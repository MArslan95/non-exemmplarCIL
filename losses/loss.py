from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


_EPS = 1e-12
_INVALID_ENERGY = 1e6


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def safe_zero_like(
    ref: Optional[torch.Tensor] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if torch.is_tensor(ref):
        return ref.sum() * 0.0
    return torch.tensor(
        0.0,
        device=device if device is not None else torch.device("cpu"),
        dtype=dtype if dtype is not None else torch.float32,
    )


def _require_finite_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a tensor.")
    if x.numel() == 0:
        return
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
        raise RuntimeError(f"{name}: tensor contains {bad} NaN/Inf values.")


def _as_1d_long(labels: torch.Tensor, *, device: torch.device, name: str = "labels") -> torch.Tensor:
    if labels is None or not torch.is_tensor(labels):
        raise TypeError(f"{name} must be a tensor.")
    return labels.to(device=device).long().flatten()


def _scalar(value: Any, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.reshape(())
        return value.float().mean()
    if isinstance(value, (int, float)):
        if torch.is_tensor(ref):
            return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
        return torch.tensor(float(value), dtype=torch.float32)
    return safe_zero_like(ref)



def _ordered_unique_ints(values: Optional[Iterable[int]]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        c = int(value)
        if c < 0:
            raise ValueError(f"class ids must be non-negative, got {c}")
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _global_to_local_labels(labels: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
    seen = _ordered_unique_ints(seen_classes)
    if not seen:
        raise ValueError("seen_classes must be non-empty for global-label mapping.")
    y = labels.long().flatten()
    mapping = {c: i for i, c in enumerate(seen)}
    local = torch.full_like(y, -1)
    for c, i in mapping.items():
        local[y == int(c)] = int(i)
    if bool((local < 0).any().item()):
        bad = sorted(set(int(v) for v in y[local < 0].detach().cpu().tolist()))
        raise RuntimeError(f"labels contain classes absent from seen_classes: bad={bad}, seen={seen}")
    return local


def _resolve_local_labels(
    labels: torch.Tensor,
    *,
    width: int,
    device: torch.device,
    seen_classes: Optional[Iterable[int]] = None,
    labels_are_global: bool = False,
    name: str = "labels",
) -> torch.Tensor:
    y = _as_1d_long(labels, device=device, name=name)
    if labels_are_global:
        if seen_classes is None:
            raise ValueError("seen_classes is required when labels_are_global=True.")
        y = _global_to_local_labels(y, seen_classes).to(device=device)
    if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= int(width)):
        raise RuntimeError(
            f"{name} must be local labels in [0,{int(width)-1}], got "
            f"{torch.unique(y).detach().cpu().tolist()}"
        )
    return y


def _class_balanced_mean(values: torch.Tensor, labels_local: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    v = values.flatten()
    y = labels_local.long().flatten().to(device=v.device)
    if v.numel() != y.numel():
        raise RuntimeError(f"class-balanced reduction mismatch: values={v.numel()}, labels={y.numel()}")
    terms = [v[y == c].mean() for c in torch.unique(y, sorted=True) if bool((y == c).any().item())]
    return torch.stack(terms).mean() if terms else v.sum() * 0.0


def _resolve_old_new_masks(
    width: int,
    *,
    device: torch.device,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    require_nonempty: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seen_classes is None:
        seen = list(range(int(width)))
    else:
        seen = _ordered_unique_ints(seen_classes)
        if len(seen) != int(width):
            raise RuntimeError(f"len(seen_classes)={len(seen)} must equal energy/logit width={int(width)}")

    if old_classes is None:
        k = int(max(0, min(int(old_class_count or 0), int(width))))
        old = seen[:k]
    else:
        old = _ordered_unique_ints(old_classes)
    if new_classes is None:
        old_set = set(old)
        new = [c for c in seen if c not in old_set]
    else:
        new = _ordered_unique_ints(new_classes)

    seen_set, old_set, new_set = set(seen), set(old), set(new)
    if not old_set.issubset(seen_set):
        raise RuntimeError(f"old_classes are not a subset of seen_classes: old={old}, seen={seen}")
    if not new_set.issubset(seen_set):
        raise RuntimeError(f"new_classes are not a subset of seen_classes: new={new}, seen={seen}")
    if old_set & new_set:
        raise RuntimeError(f"old_classes and new_classes overlap: {sorted(old_set & new_set)}")
    old_mask = torch.tensor([c in old_set for c in seen], device=device, dtype=torch.bool)
    new_mask = torch.tensor([c in new_set for c in seen], device=device, dtype=torch.bool)
    if require_nonempty and (not bool(old_mask.any().item()) or not bool(new_mask.any().item())):
        raise RuntimeError(f"old/new groups must both be non-empty: seen={seen}, old={old}, new={new}")
    return old_mask, new_mask


def _resolve_valid_class_mask(
    valid_mask: Optional[torch.Tensor],
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones((int(width),), device=device, dtype=torch.bool)
    vm = valid_mask.to(device=device).bool()
    if vm.dim() == 2:
        if vm.size(1) != int(width):
            raise RuntimeError(f"valid_mask width={vm.size(1)} does not match class width={int(width)}")
        vm = vm.all(dim=0)
    else:
        vm = vm.flatten()
    if vm.numel() != int(width):
        raise RuntimeError(f"valid_mask must contain {int(width)} class entries, got {vm.numel()}")
    return vm


def _class_centers(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_samples: int = 2,
    normalize_centers: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
        device = features.device if torch.is_tensor(features) else torch.device("cpu")
        dtype = features.dtype if torch.is_tensor(features) else torch.float32
        return (
            torch.empty(0, 0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=dtype),
        )

    if features.dim() != 2:
        raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")

    y = _as_1d_long(labels, device=features.device)
    if y.numel() != features.size(0):
        raise ValueError(f"labels/features mismatch: labels={y.numel()}, features={features.size(0)}")

    centers, class_ids, counts = [], [], []
    for cls in torch.unique(y, sorted=True):
        mask = y == cls
        n = int(mask.sum().item())
        if n < int(min_samples):
            continue
        c = features[mask].mean(dim=0)
        if normalize_centers:
            c = F.normalize(c, dim=0, eps=1e-6)
        centers.append(c)
        class_ids.append(cls)
        counts.append(torch.tensor(float(n), device=features.device, dtype=features.dtype))

    if not centers:
        return (
            torch.empty(0, features.size(1), device=features.device, dtype=features.dtype),
            torch.empty(0, device=features.device, dtype=torch.long),
            torch.empty(0, device=features.device, dtype=features.dtype),
        )

    return torch.stack(centers, dim=0), torch.stack(class_ids).long(), torch.stack(counts)


def _pairwise_center_margin_loss(centers: torch.Tensor, margin: float) -> torch.Tensor:
    if centers is None or not torch.is_tensor(centers) or centers.numel() == 0 or centers.size(0) < 2:
        return safe_zero_like(centers)
    dist = torch.cdist(centers, centers, p=2)
    eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
    pair = dist[~eye]
    if pair.numel() == 0:
        return centers.sum() * 0.0
    return F.relu(float(margin) - pair).pow(2).mean()


def _pad_to_width(x: torch.Tensor, width: int) -> torch.Tensor:
    if x.size(1) == width:
        return x
    if x.size(1) > width:
        return x[:, :width]
    return F.pad(x, (0, int(width) - int(x.size(1))))


# -----------------------------------------------------------------------------
# Spectral / band profiles
# -----------------------------------------------------------------------------

def _spectral_derivatives(spectral_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if spectral_summary.dim() != 2:
        raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
    if spectral_summary.size(1) < 2:
        z = spectral_summary.new_zeros((spectral_summary.size(0), 1))
        return z, z
    d1 = spectral_summary[:, 1:] - spectral_summary[:, :-1]
    if d1.size(1) < 2:
        d2 = d1.new_zeros((d1.size(0), 1))
    else:
        d2 = d1[:, 1:] - d1[:, :-1]
    return d1, d2


def _spectral_profile_descriptor(spectral_summary: torch.Tensor) -> torch.Tensor:
    """Derivative-aware descriptor for physical spectral-shape comparison.

    Raw HSI spectra may be normalized and may contain signed values.  Direct
    softmax over signed spectra destroys absorption/reflectance shape.  This
    descriptor preserves curve shape by standardizing each spectrum and appending
    first/second derivative information.
    """
    if spectral_summary.dim() != 2:
        raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
    s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
    s = s - s.mean(dim=1, keepdim=True)
    s = s / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    d1, d2 = _spectral_derivatives(s)
    desc = torch.cat([F.normalize(s, dim=1, eps=1e-6), F.normalize(d1, dim=1, eps=1e-6), F.normalize(d2, dim=1, eps=1e-6)], dim=1)
    return torch.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


def _band_importance_profile(band_summary: torch.Tensor) -> torch.Tensor:
    """Convert raw spectra or band summaries to a non-negative band profile.

    This is the correct base regularizer target for HSI: it compares where the
    spectrum changes/absorbs, not a hidden classifier branch.  It works for raw
    physical spectra and is still safe for reduced non-physical summaries.
    """
    if band_summary is None or not torch.is_tensor(band_summary):
        raise TypeError("band_summary must be a tensor.")
    if band_summary.dim() != 2:
        raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")

    b = torch.nan_to_num(band_summary, nan=0.0, posinf=0.0, neginf=0.0)
    B, S = b.shape
    if S <= 0:
        return b

    # Per-sample standardization preserves spectral shape under dataset scaling.
    z = b - b.mean(dim=1, keepdim=True)
    z = z / z.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    d1, d2 = _spectral_derivatives(z)
    d1e = _pad_to_width(d1.abs(), S)
    d2e = _pad_to_width(d2.abs(), S)
    profile = z.abs() + 0.50 * d1e + 0.25 * d2e
    profile = profile.clamp_min(0.0)
    uniform = torch.full_like(profile, 1.0 / float(max(S, 1)))
    denom = profile.sum(dim=1, keepdim=True)
    profile = torch.where(denom > 1e-8, profile / denom.clamp_min(1e-8), uniform)
    return torch.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)


# Backward-compatible name used by old code.
def _normalize_band_summary(band_summary: torch.Tensor) -> torch.Tensor:
    return _band_importance_profile(band_summary)


def _positive_cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1, eps=1e-6)
    b = F.normalize(b, dim=1, eps=1e-6)
    return (a @ b.t()).clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# GICS: Geometry-Involved Class Separation
# -----------------------------------------------------------------------------

def base_geometry_involved_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    key_features: Optional[torch.Tensor] = None,
    weight: float = 0.20,
    temperature: float = 0.07,
    same_class_positive: bool = True,
    class_balanced: bool = True,
    detach_key: bool = True,
    normalize: bool = True,
    return_parts: bool = True,
    **_: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Base-phase GICS on canonical projected z-space."""
    if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
        z = safe_zero_like(features)
        out = {"total": z, "gics": z, "weighted_gics": z, "valid_anchors": z, "num_anchors": z, "mean_positive_count": z}
        return out if return_parts else z

    if features.dim() != 2:
        raise ValueError(f"GICS expects projected features [B,D], got {tuple(features.shape)}")
    _require_finite_tensor(features, "gics.features")

    zq = features
    explicit_key = key_features is not None and torch.is_tensor(key_features) and key_features.numel() > 0
    zk = key_features if explicit_key else features
    if zk.dim() != 2:
        raise ValueError(f"key_features must be [B,D], got {tuple(zk.shape)}")
    if zk.size(0) != zq.size(0) or zk.size(1) != zq.size(1):
        raise ValueError(f"key_features shape mismatch: query={tuple(zq.shape)}, key={tuple(zk.shape)}")
    if detach_key:
        zk = zk.detach()

    y = _as_1d_long(labels, device=zq.device)
    if y.numel() != zq.size(0):
        raise ValueError(f"GICS labels/features mismatch: labels={y.numel()}, features={zq.size(0)}")

    q = F.normalize(zq, dim=1, eps=1e-6) if normalize else zq
    k = F.normalize(zk.to(device=zq.device, dtype=zq.dtype), dim=1, eps=1e-6) if normalize else zk.to(device=zq.device, dtype=zq.dtype)

    logits = q @ k.t()
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    B = zq.size(0)
    diag = torch.eye(B, device=zq.device, dtype=torch.bool)
    positive = y.view(-1, 1).eq(y.view(1, -1)) if same_class_positive else diag.clone()

    if explicit_key:
        positive = positive | diag
        denom_mask = torch.ones_like(positive, dtype=torch.bool)
    else:
        positive = positive & (~diag)
        denom_mask = ~diag

    pos_count = positive.float().sum(dim=1)
    valid = pos_count > 0

    if not bool(valid.any().item()):
        gics = features.sum() * 0.0
    else:
        exp_logits = torch.exp(logits).masked_fill(~denom_mask, 0.0)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - denom.log()
        per = -(positive.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
        per = per[valid]
        yv = y[valid]
        if per.numel() == 0:
            gics = features.sum() * 0.0
        elif class_balanced:
            class_terms = []
            for c in torch.unique(yv, sorted=True):
                cm = yv == c
                if bool(cm.any().item()):
                    class_terms.append(per[cm].mean())
            gics = torch.stack(class_terms).mean() if class_terms else features.sum() * 0.0
        else:
            gics = per.mean()

    total = float(weight) * gics
    if not return_parts:
        return total
    return {
        "total": total,
        "gics": gics.detach(),
        "weighted_gics": total.detach(),
        "valid_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
        "num_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
        "mean_positive_count": pos_count[valid].float().mean().detach() if bool(valid.any().item()) else features.sum().detach() * 0.0,
    }


# Backward-compatible aliases.
def base_fcs_geometry_contrastive_loss(*args: Any, **kwargs: Any):
    return base_geometry_involved_contrastive_loss(*args, **kwargs)


def base_geometry_involved_contrastive_separation_loss(*args: Any, **kwargs: Any):
    return base_geometry_involved_contrastive_loss(*args, **kwargs)


def base_supervised_contrastive_loss(*args: Any, **kwargs: Any):
    return base_geometry_involved_contrastive_loss(*args, **kwargs)


def base_hsi_supervised_contrastive_loss(*args: Any, **kwargs: Any):
    return base_geometry_involved_contrastive_loss(*args, **kwargs)


# -----------------------------------------------------------------------------
# PGR: Prospective Geometry Reserve
# -----------------------------------------------------------------------------

def _batch_subspace_overlap_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    rank: int = 3,
    min_samples: int = 6,
    max_overlap: float = 0.50,
    normalize: bool = True,
    include_mean_overlap: bool = True,
    return_parts: bool = False,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Margin-based subspace reserve.

    The old loss minimized average overlap only.  That can look nonzero in logs
    while leaving a single highly-overlapping class pair that breaks incremental
    descriptor insertion.  This version explicitly penalizes pair overlaps above
    max_overlap while still reporting mean/max overlap.
    """
    if features is None or not torch.is_tensor(features) or features.numel() == 0:
        z = safe_zero_like(features)
        if return_parts:
            return {"total": z, "pair_count": z, "valid_class_count": z, "mean_overlap": z, "max_overlap": z}
        return z
    if features.dim() != 2:
        raise ValueError(f"subspace loss expects [B,D], got {tuple(features.shape)}")

    y = _as_1d_long(labels, device=features.device)
    z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
    _, D = z.shape

    bases = []
    for cls in torch.unique(y, sorted=True):
        m = y == cls
        n = int(m.sum().item())
        if n < int(min_samples):
            continue
        xc = z[m] - z[m].mean(dim=0, keepdim=True)
        r = max(1, min(int(rank), n - 1, D))
        try:
            _, s, vh = torch.linalg.svd(xc, full_matrices=False)
        except RuntimeError:
            continue
        if s.numel() == 0 or float(s[0].detach().abs().item()) <= 1e-8:
            continue
        keep = (s[:r] / s[0].clamp_min(1e-8)) > 1e-3
        if bool(keep.any().item()):
            bases.append(vh[:r][keep].t().contiguous())

    if len(bases) < 2:
        loss = features.sum() * 0.0
        overlaps = features.new_empty(0)
        pair_count = 0
    else:
        overlap_list = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                Ui, Uj = bases[i], bases[j]
                denom = float(max(min(Ui.size(1), Uj.size(1)), 1))
                ov = (Ui.t() @ Uj).pow(2).sum() / denom
                overlap_list.append(ov)
        overlaps = torch.stack(overlap_list) if overlap_list else features.new_empty(0)
        pair_count = int(overlaps.numel())
        if pair_count == 0:
            loss = features.sum() * 0.0
        else:
            margin_loss = F.relu(overlaps - float(max_overlap)).pow(2).mean()
            mean_loss = 0.10 * overlaps.mean() if include_mean_overlap else overlaps.sum() * 0.0
            loss = margin_loss + mean_loss

    if return_parts:
        return {
            "total": loss,
            "pair_count": torch.tensor(float(pair_count), device=features.device, dtype=features.dtype),
            "valid_class_count": torch.tensor(float(len(bases)), device=features.device, dtype=features.dtype),
            "mean_overlap": overlaps.mean().detach() if overlaps.numel() else features.sum().detach() * 0.0,
            "max_overlap": overlaps.max().detach() if overlaps.numel() else features.sum().detach() * 0.0,
        }
    return loss



def risk_aware_band_discrimination_loss(
    band_summary: Optional[torch.Tensor],
    labels: torch.Tensor,
    *,
    features: Optional[torch.Tensor] = None,
    min_samples: int = 3,
    max_band_similarity: float = 0.75,
    risk_center_margin: float = 1.0,
    risk_weight: float = 1.0,
    return_parts: bool = True,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Band-guided feature-geometry reserve for HSI.

    Important: raw physical spectra are *not* trainable. Therefore this loss
    must not try to make raw band profiles dissimilar directly.  Instead, high
    band similarity marks a hard class pair and increases the gradient on the
    learned feature geometry for that pair.

    Loss = hard_band_similarity.detach() * feature_center_conflict(features)

    This keeps band information in the architecture while preserving the main
    SCTGR rule: inference and replay remain geometry-energy based.
    """
    ref = features if torch.is_tensor(features) else (band_summary if torch.is_tensor(band_summary) else labels)
    if band_summary is None or not torch.is_tensor(band_summary) or band_summary.numel() == 0:
        z = safe_zero_like(ref)
        out = {
            "total": z,
            "band": z,
            "pair_count": z,
            "valid_class_count": z,
            "mean_similarity": z,
            "max_similarity": z,
            "guided_conflict_mean": z,
            "guided_conflict_max": z,
        }
        return out if return_parts else z

    if band_summary.dim() != 2:
        raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")

    y = _as_1d_long(labels, device=band_summary.device)
    if y.numel() != band_summary.size(0):
        raise ValueError(f"band labels/batch mismatch: labels={y.numel()}, band={band_summary.size(0)}")

    b = _band_importance_profile(band_summary)
    b_centers, class_ids, _ = _class_centers(b, y, min_samples=min_samples, normalize_centers=True)
    if b_centers.size(0) < 2:
        z = band_summary.sum() * 0.0
        out = {
            "total": z,
            "band": z,
            "pair_count": z,
            "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
            "mean_similarity": z,
            "max_similarity": z,
            "guided_conflict_mean": z,
            "guided_conflict_max": z,
        }
        return out if return_parts else z

    sim = (b_centers @ b_centers.t()).clamp(0.0, 1.0)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    pair_sim = sim[~eye]
    hard_band = F.relu(pair_sim - float(max_band_similarity)) / max(1.0 - float(max_band_similarity), 1e-6)
    hard_band = hard_band.detach()

    # The only trainable part is the learned feature geometry.  If features are
    # missing or class ids cannot be aligned, the band term reports similarity
    # but contributes zero gradient instead of applying a useless constant loss.
    if features is not None and torch.is_tensor(features) and features.numel() > 0:
        zf = F.normalize(features.to(device=band_summary.device, dtype=band_summary.dtype), dim=1, eps=1e-6)
        f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
        if f_centers.size(0) == b_centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
            dist = torch.cdist(f_centers, f_centers, p=2)
            feature_conflict = F.relu(float(risk_center_margin) - dist) / max(float(risk_center_margin), 1e-6)
            feature_conflict = feature_conflict[~eye]
            guided = hard_band * feature_conflict
            loss_vec = hard_band * feature_conflict.pow(2)
            loss = float(risk_weight) * (loss_vec.mean() if loss_vec.numel() > 0 else features.sum() * 0.0)
        else:
            guided = torch.zeros_like(pair_sim)
            loss = features.sum() * 0.0
    else:
        guided = torch.zeros_like(pair_sim)
        loss = band_summary.sum() * 0.0

    if not return_parts:
        return loss
    return {
        "total": loss,
        "band": loss.detach(),
        "pair_count": torch.tensor(float(pair_sim.numel()), device=band_summary.device, dtype=band_summary.dtype),
        "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
        "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
        "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
        "guided_conflict_mean": guided.mean().detach() if guided.numel() > 0 else band_summary.sum().detach() * 0.0,
        "guided_conflict_max": guided.max().detach() if guided.numel() > 0 else band_summary.sum().detach() * 0.0,
    }


def prospective_geometry_reserve_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    band_summary: Optional[torch.Tensor] = None,
    weight: float = 0.10,
    compact_weight: float = 0.15,
    center_weight: float = 0.20,
    subspace_weight: float = 0.10,
    band_weight: float = 0.05,
    volume_weight: float = 0.05,
    center_margin: float = 1.05,
    min_class_samples: int = 3,
    subspace_min_samples: int = 6,
    subspace_rank: int = 3,
    max_band_similarity: float = 0.75,
    max_class_variance: float = 0.75,
    min_class_variance: float = 0.015,
    max_subspace_overlap: float = 0.50,
    normalize_features: bool = True,
    adaptive_component_weights: bool = True,
    return_parts: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Base-phase PGR.

    Active terms:
      - compactness: same-class spread control
      - center reserve: class centers separated by margin
      - subspace reserve: explicit margin on tangent subspace overlap
      - band reserve: risky classes avoid identical spectral-band profiles
      - volume reserve: avoids both broad blobs and collapsed zero-volume rows
    """
    if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
        z0 = safe_zero_like(features)
        out = {
            "total": z0, "pgr": z0, "weighted_pgr": z0,
            "compact": z0, "center": z0, "subspace": z0, "band": z0, "volume": z0,
            "valid_class_count": z0, "unique_class_count": z0,
            "subspace_pair_count": z0, "band_pair_count": z0,
            "compact_factor": z0, "center_factor": z0, "subspace_factor": z0,
            "band_factor": z0, "volume_factor": z0,
            "subspace_mean_overlap": z0, "subspace_max_overlap": z0,
            "band_mean_similarity": z0, "band_max_similarity": z0,
        }
        return out if return_parts else z0

    if features.dim() != 2:
        raise ValueError(f"PGR expects features [B,D], got {tuple(features.shape)}")
    _require_finite_tensor(features, "pgr.features")

    y = _as_1d_long(labels, device=features.device)
    if y.numel() != features.size(0):
        raise ValueError(f"PGR labels/features mismatch: labels={y.numel()}, features={features.size(0)}")

    # Let explicit aliases in kwargs override defaults without requiring trainer changes.
    if "pgr_max_subspace_overlap" in kwargs:
        max_subspace_overlap = float(kwargs["pgr_max_subspace_overlap"])
    if "subspace_overlap_max" in kwargs:
        max_subspace_overlap = float(kwargs["subspace_overlap_max"])
    if "pgr_min_class_variance" in kwargs:
        min_class_variance = float(kwargs["pgr_min_class_variance"])

    z = F.normalize(features, dim=1, eps=1e-6) if normalize_features else features

    compact_terms = []
    volume_terms = []
    class_vars = []
    for cls in torch.unique(y, sorted=True):
        m = y == cls
        if int(m.sum().item()) < int(min_class_samples):
            continue
        xc = z[m]
        var = (xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
        compact_terms.append(var)
        class_vars.append(var.detach())
        broad = F.relu(var - float(max_class_variance)).pow(2)
        collapsed = F.relu(float(min_class_variance) - var).pow(2)
        volume_terms.append(broad + collapsed)

    compact = torch.stack(compact_terms).mean() if compact_terms else features.sum() * 0.0
    volume = torch.stack(volume_terms).mean() if volume_terms else features.sum() * 0.0
    valid_class_count = len(compact_terms)
    unique_class_count = int(torch.unique(y).numel())

    centers, _, _ = _class_centers(z, y, min_samples=min_class_samples, normalize_centers=False)
    center = _pairwise_center_margin_loss(centers, center_margin)

    sub_obj = _batch_subspace_overlap_loss(
        z,
        y,
        rank=int(subspace_rank),
        min_samples=int(subspace_min_samples),
        max_overlap=float(max_subspace_overlap),
        normalize=False,
        return_parts=True,
    )
    subspace = sub_obj["total"]
    subspace_pair_count = sub_obj["pair_count"]

    band_obj = risk_aware_band_discrimination_loss(
        band_summary,
        y,
        features=z,
        min_samples=int(min_class_samples),
        max_band_similarity=float(max_band_similarity),
        risk_center_margin=float(center_margin),
        return_parts=True,
    )
    band = band_obj["total"] if isinstance(band_obj, dict) else band_obj
    band_pair_count = band_obj.get("pair_count", features.sum().detach() * 0.0) if isinstance(band_obj, dict) else features.sum().detach() * 0.0

    one = torch.tensor(1.0, device=features.device, dtype=features.dtype)
    zero = features.sum() * 0.0
    if adaptive_component_weights:
        compact_factor = one if valid_class_count > 0 else zero
        volume_factor = one if valid_class_count > 0 else zero
        center_factor = one if int(centers.size(0)) >= 2 else zero
        subspace_factor = one if float(subspace_pair_count.detach().item()) > 0.0 else zero
        band_factor = one if float(band_pair_count.detach().item()) > 0.0 else zero
    else:
        compact_factor = center_factor = subspace_factor = band_factor = volume_factor = one

    pgr_unweighted = (
        float(compact_weight) * compact_factor * compact
        + float(center_weight) * center_factor * center
        + float(subspace_weight) * subspace_factor * subspace
        + float(band_weight) * band_factor * band
        + float(volume_weight) * volume_factor * volume
    )
    total = float(weight) * pgr_unweighted

    if not return_parts:
        return total
    return {
        "total": total,
        "pgr": pgr_unweighted.detach(),
        "weighted_pgr": total.detach(),
        "compact": compact.detach(),
        "center": center.detach(),
        "subspace": subspace.detach(),
        "band": band.detach(),
        "volume": volume.detach(),
        "valid_class_count": torch.tensor(float(valid_class_count), device=features.device, dtype=features.dtype),
        "unique_class_count": torch.tensor(float(unique_class_count), device=features.device, dtype=features.dtype),
        "subspace_pair_count": subspace_pair_count.detach(),
        "band_pair_count": band_pair_count.detach() if torch.is_tensor(band_pair_count) else torch.tensor(float(band_pair_count), device=features.device, dtype=features.dtype),
        "compact_factor": compact_factor.detach(),
        "center_factor": center_factor.detach(),
        "subspace_factor": subspace_factor.detach(),
        "band_factor": band_factor.detach(),
        "volume_factor": volume_factor.detach(),
        "subspace_mean_overlap": sub_obj.get("mean_overlap", zero).detach(),
        "subspace_max_overlap": sub_obj.get("max_overlap", zero).detach(),
        "band_mean_similarity": band_obj.get("mean_similarity", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
        "band_max_similarity": band_obj.get("max_similarity", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
        "band_guided_conflict_mean": band_obj.get("guided_conflict_mean", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
        "band_guided_conflict_max": band_obj.get("guided_conflict_max", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
        "class_variance_mean": torch.stack(class_vars).mean() if class_vars else zero.detach(),
    }


def base_prospective_geometry_reserve_loss(*args: Any, **kwargs: Any):
    return prospective_geometry_reserve_loss(*args, **kwargs)


# -----------------------------------------------------------------------------
# Physical spectral-shape reserve
# -----------------------------------------------------------------------------


def spectral_shape_discrimination_loss(
    spectral_summary: Optional[torch.Tensor],
    labels: torch.Tensor,
    *,
    features: Optional[torch.Tensor] = None,
    spectral_summary_is_physical: bool = False,
    require_physical_summary: bool = True,
    min_samples: int = 3,
    max_shape_similarity: float = 0.75,
    risk_center_margin: float = 1.0,
    risk_weight: float = 1.0,
    return_parts: bool = True,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Spectral-shape-guided feature-geometry reserve.

    Raw wavelength spectra are fixed metadata, not trainable outputs.  High
    spectral-shape similarity should therefore identify hard HSI class pairs and
    push their learned feature centers apart.  It should not be optimized as a
    standalone raw-spectrum dissimilarity objective.
    """
    ref = features if torch.is_tensor(features) else (spectral_summary if torch.is_tensor(spectral_summary) else labels)
    if (
        spectral_summary is None
        or not torch.is_tensor(spectral_summary)
        or spectral_summary.numel() == 0
        or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
    ):
        z = safe_zero_like(ref)
        out = {
            "total": z,
            "spectral_shape": z,
            "pair_count": z,
            "valid_class_count": z,
            "mean_similarity": z,
            "max_similarity": z,
            "guided_conflict_mean": z,
            "guided_conflict_max": z,
        }
        return out if return_parts else z

    s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
    if s.dim() != 2:
        raise ValueError(f"spectral_summary must be [B,S], got {tuple(s.shape)}")
    y = _as_1d_long(labels, device=s.device)
    if y.numel() != s.size(0):
        raise ValueError(f"spectral_summary/label mismatch: spectra={s.size(0)}, labels={y.numel()}")

    desc = _spectral_profile_descriptor(s)
    centers, class_ids, _ = _class_centers(desc, y, min_samples=min_samples, normalize_centers=False)

    if centers.size(0) < 2:
        z = s.sum() * 0.0
        out = {
            "total": z,
            "spectral_shape": z,
            "pair_count": z,
            "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
            "mean_similarity": z,
            "max_similarity": z,
            "guided_conflict_mean": z,
            "guided_conflict_max": z,
        }
        return out if return_parts else z

    sim = _positive_cosine_matrix(centers, centers)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    pair_sim = sim[~eye]
    hard_shape = F.relu(pair_sim - float(max_shape_similarity)) / max(1.0 - float(max_shape_similarity), 1e-6)
    hard_shape = hard_shape.detach()

    if features is not None and torch.is_tensor(features) and features.numel() > 0:
        zf = F.normalize(features.to(device=s.device, dtype=s.dtype), dim=1, eps=1e-6)
        f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
        if f_centers.size(0) == centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
            dist = torch.cdist(f_centers, f_centers, p=2)
            feature_conflict = F.relu(float(risk_center_margin) - dist)[~eye] / max(float(risk_center_margin), 1e-6)
            guided = hard_shape * feature_conflict
            loss_vec = hard_shape * feature_conflict.pow(2)
            loss = float(risk_weight) * (loss_vec.mean() if loss_vec.numel() > 0 else features.sum() * 0.0)
        else:
            guided = torch.zeros_like(pair_sim)
            loss = features.sum() * 0.0
    else:
        guided = torch.zeros_like(pair_sim)
        loss = s.sum() * 0.0

    if not return_parts:
        return loss
    return {
        "total": loss,
        "spectral_shape": loss.detach(),
        "pair_count": torch.tensor(float(pair_sim.numel()), device=s.device, dtype=s.dtype),
        "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
        "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
        "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
        "guided_conflict_mean": guided.mean().detach() if guided.numel() > 0 else s.sum().detach() * 0.0,
        "guided_conflict_max": guided.max().detach() if guided.numel() > 0 else s.sum().detach() * 0.0,
    }


# -----------------------------------------------------------------------------
# Low-rank GeometryBank energy used by classifier/replay/margins
# -----------------------------------------------------------------------------

def _canonicalize_variances(
    *,
    eigvals: Optional[torch.Tensor] = None,
    res_vars: Optional[torch.Tensor] = None,
    variances: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if variances is not None and torch.is_tensor(variances):
        if variances.dim() != 2 or variances.size(1) < 2:
            raise ValueError(f"variances must be [C,R+1], got {tuple(variances.shape)}")
        return variances[:, :-1], variances[:, -1]
    if eigvals is None or res_vars is None:
        raise ValueError("Either variances or eigvals+res_vars must be provided.")
    return eigvals, res_vars


def _active_rank_mask(active_ranks: Optional[torch.Tensor], C: int, R: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if active_ranks is None or not torch.is_tensor(active_ranks):
        ar = torch.full((C,), int(R), device=device, dtype=torch.long)
    else:
        ar = active_ranks.to(device=device).long().flatten()
        if ar.numel() != C:
            raise ValueError(f"active_ranks must have C={C} entries, got {ar.numel()}")
        ar = ar.clamp(min=0, max=R)
    idx = torch.arange(R, device=device).view(1, R)
    mask = (idx < ar.view(C, 1)).to(dtype=dtype)
    return mask, ar




def geometry_energy_matrix(
    features: Optional[torch.Tensor] = None,
    means: Optional[torch.Tensor] = None,
    bases: Optional[torch.Tensor] = None,
    variances: Optional[torch.Tensor] = None,
    *,
    bank: Optional[Mapping[str, torch.Tensor]] = None,
    eigvals: Optional[torch.Tensor] = None,
    res_vars: Optional[torch.Tensor] = None,
    active_ranks: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    sample_counts: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    seen_classes: Optional[Iterable[int]] = None,
    class_ids: Optional[Iterable[int]] = None,
    variance_floor: float = 1e-4,
    reliability_energy_weight: float = 0.0,
    reliability_min_clamp: float = 0.05,
    residual_variance_scale: float = 1.0,
    normalize_by_dim: bool = True,
    invalid_class_energy: float = _INVALID_ENERGY,
    use_logdet_energy: bool = False,
    logdet_energy_weight: float = 0.0,
    logdet_normalize_by_dim: bool = True,
    center_logdet_energy: bool = False,
    spectral_summary: Optional[torch.Tensor] = None,
    spectral_curve_means: Optional[torch.Tensor] = None,
    spectral_curve_vars: Optional[torch.Tensor] = None,
    spectral_curve_d1: Optional[torch.Tensor] = None,
    spectral_curve_d2: Optional[torch.Tensor] = None,
    spectral_shape_reliability: Optional[torch.Tensor] = None,
    use_spectral_residual_energy: bool = False,
    spectral_energy_weight: float = 0.0,
    spectral_summary_is_physical: bool = False,
    spectral_require_physical_summary: bool = True,
    return_parts: bool = False,
    **_: Any,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """Exact GeometryBank energy shared by replay, losses, and classification.

    Main-path energy is rank-normalized parallel Mahalanobis plus
    residual-dimension-normalized orthogonal Mahalanobis. Reliability, logdet,
    spectral, calibration, and residual-rescaling terms are rejected because
    they would make loss energy differ from GeometryBank replay validation.
    """
    del (
        reliability, reliability_min_clamp, logdet_normalize_by_dim,
        spectral_summary, spectral_curve_means, spectral_curve_vars,
        spectral_curve_d1, spectral_curve_d2, spectral_shape_reliability,
        spectral_summary_is_physical, spectral_require_physical_summary,
    )
    if abs(float(residual_variance_scale) - 1.0) > 1e-8:
        raise RuntimeError("residual_variance_scale must be 1.0 for GeometryBank energy consistency.")
    if not bool(normalize_by_dim):
        raise RuntimeError("normalize_by_dim must be true for GeometryBank energy consistency.")
    if bool(use_logdet_energy) or float(logdet_energy_weight) != 0.0 or bool(center_logdet_energy):
        raise RuntimeError("Log-determinant energy is disabled; it is absent from replay-validation energy.")
    if float(reliability_energy_weight) != 0.0:
        raise RuntimeError("Reliability controls replay/trust and must not bias classifier or loss energy.")
    if bool(use_spectral_residual_energy) or float(spectral_energy_weight) != 0.0:
        raise RuntimeError("Spectral information guides replay and base reserve; it is not an inference-energy branch.")

    if isinstance(means, Mapping) and bank is None and bases is None:
        bank = means
        means = None
    selected_ids = _ordered_unique_ints(seen_classes if seen_classes is not None else class_ids)
    bank_class_ids: Optional[torch.Tensor] = None
    if bank is not None:
        means = bank.get("means", means)
        bases = bank.get("bases", bank.get("subspace_bases", bases))
        variances = bank.get("variances", variances)
        eigvals = bank.get("eigvals", bank.get("eigenvalues", eigvals))
        res_vars = bank.get("res_vars", bank.get("residual_variances", bank.get("resvars", res_vars)))
        active_ranks = bank.get("active_ranks", active_ranks)
        sample_counts = bank.get("sample_counts", sample_counts)
        valid_mask = bank.get("valid_mask", bank.get("valid_class_mask", valid_mask))
        if torch.is_tensor(bank.get("class_ids", None)):
            bank_class_ids = bank["class_ids"].long().flatten()

    if means is None or bases is None:
        raise ValueError("geometry_energy_matrix requires means and bases, or a bank mapping containing them.")
    if features is None or not torch.is_tensor(features):
        raise TypeError("features must be a tensor.")
    if features.dim() != 2:
        raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
    if means.dim() != 2 or bases.dim() != 3:
        raise ValueError(f"means/bases must be [C,D] and [C,D,R], got {tuple(means.shape)}, {tuple(bases.shape)}")

    B, D = features.shape
    C, Dm = means.shape
    if Dm != D or bases.size(0) != C or bases.size(1) != D:
        raise ValueError(f"feature/bank shape mismatch: features={tuple(features.shape)}, means={tuple(means.shape)}, bases={tuple(bases.shape)}")

    device, dtype = features.device, features.dtype
    means = means.to(device=device, dtype=dtype)
    bases = bases.to(device=device, dtype=dtype)
    eig, rv = _canonicalize_variances(eigvals=eigvals, res_vars=res_vars, variances=variances)
    eig = eig.to(device=device, dtype=dtype)
    rv = rv.to(device=device, dtype=dtype).flatten()
    R = int(bases.size(2))
    if eig.shape != (C, R) or rv.numel() != C:
        raise ValueError(f"variance shape mismatch: eig={tuple(eig.shape)}, res={rv.numel()}, expected {(C,R)} and {C}")

    row_idx = torch.arange(C, device=device, dtype=torch.long)
    global_ids = torch.arange(C, device=device, dtype=torch.long)
    if selected_ids:
        if bank_class_ids is not None:
            ids_cpu = bank_class_ids.detach().cpu().tolist()
            if len(set(int(c) for c in ids_cpu)) != len(ids_cpu):
                raise RuntimeError("bank['class_ids'] contains duplicate ids.")
            mapping = {int(c): i for i, c in enumerate(ids_cpu)}
            missing = [c for c in selected_ids if c not in mapping]
            if missing:
                raise RuntimeError(f"selected classes absent from bank: {missing}")
            row_idx = torch.as_tensor([mapping[c] for c in selected_ids], device=device, dtype=torch.long)
        else:
            missing = [c for c in selected_ids if c >= C]
            if missing:
                raise RuntimeError(f"selected classes absent from full bank rows: {missing}")
            row_idx = torch.as_tensor(selected_ids, device=device, dtype=torch.long)
        global_ids = torch.as_tensor(selected_ids, device=device, dtype=torch.long)
        means = means.index_select(0, row_idx)
        bases = bases.index_select(0, row_idx)
        eig = eig.index_select(0, row_idx)
        rv = rv.index_select(0, row_idx)
        if active_ranks is not None:
            active_ranks = active_ranks.to(device=device).flatten().index_select(0, row_idx)
        if sample_counts is not None:
            sample_counts = sample_counts.to(device=device).flatten().index_select(0, row_idx)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device).flatten().index_select(0, row_idx)
        C = len(selected_ids)

    if features.numel() == 0:
        empty = torch.empty((0, C), device=device, dtype=dtype)
        return {"energy": empty, "feature_energy": empty} if return_parts else empty

    _require_finite_tensor(features, "geometry_energy.features")
    _require_finite_tensor(means, "geometry_energy.means")
    _require_finite_tensor(bases, "geometry_energy.bases")
    _require_finite_tensor(eig, "geometry_energy.eigvals")
    _require_finite_tensor(rv, "geometry_energy.res_vars")
    if bool((rv <= 0).any().item()):
        raise RuntimeError("GeometryBank residual variances must be positive.")

    rank_mask, ar = _active_rank_mask(active_ranks, C, R, device, dtype)
    delta = features.unsqueeze(1) - means.unsqueeze(0)
    coeff = torch.einsum("bcd,cdr->bcr", delta, bases)
    coeff_active = coeff * rank_mask.view(1, C, R)
    recon = torch.einsum("bcr,cdr->bcd", coeff_active, bases)
    residual = delta - recon

    eig_safe = eig.clamp_min(float(variance_floor))
    rv_safe = rv.clamp_min(float(variance_floor))
    parallel_raw = ((coeff_active.square() / eig_safe.view(1, C, R)) * rank_mask.view(1, C, R)).sum(dim=-1)
    residual_raw = residual.square().sum(dim=-1)
    active_dims = ar.to(dtype=dtype).clamp_min(1.0)
    residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=dtype).clamp_min(1.0)
    parallel = parallel_raw / active_dims.view(1, C)
    orthogonal = residual_raw / (residual_dims.view(1, C) * rv_safe.view(1, C))
    energy = parallel + orthogonal

    valid_class = torch.isfinite(means).all(dim=1) & torch.isfinite(bases).all(dim=(1, 2))
    valid_class &= torch.isfinite(eig).all(dim=1) & torch.isfinite(rv) & (rv > 0)
    if sample_counts is not None:
        sc = sample_counts.to(device=device).flatten()
        if sc.numel() != C:
            raise ValueError(f"sample_counts must have C={C} entries, got {sc.numel()}")
        valid_class &= torch.isfinite(sc) & (sc > 0)
    if valid_mask is not None:
        valid_class &= _resolve_valid_class_mask(valid_mask, width=C, device=device)

    if bool((~valid_class).any().item()):
        mask = (~valid_class).view(1, C)
        energy = energy.masked_fill(mask, float(invalid_class_energy))
        parallel = parallel.masked_fill(mask, float(invalid_class_energy))
        orthogonal = orthogonal.masked_fill(mask, float(invalid_class_energy))
    energy = torch.nan_to_num(energy, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0)

    if not return_parts:
        return energy
    zero_c = torch.zeros((C,), device=device, dtype=dtype)
    return {
        "energy": energy,
        "feature_energy": energy,
        "parallel": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
        "orthogonal": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
        "parallel_energy": parallel,
        "residual_energy": orthogonal,
        "parallel_raw": parallel_raw,
        "residual_raw": residual_raw,
        "active_dims": active_dims,
        "residual_dims": residual_dims,
        "logdet_penalty": zero_c,
        "reliability_penalty": zero_c,
        "spectral_energy": torch.zeros_like(energy),
        "active_ranks": ar,
        "rank_mask": rank_mask,
        "valid_class_mask": valid_class,
        "global_class_ids": global_ids,
        "row_indices": row_idx,
    }



# -----------------------------------------------------------------------------
# Phase-consistent energy-margin helpers
# -----------------------------------------------------------------------------


def _energy_to_logits(
    energy: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    logit_scale: float = 8.0,
    center_per_sample: bool = True,
    clip: float = 50.0,
    invalid_logit: float = -1e9,
) -> torch.Tensor:
    """Convert lower-is-better geometry energy using the classifier contract."""
    if energy is None or not torch.is_tensor(energy) or energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,C], got {None if energy is None else tuple(energy.shape)}")
    e = torch.nan_to_num(energy.float(), nan=float(_INVALID_ENERGY), posinf=float(_INVALID_ENERGY), neginf=0.0)
    vm = _resolve_valid_class_mask(valid_mask, width=e.size(1), device=e.device)
    finite = torch.isfinite(e) & (e < 0.5 * float(_INVALID_ENERGY)) & vm.view(1, -1)
    if center_per_sample:
        masked = e.masked_fill(~finite, float("inf"))
        row_min = masked.min(dim=1, keepdim=True).values
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        e = e - row_min
    logits = -float(logit_scale) * e
    if float(clip) > 0.0:
        logits = logits.clamp(-float(clip), float(clip))
    logits = torch.nan_to_num(logits, nan=float(invalid_logit), posinf=float(invalid_logit), neginf=float(invalid_logit))
    return logits.masked_fill(~finite, float(invalid_logit))



def _ce_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    clip: float = 50.0,
    class_balanced: bool = True,
) -> torch.Tensor:
    if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
        return safe_zero_like(labels if torch.is_tensor(labels) else None)
    if logits.dim() != 2:
        raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
    y = _resolve_local_labels(labels, width=logits.size(1), device=logits.device, name="CE labels")
    if y.numel() != logits.size(0):
        raise RuntimeError(f"CE labels/logits mismatch: labels={y.numel()}, logits={logits.size(0)}")
    per = F.cross_entropy(logits.clamp(-float(clip), float(clip)), y, label_smoothing=float(label_smoothing), reduction="none")
    return _class_balanced_mean(per, y) if class_balanced else per.mean()



def _energy_margin_parts(
    energy: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
    class_balanced: bool = True,
) -> Dict[str, torch.Tensor]:
    ref = energy if torch.is_tensor(energy) else labels
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        z = safe_zero_like(ref)
        return {"total": z, "violation_rate": z.detach(), "error_rate": z.detach(), "mean_gap": z.detach(), "min_gap": z.detach(), "true_energy": z.detach(), "nearest_wrong_energy": z.detach()}
    if energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,C], got {tuple(energy.shape)}")
    y = _resolve_local_labels(labels, width=energy.size(1), device=energy.device, name="energy labels")
    if y.numel() != energy.size(0):
        raise RuntimeError("labels/energy batch mismatch")
    vm = _resolve_valid_class_mask(valid_mask, width=energy.size(1), device=energy.device)
    if not bool(vm.index_select(0, y).all().item()):
        raise RuntimeError("A target label points to an invalid class-energy column.")
    e = torch.nan_to_num(energy.float(), nan=float(_INVALID_ENERGY), posinf=float(_INVALID_ENERGY), neginf=0.0)
    true_e = e.gather(1, y.view(-1, 1)).squeeze(1)
    rival_valid = vm.view(1, -1).expand_as(e).clone()
    rival_valid.scatter_(1, y.view(-1, 1), False)
    nearest_wrong = e.masked_fill(~rival_valid, float("inf")).min(dim=1).values
    finite = torch.isfinite(nearest_wrong) & torch.isfinite(true_e)
    gap = nearest_wrong - true_e
    per = F.relu(float(margin) - gap)
    if bool(finite.any().item()):
        loss = _class_balanced_mean(per[finite], y[finite]) if class_balanced else per[finite].mean()
        violation = _class_balanced_mean((gap[finite] < float(margin)).float(), y[finite]) if class_balanced else (gap[finite] < float(margin)).float().mean()
        error = _class_balanced_mean((gap[finite] <= 0.0).float(), y[finite]) if class_balanced else (gap[finite] <= 0.0).float().mean()
        mean_gap = _class_balanced_mean(gap[finite], y[finite]) if class_balanced else gap[finite].mean()
        min_gap = gap[finite].min()
    else:
        loss = e.sum() * 0.0
        violation = error = mean_gap = min_gap = e.sum() * 0.0
    return {
        "total": loss,
        "violation_rate": violation.detach(),
        "error_rate": error.detach(),
        "mean_gap": mean_gap.detach(),
        "min_gap": min_gap.detach(),
        "true_energy": true_e[finite].mean().detach() if bool(finite.any().item()) else e.sum().detach() * 0.0,
        "nearest_wrong_energy": nearest_wrong[finite].mean().detach() if bool(finite.any().item()) else e.sum().detach() * 0.0,
    }



def _old_new_invasion_parts(
    energy: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_class_count: Optional[int] = None,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    old_mask: Optional[torch.Tensor] = None,
    new_mask: Optional[torch.Tensor] = None,
    margin: float = 0.25,
    class_balanced: bool = True,
) -> Dict[str, torch.Tensor]:
    ref = energy if torch.is_tensor(energy) else labels
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        z = safe_zero_like(ref)
        return {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach(), "old_to_new_violation": z.detach(), "new_to_old_violation": z.detach()}
    if energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,C], got {tuple(energy.shape)}")
    C = int(energy.size(1))
    if old_mask is None or new_mask is None:
        old_mask, new_mask = _resolve_old_new_masks(
            C, device=energy.device, seen_classes=seen_classes,
            old_classes=old_classes, new_classes=new_classes,
            old_class_count=old_class_count, require_nonempty=False,
        )
    else:
        old_mask = old_mask.to(device=energy.device).bool().flatten()
        new_mask = new_mask.to(device=energy.device).bool().flatten()
    if old_mask.numel() != C or new_mask.numel() != C:
        raise RuntimeError("old_mask/new_mask must match energy width")
    if not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
        z = energy.sum() * 0.0
        return {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach(), "old_to_new_violation": z.detach(), "new_to_old_violation": z.detach()}
    y = _resolve_local_labels(labels, width=C, device=energy.device, name="invasion labels")
    if y.numel() != energy.size(0):
        raise RuntimeError("labels/energy batch mismatch")
    e = torch.nan_to_num(energy.float(), nan=float(_INVALID_ENERGY), posinf=float(_INVALID_ENERGY), neginf=0.0)
    true_e = e.gather(1, y.view(-1, 1)).squeeze(1)
    old_min = e[:, old_mask].min(dim=1).values
    new_min = e[:, new_mask].min(dim=1).values
    is_old = old_mask.index_select(0, y)
    is_new = new_mask.index_select(0, y)
    if not bool((is_old | is_new).all().item()):
        raise RuntimeError("Each target class must belong to exactly one of old_classes or new_classes.")
    opposite = torch.where(is_old, new_min, old_min)
    gap = opposite - true_e
    per = F.relu(float(margin) - gap)
    finite = torch.isfinite(per)
    loss = _class_balanced_mean(per[finite], y[finite]) if bool(finite.any().item()) and class_balanced else (per[finite].mean() if bool(finite.any().item()) else e.sum() * 0.0)
    violation = _class_balanced_mean((gap[finite] < float(margin)).float(), y[finite]) if bool(finite.any().item()) and class_balanced else ((gap[finite] < float(margin)).float().mean() if bool(finite.any().item()) else e.sum() * 0.0)
    mean_gap = _class_balanced_mean(gap[finite], y[finite]) if bool(finite.any().item()) and class_balanced else (gap[finite].mean() if bool(finite.any().item()) else e.sum() * 0.0)
    old_v = (gap[is_old] < float(margin)).float().mean() if bool(is_old.any().item()) else e.sum() * 0.0
    new_v = (gap[is_new] < float(margin)).float().mean() if bool(is_new.any().item()) else e.sum() * 0.0
    return {
        "total": loss,
        "violation_rate": violation.detach(),
        "mean_gap": mean_gap.detach(),
        "old_to_new_violation": old_v.detach(),
        "new_to_old_violation": new_v.detach(),
    }


def base_local_geometry_energy_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    **_: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Removed: the base trainer owns the exact low-rank margin.
    The former helper used isotropic center distance, which is not the
    GeometryBank energy and caused a second, conflicting base-margin path.
    """
    del features, labels
    raise RuntimeError(
        "base_local_geometry_energy_margin_loss is removed. "
        "Use BasePhaseTrainer._base_batch_geometry_energy_margin so phase-0 "
        "training uses the exact rank-normalized GeometryBank energy once."
    )



def incremental_geometry_training_loss(
    *,
    labels: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    energy: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    bank: Optional[Mapping[str, torch.Tensor]] = None,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    labels_are_global: bool = False,
    old_class_count: int = 0,
    ce_weight: float = 1.0,
    joint_old_new_ce_weight: Optional[float] = None,
    geometry_energy_margin_weight: float = 0.30,
    geometry_energy_margin: float = 0.30,
    old_new_invasion_weight: float = 0.50,
    old_new_geometry_margin: float = 0.35,
    label_smoothing: float = 0.0,
    logit_scale: float = 8.0,
    ce_logit_clip: float = 50.0,
    variance_floor: float = 5e-4,
    residual_variance_scale: float = 1.0,
    reliability_energy_weight: float = 0.0,
    normalize_by_dim: bool = True,
    return_parts: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Class-balanced all-seen CE plus exact geometry and invasion margins."""
    ref = energy if torch.is_tensor(energy) else (logits if torch.is_tensor(logits) else features)
    z = safe_zero_like(ref)
    seen = _ordered_unique_ints(seen_classes)
    if energy is None and torch.is_tensor(features) and bank is not None:
        energy = geometry_energy_matrix(
            features,
            bank=bank,
            seen_classes=seen or None,
            variance_floor=float(variance_floor),
            residual_variance_scale=float(residual_variance_scale),
            reliability_energy_weight=float(reliability_energy_weight),
            normalize_by_dim=bool(normalize_by_dim),
            return_parts=False,
            **kwargs,
        )
    width = int(energy.size(1)) if torch.is_tensor(energy) else (int(logits.size(1)) if torch.is_tensor(logits) else 0)
    if width <= 0:
        return {"total": z} if return_parts else z
    y = _resolve_local_labels(
        labels, width=width, device=(energy.device if torch.is_tensor(energy) else logits.device),
        seen_classes=seen or None, labels_are_global=bool(labels_are_global), name="incremental labels",
    )
    valid = None
    if torch.is_tensor(energy):
        valid = torch.isfinite(energy).all(dim=0) & (energy < 0.5 * _INVALID_ENERGY).any(dim=0)
    if logits is None and torch.is_tensor(energy):
        logits = _energy_to_logits(energy, valid_mask=valid, logit_scale=float(logit_scale), clip=float(ce_logit_clip))
    ce_w = float(joint_old_new_ce_weight) if joint_old_new_ce_weight is not None else float(ce_weight)
    ce = _ce_from_logits(logits, y, label_smoothing=float(label_smoothing), clip=float(ce_logit_clip), class_balanced=True) if torch.is_tensor(logits) else z
    if not torch.is_tensor(energy) and torch.is_tensor(logits):
        energy = -logits.float() / max(float(logit_scale), 1e-6)
    margin_parts = _energy_margin_parts(energy, y, margin=float(geometry_energy_margin), valid_mask=valid, class_balanced=True)
    invasion_parts = _old_new_invasion_parts(
        energy, y, old_class_count=int(old_class_count), seen_classes=seen or None,
        old_classes=old_classes, new_classes=new_classes,
        margin=float(old_new_geometry_margin), class_balanced=True,
    )
    energy_margin_total = _scalar(margin_parts["total"], ref)
    invasion_total = _scalar(invasion_parts["total"], ref)
    total = ce_w * ce + float(geometry_energy_margin_weight) * energy_margin_total + float(old_new_invasion_weight) * invasion_total
    if not return_parts:
        return total
    return {
        "total": total,
        "ce": ce.detach(),
        "incremental_ce": ce.detach(),
        "geometry_energy_margin": energy_margin_total.detach(),
        "old_new_invasion": invasion_total.detach(),
        "energy_margin_violation": margin_parts["violation_rate"],
        "energy_error_rate": margin_parts["error_rate"],
        "energy_margin_gap": margin_parts["mean_gap"],
        "energy_margin_min_gap": margin_parts["min_gap"],
        "old_new_invasion_violation": invasion_parts["violation_rate"],
        "old_new_invasion_gap": invasion_parts["mean_gap"],
        "old_to_new_violation": invasion_parts["old_to_new_violation"],
        "new_to_old_violation": invasion_parts["new_to_old_violation"],
        "phase_is_base": torch.tensor(0.0, device=total.device, dtype=total.dtype),
        "phase_is_incremental": torch.tensor(1.0, device=total.device, dtype=total.dtype),
        "rank": energy_margin_total.detach(),
        "admission": invasion_total.detach(),
        "subspace": z.detach(),
        "volume": z.detach(),
        "trust": z.detach(),
        "violation_rate": margin_parts["violation_rate"],
    }


# -----------------------------------------------------------------------------
# Unified base objective
# -----------------------------------------------------------------------------


def base_geometry_preparation_loss(
    *,
    logits: Optional[torch.Tensor] = None,
    features: torch.Tensor,
    labels: torch.Tensor,
    key_features: Optional[torch.Tensor] = None,
    band_summary: Optional[torch.Tensor] = None,
    spectral_summary: Optional[torch.Tensor] = None,
    spectral_summary_is_physical: bool = False,
    ce_weight: float = 0.0,
    base_geometry_weight: float = 1.0,
    label_smoothing: float = 0.0,
    gics_weight: float = 0.20,
    gics_temperature: float = 0.07,
    pgr_weight: float = 0.10,
    pgr_compact_weight: float = 0.15,
    pgr_center_weight: float = 0.20,
    pgr_subspace_weight: float = 0.10,
    pgr_band_weight: float = 0.05,
    pgr_volume_weight: float = 0.05,
    pgr_center_margin: float = 1.05,
    pgr_max_band_similarity: float = 0.75,
    pgr_max_class_variance: float = 0.75,
    pgr_min_class_variance: float = 0.015,
    pgr_max_subspace_overlap: float = 0.50,
    subspace_rank: int = 3,
    min_class_samples: int = 3,
    subspace_min_samples: int = 6,
    spectral_shape_weight: float = 0.05,
    max_spectral_shape_similarity: float = 0.75,
    spectral_shape_risk_weight: float = 1.0,
    require_physical_summary: bool = True,
    base_energy_margin_weight: float = 0.15,
    base_energy_margin: float = 0.25,
    base_energy_variance_floor: float = 5e-4,
    return_parts: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Base CE/GICS/PGR/spectral reserve.

    The exact low-rank base energy margin is intentionally owned by
    BasePhaseTrainer._base_batch_geometry_energy_margin. Keeping another local
    approximation here would double-count the term and use a different energy.
    Margin arguments are accepted only for API compatibility.
    """
    del base_energy_margin_weight, base_energy_margin, base_energy_variance_floor
    if features is None or not torch.is_tensor(features) or features.numel() == 0:
        z = safe_zero_like(features)
        out = {"total": z, "ce": z, "base_geometry": z}
        for key in (
            "base_gics", "base_gics_weighted", "base_gics_anchors", "base_gics_pos",
            "base_pgr", "base_pgr_weighted", "base_compact", "base_center", "base_subspace", "base_band", "base_volume",
            "base_pgr_valid_class_count", "base_pgr_subspace_pair_count", "base_pgr_band_pair_count", "base_pgr_volume_factor",
            "base_pgr_subspace_max_overlap", "base_pgr_band_max_similarity", "base_pgr_band_guided_conflict_mean", "base_pgr_band_guided_conflict_max",
            "base_spectral_shape", "base_spectral_shape_raw", "base_spectral_shape_mean_similarity", "base_spectral_shape_max_similarity",
            "base_spectral_shape_pair_count", "base_spectral_shape_active", "base_spectral_shape_guided_conflict_mean", "base_spectral_shape_guided_conflict_max",
            "base_energy_margin", "base_energy_margin_weighted", "base_energy_margin_violation", "base_energy_margin_gap",
        ):
            out[key] = z
        return out if return_parts else z
    if features.dim() != 2:
        raise ValueError(f"base features must be [B,D], got {tuple(features.shape)}")
    _require_finite_tensor(features, "base.features")
    labels = _as_1d_long(labels, device=features.device)
    if labels.numel() != features.size(0):
        raise ValueError(f"base labels/features mismatch: labels={labels.numel()}, features={features.size(0)}")

    ce = safe_zero_like(features)
    if ce_weight > 0.0:
        if logits is None or not torch.is_tensor(logits):
            raise ValueError("logits are required when ce_weight > 0.")
        ce = _ce_from_logits(logits, labels, label_smoothing=float(label_smoothing), class_balanced=True)

    band_for_pgr = spectral_summary if (
        torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0 and bool(spectral_summary_is_physical)
    ) else band_summary
    gics = base_geometry_involved_contrastive_loss(
        features, labels, key_features=key_features, weight=float(gics_weight),
        temperature=float(gics_temperature), return_parts=True,
    )
    pgr = prospective_geometry_reserve_loss(
        features, labels, band_summary=band_for_pgr, weight=float(pgr_weight),
        compact_weight=float(pgr_compact_weight), center_weight=float(pgr_center_weight),
        subspace_weight=float(pgr_subspace_weight), band_weight=float(pgr_band_weight),
        volume_weight=float(pgr_volume_weight), center_margin=float(pgr_center_margin),
        min_class_samples=int(min_class_samples), subspace_min_samples=int(subspace_min_samples),
        subspace_rank=int(subspace_rank), max_band_similarity=float(pgr_max_band_similarity),
        max_class_variance=float(pgr_max_class_variance), min_class_variance=float(pgr_min_class_variance),
        max_subspace_overlap=float(kwargs.get("subspace_overlap_max", pgr_max_subspace_overlap)), return_parts=True,
    )
    shape = spectral_shape_discrimination_loss(
        spectral_summary, labels, features=features,
        spectral_summary_is_physical=bool(spectral_summary_is_physical),
        require_physical_summary=bool(require_physical_summary), min_samples=int(min_class_samples),
        max_shape_similarity=float(max_spectral_shape_similarity), risk_center_margin=float(pgr_center_margin),
        risk_weight=float(spectral_shape_risk_weight), return_parts=True,
    )
    gics_total = _scalar(gics.get("total", safe_zero_like(features)), features)
    pgr_total = _scalar(pgr.get("total", safe_zero_like(features)), features)
    shape_raw = _scalar(shape.get("total", safe_zero_like(features)), features)
    shape_total = float(spectral_shape_weight) * shape_raw
    base_geometry = float(base_geometry_weight) * (gics_total + pgr_total + shape_total)
    total = float(ce_weight) * ce + base_geometry
    if not return_parts:
        return total
    zero = features.sum() * 0.0
    spectral_active = bool(spectral_summary_is_physical) and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0
    return {
        "total": total,
        "ce": ce.detach(),
        "base_geometry": base_geometry,
        "base_gics": _scalar(gics.get("gics", zero), features).detach(),
        "base_gics_weighted": gics_total.detach(),
        "base_gics_anchors": _scalar(gics.get("valid_anchors", zero), features).detach(),
        "base_gics_pos": _scalar(gics.get("mean_positive_count", zero), features).detach(),
        "base_pgr": _scalar(pgr.get("pgr", zero), features).detach(),
        "base_pgr_weighted": pgr_total.detach(),
        "base_compact": _scalar(pgr.get("compact", zero), features).detach(),
        "base_center": _scalar(pgr.get("center", zero), features).detach(),
        "base_subspace": _scalar(pgr.get("subspace", zero), features).detach(),
        "base_band": _scalar(pgr.get("band", zero), features).detach(),
        "base_volume": _scalar(pgr.get("volume", zero), features).detach(),
        "base_spectral_shape": shape_total.detach(),
        "base_spectral_shape_raw": shape_raw.detach(),
        "base_spectral_shape_mean_similarity": _scalar(shape.get("mean_similarity", zero), features).detach(),
        "base_spectral_shape_max_similarity": _scalar(shape.get("max_similarity", zero), features).detach(),
        "base_spectral_shape_pair_count": _scalar(shape.get("pair_count", zero), features).detach(),
        "base_spectral_shape_active": torch.tensor(float(spectral_active), device=features.device, dtype=features.dtype),
        "base_energy_margin": zero.detach(),
        "base_energy_margin_weighted": zero.detach(),
        "base_energy_margin_violation": zero.detach(),
        "base_energy_margin_gap": zero.detach(),
        "base_pgr_valid_class_count": _scalar(pgr.get("valid_class_count", zero), features).detach(),
        "base_pgr_subspace_pair_count": _scalar(pgr.get("subspace_pair_count", zero), features).detach(),
        "base_pgr_band_pair_count": _scalar(pgr.get("band_pair_count", zero), features).detach(),
        "base_pgr_volume_factor": _scalar(pgr.get("volume_factor", zero), features).detach(),
        "base_pgr_subspace_max_overlap": _scalar(pgr.get("subspace_max_overlap", zero), features).detach(),
        "base_pgr_band_max_similarity": _scalar(pgr.get("band_max_similarity", zero), features).detach(),
        "base_pgr_band_guided_conflict_mean": _scalar(pgr.get("band_guided_conflict_mean", zero), features).detach(),
        "base_pgr_band_guided_conflict_max": _scalar(pgr.get("band_guided_conflict_max", zero), features).detach(),
        "base_spectral_shape_guided_conflict_mean": _scalar(shape.get("guided_conflict_mean", zero), features).detach(),
        "base_spectral_shape_guided_conflict_max": _scalar(shape.get("guided_conflict_max", zero), features).detach(),
        "base_energy_margin_owned_by_trainer": torch.tensor(1.0, device=features.device, dtype=features.dtype),
    }



def unified_spectral_geometry_loss(
    *,
    phase: str,
    labels: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    energy: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    key_features: Optional[torch.Tensor] = None,
    band_summary: Optional[torch.Tensor] = None,
    spectral_summary: Optional[torch.Tensor] = None,
    spectral_summary_is_physical: bool = False,
    bank: Optional[Mapping[str, torch.Tensor]] = None,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    labels_are_global: bool = False,
    old_class_count: int = 0,
    ce_weight: Optional[float] = None,
    joint_old_new_ce_weight: Optional[float] = None,
    geometry_energy_margin_weight: float = 0.30,
    geometry_energy_margin: float = 0.30,
    old_new_invasion_weight: float = 0.50,
    old_new_geometry_margin: float = 0.35,
    logit_scale: float = 8.0,
    label_smoothing: float = 0.0,
    return_parts: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Single public loss router for the current SCTGR architecture."""
    p = str(phase).strip().lower()
    if p in {"base", "phase0", "phase_0", "0"}:
        return base_geometry_preparation_loss(
            logits=logits, features=features, labels=labels, key_features=key_features,
            band_summary=band_summary, spectral_summary=spectral_summary,
            spectral_summary_is_physical=spectral_summary_is_physical,
            ce_weight=float(0.0 if ce_weight is None else ce_weight),
            label_smoothing=float(label_smoothing), return_parts=return_parts, **kwargs,
        )
    if p in {"incremental", "inc", "phase_inc", "phase1", "phase_1", "1", "2", "3", "4", "5"} or p.startswith("phase"):
        return incremental_geometry_training_loss(
            labels=labels, logits=logits, energy=energy, features=features, bank=bank,
            seen_classes=seen_classes, old_classes=old_classes, new_classes=new_classes,
            labels_are_global=bool(labels_are_global), old_class_count=int(old_class_count),
            ce_weight=float(1.0 if ce_weight is None else ce_weight),
            joint_old_new_ce_weight=joint_old_new_ce_weight,
            geometry_energy_margin_weight=float(geometry_energy_margin_weight),
            geometry_energy_margin=float(geometry_energy_margin),
            old_new_invasion_weight=float(old_new_invasion_weight),
            old_new_geometry_margin=float(old_new_geometry_margin),
            logit_scale=float(logit_scale), label_smoothing=float(label_smoothing),
            return_parts=return_parts, **kwargs,
        )
    raise RuntimeError(f"Unsupported phase={phase!r}. Use 'base' or 'incremental'.")


class UnifiedSpectralGeometryLoss:
    """Thin callable wrapper for compatibility with older code."""

    def __init__(self, **defaults: Any) -> None:
        self.defaults = dict(defaults)

    def __call__(self, **kwargs: Any):
        merged = dict(self.defaults)
        merged.update(kwargs)
        return unified_spectral_geometry_loss(**merged)


# -----------------------------------------------------------------------------
# Incremental margin losses used by SCTGR
# -----------------------------------------------------------------------------


def geometry_energy_margin_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    seen_classes: Optional[Iterable[int]] = None,
    labels_are_global: bool = False,
    class_balanced: bool = True,
) -> torch.Tensor:
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return safe_zero_like(labels if torch.is_tensor(labels) else None)
    y = _resolve_local_labels(labels, width=energy.size(1), device=energy.device, seen_classes=seen_classes, labels_are_global=labels_are_global)
    return _energy_margin_parts(energy, y, margin=float(margin), valid_mask=valid_mask, class_balanced=class_balanced)["total"]



def old_new_invasion_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    old_class_count: Optional[int] = None,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    labels_are_global: bool = False,
    old_mask: Optional[torch.Tensor] = None,
    new_mask: Optional[torch.Tensor] = None,
    class_balanced: bool = True,
) -> torch.Tensor:
    del valid_mask
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return safe_zero_like(labels if torch.is_tensor(labels) else None)
    y = _resolve_local_labels(labels, width=energy.size(1), device=energy.device, seen_classes=seen_classes, labels_are_global=labels_are_global)
    return _old_new_invasion_parts(
        energy, y, old_class_count=old_class_count, seen_classes=seen_classes,
        old_classes=old_classes, new_classes=new_classes, old_mask=old_mask,
        new_mask=new_mask, margin=float(margin), class_balanced=class_balanced,
    )["total"]


# -----------------------------------------------------------------------------
# Base diagnostics
# -----------------------------------------------------------------------------

@torch.no_grad()
def base_center_overlap_diagnostics(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalize: bool = True,
    min_samples: int = 2,
) -> Dict[str, torch.Tensor]:
    if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
        z = safe_zero_like(features)
        return {"compact": z, "mean_center_margin": z, "min_center_margin": z, "num_classes": z}

    z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
    y = _as_1d_long(labels, device=z.device)
    centers, _, _ = _class_centers(z, y, min_samples=min_samples)

    compact_terms = []
    for cls in torch.unique(y, sorted=True):
        m = y == cls
        if int(m.sum().item()) >= int(min_samples):
            xc = z[m]
            compact_terms.append((xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean())
    compact = torch.stack(compact_terms).mean() if compact_terms else z.sum() * 0.0

    if centers.size(0) < 2:
        mean_margin = z.sum() * 0.0
        min_margin = z.sum() * 0.0
    else:
        dist = torch.cdist(centers, centers, p=2)
        eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        pair = dist[~eye]
        mean_margin = pair.mean()
        min_margin = pair.min()

    return {
        "compact": compact.detach(),
        "mean_center_margin": mean_margin.detach(),
        "min_center_margin": min_margin.detach(),
        "num_classes": torch.tensor(float(centers.size(0)), device=z.device, dtype=z.dtype),
    }


@torch.no_grad()
def base_gics_diagnostics(features: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
    out = base_geometry_involved_contrastive_loss(features, labels, weight=1.0, return_parts=True, **kwargs)
    return {
        "gics": out["gics"].detach(),
        "valid_anchors": out["valid_anchors"].detach(),
        "positive_pairs": out["mean_positive_count"].detach(),
    }


def base_supcon_diagnostics(*args: Any, **kwargs: Any):
    return base_gics_diagnostics(*args, **kwargs)


# -----------------------------------------------------------------------------
# Disabled legacy boundary helper
# -----------------------------------------------------------------------------

def sample_boundary_geometry_features(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "sample_boundary_geometry_features is not part of  main path. "
        "Use GeometryBank synthetic replay, not boundary replay."
    )















# from __future__ import annotations
# from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
# import torch
# import torch.nn.functional as F


# _EPS = 1e-12
# _INVALID_ENERGY = 1e6


# # -----------------------------------------------------------------------------
# # Basic utilities
# # -----------------------------------------------------------------------------

# def safe_zero_like(
#     ref: Optional[torch.Tensor] = None,
#     *,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> torch.Tensor:
#     if torch.is_tensor(ref):
#         return ref.sum() * 0.0
#     return torch.tensor(
#         0.0,
#         device=device if device is not None else torch.device("cpu"),
#         dtype=dtype if dtype is not None else torch.float32,
#     )


# def _require_finite_tensor(x: torch.Tensor, name: str) -> None:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     if x.numel() == 0:
#         return
#     if not torch.isfinite(x).all():
#         bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
#         raise RuntimeError(f"{name}: tensor contains {bad} NaN/Inf values.")


# def _as_1d_long(labels: torch.Tensor, *, device: torch.device, name: str = "labels") -> torch.Tensor:
#     if labels is None or not torch.is_tensor(labels):
#         raise TypeError(f"{name} must be a tensor.")
#     return labels.to(device=device).long().flatten()


# def _scalar(value: Any, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#     if torch.is_tensor(value):
#         if value.numel() == 1:
#             return value.reshape(())
#         return value.float().mean()
#     if isinstance(value, (int, float)):
#         if torch.is_tensor(ref):
#             return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
#         return torch.tensor(float(value), dtype=torch.float32)
#     return safe_zero_like(ref)


# def _class_centers(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     min_samples: int = 2,
#     normalize_centers: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         device = features.device if torch.is_tensor(features) else torch.device("cpu")
#         dtype = features.dtype if torch.is_tensor(features) else torch.float32
#         return (
#             torch.empty(0, 0, device=device, dtype=dtype),
#             torch.empty(0, device=device, dtype=torch.long),
#             torch.empty(0, device=device, dtype=dtype),
#         )

#     if features.dim() != 2:
#         raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")

#     y = _as_1d_long(labels, device=features.device)
#     if y.numel() != features.size(0):
#         raise ValueError(f"labels/features mismatch: labels={y.numel()}, features={features.size(0)}")

#     centers, class_ids, counts = [], [], []
#     for cls in torch.unique(y, sorted=True):
#         mask = y == cls
#         n = int(mask.sum().item())
#         if n < int(min_samples):
#             continue
#         c = features[mask].mean(dim=0)
#         if normalize_centers:
#             c = F.normalize(c, dim=0, eps=1e-6)
#         centers.append(c)
#         class_ids.append(cls)
#         counts.append(torch.tensor(float(n), device=features.device, dtype=features.dtype))

#     if not centers:
#         return (
#             torch.empty(0, features.size(1), device=features.device, dtype=features.dtype),
#             torch.empty(0, device=features.device, dtype=torch.long),
#             torch.empty(0, device=features.device, dtype=features.dtype),
#         )

#     return torch.stack(centers, dim=0), torch.stack(class_ids).long(), torch.stack(counts)


# def _pairwise_center_margin_loss(centers: torch.Tensor, margin: float) -> torch.Tensor:
#     if centers is None or not torch.is_tensor(centers) or centers.numel() == 0 or centers.size(0) < 2:
#         return safe_zero_like(centers)
#     dist = torch.cdist(centers, centers, p=2)
#     eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
#     pair = dist[~eye]
#     if pair.numel() == 0:
#         return centers.sum() * 0.0
#     return F.relu(float(margin) - pair).pow(2).mean()


# def _pad_to_width(x: torch.Tensor, width: int) -> torch.Tensor:
#     if x.size(1) == width:
#         return x
#     if x.size(1) > width:
#         return x[:, :width]
#     return F.pad(x, (0, int(width) - int(x.size(1))))


# # -----------------------------------------------------------------------------
# # Spectral / band profiles
# # -----------------------------------------------------------------------------

# def _spectral_derivatives(spectral_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     if spectral_summary.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
#     if spectral_summary.size(1) < 2:
#         z = spectral_summary.new_zeros((spectral_summary.size(0), 1))
#         return z, z
#     d1 = spectral_summary[:, 1:] - spectral_summary[:, :-1]
#     if d1.size(1) < 2:
#         d2 = d1.new_zeros((d1.size(0), 1))
#     else:
#         d2 = d1[:, 1:] - d1[:, :-1]
#     return d1, d2


# def _spectral_profile_descriptor(spectral_summary: torch.Tensor) -> torch.Tensor:
#     """Derivative-aware descriptor for physical spectral-shape comparison.

#     Raw HSI spectra may be normalized and may contain signed values.  Direct
#     softmax over signed spectra destroys absorption/reflectance shape.  This
#     descriptor preserves curve shape by standardizing each spectrum and appending
#     first/second derivative information.
#     """
#     if spectral_summary.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     s = s - s.mean(dim=1, keepdim=True)
#     s = s / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
#     d1, d2 = _spectral_derivatives(s)
#     desc = torch.cat([F.normalize(s, dim=1, eps=1e-6), F.normalize(d1, dim=1, eps=1e-6), F.normalize(d2, dim=1, eps=1e-6)], dim=1)
#     return torch.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


# def _band_importance_profile(band_summary: torch.Tensor) -> torch.Tensor:
#     """Convert raw spectra or band summaries to a non-negative band profile.

#     This is the correct base regularizer target for HSI: it compares where the
#     spectrum changes/absorbs, not a hidden classifier branch.  It works for raw
#     physical spectra and is still safe for reduced non-physical summaries.
#     """
#     if band_summary is None or not torch.is_tensor(band_summary):
#         raise TypeError("band_summary must be a tensor.")
#     if band_summary.dim() != 2:
#         raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")

#     b = torch.nan_to_num(band_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     B, S = b.shape
#     if S <= 0:
#         return b

#     # Per-sample standardization preserves spectral shape under dataset scaling.
#     z = b - b.mean(dim=1, keepdim=True)
#     z = z / z.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
#     d1, d2 = _spectral_derivatives(z)
#     d1e = _pad_to_width(d1.abs(), S)
#     d2e = _pad_to_width(d2.abs(), S)
#     profile = z.abs() + 0.50 * d1e + 0.25 * d2e
#     profile = profile.clamp_min(0.0)
#     uniform = torch.full_like(profile, 1.0 / float(max(S, 1)))
#     denom = profile.sum(dim=1, keepdim=True)
#     profile = torch.where(denom > 1e-8, profile / denom.clamp_min(1e-8), uniform)
#     return torch.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)


# # Backward-compatible name used by old code.
# def _normalize_band_summary(band_summary: torch.Tensor) -> torch.Tensor:
#     return _band_importance_profile(band_summary)


# def _positive_cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     a = F.normalize(a, dim=1, eps=1e-6)
#     b = F.normalize(b, dim=1, eps=1e-6)
#     return (a @ b.t()).clamp(0.0, 1.0)


# # -----------------------------------------------------------------------------
# # GICS: Geometry-Involved Class Separation
# # -----------------------------------------------------------------------------

# def base_geometry_involved_contrastive_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     key_features: Optional[torch.Tensor] = None,
#     weight: float = 0.20,
#     temperature: float = 0.07,
#     same_class_positive: bool = True,
#     class_balanced: bool = True,
#     detach_key: bool = True,
#     normalize: bool = True,
#     return_parts: bool = True,
#     **_: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Base-phase GICS on canonical projected z-space."""
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         out = {"total": z, "gics": z, "weighted_gics": z, "valid_anchors": z, "num_anchors": z, "mean_positive_count": z}
#         return out if return_parts else z

#     if features.dim() != 2:
#         raise ValueError(f"GICS expects projected features [B,D], got {tuple(features.shape)}")
#     _require_finite_tensor(features, "gics.features")

#     zq = features
#     explicit_key = key_features is not None and torch.is_tensor(key_features) and key_features.numel() > 0
#     zk = key_features if explicit_key else features
#     if zk.dim() != 2:
#         raise ValueError(f"key_features must be [B,D], got {tuple(zk.shape)}")
#     if zk.size(0) != zq.size(0) or zk.size(1) != zq.size(1):
#         raise ValueError(f"key_features shape mismatch: query={tuple(zq.shape)}, key={tuple(zk.shape)}")
#     if detach_key:
#         zk = zk.detach()

#     y = _as_1d_long(labels, device=zq.device)
#     if y.numel() != zq.size(0):
#         raise ValueError(f"GICS labels/features mismatch: labels={y.numel()}, features={zq.size(0)}")

#     q = F.normalize(zq, dim=1, eps=1e-6) if normalize else zq
#     k = F.normalize(zk.to(device=zq.device, dtype=zq.dtype), dim=1, eps=1e-6) if normalize else zk.to(device=zq.device, dtype=zq.dtype)

#     logits = q @ k.t()
#     logits = logits / max(float(temperature), 1e-6)
#     logits = logits - logits.max(dim=1, keepdim=True).values.detach()

#     B = zq.size(0)
#     diag = torch.eye(B, device=zq.device, dtype=torch.bool)
#     positive = y.view(-1, 1).eq(y.view(1, -1)) if same_class_positive else diag.clone()

#     if explicit_key:
#         positive = positive | diag
#         denom_mask = torch.ones_like(positive, dtype=torch.bool)
#     else:
#         positive = positive & (~diag)
#         denom_mask = ~diag

#     pos_count = positive.float().sum(dim=1)
#     valid = pos_count > 0

#     if not bool(valid.any().item()):
#         gics = features.sum() * 0.0
#     else:
#         exp_logits = torch.exp(logits).masked_fill(~denom_mask, 0.0)
#         denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
#         log_prob = logits - denom.log()
#         per = -(positive.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
#         per = per[valid]
#         yv = y[valid]
#         if per.numel() == 0:
#             gics = features.sum() * 0.0
#         elif class_balanced:
#             class_terms = []
#             for c in torch.unique(yv, sorted=True):
#                 cm = yv == c
#                 if bool(cm.any().item()):
#                     class_terms.append(per[cm].mean())
#             gics = torch.stack(class_terms).mean() if class_terms else features.sum() * 0.0
#         else:
#             gics = per.mean()

#     total = float(weight) * gics
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "gics": gics.detach(),
#         "weighted_gics": total.detach(),
#         "valid_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
#         "num_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
#         "mean_positive_count": pos_count[valid].float().mean().detach() if bool(valid.any().item()) else features.sum().detach() * 0.0,
#     }


# # Backward-compatible aliases.
# def base_fcs_geometry_contrastive_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_geometry_involved_contrastive_separation_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_supervised_contrastive_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_hsi_supervised_contrastive_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # PGR: Prospective Geometry Reserve
# # -----------------------------------------------------------------------------

# def _batch_subspace_overlap_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     rank: int = 3,
#     min_samples: int = 6,
#     max_overlap: float = 0.50,
#     normalize: bool = True,
#     include_mean_overlap: bool = True,
#     return_parts: bool = False,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """Margin-based subspace reserve.

#     The old loss minimized average overlap only.  That can look nonzero in logs
#     while leaving a single highly-overlapping class pair that breaks incremental
#     descriptor insertion.  This version explicitly penalizes pair overlaps above
#     max_overlap while still reporting mean/max overlap.
#     """
#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         if return_parts:
#             return {"total": z, "pair_count": z, "valid_class_count": z, "mean_overlap": z, "max_overlap": z}
#         return z
#     if features.dim() != 2:
#         raise ValueError(f"subspace loss expects [B,D], got {tuple(features.shape)}")

#     y = _as_1d_long(labels, device=features.device)
#     z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
#     _, D = z.shape

#     bases = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         n = int(m.sum().item())
#         if n < int(min_samples):
#             continue
#         xc = z[m] - z[m].mean(dim=0, keepdim=True)
#         r = max(1, min(int(rank), n - 1, D))
#         try:
#             _, s, vh = torch.linalg.svd(xc, full_matrices=False)
#         except RuntimeError:
#             continue
#         if s.numel() == 0 or float(s[0].detach().abs().item()) <= 1e-8:
#             continue
#         keep = (s[:r] / s[0].clamp_min(1e-8)) > 1e-3
#         if bool(keep.any().item()):
#             bases.append(vh[:r][keep].t().contiguous())

#     if len(bases) < 2:
#         loss = features.sum() * 0.0
#         overlaps = features.new_empty(0)
#         pair_count = 0
#     else:
#         overlap_list = []
#         for i in range(len(bases)):
#             for j in range(i + 1, len(bases)):
#                 Ui, Uj = bases[i], bases[j]
#                 denom = float(max(min(Ui.size(1), Uj.size(1)), 1))
#                 ov = (Ui.t() @ Uj).pow(2).sum() / denom
#                 overlap_list.append(ov)
#         overlaps = torch.stack(overlap_list) if overlap_list else features.new_empty(0)
#         pair_count = int(overlaps.numel())
#         if pair_count == 0:
#             loss = features.sum() * 0.0
#         else:
#             margin_loss = F.relu(overlaps - float(max_overlap)).pow(2).mean()
#             mean_loss = 0.10 * overlaps.mean() if include_mean_overlap else overlaps.sum() * 0.0
#             loss = margin_loss + mean_loss

#     if return_parts:
#         return {
#             "total": loss,
#             "pair_count": torch.tensor(float(pair_count), device=features.device, dtype=features.dtype),
#             "valid_class_count": torch.tensor(float(len(bases)), device=features.device, dtype=features.dtype),
#             "mean_overlap": overlaps.mean().detach() if overlaps.numel() else features.sum().detach() * 0.0,
#             "max_overlap": overlaps.max().detach() if overlaps.numel() else features.sum().detach() * 0.0,
#         }
#     return loss



# def risk_aware_band_discrimination_loss(
#     band_summary: Optional[torch.Tensor],
#     labels: torch.Tensor,
#     *,
#     features: Optional[torch.Tensor] = None,
#     min_samples: int = 3,
#     max_band_similarity: float = 0.75,
#     risk_center_margin: float = 1.0,
#     risk_weight: float = 1.0,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Band-guided feature-geometry reserve for HSI.

#     Important: raw physical spectra are *not* trainable. Therefore this loss
#     must not try to make raw band profiles dissimilar directly.  Instead, high
#     band similarity marks a hard class pair and increases the gradient on the
#     learned feature geometry for that pair.

#     Loss = hard_band_similarity.detach() * feature_center_conflict(features)

#     This keeps band information in the architecture while preserving the main
#     PG-RGA rule: inference and replay remain geometry-energy based.
#     """
#     ref = features if torch.is_tensor(features) else (band_summary if torch.is_tensor(band_summary) else labels)
#     if band_summary is None or not torch.is_tensor(band_summary) or band_summary.numel() == 0:
#         z = safe_zero_like(ref)
#         out = {
#             "total": z,
#             "band": z,
#             "pair_count": z,
#             "valid_class_count": z,
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     if band_summary.dim() != 2:
#         raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")

#     y = _as_1d_long(labels, device=band_summary.device)
#     if y.numel() != band_summary.size(0):
#         raise ValueError(f"band labels/batch mismatch: labels={y.numel()}, band={band_summary.size(0)}")

#     b = _band_importance_profile(band_summary)
#     b_centers, class_ids, _ = _class_centers(b, y, min_samples=min_samples, normalize_centers=True)
#     if b_centers.size(0) < 2:
#         z = band_summary.sum() * 0.0
#         out = {
#             "total": z,
#             "band": z,
#             "pair_count": z,
#             "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     sim = (b_centers @ b_centers.t()).clamp(0.0, 1.0)
#     eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
#     pair_sim = sim[~eye]
#     hard_band = F.relu(pair_sim - float(max_band_similarity)) / max(1.0 - float(max_band_similarity), 1e-6)
#     hard_band = hard_band.detach()

#     # The only trainable part is the learned feature geometry.  If features are
#     # missing or class ids cannot be aligned, the band term reports similarity
#     # but contributes zero gradient instead of applying a useless constant loss.
#     if features is not None and torch.is_tensor(features) and features.numel() > 0:
#         zf = F.normalize(features.to(device=band_summary.device, dtype=band_summary.dtype), dim=1, eps=1e-6)
#         f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
#         if f_centers.size(0) == b_centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
#             dist = torch.cdist(f_centers, f_centers, p=2)
#             feature_conflict = F.relu(float(risk_center_margin) - dist) / max(float(risk_center_margin), 1e-6)
#             feature_conflict = feature_conflict[~eye]
#             guided = hard_band * feature_conflict
#             loss_vec = hard_band * feature_conflict.pow(2)
#             loss = float(risk_weight) * (loss_vec.mean() if loss_vec.numel() > 0 else features.sum() * 0.0)
#         else:
#             guided = torch.zeros_like(pair_sim)
#             loss = features.sum() * 0.0
#     else:
#         guided = torch.zeros_like(pair_sim)
#         loss = band_summary.sum() * 0.0

#     if not return_parts:
#         return loss
#     return {
#         "total": loss,
#         "band": loss.detach(),
#         "pair_count": torch.tensor(float(pair_sim.numel()), device=band_summary.device, dtype=band_summary.dtype),
#         "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
#         "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
#         "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
#         "guided_conflict_mean": guided.mean().detach() if guided.numel() > 0 else band_summary.sum().detach() * 0.0,
#         "guided_conflict_max": guided.max().detach() if guided.numel() > 0 else band_summary.sum().detach() * 0.0,
#     }


# def prospective_geometry_reserve_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     band_summary: Optional[torch.Tensor] = None,
#     weight: float = 0.10,
#     compact_weight: float = 0.15,
#     center_weight: float = 0.20,
#     subspace_weight: float = 0.10,
#     band_weight: float = 0.05,
#     volume_weight: float = 0.05,
#     center_margin: float = 1.05,
#     min_class_samples: int = 3,
#     subspace_min_samples: int = 6,
#     subspace_rank: int = 3,
#     max_band_similarity: float = 0.75,
#     max_class_variance: float = 0.75,
#     min_class_variance: float = 0.015,
#     max_subspace_overlap: float = 0.50,
#     normalize_features: bool = True,
#     adaptive_component_weights: bool = True,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Base-phase PGR.

#     Active terms:
#       - compactness: same-class spread control
#       - center reserve: class centers separated by margin
#       - subspace reserve: explicit margin on tangent subspace overlap
#       - band reserve: risky classes avoid identical spectral-band profiles
#       - volume reserve: avoids both broad blobs and collapsed zero-volume rows
#     """
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z0 = safe_zero_like(features)
#         out = {
#             "total": z0, "pgr": z0, "weighted_pgr": z0,
#             "compact": z0, "center": z0, "subspace": z0, "band": z0, "volume": z0,
#             "valid_class_count": z0, "unique_class_count": z0,
#             "subspace_pair_count": z0, "band_pair_count": z0,
#             "compact_factor": z0, "center_factor": z0, "subspace_factor": z0,
#             "band_factor": z0, "volume_factor": z0,
#             "subspace_mean_overlap": z0, "subspace_max_overlap": z0,
#             "band_mean_similarity": z0, "band_max_similarity": z0,
#         }
#         return out if return_parts else z0

#     if features.dim() != 2:
#         raise ValueError(f"PGR expects features [B,D], got {tuple(features.shape)}")
#     _require_finite_tensor(features, "pgr.features")

#     y = _as_1d_long(labels, device=features.device)
#     if y.numel() != features.size(0):
#         raise ValueError(f"PGR labels/features mismatch: labels={y.numel()}, features={features.size(0)}")

#     # Let explicit aliases in kwargs override defaults without requiring trainer changes.
#     if "pgr_max_subspace_overlap" in kwargs:
#         max_subspace_overlap = float(kwargs["pgr_max_subspace_overlap"])
#     if "subspace_overlap_max" in kwargs:
#         max_subspace_overlap = float(kwargs["subspace_overlap_max"])
#     if "pgr_min_class_variance" in kwargs:
#         min_class_variance = float(kwargs["pgr_min_class_variance"])

#     z = F.normalize(features, dim=1, eps=1e-6) if normalize_features else features

#     compact_terms = []
#     volume_terms = []
#     class_vars = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         if int(m.sum().item()) < int(min_class_samples):
#             continue
#         xc = z[m]
#         var = (xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
#         compact_terms.append(var)
#         class_vars.append(var.detach())
#         broad = F.relu(var - float(max_class_variance)).pow(2)
#         collapsed = F.relu(float(min_class_variance) - var).pow(2)
#         volume_terms.append(broad + collapsed)

#     compact = torch.stack(compact_terms).mean() if compact_terms else features.sum() * 0.0
#     volume = torch.stack(volume_terms).mean() if volume_terms else features.sum() * 0.0
#     valid_class_count = len(compact_terms)
#     unique_class_count = int(torch.unique(y).numel())

#     centers, _, _ = _class_centers(z, y, min_samples=min_class_samples, normalize_centers=False)
#     center = _pairwise_center_margin_loss(centers, center_margin)

#     sub_obj = _batch_subspace_overlap_loss(
#         z,
#         y,
#         rank=int(subspace_rank),
#         min_samples=int(subspace_min_samples),
#         max_overlap=float(max_subspace_overlap),
#         normalize=False,
#         return_parts=True,
#     )
#     subspace = sub_obj["total"]
#     subspace_pair_count = sub_obj["pair_count"]

#     band_obj = risk_aware_band_discrimination_loss(
#         band_summary,
#         y,
#         features=z,
#         min_samples=int(min_class_samples),
#         max_band_similarity=float(max_band_similarity),
#         risk_center_margin=float(center_margin),
#         return_parts=True,
#     )
#     band = band_obj["total"] if isinstance(band_obj, dict) else band_obj
#     band_pair_count = band_obj.get("pair_count", features.sum().detach() * 0.0) if isinstance(band_obj, dict) else features.sum().detach() * 0.0

#     one = torch.tensor(1.0, device=features.device, dtype=features.dtype)
#     zero = features.sum() * 0.0
#     if adaptive_component_weights:
#         compact_factor = one if valid_class_count > 0 else zero
#         volume_factor = one if valid_class_count > 0 else zero
#         center_factor = one if int(centers.size(0)) >= 2 else zero
#         subspace_factor = one if float(subspace_pair_count.detach().item()) > 0.0 else zero
#         band_factor = one if float(band_pair_count.detach().item()) > 0.0 else zero
#     else:
#         compact_factor = center_factor = subspace_factor = band_factor = volume_factor = one

#     pgr_unweighted = (
#         float(compact_weight) * compact_factor * compact
#         + float(center_weight) * center_factor * center
#         + float(subspace_weight) * subspace_factor * subspace
#         + float(band_weight) * band_factor * band
#         + float(volume_weight) * volume_factor * volume
#     )
#     total = float(weight) * pgr_unweighted

#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "pgr": pgr_unweighted.detach(),
#         "weighted_pgr": total.detach(),
#         "compact": compact.detach(),
#         "center": center.detach(),
#         "subspace": subspace.detach(),
#         "band": band.detach(),
#         "volume": volume.detach(),
#         "valid_class_count": torch.tensor(float(valid_class_count), device=features.device, dtype=features.dtype),
#         "unique_class_count": torch.tensor(float(unique_class_count), device=features.device, dtype=features.dtype),
#         "subspace_pair_count": subspace_pair_count.detach(),
#         "band_pair_count": band_pair_count.detach() if torch.is_tensor(band_pair_count) else torch.tensor(float(band_pair_count), device=features.device, dtype=features.dtype),
#         "compact_factor": compact_factor.detach(),
#         "center_factor": center_factor.detach(),
#         "subspace_factor": subspace_factor.detach(),
#         "band_factor": band_factor.detach(),
#         "volume_factor": volume_factor.detach(),
#         "subspace_mean_overlap": sub_obj.get("mean_overlap", zero).detach(),
#         "subspace_max_overlap": sub_obj.get("max_overlap", zero).detach(),
#         "band_mean_similarity": band_obj.get("mean_similarity", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "band_max_similarity": band_obj.get("max_similarity", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "band_guided_conflict_mean": band_obj.get("guided_conflict_mean", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "band_guided_conflict_max": band_obj.get("guided_conflict_max", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "class_variance_mean": torch.stack(class_vars).mean() if class_vars else zero.detach(),
#     }


# def base_prospective_geometry_reserve_loss(*args: Any, **kwargs: Any):
#     return prospective_geometry_reserve_loss(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # Physical spectral-shape reserve
# # -----------------------------------------------------------------------------


# def spectral_shape_discrimination_loss(
#     spectral_summary: Optional[torch.Tensor],
#     labels: torch.Tensor,
#     *,
#     features: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     require_physical_summary: bool = True,
#     min_samples: int = 3,
#     max_shape_similarity: float = 0.75,
#     risk_center_margin: float = 1.0,
#     risk_weight: float = 1.0,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Spectral-shape-guided feature-geometry reserve.

#     Raw wavelength spectra are fixed metadata, not trainable outputs.  High
#     spectral-shape similarity should therefore identify hard HSI class pairs and
#     push their learned feature centers apart.  It should not be optimized as a
#     standalone raw-spectrum dissimilarity objective.
#     """
#     ref = features if torch.is_tensor(features) else (spectral_summary if torch.is_tensor(spectral_summary) else labels)
#     if (
#         spectral_summary is None
#         or not torch.is_tensor(spectral_summary)
#         or spectral_summary.numel() == 0
#         or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
#     ):
#         z = safe_zero_like(ref)
#         out = {
#             "total": z,
#             "spectral_shape": z,
#             "pair_count": z,
#             "valid_class_count": z,
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     if s.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(s.shape)}")
#     y = _as_1d_long(labels, device=s.device)
#     if y.numel() != s.size(0):
#         raise ValueError(f"spectral_summary/label mismatch: spectra={s.size(0)}, labels={y.numel()}")

#     desc = _spectral_profile_descriptor(s)
#     centers, class_ids, _ = _class_centers(desc, y, min_samples=min_samples, normalize_centers=False)

#     if centers.size(0) < 2:
#         z = s.sum() * 0.0
#         out = {
#             "total": z,
#             "spectral_shape": z,
#             "pair_count": z,
#             "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     sim = _positive_cosine_matrix(centers, centers)
#     eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
#     pair_sim = sim[~eye]
#     hard_shape = F.relu(pair_sim - float(max_shape_similarity)) / max(1.0 - float(max_shape_similarity), 1e-6)
#     hard_shape = hard_shape.detach()

#     if features is not None and torch.is_tensor(features) and features.numel() > 0:
#         zf = F.normalize(features.to(device=s.device, dtype=s.dtype), dim=1, eps=1e-6)
#         f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
#         if f_centers.size(0) == centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
#             dist = torch.cdist(f_centers, f_centers, p=2)
#             feature_conflict = F.relu(float(risk_center_margin) - dist)[~eye] / max(float(risk_center_margin), 1e-6)
#             guided = hard_shape * feature_conflict
#             loss_vec = hard_shape * feature_conflict.pow(2)
#             loss = float(risk_weight) * (loss_vec.mean() if loss_vec.numel() > 0 else features.sum() * 0.0)
#         else:
#             guided = torch.zeros_like(pair_sim)
#             loss = features.sum() * 0.0
#     else:
#         guided = torch.zeros_like(pair_sim)
#         loss = s.sum() * 0.0

#     if not return_parts:
#         return loss
#     return {
#         "total": loss,
#         "spectral_shape": loss.detach(),
#         "pair_count": torch.tensor(float(pair_sim.numel()), device=s.device, dtype=s.dtype),
#         "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
#         "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
#         "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
#         "guided_conflict_mean": guided.mean().detach() if guided.numel() > 0 else s.sum().detach() * 0.0,
#         "guided_conflict_max": guided.max().detach() if guided.numel() > 0 else s.sum().detach() * 0.0,
#     }


# # -----------------------------------------------------------------------------
# # Low-rank GeometryBank energy used by classifier/replay/margins
# # -----------------------------------------------------------------------------

# def _canonicalize_variances(
#     *,
#     eigvals: Optional[torch.Tensor] = None,
#     res_vars: Optional[torch.Tensor] = None,
#     variances: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if variances is not None and torch.is_tensor(variances):
#         if variances.dim() != 2 or variances.size(1) < 2:
#             raise ValueError(f"variances must be [C,R+1], got {tuple(variances.shape)}")
#         return variances[:, :-1], variances[:, -1]
#     if eigvals is None or res_vars is None:
#         raise ValueError("Either variances or eigvals+res_vars must be provided.")
#     return eigvals, res_vars


# def _active_rank_mask(active_ranks: Optional[torch.Tensor], C: int, R: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
#     if active_ranks is None or not torch.is_tensor(active_ranks):
#         ar = torch.full((C,), int(R), device=device, dtype=torch.long)
#     else:
#         ar = active_ranks.to(device=device).long().flatten()
#         if ar.numel() != C:
#             raise ValueError(f"active_ranks must have C={C} entries, got {ar.numel()}")
#         ar = ar.clamp(min=0, max=R)
#     idx = torch.arange(R, device=device).view(1, R)
#     mask = (idx < ar.view(C, 1)).to(dtype=dtype)
#     return mask, ar



# def geometry_energy_matrix(
#     features: Optional[torch.Tensor] = None,
#     means: Optional[torch.Tensor] = None,
#     bases: Optional[torch.Tensor] = None,
#     variances: Optional[torch.Tensor] = None,
#     *,
#     bank: Optional[Mapping[str, torch.Tensor]] = None,
#     eigvals: Optional[torch.Tensor] = None,
#     res_vars: Optional[torch.Tensor] = None,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     reliability_energy_weight: float = 0.03,
#     reliability_min_clamp: float = 0.05,
#     residual_variance_scale: float = 0.75,
#     normalize_by_dim: bool = True,
#     invalid_class_energy: float = _INVALID_ENERGY,
#     use_logdet_energy: bool = True,
#     logdet_energy_weight: float = 0.05,
#     logdet_normalize_by_dim: bool = True,
#     center_logdet_energy: bool = True,
#     # accepted but intentionally inactive in PG-RGA main path
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_curve_means: Optional[torch.Tensor] = None,
#     spectral_curve_vars: Optional[torch.Tensor] = None,
#     spectral_curve_d1: Optional[torch.Tensor] = None,
#     spectral_curve_d2: Optional[torch.Tensor] = None,
#     spectral_shape_reliability: Optional[torch.Tensor] = None,
#     use_spectral_residual_energy: bool = False,
#     spectral_energy_weight: float = 0.0,
#     spectral_summary_is_physical: bool = False,
#     spectral_require_physical_summary: bool = True,
#     return_parts: bool = False,
#     **_: Any,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """Low-rank GeometryBank energy used by classifier/replay/margins.

#     Supports both explicit tensors and a bank mapping.  Spectral arguments are
#     accepted only for API compatibility; PG-RGA inference remains feature-geometry
#     energy, not spectral-branch energy.
#     """
#     del spectral_summary, spectral_curve_means, spectral_curve_vars, spectral_curve_d1, spectral_curve_d2, spectral_shape_reliability

#     if isinstance(means, Mapping) and bank is None and bases is None:
#         bank = means  # supports geometry_energy_matrix(features, bank=...) style through positional means
#         means = None
#     if bank is not None:
#         means = bank.get("means", means)
#         bases = bank.get("bases", bank.get("subspace_bases", bases))
#         variances = bank.get("variances", variances)
#         eigvals = bank.get("eigvals", bank.get("eigenvalues", eigvals))
#         res_vars = bank.get("res_vars", bank.get("residual_variances", res_vars))
#         active_ranks = bank.get("active_ranks", active_ranks)
#         reliability = bank.get("reliability", reliability)
#         sample_counts = bank.get("sample_counts", sample_counts)

#     if means is None or bases is None:
#         raise ValueError("geometry_energy_matrix requires means and bases, or a bank mapping containing them.")

#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         device = means.device if torch.is_tensor(means) else torch.device("cpu")
#         dtype = means.dtype if torch.is_tensor(means) else torch.float32
#         C = int(means.size(0)) if torch.is_tensor(means) and means.dim() >= 1 else 0
#         empty = torch.empty((0, C), device=device, dtype=dtype)
#         if return_parts:
#             return {"energy": empty, "feature_energy": empty, "parallel": empty, "orthogonal": empty,
#                     "parallel_energy": empty, "residual_energy": empty, "spectral_energy": empty}
#         return empty

#     if features.dim() != 2:
#         raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
#     if means.dim() != 2:
#         raise ValueError(f"means must be [C,D], got {tuple(means.shape)}")
#     if bases.dim() != 3:
#         raise ValueError(f"bases must be [C,D,R], got {tuple(bases.shape)}")
#     B, D = features.shape
#     C, Dm = means.shape
#     if Dm != D or bases.size(0) != C or bases.size(1) != D:
#         raise ValueError(f"feature/bank shape mismatch: features={tuple(features.shape)}, means={tuple(means.shape)}, bases={tuple(bases.shape)}")

#     device, dtype = features.device, features.dtype
#     means = means.to(device=device, dtype=dtype)
#     bases = bases.to(device=device, dtype=dtype)
#     eig, rv = _canonicalize_variances(eigvals=eigvals, res_vars=res_vars, variances=variances)
#     eig = eig.to(device=device, dtype=dtype)
#     rv = rv.to(device=device, dtype=dtype).flatten()

#     R = int(bases.size(2))
#     if eig.dim() != 2 or eig.size(0) != C or eig.size(1) != R:
#         raise ValueError(f"eigvals/variances rank mismatch: eig={tuple(eig.shape)}, C={C}, R={R}")
#     if rv.numel() != C:
#         raise ValueError(f"res_vars must have C={C} entries, got {rv.numel()}")

#     rank_mask, ar = _active_rank_mask(active_ranks, C, R, device, dtype)

#     delta = features.unsqueeze(1) - means.unsqueeze(0)
#     coeff = torch.einsum("bcd,cdr->bcr", delta, bases)
#     coeff_active = coeff * rank_mask.view(1, C, R)
#     recon = torch.einsum("bcr,cdr->bcd", coeff_active, bases)
#     residual = delta - recon

#     eig_safe = eig.clamp_min(float(variance_floor))
#     rv_safe = (rv * float(residual_variance_scale)).clamp_min(float(variance_floor))

#     parallel = ((coeff_active.pow(2) / eig_safe.view(1, C, R)) * rank_mask.view(1, C, R)).sum(dim=-1)
#     orthogonal = residual.pow(2).sum(dim=-1) / rv_safe.view(1, C)
#     energy = parallel + orthogonal
#     if bool(normalize_by_dim):
#         energy = energy / float(max(D, 1))

#     logdet_penalty = torch.zeros((C,), device=device, dtype=dtype)
#     if bool(use_logdet_energy) and float(logdet_energy_weight) > 0.0:
#         active_logdet = (eig_safe.log() * rank_mask).sum(dim=1)
#         residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=dtype)
#         logdet_penalty = active_logdet + residual_dims * rv_safe.log()
#         if bool(logdet_normalize_by_dim):
#             logdet_penalty = logdet_penalty / float(max(D, 1))
#         if bool(center_logdet_energy):
#             logdet_penalty = logdet_penalty - logdet_penalty.mean().detach()
#         energy = energy + float(logdet_energy_weight) * logdet_penalty.view(1, C)

#     reliability_penalty = torch.zeros((C,), device=device, dtype=dtype)
#     if reliability is not None and torch.is_tensor(reliability) and float(reliability_energy_weight) > 0.0:
#         rel = torch.nan_to_num(reliability.to(device=device, dtype=dtype).flatten(), nan=float(reliability_min_clamp), posinf=1.0, neginf=float(reliability_min_clamp))
#         if rel.numel() != C:
#             raise ValueError(f"reliability must have C={C} entries, got {rel.numel()}")
#         rel = rel.clamp(float(reliability_min_clamp), 1.0)
#         reliability_penalty = -rel.log()
#         reliability_penalty = reliability_penalty - reliability_penalty.mean().detach()
#         energy = energy + float(reliability_energy_weight) * reliability_penalty.view(1, C)

#     spectral_energy = torch.zeros_like(energy)
#     if bool(use_spectral_residual_energy) and float(spectral_energy_weight) > 0.0:
#         if bool(spectral_require_physical_summary) and not bool(spectral_summary_is_physical):
#             spectral_energy = torch.zeros_like(energy)
#         else:
#             spectral_energy = torch.zeros_like(energy)

#     energy = energy + spectral_energy

#     # Mask invalid/uninitialized classes so stale future rows can never win.
#     valid_class = torch.ones((C,), device=device, dtype=torch.bool)
#     if sample_counts is not None and torch.is_tensor(sample_counts):
#         sc = sample_counts.to(device=device).flatten()
#         if sc.numel() == C:
#             valid_class = sc > 0
#     # A valid row with active_rank == 0 can still score through pure residual energy.
#     # Do not mask tiny/low-rank classes just because their tangent rank is empty.
#     valid_class = valid_class & (ar >= 0)
#     if bool((~valid_class).any().item()):
#         energy = energy.masked_fill((~valid_class).view(1, C), float(invalid_class_energy))
#         parallel = parallel.masked_fill((~valid_class).view(1, C), float(invalid_class_energy))
#         orthogonal = orthogonal.masked_fill((~valid_class).view(1, C), float(invalid_class_energy))

#     energy = torch.nan_to_num(energy, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0)

#     if not return_parts:
#         return energy
#     return {
#         "energy": energy,
#         "feature_energy": energy,
#         "parallel": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "orthogonal": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "parallel_energy": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "residual_energy": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "logdet_penalty": logdet_penalty,
#         "reliability_penalty": reliability_penalty,
#         "spectral_energy": spectral_energy,
#         "active_ranks": ar,
#         "rank_mask": rank_mask,
#         "valid_class_mask": valid_class,
#     }



# # -----------------------------------------------------------------------------
# # Phase-consistent energy-margin helpers
# # -----------------------------------------------------------------------------

# def _energy_to_logits(
#     energy: torch.Tensor,
#     *,
#     logit_scale: float = 8.0,
#     center_per_sample: bool = True,
#     clip: float = 50.0,
# ) -> torch.Tensor:
#     """Convert lower-is-better energy to CE logits without changing class order."""
#     if energy is None or not torch.is_tensor(energy):
#         raise TypeError("energy must be a tensor.")
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     e = torch.nan_to_num(energy.float(), nan=float(_INVALID_ENERGY), posinf=float(_INVALID_ENERGY), neginf=0.0)
#     if bool(center_per_sample) and e.numel() > 0:
#         finite = torch.isfinite(e)
#         row_ref = torch.where(finite, e, torch.zeros_like(e)).mean(dim=1, keepdim=True).detach()
#         e = e - row_ref
#     logits = -float(logit_scale) * e
#     if float(clip) > 0.0:
#         logits = logits.clamp(-float(clip), float(clip))
#     return torch.nan_to_num(logits, nan=-float(clip), posinf=float(clip), neginf=-float(clip))


# def _ce_from_logits(
#     logits: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     label_smoothing: float = 0.0,
#     clip: float = 50.0,
# ) -> torch.Tensor:
#     if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
#         return safe_zero_like(labels if torch.is_tensor(labels) else None)
#     if logits.dim() != 2:
#         raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
#     y = labels.to(device=logits.device).long().flatten()
#     if y.numel() != logits.size(0):
#         raise RuntimeError(f"CE labels/logits mismatch: labels={y.numel()}, logits={logits.size(0)}")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= logits.size(1)):
#         raise RuntimeError(
#             f"CE labels outside logit range: [{int(y.min().item())},{int(y.max().item())}] vs width={logits.size(1)}"
#         )
#     return F.cross_entropy(logits.clamp(-float(clip), float(clip)), y, label_smoothing=float(label_smoothing))


# def _energy_margin_parts(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     margin: float = 0.25,
# ) -> Dict[str, torch.Tensor]:
#     ref = energy if torch.is_tensor(energy) else labels
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         z = safe_zero_like(ref)
#         return {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach(), "true_energy": z.detach(), "nearest_wrong_energy": z.detach()}
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
#         raise RuntimeError("labels outside local energy range")
#     e = torch.nan_to_num(energy.float(), nan=float(_INVALID_ENERGY), posinf=float(_INVALID_ENERGY), neginf=0.0)
#     true_e = e.gather(1, y.view(-1, 1)).squeeze(1)
#     true_mask = torch.zeros_like(e, dtype=torch.bool)
#     true_mask.scatter_(1, y.view(-1, 1), True)
#     nearest_wrong = e.masked_fill(true_mask, float("inf")).min(dim=1).values
#     gap = nearest_wrong - true_e
#     loss_vec = F.relu(float(margin) - gap)
#     finite = torch.isfinite(loss_vec)
#     loss = loss_vec[finite].mean() if bool(finite.any().item()) else e.sum() * 0.0
#     return {
#         "total": loss,
#         "violation_rate": (gap < float(margin)).float().mean().detach() if gap.numel() else e.sum().detach() * 0.0,
#         "mean_gap": gap[torch.isfinite(gap)].mean().detach() if bool(torch.isfinite(gap).any().item()) else e.sum().detach() * 0.0,
#         "true_energy": true_e.detach().mean() if true_e.numel() else e.sum().detach() * 0.0,
#         "nearest_wrong_energy": nearest_wrong[torch.isfinite(nearest_wrong)].detach().mean() if bool(torch.isfinite(nearest_wrong).any().item()) else e.sum().detach() * 0.0,
#     }


# def _old_new_invasion_parts(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     old_class_count: int,
#     margin: float = 0.25,
# ) -> Dict[str, torch.Tensor]:
#     ref = energy if torch.is_tensor(energy) else labels
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         z = safe_zero_like(ref)
#         return {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach()}
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     C = int(energy.size(1))
#     old = int(max(0, min(int(old_class_count), C)))
#     if old <= 0 or old >= C:
#         z = energy.sum() * 0.0
#         return {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach()}
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= C):
#         raise RuntimeError("labels outside local energy range")
#     e = torch.nan_to_num(energy.float(), nan=float(_INVALID_ENERGY), posinf=float(_INVALID_ENERGY), neginf=0.0)
#     true_e = e.gather(1, y.view(-1, 1)).squeeze(1)
#     old_min = e[:, :old].min(dim=1).values
#     new_min = e[:, old:].min(dim=1).values
#     is_old = y < old
#     opposite = torch.where(is_old, new_min, old_min)
#     gap = opposite - true_e
#     loss_vec = F.relu(float(margin) - gap)
#     finite = torch.isfinite(loss_vec)
#     loss = loss_vec[finite].mean() if bool(finite.any().item()) else e.sum() * 0.0
#     return {
#         "total": loss,
#         "violation_rate": (gap < float(margin)).float().mean().detach() if gap.numel() else e.sum().detach() * 0.0,
#         "mean_gap": gap[torch.isfinite(gap)].mean().detach() if bool(torch.isfinite(gap).any().item()) else e.sum().detach() * 0.0,
#     }


# def base_local_geometry_energy_margin_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     weight: float = 0.15,
#     margin: float = 0.25,
#     min_samples: int = 3,
#     variance_floor: float = 5e-4,
#     normalize_features: bool = True,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Batch-local approximation of the GeometryBank energy margin for phase 0."""
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         out = {"total": z, "raw": z, "violation_rate": z.detach(), "mean_gap": z.detach(), "valid_anchor_count": z.detach(), "valid_class_count": z.detach()}
#         return out if return_parts else z
#     if features.dim() != 2:
#         raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
#     y = _as_1d_long(labels, device=features.device)
#     if y.numel() != features.size(0):
#         raise ValueError(f"labels/features mismatch: labels={y.numel()}, features={features.size(0)}")
#     z = F.normalize(features, dim=1, eps=1e-6) if normalize_features else features
#     _, D = z.shape
#     centers, class_ids, _ = _class_centers(z, y, min_samples=int(min_samples), normalize_centers=False)
#     if centers.size(0) < 2:
#         zero = features.sum() * 0.0
#         out = {
#             "total": zero, "raw": zero.detach(), "violation_rate": zero.detach(), "mean_gap": zero.detach(),
#             "valid_anchor_count": zero.detach(),
#             "valid_class_count": torch.tensor(float(centers.size(0)), device=features.device, dtype=features.dtype),
#         }
#         return out if return_parts else zero
#     local = torch.full_like(y, -1)
#     for j, cls in enumerate(class_ids.detach().cpu().tolist()):
#         local[y == int(cls)] = int(j)
#     valid_anchor = local >= 0
#     if not bool(valid_anchor.any().item()):
#         zero = features.sum() * 0.0
#         out = {"total": zero, "raw": zero.detach(), "violation_rate": zero.detach(), "mean_gap": zero.detach(), "valid_anchor_count": zero.detach(), "valid_class_count": zero.detach()}
#         return out if return_parts else zero
#     class_vars = []
#     for j, cls in enumerate(class_ids.detach().cpu().tolist()):
#         m = y == int(cls)
#         zj = z[m]
#         cj = centers[j:j + 1]
#         var_j = (zj - cj).pow(2).sum(dim=1).mean() / float(max(D, 1))
#         class_vars.append(var_j.clamp_min(float(variance_floor)))
#     class_vars = torch.stack(class_vars).to(device=features.device, dtype=features.dtype)
#     zv = z[valid_anchor]
#     yv = local[valid_anchor]
#     dist2 = torch.cdist(zv, centers, p=2).pow(2) / float(max(D, 1))
#     energy = dist2 / class_vars.view(1, -1).clamp_min(float(variance_floor))
#     parts = _energy_margin_parts(energy, yv, margin=float(margin))
#     raw = parts["total"]
#     total = float(weight) * raw
#     out = {
#         "total": total,
#         "raw": raw.detach(),
#         "violation_rate": parts["violation_rate"].detach(),
#         "mean_gap": parts["mean_gap"].detach(),
#         "valid_anchor_count": torch.tensor(float(valid_anchor.sum().item()), device=features.device, dtype=features.dtype),
#         "valid_class_count": torch.tensor(float(centers.size(0)), device=features.device, dtype=features.dtype),
#     }
#     return out if return_parts else total


# def incremental_geometry_training_loss(
#     *,
#     labels: torch.Tensor,
#     logits: Optional[torch.Tensor] = None,
#     energy: Optional[torch.Tensor] = None,
#     features: Optional[torch.Tensor] = None,
#     bank: Optional[Mapping[str, torch.Tensor]] = None,
#     old_class_count: int = 0,
#     ce_weight: float = 1.0,
#     joint_old_new_ce_weight: Optional[float] = None,
#     geometry_energy_margin_weight: float = 0.30,
#     geometry_energy_margin: float = 0.30,
#     old_new_invasion_weight: float = 0.50,
#     old_new_geometry_margin: float = 0.35,
#     label_smoothing: float = 0.0,
#     logit_scale: float = 8.0,
#     ce_logit_clip: float = 50.0,
#     variance_floor: float = 5e-4,
#     residual_variance_scale: float = 0.75,
#     reliability_energy_weight: float = 0.03,
#     normalize_by_dim: bool = True,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Phase>=1 objective using the same GeometryBank energy as the classifier."""
#     ref = energy if torch.is_tensor(energy) else (logits if torch.is_tensor(logits) else features)
#     z = safe_zero_like(ref)
#     if energy is None and torch.is_tensor(features) and bank is not None:
#         energy = geometry_energy_matrix(
#             features,
#             bank=bank,
#             variance_floor=float(variance_floor),
#             residual_variance_scale=float(residual_variance_scale),
#             reliability_energy_weight=float(reliability_energy_weight),
#             normalize_by_dim=bool(normalize_by_dim),
#             return_parts=False,
#             **kwargs,
#         )
#     if logits is None and torch.is_tensor(energy):
#         logits = _energy_to_logits(energy, logit_scale=float(logit_scale), clip=float(ce_logit_clip))
#     ce_w = float(joint_old_new_ce_weight) if joint_old_new_ce_weight is not None else float(ce_weight)
#     ce = _ce_from_logits(logits, labels, label_smoothing=float(label_smoothing), clip=float(ce_logit_clip)) if torch.is_tensor(logits) else z
#     if not torch.is_tensor(energy) and torch.is_tensor(logits):
#         energy = -logits.float() / max(float(logit_scale), 1e-6)
#     margin_parts = _energy_margin_parts(energy, labels, margin=float(geometry_energy_margin)) if torch.is_tensor(energy) else {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach()}
#     invasion_parts = _old_new_invasion_parts(energy, labels, old_class_count=int(old_class_count), margin=float(old_new_geometry_margin)) if torch.is_tensor(energy) else {"total": z, "violation_rate": z.detach(), "mean_gap": z.detach()}
#     energy_margin_total = _scalar(margin_parts.get("total", z), ref)
#     invasion_total = _scalar(invasion_parts.get("total", z), ref)
#     total = ce_w * ce + float(geometry_energy_margin_weight) * energy_margin_total + float(old_new_invasion_weight) * invasion_total
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "ce": ce.detach(),
#         "incremental_ce": ce.detach(),
#         "geometry_energy_margin": energy_margin_total.detach(),
#         "old_new_invasion": invasion_total.detach(),
#         "energy_margin_violation": _scalar(margin_parts.get("violation_rate", z), ref).detach(),
#         "energy_margin_gap": _scalar(margin_parts.get("mean_gap", z), ref).detach(),
#         "old_new_invasion_violation": _scalar(invasion_parts.get("violation_rate", z), ref).detach(),
#         "old_new_invasion_gap": _scalar(invasion_parts.get("mean_gap", z), ref).detach(),
#         "phase_is_base": torch.tensor(0.0, device=total.device, dtype=total.dtype),
#         "phase_is_incremental": torch.tensor(1.0, device=total.device, dtype=total.dtype),
#         "rank": energy_margin_total.detach(),
#         "admission": invasion_total.detach(),
#         "subspace": z.detach(),
#         "volume": z.detach(),
#         "trust": z.detach(),
#         "violation_rate": _scalar(margin_parts.get("violation_rate", z), ref).detach(),
#     }


# # -----------------------------------------------------------------------------
# # Unified base objective
# # -----------------------------------------------------------------------------

# def base_geometry_preparation_loss(
#     *,
#     logits: Optional[torch.Tensor] = None,
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     key_features: Optional[torch.Tensor] = None,
#     band_summary: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     ce_weight: float = 0.0,
#     base_geometry_weight: float = 1.0,
#     label_smoothing: float = 0.0,
#     # GICS
#     gics_weight: float = 0.20,
#     gics_temperature: float = 0.07,
#     # PGR
#     pgr_weight: float = 0.10,
#     pgr_compact_weight: float = 0.15,
#     pgr_center_weight: float = 0.20,
#     pgr_subspace_weight: float = 0.10,
#     pgr_band_weight: float = 0.05,
#     pgr_volume_weight: float = 0.05,
#     pgr_center_margin: float = 1.05,
#     pgr_max_band_similarity: float = 0.75,
#     pgr_max_class_variance: float = 0.75,
#     pgr_min_class_variance: float = 0.015,
#     pgr_max_subspace_overlap: float = 0.50,
#     subspace_rank: int = 3,
#     min_class_samples: int = 3,
#     subspace_min_samples: int = 6,
#     # spectral shape
#     spectral_shape_weight: float = 0.05,
#     max_spectral_shape_similarity: float = 0.75,
#     spectral_shape_risk_weight: float = 1.0,
#     require_physical_summary: bool = True,
#     # base energy handoff
#     base_energy_margin_weight: float = 0.15,
#     base_energy_margin: float = 0.25,
#     base_energy_variance_floor: float = 5e-4,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Single active base-phase regularizer.

#     BasePhaseTrainer computes balanced CE separately, so ce_weight defaults to
#     0.0.  The returned 'total' remains differentiable and must be used for SRPGR.
#     """
#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         out = {
#             "total": z, "ce": z, "base_geometry": z,
#             "base_gics": z, "base_gics_anchors": z, "base_gics_pos": z,
#             "base_pgr": z, "base_compact": z, "base_center": z,
#             "base_subspace": z, "base_band": z, "base_volume": z,
#             "base_spectral_shape": z, "base_spectral_shape_raw": z,
#             "base_spectral_shape_mean_similarity": z, "base_spectral_shape_pair_count": z,
#             "base_spectral_shape_active": z,
#             "base_energy_margin": z, "base_energy_margin_weighted": z,
#             "base_energy_margin_violation": z, "base_energy_margin_gap": z,
#         }
#         return out if return_parts else z

#     if features.dim() != 2:
#         raise ValueError(f"base features must be [B,D], got {tuple(features.shape)}")
#     _require_finite_tensor(features, "base.features")

#     labels = _as_1d_long(labels, device=features.device)
#     if labels.numel() != features.size(0):
#         raise ValueError(f"base labels/features mismatch: labels={labels.numel()}, features={features.size(0)}")

#     ce = safe_zero_like(features)
#     if ce_weight > 0.0:
#         if logits is None or not torch.is_tensor(logits):
#             raise ValueError("logits are required when ce_weight > 0.")
#         if logits.dim() != 2 or logits.size(0) != labels.numel():
#             raise ValueError(f"logits must be [B,C] aligned with labels, got {tuple(logits.shape)}")
#         ce = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))

#     # If physical spectra are present, they are the correct band reserve input.
#     # This avoids PCA-30 band profiles competing with raw-200 spectral descriptors.
#     band_for_pgr = band_summary
#     if spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0 and bool(spectral_summary_is_physical):
#         band_for_pgr = spectral_summary

#     gics = base_geometry_involved_contrastive_loss(
#         features,
#         labels,
#         key_features=key_features,
#         weight=float(gics_weight),
#         temperature=float(gics_temperature),
#         return_parts=True,
#     )

#     pgr = prospective_geometry_reserve_loss(
#         features,
#         labels,
#         band_summary=band_for_pgr,
#         weight=float(pgr_weight),
#         compact_weight=float(pgr_compact_weight),
#         center_weight=float(pgr_center_weight),
#         subspace_weight=float(pgr_subspace_weight),
#         band_weight=float(pgr_band_weight),
#         volume_weight=float(pgr_volume_weight),
#         center_margin=float(pgr_center_margin),
#         min_class_samples=int(min_class_samples),
#         subspace_min_samples=int(subspace_min_samples),
#         subspace_rank=int(subspace_rank),
#         max_band_similarity=float(pgr_max_band_similarity),
#         max_class_variance=float(pgr_max_class_variance),
#         min_class_variance=float(pgr_min_class_variance),
#         max_subspace_overlap=float(kwargs.get("subspace_overlap_max", pgr_max_subspace_overlap)),
#         return_parts=True,
#     )

#     shape_raw = spectral_shape_discrimination_loss(
#         spectral_summary,
#         labels,
#         features=features,
#         spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         require_physical_summary=bool(require_physical_summary),
#         min_samples=int(min_class_samples),
#         max_shape_similarity=float(max_spectral_shape_similarity),
#         risk_center_margin=float(pgr_center_margin),
#         risk_weight=float(spectral_shape_risk_weight),
#         return_parts=True,
#     )
#     shape_total = float(spectral_shape_weight) * _scalar(shape_raw.get("total", safe_zero_like(features)), features)

#     energy_margin = base_local_geometry_energy_margin_loss(
#         features,
#         labels,
#         weight=float(base_energy_margin_weight),
#         margin=float(base_energy_margin),
#         min_samples=int(min_class_samples),
#         variance_floor=float(kwargs.get("geom_var_floor", kwargs.get("variance_floor", base_energy_variance_floor))),
#         normalize_features=True,
#         return_parts=True,
#     )

#     gics_total = _scalar(gics.get("total", safe_zero_like(features)), features)
#     pgr_total = _scalar(pgr.get("total", safe_zero_like(features)), features)
#     energy_margin_total = _scalar(energy_margin.get("total", safe_zero_like(features)), features)
#     base_geometry = float(base_geometry_weight) * (gics_total + pgr_total + shape_total + energy_margin_total)
#     total = float(ce_weight) * ce + base_geometry

#     if not return_parts:
#         return total

#     spectral_active = bool(spectral_summary_is_physical) and spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0

#     return {
#         "total": total,
#         "ce": ce.detach(),
#         # Keep differentiable value for code that still uses this key; logs can detach later.
#         "base_geometry": base_geometry,

#         "base_gics": _scalar(gics.get("gics", safe_zero_like(features)), features).detach(),
#         "base_gics_weighted": gics_total.detach(),
#         "base_gics_anchors": _scalar(gics.get("valid_anchors", safe_zero_like(features)), features).detach(),
#         "base_gics_pos": _scalar(gics.get("mean_positive_count", safe_zero_like(features)), features).detach(),

#         "base_pgr": _scalar(pgr.get("pgr", safe_zero_like(features)), features).detach(),
#         "base_pgr_weighted": pgr_total.detach(),
#         "base_compact": _scalar(pgr.get("compact", safe_zero_like(features)), features).detach(),
#         "base_center": _scalar(pgr.get("center", safe_zero_like(features)), features).detach(),
#         "base_subspace": _scalar(pgr.get("subspace", safe_zero_like(features)), features).detach(),
#         "base_band": _scalar(pgr.get("band", safe_zero_like(features)), features).detach(),
#         "base_volume": _scalar(pgr.get("volume", safe_zero_like(features)), features).detach(),

#         "base_spectral_shape": shape_total.detach(),
#         "base_spectral_shape_raw": _scalar(shape_raw.get("total", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_mean_similarity": _scalar(shape_raw.get("mean_similarity", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_max_similarity": _scalar(shape_raw.get("max_similarity", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_pair_count": _scalar(shape_raw.get("pair_count", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_active": torch.tensor(float(spectral_active), device=features.device, dtype=features.dtype),

#         "base_energy_margin": _scalar(energy_margin.get("raw", safe_zero_like(features)), features).detach(),
#         "base_energy_margin_weighted": energy_margin_total.detach(),
#         "base_energy_margin_violation": _scalar(energy_margin.get("violation_rate", safe_zero_like(features)), features).detach(),
#         "base_energy_margin_gap": _scalar(energy_margin.get("mean_gap", safe_zero_like(features)), features).detach(),

#         "base_pgr_valid_class_count": _scalar(pgr.get("valid_class_count", safe_zero_like(features)), features).detach(),
#         "base_pgr_subspace_pair_count": _scalar(pgr.get("subspace_pair_count", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_pair_count": _scalar(pgr.get("band_pair_count", safe_zero_like(features)), features).detach(),
#         "base_pgr_volume_factor": _scalar(pgr.get("volume_factor", safe_zero_like(features)), features).detach(),
#         "base_pgr_subspace_max_overlap": _scalar(pgr.get("subspace_max_overlap", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_max_similarity": _scalar(pgr.get("band_max_similarity", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_guided_conflict_mean": _scalar(pgr.get("band_guided_conflict_mean", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_guided_conflict_max": _scalar(pgr.get("band_guided_conflict_max", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_guided_conflict_mean": _scalar(shape_raw.get("guided_conflict_mean", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_guided_conflict_max": _scalar(shape_raw.get("guided_conflict_max", safe_zero_like(features)), features).detach(),
#     }


# def unified_spectral_geometry_loss(
#     *,
#     phase: str,
#     labels: torch.Tensor,
#     logits: Optional[torch.Tensor] = None,
#     energy: Optional[torch.Tensor] = None,
#     features: Optional[torch.Tensor] = None,
#     key_features: Optional[torch.Tensor] = None,
#     band_summary: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     bank: Optional[Mapping[str, torch.Tensor]] = None,
#     old_class_count: int = 0,
#     ce_weight: Optional[float] = None,
#     joint_old_new_ce_weight: Optional[float] = None,
#     geometry_energy_margin_weight: float = 0.30,
#     geometry_energy_margin: float = 0.30,
#     old_new_invasion_weight: float = 0.50,
#     old_new_geometry_margin: float = 0.35,
#     logit_scale: float = 8.0,
#     label_smoothing: float = 0.0,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Public phase-consistent loss entry for PG-RGA / NECIL-HSI."""
#     p = str(phase).strip().lower()
#     if p in {"base", "phase0", "phase_0", "0"}:
#         return base_geometry_preparation_loss(
#             logits=logits,
#             features=features,
#             labels=labels,
#             key_features=key_features,
#             band_summary=band_summary,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#             ce_weight=float(0.0 if ce_weight is None else ce_weight),
#             label_smoothing=float(label_smoothing),
#             return_parts=return_parts,
#             **kwargs,
#         )
#     if p in {"incremental", "inc", "phase_inc", "phase1", "phase_1", "1", "2", "3", "4", "5"} or p.startswith("phase"):
#         return incremental_geometry_training_loss(
#             labels=labels,
#             logits=logits,
#             energy=energy,
#             features=features,
#             bank=bank,
#             old_class_count=int(old_class_count),
#             ce_weight=float(1.0 if ce_weight is None else ce_weight),
#             joint_old_new_ce_weight=joint_old_new_ce_weight,
#             geometry_energy_margin_weight=float(geometry_energy_margin_weight),
#             geometry_energy_margin=float(geometry_energy_margin),
#             old_new_invasion_weight=float(old_new_invasion_weight),
#             old_new_geometry_margin=float(old_new_geometry_margin),
#             logit_scale=float(logit_scale),
#             label_smoothing=float(label_smoothing),
#             return_parts=return_parts,
#             **kwargs,
#         )
#     raise RuntimeError(f"Unsupported phase={phase!r}. Use 'base' or 'incremental'.")


# class UnifiedSpectralGeometryLoss:
#     """Thin callable wrapper for compatibility with older code."""

#     def __init__(self, **defaults: Any) -> None:
#         self.defaults = dict(defaults)

#     def __call__(self, **kwargs: Any):
#         merged = dict(self.defaults)
#         merged.update(kwargs)
#         return unified_spectral_geometry_loss(**merged)


# # -----------------------------------------------------------------------------
# # Incremental margin losses used by PG-RGA
# # -----------------------------------------------------------------------------

# def geometry_energy_margin_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         return safe_zero_like(labels if torch.is_tensor(labels) else None)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
#         raise RuntimeError("labels outside local energy range")
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     true_mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
#     nearest_wrong = energy.masked_fill(true_mask, float("inf")).min(dim=1).values
#     loss = F.relu(true_e + float(margin) - nearest_wrong)
#     finite = torch.isfinite(loss)
#     return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0


# def old_new_invasion_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     old_class_count: int,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         return safe_zero_like(labels if torch.is_tensor(labels) else None)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     C = int(energy.size(1))
#     old = int(max(0, min(int(old_class_count), C)))
#     if old <= 0 or old >= C:
#         return energy.sum() * 0.0
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= C):
#         raise RuntimeError("labels outside local energy range")
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     old_min = energy[:, :old].min(dim=1).values
#     new_min = energy[:, old:].min(dim=1).values
#     is_old = y < old
#     opposite = torch.where(is_old, new_min, old_min)
#     loss = F.relu(true_e + float(margin) - opposite)
#     finite = torch.isfinite(loss)
#     return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0


# # -----------------------------------------------------------------------------
# # Base diagnostics
# # -----------------------------------------------------------------------------

# @torch.no_grad()
# def base_center_overlap_diagnostics(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     normalize: bool = True,
#     min_samples: int = 2,
# ) -> Dict[str, torch.Tensor]:
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         return {"compact": z, "mean_center_margin": z, "min_center_margin": z, "num_classes": z}

#     z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
#     y = _as_1d_long(labels, device=z.device)
#     centers, _, _ = _class_centers(z, y, min_samples=min_samples)

#     compact_terms = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         if int(m.sum().item()) >= int(min_samples):
#             xc = z[m]
#             compact_terms.append((xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean())
#     compact = torch.stack(compact_terms).mean() if compact_terms else z.sum() * 0.0

#     if centers.size(0) < 2:
#         mean_margin = z.sum() * 0.0
#         min_margin = z.sum() * 0.0
#     else:
#         dist = torch.cdist(centers, centers, p=2)
#         eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
#         pair = dist[~eye]
#         mean_margin = pair.mean()
#         min_margin = pair.min()

#     return {
#         "compact": compact.detach(),
#         "mean_center_margin": mean_margin.detach(),
#         "min_center_margin": min_margin.detach(),
#         "num_classes": torch.tensor(float(centers.size(0)), device=z.device, dtype=z.dtype),
#     }


# @torch.no_grad()
# def base_gics_diagnostics(features: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#     out = base_geometry_involved_contrastive_loss(features, labels, weight=1.0, return_parts=True, **kwargs)
#     return {
#         "gics": out["gics"].detach(),
#         "valid_anchors": out["valid_anchors"].detach(),
#         "positive_pairs": out["mean_positive_count"].detach(),
#     }


# def base_supcon_diagnostics(*args: Any, **kwargs: Any):
#     return base_gics_diagnostics(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # Disabled legacy boundary helper
# # -----------------------------------------------------------------------------

# def sample_boundary_geometry_features(*args: Any, **kwargs: Any):
#     raise RuntimeError(
#         "sample_boundary_geometry_features is not part of  main path. "
#         "Use GeometryBank synthetic replay, not boundary replay."
#     )


















# from __future__ import annotations

# from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

# import torch
# import torch.nn.functional as F


# _EPS = 1e-12
# _INVALID_ENERGY = 1e6


# # -----------------------------------------------------------------------------
# # Basic utilities
# # -----------------------------------------------------------------------------

# def safe_zero_like(
#     ref: Optional[torch.Tensor] = None,
#     *,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> torch.Tensor:
#     if torch.is_tensor(ref):
#         return ref.sum() * 0.0
#     return torch.tensor(
#         0.0,
#         device=device if device is not None else torch.device("cpu"),
#         dtype=dtype if dtype is not None else torch.float32,
#     )


# def _require_finite_tensor(x: torch.Tensor, name: str) -> None:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     if x.numel() == 0:
#         return
#     if not torch.isfinite(x).all():
#         bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
#         raise RuntimeError(f"{name}: tensor contains {bad} NaN/Inf values.")


# def _as_1d_long(labels: torch.Tensor, *, device: torch.device, name: str = "labels") -> torch.Tensor:
#     if labels is None or not torch.is_tensor(labels):
#         raise TypeError(f"{name} must be a tensor.")
#     return labels.to(device=device).long().flatten()


# def _scalar(value: Any, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#     if torch.is_tensor(value):
#         if value.numel() == 1:
#             return value.reshape(())
#         return value.float().mean()
#     if isinstance(value, (int, float)):
#         if torch.is_tensor(ref):
#             return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
#         return torch.tensor(float(value), dtype=torch.float32)
#     return safe_zero_like(ref)


# def _class_centers(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     min_samples: int = 2,
#     normalize_centers: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         device = features.device if torch.is_tensor(features) else torch.device("cpu")
#         dtype = features.dtype if torch.is_tensor(features) else torch.float32
#         return (
#             torch.empty(0, 0, device=device, dtype=dtype),
#             torch.empty(0, device=device, dtype=torch.long),
#             torch.empty(0, device=device, dtype=dtype),
#         )

#     if features.dim() != 2:
#         raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")

#     y = _as_1d_long(labels, device=features.device)
#     if y.numel() != features.size(0):
#         raise ValueError(f"labels/features mismatch: labels={y.numel()}, features={features.size(0)}")

#     centers, class_ids, counts = [], [], []
#     for cls in torch.unique(y, sorted=True):
#         mask = y == cls
#         n = int(mask.sum().item())
#         if n < int(min_samples):
#             continue
#         c = features[mask].mean(dim=0)
#         if normalize_centers:
#             c = F.normalize(c, dim=0, eps=1e-6)
#         centers.append(c)
#         class_ids.append(cls)
#         counts.append(torch.tensor(float(n), device=features.device, dtype=features.dtype))

#     if not centers:
#         return (
#             torch.empty(0, features.size(1), device=features.device, dtype=features.dtype),
#             torch.empty(0, device=features.device, dtype=torch.long),
#             torch.empty(0, device=features.device, dtype=features.dtype),
#         )

#     return torch.stack(centers, dim=0), torch.stack(class_ids).long(), torch.stack(counts)


# def _pairwise_center_margin_loss(centers: torch.Tensor, margin: float) -> torch.Tensor:
#     if centers is None or not torch.is_tensor(centers) or centers.numel() == 0 or centers.size(0) < 2:
#         return safe_zero_like(centers)
#     dist = torch.cdist(centers, centers, p=2)
#     eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
#     pair = dist[~eye]
#     if pair.numel() == 0:
#         return centers.sum() * 0.0
#     return F.relu(float(margin) - pair).pow(2).mean()


# def _pad_to_width(x: torch.Tensor, width: int) -> torch.Tensor:
#     if x.size(1) == width:
#         return x
#     if x.size(1) > width:
#         return x[:, :width]
#     return F.pad(x, (0, int(width) - int(x.size(1))))


# # -----------------------------------------------------------------------------
# # Spectral / band profiles
# # -----------------------------------------------------------------------------

# def _spectral_derivatives(spectral_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     if spectral_summary.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
#     if spectral_summary.size(1) < 2:
#         z = spectral_summary.new_zeros((spectral_summary.size(0), 1))
#         return z, z
#     d1 = spectral_summary[:, 1:] - spectral_summary[:, :-1]
#     if d1.size(1) < 2:
#         d2 = d1.new_zeros((d1.size(0), 1))
#     else:
#         d2 = d1[:, 1:] - d1[:, :-1]
#     return d1, d2


# def _spectral_profile_descriptor(spectral_summary: torch.Tensor) -> torch.Tensor:
#     """Derivative-aware descriptor for physical spectral-shape comparison.

#     Raw HSI spectra may be normalized and may contain signed values.  Direct
#     softmax over signed spectra destroys absorption/reflectance shape.  This
#     descriptor preserves curve shape by standardizing each spectrum and appending
#     first/second derivative information.
#     """
#     if spectral_summary.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     s = s - s.mean(dim=1, keepdim=True)
#     s = s / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
#     d1, d2 = _spectral_derivatives(s)
#     desc = torch.cat([F.normalize(s, dim=1, eps=1e-6), F.normalize(d1, dim=1, eps=1e-6), F.normalize(d2, dim=1, eps=1e-6)], dim=1)
#     return torch.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


# def _band_importance_profile(band_summary: torch.Tensor) -> torch.Tensor:
#     """Convert raw spectra or band summaries to a non-negative band profile.

#     This is the correct base regularizer target for HSI: it compares where the
#     spectrum changes/absorbs, not a hidden classifier branch.  It works for raw
#     physical spectra and is still safe for reduced non-physical summaries.
#     """
#     if band_summary is None or not torch.is_tensor(band_summary):
#         raise TypeError("band_summary must be a tensor.")
#     if band_summary.dim() != 2:
#         raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")

#     b = torch.nan_to_num(band_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     B, S = b.shape
#     if S <= 0:
#         return b

#     # Per-sample standardization preserves spectral shape under dataset scaling.
#     z = b - b.mean(dim=1, keepdim=True)
#     z = z / z.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
#     d1, d2 = _spectral_derivatives(z)
#     d1e = _pad_to_width(d1.abs(), S)
#     d2e = _pad_to_width(d2.abs(), S)
#     profile = z.abs() + 0.50 * d1e + 0.25 * d2e
#     profile = profile.clamp_min(0.0)
#     uniform = torch.full_like(profile, 1.0 / float(max(S, 1)))
#     denom = profile.sum(dim=1, keepdim=True)
#     profile = torch.where(denom > 1e-8, profile / denom.clamp_min(1e-8), uniform)
#     return torch.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)


# # Backward-compatible name used by old code.
# def _normalize_band_summary(band_summary: torch.Tensor) -> torch.Tensor:
#     return _band_importance_profile(band_summary)


# def _positive_cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     a = F.normalize(a, dim=1, eps=1e-6)
#     b = F.normalize(b, dim=1, eps=1e-6)
#     return (a @ b.t()).clamp(0.0, 1.0)


# # -----------------------------------------------------------------------------
# # GICS: Geometry-Involved Class Separation
# # -----------------------------------------------------------------------------

# def base_geometry_involved_contrastive_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     key_features: Optional[torch.Tensor] = None,
#     weight: float = 0.20,
#     temperature: float = 0.07,
#     same_class_positive: bool = True,
#     class_balanced: bool = True,
#     detach_key: bool = True,
#     normalize: bool = True,
#     return_parts: bool = True,
#     **_: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Base-phase GICS on canonical projected z-space."""
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         out = {"total": z, "gics": z, "weighted_gics": z, "valid_anchors": z, "num_anchors": z, "mean_positive_count": z}
#         return out if return_parts else z

#     if features.dim() != 2:
#         raise ValueError(f"GICS expects projected features [B,D], got {tuple(features.shape)}")
#     _require_finite_tensor(features, "gics.features")

#     zq = features
#     explicit_key = key_features is not None and torch.is_tensor(key_features) and key_features.numel() > 0
#     zk = key_features if explicit_key else features
#     if zk.dim() != 2:
#         raise ValueError(f"key_features must be [B,D], got {tuple(zk.shape)}")
#     if zk.size(0) != zq.size(0) or zk.size(1) != zq.size(1):
#         raise ValueError(f"key_features shape mismatch: query={tuple(zq.shape)}, key={tuple(zk.shape)}")
#     if detach_key:
#         zk = zk.detach()

#     y = _as_1d_long(labels, device=zq.device)
#     if y.numel() != zq.size(0):
#         raise ValueError(f"GICS labels/features mismatch: labels={y.numel()}, features={zq.size(0)}")

#     q = F.normalize(zq, dim=1, eps=1e-6) if normalize else zq
#     k = F.normalize(zk.to(device=zq.device, dtype=zq.dtype), dim=1, eps=1e-6) if normalize else zk.to(device=zq.device, dtype=zq.dtype)

#     logits = q @ k.t()
#     logits = logits / max(float(temperature), 1e-6)
#     logits = logits - logits.max(dim=1, keepdim=True).values.detach()

#     B = zq.size(0)
#     diag = torch.eye(B, device=zq.device, dtype=torch.bool)
#     positive = y.view(-1, 1).eq(y.view(1, -1)) if same_class_positive else diag.clone()

#     if explicit_key:
#         positive = positive | diag
#         denom_mask = torch.ones_like(positive, dtype=torch.bool)
#     else:
#         positive = positive & (~diag)
#         denom_mask = ~diag

#     pos_count = positive.float().sum(dim=1)
#     valid = pos_count > 0

#     if not bool(valid.any().item()):
#         gics = features.sum() * 0.0
#     else:
#         exp_logits = torch.exp(logits).masked_fill(~denom_mask, 0.0)
#         denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
#         log_prob = logits - denom.log()
#         per = -(positive.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
#         per = per[valid]
#         yv = y[valid]
#         if per.numel() == 0:
#             gics = features.sum() * 0.0
#         elif class_balanced:
#             class_terms = []
#             for c in torch.unique(yv, sorted=True):
#                 cm = yv == c
#                 if bool(cm.any().item()):
#                     class_terms.append(per[cm].mean())
#             gics = torch.stack(class_terms).mean() if class_terms else features.sum() * 0.0
#         else:
#             gics = per.mean()

#     total = float(weight) * gics
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "gics": gics.detach(),
#         "weighted_gics": total.detach(),
#         "valid_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
#         "num_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
#         "mean_positive_count": pos_count[valid].float().mean().detach() if bool(valid.any().item()) else features.sum().detach() * 0.0,
#     }


# # Backward-compatible aliases.
# def base_fcs_geometry_contrastive_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_geometry_involved_contrastive_separation_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_supervised_contrastive_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_hsi_supervised_contrastive_loss(*args: Any, **kwargs: Any):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # PGR: Prospective Geometry Reserve
# # -----------------------------------------------------------------------------

# def _batch_subspace_overlap_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     rank: int = 3,
#     min_samples: int = 6,
#     max_overlap: float = 0.50,
#     normalize: bool = True,
#     include_mean_overlap: bool = True,
#     return_parts: bool = False,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """Margin-based subspace reserve.

#     The old loss minimized average overlap only.  That can look nonzero in logs
#     while leaving a single highly-overlapping class pair that breaks incremental
#     descriptor insertion.  This version explicitly penalizes pair overlaps above
#     max_overlap while still reporting mean/max overlap.
#     """
#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         if return_parts:
#             return {"total": z, "pair_count": z, "valid_class_count": z, "mean_overlap": z, "max_overlap": z}
#         return z
#     if features.dim() != 2:
#         raise ValueError(f"subspace loss expects [B,D], got {tuple(features.shape)}")

#     y = _as_1d_long(labels, device=features.device)
#     z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
#     _, D = z.shape

#     bases = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         n = int(m.sum().item())
#         if n < int(min_samples):
#             continue
#         xc = z[m] - z[m].mean(dim=0, keepdim=True)
#         r = max(1, min(int(rank), n - 1, D))
#         try:
#             _, s, vh = torch.linalg.svd(xc, full_matrices=False)
#         except RuntimeError:
#             continue
#         if s.numel() == 0 or float(s[0].detach().abs().item()) <= 1e-8:
#             continue
#         keep = (s[:r] / s[0].clamp_min(1e-8)) > 1e-3
#         if bool(keep.any().item()):
#             bases.append(vh[:r][keep].t().contiguous())

#     if len(bases) < 2:
#         loss = features.sum() * 0.0
#         overlaps = features.new_empty(0)
#         pair_count = 0
#     else:
#         overlap_list = []
#         for i in range(len(bases)):
#             for j in range(i + 1, len(bases)):
#                 Ui, Uj = bases[i], bases[j]
#                 denom = float(max(min(Ui.size(1), Uj.size(1)), 1))
#                 ov = (Ui.t() @ Uj).pow(2).sum() / denom
#                 overlap_list.append(ov)
#         overlaps = torch.stack(overlap_list) if overlap_list else features.new_empty(0)
#         pair_count = int(overlaps.numel())
#         if pair_count == 0:
#             loss = features.sum() * 0.0
#         else:
#             margin_loss = F.relu(overlaps - float(max_overlap)).pow(2).mean()
#             mean_loss = 0.10 * overlaps.mean() if include_mean_overlap else overlaps.sum() * 0.0
#             loss = margin_loss + mean_loss

#     if return_parts:
#         return {
#             "total": loss,
#             "pair_count": torch.tensor(float(pair_count), device=features.device, dtype=features.dtype),
#             "valid_class_count": torch.tensor(float(len(bases)), device=features.device, dtype=features.dtype),
#             "mean_overlap": overlaps.mean().detach() if overlaps.numel() else features.sum().detach() * 0.0,
#             "max_overlap": overlaps.max().detach() if overlaps.numel() else features.sum().detach() * 0.0,
#         }
#     return loss



# def risk_aware_band_discrimination_loss(
#     band_summary: Optional[torch.Tensor],
#     labels: torch.Tensor,
#     *,
#     features: Optional[torch.Tensor] = None,
#     min_samples: int = 3,
#     max_band_similarity: float = 0.75,
#     risk_center_margin: float = 1.0,
#     risk_weight: float = 1.0,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Band-guided feature-geometry reserve for HSI.

#     Important: raw physical spectra are *not* trainable. Therefore this loss
#     must not try to make raw band profiles dissimilar directly.  Instead, high
#     band similarity marks a hard class pair and increases the gradient on the
#     learned feature geometry for that pair.

#     Loss = hard_band_similarity.detach() * feature_center_conflict(features)

#     This keeps band information in the architecture while preserving the main
#     PG-RGA rule: inference and replay remain geometry-energy based.
#     """
#     ref = features if torch.is_tensor(features) else (band_summary if torch.is_tensor(band_summary) else labels)
#     if band_summary is None or not torch.is_tensor(band_summary) or band_summary.numel() == 0:
#         z = safe_zero_like(ref)
#         out = {
#             "total": z,
#             "band": z,
#             "pair_count": z,
#             "valid_class_count": z,
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     if band_summary.dim() != 2:
#         raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")

#     y = _as_1d_long(labels, device=band_summary.device)
#     if y.numel() != band_summary.size(0):
#         raise ValueError(f"band labels/batch mismatch: labels={y.numel()}, band={band_summary.size(0)}")

#     b = _band_importance_profile(band_summary)
#     b_centers, class_ids, _ = _class_centers(b, y, min_samples=min_samples, normalize_centers=True)
#     if b_centers.size(0) < 2:
#         z = band_summary.sum() * 0.0
#         out = {
#             "total": z,
#             "band": z,
#             "pair_count": z,
#             "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     sim = (b_centers @ b_centers.t()).clamp(0.0, 1.0)
#     eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
#     pair_sim = sim[~eye]
#     hard_band = F.relu(pair_sim - float(max_band_similarity)) / max(1.0 - float(max_band_similarity), 1e-6)
#     hard_band = hard_band.detach()

#     # The only trainable part is the learned feature geometry.  If features are
#     # missing or class ids cannot be aligned, the band term reports similarity
#     # but contributes zero gradient instead of applying a useless constant loss.
#     if features is not None and torch.is_tensor(features) and features.numel() > 0:
#         zf = F.normalize(features.to(device=band_summary.device, dtype=band_summary.dtype), dim=1, eps=1e-6)
#         f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
#         if f_centers.size(0) == b_centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
#             dist = torch.cdist(f_centers, f_centers, p=2)
#             feature_conflict = F.relu(float(risk_center_margin) - dist) / max(float(risk_center_margin), 1e-6)
#             feature_conflict = feature_conflict[~eye]
#             guided = hard_band * feature_conflict
#             loss_vec = hard_band * feature_conflict.pow(2)
#             loss = float(risk_weight) * (loss_vec.mean() if loss_vec.numel() > 0 else features.sum() * 0.0)
#         else:
#             guided = torch.zeros_like(pair_sim)
#             loss = features.sum() * 0.0
#     else:
#         guided = torch.zeros_like(pair_sim)
#         loss = band_summary.sum() * 0.0

#     if not return_parts:
#         return loss
#     return {
#         "total": loss,
#         "band": loss.detach(),
#         "pair_count": torch.tensor(float(pair_sim.numel()), device=band_summary.device, dtype=band_summary.dtype),
#         "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
#         "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
#         "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
#         "guided_conflict_mean": guided.mean().detach() if guided.numel() > 0 else band_summary.sum().detach() * 0.0,
#         "guided_conflict_max": guided.max().detach() if guided.numel() > 0 else band_summary.sum().detach() * 0.0,
#     }


# def prospective_geometry_reserve_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     band_summary: Optional[torch.Tensor] = None,
#     weight: float = 0.10,
#     compact_weight: float = 0.15,
#     center_weight: float = 0.20,
#     subspace_weight: float = 0.10,
#     band_weight: float = 0.05,
#     volume_weight: float = 0.05,
#     center_margin: float = 1.05,
#     min_class_samples: int = 3,
#     subspace_min_samples: int = 6,
#     subspace_rank: int = 3,
#     max_band_similarity: float = 0.75,
#     max_class_variance: float = 0.75,
#     min_class_variance: float = 0.015,
#     max_subspace_overlap: float = 0.50,
#     normalize_features: bool = True,
#     adaptive_component_weights: bool = True,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Base-phase PGR.

#     Active terms:
#       - compactness: same-class spread control
#       - center reserve: class centers separated by margin
#       - subspace reserve: explicit margin on tangent subspace overlap
#       - band reserve: risky classes avoid identical spectral-band profiles
#       - volume reserve: avoids both broad blobs and collapsed zero-volume rows
#     """
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z0 = safe_zero_like(features)
#         out = {
#             "total": z0, "pgr": z0, "weighted_pgr": z0,
#             "compact": z0, "center": z0, "subspace": z0, "band": z0, "volume": z0,
#             "valid_class_count": z0, "unique_class_count": z0,
#             "subspace_pair_count": z0, "band_pair_count": z0,
#             "compact_factor": z0, "center_factor": z0, "subspace_factor": z0,
#             "band_factor": z0, "volume_factor": z0,
#             "subspace_mean_overlap": z0, "subspace_max_overlap": z0,
#             "band_mean_similarity": z0, "band_max_similarity": z0,
#         }
#         return out if return_parts else z0

#     if features.dim() != 2:
#         raise ValueError(f"PGR expects features [B,D], got {tuple(features.shape)}")
#     _require_finite_tensor(features, "pgr.features")

#     y = _as_1d_long(labels, device=features.device)
#     if y.numel() != features.size(0):
#         raise ValueError(f"PGR labels/features mismatch: labels={y.numel()}, features={features.size(0)}")

#     # Let explicit aliases in kwargs override defaults without requiring trainer changes.
#     if "pgr_max_subspace_overlap" in kwargs:
#         max_subspace_overlap = float(kwargs["pgr_max_subspace_overlap"])
#     if "subspace_overlap_max" in kwargs:
#         max_subspace_overlap = float(kwargs["subspace_overlap_max"])
#     if "pgr_min_class_variance" in kwargs:
#         min_class_variance = float(kwargs["pgr_min_class_variance"])

#     z = F.normalize(features, dim=1, eps=1e-6) if normalize_features else features

#     compact_terms = []
#     volume_terms = []
#     class_vars = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         if int(m.sum().item()) < int(min_class_samples):
#             continue
#         xc = z[m]
#         var = (xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
#         compact_terms.append(var)
#         class_vars.append(var.detach())
#         broad = F.relu(var - float(max_class_variance)).pow(2)
#         collapsed = F.relu(float(min_class_variance) - var).pow(2)
#         volume_terms.append(broad + collapsed)

#     compact = torch.stack(compact_terms).mean() if compact_terms else features.sum() * 0.0
#     volume = torch.stack(volume_terms).mean() if volume_terms else features.sum() * 0.0
#     valid_class_count = len(compact_terms)
#     unique_class_count = int(torch.unique(y).numel())

#     centers, _, _ = _class_centers(z, y, min_samples=min_class_samples, normalize_centers=False)
#     center = _pairwise_center_margin_loss(centers, center_margin)

#     sub_obj = _batch_subspace_overlap_loss(
#         z,
#         y,
#         rank=int(subspace_rank),
#         min_samples=int(subspace_min_samples),
#         max_overlap=float(max_subspace_overlap),
#         normalize=False,
#         return_parts=True,
#     )
#     subspace = sub_obj["total"]
#     subspace_pair_count = sub_obj["pair_count"]

#     band_obj = risk_aware_band_discrimination_loss(
#         band_summary,
#         y,
#         features=z,
#         min_samples=int(min_class_samples),
#         max_band_similarity=float(max_band_similarity),
#         risk_center_margin=float(center_margin),
#         return_parts=True,
#     )
#     band = band_obj["total"] if isinstance(band_obj, dict) else band_obj
#     band_pair_count = band_obj.get("pair_count", features.sum().detach() * 0.0) if isinstance(band_obj, dict) else features.sum().detach() * 0.0

#     one = torch.tensor(1.0, device=features.device, dtype=features.dtype)
#     zero = features.sum() * 0.0
#     if adaptive_component_weights:
#         compact_factor = one if valid_class_count > 0 else zero
#         volume_factor = one if valid_class_count > 0 else zero
#         center_factor = one if int(centers.size(0)) >= 2 else zero
#         subspace_factor = one if float(subspace_pair_count.detach().item()) > 0.0 else zero
#         band_factor = one if float(band_pair_count.detach().item()) > 0.0 else zero
#     else:
#         compact_factor = center_factor = subspace_factor = band_factor = volume_factor = one

#     pgr_unweighted = (
#         float(compact_weight) * compact_factor * compact
#         + float(center_weight) * center_factor * center
#         + float(subspace_weight) * subspace_factor * subspace
#         + float(band_weight) * band_factor * band
#         + float(volume_weight) * volume_factor * volume
#     )
#     total = float(weight) * pgr_unweighted

#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "pgr": pgr_unweighted.detach(),
#         "weighted_pgr": total.detach(),
#         "compact": compact.detach(),
#         "center": center.detach(),
#         "subspace": subspace.detach(),
#         "band": band.detach(),
#         "volume": volume.detach(),
#         "valid_class_count": torch.tensor(float(valid_class_count), device=features.device, dtype=features.dtype),
#         "unique_class_count": torch.tensor(float(unique_class_count), device=features.device, dtype=features.dtype),
#         "subspace_pair_count": subspace_pair_count.detach(),
#         "band_pair_count": band_pair_count.detach() if torch.is_tensor(band_pair_count) else torch.tensor(float(band_pair_count), device=features.device, dtype=features.dtype),
#         "compact_factor": compact_factor.detach(),
#         "center_factor": center_factor.detach(),
#         "subspace_factor": subspace_factor.detach(),
#         "band_factor": band_factor.detach(),
#         "volume_factor": volume_factor.detach(),
#         "subspace_mean_overlap": sub_obj.get("mean_overlap", zero).detach(),
#         "subspace_max_overlap": sub_obj.get("max_overlap", zero).detach(),
#         "band_mean_similarity": band_obj.get("mean_similarity", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "band_max_similarity": band_obj.get("max_similarity", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "band_guided_conflict_mean": band_obj.get("guided_conflict_mean", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "band_guided_conflict_max": band_obj.get("guided_conflict_max", zero).detach() if isinstance(band_obj, dict) else zero.detach(),
#         "class_variance_mean": torch.stack(class_vars).mean() if class_vars else zero.detach(),
#     }


# def base_prospective_geometry_reserve_loss(*args: Any, **kwargs: Any):
#     return prospective_geometry_reserve_loss(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # Physical spectral-shape reserve
# # -----------------------------------------------------------------------------


# def spectral_shape_discrimination_loss(
#     spectral_summary: Optional[torch.Tensor],
#     labels: torch.Tensor,
#     *,
#     features: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     require_physical_summary: bool = True,
#     min_samples: int = 3,
#     max_shape_similarity: float = 0.75,
#     risk_center_margin: float = 1.0,
#     risk_weight: float = 1.0,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Spectral-shape-guided feature-geometry reserve.

#     Raw wavelength spectra are fixed metadata, not trainable outputs.  High
#     spectral-shape similarity should therefore identify hard HSI class pairs and
#     push their learned feature centers apart.  It should not be optimized as a
#     standalone raw-spectrum dissimilarity objective.
#     """
#     ref = features if torch.is_tensor(features) else (spectral_summary if torch.is_tensor(spectral_summary) else labels)
#     if (
#         spectral_summary is None
#         or not torch.is_tensor(spectral_summary)
#         or spectral_summary.numel() == 0
#         or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
#     ):
#         z = safe_zero_like(ref)
#         out = {
#             "total": z,
#             "spectral_shape": z,
#             "pair_count": z,
#             "valid_class_count": z,
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     if s.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(s.shape)}")
#     y = _as_1d_long(labels, device=s.device)
#     if y.numel() != s.size(0):
#         raise ValueError(f"spectral_summary/label mismatch: spectra={s.size(0)}, labels={y.numel()}")

#     desc = _spectral_profile_descriptor(s)
#     centers, class_ids, _ = _class_centers(desc, y, min_samples=min_samples, normalize_centers=False)

#     if centers.size(0) < 2:
#         z = s.sum() * 0.0
#         out = {
#             "total": z,
#             "spectral_shape": z,
#             "pair_count": z,
#             "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
#             "mean_similarity": z,
#             "max_similarity": z,
#             "guided_conflict_mean": z,
#             "guided_conflict_max": z,
#         }
#         return out if return_parts else z

#     sim = _positive_cosine_matrix(centers, centers)
#     eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
#     pair_sim = sim[~eye]
#     hard_shape = F.relu(pair_sim - float(max_shape_similarity)) / max(1.0 - float(max_shape_similarity), 1e-6)
#     hard_shape = hard_shape.detach()

#     if features is not None and torch.is_tensor(features) and features.numel() > 0:
#         zf = F.normalize(features.to(device=s.device, dtype=s.dtype), dim=1, eps=1e-6)
#         f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
#         if f_centers.size(0) == centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
#             dist = torch.cdist(f_centers, f_centers, p=2)
#             feature_conflict = F.relu(float(risk_center_margin) - dist)[~eye] / max(float(risk_center_margin), 1e-6)
#             guided = hard_shape * feature_conflict
#             loss_vec = hard_shape * feature_conflict.pow(2)
#             loss = float(risk_weight) * (loss_vec.mean() if loss_vec.numel() > 0 else features.sum() * 0.0)
#         else:
#             guided = torch.zeros_like(pair_sim)
#             loss = features.sum() * 0.0
#     else:
#         guided = torch.zeros_like(pair_sim)
#         loss = s.sum() * 0.0

#     if not return_parts:
#         return loss
#     return {
#         "total": loss,
#         "spectral_shape": loss.detach(),
#         "pair_count": torch.tensor(float(pair_sim.numel()), device=s.device, dtype=s.dtype),
#         "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
#         "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
#         "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
#         "guided_conflict_mean": guided.mean().detach() if guided.numel() > 0 else s.sum().detach() * 0.0,
#         "guided_conflict_max": guided.max().detach() if guided.numel() > 0 else s.sum().detach() * 0.0,
#     }


# # -----------------------------------------------------------------------------
# # Low-rank GeometryBank energy used by classifier/replay/margins
# # -----------------------------------------------------------------------------

# def _canonicalize_variances(
#     *,
#     eigvals: Optional[torch.Tensor] = None,
#     res_vars: Optional[torch.Tensor] = None,
#     variances: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if variances is not None and torch.is_tensor(variances):
#         if variances.dim() != 2 or variances.size(1) < 2:
#             raise ValueError(f"variances must be [C,R+1], got {tuple(variances.shape)}")
#         return variances[:, :-1], variances[:, -1]
#     if eigvals is None or res_vars is None:
#         raise ValueError("Either variances or eigvals+res_vars must be provided.")
#     return eigvals, res_vars


# def _active_rank_mask(active_ranks: Optional[torch.Tensor], C: int, R: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
#     if active_ranks is None or not torch.is_tensor(active_ranks):
#         ar = torch.full((C,), int(R), device=device, dtype=torch.long)
#     else:
#         ar = active_ranks.to(device=device).long().flatten()
#         if ar.numel() != C:
#             raise ValueError(f"active_ranks must have C={C} entries, got {ar.numel()}")
#         ar = ar.clamp(min=0, max=R)
#     idx = torch.arange(R, device=device).view(1, R)
#     mask = (idx < ar.view(C, 1)).to(dtype=dtype)
#     return mask, ar



# def geometry_energy_matrix(
#     features: Optional[torch.Tensor] = None,
#     means: Optional[torch.Tensor] = None,
#     bases: Optional[torch.Tensor] = None,
#     variances: Optional[torch.Tensor] = None,
#     *,
#     bank: Optional[Mapping[str, torch.Tensor]] = None,
#     eigvals: Optional[torch.Tensor] = None,
#     res_vars: Optional[torch.Tensor] = None,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     reliability_energy_weight: float = 0.03,
#     reliability_min_clamp: float = 0.05,
#     residual_variance_scale: float = 0.75,
#     normalize_by_dim: bool = True,
#     invalid_class_energy: float = _INVALID_ENERGY,
#     use_logdet_energy: bool = True,
#     logdet_energy_weight: float = 0.05,
#     logdet_normalize_by_dim: bool = True,
#     center_logdet_energy: bool = True,
#     # accepted but intentionally inactive in PG-RGA main path
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_curve_means: Optional[torch.Tensor] = None,
#     spectral_curve_vars: Optional[torch.Tensor] = None,
#     spectral_curve_d1: Optional[torch.Tensor] = None,
#     spectral_curve_d2: Optional[torch.Tensor] = None,
#     spectral_shape_reliability: Optional[torch.Tensor] = None,
#     use_spectral_residual_energy: bool = False,
#     spectral_energy_weight: float = 0.0,
#     spectral_summary_is_physical: bool = False,
#     spectral_require_physical_summary: bool = True,
#     return_parts: bool = False,
#     **_: Any,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """Low-rank GeometryBank energy used by classifier/replay/margins.

#     Supports both explicit tensors and a bank mapping.  Spectral arguments are
#     accepted only for API compatibility; PG-RGA inference remains feature-geometry
#     energy, not spectral-branch energy.
#     """
#     del spectral_summary, spectral_curve_means, spectral_curve_vars, spectral_curve_d1, spectral_curve_d2, spectral_shape_reliability

#     if isinstance(means, Mapping) and bank is None and bases is None:
#         bank = means  # supports geometry_energy_matrix(features, bank=...) style through positional means
#         means = None
#     if bank is not None:
#         means = bank.get("means", means)
#         bases = bank.get("bases", bank.get("subspace_bases", bases))
#         variances = bank.get("variances", variances)
#         eigvals = bank.get("eigvals", bank.get("eigenvalues", eigvals))
#         res_vars = bank.get("res_vars", bank.get("residual_variances", res_vars))
#         active_ranks = bank.get("active_ranks", active_ranks)
#         reliability = bank.get("reliability", reliability)
#         sample_counts = bank.get("sample_counts", sample_counts)

#     if means is None or bases is None:
#         raise ValueError("geometry_energy_matrix requires means and bases, or a bank mapping containing them.")

#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         device = means.device if torch.is_tensor(means) else torch.device("cpu")
#         dtype = means.dtype if torch.is_tensor(means) else torch.float32
#         C = int(means.size(0)) if torch.is_tensor(means) and means.dim() >= 1 else 0
#         empty = torch.empty((0, C), device=device, dtype=dtype)
#         if return_parts:
#             return {"energy": empty, "feature_energy": empty, "parallel": empty, "orthogonal": empty,
#                     "parallel_energy": empty, "residual_energy": empty, "spectral_energy": empty}
#         return empty

#     if features.dim() != 2:
#         raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
#     if means.dim() != 2:
#         raise ValueError(f"means must be [C,D], got {tuple(means.shape)}")
#     if bases.dim() != 3:
#         raise ValueError(f"bases must be [C,D,R], got {tuple(bases.shape)}")
#     B, D = features.shape
#     C, Dm = means.shape
#     if Dm != D or bases.size(0) != C or bases.size(1) != D:
#         raise ValueError(f"feature/bank shape mismatch: features={tuple(features.shape)}, means={tuple(means.shape)}, bases={tuple(bases.shape)}")

#     device, dtype = features.device, features.dtype
#     means = means.to(device=device, dtype=dtype)
#     bases = bases.to(device=device, dtype=dtype)
#     eig, rv = _canonicalize_variances(eigvals=eigvals, res_vars=res_vars, variances=variances)
#     eig = eig.to(device=device, dtype=dtype)
#     rv = rv.to(device=device, dtype=dtype).flatten()

#     R = int(bases.size(2))
#     if eig.dim() != 2 or eig.size(0) != C or eig.size(1) != R:
#         raise ValueError(f"eigvals/variances rank mismatch: eig={tuple(eig.shape)}, C={C}, R={R}")
#     if rv.numel() != C:
#         raise ValueError(f"res_vars must have C={C} entries, got {rv.numel()}")

#     rank_mask, ar = _active_rank_mask(active_ranks, C, R, device, dtype)

#     delta = features.unsqueeze(1) - means.unsqueeze(0)
#     coeff = torch.einsum("bcd,cdr->bcr", delta, bases)
#     coeff_active = coeff * rank_mask.view(1, C, R)
#     recon = torch.einsum("bcr,cdr->bcd", coeff_active, bases)
#     residual = delta - recon

#     eig_safe = eig.clamp_min(float(variance_floor))
#     rv_safe = (rv * float(residual_variance_scale)).clamp_min(float(variance_floor))

#     parallel = ((coeff_active.pow(2) / eig_safe.view(1, C, R)) * rank_mask.view(1, C, R)).sum(dim=-1)
#     orthogonal = residual.pow(2).sum(dim=-1) / rv_safe.view(1, C)
#     energy = parallel + orthogonal
#     if bool(normalize_by_dim):
#         energy = energy / float(max(D, 1))

#     logdet_penalty = torch.zeros((C,), device=device, dtype=dtype)
#     if bool(use_logdet_energy) and float(logdet_energy_weight) > 0.0:
#         active_logdet = (eig_safe.log() * rank_mask).sum(dim=1)
#         residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=dtype)
#         logdet_penalty = active_logdet + residual_dims * rv_safe.log()
#         if bool(logdet_normalize_by_dim):
#             logdet_penalty = logdet_penalty / float(max(D, 1))
#         if bool(center_logdet_energy):
#             logdet_penalty = logdet_penalty - logdet_penalty.mean().detach()
#         energy = energy + float(logdet_energy_weight) * logdet_penalty.view(1, C)

#     reliability_penalty = torch.zeros((C,), device=device, dtype=dtype)
#     if reliability is not None and torch.is_tensor(reliability) and float(reliability_energy_weight) > 0.0:
#         rel = torch.nan_to_num(reliability.to(device=device, dtype=dtype).flatten(), nan=float(reliability_min_clamp), posinf=1.0, neginf=float(reliability_min_clamp))
#         if rel.numel() != C:
#             raise ValueError(f"reliability must have C={C} entries, got {rel.numel()}")
#         rel = rel.clamp(float(reliability_min_clamp), 1.0)
#         reliability_penalty = -rel.log()
#         reliability_penalty = reliability_penalty - reliability_penalty.mean().detach()
#         energy = energy + float(reliability_energy_weight) * reliability_penalty.view(1, C)

#     spectral_energy = torch.zeros_like(energy)
#     if bool(use_spectral_residual_energy) and float(spectral_energy_weight) > 0.0:
#         if bool(spectral_require_physical_summary) and not bool(spectral_summary_is_physical):
#             spectral_energy = torch.zeros_like(energy)
#         else:
#             spectral_energy = torch.zeros_like(energy)

#     energy = energy + spectral_energy

#     # Mask invalid/uninitialized classes so stale future rows can never win.
#     valid_class = torch.ones((C,), device=device, dtype=torch.bool)
#     if sample_counts is not None and torch.is_tensor(sample_counts):
#         sc = sample_counts.to(device=device).flatten()
#         if sc.numel() == C:
#             valid_class = sc > 0
#     valid_class = valid_class & (ar > 0)
#     if bool((~valid_class).any().item()):
#         energy = energy.masked_fill((~valid_class).view(1, C), float(invalid_class_energy))
#         parallel = parallel.masked_fill((~valid_class).view(1, C), float(invalid_class_energy))
#         orthogonal = orthogonal.masked_fill((~valid_class).view(1, C), float(invalid_class_energy))

#     energy = torch.nan_to_num(energy, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0)

#     if not return_parts:
#         return energy
#     return {
#         "energy": energy,
#         "feature_energy": energy,
#         "parallel": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "orthogonal": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "parallel_energy": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "residual_energy": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
#         "logdet_penalty": logdet_penalty,
#         "reliability_penalty": reliability_penalty,
#         "spectral_energy": spectral_energy,
#         "active_ranks": ar,
#         "rank_mask": rank_mask,
#         "valid_class_mask": valid_class,
#     }


# # -----------------------------------------------------------------------------
# # Unified base objective
# # -----------------------------------------------------------------------------

# def base_geometry_preparation_loss(
#     *,
#     logits: Optional[torch.Tensor] = None,
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     key_features: Optional[torch.Tensor] = None,
#     band_summary: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     ce_weight: float = 0.0,
#     base_geometry_weight: float = 1.0,
#     label_smoothing: float = 0.0,
#     # GICS
#     gics_weight: float = 0.20,
#     gics_temperature: float = 0.07,
#     # PGR
#     pgr_weight: float = 0.10,
#     pgr_compact_weight: float = 0.15,
#     pgr_center_weight: float = 0.20,
#     pgr_subspace_weight: float = 0.10,
#     pgr_band_weight: float = 0.05,
#     pgr_volume_weight: float = 0.05,
#     pgr_center_margin: float = 1.05,
#     pgr_max_band_similarity: float = 0.75,
#     pgr_max_class_variance: float = 0.75,
#     pgr_min_class_variance: float = 0.015,
#     pgr_max_subspace_overlap: float = 0.50,
#     subspace_rank: int = 3,
#     min_class_samples: int = 3,
#     subspace_min_samples: int = 6,
#     # spectral shape
#     spectral_shape_weight: float = 0.05,
#     max_spectral_shape_similarity: float = 0.75,
#     spectral_shape_risk_weight: float = 1.0,
#     require_physical_summary: bool = True,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Single active base-phase regularizer.

#     BasePhaseTrainer computes balanced CE separately, so ce_weight defaults to
#     0.0.  The returned 'total' remains differentiable and must be used for SRPGR.
#     """
#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         out = {
#             "total": z, "ce": z, "base_geometry": z,
#             "base_gics": z, "base_gics_anchors": z, "base_gics_pos": z,
#             "base_pgr": z, "base_compact": z, "base_center": z,
#             "base_subspace": z, "base_band": z, "base_volume": z,
#             "base_spectral_shape": z, "base_spectral_shape_raw": z,
#             "base_spectral_shape_mean_similarity": z, "base_spectral_shape_pair_count": z,
#             "base_spectral_shape_active": z,
#         }
#         return out if return_parts else z

#     if features.dim() != 2:
#         raise ValueError(f"base features must be [B,D], got {tuple(features.shape)}")
#     _require_finite_tensor(features, "base.features")

#     labels = _as_1d_long(labels, device=features.device)
#     if labels.numel() != features.size(0):
#         raise ValueError(f"base labels/features mismatch: labels={labels.numel()}, features={features.size(0)}")

#     ce = safe_zero_like(features)
#     if ce_weight > 0.0:
#         if logits is None or not torch.is_tensor(logits):
#             raise ValueError("logits are required when ce_weight > 0.")
#         if logits.dim() != 2 or logits.size(0) != labels.numel():
#             raise ValueError(f"logits must be [B,C] aligned with labels, got {tuple(logits.shape)}")
#         ce = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))

#     # If physical spectra are present, they are the correct band reserve input.
#     # This avoids PCA-30 band profiles competing with raw-200 spectral descriptors.
#     band_for_pgr = band_summary
#     if spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0 and bool(spectral_summary_is_physical):
#         band_for_pgr = spectral_summary

#     gics = base_geometry_involved_contrastive_loss(
#         features,
#         labels,
#         key_features=key_features,
#         weight=float(gics_weight),
#         temperature=float(gics_temperature),
#         return_parts=True,
#     )

#     pgr = prospective_geometry_reserve_loss(
#         features,
#         labels,
#         band_summary=band_for_pgr,
#         weight=float(pgr_weight),
#         compact_weight=float(pgr_compact_weight),
#         center_weight=float(pgr_center_weight),
#         subspace_weight=float(pgr_subspace_weight),
#         band_weight=float(pgr_band_weight),
#         volume_weight=float(pgr_volume_weight),
#         center_margin=float(pgr_center_margin),
#         min_class_samples=int(min_class_samples),
#         subspace_min_samples=int(subspace_min_samples),
#         subspace_rank=int(subspace_rank),
#         max_band_similarity=float(pgr_max_band_similarity),
#         max_class_variance=float(pgr_max_class_variance),
#         min_class_variance=float(pgr_min_class_variance),
#         max_subspace_overlap=float(kwargs.get("subspace_overlap_max", pgr_max_subspace_overlap)),
#         return_parts=True,
#     )

#     shape_raw = spectral_shape_discrimination_loss(
#         spectral_summary,
#         labels,
#         features=features,
#         spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         require_physical_summary=bool(require_physical_summary),
#         min_samples=int(min_class_samples),
#         max_shape_similarity=float(max_spectral_shape_similarity),
#         risk_center_margin=float(pgr_center_margin),
#         risk_weight=float(spectral_shape_risk_weight),
#         return_parts=True,
#     )
#     shape_total = float(spectral_shape_weight) * _scalar(shape_raw.get("total", safe_zero_like(features)), features)

#     gics_total = _scalar(gics.get("total", safe_zero_like(features)), features)
#     pgr_total = _scalar(pgr.get("total", safe_zero_like(features)), features)
#     base_geometry = float(base_geometry_weight) * (gics_total + pgr_total + shape_total)
#     total = float(ce_weight) * ce + base_geometry

#     if not return_parts:
#         return total

#     spectral_active = bool(spectral_summary_is_physical) and spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0

#     return {
#         "total": total,
#         "ce": ce.detach(),
#         # Keep differentiable value for code that still uses this key; logs can detach later.
#         "base_geometry": base_geometry,

#         "base_gics": _scalar(gics.get("gics", safe_zero_like(features)), features).detach(),
#         "base_gics_weighted": gics_total.detach(),
#         "base_gics_anchors": _scalar(gics.get("valid_anchors", safe_zero_like(features)), features).detach(),
#         "base_gics_pos": _scalar(gics.get("mean_positive_count", safe_zero_like(features)), features).detach(),

#         "base_pgr": _scalar(pgr.get("pgr", safe_zero_like(features)), features).detach(),
#         "base_pgr_weighted": pgr_total.detach(),
#         "base_compact": _scalar(pgr.get("compact", safe_zero_like(features)), features).detach(),
#         "base_center": _scalar(pgr.get("center", safe_zero_like(features)), features).detach(),
#         "base_subspace": _scalar(pgr.get("subspace", safe_zero_like(features)), features).detach(),
#         "base_band": _scalar(pgr.get("band", safe_zero_like(features)), features).detach(),
#         "base_volume": _scalar(pgr.get("volume", safe_zero_like(features)), features).detach(),

#         "base_spectral_shape": shape_total.detach(),
#         "base_spectral_shape_raw": _scalar(shape_raw.get("total", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_mean_similarity": _scalar(shape_raw.get("mean_similarity", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_max_similarity": _scalar(shape_raw.get("max_similarity", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_pair_count": _scalar(shape_raw.get("pair_count", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_active": torch.tensor(float(spectral_active), device=features.device, dtype=features.dtype),

#         "base_pgr_valid_class_count": _scalar(pgr.get("valid_class_count", safe_zero_like(features)), features).detach(),
#         "base_pgr_subspace_pair_count": _scalar(pgr.get("subspace_pair_count", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_pair_count": _scalar(pgr.get("band_pair_count", safe_zero_like(features)), features).detach(),
#         "base_pgr_volume_factor": _scalar(pgr.get("volume_factor", safe_zero_like(features)), features).detach(),
#         "base_pgr_subspace_max_overlap": _scalar(pgr.get("subspace_max_overlap", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_max_similarity": _scalar(pgr.get("band_max_similarity", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_guided_conflict_mean": _scalar(pgr.get("band_guided_conflict_mean", safe_zero_like(features)), features).detach(),
#         "base_pgr_band_guided_conflict_max": _scalar(pgr.get("band_guided_conflict_max", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_guided_conflict_mean": _scalar(shape_raw.get("guided_conflict_mean", safe_zero_like(features)), features).detach(),
#         "base_spectral_shape_guided_conflict_max": _scalar(shape_raw.get("guided_conflict_max", safe_zero_like(features)), features).detach(),
#     }


# def unified_spectral_geometry_loss(
#     *,
#     phase: str,
#     labels: torch.Tensor,
#     logits: Optional[torch.Tensor] = None,
#     features: Optional[torch.Tensor] = None,
#     key_features: Optional[torch.Tensor] = None,
#     band_summary: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     return_parts: bool = True,
#     **kwargs: Any,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Public loss entry used by BasePhaseTrainer.

#     Incremental training should use explicit geometry_energy_matrix + margin
#     losses, not a monolithic hidden loss stack.
#     """
#     p = str(phase).strip().lower()
#     if p not in {"base", "phase0", "phase_0", "0"}:
#         raise RuntimeError(
#             "unified_spectral_geometry_loss currently owns only the base phase. "
#             "For incremental PG-RGA, use geometry_energy_matrix(), "
#             "geometry_energy_margin_loss(), and old_new_invasion_loss()."
#         )
#     return base_geometry_preparation_loss(
#         logits=logits,
#         features=features,
#         labels=labels,
#         key_features=key_features,
#         band_summary=band_summary,
#         spectral_summary=spectral_summary,
#         spectral_summary_is_physical=spectral_summary_is_physical,
#         return_parts=return_parts,
#         **kwargs,
#     )


# class UnifiedSpectralGeometryLoss:
#     """Thin callable wrapper for compatibility with older code."""

#     def __init__(self, **defaults: Any) -> None:
#         self.defaults = dict(defaults)

#     def __call__(self, **kwargs: Any):
#         merged = dict(self.defaults)
#         merged.update(kwargs)
#         return unified_spectral_geometry_loss(**merged)


# # -----------------------------------------------------------------------------
# # Incremental margin losses used by PG-RGA
# # -----------------------------------------------------------------------------

# def geometry_energy_margin_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         return safe_zero_like(labels if torch.is_tensor(labels) else None)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
#         raise RuntimeError("labels outside local energy range")
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     true_mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
#     nearest_wrong = energy.masked_fill(true_mask, float("inf")).min(dim=1).values
#     loss = F.relu(true_e + float(margin) - nearest_wrong)
#     finite = torch.isfinite(loss)
#     return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0


# def old_new_invasion_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     old_class_count: int,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         return safe_zero_like(labels if torch.is_tensor(labels) else None)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     C = int(energy.size(1))
#     old = int(max(0, min(int(old_class_count), C)))
#     if old <= 0 or old >= C:
#         return energy.sum() * 0.0
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= C):
#         raise RuntimeError("labels outside local energy range")
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     old_min = energy[:, :old].min(dim=1).values
#     new_min = energy[:, old:].min(dim=1).values
#     is_old = y < old
#     opposite = torch.where(is_old, new_min, old_min)
#     loss = F.relu(true_e + float(margin) - opposite)
#     finite = torch.isfinite(loss)
#     return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0


# # -----------------------------------------------------------------------------
# # Base diagnostics
# # -----------------------------------------------------------------------------

# @torch.no_grad()
# def base_center_overlap_diagnostics(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     normalize: bool = True,
#     min_samples: int = 2,
# ) -> Dict[str, torch.Tensor]:
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         return {"compact": z, "mean_center_margin": z, "min_center_margin": z, "num_classes": z}

#     z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
#     y = _as_1d_long(labels, device=z.device)
#     centers, _, _ = _class_centers(z, y, min_samples=min_samples)

#     compact_terms = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         if int(m.sum().item()) >= int(min_samples):
#             xc = z[m]
#             compact_terms.append((xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean())
#     compact = torch.stack(compact_terms).mean() if compact_terms else z.sum() * 0.0

#     if centers.size(0) < 2:
#         mean_margin = z.sum() * 0.0
#         min_margin = z.sum() * 0.0
#     else:
#         dist = torch.cdist(centers, centers, p=2)
#         eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
#         pair = dist[~eye]
#         mean_margin = pair.mean()
#         min_margin = pair.min()

#     return {
#         "compact": compact.detach(),
#         "mean_center_margin": mean_margin.detach(),
#         "min_center_margin": min_margin.detach(),
#         "num_classes": torch.tensor(float(centers.size(0)), device=z.device, dtype=z.dtype),
#     }


# @torch.no_grad()
# def base_gics_diagnostics(features: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#     out = base_geometry_involved_contrastive_loss(features, labels, weight=1.0, return_parts=True, **kwargs)
#     return {
#         "gics": out["gics"].detach(),
#         "valid_anchors": out["valid_anchors"].detach(),
#         "positive_pairs": out["mean_positive_count"].detach(),
#     }


# def base_supcon_diagnostics(*args: Any, **kwargs: Any):
#     return base_gics_diagnostics(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # Disabled legacy boundary helper
# # -----------------------------------------------------------------------------

# def sample_boundary_geometry_features(*args: Any, **kwargs: Any):
#     raise RuntimeError(
#         "sample_boundary_geometry_features is not part of main path. "
#         "Use GeometryBank synthetic replay, not boundary replay."
#     )
