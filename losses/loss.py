from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# PG-RGA / SRPGR loss file
# =============================================================================
# Phase-0 active objective:
#   CE is computed in BasePhaseTrainer for class balancing.
#   This file provides GICS + PGR + physical spectral-shape/band reserve.
#
# Incremental compatibility:
#   geometry_energy_matrix(), geometry_energy_margin_loss(), and
#   old_new_invasion_loss() are active because PG-RGA incremental replay/margins
#   need the same low-rank GeometryBank energy used by the classifier.
# =============================================================================

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
    """Base-phase band-signature reserve.

    Uses derivative-aware band profiles, so physical raw spectra produce a real
    spectral/band reserve.  It is a regularizer, not a classifier branch.
    """
    ref = features if torch.is_tensor(features) else labels
    if band_summary is None or not torch.is_tensor(band_summary) or band_summary.numel() == 0:
        z = safe_zero_like(ref)
        out = {"total": z, "band": z, "pair_count": z, "valid_class_count": z, "mean_similarity": z, "max_similarity": z}
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
        }
        return out if return_parts else z

    sim = (b_centers @ b_centers.t()).clamp(0.0, 1.0)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    pair_sim = sim[~eye]

    if features is not None and torch.is_tensor(features) and features.numel() > 0:
        zf = F.normalize(features.to(device=band_summary.device, dtype=band_summary.dtype), dim=1, eps=1e-6)
        f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
        if f_centers.size(0) == b_centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
            dist = torch.cdist(f_centers, f_centers, p=2)
            center_conflict = F.relu(float(risk_center_margin) - dist) / max(float(risk_center_margin), 1e-6)
            conflict_weight = center_conflict[~eye].detach()
        else:
            conflict_weight = torch.ones_like(pair_sim)
    else:
        conflict_weight = torch.ones_like(pair_sim)

    loss_vec = F.relu(pair_sim - float(max_band_similarity)).pow(2) * (1.0 + float(risk_weight) * conflict_weight)
    loss = loss_vec.mean() if loss_vec.numel() > 0 else band_summary.sum() * 0.0

    if not return_parts:
        return loss
    return {
        "total": loss,
        "band": loss.detach(),
        "pair_count": torch.tensor(float(loss_vec.numel()), device=band_summary.device, dtype=band_summary.dtype),
        "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
        "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
        "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
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
    """HSI spectral-shape reserve.

    Active only for raw wavelength-ordered spectra unless
    require_physical_summary=False.  The descriptor uses standardized spectrum +
    derivatives, making it align with current HSI spectral-shape practice.
    """
    ref = spectral_summary if torch.is_tensor(spectral_summary) else labels
    if (
        spectral_summary is None
        or not torch.is_tensor(spectral_summary)
        or spectral_summary.numel() == 0
        or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
    ):
        z = safe_zero_like(ref)
        out = {"total": z, "spectral_shape": z, "pair_count": z, "valid_class_count": z, "mean_similarity": z, "max_similarity": z}
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
        }
        return out if return_parts else z

    sim = _positive_cosine_matrix(centers, centers)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    pair_sim = sim[~eye]

    if features is not None and torch.is_tensor(features) and features.numel() > 0:
        zf = F.normalize(features.to(device=s.device, dtype=s.dtype), dim=1, eps=1e-6)
        f_centers, f_ids, _ = _class_centers(zf, y, min_samples=min_samples, normalize_centers=False)
        if f_centers.size(0) == centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
            dist = torch.cdist(f_centers, f_centers, p=2)
            conflict = F.relu(float(risk_center_margin) - dist)[~eye].detach() / max(float(risk_center_margin), 1e-6)
        else:
            conflict = torch.ones_like(pair_sim)
    else:
        conflict = torch.ones_like(pair_sim)

    loss_vec = F.relu(pair_sim - float(max_shape_similarity)).pow(2) * (1.0 + float(risk_weight) * conflict)
    loss = loss_vec.mean() if loss_vec.numel() > 0 else s.sum() * 0.0

    if not return_parts:
        return loss
    return {
        "total": loss,
        "spectral_shape": loss.detach(),
        "pair_count": torch.tensor(float(loss_vec.numel()), device=s.device, dtype=s.dtype),
        "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
        "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
        "max_similarity": pair_sim.max().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
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
    *,
    features: torch.Tensor,
    means: torch.Tensor,
    bases: torch.Tensor,
    variances: Optional[torch.Tensor] = None,
    eigvals: Optional[torch.Tensor] = None,
    res_vars: Optional[torch.Tensor] = None,
    active_ranks: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    sample_counts: Optional[torch.Tensor] = None,
    variance_floor: float = 1e-4,
    reliability_energy_weight: float = 0.03,
    reliability_min_clamp: float = 0.05,
    residual_variance_scale: float = 0.75,
    normalize_by_dim: bool = True,
    invalid_class_energy: float = _INVALID_ENERGY,
    use_logdet_energy: bool = True,
    logdet_energy_weight: float = 0.05,
    logdet_normalize_by_dim: bool = True,
    center_logdet_energy: bool = True,
    # accepted but intentionally inactive in PG-RGA main path
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
    """Low-rank Gaussian energy against explicit GeometryBank tensors.

    Returns energy [B,C].  When return_parts=True, returns a dict containing
    'energy' and diagnostic terms.  This mirrors the strict classifier energy,
    allowing trainer diagnostics and incremental margins to work even when they
    call the loss module directly.
    """
    del sample_counts, spectral_summary, spectral_curve_means, spectral_curve_vars, spectral_curve_d1, spectral_curve_d2, spectral_shape_reliability

    if features is None or not torch.is_tensor(features) or features.numel() == 0:
        z = safe_zero_like(features)
        empty = z.view(0, 0)
        return {"energy": empty, "feature_energy": empty} if return_parts else empty

    if features.dim() != 2:
        raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
    if means.dim() != 2:
        raise ValueError(f"means must be [C,D], got {tuple(means.shape)}")
    if bases.dim() != 3:
        raise ValueError(f"bases must be [C,D,R], got {tuple(bases.shape)}")
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
    if eig.dim() != 2 or eig.size(0) != C or eig.size(1) != R:
        raise ValueError(f"eigvals/variances rank mismatch: eig={tuple(eig.shape)}, C={C}, R={R}")
    if rv.numel() != C:
        raise ValueError(f"res_vars must have C={C} entries, got {rv.numel()}")

    rank_mask, ar = _active_rank_mask(active_ranks, C, R, device, dtype)

    delta = features.unsqueeze(1) - means.unsqueeze(0)                    # [B,C,D]
    coeff = torch.einsum("bcd,cdr->bcr", delta, bases)                  # [B,C,R]
    coeff_active = coeff * rank_mask.view(1, C, R)
    recon = torch.einsum("bcr,cdr->bcd", coeff_active, bases)
    residual = delta - recon

    eig_safe = eig.clamp_min(float(variance_floor))
    rv_safe = (rv * float(residual_variance_scale)).clamp_min(float(variance_floor))

    parallel = ((coeff_active.pow(2) / eig_safe.view(1, C, R)) * rank_mask.view(1, C, R)).sum(dim=-1)
    orthogonal = residual.pow(2).sum(dim=-1) / rv_safe.view(1, C)
    energy = parallel + orthogonal
    if bool(normalize_by_dim):
        energy = energy / float(max(D, 1))

    logdet_penalty = torch.zeros((C,), device=device, dtype=dtype)
    if bool(use_logdet_energy) and float(logdet_energy_weight) > 0.0:
        active_logdet = (eig_safe.log() * rank_mask).sum(dim=1)
        residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=dtype)
        logdet_penalty = active_logdet + residual_dims * rv_safe.log()
        if bool(logdet_normalize_by_dim):
            logdet_penalty = logdet_penalty / float(max(D, 1))
        if bool(center_logdet_energy):
            logdet_penalty = logdet_penalty - logdet_penalty.mean().detach()
        energy = energy + float(logdet_energy_weight) * logdet_penalty.view(1, C)

    reliability_penalty = torch.zeros((C,), device=device, dtype=dtype)
    if reliability is not None and torch.is_tensor(reliability) and float(reliability_energy_weight) > 0.0:
        rel = torch.nan_to_num(reliability.to(device=device, dtype=dtype).flatten(), nan=float(reliability_min_clamp), posinf=1.0, neginf=float(reliability_min_clamp))
        if rel.numel() != C:
            raise ValueError(f"reliability must have C={C} entries, got {rel.numel()}")
        rel = rel.clamp(float(reliability_min_clamp), 1.0)
        reliability_penalty = -rel.log()
        reliability_penalty = reliability_penalty - reliability_penalty.mean().detach()
        energy = energy + float(reliability_energy_weight) * reliability_penalty.view(1, C)

    # PG-RGA main path does not add spectral classifier energy.  The arguments
    # are accepted for compatibility; if someone tries to activate it, return a
    # zero part unless both weight and physical flag are explicit.
    spectral_energy = torch.zeros_like(energy)
    if bool(use_spectral_residual_energy) and float(spectral_energy_weight) > 0.0:
        if bool(spectral_require_physical_summary) and not bool(spectral_summary_is_physical):
            spectral_energy = torch.zeros_like(energy)
        else:
            # Deliberately zero in the strict PG-RGA loss module: spectral/band
            # descriptors shape the bank; they do not become inference energy.
            spectral_energy = torch.zeros_like(energy)

    energy = torch.nan_to_num(energy + spectral_energy, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0)

    if not return_parts:
        return energy
    return {
        "energy": energy,
        "feature_energy": energy,
        "parallel": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
        "orthogonal": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
        "parallel_energy": torch.nan_to_num(parallel, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
        "residual_energy": torch.nan_to_num(orthogonal, nan=float(invalid_class_energy), posinf=float(invalid_class_energy), neginf=0.0),
        "logdet_penalty": logdet_penalty,
        "reliability_penalty": reliability_penalty,
        "spectral_energy": spectral_energy,
        "active_ranks": ar,
        "rank_mask": rank_mask,
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
    # GICS
    gics_weight: float = 0.20,
    gics_temperature: float = 0.07,
    # PGR
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
    # spectral shape
    spectral_shape_weight: float = 0.05,
    max_spectral_shape_similarity: float = 0.75,
    spectral_shape_risk_weight: float = 1.0,
    require_physical_summary: bool = True,
    return_parts: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Single active base-phase regularizer.

    BasePhaseTrainer computes balanced CE separately, so ce_weight defaults to
    0.0.  The returned 'total' remains differentiable and must be used for SRPGR.
    """
    if features is None or not torch.is_tensor(features) or features.numel() == 0:
        z = safe_zero_like(features)
        out = {
            "total": z, "ce": z, "base_geometry": z,
            "base_gics": z, "base_gics_anchors": z, "base_gics_pos": z,
            "base_pgr": z, "base_compact": z, "base_center": z,
            "base_subspace": z, "base_band": z, "base_volume": z,
            "base_spectral_shape": z, "base_spectral_shape_raw": z,
            "base_spectral_shape_mean_similarity": z, "base_spectral_shape_pair_count": z,
            "base_spectral_shape_active": z,
        }
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
        if logits.dim() != 2 or logits.size(0) != labels.numel():
            raise ValueError(f"logits must be [B,C] aligned with labels, got {tuple(logits.shape)}")
        ce = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))

    # If physical spectra are present, they are the correct band reserve input.
    # This avoids PCA-30 band profiles competing with raw-200 spectral descriptors.
    band_for_pgr = band_summary
    if spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0 and bool(spectral_summary_is_physical):
        band_for_pgr = spectral_summary

    gics = base_geometry_involved_contrastive_loss(
        features,
        labels,
        key_features=key_features,
        weight=float(gics_weight),
        temperature=float(gics_temperature),
        return_parts=True,
    )

    pgr = prospective_geometry_reserve_loss(
        features,
        labels,
        band_summary=band_for_pgr,
        weight=float(pgr_weight),
        compact_weight=float(pgr_compact_weight),
        center_weight=float(pgr_center_weight),
        subspace_weight=float(pgr_subspace_weight),
        band_weight=float(pgr_band_weight),
        volume_weight=float(pgr_volume_weight),
        center_margin=float(pgr_center_margin),
        min_class_samples=int(min_class_samples),
        subspace_min_samples=int(subspace_min_samples),
        subspace_rank=int(subspace_rank),
        max_band_similarity=float(pgr_max_band_similarity),
        max_class_variance=float(pgr_max_class_variance),
        min_class_variance=float(pgr_min_class_variance),
        max_subspace_overlap=float(kwargs.get("subspace_overlap_max", pgr_max_subspace_overlap)),
        return_parts=True,
    )

    shape_raw = spectral_shape_discrimination_loss(
        spectral_summary,
        labels,
        features=features,
        spectral_summary_is_physical=bool(spectral_summary_is_physical),
        require_physical_summary=bool(require_physical_summary),
        min_samples=int(min_class_samples),
        max_shape_similarity=float(max_spectral_shape_similarity),
        risk_center_margin=float(pgr_center_margin),
        risk_weight=float(spectral_shape_risk_weight),
        return_parts=True,
    )
    shape_total = float(spectral_shape_weight) * _scalar(shape_raw.get("total", safe_zero_like(features)), features)

    gics_total = _scalar(gics.get("total", safe_zero_like(features)), features)
    pgr_total = _scalar(pgr.get("total", safe_zero_like(features)), features)
    base_geometry = float(base_geometry_weight) * (gics_total + pgr_total + shape_total)
    total = float(ce_weight) * ce + base_geometry

    if not return_parts:
        return total

    spectral_active = bool(spectral_summary_is_physical) and spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0

    return {
        "total": total,
        "ce": ce.detach(),
        # Keep differentiable value for code that still uses this key; logs can detach later.
        "base_geometry": base_geometry,

        "base_gics": _scalar(gics.get("gics", safe_zero_like(features)), features).detach(),
        "base_gics_weighted": gics_total.detach(),
        "base_gics_anchors": _scalar(gics.get("valid_anchors", safe_zero_like(features)), features).detach(),
        "base_gics_pos": _scalar(gics.get("mean_positive_count", safe_zero_like(features)), features).detach(),

        "base_pgr": _scalar(pgr.get("pgr", safe_zero_like(features)), features).detach(),
        "base_pgr_weighted": pgr_total.detach(),
        "base_compact": _scalar(pgr.get("compact", safe_zero_like(features)), features).detach(),
        "base_center": _scalar(pgr.get("center", safe_zero_like(features)), features).detach(),
        "base_subspace": _scalar(pgr.get("subspace", safe_zero_like(features)), features).detach(),
        "base_band": _scalar(pgr.get("band", safe_zero_like(features)), features).detach(),
        "base_volume": _scalar(pgr.get("volume", safe_zero_like(features)), features).detach(),

        "base_spectral_shape": shape_total.detach(),
        "base_spectral_shape_raw": _scalar(shape_raw.get("total", safe_zero_like(features)), features).detach(),
        "base_spectral_shape_mean_similarity": _scalar(shape_raw.get("mean_similarity", safe_zero_like(features)), features).detach(),
        "base_spectral_shape_max_similarity": _scalar(shape_raw.get("max_similarity", safe_zero_like(features)), features).detach(),
        "base_spectral_shape_pair_count": _scalar(shape_raw.get("pair_count", safe_zero_like(features)), features).detach(),
        "base_spectral_shape_active": torch.tensor(float(spectral_active), device=features.device, dtype=features.dtype),

        "base_pgr_valid_class_count": _scalar(pgr.get("valid_class_count", safe_zero_like(features)), features).detach(),
        "base_pgr_subspace_pair_count": _scalar(pgr.get("subspace_pair_count", safe_zero_like(features)), features).detach(),
        "base_pgr_band_pair_count": _scalar(pgr.get("band_pair_count", safe_zero_like(features)), features).detach(),
        "base_pgr_volume_factor": _scalar(pgr.get("volume_factor", safe_zero_like(features)), features).detach(),
        "base_pgr_subspace_max_overlap": _scalar(pgr.get("subspace_max_overlap", safe_zero_like(features)), features).detach(),
        "base_pgr_band_max_similarity": _scalar(pgr.get("band_max_similarity", safe_zero_like(features)), features).detach(),
    }


def unified_spectral_geometry_loss(
    *,
    phase: str,
    labels: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    key_features: Optional[torch.Tensor] = None,
    band_summary: Optional[torch.Tensor] = None,
    spectral_summary: Optional[torch.Tensor] = None,
    spectral_summary_is_physical: bool = False,
    return_parts: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor] | torch.Tensor:
    """Public loss entry used by BasePhaseTrainer.

    Incremental training should use explicit geometry_energy_matrix + margin
    losses, not a monolithic hidden loss stack.
    """
    p = str(phase).strip().lower()
    if p not in {"base", "phase0", "phase_0", "0"}:
        raise RuntimeError(
            "unified_spectral_geometry_loss currently owns only the base phase. "
            "For incremental PG-RGA, use geometry_energy_matrix(), "
            "geometry_energy_margin_loss(), and old_new_invasion_loss()."
        )
    return base_geometry_preparation_loss(
        logits=logits,
        features=features,
        labels=labels,
        key_features=key_features,
        band_summary=band_summary,
        spectral_summary=spectral_summary,
        spectral_summary_is_physical=spectral_summary_is_physical,
        return_parts=return_parts,
        **kwargs,
    )


class UnifiedSpectralGeometryLoss:
    """Thin callable wrapper for compatibility with older code."""

    def __init__(self, **defaults: Any) -> None:
        self.defaults = dict(defaults)

    def __call__(self, **kwargs: Any):
        merged = dict(self.defaults)
        merged.update(kwargs)
        return unified_spectral_geometry_loss(**merged)


# -----------------------------------------------------------------------------
# Incremental margin losses used by PG-RGA
# -----------------------------------------------------------------------------

def geometry_energy_margin_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del valid_mask
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return safe_zero_like(labels if torch.is_tensor(labels) else None)
    if energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
    y = labels.to(device=energy.device).long().flatten()
    if y.numel() != energy.size(0):
        raise RuntimeError("labels/energy batch mismatch")
    if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
        raise RuntimeError("labels outside local energy range")
    true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
    true_mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
    nearest_wrong = energy.masked_fill(true_mask, float("inf")).min(dim=1).values
    loss = F.relu(true_e + float(margin) - nearest_wrong)
    finite = torch.isfinite(loss)
    return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0


def old_new_invasion_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    old_class_count: int,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del valid_mask
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return safe_zero_like(labels if torch.is_tensor(labels) else None)
    if energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
    C = int(energy.size(1))
    old = int(max(0, min(int(old_class_count), C)))
    if old <= 0 or old >= C:
        return energy.sum() * 0.0
    y = labels.to(device=energy.device).long().flatten()
    if y.numel() != energy.size(0):
        raise RuntimeError("labels/energy batch mismatch")
    if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= C):
        raise RuntimeError("labels outside local energy range")
    true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
    old_min = energy[:, :old].min(dim=1).values
    new_min = energy[:, old:].min(dim=1).values
    is_old = y < old
    opposite = torch.where(is_old, new_min, old_min)
    loss = F.relu(true_e + float(margin) - opposite)
    finite = torch.isfinite(loss)
    return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0


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
        "sample_boundary_geometry_features is not part of PG-RGA main path. "
        "Use GeometryBank synthetic replay, not boundary replay."
    )












# from __future__ import annotations

# from typing import Any, Dict, Iterable, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -----------------------------------------------------------------------------
# # Basic utilities
# # -----------------------------------------------------------------------------

# def safe_zero_like(
#     ref: Optional[torch.Tensor] = None,
#     *,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> torch.Tensor:
#     if isinstance(ref, torch.Tensor):
#         return ref.sum() * 0.0
#     return torch.tensor(
#         0.0,
#         device=device if device is not None else torch.device("cpu"),
#         dtype=dtype if dtype is not None else torch.float32,
#     )


# def _require_finite_tensor(x: torch.Tensor, name: str) -> None:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     if not torch.isfinite(x).all():
#         bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
#         raise ValueError(f"{name}: tensor contains {bad} NaN/Inf values.")


# def _validate_bank_tensors(
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     *,
#     name: str = "geometry",
# ) -> None:
#     if means is None or bases is None or variances is None:
#         raise ValueError(f"{name}: means, bases, and variances must not be None.")
#     if means.dim() != 2:
#         raise ValueError(f"{name}: means must be [C,D], got {tuple(means.shape)}")
#     if bases.dim() != 3:
#         raise ValueError(f"{name}: bases must be [C,D,R], got {tuple(bases.shape)}")
#     if variances.dim() != 2:
#         raise ValueError(f"{name}: variances must be [C,R+1], got {tuple(variances.shape)}")
#     if means.size(0) != bases.size(0) or means.size(0) != variances.size(0):
#         raise ValueError(
#             f"{name}: class-count mismatch: means={means.size(0)}, "
#             f"bases={bases.size(0)}, variances={variances.size(0)}"
#         )
#     if means.size(1) != bases.size(1):
#         raise ValueError(f"{name}: feature-dim mismatch: means={means.size(1)}, bases={bases.size(1)}")
#     if bases.size(2) + 1 != variances.size(1):
#         raise ValueError(
#             f"{name}: rank/variance mismatch: rank={bases.size(2)}, "
#             f"variance dim={variances.size(1)}"
#         )


# def _valid_class_mask_from_counts(
#     sample_counts: Optional[torch.Tensor],
#     num_classes: int,
#     device: torch.device,
# ) -> torch.Tensor:
#     if sample_counts is None or not torch.is_tensor(sample_counts):
#         raise RuntimeError(
#             f"Geometry losses require sample_counts [C={num_classes}]. "
#             "Missing counts would treat unbuilt capacity rows as real classes."
#         )
#     if sample_counts.numel() != int(num_classes):
#         raise RuntimeError(
#             f"sample_counts width mismatch: expected C={int(num_classes)}, got {int(sample_counts.numel())}."
#         )
#     counts = sample_counts.to(device=device).flatten()
#     if not torch.isfinite(counts).all():
#         raise RuntimeError("sample_counts contains NaN/Inf values.")
#     return counts > 0


# def _apply_invalid_class_mask(
#     energy: torch.Tensor,
#     sample_counts: Optional[torch.Tensor],
#     invalid_class_energy: float = 1e6,
# ) -> torch.Tensor:
#     valid = _valid_class_mask_from_counts(sample_counts, energy.size(1), energy.device)
#     if bool(valid.all().item()):
#         return energy
#     return energy.masked_fill(~valid.view(1, -1), float(invalid_class_energy))


# def _active_rank_mask(
#     active_ranks: Optional[torch.Tensor],
#     num_classes: int,
#     rank: int,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if active_ranks is None or not torch.is_tensor(active_ranks) or active_ranks.numel() != num_classes:
#         ar = torch.full((num_classes,), rank, device=device, dtype=torch.long)
#     else:
#         ar = active_ranks.to(device=device).long().flatten().clamp(min=0, max=rank)
#     mask = torch.arange(rank, device=device).view(1, rank) < ar.view(num_classes, 1)
#     return mask.to(dtype=dtype), ar


# def _low_rank_logdet_penalty(
#     eigvals: torch.Tensor,
#     resvars: torch.Tensor,
#     rank_mask: torch.Tensor,
#     active_ranks: torch.Tensor,
#     feature_dim: int,
#     valid_mask: torch.Tensor,
#     *,
#     variance_floor: float = 1e-4,
#     normalize_by_dim: bool = True,
#     center: bool = True,
#     invalid_class_energy: float = 1e6,
# ) -> torch.Tensor:
#     """Approximate class covariance volume for low-rank residual geometry.

#     For class c:
#         log |Sigma_c| ~= sum_{i<=r_c} log(lambda_ci)
#                         + (D - r_c) log(sigma_c^2)

#     This is the same volume correction used by the classifier.  It prevents
#     broad descriptors from receiving artificially low Mahalanobis energy and
#     stealing old samples during incremental phases.
#     """
#     if eigvals is None or eigvals.numel() == 0:
#         return torch.empty((0,), device=resvars.device, dtype=resvars.dtype)
#     D = int(max(feature_dim, 1))
#     log_eig = eigvals.clamp_min(float(variance_floor)).log()
#     log_res = resvars.clamp_min(float(variance_floor)).log()
#     active_logdet = (log_eig * rank_mask).sum(dim=1)
#     residual_dims = (D - active_ranks.to(device=eigvals.device).long().clamp(min=0, max=D)).to(dtype=eigvals.dtype)
#     logdet = active_logdet + residual_dims * log_res
#     if bool(normalize_by_dim):
#         logdet = logdet / float(D)
#     logdet = torch.nan_to_num(
#         logdet,
#         nan=0.0,
#         posinf=float(invalid_class_energy),
#         neginf=-float(invalid_class_energy),
#     )
#     if bool(center) and torch.is_tensor(valid_mask) and bool(valid_mask.any().item()):
#         logdet = logdet - logdet[valid_mask].mean().detach()
#     return logdet


# def _class_centers(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     min_samples: int = 2,
#     normalize_centers: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
#         device = features.device if torch.is_tensor(features) else torch.device("cpu")
#         dtype = features.dtype if torch.is_tensor(features) else torch.float32
#         return (
#             torch.empty(0, 0, device=device, dtype=dtype),
#             torch.empty(0, device=device, dtype=torch.long),
#             torch.empty(0, device=device, dtype=dtype),
#         )
#     if features.dim() != 2:
#         raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
#     y = labels.to(device=features.device).long().flatten()
#     if y.numel() != features.size(0):
#         raise ValueError(f"labels/features mismatch: {y.numel()} vs {features.size(0)}")

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



# def _normalize_band_summary(band_summary: torch.Tensor) -> torch.Tensor:
#     """
#     Convert band summaries to comparable non-negative class signatures.

#     Dataset-robust behavior:
#         - signed/logit-like rows are converted with softmax;
#         - non-negative rows are sum-normalized;
#         - all-zero/near-zero rows fall back to a uniform distribution.

#     This avoids NaNs and prevents a mini-batch with weak PCA/band responses from
#     creating fake band evidence.
#     """
#     if band_summary is None or not torch.is_tensor(band_summary):
#         raise TypeError("band_summary must be a tensor.")
#     if band_summary.dim() != 2:
#         raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")
#     b = torch.nan_to_num(band_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     if b.size(1) <= 0:
#         return b
#     if bool((b < 0).any().item()):
#         return F.softmax(b, dim=1)
#     b = b.clamp_min(0.0)
#     denom = b.sum(dim=1, keepdim=True)
#     uniform = torch.full_like(b, 1.0 / max(int(b.size(1)), 1))
#     return torch.where(denom > 1e-8, b / denom.clamp_min(1e-8), uniform)


# # -----------------------------------------------------------------------------
# # Base phase: geometry-compatible representation shaping
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
#     **_: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """
#     Geometry-Involved Class Separation (GICS), cleaned.

#     Base-phase only. It shapes the same canonical projected z-space that later
#     populates the GeometryBank. If a key view is provided, the diagonal means
#     query-vs-key positive. If no key view is provided, self-pairs are excluded so
#     the loss does not become a trivial identity objective.
#     """
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         if return_parts:
#             return {"total": z, "gics": z, "loss": z, "weighted_gics": z, "valid_anchors": z, "mean_positive_count": z}
#         return z
#     if features.dim() != 2:
#         raise ValueError(f"GICS expects projected features [B,D], got {tuple(features.shape)}")

#     zq = features
#     explicit_key = key_features is not None and torch.is_tensor(key_features) and key_features.numel() > 0
#     zk = key_features if explicit_key else features
#     if zk.dim() != 2:
#         raise ValueError(f"key_features must be [B,D], got {tuple(zk.shape)}")
#     if zk.size(0) != zq.size(0):
#         raise ValueError(f"key_features batch mismatch: {zk.size(0)} vs {zq.size(0)}")
#     if detach_key:
#         zk = zk.detach()

#     y = labels.to(device=zq.device).long().flatten()
#     if y.numel() != zq.size(0):
#         raise ValueError(f"GICS labels/features mismatch: {y.numel()} vs {zq.size(0)}")

#     q = F.normalize(zq, dim=1, eps=1e-6) if normalize else zq
#     k = F.normalize(zk.to(device=zq.device, dtype=zq.dtype), dim=1, eps=1e-6) if normalize else zk.to(device=zq.device, dtype=zq.dtype)
#     logits = q @ k.t() / max(float(temperature), 1e-6)
#     logits = logits - logits.max(dim=1, keepdim=True).values.detach()

#     B = zq.size(0)
#     target = y.view(-1, 1).eq(y.view(1, -1)) if same_class_positive else torch.eye(B, device=zq.device, dtype=torch.bool)
#     diag = torch.eye(B, device=zq.device, dtype=torch.bool)
#     if explicit_key:
#         target = target | diag
#     else:
#         target = target & (~diag)

#     pos_count = target.float().sum(dim=1)
#     valid = pos_count > 0
#     if not bool(valid.any().item()):
#         gics = features.sum() * 0.0
#     else:
#         exp_logits = torch.exp(logits)
#         if not explicit_key:
#             exp_logits = exp_logits.masked_fill(diag, 0.0)
#         denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
#         log_prob = logits - denom.log()
#         per = -(target.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
#         per = per[valid]
#         yv = y[valid]
#         if per.numel() == 0:
#             gics = features.sum() * 0.0
#         elif class_balanced:
#             terms = [per[yv == c].mean() for c in torch.unique(yv) if bool((yv == c).any().item())]
#             gics = torch.stack(terms).mean() if terms else features.sum() * 0.0
#         else:
#             gics = per.mean()

#     total = float(weight) * gics
#     if return_parts:
#         return {
#             "total": total,
#             "gics": gics.detach(),
#             "loss": gics.detach(),
#             "weighted_gics": total.detach(),
#             "valid_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
#             "num_anchors": torch.tensor(float(valid.sum().item()), device=features.device, dtype=features.dtype),
#             "mean_positive_count": pos_count[valid].float().mean().detach() if bool(valid.any().item()) else features.sum().detach() * 0.0,
#         }
#     return total

# def base_fcs_geometry_contrastive_loss(*args, **kwargs):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_geometry_involved_contrastive_separation_loss(*args, **kwargs):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_supervised_contrastive_loss(*args, **kwargs):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def base_hsi_supervised_contrastive_loss(*args, **kwargs):
#     return base_geometry_involved_contrastive_loss(*args, **kwargs)


# def _pairwise_center_margin_loss(centers: torch.Tensor, margin: float) -> torch.Tensor:
#     if centers is None or centers.numel() == 0 or centers.size(0) < 2:
#         return safe_zero_like(centers)
#     dist = torch.cdist(centers, centers, p=2)
#     eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
#     pair = dist[~eye]
#     return F.relu(float(margin) - pair).pow(2).mean() if pair.numel() > 0 else centers.sum() * 0.0



# def _batch_subspace_overlap_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     rank: int = 3,
#     min_samples: int = 6,
#     normalize: bool = True,
#     return_parts: bool = False,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     if features is None or features.numel() == 0:
#         z = safe_zero_like(features)
#         if return_parts:
#             return {"total": z, "pair_count": z, "valid_class_count": z}
#         return z
#     if features.dim() != 2:
#         raise ValueError(f"subspace loss expects [B,D], got {tuple(features.shape)}")
#     y = labels.to(device=features.device).long().flatten()
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
#         pair_count = 0
#     else:
#         loss = features.sum() * 0.0
#         pair_count = 0
#         for i in range(len(bases)):
#             for j in range(i + 1, len(bases)):
#                 Ui, Uj = bases[i], bases[j]
#                 denom = float(max(min(Ui.size(1), Uj.size(1)), 1))
#                 loss = loss + (Ui.t() @ Uj).pow(2).sum() / denom
#                 pair_count += 1
#         loss = loss / max(pair_count, 1)
#     if return_parts:
#         return {
#             "total": loss,
#             "pair_count": torch.tensor(float(pair_count), device=features.device, dtype=features.dtype),
#             "valid_class_count": torch.tensor(float(len(bases)), device=features.device, dtype=features.dtype),
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
#     """
#     HSI-specific base loss: prevent high-risk class pairs from using identical
#     band signatures.

#     This is not a second classifier branch. It shapes the base feature/band space
#     so the later GeometryBank's band_signature is class-discriminative. The
#     normalization is robust to PCA datasets where band summaries may be signed or
#     near-zero.
#     """
#     ref = features if isinstance(features, torch.Tensor) else labels
#     if band_summary is None or not torch.is_tensor(band_summary) or band_summary.numel() == 0:
#         z = safe_zero_like(ref)
#         if return_parts:
#             return {"total": z, "band": z, "pair_count": z, "valid_class_count": z, "mean_similarity": z}
#         return z
#     if band_summary.dim() != 2:
#         raise ValueError(f"band_summary must be [B,S], got {tuple(band_summary.shape)}")
#     y = labels.to(device=band_summary.device).long().flatten()
#     if y.numel() != band_summary.size(0):
#         raise ValueError(f"band labels/batch mismatch: {y.numel()} vs {band_summary.size(0)}")

#     b = _normalize_band_summary(band_summary)
#     b_centers, class_ids, _ = _class_centers(b, y, min_samples=min_samples, normalize_centers=True)
#     if b_centers.numel() == 0 or b_centers.size(0) < 2:
#         z = band_summary.sum() * 0.0
#         if return_parts:
#             return {
#                 "total": z,
#                 "band": z,
#                 "pair_count": z,
#                 "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
#                 "mean_similarity": z,
#             }
#         return z

#     sim = (b_centers @ b_centers.t()).clamp(0.0, 1.0)
#     eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
#     pair_sim = sim[~eye]

#     if features is not None and torch.is_tensor(features) and features.numel() > 0:
#         z = F.normalize(features.to(device=band_summary.device, dtype=band_summary.dtype), dim=1, eps=1e-6)
#         f_centers, f_ids, _ = _class_centers(z, y, min_samples=min_samples, normalize_centers=False)
#         if f_centers.size(0) == b_centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
#             dist = torch.cdist(f_centers, f_centers, p=2)
#             center_risk = F.relu(float(risk_center_margin) - dist) / max(float(risk_center_margin), 1e-6)
#             risk = center_risk[~eye].detach()
#         else:
#             risk = torch.ones_like(pair_sim)
#     else:
#         risk = torch.ones_like(pair_sim)

#     loss_vec = F.relu(pair_sim - float(max_band_similarity)).pow(2) * (1.0 + float(risk_weight) * risk)
#     loss = loss_vec.mean() if loss_vec.numel() > 0 else band_summary.sum() * 0.0
#     if return_parts:
#         return {
#             "total": loss,
#             "band": loss.detach(),
#             "pair_count": torch.tensor(float(loss_vec.numel()), device=band_summary.device, dtype=band_summary.dtype),
#             "valid_class_count": torch.tensor(float(b_centers.size(0)), device=band_summary.device, dtype=band_summary.dtype),
#             "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else band_summary.sum().detach() * 0.0,
#         }
#     return loss


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
#     normalize_features: bool = True,
#     adaptive_component_weights: bool = True,
#     return_parts: bool = True,
#     **_: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """
#     Prospective Geometry Reserve (PGR), dataset-robust.

#     Base-phase only. It prepares the canonical feature space for future HSI class
#     insertion through compactness, center separation, subspace de-overlap,
#     controlled class volume, and band-signature discrimination.

#     ``adaptive_component_weights`` prevents noisy terms from dominating on small
#     HSI batches/classes by activating each term only when the mini-batch contains
#     enough evidence for that term.
#     """
#     if features is None or labels is None or not torch.is_tensor(features) or features.numel() == 0:
#         z0 = safe_zero_like(features)
#         if return_parts:
#             return {
#                 "total": z0, "pgr": z0, "compact": z0, "center": z0, "subspace": z0, "band": z0, "volume": z0,
#                 "valid_class_count": z0, "unique_class_count": z0, "subspace_pair_count": z0, "band_pair_count": z0,
#                 "compact_factor": z0, "center_factor": z0, "subspace_factor": z0, "band_factor": z0, "volume_factor": z0,
#             }
#         return z0
#     if features.dim() != 2:
#         raise ValueError(f"PGR expects features [B,D], got {tuple(features.shape)}")
#     y = labels.to(device=features.device).long().flatten()
#     if y.numel() != features.size(0):
#         raise ValueError(f"PGR labels/features mismatch: {y.numel()} vs {features.size(0)}")

#     z = F.normalize(features, dim=1, eps=1e-6) if normalize_features else features

#     compact_terms = []
#     volume_terms = []
#     for cls in torch.unique(y, sorted=True):
#         m = y == cls
#         if int(m.sum().item()) < int(min_class_samples):
#             continue
#         xc = z[m]
#         var = (xc - xc.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
#         compact_terms.append(var)
#         volume_terms.append(F.relu(var - float(max_class_variance)).pow(2))
#     compact = torch.stack(compact_terms).mean() if compact_terms else features.sum() * 0.0
#     volume = torch.stack(volume_terms).mean() if volume_terms else features.sum() * 0.0
#     valid_class_count = len(compact_terms)
#     unique_class_count = int(torch.unique(y).numel())

#     centers, _, _ = _class_centers(z, y, min_samples=min_class_samples, normalize_centers=False)
#     center = _pairwise_center_margin_loss(centers, center_margin)

#     sub_obj = _batch_subspace_overlap_loss(
#         z,
#         y,
#         rank=subspace_rank,
#         min_samples=subspace_min_samples,
#         normalize=False,
#         return_parts=True,
#     )
#     subspace = sub_obj["total"]
#     subspace_pair_count = sub_obj["pair_count"]

#     band_obj = risk_aware_band_discrimination_loss(
#         band_summary,
#         y,
#         features=z,
#         min_samples=min_class_samples,
#         max_band_similarity=max_band_similarity,
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
#         compact_factor = volume_factor = center_factor = subspace_factor = band_factor = one

#     unweighted = (
#         float(compact_weight) * compact_factor * compact
#         + float(center_weight) * center_factor * center
#         + float(subspace_weight) * subspace_factor * subspace
#         + float(band_weight) * band_factor * band
#         + float(volume_weight) * volume_factor * volume
#     )
#     total = float(weight) * unweighted
#     if return_parts:
#         return {
#             "total": total,
#             "pgr": unweighted.detach(),
#             "compact": compact.detach(),
#             "center": center.detach(),
#             "subspace": subspace.detach(),
#             "band": band.detach(),
#             "volume": volume.detach(),
#             "weighted_pgr": total.detach(),
#             "valid_class_count": torch.tensor(float(valid_class_count), device=features.device, dtype=features.dtype),
#             "unique_class_count": torch.tensor(float(unique_class_count), device=features.device, dtype=features.dtype),
#             "subspace_pair_count": subspace_pair_count.detach(),
#             "band_pair_count": band_pair_count.detach() if torch.is_tensor(band_pair_count) else torch.tensor(float(band_pair_count), device=features.device, dtype=features.dtype),
#             "compact_factor": compact_factor.detach(),
#             "center_factor": center_factor.detach(),
#             "subspace_factor": subspace_factor.detach(),
#             "band_factor": band_factor.detach(),
#             "volume_factor": volume_factor.detach(),
#         }
#     return total

# def base_prospective_geometry_reserve_loss(*args, **kwargs):
#     return prospective_geometry_reserve_loss(*args, **kwargs)


# @torch.no_grad()
# def base_center_overlap_diagnostics(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     normalize: bool = True,
#     min_samples: int = 2,
# ) -> Dict[str, torch.Tensor]:
#     if features is None or labels is None or features.numel() == 0:
#         z = safe_zero_like(features)
#         return {"compact": z, "mean_center_margin": z, "min_center_margin": z, "num_classes": z}
#     z = F.normalize(features, dim=1, eps=1e-6) if normalize else features
#     y = labels.to(device=z.device).long().flatten()
#     centers, _, _ = _class_centers(z, y, min_samples=min_samples)
#     compact_terms = []
#     for cls in torch.unique(y):
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
# def base_gics_diagnostics(features: torch.Tensor, labels: torch.Tensor, **kwargs: object) -> Dict[str, torch.Tensor]:
#     out = base_geometry_involved_contrastive_loss(features, labels, weight=1.0, return_parts=True, **kwargs)
#     return {
#         "gics": out["gics"].detach(),
#         "valid_anchors": out["valid_anchors"].detach(),
#         "positive_pairs": out["mean_positive_count"].detach(),
#     }


# def base_supcon_diagnostics(*args, **kwargs):
#     return base_gics_diagnostics(*args, **kwargs)


# # -----------------------------------------------------------------------------
# # Geometry energy and geometry objectives
# # -----------------------------------------------------------------------------

# def geometry_energy_matrix(
#     features: torch.Tensor,
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     *,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     reliability_energy_weight: float = 0.05,
#     residual_variance_scale: float = 0.75,
#     normalize_by_dim: bool = True,
#     invalid_class_energy: float = 1e6,
#     return_parts: bool = False,
#     var_floor: Optional[float] = None,
#     reliability_penalty: Optional[float] = None,
#     use_logdet_energy: bool = True,
#     logdet_energy_weight: float = 0.05,
#     logdet_normalize_by_dim: bool = True,
#     center_logdet_energy: bool = True,
#     name: str = "geometry",
#     **_: object,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """Covariance-consistent low-rank Gaussian energy.

#     This mirrors ``GeometryEnergyClassifier.geometry_energy``.  Incremental
#     descriptor refinement, replay CE, margin losses, and diagnostics must optimize
#     the same energy used at evaluation time:

#         E = low-rank Mahalanobis + residual Mahalanobis
#             + beta * centered logdet(Sigma_c)
#             + gamma * centered reliability penalty

#     Missing ``sample_counts`` is an error because allocated capacity rows are not
#     scoreable classes.
#     """
#     if var_floor is not None:
#         variance_floor = float(var_floor)
#     if reliability_penalty is not None:
#         reliability_energy_weight = float(reliability_penalty)

#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         device = means.device if torch.is_tensor(means) else torch.device("cpu")
#         dtype = means.dtype if torch.is_tensor(means) else torch.float32
#         z = torch.empty(0, 0, device=device, dtype=dtype)
#         if return_parts:
#             return {
#                 "energy": z,
#                 "parallel": z,
#                 "orthogonal": z,
#                 "logdet_penalty": torch.empty(0, device=device, dtype=dtype),
#                 "reliability_penalty": torch.empty(0, device=device, dtype=dtype),
#                 "active_ranks": torch.empty(0, device=device, dtype=torch.long),
#                 "rank_mask": z,
#                 "valid_mask": torch.empty(0, device=device, dtype=torch.bool),
#             }
#         return z

#     _validate_bank_tensors(means, bases, variances, name=name)
#     if features.dim() != 2:
#         raise ValueError(f"{name}: features must be [B,D], got {tuple(features.shape)}")
#     if features.size(1) != means.size(1):
#         raise ValueError(f"{name}: feature dim mismatch: features={features.size(1)}, means={means.size(1)}")

#     _require_finite_tensor(features, f"{name}.features")
#     _require_finite_tensor(means, f"{name}.means")
#     _require_finite_tensor(bases, f"{name}.bases")
#     _require_finite_tensor(variances, f"{name}.variances")

#     z = features
#     means = torch.nan_to_num(means.to(device=z.device, dtype=z.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#     bases = torch.nan_to_num(bases.to(device=z.device, dtype=z.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#     variances = torch.nan_to_num(
#         variances.to(device=z.device, dtype=z.dtype),
#         nan=float(variance_floor),
#         posinf=float(invalid_class_energy),
#         neginf=float(variance_floor),
#     )

#     _, D = z.shape
#     C, _, R = bases.shape
#     valid = _valid_class_mask_from_counts(sample_counts, C, z.device)
#     rank_mask, ar = _active_rank_mask(active_ranks, C, R, z.device, z.dtype)

#     delta = z.unsqueeze(1) - means.unsqueeze(0)                       # [B,C,D]
#     coeff = torch.einsum("bcd,cdr->bcr", delta, bases)               # [B,C,R]
#     coeff_active = coeff * rank_mask.unsqueeze(0)
#     recon = torch.einsum("bcr,cdr->bcd", coeff_active, bases)
#     residual = delta - recon

#     eigvals = variances[:, :-1].clamp_min(float(variance_floor))
#     resvars = (variances[:, -1] * float(residual_variance_scale)).clamp_min(float(variance_floor))

#     parallel = ((coeff_active.pow(2) / eigvals.unsqueeze(0)) * rank_mask.unsqueeze(0)).sum(dim=-1)
#     orthogonal = residual.pow(2).sum(dim=-1) / resvars.unsqueeze(0)

#     energy = parallel + orthogonal
#     if normalize_by_dim:
#         energy = energy / max(D, 1)

#     logdet_pen = _low_rank_logdet_penalty(
#         eigvals=eigvals,
#         resvars=resvars,
#         rank_mask=rank_mask,
#         active_ranks=ar,
#         feature_dim=D,
#         valid_mask=valid,
#         variance_floor=float(variance_floor),
#         normalize_by_dim=bool(logdet_normalize_by_dim),
#         center=bool(center_logdet_energy),
#         invalid_class_energy=float(invalid_class_energy),
#     )
#     if bool(use_logdet_energy) and float(logdet_energy_weight) > 0.0:
#         energy = energy + float(logdet_energy_weight) * logdet_pen.view(1, C)

#     rel_pen = torch.zeros((C,), device=z.device, dtype=z.dtype)
#     if reliability is not None and torch.is_tensor(reliability) and reliability.numel() == C:
#         rel = torch.nan_to_num(
#             reliability.to(device=z.device, dtype=z.dtype).flatten(),
#             nan=1e-6,
#             posinf=1.0,
#             neginf=1e-6,
#         ).clamp(1e-6, 1.0)
#         rel_pen = -torch.log(rel)
#         if bool(valid.any().item()):
#             rel_pen = rel_pen - rel_pen[valid].mean().detach()
#         energy = energy + float(reliability_energy_weight) * rel_pen.unsqueeze(0)

#     energy = torch.nan_to_num(
#         energy,
#         nan=float(invalid_class_energy),
#         posinf=float(invalid_class_energy),
#         neginf=0.0,
#     ).masked_fill(~valid.view(1, C), float(invalid_class_energy))

#     if return_parts:
#         parallel = torch.nan_to_num(
#             parallel,
#             nan=float(invalid_class_energy),
#             posinf=float(invalid_class_energy),
#             neginf=0.0,
#         ).masked_fill(~valid.view(1, C), float(invalid_class_energy))
#         orthogonal = torch.nan_to_num(
#             orthogonal,
#             nan=float(invalid_class_energy),
#             posinf=float(invalid_class_energy),
#             neginf=0.0,
#         ).masked_fill(~valid.view(1, C), float(invalid_class_energy))
#         return {
#             "energy": energy,
#             "parallel": parallel,
#             "orthogonal": orthogonal,
#             "logdet_penalty": logdet_pen.masked_fill(~valid, 0.0),
#             "reliability_penalty": rel_pen,
#             "active_ranks": ar,
#             "rank_mask": rank_mask,
#             "valid_mask": valid,
#         }
#     return energy


# # Clean compatibility alias. Dual spectral geometry is removed from core.
# def dual_geometry_energy_matrix(
#     features: torch.Tensor,
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     **kwargs: object,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     return geometry_energy_matrix(features, means, bases, variances, **kwargs)


# def geometry_logits_from_energy(energy: torch.Tensor, logit_scale: float = 8.0) -> torch.Tensor:
#     if energy.dim() != 2:
#         raise ValueError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     e = torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=0.0)
#     row_min = e.min(dim=1, keepdim=True).values
#     return -float(logit_scale) * (e - row_min)


# def relative_energy_logits(
#     energy: torch.Tensor,
#     *,
#     sample_counts: Optional[torch.Tensor] = None,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     logit_scale: float = 8.0,
#     min_energy_scale: float = 1.0,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Convert energies to logits without row-wise standardization.

#     ``min_energy_scale`` is accepted for backward compatibility but is not used
#     to divide per-row energies.  Per-sample energy standardization hides old/new
#     calibration errors and was one of the reasons incremental phases looked
#     stable during optimization but collapsed at evaluation.
#     """
#     del min_energy_scale
#     if energy.dim() != 2:
#         raise ValueError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     B, C = energy.shape
#     del B
#     if valid_class_mask is not None:
#         valid = valid_class_mask.to(device=energy.device, dtype=torch.bool).flatten()
#         if valid.numel() != C:
#             raise ValueError(f"valid_class_mask must have C={C} entries, got {valid.numel()}")
#     else:
#         valid = _valid_class_mask_from_counts(sample_counts, C, energy.device)
#     if int(valid.sum().item()) <= 0:
#         raise RuntimeError("No valid GeometryBank rows available.")

#     e = torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=0.0)
#     masked = e.masked_fill(~valid.view(1, C), float("inf"))
#     row_min = masked.min(dim=1, keepdim=True).values
#     rel = torch.nan_to_num(masked - row_min, nan=0.0, posinf=1e6, neginf=0.0)

#     logits = -float(logit_scale) * rel
#     logits = logits.masked_fill(~valid.view(1, C), -1e9)
#     return torch.nan_to_num(logits, nan=-1e9, posinf=1e4, neginf=-1e9), rel, valid


# def valid_energy_objective(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     sample_counts: Optional[torch.Tensor] = None,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     logit_scale: float = 8.0,
#     min_energy_scale: float = 1.0,
#     rank_margin: float = 0.25,
#     label_smoothing: float = 0.0,
# ) -> Dict[str, torch.Tensor]:
#     if energy.dim() != 2:
#         raise ValueError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     labels = labels.to(device=energy.device).long().flatten()
#     B, C = energy.shape
#     if labels.numel() != B:
#         raise ValueError(f"labels size mismatch: labels={labels.numel()}, batch={B}")
#     if int(labels.min().item()) < 0 or int(labels.max().item()) >= C:
#         raise ValueError(f"label range incompatible with C={C}")

#     logits, norm_rel, valid = relative_energy_logits(
#         energy,
#         sample_counts=sample_counts,
#         valid_class_mask=valid_class_mask,
#         logit_scale=logit_scale,
#         min_energy_scale=min_energy_scale,
#     )
#     true_valid = valid.gather(0, labels)
#     if not bool(true_valid.all().item()):
#         bad = labels[~true_valid].detach().cpu().unique().tolist()
#         raise RuntimeError(f"Labels include unbuilt GeometryBank classes: {bad}")

#     ce = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
#     true_mask = torch.zeros((B, C), device=energy.device, dtype=torch.bool).scatter(1, labels.view(-1, 1), True)
#     neg_mask = valid.view(1, C).expand(B, C) & (~true_mask)
#     own = energy.gather(1, labels.view(-1, 1)).squeeze(1)
#     if bool(neg_mask.any(dim=1).all().item()):
#         nearest = energy.masked_fill(~neg_mask, float("inf")).min(dim=1).values
#         # Do not normalize the rank loss by each row's energy std.  Absolute
#         # old/new energy scale is meaningful after logdet/reliability calibration.
#         rank = F.softplus(own - nearest + float(rank_margin)).mean()
#         violation = ((own + float(rank_margin)) >= nearest).float().mean().detach()
#         raw_gap = (nearest - own).mean().detach()
#     else:
#         nearest = own.detach() * 0.0
#         rank = energy.sum() * 0.0
#         violation = energy.sum().detach() * 0.0
#         raw_gap = energy.sum().detach() * 0.0
#     compact = torch.log1p(own.clamp_min(0.0)).mean()
#     return {
#         "ce": ce,
#         "rank": rank,
#         "compact": compact,
#         "logits": logits,
#         "norm_rel_energy": norm_rel,
#         "valid_mask": valid,
#         "own_energy": own.detach().mean(),
#         "nearest_energy": nearest.detach().mean(),
#         "raw_gap": raw_gap,
#         "violation_rate": violation,
#     }


# @torch.no_grad()
# def energy_margin_statistics(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     sample_counts: Optional[torch.Tensor] = None,
# ) -> Dict[str, torch.Tensor]:
#     if energy is None or energy.numel() == 0:
#         z = safe_zero_like(energy)
#         return {"mean_margin": z, "min_margin": z, "violation_rate": z, "accuracy": z}
#     labels = labels.to(device=energy.device).long().flatten()
#     valid = _valid_class_mask_from_counts(sample_counts, energy.size(1), energy.device)
#     masked = energy.masked_fill(~valid.view(1, -1), float("inf"))
#     true_e = masked.gather(1, labels.view(-1, 1)).squeeze(1)
#     true_mask = torch.zeros_like(masked, dtype=torch.bool).scatter(1, labels.view(-1, 1), True)
#     wrong = masked.masked_fill(true_mask, float("inf"))
#     nearest_wrong = wrong.min(dim=1).values
#     margin = nearest_wrong - true_e
#     pred = masked.argmin(dim=1)
#     return {
#         "mean_margin": margin.mean(),
#         "min_margin": margin.min(),
#         "violation_rate": (margin <= 0).float().mean(),
#         "accuracy": (pred == labels).float().mean(),
#     }


# # -----------------------------------------------------------------------------
# # Incremental overlap / invasion objectives
# # -----------------------------------------------------------------------------

# def old_new_geometry_risk(
#     old_means: torch.Tensor,
#     old_bases: torch.Tensor,
#     new_means: torch.Tensor,
#     new_bases: torch.Tensor,
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     old_reliability: Optional[torch.Tensor] = None,
#     center_weight: float = 0.50,
#     subspace_weight: float = 0.50,
# ) -> torch.Tensor:
#     if old_means is None or new_means is None or old_means.numel() == 0 or new_means.numel() == 0:
#         device = old_means.device if torch.is_tensor(old_means) else torch.device("cpu")
#         dtype = old_means.dtype if torch.is_tensor(old_means) else torch.float32
#         return torch.empty(0, 0, device=device, dtype=dtype)
#     old_means = old_means.to(device=new_means.device, dtype=new_means.dtype)
#     old_bases = old_bases.to(device=new_means.device, dtype=new_means.dtype)
#     new_bases = new_bases.to(device=new_means.device, dtype=new_means.dtype)
#     O, D = old_means.shape
#     N = new_means.size(0)
#     dist = torch.cdist(old_means, new_means, p=2) / max(D ** 0.5, 1.0)
#     center_sim = torch.exp(-dist)
#     old_R = old_bases.size(2)
#     new_R = new_bases.size(2)
#     old_ar = torch.full((O,), old_R, device=new_means.device, dtype=torch.long) if old_active_ranks is None else old_active_ranks.to(new_means.device).long().clamp(0, old_R)
#     new_ar = torch.full((N,), new_R, device=new_means.device, dtype=torch.long) if new_active_ranks is None else new_active_ranks.to(new_means.device).long().clamp(0, new_R)
#     overlap = torch.zeros(O, N, device=new_means.device, dtype=new_means.dtype)
#     for o in range(O):
#         ro = int(old_ar[o].item())
#         if ro <= 0:
#             continue
#         Uo = old_bases[o, :, :ro]
#         for n in range(N):
#             rn = int(new_ar[n].item())
#             if rn <= 0:
#                 continue
#             Un = new_bases[n, :, :rn]
#             denom = float(max(min(ro, rn), 1))
#             overlap[o, n] = (Uo.t() @ Un).pow(2).sum() / denom
#     risk = float(center_weight) * center_sim + float(subspace_weight) * overlap
#     if old_reliability is not None and torch.is_tensor(old_reliability) and old_reliability.numel() == O:
#         rel = old_reliability.to(device=new_means.device, dtype=new_means.dtype).flatten().clamp(0.05, 1.0)
#         risk = risk * (2.0 - rel).view(-1, 1).clamp(1.0, 1.95)
#     return risk.clamp_min(0.0)



# def descriptor_subspace_collision_loss(
#     old_bases: Optional[torch.Tensor],
#     new_bases: Optional[torch.Tensor],
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     target_overlap: float = 0.35,
#     reliability: Optional[torch.Tensor] = None,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Penalize new descriptors that reuse old low-rank tangent directions."""
#     ref = new_bases if torch.is_tensor(new_bases) else old_bases
#     if old_bases is None or new_bases is None or not torch.is_tensor(old_bases) or not torch.is_tensor(new_bases):
#         z = safe_zero_like(ref)
#         return {"total": z, "mean_overlap": z, "max_overlap": z, "pair_count": z} if return_parts else z
#     if old_bases.numel() == 0 or new_bases.numel() == 0:
#         z = safe_zero_like(ref)
#         return {"total": z, "mean_overlap": z, "max_overlap": z, "pair_count": z} if return_parts else z
#     if old_bases.dim() != 3 or new_bases.dim() != 3:
#         raise ValueError("old_bases and new_bases must be [C,D,R].")
#     old = old_bases.to(device=new_bases.device, dtype=new_bases.dtype)
#     new = new_bases
#     O, D, Ro = old.shape
#     N, Dn, Rn = new.shape
#     if D != Dn:
#         raise ValueError(f"basis dimension mismatch: old D={D}, new D={Dn}")
#     old_ar = torch.full((O,), Ro, device=new.device, dtype=torch.long) if old_active_ranks is None else old_active_ranks.to(new.device).long().clamp(0, Ro)
#     new_ar = torch.full((N,), Rn, device=new.device, dtype=torch.long) if new_active_ranks is None else new_active_ranks.to(new.device).long().clamp(0, Rn)
#     losses, overlaps = [], []
#     for o in range(O):
#         ro = int(old_ar[o].detach().cpu().item())
#         if ro <= 0:
#             continue
#         Uo = old[o, :, :ro]
#         for n in range(N):
#             rn = int(new_ar[n].detach().cpu().item())
#             if rn <= 0:
#                 continue
#             Un = new[n, :, :rn]
#             denom = float(max(min(ro, rn), 1))
#             s = (Uo.t() @ Un).pow(2).sum() / denom
#             w = 1.0
#             if reliability is not None and torch.is_tensor(reliability) and reliability.numel() > o:
#                 rho = reliability.to(device=new.device, dtype=new.dtype).flatten()[o].clamp(0.05, 1.0)
#                 w = float((2.0 - rho).detach().cpu().item())
#             overlaps.append(s)
#             losses.append(float(w) * F.relu(s - float(target_overlap)).pow(2))
#     if not losses:
#         z = safe_zero_like(ref)
#         return {"total": z, "mean_overlap": z, "max_overlap": z, "pair_count": z} if return_parts else z
#     ov = torch.stack(overlaps)
#     loss = torch.stack(losses).mean()
#     if return_parts:
#         return {
#             "total": loss,
#             "mean_overlap": ov.mean().detach(),
#             "max_overlap": ov.max().detach(),
#             "pair_count": torch.tensor(float(ov.numel()), device=new.device, dtype=new.dtype),
#         }
#     return loss


# def center_to_old_ellipsoid_loss(
#     new_means: Optional[torch.Tensor],
#     old_means: Optional[torch.Tensor],
#     old_bases: Optional[torch.Tensor],
#     old_variances: Optional[torch.Tensor],
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     old_reliability: Optional[torch.Tensor] = None,
#     old_sample_counts: Optional[torch.Tensor] = None,
#     margin: float = 1.0,
#     variance_floor: float = 1e-4,
#     reliability_energy_weight: float = 0.05,
#     residual_variance_scale: float = 0.75,
#     normalize_by_dim: bool = True,
#     use_logdet_energy: bool = True,
#     logdet_energy_weight: float = 0.05,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Keep new class centers outside frozen old low-rank ellipsoids."""
#     ref = new_means if torch.is_tensor(new_means) else old_means
#     if new_means is None or old_means is None or old_bases is None or old_variances is None:
#         z = safe_zero_like(ref)
#         return {"total": z, "min_old_energy": z, "mean_old_energy": z} if return_parts else z
#     if new_means.numel() == 0 or old_means.numel() == 0:
#         z = safe_zero_like(ref)
#         return {"total": z, "min_old_energy": z, "mean_old_energy": z} if return_parts else z
#     # If no old counts were supplied, treat all old rows as valid.  This function
#     # is often called on already-sliced old snapshots.
#     if old_sample_counts is None:
#         old_sample_counts = torch.ones((old_means.size(0),), device=old_means.device, dtype=old_means.dtype)
#     e = geometry_energy_matrix(
#         new_means,
#         old_means,
#         old_bases,
#         old_variances,
#         active_ranks=old_active_ranks,
#         reliability=old_reliability,
#         sample_counts=old_sample_counts,
#         variance_floor=float(variance_floor),
#         reliability_energy_weight=float(reliability_energy_weight),
#         residual_variance_scale=float(residual_variance_scale),
#         normalize_by_dim=bool(normalize_by_dim),
#         use_logdet_energy=bool(use_logdet_energy),
#         logdet_energy_weight=float(logdet_energy_weight),
#     )
#     nearest_old = e.min(dim=1).values
#     loss = F.relu(float(margin) - nearest_old).pow(2).mean()
#     if return_parts:
#         return {
#             "total": loss,
#             "min_old_energy": nearest_old.min().detach(),
#             "mean_old_energy": nearest_old.mean().detach(),
#         }
#     return loss


# def descriptor_volume_control_loss(
#     variances: Optional[torch.Tensor],
#     *,
#     active_ranks: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     feature_dim: Optional[int] = None,
#     max_logdet: Optional[float] = None,
#     reference_variances: Optional[torch.Tensor] = None,
#     reference_active_ranks: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     normalize_by_dim: bool = True,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Prevent new descriptors from becoming broad covariance blobs."""
#     if variances is None or not torch.is_tensor(variances) or variances.numel() == 0:
#         z = safe_zero_like(variances)
#         return {"total": z, "mean_logdet": z, "max_logdet": z, "cap": z} if return_parts else z
#     if variances.dim() != 2:
#         raise ValueError(f"variances must be [C,R+1], got {tuple(variances.shape)}")
#     C, Rp1 = variances.shape
#     R = Rp1 - 1
#     D = int(feature_dim or max(R, 1))
#     device, dtype = variances.device, variances.dtype
#     if active_ranks is None or not torch.is_tensor(active_ranks) or active_ranks.numel() != C:
#         ar = torch.full((C,), R, device=device, dtype=torch.long)
#     else:
#         ar = active_ranks.to(device=device).long().clamp(0, R)
#     valid = torch.ones((C,), device=device, dtype=torch.bool)
#     if sample_counts is not None and torch.is_tensor(sample_counts) and sample_counts.numel() == C:
#         valid = sample_counts.to(device=device).flatten() > 0
#     mask = (torch.arange(R, device=device).view(1, R) < ar.view(C, 1)).to(dtype=dtype)
#     logdet = _low_rank_logdet_penalty(
#         eigvals=variances[:, :-1].clamp_min(float(variance_floor)),
#         resvars=variances[:, -1].clamp_min(float(variance_floor)),
#         rank_mask=mask,
#         active_ranks=ar,
#         feature_dim=D,
#         valid_mask=valid,
#         variance_floor=float(variance_floor),
#         normalize_by_dim=bool(normalize_by_dim),
#         center=False,
#     )
#     if max_logdet is None:
#         if reference_variances is not None and torch.is_tensor(reference_variances) and reference_variances.numel() > 0:
#             rC, rRp1 = reference_variances.shape
#             rR = rRp1 - 1
#             if reference_active_ranks is None or not torch.is_tensor(reference_active_ranks) or reference_active_ranks.numel() != rC:
#                 rar = torch.full((rC,), rR, device=device, dtype=torch.long)
#             else:
#                 rar = reference_active_ranks.to(device=device).long().clamp(0, rR)
#             rmask = (torch.arange(rR, device=device).view(1, rR) < rar.view(rC, 1)).to(dtype=dtype)
#             rvalid = torch.ones((rC,), device=device, dtype=torch.bool)
#             rlogdet = _low_rank_logdet_penalty(
#                 eigvals=reference_variances.to(device=device, dtype=dtype)[:, :-1].clamp_min(float(variance_floor)),
#                 resvars=reference_variances.to(device=device, dtype=dtype)[:, -1].clamp_min(float(variance_floor)),
#                 rank_mask=rmask,
#                 active_ranks=rar,
#                 feature_dim=D,
#                 valid_mask=rvalid,
#                 variance_floor=float(variance_floor),
#                 normalize_by_dim=bool(normalize_by_dim),
#                 center=False,
#             )
#             cap = rlogdet.mean().detach() + rlogdet.std(unbiased=False).detach().clamp_min(0.0)
#         else:
#             cap = logdet[valid].mean().detach() if bool(valid.any().item()) else logdet.mean().detach()
#     else:
#         cap = torch.tensor(float(max_logdet), device=device, dtype=dtype)
#     over = F.relu(logdet - cap).pow(2)
#     loss = over[valid].mean() if bool(valid.any().item()) else variances.sum() * 0.0
#     if return_parts:
#         return {
#             "total": loss,
#             "mean_logdet": logdet[valid].mean().detach() if bool(valid.any().item()) else variances.sum().detach() * 0.0,
#             "max_logdet": logdet[valid].max().detach() if bool(valid.any().item()) else variances.sum().detach() * 0.0,
#             "cap": cap.detach(),
#         }
#     return loss


# def descriptor_trust_region_loss(
#     current_means: Optional[torch.Tensor],
#     current_bases: Optional[torch.Tensor],
#     current_variances: Optional[torch.Tensor],
#     init_means: Optional[torch.Tensor],
#     init_bases: Optional[torch.Tensor],
#     init_variances: Optional[torch.Tensor],
#     *,
#     active_ranks: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     mean_weight: float = 1.0,
#     basis_weight: float = 1.0,
#     variance_weight: float = 1.0,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Bound descriptor-only plasticity around the inserted estimate."""
#     if any(v is None for v in [current_means, current_bases, current_variances, init_means, init_bases, init_variances]):
#         z = safe_zero_like(current_means if torch.is_tensor(current_means) else init_means)
#         return {"total": z, "mean": z, "basis": z, "variance": z} if return_parts else z
#     n = min(current_means.size(0), init_means.size(0))
#     if n <= 0:
#         z = safe_zero_like(current_means)
#         return {"total": z, "mean": z, "basis": z, "variance": z} if return_parts else z
#     cur_m = current_means[:n]
#     init_m = init_means[:n].to(cur_m.device, cur_m.dtype)
#     cur_b = current_bases[:n]
#     init_b = init_bases[:n].to(cur_b.device, cur_b.dtype)
#     cur_v = current_variances[:n]
#     init_v = init_variances[:n].to(cur_v.device, cur_v.dtype)
#     ar = active_ranks[:n] if active_ranks is not None and torch.is_tensor(active_ranks) and active_ranks.numel() >= n else None
#     mean_loss = (cur_m - init_m).pow(2).mean()
#     basis_loss = (projectors_from_basis(cur_b, ar) - projectors_from_basis(init_b, ar)).pow(2).mean()
#     var_loss = (
#         torch.log(cur_v.clamp_min(float(variance_floor)))
#         - torch.log(init_v.clamp_min(float(variance_floor)))
#     ).pow(2).mean()
#     total = float(mean_weight) * mean_loss + float(basis_weight) * basis_loss + float(variance_weight) * var_loss
#     if return_parts:
#         return {"total": total, "mean": mean_loss.detach(), "basis": basis_loss.detach(), "variance": var_loss.detach()}
#     return total


# # Compatibility alias: dual risk now means feature geometry risk only.
# def old_new_dual_geometry_risk(**kwargs: object) -> torch.Tensor:
#     return old_new_geometry_risk(
#         kwargs["old_means"],
#         kwargs["old_bases"],
#         kwargs["new_means"],
#         kwargs["new_bases"],
#         old_active_ranks=kwargs.get("old_active_ranks"),
#         new_active_ranks=kwargs.get("new_active_ranks"),
#         old_reliability=kwargs.get("old_reliability"),
#         center_weight=float(kwargs.get("feature_center_weight", 0.5)),
#         subspace_weight=float(kwargs.get("feature_subspace_weight", 0.5)),
#     )


# class OldGeometryInvasionLoss(nn.Module):
#     """
#     Penalizes current new features that fall inside old geometry basins.

#     This is the clean incremental separation term. It uses the stored old bank as
#     a geometric constraint and does not require old raw samples.
#     """

#     def __init__(
#         self,
#         margin: float = 0.25,
#         variance_floor: float = 1e-4,
#         reliability_energy_weight: float = 0.05,
#         residual_variance_scale: float = 0.75,
#         invalid_class_energy: float = 1e6,
#         normalize_by_dim: bool = True,
#         use_logdet_energy: bool = True,
#         logdet_energy_weight: float = 0.05,
#     ) -> None:
#         super().__init__()
#         self.margin = float(margin)
#         self.variance_floor = float(variance_floor)
#         self.reliability_energy_weight = float(reliability_energy_weight)
#         self.residual_variance_scale = float(residual_variance_scale)
#         self.invalid_class_energy = float(invalid_class_energy)
#         self.normalize_by_dim = bool(normalize_by_dim)
#         self.use_logdet_energy = bool(use_logdet_energy)
#         self.logdet_energy_weight = float(logdet_energy_weight)

#     def forward(
#         self,
#         features: Optional[torch.Tensor],
#         labels: Optional[torch.Tensor],
#         *,
#         old_class_count: int,
#         means: Optional[torch.Tensor],
#         bases: Optional[torch.Tensor],
#         variances: Optional[torch.Tensor],
#         active_ranks: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         **_: object,
#     ) -> Dict[str, torch.Tensor]:
#         if features is None or labels is None or means is None or bases is None or variances is None:
#             z = safe_zero_like(features if torch.is_tensor(features) else means)
#             return {"total": z, "own": z, "old": z, "active": z}
#         if features.numel() == 0 or means.numel() == 0:
#             z = safe_zero_like(features)
#             return {"total": z, "own": z, "old": z, "active": z}
#         old_class_count = int(old_class_count)
#         if old_class_count <= 0 or old_class_count >= means.size(0):
#             z = features.sum() * 0.0
#             return {"total": z, "own": z, "old": z, "active": z}
#         y = labels.to(device=features.device).long().flatten()
#         valid = (y >= old_class_count) & (y < means.size(0))
#         if not bool(valid.any().item()):
#             z = features.sum() * 0.0
#             return {"total": z, "own": z, "old": z, "active": z}
#         z_new = features[valid]
#         y_new = y[valid]
#         energy = geometry_energy_matrix(
#             z_new,
#             means,
#             bases,
#             variances,
#             active_ranks=active_ranks,
#             reliability=reliability,
#             sample_counts=sample_counts,
#             variance_floor=self.variance_floor,
#             reliability_energy_weight=self.reliability_energy_weight,
#             residual_variance_scale=self.residual_variance_scale,
#             normalize_by_dim=self.normalize_by_dim,
#             invalid_class_energy=self.invalid_class_energy,
#             use_logdet_energy=self.use_logdet_energy,
#             logdet_energy_weight=self.logdet_energy_weight,
#         )
#         own = energy.gather(1, y_new.view(-1, 1)).squeeze(1)
#         nearest_old = energy[:, :old_class_count].min(dim=1).values
#         loss = F.relu(own + self.margin - nearest_old).mean()
#         return {
#             "total": loss,
#             "own": own.detach().mean(),
#             "old": nearest_old.detach().mean(),
#             "active": torch.tensor(float(z_new.size(0)), device=features.device, dtype=features.dtype),
#         }


# # Compatibility names for older trainer imports.
# BoundarySubspaceSeparationLoss = OldGeometryInvasionLoss
# RiskAwareBoundarySubspaceSeparationLoss = OldGeometryInvasionLoss
# SymmetricBoundarySubspaceSeparationLoss = OldGeometryInvasionLoss


# # -----------------------------------------------------------------------------
# # Synthetic replay from geometry
# # -----------------------------------------------------------------------------

# @torch.no_grad()
# def sample_geometry_features(
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     *,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     samples_per_class: int = 16,
#     variance_floor: float = 1e-4,
#     parallel_scale: float = 1.0,
#     residual_scale: float = 0.25,
#     reliability_gated: bool = False,
#     skip_invalid_classes: bool = True,
#     class_ids: Optional[Iterable[int]] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Sample synthetic features from stored low-rank geometry.

#     Labels are global class ids when ``class_ids`` is provided.  If omitted,
#     labels default to row indices.  This keeps old-snapshot replay correct for
#     contiguous old rows while allowing future non-contiguous class-order tests.
#     """
#     if means is None or not torch.is_tensor(means) or means.numel() == 0 or samples_per_class <= 0:
#         device = means.device if torch.is_tensor(means) else torch.device("cpu")
#         return torch.empty(0, 0, device=device), torch.empty(0, dtype=torch.long, device=device)
#     _validate_bank_tensors(means, bases, variances, name="sample_geometry")
#     C, D = means.shape
#     R = bases.size(2)
#     device, dtype = means.device, means.dtype

#     if class_ids is None:
#         label_ids = list(range(C))
#     else:
#         label_ids = [int(c) for c in class_ids]
#         if len(label_ids) != C:
#             raise ValueError(f"class_ids length must match bank rows C={C}, got {len(label_ids)}")

#     if active_ranks is None or not torch.is_tensor(active_ranks) or active_ranks.numel() != C:
#         active_ranks = torch.full((C,), R, device=device, dtype=torch.long)
#     else:
#         active_ranks = active_ranks.to(device=device).long().clamp(0, R)
#     valid = _valid_class_mask_from_counts(sample_counts, C, device)

#     feats, labels = [], []
#     for row in range(C):
#         if skip_invalid_classes and not bool(valid[row].item()):
#             continue
#         n = int(samples_per_class)
#         if n <= 0:
#             continue
#         mu = means[row]
#         z = mu.unsqueeze(0).expand(n, D).clone()
#         r = int(active_ranks[row].item())
#         rel_gate = torch.tensor(1.0, device=device, dtype=dtype)
#         if reliability_gated and reliability is not None and torch.is_tensor(reliability) and reliability.numel() == C:
#             rel_gate = reliability.to(device=device, dtype=dtype).flatten()[row].clamp(0.05, 1.0)

#         if r > 0:
#             U = bases[row, :, :r]
#             eig = variances[row, :r].clamp_min(float(variance_floor))
#             if reliability_gated:
#                 eig = rel_gate * eig + (1.0 - rel_gate) * torch.full_like(eig, float(variance_floor))
#             eps = torch.randn(n, r, device=device, dtype=dtype) * eig.sqrt().unsqueeze(0)
#             z = z + float(parallel_scale) * (eps @ U.t())

#         res = variances[row, -1].clamp_min(float(variance_floor))
#         if reliability_gated:
#             res = rel_gate * res + (1.0 - rel_gate) * torch.as_tensor(float(variance_floor), device=device, dtype=dtype)
#         z = z + float(residual_scale) * torch.randn(n, D, device=device, dtype=dtype) * res.sqrt()

#         feats.append(z)
#         labels.append(torch.full((n,), int(label_ids[row]), device=device, dtype=torch.long))

#     if not feats:
#         return torch.empty(0, D, device=device, dtype=dtype), torch.empty(0, dtype=torch.long, device=device)
#     return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# class GeometryFeatureAnchoringLoss(nn.Module):
#     """Old geometry replay CE using synthetic features sampled from the bank."""

#     def __init__(
#         self,
#         samples_per_class: int = 16,
#         variance_floor: float = 1e-4,
#         logit_scale: float = 8.0,
#         reliability_energy_weight: float = 0.05,
#         residual_variance_scale: float = 0.75,
#         invalid_class_energy: float = 1e6,
#         parallel_scale: float = 1.0,
#         residual_scale: float = 0.25,
#         normalize_by_dim: bool = True,
#         reliability_gated: bool = False,
#         use_logdet_energy: bool = True,
#         logdet_energy_weight: float = 0.05,
#     ) -> None:
#         super().__init__()
#         self.samples_per_class = int(samples_per_class)
#         self.variance_floor = float(variance_floor)
#         self.logit_scale = float(logit_scale)
#         self.reliability_energy_weight = float(reliability_energy_weight)
#         self.residual_variance_scale = float(residual_variance_scale)
#         self.invalid_class_energy = float(invalid_class_energy)
#         self.parallel_scale = float(parallel_scale)
#         self.residual_scale = float(residual_scale)
#         self.normalize_by_dim = bool(normalize_by_dim)
#         self.reliability_gated = bool(reliability_gated)
#         self.use_logdet_energy = bool(use_logdet_energy)
#         self.logdet_energy_weight = float(logdet_energy_weight)

#     def forward(
#         self,
#         old_means: Optional[torch.Tensor],
#         old_bases: Optional[torch.Tensor],
#         old_variances: Optional[torch.Tensor],
#         *,
#         all_means: Optional[torch.Tensor] = None,
#         all_bases: Optional[torch.Tensor] = None,
#         all_variances: Optional[torch.Tensor] = None,
#         old_active_ranks: Optional[torch.Tensor] = None,
#         all_active_ranks: Optional[torch.Tensor] = None,
#         old_reliability: Optional[torch.Tensor] = None,
#         all_reliability: Optional[torch.Tensor] = None,
#         old_sample_counts: Optional[torch.Tensor] = None,
#         all_sample_counts: Optional[torch.Tensor] = None,
#         old_class_ids: Optional[Iterable[int]] = None,
#         return_anchors: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         if old_means is None or old_bases is None or old_variances is None or old_means.numel() == 0:
#             z = safe_zero_like(old_means)
#             out = {"total": z, "ce": z, "num_anchors": z}
#             if return_anchors:
#                 out.update({"anchor_features": torch.empty(0, 0), "anchor_labels": torch.empty(0, dtype=torch.long)})
#             return out
#         anchor_x, anchor_y = sample_geometry_features(
#             old_means,
#             old_bases,
#             old_variances,
#             active_ranks=old_active_ranks,
#             reliability=old_reliability,
#             sample_counts=old_sample_counts,
#             samples_per_class=self.samples_per_class,
#             variance_floor=self.variance_floor,
#             parallel_scale=self.parallel_scale,
#             residual_scale=self.residual_scale,
#             reliability_gated=self.reliability_gated,
#             skip_invalid_classes=True,
#             class_ids=old_class_ids,
#         )
#         if anchor_x.numel() == 0:
#             z = safe_zero_like(old_means)
#             out = {"total": z, "ce": z, "num_anchors": z}
#             if return_anchors:
#                 out.update({"anchor_features": anchor_x, "anchor_labels": anchor_y})
#             return out

#         means = all_means if all_means is not None and torch.is_tensor(all_means) and all_means.numel() > 0 else old_means
#         bases = all_bases if all_bases is not None and torch.is_tensor(all_bases) and all_bases.numel() > 0 else old_bases
#         variances = all_variances if all_variances is not None and torch.is_tensor(all_variances) and all_variances.numel() > 0 else old_variances
#         active_ranks = all_active_ranks if all_active_ranks is not None else old_active_ranks
#         reliability = all_reliability if all_reliability is not None else old_reliability
#         sample_counts = all_sample_counts if all_sample_counts is not None else old_sample_counts
#         energy = geometry_energy_matrix(
#             anchor_x,
#             means,
#             bases,
#             variances,
#             active_ranks=active_ranks,
#             reliability=reliability,
#             sample_counts=sample_counts,
#             variance_floor=self.variance_floor,
#             reliability_energy_weight=self.reliability_energy_weight,
#             residual_variance_scale=self.residual_variance_scale,
#             normalize_by_dim=self.normalize_by_dim,
#             invalid_class_energy=self.invalid_class_energy,
#             use_logdet_energy=self.use_logdet_energy,
#             logdet_energy_weight=self.logdet_energy_weight,
#         )
#         obj = valid_energy_objective(
#             energy,
#             anchor_y,
#             sample_counts=sample_counts,
#             logit_scale=self.logit_scale,
#             min_energy_scale=1.0,
#             rank_margin=0.25,
#             label_smoothing=0.0,
#         )
#         out = {
#             "total": obj["ce"],
#             "ce": obj["ce"],
#             "num_anchors": torch.tensor(float(anchor_x.size(0)), device=anchor_x.device, dtype=anchor_x.dtype),
#         }
#         if return_anchors:
#             out.update({"anchor_features": anchor_x, "anchor_labels": anchor_y})
#         return out


# # Hard boundary sampler compatibility: now uses faithful geometry replay.
# sample_boundary_geometry_features = sample_geometry_features
# BoundaryGeometryAnchoringLoss = GeometryFeatureAnchoringLoss


# # -----------------------------------------------------------------------------
# # Geometry drift regularization; kept as optional guard, not core method
# # -----------------------------------------------------------------------------

# def projectors_from_basis(bases: torch.Tensor, active_ranks: Optional[torch.Tensor] = None) -> torch.Tensor:
#     if bases.dim() == 2:
#         r = bases.size(1)
#         if active_ranks is not None:
#             r = int(torch.as_tensor(active_ranks).detach().cpu().item())
#             r = max(0, min(r, bases.size(1)))
#         if r <= 0:
#             return torch.zeros(bases.size(0), bases.size(0), device=bases.device, dtype=bases.dtype)
#         U = bases[:, :r]
#         return U @ U.t()
#     if bases.dim() != 3:
#         raise ValueError(f"bases must be [D,R] or [C,D,R], got {tuple(bases.shape)}")
#     C, D, R = bases.shape
#     out = torch.zeros(C, D, D, device=bases.device, dtype=bases.dtype)
#     if active_ranks is None:
#         ar = torch.full((C,), R, device=bases.device, dtype=torch.long)
#     else:
#         ar = active_ranks.to(device=bases.device).long().clamp(0, R)
#     for c in range(C):
#         r = int(ar[c].item())
#         if r > 0:
#             U = bases[c, :, :r]
#             out[c] = U @ U.t()
#     return out


# class GeometryDriftRegularization(nn.Module):
#     def __init__(
#         self,
#         variance_floor: float = 1e-4,
#         mean_weight: float = 1.0,
#         basis_weight: float = 1.0,
#         variance_weight: float = 1.0,
#         reliability_weighted: bool = True,
#     ) -> None:
#         super().__init__()
#         self.variance_floor = float(variance_floor)
#         self.mean_weight = float(mean_weight)
#         self.basis_weight = float(basis_weight)
#         self.variance_weight = float(variance_weight)
#         self.reliability_weighted = bool(reliability_weighted)

#     def forward(
#         self,
#         current_means: Optional[torch.Tensor],
#         current_bases: Optional[torch.Tensor],
#         current_variances: Optional[torch.Tensor],
#         snapshot_means: Optional[torch.Tensor],
#         snapshot_bases: Optional[torch.Tensor],
#         snapshot_variances: Optional[torch.Tensor],
#         *,
#         active_ranks: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         if any(v is None for v in [current_means, current_bases, current_variances, snapshot_means, snapshot_bases, snapshot_variances]):
#             z = safe_zero_like(current_means)
#             return {"total": z, "mean": z, "basis": z, "variance": z}
#         if current_means.numel() == 0 or snapshot_means.numel() == 0:
#             z = safe_zero_like(current_means)
#             return {"total": z, "mean": z, "basis": z, "variance": z}
#         n = min(current_means.size(0), snapshot_means.size(0))
#         cur_m = current_means[:n]
#         old_m = snapshot_means[:n].to(cur_m.device, cur_m.dtype)
#         cur_b = current_bases[:n]
#         old_b = snapshot_bases[:n].to(cur_b.device, cur_b.dtype)
#         cur_v = current_variances[:n]
#         old_v = snapshot_variances[:n].to(cur_v.device, cur_v.dtype)
#         ar = active_ranks[:n] if active_ranks is not None and active_ranks.numel() >= n else None
#         mean_per = (cur_m - old_m).pow(2).mean(dim=1)
#         basis_per = (projectors_from_basis(cur_b, ar) - projectors_from_basis(old_b, ar)).pow(2).flatten(1).mean(dim=1)
#         var_per = (torch.log(cur_v.clamp_min(self.variance_floor)) - torch.log(old_v.clamp_min(self.variance_floor))).pow(2).mean(dim=1)
#         if self.reliability_weighted and reliability is not None and torch.is_tensor(reliability) and reliability.numel() >= n:
#             w = reliability[:n].to(device=cur_m.device, dtype=cur_m.dtype).clamp(0.05, 1.0)
#             w = w / w.mean().clamp_min(1e-6)
#             mean_loss = (w * mean_per).mean()
#             basis_loss = (w * basis_per).mean()
#             var_loss = (w * var_per).mean()
#         else:
#             mean_loss = mean_per.mean()
#             basis_loss = basis_per.mean()
#             var_loss = var_per.mean()
#         total = self.mean_weight * mean_loss + self.basis_weight * basis_loss + self.variance_weight * var_loss
#         return {"total": total, "mean": mean_loss, "basis": basis_loss, "variance": var_loss}



# def _deprecated_clean_zero_loss(name: str, args: tuple, kwargs: Dict[str, object]) -> Dict[str, torch.Tensor]:
#     """Compatibility hook for removed losses.

#     These losses are not part of the clean method.  They may be called only with
#     zero weight by stale code; a positive weight is a configuration error.
#     """
#     weight_keys = (
#         "weight",
#         "loss_weight",
#         "ssgl_weight",
#         "shared_weight",
#         "feature_weight",
#         "spectral_weight",
#         "gdr_weight",
#         "bss_weight",
#         "sym_bss_weight",
#     )
#     for key in weight_keys:
#         if key in kwargs:
#             try:
#                 if float(kwargs[key]) != 0.0:
#                     raise RuntimeError(f"{name} is removed from the clean NECIL-HSI path; set {key}=0.0.")
#             except (TypeError, ValueError):
#                 pass
#     ref = args[0] if args else None
#     z = safe_zero_like(ref)
#     return {
#         "total": z,
#         "coverage": z,
#         "exclusion": z,
#         "reserve": z,
#         "mean_margin": z,
#         "min_margin": z,
#         "violation_rate": z,
#         "valid_samples": z,
#         "reserve_active": z,
#         "feature": z,
#         "spectral": z,
#         "num_pairs": z,
#     }


# def strategic_spectral_spatial_geometry_loss(*args, **kwargs):
#     out = _deprecated_clean_zero_loss("strategic_spectral_spatial_geometry_loss", args, kwargs)
#     return {
#         "total": out["total"],
#         "coverage": out["coverage"],
#         "exclusion": out["exclusion"],
#         "reserve": out["reserve"],
#         "mean_margin": out["mean_margin"],
#         "min_margin": out["min_margin"],
#         "violation_rate": out["violation_rate"],
#         "valid_samples": out["valid_samples"],
#         "reserve_active": out["reserve_active"],
#     }


# def ssgl_loss(*args, **kwargs):
#     return strategic_spectral_spatial_geometry_loss(*args, **kwargs)


# def shared_mode_tangent_decorrelation_loss(*args, **kwargs):
#     out = _deprecated_clean_zero_loss("shared_mode_tangent_decorrelation_loss", args, kwargs)
#     return {"total": out["total"], "feature": out["feature"], "spectral": out["spectral"], "num_pairs": out["num_pairs"]}


# class OldAnchorConsistencyLoss(nn.Module):
#     """Compatibility hook. Plain MSE in projected geometry space; no KD logits."""

#     def __init__(self, feature_weight: float = 1.0, logit_weight: float = 0.0, temperature: float = 2.0) -> None:
#         super().__init__()
#         self.feature_weight = float(feature_weight)
#         self.logit_weight = 0.0
#         self.temperature = float(temperature)

#     def forward(
#         self,
#         z_before: Optional[torch.Tensor],
#         z_after: Optional[torch.Tensor],
#         *,
#         logits_before: Optional[torch.Tensor] = None,
#         logits_after: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         del logits_before, logits_after
#         ref = z_after if isinstance(z_after, torch.Tensor) else z_before
#         if z_before is None or z_after is None or not torch.is_tensor(z_before) or not torch.is_tensor(z_after) or z_before.numel() == 0 or z_after.numel() == 0:
#             z = safe_zero_like(ref)
#             return {"total": z, "feature": z, "logit": z}
#         if z_before.shape != z_after.shape:
#             raise ValueError(f"OldAnchorConsistency shape mismatch: {tuple(z_before.shape)} vs {tuple(z_after.shape)}")
#         feature_loss = F.mse_loss(z_after, z_before.detach())
#         z = safe_zero_like(z_after)
#         return {"total": self.feature_weight * feature_loss, "feature": feature_loss, "logit": z}







# # -----------------------------------------------------------------------------
# # SRGP additions: spectral-residual losses and HSI-aware old/new separation
# # -----------------------------------------------------------------------------
# # The definitions below intentionally override a few earlier compatibility
# # functions.  This keeps stale trainer imports working while making the active
# # behavior consistent with the SRGP GeometryBank and classifier.

# _feature_geometry_energy_matrix = geometry_energy_matrix
# _feature_old_new_geometry_risk = old_new_geometry_risk
# _feature_descriptor_subspace_collision_loss = descriptor_subspace_collision_loss


# def _spectral_derivatives(spectral_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """First and second derivatives along physical spectral-band order."""
#     if spectral_summary.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
#     if spectral_summary.size(1) >= 2:
#         d1 = spectral_summary[:, 1:] - spectral_summary[:, :-1]
#     else:
#         d1 = spectral_summary.new_empty((spectral_summary.size(0), 0))
#     if spectral_summary.size(1) >= 3:
#         d2 = spectral_summary[:, 2:] - 2.0 * spectral_summary[:, 1:-1] + spectral_summary[:, :-2]
#     else:
#         d2 = spectral_summary.new_empty((spectral_summary.size(0), 0))
#     return d1, d2


# def _positive_cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """Positive cosine only. Opposite spectral slopes are not treated as conflicts."""
#     if a.numel() == 0 or b.numel() == 0:
#         return torch.zeros((a.size(0), b.size(0)), device=a.device, dtype=a.dtype)
#     aa = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
#     bb = torch.nan_to_num(b.to(device=a.device, dtype=a.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#     aa = aa - aa.mean(dim=1, keepdim=True)
#     bb = bb - bb.mean(dim=1, keepdim=True)
#     aa = F.normalize(aa, dim=1, eps=1e-8)
#     bb = F.normalize(bb, dim=1, eps=1e-8)
#     return (aa @ bb.t()).clamp(0.0, 1.0)


# def spectral_shape_similarity_matrix(
#     old_spectral_curve_means: Optional[torch.Tensor],
#     new_spectral_curve_means: Optional[torch.Tensor],
#     *,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_reliability: Optional[torch.Tensor] = None,
#     new_reliability: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """HSI spectral-shape similarity between old and new class descriptors.

#     The similarity uses curve, first derivative, and second derivative descriptors
#     when available.  It returns positive cosine similarity in [0,1].  This should
#     be used for conflict weighting and adaptive old/new subspace margins, not as
#     an exemplar memory.
#     """
#     ref = old_spectral_curve_means if torch.is_tensor(old_spectral_curve_means) else new_spectral_curve_means
#     if ref is None or not torch.is_tensor(ref):
#         return torch.empty((0, 0))
#     device, dtype = ref.device, ref.dtype
#     if old_spectral_curve_means is None or new_spectral_curve_means is None:
#         O = old_spectral_curve_means.size(0) if torch.is_tensor(old_spectral_curve_means) else 0
#         N = new_spectral_curve_means.size(0) if torch.is_tensor(new_spectral_curve_means) else 0
#         return torch.zeros((O, N), device=device, dtype=dtype)
#     if old_spectral_curve_means.numel() == 0 or new_spectral_curve_means.numel() == 0:
#         return torch.zeros((old_spectral_curve_means.size(0), new_spectral_curve_means.size(0)), device=device, dtype=dtype)

#     old_parts = [old_spectral_curve_means]
#     new_parts = [new_spectral_curve_means]
#     if old_spectral_curve_d1 is not None and new_spectral_curve_d1 is not None and torch.is_tensor(old_spectral_curve_d1) and torch.is_tensor(new_spectral_curve_d1):
#         if old_spectral_curve_d1.numel() > 0 and new_spectral_curve_d1.numel() > 0 and old_spectral_curve_d1.size(1) == new_spectral_curve_d1.size(1):
#             old_parts.append(old_spectral_curve_d1)
#             new_parts.append(new_spectral_curve_d1)
#     if old_spectral_curve_d2 is not None and new_spectral_curve_d2 is not None and torch.is_tensor(old_spectral_curve_d2) and torch.is_tensor(new_spectral_curve_d2):
#         if old_spectral_curve_d2.numel() > 0 and new_spectral_curve_d2.numel() > 0 and old_spectral_curve_d2.size(1) == new_spectral_curve_d2.size(1):
#             old_parts.append(old_spectral_curve_d2)
#             new_parts.append(new_spectral_curve_d2)

#     old_v = torch.cat([p.to(device=device, dtype=dtype) for p in old_parts], dim=1)
#     new_v = torch.cat([p.to(device=device, dtype=dtype) for p in new_parts], dim=1)
#     sim = _positive_cosine_matrix(old_v, new_v)
#     if old_reliability is not None and torch.is_tensor(old_reliability) and old_reliability.numel() == sim.size(0):
#         sim = sim * old_reliability.to(device=device, dtype=dtype).flatten().clamp(0.05, 1.0).view(-1, 1)
#     if new_reliability is not None and torch.is_tensor(new_reliability) and new_reliability.numel() == sim.size(1):
#         sim = sim * new_reliability.to(device=device, dtype=dtype).flatten().clamp(0.05, 1.0).view(1, -1)
#     return torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


# def spectral_residual_energy_matrix(
#     spectral_summary: Optional[torch.Tensor],
#     spectral_curve_means: Optional[torch.Tensor],
#     spectral_curve_vars: Optional[torch.Tensor],
#     *,
#     spectral_curve_d1: Optional[torch.Tensor] = None,
#     spectral_curve_d2: Optional[torch.Tensor] = None,
#     spectral_shape_reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     derivative_weight: float = 0.50,
#     second_derivative_weight: float = 0.25,
#     normalize_by_components: bool = True,
#     spectral_summary_is_physical: bool = False,
#     require_physical_summary: bool = True,
#     invalid_class_energy: float = 1e6,
#     return_parts: bool = False,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """SRGP spectral-residual energy for physical HSI spectra.

#     This mirrors the SRGP classifier branch.  It is intentionally dormant unless
#     physical spectral summaries and bank spectral-shape descriptors are supplied.
#     Synthetic replay features normally have no physical spectra, so this returns
#     zero energy for that path.
#     """
#     ref = spectral_summary if torch.is_tensor(spectral_summary) else spectral_curve_means
#     if ref is None or not torch.is_tensor(ref):
#         z = torch.tensor(0.0)
#         out = {"spectral_energy": z.view(1, 1) * 0.0}
#         return out if return_parts else out["spectral_energy"]
#     if (
#         spectral_summary is None
#         or spectral_curve_means is None
#         or not torch.is_tensor(spectral_summary)
#         or not torch.is_tensor(spectral_curve_means)
#         or spectral_summary.numel() == 0
#         or spectral_curve_means.numel() == 0
#         or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
#     ):
#         z = ref.sum() * 0.0
#         out = {"spectral_energy": z.view(1, 1) * 0.0}
#         return out if return_parts else out["spectral_energy"]

#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     if s.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(s.shape)}")
#     means = torch.nan_to_num(spectral_curve_means.to(device=s.device, dtype=s.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#     if means.dim() != 2:
#         raise ValueError(f"spectral_curve_means must be [C,S], got {tuple(means.shape)}")
#     if means.size(1) != s.size(1):
#         raise ValueError(
#             f"spectral_summary width {s.size(1)} does not match bank spectral width {means.size(1)}. "
#             "Use raw/physical spectral summaries aligned with GeometryBank extraction."
#         )
#     B, S = s.shape
#     C = means.size(0)
#     if sample_counts is not None and torch.is_tensor(sample_counts) and sample_counts.numel() == C:
#         valid = _valid_class_mask_from_counts(sample_counts, C, s.device)
#     else:
#         valid = torch.ones((C,), device=s.device, dtype=torch.bool)

#     if spectral_curve_vars is not None and torch.is_tensor(spectral_curve_vars) and spectral_curve_vars.numel() > 0:
#         var = torch.nan_to_num(spectral_curve_vars.to(device=s.device, dtype=s.dtype), nan=float(variance_floor), posinf=float(invalid_class_energy), neginf=float(variance_floor))
#         if var.shape != means.shape:
#             raise ValueError(f"spectral_curve_vars shape {tuple(var.shape)} must match means {tuple(means.shape)}")
#         var = var.clamp_min(float(variance_floor))
#     else:
#         var = torch.full_like(means, float(variance_floor))

#     curve = ((s.unsqueeze(1) - means.unsqueeze(0)).pow(2) / var.unsqueeze(0)).mean(dim=-1)
#     d1_energy = torch.zeros((B, C), device=s.device, dtype=s.dtype)
#     d2_energy = torch.zeros((B, C), device=s.device, dtype=s.dtype)
#     d1, d2 = _spectral_derivatives(s)

#     if derivative_weight > 0.0 and d1.numel() > 0 and spectral_curve_d1 is not None and torch.is_tensor(spectral_curve_d1) and spectral_curve_d1.numel() > 0:
#         bank_d1 = torch.nan_to_num(spectral_curve_d1.to(device=s.device, dtype=s.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#         if bank_d1.shape == (C, S - 1):
#             var_d1 = (var[:, 1:] + var[:, :-1]).clamp_min(float(variance_floor))
#             d1_energy = ((d1.unsqueeze(1) - bank_d1.unsqueeze(0)).pow(2) / var_d1.unsqueeze(0)).mean(dim=-1)
#     if second_derivative_weight > 0.0 and d2.numel() > 0 and spectral_curve_d2 is not None and torch.is_tensor(spectral_curve_d2) and spectral_curve_d2.numel() > 0:
#         bank_d2 = torch.nan_to_num(spectral_curve_d2.to(device=s.device, dtype=s.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#         if bank_d2.shape == (C, S - 2):
#             var_d2 = (var[:, 2:] + 4.0 * var[:, 1:-1] + var[:, :-2]).clamp_min(float(variance_floor))
#             d2_energy = ((d2.unsqueeze(1) - bank_d2.unsqueeze(0)).pow(2) / var_d2.unsqueeze(0)).mean(dim=-1)

#     energy = curve + float(derivative_weight) * d1_energy + float(second_derivative_weight) * d2_energy
#     if bool(normalize_by_components):
#         energy = energy / max(1.0 + float(derivative_weight) + float(second_derivative_weight), 1e-8)

#     rel = torch.ones((C,), device=s.device, dtype=s.dtype)
#     if spectral_shape_reliability is not None and torch.is_tensor(spectral_shape_reliability) and spectral_shape_reliability.numel() == C:
#         rel = torch.nan_to_num(spectral_shape_reliability.to(device=s.device, dtype=s.dtype).flatten(), nan=0.05, posinf=1.0, neginf=0.05).clamp(0.05, 1.0)
#     # Low-reliability spectral descriptors should have weak impact, not a harsh penalty.
#     energy = energy * rel.view(1, C)
#     energy = torch.nan_to_num(energy, nan=0.0, posinf=float(invalid_class_energy), neginf=0.0).masked_fill(~valid.view(1, C), float(invalid_class_energy))

#     if return_parts:
#         return {
#             "spectral_energy": energy,
#             "spectral_curve_energy": curve.masked_fill(~valid.view(1, C), float(invalid_class_energy)),
#             "spectral_d1_energy": d1_energy.masked_fill(~valid.view(1, C), float(invalid_class_energy)),
#             "spectral_d2_energy": d2_energy.masked_fill(~valid.view(1, C), float(invalid_class_energy)),
#             "spectral_reliability": rel,
#             "valid_mask": valid,
#         }
#     return energy


# def geometry_energy_matrix(
#     features: torch.Tensor,
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     *,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_curve_means: Optional[torch.Tensor] = None,
#     spectral_curve_vars: Optional[torch.Tensor] = None,
#     spectral_curve_d1: Optional[torch.Tensor] = None,
#     spectral_curve_d2: Optional[torch.Tensor] = None,
#     spectral_shape_reliability: Optional[torch.Tensor] = None,
#     use_spectral_residual_energy: bool = False,
#     spectral_energy_weight: float = 0.0,
#     spectral_derivative_weight: float = 0.50,
#     spectral_second_derivative_weight: float = 0.25,
#     spectral_summary_is_physical: bool = False,
#     spectral_require_physical_summary: bool = True,
#     return_parts: bool = False,
#     **kwargs: object,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """SRGP energy matrix: low-rank feature energy + optional spectral residual.

#     This wrapper preserves the original feature geometry energy and adds the HSI
#     spectral-shape term only when explicitly enabled and physical spectra are
#     available.  The default remains feature-only so synthetic replay is safe.
#     """
#     feature_parts = _feature_geometry_energy_matrix(
#         features,
#         means,
#         bases,
#         variances,
#         active_ranks=active_ranks,
#         reliability=reliability,
#         sample_counts=sample_counts,
#         return_parts=True,
#         **kwargs,
#     )
#     energy = feature_parts["energy"]
#     spectral_parts = spectral_residual_energy_matrix(
#         spectral_summary=spectral_summary,
#         spectral_curve_means=spectral_curve_means,
#         spectral_curve_vars=spectral_curve_vars,
#         spectral_curve_d1=spectral_curve_d1,
#         spectral_curve_d2=spectral_curve_d2,
#         spectral_shape_reliability=spectral_shape_reliability,
#         sample_counts=sample_counts,
#         variance_floor=float(kwargs.get("variance_floor", kwargs.get("var_floor", 1e-4))),
#         derivative_weight=float(spectral_derivative_weight),
#         second_derivative_weight=float(spectral_second_derivative_weight),
#         spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         require_physical_summary=bool(spectral_require_physical_summary),
#         invalid_class_energy=float(kwargs.get("invalid_class_energy", 1e6)),
#         return_parts=True,
#     )
#     spectral_energy = spectral_parts["spectral_energy"]
#     if bool(use_spectral_residual_energy) and float(spectral_energy_weight) > 0.0:
#         if spectral_energy.numel() == 1:
#             spectral_energy = torch.zeros_like(energy)
#         if spectral_energy.shape != energy.shape:
#             raise ValueError(f"spectral energy shape {tuple(spectral_energy.shape)} does not match feature energy {tuple(energy.shape)}")
#         energy = energy + float(spectral_energy_weight) * spectral_energy

#     if return_parts:
#         out = dict(feature_parts)
#         out["energy"] = energy
#         out["feature_energy"] = feature_parts["energy"]
#         out["spectral_energy"] = spectral_energy if spectral_energy.shape == energy.shape else torch.zeros_like(energy)
#         out["spectral_curve_energy"] = spectral_parts.get("spectral_curve_energy", torch.zeros_like(energy)) if isinstance(spectral_parts, dict) else torch.zeros_like(energy)
#         out["spectral_d1_energy"] = spectral_parts.get("spectral_d1_energy", torch.zeros_like(energy)) if isinstance(spectral_parts, dict) else torch.zeros_like(energy)
#         out["spectral_d2_energy"] = spectral_parts.get("spectral_d2_energy", torch.zeros_like(energy)) if isinstance(spectral_parts, dict) else torch.zeros_like(energy)
#         out["uses_spectral_residual_energy"] = torch.tensor(bool(use_spectral_residual_energy and float(spectral_energy_weight) > 0.0), device=energy.device)
#         return out
#     return energy


# def dual_geometry_energy_matrix(
#     features: torch.Tensor,
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     **kwargs: object,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """Compatibility alias now maps to SRGP energy."""
#     return geometry_energy_matrix(features, means, bases, variances, **kwargs)


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
#     """Base-phase HSI spectral-shape separation.

#     It discourages base classes from having identical mean spectral derivative
#     shapes.  This prepares the SRGP bank to store discriminative spectral-shape
#     descriptors.  It is skipped for PCA/non-physical summaries.
#     """
#     ref = spectral_summary if torch.is_tensor(spectral_summary) else labels
#     if (
#         spectral_summary is None
#         or not torch.is_tensor(spectral_summary)
#         or spectral_summary.numel() == 0
#         or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
#     ):
#         z = safe_zero_like(ref)
#         if return_parts:
#             return {"total": z, "spectral_shape": z, "pair_count": z, "valid_class_count": z, "mean_similarity": z}
#         return z
#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     y = labels.to(device=s.device).long().flatten()
#     if s.dim() != 2 or y.numel() != s.size(0):
#         raise ValueError("spectral_summary must be [B,S] aligned with labels")
#     d1, d2 = _spectral_derivatives(s)
#     desc = torch.cat([s, d1, d2], dim=1)
#     centers, class_ids, _ = _class_centers(desc, y, min_samples=min_samples, normalize_centers=False)
#     if centers.size(0) < 2:
#         z = s.sum() * 0.0
#         if return_parts:
#             return {"total": z, "spectral_shape": z, "pair_count": z, "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype), "mean_similarity": z}
#         return z
#     sim = _positive_cosine_matrix(centers, centers)
#     eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
#     pair_sim = sim[~eye]
#     if features is not None and torch.is_tensor(features) and features.numel() > 0:
#         zfeat = F.normalize(features.to(device=s.device, dtype=s.dtype), dim=1, eps=1e-6)
#         f_centers, f_ids, _ = _class_centers(zfeat, y, min_samples=min_samples, normalize_centers=False)
#         if f_centers.size(0) == centers.size(0) and torch.equal(f_ids.to(class_ids.device), class_ids):
#             dist = torch.cdist(f_centers, f_centers, p=2)
#             risk = F.relu(float(risk_center_margin) - dist)[~eye].detach() / max(float(risk_center_margin), 1e-6)
#         else:
#             risk = torch.ones_like(pair_sim)
#     else:
#         risk = torch.ones_like(pair_sim)
#     loss_vec = F.relu(pair_sim - float(max_shape_similarity)).pow(2) * (1.0 + float(risk_weight) * risk)
#     loss = loss_vec.mean() if loss_vec.numel() > 0 else s.sum() * 0.0
#     if return_parts:
#         return {
#             "total": loss,
#             "spectral_shape": loss.detach(),
#             "pair_count": torch.tensor(float(loss_vec.numel()), device=s.device, dtype=s.dtype),
#             "valid_class_count": torch.tensor(float(centers.size(0)), device=s.device, dtype=s.dtype),
#             "mean_similarity": pair_sim.mean().detach() if pair_sim.numel() > 0 else s.sum().detach() * 0.0,
#         }
#     return loss


# def old_new_geometry_risk(
#     old_means: torch.Tensor,
#     old_bases: torch.Tensor,
#     new_means: torch.Tensor,
#     new_bases: torch.Tensor,
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     old_reliability: Optional[torch.Tensor] = None,
#     center_weight: float = 0.40,
#     subspace_weight: float = 0.40,
#     spectral_weight: float = 0.20,
#     old_spectral_curve_means: Optional[torch.Tensor] = None,
#     new_spectral_curve_means: Optional[torch.Tensor] = None,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_spectral_reliability: Optional[torch.Tensor] = None,
#     new_spectral_reliability: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """SRGP old/new risk: feature-center + subspace + spectral-shape conflict."""
#     base = _feature_old_new_geometry_risk(
#         old_means,
#         old_bases,
#         new_means,
#         new_bases,
#         old_active_ranks=old_active_ranks,
#         new_active_ranks=new_active_ranks,
#         old_reliability=old_reliability,
#         center_weight=float(center_weight),
#         subspace_weight=float(subspace_weight),
#     )
#     if base.numel() == 0 or float(spectral_weight) <= 0.0:
#         return base
#     spec = spectral_shape_similarity_matrix(
#         old_spectral_curve_means,
#         new_spectral_curve_means,
#         old_spectral_curve_d1=old_spectral_curve_d1,
#         new_spectral_curve_d1=new_spectral_curve_d1,
#         old_spectral_curve_d2=old_spectral_curve_d2,
#         new_spectral_curve_d2=new_spectral_curve_d2,
#         old_reliability=old_spectral_reliability,
#         new_reliability=new_spectral_reliability,
#     )
#     if spec.shape == base.shape:
#         base = base + float(spectral_weight) * spec.to(device=base.device, dtype=base.dtype)
#     return torch.nan_to_num(base, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)


# def spectral_aware_subspace_separation_loss(
#     old_bases: Optional[torch.Tensor],
#     new_bases: Optional[torch.Tensor],
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     old_spectral_curve_means: Optional[torch.Tensor] = None,
#     new_spectral_curve_means: Optional[torch.Tensor] = None,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_spectral_reliability: Optional[torch.Tensor] = None,
#     new_spectral_reliability: Optional[torch.Tensor] = None,
#     target_overlap: float = 0.35,
#     spectral_margin_strength: float = 0.20,
#     min_target_overlap: float = 0.05,
#     reliability: Optional[torch.Tensor] = None,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Old-new subspace separation with HSI spectral-aware adaptive margin.

#     If old and new classes have similar spectral shapes, the allowed subspace
#     overlap is reduced.  This directly targets HSI feature overlap between
#     spectrally similar classes.
#     """
#     ref = new_bases if torch.is_tensor(new_bases) else old_bases
#     if old_bases is None or new_bases is None or not torch.is_tensor(old_bases) or not torch.is_tensor(new_bases):
#         z = safe_zero_like(ref)
#         return {"total": z, "mean_overlap": z, "max_overlap": z, "mean_target": z, "pair_count": z} if return_parts else z
#     if old_bases.numel() == 0 or new_bases.numel() == 0:
#         z = safe_zero_like(ref)
#         return {"total": z, "mean_overlap": z, "max_overlap": z, "mean_target": z, "pair_count": z} if return_parts else z
#     if old_bases.dim() != 3 or new_bases.dim() != 3:
#         raise ValueError("old_bases and new_bases must be [C,D,R].")
#     old = old_bases.to(device=new_bases.device, dtype=new_bases.dtype)
#     new = new_bases
#     O, D, Ro = old.shape
#     N, Dn, Rn = new.shape
#     if D != Dn:
#         raise ValueError(f"basis dimension mismatch: old D={D}, new D={Dn}")
#     old_ar = torch.full((O,), Ro, device=new.device, dtype=torch.long) if old_active_ranks is None else old_active_ranks.to(new.device).long().clamp(0, Ro)
#     new_ar = torch.full((N,), Rn, device=new.device, dtype=torch.long) if new_active_ranks is None else new_active_ranks.to(new.device).long().clamp(0, Rn)
#     spec = spectral_shape_similarity_matrix(
#         old_spectral_curve_means,
#         new_spectral_curve_means,
#         old_spectral_curve_d1=old_spectral_curve_d1,
#         new_spectral_curve_d1=new_spectral_curve_d1,
#         old_spectral_curve_d2=old_spectral_curve_d2,
#         new_spectral_curve_d2=new_spectral_curve_d2,
#         old_reliability=old_spectral_reliability,
#         new_reliability=new_spectral_reliability,
#     )
#     losses, overlaps, targets = [], [], []
#     for o in range(O):
#         ro = int(old_ar[o].detach().cpu().item())
#         if ro <= 0:
#             continue
#         Uo = old[o, :, :ro]
#         for n in range(N):
#             rn = int(new_ar[n].detach().cpu().item())
#             if rn <= 0:
#                 continue
#             Un = new[n, :, :rn]
#             denom = float(max(min(ro, rn), 1))
#             overlap = (Uo.t() @ Un).pow(2).sum() / denom
#             sim = spec[o, n].to(device=new.device, dtype=new.dtype) if spec.shape == (O, N) else torch.tensor(0.0, device=new.device, dtype=new.dtype)
#             target = torch.tensor(float(target_overlap), device=new.device, dtype=new.dtype) - float(spectral_margin_strength) * sim
#             target = target.clamp(min=float(min_target_overlap), max=float(target_overlap))
#             w = 1.0
#             if reliability is not None and torch.is_tensor(reliability) and reliability.numel() > o:
#                 rho = reliability.to(device=new.device, dtype=new.dtype).flatten()[o].clamp(0.05, 1.0)
#                 w = float((2.0 - rho).detach().cpu().item())
#             overlaps.append(overlap)
#             targets.append(target)
#             losses.append(float(w) * F.relu(overlap - target).pow(2))
#     if not losses:
#         z = safe_zero_like(ref)
#         return {"total": z, "mean_overlap": z, "max_overlap": z, "mean_target": z, "pair_count": z} if return_parts else z
#     ov = torch.stack(overlaps)
#     tg = torch.stack(targets)
#     loss = torch.stack(losses).mean()
#     if return_parts:
#         return {
#             "total": loss,
#             "mean_overlap": ov.mean().detach(),
#             "max_overlap": ov.max().detach(),
#             "mean_target": tg.mean().detach(),
#             "pair_count": torch.tensor(float(ov.numel()), device=new.device, dtype=new.dtype),
#         }
#     return loss


# def descriptor_subspace_collision_loss(
#     old_bases: Optional[torch.Tensor],
#     new_bases: Optional[torch.Tensor],
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     target_overlap: float = 0.35,
#     reliability: Optional[torch.Tensor] = None,
#     old_spectral_curve_means: Optional[torch.Tensor] = None,
#     new_spectral_curve_means: Optional[torch.Tensor] = None,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_spectral_reliability: Optional[torch.Tensor] = None,
#     new_spectral_reliability: Optional[torch.Tensor] = None,
#     spectral_margin_strength: float = 0.20,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Compatibility name now maps to spectral-aware SRGP separation."""
#     return spectral_aware_subspace_separation_loss(
#         old_bases,
#         new_bases,
#         old_active_ranks=old_active_ranks,
#         new_active_ranks=new_active_ranks,
#         old_spectral_curve_means=old_spectral_curve_means,
#         new_spectral_curve_means=new_spectral_curve_means,
#         old_spectral_curve_d1=old_spectral_curve_d1,
#         new_spectral_curve_d1=new_spectral_curve_d1,
#         old_spectral_curve_d2=old_spectral_curve_d2,
#         new_spectral_curve_d2=new_spectral_curve_d2,
#         old_spectral_reliability=old_spectral_reliability,
#         new_spectral_reliability=new_spectral_reliability,
#         target_overlap=target_overlap,
#         spectral_margin_strength=spectral_margin_strength,
#         reliability=reliability,
#         return_parts=return_parts,
#     )


# class SpectralResidualConsistencyLoss(nn.Module):
#     """Optional SRGP spectral identity loss for real HSI new samples.

#     Use it only on real samples with physical spectra.  It should not be applied
#     to synthetic replay features because those features do not have spectra.
#     """
#     def __init__(
#         self,
#         weight: float = 1.0,
#         derivative_weight: float = 0.50,
#         second_derivative_weight: float = 0.25,
#         variance_floor: float = 1e-4,
#         require_physical_summary: bool = True,
#     ) -> None:
#         super().__init__()
#         self.weight = float(weight)
#         self.derivative_weight = float(derivative_weight)
#         self.second_derivative_weight = float(second_derivative_weight)
#         self.variance_floor = float(variance_floor)
#         self.require_physical_summary = bool(require_physical_summary)

#     def forward(
#         self,
#         spectral_summary: Optional[torch.Tensor],
#         labels: Optional[torch.Tensor],
#         *,
#         spectral_curve_means: Optional[torch.Tensor],
#         spectral_curve_vars: Optional[torch.Tensor],
#         spectral_curve_d1: Optional[torch.Tensor] = None,
#         spectral_curve_d2: Optional[torch.Tensor] = None,
#         spectral_shape_reliability: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         ref = spectral_summary if torch.is_tensor(spectral_summary) else spectral_curve_means
#         if spectral_summary is None or labels is None or not torch.is_tensor(spectral_summary) or spectral_summary.numel() == 0:
#             z = safe_zero_like(ref)
#             return {"total": z, "spectral": z, "active": z}
#         if self.require_physical_summary and not bool(spectral_summary_is_physical):
#             z = spectral_summary.sum() * 0.0
#             return {"total": z, "spectral": z, "active": z}
#         parts = spectral_residual_energy_matrix(
#             spectral_summary,
#             spectral_curve_means,
#             spectral_curve_vars,
#             spectral_curve_d1=spectral_curve_d1,
#             spectral_curve_d2=spectral_curve_d2,
#             spectral_shape_reliability=spectral_shape_reliability,
#             sample_counts=sample_counts,
#             variance_floor=self.variance_floor,
#             derivative_weight=self.derivative_weight,
#             second_derivative_weight=self.second_derivative_weight,
#             spectral_summary_is_physical=bool(spectral_summary_is_physical),
#             require_physical_summary=self.require_physical_summary,
#             return_parts=True,
#         )
#         e = parts["spectral_energy"]
#         y = labels.to(device=e.device).long().flatten()
#         if y.numel() != e.size(0):
#             raise ValueError(f"labels/spectral batch mismatch: {y.numel()} vs {e.size(0)}")
#         if int(y.min().item()) < 0 or int(y.max().item()) >= e.size(1):
#             raise ValueError("labels out of spectral descriptor class range")
#         own = e.gather(1, y.view(-1, 1)).squeeze(1)
#         finite = torch.isfinite(own) & (own < 1e5)
#         loss = own[finite].mean() if bool(finite.any().item()) else e.sum() * 0.0
#         return {"total": float(self.weight) * loss, "spectral": loss.detach(), "active": torch.tensor(float(finite.sum().item()), device=e.device, dtype=e.dtype)}

# # -----------------------------------------------------------------------------
# # Unified SRGP / RSGI objective wrappers
# # -----------------------------------------------------------------------------

# def _scalar_from_parts(parts: Dict[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor:
#     value = parts.get(key, None) if isinstance(parts, dict) else None
#     if torch.is_tensor(value):
#         return value
#     return safe_zero_like(ref)


# def spectral_residual_prospective_geometry_loss(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     key_features: Optional[torch.Tensor] = None,
#     band_summary: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     # global / component weights
#     weight: float = 1.0,
#     gics_weight: float = 0.20,
#     pgr_weight: float = 0.10,
#     spectral_shape_weight: float = 0.05,
#     # GICS knobs
#     gics_temperature: float = 0.07,
#     # PGR knobs
#     pgr_compact_weight: float = 0.15,
#     pgr_center_weight: float = 0.20,
#     pgr_subspace_weight: float = 0.10,
#     pgr_band_weight: float = 0.05,
#     pgr_volume_weight: float = 0.05,
#     pgr_center_margin: float = 1.05,
#     pgr_max_band_similarity: float = 0.75,
#     pgr_max_class_variance: float = 0.75,
#     subspace_rank: int = 3,
#     min_class_samples: int = 3,
#     subspace_min_samples: int = 6,
#     # spectral-shape knobs
#     max_spectral_shape_similarity: float = 0.75,
#     spectral_shape_risk_weight: float = 1.0,
#     require_physical_summary: bool = True,
#     return_parts: bool = True,
#     **kwargs: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Unified base-phase SRPGR objective.

#     This wrapper is the paper-level base regularizer:

#         SRPGR = compact/class separation + prospective geometry reserve
#                 + physical spectral-shape reserve

#     CE remains outside this function in the base trainer.  GICS and PGR are kept
#     as internal components so old imports still work, but this function exposes
#     the single coherent SRGP base objective.
#     """
#     if features is None or not torch.is_tensor(features) or features.numel() == 0:
#         z = safe_zero_like(features)
#         out = {
#             "total": z, "srpgr": z, "gics": z, "pgr": z, "spectral_shape": z,
#             "compact": z, "center": z, "subspace": z, "band": z, "volume": z,
#             "spectral_shape_active": z,
#         }
#         return out if return_parts else z

#     gics = base_geometry_involved_contrastive_loss(
#         features,
#         labels,
#         key_features=key_features,
#         weight=float(gics_weight),
#         temperature=float(gics_temperature),
#         return_parts=True,
#         **kwargs,
#     )
#     pgr = prospective_geometry_reserve_loss(
#         features,
#         labels,
#         band_summary=band_summary,
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
#         return_parts=True,
#         **kwargs,
#     )
#     spec_raw = spectral_shape_discrimination_loss(
#         spectral_summary,
#         labels,
#         features=features,
#         spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         require_physical_summary=bool(require_physical_summary),
#         min_samples=int(min_class_samples),
#         max_shape_similarity=float(max_spectral_shape_similarity),
#         risk_weight=float(spectral_shape_risk_weight),
#         return_parts=True,
#     )
#     spec_total = float(spectral_shape_weight) * _scalar_from_parts(spec_raw, "total", features)
#     total = float(weight) * (_scalar_from_parts(gics, "total", features) + _scalar_from_parts(pgr, "total", features) + spec_total)

#     if not return_parts:
#         return total
#     out = {
#         "total": total,
#         "srpgr": (total / max(float(weight), 1e-12)).detach(),
#         "gics": _scalar_from_parts(gics, "gics", features).detach(),
#         "gics_total": _scalar_from_parts(gics, "total", features).detach(),
#         "pgr": _scalar_from_parts(pgr, "pgr", features).detach(),
#         "pgr_total": _scalar_from_parts(pgr, "total", features).detach(),
#         "compact": _scalar_from_parts(pgr, "compact", features).detach(),
#         "center": _scalar_from_parts(pgr, "center", features).detach(),
#         "subspace": _scalar_from_parts(pgr, "subspace", features).detach(),
#         "band": _scalar_from_parts(pgr, "band", features).detach(),
#         "volume": _scalar_from_parts(pgr, "volume", features).detach(),
#         "spectral_shape": spec_total.detach(),
#         "spectral_shape_raw": _scalar_from_parts(spec_raw, "total", features).detach(),
#         "spectral_shape_mean_similarity": _scalar_from_parts(spec_raw, "mean_similarity", features).detach(),
#         "spectral_shape_pair_count": _scalar_from_parts(spec_raw, "pair_count", features).detach(),
#         "spectral_shape_active": torch.tensor(
#             float(bool(spectral_summary_is_physical) and (spectral_summary is not None) and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0),
#             device=features.device,
#             dtype=features.dtype,
#         ),
#     }
#     return out


# # Alias names used in notes/trainer variants.
# spectral_residual_prospective_reserve_loss = spectral_residual_prospective_geometry_loss
# srpgr_loss = spectral_residual_prospective_geometry_loss


# def old_geometry_invasion_loss(
#     features: Optional[torch.Tensor],
#     labels: Optional[torch.Tensor],
#     *,
#     old_class_count: int,
#     means: Optional[torch.Tensor],
#     bases: Optional[torch.Tensor],
#     variances: Optional[torch.Tensor],
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     margin: float = 0.30,
#     variance_floor: float = 1e-4,
#     reliability_energy_weight: float = 0.05,
#     residual_variance_scale: float = 0.75,
#     normalize_by_dim: bool = True,
#     use_logdet_energy: bool = True,
#     logdet_energy_weight: float = 0.05,
#     return_parts: bool = True,
#     **kwargs: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Functional wrapper around :class:`OldGeometryInvasionLoss`.

#     Keeps stale trainer variants aligned with the RSGI objective name while using
#     the same covariance-consistent energy as the classifier.
#     """
#     obj = OldGeometryInvasionLoss(
#         margin=float(margin),
#         variance_floor=float(variance_floor),
#         reliability_energy_weight=float(reliability_energy_weight),
#         residual_variance_scale=float(residual_variance_scale),
#         normalize_by_dim=bool(normalize_by_dim),
#         use_logdet_energy=bool(use_logdet_energy),
#         logdet_energy_weight=float(logdet_energy_weight),
#     )
#     out = obj(
#         features,
#         labels,
#         old_class_count=int(old_class_count),
#         means=means,
#         bases=bases,
#         variances=variances,
#         active_ranks=active_ranks,
#         reliability=reliability,
#         sample_counts=sample_counts,
#         **kwargs,
#     )
#     return out if return_parts else out["total"]


# def risk_aware_old_new_subspace_separation_loss(
#     old_bases: Optional[torch.Tensor],
#     new_bases: Optional[torch.Tensor],
#     *,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     old_spectral_curve_means: Optional[torch.Tensor] = None,
#     new_spectral_curve_means: Optional[torch.Tensor] = None,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_spectral_reliability: Optional[torch.Tensor] = None,
#     new_spectral_reliability: Optional[torch.Tensor] = None,
#     target_overlap: float = 0.35,
#     spectral_margin_strength: float = 0.20,
#     reliability: Optional[torch.Tensor] = None,
#     return_parts: bool = True,
#     **_: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """RSGI old/new descriptor-separation wrapper."""
#     return spectral_aware_subspace_separation_loss(
#         old_bases,
#         new_bases,
#         old_active_ranks=old_active_ranks,
#         new_active_ranks=new_active_ranks,
#         old_spectral_curve_means=old_spectral_curve_means,
#         new_spectral_curve_means=new_spectral_curve_means,
#         old_spectral_curve_d1=old_spectral_curve_d1,
#         new_spectral_curve_d1=new_spectral_curve_d1,
#         old_spectral_curve_d2=old_spectral_curve_d2,
#         new_spectral_curve_d2=new_spectral_curve_d2,
#         old_spectral_reliability=old_spectral_reliability,
#         new_spectral_reliability=new_spectral_reliability,
#         target_overlap=float(target_overlap),
#         spectral_margin_strength=float(spectral_margin_strength),
#         reliability=reliability,
#         return_parts=return_parts,
#     )


# def risk_aware_srgp_insertion_loss(
#     *,
#     energy: Optional[torch.Tensor] = None,
#     labels: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     old_class_count: int = 0,
#     old_bases: Optional[torch.Tensor] = None,
#     new_bases: Optional[torch.Tensor] = None,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     old_spectral_curve_means: Optional[torch.Tensor] = None,
#     new_spectral_curve_means: Optional[torch.Tensor] = None,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_spectral_reliability: Optional[torch.Tensor] = None,
#     new_spectral_reliability: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     logit_scale: float = 8.0,
#     ce_weight: float = 1.0,
#     rank_weight: float = 0.10,
#     invasion_weight: float = 0.10,
#     subspace_weight: float = 0.10,
#     rank_margin: float = 0.25,
#     old_new_margin: float = 0.30,
#     target_overlap: float = 0.35,
#     return_parts: bool = True,
#     **_: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Unified incremental RSGI loss wrapper.

#     The trainer may still compute the terms manually, but this function provides
#     a single consistent objective for variants:

#         RSGI = CE/rank energy objective + old/new invasion + descriptor separation.

#     It never samples or stores exemplars; it operates only on energy matrices and
#     compact descriptor tensors.
#     """
#     ref = energy if torch.is_tensor(energy) else (new_bases if torch.is_tensor(new_bases) else old_bases)
#     z = safe_zero_like(ref)
#     ce = rank = invasion = sep = z
#     violation = z.detach()

#     if energy is not None and labels is not None and torch.is_tensor(energy) and energy.numel() > 0:
#         obj = valid_energy_objective(
#             energy,
#             labels,
#             sample_counts=sample_counts,
#             logit_scale=float(logit_scale),
#             rank_margin=float(rank_margin),
#         )
#         ce = obj["ce"]
#         rank = obj["rank"]
#         violation = obj["violation_rate"].detach()
#         if int(old_class_count) > 0 and int(old_class_count) < energy.size(1):
#             y = labels.to(device=energy.device).long().flatten()
#             valid_mask = _valid_class_mask_from_counts(sample_counts, energy.size(1), energy.device)
#             masked = energy.masked_fill(~valid_mask.view(1, -1), float("inf"))
#             own = masked.gather(1, y.view(-1, 1)).squeeze(1)
#             losses = []
#             new_mask = y >= int(old_class_count)
#             old_mask = y < int(old_class_count)
#             if bool(new_mask.any().item()):
#                 nearest_old = masked[new_mask, :int(old_class_count)].min(dim=1).values
#                 losses.append(F.relu(own[new_mask] + float(old_new_margin) - nearest_old))
#             if bool(old_mask.any().item()):
#                 nearest_new = masked[old_mask, int(old_class_count):].min(dim=1).values
#                 losses.append(F.relu(own[old_mask] + float(old_new_margin) - nearest_new))
#             if losses:
#                 invasion = torch.cat(losses).mean()

#     sep_out = risk_aware_old_new_subspace_separation_loss(
#         old_bases,
#         new_bases,
#         old_active_ranks=old_active_ranks,
#         new_active_ranks=new_active_ranks,
#         old_spectral_curve_means=old_spectral_curve_means,
#         new_spectral_curve_means=new_spectral_curve_means,
#         old_spectral_curve_d1=old_spectral_curve_d1,
#         new_spectral_curve_d1=new_spectral_curve_d1,
#         old_spectral_curve_d2=old_spectral_curve_d2,
#         new_spectral_curve_d2=new_spectral_curve_d2,
#         old_spectral_reliability=old_spectral_reliability,
#         new_spectral_reliability=new_spectral_reliability,
#         target_overlap=float(target_overlap),
#         reliability=reliability,
#         return_parts=True,
#     )
#     sep = _scalar_from_parts(sep_out, "total", ref if torch.is_tensor(ref) else z)
#     total = float(ce_weight) * ce + float(rank_weight) * rank + float(invasion_weight) * invasion + float(subspace_weight) * sep

#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "ce": ce.detach(),
#         "rank": rank.detach(),
#         "invasion": invasion.detach(),
#         "subspace": sep.detach(),
#         "violation_rate": violation.detach(),
#         "mean_overlap": _scalar_from_parts(sep_out, "mean_overlap", ref if torch.is_tensor(ref) else z).detach(),
#         "max_overlap": _scalar_from_parts(sep_out, "max_overlap", ref if torch.is_tensor(ref) else z).detach(),
#         "pair_count": _scalar_from_parts(sep_out, "pair_count", ref if torch.is_tensor(ref) else z).detach(),
#     }


# # Additional compatibility names used in iterative trainer drafts.
# rsgi_loss = risk_aware_srgp_insertion_loss
# risk_aware_spectral_residual_geometry_insertion_loss = risk_aware_srgp_insertion_loss
# spectral_aware_descriptor_subspace_collision_loss = spectral_aware_subspace_separation_loss





# # -----------------------------------------------------------------------------
# # G2RPA additions: geometry-gated residual plastic adapter losses
# # -----------------------------------------------------------------------------
# # These losses are intentionally feature-geometry losses.  They do not require
# # old raw patches, an old model, KD targets, or stored old feature samples.  They
# # operate on:
# #   - synthetic old features sampled from the frozen GeometryBank,
# #   - their adapter outputs,
# #   - real new-sample adapted features,
# #   - the current compact GeometryBank descriptors.
# #
# # Contract for the model/trainer:
# #   z_base  = canonical projected feature before the residual adapter
# #   z_adapt = canonical projected feature after the residual adapter
# #   gate    = geometry-gated adapter gate in [0, 1]
# #
# # Old synthetic replay MUST pass through the adapter, then these losses enforce:
# #   gate_old -> 0, z_old_adapt -> z_old_base, and old energy remains correct.
# # This is the mechanism that prevents the architecture-level plasticity from
# # destroying the incremental phase.


# def _as_gate_column(gate: Optional[torch.Tensor], ref: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
#     """Return gate as [B,1] clamped to [0,1], or None when absent."""
#     if gate is None or not torch.is_tensor(gate) or gate.numel() == 0:
#         return None
#     g = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).float()
#     if ref is not None and torch.is_tensor(ref):
#         g = g.to(device=ref.device, dtype=ref.dtype)
#     if g.dim() == 1:
#         g = g.view(-1, 1)
#     elif g.dim() > 2:
#         g = g.flatten(1).mean(dim=1, keepdim=True)
#     elif g.dim() == 2 and g.size(1) != 1:
#         g = g.mean(dim=1, keepdim=True)
#     return g.clamp(0.0, 1.0)


# def adapter_delta_norm_loss(
#     z_base: Optional[torch.Tensor],
#     z_adapt: Optional[torch.Tensor],
#     *,
#     gate: Optional[torch.Tensor] = None,
#     detach_base: bool = True,
#     squared: bool = True,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Bound residual adapter movement in canonical geometry space.

#     This loss should be applied to both real new samples and synthetic old replay.
#     For old replay it is usually weighted strongly.  For real new samples it is a
#     soft regularizer that prevents the adapter from rewriting the whole z-space.
#     """
#     ref = z_adapt if torch.is_tensor(z_adapt) else z_base
#     if z_base is None or z_adapt is None or not torch.is_tensor(z_base) or not torch.is_tensor(z_adapt):
#         z = safe_zero_like(ref)
#         return {"total": z, "delta": z, "mean_norm": z, "gate_mean": z} if return_parts else z
#     if z_base.numel() == 0 or z_adapt.numel() == 0:
#         z = safe_zero_like(ref)
#         return {"total": z, "delta": z, "mean_norm": z, "gate_mean": z} if return_parts else z
#     if z_base.shape != z_adapt.shape:
#         raise ValueError(f"adapter_delta_norm_loss shape mismatch: {tuple(z_base.shape)} vs {tuple(z_adapt.shape)}")
#     base = z_base.detach() if detach_base else z_base
#     delta = z_adapt - base
#     per = delta.pow(2).mean(dim=1) if squared else delta.norm(p=2, dim=1) / max(float(delta.size(1)) ** 0.5, 1.0)
#     loss = per.mean()
#     g = _as_gate_column(gate, z_adapt)
#     gate_mean = g.mean() if g is not None and g.numel() > 0 else z_adapt.sum() * 0.0
#     if return_parts:
#         return {
#             "total": loss,
#             "delta": loss.detach(),
#             "mean_norm": delta.norm(p=2, dim=1).mean().detach(),
#             "gate_mean": gate_mean.detach(),
#         }
#     return loss


# def gate_old_suppression_loss(
#     gate_old: Optional[torch.Tensor],
#     *,
#     target: float = 0.05,
#     squared: bool = True,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Force adapter gate to stay closed on old synthetic geometry replay."""
#     if gate_old is None or not torch.is_tensor(gate_old) or gate_old.numel() == 0:
#         z = safe_zero_like(gate_old)
#         return {"total": z, "gate": z, "mean_gate": z, "violation_rate": z} if return_parts else z
#     g = _as_gate_column(gate_old, gate_old)
#     assert g is not None
#     if squared:
#         loss_vec = F.relu(g - float(target)).pow(2)
#     else:
#         loss_vec = F.relu(g - float(target))
#     loss = loss_vec.mean()
#     violation = (g > float(target)).float().mean()
#     if return_parts:
#         return {
#             "total": loss,
#             "gate": loss.detach(),
#             "mean_gate": g.mean().detach(),
#             "violation_rate": violation.detach(),
#         }
#     return loss


# def gate_new_utilization_loss(
#     gate_new: Optional[torch.Tensor],
#     *,
#     target: float = 0.25,
#     max_target: float = 0.75,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Encourage some plasticity for real new samples without opening all gates.

#     The lower target prevents a dead adapter.  The upper target prevents the
#     adapter from becoming a global feature rewrite.
#     """
#     if gate_new is None or not torch.is_tensor(gate_new) or gate_new.numel() == 0:
#         z = safe_zero_like(gate_new)
#         return {"total": z, "under": z, "over": z, "mean_gate": z} if return_parts else z
#     g = _as_gate_column(gate_new, gate_new)
#     assert g is not None
#     mean_gate = g.mean()
#     under = F.relu(float(target) - mean_gate).pow(2)
#     over = F.relu(mean_gate - float(max_target)).pow(2)
#     total = under + over
#     if return_parts:
#         return {"total": total, "under": under.detach(), "over": over.detach(), "mean_gate": mean_gate.detach()}
#     return total


# def old_adapter_invariance_loss(
#     z_old_base: Optional[torch.Tensor],
#     z_old_adapt: Optional[torch.Tensor],
#     labels: Optional[torch.Tensor] = None,
#     *,
#     gate_old: Optional[torch.Tensor] = None,
#     means: Optional[torch.Tensor] = None,
#     bases: Optional[torch.Tensor] = None,
#     variances: Optional[torch.Tensor] = None,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     reliability_energy_weight: float = 0.05,
#     residual_variance_scale: float = 0.75,
#     normalize_by_dim: bool = True,
#     use_logdet_energy: bool = True,
#     logdet_energy_weight: float = 0.05,
#     logit_scale: float = 8.0,
#     margin: float = 0.25,
#     delta_weight: float = 1.0,
#     energy_weight: float = 0.25,
#     margin_weight: float = 0.25,
#     gate_weight: float = 0.50,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Protect old geometry when using the geometry-gated residual adapter.

#     Use on synthetic old replay after passing replay features through
#     ``model.adapt_projected_features``.  The function enforces three constraints:
#       1. old adapted features stay close to original old replay features;
#       2. old adapted features remain low-energy under their own old class;
#       3. old adapter gates stay closed.
#     """
#     ref = z_old_adapt if torch.is_tensor(z_old_adapt) else z_old_base
#     if z_old_base is None or z_old_adapt is None or not torch.is_tensor(z_old_base) or not torch.is_tensor(z_old_adapt):
#         z = safe_zero_like(ref)
#         out = {"total": z, "delta": z, "energy": z, "margin": z, "gate": z, "mean_gate": z, "active": z, "accuracy": z}
#         return out if return_parts else z
#     if z_old_base.numel() == 0 or z_old_adapt.numel() == 0:
#         z = safe_zero_like(ref)
#         out = {"total": z, "delta": z, "energy": z, "margin": z, "gate": z, "mean_gate": z, "active": z, "accuracy": z}
#         return out if return_parts else z

#     delta_parts = adapter_delta_norm_loss(z_old_base, z_old_adapt, gate=gate_old, return_parts=True)
#     delta_loss = delta_parts["total"]
#     gate_parts = gate_old_suppression_loss(gate_old, return_parts=True)
#     gate_loss = gate_parts["total"]
#     mean_gate = gate_parts["mean_gate"]

#     energy_loss = z_old_adapt.sum() * 0.0
#     margin_loss = z_old_adapt.sum() * 0.0
#     acc = z_old_adapt.sum().detach() * 0.0
#     active = torch.tensor(float(z_old_adapt.size(0)), device=z_old_adapt.device, dtype=z_old_adapt.dtype)

#     if (
#         labels is not None
#         and torch.is_tensor(labels)
#         and means is not None and bases is not None and variances is not None
#         and torch.is_tensor(means) and torch.is_tensor(bases) and torch.is_tensor(variances)
#         and means.numel() > 0 and bases.numel() > 0 and variances.numel() > 0
#     ):
#         y = labels.to(device=z_old_adapt.device).long().flatten()
#         if y.numel() != z_old_adapt.size(0):
#             raise ValueError(f"old_adapter_invariance_loss labels/features mismatch: {y.numel()} vs {z_old_adapt.size(0)}")
#         bank_means = means.to(device=z_old_adapt.device, dtype=z_old_adapt.dtype)
#         bank_bases = bases.to(device=z_old_adapt.device, dtype=z_old_adapt.dtype)
#         bank_variances = variances.to(device=z_old_adapt.device, dtype=z_old_adapt.dtype)
#         if sample_counts is None or not torch.is_tensor(sample_counts) or sample_counts.numel() != bank_means.size(0):
#             sample_counts = torch.ones((bank_means.size(0),), device=z_old_adapt.device, dtype=z_old_adapt.dtype)
#         energy = geometry_energy_matrix(
#             z_old_adapt,
#             bank_means,
#             bank_bases,
#             bank_variances,
#             active_ranks=active_ranks,
#             reliability=reliability,
#             sample_counts=sample_counts,
#             variance_floor=float(variance_floor),
#             reliability_energy_weight=float(reliability_energy_weight),
#             residual_variance_scale=float(residual_variance_scale),
#             normalize_by_dim=bool(normalize_by_dim),
#             use_logdet_energy=bool(use_logdet_energy),
#             logdet_energy_weight=float(logdet_energy_weight),
#         )
#         if int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1):
#             raise ValueError("old_adapter_invariance_loss labels outside GeometryBank class dimension")
#         true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#         energy_loss = torch.log1p(true_e.clamp_min(0.0)).mean()
#         logits, _, valid = relative_energy_logits(
#             energy,
#             sample_counts=sample_counts,
#             logit_scale=float(logit_scale),
#         )
#         pred = logits.argmax(dim=1)
#         acc = (pred == y).float().mean().detach()
#         true_mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
#         neg_mask = valid.view(1, -1).expand_as(energy) & (~true_mask)
#         if bool(neg_mask.any(dim=1).all().item()):
#             nearest_wrong = energy.masked_fill(~neg_mask, float("inf")).min(dim=1).values
#             margin_loss = F.relu(true_e + float(margin) - nearest_wrong).mean()

#     total = (
#         float(delta_weight) * delta_loss
#         + float(energy_weight) * energy_loss
#         + float(margin_weight) * margin_loss
#         + float(gate_weight) * gate_loss
#     )
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "delta": delta_loss.detach(),
#         "energy": energy_loss.detach(),
#         "margin": margin_loss.detach(),
#         "gate": gate_loss.detach(),
#         "mean_gate": mean_gate.detach(),
#         "active": active.detach(),
#         "accuracy": acc.detach(),
#     }


# class GeometryGatedAdapterLoss(nn.Module):
#     """Unified regularizer for G2RPA incremental training.

#     This class combines:
#       - old replay invariance/protection,
#       - real-new adapter movement control,
#       - new gate utilization.

#     It deliberately does not compute the main real-new CE or replay CE.  The
#     trainer should still compute those using the same GeometryEnergyClassifier
#     used at evaluation.  This module only makes architecture-level plasticity
#     safe for NECIL.
#     """

#     def __init__(
#         self,
#         old_delta_weight: float = 1.0,
#         old_energy_weight: float = 0.25,
#         old_margin_weight: float = 0.25,
#         old_gate_weight: float = 0.50,
#         new_delta_weight: float = 0.10,
#         new_gate_weight: float = 0.05,
#         new_gate_target: float = 0.25,
#         new_gate_max_target: float = 0.75,
#         margin: float = 0.25,
#         **energy_kwargs: object,
#     ) -> None:
#         super().__init__()
#         self.old_delta_weight = float(old_delta_weight)
#         self.old_energy_weight = float(old_energy_weight)
#         self.old_margin_weight = float(old_margin_weight)
#         self.old_gate_weight = float(old_gate_weight)
#         self.new_delta_weight = float(new_delta_weight)
#         self.new_gate_weight = float(new_gate_weight)
#         self.new_gate_target = float(new_gate_target)
#         self.new_gate_max_target = float(new_gate_max_target)
#         self.margin = float(margin)
#         self.energy_kwargs = dict(energy_kwargs)

#     def forward(
#         self,
#         *,
#         z_old_base: Optional[torch.Tensor] = None,
#         z_old_adapt: Optional[torch.Tensor] = None,
#         y_old: Optional[torch.Tensor] = None,
#         gate_old: Optional[torch.Tensor] = None,
#         z_new_base: Optional[torch.Tensor] = None,
#         z_new_adapt: Optional[torch.Tensor] = None,
#         gate_new: Optional[torch.Tensor] = None,
#         means: Optional[torch.Tensor] = None,
#         bases: Optional[torch.Tensor] = None,
#         variances: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         ref = z_new_adapt if torch.is_tensor(z_new_adapt) else z_old_adapt
#         z = safe_zero_like(ref)

#         old = old_adapter_invariance_loss(
#             z_old_base,
#             z_old_adapt,
#             y_old,
#             gate_old=gate_old,
#             means=means,
#             bases=bases,
#             variances=variances,
#             active_ranks=active_ranks,
#             reliability=reliability,
#             sample_counts=sample_counts,
#             margin=self.margin,
#             delta_weight=self.old_delta_weight,
#             energy_weight=self.old_energy_weight,
#             margin_weight=self.old_margin_weight,
#             gate_weight=self.old_gate_weight,
#             return_parts=True,
#             **self.energy_kwargs,
#         )
#         new_delta = adapter_delta_norm_loss(z_new_base, z_new_adapt, gate=gate_new, return_parts=True)
#         new_gate = gate_new_utilization_loss(
#             gate_new,
#             target=self.new_gate_target,
#             max_target=self.new_gate_max_target,
#             return_parts=True,
#         )
#         total = old["total"] + float(self.new_delta_weight) * new_delta["total"] + float(self.new_gate_weight) * new_gate["total"]
#         return {
#             "total": total,
#             "old_total": old["total"].detach(),
#             "old_delta": old["delta"].detach(),
#             "old_energy": old["energy"].detach(),
#             "old_margin": old["margin"].detach(),
#             "old_gate": old["gate"].detach(),
#             "old_mean_gate": old["mean_gate"].detach(),
#             "old_adapter_acc": old["accuracy"].detach(),
#             "new_delta": new_delta["total"].detach(),
#             "new_gate": new_gate["total"].detach(),
#             "new_mean_gate": new_gate["mean_gate"].detach(),
#         }


# def geometry_gated_adapter_loss(*args: object, **kwargs: object) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """Functional compatibility wrapper around :class:`GeometryGatedAdapterLoss`.

#     Constructor arguments are separated from forward arguments so trainers can
#     call this as a plain function without manually instantiating the module.
#     """
#     init_keys = {
#         "old_delta_weight", "old_energy_weight", "old_margin_weight", "old_gate_weight",
#         "new_delta_weight", "new_gate_weight", "new_gate_target", "new_gate_max_target",
#         "margin", "variance_floor", "reliability_energy_weight", "residual_variance_scale",
#         "normalize_by_dim", "use_logdet_energy", "logdet_energy_weight", "logit_scale",
#     }
#     init_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in init_keys}
#     return GeometryGatedAdapterLoss(**init_kwargs)(*args, **kwargs)


# # Compatibility aliases for trainer variants.
# g2rpa_loss = geometry_gated_adapter_loss
# geometry_gated_residual_plasticity_loss = geometry_gated_adapter_loss
# old_geometry_adapter_invariance_loss = old_adapter_invariance_loss


# __all__ = [name for name in globals().keys() if not name.startswith("_")]



# # =============================================================================
# # SCB-GR corrections: boundary replay + geometry-state admission
# # =============================================================================
# # These definitions intentionally appear at the end of the file so they override
# # earlier compatibility aliases without touching the rest of the training stack.
# # They keep the current GeometryBank API unchanged.


# def _neutralize_unreliable_class_energy(
#     energy: torch.Tensor,
#     reliability: Optional[torch.Tensor],
#     valid_mask: Optional[torch.Tensor] = None,
#     *,
#     reliability_floor: float = 0.05,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Make low-reliability class-specific energy neutral instead of attractive.

#     Lower energy means a better match.  Therefore multiplying energy by a small
#     reliability value is wrong: it makes unreliable classes easier to select.
#     This function mixes unreliable class energy toward a per-sample neutral
#     baseline, so weak spectral descriptors neither help nor strongly hurt.
#     """
#     if energy.dim() != 2:
#         raise ValueError(f"energy must be [B,C], got {tuple(energy.shape)}")
#     B, C = energy.shape
#     if reliability is None or not torch.is_tensor(reliability) or reliability.numel() != C:
#         rel = torch.ones((C,), device=energy.device, dtype=energy.dtype)
#         return energy, rel
#     rel = torch.nan_to_num(
#         reliability.to(device=energy.device, dtype=energy.dtype).flatten(),
#         nan=float(reliability_floor),
#         posinf=1.0,
#         neginf=float(reliability_floor),
#     ).clamp(float(reliability_floor), 1.0)
#     if valid_mask is not None and torch.is_tensor(valid_mask) and valid_mask.numel() == C and bool(valid_mask.any().item()):
#         valid = valid_mask.to(device=energy.device).bool().flatten()
#         neutral = energy.masked_fill(~valid.view(1, C), 0.0).sum(dim=1, keepdim=True) / valid.to(dtype=energy.dtype).sum().clamp_min(1.0)
#     else:
#         neutral = energy.mean(dim=1, keepdim=True)
#     mixed = rel.view(1, C) * energy + (1.0 - rel.view(1, C)) * neutral.detach()
#     return mixed, rel


# def spectral_residual_energy_matrix(
#     spectral_summary: Optional[torch.Tensor],
#     spectral_curve_means: Optional[torch.Tensor],
#     spectral_curve_vars: Optional[torch.Tensor],
#     *,
#     spectral_curve_d1: Optional[torch.Tensor] = None,
#     spectral_curve_d2: Optional[torch.Tensor] = None,
#     spectral_shape_reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     variance_floor: float = 1e-4,
#     derivative_weight: float = 0.50,
#     second_derivative_weight: float = 0.25,
#     normalize_by_components: bool = True,
#     spectral_summary_is_physical: bool = False,
#     require_physical_summary: bool = True,
#     invalid_class_energy: float = 1e6,
#     return_parts: bool = False,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """SRGP spectral-residual energy with reliability-neutral gating.

#     It is dormant unless physical spectra and bank spectral-shape descriptors are
#     available.  Low spectral reliability does not lower a class energy; it moves
#     that spectral score toward a neutral per-sample baseline.
#     """
#     ref = spectral_summary if torch.is_tensor(spectral_summary) else spectral_curve_means
#     if ref is None or not torch.is_tensor(ref):
#         z = torch.tensor(0.0)
#         out = {"spectral_energy": z.view(1, 1) * 0.0, "spectral_active": z.view(()) * 0.0}
#         return out if return_parts else out["spectral_energy"]
#     if (
#         spectral_summary is None
#         or spectral_curve_means is None
#         or not torch.is_tensor(spectral_summary)
#         or not torch.is_tensor(spectral_curve_means)
#         or spectral_summary.numel() == 0
#         or spectral_curve_means.numel() == 0
#         or (bool(require_physical_summary) and not bool(spectral_summary_is_physical))
#     ):
#         z = ref.sum() * 0.0
#         out = {"spectral_energy": z.view(1, 1) * 0.0, "spectral_active": z.view(()) * 0.0}
#         return out if return_parts else out["spectral_energy"]

#     s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#     if s.dim() != 2:
#         raise ValueError(f"spectral_summary must be [B,S], got {tuple(s.shape)}")
#     means = torch.nan_to_num(spectral_curve_means.to(device=s.device, dtype=s.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#     if means.dim() != 2:
#         raise ValueError(f"spectral_curve_means must be [C,S], got {tuple(means.shape)}")
#     if means.size(1) != s.size(1):
#         raise ValueError(
#             f"spectral_summary width {s.size(1)} does not match bank spectral width {means.size(1)}. "
#             "Use raw/physical spectral summaries aligned with GeometryBank extraction."
#         )
#     B, S = s.shape
#     C = means.size(0)
#     if sample_counts is not None and torch.is_tensor(sample_counts) and sample_counts.numel() == C:
#         valid = _valid_class_mask_from_counts(sample_counts, C, s.device)
#     else:
#         valid = torch.ones((C,), device=s.device, dtype=torch.bool)

#     if spectral_curve_vars is not None and torch.is_tensor(spectral_curve_vars) and spectral_curve_vars.numel() > 0:
#         var = torch.nan_to_num(
#             spectral_curve_vars.to(device=s.device, dtype=s.dtype),
#             nan=float(variance_floor),
#             posinf=float(invalid_class_energy),
#             neginf=float(variance_floor),
#         )
#         if var.shape != means.shape:
#             raise ValueError(f"spectral_curve_vars shape {tuple(var.shape)} must match means {tuple(means.shape)}")
#         var = var.clamp_min(float(variance_floor))
#     else:
#         var = torch.full_like(means, float(variance_floor))

#     curve = ((s.unsqueeze(1) - means.unsqueeze(0)).pow(2) / var.unsqueeze(0)).mean(dim=-1)
#     d1_energy = torch.zeros((B, C), device=s.device, dtype=s.dtype)
#     d2_energy = torch.zeros((B, C), device=s.device, dtype=s.dtype)
#     d1, d2 = _spectral_derivatives(s)

#     if derivative_weight > 0.0 and d1.numel() > 0 and spectral_curve_d1 is not None and torch.is_tensor(spectral_curve_d1) and spectral_curve_d1.numel() > 0:
#         bank_d1 = torch.nan_to_num(spectral_curve_d1.to(device=s.device, dtype=s.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#         if bank_d1.shape == (C, S - 1):
#             var_d1 = (var[:, 1:] + var[:, :-1]).clamp_min(float(variance_floor))
#             d1_energy = ((d1.unsqueeze(1) - bank_d1.unsqueeze(0)).pow(2) / var_d1.unsqueeze(0)).mean(dim=-1)
#     if second_derivative_weight > 0.0 and d2.numel() > 0 and spectral_curve_d2 is not None and torch.is_tensor(spectral_curve_d2) and spectral_curve_d2.numel() > 0:
#         bank_d2 = torch.nan_to_num(spectral_curve_d2.to(device=s.device, dtype=s.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#         if bank_d2.shape == (C, S - 2):
#             var_d2 = (var[:, 2:] + 4.0 * var[:, 1:-1] + var[:, :-2]).clamp_min(float(variance_floor))
#             d2_energy = ((d2.unsqueeze(1) - bank_d2.unsqueeze(0)).pow(2) / var_d2.unsqueeze(0)).mean(dim=-1)

#     energy = curve + float(derivative_weight) * d1_energy + float(second_derivative_weight) * d2_energy
#     if bool(normalize_by_components):
#         energy = energy / max(1.0 + float(derivative_weight) + float(second_derivative_weight), 1e-8)

#     energy, rel = _neutralize_unreliable_class_energy(energy, spectral_shape_reliability, valid)
#     energy = torch.nan_to_num(energy, nan=0.0, posinf=float(invalid_class_energy), neginf=0.0).masked_fill(~valid.view(1, C), float(invalid_class_energy))

#     if return_parts:
#         return {
#             "spectral_energy": energy,
#             "spectral_curve_energy": curve.masked_fill(~valid.view(1, C), float(invalid_class_energy)),
#             "spectral_d1_energy": d1_energy.masked_fill(~valid.view(1, C), float(invalid_class_energy)),
#             "spectral_d2_energy": d2_energy.masked_fill(~valid.view(1, C), float(invalid_class_energy)),
#             "spectral_reliability": rel,
#             "valid_mask": valid,
#             "spectral_active": torch.tensor(1.0, device=s.device, dtype=s.dtype),
#         }
#     return energy


# def geometry_energy_matrix(
#     features: torch.Tensor,
#     means: torch.Tensor,
#     bases: torch.Tensor,
#     variances: torch.Tensor,
#     *,
#     active_ranks: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_curve_means: Optional[torch.Tensor] = None,
#     spectral_curve_vars: Optional[torch.Tensor] = None,
#     spectral_curve_d1: Optional[torch.Tensor] = None,
#     spectral_curve_d2: Optional[torch.Tensor] = None,
#     spectral_shape_reliability: Optional[torch.Tensor] = None,
#     use_spectral_residual_energy: bool = False,
#     spectral_energy_weight: float = 0.0,
#     spectral_derivative_weight: float = 0.50,
#     spectral_second_derivative_weight: float = 0.25,
#     spectral_summary_is_physical: bool = False,
#     spectral_require_physical_summary: bool = True,
#     return_parts: bool = False,
#     **kwargs: object,
# ) -> torch.Tensor | Dict[str, torch.Tensor]:
#     """SCB-GR energy matrix: feature geometry plus optional physical spectral residual.

#     The default remains feature-only so synthetic replay and boundary anchors are
#     safe.  This mirrors the classifier's scoring contract.
#     """
#     feature_parts = _feature_geometry_energy_matrix(
#         features,
#         means,
#         bases,
#         variances,
#         active_ranks=active_ranks,
#         reliability=reliability,
#         sample_counts=sample_counts,
#         return_parts=True,
#         **kwargs,
#     )
#     feature_energy = feature_parts["energy"]
#     energy = feature_energy
#     spectral_parts = spectral_residual_energy_matrix(
#         spectral_summary=spectral_summary,
#         spectral_curve_means=spectral_curve_means,
#         spectral_curve_vars=spectral_curve_vars,
#         spectral_curve_d1=spectral_curve_d1,
#         spectral_curve_d2=spectral_curve_d2,
#         spectral_shape_reliability=spectral_shape_reliability,
#         sample_counts=sample_counts,
#         variance_floor=float(kwargs.get("variance_floor", kwargs.get("var_floor", 1e-4))),
#         derivative_weight=float(spectral_derivative_weight),
#         second_derivative_weight=float(spectral_second_derivative_weight),
#         spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         require_physical_summary=bool(spectral_require_physical_summary),
#         invalid_class_energy=float(kwargs.get("invalid_class_energy", 1e6)),
#         return_parts=True,
#     )
#     spectral_energy = spectral_parts["spectral_energy"]
#     spectral_active = bool(
#         use_spectral_residual_energy
#         and float(spectral_energy_weight) > 0.0
#         and torch.is_tensor(spectral_energy)
#         and spectral_energy.shape == feature_energy.shape
#         and bool(spectral_parts.get("spectral_active", torch.tensor(0.0)).detach().cpu().item() > 0.0)
#     )
#     if spectral_active:
#         energy = feature_energy + float(spectral_energy_weight) * spectral_energy
#     else:
#         spectral_energy = torch.zeros_like(feature_energy)

#     if return_parts:
#         out = dict(feature_parts)
#         out["energy"] = energy
#         out["feature_energy"] = feature_energy
#         out["spectral_energy"] = spectral_energy
#         out["spectral_curve_energy"] = spectral_parts.get("spectral_curve_energy", torch.zeros_like(feature_energy)) if isinstance(spectral_parts, dict) else torch.zeros_like(feature_energy)
#         out["spectral_d1_energy"] = spectral_parts.get("spectral_d1_energy", torch.zeros_like(feature_energy)) if isinstance(spectral_parts, dict) else torch.zeros_like(feature_energy)
#         out["spectral_d2_energy"] = spectral_parts.get("spectral_d2_energy", torch.zeros_like(feature_energy)) if isinstance(spectral_parts, dict) else torch.zeros_like(feature_energy)
#         out["uses_spectral_residual_energy"] = torch.tensor(bool(spectral_active), device=feature_energy.device)
#         return out
#     return energy


# def _parse_boundary_pairs(
#     risk_pairs: Optional[object],
#     old_count: int,
#     new_count: int,
#     device: torch.device,
# ) -> torch.Tensor:
#     if risk_pairs is None:
#         if old_count <= 0 or new_count <= 0:
#             return torch.empty((0, 2), device=device, dtype=torch.long)
#         return torch.tensor([(o, n) for o in range(old_count) for n in range(new_count)], device=device, dtype=torch.long)
#     if torch.is_tensor(risk_pairs):
#         pairs = risk_pairs.to(device=device).long()
#         if pairs.numel() == 0:
#             return torch.empty((0, 2), device=device, dtype=torch.long)
#         pairs = pairs.view(-1, pairs.size(-1))[:, :2]
#     else:
#         parsed = []
#         for p in risk_pairs:
#             if isinstance(p, dict):
#                 parsed.append((int(p.get("old", p.get("old_row", p.get("old_idx", 0)))), int(p.get("new", p.get("new_row", p.get("new_idx", 0))))))
#             else:
#                 parsed.append((int(p[0]), int(p[1])))
#         pairs = torch.tensor(parsed, device=device, dtype=torch.long) if parsed else torch.empty((0, 2), device=device, dtype=torch.long)
#     if pairs.numel() == 0:
#         return pairs.view(0, 2)
#     valid = (pairs[:, 0] >= 0) & (pairs[:, 0] < int(old_count)) & (pairs[:, 1] >= 0) & (pairs[:, 1] < int(new_count))
#     return pairs[valid].view(-1, 2)


# @torch.no_grad()
# def sample_boundary_geometry_features(
#     old_means: torch.Tensor,
#     old_bases: torch.Tensor,
#     old_variances: torch.Tensor,
#     *,
#     new_means: Optional[torch.Tensor] = None,
#     new_bases: Optional[torch.Tensor] = None,
#     risk_pairs: Optional[object] = None,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     old_reliability: Optional[torch.Tensor] = None,
#     old_sample_counts: Optional[torch.Tensor] = None,
#     old_class_ids: Optional[Iterable[int]] = None,
#     samples_per_pair: int = 12,
#     alphas: Iterable[float] = (1.0, 1.5, 2.0),
#     variance_floor: float = 1e-4,
#     parallel_scale: float = 0.15,
#     residual_scale: float = 0.05,
#     skip_invalid_classes: bool = True,
#     fallback_samples_per_class: int = 16,
#     return_metadata: bool = False,
#     **_: object,
# ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
#     """Deterministic old-boundary anchors toward risky new class states.

#     This is the loss-file SCB-GR fix.  It does not require new GeometryBank
#     methods.  If ``new_means`` is absent, it falls back to ordinary geometry
#     replay for compatibility.
#     """
#     if old_means is None or not torch.is_tensor(old_means) or old_means.numel() == 0:
#         device = old_means.device if torch.is_tensor(old_means) else torch.device("cpu")
#         x = torch.empty(0, 0, device=device)
#         y = torch.empty(0, dtype=torch.long, device=device)
#         meta = {"pair_count": torch.tensor(0.0, device=device), "boundary_anchor_count": torch.tensor(0.0, device=device)}
#         return (x, y, meta) if return_metadata else (x, y)
#     if new_means is None or not torch.is_tensor(new_means) or new_means.numel() == 0:
#         x, y = sample_geometry_features(
#             old_means,
#             old_bases,
#             old_variances,
#             active_ranks=old_active_ranks,
#             reliability=old_reliability,
#             sample_counts=old_sample_counts,
#             samples_per_class=int(fallback_samples_per_class),
#             variance_floor=float(variance_floor),
#             parallel_scale=1.0,
#             residual_scale=0.25,
#             reliability_gated=False,
#             skip_invalid_classes=skip_invalid_classes,
#             class_ids=old_class_ids,
#         )
#         meta = {"pair_count": torch.tensor(0.0, device=old_means.device), "boundary_anchor_count": torch.tensor(float(x.size(0)), device=old_means.device)}
#         return (x, y, meta) if return_metadata else (x, y)

#     _validate_bank_tensors(old_means, old_bases, old_variances, name="sample_boundary_geometry.old")
#     O, D = old_means.shape
#     R = old_bases.size(2)
#     device, dtype = old_means.device, old_means.dtype
#     new_means = new_means.to(device=device, dtype=dtype)
#     N = int(new_means.size(0))
#     pairs = _parse_boundary_pairs(risk_pairs, O, N, device)
#     if pairs.numel() == 0:
#         x = torch.empty(0, D, device=device, dtype=dtype)
#         y = torch.empty(0, dtype=torch.long, device=device)
#         meta = {"pair_count": torch.tensor(0.0, device=device, dtype=dtype), "boundary_anchor_count": torch.tensor(0.0, device=device, dtype=dtype)}
#         return (x, y, meta) if return_metadata else (x, y)

#     if old_active_ranks is None or not torch.is_tensor(old_active_ranks) or old_active_ranks.numel() != O:
#         active = torch.full((O,), R, device=device, dtype=torch.long)
#     else:
#         active = old_active_ranks.to(device=device).long().clamp(0, R)
#     if old_sample_counts is None or not torch.is_tensor(old_sample_counts) or old_sample_counts.numel() != O:
#         valid = torch.ones((O,), device=device, dtype=torch.bool)
#     else:
#         valid = old_sample_counts.to(device=device).flatten() > 0
#     if old_class_ids is None:
#         label_ids = list(range(O))
#     else:
#         label_ids = [int(c) for c in old_class_ids]
#         if len(label_ids) != O:
#             raise ValueError(f"old_class_ids length must match O={O}, got {len(label_ids)}")
#     alpha_list = [float(a) for a in alphas]
#     if not alpha_list:
#         alpha_list = [1.0, 1.5, 2.0]

#     feats, labels, used_pairs = [], [], []
#     n_per = max(1, int(samples_per_pair))
#     eye_dirs = torch.eye(D, device=device, dtype=dtype)
#     for pair in pairs:
#         o = int(pair[0].item())
#         n = int(pair[1].item())
#         if skip_invalid_classes and not bool(valid[o].item()):
#             continue
#         mu = old_means[o]
#         direction = new_means[n] - mu
#         norm = direction.norm().clamp_min(1e-8)
#         direction = direction / norm
#         r = int(active[o].item())
#         U = old_bases[o, :, :r] if r > 0 else old_bases.new_empty(D, 0)
#         eig = old_variances[o, :r].clamp_min(float(variance_floor)) if r > 0 else old_variances.new_empty(0)
#         res = old_variances[o, -1].clamp_min(float(variance_floor))
#         if r > 0:
#             coeff_dir = U.t() @ direction
#             low_var = (coeff_dir.pow(2) * eig).sum()
#             residual_dir = direction - U @ coeff_dir
#             residual_fraction = residual_dir.pow(2).sum().clamp(0.0, 1.0)
#             dir_var = (low_var + residual_fraction * res).clamp_min(float(variance_floor))
#             if residual_dir.norm() <= 1e-8:
#                 # Deterministic fallback residual direction that is not tied to RNG.
#                 residual_dir = eye_dirs[(o + n) % D]
#             residual_dir = F.normalize(residual_dir, dim=0, eps=1e-8)
#         else:
#             dir_var = res.clamp_min(float(variance_floor))
#             residual_dir = direction
#         dir_std = dir_var.sqrt()

#         for k in range(n_per):
#             alpha = alpha_list[k % len(alpha_list)]
#             sign = 1.0 if ((k // len(alpha_list)) % 2 == 0) else -1.0
#             z = mu + alpha * dir_std * direction
#             if r > 0 and float(parallel_scale) > 0.0:
#                 axis = k % r
#                 z = z + sign * float(parallel_scale) * eig[axis].sqrt() * U[:, axis]
#             if float(residual_scale) > 0.0:
#                 z = z + sign * float(residual_scale) * res.sqrt() * residual_dir
#             feats.append(z)
#             labels.append(torch.tensor(int(label_ids[o]), device=device, dtype=torch.long))
#             used_pairs.append((o, n))
#     if not feats:
#         x = torch.empty(0, D, device=device, dtype=dtype)
#         y = torch.empty(0, dtype=torch.long, device=device)
#     else:
#         x = torch.stack(feats, dim=0)
#         y = torch.stack(labels, dim=0)
#     meta = {
#         "pair_count": torch.tensor(float(len(set(used_pairs))), device=device, dtype=dtype),
#         "boundary_anchor_count": torch.tensor(float(x.size(0)), device=device, dtype=dtype),
#     }
#     return (x, y, meta) if return_metadata else (x, y)


# class BoundaryGeometryAnchoringLoss(nn.Module):
#     """SCB-GR old-boundary replay CE from frozen old geometry toward new states."""

#     def __init__(
#         self,
#         samples_per_pair: int = 12,
#         variance_floor: float = 1e-4,
#         logit_scale: float = 8.0,
#         reliability_energy_weight: float = 0.05,
#         residual_variance_scale: float = 0.75,
#         invalid_class_energy: float = 1e6,
#         parallel_scale: float = 0.15,
#         residual_scale: float = 0.05,
#         normalize_by_dim: bool = True,
#         use_logdet_energy: bool = True,
#         logdet_energy_weight: float = 0.05,
#         rank_margin: float = 0.25,
#     ) -> None:
#         super().__init__()
#         self.samples_per_pair = int(samples_per_pair)
#         self.variance_floor = float(variance_floor)
#         self.logit_scale = float(logit_scale)
#         self.reliability_energy_weight = float(reliability_energy_weight)
#         self.residual_variance_scale = float(residual_variance_scale)
#         self.invalid_class_energy = float(invalid_class_energy)
#         self.parallel_scale = float(parallel_scale)
#         self.residual_scale = float(residual_scale)
#         self.normalize_by_dim = bool(normalize_by_dim)
#         self.use_logdet_energy = bool(use_logdet_energy)
#         self.logdet_energy_weight = float(logdet_energy_weight)
#         self.rank_margin = float(rank_margin)

#     def forward(
#         self,
#         old_means: Optional[torch.Tensor],
#         old_bases: Optional[torch.Tensor],
#         old_variances: Optional[torch.Tensor],
#         *,
#         new_means: Optional[torch.Tensor] = None,
#         new_bases: Optional[torch.Tensor] = None,
#         risk_pairs: Optional[object] = None,
#         all_means: Optional[torch.Tensor] = None,
#         all_bases: Optional[torch.Tensor] = None,
#         all_variances: Optional[torch.Tensor] = None,
#         old_active_ranks: Optional[torch.Tensor] = None,
#         all_active_ranks: Optional[torch.Tensor] = None,
#         old_reliability: Optional[torch.Tensor] = None,
#         all_reliability: Optional[torch.Tensor] = None,
#         old_sample_counts: Optional[torch.Tensor] = None,
#         all_sample_counts: Optional[torch.Tensor] = None,
#         old_class_ids: Optional[Iterable[int]] = None,
#         return_anchors: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         ref = old_means if torch.is_tensor(old_means) else all_means
#         if old_means is None or old_bases is None or old_variances is None or not torch.is_tensor(old_means) or old_means.numel() == 0:
#             z = safe_zero_like(ref)
#             out = {"total": z, "ce": z, "num_anchors": z, "pair_count": z}
#             if return_anchors:
#                 out.update({"anchor_features": torch.empty(0, 0), "anchor_labels": torch.empty(0, dtype=torch.long)})
#             return out
#         anchor_x, anchor_y, meta = sample_boundary_geometry_features(
#             old_means,
#             old_bases,
#             old_variances,
#             new_means=new_means,
#             new_bases=new_bases,
#             risk_pairs=risk_pairs,
#             old_active_ranks=old_active_ranks,
#             old_reliability=old_reliability,
#             old_sample_counts=old_sample_counts,
#             old_class_ids=old_class_ids,
#             samples_per_pair=self.samples_per_pair,
#             variance_floor=self.variance_floor,
#             parallel_scale=self.parallel_scale,
#             residual_scale=self.residual_scale,
#             return_metadata=True,
#         )
#         if anchor_x.numel() == 0:
#             z = old_means.sum() * 0.0
#             out = {"total": z, "ce": z, "num_anchors": z, "pair_count": meta["pair_count"]}
#             if return_anchors:
#                 out.update({"anchor_features": anchor_x, "anchor_labels": anchor_y})
#             return out
#         means = all_means if torch.is_tensor(all_means) and all_means.numel() > 0 else old_means
#         bases = all_bases if torch.is_tensor(all_bases) and all_bases.numel() > 0 else old_bases
#         variances = all_variances if torch.is_tensor(all_variances) and all_variances.numel() > 0 else old_variances
#         active_ranks = all_active_ranks if all_active_ranks is not None else old_active_ranks
#         reliability = all_reliability if all_reliability is not None else old_reliability
#         sample_counts = all_sample_counts if all_sample_counts is not None else old_sample_counts
#         energy = geometry_energy_matrix(
#             anchor_x,
#             means,
#             bases,
#             variances,
#             active_ranks=active_ranks,
#             reliability=reliability,
#             sample_counts=sample_counts,
#             variance_floor=self.variance_floor,
#             reliability_energy_weight=self.reliability_energy_weight,
#             residual_variance_scale=self.residual_variance_scale,
#             normalize_by_dim=self.normalize_by_dim,
#             invalid_class_energy=self.invalid_class_energy,
#             use_logdet_energy=self.use_logdet_energy,
#             logdet_energy_weight=self.logdet_energy_weight,
#         )
#         obj = valid_energy_objective(
#             energy,
#             anchor_y,
#             sample_counts=sample_counts,
#             logit_scale=self.logit_scale,
#             min_energy_scale=1.0,
#             rank_margin=self.rank_margin,
#             label_smoothing=0.0,
#         )
#         out = {
#             "total": obj["ce"] + 0.10 * obj["rank"],
#             "ce": obj["ce"].detach(),
#             "rank": obj["rank"].detach(),
#             "num_anchors": meta["boundary_anchor_count"].to(device=anchor_x.device, dtype=anchor_x.dtype),
#             "pair_count": meta["pair_count"].to(device=anchor_x.device, dtype=anchor_x.dtype),
#             "violation_rate": obj["violation_rate"].detach(),
#         }
#         if return_anchors:
#             out.update({"anchor_features": anchor_x, "anchor_labels": anchor_y})
#         return out


# def geometry_state_admission_report_from_energy(
#     energy_new: Optional[torch.Tensor],
#     labels_new: Optional[torch.Tensor],
#     *,
#     old_class_count: int,
#     energy_old_boundary: Optional[torch.Tensor] = None,
#     labels_old_boundary: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     margin: float = 0.0,
#     max_new_violation: float = 0.25,
#     max_old_boundary_violation: float = 0.25,
# ) -> Dict[str, torch.Tensor]:
#     """Admission diagnostics from precomputed energy matrices."""
#     ref = energy_new if torch.is_tensor(energy_new) else energy_old_boundary
#     z = safe_zero_like(ref)
#     out = {
#         "safe": z,
#         "new_margin_mean": z,
#         "new_margin_min": z,
#         "new_violation_rate": z,
#         "old_boundary_margin_mean": z,
#         "old_boundary_margin_min": z,
#         "old_boundary_violation_rate": z,
#         "new_active": z,
#         "old_boundary_active": z,
#     }
#     old_class_count = int(old_class_count)
#     if energy_new is not None and torch.is_tensor(energy_new) and labels_new is not None and torch.is_tensor(labels_new) and energy_new.numel() > 0:
#         e = energy_new
#         y = labels_new.to(device=e.device).long().flatten()
#         C = int(e.size(1))
#         if y.numel() != e.size(0):
#             raise ValueError(f"labels_new/energy_new mismatch: {y.numel()} vs {e.size(0)}")
#         valid = _valid_class_mask_from_counts(sample_counts, C, e.device) if sample_counts is not None else torch.ones((C,), device=e.device, dtype=torch.bool)
#         masked = e.masked_fill(~valid.view(1, C), float("inf"))
#         new_mask = (y >= old_class_count) & (y < C)
#         if bool(new_mask.any().item()) and old_class_count > 0:
#             en = masked[new_mask]
#             yn = y[new_mask]
#             own = en.gather(1, yn.view(-1, 1)).squeeze(1)
#             nearest_old = en[:, :old_class_count].min(dim=1).values
#             margin_new = nearest_old - own
#             out["new_margin_mean"] = margin_new.mean().detach()
#             out["new_margin_min"] = margin_new.min().detach()
#             out["new_violation_rate"] = (margin_new <= float(margin)).float().mean().detach()
#             out["new_active"] = torch.tensor(float(margin_new.numel()), device=e.device, dtype=e.dtype)
#     if energy_old_boundary is not None and torch.is_tensor(energy_old_boundary) and labels_old_boundary is not None and torch.is_tensor(labels_old_boundary) and energy_old_boundary.numel() > 0:
#         e = energy_old_boundary
#         y = labels_old_boundary.to(device=e.device).long().flatten()
#         C = int(e.size(1))
#         if y.numel() != e.size(0):
#             raise ValueError(f"labels_old_boundary/energy_old_boundary mismatch: {y.numel()} vs {e.size(0)}")
#         valid = _valid_class_mask_from_counts(sample_counts, C, e.device) if sample_counts is not None else torch.ones((C,), device=e.device, dtype=torch.bool)
#         masked = e.masked_fill(~valid.view(1, C), float("inf"))
#         old_mask = (y >= 0) & (y < old_class_count)
#         if bool(old_mask.any().item()) and old_class_count > 0 and old_class_count < C:
#             eo = masked[old_mask]
#             yo = y[old_mask]
#             own = eo.gather(1, yo.view(-1, 1)).squeeze(1)
#             nearest_new = eo[:, old_class_count:].min(dim=1).values
#             margin_old = nearest_new - own
#             out["old_boundary_margin_mean"] = margin_old.mean().detach()
#             out["old_boundary_margin_min"] = margin_old.min().detach()
#             out["old_boundary_violation_rate"] = (margin_old <= float(margin)).float().mean().detach()
#             out["old_boundary_active"] = torch.tensor(float(margin_old.numel()), device=e.device, dtype=e.dtype)
#     new_ok = bool(out["new_active"].detach().cpu().item() <= 0.0) or float(out["new_violation_rate"].detach().cpu().item()) <= float(max_new_violation)
#     boundary_required = old_class_count > 0
#     old_ok = (not boundary_required) or (float(out["old_boundary_active"].detach().cpu().item()) > 0.0 and float(out["old_boundary_violation_rate"].detach().cpu().item()) <= float(max_old_boundary_violation))
#     out["safe"] = torch.tensor(float(new_ok and old_ok), device=(ref.device if torch.is_tensor(ref) else torch.device("cpu")), dtype=(ref.dtype if torch.is_tensor(ref) and ref.is_floating_point() else torch.float32))
#     return out


# def geometry_state_admission_loss(
#     *,
#     energy_new: Optional[torch.Tensor] = None,
#     labels_new: Optional[torch.Tensor] = None,
#     energy_old_boundary: Optional[torch.Tensor] = None,
#     labels_old_boundary: Optional[torch.Tensor] = None,
#     old_class_count: int = 0,
#     sample_counts: Optional[torch.Tensor] = None,
#     margin: float = 0.25,
#     new_weight: float = 1.0,
#     old_boundary_weight: float = 1.0,
#     return_parts: bool = True,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """SCB-GR admission loss: new samples stay new, old boundary anchors stay old."""
#     ref = energy_new if torch.is_tensor(energy_new) else energy_old_boundary
#     z = safe_zero_like(ref)
#     new_loss = z
#     old_loss = z
#     old_class_count = int(old_class_count)
#     if energy_new is not None and torch.is_tensor(energy_new) and labels_new is not None and torch.is_tensor(labels_new) and energy_new.numel() > 0 and old_class_count > 0:
#         e = energy_new
#         y = labels_new.to(device=e.device).long().flatten()
#         C = int(e.size(1))
#         valid = _valid_class_mask_from_counts(sample_counts, C, e.device) if sample_counts is not None else torch.ones((C,), device=e.device, dtype=torch.bool)
#         masked = e.masked_fill(~valid.view(1, C), float("inf"))
#         m = (y >= old_class_count) & (y < C)
#         if bool(m.any().item()):
#             em = masked[m]
#             ym = y[m]
#             own = em.gather(1, ym.view(-1, 1)).squeeze(1)
#             nearest_old = em[:, :old_class_count].min(dim=1).values
#             new_loss = F.relu(own + float(margin) - nearest_old).mean()
#     if energy_old_boundary is not None and torch.is_tensor(energy_old_boundary) and labels_old_boundary is not None and torch.is_tensor(labels_old_boundary) and energy_old_boundary.numel() > 0 and old_class_count > 0:
#         e = energy_old_boundary
#         y = labels_old_boundary.to(device=e.device).long().flatten()
#         C = int(e.size(1))
#         valid = _valid_class_mask_from_counts(sample_counts, C, e.device) if sample_counts is not None else torch.ones((C,), device=e.device, dtype=torch.bool)
#         masked = e.masked_fill(~valid.view(1, C), float("inf"))
#         m = (y >= 0) & (y < old_class_count)
#         if bool(m.any().item()) and old_class_count < C:
#             em = masked[m]
#             ym = y[m]
#             own = em.gather(1, ym.view(-1, 1)).squeeze(1)
#             nearest_new = em[:, old_class_count:].min(dim=1).values
#             old_loss = F.relu(own + float(margin) - nearest_new).mean()
#     total = float(new_weight) * new_loss + float(old_boundary_weight) * old_loss
#     if not return_parts:
#         return total
#     report = geometry_state_admission_report_from_energy(
#         energy_new,
#         labels_new,
#         old_class_count=old_class_count,
#         energy_old_boundary=energy_old_boundary,
#         labels_old_boundary=labels_old_boundary,
#         sample_counts=sample_counts,
#         margin=0.0,
#     )
#     report.update({"total": total, "new_loss": new_loss.detach(), "old_boundary_loss": old_loss.detach()})
#     return report


# # Clear paper-level aliases.  Keep old names alive for compatibility.
# class_geometry_state_admission_loss = geometry_state_admission_loss
# class_geometry_state_admission_report_from_energy = geometry_state_admission_report_from_energy
# scbgr_admission_loss = geometry_state_admission_loss
# scbgr_boundary_anchor_loss = BoundaryGeometryAnchoringLoss

# # Keep compatibility alias updated after overriding BoundaryGeometryAnchoringLoss.
# sample_scbgr_boundary_geometry_features = sample_boundary_geometry_features



# # =============================================================================
# # Unified one-loss entry point for both base and incremental phases
# # =============================================================================
# # This is the public loss the trainer should call.  It deliberately wraps the
# # base SRPGR objective and the incremental BAGE/SCB-GR objective behind one API.
# # Internally it still reports components because a single scalar with no
# # diagnostics is useless when old classes collapse.

# def _canonical_phase_name(phase: str) -> str:
#     p = str(phase or "base").lower().strip()
#     aliases = {
#         "0": "base",
#         "phase0": "base",
#         "phase_0": "base",
#         "pretrain": "base",
#         "base_phase": "base",
#         "inc": "incremental",
#         "increment": "incremental",
#         "incremental_phase": "incremental",
#         "scbgr": "incremental",
#         "bage": "incremental",
#         "rsgi": "incremental",
#     }
#     return aliases.get(p, p)


# def _cross_entropy_from_logits_or_energy(
#     *,
#     logits: Optional[torch.Tensor],
#     energy: Optional[torch.Tensor],
#     labels: Optional[torch.Tensor],
#     sample_counts: Optional[torch.Tensor],
#     logit_scale: float,
#     label_smoothing: float,
#     rank_margin: float,
# ) -> Dict[str, torch.Tensor]:
#     """Return CE/rank objective from logits when available, otherwise energy.

#     Base phase usually has classifier logits before the GeometryBank is fully
#     reliable.  Incremental phase should use geometry energy from the bank.
#     """
#     ref = logits if torch.is_tensor(logits) else energy
#     if labels is None or not torch.is_tensor(labels):
#         z = safe_zero_like(ref)
#         return {"ce": z, "rank": z, "compact": z, "violation_rate": z, "logits": z, "valid_mask": z}

#     y = labels.long().flatten()
#     if logits is not None and torch.is_tensor(logits) and logits.numel() > 0:
#         if logits.dim() != 2:
#             raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}")
#         if y.numel() != logits.size(0):
#             raise ValueError(f"labels/logits mismatch: {y.numel()} vs {logits.size(0)}")
#         if int(y.min().item()) < 0 or int(y.max().item()) >= logits.size(1):
#             raise ValueError(f"labels outside logits class range C={logits.size(1)}")
#         ce = F.cross_entropy(logits, y.to(device=logits.device), label_smoothing=float(label_smoothing))
#         z = logits.sum() * 0.0
#         pred = logits.argmax(dim=1)
#         return {
#             "ce": ce,
#             "rank": z,
#             "compact": z,
#             "violation_rate": (pred != y.to(device=logits.device)).float().mean().detach(),
#             "logits": logits,
#             "valid_mask": torch.ones((logits.size(1),), device=logits.device, dtype=torch.bool),
#         }

#     if energy is not None and torch.is_tensor(energy) and energy.numel() > 0:
#         obj = valid_energy_objective(
#             energy,
#             y.to(device=energy.device),
#             sample_counts=sample_counts,
#             logit_scale=float(logit_scale),
#             min_energy_scale=1.0,
#             rank_margin=float(rank_margin),
#             label_smoothing=float(label_smoothing),
#         )
#         return obj

#     z = safe_zero_like(ref)
#     return {"ce": z, "rank": z, "compact": z, "violation_rate": z, "logits": z, "valid_mask": z}


# def _split_incremental_energy_by_role(
#     energy: Optional[torch.Tensor],
#     labels: Optional[torch.Tensor],
#     *,
#     old_class_count: int,
#     batch_role: Optional[torch.Tensor] = None,
# ) -> Dict[str, Optional[torch.Tensor]]:
#     """Split joint incremental batch into new-real and old-boundary groups.

#     Role convention, when supplied:
#         0 = new real/current sample
#         1 = old boundary/replay anchor

#     If role is absent, labels >= old_class_count are treated as new samples and
#     labels < old_class_count as old boundary/replay anchors.
#     """
#     out: Dict[str, Optional[torch.Tensor]] = {
#         "energy_new": None,
#         "labels_new": None,
#         "energy_old_boundary": None,
#         "labels_old_boundary": None,
#         "new_mask": None,
#         "old_boundary_mask": None,
#     }
#     if energy is None or labels is None or not torch.is_tensor(energy) or not torch.is_tensor(labels) or energy.numel() == 0:
#         return out
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise ValueError(f"labels/energy mismatch: {y.numel()} vs {energy.size(0)}")
#     old_class_count = int(old_class_count)
#     if batch_role is not None and torch.is_tensor(batch_role) and batch_role.numel() == y.numel():
#         r = batch_role.to(device=energy.device).long().flatten()
#         new_mask = r == 0
#         old_boundary_mask = r == 1
#     else:
#         new_mask = y >= old_class_count
#         old_boundary_mask = y < old_class_count
#     if bool(new_mask.any().item()):
#         out["energy_new"] = energy[new_mask]
#         out["labels_new"] = y[new_mask]
#     if bool(old_boundary_mask.any().item()):
#         out["energy_old_boundary"] = energy[old_boundary_mask]
#         out["labels_old_boundary"] = y[old_boundary_mask]
#     out["new_mask"] = new_mask
#     out["old_boundary_mask"] = old_boundary_mask
#     return out


# def unified_spectral_geometry_loss(
#     *,
#     phase: str,
#     labels: torch.Tensor,
#     # Common supervised inputs.
#     logits: Optional[torch.Tensor] = None,
#     energy: Optional[torch.Tensor] = None,
#     features: Optional[torch.Tensor] = None,
#     sample_counts: Optional[torch.Tensor] = None,
#     # Base-phase metadata.
#     key_features: Optional[torch.Tensor] = None,
#     band_summary: Optional[torch.Tensor] = None,
#     spectral_summary: Optional[torch.Tensor] = None,
#     spectral_summary_is_physical: bool = False,
#     # Incremental phase role/memory.
#     old_class_count: int = 0,
#     batch_role: Optional[torch.Tensor] = None,
#     old_bases: Optional[torch.Tensor] = None,
#     new_bases: Optional[torch.Tensor] = None,
#     old_active_ranks: Optional[torch.Tensor] = None,
#     new_active_ranks: Optional[torch.Tensor] = None,
#     old_spectral_curve_means: Optional[torch.Tensor] = None,
#     new_spectral_curve_means: Optional[torch.Tensor] = None,
#     old_spectral_curve_d1: Optional[torch.Tensor] = None,
#     new_spectral_curve_d1: Optional[torch.Tensor] = None,
#     old_spectral_curve_d2: Optional[torch.Tensor] = None,
#     new_spectral_curve_d2: Optional[torch.Tensor] = None,
#     old_spectral_reliability: Optional[torch.Tensor] = None,
#     new_spectral_reliability: Optional[torch.Tensor] = None,
#     reliability: Optional[torch.Tensor] = None,
#     # New geometry-state trust/volume controls.
#     new_means: Optional[torch.Tensor] = None,
#     new_variances: Optional[torch.Tensor] = None,
#     init_new_means: Optional[torch.Tensor] = None,
#     init_new_bases: Optional[torch.Tensor] = None,
#     init_new_variances: Optional[torch.Tensor] = None,
#     reference_old_variances: Optional[torch.Tensor] = None,
#     reference_old_active_ranks: Optional[torch.Tensor] = None,
#     # Global weights.
#     ce_weight: float = 1.0,
#     rank_weight: float = 0.10,
#     base_geometry_weight: float = 1.0,
#     admission_weight: float = 1.0,
#     subspace_weight: float = 0.10,
#     volume_weight: float = 0.05,
#     trust_weight: float = 0.05,
#     # Shared knobs.
#     logit_scale: float = 8.0,
#     label_smoothing: float = 0.0,
#     rank_margin: float = 0.25,
#     admission_margin: float = 0.25,
#     target_overlap: float = 0.35,
#     spectral_margin_strength: float = 0.20,
#     # Base SRPGR knobs.
#     gics_weight: float = 0.20,
#     pgr_weight: float = 0.10,
#     spectral_shape_weight: float = 0.05,
#     return_parts: bool = True,
#     **kwargs: object,
# ) -> Dict[str, torch.Tensor] | torch.Tensor:
#     """One public loss for both phases: USG / BAGE-style loss.

#     Base phase:
#         L = CE + SRPGR
#         where SRPGR builds a compact, separated, low-rank-ready HSI feature space.

#     Incremental phase:
#         L = Energy CE + global energy margin + geometry-state admission
#             + spectral-aware old/new subspace separation
#             + new-state volume control + new-state trust region.

#     This function is intentionally the only loss entry point the trainer needs.
#     Component values are returned only for diagnostics.
#     """
#     mode = _canonical_phase_name(phase)
#     ref = features if torch.is_tensor(features) else (energy if torch.is_tensor(energy) else logits)
#     z = safe_zero_like(ref)

#     ce_obj = _cross_entropy_from_logits_or_energy(
#         logits=logits,
#         energy=energy,
#         labels=labels,
#         sample_counts=sample_counts,
#         logit_scale=float(logit_scale),
#         label_smoothing=float(label_smoothing),
#         rank_margin=float(rank_margin),
#     )
#     ce = ce_obj["ce"]
#     rank = ce_obj["rank"] if torch.is_tensor(ce_obj.get("rank", None)) else z

#     base_geom_total = z
#     admission_total = z
#     subspace_total = z
#     volume_total = z
#     trust_total = z

#     base_geom_parts: Dict[str, torch.Tensor] = {}
#     admission_parts: Dict[str, torch.Tensor] = {}
#     subspace_parts: Dict[str, torch.Tensor] = {}
#     volume_parts: Dict[str, torch.Tensor] = {}
#     trust_parts: Dict[str, torch.Tensor] = {}

#     if mode == "base":
#         if features is not None and torch.is_tensor(features) and features.numel() > 0:
#             base_geom = spectral_residual_prospective_geometry_loss(
#                 features,
#                 labels,
#                 key_features=key_features,
#                 band_summary=band_summary,
#                 spectral_summary=spectral_summary,
#                 spectral_summary_is_physical=bool(spectral_summary_is_physical),
#                 weight=1.0,
#                 gics_weight=float(gics_weight),
#                 pgr_weight=float(pgr_weight),
#                 spectral_shape_weight=float(spectral_shape_weight),
#                 return_parts=True,
#                 **kwargs,
#             )
#             base_geom_total = _scalar_from_parts(base_geom, "total", features)
#             base_geom_parts = base_geom if isinstance(base_geom, dict) else {}
#         total = float(ce_weight) * ce + float(base_geometry_weight) * base_geom_total

#     elif mode == "incremental":
#         split = _split_incremental_energy_by_role(
#             energy,
#             labels,
#             old_class_count=int(old_class_count),
#             batch_role=batch_role,
#         )
#         if energy is not None and torch.is_tensor(energy) and energy.numel() > 0:
#             admission = geometry_state_admission_loss(
#                 energy_new=split["energy_new"],
#                 labels_new=split["labels_new"],
#                 energy_old_boundary=split["energy_old_boundary"],
#                 labels_old_boundary=split["labels_old_boundary"],
#                 old_class_count=int(old_class_count),
#                 sample_counts=sample_counts,
#                 margin=float(admission_margin),
#                 new_weight=1.0,
#                 old_boundary_weight=1.0,
#                 return_parts=True,
#             )
#             admission_total = _scalar_from_parts(admission, "total", energy)
#             admission_parts = admission if isinstance(admission, dict) else {}

#         if old_bases is not None and new_bases is not None and torch.is_tensor(old_bases) and torch.is_tensor(new_bases):
#             sep = risk_aware_old_new_subspace_separation_loss(
#                 old_bases,
#                 new_bases,
#                 old_active_ranks=old_active_ranks,
#                 new_active_ranks=new_active_ranks,
#                 old_spectral_curve_means=old_spectral_curve_means,
#                 new_spectral_curve_means=new_spectral_curve_means,
#                 old_spectral_curve_d1=old_spectral_curve_d1,
#                 new_spectral_curve_d1=new_spectral_curve_d1,
#                 old_spectral_curve_d2=old_spectral_curve_d2,
#                 new_spectral_curve_d2=new_spectral_curve_d2,
#                 old_spectral_reliability=old_spectral_reliability,
#                 new_spectral_reliability=new_spectral_reliability,
#                 target_overlap=float(target_overlap),
#                 spectral_margin_strength=float(spectral_margin_strength),
#                 reliability=reliability,
#                 return_parts=True,
#             )
#             subspace_total = _scalar_from_parts(sep, "total", new_bases)
#             subspace_parts = sep if isinstance(sep, dict) else {}

#         if new_variances is not None and torch.is_tensor(new_variances) and new_variances.numel() > 0:
#             vol = descriptor_volume_control_loss(
#                 new_variances,
#                 active_ranks=new_active_ranks,
#                 feature_dim=(features.size(1) if torch.is_tensor(features) and features.dim() == 2 else None),
#                 reference_variances=reference_old_variances,
#                 reference_active_ranks=reference_old_active_ranks,
#                 return_parts=True,
#             )
#             volume_total = _scalar_from_parts(vol, "total", new_variances)
#             volume_parts = vol if isinstance(vol, dict) else {}

#         if (
#             new_means is not None and new_bases is not None and new_variances is not None
#             and init_new_means is not None and init_new_bases is not None and init_new_variances is not None
#             and torch.is_tensor(new_means) and torch.is_tensor(new_bases) and torch.is_tensor(new_variances)
#         ):
#             trust = descriptor_trust_region_loss(
#                 new_means,
#                 new_bases,
#                 new_variances,
#                 init_new_means,
#                 init_new_bases,
#                 init_new_variances,
#                 active_ranks=new_active_ranks,
#                 return_parts=True,
#             )
#             trust_total = _scalar_from_parts(trust, "total", new_means)
#             trust_parts = trust if isinstance(trust, dict) else {}

#         total = (
#             float(ce_weight) * ce
#             + float(rank_weight) * rank
#             + float(admission_weight) * admission_total
#             + float(subspace_weight) * subspace_total
#             + float(volume_weight) * volume_total
#             + float(trust_weight) * trust_total
#         )
#     else:
#         raise ValueError("phase must be 'base' or 'incremental'.")

#     if not return_parts:
#         return total

#     def _detach(v: torch.Tensor) -> torch.Tensor:
#         return v.detach() if torch.is_tensor(v) else z.detach()

#     out = {
#         "total": total,
#         "phase_is_base": torch.tensor(float(mode == "base"), device=(total.device if torch.is_tensor(total) else torch.device("cpu"))),
#         "phase_is_incremental": torch.tensor(float(mode == "incremental"), device=(total.device if torch.is_tensor(total) else torch.device("cpu"))),
#         "ce": _detach(ce),
#         "rank": _detach(rank),
#         "base_geometry": _detach(base_geom_total),
#         "admission": _detach(admission_total),
#         "subspace": _detach(subspace_total),
#         "volume": _detach(volume_total),
#         "trust": _detach(trust_total),
#         "violation_rate": _detach(ce_obj.get("violation_rate", z)),
#     }

#     # Base diagnostics.
#     for key in ["gics", "pgr", "compact", "center", "band", "spectral_shape", "spectral_shape_active"]:
#         if key in base_geom_parts and torch.is_tensor(base_geom_parts[key]):
#             out[f"base_{key}"] = base_geom_parts[key].detach()

#     # Incremental diagnostics.
#     for key in [
#         "safe",
#         "new_margin_mean",
#         "new_margin_min",
#         "new_violation_rate",
#         "old_boundary_margin_mean",
#         "old_boundary_margin_min",
#         "old_boundary_violation_rate",
#         "new_loss",
#         "old_boundary_loss",
#     ]:
#         if key in admission_parts and torch.is_tensor(admission_parts[key]):
#             out[f"admission_{key}"] = admission_parts[key].detach()

#     for key in ["mean_overlap", "max_overlap", "mean_target", "pair_count"]:
#         if key in subspace_parts and torch.is_tensor(subspace_parts[key]):
#             out[f"subspace_{key}"] = subspace_parts[key].detach()

#     for key in ["mean_logdet", "max_logdet", "cap"]:
#         if key in volume_parts and torch.is_tensor(volume_parts[key]):
#             out[f"volume_{key}"] = volume_parts[key].detach()

#     for key in ["mean", "basis", "variance"]:
#         if key in trust_parts and torch.is_tensor(trust_parts[key]):
#             out[f"trust_{key}"] = trust_parts[key].detach()

#     return out


# class UnifiedSpectralGeometryLoss(nn.Module):
#     """nn.Module wrapper for the single base+incremental NECIL-HSI loss."""

#     def __init__(
#         self,
#         *,
#         ce_weight: float = 1.0,
#         rank_weight: float = 0.10,
#         base_geometry_weight: float = 1.0,
#         admission_weight: float = 1.0,
#         subspace_weight: float = 0.10,
#         volume_weight: float = 0.05,
#         trust_weight: float = 0.05,
#         logit_scale: float = 8.0,
#         label_smoothing: float = 0.0,
#         rank_margin: float = 0.25,
#         admission_margin: float = 0.25,
#         target_overlap: float = 0.35,
#         spectral_margin_strength: float = 0.20,
#         gics_weight: float = 0.20,
#         pgr_weight: float = 0.10,
#         spectral_shape_weight: float = 0.05,
#         **kwargs: object,
#     ) -> None:
#         super().__init__()
#         self.defaults = dict(
#             ce_weight=float(ce_weight),
#             rank_weight=float(rank_weight),
#             base_geometry_weight=float(base_geometry_weight),
#             admission_weight=float(admission_weight),
#             subspace_weight=float(subspace_weight),
#             volume_weight=float(volume_weight),
#             trust_weight=float(trust_weight),
#             logit_scale=float(logit_scale),
#             label_smoothing=float(label_smoothing),
#             rank_margin=float(rank_margin),
#             admission_margin=float(admission_margin),
#             target_overlap=float(target_overlap),
#             spectral_margin_strength=float(spectral_margin_strength),
#             gics_weight=float(gics_weight),
#             pgr_weight=float(pgr_weight),
#             spectral_shape_weight=float(spectral_shape_weight),
#         )
#         self.extra_defaults = dict(kwargs)

#     def forward(self, **kwargs: object) -> Dict[str, torch.Tensor] | torch.Tensor:
#         merged = dict(self.extra_defaults)
#         merged.update(self.defaults)
#         merged.update(kwargs)
#         return unified_spectral_geometry_loss(**merged)


# # Paper/code aliases.
# unified_necil_hsi_loss = unified_spectral_geometry_loss
# unified_geometry_state_loss = unified_spectral_geometry_loss
# base_incremental_geometry_loss = unified_spectral_geometry_loss
# BAGELoss = UnifiedSpectralGeometryLoss
# UnifiedNECILHSILoss = UnifiedSpectralGeometryLoss

# # Refresh __all__ after appending unified-loss names.
# __all__ = [name for name in globals().keys() if not name.startswith("_")]

