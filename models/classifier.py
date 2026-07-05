
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-12
_INVALID_LOGIT = -1e9


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", "none", "null", ""}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")
    return bool(value)


def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        c = int(value)
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _as_seen_list(
    seen_classes: Optional[Iterable[int]],
    *,
    fallback_count: Optional[int] = None,
) -> List[int]:
    if seen_classes is None:
        if fallback_count is None:
            raise ValueError(
                "seen_classes is required. The classifier must know the current "
                "seen-class order so logits are [B, len(seen_classes)]."
            )
        seen = list(range(int(fallback_count)))
    else:
        seen = [int(c) for c in seen_classes]

    if not seen:
        raise ValueError("seen_classes is empty.")
    if len(set(seen)) != len(seen):
        raise ValueError(f"seen_classes contains duplicates: {seen}")
    if min(seen) < 0:
        raise ValueError(f"seen_classes contains negative ids: {seen}")
    return seen


def _as_long_1d(x: torch.Tensor, *, device: torch.device, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a tensor.")
    return x.to(device=device).long().flatten()


def _finite_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a tensor.")
    if x.numel() == 0:
        raise ValueError(f"{name} is empty.")
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
        raise RuntimeError(f"{name} contains {bad} NaN/Inf values.")
    return x


def _zero_like(ref: Optional[torch.Tensor] = None, *, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch.is_tensor(ref):
        return ref.sum() * 0.0
    return torch.tensor(0.0, device=device if device is not None else torch.device("cpu"))


def _tensor_from_bank(bank: Mapping[str, Any], *names: str) -> torch.Tensor:
    for name in names:
        value = bank.get(name, None)
        if torch.is_tensor(value):
            return value
    raise KeyError(f"GeometryBank is missing required tensor. Tried keys={names}")


# -----------------------------------------------------------------------------
# Optional ablation: bounded old/new logit bias
# -----------------------------------------------------------------------------


class OldNewLogitCalibrator(nn.Module):
    """Small bounded old/new logit bias.

    This module is not part of the main PG-RGA path. It is kept as an explicit
    ablation only. The main method should use geometry replay + energy margins,
    not post-hoc score shifting.
    """

    def __init__(self, max_abs_bias: float = 1.0) -> None:
        super().__init__()
        self.max_abs_bias = float(max(0.0, max_abs_bias))
        self.old_bias_raw = nn.Parameter(torch.tensor(0.0))

    def bias_value(self) -> torch.Tensor:
        return self.max_abs_bias * torch.tanh(self.old_bias_raw)

    def forward(self, logits: torch.Tensor, old_mask: torch.Tensor, new_mask: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be [B,S], got {tuple(logits.shape)}")
        if old_mask.numel() != logits.size(1) or new_mask.numel() != logits.size(1):
            raise ValueError("old/new masks must match logit width.")
        if not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
            return logits
        bias = self.bias_value().to(device=logits.device, dtype=logits.dtype)
        out = logits.clone()
        out[:, old_mask] = out[:, old_mask] + bias
        out[:, new_mask] = out[:, new_mask] - bias
        return out

    @torch.no_grad()
    def summary(self) -> Dict[str, float]:
        return {"old_new_logit_bias": float(self.bias_value().detach().cpu().item())}


# -----------------------------------------------------------------------------
# Main classifier
# -----------------------------------------------------------------------------


class GeometryEnergyClassifier(nn.Module):
    """Strict seen-class low-rank GeometryBank classifier for PG-RGA / NECIL-HSI.

    Contract:
        * Input features are canonical projected z-space features [B, D].
        * GeometryBank stores compact descriptors only.
        * Output logits are [B, len(seen_classes)] in exactly seen_classes order.
        * CE targets for returned logits must be seen-local labels.
        * No hidden future class columns.
        * No prototype/concept branch.
        * No adaptive boundary as the main classifier path.
        * No measured energy normalization.

    The classifier implements the same low-rank Gaussian energy used by replay,
    diagnostics, and old/new margin losses:
        E = low-rank Mahalanobis + residual Mahalanobis
            + optional centered logdet penalty
            + optional centered reliability penalty.
    """

    def __init__(
        self,
        initial_classes: int = 0,
        d_model: int = 128,
        logit_scale: float = 8.0,
        variance_floor: float = 1e-4,
        residual_variance_scale: float = 0.75,
        normalize_energy_by_dim: bool = True,
        energy_normalize_by_dim: Optional[bool] = None,
        use_logdet_energy: bool = True,
        logdet_energy_weight: float = 0.05,
        logdet_normalize_by_dim: bool = True,
        center_logdet_energy: bool = True,
        use_reliability_penalty: bool = True,
        reliability_energy_weight: float = 0.03,
        reliability_min_clamp: float = 0.05,
        center_reliability_energy: bool = True,
        use_old_new_calibration: bool = False,
        use_energy_calibrator: Optional[bool] = None,
        energy_calibrator_type: str = "none",
        calibration_max_abs_bias: float = 1.0,
        energy_calibrator_max_bias: Optional[float] = None,
        logit_clip: float = 0.0,
        invalid_logit: float = _INVALID_LOGIT,
        invalid_class_energy: float = 1e6,
        # accepted legacy args; explicitly disabled in the main method
        use_measured_energy_calibration: bool = False,
        use_adaptive_boundary: bool = False,
        use_spectral_geometry: bool = False,
        spectral_energy_weight: float = 0.0,
        band_energy_weight: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()

        if _to_bool(use_measured_energy_calibration, False):
            raise RuntimeError(
                "use_measured_energy_calibration is disabled. It hides old/new "
                "geometry-scale errors and label bugs."
            )
        if _to_bool(use_adaptive_boundary, False):
            raise RuntimeError(
                "use_adaptive_boundary is not part of the PG-RGA main classifier. "
                "Use GeometryBank replay + old/new energy margin instead."
            )
        if float(band_energy_weight) != 0.0:
            raise RuntimeError(
                "Direct band-energy classifier branch is disabled. Band/spectral information "
                "belongs in base geometry preparation and GeometryBank diagnostics, not as a "
                "hidden inference branch."
            )
        if _to_bool(use_spectral_geometry, False) or float(spectral_energy_weight) > 0.0:
            raise RuntimeError(
                "Spectral residual classifier energy is disabled in the PG-RGA main path. "
                "Use geometry_only scoring; physical spectral shape is handled during base "
                "geometry preparation, not old replay scoring."
            )

        if energy_normalize_by_dim is not None:
            normalize_energy_by_dim = _to_bool(energy_normalize_by_dim, True)
        if use_energy_calibrator is not None:
            use_old_new_calibration = _to_bool(use_energy_calibrator, False)
        if energy_calibrator_max_bias is not None:
            calibration_max_abs_bias = float(energy_calibrator_max_bias)

        ct = str(energy_calibrator_type or "none").strip().lower()
        if ct in {"none", "off", "false", ""}:
            ct = "none"
        if ct not in {"none", "old_new"}:
            raise ValueError("energy_calibrator_type must be 'none' or 'old_new'.")

        self.num_classes = int(max(0, initial_classes))
        self.d_model = int(d_model)
        self.logit_scale = float(logit_scale)
        self.variance_floor = float(max(variance_floor, 1e-12))
        self.residual_variance_scale = float(max(residual_variance_scale, 1e-8))
        self.normalize_energy_by_dim = _to_bool(normalize_energy_by_dim, True)
        self.use_logdet_energy = _to_bool(use_logdet_energy, True)
        self.logdet_energy_weight = float(max(logdet_energy_weight, 0.0))
        self.logdet_normalize_by_dim = _to_bool(logdet_normalize_by_dim, True)
        self.center_logdet_energy = _to_bool(center_logdet_energy, True)
        self.use_reliability_penalty = _to_bool(use_reliability_penalty, True)
        self.reliability_energy_weight = float(max(reliability_energy_weight, 0.0))
        self.reliability_min_clamp = float(max(min(reliability_min_clamp, 1.0), 1e-8))
        self.center_reliability_energy = _to_bool(center_reliability_energy, True)
        self.logit_clip = float(max(logit_clip, 0.0))
        self.invalid_logit = float(invalid_logit)
        self.invalid_class_energy = float(max(invalid_class_energy, 1.0))

        self.energy_calibrator_type = ct
        self.use_old_new_calibration = _to_bool(use_old_new_calibration, False) and ct == "old_new"
        self.calibrator = OldNewLogitCalibrator(max_abs_bias=float(calibration_max_abs_bias))
        if not self.use_old_new_calibration:
            self.freeze_all_adaptation()

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self._last_seen_classes: List[int] = list(range(self.num_classes))

    # ------------------------------------------------------------------
    # Compatibility / adaptation controls
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_mode(mode: str) -> str:
        m = str(mode or "geometry_only").lower().strip()
        aliases = {
            "geo": "geometry_only",
            "geometry": "geometry_only",
            "geometry-only": "geometry_only",
            "feature_geometry": "geometry_only",
            "feature-only": "geometry_only",
            "feature_only": "geometry_only",
            "low_rank_geometry": "geometry_only",
            "replay": "geometry_only",
            "synthetic_replay": "geometry_only",
            "anchor": "geometry_only",
            "anchor_concept": "geometry_only",
            "anchor_concept_geometry": "geometry_only",
            "srgp": "geometry_only",
            "srgp_geometry": "geometry_only",
            "spectral_geometry": "geometry_only",
            "spectral_residual": "geometry_only",
            "calibrated_geometry": "calibrated_geometry",
            "topology_calibrated_geometry": "calibrated_geometry",
        }
        out = aliases.get(m, m)
        if out not in {"geometry_only", "calibrated_geometry"}:
            raise ValueError(f"Unsupported classifier mode {mode!r}. Use 'geometry_only' or 'calibrated_geometry'.")
        return out

    def expand(self, num_new_classes: int, phase: int = 0) -> None:
        del phase
        self.num_classes += int(max(0, num_new_classes))

    def expand_to_seen_classes(self, seen_classes: Iterable[int]) -> None:
        seen = _as_seen_list(seen_classes)
        self._last_seen_classes = seen
        self.num_classes = len(seen)

    def freeze_all_adaptation(self) -> None:
        for p in self.calibrator.parameters():
            p.requires_grad = False

    def unfreeze_all_adaptation(self) -> None:
        if self.use_old_new_calibration:
            for p in self.calibrator.parameters():
                p.requires_grad = True
        else:
            self.freeze_all_adaptation()

    def freeze_old_adaptation(self, old_class_count: int) -> None:
        del old_class_count
        self.freeze_all_adaptation()

    def freeze_fusion_module(self) -> None:
        return

    def unfreeze_fusion_module(self) -> None:
        return

    def adaptation_regularization_loss(self, num_classes: Optional[int] = None) -> Dict[str, torch.Tensor]:
        del num_classes
        z = self._zero * 0.0
        if not self.use_old_new_calibration:
            return {"total": z, "bias": z, "temp": z, "alpha": z, "energy_cal": z, "adaptive_boundary": z}
        loss = self.calibrator.old_bias_raw.pow(2)
        return {"total": loss, "bias": loss, "temp": z, "alpha": z, "energy_cal": loss, "adaptive_boundary": z}

    def energy_calibration_regularization_loss(self, num_classes: Optional[int] = None) -> torch.Tensor:
        return self.adaptation_regularization_loss(num_classes=num_classes)["energy_cal"]

    def enable_energy_calibration(self, enabled: bool = True, calibrator_type: Optional[str] = None) -> None:
        enabled_b = _to_bool(enabled, False)
        if calibrator_type is not None:
            self.energy_calibrator_type = "old_new" if str(calibrator_type).lower().strip() == "old_new" else "none"
        self.use_old_new_calibration = enabled_b and self.energy_calibrator_type == "old_new"
        self.unfreeze_all_adaptation()

    # Adaptive-boundary compatibility no-ops. The main architecture does not use them.
    def boundary_parameters(self) -> Iterable[nn.Parameter]:
        return []

    def freeze_all_boundary_radii(self) -> None:
        return

    def unfreeze_all_boundary_radii(self) -> None:
        raise RuntimeError("Adaptive boundary radii are disabled in the PG-RGA main classifier.")

    def freeze_old_boundary_radii(self, old_class_count: int) -> None:
        del old_class_count
        return

    def adaptive_boundary_state(self, num_classes: Optional[int] = None, old_class_count: int = 0) -> Dict[str, float]:
        del num_classes, old_class_count
        return {
            "adaptive_boundary_enabled": 0.0,
            "boundary_radius_mean": 0.0,
            "boundary_radius_min": 0.0,
            "boundary_radius_max": 0.0,
            "old_boundary_radius_mean": 0.0,
            "new_boundary_radius_mean": 0.0,
        }

    def adaptive_boundary_loss(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        del args, kwargs
        z = self._zero * 0.0
        return {"total": z, "boundary": z.detach(), "old_new": z.detach(), "radius_reg": z.detach()}

    # ------------------------------------------------------------------
    # Bank handling
    # ------------------------------------------------------------------
    def _bank_dict(self, geometry_bank: Any) -> Dict[str, torch.Tensor]:
        if geometry_bank is None:
            raise ValueError("geometry_bank is required for geometry scoring.")
        if isinstance(geometry_bank, dict):
            return dict(geometry_bank)
        if hasattr(geometry_bank, "get_bank") and callable(geometry_bank.get_bank):
            return geometry_bank.get_bank()
        if hasattr(geometry_bank, "get_subspace_bank") and callable(geometry_bank.get_subspace_bank):
            return geometry_bank.get_subspace_bank()
        raise TypeError("geometry_bank must be a dict or an object exposing get_bank()/get_subspace_bank().")

    @staticmethod
    def _bank_variances(bank: Mapping[str, Any]) -> torch.Tensor:
        if "variances" in bank and torch.is_tensor(bank["variances"]):
            return bank["variances"]
        eig = _tensor_from_bank(bank, "eigvals")
        if "res_vars" in bank and torch.is_tensor(bank["res_vars"]):
            res = bank["res_vars"]
        else:
            res = _tensor_from_bank(bank, "resvars")
        return torch.cat([eig, res.unsqueeze(-1)], dim=-1)

    def _infer_seen_from_bank(self, bank: Mapping[str, Any]) -> List[int]:
        counts = _tensor_from_bank(bank, "sample_counts").detach().cpu().flatten()
        valid = torch.isfinite(counts) & (counts > 0)
        if "class_ids" in bank and torch.is_tensor(bank["class_ids"]):
            ids = bank["class_ids"].detach().cpu().long().flatten()
            if ids.numel() == counts.numel():
                return [int(ids[i].item()) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
        return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]

    def _resolve_row_indices(
        self,
        bank: Mapping[str, Any],
        seen_classes: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        means = _tensor_from_bank(bank, "means")
        C = int(means.size(0))
        if "class_ids" in bank and torch.is_tensor(bank["class_ids"]):
            bank_class_ids = bank["class_ids"].detach().cpu().long().flatten().tolist()
            if len(bank_class_ids) != C:
                raise RuntimeError(f"bank['class_ids'] length {len(bank_class_ids)} does not match rows {C}.")
            mapping = {int(c): i for i, c in enumerate(bank_class_ids)}
            missing = [int(c) for c in seen_classes if int(c) not in mapping]
            if missing:
                raise IndexError(f"seen_classes absent from sliced GeometryBank class_ids: {missing}")
            return torch.as_tensor([mapping[int(c)] for c in seen_classes], device=device, dtype=torch.long)

        # Full bank: row index is global class id.
        missing = [int(c) for c in seen_classes if int(c) < 0 or int(c) >= C]
        if missing:
            raise IndexError(f"seen_classes contain ids absent from full GeometryBank: {missing}; bank_rows={C}")
        return torch.as_tensor([int(c) for c in seen_classes], device=device, dtype=torch.long)

    def _select_bank_rows(
        self,
        bank: Mapping[str, Any],
        seen_classes: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        means = _tensor_from_bank(bank, "means").to(device=device, dtype=dtype)
        bases = _tensor_from_bank(bank, "bases", "raw_bases").to(device=device, dtype=dtype)
        variances = self._bank_variances(bank).to(device=device, dtype=dtype)
        sample_counts = _tensor_from_bank(bank, "sample_counts").to(device=device, dtype=dtype).flatten()

        if means.dim() != 2 or means.size(1) != self.d_model:
            raise ValueError(f"bank means must be [C,{self.d_model}], got {tuple(means.shape)}")
        if bases.dim() != 3 or bases.size(1) != self.d_model:
            raise ValueError(f"bank bases must be [C,{self.d_model},R], got {tuple(bases.shape)}")
        if variances.dim() != 2 or variances.size(0) != means.size(0) or variances.size(1) != bases.size(2) + 1:
            raise ValueError(
                f"bank variances must be [C,R+1], got {tuple(variances.shape)} for bases {tuple(bases.shape)}"
            )
        if sample_counts.numel() != means.size(0):
            raise ValueError(f"sample_counts width mismatch: {sample_counts.numel()} vs rows {means.size(0)}")

        row_idx = self._resolve_row_indices(bank, seen_classes, device=device)
        counts_seen = sample_counts.index_select(0, row_idx)
        missing = [int(seen_classes[i]) for i in range(len(seen_classes)) if float(counts_seen[i].detach().cpu().item()) <= 0.0]
        if missing:
            raise RuntimeError(f"Geometry scoring requested classes with no valid GeometryBank row: {missing}")

        reliability = None
        if "reliability" in bank and torch.is_tensor(bank["reliability"]):
            reliability = bank["reliability"].to(device=device, dtype=dtype).index_select(0, row_idx)
        active_ranks = None
        if "active_ranks" in bank and torch.is_tensor(bank["active_ranks"]):
            active_ranks = bank["active_ranks"].to(device=device).long().index_select(0, row_idx)

        return {
            "means": means.index_select(0, row_idx),
            "bases": bases.index_select(0, row_idx),
            "eigvals": variances.index_select(0, row_idx)[:, :-1].clamp_min(self.variance_floor),
            "res_vars": variances.index_select(0, row_idx)[:, -1].flatten().clamp_min(self.variance_floor),
            "sample_counts": counts_seen,
            "reliability": reliability,
            "active_ranks": active_ranks,
            "global_class_ids": torch.as_tensor([int(c) for c in seen_classes], device=device, dtype=torch.long),
            "row_indices": row_idx,
        }

    # ------------------------------------------------------------------
    # Labels and old/new masks
    # ------------------------------------------------------------------
    @staticmethod
    def global_to_local_labels(labels: torch.Tensor, seen_classes: Sequence[int]) -> torch.Tensor:
        if not torch.is_tensor(labels):
            raise TypeError("labels must be a tensor.")
        seen = [int(c) for c in seen_classes]
        mapping = {c: i for i, c in enumerate(seen)}
        y = labels.long().flatten()
        out = torch.empty_like(y)
        bad: List[int] = []
        for i, value in enumerate(y.detach().cpu().tolist()):
            v = int(value)
            if v not in mapping:
                bad.append(v)
                out[i] = -1
            else:
                out[i] = mapping[v]
        if bad:
            raise RuntimeError(f"labels contain classes not in seen_classes. bad={sorted(set(bad))}, seen={seen}")
        return out.to(device=labels.device)

    def _old_new_masks(
        self,
        seen_classes: Sequence[int],
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
        *,
        require_prefix: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seen = [int(c) for c in seen_classes]
        seen_set = set(seen)

        if old_classes is None and old_class_count is not None:
            k = int(max(0, min(int(old_class_count), len(seen))))
            old_list = seen[:k]
        else:
            old_list = _ordered_unique_ints(old_classes or [])

        if new_classes is None:
            new_list = [c for c in seen if c not in set(old_list)]
        else:
            new_list = _ordered_unique_ints(new_classes)

        old_set = set(old_list)
        new_set = set(new_list)
        if not old_set.issubset(seen_set):
            raise RuntimeError(f"old_classes not subset of seen_classes: old={sorted(old_set)}, seen={seen}")
        if not new_set.issubset(seen_set):
            raise RuntimeError(f"new_classes not subset of seen_classes: new={sorted(new_set)}, seen={seen}")
        if old_set & new_set:
            raise RuntimeError(f"old/new classes overlap: {sorted(old_set & new_set)}")

        old_mask = torch.tensor([c in old_set for c in seen], dtype=torch.bool)
        new_mask = torch.tensor([c in new_set for c in seen], dtype=torch.bool)

        old_prefix_count = int(old_mask.sum().item())
        if require_prefix and old_prefix_count > 0:
            expected = seen[:old_prefix_count]
            if set(expected) != old_set or any(c not in old_set for c in expected):
                raise RuntimeError(
                    "Old/new energy losses require old classes to be the prefix of seen_classes. "
                    f"seen={seen}, old={sorted(old_set)}"
                )

        return old_mask, new_mask, old_prefix_count

    def assert_logits_valid(
        self,
        logits: torch.Tensor,
        *,
        seen_classes: Sequence[int],
        targets: Optional[torch.Tensor] = None,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        context: str = "classifier",
    ) -> None:
        if not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError(f"{context}: logits must be [B,S], got {None if logits is None else tuple(logits.shape)}")
        S = len(seen_classes)
        if int(logits.size(1)) != S:
            raise RuntimeError(f"{context}: output width={logits.size(1)} but len(seen_classes)={S}")
        if not torch.isfinite(logits).all():
            bad = int((~torch.isfinite(logits)).sum().detach().cpu().item())
            raise RuntimeError(f"{context}: logits contain {bad} NaN/Inf values.")
        if targets is not None:
            y = _as_long_1d(targets, device=logits.device, name=f"{context}.targets")
            if y.numel() != int(logits.size(0)):
                raise RuntimeError(f"{context}: target/logit batch mismatch: {y.numel()} vs {logits.size(0)}")
            if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= S):
                raise RuntimeError(
                    f"{context}: local targets must be in [0,{S - 1}], got unique="
                    f"{torch.unique(y).detach().cpu().tolist()}"
                )
        self._old_new_masks(seen_classes, old_classes=old_classes, new_classes=new_classes)

    # ------------------------------------------------------------------
    # Core geometry energy
    # ------------------------------------------------------------------
    def _active_rank_mask(
        self,
        active_ranks: Optional[torch.Tensor],
        num_classes: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if active_ranks is None or not torch.is_tensor(active_ranks) or active_ranks.numel() != num_classes:
            ar = torch.full((num_classes,), rank, device=device, dtype=torch.long)
        else:
            ar = active_ranks.to(device=device).long().flatten().clamp(min=0, max=rank)
        mask = torch.arange(rank, device=device).view(1, rank) < ar.view(num_classes, 1)
        return mask.to(dtype=dtype), ar

    def compute_geometry_energy(
        self,
        features: torch.Tensor,
        *,
        seen_classes: Iterable[int],
        geometry_bank: Any,
        return_parts: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _finite_tensor(features, "features")
        if features.dim() != 2 or features.size(1) != self.d_model:
            raise RuntimeError(f"features must be [B,{self.d_model}], got {tuple(features.shape)}")

        seen = _as_seen_list(seen_classes)
        bank = self._bank_dict(geometry_bank)
        rows = self._select_bank_rows(bank, seen, device=features.device, dtype=features.dtype)

        means = rows["means"]
        bases = rows["bases"]
        eigvals = rows["eigvals"]
        res_vars = rows["res_vars"]
        reliability = rows["reliability"]
        active_ranks = rows["active_ranks"]

        S, D, R = bases.shape
        rank_mask, ar = self._active_rank_mask(active_ranks, S, R, features.device, features.dtype)

        delta = features.unsqueeze(1) - means.unsqueeze(0)                    # [B,S,D]
        coeff = torch.einsum("bsd,sdr->bsr", delta, bases)                  # [B,S,R]
        coeff_active = coeff * rank_mask.view(1, S, R)
        recon = torch.einsum("bsr,sdr->bsd", coeff_active, bases)
        residual = delta - recon

        eig = eigvals.clamp_min(self.variance_floor)
        rv = (res_vars * self.residual_variance_scale).clamp_min(self.variance_floor)

        parallel = ((coeff_active.pow(2) / eig.view(1, S, R)) * rank_mask.view(1, S, R)).sum(dim=-1)
        orthogonal = residual.pow(2).sum(dim=-1) / rv.view(1, S)

        energy = parallel + orthogonal
        if self.normalize_energy_by_dim:
            energy = energy / float(max(D, 1))

        logdet_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)
        if self.use_logdet_energy and self.logdet_energy_weight > 0.0:
            active_logdet = (eig.log() * rank_mask).sum(dim=1)
            residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=features.dtype)
            logdet_penalty = active_logdet + residual_dims * rv.log()
            if self.logdet_normalize_by_dim:
                logdet_penalty = logdet_penalty / float(max(D, 1))
            if self.center_logdet_energy:
                logdet_penalty = logdet_penalty - logdet_penalty.mean().detach()
            energy = energy + self.logdet_energy_weight * logdet_penalty.view(1, S)

        reliability_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)
        if self.use_reliability_penalty and self.reliability_energy_weight > 0.0 and reliability is not None:
            rel = torch.nan_to_num(
                reliability.to(device=features.device, dtype=features.dtype).flatten(),
                nan=self.reliability_min_clamp,
                posinf=1.0,
                neginf=self.reliability_min_clamp,
            ).clamp(self.reliability_min_clamp, 1.0)
            reliability_penalty = -rel.log()
            if self.center_reliability_energy:
                reliability_penalty = reliability_penalty - reliability_penalty.mean().detach()
            energy = energy + self.reliability_energy_weight * reliability_penalty.view(1, S)

        energy = torch.nan_to_num(energy, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0)

        parts: Dict[str, torch.Tensor] = {
            "energy": energy,
            "feature_energy": energy,
            "parallel": torch.nan_to_num(parallel, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "orthogonal": torch.nan_to_num(orthogonal, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "parallel_energy": torch.nan_to_num(parallel, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "residual_energy": torch.nan_to_num(orthogonal, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "logdet_penalty": logdet_penalty,
            "reliability_penalty": reliability_penalty,
            "active_ranks": ar,
            "rank_mask": rank_mask,
            "sample_counts": rows["sample_counts"],
            "global_class_ids": rows["global_class_ids"],
            "row_indices": rows["row_indices"],
        }
        return energy, parts if return_parts else {}

    def _energy_to_logits(self, energy: torch.Tensor) -> torch.Tensor:
        if energy.dim() != 2:
            raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
        row_min = energy.min(dim=1, keepdim=True).values
        rel = energy - row_min
        logits = -self.logit_scale * rel
        if self.logit_clip > 0.0:
            logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
        return torch.nan_to_num(logits, nan=self.invalid_logit, posinf=1e4, neginf=self.invalid_logit)

    def compute_geometry_logits(
        self,
        features: torch.Tensor,
        *,
        seen_classes: Iterable[int],
        geometry_bank: Any,
        return_parts: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        seen = _as_seen_list(seen_classes)
        energy, parts = self.compute_geometry_energy(
            features,
            seen_classes=seen,
            geometry_bank=geometry_bank,
            return_parts=True,
        )
        logits = self._energy_to_logits(energy)
        self.assert_logits_valid(logits, seen_classes=seen, context="compute_geometry_logits")
        if not return_parts:
            return logits
        out: Dict[str, torch.Tensor] = {"logits": logits, "energy": energy, "raw_energy": energy}
        out.update(parts)
        return out

    # Backward-compatible direct energy/logit methods.
    def geometry_energy(
        self,
        features: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        active_ranks: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
        return_parts: bool = False,
        **_: Any,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bank = {
            "means": means,
            "bases": bases,
            "variances": variances,
            "sample_counts": sample_counts,
        }
        if reliability is not None:
            bank["reliability"] = reliability
        if active_ranks is not None:
            bank["active_ranks"] = active_ranks
        seen = self._infer_seen_from_bank(bank)
        energy, parts = self.compute_geometry_energy(features, seen_classes=seen, geometry_bank=bank, return_parts=True)
        if return_parts:
            out = {"energy": energy}
            out.update(parts)
            return out
        return energy

    def geometry_logits(
        self,
        features: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        active_ranks: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        bank = {
            "means": means,
            "bases": bases,
            "variances": variances,
            "sample_counts": sample_counts,
        }
        if reliability is not None:
            bank["reliability"] = reliability
        if active_ranks is not None:
            bank["active_ranks"] = active_ranks
        seen = self._infer_seen_from_bank(bank)
        return self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank)

    def geometry_logits_from_bank(
        self,
        features: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        *,
        seen_classes: Optional[Iterable[int]] = None,
        apply_energy_calibration: bool = False,
        old_class_count: int = 0,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        return_parts: bool = False,
        **_: Any,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if seen_classes is None:
            seen_classes = self._infer_seen_from_bank(bank)
        seen = _as_seen_list(seen_classes)
        out = self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank, return_parts=True)
        logits = out["logits"]
        if apply_energy_calibration:
            logits = self.calibrate_old_new_logits(
                logits,
                seen_classes=seen,
                old_classes=old_classes,
                new_classes=new_classes,
                old_class_count=old_class_count,
            )
            out["logits"] = logits
        if return_parts:
            return out
        return logits

    def geometry_energy_from_bank(
        self,
        features: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        *,
        seen_classes: Optional[Iterable[int]] = None,
        return_parts: bool = False,
        **_: Any,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if seen_classes is None:
            seen_classes = self._infer_seen_from_bank(bank)
        energy, parts = self.compute_geometry_energy(features, seen_classes=seen_classes, geometry_bank=bank, return_parts=True)
        if return_parts:
            out = {"energy": energy}
            out.update(parts)
            return out
        return energy

    # Legacy aliases
    def _geometry_energy(
        self,
        f: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        vars_: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        active_ranks: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.geometry_energy(f, means, bases, vars_, reliability, active_ranks, sample_counts)

    def _geometry_logits(
        self,
        f: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        vars_: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        active_ranks: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.geometry_logits(f, means, bases, vars_, reliability, active_ranks, sample_counts)

    # ------------------------------------------------------------------
    # Calibration and diagnostics
    # ------------------------------------------------------------------
    def calibrate_old_new_logits(
        self,
        logits: torch.Tensor,
        *,
        seen_classes: Iterable[int],
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
    ) -> torch.Tensor:
        seen = _as_seen_list(seen_classes)
        old_mask, new_mask, _ = self._old_new_masks(
            seen,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
        )
        old_mask = old_mask.to(device=logits.device)
        new_mask = new_mask.to(device=logits.device)
        self.assert_logits_valid(logits, seen_classes=seen, context="calibrate_old_new_logits")
        if not self.use_old_new_calibration:
            return logits
        return self.calibrator(logits, old_mask, new_mask)

    @torch.no_grad()
    def classifier_diagnostics(
        self,
        logits: torch.Tensor,
        *,
        seen_classes: Iterable[int],
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
        targets_local: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        seen = _as_seen_list(seen_classes)
        self.assert_logits_valid(logits, seen_classes=seen, targets=targets_local, context="classifier_diagnostics")
        old_mask, new_mask, _ = self._old_new_masks(
            seen,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
        )
        old_mask = old_mask.to(device=logits.device)
        new_mask = new_mask.to(device=logits.device)

        pred_local = logits.argmax(dim=1)
        seen_tensor = torch.as_tensor(seen, device=logits.device, dtype=torch.long)
        pred_global = seen_tensor.index_select(0, pred_local) if pred_local.numel() else torch.empty((0,), device=logits.device, dtype=torch.long)
        counts = torch.bincount(pred_local, minlength=len(seen)).detach().cpu()

        old_logits = logits[:, old_mask] if bool(old_mask.any().item()) else logits.new_empty((logits.size(0), 0))
        new_logits = logits[:, new_mask] if bool(new_mask.any().item()) else logits.new_empty((logits.size(0), 0))

        old_mean = old_logits.mean() if old_logits.numel() else logits.sum() * 0.0
        new_mean = new_logits.mean() if new_logits.numel() else logits.sum() * 0.0
        old_max = old_logits.max() if old_logits.numel() else logits.sum() * 0.0
        new_max = new_logits.max() if new_logits.numel() else logits.sum() * 0.0

        out: Dict[str, Any] = {
            "seen_classes": [int(c) for c in seen],
            "classifier_output_dim": int(logits.size(1)),
            "prediction_global": pred_global.detach().cpu().tolist(),
            "old_logit_mean": float(old_mean.detach().cpu().item()),
            "new_logit_mean": float(new_mean.detach().cpu().item()),
            "old_new_logit_gap": float((old_mean - new_mean).detach().cpu().item()),
            "max_old_logit": float(old_max.detach().cpu().item()),
            "max_new_logit": float(new_max.detach().cpu().item()),
            "invalid_prediction_rate": 0.0,
            "prediction_distribution": {int(seen[i]): int(counts[i].item()) for i in range(len(seen))},
            "per_class_prediction_count": {int(seen[i]): int(counts[i].item()) for i in range(len(seen))},
            "calibration_bias_value": float(self.calibrator.bias_value().detach().cpu().item()) if self.use_old_new_calibration else 0.0,
        }

        if targets_local is not None:
            y = targets_local.to(device=logits.device).long().flatten()
            if y.numel() != logits.size(0):
                raise RuntimeError("classifier_diagnostics: targets/logits batch mismatch.")
            correct = pred_local.eq(y)
            out["accuracy"] = float(correct.float().mean().detach().cpu().item()) if y.numel() else 0.0
            old_y = old_mask.index_select(0, y) if y.numel() else torch.empty((0,), device=logits.device, dtype=torch.bool)
            new_y = new_mask.index_select(0, y) if y.numel() else torch.empty((0,), device=logits.device, dtype=torch.bool)
            out["old_accuracy"] = float(correct[old_y].float().mean().detach().cpu().item()) if bool(old_y.any().item()) else 0.0
            out["new_accuracy"] = float(correct[new_y].float().mean().detach().cpu().item()) if bool(new_y.any().item()) else 0.0
        return out

    @torch.no_grad()
    def energy_margin_statistics(
        self,
        energy: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del sample_counts
        if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
            z = self._zero * 0.0
            return {"mean_margin": z, "min_margin": z, "violation_rate": z, "accuracy": z}
        if energy.dim() != 2:
            raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
        y = _as_long_1d(labels, device=energy.device, name="labels")
        if y.numel() != energy.size(0):
            raise RuntimeError("labels/energy batch mismatch")
        if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
            raise RuntimeError("labels outside local energy range")
        true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
        true_mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
        nearest_wrong = energy.masked_fill(true_mask, float("inf")).min(dim=1).values
        margin = nearest_wrong - true_e
        pred = energy.argmin(dim=1)
        return {
            "mean_margin": margin.mean(),
            "min_margin": margin.min(),
            "violation_rate": (margin <= 0).float().mean(),
            "accuracy": (pred == y).float().mean(),
        }

    @torch.no_grad()
    def old_new_energy_statistics(
        self,
        energy: torch.Tensor,
        labels: torch.Tensor,
        *,
        old_class_count: int,
        sample_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del sample_counts
        z = energy.sum() * 0.0 if torch.is_tensor(energy) else self._zero * 0.0
        if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
            return {
                "new_into_old_rate": z,
                "old_into_new_rate": z,
                "old_group_win_rate": z,
                "new_group_win_rate": z,
                "mean_old_new_gap": z,
            }
        C = int(energy.size(1))
        old = int(max(0, min(old_class_count, C)))
        if old <= 0 or old >= C:
            return {
                "new_into_old_rate": z,
                "old_into_new_rate": z,
                "old_group_win_rate": z,
                "new_group_win_rate": z,
                "mean_old_new_gap": z,
            }
        y = _as_long_1d(labels, device=energy.device, name="labels")
        old_min = energy[:, :old].min(dim=1).values
        new_min = energy[:, old:].min(dim=1).values
        old_win = old_min < new_min
        new_win = new_min < old_min
        old_labels = y < old
        new_labels = y >= old
        return {
            "new_into_old_rate": old_win[new_labels].float().mean() if bool(new_labels.any().item()) else z,
            "old_into_new_rate": new_win[old_labels].float().mean() if bool(old_labels.any().item()) else z,
            "old_group_win_rate": old_win.float().mean(),
            "new_group_win_rate": new_win.float().mean(),
            "mean_old_new_gap": (new_min - old_min).mean(),
        }

    @torch.no_grad()
    def old_new_margin_report_from_energy(
        self,
        energy: torch.Tensor,
        labels: torch.Tensor,
        *,
        old_class_count: int,
        margin: float = 0.25,
        sample_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del sample_counts
        z = energy.sum() * 0.0
        if energy.dim() != 2:
            raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
        y = _as_long_1d(labels, device=energy.device, name="labels")
        if y.numel() != energy.size(0):
            raise RuntimeError("labels/energy batch mismatch")
        C = int(energy.size(1))
        old = int(max(0, min(int(old_class_count), C)))
        pred = energy.argmin(dim=1)
        acc = (pred == y).float().mean() if y.numel() else z

        if old <= 0 or old >= C:
            return {
                "accuracy": acc,
                "old_accuracy": z,
                "new_accuracy": acc,
                "hm": z,
                "old_win_rate": z,
                "new_win_rate": z,
                "new_into_old_rate": z,
                "old_into_new_rate": z,
                "new_margin_mean": z,
                "new_margin_min": z,
                "new_violation_rate": z,
                "old_boundary_margin_mean": z,
                "old_boundary_margin_min": z,
                "old_boundary_violation_rate": z,
                "mean_true_vs_opposite_margin": z,
            }

        old_labels = y < old
        new_labels = y >= old
        old_min = energy[:, :old].min(dim=1).values
        new_min = energy[:, old:].min(dim=1).values
        old_win = old_min < new_min
        new_win = new_min < old_min
        true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)

        new_margin = old_min[new_labels] - true_e[new_labels] if bool(new_labels.any().item()) else energy.new_empty((0,))
        old_margin = new_min[old_labels] - true_e[old_labels] if bool(old_labels.any().item()) else energy.new_empty((0,))
        old_acc = (pred[old_labels] == y[old_labels]).float().mean() if bool(old_labels.any().item()) else z
        new_acc = (pred[new_labels] == y[new_labels]).float().mean() if bool(new_labels.any().item()) else z
        hm = (2 * old_acc * new_acc / (old_acc + new_acc + 1e-8)) if bool(old_labels.any().item()) and bool(new_labels.any().item()) else z

        return {
            "accuracy": acc,
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "hm": hm,
            "old_win_rate": old_win.float().mean(),
            "new_win_rate": new_win.float().mean(),
            "new_into_old_rate": old_win[new_labels].float().mean() if bool(new_labels.any().item()) else z,
            "old_into_new_rate": new_win[old_labels].float().mean() if bool(old_labels.any().item()) else z,
            "new_margin_mean": new_margin.mean() if new_margin.numel() else z,
            "new_margin_min": new_margin.min() if new_margin.numel() else z,
            "new_violation_rate": (new_margin <= float(margin)).float().mean() if new_margin.numel() else z,
            "old_boundary_margin_mean": old_margin.mean() if old_margin.numel() else z,
            "old_boundary_margin_min": old_margin.min() if old_margin.numel() else z,
            "old_boundary_violation_rate": (old_margin <= float(margin)).float().mean() if old_margin.numel() else z,
            "mean_true_vs_opposite_margin": torch.cat([new_margin, old_margin]).mean() if (new_margin.numel() + old_margin.numel()) else z,
        }

    @torch.no_grad()
    def old_geometry_risk_features_from_bank(
        self,
        features: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        old_class_count: int,
    ) -> Dict[str, torch.Tensor]:
        if int(old_class_count) <= 0:
            z = torch.zeros((features.size(0),), device=features.device, dtype=features.dtype)
            return {
                "nearest_old_energy": z,
                "old_energy_margin": z,
                "nearest_old_reliability": torch.ones_like(z),
                "nearest_old_residual_variance": z,
                "nearest_old_class": torch.zeros_like(z, dtype=torch.long),
                "old_membership": z,
                "risk_features": torch.zeros((features.size(0), 4), device=features.device, dtype=features.dtype),
            }

        full_seen = self._infer_seen_from_bank(bank)
        old_seen = full_seen[: int(old_class_count)]
        energy, parts = self.compute_geometry_energy(features, seen_classes=old_seen, geometry_bank=bank, return_parts=True)
        C_old = int(energy.size(1))
        sorted_e, sorted_idx = torch.sort(energy, dim=1)
        nearest = sorted_e[:, 0]
        margin = sorted_e[:, 1] - sorted_e[:, 0] if C_old > 1 else torch.ones_like(nearest)
        nearest_local = sorted_idx[:, 0].long()
        nearest_global = parts["global_class_ids"].index_select(0, nearest_local)

        rel = torch.ones_like(nearest)
        res = torch.zeros_like(nearest)
        bank_dict = self._bank_dict(bank)
        if "reliability" in bank_dict and torch.is_tensor(bank_dict["reliability"]):
            row_idx = parts["row_indices"].index_select(0, nearest_local)
            rel_all = bank_dict["reliability"].to(device=features.device, dtype=features.dtype)
            rel = rel_all.index_select(0, row_idx).clamp(0.0, 1.0)
        if "res_vars" in bank_dict and torch.is_tensor(bank_dict["res_vars"]):
            row_idx = parts["row_indices"].index_select(0, nearest_local)
            rv_all = bank_dict["res_vars"].to(device=features.device, dtype=features.dtype)
            res = rv_all.index_select(0, row_idx).clamp_min(0.0)

        risk_features = torch.stack(
            [
                torch.log1p(nearest.clamp_min(0.0)),
                torch.log1p(margin.clamp_min(0.0)),
                rel,
                torch.log1p(res),
            ],
            dim=1,
        )
        risk_features = torch.nan_to_num(risk_features, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        return {
            "nearest_old_energy": nearest,
            "old_energy_margin": margin,
            "nearest_old_reliability": rel,
            "nearest_old_residual_variance": res,
            "nearest_old_class": nearest_global,
            "old_membership": torch.exp(-nearest.clamp_min(0.0)),
            "risk_features": risk_features,
        }

    # Candidate/admission reports kept as diagnostics for incremental phase.
    @torch.no_grad()
    def geometry_state_admission_report(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        geometry_bank: Any,
        seen_classes: Iterable[int],
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
        margin: float = 0.25,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        seen = _as_seen_list(seen_classes)
        out = self.forward(
            features,
            seen_classes=seen,
            geometry_bank=geometry_bank,
            targets=labels,
            targets_are_global=True,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
            return_energy=True,
            return_parts=True,
        )
        targets_local = self.global_to_local_labels(labels, seen)
        _, _, old_prefix = self._old_new_masks(
            seen,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
            require_prefix=True,
        )
        return self.old_new_margin_report_from_energy(
            out["energy"],
            targets_local,
            old_class_count=old_prefix,
            margin=margin,
        )

    sglat_candidate_admission_report = geometry_state_admission_report
    candidate_admission_report = geometry_state_admission_report

    @torch.no_grad()
    def transport_effect_report(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        del args, kwargs
        z = self._zero * 0.0
        return {"total": z, "new_violation_rate": z, "old_boundary_violation_rate": z}

    @torch.no_grad()
    def method_summary(self) -> Dict[str, object]:
        return {
            "method_path": "pg_rga_strict_low_rank_geometry",
            "architecture": "PG-RGA-HSI",
            "output_contract": "[B, len(seen_classes)]",
            "uses_geometry_bank": True,
            "uses_feature_low_rank_energy": True,
            "uses_low_rank_logdet_energy": bool(self.use_logdet_energy and self.logdet_energy_weight > 0.0),
            "uses_reliability_penalty": bool(self.use_reliability_penalty and self.reliability_energy_weight > 0.0),
            "uses_old_new_logit_calibration": bool(self.use_old_new_calibration),
            "uses_measured_energy_calibration": False,
            "uses_adaptive_boundary": False,
            "uses_spectral_classifier_energy": False,
            "uses_band_energy": False,
            "logit_scale": float(self.logit_scale),
            "variance_floor": float(self.variance_floor),
            "residual_variance_scale": float(self.residual_variance_scale),
            "logdet_energy_weight": float(self.logdet_energy_weight),
            "reliability_energy_weight": float(self.reliability_energy_weight),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        seen_classes: Optional[Iterable[int]] = None,
        geometry_bank: Any = None,
        *,
        bank: Any = None,
        mode: str = "geometry_only",
        targets: Optional[torch.Tensor] = None,
        targets_are_global: bool = False,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
        return_energy: bool = False,
        return_parts: bool = False,
        return_diagnostics: bool = False,
        **legacy_kwargs: Any,
    ) -> torch.Tensor | Dict[str, Any]:
        supplied_bank = geometry_bank if geometry_bank is not None else bank

        # Compatibility: older trainer may pass subspace tensors directly.
        if supplied_bank is None and "subspace_means" in legacy_kwargs:
            means = legacy_kwargs.get("subspace_means")
            bases = legacy_kwargs.get("subspace_bases")
            variances = legacy_kwargs.get("subspace_variances")
            if variances is None and "subspace_eigvals" in legacy_kwargs:
                eig = legacy_kwargs.get("subspace_eigvals")
                rv = legacy_kwargs.get("subspace_res_vars", legacy_kwargs.get("subspace_resvars"))
                if torch.is_tensor(eig) and torch.is_tensor(rv):
                    variances = torch.cat([eig, rv.unsqueeze(-1)], dim=-1)
            supplied_bank = {
                "means": means,
                "bases": bases,
                "variances": variances,
                "sample_counts": legacy_kwargs.get("subspace_sample_counts"),
                "reliability": legacy_kwargs.get("subspace_reliability"),
                "active_ranks": legacy_kwargs.get("subspace_active_ranks"),
            }

        if supplied_bank is None:
            raise ValueError("forward requires geometry_bank/bank or subspace_* tensors.")

        bank_dict = self._bank_dict(supplied_bank)
        if seen_classes is None:
            seen_classes = self._infer_seen_from_bank(bank_dict)
        seen = _as_seen_list(seen_classes)
        mode_norm = self.normalize_mode(mode)
        self.expand_to_seen_classes(seen)

        parts = self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank_dict, return_parts=True)
        logits = parts["logits"]

        if mode_norm == "calibrated_geometry":
            logits = self.calibrate_old_new_logits(
                logits,
                seen_classes=seen,
                old_classes=old_classes,
                new_classes=new_classes,
                old_class_count=old_class_count,
            )

        targets_local = None
        if targets is not None:
            targets_local = self.global_to_local_labels(targets, seen) if targets_are_global else targets.to(device=features.device).long().flatten()

        self.assert_logits_valid(
            logits,
            seen_classes=seen,
            targets=targets_local,
            old_classes=old_classes,
            new_classes=new_classes,
            context="classifier.forward",
        )

        if not (return_energy or return_parts or return_diagnostics):
            return logits

        out: Dict[str, Any] = {
            "logits": logits,
            "seen_classes": torch.as_tensor(seen, device=features.device, dtype=torch.long),
            "mode": mode_norm,
            "energy_calibrated": torch.tensor(mode_norm == "calibrated_geometry" and self.use_old_new_calibration, device=features.device),
        }
        if return_energy or return_parts:
            out["energy"] = parts["energy"]
        if return_parts:
            out.update(parts)
            out["logits"] = logits
        if return_diagnostics:
            out["diagnostics"] = self.classifier_diagnostics(
                logits,
                seen_classes=seen,
                old_classes=old_classes,
                new_classes=new_classes,
                old_class_count=old_class_count,
                targets_local=targets_local,
            )
        return out


# -----------------------------------------------------------------------------
# Standalone loss helpers used by trainers
# -----------------------------------------------------------------------------


def geometry_energy_margin_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del valid_mask
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return _zero_like(labels if torch.is_tensor(labels) else None)
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
        return _zero_like(labels if torch.is_tensor(labels) else None)
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





















# from __future__ import annotations
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# _EPS = 1e-12
# _INVALID_LOGIT = -1e9


# def _to_bool(x: object, default: bool = False) -> bool:
#     if x is None:
#         return bool(default)
#     if isinstance(x, bool):
#         return x
#     if isinstance(x, (int, float)):
#         return bool(x)
#     if isinstance(x, str):
#         v = x.strip().lower()
#         if v in {"1", "true", "yes", "y", "on"}:
#             return True
#         if v in {"0", "false", "no", "n", "off", "none", ""}:
#             return False
#     return bool(x)


# def _as_seen_list(seen_classes: Optional[Iterable[int]], *, fallback_count: Optional[int] = None) -> List[int]:
#     if seen_classes is None:
#         if fallback_count is None:
#             raise ValueError("seen_classes is required when it cannot be inferred from GeometryBank sample_counts.")
#         return list(range(int(fallback_count)))
#     out = [int(c) for c in seen_classes]
#     if not out:
#         raise ValueError("seen_classes is empty.")
#     if len(set(out)) != len(out):
#         raise ValueError(f"seen_classes contains duplicates: {out}")
#     if min(out) < 0:
#         raise ValueError(f"seen_classes contains negative class ids: {out}")
#     return out


# def _as_long_1d(x: torch.Tensor, *, device: torch.device, name: str) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     return x.to(device=device).long().flatten()


# def _finite(x: torch.Tensor, name: str) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     if not torch.isfinite(x).all():
#         bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
#         raise RuntimeError(f"{name} contains {bad} NaN/Inf values.")
#     return x


# def _tensor_from_bank(bank: Dict[str, torch.Tensor], *names: str) -> torch.Tensor:
#     for n in names:
#         if n in bank and torch.is_tensor(bank[n]):
#             return bank[n]
#     raise KeyError(f"GeometryBank missing required tensor. Tried keys={names}")


# class OldNewLogitCalibrator(nn.Module):
#     """Bounded old/new logit bias.

#     This module is intentionally tiny. It can correct a measured old/new prior
#     shift, but it cannot hide label bugs because the caller must provide valid
#     old/new masks in the current seen-class output space.
#     """

#     def __init__(self, max_abs_bias: float = 1.0) -> None:
#         super().__init__()
#         self.max_abs_bias = float(max(0.0, max_abs_bias))
#         self.old_bias_raw = nn.Parameter(torch.tensor(0.0))

#     def bias_value(self) -> torch.Tensor:
#         return self.max_abs_bias * torch.tanh(self.old_bias_raw)

#     def forward(self, logits: torch.Tensor, old_mask: torch.Tensor, new_mask: torch.Tensor) -> torch.Tensor:
#         if logits.dim() != 2:
#             raise ValueError(f"logits must be [B,S], got {tuple(logits.shape)}")
#         if old_mask.numel() != logits.size(1) or new_mask.numel() != logits.size(1):
#             raise ValueError("old/new masks must match logit width.")
#         if not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
#             return logits
#         b = self.bias_value().to(device=logits.device, dtype=logits.dtype)
#         out = logits.clone()
#         out[:, old_mask] = out[:, old_mask] + b
#         out[:, new_mask] = out[:, new_mask] - b
#         return out

#     @torch.no_grad()
#     def summary(self) -> Dict[str, float]:
#         return {"old_new_logit_bias": float(self.bias_value().detach().cpu().item())}


# class GeometryEnergyClassifier(nn.Module):
#     """Strict seen-class GeometryBank energy classifier for NECIL-HSI.

#     Contract:
#         - Input features are projected geometry-space features [B,D].
#         - GeometryBank stores class-level descriptors only.
#         - Output logits are [B, len(seen_classes)] in exactly seen_classes order.
#         - Targets used with the returned logits must be local seen indices.

#     The classifier has no raw-sample memory, no prototype/concept branch, no
#     adapter, no transport, and no hidden future-class columns. Legacy modes are
#     normalized to geometry scoring so training and evaluation cannot silently use
#     different metrics.
#     """

#     def __init__(
#         self,
#         initial_classes: int = 0,
#         d_model: int = 128,
#         logit_scale: float = 8.0,
#         variance_floor: float = 1e-4,
#         residual_variance_scale: float = 0.75,
#         normalize_energy_by_dim: bool = True,
#         use_logdet_energy: bool = True,
#         logdet_energy_weight: float = 0.05,
#         use_reliability_penalty: bool = True,
#         reliability_energy_weight: float = 0.03,
#         use_old_new_calibration: bool = False,
#         calibration_max_abs_bias: float = 1.0,
#         logit_clip: float = 0.0,
#         invalid_logit: float = _INVALID_LOGIT,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__()
#         # Accept old constructor names without activating old paths.
#         if "energy_normalize_by_dim" in kwargs:
#             normalize_energy_by_dim = _to_bool(kwargs.pop("energy_normalize_by_dim"), normalize_energy_by_dim)
#         if "use_energy_calibrator" in kwargs:
#             use_old_new_calibration = _to_bool(kwargs.pop("use_energy_calibrator"), use_old_new_calibration)
#         if "energy_calibrator_max_bias" in kwargs:
#             calibration_max_abs_bias = float(kwargs.pop("energy_calibrator_max_bias"))
#         self.use_adaptive_boundary = _to_bool(kwargs.pop("use_adaptive_boundary", False), False)
#         self.boundary_radius_min = float(kwargs.pop("boundary_radius_min", 0.50))
#         self.boundary_radius_max = float(kwargs.pop("boundary_radius_max", 2.00))
#         self.boundary_init_radius = float(kwargs.pop("boundary_init_radius", 1.00))
#         self.boundary_radius_reg_weight = float(kwargs.pop("boundary_radius_reg_weight", 0.01))
#         self.boundary_old_new_constraint_weight = float(kwargs.pop("boundary_old_new_constraint_weight", 0.20))
#         self.boundary_old_new_margin_base = float(kwargs.pop("boundary_old_new_margin_base", 0.05))
#         self.boundary_old_new_margin_scale = float(kwargs.pop("boundary_old_new_margin_scale", 0.25))
#         if self.boundary_radius_max <= self.boundary_radius_min:
#             self.boundary_radius_max = self.boundary_radius_min + 1e-3
#         self.boundary_init_radius = float(min(max(self.boundary_init_radius, self.boundary_radius_min), self.boundary_radius_max))
#         if "use_measured_energy_calibration" in kwargs and _to_bool(kwargs.pop("use_measured_energy_calibration"), False):
#             raise RuntimeError(
#                 "Measured old/new energy normalization is disabled. It hides geometry/label bugs."
#             )

#         self.num_classes = int(max(0, initial_classes))
#         self.d_model = int(d_model)
#         self.logit_scale = float(logit_scale)
#         self.variance_floor = float(max(variance_floor, 1e-12))
#         self.residual_variance_scale = float(max(residual_variance_scale, 1e-8))
#         self.normalize_energy_by_dim = _to_bool(normalize_energy_by_dim, True)
#         self.use_logdet_energy = _to_bool(use_logdet_energy, True)
#         self.logdet_energy_weight = float(max(logdet_energy_weight, 0.0))
#         self.use_reliability_penalty = _to_bool(use_reliability_penalty, True)
#         self.reliability_energy_weight = float(max(reliability_energy_weight, 0.0))
#         self.use_old_new_calibration = _to_bool(use_old_new_calibration, False)
#         self.logit_clip = float(max(logit_clip, 0.0))
#         self.invalid_logit = float(invalid_logit)
#         self.calibrator = OldNewLogitCalibrator(max_abs_bias=float(calibration_max_abs_bias))
#         if not self.use_old_new_calibration:
#             for p in self.calibrator.parameters():
#                 p.requires_grad = False

#         init_log_radius = math.log(max(float(self.boundary_init_radius), 1e-8))
#         initial_boundary = torch.full((int(max(0, initial_classes)),), init_log_radius, dtype=torch.float32)
#         self.boundary_log_radius = nn.Parameter(initial_boundary, requires_grad=bool(self.use_adaptive_boundary))
#         self.register_buffer("_boundary_grad_mask", torch.ones_like(initial_boundary), persistent=False)
#         self._boundary_hook_handle = None
#         self._register_boundary_mask_hook()

#         self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
#         self._last_seen_classes: List[int] = list(range(self.num_classes))

#     # ------------------------------------------------------------------
#     # Mode and compatibility
#     # ------------------------------------------------------------------
#     @staticmethod
#     def normalize_mode(mode: str) -> str:
#         m = str(mode or "geometry").lower().strip()
#         aliases = {
#             "geo": "geometry",
#             "geometry_only": "geometry",
#             "geometry-only": "geometry",
#             "feature_geometry": "geometry",
#             "low_rank_geometry": "geometry",
#             "replay": "geometry",
#             "synthetic_replay": "geometry",
#             "anchor": "geometry",
#             "anchor_concept": "geometry",
#             "anchor_concept_geometry": "geometry",
#             "srgp": "geometry",
#             "srgp_geometry": "geometry",
#             "spectral_geometry": "geometry",
#             "calibrated_geometry": "calibrated_geometry",
#             "topology_calibrated_geometry": "calibrated_geometry",
#         }
#         out = aliases.get(m, m)
#         if out not in {"geometry", "calibrated_geometry"}:
#             raise ValueError(f"Unsupported classifier mode {mode!r}. Use 'geometry' or 'calibrated_geometry'.")
#         return out

#     def expand(self, num_new_classes: int, phase: int = 0) -> None:
#         del phase
#         self.num_classes += int(max(0, num_new_classes))

#     def expand_to_seen_classes(self, seen_classes: Iterable[int]) -> None:
#         seen = _as_seen_list(seen_classes)
#         self._last_seen_classes = seen
#         self.num_classes = int(len(seen))

#     def freeze_all_adaptation(self) -> None:
#         for p in self.calibrator.parameters():
#             p.requires_grad = False

#     def unfreeze_all_adaptation(self) -> None:
#         if self.use_old_new_calibration:
#             for p in self.calibrator.parameters():
#                 p.requires_grad = True

#     def adaptation_regularization_loss(self, num_classes: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         del num_classes
#         if self.use_old_new_calibration:
#             loss = self.calibrator.old_bias_raw.pow(2)
#         else:
#             loss = self._zero * 0.0
#         z = self._zero * 0.0
#         return {"total": loss, "bias": loss, "temp": z, "alpha": z, "energy_cal": loss, "adaptive_boundary": z}

#     def energy_calibration_regularization_loss(self, num_classes: Optional[int] = None) -> torch.Tensor:
#         return self.adaptation_regularization_loss(num_classes=num_classes)["energy_cal"]

#     # ------------------------------------------------------------------
#     # Adaptive boundary radii
#     # ------------------------------------------------------------------
#     def _register_boundary_mask_hook(self) -> None:
#         """Mask gradients for frozen old boundary radii after parameter resizing.

#         PyTorch only allows hooks on tensors with ``requires_grad=True``.  The
#         classifier is constructed before the trainer decides whether adaptive
#         boundary radii are trainable, so construction may legitimately create
#         ``boundary_log_radius`` frozen.  In that state we must not register the
#         hook yet; ``unfreeze_all_boundary_radii`` registers it when the trainer
#         enables boundary training.
#         """
#         if not hasattr(self, "boundary_log_radius"):
#             return
#         try:
#             if self._boundary_hook_handle is not None:
#                 self._boundary_hook_handle.remove()
#         except Exception:
#             pass
#         self._boundary_hook_handle = None

#         if not bool(getattr(self.boundary_log_radius, "requires_grad", False)):
#             return

#         def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
#             mask = getattr(self, "_boundary_grad_mask", None)
#             if torch.is_tensor(mask) and mask.shape == grad.shape:
#                 return grad * mask.to(device=grad.device, dtype=grad.dtype)
#             return grad

#         self._boundary_hook_handle = self.boundary_log_radius.register_hook(_mask_grad)

#     def _ensure_boundary_capacity(
#         self,
#         num_classes: int,
#         *,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         """Ensure global class-wise boundary radii exist up to num_classes."""
#         count = int(max(0, num_classes))
#         device = device or self.boundary_log_radius.device
#         dtype = dtype or self.boundary_log_radius.dtype
#         cur = int(self.boundary_log_radius.numel())
#         if count <= cur:
#             if self.boundary_log_radius.device != device or self.boundary_log_radius.dtype != dtype:
#                 with torch.no_grad():
#                     vals = self.boundary_log_radius.detach().to(device=device, dtype=dtype)
#                     mask = self._boundary_grad_mask.detach().to(device=device, dtype=dtype)
#                 self.boundary_log_radius = nn.Parameter(vals, requires_grad=bool(self.use_adaptive_boundary))
#                 self._boundary_grad_mask = mask
#                 self._register_boundary_mask_hook()
#             return
#         init_log_radius = math.log(max(float(self.boundary_init_radius), 1e-8))
#         with torch.no_grad():
#             old = self.boundary_log_radius.detach().to(device=device, dtype=dtype)
#             extra = torch.full((count - cur,), init_log_radius, device=device, dtype=dtype)
#             new_param = torch.cat([old, extra], dim=0)
#             old_mask = self._boundary_grad_mask.detach().to(device=device, dtype=dtype) if torch.is_tensor(self._boundary_grad_mask) else torch.ones((cur,), device=device, dtype=dtype)
#             new_mask = torch.cat([old_mask, torch.ones((count - cur,), device=device, dtype=dtype)], dim=0)
#         self.boundary_log_radius = nn.Parameter(new_param, requires_grad=bool(self.use_adaptive_boundary))
#         self._boundary_grad_mask = new_mask
#         self._register_boundary_mask_hook()

#     def _boundary_radii_for_seen(
#         self,
#         seen_classes: Sequence[int],
#         *,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> torch.Tensor:
#         seen = [int(c) for c in seen_classes]
#         if not seen:
#             return torch.empty((0,), device=device, dtype=dtype)
#         self._ensure_boundary_capacity(max(seen) + 1, device=device, dtype=dtype)
#         idx = torch.as_tensor(seen, device=device, dtype=torch.long)
#         raw = self.boundary_log_radius.to(device=device, dtype=dtype).index_select(0, idx)
#         radii = raw.exp().clamp(min=float(self.boundary_radius_min), max=float(self.boundary_radius_max))
#         return radii

#     def _apply_adaptive_boundary_to_energy(
#         self,
#         energy: torch.Tensor,
#         seen_classes: Sequence[int],
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if not bool(getattr(self, "use_adaptive_boundary", False)):
#             return energy, energy.new_ones((energy.size(1),))
#         radii = self._boundary_radii_for_seen(seen_classes, device=energy.device, dtype=energy.dtype).clamp_min(1e-6)
#         if radii.numel() != energy.size(1):
#             return energy, energy.new_ones((energy.size(1),))
#         # Geometry energy with an adaptive class radius: larger rho expands the
#         # accepted region; log(rho) prevents unbounded expansion.
#         adjusted = energy / radii.view(1, -1) + radii.log().view(1, -1)
#         adjusted = torch.nan_to_num(adjusted, nan=1e6, posinf=1e6, neginf=0.0)
#         return adjusted, radii

#     def boundary_parameters(self) -> Iterable[nn.Parameter]:
#         return [self.boundary_log_radius]

#     def freeze_all_boundary_radii(self) -> None:
#         self.boundary_log_radius.requires_grad_(False)

#     def unfreeze_all_boundary_radii(self) -> None:
#         if bool(getattr(self, "use_adaptive_boundary", False)):
#             self.boundary_log_radius.requires_grad_(True)
#             if torch.is_tensor(self._boundary_grad_mask):
#                 self._boundary_grad_mask.fill_(1.0)
#             self._register_boundary_mask_hook()

#     def freeze_old_boundary_radii(self, old_class_count: int) -> None:
#         old = int(max(0, old_class_count))
#         self.unfreeze_all_boundary_radii()
#         self._ensure_boundary_capacity(max(old, int(self.boundary_log_radius.numel())))
#         if old > 0 and torch.is_tensor(self._boundary_grad_mask):
#             self._boundary_grad_mask[:old] = 0.0

#     def adaptive_boundary_state(self, num_classes: Optional[int] = None, old_class_count: int = 0) -> Dict[str, float]:
#         count = int(num_classes or self.boundary_log_radius.numel())
#         if count <= 0 or self.boundary_log_radius.numel() == 0:
#             return {
#                 "adaptive_boundary_enabled": float(bool(getattr(self, "use_adaptive_boundary", False))),
#                 "boundary_radius_mean": 0.0,
#                 "old_boundary_radius_mean": 0.0,
#                 "new_boundary_radius_mean": 0.0,
#             }
#         self._ensure_boundary_capacity(count)
#         radii = self.boundary_log_radius[:count].detach().exp().clamp(float(self.boundary_radius_min), float(self.boundary_radius_max))
#         old = int(max(0, min(old_class_count, count)))
#         old_r = radii[:old] if old > 0 else radii.new_empty((0,))
#         new_r = radii[old:] if old < count else radii.new_empty((0,))
#         return {
#             "adaptive_boundary_enabled": float(bool(getattr(self, "use_adaptive_boundary", False))),
#             "boundary_radius_mean": float(radii.mean().cpu().item()),
#             "boundary_radius_min": float(radii.min().cpu().item()),
#             "boundary_radius_max": float(radii.max().cpu().item()),
#             "old_boundary_radius_mean": float(old_r.mean().cpu().item()) if old_r.numel() else 0.0,
#             "new_boundary_radius_mean": float(new_r.mean().cpu().item()) if new_r.numel() else 0.0,
#         }

#     def adaptive_boundary_loss(
#         self,
#         logits: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         old_class_count: int = 0,
#         seen_classes: Optional[Iterable[int]] = None,
#         **_: Any,
#     ) -> Dict[str, torch.Tensor]:
#         """Trainable boundary-radius regularizer and old/new separation loss.

#         The trainer passes logits that already depend on boundary radii.  This
#         loss therefore gives gradients to boundary_log_radius without touching
#         old bank rows or raw samples.
#         """
#         ref = logits if torch.is_tensor(logits) else self.boundary_log_radius
#         z = ref.sum() * 0.0
#         if not bool(getattr(self, "use_adaptive_boundary", False)):
#             return {"total": z, "boundary": z.detach(), "old_new": z.detach(), "radius_reg": z.detach()}

#         # Radius regularization: keep rho close to 1 unless validation/replay
#         # gradients prove a class needs expansion or contraction.
#         seen = _as_seen_list(seen_classes, fallback_count=(logits.size(1) if torch.is_tensor(logits) and logits.dim() == 2 else None)) if seen_classes is not None or (torch.is_tensor(logits) and logits.dim() == 2) else list(range(int(self.boundary_log_radius.numel())))
#         radii = self._boundary_radii_for_seen(seen, device=ref.device, dtype=ref.dtype) if seen else ref.new_empty((0,))
#         radius_reg = (radii - 1.0).pow(2).mean() if radii.numel() else z

#         old_new = z
#         if torch.is_tensor(logits) and torch.is_tensor(labels) and logits.dim() == 2 and logits.size(1) >= 2:
#             y = labels.to(device=logits.device).long().flatten()
#             if y.numel() == logits.size(0) and y.numel() > 0 and int(y.min().item()) >= 0 and int(y.max().item()) < logits.size(1):
#                 C = int(logits.size(1))
#                 old = int(max(0, min(old_class_count, C)))
#                 if old > 0 and old < C:
#                     true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
#                     old_max = logits[:, :old].max(dim=1).values
#                     new_max = logits[:, old:].max(dim=1).values
#                     is_old = y < old
#                     is_new = ~is_old
#                     margin = float(self.boundary_old_new_margin_base) + float(self.boundary_old_new_margin_scale) * 0.1
#                     terms = []
#                     if bool(is_old.any().item()):
#                         terms.append(F.relu(new_max[is_old] + margin - true_logit[is_old]).mean())
#                     if bool(is_new.any().item()):
#                         terms.append(F.relu(old_max[is_new] + margin - true_logit[is_new]).mean())
#                     if terms:
#                         old_new = torch.stack(terms).mean()

#         total = float(self.boundary_radius_reg_weight) * radius_reg + float(self.boundary_old_new_constraint_weight) * old_new
#         return {
#             "total": total,
#             "boundary": old_new.detach(),
#             "old_new": old_new.detach(),
#             "radius_reg": radius_reg.detach(),
#         }

#     # ------------------------------------------------------------------
#     # Bank handling
#     # ------------------------------------------------------------------
#     def _bank_dict(self, geometry_bank: Any) -> Dict[str, torch.Tensor]:
#         if geometry_bank is None:
#             raise ValueError("geometry_bank is required for geometry scoring.")
#         if isinstance(geometry_bank, dict):
#             return geometry_bank
#         if hasattr(geometry_bank, "get_bank") and callable(geometry_bank.get_bank):
#             return geometry_bank.get_bank()
#         if hasattr(geometry_bank, "get_subspace_bank") and callable(geometry_bank.get_subspace_bank):
#             return geometry_bank.get_subspace_bank()
#         raise TypeError("geometry_bank must be a dict or an object exposing get_bank()/get_subspace_bank().")

#     def _infer_seen_from_bank(self, bank: Dict[str, torch.Tensor]) -> List[int]:
#         counts = _tensor_from_bank(bank, "sample_counts").detach().cpu().flatten()
#         valid = torch.isfinite(counts) & (counts > 0)
#         return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]

#     def _select_bank_rows(
#         self,
#         bank: Dict[str, torch.Tensor],
#         seen_classes: Sequence[int],
#         *,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> Dict[str, torch.Tensor]:
#         means = _tensor_from_bank(bank, "means").to(device=device, dtype=dtype)
#         bases = _tensor_from_bank(bank, "bases", "raw_bases").to(device=device, dtype=dtype)
#         eigvals = _tensor_from_bank(bank, "eigvals").to(device=device, dtype=dtype)
#         if "res_vars" in bank and torch.is_tensor(bank["res_vars"]):
#             res_vars = bank["res_vars"].to(device=device, dtype=dtype)
#         else:
#             res_vars = _tensor_from_bank(bank, "resvars").to(device=device, dtype=dtype)
#         counts = _tensor_from_bank(bank, "sample_counts").to(device=device, dtype=dtype)

#         if means.dim() != 2 or means.size(1) != self.d_model:
#             raise ValueError(f"bank means must be [C,{self.d_model}], got {tuple(means.shape)}")
#         if bases.dim() != 3 or bases.size(1) != self.d_model:
#             raise ValueError(f"bank bases must be [C,{self.d_model},R], got {tuple(bases.shape)}")
#         C_total = int(means.size(0))
#         bad = [c for c in seen_classes if c < 0 or c >= C_total]
#         if bad:
#             raise IndexError(f"seen_classes contain ids absent from GeometryBank: {bad}; bank_rows={C_total}")
#         idx = torch.as_tensor(list(seen_classes), device=device, dtype=torch.long)
#         counts_seen = counts.index_select(0, idx)
#         missing = [int(seen_classes[i]) for i in range(len(seen_classes)) if float(counts_seen[i].detach().cpu().item()) <= 0.0]
#         if missing:
#             raise RuntimeError(f"Geometry scoring requested classes with no geometry/sample_count: {missing}")

#         reliability = None
#         if "reliability" in bank and torch.is_tensor(bank["reliability"]):
#             reliability = bank["reliability"].to(device=device, dtype=dtype).index_select(0, idx)
#         active_ranks = None
#         if "active_ranks" in bank and torch.is_tensor(bank["active_ranks"]):
#             active_ranks = bank["active_ranks"].to(device=device).long().index_select(0, idx)

#         return {
#             "means": means.index_select(0, idx),
#             "bases": bases.index_select(0, idx),
#             "eigvals": eigvals.index_select(0, idx).clamp_min(self.variance_floor),
#             "res_vars": res_vars.index_select(0, idx).flatten().clamp_min(self.variance_floor),
#             "sample_counts": counts_seen.flatten(),
#             "reliability": reliability,
#             "active_ranks": active_ranks,
#             "global_class_ids": idx,
#         }

#     # ------------------------------------------------------------------
#     # Label mapping and assertions
#     # ------------------------------------------------------------------
#     @staticmethod
#     def global_to_local_labels(labels: torch.Tensor, seen_classes: Sequence[int]) -> torch.Tensor:
#         if not torch.is_tensor(labels):
#             raise TypeError("labels must be a tensor.")
#         seen = [int(c) for c in seen_classes]
#         mapping = {c: i for i, c in enumerate(seen)}
#         y = labels.long().flatten()
#         out = torch.empty_like(y)
#         bad: List[int] = []
#         for i, v in enumerate(y.detach().cpu().tolist()):
#             if int(v) not in mapping:
#                 bad.append(int(v))
#                 out[i] = -1
#             else:
#                 out[i] = mapping[int(v)]
#         if bad:
#             raise RuntimeError(f"labels contain classes not in seen_classes. bad={sorted(set(bad))}, seen={seen}")
#         return out.to(device=labels.device)

#     def _old_new_masks(
#         self,
#         seen_classes: Sequence[int],
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         seen = [int(c) for c in seen_classes]
#         if old_classes is None and old_class_count is not None:
#             old_set = set(seen[: int(max(0, min(old_class_count, len(seen))))])
#         else:
#             old_set = set(int(c) for c in (old_classes or []))
#         if new_classes is None:
#             new_set = set(seen) - old_set
#         else:
#             new_set = set(int(c) for c in new_classes)
#         if not old_set and not new_set:
#             old_set = set()
#             new_set = set(seen)
#         if not old_set.issubset(set(seen)):
#             raise RuntimeError(f"old_classes not subset of seen_classes: old={sorted(old_set)}, seen={seen}")
#         if not new_set.issubset(set(seen)):
#             raise RuntimeError(f"new_classes not subset of seen_classes: new={sorted(new_set)}, seen={seen}")
#         if old_set & new_set:
#             raise RuntimeError(f"old/new classes overlap: {sorted(old_set & new_set)}")
#         old_mask = torch.tensor([c in old_set for c in seen], dtype=torch.bool)
#         new_mask = torch.tensor([c in new_set for c in seen], dtype=torch.bool)
#         return old_mask, new_mask

#     def assert_logits_valid(
#         self,
#         logits: torch.Tensor,
#         *,
#         seen_classes: Sequence[int],
#         targets: Optional[torch.Tensor] = None,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         context: str = "classifier",
#     ) -> None:
#         if not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError(f"{context}: logits must be [B,S], got {None if logits is None else tuple(logits.shape)}")
#         S = len(seen_classes)
#         if int(logits.size(1)) != S:
#             raise RuntimeError(f"{context}: classifier output width={logits.size(1)} but len(seen_classes)={S}")
#         if not torch.isfinite(logits).all():
#             bad = int((~torch.isfinite(logits)).sum().detach().cpu().item())
#             raise RuntimeError(f"{context}: logits contain {bad} NaN/Inf values.")
#         if targets is not None:
#             y = _as_long_1d(targets, device=logits.device, name=f"{context}.targets")
#             if y.numel() != int(logits.size(0)):
#                 raise RuntimeError(f"{context}: target/logit batch mismatch: {y.numel()} vs {logits.size(0)}")
#             if y.numel() > 0 and (int(y.min().item()) < 0 or int(y.max().item()) >= S):
#                 raise RuntimeError(
#                     f"{context}: local targets must be in [0,{S - 1}], got unique={torch.unique(y).detach().cpu().tolist()}"
#                 )
#         old_mask, new_mask = self._old_new_masks(seen_classes, old_classes=old_classes, new_classes=new_classes)
#         if old_classes is not None and not bool(old_mask.any().item()):
#             raise RuntimeError(f"{context}: old_classes provided but old mask is empty.")
#         if new_classes is not None and not bool(new_mask.any().item()):
#             raise RuntimeError(f"{context}: new_classes provided but new mask is empty.")

#     # ------------------------------------------------------------------
#     # Core geometry energy/logit computation
#     # ------------------------------------------------------------------
#     def _active_rank_mask(self, active_ranks: Optional[torch.Tensor], S: int, R: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#         if active_ranks is None or active_ranks.numel() != S:
#             ar = torch.full((S,), R, device=device, dtype=torch.long)
#         else:
#             ar = active_ranks.to(device=device).long().flatten().clamp(min=0, max=R)
#         mask = torch.arange(R, device=device).view(1, R) < ar.view(S, 1)
#         return mask, ar

#     def compute_geometry_energy(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         _finite(features, "features")
#         if features.dim() != 2 or features.size(1) != self.d_model:
#             raise RuntimeError(f"features must be [B,{self.d_model}], got {tuple(features.shape)}")
#         seen = _as_seen_list(seen_classes)
#         bank = self._bank_dict(geometry_bank)
#         rows = self._select_bank_rows(bank, seen, device=features.device, dtype=features.dtype)

#         means = rows["means"]
#         bases = rows["bases"]
#         eigvals = rows["eigvals"]
#         res_vars = rows["res_vars"]
#         reliability = rows["reliability"]
#         active_ranks = rows["active_ranks"]

#         S, D, R = bases.shape
#         if S != len(seen):
#             raise RuntimeError("internal bank selection width mismatch.")
#         mask, ar = self._active_rank_mask(active_ranks, S, R, features.device)
#         mask_f = mask.to(dtype=features.dtype)

#         delta = features.unsqueeze(1) - means.unsqueeze(0)                    # [B,S,D]
#         coeff = torch.einsum("bsd,sdr->bsr", delta, bases)                  # [B,S,R]
#         coeff_active = coeff * mask_f.view(1, S, R)
#         recon = torch.einsum("bsr,sdr->bsd", coeff_active, bases)
#         residual = delta - recon

#         eig = eigvals.clamp_min(self.variance_floor)
#         rv = (res_vars * self.residual_variance_scale).clamp_min(self.variance_floor)
#         parallel = ((coeff_active.pow(2) / eig.view(1, S, R)) * mask_f.view(1, S, R)).sum(dim=-1)
#         orthogonal = residual.pow(2).sum(dim=-1) / rv.view(1, S)
#         energy = parallel + orthogonal
#         if self.normalize_energy_by_dim:
#             energy = energy / float(max(D, 1))

#         logdet = torch.zeros((S,), device=features.device, dtype=features.dtype)
#         if self.use_logdet_energy and self.logdet_energy_weight > 0.0:
#             log_eig = eig.log() * mask_f
#             residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=features.dtype)
#             logdet = (log_eig.sum(dim=1) + residual_dims * rv.log()) / float(max(D, 1))
#             logdet = logdet - logdet.mean().detach()
#             energy = energy + self.logdet_energy_weight * logdet.view(1, S)

#         reliability_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)
#         if self.use_reliability_penalty and self.reliability_energy_weight > 0.0 and reliability is not None:
#             rel = reliability.to(device=features.device, dtype=features.dtype).flatten().clamp(1e-6, 1.0)
#             reliability_penalty = -rel.log()
#             reliability_penalty = reliability_penalty - reliability_penalty.mean().detach()
#             energy = energy + self.reliability_energy_weight * reliability_penalty.view(1, S)

#         energy = torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=0.0)
#         parts = {
#             "energy": energy,
#             "parallel_energy": parallel,
#             "residual_energy": orthogonal,
#             "logdet_penalty": logdet,
#             "reliability_penalty": reliability_penalty,
#             "active_ranks": ar,
#             "sample_counts": rows["sample_counts"],
#             "global_class_ids": rows["global_class_ids"],
#         }
#         return energy, parts

#     def _energy_to_logits(self, energy: torch.Tensor) -> torch.Tensor:
#         if energy.dim() != 2:
#             raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#         row_min = energy.min(dim=1, keepdim=True).values
#         logits = -self.logit_scale * (energy - row_min)
#         if self.logit_clip > 0:
#             logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
#         return torch.nan_to_num(logits, nan=self.invalid_logit, posinf=1e4, neginf=self.invalid_logit)

#     def compute_geometry_logits(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#         return_parts: bool = False,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         seen = _as_seen_list(seen_classes)
#         energy_raw, parts = self.compute_geometry_energy(features, seen_classes=seen, geometry_bank=geometry_bank)
#         energy, boundary_radii = self._apply_adaptive_boundary_to_energy(energy_raw, seen)
#         logits = self._energy_to_logits(energy)
#         self.assert_logits_valid(logits, seen_classes=seen, context="compute_geometry_logits")
#         if return_parts:
#             out = {"logits": logits, "energy": energy, "raw_energy": energy_raw, "boundary_radii": boundary_radii}
#             out.update(parts)
#             out["energy"] = energy
#             out["raw_energy"] = energy_raw
#             out["boundary_radii"] = boundary_radii
#             return out
#         return logits

#     def calibrate_old_new_logits(
#         self,
#         logits: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#     ) -> torch.Tensor:
#         seen = _as_seen_list(seen_classes)
#         old_mask, new_mask = self._old_new_masks(seen, old_classes=old_classes, new_classes=new_classes, old_class_count=old_class_count)
#         old_mask = old_mask.to(device=logits.device)
#         new_mask = new_mask.to(device=logits.device)
#         self.assert_logits_valid(logits, seen_classes=seen, old_classes=[seen[i] for i, m in enumerate(old_mask.cpu().tolist()) if m], new_classes=[seen[i] for i, m in enumerate(new_mask.cpu().tolist()) if m], context="calibrate_old_new_logits")
#         if not self.use_old_new_calibration:
#             return logits
#         return self.calibrator(logits, old_mask, new_mask)

#     # Backward-compatible names.
#     def geometry_logits_from_bank(self, features: torch.Tensor, bank: Dict[str, torch.Tensor], *, seen_classes: Optional[Iterable[int]] = None, apply_energy_calibration: bool = False, old_class_count: int = 0, return_parts: bool = False, **_: Any) -> torch.Tensor | Dict[str, torch.Tensor]:
#         if seen_classes is None:
#             seen_classes = self._infer_seen_from_bank(bank)
#         out = self.compute_geometry_logits(features, seen_classes=seen_classes, geometry_bank=bank, return_parts=return_parts)
#         if return_parts:
#             logits = out["logits"]
#             if apply_energy_calibration:
#                 out["logits"] = self.calibrate_old_new_logits(logits, seen_classes=seen_classes, old_class_count=old_class_count)
#             return out
#         if apply_energy_calibration:
#             return self.calibrate_old_new_logits(out, seen_classes=seen_classes, old_class_count=old_class_count)
#         return out

#     def geometry_energy_from_bank(self, features: torch.Tensor, bank: Dict[str, torch.Tensor], *, seen_classes: Optional[Iterable[int]] = None, return_parts: bool = False, **_: Any) -> torch.Tensor | Dict[str, torch.Tensor]:
#         if seen_classes is None:
#             seen_classes = self._infer_seen_from_bank(bank)
#         energy, parts = self.compute_geometry_energy(features, seen_classes=seen_classes, geometry_bank=bank)
#         if return_parts:
#             out = {"energy": energy}
#             out.update(parts)
#             return out
#         return energy

#     # ------------------------------------------------------------------
#     # Diagnostics
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def classifier_diagnostics(
#         self,
#         logits: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         targets_local: Optional[torch.Tensor] = None,
#     ) -> Dict[str, Any]:
#         seen = _as_seen_list(seen_classes)
#         self.assert_logits_valid(logits, seen_classes=seen, targets=targets_local, context="classifier_diagnostics")
#         old_mask, new_mask = self._old_new_masks(seen, old_classes=old_classes, new_classes=new_classes, old_class_count=old_class_count)
#         old_mask = old_mask.to(device=logits.device)
#         new_mask = new_mask.to(device=logits.device)
#         pred_local = logits.argmax(dim=1)
#         pred_global = torch.as_tensor(seen, device=logits.device, dtype=torch.long).index_select(0, pred_local)
#         counts = torch.bincount(pred_local, minlength=len(seen)).detach().cpu()
#         old_logits = logits[:, old_mask] if bool(old_mask.any().item()) else logits.new_empty((logits.size(0), 0))
#         new_logits = logits[:, new_mask] if bool(new_mask.any().item()) else logits.new_empty((logits.size(0), 0))
#         old_mean = old_logits.mean() if old_logits.numel() else logits.sum() * 0.0
#         new_mean = new_logits.mean() if new_logits.numel() else logits.sum() * 0.0
#         old_max = old_logits.max() if old_logits.numel() else logits.sum() * 0.0
#         new_max = new_logits.max() if new_logits.numel() else logits.sum() * 0.0
#         out: Dict[str, Any] = {
#             "seen_classes": [int(c) for c in seen],
#             "classifier_output_dim": int(logits.size(1)),
#             "old_logit_mean": float(old_mean.detach().cpu().item()),
#             "new_logit_mean": float(new_mean.detach().cpu().item()),
#             "old_new_logit_gap": float((old_mean - new_mean).detach().cpu().item()),
#             "max_old_logit": float(old_max.detach().cpu().item()),
#             "max_new_logit": float(new_max.detach().cpu().item()),
#             "invalid_prediction_rate": 0.0,
#             "prediction_distribution": {int(seen[i]): int(counts[i].item()) for i in range(len(seen))},
#             "per_class_prediction_count": {int(seen[i]): int(counts[i].item()) for i in range(len(seen))},
#             "calibration_bias_value": float(self.calibrator.bias_value().detach().cpu().item()) if self.use_old_new_calibration else 0.0,
#         }
#         if targets_local is not None:
#             y = targets_local.to(device=logits.device).long().flatten()
#             correct = pred_local.eq(y)
#             out["accuracy"] = float(correct.float().mean().detach().cpu().item()) if y.numel() else 0.0
#             old_y = old_mask.index_select(0, y) if y.numel() else torch.empty((0,), device=logits.device, dtype=torch.bool)
#             new_y = new_mask.index_select(0, y) if y.numel() else torch.empty((0,), device=logits.device, dtype=torch.bool)
#             out["old_accuracy"] = float(correct[old_y].float().mean().detach().cpu().item()) if bool(old_y.any().item()) else 0.0
#             out["new_accuracy"] = float(correct[new_y].float().mean().detach().cpu().item()) if bool(new_y.any().item()) else 0.0
#         return out

#     @torch.no_grad()
#     def energy_margin_statistics(self, energy: torch.Tensor, labels: torch.Tensor, *, sample_counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         del sample_counts
#         if energy.dim() != 2:
#             raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#         y = _as_long_1d(labels, device=energy.device, name="labels")
#         if y.numel() != energy.size(0):
#             raise RuntimeError("labels/energy batch mismatch")
#         if y.numel() > 0 and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
#             raise RuntimeError("labels outside local energy range")
#         true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#         mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
#         wrong = energy.masked_fill(mask, float("inf")).min(dim=1).values
#         margin = wrong - true_e
#         pred = energy.argmin(dim=1)
#         return {
#             "mean_margin": margin.mean(),
#             "min_margin": margin.min(),
#             "violation_rate": (margin <= 0).float().mean(),
#             "accuracy": (pred == y).float().mean(),
#         }

#     @torch.no_grad()
#     def old_new_energy_statistics(self, energy: torch.Tensor, labels: torch.Tensor, *, old_class_count: int, sample_counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         del sample_counts
#         z = energy.sum() * 0.0
#         C = int(energy.size(1))
#         old = int(max(0, min(old_class_count, C)))
#         if old <= 0 or old >= C:
#             return {"new_into_old_rate": z, "old_into_new_rate": z, "old_group_win_rate": z, "new_group_win_rate": z, "mean_old_new_gap": z}
#         y = _as_long_1d(labels, device=energy.device, name="labels")
#         old_min = energy[:, :old].min(dim=1).values
#         new_min = energy[:, old:].min(dim=1).values
#         old_win = old_min < new_min
#         new_win = new_min < old_min
#         old_labels = y < old
#         new_labels = y >= old
#         return {
#             "new_into_old_rate": old_win[new_labels].float().mean() if bool(new_labels.any().item()) else z,
#             "old_into_new_rate": new_win[old_labels].float().mean() if bool(old_labels.any().item()) else z,
#             "old_group_win_rate": old_win.float().mean(),
#             "new_group_win_rate": new_win.float().mean(),
#             "mean_old_new_gap": (old_min - new_min).mean(),
#         }

#     @torch.no_grad()
#     def method_summary(self) -> Dict[str, object]:
#         return {
#             "method_path": "strict_geometry_seen_space",
#             "output_contract": "[B,len(seen_classes)]",
#             "uses_geometry_bank": True,
#             "uses_anchor_concept_branch": False,
#             "uses_adaptive_boundary": bool(getattr(self, "use_adaptive_boundary", False)),
#             "uses_measured_energy_calibration": False,
#             "uses_old_new_logit_calibration": bool(self.use_old_new_calibration),
#             "logit_scale": float(self.logit_scale),
#             "variance_floor": float(self.variance_floor),
#             "residual_variance_scale": float(self.residual_variance_scale),
#             "logdet_energy_weight": float(self.logdet_energy_weight),
#             "reliability_energy_weight": float(self.reliability_energy_weight),
#         }

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------
#     def forward(
#         self,
#         features: torch.Tensor,
#         seen_classes: Optional[Iterable[int]] = None,
#         geometry_bank: Any = None,
#         *,
#         bank: Any = None,
#         mode: str = "geometry",
#         targets: Optional[torch.Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#         **legacy_kwargs: Any,
#     ) -> torch.Tensor | Dict[str, Any]:
#         # Compatibility: old model may pass a bank dict under geometry_bank/bank,
#         # or pass subspace tensors directly.
#         supplied_bank = geometry_bank if geometry_bank is not None else bank
#         if supplied_bank is None and "subspace_means" in legacy_kwargs:
#             means = legacy_kwargs.get("subspace_means")
#             bases = legacy_kwargs.get("subspace_bases")
#             variances = legacy_kwargs.get("subspace_variances")
#             if variances is None and "subspace_eigvals" in legacy_kwargs:
#                 eig = legacy_kwargs.get("subspace_eigvals")
#                 rv = legacy_kwargs.get("subspace_res_vars", legacy_kwargs.get("subspace_resvars"))
#                 if torch.is_tensor(eig) and torch.is_tensor(rv):
#                     variances = torch.cat([eig, rv.unsqueeze(-1)], dim=-1)
#             if variances is not None and torch.is_tensor(variances):
#                 supplied_bank = {
#                     "means": means,
#                     "bases": bases,
#                     "eigvals": variances[:, :-1],
#                     "res_vars": variances[:, -1],
#                     "sample_counts": legacy_kwargs.get("subspace_sample_counts"),
#                     "reliability": legacy_kwargs.get("subspace_reliability"),
#                     "active_ranks": legacy_kwargs.get("subspace_active_ranks"),
#                 }
#         if supplied_bank is None:
#             raise ValueError("forward requires geometry_bank/bank or subspace_* tensors.")

#         bank_dict = self._bank_dict(supplied_bank)
#         if seen_classes is None:
#             seen_classes = self._infer_seen_from_bank(bank_dict)
#         seen = _as_seen_list(seen_classes)
#         mode_norm = self.normalize_mode(mode)
#         self.expand_to_seen_classes(seen)

#         parts = self.compute_geometry_logits(
#             features,
#             seen_classes=seen,
#             geometry_bank=bank_dict,
#             return_parts=True,
#         )
#         logits = parts["logits"]
#         if mode_norm == "calibrated_geometry":
#             logits = self.calibrate_old_new_logits(
#                 logits,
#                 seen_classes=seen,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#                 old_class_count=old_class_count,
#             )
#         targets_local = None
#         if targets is not None:
#             targets_local = self.global_to_local_labels(targets, seen) if targets_are_global else targets.to(device=features.device).long().flatten()
#         self.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets=targets_local,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             context="classifier.forward",
#         )
#         if not (return_energy or return_parts or return_diagnostics):
#             return logits
#         out: Dict[str, Any] = {
#             "logits": logits,
#             "seen_classes": torch.as_tensor(seen, device=features.device, dtype=torch.long),
#             "mode": mode_norm,
#             "energy_calibrated": torch.tensor(mode_norm == "calibrated_geometry" and self.use_old_new_calibration, device=features.device),
#         }
#         if return_energy or return_parts:
#             out["energy"] = parts["energy"]
#         if return_parts:
#             out.update(parts)
#             out["logits"] = logits
#         if return_diagnostics:
#             out["diagnostics"] = self.classifier_diagnostics(
#                 logits,
#                 seen_classes=seen,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#                 old_class_count=old_class_count,
#                 targets_local=targets_local,
#             )
#         return out


# # Backward-compatible class aliases used by older model code.
# SemanticClassifier = GeometryEnergyClassifier
# NECILClassifier = GeometryEnergyClassifier


# # -----------------------------------------------------------------------------
# # Loss helpers retained for trainer compatibility
# # -----------------------------------------------------------------------------


# def geometry_energy_margin_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         device = labels.device if torch.is_tensor(labels) else torch.device("cpu")
#         return torch.tensor(0.0, device=device)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() > 0 and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
#         raise RuntimeError("labels outside local energy range")
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
#     nearest_wrong = energy.masked_fill(mask, float("inf")).min(dim=1).values
#     loss = F.relu(true_e + float(margin) - nearest_wrong)
#     return loss[torch.isfinite(loss)].mean() if bool(torch.isfinite(loss).any().item()) else energy.sum() * 0.0


# def old_new_invasion_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     old_class_count: int,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         device = labels.device if torch.is_tensor(labels) else torch.device("cpu")
#         return torch.tensor(0.0, device=device)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     C = int(energy.size(1))
#     old = int(max(0, min(old_class_count, C)))
#     if old <= 0 or old >= C:
#         return energy.sum() * 0.0
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() > 0 and (int(y.min().item()) < 0 or int(y.max().item()) >= C):
#         raise RuntimeError("labels outside local energy range")
#     old_min = energy[:, :old].min(dim=1).values
#     new_min = energy[:, old:].min(dim=1).values
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     is_old = y < old
#     opposite = torch.where(is_old, new_min, old_min)
#     loss = F.relu(true_e + float(margin) - opposite)
#     return loss[torch.isfinite(loss)].mean() if bool(torch.isfinite(loss).any().item()) else energy.sum() * 0.0
