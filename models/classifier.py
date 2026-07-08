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


def _tensor_from_bank(bank: Mapping[str, Any], *names: str, required: bool = True) -> Optional[torch.Tensor]:
    for name in names:
        value = bank.get(name, None)
        if torch.is_tensor(value):
            return value
    if required:
        raise KeyError(f"GeometryBank is missing required tensor. Tried keys={names}")
    return None


def _truthy_legacy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _to_bool(value, False)
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if torch.is_tensor(value):
        return bool(value.numel() > 0 and float(value.detach().abs().max().cpu().item()) != 0.0)
    return bool(value)


# -----------------------------------------------------------------------------
# Strict low-rank geometry classifier
# -----------------------------------------------------------------------------


class GeometryEnergyClassifier(nn.Module):
    """Strict seen-class low-rank GeometryBank classifier for HSI NECIL.

    Method contract:
        * Input features are canonical projected z-space features [B, D].
        * GeometryBank stores compact low-rank descriptors only.
        * Output logits are [B, len(seen_classes)] in exactly seen_classes order.
        * CE targets for returned logits must be seen-local labels.
        * No prototype classifier.
        * No direct spectral/band inference branch.
        * No old/new score calibrator.
        * No adaptive boundary branch.

    Energy:
        E_c(z) = mean_j[(u_j^T(z-mu_c))^2 / lambda_j]
               + ||P_c^perp(z-mu_c)||^2 / ((D-r_c) sigma_perp,c^2).

    The scoring energy is deliberately identical to GeometryBank replay
    validation. Reliability, spectral descriptors, and coupling confidence
    control replay allocation and descriptor trust; they do not bias logits.
    """

    _DISALLOWED_TRUTHY_ARGS = {
        "use_old_new_calibration",
        "use_energy_calibrator",
        "use_measured_energy_calibration",
        "use_adaptive_boundary",
        "use_spectral_geometry",
        "use_spectral_residual_energy",
    }
    _DISALLOWED_NONZERO_ARGS = {
        "spectral_energy_weight",
        "band_energy_weight",
        "energy_calibrator_weight",
        "adapter_energy_weight",
        "logdet_energy_weight",
        "reliability_energy_weight",
    }

    def __init__(
        self,
        initial_classes: int = 0,
        d_model: int = 128,
        logit_scale: float = 8.0,
        variance_floor: float = 1e-4,
        residual_variance_scale: float = 1.0,
        normalize_energy_by_dim: bool = True,
        energy_normalize_by_dim: Optional[bool] = None,
        use_logdet_energy: bool = False,
        logdet_energy_weight: float = 0.0,
        logdet_normalize_by_dim: bool = True,
        center_logdet_energy: bool = False,
        use_reliability_penalty: bool = False,
        reliability_energy_weight: float = 0.0,
        reliability_min_clamp: float = 0.05,
        center_reliability_energy: bool = False,
        logit_clip: float = 0.0,
        invalid_logit: float = _INVALID_LOGIT,
        invalid_class_energy: float = 1e6,
        # Legacy arguments are accepted only when they are explicitly off.
        energy_calibrator_type: str = "none",
        calibration_max_abs_bias: float = 1.0,
        energy_calibrator_max_bias: Optional[float] = None,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        del calibration_max_abs_bias, energy_calibrator_max_bias

        ct = str(energy_calibrator_type or "none").strip().lower()
        if ct not in {"", "none", "off", "false"}:
            raise RuntimeError(
                "Classifier calibrators are removed from the main method. "
                "Use low-rank geometry replay and residual geometry adaptation losses instead."
            )
        for name in sorted(self._DISALLOWED_TRUTHY_ARGS):
            if name in legacy_kwargs and _truthy_legacy(legacy_kwargs[name]):
                raise RuntimeError(
                    f"{name} is disabled. The classifier has one strict path: "
                    "seen-class low-rank geometry energy."
                )
        for name in sorted(self._DISALLOWED_NONZERO_ARGS):
            if name in legacy_kwargs and _truthy_legacy(legacy_kwargs[name]):
                raise RuntimeError(
                    f"{name} is disabled. Spectral/band signals belong in base loss, "
                    "GeometryBank risk reports, and replay scheduling, not classifier logits."
                )

        # The replay filter in GeometryBank uses only rank-normalized parallel
        # and orthogonal energy. Adding logdet/reliability terms here would make
        # training, replay acceptance, and stored energy quantiles inconsistent.
        if _to_bool(use_logdet_energy, False) or float(logdet_energy_weight) != 0.0:
            raise RuntimeError(
                "Log-determinant scoring is disabled in the strict classifier. "
                "Set use_logdet_energy=false and logdet_energy_weight=0.0."
            )
        if _to_bool(use_reliability_penalty, False) or float(reliability_energy_weight) != 0.0:
            raise RuntimeError(
                "Reliability must control replay allocation/trust, not classifier logits. "
                "Set use_reliability_penalty=false and reliability_energy_weight=0.0."
            )
        if abs(float(residual_variance_scale) - 1.0) > 1e-8:
            raise RuntimeError(
                "residual_variance_scale must be 1.0 so classifier energy exactly matches "
                "GeometryBank replay-validation energy."
            )

        if energy_normalize_by_dim is not None:
            normalize_energy_by_dim = _to_bool(energy_normalize_by_dim, True)

        self.num_classes = int(max(0, initial_classes))
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        self.logit_scale = float(logit_scale)
        if self.logit_scale <= 0.0:
            raise ValueError("logit_scale must be positive.")
        self.variance_floor = float(max(variance_floor, 1e-12))
        self.residual_variance_scale = float(max(residual_variance_scale, 1e-8))
        self.normalize_energy_by_dim = _to_bool(normalize_energy_by_dim, True)
        self.use_logdet_energy = False
        self.logdet_energy_weight = 0.0
        self.logdet_normalize_by_dim = _to_bool(logdet_normalize_by_dim, True)
        self.center_logdet_energy = False
        self.use_reliability_penalty = False
        self.reliability_energy_weight = 0.0
        self.reliability_min_clamp = float(max(min(reliability_min_clamp, 1.0), 1e-8))
        self.center_reliability_energy = False
        self.logit_clip = float(max(logit_clip, 0.0))
        self.invalid_logit = float(invalid_logit)
        self.invalid_class_energy = float(max(invalid_class_energy, 1.0))

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self._last_seen_classes: List[int] = list(range(self.num_classes))

    # ------------------------------------------------------------------
    # Compatibility controls: strict no-op or hard error
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
            "low-rank-geometry": "geometry_only",
            "geometry_replay": "geometry_only",
        }
        out = aliases.get(m, m)
        if out != "geometry_only":
            raise RuntimeError(
                f"Unsupported classifier mode {mode!r}. This method uses only 'geometry_only'. "
                "Remove calibrators/adaptive-boundary modes from the trainer config."
            )
        return out

    def expand(self, num_new_classes: int, phase: int = 0) -> None:
        del phase
        self.num_classes += int(max(0, num_new_classes))

    def expand_to_seen_classes(self, seen_classes: Iterable[int]) -> None:
        seen = _as_seen_list(seen_classes)
        self._last_seen_classes = seen
        self.num_classes = len(seen)

    def freeze_all_adaptation(self) -> None:
        return

    def unfreeze_all_adaptation(self) -> None:
        return

    def freeze_old_adaptation(self, old_class_count: int) -> None:
        del old_class_count
        return

    def freeze_fusion_module(self) -> None:
        return

    def unfreeze_fusion_module(self) -> None:
        return

    def adaptation_regularization_loss(self, num_classes: Optional[int] = None) -> Dict[str, torch.Tensor]:
        del num_classes
        z = self._zero * 0.0
        return {"total": z, "bias": z, "temp": z, "alpha": z, "energy_cal": z, "adaptive_boundary": z}

    def energy_calibration_regularization_loss(self, num_classes: Optional[int] = None) -> torch.Tensor:
        return self.adaptation_regularization_loss(num_classes=num_classes)["energy_cal"]

    def enable_energy_calibration(self, enabled: bool = True, calibrator_type: Optional[str] = None) -> None:
        del calibrator_type
        if _to_bool(enabled, False):
            raise RuntimeError("Energy calibration is removed. Use geometry replay + energy margins.")

    def boundary_parameters(self) -> Iterable[nn.Parameter]:
        return []

    def freeze_all_boundary_radii(self) -> None:
        return

    def unfreeze_all_boundary_radii(self) -> None:
        raise RuntimeError("Adaptive boundary radii are removed from the classifier.")

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
        raise TypeError("geometry_bank must be a dict or expose get_bank()/get_subspace_bank().")

    @staticmethod
    def _bank_variances(bank: Mapping[str, Any]) -> torch.Tensor:
        if "variances" in bank and torch.is_tensor(bank["variances"]):
            return bank["variances"]
        eig = _tensor_from_bank(bank, "eigvals")
        res = _tensor_from_bank(bank, "res_vars", "resvars")
        return torch.cat([eig, res.unsqueeze(-1)], dim=-1)

    def _bank_valid_mask(self, bank: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
        means = _tensor_from_bank(bank, "means").to(device=device)
        C = int(means.size(0))
        valid = torch.isfinite(means).all(dim=1)

        bases = _tensor_from_bank(bank, "bases", "raw_bases", "subspace_bases", required=False)
        if torch.is_tensor(bases) and bases.dim() == 3 and bases.size(0) == C:
            valid = valid & torch.isfinite(bases.to(device=device)).flatten(1).all(dim=1)

        try:
            variances = self._bank_variances(bank).to(device=device)
        except (KeyError, ValueError, RuntimeError):
            variances = None
        if torch.is_tensor(variances) and variances.dim() == 2 and variances.size(0) == C:
            var_finite = torch.isfinite(variances).all(dim=1)
            residual_positive = variances[:, -1] > 0
            valid = valid & var_finite & residual_positive

        if "valid_mask" in bank and torch.is_tensor(bank["valid_mask"]):
            vm = bank["valid_mask"].to(device=device).bool().flatten()
            if vm.numel() != C:
                raise ValueError(f"valid_mask width mismatch: {vm.numel()} vs bank rows {C}")
            valid = valid & vm
        if "sample_counts" in bank and torch.is_tensor(bank["sample_counts"]):
            counts = bank["sample_counts"].to(device=device).flatten()
            if counts.numel() != C:
                raise ValueError(f"sample_counts width mismatch: {counts.numel()} vs bank rows {C}")
            valid = valid & torch.isfinite(counts) & (counts > 0)
        if "active_ranks" in bank and torch.is_tensor(bank["active_ranks"]):
            ranks = bank["active_ranks"].to(device=device).long().flatten()
            if ranks.numel() != C:
                raise ValueError(f"active_ranks width mismatch: {ranks.numel()} vs bank rows {C}")
            max_rank = int(bases.size(2)) if torch.is_tensor(bases) and bases.dim() == 3 else int(ranks.max().item())
            valid = valid & (ranks >= 0) & (ranks <= max_rank)
        return valid

    def _sample_counts_or_valid(self, bank: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        means = _tensor_from_bank(bank, "means")
        C = int(means.size(0))
        if "sample_counts" in bank and torch.is_tensor(bank["sample_counts"]):
            counts = bank["sample_counts"].to(device=device, dtype=dtype).flatten()
            if counts.numel() == C:
                return counts
        valid = self._bank_valid_mask(bank, device=device).to(dtype=dtype)
        return valid

    def _infer_seen_from_bank(self, bank: Mapping[str, Any]) -> List[int]:
        means = _tensor_from_bank(bank, "means")
        C = int(means.size(0))
        device = means.device
        valid = self._bank_valid_mask(bank, device=device).detach().cpu().flatten()
        if "class_ids" in bank and torch.is_tensor(bank["class_ids"]):
            ids = bank["class_ids"].detach().cpu().long().flatten()
            if ids.numel() == C:
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
            if len(set(int(c) for c in bank_class_ids)) != len(bank_class_ids):
                raise RuntimeError("bank['class_ids'] contains duplicate global class ids.")
            mapping = {int(c): i for i, c in enumerate(bank_class_ids)}
            missing = [int(c) for c in seen_classes if int(c) not in mapping]
            if missing:
                raise IndexError(f"seen_classes absent from sliced GeometryBank class_ids: {missing}")
            return torch.as_tensor([mapping[int(c)] for c in seen_classes], device=device, dtype=torch.long)

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
        bases = _tensor_from_bank(bank, "bases", "raw_bases", "subspace_bases").to(device=device, dtype=dtype)
        variances = self._bank_variances(bank).to(device=device, dtype=dtype)
        sample_counts = self._sample_counts_or_valid(bank, device=device, dtype=dtype).flatten()
        valid_all = self._bank_valid_mask(bank, device=device)

        if means.dim() != 2 or means.size(1) != self.d_model:
            raise ValueError(f"bank means must be [C,{self.d_model}], got {tuple(means.shape)}")
        if bases.dim() != 3 or bases.size(1) != self.d_model:
            raise ValueError(f"bank bases must be [C,{self.d_model},R], got {tuple(bases.shape)}")
        if variances.dim() != 2 or variances.size(0) != means.size(0) or variances.size(1) != bases.size(2) + 1:
            raise ValueError(
                f"bank variances must be [C,R+1], got {tuple(variances.shape)} for bases {tuple(bases.shape)}"
            )
        if sample_counts.numel() != means.size(0):
            raise ValueError(f"sample_counts/valid width mismatch: {sample_counts.numel()} vs rows {means.size(0)}")

        row_idx = self._resolve_row_indices(bank, seen_classes, device=device)
        counts_seen = sample_counts.index_select(0, row_idx)
        valid_seen = valid_all.index_select(0, row_idx)
        missing = [int(seen_classes[i]) for i in range(len(seen_classes)) if not bool(valid_seen[i].item())]
        if missing:
            raise RuntimeError(f"Geometry scoring requested classes with no valid GeometryBank row: {missing}")

        reliability = None
        if "reliability" in bank and torch.is_tensor(bank["reliability"]):
            rel_all = bank["reliability"].to(device=device, dtype=dtype).flatten()
            if rel_all.numel() != means.size(0):
                raise ValueError(f"reliability width mismatch: {rel_all.numel()} vs rows {means.size(0)}")
            reliability = rel_all.index_select(0, row_idx)
        elif "feature_reliability" in bank and torch.is_tensor(bank["feature_reliability"]):
            rel_all = bank["feature_reliability"].to(device=device, dtype=dtype).flatten()
            if rel_all.numel() != means.size(0):
                raise ValueError(f"feature_reliability width mismatch: {rel_all.numel()} vs rows {means.size(0)}")
            reliability = rel_all.index_select(0, row_idx)

        active_ranks = None
        if "active_ranks" in bank and torch.is_tensor(bank["active_ranks"]):
            rank_all = bank["active_ranks"].to(device=device).long().flatten()
            if rank_all.numel() != means.size(0):
                raise ValueError(f"active_ranks width mismatch: {rank_all.numel()} vs rows {means.size(0)}")
            active_ranks = rank_all.index_select(0, row_idx)

        return {
            "means": means.index_select(0, row_idx),
            "bases": bases.index_select(0, row_idx),
            "eigvals": variances.index_select(0, row_idx)[:, :-1].clamp_min(self.variance_floor),
            "res_vars": variances.index_select(0, row_idx)[:, -1].flatten().clamp_min(self.variance_floor),
            "sample_counts": counts_seen,
            "reliability": reliability,
            "active_ranks": active_ranks,
            "valid_class_mask": valid_seen,
            "global_class_ids": torch.as_tensor([int(c) for c in seen_classes], device=device, dtype=torch.long),
            "row_indices": row_idx,
        }

    # ------------------------------------------------------------------
    # Labels and masks
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
        device: Optional[torch.device] = None,
        require_nonempty: bool = False,
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

        old_mask = torch.tensor([c in old_set for c in seen], dtype=torch.bool, device=device)
        new_mask = torch.tensor([c in new_set for c in seen], dtype=torch.bool, device=device)
        if require_nonempty and (not bool(old_mask.any().item()) or not bool(new_mask.any().item())):
            raise RuntimeError(f"old/new masks must both be non-empty. seen={seen}, old={old_list}, new={new_list}")
        return old_mask, new_mask, int(old_mask.sum().item())

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
        self._old_new_masks(seen_classes, old_classes=old_classes, new_classes=new_classes, device=logits.device)

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

    @staticmethod
    def _center_vector_on_valid(vec: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if vec.numel() == 0 or valid_mask.numel() != vec.numel() or not bool(valid_mask.any().item()):
            return vec
        out = vec.clone()
        out[valid_mask] = out[valid_mask] - out[valid_mask].mean().detach()
        out[~valid_mask] = 0.0
        return out

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
        valid_class_mask = rows["valid_class_mask"].to(device=features.device)

        S, D, R = bases.shape
        rank_mask, ar = self._active_rank_mask(active_ranks, S, R, features.device, features.dtype)

        delta = features.unsqueeze(1) - means.unsqueeze(0)                    # [B,S,D]
        coeff = torch.einsum("bsd,sdr->bsr", delta, bases)                  # [B,S,R]
        coeff_active = coeff * rank_mask.view(1, S, R)
        recon = torch.einsum("bsr,sdr->bsd", coeff_active, bases)
        residual = delta - recon

        eig = eigvals.clamp_min(self.variance_floor)
        rv = res_vars.clamp_min(self.variance_floor)

        parallel_raw = ((coeff_active.pow(2) / eig.view(1, S, R)) * rank_mask.view(1, S, R)).sum(dim=-1)
        residual_raw = residual.pow(2).sum(dim=-1)
        active_dims = ar.to(dtype=features.dtype).clamp_min(1.0)
        residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=features.dtype).clamp_min(1.0)

        if self.normalize_energy_by_dim:
            parallel = parallel_raw / active_dims.view(1, S)
            orthogonal = residual_raw / (residual_dims.view(1, S) * rv.view(1, S))
        else:
            parallel = parallel_raw
            orthogonal = residual_raw / rv.view(1, S)
        energy = parallel + orthogonal

        # Diagnostic-only vectors retained for API compatibility. They are not
        # added to energy because that would diverge from GeometryBank replay
        # validation and would penalize small/low-reliability HSI classes.
        logdet_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)
        reliability_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)

        invalid_mask = ~valid_class_mask.view(1, S)
        if bool(invalid_mask.any().item()):
            energy = energy.masked_fill(invalid_mask, self.invalid_class_energy)
            parallel = parallel.masked_fill(invalid_mask, self.invalid_class_energy)
            orthogonal = orthogonal.masked_fill(invalid_mask, self.invalid_class_energy)

        energy = torch.nan_to_num(energy, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0)

        parts: Dict[str, torch.Tensor] = {
            "energy": energy,
            "feature_energy": energy,
            "parallel": torch.nan_to_num(parallel, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "orthogonal": torch.nan_to_num(orthogonal, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "parallel_energy": torch.nan_to_num(parallel, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "residual_energy": torch.nan_to_num(orthogonal, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "parallel_raw": torch.nan_to_num(parallel_raw, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "residual_raw": torch.nan_to_num(residual_raw, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
            "active_dims": active_dims,
            "residual_dims": residual_dims,
            "logdet_penalty": logdet_penalty,
            "reliability_penalty": reliability_penalty,
            "active_ranks": ar,
            "rank_mask": rank_mask,
            "sample_counts": rows["sample_counts"],
            "global_class_ids": rows["global_class_ids"],
            "row_indices": rows["row_indices"],
            "valid_class_mask": valid_class_mask,
        }
        return energy, parts if return_parts else {}

    def _energy_to_logits(self, energy: torch.Tensor, valid_class_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if energy.dim() != 2:
            raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
        finite_mask = torch.isfinite(energy) & (energy < 0.5 * self.invalid_class_energy)
        if valid_class_mask is not None and valid_class_mask.numel() == energy.size(1):
            finite_mask = finite_mask & valid_class_mask.to(device=energy.device).bool().view(1, -1)
        masked_energy = energy.masked_fill(~finite_mask, float("inf"))
        row_min = masked_energy.min(dim=1, keepdim=True).values
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        rel = energy - row_min
        logits = -self.logit_scale * rel
        if self.logit_clip > 0.0:
            logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
        logits = torch.nan_to_num(logits, nan=self.invalid_logit, posinf=self.invalid_logit, neginf=self.invalid_logit)
        logits = logits.masked_fill(~finite_mask, self.invalid_logit)
        return logits

    def compute_geometry_logits(
        self,
        features: torch.Tensor,
        *,
        seen_classes: Iterable[int],
        geometry_bank: Any,
        return_parts: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        seen = _as_seen_list(seen_classes)
        energy, parts = self.compute_geometry_energy(features, seen_classes=seen, geometry_bank=geometry_bank, return_parts=True)
        logits = self._energy_to_logits(energy, parts.get("valid_class_mask"))
        self.assert_logits_valid(logits, seen_classes=seen, context="compute_geometry_logits")
        if not return_parts:
            return logits
        out: Dict[str, torch.Tensor] = {"logits": logits, "energy": energy, "raw_energy": energy}
        out.update(parts)
        return out

    # ------------------------------------------------------------------
    # Compatibility tensor APIs
    # ------------------------------------------------------------------
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
        bank: Dict[str, torch.Tensor] = {"means": means, "bases": bases, "variances": variances}
        if sample_counts is not None:
            bank["sample_counts"] = sample_counts
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
        bank: Dict[str, torch.Tensor] = {"means": means, "bases": bases, "variances": variances}
        if sample_counts is not None:
            bank["sample_counts"] = sample_counts
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
        del old_class_count, old_classes, new_classes
        if apply_energy_calibration:
            raise RuntimeError("apply_energy_calibration=True is removed from the strict classifier.")
        if seen_classes is None:
            seen_classes = self._infer_seen_from_bank(bank)
        seen = _as_seen_list(seen_classes)
        out = self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank, return_parts=True)
        if return_parts:
            return out
        return out["logits"]

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
    # Diagnostics
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
        del old_classes, new_classes, old_class_count
        seen = _as_seen_list(seen_classes)
        self.assert_logits_valid(logits, seen_classes=seen, context="calibrate_old_new_logits")
        raise RuntimeError("Old/new logit calibration is removed from the strict classifier.")

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
        energy: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        seen = _as_seen_list(seen_classes)
        self.assert_logits_valid(logits, seen_classes=seen, targets=targets_local, context="classifier_diagnostics")
        old_mask, new_mask, old_prefix = self._old_new_masks(
            seen,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
            device=logits.device,
        )

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
            "uses_old_new_logit_calibration": False,
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

        if energy is not None and targets_local is not None:
            em = self.energy_margin_statistics(energy, targets_local)
            out.update({
                "energy_mean_margin": float(em["mean_margin"].detach().cpu().item()),
                "energy_min_margin": float(em["min_margin"].detach().cpu().item()),
                "energy_violation_rate": float(em["violation_rate"].detach().cpu().item()),
                "energy_accuracy": float(em["accuracy"].detach().cpu().item()),
            })
            on = self.old_new_energy_statistics(
                energy,
                targets_local,
                old_class_count=old_prefix,
                old_mask=old_mask,
                new_mask=new_mask,
            )
            out.update({f"old_new_{k}": float(v.detach().cpu().item()) for k, v in on.items()})
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
        wrong = energy.masked_fill(true_mask, float("inf"))
        nearest_wrong = wrong.min(dim=1).values
        valid = torch.isfinite(nearest_wrong)
        margin = nearest_wrong[valid] - true_e[valid]
        pred = energy.argmin(dim=1)
        z = energy.sum() * 0.0
        return {
            "mean_margin": margin.mean() if margin.numel() else z,
            "min_margin": margin.min() if margin.numel() else z,
            "violation_rate": (margin <= 0).float().mean() if margin.numel() else z,
            "accuracy": (pred == y).float().mean() if y.numel() else z,
        }

    @torch.no_grad()
    def old_new_energy_statistics(
        self,
        energy: torch.Tensor,
        labels: torch.Tensor,
        *,
        old_class_count: Optional[int] = None,
        old_mask: Optional[torch.Tensor] = None,
        new_mask: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del sample_counts
        z = energy.sum() * 0.0 if torch.is_tensor(energy) else self._zero * 0.0
        if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
            return {"new_into_old_rate": z, "old_into_new_rate": z, "old_group_win_rate": z, "new_group_win_rate": z, "mean_old_new_gap": z}
        C = int(energy.size(1))
        if old_mask is None or new_mask is None:
            old = int(max(0, min(int(old_class_count or 0), C)))
            old_mask = torch.arange(C, device=energy.device) < old
            new_mask = ~old_mask
        else:
            old_mask = old_mask.to(device=energy.device).bool().flatten()
            new_mask = new_mask.to(device=energy.device).bool().flatten()
        if old_mask.numel() != C or new_mask.numel() != C or not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
            return {"new_into_old_rate": z, "old_into_new_rate": z, "old_group_win_rate": z, "new_group_win_rate": z, "mean_old_new_gap": z}
        y = _as_long_1d(labels, device=energy.device, name="labels")
        old_min = energy[:, old_mask].min(dim=1).values
        new_min = energy[:, new_mask].min(dim=1).values
        old_win = old_min < new_min
        new_win = new_min < old_min
        old_labels = old_mask.index_select(0, y)
        new_labels = new_mask.index_select(0, y)
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
        old_class_count: Optional[int] = None,
        old_mask: Optional[torch.Tensor] = None,
        new_mask: Optional[torch.Tensor] = None,
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
        if old_mask is None or new_mask is None:
            old = int(max(0, min(int(old_class_count or 0), C)))
            old_mask = torch.arange(C, device=energy.device) < old
            new_mask = ~old_mask
        else:
            old_mask = old_mask.to(device=energy.device).bool().flatten()
            new_mask = new_mask.to(device=energy.device).bool().flatten()
        pred = energy.argmin(dim=1)
        acc = (pred == y).float().mean() if y.numel() else z
        if old_mask.numel() != C or new_mask.numel() != C or not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
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
        old_labels = old_mask.index_select(0, y)
        new_labels = new_mask.index_select(0, y)
        old_min = energy[:, old_mask].min(dim=1).values
        new_min = energy[:, new_mask].min(dim=1).values
        old_win = old_min < new_min
        new_win = new_min < old_min
        true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
        new_margin = old_min[new_labels] - true_e[new_labels] if bool(new_labels.any().item()) else energy.new_empty((0,))
        old_margin = new_min[old_labels] - true_e[old_labels] if bool(old_labels.any().item()) else energy.new_empty((0,))
        old_acc = (pred[old_labels] == y[old_labels]).float().mean() if bool(old_labels.any().item()) else z
        new_acc = (pred[new_labels] == y[new_labels]).float().mean() if bool(new_labels.any().item()) else z
        hm = (2 * old_acc * new_acc / (old_acc + new_acc + 1e-8)) if bool(old_labels.any().item()) and bool(new_labels.any().item()) else z
        both = torch.cat([new_margin, old_margin]) if (new_margin.numel() + old_margin.numel()) else energy.new_empty((0,))
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
            "mean_true_vs_opposite_margin": both.mean() if both.numel() else z,
        }

    @torch.no_grad()
    def old_geometry_risk_features_from_bank(
        self,
        features: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        old_class_count: Optional[int] = None,
        old_classes: Optional[Iterable[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        explicit_old = _ordered_unique_ints(old_classes or [])
        if not explicit_old and int(old_class_count or 0) <= 0:
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
        old_seen = explicit_old if explicit_old else full_seen[: int(old_class_count or 0)]
        missing_old = [c for c in old_seen if c not in set(full_seen)]
        if missing_old:
            raise RuntimeError(f"old_classes absent from GeometryBank: {missing_old}")
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
        row_idx = parts["row_indices"].index_select(0, nearest_local)
        if "reliability" in bank_dict and torch.is_tensor(bank_dict["reliability"]):
            rel_all = bank_dict["reliability"].to(device=features.device, dtype=features.dtype)
            rel = rel_all.index_select(0, row_idx).clamp(0.0, 1.0)
        if "res_vars" in bank_dict and torch.is_tensor(bank_dict["res_vars"]):
            rv_all = bank_dict["res_vars"].to(device=features.device, dtype=features.dtype)
            res = rv_all.index_select(0, row_idx).clamp_min(0.0)
        risk_features = torch.stack(
            [torch.log1p(nearest.clamp_min(0.0)), torch.log1p(margin.clamp_min(0.0)), rel, torch.log1p(res)], dim=1
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
        old_mask, new_mask, old_prefix = self._old_new_masks(
            seen,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
            device=features.device,
        )
        return self.old_new_margin_report_from_energy(
            out["energy"],
            targets_local,
            old_class_count=old_prefix,
            old_mask=old_mask,
            new_mask=new_mask,
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
            "method_path": "spectral_coupled_tangent_geometry_replay_hsi_necil",
            "architecture": "Spectral-Coupled Tangent Geometry Replay with New-Row Descriptor Adaptation",
            "output_contract": "[B, len(seen_classes)]",
            "uses_geometry_bank": True,
            "uses_feature_low_rank_energy": True,
            "uses_low_rank_logdet_energy": False,
            "uses_reliability_penalty": False,
            "energy_matches_geometry_bank_replay": True,
            "uses_rank_normalized_parallel_energy": bool(self.normalize_energy_by_dim),
            "uses_residual_dimension_normalization": bool(self.normalize_energy_by_dim),
            "uses_old_new_logit_calibration": False,
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
            "energy_calibrated": torch.tensor(False, device=features.device),
        }
        if return_energy or return_parts or return_diagnostics:
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
                energy=parts["energy"],
            )
        return out


# -----------------------------------------------------------------------------
# Standalone loss helpers used by trainers
# -----------------------------------------------------------------------------



def _labels_to_seen_local(
    labels: torch.Tensor,
    *,
    width: int,
    seen_classes: Optional[Iterable[int]] = None,
    targets_are_global: bool = False,
    context: str,
) -> torch.Tensor:
    y = labels.long().flatten()
    if targets_are_global:
        if seen_classes is None:
            raise RuntimeError(f"{context}: seen_classes is required for global labels.")
        y = GeometryEnergyClassifier.global_to_local_labels(y, _as_seen_list(seen_classes))
    if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= int(width)):
        raise RuntimeError(f"{context}: labels outside local range [0,{int(width)-1}].")
    return y


def _valid_energy_columns(
    energy: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    valid = torch.isfinite(energy).all(dim=0)
    if valid_mask is not None:
        vm = valid_mask.to(device=energy.device).bool().flatten()
        if vm.numel() != energy.size(1):
            raise RuntimeError(f"valid_mask width {vm.numel()} != energy width {energy.size(1)}")
        valid = valid & vm
    return valid


def _reduce_per_sample_by_class(
    values: torch.Tensor,
    labels_local: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    reduction = str(reduction).lower().strip()
    finite = torch.isfinite(values)
    if reduction == "none":
        return torch.where(finite, values, torch.zeros_like(values))
    if not bool(finite.any().item()):
        return values.sum() * 0.0
    if reduction == "sum":
        return values[finite].sum()
    if reduction in {"class_mean", "balanced", "class_balanced"}:
        terms: List[torch.Tensor] = []
        for c in torch.unique(labels_local[finite], sorted=True):
            mask = finite & labels_local.eq(c)
            if bool(mask.any().item()):
                terms.append(values[mask].mean())
        return torch.stack(terms).mean() if terms else values.sum() * 0.0
    return values[finite].mean()


def geometry_energy_margin_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    seen_classes: Optional[Iterable[int]] = None,
    targets_are_global: bool = False,
    reduction: str = "class_mean",
) -> torch.Tensor:
    """Correct-class energy must beat the nearest valid rival by ``margin``."""
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return _zero_like(labels if torch.is_tensor(labels) else None)
    if energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
    y = _labels_to_seen_local(
        labels.to(device=energy.device),
        width=int(energy.size(1)),
        seen_classes=seen_classes,
        targets_are_global=bool(targets_are_global),
        context="geometry_energy_margin_loss",
    )
    if y.numel() != energy.size(0):
        raise RuntimeError("labels/energy batch mismatch")
    valid_cols = _valid_energy_columns(energy, valid_mask)
    if not bool(valid_cols.index_select(0, y).all().item()):
        raise RuntimeError("At least one target class is invalid in valid_mask.")
    true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
    rival_mask = valid_cols.view(1, -1).expand_as(energy).clone()
    rival_mask.scatter_(1, y.view(-1, 1), False)
    nearest_wrong = energy.masked_fill(~rival_mask, float("inf")).min(dim=1).values
    loss = F.relu(true_e + float(margin) - nearest_wrong)
    return _reduce_per_sample_by_class(loss, y, reduction=reduction)


def old_new_invasion_loss(
    energy: torch.Tensor,
    labels: torch.Tensor,
    old_class_count: Optional[int] = None,
    margin: float = 0.25,
    valid_mask: Optional[torch.Tensor] = None,
    old_mask: Optional[torch.Tensor] = None,
    new_mask: Optional[torch.Tensor] = None,
    *,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    targets_are_global: bool = False,
    reduction: str = "class_mean",
) -> torch.Tensor:
    """Bidirectional old/new invasion loss with explicit global class lists."""
    if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
        return _zero_like(labels if torch.is_tensor(labels) else None)
    if energy.dim() != 2:
        raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
    C = int(energy.size(1))
    seen = _as_seen_list(seen_classes, fallback_count=C)
    if len(seen) != C:
        raise RuntimeError(f"seen_classes width {len(seen)} != energy width {C}")
    if old_mask is None or new_mask is None:
        helper = GeometryEnergyClassifier(d_model=1)
        old_mask, new_mask, _ = helper._old_new_masks(
            seen,
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=old_class_count,
            device=energy.device,
            require_nonempty=True,
        )
    else:
        old_mask = old_mask.to(device=energy.device).bool().flatten()
        new_mask = new_mask.to(device=energy.device).bool().flatten()
    if old_mask.numel() != C or new_mask.numel() != C:
        raise RuntimeError("old_mask/new_mask must match energy width")
    if bool((old_mask & new_mask).any().item()) or not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
        raise RuntimeError("old/new masks must be disjoint and non-empty.")
    valid_cols = _valid_energy_columns(energy, valid_mask)
    old_valid = old_mask & valid_cols
    new_valid = new_mask & valid_cols
    if not bool(old_valid.any().item()) or not bool(new_valid.any().item()):
        return energy.sum() * 0.0
    y = _labels_to_seen_local(
        labels.to(device=energy.device),
        width=C,
        seen_classes=seen,
        targets_are_global=bool(targets_are_global),
        context="old_new_invasion_loss",
    )
    if y.numel() != energy.size(0):
        raise RuntimeError("labels/energy batch mismatch")
    true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
    old_min = energy[:, old_valid].min(dim=1).values
    new_min = energy[:, new_valid].min(dim=1).values
    is_old = old_mask.index_select(0, y)
    is_new = new_mask.index_select(0, y)
    if not bool((is_old | is_new).all().item()):
        raise RuntimeError("Some labels are not covered by old/new class masks.")
    opposite = torch.where(is_old, new_min, old_min)
    loss = F.relu(true_e + float(margin) - opposite)
    return _reduce_per_sample_by_class(loss, y, reduction=reduction)















# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# _EPS = 1e-12
# _INVALID_LOGIT = -1e9


# # -----------------------------------------------------------------------------
# # Utilities
# # -----------------------------------------------------------------------------


# def _to_bool(value: object, default: bool = False) -> bool:
#     if value is None:
#         return bool(default)
#     if isinstance(value, bool):
#         return value
#     if isinstance(value, (int, float)):
#         return bool(value)
#     if isinstance(value, str):
#         v = value.strip().lower()
#         if v in {"1", "true", "yes", "y", "on"}:
#             return True
#         if v in {"0", "false", "no", "n", "off", "none", "null", ""}:
#             return False
#         raise ValueError(f"Cannot parse boolean value: {value!r}")
#     return bool(value)


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     out: List[int] = []
#     seen = set()
#     for value in values:
#         c = int(value)
#         if c not in seen:
#             out.append(c)
#             seen.add(c)
#     return out


# def _as_seen_list(
#     seen_classes: Optional[Iterable[int]],
#     *,
#     fallback_count: Optional[int] = None,
# ) -> List[int]:
#     if seen_classes is None:
#         if fallback_count is None:
#             raise ValueError(
#                 "seen_classes is required. The classifier must know the current "
#                 "seen-class order so logits are [B, len(seen_classes)]."
#             )
#         seen = list(range(int(fallback_count)))
#     else:
#         seen = [int(c) for c in seen_classes]

#     if not seen:
#         raise ValueError("seen_classes is empty.")
#     if len(set(seen)) != len(seen):
#         raise ValueError(f"seen_classes contains duplicates: {seen}")
#     if min(seen) < 0:
#         raise ValueError(f"seen_classes contains negative ids: {seen}")
#     return seen


# def _as_long_1d(x: torch.Tensor, *, device: torch.device, name: str) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     return x.to(device=device).long().flatten()


# def _finite_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     if x.numel() == 0:
#         raise ValueError(f"{name} is empty.")
#     if not torch.isfinite(x).all():
#         bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
#         raise RuntimeError(f"{name} contains {bad} NaN/Inf values.")
#     return x


# def _zero_like(ref: Optional[torch.Tensor] = None, *, device: Optional[torch.device] = None) -> torch.Tensor:
#     if torch.is_tensor(ref):
#         return ref.sum() * 0.0
#     return torch.tensor(0.0, device=device if device is not None else torch.device("cpu"))


# def _tensor_from_bank(bank: Mapping[str, Any], *names: str, required: bool = True) -> Optional[torch.Tensor]:
#     for name in names:
#         value = bank.get(name, None)
#         if torch.is_tensor(value):
#             return value
#     if required:
#         raise KeyError(f"GeometryBank is missing required tensor. Tried keys={names}")
#     return None


# def _truthy_legacy(value: Any) -> bool:
#     if value is None:
#         return False
#     if isinstance(value, str):
#         return _to_bool(value, False)
#     if isinstance(value, (int, float)):
#         return float(value) != 0.0
#     if torch.is_tensor(value):
#         return bool(value.numel() > 0 and float(value.detach().abs().max().cpu().item()) != 0.0)
#     return bool(value)


# # -----------------------------------------------------------------------------
# # Strict low-rank geometry classifier
# # -----------------------------------------------------------------------------


# class GeometryEnergyClassifier(nn.Module):
#     """Strict seen-class low-rank GeometryBank classifier for HSI NECIL.

#     Method contract:
#         * Input features are canonical projected z-space features [B, D].
#         * GeometryBank stores compact low-rank descriptors only.
#         * Output logits are [B, len(seen_classes)] in exactly seen_classes order.
#         * CE targets for returned logits must be seen-local labels.
#         * No prototype classifier.
#         * No direct spectral/band inference branch.
#         * No old/new score calibrator.
#         * No adaptive boundary branch.

#     Energy:
#         E = parallel low-rank Mahalanobis
#           + residual Mahalanobis
#           + centered logdet penalty
#           + centered reliability penalty.
#     """

#     _DISALLOWED_TRUTHY_ARGS = {
#         "use_old_new_calibration",
#         "use_energy_calibrator",
#         "use_measured_energy_calibration",
#         "use_adaptive_boundary",
#         "use_spectral_geometry",
#         "use_spectral_residual_energy",
#     }
#     _DISALLOWED_NONZERO_ARGS = {
#         "spectral_energy_weight",
#         "band_energy_weight",
#         "energy_calibrator_weight",
#         "adapter_energy_weight",
#     }

#     def __init__(
#         self,
#         initial_classes: int = 0,
#         d_model: int = 128,
#         logit_scale: float = 8.0,
#         variance_floor: float = 1e-4,
#         residual_variance_scale: float = 0.75,
#         normalize_energy_by_dim: bool = True,
#         energy_normalize_by_dim: Optional[bool] = None,
#         use_logdet_energy: bool = True,
#         logdet_energy_weight: float = 0.05,
#         logdet_normalize_by_dim: bool = True,
#         center_logdet_energy: bool = True,
#         use_reliability_penalty: bool = True,
#         reliability_energy_weight: float = 0.03,
#         reliability_min_clamp: float = 0.05,
#         center_reliability_energy: bool = True,
#         logit_clip: float = 0.0,
#         invalid_logit: float = _INVALID_LOGIT,
#         invalid_class_energy: float = 1e6,
#         # Legacy arguments are accepted only when they are explicitly off.
#         energy_calibrator_type: str = "none",
#         calibration_max_abs_bias: float = 1.0,
#         energy_calibrator_max_bias: Optional[float] = None,
#         **legacy_kwargs: Any,
#     ) -> None:
#         super().__init__()
#         del calibration_max_abs_bias, energy_calibrator_max_bias

#         ct = str(energy_calibrator_type or "none").strip().lower()
#         if ct not in {"", "none", "off", "false"}:
#             raise RuntimeError(
#                 "Classifier calibrators are removed from the main method. "
#                 "Use low-rank geometry replay and residual geometry adaptation losses instead."
#             )
#         for name in sorted(self._DISALLOWED_TRUTHY_ARGS):
#             if name in legacy_kwargs and _truthy_legacy(legacy_kwargs[name]):
#                 raise RuntimeError(
#                     f"{name} is disabled. The classifier has one strict path: "
#                     "seen-class low-rank geometry energy."
#                 )
#         for name in sorted(self._DISALLOWED_NONZERO_ARGS):
#             if name in legacy_kwargs and _truthy_legacy(legacy_kwargs[name]):
#                 raise RuntimeError(
#                     f"{name} is disabled. Spectral/band signals belong in base loss, "
#                     "GeometryBank risk reports, and replay scheduling, not classifier logits."
#                 )

#         if energy_normalize_by_dim is not None:
#             normalize_energy_by_dim = _to_bool(energy_normalize_by_dim, True)

#         self.num_classes = int(max(0, initial_classes))
#         self.d_model = int(d_model)
#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive.")
#         self.logit_scale = float(logit_scale)
#         if self.logit_scale <= 0.0:
#             raise ValueError("logit_scale must be positive.")
#         self.variance_floor = float(max(variance_floor, 1e-12))
#         self.residual_variance_scale = float(max(residual_variance_scale, 1e-8))
#         self.normalize_energy_by_dim = _to_bool(normalize_energy_by_dim, True)
#         self.use_logdet_energy = _to_bool(use_logdet_energy, True)
#         self.logdet_energy_weight = float(max(logdet_energy_weight, 0.0))
#         self.logdet_normalize_by_dim = _to_bool(logdet_normalize_by_dim, True)
#         self.center_logdet_energy = _to_bool(center_logdet_energy, True)
#         self.use_reliability_penalty = _to_bool(use_reliability_penalty, True)
#         self.reliability_energy_weight = float(max(reliability_energy_weight, 0.0))
#         self.reliability_min_clamp = float(max(min(reliability_min_clamp, 1.0), 1e-8))
#         self.center_reliability_energy = _to_bool(center_reliability_energy, True)
#         self.logit_clip = float(max(logit_clip, 0.0))
#         self.invalid_logit = float(invalid_logit)
#         self.invalid_class_energy = float(max(invalid_class_energy, 1.0))

#         self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
#         self._last_seen_classes: List[int] = list(range(self.num_classes))

#     # ------------------------------------------------------------------
#     # Compatibility controls: strict no-op or hard error
#     # ------------------------------------------------------------------
#     @staticmethod
#     def normalize_mode(mode: str) -> str:
#         m = str(mode or "geometry_only").lower().strip()
#         aliases = {
#             "geo": "geometry_only",
#             "geometry": "geometry_only",
#             "geometry-only": "geometry_only",
#             "feature_geometry": "geometry_only",
#             "feature-only": "geometry_only",
#             "feature_only": "geometry_only",
#             "low_rank_geometry": "geometry_only",
#             "low-rank-geometry": "geometry_only",
#             "geometry_replay": "geometry_only",
#         }
#         out = aliases.get(m, m)
#         if out != "geometry_only":
#             raise RuntimeError(
#                 f"Unsupported classifier mode {mode!r}. This method uses only 'geometry_only'. "
#                 "Remove calibrators/adaptive-boundary modes from the trainer config."
#             )
#         return out

#     def expand(self, num_new_classes: int, phase: int = 0) -> None:
#         del phase
#         self.num_classes += int(max(0, num_new_classes))

#     def expand_to_seen_classes(self, seen_classes: Iterable[int]) -> None:
#         seen = _as_seen_list(seen_classes)
#         self._last_seen_classes = seen
#         self.num_classes = len(seen)

#     def freeze_all_adaptation(self) -> None:
#         return

#     def unfreeze_all_adaptation(self) -> None:
#         return

#     def freeze_old_adaptation(self, old_class_count: int) -> None:
#         del old_class_count
#         return

#     def freeze_fusion_module(self) -> None:
#         return

#     def unfreeze_fusion_module(self) -> None:
#         return

#     def adaptation_regularization_loss(self, num_classes: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         del num_classes
#         z = self._zero * 0.0
#         return {"total": z, "bias": z, "temp": z, "alpha": z, "energy_cal": z, "adaptive_boundary": z}

#     def energy_calibration_regularization_loss(self, num_classes: Optional[int] = None) -> torch.Tensor:
#         return self.adaptation_regularization_loss(num_classes=num_classes)["energy_cal"]

#     def enable_energy_calibration(self, enabled: bool = True, calibrator_type: Optional[str] = None) -> None:
#         del calibrator_type
#         if _to_bool(enabled, False):
#             raise RuntimeError("Energy calibration is removed. Use geometry replay + energy margins.")

#     def boundary_parameters(self) -> Iterable[nn.Parameter]:
#         return []

#     def freeze_all_boundary_radii(self) -> None:
#         return

#     def unfreeze_all_boundary_radii(self) -> None:
#         raise RuntimeError("Adaptive boundary radii are removed from the classifier.")

#     def freeze_old_boundary_radii(self, old_class_count: int) -> None:
#         del old_class_count
#         return

#     def adaptive_boundary_state(self, num_classes: Optional[int] = None, old_class_count: int = 0) -> Dict[str, float]:
#         del num_classes, old_class_count
#         return {
#             "adaptive_boundary_enabled": 0.0,
#             "boundary_radius_mean": 0.0,
#             "boundary_radius_min": 0.0,
#             "boundary_radius_max": 0.0,
#             "old_boundary_radius_mean": 0.0,
#             "new_boundary_radius_mean": 0.0,
#         }

#     def adaptive_boundary_loss(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         del args, kwargs
#         z = self._zero * 0.0
#         return {"total": z, "boundary": z.detach(), "old_new": z.detach(), "radius_reg": z.detach()}

#     # ------------------------------------------------------------------
#     # Bank handling
#     # ------------------------------------------------------------------
#     def _bank_dict(self, geometry_bank: Any) -> Dict[str, torch.Tensor]:
#         if geometry_bank is None:
#             raise ValueError("geometry_bank is required for geometry scoring.")
#         if isinstance(geometry_bank, dict):
#             return dict(geometry_bank)
#         if hasattr(geometry_bank, "get_bank") and callable(geometry_bank.get_bank):
#             return geometry_bank.get_bank()
#         if hasattr(geometry_bank, "get_subspace_bank") and callable(geometry_bank.get_subspace_bank):
#             return geometry_bank.get_subspace_bank()
#         raise TypeError("geometry_bank must be a dict or expose get_bank()/get_subspace_bank().")

#     @staticmethod
#     def _bank_variances(bank: Mapping[str, Any]) -> torch.Tensor:
#         if "variances" in bank and torch.is_tensor(bank["variances"]):
#             return bank["variances"]
#         eig = _tensor_from_bank(bank, "eigvals")
#         res = _tensor_from_bank(bank, "res_vars", "resvars")
#         return torch.cat([eig, res.unsqueeze(-1)], dim=-1)

#     def _bank_valid_mask(self, bank: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
#         means = _tensor_from_bank(bank, "means").to(device=device)
#         C = int(means.size(0))
#         finite = torch.isfinite(means).all(dim=1)
#         if "valid_mask" in bank and torch.is_tensor(bank["valid_mask"]):
#             vm = bank["valid_mask"].to(device=device).bool().flatten()
#             if vm.numel() == C:
#                 finite = finite & vm
#         if "sample_counts" in bank and torch.is_tensor(bank["sample_counts"]):
#             counts = bank["sample_counts"].to(device=device).flatten()
#             if counts.numel() == C:
#                 finite = finite & torch.isfinite(counts) & (counts > 0)
#         return finite

#     def _sample_counts_or_valid(self, bank: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
#         means = _tensor_from_bank(bank, "means")
#         C = int(means.size(0))
#         if "sample_counts" in bank and torch.is_tensor(bank["sample_counts"]):
#             counts = bank["sample_counts"].to(device=device, dtype=dtype).flatten()
#             if counts.numel() == C:
#                 return counts
#         valid = self._bank_valid_mask(bank, device=device).to(dtype=dtype)
#         return valid

#     def _infer_seen_from_bank(self, bank: Mapping[str, Any]) -> List[int]:
#         means = _tensor_from_bank(bank, "means")
#         C = int(means.size(0))
#         device = means.device
#         valid = self._bank_valid_mask(bank, device=device).detach().cpu().flatten()
#         if "class_ids" in bank and torch.is_tensor(bank["class_ids"]):
#             ids = bank["class_ids"].detach().cpu().long().flatten()
#             if ids.numel() == C:
#                 return [int(ids[i].item()) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
#         return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]

#     def _resolve_row_indices(
#         self,
#         bank: Mapping[str, Any],
#         seen_classes: Sequence[int],
#         *,
#         device: torch.device,
#     ) -> torch.Tensor:
#         means = _tensor_from_bank(bank, "means")
#         C = int(means.size(0))
#         if "class_ids" in bank and torch.is_tensor(bank["class_ids"]):
#             bank_class_ids = bank["class_ids"].detach().cpu().long().flatten().tolist()
#             if len(bank_class_ids) != C:
#                 raise RuntimeError(f"bank['class_ids'] length {len(bank_class_ids)} does not match rows {C}.")
#             mapping = {int(c): i for i, c in enumerate(bank_class_ids)}
#             missing = [int(c) for c in seen_classes if int(c) not in mapping]
#             if missing:
#                 raise IndexError(f"seen_classes absent from sliced GeometryBank class_ids: {missing}")
#             return torch.as_tensor([mapping[int(c)] for c in seen_classes], device=device, dtype=torch.long)

#         missing = [int(c) for c in seen_classes if int(c) < 0 or int(c) >= C]
#         if missing:
#             raise IndexError(f"seen_classes contain ids absent from full GeometryBank: {missing}; bank_rows={C}")
#         return torch.as_tensor([int(c) for c in seen_classes], device=device, dtype=torch.long)

#     def _select_bank_rows(
#         self,
#         bank: Mapping[str, Any],
#         seen_classes: Sequence[int],
#         *,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> Dict[str, torch.Tensor]:
#         means = _tensor_from_bank(bank, "means").to(device=device, dtype=dtype)
#         bases = _tensor_from_bank(bank, "bases", "raw_bases", "subspace_bases").to(device=device, dtype=dtype)
#         variances = self._bank_variances(bank).to(device=device, dtype=dtype)
#         sample_counts = self._sample_counts_or_valid(bank, device=device, dtype=dtype).flatten()
#         valid_all = self._bank_valid_mask(bank, device=device)

#         if means.dim() != 2 or means.size(1) != self.d_model:
#             raise ValueError(f"bank means must be [C,{self.d_model}], got {tuple(means.shape)}")
#         if bases.dim() != 3 or bases.size(1) != self.d_model:
#             raise ValueError(f"bank bases must be [C,{self.d_model},R], got {tuple(bases.shape)}")
#         if variances.dim() != 2 or variances.size(0) != means.size(0) or variances.size(1) != bases.size(2) + 1:
#             raise ValueError(
#                 f"bank variances must be [C,R+1], got {tuple(variances.shape)} for bases {tuple(bases.shape)}"
#             )
#         if sample_counts.numel() != means.size(0):
#             raise ValueError(f"sample_counts/valid width mismatch: {sample_counts.numel()} vs rows {means.size(0)}")

#         row_idx = self._resolve_row_indices(bank, seen_classes, device=device)
#         counts_seen = sample_counts.index_select(0, row_idx)
#         valid_seen = valid_all.index_select(0, row_idx)
#         missing = [int(seen_classes[i]) for i in range(len(seen_classes)) if not bool(valid_seen[i].item())]
#         if missing:
#             raise RuntimeError(f"Geometry scoring requested classes with no valid GeometryBank row: {missing}")

#         reliability = None
#         if "reliability" in bank and torch.is_tensor(bank["reliability"]):
#             reliability = bank["reliability"].to(device=device, dtype=dtype).index_select(0, row_idx)
#         elif "feature_reliability" in bank and torch.is_tensor(bank["feature_reliability"]):
#             reliability = bank["feature_reliability"].to(device=device, dtype=dtype).index_select(0, row_idx)

#         active_ranks = None
#         if "active_ranks" in bank and torch.is_tensor(bank["active_ranks"]):
#             active_ranks = bank["active_ranks"].to(device=device).long().index_select(0, row_idx)

#         return {
#             "means": means.index_select(0, row_idx),
#             "bases": bases.index_select(0, row_idx),
#             "eigvals": variances.index_select(0, row_idx)[:, :-1].clamp_min(self.variance_floor),
#             "res_vars": variances.index_select(0, row_idx)[:, -1].flatten().clamp_min(self.variance_floor),
#             "sample_counts": counts_seen,
#             "reliability": reliability,
#             "active_ranks": active_ranks,
#             "valid_class_mask": valid_seen,
#             "global_class_ids": torch.as_tensor([int(c) for c in seen_classes], device=device, dtype=torch.long),
#             "row_indices": row_idx,
#         }

#     # ------------------------------------------------------------------
#     # Labels and masks
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
#         for i, value in enumerate(y.detach().cpu().tolist()):
#             v = int(value)
#             if v not in mapping:
#                 bad.append(v)
#                 out[i] = -1
#             else:
#                 out[i] = mapping[v]
#         if bad:
#             raise RuntimeError(f"labels contain classes not in seen_classes. bad={sorted(set(bad))}, seen={seen}")
#         return out.to(device=labels.device)

#     def _old_new_masks(
#         self,
#         seen_classes: Sequence[int],
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         *,
#         device: Optional[torch.device] = None,
#         require_nonempty: bool = False,
#     ) -> Tuple[torch.Tensor, torch.Tensor, int]:
#         seen = [int(c) for c in seen_classes]
#         seen_set = set(seen)

#         if old_classes is None and old_class_count is not None:
#             k = int(max(0, min(int(old_class_count), len(seen))))
#             old_list = seen[:k]
#         else:
#             old_list = _ordered_unique_ints(old_classes or [])

#         if new_classes is None:
#             new_list = [c for c in seen if c not in set(old_list)]
#         else:
#             new_list = _ordered_unique_ints(new_classes)

#         old_set = set(old_list)
#         new_set = set(new_list)
#         if not old_set.issubset(seen_set):
#             raise RuntimeError(f"old_classes not subset of seen_classes: old={sorted(old_set)}, seen={seen}")
#         if not new_set.issubset(seen_set):
#             raise RuntimeError(f"new_classes not subset of seen_classes: new={sorted(new_set)}, seen={seen}")
#         if old_set & new_set:
#             raise RuntimeError(f"old/new classes overlap: {sorted(old_set & new_set)}")

#         old_mask = torch.tensor([c in old_set for c in seen], dtype=torch.bool, device=device)
#         new_mask = torch.tensor([c in new_set for c in seen], dtype=torch.bool, device=device)
#         if require_nonempty and (not bool(old_mask.any().item()) or not bool(new_mask.any().item())):
#             raise RuntimeError(f"old/new masks must both be non-empty. seen={seen}, old={old_list}, new={new_list}")
#         return old_mask, new_mask, int(old_mask.sum().item())

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
#             raise RuntimeError(f"{context}: output width={logits.size(1)} but len(seen_classes)={S}")
#         if not torch.isfinite(logits).all():
#             bad = int((~torch.isfinite(logits)).sum().detach().cpu().item())
#             raise RuntimeError(f"{context}: logits contain {bad} NaN/Inf values.")
#         if targets is not None:
#             y = _as_long_1d(targets, device=logits.device, name=f"{context}.targets")
#             if y.numel() != int(logits.size(0)):
#                 raise RuntimeError(f"{context}: target/logit batch mismatch: {y.numel()} vs {logits.size(0)}")
#             if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= S):
#                 raise RuntimeError(
#                     f"{context}: local targets must be in [0,{S - 1}], got unique="
#                     f"{torch.unique(y).detach().cpu().tolist()}"
#                 )
#         self._old_new_masks(seen_classes, old_classes=old_classes, new_classes=new_classes, device=logits.device)

#     # ------------------------------------------------------------------
#     # Core geometry energy
#     # ------------------------------------------------------------------
#     def _active_rank_mask(
#         self,
#         active_ranks: Optional[torch.Tensor],
#         num_classes: int,
#         rank: int,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if active_ranks is None or not torch.is_tensor(active_ranks) or active_ranks.numel() != num_classes:
#             ar = torch.full((num_classes,), rank, device=device, dtype=torch.long)
#         else:
#             ar = active_ranks.to(device=device).long().flatten().clamp(min=0, max=rank)
#         mask = torch.arange(rank, device=device).view(1, rank) < ar.view(num_classes, 1)
#         return mask.to(dtype=dtype), ar

#     @staticmethod
#     def _center_vector_on_valid(vec: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
#         if vec.numel() == 0 or valid_mask.numel() != vec.numel() or not bool(valid_mask.any().item()):
#             return vec
#         out = vec.clone()
#         out[valid_mask] = out[valid_mask] - out[valid_mask].mean().detach()
#         out[~valid_mask] = 0.0
#         return out

#     def compute_geometry_energy(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#         return_parts: bool = True,
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         _finite_tensor(features, "features")
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
#         valid_class_mask = rows["valid_class_mask"].to(device=features.device)

#         S, D, R = bases.shape
#         rank_mask, ar = self._active_rank_mask(active_ranks, S, R, features.device, features.dtype)

#         delta = features.unsqueeze(1) - means.unsqueeze(0)                    # [B,S,D]
#         coeff = torch.einsum("bsd,sdr->bsr", delta, bases)                  # [B,S,R]
#         coeff_active = coeff * rank_mask.view(1, S, R)
#         recon = torch.einsum("bsr,sdr->bsd", coeff_active, bases)
#         residual = delta - recon

#         eig = eigvals.clamp_min(self.variance_floor)
#         rv = (res_vars * self.residual_variance_scale).clamp_min(self.variance_floor)

#         parallel = ((coeff_active.pow(2) / eig.view(1, S, R)) * rank_mask.view(1, S, R)).sum(dim=-1)
#         orthogonal = residual.pow(2).sum(dim=-1) / rv.view(1, S)
#         energy = parallel + orthogonal
#         if self.normalize_energy_by_dim:
#             energy = energy / float(max(D, 1))

#         logdet_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)
#         if self.use_logdet_energy and self.logdet_energy_weight > 0.0:
#             active_logdet = (eig.log() * rank_mask).sum(dim=1)
#             residual_dims = (D - ar.clamp(min=0, max=D)).to(dtype=features.dtype)
#             logdet_penalty = active_logdet + residual_dims * rv.log()
#             if self.logdet_normalize_by_dim:
#                 logdet_penalty = logdet_penalty / float(max(D, 1))
#             if self.center_logdet_energy:
#                 logdet_penalty = self._center_vector_on_valid(logdet_penalty, valid_class_mask)
#             energy = energy + self.logdet_energy_weight * logdet_penalty.view(1, S)

#         reliability_penalty = torch.zeros((S,), device=features.device, dtype=features.dtype)
#         if self.use_reliability_penalty and self.reliability_energy_weight > 0.0 and reliability is not None:
#             rel = torch.nan_to_num(
#                 reliability.to(device=features.device, dtype=features.dtype).flatten(),
#                 nan=self.reliability_min_clamp,
#                 posinf=1.0,
#                 neginf=self.reliability_min_clamp,
#             ).clamp(self.reliability_min_clamp, 1.0)
#             reliability_penalty = -rel.log()
#             if self.center_reliability_energy:
#                 reliability_penalty = self._center_vector_on_valid(reliability_penalty, valid_class_mask)
#             energy = energy + self.reliability_energy_weight * reliability_penalty.view(1, S)

#         invalid_mask = ~valid_class_mask.view(1, S)
#         if bool(invalid_mask.any().item()):
#             energy = energy.masked_fill(invalid_mask, self.invalid_class_energy)
#             parallel = parallel.masked_fill(invalid_mask, self.invalid_class_energy)
#             orthogonal = orthogonal.masked_fill(invalid_mask, self.invalid_class_energy)

#         energy = torch.nan_to_num(energy, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0)

#         parts: Dict[str, torch.Tensor] = {
#             "energy": energy,
#             "feature_energy": energy,
#             "parallel": torch.nan_to_num(parallel, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
#             "orthogonal": torch.nan_to_num(orthogonal, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
#             "parallel_energy": torch.nan_to_num(parallel, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
#             "residual_energy": torch.nan_to_num(orthogonal, nan=self.invalid_class_energy, posinf=self.invalid_class_energy, neginf=0.0),
#             "logdet_penalty": logdet_penalty,
#             "reliability_penalty": reliability_penalty,
#             "active_ranks": ar,
#             "rank_mask": rank_mask,
#             "sample_counts": rows["sample_counts"],
#             "global_class_ids": rows["global_class_ids"],
#             "row_indices": rows["row_indices"],
#             "valid_class_mask": valid_class_mask,
#         }
#         return energy, parts if return_parts else {}

#     def _energy_to_logits(self, energy: torch.Tensor, valid_class_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if energy.dim() != 2:
#             raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#         finite_mask = torch.isfinite(energy) & (energy < 0.5 * self.invalid_class_energy)
#         if valid_class_mask is not None and valid_class_mask.numel() == energy.size(1):
#             finite_mask = finite_mask & valid_class_mask.to(device=energy.device).bool().view(1, -1)
#         masked_energy = energy.masked_fill(~finite_mask, float("inf"))
#         row_min = masked_energy.min(dim=1, keepdim=True).values
#         row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
#         rel = energy - row_min
#         logits = -self.logit_scale * rel
#         if self.logit_clip > 0.0:
#             logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
#         logits = torch.nan_to_num(logits, nan=self.invalid_logit, posinf=self.invalid_logit, neginf=self.invalid_logit)
#         logits = logits.masked_fill(~finite_mask, self.invalid_logit)
#         return logits

#     def compute_geometry_logits(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#         return_parts: bool = False,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         seen = _as_seen_list(seen_classes)
#         energy, parts = self.compute_geometry_energy(features, seen_classes=seen, geometry_bank=geometry_bank, return_parts=True)
#         logits = self._energy_to_logits(energy, parts.get("valid_class_mask"))
#         self.assert_logits_valid(logits, seen_classes=seen, context="compute_geometry_logits")
#         if not return_parts:
#             return logits
#         out: Dict[str, torch.Tensor] = {"logits": logits, "energy": energy, "raw_energy": energy}
#         out.update(parts)
#         return out

#     # ------------------------------------------------------------------
#     # Compatibility tensor APIs
#     # ------------------------------------------------------------------
#     def geometry_energy(
#         self,
#         features: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         return_parts: bool = False,
#         **_: Any,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         bank: Dict[str, torch.Tensor] = {"means": means, "bases": bases, "variances": variances}
#         if sample_counts is not None:
#             bank["sample_counts"] = sample_counts
#         if reliability is not None:
#             bank["reliability"] = reliability
#         if active_ranks is not None:
#             bank["active_ranks"] = active_ranks
#         seen = self._infer_seen_from_bank(bank)
#         energy, parts = self.compute_geometry_energy(features, seen_classes=seen, geometry_bank=bank, return_parts=True)
#         if return_parts:
#             out = {"energy": energy}
#             out.update(parts)
#             return out
#         return energy

#     def geometry_logits(
#         self,
#         features: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         **_: Any,
#     ) -> torch.Tensor:
#         bank: Dict[str, torch.Tensor] = {"means": means, "bases": bases, "variances": variances}
#         if sample_counts is not None:
#             bank["sample_counts"] = sample_counts
#         if reliability is not None:
#             bank["reliability"] = reliability
#         if active_ranks is not None:
#             bank["active_ranks"] = active_ranks
#         seen = self._infer_seen_from_bank(bank)
#         return self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank)

#     def geometry_logits_from_bank(
#         self,
#         features: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         apply_energy_calibration: bool = False,
#         old_class_count: int = 0,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_parts: bool = False,
#         **_: Any,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         del old_class_count, old_classes, new_classes
#         if apply_energy_calibration:
#             raise RuntimeError("apply_energy_calibration=True is removed from the strict classifier.")
#         if seen_classes is None:
#             seen_classes = self._infer_seen_from_bank(bank)
#         seen = _as_seen_list(seen_classes)
#         out = self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank, return_parts=True)
#         if return_parts:
#             return out
#         return out["logits"]

#     def geometry_energy_from_bank(
#         self,
#         features: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         return_parts: bool = False,
#         **_: Any,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         if seen_classes is None:
#             seen_classes = self._infer_seen_from_bank(bank)
#         energy, parts = self.compute_geometry_energy(features, seen_classes=seen_classes, geometry_bank=bank, return_parts=True)
#         if return_parts:
#             out = {"energy": energy}
#             out.update(parts)
#             return out
#         return energy

#     def _geometry_energy(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         return self.geometry_energy(f, means, bases, vars_, reliability, active_ranks, sample_counts)

#     def _geometry_logits(
#         self,
#         f: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         vars_: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         active_ranks: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         return self.geometry_logits(f, means, bases, vars_, reliability, active_ranks, sample_counts)

#     # ------------------------------------------------------------------
#     # Diagnostics
#     # ------------------------------------------------------------------
#     def calibrate_old_new_logits(
#         self,
#         logits: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#     ) -> torch.Tensor:
#         del old_classes, new_classes, old_class_count
#         seen = _as_seen_list(seen_classes)
#         self.assert_logits_valid(logits, seen_classes=seen, context="calibrate_old_new_logits")
#         raise RuntimeError("Old/new logit calibration is removed from the strict classifier.")

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
#         energy: Optional[torch.Tensor] = None,
#     ) -> Dict[str, Any]:
#         seen = _as_seen_list(seen_classes)
#         self.assert_logits_valid(logits, seen_classes=seen, targets=targets_local, context="classifier_diagnostics")
#         old_mask, new_mask, old_prefix = self._old_new_masks(
#             seen,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=old_class_count,
#             device=logits.device,
#         )

#         pred_local = logits.argmax(dim=1)
#         seen_tensor = torch.as_tensor(seen, device=logits.device, dtype=torch.long)
#         pred_global = seen_tensor.index_select(0, pred_local) if pred_local.numel() else torch.empty((0,), device=logits.device, dtype=torch.long)
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
#             "prediction_global": pred_global.detach().cpu().tolist(),
#             "old_logit_mean": float(old_mean.detach().cpu().item()),
#             "new_logit_mean": float(new_mean.detach().cpu().item()),
#             "old_new_logit_gap": float((old_mean - new_mean).detach().cpu().item()),
#             "max_old_logit": float(old_max.detach().cpu().item()),
#             "max_new_logit": float(new_max.detach().cpu().item()),
#             "invalid_prediction_rate": 0.0,
#             "prediction_distribution": {int(seen[i]): int(counts[i].item()) for i in range(len(seen))},
#             "per_class_prediction_count": {int(seen[i]): int(counts[i].item()) for i in range(len(seen))},
#             "uses_old_new_logit_calibration": False,
#         }

#         if targets_local is not None:
#             y = targets_local.to(device=logits.device).long().flatten()
#             if y.numel() != logits.size(0):
#                 raise RuntimeError("classifier_diagnostics: targets/logits batch mismatch.")
#             correct = pred_local.eq(y)
#             out["accuracy"] = float(correct.float().mean().detach().cpu().item()) if y.numel() else 0.0
#             old_y = old_mask.index_select(0, y) if y.numel() else torch.empty((0,), device=logits.device, dtype=torch.bool)
#             new_y = new_mask.index_select(0, y) if y.numel() else torch.empty((0,), device=logits.device, dtype=torch.bool)
#             out["old_accuracy"] = float(correct[old_y].float().mean().detach().cpu().item()) if bool(old_y.any().item()) else 0.0
#             out["new_accuracy"] = float(correct[new_y].float().mean().detach().cpu().item()) if bool(new_y.any().item()) else 0.0

#         if energy is not None and targets_local is not None:
#             em = self.energy_margin_statistics(energy, targets_local)
#             out.update({
#                 "energy_mean_margin": float(em["mean_margin"].detach().cpu().item()),
#                 "energy_min_margin": float(em["min_margin"].detach().cpu().item()),
#                 "energy_violation_rate": float(em["violation_rate"].detach().cpu().item()),
#                 "energy_accuracy": float(em["accuracy"].detach().cpu().item()),
#             })
#             on = self.old_new_energy_statistics(
#                 energy,
#                 targets_local,
#                 old_class_count=old_prefix,
#                 old_mask=old_mask,
#                 new_mask=new_mask,
#             )
#             out.update({f"old_new_{k}": float(v.detach().cpu().item()) for k, v in on.items()})
#         return out

#     @torch.no_grad()
#     def energy_margin_statistics(
#         self,
#         energy: torch.Tensor,
#         labels: torch.Tensor,
#         *,
#         sample_counts: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         del sample_counts
#         if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#             z = self._zero * 0.0
#             return {"mean_margin": z, "min_margin": z, "violation_rate": z, "accuracy": z}
#         if energy.dim() != 2:
#             raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#         y = _as_long_1d(labels, device=energy.device, name="labels")
#         if y.numel() != energy.size(0):
#             raise RuntimeError("labels/energy batch mismatch")
#         if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= energy.size(1)):
#             raise RuntimeError("labels outside local energy range")
#         true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#         true_mask = torch.zeros_like(energy, dtype=torch.bool).scatter(1, y.view(-1, 1), True)
#         wrong = energy.masked_fill(true_mask, float("inf"))
#         nearest_wrong = wrong.min(dim=1).values
#         valid = torch.isfinite(nearest_wrong)
#         margin = nearest_wrong[valid] - true_e[valid]
#         pred = energy.argmin(dim=1)
#         z = energy.sum() * 0.0
#         return {
#             "mean_margin": margin.mean() if margin.numel() else z,
#             "min_margin": margin.min() if margin.numel() else z,
#             "violation_rate": (margin <= 0).float().mean() if margin.numel() else z,
#             "accuracy": (pred == y).float().mean() if y.numel() else z,
#         }

#     @torch.no_grad()
#     def old_new_energy_statistics(
#         self,
#         energy: torch.Tensor,
#         labels: torch.Tensor,
#         *,
#         old_class_count: Optional[int] = None,
#         old_mask: Optional[torch.Tensor] = None,
#         new_mask: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         del sample_counts
#         z = energy.sum() * 0.0 if torch.is_tensor(energy) else self._zero * 0.0
#         if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#             return {"new_into_old_rate": z, "old_into_new_rate": z, "old_group_win_rate": z, "new_group_win_rate": z, "mean_old_new_gap": z}
#         C = int(energy.size(1))
#         if old_mask is None or new_mask is None:
#             old = int(max(0, min(int(old_class_count or 0), C)))
#             old_mask = torch.arange(C, device=energy.device) < old
#             new_mask = ~old_mask
#         else:
#             old_mask = old_mask.to(device=energy.device).bool().flatten()
#             new_mask = new_mask.to(device=energy.device).bool().flatten()
#         if old_mask.numel() != C or new_mask.numel() != C or not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
#             return {"new_into_old_rate": z, "old_into_new_rate": z, "old_group_win_rate": z, "new_group_win_rate": z, "mean_old_new_gap": z}
#         y = _as_long_1d(labels, device=energy.device, name="labels")
#         old_min = energy[:, old_mask].min(dim=1).values
#         new_min = energy[:, new_mask].min(dim=1).values
#         old_win = old_min < new_min
#         new_win = new_min < old_min
#         old_labels = old_mask.index_select(0, y)
#         new_labels = new_mask.index_select(0, y)
#         return {
#             "new_into_old_rate": old_win[new_labels].float().mean() if bool(new_labels.any().item()) else z,
#             "old_into_new_rate": new_win[old_labels].float().mean() if bool(old_labels.any().item()) else z,
#             "old_group_win_rate": old_win.float().mean(),
#             "new_group_win_rate": new_win.float().mean(),
#             "mean_old_new_gap": (new_min - old_min).mean(),
#         }

#     @torch.no_grad()
#     def old_new_margin_report_from_energy(
#         self,
#         energy: torch.Tensor,
#         labels: torch.Tensor,
#         *,
#         old_class_count: Optional[int] = None,
#         old_mask: Optional[torch.Tensor] = None,
#         new_mask: Optional[torch.Tensor] = None,
#         margin: float = 0.25,
#         sample_counts: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         del sample_counts
#         z = energy.sum() * 0.0
#         if energy.dim() != 2:
#             raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#         y = _as_long_1d(labels, device=energy.device, name="labels")
#         if y.numel() != energy.size(0):
#             raise RuntimeError("labels/energy batch mismatch")
#         C = int(energy.size(1))
#         if old_mask is None or new_mask is None:
#             old = int(max(0, min(int(old_class_count or 0), C)))
#             old_mask = torch.arange(C, device=energy.device) < old
#             new_mask = ~old_mask
#         else:
#             old_mask = old_mask.to(device=energy.device).bool().flatten()
#             new_mask = new_mask.to(device=energy.device).bool().flatten()
#         pred = energy.argmin(dim=1)
#         acc = (pred == y).float().mean() if y.numel() else z
#         if old_mask.numel() != C or new_mask.numel() != C or not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
#             return {
#                 "accuracy": acc,
#                 "old_accuracy": z,
#                 "new_accuracy": acc,
#                 "hm": z,
#                 "old_win_rate": z,
#                 "new_win_rate": z,
#                 "new_into_old_rate": z,
#                 "old_into_new_rate": z,
#                 "new_margin_mean": z,
#                 "new_margin_min": z,
#                 "new_violation_rate": z,
#                 "old_boundary_margin_mean": z,
#                 "old_boundary_margin_min": z,
#                 "old_boundary_violation_rate": z,
#                 "mean_true_vs_opposite_margin": z,
#             }
#         old_labels = old_mask.index_select(0, y)
#         new_labels = new_mask.index_select(0, y)
#         old_min = energy[:, old_mask].min(dim=1).values
#         new_min = energy[:, new_mask].min(dim=1).values
#         old_win = old_min < new_min
#         new_win = new_min < old_min
#         true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#         new_margin = old_min[new_labels] - true_e[new_labels] if bool(new_labels.any().item()) else energy.new_empty((0,))
#         old_margin = new_min[old_labels] - true_e[old_labels] if bool(old_labels.any().item()) else energy.new_empty((0,))
#         old_acc = (pred[old_labels] == y[old_labels]).float().mean() if bool(old_labels.any().item()) else z
#         new_acc = (pred[new_labels] == y[new_labels]).float().mean() if bool(new_labels.any().item()) else z
#         hm = (2 * old_acc * new_acc / (old_acc + new_acc + 1e-8)) if bool(old_labels.any().item()) and bool(new_labels.any().item()) else z
#         both = torch.cat([new_margin, old_margin]) if (new_margin.numel() + old_margin.numel()) else energy.new_empty((0,))
#         return {
#             "accuracy": acc,
#             "old_accuracy": old_acc,
#             "new_accuracy": new_acc,
#             "hm": hm,
#             "old_win_rate": old_win.float().mean(),
#             "new_win_rate": new_win.float().mean(),
#             "new_into_old_rate": old_win[new_labels].float().mean() if bool(new_labels.any().item()) else z,
#             "old_into_new_rate": new_win[old_labels].float().mean() if bool(old_labels.any().item()) else z,
#             "new_margin_mean": new_margin.mean() if new_margin.numel() else z,
#             "new_margin_min": new_margin.min() if new_margin.numel() else z,
#             "new_violation_rate": (new_margin <= float(margin)).float().mean() if new_margin.numel() else z,
#             "old_boundary_margin_mean": old_margin.mean() if old_margin.numel() else z,
#             "old_boundary_margin_min": old_margin.min() if old_margin.numel() else z,
#             "old_boundary_violation_rate": (old_margin <= float(margin)).float().mean() if old_margin.numel() else z,
#             "mean_true_vs_opposite_margin": both.mean() if both.numel() else z,
#         }

#     @torch.no_grad()
#     def old_geometry_risk_features_from_bank(
#         self,
#         features: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#     ) -> Dict[str, torch.Tensor]:
#         if int(old_class_count) <= 0:
#             z = torch.zeros((features.size(0),), device=features.device, dtype=features.dtype)
#             return {
#                 "nearest_old_energy": z,
#                 "old_energy_margin": z,
#                 "nearest_old_reliability": torch.ones_like(z),
#                 "nearest_old_residual_variance": z,
#                 "nearest_old_class": torch.zeros_like(z, dtype=torch.long),
#                 "old_membership": z,
#                 "risk_features": torch.zeros((features.size(0), 4), device=features.device, dtype=features.dtype),
#             }
#         full_seen = self._infer_seen_from_bank(bank)
#         old_seen = full_seen[: int(old_class_count)]
#         energy, parts = self.compute_geometry_energy(features, seen_classes=old_seen, geometry_bank=bank, return_parts=True)
#         C_old = int(energy.size(1))
#         sorted_e, sorted_idx = torch.sort(energy, dim=1)
#         nearest = sorted_e[:, 0]
#         margin = sorted_e[:, 1] - sorted_e[:, 0] if C_old > 1 else torch.ones_like(nearest)
#         nearest_local = sorted_idx[:, 0].long()
#         nearest_global = parts["global_class_ids"].index_select(0, nearest_local)
#         rel = torch.ones_like(nearest)
#         res = torch.zeros_like(nearest)
#         bank_dict = self._bank_dict(bank)
#         row_idx = parts["row_indices"].index_select(0, nearest_local)
#         if "reliability" in bank_dict and torch.is_tensor(bank_dict["reliability"]):
#             rel_all = bank_dict["reliability"].to(device=features.device, dtype=features.dtype)
#             rel = rel_all.index_select(0, row_idx).clamp(0.0, 1.0)
#         if "res_vars" in bank_dict and torch.is_tensor(bank_dict["res_vars"]):
#             rv_all = bank_dict["res_vars"].to(device=features.device, dtype=features.dtype)
#             res = rv_all.index_select(0, row_idx).clamp_min(0.0)
#         risk_features = torch.stack(
#             [torch.log1p(nearest.clamp_min(0.0)), torch.log1p(margin.clamp_min(0.0)), rel, torch.log1p(res)], dim=1
#         )
#         risk_features = torch.nan_to_num(risk_features, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
#         return {
#             "nearest_old_energy": nearest,
#             "old_energy_margin": margin,
#             "nearest_old_reliability": rel,
#             "nearest_old_residual_variance": res,
#             "nearest_old_class": nearest_global,
#             "old_membership": torch.exp(-nearest.clamp_min(0.0)),
#             "risk_features": risk_features,
#         }

#     @torch.no_grad()
#     def geometry_state_admission_report(
#         self,
#         features: torch.Tensor,
#         labels: torch.Tensor,
#         *,
#         geometry_bank: Any,
#         seen_classes: Iterable[int],
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         margin: float = 0.25,
#         **_: Any,
#     ) -> Dict[str, torch.Tensor]:
#         seen = _as_seen_list(seen_classes)
#         out = self.forward(
#             features,
#             seen_classes=seen,
#             geometry_bank=geometry_bank,
#             targets=labels,
#             targets_are_global=True,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=old_class_count,
#             return_energy=True,
#             return_parts=True,
#         )
#         targets_local = self.global_to_local_labels(labels, seen)
#         old_mask, new_mask, old_prefix = self._old_new_masks(
#             seen,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=old_class_count,
#             device=features.device,
#         )
#         return self.old_new_margin_report_from_energy(
#             out["energy"],
#             targets_local,
#             old_class_count=old_prefix,
#             old_mask=old_mask,
#             new_mask=new_mask,
#             margin=margin,
#         )

#     sglat_candidate_admission_report = geometry_state_admission_report
#     candidate_admission_report = geometry_state_admission_report

#     @torch.no_grad()
#     def transport_effect_report(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         del args, kwargs
#         z = self._zero * 0.0
#         return {"total": z, "new_violation_rate": z, "old_boundary_violation_rate": z}

#     @torch.no_grad()
#     def method_summary(self) -> Dict[str, object]:
#         return {
#             "method_path": "low_rank_geometry_replay_residual_geometry_adaptation_hsi_necil",
#             "architecture": "Low-Rank Geometry Replay and Residual Geometry Adaptation for HSI NECIL",
#             "output_contract": "[B, len(seen_classes)]",
#             "uses_geometry_bank": True,
#             "uses_feature_low_rank_energy": True,
#             "uses_low_rank_logdet_energy": bool(self.use_logdet_energy and self.logdet_energy_weight > 0.0),
#             "uses_reliability_penalty": bool(self.use_reliability_penalty and self.reliability_energy_weight > 0.0),
#             "uses_old_new_logit_calibration": False,
#             "uses_measured_energy_calibration": False,
#             "uses_adaptive_boundary": False,
#             "uses_spectral_classifier_energy": False,
#             "uses_band_energy": False,
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
#         mode: str = "geometry_only",
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
#             supplied_bank = {
#                 "means": means,
#                 "bases": bases,
#                 "variances": variances,
#                 "sample_counts": legacy_kwargs.get("subspace_sample_counts"),
#                 "reliability": legacy_kwargs.get("subspace_reliability"),
#                 "active_ranks": legacy_kwargs.get("subspace_active_ranks"),
#             }

#         if supplied_bank is None:
#             raise ValueError("forward requires geometry_bank/bank or subspace_* tensors.")

#         bank_dict = self._bank_dict(supplied_bank)
#         if seen_classes is None:
#             seen_classes = self._infer_seen_from_bank(bank_dict)
#         seen = _as_seen_list(seen_classes)
#         mode_norm = self.normalize_mode(mode)
#         self.expand_to_seen_classes(seen)

#         parts = self.compute_geometry_logits(features, seen_classes=seen, geometry_bank=bank_dict, return_parts=True)
#         logits = parts["logits"]

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
#             "energy_calibrated": torch.tensor(False, device=features.device),
#         }
#         if return_energy or return_parts or return_diagnostics:
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
#                 energy=parts["energy"],
#             )
#         return out


# # -----------------------------------------------------------------------------
# # Standalone loss helpers used by trainers
# # -----------------------------------------------------------------------------


# def geometry_energy_margin_loss(
#     energy: torch.Tensor,
#     labels: torch.Tensor,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         return _zero_like(labels if torch.is_tensor(labels) else None)
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
#     old_class_count: Optional[int] = None,
#     margin: float = 0.25,
#     valid_mask: Optional[torch.Tensor] = None,
#     old_mask: Optional[torch.Tensor] = None,
#     new_mask: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     del valid_mask
#     if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#         return _zero_like(labels if torch.is_tensor(labels) else None)
#     if energy.dim() != 2:
#         raise RuntimeError(f"energy must be [B,S], got {tuple(energy.shape)}")
#     C = int(energy.size(1))
#     if old_mask is None or new_mask is None:
#         old = int(max(0, min(int(old_class_count or 0), C)))
#         old_mask = torch.arange(C, device=energy.device) < old
#         new_mask = ~old_mask
#     else:
#         old_mask = old_mask.to(device=energy.device).bool().flatten()
#         new_mask = new_mask.to(device=energy.device).bool().flatten()
#     if old_mask.numel() != C or new_mask.numel() != C:
#         raise RuntimeError("old_mask/new_mask must match energy width")
#     if not bool(old_mask.any().item()) or not bool(new_mask.any().item()):
#         return energy.sum() * 0.0
#     y = labels.to(device=energy.device).long().flatten()
#     if y.numel() != energy.size(0):
#         raise RuntimeError("labels/energy batch mismatch")
#     if y.numel() and (int(y.min().item()) < 0 or int(y.max().item()) >= C):
#         raise RuntimeError("labels outside local energy range")
#     true_e = energy.gather(1, y.view(-1, 1)).squeeze(1)
#     old_min = energy[:, old_mask].min(dim=1).values
#     new_min = energy[:, new_mask].min(dim=1).values
#     is_old = old_mask.index_select(0, y)
#     opposite = torch.where(is_old, new_min, old_min)
#     loss = F.relu(true_e + float(margin) - opposite)
#     finite = torch.isfinite(loss)
#     return loss[finite].mean() if bool(finite.any().item()) else energy.sum() * 0.0

