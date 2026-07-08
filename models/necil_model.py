from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SSMBackbone
from models.geometry_bank import GeometryBank
from models.classifier import GeometryEnergyClassifier


def _to_bool(value: Any, default: bool = False) -> bool:
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
    return bool(value)


def _filter_supported_kwargs(cls_or_fn: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters)
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        c = int(value)
        if c < 0:
            raise ValueError(f"Class ids must be non-negative, got {c}.")
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _normalize_classifier_mode(mode: Optional[str], default: str = "geometry") -> str:
    m = str(default if mode is None else mode).lower().strip()
    aliases = {
        "": "geometry",
        "none": "geometry",
        "geo": "geometry",
        "geometry": "geometry",
        "geometry_only": "geometry",
        "geometry-only": "geometry",
        "feature_geometry": "geometry",
        "low_rank_geometry": "geometry",
        "spectral_geometry": "geometry",
        "spectral_coupled_geometry": "geometry",
        "calibrated": "geometry",
        "calibrated_geometry": "geometry",
        "base_ce": "base_ce",
    }
    out = aliases.get(m, m)
    if out not in {"geometry", "base_ce"}:
        raise ValueError(f"Unsupported classifier mode {mode!r}. Use geometry_only/geometry or base_ce.")
    return out


def _normalize_incremental_update_mode(mode: Optional[str]) -> str:
    """Return the only legal incremental architecture identity.

    Feature adapters, transport, and score calibration are not aliases for the
    method. They are different architectures and therefore fail loudly.
    """
    m = str(mode or "spectral_coupled_geometry_replay").lower().strip()
    aliases = {
        "": "spectral_coupled_geometry_replay",
        "none": "spectral_coupled_geometry_replay",
        "clean": "spectral_coupled_geometry_replay",
        "main": "spectral_coupled_geometry_replay",
        "descriptor": "spectral_coupled_geometry_replay",
        "descriptor_only": "spectral_coupled_geometry_replay",
        "descriptor_refinement": "spectral_coupled_geometry_replay",
        "scbgr": "spectral_coupled_geometry_replay",
        "scb-gr": "spectral_coupled_geometry_replay",
        "sctgr": "spectral_coupled_geometry_replay",
        "sctgr_rga": "spectral_coupled_geometry_replay",
        "spectral_coupled": "spectral_coupled_geometry_replay",
        "spectral_coupled_replay": "spectral_coupled_geometry_replay",
        "spectral_coupled_geometry_replay": "spectral_coupled_geometry_replay",
    }
    forbidden = {
        "geometry_gated_adapter", "geometry_adapter", "gated_adapter", "adapter",
        "g2rpa", "g2-rpa", "transport", "geometry_transport",
    }
    if m in forbidden:
        raise ValueError(
            f"incremental_update_mode={mode!r} selects a forbidden architecture. "
            "Use spectral_coupled_geometry_replay."
        )
    out = aliases.get(m, m)
    if out != "spectral_coupled_geometry_replay":
        raise ValueError(f"Unsupported incremental_update_mode={mode!r}.")
    return out


class NECILModel(nn.Module):
    """NECIL-HSI model with one canonical feature space and one geometry memory.

    Architectural contract
    ----------------------
    * Base and incremental samples use the same canonical projected feature z.
    * GeometryBank is the only old-class memory.
    * Physical spectra are used to build spectral tangent/coupling descriptors,
      never as a second inference classifier.
    * Incremental backbone, projection, norm, and classifier are frozen.
    * Only temporary new-row descriptor residuals are optimized by the trainer.
    * No feature adapter, transport, calibration, teacher, or prototype branch.
    """

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.d_model = int(getattr(args, "d_model", 128))
        self.subspace_rank = int(getattr(args, "subspace_rank", getattr(args, "rank", 5)))
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.subspace_rank < 0:
            raise ValueError("subspace_rank must be non-negative.")

        self.current_phase = 0
        self.old_class_count = 0
        self.old_classes: List[int] = []
        self.current_num_classes = 0
        self.seen_classes: List[int] = []
        self.base_mode_active = True
        self.incremental_mode_active = False
        self._incremental_frozen_modules: List[str] = []

        self.incremental_update_mode = _normalize_incremental_update_mode(
            getattr(args, "incremental_update_mode", None)
        )
        try:
            setattr(args, "incremental_update_mode", self.incremental_update_mode)
        except Exception:
            pass

        # One exact energy is used by GeometryBank replay admission, classifier
        # scoring, base margin training, incremental losses, and evaluation.
        # Normalize it before constructing the classifier so stale command-line
        # or checkpoint-era arguments cannot reactivate a different score.
        self._install_strict_energy_contract(args)

        # Runtime identity flags used by trainers. All alternative mechanisms are
        # hard-disabled rather than merely left unused.
        self.use_incremental_adapter = False
        self.use_geometry_gated_adapter = False
        self.use_geometry_calibrator = False
        self.use_energy_calibrator = False
        self.use_adaptive_boundary = False
        self.use_bicyc_geometry_cycle = False
        self.use_geometry_transport = False
        self.use_sglat_transport = False
        self.use_spectral_geometry = False
        self.geometry_plastic_adapter: Optional[nn.Module] = None
        self.semantic_encoder: Optional[nn.Module] = None
        self.concept_encoder: Optional[nn.Module] = None
        self.default_eval_classifier_mode = "geometry_only"
        self.default_incremental_classifier_mode = "geometry_only"

        self.backbone = SSMBackbone(args)
        projection_dropout = float(getattr(args, "projection_dropout", 0.0))
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.norm = nn.LayerNorm(self.d_model)

        self.normalize_geometry_features = _to_bool(
            getattr(args, "normalize_geometry_features", True), True
        )
        raw_scale = float(getattr(args, "geometry_feature_scale", 0.0) or 0.0)
        self.geometry_feature_scale = raw_scale if raw_scale > 0 else math.sqrt(float(self.d_model))
        self.geometry_feature_clamp = float(getattr(args, "geometry_feature_clamp", 0.0) or 0.0)
        self.spectral_summary_mode = str(getattr(args, "spectral_summary_mode", "center")).lower().strip()
        if self.spectral_summary_mode not in {"center", "mean"}:
            raise ValueError("spectral_summary_mode must be 'center' or 'mean'.")
        pca_components = int(getattr(args, "pca_components", 0) or 0)
        self.default_spectral_physical = _to_bool(
            getattr(args, "spectral_summary_is_physical", pca_components <= 0),
            pca_components <= 0,
        )
        self.min_band_mass = float(getattr(args, "min_band_mass", 1e-8))

        bank_kwargs = _filter_supported_kwargs(
            GeometryBank.__init__,
            {
                "device": self.device,
                "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
                "variance_shrinkage": float(getattr(args, "geometry_variance_shrinkage", 0.10)),
                "max_variance_ratio": float(getattr(args, "geometry_max_variance_ratio", 50.0)),
                "min_reliability": float(getattr(args, "geometry_min_reliability", 0.05)),
                "reliability_sample_alpha": float(getattr(args, "reliability_sample_alpha", 20.0)),
                "rank_energy_threshold": float(getattr(args, "rank_energy_threshold", 0.95)),
                "rank_eigen_ratio_threshold": float(getattr(args, "rank_eigen_ratio_threshold", 1e-3)),
                "min_active_rank": int(getattr(args, "min_active_rank", 1)),
                "small_class_rank_threshold_1": int(getattr(args, "small_class_rank_threshold_1", 30)),
                "small_class_rank_threshold_2": int(getattr(args, "small_class_rank_threshold_2", 80)),
                "small_class_rank_threshold_3": int(getattr(args, "small_class_rank_threshold_3", 150)),
                "small_class_rank_cap_1": int(getattr(args, "small_class_rank_cap_1", 1)),
                "small_class_rank_cap_2": int(getattr(args, "small_class_rank_cap_2", 3)),
                "small_class_rank_cap_3": int(getattr(args, "small_class_rank_cap_3", 4)),
                "small_class_extra_shrinkage": float(getattr(args, "small_class_extra_shrinkage", 0.35)),
                "spectral_rank": int(getattr(args, "spectral_geometry_rank", getattr(args, "spectral_rank", 6))),
                "spectral_rank_energy_threshold": float(getattr(args, "spectral_rank_energy_threshold", 0.95)),
                "spectral_rank_eigen_ratio_threshold": float(getattr(args, "spectral_rank_eigen_ratio_threshold", 1e-3)),
                "spectral_variance_floor": float(getattr(args, "spectral_variance_floor", 1e-6)),
                "coupling_ridge": float(getattr(args, "coupling_ridge", 1e-3)),
                "coupling_min_reliability": float(getattr(args, "coupling_min_reliability", 0.20)),
                "spectral_tangent_clip": float(getattr(args, "spectral_tangent_clip", 2.5)),
                "replay_candidate_multiplier": int(getattr(args, "replay_candidate_multiplier", 4)),
            },
        )
        self.geometry_bank = GeometryBank(self.d_model, self.subspace_rank, **bank_kwargs)

        clf_kwargs = _filter_supported_kwargs(
            GeometryEnergyClassifier.__init__,
            {
                "initial_classes": 0,
                "d_model": self.d_model,
                "logit_scale": float(getattr(args, "loss_scale", getattr(args, "logit_scale", 8.0))),
                "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
                "residual_variance_scale": 1.0,
                "normalize_energy_by_dim": True,
                "energy_normalize_by_dim": True,
                "use_logdet_energy": False,
                "logdet_energy_weight": 0.0,
                "logdet_normalize_by_dim": False,
                "center_logdet_energy": False,
                "use_reliability_penalty": False,
                "reliability_energy_weight": 0.0,
                "center_reliability_energy": False,
                "use_old_new_calibration": False,
                "invalid_class_energy": float(getattr(args, "invalid_class_energy", 1e6)),
                "logit_clip": float(getattr(args, "geometry_logit_clip", 0.0)),
            },
        )
        self.classifier = GeometryEnergyClassifier(**clf_kwargs)
        self._assert_strict_energy_contract()

        self.base_ce_head: Optional[nn.Linear] = None
        self.base_ce_num_classes = 0
        self.to(self.device)

    def _install_strict_energy_contract(self, args: Any) -> None:
        """Install the single SCTGR geometry-energy contract.

        Reliability and spectral coupling control replay allocation, trust, and
        candidate generation only. They must never modify classifier logits.
        """
        contract = {
            "residual_variance_scale": 1.0,
            "energy_normalize_by_dim": True,
            "use_logdet_energy": False,
            "logdet_energy_weight": 0.0,
            "logdet_normalize_by_dim": False,
            "center_logdet_energy": False,
            "use_reliability_penalty": False,
            "reliability_energy_weight": 0.0,
            "center_reliability_energy": False,
            "use_spectral_geometry": False,
            "spectral_energy_weight": 0.0,
            "band_energy_weight": 0.0,
        }
        for name, value in contract.items():
            try:
                setattr(args, name, value)
            except Exception:
                pass
            setattr(self, name, value)

    def _assert_strict_energy_contract(self) -> None:
        expected = {
            "residual_variance_scale": 1.0,
            "energy_normalize_by_dim": True,
            "use_logdet_energy": False,
            "logdet_energy_weight": 0.0,
            "center_logdet_energy": False,
            "use_reliability_penalty": False,
            "reliability_energy_weight": 0.0,
            "use_spectral_geometry": False,
            "spectral_energy_weight": 0.0,
            "band_energy_weight": 0.0,
        }
        errors: List[str] = []
        for name, wanted in expected.items():
            actual = getattr(self, name, None)
            if isinstance(wanted, bool):
                if bool(actual) != wanted:
                    errors.append(f"model.{name}={actual!r}, expected {wanted!r}")
            else:
                try:
                    if abs(float(actual) - float(wanted)) > 1e-8:
                        errors.append(f"model.{name}={actual!r}, expected {wanted!r}")
                except Exception:
                    errors.append(f"model.{name}={actual!r}, expected {wanted!r}")

        clf = getattr(self, "classifier", None)
        if clf is not None:
            classifier_expected = {
                "residual_variance_scale": 1.0,
                "normalize_energy_by_dim": True,
                "use_logdet_energy": False,
                "logdet_energy_weight": 0.0,
                "center_logdet_energy": False,
                "use_reliability_penalty": False,
                "reliability_energy_weight": 0.0,
            }
            for name, wanted in classifier_expected.items():
                if not hasattr(clf, name):
                    continue
                actual = getattr(clf, name)
                if isinstance(wanted, bool):
                    if bool(actual) != wanted:
                        errors.append(f"classifier.{name}={actual!r}, expected {wanted!r}")
                elif abs(float(actual) - float(wanted)) > 1e-8:
                    errors.append(f"classifier.{name}={actual!r}, expected {wanted!r}")

        if errors:
            raise RuntimeError(
                "SCTGR energy contract mismatch: " + "; ".join(errors)
            )

    # ------------------------------------------------------------------
    # Canonical feature and physical-spectrum extraction
    # ------------------------------------------------------------------
    def _validate_feature_tensor(
        self,
        features: torch.Tensor,
        name: str,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError(f"{name} must be a tensor, got {type(features)}")
        if features.dim() != 2 or features.size(1) != self.d_model:
            raise RuntimeError(f"{name} must be [B,{self.d_model}], got {tuple(features.shape)}")
        if batch_size is not None and features.size(0) != int(batch_size):
            raise RuntimeError(f"{name} batch mismatch: {features.size(0)} != {int(batch_size)}")
        if not torch.isfinite(features).all():
            raise RuntimeError(f"{name} contains NaN/Inf.")
        return features

    def _canonicalize(self, z: torch.Tensor, *, name: str) -> torch.Tensor:
        z = self._validate_feature_tensor(z, name)
        if self.normalize_geometry_features:
            z = F.normalize(z, p=2, dim=1, eps=1e-8) * float(self.geometry_feature_scale)
        if self.geometry_feature_clamp > 0:
            z = z.clamp(-self.geometry_feature_clamp, self.geometry_feature_clamp)
        if not torch.isfinite(z).all():
            raise RuntimeError(f"{name} became non-finite after canonicalization.")
        return z

    def _center_spectrum_from_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x[:, :, x.size(-2) // 2, x.size(-1) // 2] if self.spectral_summary_mode == "center" else x.mean((-1, -2))
        if x.dim() == 3:
            return x[:, :, x.size(-1) // 2] if self.spectral_summary_mode == "center" else x.mean(-1)
        if x.dim() == 2:
            return x
        raise RuntimeError(f"Unsupported input shape for spectral summary: {tuple(x.shape)}")

    def _prepare_spectral_summary(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        spectral_summary: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, bool]:
        if spectral_summary is None or not torch.is_tensor(spectral_summary) or spectral_summary.numel() == 0:
            s = self._center_spectrum_from_input(x).to(features.device, features.dtype)
            physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
        else:
            s = spectral_summary.to(features.device, features.dtype)
            if s.dim() == 4:
                s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
            elif s.dim() == 3:
                s = s[:, :, s.size(-1) // 2] if s.size(0) == features.size(0) else s.reshape(features.size(0), -1)
            elif s.dim() == 1:
                if s.numel() % max(features.size(0), 1) != 0:
                    raise RuntimeError("1-D spectral_summary cannot be aligned with the batch.")
                s = s.reshape(features.size(0), -1)
            elif s.dim() != 2:
                s = s.reshape(features.size(0), -1)
            physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
        if s.dim() != 2 or s.size(0) != features.size(0):
            raise RuntimeError(f"spectral_summary must resolve to [B,S], got {tuple(s.shape)}")
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        return s, bool(physical and s.size(1) > 0)

    def _band_summary(
        self,
        spectral_summary: torch.Tensor,
        band_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if torch.is_tensor(band_weights) and band_weights.numel() > 0:
            bw = band_weights.to(spectral_summary.device, spectral_summary.dtype)
            if bw.shape == spectral_summary.shape:
                bw = torch.nan_to_num(bw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                mass = bw.sum(1, keepdim=True)
                if bool((mass > self.min_band_mass).all().item()):
                    return bw / mass.clamp_min(self.min_band_mass)
        b = spectral_summary.abs()
        mass = b.sum(1, keepdim=True)
        uniform = torch.full_like(b, 1.0 / float(max(b.size(1), 1)))
        return torch.where(mass > self.min_band_mass, b / mass.clamp_min(self.min_band_mass), uniform)

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a tensor, got {type(x)}")
        out = self.backbone(x.to(self.device))
        if isinstance(out, dict):
            if not torch.is_tensor(out.get("features", None)):
                raise RuntimeError("backbone output dict must contain tensor 'features'.")
            result = dict(out)
            result["features"] = self._validate_feature_tensor(out["features"], "backbone.features", x.size(0))
            result.setdefault("backbone_features", result["features"])
            return result
        h = self._validate_feature_tensor(out, "backbone.features", x.size(0))
        return {"features": h, "backbone_features": h}

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        spectral_summary: Optional[torch.Tensor] = None,
        band_weights: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: Optional[bool] = None,
        apply_adapter: Optional[bool] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        if bool(apply_adapter):
            raise RuntimeError("Feature adapters are not part of spectral_coupled_geometry_replay.")
        raw = self.extract_features(x)
        h = self._validate_feature_tensor(raw["features"], "preproject_features", x.size(0))
        z = self._canonicalize(self.norm(self.projection(h) + h), name="canonical_geometry_features")
        s, physical = self._prepare_spectral_summary(
            x.to(self.device), z,
            spectral_summary=spectral_summary,
            spectral_summary_is_physical=spectral_summary_is_physical,
        )
        candidate_bw = band_weights if band_weights is not None else raw.get("band_weights", None)
        band = self._band_summary(s, candidate_bw)
        return {
            "features": z,
            "projected_features": z,
            "geometry_features": z,
            "canonical_features": z,
            "canonical_projected_features": z,
            "pre_adapter_features": z,
            "preproject_features": h,
            "backbone_features": h,
            "spectral_summary": s,
            "spectral_summary_is_physical": torch.tensor(bool(physical), device=z.device, dtype=torch.bool),
            "band_summary": band,
            "band_importance": band,
            "band_weights": candidate_bw if torch.is_tensor(candidate_bw) else None,
            "spectral_features": raw.get("spectral_features", h),
            "spatial_features": raw.get("spatial_features", h),
            "adapter_active": torch.tensor(False, device=z.device),
        }

    def extract_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        kwargs = dict(kwargs)
        kwargs["apply_adapter"] = False
        return self.forward_features(x, **kwargs)

    extract_canonical_projected_features = extract_projected_features

    def extract_adapted_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        raise RuntimeError("Adapted feature extraction is forbidden; use canonical projected features.")

    def extract_geometry_features(
        self,
        x: torch.Tensor,
        *,
        return_dict: bool = False,
        space: str = "canonical",
        **kwargs: Any,
    ):
        space_norm = str(space or "canonical").lower().strip()
        if space_norm not in {"canonical", "pre_adapter", "base"}:
            raise RuntimeError("GeometryBank supports canonical feature space only.")
        out = self.extract_projected_features(x, **kwargs)
        out["geometry_feature_space"] = "canonical"
        out["classifier_feature_space"] = "canonical"
        return out if bool(return_dict) else out["features"]

    @torch.no_grad()
    def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)

    # Legacy adapter APIs fail or return an explicit identity view.
    def adapt_projected_features(self, features: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        z = self._validate_feature_tensor(features, "adapt_projected_features.features")
        zero = torch.zeros_like(z)
        gate = torch.zeros((z.size(0), 1), device=z.device, dtype=z.dtype)
        return {
            "features": z,
            "projected_features": z,
            "geometry_features": z,
            "delta": zero,
            "gate": gate,
            "adapter_delta": zero,
            "adapter_gate": gate,
            "adapter_active": torch.tensor(False, device=z.device),
        }

    # ------------------------------------------------------------------
    # GeometryBank capacity and scoring
    # ------------------------------------------------------------------
    def _infer_seen_classes(self, geometry_bank: Optional[Any] = None) -> List[int]:
        bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
        if hasattr(bank_obj, "get_valid_mask"):
            valid = bank_obj.get_valid_mask().detach().cpu().flatten()
            return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
        bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
        if not isinstance(bank, dict) or not torch.is_tensor(bank.get("sample_counts", None)):
            return list(self.seen_classes)
        counts = bank["sample_counts"].detach().cpu().flatten()
        return [int(i) for i in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()]

    def ensure_class_capacity(
        self,
        class_count: int,
        spectral_dim: int = 0,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        count = int(max(0, class_count))
        dtype = dtype or next(self.parameters()).dtype
        self.geometry_bank.ensure_class_count(count, spectral_dim=int(spectral_dim), dtype=dtype)
        self.current_num_classes = max(self.current_num_classes, count)

    def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
        bank = dict(self.geometry_bank.get_bank())
        if "variances" not in bank:
            bank["variances"] = torch.cat([bank["eigvals"], bank["res_vars"].unsqueeze(-1)], dim=-1)
        bank.setdefault("resvars", bank["res_vars"])
        return bank

    def get_old_subspace_bank(
        self,
        old_class_count: Optional[int] = None,
        old_classes: Optional[Iterable[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        ids = _ordered_unique_ints(old_classes) if old_classes is not None else list(range(int(self.old_class_count if old_class_count is None else old_class_count)))
        bank = self.get_subspace_bank()
        if not ids:
            return {k: v[:0] for k, v in bank.items() if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == len(self.geometry_bank)}
        idx = torch.as_tensor(ids, device=self.geometry_bank.device, dtype=torch.long)
        out: Dict[str, torch.Tensor] = {"class_ids": idx}
        for key, value in bank.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == len(self.geometry_bank):
                out[key] = value.index_select(0, idx)
            elif torch.is_tensor(value):
                out[key] = value
        return out

    def compute_logits_from_features(
        self,
        features: torch.Tensor,
        seen_classes: Optional[Iterable[int]] = None,
        geometry_bank: Optional[Any] = None,
        mode: str = "geometry",
        *,
        classifier_mode: Optional[str] = None,
        targets: Optional[torch.Tensor] = None,
        targets_are_global: bool = False,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        return_energy: bool = False,
        return_parts: bool = False,
        return_diagnostics: bool = False,
        **_: Any,
    ):
        z = self._validate_feature_tensor(features, "compute_logits_from_features.features")
        bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
        seen = _ordered_unique_ints(seen_classes if seen_classes is not None else self._infer_seen_classes(bank_obj))
        if not seen:
            raise RuntimeError("seen_classes is empty. Build GeometryBank rows first.")
        mode_norm = _normalize_classifier_mode(classifier_mode if classifier_mode is not None else mode)
        if mode_norm != "geometry":
            raise RuntimeError("compute_logits_from_features supports geometry-only scoring.")
        self.assert_phase_ready(seen, mode="geometry", require_geometry=True)
        out = self.classifier(
            z,
            seen_classes=seen,
            geometry_bank=bank_obj,
            mode="geometry_only",
            targets=targets,
            targets_are_global=bool(targets_are_global),
            old_classes=old_classes,
            new_classes=new_classes,
            old_class_count=int(self.old_class_count),
            return_energy=return_energy,
            return_parts=return_parts,
            return_diagnostics=return_diagnostics,
        )
        logits = out["logits"] if isinstance(out, dict) else out
        target_local = self.classifier.global_to_local_labels(targets, seen) if targets is not None and targets_are_global else targets
        self.classifier.assert_logits_valid(
            logits,
            seen_classes=seen,
            targets=target_local,
            old_classes=old_classes,
            new_classes=new_classes,
            context="NECILModel.compute_logits_from_features",
        )
        self.current_num_classes = max(self.current_num_classes, max(seen) + 1)
        self.seen_classes = list(seen)
        return out

    forward_classifier = compute_logits_from_features

    def compute_energy_from_features(
        self,
        features: torch.Tensor,
        seen_classes: Optional[Iterable[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        out = self.compute_logits_from_features(
            features,
            seen_classes=seen_classes,
            return_energy=True,
            **kwargs,
        )
        if not isinstance(out, dict) or not torch.is_tensor(out.get("energy", None)):
            raise RuntimeError("Classifier did not return energy.")
        return out["energy"]

    # ------------------------------------------------------------------
    # Replay-ready row construction and safe row refresh
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refresh_geometry_for_classes(
        self,
        class_ids: Iterable[int],
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        spectral_summary: Optional[torch.Tensor] = None,
        band_weights: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: bool = False,
        phase_created: Optional[int] = None,
        freeze_after: bool = False,
    ) -> Dict[str, Any]:
        z = self._validate_feature_tensor(features, "refresh_geometry_for_classes.features")
        y = labels.to(z.device).long().flatten()
        ids = _ordered_unique_ints(class_ids)
        if not ids:
            raise RuntimeError("refresh_geometry_for_classes received no classes.")
        if y.numel() != z.size(0):
            raise RuntimeError("labels/features batch mismatch in refresh_geometry_for_classes.")
        bad = sorted(set(torch.unique(y).detach().cpu().tolist()).difference(ids))
        if bad:
            raise RuntimeError(f"labels contain classes outside refresh ids: {bad}")
        spectral_dim = int(spectral_summary.size(1)) if torch.is_tensor(spectral_summary) and spectral_summary.dim() == 2 else 0
        self.ensure_class_capacity(max(ids) + 1, spectral_dim=spectral_dim, dtype=z.dtype)
        rows = self.geometry_bank.build_candidate_geometry_rows(
            z,
            y,
            spectral_summary=spectral_summary,
            band_weights=band_weights,
            spectral_summary_is_physical=bool(spectral_summary_is_physical),
            class_ids=ids,
        )
        result = self.geometry_bank.commit_candidate_geometry_rows(
            rows,
            allow_frozen_update=False,
            phase_created=int(self.current_phase if phase_created is None else phase_created),
            freeze=bool(freeze_after),
            context="NECILModel.refresh_geometry_for_classes",
        )
        self.geometry_bank.assert_bank_valid(seen_classes=ids, strict=True)
        self.current_num_classes = max(self.current_num_classes, max(ids) + 1)
        return result

    def _row_metadata_for_refresh(self, class_id: int) -> Dict[str, torch.Tensor]:
        c = int(class_id)
        if c < 0 or c >= len(self.geometry_bank):
            return {}
        if float(self.geometry_bank.sample_counts[c].detach().cpu().item()) <= 0:
            return {}
        row = self.geometry_bank.get_class_geometry(c)
        mapping = {
            "spectral_prototype": "spectral_prototype",
            "band_importance": "band_importance",
            "sample_count": "sample_count",
            "active_rank": "active_rank",
            "reliability": "reliability",
            "feature_reliability": "feature_reliability",
            "band_reliability": "band_reliability",
            "spectral_reliability": "spectral_reliability",
            "spectral_basis": "spectral_basis",
            "spectral_eigvals": "spectral_eigvals",
            "spectral_res_var": "spectral_res_var",
            "spectral_active_rank": "spectral_active_rank",
            "spectral_to_feature": "spectral_to_feature",
            "coupling_residual_vars": "coupling_residual_vars",
            "coupling_reliability": "coupling_reliability",
            "spectral_sam_limit": "spectral_sam_limit",
            "spectral_d1_limit": "spectral_d1_limit",
            "spectral_d2_limit": "spectral_d2_limit",
            "energy_quantiles": "energy_quantiles",
            "margin_quantiles": "margin_quantiles",
        }
        return {dst: row[src] for dst, src in mapping.items() if src in row}

    @torch.no_grad()
    def refresh_class_subspace(
        self,
        cls: int,
        mean: torch.Tensor,
        basis: torch.Tensor,
        eigvals: torch.Tensor,
        res_var: Optional[torch.Tensor] = None,
        resvar: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """Compatibility row refresh that never drops spectral coupling metadata."""
        rv = res_var if res_var is not None else resvar
        if rv is None:
            raise ValueError("refresh_class_subspace requires res_var/resvar.")
        c = int(cls)
        self.ensure_class_capacity(c + 1)
        preserved = self._row_metadata_for_refresh(c)
        aliases = {
            "spectral_proto": "spectral_prototype",
            "spectral_curve_mean": "spectral_prototype",
            "spectral_shape_reliability": "spectral_reliability",
        }
        for key, value in kwargs.items():
            target = aliases.get(key, key)
            if value is not None:
                preserved[target] = value
        self.geometry_bank.add_or_update_class_geometry(
            c,
            mean=mean,
            basis=basis,
            eigvals=eigvals,
            res_var=rv,
            phase_created=int(kwargs.get("phase_created", self.current_phase)),
            freeze=bool(kwargs.get("freeze", False)),
            allow_frozen_update=bool(kwargs.get("allow_frozen_update", False)),
            **{k: v for k, v in preserved.items() if k not in {"phase_created", "freeze", "allow_frozen_update"}},
        )

    @torch.no_grad()
    def sample_geometry_replay(
        self,
        class_ids: Iterable[int],
        samples_per_class: int | Mapping[int, int] = 16,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        return self.geometry_bank.sample_replay(
            class_ids,
            samples_per_class=samples_per_class,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Memory snapshot and checkpoint compatibility
    # ------------------------------------------------------------------
    @torch.no_grad()
    def export_memory_snapshot(self) -> Dict[str, Any]:
        snapshot = self.geometry_bank.export_snapshot()
        snapshot.update({
            "current_phase": int(self.current_phase),
            "old_class_count": int(self.old_class_count),
            "old_classes": list(self.old_classes),
            "current_num_classes": int(self.current_num_classes),
            "seen_classes": list(self.seen_classes),
            "feature_contract": self.feature_contract(),
        })
        return snapshot

    @torch.no_grad()
    def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True) -> None:
        if not snapshot:
            if strict:
                raise ValueError("empty memory snapshot")
            return
        self._assert_snapshot_feature_contract(snapshot, strict=strict)
        self.geometry_bank.load_snapshot(snapshot, strict=strict)
        self.current_phase = int(snapshot.get("current_phase", self.current_phase))
        self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
        self.old_classes = [int(c) for c in snapshot.get("old_classes", list(range(self.old_class_count)))]
        self.current_num_classes = int(snapshot.get("current_num_classes", len(self.geometry_bank)))
        self.seen_classes = [int(c) for c in snapshot.get("seen_classes", self._infer_seen_classes())]

    def feature_contract(self) -> Dict[str, Any]:
        return {
            "d_model": int(self.d_model),
            "subspace_rank": int(self.subspace_rank),
            "spectral_rank": int(self.geometry_bank.spectral_rank),
            "normalize_geometry_features": bool(self.normalize_geometry_features),
            "geometry_feature_scale": float(self.geometry_feature_scale),
            "spectral_summary_mode": str(self.spectral_summary_mode),
            "classifier_contract": "logits[B,len(seen_classes)]",
            "incremental_update_mode": str(self.incremental_update_mode),
            "geometry_feature_space": "canonical",
            "spectral_coupled_replay": True,
            "feature_adapter_available": False,
            "energy_contract": {
                "residual_variance_scale": 1.0,
                "normalize_energy_by_dim": True,
                "use_logdet_energy": False,
                "logdet_energy_weight": 0.0,
                "center_logdet_energy": False,
                "use_reliability_penalty": False,
                "reliability_energy_weight": 0.0,
            },
        }

    def _assert_snapshot_feature_contract(self, snapshot: Dict[str, Any], *, strict: bool) -> None:
        if not strict:
            return
        old = snapshot.get("feature_contract", snapshot.get("geometry_feature_contract", None))
        if not isinstance(old, dict):
            return
        cur = self.feature_contract()
        mismatches = []
        for key in ("d_model", "subspace_rank", "normalize_geometry_features", "spectral_summary_mode"):
            if key in old and old[key] != cur[key]:
                mismatches.append(f"{key}: snapshot={old[key]!r}, current={cur[key]!r}")
        if "geometry_feature_scale" in old and abs(float(old["geometry_feature_scale"]) - float(cur["geometry_feature_scale"])) > 1e-6:
            mismatches.append(
                f"geometry_feature_scale: snapshot={old['geometry_feature_scale']!r}, current={cur['geometry_feature_scale']!r}"
            )
        if mismatches:
            raise RuntimeError("Memory snapshot was built under a different feature contract: " + "; ".join(mismatches))

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # type: ignore[override]
        """Load old checkpoints while discarding removed adapter parameters only."""
        cleaned = {
            k: v for k, v in state_dict.items()
            if "geometry_plastic_adapter" not in k and "incremental_adapter" not in k
        }
        try:
            return super().load_state_dict(cleaned, strict=strict, assign=assign)
        except TypeError:  # older PyTorch
            return super().load_state_dict(cleaned, strict=strict)

    # ------------------------------------------------------------------
    # Phase modes and freezing
    # ------------------------------------------------------------------
    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], value: bool) -> None:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = bool(value)

    def freeze_backbone_except_allowed(
        self,
        *,
        allow_last_blocks: bool = False,
        allow_projection: bool = False,
        allow_norm: Optional[bool] = None,
        **_: Any,
    ) -> None:
        self._set_requires_grad(self.backbone, False)
        if bool(allow_last_blocks) and hasattr(self.backbone, "get_last_blocks"):
            for block in self.backbone.get_last_blocks():
                self._set_requires_grad(block, True)
        self._set_requires_grad(self.projection, bool(allow_projection))
        self._set_requires_grad(self.norm, bool(allow_projection if allow_norm is None else allow_norm))

    def freeze_backbone_only(self) -> None:
        self._set_requires_grad(self.backbone, False)

    def unfreeze_backbone(self) -> None:
        self._set_requires_grad(self.backbone, True)

    def freeze_projection_head(self) -> None:
        self._set_requires_grad(self.projection, False)
        self._set_requires_grad(self.norm, False)

    def unfreeze_projection_head(self) -> None:
        self._set_requires_grad(self.projection, True)
        self._set_requires_grad(self.norm, True)

    def freeze_semantic_encoder(self) -> None:
        self._set_requires_grad(self.semantic_encoder, False)
        self._set_requires_grad(self.concept_encoder, False)

    def freeze_classifier(self) -> None:
        self._set_requires_grad(self.classifier, False)

    def unfreeze_classifier(self) -> None:
        raise RuntimeError("The geometry classifier is parameter-free/frozen in the main architecture.")

    def freeze_classes(self, class_ids_or_count: Iterable[int] | int) -> None:
        if isinstance(class_ids_or_count, int):
            self.geometry_bank.freeze_classes_up_to(int(max(0, class_ids_or_count)))
        else:
            self.geometry_bank.freeze_classes(_ordered_unique_ints(class_ids_or_count))

    def freeze_old_geometry_states(self, old_class_count: Optional[Any] = None) -> None:
        if old_class_count is None:
            ids = list(self.old_classes) if self.old_classes else list(range(self.old_class_count))
        elif isinstance(old_class_count, int):
            ids = list(range(int(max(0, old_class_count))))
        else:
            ids = _ordered_unique_ints(old_class_count)
        self.old_classes = list(ids)
        self.old_class_count = len(ids)
        self.freeze_classes(ids)

    def freeze_base_ce_head(self) -> None:
        self._set_requires_grad(self.base_ce_head, False)

    def unfreeze_base_ce_head(self) -> None:
        self._set_requires_grad(self.base_ce_head, True)

    def set_base_mode(self, *, train_backbone: bool = True, train_projection: bool = True) -> None:
        self.current_phase = 0
        self.old_class_count = 0
        self.old_classes = []
        self.base_mode_active = True
        self.incremental_mode_active = False
        super().train(True)
        self._set_requires_grad(self.backbone, bool(train_backbone))
        self._set_requires_grad(self.projection, bool(train_projection))
        self._set_requires_grad(self.norm, bool(train_projection))
        self.freeze_classifier()
        self.freeze_semantic_encoder()
        if self.base_ce_head is not None:
            self.unfreeze_base_ce_head()

    def set_incremental_mode(
        self,
        *,
        phase: Optional[int] = None,
        old_class_count: Optional[int] = None,
        old_classes: Optional[Iterable[int]] = None,
        train_classifier_calibration: bool = False,
        train_geometry_adapter: Optional[bool] = None,
        **_: Any,
    ) -> None:
        if bool(train_classifier_calibration) or bool(train_geometry_adapter):
            raise RuntimeError("Classifier calibration and feature adapters are forbidden in SCTGR-RGA.")
        if phase is not None:
            self.current_phase = int(phase)
        if old_classes is not None:
            ids = _ordered_unique_ints(old_classes)
        else:
            count = int(self.old_class_count if old_class_count is None else old_class_count)
            ids = list(range(max(count, 0)))
        self.old_classes = ids
        self.old_class_count = len(ids)
        self.base_mode_active = False
        self.incremental_mode_active = True
        self.freeze_backbone_except_allowed(allow_last_blocks=False, allow_projection=False)
        self.freeze_classifier()
        self.freeze_semantic_encoder()
        self.freeze_base_ce_head()
        self.freeze_old_geometry_states(ids)
        self.backbone.eval()
        self.projection.eval()
        self.norm.eval()
        self._incremental_frozen_modules = ["backbone", "projection", "norm", "classifier"]

    def assert_frozen_modules(self) -> None:
        bad_req: List[str] = []
        bad_grad: List[str] = []
        for prefix, module in {
            "backbone": self.backbone,
            "projection": self.projection,
            "norm": self.norm,
            "classifier": self.classifier,
        }.items():
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    bad_req.append(f"{prefix}.{name}")
                if parameter.grad is not None and float(parameter.grad.detach().abs().sum().cpu().item()) != 0:
                    bad_grad.append(f"{prefix}.{name}")
        if bad_req:
            raise RuntimeError(f"Frozen modules still have trainable parameters: {bad_req[:20]}")
        if bad_grad:
            raise RuntimeError(f"Frozen modules have nonzero gradients: {bad_grad[:20]}")

    # Compatibility methods for removed branches.
    def freeze_incremental_adapter(self) -> None: return
    def freeze_geometry_plastic_adapter(self) -> None: return
    def disable_incremental_adapter(self) -> None:
        self.use_incremental_adapter = False
        self.use_geometry_gated_adapter = False
    def enable_incremental_adapter(self) -> None:
        raise RuntimeError("Feature adapters are removed from the main architecture.")
    def unfreeze_incremental_adapter(self) -> None:
        raise RuntimeError("Feature adapters are removed from the main architecture.")
    def unfreeze_geometry_plastic_adapter(self) -> None:
        raise RuntimeError("Feature adapters are removed from the main architecture.")
    def freeze_geometry_calibrator(self) -> None: self.use_geometry_calibrator = False
    def unfreeze_geometry_calibrator(self) -> None: raise RuntimeError("Geometry calibration is disabled.")
    def freeze_energy_calibrator(self) -> None:
        if hasattr(self.classifier, "freeze_all_adaptation"):
            self.classifier.freeze_all_adaptation()
    def unfreeze_energy_calibrator(self) -> None: raise RuntimeError("Energy calibration is disabled.")
    def adaptive_boundary_parameters(self) -> List[nn.Parameter]: return []
    def ensure_adaptive_boundary_capacity(self, class_count: int) -> None: del class_count
    def adaptive_boundary_state(self, old_class_count: int = 0) -> Dict[str, float]:
        return {"old_class_count": float(old_class_count), "adaptive_boundary_available": 0.0}

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.incremental_mode_active:
            self.backbone.eval()
            self.projection.eval()
            self.norm.eval()
            self.classifier.eval()
        return self

    # ------------------------------------------------------------------
    # Base CE head
    # ------------------------------------------------------------------
    def ensure_base_ce_head(self, num_base_classes: int, feature_dim: Optional[int] = None) -> nn.Linear:
        n = int(num_base_classes)
        d = int(feature_dim or self.d_model)
        if d != self.d_model:
            raise RuntimeError(f"base CE feature_dim must equal d_model={self.d_model}, got {d}")
        if self.base_ce_head is None or self.base_ce_num_classes != n:
            self.base_ce_head = nn.Linear(self.d_model, n).to(self.device)
            nn.init.normal_(self.base_ce_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.base_ce_head.bias)
            self.base_ce_num_classes = n
        return self.base_ce_head

    def base_ce_logits(self, features: torch.Tensor, num_base_classes: Optional[int] = None) -> torch.Tensor:
        z = self._validate_feature_tensor(features, "base_ce_logits.features")
        if self.base_ce_head is None:
            if num_base_classes is None:
                raise RuntimeError("base_ce_head is not initialized.")
            self.ensure_base_ce_head(int(num_base_classes))
        return self.base_ce_head(z)

    def forward_base_ce(self, x: torch.Tensor, num_base_classes: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        out = self.forward_features(
            x,
            spectral_summary=kwargs.get("spectral_summary"),
            band_weights=kwargs.get("band_weights"),
            spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical"),
        )
        logits = self.base_ce_logits(out["features"], num_base_classes)
        result = dict(out)
        result["base_logits"] = logits
        result["logits"] = logits
        return result

    def drop_base_ce_head(self) -> None:
        self.base_ce_head = None
        self.base_ce_num_classes = 0

    ensure_base_prl_head = ensure_base_ce_head
    base_prl_logits = base_ce_logits
    drop_base_prl_head = drop_base_ce_head
    freeze_base_prl_head = freeze_base_ce_head
    unfreeze_base_prl_head = unfreeze_base_ce_head

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------
    @torch.no_grad()
    def assert_base_handoff_ready(
        self,
        base_class_ids: Iterable[int],
        *,
        freeze: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        ids = _ordered_unique_ints(base_class_ids)
        result = self.geometry_bank.assert_phase0_base_handoff_ready(
            ids,
            freeze=bool(freeze),
            strict=bool(strict),
        )
        self.old_classes = list(ids)
        self.old_class_count = len(ids)
        self.current_num_classes = max(self.current_num_classes, max(ids) + 1 if ids else 0)
        self.seen_classes = list(ids)
        return result

    def assert_phase_ready(
        self,
        seen_classes: Iterable[int],
        *,
        mode: str = "geometry",
        require_geometry: bool = True,
    ) -> None:
        seen = _ordered_unique_ints(seen_classes)
        if not seen:
            raise RuntimeError("seen_classes is empty.")
        if self.geometry_bank.d_model != self.d_model:
            raise RuntimeError("GeometryBank/model feature dimensions differ.")
        if _normalize_classifier_mode(mode) == "geometry" and require_geometry:
            self.geometry_bank.assert_bank_valid(seen_classes=seen, strict=True)
        if self.incremental_mode_active:
            bad_train = [name for name in ("backbone", "projection", "norm") if getattr(self, name).training]
            if bad_train:
                raise RuntimeError(f"Frozen incremental modules must be eval(): {bad_train}")
        self.classifier.expand_to_seen_classes(seen)

    def assert_no_missing_class_geometry(self, seen_classes: Iterable[int]) -> None:
        self.geometry_bank.assert_bank_valid(seen_classes=seen_classes, strict=True)

    def assert_method_identity(self) -> None:
        if self.incremental_update_mode != "spectral_coupled_geometry_replay":
            raise RuntimeError(f"Unexpected incremental_update_mode={self.incremental_update_mode!r}")
        forbidden = {
            "use_geometry_gated_adapter": self.use_geometry_gated_adapter,
            "use_incremental_adapter": self.use_incremental_adapter,
            "use_geometry_transport": self.use_geometry_transport,
            "use_sglat_transport": self.use_sglat_transport,
            "use_geometry_calibrator": self.use_geometry_calibrator,
            "use_energy_calibrator": self.use_energy_calibrator,
            "use_adaptive_boundary": self.use_adaptive_boundary,
        }
        active = [name for name, value in forbidden.items() if bool(value)]
        if active:
            raise RuntimeError(f"Forbidden architecture branches are active: {active}")
        if self.geometry_plastic_adapter is not None:
            raise RuntimeError("Feature adapter module must not exist in the SCTGR-RGA model.")
        self._assert_strict_energy_contract()

    def assert_pg_rga_contract(self, seen_classes: Iterable[int], *, phase: str = "base") -> None:
        self.assert_method_identity()
        self.assert_phase_ready(seen_classes, mode="geometry", require_geometry=True)
        if not str(phase).lower().startswith("base"):
            self.assert_frozen_modules()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        mode = _normalize_classifier_mode(kwargs.get("classifier_mode", kwargs.get("mode", "geometry")))
        if bool(kwargs.get("apply_adapter", False)):
            raise RuntimeError("Feature adapter routing is forbidden.")
        features_out = self.forward_features(
            x,
            spectral_summary=kwargs.get("spectral_summary"),
            band_weights=kwargs.get("band_weights"),
            spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical"),
            apply_adapter=False,
        )
        return_features_only = _to_bool(kwargs.get("return_features_only", False), False)
        if mode == "base_ce" or return_features_only:
            if mode == "base_ce" or "num_base_classes" in kwargs:
                nbase = int(kwargs.get("num_base_classes", self.base_ce_num_classes))
                if nbase <= 0:
                    raise RuntimeError("num_base_classes is required for base_ce mode.")
                features_out = dict(features_out)
                features_out["logits"] = self.base_ce_logits(features_out["features"], nbase)
            return features_out

        seen_classes = kwargs.get("seen_classes", None)
        if seen_classes is None:
            seen_classes = self._infer_seen_classes()
        logits_out = self.compute_logits_from_features(
            features_out["features"],
            seen_classes=seen_classes,
            geometry_bank=self.geometry_bank,
            mode="geometry",
            targets=kwargs.get("targets", kwargs.get("labels")),
            targets_are_global=_to_bool(kwargs.get("targets_are_global", kwargs.get("labels_are_global", False))),
            old_classes=kwargs.get("old_classes"),
            new_classes=kwargs.get("new_classes"),
            return_energy=_to_bool(kwargs.get("return_energy", False)),
            return_parts=_to_bool(kwargs.get("return_parts", False)),
            return_diagnostics=_to_bool(kwargs.get("return_diagnostics", False)),
        )
        out: Dict[str, Any] = dict(features_out)
        if isinstance(logits_out, dict):
            out.update(logits_out)
        else:
            out["logits"] = logits_out
        return out










# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import inspect
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.backbone import SSMBackbone
# from models.geometry_bank import GeometryBank
# from models.classifier import GeometryEnergyClassifier


# def _to_bool(value: Any, default: bool = False) -> bool:
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
#     return bool(value)


# def _filter_supported_kwargs(cls_or_fn: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
#     try:
#         sig = inspect.signature(cls_or_fn)
#     except (TypeError, ValueError):
#         return dict(kwargs)
#     if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
#         return dict(kwargs)
#     allowed = set(sig.parameters.keys())
#     allowed.discard("self")
#     return {k: v for k, v in kwargs.items() if k in allowed}


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     out: List[int] = []
#     seen = set()
#     for v in values:
#         iv = int(v)
#         if iv not in seen:
#             out.append(iv)
#             seen.add(iv)
#     return out


# def _normalize_classifier_mode(mode: Optional[str], default: str = "geometry") -> str:
#     """Normalize classifier mode for the clean PG-RGA model.

#     Public commands may use ``geometry_only`` while the classifier implementation
#     uses ``geometry`` internally.  Calibrated/topology aliases are accepted only
#     for backward compatibility and are routed to plain geometry, because the
#     main path must remain geometry-only.
#     """
#     m = str(default if mode is None else mode).lower().strip()
#     aliases = {
#         "": "geometry",
#         "none": "geometry",
#         "geo": "geometry",
#         "geometry": "geometry",
#         "geometry_only": "geometry",
#         "geometry-only": "geometry",
#         "feature_geometry": "geometry",
#         "low_rank_geometry": "geometry",
#         "anchor": "geometry",
#         "anchor_concept": "geometry",
#         "anchor_concept_geometry": "geometry",
#         "srgp": "geometry",
#         "srgp_geometry": "geometry",
#         "spectral_geometry": "geometry",
#         "spectral_residual": "geometry",
#         "calibrated": "geometry",
#         "calibrated_geometry": "geometry",
#         "topology_calibrated_geometry": "geometry",
#         "base_ce": "base_ce",
#     }
#     out = aliases.get(m, m)
#     if out not in {"geometry", "base_ce"}:
#         raise ValueError(f"Unsupported classifier mode {mode!r}. Use geometry_only/geometry or base_ce.")
#     return out



# def _normalize_incremental_update_mode(mode: Optional[str]) -> str:
#     """Normalize incremental update mode for the clean PG-RGA architecture.

#     The main method has exactly one model-plasticity path: the bounded
#     geometry-gated residual adapter.  Legacy descriptor/frozen names are
#     accepted only so old commands do not route to a different implementation.
#     """
#     m = str(mode or "geometry_gated_adapter").lower().strip()
#     aliases = {
#         "": "geometry_gated_adapter",
#         "none": "geometry_gated_adapter",
#         "false": "geometry_gated_adapter",
#         "off": "geometry_gated_adapter",
#         "clean": "geometry_gated_adapter",
#         "main": "geometry_gated_adapter",
#         "pg_rga": "geometry_gated_adapter",
#         "pg-rga": "geometry_gated_adapter",
#         "pgrga": "geometry_gated_adapter",
#         "geometry_gated_adapter": "geometry_gated_adapter",
#         "geometry_gated": "geometry_gated_adapter",
#         "gated_adapter": "geometry_gated_adapter",
#         "geometry_adapter": "geometry_gated_adapter",
#         "adapter": "geometry_gated_adapter",
#         "g2rpa": "geometry_gated_adapter",
#         "g2-rpa": "geometry_gated_adapter",
#         "descriptor": "geometry_gated_adapter",
#         "descriptor_only": "geometry_gated_adapter",
#         "descriptor_refinement": "geometry_gated_adapter",
#         "frozen": "geometry_gated_adapter",
#         "frozen_geometry": "geometry_gated_adapter",
#         "scbgr": "geometry_gated_adapter",
#         "scb-gr": "geometry_gated_adapter",
#         "rsgi": "geometry_gated_adapter",
#     }
#     out = aliases.get(m, m)
#     if out != "geometry_gated_adapter":
#         raise ValueError(
#             f"Unsupported incremental_update_mode={mode!r}. "
#             "Use geometry_gated_adapter for the PG-RGA main path."
#         )
#     return out


# def _module_has_trainable_params(module: Optional[nn.Module]) -> bool:
#     return bool(module is not None and any(p.requires_grad for p in module.parameters()))


# class GeometryPlasticAdapter(nn.Module):
#     """Bounded geometry-gated residual adapter for controlled incremental plasticity.

#     This module operates directly in the canonical projected GeometryBank space.
#     It is deliberately small: it can make a bounded residual correction for new
#     classes, but it cannot rewrite the backbone/projection or old bank rows.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         *,
#         bottleneck: int = 32,
#         max_scale: float = 0.10,
#         dropout: float = 0.0,
#         gate_bias_init: float = -3.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.bottleneck = int(max(1, bottleneck))
#         self.max_scale = float(max(0.0, max_scale))
#         self.norm = nn.LayerNorm(self.d_model)
#         self.delta = nn.Sequential(
#             nn.Linear(self.d_model, self.bottleneck),
#             nn.GELU(),
#             nn.Dropout(float(max(0.0, dropout))),
#             nn.Linear(self.bottleneck, self.d_model),
#         )
#         self.gate = nn.Sequential(
#             nn.Linear(self.d_model, self.bottleneck),
#             nn.GELU(),
#             nn.Linear(self.bottleneck, 1),
#         )
#         self.reset_parameters(float(gate_bias_init))

#     def reset_parameters(self, gate_bias_init: float = -3.0) -> None:
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.zeros_(module.bias)
#         # Start as an identity map. The adapter earns plasticity only through
#         # incremental new/replay losses.
#         last_delta = self.delta[-1]
#         if isinstance(last_delta, nn.Linear):
#             nn.init.zeros_(last_delta.weight)
#             nn.init.zeros_(last_delta.bias)
#         last_gate = self.gate[-1]
#         if isinstance(last_gate, nn.Linear):
#             nn.init.zeros_(last_gate.weight)
#             nn.init.constant_(last_gate.bias, float(gate_bias_init))

#     def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
#         if features.dim() != 2:
#             raise RuntimeError(f"GeometryPlasticAdapter expects [B,D], got {tuple(features.shape)}")
#         h = self.norm(features)
#         gate = torch.sigmoid(self.gate(h))
#         direction = torch.tanh(self.delta(h))
#         delta = gate * float(self.max_scale) * direction
#         adapted = features + delta
#         return {
#             "features": adapted,
#             "projected_features": adapted,
#             "delta": delta,
#             "gate": gate,
#             "adapter_delta": delta,
#             "adapter_gate": gate,
#         }


# class NECILModel(nn.Module):
#     """Clean NECIL-HSI model router.

#     Contract:
#         - One canonical projected feature space ``z`` is used for base geometry,
#           incremental geometry insertion, synthetic replay, and evaluation.
#         - GeometryBank is the only non-exemplar memory.
#         - Classifier always receives explicit ``seen_classes`` and returns
#           logits [B, len(seen_classes)].
#         - Semantic/concept/adaptor/transport paths are not part of this clean
#           main model. Their hooks are kept as no-ops or hard-disabled aliases so
#           stale trainer code does not silently route through a different space.
#     """

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.args = args
#         self.device = torch.device(getattr(args, "device", "cpu"))
#         self.d_model = int(getattr(args, "d_model", 128))
#         self.subspace_rank = int(getattr(args, "subspace_rank", getattr(args, "rank", 5)))
#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive.")
#         if self.subspace_rank < 0:
#             raise ValueError("subspace_rank must be non-negative.")

#         self.current_phase = 0
#         self.old_class_count = 0
#         self.current_num_classes = 0
#         self.seen_classes: List[int] = []
#         self.base_mode_active = True
#         self.incremental_mode_active = False
#         self._incremental_frozen_modules: List[str] = []

#         # Main architecture switch.
#         # PG-RGA uses a bounded residual geometry adapter during incremental
#         # learning, while the backbone/projection and old GeometryBank rows stay
#         # frozen. Descriptor-only/frozen modes remain available as ablations.
#         self.incremental_update_mode = _normalize_incremental_update_mode(
#             getattr(args, "incremental_update_mode", None)
#         )

#         # Hard-disable stale/unsafe paths in the model object. They can exist in
#         # other files for ablation compatibility, but this clean model must not
#         # silently route through them.
#         self.use_incremental_adapter = False
#         self.use_geometry_calibrator = False
#         self.use_bicyc_geometry_cycle = False
#         self.use_geometry_transport = False
#         self.use_sglat_transport = False
#         self.use_geometry_gated_adapter = (
#             self.incremental_update_mode == "geometry_gated_adapter"
#             or _to_bool(getattr(args, "use_geometry_gated_adapter", False), False)
#         )
#         self.semantic_encoder: Optional[nn.Module] = None
#         self.concept_encoder: Optional[nn.Module] = None

#         self.backbone = SSMBackbone(args)
#         dropout = float(getattr(args, "projection_dropout", 0.0))
#         self.projection = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#         self.normalize_geometry_features = _to_bool(getattr(args, "normalize_geometry_features", True), True)
#         raw_scale = float(getattr(args, "geometry_feature_scale", 0.0) or 0.0)
#         self.geometry_feature_scale = raw_scale if raw_scale > 0.0 else math.sqrt(float(self.d_model))
#         self.geometry_feature_clamp = float(getattr(args, "geometry_feature_clamp", 0.0) or 0.0)
#         self.spectral_summary_mode = str(getattr(args, "spectral_summary_mode", "center")).lower().strip()
#         if self.spectral_summary_mode not in {"center", "mean"}:
#             raise ValueError("spectral_summary_mode must be 'center' or 'mean'.")
#         pca_components = int(getattr(args, "pca_components", 0) or 0)
#         self.default_spectral_physical = bool(getattr(args, "spectral_summary_is_physical", pca_components <= 0))
#         self.min_band_mass = float(getattr(args, "min_band_mass", 1e-8))

#         bank_kwargs = _filter_supported_kwargs(
#             GeometryBank.__init__,
#             {
#                 "device": self.device,
#                 "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
#                 "variance_shrinkage": float(getattr(args, "geometry_variance_shrinkage", 0.10)),
#                 "max_variance_ratio": float(getattr(args, "geometry_max_variance_ratio", 50.0)),
#                 "min_reliability": float(getattr(args, "geometry_min_reliability", 0.05)),
#                 "reliability_sample_alpha": float(getattr(args, "reliability_sample_alpha", 20.0)),
#                 "rank_energy_threshold": float(getattr(args, "rank_energy_threshold", 0.95)),
#                 "rank_eigen_ratio_threshold": float(getattr(args, "rank_eigen_ratio_threshold", 1e-3)),
#                 "min_active_rank": int(getattr(args, "min_active_rank", 1)),
#             },
#         )
#         self.geometry_bank = GeometryBank(self.d_model, self.subspace_rank, **bank_kwargs)

#         clf_kwargs = _filter_supported_kwargs(
#             GeometryEnergyClassifier.__init__,
#             {
#                 "initial_classes": 0,
#                 "d_model": self.d_model,
#                 "logit_scale": float(getattr(args, "loss_scale", getattr(args, "logit_scale", 8.0))),
#                 "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
#                 "residual_variance_scale": float(getattr(args, "residual_variance_scale", 0.75)),
#                 "normalize_energy_by_dim": _to_bool(getattr(args, "energy_normalize_by_dim", True), True),
#                 "use_logdet_energy": _to_bool(getattr(args, "use_logdet_energy", True), True),
#                 "logdet_energy_weight": float(getattr(args, "logdet_energy_weight", 0.05)),
#                 "use_reliability_penalty": _to_bool(getattr(args, "use_reliability_penalty", True), True),
#                 "reliability_energy_weight": float(getattr(args, "reliability_energy_weight", 0.03)),
#                 "use_old_new_calibration": False,
#                 "calibration_max_abs_bias": float(getattr(args, "calibration_max_abs_bias", getattr(args, "energy_calibrator_max_bias", 1.0))),
#                 "logit_clip": float(getattr(args, "geometry_logit_clip", 0.0)),
#             },
#         )
#         self.classifier = GeometryEnergyClassifier(**clf_kwargs)

#         self.geometry_plastic_adapter = GeometryPlasticAdapter(
#             self.d_model,
#             bottleneck=int(getattr(args, "adapter_bottleneck", 32)),
#             max_scale=float(getattr(args, "adapter_max_scale", 0.10)),
#             dropout=float(getattr(args, "adapter_dropout", 0.0)),
#             gate_bias_init=float(getattr(args, "adapter_gate_bias_init", -3.0)),
#         )
#         self._set_requires_grad(self.geometry_plastic_adapter, False)

#         self.base_ce_head: Optional[nn.Linear] = None
#         self.base_ce_num_classes = 0

#         self.to(self.device)

#     # ------------------------------------------------------------------
#     # Input / feature utilities
#     # ------------------------------------------------------------------
#     def _validate_feature_tensor(self, features: torch.Tensor, name: str, batch_size: Optional[int] = None) -> torch.Tensor:
#         if not torch.is_tensor(features):
#             raise TypeError(f"{name} must be a torch.Tensor, got {type(features)}")
#         if features.dim() != 2:
#             raise RuntimeError(f"{name} must be [B,D], got {tuple(features.shape)}")
#         if features.size(1) != self.d_model:
#             raise RuntimeError(f"{name} dim mismatch: expected {self.d_model}, got {features.size(1)}")
#         if batch_size is not None and features.size(0) != int(batch_size):
#             raise RuntimeError(f"{name} batch mismatch: got {features.size(0)}, expected {int(batch_size)}")
#         if not torch.isfinite(features).all():
#             raise RuntimeError(f"{name} contains NaN/Inf values.")
#         return features

#     def _canonicalize(self, z: torch.Tensor, *, name: str) -> torch.Tensor:
#         z = self._validate_feature_tensor(z, name)
#         if self.normalize_geometry_features:
#             z = F.normalize(z, p=2, dim=1, eps=1e-8) * float(self.geometry_feature_scale)
#         if self.geometry_feature_clamp > 0.0:
#             z = z.clamp(-self.geometry_feature_clamp, self.geometry_feature_clamp)
#         if not torch.isfinite(z).all():
#             raise RuntimeError(f"{name} contains NaN/Inf after canonicalization.")
#         return z

#     def _center_spectrum_from_input(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 4:
#             if self.spectral_summary_mode == "center":
#                 return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
#             return x.mean(dim=(-1, -2))
#         if x.dim() == 3:
#             if self.spectral_summary_mode == "center":
#                 return x[:, :, x.size(-1) // 2]
#             return x.mean(dim=-1)
#         if x.dim() == 2:
#             return x
#         raise RuntimeError(f"Unsupported input shape for spectral summary: {tuple(x.shape)}")

#     def _prepare_spectral_summary(
#         self,
#         x: torch.Tensor,
#         features: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#     ) -> Tuple[torch.Tensor, bool]:
#         if spectral_summary is None or not torch.is_tensor(spectral_summary) or spectral_summary.numel() == 0:
#             s = self._center_spectrum_from_input(x).to(device=features.device, dtype=features.dtype)
#             physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
#         else:
#             s = spectral_summary.to(device=features.device, dtype=features.dtype)
#             if s.dim() == 4:
#                 s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#             elif s.dim() == 3:
#                 if s.size(0) == features.size(0) and s.size(-1) > 1:
#                     s = s[:, :, s.size(-1) // 2]
#                 else:
#                     s = s.reshape(features.size(0), -1)
#             elif s.dim() == 1:
#                 if s.numel() % max(int(features.size(0)), 1) != 0:
#                     raise RuntimeError(f"1-D spectral_summary cannot be reshaped to batch size {features.size(0)}")
#                 s = s.reshape(features.size(0), -1)
#             elif s.dim() > 4:
#                 s = s.flatten(1)
#             physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
#         if s.dim() != 2 or s.size(0) != features.size(0):
#             raise RuntimeError(f"spectral_summary must resolve to [B,S], got {tuple(s.shape)}")
#         s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#         if s.size(1) == 0:
#             physical = False
#         return s, bool(physical)

#     def _band_summary(self, spectral_summary: torch.Tensor, band_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if band_weights is not None and torch.is_tensor(band_weights) and band_weights.numel() > 0:
#             bw = band_weights.to(device=spectral_summary.device, dtype=spectral_summary.dtype)
#             if bw.dim() == 2 and bw.size(0) == spectral_summary.size(0) and bw.size(1) == spectral_summary.size(1):
#                 bw = torch.nan_to_num(bw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
#                 denom = bw.sum(dim=1, keepdim=True)
#                 if bool((denom > self.min_band_mass).all().item()):
#                     return bw / denom.clamp_min(self.min_band_mass)
#         b = spectral_summary.abs()
#         denom = b.sum(dim=1, keepdim=True)
#         uniform = torch.full_like(b, 1.0 / float(max(int(b.size(1)), 1)))
#         b = torch.where(denom > self.min_band_mass, b / denom.clamp_min(self.min_band_mass), uniform)
#         return b

#     def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """Backbone-only feature extraction. Does not touch GeometryBank/classifier."""
#         if not torch.is_tensor(x):
#             raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
#         out = self.backbone(x.to(self.device))
#         if isinstance(out, dict):
#             if "features" not in out:
#                 raise RuntimeError("backbone output dict must contain key 'features'.")
#             features = self._validate_feature_tensor(out["features"], "backbone.features", int(x.size(0)))
#             result = dict(out)
#             result["features"] = features
#             result.setdefault("backbone_features", features)
#             return result
#         features = self._validate_feature_tensor(out, "backbone.features", int(x.size(0)))
#         return {"features": features, "backbone_features": features}

#     def forward_features(
#         self,
#         x: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#         apply_adapter: Optional[bool] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Return projected geometry features.

#         Important contract:
#             - ``canonical_features`` are always the backbone/projection z-space
#               used to build the base GeometryBank.
#             - ``features`` / ``projected_features`` are the scoring features.
#               They equal canonical z in base mode and become adapted z only when
#               PG-RGA adapter routing is explicitly active.
#         """
#         raw = self.extract_features(x)
#         h = self._validate_feature_tensor(raw["features"], "preproject_features", int(x.size(0)))
#         z_canonical = self.norm(self.projection(h) + h)
#         z_canonical = self._canonicalize(z_canonical, name="canonical_geometry_features")

#         use_adapter = False if apply_adapter is None else bool(apply_adapter)
#         z = z_canonical
#         adapter_out: Optional[Dict[str, torch.Tensor]] = None
#         if use_adapter:
#             adapter_out = self.adapt_projected_features(
#                 z_canonical,
#                 force=(apply_adapter is True),
#                 return_delta=True,
#                 recanonicalize=True,
#             )
#             z = adapter_out["features"]

#         s, physical = self._prepare_spectral_summary(
#             x.to(self.device),
#             z,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#         )
#         candidate_bw = band_weights if band_weights is not None else raw.get("band_weights", None)
#         band = self._band_summary(s, candidate_bw)
#         result = {
#             "features": z,
#             "projected_features": z,
#             "geometry_features": z,
#             "canonical_features": z_canonical,
#             "canonical_projected_features": z_canonical,
#             "pre_adapter_features": z_canonical,
#             "preproject_features": h,
#             "backbone_features": h,
#             "spectral_summary": s,
#             "spectral_summary_is_physical": torch.tensor(bool(physical), device=z.device, dtype=torch.bool),
#             "band_summary": band,
#             "band_importance": band,
#             "band_weights": candidate_bw if torch.is_tensor(candidate_bw) else None,
#             "spectral_features": raw.get("spectral_features", h),
#             "spatial_features": raw.get("spatial_features", h),
#         }
#         if adapter_out is not None:
#             result["adapter_delta"] = adapter_out["adapter_delta"]
#             result["adapter_gate"] = adapter_out["adapter_gate"]
#             result["adapter_active"] = adapter_out["adapter_active"]
#         return result

#     # Compatibility aliases expected by existing trainers.
#     def extract_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         # Projected features are canonical by contract.  Incremental scoring uses
#         # the adapter only through forward(...)/extract_adapted_projected_features.
#         kwargs = dict(kwargs)
#         kwargs["apply_adapter"] = False
#         return self.forward_features(x, **kwargs)

#     def extract_canonical_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         kwargs = dict(kwargs)
#         kwargs["apply_adapter"] = False
#         return self.forward_features(x, **kwargs)

#     def extract_adapted_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         kwargs = dict(kwargs)
#         kwargs["apply_adapter"] = True
#         return self.forward_features(x, **kwargs)

#     def extract_geometry_features(self, x: torch.Tensor, *, return_dict: bool = False, space: str = "canonical", **kwargs: Any):
#         space_norm = str(space or "canonical").lower().strip()
#         if space_norm in {"canonical", "pre_adapter", "base"}:
#             out = self.extract_canonical_projected_features(x, **kwargs)
#         elif space_norm in {"scoring", "adapted", "post_adapter"}:
#             out = self.extract_adapted_projected_features(x, **kwargs)
#         else:
#             raise RuntimeError(f"Unsupported geometry feature space {space!r}; use canonical or scoring.")
#         out["geometry_feature_space"] = "scoring" if space_norm in {"scoring", "adapted", "post_adapter"} else "canonical"
#         return out if bool(return_dict) else out["features"]

#     @torch.no_grad()
#     def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         return self.forward_features(x)

#     # ------------------------------------------------------------------
#     # Controlled geometry-gated adapter
#     # ------------------------------------------------------------------
#     def _adapter_runtime_enabled(self, *, force: bool = False) -> bool:
#         return bool(
#             getattr(self, "geometry_plastic_adapter", None) is not None
#             and (bool(force) or (bool(getattr(self, "use_geometry_gated_adapter", False)) and bool(self.incremental_mode_active)))
#         )

#     def adapt_projected_features(
#         self,
#         features: torch.Tensor,
#         *,
#         force: bool = False,
#         return_delta: bool = True,
#         recanonicalize: bool = True,
#         **_: Any,
#     ) -> Dict[str, torch.Tensor]:
#         """Apply the bounded adapter directly to canonical z-space features.

#         This is required for synthetic old replay: replay samples already live
#         in GeometryBank space and cannot pass through the backbone. The adapter
#         must still see them so replay CE can close old-region gates.
#         """
#         z0 = self._validate_feature_tensor(features, "adapt_projected_features.features")
#         if not self._adapter_runtime_enabled(force=force):
#             zero_delta = torch.zeros_like(z0)
#             zero_gate = torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype)
#             return {
#                 "features": z0,
#                 "projected_features": z0,
#                 "geometry_features": z0,
#                 "delta": zero_delta,
#                 "gate": zero_gate,
#                 "adapter_delta": zero_delta,
#                 "adapter_gate": zero_gate,
#                 "adapter_active": torch.tensor(False, device=z0.device),
#             }
#         raw = self.geometry_plastic_adapter(z0)
#         z = raw["features"]
#         if bool(recanonicalize):
#             z = self._canonicalize(z, name="adapted_geometry_features")
#         delta = z - z0
#         gate = raw.get("gate", torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype))
#         return {
#             "features": z,
#             "projected_features": z,
#             "geometry_features": z,
#             "delta": delta,
#             "gate": gate,
#             "adapter_delta": delta,
#             "adapter_gate": gate,
#             "adapter_active": torch.tensor(True, device=z0.device),
#         }

#     # ------------------------------------------------------------------
#     # Seen classes / bank helpers
#     # ------------------------------------------------------------------
#     def _infer_seen_classes(self, geometry_bank: Optional[Any] = None) -> List[int]:
#         bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
#         if hasattr(bank_obj, "get_valid_mask"):
#             valid = bank_obj.get_valid_mask().detach().cpu().flatten()
#             return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
#         bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
#         if not isinstance(bank, dict) or "sample_counts" not in bank:
#             return list(self.seen_classes)
#         counts = bank["sample_counts"].detach().cpu().flatten()
#         return [int(i) for i in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()]

#     def ensure_class_capacity(self, class_count: int, spectral_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
#         count = int(max(0, class_count))
#         dtype = dtype or next(self.parameters()).dtype
#         if hasattr(self.geometry_bank, "ensure_class_count"):
#             self.geometry_bank.ensure_class_count(count, spectral_dim=int(spectral_dim), dtype=dtype)
#         elif hasattr(self.geometry_bank, "ensure_num_classes"):
#             self.geometry_bank.ensure_num_classes(count)
#         self.current_num_classes = max(self.current_num_classes, count)

#     def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.geometry_bank, "get_bank"):
#             raise RuntimeError("GeometryBank must expose get_bank().")
#         bank = self.geometry_bank.get_bank()
#         if "variances" not in bank and "eigvals" in bank and "res_vars" in bank:
#             bank = dict(bank)
#             bank["variances"] = torch.cat([bank["eigvals"], bank["res_vars"].unsqueeze(-1)], dim=-1)
#         if "resvars" not in bank and "res_vars" in bank:
#             bank = dict(bank)
#             bank["resvars"] = bank["res_vars"]
#         return bank

#     def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()
#         old = max(0, min(old, int(bank["means"].size(0))))
#         out = {}
#         for k, v in bank.items():
#             if torch.is_tensor(v) and v.dim() > 0 and v.size(0) >= old:
#                 out[k] = v[:old]
#         return out

#     # ------------------------------------------------------------------
#     # Classifier routing
#     # ------------------------------------------------------------------
#     def forward_classifier(
#         self,
#         features: torch.Tensor,
#         seen_classes: Iterable[int],
#         mode: str = "geometry",
#         *,
#         geometry_bank: Optional[Any] = None,
#         targets: Optional[torch.Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#     ):
#         return self.compute_logits_from_features(
#             features=features,
#             seen_classes=seen_classes,
#             geometry_bank=geometry_bank,
#             mode=mode,
#             targets=targets,
#             targets_are_global=targets_are_global,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )

#     def compute_logits_from_features(
#         self,
#         features: torch.Tensor,
#         seen_classes: Optional[Iterable[int]] = None,
#         geometry_bank: Optional[Any] = None,
#         mode: str = "geometry",
#         *,
#         classifier_mode: Optional[str] = None,
#         targets: Optional[torch.Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#         **_: Any,
#     ):
#         features = self._validate_feature_tensor(features, "compute_logits_from_features.features")
#         bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
#         seen = _ordered_unique_ints(seen_classes if seen_classes is not None else self._infer_seen_classes(bank_obj))
#         if not seen:
#             raise RuntimeError("seen_classes is empty. Build GeometryBank rows or pass seen_classes explicitly.")
#         mode_norm = _normalize_classifier_mode(classifier_mode if classifier_mode is not None else mode, "geometry")
#         self.assert_phase_ready(seen, mode=mode_norm, require_geometry=True)
#         out = self.classifier(
#             features,
#             seen_classes=seen,
#             geometry_bank=bank_obj,
#             mode=mode_norm,
#             targets=targets,
#             targets_are_global=bool(targets_are_global),
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=int(self.old_class_count),
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )
#         logits = out["logits"] if isinstance(out, dict) else out
#         self.classifier.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets=(self.classifier.global_to_local_labels(targets, seen) if targets is not None and targets_are_global else targets),
#             old_classes=old_classes,
#             new_classes=new_classes,
#             context="NECILModel.compute_logits_from_features",
#         )
#         self.current_num_classes = max(self.current_num_classes, max(seen) + 1)
#         self.seen_classes = list(seen)
#         return out

#     def compute_energy_from_features(self, features: torch.Tensor, seen_classes: Optional[Iterable[int]] = None, **kwargs: Any):
#         out = self.compute_logits_from_features(features, seen_classes=seen_classes, return_energy=True, **kwargs)
#         if not isinstance(out, dict) or "energy" not in out:
#             raise RuntimeError("Expected dict with energy from compute_logits_from_features(return_energy=True).")
#         return out["energy"]

#     # ------------------------------------------------------------------
#     # Geometry refresh / memory snapshots
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def refresh_geometry_for_classes(
#         self,
#         class_ids: Iterable[int],
#         features: torch.Tensor,
#         labels: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#         phase_created: Optional[int] = None,
#         freeze_after: bool = False,
#     ) -> Dict[str, Any]:
#         features = self._validate_feature_tensor(features, "refresh_geometry_for_classes.features")
#         labels = labels.to(device=features.device).long().flatten()
#         ids = _ordered_unique_ints(class_ids)
#         if labels.numel() != features.size(0):
#             raise RuntimeError("labels/features batch mismatch in refresh_geometry_for_classes")
#         bad = sorted(set(int(v) for v in torch.unique(labels).detach().cpu().tolist()).difference(set(ids)))
#         if bad:
#             raise RuntimeError(f"labels contain classes outside requested refresh ids: bad={bad}, class_ids={ids}")
#         band_dim = 0
#         if spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0:
#             band_dim = int(spectral_summary.reshape(features.size(0), -1).size(1))
#         elif band_weights is not None and torch.is_tensor(band_weights) and band_weights.numel() > 0:
#             band_dim = int(band_weights.reshape(features.size(0), -1).size(1))
#         self.ensure_class_capacity(max(ids) + 1 if ids else 0, spectral_dim=band_dim, dtype=features.dtype)
#         rows = self.geometry_bank.extract_geometry(
#             features,
#             labels,
#             spectral_summary=spectral_summary,
#             band_weights=band_weights,
#             spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         )
#         committed: List[int] = []
#         for c in ids:
#             if c not in rows:
#                 raise RuntimeError(f"No geometry could be extracted for class {c}.")
#             row = rows[c]
#             self.geometry_bank.add_or_update_class_geometry(
#                 c,
#                 mean=row["mean"],
#                 basis=row["basis"],
#                 eigvals=row["eigvals"],
#                 res_var=row["res_var"],
#                 spectral_prototype=row.get("spectral_prototype"),
#                 band_importance=row.get("band_importance"),
#                 sample_count=row.get("sample_count"),
#                 active_rank=row.get("active_rank"),
#                 reliability=row.get("reliability"),
#                 feature_reliability=row.get("feature_reliability"),
#                 band_reliability=row.get("band_reliability"),
#                 spectral_reliability=row.get("spectral_reliability"),
#                 phase_created=int(self.current_phase if phase_created is None else phase_created),
#                 allow_frozen_update=False,
#             )
#             committed.append(c)
#         if bool(freeze_after):
#             self.freeze_classes(committed)
#         self.geometry_bank.assert_bank_valid(seen_classes=committed, strict=True)
#         self.current_num_classes = max(self.current_num_classes, max(committed) + 1 if committed else self.current_num_classes)
#         return {"committed_class_ids": committed, "phase_created": int(self.current_phase if phase_created is None else phase_created)}

#     @torch.no_grad()
#     def refresh_class_subspace(self, cls: int, mean: torch.Tensor, basis: torch.Tensor, eigvals: torch.Tensor, res_var: Optional[torch.Tensor] = None, resvar: Optional[torch.Tensor] = None, **kwargs: Any) -> None:
#         rv = res_var if res_var is not None else resvar
#         if rv is None:
#             raise ValueError("refresh_class_subspace requires res_var/resvar.")
#         c = int(cls)
#         self.ensure_class_capacity(c + 1)
#         self.geometry_bank.add_or_update_class_geometry(
#             c,
#             mean=mean,
#             basis=basis,
#             eigvals=eigvals,
#             res_var=rv,
#             spectral_prototype=kwargs.get("spectral_prototype", kwargs.get("spectral_proto", None)),
#             band_importance=kwargs.get("band_importance", None),
#             sample_count=kwargs.get("sample_count", None),
#             active_rank=kwargs.get("active_rank", None),
#             reliability=kwargs.get("reliability", None),
#             feature_reliability=kwargs.get("feature_reliability", None),
#             band_reliability=kwargs.get("band_reliability", None),
#             phase_created=kwargs.get("phase_created", self.current_phase),
#             allow_frozen_update=bool(kwargs.get("allow_frozen_update", False)),
#         )

#     @torch.no_grad()
#     def export_memory_snapshot(self) -> Dict[str, Any]:
#         snap = self.geometry_bank.export_snapshot()
#         snap.update(
#             {
#                 "current_phase": int(self.current_phase),
#                 "old_class_count": int(self.old_class_count),
#                 "current_num_classes": int(self.current_num_classes),
#                 "seen_classes": list(self.seen_classes),
#                 "feature_contract": self.feature_contract(),
#             }
#         )
#         return snap

#     @torch.no_grad()
#     def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True) -> None:
#         if not snapshot:
#             if strict:
#                 raise ValueError("empty memory snapshot")
#             return
#         self._assert_snapshot_feature_contract(snapshot, strict=strict)
#         self.geometry_bank.load_snapshot(snapshot, strict=strict)
#         self.current_phase = int(snapshot.get("current_phase", self.current_phase))
#         self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
#         self.current_num_classes = int(snapshot.get("current_num_classes", len(self.geometry_bank)))
#         self.seen_classes = [int(c) for c in snapshot.get("seen_classes", self._infer_seen_classes())]

#     def feature_contract(self) -> Dict[str, Any]:
#         return {
#             "d_model": int(self.d_model),
#             "subspace_rank": int(self.subspace_rank),
#             "normalize_geometry_features": bool(self.normalize_geometry_features),
#             "geometry_feature_scale": float(self.geometry_feature_scale),
#             "spectral_summary_mode": str(self.spectral_summary_mode),
#             "classifier_contract": "logits[B,len(seen_classes)]",
#             "incremental_update_mode": str(self.incremental_update_mode),
#             "geometry_gated_adapter_available": hasattr(self, "geometry_plastic_adapter"),
#             "geometry_gated_adapter_enabled": bool(self.use_geometry_gated_adapter),
#             "semantic_encoder_enabled": False,
#             "concept_encoder_enabled": False,
#         }

#     def _assert_snapshot_feature_contract(self, snapshot: Dict[str, Any], *, strict: bool) -> None:
#         if not strict:
#             return
#         old = snapshot.get("feature_contract", snapshot.get("geometry_feature_contract", None))
#         if not isinstance(old, dict):
#             return
#         cur = self.feature_contract()
#         mismatches = []
#         for k in ("d_model", "subspace_rank", "normalize_geometry_features", "spectral_summary_mode"):
#             if k in old and old[k] != cur[k]:
#                 mismatches.append(f"{k}: snapshot={old[k]!r}, current={cur[k]!r}")
#         if "geometry_feature_scale" in old and abs(float(old["geometry_feature_scale"]) - float(cur["geometry_feature_scale"])) > 1e-6:
#             mismatches.append(f"geometry_feature_scale: snapshot={old['geometry_feature_scale']!r}, current={cur['geometry_feature_scale']!r}")
#         if mismatches:
#             raise RuntimeError("Memory snapshot was built under a different feature contract: " + "; ".join(mismatches))

#     # ------------------------------------------------------------------
#     # Freezing / phase modes
#     # ------------------------------------------------------------------
#     def _set_requires_grad(self, module: Optional[nn.Module], value: bool) -> None:
#         if module is None:
#             return
#         for p in module.parameters():
#             p.requires_grad = bool(value)

#     def freeze_backbone_except_allowed(self, *, allow_last_blocks: bool = False, allow_projection: bool = False, allow_norm: Optional[bool] = None) -> None:
#         self._set_requires_grad(self.backbone, False)
#         if bool(allow_last_blocks) and hasattr(self.backbone, "get_last_blocks"):
#             for block in self.backbone.get_last_blocks():
#                 self._set_requires_grad(block, True)
#         self._set_requires_grad(self.projection, bool(allow_projection))
#         self._set_requires_grad(self.norm, bool(allow_projection if allow_norm is None else allow_norm))

#     def freeze_backbone_only(self) -> None:
#         self._set_requires_grad(self.backbone, False)

#     def unfreeze_backbone(self) -> None:
#         self._set_requires_grad(self.backbone, True)

#     def freeze_projection_head(self) -> None:
#         self._set_requires_grad(self.projection, False)
#         self._set_requires_grad(self.norm, False)

#     def unfreeze_projection_head(self) -> None:
#         self._set_requires_grad(self.projection, True)
#         self._set_requires_grad(self.norm, True)

#     def freeze_semantic_encoder(self) -> None:
#         self._set_requires_grad(self.semantic_encoder, False)
#         self._set_requires_grad(self.concept_encoder, False)

#     def freeze_classifier(self) -> None:
#         self._set_requires_grad(self.classifier, False)

#     def unfreeze_classifier(self) -> None:
#         self._set_requires_grad(self.classifier, True)

#     def freeze_classes(self, class_ids_or_count: Iterable[int] | int) -> None:
#         # GeometryBank.freeze_classes expects an iterable; freeze_classes_up_to
#         # expects a count. Keep both contracts explicit to avoid treating an int
#         # as an iterable during base handoff.
#         if isinstance(class_ids_or_count, int):
#             count = int(max(0, class_ids_or_count))
#             if hasattr(self.geometry_bank, "freeze_classes_up_to"):
#                 self.geometry_bank.freeze_classes_up_to(count)
#             elif hasattr(self.geometry_bank, "freeze_classes"):
#                 self.geometry_bank.freeze_classes(range(count))
#             return
#         ids = _ordered_unique_ints(class_ids_or_count)
#         if hasattr(self.geometry_bank, "freeze_classes"):
#             self.geometry_bank.freeze_classes(ids)
#         elif hasattr(self.geometry_bank, "freeze_classes_up_to"):
#             count = max(ids) + 1 if ids else 0
#             self.geometry_bank.freeze_classes_up_to(count)

#     def freeze_old_geometry_states(self, old_class_count: Optional[Any] = None) -> None:
#         """Freeze old GeometryBank rows without assuming a hidden update.

#         ``old_class_count`` may be an int count for the standard sequential IP/HC
#         protocol or an explicit iterable of global class IDs.
#         """
#         if old_class_count is None:
#             old = int(self.old_class_count)
#             self.freeze_classes(range(old))
#             return
#         if isinstance(old_class_count, int):
#             old = int(max(0, old_class_count))
#             self.old_class_count = old
#             self.freeze_classes(range(old))
#             return
#         ids = _ordered_unique_ints(old_class_count)
#         self.old_class_count = len(ids)
#         self.freeze_classes(ids)

#     def freeze_base_ce_head(self) -> None:
#         self._set_requires_grad(self.base_ce_head, False)

#     def unfreeze_base_ce_head(self) -> None:
#         self._set_requires_grad(self.base_ce_head, True)

#     def set_base_mode(self, *, train_backbone: bool = True, train_projection: bool = True) -> None:
#         self.current_phase = 0
#         self.old_class_count = 0
#         self.base_mode_active = True
#         self.incremental_mode_active = False
#         self.train()
#         self._set_requires_grad(self.backbone, bool(train_backbone))
#         self._set_requires_grad(self.projection, bool(train_projection))
#         self._set_requires_grad(self.norm, bool(train_projection))
#         self.freeze_semantic_encoder()
#         self.freeze_classifier()
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.eval()
#         if self.base_ce_head is not None:
#             self.unfreeze_base_ce_head()

#     def set_incremental_mode(
#         self,
#         *,
#         phase: Optional[int] = None,
#         old_class_count: Optional[int] = None,
#         old_classes: Optional[Iterable[int]] = None,
#         train_classifier_calibration: bool = False,
#         train_geometry_adapter: Optional[bool] = None,
#     ) -> None:
#         if phase is not None:
#             self.current_phase = int(phase)
#         if old_class_count is not None:
#             self.old_class_count = int(old_class_count)
#         self.base_mode_active = False
#         self.incremental_mode_active = True

#         # Freeze backbone/projection and put them in eval mode to kill dropout.
#         self.freeze_backbone_except_allowed(allow_last_blocks=False, allow_projection=False)
#         self.backbone.eval()
#         self.projection.eval()
#         self.norm.eval()
#         self.freeze_semantic_encoder()
#         self.freeze_base_ce_head()
#         self.freeze_old_geometry_states(old_classes if old_classes is not None else self.old_class_count)

#         if bool(train_classifier_calibration):
#             # Only classifier calibration parameters may be trainable if enabled.
#             self.unfreeze_classifier()
#         else:
#             self.freeze_classifier()
#         # PG-RGA main path: train only the bounded residual geometry adapter.
#         # Descriptor-only/frozen ablations keep it disabled.
#         if train_geometry_adapter is None:
#             train_geometry_adapter = bool(self.use_geometry_gated_adapter)
#         if bool(train_geometry_adapter):
#             self.unfreeze_geometry_plastic_adapter()
#         else:
#             self.freeze_geometry_plastic_adapter()
#         self._incremental_frozen_modules = ["backbone", "projection", "norm", "semantic_encoder", "concept_encoder"]

#     def assert_frozen_modules(self) -> None:
#         modules = {
#             "backbone": self.backbone,
#             "projection": self.projection,
#             "norm": self.norm,
#             "semantic_encoder": self.semantic_encoder,
#             "concept_encoder": self.concept_encoder,
#         }
#         bad_req: List[str] = []
#         bad_grad: List[str] = []
#         for prefix, module in modules.items():
#             if module is None:
#                 continue
#             for name, p in module.named_parameters():
#                 full = f"{prefix}.{name}"
#                 if p.requires_grad:
#                     bad_req.append(full)
#                 if p.grad is not None and torch.is_tensor(p.grad) and float(p.grad.detach().abs().sum().cpu().item()) != 0.0:
#                     bad_grad.append(full)
#         if bad_req:
#             raise RuntimeError(f"Frozen modules still have requires_grad=True: {bad_req[:20]}")
#         if bad_grad:
#             raise RuntimeError(f"Frozen modules have nonzero gradients: {bad_grad[:20]}")

#     # Legacy aliases kept safe, but routed to the bounded geometry adapter when
#     # the explicit geometry_gated_adapter ablation is selected.
#     def freeze_incremental_adapter(self) -> None:
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.eval()

#     def unfreeze_incremental_adapter(self) -> None:
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), True)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.train()

#     def disable_incremental_adapter(self) -> None:
#         self.use_incremental_adapter = False
#         self.freeze_incremental_adapter()

#     def enable_incremental_adapter(self) -> None:
#         self.use_geometry_gated_adapter = True
#         self.unfreeze_incremental_adapter()

#     def freeze_geometry_plastic_adapter(self) -> None:
#         self.freeze_incremental_adapter()

#     def unfreeze_geometry_plastic_adapter(self) -> None:
#         self.use_geometry_gated_adapter = True
#         self.unfreeze_incremental_adapter()

#     def adaptive_boundary_parameters(self) -> List[nn.Parameter]:
#         clf = getattr(self, "classifier", None)
#         if clf is not None and hasattr(clf, "boundary_parameters"):
#             return list(clf.boundary_parameters())
#         return []

#     def ensure_adaptive_boundary_capacity(self, class_count: int) -> None:
#         clf = getattr(self, "classifier", None)
#         if clf is not None and hasattr(clf, "ensure_class_capacity"):
#             try:
#                 clf.ensure_class_capacity(int(class_count))
#             except TypeError:
#                 pass
#         if clf is not None and hasattr(clf, "expand_to_seen_classes"):
#             try:
#                 clf.expand_to_seen_classes(list(range(int(class_count))))
#             except TypeError:
#                 pass

#     def adaptive_boundary_state(self, old_class_count: int = 0) -> Dict[str, float]:
#         clf = getattr(self, "classifier", None)
#         if clf is not None and hasattr(clf, "adaptive_boundary_state"):
#             try:
#                 return {k: float(v) for k, v in clf.adaptive_boundary_state(old_class_count=int(old_class_count)).items()}
#             except TypeError:
#                 try:
#                     return {k: float(v) for k, v in clf.adaptive_boundary_state(int(old_class_count)).items()}
#                 except Exception:
#                     pass
#         return {"old_class_count": float(old_class_count), "adaptive_boundary_available": float(bool(clf is not None and hasattr(clf, "boundary_parameters")))}

#     def freeze_geometry_calibrator(self) -> None: self.use_geometry_calibrator = False
#     def unfreeze_geometry_calibrator(self) -> None:
#         raise RuntimeError("Legacy geometry calibrator is disabled in the clean NECILModel.")
#     def freeze_energy_calibrator(self) -> None:
#         if hasattr(self.classifier, "freeze_all_adaptation"):
#             self.classifier.freeze_all_adaptation()
#     def unfreeze_energy_calibrator(self) -> None:
#         raise RuntimeError("Energy/logit calibration is disabled in the clean PG-RGA model.")

#     # ------------------------------------------------------------------
#     # Base CE head
#     # ------------------------------------------------------------------
#     def ensure_base_ce_head(self, num_base_classes: int, feature_dim: Optional[int] = None) -> nn.Linear:
#         num_base_classes = int(num_base_classes)
#         feature_dim = int(feature_dim or self.d_model)
#         if feature_dim != self.d_model:
#             raise RuntimeError(f"base CE head feature_dim must equal d_model={self.d_model}, got {feature_dim}")
#         if self.base_ce_head is None or self.base_ce_num_classes != num_base_classes:
#             self.base_ce_head = nn.Linear(self.d_model, num_base_classes).to(self.device)
#             nn.init.normal_(self.base_ce_head.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(self.base_ce_head.bias)
#             self.base_ce_num_classes = num_base_classes
#         return self.base_ce_head

#     def base_ce_logits(self, features: torch.Tensor, num_base_classes: Optional[int] = None) -> torch.Tensor:
#         features = self._validate_feature_tensor(features, "base_ce_logits.features")
#         if self.base_ce_head is None:
#             if num_base_classes is None:
#                 raise RuntimeError("base_ce_head is not initialized.")
#             self.ensure_base_ce_head(int(num_base_classes))
#         logits = self.base_ce_head(features)
#         if logits.dim() != 2 or logits.size(0) != features.size(0):
#             raise RuntimeError("base_ce_head returned invalid logits")
#         return logits

#     def forward_base_ce(self, x: torch.Tensor, num_base_classes: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         out = self.forward_features(
#             x,
#             spectral_summary=kwargs.get("spectral_summary", None),
#             band_weights=kwargs.get("band_weights", None),
#             spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
#         )
#         logits = self.base_ce_logits(out["features"], num_base_classes=int(num_base_classes))
#         out = dict(out)
#         out["base_logits"] = logits
#         out["logits"] = logits
#         return out

#     def drop_base_ce_head(self) -> None:
#         self.base_ce_head = None
#         self.base_ce_num_classes = 0

#     # PRL aliases from older trainer code.
#     ensure_base_prl_head = ensure_base_ce_head
#     base_prl_logits = base_ce_logits
#     def drop_base_prl_head(self) -> None: self.drop_base_ce_head()
#     def freeze_base_prl_head(self) -> None: self.freeze_base_ce_head()
#     def unfreeze_base_prl_head(self) -> None: self.unfreeze_base_ce_head()

#     def train(self, mode: bool = True):  # type: ignore[override]
#         super().train(mode)
#         if bool(getattr(self, "incremental_mode_active", False)):
#             # Frozen feature modules must remain deterministic during adapter
#             # training. Calling model.train() from the trainer should not turn
#             # dropout/batch statistics back on for backbone/projection.
#             self.backbone.eval()
#             self.projection.eval()
#             self.norm.eval()
#             if getattr(self, "geometry_plastic_adapter", None) is not None:
#                 self.geometry_plastic_adapter.train(mode and _module_has_trainable_params(self.geometry_plastic_adapter))
#         return self


#     # ------------------------------------------------------------------
#     # PG-RGA helper APIs used by the incremental trainer
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def sample_geometry_replay(
#         self,
#         class_ids: Iterable[int],
#         samples_per_class: int | Mapping[int, int] = 16,
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         label_to_local: Optional[Mapping[int, int]] = None,
#         parallel_scale: float = 1.0,
#         residual_scale: float = 0.25,
#         reliability_gated: bool = True,
#     ) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.geometry_bank, "sample_replay"):
#             raise RuntimeError("GeometryBank must expose sample_replay for PG-RGA old geometry replay.")
#         return self.geometry_bank.sample_replay(
#             class_ids,
#             samples_per_class=samples_per_class,
#             seen_classes=seen_classes,
#             label_to_local=label_to_local,
#             parallel_scale=float(parallel_scale),
#             residual_scale=float(residual_scale),
#             reliability_gated=bool(reliability_gated),
#         )

#     def compute_old_geometry_risk_features(
#         self,
#         features: torch.Tensor,
#         *,
#         old_class_count: Optional[int] = None,
#         geometry_bank: Optional[Any] = None,
#     ) -> Dict[str, torch.Tensor]:
#         z = self._validate_feature_tensor(features, "compute_old_geometry_risk_features.features")
#         old = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
#         bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
#         if not hasattr(self.classifier, "old_geometry_risk_features_from_bank"):
#             raise RuntimeError("Classifier must expose old_geometry_risk_features_from_bank for PG-RGA gating.")
#         return self.classifier.old_geometry_risk_features_from_bank(z, bank, old_class_count=old)

#     @torch.no_grad()
#     def assert_base_handoff_ready(self, base_class_ids: Iterable[int], *, freeze: bool = True, strict: bool = True) -> Dict[str, Any]:
#         ids = _ordered_unique_ints(base_class_ids)
#         if hasattr(self.geometry_bank, "assert_phase0_base_handoff_ready"):
#             result = self.geometry_bank.assert_phase0_base_handoff_ready(ids, freeze=bool(freeze), strict=bool(strict))
#         else:
#             if bool(freeze):
#                 self.freeze_classes(ids)
#             result = self.geometry_bank.assert_bank_valid(seen_classes=ids, strict=bool(strict))
#         self.old_class_count = len(ids)
#         self.current_num_classes = max(self.current_num_classes, max(ids) + 1 if ids else 0)
#         self.seen_classes = list(ids)
#         return result

#     def assert_pg_rga_contract(self, seen_classes: Iterable[int], *, phase: str = "base") -> None:
#         seen = _ordered_unique_ints(seen_classes)
#         self.assert_phase_ready(seen, mode="geometry", require_geometry=True)
#         if str(phase).lower().startswith("base"):
#             if self.incremental_mode_active:
#                 raise RuntimeError("Base contract violation: incremental_mode_active=True during base phase.")
#             if _module_has_trainable_params(self.geometry_plastic_adapter):
#                 raise RuntimeError("Base contract violation: geometry_plastic_adapter is trainable during base phase.")
#         else:
#             self.assert_frozen_modules()
#             if self.use_geometry_gated_adapter and not _module_has_trainable_params(self.geometry_plastic_adapter):
#                 raise RuntimeError("PG-RGA incremental contract violation: geometry_plastic_adapter is not trainable.")

#     # ------------------------------------------------------------------
#     # Assertions
#     # ------------------------------------------------------------------
#     def assert_phase_ready(self, seen_classes: Iterable[int], *, mode: str = "geometry", require_geometry: bool = True) -> None:
#         seen = _ordered_unique_ints(seen_classes)
#         if not seen:
#             raise RuntimeError("seen_classes is empty.")
#         if self.geometry_bank.d_model != self.d_model:
#             raise RuntimeError(f"GeometryBank d_model={self.geometry_bank.d_model} != model d_model={self.d_model}")
#         if _normalize_classifier_mode(mode, "geometry") in {"geometry", "calibrated_geometry"} and require_geometry:
#             if hasattr(self.geometry_bank, "assert_bank_valid"):
#                 self.geometry_bank.assert_bank_valid(seen_classes=seen, strict=True)
#             else:
#                 bank = self.get_subspace_bank()
#                 missing = [c for c in seen if c >= bank["sample_counts"].numel() or float(bank["sample_counts"][c].item()) <= 0.0]
#                 if missing:
#                     raise RuntimeError(f"Missing class geometry for seen classes: {missing}")
#         if self.incremental_mode_active:
#             # In incremental mode, frozen feature modules must stay eval to avoid dropout drift.
#             bad_train = []
#             for name in ("backbone", "projection"):
#                 m = getattr(self, name)
#                 if m.training:
#                     bad_train.append(name)
#             if bad_train:
#                 raise RuntimeError(f"Incremental frozen modules must be eval(), but these are train(): {bad_train}")
#         self.classifier.expand_to_seen_classes(seen)

#     def assert_no_missing_class_geometry(self, seen_classes: Iterable[int]) -> None:
#         if hasattr(self.geometry_bank, "assert_bank_valid"):
#             self.geometry_bank.assert_bank_valid(seen_classes=seen_classes, strict=True)


#     def assert_method_identity(self) -> None:
#         """Hard runtime check used by main.py/trainer.py."""
#         if self.incremental_update_mode != "geometry_gated_adapter":
#             raise RuntimeError(f"Unexpected incremental_update_mode={self.incremental_update_mode!r}")
#         if not hasattr(self, "geometry_plastic_adapter"):
#             raise RuntimeError("PG-RGA model missing geometry_plastic_adapter.")
#         if bool(getattr(self, "use_geometry_transport", False)) or bool(getattr(self, "use_sglat_transport", False)):
#             raise RuntimeError("Transport is disabled in the PG-RGA main model.")
#         if bool(getattr(self, "use_geometry_calibrator", False)):
#             raise RuntimeError("Geometry/logit calibrator is disabled in the PG-RGA main model.")
#         if bool(getattr(self, "use_incremental_adapter", False)):
#             raise RuntimeError("Legacy incremental_adapter must remain disabled; use geometry_plastic_adapter only.")

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------
#     def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
#         mode = _normalize_classifier_mode(kwargs.get("classifier_mode", kwargs.get("mode", "geometry")), "geometry")
#         return_features_only = _to_bool(kwargs.get("return_features_only", False), False)
#         seen_classes = kwargs.get("seen_classes", None)
#         old_classes = kwargs.get("old_classes", None)
#         new_classes = kwargs.get("new_classes", None)
#         targets = kwargs.get("targets", kwargs.get("labels", None))
#         targets_are_global = _to_bool(kwargs.get("targets_are_global", kwargs.get("labels_are_global", False)), False)
#         return_energy = _to_bool(kwargs.get("return_energy", False), False)
#         return_parts = _to_bool(kwargs.get("return_parts", False), False)
#         return_diagnostics = _to_bool(kwargs.get("return_diagnostics", False), False)

#         apply_adapter = kwargs.get("apply_adapter", None)
#         if mode == "base_ce":
#             apply_adapter = False
#         elif apply_adapter is None:
#             apply_adapter = bool(self._adapter_runtime_enabled(force=False))
#         features_out = self.forward_features(
#             x,
#             spectral_summary=kwargs.get("spectral_summary", None),
#             band_weights=kwargs.get("band_weights", None),
#             spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
#             apply_adapter=apply_adapter,
#         )
#         if return_features_only or mode == "base_ce":
#             if mode == "base_ce" or "num_base_classes" in kwargs:
#                 nbase = int(kwargs.get("num_base_classes", self.base_ce_num_classes))
#                 if nbase <= 0:
#                     raise RuntimeError("num_base_classes is required for base_ce mode.")
#                 features_out = dict(features_out)
#                 features_out["logits"] = self.base_ce_logits(features_out["features"], nbase)
#             return features_out

#         if seen_classes is None:
#             seen_classes = self._infer_seen_classes(self.geometry_bank)
#         logits_out = self.compute_logits_from_features(
#             features_out["features"],
#             seen_classes=seen_classes,
#             geometry_bank=self.geometry_bank,
#             mode=mode,
#             targets=targets,
#             targets_are_global=targets_are_global,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )
#         out: Dict[str, Any] = dict(features_out)
#         if isinstance(logits_out, dict):
#             out.update(logits_out)
#         else:
#             out["logits"] = logits_out
#         return out

















# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import inspect
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.backbone import SSMBackbone
# from models.geometry_bank import GeometryBank
# from models.classifier import GeometryEnergyClassifier


# def _to_bool(value: Any, default: bool = False) -> bool:
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
#     return bool(value)


# def _filter_supported_kwargs(cls_or_fn: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
#     try:
#         sig = inspect.signature(cls_or_fn)
#     except (TypeError, ValueError):
#         return dict(kwargs)
#     if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
#         return dict(kwargs)
#     allowed = set(sig.parameters.keys())
#     allowed.discard("self")
#     return {k: v for k, v in kwargs.items() if k in allowed}


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     out: List[int] = []
#     seen = set()
#     for v in values:
#         iv = int(v)
#         if iv not in seen:
#             out.append(iv)
#             seen.add(iv)
#     return out


# def _normalize_classifier_mode(mode: Optional[str], default: str = "geometry") -> str:
#     m = str(default if mode is None else mode).lower().strip()
#     aliases = {
#         "": default,
#         "none": default,
#         "geo": "geometry",
#         "geometry_only": "geometry",
#         "geometry-only": "geometry",
#         "feature_geometry": "geometry",
#         "low_rank_geometry": "geometry",
#         "anchor": "geometry",
#         "anchor_concept": "geometry",
#         "anchor_concept_geometry": "geometry",
#         "srgp": "geometry",
#         "srgp_geometry": "geometry",
#         "spectral_geometry": "geometry",
#         "spectral_residual": "geometry",
#         "calibrated_geometry": "calibrated_geometry",
#         "topology_calibrated_geometry": "calibrated_geometry",
#     }
#     out = aliases.get(m, m)
#     if out not in {"geometry", "calibrated_geometry", "base_ce"}:
#         raise ValueError(f"Unsupported classifier mode {mode!r}. Use geometry, calibrated_geometry, or base_ce.")
#     return out



# def _normalize_incremental_update_mode(mode: Optional[str]) -> str:
#     """Normalize incremental update mode for the clean PG-RGA architecture."""
#     m = str(mode or "geometry_gated_adapter").lower().strip()
#     aliases = {
#         "": "geometry_gated_adapter",
#         "none": "frozen_geometry",
#         "false": "frozen_geometry",
#         "off": "frozen_geometry",
#         "geometry_gated": "geometry_gated_adapter",
#         "gated_adapter": "geometry_gated_adapter",
#         "pg_rga": "geometry_gated_adapter",
#         "pgrga": "geometry_gated_adapter",
#         "sgrga": "geometry_gated_adapter",
#         "adapter": "geometry_gated_adapter",
#         "descriptor": "descriptor_only",
#         "descriptor_refinement": "descriptor_only",
#         "frozen": "frozen_geometry",
#         "frozen_geometry": "frozen_geometry",
#     }
#     out = aliases.get(m, m)
#     if out not in {"geometry_gated_adapter", "descriptor_only", "frozen_geometry"}:
#         raise ValueError(
#             f"Unsupported incremental_update_mode={mode!r}. "
#             "Use geometry_gated_adapter for the main PG-RGA path."
#         )
#     return out


# def _module_has_trainable_params(module: Optional[nn.Module]) -> bool:
#     return bool(module is not None and any(p.requires_grad for p in module.parameters()))


# class GeometryPlasticAdapter(nn.Module):
#     """Bounded geometry-gated residual adapter for controlled incremental plasticity.

#     This module operates directly in the canonical projected GeometryBank space.
#     It is deliberately small: it can make a bounded residual correction for new
#     classes, but it cannot rewrite the backbone/projection or old bank rows.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         *,
#         bottleneck: int = 32,
#         max_scale: float = 0.10,
#         dropout: float = 0.0,
#         gate_bias_init: float = -3.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.bottleneck = int(max(1, bottleneck))
#         self.max_scale = float(max(0.0, max_scale))
#         self.norm = nn.LayerNorm(self.d_model)
#         self.delta = nn.Sequential(
#             nn.Linear(self.d_model, self.bottleneck),
#             nn.GELU(),
#             nn.Dropout(float(max(0.0, dropout))),
#             nn.Linear(self.bottleneck, self.d_model),
#         )
#         self.gate = nn.Sequential(
#             nn.Linear(self.d_model, self.bottleneck),
#             nn.GELU(),
#             nn.Linear(self.bottleneck, 1),
#         )
#         self.reset_parameters(float(gate_bias_init))

#     def reset_parameters(self, gate_bias_init: float = -3.0) -> None:
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.zeros_(module.bias)
#         # Start as an identity map. The adapter earns plasticity only through
#         # incremental new/replay losses.
#         last_delta = self.delta[-1]
#         if isinstance(last_delta, nn.Linear):
#             nn.init.zeros_(last_delta.weight)
#             nn.init.zeros_(last_delta.bias)
#         last_gate = self.gate[-1]
#         if isinstance(last_gate, nn.Linear):
#             nn.init.zeros_(last_gate.weight)
#             nn.init.constant_(last_gate.bias, float(gate_bias_init))

#     def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
#         if features.dim() != 2:
#             raise RuntimeError(f"GeometryPlasticAdapter expects [B,D], got {tuple(features.shape)}")
#         h = self.norm(features)
#         gate = torch.sigmoid(self.gate(h))
#         direction = torch.tanh(self.delta(h))
#         delta = gate * float(self.max_scale) * direction
#         adapted = features + delta
#         return {
#             "features": adapted,
#             "projected_features": adapted,
#             "delta": delta,
#             "gate": gate,
#             "adapter_delta": delta,
#             "adapter_gate": gate,
#         }


# class NECILModel(nn.Module):
#     """Clean NECIL-HSI model router.

#     Contract:
#         - One canonical projected feature space ``z`` is used for base geometry,
#           incremental geometry insertion, synthetic replay, and evaluation.
#         - GeometryBank is the only non-exemplar memory.
#         - Classifier always receives explicit ``seen_classes`` and returns
#           logits [B, len(seen_classes)].
#         - Semantic/concept/adaptor/transport paths are not part of this clean
#           main model. Their hooks are kept as no-ops or hard-disabled aliases so
#           stale trainer code does not silently route through a different space.
#     """

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.args = args
#         self.device = torch.device(getattr(args, "device", "cpu"))
#         self.d_model = int(getattr(args, "d_model", 128))
#         self.subspace_rank = int(getattr(args, "subspace_rank", getattr(args, "rank", 5)))
#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive.")
#         if self.subspace_rank < 0:
#             raise ValueError("subspace_rank must be non-negative.")

#         self.current_phase = 0
#         self.old_class_count = 0
#         self.current_num_classes = 0
#         self.seen_classes: List[int] = []
#         self.base_mode_active = True
#         self.incremental_mode_active = False
#         self._incremental_frozen_modules: List[str] = []

#         # Main architecture switch.
#         # PG-RGA uses a bounded residual geometry adapter during incremental
#         # learning, while the backbone/projection and old GeometryBank rows stay
#         # frozen. Descriptor-only/frozen modes remain available as ablations.
#         self.incremental_update_mode = _normalize_incremental_update_mode(
#             getattr(args, "incremental_update_mode", None)
#         )

#         # Hard-disable stale/unsafe paths in the model object. They can exist in
#         # other files for ablation compatibility, but this clean model must not
#         # silently route through them.
#         self.use_incremental_adapter = False
#         self.use_geometry_calibrator = False
#         self.use_bicyc_geometry_cycle = False
#         self.use_geometry_transport = False
#         self.use_sglat_transport = False
#         self.use_geometry_gated_adapter = (
#             self.incremental_update_mode == "geometry_gated_adapter"
#             or _to_bool(getattr(args, "use_geometry_gated_adapter", False), False)
#         )
#         self.semantic_encoder: Optional[nn.Module] = None
#         self.concept_encoder: Optional[nn.Module] = None

#         self.backbone = SSMBackbone(args)
#         dropout = float(getattr(args, "projection_dropout", 0.0))
#         self.projection = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#         self.normalize_geometry_features = _to_bool(getattr(args, "normalize_geometry_features", True), True)
#         raw_scale = float(getattr(args, "geometry_feature_scale", 0.0) or 0.0)
#         self.geometry_feature_scale = raw_scale if raw_scale > 0.0 else math.sqrt(float(self.d_model))
#         self.geometry_feature_clamp = float(getattr(args, "geometry_feature_clamp", 0.0) or 0.0)
#         self.spectral_summary_mode = str(getattr(args, "spectral_summary_mode", "center")).lower().strip()
#         if self.spectral_summary_mode not in {"center", "mean"}:
#             raise ValueError("spectral_summary_mode must be 'center' or 'mean'.")
#         pca_components = int(getattr(args, "pca_components", 0) or 0)
#         self.default_spectral_physical = bool(getattr(args, "spectral_summary_is_physical", pca_components <= 0))
#         self.min_band_mass = float(getattr(args, "min_band_mass", 1e-8))

#         bank_kwargs = _filter_supported_kwargs(
#             GeometryBank.__init__,
#             {
#                 "device": self.device,
#                 "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
#                 "variance_shrinkage": float(getattr(args, "geometry_variance_shrinkage", 0.10)),
#                 "max_variance_ratio": float(getattr(args, "geometry_max_variance_ratio", 50.0)),
#                 "min_reliability": float(getattr(args, "geometry_min_reliability", 0.05)),
#                 "reliability_sample_alpha": float(getattr(args, "reliability_sample_alpha", 20.0)),
#                 "rank_energy_threshold": float(getattr(args, "rank_energy_threshold", 0.95)),
#                 "rank_eigen_ratio_threshold": float(getattr(args, "rank_eigen_ratio_threshold", 1e-3)),
#                 "min_active_rank": int(getattr(args, "min_active_rank", 1)),
#             },
#         )
#         self.geometry_bank = GeometryBank(self.d_model, self.subspace_rank, **bank_kwargs)

#         clf_kwargs = _filter_supported_kwargs(
#             GeometryEnergyClassifier.__init__,
#             {
#                 "initial_classes": 0,
#                 "d_model": self.d_model,
#                 "logit_scale": float(getattr(args, "loss_scale", getattr(args, "logit_scale", 8.0))),
#                 "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
#                 "residual_variance_scale": float(getattr(args, "residual_variance_scale", 0.75)),
#                 "normalize_energy_by_dim": _to_bool(getattr(args, "energy_normalize_by_dim", True), True),
#                 "use_logdet_energy": _to_bool(getattr(args, "use_logdet_energy", True), True),
#                 "logdet_energy_weight": float(getattr(args, "logdet_energy_weight", 0.05)),
#                 "use_reliability_penalty": _to_bool(getattr(args, "use_reliability_penalty", True), True),
#                 "reliability_energy_weight": float(getattr(args, "reliability_energy_weight", 0.03)),
#                 "use_old_new_calibration": _to_bool(getattr(args, "use_old_new_calibration", getattr(args, "use_energy_calibrator", False)), False),
#                 "calibration_max_abs_bias": float(getattr(args, "calibration_max_abs_bias", getattr(args, "energy_calibrator_max_bias", 1.0))),
#                 "logit_clip": float(getattr(args, "geometry_logit_clip", 0.0)),
#             },
#         )
#         self.classifier = GeometryEnergyClassifier(**clf_kwargs)

#         self.geometry_plastic_adapter = GeometryPlasticAdapter(
#             self.d_model,
#             bottleneck=int(getattr(args, "adapter_bottleneck", 32)),
#             max_scale=float(getattr(args, "adapter_max_scale", 0.10)),
#             dropout=float(getattr(args, "adapter_dropout", 0.0)),
#             gate_bias_init=float(getattr(args, "adapter_gate_bias_init", -3.0)),
#         )
#         self._set_requires_grad(self.geometry_plastic_adapter, False)

#         self.base_ce_head: Optional[nn.Linear] = None
#         self.base_ce_num_classes = 0

#         self.to(self.device)

#     # ------------------------------------------------------------------
#     # Input / feature utilities
#     # ------------------------------------------------------------------
#     def _validate_feature_tensor(self, features: torch.Tensor, name: str, batch_size: Optional[int] = None) -> torch.Tensor:
#         if not torch.is_tensor(features):
#             raise TypeError(f"{name} must be a torch.Tensor, got {type(features)}")
#         if features.dim() != 2:
#             raise RuntimeError(f"{name} must be [B,D], got {tuple(features.shape)}")
#         if features.size(1) != self.d_model:
#             raise RuntimeError(f"{name} dim mismatch: expected {self.d_model}, got {features.size(1)}")
#         if batch_size is not None and features.size(0) != int(batch_size):
#             raise RuntimeError(f"{name} batch mismatch: got {features.size(0)}, expected {int(batch_size)}")
#         if not torch.isfinite(features).all():
#             raise RuntimeError(f"{name} contains NaN/Inf values.")
#         return features

#     def _canonicalize(self, z: torch.Tensor, *, name: str) -> torch.Tensor:
#         z = self._validate_feature_tensor(z, name)
#         if self.normalize_geometry_features:
#             z = F.normalize(z, p=2, dim=1, eps=1e-8) * float(self.geometry_feature_scale)
#         if self.geometry_feature_clamp > 0.0:
#             z = z.clamp(-self.geometry_feature_clamp, self.geometry_feature_clamp)
#         if not torch.isfinite(z).all():
#             raise RuntimeError(f"{name} contains NaN/Inf after canonicalization.")
#         return z

#     def _center_spectrum_from_input(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 4:
#             if self.spectral_summary_mode == "center":
#                 return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
#             return x.mean(dim=(-1, -2))
#         if x.dim() == 3:
#             if self.spectral_summary_mode == "center":
#                 return x[:, :, x.size(-1) // 2]
#             return x.mean(dim=-1)
#         if x.dim() == 2:
#             return x
#         raise RuntimeError(f"Unsupported input shape for spectral summary: {tuple(x.shape)}")

#     def _prepare_spectral_summary(
#         self,
#         x: torch.Tensor,
#         features: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#     ) -> Tuple[torch.Tensor, bool]:
#         if spectral_summary is None or not torch.is_tensor(spectral_summary) or spectral_summary.numel() == 0:
#             s = self._center_spectrum_from_input(x).to(device=features.device, dtype=features.dtype)
#             physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
#         else:
#             s = spectral_summary.to(device=features.device, dtype=features.dtype)
#             if s.dim() == 4:
#                 s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#             elif s.dim() == 3:
#                 if s.size(0) == features.size(0) and s.size(-1) > 1:
#                     s = s[:, :, s.size(-1) // 2]
#                 else:
#                     s = s.reshape(features.size(0), -1)
#             elif s.dim() == 1:
#                 if s.numel() % max(int(features.size(0)), 1) != 0:
#                     raise RuntimeError(f"1-D spectral_summary cannot be reshaped to batch size {features.size(0)}")
#                 s = s.reshape(features.size(0), -1)
#             elif s.dim() > 4:
#                 s = s.flatten(1)
#             physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
#         if s.dim() != 2 or s.size(0) != features.size(0):
#             raise RuntimeError(f"spectral_summary must resolve to [B,S], got {tuple(s.shape)}")
#         s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#         if s.size(1) == 0:
#             physical = False
#         return s, bool(physical)

#     def _band_summary(self, spectral_summary: torch.Tensor, band_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if band_weights is not None and torch.is_tensor(band_weights) and band_weights.numel() > 0:
#             bw = band_weights.to(device=spectral_summary.device, dtype=spectral_summary.dtype)
#             if bw.dim() == 2 and bw.size(0) == spectral_summary.size(0) and bw.size(1) == spectral_summary.size(1):
#                 bw = torch.nan_to_num(bw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
#                 denom = bw.sum(dim=1, keepdim=True)
#                 if bool((denom > self.min_band_mass).all().item()):
#                     return bw / denom.clamp_min(self.min_band_mass)
#         b = spectral_summary.abs()
#         denom = b.sum(dim=1, keepdim=True)
#         uniform = torch.full_like(b, 1.0 / float(max(int(b.size(1)), 1)))
#         b = torch.where(denom > self.min_band_mass, b / denom.clamp_min(self.min_band_mass), uniform)
#         return b

#     def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """Backbone-only feature extraction. Does not touch GeometryBank/classifier."""
#         if not torch.is_tensor(x):
#             raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
#         out = self.backbone(x.to(self.device))
#         if isinstance(out, dict):
#             if "features" not in out:
#                 raise RuntimeError("backbone output dict must contain key 'features'.")
#             features = self._validate_feature_tensor(out["features"], "backbone.features", int(x.size(0)))
#             result = dict(out)
#             result["features"] = features
#             result.setdefault("backbone_features", features)
#             return result
#         features = self._validate_feature_tensor(out, "backbone.features", int(x.size(0)))
#         return {"features": features, "backbone_features": features}

#     def forward_features(
#         self,
#         x: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#         apply_adapter: Optional[bool] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Return projected geometry features.

#         Important contract:
#             - ``canonical_features`` are always the backbone/projection z-space
#               used to build the base GeometryBank.
#             - ``features`` / ``projected_features`` are the scoring features.
#               They equal canonical z in base mode and become adapted z only when
#               PG-RGA adapter routing is explicitly active.
#         """
#         raw = self.extract_features(x)
#         h = self._validate_feature_tensor(raw["features"], "preproject_features", int(x.size(0)))
#         z_canonical = self.norm(self.projection(h) + h)
#         z_canonical = self._canonicalize(z_canonical, name="canonical_geometry_features")

#         use_adapter = self._adapter_runtime_enabled(force=False) if apply_adapter is None else bool(apply_adapter)
#         z = z_canonical
#         adapter_out: Optional[Dict[str, torch.Tensor]] = None
#         if use_adapter:
#             adapter_out = self.adapt_projected_features(
#                 z_canonical,
#                 force=(apply_adapter is True),
#                 return_delta=True,
#                 recanonicalize=True,
#             )
#             z = adapter_out["features"]

#         s, physical = self._prepare_spectral_summary(
#             x.to(self.device),
#             z,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#         )
#         candidate_bw = band_weights if band_weights is not None else raw.get("band_weights", None)
#         band = self._band_summary(s, candidate_bw)
#         result = {
#             "features": z,
#             "projected_features": z,
#             "geometry_features": z,
#             "canonical_features": z_canonical,
#             "canonical_projected_features": z_canonical,
#             "pre_adapter_features": z_canonical,
#             "preproject_features": h,
#             "backbone_features": h,
#             "spectral_summary": s,
#             "spectral_summary_is_physical": torch.tensor(bool(physical), device=z.device, dtype=torch.bool),
#             "band_summary": band,
#             "band_importance": band,
#             "band_weights": candidate_bw if torch.is_tensor(candidate_bw) else None,
#             "spectral_features": raw.get("spectral_features", h),
#             "spatial_features": raw.get("spatial_features", h),
#         }
#         if adapter_out is not None:
#             result["adapter_delta"] = adapter_out["adapter_delta"]
#             result["adapter_gate"] = adapter_out["adapter_gate"]
#             result["adapter_active"] = adapter_out["adapter_active"]
#         return result

#     # Compatibility aliases expected by existing trainers.
#     def extract_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         return self.forward_features(x, **kwargs)

#     def extract_canonical_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         kwargs = dict(kwargs)
#         kwargs["apply_adapter"] = False
#         return self.forward_features(x, **kwargs)

#     def extract_adapted_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         kwargs = dict(kwargs)
#         kwargs["apply_adapter"] = True
#         return self.forward_features(x, **kwargs)

#     def extract_geometry_features(self, x: torch.Tensor, *, return_dict: bool = False, **kwargs: Any):
#         out = self.forward_features(x, **kwargs)
#         return out if bool(return_dict) else out["features"]

#     @torch.no_grad()
#     def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         return self.forward_features(x)

#     # ------------------------------------------------------------------
#     # Controlled geometry-gated adapter
#     # ------------------------------------------------------------------
#     def _adapter_runtime_enabled(self, *, force: bool = False) -> bool:
#         return bool(
#             getattr(self, "geometry_plastic_adapter", None) is not None
#             and (bool(force) or (bool(getattr(self, "use_geometry_gated_adapter", False)) and bool(self.incremental_mode_active)))
#         )

#     def adapt_projected_features(
#         self,
#         features: torch.Tensor,
#         *,
#         force: bool = False,
#         return_delta: bool = True,
#         recanonicalize: bool = True,
#         **_: Any,
#     ) -> Dict[str, torch.Tensor]:
#         """Apply the bounded adapter directly to canonical z-space features.

#         This is required for synthetic old replay: replay samples already live
#         in GeometryBank space and cannot pass through the backbone. The adapter
#         must still see them so replay CE can close old-region gates.
#         """
#         z0 = self._validate_feature_tensor(features, "adapt_projected_features.features")
#         if not self._adapter_runtime_enabled(force=force):
#             zero_delta = torch.zeros_like(z0)
#             zero_gate = torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype)
#             return {
#                 "features": z0,
#                 "projected_features": z0,
#                 "geometry_features": z0,
#                 "delta": zero_delta,
#                 "gate": zero_gate,
#                 "adapter_delta": zero_delta,
#                 "adapter_gate": zero_gate,
#                 "adapter_active": torch.tensor(False, device=z0.device),
#             }
#         raw = self.geometry_plastic_adapter(z0)
#         z = raw["features"]
#         if bool(recanonicalize):
#             z = self._canonicalize(z, name="adapted_geometry_features")
#         delta = z - z0
#         gate = raw.get("gate", torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype))
#         return {
#             "features": z,
#             "projected_features": z,
#             "geometry_features": z,
#             "delta": delta,
#             "gate": gate,
#             "adapter_delta": delta,
#             "adapter_gate": gate,
#             "adapter_active": torch.tensor(True, device=z0.device),
#         }

#     # ------------------------------------------------------------------
#     # Seen classes / bank helpers
#     # ------------------------------------------------------------------
#     def _infer_seen_classes(self, geometry_bank: Optional[Any] = None) -> List[int]:
#         bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
#         if hasattr(bank_obj, "get_valid_mask"):
#             valid = bank_obj.get_valid_mask().detach().cpu().flatten()
#             return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
#         bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
#         if not isinstance(bank, dict) or "sample_counts" not in bank:
#             return list(self.seen_classes)
#         counts = bank["sample_counts"].detach().cpu().flatten()
#         return [int(i) for i in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()]

#     def ensure_class_capacity(self, class_count: int, spectral_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
#         count = int(max(0, class_count))
#         dtype = dtype or next(self.parameters()).dtype
#         if hasattr(self.geometry_bank, "ensure_class_count"):
#             self.geometry_bank.ensure_class_count(count, spectral_dim=int(spectral_dim), dtype=dtype)
#         elif hasattr(self.geometry_bank, "ensure_num_classes"):
#             self.geometry_bank.ensure_num_classes(count)
#         self.current_num_classes = max(self.current_num_classes, count)

#     def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.geometry_bank, "get_bank"):
#             raise RuntimeError("GeometryBank must expose get_bank().")
#         bank = self.geometry_bank.get_bank()
#         if "variances" not in bank and "eigvals" in bank and "res_vars" in bank:
#             bank = dict(bank)
#             bank["variances"] = torch.cat([bank["eigvals"], bank["res_vars"].unsqueeze(-1)], dim=-1)
#         if "resvars" not in bank and "res_vars" in bank:
#             bank = dict(bank)
#             bank["resvars"] = bank["res_vars"]
#         return bank

#     def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank = self.get_subspace_bank()
#         old = max(0, min(old, int(bank["means"].size(0))))
#         out = {}
#         for k, v in bank.items():
#             if torch.is_tensor(v) and v.dim() > 0 and v.size(0) >= old:
#                 out[k] = v[:old]
#         return out

#     # ------------------------------------------------------------------
#     # Classifier routing
#     # ------------------------------------------------------------------
#     def forward_classifier(
#         self,
#         features: torch.Tensor,
#         seen_classes: Iterable[int],
#         mode: str = "geometry",
#         *,
#         geometry_bank: Optional[Any] = None,
#         targets: Optional[torch.Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#     ):
#         return self.compute_logits_from_features(
#             features=features,
#             seen_classes=seen_classes,
#             geometry_bank=geometry_bank,
#             mode=mode,
#             targets=targets,
#             targets_are_global=targets_are_global,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )

#     def compute_logits_from_features(
#         self,
#         features: torch.Tensor,
#         seen_classes: Optional[Iterable[int]] = None,
#         geometry_bank: Optional[Any] = None,
#         mode: str = "geometry",
#         *,
#         classifier_mode: Optional[str] = None,
#         targets: Optional[torch.Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#         **_: Any,
#     ):
#         features = self._validate_feature_tensor(features, "compute_logits_from_features.features")
#         bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
#         seen = _ordered_unique_ints(seen_classes if seen_classes is not None else self._infer_seen_classes(bank_obj))
#         if not seen:
#             raise RuntimeError("seen_classes is empty. Build GeometryBank rows or pass seen_classes explicitly.")
#         mode_norm = _normalize_classifier_mode(classifier_mode if classifier_mode is not None else mode, "geometry")
#         self.assert_phase_ready(seen, mode=mode_norm, require_geometry=True)
#         out = self.classifier(
#             features,
#             seen_classes=seen,
#             geometry_bank=bank_obj,
#             mode=mode_norm,
#             targets=targets,
#             targets_are_global=bool(targets_are_global),
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=int(self.old_class_count),
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )
#         logits = out["logits"] if isinstance(out, dict) else out
#         self.classifier.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets=(self.classifier.global_to_local_labels(targets, seen) if targets is not None and targets_are_global else targets),
#             old_classes=old_classes,
#             new_classes=new_classes,
#             context="NECILModel.compute_logits_from_features",
#         )
#         self.current_num_classes = max(self.current_num_classes, max(seen) + 1)
#         self.seen_classes = list(seen)
#         return out

#     def compute_energy_from_features(self, features: torch.Tensor, seen_classes: Optional[Iterable[int]] = None, **kwargs: Any):
#         out = self.compute_logits_from_features(features, seen_classes=seen_classes, return_energy=True, **kwargs)
#         if not isinstance(out, dict) or "energy" not in out:
#             raise RuntimeError("Expected dict with energy from compute_logits_from_features(return_energy=True).")
#         return out["energy"]

#     # ------------------------------------------------------------------
#     # Geometry refresh / memory snapshots
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def refresh_geometry_for_classes(
#         self,
#         class_ids: Iterable[int],
#         features: torch.Tensor,
#         labels: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#         phase_created: Optional[int] = None,
#         freeze_after: bool = False,
#     ) -> Dict[str, Any]:
#         features = self._validate_feature_tensor(features, "refresh_geometry_for_classes.features")
#         labels = labels.to(device=features.device).long().flatten()
#         ids = _ordered_unique_ints(class_ids)
#         if labels.numel() != features.size(0):
#             raise RuntimeError("labels/features batch mismatch in refresh_geometry_for_classes")
#         bad = sorted(set(int(v) for v in torch.unique(labels).detach().cpu().tolist()).difference(set(ids)))
#         if bad:
#             raise RuntimeError(f"labels contain classes outside requested refresh ids: bad={bad}, class_ids={ids}")
#         band_dim = 0
#         if spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0:
#             band_dim = int(spectral_summary.reshape(features.size(0), -1).size(1))
#         elif band_weights is not None and torch.is_tensor(band_weights) and band_weights.numel() > 0:
#             band_dim = int(band_weights.reshape(features.size(0), -1).size(1))
#         self.ensure_class_capacity(max(ids) + 1 if ids else 0, spectral_dim=band_dim, dtype=features.dtype)
#         rows = self.geometry_bank.extract_geometry(
#             features,
#             labels,
#             spectral_summary=spectral_summary,
#             band_weights=band_weights,
#             spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         )
#         committed: List[int] = []
#         for c in ids:
#             if c not in rows:
#                 raise RuntimeError(f"No geometry could be extracted for class {c}.")
#             row = rows[c]
#             self.geometry_bank.add_or_update_class_geometry(
#                 c,
#                 mean=row["mean"],
#                 basis=row["basis"],
#                 eigvals=row["eigvals"],
#                 res_var=row["res_var"],
#                 spectral_prototype=row.get("spectral_prototype"),
#                 band_importance=row.get("band_importance"),
#                 sample_count=row.get("sample_count"),
#                 active_rank=row.get("active_rank"),
#                 reliability=row.get("reliability"),
#                 feature_reliability=row.get("feature_reliability"),
#                 band_reliability=row.get("band_reliability"),
#                 spectral_reliability=row.get("spectral_reliability"),
#                 phase_created=int(self.current_phase if phase_created is None else phase_created),
#                 allow_frozen_update=False,
#             )
#             committed.append(c)
#         if bool(freeze_after):
#             self.freeze_classes(committed)
#         self.geometry_bank.assert_bank_valid(seen_classes=committed, strict=True)
#         self.current_num_classes = max(self.current_num_classes, max(committed) + 1 if committed else self.current_num_classes)
#         return {"committed_class_ids": committed, "phase_created": int(self.current_phase if phase_created is None else phase_created)}

#     @torch.no_grad()
#     def refresh_class_subspace(self, cls: int, mean: torch.Tensor, basis: torch.Tensor, eigvals: torch.Tensor, res_var: Optional[torch.Tensor] = None, resvar: Optional[torch.Tensor] = None, **kwargs: Any) -> None:
#         rv = res_var if res_var is not None else resvar
#         if rv is None:
#             raise ValueError("refresh_class_subspace requires res_var/resvar.")
#         c = int(cls)
#         self.ensure_class_capacity(c + 1)
#         self.geometry_bank.add_or_update_class_geometry(
#             c,
#             mean=mean,
#             basis=basis,
#             eigvals=eigvals,
#             res_var=rv,
#             spectral_prototype=kwargs.get("spectral_prototype", kwargs.get("spectral_proto", None)),
#             band_importance=kwargs.get("band_importance", None),
#             sample_count=kwargs.get("sample_count", None),
#             active_rank=kwargs.get("active_rank", None),
#             reliability=kwargs.get("reliability", None),
#             feature_reliability=kwargs.get("feature_reliability", None),
#             band_reliability=kwargs.get("band_reliability", None),
#             phase_created=kwargs.get("phase_created", self.current_phase),
#             allow_frozen_update=bool(kwargs.get("allow_frozen_update", False)),
#         )

#     @torch.no_grad()
#     def export_memory_snapshot(self) -> Dict[str, Any]:
#         snap = self.geometry_bank.export_snapshot()
#         snap.update(
#             {
#                 "current_phase": int(self.current_phase),
#                 "old_class_count": int(self.old_class_count),
#                 "current_num_classes": int(self.current_num_classes),
#                 "seen_classes": list(self.seen_classes),
#                 "feature_contract": self.feature_contract(),
#             }
#         )
#         return snap

#     @torch.no_grad()
#     def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True) -> None:
#         if not snapshot:
#             if strict:
#                 raise ValueError("empty memory snapshot")
#             return
#         self._assert_snapshot_feature_contract(snapshot, strict=strict)
#         self.geometry_bank.load_snapshot(snapshot, strict=strict)
#         self.current_phase = int(snapshot.get("current_phase", self.current_phase))
#         self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
#         self.current_num_classes = int(snapshot.get("current_num_classes", len(self.geometry_bank)))
#         self.seen_classes = [int(c) for c in snapshot.get("seen_classes", self._infer_seen_classes())]

#     def feature_contract(self) -> Dict[str, Any]:
#         return {
#             "d_model": int(self.d_model),
#             "subspace_rank": int(self.subspace_rank),
#             "normalize_geometry_features": bool(self.normalize_geometry_features),
#             "geometry_feature_scale": float(self.geometry_feature_scale),
#             "spectral_summary_mode": str(self.spectral_summary_mode),
#             "classifier_contract": "logits[B,len(seen_classes)]",
#             "incremental_update_mode": str(self.incremental_update_mode),
#             "geometry_gated_adapter_available": hasattr(self, "geometry_plastic_adapter"),
#             "geometry_gated_adapter_enabled": bool(self.use_geometry_gated_adapter),
#             "semantic_encoder_enabled": False,
#             "concept_encoder_enabled": False,
#         }

#     def _assert_snapshot_feature_contract(self, snapshot: Dict[str, Any], *, strict: bool) -> None:
#         if not strict:
#             return
#         old = snapshot.get("feature_contract", snapshot.get("geometry_feature_contract", None))
#         if not isinstance(old, dict):
#             return
#         cur = self.feature_contract()
#         mismatches = []
#         for k in ("d_model", "subspace_rank", "normalize_geometry_features", "spectral_summary_mode"):
#             if k in old and old[k] != cur[k]:
#                 mismatches.append(f"{k}: snapshot={old[k]!r}, current={cur[k]!r}")
#         if "geometry_feature_scale" in old and abs(float(old["geometry_feature_scale"]) - float(cur["geometry_feature_scale"])) > 1e-6:
#             mismatches.append(f"geometry_feature_scale: snapshot={old['geometry_feature_scale']!r}, current={cur['geometry_feature_scale']!r}")
#         if mismatches:
#             raise RuntimeError("Memory snapshot was built under a different feature contract: " + "; ".join(mismatches))

#     # ------------------------------------------------------------------
#     # Freezing / phase modes
#     # ------------------------------------------------------------------
#     def _set_requires_grad(self, module: Optional[nn.Module], value: bool) -> None:
#         if module is None:
#             return
#         for p in module.parameters():
#             p.requires_grad = bool(value)

#     def freeze_backbone_except_allowed(self, *, allow_last_blocks: bool = False, allow_projection: bool = False, allow_norm: Optional[bool] = None) -> None:
#         self._set_requires_grad(self.backbone, False)
#         if bool(allow_last_blocks) and hasattr(self.backbone, "get_last_blocks"):
#             for block in self.backbone.get_last_blocks():
#                 self._set_requires_grad(block, True)
#         self._set_requires_grad(self.projection, bool(allow_projection))
#         self._set_requires_grad(self.norm, bool(allow_projection if allow_norm is None else allow_norm))

#     def freeze_backbone_only(self) -> None:
#         self._set_requires_grad(self.backbone, False)

#     def unfreeze_backbone(self) -> None:
#         self._set_requires_grad(self.backbone, True)

#     def freeze_projection_head(self) -> None:
#         self._set_requires_grad(self.projection, False)
#         self._set_requires_grad(self.norm, False)

#     def unfreeze_projection_head(self) -> None:
#         self._set_requires_grad(self.projection, True)
#         self._set_requires_grad(self.norm, True)

#     def freeze_semantic_encoder(self) -> None:
#         self._set_requires_grad(self.semantic_encoder, False)
#         self._set_requires_grad(self.concept_encoder, False)

#     def freeze_classifier(self) -> None:
#         self._set_requires_grad(self.classifier, False)

#     def unfreeze_classifier(self) -> None:
#         self._set_requires_grad(self.classifier, True)

#     def freeze_classes(self, class_ids_or_count: Iterable[int] | int) -> None:
#         # GeometryBank.freeze_classes expects an iterable; freeze_classes_up_to
#         # expects a count. Keep both contracts explicit to avoid treating an int
#         # as an iterable during base handoff.
#         if isinstance(class_ids_or_count, int):
#             count = int(max(0, class_ids_or_count))
#             if hasattr(self.geometry_bank, "freeze_classes_up_to"):
#                 self.geometry_bank.freeze_classes_up_to(count)
#             elif hasattr(self.geometry_bank, "freeze_classes"):
#                 self.geometry_bank.freeze_classes(range(count))
#             return
#         ids = _ordered_unique_ints(class_ids_or_count)
#         if hasattr(self.geometry_bank, "freeze_classes"):
#             self.geometry_bank.freeze_classes(ids)
#         elif hasattr(self.geometry_bank, "freeze_classes_up_to"):
#             count = max(ids) + 1 if ids else 0
#             self.geometry_bank.freeze_classes_up_to(count)

#     def freeze_old_geometry_states(self, old_class_count: Optional[int] = None) -> None:
#         old = int(self.old_class_count if old_class_count is None else old_class_count)
#         self.old_class_count = old
#         self.freeze_classes(range(old))

#     def freeze_base_ce_head(self) -> None:
#         self._set_requires_grad(self.base_ce_head, False)

#     def unfreeze_base_ce_head(self) -> None:
#         self._set_requires_grad(self.base_ce_head, True)

#     def set_base_mode(self, *, train_backbone: bool = True, train_projection: bool = True) -> None:
#         self.current_phase = 0
#         self.old_class_count = 0
#         self.base_mode_active = True
#         self.incremental_mode_active = False
#         self.train()
#         self._set_requires_grad(self.backbone, bool(train_backbone))
#         self._set_requires_grad(self.projection, bool(train_projection))
#         self._set_requires_grad(self.norm, bool(train_projection))
#         self.freeze_semantic_encoder()
#         self.freeze_classifier()
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.eval()
#         if self.base_ce_head is not None:
#             self.unfreeze_base_ce_head()

#     def set_incremental_mode(
#         self,
#         *,
#         phase: Optional[int] = None,
#         old_class_count: Optional[int] = None,
#         train_classifier_calibration: bool = False,
#         train_geometry_adapter: Optional[bool] = None,
#     ) -> None:
#         if phase is not None:
#             self.current_phase = int(phase)
#         if old_class_count is not None:
#             self.old_class_count = int(old_class_count)
#         self.base_mode_active = False
#         self.incremental_mode_active = True

#         # Freeze backbone/projection and put them in eval mode to kill dropout.
#         self.freeze_backbone_except_allowed(allow_last_blocks=False, allow_projection=False)
#         self.backbone.eval()
#         self.projection.eval()
#         self.norm.eval()
#         self.freeze_semantic_encoder()
#         self.freeze_base_ce_head()
#         self.freeze_old_geometry_states(self.old_class_count)

#         if bool(train_classifier_calibration):
#             # Only classifier calibration parameters may be trainable if enabled.
#             self.unfreeze_classifier()
#         else:
#             self.freeze_classifier()
#         # PG-RGA main path: train only the bounded residual geometry adapter.
#         # Descriptor-only/frozen ablations keep it disabled.
#         if train_geometry_adapter is None:
#             train_geometry_adapter = bool(self.use_geometry_gated_adapter)
#         if bool(train_geometry_adapter):
#             self.unfreeze_geometry_plastic_adapter()
#         else:
#             self.freeze_geometry_plastic_adapter()
#         self._incremental_frozen_modules = ["backbone", "projection", "norm", "semantic_encoder", "concept_encoder"]

#     def assert_frozen_modules(self) -> None:
#         modules = {
#             "backbone": self.backbone,
#             "projection": self.projection,
#             "norm": self.norm,
#             "semantic_encoder": self.semantic_encoder,
#             "concept_encoder": self.concept_encoder,
#         }
#         bad_req: List[str] = []
#         bad_grad: List[str] = []
#         for prefix, module in modules.items():
#             if module is None:
#                 continue
#             for name, p in module.named_parameters():
#                 full = f"{prefix}.{name}"
#                 if p.requires_grad:
#                     bad_req.append(full)
#                 if p.grad is not None and torch.is_tensor(p.grad) and float(p.grad.detach().abs().sum().cpu().item()) != 0.0:
#                     bad_grad.append(full)
#         if bad_req:
#             raise RuntimeError(f"Frozen modules still have requires_grad=True: {bad_req[:20]}")
#         if bad_grad:
#             raise RuntimeError(f"Frozen modules have nonzero gradients: {bad_grad[:20]}")

#     # Legacy aliases kept safe, but routed to the bounded geometry adapter when
#     # the explicit geometry_gated_adapter ablation is selected.
#     def freeze_incremental_adapter(self) -> None:
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.eval()

#     def unfreeze_incremental_adapter(self) -> None:
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), True)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.train()

#     def disable_incremental_adapter(self) -> None:
#         self.use_incremental_adapter = False
#         self.freeze_incremental_adapter()

#     def enable_incremental_adapter(self) -> None:
#         self.use_geometry_gated_adapter = True
#         self.unfreeze_incremental_adapter()

#     def freeze_geometry_plastic_adapter(self) -> None:
#         self.freeze_incremental_adapter()

#     def unfreeze_geometry_plastic_adapter(self) -> None:
#         self.use_geometry_gated_adapter = True
#         self.unfreeze_incremental_adapter()

#     def adaptive_boundary_parameters(self) -> List[nn.Parameter]:
#         clf = getattr(self, "classifier", None)
#         if clf is not None and hasattr(clf, "boundary_parameters"):
#             return list(clf.boundary_parameters())
#         return []

#     def ensure_adaptive_boundary_capacity(self, class_count: int) -> None:
#         clf = getattr(self, "classifier", None)
#         if clf is not None and hasattr(clf, "ensure_class_capacity"):
#             try:
#                 clf.ensure_class_capacity(int(class_count))
#             except TypeError:
#                 pass
#         if clf is not None and hasattr(clf, "expand_to_seen_classes"):
#             try:
#                 clf.expand_to_seen_classes(list(range(int(class_count))))
#             except TypeError:
#                 pass

#     def adaptive_boundary_state(self, old_class_count: int = 0) -> Dict[str, float]:
#         clf = getattr(self, "classifier", None)
#         if clf is not None and hasattr(clf, "adaptive_boundary_state"):
#             try:
#                 return {k: float(v) for k, v in clf.adaptive_boundary_state(old_class_count=int(old_class_count)).items()}
#             except TypeError:
#                 try:
#                     return {k: float(v) for k, v in clf.adaptive_boundary_state(int(old_class_count)).items()}
#                 except Exception:
#                     pass
#         return {"old_class_count": float(old_class_count), "adaptive_boundary_available": float(bool(clf is not None and hasattr(clf, "boundary_parameters")))}

#     def freeze_geometry_calibrator(self) -> None: self.use_geometry_calibrator = False
#     def unfreeze_geometry_calibrator(self) -> None:
#         raise RuntimeError("Legacy geometry calibrator is disabled in the clean NECILModel.")
#     def freeze_energy_calibrator(self) -> None:
#         if hasattr(self.classifier, "freeze_all_adaptation"):
#             self.classifier.freeze_all_adaptation()
#     def unfreeze_energy_calibrator(self) -> None:
#         if hasattr(self.classifier, "unfreeze_all_adaptation"):
#             self.classifier.unfreeze_all_adaptation()

#     # ------------------------------------------------------------------
#     # Base CE head
#     # ------------------------------------------------------------------
#     def ensure_base_ce_head(self, num_base_classes: int, feature_dim: Optional[int] = None) -> nn.Linear:
#         num_base_classes = int(num_base_classes)
#         feature_dim = int(feature_dim or self.d_model)
#         if feature_dim != self.d_model:
#             raise RuntimeError(f"base CE head feature_dim must equal d_model={self.d_model}, got {feature_dim}")
#         if self.base_ce_head is None or self.base_ce_num_classes != num_base_classes:
#             self.base_ce_head = nn.Linear(self.d_model, num_base_classes).to(self.device)
#             nn.init.normal_(self.base_ce_head.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(self.base_ce_head.bias)
#             self.base_ce_num_classes = num_base_classes
#         return self.base_ce_head

#     def base_ce_logits(self, features: torch.Tensor, num_base_classes: Optional[int] = None) -> torch.Tensor:
#         features = self._validate_feature_tensor(features, "base_ce_logits.features")
#         if self.base_ce_head is None:
#             if num_base_classes is None:
#                 raise RuntimeError("base_ce_head is not initialized.")
#             self.ensure_base_ce_head(int(num_base_classes))
#         logits = self.base_ce_head(features)
#         if logits.dim() != 2 or logits.size(0) != features.size(0):
#             raise RuntimeError("base_ce_head returned invalid logits")
#         return logits

#     def forward_base_ce(self, x: torch.Tensor, num_base_classes: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         out = self.forward_features(
#             x,
#             spectral_summary=kwargs.get("spectral_summary", None),
#             band_weights=kwargs.get("band_weights", None),
#             spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
#         )
#         logits = self.base_ce_logits(out["features"], num_base_classes=int(num_base_classes))
#         out = dict(out)
#         out["base_logits"] = logits
#         out["logits"] = logits
#         return out

#     def drop_base_ce_head(self) -> None:
#         self.base_ce_head = None
#         self.base_ce_num_classes = 0

#     # PRL aliases from older trainer code.
#     ensure_base_prl_head = ensure_base_ce_head
#     base_prl_logits = base_ce_logits
#     def drop_base_prl_head(self) -> None: self.drop_base_ce_head()
#     def freeze_base_prl_head(self) -> None: self.freeze_base_ce_head()
#     def unfreeze_base_prl_head(self) -> None: self.unfreeze_base_ce_head()

#     def train(self, mode: bool = True):  # type: ignore[override]
#         super().train(mode)
#         if bool(getattr(self, "incremental_mode_active", False)):
#             # Frozen feature modules must remain deterministic during adapter
#             # training. Calling model.train() from the trainer should not turn
#             # dropout/batch statistics back on for backbone/projection.
#             self.backbone.eval()
#             self.projection.eval()
#             self.norm.eval()
#             if getattr(self, "geometry_plastic_adapter", None) is not None:
#                 self.geometry_plastic_adapter.train(mode and _module_has_trainable_params(self.geometry_plastic_adapter))
#         return self


#     # ------------------------------------------------------------------
#     # PG-RGA helper APIs used by the incremental trainer
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def sample_geometry_replay(
#         self,
#         class_ids: Iterable[int],
#         samples_per_class: int | Mapping[int, int] = 16,
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         label_to_local: Optional[Mapping[int, int]] = None,
#         parallel_scale: float = 1.0,
#         residual_scale: float = 0.25,
#         reliability_gated: bool = True,
#     ) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.geometry_bank, "sample_replay"):
#             raise RuntimeError("GeometryBank must expose sample_replay for PG-RGA old geometry replay.")
#         return self.geometry_bank.sample_replay(
#             class_ids,
#             samples_per_class=samples_per_class,
#             seen_classes=seen_classes,
#             label_to_local=label_to_local,
#             parallel_scale=float(parallel_scale),
#             residual_scale=float(residual_scale),
#             reliability_gated=bool(reliability_gated),
#         )

#     def compute_old_geometry_risk_features(
#         self,
#         features: torch.Tensor,
#         *,
#         old_class_count: Optional[int] = None,
#         geometry_bank: Optional[Any] = None,
#     ) -> Dict[str, torch.Tensor]:
#         z = self._validate_feature_tensor(features, "compute_old_geometry_risk_features.features")
#         old = int(self.old_class_count if old_class_count is None else old_class_count)
#         bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
#         bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
#         if not hasattr(self.classifier, "old_geometry_risk_features_from_bank"):
#             raise RuntimeError("Classifier must expose old_geometry_risk_features_from_bank for PG-RGA gating.")
#         return self.classifier.old_geometry_risk_features_from_bank(z, bank, old_class_count=old)

#     @torch.no_grad()
#     def assert_base_handoff_ready(self, base_class_ids: Iterable[int], *, freeze: bool = True, strict: bool = True) -> Dict[str, Any]:
#         ids = _ordered_unique_ints(base_class_ids)
#         if hasattr(self.geometry_bank, "assert_phase0_base_handoff_ready"):
#             result = self.geometry_bank.assert_phase0_base_handoff_ready(ids, freeze=bool(freeze), strict=bool(strict))
#         else:
#             if bool(freeze):
#                 self.freeze_classes(ids)
#             result = self.geometry_bank.assert_bank_valid(seen_classes=ids, strict=bool(strict))
#         self.old_class_count = len(ids)
#         self.current_num_classes = max(self.current_num_classes, max(ids) + 1 if ids else 0)
#         self.seen_classes = list(ids)
#         return result

#     def assert_pg_rga_contract(self, seen_classes: Iterable[int], *, phase: str = "base") -> None:
#         seen = _ordered_unique_ints(seen_classes)
#         self.assert_phase_ready(seen, mode="geometry", require_geometry=True)
#         if str(phase).lower().startswith("base"):
#             if self.incremental_mode_active:
#                 raise RuntimeError("Base contract violation: incremental_mode_active=True during base phase.")
#             if _module_has_trainable_params(self.geometry_plastic_adapter):
#                 raise RuntimeError("Base contract violation: geometry_plastic_adapter is trainable during base phase.")
#         else:
#             self.assert_frozen_modules()
#             if self.use_geometry_gated_adapter and not _module_has_trainable_params(self.geometry_plastic_adapter):
#                 raise RuntimeError("PG-RGA incremental contract violation: geometry_plastic_adapter is not trainable.")

#     # ------------------------------------------------------------------
#     # Assertions
#     # ------------------------------------------------------------------
#     def assert_phase_ready(self, seen_classes: Iterable[int], *, mode: str = "geometry", require_geometry: bool = True) -> None:
#         seen = _ordered_unique_ints(seen_classes)
#         if not seen:
#             raise RuntimeError("seen_classes is empty.")
#         if self.geometry_bank.d_model != self.d_model:
#             raise RuntimeError(f"GeometryBank d_model={self.geometry_bank.d_model} != model d_model={self.d_model}")
#         if _normalize_classifier_mode(mode, "geometry") in {"geometry", "calibrated_geometry"} and require_geometry:
#             if hasattr(self.geometry_bank, "assert_bank_valid"):
#                 self.geometry_bank.assert_bank_valid(seen_classes=seen, strict=True)
#             else:
#                 bank = self.get_subspace_bank()
#                 missing = [c for c in seen if c >= bank["sample_counts"].numel() or float(bank["sample_counts"][c].item()) <= 0.0]
#                 if missing:
#                     raise RuntimeError(f"Missing class geometry for seen classes: {missing}")
#         if self.incremental_mode_active:
#             # In incremental mode, frozen feature modules must stay eval to avoid dropout drift.
#             bad_train = []
#             for name in ("backbone", "projection"):
#                 m = getattr(self, name)
#                 if m.training:
#                     bad_train.append(name)
#             if bad_train:
#                 raise RuntimeError(f"Incremental frozen modules must be eval(), but these are train(): {bad_train}")
#         self.classifier.expand_to_seen_classes(seen)

#     def assert_no_missing_class_geometry(self, seen_classes: Iterable[int]) -> None:
#         if hasattr(self.geometry_bank, "assert_bank_valid"):
#             self.geometry_bank.assert_bank_valid(seen_classes=seen_classes, strict=True)

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------
#     def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
#         mode = _normalize_classifier_mode(kwargs.get("classifier_mode", kwargs.get("mode", "geometry")), "geometry")
#         return_features_only = _to_bool(kwargs.get("return_features_only", False), False)
#         seen_classes = kwargs.get("seen_classes", None)
#         old_classes = kwargs.get("old_classes", None)
#         new_classes = kwargs.get("new_classes", None)
#         targets = kwargs.get("targets", kwargs.get("labels", None))
#         targets_are_global = _to_bool(kwargs.get("targets_are_global", kwargs.get("labels_are_global", False)), False)
#         return_energy = _to_bool(kwargs.get("return_energy", False), False)
#         return_parts = _to_bool(kwargs.get("return_parts", False), False)
#         return_diagnostics = _to_bool(kwargs.get("return_diagnostics", False), False)

#         apply_adapter = kwargs.get("apply_adapter", None)
#         if mode == "base_ce":
#             apply_adapter = False
#         features_out = self.forward_features(
#             x,
#             spectral_summary=kwargs.get("spectral_summary", None),
#             band_weights=kwargs.get("band_weights", None),
#             spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
#             apply_adapter=apply_adapter,
#         )
#         if return_features_only or mode == "base_ce":
#             if mode == "base_ce" or "num_base_classes" in kwargs:
#                 nbase = int(kwargs.get("num_base_classes", self.base_ce_num_classes))
#                 if nbase <= 0:
#                     raise RuntimeError("num_base_classes is required for base_ce mode.")
#                 features_out = dict(features_out)
#                 features_out["logits"] = self.base_ce_logits(features_out["features"], nbase)
#             return features_out

#         if seen_classes is None:
#             seen_classes = self._infer_seen_classes(self.geometry_bank)
#         logits_out = self.compute_logits_from_features(
#             features_out["features"],
#             seen_classes=seen_classes,
#             geometry_bank=self.geometry_bank,
#             mode=mode,
#             targets=targets,
#             targets_are_global=targets_are_global,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )
#         out: Dict[str, Any] = dict(features_out)
#         if isinstance(logits_out, dict):
#             out.update(logits_out)
#         else:
#             out["logits"] = logits_out
#         return out
