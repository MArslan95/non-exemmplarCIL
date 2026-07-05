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
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for v in values:
        iv = int(v)
        if iv not in seen:
            out.append(iv)
            seen.add(iv)
    return out


def _normalize_classifier_mode(mode: Optional[str], default: str = "geometry") -> str:
    m = str(default if mode is None else mode).lower().strip()
    aliases = {
        "": default,
        "none": default,
        "geo": "geometry",
        "geometry_only": "geometry",
        "geometry-only": "geometry",
        "feature_geometry": "geometry",
        "low_rank_geometry": "geometry",
        "anchor": "geometry",
        "anchor_concept": "geometry",
        "anchor_concept_geometry": "geometry",
        "srgp": "geometry",
        "srgp_geometry": "geometry",
        "spectral_geometry": "geometry",
        "spectral_residual": "geometry",
        "calibrated_geometry": "calibrated_geometry",
        "topology_calibrated_geometry": "calibrated_geometry",
    }
    out = aliases.get(m, m)
    if out not in {"geometry", "calibrated_geometry", "base_ce"}:
        raise ValueError(f"Unsupported classifier mode {mode!r}. Use geometry, calibrated_geometry, or base_ce.")
    return out



def _normalize_incremental_update_mode(mode: Optional[str]) -> str:
    """Normalize incremental update mode for the clean PG-RGA architecture."""
    m = str(mode or "geometry_gated_adapter").lower().strip()
    aliases = {
        "": "geometry_gated_adapter",
        "none": "frozen_geometry",
        "false": "frozen_geometry",
        "off": "frozen_geometry",
        "geometry_gated": "geometry_gated_adapter",
        "gated_adapter": "geometry_gated_adapter",
        "pg_rga": "geometry_gated_adapter",
        "pgrga": "geometry_gated_adapter",
        "sgrga": "geometry_gated_adapter",
        "adapter": "geometry_gated_adapter",
        "descriptor": "descriptor_only",
        "descriptor_refinement": "descriptor_only",
        "frozen": "frozen_geometry",
        "frozen_geometry": "frozen_geometry",
    }
    out = aliases.get(m, m)
    if out not in {"geometry_gated_adapter", "descriptor_only", "frozen_geometry"}:
        raise ValueError(
            f"Unsupported incremental_update_mode={mode!r}. "
            "Use geometry_gated_adapter for the main PG-RGA path."
        )
    return out


def _module_has_trainable_params(module: Optional[nn.Module]) -> bool:
    return bool(module is not None and any(p.requires_grad for p in module.parameters()))


class GeometryPlasticAdapter(nn.Module):
    """Bounded geometry-gated residual adapter for controlled incremental plasticity.

    This module operates directly in the canonical projected GeometryBank space.
    It is deliberately small: it can make a bounded residual correction for new
    classes, but it cannot rewrite the backbone/projection or old bank rows.
    """

    def __init__(
        self,
        d_model: int,
        *,
        bottleneck: int = 32,
        max_scale: float = 0.10,
        dropout: float = 0.0,
        gate_bias_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.bottleneck = int(max(1, bottleneck))
        self.max_scale = float(max(0.0, max_scale))
        self.norm = nn.LayerNorm(self.d_model)
        self.delta = nn.Sequential(
            nn.Linear(self.d_model, self.bottleneck),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(self.bottleneck, self.d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, self.bottleneck),
            nn.GELU(),
            nn.Linear(self.bottleneck, 1),
        )
        self.reset_parameters(float(gate_bias_init))

    def reset_parameters(self, gate_bias_init: float = -3.0) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Start as an identity map. The adapter earns plasticity only through
        # incremental new/replay losses.
        last_delta = self.delta[-1]
        if isinstance(last_delta, nn.Linear):
            nn.init.zeros_(last_delta.weight)
            nn.init.zeros_(last_delta.bias)
        last_gate = self.gate[-1]
        if isinstance(last_gate, nn.Linear):
            nn.init.zeros_(last_gate.weight)
            nn.init.constant_(last_gate.bias, float(gate_bias_init))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.dim() != 2:
            raise RuntimeError(f"GeometryPlasticAdapter expects [B,D], got {tuple(features.shape)}")
        h = self.norm(features)
        gate = torch.sigmoid(self.gate(h))
        direction = torch.tanh(self.delta(h))
        delta = gate * float(self.max_scale) * direction
        adapted = features + delta
        return {
            "features": adapted,
            "projected_features": adapted,
            "delta": delta,
            "gate": gate,
            "adapter_delta": delta,
            "adapter_gate": gate,
        }


class NECILModel(nn.Module):
    """Clean NECIL-HSI model router.

    Contract:
        - One canonical projected feature space ``z`` is used for base geometry,
          incremental geometry insertion, synthetic replay, and evaluation.
        - GeometryBank is the only non-exemplar memory.
        - Classifier always receives explicit ``seen_classes`` and returns
          logits [B, len(seen_classes)].
        - Semantic/concept/adaptor/transport paths are not part of this clean
          main model. Their hooks are kept as no-ops or hard-disabled aliases so
          stale trainer code does not silently route through a different space.
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
        self.current_num_classes = 0
        self.seen_classes: List[int] = []
        self.base_mode_active = True
        self.incremental_mode_active = False
        self._incremental_frozen_modules: List[str] = []

        # Main architecture switch.
        # PG-RGA uses a bounded residual geometry adapter during incremental
        # learning, while the backbone/projection and old GeometryBank rows stay
        # frozen. Descriptor-only/frozen modes remain available as ablations.
        self.incremental_update_mode = _normalize_incremental_update_mode(
            getattr(args, "incremental_update_mode", None)
        )

        # Hard-disable stale/unsafe paths in the model object. They can exist in
        # other files for ablation compatibility, but this clean model must not
        # silently route through them.
        self.use_incremental_adapter = False
        self.use_geometry_calibrator = False
        self.use_bicyc_geometry_cycle = False
        self.use_geometry_transport = False
        self.use_sglat_transport = False
        self.use_geometry_gated_adapter = (
            self.incremental_update_mode == "geometry_gated_adapter"
            or _to_bool(getattr(args, "use_geometry_gated_adapter", False), False)
        )
        self.semantic_encoder: Optional[nn.Module] = None
        self.concept_encoder: Optional[nn.Module] = None

        self.backbone = SSMBackbone(args)
        dropout = float(getattr(args, "projection_dropout", 0.0))
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.norm = nn.LayerNorm(self.d_model)

        self.normalize_geometry_features = _to_bool(getattr(args, "normalize_geometry_features", True), True)
        raw_scale = float(getattr(args, "geometry_feature_scale", 0.0) or 0.0)
        self.geometry_feature_scale = raw_scale if raw_scale > 0.0 else math.sqrt(float(self.d_model))
        self.geometry_feature_clamp = float(getattr(args, "geometry_feature_clamp", 0.0) or 0.0)
        self.spectral_summary_mode = str(getattr(args, "spectral_summary_mode", "center")).lower().strip()
        if self.spectral_summary_mode not in {"center", "mean"}:
            raise ValueError("spectral_summary_mode must be 'center' or 'mean'.")
        pca_components = int(getattr(args, "pca_components", 0) or 0)
        self.default_spectral_physical = bool(getattr(args, "spectral_summary_is_physical", pca_components <= 0))
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
                "residual_variance_scale": float(getattr(args, "residual_variance_scale", 0.75)),
                "normalize_energy_by_dim": _to_bool(getattr(args, "energy_normalize_by_dim", True), True),
                "use_logdet_energy": _to_bool(getattr(args, "use_logdet_energy", True), True),
                "logdet_energy_weight": float(getattr(args, "logdet_energy_weight", 0.05)),
                "use_reliability_penalty": _to_bool(getattr(args, "use_reliability_penalty", True), True),
                "reliability_energy_weight": float(getattr(args, "reliability_energy_weight", 0.03)),
                "use_old_new_calibration": _to_bool(getattr(args, "use_old_new_calibration", getattr(args, "use_energy_calibrator", False)), False),
                "calibration_max_abs_bias": float(getattr(args, "calibration_max_abs_bias", getattr(args, "energy_calibrator_max_bias", 1.0))),
                "logit_clip": float(getattr(args, "geometry_logit_clip", 0.0)),
            },
        )
        self.classifier = GeometryEnergyClassifier(**clf_kwargs)

        self.geometry_plastic_adapter = GeometryPlasticAdapter(
            self.d_model,
            bottleneck=int(getattr(args, "adapter_bottleneck", 32)),
            max_scale=float(getattr(args, "adapter_max_scale", 0.10)),
            dropout=float(getattr(args, "adapter_dropout", 0.0)),
            gate_bias_init=float(getattr(args, "adapter_gate_bias_init", -3.0)),
        )
        self._set_requires_grad(self.geometry_plastic_adapter, False)

        self.base_ce_head: Optional[nn.Linear] = None
        self.base_ce_num_classes = 0

        self.to(self.device)

    # ------------------------------------------------------------------
    # Input / feature utilities
    # ------------------------------------------------------------------
    def _validate_feature_tensor(self, features: torch.Tensor, name: str, batch_size: Optional[int] = None) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(features)}")
        if features.dim() != 2:
            raise RuntimeError(f"{name} must be [B,D], got {tuple(features.shape)}")
        if features.size(1) != self.d_model:
            raise RuntimeError(f"{name} dim mismatch: expected {self.d_model}, got {features.size(1)}")
        if batch_size is not None and features.size(0) != int(batch_size):
            raise RuntimeError(f"{name} batch mismatch: got {features.size(0)}, expected {int(batch_size)}")
        if not torch.isfinite(features).all():
            raise RuntimeError(f"{name} contains NaN/Inf values.")
        return features

    def _canonicalize(self, z: torch.Tensor, *, name: str) -> torch.Tensor:
        z = self._validate_feature_tensor(z, name)
        if self.normalize_geometry_features:
            z = F.normalize(z, p=2, dim=1, eps=1e-8) * float(self.geometry_feature_scale)
        if self.geometry_feature_clamp > 0.0:
            z = z.clamp(-self.geometry_feature_clamp, self.geometry_feature_clamp)
        if not torch.isfinite(z).all():
            raise RuntimeError(f"{name} contains NaN/Inf after canonicalization.")
        return z

    def _center_spectrum_from_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if self.spectral_summary_mode == "center":
                return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
            return x.mean(dim=(-1, -2))
        if x.dim() == 3:
            if self.spectral_summary_mode == "center":
                return x[:, :, x.size(-1) // 2]
            return x.mean(dim=-1)
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
            s = self._center_spectrum_from_input(x).to(device=features.device, dtype=features.dtype)
            physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
        else:
            s = spectral_summary.to(device=features.device, dtype=features.dtype)
            if s.dim() == 4:
                s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
            elif s.dim() == 3:
                if s.size(0) == features.size(0) and s.size(-1) > 1:
                    s = s[:, :, s.size(-1) // 2]
                else:
                    s = s.reshape(features.size(0), -1)
            elif s.dim() == 1:
                if s.numel() % max(int(features.size(0)), 1) != 0:
                    raise RuntimeError(f"1-D spectral_summary cannot be reshaped to batch size {features.size(0)}")
                s = s.reshape(features.size(0), -1)
            elif s.dim() > 4:
                s = s.flatten(1)
            physical = self.default_spectral_physical if spectral_summary_is_physical is None else bool(spectral_summary_is_physical)
        if s.dim() != 2 or s.size(0) != features.size(0):
            raise RuntimeError(f"spectral_summary must resolve to [B,S], got {tuple(s.shape)}")
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if s.size(1) == 0:
            physical = False
        return s, bool(physical)

    def _band_summary(self, spectral_summary: torch.Tensor, band_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if band_weights is not None and torch.is_tensor(band_weights) and band_weights.numel() > 0:
            bw = band_weights.to(device=spectral_summary.device, dtype=spectral_summary.dtype)
            if bw.dim() == 2 and bw.size(0) == spectral_summary.size(0) and bw.size(1) == spectral_summary.size(1):
                bw = torch.nan_to_num(bw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                denom = bw.sum(dim=1, keepdim=True)
                if bool((denom > self.min_band_mass).all().item()):
                    return bw / denom.clamp_min(self.min_band_mass)
        b = spectral_summary.abs()
        denom = b.sum(dim=1, keepdim=True)
        uniform = torch.full_like(b, 1.0 / float(max(int(b.size(1)), 1)))
        b = torch.where(denom > self.min_band_mass, b / denom.clamp_min(self.min_band_mass), uniform)
        return b

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Backbone-only feature extraction. Does not touch GeometryBank/classifier."""
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        out = self.backbone(x.to(self.device))
        if isinstance(out, dict):
            if "features" not in out:
                raise RuntimeError("backbone output dict must contain key 'features'.")
            features = self._validate_feature_tensor(out["features"], "backbone.features", int(x.size(0)))
            result = dict(out)
            result["features"] = features
            result.setdefault("backbone_features", features)
            return result
        features = self._validate_feature_tensor(out, "backbone.features", int(x.size(0)))
        return {"features": features, "backbone_features": features}

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        spectral_summary: Optional[torch.Tensor] = None,
        band_weights: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: Optional[bool] = None,
        apply_adapter: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return projected geometry features.

        Important contract:
            - ``canonical_features`` are always the backbone/projection z-space
              used to build the base GeometryBank.
            - ``features`` / ``projected_features`` are the scoring features.
              They equal canonical z in base mode and become adapted z only when
              PG-RGA adapter routing is explicitly active.
        """
        raw = self.extract_features(x)
        h = self._validate_feature_tensor(raw["features"], "preproject_features", int(x.size(0)))
        z_canonical = self.norm(self.projection(h) + h)
        z_canonical = self._canonicalize(z_canonical, name="canonical_geometry_features")

        use_adapter = self._adapter_runtime_enabled(force=False) if apply_adapter is None else bool(apply_adapter)
        z = z_canonical
        adapter_out: Optional[Dict[str, torch.Tensor]] = None
        if use_adapter:
            adapter_out = self.adapt_projected_features(
                z_canonical,
                force=(apply_adapter is True),
                return_delta=True,
                recanonicalize=True,
            )
            z = adapter_out["features"]

        s, physical = self._prepare_spectral_summary(
            x.to(self.device),
            z,
            spectral_summary=spectral_summary,
            spectral_summary_is_physical=spectral_summary_is_physical,
        )
        candidate_bw = band_weights if band_weights is not None else raw.get("band_weights", None)
        band = self._band_summary(s, candidate_bw)
        result = {
            "features": z,
            "projected_features": z,
            "geometry_features": z,
            "canonical_features": z_canonical,
            "canonical_projected_features": z_canonical,
            "pre_adapter_features": z_canonical,
            "preproject_features": h,
            "backbone_features": h,
            "spectral_summary": s,
            "spectral_summary_is_physical": torch.tensor(bool(physical), device=z.device, dtype=torch.bool),
            "band_summary": band,
            "band_importance": band,
            "band_weights": candidate_bw if torch.is_tensor(candidate_bw) else None,
            "spectral_features": raw.get("spectral_features", h),
            "spatial_features": raw.get("spatial_features", h),
        }
        if adapter_out is not None:
            result["adapter_delta"] = adapter_out["adapter_delta"]
            result["adapter_gate"] = adapter_out["adapter_gate"]
            result["adapter_active"] = adapter_out["adapter_active"]
        return result

    # Compatibility aliases expected by existing trainers.
    def extract_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.forward_features(x, **kwargs)

    def extract_canonical_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        kwargs = dict(kwargs)
        kwargs["apply_adapter"] = False
        return self.forward_features(x, **kwargs)

    def extract_adapted_projected_features(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        kwargs = dict(kwargs)
        kwargs["apply_adapter"] = True
        return self.forward_features(x, **kwargs)

    def extract_geometry_features(self, x: torch.Tensor, *, return_dict: bool = False, **kwargs: Any):
        out = self.forward_features(x, **kwargs)
        return out if bool(return_dict) else out["features"]

    @torch.no_grad()
    def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)

    # ------------------------------------------------------------------
    # Controlled geometry-gated adapter
    # ------------------------------------------------------------------
    def _adapter_runtime_enabled(self, *, force: bool = False) -> bool:
        return bool(
            getattr(self, "geometry_plastic_adapter", None) is not None
            and (bool(force) or (bool(getattr(self, "use_geometry_gated_adapter", False)) and bool(self.incremental_mode_active)))
        )

    def adapt_projected_features(
        self,
        features: torch.Tensor,
        *,
        force: bool = False,
        return_delta: bool = True,
        recanonicalize: bool = True,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        """Apply the bounded adapter directly to canonical z-space features.

        This is required for synthetic old replay: replay samples already live
        in GeometryBank space and cannot pass through the backbone. The adapter
        must still see them so replay CE can close old-region gates.
        """
        z0 = self._validate_feature_tensor(features, "adapt_projected_features.features")
        if not self._adapter_runtime_enabled(force=force):
            zero_delta = torch.zeros_like(z0)
            zero_gate = torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype)
            return {
                "features": z0,
                "projected_features": z0,
                "geometry_features": z0,
                "delta": zero_delta,
                "gate": zero_gate,
                "adapter_delta": zero_delta,
                "adapter_gate": zero_gate,
                "adapter_active": torch.tensor(False, device=z0.device),
            }
        raw = self.geometry_plastic_adapter(z0)
        z = raw["features"]
        if bool(recanonicalize):
            z = self._canonicalize(z, name="adapted_geometry_features")
        delta = z - z0
        gate = raw.get("gate", torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype))
        return {
            "features": z,
            "projected_features": z,
            "geometry_features": z,
            "delta": delta,
            "gate": gate,
            "adapter_delta": delta,
            "adapter_gate": gate,
            "adapter_active": torch.tensor(True, device=z0.device),
        }

    # ------------------------------------------------------------------
    # Seen classes / bank helpers
    # ------------------------------------------------------------------
    def _infer_seen_classes(self, geometry_bank: Optional[Any] = None) -> List[int]:
        bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
        if hasattr(bank_obj, "get_valid_mask"):
            valid = bank_obj.get_valid_mask().detach().cpu().flatten()
            return [int(i) for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
        bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
        if not isinstance(bank, dict) or "sample_counts" not in bank:
            return list(self.seen_classes)
        counts = bank["sample_counts"].detach().cpu().flatten()
        return [int(i) for i in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()]

    def ensure_class_capacity(self, class_count: int, spectral_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
        count = int(max(0, class_count))
        dtype = dtype or next(self.parameters()).dtype
        if hasattr(self.geometry_bank, "ensure_class_count"):
            self.geometry_bank.ensure_class_count(count, spectral_dim=int(spectral_dim), dtype=dtype)
        elif hasattr(self.geometry_bank, "ensure_num_classes"):
            self.geometry_bank.ensure_num_classes(count)
        self.current_num_classes = max(self.current_num_classes, count)

    def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
        if not hasattr(self.geometry_bank, "get_bank"):
            raise RuntimeError("GeometryBank must expose get_bank().")
        bank = self.geometry_bank.get_bank()
        if "variances" not in bank and "eigvals" in bank and "res_vars" in bank:
            bank = dict(bank)
            bank["variances"] = torch.cat([bank["eigvals"], bank["res_vars"].unsqueeze(-1)], dim=-1)
        if "resvars" not in bank and "res_vars" in bank:
            bank = dict(bank)
            bank["resvars"] = bank["res_vars"]
        return bank

    def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
        old = int(self.old_class_count if old_class_count is None else old_class_count)
        bank = self.get_subspace_bank()
        old = max(0, min(old, int(bank["means"].size(0))))
        out = {}
        for k, v in bank.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.size(0) >= old:
                out[k] = v[:old]
        return out

    # ------------------------------------------------------------------
    # Classifier routing
    # ------------------------------------------------------------------
    def forward_classifier(
        self,
        features: torch.Tensor,
        seen_classes: Iterable[int],
        mode: str = "geometry",
        *,
        geometry_bank: Optional[Any] = None,
        targets: Optional[torch.Tensor] = None,
        targets_are_global: bool = False,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        return_energy: bool = False,
        return_parts: bool = False,
        return_diagnostics: bool = False,
    ):
        return self.compute_logits_from_features(
            features=features,
            seen_classes=seen_classes,
            geometry_bank=geometry_bank,
            mode=mode,
            targets=targets,
            targets_are_global=targets_are_global,
            old_classes=old_classes,
            new_classes=new_classes,
            return_energy=return_energy,
            return_parts=return_parts,
            return_diagnostics=return_diagnostics,
        )

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
        features = self._validate_feature_tensor(features, "compute_logits_from_features.features")
        bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
        seen = _ordered_unique_ints(seen_classes if seen_classes is not None else self._infer_seen_classes(bank_obj))
        if not seen:
            raise RuntimeError("seen_classes is empty. Build GeometryBank rows or pass seen_classes explicitly.")
        mode_norm = _normalize_classifier_mode(classifier_mode if classifier_mode is not None else mode, "geometry")
        self.assert_phase_ready(seen, mode=mode_norm, require_geometry=True)
        out = self.classifier(
            features,
            seen_classes=seen,
            geometry_bank=bank_obj,
            mode=mode_norm,
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
        self.classifier.assert_logits_valid(
            logits,
            seen_classes=seen,
            targets=(self.classifier.global_to_local_labels(targets, seen) if targets is not None and targets_are_global else targets),
            old_classes=old_classes,
            new_classes=new_classes,
            context="NECILModel.compute_logits_from_features",
        )
        self.current_num_classes = max(self.current_num_classes, max(seen) + 1)
        self.seen_classes = list(seen)
        return out

    def compute_energy_from_features(self, features: torch.Tensor, seen_classes: Optional[Iterable[int]] = None, **kwargs: Any):
        out = self.compute_logits_from_features(features, seen_classes=seen_classes, return_energy=True, **kwargs)
        if not isinstance(out, dict) or "energy" not in out:
            raise RuntimeError("Expected dict with energy from compute_logits_from_features(return_energy=True).")
        return out["energy"]

    # ------------------------------------------------------------------
    # Geometry refresh / memory snapshots
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
        features = self._validate_feature_tensor(features, "refresh_geometry_for_classes.features")
        labels = labels.to(device=features.device).long().flatten()
        ids = _ordered_unique_ints(class_ids)
        if labels.numel() != features.size(0):
            raise RuntimeError("labels/features batch mismatch in refresh_geometry_for_classes")
        bad = sorted(set(int(v) for v in torch.unique(labels).detach().cpu().tolist()).difference(set(ids)))
        if bad:
            raise RuntimeError(f"labels contain classes outside requested refresh ids: bad={bad}, class_ids={ids}")
        band_dim = 0
        if spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0:
            band_dim = int(spectral_summary.reshape(features.size(0), -1).size(1))
        elif band_weights is not None and torch.is_tensor(band_weights) and band_weights.numel() > 0:
            band_dim = int(band_weights.reshape(features.size(0), -1).size(1))
        self.ensure_class_capacity(max(ids) + 1 if ids else 0, spectral_dim=band_dim, dtype=features.dtype)
        rows = self.geometry_bank.extract_geometry(
            features,
            labels,
            spectral_summary=spectral_summary,
            band_weights=band_weights,
            spectral_summary_is_physical=bool(spectral_summary_is_physical),
        )
        committed: List[int] = []
        for c in ids:
            if c not in rows:
                raise RuntimeError(f"No geometry could be extracted for class {c}.")
            row = rows[c]
            self.geometry_bank.add_or_update_class_geometry(
                c,
                mean=row["mean"],
                basis=row["basis"],
                eigvals=row["eigvals"],
                res_var=row["res_var"],
                spectral_prototype=row.get("spectral_prototype"),
                band_importance=row.get("band_importance"),
                sample_count=row.get("sample_count"),
                active_rank=row.get("active_rank"),
                reliability=row.get("reliability"),
                feature_reliability=row.get("feature_reliability"),
                band_reliability=row.get("band_reliability"),
                spectral_reliability=row.get("spectral_reliability"),
                phase_created=int(self.current_phase if phase_created is None else phase_created),
                allow_frozen_update=False,
            )
            committed.append(c)
        if bool(freeze_after):
            self.freeze_classes(committed)
        self.geometry_bank.assert_bank_valid(seen_classes=committed, strict=True)
        self.current_num_classes = max(self.current_num_classes, max(committed) + 1 if committed else self.current_num_classes)
        return {"committed_class_ids": committed, "phase_created": int(self.current_phase if phase_created is None else phase_created)}

    @torch.no_grad()
    def refresh_class_subspace(self, cls: int, mean: torch.Tensor, basis: torch.Tensor, eigvals: torch.Tensor, res_var: Optional[torch.Tensor] = None, resvar: Optional[torch.Tensor] = None, **kwargs: Any) -> None:
        rv = res_var if res_var is not None else resvar
        if rv is None:
            raise ValueError("refresh_class_subspace requires res_var/resvar.")
        c = int(cls)
        self.ensure_class_capacity(c + 1)
        self.geometry_bank.add_or_update_class_geometry(
            c,
            mean=mean,
            basis=basis,
            eigvals=eigvals,
            res_var=rv,
            spectral_prototype=kwargs.get("spectral_prototype", kwargs.get("spectral_proto", None)),
            band_importance=kwargs.get("band_importance", None),
            sample_count=kwargs.get("sample_count", None),
            active_rank=kwargs.get("active_rank", None),
            reliability=kwargs.get("reliability", None),
            feature_reliability=kwargs.get("feature_reliability", None),
            band_reliability=kwargs.get("band_reliability", None),
            phase_created=kwargs.get("phase_created", self.current_phase),
            allow_frozen_update=bool(kwargs.get("allow_frozen_update", False)),
        )

    @torch.no_grad()
    def export_memory_snapshot(self) -> Dict[str, Any]:
        snap = self.geometry_bank.export_snapshot()
        snap.update(
            {
                "current_phase": int(self.current_phase),
                "old_class_count": int(self.old_class_count),
                "current_num_classes": int(self.current_num_classes),
                "seen_classes": list(self.seen_classes),
                "feature_contract": self.feature_contract(),
            }
        )
        return snap

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
        self.current_num_classes = int(snapshot.get("current_num_classes", len(self.geometry_bank)))
        self.seen_classes = [int(c) for c in snapshot.get("seen_classes", self._infer_seen_classes())]

    def feature_contract(self) -> Dict[str, Any]:
        return {
            "d_model": int(self.d_model),
            "subspace_rank": int(self.subspace_rank),
            "normalize_geometry_features": bool(self.normalize_geometry_features),
            "geometry_feature_scale": float(self.geometry_feature_scale),
            "spectral_summary_mode": str(self.spectral_summary_mode),
            "classifier_contract": "logits[B,len(seen_classes)]",
            "incremental_update_mode": str(self.incremental_update_mode),
            "geometry_gated_adapter_available": hasattr(self, "geometry_plastic_adapter"),
            "geometry_gated_adapter_enabled": bool(self.use_geometry_gated_adapter),
            "semantic_encoder_enabled": False,
            "concept_encoder_enabled": False,
        }

    def _assert_snapshot_feature_contract(self, snapshot: Dict[str, Any], *, strict: bool) -> None:
        if not strict:
            return
        old = snapshot.get("feature_contract", snapshot.get("geometry_feature_contract", None))
        if not isinstance(old, dict):
            return
        cur = self.feature_contract()
        mismatches = []
        for k in ("d_model", "subspace_rank", "normalize_geometry_features", "spectral_summary_mode"):
            if k in old and old[k] != cur[k]:
                mismatches.append(f"{k}: snapshot={old[k]!r}, current={cur[k]!r}")
        if "geometry_feature_scale" in old and abs(float(old["geometry_feature_scale"]) - float(cur["geometry_feature_scale"])) > 1e-6:
            mismatches.append(f"geometry_feature_scale: snapshot={old['geometry_feature_scale']!r}, current={cur['geometry_feature_scale']!r}")
        if mismatches:
            raise RuntimeError("Memory snapshot was built under a different feature contract: " + "; ".join(mismatches))

    # ------------------------------------------------------------------
    # Freezing / phase modes
    # ------------------------------------------------------------------
    def _set_requires_grad(self, module: Optional[nn.Module], value: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = bool(value)

    def freeze_backbone_except_allowed(self, *, allow_last_blocks: bool = False, allow_projection: bool = False, allow_norm: Optional[bool] = None) -> None:
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
        self._set_requires_grad(self.classifier, True)

    def freeze_classes(self, class_ids_or_count: Iterable[int] | int) -> None:
        # GeometryBank.freeze_classes expects an iterable; freeze_classes_up_to
        # expects a count. Keep both contracts explicit to avoid treating an int
        # as an iterable during base handoff.
        if isinstance(class_ids_or_count, int):
            count = int(max(0, class_ids_or_count))
            if hasattr(self.geometry_bank, "freeze_classes_up_to"):
                self.geometry_bank.freeze_classes_up_to(count)
            elif hasattr(self.geometry_bank, "freeze_classes"):
                self.geometry_bank.freeze_classes(range(count))
            return
        ids = _ordered_unique_ints(class_ids_or_count)
        if hasattr(self.geometry_bank, "freeze_classes"):
            self.geometry_bank.freeze_classes(ids)
        elif hasattr(self.geometry_bank, "freeze_classes_up_to"):
            count = max(ids) + 1 if ids else 0
            self.geometry_bank.freeze_classes_up_to(count)

    def freeze_old_geometry_states(self, old_class_count: Optional[int] = None) -> None:
        old = int(self.old_class_count if old_class_count is None else old_class_count)
        self.old_class_count = old
        self.freeze_classes(range(old))

    def freeze_base_ce_head(self) -> None:
        self._set_requires_grad(self.base_ce_head, False)

    def unfreeze_base_ce_head(self) -> None:
        self._set_requires_grad(self.base_ce_head, True)

    def set_base_mode(self, *, train_backbone: bool = True, train_projection: bool = True) -> None:
        self.current_phase = 0
        self.old_class_count = 0
        self.base_mode_active = True
        self.incremental_mode_active = False
        self.train()
        self._set_requires_grad(self.backbone, bool(train_backbone))
        self._set_requires_grad(self.projection, bool(train_projection))
        self._set_requires_grad(self.norm, bool(train_projection))
        self.freeze_semantic_encoder()
        self.freeze_classifier()
        self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
        if getattr(self, "geometry_plastic_adapter", None) is not None:
            self.geometry_plastic_adapter.eval()
        if self.base_ce_head is not None:
            self.unfreeze_base_ce_head()

    def set_incremental_mode(
        self,
        *,
        phase: Optional[int] = None,
        old_class_count: Optional[int] = None,
        train_classifier_calibration: bool = False,
        train_geometry_adapter: Optional[bool] = None,
    ) -> None:
        if phase is not None:
            self.current_phase = int(phase)
        if old_class_count is not None:
            self.old_class_count = int(old_class_count)
        self.base_mode_active = False
        self.incremental_mode_active = True

        # Freeze backbone/projection and put them in eval mode to kill dropout.
        self.freeze_backbone_except_allowed(allow_last_blocks=False, allow_projection=False)
        self.backbone.eval()
        self.projection.eval()
        self.norm.eval()
        self.freeze_semantic_encoder()
        self.freeze_base_ce_head()
        self.freeze_old_geometry_states(self.old_class_count)

        if bool(train_classifier_calibration):
            # Only classifier calibration parameters may be trainable if enabled.
            self.unfreeze_classifier()
        else:
            self.freeze_classifier()
        # PG-RGA main path: train only the bounded residual geometry adapter.
        # Descriptor-only/frozen ablations keep it disabled.
        if train_geometry_adapter is None:
            train_geometry_adapter = bool(self.use_geometry_gated_adapter)
        if bool(train_geometry_adapter):
            self.unfreeze_geometry_plastic_adapter()
        else:
            self.freeze_geometry_plastic_adapter()
        self._incremental_frozen_modules = ["backbone", "projection", "norm", "semantic_encoder", "concept_encoder"]

    def assert_frozen_modules(self) -> None:
        modules = {
            "backbone": self.backbone,
            "projection": self.projection,
            "norm": self.norm,
            "semantic_encoder": self.semantic_encoder,
            "concept_encoder": self.concept_encoder,
        }
        bad_req: List[str] = []
        bad_grad: List[str] = []
        for prefix, module in modules.items():
            if module is None:
                continue
            for name, p in module.named_parameters():
                full = f"{prefix}.{name}"
                if p.requires_grad:
                    bad_req.append(full)
                if p.grad is not None and torch.is_tensor(p.grad) and float(p.grad.detach().abs().sum().cpu().item()) != 0.0:
                    bad_grad.append(full)
        if bad_req:
            raise RuntimeError(f"Frozen modules still have requires_grad=True: {bad_req[:20]}")
        if bad_grad:
            raise RuntimeError(f"Frozen modules have nonzero gradients: {bad_grad[:20]}")

    # Legacy aliases kept safe, but routed to the bounded geometry adapter when
    # the explicit geometry_gated_adapter ablation is selected.
    def freeze_incremental_adapter(self) -> None:
        self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
        if getattr(self, "geometry_plastic_adapter", None) is not None:
            self.geometry_plastic_adapter.eval()

    def unfreeze_incremental_adapter(self) -> None:
        self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), True)
        if getattr(self, "geometry_plastic_adapter", None) is not None:
            self.geometry_plastic_adapter.train()

    def disable_incremental_adapter(self) -> None:
        self.use_incremental_adapter = False
        self.freeze_incremental_adapter()

    def enable_incremental_adapter(self) -> None:
        self.use_geometry_gated_adapter = True
        self.unfreeze_incremental_adapter()

    def freeze_geometry_plastic_adapter(self) -> None:
        self.freeze_incremental_adapter()

    def unfreeze_geometry_plastic_adapter(self) -> None:
        self.use_geometry_gated_adapter = True
        self.unfreeze_incremental_adapter()

    def adaptive_boundary_parameters(self) -> List[nn.Parameter]:
        clf = getattr(self, "classifier", None)
        if clf is not None and hasattr(clf, "boundary_parameters"):
            return list(clf.boundary_parameters())
        return []

    def ensure_adaptive_boundary_capacity(self, class_count: int) -> None:
        clf = getattr(self, "classifier", None)
        if clf is not None and hasattr(clf, "ensure_class_capacity"):
            try:
                clf.ensure_class_capacity(int(class_count))
            except TypeError:
                pass
        if clf is not None and hasattr(clf, "expand_to_seen_classes"):
            try:
                clf.expand_to_seen_classes(list(range(int(class_count))))
            except TypeError:
                pass

    def adaptive_boundary_state(self, old_class_count: int = 0) -> Dict[str, float]:
        clf = getattr(self, "classifier", None)
        if clf is not None and hasattr(clf, "adaptive_boundary_state"):
            try:
                return {k: float(v) for k, v in clf.adaptive_boundary_state(old_class_count=int(old_class_count)).items()}
            except TypeError:
                try:
                    return {k: float(v) for k, v in clf.adaptive_boundary_state(int(old_class_count)).items()}
                except Exception:
                    pass
        return {"old_class_count": float(old_class_count), "adaptive_boundary_available": float(bool(clf is not None and hasattr(clf, "boundary_parameters")))}

    def freeze_geometry_calibrator(self) -> None: self.use_geometry_calibrator = False
    def unfreeze_geometry_calibrator(self) -> None:
        raise RuntimeError("Legacy geometry calibrator is disabled in the clean NECILModel.")
    def freeze_energy_calibrator(self) -> None:
        if hasattr(self.classifier, "freeze_all_adaptation"):
            self.classifier.freeze_all_adaptation()
    def unfreeze_energy_calibrator(self) -> None:
        if hasattr(self.classifier, "unfreeze_all_adaptation"):
            self.classifier.unfreeze_all_adaptation()

    # ------------------------------------------------------------------
    # Base CE head
    # ------------------------------------------------------------------
    def ensure_base_ce_head(self, num_base_classes: int, feature_dim: Optional[int] = None) -> nn.Linear:
        num_base_classes = int(num_base_classes)
        feature_dim = int(feature_dim or self.d_model)
        if feature_dim != self.d_model:
            raise RuntimeError(f"base CE head feature_dim must equal d_model={self.d_model}, got {feature_dim}")
        if self.base_ce_head is None or self.base_ce_num_classes != num_base_classes:
            self.base_ce_head = nn.Linear(self.d_model, num_base_classes).to(self.device)
            nn.init.normal_(self.base_ce_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.base_ce_head.bias)
            self.base_ce_num_classes = num_base_classes
        return self.base_ce_head

    def base_ce_logits(self, features: torch.Tensor, num_base_classes: Optional[int] = None) -> torch.Tensor:
        features = self._validate_feature_tensor(features, "base_ce_logits.features")
        if self.base_ce_head is None:
            if num_base_classes is None:
                raise RuntimeError("base_ce_head is not initialized.")
            self.ensure_base_ce_head(int(num_base_classes))
        logits = self.base_ce_head(features)
        if logits.dim() != 2 or logits.size(0) != features.size(0):
            raise RuntimeError("base_ce_head returned invalid logits")
        return logits

    def forward_base_ce(self, x: torch.Tensor, num_base_classes: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        out = self.forward_features(
            x,
            spectral_summary=kwargs.get("spectral_summary", None),
            band_weights=kwargs.get("band_weights", None),
            spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
        )
        logits = self.base_ce_logits(out["features"], num_base_classes=int(num_base_classes))
        out = dict(out)
        out["base_logits"] = logits
        out["logits"] = logits
        return out

    def drop_base_ce_head(self) -> None:
        self.base_ce_head = None
        self.base_ce_num_classes = 0

    # PRL aliases from older trainer code.
    ensure_base_prl_head = ensure_base_ce_head
    base_prl_logits = base_ce_logits
    def drop_base_prl_head(self) -> None: self.drop_base_ce_head()
    def freeze_base_prl_head(self) -> None: self.freeze_base_ce_head()
    def unfreeze_base_prl_head(self) -> None: self.unfreeze_base_ce_head()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if bool(getattr(self, "incremental_mode_active", False)):
            # Frozen feature modules must remain deterministic during adapter
            # training. Calling model.train() from the trainer should not turn
            # dropout/batch statistics back on for backbone/projection.
            self.backbone.eval()
            self.projection.eval()
            self.norm.eval()
            if getattr(self, "geometry_plastic_adapter", None) is not None:
                self.geometry_plastic_adapter.train(mode and _module_has_trainable_params(self.geometry_plastic_adapter))
        return self


    # ------------------------------------------------------------------
    # PG-RGA helper APIs used by the incremental trainer
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_geometry_replay(
        self,
        class_ids: Iterable[int],
        samples_per_class: int | Mapping[int, int] = 16,
        *,
        seen_classes: Optional[Iterable[int]] = None,
        label_to_local: Optional[Mapping[int, int]] = None,
        parallel_scale: float = 1.0,
        residual_scale: float = 0.25,
        reliability_gated: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(self.geometry_bank, "sample_replay"):
            raise RuntimeError("GeometryBank must expose sample_replay for PG-RGA old geometry replay.")
        return self.geometry_bank.sample_replay(
            class_ids,
            samples_per_class=samples_per_class,
            seen_classes=seen_classes,
            label_to_local=label_to_local,
            parallel_scale=float(parallel_scale),
            residual_scale=float(residual_scale),
            reliability_gated=bool(reliability_gated),
        )

    def compute_old_geometry_risk_features(
        self,
        features: torch.Tensor,
        *,
        old_class_count: Optional[int] = None,
        geometry_bank: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self._validate_feature_tensor(features, "compute_old_geometry_risk_features.features")
        old = int(self.old_class_count if old_class_count is None else old_class_count)
        bank_obj = self.geometry_bank if geometry_bank is None else geometry_bank
        bank = bank_obj.get_bank() if hasattr(bank_obj, "get_bank") else bank_obj
        if not hasattr(self.classifier, "old_geometry_risk_features_from_bank"):
            raise RuntimeError("Classifier must expose old_geometry_risk_features_from_bank for PG-RGA gating.")
        return self.classifier.old_geometry_risk_features_from_bank(z, bank, old_class_count=old)

    @torch.no_grad()
    def assert_base_handoff_ready(self, base_class_ids: Iterable[int], *, freeze: bool = True, strict: bool = True) -> Dict[str, Any]:
        ids = _ordered_unique_ints(base_class_ids)
        if hasattr(self.geometry_bank, "assert_phase0_base_handoff_ready"):
            result = self.geometry_bank.assert_phase0_base_handoff_ready(ids, freeze=bool(freeze), strict=bool(strict))
        else:
            if bool(freeze):
                self.freeze_classes(ids)
            result = self.geometry_bank.assert_bank_valid(seen_classes=ids, strict=bool(strict))
        self.old_class_count = len(ids)
        self.current_num_classes = max(self.current_num_classes, max(ids) + 1 if ids else 0)
        self.seen_classes = list(ids)
        return result

    def assert_pg_rga_contract(self, seen_classes: Iterable[int], *, phase: str = "base") -> None:
        seen = _ordered_unique_ints(seen_classes)
        self.assert_phase_ready(seen, mode="geometry", require_geometry=True)
        if str(phase).lower().startswith("base"):
            if self.incremental_mode_active:
                raise RuntimeError("Base contract violation: incremental_mode_active=True during base phase.")
            if _module_has_trainable_params(self.geometry_plastic_adapter):
                raise RuntimeError("Base contract violation: geometry_plastic_adapter is trainable during base phase.")
        else:
            self.assert_frozen_modules()
            if self.use_geometry_gated_adapter and not _module_has_trainable_params(self.geometry_plastic_adapter):
                raise RuntimeError("PG-RGA incremental contract violation: geometry_plastic_adapter is not trainable.")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    def assert_phase_ready(self, seen_classes: Iterable[int], *, mode: str = "geometry", require_geometry: bool = True) -> None:
        seen = _ordered_unique_ints(seen_classes)
        if not seen:
            raise RuntimeError("seen_classes is empty.")
        if self.geometry_bank.d_model != self.d_model:
            raise RuntimeError(f"GeometryBank d_model={self.geometry_bank.d_model} != model d_model={self.d_model}")
        if _normalize_classifier_mode(mode, "geometry") in {"geometry", "calibrated_geometry"} and require_geometry:
            if hasattr(self.geometry_bank, "assert_bank_valid"):
                self.geometry_bank.assert_bank_valid(seen_classes=seen, strict=True)
            else:
                bank = self.get_subspace_bank()
                missing = [c for c in seen if c >= bank["sample_counts"].numel() or float(bank["sample_counts"][c].item()) <= 0.0]
                if missing:
                    raise RuntimeError(f"Missing class geometry for seen classes: {missing}")
        if self.incremental_mode_active:
            # In incremental mode, frozen feature modules must stay eval to avoid dropout drift.
            bad_train = []
            for name in ("backbone", "projection"):
                m = getattr(self, name)
                if m.training:
                    bad_train.append(name)
            if bad_train:
                raise RuntimeError(f"Incremental frozen modules must be eval(), but these are train(): {bad_train}")
        self.classifier.expand_to_seen_classes(seen)

    def assert_no_missing_class_geometry(self, seen_classes: Iterable[int]) -> None:
        if hasattr(self.geometry_bank, "assert_bank_valid"):
            self.geometry_bank.assert_bank_valid(seen_classes=seen_classes, strict=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        mode = _normalize_classifier_mode(kwargs.get("classifier_mode", kwargs.get("mode", "geometry")), "geometry")
        return_features_only = _to_bool(kwargs.get("return_features_only", False), False)
        seen_classes = kwargs.get("seen_classes", None)
        old_classes = kwargs.get("old_classes", None)
        new_classes = kwargs.get("new_classes", None)
        targets = kwargs.get("targets", kwargs.get("labels", None))
        targets_are_global = _to_bool(kwargs.get("targets_are_global", kwargs.get("labels_are_global", False)), False)
        return_energy = _to_bool(kwargs.get("return_energy", False), False)
        return_parts = _to_bool(kwargs.get("return_parts", False), False)
        return_diagnostics = _to_bool(kwargs.get("return_diagnostics", False), False)

        apply_adapter = kwargs.get("apply_adapter", None)
        if mode == "base_ce":
            apply_adapter = False
        features_out = self.forward_features(
            x,
            spectral_summary=kwargs.get("spectral_summary", None),
            band_weights=kwargs.get("band_weights", None),
            spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
            apply_adapter=apply_adapter,
        )
        if return_features_only or mode == "base_ce":
            if mode == "base_ce" or "num_base_classes" in kwargs:
                nbase = int(kwargs.get("num_base_classes", self.base_ce_num_classes))
                if nbase <= 0:
                    raise RuntimeError("num_base_classes is required for base_ce mode.")
                features_out = dict(features_out)
                features_out["logits"] = self.base_ce_logits(features_out["features"], nbase)
            return features_out

        if seen_classes is None:
            seen_classes = self._infer_seen_classes(self.geometry_bank)
        logits_out = self.compute_logits_from_features(
            features_out["features"],
            seen_classes=seen_classes,
            geometry_bank=self.geometry_bank,
            mode=mode,
            targets=targets,
            targets_are_global=targets_are_global,
            old_classes=old_classes,
            new_classes=new_classes,
            return_energy=return_energy,
            return_parts=return_parts,
            return_diagnostics=return_diagnostics,
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

#         # Hard-disable stale/unsafe paths in the model object.
#         self.use_incremental_adapter = False
#         self.use_geometry_calibrator = False
#         self.use_bicyc_geometry_cycle = False
#         self.use_geometry_transport = False
#         self.use_sglat_transport = False
#         self.use_geometry_gated_adapter = _to_bool(getattr(args, "use_geometry_gated_adapter", False), False)
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
#     ) -> Dict[str, torch.Tensor]:
#         """Return the single canonical geometry representation used everywhere."""
#         raw = self.extract_features(x)
#         h = self._validate_feature_tensor(raw["features"], "preproject_features", int(x.size(0)))
#         z = self.norm(self.projection(h) + h)
#         z = self._canonicalize(z, name="canonical_geometry_features")
#         adapter_out: Optional[Dict[str, torch.Tensor]] = None
#         if self._adapter_runtime_enabled(force=False):
#             adapter_out = self.adapt_projected_features(z, force=False, return_delta=True, recanonicalize=True)
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
#             "geometry_gated_adapter_available": hasattr(self, "geometry_plastic_adapter"),
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
#         if hasattr(self.geometry_bank, "freeze_classes"):
#             self.geometry_bank.freeze_classes(class_ids_or_count)
#         elif hasattr(self.geometry_bank, "freeze_classes_up_to"):
#             count = int(class_ids_or_count) if isinstance(class_ids_or_count, int) else max(_ordered_unique_ints(class_ids_or_count)) + 1
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
#         # The trainer explicitly unfreezes geometry_plastic_adapter only for the
#         # geometry_gated_adapter ablation. Keep it frozen here to avoid leakage
#         # in descriptor-only runs.
#         self._set_requires_grad(getattr(self, "geometry_plastic_adapter", None), False)
#         if getattr(self, "geometry_plastic_adapter", None) is not None:
#             self.geometry_plastic_adapter.eval()
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

#         features_out = self.forward_features(
#             x,
#             spectral_summary=kwargs.get("spectral_summary", None),
#             band_weights=kwargs.get("band_weights", None),
#             spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
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

# from typing import Any, Dict, Iterable, Optional

# import copy
# import inspect
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.backbone import SSMBackbone
# from models.geometry_bank import GeometryBank
# from models.classifier import GeometryEnergyClassifier


# def _filter_supported_kwargs(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
#     """Keep only kwargs accepted by a class/function signature."""
#     try:
#         sig = inspect.signature(cls_or_fn)
#     except (TypeError, ValueError):
#         return kwargs
#     if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
#         return kwargs
#     allowed = set(sig.parameters.keys())
#     allowed.discard("self")
#     return {k: v for k, v in kwargs.items() if k in allowed}


# def _as_bool(value: Any, default: bool = False) -> bool:
#     """Robust argparse-safe bool parsing. bool('false') is True, so never use bool() on raw CLI strings."""
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



# def _normalize_classifier_mode(mode: Optional[str], default: str = "geometry_only") -> str:
#     """Normalize classifier-mode aliases used by main/trainer/classifier.

#     Critical SRGP rule:
#         geometry_only = feature geometry only, used for synthetic replay.
#         srgp          = feature geometry + physical spectral residual energy.
#     """
#     if mode is None:
#         mode = default
#     m = str(mode).lower().strip()
#     aliases = {
#         "": default,
#         "none": default,
#         "geometry": "geometry_only",
#         "geo": "geometry_only",
#         "feature_geometry": "geometry_only",
#         "low_rank_geometry": "geometry_only",
#         "spectral_geometry": "srgp",
#         "spectral_residual": "srgp",
#         "spectral_residual_geometry": "srgp",
#         "srgp_geometry": "srgp",
#         # Legacy names must not silently activate spectral scoring during
#         # incremental replay/transport. Synthetic old anchors have no physical
#         # spectra, so these route to feature-only geometry unless SRGP is
#         # explicitly requested.
#         "anchor": "geometry_only",
#         "anchor_concept": "geometry_only",
#         "prototype": "geometry_only",
#         "linear": "geometry_only",
#         "calibrated_geometry": "geometry_only",
#         "topology_calibrated_geometry": "geometry_only",
#         "srgp_real": "srgp",
#         "real_spectral": "srgp",
#     }
#     return aliases.get(m, m)


# def _normalize_incremental_update_mode(mode: Optional[str], default: str = "scbgr") -> str:
#     """Normalize incremental architecture aliases without changing the bank.

#     Model-level meaning:
#         scbgr / rsgi / descriptor_only:
#             No model-side feature plasticity.  The trainer performs geometry-state
#             insertion/refinement/admission using the existing GeometryBank rows.

#         geometry_gated_adapter:
#             Explicit ablation/escape hatch.  Enables the bounded residual adapter
#             after canonical z and must be trained with old replay invariance.

#     This function deliberately does not require new GeometryBank methods.
#     """
#     if mode is None:
#         mode = default
#     m = str(mode).lower().strip()
#     aliases = {
#         "": default,
#         "none": default,
#         "clean": "scbgr",
#         "rsgi": "scbgr",
#         "sgdr": "scbgr",
#         "scbgr": "scbgr",
#         "scb-gr": "scbgr",
#         "spectral_risk_boundary": "scbgr",
#         "boundary_geometry": "scbgr",
#         "geometry_state_admission": "scbgr",
#         # Old name kept so existing commands do not break.  In the cleaned
#         # architecture, descriptor-only at the model level means: no adapter;
#         # trainer-side geometry-state admission still owns the incremental update.
#         "descriptor": "scbgr",
#         "descriptor_only": "scbgr",
#         "geometry_gated_adapter": "geometry_gated_adapter",
#         "g2rpa": "geometry_gated_adapter",
#         "g2-rpa": "geometry_gated_adapter",
#         "g²rpa": "geometry_gated_adapter",
#         "adapter": "geometry_gated_adapter",
#         "gated_adapter": "geometry_gated_adapter",
#         "geometry_adapter": "geometry_gated_adapter",
#     }
#     out = aliases.get(m, m)
#     if out not in {"scbgr", "geometry_gated_adapter"}:
#         raise RuntimeError(
#             f"Unsupported incremental_update_mode={mode!r}. "
#             "Allowed model-level modes: scbgr, geometry_gated_adapter."
#         )
#     return out


# def _force_clean_model_args(args: Any) -> None:
#     """Disable dangerous legacy paths while keeping the SRGP spectral path alive.

#     Earlier clean versions forced every spectral classifier flag off. That is no
#     longer correct: SRGP requires feature geometry plus physical spectral-shape
#     descriptors. This function only disables prompt/adaptor/transport/KD-like
#     machinery and sets safe SRGP/SGLAT defaults when an argument is missing.
#     SGLAT transport is not a legacy geometry calibrator: it is estimated by
#     models.geometry_transport and applied explicitly to frozen GeometryBank rows.
#     """
#     forced_off = {
#         "use_incremental_adapter": False,
#         "disable_incremental_adapter": True,
#         "incremental_adapter_normalize": False,
#         "use_geometry_calibrator": False,
#         "use_bicyc_geometry_cycle": False,
#         "bicyc_cycle_weight": 0.0,
#         "bicyc_reg_weight": 0.0,
#         "use_shared_private_geometry": False,
#         "shared_geometry_energy_weight": 0.0,
#         "allow_incremental_projection_training": False,
#         "freeze_projection_during_incremental": True,
#         "band_energy_weight": 0.0,  # band signature is memory/risk, not a direct classifier branch
#     }
#     for key, value in forced_off.items():
#         try:
#             setattr(args, key, value)
#         except Exception:
#             pass

#     # Safe defaults for the SRGP path. Do not overwrite explicit user choices
#     # unless they point to removed legacy modes.
#     defaults = {
#         "base_classifier_mode": "srgp",
#         # Incremental/replay/transport must default to feature-only geometry.
#         # SRGP spectral scoring is available only when explicitly requested for
#         # real HSI batches with physical spectra.
#         "incremental_classifier_mode": "geometry_only",
#         "eval_classifier_mode": "geometry_only",
#         "use_spectral_geometry": True,
#         "spectral_energy_weight": 0.05,
#         "spectral_derivative_weight": 0.50,
#         "spectral_second_derivative_weight": 0.25,
#         "spectral_require_physical_summary": True,
#         "max_charts_per_class": 1,
#         "spectral_shape_weight": 0.25,
#         # Decision-time old/new tangent barrier. This must be routed by NECILModel,
#         # otherwise classifier.py can contain the fix but the model still evaluates
#         # the unsafe raw geometry energy.
#         "use_old_new_overlap_barrier": True,
#         "old_new_overlap_barrier_weight": 0.35,
#         "old_new_overlap_barrier_threshold": 0.60,
#         "old_new_overlap_barrier_temperature": 0.15,
#         "old_new_overlap_barrier_topk": 3,
#         # ADBS-style adaptive decision boundary in GeometryBank-energy space.
#         # This is the actual old/new decision-radius controller used by the
#         # incremental trainer; old radii are frozen and new radii are optimized.
#         "use_adaptive_boundary": True,
#         "boundary_radius_min": 0.50,
#         "boundary_radius_max": 2.00,
#         "boundary_init_radius": 1.00,
#         "boundary_radius_reg_weight": 0.01,
#         "boundary_old_new_constraint_weight": 0.20,
#         "boundary_old_new_margin_base": 0.05,
#         "boundary_old_new_margin_scale": 0.25,
#         "adaptive_boundary_loss_weight": 1.00,
#         "adaptive_boundary_lr": 1e-4,
#         # SGLAT-HSI transport defaults. These are allowed architectural flags;
#         # they do not enable legacy BiCyc/geometry-calibrator paths.
#         # Transport is no longer forced on by the model. The trainer/command
#         # must enable it explicitly, and when enabled it uses low-rank residual
#         # drift defaults aligned with GeometryBank transport.
#         "use_sglat_transport": False,
#         "use_geometry_transport": False,
#         "allow_old_model_transport": True,
#         "allow_transport_without_adapter": False,
#         "transport_type": "ridge",
#         "transport_ridge": 1e-3,
#         "transport_ema": 0.97,
#         "transport_identity_blend": 0.75,
#         "transport_low_rank": 4,
#         "transport_after_adapter_epoch": 3,
#         "transport_spectral_reliability_gate": True,
#         "transport_min_reliability_gate": 0.30,
#         "transport_max_a_minus_i_fro": 1.5,
#         "transport_max_b_norm": 0.75,
#         "transport_residual_scale": 0.50,
#         "transport_min_rmse_gain": 1e-5,
#         "transport_max_rmse_ratio": 0.98,
#         "transport_min_old_anchor_acc": 95.0,
#     }
#     for key, value in defaults.items():
#         try:
#             cur = getattr(args, key, None)
#             if cur is None:
#                 setattr(args, key, value)
#         except Exception:
#             try:
#                 setattr(args, key, value)
#             except Exception:
#                 pass

#     # Map removed/stale modes to SRGP/geometry-only. Base may use SRGP;
#     # incremental/eval default to geometry-only unless the caller explicitly
#     # requests spectral SRGP with physical spectra.
#     mode_defaults = {
#         "base_classifier_mode": "srgp",
#         "incremental_classifier_mode": "geometry_only",
#         "eval_classifier_mode": "geometry_only",
#     }
#     for key, fallback in mode_defaults.items():
#         try:
#             setattr(args, key, _normalize_classifier_mode(getattr(args, key, fallback), fallback))
#         except Exception:
#             pass

# def _zero_scalar(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
#     return torch.tensor(0.0, device=device, dtype=dtype)


# class GeometryGatedResidualAdapter(nn.Module):
#     """
#     Geometry-gated residual plasticity module for HSI NECIL.

#     It is intentionally small and placed after the canonical projected geometry
#     feature z.  The adapter is not a global backbone update.  It learns a bounded
#     residual correction and opens/closes that correction using old-bank geometry
#     risk statistics.  Old-like samples should receive a near-zero gate; new or
#     ambiguous samples may receive a larger gate.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         bottleneck: int = 32,
#         risk_dim: int = 4,
#         max_scale: float = 0.35,
#         dropout: float = 0.0,
#         gate_bias_init: float = -3.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.risk_dim = int(risk_dim)
#         hidden = max(4, min(int(bottleneck), int(d_model)))
#         self.max_scale = float(max(0.0, max_scale))

#         self.delta = nn.Sequential(
#             nn.LayerNorm(self.d_model),
#             nn.Linear(self.d_model, hidden),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(hidden, self.d_model),
#         )
#         self.risk_norm = nn.LayerNorm(self.risk_dim)
#         self.gate = nn.Sequential(
#             nn.LayerNorm(self.d_model + self.risk_dim),
#             nn.Linear(self.d_model + self.risk_dim, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#         )
#         self.reset_parameters(float(gate_bias_init))

#     def reset_parameters(self, gate_bias_init: float = -3.0) -> None:
#         # Start as exact/no-near identity.  This avoids destroying the existing
#         # base GeometryBank when the architecture is first enabled.
#         last_delta = self.delta[-1]
#         if isinstance(last_delta, nn.Linear):
#             nn.init.zeros_(last_delta.weight)
#             nn.init.zeros_(last_delta.bias)
#         last_gate = self.gate[-1]
#         if isinstance(last_gate, nn.Linear):
#             nn.init.zeros_(last_gate.weight)
#             nn.init.constant_(last_gate.bias, float(gate_bias_init))

#     def forward(
#         self,
#         z: torch.Tensor,
#         risk_features: Optional[torch.Tensor] = None,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if z.dim() != 2:
#             raise ValueError(f"z must be [B,D], got {tuple(z.shape)}")
#         if risk_features is None:
#             risk_features = torch.zeros((z.size(0), self.risk_dim), device=z.device, dtype=z.dtype)
#         else:
#             risk_features = risk_features.to(device=z.device, dtype=z.dtype)
#             if risk_features.dim() != 2 or risk_features.size(0) != z.size(0):
#                 raise ValueError(
#                     f"risk_features must be [B,{self.risk_dim}], got {tuple(risk_features.shape)}"
#                 )
#             if risk_features.size(1) != self.risk_dim:
#                 raise ValueError(f"risk_features dim mismatch: {risk_features.size(1)} vs {self.risk_dim}")

#         risk = torch.nan_to_num(risk_features, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
#         risk = self.risk_norm(risk)
#         gate_logits = self.gate(torch.cat([z, risk], dim=1))
#         gate = torch.sigmoid(gate_logits)
#         raw_delta = self.delta(z)
#         scaled_delta = float(self.max_scale) * gate * raw_delta
#         return z + scaled_delta, gate, scaled_delta


# class NECILModel(nn.Module):
#     """
#     Minimal geometry-aware NECIL-HSI model.

#     The model owns only:
#       1) backbone,
#       2) projection + LayerNorm canonical feature space,
#       3) temporary base CE head,
#       4) clean GeometryBank,
#       5) clean GeometryEnergyClassifier,
#       6) optional geometry-gated residual plastic adapter for incremental phases,
#       7) SGLAT wrappers for canonical z extraction and GeometryBank transport.

#     The adapter is disabled by default and identity-initialized. When enabled, it
#     gives controlled local plasticity after canonical z while old rows remain
#     protected by GeometryBank and old synthetic replay invariance losses.
#     """

#     def __init__(self, args) -> None:
#         super().__init__()
#         _force_clean_model_args(args)
#         self.args = args
#         self.device = torch.device(getattr(args, "device", "cpu"))
#         self.d_model = int(getattr(args, "d_model", 128))
#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))

#         self.current_num_classes = 0
#         self.old_class_count = 0
#         self.current_phase = 0
#         self.default_base_classifier_mode = _normalize_classifier_mode(getattr(args, "base_classifier_mode", "srgp"), "srgp")
#         self.default_incremental_classifier_mode = _normalize_classifier_mode(getattr(args, "incremental_classifier_mode", "geometry_only"), "geometry_only")
#         self.default_eval_classifier_mode = _normalize_classifier_mode(getattr(args, "eval_classifier_mode", "geometry_only"), "geometry_only")

#         self.backbone = SSMBackbone(args)

#         # Canonical geometry feature z. GeometryBank extraction, base losses,
#         # replay, and final classifier must all use this same space.
#         self.projection = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Dropout(float(getattr(args, "projection_dropout", 0.1))),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#         # Canonical z-space contract. Base GeometryBank extraction, incremental
#         # descriptor insertion/replay, and evaluation must all consume exactly
#         # this feature representation. L2 normalization makes the feature scale
#         # stable across phases; the sqrt(D) scale keeps covariance magnitudes
#         # interpretable instead of collapsing all variances toward zero.
#         self.normalize_geometry_features = _as_bool(
#             getattr(args, "normalize_geometry_features", True), True
#         )
#         raw_scale = float(getattr(args, "geometry_feature_scale", 0.0))
#         self.geometry_feature_scale = raw_scale if raw_scale > 0.0 else math.sqrt(float(self.d_model))
#         self.geometry_feature_clamp = float(getattr(args, "geometry_feature_clamp", 0.0))
#         self.strict_feature_contract = _as_bool(getattr(args, "strict_feature_contract", True), True)
#         # SGLAT-HSI transport is model-external estimation + bank-internal application.
#         # It is not the removed BiCyc/geometry-calibrator path. The model only
#         # exposes canonical z extraction and frozen snapshots for transport pairs.
#         self.use_sglat_transport = _as_bool(getattr(args, "use_sglat_transport", False), False)
#         self.use_geometry_transport = self.use_sglat_transport or _as_bool(getattr(args, "use_geometry_transport", False), False)
#         self.allow_old_model_transport = _as_bool(getattr(args, "allow_old_model_transport", True), True)
#         self.allow_transport_without_adapter = _as_bool(getattr(args, "allow_transport_without_adapter", False), False)
#         self.transport_defaults = {
#             "ema": float(getattr(args, "transport_ema", 0.97)),
#             "identity_blend": float(getattr(args, "transport_identity_blend", 0.75)),
#             "residual_scale": float(getattr(args, "transport_residual_scale", 0.50)),
#             "min_reliability_gate": float(getattr(args, "transport_min_reliability_gate", 0.30)),
#             "low_rank_delta": int(getattr(args, "transport_low_rank", 4)),
#             "max_delta_fro": float(getattr(args, "transport_max_a_minus_i_fro", 1.5)),
#             "max_b_norm": float(getattr(args, "transport_max_b_norm", 0.75)),
#             "require_frozen": True,
#         }

#         # Spectral consistency is exposed as a band/reliability signal, not as a
#         # second classifier branch. For center-pixel HSI classification, the
#         # safest default spectral summary is the center spectrum, not the whole
#         # patch mean, because the label belongs to the center pixel and neighbors
#         # may include mixed/other classes.
#         self.spectral_summary_mode = str(getattr(args, "spectral_summary_mode", "center")).lower().strip()
#         if self.spectral_summary_mode not in {"center", "mean"}:
#             raise ValueError("spectral_summary_mode must be 'center' or 'mean'.")
#         # If PCA was applied before the model input, derivatives over x channels
#         # are not physical wavelength derivatives. Trainers can pass a raw
#         # spectrum through forward(..., spectral_summary=raw_spectrum,
#         # spectral_summary_is_physical=True) to activate the SRGP spectral scorer.
#         pca_components = int(getattr(args, "pca_components", 0) or 0)
#         default_physical = False if pca_components > 0 else True
#         self.spectral_summary_is_physical = _as_bool(
#             getattr(args, "spectral_summary_is_physical", default_physical),
#             default_physical,
#         )
#         self.allow_nonphysical_spectral_summary = _as_bool(
#             getattr(args, "allow_nonphysical_spectral_summary", True),
#             True,
#         )
#         self.min_band_mass = float(getattr(args, "min_band_mass", 1e-8))

#         # Temporary base CE head only. Not memory, not a prototype, not used as
#         # incremental classifier.
#         self.base_ce_head: Optional[nn.Linear] = None
#         self.base_ce_num_classes: int = 0

#         bank_kwargs = _filter_supported_kwargs(
#             GeometryBank.__init__,
#             {
#                 "device": str(self.device),
#                 "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
#                 "variance_shrinkage": float(getattr(args, "geometry_variance_shrinkage", 0.10)),
#                 "max_variance_ratio": float(getattr(args, "geometry_max_variance_ratio", 50.0)),
#                 "min_reliability": float(getattr(args, "geometry_min_reliability", 0.05)),
#                 "reliability_sample_alpha": float(getattr(args, "reliability_sample_alpha", 20.0)),
#                 "reliability_sample_weight": float(getattr(args, "reliability_sample_weight", 0.25)),
#                 "reliability_rank_weight": float(getattr(args, "reliability_rank_weight", 0.20)),
#                 "reliability_compact_weight": float(getattr(args, "reliability_compact_weight", 0.35)),
#                 "reliability_band_weight": float(getattr(args, "reliability_band_weight", 0.20)),
#                 "rank_energy_threshold": float(getattr(args, "rank_energy_threshold", 0.95)),
#                 "rank_eigen_ratio_threshold": float(getattr(args, "rank_eigen_ratio_threshold", 1e-3)),
#                 "min_active_rank": int(getattr(args, "min_active_rank", 1)),
#                 "residual_fraction_floor": float(getattr(args, "residual_fraction_floor", 1e-6)),
#                 # SRGP: charts are dormant by default; spectral-shape conflict remains active.
#                 "max_charts_per_class": int(getattr(args, "max_charts_per_class", 1)),
#                 "spectral_shape_weight": float(getattr(args, "spectral_shape_weight", 0.25)),
#                 # accepted/ignored by clean bank if present in older code
#                 "spectral_rank": int(getattr(args, "spectral_rank", self.subspace_rank)),
#             },
#         )
#         self.geometry_bank = GeometryBank(self.d_model, self.subspace_rank, **bank_kwargs)

#         classifier_kwargs = _filter_supported_kwargs(
#             GeometryEnergyClassifier.__init__,
#             {
#                 "initial_classes": 0,
#                 "d_model": self.d_model,
#                 "logit_scale": float(getattr(args, "loss_scale", 8.0)),
#                 "variance_floor": float(getattr(args, "geom_var_floor", 1e-4)),
#                 "reliability_energy_weight": float(getattr(args, "reliability_energy_weight", 0.03)),
#                 "residual_variance_scale": float(getattr(args, "residual_variance_scale", 0.75)),
#                 "energy_normalize_by_dim": _as_bool(getattr(args, "energy_normalize_by_dim", True), True),
#                 "normalize_logits": _as_bool(getattr(args, "geometry_normalize_logits", False), False),
#                 "logit_clip": float(getattr(args, "geometry_logit_clip", 0.0)),
#                 "invalid_class_energy": float(getattr(args, "invalid_class_energy", 1e6)),
#                 "reliability_min_clamp": float(getattr(args, "geometry_min_reliability", 0.05)),
#                 "center_reliability_energy": _as_bool(getattr(args, "center_reliability_energy", True), True),
#                 # Covariance-consistent geometry energy. Unsupported keys are
#                 # filtered out for older classifier.py versions.
#                 "use_logdet_energy": _as_bool(getattr(args, "use_logdet_energy", True), True),
#                 "logdet_energy_weight": float(getattr(args, "logdet_energy_weight", 0.05)),
#                 # Optional old/new score calibration only; not the main solver.
#                 "use_energy_calibrator": _as_bool(getattr(args, "use_energy_calibrator", False), False),
#                 "energy_calibrator_type": str(getattr(args, "energy_calibrator_type", "none")),
#                 "energy_calibrator_max_log_scale": float(getattr(args, "energy_calibrator_max_log_scale", 0.35)),
#                 "energy_calibrator_max_bias": float(getattr(args, "energy_calibrator_max_bias", 1.0)),
#                 # SRGP spectral-residual classifier. Unsupported keys are
#                 # filtered out for older classifier.py versions.
#                 "use_spectral_geometry": _as_bool(getattr(args, "use_spectral_geometry", True), True),
#                 "spectral_energy_weight": float(getattr(args, "spectral_energy_weight", 0.05)),
#                 "spectral_derivative_weight": float(getattr(args, "spectral_derivative_weight", 0.50)),
#                 "spectral_second_derivative_weight": float(getattr(args, "spectral_second_derivative_weight", 0.25)),
#                 "spectral_require_physical_summary": _as_bool(getattr(args, "spectral_require_physical_summary", True), True),
#                 "band_energy_weight": 0.0,
#                 "use_shared_private_geometry": False,
#                 # Decision-time barrier for persistent old/new tangent collisions
#                 # such as Corn-notill <-> Soybean-notill. These are filtered out
#                 # automatically when an older classifier.py is installed.
#                 "use_old_new_overlap_barrier": _as_bool(getattr(args, "use_old_new_overlap_barrier", True), True),
#                 "old_new_overlap_barrier_weight": float(getattr(args, "old_new_overlap_barrier_weight", 0.35)),
#                 "old_new_overlap_barrier_threshold": float(getattr(args, "old_new_overlap_barrier_threshold", 0.60)),
#                 "old_new_overlap_barrier_temperature": float(getattr(args, "old_new_overlap_barrier_temperature", 0.15)),
#                 "old_new_overlap_barrier_topk": int(getattr(args, "old_new_overlap_barrier_topk", 3)),
#                 # Adaptive decision boundary.  These kwargs are filtered out if an
#                 # older classifier.py is accidentally installed, but with the
#                 # updated classifier they create one learnable energy radius per
#                 # class: E'_c = E_c / rho_c + log(rho_c).
#                 "use_adaptive_boundary": _as_bool(getattr(args, "use_adaptive_boundary", True), True),
#                 "boundary_radius_min": float(getattr(args, "boundary_radius_min", 0.50)),
#                 "boundary_radius_max": float(getattr(args, "boundary_radius_max", 2.00)),
#                 "boundary_init_radius": float(getattr(args, "boundary_init_radius", 1.00)),
#                 "boundary_radius_reg_weight": float(getattr(args, "boundary_radius_reg_weight", 0.01)),
#                 "boundary_old_new_constraint_weight": float(getattr(args, "boundary_old_new_constraint_weight", 0.20)),
#                 "boundary_old_new_margin_base": float(getattr(args, "boundary_old_new_margin_base", 0.05)),
#                 "boundary_old_new_margin_scale": float(getattr(args, "boundary_old_new_margin_scale", 0.25)),
#             },
#         )
#         self.classifier = GeometryEnergyClassifier(**classifier_kwargs)

#         # Geometry-gated residual plasticity.  This is the architectural escape
#         # hatch for HSI class overlap: descriptor-only insertion cannot separate
#         # new classes if the frozen base z-space already maps them inside old
#         # basins.  The module is identity-initialized and only active when the
#         # trainer/main sets incremental_update_mode='geometry_gated_adapter'.
#         self.incremental_update_mode = _normalize_incremental_update_mode(
#             getattr(args, "incremental_update_mode", "scbgr"),
#             default="scbgr",
#         )
#         # Do not write the normalized value back into args here.  The trainer may
#         # still need its own alias handling, and mutating shared args inside the
#         # model is how earlier code silently disabled the intended incremental path.
#         self.use_geometry_gated_adapter = self.incremental_update_mode == "geometry_gated_adapter"
#         self.adapter_risk_dim = 4
#         self.geometry_plastic_adapter = GeometryGatedResidualAdapter(
#             d_model=self.d_model,
#             bottleneck=int(getattr(args, "adapter_bottleneck", 32)),
#             risk_dim=self.adapter_risk_dim,
#             max_scale=float(getattr(args, "adapter_max_scale", 0.35)),
#             dropout=float(getattr(args, "adapter_dropout", 0.0)),
#             gate_bias_init=float(getattr(args, "adapter_gate_bias_init", -3.0)),
#         ).to(self.device)
#         if not self.use_geometry_gated_adapter:
#             self.freeze_geometry_plastic_adapter()

#         # Hard-disable stale feature adapters and geometry-cycle transports in the clean model.
#         # Descriptor refinement is implemented outside the model by the incremental trainer
#         # and commits only new GeometryBank rows.  The optional geometry-gated adapter is not
#         # an old<->new transport path; it is a bounded residual correction after z and must be
#         # trained with old synthetic replay invariance.
#         self.use_incremental_adapter = False
#         self.use_bicyc_geometry_cycle = False
#         self.use_geometry_calibrator = False
#         self.use_energy_calibrator = _as_bool(getattr(self.classifier, "use_energy_calibrator", False), False)
#         self.energy_calibrator_type = str(getattr(self.classifier, "energy_calibrator_type", "none"))
#         if not self.use_energy_calibrator:
#             self.freeze_energy_calibrator()

#     # ------------------------------------------------------------------
#     # State
#     # ------------------------------------------------------------------
#     def set_phase(self, phase: int) -> None:
#         self.current_phase = int(phase)

#     def set_old_class_count(self, old_class_count: int) -> None:
#         self.old_class_count = int(old_class_count)

#     def _resolve_classifier_mode(self, classifier_mode: Optional[str]) -> str:
#         if classifier_mode is None:
#             default = self.default_base_classifier_mode if int(self.current_phase) == 0 else self.default_incremental_classifier_mode
#             fallback = "srgp" if int(self.current_phase) == 0 else "geometry_only"
#             return _normalize_classifier_mode(default, fallback)
#         return _normalize_classifier_mode(classifier_mode, "geometry_only")

#     # ------------------------------------------------------------------
#     # Temporary base CE head
#     # ------------------------------------------------------------------
#     def ensure_base_ce_head(self, num_base_classes: int, feature_dim: Optional[int] = None) -> nn.Linear:
#         num_base_classes = int(num_base_classes)
#         feature_dim = int(feature_dim or self.d_model)
#         if num_base_classes <= 0:
#             raise ValueError(f"num_base_classes must be positive, got {num_base_classes}")
#         if feature_dim != self.d_model:
#             raise RuntimeError(f"base CE head feature_dim must equal d_model={self.d_model}, got {feature_dim}")
#         if self.base_ce_head is None or self.base_ce_num_classes != num_base_classes:
#             head = nn.Linear(self.d_model, num_base_classes, bias=True).to(self.device)
#             nn.init.normal_(head.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(head.bias)
#             self.base_ce_head = head
#             self.base_ce_num_classes = num_base_classes
#         return self.base_ce_head

#     def base_ce_logits(self, features: torch.Tensor, num_base_classes: Optional[int] = None) -> torch.Tensor:
#         features = self._validate_feature_tensor(features, "base_ce_logits.features", int(features.size(0)))
#         if self.base_ce_head is None:
#             if num_base_classes is None:
#                 raise RuntimeError("base_ce_head is not initialized. Call ensure_base_ce_head() first.")
#             self.ensure_base_ce_head(int(num_base_classes), int(features.size(1)))
#         return self.base_ce_head(features)

#     def forward_base_ce(self, x: torch.Tensor, num_base_classes: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         out = self.extract_projected_features(
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

#     def freeze_base_ce_head(self) -> None:
#         if self.base_ce_head is not None:
#             for p in self.base_ce_head.parameters():
#                 p.requires_grad = False

#     def unfreeze_base_ce_head(self) -> None:
#         if self.base_ce_head is not None:
#             for p in self.base_ce_head.parameters():
#                 p.requires_grad = True

#     # Legacy PRL/PES aliases retained for old trainer calls.
#     def ensure_base_prl_head(self, num_base_classes: int, feature_dim: Optional[int] = None) -> nn.Linear:
#         return self.ensure_base_ce_head(num_base_classes, feature_dim)

#     def base_prl_logits(self, features: torch.Tensor, num_base_classes: Optional[int] = None) -> torch.Tensor:
#         return self.base_ce_logits(features, num_base_classes)

#     def drop_base_prl_head(self) -> None:
#         self.drop_base_ce_head()

#     def freeze_base_prl_head(self) -> None:
#         self.freeze_base_ce_head()

#     def unfreeze_base_prl_head(self) -> None:
#         self.unfreeze_base_ce_head()

#     # ------------------------------------------------------------------
#     # Geometry-gated residual plasticity
#     # ------------------------------------------------------------------
#     def adapter_enabled(self) -> bool:
#         return bool(getattr(self, "use_geometry_gated_adapter", False))

#     def freeze_geometry_plastic_adapter(self) -> None:
#         if hasattr(self, "geometry_plastic_adapter"):
#             for p in self.geometry_plastic_adapter.parameters():
#                 p.requires_grad = False

#     def unfreeze_geometry_plastic_adapter(self) -> None:
#         if not self.adapter_enabled():
#             self.freeze_geometry_plastic_adapter()
#             return
#         for p in self.geometry_plastic_adapter.parameters():
#             p.requires_grad = True

#     @torch.no_grad()
#     def _old_bank_risk_features_no_grad(self, z: torch.Tensor) -> torch.Tensor:
#         return self.compute_old_geometry_risk_features(z).detach()

#     def compute_old_geometry_risk_features(self, z: torch.Tensor) -> torch.Tensor:
#         """Return [B,4] old-bank risk features for adapter gating.

#         Columns are bounded transforms of:
#           1) nearest old energy,
#           2) old energy ambiguity margin,
#           3) nearest-old reliability,
#           4) nearest-old residual variance.

#         These are not classifier outputs. They are gate context that tells the
#         adapter whether a feature lies in an old protected basin or in an
#         ambiguous/new region.  If no old bank exists, return zeros so the adapter
#         behaves as a normal bounded residual module for the first incremental
#         insertion.
#         """
#         z = self._validate_feature_tensor(z, "old_geometry_risk_features.z", int(z.size(0)))
#         B = int(z.size(0))
#         if int(getattr(self, "old_class_count", 0)) <= 0:
#             return torch.zeros((B, self.adapter_risk_dim), device=z.device, dtype=z.dtype)

#         bank = self.get_old_subspace_bank(int(self.old_class_count))
#         if not bank or "means" not in bank or not torch.is_tensor(bank["means"]) or bank["means"].numel() == 0:
#             return torch.zeros((B, self.adapter_risk_dim), device=z.device, dtype=z.dtype)

#         # Prefer the project classifier's exact geometry energy implementation so
#         # the gate is aligned with the final decision rule.
#         try:
#             energy = self.classifier.geometry_energy(
#                 features=z,
#                 means=bank["means"].to(device=z.device, dtype=z.dtype),
#                 bases=bank["bases"].to(device=z.device, dtype=z.dtype),
#                 variances=bank["variances"].to(device=z.device, dtype=z.dtype),
#                 reliability=bank.get("reliability", None),
#                 active_ranks=bank.get("active_ranks", None),
#                 sample_counts=bank.get("sample_counts", None),
#                 return_parts=False,
#             )
#             if isinstance(energy, dict):
#                 energy = energy.get("energy", None)
#         except Exception:
#             # Conservative fallback: squared center distance to old means.
#             means = bank["means"].to(device=z.device, dtype=z.dtype)
#             energy = torch.cdist(z, means, p=2).pow(2)

#         if energy is None or not torch.is_tensor(energy) or energy.numel() == 0:
#             return torch.zeros((B, self.adapter_risk_dim), device=z.device, dtype=z.dtype)
#         energy = torch.nan_to_num(energy.to(device=z.device, dtype=z.dtype), nan=1e6, posinf=1e6, neginf=1e6)
#         C_old = int(energy.size(1))
#         k = 2 if C_old >= 2 else 1
#         top = torch.topk(energy, k=k, dim=1, largest=False).values
#         nearest = top[:, 0]
#         if k == 2:
#             margin = top[:, 1] - top[:, 0]
#         else:
#             margin = torch.zeros_like(nearest)
#         nearest_idx = energy.argmin(dim=1)

#         reliability = bank.get("reliability", None)
#         if torch.is_tensor(reliability) and reliability.numel() >= C_old:
#             rel = reliability.to(device=z.device, dtype=z.dtype).flatten()[:C_old][nearest_idx]
#         else:
#             rel = torch.zeros_like(nearest)
#         variances = bank.get("variances", None)
#         if torch.is_tensor(variances) and variances.size(0) >= C_old:
#             res = variances.to(device=z.device, dtype=z.dtype)[:C_old, -1][nearest_idx]
#         else:
#             res = torch.zeros_like(nearest)

#         # Bounded transforms keep gate training stable across datasets/scales.
#         f0 = torch.log1p(nearest.clamp_min(0.0))
#         f1 = torch.log1p(margin.clamp_min(0.0))
#         f2 = rel.clamp(0.0, 1.0)
#         f3 = torch.log1p(res.clamp_min(0.0))
#         risk = torch.stack([f0, f1, f2, f3], dim=1)
#         return torch.nan_to_num(risk, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

#     def _apply_geometry_adapter(
#         self,
#         z_base: torch.Tensor,
#         *,
#         risk_features: Optional[torch.Tensor] = None,
#         force: bool = False,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         z_base = self._validate_feature_tensor(z_base, "adapter.z_base", int(z_base.size(0)))
#         active = bool(force or (self.adapter_enabled() and int(self.current_phase) > 0))
#         if not active:
#             gate = torch.zeros((z_base.size(0), 1), device=z_base.device, dtype=z_base.dtype)
#             return z_base, gate, torch.zeros_like(z_base)
#         if risk_features is None:
#             # Do not let gate-context energy become a loophole for moving old
#             # regions; the adapter receives risk as detached context.
#             risk_features = self._old_bank_risk_features_no_grad(z_base)
#         raw_adapt, gate, raw_delta = self.geometry_plastic_adapter(z_base, risk_features.detach())
#         z_adapt = self._canonicalize_geometry_features(raw_adapt, name="adapted_projected_z")
#         delta = z_adapt - z_base
#         return z_adapt, gate, delta

#     # ------------------------------------------------------------------
#     # Feature extraction
#     # ------------------------------------------------------------------
#     def _validate_feature_tensor(self, feat: torch.Tensor, name: str, batch_size: int) -> torch.Tensor:
#         if not torch.is_tensor(feat):
#             raise TypeError(f"{name} must be a torch.Tensor, got {type(feat)}")
#         if feat.dim() != 2:
#             raise RuntimeError(f"{name} must be [B,D], got {tuple(feat.shape)}")
#         if feat.size(0) != int(batch_size):
#             raise RuntimeError(f"{name} batch mismatch: {feat.size(0)} vs expected {int(batch_size)}")
#         if feat.size(1) != self.d_model:
#             raise RuntimeError(
#                 f"{name} dim mismatch: expected d_model={self.d_model}, got {feat.size(1)}. "
#                 "GeometryBank must be built from projected geometry features."
#             )
#         if not torch.isfinite(feat).all():
#             raise RuntimeError(f"{name} contains NaN/Inf values.")
#         return feat

#     def _spectral_summary(self, x: torch.Tensor) -> torch.Tensor:
#         """Return the spectral vector used for band signatures/reliability.

#         This is intentionally not a classifier branch. The output is consumed by
#         base PGR/band diagnostics and GeometryBank band signatures. For 4-D HSI
#         patches [B,S,H,W], the default is the center spectrum because labels are
#         center-pixel labels; using a patch mean can smear class spectra with
#         neighboring pixels and poison incremental old/new spectral risk.
#         """
#         if x.dim() == 4:
#             if self.spectral_summary_mode == "center":
#                 return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
#             return x.mean(dim=(-1, -2))
#         if x.dim() == 3:
#             # Common compact layouts are [B,S,L] or [B,S,P]. Treat the last
#             # dimension as spatial/sequence and take center/mean consistently.
#             if self.spectral_summary_mode == "center":
#                 return x[:, :, x.size(-1) // 2]
#             return x.mean(dim=-1)
#         if x.dim() == 2:
#             return x
#         raise RuntimeError(f"Unsupported HSI input shape for spectral summary: {tuple(x.shape)}")

#     def _band_summary_from_spectral(self, spectral_summary: torch.Tensor) -> torch.Tensor:
#         """Convert a raw/signed spectral vector into a stable band signature.

#         PCA/whitened bands may be signed. We therefore use magnitude as band
#         evidence and normalize it to a probability-like vector. Degenerate rows
#         fall back to a uniform signature and should later receive low reliability
#         inside GeometryBank.
#         """
#         if spectral_summary.dim() != 2:
#             raise RuntimeError(f"spectral_summary must be [B,S], got {tuple(spectral_summary.shape)}")
#         if spectral_summary.size(1) == 0:
#             return torch.empty((spectral_summary.size(0), 0), device=spectral_summary.device, dtype=spectral_summary.dtype)
#         if not torch.isfinite(spectral_summary).all():
#             raise RuntimeError("spectral_summary contains NaN/Inf values.")
#         band = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0).abs()
#         denom = band.sum(dim=1, keepdim=True)
#         bad = denom <= self.min_band_mass
#         if bool(bad.any().item()):
#             uniform = torch.full_like(band, 1.0 / max(int(band.size(1)), 1))
#             band = torch.where(bad.expand_as(band), uniform, band)
#             denom = band.sum(dim=1, keepdim=True)
#         return band / denom.clamp_min(self.min_band_mass)

#     def _prepare_external_spectral_summary(
#         self,
#         spectral_summary: Optional[torch.Tensor],
#         *,
#         x: torch.Tensor,
#         projected: torch.Tensor,
#         spectral_summary_is_physical: Optional[bool],
#     ) -> tuple[torch.Tensor, bool]:
#         """Prepare spectral metadata for GeometryBank/classifier calls.

#         Empty metadata tensors from the dataset are treated as absent.  When no
#         external raw spectra are supplied, the model falls back to the input
#         center vector and marks it physical only if the model input is known to
#         be wavelength ordered.  This preserves reduced/PCA metadata for band
#         signatures without letting PCA components activate derivative energy.
#         """
#         external = spectral_summary is not None and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0
#         if external:
#             s = torch.as_tensor(spectral_summary, device=projected.device, dtype=projected.dtype)
#             if s.dim() == 4:
#                 s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#             elif s.dim() == 3:
#                 # HSI safety: metadata may arrive as [B,S,L]. The label belongs
#                 # to the center pixel/token, so do not flatten the whole patch.
#                 # Flattening would mix neighboring pixels into the class spectrum
#                 # and poison spectral-shape risk/admission.
#                 if s.size(0) != projected.size(0):
#                     if s.numel() % max(int(projected.size(0)), 1) != 0:
#                         raise RuntimeError(
#                             f"3-D spectral_summary cannot be reshaped to batch size {projected.size(0)}: {tuple(s.shape)}"
#                         )
#                     s = s.reshape(projected.size(0), -1)
#                 elif s.size(1) > 0 and s.size(2) > 1:
#                     s = s[:, :, s.size(-1) // 2]
#                 else:
#                     s = s.reshape(projected.size(0), -1)
#             elif s.dim() == 1:
#                 if s.numel() % max(int(projected.size(0)), 1) != 0:
#                     raise RuntimeError(
#                         f"1-D spectral_summary cannot be reshaped to batch size {projected.size(0)}: {tuple(s.shape)}"
#                     )
#                 s = s.view(projected.size(0), -1)
#             elif s.dim() > 4:
#                 s = s.flatten(1)
#             physical_flag = bool(
#                 self.spectral_summary_is_physical
#                 if spectral_summary_is_physical is None
#                 else spectral_summary_is_physical
#             )
#         else:
#             s = self._spectral_summary(x).to(device=projected.device, dtype=projected.dtype)
#             physical_flag = bool(
#                 self.spectral_summary_is_physical
#                 if spectral_summary_is_physical is None
#                 else spectral_summary_is_physical
#             )

#         if s.dim() != 2 or s.size(0) != projected.size(0):
#             raise RuntimeError(f"spectral_summary must resolve to [B,S], got {tuple(s.shape)}")
#         s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#         if s.size(1) == 0:
#             physical_flag = False
#         if not physical_flag and not self.allow_nonphysical_spectral_summary:
#             raise RuntimeError(
#                 "spectral_summary_is_physical=False. Pass raw wavelength-ordered spectra or enable "
#                 "allow_nonphysical_spectral_summary for band diagnostics only."
#             )
#         return s, bool(physical_flag)

#     def _validate_optional_band_weights(
#         self,
#         band_weights: Optional[torch.Tensor],
#         *,
#         batch_size: int,
#         spectral_dim: int,
#         dtype: torch.dtype,
#         device: torch.device,
#     ) -> Optional[torch.Tensor]:
#         if int(spectral_dim) <= 0:
#             return None
#         if band_weights is None or not torch.is_tensor(band_weights) or band_weights.numel() == 0:
#             return None
#         bw = band_weights.to(device=device, dtype=dtype)
#         if bw.dim() != 2:
#             raise RuntimeError(f"band_weights must be [B,S], got {tuple(bw.shape)}")
#         if bw.size(0) != int(batch_size):
#             raise RuntimeError(f"band_weights batch mismatch: {bw.size(0)} vs {int(batch_size)}")
#         if bw.size(1) != int(spectral_dim):
#             # Reduced/raw metadata safety.  A common valid configuration is
#             # model input = PCA components (e.g. 30) while an older loader still
#             # supplies raw physical spectra (e.g. 200) as spectral_summary.
#             # In that case backbone band_weights live in input-channel space and
#             # cannot weight the external spectral_summary.  Drop them instead of
#             # crashing; the caller will fall back to a summary derived from
#             # spectral_summary itself.
#             return None
#         if not torch.isfinite(bw).all():
#             raise RuntimeError("band_weights contains NaN/Inf values.")
#         bw = bw.clamp_min(0.0)
#         denom = bw.sum(dim=1, keepdim=True)
#         if bool((denom <= 1e-8).any().item()):
#             return None
#         return bw / denom.clamp_min(1e-8)

#     def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         out = self.backbone(x)
#         if not isinstance(out, dict):
#             feat = self._validate_feature_tensor(out, "backbone features", int(x.size(0)))
#             return {
#                 "features": feat,
#                 "backbone_features": feat,
#                 "band_weights": None,
#                 "spectral_features": feat,
#                 "spatial_features": feat,
#             }
#         if "features" not in out:
#             raise RuntimeError("backbone output dict must contain key 'features'.")
#         feat = self._validate_feature_tensor(out["features"], "backbone features", int(x.size(0)))
#         return {
#             "features": feat,
#             "backbone_features": feat,
#             "band_weights": out.get("band_weights", None),
#             "spectral_features": out.get("spectral_features", feat),
#             "spatial_features": out.get("spatial_features", feat),
#         }

#     def _canonicalize_geometry_features(self, z: torch.Tensor, *, name: str) -> torch.Tensor:
#         z = self._validate_feature_tensor(z, name, int(z.size(0)))
#         if self.normalize_geometry_features:
#             z = F.normalize(z, p=2, dim=1, eps=1e-8) * float(self.geometry_feature_scale)
#         if self.geometry_feature_clamp > 0.0:
#             z = z.clamp(min=-self.geometry_feature_clamp, max=self.geometry_feature_clamp)
#         if not torch.isfinite(z).all():
#             raise RuntimeError(f"{name} contains NaN/Inf after canonicalization.")
#         return z

#     def _project_without_adapter(self, feat: torch.Tensor) -> torch.Tensor:
#         # Residual projection learns the PRL-inspired base geometry space.
#         # Canonicalization then fixes the coordinate contract used by base bank
#         # extraction, incremental descriptor insertion, replay, and evaluation.
#         z = self.norm(self.projection(feat) + feat)
#         return self._canonicalize_geometry_features(z, name="canonical_projected_z")

#     def _project(self, feat: torch.Tensor, *, return_pre_adapter: bool = False):
#         z_base = self._project_without_adapter(feat)
#         z_adapt, gate, delta = self._apply_geometry_adapter(z_base)
#         if return_pre_adapter:
#             return z_adapt, z_base, delta, gate
#         return z_adapt

#     def project_features(self, feat: torch.Tensor) -> torch.Tensor:
#         feat = self._validate_feature_tensor(feat, "project_features.input", int(feat.size(0)))
#         return self._project(feat)

#     def adapt_projected_features(self, z: torch.Tensor, *, force: bool = False, return_delta: bool = False):
#         """Apply the geometry-gated adapter directly to already-projected z.

#         This is required for old synthetic replay.  Replay samples live directly
#         in GeometryBank z-space, so they must not pass through the backbone or
#         projection again.  They still must pass through the adapter during
#         adapter training so the model learns gate≈0 / delta≈0 in old basins.
#         """
#         z = self._validate_feature_tensor(z, "adapt_projected_features.input", int(z.size(0)))
#         z_adapt, gate, delta = self._apply_geometry_adapter(z, force=force)
#         if return_delta:
#             return {"features": z_adapt, "base_features": z, "gate": gate, "delta": delta}
#         return z_adapt

#     def incremental_adapter_active(self) -> bool:
#         return bool(self.adapter_enabled() and int(self.current_phase) > 0)

#     def extract_projected_features(
#         self,
#         x: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#     ) -> Dict[str, torch.Tensor]:
#         if not torch.is_tensor(x):
#             raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
#         if x.dim() not in {2, 3, 4}:
#             raise RuntimeError(f"Unsupported HSI input shape: {tuple(x.shape)}")

#         out = self.extract_features(x)
#         raw_feat = self._validate_feature_tensor(out["features"], "preproject_features", int(x.size(0)))
#         projected, pre_adapter_projected, adapter_delta, adapter_gate = self._project(raw_feat, return_pre_adapter=True)
#         projected = self._validate_feature_tensor(projected, "projected geometry features", int(x.size(0)))

#         spectral_summary_t, physical_flag = self._prepare_external_spectral_summary(
#             spectral_summary,
#             x=x,
#             projected=projected,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#         )

#         candidate_band_weights = band_weights if band_weights is not None else out.get("band_weights", None)
#         band_weights = self._validate_optional_band_weights(
#             candidate_band_weights,
#             batch_size=int(projected.size(0)),
#             spectral_dim=int(spectral_summary_t.size(1)),
#             dtype=projected.dtype,
#             device=projected.device,
#         )
#         band_summary = band_weights if band_weights is not None else self._band_summary_from_spectral(spectral_summary_t)

#         return {
#             "features": projected,
#             "projected_features": projected,
#             "pre_adapter_features": pre_adapter_projected,
#             "base_features": pre_adapter_projected,
#             "adapter_delta": adapter_delta,
#             "adapter_gate": adapter_gate,
#             "adapter_active": torch.tensor(float(self.incremental_adapter_active()), device=projected.device, dtype=projected.dtype),
#             "preproject_features": raw_feat,
#             "backbone_features": raw_feat,
#             "band_weights": band_weights,
#             "band_summary": band_summary,
#             "band_importance": band_summary,
#             "spectral_summary": spectral_summary_t,
#             "spectral_summary_is_physical": torch.tensor(bool(physical_flag), device=projected.device, dtype=torch.bool),
#             "spectral_features": out.get("spectral_features", raw_feat),
#             "spatial_features": out.get("spatial_features", raw_feat),
#         }

#     def extract_geometry_features(
#         self,
#         x: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#         space: str = "scoring",
#         pre_adapter: Optional[bool] = None,
#         return_dict: bool = False,
#     ):
#         """Return GeometryBank-compatible feature coordinates.

#         This is the model-side contract for transport.  The old and current
#         model must produce paired features in the same named space:

#             canonical: projected + normalized z before geometry_plastic_adapter
#             scoring:   actual z used by classifier/GeometryBank scoring
#             both:      return both canonical and scoring features

#         Legacy ``pre_adapter=True`` maps to ``space='canonical'``.  Do not use
#         raw backbone features for transport; transported GeometryBank rows are
#         only compatible with the projected z-space returned here.
#         """
#         if pre_adapter is not None:
#             space = "canonical" if bool(pre_adapter) else str(space or "scoring")
#         s = str(space or "scoring").lower().strip()
#         aliases = {
#             "base": "canonical",
#             "pre": "canonical",
#             "pre_adapter": "canonical",
#             "canonical_z": "canonical",
#             "adapted": "scoring",
#             "post_adapter": "scoring",
#             "score": "scoring",
#             "features": "scoring",
#             "z": "scoring",
#         }
#         s = aliases.get(s, s)
#         if s not in {"canonical", "scoring", "both"}:
#             raise ValueError("space must be one of {'canonical','scoring','both'}, got %r" % (space,))

#         out = self.extract_projected_features(
#             x,
#             spectral_summary=spectral_summary,
#             band_weights=band_weights,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#         )
#         canonical = self._validate_feature_tensor(
#             out.get("pre_adapter_features", out["features"]),
#             "extract_geometry_features.canonical",
#             int(x.size(0)),
#         )
#         scoring = self._validate_feature_tensor(
#             out["features"],
#             "extract_geometry_features.scoring",
#             int(x.size(0)),
#         )
#         selected = canonical if s == "canonical" else scoring

#         if bool(return_dict) or s == "both":
#             out = dict(out)
#             out["canonical_features"] = canonical
#             out["geometry_features_pre_adapter"] = canonical
#             out["scoring_features"] = scoring
#             out["geometry_features_scoring"] = scoring
#             out["geometry_space"] = s
#             out["features"] = selected
#             out["projected_features"] = selected
#             out["geometry_features"] = selected
#             return out
#         return selected

#     def clone_frozen_for_transport(self):
#         """Return a frozen old-model snapshot for SGLAT transport pairs.

#         The snapshot is used only to compute ``z_old = f_{t-1}(x_new)`` on
#         current-phase samples. It is not a KD teacher, does not store old HSI
#         patches, and does not store old features. The adapter state is preserved
#         because it is part of the previous phase feature coordinate system.
#         """
#         if not bool(getattr(self, "allow_old_model_transport", True)):
#             raise RuntimeError(
#                 "SGLAT old-model transport snapshot is disabled. Set "
#                 "allow_old_model_transport=True to estimate z_old/z_new pairs."
#             )
#         snap = copy.deepcopy(self)
#         snap.eval()
#         for p in snap.parameters():
#             p.requires_grad = False
#         # Explicitly keep SGLAT flags visible in the snapshot metadata.
#         snap.use_sglat_transport = bool(getattr(self, "use_sglat_transport", False))
#         snap.use_geometry_transport = bool(getattr(self, "use_geometry_transport", False))
#         snap.allow_old_model_transport = False
#         return snap

#     @torch.no_grad()
#     def extract_backbone_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         # Legacy name. It intentionally returns projected z, not raw backbone output.
#         return self.extract_projected_features(x)

#     # ------------------------------------------------------------------
#     # Geometry memory
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def ensure_class_capacity(self, class_count: int, spectral_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
#         class_count = int(class_count)
#         dtype = dtype or self.projection[0].weight.dtype
#         while getattr(self.classifier, "num_classes", 0) < class_count:
#             self.classifier.expand(1, self.current_phase)
#         # Classifier.expand() in the adaptive-boundary classifier already expands
#         # boundary_log_radius.  This explicit call keeps older expansion paths
#         # honest and fails early if the classifier boundary vector is stale.
#         if hasattr(self.classifier, "_ensure_boundary_capacity"):
#             self.classifier._ensure_boundary_capacity(class_count, device=self.device, dtype=dtype)
#         if hasattr(self.geometry_bank, "ensure_class_count"):
#             self.geometry_bank.ensure_class_count(count=class_count, spectral_dim=int(spectral_dim), dtype=dtype)
#         else:
#             self.geometry_bank.ensure_num_classes(class_count)
#         self.current_num_classes = max(self.current_num_classes, class_count)

#     @torch.no_grad()
#     def refresh_class_subspace(
#         self,
#         cls: int,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         eigvals: torch.Tensor,
#         res_var=None,
#         resvar=None,
#         spectral_proto=None,
#         spectral_mean=None,
#         spectral_basis=None,
#         spectral_eigvals=None,
#         spectral_res_var=None,
#         spectral_active_rank=None,
#         spectral_reliability=None,
#         band_importance=None,
#         band_reliability=None,
#         active_rank=None,
#         reliability=None,
#         sample_count=None,
#         feature_reliability=None,
#         spectral_shape: Optional[Dict[str, torch.Tensor]] = None,
#         spectral_curve_mean=None,
#         spectral_curve_var=None,
#         spectral_curve_d1=None,
#         spectral_curve_d2=None,
#         spectral_shape_reliability=None,
#         **_: Any,
#     ) -> None:
#         """
#         Insert/update one class row in the clean bank.

#         All spectral low-rank arguments are accepted for compatibility and ignored.
#         Only band_importance is retained because the clean bank stores band signatures.
#         """
#         del spectral_proto, spectral_mean, spectral_basis, spectral_eigvals
#         del spectral_res_var, spectral_active_rank, spectral_reliability
#         cls = int(cls)
#         rv = res_var if res_var is not None else resvar
#         if rv is None:
#             raise ValueError("refresh_class_subspace requires res_var/resvar.")

#         band_dim = int(torch.as_tensor(band_importance).numel()) if band_importance is not None and torch.as_tensor(band_importance).numel() > 0 else 0
#         self.ensure_class_capacity(cls + 1, spectral_dim=band_dim)
#         if spectral_shape is None and any(v is not None for v in (spectral_curve_mean, spectral_curve_var, spectral_curve_d1, spectral_curve_d2, spectral_shape_reliability)):
#             spectral_shape = {
#                 "mean": spectral_curve_mean,
#                 "var": spectral_curve_var,
#                 "d1": spectral_curve_d1,
#                 "d2": spectral_curve_d2,
#                 "reliability": spectral_shape_reliability,
#             }

#         self.geometry_bank.update_class_geometry(
#             class_id=cls,
#             mean=mean.to(self.device).float(),
#             basis=basis.to(self.device).float(),
#             eigvals=eigvals.to(self.device).float(),
#             resvar=torch.as_tensor(rv, device=self.device, dtype=torch.float32),
#             band_importance=band_importance,
#             band_reliability=band_reliability,
#             active_rank=active_rank,
#             reliability=reliability,
#             sample_count=sample_count,
#             feature_reliability=feature_reliability,
#             spectral_shape=spectral_shape,
#         )
#         self.current_num_classes = max(self.current_num_classes, cls + 1)

#     def get_subspace_bank(self) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.geometry_bank, "get_bank"):
#             raise RuntimeError("GeometryBank must expose get_bank() for NECILModel.")
#         bank = self.geometry_bank.get_bank()
#         if "res_vars" not in bank and "resvars" in bank:
#             bank["res_vars"] = bank["resvars"]
#         if "resvars" not in bank and "res_vars" in bank:
#             bank["resvars"] = bank["res_vars"]
#         if "variances" not in bank and "eigvals" in bank and "res_vars" in bank:
#             bank["variances"] = torch.cat([bank["eigvals"], bank["res_vars"].unsqueeze(-1)], dim=-1)
#         if "valid_mask" not in bank and "sample_counts" in bank and torch.is_tensor(bank["sample_counts"]):
#             bank["valid_mask"] = bank["sample_counts"].to(self.device).flatten() > 0
#         # Backward compatibility aliases expected by older trainer/evaluator code.
#         bank.setdefault("spectral_means", torch.empty((len(self.geometry_bank), 0), device=self.device))
#         bank.setdefault("spectral_protos", bank["spectral_means"])
#         bank.setdefault("spectral_bases", torch.empty((len(self.geometry_bank), 0, 0), device=self.device))
#         bank.setdefault("spectral_variances", torch.empty((len(self.geometry_bank), 0), device=self.device))
#         bank.setdefault("spectral_reliability", torch.empty((len(self.geometry_bank),), device=self.device))
#         bank.setdefault("spectral_active_ranks", torch.empty((len(self.geometry_bank),), dtype=torch.long, device=self.device))
#         return bank

#     def get_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         old_class_count = int(self.old_class_count if old_class_count is None else old_class_count)
#         old_class_count = max(0, min(old_class_count, len(self.geometry_bank)))
#         bank = self.get_subspace_bank()
#         required = ["means", "bases", "variances", "reliability", "active_ranks", "sample_counts"]
#         for key in required:
#             if key not in bank:
#                 raise RuntimeError(f"GeometryBank missing required key '{key}'.")
#         out = {k: bank[k][:old_class_count] for k in required}
#         for key in [
#             "band_importances", "band_importance", "band_reliability", "valid_mask", "frozen_class_mask",
#             "spectral_curve_means", "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2",
#             "spectral_shape_reliability",
#         ]:
#             if key in bank and torch.is_tensor(bank[key]):
#                 out[key] = bank[key][:old_class_count]
#         # Empty compatibility keys.
#         out["spectral_means"] = bank.get("spectral_means", torch.empty((old_class_count, 0), device=self.device))[:old_class_count]
#         out["spectral_protos"] = out["spectral_means"]
#         out["spectral_bases"] = bank.get("spectral_bases", torch.empty((old_class_count, 0, 0), device=self.device))[:old_class_count]
#         out["spectral_variances"] = bank.get("spectral_variances", torch.empty((old_class_count, 0), device=self.device))[:old_class_count]
#         out["spectral_reliability"] = bank.get("spectral_reliability", torch.empty((old_class_count,), device=self.device))[:old_class_count]
#         out["spectral_active_ranks"] = bank.get("spectral_active_ranks", torch.empty((old_class_count,), dtype=torch.long, device=self.device))[:old_class_count]
#         return out

#     def get_calibrated_old_subspace_bank(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         # Clean architecture has no geometry calibrator. Return raw old bank.
#         return self.get_old_subspace_bank(old_class_count)

#     @torch.no_grad()
#     def freeze_old_geometry_states(self, old_class_count: Optional[int] = None) -> None:
#         """Freeze old GeometryBank rows through the current bank API.

#         This keeps the bank unchanged.  The model simply exposes a stable wrapper
#         so trainer code does not need to know the exact bank method name.
#         """
#         count = int(self.old_class_count if old_class_count is None else old_class_count)
#         count = max(0, min(count, len(self.geometry_bank)))
#         self.set_old_class_count(count)
#         if hasattr(self.geometry_bank, "freeze_classes_up_to"):
#             self.geometry_bank.freeze_classes_up_to(count)

#     @torch.no_grad()
#     def snapshot_rows_for_transport(self, class_ids: Iterable[int]) -> Dict[str, torch.Tensor]:
#         """Snapshot compact bank rows before a mutating transport operation."""
#         if not hasattr(self.geometry_bank, "snapshot_rows_for_transport"):
#             ids = torch.as_tensor([int(c) for c in class_ids], device=self.device, dtype=torch.long)
#             bank = self.get_subspace_bank()
#             return {"ids": ids, **{k: v.index_select(0, ids).detach().clone() for k, v in bank.items() if torch.is_tensor(v) and v.dim() > 0 and v.size(0) >= len(self.geometry_bank)}}
#         return self.geometry_bank.snapshot_rows_for_transport(class_ids)

#     @torch.no_grad()
#     def restore_rows_from_transport_snapshot(self, snapshot: Dict[str, torch.Tensor]) -> None:
#         """Restore compact bank rows after rejected transport."""
#         if hasattr(self.geometry_bank, "restore_rows_from_transport_snapshot"):
#             self.geometry_bank.restore_rows_from_transport_snapshot(snapshot)
#             return
#         if not isinstance(snapshot, dict) or "ids" not in snapshot:
#             raise RuntimeError("Invalid transport snapshot.")
#         ids = snapshot["ids"].to(self.device).long()
#         for key, value in snapshot.items():
#             if key == "ids" or not hasattr(self.geometry_bank, key):
#                 continue
#             target = getattr(self.geometry_bank, key)
#             if torch.is_tensor(target) and torch.is_tensor(value) and target.dim() > 0:
#                 target.index_copy_(0, ids, value.to(device=target.device, dtype=target.dtype))
#         self.validate_geometry_memory(strict=True)

#     @torch.no_grad()
#     def transport_frozen_geometry(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
#         """Apply HSI low-rank residual transport to frozen old GeometryBank rows.

#         Estimation of ``A,b`` happens in ``geometry_transport.py``.  This wrapper
#         injects the model's transport contract before delegating to the bank:
#         strong EMA, reliability gate, low-rank residual projection, bounded
#         translation, and frozen-row enforcement.
#         """
#         if not bool(getattr(self, "use_sglat_transport", False) or getattr(self, "use_geometry_transport", False)):
#             raise RuntimeError("SGLAT transport is disabled on the model. Enable use_sglat_transport explicitly.")
#         if not hasattr(self.geometry_bank, "transport_frozen_geometry"):
#             raise RuntimeError("Updated GeometryBank with transport_frozen_geometry() is required for HSI transport.")
#         defaults = dict(getattr(self, "transport_defaults", {}))
#         for k, v in defaults.items():
#             kwargs.setdefault(k, v)
#         # Older bank versions do not accept the new guard keys; filter safely.
#         filtered = _filter_supported_kwargs(self.geometry_bank.transport_frozen_geometry, kwargs)
#         return self.geometry_bank.transport_frozen_geometry(*args, **filtered)

#     @torch.no_grad()
#     def build_candidate_geometry_rows(self, *args: Any, **kwargs: Any) -> Dict[int, Dict[str, torch.Tensor]]:
#         if not hasattr(self.geometry_bank, "build_candidate_geometry_rows"):
#             raise RuntimeError("GeometryBank must expose build_candidate_geometry_rows() for admission.")
#         return self.geometry_bank.build_candidate_geometry_rows(*args, **kwargs)

#     @torch.no_grad()
#     def candidate_old_new_risk_report(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
#         """Return old/new bank conflict without requiring a bank-specific helper.

#         Preferred path delegates to GeometryBank.  Fallback computes risk from the
#         committed compact rows, which is enough for trainer stop/reject gates.
#         """
#         if hasattr(self.geometry_bank, "candidate_old_new_risk_report"):
#             return self.geometry_bank.candidate_old_new_risk_report(*args, **kwargs)

#         old_count = int(kwargs.get("old_class_count", self.old_class_count))
#         new_ids = kwargs.get("new_class_ids", kwargs.get("class_ids", None))
#         old_count = max(0, min(old_count, len(self.geometry_bank)))
#         if new_ids is None:
#             new_ids = list(range(old_count, len(self.geometry_bank)))
#         new_ids = [int(c) for c in new_ids if old_count <= int(c) < len(self.geometry_bank)]
#         if old_count <= 0 or not new_ids:
#             return {"active": 0, "safe": True, "max_old_new_risk": 0.0, "max_old_new_overlap": 0.0, "new_class_ids": new_ids}

#         risk = self.geometry_bank.geometry_conflict_matrix() if hasattr(self.geometry_bank, "geometry_conflict_matrix") else None
#         overlap = self.geometry_bank.pairwise_subspace_overlap() if hasattr(self.geometry_bank, "pairwise_subspace_overlap") else None
#         new_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
#         if torch.is_tensor(risk) and risk.numel() > 0:
#             ron = risk[:old_count].index_select(1, new_t)
#             max_risk = float(ron.max().detach().cpu().item()) if ron.numel() else 0.0
#             mean_risk = float(ron.mean().detach().cpu().item()) if ron.numel() else 0.0
#         else:
#             max_risk = mean_risk = 0.0
#         if torch.is_tensor(overlap) and overlap.numel() > 0:
#             oon = overlap[:old_count].index_select(1, new_t)
#             max_overlap = float(oon.max().detach().cpu().item()) if oon.numel() else 0.0
#             mean_overlap = float(oon.mean().detach().cpu().item()) if oon.numel() else 0.0
#         else:
#             max_overlap = mean_overlap = 0.0
#         risk_thr = float(kwargs.get("max_old_new_risk", getattr(self.args, "max_old_new_risk", 1.0)))
#         overlap_thr = float(kwargs.get("max_old_new_overlap", getattr(self.args, "max_old_new_overlap", 0.55)))
#         return {
#             "active": 1,
#             "old_class_count": old_count,
#             "new_class_ids": new_ids,
#             "max_old_new_risk": max_risk,
#             "mean_old_new_risk": mean_risk,
#             "old_new_risk_max": max_risk,
#             "old_new_risk_mean": mean_risk,
#             "max_old_new_overlap": max_overlap,
#             "mean_old_new_overlap": mean_overlap,
#             "old_new_subspace_overlap_max": max_overlap,
#             "old_new_subspace_overlap_mean": mean_overlap,
#             "safe": bool(max_risk <= risk_thr and max_overlap <= overlap_thr),
#         }

#     @torch.no_grad()
#     def correct_candidate_rows_against_old(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
#         """Correct risky new compact descriptors against frozen old rows."""
#         if hasattr(self.geometry_bank, "correct_candidate_rows_against_old"):
#             return self.geometry_bank.correct_candidate_rows_against_old(*args, **kwargs)
#         if not hasattr(self.geometry_bank, "correct_new_descriptors_against_old"):
#             raise RuntimeError("GeometryBank must expose descriptor correction for old/new admission.")
#         old_count = int(kwargs.pop("old_class_count", self.old_class_count))
#         new_ids = kwargs.pop("new_class_ids", kwargs.pop("class_ids", None))
#         if new_ids is None:
#             new_ids = list(range(old_count, len(self.geometry_bank)))
#         supported = _filter_supported_kwargs(self.geometry_bank.correct_new_descriptors_against_old, kwargs)
#         return self.geometry_bank.correct_new_descriptors_against_old(old_count, new_ids, **supported)

#     @torch.no_grad()
#     def transport_effect_report(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
#         """Model wrapper for classifier-side before/after transport evaluation."""
#         if not hasattr(self.classifier, "transport_effect_report"):
#             raise RuntimeError("Updated classifier with transport_effect_report() is required.")
#         return self.classifier.transport_effect_report(*args, **kwargs)

#     @torch.no_grad()
#     def commit_candidate_geometry_rows(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
#         if not hasattr(self.geometry_bank, "commit_candidate_geometry_rows"):
#             raise RuntimeError("GeometryBank must expose commit_candidate_geometry_rows() for SGLAT admission.")
#         out = self.geometry_bank.commit_candidate_geometry_rows(*args, **kwargs)
#         self.current_num_classes = max(self.current_num_classes, len(self.geometry_bank))
#         return out

#     @torch.no_grad()
#     def validate_geometry_memory(self, strict: bool = True) -> Dict[str, Any]:
#         """Validate the current GeometryBank when the bank supports validation."""
#         if hasattr(self.geometry_bank, "validate_consistency"):
#             return self.geometry_bank.validate_consistency(strict=bool(strict))
#         bank = self.get_subspace_bank()
#         required = ("means", "bases", "variances", "sample_counts")
#         missing = [k for k in required if k not in bank or not torch.is_tensor(bank[k])]
#         ok = len(missing) == 0
#         if strict and not ok:
#             raise RuntimeError("GeometryBank is missing required tensors: " + ", ".join(missing))
#         return {"ok": ok, "missing": missing, "num_rows": len(self.geometry_bank)}

#     @torch.no_grad()
#     def geometry_memory_ready(self) -> bool:
#         """Return True if at least one valid GeometryBank row is available."""
#         try:
#             return bool(self._bank_is_ready_for_energy(self.get_subspace_bank()))
#         except Exception:
#             return False

#     def calibration_regularization_loss(self, old_class_count: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         del old_class_count
#         z = _zero_scalar(self.device)
#         return {"total": z, "mean": z, "var": z}

#     # ------------------------------------------------------------------
#     # Energy / logits
#     # ------------------------------------------------------------------
#     def _bank_is_ready_for_energy(self, bank: Optional[Dict[str, torch.Tensor]] = None) -> bool:
#         bank = self.get_subspace_bank() if bank is None else bank
#         for key in ("means", "bases", "variances", "sample_counts"):
#             if key not in bank or not torch.is_tensor(bank[key]) or bank[key].numel() == 0:
#                 return False
#         return bool((bank["sample_counts"].detach().flatten() > 0).any().item())

#     def compute_current_geometry_energy(
#         self,
#         features: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor] = None,
#         *,
#         bank: Optional[Dict[str, torch.Tensor]] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#         classifier_mode: Optional[str] = None,
#         old_class_count: Optional[int] = None,
#         return_parts: bool = False,
#         strict: bool = True,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         """Compute decision-aligned geometry energy.

#         This method used to bypass ``old_class_count`` and always pass spectral
#         summaries when available. That created a silent mismatch:

#             forward()/logits path:      geometry_only + old/new barrier
#             energy/health path:         raw bank energy without old/new barrier

#         The Phase-2 runs kept looking identical because the evaluation/health
#         code could still score the unsafe raw energy even after classifier.py was
#         patched.  This wrapper now resolves the same classifier mode and
#         old_class_count used by final logits.
#         """
#         self.assert_incremental_feature_contract()
#         features = self._validate_feature_tensor(features, "compute_current_geometry_energy.features", int(features.size(0)))
#         bank = self.get_subspace_bank() if bank is None else bank
#         if not self._bank_is_ready_for_energy(bank):
#             if strict:
#                 raise RuntimeError("GeometryBank is not ready. Build/refresh it from projected z first.")
#             empty = torch.empty((features.size(0), 0), device=features.device, dtype=features.dtype)
#             return {"energy": empty} if return_parts else empty

#         mode = self._resolve_classifier_mode(classifier_mode)
#         if mode not in {"geometry_only", "srgp"}:
#             raise RuntimeError(f"compute_current_geometry_energy supports classifier_mode in {{'geometry_only','srgp'}}, got {mode!r}.")
#         oc = int(self.old_class_count if old_class_count is None else old_class_count)
#         spectral_for_call = (
#             spectral_summary
#             if mode == "srgp" and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0
#             else None
#         )
#         physical_for_call = bool(
#             False if spectral_for_call is None
#             else (self.spectral_summary_is_physical if spectral_summary_is_physical is None else spectral_summary_is_physical)
#         )

#         if hasattr(self.classifier, "geometry_energy_from_bank"):
#             try:
#                 return self.classifier.geometry_energy_from_bank(
#                     features=features,
#                     bank=bank,
#                     spectral_summary=spectral_for_call,
#                     spectral_summary_is_physical=physical_for_call,
#                     old_class_count=oc,
#                     return_parts=return_parts,
#                 )
#             except TypeError as exc:
#                 # Compatibility fallback for older classifier.py.  Do not hide
#                 # unrelated TypeErrors from inside classifier code.
#                 if "old_class_count" not in str(exc):
#                     raise
#                 return self.classifier.geometry_energy_from_bank(
#                     features=features,
#                     bank=bank,
#                     spectral_summary=spectral_for_call,
#                     spectral_summary_is_physical=physical_for_call,
#                     return_parts=return_parts,
#                 )

#         kwargs = dict(
#             features=features,
#             means=bank["means"],
#             bases=bank["bases"],
#             variances=bank["variances"],
#             reliability=bank.get("reliability", None),
#             active_ranks=bank.get("active_ranks", None),
#             sample_counts=bank.get("sample_counts", None),
#             spectral_summary=spectral_for_call,
#             spectral_curve_means=bank.get("spectral_curve_means", None),
#             spectral_curve_vars=bank.get("spectral_curve_vars", None),
#             spectral_curve_d1=bank.get("spectral_curve_d1", None),
#             spectral_curve_d2=bank.get("spectral_curve_d2", None),
#             spectral_shape_reliability=bank.get("spectral_shape_reliability", None),
#             spectral_summary_is_physical=physical_for_call,
#             old_class_count=oc,
#             return_parts=return_parts,
#         )
#         try:
#             return self.classifier.geometry_energy(**kwargs)
#         except TypeError as exc:
#             if "old_class_count" not in str(exc):
#                 raise
#             kwargs.pop("old_class_count", None)
#             return self.classifier.geometry_energy(**kwargs)

#     def geometry_energy_for_projected_batch(
#         self,
#         projected_out: Dict[str, torch.Tensor],
#         *,
#         classifier_mode: Optional[str] = None,
#         old_class_count: Optional[int] = None,
#         return_parts: bool = False,
#         strict: bool = True,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         if "features" not in projected_out:
#             raise RuntimeError("projected_out must contain key 'features'.")
#         return self.compute_current_geometry_energy(
#             projected_out["features"],
#             spectral_summary=projected_out.get("spectral_summary", None),
#             spectral_summary_is_physical=bool(projected_out.get("spectral_summary_is_physical", torch.tensor(self.spectral_summary_is_physical)).detach().cpu().item())
#             if torch.is_tensor(projected_out.get("spectral_summary_is_physical", None)) else self.spectral_summary_is_physical,
#             classifier_mode=classifier_mode,
#             old_class_count=old_class_count,
#             return_parts=return_parts,
#             strict=strict,
#         )

#     def compute_logits_from_features(
#         self,
#         features: torch.Tensor,
#         classifier_mode: str = "geometry_only",
#         return_energy: bool = False,
#         return_parts: bool = False,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#     ):
#         mode = self._resolve_classifier_mode(classifier_mode)
#         if mode not in {"geometry_only", "srgp"}:
#             raise RuntimeError(f"SRGP NECIL-HSI supports classifier_mode in {{'geometry_only','srgp'}}, got {mode!r}.")
#         self.assert_incremental_feature_contract()
#         features = self._validate_feature_tensor(features, "compute_logits_from_features.features", int(features.size(0)))
#         # Features passed here must already be canonical projected z. Re-normalizing
#         # synthetic replay samples would distort the stored GeometryBank distribution.
#         bank = self.get_subspace_bank()
#         calibrated_old = self.get_calibrated_old_subspace_bank(self.old_class_count)
#         spectral_for_call = spectral_summary if mode == "srgp" and torch.is_tensor(spectral_summary) and spectral_summary.numel() > 0 else None
#         physical_for_call = bool(
#             False if spectral_for_call is None
#             else (self.spectral_summary_is_physical if spectral_summary_is_physical is None else spectral_summary_is_physical)
#         )

#         return self.classifier(
#             features,
#             geometry_bank=bank,
#             subspace_means=bank["means"] if bank["means"].numel() > 0 else None,
#             subspace_bases=bank["bases"] if bank["bases"].numel() > 0 else None,
#             subspace_variances=bank["variances"] if bank["variances"].numel() > 0 else None,
#             subspace_reliability=bank.get("reliability", None),
#             subspace_active_ranks=bank.get("active_ranks", None),
#             subspace_sample_counts=bank.get("sample_counts", None),
#             calibrated_old_means=calibrated_old.get("means", None),
#             calibrated_old_bases=calibrated_old.get("bases", None),
#             calibrated_old_variances=calibrated_old.get("variances", None),
#             calibrated_old_reliability=calibrated_old.get("reliability", None),
#             calibrated_old_active_ranks=calibrated_old.get("active_ranks", None),
#             calibrated_old_sample_counts=calibrated_old.get("sample_counts", None),
#             spectral_summary=spectral_for_call,
#             spectral_summary_is_physical=physical_for_call,
#             mode=mode,
#             old_class_count=int(self.old_class_count),
#             return_energy=return_energy,
#             return_parts=return_parts,
#         )

#     def compute_energy_from_features(
#         self,
#         features: torch.Tensor,
#         classifier_mode: str = "geometry_only",
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#         return_parts: bool = False,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         out = self.compute_logits_from_features(
#             features=features,
#             classifier_mode=classifier_mode,
#             return_energy=True,
#             return_parts=return_parts,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#         )
#         if not isinstance(out, dict) or "energy" not in out:
#             raise RuntimeError("compute_logits_from_features(return_energy=True) did not return energy.")
#         return out if return_parts else out["energy"]

#     def compute_logits_and_energy_from_features(
#         self,
#         features: torch.Tensor,
#         classifier_mode: str = "geometry_only",
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#         return_parts: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         out = self.compute_logits_from_features(
#             features=features,
#             classifier_mode=classifier_mode,
#             return_energy=True,
#             return_parts=return_parts,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=spectral_summary_is_physical,
#         )
#         if not isinstance(out, dict):
#             raise RuntimeError("Expected dict output when return_energy=True.")
#         return out

#     # ------------------------------------------------------------------
#     # Trainability controls
#     # ------------------------------------------------------------------
#     def freeze_backbone_only(self) -> None:
#         for p in self.backbone.parameters():
#             p.requires_grad = False

#     def unfreeze_backbone(self) -> None:
#         for p in self.backbone.parameters():
#             p.requires_grad = True

#     def freeze_projection_head(self) -> None:
#         for p in self.projection.parameters():
#             p.requires_grad = False
#         for p in self.norm.parameters():
#             p.requires_grad = False

#     def unfreeze_projection_head(self) -> None:
#         for p in self.projection.parameters():
#             p.requires_grad = True
#         for p in self.norm.parameters():
#             p.requires_grad = True

#     def freeze_adapter(self) -> None:
#         self.freeze_geometry_plastic_adapter()

#     def unfreeze_adapter(self) -> None:
#         self.unfreeze_geometry_plastic_adapter()

#     def freeze_energy_calibrator(self) -> None:
#         if hasattr(self.classifier, "freeze_all_adaptation"):
#             self.classifier.freeze_all_adaptation()

#     def unfreeze_energy_calibrator(self) -> None:
#         if hasattr(self.classifier, "unfreeze_all_adaptation"):
#             self.classifier.unfreeze_all_adaptation()

#     def enable_energy_calibration(self, enabled: bool = True, calibrator_type: Optional[str] = None) -> None:
#         if hasattr(self.classifier, "enable_energy_calibration"):
#             self.classifier.enable_energy_calibration(enabled=enabled, calibrator_type=calibrator_type)
#         self.use_energy_calibrator = _as_bool(getattr(self.classifier, "use_energy_calibrator", False), False)
#         self.energy_calibrator_type = str(getattr(self.classifier, "energy_calibrator_type", "none"))

#     def energy_calibration_regularization_loss(self) -> torch.Tensor:
#         if hasattr(self.classifier, "energy_calibration_regularization_loss"):
#             try:
#                 return self.classifier.energy_calibration_regularization_loss(num_classes=int(self.current_num_classes))
#             except TypeError:
#                 return self.classifier.energy_calibration_regularization_loss()
#         return _zero_scalar(self.device)

#     @torch.no_grad()
#     def energy_calibration_state(self) -> Dict[str, float]:
#         if hasattr(self.classifier, "energy_calibration_state"):
#             return self.classifier.energy_calibration_state()
#         return {}

#     # ------------------------------------------------------------------
#     # Adaptive GeometryBank-energy decision boundary
#     # ------------------------------------------------------------------
#     def adaptive_boundary_enabled(self) -> bool:
#         return bool(getattr(self.classifier, "use_adaptive_boundary", False))

#     @torch.no_grad()
#     def ensure_adaptive_boundary_capacity(self, class_count: Optional[int] = None) -> None:
#         """Ensure the classifier owns one boundary radius for every seen class."""
#         count = int(self.current_num_classes if class_count is None else class_count)
#         count = max(0, count)
#         if hasattr(self.classifier, "_ensure_boundary_capacity"):
#             dtype = self.projection[0].weight.dtype
#             self.classifier._ensure_boundary_capacity(count, device=self.device, dtype=dtype)
#         elif hasattr(self.classifier, "expand"):
#             while int(getattr(self.classifier, "num_classes", 0)) < count:
#                 self.classifier.expand(1, self.current_phase)

#     def adaptive_boundary_parameters(self):
#         """Return trainable adaptive-boundary parameters for optimizer groups."""
#         if not self.adaptive_boundary_enabled() or not hasattr(self.classifier, "boundary_parameters"):
#             return []
#         self.ensure_adaptive_boundary_capacity(self.current_num_classes)
#         return list(self.classifier.boundary_parameters())

#     def freeze_all_boundary_radii(self) -> None:
#         if hasattr(self.classifier, "freeze_all_boundary_radii"):
#             self.classifier.freeze_all_boundary_radii()

#     def unfreeze_all_boundary_radii(self) -> None:
#         if hasattr(self.classifier, "unfreeze_all_boundary_radii"):
#             self.classifier.unfreeze_all_boundary_radii()

#     def freeze_old_boundary_radii(self, old_class_count: Optional[int] = None) -> None:
#         """Freeze old decision radii while leaving new radii trainable.

#         PyTorch cannot set requires_grad for only part of a Parameter, so the
#         classifier uses a gradient hook.  This wrapper is what the incremental
#         trainer should call before building the optimizer.
#         """
#         count = int(self.old_class_count if old_class_count is None else old_class_count)
#         self.ensure_adaptive_boundary_capacity(self.current_num_classes)
#         if hasattr(self.classifier, "freeze_old_boundary_radii"):
#             self.classifier.freeze_old_boundary_radii(count)

#     def adaptive_boundary_loss(
#         self,
#         *,
#         old_class_count: Optional[int] = None,
#         bank: Optional[Dict[str, torch.Tensor]] = None,
#         num_classes: Optional[int] = None,
#     ) -> torch.Tensor:
#         """Geometry-risk constraint for class-wise decision radii.

#         The loss is intentionally classifier-owned: it uses GeometryBank bases,
#         active ranks, sample counts, and old_class_count to stop risky new rows
#         from expanding their decision radius into frozen old basins.
#         """
#         if not self.adaptive_boundary_enabled() or not hasattr(self.classifier, "adaptive_boundary_loss"):
#             return _zero_scalar(self.device)
#         bank = self.get_subspace_bank() if bank is None else bank
#         count = int(self.old_class_count if old_class_count is None else old_class_count)
#         C = int(num_classes if num_classes is not None else (bank["sample_counts"].numel() if "sample_counts" in bank and torch.is_tensor(bank["sample_counts"]) else self.current_num_classes))
#         self.ensure_adaptive_boundary_capacity(C)
#         return self.classifier.adaptive_boundary_loss(
#             old_class_count=count,
#             bases=bank.get("bases", None),
#             active_ranks=bank.get("active_ranks", None),
#             sample_counts=bank.get("sample_counts", None),
#             num_classes=C,
#         )

#     @torch.no_grad()
#     def adaptive_boundary_state(self, old_class_count: Optional[int] = None) -> Dict[str, float]:
#         if not hasattr(self.classifier, "adaptive_boundary_state"):
#             return {
#                 "adaptive_boundary_enabled": 0.0,
#                 "boundary_radius_mean": 0.0,
#                 "old_boundary_radius_mean": 0.0,
#                 "new_boundary_radius_mean": 0.0,
#             }
#         count = int(self.old_class_count if old_class_count is None else old_class_count)
#         try:
#             bank = self.get_subspace_bank()
#             C = int(bank["sample_counts"].numel()) if torch.is_tensor(bank.get("sample_counts", None)) else int(self.current_num_classes)
#         except Exception:
#             C = int(self.current_num_classes)
#         self.ensure_adaptive_boundary_capacity(C)
#         return self.classifier.adaptive_boundary_state(num_classes=C, old_class_count=count)

#     # Geometry-cycle compatibility hooks. Clean NECIL-HSI has no BiCyc/transport path.
#     def freeze_geometry_calibrator(self) -> None:
#         return None

#     def unfreeze_geometry_calibrator(self) -> None:
#         raise RuntimeError(
#             "Geometry-cycle/BiCyc calibration is not part of the clean NECIL-HSI model. "
#             "Use descriptor-only new-row refinement in the incremental trainer instead."
#         )

#     def geometry_cycle_old_to_new(self, features: torch.Tensor) -> torch.Tensor:
#         return self._validate_feature_tensor(features, "geometry_cycle_old_to_new.features", int(features.size(0)))

#     def geometry_cycle_new_to_old(self, features: torch.Tensor) -> torch.Tensor:
#         return self._validate_feature_tensor(features, "geometry_cycle_new_to_old.features", int(features.size(0)))

#     def geometry_cycle_loss(self, old_features: Optional[torch.Tensor] = None, new_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         ref = old_features if torch.is_tensor(old_features) and old_features.numel() > 0 else new_features
#         if not torch.is_tensor(ref):
#             z = _zero_scalar(self.device)
#         else:
#             z = ref.sum() * 0.0
#         return {"total": z, "old_cycle": z, "new_cycle": z, "reg": z}

#     @torch.no_grad()
#     def geometry_cycle_state(self) -> Dict[str, float]:
#         return {"enabled": 0.0, "max_delta_scale": 0.0, "trainable_tensors": 0.0}

#     def freeze_incremental_adapter(self) -> None: self.freeze_geometry_plastic_adapter()
#     def unfreeze_incremental_adapter(self) -> None: self.unfreeze_geometry_plastic_adapter()
#     def enable_incremental_adapter(self) -> None:
#         # Legacy name.  Do not enable stale adapter flags; enable only the
#         # geometry-gated adapter if that architecture was selected.
#         self.use_incremental_adapter = False
#         if self.use_geometry_gated_adapter:
#             self.unfreeze_geometry_plastic_adapter()
#     def disable_incremental_adapter(self) -> None:
#         self.use_incremental_adapter = False
#         self.freeze_geometry_plastic_adapter()
#     def freeze_semantic_encoder(self) -> None: return None
#     def unfreeze_semantic_encoder(self) -> None: return None
#     def freeze_old_anchor_deltas(self, old_class_count: int) -> None: del old_class_count; return None
#     def unfreeze_new_anchor_deltas(self, old_class_count: int) -> None: del old_class_count; return None
#     def freeze_old_concept_deltas(self, old_class_count: int) -> None: del old_class_count; return None
#     def unfreeze_new_concept_deltas(self, old_class_count: int) -> None: del old_class_count; return None
#     def freeze_classifier_adaptation(self) -> None: self.freeze_energy_calibrator()
#     def freeze_old_classifier_adaptation(self, old_class_count: int) -> None: del old_class_count; self.freeze_energy_calibrator()
#     def unfreeze_classifier_adaptation(self) -> None: self.unfreeze_energy_calibrator()
#     # Explicit aliases used by adaptive-boundary incremental trainers.
#     def freeze_adaptive_boundary(self) -> None: self.freeze_all_boundary_radii()
#     def unfreeze_adaptive_boundary(self) -> None: self.unfreeze_all_boundary_radii()
#     def freeze_old_adaptive_boundary(self, old_class_count: int) -> None: self.freeze_old_boundary_radii(old_class_count)
#     def freeze_fusion_module(self) -> None: return None
#     def unfreeze_fusion_module(self) -> None: return None

#     # ------------------------------------------------------------------
#     # Base/incremental feature contract
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def geometry_feature_contract(self) -> Dict[str, Any]:
#         """Serializable contract tying base and incremental phases together."""
#         return {
#             "d_model": int(self.d_model),
#             "subspace_rank": int(self.subspace_rank),
#             "normalize_geometry_features": bool(self.normalize_geometry_features),
#             "geometry_feature_scale": float(self.geometry_feature_scale),
#             "geometry_feature_clamp": float(self.geometry_feature_clamp),
#             "spectral_summary_mode": str(self.spectral_summary_mode),
#             "spectral_summary_is_physical": bool(self.spectral_summary_is_physical),
#             "base_classifier_mode": str(self.default_base_classifier_mode),
#             "incremental_classifier_mode": str(self.default_incremental_classifier_mode),
#             "eval_classifier_mode": str(self.default_eval_classifier_mode),
#             "has_incremental_adapter": bool(self.adapter_enabled()),
#             "incremental_update_mode": str(self.incremental_update_mode),
#             "adapter_max_scale": float(getattr(self.geometry_plastic_adapter, "max_scale", 0.0)),
#             "has_geometry_transport": bool(getattr(self, "use_geometry_transport", False)),
#             "has_sglat_transport": bool(getattr(self, "use_sglat_transport", False)),
#             "allow_old_model_transport": bool(getattr(self, "allow_old_model_transport", True)),
#             "has_spectral_classifier_branch": True,
#             "has_old_new_overlap_barrier": bool(getattr(self.classifier, "use_old_new_overlap_barrier", False)),
#             "old_new_overlap_barrier_weight": float(getattr(self.classifier, "old_new_overlap_barrier_weight", 0.0)),
#             "old_new_overlap_barrier_threshold": float(getattr(self.classifier, "old_new_overlap_barrier_threshold", 0.0)),
#             "has_adaptive_boundary": bool(getattr(self.classifier, "use_adaptive_boundary", False)),
#             "adaptive_boundary_radius_min": float(getattr(self.classifier, "boundary_radius_min", 0.0)),
#             "adaptive_boundary_radius_max": float(getattr(self.classifier, "boundary_radius_max", 0.0)),
#             "adaptive_boundary_state": self.adaptive_boundary_state(int(self.old_class_count)) if hasattr(self, "adaptive_boundary_state") else {},
#         }

#     def assert_incremental_feature_contract(self) -> None:
#         """Fail loudly if any forbidden feature-space plasticity path is active."""
#         if bool(getattr(self, "use_incremental_adapter", False)):
#             raise RuntimeError("Incremental adapter is active; this violates the clean base-to-incremental geometry contract.")
#         if bool(getattr(self, "use_geometry_calibrator", False)):
#             raise RuntimeError("Legacy geometry calibrator is active; SGLAT uses explicit bank transport, not a calibrator head.")
#         if self.adapter_enabled() and float(getattr(self.geometry_plastic_adapter, "max_scale", 0.0)) <= 0.0:
#             raise RuntimeError("geometry_gated_adapter selected but adapter_max_scale <= 0; no plasticity is possible.")
#         if not self.normalize_geometry_features and self.strict_feature_contract:
#             raise RuntimeError("normalize_geometry_features is disabled; GeometryBank covariance scale may drift across phases.")

#     def _assert_snapshot_contract_compatible(self, snapshot: Dict[str, Any], *, strict: bool) -> None:
#         if not strict:
#             return
#         contract = snapshot.get("geometry_feature_contract", None)
#         if not isinstance(contract, dict):
#             return
#         current = self.geometry_feature_contract()
#         checks = (
#             "d_model",
#             "subspace_rank",
#             "normalize_geometry_features",
#             "spectral_summary_mode",
#             "spectral_summary_is_physical",
#         )
#         mismatches = []
#         for key in checks:
#             if key in contract and contract[key] != current[key]:
#                 mismatches.append(f"{key}: snapshot={contract[key]!r}, current={current[key]!r}")
#         if "geometry_feature_scale" in contract:
#             if abs(float(contract["geometry_feature_scale"]) - float(current["geometry_feature_scale"])) > 1e-6:
#                 mismatches.append(
#                     f"geometry_feature_scale: snapshot={contract['geometry_feature_scale']!r}, "
#                     f"current={current['geometry_feature_scale']!r}"
#                 )
#         if mismatches:
#             raise RuntimeError(
#                 "Loaded GeometryBank was built under a different feature contract. "
#                 "Do not evaluate/increment with a mismatched z-space. "
#                 + "; ".join(mismatches)
#             )

#     # ------------------------------------------------------------------
#     # Snapshot
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def export_memory_snapshot(self) -> Dict[str, Any]:
#         snap = self.geometry_bank.export_snapshot()
#         snap.update(
#             {
#                 "current_num_classes": int(self.current_num_classes),
#                 "old_class_count": int(self.old_class_count),
#                 "current_phase": int(self.current_phase),
#                 "base_ce_head_present": False,
#                 "base_prl_head_present": False,
#                 "use_incremental_adapter": False,
#                 "use_geometry_gated_adapter": bool(self.adapter_enabled()),
#                 "incremental_update_mode": str(self.incremental_update_mode),
#                 "incremental_adapter_active": bool(self.incremental_adapter_active()),
#                 "use_geometry_calibrator": False,
#                 "use_sglat_transport": bool(getattr(self, "use_sglat_transport", False)),
#                 "use_geometry_transport": bool(getattr(self, "use_geometry_transport", False)),
#                 "allow_old_model_transport": bool(getattr(self, "allow_old_model_transport", True)),
#                 "use_energy_calibrator": _as_bool(getattr(self.classifier, "use_energy_calibrator", False), False),
#                 "energy_calibrator_type": str(getattr(self.classifier, "energy_calibrator_type", "none")),
#                 "energy_calibration_state": self.energy_calibration_state(),
#                 "geometry_feature_contract": self.geometry_feature_contract(),
#                 "clean_geometry_model": True,
#                 "use_old_new_overlap_barrier": bool(getattr(self.classifier, "use_old_new_overlap_barrier", False)),
#                 "old_new_overlap_barrier_weight": float(getattr(self.classifier, "old_new_overlap_barrier_weight", 0.0)),
#                 "old_new_overlap_barrier_threshold": float(getattr(self.classifier, "old_new_overlap_barrier_threshold", 0.0)),
#                 "use_adaptive_boundary": bool(getattr(self.classifier, "use_adaptive_boundary", False)),
#                 "adaptive_boundary_state": self.adaptive_boundary_state(int(self.old_class_count)) if hasattr(self, "adaptive_boundary_state") else {},
#                 "boundary_log_radius": getattr(self.classifier, "boundary_log_radius", torch.empty(0, device=self.device)).detach().clone()
#                 if torch.is_tensor(getattr(self.classifier, "boundary_log_radius", None)) else torch.empty(0, device=self.device),
#             }
#         )
#         return snap

#     @torch.no_grad()
#     def load_memory_snapshot(self, snapshot: Dict[str, Any], strict: bool = True) -> None:
#         if snapshot is None:
#             if strict:
#                 raise ValueError("snapshot is None")
#             return
#         means = snapshot.get("means", None)
#         class_count = int(means.size(0)) if torch.is_tensor(means) else int(snapshot.get("current_num_classes", 0))
#         _snap_sdim = snapshot.get("spectral_dim", snapshot.get("band_dim", 0))
#         spectral_dim = int(_snap_sdim.item()) if torch.is_tensor(_snap_sdim) else int(_snap_sdim or 0)
#         self._assert_snapshot_contract_compatible(snapshot, strict=strict)
#         self.ensure_class_capacity(class_count=class_count, spectral_dim=spectral_dim)
#         self.geometry_bank.load_snapshot(snapshot, strict=strict)
#         self.current_num_classes = class_count
#         self.old_class_count = int(snapshot.get("old_class_count", self.old_class_count))
#         self.current_phase = int(snapshot.get("current_phase", self.current_phase))
#         if "use_energy_calibrator" in snapshot:
#             self.enable_energy_calibration(
#                 _as_bool(snapshot.get("use_energy_calibrator", False), False),
#                 calibrator_type=str(snapshot.get("energy_calibrator_type", getattr(self.classifier, "energy_calibrator_type", "none"))),
#             )
#         # Restore adaptive-boundary radii for memory-snapshot workflows.  Normal
#         # PyTorch checkpoints restore this through state_dict, but exported
#         # GeometryBank snapshots would otherwise lose the learned decision radii.
#         b_log = snapshot.get("boundary_log_radius", None)
#         if torch.is_tensor(b_log) and hasattr(self.classifier, "boundary_log_radius"):
#             self.ensure_adaptive_boundary_capacity(class_count)
#             with torch.no_grad():
#                 dst = self.classifier.boundary_log_radius
#                 n = min(int(dst.numel()), int(b_log.numel()))
#                 if n > 0:
#                     dst[:n].copy_(b_log[:n].to(device=dst.device, dtype=dst.dtype))
#         self.drop_base_ce_head()

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------
#     def forward(self, x: torch.Tensor, **kwargs):
#         classifier_mode = self._resolve_classifier_mode(kwargs.get("classifier_mode", None))
#         return_energy = _as_bool(kwargs.get("return_energy", False), False)
#         return_parts = _as_bool(kwargs.get("return_parts", False), False)

#         projected_out = self.extract_projected_features(
#             x,
#             spectral_summary=kwargs.get("spectral_summary", None),
#             band_weights=kwargs.get("band_weights", None),
#             spectral_summary_is_physical=kwargs.get("spectral_summary_is_physical", None),
#         )
#         projected = projected_out["features"]

#         out_logits = self.compute_logits_from_features(
#             projected,
#             classifier_mode=classifier_mode,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             spectral_summary=projected_out.get("spectral_summary", None),
#             spectral_summary_is_physical=bool(projected_out.get("spectral_summary_is_physical", torch.tensor(self.spectral_summary_is_physical)).detach().cpu().item())
#             if torch.is_tensor(projected_out.get("spectral_summary_is_physical", None)) else self.spectral_summary_is_physical,
#         )

#         if isinstance(out_logits, dict):
#             logits = out_logits["logits"]
#             energy = out_logits.get("energy", None)
#             raw_energy = out_logits.get("raw_energy", None)
#             energy_calibrated = out_logits.get("energy_calibrated", None)
#         else:
#             logits = out_logits
#             energy = raw_energy = energy_calibrated = None

#         bank = self.get_subspace_bank()
#         energy_cal_reg = self.energy_calibration_regularization_loss()
#         zero = _zero_scalar(projected.device, projected.dtype)
#         cal_reg = {"total": energy_cal_reg, "mean": zero, "var": zero, "energy": energy_cal_reg, "energy_cal": energy_cal_reg}

#         out = {
#             "logits": logits,
#             "features": projected,
#             "projected_features": projected,
#             "pre_adapter_features": projected_out.get("pre_adapter_features", projected),
#             "base_features": projected_out.get("base_features", projected_out.get("pre_adapter_features", projected)),
#             "adapter_delta": projected_out.get("adapter_delta", torch.zeros_like(projected)),
#             "adapter_gate": projected_out.get("adapter_gate", torch.zeros((projected.size(0), 1), device=projected.device, dtype=projected.dtype)),
#             "adapter_active": torch.tensor(float(self.incremental_adapter_active()), device=projected.device, dtype=projected.dtype),
#             "preproject_features": projected_out["preproject_features"],
#             "backbone_features": projected_out["backbone_features"],
#             "band_weights": projected_out.get("band_weights", None),
#             "band_summary": projected_out.get("band_summary", None),
#             "band_importance": projected_out.get("band_importance", projected_out.get("band_summary", None)),
#             "spectral_summary": projected_out["spectral_summary"],
#             "spectral_summary_is_physical": projected_out.get("spectral_summary_is_physical", torch.tensor(self.spectral_summary_is_physical, device=projected.device, dtype=torch.bool)),
#             "subspace_means": bank["means"],
#             "subspace_bases": bank["bases"],
#             "subspace_variances": bank["variances"],
#             "subspace_reliability": bank.get("reliability", None),
#             "subspace_active_ranks": bank.get("active_ranks", None),
#             "subspace_sample_counts": bank.get("sample_counts", None),
#             "band_importances": bank.get("band_importances", None),
#             "spectral_curve_means": bank.get("spectral_curve_means", None),
#             "spectral_curve_vars": bank.get("spectral_curve_vars", None),
#             "spectral_curve_d1": bank.get("spectral_curve_d1", None),
#             "spectral_curve_d2": bank.get("spectral_curve_d2", None),
#             "spectral_shape_reliability": bank.get("spectral_shape_reliability", None),

#             "spectral_means": bank.get("spectral_means", None),
#             "spectral_protos": bank.get("spectral_protos", None),
#             "spectral_bases": bank.get("spectral_bases", None),
#             "spectral_variances": bank.get("spectral_variances", None),
#             "spectral_reliability": bank.get("spectral_reliability", None),
#             "spectral_active_ranks": bank.get("spectral_active_ranks", None),
#             "calibration_reg": cal_reg,
#             "energy_calibration_state": self.energy_calibration_state(),
#             "adaptive_boundary_state": self.adaptive_boundary_state(int(self.old_class_count)) if hasattr(self, "adaptive_boundary_state") else {},
#         }
#         if energy is not None:
#             out["energy"] = energy
#         if raw_energy is not None:
#             out["raw_energy"] = raw_energy
#         if energy_calibrated is not None:
#             out["energy_calibrated"] = energy_calibrated
#         if isinstance(out_logits, dict) and return_parts:
#             for k, v in out_logits.items():
#                 if k not in out:
#                     out[k] = v
#         return out
