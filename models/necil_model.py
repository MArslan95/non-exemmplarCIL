from __future__ import annotations

"""Integrated NECIL model for transport-verified HSI factor geometry.

Persistent class memory
-----------------------
    p(z | c) = N(mu_c, L_c L_c^T + Psi_c)

where
    z = [z_s ; z_p]
    Psi_c = diag(psi_c,s I_Ds, psi_c,p I_Dp).

The raw ordered-spectrum relation p(h | c) is stored only for class-pair risk.
It never becomes a second inference classifier.

Incremental representation evolution
------------------------------------
The backbone may use either:

1. a frozen incremental baseline; or
2. controlled late-layer plasticity.

For controlled plasticity, a temporary frozen phase-start observer is applied
only to current-phase samples. A branchwise similarity transform is fitted
analytically on support pairs, validated on disjoint current pairs, and then
used to transport aggregate old rows exactly. There is no trainable transport
network and no old-feature replay objective.
"""

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from models.backbone import SSMBackbone
from models.classifier import GeometryEnergyClassifier
from models.geometry_bank import ( BranchSimilarityTransform, GeometryBank, )


Tensor = torch.Tensor
Row = Dict[str, Tensor]


# =============================================================================
# Shared validation
# =============================================================================


def _unique_ids(
    values: Iterable[int],
    *,
    name: str,
    allow_empty: bool = False,
) -> List[int]:
    output: List[int] = []
    observed: set[int] = set()
    for value in values:
        class_id = int(value)
        if class_id < 0:
            raise ValueError(f"{name} contains negative class ID {class_id}")
        if class_id in observed:
            raise ValueError(f"{name} contains duplicate class ID {class_id}")
        observed.add(class_id)
        output.append(class_id)
    if not output and not allow_empty:
        raise ValueError(f"{name} is empty")
    return output


def _feature_matrix(
    value: Tensor,
    *,
    expected_dim: int,
    name: str,
) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.dim() != 2 or value.size(1) != int(expected_dim):
        raise ValueError(
            f"{name} must be [N,{int(expected_dim)}], "
            f"got {tuple(value.shape)}"
        )
    if value.size(0) == 0:
        raise ValueError(f"{name} is empty")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    return value


def _weights(
    value: Optional[Tensor],
    *,
    sample_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if value is None:
        return torch.ones(sample_count, device=device, dtype=dtype)
    result = torch.as_tensor(
        value,
        device=device,
        dtype=dtype,
    ).flatten()
    if result.numel() != sample_count:
        raise ValueError(
            f"sample_weights has {result.numel()} entries; "
            f"expected {sample_count}"
        )
    if not torch.isfinite(result).all():
        raise RuntimeError("sample_weights contains NaN/Inf")
    if bool(result.lt(0.0).any()):
        raise ValueError("sample_weights must be non-negative")
    if float(result.sum().item()) <= 0.0:
        raise ValueError("sample_weights sums to zero")
    return result


def _relative_improvement(
    reference: float,
    candidate: float,
) -> float:
    return (reference - candidate) / max(reference, 1e-12)


# =============================================================================
# Temporary phase context
# =============================================================================


@dataclass
class IncrementalPhaseContext:
    """Temporary, explicitly managed state for one incremental phase.

    This object is not registered inside ``NECILModel`` and therefore is not
    silently persisted as exemplar memory. A trainer may checkpoint
    ``export_state()`` separately to resume an interrupted phase.
    """

    phase: int
    old_class_ids: List[int]
    new_class_ids: List[int]
    old_rows_digest: str
    bank_contract_digest: str
    observer: SSMBackbone
    parameter_snapshot: Dict[str, Tensor]
    controlled_plasticity: bool

    def export_state(self) -> Dict[str, Any]:
        return {
            "phase": int(self.phase),
            "old_class_ids": list(self.old_class_ids),
            "new_class_ids": list(self.new_class_ids),
            "old_rows_digest": str(self.old_rows_digest),
            "bank_contract_digest": str(self.bank_contract_digest),
            "observer_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.observer.state_dict().items()
            },
            "parameter_snapshot": {
                name: value.detach().cpu().clone()
                for name, value in self.parameter_snapshot.items()
            },
            "controlled_plasticity": bool(self.controlled_plasticity),
        }


# =============================================================================
# Evidence-gated analytical branch transport
# =============================================================================


class EvidenceGatedSimilarityEstimator:
    """Fit and validate branchwise Level-0/1/2 similarity transforms.

    Hierarchy
    ---------
    Level 0:
        identity or translation.

    Level 1:
        positive isotropic scale + translation.

    Level 2:
        positive isotropic scale + proper orthogonal rotation + translation.

    Candidates are fitted on support pairs and selected using disjoint
    validation pairs. More complex levels are accepted only when identifiable
    and when they improve normalized held-out RMSE by a configured amount.
    """

    def __init__(
        self,
        *,
        spectral_dim: int,
        spatial_dim: int,
        minimum_scale_samples: int = 8,
        minimum_rotation_samples: Optional[int] = None,
        minimum_rotation_rank_fraction: float = 0.50,
        rank_tolerance: float = 1e-4,
        minimum_relative_improvement: float = 0.02,
        maximum_normalized_rmse: float = 1.50,
        minimum_scale: float = 0.70,
        maximum_scale: float = 1.30,
    ) -> None:
        self.spectral_dim = int(spectral_dim)
        self.spatial_dim = int(spatial_dim)
        if self.spectral_dim <= 0 or self.spatial_dim <= 0:
            raise ValueError("branch dimensions must be positive")

        self.minimum_scale_samples = int(max(minimum_scale_samples, 3))
        self.minimum_rotation_samples = (
            None
            if minimum_rotation_samples is None
            else int(max(minimum_rotation_samples, 3))
        )
        self.minimum_rotation_rank_fraction = float(
            min(max(minimum_rotation_rank_fraction, 0.0), 1.0)
        )
        self.rank_tolerance = float(max(rank_tolerance, 1e-12))
        self.minimum_relative_improvement = float(
            max(minimum_relative_improvement, 0.0)
        )
        self.maximum_normalized_rmse = float(
            max(maximum_normalized_rmse, 0.0)
        )
        self.minimum_scale = float(max(minimum_scale, 1e-4))
        self.maximum_scale = float(maximum_scale)
        if self.maximum_scale < self.minimum_scale:
            raise ValueError("maximum_scale must be >= minimum_scale")

    @staticmethod
    def _weighted_center(
        value: Tensor,
        weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        normalized = weights / weights.sum().clamp_min(1e-12)
        mean = (normalized[:, None] * value).sum(dim=0)
        return mean, value - mean

    def _effective_rank(
        self,
        centered: Tensor,
        weights: Tensor,
    ) -> int:
        weighted = centered * weights.sqrt().unsqueeze(1)
        singular = torch.linalg.svdvals(weighted)
        if singular.numel() == 0:
            return 0
        threshold = singular.max().clamp_min(1e-12) * self.rank_tolerance
        return int(singular.gt(threshold).sum().item())

    @staticmethod
    def _apply(
        value: Tensor,
        *,
        rotation: Tensor,
        scale: float,
        bias: Tensor,
    ) -> Tensor:
        return (
            value
            @ (float(scale) * rotation).transpose(0, 1)
            + bias.view(1, -1)
        )

    @staticmethod
    def _normalized_rmse(
        predicted: Tensor,
        target: Tensor,
        weights: Tensor,
    ) -> float:
        normalized = weights / weights.sum().clamp_min(1e-12)
        squared_error = (
            predicted - target
        ).square().mean(dim=1)
        mse = (normalized * squared_error).sum()
        target_mean = (
            normalized[:, None] * target
        ).sum(dim=0)
        variance = (
            normalized[:, None]
            * (target - target_mean).square()
        ).sum(dim=0).mean().clamp_min(1e-8)
        return float((mse / variance).sqrt().item())

    @staticmethod
    def _identity_candidate(
        dimension: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        return {
            "rotation": torch.eye(
                dimension,
                device=device,
                dtype=dtype,
            ),
            "scale": 1.0,
            "bias": torch.zeros(
                dimension,
                device=device,
                dtype=dtype,
            ),
            "level": 0,
            "level0_mode": "identity",
        }

    def _translation_candidate(
        self,
        source: Tensor,
        target: Tensor,
        weights: Tensor,
    ) -> Dict[str, Any]:
        source_mean, _ = self._weighted_center(source, weights)
        target_mean, _ = self._weighted_center(target, weights)
        candidate = self._identity_candidate(
            source.size(1),
            device=source.device,
            dtype=source.dtype,
        )
        candidate["bias"] = target_mean - source_mean
        candidate["level0_mode"] = "translation"
        return candidate

    def _scale_candidate(
        self,
        source: Tensor,
        target: Tensor,
        weights: Tensor,
    ) -> Optional[Dict[str, Any]]:
        if source.size(0) < self.minimum_scale_samples:
            return None

        source_mean, source_centered = self._weighted_center(
            source,
            weights,
        )
        target_mean, target_centered = self._weighted_center(
            target,
            weights,
        )
        denominator = (
            weights[:, None] * source_centered.square()
        ).sum().clamp_min(1e-12)
        numerator = (
            weights[:, None]
            * source_centered
            * target_centered
        ).sum()
        scale = float(
            (numerator / denominator)
            .clamp(self.minimum_scale, self.maximum_scale)
            .item()
        )
        rotation = torch.eye(
            source.size(1),
            device=source.device,
            dtype=source.dtype,
        )
        bias = target_mean - scale * source_mean
        return {
            "rotation": rotation,
            "scale": scale,
            "bias": bias,
            "level": 1,
            "level0_mode": "not_applicable",
        }

    def _rotation_candidate(
        self,
        source: Tensor,
        target: Tensor,
        weights: Tensor,
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        dimension = source.size(1)
        minimum_samples = (
            max(2 * dimension, dimension + 2)
            if self.minimum_rotation_samples is None
            else max(self.minimum_rotation_samples, dimension + 2)
        )
        source_mean, source_centered = self._weighted_center(
            source,
            weights,
        )
        target_mean, target_centered = self._weighted_center(
            target,
            weights,
        )
        effective_rank = self._effective_rank(
            source_centered,
            weights,
        )
        required_rank = max(
            2,
            int(math.ceil(
                self.minimum_rotation_rank_fraction * dimension
            )),
        )
        if (
            source.size(0) < minimum_samples
            or effective_rank < required_rank
        ):
            return None, effective_rank

        weighted_source = source_centered * weights.sqrt().unsqueeze(1)
        weighted_target = target_centered * weights.sqrt().unsqueeze(1)
        cross = weighted_source.transpose(0, 1) @ weighted_target

        u, singular, vh = torch.linalg.svd(
            cross,
            full_matrices=False,
        )
        q = u @ vh
        # Proper orthogonal Procrustes: reject a reflection.
        if float(torch.linalg.det(q).item()) < 0.0:
            u = u.clone()
            u[:, -1] *= -1.0
            q = u @ vh

        denominator = weighted_source.square().sum().clamp_min(1e-12)
        aligned = source_centered @ q
        numerator = (
            weights[:, None] * aligned * target_centered
        ).sum()
        scale = float(
            (numerator / denominator)
            .clamp(self.minimum_scale, self.maximum_scale)
            .item()
        )

        # BranchSimilarityTransform uses y = x @ (scale * R).T + b.
        rotation = q.transpose(0, 1).contiguous()
        bias = (
            target_mean
            - source_mean
            @ (scale * rotation).transpose(0, 1)
        )
        return {
            "rotation": rotation,
            "scale": scale,
            "bias": bias,
            "level": 2,
            "level0_mode": "not_applicable",
            "singular_value_sum": float(singular.sum().item()),
        }, effective_rank

    def _fit_branch(
        self,
        support_source: Tensor,
        support_target: Tensor,
        validation_source: Tensor,
        validation_target: Tensor,
        *,
        support_weights: Tensor,
        validation_weights: Tensor,
        branch_name: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dimension = support_source.size(1)
        identity = self._identity_candidate(
            dimension,
            device=support_source.device,
            dtype=support_source.dtype,
        )
        translation = self._translation_candidate(
            support_source,
            support_target,
            support_weights,
        )
        scale = self._scale_candidate(
            support_source,
            support_target,
            support_weights,
        )
        rotation, effective_rank = self._rotation_candidate(
            support_source,
            support_target,
            support_weights,
        )

        candidates: Dict[str, Optional[Dict[str, Any]]] = {
            "identity": identity,
            "translation": translation,
            "scale_translation": scale,
            "scale_rotation_translation": rotation,
        }
        rmse: Dict[str, float] = {}
        for name, candidate in candidates.items():
            if candidate is None:
                rmse[name] = float("inf")
                continue
            predicted = self._apply(
                validation_source,
                rotation=candidate["rotation"],
                scale=candidate["scale"],
                bias=candidate["bias"],
            )
            rmse[name] = self._normalized_rmse(
                predicted,
                validation_target,
                validation_weights,
            )

        selected = identity
        selected_name = "identity"
        selected_rmse = rmse[selected_name]

        if _relative_improvement(
            selected_rmse,
            rmse["translation"],
        ) >= self.minimum_relative_improvement:
            selected = translation
            selected_name = "translation"
            selected_rmse = rmse[selected_name]

        if (
            scale is not None
            and _relative_improvement(
                selected_rmse,
                rmse["scale_translation"],
            ) >= self.minimum_relative_improvement
        ):
            selected = scale
            selected_name = "scale_translation"
            selected_rmse = rmse[selected_name]

        if (
            rotation is not None
            and _relative_improvement(
                selected_rmse,
                rmse["scale_rotation_translation"],
            ) >= self.minimum_relative_improvement
        ):
            selected = rotation
            selected_name = "scale_rotation_translation"
            selected_rmse = rmse[selected_name]

        rejected_for_error = False
        if (
            self.maximum_normalized_rmse > 0.0
            and selected_rmse > self.maximum_normalized_rmse
        ):
            selected = identity
            selected_name = "identity"
            selected_rmse = rmse["identity"]
            rejected_for_error = True

        report = {
            "branch": branch_name,
            "selected_candidate": selected_name,
            "selected_level": int(selected["level"]),
            "level0_mode": selected["level0_mode"],
            "selected_normalized_rmse": float(selected_rmse),
            "candidate_normalized_rmse": dict(rmse),
            "support_samples": int(support_source.size(0)),
            "validation_samples": int(validation_source.size(0)),
            "support_effective_rank": int(effective_rank),
            "rotation_candidate_available": rotation is not None,
            "scale_candidate_available": scale is not None,
            "rejected_for_excessive_error": rejected_for_error,
        }
        return selected, report

    def fit(
        self,
        support_previous: Tensor,
        support_current: Tensor,
        validation_previous: Tensor,
        validation_current: Tensor,
        *,
        support_weights: Optional[Tensor] = None,
        validation_weights: Optional[Tensor] = None,
    ) -> Tuple[BranchSimilarityTransform, Dict[str, Any]]:
        feature_dim = self.spectral_dim + self.spatial_dim
        support_previous = _feature_matrix(
            support_previous,
            expected_dim=feature_dim,
            name="support_previous",
        )
        support_current = _feature_matrix(
            support_current,
            expected_dim=feature_dim,
            name="support_current",
        )
        validation_previous = _feature_matrix(
            validation_previous,
            expected_dim=feature_dim,
            name="validation_previous",
        )
        validation_current = _feature_matrix(
            validation_current,
            expected_dim=feature_dim,
            name="validation_current",
        )
        if support_previous.shape != support_current.shape:
            raise ValueError("support feature pairs are not aligned")
        if validation_previous.shape != validation_current.shape:
            raise ValueError("validation feature pairs are not aligned")
        if support_previous.device != support_current.device:
            raise RuntimeError("support feature pairs must share one device")
        if validation_previous.device != validation_current.device:
            raise RuntimeError("validation feature pairs must share one device")
        if validation_previous.device != support_previous.device:
            raise RuntimeError("support and validation pairs must share one device")

        support_weight = _weights(
            support_weights,
            sample_count=support_previous.size(0),
            device=support_previous.device,
            dtype=support_previous.dtype,
        )
        validation_weight = _weights(
            validation_weights,
            sample_count=validation_previous.size(0),
            device=validation_previous.device,
            dtype=validation_previous.dtype,
        )

        ds = self.spectral_dim
        spectral, spectral_report = self._fit_branch(
            support_previous[:, :ds],
            support_current[:, :ds],
            validation_previous[:, :ds],
            validation_current[:, :ds],
            support_weights=support_weight,
            validation_weights=validation_weight,
            branch_name="spectral",
        )
        spatial, spatial_report = self._fit_branch(
            support_previous[:, ds:],
            support_current[:, ds:],
            validation_previous[:, ds:],
            validation_current[:, ds:],
            support_weights=support_weight,
            validation_weights=validation_weight,
            branch_name="spatial",
        )

        transform = BranchSimilarityTransform(
            spectral_rotation=spectral["rotation"].detach(),
            spatial_rotation=spatial["rotation"].detach(),
            spectral_scale=float(spectral["scale"]),
            spatial_scale=float(spatial["scale"]),
            spectral_bias=spectral["bias"].detach(),
            spatial_bias=spatial["bias"].detach(),
            spectral_level=int(spectral["level"]),
            spatial_level=int(spatial["level"]),
        )
        return transform, {
            "spectral": spectral_report,
            "spatial": spatial_report,
            "support_validation_disjoint_required": True,
            "analytical_fit": True,
            "trainable_transport_parameters": False,
        }


# =============================================================================
# Integrated model
# =============================================================================


class NECILModel(nn.Module):
    """Transport-verified factor-geometry NECIL architecture."""

    ARCHITECTURE_VERSION = 2
    METHOD_NAME = "Transport-verified spectral-spatial factor geometry"

    FEATURE_MODE = "features"
    BASE_CE_MODE = "base_ce"
    GEOMETRY_MODE = "geometry"

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device(
            getattr(args, "device", "cpu")
        )

        self.backbone = SSMBackbone(args)
        self.spectral_dim = int(
            self.backbone.spectral_feature_dim
        )
        self.spatial_dim = int(
            self.backbone.spatial_feature_dim
        )
        self.feature_dim = int(self.backbone.feature_dim)
        self.maximum_rank = int(
            getattr(
                args,
                "subspace_rank",
                getattr(args, "rank", 4),
            )
        )

        self.geometry_bank = GeometryBank(
            spectral_dim=self.spectral_dim,
            spatial_dim=self.spatial_dim,
            maximum_rank=self.maximum_rank,
            raw_spectral_dim=int(
                self.backbone.raw_num_bands
            ),
            spectral_resample_length=int(
                getattr(args, "spectral_resample_length", 32)
            ),
            spectral_derivative_weight=float(
                getattr(
                    args,
                    "spectral_derivative_weight",
                    0.50,
                )
            ),
            device=self.device,
            variance_floor_absolute=float(
                getattr(args, "geom_var_floor", 1e-4)
            ),
            variance_floor_relative=float(
                getattr(
                    args,
                    "geometry_variance_floor_relative",
                    1e-3,
                )
            ),
            minimum_variance_shrinkage=float(
                getattr(
                    args,
                    "geometry_minimum_variance_shrinkage",
                    0.10,
                )
            ),
            extra_rare_class_shrinkage=float(
                getattr(
                    args,
                    "geometry_extra_rare_class_shrinkage",
                    0.35,
                )
            ),
            shrinkage_sample_strength=float(
                getattr(
                    args,
                    "geometry_shrinkage_tau",
                    20.0,
                )
            ),
            factor_fit_iterations=int(
                getattr(args, "factor_fit_iterations", 3)
            ),
            whitened_eigen_excess_threshold=float(
                getattr(
                    args,
                    "whitened_eigen_excess_threshold",
                    0.50,
                )
            ),
            rank_energy_threshold=float(
                getattr(
                    args,
                    "rank_energy_threshold",
                    0.95,
                )
            ),
            rank_eigen_ratio_threshold=float(
                getattr(
                    args,
                    "rank_eigen_ratio_threshold",
                    1e-3,
                )
            ),
            volume_weight=float(
                getattr(
                    args,
                    "geometry_volume_weight",
                    getattr(args, "energy_logdet_weight", 0.50),
                )
            ),
            robust_iterations=int(
                getattr(
                    args,
                    "robust_geometry_iterations",
                    3,
                )
            ),
            robust_huber_delta=float(
                getattr(
                    args,
                    "robust_geometry_huber_delta",
                    2.5,
                )
            ),
            spatial_purity_power=float(
                getattr(args, "spatial_purity_power", 1.0)
            ),
            reliability_sample_strength=float(
                getattr(
                    args,
                    "geometry_reliability_sample_strength",
                    20.0,
                )
            ),
        )

        self.classifier = GeometryEnergyClassifier(
            feature_dim=self.feature_dim,
            temperature=float(
                getattr(args, "energy_temperature", 1.0)
            ),
            expected_spectral_dim=self.spectral_dim,
            expected_spatial_dim=self.spatial_dim,
            expected_bank_schema_version=int(
                self.geometry_bank.SCHEMA_VERSION
            ),
            require_bound_contract=False,
        )

        self.transport_estimator = (
            EvidenceGatedSimilarityEstimator(
                spectral_dim=self.spectral_dim,
                spatial_dim=self.spatial_dim,
                minimum_scale_samples=int(
                    getattr(
                        args,
                        "transport_minimum_scale_samples",
                        8,
                    )
                ),
                minimum_rotation_samples=getattr(
                    args,
                    "transport_minimum_rotation_samples",
                    None,
                ),
                minimum_rotation_rank_fraction=float(
                    getattr(
                        args,
                        "transport_minimum_rank_fraction",
                        0.50,
                    )
                ),
                rank_tolerance=float(
                    getattr(
                        args,
                        "transport_rank_tolerance",
                        1e-4,
                    )
                ),
                minimum_relative_improvement=float(
                    getattr(
                        args,
                        "transport_minimum_relative_improvement",
                        0.02,
                    )
                ),
                maximum_normalized_rmse=float(
                    getattr(
                        args,
                        "transport_maximum_normalized_rmse",
                        1.50,
                    )
                ),
                minimum_scale=float(
                    getattr(args, "transport_minimum_scale", 0.70)
                ),
                maximum_scale=float(
                    getattr(args, "transport_maximum_scale", 1.30)
                ),
            )
        )

        self.base_ce_head: Optional[nn.Linear] = None
        self.base_ce_class_ids: List[int] = []

        self.current_phase = 0
        self.phase_mode = "base"
        self.seen_classes: List[int] = []
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []
        self.phase_old_digest: Optional[str] = None
        self.base_handoff: Optional[Dict[str, Any]] = None

        self.to(self.device)
        self.assert_architecture_contract()

    # ------------------------------------------------------------------
    # Architecture and checkpoint contract
    # ------------------------------------------------------------------

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "architecture_version":
                self.ARCHITECTURE_VERSION,
            "method": self.METHOD_NAME,
            "classification_factorization": "p(z|c)",
            "spectral_relation_factorization": "p(h|c)",
            "joint_feature": "direct_[z_s;z_p]",
            "feature_dim": self.feature_dim,
            "spectral_dim": self.spectral_dim,
            "spatial_dim": self.spatial_dim,
            "maximum_factor_rank": self.maximum_rank,
            "phase_observer": (
                "temporary_external_frozen_encoder"
            ),
            "phase_transport": (
                "analytical_evidence_gated_branch_similarity"
            ),
            "trainable_transport_network": False,
            "old_rows_evolve": True,
            "uses_geometry_replay_for_training": False,
            "stores_exemplars": False,
            "stores_old_features": False,
            "stores_old_spectra": False,
            "uses_knowledge_distillation": False,
            "uses_task_adapters": False,
            "classifier": (
                "parameter_free_equal_prior_factor_energy"
            ),
        }

    def architecture_digest(self) -> str:
        payload = (
            self.architecture_summary(),
            self.backbone.backbone_contract(),
            self.classifier.classifier_contract(),
        )
        return hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()

    def assert_architecture_contract(self) -> None:
        if self.feature_dim != (
            self.spectral_dim + self.spatial_dim
        ):
            raise RuntimeError(
                "backbone branch dimensions do not sum to feature_dim"
            )
        if self.geometry_bank.feature_dim != self.feature_dim:
            raise RuntimeError(
                "GeometryBank and backbone feature dimensions differ"
            )
        if self.geometry_bank.spectral_dim != self.spectral_dim:
            raise RuntimeError(
                "GeometryBank spectral dimension is inconsistent"
            )
        if self.geometry_bank.spatial_dim != self.spatial_dim:
            raise RuntimeError(
                "GeometryBank spatial dimension is inconsistent"
            )
        if self.geometry_bank.raw_spectral_dim != (
            self.backbone.raw_num_bands
        ):
            raise RuntimeError(
                "GeometryBank and backbone raw spectral widths differ"
            )

        required_bank_methods = (
            "fit_global_priors",
            "extract_rows",
            "energy_matrix",
            "pair_risk_matrix",
            "build_transported_rows",
            "commit_base_rows",
            "commit_incremental_phase",
            "rows_digest",
            "export_snapshot",
            "load_snapshot",
        )
        missing = [
            name
            for name in required_bank_methods
            if not callable(
                getattr(self.geometry_bank, name, None)
            )
        ]
        if missing:
            raise RuntimeError(
                f"GeometryBank lacks required methods: {missing}"
            )

        if any(
            parameter.requires_grad
            for parameter in self.geometry_bank.parameters()
        ):
            raise RuntimeError(
                "GeometryBank must contain buffers only"
            )
        if any(
            parameter.requires_grad
            for parameter in self.classifier.parameters()
        ):
            raise RuntimeError(
                "factor-energy classifier must be parameter-free"
            )

        classifier_contract = (
            self.classifier.classifier_contract()
        )
        if classifier_contract.get(
            "classification_factorization"
        ) != "p(z|c)":
            raise RuntimeError(
                "classifier factorization is inconsistent"
            )
        if classifier_contract.get(
            "uses_raw_spectra_at_inference"
        ) is not False:
            raise RuntimeError(
                "raw spectra must not enter classifier inference"
            )

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "architecture_version":
                self.ARCHITECTURE_VERSION,
            "architecture_digest":
                self.architecture_digest(),
            "current_phase": int(self.current_phase),
            "phase_mode": str(self.phase_mode),
            "seen_classes": list(self.seen_classes),
            "old_classes": list(self.old_classes),
            "new_classes": list(self.new_classes),
            "phase_old_digest": self.phase_old_digest,
            "base_ce_class_ids":
                list(self.base_ce_class_ids),
            "base_handoff":
                copy.deepcopy(self.base_handoff),
            "phase_context_stored_in_model": False,
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            return
        version = int(
            state.get("architecture_version", -1)
        )
        if version != self.ARCHITECTURE_VERSION:
            raise RuntimeError(
                "NECILModel architecture version mismatch: "
                f"{version} vs {self.ARCHITECTURE_VERSION}"
            )

        self.current_phase = int(
            state.get("current_phase", 0)
        )
        self.phase_mode = str(
            state.get("phase_mode", "base")
        )
        self.seen_classes = [
            int(value)
            for value in state.get("seen_classes", [])
        ]
        self.old_classes = [
            int(value)
            for value in state.get("old_classes", [])
        ]
        self.new_classes = [
            int(value)
            for value in state.get("new_classes", [])
        ]
        digest = state.get("phase_old_digest")
        self.phase_old_digest = (
            None if digest is None else str(digest)
        )
        self.base_ce_class_ids = [
            int(value)
            for value in state.get(
                "base_ce_class_ids",
                [],
            )
        ]
        if (
            self.base_ce_class_ids
            and self.base_ce_head is None
        ):
            self.ensure_base_ce_head(
                self.base_ce_class_ids
            )
        self.base_handoff = copy.deepcopy(
            state.get("base_handoff")
        )

    # ------------------------------------------------------------------
    # Backbone output and feature extraction
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        output: Mapping[str, Any],
    ) -> Dict[str, Any]:
        required = (
            "spectral_feature",
            "spatial_feature",
            "joint_feature",
            "raw_center_spectrum",
        )
        missing = [
            name for name in required
            if name not in output
        ]
        if missing:
            raise RuntimeError(
                f"backbone output lacks {missing}"
            )

        result = dict(output)
        spectral = _feature_matrix(
            result["spectral_feature"],
            expected_dim=self.spectral_dim,
            name="spectral_feature",
        )
        spatial = _feature_matrix(
            result["spatial_feature"],
            expected_dim=self.spatial_dim,
            name="spatial_feature",
        )
        joint = _feature_matrix(
            result["joint_feature"],
            expected_dim=self.feature_dim,
            name="joint_feature",
        )
        if not torch.equal(
            joint,
            torch.cat([spectral, spatial], dim=1),
        ):
            raise RuntimeError(
                "joint feature is not exact [z_s;z_p]"
            )

        spectra = torch.as_tensor(
            result["raw_center_spectrum"],
            device=joint.device,
            dtype=joint.dtype,
        )
        if spectra.shape != (
            joint.size(0),
            self.backbone.raw_num_bands,
        ):
            raise RuntimeError(
                "raw_center_spectrum shape is invalid"
            )
        if not torch.isfinite(spectra).all():
            raise RuntimeError(
                "raw_center_spectrum contains NaN/Inf"
            )

        result["spectral_feature"] = spectral
        result["spatial_feature"] = spatial
        result["joint_feature"] = joint
        result["joint_features"] = joint
        result["geometry_features"] = joint
        result["features"] = joint
        result["raw_center_spectrum"] = spectra
        result["raw_center_spectra"] = spectra
        return result

    def forward_features(
        self,
        model_patch: Tensor,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
        *,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        patch = model_patch.to(self.device)
        raw_patch = (
            None
            if raw_spectral_patch is None
            else raw_spectral_patch.to(self.device)
        )
        raw_center = (
            None
            if raw_center_spectrum is None
            else raw_center_spectrum.to(self.device)
        )
        output = (
            self.backbone.forward_geometry(
                patch,
                raw_spectral_patch=raw_patch,
                raw_center_spectrum=raw_center,
            )
            if deterministic
            else self.backbone(
                patch,
                raw_spectral_patch=raw_patch,
                raw_center_spectrum=raw_center,
            )
        )
        return self._validate_output(output)

    extract_geometry_features = forward_features

    @torch.no_grad()
    def encode_observer(
        self,
        context: IncrementalPhaseContext,
        model_patch: Tensor,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        self._validate_phase_context(context)
        output = context.observer.forward_geometry(
            model_patch.to(self.device),
            raw_spectral_patch=(
                None
                if raw_spectral_patch is None
                else raw_spectral_patch.to(self.device)
            ),
            raw_center_spectrum=(
                None
                if raw_center_spectrum is None
                else raw_center_spectrum.to(self.device)
            ),
        )
        return self._validate_output(output)

    # ------------------------------------------------------------------
    # Classifier interface
    # ------------------------------------------------------------------

    def infer_seen_classes(self) -> List[int]:
        valid = self.geometry_bank.valid_mask()
        return torch.nonzero(
            valid.detach().cpu().bool(),
            as_tuple=False,
        ).flatten().tolist()

    def compute_logits_from_features(
        self,
        features: Tensor,
        *,
        class_ids: Sequence[int],
        temporary_rows: Optional[
            Mapping[int, Mapping[str, Any]]
        ] = None,
        targets: Optional[Tensor] = None,
        targets_are_global: bool = False,
        old_classes: Optional[Sequence[int]] = None,
        new_classes: Optional[Sequence[int]] = None,
        mode: str = "factor_geometry",
        return_energy: bool = False,
        return_parts: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        return self.classifier(
            features,
            geometry_bank=self.geometry_bank,
            class_ids=class_ids,
            temporary_rows=temporary_rows,
            targets=targets,
            targets_are_global=targets_are_global,
            old_classes=old_classes,
            new_classes=new_classes,
            mode=mode,
            return_energy=return_energy,
            return_parts=return_parts,
            return_diagnostics=return_diagnostics,
        )

    def forward(
        self,
        model_patch: Tensor,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
        *,
        mode: str = GEOMETRY_MODE,
        class_ids: Optional[Sequence[int]] = None,
        temporary_rows: Optional[
            Mapping[int, Mapping[str, Any]]
        ] = None,
        targets: Optional[Tensor] = None,
        targets_are_global: bool = False,
        old_classes: Optional[Sequence[int]] = None,
        new_classes: Optional[Sequence[int]] = None,
        classifier_mode: str = "factor_geometry",
        deterministic: bool = True,
        return_energy: bool = False,
        return_parts: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        output = self.forward_features(
            model_patch,
            raw_spectral_patch,
            raw_center_spectrum,
            deterministic=deterministic,
        )
        selected = str(mode).strip().lower()

        if selected == self.FEATURE_MODE:
            return output

        if selected == self.BASE_CE_MODE:
            logits = self.base_ce_logits(
                output["joint_feature"]
            )
            return {
                **output,
                "logits": logits,
                "class_ids": torch.tensor(
                    self.base_ce_class_ids,
                    device=logits.device,
                    dtype=torch.long,
                ),
            }

        if selected != self.GEOMETRY_MODE:
            raise ValueError(
                "mode must be one of "
                "features/base_ce/geometry"
            )

        ids = (
            self.infer_seen_classes()
            if class_ids is None
            else _unique_ids(
                class_ids,
                name="class_ids",
            )
        )
        if not ids:
            raise RuntimeError(
                "no class rows are available for geometry inference"
            )

        result = self.compute_logits_from_features(
            output["joint_feature"],
            class_ids=ids,
            temporary_rows=temporary_rows,
            targets=targets,
            targets_are_global=targets_are_global,
            old_classes=old_classes,
            new_classes=new_classes,
            mode=classifier_mode,
            return_energy=return_energy,
            return_parts=return_parts,
            return_diagnostics=return_diagnostics,
        )
        if isinstance(result, Mapping):
            return {**output, **dict(result)}
        return result

    # ------------------------------------------------------------------
    # Base warm-up and bank handoff
    # ------------------------------------------------------------------

    def ensure_base_ce_head(
        self,
        base_class_ids: Sequence[int],
    ) -> nn.Linear:
        ids = _unique_ids(
            base_class_ids,
            name="base_class_ids",
        )
        if len(ids) < 2:
            raise ValueError(
                "base phase requires at least two classes"
            )

        if (
            self.base_ce_head is None
            or self.base_ce_head.out_features != len(ids)
        ):
            self.base_ce_head = nn.Linear(
                self.feature_dim,
                len(ids),
            ).to(self.device)
            nn.init.normal_(
                self.base_ce_head.weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.base_ce_head.bias)
        self.base_ce_class_ids = list(ids)
        return self.base_ce_head

    def base_ce_logits(self, features: Tensor) -> Tensor:
        if self.base_ce_head is None:
            raise RuntimeError(
                "base CE head is not initialized"
            )
        value = _feature_matrix(
            features,
            expected_dim=self.feature_dim,
            name="base CE features",
        )
        return self.base_ce_head(value)

    def drop_base_ce_head(self) -> None:
        self.base_ce_head = None
        self.base_ce_class_ids = []

    @torch.no_grad()
    def fit_global_priors(
        self,
        features: Tensor,
        raw_center_spectra: Tensor,
        *,
        freeze: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, float]:
        if (
            self.current_phase != 0
            or self.phase_mode != "base"
        ):
            raise RuntimeError(
                "global priors can only be fitted in base phase"
            )
        return self.geometry_bank.fit_global_priors(
            features,
            raw_center_spectra,
            freeze=freeze,
            overwrite=overwrite,
        )

    @torch.no_grad()
    def build_geometry_rows(
        self,
        *,
        class_ids: Sequence[int],
        features: Tensor,
        raw_center_spectra: Tensor,
        labels: Tensor,
        sample_weights: Optional[Tensor] = None,
    ) -> Dict[int, Row]:
        ids = _unique_ids(
            class_ids,
            name="class_ids",
        )
        rows = self.geometry_bank.extract_rows(
            features,
            labels,
            raw_spectra=raw_center_spectra,
            sample_weights=sample_weights,
            class_ids=ids,
        )
        if set(rows) != set(ids):
            raise RuntimeError(
                "GeometryBank returned an incomplete row set"
            )
        return rows

    @torch.no_grad()
    def build_geometry_rows_from_output(
        self,
        output: Mapping[str, Any],
        labels: Tensor,
        class_ids: Sequence[int],
        *,
        sample_weights: Optional[Tensor] = None,
    ) -> Dict[int, Row]:
        checked = self._validate_output(output)
        return self.build_geometry_rows(
            class_ids=class_ids,
            features=checked["joint_feature"],
            raw_center_spectra=checked[
                "raw_center_spectrum"
            ],
            labels=labels,
            sample_weights=sample_weights,
        )

    @torch.no_grad()
    def finalize_base_geometry(
        self,
        *,
        base_class_ids: Sequence[int],
        features: Tensor,
        raw_center_spectra: Tensor,
        labels: Tensor,
        sample_weights: Optional[Tensor] = None,
        update_statistics: bool = True,
        fit_overlap_temperatures: bool = True,
        drop_ce_head: bool = True,
        calibrated_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        if (
            self.current_phase != 0
            or self.phase_mode != "base"
        ):
            raise RuntimeError(
                "finalize_base_geometry is base-phase only"
            )
        ids = _unique_ids(
            base_class_ids,
            name="base_class_ids",
        )
        self.geometry_bank.assert_global_priors_ready()
        if not bool(
            self.geometry_bank.global_priors_frozen.item()
        ):
            raise RuntimeError(
                "base-fitted global priors must be frozen"
            )
        if bool(
            self.geometry_bank.valid_mask().any().item()
        ):
            raise RuntimeError(
                "base geometry can only be committed once"
            )

        bank_snapshot = (
            self.geometry_bank.export_snapshot()
        )
        classifier_state = copy.deepcopy(
            self.classifier.state_dict()
        )
        previous_state = (
            list(self.seen_classes),
            list(self.old_classes),
            list(self.new_classes),
            str(self.phase_mode),
            copy.deepcopy(self.base_handoff),
        )

        try:
            rows = self.build_geometry_rows(
                class_ids=ids,
                features=features,
                raw_center_spectra=
                    raw_center_spectra,
                labels=labels,
                sample_weights=sample_weights,
            )
            commit = (
                self.geometry_bank.commit_base_rows(
                    rows,
                    base_class_ids=ids,
                    phase=0,
                )
            )

            overlap_report = None
            if fit_overlap_temperatures:
                overlap_report = (
                    self.geometry_bank
                    .fit_overlap_temperatures(
                        ids,
                        freeze=True,
                    )
                )

            statistics = None
            if update_statistics:
                statistics = (
                    self.geometry_bank.update_statistics(
                        features,
                        labels,
                        ids,
                    )
                )

            admission = (
                self.geometry_bank.admission_report(
                    ids,
                    maximum_reconstruction_error=float(
                        getattr(
                            self.args,
                            "maximum_geometry_reconstruction_error",
                            0.75,
                        )
                    ),
                    minimum_effective_dimension=float(
                        getattr(
                            self.args,
                            "minimum_geometry_effective_dimension",
                            1.25,
                        )
                    ),
                    require_statistics=update_statistics,
                )
            )
            if not admission["ok"]:
                raise RuntimeError(
                    "base geometry admission failed: "
                    + "; ".join(admission["errors"])
                )

            if calibrated_temperature is not None:
                self.classifier.set_temperature(
                    calibrated_temperature
                )

            contract_digest = (
                self.classifier
                .bind_geometry_bank_contract(
                    self.geometry_bank,
                    require_committed_rows=True,
                    enforce_after_binding=True,
                )
            )

            self.seen_classes = list(ids)
            self.old_classes = list(ids)
            self.new_classes = []
            self.phase_mode = "evaluation"
            self.backbone.freeze_all()
            if drop_ce_head:
                self.drop_base_ce_head()

            self.base_handoff = {
                "commit": commit,
                "overlap_temperatures":
                    overlap_report,
                "statistics": statistics,
                "admission": admission,
                "bank_contract_digest":
                    contract_digest,
                "classification_factorization":
                    "p(z|c)",
                "spectral_relation_factorization":
                    "p(h|c)",
                "atomic": True,
            }
            return copy.deepcopy(
                self.base_handoff
            )
        except Exception:
            self.geometry_bank.load_snapshot(
                bank_snapshot,
                strict=True,
            )
            self.classifier.load_state_dict(
                classifier_state
            )
            (
                self.seen_classes,
                self.old_classes,
                self.new_classes,
                self.phase_mode,
                self.base_handoff,
            ) = previous_state
            raise

    # ------------------------------------------------------------------
    # Incremental phase lifecycle
    # ------------------------------------------------------------------

    def set_base_mode(self) -> None:
        if bool(
            self.geometry_bank.valid_mask().any().item()
        ):
            raise RuntimeError(
                "committed rows exist; reset the bank before "
                "restarting base training"
            )
        self.current_phase = 0
        self.phase_mode = "base"
        self.seen_classes = []
        self.old_classes = []
        self.new_classes = []
        self.phase_old_digest = None
        self.base_handoff = None
        self.backbone.enable_base_training()
        self.classifier.require_bound_contract = False

    def _static_bank_digest(self) -> str:
        return self.classifier.bank_contract_digest(
            self.geometry_bank
        )

    def begin_incremental_phase(
        self,
        *,
        phase: int,
        old_class_ids: Sequence[int],
        new_class_ids: Sequence[int],
        controlled_plasticity: bool = True,
    ) -> IncrementalPhaseContext:
        phase = int(phase)
        if phase <= 0:
            raise ValueError(
                "incremental phase must be >= 1"
            )
        old_ids = _unique_ids(
            old_class_ids,
            name="old_class_ids",
        )
        new_ids = _unique_ids(
            new_class_ids,
            name="new_class_ids",
        )
        if set(old_ids) & set(new_ids):
            raise RuntimeError(
                "old and new class IDs overlap"
            )
        committed = self.infer_seen_classes()
        if set(committed) != set(old_ids):
            raise RuntimeError(
                f"old classes {sorted(old_ids)} do not match "
                f"committed rows {sorted(committed)}"
            )
        self.geometry_bank.assert_valid(
            old_ids,
            strict=True,
        )
        self.geometry_bank.assert_global_priors_ready()

        current_contract = self._static_bank_digest()
        if (
            self.classifier.bound_bank_contract_digest
            != current_contract
        ):
            raise RuntimeError(
                "finalize and bind base geometry before "
                "starting an incremental phase"
            )

        # Capture the accepted phase-start coordinates before changing
        # trainability or optimizer state.
        observer = self.backbone.make_phase_observer()

        self.current_phase = phase
        self.phase_mode = "incremental"
        self.old_classes = list(old_ids)
        self.new_classes = list(new_ids)
        self.seen_classes = [
            *old_ids,
            *new_ids,
        ]
        self.phase_old_digest = (
            self.geometry_bank.rows_digest(old_ids)
        )

        if controlled_plasticity:
            self.backbone.enable_incremental_plasticity()
            parameter_snapshot = {
                name: parameter.detach().clone()
                for name, parameter
                in self.backbone.named_parameters()
                if parameter.requires_grad
            }
        else:
            self.backbone.freeze_for_incremental()
            parameter_snapshot = {}

        self.backbone.assert_incremental_policy()
        return IncrementalPhaseContext(
            phase=phase,
            old_class_ids=list(old_ids),
            new_class_ids=list(new_ids),
            old_rows_digest=str(
                self.phase_old_digest
            ),
            bank_contract_digest=current_contract,
            observer=observer,
            parameter_snapshot=
                parameter_snapshot,
            controlled_plasticity=bool(
                controlled_plasticity
            ),
        )

    def restore_incremental_context(
        self,
        state: Mapping[str, Any],
    ) -> IncrementalPhaseContext:
        if self.phase_mode != "incremental":
            raise RuntimeError(
                "model must be in incremental mode before "
                "restoring phase context"
            )
        observer = copy.deepcopy(
            self.backbone
        ).to(self.device)
        observer.load_state_dict(
            state["observer_state_dict"],
            strict=True,
        )
        observer.freeze_all()
        observer.eval()

        snapshot = {
            str(name): torch.as_tensor(
                value,
                device=self.device,
            ).detach().clone()
            for name, value in dict(
                state.get(
                    "parameter_snapshot",
                    {},
                )
            ).items()
        }
        context = IncrementalPhaseContext(
            phase=int(state["phase"]),
            old_class_ids=[
                int(value)
                for value in state["old_class_ids"]
            ],
            new_class_ids=[
                int(value)
                for value in state["new_class_ids"]
            ],
            old_rows_digest=str(
                state["old_rows_digest"]
            ),
            bank_contract_digest=str(
                state["bank_contract_digest"]
            ),
            observer=observer,
            parameter_snapshot=snapshot,
            controlled_plasticity=bool(
                state["controlled_plasticity"]
            ),
        )
        self._validate_phase_context(context)
        return context

    def _validate_phase_context(
        self,
        context: IncrementalPhaseContext,
    ) -> None:
        if not isinstance(
            context,
            IncrementalPhaseContext,
        ):
            raise TypeError(
                "context must be IncrementalPhaseContext"
            )
        if self.phase_mode != "incremental":
            raise RuntimeError(
                "no incremental phase is active"
            )
        if context.phase != self.current_phase:
            raise RuntimeError(
                "context phase does not match active phase"
            )
        if context.old_class_ids != self.old_classes:
            raise RuntimeError(
                "context old classes do not match active phase"
            )
        if context.new_class_ids != self.new_classes:
            raise RuntimeError(
                "context new classes do not match active phase"
            )
        if context.old_rows_digest != self.phase_old_digest:
            raise RuntimeError(
                "context old-row digest mismatch"
            )
        if (
            context.bank_contract_digest
            != self._static_bank_digest()
        ):
            raise RuntimeError(
                "context GeometryBank contract mismatch"
            )

    def incremental_trainable_parameters(
        self,
    ) -> List[nn.Parameter]:
        if self.phase_mode != "incremental":
            raise RuntimeError(
                "incremental parameters require active phase"
            )
        return (
            self.backbone
            .incremental_trainable_parameters()
        )

    def incremental_trainable_parameter_names(
        self,
    ) -> List[str]:
        return (
            self.backbone
            .incremental_trainable_parameter_names()
        )

    @torch.no_grad()
    def fit_phase_transform(
        self,
        context: IncrementalPhaseContext,
        *,
        support_previous_features: Tensor,
        support_current_features: Tensor,
        validation_previous_features: Tensor,
        validation_current_features: Tensor,
        support_weights: Optional[Tensor] = None,
        validation_weights: Optional[Tensor] = None,
    ) -> Tuple[
        BranchSimilarityTransform,
        Dict[str, Any],
    ]:
        self._validate_phase_context(context)
        return self.transport_estimator.fit(
            support_previous_features,
            support_current_features,
            validation_previous_features,
            validation_current_features,
            support_weights=support_weights,
            validation_weights=validation_weights,
        )

    @torch.no_grad()
    def build_transported_old_rows(
        self,
        *,
        transform: BranchSimilarityTransform,
    ) -> Tuple[Dict[int, Row], Dict[str, Any]]:
        if self.phase_mode != "incremental":
            raise RuntimeError(
                "transported rows require active incremental phase"
            )
        return self.geometry_bank.build_transported_rows(
            self.old_classes,
            transform=transform,
        )

    @torch.no_grad()
    def build_phase_rows(
        self,
        *,
        transform: BranchSimilarityTransform,
        new_support_features: Tensor,
        new_support_raw_spectra: Tensor,
        new_support_labels: Tensor,
        new_support_weights: Optional[Tensor] = None,
    ) -> Tuple[Dict[int, Row], Dict[str, Any]]:
        transported, report = (
            self.build_transported_old_rows(
                transform=transform,
            )
        )
        new_rows = self.build_geometry_rows(
            class_ids=self.new_classes,
            features=new_support_features,
            raw_center_spectra=
                new_support_raw_spectra,
            labels=new_support_labels,
            sample_weights=new_support_weights,
        )
        overlap = set(transported) & set(new_rows)
        if overlap:
            raise RuntimeError(
                "old/new temporary row overlap: "
                f"{sorted(overlap)}"
            )
        return {
            **transported,
            **new_rows,
        }, report

    @torch.no_grad()
    def phase_pair_risk(
        self,
        temporary_rows: Mapping[
            int,
            Mapping[str, Any],
        ],
    ) -> Tensor:
        if self.phase_mode != "incremental":
            raise RuntimeError(
                "phase pair risk requires active incremental phase"
            )
        expected = set(self.seen_classes)
        if set(int(key) for key in temporary_rows) != expected:
            raise RuntimeError(
                "temporary rows do not cover all seen classes"
            )
        return self.geometry_bank.pair_risk_matrix(
            self.seen_classes,
            rows=temporary_rows,
        )

    @torch.no_grad()
    def commit_incremental_phase(
        self,
        *,
        transform: BranchSimilarityTransform,
        new_rows: Mapping[int, Mapping[str, Any]],
        phase: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.phase_mode != "incremental":
            raise RuntimeError("no active incremental phase")
        phase_value = self.current_phase if phase is None else int(phase)
        if phase_value != self.current_phase:
            raise RuntimeError("phase value does not match active phase")
        if set(int(key) for key in new_rows) != set(self.new_classes):
            raise RuntimeError("new row IDs do not match active new classes")

        contract_before = self._static_bank_digest()
        old_ids = list(self.old_classes)
        old_index = torch.tensor(
            old_ids, device=self.geometry_bank.device, dtype=torch.long
        )
        old_energy_quantiles = (
            self.geometry_bank.energy_quantiles.index_select(0, old_index).clone()
        )
        old_margin_quantiles = (
            self.geometry_bank.margin_quantiles.index_select(0, old_index).clone()
        )
        old_statistics_ready = (
            self.geometry_bank.statistics_ready.index_select(0, old_index).clone()
        )

        transported, transport_report = self.build_transported_old_rows(
            transform=transform
        )
        result = self.geometry_bank.commit_incremental_phase(
            transported_old_rows=transported,
            new_rows=new_rows,
            old_class_ids=old_ids,
            new_class_ids=self.new_classes,
            phase=phase_value,
            expected_old_digest=self.phase_old_digest,
        )

        # Exact similarity pushforward preserves every Mahalanobis quadratic
        # term and every inter-class energy margin.  Only the log-volume term
        # receives one class-independent additive shift.  Preserve old
        # diagnostics analytically instead of requiring forbidden old samples.
        logdet_shift = 2.0 * (
            self.spectral_dim * math.log(float(transform.spectral_scale))
            + self.spatial_dim * math.log(float(transform.spatial_scale))
        )
        energy_shift = (
            float(self.geometry_bank.volume_weight)
            * logdet_shift
            / float(self.feature_dim)
        )
        self.geometry_bank.energy_quantiles.index_copy_(
            0, old_index, old_energy_quantiles + energy_shift
        )
        self.geometry_bank.margin_quantiles.index_copy_(
            0, old_index, old_margin_quantiles
        )
        self.geometry_bank.statistics_ready.index_copy_(
            0, old_index, old_statistics_ready
        )

        contract_after = self._static_bank_digest()
        if contract_after != contract_before:
            raise RuntimeError("row commit changed the static bank contract")
        if self.classifier.bound_bank_contract_digest != contract_after:
            raise RuntimeError("classifier/bank contract diverged after commit")

        self.seen_classes = [*old_ids, *self.new_classes]
        self.old_classes = list(self.seen_classes)
        self.new_classes = []
        self.phase_old_digest = None
        self.phase_mode = "evaluation"
        self.backbone.freeze_all()

        result["transport_report"] = transport_report
        result["old_statistics_preserved_analytically"] = True
        result["old_energy_quantile_shift"] = float(energy_shift)
        result["classification_factorization"] = "p(z|c)"
        result["spectral_relation_factorization"] = "p(h|c)"
        result["transport_absorbed"] = True
        result["temporary_phase_context_must_be_deleted"] = True
        return result

    def abort_incremental_phase(self) -> None:
        """Discard phase metadata; trainer restores accepted checkpoint."""
        self.phase_old_digest = None
        self.new_classes = []
        self.seen_classes = (
            self.infer_seen_classes()
        )
        self.old_classes = list(
            self.seen_classes
        )
        self.phase_mode = "evaluation"
        self.backbone.freeze_all()

    # ------------------------------------------------------------------
    # Diagnostics and memory
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_geometry_statistics(
        self,
        *,
        features: Tensor,
        labels: Tensor,
        class_ids: Sequence[int],
        rows: Optional[
            Mapping[int, Mapping[str, Any]]
        ] = None,
    ) -> Dict[str, float]:
        return self.geometry_bank.update_statistics(
            features,
            labels,
            class_ids,
            rows=rows,
        )

    @torch.no_grad()
    def geometry_admission_report(
        self,
        class_ids: Optional[Sequence[int]] = None,
        *,
        maximum_reconstruction_error: float = 0.75,
        minimum_effective_dimension: float = 1.25,
        require_statistics: bool = True,
    ) -> Dict[str, Any]:
        ids = (
            self.infer_seen_classes()
            if class_ids is None
            else _unique_ids(
                class_ids,
                name="class_ids",
            )
        )
        return self.geometry_bank.admission_report(
            ids,
            maximum_reconstruction_error=
                maximum_reconstruction_error,
            minimum_effective_dimension=
                minimum_effective_dimension,
            require_statistics=require_statistics,
        )

    @torch.no_grad()
    def memory_snapshot(self) -> Dict[str, Any]:
        ids = self.infer_seen_classes()
        return {
            "bank":
                self.geometry_bank.export_snapshot(),
            "bank_contract_digest":
                self._static_bank_digest(),
            "rows_digest": (
                self.geometry_bank.rows_digest(ids)
                if ids else None
            ),
            "phase": int(self.current_phase),
            "phase_mode": str(self.phase_mode),
            "seen_classes":
                list(self.seen_classes),
            "classification_factorization":
                "p(z|c)",
            "spectral_relation_factorization":
                "p(h|c)",
            "stores_sample_level_memory": False,
            "stores_phase_observer": False,
        }


TransportVerifiedNECILModel = NECILModel













# from __future__ import annotations

# import copy
# import hashlib
# import math
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.backbone import SSMBackbone
# from models.classifier import GeometryEnergyClassifier
# from models.geometry_bank import GeometryBank


# Tensor = torch.Tensor
# Row = Dict[str, Tensor]


# def _unique_ids(
#     values: Iterable[int],
#     *,
#     name: str,
#     allow_empty: bool = False,
# ) -> List[int]:
#     output: List[int] = []
#     observed: set[int] = set()
#     for value in values:
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError(f"{name} contains negative class ID {class_id}")
#         if class_id in observed:
#             raise ValueError(f"{name} contains duplicate class ID {class_id}")
#         observed.add(class_id)
#         output.append(class_id)
#     if not output and not allow_empty:
#         raise ValueError(f"{name} is empty")
#     return output


# def _bounded_frobenius(value: Tensor, maximum_norm: float) -> Tensor:
#     """Smoothly bound a tensor's Frobenius norm below ``maximum_norm``."""
#     maximum = float(maximum_norm)
#     if maximum <= 0.0:
#         return torch.zeros_like(value)
#     norm = value.norm()
#     return maximum * value / (maximum + norm)


# class SpectralConditionedPhaseTransport(nn.Module):
#     """One phase-wise invertible map ``T(z,s)=Mz+Hs+b``.

#     ``M`` is a low-rank residual around the identity. Its residual Frobenius
#     norm is bounded below one, which guarantees invertibility by
#     ``sigma_min(I+R) >= 1 - ||R||_2``. ``H`` and ``b`` are also norm bounded.

#     The module is temporary: its effect is analytically absorbed into the
#     GeometryBank at phase commit and the module is then discarded.
#     """

#     def __init__(
#         self,
#         *,
#         d_model: int,
#         spectral_anchor_dim: int,
#         feature_rank: int = 8,
#         spectral_rank: int = 8,
#         maximum_matrix_residual_norm: float = 0.20,
#         maximum_spectral_shift_norm: float = 0.50,
#         maximum_bias_norm: float = 0.25,
#         initialization_std: float = 0.02,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.spectral_anchor_dim = int(spectral_anchor_dim)
#         self.feature_rank = int(feature_rank)
#         self.spectral_rank = int(spectral_rank)
#         self.maximum_matrix_residual_norm = float(
#             maximum_matrix_residual_norm
#         )
#         self.maximum_spectral_shift_norm = float(
#             maximum_spectral_shift_norm
#         )
#         self.maximum_bias_norm = float(maximum_bias_norm)

#         if self.d_model <= 0 or self.spectral_anchor_dim <= 0:
#             raise ValueError("transport dimensions must be positive")
#         if not 1 <= self.feature_rank <= self.d_model:
#             raise ValueError("feature_rank must be in [1,d_model]")
#         if not 1 <= self.spectral_rank <= min(
#             self.d_model, self.spectral_anchor_dim
#         ):
#             raise ValueError(
#                 "spectral_rank must be in [1,min(d_model,anchor_dim)]"
#             )
#         if not 0.0 < self.maximum_matrix_residual_norm < 1.0:
#             raise ValueError(
#                 "maximum_matrix_residual_norm must lie strictly in (0,1)"
#             )
#         if self.maximum_spectral_shift_norm < 0.0:
#             raise ValueError("maximum_spectral_shift_norm must be non-negative")
#         if self.maximum_bias_norm < 0.0:
#             raise ValueError("maximum_bias_norm must be non-negative")

#         std = float(max(initialization_std, 1e-6))
#         self.feature_left = nn.Parameter(
#             torch.randn(self.d_model, self.feature_rank) * std
#         )
#         self.feature_right = nn.Parameter(
#             torch.zeros(self.d_model, self.feature_rank)
#         )
#         self.spectral_left = nn.Parameter(
#             torch.randn(self.d_model, self.spectral_rank) * std
#         )
#         self.spectral_right = nn.Parameter(
#             torch.zeros(self.spectral_anchor_dim, self.spectral_rank)
#         )
#         self.raw_bias = nn.Parameter(torch.zeros(self.d_model))

#     def matrix_residual(self) -> Tensor:
#         raw = (
#             self.feature_left @ self.feature_right.transpose(0, 1)
#         ) / math.sqrt(float(self.feature_rank))
#         return _bounded_frobenius(
#             raw, self.maximum_matrix_residual_norm
#         )

#     def matrix(self) -> Tensor:
#         identity = torch.eye(
#             self.d_model,
#             device=self.feature_left.device,
#             dtype=self.feature_left.dtype,
#         )
#         return identity + self.matrix_residual()

#     def spectral_shift(self) -> Tensor:
#         raw = (
#             self.spectral_left @ self.spectral_right.transpose(0, 1)
#         ) / math.sqrt(float(self.spectral_rank))
#         return _bounded_frobenius(
#             raw, self.maximum_spectral_shift_norm
#         )

#     def bias(self) -> Tensor:
#         return _bounded_frobenius(
#             self.raw_bias, self.maximum_bias_norm
#         )

#     def forward(self, features: Tensor, spectral_anchors: Tensor) -> Tensor:
#         if features.dim() != 2 or features.size(1) != self.d_model:
#             raise ValueError(
#                 f"features must be [N,{self.d_model}], got {tuple(features.shape)}"
#             )
#         if spectral_anchors.dim() != 2 or spectral_anchors.size(1) != (
#             self.spectral_anchor_dim
#         ):
#             raise ValueError(
#                 "spectral_anchors must be "
#                 f"[N,{self.spectral_anchor_dim}], got "
#                 f"{tuple(spectral_anchors.shape)}"
#             )
#         if features.size(0) != spectral_anchors.size(0):
#             raise ValueError("features and spectral anchors are not aligned")
#         if not torch.isfinite(features).all() or not torch.isfinite(
#             spectral_anchors
#         ).all():
#             raise RuntimeError("transport inputs contain NaN/Inf")

#         matrix = self.matrix().to(
#             device=features.device, dtype=features.dtype
#         )
#         spectral_shift = self.spectral_shift().to(
#             device=features.device, dtype=features.dtype
#         )
#         bias = self.bias().to(device=features.device, dtype=features.dtype)
#         return (
#             features @ matrix.transpose(0, 1)
#             + spectral_anchors @ spectral_shift.transpose(0, 1)
#             + bias.view(1, -1)
#         )

#     def inverse(self, current_features: Tensor, spectral_anchors: Tensor) -> Tensor:
#         if current_features.dim() != 2 or current_features.size(1) != (
#             self.d_model
#         ):
#             raise ValueError("current_features have the wrong shape")
#         if spectral_anchors.shape != (
#             current_features.size(0),
#             self.spectral_anchor_dim,
#         ):
#             raise ValueError("spectral_anchors have the wrong shape")
#         matrix = self.matrix().to(
#             device=current_features.device, dtype=current_features.dtype
#         )
#         spectral_shift = self.spectral_shift().to(
#             device=current_features.device, dtype=current_features.dtype
#         )
#         bias = self.bias().to(
#             device=current_features.device, dtype=current_features.dtype
#         )
#         centered = (
#             current_features
#             - spectral_anchors @ spectral_shift.transpose(0, 1)
#             - bias.view(1, -1)
#         )
#         return torch.linalg.solve(
#             matrix, centered.transpose(0, 1)
#         ).transpose(0, 1)

#     def tensors(self, *, detach: bool = False) -> Dict[str, Tensor]:
#         values = {
#             "matrix": self.matrix(),
#             "spectral_shift": self.spectral_shift(),
#             "bias": self.bias(),
#         }
#         if detach:
#             return {name: value.detach() for name, value in values.items()}
#         return values

#     @torch.no_grad()
#     def diagnostics(self) -> Dict[str, float]:
#         matrix = self.matrix()
#         singular = torch.linalg.svdvals(matrix)
#         minimum = float(singular.min().item())
#         maximum = float(singular.max().item())
#         return {
#             "matrix_residual_frobenius": float(
#                 self.matrix_residual().norm().item()
#             ),
#             "spectral_shift_frobenius": float(
#                 self.spectral_shift().norm().item()
#             ),
#             "bias_norm": float(self.bias().norm().item()),
#             "minimum_singular_value": minimum,
#             "maximum_singular_value": maximum,
#             "condition_number": maximum / max(minimum, 1e-12),
#         }


# class NECILModel(nn.Module):
#     """Evolving spectral-conditioned geometry architecture.

#     Main representation
#     -------------------
#     ``x -> plastic spectral-spatial backbone -> z_t``

#     Persistent memory
#     -----------------
#     ``p(s|c) p(z_t|s,c)`` where ``s`` is a frozen base-fitted anchor of
#     ordered physical spectra. No old image, spectrum, feature, prototype
#     classifier weight, teacher, or task adapter is stored.

#     Incremental synchronization
#     ---------------------------
#     A temporary phase transport ``T_t(z,s)=M_t z + H_t s + b_t`` maps old
#     aggregate geometry and old geometry replay into the current feature space.
#     Its effect is absorbed into the GeometryBank at commit.
#     """

#     ARCHITECTURE_VERSION = 1
#     METHOD_NAME = "Evolving spectral-conditioned geometry"

#     FEATURE_MODE = "features"
#     BASE_CE_MODE = "base_ce"
#     GEOMETRY_MODE = "geometry"

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.args = args
#         self.device = torch.device(getattr(args, "device", "cpu"))
#         self.d_model = int(getattr(args, "d_model", 128))
#         self.feature_rank = int(
#             getattr(args, "subspace_rank", getattr(args, "rank", 5))
#         )
#         self.spectral_anchor_dim = int(
#             getattr(args, "spectral_anchor_dim", 12)
#         )
#         self.spectral_rank = int(getattr(args, "spectral_rank", 4))

#         self.backbone = SSMBackbone(args)
#         self.geometry_bank = GeometryBank(
#             d_model=self.d_model,
#             rank=self.feature_rank,
#             raw_spectral_dim=int(self.backbone.raw_num_bands),
#             spectral_anchor_dim=self.spectral_anchor_dim,
#             spectral_rank=self.spectral_rank,
#             device=self.device,
#             feature_variance_floor=float(
#                 getattr(args, "geom_var_floor", 1e-4)
#             ),
#             spectral_variance_floor=float(
#                 getattr(args, "spectral_variance_floor", 1e-4)
#             ),
#             feature_shrinkage=float(
#                 getattr(args, "geometry_variance_shrinkage", 0.10)
#             ),
#             spectral_shrinkage=float(
#                 getattr(args, "spectral_variance_shrinkage", 0.20)
#             ),
#             shrinkage_tau=float(
#                 getattr(args, "geometry_shrinkage_tau", 20.0)
#             ),
#             retained_energy=float(
#                 getattr(args, "rank_energy_threshold", 0.95)
#             ),
#             noise_ratio=float(
#                 getattr(args, "rank_noise_ratio_threshold", 1.50)
#             ),
#             coupling_ridge=float(
#                 getattr(args, "spectral_coupling_ridge", 1e-2)
#             ),
#             coupling_shrinkage_tau=float(
#                 getattr(args, "spectral_coupling_shrinkage_tau", 20.0)
#             ),
#             robust_iterations=int(
#                 getattr(args, "robust_geometry_iterations", 3)
#             ),
#             robust_huber_delta=float(
#                 getattr(args, "robust_geometry_huber_delta", 2.5)
#             ),
#             spatial_purity_power=float(
#                 getattr(args, "spatial_purity_power", 1.0)
#             ),
#             spectral_weight=float(
#                 getattr(args, "spectral_weight", 1.0)
#             ),
#             feature_weight=float(
#                 getattr(args, "feature_weight", 1.0)
#             ),
#             logdet_weight=float(
#                 getattr(args, "energy_logdet_weight", 1.0)
#             ),
#             boundary_oversample_factor=int(
#                 getattr(args, "boundary_oversample_factor", 8)
#             ),
#             maximum_transport_condition=float(
#                 getattr(args, "maximum_transport_condition", 2.0)
#             ),
#         )

#         temperature = float(getattr(args, "energy_temperature", 1.0))
#         self.classifier = GeometryEnergyClassifier(
#             d_model=self.d_model,
#             temperature=temperature,
#             expected_spectral_anchor_dim=self.spectral_anchor_dim,
#             expected_bank_schema_version=int(
#                 self.geometry_bank.SCHEMA_VERSION
#             ),
#             require_bound_contract=False,
#         )

#         self.phase_transport: Optional[
#             SpectralConditionedPhaseTransport
#         ] = None
#         self.base_ce_head: Optional[nn.Linear] = None
#         self.base_ce_class_ids: List[int] = []

#         self.current_phase = 0
#         self.seen_classes: List[int] = []
#         self.old_classes: List[int] = []
#         self.new_classes: List[int] = []
#         self.phase_old_digest: Optional[str] = None
#         self.base_handoff: Optional[Dict[str, Any]] = None
#         self.phase_mode = "base"

#         self.to(self.device)
#         self.assert_architecture_contract()

#     def _new_phase_transport(self) -> SpectralConditionedPhaseTransport:
#         return SpectralConditionedPhaseTransport(
#             d_model=self.d_model,
#             spectral_anchor_dim=self.spectral_anchor_dim,
#             feature_rank=int(getattr(self.args, "transport_rank", 8)),
#             spectral_rank=int(
#                 getattr(self.args, "transport_spectral_rank", 8)
#             ),
#             maximum_matrix_residual_norm=float(
#                 getattr(
#                     self.args,
#                     "transport_max_matrix_residual_norm",
#                     0.20,
#                 )
#             ),
#             maximum_spectral_shift_norm=float(
#                 getattr(
#                     self.args,
#                     "transport_max_spectral_shift_norm",
#                     0.50,
#                 )
#             ),
#             maximum_bias_norm=float(
#                 getattr(self.args, "transport_max_bias_norm", 0.25)
#             ),
#         ).to(self.device)

#     # ------------------------------------------------------------------
#     # Architecture and checkpoint contract
#     # ------------------------------------------------------------------
#     def architecture_summary(self) -> Dict[str, Any]:
#         return {
#             "architecture_version": self.ARCHITECTURE_VERSION,
#             "method": self.METHOD_NAME,
#             "factorization": "p(s|c)p(z|s,c)",
#             "encoder": "plastic spectral-spatial S4D bridge",
#             "spectral_object": "base-fitted frozen physical spectral anchor",
#             "phase_transport": "low-rank invertible T(z,s)=Mz+Hs+b",
#             "classifier": "parameter-free joint energy",
#             "old_rows_evolve": True,
#             "stores_exemplars": False,
#             "stores_old_features": False,
#             "stores_old_spectra": False,
#             "uses_teacher": False,
#             "uses_task_adapters": False,
#             "uses_candidate_row_corrections": False,
#         }

#     def architecture_digest(self) -> str:
#         payload = (
#             self.architecture_summary(),
#             self.backbone.backbone_contract(),
#             self.classifier.classifier_contract(),
#         )
#         return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

#     def assert_architecture_contract(self) -> None:
#         if self.backbone.d_model != self.d_model:
#             raise RuntimeError("backbone and model feature widths differ")
#         if self.geometry_bank.d_model != self.d_model:
#             raise RuntimeError("bank and model feature widths differ")
#         if self.geometry_bank.raw_spectral_dim != (
#             self.backbone.raw_num_bands
#         ):
#             raise RuntimeError("backbone and bank raw spectral widths differ")
#         if self.geometry_bank.spectral_anchor_dim != (
#             self.spectral_anchor_dim
#         ):
#             raise RuntimeError("bank spectral-anchor width is inconsistent")
#         required_bank = (
#             "fit_spectral_anchor",
#             "encode_spectra",
#             "extract_rows",
#             "joint_energy_matrix",
#             "sample_replay",
#             "build_transported_rows",
#             "commit_incremental_phase",
#             "contract_digest",
#         )
#         missing = [
#             name
#             for name in required_bank
#             if not callable(getattr(self.geometry_bank, name, None))
#         ]
#         if missing:
#             raise RuntimeError(f"GeometryBank lacks required methods: {missing}")
#         if any(
#             parameter.requires_grad
#             for parameter in self.geometry_bank.parameters()
#         ):
#             raise RuntimeError("GeometryBank must contain buffers only")
#         if any(
#             parameter.requires_grad
#             for parameter in self.classifier.parameters()
#         ):
#             raise RuntimeError("classifier must be parameter-free")
#         contract = self.classifier.classifier_contract()
#         if contract.get("factorization") != "p(s|c)p(z|s,c)":
#             raise RuntimeError("classifier factorization is inconsistent")

#     def get_extra_state(self) -> Dict[str, Any]:
#         return {
#             "architecture_version": self.ARCHITECTURE_VERSION,
#             "architecture_digest": self.architecture_digest(),
#             "current_phase": self.current_phase,
#             "seen_classes": list(self.seen_classes),
#             "old_classes": list(self.old_classes),
#             "new_classes": list(self.new_classes),
#             "phase_old_digest": self.phase_old_digest,
#             "phase_mode": self.phase_mode,
#             "base_ce_class_ids": list(self.base_ce_class_ids),
#             "base_handoff": copy.deepcopy(self.base_handoff),
#         }

#     def set_extra_state(self, state: Any) -> None:
#         if not isinstance(state, Mapping):
#             return
#         version = int(state.get("architecture_version", -1))
#         if version != self.ARCHITECTURE_VERSION:
#             raise RuntimeError(
#                 f"NECILModel architecture version mismatch: {version}"
#             )
#         self.current_phase = int(state.get("current_phase", 0))
#         self.seen_classes = [int(v) for v in state.get("seen_classes", [])]
#         self.old_classes = [int(v) for v in state.get("old_classes", [])]
#         self.new_classes = [int(v) for v in state.get("new_classes", [])]
#         digest = state.get("phase_old_digest")
#         self.phase_old_digest = None if digest is None else str(digest)
#         self.phase_mode = str(state.get("phase_mode", "base"))
#         self.base_ce_class_ids = [
#             int(v) for v in state.get("base_ce_class_ids", [])
#         ]
#         if self.base_ce_class_ids and self.base_ce_head is None:
#             self.ensure_base_ce_head(self.base_ce_class_ids)
#         if self.phase_mode == "incremental" and self.phase_transport is None:
#             self.phase_transport = self._new_phase_transport()
#         self.base_handoff = copy.deepcopy(state.get("base_handoff"))

#     # ------------------------------------------------------------------
#     # Feature and spectrum extraction
#     # ------------------------------------------------------------------
#     def _validate_output(self, output: Mapping[str, Any]) -> Dict[str, Any]:
#         required = (
#             "geometry_features",
#             "raw_center_spectra",
#             "spatial_purity",
#         )
#         missing = [name for name in required if name not in output]
#         if missing:
#             raise RuntimeError(f"backbone output lacks {missing}")
#         result = dict(output)
#         features = result["geometry_features"]
#         spectra = result["raw_center_spectra"]
#         purity = result["spatial_purity"]
#         if not torch.is_tensor(features) or features.dim() != 2:
#             raise RuntimeError("geometry_features must be [N,D]")
#         if features.size(1) != self.d_model:
#             raise RuntimeError("geometry feature width changed")
#         if not torch.is_tensor(spectra) or spectra.shape != (
#             features.size(0),
#             self.backbone.raw_num_bands,
#         ):
#             raise RuntimeError("raw_center_spectra shape is invalid")
#         if not torch.is_tensor(purity) or purity.numel() != features.size(0):
#             raise RuntimeError("spatial_purity shape is invalid")
#         if not all(
#             torch.isfinite(value).all()
#             for value in (features, spectra, purity)
#         ):
#             raise RuntimeError("backbone output contains NaN/Inf")
#         result["spatial_purity"] = purity.flatten().clamp(0.0, 1.0)
#         if bool(self.geometry_bank.anchor_ready.item()):
#             result["spectral_anchors"] = self.geometry_bank.encode_spectra(
#                 spectra
#             )
#         return result

#     def forward_features(
#         self,
#         inputs: Tensor,
#         raw_spectral_patch: Optional[Tensor] = None,
#         *,
#         deterministic: bool = True,
#     ) -> Dict[str, Any]:
#         inputs = inputs.to(self.device)
#         raw = (
#             None
#             if raw_spectral_patch is None
#             else raw_spectral_patch.to(self.device)
#         )
#         output = (
#             self.backbone.forward_geometry(inputs, raw)
#             if deterministic
#             else self.backbone(inputs, raw)
#         )
#         return self._validate_output(output)

#     extract_geometry_features = forward_features

#     def create_reference_backbone(self) -> SSMBackbone:
#         """Return a non-persistent frozen snapshot for one incremental phase."""
#         if self.phase_mode != "incremental":
#             raise RuntimeError(
#                 "create_reference_backbone requires incremental mode"
#             )
#         reference = copy.deepcopy(self.backbone).to(self.device)
#         reference.freeze_all()
#         reference.eval()
#         return reference

#     @torch.no_grad()
#     def encode_reference(
#         self,
#         reference_backbone: SSMBackbone,
#         inputs: Tensor,
#         raw_spectral_patch: Optional[Tensor] = None,
#     ) -> Dict[str, Any]:
#         output = reference_backbone.forward_geometry(
#             inputs.to(self.device),
#             None
#             if raw_spectral_patch is None
#             else raw_spectral_patch.to(self.device),
#         )
#         return self._validate_output(output)

#     # ------------------------------------------------------------------
#     # Classifier interface
#     # ------------------------------------------------------------------
#     def infer_seen_classes(self) -> List[int]:
#         valid = self.geometry_bank.valid_mask()
#         return torch.nonzero(
#             valid.detach().cpu().bool(), as_tuple=False
#         ).flatten().tolist()

#     def compute_logits_from_features(
#         self,
#         features: Tensor,
#         *,
#         seen_classes: Sequence[int],
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         targets: Optional[Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Sequence[int]] = None,
#         new_classes: Optional[Sequence[int]] = None,
#         mode: str = "spectral_conditioned_joint",
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#     ) -> Union[Tensor, Dict[str, Any]]:
#         return self.classifier(
#             features,
#             geometry_bank=self.geometry_bank,
#             seen_classes=seen_classes,
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#             temporary_rows=temporary_rows,
#             targets=targets,
#             targets_are_global=targets_are_global,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             mode=mode,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )

#     def forward(
#         self,
#         inputs: Tensor,
#         raw_spectral_patch: Optional[Tensor] = None,
#         *,
#         mode: str = GEOMETRY_MODE,
#         seen_classes: Optional[Sequence[int]] = None,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         targets: Optional[Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Sequence[int]] = None,
#         new_classes: Optional[Sequence[int]] = None,
#         classifier_mode: str = "spectral_conditioned_joint",
#         deterministic: bool = True,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#     ) -> Union[Tensor, Dict[str, Any]]:
#         output = self.forward_features(
#             inputs,
#             raw_spectral_patch,
#             deterministic=deterministic,
#         )
#         selected = str(mode).strip().lower()
#         if selected == self.FEATURE_MODE:
#             return output
#         if selected == self.BASE_CE_MODE:
#             logits = self.base_ce_logits(output["geometry_features"])
#             return {
#                 "logits": logits,
#                 "geometry_features": output["geometry_features"],
#                 "raw_center_spectra": output["raw_center_spectra"],
#                 "spatial_purity": output["spatial_purity"],
#             }
#         if selected != self.GEOMETRY_MODE:
#             raise ValueError(
#                 f"mode must be one of features/base_ce/geometry, got {mode!r}"
#             )
#         seen = (
#             self.infer_seen_classes()
#             if seen_classes is None
#             else _unique_ids(seen_classes, name="seen_classes")
#         )
#         result = self.compute_logits_from_features(
#             output["geometry_features"],
#             seen_classes=seen,
#             raw_spectra=output["raw_center_spectra"],
#             temporary_rows=temporary_rows,
#             targets=targets,
#             targets_are_global=targets_are_global,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             mode=classifier_mode,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             return_diagnostics=return_diagnostics,
#         )
#         if isinstance(result, Mapping):
#             merged = dict(result)
#             merged["geometry_features"] = output["geometry_features"]
#             merged["raw_center_spectra"] = output["raw_center_spectra"]
#             merged["spatial_purity"] = output["spatial_purity"]
#             return merged
#         return result

#     # ------------------------------------------------------------------
#     # Base CE warm-up and geometry handoff
#     # ------------------------------------------------------------------
#     def ensure_base_ce_head(self, base_class_ids: Sequence[int]) -> nn.Linear:
#         ids = _unique_ids(base_class_ids, name="base_class_ids")
#         if len(ids) < 2:
#             raise ValueError("base phase requires at least two classes")
#         if self.base_ce_head is None or self.base_ce_head.out_features != len(ids):
#             self.base_ce_head = nn.Linear(self.d_model, len(ids)).to(self.device)
#             nn.init.normal_(self.base_ce_head.weight, mean=0.0, std=0.01)
#             nn.init.zeros_(self.base_ce_head.bias)
#         self.base_ce_class_ids = list(ids)
#         return self.base_ce_head

#     def base_ce_logits(self, features: Tensor) -> Tensor:
#         if self.base_ce_head is None:
#             raise RuntimeError("base CE head is not initialized")
#         if features.dim() != 2 or features.size(1) != self.d_model:
#             raise ValueError("base CE features have the wrong shape")
#         return self.base_ce_head(features)

#     def drop_base_ce_head(self) -> None:
#         self.base_ce_head = None
#         self.base_ce_class_ids = []

#     @torch.no_grad()
#     def fit_spectral_anchor(
#         self,
#         raw_center_spectra: Tensor,
#         *,
#         freeze: bool = True,
#         overwrite: bool = False,
#     ) -> Dict[str, Any]:
#         if self.current_phase != 0 or self.phase_mode != "base":
#             raise RuntimeError("spectral anchor can only be fitted in base phase")
#         return self.geometry_bank.fit_spectral_anchor(
#             raw_center_spectra,
#             freeze=freeze,
#             overwrite=overwrite,
#         )

#     @torch.no_grad()
#     def build_geometry_rows(
#         self,
#         *,
#         class_ids: Sequence[int],
#         features: Tensor,
#         raw_center_spectra: Tensor,
#         labels: Tensor,
#         spatial_purity: Optional[Tensor] = None,
#     ) -> Dict[int, Row]:
#         ids = _unique_ids(class_ids, name="class_ids")
#         rows = self.geometry_bank.extract_rows(
#             features,
#             labels,
#             raw_spectra=raw_center_spectra,
#             sample_weights=spatial_purity,
#             class_ids=ids,
#         )
#         if set(rows) != set(ids):
#             raise RuntimeError("GeometryBank returned an incomplete row set")
#         return rows

#     @torch.no_grad()
#     def build_geometry_rows_from_output(
#         self,
#         output: Mapping[str, Any],
#         labels: Tensor,
#         class_ids: Sequence[int],
#     ) -> Dict[int, Row]:
#         return self.build_geometry_rows(
#             class_ids=class_ids,
#             features=output["geometry_features"],
#             raw_center_spectra=output["raw_center_spectra"],
#             labels=labels,
#             spatial_purity=output.get("spatial_purity"),
#         )

#     @torch.no_grad()
#     def finalize_base_geometry(
#         self,
#         *,
#         base_class_ids: Sequence[int],
#         features: Tensor,
#         raw_center_spectra: Tensor,
#         labels: Tensor,
#         spatial_purity: Optional[Tensor] = None,
#         update_statistics: bool = True,
#         drop_ce_head: bool = True,
#         calibrated_temperature: Optional[float] = None,
#     ) -> Dict[str, Any]:
#         if self.current_phase != 0 or self.phase_mode != "base":
#             raise RuntimeError("finalize_base_geometry is base-phase only")
#         ids = _unique_ids(base_class_ids, name="base_class_ids")
#         self.geometry_bank.assert_anchor_ready()
#         if not bool(self.geometry_bank.anchor_frozen.item()):
#             raise RuntimeError("base spectral anchor must be frozen")
#         if bool(self.geometry_bank.valid_mask().any().item()):
#             raise RuntimeError("base geometry can only be committed once")

#         bank_snapshot = self.geometry_bank.export_snapshot()
#         classifier_state = copy.deepcopy(self.classifier.state_dict())
#         previous_state = (
#             list(self.seen_classes),
#             list(self.old_classes),
#             list(self.new_classes),
#             copy.deepcopy(self.base_handoff),
#         )
#         try:
#             rows = self.build_geometry_rows(
#                 class_ids=ids,
#                 features=features,
#                 raw_center_spectra=raw_center_spectra,
#                 labels=labels,
#                 spatial_purity=spatial_purity,
#             )
#             commit = self.geometry_bank.commit_base_rows(
#                 rows,
#                 base_class_ids=ids,
#                 phase=0,
#             )
#             statistics = None
#             if update_statistics:
#                 statistics = self.geometry_bank.update_statistics(
#                     features,
#                     labels,
#                     ids,
#                     raw_spectra=raw_center_spectra,
#                 )
#             admission = self.geometry_bank.admission_report(
#                 ids,
#                 require_statistics=update_statistics,
#             )
#             if not admission["ok"]:
#                 raise RuntimeError(
#                     "base geometry admission failed: "
#                     + "; ".join(admission["errors"])
#                 )
#             if calibrated_temperature is not None:
#                 self.classifier.set_temperature(calibrated_temperature)
#             contract_digest = self.classifier.bind_geometry_bank_contract(
#                 self.geometry_bank,
#                 require_frozen_anchor=True,
#                 require_committed_rows=True,
#                 enforce_after_binding=True,
#             )
#             self.seen_classes = list(ids)
#             self.old_classes = list(ids)
#             self.new_classes = []
#             if drop_ce_head:
#                 self.drop_base_ce_head()
#             self.backbone.freeze_all()
#             self.phase_mode = "evaluation"
#             self.base_handoff = {
#                 "commit": commit,
#                 "statistics": statistics,
#                 "admission": admission,
#                 "bank_contract_digest": contract_digest,
#                 "factorization": "p(s|c)p(z|s,c)",
#                 "feature_space_frozen": False,
#                 "atomic": True,
#             }
#             return copy.deepcopy(self.base_handoff)
#         except Exception:
#             self.geometry_bank.load_snapshot(bank_snapshot, strict=True)
#             self.classifier.load_state_dict(classifier_state)
#             (
#                 self.seen_classes,
#                 self.old_classes,
#                 self.new_classes,
#                 self.base_handoff,
#             ) = previous_state
#             raise

#     # ------------------------------------------------------------------
#     # Incremental phase lifecycle
#     # ------------------------------------------------------------------
#     def set_base_mode(self) -> None:
#         if bool(self.geometry_bank.valid_mask().any().item()):
#             raise RuntimeError(
#                 "committed rows exist; reset the bank before restarting base training"
#             )
#         self.current_phase = 0
#         self.phase_mode = "base"
#         self.seen_classes = []
#         self.old_classes = []
#         self.new_classes = []
#         self.phase_old_digest = None
#         self.phase_transport = None
#         self.backbone.enable_base_training()
#         self.classifier.require_bound_contract = False

#     def begin_incremental_phase(
#         self,
#         *,
#         phase: int,
#         old_class_ids: Sequence[int],
#         new_class_ids: Sequence[int],
#     ) -> Dict[str, Any]:
#         phase = int(phase)
#         if phase <= 0:
#             raise ValueError("incremental phase must be >= 1")
#         old_ids = _unique_ids(old_class_ids, name="old_class_ids")
#         new_ids = _unique_ids(new_class_ids, name="new_class_ids")
#         if set(old_ids) & set(new_ids):
#             raise RuntimeError("old and new class IDs overlap")
#         committed = self.infer_seen_classes()
#         if set(committed) != set(old_ids):
#             raise RuntimeError(
#                 f"old classes {sorted(old_ids)} do not match committed rows "
#                 f"{sorted(committed)}"
#             )
#         self.geometry_bank.assert_valid(old_ids, strict=True)
#         self.geometry_bank.assert_anchor_ready()
#         if not bool(self.geometry_bank.anchor_frozen.item()):
#             raise RuntimeError("incremental phase requires a frozen anchor")
#         if self.classifier.bound_bank_contract_digest != (
#             self.geometry_bank.contract_digest()
#         ):
#             raise RuntimeError("finalize and bind base geometry first")

#         self.current_phase = phase
#         self.old_classes = list(old_ids)
#         self.new_classes = list(new_ids)
#         self.seen_classes = [*old_ids, *new_ids]
#         self.phase_old_digest = self.geometry_bank.rows_digest(old_ids)
#         self.phase_mode = "incremental"
#         self.phase_transport = self._new_phase_transport()
#         self.backbone.enable_incremental_plasticity()
#         self.classifier.eval()
#         return {
#             "phase": phase,
#             "old_class_ids": old_ids,
#             "new_class_ids": new_ids,
#             "old_digest": self.phase_old_digest,
#             "transport": self.phase_transport.diagnostics(),
#         }

#     def _require_transport(self) -> SpectralConditionedPhaseTransport:
#         if self.phase_mode != "incremental" or self.phase_transport is None:
#             raise RuntimeError("no active incremental phase transport")
#         return self.phase_transport

#     def encoder_optimizer_groups(
#         self,
#         *,
#         base_lr: float,
#         weight_decay: float = 0.0,
#     ) -> List[Dict[str, Any]]:
#         if self.phase_mode != "incremental":
#             raise RuntimeError("encoder groups require incremental mode")
#         self.backbone.enable_incremental_plasticity()
#         return self.backbone.incremental_parameter_groups(
#             base_lr=base_lr,
#             weight_decay=weight_decay,
#         )

#     def transport_parameters(self) -> List[nn.Parameter]:
#         transport = self._require_transport()
#         return list(transport.parameters())

#     def configure_transport_step(self) -> None:
#         """Freeze encoder; update transport from pair and old-replay losses."""
#         transport = self._require_transport()
#         self.backbone.freeze_all()
#         for parameter in transport.parameters():
#             parameter.requires_grad_(True)
#             parameter.grad = None
#         transport.train(True)

#     def configure_encoder_step(self) -> None:
#         """Freeze transport; update encoder using real new query samples."""
#         transport = self._require_transport()
#         self.backbone.enable_incremental_plasticity()
#         for parameter in transport.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None
#         transport.eval()

#     def transport_pair_loss(
#         self,
#         previous_features: Tensor,
#         current_features: Tensor,
#         *,
#         raw_center_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         beta: float = 0.5,
#     ) -> Dict[str, Tensor]:
#         """Fit transport to observed current-task coordinate correspondences.

#         The current feature target is detached. This loss updates transport,
#         not the current encoder.
#         """
#         transport = self._require_transport()
#         if (raw_center_spectra is None) == (spectral_anchors is None):
#             raise ValueError(
#                 "pass exactly one of raw_center_spectra or spectral_anchors"
#             )
#         anchors = (
#             self.geometry_bank.encode_spectra(raw_center_spectra)
#             if raw_center_spectra is not None
#             else spectral_anchors
#         )
#         assert anchors is not None
#         predicted = transport(previous_features.detach(), anchors.detach())
#         target = current_features.detach()
#         pair = F.smooth_l1_loss(
#             predicted,
#             target,
#             beta=float(max(beta, 1e-6)),
#             reduction="mean",
#         )
#         inverse = transport.inverse(predicted, anchors.detach())
#         inverse_error = F.mse_loss(
#             inverse, previous_features.detach(), reduction="mean"
#         )
#         return {
#             "loss": pair,
#             "pair_loss": pair,
#             "inverse_error": inverse_error.detach(),
#         }

#     def current_transport_tensors(
#         self, *, detach: bool = True
#     ) -> Dict[str, Tensor]:
#         return self._require_transport().tensors(detach=detach)

#     @torch.no_grad()
#     def build_transported_old_rows(self) -> Tuple[Dict[int, Row], Dict[str, Any]]:
#         tensors = self.current_transport_tensors(detach=True)
#         return self.geometry_bank.build_transported_rows(
#             self.old_classes,
#             matrix=tensors["matrix"],
#             spectral_shift=tensors["spectral_shift"],
#             bias=tensors["bias"],
#         )

#     @torch.no_grad()
#     def build_phase_rows(
#         self,
#         *,
#         new_support_features: Tensor,
#         new_support_raw_spectra: Tensor,
#         new_support_labels: Tensor,
#         new_support_purity: Optional[Tensor] = None,
#     ) -> Tuple[Dict[int, Row], Dict[str, Any]]:
#         transported, report = self.build_transported_old_rows()
#         new_rows = self.build_geometry_rows(
#             class_ids=self.new_classes,
#             features=new_support_features,
#             raw_center_spectra=new_support_raw_spectra,
#             labels=new_support_labels,
#             spatial_purity=new_support_purity,
#         )
#         overlap = set(transported) & set(new_rows)
#         if overlap:
#             raise RuntimeError(f"old/new temporary row overlap: {sorted(overlap)}")
#         return {**transported, **new_rows}, report

#     def sample_transported_replay(
#         self,
#         *,
#         samples_per_class: int,
#         generator: Optional[torch.Generator] = None,
#     ) -> Dict[str, Tensor]:
#         """Sample old geometry and map it through the active transport.

#         The sampled source tensors are detached aggregate replay. The returned
#         current features retain gradients to the transport when its parameters
#         are enabled.
#         """
#         transport = self._require_transport()
#         source = self.geometry_bank.sample_replay(
#             self.old_classes,
#             samples_per_class=samples_per_class,
#             generator=generator,
#         )
#         current = transport(
#             source["features"], source["spectral_anchors"]
#         )
#         return {
#             "source_features": source["features"],
#             "features": current,
#             "spectral_anchors": source["spectral_anchors"],
#             "labels": source["labels"],
#         }

#     def sample_phase_boundary_replay(
#         self,
#         *,
#         temporary_rows: Mapping[int, Mapping[str, Any]],
#         typical_per_class: int,
#         boundary_per_class: int,
#         oversample_factor: Optional[int] = None,
#         typical_quantile: float = 0.95,
#         generator: Optional[torch.Generator] = None,
#     ) -> Dict[str, Tensor]:
#         """Select typical and low-positive-margin old replay.

#         Selection is non-differentiable, but indexing is applied to the
#         differentiable transported feature tensor, so the selected replay loss
#         still updates the phase transport.
#         """
#         typical_count = int(typical_per_class)
#         boundary_count = int(boundary_per_class)
#         if typical_count < 0 or boundary_count < 0:
#             raise ValueError("replay counts must be non-negative")
#         total = typical_count + boundary_count
#         if total <= 0:
#             raise ValueError("at least one replay sample is required")
#         factor = int(
#             oversample_factor
#             if oversample_factor is not None
#             else self.geometry_bank.boundary_oversample_factor
#         )
#         factor = max(factor, 2)
#         replay = self.sample_transported_replay(
#             samples_per_class=total * factor,
#             generator=generator,
#         )
#         seen = [*self.old_classes, *self.new_classes]
#         with torch.no_grad():
#             energy = self.geometry_bank.joint_energy_matrix(
#                 replay["features"].detach(),
#                 seen,
#                 spectral_anchors=replay["spectral_anchors"],
#                 rows=temporary_rows,
#             )
#         positions = {class_id: index for index, class_id in enumerate(seen)}
#         selected: List[Tensor] = []
#         replay_kind: List[Tensor] = []
#         rivals: List[Tensor] = []
#         margins: List[Tensor] = []

#         for class_id in self.old_classes:
#             global_indices = torch.nonzero(
#                 replay["labels"].eq(class_id), as_tuple=False
#             ).flatten()
#             class_energy = energy.index_select(0, global_indices)
#             true_position = positions[class_id]
#             true_energy = class_energy[:, true_position]
#             rival_energy = class_energy.clone()
#             rival_energy[:, true_position] = float("inf")
#             nearest_energy, nearest_position = rival_energy.min(dim=1)
#             margin = nearest_energy - true_energy
#             cutoff = torch.quantile(
#                 true_energy,
#                 float(min(max(typical_quantile, 0.50), 0.999)),
#             )
#             typical_mask = true_energy.le(cutoff)
#             boundary_candidates = torch.nonzero(
#                 typical_mask & margin.ge(0.0), as_tuple=False
#             ).flatten()
#             if boundary_candidates.numel() > 0:
#                 boundary_order = torch.argsort(
#                     margin.index_select(0, boundary_candidates)
#                 )
#                 boundary_local = boundary_candidates.index_select(
#                     0, boundary_order
#                 )[:boundary_count]
#             else:
#                 boundary_local = global_indices.new_empty((0,))

#             typical_candidates = torch.nonzero(
#                 typical_mask, as_tuple=False
#             ).flatten()
#             if boundary_local.numel() > 0:
#                 keep = torch.ones(
#                     typical_candidates.numel(),
#                     device=self.device,
#                     dtype=torch.bool,
#                 )
#                 for chosen in boundary_local.tolist():
#                     keep &= typical_candidates.ne(int(chosen))
#                 typical_candidates = typical_candidates[keep]
#             if generator is not None and typical_candidates.numel() > 0:
#                 permutation = torch.randperm(
#                     typical_candidates.numel(),
#                     device=self.device,
#                     generator=generator,
#                 )
#                 typical_candidates = typical_candidates.index_select(
#                     0, permutation
#                 )
#             typical_local = typical_candidates[:typical_count]

#             if boundary_local.numel() < boundary_count:
#                 missing = boundary_count - boundary_local.numel()
#                 typical_by_margin = typical_candidates.index_select(
#                     0, torch.argsort(margin.index_select(0, typical_candidates))
#                 )
#                 already = set(int(v) for v in boundary_local.tolist())
#                 additions = [
#                     int(v)
#                     for v in typical_by_margin.tolist()
#                     if int(v) not in already
#                 ][:missing]
#                 if additions:
#                     boundary_local = torch.cat(
#                         [
#                             boundary_local,
#                             torch.tensor(
#                                 additions,
#                                 device=self.device,
#                                 dtype=torch.long,
#                             ),
#                         ]
#                     )
#             if typical_local.numel() < typical_count:
#                 missing = typical_count - typical_local.numel()
#                 typical_by_energy = typical_candidates.index_select(
#                     0, torch.argsort(true_energy.index_select(0, typical_candidates))
#                 )
#                 already = set(
#                     int(v)
#                     for v in torch.cat(
#                         [typical_local, boundary_local], dim=0
#                     ).tolist()
#                 )
#                 additions = [
#                     int(v)
#                     for v in typical_by_energy.tolist()
#                     if int(v) not in already
#                 ][:missing]
#                 if additions:
#                     typical_local = torch.cat(
#                         [
#                             typical_local,
#                             torch.tensor(
#                                 additions,
#                                 device=self.device,
#                                 dtype=torch.long,
#                             ),
#                         ]
#                     )
#             if typical_local.numel() != typical_count or (
#                 boundary_local.numel() != boundary_count
#             ):
#                 raise RuntimeError(
#                     f"insufficient replay candidates for class {class_id}"
#                 )

#             local = torch.cat([typical_local, boundary_local], dim=0)
#             chosen_global = global_indices.index_select(0, local)
#             selected.append(chosen_global)
#             replay_kind.append(
#                 torch.cat(
#                     [
#                         torch.zeros(
#                             typical_count,
#                             device=self.device,
#                             dtype=torch.long,
#                         ),
#                         torch.ones(
#                             boundary_count,
#                             device=self.device,
#                             dtype=torch.long,
#                         ),
#                     ]
#                 )
#             )
#             rivals.append(
#                 torch.tensor(
#                     seen,
#                     device=self.device,
#                     dtype=torch.long,
#                 ).index_select(
#                     0, nearest_position.index_select(0, local)
#                 )
#             )
#             margins.append(margin.index_select(0, local))

#         index = torch.cat(selected, dim=0)
#         return {
#             "features": replay["features"].index_select(0, index),
#             "source_features": replay["source_features"].index_select(
#                 0, index
#             ),
#             "spectral_anchors": replay["spectral_anchors"].index_select(
#                 0, index
#             ),
#             "labels": replay["labels"].index_select(0, index),
#             "replay_kind": torch.cat(replay_kind, dim=0),
#             "rival_labels": torch.cat(rivals, dim=0),
#             "margins": torch.cat(margins, dim=0),
#         }

#     @torch.no_grad()
#     def commit_incremental_phase(
#         self,
#         *,
#         new_rows: Mapping[int, Mapping[str, Any]],
#         phase: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         self._require_transport()
#         phase_value = self.current_phase if phase is None else int(phase)
#         if phase_value != self.current_phase:
#             raise RuntimeError("phase value does not match active phase")
#         if set(int(key) for key in new_rows) != set(self.new_classes):
#             raise RuntimeError("new row IDs do not match active new classes")
#         transported, transport_report = self.build_transported_old_rows()
#         result = self.geometry_bank.commit_incremental_phase(
#             transported_old_rows=transported,
#             new_rows=new_rows,
#             old_class_ids=self.old_classes,
#             new_class_ids=self.new_classes,
#             phase=phase_value,
#             expected_old_digest=self.phase_old_digest,
#         )
#         if self.classifier.bound_bank_contract_digest != (
#             self.geometry_bank.contract_digest()
#         ):
#             raise RuntimeError(
#                 "row evolution unexpectedly changed the immutable bank contract"
#             )
#         self.seen_classes = [*self.old_classes, *self.new_classes]
#         self.old_classes = list(self.seen_classes)
#         self.new_classes = []
#         self.phase_old_digest = None
#         self.phase_transport = None
#         self.phase_mode = "evaluation"
#         self.backbone.freeze_all()
#         result["transport_report"] = transport_report
#         result["factorization"] = "p(s|c)p(z|s,c)"
#         result["transport_absorbed"] = True
#         return result

#     def abort_incremental_phase(self) -> None:
#         """Discard temporary transport; trainer must restore encoder checkpoint."""
#         self.phase_transport = None
#         self.phase_old_digest = None
#         self.new_classes = []
#         self.seen_classes = self.infer_seen_classes()
#         self.old_classes = list(self.seen_classes)
#         self.phase_mode = "evaluation"
#         self.backbone.freeze_all()

#     # ------------------------------------------------------------------
#     # Geometry wrappers and validation
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def update_geometry_statistics(
#         self,
#         *,
#         features: Tensor,
#         raw_center_spectra: Tensor,
#         labels: Tensor,
#         class_ids: Sequence[int],
#     ) -> Dict[str, float]:
#         return self.geometry_bank.update_statistics(
#             features,
#             labels,
#             class_ids,
#             raw_spectra=raw_center_spectra,
#         )

#     @torch.no_grad()
#     def geometry_admission_report(
#         self,
#         class_ids: Optional[Sequence[int]] = None,
#         *,
#         minimum_effective_dimension: float = 1.25,
#         require_statistics: bool = True,
#     ) -> Dict[str, Any]:
#         ids = (
#             self.infer_seen_classes()
#             if class_ids is None
#             else _unique_ids(class_ids, name="class_ids")
#         )
#         return self.geometry_bank.admission_report(
#             ids,
#             minimum_effective_dimension=minimum_effective_dimension,
#             require_statistics=require_statistics,
#         )

#     @torch.no_grad()
#     def memory_snapshot(self) -> Dict[str, Any]:
#         return {
#             "bank": self.geometry_bank.export_snapshot(),
#             "bank_contract_digest": (
#                 self.geometry_bank.contract_digest()
#                 if bool(self.geometry_bank.anchor_ready.item())
#                 else None
#             ),
#             "rows_digest": (
#                 self.geometry_bank.rows_digest(self.infer_seen_classes())
#                 if self.infer_seen_classes()
#                 else None
#             ),
#             "phase": self.current_phase,
#             "seen_classes": list(self.seen_classes),
#             "factorization": "p(s|c)p(z|s,c)",
#         }
