from __future__ import annotations

"""Losses for transport-verified spectral-spatial factor geometry.

Deployed class model
--------------------
    p(z | c) = N(mu_c, L_c L_c^T + Psi_c)

where
    z = [z_s ; z_p]
    Psi_c = diag(psi_c,s I_Ds, psi_c,p I_Dp).

The raw ordered-spectrum model p(h | c) is NOT a second classifier. It is used
only by the GeometryBank to produce a detached pair-risk matrix R[c, j], which
modulates the training margin between class pairs.

Official optimization routes
----------------------------
Base warm-up:
    temporary linear-head cross entropy on real base samples.

Base geometry stage:
    risk-guided all-rival factor-energy loss on cross-fitted query samples.

Incremental stage:
    risk-guided all-rival factor-energy loss on real current query samples
    + branchwise coordinate transport consistency on disjoint current queries
    + optional relative parameter trust on controlled-plasticity parameters.

There is no old-feature replay loss, trainable generic affine transport,
spectral-anchor classifier loss, old/new logit offset, class-specific bias,
knowledge distillation, or persistent pseudo-sample objective in this file.
"""

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
_EPS = 1e-12

__all__ = [
    "global_to_local_targets",
    "local_to_global_targets",
    "class_balanced_cross_entropy",
    "risk_guided_all_rival_geometry_loss",
    "base_ce_warmup_objective",
    "base_geometry_objective",
    "base_training_objective",
    "coordinate_transport_loss",
    "snapshot_trainable_parameters",
    "relative_parameter_trust_loss",
    "incremental_geometry_objective",
    "pairwise_directional_invasion_matrix",
    "pair_risk_confusion_statistics",
    # Explicit migration aliases.
    "joint_energy_margin_loss",
    "base_geometry_loss",
]


# =============================================================================
# Validation and reductions
# =============================================================================


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _nonnegative(value: float, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _positive(value: float, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _matrix(
    value: Tensor,
    *,
    name: str,
    minimum_columns: int = 2,
) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.dim() != 2 or value.size(0) == 0:
        raise ValueError(f"{name} must be non-empty [N,C]")
    if value.size(1) < int(minimum_columns):
        raise ValueError(
            f"{name} must contain at least {minimum_columns} columns"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    return value


def _feature_matrix(value: Tensor, *, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.dim() != 2 or value.size(0) == 0 or value.size(1) == 0:
        raise ValueError(f"{name} must be non-empty [N,D]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    return value


def _class_ids(
    values: Union[Sequence[int], Tensor],
    *,
    device: torch.device,
    name: str = "class_ids",
) -> Tensor:
    result = torch.as_tensor(values, device=device, dtype=torch.long).flatten()
    if result.numel() == 0:
        raise ValueError(f"{name} is empty")
    if bool(result.lt(0).any()):
        bad = result[result.lt(0)].detach().cpu().unique().tolist()
        raise ValueError(f"{name} contains negative IDs: {bad}")
    if result.unique().numel() != result.numel():
        raise ValueError(f"{name} contains duplicate class IDs")
    return result


def global_to_local_targets(
    targets_global: Tensor,
    class_ids: Union[Sequence[int], Tensor],
) -> Tensor:
    """Map arbitrary global class IDs to exact energy-column indices."""
    if not torch.is_tensor(targets_global):
        raise TypeError("targets_global must be a tensor")
    targets = targets_global.long().flatten()
    classes = _class_ids(
        class_ids,
        device=targets.device,
    )
    matches = targets[:, None].eq(classes[None, :])
    match_count = matches.sum(dim=1)
    if bool(match_count.ne(1).any()):
        missing = (
            targets[match_count.eq(0)]
            .detach()
            .cpu()
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"targets contain classes outside class_ids: {missing}"
        )
    return matches.to(torch.long).argmax(dim=1)


def local_to_global_targets(
    targets_local: Tensor,
    class_ids: Union[Sequence[int], Tensor],
) -> Tensor:
    if not torch.is_tensor(targets_local):
        raise TypeError("targets_local must be a tensor")
    local = targets_local.long().flatten()
    classes = _class_ids(
        class_ids,
        device=local.device,
    )
    if local.numel() and (
        int(local.min().item()) < 0
        or int(local.max().item()) >= classes.numel()
    ):
        raise RuntimeError("targets_local is outside the class-column range")
    return classes.index_select(0, local)


def _sample_weights(
    value: Optional[Tensor],
    *,
    sample_count: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "sample_weights",
) -> Tensor:
    if value is None:
        return torch.ones(sample_count, device=device, dtype=dtype)
    result = torch.as_tensor(value, device=device, dtype=dtype).flatten()
    if result.numel() != int(sample_count):
        raise ValueError(
            f"{name} has {result.numel()} entries; expected {sample_count}"
        )
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if bool(result.lt(0.0).any()):
        raise ValueError(f"{name} must be non-negative")
    if float(result.sum().detach().item()) <= 0.0:
        raise ValueError(f"{name} sums to zero")
    return result


def _valid_class_mask(
    value: Optional[Tensor],
    *,
    class_count: int,
    device: torch.device,
) -> Tensor:
    if value is None:
        return torch.ones(class_count, device=device, dtype=torch.bool)
    result = torch.as_tensor(value, device=device, dtype=torch.bool).flatten()
    if result.numel() != int(class_count):
        raise ValueError(
            f"valid_class_mask must contain C={class_count} values"
        )
    if not bool(result.any()):
        raise ValueError("valid_class_mask selects no class")
    return result


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(_EPS)


def _class_balanced_mean(
    values: Tensor,
    class_labels: Tensor,
    weights: Tensor,
) -> Tensor:
    terms: List[Tensor] = []
    for class_id in torch.unique(class_labels, sorted=True):
        selected = class_labels.eq(class_id)
        selected_weights = weights[selected]
        if float(selected_weights.sum().detach().item()) <= 0.0:
            raise RuntimeError(
                f"class {int(class_id.item())} has zero effective weight"
            )
        terms.append(
            _weighted_mean(values[selected], selected_weights)
        )
    if not terms:
        raise RuntimeError("class-balanced reduction received no samples")
    return torch.stack(terms).mean()


def _reduce(
    values: Tensor,
    class_labels: Tensor,
    weights: Tensor,
    *,
    class_balanced: bool,
) -> Tensor:
    if class_balanced:
        return _class_balanced_mean(values, class_labels, weights)
    return _weighted_mean(values, weights)


# =============================================================================
# Base warm-up cross entropy
# =============================================================================


def class_balanced_cross_entropy(
    logits: Tensor,
    targets_global: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    *,
    sample_weights: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
    class_balanced: bool = True,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Cross entropy using explicit global-ID to column mapping.

    Query purity should not normally be passed as ``sample_weights``. Spatial
    purity is an estimator weight for support-row fitting, not a mechanism for
    suppressing difficult query gradients.
    """
    scores = _matrix(logits, name="logits")
    classes = _class_ids(class_ids, device=scores.device)
    if scores.size(1) != classes.numel():
        raise ValueError(
            "logit width does not match class_ids length"
        )
    targets = targets_global.to(scores.device).long().flatten()
    if targets.numel() != scores.size(0):
        raise ValueError("target/logit batch mismatch")
    local = global_to_local_targets(targets, classes)

    weights = _sample_weights(
        sample_weights,
        sample_count=scores.size(0),
        device=scores.device,
        dtype=scores.dtype,
    )
    smoothing = _nonnegative(
        label_smoothing,
        name="label_smoothing",
    )
    if smoothing >= 1.0:
        raise ValueError("label_smoothing must be smaller than one")

    per_sample = F.cross_entropy(
        scores,
        local,
        reduction="none",
        label_smoothing=smoothing,
    )
    total = _reduce(
        per_sample,
        targets,
        weights,
        class_balanced=class_balanced,
    )
    if total.dim() != 0 or not torch.isfinite(total):
        raise RuntimeError("cross entropy is not a finite scalar")

    if not return_parts:
        return total
    return {
        "total": total,
        "per_sample": per_sample,
        "targets_local": local,
        "accuracy": scores.argmax(dim=1).eq(local).float().mean().detach(),
    }


# =============================================================================
# Risk-guided all-rival factor-energy objective
# =============================================================================


def _validate_pair_risk(
    pair_risk: Optional[Tensor],
    *,
    class_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if pair_risk is None:
        return torch.zeros(
            (class_count, class_count),
            device=device,
            dtype=dtype,
        )

    risk = torch.as_tensor(
        pair_risk,
        device=device,
        dtype=dtype,
    )
    if risk.shape != (class_count, class_count):
        raise ValueError(
            f"pair_risk must be [{class_count},{class_count}], "
            f"got {tuple(risk.shape)}"
        )
    if not torch.isfinite(risk).all():
        raise RuntimeError("pair_risk contains NaN/Inf")
    if bool(risk.lt(-1e-6).any()) or bool(risk.gt(1.0 + 1e-6).any()):
        raise ValueError("pair_risk values must lie in [0,1]")
    risk = risk.clamp(0.0, 1.0)

    # The intended risk is a class-pair relation, not a directional logit
    # correction. Reject accidental asymmetric matrices instead of silently
    # introducing class-order bias.
    if not torch.allclose(
        risk,
        risk.transpose(0, 1),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("pair_risk must be symmetric")
    if not torch.allclose(
        torch.diagonal(risk),
        torch.zeros(class_count, device=device, dtype=dtype),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError("pair_risk diagonal must be zero")
    return risk


def risk_guided_all_rival_geometry_loss(
    energy: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    targets_global: Tensor,
    *,
    pair_risk: Optional[Tensor] = None,
    base_margin: float = 0.50,
    risk_strength: float = 0.50,
    temperature: float = 0.50,
    maximum_margin: Optional[float] = None,
    sample_weights: Optional[Tensor] = None,
    valid_class_mask: Optional[Tensor] = None,
    class_balanced: bool = True,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Risk-guided all-rival loss on lower-is-better factor energy.

    For query i with global target y_i and rival j:

        m_{y_i,j} = m_0 (1 + kappa R_{y_i,j})

        l_i = tau log[
            1 + sum_{j != y_i}
            exp((m_{y_i,j} + E_{i,y_i} - E_{i,j}) / tau)
        ].

    The pair-risk matrix must follow the exact ``class_ids`` column order.
    It is detached internally so spectral-shape and bank statistics are not
    optimized through the query objective.
    """
    scores = _matrix(energy, name="energy")
    n, c = scores.shape
    classes = _class_ids(class_ids, device=scores.device)
    if classes.numel() != c:
        raise ValueError(
            "energy width does not match class_ids length"
        )

    targets = targets_global.to(scores.device).long().flatten()
    if targets.numel() != n:
        raise ValueError("target/energy batch mismatch")
    local = global_to_local_targets(targets, classes)

    valid = _valid_class_mask(
        valid_class_mask,
        class_count=c,
        device=scores.device,
    )
    if not bool(valid.index_select(0, local).all()):
        raise RuntimeError("targets reference invalid class columns")

    m0 = _nonnegative(base_margin, name="base_margin")
    kappa = _nonnegative(risk_strength, name="risk_strength")
    tau = _positive(temperature, name="temperature")
    max_margin = (
        None
        if maximum_margin is None
        else _positive(maximum_margin, name="maximum_margin")
    )
    if max_margin is not None and max_margin < m0:
        raise ValueError("maximum_margin must be >= base_margin")

    risk = _validate_pair_risk(
        pair_risk,
        class_count=c,
        device=scores.device,
        dtype=scores.dtype,
    ).detach()
    margin_matrix = m0 * (1.0 + kappa * risk)
    if max_margin is not None:
        margin_matrix = margin_matrix.clamp_max(max_margin)
    margin_matrix = margin_matrix.clone()
    margin_matrix.fill_diagonal_(0.0)

    weights = _sample_weights(
        sample_weights,
        sample_count=n,
        device=scores.device,
        dtype=scores.dtype,
    )

    true_energy = scores.gather(1, local[:, None]).squeeze(1)
    pair_margins = margin_matrix.index_select(0, local)

    rivals_allowed = valid.view(1, c).expand(n, c).clone()
    rivals_allowed.scatter_(1, local[:, None], False)
    if not bool(rivals_allowed.any(dim=1).all()):
        raise RuntimeError("one or more queries have no valid rival")

    scaled_violation = (
        pair_margins
        + true_energy[:, None]
        - scores
    ) / tau
    scaled_violation = scaled_violation.masked_fill(
        ~rivals_allowed,
        float("-inf"),
    )
    zero = torch.zeros(
        (n, 1),
        device=scores.device,
        dtype=scores.dtype,
    )
    per_sample = tau * torch.logsumexp(
        torch.cat([zero, scaled_violation], dim=1),
        dim=1,
    )

    total = _reduce(
        per_sample,
        targets,
        weights,
        class_balanced=class_balanced,
    )
    if total.dim() != 0 or not torch.isfinite(total):
        raise RuntimeError(
            "risk-guided geometry loss is not a finite scalar"
        )

    masked_rivals = scores.masked_fill(
        ~rivals_allowed,
        float("inf"),
    )
    nearest_rival_energy, nearest_rival_local = masked_rivals.min(dim=1)
    nearest_gap = nearest_rival_energy - true_energy
    nearest_required_margin = pair_margins.gather(
        1,
        nearest_rival_local[:, None],
    ).squeeze(1)

    margin_violation = _reduce(
        nearest_gap.lt(nearest_required_margin).to(scores.dtype),
        targets,
        weights,
        class_balanced=class_balanced,
    )
    classification_violation = _reduce(
        nearest_gap.le(0.0).to(scores.dtype),
        targets,
        weights,
        class_balanced=class_balanced,
    )

    predicted_local = scores.argmin(dim=1)
    predicted_global = classes.index_select(0, predicted_local)
    nearest_rival_global = classes.index_select(
        0,
        nearest_rival_local,
    )

    if not return_parts:
        return total
    return {
        "total": total,
        "per_sample": per_sample,
        "targets_local": local,
        "true_energy": true_energy,
        "nearest_rival_energy": nearest_rival_energy,
        "nearest_rival_local": nearest_rival_local.detach(),
        "nearest_rival_global": nearest_rival_global.detach(),
        "nearest_required_margin": nearest_required_margin.detach(),
        "nearest_pair_risk": risk[
            local,
            nearest_rival_local,
        ].detach(),
        "nearest_gap": nearest_gap,
        "mean_gap": _reduce(
            nearest_gap,
            targets,
            weights,
            class_balanced=class_balanced,
        ).detach(),
        "minimum_gap": nearest_gap.min().detach(),
        "q01_gap": torch.quantile(nearest_gap.detach(), 0.01),
        "q05_gap": torch.quantile(nearest_gap.detach(), 0.05),
        "margin_violation_rate": margin_violation.detach(),
        "classification_violation_rate":
            classification_violation.detach(),
        "accuracy": predicted_global.eq(targets).float().mean().detach(),
        "margin_matrix": margin_matrix.detach(),
        "pair_risk": risk.detach(),
    }


# =============================================================================
# Base objectives
# =============================================================================


def base_ce_warmup_objective(
    logits: Tensor,
    targets_global: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    *,
    label_smoothing: float = 0.0,
    class_balanced: bool = True,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Temporary base-head objective before geometry shaping."""
    return class_balanced_cross_entropy(
        logits,
        targets_global,
        class_ids,
        label_smoothing=label_smoothing,
        class_balanced=class_balanced,
        return_parts=return_parts,
    )


def base_geometry_objective(
    query_energy: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    query_targets_global: Tensor,
    *,
    pair_risk: Optional[Tensor] = None,
    base_margin: float = 0.50,
    risk_strength: float = 0.50,
    temperature: float = 0.50,
    maximum_margin: Optional[float] = None,
    class_balanced: bool = True,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Cross-fitted base geometry objective on query samples.

    Support purity belongs in detached support-row fitting. Query samples are
    intentionally unweighted so difficult boundary pixels retain full gradient.
    """
    return risk_guided_all_rival_geometry_loss(
        query_energy,
        class_ids,
        query_targets_global,
        pair_risk=pair_risk,
        base_margin=base_margin,
        risk_strength=risk_strength,
        temperature=temperature,
        maximum_margin=maximum_margin,
        sample_weights=None,
        class_balanced=class_balanced,
        return_parts=return_parts,
    )


def base_training_objective(
    base_logits: Tensor,
    base_targets_global: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    *,
    query_energy: Optional[Tensor] = None,
    query_targets_global: Optional[Tensor] = None,
    pair_risk: Optional[Tensor] = None,
    geometry_weight: float = 0.0,
    base_margin: float = 0.50,
    risk_strength: float = 0.50,
    geometry_temperature: float = 0.50,
    maximum_margin: Optional[float] = None,
    label_smoothing: float = 0.0,
    class_balanced: bool = True,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Base objective with an explicit geometry-ramp weight.

    During warm-up, set ``geometry_weight=0`` and omit query energy.
    During geometry shaping, provide cross-fitted query energy and ramp
    ``geometry_weight`` from a small value toward one.
    """
    if (query_energy is None) != (query_targets_global is None):
        raise ValueError(
            "query_energy and query_targets_global must be provided together"
        )
    geometry_gain = _nonnegative(
        geometry_weight,
        name="geometry_weight",
    )
    if geometry_gain > 0.0 and query_energy is None:
        raise ValueError(
            "query_energy is required when geometry_weight > 0"
        )

    ce = class_balanced_cross_entropy(
        base_logits,
        base_targets_global,
        class_ids,
        label_smoothing=label_smoothing,
        class_balanced=class_balanced,
        return_parts=True,
    )
    assert isinstance(ce, dict)

    geometry_total = base_logits.sum() * 0.0
    geometry_parts: Optional[Dict[str, Tensor]] = None
    if query_energy is not None:
        geometry_parts = base_geometry_objective(
            query_energy,
            class_ids,
            query_targets_global,
            pair_risk=pair_risk,
            base_margin=base_margin,
            risk_strength=risk_strength,
            temperature=geometry_temperature,
            maximum_margin=maximum_margin,
            class_balanced=class_balanced,
            return_parts=True,
        )
        assert isinstance(geometry_parts, dict)
        geometry_total = geometry_parts["total"]

    total = ce["total"] + geometry_gain * geometry_total
    if total.dim() != 0 or not torch.isfinite(total):
        raise RuntimeError("base training objective is not finite")

    if not return_parts:
        return total
    zero = total.detach() * 0.0
    return {
        "total": total,
        "ce": ce["total"],
        "geometry": geometry_total,
        "geometry_weight": total.new_tensor(geometry_gain),
        "ce_accuracy": ce["accuracy"],
        "geometry_accuracy": (
            geometry_parts["accuracy"]
            if geometry_parts is not None
            else zero
        ),
        "geometry_mean_gap": (
            geometry_parts["mean_gap"]
            if geometry_parts is not None
            else zero
        ),
        "geometry_q05_gap": (
            geometry_parts["q05_gap"]
            if geometry_parts is not None
            else zero
        ),
        "classification_violation_rate": (
            geometry_parts["classification_violation_rate"]
            if geometry_parts is not None
            else zero
        ),
        "margin_violation_rate": (
            geometry_parts["margin_violation_rate"]
            if geometry_parts is not None
            else zero
        ),
    }


# =============================================================================
# Coordinate transport consistency
# =============================================================================


def _transform_field(
    transform: Any,
    name: str,
) -> Any:
    if not hasattr(transform, name):
        raise TypeError(
            f"transform lacks required field {name!r}"
        )
    return getattr(transform, name)


def _validate_branch_transform(
    transform: Any,
    *,
    spectral_dim: int,
    spatial_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Tensor]:
    spectral_rotation = torch.as_tensor(
        _transform_field(transform, "spectral_rotation"),
        device=device,
        dtype=dtype,
    )
    spatial_rotation = torch.as_tensor(
        _transform_field(transform, "spatial_rotation"),
        device=device,
        dtype=dtype,
    )
    spectral_bias = torch.as_tensor(
        _transform_field(transform, "spectral_bias"),
        device=device,
        dtype=dtype,
    ).flatten()
    spatial_bias = torch.as_tensor(
        _transform_field(transform, "spatial_bias"),
        device=device,
        dtype=dtype,
    ).flatten()
    spectral_scale = _positive(
        float(_transform_field(transform, "spectral_scale")),
        name="spectral_scale",
    )
    spatial_scale = _positive(
        float(_transform_field(transform, "spatial_scale")),
        name="spatial_scale",
    )

    expected = {
        "spectral_rotation":
            (spectral_rotation, (spectral_dim, spectral_dim)),
        "spatial_rotation":
            (spatial_rotation, (spatial_dim, spatial_dim)),
        "spectral_bias":
            (spectral_bias, (spectral_dim,)),
        "spatial_bias":
            (spatial_bias, (spatial_dim,)),
    }
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != {shape}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN/Inf")

    eye_s = torch.eye(
        spectral_dim,
        device=device,
        dtype=dtype,
    )
    eye_p = torch.eye(
        spatial_dim,
        device=device,
        dtype=dtype,
    )
    spectral_orthogonality = (
        spectral_rotation.transpose(0, 1)
        @ spectral_rotation
        - eye_s
    ).norm()
    spatial_orthogonality = (
        spatial_rotation.transpose(0, 1)
        @ spatial_rotation
        - eye_p
    ).norm()
    if float(spectral_orthogonality.detach().item()) > 1e-3:
        raise RuntimeError(
            "spectral rotation is not sufficiently orthogonal"
        )
    if float(spatial_orthogonality.detach().item()) > 1e-3:
        raise RuntimeError(
            "spatial rotation is not sufficiently orthogonal"
        )

    # The accepted transform is a closed-form support estimate, not a trainable
    # network component. Detach every field before constructing query targets.
    return {
        "spectral_rotation": spectral_rotation.detach(),
        "spatial_rotation": spatial_rotation.detach(),
        "spectral_bias": spectral_bias.detach(),
        "spatial_bias": spatial_bias.detach(),
        "spectral_scale": torch.tensor(
            spectral_scale,
            device=device,
            dtype=dtype,
        ),
        "spatial_scale": torch.tensor(
            spatial_scale,
            device=device,
            dtype=dtype,
        ),
        "spectral_orthogonality_error":
            spectral_orthogonality.detach(),
        "spatial_orthogonality_error":
            spatial_orthogonality.detach(),
    }


def coordinate_transport_loss(
    previous_joint_features: Tensor,
    current_joint_features: Tensor,
    *,
    spectral_dim: int,
    transform: Any,
    class_targets_global: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    class_balanced: bool = True,
    variance_floor: float = 1e-6,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Keep current query coordinates near the accepted support transform.

    The accepted transform is fitted on support folds and detached. For query
    feature z^- from the frozen phase-start observer and current feature z^+:

        z_s,target = a_s z_s^- R_s^T + b_s
        z_p,target = a_p z_p^- R_p^T + b_p

        L_coord =
            ||z_s^+ - z_s,target||^2 /
            [D_s (Var(z_s^+) + eps)]
          + ||z_p^+ - z_p,target||^2 /
            [D_p (Var(z_p^+) + eps)].

    Gradients flow only through ``current_joint_features``.
    """
    previous = _feature_matrix(
        previous_joint_features,
        name="previous_joint_features",
    )
    current = _feature_matrix(
        current_joint_features,
        name="current_joint_features",
    )
    if previous.shape != current.shape:
        raise ValueError(
            "previous and current joint features must have identical shape"
        )

    d = current.size(1)
    ds = int(spectral_dim)
    if not 0 < ds < d:
        raise ValueError("spectral_dim must lie inside joint feature width")
    dp = d - ds

    accepted = _validate_branch_transform(
        transform,
        spectral_dim=ds,
        spatial_dim=dp,
        device=current.device,
        dtype=current.dtype,
    )

    previous_detached = previous.detach()
    previous_s = previous_detached[:, :ds]
    previous_p = previous_detached[:, ds:]
    current_s = current[:, :ds]
    current_p = current[:, ds:]

    target_s = (
        previous_s
        @ (
            accepted["spectral_scale"]
            * accepted["spectral_rotation"]
        ).transpose(0, 1)
        + accepted["spectral_bias"].view(1, -1)
    )
    target_p = (
        previous_p
        @ (
            accepted["spatial_scale"]
            * accepted["spatial_rotation"]
        ).transpose(0, 1)
        + accepted["spatial_bias"].view(1, -1)
    )

    floor = _positive(
        variance_floor,
        name="variance_floor",
    )
    # Detach normalizers to prevent the encoder from reducing the objective by
    # inflating branch variance.
    variance_s = current_s.detach().var(
        dim=0,
        unbiased=False,
    ).mean().clamp_min(floor)
    variance_p = current_p.detach().var(
        dim=0,
        unbiased=False,
    ).mean().clamp_min(floor)

    spectral_sq = (
        current_s - target_s
    ).square().sum(dim=1) / (float(ds) * variance_s)
    spatial_sq = (
        current_p - target_p
    ).square().sum(dim=1) / (float(dp) * variance_p)
    per_sample = spectral_sq + spatial_sq

    weights = _sample_weights(
        sample_weights,
        sample_count=current.size(0),
        device=current.device,
        dtype=current.dtype,
    )
    if class_balanced:
        if class_targets_global is None:
            raise ValueError(
                "class_targets_global is required when class_balanced=True"
            )
        labels = class_targets_global.to(
            current.device,
        ).long().flatten()
        if labels.numel() != current.size(0):
            raise ValueError("class_targets_global length mismatch")
        if bool(labels.lt(0).any()):
            raise ValueError("class_targets_global contains negative IDs")
    else:
        labels = torch.zeros(
            current.size(0),
            device=current.device,
            dtype=torch.long,
        )

    total = _reduce(
        per_sample,
        labels,
        weights,
        class_balanced=class_balanced,
    )
    if total.dim() != 0 or not torch.isfinite(total):
        raise RuntimeError("coordinate transport loss is not finite")

    residual_s = (current_s.detach() - target_s).norm(dim=1)
    residual_p = (current_p.detach() - target_p).norm(dim=1)
    if not return_parts:
        return total
    return {
        "total": total,
        "per_sample": per_sample,
        "spectral_term": _reduce(
            spectral_sq,
            labels,
            weights,
            class_balanced=class_balanced,
        ),
        "spatial_term": _reduce(
            spatial_sq,
            labels,
            weights,
            class_balanced=class_balanced,
        ),
        "spectral_rmse": (
            current_s.detach() - target_s
        ).square().mean().sqrt(),
        "spatial_rmse": (
            current_p.detach() - target_p
        ).square().mean().sqrt(),
        "spectral_mean_l2": residual_s.mean(),
        "spatial_mean_l2": residual_p.mean(),
        "spectral_variance_normalizer": variance_s.detach(),
        "spatial_variance_normalizer": variance_p.detach(),
        "spectral_orthogonality_error":
            accepted["spectral_orthogonality_error"],
        "spatial_orthogonality_error":
            accepted["spatial_orthogonality_error"],
    }


# =============================================================================
# Controlled-plasticity parameter trust
# =============================================================================


@torch.no_grad()
def snapshot_trainable_parameters(
    module: nn.Module,
) -> Dict[str, Tensor]:
    """Clone only parameters that are trainable at phase start."""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not snapshot:
        raise RuntimeError(
            "module has no trainable parameters to snapshot"
        )
    return snapshot


def relative_parameter_trust_loss(
    module: nn.Module,
    reference_parameters: Mapping[str, Tensor],
    *,
    parameter_names: Optional[Iterable[str]] = None,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Relative Frobenius trust penalty for controlled late modules.

        L_param =
            sum_l ||W_l,t - W_l,t-1||_F^2 /
                  (||W_l,t-1||_F^2 + eps).

    This is a stabilizer, not the paper's contribution.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    if not isinstance(reference_parameters, Mapping):
        raise TypeError("reference_parameters must be a mapping")

    current = dict(module.named_parameters())
    selected_names = (
        [str(name) for name in parameter_names]
        if parameter_names is not None
        else [
            name
            for name, parameter in current.items()
            if parameter.requires_grad
        ]
    )
    if not selected_names:
        zero = next(module.parameters()).sum() * 0.0
        result = {
            "total": zero,
            "parameter_count": zero.detach(),
            "maximum_relative_term": zero.detach(),
        }
        return result if return_parts else zero

    missing_current = [
        name for name in selected_names
        if name not in current
    ]
    missing_reference = [
        name for name in selected_names
        if name not in reference_parameters
    ]
    if missing_current or missing_reference:
        raise RuntimeError(
            "parameter trust mapping mismatch: "
            f"missing_current={missing_current[:8]}, "
            f"missing_reference={missing_reference[:8]}"
        )

    terms: List[Tensor] = []
    for name in selected_names:
        parameter = current[name]
        reference = torch.as_tensor(
            reference_parameters[name],
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if reference.shape != parameter.shape:
            raise RuntimeError(
                f"reference shape mismatch for {name}: "
                f"{tuple(reference.shape)} vs {tuple(parameter.shape)}"
            )
        if not torch.isfinite(reference).all():
            raise RuntimeError(
                f"reference parameter {name} contains NaN/Inf"
            )
        numerator = (
            parameter - reference.detach()
        ).square().sum()
        denominator = reference.detach().square().sum().clamp_min(_EPS)
        terms.append(numerator / denominator)

    stacked = torch.stack(terms)
    total = stacked.sum()
    if total.dim() != 0 or not torch.isfinite(total):
        raise RuntimeError("parameter trust loss is not finite")

    if not return_parts:
        return total
    return {
        "total": total,
        "parameter_count": total.new_tensor(float(len(terms))),
        "mean_relative_term": stacked.mean().detach(),
        "maximum_relative_term": stacked.max().detach(),
    }


# =============================================================================
# Incremental objective
# =============================================================================


def incremental_geometry_objective(
    current_query_energy: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    current_targets_global: Tensor,
    *,
    pair_risk: Optional[Tensor] = None,
    coordinate_loss: Optional[Union[Tensor, Mapping[str, Tensor]]] = None,
    parameter_trust_loss: Optional[Union[Tensor, Mapping[str, Tensor]]] = None,
    geometry_weight: float = 1.0,
    coordinate_weight: float = 0.10,
    parameter_trust_weight: float = 0.0,
    base_margin: float = 0.50,
    risk_strength: float = 0.50,
    temperature: float = 0.50,
    maximum_margin: Optional[float] = None,
    class_balanced: bool = True,
    return_parts: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Compose the official incremental objective.

        L_inc =
            lambda_geom L_geom
          + lambda_coord L_coord
          + lambda_param L_param.

    ``L_geom`` uses only real current query samples scored against one
    provisional row set containing transported old rows and support-fitted new
    rows. No old feature replay is optimized here.
    """
    geometry = risk_guided_all_rival_geometry_loss(
        current_query_energy,
        class_ids,
        current_targets_global,
        pair_risk=pair_risk,
        base_margin=base_margin,
        risk_strength=risk_strength,
        temperature=temperature,
        maximum_margin=maximum_margin,
        sample_weights=None,
        class_balanced=class_balanced,
        return_parts=True,
    )
    assert isinstance(geometry, dict)

    def resolve(
        value: Optional[Union[Tensor, Mapping[str, Tensor]]],
        *,
        name: str,
        reference: Tensor,
    ) -> Tensor:
        if value is None:
            return reference.sum() * 0.0
        result = value["total"] if isinstance(value, Mapping) else value
        if not torch.is_tensor(result) or result.dim() != 0:
            raise TypeError(f"{name} must be a scalar tensor or mapping with total")
        if not torch.isfinite(result):
            raise RuntimeError(f"{name} is not finite")
        return result

    coordinate = resolve(
        coordinate_loss,
        name="coordinate_loss",
        reference=current_query_energy,
    )
    parameter = resolve(
        parameter_trust_loss,
        name="parameter_trust_loss",
        reference=current_query_energy,
    )

    geometry_gain = _nonnegative(
        geometry_weight,
        name="geometry_weight",
    )
    coordinate_gain = _nonnegative(
        coordinate_weight,
        name="coordinate_weight",
    )
    parameter_gain = _nonnegative(
        parameter_trust_weight,
        name="parameter_trust_weight",
    )
    if geometry_gain == 0.0 and coordinate_gain == 0.0:
        raise ValueError(
            "geometry_weight and coordinate_weight cannot both be zero"
        )

    total = (
        geometry_gain * geometry["total"]
        + coordinate_gain * coordinate
        + parameter_gain * parameter
    )
    if total.dim() != 0 or not torch.isfinite(total):
        raise RuntimeError("incremental geometry objective is not finite")

    if not return_parts:
        return total
    return {
        "total": total,
        "geometry": geometry["total"],
        "coordinate": coordinate,
        "parameter_trust": parameter,
        "geometry_weight": total.new_tensor(geometry_gain),
        "coordinate_weight": total.new_tensor(coordinate_gain),
        "parameter_trust_weight": total.new_tensor(parameter_gain),
        "accuracy": geometry["accuracy"],
        "mean_gap": geometry["mean_gap"],
        "q05_gap": geometry["q05_gap"],
        "classification_violation_rate":
            geometry["classification_violation_rate"],
        "margin_violation_rate":
            geometry["margin_violation_rate"],
        "mean_nearest_pair_risk":
            geometry["nearest_pair_risk"].mean().detach(),
        "mean_nearest_required_margin":
            geometry["nearest_required_margin"].mean().detach(),
    }


# =============================================================================
# Diagnostics
# =============================================================================


@torch.no_grad()
def pairwise_directional_invasion_matrix(
    energy: Tensor,
    class_ids: Union[Sequence[int], Tensor],
    targets_global: Tensor,
    *,
    valid_class_mask: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Ordered source-to-rival invasion rates and Q05 energy gaps."""
    scores = _matrix(energy, name="energy")
    n, c = scores.shape
    classes = _class_ids(class_ids, device=scores.device)
    if classes.numel() != c:
        raise ValueError("energy width does not match class_ids")

    targets = targets_global.to(scores.device).long().flatten()
    if targets.numel() != n:
        raise ValueError("target/energy batch mismatch")
    local = global_to_local_targets(targets, classes)

    valid = _valid_class_mask(
        valid_class_mask,
        class_count=c,
        device=scores.device,
    )
    if not bool(valid.index_select(0, local).all()):
        raise RuntimeError("targets reference invalid class columns")
    weights = _sample_weights(
        sample_weights,
        sample_count=n,
        device=scores.device,
        dtype=torch.float64,
    )

    invasion = torch.zeros(
        (c, c),
        device=scores.device,
        dtype=torch.float64,
    )
    q05 = torch.full_like(invasion, float("nan"))
    source_weight = torch.zeros(
        c,
        device=scores.device,
        dtype=torch.float64,
    )

    for source in range(c):
        if not bool(valid[source]):
            continue
        selected = local.eq(source)
        if not bool(selected.any()):
            continue
        local_weights = weights[selected]
        denominator = local_weights.sum().clamp_min(_EPS)
        source_weight[source] = denominator
        true = scores[selected, source]
        for rival in range(c):
            if rival == source or not bool(valid[rival]):
                continue
            gap = scores[selected, rival] - true
            invasion[source, rival] = (
                gap.le(0.0).double() * local_weights
            ).sum() / denominator
            q05[source, rival] = torch.quantile(
                gap.double(),
                0.05,
            )

    off_diagonal = (
        valid[:, None]
        & valid[None, :]
        & ~torch.eye(
            c,
            device=scores.device,
            dtype=torch.bool,
        )
    )
    values = invasion[off_diagonal]
    return {
        "class_ids": classes,
        "invasion_matrix": invasion,
        "gap_q05_matrix": q05,
        "source_weight": source_weight,
        "maximum_directional_invasion": (
            values.max()
            if values.numel()
            else invasion.new_tensor(0.0)
        ),
        "mean_directional_invasion": (
            values.mean()
            if values.numel()
            else invasion.new_tensor(0.0)
        ),
    }


@torch.no_grad()
def pair_risk_confusion_statistics(
    pair_risk: Tensor,
    invasion_matrix: Tensor,
) -> Dict[str, Tensor]:
    """Check whether proposed pair risk predicts observed pair confusion."""
    risk = torch.as_tensor(pair_risk).float()
    invasion = torch.as_tensor(
        invasion_matrix,
        device=risk.device,
        dtype=risk.dtype,
    )
    if risk.dim() != 2 or risk.size(0) != risk.size(1):
        raise ValueError("pair_risk must be square")
    if invasion.shape != risk.shape:
        raise ValueError(
            "invasion_matrix and pair_risk must have identical shape"
        )
    if not torch.isfinite(risk).all() or not torch.isfinite(invasion).all():
        raise RuntimeError("risk/invasion contains NaN/Inf")

    count = risk.size(0)
    mask = ~torch.eye(
        count,
        device=risk.device,
        dtype=torch.bool,
    )
    risk_values = risk[mask]
    invasion_values = invasion[mask]
    if risk_values.numel() < 2:
        correlation = risk.new_tensor(float("nan"))
    else:
        risk_centered = risk_values - risk_values.mean()
        invasion_centered = invasion_values - invasion_values.mean()
        denominator = (
            risk_centered.square().sum().sqrt()
            * invasion_centered.square().sum().sqrt()
        )
        correlation = (
            (risk_centered * invasion_centered).sum()
            / denominator.clamp_min(_EPS)
        )

    median = torch.median(risk_values)
    high = risk_values.ge(median)
    low = ~high
    return {
        "pearson_risk_invasion": correlation,
        "mean_high_risk_invasion": (
            invasion_values[high].mean()
            if bool(high.any())
            else invasion.new_tensor(float("nan"))
        ),
        "mean_low_risk_invasion": (
            invasion_values[low].mean()
            if bool(low.any())
            else invasion.new_tensor(float("nan"))
        ),
        "risk_median": median,
    }


# =============================================================================
# Explicit migration aliases
# =============================================================================


def joint_energy_margin_loss(
    energy: Tensor,
    targets_global: Tensor,
    *,
    class_ids: Union[Sequence[int], Tensor],
    pair_risk: Optional[Tensor] = None,
    margin: float = 0.50,
    risk_strength: float = 0.50,
    temperature: float = 0.50,
    sample_weights: Optional[Tensor] = None,
    valid_class_mask: Optional[Tensor] = None,
    class_balanced: bool = True,
    return_parts: bool = True,
    **retired_arguments: Any,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Compatibility name with the new factor-geometry contract.

    The wrapper intentionally requires explicit ``class_ids``. Retired
    spectral-conditioned or rival-restriction arguments are rejected.
    """
    if retired_arguments:
        raise RuntimeError(
            "retired joint-energy arguments were supplied: "
            f"{sorted(retired_arguments)}"
        )
    return risk_guided_all_rival_geometry_loss(
        energy,
        class_ids,
        targets_global,
        pair_risk=pair_risk,
        base_margin=margin,
        risk_strength=risk_strength,
        temperature=temperature,
        sample_weights=sample_weights,
        valid_class_mask=valid_class_mask,
        class_balanced=class_balanced,
        return_parts=return_parts,
    )


def base_geometry_loss(
    base_logits: Tensor,
    targets_global: Tensor,
    *,
    class_ids: Union[Sequence[int], Tensor],
    query_energy: Optional[Tensor] = None,
    query_targets_global: Optional[Tensor] = None,
    pair_risk: Optional[Tensor] = None,
    geometry_weight: float = 0.25,
    geometry_margin: float = 0.50,
    risk_strength: float = 0.50,
    temperature: float = 0.50,
    label_smoothing: float = 0.0,
    class_balanced: bool = True,
    return_parts: bool = True,
    **retired_arguments: Any,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Compatibility wrapper for CE plus cross-fitted factor geometry."""
    if retired_arguments:
        raise RuntimeError(
            "retired base-geometry arguments were supplied: "
            f"{sorted(retired_arguments)}"
        )
    return base_training_objective(
        base_logits,
        targets_global,
        class_ids,
        query_energy=query_energy,
        query_targets_global=query_targets_global,
        pair_risk=pair_risk,
        geometry_weight=geometry_weight,
        base_margin=geometry_margin,
        risk_strength=risk_strength,
        geometry_temperature=temperature,
        label_smoothing=label_smoothing,
        class_balanced=class_balanced,
        return_parts=return_parts,
    )


















# from __future__ import annotations

# """Losses for strict non-exemplar HSI class-incremental geometry learning.

# Deployed classifier contract
# ----------------------------
#     p(z, q | c) = p(z | c) p(q | z, c)

# The backbone/projection and committed old rows are frozen after phase 0.
# Incremental optimization is therefore allowed to update only bounded
# provisional new-class row corrections.  The main method deliberately uses only:

# Base:
#     L_base = L_CE + lambda_g L_joint(optional)

# Incremental:
#     L_inc = average(L_new, L_old_to_new) + lambda_tr L_trust(optional)

# `L_new` compares each real new-class sample against every seen rival.
# `L_old_to_new` compares aggregate old replay only against provisional new rows,
# which guarantees that replay produces a gradient for the trainable candidate
# rows instead of selecting another immutable old row as the nearest rival.
# """

# import math
# from typing import Dict, List, Mapping, Optional, Tuple, Union

# import torch
# import torch.nn.functional as F


# __all__ = [
#     "joint_energy_margin_loss",
#     "old_to_new_boundary_loss",
#     "candidate_trust_region_loss",
#     "base_geometry_loss",
#     "incremental_geometry_loss",
#     "pairwise_directional_invasion_matrix",
#     # Clear project-facing aliases.
#     "hsi_joint_energy_margin_loss",
#     "hsi_boundary_replay_loss",
#     "hsi_candidate_trust_region_loss",
#     "base_hsi_geometry_objective",
#     "incremental_hsi_geometry_objective",
#     # Narrow migration aliases used by older trainer imports.
#     "phase_consistent_conditional_joint_consolidation_loss",
#     "phase_consistent_spectral_geometry_consolidation_loss",
#     "pc_stgb_loss",
#     "pc_sgc_loss",
#     "candidate_descriptor_trust_region_loss",
# ]


# _DIRECT_FACTORIZATIONS = {
#     "p(z|c)p(q|z,c)",
#     "p(z,q|c)=p(z|c)p(q|z,c)",
# }


# def _finite(value: float, *, name: str) -> float:
#     result = float(value)
#     if not math.isfinite(result):
#         raise ValueError(f"{name} must be finite, got {value!r}")
#     return result


# def _nonnegative(value: float, *, name: str) -> float:
#     result = _finite(value, name=name)
#     if result < 0.0:
#         raise ValueError(f"{name} must be non-negative, got {value!r}")
#     return result


# def _positive(value: float, *, name: str) -> float:
#     result = _finite(value, name=name)
#     if result <= 0.0:
#         raise ValueError(f"{name} must be positive, got {value!r}")
#     return result


# def _validate_factorization(value: Optional[str]) -> None:
#     if value is None:
#         return
#     token = str(value).replace(" ", "")
#     if token not in _DIRECT_FACTORIZATIONS:
#         raise RuntimeError(
#             "loss expects direct HSI joint energy p(z|c)p(q|z,c); "
#             "finite-difference tangent/response energy is incompatible"
#         )


# def _energy(value: torch.Tensor, *, name: str) -> torch.Tensor:
#     if not torch.is_tensor(value):
#         raise TypeError(f"{name} must be a tensor")
#     if value.dim() != 2 or value.size(0) == 0 or value.size(1) < 2:
#         raise ValueError(
#             f"{name} must be non-empty [N,C] with C>=2, got {tuple(value.shape)}"
#         )
#     if not torch.is_floating_point(value):
#         raise TypeError(f"{name} must use a floating dtype")
#     if not torch.isfinite(value).all():
#         raise RuntimeError(f"{name} contains NaN/Inf")
#     return value


# def _targets(
#     value: torch.Tensor,
#     *,
#     sample_count: int,
#     class_count: int,
#     device: torch.device,
#     name: str,
# ) -> torch.Tensor:
#     if not torch.is_tensor(value):
#         raise TypeError(f"{name} must be a tensor")
#     result = value.to(device=device, dtype=torch.long).flatten()
#     if result.numel() != sample_count:
#         raise ValueError(
#             f"{name} has {result.numel()} entries; expected {sample_count}"
#         )
#     invalid = (result < 0) | (result >= class_count)
#     if bool(invalid.any().item()):
#         bad = torch.unique(result[invalid]).detach().cpu().tolist()
#         raise ValueError(
#             f"{name} contains seen-local columns outside [0,{class_count - 1}]: {bad}"
#         )
#     return result


# def _class_mask(
#     value: Optional[torch.Tensor],
#     *,
#     class_count: int,
#     device: torch.device,
#     name: str,
#     default: bool,
#     require_any: bool,
# ) -> torch.Tensor:
#     if value is None:
#         result = torch.full(
#             (class_count,), default, device=device, dtype=torch.bool
#         )
#     else:
#         result = torch.as_tensor(
#             value, device=device, dtype=torch.bool
#         ).flatten()
#     if result.numel() != class_count:
#         raise ValueError(f"{name} must contain C={class_count} values")
#     if require_any and not bool(result.any().item()):
#         raise ValueError(f"{name} must select at least one class")
#     return result


# def _weights(
#     value: Optional[torch.Tensor],
#     *,
#     sample_count: int,
#     device: torch.device,
#     dtype: torch.dtype,
#     name: str,
# ) -> torch.Tensor:
#     if value is None:
#         return torch.ones(sample_count, device=device, dtype=dtype)
#     result = torch.as_tensor(
#         value, device=device, dtype=dtype
#     ).flatten()
#     if result.numel() != sample_count:
#         raise ValueError(
#             f"{name} has {result.numel()} entries; expected {sample_count}"
#         )
#     if not torch.isfinite(result).all():
#         raise ValueError(f"{name} contains NaN/Inf")
#     if bool((result < 0.0).any().item()):
#         raise ValueError(f"{name} must be non-negative")
#     if float(result.sum().detach().item()) <= 0.0:
#         raise ValueError(f"{name} sums to zero")
#     return result


# def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#     eps = torch.finfo(values.dtype).eps
#     return (values * weights).sum() / weights.sum().clamp_min(eps)


# def _class_balanced_mean(
#     values: torch.Tensor,
#     targets: torch.Tensor,
#     weights: torch.Tensor,
# ) -> torch.Tensor:
#     terms: List[torch.Tensor] = []
#     for class_id in torch.unique(targets, sorted=True):
#         selected = targets.eq(class_id)
#         class_weights = weights[selected]
#         if float(class_weights.sum().detach().item()) <= 0.0:
#             raise RuntimeError(
#                 f"class {int(class_id.item())} has zero effective sample weight"
#             )
#         terms.append(_weighted_mean(values[selected], class_weights))
#     if not terms:
#         raise RuntimeError("class-balanced reduction received no samples")
#     return torch.stack(terms).mean()


# def _reduce(
#     values: torch.Tensor,
#     targets: torch.Tensor,
#     weights: torch.Tensor,
#     *,
#     class_balanced: bool,
# ) -> torch.Tensor:
#     if class_balanced:
#         return _class_balanced_mean(values, targets, weights)
#     return _weighted_mean(values, weights)


# def _scalar_optional(
#     value: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]],
#     *,
#     name: str,
#     reference: torch.Tensor,
# ) -> torch.Tensor:
#     if value is None:
#         return reference.sum() * 0.0
#     result = value.get("total") if isinstance(value, Mapping) else value
#     if not torch.is_tensor(result) or result.dim() != 0:
#         raise TypeError(f"{name} must be a scalar tensor or mapping with 'total'")
#     if not torch.isfinite(result):
#         raise RuntimeError(f"{name} is NaN/Inf")
#     return result


# def joint_energy_margin_loss(
#     joint_energy: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     margin: float = 0.50,
#     temperature: float = 0.15,
#     sample_weights: Optional[torch.Tensor] = None,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     rival_class_mask: Optional[torch.Tensor] = None,
#     rival_indices: Optional[torch.Tensor] = None,
#     class_balanced: bool = True,
#     joint_factorization: Optional[str] = "p(z|c)p(q|z,c)",
#     return_parts: bool = True,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Soft margin on the exact deployed joint energy.

#     `targets`, `rival_indices`, and masks use seen-local classifier columns.
#     Lower energy is better.  For each sample, the protected gap is

#         gap = E_rival - E_target.

#     The nearest allowed rival is selected unless `rival_indices` is supplied.
#     """
#     _validate_factorization(joint_factorization)
#     energy = _energy(joint_energy, name="joint_energy")
#     n, c = energy.shape
#     y = _targets(
#         targets,
#         sample_count=n,
#         class_count=c,
#         device=energy.device,
#         name="targets",
#     )
#     margin_value = _nonnegative(margin, name="margin")
#     tau = _positive(temperature, name="temperature")
#     weights = _weights(
#         sample_weights,
#         sample_count=n,
#         device=energy.device,
#         dtype=energy.dtype,
#         name="sample_weights",
#     )
#     valid = _class_mask(
#         valid_class_mask,
#         class_count=c,
#         device=energy.device,
#         name="valid_class_mask",
#         default=True,
#         require_any=True,
#     )
#     rivals_allowed = _class_mask(
#         rival_class_mask,
#         class_count=c,
#         device=energy.device,
#         name="rival_class_mask",
#         default=True,
#         require_any=True,
#     )
#     if not bool(valid.index_select(0, y).all().item()):
#         raise RuntimeError("targets reference invalid class columns")

#     allowed = (valid & rivals_allowed)[None, :].expand_as(energy).clone()
#     allowed.scatter_(1, y[:, None], False)
#     if not bool(allowed.any(dim=1).all().item()):
#         raise RuntimeError("one or more samples have no allowed rival class")

#     if rival_indices is None:
#         rival_energy, rivals = energy.masked_fill(
#             ~allowed, float("inf")
#         ).min(dim=1)
#     else:
#         rivals = _targets(
#             rival_indices,
#             sample_count=n,
#             class_count=c,
#             device=energy.device,
#             name="rival_indices",
#         )
#         chosen_allowed = allowed.gather(1, rivals[:, None]).squeeze(1)
#         if not bool(chosen_allowed.all().item()):
#             bad = torch.nonzero(
#                 ~chosen_allowed, as_tuple=False
#             ).flatten().detach().cpu().tolist()
#             raise ValueError(
#                 "rival_indices contain target or excluded columns; "
#                 f"bad sample indices={bad[:20]}"
#             )
#         rival_energy = energy.gather(1, rivals[:, None]).squeeze(1)

#     true_energy = energy.gather(1, y[:, None]).squeeze(1)
#     gap = rival_energy - true_energy
#     deficiency = margin_value - gap
#     per_sample = tau * F.softplus(deficiency / tau)
#     total = _reduce(
#         per_sample, y, weights, class_balanced=class_balanced
#     )
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("joint-energy margin loss is not a finite scalar")
#     if not return_parts:
#         return total

#     margin_violations = _reduce(
#         gap.lt(margin_value).to(energy.dtype),
#         y,
#         weights,
#         class_balanced=class_balanced,
#     )
#     classification_violations = _reduce(
#         gap.le(0.0).to(energy.dtype),
#         y,
#         weights,
#         class_balanced=class_balanced,
#     )
#     mean_gap = _reduce(
#         gap, y, weights, class_balanced=class_balanced
#     )
#     return {
#         "total": total,
#         "per_sample": per_sample,
#         "true_energy": true_energy,
#         "rival_energy": rival_energy,
#         "rival_indices": rivals.detach(),
#         "gap": gap,
#         "mean_gap": mean_gap.detach(),
#         "q05_gap": torch.quantile(gap.detach(), 0.05),
#         "minimum_gap": gap.min().detach(),
#         "margin_violation_rate": margin_violations.detach(),
#         "classification_violation_rate": classification_violations.detach(),
#         "accuracy": energy.argmin(dim=1).eq(y).float().mean().detach(),
#     }


# def old_to_new_boundary_loss(
#     old_joint_energy: torch.Tensor,
#     old_targets: torch.Tensor,
#     *,
#     new_class_mask: torch.Tensor,
#     margin: float = 0.25,
#     temperature: float = 0.15,
#     sample_weights: Optional[torch.Tensor] = None,
#     class_balanced: bool = True,
#     return_parts: bool = True,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Prevent provisional new rows from invading aggregate old replay.

#     Rivals are *restricted to new-class columns*.  This restriction is essential:
#     if another immutable old row were selected as the closest rival, the replay
#     loss could have no gradient with respect to the trainable candidate rows.
#     """
#     energy = _energy(old_joint_energy, name="old_joint_energy")
#     n, c = energy.shape
#     y = _targets(
#         old_targets,
#         sample_count=n,
#         class_count=c,
#         device=energy.device,
#         name="old_targets",
#     )
#     new_mask = _class_mask(
#         new_class_mask,
#         class_count=c,
#         device=energy.device,
#         name="new_class_mask",
#         default=False,
#         require_any=True,
#     )
#     if bool(new_mask.index_select(0, y).any().item()):
#         raise RuntimeError("old_targets include a new-class column")
#     result = joint_energy_margin_loss(
#         energy,
#         y,
#         margin=margin,
#         temperature=temperature,
#         sample_weights=sample_weights,
#         rival_class_mask=new_mask,
#         class_balanced=class_balanced,
#         return_parts=True,
#     )
#     assert isinstance(result, dict)
#     if not return_parts:
#         return result["total"]
#     return {
#         **result,
#         "old_to_new_invasion_rate": result[
#             "classification_violation_rate"
#         ],
#     }


# def candidate_trust_region_loss(
#     *,
#     feature_mean_deltas: Mapping[int, torch.Tensor],
#     feature_log_eigval_deltas: Mapping[int, torch.Tensor],
#     feature_log_residual_deltas: Mapping[int, torch.Tensor],
#     spectral_mean_deltas: Mapping[int, torch.Tensor],
#     spectral_log_eigval_deltas: Mapping[int, torch.Tensor],
#     spectral_log_residual_deltas: Mapping[int, torch.Tensor],
#     feature_mean_scale: float = 0.25,
#     feature_log_variance_scale: float = 0.15,
#     spectral_mean_scale: float = 0.25,
#     spectral_log_variance_scale: float = 0.15,
#     class_weights: Optional[Mapping[int, float]] = None,
#     return_parts: bool = True,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Prefer small candidate-row corrections inside the bank's hard bounds.

#     This objective is used only when candidate-row deltas are trainable.  The
#     GeometryBank correction limits remain the hard safety mechanism.
#     """
#     groups: Tuple[
#         Tuple[str, Mapping[int, torch.Tensor], float], ...
#     ] = (
#         ("feature_mean", feature_mean_deltas, _positive(
#             feature_mean_scale, name="feature_mean_scale"
#         )),
#         ("feature_log_eigval", feature_log_eigval_deltas, _positive(
#             feature_log_variance_scale, name="feature_log_variance_scale"
#         )),
#         ("feature_log_residual", feature_log_residual_deltas, _positive(
#             feature_log_variance_scale, name="feature_log_variance_scale"
#         )),
#         ("spectral_mean", spectral_mean_deltas, _positive(
#             spectral_mean_scale, name="spectral_mean_scale"
#         )),
#         ("spectral_log_eigval", spectral_log_eigval_deltas, _positive(
#             spectral_log_variance_scale, name="spectral_log_variance_scale"
#         )),
#         ("spectral_log_residual", spectral_log_residual_deltas, _positive(
#             spectral_log_variance_scale, name="spectral_log_variance_scale"
#         )),
#     )
#     key_sets = [set(int(key) for key in mapping) for _, mapping, _ in groups]
#     if not key_sets or not key_sets[0]:
#         raise ValueError("candidate delta mappings are empty")
#     expected = key_sets[0]
#     for (name, _, _), keys in zip(groups, key_sets):
#         if keys != expected:
#             raise RuntimeError(f"{name}_deltas keys do not match candidate classes")

#     reference: Optional[torch.Tensor] = None
#     for _, mapping, _ in groups:
#         for tensor in mapping.values():
#             if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
#                 raise TypeError("candidate deltas must be floating tensors")
#             if not torch.isfinite(tensor).all():
#                 raise RuntimeError("candidate deltas contain NaN/Inf")
#             reference = tensor if reference is None else reference
#     assert reference is not None
#     device, dtype = reference.device, reference.dtype

#     class_terms: List[torch.Tensor] = []
#     class_gains: List[torch.Tensor] = []
#     component_values: Dict[str, List[torch.Tensor]] = {
#         name: [] for name, _, _ in groups
#     }
#     for class_id in sorted(expected):
#         components: List[torch.Tensor] = []
#         for name, mapping, scale in groups:
#             tensor = mapping[class_id]
#             if tensor.device != device:
#                 raise RuntimeError("all candidate deltas must share one device")
#             normalized = tensor.to(dtype=dtype) / scale
#             term = (
#                 F.smooth_l1_loss(
#                     normalized,
#                     torch.zeros_like(normalized),
#                     reduction="mean",
#                     beta=1.0,
#                 )
#                 if normalized.numel()
#                 else normalized.sum() * 0.0
#             )
#             component_values[name].append(term)
#             components.append(term)
#         class_term = torch.stack(components).mean()
#         gain = 1.0 if class_weights is None else _nonnegative(
#             class_weights[class_id], name=f"class_weights[{class_id}]"
#         )
#         if gain <= 0.0:
#             continue
#         class_terms.append(class_term)
#         class_gains.append(torch.tensor(gain, device=device, dtype=dtype))

#     if not class_terms:
#         raise ValueError("all candidate class weights are zero")
#     terms = torch.stack(class_terms)
#     gains = torch.stack(class_gains)
#     total = _weighted_mean(terms, gains)
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("candidate trust-region loss is not finite")
#     if not return_parts:
#         return total
#     output: Dict[str, torch.Tensor] = {
#         "total": total,
#         "mean_class_penalty": terms.mean().detach(),
#         "maximum_class_penalty": terms.max().detach(),
#     }
#     for name, values in component_values.items():
#         output[f"{name}_penalty"] = torch.stack(values).mean().detach()
#     return output


# def base_geometry_loss(
#     base_logits: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     query_joint_energy: Optional[torch.Tensor] = None,
#     query_targets: Optional[torch.Tensor] = None,
#     ce_sample_weights: Optional[torch.Tensor] = None,
#     query_sample_weights: Optional[torch.Tensor] = None,
#     geometry_weight: float = 0.25,
#     geometry_margin: float = 0.50,
#     temperature: float = 0.20,
#     label_smoothing: float = 0.0,
#     class_balanced: bool = True,
#     return_parts: bool = True,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Temporary base CE plus optional held-out joint-energy preparation."""
#     if not torch.is_tensor(base_logits) or base_logits.dim() != 2:
#         raise ValueError("base_logits must be [N,C]")
#     if not torch.isfinite(base_logits).all():
#         raise RuntimeError("base_logits contain NaN/Inf")
#     n, c = base_logits.shape
#     y = _targets(
#         targets,
#         sample_count=n,
#         class_count=c,
#         device=base_logits.device,
#         name="targets",
#     )
#     weights = _weights(
#         ce_sample_weights,
#         sample_count=n,
#         device=base_logits.device,
#         dtype=base_logits.dtype,
#         name="ce_sample_weights",
#     )
#     smoothing = _nonnegative(label_smoothing, name="label_smoothing")
#     if smoothing >= 1.0:
#         raise ValueError("label_smoothing must be smaller than one")
#     ce_per_sample = F.cross_entropy(
#         base_logits,
#         y,
#         reduction="none",
#         label_smoothing=smoothing,
#     )
#     ce = _reduce(
#         ce_per_sample, y, weights, class_balanced=class_balanced
#     )

#     geometry_result: Optional[Dict[str, torch.Tensor]] = None
#     geometry = base_logits.sum() * 0.0
#     if query_joint_energy is not None or query_targets is not None:
#         if query_joint_energy is None or query_targets is None:
#             raise ValueError(
#                 "query_joint_energy and query_targets must be provided together"
#             )
#         geometry_result = joint_energy_margin_loss(
#             query_joint_energy,
#             query_targets,
#             margin=geometry_margin,
#             temperature=temperature,
#             sample_weights=query_sample_weights,
#             class_balanced=class_balanced,
#             return_parts=True,
#         )
#         assert isinstance(geometry_result, dict)
#         geometry = geometry_result["total"]

#     geometry_gain = _nonnegative(geometry_weight, name="geometry_weight")
#     total = ce + geometry_gain * geometry
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("base geometry loss is not finite")
#     if not return_parts:
#         return total
#     zero = total.detach() * 0.0
#     return {
#         "total": total,
#         "ce": ce,
#         "joint_geometry": geometry,
#         "ce_accuracy": base_logits.argmax(dim=1).eq(y).float().mean().detach(),
#         "joint_mean_gap": (
#             geometry_result["mean_gap"] if geometry_result is not None else zero
#         ),
#     }


# def incremental_geometry_loss(
#     new_joint_energy: torch.Tensor,
#     new_targets: torch.Tensor,
#     *,
#     old_boundary_joint_energy: torch.Tensor,
#     old_boundary_targets: torch.Tensor,
#     new_class_mask: torch.Tensor,
#     new_sample_weights: Optional[torch.Tensor] = None,
#     old_boundary_weights: Optional[torch.Tensor] = None,
#     trust_region: Optional[
#         Union[torch.Tensor, Mapping[str, torch.Tensor]]
#     ] = None,
#     new_margin: float = 0.50,
#     old_boundary_margin: float = 0.25,
#     temperature: float = 0.15,
#     new_weight: float = 1.0,
#     old_boundary_weight: float = 1.0,
#     trust_region_weight: float = 1e-3,
#     class_balanced: bool = True,
#     return_parts: bool = True,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Final incremental objective for candidate-row refinement.

#     The classification component is normalized by its active weights, keeping
#     its scale stable when the new/replay balance is changed.  The trust term is
#     added separately because it regularizes parameter displacement rather than
#     classification risk.
#     """
#     new_result = joint_energy_margin_loss(
#         new_joint_energy,
#         new_targets,
#         margin=new_margin,
#         temperature=temperature,
#         sample_weights=new_sample_weights,
#         class_balanced=class_balanced,
#         return_parts=True,
#     )
#     assert isinstance(new_result, dict)
#     old_result = old_to_new_boundary_loss(
#         old_boundary_joint_energy,
#         old_boundary_targets,
#         new_class_mask=new_class_mask,
#         margin=old_boundary_margin,
#         temperature=temperature,
#         sample_weights=old_boundary_weights,
#         class_balanced=class_balanced,
#         return_parts=True,
#     )
#     assert isinstance(old_result, dict)

#     new_gain = _positive(new_weight, name="new_weight")
#     replay_gain = _positive(
#         old_boundary_weight, name="old_boundary_weight"
#     )
#     classification = (
#         new_gain * new_result["total"]
#         + replay_gain * old_result["total"]
#     ) / (new_gain + replay_gain)

#     trust = _scalar_optional(
#         trust_region,
#         name="trust_region",
#         reference=classification,
#     )
#     trust_gain = _nonnegative(
#         trust_region_weight, name="trust_region_weight"
#     )
#     total = classification + trust_gain * trust
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("incremental geometry loss is not finite")
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "classification": classification,
#         "new_margin": new_result["total"],
#         "old_to_new_boundary": old_result["total"],
#         "trust_region": trust,
#         "new_mean_gap": new_result["mean_gap"],
#         "new_q05_gap": new_result["q05_gap"],
#         "old_to_new_mean_gap": old_result["mean_gap"],
#         "old_to_new_q05_gap": old_result["q05_gap"],
#         "new_classification_violation_rate": new_result[
#             "classification_violation_rate"
#         ],
#         "old_to_new_invasion_rate": old_result[
#             "old_to_new_invasion_rate"
#         ],
#     }


# @torch.no_grad()
# def pairwise_directional_invasion_matrix(
#     joint_energy: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     sample_weights: Optional[torch.Tensor] = None,
# ) -> Dict[str, torch.Tensor]:
#     """Diagnostic ordered source-to-rival invasion rates."""
#     energy = _energy(joint_energy, name="joint_energy")
#     n, c = energy.shape
#     y = _targets(
#         targets,
#         sample_count=n,
#         class_count=c,
#         device=energy.device,
#         name="targets",
#     )
#     valid = _class_mask(
#         valid_class_mask,
#         class_count=c,
#         device=energy.device,
#         name="valid_class_mask",
#         default=True,
#         require_any=True,
#     )
#     if not bool(valid.index_select(0, y).all().item()):
#         raise RuntimeError("targets reference invalid class columns")
#     weights = _weights(
#         sample_weights,
#         sample_count=n,
#         device=energy.device,
#         dtype=torch.float64,
#         name="sample_weights",
#     )
#     invasion = torch.zeros((c, c), device=energy.device, dtype=torch.float64)
#     q05 = torch.full_like(invasion, float("nan"))
#     source_weight = torch.zeros(c, device=energy.device, dtype=torch.float64)
#     for source in range(c):
#         if not bool(valid[source].item()):
#             continue
#         selected = y.eq(source)
#         if not bool(selected.any().item()):
#             continue
#         local_weights = weights[selected]
#         denominator = local_weights.sum().clamp_min(
#             torch.finfo(torch.float64).eps
#         )
#         source_weight[source] = denominator
#         true = energy[selected, source]
#         for rival in range(c):
#             if rival == source or not bool(valid[rival].item()):
#                 continue
#             gap = energy[selected, rival] - true
#             invasion[source, rival] = (
#                 gap.le(0.0).double() * local_weights
#             ).sum() / denominator
#             q05[source, rival] = torch.quantile(gap.double(), 0.05)
#     off_diagonal = (
#         valid[:, None]
#         & valid[None, :]
#         & ~torch.eye(c, device=energy.device, dtype=torch.bool)
#     )
#     values = invasion[off_diagonal]
#     return {
#         "invasion_matrix": invasion,
#         "gap_q05_matrix": q05,
#         "source_weight": source_weight,
#         "maximum_directional_invasion": (
#             values.max() if values.numel() else invasion.new_tensor(0.0)
#         ),
#         "mean_directional_invasion": (
#             values.mean() if values.numel() else invasion.new_tensor(0.0)
#         ),
#     }


# # Project-facing aliases.
# hsi_joint_energy_margin_loss = joint_energy_margin_loss
# hsi_boundary_replay_loss = old_to_new_boundary_loss
# hsi_candidate_trust_region_loss = candidate_trust_region_loss
# base_hsi_geometry_objective = base_geometry_loss
# incremental_hsi_geometry_objective = incremental_geometry_loss

# # Narrow migration aliases.  They all resolve to the direct-descriptor rule.
# phase_consistent_conditional_joint_consolidation_loss = joint_energy_margin_loss
# phase_consistent_spectral_geometry_consolidation_loss = joint_energy_margin_loss
# pc_stgb_loss = joint_energy_margin_loss
# pc_sgc_loss = joint_energy_margin_loss
# candidate_descriptor_trust_region_loss = candidate_trust_region_loss


# def _retired(name: str, replacement: str):
#     def fail(*_args, **_kwargs):
#         raise RuntimeError(f"{name} is retired. {replacement}")

#     fail.__name__ = name
#     return fail


# # These objectives duplicate the two exact margins or belong to the removed
# # tangent-response architecture.  Explicit failure is safer than silent reuse.
# hsi_two_sided_invasion_loss = _retired(
#     "hsi_two_sided_invasion_loss",
#     "new-sample all-rival margin plus old_to_new_boundary_loss already protect both directions",
# )
# two_sided_old_new_invasion_loss = hsi_two_sided_invasion_loss
# hsi_candidate_topology_barrier_loss = _retired(
#     "hsi_candidate_topology_barrier_loss",
#     "use GeometryBank pairwise_structure only as a diagnostic in the frozen-coordinate main method",
# )
# joint_energy_boundary_certificate_loss = _retired(
#     "joint_energy_boundary_certificate_loss",
#     "use aggregate boundary/risk-directed replay with old_to_new_boundary_loss",
# )
# spectral_tangent_step_consistency_loss = _retired(
#     "spectral_tangent_step_consistency_loss",
#     "the deployed method uses direct spectral descriptors, not finite-difference tangents",
# )





















# from __future__ import annotations

# from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# import math

# import torch
# import torch.nn.functional as F


# __all__ = [
#     "phase_consistent_conditional_joint_consolidation_loss",
#     "phase_consistent_spectral_geometry_consolidation_loss",
#     "pc_stgb_loss",
#     "pc_sgc_loss",
#     "joint_energy_boundary_certificate_loss",
#     "two_sided_old_new_invasion_loss",
#     "spectral_tangent_step_consistency_loss",
#     "candidate_descriptor_trust_region_loss",
#     "pairwise_directional_invasion_matrix",
# ]


# # =============================================================================
# # Shared validation and weighted reductions
# # =============================================================================


# def _require_energy_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
#     if not torch.is_tensor(value):
#         raise TypeError(f"{name} must be a tensor")
#     if value.dim() != 2 or value.size(0) == 0 or value.size(1) < 2:
#         raise ValueError(
#             f"{name} must be a non-empty [N,C] tensor with C>=2; "
#             f"got {tuple(value.shape)}"
#         )
#     if not torch.is_floating_point(value):
#         raise TypeError(f"{name} must use a floating dtype")
#     if not torch.isfinite(value).all():
#         bad = int((~torch.isfinite(value)).sum().detach().cpu().item())
#         raise RuntimeError(f"{name} contains {bad} NaN/Inf values")
#     return value


# def _require_targets(
#     value: torch.Tensor,
#     *,
#     sample_count: int,
#     class_count: int,
#     device: torch.device,
#     name: str = "targets",
# ) -> torch.Tensor:
#     if not torch.is_tensor(value):
#         raise TypeError(f"{name} must be a tensor")
#     targets = value.to(device=device, dtype=torch.long).flatten()
#     if targets.numel() != int(sample_count):
#         raise ValueError(
#             f"{name} contains {targets.numel()} values; expected {sample_count}"
#         )
#     if targets.numel() == 0:
#         raise ValueError(f"{name} is empty")
#     invalid = (targets < 0) | (targets >= int(class_count))
#     if bool(invalid.any().item()):
#         bad = torch.unique(targets[invalid]).detach().cpu().tolist()
#         raise ValueError(
#             f"{name} contains class columns outside [0,{class_count - 1}]: {bad}"
#         )
#     return targets


# def _require_nonnegative_finite(value: float, *, name: str) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0.0:
#         raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
#     return result


# def _require_positive_finite(value: float, *, name: str) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result <= 0.0:
#         raise ValueError(f"{name} must be finite and positive, got {value!r}")
#     return result


# def _require_valid_class_mask(
#     value: Optional[torch.Tensor],
#     *,
#     class_count: int,
#     device: torch.device,
# ) -> torch.Tensor:
#     if value is None:
#         return torch.ones(class_count, device=device, dtype=torch.bool)
#     mask = torch.as_tensor(value, device=device, dtype=torch.bool).flatten()
#     if mask.numel() != int(class_count):
#         raise ValueError(
#             f"valid_class_mask contains {mask.numel()} values; expected {class_count}"
#         )
#     if int(mask.sum().item()) < 2:
#         raise RuntimeError("at least two valid class columns are required")
#     return mask


# def _require_partition_mask(
#     value: torch.Tensor,
#     *,
#     class_count: int,
#     device: torch.device,
#     name: str,
# ) -> torch.Tensor:
#     mask = torch.as_tensor(value, device=device, dtype=torch.bool).flatten()
#     if mask.numel() != int(class_count):
#         raise ValueError(f"{name} must contain C={class_count} values")
#     if not bool(mask.any().item()):
#         raise ValueError(f"{name} must contain at least one class")
#     return mask


# def _require_sample_weights(
#     value: Optional[torch.Tensor],
#     *,
#     sample_count: int,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> torch.Tensor:
#     if value is None:
#         return torch.ones(sample_count, device=device, dtype=dtype)
#     weights = torch.as_tensor(value, device=device, dtype=dtype).flatten()
#     if weights.numel() != int(sample_count):
#         raise ValueError(
#             f"sample_weights contain {weights.numel()} values; expected {sample_count}"
#         )
#     if not torch.isfinite(weights).all():
#         raise ValueError("sample_weights contain NaN/Inf")
#     if bool((weights < 0.0).any().item()):
#         raise ValueError("sample_weights must be non-negative")
#     if float(weights.sum().detach().item()) <= 0.0:
#         raise ValueError("sample_weights sum to zero")
#     return weights


# def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#     epsilon = torch.finfo(values.dtype).eps
#     return (values * weights).sum() / weights.sum().clamp_min(epsilon)


# def _weighted_population_variance(
#     values: torch.Tensor,
#     weights: torch.Tensor,
#     mean: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     center = _weighted_mean(values, weights) if mean is None else mean
#     epsilon = torch.finfo(values.dtype).eps
#     return (
#         weights * (values - center).square()
#     ).sum() / weights.sum().clamp_min(epsilon)


# def _effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
#     epsilon = torch.finfo(weights.dtype).eps
#     numerator = weights.sum().square()
#     denominator = weights.square().sum().clamp_min(epsilon)
#     return numerator / denominator


# def _class_balanced_weighted_mean(
#     values: torch.Tensor,
#     targets: torch.Tensor,
#     sample_weights: torch.Tensor,
# ) -> torch.Tensor:
#     terms: List[torch.Tensor] = []
#     for class_id in torch.unique(targets, sorted=True):
#         mask = targets.eq(class_id)
#         class_weights = sample_weights[mask]
#         if float(class_weights.sum().detach().item()) <= 0.0:
#             raise RuntimeError(
#                 f"class {int(class_id.item())} has zero total sample weight"
#             )
#         terms.append(_weighted_mean(values[mask], class_weights))
#     if not terms:
#         raise RuntimeError("class-balanced reduction received no classes")
#     return torch.stack(terms).mean()


# def _require_pair_matrix(
#     value: Optional[torch.Tensor],
#     *,
#     class_count: int,
#     device: torch.device,
#     dtype: torch.dtype,
#     name: str,
#     default: Optional[float] = None,
#     boolean: bool = False,
# ) -> Optional[torch.Tensor]:
#     if value is None:
#         if default is None:
#             return None
#         if boolean:
#             return torch.full(
#                 (class_count, class_count),
#                 bool(default),
#                 device=device,
#                 dtype=torch.bool,
#             )
#         return torch.full(
#             (class_count, class_count),
#             float(default),
#             device=device,
#             dtype=dtype,
#         )
#     if boolean:
#         matrix = torch.as_tensor(value, device=device, dtype=torch.bool)
#     else:
#         matrix = torch.as_tensor(value, device=device, dtype=dtype)
#     if matrix.shape != (class_count, class_count):
#         raise ValueError(
#             f"{name} must be [C,C]; got {tuple(matrix.shape)} for C={class_count}"
#         )
#     if not boolean:
#         if not torch.isfinite(matrix).all():
#             raise RuntimeError(f"{name} contains NaN/Inf")
#         if bool((matrix < 0.0).any().item()):
#             raise ValueError(f"{name} must be non-negative")
#         matrix = matrix.detach()
#     return matrix


# def _validate_joint_factorization(value: Optional[str]) -> None:
#     if value is None:
#         return
#     token = str(value).replace(" ", "")
#     accepted = {
#         "p(z|c)prod_kp(g_k|z,c)",
#         "p(z,g|c)=p(z|c)prod_kp(g_k|z,c)",
#     }
#     if token not in accepted:
#         raise RuntimeError(
#             "loss received energy from an incompatible factorization; "
#             "PC-STGB requires p(z|c)prod_k p(g_k|z,c)"
#         )


# # =============================================================================
# # Exact deployed-energy consolidation
# # =============================================================================


# def phase_consistent_conditional_joint_consolidation_loss(
#     joint_energy: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     margin: float,
#     temperature: float = 0.20,
#     pair_margin_matrix: Optional[torch.Tensor] = None,
#     sample_weights: Optional[torch.Tensor] = None,
#     rival_indices: Optional[torch.Tensor] = None,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     class_balanced: bool = True,
#     joint_factorization: Optional[str] = None,
#     return_parts: bool = True,
# ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#     """PC-STGB margin on the exact deployed conditional joint energy.

#     The caller must pass

#         E_c(z,g) = E_c^occ(z) + beta_T E_c^tan(g | z).

#     No feature distance, prototype distance, or independently computed response
#     score is reconstructed inside this loss.  The same objective is therefore
#     valid for held-out base tuples, current new tuples, and coupled old replay.
#     """
#     _validate_joint_factorization(joint_factorization)
#     energy = _require_energy_matrix(joint_energy, name="joint_energy")
#     sample_count, class_count = energy.shape
#     y = _require_targets(
#         targets,
#         sample_count=sample_count,
#         class_count=class_count,
#         device=energy.device,
#     )
#     margin_value = _require_nonnegative_finite(margin, name="margin")
#     temperature_value = _require_positive_finite(temperature, name="temperature")

#     valid = _require_valid_class_mask(
#         valid_class_mask,
#         class_count=class_count,
#         device=energy.device,
#     )
#     target_valid = valid.index_select(0, y)
#     if not bool(target_valid.all().item()):
#         bad = torch.unique(y[~target_valid]).detach().cpu().tolist()
#         raise RuntimeError(f"targets reference invalid class columns: {bad}")

#     weights = _require_sample_weights(
#         sample_weights,
#         sample_count=sample_count,
#         device=energy.device,
#         dtype=energy.dtype,
#     )
#     pair_margins = _require_pair_matrix(
#         pair_margin_matrix,
#         class_count=class_count,
#         device=energy.device,
#         dtype=energy.dtype,
#         name="pair_margin_matrix",
#     )

#     true_energy = energy.gather(1, y[:, None]).squeeze(1)
#     rival_mask = valid[None, :].expand_as(energy).clone()
#     rival_mask.scatter_(1, y[:, None], False)
#     if not bool(rival_mask.any(dim=1).all().item()):
#         raise RuntimeError("at least one sample has no valid rival class")

#     if rival_indices is None:
#         masked = energy.masked_fill(~rival_mask, float("inf"))
#         rival_energy, rivals = masked.min(dim=1)
#     else:
#         rivals = _require_targets(
#             rival_indices,
#             sample_count=sample_count,
#             class_count=class_count,
#             device=energy.device,
#             name="rival_indices",
#         )
#         invalid = rivals.eq(y) | ~valid.index_select(0, rivals)
#         if bool(invalid.any().item()):
#             bad = torch.nonzero(invalid, as_tuple=False).flatten().detach().cpu().tolist()
#             raise ValueError(
#                 "rival_indices must reference valid non-target columns; "
#                 f"bad sample indices={bad[:20]}"
#             )
#         rival_energy = energy.gather(1, rivals[:, None]).squeeze(1)

#     gap = rival_energy - true_energy
#     required_margin = energy.new_full((sample_count,), margin_value)
#     if pair_margins is not None:
#         required_margin = torch.maximum(required_margin, pair_margins[y, rivals])

#     deficiency = required_margin - gap
#     per_sample = temperature_value * F.softplus(deficiency / temperature_value)

#     reducer = _class_balanced_weighted_mean if class_balanced else _weighted_mean
#     if class_balanced:
#         total = reducer(per_sample, y, weights)
#         mean_gap = reducer(gap, y, weights)
#         margin_violation = reducer(gap.lt(required_margin).to(energy.dtype), y, weights)
#         classification_violation = reducer(gap.le(0.0).to(energy.dtype), y, weights)
#     else:
#         total = reducer(per_sample, weights)
#         mean_gap = reducer(gap, weights)
#         margin_violation = reducer(gap.lt(required_margin).to(energy.dtype), weights)
#         classification_violation = reducer(gap.le(0.0).to(energy.dtype), weights)

#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("PC-STGB consolidation loss is not a finite scalar")
#     if not return_parts:
#         return total

#     collapse = torch.relu(-gap)
#     return {
#         "total": total,
#         "per_sample": per_sample,
#         "true_energy": true_energy,
#         "rival_energy": rival_energy,
#         "best_rival_energy": rival_energy,
#         "rival_indices": rivals.detach(),
#         "best_rival": rivals.detach(),
#         "gap": gap,
#         "required_margin": required_margin,
#         "mean_gap": mean_gap.detach(),
#         "q05_gap": torch.quantile(gap.detach(), 0.05),
#         "minimum_gap": gap.min().detach(),
#         "mean_required_margin": _weighted_mean(required_margin, weights).detach(),
#         "margin_violation_rate": margin_violation.detach(),
#         "classification_violation_rate": classification_violation.detach(),
#         "mean_collapse_severity": _weighted_mean(collapse, weights).detach(),
#         "maximum_collapse_severity": collapse.max().detach(),
#         "active_class_count": valid.sum().to(energy.dtype).detach(),
#         "sample_count": energy.new_tensor(float(sample_count)),
#         "uses_exact_joint_energy": energy.new_tensor(True, dtype=torch.bool),
#     }


# # Public compatibility name used by existing trainer code.  It now implements
# # the conditional PC-STGB objective rather than the old independent PC-SIRG rule.
# phase_consistent_spectral_geometry_consolidation_loss = (
#     phase_consistent_conditional_joint_consolidation_loss
# )
# pc_stgb_loss = phase_consistent_conditional_joint_consolidation_loss
# pc_sgc_loss = phase_consistent_conditional_joint_consolidation_loss


# # =============================================================================
# # Robust pairwise decision-boundary certificate
# # =============================================================================


# def joint_energy_boundary_certificate_loss(
#     joint_energy: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     margin: float = 0.0,
#     temperature: float = 0.10,
#     confidence_multiplier: float = 1.0,
#     pair_margin_matrix: Optional[torch.Tensor] = None,
#     pair_mask: Optional[torch.Tensor] = None,
#     pair_weights: Optional[torch.Tensor] = None,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     sample_weights: Optional[torch.Tensor] = None,
#     minimum_samples_per_source: int = 2,
#     joint_factorization: Optional[str] = None,
#     return_parts: bool = True,
# ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#     """Protect a lower confidence bound of each ordered joint-energy gap.

#     For source class ``c`` and rival ``j``:

#         Delta_cj = E_j - E_c
#         LCB_cj   = mean(Delta_cj) - kappa * std(Delta_cj)

#     The distributional standard deviation is used deliberately rather than the
#     standard error.  The objective therefore protects the lower tail of the
#     class region, not merely confidence in the estimated mean.

#     This is the boundary reserve for PC-STGB.  It uses the same conditional
#     joint energy as inference and does not construct a second Euclidean
#     ellipsoid or manually scale tangent responses.
#     """
#     _validate_joint_factorization(joint_factorization)
#     energy = _require_energy_matrix(joint_energy, name="joint_energy")
#     n, c = energy.shape
#     y = _require_targets(targets, sample_count=n, class_count=c, device=energy.device)
#     margin_value = _require_nonnegative_finite(margin, name="margin")
#     temperature_value = _require_positive_finite(temperature, name="temperature")
#     kappa = _require_nonnegative_finite(
#         confidence_multiplier, name="confidence_multiplier"
#     )
#     minimum_support = int(minimum_samples_per_source)
#     if minimum_support < 2:
#         raise ValueError("minimum_samples_per_source must be at least two")

#     valid = _require_valid_class_mask(
#         valid_class_mask, class_count=c, device=energy.device
#     )
#     if not bool(valid.index_select(0, y).all().item()):
#         raise RuntimeError("targets reference invalid class columns")
#     weights = _require_sample_weights(
#         sample_weights,
#         sample_count=n,
#         device=energy.device,
#         dtype=energy.dtype,
#     )
#     margins = _require_pair_matrix(
#         pair_margin_matrix,
#         class_count=c,
#         device=energy.device,
#         dtype=energy.dtype,
#         name="pair_margin_matrix",
#     )
#     selected = _require_pair_matrix(
#         pair_mask,
#         class_count=c,
#         device=energy.device,
#         dtype=energy.dtype,
#         name="pair_mask",
#         default=1.0,
#         boolean=True,
#     )
#     assert selected is not None
#     selected = selected.clone()
#     selected.fill_diagonal_(False)
#     selected &= valid[:, None] & valid[None, :]

#     configured_weights = _require_pair_matrix(
#         pair_weights,
#         class_count=c,
#         device=energy.device,
#         dtype=energy.dtype,
#         name="pair_weights",
#         default=1.0,
#     )
#     assert configured_weights is not None

#     nan_matrix = energy.new_full((c, c), float("nan"))
#     gap_mean_matrix = nan_matrix.clone()
#     gap_std_matrix = nan_matrix.clone()
#     gap_lcb_matrix = nan_matrix.clone()
#     gap_q05_matrix = nan_matrix.clone()
#     required_margin_matrix = nan_matrix.clone()
#     violation_matrix = energy.new_zeros((c, c))
#     classification_invasion_matrix = energy.new_zeros((c, c))
#     effective_n_matrix = energy.new_zeros((c, c))

#     pair_losses: List[torch.Tensor] = []
#     active_weights: List[torch.Tensor] = []
#     lcb_values: List[torch.Tensor] = []
#     required_values: List[torch.Tensor] = []
#     pair_indices: List[torch.Tensor] = []

#     for source in range(c):
#         if not bool(valid[source].item()):
#             continue
#         source_mask = y.eq(source)
#         if int(source_mask.sum().item()) < minimum_support:
#             continue
#         local_weights = weights[source_mask]
#         effective_n = _effective_sample_size(local_weights)
#         if float(effective_n.detach().item()) < float(minimum_support):
#             continue
#         true = energy[source_mask, source]
#         for rival in range(c):
#             if not bool(selected[source, rival].item()):
#                 continue
#             gap = energy[source_mask, rival] - true
#             mean = _weighted_mean(gap, local_weights)
#             variance = _weighted_population_variance(gap, local_weights, mean)
#             std = torch.sqrt(variance.clamp_min(0.0) + torch.finfo(gap.dtype).eps)
#             lcb = mean - kappa * std
#             required = energy.new_tensor(margin_value)
#             if margins is not None:
#                 required = torch.maximum(required, margins[source, rival])
#             deficiency = required - lcb
#             pair_loss = temperature_value * F.softplus(
#                 deficiency / temperature_value
#             )
#             pair_weight = configured_weights[source, rival]
#             if float(pair_weight.detach().item()) <= 0.0:
#                 continue

#             pair_losses.append(pair_loss)
#             active_weights.append(pair_weight)
#             lcb_values.append(lcb)
#             required_values.append(required)
#             pair_indices.append(
#                 torch.tensor([source, rival], device=energy.device, dtype=torch.long)
#             )

#             gap_mean_matrix[source, rival] = mean
#             gap_std_matrix[source, rival] = std
#             gap_lcb_matrix[source, rival] = lcb
#             gap_q05_matrix[source, rival] = torch.quantile(gap.detach(), 0.05)
#             required_margin_matrix[source, rival] = required
#             violation_matrix[source, rival] = lcb.lt(required).to(energy.dtype)
#             classification_invasion_matrix[source, rival] = (
#                 gap.le(0.0).to(energy.dtype) * local_weights
#             ).sum() / local_weights.sum().clamp_min(torch.finfo(energy.dtype).eps)
#             effective_n_matrix[source, rival] = effective_n

#     zero = energy.sum() * 0.0
#     if not pair_losses:
#         result = {
#             "total": zero,
#             "pair_count": zero.detach(),
#             "mean_lcb": zero.detach(),
#             "minimum_lcb": zero.detach(),
#             "q05_lcb": zero.detach(),
#             "certificate_violation_rate": zero.detach(),
#             "classification_invasion_rate": zero.detach(),
#             "gap_mean_matrix": gap_mean_matrix.detach(),
#             "gap_std_matrix": gap_std_matrix.detach(),
#             "gap_lcb_matrix": gap_lcb_matrix.detach(),
#             "gap_q05_matrix": gap_q05_matrix.detach(),
#             "required_margin_matrix": required_margin_matrix.detach(),
#             "certificate_violation_matrix": violation_matrix.detach(),
#             "classification_invasion_matrix": classification_invasion_matrix.detach(),
#             "effective_sample_size_matrix": effective_n_matrix.detach(),
#             "pair_indices": torch.empty((0, 2), device=energy.device, dtype=torch.long),
#         }
#         return result if return_parts else zero

#     loss_vector = torch.stack(pair_losses)
#     pair_weight_vector = torch.stack(active_weights).to(loss_vector)
#     total = _weighted_mean(loss_vector, pair_weight_vector)
#     lcb_vector = torch.stack(lcb_values)
#     required_vector = torch.stack(required_values)
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("boundary certificate loss is not a finite scalar")
#     if not return_parts:
#         return total

#     active_invasion = classification_invasion_matrix[
#         torch.isfinite(gap_lcb_matrix)
#     ]
#     return {
#         "total": total,
#         "pair_count": energy.new_tensor(float(len(pair_losses))),
#         "mean_lcb": lcb_vector.mean().detach(),
#         "minimum_lcb": lcb_vector.min().detach(),
#         "q05_lcb": torch.quantile(lcb_vector.detach(), 0.05),
#         "mean_required_margin": required_vector.mean().detach(),
#         "certificate_violation_rate": lcb_vector.lt(required_vector).to(energy.dtype).mean().detach(),
#         "classification_invasion_rate": active_invasion.mean().detach(),
#         "gap_mean_matrix": gap_mean_matrix.detach(),
#         "gap_std_matrix": gap_std_matrix.detach(),
#         "gap_lcb_matrix": gap_lcb_matrix.detach(),
#         "gap_q05_matrix": gap_q05_matrix.detach(),
#         "required_margin_matrix": required_margin_matrix.detach(),
#         "certificate_violation_matrix": violation_matrix.detach(),
#         "classification_invasion_matrix": classification_invasion_matrix.detach(),
#         "effective_sample_size_matrix": effective_n_matrix.detach(),
#         "pair_indices": torch.stack(pair_indices),
#     }


# # =============================================================================
# # Exact two-sided old/new invasion control
# # =============================================================================


# def _cross_partition_direction(
#     energy: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     rival_mask: torch.Tensor,
#     sample_weights: torch.Tensor,
#     margin: float,
#     temperature: float,
#     class_balanced: bool,
#     name: str,
# ) -> Dict[str, torch.Tensor]:
#     true = energy.gather(1, targets[:, None]).squeeze(1)
#     masked = energy.masked_fill(~rival_mask.view(1, -1), float("inf"))
#     rival_energy, rival_indices = masked.min(dim=1)
#     if not torch.isfinite(rival_energy).all():
#         raise RuntimeError(f"{name} has no valid rival partition")
#     gap = rival_energy - true
#     per_sample = temperature * F.softplus((margin - gap) / temperature)
#     if class_balanced:
#         total = _class_balanced_weighted_mean(per_sample, targets, sample_weights)
#         mean_gap = _class_balanced_weighted_mean(gap, targets, sample_weights)
#         violation = _class_balanced_weighted_mean(
#             gap.lt(margin).to(energy.dtype), targets, sample_weights
#         )
#         invasion = _class_balanced_weighted_mean(
#             gap.le(0.0).to(energy.dtype), targets, sample_weights
#         )
#     else:
#         total = _weighted_mean(per_sample, sample_weights)
#         mean_gap = _weighted_mean(gap, sample_weights)
#         violation = _weighted_mean(gap.lt(margin).to(energy.dtype), sample_weights)
#         invasion = _weighted_mean(gap.le(0.0).to(energy.dtype), sample_weights)
#     return {
#         "total": total,
#         "gap": gap,
#         "per_sample": per_sample,
#         "rival_indices": rival_indices.detach(),
#         "mean_gap": mean_gap.detach(),
#         "q05_gap": torch.quantile(gap.detach(), 0.05),
#         "minimum_gap": gap.min().detach(),
#         "margin_violation_rate": violation.detach(),
#         "invasion_rate": invasion.detach(),
#     }


# def two_sided_old_new_invasion_loss(
#     old_joint_energy: torch.Tensor,
#     old_targets: torch.Tensor,
#     new_joint_energy: torch.Tensor,
#     new_targets: torch.Tensor,
#     *,
#     old_class_mask: torch.Tensor,
#     new_class_mask: torch.Tensor,
#     margin: float = 0.0,
#     temperature: float = 0.20,
#     old_to_new_weight: float = 1.0,
#     new_to_old_weight: float = 1.0,
#     old_sample_weights: Optional[torch.Tensor] = None,
#     new_sample_weights: Optional[torch.Tensor] = None,
#     class_balanced: bool = True,
#     joint_factorization: Optional[str] = None,
#     return_parts: bool = True,
# ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#     """Control both directions of old/new interference using exact energy.

#     * old -> new: a new candidate row must not steal coupled old replay;
#     * new -> old: committed old rows must not absorb current real samples.

#     Both matrices must use the same seen-class column order and the exact
#     conditional joint energy returned by the PC-STGB classifier.
#     """
#     _validate_joint_factorization(joint_factorization)
#     old_energy = _require_energy_matrix(old_joint_energy, name="old_joint_energy")
#     new_energy = _require_energy_matrix(new_joint_energy, name="new_joint_energy")
#     if old_energy.size(1) != new_energy.size(1):
#         raise ValueError("old and new joint energies must use the same class columns")
#     c = old_energy.size(1)
#     old_mask = _require_partition_mask(
#         old_class_mask, class_count=c, device=old_energy.device, name="old_class_mask"
#     )
#     new_mask = _require_partition_mask(
#         new_class_mask, class_count=c, device=old_energy.device, name="new_class_mask"
#     )
#     if old_energy.device != new_energy.device:
#         raise RuntimeError("old and new joint energies must share one device")
#     new_mask = new_mask.to(new_energy.device)
#     if bool((old_mask & new_mask).any().item()):
#         raise ValueError("old_class_mask and new_class_mask overlap")
#     if not bool((old_mask | new_mask).all().item()):
#         raise ValueError("old/new class masks must partition every energy column")

#     old_y = _require_targets(
#         old_targets,
#         sample_count=old_energy.size(0),
#         class_count=c,
#         device=old_energy.device,
#         name="old_targets",
#     )
#     new_y = _require_targets(
#         new_targets,
#         sample_count=new_energy.size(0),
#         class_count=c,
#         device=new_energy.device,
#         name="new_targets",
#     )
#     if not bool(old_mask.index_select(0, old_y).all().item()):
#         raise RuntimeError("old_targets contain non-old class columns")
#     if not bool(new_mask.index_select(0, new_y).all().item()):
#         raise RuntimeError("new_targets contain non-new class columns")

#     margin_value = _require_nonnegative_finite(margin, name="margin")
#     temperature_value = _require_positive_finite(temperature, name="temperature")
#     old_direction_weight = _require_nonnegative_finite(
#         old_to_new_weight, name="old_to_new_weight"
#     )
#     new_direction_weight = _require_nonnegative_finite(
#         new_to_old_weight, name="new_to_old_weight"
#     )
#     if old_direction_weight + new_direction_weight <= 0.0:
#         raise ValueError("at least one directional weight must be positive")

#     old_weights = _require_sample_weights(
#         old_sample_weights,
#         sample_count=old_energy.size(0),
#         device=old_energy.device,
#         dtype=old_energy.dtype,
#     )
#     new_weights = _require_sample_weights(
#         new_sample_weights,
#         sample_count=new_energy.size(0),
#         device=new_energy.device,
#         dtype=new_energy.dtype,
#     )

#     old_to_new = _cross_partition_direction(
#         old_energy,
#         old_y,
#         rival_mask=new_mask.to(old_energy.device),
#         sample_weights=old_weights,
#         margin=margin_value,
#         temperature=temperature_value,
#         class_balanced=class_balanced,
#         name="old_to_new",
#     )
#     new_to_old = _cross_partition_direction(
#         new_energy,
#         new_y,
#         rival_mask=old_mask.to(new_energy.device),
#         sample_weights=new_weights,
#         margin=margin_value,
#         temperature=temperature_value,
#         class_balanced=class_balanced,
#         name="new_to_old",
#     )
#     total = (
#         old_direction_weight * old_to_new["total"]
#         + new_direction_weight * new_to_old["total"]
#     ) / (old_direction_weight + new_direction_weight)
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("two-sided invasion loss is not a finite scalar")
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "old_to_new_total": old_to_new["total"],
#         "new_to_old_total": new_to_old["total"],
#         "old_to_new_gap": old_to_new["gap"],
#         "new_to_old_gap": new_to_old["gap"],
#         "old_to_new_rival_indices": old_to_new["rival_indices"],
#         "new_to_old_rival_indices": new_to_old["rival_indices"],
#         "old_to_new_mean_gap": old_to_new["mean_gap"],
#         "new_to_old_mean_gap": new_to_old["mean_gap"],
#         "old_to_new_q05_gap": old_to_new["q05_gap"],
#         "new_to_old_q05_gap": new_to_old["q05_gap"],
#         "old_to_new_minimum_gap": old_to_new["minimum_gap"],
#         "new_to_old_minimum_gap": new_to_old["minimum_gap"],
#         "old_to_new_margin_violation_rate": old_to_new["margin_violation_rate"],
#         "new_to_old_margin_violation_rate": new_to_old["margin_violation_rate"],
#         "old_to_new_invasion_rate": old_to_new["invasion_rate"],
#         "new_to_old_invasion_rate": new_to_old["invasion_rate"],
#     }


# # =============================================================================
# # Physical finite-difference validity regularizer
# # =============================================================================


# def spectral_tangent_step_consistency_loss(
#     coarse_responses: torch.Tensor,
#     fine_responses: torch.Tensor,
#     *,
#     sample_weights: Optional[torch.Tensor] = None,
#     magnitude_weight: float = 0.5,
#     direction_weight: float = 0.5,
#     minimum_reference_norm: float = 1e-6,
#     return_parts: bool = True,
# ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#     """Check whether two central-difference step sizes estimate one tangent.

#     ``coarse_responses`` and ``fine_responses`` must be computed from the same
#     original samples and intervention definitions, for example with step sizes
#     ``h`` and ``h/2``.  This regularizer validates the local derivative object;
#     it is not a classifier and cannot replace the exact joint-energy losses.
#     """
#     if not torch.is_tensor(coarse_responses) or not torch.is_tensor(fine_responses):
#         raise TypeError("coarse_responses and fine_responses must be tensors")
#     if coarse_responses.shape != fine_responses.shape or coarse_responses.dim() != 3:
#         raise ValueError("response tensors must have identical [N,K,D] shapes")
#     if not torch.is_floating_point(coarse_responses):
#         raise TypeError("response tensors must use a floating dtype")
#     if fine_responses.device != coarse_responses.device:
#         raise RuntimeError("response tensors must share one device")
#     fine = fine_responses.to(dtype=coarse_responses.dtype)
#     if not torch.isfinite(coarse_responses).all() or not torch.isfinite(fine).all():
#         raise RuntimeError("response tensors contain NaN/Inf")

#     magnitude_gain = _require_nonnegative_finite(magnitude_weight, name="magnitude_weight")
#     direction_gain = _require_nonnegative_finite(direction_weight, name="direction_weight")
#     if magnitude_gain + direction_gain <= 0.0:
#         raise ValueError("at least one consistency component must be active")
#     floor = _require_positive_finite(
#         minimum_reference_norm, name="minimum_reference_norm"
#     )
#     weights = _require_sample_weights(
#         sample_weights,
#         sample_count=coarse_responses.size(0),
#         device=coarse_responses.device,
#         dtype=coarse_responses.dtype,
#     )

#     coarse_norm = coarse_responses.norm(dim=2)
#     fine_norm = fine.norm(dim=2)
#     reference = 0.5 * (coarse_norm + fine_norm)
#     relative_magnitude_error = (coarse_norm - fine_norm).abs() / reference.clamp_min(floor)
#     cosine = F.cosine_similarity(coarse_responses, fine, dim=2, eps=floor)
#     direction_error = (1.0 - cosine).clamp_min(0.0)
#     valid_direction = reference.gt(floor)
#     direction_error = torch.where(valid_direction, direction_error, torch.zeros_like(direction_error))

#     per_intervention = (
#         magnitude_gain * relative_magnitude_error
#         + direction_gain * direction_error
#     ) / (magnitude_gain + direction_gain)
#     per_sample = per_intervention.mean(dim=1)
#     total = _weighted_mean(per_sample, weights)
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("spectral tangent consistency loss is not a finite scalar")
#     if not return_parts:
#         return total
#     return {
#         "total": total,
#         "per_sample": per_sample,
#         "relative_magnitude_error": relative_magnitude_error,
#         "direction_error": direction_error,
#         "mean_relative_magnitude_error": relative_magnitude_error.mean().detach(),
#         "mean_direction_error": direction_error.mean().detach(),
#         "coarse_response_norm_mean": coarse_norm.mean().detach(),
#         "fine_response_norm_mean": fine_norm.mean().detach(),
#         "near_zero_response_rate": reference.le(floor).to(coarse_responses.dtype).mean().detach(),
#     }


# # =============================================================================
# # Bounded candidate-descriptor trust region
# # =============================================================================


# def candidate_descriptor_trust_region_loss(
#     *,
#     mean_deltas: Mapping[int, torch.Tensor],
#     log_eigval_deltas: Mapping[int, torch.Tensor],
#     log_residual_deltas: Mapping[int, torch.Tensor],
#     response_mean_deltas: Mapping[int, torch.Tensor],
#     response_log_eigval_deltas: Mapping[int, torch.Tensor],
#     response_log_residual_deltas: Mapping[int, torch.Tensor],
#     mean_scales: Union[float, Mapping[int, float]] = 1.0,
#     log_eigval_scales: Union[float, Mapping[int, float]] = 1.0,
#     log_residual_scales: Union[float, Mapping[int, float]] = 1.0,
#     response_mean_scales: Union[float, Mapping[int, float]] = 1.0,
#     response_log_eigval_scales: Union[float, Mapping[int, float]] = 1.0,
#     response_log_residual_scales: Union[float, Mapping[int, float]] = 1.0,
#     class_weights: Optional[Mapping[int, float]] = None,
#     return_parts: bool = True,
# ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#     """Regularize provisional new-row corrections using uncertainty scales.

#     The scales should be derived from bootstrap uncertainty or from the same
#     statistically justified limits used by ``GeometryBank.refine_candidate_joint_rows``.
#     This loss does not replace the bank's hard bounds.  It makes zero/small
#     corrections preferable inside those bounds and prevents descriptor
#     refinement from exploiting old replay through unnecessarily large changes.
#     """
#     mappings: Tuple[Tuple[str, Mapping[int, torch.Tensor]], ...] = (
#         ("mean", mean_deltas),
#         ("log_eigval", log_eigval_deltas),
#         ("log_residual", log_residual_deltas),
#         ("response_mean", response_mean_deltas),
#         ("response_log_eigval", response_log_eigval_deltas),
#         ("response_log_residual", response_log_residual_deltas),
#     )
#     key_sets = [set(int(key) for key in mapping) for _, mapping in mappings]
#     if not key_sets or not key_sets[0]:
#         raise ValueError("candidate delta mappings are empty")
#     expected = key_sets[0]
#     for (name, _), keys in zip(mappings, key_sets):
#         if keys != expected:
#             raise RuntimeError(f"{name}_deltas keys do not match candidate classes")

#     first_tensor: Optional[torch.Tensor] = None
#     for _, mapping in mappings:
#         for tensor in mapping.values():
#             if not torch.is_tensor(tensor):
#                 raise TypeError("candidate deltas must be tensors")
#             if not torch.is_floating_point(tensor):
#                 raise TypeError("candidate deltas must use floating dtypes")
#             if not torch.isfinite(tensor).all():
#                 raise RuntimeError("candidate deltas contain NaN/Inf")
#             if first_tensor is None:
#                 first_tensor = tensor
#     assert first_tensor is not None
#     device, dtype = first_tensor.device, first_tensor.dtype
#     for _, mapping in mappings:
#         for tensor in mapping.values():
#             if tensor.device != device:
#                 raise RuntimeError("all candidate deltas must share one device")

#     def resolve_scale(
#         configured: Union[float, Mapping[int, float]],
#         class_id: int,
#         name: str,
#     ) -> float:
#         value = configured[class_id] if isinstance(configured, Mapping) else configured
#         return _require_positive_finite(float(value), name=f"{name}[{class_id}]")

#     scale_configs: Tuple[Tuple[str, Union[float, Mapping[int, float]]], ...] = (
#         ("mean", mean_scales),
#         ("log_eigval", log_eigval_scales),
#         ("log_residual", log_residual_scales),
#         ("response_mean", response_mean_scales),
#         ("response_log_eigval", response_log_eigval_scales),
#         ("response_log_residual", response_log_residual_scales),
#     )

#     class_terms: List[torch.Tensor] = []
#     class_weight_values: List[torch.Tensor] = []
#     component_accumulator: Dict[str, List[torch.Tensor]] = {
#         name: [] for name, _ in mappings
#     }
#     ordered_ids = sorted(expected)
#     for class_id in ordered_ids:
#         component_terms: List[torch.Tensor] = []
#         for (name, mapping), (_, configured_scale) in zip(mappings, scale_configs):
#             tensor = mapping[class_id].to(dtype=dtype)
#             scale = resolve_scale(configured_scale, class_id, name)
#             normalized_square = (tensor / scale).square()
#             term = normalized_square.mean() if normalized_square.numel() else tensor.sum() * 0.0
#             component_accumulator[name].append(term)
#             component_terms.append(term)
#         class_term = torch.stack(component_terms).mean()
#         class_weight = 1.0 if class_weights is None else float(class_weights[class_id])
#         class_weight = _require_nonnegative_finite(
#             class_weight, name=f"class_weights[{class_id}]"
#         )
#         if class_weight <= 0.0:
#             continue
#         class_terms.append(class_term)
#         class_weight_values.append(
#             torch.tensor(class_weight, device=device, dtype=dtype)
#         )

#     if not class_terms:
#         raise ValueError("all candidate class weights are zero")
#     terms = torch.stack(class_terms)
#     weights = torch.stack(class_weight_values)
#     total = _weighted_mean(terms, weights)
#     if total.dim() != 0 or not torch.isfinite(total):
#         raise RuntimeError("candidate trust-region loss is not a finite scalar")
#     if not return_parts:
#         return total

#     output: Dict[str, torch.Tensor] = {
#         "total": total,
#         "class_count": torch.tensor(float(len(class_terms)), device=device, dtype=dtype),
#         "mean_class_penalty": terms.mean().detach(),
#         "maximum_class_penalty": terms.max().detach(),
#     }
#     for name, values in component_accumulator.items():
#         output[f"{name}_penalty"] = torch.stack(values).mean().detach()
#     return output


# # =============================================================================
# # Boundary-risk diagnostics
# # =============================================================================


# @torch.no_grad()
# def pairwise_directional_invasion_matrix(
#     joint_energy: torch.Tensor,
#     targets: torch.Tensor,
#     *,
#     valid_class_mask: Optional[torch.Tensor] = None,
#     sample_weights: Optional[torch.Tensor] = None,
#     joint_factorization: Optional[str] = None,
# ) -> Dict[str, torch.Tensor]:
#     """Measure ordered source-to-rival invasion under deployed joint energy."""
#     _validate_joint_factorization(joint_factorization)
#     energy = _require_energy_matrix(joint_energy, name="joint_energy")
#     n, c = energy.shape
#     y = _require_targets(targets, sample_count=n, class_count=c, device=energy.device)
#     valid = _require_valid_class_mask(
#         valid_class_mask, class_count=c, device=energy.device
#     )
#     if not bool(valid.index_select(0, y).all().item()):
#         raise RuntimeError("directional invasion targets reference invalid rows")
#     weights = _require_sample_weights(
#         sample_weights,
#         sample_count=n,
#         device=energy.device,
#         dtype=torch.float64,
#     )

#     invasion = torch.zeros((c, c), device=energy.device, dtype=torch.float64)
#     source_weight = torch.zeros(c, device=energy.device, dtype=torch.float64)
#     gap_q05 = torch.full((c, c), float("nan"), device=energy.device, dtype=torch.float64)
#     for source in range(c):
#         if not bool(valid[source].item()):
#             continue
#         mask = y.eq(source)
#         if not bool(mask.any().item()):
#             continue
#         local_weights = weights[mask]
#         denominator = local_weights.sum().clamp_min(torch.finfo(torch.float64).eps)
#         source_weight[source] = denominator
#         true = energy[mask, source]
#         for rival in range(c):
#             if rival == source or not bool(valid[rival].item()):
#                 continue
#             gap = energy[mask, rival] - true
#             invaded = gap.le(0.0).to(torch.float64)
#             invasion[source, rival] = (invaded * local_weights).sum() / denominator
#             gap_q05[source, rival] = torch.quantile(gap.double(), 0.05)

#     offdiag_mask = (
#         valid[:, None]
#         & valid[None, :]
#         & ~torch.eye(c, device=energy.device, dtype=torch.bool)
#     )
#     offdiag = invasion[offdiag_mask]
#     return {
#         "invasion_matrix": invasion,
#         "gap_q05_matrix": gap_q05,
#         "source_weight": source_weight,
#         "source_counts": source_weight,
#         "maximum_directional_invasion": (
#             offdiag.max() if offdiag.numel() else invasion.new_tensor(0.0)
#         ),
#         "mean_directional_invasion": (
#             offdiag.mean() if offdiag.numel() else invasion.new_tensor(0.0)
#         ),
#     }


# # =============================================================================
# # Explicit guards for incompatible objectives
# # =============================================================================


# def _retired_loss(name: str, replacement: str) -> Callable[..., Any]:
#     def retired(*_: Any, **__: Any) -> Any:
#         raise RuntimeError(
#             f"{name} constructs an objective outside the conditional PC-STGB "
#             f"decision model. {replacement}"
#         )

#     retired.__name__ = name
#     return retired


# response_conditioned_directional_clearance_reserve_loss = _retired_loss(
#     "response_conditioned_directional_clearance_reserve_loss",
#     "Use joint_energy_boundary_certificate_loss() on the exact conditional "
#     "joint energy instead of a second mean/std Euclidean boundary model.",
# )
# pc_sirg_overlap_reserve_loss = response_conditioned_directional_clearance_reserve_loss

# base_geometry_preparation_loss = _retired_loss(
#     "base_geometry_preparation_loss",
#     "Use phase_consistent_conditional_joint_consolidation_loss() and "
#     "joint_energy_boundary_certificate_loss().",
# )
# base_geometry_involved_contrastive_loss = _retired_loss(
#     "base_geometry_involved_contrastive_loss",
#     "Use held-out conditional joint-energy consolidation.",
# )
# prospective_geometry_reserve_loss = _retired_loss(
#     "prospective_geometry_reserve_loss",
#     "Use joint_energy_boundary_certificate_loss() in every phase.",
# )
# prospective_admission_loss = _retired_loss(
#     "prospective_admission_loss",
#     "Use the same joint-energy consolidation, certificate, and two-sided "
#     "invasion objectives for current real and coupled old replay tuples.",
# )
# spectral_conditioned_margin_matrix = _retired_loss(
#     "spectral_conditioned_margin_matrix",
#     "Build pair-specific margins externally from certified joint-energy gap "
#     "statistics; do not construct a second spectral score.",
# )
# spectral_conditioned_geometry_consistency_loss = _retired_loss(
#     "spectral_conditioned_geometry_consistency_loss",
#     "Use spectral_tangent_step_consistency_loss() only to validate finite "
#     "differences, and use conditional joint energy for classification.",
# )
# cross_fitted_joint_spectral_feature_loss = _retired_loss(
#     "cross_fitted_joint_spectral_feature_loss",
#     "Use the bank's occupancy-conditioned tangent factorization.",
# )
# geometry_energy_matrix = _retired_loss(
#     "geometry_energy_matrix",
#     "The GeometryBank/classifier is the sole energy authority.",
# )
# incremental_low_rank_boundary_loss = _retired_loss(
#     "incremental_low_rank_boundary_loss",
#     "Use joint_energy_boundary_certificate_loss() and "
#     "two_sided_old_new_invasion_loss().",
# )
# incremental_geometry_training_loss = incremental_low_rank_boundary_loss
