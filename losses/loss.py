from __future__ import annotations

"""Geometry objectives for one-space NECIL-HSI.

The classifier uses the persistent pairwise decision geometry

    h_ab(z) = n_ab^T z + q_ab,     a < b,

and class energy

    E_c(z) = - min_{j != c} s_cj(z).

Training therefore uses the same geometry for two complementary purposes:

1. global classification over all seen classes with logits = -E;
2. pairwise distribution separation over the trainable current-phase pairs.

For a trainable pair (a,b), the separator itself provides the discriminative
axis.  The loss balances both class sides irrespective of sample count and
separates their projected distributions without prototypes or an arbitrary
margin:

    L_side(a,b)
      = 1/2 [ mean softplus(-h_ab(z_a))
            + mean softplus( h_ab(z_b)) ]

    L_order(a,b)
      = mean_{i,j} softplus(-(h_ab(z_i^a) - h_ab(z_j^b)))

    L_sep(a,b) = 1/2 [L_side(a,b) + L_order(a,b)].

At base phase the candidate contains every base-base pair.  At an incremental
phase it contains exactly the old-new and new-new pairs, so the same objective
naturally trains only newly introduced discrimination while committed old-old
geometry remains fixed.

Historical replay uses a separate preservation objective on class-incident
old boundary responses.  For replay item i of old class c,

    r_c(z_i) = [s_cj(z_i)]_{j != c}

is cached at phase start and compared with the current response while the
backbone evolves.  This preserves decision-relevant historical coordinates
without freezing the complete feature vector or using a teacher/prototype.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

from models.classifier import ClassifierOutput, GeometryClassifier
from models.geometry_bank import (
    BoundaryCandidate,
    BoundaryGeometryBank,
    ClassBoundaryResponse,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class GeometryTrainingObjective:
    """Global classification plus pairwise distribution separation."""

    total: Tensor
    classification: Tensor
    separation: Tensor
    accuracy: Tensor
    active_pair_count: int


@dataclass(frozen=True)
class HistoricalPreservationObjective:
    """Decision-coordinate preservation for historical replay."""

    total: Tensor
    mean_absolute_drift: Tensor
    max_absolute_drift: Tensor


def _weight(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _class_weights(weights: Tensor, *, class_count: int, reference: Tensor) -> Tensor:
    value = torch.as_tensor(weights)
    if value.device != reference.device:
        raise ValueError("class_risk_weights and classifier output must share a device")
    value = value.to(dtype=reference.dtype).flatten()
    if value.shape != (class_count,):
        raise ValueError("class_risk_weights must contain one weight per class column")
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError("class_risk_weights must be finite and positive")
    return value


def _weighted_mean(
    values: Tensor,
    targets_local: Tensor,
    class_risk_weights: Tensor,
    *,
    class_count: int,
) -> Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be non-empty [N]")
    weights = _class_weights(
        class_risk_weights,
        class_count=class_count,
        reference=values,
    )
    return (weights.index_select(0, targets_local) * values).mean()


def _validate_output(output: ClassifierOutput) -> None:
    if not isinstance(output, ClassifierOutput):
        raise TypeError("output must be ClassifierOutput")
    if (
        output.energy.ndim != 2
        or output.energy.size(0) == 0
        or output.energy.size(1) == 0
    ):
        raise ValueError("energy must be non-empty [N,C]")
    if output.logits.shape != output.energy.shape:
        raise ValueError("logits and energy must share [N,C]")
    if not bool(torch.equal(output.logits, -output.energy)):
        raise ValueError("classifier logits must equal -energy exactly")
    if not bool(torch.isfinite(output.energy).all()):
        raise ValueError("classifier output contains NaN/Inf")
    expected = output.class_ids.index_select(0, output.energy.argmin(dim=1))
    if not bool(torch.equal(output.prediction, expected)):
        raise ValueError("prediction must be the minimum-energy class")


def _require_all_seen_classes(
    output: ClassifierOutput,
    geometry_bank: BoundaryGeometryBank,
    candidate: BoundaryCandidate,
) -> tuple[int, ...]:
    committed = tuple(
        int(v) for v in geometry_bank.class_ids.detach().cpu().tolist()
    )
    new_ids = candidate.new_class_ids
    if set(committed).intersection(new_ids):
        raise ValueError("candidate contains already committed classes")
    expected = committed + new_ids
    if len(expected) != len(set(expected)):
        raise RuntimeError("visible class IDs are not unique")
    actual = tuple(int(v) for v in output.class_ids.detach().cpu().tolist())
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError("training classification must include every seen class")
    return actual


def _pairwise_distribution_separation(
    *,
    coordinates: Tensor,
    labels: Tensor,
    geometry_bank: BoundaryGeometryBank,
    candidate: BoundaryCandidate,
) -> tuple[Tensor, int]:
    """Separate both sides of every candidate pair represented in this batch.

    Each pair contributes equally, and each class side contributes equally
    inside a pair.  Therefore the objective is intrinsically insensitive to a
    2-vs-200 old/new sample-count imbalance for that boundary.

    Pairs for which one side is absent from the current batch are skipped; this
    is required for minibatch training and introduces no pair-specific threshold.
    The caller receives ``active_pair_count`` for diagnostics.
    """
    z = torch.as_tensor(coordinates)
    if z.ndim != 2 or z.size(0) == 0:
        raise ValueError("coordinates must be non-empty [N,D]")
    if z.device != geometry_bank.device or z.dtype != geometry_bank.dtype:
        raise ValueError("coordinates must share geometry device and dtype")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("coordinates contain NaN/Inf")

    y = torch.as_tensor(labels, device=z.device).flatten().to(dtype=torch.long)
    if y.shape != (z.size(0),):
        raise ValueError("labels and coordinates are row-misaligned")

    candidate.validate_state()
    if candidate.pair_ids.size(0) == 0:
        return z.sum() * 0.0, 0

    pair_geometry = geometry_bank.pair_values(
        z,
        pair_ids=candidate.pair_ids,
        candidate=candidate,
    )

    pair_losses: list[Tensor] = []
    for column, pair_row in enumerate(pair_geometry.pair_ids):
        left = int(pair_row[0].item())
        right = int(pair_row[1].item())
        left_values = pair_geometry.values[y.eq(left), column]
        right_values = pair_geometry.values[y.eq(right), column]
        if left_values.numel() == 0 or right_values.numel() == 0:
            continue

        # Positive h_ab favors left class; negative h_ab favors right class.
        side = 0.5 * (
            F.softplus(-left_values).mean()
            + F.softplus(right_values).mean()
        )

        # Direct distribution ordering along the actual discriminative axis.
        # Batch sizes in this project are small enough that the exact Cartesian
        # comparison is inexpensive and avoids sampling another approximation.
        pairwise_difference = (
            left_values.unsqueeze(1) - right_values.unsqueeze(0)
        )
        order = F.softplus(-pairwise_difference).mean()

        pair_losses.append(0.5 * (side + order))

    if not pair_losses:
        # A differentiable zero is necessary for batches containing only one
        # side of all candidate pairs. Global CE still trains that batch.
        return z.sum() * 0.0, 0

    separation = torch.stack(pair_losses).mean()
    if not bool(torch.isfinite(separation)):
        raise RuntimeError("pairwise distribution separation is NaN/Inf")
    return separation, len(pair_losses)


def geometry_training_objective(
    *,
    output: ClassifierOutput,
    coordinates: Tensor,
    labels_global: Tensor,
    geometry_bank: BoundaryGeometryBank,
    candidate: BoundaryCandidate,
    class_risk_weights: Tensor,
    classification_weight: float = 1.0,
    separation_weight: float = 1.0,
    separation_coordinates: Optional[Tensor] = None,
    separation_labels_global: Optional[Tensor] = None,
) -> GeometryTrainingObjective:
    """Train classification and current-phase pairwise geometry.

    ``coordinates`` / ``labels_global`` define the classification stream.
    The optional separation stream is useful in incremental phases where random
    real-new minibatches do not guarantee that every new-new class pair
    co-occurs. Base training omits it and is unchanged.
    """
    _validate_output(output)
    if not isinstance(geometry_bank, BoundaryGeometryBank):
        raise TypeError("geometry_bank must be BoundaryGeometryBank")
    if not isinstance(candidate, BoundaryCandidate):
        raise TypeError("candidate must be BoundaryCandidate")
    geometry_bank.validate_bank_state()
    candidate.validate_state()
    visible_ids = _require_all_seen_classes(output, geometry_bank, candidate)

    z = torch.as_tensor(coordinates)
    if z.device != output.energy.device:
        raise ValueError("coordinates and classifier output must share a device")
    if z.dtype != output.energy.dtype:
        raise ValueError("coordinates and classifier output must share a dtype")
    if z.ndim != 2 or z.size(0) != output.energy.size(0):
        raise ValueError("coordinates must align with classifier rows")

    labels = torch.as_tensor(labels_global)
    if labels.device != output.class_ids.device:
        raise ValueError("labels_global and classifier output must share a device")
    labels = labels.flatten().to(dtype=torch.long)
    targets = GeometryClassifier.targets_local(labels, output.class_ids)
    if targets.numel() != output.energy.size(0):
        raise ValueError("labels and classifier rows are misaligned")

    # Protect against accidentally training with a label outside all seen classes.
    if not set(int(v) for v in labels.detach().cpu().unique().tolist()).issubset(
        set(visible_ids)
    ):
        raise ValueError("labels contain classes outside the visible geometry")

    class_count = output.energy.size(1)
    per_sample_ce = F.cross_entropy(output.logits, targets, reduction="none")
    classification = _weighted_mean(
        per_sample_ce,
        targets,
        class_risk_weights,
        class_count=class_count,
    )

    # Classification and pair separation need not use identical row sets.
    # Base training leaves these arguments unset and therefore behaves exactly
    # as before. During incremental training, classification keeps the natural
    # real-new + selected-old exposure, while pair separation can receive a
    # class-complete current-phase support set so no candidate relation depends
    # on accidental minibatch co-occurrence.
    if (separation_coordinates is None) != (separation_labels_global is None):
        raise ValueError(
            "separation_coordinates and separation_labels_global must be "
            "provided together"
        )

    if separation_coordinates is None:
        separation_z = z
        separation_labels = labels
    else:
        separation_z = torch.as_tensor(separation_coordinates)
        if (
            separation_z.device != z.device
            or separation_z.dtype != z.dtype
            or separation_z.ndim != 2
            or separation_z.size(1) != z.size(1)
            or separation_z.size(0) == 0
        ):
            raise ValueError(
                "separation_coordinates must be non-empty [N,D] on the same "
                "device/dtype and representation dimension as coordinates"
            )
        if not bool(torch.isfinite(separation_z).all()):
            raise ValueError("separation_coordinates contain NaN/Inf")

        separation_labels = torch.as_tensor(
            separation_labels_global,
            device=z.device,
        ).flatten().to(dtype=torch.long)
        if separation_labels.shape != (separation_z.size(0),):
            raise ValueError(
                "separation_labels_global and separation_coordinates are "
                "row-misaligned"
            )
        if not set(
            int(v)
            for v in separation_labels.detach().cpu().unique().tolist()
        ).issubset(set(visible_ids)):
            raise ValueError(
                "separation labels contain classes outside the visible geometry"
            )

    separation, active_pair_count = _pairwise_distribution_separation(
        coordinates=separation_z,
        labels=separation_labels,
        geometry_bank=geometry_bank,
        candidate=candidate,
    )

    cls_w = _weight("classification_weight", classification_weight)
    sep_w = _weight("separation_weight", separation_weight)
    if cls_w == 0.0 and sep_w == 0.0:
        raise ValueError("at least one training objective weight must be positive")

    total = cls_w * classification + sep_w * separation
    if not bool(torch.isfinite(total)):
        raise RuntimeError("geometry objective is NaN/Inf")

    accuracy = output.prediction.eq(labels).to(torch.float32).mean()
    return GeometryTrainingObjective(
        total=total,
        classification=classification,
        separation=separation,
        accuracy=accuracy,
        active_pair_count=active_pair_count,
    )


def historical_response_preservation_objective(
    *,
    current: ClassBoundaryResponse,
    target_margins: Tensor,
    target_rival_class_ids: Tensor,
    weight: float = 1.0,
) -> HistoricalPreservationObjective:
    """Preserve old class-incident decision coordinates during adaptation.

    ``current`` must be computed from committed old-old geometry only.  Target
    margins/rival IDs are cached at phase start for the same replay rows.  Rival
    IDs are checked exactly so the loss cannot silently compare mismatched
    boundary columns.
    """
    if not isinstance(current, ClassBoundaryResponse):
        raise TypeError("current must be ClassBoundaryResponse")

    target = torch.as_tensor(target_margins)
    rivals = torch.as_tensor(target_rival_class_ids)
    if target.device != current.margins.device or rivals.device != current.margins.device:
        raise ValueError("historical targets and current response must share a device")
    target = target.to(dtype=current.margins.dtype)
    rivals = rivals.to(dtype=torch.long)

    if target.shape != current.margins.shape:
        raise ValueError("target historical margins have incompatible shape")
    if rivals.shape != current.rival_class_ids.shape:
        raise ValueError("target historical rival IDs have incompatible shape")
    if not bool(torch.equal(rivals, current.rival_class_ids)):
        raise ValueError("historical rival columns do not match current geometry response")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("historical target margins contain NaN/Inf")

    absolute_drift = (current.margins - target).abs()
    mean_drift = absolute_drift.mean()
    max_drift = absolute_drift.amax()
    preserve_w = _weight("preservation_weight", weight)
    total = preserve_w * mean_drift
    if not bool(torch.isfinite(total)):
        raise RuntimeError("historical response preservation is NaN/Inf")

    return HistoricalPreservationObjective(
        total=total,
        mean_absolute_drift=mean_drift,
        max_absolute_drift=max_drift,
    )


__all__ = [
    "GeometryTrainingObjective",
    "HistoricalPreservationObjective",
    "geometry_training_objective",
    "historical_response_preservation_objective",
]















# from __future__ import annotations

# """Geometry objectives for one-space NECIL-HSI.

# The classifier uses the persistent pairwise decision geometry

#     h_ab(z) = n_ab^T z + q_ab,     a < b,

# and class energy

#     E_c(z) = - min_{j != c} s_cj(z).

# Training therefore uses the same geometry for two complementary purposes:

# 1. global classification over all seen classes with logits = -E;
# 2. pairwise distribution separation over the trainable current-phase pairs.

# For a trainable pair (a,b), the separator itself provides the discriminative
# axis.  The loss balances both class sides irrespective of sample count and
# separates their projected distributions without prototypes or an arbitrary
# margin:

#     L_side(a,b)
#       = 1/2 [ mean softplus(-h_ab(z_a))
#             + mean softplus( h_ab(z_b)) ]

#     L_order(a,b)
#       = mean_{i,j} softplus(-(h_ab(z_i^a) - h_ab(z_j^b)))

#     L_sep(a,b) = 1/2 [L_side(a,b) + L_order(a,b)].

# At base phase the candidate contains every base-base pair.  At an incremental
# phase it contains exactly the old-new and new-new pairs, so the same objective
# naturally trains only newly introduced discrimination while committed old-old
# geometry remains fixed.

# Historical replay uses a separate preservation objective on class-incident
# old boundary responses.  For replay item i of old class c,

#     r_c(z_i) = [s_cj(z_i)]_{j != c}

# is cached at phase start and compared with the current response while the
# backbone evolves.  This preserves decision-relevant historical coordinates
# without freezing the complete feature vector or using a teacher/prototype.
# """

# from dataclasses import dataclass
# import math
# from typing import Optional

# import torch
# import torch.nn.functional as F

# from models.classifier import ClassifierOutput, GeometryClassifier
# from models.geometry_bank import (
#     BoundaryCandidate,
#     BoundaryGeometryBank,
#     ClassBoundaryResponse,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class GeometryTrainingObjective:
#     """Global classification plus pairwise distribution separation."""

#     total: Tensor
#     classification: Tensor
#     separation: Tensor
#     accuracy: Tensor
#     active_pair_count: int


# @dataclass(frozen=True)
# class HistoricalPreservationObjective:
#     """Decision-coordinate preservation for historical replay."""

#     total: Tensor
#     mean_absolute_drift: Tensor
#     max_absolute_drift: Tensor


# def _weight(name: str, value: float) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0.0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# def _class_weights(weights: Tensor, *, class_count: int, reference: Tensor) -> Tensor:
#     value = torch.as_tensor(weights)
#     if value.device != reference.device:
#         raise ValueError("class_risk_weights and classifier output must share a device")
#     value = value.to(dtype=reference.dtype).flatten()
#     if value.shape != (class_count,):
#         raise ValueError("class_risk_weights must contain one weight per class column")
#     if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
#         raise ValueError("class_risk_weights must be finite and positive")
#     return value


# def _weighted_mean(
#     values: Tensor,
#     targets_local: Tensor,
#     class_risk_weights: Tensor,
#     *,
#     class_count: int,
# ) -> Tensor:
#     if values.ndim != 1 or values.numel() == 0:
#         raise ValueError("values must be non-empty [N]")
#     weights = _class_weights(
#         class_risk_weights,
#         class_count=class_count,
#         reference=values,
#     )
#     return (weights.index_select(0, targets_local) * values).mean()


# def _validate_output(output: ClassifierOutput) -> None:
#     if not isinstance(output, ClassifierOutput):
#         raise TypeError("output must be ClassifierOutput")
#     if (
#         output.energy.ndim != 2
#         or output.energy.size(0) == 0
#         or output.energy.size(1) == 0
#     ):
#         raise ValueError("energy must be non-empty [N,C]")
#     if output.logits.shape != output.energy.shape:
#         raise ValueError("logits and energy must share [N,C]")
#     if not bool(torch.equal(output.logits, -output.energy)):
#         raise ValueError("classifier logits must equal -energy exactly")
#     if not bool(torch.isfinite(output.energy).all()):
#         raise ValueError("classifier output contains NaN/Inf")
#     expected = output.class_ids.index_select(0, output.energy.argmin(dim=1))
#     if not bool(torch.equal(output.prediction, expected)):
#         raise ValueError("prediction must be the minimum-energy class")


# def _require_all_seen_classes(
#     output: ClassifierOutput,
#     geometry_bank: BoundaryGeometryBank,
#     candidate: BoundaryCandidate,
# ) -> tuple[int, ...]:
#     committed = tuple(
#         int(v) for v in geometry_bank.class_ids.detach().cpu().tolist()
#     )
#     new_ids = candidate.new_class_ids
#     if set(committed).intersection(new_ids):
#         raise ValueError("candidate contains already committed classes")
#     expected = committed + new_ids
#     if len(expected) != len(set(expected)):
#         raise RuntimeError("visible class IDs are not unique")
#     actual = tuple(int(v) for v in output.class_ids.detach().cpu().tolist())
#     if set(actual) != set(expected) or len(actual) != len(expected):
#         raise ValueError("training classification must include every seen class")
#     return actual


# def _pairwise_distribution_separation(
#     *,
#     coordinates: Tensor,
#     labels: Tensor,
#     geometry_bank: BoundaryGeometryBank,
#     candidate: BoundaryCandidate,
# ) -> tuple[Tensor, int]:
#     """Separate both sides of every candidate pair represented in this batch.

#     Each pair contributes equally, and each class side contributes equally
#     inside a pair.  Therefore the objective is intrinsically insensitive to a
#     2-vs-200 old/new sample-count imbalance for that boundary.

#     Pairs for which one side is absent from the current batch are skipped; this
#     is required for minibatch training and introduces no pair-specific threshold.
#     The caller receives ``active_pair_count`` for diagnostics.
#     """
#     z = torch.as_tensor(coordinates)
#     if z.ndim != 2 or z.size(0) == 0:
#         raise ValueError("coordinates must be non-empty [N,D]")
#     if z.device != geometry_bank.device or z.dtype != geometry_bank.dtype:
#         raise ValueError("coordinates must share geometry device and dtype")
#     if not bool(torch.isfinite(z).all()):
#         raise ValueError("coordinates contain NaN/Inf")

#     y = torch.as_tensor(labels, device=z.device).flatten().to(dtype=torch.long)
#     if y.shape != (z.size(0),):
#         raise ValueError("labels and coordinates are row-misaligned")

#     candidate.validate_state()
#     if candidate.pair_ids.size(0) == 0:
#         return z.sum() * 0.0, 0

#     pair_geometry = geometry_bank.pair_values(
#         z,
#         pair_ids=candidate.pair_ids,
#         candidate=candidate,
#     )

#     pair_losses: list[Tensor] = []
#     for column, pair_row in enumerate(pair_geometry.pair_ids):
#         left = int(pair_row[0].item())
#         right = int(pair_row[1].item())
#         left_values = pair_geometry.values[y.eq(left), column]
#         right_values = pair_geometry.values[y.eq(right), column]
#         if left_values.numel() == 0 or right_values.numel() == 0:
#             continue

#         # Positive h_ab favors left class; negative h_ab favors right class.
#         side = 0.5 * (
#             F.softplus(-left_values).mean()
#             + F.softplus(right_values).mean()
#         )

#         # Direct distribution ordering along the actual discriminative axis.
#         # Batch sizes in this project are small enough that the exact Cartesian
#         # comparison is inexpensive and avoids sampling another approximation.
#         pairwise_difference = (
#             left_values.unsqueeze(1) - right_values.unsqueeze(0)
#         )
#         order = F.softplus(-pairwise_difference).mean()

#         pair_losses.append(0.5 * (side + order))

#     if not pair_losses:
#         # A differentiable zero is necessary for batches containing only one
#         # side of all candidate pairs. Global CE still trains that batch.
#         return z.sum() * 0.0, 0

#     separation = torch.stack(pair_losses).mean()
#     if not bool(torch.isfinite(separation)):
#         raise RuntimeError("pairwise distribution separation is NaN/Inf")
#     return separation, len(pair_losses)


# def geometry_training_objective(
#     *,
#     output: ClassifierOutput,
#     coordinates: Tensor,
#     labels_global: Tensor,
#     geometry_bank: BoundaryGeometryBank,
#     candidate: BoundaryCandidate,
#     class_risk_weights: Tensor,
#     classification_weight: float = 1.0,
#     separation_weight: float = 1.0,
# ) -> GeometryTrainingObjective:
#     """Train global classification and the current phase's pairwise geometry."""
#     _validate_output(output)
#     if not isinstance(geometry_bank, BoundaryGeometryBank):
#         raise TypeError("geometry_bank must be BoundaryGeometryBank")
#     if not isinstance(candidate, BoundaryCandidate):
#         raise TypeError("candidate must be BoundaryCandidate")
#     geometry_bank.validate_bank_state()
#     candidate.validate_state()
#     visible_ids = _require_all_seen_classes(output, geometry_bank, candidate)

#     z = torch.as_tensor(coordinates)
#     if z.device != output.energy.device:
#         raise ValueError("coordinates and classifier output must share a device")
#     if z.dtype != output.energy.dtype:
#         raise ValueError("coordinates and classifier output must share a dtype")
#     if z.ndim != 2 or z.size(0) != output.energy.size(0):
#         raise ValueError("coordinates must align with classifier rows")

#     labels = torch.as_tensor(labels_global)
#     if labels.device != output.class_ids.device:
#         raise ValueError("labels_global and classifier output must share a device")
#     labels = labels.flatten().to(dtype=torch.long)
#     targets = GeometryClassifier.targets_local(labels, output.class_ids)
#     if targets.numel() != output.energy.size(0):
#         raise ValueError("labels and classifier rows are misaligned")

#     # Protect against accidentally training with a label outside all seen classes.
#     if not set(int(v) for v in labels.detach().cpu().unique().tolist()).issubset(
#         set(visible_ids)
#     ):
#         raise ValueError("labels contain classes outside the visible geometry")

#     class_count = output.energy.size(1)
#     per_sample_ce = F.cross_entropy(output.logits, targets, reduction="none")
#     classification = _weighted_mean(
#         per_sample_ce,
#         targets,
#         class_risk_weights,
#         class_count=class_count,
#     )

#     separation, active_pair_count = _pairwise_distribution_separation(
#         coordinates=z,
#         labels=labels,
#         geometry_bank=geometry_bank,
#         candidate=candidate,
#     )

#     cls_w = _weight("classification_weight", classification_weight)
#     sep_w = _weight("separation_weight", separation_weight)
#     if cls_w == 0.0 and sep_w == 0.0:
#         raise ValueError("at least one training objective weight must be positive")

#     total = cls_w * classification + sep_w * separation
#     if not bool(torch.isfinite(total)):
#         raise RuntimeError("geometry objective is NaN/Inf")

#     accuracy = output.prediction.eq(labels).to(torch.float32).mean()
#     return GeometryTrainingObjective(
#         total=total,
#         classification=classification,
#         separation=separation,
#         accuracy=accuracy,
#         active_pair_count=active_pair_count,
#     )


# def historical_response_preservation_objective(
#     *,
#     current: ClassBoundaryResponse,
#     target_margins: Tensor,
#     target_rival_class_ids: Tensor,
#     weight: float = 1.0,
# ) -> HistoricalPreservationObjective:
#     """Preserve old class-incident decision coordinates during adaptation.

#     ``current`` must be computed from committed old-old geometry only.  Target
#     margins/rival IDs are cached at phase start for the same replay rows.  Rival
#     IDs are checked exactly so the loss cannot silently compare mismatched
#     boundary columns.
#     """
#     if not isinstance(current, ClassBoundaryResponse):
#         raise TypeError("current must be ClassBoundaryResponse")

#     target = torch.as_tensor(target_margins)
#     rivals = torch.as_tensor(target_rival_class_ids)
#     if target.device != current.margins.device or rivals.device != current.margins.device:
#         raise ValueError("historical targets and current response must share a device")
#     target = target.to(dtype=current.margins.dtype)
#     rivals = rivals.to(dtype=torch.long)

#     if target.shape != current.margins.shape:
#         raise ValueError("target historical margins have incompatible shape")
#     if rivals.shape != current.rival_class_ids.shape:
#         raise ValueError("target historical rival IDs have incompatible shape")
#     if not bool(torch.equal(rivals, current.rival_class_ids)):
#         raise ValueError("historical rival columns do not match current geometry response")
#     if not bool(torch.isfinite(target).all()):
#         raise ValueError("historical target margins contain NaN/Inf")

#     absolute_drift = (current.margins - target).abs()
#     mean_drift = absolute_drift.mean()
#     max_drift = absolute_drift.amax()
#     preserve_w = _weight("preservation_weight", weight)
#     total = preserve_w * mean_drift
#     if not bool(torch.isfinite(total)):
#         raise RuntimeError("historical response preservation is NaN/Inf")

#     return HistoricalPreservationObjective(
#         total=total,
#         mean_absolute_drift=mean_drift,
#         max_absolute_drift=max_drift,
#     )


# __all__ = [
#     "GeometryTrainingObjective",
#     "HistoricalPreservationObjective",
#     "geometry_training_objective",
#     "historical_response_preservation_objective",
# ]
