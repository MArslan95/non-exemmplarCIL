from __future__ import annotations

"""Read-only evaluation for one-space NECIL-HSI pairwise decision geometry.

Evaluation follows the deployed classifier exactly.  In addition to accuracy,
CE and decision-cell diagnostics, it reports the quantities required to judge
whether the learned base representation is suitable for incremental learning:

    true-pair violation rate:
        fraction of class-vs-rival relations with s_yj(z) < 0;

    no-cell rate:
        fraction of samples for which min_c E_c(z) > 0;

    minimum true-pair margin:
        min_{j != y} s_yj(z), the weakest true decision relation;

    explicit class-pair boundary diagnostics:
        for each evaluated pair (a,b), report the violation rate and signed
        margin on both class sides of the shared boundary h_ab(z).

The implementation is class-count agnostic.  With four base classes it
evaluates exactly C(4,2)=6 base-base boundaries; with six classes it evaluates
15, and so on.

``cell_fit`` is retained only as a backward-compatible reporting alias for
``relu(E_y)``.  It is not the current training objective.
"""

from contextlib import contextmanager
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.geometry_bank import BoundaryCandidate, BoundaryGeometryBank

Tensor = torch.Tensor


def _as_int(value: object, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be an integer")
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    raise ValueError(f"{name} must be an integer")


def _class_ids(
    values: Sequence[int],
    *,
    name: str = "class_ids",
) -> list[int]:
    ids = [_as_int(value, name) for value in values]
    if not ids or len(ids) != len(set(ids)) or any(value < 0 for value in ids):
        raise ValueError(f"{name} must contain unique non-negative IDs")
    return ids



def _canonical_pairs(class_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Return all unordered class pairs in deterministic classifier order."""
    ids = _class_ids(class_ids)
    return [
        (ids[left], ids[right])
        for left in range(len(ids))
        for right in range(left + 1, len(ids))
    ]



@contextmanager
def _temporary_eval_state(
    model: Any,
    candidate: Optional[BoundaryCandidate],
):
    states = {
        module: bool(module.training)
        for module in model.modules()
    }
    candidate_state = None if candidate is None else bool(candidate.training)
    try:
        model.eval()
        if candidate is not None:
            candidate.eval()
        yield
    finally:
        for module, state in states.items():
            module.training = state
        if candidate is not None and candidate_state is not None:
            candidate.training = candidate_state


def _confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    ids: Sequence[int],
) -> np.ndarray:
    index = {class_id: row for row, class_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if target not in index or prediction not in index:
            raise RuntimeError(
                "confusion input contains an unknown class"
            )
        matrix[index[target], index[prediction]] += 1
    return matrix


def _kappa(matrix: np.ndarray) -> float:
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("confusion matrix is empty")
    observed = float(np.trace(matrix)) / total
    expected = float(
        (matrix.sum(axis=1) * matrix.sum(axis=0)).sum()
    ) / (total * total)
    denominator = 1.0 - expected
    return (
        0.0
        if denominator == 0.0
        else (observed - expected) / denominator
    )


@torch.no_grad()
def evaluate_loader(
    model: Any,
    loader: Any,
    *,
    class_ids: Sequence[int],
    device: str | torch.device,
    target_class_ids: Optional[Sequence[int]] = None,
    candidate: Optional[BoundaryCandidate] = None,
) -> Dict[str, Any]:
    """Classify against all ``class_ids`` and report the requested targets."""
    ids = _class_ids(class_ids)
    target_ids = (
        ids
        if target_class_ids is None
        else _class_ids(target_class_ids, name="target_class_ids")
    )
    if not set(target_ids).issubset(ids):
        raise ValueError(
            "target_class_ids must be a subset of class_ids"
        )

    dev = torch.device(device)
    if torch.device(model.device) != dev:
        raise ValueError("model and evaluation device disagree")
    if candidate is not None:
        candidate.validate_state()

    bank = getattr(model, "geometry_bank", None)
    if not isinstance(bank, BoundaryGeometryBank):
        raise TypeError("model geometry_bank must be BoundaryGeometryBank")
    dtype = bank.dtype

    total = 0
    correct = 0
    ce_sum = 0.0
    true_cell_violation_sum = 0.0
    true_energy_sum = 0.0
    rival_energy_sum = 0.0
    margin_sum = 0.0
    min_true_pair_margin_sum = 0.0

    true_inside_count = 0
    rival_inside_count = 0
    no_cell_count = 0
    pair_violation_count = 0
    pair_relation_count = 0

    has_rivals = len(ids) > 1

    # Explicit boundary diagnostics are evaluated for every unordered pair of
    # target classes.  This is independent of the classifier column count and
    # remains valid for base-4, base-6 and cumulative incremental evaluation.
    diagnostic_pairs = _canonical_pairs(target_ids)
    pair_accumulators: dict[tuple[int, int], Dict[str, Any]] = {
        pair: {
            "left_count": 0,
            "right_count": 0,
            "left_violation_count": 0,
            "right_violation_count": 0,
            "left_margin_sum": 0.0,
            "right_margin_sum": 0.0,
            "left_min_margin": float("inf"),
            "right_min_margin": float("inf"),
        }
        for pair in diagnostic_pairs
    }

    class_total = {class_id: 0 for class_id in target_ids}
    class_correct = {class_id: 0 for class_id in target_ids}
    class_ce = {class_id: 0.0 for class_id in target_ids}
    class_cell_violation = {class_id: 0.0 for class_id in target_ids}
    class_inside = {class_id: 0 for class_id in target_ids}
    class_rival_inside = {class_id: 0 for class_id in target_ids}
    class_no_cell = {class_id: 0 for class_id in target_ids}
    class_pair_violation = {class_id: 0 for class_id in target_ids}
    class_pair_relation = {class_id: 0 for class_id in target_ids}
    class_min_pair_margin = {class_id: 0.0 for class_id in target_ids}
    class_true_energy = {class_id: 0.0 for class_id in target_ids}
    class_rival_energy = {class_id: 0.0 for class_id in target_ids}
    class_margin = {class_id: 0.0 for class_id in target_ids}

    targets_all: list[Tensor] = []
    predictions_all: list[Tensor] = []

    with _temporary_eval_state(model, candidate):
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("evaluation batches must be mappings")
            required = {"image", "raw_center_spectrum", "label"}
            missing = required - set(batch)
            if missing:
                raise KeyError(f"evaluation batch lacks {sorted(missing)}")

            patch = torch.as_tensor(
                batch["image"], device=dev, dtype=dtype
            )
            spectrum = torch.as_tensor(
                batch["raw_center_spectrum"],
                device=dev,
                dtype=dtype,
            )
            labels = torch.as_tensor(
                batch["label"], device=dev
            ).flatten()
            if labels.dtype == torch.bool or labels.is_complex():
                raise RuntimeError("evaluation labels must be integer IDs")
            if torch.is_floating_point(labels):
                if not bool(torch.isfinite(labels).all()) or not bool(
                    labels.eq(labels.round()).all()
                ):
                    raise RuntimeError(
                        "evaluation labels must contain finite integer IDs"
                    )
            labels = labels.to(torch.long)

            observed = set(
                int(value)
                for value in labels.unique().detach().cpu().tolist()
            )
            outside = sorted(observed - set(target_ids))
            if outside:
                raise RuntimeError(
                    "evaluation loader contains labels outside target "
                    f"classes: {outside}"
                )

            result = model(
                patch,
                center_spectrum=spectrum,
                class_ids=ids,
                candidate=candidate,
                return_aux=False,
            )
            representation = result.representation.coordinates
            output = result.classification

            actual_ids = [
                int(value)
                for value in output.class_ids.detach().cpu().tolist()
            ]
            if actual_ids != ids:
                raise RuntimeError(
                    "classifier columns do not match requested classes"
                )

            targets = model.classifier.targets_local(
                labels,
                output.class_ids,
            )
            rows = torch.arange(labels.numel(), device=dev)
            true_energy = output.energy[rows, targets]
            inside = true_energy <= 0
            no_cell = output.energy.amin(dim=1) > 0

            per_ce = F.cross_entropy(
                output.logits,
                targets,
                reduction="none",
            )
            # Compatibility diagnostic only; not the current training loss.
            per_cell_violation = F.relu(true_energy)

            ce_sum += float(per_ce.sum().item())
            true_cell_violation_sum += float(
                per_cell_violation.sum().item()
            )
            true_energy_sum += float(true_energy.sum().item())
            true_inside_count += int(inside.sum().item())
            no_cell_count += int(no_cell.sum().item())

            pair_margins = None
            min_pair_margin = None
            pair_violations = None
            rival_energy = None
            margin = None
            rival_inside = None

            if has_rivals:
                pair_margins = model.true_pair_margins(
                    representation,
                    labels,
                    class_ids=ids,
                    candidate=candidate,
                )
                if pair_margins.shape != (
                    labels.numel(),
                    len(ids) - 1,
                ):
                    raise RuntimeError(
                        "true pair margins have an invalid shape"
                    )
                pair_violations = pair_margins < 0
                min_pair_margin = pair_margins.amin(dim=1)

                pair_violation_count += int(
                    pair_violations.sum().item()
                )
                pair_relation_count += int(pair_violations.numel())
                min_true_pair_margin_sum += float(
                    min_pair_margin.sum().item()
                )

                target_mask = F.one_hot(
                    targets, num_classes=len(ids)
                ).to(torch.bool)
                rival_energy = output.energy.masked_fill(
                    target_mask, torch.inf
                ).amin(dim=1)
                rival_inside = rival_energy < 0
                if bool((inside & rival_inside).any()):
                    raise RuntimeError(
                        "pairwise geometry invariant violated: "
                        "a sample lies in two strict class interiors"
                    )
                margin = rival_energy - true_energy
                rival_inside_count += int(rival_inside.sum().item())
                rival_energy_sum += float(rival_energy.sum().item())
                margin_sum += float(margin.sum().item())

            if diagnostic_pairs:
                pair_geometry = model.pair_values(
                    representation,
                    pair_ids=diagnostic_pairs,
                    candidate=candidate,
                )
                expected_pair_ids = torch.tensor(
                    diagnostic_pairs,
                    device=pair_geometry.pair_ids.device,
                    dtype=torch.long,
                )
                if pair_geometry.values.shape != (
                    labels.numel(),
                    len(diagnostic_pairs),
                ):
                    raise RuntimeError(
                        "pairwise geometry values have an invalid shape"
                    )
                if not bool(torch.equal(pair_geometry.pair_ids, expected_pair_ids)):
                    raise RuntimeError(
                        "pairwise geometry returned pairs in an unexpected order"
                    )

                for pair_index, pair in enumerate(diagnostic_pairs):
                    left_id, right_id = pair
                    h = pair_geometry.values[:, pair_index]
                    left_mask = labels.eq(left_id)
                    right_mask = labels.eq(right_id)
                    left_count = int(left_mask.sum().item())
                    right_count = int(right_mask.sum().item())
                    stats = pair_accumulators[pair]

                    if left_count:
                        left_margin = h[left_mask]
                        stats["left_count"] += left_count
                        stats["left_violation_count"] += int(
                            (left_margin < 0).sum().item()
                        )
                        stats["left_margin_sum"] += float(
                            left_margin.sum().item()
                        )
                        stats["left_min_margin"] = min(
                            float(stats["left_min_margin"]),
                            float(left_margin.amin().item()),
                        )

                    if right_count:
                        # Positive h_ab belongs to the lower/canonical class a.
                        # Therefore the class-b oriented margin is -h_ab.
                        right_margin = -h[right_mask]
                        stats["right_count"] += right_count
                        stats["right_violation_count"] += int(
                            (right_margin < 0).sum().item()
                        )
                        stats["right_margin_sum"] += float(
                            right_margin.sum().item()
                        )
                        stats["right_min_margin"] = min(
                            float(stats["right_min_margin"]),
                            float(right_margin.amin().item()),
                        )

            prediction = output.prediction
            batch_count = int(labels.numel())
            if batch_count <= 0:
                raise RuntimeError("evaluation produced an empty batch")
            total += batch_count
            correct += int(prediction.eq(labels).sum().item())
            targets_all.append(labels.detach().cpu())
            predictions_all.append(prediction.detach().cpu())

            for class_id in target_ids:
                class_mask = labels.eq(class_id)
                count = int(class_mask.sum().item())
                if count == 0:
                    continue

                class_total[class_id] += count
                class_correct[class_id] += int(
                    prediction[class_mask]
                    .eq(labels[class_mask])
                    .sum()
                    .item()
                )
                class_ce[class_id] += float(
                    per_ce[class_mask].sum().item()
                )
                class_cell_violation[class_id] += float(
                    per_cell_violation[class_mask].sum().item()
                )
                class_inside[class_id] += int(
                    inside[class_mask].sum().item()
                )
                class_no_cell[class_id] += int(
                    no_cell[class_mask].sum().item()
                )
                class_true_energy[class_id] += float(
                    true_energy[class_mask].sum().item()
                )

                if has_rivals:
                    assert (
                        pair_violations is not None
                        and min_pair_margin is not None
                        and rival_inside is not None
                        and rival_energy is not None
                        and margin is not None
                    )
                    class_pair_violation[class_id] += int(
                        pair_violations[class_mask].sum().item()
                    )
                    class_pair_relation[class_id] += (
                        count * (len(ids) - 1)
                    )
                    class_min_pair_margin[class_id] += float(
                        min_pair_margin[class_mask].sum().item()
                    )
                    class_rival_inside[class_id] += int(
                        rival_inside[class_mask].sum().item()
                    )
                    class_rival_energy[class_id] += float(
                        rival_energy[class_mask].sum().item()
                    )
                    class_margin[class_id] += float(
                        margin[class_mask].sum().item()
                    )

    if total == 0:
        raise RuntimeError("evaluation loader is empty")

    missing_targets = [
        class_id
        for class_id in target_ids
        if class_total[class_id] == 0
    ]
    if missing_targets:
        raise RuntimeError(
            "evaluation split is missing target classes: "
            f"{missing_targets}"
        )

    per_acc = {
        class_id: class_correct[class_id] / class_total[class_id]
        for class_id in target_ids
    }
    per_ce_mean = {
        class_id: class_ce[class_id] / class_total[class_id]
        for class_id in target_ids
    }
    per_cell_violation_mean = {
        class_id: (
            class_cell_violation[class_id] / class_total[class_id]
        )
        for class_id in target_ids
    }
    per_cov = {
        class_id: class_inside[class_id] / class_total[class_id]
        for class_id in target_ids
    }
    per_inv = {
        class_id: (
            class_rival_inside[class_id] / class_total[class_id]
            if has_rivals else 0.0
        )
        for class_id in target_ids
    }
    per_no_cell = {
        class_id: class_no_cell[class_id] / class_total[class_id]
        for class_id in target_ids
    }
    per_pair_violation = {
        class_id: (
            class_pair_violation[class_id]
            / class_pair_relation[class_id]
            if has_rivals else 0.0
        )
        for class_id in target_ids
    }
    per_min_pair_margin = {
        class_id: (
            class_min_pair_margin[class_id] / class_total[class_id]
            if has_rivals else None
        )
        for class_id in target_ids
    }
    per_true = {
        class_id: (
            class_true_energy[class_id] / class_total[class_id]
        )
        for class_id in target_ids
    }
    per_rival = {
        class_id: (
            class_rival_energy[class_id] / class_total[class_id]
            if has_rivals else None
        )
        for class_id in target_ids
    }
    per_margin = {
        class_id: (
            class_margin[class_id] / class_total[class_id]
            if has_rivals else None
        )
        for class_id in target_ids
    }

    def macro(values: Mapping[int, float]) -> float:
        return (
            sum(float(values[class_id]) for class_id in target_ids)
            / len(target_ids)
        )

    macro_min_pair_margin = (
        macro({
            class_id: float(per_min_pair_margin[class_id])
            for class_id in target_ids
        })
        if has_rivals else None
    )
    macro_rival = (
        macro({
            class_id: float(per_rival[class_id])
            for class_id in target_ids
        })
        if has_rivals else None
    )
    macro_margin = (
        macro({
            class_id: float(per_margin[class_id])
            for class_id in target_ids
        })
        if has_rivals else None
    )

    target_np = torch.cat(targets_all).numpy()
    prediction_np = torch.cat(predictions_all).numpy()
    matrix = _confusion(target_np, prediction_np, ids)

    pairwise_boundary_metrics: Dict[str, Any] = {}
    for left_id, right_id in diagnostic_pairs:
        stats = pair_accumulators[(left_id, right_id)]
        left_count = int(stats["left_count"])
        right_count = int(stats["right_count"])
        if left_count <= 0 or right_count <= 0:
            raise RuntimeError(
                "pairwise diagnostics require both evaluated classes for "
                f"pair ({left_id},{right_id})"
            )

        left_mean = float(stats["left_margin_sum"]) / left_count
        right_mean = float(stats["right_margin_sum"]) / right_count
        left_violation = int(stats["left_violation_count"]) / left_count
        right_violation = int(stats["right_violation_count"]) / right_count
        combined_count = left_count + right_count
        combined_violation = (
            int(stats["left_violation_count"])
            + int(stats["right_violation_count"])
        ) / combined_count

        pairwise_boundary_metrics[f"{left_id}-{right_id}"] = {
            "left_class_id": left_id,
            "right_class_id": right_id,
            "left_count": left_count,
            "right_count": right_count,
            "left_violation_rate": left_violation,
            "right_violation_rate": right_violation,
            "combined_violation_rate": combined_violation,
            "left_mean_oriented_margin": left_mean,
            "right_mean_oriented_margin": right_mean,
            "left_minimum_oriented_margin": float(
                stats["left_min_margin"]
            ),
            "right_minimum_oriented_margin": float(
                stats["right_min_margin"]
            ),
            # Since the right oriented margin is -h_ab, this equals
            # mean[h_ab(Z_left)] - mean[h_ab(Z_right)].
            "mean_distribution_order_gap": left_mean + right_mean,
            "minimum_side_mean_margin": min(left_mean, right_mean),
        }

    if pairwise_boundary_metrics:
        worst_pair_key = max(
            pairwise_boundary_metrics,
            key=lambda key: pairwise_boundary_metrics[key][
                "combined_violation_rate"
            ],
        )
        weakest_margin_pair_key = min(
            pairwise_boundary_metrics,
            key=lambda key: pairwise_boundary_metrics[key][
                "minimum_side_mean_margin"
            ],
        )
        pairwise_boundary_summary: Dict[str, Any] = {
            "pair_count": len(pairwise_boundary_metrics),
            "worst_violation_pair": worst_pair_key,
            "worst_combined_violation_rate": float(
                pairwise_boundary_metrics[worst_pair_key][
                    "combined_violation_rate"
                ]
            ),
            "weakest_mean_margin_pair": weakest_margin_pair_key,
            "weakest_minimum_side_mean_margin": float(
                pairwise_boundary_metrics[weakest_margin_pair_key][
                    "minimum_side_mean_margin"
                ]
            ),
        }
    else:
        pairwise_boundary_summary = {
            "pair_count": 0,
            "worst_violation_pair": None,
            "worst_combined_violation_rate": 0.0,
            "weakest_mean_margin_pair": None,
            "weakest_minimum_side_mean_margin": None,
        }

    return {
        "classification": ce_sum / total,
        "macro_classification": macro(per_ce_mean),

        # Backward-compatible diagnostic aliases.
        "true_cell_violation": true_cell_violation_sum / total,
        "macro_true_cell_violation": macro(per_cell_violation_mean),
        "per_class_true_cell_violation": per_cell_violation_mean,
        "cell_fit": true_cell_violation_sum / total,
        "macro_cell_fit": macro(per_cell_violation_mean),
        "per_class_cell_fit": per_cell_violation_mean,

        "overall_accuracy": correct / total,
        "accuracy": correct / total,
        "balanced_accuracy": macro(per_acc),
        "minimum_class_accuracy": min(per_acc.values()),
        "kappa": _kappa(matrix),
        "confusion_matrix": matrix.tolist(),

        "evaluated_class_ids": list(ids),
        "target_class_ids": list(target_ids),
        "class_counts": {
            class_id: class_total[class_id]
            for class_id in target_ids
        },
        "per_class_accuracy": per_acc,

        "true_cell_coverage": true_inside_count / total,
        "macro_true_cell_coverage": macro(per_cov),
        "per_class_true_cell_coverage": per_cov,

        "rival_cell_invasion_rate": (
            rival_inside_count / total if has_rivals else 0.0
        ),
        "macro_rival_cell_invasion_rate": macro(per_inv),
        "per_class_rival_cell_invasion_rate": per_inv,

        "no_cell_rate": no_cell_count / total,
        "macro_no_cell_rate": macro(per_no_cell),
        "per_class_no_cell_rate": per_no_cell,

        "true_pair_violation_rate": (
            pair_violation_count / pair_relation_count
            if pair_relation_count else 0.0
        ),
        "macro_true_pair_violation_rate": macro(per_pair_violation),
        "per_class_true_pair_violation_rate": per_pair_violation,

        "pairwise_boundary_metrics": pairwise_boundary_metrics,
        "pairwise_boundary_summary": pairwise_boundary_summary,

        "mean_minimum_true_pair_margin": (
            min_true_pair_margin_sum / total if has_rivals else None
        ),
        "macro_mean_minimum_true_pair_margin": macro_min_pair_margin,
        "per_class_mean_minimum_true_pair_margin": per_min_pair_margin,

        "mean_true_energy": true_energy_sum / total,
        "macro_mean_true_energy": macro(per_true),
        "per_class_mean_true_energy": per_true,

        "mean_nearest_rival_energy": (
            rival_energy_sum / total if has_rivals else None
        ),
        "macro_mean_nearest_rival_energy": macro_rival,
        "per_class_mean_nearest_rival_energy": per_rival,

        "mean_decision_margin": (
            margin_sum / total if has_rivals else None
        ),
        "macro_mean_decision_margin": macro_margin,
        "per_class_mean_decision_margin": per_margin,

        "strict_cell_conflict_rate": 0.0,
        "energy_convention": {
            "class_cell": (
                "E_c(z) <= 0 iff all pairwise boundaries support class c"
            ),
            "class_score": (
                "E_c(z) = -minimum oriented pairwise signed distance"
            ),
            "decision": "argmin_c E_c(z)",
            "pair_violation": "s_yj(z) < 0",
            "pair_boundary_orientation": (
                "for canonical pair (a,b) with a<b: h_ab>0 supports a "
                "and h_ab<0 supports b"
            ),
            "no_cell": "min_c E_c(z) > 0",
            "strict_interior_overlap": (
                "impossible by shared opposite pair boundaries"
            ),
        },
    }


def summarize_class_group(
    metrics: Mapping[str, Any],
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    """Summarize a class group from cumulative seen-class evaluation."""
    ids = _class_ids(class_ids)

    fields = {
        "balanced_accuracy": "per_class_accuracy",
        "macro_true_cell_coverage": "per_class_true_cell_coverage",
        "macro_rival_cell_invasion_rate": (
            "per_class_rival_cell_invasion_rate"
        ),
        "macro_no_cell_rate": "per_class_no_cell_rate",
        "macro_true_pair_violation_rate": (
            "per_class_true_pair_violation_rate"
        ),
        "macro_true_cell_violation": (
            "per_class_true_cell_violation"
        ),
        "macro_mean_minimum_true_pair_margin": (
            "per_class_mean_minimum_true_pair_margin"
        ),
        "macro_mean_decision_margin": (
            "per_class_mean_decision_margin"
        ),
    }

    result: Dict[str, Any] = {"class_ids": ids}
    for output_name, source_name in fields.items():
        source = metrics.get(source_name)
        if not isinstance(source, Mapping):
            raise ValueError(f"metrics lacks {source_name}")

        values: list[float] = []
        for class_id in ids:
            raw = (
                source[class_id]
                if class_id in source
                else source.get(str(class_id))
            )
            if raw is None:
                raise ValueError(
                    f"{source_name} lacks class {class_id}"
                )
            values.append(float(raw))
        result[output_name] = sum(values) / len(values)

    accuracy_source = metrics.get("per_class_accuracy")
    if not isinstance(accuracy_source, Mapping):
        raise ValueError("metrics lacks per_class_accuracy")
    accuracies = [
        float(
            accuracy_source[class_id]
            if class_id in accuracy_source
            else accuracy_source[str(class_id)]
        )
        for class_id in ids
    ]
    result["minimum_class_accuracy"] = min(accuracies)

    # Compatibility alias; this is not a training objective.
    result["macro_cell_fit"] = result["macro_true_cell_violation"]
    return result


def old_new_summary(
    metrics: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
) -> Dict[str, Any]:
    old = summarize_class_group(metrics, old_class_ids)
    new = summarize_class_group(metrics, new_class_ids)

    old_accuracy = float(old["balanced_accuracy"])
    new_accuracy = float(new["balanced_accuracy"])
    denominator = old_accuracy + new_accuracy
    harmonic = (
        0.0
        if denominator == 0.0
        else 2.0 * old_accuracy * new_accuracy / denominator
    )
    return {
        "old": old,
        "new": new,
        "harmonic_old_new_accuracy": harmonic,
    }


def boundary_preservation_delta(
    previous_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
) -> Dict[str, Any]:
    """Measure deployed historical-class change between finalized phases."""
    ids = _class_ids(old_class_ids)

    def per_class_delta(field: str) -> Dict[int, float]:
        previous = previous_metrics.get(field)
        current = current_metrics.get(field)
        if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
            raise ValueError(f"both metric sets must contain {field}")

        result: Dict[int, float] = {}
        for class_id in ids:
            old_value = (
                previous[class_id]
                if class_id in previous
                else previous.get(str(class_id))
            )
            new_value = (
                current[class_id]
                if class_id in current
                else current.get(str(class_id))
            )
            if old_value is None or new_value is None:
                raise ValueError(
                    f"{field} lacks historical class {class_id}"
                )
            result[class_id] = float(new_value) - float(old_value)
        return result

    accuracy = per_class_delta("per_class_accuracy")
    coverage = per_class_delta("per_class_true_cell_coverage")
    pair_violation = per_class_delta(
        "per_class_true_pair_violation_rate"
    )
    no_cell = per_class_delta("per_class_no_cell_rate")
    invasion = per_class_delta(
        "per_class_rival_cell_invasion_rate"
    )
    margin = per_class_delta("per_class_mean_decision_margin")

    def mean(values: Mapping[int, float]) -> float:
        return sum(values.values()) / len(values)

    return {
        "class_ids": ids,
        "per_class_accuracy_delta": accuracy,
        "per_class_cell_coverage_delta": coverage,
        "per_class_pair_violation_delta": pair_violation,
        "per_class_no_cell_rate_delta": no_cell,
        "per_class_rival_invasion_delta": invasion,
        "per_class_decision_margin_delta": margin,
        "mean_accuracy_delta": mean(accuracy),
        "mean_cell_coverage_delta": mean(coverage),
        "mean_pair_violation_delta": mean(pair_violation),
        "mean_no_cell_rate_delta": mean(no_cell),
        "mean_rival_invasion_delta": mean(invasion),
        "mean_decision_margin_delta": mean(margin),
    }


@torch.no_grad()
def geometry_diagnostics(
    model: Any,
    class_ids: Sequence[int],
    *,
    candidate: Optional[BoundaryCandidate] = None,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Report the committed structural pairwise geometry."""
    if candidate is not None:
        raise ValueError(
            "structural diagnostics require committed geometry"
        )

    ids = _class_ids(class_ids)
    bank = model.geometry_bank
    if not isinstance(bank, BoundaryGeometryBank):
        raise TypeError(
            "model geometry is not BoundaryGeometryBank"
        )
    bank.validate_bank_state()

    committed = [
        int(value)
        for value in bank.class_ids.detach().cpu().tolist()
    ]
    if committed != ids:
        raise RuntimeError(
            "committed geometry classes do not match diagnostics classes"
        )

    names = list(target_names or [])
    pairs: list[Dict[str, Any]] = []
    for raw_pair in bank.pair_ids.detach().cpu().tolist():
        left, right = map(int, raw_pair)
        geometry = bank.get_pair_geometry(left, right)
        normal = geometry["normal"]
        offset = geometry["offset"]
        pairs.append(
            {
                "left_class_id": left,
                "right_class_id": right,
                "left_class_name": (
                    names[left]
                    if left < len(names)
                    else f"Class {left + 1}"
                ),
                "right_class_name": (
                    names[right]
                    if right < len(names)
                    else f"Class {right + 1}"
                ),
                "normal_norm": float(
                    torch.linalg.vector_norm(normal).item()
                ),
                "offset": float(offset.item()),
                "shared_boundary": True,
                "strict_interior_overlap": False,
            }
        )

    expected_pairs = len(ids) * (len(ids) - 1) // 2
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            "committed pairwise geometry is incomplete"
        )

    return {
        "class_ids": ids,
        "class_count": len(ids),
        "pair_count": len(pairs),
        "expected_pair_count": expected_pairs,
        "representation_dim": int(bank.representation_dim),
        "strict_interior_overlap": "impossible by construction",
        "pairs": pairs,
    }


__all__ = [
    "boundary_preservation_delta",
    "evaluate_loader",
    "geometry_diagnostics",
    "old_new_summary",
    "summarize_class_group",
]











# from __future__ import annotations

# """Read-only evaluation for one-space NECIL-HSI pairwise decision geometry.

# Evaluation follows the deployed classifier exactly.  In addition to accuracy,
# CE and decision-cell diagnostics, it reports the quantities required to judge
# whether the learned base representation is suitable for incremental learning:

#     true-pair violation rate:
#         fraction of class-vs-rival relations with s_yj(z) < 0;

#     no-cell rate:
#         fraction of samples for which min_c E_c(z) > 0;

#     minimum true-pair margin:
#         min_{j != y} s_yj(z), the weakest true decision relation.

# ``cell_fit`` is retained only as a backward-compatible reporting alias for
# ``relu(E_y)``.  It is not the current training objective.
# """

# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch
# import torch.nn.functional as F

# from models.geometry_bank import BoundaryCandidate, BoundaryGeometryBank

# Tensor = torch.Tensor


# def _as_int(value: object, name: str) -> int:
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer")
#         value = value.item()
#     if isinstance(value, bool):
#         raise ValueError(f"{name} must be an integer")
#     if isinstance(value, Integral):
#         return int(value)
#     if isinstance(value, Real):
#         number = float(value)
#         if math.isfinite(number) and number.is_integer():
#             return int(number)
#     raise ValueError(f"{name} must be an integer")


# def _class_ids(
#     values: Sequence[int],
#     *,
#     name: str = "class_ids",
# ) -> list[int]:
#     ids = [_as_int(value, name) for value in values]
#     if not ids or len(ids) != len(set(ids)) or any(value < 0 for value in ids):
#         raise ValueError(f"{name} must contain unique non-negative IDs")
#     return ids


# @contextmanager
# def _temporary_eval_state(
#     model: Any,
#     candidate: Optional[BoundaryCandidate],
# ):
#     states = {
#         module: bool(module.training)
#         for module in model.modules()
#     }
#     candidate_state = None if candidate is None else bool(candidate.training)
#     try:
#         model.eval()
#         if candidate is not None:
#             candidate.eval()
#         yield
#     finally:
#         for module, state in states.items():
#             module.training = state
#         if candidate is not None and candidate_state is not None:
#             candidate.training = candidate_state


# def _confusion(
#     targets: np.ndarray,
#     predictions: np.ndarray,
#     ids: Sequence[int],
# ) -> np.ndarray:
#     index = {class_id: row for row, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
#     for target, prediction in zip(targets.tolist(), predictions.tolist()):
#         if target not in index or prediction not in index:
#             raise RuntimeError(
#                 "confusion input contains an unknown class"
#             )
#         matrix[index[target], index[prediction]] += 1
#     return matrix


# def _kappa(matrix: np.ndarray) -> float:
#     total = float(matrix.sum())
#     if total <= 0:
#         raise ValueError("confusion matrix is empty")
#     observed = float(np.trace(matrix)) / total
#     expected = float(
#         (matrix.sum(axis=1) * matrix.sum(axis=0)).sum()
#     ) / (total * total)
#     denominator = 1.0 - expected
#     return (
#         0.0
#         if denominator == 0.0
#         else (observed - expected) / denominator
#     )


# @torch.no_grad()
# def evaluate_loader(
#     model: Any,
#     loader: Any,
#     *,
#     class_ids: Sequence[int],
#     device: str | torch.device,
#     target_class_ids: Optional[Sequence[int]] = None,
#     candidate: Optional[BoundaryCandidate] = None,
# ) -> Dict[str, Any]:
#     """Classify against all ``class_ids`` and report the requested targets."""
#     ids = _class_ids(class_ids)
#     target_ids = (
#         ids
#         if target_class_ids is None
#         else _class_ids(target_class_ids, name="target_class_ids")
#     )
#     if not set(target_ids).issubset(ids):
#         raise ValueError(
#             "target_class_ids must be a subset of class_ids"
#         )

#     dev = torch.device(device)
#     if torch.device(model.device) != dev:
#         raise ValueError("model and evaluation device disagree")
#     if candidate is not None:
#         candidate.validate_state()

#     bank = getattr(model, "geometry_bank", None)
#     if not isinstance(bank, BoundaryGeometryBank):
#         raise TypeError("model geometry_bank must be BoundaryGeometryBank")
#     dtype = bank.dtype

#     total = 0
#     correct = 0
#     ce_sum = 0.0
#     true_cell_violation_sum = 0.0
#     true_energy_sum = 0.0
#     rival_energy_sum = 0.0
#     margin_sum = 0.0
#     min_true_pair_margin_sum = 0.0

#     true_inside_count = 0
#     rival_inside_count = 0
#     no_cell_count = 0
#     pair_violation_count = 0
#     pair_relation_count = 0

#     has_rivals = len(ids) > 1

#     class_total = {class_id: 0 for class_id in target_ids}
#     class_correct = {class_id: 0 for class_id in target_ids}
#     class_ce = {class_id: 0.0 for class_id in target_ids}
#     class_cell_violation = {class_id: 0.0 for class_id in target_ids}
#     class_inside = {class_id: 0 for class_id in target_ids}
#     class_rival_inside = {class_id: 0 for class_id in target_ids}
#     class_no_cell = {class_id: 0 for class_id in target_ids}
#     class_pair_violation = {class_id: 0 for class_id in target_ids}
#     class_pair_relation = {class_id: 0 for class_id in target_ids}
#     class_min_pair_margin = {class_id: 0.0 for class_id in target_ids}
#     class_true_energy = {class_id: 0.0 for class_id in target_ids}
#     class_rival_energy = {class_id: 0.0 for class_id in target_ids}
#     class_margin = {class_id: 0.0 for class_id in target_ids}

#     targets_all: list[Tensor] = []
#     predictions_all: list[Tensor] = []

#     with _temporary_eval_state(model, candidate):
#         for batch in loader:
#             if not isinstance(batch, Mapping):
#                 raise TypeError("evaluation batches must be mappings")
#             required = {"image", "raw_center_spectrum", "label"}
#             missing = required - set(batch)
#             if missing:
#                 raise KeyError(f"evaluation batch lacks {sorted(missing)}")

#             patch = torch.as_tensor(
#                 batch["image"], device=dev, dtype=dtype
#             )
#             spectrum = torch.as_tensor(
#                 batch["raw_center_spectrum"],
#                 device=dev,
#                 dtype=dtype,
#             )
#             labels = torch.as_tensor(
#                 batch["label"], device=dev
#             ).flatten()
#             if labels.dtype == torch.bool or labels.is_complex():
#                 raise RuntimeError("evaluation labels must be integer IDs")
#             if torch.is_floating_point(labels):
#                 if not bool(torch.isfinite(labels).all()) or not bool(
#                     labels.eq(labels.round()).all()
#                 ):
#                     raise RuntimeError(
#                         "evaluation labels must contain finite integer IDs"
#                     )
#             labels = labels.to(torch.long)

#             observed = set(
#                 int(value)
#                 for value in labels.unique().detach().cpu().tolist()
#             )
#             outside = sorted(observed - set(target_ids))
#             if outside:
#                 raise RuntimeError(
#                     "evaluation loader contains labels outside target "
#                     f"classes: {outside}"
#                 )

#             result = model(
#                 patch,
#                 center_spectrum=spectrum,
#                 class_ids=ids,
#                 candidate=candidate,
#                 return_aux=False,
#             )
#             representation = result.representation.coordinates
#             output = result.classification

#             actual_ids = [
#                 int(value)
#                 for value in output.class_ids.detach().cpu().tolist()
#             ]
#             if actual_ids != ids:
#                 raise RuntimeError(
#                     "classifier columns do not match requested classes"
#                 )

#             targets = model.classifier.targets_local(
#                 labels,
#                 output.class_ids,
#             )
#             rows = torch.arange(labels.numel(), device=dev)
#             true_energy = output.energy[rows, targets]
#             inside = true_energy <= 0
#             no_cell = output.energy.amin(dim=1) > 0

#             per_ce = F.cross_entropy(
#                 output.logits,
#                 targets,
#                 reduction="none",
#             )
#             # Compatibility diagnostic only; not the current training loss.
#             per_cell_violation = F.relu(true_energy)

#             ce_sum += float(per_ce.sum().item())
#             true_cell_violation_sum += float(
#                 per_cell_violation.sum().item()
#             )
#             true_energy_sum += float(true_energy.sum().item())
#             true_inside_count += int(inside.sum().item())
#             no_cell_count += int(no_cell.sum().item())

#             pair_margins = None
#             min_pair_margin = None
#             pair_violations = None
#             rival_energy = None
#             margin = None
#             rival_inside = None

#             if has_rivals:
#                 pair_margins = model.true_pair_margins(
#                     representation,
#                     labels,
#                     class_ids=ids,
#                     candidate=candidate,
#                 )
#                 if pair_margins.shape != (
#                     labels.numel(),
#                     len(ids) - 1,
#                 ):
#                     raise RuntimeError(
#                         "true pair margins have an invalid shape"
#                     )
#                 pair_violations = pair_margins < 0
#                 min_pair_margin = pair_margins.amin(dim=1)

#                 pair_violation_count += int(
#                     pair_violations.sum().item()
#                 )
#                 pair_relation_count += int(pair_violations.numel())
#                 min_true_pair_margin_sum += float(
#                     min_pair_margin.sum().item()
#                 )

#                 target_mask = F.one_hot(
#                     targets, num_classes=len(ids)
#                 ).to(torch.bool)
#                 rival_energy = output.energy.masked_fill(
#                     target_mask, torch.inf
#                 ).amin(dim=1)
#                 rival_inside = rival_energy < 0
#                 if bool((inside & rival_inside).any()):
#                     raise RuntimeError(
#                         "pairwise geometry invariant violated: "
#                         "a sample lies in two strict class interiors"
#                     )
#                 margin = rival_energy - true_energy
#                 rival_inside_count += int(rival_inside.sum().item())
#                 rival_energy_sum += float(rival_energy.sum().item())
#                 margin_sum += float(margin.sum().item())

#             prediction = output.prediction
#             batch_count = int(labels.numel())
#             if batch_count <= 0:
#                 raise RuntimeError("evaluation produced an empty batch")
#             total += batch_count
#             correct += int(prediction.eq(labels).sum().item())
#             targets_all.append(labels.detach().cpu())
#             predictions_all.append(prediction.detach().cpu())

#             for class_id in target_ids:
#                 class_mask = labels.eq(class_id)
#                 count = int(class_mask.sum().item())
#                 if count == 0:
#                     continue

#                 class_total[class_id] += count
#                 class_correct[class_id] += int(
#                     prediction[class_mask]
#                     .eq(labels[class_mask])
#                     .sum()
#                     .item()
#                 )
#                 class_ce[class_id] += float(
#                     per_ce[class_mask].sum().item()
#                 )
#                 class_cell_violation[class_id] += float(
#                     per_cell_violation[class_mask].sum().item()
#                 )
#                 class_inside[class_id] += int(
#                     inside[class_mask].sum().item()
#                 )
#                 class_no_cell[class_id] += int(
#                     no_cell[class_mask].sum().item()
#                 )
#                 class_true_energy[class_id] += float(
#                     true_energy[class_mask].sum().item()
#                 )

#                 if has_rivals:
#                     assert (
#                         pair_violations is not None
#                         and min_pair_margin is not None
#                         and rival_inside is not None
#                         and rival_energy is not None
#                         and margin is not None
#                     )
#                     class_pair_violation[class_id] += int(
#                         pair_violations[class_mask].sum().item()
#                     )
#                     class_pair_relation[class_id] += (
#                         count * (len(ids) - 1)
#                     )
#                     class_min_pair_margin[class_id] += float(
#                         min_pair_margin[class_mask].sum().item()
#                     )
#                     class_rival_inside[class_id] += int(
#                         rival_inside[class_mask].sum().item()
#                     )
#                     class_rival_energy[class_id] += float(
#                         rival_energy[class_mask].sum().item()
#                     )
#                     class_margin[class_id] += float(
#                         margin[class_mask].sum().item()
#                     )

#     if total == 0:
#         raise RuntimeError("evaluation loader is empty")

#     missing_targets = [
#         class_id
#         for class_id in target_ids
#         if class_total[class_id] == 0
#     ]
#     if missing_targets:
#         raise RuntimeError(
#             "evaluation split is missing target classes: "
#             f"{missing_targets}"
#         )

#     per_acc = {
#         class_id: class_correct[class_id] / class_total[class_id]
#         for class_id in target_ids
#     }
#     per_ce_mean = {
#         class_id: class_ce[class_id] / class_total[class_id]
#         for class_id in target_ids
#     }
#     per_cell_violation_mean = {
#         class_id: (
#             class_cell_violation[class_id] / class_total[class_id]
#         )
#         for class_id in target_ids
#     }
#     per_cov = {
#         class_id: class_inside[class_id] / class_total[class_id]
#         for class_id in target_ids
#     }
#     per_inv = {
#         class_id: (
#             class_rival_inside[class_id] / class_total[class_id]
#             if has_rivals else 0.0
#         )
#         for class_id in target_ids
#     }
#     per_no_cell = {
#         class_id: class_no_cell[class_id] / class_total[class_id]
#         for class_id in target_ids
#     }
#     per_pair_violation = {
#         class_id: (
#             class_pair_violation[class_id]
#             / class_pair_relation[class_id]
#             if has_rivals else 0.0
#         )
#         for class_id in target_ids
#     }
#     per_min_pair_margin = {
#         class_id: (
#             class_min_pair_margin[class_id] / class_total[class_id]
#             if has_rivals else None
#         )
#         for class_id in target_ids
#     }
#     per_true = {
#         class_id: (
#             class_true_energy[class_id] / class_total[class_id]
#         )
#         for class_id in target_ids
#     }
#     per_rival = {
#         class_id: (
#             class_rival_energy[class_id] / class_total[class_id]
#             if has_rivals else None
#         )
#         for class_id in target_ids
#     }
#     per_margin = {
#         class_id: (
#             class_margin[class_id] / class_total[class_id]
#             if has_rivals else None
#         )
#         for class_id in target_ids
#     }

#     def macro(values: Mapping[int, float]) -> float:
#         return (
#             sum(float(values[class_id]) for class_id in target_ids)
#             / len(target_ids)
#         )

#     macro_min_pair_margin = (
#         macro({
#             class_id: float(per_min_pair_margin[class_id])
#             for class_id in target_ids
#         })
#         if has_rivals else None
#     )
#     macro_rival = (
#         macro({
#             class_id: float(per_rival[class_id])
#             for class_id in target_ids
#         })
#         if has_rivals else None
#     )
#     macro_margin = (
#         macro({
#             class_id: float(per_margin[class_id])
#             for class_id in target_ids
#         })
#         if has_rivals else None
#     )

#     target_np = torch.cat(targets_all).numpy()
#     prediction_np = torch.cat(predictions_all).numpy()
#     matrix = _confusion(target_np, prediction_np, ids)

#     return {
#         "classification": ce_sum / total,
#         "macro_classification": macro(per_ce_mean),

#         # Backward-compatible diagnostic aliases.
#         "true_cell_violation": true_cell_violation_sum / total,
#         "macro_true_cell_violation": macro(per_cell_violation_mean),
#         "per_class_true_cell_violation": per_cell_violation_mean,
#         "cell_fit": true_cell_violation_sum / total,
#         "macro_cell_fit": macro(per_cell_violation_mean),
#         "per_class_cell_fit": per_cell_violation_mean,

#         "overall_accuracy": correct / total,
#         "accuracy": correct / total,
#         "balanced_accuracy": macro(per_acc),
#         "minimum_class_accuracy": min(per_acc.values()),
#         "kappa": _kappa(matrix),
#         "confusion_matrix": matrix.tolist(),

#         "evaluated_class_ids": list(ids),
#         "target_class_ids": list(target_ids),
#         "class_counts": {
#             class_id: class_total[class_id]
#             for class_id in target_ids
#         },
#         "per_class_accuracy": per_acc,

#         "true_cell_coverage": true_inside_count / total,
#         "macro_true_cell_coverage": macro(per_cov),
#         "per_class_true_cell_coverage": per_cov,

#         "rival_cell_invasion_rate": (
#             rival_inside_count / total if has_rivals else 0.0
#         ),
#         "macro_rival_cell_invasion_rate": macro(per_inv),
#         "per_class_rival_cell_invasion_rate": per_inv,

#         "no_cell_rate": no_cell_count / total,
#         "macro_no_cell_rate": macro(per_no_cell),
#         "per_class_no_cell_rate": per_no_cell,

#         "true_pair_violation_rate": (
#             pair_violation_count / pair_relation_count
#             if pair_relation_count else 0.0
#         ),
#         "macro_true_pair_violation_rate": macro(per_pair_violation),
#         "per_class_true_pair_violation_rate": per_pair_violation,

#         "mean_minimum_true_pair_margin": (
#             min_true_pair_margin_sum / total if has_rivals else None
#         ),
#         "macro_mean_minimum_true_pair_margin": macro_min_pair_margin,
#         "per_class_mean_minimum_true_pair_margin": per_min_pair_margin,

#         "mean_true_energy": true_energy_sum / total,
#         "macro_mean_true_energy": macro(per_true),
#         "per_class_mean_true_energy": per_true,

#         "mean_nearest_rival_energy": (
#             rival_energy_sum / total if has_rivals else None
#         ),
#         "macro_mean_nearest_rival_energy": macro_rival,
#         "per_class_mean_nearest_rival_energy": per_rival,

#         "mean_decision_margin": (
#             margin_sum / total if has_rivals else None
#         ),
#         "macro_mean_decision_margin": macro_margin,
#         "per_class_mean_decision_margin": per_margin,

#         "strict_cell_conflict_rate": 0.0,
#         "energy_convention": {
#             "class_cell": (
#                 "E_c(z) <= 0 iff all pairwise boundaries support class c"
#             ),
#             "class_score": (
#                 "E_c(z) = -minimum oriented pairwise signed distance"
#             ),
#             "decision": "argmin_c E_c(z)",
#             "pair_violation": "s_yj(z) < 0",
#             "no_cell": "min_c E_c(z) > 0",
#             "strict_interior_overlap": (
#                 "impossible by shared opposite pair boundaries"
#             ),
#         },
#     }


# def summarize_class_group(
#     metrics: Mapping[str, Any],
#     class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     """Summarize a class group from cumulative seen-class evaluation."""
#     ids = _class_ids(class_ids)

#     fields = {
#         "balanced_accuracy": "per_class_accuracy",
#         "macro_true_cell_coverage": "per_class_true_cell_coverage",
#         "macro_rival_cell_invasion_rate": (
#             "per_class_rival_cell_invasion_rate"
#         ),
#         "macro_no_cell_rate": "per_class_no_cell_rate",
#         "macro_true_pair_violation_rate": (
#             "per_class_true_pair_violation_rate"
#         ),
#         "macro_true_cell_violation": (
#             "per_class_true_cell_violation"
#         ),
#         "macro_mean_minimum_true_pair_margin": (
#             "per_class_mean_minimum_true_pair_margin"
#         ),
#         "macro_mean_decision_margin": (
#             "per_class_mean_decision_margin"
#         ),
#     }

#     result: Dict[str, Any] = {"class_ids": ids}
#     for output_name, source_name in fields.items():
#         source = metrics.get(source_name)
#         if not isinstance(source, Mapping):
#             raise ValueError(f"metrics lacks {source_name}")

#         values: list[float] = []
#         for class_id in ids:
#             raw = (
#                 source[class_id]
#                 if class_id in source
#                 else source.get(str(class_id))
#             )
#             if raw is None:
#                 raise ValueError(
#                     f"{source_name} lacks class {class_id}"
#                 )
#             values.append(float(raw))
#         result[output_name] = sum(values) / len(values)

#     accuracy_source = metrics.get("per_class_accuracy")
#     if not isinstance(accuracy_source, Mapping):
#         raise ValueError("metrics lacks per_class_accuracy")
#     accuracies = [
#         float(
#             accuracy_source[class_id]
#             if class_id in accuracy_source
#             else accuracy_source[str(class_id)]
#         )
#         for class_id in ids
#     ]
#     result["minimum_class_accuracy"] = min(accuracies)

#     # Compatibility alias; this is not a training objective.
#     result["macro_cell_fit"] = result["macro_true_cell_violation"]
#     return result


# def old_new_summary(
#     metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old = summarize_class_group(metrics, old_class_ids)
#     new = summarize_class_group(metrics, new_class_ids)

#     old_accuracy = float(old["balanced_accuracy"])
#     new_accuracy = float(new["balanced_accuracy"])
#     denominator = old_accuracy + new_accuracy
#     harmonic = (
#         0.0
#         if denominator == 0.0
#         else 2.0 * old_accuracy * new_accuracy / denominator
#     )
#     return {
#         "old": old,
#         "new": new,
#         "harmonic_old_new_accuracy": harmonic,
#     }


# def boundary_preservation_delta(
#     previous_metrics: Mapping[str, Any],
#     current_metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     """Measure deployed historical-class change between finalized phases."""
#     ids = _class_ids(old_class_ids)

#     def per_class_delta(field: str) -> Dict[int, float]:
#         previous = previous_metrics.get(field)
#         current = current_metrics.get(field)
#         if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
#             raise ValueError(f"both metric sets must contain {field}")

#         result: Dict[int, float] = {}
#         for class_id in ids:
#             old_value = (
#                 previous[class_id]
#                 if class_id in previous
#                 else previous.get(str(class_id))
#             )
#             new_value = (
#                 current[class_id]
#                 if class_id in current
#                 else current.get(str(class_id))
#             )
#             if old_value is None or new_value is None:
#                 raise ValueError(
#                     f"{field} lacks historical class {class_id}"
#                 )
#             result[class_id] = float(new_value) - float(old_value)
#         return result

#     accuracy = per_class_delta("per_class_accuracy")
#     coverage = per_class_delta("per_class_true_cell_coverage")
#     pair_violation = per_class_delta(
#         "per_class_true_pair_violation_rate"
#     )
#     no_cell = per_class_delta("per_class_no_cell_rate")
#     invasion = per_class_delta(
#         "per_class_rival_cell_invasion_rate"
#     )
#     margin = per_class_delta("per_class_mean_decision_margin")

#     def mean(values: Mapping[int, float]) -> float:
#         return sum(values.values()) / len(values)

#     return {
#         "class_ids": ids,
#         "per_class_accuracy_delta": accuracy,
#         "per_class_cell_coverage_delta": coverage,
#         "per_class_pair_violation_delta": pair_violation,
#         "per_class_no_cell_rate_delta": no_cell,
#         "per_class_rival_invasion_delta": invasion,
#         "per_class_decision_margin_delta": margin,
#         "mean_accuracy_delta": mean(accuracy),
#         "mean_cell_coverage_delta": mean(coverage),
#         "mean_pair_violation_delta": mean(pair_violation),
#         "mean_no_cell_rate_delta": mean(no_cell),
#         "mean_rival_invasion_delta": mean(invasion),
#         "mean_decision_margin_delta": mean(margin),
#     }


# @torch.no_grad()
# def geometry_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     candidate: Optional[BoundaryCandidate] = None,
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     """Report the committed structural pairwise geometry."""
#     if candidate is not None:
#         raise ValueError(
#             "structural diagnostics require committed geometry"
#         )

#     ids = _class_ids(class_ids)
#     bank = model.geometry_bank
#     if not isinstance(bank, BoundaryGeometryBank):
#         raise TypeError(
#             "model geometry is not BoundaryGeometryBank"
#         )
#     bank.validate_bank_state()

#     committed = [
#         int(value)
#         for value in bank.class_ids.detach().cpu().tolist()
#     ]
#     if committed != ids:
#         raise RuntimeError(
#             "committed geometry classes do not match diagnostics classes"
#         )

#     names = list(target_names or [])
#     pairs: list[Dict[str, Any]] = []
#     for raw_pair in bank.pair_ids.detach().cpu().tolist():
#         left, right = map(int, raw_pair)
#         geometry = bank.get_pair_geometry(left, right)
#         normal = geometry["normal"]
#         offset = geometry["offset"]
#         pairs.append(
#             {
#                 "left_class_id": left,
#                 "right_class_id": right,
#                 "left_class_name": (
#                     names[left]
#                     if left < len(names)
#                     else f"Class {left + 1}"
#                 ),
#                 "right_class_name": (
#                     names[right]
#                     if right < len(names)
#                     else f"Class {right + 1}"
#                 ),
#                 "normal_norm": float(
#                     torch.linalg.vector_norm(normal).item()
#                 ),
#                 "offset": float(offset.item()),
#                 "shared_boundary": True,
#                 "strict_interior_overlap": False,
#             }
#         )

#     expected_pairs = len(ids) * (len(ids) - 1) // 2
#     if len(pairs) != expected_pairs:
#         raise RuntimeError(
#             "committed pairwise geometry is incomplete"
#         )

#     return {
#         "class_ids": ids,
#         "class_count": len(ids),
#         "pair_count": len(pairs),
#         "representation_dim": int(bank.representation_dim),
#         "strict_interior_overlap": "impossible by construction",
#         "pairs": pairs,
#     }


# __all__ = [
#     "boundary_preservation_delta",
#     "evaluate_loader",
#     "geometry_diagnostics",
#     "old_new_summary",
#     "summarize_class_group",
# ]

