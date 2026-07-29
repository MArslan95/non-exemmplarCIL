from __future__ import annotations

"""Evaluation for transport-verified spectral-spatial factor-geometry NECIL.

Deployed score
--------------
    p(z | c) = N(mu_c, L_c L_c^T + Psi_c),  z = [z_s ; z_p].

The ordered physical-spectrum relation p(h | c) is used only to construct the
detached pair-risk matrix. It is not an inference score.

This module evaluates every accepted phase in the exact cumulative global-class
order. It reports OA, AA, Kappa, macro-F1, old/new accuracy, harmonic mean,
old-to-new and new-to-old errors, geometry margins, directional invasion,
per-class accuracy, forgetting, and backward transfer.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


_EPS = 1e-12
CLASSIFICATION_FACTORIZATION = "p(z|c)"
SPECTRAL_RELATION_FACTORIZATION = "p(h|c)"


# =============================================================================
# Conversion and class-order validation
# =============================================================================


def make_json_serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, set):
        return [
            make_json_serializable(item)
            for item in sorted(value, key=str)
        ]
    return value


def save_json(path: str, value: Any) -> str:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(
            make_json_serializable(value),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, absolute)
    return absolute


def _ordered_unique(
    values: Iterable[int],
    *,
    name: str,
    allow_empty: bool = False,
) -> List[int]:
    result: List[int] = []
    observed: set[int] = set()
    for value in values:
        class_id = int(value)
        if class_id < 0:
            raise ValueError(f"{name} contains negative class ID {class_id}")
        if class_id in observed:
            raise ValueError(f"{name} contains duplicate class ID {class_id}")
        observed.add(class_id)
        result.append(class_id)
    if not result and not allow_empty:
        raise ValueError(f"{name} is empty")
    return result


def _as_numpy_labels(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = array.reshape(-1).astype(np.int64, copy=False)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _resolve_old_new(
    seen_classes: Sequence[int],
    *,
    old_classes: Optional[Iterable[int]],
    new_classes: Optional[Iterable[int]],
) -> Tuple[List[int], List[int]]:
    seen = _ordered_unique(seen_classes, name="seen_classes")
    seen_set = set(seen)

    if old_classes is None and new_classes is None:
        return [], list(seen)

    old = (
        []
        if old_classes is None
        else _ordered_unique(
            old_classes, name="old_classes", allow_empty=True
        )
    )
    new = (
        []
        if new_classes is None
        else _ordered_unique(
            new_classes, name="new_classes", allow_empty=True
        )
    )
    if old_classes is None:
        old = [class_id for class_id in seen if class_id not in set(new)]
    if new_classes is None:
        new = [class_id for class_id in seen if class_id not in set(old)]

    old_set, new_set = set(old), set(new)
    if not old_set.issubset(seen_set):
        raise RuntimeError(
            f"old classes outside seen classes: {sorted(old_set - seen_set)}"
        )
    if not new_set.issubset(seen_set):
        raise RuntimeError(
            f"new classes outside seen classes: {sorted(new_set - seen_set)}"
        )
    if old_set & new_set:
        raise RuntimeError(
            f"old/new class overlap: {sorted(old_set & new_set)}"
        )
    if old_set | new_set != seen_set:
        raise RuntimeError("old/new classes do not partition seen classes")
    return (
        [class_id for class_id in seen if class_id in old_set],
        [class_id for class_id in seen if class_id in new_set],
    )


def _class_name(
    target_names: Optional[Sequence[str]],
    class_id: int,
) -> str:
    class_id = int(class_id)
    if (
        target_names is not None
        and 0 <= class_id < len(target_names)
        and str(target_names[class_id]).strip()
    ):
        return str(target_names[class_id]).strip()
    return f"Class-{class_id}"


# =============================================================================
# Batch contract and deployed factor-geometry evaluation
# =============================================================================


def _mapping_value(
    batch: Mapping[str, Any],
    names: Sequence[str],
) -> Any:
    for name in names:
        if name in batch and batch[name] is not None:
            return batch[name]
    return None


def unpack_factor_eval_batch(
    batch: Any,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Return processed patch, raw patch/center, labels, coords, sample IDs."""
    if isinstance(batch, Mapping):
        processed = _mapping_value(
            batch,
            ("image", "patch", "patches", "x", "input"),
        )
        raw_patch = _mapping_value(
            batch,
            (
                "raw_spectral_patch",
                "raw_spectral_patches",
                "physical_spectral_patch",
                "physical_patch",
                "raw_patch",
            ),
        )
        raw_center = _mapping_value(
            batch,
            (
                "raw_center_spectrum",
                "raw_center_spectra",
                "center_spectrum_raw",
            ),
        )
        labels = _mapping_value(
            batch,
            ("label", "labels", "target", "y"),
        )
        coords = _mapping_value(
            batch,
            ("coord", "coords", "coordinate"),
        )
        sample_ids = _mapping_value(
            batch,
            ("sample_index", "sample_indices", "index", "indices"),
        )
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        processed = batch[0]
        labels = batch[1]
        raw_patch = batch[2] if len(batch) >= 3 else None
        raw_center = None
        coords = batch[3] if len(batch) >= 4 else None
        sample_ids = batch[4] if len(batch) >= 5 else None
    else:
        raise RuntimeError(
            "factor evaluation requires a mapping or tuple/list batch"
        )

    if processed is None or labels is None:
        raise RuntimeError("evaluation batch lacks processed patches or labels")
    if raw_patch is None and raw_center is None:
        raise RuntimeError(
            "evaluation batch must contain raw spectral patches or center spectra"
        )

    processed_t = torch.as_tensor(processed)
    labels_t = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if processed_t.ndim != 4:
        raise RuntimeError(
            f"processed patches must be [B,C,H,W], got {tuple(processed_t.shape)}"
        )
    batch_size = int(processed_t.size(0))
    if labels_t.numel() != batch_size:
        raise RuntimeError("processed-patch/label batch mismatch")

    raw_patch_t = (
        None if raw_patch is None else torch.as_tensor(raw_patch)
    )
    raw_center_t = (
        None if raw_center is None else torch.as_tensor(raw_center)
    )
    if raw_patch_t is not None:
        if raw_patch_t.ndim != 4 or raw_patch_t.size(0) != batch_size:
            raise RuntimeError(
                "raw spectral patches must be [B,Bands,H,W]"
            )
        if raw_patch_t.shape[2:] != processed_t.shape[2:]:
            raise RuntimeError("processed and raw patch shapes are misaligned")
    if raw_center_t is not None:
        if raw_center_t.ndim != 2 or raw_center_t.size(0) != batch_size:
            raise RuntimeError(
                "raw center spectra must be [B,Bands]"
            )

    coords_t = None if coords is None else torch.as_tensor(coords, dtype=torch.long)
    if coords_t is not None and tuple(coords_t.shape) != (batch_size, 2):
        raise RuntimeError(
            f"coordinates must be [B,2], got {tuple(coords_t.shape)}"
        )
    sample_t = (
        None
        if sample_ids is None
        else torch.as_tensor(sample_ids, dtype=torch.long).reshape(-1)
    )
    if sample_t is not None and sample_t.numel() != batch_size:
        raise RuntimeError("sample-index batch mismatch")

    for name, tensor in (
        ("processed patches", processed_t),
        ("raw patches", raw_patch_t),
        ("raw centers", raw_center_t),
    ):
        if tensor is not None and not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"{name} contain NaN/Inf")

    return (
        processed_t,
        raw_patch_t,
        raw_center_t,
        labels_t,
        coords_t,
        sample_t,
    )


@torch.no_grad()
def evaluate_factor_geometry_loader(
    model: torch.nn.Module,
    loader: Any,
    *,
    seen_classes: Iterable[int],
    device: str | torch.device,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    energy_margin: float = 0.0,
    return_arrays: bool = True,
) -> Dict[str, Any]:
    """Evaluate the deployed parameter-free factor-energy classifier."""
    seen = _ordered_unique(seen_classes, name="seen_classes")
    old, new = _resolve_old_new(
        seen,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    requested_device = torch.device(device)
    if (
        requested_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but is unavailable")

    expected_rows = list(model.infer_seen_classes())
    if set(expected_rows) != set(seen):
        raise RuntimeError(
            f"committed rows {expected_rows} do not match evaluation classes {seen}"
        )
    model.geometry_bank.assert_valid(seen, strict=True)

    previous_training = bool(model.training)
    model.eval()
    y_true_parts: List[torch.Tensor] = []
    y_pred_parts: List[torch.Tensor] = []
    energy_parts: List[torch.Tensor] = []
    quadratic_parts: List[torch.Tensor] = []
    volume_parts: List[torch.Tensor] = []
    feature_parts: List[torch.Tensor] = []
    spectral_feature_parts: List[torch.Tensor] = []
    spatial_feature_parts: List[torch.Tensor] = []
    raw_spectrum_parts: List[torch.Tensor] = []
    coord_parts: List[torch.Tensor] = []
    sample_parts: List[torch.Tensor] = []
    class_tensor = torch.tensor(
        seen,
        device=requested_device,
        dtype=torch.long,
    )

    try:
        for batch in loader:
            (
                processed,
                raw_patch,
                raw_center,
                labels,
                coords,
                sample_ids,
            ) = unpack_factor_eval_batch(batch)
            processed = processed.to(
                requested_device,
                dtype=torch.float32,
                non_blocking=True,
            )
            raw_patch = (
                None
                if raw_patch is None
                else raw_patch.to(
                    requested_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )
            raw_center = (
                None
                if raw_center is None
                else raw_center.to(
                    requested_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )
            labels = labels.to(
                requested_device,
                dtype=torch.long,
                non_blocking=True,
            )
            unexpected = sorted(
                set(int(value) for value in labels.detach().cpu().tolist())
                - set(seen)
            )
            if unexpected:
                raise RuntimeError(
                    f"evaluation batch contains classes outside seen set: {unexpected}"
                )

            output = model.forward_features(
                processed,
                raw_spectral_patch=raw_patch,
                raw_center_spectrum=raw_center,
                deterministic=True,
            )
            scored = model.compute_logits_from_features(
                output["joint_feature"],
                class_ids=seen,
                targets=labels,
                targets_are_global=True,
                old_classes=old,
                new_classes=new,
                mode="factor_geometry",
                return_parts=True,
                return_diagnostics=True,
            )
            if not isinstance(scored, Mapping):
                raise RuntimeError("factor classifier did not return a mapping")
            if scored.get("classification_factorization") != CLASSIFICATION_FACTORIZATION:
                raise RuntimeError(
                    "evaluation received a non-factor classifier result"
                )
            returned_ids = scored["class_ids"].detach().cpu().tolist()
            if returned_ids != seen:
                raise RuntimeError(
                    f"classifier column order {returned_ids} != requested {seen}"
                )
            energy = scored["energy"]
            expected_shape = (processed.size(0), len(seen))
            for name, tensor in (
                ("energy", energy),
                ("quadratic", scored["quadratic"]),
                ("volume", scored["volume"]),
            ):
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"{name} shape {tuple(tensor.shape)} != {expected_shape}"
                    )
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"{name} contains NaN/Inf")

            prediction = class_tensor.index_select(
                0,
                energy.argmin(dim=1),
            )
            y_true_parts.append(labels.detach().cpu())
            y_pred_parts.append(prediction.detach().cpu())
            energy_parts.append(energy.detach().cpu())
            quadratic_parts.append(scored["quadratic"].detach().cpu())
            volume_parts.append(scored["volume"].detach().cpu())
            feature_parts.append(output["joint_feature"].detach().cpu())
            spectral_feature_parts.append(
                output["spectral_feature"].detach().cpu()
            )
            spatial_feature_parts.append(
                output["spatial_feature"].detach().cpu()
            )
            raw_spectrum_parts.append(
                output["raw_center_spectrum"].detach().cpu()
            )
            if coords is not None:
                coord_parts.append(coords.detach().cpu())
            if sample_ids is not None:
                sample_parts.append(sample_ids.detach().cpu())
    finally:
        model.train(previous_training)

    if not y_true_parts:
        raise RuntimeError("evaluation loader is empty")

    y_true = torch.cat(y_true_parts)
    y_pred = torch.cat(y_pred_parts)
    energy = torch.cat(energy_parts)
    quadratic = torch.cat(quadratic_parts)
    volume = torch.cat(volume_parts)

    metrics = calculate_metrics_torch(
        y_true,
        y_pred,
        seen_classes=seen,
        old_classes=old,
        new_classes=new,
        energy=energy,
        energy_margin=float(energy_margin),
    )
    metrics.update(
        pairwise_directional_energy_diagnostics(
            energy,
            y_true,
            seen_classes=seen,
        )
    )
    pair_risk = model.geometry_bank.pair_risk_matrix(seen)
    metrics["pair_risk_matrix"] = pair_risk.detach().cpu()
    metrics["classification_factorization"] = CLASSIFICATION_FACTORIZATION
    metrics["spectral_relation_factorization"] = (
        SPECTRAL_RELATION_FACTORIZATION
    )
    metrics["spectral_shape_used_for_inference"] = False
    metrics["class_order"] = list(seen)
    metrics["bank_rows_digest"] = model.geometry_bank.rows_digest(seen)
    metrics["bank_contract_digest"] = (
        model.classifier.bank_contract_digest(model.geometry_bank)
    )

    output: Dict[str, Any] = {
        "metrics": metrics,
        "seen_classes": seen,
        "old_classes": old,
        "new_classes": new,
    }
    if return_arrays:
        output.update(
            {
                "y_true": y_true.numpy(),
                "y_pred": y_pred.numpy(),
                "energy": energy,
                "quadratic": quadratic,
                "volume": volume,
                "joint_features": torch.cat(feature_parts),
                "spectral_features": torch.cat(spectral_feature_parts),
                "spatial_features": torch.cat(spatial_feature_parts),
                "raw_center_spectra": torch.cat(raw_spectrum_parts),
                "coords": (
                    torch.cat(coord_parts).numpy()
                    if coord_parts else None
                ),
                "sample_indices": (
                    torch.cat(sample_parts).numpy()
                    if sample_parts else None
                ),
            }
        )
    return output


# Backward-compatible public name.
evaluate_geometry_loader = evaluate_factor_geometry_loader


# =============================================================================
# Metrics
# =============================================================================


def confusion_matrix_for_labels(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Sequence[int],
) -> torch.Tensor:
    ids = _ordered_unique(labels, name="confusion labels")
    true = torch.as_tensor(y_true, dtype=torch.long).reshape(-1)
    pred = torch.as_tensor(y_pred, dtype=torch.long).reshape(-1)
    if true.numel() != pred.numel():
        raise ValueError("y_true/y_pred length mismatch")

    mapping = {class_id: index for index, class_id in enumerate(ids)}
    true_local = torch.full_like(true, -1)
    pred_local = torch.full_like(pred, -1)
    for class_id, local in mapping.items():
        true_local[true.eq(class_id)] = local
        pred_local[pred.eq(class_id)] = local
    if bool(true_local.lt(0).any()):
        bad = true[true_local.lt(0)].unique().tolist()
        raise RuntimeError(f"true labels missing from confusion order: {bad}")
    if bool(pred_local.lt(0).any()):
        bad = pred[pred_local.lt(0)].unique().tolist()
        raise RuntimeError(f"predictions missing from confusion order: {bad}")
    count = len(ids)
    indices = true_local * count + pred_local
    return torch.bincount(
        indices,
        minlength=count * count,
    ).reshape(count, count)


def metrics_from_confusion(
    confusion: torch.Tensor,
    *,
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    matrix = confusion.detach().double()
    ids = _ordered_unique(class_ids, name="class_ids")
    if matrix.shape != (len(ids), len(ids)):
        raise ValueError("confusion matrix shape does not match class IDs")

    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    total = matrix.sum()
    recall = torch.where(
        support > 0,
        true_positive / support,
        torch.zeros_like(support),
    )
    precision = torch.where(
        predicted > 0,
        true_positive / predicted,
        torch.zeros_like(predicted),
    )
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    valid = support > 0
    observed = true_positive.sum() / total.clamp_min(_EPS)
    expected = (
        (support * predicted).sum()
        / (total.square().clamp_min(_EPS))
    )
    kappa = (
        (observed - expected) / (1.0 - expected).clamp_min(_EPS)
    )
    per_class_accuracy = {
        class_id: 100.0 * float(recall[index].item())
        for index, class_id in enumerate(ids)
    }
    return {
        "overall_accuracy": 100.0 * float(observed.item()),
        "average_accuracy": (
            100.0 * float(recall[valid].mean().item())
            if bool(valid.any()) else 0.0
        ),
        "minimum_per_class_accuracy": min(
            per_class_accuracy.values(),
            default=0.0,
        ),
        "kappa": float(kappa.item()),
        "f1_macro": (
            100.0 * float(f1[valid].mean().item())
            if bool(valid.any()) else 0.0
        ),
        "macro_precision": (
            100.0 * float(precision[valid].mean().item())
            if bool(valid.any()) else 0.0
        ),
        "macro_recall": (
            100.0 * float(recall[valid].mean().item())
            if bool(valid.any()) else 0.0
        ),
        "per_class_accuracy": per_class_accuracy,
        "precision": {
            class_id: 100.0 * float(precision[index].item())
            for index, class_id in enumerate(ids)
        },
        "recall": {
            class_id: 100.0 * float(recall[index].item())
            for index, class_id in enumerate(ids)
        },
        "f1_per_class": {
            class_id: 100.0 * float(f1[index].item())
            for index, class_id in enumerate(ids)
        },
        "support": {
            class_id: int(support[index].item())
            for index, class_id in enumerate(ids)
        },
        "predicted_count": {
            class_id: int(predicted[index].item())
            for index, class_id in enumerate(ids)
        },
        "confusion_matrix": matrix.long().cpu(),
        "confusion_matrix_labels": list(ids),
    }


def calculate_metrics_torch(
    y_true: Any,
    y_pred: Any,
    *,
    seen_classes: Iterable[int],
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Dict[str, Any]:
    true = _as_numpy_labels(y_true, name="y_true")
    pred = _as_numpy_labels(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true/y_pred shape mismatch")

    seen = _ordered_unique(seen_classes, name="seen_classes")
    old, new = _resolve_old_new(
        seen,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    true_outside = sorted(set(np.unique(true).tolist()) - set(seen))
    if true_outside:
        raise RuntimeError(
            f"true labels outside seen classes: {true_outside}"
        )
    invalid_mask = ~np.isin(pred, np.asarray(seen, dtype=np.int64))
    invalid_classes = sorted(
        int(value)
        for value in np.unique(pred[invalid_mask]).tolist()
    )
    full_labels = [
        *seen,
        *[
            class_id
            for class_id in invalid_classes
            if class_id not in set(seen)
        ],
    ]
    confusion = confusion_matrix_for_labels(
        true,
        pred,
        labels=full_labels,
    )
    full_metrics = metrics_from_confusion(
        confusion,
        class_ids=full_labels,
    )

    seen_positions = torch.tensor(
        [full_labels.index(class_id) for class_id in seen],
        dtype=torch.long,
    )
    seen_confusion = confusion.index_select(
        0,
        seen_positions,
    ).index_select(1, seen_positions)
    seen_metrics = metrics_from_confusion(
        seen_confusion,
        class_ids=seen,
    )

    correct = pred == true
    old_mask = np.isin(true, np.asarray(old, dtype=np.int64))
    new_mask = np.isin(true, np.asarray(new, dtype=np.int64))
    old_accuracy = (
        100.0 * float(correct[old_mask].mean())
        if old_mask.any() else 0.0
    )
    new_accuracy = (
        100.0 * float(correct[new_mask].mean())
        if new_mask.any() else 0.0
    )
    split_available = bool(old_mask.any() and new_mask.any())
    harmonic = (
        2.0 * old_accuracy * new_accuracy
        / max(old_accuracy + new_accuracy, _EPS)
        if split_available
        else seen_metrics["overall_accuracy"]
    )
    pred_old = np.isin(pred, np.asarray(old, dtype=np.int64))
    pred_new = np.isin(pred, np.asarray(new, dtype=np.int64))

    metrics = dict(seen_metrics)
    metrics.update(
        {
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "harmonic_mean": float(harmonic),
            "old_count": int(old_mask.sum()),
            "new_count": int(new_mask.sum()),
            "old_class_ids": list(old),
            "new_class_ids": list(new),
            "old_new_split_available": split_available,
            "old_to_new_rate": (
                100.0 * float((old_mask & pred_new).sum())
                / max(int(old_mask.sum()), 1)
            ),
            "new_to_old_rate": (
                100.0 * float((new_mask & pred_old).sum())
                / max(int(new_mask.sum()), 1)
            ),
            "invalid_prediction_rate": (
                100.0 * float(invalid_mask.mean())
            ),
            "predicted_unseen_count": int(invalid_mask.sum()),
            "predicted_unseen_classes": invalid_classes,
            "num_samples": int(true.size),
            "num_classes": len(seen),
            "classes": list(seen),
            "confusion_matrix_full": confusion.cpu(),
            "confusion_matrix_full_labels": full_labels,
            "per_class_accuracy_all": full_metrics[
                "per_class_accuracy"
            ],
        }
    )
    if energy is not None:
        metrics.update(
            factor_energy_diagnostics(
                energy,
                true,
                seen_classes=seen,
                old_classes=old,
                new_classes=new,
                margin=float(energy_margin),
            )
        )

    # Compact aliases used by aggregation code.
    metrics["oa"] = metrics["overall_accuracy"]
    metrics["aa"] = metrics["average_accuracy"]
    metrics["macro_f1"] = metrics["f1_macro"]
    metrics["old_acc"] = metrics["old_accuracy"]
    metrics["new_acc"] = metrics["new_accuracy"]
    metrics["hm"] = metrics["harmonic_mean"]
    return metrics


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    seen_classes: Iterable[int],
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    return calculate_metrics_torch(
        y_true,
        y_pred,
        seen_classes=seen_classes,
        old_classes=old_classes,
        new_classes=new_classes,
    )


def factor_energy_diagnostics(
    energy: torch.Tensor,
    labels_global: Any,
    *,
    seen_classes: Iterable[int],
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    margin: float = 0.0,
) -> Dict[str, Any]:
    if not torch.is_tensor(energy) or energy.dim() != 2:
        raise ValueError("energy must be [N,C]")
    seen = _ordered_unique(seen_classes, name="seen_classes")
    if energy.size(1) != len(seen):
        raise RuntimeError("energy width does not match seen-class order")
    labels = torch.as_tensor(
        labels_global,
        device=energy.device,
        dtype=torch.long,
    ).reshape(-1)
    if labels.numel() != energy.size(0):
        raise RuntimeError("energy/label batch mismatch")
    mapping = {class_id: index for index, class_id in enumerate(seen)}
    local = torch.full_like(labels, -1)
    for class_id, column in mapping.items():
        local[labels.eq(class_id)] = column
    if bool(local.lt(0).any()):
        bad = labels[local.lt(0)].unique().tolist()
        raise RuntimeError(f"labels outside seen classes: {bad}")

    true_energy = energy.gather(1, local[:, None]).squeeze(1)
    rivals = energy.clone()
    rivals.scatter_(1, local[:, None], float("inf"))
    rival_energy, rival_local = rivals.min(dim=1)
    gap = rival_energy - true_energy
    prediction_local = energy.argmin(dim=1)

    old, new = _resolve_old_new(
        seen,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    old_columns = torch.tensor(
        [mapping[class_id] for class_id in old],
        device=energy.device,
        dtype=torch.long,
    )
    new_columns = torch.tensor(
        [mapping[class_id] for class_id in new],
        device=energy.device,
        dtype=torch.long,
    )
    old_samples = torch.zeros_like(labels, dtype=torch.bool)
    new_samples = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in old:
        old_samples |= labels.eq(class_id)
    for class_id in new:
        new_samples |= labels.eq(class_id)

    old_to_new_gap = energy.new_empty((0,))
    new_to_old_gap = energy.new_empty((0,))
    if old_columns.numel() and new_columns.numel():
        old_min = energy.index_select(1, old_columns).min(dim=1).values
        new_min = energy.index_select(1, new_columns).min(dim=1).values
        if bool(old_samples.any()):
            old_to_new_gap = (
                new_min[old_samples] - true_energy[old_samples]
            )
        if bool(new_samples.any()):
            new_to_old_gap = (
                old_min[new_samples] - true_energy[new_samples]
            )

    per_class: Dict[int, Dict[str, float]] = {}
    for class_id in seen:
        selected = labels.eq(class_id)
        class_gap = gap[selected]
        per_class[class_id] = {
            "count": int(selected.sum().item()),
            "accuracy": (
                100.0
                * float(
                    prediction_local[selected]
                    .eq(mapping[class_id])
                    .float()
                    .mean()
                    .item()
                )
                if bool(selected.any()) else 0.0
            ),
            "mean_gap": (
                float(class_gap.mean().item())
                if class_gap.numel() else 0.0
            ),
            "q05_gap": (
                float(torch.quantile(class_gap, 0.05).item())
                if class_gap.numel() else 0.0
            ),
            "minimum_gap": (
                float(class_gap.min().item())
                if class_gap.numel() else 0.0
            ),
            "classification_violation_rate": (
                100.0
                * float(class_gap.le(0.0).float().mean().item())
                if class_gap.numel() else 0.0
            ),
            "margin_violation_rate": (
                100.0
                * float(class_gap.le(float(margin)).float().mean().item())
                if class_gap.numel() else 0.0
            ),
        }

    def percentage(mask: torch.Tensor) -> float:
        return (
            100.0 * float(mask.float().mean().item())
            if mask.numel() else 0.0
        )

    return {
        "geometry_energy_accuracy": (
            100.0
            * float(prediction_local.eq(local).float().mean().item())
        ),
        "geometry_margin_mean": float(gap.mean().item()),
        "geometry_margin_q01": float(torch.quantile(gap, 0.01).item()),
        "geometry_margin_q05": float(torch.quantile(gap, 0.05).item()),
        "geometry_margin_min": float(gap.min().item()),
        "geometry_margin_threshold": float(margin),
        "geometry_margin_violation_rate": percentage(
            gap.le(float(margin))
        ),
        "geometry_error_rate": percentage(gap.le(0.0)),
        "old_to_new_energy_invasion_rate": percentage(
            old_to_new_gap.le(float(margin))
        ),
        "new_to_old_energy_invasion_rate": percentage(
            new_to_old_gap.le(float(margin))
        ),
        "old_to_new_energy_error_rate": percentage(
            old_to_new_gap.le(0.0)
        ),
        "new_to_old_energy_error_rate": percentage(
            new_to_old_gap.le(0.0)
        ),
        "old_to_new_energy_gap_mean": (
            float(old_to_new_gap.mean().item())
            if old_to_new_gap.numel() else 0.0
        ),
        "new_to_old_energy_gap_mean": (
            float(new_to_old_gap.mean().item())
            if new_to_old_gap.numel() else 0.0
        ),
        "per_class_geometry": per_class,
        "nearest_rival_global": torch.tensor(
            [seen[index] for index in rival_local.detach().cpu().tolist()],
            dtype=torch.long,
        ),
    }


def pairwise_directional_energy_diagnostics(
    energy: torch.Tensor,
    labels_global: Any,
    *,
    seen_classes: Iterable[int],
) -> Dict[str, Any]:
    seen = _ordered_unique(seen_classes, name="seen_classes")
    labels = torch.as_tensor(labels_global, dtype=torch.long).reshape(-1)
    scores = torch.as_tensor(energy).float().cpu()
    if scores.shape != (labels.numel(), len(seen)):
        raise RuntimeError("pairwise energy shape mismatch")
    mapping = {class_id: index for index, class_id in enumerate(seen)}

    invasion = torch.zeros((len(seen), len(seen)), dtype=torch.float64)
    q05 = torch.full_like(invasion, float("nan"))
    mean_gap = torch.full_like(invasion, float("nan"))
    for source_class in seen:
        source = mapping[source_class]
        selected = labels.eq(source_class)
        if not bool(selected.any()):
            continue
        true_energy = scores[selected, source]
        for rival_class in seen:
            rival = mapping[rival_class]
            if source == rival:
                invasion[source, rival] = 0.0
                q05[source, rival] = 0.0
                mean_gap[source, rival] = 0.0
                continue
            gap = scores[selected, rival] - true_energy
            invasion[source, rival] = gap.le(0.0).double().mean()
            q05[source, rival] = torch.quantile(gap.double(), 0.05)
            mean_gap[source, rival] = gap.double().mean()

    off_diagonal = ~torch.eye(len(seen), dtype=torch.bool)
    return {
        "directional_invasion_matrix": invasion,
        "directional_gap_q05_matrix": q05,
        "directional_gap_mean_matrix": mean_gap,
        "maximum_directional_invasion": (
            float(invasion[off_diagonal].max().item())
            if bool(off_diagonal.any()) else 0.0
        ),
        "mean_directional_invasion": (
            float(invasion[off_diagonal].mean().item())
            if bool(off_diagonal.any()) else 0.0
        ),
    }


# =============================================================================
# GeometryBank and diagnostic sampling reports
# =============================================================================


@torch.no_grad()
def geometry_sampling_health(
    model: torch.nn.Module,
    class_ids: Sequence[int],
    *,
    samples_per_class: int = 32,
) -> Dict[str, Any]:
    ids = _ordered_unique(class_ids, name="class_ids")
    sampled = model.geometry_bank.sample_geometry(
        ids,
        int(samples_per_class),
    )
    scored = model.compute_logits_from_features(
        sampled["features"],
        class_ids=ids,
        targets=sampled["labels"],
        targets_are_global=True,
        return_parts=True,
        return_diagnostics=True,
    )
    return {
        "classification_factorization": CLASSIFICATION_FACTORIZATION,
        "sample_count": int(sampled["labels"].numel()),
        "accuracy": 100.0 * float(
            scored["diagnostics"]["accuracy"]
        ),
        "mean_gap": float(
            scored["diagnostics"]["mean_margin"]
        ),
        "q05_gap": float(
            scored["diagnostics"]["q05_margin"]
        ),
        "minimum_gap": float(
            scored["diagnostics"]["minimum_margin"]
        ),
        "violation_rate": 100.0 * float(
            scored["diagnostics"]["violation_rate"]
        ),
        "training_use": False,
        "persistent_samples": False,
    }


@torch.no_grad()
def geometry_bank_diagnostics(
    model: torch.nn.Module,
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    ids = _ordered_unique(class_ids, name="class_ids")
    bank = model.geometry_bank
    bank.assert_valid(ids, strict=True)
    effective_dimension = bank.effective_dimension(ids).detach().cpu()
    pair_risk = bank.pair_risk_matrix(ids).detach().cpu()
    spectral_distance = bank.spectral_shape_distance_matrix(ids).detach().cpu()
    factor_distance = bank.factor_bhattacharyya_distance_matrix(ids).detach().cpu()
    rows: Dict[int, Dict[str, Any]] = {}
    for index, class_id in enumerate(ids):
        row = bank.get_class_row(class_id)
        rows[class_id] = {
            "sample_count": int(row["sample_count"].item()),
            "effective_sample_count": float(
                row["effective_sample_count"].item()
            ),
            "active_rank": int(row["active_rank"].item()),
            "spectral_residual_variance": float(
                row["residual_var_spectral"].item()
            ),
            "spatial_residual_variance": float(
                row["residual_var_spatial"].item()
            ),
            "spectral_shape_reliability": float(
                row["spectral_shape_reliability"].item()
            ),
            "geometry_reliability": float(
                row["geometry_reliability"].item()
            ),
            "reconstruction_error": float(
                row["reconstruction_error"].item()
            ),
            "outlier_rate": float(row["outlier_rate"].item()),
            "effective_dimension": float(
                effective_dimension[index].item()
            ),
        }
    return {
        "classification_factorization": CLASSIFICATION_FACTORIZATION,
        "spectral_relation_factorization": (
            SPECTRAL_RELATION_FACTORIZATION
        ),
        "memory_cost": bank.memory_cost_summary(),
        "admission": model.geometry_admission_report(
            ids,
            maximum_reconstruction_error=float(
                getattr(
                    model.args,
                    "maximum_geometry_reconstruction_error",
                    0.75,
                )
            ),
            minimum_effective_dimension=float(
                getattr(
                    model.args,
                    "minimum_geometry_effective_dimension",
                    1.25,
                )
            ),
            require_statistics=True,
        ),
        "pair_risk_matrix": pair_risk,
        "spectral_shape_distance_matrix": spectral_distance,
        "factor_bhattacharyya_distance_matrix": factor_distance,
        "classes": rows,
        "rows_digest": bank.rows_digest(ids),
    }


# =============================================================================
# Reports
# =============================================================================


def save_classification_report(
    y_true: Any,
    y_pred: Any,
    *,
    target_names: Optional[Sequence[str]],
    save_dir: str,
    phase: int,
    seen_classes: Iterable[int],
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    metrics = calculate_metrics_torch(
        y_true,
        y_pred,
        seen_classes=seen_classes,
        old_classes=old_classes,
        new_classes=new_classes,
        energy=energy,
        energy_margin=energy_margin,
    )
    seen = list(metrics["classes"])
    confusion = metrics["confusion_matrix"].numpy()
    names = [_class_name(target_names, class_id) for class_id in seen]

    prefix = f"phase_{int(phase)}"
    text_path = os.path.join(
        save_dir,
        f"{prefix}_classification_report.txt",
    )
    json_path = os.path.join(
        save_dir,
        f"{prefix}_classification_report.json",
    )
    confusion_csv = os.path.join(
        save_dir,
        f"{prefix}_confusion_matrix.csv",
    )
    confusion_npy = os.path.join(
        save_dir,
        f"{prefix}_confusion_matrix.npy",
    )
    per_class_csv = os.path.join(
        save_dir,
        f"{prefix}_per_class_metrics.csv",
    )

    lines = [
        f"Factor-Geometry NECIL Classification Report — Phase {phase}",
        "=" * 96,
        "",
        f"Classification factorization: {CLASSIFICATION_FACTORIZATION}",
        (
            "Physical spectral relation: "
            f"{SPECTRAL_RELATION_FACTORIZATION} (pair risk only)"
        ),
        f"Seen class order: {seen}",
        f"Old classes: {metrics['old_class_ids']}",
        f"New classes: {metrics['new_class_ids']}",
        "",
        (
            f"OA={metrics['overall_accuracy']:.4f}% | "
            f"AA={metrics['average_accuracy']:.4f}% | "
            f"Kappa={metrics['kappa']:.6f} | "
            f"Macro-F1={metrics['f1_macro']:.4f}%"
        ),
        (
            f"Old={metrics['old_accuracy']:.4f}% | "
            f"New={metrics['new_accuracy']:.4f}% | "
            f"H={metrics['harmonic_mean']:.4f}%"
        ),
        (
            f"Old->New={metrics['old_to_new_rate']:.4f}% | "
            f"New->Old={metrics['new_to_old_rate']:.4f}% | "
            f"Invalid={metrics['invalid_prediction_rate']:.4f}%"
        ),
    ]
    if "geometry_margin_q05" in metrics:
        lines.extend(
            [
                (
                    f"MeanGap={metrics['geometry_margin_mean']:.6f} | "
                    f"Q05Gap={metrics['geometry_margin_q05']:.6f} | "
                    f"MinGap={metrics['geometry_margin_min']:.6f}"
                ),
                (
                    "Geometry violation="
                    f"{metrics['geometry_error_rate']:.4f}% | "
                    "Margin violation="
                    f"{metrics['geometry_margin_violation_rate']:.4f}%"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Class-wise performance",
            "-" * 96,
            (
                "id  class                              support   accuracy  "
                "precision     recall         f1"
            ),
        ]
    )
    for class_id, name in zip(seen, names):
        lines.append(
            f"{class_id:2d}  {name[:32]:32s} "
            f"{metrics['support'][class_id]:7d} "
            f"{metrics['per_class_accuracy'][class_id]:9.2f}% "
            f"{metrics['precision'][class_id]:9.2f}% "
            f"{metrics['recall'][class_id]:9.2f}% "
            f"{metrics['f1_per_class'][class_id]:9.2f}%"
        )

    with open(text_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    save_json(json_path, metrics)
    np.save(confusion_npy, confusion)

    with open(
        confusion_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["true\\pred"]
            + [
                f"{name} [{class_id}]"
                for class_id, name in zip(seen, names)
            ]
        )
        for index, (class_id, name) in enumerate(zip(seen, names)):
            writer.writerow(
                [f"{name} [{class_id}]"]
                + [int(value) for value in confusion[index].tolist()]
            )

    with open(
        per_class_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "phase_role",
                "support",
                "accuracy_percent",
                "precision_percent",
                "recall_percent",
                "f1_percent",
            ]
        )
        old_set = set(metrics["old_class_ids"])
        new_set = set(metrics["new_class_ids"])
        for class_id, name in zip(seen, names):
            role = (
                "new"
                if class_id in new_set
                else "old"
                if class_id in old_set
                else "base"
            )
            writer.writerow(
                [
                    class_id,
                    name,
                    role,
                    metrics["support"][class_id],
                    metrics["per_class_accuracy"][class_id],
                    metrics["precision"][class_id],
                    metrics["recall"][class_id],
                    metrics["f1_per_class"][class_id],
                ]
            )

    return {
        "txt_path": text_path,
        "json_path": json_path,
        "confusion_matrix_csv_path": confusion_csv,
        "confusion_matrix_npy_path": confusion_npy,
        "per_class_csv_path": per_class_csv,
        "metrics": metrics,
    }


# =============================================================================
# Across-phase NECIL evaluator
# =============================================================================


class NECILEvaluator:
    def __init__(self) -> None:
        self.phase_history: Dict[int, Dict[str, Any]] = {}
        self.phases_seen: List[int] = []
        self.class_history: defaultdict[int, List[float]] = defaultdict(list)

    def restore(self, payload: Mapping[str, Any]) -> None:
        """Restore prior phase metrics for resumed evaluation summaries."""
        source = payload.get("phase_history", payload)
        if not isinstance(source, Mapping):
            raise TypeError("evaluator payload must contain phase_history")
        restored: Dict[int, Dict[str, Any]] = {}
        for phase, metrics in source.items():
            if not isinstance(metrics, Mapping):
                continue
            restored[int(phase)] = dict(metrics)
        self.phase_history = restored
        self.phases_seen = sorted(restored)
        self._rebuild_class_history()

    def update(
        self,
        phase: int,
        y_true: Any,
        y_pred: Any,
        *,
        seen_classes: Iterable[int],
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        energy: Optional[torch.Tensor] = None,
        energy_margin: float = 0.0,
    ) -> Dict[str, Any]:
        phase = int(phase)
        metrics = calculate_metrics_torch(
            y_true,
            y_pred,
            seen_classes=seen_classes,
            old_classes=old_classes,
            new_classes=new_classes,
            energy=energy,
            energy_margin=energy_margin,
        )
        self.phase_history[phase] = metrics
        self.phases_seen = sorted(self.phase_history)
        self._rebuild_class_history()
        return metrics

    def _rebuild_class_history(self) -> None:
        classes = sorted(
            {
                int(class_id)
                for phase in self.phases_seen
                for class_id, support in self.phase_history[phase][
                    "support"
                ].items()
                if int(support) > 0
            }
        )
        history: defaultdict[int, List[float]] = defaultdict(list)
        for class_id in classes:
            for phase in self.phases_seen:
                metrics = self.phase_history[phase]
                if int(metrics["support"].get(class_id, 0)) > 0:
                    history[class_id].append(
                        float(
                            metrics["per_class_accuracy"].get(
                                class_id,
                                float("nan"),
                            )
                        )
                    )
                else:
                    history[class_id].append(float("nan"))
        self.class_history = history

    def forgetting_per_class(self) -> Dict[int, float]:
        result: Dict[int, float] = {}
        for class_id, values in self.class_history.items():
            observed = np.asarray(values, dtype=np.float64)
            observed = observed[np.isfinite(observed)]
            if observed.size >= 2:
                result[class_id] = max(
                    0.0,
                    float(observed[:-1].max() - observed[-1]),
                )
        return result

    def backward_transfer(self) -> float:
        values: List[float] = []
        for history in self.class_history.values():
            observed = np.asarray(history, dtype=np.float64)
            observed = observed[np.isfinite(observed)]
            if observed.size >= 2:
                values.append(float(observed[-1] - observed[0]))
        return float(np.mean(values)) if values else 0.0

    def phase_table(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for phase in self.phases_seen:
            metrics = self.phase_history[phase]
            rows.append(
                {
                    "phase": phase,
                    "OA": metrics["overall_accuracy"],
                    "AA": metrics["average_accuracy"],
                    "Kappa": metrics["kappa"],
                    "F1": metrics["f1_macro"],
                    "Old": metrics["old_accuracy"],
                    "New": metrics["new_accuracy"],
                    "H": metrics["harmonic_mean"],
                    "OldToNew": metrics["old_to_new_rate"],
                    "NewToOld": metrics["new_to_old_rate"],
                    "GeometryViolation": metrics.get(
                        "geometry_error_rate",
                        0.0,
                    ),
                    "Q05Gap": metrics.get(
                        "geometry_margin_q05",
                        0.0,
                    ),
                    "Samples": metrics["num_samples"],
                }
            )
        return rows

    def standard_metrics(self) -> Dict[str, float]:
        if not self.phases_seen:
            return {}
        final = self.phase_history[self.phases_seen[-1]]
        forgetting = self.forgetting_per_class()
        incremental_h = [
            self.phase_history[phase]["harmonic_mean"]
            for phase in self.phases_seen
            if self.phase_history[phase]["old_new_split_available"]
        ]
        return {
            "A_last": final["overall_accuracy"],
            "A_avg": float(
                np.mean(
                    [
                        self.phase_history[phase]["overall_accuracy"]
                        for phase in self.phases_seen
                    ]
                )
            ),
            "AA_last": final["average_accuracy"],
            "H_last": final["harmonic_mean"],
            "H_avg_incremental": (
                float(np.mean(incremental_h))
                if incremental_h else 0.0
            ),
            "F_avg": (
                float(np.mean(list(forgetting.values())))
                if forgetting else 0.0
            ),
            "BWT": self.backward_transfer(),
            "Old_last": final["old_accuracy"],
            "New_last": final["new_accuracy"],
            "Kappa_last": final["kappa"],
            "F1_last": final["f1_macro"],
            "OldToNew_last": final["old_to_new_rate"],
            "NewToOld_last": final["new_to_old_rate"],
            "GeometryViolation_last": final.get(
                "geometry_error_rate",
                0.0,
            ),
            "Phases": float(len(self.phases_seen)),
        }

    def per_class_summary(self) -> Dict[int, Dict[str, float]]:
        forgetting = self.forgetting_per_class()
        output: Dict[int, Dict[str, float]] = {}
        for class_id, history in self.class_history.items():
            observed = np.asarray(history, dtype=np.float64)
            phases = np.asarray(self.phases_seen)
            valid = np.isfinite(observed)
            if not valid.any():
                continue
            values = observed[valid]
            present_phases = phases[valid]
            output[class_id] = {
                "introduction_phase": int(present_phases[0]),
                "first_accuracy": float(values[0]),
                "best_accuracy": float(values.max()),
                "last_accuracy": float(values[-1]),
                "forgetting": float(forgetting.get(class_id, 0.0)),
                "backward_transfer": float(values[-1] - values[0]),
            }
        return output

    def to_dict(self) -> Dict[str, Any]:
        return make_json_serializable(
            {
                "phase_history": self.phase_history,
                "phase_table": self.phase_table(),
                "standard_metrics": self.standard_metrics(),
                "per_class_summary": self.per_class_summary(),
            }
        )

    def save(self, save_dir: str) -> Dict[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        json_path = save_json(
            os.path.join(save_dir, "necil_evaluation_summary.json"),
            self.to_dict(),
        )
        csv_path = os.path.join(save_dir, "necil_phase_table.csv")
        rows = self.phase_table()
        if rows:
            with open(
                csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(rows[0]),
                )
                writer.writeheader()
                writer.writerows(rows)
        return {
            "json_path": json_path,
            "phase_table_csv_path": csv_path,
        }

    def print_summary(self) -> None:
        if not self.phases_seen:
            print("[NECILEvaluator] no evaluated phases")
            return
        metrics = self.standard_metrics()
        print("\n" + "=" * 72)
        print("Transport-Verified Factor-Geometry NECIL Summary")
        print("=" * 72)
        print(
            f"A_last={metrics['A_last']:.2f}% | "
            f"A_avg={metrics['A_avg']:.2f}% | "
            f"F_avg={metrics['F_avg']:.2f}% | "
            f"BWT={metrics['BWT']:.2f}%"
        )
        print(
            f"Old={metrics['Old_last']:.2f}% | "
            f"New={metrics['New_last']:.2f}% | "
            f"H={metrics['H_last']:.2f}% | "
            f"GeometryViolation={metrics['GeometryViolation_last']:.2f}%"
        )



















# """Evaluation utilities for PC-SIRG / PC-SGC NECIL-HSI.

# Core contract
# -------------
# - Class ids are global dataset ids and may be non-contiguous.
# - ``seen_classes`` defines the exact classifier column order.
# - Deployed evaluation uses joint occupancy-response energy; feature-only scoring
#   is an explicit ablation, never the default.
# - Evaluation batches must carry paired positive/negative intervention patches
#   and the exact finite-difference step sizes. Raw physical spectra and latent
#   spectral-feature coupling are not part of the scoring path.
# - Predictions outside ``seen_classes`` are reported as leakage, never removed.
# - Kappa is returned as a unitless coefficient in [-1,1]; OA, AA and F1 are
#   percentages.
# """

# from __future__ import annotations

# import csv
# import json
# import os
# from collections import defaultdict
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from sklearn.metrics import classification_report


# _EPS = 1e-12


# # ============================================================
# # Generic conversion / serialization
# # ============================================================
# def make_json_serializable(obj: Any) -> Any:
#     if isinstance(obj, Mapping):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [make_json_serializable(v) for v in obj]
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.bool_):
#         return bool(obj)
#     if torch.is_tensor(obj):
#         if obj.numel() == 1:
#             return obj.detach().cpu().item()
#         return obj.detach().cpu().tolist()
#     return obj


# def _as_1d_np(x: Any, name: str) -> np.ndarray:
#     if torch.is_tensor(x):
#         arr = x.detach().cpu().numpy().reshape(-1)
#     else:
#         arr = np.asarray(x).reshape(-1)
#     if arr.size == 0:
#         raise ValueError(f"{name} is empty.")
#     return arr.astype(np.int64, copy=False)


# def _to_1d_long_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
#     t = x.detach() if torch.is_tensor(x) else torch.as_tensor(x)
#     t = t.long().view(-1)
#     return t.to(device) if device is not None else t


# def _safe_class_name(target_names: Optional[Sequence[str]], cls: int) -> str:
#     cls = int(cls)
#     if target_names is not None and 0 <= cls < len(target_names):
#         return str(target_names[cls])
#     return f"Class {cls}"


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     out: List[int] = []
#     seen = set()
#     for value in values:
#         c = int(value)
#         if c not in seen:
#             out.append(c)
#             seen.add(c)
#     return out


# def _seen_list(
#     seen_classes: Optional[Iterable[int]],
#     y_true: Optional[np.ndarray] = None,
# ) -> Optional[List[int]]:
#     if seen_classes is None:
#         return None
#     out = _ordered_unique_ints(seen_classes)
#     if not out:
#         raise ValueError("seen_classes was provided but empty.")
#     if y_true is not None:
#         true_set = set(int(c) for c in np.unique(y_true).tolist())
#         bad_true = sorted(true_set.difference(out))
#         if bad_true:
#             raise ValueError(
#                 f"y_true contains labels outside seen_classes: {bad_true}. "
#                 "This is a phase/dataset split bug."
#             )
#     return out


# def _resolve_old_new_classes(
#     seen_classes: Sequence[int],
#     *,
#     old_class_count: Optional[int] = None,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
# ) -> Tuple[List[int], List[int]]:
#     """Resolve an exact global-id partition of seen classes.

#     Explicit class lists take precedence. Prefix splitting by ``old_class_count``
#     is retained only for legacy sequential protocols.
#     """
#     seen = _ordered_unique_ints(seen_classes)
#     seen_set = set(seen)

#     old_explicit = old_classes is not None
#     new_explicit = new_classes is not None
#     if old_explicit or new_explicit:
#         old = _ordered_unique_ints(old_classes or [])
#         new = _ordered_unique_ints(new_classes or [])
#         if old_explicit and not new_explicit:
#             new = [c for c in seen if c not in set(old)]
#         elif new_explicit and not old_explicit:
#             old = [c for c in seen if c not in set(new)]
#     else:
#         k = int(old_class_count or 0)
#         if k < 0 or k > len(seen):
#             raise ValueError(
#                 f"old_class_count={k} must be in [0,{len(seen)}] for seen_classes={seen}."
#             )
#         old = seen[:k]
#         new = seen[k:]

#     old_set, new_set = set(old), set(new)
#     if not old_set.issubset(seen_set):
#         raise ValueError(f"old_classes are not a subset of seen_classes: {sorted(old_set - seen_set)}")
#     if not new_set.issubset(seen_set):
#         raise ValueError(f"new_classes are not a subset of seen_classes: {sorted(new_set - seen_set)}")
#     if old_set & new_set:
#         raise ValueError(f"old_classes/new_classes overlap: {sorted(old_set & new_set)}")
#     if old_set | new_set != seen_set:
#         missing = [c for c in seen if c not in old_set and c not in new_set]
#         raise ValueError(f"old/new partition does not cover seen_classes. missing={missing}")

#     return [c for c in seen if c in old_set], [c for c in seen if c in new_set]


# def _filter_true_labels(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     *,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
# ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
#     y_true = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
#     y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64, copy=False)
#     if y_true.shape[0] != y_pred.shape[0]:
#         raise ValueError(f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")

#     valid = np.ones_like(y_true, dtype=bool)
#     if ignore_index is not None:
#         valid &= y_true != int(ignore_index)
#     y_true = y_true[valid]
#     y_pred = y_pred[valid]
#     if y_true.size == 0:
#         raise ValueError("No valid labels remain after ignore_index filtering.")

#     seen = _seen_list(seen_classes, y_true)
#     if seen is None:
#         seen = sorted(int(c) for c in np.unique(y_true).tolist())
#     return y_true, y_pred, seen


# def _prediction_leakage_np(y_pred: np.ndarray, seen: Sequence[int]) -> Dict[str, Any]:
#     seen_set = set(int(c) for c in seen)
#     invalid = np.asarray([int(v) not in seen_set for v in y_pred], dtype=bool)
#     return {
#         "invalid_prediction_rate": float(invalid.mean() * 100.0) if invalid.size else 0.0,
#         "predicted_unseen_count": int(invalid.sum()),
#         "predicted_unseen_classes": (
#             sorted(int(c) for c in np.unique(y_pred[invalid]).tolist()) if invalid.any() else []
#         ),
#         "negative_prediction_count": int((np.asarray(y_pred) < 0).sum()),
#     }


# def _add_metric_aliases(metrics: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(metrics, dict):
#         return metrics
#     metrics.setdefault("oa", float(metrics.get("overall_accuracy", 0.0)))
#     metrics.setdefault("acc", float(metrics.get("overall_accuracy", 0.0)))
#     metrics.setdefault("aa", float(metrics.get("average_accuracy", metrics.get("balanced_accuracy", 0.0))))
#     metrics.setdefault("macro_f1", float(metrics.get("f1_macro", 0.0)))
#     metrics.setdefault("old_acc", float(metrics.get("old_accuracy", 0.0)))
#     metrics.setdefault("new_acc", float(metrics.get("new_accuracy", 0.0)))
#     metrics.setdefault("hm", float(metrics.get("harmonic_mean", 0.0)))
#     metrics.setdefault("h", float(metrics.get("harmonic_mean", 0.0)))
#     metrics.setdefault("invalid", float(metrics.get("invalid_prediction_rate", 0.0)))
#     return metrics


# def _compact_labels(seen: Sequence[int], y_pred: np.ndarray) -> List[int]:
#     seen_list = _ordered_unique_ints(seen)
#     seen_set = set(seen_list)
#     invalid_pred = sorted(int(c) for c in np.unique(y_pred).tolist() if int(c) not in seen_set)
#     return seen_list + invalid_pred


# def _report_class_name(
#     target_names: Optional[Sequence[str]],
#     cls: int,
#     seen: Sequence[int],
# ) -> str:
#     cls = int(cls)
#     if cls in set(int(c) for c in seen):
#         return _safe_class_name(target_names, cls)
#     if cls < 0:
#         return f"INVALID-PRED-{cls}"
#     return f"UNSEEN-PRED-{cls}"


# # ============================================================
# # Strict seen-local score helpers
# # ============================================================
# def validate_seen_local_outputs(
#     *,
#     seen_classes: Iterable[int],
#     logits: Optional[torch.Tensor] = None,
#     energy: Optional[torch.Tensor] = None,
#     batch_size: Optional[int] = None,
# ) -> List[int]:
#     seen = _ordered_unique_ints(seen_classes)
#     if not seen:
#         raise ValueError("seen_classes is empty.")
#     for name, tensor in (("logits", logits), ("energy", energy)):
#         if tensor is None:
#             continue
#         if not torch.is_tensor(tensor) or tensor.dim() != 2:
#             raise RuntimeError(f"{name} must be [B,len(seen_classes)], got {type(tensor)}")
#         if tensor.size(1) != len(seen):
#             raise RuntimeError(
#                 f"{name} width={tensor.size(1)} but len(seen_classes)={len(seen)}. "
#                 "Evaluation requires strict seen-local outputs."
#             )
#         if batch_size is not None and tensor.size(0) != int(batch_size):
#             raise RuntimeError(f"{name} batch={tensor.size(0)} but expected {int(batch_size)}")
#         if not torch.isfinite(tensor).all():
#             raise RuntimeError(f"{name} contains NaN/Inf values.")
#     return seen


# @torch.no_grad()
# def predictions_from_seen_local_scores(
#     scores: torch.Tensor,
#     seen_classes: Iterable[int],
#     *,
#     lower_is_better: bool = False,
# ) -> torch.Tensor:
#     seen = validate_seen_local_outputs(
#         seen_classes=seen_classes,
#         energy=scores if lower_is_better else None,
#         logits=None if lower_is_better else scores,
#     )
#     local = scores.argmin(dim=1) if lower_is_better else scores.argmax(dim=1)
#     mapping = torch.as_tensor(seen, device=scores.device, dtype=torch.long)
#     return mapping.index_select(0, local.long())


# # ============================================================
# # Torch-native compact confusion metrics
# # ============================================================
# @torch.no_grad()
# def torch_confusion_matrix(
#     y_true: Any,
#     y_pred: Any,
#     num_classes: int,
#     device: Optional[str] = "cpu",
# ) -> torch.Tensor:
#     """Compatibility dense confusion matrix for labels ``0..num_classes-1``."""
#     if int(num_classes) <= 0:
#         raise ValueError("num_classes must be positive.")
#     labels = list(range(int(num_classes)))
#     return torch_confusion_matrix_for_labels(y_true, y_pred, labels=labels, device=device)


# @torch.no_grad()
# def torch_confusion_matrix_for_labels(
#     y_true: Any,
#     y_pred: Any,
#     *,
#     labels: Sequence[int],
#     device: Optional[str] = "cpu",
#     strict_true_labels: bool = True,
# ) -> torch.Tensor:
#     """Rows=true labels, columns=predicted labels in explicit compact order."""
#     label_list = _ordered_unique_ints(labels)
#     if not label_list:
#         raise ValueError("labels must be non-empty.")
#     dev = torch.device(device) if device is not None else None
#     yt = _to_1d_long_tensor(y_true, dev)
#     yp = _to_1d_long_tensor(y_pred, dev)
#     if yt.numel() != yp.numel():
#         raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

#     true_local = torch.full_like(yt, -1)
#     pred_local = torch.full_like(yp, -1)
#     for i, cls in enumerate(label_list):
#         true_local[yt == int(cls)] = int(i)
#         pred_local[yp == int(cls)] = int(i)

#     if strict_true_labels and bool((true_local < 0).any().item()):
#         bad = torch.unique(yt[true_local < 0]).detach().cpu().tolist()
#         raise ValueError(f"True labels are absent from confusion labels: {bad}")
#     if bool((pred_local < 0).any().item()):
#         bad = torch.unique(yp[pred_local < 0]).detach().cpu().tolist()
#         raise ValueError(
#             f"Predictions are absent from confusion labels: {bad}. "
#             "Include every predicted invalid/unseen id in labels."
#         )

#     k = len(label_list)
#     idx = true_local * k + pred_local
#     return torch.bincount(idx, minlength=k * k).reshape(k, k)


# @torch.no_grad()
# def torch_metrics_from_confusion_matrix(
#     cm: torch.Tensor,
#     eps: float = _EPS,
#     *,
#     class_ids: Optional[Sequence[int]] = None,
# ) -> Dict[str, Any]:
#     cm = cm.detach().float()
#     if cm.dim() != 2 or cm.size(0) != cm.size(1):
#         raise ValueError(f"confusion matrix must be square [C,C], got {tuple(cm.shape)}")
#     ids = list(range(cm.size(0))) if class_ids is None else _ordered_unique_ints(class_ids)
#     if len(ids) != cm.size(0):
#         raise ValueError(f"class_ids length={len(ids)} but confusion size={cm.size(0)}")

#     tp = torch.diag(cm)
#     support = cm.sum(dim=1)
#     predicted = cm.sum(dim=0)
#     total = cm.sum()
#     recall = tp / (support + eps)
#     precision = tp / (predicted + eps)
#     f1 = 2.0 * precision * recall / (precision + recall + eps)
#     valid_true = support > 0

#     oa = 100.0 * tp.sum() / (total + eps)
#     aa = 100.0 * recall[valid_true].mean() if bool(valid_true.any().item()) else cm.new_tensor(0.0)
#     macro_f1 = 100.0 * f1[valid_true].mean() if bool(valid_true.any().item()) else cm.new_tensor(0.0)
#     po = tp.sum() / (total + eps)
#     pe = (support * predicted).sum() / (total * total + eps)
#     kappa = (po - pe) / (1.0 - pe + eps)

#     return {
#         "overall_accuracy": float(oa.item()),
#         "balanced_accuracy": float(aa.item()),
#         "average_accuracy": float(aa.item()),
#         "kappa": float(kappa.item()),
#         "f1_macro": float(macro_f1.item()),
#         "per_class_accuracy": {int(ids[i]): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
#         "precision": {int(ids[i]): float((100.0 * precision[i]).item()) for i in range(cm.size(0))},
#         "recall": {int(ids[i]): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
#         "f1_per_class": {int(ids[i]): float((100.0 * f1[i]).item()) for i in range(cm.size(0))},
#         "support": {int(ids[i]): int(support[i].item()) for i in range(cm.size(0))},
#         "predicted_count": {int(ids[i]): int(predicted[i].item()) for i in range(cm.size(0))},
#         "confusion_matrix": cm.detach().cpu(),
#         "confusion_matrix_labels": [int(c) for c in ids],
#     }


# @torch.no_grad()
# def torch_old_new_metrics(
#     y_true: Any,
#     y_pred: Any,
#     old_class_count: Optional[int] = None,
#     *,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
#     eps: float = _EPS,
# ) -> Dict[str, Any]:
#     yt = _to_1d_long_tensor(y_true)
#     yp = _to_1d_long_tensor(y_pred, yt.device)
#     if yt.numel() != yp.numel():
#         raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

#     seen = (
#         _ordered_unique_ints(seen_classes)
#         if seen_classes is not None
#         else sorted(int(c) for c in torch.unique(yt).detach().cpu().tolist())
#     )
#     old_ids, new_ids = _resolve_old_new_classes(
#         seen,
#         old_class_count=old_class_count,
#         old_classes=old_classes,
#         new_classes=new_classes,
#     )
#     seen_t = torch.as_tensor(seen, device=yt.device, dtype=torch.long)
#     old_t = torch.as_tensor(old_ids, device=yt.device, dtype=torch.long)
#     new_t = torch.as_tensor(new_ids, device=yt.device, dtype=torch.long)

#     def membership(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
#         if ids.numel() == 0:
#             return torch.zeros_like(values, dtype=torch.bool)
#         return (values.view(-1, 1) == ids.view(1, -1)).any(dim=1)

#     old_true = membership(yt, old_t)
#     new_true = membership(yt, new_t)
#     pred_old = membership(yp, old_t)
#     pred_new = membership(yp, new_t)
#     pred_seen = membership(yp, seen_t)
#     pred_invalid = ~pred_seen
#     correct = yp.eq(yt)

#     overall = 100.0 * correct.float().mean() if yt.numel() else yt.new_tensor(0.0, dtype=torch.float32)
#     old_total = int(old_true.sum().item())
#     new_total = int(new_true.sum().item())
#     old_acc = 100.0 * correct[old_true].float().mean() if old_total else overall.new_tensor(0.0)
#     new_acc = 100.0 * correct[new_true].float().mean() if new_total else overall.new_tensor(0.0)

#     old_to_new_count = int((old_true & pred_new).sum().item())
#     new_to_old_count = int((new_true & pred_old).sum().item())
#     old_invalid_count = int((old_true & pred_invalid).sum().item())
#     new_invalid_count = int((new_true & pred_invalid).sum().item())

#     split_available = old_total > 0 and new_total > 0
#     if split_available:
#         h = 2.0 * old_acc * new_acc / (old_acc + new_acc + eps)
#     else:
#         h = overall

#     return {
#         "old_accuracy": float(old_acc.item()),
#         "new_accuracy": float(new_acc.item()),
#         "harmonic_mean": float(h.item()),
#         "old_count": old_total,
#         "new_count": new_total,
#         "old_class_ids": [int(c) for c in old_ids],
#         "new_class_ids": [int(c) for c in new_ids],
#         "old_new_split_available": bool(split_available),
#         "old_to_new_rate": float(100.0 * old_to_new_count / max(old_total, 1)),
#         "new_to_old_rate": float(100.0 * new_to_old_count / max(new_total, 1)),
#         "old_to_new_count": old_to_new_count,
#         "new_to_old_count": new_to_old_count,
#         "old_invalid_prediction_rate": float(100.0 * old_invalid_count / max(old_total, 1)),
#         "new_invalid_prediction_rate": float(100.0 * new_invalid_count / max(new_total, 1)),
#         "old_invalid_prediction_count": old_invalid_count,
#         "new_invalid_prediction_count": new_invalid_count,
#     }


# @torch.no_grad()
# def geometry_energy_diagnostics(
#     energy: torch.Tensor,
#     labels_global: Any,
#     *,
#     seen_classes: Iterable[int],
#     old_class_count: Optional[int] = None,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
#     margin: float = 0.0,
# ) -> Dict[str, Any]:
#     """Evaluate geometry-energy health in the strict seen-local class order."""
#     seen = validate_seen_local_outputs(seen_classes=seen_classes, energy=energy)
#     y_global = _to_1d_long_tensor(labels_global, energy.device)
#     if y_global.numel() != energy.size(0):
#         raise ValueError(f"energy/labels mismatch: {energy.size(0)} vs {y_global.numel()}")
#     mapping = {int(c): i for i, c in enumerate(seen)}
#     y_local = torch.full_like(y_global, -1)
#     for cls, local in mapping.items():
#         y_local[y_global == int(cls)] = int(local)
#     if bool((y_local < 0).any().item()):
#         bad = torch.unique(y_global[y_local < 0]).detach().cpu().tolist()
#         raise ValueError(f"labels_global contains classes outside seen_classes: {bad}")

#     e = energy.float()
#     true_e = e.gather(1, y_local.view(-1, 1)).squeeze(1)
#     true_mask = torch.zeros_like(e, dtype=torch.bool).scatter(1, y_local.view(-1, 1), True)
#     rival_e = e.masked_fill(true_mask, float("inf")).min(dim=1).values
#     gap = rival_e - true_e
#     pred_local = e.argmin(dim=1)

#     old_ids, new_ids = _resolve_old_new_classes(
#         seen,
#         old_class_count=old_class_count,
#         old_classes=old_classes,
#         new_classes=new_classes,
#     )
#     old_cols = torch.as_tensor([seen.index(c) for c in old_ids], device=e.device, dtype=torch.long)
#     new_cols = torch.as_tensor([seen.index(c) for c in new_ids], device=e.device, dtype=torch.long)
#     old_true = torch.zeros_like(y_global, dtype=torch.bool)
#     new_true = torch.zeros_like(y_global, dtype=torch.bool)
#     for c in old_ids:
#         old_true |= y_global.eq(int(c))
#     for c in new_ids:
#         new_true |= y_global.eq(int(c))

#     def pct(mask: torch.Tensor) -> float:
#         return float(100.0 * mask.float().mean().item()) if mask.numel() else 0.0

#     per_class: Dict[int, Dict[str, float]] = {}
#     for cls in seen:
#         mask = y_global.eq(int(cls))
#         cls_gap = gap[mask]
#         per_class[int(cls)] = {
#             "count": int(mask.sum().item()),
#             "accuracy": float(100.0 * pred_local[mask].eq(seen.index(cls)).float().mean().item()) if bool(mask.any().item()) else 0.0,
#             "mean_margin": float(cls_gap.mean().item()) if cls_gap.numel() else 0.0,
#             "min_margin": float(cls_gap.min().item()) if cls_gap.numel() else 0.0,
#             "margin_violation_rate": pct(cls_gap <= float(margin)) if cls_gap.numel() else 0.0,
#             "energy_error_rate": pct(cls_gap <= 0.0) if cls_gap.numel() else 0.0,
#         }

#     zero = e.new_zeros(())
#     old_opposite_gap = e.new_empty((0,))
#     new_opposite_gap = e.new_empty((0,))
#     if old_cols.numel() and new_cols.numel():
#         old_min = e.index_select(1, old_cols).min(dim=1).values
#         new_min = e.index_select(1, new_cols).min(dim=1).values
#         if bool(old_true.any().item()):
#             old_opposite_gap = new_min[old_true] - true_e[old_true]
#         if bool(new_true.any().item()):
#             new_opposite_gap = old_min[new_true] - true_e[new_true]

#     return {
#         "geometry_energy_accuracy": float(100.0 * pred_local.eq(y_local).float().mean().item()),
#         "geometry_margin_mean": float(gap.mean().item()) if gap.numel() else 0.0,
#         "geometry_margin_min": float(gap.min().item()) if gap.numel() else 0.0,
#         "geometry_margin_violation_rate": pct(gap <= float(margin)),
#         "geometry_error_rate": pct(gap <= 0.0),
#         "geometry_margin_threshold": float(margin),
#         "old_to_new_energy_invasion_rate": pct(old_opposite_gap <= float(margin)) if old_opposite_gap.numel() else 0.0,
#         "new_to_old_energy_invasion_rate": pct(new_opposite_gap <= float(margin)) if new_opposite_gap.numel() else 0.0,
#         "old_to_new_energy_error_rate": pct(old_opposite_gap <= 0.0) if old_opposite_gap.numel() else 0.0,
#         "new_to_old_energy_error_rate": pct(new_opposite_gap <= 0.0) if new_opposite_gap.numel() else 0.0,
#         "old_to_new_energy_gap_mean": float(old_opposite_gap.mean().item()) if old_opposite_gap.numel() else float(zero.item()),
#         "new_to_old_energy_gap_mean": float(new_opposite_gap.mean().item()) if new_opposite_gap.numel() else float(zero.item()),
#         "per_class_geometry": per_class,
#     }


# @torch.no_grad()
# def calculate_metrics_torch(
#     y_true: Any,
#     y_pred: Any,
#     num_classes: Optional[int] = None,
#     old_class_count: Optional[int] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
#     device: str = "cpu",
#     *,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
#     energy: Optional[torch.Tensor] = None,
#     energy_margin: float = 0.0,
# ) -> Dict[str, Any]:
#     yt, yp, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )
#     labels_full = _compact_labels(seen, yp)
#     cm_full = torch_confusion_matrix_for_labels(
#         yt,
#         yp,
#         labels=labels_full,
#         device=device,
#         strict_true_labels=True,
#     )
#     all_metrics = torch_metrics_from_confusion_matrix(cm_full, class_ids=labels_full)
#     metrics = dict(all_metrics)

#     # Preserve all-label metrics for leakage reports, then expose seen-only
#     # dictionaries under the historical public keys.
#     for key in ("per_class_accuracy", "precision", "recall", "f1_per_class", "support", "predicted_count"):
#         metrics[f"{key}_all"] = dict(all_metrics[key])
#         metrics[key] = {int(c): all_metrics[key].get(int(c), 0.0) for c in seen}

#     split = torch_old_new_metrics(
#         yt,
#         yp,
#         old_class_count=old_class_count,
#         seen_classes=seen,
#         old_classes=old_classes,
#         new_classes=new_classes,
#     )
#     metrics.update(split)
#     metrics.update(_prediction_leakage_np(yp, seen))
#     metrics["negative_prediction_overflow_class"] = None
#     metrics["num_samples"] = int(yt.size)
#     metrics["num_classes"] = int(len(seen))
#     metrics["classes"] = [int(c) for c in seen]
#     metrics["requested_num_classes"] = None if num_classes is None else int(num_classes)
#     metrics["confusion_matrix_full"] = cm_full.detach().cpu()
#     metrics["confusion_matrix_labels"] = [int(c) for c in labels_full]
#     seen_pos = torch.as_tensor([labels_full.index(int(c)) for c in seen], dtype=torch.long)
#     metrics["confusion_matrix_seen"] = cm_full.detach().cpu().index_select(0, seen_pos).index_select(1, seen_pos)
#     metrics["confusion_matrix_seen_labels"] = [int(c) for c in seen]
#     metrics["confusion_matrix"] = cm_full.detach().cpu()

#     if energy is not None:
#         e = energy
#         if ignore_index is not None:
#             raw_true = _as_1d_np(y_true, "y_true")
#             keep = raw_true != int(ignore_index)
#             e = energy[torch.as_tensor(keep, device=energy.device, dtype=torch.bool)]
#         metrics.update(
#             geometry_energy_diagnostics(
#                 e,
#                 yt,
#                 seen_classes=seen,
#                 old_class_count=old_class_count,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#                 margin=float(energy_margin),
#             )
#         )
#     return _add_metric_aliases(metrics)


# def calculate_metrics(
#     y_true: Any,
#     y_pred: Any,
#     class_names: Optional[List[str]] = None,
#     old_class_count: Optional[int] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
#     *,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
# ) -> Dict[str, Any]:
#     del class_names
#     return calculate_metrics_torch(
#         y_true=y_true,
#         y_pred=y_pred,
#         num_classes=None,
#         old_class_count=old_class_count,
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#         device="cpu",
#         old_classes=old_classes,
#         new_classes=new_classes,
#     )



# # ============================================================
# # PC-SIRG loader evaluation
# # ============================================================

# def _mapping_value(batch: Mapping[str, Any], names: Sequence[str]) -> Any:
#     for name in names:
#         if name in batch and batch[name] is not None:
#             return batch[name]
#     return None


# def _unpack_pc_sirg_eval_batch(
#     batch: Any,
# ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Return original patches, labels, coords, paired views and step sizes."""
#     if not isinstance(batch, Mapping):
#         raise RuntimeError(
#             "PC-SIRG evaluation requires mapping batches with paired response views"
#         )
#     patches = _mapping_value(batch, ("image", "patch", "patches", "x", "input"))
#     labels = _mapping_value(batch, ("label", "labels", "target", "y"))
#     coords = _mapping_value(batch, ("coord", "coords", "coordinate"))
#     positive = _mapping_value(
#         batch,
#         ("spectral_positive_patches", "positive_patches", "response_positive_patches"),
#     )
#     negative = _mapping_value(
#         batch,
#         ("spectral_negative_patches", "negative_patches", "response_negative_patches"),
#     )
#     steps = _mapping_value(
#         batch,
#         ("spectral_step_sizes", "step_sizes", "response_step_sizes"),
#     )
#     missing = [
#         name
#         for name, value in (
#             ("image", patches),
#             ("label", labels),
#             ("spectral_positive_patches", positive),
#             ("spectral_negative_patches", negative),
#             ("spectral_step_sizes", steps),
#         )
#         if value is None
#     ]
#     if missing:
#         raise RuntimeError(f"evaluation batch is missing {missing}")

#     x = torch.as_tensor(patches)
#     y = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
#     coord_t = None if coords is None else torch.as_tensor(coords, dtype=torch.long)
#     pos = torch.as_tensor(positive)
#     neg = torch.as_tensor(negative)
#     step = torch.as_tensor(steps)
#     batch_size = int(x.shape[0])
#     if x.ndim != 4:
#         raise RuntimeError(f"evaluation patches must be [B,C,H,W], got {tuple(x.shape)}")
#     if y.numel() != batch_size:
#         raise RuntimeError("evaluation patch/label mismatch")
#     if coord_t is not None and coord_t.shape != (batch_size, 2):
#         raise RuntimeError(f"evaluation coordinates must be [B,2], got {tuple(coord_t.shape)}")
#     if pos.ndim != 5 or pos.shape != neg.shape:
#         raise RuntimeError("paired intervention views must share [B,K,C,H,W]")
#     if pos.shape[0] != batch_size or tuple(pos.shape[2:]) != tuple(x.shape[1:]):
#         raise RuntimeError("paired intervention views are not aligned with patches")
#     if step.ndim == 1:
#         if step.numel() == pos.shape[1]:
#             step = step.unsqueeze(0).expand(batch_size, -1)
#         elif step.numel() == batch_size and pos.shape[1] == 1:
#             step = step[:, None]
#         else:
#             raise RuntimeError("1-D step sizes must contain K values")
#     if step.ndim != 2 or tuple(step.shape) != tuple(pos.shape[:2]):
#         raise RuntimeError(f"step sizes must be [B,K], got {tuple(step.shape)}")
#     for name, tensor in (("patches", x), ("positive", pos), ("negative", neg), ("steps", step)):
#         if not torch.isfinite(tensor.float()).all():
#             raise RuntimeError(f"{name} contain NaN/Inf")
#     if bool((step.abs() <= 1e-12).any().item()):
#         raise RuntimeError("step sizes must be non-zero")
#     return x, y, coord_t, pos, neg, step


# @torch.no_grad()
# def evaluate_pc_sirg_loader(
#     model: torch.nn.Module,
#     loader: Any,
#     *,
#     seen_classes: Iterable[int],
#     device: str | torch.device,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
#     energy_margin: float = 0.0,
#     return_arrays: bool = True,
# ) -> Dict[str, Any]:
#     """Evaluate the deployed PC-SIRG joint occupancy-response classifier.

#     This function is the single recommended evaluation path for base validation,
#     base testing, incremental testing and map prediction diagnostics. It does not
#     accept raw physical spectra because they are not part of the deployed score.
#     """
#     seen = _ordered_unique_ints(seen_classes)
#     if not seen:
#         raise ValueError("seen_classes is empty")
#     requested_device = torch.device(device)
#     if requested_device.type == "cuda" and not torch.cuda.is_available():
#         raise RuntimeError("CUDA was requested but is unavailable")

#     extractor = getattr(model, "extract_canonical_geometry_features", None)
#     response_extractor = getattr(model, "compute_spectral_responses_from_views", None)
#     scorer = getattr(model, "compute_logits_from_features", None)
#     if not callable(extractor) or not callable(response_extractor) or not callable(scorer):
#         raise RuntimeError(
#             "model must expose extract_canonical_geometry_features(), "
#             "compute_spectral_responses_from_views(), and compute_logits_from_features()"
#         )

#     previous_training = bool(model.training)
#     model.eval()
#     y_true_chunks: List[torch.Tensor] = []
#     y_pred_chunks: List[torch.Tensor] = []
#     energy_chunks: List[torch.Tensor] = []
#     occupancy_chunks: List[torch.Tensor] = []
#     response_chunks: List[torch.Tensor] = []
#     coord_chunks: List[torch.Tensor] = []
#     seen_tensor = torch.as_tensor(seen, device=requested_device, dtype=torch.long)
#     try:
#         for batch in loader:
#             x, y, coords, positive, negative, steps = _unpack_pc_sirg_eval_batch(batch)
#             x = x.to(requested_device, dtype=torch.float32, non_blocking=True)
#             y = y.to(requested_device, dtype=torch.long, non_blocking=True)
#             positive = positive.to(requested_device, dtype=torch.float32, non_blocking=True)
#             negative = negative.to(requested_device, dtype=torch.float32, non_blocking=True)
#             steps = steps.to(requested_device, dtype=torch.float32, non_blocking=True)

#             unexpected = sorted(set(int(v) for v in y.detach().cpu().tolist()) - set(seen))
#             if unexpected:
#                 raise RuntimeError(f"evaluation batch contains unseen true classes {unexpected}")

#             extracted = extractor(x, deterministic=True, return_dict=True)
#             features = extracted.get("features") if isinstance(extracted, Mapping) else extracted
#             if not torch.is_tensor(features) or features.ndim != 2:
#                 raise RuntimeError("canonical feature extraction returned no [B,D] tensor")
#             response_payload = response_extractor(
#                 positive,
#                 negative,
#                 step_sizes=steps,
#                 deterministic=True,
#             )
#             responses = response_payload.get("spectral_responses")
#             if not torch.is_tensor(responses) or responses.ndim != 3:
#                 raise RuntimeError("spectral response extraction returned no [B,K,D] tensor")

#             scored = scorer(
#                 features,
#                 spectral_responses=responses,
#                 seen_classes=seen,
#                 mode="pc_sirg",
#                 return_energy=True,
#                 return_parts=True,
#             )
#             if not isinstance(scored, Mapping):
#                 raise RuntimeError("PC-SIRG scorer must return a mapping")
#             energy = scored.get("energy", scored.get("joint_energy", scored.get("raw_energy")))
#             logits = scored.get("logits")
#             occupancy = scored.get("occupancy_energy")
#             response = scored.get("response_energy")
#             validate_seen_local_outputs(
#                 seen_classes=seen,
#                 logits=logits,
#                 energy=energy,
#                 batch_size=x.size(0),
#             )
#             if not torch.is_tensor(occupancy) or occupancy.shape != energy.shape:
#                 raise RuntimeError("PC-SIRG scorer returned no occupancy_energy")
#             if not torch.is_tensor(response) or response.shape != energy.shape:
#                 raise RuntimeError("PC-SIRG scorer returned no response_energy")

#             pred_global = seen_tensor.index_select(0, energy.argmin(dim=1))
#             y_true_chunks.append(y.detach().cpu())
#             y_pred_chunks.append(pred_global.detach().cpu())
#             energy_chunks.append(energy.detach().cpu())
#             occupancy_chunks.append(occupancy.detach().cpu())
#             response_chunks.append(response.detach().cpu())
#             if coords is not None:
#                 coord_chunks.append(coords.detach().cpu())
#     finally:
#         model.train(previous_training)

#     if not y_true_chunks:
#         raise RuntimeError("evaluation loader is empty")
#     y_true = torch.cat(y_true_chunks)
#     y_pred = torch.cat(y_pred_chunks)
#     energy = torch.cat(energy_chunks)
#     occupancy_energy = torch.cat(occupancy_chunks)
#     response_energy = torch.cat(response_chunks)

#     metrics = calculate_metrics_torch(
#         y_true=y_true,
#         y_pred=y_pred,
#         seen_classes=seen,
#         old_classes=old_classes,
#         new_classes=new_classes,
#         device="cpu",
#         energy=energy,
#         energy_margin=float(energy_margin),
#     )
#     local_by_global = {class_id: index for index, class_id in enumerate(seen)}
#     local_true = torch.as_tensor([local_by_global[int(v)] for v in y_true.tolist()], dtype=torch.long)
#     occupancy_pred = occupancy_energy.argmin(dim=1)
#     response_pred = response_energy.argmin(dim=1)
#     joint_pred = energy.argmin(dim=1)
#     occupancy_correct = occupancy_pred.eq(local_true)
#     response_correct = response_pred.eq(local_true)
#     joint_correct = joint_pred.eq(local_true)
#     metrics.update(
#         {
#             "joint_accuracy": 100.0 * float(joint_correct.float().mean().item()),
#             "occupancy_only_accuracy": 100.0 * float(occupancy_correct.float().mean().item()),
#             "response_only_accuracy": 100.0 * float(response_correct.float().mean().item()),
#             "response_help_rate": float(((~occupancy_correct) & joint_correct).float().mean().item()),
#             "response_harm_rate": float((occupancy_correct & (~joint_correct)).float().mean().item()),
#             "response_help_minus_harm": float(
#                 (((~occupancy_correct) & joint_correct).float().mean()
#                  - (occupancy_correct & (~joint_correct)).float().mean()).item()
#             ),
#         }
#     )
#     output: Dict[str, Any] = {
#         "metrics": metrics,
#         "seen_classes": seen,
#     }
#     if return_arrays:
#         output.update(
#             {
#                 "y_true": y_true.numpy(),
#                 "y_pred": y_pred.numpy(),
#                 "joint_energy": energy,
#                 "occupancy_energy": occupancy_energy,
#                 "response_energy": response_energy,
#                 "coords": torch.cat(coord_chunks).numpy() if coord_chunks else None,
#             }
#         )
#     return output


# # ============================================================
# # Report writing
# # ============================================================
# def _report_labels(
#     seen: Sequence[int],
#     y_pred: np.ndarray,
#     include_predicted_unseen: bool,
# ) -> List[int]:
#     if include_predicted_unseen:
#         return _compact_labels(seen, y_pred)
#     return _ordered_unique_ints(seen)


# def save_structured_classification_report(
#     y_true: Any,
#     y_pred: Any,
#     target_names: Optional[List[str]] = None,
#     save_dir: str = "./results",
#     phase: int = 0,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_class_count: Optional[int] = None,
#     ignore_index: Optional[int] = None,
#     include_predicted_unseen: bool = True,
#     *,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
#     energy: Optional[torch.Tensor] = None,
#     energy_margin: float = 0.0,
# ) -> Dict[str, Any]:
#     os.makedirs(save_dir, exist_ok=True)
#     yt, yp, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )
#     labels = _report_labels(seen, yp, include_predicted_unseen)
#     names = [_report_class_name(target_names, c, seen) for c in labels]

#     metrics = calculate_metrics_torch(
#         y_true=yt,
#         y_pred=yp,
#         num_classes=None,
#         old_class_count=old_class_count,
#         seen_classes=seen,
#         ignore_index=None,
#         device="cpu",
#         old_classes=old_classes,
#         new_classes=new_classes,
#         energy=energy,
#         energy_margin=energy_margin,
#     )
#     full_labels = metrics["confusion_matrix_labels"]
#     cm_full = metrics["confusion_matrix_full"].detach().cpu().numpy()
#     positions = [full_labels.index(int(c)) for c in labels]
#     cm_report = cm_full[np.ix_(positions, positions)]

#     report_dict = classification_report(
#         yt,
#         yp,
#         labels=labels,
#         target_names=names,
#         zero_division=0,
#         output_dict=True,
#     )
#     report_text = classification_report(
#         yt,
#         yp,
#         labels=labels,
#         target_names=names,
#         zero_division=0,
#         digits=4,
#     )
#     for cls, name in zip(labels, names):
#         report_dict.setdefault(name, {})
#         report_dict[name]["precision"] = float(metrics["precision_all"].get(cls, 0.0)) / 100.0
#         report_dict[name]["recall"] = float(metrics["recall_all"].get(cls, 0.0)) / 100.0
#         report_dict[name]["f1-score"] = float(metrics["f1_per_class_all"].get(cls, 0.0)) / 100.0
#         report_dict[name]["support"] = int(metrics["support_all"].get(cls, 0))
#     report_dict["torch_metrics"] = make_json_serializable(metrics)
#     report_dict["old_new_split"] = {
#         k: metrics.get(k)
#         for k in (
#             "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
#             "old_class_ids", "new_class_ids", "old_new_split_available",
#             "old_to_new_rate", "new_to_old_rate", "old_to_new_count", "new_to_old_count",
#             "old_invalid_prediction_rate", "new_invalid_prediction_rate",
#         )
#     }

#     base = f"phase_{int(phase)}"
#     txt_path = os.path.join(save_dir, f"{base}_classification_report.txt")
#     json_path = os.path.join(save_dir, f"{base}_classification_report.json")
#     cm_csv_path = os.path.join(save_dir, f"{base}_confusion_matrix.csv")
#     cm_npy_path = os.path.join(save_dir, f"{base}_confusion_matrix.npy")
#     per_class_csv_path = os.path.join(save_dir, f"{base}_per_class_metrics.csv")

#     with open(txt_path, "w", encoding="utf-8") as f:
#         f.write(f"Classification Report - Phase {phase}\n")
#         f.write("=" * 90 + "\n\n")
#         f.write(f"Seen class order: {seen}\n")
#         f.write(f"Old class IDs: {metrics.get('old_class_ids', [])}\n")
#         f.write(f"New class IDs: {metrics.get('new_class_ids', [])}\n")
#         f.write(f"OA: {metrics['overall_accuracy']:.4f}%\n")
#         f.write(f"AA: {metrics['average_accuracy']:.4f}%\n")
#         f.write(f"Kappa: {metrics['kappa']:.4f}\n")
#         f.write(f"Macro-F1: {metrics['f1_macro']:.4f}%\n")
#         f.write(f"Old Accuracy: {metrics.get('old_accuracy', 0.0):.4f}%\n")
#         f.write(f"New Accuracy: {metrics.get('new_accuracy', 0.0):.4f}%\n")
#         f.write(f"Harmonic Mean: {metrics.get('harmonic_mean', 0.0):.4f}%\n")
#         f.write(f"Invalid Prediction Rate: {metrics.get('invalid_prediction_rate', 0.0):.4f}%\n")
#         f.write(f"Predicted unseen IDs: {metrics.get('predicted_unseen_classes', [])}\n")
#         f.write(f"Old -> New error rate: {metrics.get('old_to_new_rate', 0.0):.4f}%\n")
#         f.write(f"New -> Old error rate: {metrics.get('new_to_old_rate', 0.0):.4f}%\n")
#         if "geometry_energy_accuracy" in metrics:
#             f.write(f"Geometry-energy accuracy: {metrics['geometry_energy_accuracy']:.4f}%\n")
#             f.write(f"Geometry margin violation: {metrics['geometry_margin_violation_rate']:.4f}%\n")
#             f.write(f"Old -> New energy invasion: {metrics['old_to_new_energy_invasion_rate']:.4f}%\n")
#             f.write(f"New -> Old energy invasion: {metrics['new_to_old_energy_invasion_rate']:.4f}%\n")
#         f.write("\n" + report_text)

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(report_dict), f, indent=2)

#     with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["true\\pred"] + [f"{name} [{cls}]" for cls, name in zip(labels, names)])
#         for i, (cls, name) in enumerate(zip(labels, names)):
#             writer.writerow([f"{name} [{cls}]"] + [int(v) for v in cm_report[i].tolist()])
#     np.save(cm_npy_path, cm_report)

#     seen_set = set(seen)
#     with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "class_id", "class_name", "seen_class", "old_class", "new_class",
#             "precision_percent", "recall_percent", "f1_percent", "accuracy_percent",
#             "support", "predicted_count",
#         ])
#         old_set = set(metrics.get("old_class_ids", []))
#         new_set = set(metrics.get("new_class_ids", []))
#         for cls, name in zip(labels, names):
#             writer.writerow([
#                 int(cls), name, bool(cls in seen_set), bool(cls in old_set), bool(cls in new_set),
#                 float(metrics["precision_all"].get(cls, 0.0)),
#                 float(metrics["recall_all"].get(cls, 0.0)),
#                 float(metrics["f1_per_class_all"].get(cls, 0.0)),
#                 float(metrics["per_class_accuracy_all"].get(cls, 0.0)),
#                 int(metrics["support_all"].get(cls, 0)),
#                 int(metrics["predicted_count_all"].get(cls, 0)),
#             ])

#     print(f"[Report] Saved structured classification report: {txt_path}")
#     return {
#         "txt_path": txt_path,
#         "json_path": json_path,
#         "confusion_matrix_csv_path": cm_csv_path,
#         "confusion_matrix_npy_path": cm_npy_path,
#         "per_class_csv_path": per_class_csv_path,
#         "report": report_dict,
#         "torch_metrics": metrics,
#         "confusion_matrix": cm_report,
#         "labels": labels,
#         "names": names,
#     }


# def save_hsi_style_classification_report(
#     y_true: Any,
#     y_pred: Any,
#     target_names: Optional[List[str]] = None,
#     save_path: str = "./classification_report.csv",
#     tr_time: Optional[float] = None,
#     te_time: Optional[float] = None,
#     dl_time: Optional[float] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_class_count: Optional[int] = None,
#     ignore_index: Optional[int] = None,
#     *,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
# ) -> Dict[str, Any]:
#     parent = os.path.dirname(save_path)
#     if parent:
#         os.makedirs(parent, exist_ok=True)
#     yt, yp, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )
#     metrics = calculate_metrics_torch(
#         yt,
#         yp,
#         old_class_count=old_class_count,
#         seen_classes=seen,
#         device="cpu",
#         old_classes=old_classes,
#         new_classes=new_classes,
#     )
#     labels = list(seen)
#     names = [_safe_class_name(target_names, c) for c in labels]
#     cm_seen = metrics["confusion_matrix_seen"].detach().cpu().numpy()
#     per_class_acc = [float(metrics["per_class_accuracy"].get(int(c), 0.0)) for c in labels]
#     report_text = classification_report(yt, yp, labels=labels, target_names=names, digits=4, zero_division=0)

#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write(f"{0.0 if tr_time is None else float(tr_time)} Tr_Time\n")
#         f.write(f"{0.0 if te_time is None else float(te_time)} Te_Time\n")
#         f.write(f"{0.0 if dl_time is None else float(dl_time)} DL_Time\n")
#         f.write(f"{seen} Seen class order\n")
#         f.write(f"{metrics.get('old_class_ids', [])} Old class ids\n")
#         f.write(f"{metrics.get('new_class_ids', [])} New class ids\n")
#         f.write(f"{metrics['kappa']} Kappa coefficient\n")
#         f.write(f"{metrics['overall_accuracy']} Overall accuracy (%)\n")
#         f.write(f"{metrics['average_accuracy']} Average accuracy (%)\n")
#         f.write(f"{metrics['f1_macro']} Macro F1 (%)\n")
#         f.write(f"{metrics.get('old_accuracy', 0.0)} Old accuracy (%)\n")
#         f.write(f"{metrics.get('new_accuracy', 0.0)} New accuracy (%)\n")
#         f.write(f"{metrics.get('harmonic_mean', 0.0)} Harmonic mean (%)\n")
#         f.write(f"{metrics.get('invalid_prediction_rate', 0.0)} Invalid prediction rate (%)\n")
#         f.write(f"{metrics.get('old_to_new_rate', 0.0)} Old-to-new error rate (%)\n")
#         f.write(f"{metrics.get('new_to_old_rate', 0.0)} New-to-old error rate (%)\n")
#         f.write(report_text + "\n")
#         f.write(str(np.asarray(per_class_acc)) + "\n")
#         f.write(str(cm_seen) + "\n")

#     print(f"[Report] Saved HSI-style classification report: {save_path}")
#     return {
#         "save_path": save_path,
#         "overall_accuracy": metrics["overall_accuracy"],
#         "average_accuracy": metrics["average_accuracy"],
#         "kappa": metrics["kappa"],
#         "f1_macro": metrics["f1_macro"],
#         "per_class_accuracy": per_class_acc,
#         "confusion_matrix": cm_seen,
#         "old_new_split": {
#             k: metrics.get(k)
#             for k in (
#                 "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
#                 "old_class_ids", "new_class_ids", "old_new_split_available",
#             )
#         },
#         "torch_metrics": metrics,
#     }


# def save_classification_report(
#     y_true: Any,
#     y_pred: Any,
#     target_names: Optional[List[str]] = None,
#     save_dir: str = "./results",
#     phase: int = 0,
#     seen_classes: Optional[Iterable[int]] = None,
#     old_class_count: Optional[int] = None,
#     ignore_index: Optional[int] = None,
#     include_predicted_unseen: bool = True,
#     save_hsi_style: bool = True,
#     save_structured: bool = True,
#     tr_time: Optional[float] = None,
#     te_time: Optional[float] = None,
#     dl_time: Optional[float] = None,
#     save_path: Optional[str] = None,
#     *,
#     old_classes: Optional[Iterable[int]] = None,
#     new_classes: Optional[Iterable[int]] = None,
#     energy: Optional[torch.Tensor] = None,
#     energy_margin: float = 0.0,
# ) -> Dict[str, Any]:
#     if save_path is not None:
#         return save_hsi_style_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_path=save_path,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#             old_classes=old_classes,
#             new_classes=new_classes,
#         )

#     os.makedirs(save_dir, exist_ok=True)
#     output: Dict[str, Any] = {}
#     if save_structured:
#         output["structured"] = save_structured_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_dir=save_dir,
#             phase=phase,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#             include_predicted_unseen=include_predicted_unseen,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             energy=energy,
#             energy_margin=energy_margin,
#         )
#     if save_hsi_style:
#         hsi_path = os.path.join(save_dir, f"phase_{phase}_HSI_Classification_Report.csv")
#         output["hsi_style"] = save_hsi_style_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_path=hsi_path,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#             old_classes=old_classes,
#             new_classes=new_classes,
#         )
#     if "structured" in output:
#         for key in (
#             "txt_path", "json_path", "confusion_matrix_csv_path",
#             "confusion_matrix_npy_path", "per_class_csv_path",
#         ):
#             output[key] = output["structured"].get(key)
#     if "hsi_style" in output:
#         output["hsi_style_path"] = output["hsi_style"].get("save_path")
#     return output


# # ============================================================
# # NECIL evaluator
# # ============================================================
# class NECILEvaluator:
#     def __init__(self) -> None:
#         self.phase_history: Dict[int, Dict[str, Any]] = {}
#         self.class_acc_history: defaultdict[int, List[float]] = defaultdict(list)
#         self.class_presence_history: defaultdict[int, List[bool]] = defaultdict(list)
#         self.class_introduction_phase: Dict[int, int] = {}
#         self.phases_seen: List[int] = []

#     def _sanity_check_labels(self, y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
#         yt = _as_1d_np(y_true, "y_true")
#         yp = _as_1d_np(y_pred, "y_pred")
#         if yt.shape[0] != yp.shape[0]:
#             raise ValueError(f"y_true/y_pred length mismatch: {len(yt)} vs {len(yp)}")
#         return yt, yp

#     def update(
#         self,
#         phase: int,
#         y_true: Any,
#         y_pred: Any,
#         old_class_count: Optional[int] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#         ignore_index: Optional[int] = None,
#         *,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         energy: Optional[torch.Tensor] = None,
#         energy_margin: float = 0.0,
#     ) -> None:
#         phase = int(phase)
#         yt, yp = self._sanity_check_labels(y_true, y_pred)
#         seen = (
#             _ordered_unique_ints(seen_classes)
#             if seen_classes is not None
#             else sorted(np.unique(yt).astype(int).tolist())
#         )
#         metrics = calculate_metrics_torch(
#             y_true=yt,
#             y_pred=yp,
#             old_class_count=old_class_count,
#             seen_classes=seen,
#             ignore_index=ignore_index,
#             device="cpu",
#             old_classes=old_classes,
#             new_classes=new_classes,
#             energy=energy,
#             energy_margin=energy_margin,
#         )
#         self.phase_history[phase] = metrics
#         if phase not in self.phases_seen:
#             self.phases_seen.append(phase)
#             self.phases_seen.sort()
#         for cls in metrics.get("new_class_ids", []):
#             self.class_introduction_phase.setdefault(int(cls), phase)
#         if phase == min(self.phases_seen):
#             for cls in metrics.get("classes", []):
#                 self.class_introduction_phase.setdefault(int(cls), phase)
#         self._rebuild_class_history()

#     def save_phase_report(
#         self,
#         phase: int,
#         y_true: Any,
#         y_pred: Any,
#         target_names: Optional[List[str]] = None,
#         save_dir: str = "./results",
#         seen_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         ignore_index: Optional[int] = None,
#         tr_time: Optional[float] = None,
#         te_time: Optional[float] = None,
#         dl_time: Optional[float] = None,
#         *,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         energy: Optional[torch.Tensor] = None,
#         energy_margin: float = 0.0,
#     ) -> Dict[str, Any]:
#         return save_classification_report(
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_dir=save_dir,
#             phase=phase,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             ignore_index=ignore_index,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=dl_time,
#             save_hsi_style=True,
#             save_structured=True,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             energy=energy,
#             energy_margin=energy_margin,
#         )

#     def _rebuild_class_history(self) -> None:
#         # A class is considered present only when the evaluated split contains
#         # at least one true sample. Merely being listed in seen_classes is not
#         # enough; otherwise a zero-support validation split creates fake 0%
#         # accuracy and artificial forgetting.
#         all_classes = sorted({
#             int(c)
#             for phase in self.phases_seen
#             for c, support in self.phase_history[phase].get("support", {}).items()
#             if int(support) > 0
#         })
#         introductions: Dict[int, int] = {}
#         for cls in all_classes:
#             phases_with_support = [
#                 phase
#                 for phase in self.phases_seen
#                 if int(self.phase_history[phase].get("support", {}).get(cls, 0)) > 0
#             ]
#             if phases_with_support:
#                 introductions[int(cls)] = int(min(phases_with_support))
#         self.class_introduction_phase = introductions

#         new_hist: defaultdict[int, List[float]] = defaultdict(list)
#         presence: defaultdict[int, List[bool]] = defaultdict(list)
#         for cls in all_classes:
#             for phase in self.phases_seen:
#                 per_class = self.phase_history[phase].get("per_class_accuracy", {})
#                 support = int(self.phase_history[phase].get("support", {}).get(cls, 0))
#                 present = support > 0 and cls in per_class
#                 new_hist[cls].append(float(per_class[cls]) if present else float("nan"))
#                 presence[cls].append(bool(present))
#         self.class_acc_history = new_hist
#         self.class_presence_history = presence

#     def calculate_forgetting_per_class(self) -> Dict[int, float]:
#         if len(self.phases_seen) < 2:
#             return {}
#         forgetting: Dict[int, float] = {}
#         for cls, history in self.class_acc_history.items():
#             vals = np.asarray(history, dtype=float)
#             vals = vals[~np.isnan(vals)]
#             if vals.size >= 2:
#                 forgetting[int(cls)] = float(max(0.0, float(np.max(vals[:-1])) - float(vals[-1])))
#         return forgetting

#     def calculate_backward_transfer(self) -> float:
#         """Final accuracy minus accuracy at each class's introduction phase."""
#         values = []
#         for history in self.class_acc_history.values():
#             vals = np.asarray(history, dtype=float)
#             vals = vals[~np.isnan(vals)]
#             if vals.size >= 2:
#                 values.append(float(vals[-1] - vals[0]))
#         return float(np.mean(values)) if values else 0.0

#     def get_standard_metrics(self) -> Dict[str, float]:
#         if not self.phases_seen:
#             return {}
#         last_phase = self.phases_seen[-1]
#         all_oa = [float(self.phase_history[p].get("overall_accuracy", 0.0)) for p in self.phases_seen]
#         inc_h = [
#             float(self.phase_history[p].get("harmonic_mean", 0.0))
#             for p in self.phases_seen
#             if bool(self.phase_history[p].get("old_new_split_available", False))
#         ]
#         forgetting = self.calculate_forgetting_per_class()
#         last = self.phase_history[last_phase]
#         return {
#             "A_last (Final Accuracy)": float(last.get("overall_accuracy", 0.0)),
#             "A_avg (Avg Accuracy)": float(np.mean(all_oa)) if all_oa else 0.0,
#             "H_last (Final Harmonic Mean)": float(last.get("harmonic_mean", 0.0)),
#             "H_avg (Avg Harmonic Mean)": float(np.mean(inc_h)) if inc_h else 0.0,
#             "H_avg_inc_only": float(np.mean(inc_h)) if inc_h else 0.0,
#             "F_avg (Avg Forgetting)": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
#             "BWT (Backward Transfer)": self.calculate_backward_transfer(),
#             "Old_last (Final Old Accuracy)": float(last.get("old_accuracy", 0.0)),
#             "New_last (Final New Accuracy)": float(last.get("new_accuracy", 0.0)),
#             "AA_last (Final Avg Accuracy)": float(last.get("average_accuracy", 0.0)),
#             "Kappa_last": float(last.get("kappa", 0.0)),
#             "F1_last": float(last.get("f1_macro", 0.0)),
#             "Invalid_last": float(last.get("invalid_prediction_rate", 0.0)),
#             "OldToNew_last": float(last.get("old_to_new_rate", 0.0)),
#             "NewToOld_last": float(last.get("new_to_old_rate", 0.0)),
#             "GeometryViolation_last": float(last.get("geometry_margin_violation_rate", 0.0)),
#             "Phases": len(self.phases_seen),
#         }

#     def get_phase_table(self) -> List[Dict[str, Any]]:
#         rows = []
#         for phase in self.phases_seen:
#             m = self.phase_history[phase]
#             rows.append({
#                 "phase": int(phase),
#                 "OA": float(m.get("overall_accuracy", 0.0)),
#                 "AA": float(m.get("average_accuracy", 0.0)),
#                 "Kappa": float(m.get("kappa", 0.0)),
#                 "F1": float(m.get("f1_macro", 0.0)),
#                 "Old": float(m.get("old_accuracy", 0.0)),
#                 "New": float(m.get("new_accuracy", 0.0)),
#                 "H": float(m.get("harmonic_mean", 0.0)),
#                 "Invalid": float(m.get("invalid_prediction_rate", 0.0)),
#                 "OldToNew": float(m.get("old_to_new_rate", 0.0)),
#                 "NewToOld": float(m.get("new_to_old_rate", 0.0)),
#                 "EnergyViolation": float(m.get("geometry_margin_violation_rate", 0.0)),
#                 "OldEnergyInvasion": float(m.get("old_to_new_energy_invasion_rate", 0.0)),
#                 "NewEnergyInvasion": float(m.get("new_to_old_energy_invasion_rate", 0.0)),
#                 "SplitAvailable": bool(m.get("old_new_split_available", False)),
#                 "Samples": int(m.get("num_samples", 0)),
#             })
#         return rows

#     def get_per_class_summary(self) -> Dict[int, Dict[str, float]]:
#         forgetting = self.calculate_forgetting_per_class()
#         out: Dict[int, Dict[str, float]] = {}
#         for cls, history in self.class_acc_history.items():
#             vals = np.asarray(history, dtype=float)
#             vals = vals[~np.isnan(vals)]
#             if vals.size:
#                 out[int(cls)] = {
#                     "introduction_phase": float(self.class_introduction_phase.get(int(cls), 0)),
#                     "first": float(vals[0]),
#                     "best": float(np.max(vals)),
#                     "last": float(vals[-1]),
#                     "forgetting": float(forgetting.get(int(cls), 0.0)),
#                     "backward_transfer": float(vals[-1] - vals[0]),
#                 }
#         return out

#     def print_summary(self) -> None:
#         if not self.phases_seen:
#             print("[NECILEvaluator] No phases evaluated yet.")
#             return
#         metrics = self.get_standard_metrics()
#         last_phase = self.phases_seen[-1]
#         m = self.phase_history[last_phase]
#         print("\n" + "=" * 64)
#         print(f" NECIL-HSI Evaluation Report (Phase {last_phase})")
#         print("=" * 64)
#         print(f" 1. Final Accuracy (A_last):       {metrics.get('A_last (Final Accuracy)', 0):.2f}%")
#         print(f" 2. Avg Accuracy (A_avg):          {metrics.get('A_avg (Avg Accuracy)', 0):.2f}%")
#         print(f" 3. Avg Forgetting (F_avg):        {metrics.get('F_avg (Avg Forgetting)', 0):.2f}%")
#         print(f" 4. Backward Transfer (BWT):       {metrics.get('BWT (Backward Transfer)', 0):.2f}%")
#         print(f" 5. Old / New Accuracy:            {m.get('old_accuracy', 0):.2f}% / {m.get('new_accuracy', 0):.2f}%")
#         print(f" 6. Harmonic Mean:                 {m.get('harmonic_mean', 0):.2f}%")
#         print(f" 7. Invalid Prediction Rate:       {m.get('invalid_prediction_rate', 0):.2f}%")
#         print(f" 8. Old -> New / New -> Old:       {m.get('old_to_new_rate', 0):.2f}% / {m.get('new_to_old_rate', 0):.2f}%")
#         print(f" 9. Geometry Margin Violation:     {m.get('geometry_margin_violation_rate', 0):.2f}%")
#         print(f"10. Energy Old->New / New->Old:    {m.get('old_to_new_energy_invasion_rate', 0):.2f}% / {m.get('new_to_old_energy_invasion_rate', 0):.2f}%")
#         print(f"11. AA / Kappa / F1:               {m.get('average_accuracy', 0):.2f}% / {m.get('kappa', 0):.2f}% / {m.get('f1_macro', 0):.2f}%")
#         print("-" * 64)

#     def save_phase_table_csv(self, save_path: str) -> str:
#         parent = os.path.dirname(save_path)
#         if parent:
#             os.makedirs(parent, exist_ok=True)
#         rows = self.get_phase_table()
#         fields = [
#             "phase", "OA", "AA", "Kappa", "F1", "Old", "New", "H", "Invalid",
#             "OldToNew", "NewToOld", "EnergyViolation", "OldEnergyInvasion",
#             "NewEnergyInvasion", "SplitAvailable", "Samples",
#         ]
#         with open(save_path, "w", encoding="utf-8", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fields)
#             writer.writeheader()
#             writer.writerows(rows)
#         return save_path

#     def to_dict(self) -> Dict[str, Any]:
#         return make_json_serializable({
#             "phase_history": self.phase_history,
#             "class_introduction_phase": self.class_introduction_phase,
#             "standard_metrics": self.get_standard_metrics(),
#             "phase_table": self.get_phase_table(),
#             "per_class_summary": self.get_per_class_summary(),
#         })

