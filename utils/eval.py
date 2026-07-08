"""Evaluation utilities for spectral-coupled geometry replay NECIL-HSI.

Core contract
-------------
- Class ids are global dataset ids and may be non-contiguous.
- ``seen_classes`` defines the exact classifier column order for a phase.
- ``old_classes``/``new_classes`` are explicit global-id partitions whenever
  available. ``old_class_count`` is only a compatibility fallback.
- Predictions outside ``seen_classes`` are counted as errors and reported as
  leakage; they are never silently removed.
- Confusion matrices are built in a compact explicit-label space, so sparse or
  large global ids do not allocate huge dense matrices.
- Optional geometry-energy diagnostics use the same seen-class order as the
  classifier and expose margin violations and bidirectional old/new invasion.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report


_EPS = 1e-12


# ============================================================
# Generic conversion / serialization
# ============================================================
def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def _as_1d_np(x: Any, name: str) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr.astype(np.int64, copy=False)


def _to_1d_long_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    t = x.detach() if torch.is_tensor(x) else torch.as_tensor(x)
    t = t.long().view(-1)
    return t.to(device) if device is not None else t


def _safe_class_name(target_names: Optional[Sequence[str]], cls: int) -> str:
    cls = int(cls)
    if target_names is not None and 0 <= cls < len(target_names):
        return str(target_names[cls])
    return f"Class {cls}"


def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        c = int(value)
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _seen_list(
    seen_classes: Optional[Iterable[int]],
    y_true: Optional[np.ndarray] = None,
) -> Optional[List[int]]:
    if seen_classes is None:
        return None
    out = _ordered_unique_ints(seen_classes)
    if not out:
        raise ValueError("seen_classes was provided but empty.")
    if y_true is not None:
        true_set = set(int(c) for c in np.unique(y_true).tolist())
        bad_true = sorted(true_set.difference(out))
        if bad_true:
            raise ValueError(
                f"y_true contains labels outside seen_classes: {bad_true}. "
                "This is a phase/dataset split bug."
            )
    return out


def _resolve_old_new_classes(
    seen_classes: Sequence[int],
    *,
    old_class_count: Optional[int] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Resolve an exact global-id partition of seen classes.

    Explicit class lists take precedence. Prefix splitting by ``old_class_count``
    is retained only for legacy sequential protocols.
    """
    seen = _ordered_unique_ints(seen_classes)
    seen_set = set(seen)

    old_explicit = old_classes is not None
    new_explicit = new_classes is not None
    if old_explicit or new_explicit:
        old = _ordered_unique_ints(old_classes or [])
        new = _ordered_unique_ints(new_classes or [])
        if old_explicit and not new_explicit:
            new = [c for c in seen if c not in set(old)]
        elif new_explicit and not old_explicit:
            old = [c for c in seen if c not in set(new)]
    else:
        k = int(old_class_count or 0)
        if k < 0 or k > len(seen):
            raise ValueError(
                f"old_class_count={k} must be in [0,{len(seen)}] for seen_classes={seen}."
            )
        old = seen[:k]
        new = seen[k:]

    old_set, new_set = set(old), set(new)
    if not old_set.issubset(seen_set):
        raise ValueError(f"old_classes are not a subset of seen_classes: {sorted(old_set - seen_set)}")
    if not new_set.issubset(seen_set):
        raise ValueError(f"new_classes are not a subset of seen_classes: {sorted(new_set - seen_set)}")
    if old_set & new_set:
        raise ValueError(f"old_classes/new_classes overlap: {sorted(old_set & new_set)}")
    if old_set | new_set != seen_set:
        missing = [c for c in seen if c not in old_set and c not in new_set]
        raise ValueError(f"old/new partition does not cover seen_classes. missing={missing}")

    return [c for c in seen if c in old_set], [c for c in seen if c in new_set]


def _filter_true_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64, copy=False)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")

    valid = np.ones_like(y_true, dtype=bool)
    if ignore_index is not None:
        valid &= y_true != int(ignore_index)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        raise ValueError("No valid labels remain after ignore_index filtering.")

    seen = _seen_list(seen_classes, y_true)
    if seen is None:
        seen = sorted(int(c) for c in np.unique(y_true).tolist())
    return y_true, y_pred, seen


def _prediction_leakage_np(y_pred: np.ndarray, seen: Sequence[int]) -> Dict[str, Any]:
    seen_set = set(int(c) for c in seen)
    invalid = np.asarray([int(v) not in seen_set for v in y_pred], dtype=bool)
    return {
        "invalid_prediction_rate": float(invalid.mean() * 100.0) if invalid.size else 0.0,
        "predicted_unseen_count": int(invalid.sum()),
        "predicted_unseen_classes": (
            sorted(int(c) for c in np.unique(y_pred[invalid]).tolist()) if invalid.any() else []
        ),
        "negative_prediction_count": int((np.asarray(y_pred) < 0).sum()),
    }


def _add_metric_aliases(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return metrics
    metrics.setdefault("oa", float(metrics.get("overall_accuracy", 0.0)))
    metrics.setdefault("acc", float(metrics.get("overall_accuracy", 0.0)))
    metrics.setdefault("aa", float(metrics.get("average_accuracy", metrics.get("balanced_accuracy", 0.0))))
    metrics.setdefault("macro_f1", float(metrics.get("f1_macro", 0.0)))
    metrics.setdefault("old_acc", float(metrics.get("old_accuracy", 0.0)))
    metrics.setdefault("new_acc", float(metrics.get("new_accuracy", 0.0)))
    metrics.setdefault("hm", float(metrics.get("harmonic_mean", 0.0)))
    metrics.setdefault("h", float(metrics.get("harmonic_mean", 0.0)))
    metrics.setdefault("invalid", float(metrics.get("invalid_prediction_rate", 0.0)))
    return metrics


def _compact_labels(seen: Sequence[int], y_pred: np.ndarray) -> List[int]:
    seen_list = _ordered_unique_ints(seen)
    seen_set = set(seen_list)
    invalid_pred = sorted(int(c) for c in np.unique(y_pred).tolist() if int(c) not in seen_set)
    return seen_list + invalid_pred


def _report_class_name(
    target_names: Optional[Sequence[str]],
    cls: int,
    seen: Sequence[int],
) -> str:
    cls = int(cls)
    if cls in set(int(c) for c in seen):
        return _safe_class_name(target_names, cls)
    if cls < 0:
        return f"INVALID-PRED-{cls}"
    return f"UNSEEN-PRED-{cls}"


# ============================================================
# Strict seen-local score helpers
# ============================================================
def validate_seen_local_outputs(
    *,
    seen_classes: Iterable[int],
    logits: Optional[torch.Tensor] = None,
    energy: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> List[int]:
    seen = _ordered_unique_ints(seen_classes)
    if not seen:
        raise ValueError("seen_classes is empty.")
    for name, tensor in (("logits", logits), ("energy", energy)):
        if tensor is None:
            continue
        if not torch.is_tensor(tensor) or tensor.dim() != 2:
            raise RuntimeError(f"{name} must be [B,len(seen_classes)], got {type(tensor)}")
        if tensor.size(1) != len(seen):
            raise RuntimeError(
                f"{name} width={tensor.size(1)} but len(seen_classes)={len(seen)}. "
                "Evaluation requires strict seen-local outputs."
            )
        if batch_size is not None and tensor.size(0) != int(batch_size):
            raise RuntimeError(f"{name} batch={tensor.size(0)} but expected {int(batch_size)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN/Inf values.")
    return seen


@torch.no_grad()
def predictions_from_seen_local_scores(
    scores: torch.Tensor,
    seen_classes: Iterable[int],
    *,
    lower_is_better: bool = False,
) -> torch.Tensor:
    seen = validate_seen_local_outputs(
        seen_classes=seen_classes,
        energy=scores if lower_is_better else None,
        logits=None if lower_is_better else scores,
    )
    local = scores.argmin(dim=1) if lower_is_better else scores.argmax(dim=1)
    mapping = torch.as_tensor(seen, device=scores.device, dtype=torch.long)
    return mapping.index_select(0, local.long())


# ============================================================
# Torch-native compact confusion metrics
# ============================================================
@torch.no_grad()
def torch_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    num_classes: int,
    device: Optional[str] = "cpu",
) -> torch.Tensor:
    """Compatibility dense confusion matrix for labels ``0..num_classes-1``."""
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive.")
    labels = list(range(int(num_classes)))
    return torch_confusion_matrix_for_labels(y_true, y_pred, labels=labels, device=device)


@torch.no_grad()
def torch_confusion_matrix_for_labels(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Sequence[int],
    device: Optional[str] = "cpu",
    strict_true_labels: bool = True,
) -> torch.Tensor:
    """Rows=true labels, columns=predicted labels in explicit compact order."""
    label_list = _ordered_unique_ints(labels)
    if not label_list:
        raise ValueError("labels must be non-empty.")
    dev = torch.device(device) if device is not None else None
    yt = _to_1d_long_tensor(y_true, dev)
    yp = _to_1d_long_tensor(y_pred, dev)
    if yt.numel() != yp.numel():
        raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

    true_local = torch.full_like(yt, -1)
    pred_local = torch.full_like(yp, -1)
    for i, cls in enumerate(label_list):
        true_local[yt == int(cls)] = int(i)
        pred_local[yp == int(cls)] = int(i)

    if strict_true_labels and bool((true_local < 0).any().item()):
        bad = torch.unique(yt[true_local < 0]).detach().cpu().tolist()
        raise ValueError(f"True labels are absent from confusion labels: {bad}")
    if bool((pred_local < 0).any().item()):
        bad = torch.unique(yp[pred_local < 0]).detach().cpu().tolist()
        raise ValueError(
            f"Predictions are absent from confusion labels: {bad}. "
            "Include every predicted invalid/unseen id in labels."
        )

    k = len(label_list)
    idx = true_local * k + pred_local
    return torch.bincount(idx, minlength=k * k).reshape(k, k)


@torch.no_grad()
def torch_metrics_from_confusion_matrix(
    cm: torch.Tensor,
    eps: float = _EPS,
    *,
    class_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    cm = cm.detach().float()
    if cm.dim() != 2 or cm.size(0) != cm.size(1):
        raise ValueError(f"confusion matrix must be square [C,C], got {tuple(cm.shape)}")
    ids = list(range(cm.size(0))) if class_ids is None else _ordered_unique_ints(class_ids)
    if len(ids) != cm.size(0):
        raise ValueError(f"class_ids length={len(ids)} but confusion size={cm.size(0)}")

    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    total = cm.sum()
    recall = tp / (support + eps)
    precision = tp / (predicted + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    valid_true = support > 0

    oa = 100.0 * tp.sum() / (total + eps)
    aa = 100.0 * recall[valid_true].mean() if bool(valid_true.any().item()) else cm.new_tensor(0.0)
    macro_f1 = 100.0 * f1[valid_true].mean() if bool(valid_true.any().item()) else cm.new_tensor(0.0)
    po = tp.sum() / (total + eps)
    pe = (support * predicted).sum() / (total * total + eps)
    kappa = 100.0 * (po - pe) / (1.0 - pe + eps)

    return {
        "overall_accuracy": float(oa.item()),
        "balanced_accuracy": float(aa.item()),
        "average_accuracy": float(aa.item()),
        "kappa": float(kappa.item()),
        "f1_macro": float(macro_f1.item()),
        "per_class_accuracy": {int(ids[i]): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
        "precision": {int(ids[i]): float((100.0 * precision[i]).item()) for i in range(cm.size(0))},
        "recall": {int(ids[i]): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
        "f1_per_class": {int(ids[i]): float((100.0 * f1[i]).item()) for i in range(cm.size(0))},
        "support": {int(ids[i]): int(support[i].item()) for i in range(cm.size(0))},
        "predicted_count": {int(ids[i]): int(predicted[i].item()) for i in range(cm.size(0))},
        "confusion_matrix": cm.detach().cpu(),
        "confusion_matrix_labels": [int(c) for c in ids],
    }


@torch.no_grad()
def torch_old_new_metrics(
    y_true: Any,
    y_pred: Any,
    old_class_count: Optional[int] = None,
    *,
    seen_classes: Optional[Iterable[int]] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    eps: float = _EPS,
) -> Dict[str, Any]:
    yt = _to_1d_long_tensor(y_true)
    yp = _to_1d_long_tensor(y_pred, yt.device)
    if yt.numel() != yp.numel():
        raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

    seen = (
        _ordered_unique_ints(seen_classes)
        if seen_classes is not None
        else sorted(int(c) for c in torch.unique(yt).detach().cpu().tolist())
    )
    old_ids, new_ids = _resolve_old_new_classes(
        seen,
        old_class_count=old_class_count,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    seen_t = torch.as_tensor(seen, device=yt.device, dtype=torch.long)
    old_t = torch.as_tensor(old_ids, device=yt.device, dtype=torch.long)
    new_t = torch.as_tensor(new_ids, device=yt.device, dtype=torch.long)

    def membership(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if ids.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        return (values.view(-1, 1) == ids.view(1, -1)).any(dim=1)

    old_true = membership(yt, old_t)
    new_true = membership(yt, new_t)
    pred_old = membership(yp, old_t)
    pred_new = membership(yp, new_t)
    pred_seen = membership(yp, seen_t)
    pred_invalid = ~pred_seen
    correct = yp.eq(yt)

    overall = 100.0 * correct.float().mean() if yt.numel() else yt.new_tensor(0.0, dtype=torch.float32)
    old_total = int(old_true.sum().item())
    new_total = int(new_true.sum().item())
    old_acc = 100.0 * correct[old_true].float().mean() if old_total else overall.new_tensor(0.0)
    new_acc = 100.0 * correct[new_true].float().mean() if new_total else overall.new_tensor(0.0)

    old_to_new_count = int((old_true & pred_new).sum().item())
    new_to_old_count = int((new_true & pred_old).sum().item())
    old_invalid_count = int((old_true & pred_invalid).sum().item())
    new_invalid_count = int((new_true & pred_invalid).sum().item())

    split_available = old_total > 0 and new_total > 0
    if split_available:
        h = 2.0 * old_acc * new_acc / (old_acc + new_acc + eps)
    else:
        h = overall

    return {
        "old_accuracy": float(old_acc.item()),
        "new_accuracy": float(new_acc.item()),
        "harmonic_mean": float(h.item()),
        "old_count": old_total,
        "new_count": new_total,
        "old_class_ids": [int(c) for c in old_ids],
        "new_class_ids": [int(c) for c in new_ids],
        "old_new_split_available": bool(split_available),
        "old_to_new_rate": float(100.0 * old_to_new_count / max(old_total, 1)),
        "new_to_old_rate": float(100.0 * new_to_old_count / max(new_total, 1)),
        "old_to_new_count": old_to_new_count,
        "new_to_old_count": new_to_old_count,
        "old_invalid_prediction_rate": float(100.0 * old_invalid_count / max(old_total, 1)),
        "new_invalid_prediction_rate": float(100.0 * new_invalid_count / max(new_total, 1)),
        "old_invalid_prediction_count": old_invalid_count,
        "new_invalid_prediction_count": new_invalid_count,
    }


@torch.no_grad()
def geometry_energy_diagnostics(
    energy: torch.Tensor,
    labels_global: Any,
    *,
    seen_classes: Iterable[int],
    old_class_count: Optional[int] = None,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    margin: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate geometry-energy health in the strict seen-local class order."""
    seen = validate_seen_local_outputs(seen_classes=seen_classes, energy=energy)
    y_global = _to_1d_long_tensor(labels_global, energy.device)
    if y_global.numel() != energy.size(0):
        raise ValueError(f"energy/labels mismatch: {energy.size(0)} vs {y_global.numel()}")
    mapping = {int(c): i for i, c in enumerate(seen)}
    y_local = torch.full_like(y_global, -1)
    for cls, local in mapping.items():
        y_local[y_global == int(cls)] = int(local)
    if bool((y_local < 0).any().item()):
        bad = torch.unique(y_global[y_local < 0]).detach().cpu().tolist()
        raise ValueError(f"labels_global contains classes outside seen_classes: {bad}")

    e = energy.float()
    true_e = e.gather(1, y_local.view(-1, 1)).squeeze(1)
    true_mask = torch.zeros_like(e, dtype=torch.bool).scatter(1, y_local.view(-1, 1), True)
    rival_e = e.masked_fill(true_mask, float("inf")).min(dim=1).values
    gap = rival_e - true_e
    pred_local = e.argmin(dim=1)

    old_ids, new_ids = _resolve_old_new_classes(
        seen,
        old_class_count=old_class_count,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    old_cols = torch.as_tensor([seen.index(c) for c in old_ids], device=e.device, dtype=torch.long)
    new_cols = torch.as_tensor([seen.index(c) for c in new_ids], device=e.device, dtype=torch.long)
    old_true = torch.zeros_like(y_global, dtype=torch.bool)
    new_true = torch.zeros_like(y_global, dtype=torch.bool)
    for c in old_ids:
        old_true |= y_global.eq(int(c))
    for c in new_ids:
        new_true |= y_global.eq(int(c))

    def pct(mask: torch.Tensor) -> float:
        return float(100.0 * mask.float().mean().item()) if mask.numel() else 0.0

    per_class: Dict[int, Dict[str, float]] = {}
    for cls in seen:
        mask = y_global.eq(int(cls))
        cls_gap = gap[mask]
        per_class[int(cls)] = {
            "count": int(mask.sum().item()),
            "accuracy": float(100.0 * pred_local[mask].eq(seen.index(cls)).float().mean().item()) if bool(mask.any().item()) else 0.0,
            "mean_margin": float(cls_gap.mean().item()) if cls_gap.numel() else 0.0,
            "min_margin": float(cls_gap.min().item()) if cls_gap.numel() else 0.0,
            "margin_violation_rate": pct(cls_gap <= float(margin)) if cls_gap.numel() else 0.0,
            "energy_error_rate": pct(cls_gap <= 0.0) if cls_gap.numel() else 0.0,
        }

    zero = e.new_zeros(())
    old_opposite_gap = e.new_empty((0,))
    new_opposite_gap = e.new_empty((0,))
    if old_cols.numel() and new_cols.numel():
        old_min = e.index_select(1, old_cols).min(dim=1).values
        new_min = e.index_select(1, new_cols).min(dim=1).values
        if bool(old_true.any().item()):
            old_opposite_gap = new_min[old_true] - true_e[old_true]
        if bool(new_true.any().item()):
            new_opposite_gap = old_min[new_true] - true_e[new_true]

    return {
        "geometry_energy_accuracy": float(100.0 * pred_local.eq(y_local).float().mean().item()),
        "geometry_margin_mean": float(gap.mean().item()) if gap.numel() else 0.0,
        "geometry_margin_min": float(gap.min().item()) if gap.numel() else 0.0,
        "geometry_margin_violation_rate": pct(gap <= float(margin)),
        "geometry_error_rate": pct(gap <= 0.0),
        "geometry_margin_threshold": float(margin),
        "old_to_new_energy_invasion_rate": pct(old_opposite_gap <= float(margin)) if old_opposite_gap.numel() else 0.0,
        "new_to_old_energy_invasion_rate": pct(new_opposite_gap <= float(margin)) if new_opposite_gap.numel() else 0.0,
        "old_to_new_energy_error_rate": pct(old_opposite_gap <= 0.0) if old_opposite_gap.numel() else 0.0,
        "new_to_old_energy_error_rate": pct(new_opposite_gap <= 0.0) if new_opposite_gap.numel() else 0.0,
        "old_to_new_energy_gap_mean": float(old_opposite_gap.mean().item()) if old_opposite_gap.numel() else float(zero.item()),
        "new_to_old_energy_gap_mean": float(new_opposite_gap.mean().item()) if new_opposite_gap.numel() else float(zero.item()),
        "per_class_geometry": per_class,
    }


@torch.no_grad()
def calculate_metrics_torch(
    y_true: Any,
    y_pred: Any,
    num_classes: Optional[int] = None,
    old_class_count: Optional[int] = None,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
    device: str = "cpu",
    *,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Dict[str, Any]:
    yt, yp, seen = _filter_true_labels(
        _as_1d_np(y_true, "y_true"),
        _as_1d_np(y_pred, "y_pred"),
        seen_classes=seen_classes,
        ignore_index=ignore_index,
    )
    labels_full = _compact_labels(seen, yp)
    cm_full = torch_confusion_matrix_for_labels(
        yt,
        yp,
        labels=labels_full,
        device=device,
        strict_true_labels=True,
    )
    all_metrics = torch_metrics_from_confusion_matrix(cm_full, class_ids=labels_full)
    metrics = dict(all_metrics)

    # Preserve all-label metrics for leakage reports, then expose seen-only
    # dictionaries under the historical public keys.
    for key in ("per_class_accuracy", "precision", "recall", "f1_per_class", "support", "predicted_count"):
        metrics[f"{key}_all"] = dict(all_metrics[key])
        metrics[key] = {int(c): all_metrics[key].get(int(c), 0.0) for c in seen}

    split = torch_old_new_metrics(
        yt,
        yp,
        old_class_count=old_class_count,
        seen_classes=seen,
        old_classes=old_classes,
        new_classes=new_classes,
    )
    metrics.update(split)
    metrics.update(_prediction_leakage_np(yp, seen))
    metrics["negative_prediction_overflow_class"] = None
    metrics["num_samples"] = int(yt.size)
    metrics["num_classes"] = int(len(seen))
    metrics["classes"] = [int(c) for c in seen]
    metrics["requested_num_classes"] = None if num_classes is None else int(num_classes)
    metrics["confusion_matrix_full"] = cm_full.detach().cpu()
    metrics["confusion_matrix_labels"] = [int(c) for c in labels_full]
    seen_pos = torch.as_tensor([labels_full.index(int(c)) for c in seen], dtype=torch.long)
    metrics["confusion_matrix_seen"] = cm_full.detach().cpu().index_select(0, seen_pos).index_select(1, seen_pos)
    metrics["confusion_matrix_seen_labels"] = [int(c) for c in seen]
    metrics["confusion_matrix"] = cm_full.detach().cpu()

    if energy is not None:
        e = energy
        if ignore_index is not None:
            raw_true = _as_1d_np(y_true, "y_true")
            keep = raw_true != int(ignore_index)
            e = energy[torch.as_tensor(keep, device=energy.device, dtype=torch.bool)]
        metrics.update(
            geometry_energy_diagnostics(
                e,
                yt,
                seen_classes=seen,
                old_class_count=old_class_count,
                old_classes=old_classes,
                new_classes=new_classes,
                margin=float(energy_margin),
            )
        )
    return _add_metric_aliases(metrics)


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    class_names: Optional[List[str]] = None,
    old_class_count: Optional[int] = None,
    seen_classes: Optional[Iterable[int]] = None,
    ignore_index: Optional[int] = None,
    *,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    del class_names
    return calculate_metrics_torch(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=None,
        old_class_count=old_class_count,
        seen_classes=seen_classes,
        ignore_index=ignore_index,
        device="cpu",
        old_classes=old_classes,
        new_classes=new_classes,
    )


# ============================================================
# Report writing
# ============================================================
def _report_labels(
    seen: Sequence[int],
    y_pred: np.ndarray,
    include_predicted_unseen: bool,
) -> List[int]:
    if include_predicted_unseen:
        return _compact_labels(seen, y_pred)
    return _ordered_unique_ints(seen)


def save_structured_classification_report(
    y_true: Any,
    y_pred: Any,
    target_names: Optional[List[str]] = None,
    save_dir: str = "./results",
    phase: int = 0,
    seen_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    ignore_index: Optional[int] = None,
    include_predicted_unseen: bool = True,
    *,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    yt, yp, seen = _filter_true_labels(
        _as_1d_np(y_true, "y_true"),
        _as_1d_np(y_pred, "y_pred"),
        seen_classes=seen_classes,
        ignore_index=ignore_index,
    )
    labels = _report_labels(seen, yp, include_predicted_unseen)
    names = [_report_class_name(target_names, c, seen) for c in labels]

    metrics = calculate_metrics_torch(
        y_true=yt,
        y_pred=yp,
        num_classes=None,
        old_class_count=old_class_count,
        seen_classes=seen,
        ignore_index=None,
        device="cpu",
        old_classes=old_classes,
        new_classes=new_classes,
        energy=energy,
        energy_margin=energy_margin,
    )
    full_labels = metrics["confusion_matrix_labels"]
    cm_full = metrics["confusion_matrix_full"].detach().cpu().numpy()
    positions = [full_labels.index(int(c)) for c in labels]
    cm_report = cm_full[np.ix_(positions, positions)]

    report_dict = classification_report(
        yt,
        yp,
        labels=labels,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        yt,
        yp,
        labels=labels,
        target_names=names,
        zero_division=0,
        digits=4,
    )
    for cls, name in zip(labels, names):
        report_dict.setdefault(name, {})
        report_dict[name]["precision"] = float(metrics["precision_all"].get(cls, 0.0)) / 100.0
        report_dict[name]["recall"] = float(metrics["recall_all"].get(cls, 0.0)) / 100.0
        report_dict[name]["f1-score"] = float(metrics["f1_per_class_all"].get(cls, 0.0)) / 100.0
        report_dict[name]["support"] = int(metrics["support_all"].get(cls, 0))
    report_dict["torch_metrics"] = make_json_serializable(metrics)
    report_dict["old_new_split"] = {
        k: metrics.get(k)
        for k in (
            "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
            "old_class_ids", "new_class_ids", "old_new_split_available",
            "old_to_new_rate", "new_to_old_rate", "old_to_new_count", "new_to_old_count",
            "old_invalid_prediction_rate", "new_invalid_prediction_rate",
        )
    }

    base = f"phase_{int(phase)}"
    txt_path = os.path.join(save_dir, f"{base}_classification_report.txt")
    json_path = os.path.join(save_dir, f"{base}_classification_report.json")
    cm_csv_path = os.path.join(save_dir, f"{base}_confusion_matrix.csv")
    cm_npy_path = os.path.join(save_dir, f"{base}_confusion_matrix.npy")
    per_class_csv_path = os.path.join(save_dir, f"{base}_per_class_metrics.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - Phase {phase}\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Seen class order: {seen}\n")
        f.write(f"Old class IDs: {metrics.get('old_class_ids', [])}\n")
        f.write(f"New class IDs: {metrics.get('new_class_ids', [])}\n")
        f.write(f"OA: {metrics['overall_accuracy']:.4f}%\n")
        f.write(f"AA: {metrics['average_accuracy']:.4f}%\n")
        f.write(f"Kappa: {metrics['kappa']:.4f}%\n")
        f.write(f"Macro-F1: {metrics['f1_macro']:.4f}%\n")
        f.write(f"Old Accuracy: {metrics.get('old_accuracy', 0.0):.4f}%\n")
        f.write(f"New Accuracy: {metrics.get('new_accuracy', 0.0):.4f}%\n")
        f.write(f"Harmonic Mean: {metrics.get('harmonic_mean', 0.0):.4f}%\n")
        f.write(f"Invalid Prediction Rate: {metrics.get('invalid_prediction_rate', 0.0):.4f}%\n")
        f.write(f"Predicted unseen IDs: {metrics.get('predicted_unseen_classes', [])}\n")
        f.write(f"Old -> New error rate: {metrics.get('old_to_new_rate', 0.0):.4f}%\n")
        f.write(f"New -> Old error rate: {metrics.get('new_to_old_rate', 0.0):.4f}%\n")
        if "geometry_energy_accuracy" in metrics:
            f.write(f"Geometry-energy accuracy: {metrics['geometry_energy_accuracy']:.4f}%\n")
            f.write(f"Geometry margin violation: {metrics['geometry_margin_violation_rate']:.4f}%\n")
            f.write(f"Old -> New energy invasion: {metrics['old_to_new_energy_invasion_rate']:.4f}%\n")
            f.write(f"New -> Old energy invasion: {metrics['new_to_old_energy_invasion_rate']:.4f}%\n")
        f.write("\n" + report_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(report_dict), f, indent=2)

    with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [f"{name} [{cls}]" for cls, name in zip(labels, names)])
        for i, (cls, name) in enumerate(zip(labels, names)):
            writer.writerow([f"{name} [{cls}]"] + [int(v) for v in cm_report[i].tolist()])
    np.save(cm_npy_path, cm_report)

    seen_set = set(seen)
    with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name", "seen_class", "old_class", "new_class",
            "precision_percent", "recall_percent", "f1_percent", "accuracy_percent",
            "support", "predicted_count",
        ])
        old_set = set(metrics.get("old_class_ids", []))
        new_set = set(metrics.get("new_class_ids", []))
        for cls, name in zip(labels, names):
            writer.writerow([
                int(cls), name, bool(cls in seen_set), bool(cls in old_set), bool(cls in new_set),
                float(metrics["precision_all"].get(cls, 0.0)),
                float(metrics["recall_all"].get(cls, 0.0)),
                float(metrics["f1_per_class_all"].get(cls, 0.0)),
                float(metrics["per_class_accuracy_all"].get(cls, 0.0)),
                int(metrics["support_all"].get(cls, 0)),
                int(metrics["predicted_count_all"].get(cls, 0)),
            ])

    print(f"[Report] Saved structured classification report: {txt_path}")
    return {
        "txt_path": txt_path,
        "json_path": json_path,
        "confusion_matrix_csv_path": cm_csv_path,
        "confusion_matrix_npy_path": cm_npy_path,
        "per_class_csv_path": per_class_csv_path,
        "report": report_dict,
        "torch_metrics": metrics,
        "confusion_matrix": cm_report,
        "labels": labels,
        "names": names,
    }


def save_hsi_style_classification_report(
    y_true: Any,
    y_pred: Any,
    target_names: Optional[List[str]] = None,
    save_path: str = "./classification_report.csv",
    tr_time: Optional[float] = None,
    te_time: Optional[float] = None,
    dl_time: Optional[float] = None,
    seen_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    ignore_index: Optional[int] = None,
    *,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    yt, yp, seen = _filter_true_labels(
        _as_1d_np(y_true, "y_true"),
        _as_1d_np(y_pred, "y_pred"),
        seen_classes=seen_classes,
        ignore_index=ignore_index,
    )
    metrics = calculate_metrics_torch(
        yt,
        yp,
        old_class_count=old_class_count,
        seen_classes=seen,
        device="cpu",
        old_classes=old_classes,
        new_classes=new_classes,
    )
    labels = list(seen)
    names = [_safe_class_name(target_names, c) for c in labels]
    cm_seen = metrics["confusion_matrix_seen"].detach().cpu().numpy()
    per_class_acc = [float(metrics["per_class_accuracy"].get(int(c), 0.0)) for c in labels]
    report_text = classification_report(yt, yp, labels=labels, target_names=names, digits=4, zero_division=0)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{0.0 if tr_time is None else float(tr_time)} Tr_Time\n")
        f.write(f"{0.0 if te_time is None else float(te_time)} Te_Time\n")
        f.write(f"{0.0 if dl_time is None else float(dl_time)} DL_Time\n")
        f.write(f"{seen} Seen class order\n")
        f.write(f"{metrics.get('old_class_ids', [])} Old class ids\n")
        f.write(f"{metrics.get('new_class_ids', [])} New class ids\n")
        f.write(f"{metrics['kappa']} Kappa accuracy (%)\n")
        f.write(f"{metrics['overall_accuracy']} Overall accuracy (%)\n")
        f.write(f"{metrics['average_accuracy']} Average accuracy (%)\n")
        f.write(f"{metrics['f1_macro']} Macro F1 (%)\n")
        f.write(f"{metrics.get('old_accuracy', 0.0)} Old accuracy (%)\n")
        f.write(f"{metrics.get('new_accuracy', 0.0)} New accuracy (%)\n")
        f.write(f"{metrics.get('harmonic_mean', 0.0)} Harmonic mean (%)\n")
        f.write(f"{metrics.get('invalid_prediction_rate', 0.0)} Invalid prediction rate (%)\n")
        f.write(f"{metrics.get('old_to_new_rate', 0.0)} Old-to-new error rate (%)\n")
        f.write(f"{metrics.get('new_to_old_rate', 0.0)} New-to-old error rate (%)\n")
        f.write(report_text + "\n")
        f.write(str(np.asarray(per_class_acc)) + "\n")
        f.write(str(cm_seen) + "\n")

    print(f"[Report] Saved HSI-style classification report: {save_path}")
    return {
        "save_path": save_path,
        "overall_accuracy": metrics["overall_accuracy"],
        "average_accuracy": metrics["average_accuracy"],
        "kappa": metrics["kappa"],
        "f1_macro": metrics["f1_macro"],
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm_seen,
        "old_new_split": {
            k: metrics.get(k)
            for k in (
                "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
                "old_class_ids", "new_class_ids", "old_new_split_available",
            )
        },
        "torch_metrics": metrics,
    }


def save_classification_report(
    y_true: Any,
    y_pred: Any,
    target_names: Optional[List[str]] = None,
    save_dir: str = "./results",
    phase: int = 0,
    seen_classes: Optional[Iterable[int]] = None,
    old_class_count: Optional[int] = None,
    ignore_index: Optional[int] = None,
    include_predicted_unseen: bool = True,
    save_hsi_style: bool = True,
    save_structured: bool = True,
    tr_time: Optional[float] = None,
    te_time: Optional[float] = None,
    dl_time: Optional[float] = None,
    save_path: Optional[str] = None,
    *,
    old_classes: Optional[Iterable[int]] = None,
    new_classes: Optional[Iterable[int]] = None,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Dict[str, Any]:
    if save_path is not None:
        return save_hsi_style_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_path=save_path,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
            old_classes=old_classes,
            new_classes=new_classes,
        )

    os.makedirs(save_dir, exist_ok=True)
    output: Dict[str, Any] = {}
    if save_structured:
        output["structured"] = save_structured_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_dir=save_dir,
            phase=phase,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
            include_predicted_unseen=include_predicted_unseen,
            old_classes=old_classes,
            new_classes=new_classes,
            energy=energy,
            energy_margin=energy_margin,
        )
    if save_hsi_style:
        hsi_path = os.path.join(save_dir, f"phase_{phase}_HSI_Classification_Report.csv")
        output["hsi_style"] = save_hsi_style_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_path=hsi_path,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
            old_classes=old_classes,
            new_classes=new_classes,
        )
    if "structured" in output:
        for key in (
            "txt_path", "json_path", "confusion_matrix_csv_path",
            "confusion_matrix_npy_path", "per_class_csv_path",
        ):
            output[key] = output["structured"].get(key)
    if "hsi_style" in output:
        output["hsi_style_path"] = output["hsi_style"].get("save_path")
    return output


# ============================================================
# NECIL evaluator
# ============================================================
class NECILEvaluator:
    def __init__(self) -> None:
        self.phase_history: Dict[int, Dict[str, Any]] = {}
        self.class_acc_history: defaultdict[int, List[float]] = defaultdict(list)
        self.class_presence_history: defaultdict[int, List[bool]] = defaultdict(list)
        self.class_introduction_phase: Dict[int, int] = {}
        self.phases_seen: List[int] = []

    def _sanity_check_labels(self, y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
        yt = _as_1d_np(y_true, "y_true")
        yp = _as_1d_np(y_pred, "y_pred")
        if yt.shape[0] != yp.shape[0]:
            raise ValueError(f"y_true/y_pred length mismatch: {len(yt)} vs {len(yp)}")
        return yt, yp

    def update(
        self,
        phase: int,
        y_true: Any,
        y_pred: Any,
        old_class_count: Optional[int] = None,
        seen_classes: Optional[Iterable[int]] = None,
        ignore_index: Optional[int] = None,
        *,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        energy: Optional[torch.Tensor] = None,
        energy_margin: float = 0.0,
    ) -> None:
        phase = int(phase)
        yt, yp = self._sanity_check_labels(y_true, y_pred)
        seen = (
            _ordered_unique_ints(seen_classes)
            if seen_classes is not None
            else sorted(np.unique(yt).astype(int).tolist())
        )
        metrics = calculate_metrics_torch(
            y_true=yt,
            y_pred=yp,
            old_class_count=old_class_count,
            seen_classes=seen,
            ignore_index=ignore_index,
            device="cpu",
            old_classes=old_classes,
            new_classes=new_classes,
            energy=energy,
            energy_margin=energy_margin,
        )
        self.phase_history[phase] = metrics
        if phase not in self.phases_seen:
            self.phases_seen.append(phase)
            self.phases_seen.sort()
        for cls in metrics.get("new_class_ids", []):
            self.class_introduction_phase.setdefault(int(cls), phase)
        if phase == min(self.phases_seen):
            for cls in metrics.get("classes", []):
                self.class_introduction_phase.setdefault(int(cls), phase)
        self._rebuild_class_history()

    def save_phase_report(
        self,
        phase: int,
        y_true: Any,
        y_pred: Any,
        target_names: Optional[List[str]] = None,
        save_dir: str = "./results",
        seen_classes: Optional[Iterable[int]] = None,
        old_class_count: Optional[int] = None,
        ignore_index: Optional[int] = None,
        tr_time: Optional[float] = None,
        te_time: Optional[float] = None,
        dl_time: Optional[float] = None,
        *,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        energy: Optional[torch.Tensor] = None,
        energy_margin: float = 0.0,
    ) -> Dict[str, Any]:
        return save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_dir=save_dir,
            phase=phase,
            seen_classes=seen_classes,
            old_class_count=old_class_count,
            ignore_index=ignore_index,
            tr_time=tr_time,
            te_time=te_time,
            dl_time=dl_time,
            save_hsi_style=True,
            save_structured=True,
            old_classes=old_classes,
            new_classes=new_classes,
            energy=energy,
            energy_margin=energy_margin,
        )

    def _rebuild_class_history(self) -> None:
        # A class is considered present only when the evaluated split contains
        # at least one true sample. Merely being listed in seen_classes is not
        # enough; otherwise a zero-support validation split creates fake 0%
        # accuracy and artificial forgetting.
        all_classes = sorted({
            int(c)
            for phase in self.phases_seen
            for c, support in self.phase_history[phase].get("support", {}).items()
            if int(support) > 0
        })
        introductions: Dict[int, int] = {}
        for cls in all_classes:
            phases_with_support = [
                phase
                for phase in self.phases_seen
                if int(self.phase_history[phase].get("support", {}).get(cls, 0)) > 0
            ]
            if phases_with_support:
                introductions[int(cls)] = int(min(phases_with_support))
        self.class_introduction_phase = introductions

        new_hist: defaultdict[int, List[float]] = defaultdict(list)
        presence: defaultdict[int, List[bool]] = defaultdict(list)
        for cls in all_classes:
            for phase in self.phases_seen:
                per_class = self.phase_history[phase].get("per_class_accuracy", {})
                support = int(self.phase_history[phase].get("support", {}).get(cls, 0))
                present = support > 0 and cls in per_class
                new_hist[cls].append(float(per_class[cls]) if present else float("nan"))
                presence[cls].append(bool(present))
        self.class_acc_history = new_hist
        self.class_presence_history = presence

    def calculate_forgetting_per_class(self) -> Dict[int, float]:
        if len(self.phases_seen) < 2:
            return {}
        forgetting: Dict[int, float] = {}
        for cls, history in self.class_acc_history.items():
            vals = np.asarray(history, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size >= 2:
                forgetting[int(cls)] = float(max(0.0, float(np.max(vals[:-1])) - float(vals[-1])))
        return forgetting

    def calculate_backward_transfer(self) -> float:
        """Final accuracy minus accuracy at each class's introduction phase."""
        values = []
        for history in self.class_acc_history.values():
            vals = np.asarray(history, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size >= 2:
                values.append(float(vals[-1] - vals[0]))
        return float(np.mean(values)) if values else 0.0

    def get_standard_metrics(self) -> Dict[str, float]:
        if not self.phases_seen:
            return {}
        last_phase = self.phases_seen[-1]
        all_oa = [float(self.phase_history[p].get("overall_accuracy", 0.0)) for p in self.phases_seen]
        inc_h = [
            float(self.phase_history[p].get("harmonic_mean", 0.0))
            for p in self.phases_seen
            if bool(self.phase_history[p].get("old_new_split_available", False))
        ]
        forgetting = self.calculate_forgetting_per_class()
        last = self.phase_history[last_phase]
        return {
            "A_last (Final Accuracy)": float(last.get("overall_accuracy", 0.0)),
            "A_avg (Avg Accuracy)": float(np.mean(all_oa)) if all_oa else 0.0,
            "H_last (Final Harmonic Mean)": float(last.get("harmonic_mean", 0.0)),
            "H_avg (Avg Harmonic Mean)": float(np.mean(inc_h)) if inc_h else 0.0,
            "H_avg_inc_only": float(np.mean(inc_h)) if inc_h else 0.0,
            "F_avg (Avg Forgetting)": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
            "BWT (Backward Transfer)": self.calculate_backward_transfer(),
            "Old_last (Final Old Accuracy)": float(last.get("old_accuracy", 0.0)),
            "New_last (Final New Accuracy)": float(last.get("new_accuracy", 0.0)),
            "AA_last (Final Avg Accuracy)": float(last.get("average_accuracy", 0.0)),
            "Kappa_last": float(last.get("kappa", 0.0)),
            "F1_last": float(last.get("f1_macro", 0.0)),
            "Invalid_last": float(last.get("invalid_prediction_rate", 0.0)),
            "OldToNew_last": float(last.get("old_to_new_rate", 0.0)),
            "NewToOld_last": float(last.get("new_to_old_rate", 0.0)),
            "GeometryViolation_last": float(last.get("geometry_margin_violation_rate", 0.0)),
            "Phases": len(self.phases_seen),
        }

    def get_phase_table(self) -> List[Dict[str, Any]]:
        rows = []
        for phase in self.phases_seen:
            m = self.phase_history[phase]
            rows.append({
                "phase": int(phase),
                "OA": float(m.get("overall_accuracy", 0.0)),
                "AA": float(m.get("average_accuracy", 0.0)),
                "Kappa": float(m.get("kappa", 0.0)),
                "F1": float(m.get("f1_macro", 0.0)),
                "Old": float(m.get("old_accuracy", 0.0)),
                "New": float(m.get("new_accuracy", 0.0)),
                "H": float(m.get("harmonic_mean", 0.0)),
                "Invalid": float(m.get("invalid_prediction_rate", 0.0)),
                "OldToNew": float(m.get("old_to_new_rate", 0.0)),
                "NewToOld": float(m.get("new_to_old_rate", 0.0)),
                "EnergyViolation": float(m.get("geometry_margin_violation_rate", 0.0)),
                "OldEnergyInvasion": float(m.get("old_to_new_energy_invasion_rate", 0.0)),
                "NewEnergyInvasion": float(m.get("new_to_old_energy_invasion_rate", 0.0)),
                "SplitAvailable": bool(m.get("old_new_split_available", False)),
                "Samples": int(m.get("num_samples", 0)),
            })
        return rows

    def get_per_class_summary(self) -> Dict[int, Dict[str, float]]:
        forgetting = self.calculate_forgetting_per_class()
        out: Dict[int, Dict[str, float]] = {}
        for cls, history in self.class_acc_history.items():
            vals = np.asarray(history, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size:
                out[int(cls)] = {
                    "introduction_phase": float(self.class_introduction_phase.get(int(cls), 0)),
                    "first": float(vals[0]),
                    "best": float(np.max(vals)),
                    "last": float(vals[-1]),
                    "forgetting": float(forgetting.get(int(cls), 0.0)),
                    "backward_transfer": float(vals[-1] - vals[0]),
                }
        return out

    def print_summary(self) -> None:
        if not self.phases_seen:
            print("[NECILEvaluator] No phases evaluated yet.")
            return
        metrics = self.get_standard_metrics()
        last_phase = self.phases_seen[-1]
        m = self.phase_history[last_phase]
        print("\n" + "=" * 64)
        print(f" NECIL-HSI Evaluation Report (Phase {last_phase})")
        print("=" * 64)
        print(f" 1. Final Accuracy (A_last):       {metrics.get('A_last (Final Accuracy)', 0):.2f}%")
        print(f" 2. Avg Accuracy (A_avg):          {metrics.get('A_avg (Avg Accuracy)', 0):.2f}%")
        print(f" 3. Avg Forgetting (F_avg):        {metrics.get('F_avg (Avg Forgetting)', 0):.2f}%")
        print(f" 4. Backward Transfer (BWT):       {metrics.get('BWT (Backward Transfer)', 0):.2f}%")
        print(f" 5. Old / New Accuracy:            {m.get('old_accuracy', 0):.2f}% / {m.get('new_accuracy', 0):.2f}%")
        print(f" 6. Harmonic Mean:                 {m.get('harmonic_mean', 0):.2f}%")
        print(f" 7. Invalid Prediction Rate:       {m.get('invalid_prediction_rate', 0):.2f}%")
        print(f" 8. Old -> New / New -> Old:       {m.get('old_to_new_rate', 0):.2f}% / {m.get('new_to_old_rate', 0):.2f}%")
        print(f" 9. Geometry Margin Violation:     {m.get('geometry_margin_violation_rate', 0):.2f}%")
        print(f"10. Energy Old->New / New->Old:    {m.get('old_to_new_energy_invasion_rate', 0):.2f}% / {m.get('new_to_old_energy_invasion_rate', 0):.2f}%")
        print(f"11. AA / Kappa / F1:               {m.get('average_accuracy', 0):.2f}% / {m.get('kappa', 0):.2f}% / {m.get('f1_macro', 0):.2f}%")
        print("-" * 64)

    def save_phase_table_csv(self, save_path: str) -> str:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        rows = self.get_phase_table()
        fields = [
            "phase", "OA", "AA", "Kappa", "F1", "Old", "New", "H", "Invalid",
            "OldToNew", "NewToOld", "EnergyViolation", "OldEnergyInvasion",
            "NewEnergyInvasion", "SplitAvailable", "Samples",
        ]
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return save_path

    def to_dict(self) -> Dict[str, Any]:
        return make_json_serializable({
            "phase_history": self.phase_history,
            "class_introduction_phase": self.class_introduction_phase,
            "standard_metrics": self.get_standard_metrics(),
            "phase_table": self.get_phase_table(),
            "per_class_summary": self.get_per_class_summary(),
        })










# """
# Clean evaluation utilities for PG-RGA geometry-native NECIL-HSI.

# Contract
# --------
# - Labels are sequential HSI class ids: 0..C-1 after background removal.
# - `seen_classes` defines the only valid prediction columns for a phase.
# - Future/unseen predictions are NOT silently dropped; they are counted as wrong
#   and reported as leakage.
# - Old/new split follows the provided phase class order when available; otherwise it falls back to global sequential id < old_class_count.

# Outputs
# -------
# - phase_X_classification_report.txt
# - phase_X_classification_report.json
# - phase_X_confusion_matrix.csv
# - phase_X_confusion_matrix.npy
# - phase_X_per_class_metrics.csv
# - phase_X_HSI_Classification_Report.csv
# """

# from __future__ import annotations

# import csv
# import json
# import os
# from collections import defaultdict
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from sklearn.metrics import classification_report


# # ============================================================
# # Generic conversion / serialization
# # ============================================================
# def make_json_serializable(obj: Any) -> Any:
#     if isinstance(obj, dict):
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
#     arr = np.asarray(x).reshape(-1)
#     if arr.size == 0:
#         raise ValueError(f"{name} is empty.")
#     return arr.astype(np.int64, copy=False)


# def _to_1d_long_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
#     if torch.is_tensor(x):
#         t = x.detach()
#     else:
#         t = torch.as_tensor(x)
#     t = t.long().view(-1)
#     return t.to(device) if device is not None else t


# def _safe_class_name(target_names: Optional[Sequence[str]], cls: int) -> str:
#     cls = int(cls)
#     if target_names is not None and 0 <= cls < len(target_names):
#         return str(target_names[cls])
#     return f"Class {cls}"


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     """Preserve phase/class-order while removing duplicates."""
#     out: List[int] = []
#     seen = set()
#     for value in values:
#         c = int(value)
#         if c not in seen:
#             out.append(c)
#             seen.add(c)
#     return out


# def _seen_list(seen_classes: Optional[Iterable[int]], y_true: Optional[np.ndarray] = None) -> Optional[List[int]]:
#     if seen_classes is None:
#         return None
#     out = _ordered_unique_ints(seen_classes)
#     if not out:
#         raise ValueError("seen_classes was provided but empty.")
#     if y_true is not None:
#         true_set = set(int(c) for c in np.unique(y_true).tolist())
#         bad_true = sorted(true_set.difference(set(out)))
#         if bad_true:
#             raise ValueError(
#                 f"y_true contains labels outside seen_classes: {bad_true}. "
#                 "This is usually a phase/dataset split bug, not an evaluation issue."
#             )
#     return out


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
#     """Report predictions outside the currently seen phase classes.

#     In NECIL evaluation, future/unseen columns are not ignored. They are counted
#     as wrong and explicitly reported. Negative predictions are also treated as
#     invalid predictions; this prevents a rare but dangerous failure where a
#     negative prediction would be dropped by a confusion-matrix filter and inflate
#     OA.
#     """
#     seen_set = set(int(c) for c in seen)
#     invalid = np.asarray([int(v) not in seen_set for v in y_pred], dtype=bool)
#     return {
#         "invalid_prediction_rate": float(invalid.mean() * 100.0) if invalid.size else 0.0,
#         "predicted_unseen_count": int(invalid.sum()),
#         "predicted_unseen_classes": sorted(int(c) for c in np.unique(y_pred[invalid]).tolist()) if invalid.any() else [],
#     }


# def _add_metric_aliases(metrics: Dict[str, Any]) -> Dict[str, Any]:
#     """Add stable aliases consumed by older main.py/reporting code.

#     The project has used both evaluator names (overall_accuracy/harmonic_mean)
#     and trainer names (acc/hm/old_acc/new_acc).  Keeping aliases here prevents
#     phase summaries from silently showing zeros when one side changes key names.
#     """
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


# def _remap_negative_predictions(y_pred: np.ndarray, base_num_classes: int) -> Tuple[np.ndarray, int, Optional[int]]:
#     """Map negative prediction ids to an overflow column so they count as wrong."""
#     yp = np.asarray(y_pred).reshape(-1).astype(np.int64, copy=True)
#     if yp.size == 0 or not np.any(yp < 0):
#         return yp, int(base_num_classes), None
#     overflow = int(max(base_num_classes, int(yp.max()) + 1, 0))
#     yp[yp < 0] = overflow
#     return yp, overflow + 1, overflow

# def _prepare_report_predictions(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     seen: Sequence[int],
# ) -> Tuple[np.ndarray, int, Optional[int]]:
#     """Return predictions safe for sklearn/report indexing.

#     Metric code can keep raw negative predictions and remap them internally, but
#     report code cannot use a negative class id as a NumPy index.  This helper
#     maps negative predictions to an explicit overflow class id before building
#     report labels/confusion matrices.
#     """
#     yt = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
#     yp = np.asarray(y_pred).reshape(-1).astype(np.int64, copy=False)
#     max_seen = max([int(c) for c in seen]) if seen else -1
#     max_true = int(yt.max()) if yt.size else -1
#     max_pred_pos = int(yp[yp >= 0].max()) if np.any(yp >= 0) else -1
#     base_c = max(max_seen + 1, max_true + 1, max_pred_pos + 1, 1)
#     yp_report, report_c, neg_overflow = _remap_negative_predictions(yp, base_c)
#     return yp_report, int(report_c), neg_overflow


# def _report_class_name(
#     target_names: Optional[Sequence[str]],
#     cls: int,
#     seen: Sequence[int],
#     negative_overflow_class: Optional[int],
# ) -> str:
#     cls = int(cls)
#     if cls in set(int(c) for c in seen):
#         return _safe_class_name(target_names, cls)
#     if negative_overflow_class is not None and cls == int(negative_overflow_class):
#         return "INVALID-NEGATIVE-PRED"
#     return f"UNSEEN-PRED-{cls}"



# # ============================================================
# # Torch-native metrics
# # ============================================================
# @torch.no_grad()
# def torch_confusion_matrix(y_true: Any, y_pred: Any, num_classes: int, device: Optional[str] = "cpu") -> torch.Tensor:
#     """Rows=true labels, columns=predicted labels."""
#     if int(num_classes) <= 0:
#         raise ValueError("num_classes must be positive.")
#     dev = torch.device(device) if device is not None else None
#     yt = _to_1d_long_tensor(y_true, dev)
#     yp = _to_1d_long_tensor(y_pred, dev)
#     if yt.numel() != yp.numel():
#         raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

#     valid = (yt >= 0) & (yt < int(num_classes)) & (yp >= 0) & (yp < int(num_classes))
#     yt = yt[valid]
#     yp = yp[valid]
#     idx = yt * int(num_classes) + yp
#     return torch.bincount(idx, minlength=int(num_classes) * int(num_classes)).reshape(int(num_classes), int(num_classes))


# @torch.no_grad()
# def torch_metrics_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-12) -> Dict[str, Any]:
#     cm = cm.detach().float()
#     if cm.dim() != 2 or cm.size(0) != cm.size(1):
#         raise ValueError(f"confusion matrix must be square [C,C], got {tuple(cm.shape)}")

#     tp = torch.diag(cm)
#     support = cm.sum(dim=1)
#     predicted = cm.sum(dim=0)
#     total = cm.sum()

#     recall = tp / (support + eps)
#     precision = tp / (predicted + eps)
#     f1 = 2.0 * precision * recall / (precision + recall + eps)
#     valid = support > 0

#     oa = 100.0 * tp.sum() / (total + eps)
#     aa = 100.0 * recall[valid].mean() if bool(valid.any().item()) else torch.tensor(0.0, device=cm.device)
#     macro_f1 = 100.0 * f1[valid].mean() if bool(valid.any().item()) else torch.tensor(0.0, device=cm.device)

#     po = tp.sum() / (total + eps)
#     pe = (cm.sum(dim=1) * cm.sum(dim=0)).sum() / (total * total + eps)
#     kappa = 100.0 * (po - pe) / (1.0 - pe + eps)

#     return {
#         "overall_accuracy": float(oa.item()),
#         "balanced_accuracy": float(aa.item()),
#         "average_accuracy": float(aa.item()),
#         "kappa": float(kappa.item()),
#         "f1_macro": float(macro_f1.item()),
#         "per_class_accuracy": {int(i): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
#         "precision": {int(i): float((100.0 * precision[i]).item()) for i in range(cm.size(0))},
#         "recall": {int(i): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
#         "f1_per_class": {int(i): float((100.0 * f1[i]).item()) for i in range(cm.size(0))},
#         "support": {int(i): int(support[i].item()) for i in range(cm.size(0))},
#         "predicted_count": {int(i): int(predicted[i].item()) for i in range(cm.size(0))},
#         "confusion_matrix": cm.detach().cpu(),
#     }


# @torch.no_grad()
# def torch_old_new_metrics(
#     y_true: Any,
#     y_pred: Any,
#     old_class_count: Optional[int] = None,
#     *,
#     seen_classes: Optional[Iterable[int]] = None,
#     eps: float = 1e-12,
# ) -> Dict[str, Any]:
#     """Compute old/new/HM and explicit old<->new stealing rates.

#     PG-RGA evaluation uses global class ids for y_true/y_pred and an ordered
#     seen-class list for the phase.  The first ``old_class_count`` entries in
#     ``seen_classes`` are old classes; the remaining entries are current new
#     classes.  Predictions outside the seen set are *not* assigned to old or new;
#     they are reported separately as invalid/unseen leakage.
#     """
#     yt = _to_1d_long_tensor(y_true)
#     yp = _to_1d_long_tensor(y_pred, yt.device)
#     if yt.numel() != yp.numel():
#         raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

#     overall = 100.0 * (yp == yt).float().mean() if yt.numel() else torch.tensor(0.0, device=yt.device)
#     old_class_count_i = 0 if old_class_count is None else int(old_class_count)

#     if seen_classes is not None:
#         seen_order = _ordered_unique_ints(seen_classes)
#     else:
#         seen_order = sorted(int(c) for c in torch.unique(yt).cpu().tolist())

#     if old_class_count_i < 0:
#         raise ValueError(f"old_class_count must be >= 0, got {old_class_count_i}")
#     if old_class_count_i > len(seen_order):
#         raise ValueError(
#             f"old_class_count={old_class_count_i} exceeds len(seen_classes)={len(seen_order)}."
#         )

#     # Phase 0: there is no old/new split. Keep H=OA for readable phase tables,
#     # but do not fake old accuracy as a real old-class measurement.
#     if old_class_count_i <= 0:
#         return {
#             "old_accuracy": 0.0,
#             "new_accuracy": float(overall.item()),
#             "harmonic_mean": float(overall.item()),
#             "old_count": 0,
#             "new_count": int(yt.numel()),
#             "old_class_ids": [],
#             "new_class_ids": [int(c) for c in seen_order],
#             "old_new_split_available": False,
#             "old_to_new_rate": 0.0,
#             "new_to_old_rate": 0.0,
#             "old_to_new_count": 0,
#             "new_to_old_count": 0,
#         }

#     old_ids = seen_order[:old_class_count_i]
#     new_ids = seen_order[old_class_count_i:]
#     old_t = torch.tensor(old_ids, device=yt.device, dtype=torch.long) if old_ids else torch.empty(0, device=yt.device, dtype=torch.long)
#     new_t = torch.tensor(new_ids, device=yt.device, dtype=torch.long) if new_ids else torch.empty(0, device=yt.device, dtype=torch.long)

#     old_mask = (yt.view(-1, 1) == old_t.view(1, -1)).any(dim=1) if old_t.numel() else torch.zeros_like(yt, dtype=torch.bool)
#     new_mask = (yt.view(-1, 1) == new_t.view(1, -1)).any(dim=1) if new_t.numel() else torch.zeros_like(yt, dtype=torch.bool)
#     pred_old = (yp.view(-1, 1) == old_t.view(1, -1)).any(dim=1) if old_t.numel() else torch.zeros_like(yp, dtype=torch.bool)
#     pred_new = (yp.view(-1, 1) == new_t.view(1, -1)).any(dim=1) if new_t.numel() else torch.zeros_like(yp, dtype=torch.bool)

#     old_total = int(old_mask.sum().item())
#     new_total = int(new_mask.sum().item())
#     old_acc = 100.0 * (yp[old_mask] == yt[old_mask]).float().mean() if old_total else torch.tensor(0.0, device=yt.device)
#     new_acc = 100.0 * (yp[new_mask] == yt[new_mask]).float().mean() if new_total else torch.tensor(0.0, device=yt.device)

#     old_to_new_count = int((old_mask & pred_new).sum().item())
#     new_to_old_count = int((new_mask & pred_old).sum().item())
#     old_to_new_rate = 100.0 * old_to_new_count / max(old_total, 1)
#     new_to_old_rate = 100.0 * new_to_old_count / max(new_total, 1)

#     if old_total > 0 and new_total > 0:
#         h = 2.0 * old_acc * new_acc / (old_acc + new_acc + eps)
#         split_available = True
#     else:
#         h = overall
#         split_available = False

#     return {
#         "old_accuracy": float(old_acc.item()),
#         "new_accuracy": float(new_acc.item()),
#         "harmonic_mean": float(h.item()),
#         "old_count": old_total,
#         "new_count": new_total,
#         "old_class_ids": [int(c) for c in old_ids],
#         "new_class_ids": [int(c) for c in new_ids],
#         "old_new_split_available": bool(split_available),
#         "old_to_new_rate": float(old_to_new_rate),
#         "new_to_old_rate": float(new_to_old_rate),
#         "old_to_new_count": int(old_to_new_count),
#         "new_to_old_count": int(new_to_old_count),
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
# ) -> Dict[str, Any]:
#     yt_np, yp_np_raw, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )

#     # Count unseen predictions as wrong by keeping y_pred in the full CM width.
#     # Positive future-class ids expand the confusion matrix. Negative predictions
#     # are mapped to a dedicated overflow column instead of being dropped.
#     max_seen = max(seen) if seen else -1
#     max_true = int(yt_np.max()) if yt_np.size else -1
#     max_pred_pos = int(yp_np_raw[yp_np_raw >= 0].max()) if np.any(yp_np_raw >= 0) else -1
#     C_base = max(int(num_classes or 0), max_seen + 1, max_true + 1, max_pred_pos + 1)
#     if C_base <= 0:
#         raise ValueError("Cannot infer num_classes.")
#     yp_np_cm, C, negative_overflow_class = _remap_negative_predictions(yp_np_raw, C_base)

#     cm_full = torch_confusion_matrix(yt_np, yp_np_cm, C, device=device)
#     metrics = torch_metrics_from_confusion_matrix(cm_full)

#     seen_set = set(seen)
#     for key in ("per_class_accuracy", "precision", "recall", "f1_per_class", "support", "predicted_count"):
#         metrics[key] = {int(k): v for k, v in metrics[key].items() if int(k) in seen_set}

#     split = torch_old_new_metrics(yt_np, yp_np_raw, old_class_count=old_class_count, seen_classes=seen)
#     metrics.update(split)
#     metrics.update(_prediction_leakage_np(yp_np_raw, seen))
#     metrics["negative_prediction_count"] = int((yp_np_raw < 0).sum())
#     metrics["negative_prediction_overflow_class"] = None if negative_overflow_class is None else int(negative_overflow_class)
#     metrics["num_samples"] = int(yt_np.size)
#     metrics["num_classes"] = int(len(seen))
#     metrics["classes"] = [int(c) for c in seen]
#     seen_idx = torch.as_tensor([int(c) for c in seen], dtype=torch.long)
#     metrics["confusion_matrix_full"] = cm_full.detach().cpu()
#     if seen_idx.numel() > 0:
#         metrics["confusion_matrix_seen"] = cm_full.detach().cpu().index_select(0, seen_idx).index_select(1, seen_idx)
#     else:
#         metrics["confusion_matrix_seen"] = torch.empty((0, 0), dtype=cm_full.dtype)
#     # Keep legacy key as full matrix so predicted-unseen leakage is not hidden.
#     metrics["confusion_matrix"] = cm_full.detach().cpu()
#     return _add_metric_aliases(metrics)


# def calculate_metrics(
#     y_true: Any,
#     y_pred: Any,
#     class_names: Optional[List[str]] = None,
#     old_class_count: Optional[int] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
# ) -> Dict[str, Any]:
#     del class_names
#     seen = list(seen_classes) if seen_classes is not None else None
#     if seen:
#         num_classes = max(seen) + 1
#     else:
#         yt = np.asarray(y_true).reshape(-1)
#         yp = np.asarray(y_pred).reshape(-1)
#         num_classes = int(max(yt.max(), yp.max())) + 1 if yt.size and yp.size else None
#     return calculate_metrics_torch(
#         y_true=y_true,
#         y_pred=y_pred,
#         num_classes=num_classes,
#         old_class_count=old_class_count,
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#         device="cpu",
#     )


# # ============================================================
# # Report writing
# # ============================================================
# def _report_labels(seen: Sequence[int], y_pred: np.ndarray, include_predicted_unseen: bool) -> List[int]:
#     labels = list(int(c) for c in seen)
#     if include_predicted_unseen:
#         label_set = set(labels)
#         labels += sorted(int(c) for c in np.unique(y_pred).tolist() if int(c) not in label_set)
#     return labels


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
# ) -> Dict[str, Any]:
#     os.makedirs(save_dir, exist_ok=True)

#     yt, yp_raw, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )
#     yp_report, C, negative_overflow_class = _prepare_report_predictions(yt, yp_raw, seen)
#     labels = _report_labels(seen, yp_report, include_predicted_unseen=include_predicted_unseen)
#     names = [_report_class_name(target_names, c, seen, negative_overflow_class) for c in labels]

#     metrics = calculate_metrics_torch(
#         y_true=yt,
#         y_pred=yp_raw,
#         num_classes=C,
#         old_class_count=old_class_count,
#         seen_classes=seen,
#         ignore_index=None,
#         device="cpu",
#     )
#     cm_full = metrics["confusion_matrix_full"].detach().cpu().numpy()
#     if labels and max(labels) >= cm_full.shape[0]:
#         raise RuntimeError(
#             f"Report labels exceed confusion-matrix width: max_label={max(labels)}, cm_shape={cm_full.shape}"
#         )
#     cm_report = cm_full[np.ix_(labels, labels)]

#     # sklearn report is for display only; scalar truth comes from torch metrics above.
#     report_dict = classification_report(
#         yt,
#         yp_report,
#         labels=labels,
#         target_names=names,
#         zero_division=0,
#         output_dict=True,
#     )
#     report_text = classification_report(
#         yt,
#         yp_report,
#         labels=labels,
#         target_names=names,
#         zero_division=0,
#         digits=4,
#     )

#     for cls, name in zip(labels, names):
#         report_dict.setdefault(name, {})
#         report_dict[name]["precision"] = float(metrics.get("precision", {}).get(cls, 0.0)) / 100.0
#         report_dict[name]["recall"] = float(metrics.get("recall", {}).get(cls, 0.0)) / 100.0
#         report_dict[name]["f1-score"] = float(metrics.get("f1_per_class", {}).get(cls, 0.0)) / 100.0
#         report_dict[name]["support"] = int(metrics.get("support", {}).get(cls, 0))
#     report_dict["torch_metrics"] = make_json_serializable(metrics)
#     report_dict["old_new_split"] = {
#         k: metrics.get(k)
#         for k in [
#             "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
#             "old_class_ids", "new_class_ids", "old_new_split_available",
#             "old_to_new_rate", "new_to_old_rate", "old_to_new_count", "new_to_old_count",
#         ]
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
#         f.write(f"OA: {metrics['overall_accuracy']:.4f}%\n")
#         f.write(f"AA: {metrics['average_accuracy']:.4f}%\n")
#         f.write(f"Kappa: {metrics['kappa']:.4f}%\n")
#         f.write(f"Macro-F1: {metrics['f1_macro']:.4f}%\n")
#         f.write(f"Old Accuracy: {metrics.get('old_accuracy', 0.0):.4f}%\n")
#         f.write(f"New Accuracy: {metrics.get('new_accuracy', 0.0):.4f}%\n")
#         f.write(f"Harmonic Mean: {metrics.get('harmonic_mean', 0.0):.4f}%\n")
#         f.write(f"Old/New Split Available: {bool(metrics.get('old_new_split_available', False))}\n")
#         f.write(f"Old Class IDs: {metrics.get('old_class_ids', [])}\n")
#         f.write(f"New Class IDs: {metrics.get('new_class_ids', [])}\n")
#         f.write(f"Invalid Prediction Rate: {metrics.get('invalid_prediction_rate', 0.0):.4f}%\n")
#         f.write(f"Predicted unseen classes: {metrics.get('predicted_unseen_classes', [])}\n")
#         f.write(f"Predicted unseen count: {metrics.get('predicted_unseen_count', 0)}\n")
#         f.write(f"Old -> New error rate: {metrics.get('old_to_new_rate', 0.0):.4f}% ({metrics.get('old_to_new_count', 0)} samples)\n")
#         f.write(f"New -> Old error rate: {metrics.get('new_to_old_rate', 0.0):.4f}% ({metrics.get('new_to_old_count', 0)} samples)\n\n")
#         f.write(report_text)

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(report_dict), f, indent=2)

#     with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["true\\pred"] + names)
#         for i, name in enumerate(names):
#             writer.writerow([name] + [int(v) for v in cm_report[i].tolist()])
#     np.save(cm_npy_path, cm_report)

#     with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "class_id", "class_name", "seen_class", "precision_percent", "recall_percent",
#             "f1_percent", "accuracy_percent", "support", "predicted_count",
#         ])
#         for cls, name in zip(labels, names):
#             writer.writerow([
#                 int(cls),
#                 name,
#                 bool(int(cls) in set(seen)),
#                 float(metrics.get("precision", {}).get(cls, 0.0)),
#                 float(metrics.get("recall", {}).get(cls, 0.0)),
#                 float(metrics.get("f1_per_class", {}).get(cls, 0.0)),
#                 float(metrics.get("per_class_accuracy", {}).get(cls, 0.0)),
#                 int(metrics.get("support", {}).get(cls, 0)),
#                 int(metrics.get("predicted_count", {}).get(cls, 0)),
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
# ) -> Dict[str, Any]:
#     parent = os.path.dirname(save_path)
#     if parent:
#         os.makedirs(parent, exist_ok=True)

#     yt, yp_raw, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )
#     yp_report, C, _ = _prepare_report_predictions(yt, yp_raw, seen)
#     metrics = calculate_metrics_torch(yt, yp_raw, C, old_class_count, seen, None, device="cpu")
#     labels = list(seen)
#     names = [_safe_class_name(target_names, c) for c in labels]
#     cm_report = metrics["confusion_matrix_seen"].detach().cpu().numpy()
#     per_class_acc = [float(metrics.get("per_class_accuracy", {}).get(int(c), 0.0)) for c in labels]

#     report_text = classification_report(yt, yp_report, labels=labels, target_names=names, digits=4, zero_division=0)
#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write(f"{0.0 if tr_time is None else float(tr_time)} Tr_Time\n")
#         f.write(f"{0.0 if te_time is None else float(te_time)} Te_Time\n")
#         f.write(f"{0.0 if dl_time is None else float(dl_time)} DL_Time\n")
#         f.write(f"{metrics['kappa']} Kappa accuracy (%)\n")
#         f.write(f"{metrics['overall_accuracy']} Overall accuracy (%)\n")
#         f.write(f"{metrics['average_accuracy']} Average accuracy (%)\n")
#         f.write(f"{metrics['f1_macro']} Macro F1 (%)\n")
#         f.write(f"{metrics.get('old_accuracy', 0.0)} Old accuracy (%)\n")
#         f.write(f"{metrics.get('new_accuracy', 0.0)} New accuracy (%)\n")
#         f.write(f"{metrics.get('harmonic_mean', 0.0)} Harmonic mean (%)\n")
#         f.write(f"{bool(metrics.get('old_new_split_available', False))} Old/new split available\n")
#         f.write(f"{metrics.get('old_class_ids', [])} Old class ids\n")
#         f.write(f"{metrics.get('new_class_ids', [])} New class ids\n")
#         f.write(f"{metrics.get('invalid_prediction_rate', 0.0)} Invalid prediction rate (%)\n")
#         f.write(f"{metrics.get('old_to_new_rate', 0.0)} Old-to-new error rate (%)\n")
#         f.write(f"{metrics.get('new_to_old_rate', 0.0)} New-to-old error rate (%)\n")
#         f.write(report_text + "\n")
#         f.write(str(np.asarray(per_class_acc)) + "\n")
#         f.write(str(cm_report) + "\n")

#     print(f"[Report] Saved HSI-style classification report: {save_path}")
#     return {
#         "save_path": save_path,
#         "overall_accuracy": metrics["overall_accuracy"],
#         "average_accuracy": metrics["average_accuracy"],
#         "kappa": metrics["kappa"],
#         "f1_macro": metrics["f1_macro"],
#         "per_class_accuracy": per_class_acc,
#         "confusion_matrix": cm_report,
#         "old_new_split": {
#             k: metrics.get(k)
#             for k in [
#                 "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
#                 "old_class_ids", "new_class_ids", "old_new_split_available",
#             ]
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
#         )
#     if "structured" in output:
#         for k in ("txt_path", "json_path", "confusion_matrix_csv_path", "confusion_matrix_npy_path", "per_class_csv_path"):
#             output[k] = output["structured"].get(k)
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
#         self.phases_seen: List[int] = []

#     def _sanity_check_labels(self, y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
#         yt = _as_1d_np(y_true, "y_true")
#         yp = _as_1d_np(y_pred, "y_pred")
#         if yt.shape[0] != yp.shape[0]:
#             raise ValueError(f"y_true/y_pred length mismatch: {len(yt)} vs {len(yp)}")
#         if yt.min() < 0:
#             raise ValueError(f"Negative true labels detected: min={yt.min()}")
#         return yt, yp

#     def update(
#         self,
#         phase: int,
#         y_true: Any,
#         y_pred: Any,
#         old_class_count: Optional[int] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#         ignore_index: Optional[int] = None,
#     ) -> None:
#         phase = int(phase)
#         yt, yp = self._sanity_check_labels(y_true, y_pred)
#         seen = _ordered_unique_ints(seen_classes) if seen_classes is not None else sorted(np.unique(yt).astype(int).tolist())
#         if not seen:
#             raise ValueError("seen_classes resolved to empty in evaluator.update().")
#         max_pred_pos = int(yp[yp >= 0].max()) if np.any(yp >= 0) else -1
#         C = max(max(seen) + 1, int(yt.max()) + 1, max_pred_pos + 1)
#         metrics = calculate_metrics_torch(
#             y_true=yt,
#             y_pred=yp,
#             num_classes=C,
#             old_class_count=old_class_count,
#             seen_classes=seen,
#             ignore_index=ignore_index,
#             device="cpu",
#         )
#         self.phase_history[phase] = metrics
#         if phase not in self.phases_seen:
#             self.phases_seen.append(phase)
#             self.phases_seen.sort()
#         self._rebuild_class_history()

#     def save_phase_report(self, phase: int, y_true: Any, y_pred: Any, target_names: Optional[List[str]] = None, save_dir: str = "./results", seen_classes: Optional[Iterable[int]] = None, old_class_count: Optional[int] = None, ignore_index: Optional[int] = None, tr_time: Optional[float] = None, te_time: Optional[float] = None, dl_time: Optional[float] = None) -> Dict[str, Any]:
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
#         )

#     def _rebuild_class_history(self) -> None:
#         all_classes = set()
#         for p in self.phases_seen:
#             all_classes.update(int(k) for k in self.phase_history[p].get("per_class_accuracy", {}).keys())
#         new_hist: defaultdict[int, List[float]] = defaultdict(list)
#         presence: defaultdict[int, List[bool]] = defaultdict(list)
#         for cls in sorted(all_classes):
#             for p in self.phases_seen:
#                 per_class = self.phase_history[p].get("per_class_accuracy", {})
#                 if cls in per_class:
#                     new_hist[cls].append(float(per_class[cls]))
#                     presence[cls].append(True)
#                 else:
#                     new_hist[cls].append(float("nan"))
#                     presence[cls].append(False)
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
#         vals = []
#         for history in self.class_acc_history.values():
#             h = np.asarray(history, dtype=float)
#             h = h[~np.isnan(h)]
#             if h.size >= 2:
#                 vals.append(float(h[-1] - h[0]))
#         return float(np.mean(vals)) if vals else 0.0

#     def get_standard_metrics(self) -> Dict[str, float]:
#         if not self.phases_seen:
#             return {}
#         last_phase = self.phases_seen[-1]
#         all_oa = [float(self.phase_history[p].get("overall_accuracy", 0.0)) for p in self.phases_seen]
#         # Average H over phases where old/new split exists. Phase 0 mirrors OA for
#         # reporting readability but should not inflate incremental H_avg.
#         inc_h = [
#             float(self.phase_history[p].get("harmonic_mean", 0.0))
#             for p in self.phases_seen
#             if bool(self.phase_history[p].get("old_new_split_available", False))
#         ]
#         all_h_fallback = [float(self.phase_history[p].get("harmonic_mean", 0.0)) for p in self.phases_seen]
#         forgetting = self.calculate_forgetting_per_class()
#         last = self.phase_history[last_phase]
#         return {
#             "A_last (Final Accuracy)": float(last.get("overall_accuracy", 0.0)),
#             "A_avg (Avg Accuracy)": float(np.mean(all_oa)) if all_oa else 0.0,
#             "H_last (Final Harmonic Mean)": float(last.get("harmonic_mean", 0.0)),
#             "H_avg (Avg Harmonic Mean)": float(np.mean(inc_h)) if inc_h else (float(np.mean(all_h_fallback)) if all_h_fallback else 0.0),
#             "H_avg_inc_only": float(np.mean(inc_h)) if inc_h else 0.0,
#             "F_avg (Avg Forgetting)": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
#             "BWT (Backward Transfer)": self.calculate_backward_transfer(),
#             "Old_last (Final Old Accuracy)": float(last.get("old_accuracy", 0.0)),
#             "New_last (Final New Accuracy)": float(last.get("new_accuracy", 0.0)),
#             "AA_last (Final Avg Accuracy)": float(last.get("average_accuracy", 0.0)),
#             "Kappa_last": float(last.get("kappa", 0.0)),
#             "F1_last": float(last.get("f1_macro", 0.0)),
#             "Invalid_last": float(last.get("invalid_prediction_rate", 0.0)),
#             "Phases": len(self.phases_seen),
#         }

#     def get_phase_table(self) -> List[Dict[str, Any]]:
#         rows = []
#         for p in self.phases_seen:
#             m = self.phase_history[p]
#             rows.append({
#                 "phase": int(p),
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
#                     "first": float(vals[0]),
#                     "best": float(np.max(vals)),
#                     "last": float(vals[-1]),
#                     "forgetting": float(forgetting.get(int(cls), 0.0)),
#                 }
#         return out

#     def print_summary(self) -> None:
#         if not self.phases_seen:
#             print("[NECILEvaluator] No phases evaluated yet.")
#             return
#         metrics = self.get_standard_metrics()
#         last_phase = self.phases_seen[-1]
#         phase_metrics = self.phase_history[last_phase]
#         print("\n" + "=" * 58)
#         print(f" NECIL-HSI Evaluation Report (Phase {last_phase})")
#         print("=" * 58)
#         print(f" 1. Final Accuracy (A_last):      {metrics.get('A_last (Final Accuracy)', 0):.2f}%")
#         print(f" 2. Avg Accuracy (A_avg):         {metrics.get('A_avg (Avg Accuracy)', 0):.2f}%")
#         print(f" 3. Avg Forgetting (F_avg):       {metrics.get('F_avg (Avg Forgetting)', 0):.2f}%")
#         print(f" 4. Backward Transfer (BWT):      {metrics.get('BWT (Backward Transfer)', 0):.2f}%")
#         print(f" 5. Old Accuracy:                 {phase_metrics.get('old_accuracy', 0):.2f}%")
#         print(f" 6. New Accuracy:                 {phase_metrics.get('new_accuracy', 0):.2f}%")
#         print(f" 7. Harmonic Mean:                {phase_metrics.get('harmonic_mean', 0):.2f}%")
#         print(f" 8. Old/New Split Available:      {bool(phase_metrics.get('old_new_split_available', False))}")
#         print(f" 9. Invalid Prediction Rate:      {phase_metrics.get('invalid_prediction_rate', 0):.2f}%")
#         print(f"10. Old -> New Error Rate:        {phase_metrics.get('old_to_new_rate', 0):.2f}%")
#         print(f"11. New -> Old Error Rate:        {phase_metrics.get('new_to_old_rate', 0):.2f}%")
#         print(f"12. AA / Kappa / F1:              {phase_metrics.get('average_accuracy', 0):.2f}% / {phase_metrics.get('kappa', 0):.2f}% / {phase_metrics.get('f1_macro', 0):.2f}%")
#         print("-" * 58)

#     def save_phase_table_csv(self, save_path: str) -> str:
#         parent = os.path.dirname(save_path)
#         if parent:
#             os.makedirs(parent, exist_ok=True)
#         rows = self.get_phase_table()
#         fields = ["phase", "OA", "AA", "Kappa", "F1", "Old", "New", "H", "Invalid", "OldToNew", "NewToOld", "SplitAvailable", "Samples"]
#         with open(save_path, "w", encoding="utf-8", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fields)
#             writer.writeheader()
#             for row in rows:
#                 writer.writerow(row)
#         return save_path

#     def to_dict(self) -> Dict[str, Any]:
#         return make_json_serializable({
#             "phase_history": self.phase_history,
#             "standard_metrics": self.get_standard_metrics(),
#             "phase_table": self.get_phase_table(),
#             "per_class_summary": self.get_per_class_summary(),
#         })






















# """
# Clean evaluation utilities for geometry-native NECIL-HSI.

# Contract
# --------
# - Labels are sequential HSI class ids: 0..C-1 after background removal.
# - `seen_classes` defines the only valid prediction columns for a phase.
# - Future/unseen predictions are NOT silently dropped; they are counted as wrong
#   and reported as leakage.
# - Old/new split follows the provided phase class order when available; otherwise it falls back to global sequential id < old_class_count.

# Outputs
# -------
# - phase_X_classification_report.txt
# - phase_X_classification_report.json
# - phase_X_confusion_matrix.csv
# - phase_X_confusion_matrix.npy
# - phase_X_per_class_metrics.csv
# - phase_X_HSI_Classification_Report.csv
# """

# from __future__ import annotations

# import csv
# import json
# import os
# from collections import defaultdict
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from sklearn.metrics import classification_report


# # ============================================================
# # Generic conversion / serialization
# # ============================================================
# def make_json_serializable(obj: Any) -> Any:
#     if isinstance(obj, dict):
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
#     arr = np.asarray(x).reshape(-1)
#     if arr.size == 0:
#         raise ValueError(f"{name} is empty.")
#     return arr.astype(np.int64, copy=False)


# def _to_1d_long_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
#     if torch.is_tensor(x):
#         t = x.detach()
#     else:
#         t = torch.as_tensor(x)
#     t = t.long().view(-1)
#     return t.to(device) if device is not None else t


# def _safe_class_name(target_names: Optional[Sequence[str]], cls: int) -> str:
#     cls = int(cls)
#     if target_names is not None and 0 <= cls < len(target_names):
#         return str(target_names[cls])
#     return f"Class {cls}"


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     """Preserve phase/class-order while removing duplicates."""
#     out: List[int] = []
#     seen = set()
#     for value in values:
#         c = int(value)
#         if c not in seen:
#             out.append(c)
#             seen.add(c)
#     return out


# def _seen_list(seen_classes: Optional[Iterable[int]], y_true: Optional[np.ndarray] = None) -> Optional[List[int]]:
#     if seen_classes is None:
#         return None
#     out = _ordered_unique_ints(seen_classes)
#     if not out:
#         raise ValueError("seen_classes was provided but empty.")
#     if y_true is not None:
#         true_set = set(int(c) for c in np.unique(y_true).tolist())
#         bad_true = sorted(true_set.difference(set(out)))
#         if bad_true:
#             raise ValueError(
#                 f"y_true contains labels outside seen_classes: {bad_true}. "
#                 "This is usually a phase/dataset split bug, not an evaluation issue."
#             )
#     return out


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
#         "predicted_unseen_classes": sorted(int(c) for c in np.unique(y_pred[invalid]).tolist()) if invalid.any() else [],
#     }


# # ============================================================
# # Torch-native metrics
# # ============================================================
# @torch.no_grad()
# def torch_confusion_matrix(y_true: Any, y_pred: Any, num_classes: int, device: Optional[str] = "cpu") -> torch.Tensor:
#     """Rows=true labels, columns=predicted labels."""
#     if int(num_classes) <= 0:
#         raise ValueError("num_classes must be positive.")
#     dev = torch.device(device) if device is not None else None
#     yt = _to_1d_long_tensor(y_true, dev)
#     yp = _to_1d_long_tensor(y_pred, dev)
#     if yt.numel() != yp.numel():
#         raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

#     valid = (yt >= 0) & (yt < int(num_classes)) & (yp >= 0) & (yp < int(num_classes))
#     yt = yt[valid]
#     yp = yp[valid]
#     idx = yt * int(num_classes) + yp
#     return torch.bincount(idx, minlength=int(num_classes) * int(num_classes)).reshape(int(num_classes), int(num_classes))


# @torch.no_grad()
# def torch_metrics_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-12) -> Dict[str, Any]:
#     cm = cm.detach().float()
#     if cm.dim() != 2 or cm.size(0) != cm.size(1):
#         raise ValueError(f"confusion matrix must be square [C,C], got {tuple(cm.shape)}")

#     tp = torch.diag(cm)
#     support = cm.sum(dim=1)
#     predicted = cm.sum(dim=0)
#     total = cm.sum()

#     recall = tp / (support + eps)
#     precision = tp / (predicted + eps)
#     f1 = 2.0 * precision * recall / (precision + recall + eps)
#     valid = support > 0

#     oa = 100.0 * tp.sum() / (total + eps)
#     aa = 100.0 * recall[valid].mean() if bool(valid.any().item()) else torch.tensor(0.0, device=cm.device)
#     macro_f1 = 100.0 * f1[valid].mean() if bool(valid.any().item()) else torch.tensor(0.0, device=cm.device)

#     po = tp.sum() / (total + eps)
#     pe = (cm.sum(dim=1) * cm.sum(dim=0)).sum() / (total * total + eps)
#     kappa = 100.0 * (po - pe) / (1.0 - pe + eps)

#     return {
#         "overall_accuracy": float(oa.item()),
#         "balanced_accuracy": float(aa.item()),
#         "average_accuracy": float(aa.item()),
#         "kappa": float(kappa.item()),
#         "f1_macro": float(macro_f1.item()),
#         "per_class_accuracy": {int(i): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
#         "precision": {int(i): float((100.0 * precision[i]).item()) for i in range(cm.size(0))},
#         "recall": {int(i): float((100.0 * recall[i]).item()) for i in range(cm.size(0))},
#         "f1_per_class": {int(i): float((100.0 * f1[i]).item()) for i in range(cm.size(0))},
#         "support": {int(i): int(support[i].item()) for i in range(cm.size(0))},
#         "predicted_count": {int(i): int(predicted[i].item()) for i in range(cm.size(0))},
#         "confusion_matrix": cm.detach().cpu(),
#     }


# @torch.no_grad()
# def torch_old_new_metrics(
#     y_true: Any,
#     y_pred: Any,
#     old_class_count: Optional[int] = None,
#     *,
#     seen_classes: Optional[Iterable[int]] = None,
#     eps: float = 1e-12,
# ) -> Dict[str, Any]:
#     """Compute old/new/HM without breaking phase-0 or shuffled class orders.

#     If ``seen_classes`` is provided, the first ``old_class_count`` entries are
#     treated as the old set. This preserves the actual CIL class order. If it is
#     not provided, the legacy fallback is global id < old_class_count.
#     """
#     yt = _to_1d_long_tensor(y_true)
#     yp = _to_1d_long_tensor(y_pred, yt.device)
#     if yt.numel() != yp.numel():
#         raise ValueError(f"y_true/y_pred length mismatch: {yt.numel()} vs {yp.numel()}")

#     overall = 100.0 * (yp == yt).float().mean() if yt.numel() else torch.tensor(0.0, device=yt.device)
#     old_class_count_i = 0 if old_class_count is None else int(old_class_count)

#     if old_class_count_i <= 0:
#         # Phase 0 has no old/new split. Reporting H=0 makes base summaries and
#         # multi-run tables look falsely broken, so expose split availability and
#         # mirror the overall accuracy into old/new/HM for base-phase readability.
#         return {
#             "old_accuracy": float(overall.item()),
#             "new_accuracy": float(overall.item()),
#             "harmonic_mean": float(overall.item()),
#             "old_count": 0,
#             "new_count": int(yt.numel()),
#             "old_class_ids": [],
#             "new_class_ids": _ordered_unique_ints(seen_classes) if seen_classes is not None else sorted(int(c) for c in torch.unique(yt).cpu().tolist()),
#             "old_new_split_available": False,
#         }

#     if seen_classes is not None:
#         seen_order = _ordered_unique_ints(seen_classes)
#         old_ids = seen_order[:old_class_count_i]
#         new_ids = seen_order[old_class_count_i:]
#         if not old_ids or not new_ids:
#             old_mask = yt < old_class_count_i
#             new_mask = yt >= old_class_count_i
#         else:
#             old_t = torch.tensor(old_ids, device=yt.device, dtype=torch.long)
#             new_t = torch.tensor(new_ids, device=yt.device, dtype=torch.long)
#             old_mask = (yt.view(-1, 1) == old_t.view(1, -1)).any(dim=1)
#             new_mask = (yt.view(-1, 1) == new_t.view(1, -1)).any(dim=1)
#     else:
#         old_ids = list(range(old_class_count_i))
#         unique_true = sorted(int(c) for c in torch.unique(yt).cpu().tolist())
#         new_ids = [c for c in unique_true if c >= old_class_count_i]
#         old_mask = yt < old_class_count_i
#         new_mask = yt >= old_class_count_i

#     old_total = int(old_mask.sum().item())
#     new_total = int(new_mask.sum().item())
#     old_acc = 100.0 * (yp[old_mask] == yt[old_mask]).float().mean() if old_total else torch.tensor(0.0, device=yt.device)
#     new_acc = 100.0 * (yp[new_mask] == yt[new_mask]).float().mean() if new_total else torch.tensor(0.0, device=yt.device)
#     if old_total > 0 and new_total > 0:
#         h = 2.0 * old_acc * new_acc / (old_acc + new_acc + eps)
#         split_available = True
#     else:
#         h = torch.tensor(0.0, device=yt.device)
#         split_available = False
#     return {
#         "old_accuracy": float(old_acc.item()),
#         "new_accuracy": float(new_acc.item()),
#         "harmonic_mean": float(h.item()),
#         "old_count": old_total,
#         "new_count": new_total,
#         "old_class_ids": [int(c) for c in old_ids],
#         "new_class_ids": [int(c) for c in new_ids],
#         "old_new_split_available": bool(split_available),
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
# ) -> Dict[str, Any]:
#     yt_np, yp_np, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )

#     # Count unseen predictions as wrong by keeping y_pred in the full CM width.
#     max_seen = max(seen) if seen else -1
#     max_true = int(yt_np.max()) if yt_np.size else -1
#     max_pred = int(yp_np.max()) if yp_np.size else -1
#     C = max(int(num_classes or 0), max_seen + 1, max_true + 1, max_pred + 1)
#     if C <= 0:
#         raise ValueError("Cannot infer num_classes.")

#     cm_full = torch_confusion_matrix(yt_np, yp_np, C, device=device)
#     metrics = torch_metrics_from_confusion_matrix(cm_full)

#     seen_set = set(seen)
#     for key in ("per_class_accuracy", "precision", "recall", "f1_per_class", "support", "predicted_count"):
#         metrics[key] = {int(k): v for k, v in metrics[key].items() if int(k) in seen_set}

#     split = torch_old_new_metrics(yt_np, yp_np, old_class_count=old_class_count, seen_classes=seen)
#     metrics.update(split)
#     metrics.update(_prediction_leakage_np(yp_np, seen))
#     metrics["num_samples"] = int(yt_np.size)
#     metrics["num_classes"] = int(len(seen))
#     metrics["classes"] = [int(c) for c in seen]
#     metrics["confusion_matrix"] = cm_full.detach().cpu()
#     return metrics


# def calculate_metrics(
#     y_true: Any,
#     y_pred: Any,
#     class_names: Optional[List[str]] = None,
#     old_class_count: Optional[int] = None,
#     seen_classes: Optional[Iterable[int]] = None,
#     ignore_index: Optional[int] = None,
# ) -> Dict[str, Any]:
#     del class_names
#     seen = list(seen_classes) if seen_classes is not None else None
#     if seen:
#         num_classes = max(seen) + 1
#     else:
#         yt = np.asarray(y_true).reshape(-1)
#         yp = np.asarray(y_pred).reshape(-1)
#         num_classes = int(max(yt.max(), yp.max())) + 1 if yt.size and yp.size else None
#     return calculate_metrics_torch(
#         y_true=y_true,
#         y_pred=y_pred,
#         num_classes=num_classes,
#         old_class_count=old_class_count,
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#         device="cpu",
#     )


# # ============================================================
# # Report writing
# # ============================================================
# def _report_labels(seen: Sequence[int], y_pred: np.ndarray, include_predicted_unseen: bool) -> List[int]:
#     labels = list(int(c) for c in seen)
#     if include_predicted_unseen:
#         label_set = set(labels)
#         labels += sorted(int(c) for c in np.unique(y_pred).tolist() if int(c) not in label_set)
#     return labels


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
# ) -> Dict[str, Any]:
#     os.makedirs(save_dir, exist_ok=True)

#     yt, yp, seen = _filter_true_labels(
#         _as_1d_np(y_true, "y_true"),
#         _as_1d_np(y_pred, "y_pred"),
#         seen_classes=seen_classes,
#         ignore_index=ignore_index,
#     )
#     labels = _report_labels(seen, yp, include_predicted_unseen=include_predicted_unseen)
#     names = [_safe_class_name(target_names, c) if c in seen else f"UNSEEN-PRED-{c}" for c in labels]
#     C = max(max(labels) + 1 if labels else 0, int(yt.max()) + 1, int(yp.max()) + 1)

#     metrics = calculate_metrics_torch(
#         y_true=yt,
#         y_pred=yp,
#         num_classes=C,
#         old_class_count=old_class_count,
#         seen_classes=seen,
#         ignore_index=None,
#         device="cpu",
#     )
#     cm_full = metrics["confusion_matrix"].detach().cpu().numpy()
#     cm_report = cm_full[np.ix_(labels, labels)]

#     # sklearn report is for display only; scalar truth comes from torch metrics above.
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
#         report_dict[name]["precision"] = float(metrics.get("precision", {}).get(cls, 0.0)) / 100.0
#         report_dict[name]["recall"] = float(metrics.get("recall", {}).get(cls, 0.0)) / 100.0
#         report_dict[name]["f1-score"] = float(metrics.get("f1_per_class", {}).get(cls, 0.0)) / 100.0
#         report_dict[name]["support"] = int(metrics.get("support", {}).get(cls, 0))
#     report_dict["torch_metrics"] = make_json_serializable(metrics)
#     report_dict["old_new_split"] = {
#         k: metrics.get(k)
#         for k in [
#             "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
#             "old_class_ids", "new_class_ids", "old_new_split_available",
#         ]
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
#         f.write(f"OA: {metrics['overall_accuracy']:.4f}%\n")
#         f.write(f"AA: {metrics['average_accuracy']:.4f}%\n")
#         f.write(f"Kappa: {metrics['kappa']:.4f}%\n")
#         f.write(f"Macro-F1: {metrics['f1_macro']:.4f}%\n")
#         f.write(f"Old Accuracy: {metrics.get('old_accuracy', 0.0):.4f}%\n")
#         f.write(f"New Accuracy: {metrics.get('new_accuracy', 0.0):.4f}%\n")
#         f.write(f"Harmonic Mean: {metrics.get('harmonic_mean', 0.0):.4f}%\n")
#         f.write(f"Old/New Split Available: {bool(metrics.get('old_new_split_available', False))}\n")
#         f.write(f"Old Class IDs: {metrics.get('old_class_ids', [])}\n")
#         f.write(f"New Class IDs: {metrics.get('new_class_ids', [])}\n")
#         f.write(f"Invalid Prediction Rate: {metrics.get('invalid_prediction_rate', 0.0):.4f}%\n")
#         f.write(f"Predicted unseen classes: {metrics.get('predicted_unseen_classes', [])}\n")
#         f.write(f"Predicted unseen count: {metrics.get('predicted_unseen_count', 0)}\n\n")
#         f.write(report_text)

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(report_dict), f, indent=2)

#     with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["true\\pred"] + names)
#         for i, name in enumerate(names):
#             writer.writerow([name] + [int(v) for v in cm_report[i].tolist()])
#     np.save(cm_npy_path, cm_report)

#     with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "class_id", "class_name", "seen_class", "precision_percent", "recall_percent",
#             "f1_percent", "accuracy_percent", "support", "predicted_count",
#         ])
#         for cls, name in zip(labels, names):
#             writer.writerow([
#                 int(cls),
#                 name,
#                 bool(int(cls) in set(seen)),
#                 float(metrics.get("precision", {}).get(cls, 0.0)),
#                 float(metrics.get("recall", {}).get(cls, 0.0)),
#                 float(metrics.get("f1_per_class", {}).get(cls, 0.0)),
#                 float(metrics.get("per_class_accuracy", {}).get(cls, 0.0)),
#                 int(metrics.get("support", {}).get(cls, 0)),
#                 int(metrics.get("predicted_count", {}).get(cls, 0)),
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
#     C = max(max(seen) + 1, int(yt.max()) + 1, int(yp.max()) + 1)
#     metrics = calculate_metrics_torch(yt, yp, C, old_class_count, seen, None, device="cpu")
#     labels = list(seen)
#     names = [_safe_class_name(target_names, c) for c in labels]
#     cm_full = metrics["confusion_matrix"].detach().cpu().numpy()
#     cm_report = cm_full[np.ix_(labels, labels)]
#     per_class_acc = [float(metrics.get("per_class_accuracy", {}).get(int(c), 0.0)) for c in labels]

#     report_text = classification_report(yt, yp, labels=labels, target_names=names, digits=4, zero_division=0)
#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write(f"{0.0 if tr_time is None else float(tr_time)} Tr_Time\n")
#         f.write(f"{0.0 if te_time is None else float(te_time)} Te_Time\n")
#         f.write(f"{0.0 if dl_time is None else float(dl_time)} DL_Time\n")
#         f.write(f"{metrics['kappa']} Kappa accuracy (%)\n")
#         f.write(f"{metrics['overall_accuracy']} Overall accuracy (%)\n")
#         f.write(f"{metrics['average_accuracy']} Average accuracy (%)\n")
#         f.write(f"{metrics['f1_macro']} Macro F1 (%)\n")
#         f.write(f"{metrics.get('old_accuracy', 0.0)} Old accuracy (%)\n")
#         f.write(f"{metrics.get('new_accuracy', 0.0)} New accuracy (%)\n")
#         f.write(f"{metrics.get('harmonic_mean', 0.0)} Harmonic mean (%)\n")
#         f.write(f"{bool(metrics.get('old_new_split_available', False))} Old/new split available\n")
#         f.write(f"{metrics.get('old_class_ids', [])} Old class ids\n")
#         f.write(f"{metrics.get('new_class_ids', [])} New class ids\n")
#         f.write(f"{metrics.get('invalid_prediction_rate', 0.0)} Invalid prediction rate (%)\n")
#         f.write(report_text + "\n")
#         f.write(str(np.asarray(per_class_acc)) + "\n")
#         f.write(str(cm_report) + "\n")

#     print(f"[Report] Saved HSI-style classification report: {save_path}")
#     return {
#         "save_path": save_path,
#         "overall_accuracy": metrics["overall_accuracy"],
#         "average_accuracy": metrics["average_accuracy"],
#         "kappa": metrics["kappa"],
#         "f1_macro": metrics["f1_macro"],
#         "per_class_accuracy": per_class_acc,
#         "confusion_matrix": cm_report,
#         "old_new_split": {
#             k: metrics.get(k)
#             for k in [
#                 "old_accuracy", "new_accuracy", "harmonic_mean", "old_count", "new_count",
#                 "old_class_ids", "new_class_ids", "old_new_split_available",
#             ]
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
#         )
#     if "structured" in output:
#         for k in ("txt_path", "json_path", "confusion_matrix_csv_path", "confusion_matrix_npy_path", "per_class_csv_path"):
#             output[k] = output["structured"].get(k)
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
#         self.phases_seen: List[int] = []

#     def _sanity_check_labels(self, y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
#         yt = _as_1d_np(y_true, "y_true")
#         yp = _as_1d_np(y_pred, "y_pred")
#         if yt.shape[0] != yp.shape[0]:
#             raise ValueError(f"y_true/y_pred length mismatch: {len(yt)} vs {len(yp)}")
#         if yt.min() < 0:
#             raise ValueError(f"Negative true labels detected: min={yt.min()}")
#         return yt, yp

#     def update(
#         self,
#         phase: int,
#         y_true: Any,
#         y_pred: Any,
#         old_class_count: Optional[int] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#         ignore_index: Optional[int] = None,
#     ) -> None:
#         phase = int(phase)
#         yt, yp = self._sanity_check_labels(y_true, y_pred)
#         seen = list(seen_classes) if seen_classes is not None else sorted(np.unique(yt).astype(int).tolist())
#         C = max(max(seen) + 1, int(yt.max()) + 1, int(yp.max()) + 1)
#         metrics = calculate_metrics_torch(
#             y_true=yt,
#             y_pred=yp,
#             num_classes=C,
#             old_class_count=old_class_count,
#             seen_classes=seen,
#             ignore_index=ignore_index,
#             device="cpu",
#         )
#         self.phase_history[phase] = metrics
#         if phase not in self.phases_seen:
#             self.phases_seen.append(phase)
#             self.phases_seen.sort()
#         self._rebuild_class_history()

#     def save_phase_report(self, phase: int, y_true: Any, y_pred: Any, target_names: Optional[List[str]] = None, save_dir: str = "./results", seen_classes: Optional[Iterable[int]] = None, old_class_count: Optional[int] = None, ignore_index: Optional[int] = None, tr_time: Optional[float] = None, te_time: Optional[float] = None, dl_time: Optional[float] = None) -> Dict[str, Any]:
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
#         )

#     def _rebuild_class_history(self) -> None:
#         all_classes = set()
#         for p in self.phases_seen:
#             all_classes.update(int(k) for k in self.phase_history[p].get("per_class_accuracy", {}).keys())
#         new_hist: defaultdict[int, List[float]] = defaultdict(list)
#         presence: defaultdict[int, List[bool]] = defaultdict(list)
#         for cls in sorted(all_classes):
#             for p in self.phases_seen:
#                 per_class = self.phase_history[p].get("per_class_accuracy", {})
#                 if cls in per_class:
#                     new_hist[cls].append(float(per_class[cls]))
#                     presence[cls].append(True)
#                 else:
#                     new_hist[cls].append(float("nan"))
#                     presence[cls].append(False)
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
#         vals = []
#         for history in self.class_acc_history.values():
#             h = np.asarray(history, dtype=float)
#             h = h[~np.isnan(h)]
#             if h.size >= 2:
#                 vals.append(float(h[-1] - h[0]))
#         return float(np.mean(vals)) if vals else 0.0

#     def get_standard_metrics(self) -> Dict[str, float]:
#         if not self.phases_seen:
#             return {}
#         last_phase = self.phases_seen[-1]
#         all_oa = [float(self.phase_history[p].get("overall_accuracy", 0.0)) for p in self.phases_seen]
#         # Average H over phases where old/new split exists. Phase 0 mirrors OA for
#         # reporting readability but should not inflate incremental H_avg.
#         inc_h = [
#             float(self.phase_history[p].get("harmonic_mean", 0.0))
#             for p in self.phases_seen
#             if bool(self.phase_history[p].get("old_new_split_available", False))
#         ]
#         all_h_fallback = [float(self.phase_history[p].get("harmonic_mean", 0.0)) for p in self.phases_seen]
#         forgetting = self.calculate_forgetting_per_class()
#         last = self.phase_history[last_phase]
#         return {
#             "A_last (Final Accuracy)": float(last.get("overall_accuracy", 0.0)),
#             "A_avg (Avg Accuracy)": float(np.mean(all_oa)) if all_oa else 0.0,
#             "H_last (Final Harmonic Mean)": float(last.get("harmonic_mean", 0.0)),
#             "H_avg (Avg Harmonic Mean)": float(np.mean(inc_h)) if inc_h else (float(np.mean(all_h_fallback)) if all_h_fallback else 0.0),
#             "H_avg_inc_only": float(np.mean(inc_h)) if inc_h else 0.0,
#             "F_avg (Avg Forgetting)": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
#             "BWT (Backward Transfer)": self.calculate_backward_transfer(),
#             "Old_last (Final Old Accuracy)": float(last.get("old_accuracy", 0.0)),
#             "New_last (Final New Accuracy)": float(last.get("new_accuracy", 0.0)),
#             "AA_last (Final Avg Accuracy)": float(last.get("average_accuracy", 0.0)),
#             "Kappa_last": float(last.get("kappa", 0.0)),
#             "F1_last": float(last.get("f1_macro", 0.0)),
#             "Invalid_last": float(last.get("invalid_prediction_rate", 0.0)),
#             "Phases": len(self.phases_seen),
#         }

#     def get_phase_table(self) -> List[Dict[str, Any]]:
#         rows = []
#         for p in self.phases_seen:
#             m = self.phase_history[p]
#             rows.append({
#                 "phase": int(p),
#                 "OA": float(m.get("overall_accuracy", 0.0)),
#                 "AA": float(m.get("average_accuracy", 0.0)),
#                 "Kappa": float(m.get("kappa", 0.0)),
#                 "F1": float(m.get("f1_macro", 0.0)),
#                 "Old": float(m.get("old_accuracy", 0.0)),
#                 "New": float(m.get("new_accuracy", 0.0)),
#                 "H": float(m.get("harmonic_mean", 0.0)),
#                 "Invalid": float(m.get("invalid_prediction_rate", 0.0)),
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
#                     "first": float(vals[0]),
#                     "best": float(np.max(vals)),
#                     "last": float(vals[-1]),
#                     "forgetting": float(forgetting.get(int(cls), 0.0)),
#                 }
#         return out

#     def print_summary(self) -> None:
#         if not self.phases_seen:
#             print("[NECILEvaluator] No phases evaluated yet.")
#             return
#         metrics = self.get_standard_metrics()
#         last_phase = self.phases_seen[-1]
#         phase_metrics = self.phase_history[last_phase]
#         print("\n" + "=" * 58)
#         print(f" NECIL-HSI Evaluation Report (Phase {last_phase})")
#         print("=" * 58)
#         print(f" 1. Final Accuracy (A_last):      {metrics.get('A_last (Final Accuracy)', 0):.2f}%")
#         print(f" 2. Avg Accuracy (A_avg):         {metrics.get('A_avg (Avg Accuracy)', 0):.2f}%")
#         print(f" 3. Avg Forgetting (F_avg):       {metrics.get('F_avg (Avg Forgetting)', 0):.2f}%")
#         print(f" 4. Backward Transfer (BWT):      {metrics.get('BWT (Backward Transfer)', 0):.2f}%")
#         print(f" 5. Old Accuracy:                 {phase_metrics.get('old_accuracy', 0):.2f}%")
#         print(f" 6. New Accuracy:                 {phase_metrics.get('new_accuracy', 0):.2f}%")
#         print(f" 7. Harmonic Mean:                {phase_metrics.get('harmonic_mean', 0):.2f}%")
#         print(f" 8. Old/New Split Available:      {bool(phase_metrics.get('old_new_split_available', False))}")
#         print(f" 9. Invalid Prediction Rate:      {phase_metrics.get('invalid_prediction_rate', 0):.2f}%")
#         print(f"10. AA / Kappa / F1:              {phase_metrics.get('average_accuracy', 0):.2f}% / {phase_metrics.get('kappa', 0):.2f}% / {phase_metrics.get('f1_macro', 0):.2f}%")
#         print("-" * 58)

#     def save_phase_table_csv(self, save_path: str) -> str:
#         parent = os.path.dirname(save_path)
#         if parent:
#             os.makedirs(parent, exist_ok=True)
#         rows = self.get_phase_table()
#         fields = ["phase", "OA", "AA", "Kappa", "F1", "Old", "New", "H", "Invalid", "SplitAvailable", "Samples"]
#         with open(save_path, "w", encoding="utf-8", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fields)
#             writer.writeheader()
#             for row in rows:
#                 writer.writerow(row)
#         return save_path

#     def to_dict(self) -> Dict[str, Any]:
#         return make_json_serializable({
#             "phase_history": self.phase_history,
#             "standard_metrics": self.get_standard_metrics(),
#             "phase_table": self.get_phase_table(),
#             "per_class_summary": self.get_per_class_summary(),
#         })
