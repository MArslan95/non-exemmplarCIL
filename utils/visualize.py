from __future__ import annotations

"""Reporting-only visualization for the one-space NECIL-HSI base phase.

Internal labels are zero-based global class IDs. Rendered value 0 is reserved
for background/unseen pixels; semantic class c is rendered as c+1.

This module never changes model parameters, class geometry, normalization, or
phase state.
"""

import math
import os
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


def _as_int(value: object, name: str) -> int:
    if isinstance(value, (np.bool_, bool)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    raise ValueError(f"{name} must be an integer")


def _class_ids(values: Sequence[int], *, total_classes: Optional[int] = None) -> list[int]:
    ids = [_as_int(value, "class_id") for value in values]
    if not ids:
        raise ValueError("class_ids must not be empty")
    if len(ids) != len(set(ids)) or any(value < 0 for value in ids):
        raise ValueError("class_ids must contain unique non-negative IDs")
    if total_classes is not None:
        total = _as_int(total_classes, "total_classes")
        if total <= 0 or max(ids) >= total:
            raise ValueError("total_classes is inconsistent with class_ids")
    return ids


def _class_name(
    class_id: int,
    target_names: Optional[Sequence[str]],
) -> str:
    names = list(target_names or [])
    return str(names[class_id]) if class_id < len(names) else f"Class {class_id + 1}"


def _palette(
    total_classes: int,
    cmap_name: str,
) -> tuple[ListedColormap, BoundaryNorm]:
    total = _as_int(total_classes, "total_classes")
    if total <= 0:
        raise ValueError("total_classes must be positive")
    try:
        source = plt.get_cmap(str(cmap_name))
    except ValueError as exc:
        raise ValueError(f"unknown matplotlib colormap {cmap_name!r}") from exc

    # Mid-bin samples avoid the continuous-map endpoints and remain fixed for
    # every phase because the full dataset class count is used.
    positions = (np.arange(total, dtype=np.float64) + 0.5) / float(total)
    class_colours = source(positions)
    background = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    cmap = ListedColormap(np.vstack((background, class_colours)))
    boundaries = np.arange(-0.5, total + 1.5, 1.0, dtype=np.float64)
    return cmap, BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)


def _validated_labels(
    values: np.ndarray,
    *,
    name: str,
    class_ids: Sequence[int],
    total_classes: int,
) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)
    if raw.dtype == np.bool_ or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain integer class IDs")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
            raise ValueError(f"{name} must contain finite integer class IDs")
    result = raw.astype(np.int64, copy=False)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(result < 0) or np.any(result >= int(total_classes)):
        raise ValueError(f"{name} contains IDs outside the dataset label range")
    outside = sorted(set(result.tolist()) - set(int(value) for value in class_ids))
    if outside:
        raise ValueError(f"{name} contains phase-invisible classes: {outside}")
    return result


def _validated_coordinates(
    coords: np.ndarray,
    *,
    count: int,
    gt_shape: Sequence[int],
) -> tuple[np.ndarray, tuple[int, int]]:
    raw = np.asarray(coords)
    if raw.dtype == np.bool_ or np.iscomplexobj(raw):
        raise ValueError("coords must contain integer row/column indices")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
            raise ValueError("coords must contain finite integer indices")
    result = raw.astype(np.int64, copy=False)
    if result.shape != (int(count), 2):
        raise ValueError(f"coords must have shape {(int(count), 2)}")
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("coords contain duplicate image locations")

    if len(gt_shape) != 2:
        raise ValueError("gt_shape must contain [height,width]")
    shape = (int(gt_shape[0]), int(gt_shape[1]))
    if min(shape) <= 0:
        raise ValueError("gt_shape must be positive")
    rows, cols = result[:, 0], result[:, 1]
    if (
        np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(cols < 0)
        or np.any(cols >= shape[1])
    ):
        raise ValueError("coords contain positions outside gt_shape")
    return result, shape


def _render_map(
    path: str,
    values: np.ndarray,
    *,
    total_classes: int,
    cmap_name: str,
    dpi: int,
) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2:
        raise ValueError("qualitative map must be two-dimensional")
    if np.any(array < 0) or np.any(array > int(total_classes)):
        raise ValueError("qualitative map contains invalid rendered IDs")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    cmap, norm = _palette(total_classes, cmap_name)
    height, width = array.shape
    long_side = 7.2
    aspect = float(width) / float(height)
    figure_size = (
        (long_side, long_side / aspect)
        if aspect >= 1.0
        else (long_side * aspect, long_side)
    )
    figure = plt.figure(figsize=figure_size, dpi=int(dpi), frameon=False)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_axis_off()
    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches=None,
        transparent=False,
        pad_inches=0.0,
    )
    plt.close(figure)
    return destination


def _render_legend(
    path: str,
    *,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    total_classes: int,
    cmap_name: str,
    dpi: int,
) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    cmap, norm = _palette(total_classes, cmap_name)
    handles = [
        Patch(facecolor=cmap(norm(0)), edgecolor="none", label="Background / unseen")
    ]
    handles.extend(
        Patch(
            facecolor=cmap(norm(class_id + 1)),
            edgecolor="none",
            label=f"{class_id + 1}. {_class_name(class_id, target_names)}",
        )
        for class_id in class_ids
    )
    columns = 1 if len(handles) <= 8 else 2
    rows = int(math.ceil(len(handles) / columns))
    figure = plt.figure(
        figsize=(4.6 * columns, max(1.2, 0.42 * rows)),
        dpi=int(dpi),
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_axis_off()
    axis.legend(
        handles=handles,
        loc="center",
        ncol=columns,
        frameon=False,
        fontsize=10,
    )
    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches="tight",
        transparent=False,
        pad_inches=0.04,
    )
    plt.close(figure)
    return destination


def save_full_phase_qualitative_maps(
    *,
    output_dir: str,
    phase: int,
    gt_shape: Sequence[int],
    coords: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_ids: Sequence[int],
    total_classes: int,
    target_names: Optional[Sequence[str]] = None,
    cmap_name: str = "nipy_spectral",
    dpi: int = 300,
) -> Dict[str, str]:
    """Save full labeled-scene ground-truth, prediction, and legend images."""
    phase_id = _as_int(phase, "phase")
    if phase_id < 0:
        raise ValueError("phase must be non-negative")
    ids = _class_ids(class_ids, total_classes=total_classes)
    if target_names is not None and len(target_names) != int(total_classes):
        raise ValueError("target_names must contain one name per dataset class")

    truth = _validated_labels(
        targets,
        name="targets",
        class_ids=ids,
        total_classes=total_classes,
    )
    prediction = _validated_labels(
        predictions,
        name="predictions",
        class_ids=ids,
        total_classes=total_classes,
    )
    if prediction.size != truth.size:
        raise ValueError("targets and predictions must align")
    coordinates, shape = _validated_coordinates(
        coords,
        count=truth.size,
        gt_shape=gt_shape,
    )

    gt_map = np.zeros(shape, dtype=np.int32)
    pred_map = np.zeros(shape, dtype=np.int32)
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    gt_map[rows, cols] = truth.astype(np.int32) + 1
    pred_map[rows, cols] = prediction.astype(np.int32) + 1

    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)
    prefix = f"phase_{phase_id:02d}"
    paths = {
        "ground_truth": _render_map(
            os.path.join(root, f"{prefix}_ground_truth.png"),
            gt_map,
            total_classes=int(total_classes),
            cmap_name=str(cmap_name),
            dpi=int(dpi),
        ),
        "prediction": _render_map(
            os.path.join(root, f"{prefix}_prediction.png"),
            pred_map,
            total_classes=int(total_classes),
            cmap_name=str(cmap_name),
            dpi=int(dpi),
        ),
        "legend": _render_legend(
            os.path.join(root, f"{prefix}_legend.png"),
            class_ids=ids,
            target_names=target_names,
            total_classes=int(total_classes),
            cmap_name=str(cmap_name),
            dpi=int(dpi),
        ),
    }
    return paths


def _metric(
    metrics: Mapping[str, Any],
    key: str,
    class_id: int,
) -> float:
    values = metrics.get(key)
    if not isinstance(values, Mapping):
        raise ValueError(f"metrics lack {key}")
    value = values.get(class_id, values.get(str(class_id)))
    if value is None:
        raise ValueError(f"{key} lacks class {class_id}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def save_geometry_diagnostic_figures(
    *,
    output_dir: str,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    structural_geometry: Mapping[str, Any],
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]] = None,
    dpi: int = 300,
) -> Dict[str, str]:
    """Save diagnostics for the deployed pairwise decision-cell geometry.

    This function is reporting-only.  It consumes the metric contract produced
    by the pairwise evaluator:

    - per_class_accuracy
    - per_class_true_cell_coverage
    - per_class_rival_cell_invasion_rate
    - per_class_cell_fit
    - per_class_mean_decision_margin

    Structural diagnostics consume the committed pairwise-boundary contract:

    - left_class_id / right_class_id
    - normal_norm
    - offset
    - shared_boundary
    - strict_interior_overlap

    No box-support width, support-overlap gap, Gaussian statistic, or validity
    threshold is used.
    """
    ids = _class_ids(class_ids)
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    structural_ids = structural_geometry.get("class_ids")
    if not isinstance(structural_ids, Sequence):
        raise ValueError("structural_geometry lacks class_ids")
    if [int(v) for v in structural_ids] != ids:
        raise ValueError("structural_geometry class_ids disagree with requested classes")

    expected_pairs = len(ids) * (len(ids) - 1) // 2
    if int(structural_geometry.get("pair_count", -1)) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} committed pair boundaries, "
            f"got {structural_geometry.get('pair_count')}"
        )
    if structural_geometry.get("strict_interior_overlap") != "impossible by construction":
        raise ValueError("structural geometry does not expose the pairwise-cell invariant")

    labels = [
        f"{class_id + 1}\n{_class_name(class_id, target_names)}"
        for class_id in ids
    ]
    x = np.arange(len(ids), dtype=np.float64)
    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)
    paths: Dict[str, str] = {}

    split_specs = (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    )

    # 1. Per-class accuracy on all three splits.
    accuracy = np.asarray(
        [
            [_metric(metrics, "per_class_accuracy", class_id) for class_id in ids]
            for _, metrics in split_specs
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.25 * len(ids)), 4.8),
        constrained_layout=True,
    )
    width = 0.24
    for index, ((name, _), values) in enumerate(zip(split_specs, accuracy)):
        axis.bar(x + (index - 1) * width, values, width=width, label=name)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Per-class accuracy")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    axis.legend(frameon=False)
    path = os.path.join(root, "geometry_class_accuracy.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["class_accuracy"] = path

    # 2. Own decision-cell coverage across train/validation/test.
    coverage = np.asarray(
        [
            [
                _metric(metrics, "per_class_true_cell_coverage", class_id)
                for class_id in ids
            ]
            for _, metrics in split_specs
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.25 * len(ids)), 4.8),
        constrained_layout=True,
    )
    for index, ((name, _), values) in enumerate(zip(split_specs, coverage)):
        axis.bar(x + (index - 1) * width, values, width=width, label=name)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Own decision-cell coverage")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    axis.legend(frameon=False)
    path = os.path.join(root, "geometry_class_cell_coverage.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["class_cell_coverage"] = path

    # 3. Rival-cell invasion across train/validation/test.
    invasion = np.asarray(
        [
            [
                _metric(metrics, "per_class_rival_cell_invasion_rate", class_id)
                for class_id in ids
            ]
            for _, metrics in split_specs
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.25 * len(ids)), 4.8),
        constrained_layout=True,
    )
    for index, ((name, _), values) in enumerate(zip(split_specs, invasion)):
        axis.bar(x + (index - 1) * width, values, width=width, label=name)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Rival decision-cell invasion rate")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    axis.legend(frameon=False)
    path = os.path.join(root, "geometry_class_rival_invasion.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["class_rival_invasion"] = path

    # 4. Cell-fit violation magnitude.  Zero is the intrinsic satisfied state.
    cell_fit = np.asarray(
        [
            [_metric(metrics, "per_class_cell_fit", class_id) for class_id in ids]
            for _, metrics in split_specs
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.25 * len(ids)), 4.8),
        constrained_layout=True,
    )
    for index, ((name, _), values) in enumerate(zip(split_specs, cell_fit)):
        axis.bar(x + (index - 1) * width, values, width=width, label=name)
    axis.set_ylabel("Mean decision-cell violation  ReLU(E_true)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    axis.legend(frameon=False)
    path = os.path.join(root, "geometry_class_cell_fit.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["class_cell_fit"] = path

    # 5. Deployed test decision margin.
    margins = np.asarray(
        [
            _metric(test, "per_class_mean_decision_margin", class_id)
            for class_id in ids
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.25 * len(ids)), 4.6),
        constrained_layout=True,
    )
    axis.bar(x, margins, width=0.62)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_ylabel("Test mean decision margin")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "geometry_class_margin.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["class_margin"] = path

    # 6. Pairwise affine-boundary offsets.
    #
    # For stored pair (a,b), a<b:
    #     h_ab(z) = n_ab^T z + q_ab
    # and the reverse orientation is -h_ab.  The matrix therefore stores q_ab
    # at [a,b] and -q_ab at [b,a].  This is a structural sanity visualization,
    # not a class-separation "gap" metric.
    pairs = structural_geometry.get("pairs")
    if not isinstance(pairs, Sequence):
        raise ValueError("structural_geometry lacks pair diagnostics")
    if len(pairs) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} unordered pair diagnostics, got {len(pairs)}"
        )

    if expected_pairs:
        index_of = {class_id: index for index, class_id in enumerate(ids)}
        offset_matrix = np.full((len(ids), len(ids)), np.nan, dtype=np.float64)
        observed: set[tuple[int, int]] = set()

        for row in pairs:
            if not isinstance(row, Mapping):
                raise ValueError("pair diagnostics must be mappings")
            left = int(row["left_class_id"])
            right = int(row["right_class_id"])
            if left not in index_of or right not in index_of or left >= right:
                raise ValueError(
                    "pair diagnostics must use valid ordered class IDs left < right"
                )
            key = (left, right)
            if key in observed:
                raise ValueError(f"duplicate pair diagnostic {key}")
            observed.add(key)

            if row.get("shared_boundary") is not True:
                raise ValueError(f"pair {key} is not marked as a shared boundary")
            if row.get("strict_interior_overlap") is not False:
                raise ValueError(
                    f"pair {key} violates the strict-interior non-overlap contract"
                )

            normal_norm = float(row["normal_norm"])
            offset = float(row["offset"])
            if not math.isfinite(normal_norm) or normal_norm <= 0.0:
                raise ValueError(f"pair {key} has an invalid boundary normal norm")
            if not math.isfinite(offset):
                raise ValueError(f"pair {key} has a non-finite boundary offset")

            i, j = index_of[left], index_of[right]
            offset_matrix[i, j] = offset
            offset_matrix[j, i] = -offset

        finite = offset_matrix[np.isfinite(offset_matrix)]
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        limit = max(limit, np.finfo(np.float64).eps)

        figure, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        image = axis.imshow(
            np.ma.masked_invalid(offset_matrix),
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            interpolation="nearest",
        )
        display_ids = [str(class_id + 1) for class_id in ids]
        axis.set_xticks(np.arange(len(ids)), display_ids)
        axis.set_yticks(np.arange(len(ids)), display_ids)
        axis.set_xlabel("Rival class display ID")
        axis.set_ylabel("Reference class display ID")
        colour_bar = figure.colorbar(image, ax=axis, shrink=0.88)
        colour_bar.set_label("Oriented affine-boundary offset")
        path = os.path.join(root, "geometry_pairwise_boundary_offset.png")
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        paths["pairwise_boundary_offset"] = path

    return paths


def save_incremental_geometry_diagnostic_figures(
    *,
    output_dir: str,
    test: Mapping[str, Any],
    old_test: Mapping[str, Any],
    new_test: Mapping[str, Any],
    boundary_preservation: Mapping[str, Any],
    class_ids: Sequence[int],
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    target_names: Optional[Sequence[str]] = None,
    dpi: int = 300,
) -> Dict[str, str]:
    """Save phase-t diagnostics without pretending old TRAIN/VAL still exist.

    Incremental optimization has real TRAIN/VAL only for current classes, so
    cumulative figures are based on the cumulative TEST split. Old/new group
    balance and forgetting are visualized from the finalized phase summaries.
    """
    ids = _class_ids(class_ids)
    old_ids = [int(value) for value in old_class_ids]
    new_ids = [int(value) for value in new_class_ids]
    if ids != old_ids + new_ids:
        raise ValueError(
            "incremental diagnostic classes must be old_ids + new_ids"
        )
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    labels = [
        f"{class_id + 1}\n{_class_name(class_id, target_names)}"
        for class_id in ids
    ]
    x = np.arange(len(ids), dtype=np.float64)
    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)
    paths: Dict[str, str] = {}

    # 1) Cumulative test per-class accuracy.
    accuracy = np.asarray(
        [
            _metric(test, "per_class_accuracy", class_id)
            for class_id in ids
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(8.0, 1.05 * len(ids)), 4.8),
        constrained_layout=True,
    )
    axis.bar(x, accuracy, width=0.68)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Cumulative test accuracy")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "incremental_test_class_accuracy.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["test_class_accuracy"] = path

    # 2) Cumulative true pair-violation rate per class.
    pair_violation = np.asarray(
        [
            _metric(
                test,
                "per_class_true_pair_violation_rate",
                class_id,
            )
            for class_id in ids
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(8.0, 1.05 * len(ids)), 4.8),
        constrained_layout=True,
    )
    axis.bar(x, pair_violation, width=0.68)
    axis.set_ylim(0.0, max(1.0, float(pair_violation.max()) * 1.05))
    axis.set_ylabel("Cumulative test pair-violation rate")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "incremental_test_pair_violation.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["test_pair_violation"] = path

    # 3) Cumulative test decision margin per class.
    margins = np.asarray(
        [
            _metric(
                test,
                "per_class_mean_decision_margin",
                class_id,
            )
            for class_id in ids
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(max(8.0, 1.05 * len(ids)), 4.8),
        constrained_layout=True,
    )
    axis.bar(x, margins, width=0.68)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_ylabel("Cumulative test mean decision margin")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "incremental_test_decision_margin.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["test_decision_margin"] = path

    # 4) Own-cell coverage vs rival invasion.
    coverage = np.asarray(
        [
            _metric(
                test,
                "per_class_true_cell_coverage",
                class_id,
            )
            for class_id in ids
        ],
        dtype=np.float64,
    )
    invasion = np.asarray(
        [
            _metric(
                test,
                "per_class_rival_cell_invasion_rate",
                class_id,
            )
            for class_id in ids
        ],
        dtype=np.float64,
    )
    width = 0.36
    figure, axis = plt.subplots(
        figsize=(max(8.0, 1.05 * len(ids)), 4.8),
        constrained_layout=True,
    )
    axis.bar(x - width / 2.0, coverage, width=width, label="Own-cell coverage")
    axis.bar(x + width / 2.0, invasion, width=width, label="Rival invasion")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Cumulative test rate")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    axis.legend(frameon=False)
    path = os.path.join(root, "incremental_test_cell_behavior.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["test_cell_behavior"] = path

    # 5) Old/new stability-plasticity balance.
    old_ba = float(old_test["balanced_accuracy"])
    new_ba = float(new_test["balanced_accuracy"])
    denominator = old_ba + new_ba
    harmonic = (
        0.0
        if denominator == 0.0
        else 2.0 * old_ba * new_ba / denominator
    )
    values = np.asarray([old_ba, new_ba, harmonic], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    axis.bar(np.arange(3), values, width=0.62)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Balanced accuracy")
    axis.set_xticks(np.arange(3), ["Old BA", "New BA", "Harmonic"])
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "incremental_old_new_balance.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["old_new_balance"] = path

    # 6) Historical change summary. Mixed units are kept in separate figures:
    # rates in percentage-point units here; margin delta below.
    delta_fields = (
        ("Old BA", "old_balanced_accuracy_delta"),
        ("Cell coverage", "old_cell_coverage_delta"),
        ("Pair violation", "old_pair_violation_delta"),
        ("No-cell", "old_no_cell_rate_delta"),
        ("Rival invasion", "old_rival_invasion_delta"),
    )
    delta_values = np.asarray(
        [
            100.0 * float(boundary_preservation[field])
            for _, field in delta_fields
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axis.bar(np.arange(len(delta_fields)), delta_values, width=0.62)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_ylabel("Historical change (percentage points)")
    axis.set_xticks(
        np.arange(len(delta_fields)),
        [label for label, _ in delta_fields],
        rotation=18,
        ha="right",
    )
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "incremental_old_rate_deltas.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["old_rate_deltas"] = path

    margin_delta = float(
        boundary_preservation["old_decision_margin_delta"]
    )
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    axis.bar([0], [margin_delta], width=0.55)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_ylabel("Historical mean decision-margin change")
    axis.set_xticks([0], ["Old classes"])
    axis.grid(axis="y", linewidth=0.8, alpha=0.35)
    path = os.path.join(root, "incremental_old_margin_delta.png")
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    paths["old_margin_delta"] = path

    # 7) Pairwise violation heatmap when evaluator provides pair diagnostics.
    pair_metrics = test.get("pairwise_boundary_metrics")
    if isinstance(pair_metrics, Mapping) and pair_metrics:
        index_of = {class_id: index for index, class_id in enumerate(ids)}
        matrix = np.full((len(ids), len(ids)), np.nan, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)

        for key, row in pair_metrics.items():
            if not isinstance(row, Mapping):
                continue
            left = int(row["left_class_id"])
            right = int(row["right_class_id"])
            if left not in index_of or right not in index_of:
                continue
            value = float(row["combined_violation_rate"])
            i, j = index_of[left], index_of[right]
            matrix[i, j] = matrix[j, i] = value

        figure, axis = plt.subplots(
            figsize=(max(6.0, 0.48 * len(ids) + 2.0), max(5.4, 0.45 * len(ids) + 1.8)),
            constrained_layout=True,
        )
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            interpolation="nearest",
            vmin=0.0,
            vmax=max(1.0, float(np.nanmax(matrix))),
        )
        display_ids = [str(class_id + 1) for class_id in ids]
        axis.set_xticks(np.arange(len(ids)), display_ids)
        axis.set_yticks(np.arange(len(ids)), display_ids)
        axis.set_xlabel("Class display ID")
        axis.set_ylabel("Class display ID")
        colour_bar = figure.colorbar(image, ax=axis, shrink=0.88)
        colour_bar.set_label("Combined pair-violation rate")
        path = os.path.join(root, "incremental_pairwise_violation_matrix.png")
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        paths["pairwise_violation_matrix"] = path

    return paths


__all__ = [
    "save_full_phase_qualitative_maps",
    "save_geometry_diagnostic_figures",
    "save_incremental_geometry_diagnostic_figures",
]

