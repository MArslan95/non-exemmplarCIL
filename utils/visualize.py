from __future__ import annotations

"""Phase-wise visualizations for factor-geometry NECIL-HSI.

Required evidence retained
--------------------------
* A separate ground-truth map and predicted-GT map for every accepted phase.
* Official held-out cumulative-test maps.
* Qualitative cumulative all-labeled maps.
* Confusion matrix and class-wise precision/recall/F1.
* Base and incremental training dynamics.
* Spectral-spatial factor GeometryBank diagnostics.
* Pair-risk, factor-overlap, spectral-shape, and directional-invasion reviews.
* Across-phase OA/AA/old/new/H/forgetting summaries.

Display value zero is always background/not evaluated. Semantic NumPy maps use
-1 for background and global sequential class IDs for visible pixels.
"""

import os
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from utils.eval import (
        NECILEvaluator,  evaluate_factor_geometry_loader,
        unpack_factor_eval_batch,
    )


CLASSIFICATION_FACTORIZATION = "p(z|c)"
SPECTRAL_RELATION_FACTORIZATION = "p(h|c)"


# =============================================================================
# Shared helpers
# =============================================================================


def _ordered_unique(values: Iterable[int], *, name: str) -> List[int]:
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
    if not output:
        raise ValueError(f"{name} is empty")
    return output


def _safe_name(
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


def _seen_classes(
    dataset_manager: Any,
    phase: int,
) -> List[int]:
    getter = getattr(dataset_manager, "get_seen_classes", None)
    if callable(getter):
        return _ordered_unique(
            getter(int(phase)),
            name="seen_classes",
        )
    schedule = getattr(dataset_manager, "phase_to_classes", None)
    if not isinstance(schedule, Mapping):
        raise RuntimeError(
            "dataset manager lacks get_seen_classes or phase_to_classes"
        )
    output: List[int] = []
    for index in range(int(phase) + 1):
        output.extend(int(value) for value in schedule[index])
    return _ordered_unique(output, name="seen_classes")


def _phase_classes(
    dataset_manager: Any,
    phase: int,
) -> Tuple[List[int], List[int], List[int]]:
    seen = _seen_classes(dataset_manager, phase)
    schedule = getattr(dataset_manager, "phase_to_classes")
    new = _ordered_unique(
        schedule[int(phase)],
        name="new_classes",
    )
    new_set = set(new)
    old = [class_id for class_id in seen if class_id not in new_set]
    return old, new, seen


def _phase_loader(
    dataset_manager: Any,
    *,
    phase: int,
    split: str,
    batch_size: int,
) -> Any:
    loader = getattr(
        dataset_manager,
        "get_cumulative_dataloader",
        None,
    )
    if not callable(loader):
        raise RuntimeError(
            "dataset manager must expose get_cumulative_dataloader"
        )
    return loader(
        int(phase),
        split=str(split),
        batch_size=int(batch_size),
        shuffle=False,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _history_rows(
    history: Mapping[str, Any],
    key: str,
) -> List[Mapping[str, Any]]:
    values = history.get(key, [])
    if not isinstance(values, (tuple, list)):
        return []
    return [
        value
        for value in values
        if isinstance(value, Mapping)
    ]


def _series(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    nested: Optional[str] = None,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    epochs: List[int] = []
    values: List[float] = []
    for index, row in enumerate(rows):
        source: Any = row
        if nested is not None:
            source = row.get(nested, {})
        if not isinstance(source, Mapping):
            continue
        value = source.get(key)
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(scalar):
            continue
        epochs.append(int(row.get("epoch", index + 1)))
        values.append(scalar * float(scale))
    return (
        np.asarray(epochs, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


# =============================================================================
# Stable class color mapping and publication maps
# =============================================================================


def _normalize_class_cmap_name(cmap_name: str) -> str:
    """Return a valid Matplotlib class colormap name.

    ``npy_spectral`` is accepted as a user-facing alias for Matplotlib's exact
    colormap name ``nipy_spectral``.
    """
    token = str(cmap_name or "nipy_spectral").strip()
    normalized = token.lower().replace("-", "_")
    aliases = {
        "npy_spectral": "nipy_spectral",
        "nipy_spectral": "nipy_spectral",
    }
    return aliases.get(normalized, token)


def _build_cmap(
    class_ids: Sequence[int],
    *,
    total_class_count: int,
    cmap_name: str,
    background_color: str,
    color_scope: str = "visible",
) -> ListedColormap:
    """Build the discrete publication palette.

    ``color_scope='visible'`` reproduces the supplied reference figure:
    visible classes are sampled uniformly from ``nipy_spectral`` over
    [0.08, 0.92]. For six classes, the rendered RGB values are exactly

        #800091, #0070DD, #00A45D, #00F400, #FFC900, #D50000.

    ``color_scope='global'`` keeps a class's color fixed across phases by
    sampling the complete dataset class range and selecting by global ID.
    """
    ids = _ordered_unique(class_ids, name="class_ids")
    total = int(total_class_count)
    if total <= max(ids):
        raise ValueError(
            "total_class_count does not cover all visible class IDs"
        )

    normalized_name = _normalize_class_cmap_name(cmap_name)
    source = plt.get_cmap(normalized_name)
    scope = str(color_scope or "visible").strip().lower().replace("-", "_")
    if scope in {"visible", "phase", "phase_visible", "seen"}:
        positions = np.linspace(0.08, 0.92, len(ids), dtype=np.float64)
        class_colors = [source(float(position)) for position in positions]
    elif scope in {"global", "stable", "global_stable"}:
        positions = np.linspace(0.08, 0.92, total, dtype=np.float64)
        global_colors = [source(float(position)) for position in positions]
        class_colors = [global_colors[class_id] for class_id in ids]
    else:
        raise ValueError(
            "color_scope must be 'visible' or 'global', "
            f"got {color_scope!r}"
        )

    return ListedColormap(
        [background_color, *class_colors],
        name=f"{normalized_name}_{scope}_{len(ids)}",
    )


def _palette_rgb_hex(cmap: ListedColormap) -> List[str]:
    """Return rendered RGB hex values for reproducibility reports."""
    output: List[str] = []
    for color in cmap.colors:
        if isinstance(color, str):
            rgba = matplotlib.colors.to_rgba(color)
        else:
            rgba = color
        # Matplotlib rasterization converts normalized RGB values to uint8
        # by truncation. Use the same conversion so the recorded palette
        # matches the PNG pixels exactly.
        rgb = np.clip(
            255.0 * np.asarray(rgba[:3], dtype=np.float64),
            0,
            255,
        ).astype(np.uint8)
        output.append(
            "#{:02X}{:02X}{:02X}".format(
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2]),
            )
        )
    return output


def _legend_handles(
    cmap: ListedColormap,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    *,
    background_label: str,
) -> List[mpatches.Patch]:
    handles = [
        mpatches.Patch(
            facecolor=cmap.colors[0],
            edgecolor="none",
            label=background_label,
        )
    ]
    for display_id, class_id in enumerate(class_ids, start=1):
        handles.append(
            mpatches.Patch(
                facecolor=cmap.colors[display_id],
                edgecolor="none",
                label=(
                    f"{class_id}: "
                    f"{_safe_name(target_names, class_id)}"
                ),
            )
        )
    return handles


def _format_map_metric(label: str, value: float) -> str:
    if "pixels" in str(label).lower():
        return f"{label}: {int(round(float(value)))}"
    return f"{label}: {float(value):.2f}%"


def _save_map(
    class_map: np.ndarray,
    *,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    title: str,
    save_path: str,
    legend_handles: Sequence[mpatches.Patch],
    metrics: Optional[Mapping[str, float]],
    dpi: int,
) -> str:
    """Render the large single-map publication layout used in the reference."""
    values = np.asarray(class_map, dtype=np.int32)
    if values.ndim != 2:
        raise ValueError("class_map must be two-dimensional")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    legend_columns = min(4, max(1, len(legend_handles)))
    legend_rows = int(
        np.ceil(len(legend_handles) / float(legend_columns))
    )

    # For a square Indian Pines scene this produces the reference 11x14 layout.
    extra_height = 0.34 * max(0, legend_rows - 2)
    figure = plt.figure(
        figsize=(11.0, 14.0 + extra_height),
        facecolor="white",
    )

    # Keep a large, nearly edge-to-edge scene with room for title and legend.
    legend_fraction = min(0.30, 0.105 + 0.036 * legend_rows)
    axis_bottom = legend_fraction
    axis_top = 0.935
    axis = figure.add_axes(
        [0.015, axis_bottom, 0.970, axis_top - axis_bottom]
    )
    axis.imshow(
        values,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        resample=False,
        aspect="equal",
    )
    axis.set_title(
        title,
        fontsize=31,
        fontweight="bold",
        pad=22,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    if metrics:
        text = "\n".join(
            _format_map_metric(label, value)
            for label, value in metrics.items()
        )
        axis.text(
            0.985,
            0.020,
            text,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=13.5,
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.46",
                "facecolor": "#ECEDED",
                "edgecolor": "#C8C9CA",
                "linewidth": 1.2,
                "alpha": 0.97,
            },
        )

    figure.legend(
        handles=list(legend_handles),
        loc="lower center",
        ncol=legend_columns,
        bbox_to_anchor=(0.5, 0.018),
        frameon=False,
        fontsize=11.2 if legend_rows <= 3 else 9.2,
        handlelength=1.05,
        handleheight=0.90,
        columnspacing=1.55,
        labelspacing=0.64,
    )

    absolute = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    figure.savefig(
        absolute,
        dpi=int(dpi),
        facecolor="white",
        edgecolor="none",
        # Deliberately avoid bbox_inches='tight': the 11x14 publication
        # proportions and title/legend spacing must remain deterministic.
    )
    plt.close(figure)
    return absolute


def _phase_map_title(
    phase: int,
    split_token: str,
    *,
    prediction: bool,
) -> str:
    phase_name = (
        "Base Phase"
        if int(phase) == 0
        else f"Incremental Phase {int(phase)}"
    )
    output_name = "Predicted Output" if prediction else "Ground Truth"
    if split_token == "all":
        return f"{phase_name} {output_name}"
    return f"{phase_name} {split_token.title()} {output_name}"


def _phase_map_paths(
    save_dir: str,
    phase: int,
    split_token: str,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Return canonical paths and backward-compatible alias paths."""
    phase = int(phase)
    aliases: List[Tuple[str, str]] = []

    if split_token == "all" and phase == 0:
        gt_path = os.path.join(save_dir, "base_phase_ground_truth.png")
        prediction_path = os.path.join(
            save_dir,
            "base_phase_predicted_output.png",
        )
        aliases.extend(
            [
                (
                    gt_path,
                    os.path.join(
                        save_dir,
                        "phase_0_all_ground_truth.png",
                    ),
                ),
                (
                    prediction_path,
                    os.path.join(
                        save_dir,
                        "phase_0_all_predicted_gt.png",
                    ),
                ),
            ]
        )
    elif split_token == "all":
        gt_path = os.path.join(
            save_dir,
            f"phase_{phase}_ground_truth.png",
        )
        prediction_path = os.path.join(
            save_dir,
            f"phase_{phase}_predicted_output.png",
        )
        aliases.extend(
            [
                (
                    gt_path,
                    os.path.join(
                        save_dir,
                        f"phase_{phase}_all_ground_truth.png",
                    ),
                ),
                (
                    prediction_path,
                    os.path.join(
                        save_dir,
                        f"phase_{phase}_all_predicted_gt.png",
                    ),
                ),
            ]
        )
    else:
        gt_path = os.path.join(
            save_dir,
            f"phase_{phase}_{split_token}_ground_truth.png",
        )
        prediction_path = os.path.join(
            save_dir,
            f"phase_{phase}_{split_token}_predicted_output.png",
        )
        aliases.append(
            (
                prediction_path,
                os.path.join(
                    save_dir,
                    f"phase_{phase}_{split_token}_predicted_gt.png",
                ),
            )
        )
    return gt_path, prediction_path, aliases


def _copy_map_aliases(
    aliases: Sequence[Tuple[str, str]],
) -> List[str]:
    saved: List[str] = []
    for source_path, alias_path in aliases:
        if os.path.abspath(source_path) == os.path.abspath(alias_path):
            continue
        os.makedirs(
            os.path.dirname(os.path.abspath(alias_path)),
            exist_ok=True,
        )
        shutil.copyfile(source_path, alias_path)
        saved.append(alias_path)
    return saved


@torch.no_grad()
def predict_phase_grid(
    model: torch.nn.Module,
    dataset_manager: Any,
    phase: int,
    target_names: Optional[Sequence[str]],
    *,
    save_dir: str,
    device: str = "cuda:0",
    batch_size: int = 256,
    split: str = "test",
    class_cmap: str = "nipy_spectral",
    background_color: str = "#20252B",
    color_scope: str = "visible",
    save_numpy: bool = True,
    save_combined: bool = True,
    save_legacy_names: bool = True,
    return_outputs: bool = True,
    dpi: int = 300,
    **_: Any,
) -> Any:
    """Save phase maps in the supplied publication style.

    The official test map remains sparse because it contains held-out test
    coordinates only. ``split='all'`` predicts all labeled pixels belonging to
    classes seen at this phase and produces the dense qualitative output.
    No interpolation, dilation, smoothing, or fabricated filling is used.
    """
    phase = int(phase)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    requested_device = torch.device(str(device))
    if (
        requested_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested for visualization but is unavailable"
        )

    old_classes, new_classes, seen = _phase_classes(
        dataset_manager,
        phase,
    )
    if set(model.infer_seen_classes()) != set(seen):
        raise RuntimeError(
            "model rows do not match the requested visualization phase"
        )

    gt_shape = tuple(int(value) for value in dataset_manager.gt_shape)
    if len(gt_shape) != 2:
        raise RuntimeError("dataset gt_shape must be [H,W]")

    gt_display = np.zeros(gt_shape, dtype=np.int32)
    pred_display = np.zeros(gt_shape, dtype=np.int32)
    gt_semantic = np.full(gt_shape, -1, dtype=np.int32)
    pred_semantic = np.full(gt_shape, -1, dtype=np.int32)
    display_by_class = {
        class_id: display_id
        for display_id, class_id in enumerate(seen, start=1)
    }
    class_tensor = torch.tensor(
        seen,
        device=requested_device,
        dtype=torch.long,
    )
    loader = _phase_loader(
        dataset_manager,
        phase=phase,
        split=split,
        batch_size=batch_size,
    )

    previous_training = bool(model.training)
    model.eval()
    y_true_parts: List[torch.Tensor] = []
    y_pred_parts: List[torch.Tensor] = []
    visited: set[Tuple[int, int]] = set()
    try:
        for batch in loader:
            (
                processed,
                raw_patch,
                raw_center,
                labels,
                coords,
                _,
            ) = unpack_factor_eval_batch(batch)
            if coords is None:
                raise RuntimeError(
                    "prediction maps require batch coordinates"
                )
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
            output = model.forward_features(
                processed,
                raw_spectral_patch=raw_patch,
                raw_center_spectrum=raw_center,
                deterministic=True,
            )
            scored = model.compute_logits_from_features(
                output["joint_feature"],
                class_ids=seen,
                old_classes=old_classes,
                new_classes=new_classes,
                mode="factor_geometry",
                return_parts=True,
            )
            prediction = class_tensor.index_select(
                0,
                scored["energy"].argmin(dim=1),
            )
            y_true_parts.append(labels.detach().cpu())
            y_pred_parts.append(prediction.detach().cpu())

            coords_np = coords.detach().cpu().numpy().astype(np.int64)
            labels_np = labels.detach().cpu().numpy().astype(np.int64)
            predictions_np = (
                prediction.detach().cpu().numpy().astype(np.int64)
            )
            for index, (row, column) in enumerate(coords_np.tolist()):
                row, column = int(row), int(column)
                if not (
                    0 <= row < gt_shape[0]
                    and 0 <= column < gt_shape[1]
                ):
                    raise RuntimeError(
                        f"coordinate {(row, column)} outside {gt_shape}"
                    )
                if (row, column) in visited:
                    raise RuntimeError(
                        f"duplicate coordinate {(row, column)}"
                    )
                visited.add((row, column))
                target = int(labels_np[index])
                predicted = int(predictions_np[index])
                if target not in display_by_class:
                    raise RuntimeError(
                        f"target class {target} is outside seen classes {seen}"
                    )
                if predicted not in display_by_class:
                    raise RuntimeError(
                        f"predicted class {predicted} is outside seen classes {seen}"
                    )
                gt_semantic[row, column] = target
                pred_semantic[row, column] = predicted
                gt_display[row, column] = display_by_class[target]
                pred_display[row, column] = display_by_class[predicted]
    finally:
        model.train(previous_training)

    if not y_true_parts:
        raise RuntimeError(
            f"visualization loader for split={split!r} is empty"
        )
    y_true = torch.cat(y_true_parts)
    y_pred = torch.cat(y_pred_parts)
    overall = 100.0 * float(
        y_pred.eq(y_true).float().mean().item()
    )
    class_accuracy: Dict[int, float] = {}
    for class_id in seen:
        selected = y_true.eq(class_id)
        class_accuracy[class_id] = (
            100.0
            * float(
                y_pred[selected]
                .eq(y_true[selected])
                .float()
                .mean()
                .item()
            )
            if bool(selected.any()) else float("nan")
        )
    finite = [
        value
        for value in class_accuracy.values()
        if np.isfinite(value)
    ]
    average = float(np.mean(finite)) if finite else 0.0

    token = str(split).strip().lower()
    aliases = {
        "validation": "val",
        "all_labeled": "all",
        "full": "all",
    }
    token = aliases.get(token, token)
    if token not in {"train", "val", "test", "all"}:
        raise ValueError(f"unsupported map split {split!r}")
    qualitative = token == "all"

    total_class_count = int(
        getattr(
            dataset_manager,
            "total_class_count",
            (
                len(target_names)
                if target_names is not None
                else max(seen) + 1
            ),
        )
    )
    cmap = _build_cmap(
        seen,
        total_class_count=total_class_count,
        cmap_name=class_cmap,
        background_color=background_color,
        color_scope=color_scope,
    )
    norm = BoundaryNorm(
        np.arange(-0.5, len(seen) + 1.5, 1.0),
        cmap.N,
    )

    if qualitative and phase == 0:
        background_label = "Background / non-base"
    elif qualitative:
        background_label = "Background / non-seen"
    else:
        background_label = "Background / not in split"
    handles = _legend_handles(
        cmap,
        seen,
        target_names,
        background_label=background_label,
    )

    os.makedirs(save_dir, exist_ok=True)
    gt_path, prediction_path, legacy_alias_pairs = _phase_map_paths(
        save_dir,
        phase,
        token,
    )
    _save_map(
        gt_display,
        cmap=cmap,
        norm=norm,
        title=_phase_map_title(
            phase,
            token,
            prediction=False,
        ),
        save_path=gt_path,
        legend_handles=handles,
        metrics=None,
        dpi=dpi,
    )

    if qualitative:
        pixel_label = (
            "Visible base pixels"
            if phase == 0
            else "Visible seen pixels"
        )
        metric_payload = {
            "Qualitative labeled-pixel accuracy": overall,
            pixel_label: float(y_true.numel()),
        }
    else:
        metric_payload = {
            f"{token.title()} labeled-pixel accuracy": overall,
            f"Visible {token} pixels": float(y_true.numel()),
        }
    _save_map(
        pred_display,
        cmap=cmap,
        norm=norm,
        title=_phase_map_title(
            phase,
            token,
            prediction=True,
        ),
        save_path=prediction_path,
        legend_handles=handles,
        metrics=metric_payload,
        dpi=dpi,
    )

    legacy_paths: List[str] = []
    if save_legacy_names:
        legacy_paths = _copy_map_aliases(legacy_alias_pairs)

    combined_path: Optional[str] = None
    if save_combined:
        combined_path = os.path.join(
            save_dir,
            f"phase_{phase}_{token}_gt_vs_prediction.png",
        )
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(16.0, 7.2),
            facecolor="white",
        )
        for axis, values, title in (
            (axes[0], gt_display, "Ground Truth"),
            (axes[1], pred_display, "Predicted Output"),
        ):
            axis.imshow(
                values,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                resample=False,
                aspect="equal",
            )
            axis.set_title(title, fontsize=15, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        figure.suptitle(
            (
                "Base Phase"
                if phase == 0
                else f"Incremental Phase {phase}"
            ),
            fontsize=18,
            fontweight="bold",
        )
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            fontsize=8.5,
        )
        figure.tight_layout(rect=(0, 0.10, 1, 0.95))
        figure.savefig(
            combined_path,
            dpi=int(dpi),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    prefix = f"phase_{phase}_{token}"
    numpy_paths: Dict[str, str] = {}
    if save_numpy:
        arrays = {
            "ground_truth_display": gt_display,
            "predicted_output_display": pred_display,
            "ground_truth_semantic": gt_semantic,
            "predicted_output_semantic": pred_semantic,
        }
        for name, array in arrays.items():
            path = os.path.join(
                save_dir,
                f"{prefix}_{name}.npy",
            )
            np.save(path, array)
            numpy_paths[name] = path

    palette = {
        "matplotlib_colormap": _normalize_class_cmap_name(class_cmap),
        "color_scope": str(color_scope),
        "sampling_interval": [0.08, 0.92],
        "background": _palette_rgb_hex(cmap)[0],
        "class_rgb_hex": {
            int(class_id): color
            for class_id, color in zip(
                seen,
                _palette_rgb_hex(cmap)[1:],
            )
        },
    }

    output = {
        "phase": phase,
        "split": token,
        "seen_classes": seen,
        "old_classes": old_classes,
        "new_classes": new_classes,
        "ground_truth_path": gt_path,
        "prediction_path": prediction_path,
        "combined_path": combined_path,
        "legacy_alias_paths": legacy_paths,
        "numpy_paths": numpy_paths,
        "palette": palette,
        "metrics": {
            "overall_accuracy": overall,
            "average_accuracy": average,
            "per_class_accuracy": class_accuracy,
            "visible_pixels": int(y_true.numel()),
            "official_test_evidence": token == "test",
            "qualitative_only": qualitative,
        },
    }
    print(f"[Map] Phase {phase} GT: {gt_path}")
    print(f"[Map] Phase {phase} prediction: {prediction_path}")
    print(
        "[Map palette] "
        f"{palette['matplotlib_colormap']} | "
        f"scope={palette['color_scope']} | "
        f"colors={palette['class_rgb_hex']}"
    )
    return output if return_outputs else prediction_path


def predict_base_grid(
    model: torch.nn.Module,
    dataset_manager: Any,
    target_names: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    """Compatibility wrapper preserving the former base-map API."""
    return predict_phase_grid(
        model,
        dataset_manager,
        0,
        target_names,
        **kwargs,
    )


# =============================================================================
# Classification diagnostics
# =============================================================================


def save_classification_diagnostics(
    *,
    confusion_matrix: np.ndarray,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    save_path: str,
    title: str,
    dpi: int = 240,
) -> str:
    ids = _ordered_unique(class_ids, name="class_ids")
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    if matrix.shape != (len(ids), len(ids)):
        raise ValueError(
            "confusion matrix shape does not match class IDs"
        )
    names = [_safe_name(target_names, class_id) for class_id in ids]
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall > 0,
    )

    height = max(6.0, 0.48 * len(ids) + 2.5)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16.8, height),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
        facecolor="white",
    )
    figure.suptitle(title, fontsize=17, fontweight="bold")
    image = axes[0].imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )
    axes[0].set_title("Confusion matrix")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_xticks(
        range(len(names)),
        names,
        rotation=45,
        ha="right",
    )
    axes[0].set_yticks(range(len(names)), names)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(len(ids)):
        for column in range(len(ids)):
            axes[0].text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    positions = np.arange(len(ids), dtype=float)
    width = 0.24
    axes[1].barh(
        positions - width,
        100.0 * precision,
        height=width,
        label="Precision",
    )
    axes[1].barh(
        positions,
        100.0 * recall,
        height=width,
        label="Recall",
    )
    axes[1].barh(
        positions + width,
        100.0 * f1,
        height=width,
        label="F1",
    )
    axes[1].set_yticks(positions, names)
    axes[1].set_xlim(0.0, 101.0)
    axes[1].set_xlabel("Score (%)")
    axes[1].set_title("Class-wise performance")
    axes[1].grid(True, axis="x", alpha=0.22)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(frameon=False)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return save_path


# =============================================================================
# Base and incremental training dynamics
# =============================================================================


def _save_line_plot(
    *,
    rows: Sequence[Mapping[str, Any]],
    specifications: Sequence[
        Tuple[str, str, float, Optional[str]]
    ],
    title: str,
    ylabel: str,
    save_path: str,
    dpi: int,
    zero_line: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    figure, axis = plt.subplots(
        figsize=(10.4, 6.0),
        facecolor="white",
    )
    plotted = 0
    for key, label, scale, nested in specifications:
        epochs, values = _series(
            rows,
            key,
            nested=nested,
            scale=scale,
        )
        if values.size == 0:
            continue
        axis.plot(
            epochs,
            values,
            linewidth=2.0,
            label=label,
        )
        plotted += 1
    if plotted == 0:
        plt.close(figure)
        return None
    if zero_line:
        axis.axhline(0.0, linewidth=1.0, alpha=0.45)
    axis.set_title(title, fontsize=15, fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return save_path


def plot_base_training_dynamics(
    history: Mapping[str, Any],
    save_path: str,
    *,
    dpi: int = 240,
    save_separate: bool = True,
) -> Dict[str, str]:
    rows = _history_rows(history, "train")
    validation = _history_rows(history, "validation")
    if not rows:
        raise ValueError("base history has no train rows")

    # Merge validation entries into epoch-indexed base rows for plotting.
    validation_by_epoch = {
        int(row.get("epoch", index + 1)): row
        for index, row in enumerate(validation)
    }
    combined: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        epoch = int(row.get("epoch", index + 1))
        combined.append(
            {
                **dict(row),
                "validation": validation_by_epoch.get(epoch, {}),
            }
        )

    root, extension = os.path.splitext(save_path)
    extension = extension or ".png"
    output_dir = root + "_separate"
    os.makedirs(output_dir, exist_ok=True)
    plots = {
        "objective": (
            (
                ("total", "Total", 1.0, None),
                ("ce", "Temporary CE", 1.0, None),
                ("geometry", "Factor geometry", 1.0, None),
            ),
            "Base objective",
            "Loss",
            False,
            None,
        ),
        "accuracy": (
            (
                ("head_accuracy", "Head", 100.0, None),
                ("geometry_accuracy", "Query geometry", 100.0, None),
                ("accuracy", "Validation geometry", 1.0, "validation"),
                (
                    "minimum_per_class_accuracy",
                    "Validation minimum class",
                    1.0,
                    "validation",
                ),
            ),
            "Base geometry accuracy",
            "Accuracy (%)",
            False,
            (0.0, 101.0),
        ),
        "gap": (
            (
                ("mean_gap", "Training mean gap", 1.0, None),
                ("q05_gap", "Training Q05 gap", 1.0, None),
                ("mean_gap", "Validation mean gap", 1.0, "validation"),
                ("q05_gap", "Validation Q05 gap", 1.0, "validation"),
            ),
            "Factor-energy separation",
            "Rival minus target energy",
            True,
            None,
        ),
        "risk": (
            (
                (
                    "classification_violation_rate",
                    "Training classification violation",
                    100.0,
                    None,
                ),
                (
                    "margin_violation_rate",
                    "Training margin violation",
                    100.0,
                    None,
                ),
                (
                    "classification_violation_rate",
                    "Validation classification violation",
                    100.0,
                    "validation",
                ),
            ),
            "Boundary violations",
            "Rate (%)",
            False,
            (0.0, 101.0),
        ),
        "schedule": (
            (
                ("geometry_weight", "Geometry weight", 1.0, None),
                ("learning_rate", "Learning rate", 1.0, None),
                ("gradient_norm", "Gradient norm", 1.0, None),
            ),
            "Base optimization schedule",
            "Value",
            False,
            None,
        ),
    }
    outputs: Dict[str, str] = {}
    if save_separate:
        for name, (
            specs,
            title,
            ylabel,
            zero_line,
            ylim,
        ) in plots.items():
            saved = _save_line_plot(
                rows=combined,
                specifications=specs,
                title=title,
                ylabel=ylabel,
                save_path=os.path.join(
                    output_dir,
                    f"{name}{extension}",
                ),
                dpi=dpi,
                zero_line=zero_line,
                ylim=ylim,
            )
            if saved:
                outputs[name] = saved

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        facecolor="white",
    )
    for axis, (
        name,
        (
            specs,
            title,
            ylabel,
            zero_line,
            ylim,
        ),
    ) in zip(axes.flat, list(plots.items())[:4]):
        del name
        for key, label, scale, nested in specs:
            epochs, values = _series(
                combined,
                key,
                nested=nested,
                scale=scale,
            )
            if values.size:
                axis.plot(
                    epochs,
                    values,
                    linewidth=1.8,
                    label=label,
                )
        if zero_line:
            axis.axhline(0.0, linewidth=1.0, alpha=0.45)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if axis.lines:
            axis.legend(frameon=False, fontsize=7.5)
    figure.suptitle(
        "Base-phase spectral-spatial factor-geometry training",
        fontsize=17,
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    outputs["dashboard"] = save_path
    return outputs


def plot_incremental_training_dynamics(
    history: Mapping[str, Any],
    save_path: str,
    *,
    dpi: int = 240,
    save_separate: bool = True,
) -> Dict[str, str]:
    rows = _history_rows(history, "epoch_metrics")
    if not rows:
        raise ValueError("incremental history has no epoch_metrics")
    root, extension = os.path.splitext(save_path)
    extension = extension or ".png"
    output_dir = root + "_separate"
    os.makedirs(output_dir, exist_ok=True)

    plots = {
        "objective": (
            (
                ("total", "Total", 1.0, None),
                ("geometry", "Geometry", 1.0, None),
                ("coordinate", "Coordinate", 1.0, None),
                ("parameter_trust", "Parameter trust", 1.0, None),
            ),
            "Incremental objective",
            "Loss",
            False,
            None,
        ),
        "accuracy": (
            (
                ("accuracy", "Current query", 1.0, None),
                (
                    "accuracy",
                    "Current validation",
                    1.0,
                    "current_validation",
                ),
                (
                    "minimum_per_class_accuracy",
                    "Validation minimum class",
                    1.0,
                    "current_validation",
                ),
            ),
            "Current-class geometry accuracy",
            "Accuracy (%)",
            False,
            (0.0, 101.0),
        ),
        "gap": (
            (
                ("mean_gap", "Query mean gap", 1.0, None),
                ("q05_gap", "Query Q05 gap", 1.0, None),
                (
                    "mean_gap",
                    "Validation mean gap",
                    1.0,
                    "current_validation",
                ),
                (
                    "q05_gap",
                    "Validation Q05 gap",
                    1.0,
                    "current_validation",
                ),
            ),
            "Incremental factor-energy separation",
            "Rival minus target energy",
            True,
            None,
        ),
        "transport": (
            (
                (
                    "spectral_coordinate_rmse",
                    "Spectral coordinate RMSE",
                    1.0,
                    None,
                ),
                (
                    "spatial_coordinate_rmse",
                    "Spatial coordinate RMSE",
                    1.0,
                    None,
                ),
                (
                    "spectral_transport_rmse",
                    "Spectral held-out transport",
                    1.0,
                    None,
                ),
                (
                    "spatial_transport_rmse",
                    "Spatial held-out transport",
                    1.0,
                    None,
                ),
                (
                    "transport_closure_error",
                    "Transport closure",
                    1.0,
                    None,
                ),
            ),
            "Branchwise coordinate transport",
            "Normalized error",
            False,
            None,
        ),
        "risk": (
            (
                (
                    "classification_violation_rate",
                    "Classification violation",
                    100.0,
                    None,
                ),
                (
                    "margin_violation_rate",
                    "Margin violation",
                    100.0,
                    None,
                ),
                ("gradient_norm", "Gradient norm", 1.0, None),
                ("learning_rate", "Learning rate", 1.0, None),
            ),
            "Incremental risk and optimization",
            "Value",
            False,
            None,
        ),
    }

    outputs: Dict[str, str] = {}
    if save_separate:
        for name, (
            specs,
            title,
            ylabel,
            zero_line,
            ylim,
        ) in plots.items():
            saved = _save_line_plot(
                rows=rows,
                specifications=specs,
                title=title,
                ylabel=ylabel,
                save_path=os.path.join(
                    output_dir,
                    f"{name}{extension}",
                ),
                dpi=dpi,
                zero_line=zero_line,
                ylim=ylim,
            )
            if saved:
                outputs[name] = saved

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        facecolor="white",
    )
    for axis, (
        _,
        (
            specs,
            title,
            ylabel,
            zero_line,
            ylim,
        ),
    ) in zip(axes.flat, list(plots.items())[:4]):
        for key, label, scale, nested in specs:
            epochs, values = _series(
                rows,
                key,
                nested=nested,
                scale=scale,
            )
            if values.size:
                axis.plot(
                    epochs,
                    values,
                    linewidth=1.8,
                    label=label,
                )
        if zero_line:
            axis.axhline(0.0, linewidth=1.0, alpha=0.45)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if axis.lines:
            axis.legend(frameon=False, fontsize=7.4)
    figure.suptitle(
        f"Incremental phase {history.get('phase')} — "
        "transport-verified factor geometry",
        fontsize=17,
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    outputs["dashboard"] = save_path
    return outputs


def plot_phase_training_dynamics(
    history: Mapping[str, Any],
    save_path: str,
    *,
    dpi: int = 240,
    save_separate: bool = True,
) -> Dict[str, str]:
    if "epoch_metrics" in history:
        return plot_incremental_training_dynamics(
            history,
            save_path,
            dpi=dpi,
            save_separate=save_separate,
        )
    return plot_base_training_dynamics(
        history,
        save_path,
        dpi=dpi,
        save_separate=save_separate,
    )


def plot_training_history(
    history: Mapping[str, Any],
    save_path: str,
    *,
    dpi: int = 240,
    save_separate: bool = True,
) -> Dict[str, str]:
    return plot_phase_training_dynamics(
        history,
        save_path,
        dpi=dpi,
        save_separate=save_separate,
    )


# =============================================================================
# Factor GeometryBank pair diagnostics
# =============================================================================


def _covariance_from_row(
    row: Mapping[str, torch.Tensor],
    *,
    spectral_dim: int,
    spatial_dim: int,
) -> torch.Tensor:
    mean = row["mean"]
    dimension = spectral_dim + spatial_dim
    covariance = torch.zeros(
        (dimension, dimension),
        device=mean.device,
        dtype=mean.dtype,
    )
    covariance[:spectral_dim, :spectral_dim].fill_diagonal_(
        float(row["residual_var_spectral"].item())
    )
    covariance[spectral_dim:, spectral_dim:].fill_diagonal_(
        float(row["residual_var_spatial"].item())
    )
    rank = int(row["active_rank"].item())
    if rank > 0:
        loading = row["loading"][:, :rank]
        covariance = covariance + loading @ loading.transpose(0, 1)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def build_geometry_overlap_review(
    *,
    model: torch.nn.Module,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    directional_invasion_matrix: Any,
) -> Dict[str, Any]:
    ids = _ordered_unique(class_ids, name="class_ids")
    bank = model.geometry_bank
    bank.assert_valid(ids, strict=True)
    rows = {
        class_id: bank.get_class_row(class_id)
        for class_id in ids
    }
    invasion = _to_numpy(
        directional_invasion_matrix
    ).astype(np.float64)
    expected = (len(ids), len(ids))
    if invasion.shape != expected:
        raise ValueError(
            f"invasion matrix must be {expected}, got {invasion.shape}"
        )

    pair_risk = _to_numpy(
        bank.pair_risk_matrix(ids)
    ).astype(np.float64)
    spectral_distance = _to_numpy(
        bank.spectral_shape_distance_matrix(ids)
    ).astype(np.float64)
    factor_distance = _to_numpy(
        bank.factor_bhattacharyya_distance_matrix(ids)
    ).astype(np.float64)
    effective_dimension = _to_numpy(
        bank.effective_dimension(ids)
    ).astype(np.float64)

    normalized_center_distance = np.zeros(
        expected,
        dtype=np.float64,
    )
    directional_clearance = np.full(
        expected,
        np.nan,
        dtype=np.float64,
    )
    for first, class_i in enumerate(ids):
        row_i = rows[class_i]
        covariance_i = _covariance_from_row(
            row_i,
            spectral_dim=bank.spectral_dim,
            spatial_dim=bank.spatial_dim,
        )
        for second, class_j in enumerate(ids):
            if first == second:
                normalized_center_distance[first, second] = 0.0
                directional_clearance[first, second] = 0.0
                continue
            row_j = rows[class_j]
            covariance_j = _covariance_from_row(
                row_j,
                spectral_dim=bank.spectral_dim,
                spatial_dim=bank.spatial_dim,
            )
            difference = row_j["mean"] - row_i["mean"]
            distance = difference.norm().clamp_min(1e-12)
            direction = difference / distance
            radius_i = torch.sqrt(
                direction @ covariance_i @ direction
            ).clamp_min(1e-12)
            radius_j = torch.sqrt(
                direction @ covariance_j @ direction
            ).clamp_min(1e-12)
            directional_clearance[first, second] = float(
                (distance - 2.0 * (radius_i + radius_j)).item()
            )
            scale = (
                torch.sqrt(
                    torch.trace(covariance_i)
                    / max(covariance_i.size(0), 1)
                )
                + torch.sqrt(
                    torch.trace(covariance_j)
                    / max(covariance_j.size(0), 1)
                )
            )
            normalized_center_distance[first, second] = float(
                (distance / scale.clamp_min(1e-12)).item()
            )

    pairs: List[Dict[str, Any]] = []
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            class_i, class_j = ids[first], ids[second]
            risk = float(pair_risk[first, second])
            maximum_invasion = max(
                float(invasion[first, second]),
                float(invasion[second, first]),
            )
            clearance = min(
                float(directional_clearance[first, second]),
                float(directional_clearance[second, first]),
            )
            intersects = bool(np.isfinite(clearance) and clearance <= 0.0)
            if risk >= 0.50 and maximum_invasion >= 0.05:
                status = "HIGH"
            elif (
                risk >= 0.25
                or maximum_invasion >= 0.05
                or intersects
            ):
                status = "MODERATE"
            else:
                status = "LOW"
            reasons: List[str] = []
            if risk >= 0.50:
                reasons.append("high spectral-geometry pair risk")
            elif risk >= 0.25:
                reasons.append("moderate spectral-geometry pair risk")
            if maximum_invasion >= 0.05:
                reasons.append("empirical directional invasion >= 5%")
            if intersects:
                reasons.append("two-sigma directional radii intersect")
            if not reasons:
                reasons.append("low observed pair risk")
            pairs.append(
                {
                    "class_i": class_i,
                    "class_j": class_j,
                    "class_i_name": _safe_name(target_names, class_i),
                    "class_j_name": _safe_name(target_names, class_j),
                    "pair_risk": risk,
                    "spectral_shape_distance": float(
                        spectral_distance[first, second]
                    ),
                    "factor_bhattacharyya_distance": float(
                        factor_distance[first, second]
                    ),
                    "normalized_center_distance": float(
                        normalized_center_distance[first, second]
                    ),
                    "directional_clearance": clearance,
                    "invasion_i_to_j": float(invasion[first, second]),
                    "invasion_j_to_i": float(invasion[second, first]),
                    "maximum_bidirectional_invasion": maximum_invasion,
                    "ellipsoids_intersect": intersects,
                    "status": status,
                    "reason": "; ".join(reasons),
                }
            )
    priority = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    pairs.sort(
        key=lambda row: (
            priority[row["status"]],
            -float(row["pair_risk"]),
            -float(row["maximum_bidirectional_invasion"]),
        )
    )
    off_diagonal = ~np.eye(len(ids), dtype=bool)
    finite_clearance = directional_clearance[
        off_diagonal & np.isfinite(directional_clearance)
    ]
    return {
        "definition": (
            "Pair risk combines raw ordered spectral-shape distance, "
            "factor-distribution distance, and row reliability. "
            "Directional invasion is measured on real evaluation samples."
        ),
        "class_ids": ids,
        "high_risk_pair_count": sum(
            row["status"] == "HIGH" for row in pairs
        ),
        "moderate_risk_pair_count": sum(
            row["status"] == "MODERATE" for row in pairs
        ),
        "maximum_pair_risk": (
            float(pair_risk[off_diagonal].max())
            if bool(off_diagonal.any()) else 0.0
        ),
        "maximum_directional_invasion": (
            float(invasion[off_diagonal].max())
            if bool(off_diagonal.any()) else 0.0
        ),
        "minimum_directional_clearance": (
            float(finite_clearance.min())
            if finite_clearance.size else 0.0
        ),
        "pair_risk_matrix": pair_risk,
        "spectral_shape_distance_matrix": spectral_distance,
        "factor_bhattacharyya_distance_matrix": factor_distance,
        "normalized_center_distance_matrix": normalized_center_distance,
        "directional_clearance_matrix": directional_clearance,
        "directional_invasion_matrix": invasion,
        "effective_dimension": effective_dimension,
        "pair_geometry": pairs,
    }


def save_geometry_pair_diagnostics(
    *,
    review: Mapping[str, Any],
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    save_path: str,
    dpi: int = 240,
) -> str:
    ids = _ordered_unique(class_ids, name="class_ids")
    names = [_safe_name(target_names, class_id) for class_id in ids]
    matrices = (
        (
            100.0
            * np.asarray(
                review["directional_invasion_matrix"],
                dtype=float,
            ),
            "Directional invasion (%)",
        ),
        (
            np.asarray(review["pair_risk_matrix"], dtype=float),
            "Spectral-geometry pair risk",
        ),
        (
            np.asarray(
                review["factor_bhattacharyya_distance_matrix"],
                dtype=float,
            ),
            "Factor Bhattacharyya distance",
        ),
        (
            np.asarray(
                review["spectral_shape_distance_matrix"],
                dtype=float,
            ),
            "Ordered spectral-shape distance",
        ),
        (
            np.asarray(
                review["directional_clearance_matrix"],
                dtype=float,
            ),
            "Two-sigma directional clearance",
        ),
    )
    effective_dimension = np.asarray(
        review["effective_dimension"],
        dtype=float,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(20.0, 12.0),
        facecolor="white",
    )
    figure.suptitle(
        "Spectral-spatial factor GeometryBank diagnostics",
        fontsize=17,
        fontweight="bold",
    )
    for axis, (matrix, title) in zip(axes.flat[:5], matrices):
        image = axis.imshow(matrix, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks(
            range(len(ids)),
            names,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(ids)), names)
        for row in range(len(ids)):
            for column in range(len(ids)):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                    )
        figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

    axes[1, 2].barh(
        np.arange(len(ids)),
        effective_dimension,
    )
    axes[1, 2].set_yticks(np.arange(len(ids)), names)
    axes[1, 2].set_xlabel("Effective covariance dimension")
    axes[1, 2].set_title("Per-class effective dimension")
    axes[1, 2].grid(True, axis="x", alpha=0.22)
    axes[1, 2].spines["top"].set_visible(False)
    axes[1, 2].spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return save_path


def save_spectral_spatial_geometry_diagnostics(
    *,
    model: torch.nn.Module,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    save_path: str,
    dpi: int = 240,
) -> str:
    ids = _ordered_unique(class_ids, name="class_ids")
    bank = model.geometry_bank
    bank.assert_valid(ids, strict=True)
    names = [_safe_name(target_names, class_id) for class_id in ids]

    ranks: List[int] = []
    spectral_variance: List[float] = []
    spatial_variance: List[float] = []
    spectral_reliability: List[float] = []
    geometry_reliability: List[float] = []
    reconstruction: List[float] = []
    for class_id in ids:
        row = bank.get_class_row(class_id)
        ranks.append(int(row["active_rank"].item()))
        spectral_variance.append(
            float(row["residual_var_spectral"].item())
        )
        spatial_variance.append(
            float(row["residual_var_spatial"].item())
        )
        spectral_reliability.append(
            float(row["spectral_shape_reliability"].item())
        )
        geometry_reliability.append(
            float(row["geometry_reliability"].item())
        )
        reconstruction.append(
            float(row["reconstruction_error"].item())
        )

    positions = np.arange(len(ids), dtype=float)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.8, 10.5),
        facecolor="white",
    )
    figure.suptitle(
        "Spectral-spatial factor geometry audit",
        fontsize=17,
        fontweight="bold",
    )

    axes[0, 0].barh(positions, ranks)
    axes[0, 0].set_yticks(positions, names)
    axes[0, 0].set_xlabel("Active factor rank")
    axes[0, 0].set_title("Low-rank correlated geometry")

    width = 0.36
    axes[0, 1].barh(
        positions - width / 2,
        spectral_variance,
        height=width,
        label="Spectral residual",
    )
    axes[0, 1].barh(
        positions + width / 2,
        spatial_variance,
        height=width,
        label="Spatial residual",
    )
    axes[0, 1].set_yticks(positions, names)
    axes[0, 1].set_xlabel("Residual variance")
    axes[0, 1].set_title("Branch residual uncertainty")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].barh(
        positions - width / 2,
        100.0 * np.asarray(spectral_reliability),
        height=width,
        label="Spectral-shape reliability",
    )
    axes[1, 0].barh(
        positions + width / 2,
        100.0 * np.asarray(geometry_reliability),
        height=width,
        label="Geometry reliability",
    )
    axes[1, 0].set_yticks(positions, names)
    axes[1, 0].set_xlim(0.0, 100.0)
    axes[1, 0].set_xlabel("Reliability (%)")
    axes[1, 0].set_title("Aggregate-row reliability")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].barh(positions, reconstruction)
    axes[1, 1].set_yticks(positions, names)
    axes[1, 1].set_xlabel("Relative covariance reconstruction error")
    axes[1, 1].set_title("Factor approximation quality")

    for axis in axes.flat:
        axis.grid(True, axis="x", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return save_path


# Compatibility alias. The output is intentionally not a second classifier.
save_spectral_conditioning_diagnostics = (
    save_spectral_spatial_geometry_diagnostics
)


def save_transport_diagnostics(
    *,
    history: Mapping[str, Any],
    save_path: str,
    dpi: int = 240,
) -> Optional[str]:
    report = history.get(
        "best_candidate_report",
        history.get("candidate_report"),
    )
    if not isinstance(report, Mapping):
        return None
    transform = report.get(
        "transform_report",
        report,
    )
    if not isinstance(transform, Mapping):
        return None
    spectral = transform.get("spectral", {})
    spatial = transform.get("spatial", {})
    if not isinstance(spectral, Mapping) or not isinstance(spatial, Mapping):
        return None

    labels = ["Spectral", "Spatial"]
    selected_level = [
        float(spectral.get("selected_level", 0)),
        float(spatial.get("selected_level", 0)),
    ]
    normalized_rmse = [
        float(spectral.get("selected_normalized_rmse", np.nan)),
        float(spatial.get("selected_normalized_rmse", np.nan)),
    ]
    rank = [
        float(spectral.get("support_effective_rank", 0)),
        float(spatial.get("support_effective_rank", 0)),
    ]
    support = [
        float(spectral.get("support_samples", 0)),
        float(spatial.get("support_samples", 0)),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.5),
        facecolor="white",
    )
    figure.suptitle(
        f"Phase {history.get('phase')} branch-transport audit",
        fontsize=17,
        fontweight="bold",
    )
    for axis, values, title, ylabel in (
        (
            axes[0, 0],
            selected_level,
            "Selected transport complexity",
            "Level (0/1/2)",
        ),
        (
            axes[0, 1],
            normalized_rmse,
            "Held-out registration error",
            "Normalized RMSE",
        ),
        (
            axes[1, 0],
            rank,
            "Support effective rank",
            "Rank",
        ),
        (
            axes[1, 1],
            support,
            "Transform support size",
            "Samples",
        ),
    ):
        axis.bar(labels, values)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return save_path


# =============================================================================
# Across-phase visualization
# =============================================================================


def plot_necil_phase_summary(
    evaluator: NECILEvaluator,
    save_path: str,
    *,
    dpi: int = 240,
) -> str:
    rows = evaluator.phase_table()
    if not rows:
        raise ValueError("NECILEvaluator has no phase results")
    phases = np.asarray(
        [int(row["phase"]) for row in rows],
        dtype=np.int64,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        facecolor="white",
    )
    axes[0, 0].plot(
        phases,
        [row["OA"] for row in rows],
        marker="o",
        label="OA",
    )
    axes[0, 0].plot(
        phases,
        [row["AA"] for row in rows],
        marker="o",
        label="AA",
    )
    axes[0, 0].plot(
        phases,
        [row["F1"] for row in rows],
        marker="o",
        label="Macro-F1",
    )
    axes[0, 0].set_title("Cumulative classification")
    axes[0, 0].set_ylabel("Score (%)")

    axes[0, 1].plot(
        phases,
        [row["Old"] for row in rows],
        marker="o",
        label="Old",
    )
    axes[0, 1].plot(
        phases,
        [row["New"] for row in rows],
        marker="o",
        label="New",
    )
    axes[0, 1].plot(
        phases,
        [row["H"] for row in rows],
        marker="o",
        label="Harmonic mean",
    )
    axes[0, 1].set_title("Stability–plasticity balance")
    axes[0, 1].set_ylabel("Accuracy (%)")

    axes[1, 0].plot(
        phases,
        [row["OldToNew"] for row in rows],
        marker="o",
        label="Old → new",
    )
    axes[1, 0].plot(
        phases,
        [row["NewToOld"] for row in rows],
        marker="o",
        label="New → old",
    )
    axes[1, 0].plot(
        phases,
        [row["GeometryViolation"] for row in rows],
        marker="o",
        label="Geometry violation",
    )
    axes[1, 0].set_title("Classifier and geometry invasion")
    axes[1, 0].set_ylabel("Rate (%)")

    per_class = evaluator.per_class_summary()
    class_ids = sorted(per_class)
    forgetting = [
        per_class[class_id]["forgetting"]
        for class_id in class_ids
    ]
    axes[1, 1].bar(
        [str(class_id) for class_id in class_ids],
        forgetting,
    )
    axes[1, 1].set_title("Final per-class forgetting")
    axes[1, 1].set_xlabel("Global class ID")
    axes[1, 1].set_ylabel("Forgetting (percentage points)")

    for axis in axes.flat:
        if axis is not axes[1, 1]:
            axis.set_xlabel("Phase")
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if axis.lines:
            axis.legend(frameon=False, fontsize=8.5)
    figure.suptitle(
        "NECIL phase-wise evaluation",
        fontsize=17,
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        save_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return save_path
























# """Base-phase visualizations for evolving spectral-conditioned HSI geometry.

# Active contract
# ---------------
# * Deployed score: p(s|c) p(z|s,c).
# * The neural encoder receives a processed/PCA patch.
# * The geometry branch receives the aligned ordered physical-band patch.
# * Display value 0 is reserved for background/not evaluated.
# * Held-out test maps are official evidence; all-labeled maps are qualitative.
# """

# from __future__ import annotations

# import os
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import matplotlib
# matplotlib.use("Agg", force=True)

# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from matplotlib.colors import BoundaryNorm, ListedColormap


# JOINT_FACTORIZATION = "p(s|c)p(z|s,c)"


# def _ordered_unique(values: Iterable[int]) -> List[int]:
#     result: List[int] = []
#     seen: set[int] = set()
#     for value in values:
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError(f"class IDs must be non-negative, got {class_id}")
#         if class_id in seen:
#             raise ValueError(f"duplicate class ID {class_id}")
#         seen.add(class_id)
#         result.append(class_id)
#     if not result:
#         raise ValueError("class ID list is empty")
#     return result


# def _safe_name(target_names: Optional[Sequence[str]], class_id: int) -> str:
#     class_id = int(class_id)
#     if target_names is not None and 0 <= class_id < len(target_names):
#         value = str(target_names[class_id]).strip()
#         if value:
#             return value
#     return f"Class {class_id}"


# def _base_classes(dataset_manager: Any) -> List[int]:
#     getter = getattr(dataset_manager, "get_seen_classes", None)
#     if callable(getter):
#         return _ordered_unique(getter(0))
#     phase_to_classes = getattr(dataset_manager, "phase_to_classes", None)
#     if phase_to_classes is None:
#         raise AttributeError(
#             "dataset manager must expose get_seen_classes(0) or phase_to_classes"
#         )
#     return _ordered_unique(phase_to_classes[0])


# def _base_loader(
#     dataset_manager: Any,
#     *,
#     split: str,
#     batch_size: int,
# ) -> Any:
#     cumulative = getattr(dataset_manager, "get_cumulative_dataloader", None)
#     if callable(cumulative):
#         try:
#             return cumulative(
#                 0,
#                 split=str(split),
#                 batch_size=int(batch_size),
#                 shuffle=False,
#             )
#         except TypeError:
#             return cumulative(
#                 phase=0,
#                 split=str(split),
#                 batch_size=int(batch_size),
#                 shuffle=False,
#             )
#     phase_loader = getattr(dataset_manager, "get_phase_dataloader", None)
#     if callable(phase_loader):
#         return phase_loader(
#             0,
#             split=str(split),
#             batch_size=int(batch_size),
#             shuffle=False,
#         )
#     raise RuntimeError("dataset manager exposes no base evaluation loader")


# def _batch_value(batch: Mapping[str, Any], names: Sequence[str]) -> Any:
#     for name in names:
#         if name in batch and batch[name] is not None:
#             return batch[name]
#     return None


# def _unpack_hsi_batch(
#     batch: Any,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Return processed patch, physical patch, global label, and coordinate."""
#     if not isinstance(batch, Mapping):
#         raise RuntimeError("base visualization requires mapping batches")
#     processed = _batch_value(batch, ("image", "patch", "patches", "x", "input"))
#     raw = _batch_value(
#         batch,
#         (
#             "raw_spectral_patch",
#             "raw_spectral_patches",
#             "physical_spectral_patch",
#             "physical_patch",
#             "raw_patch",
#         ),
#     )
#     labels = _batch_value(batch, ("label", "labels", "target", "y"))
#     coords = _batch_value(batch, ("coord", "coords", "coordinate"))
#     missing = [
#         name
#         for name, value in (
#             ("image", processed),
#             ("raw_spectral_patch", raw),
#             ("label", labels),
#             ("coord", coords),
#         )
#         if value is None
#     ]
#     if missing:
#         raise RuntimeError(f"visualization batch is missing {missing}")

#     processed_t = torch.as_tensor(processed)
#     raw_t = torch.as_tensor(raw)
#     labels_t = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
#     coords_t = torch.as_tensor(coords, dtype=torch.long)
#     if processed_t.ndim != 4 or raw_t.ndim != 4:
#         raise RuntimeError("processed and raw patches must be [B,C,H,W]")
#     batch_size = int(processed_t.shape[0])
#     if raw_t.shape[0] != batch_size or raw_t.shape[2:] != processed_t.shape[2:]:
#         raise RuntimeError("processed and raw patches are misaligned")
#     if labels_t.numel() != batch_size:
#         raise RuntimeError("patch/label batch mismatch")
#     if tuple(coords_t.shape) != (batch_size, 2):
#         raise RuntimeError(f"coordinates must be [B,2], got {tuple(coords_t.shape)}")
#     if not torch.isfinite(processed_t.float()).all():
#         raise RuntimeError("processed patches contain NaN/Inf")
#     if not torch.isfinite(raw_t.float()).all():
#         raise RuntimeError("raw spectral patches contain NaN/Inf")
#     return processed_t, raw_t, labels_t, coords_t


# @torch.no_grad()
# def _joint_scores(
#     model: torch.nn.Module,
#     processed: torch.Tensor,
#     raw_spectral_patch: torch.Tensor,
#     base_classes: Sequence[int],
# ) -> Mapping[str, torch.Tensor]:
#     output = model.forward_features(
#         processed,
#         raw_spectral_patch=raw_spectral_patch,
#         deterministic=True,
#     )
#     features = output.get("geometry_features")
#     spectra = output.get("raw_center_spectra")
#     if not torch.is_tensor(features) or features.ndim != 2:
#         raise RuntimeError("forward_features returned no geometry_features [B,D]")
#     if not torch.is_tensor(spectra) or spectra.ndim != 2:
#         raise RuntimeError("forward_features returned no raw_center_spectra [B,Bands]")

#     scored = model.compute_logits_from_features(
#         features,
#         seen_classes=[int(value) for value in base_classes],
#         raw_spectra=spectra,
#         mode="spectral_conditioned_joint",
#         return_energy=True,
#         return_parts=True,
#         return_diagnostics=True,
#     )
#     if not isinstance(scored, Mapping):
#         raise RuntimeError("joint geometry scoring must return a mapping")
#     if scored.get("joint_factorization") != JOINT_FACTORIZATION:
#         raise RuntimeError(
#             f"classifier factorization={scored.get('joint_factorization')!r}, "
#             f"expected {JOINT_FACTORIZATION!r}"
#         )
#     required = (
#         "joint_energy",
#         "weighted_feature_energy",
#         "weighted_spectral_energy",
#     )
#     for name in required:
#         value = scored.get(name)
#         if not torch.is_tensor(value):
#             raise RuntimeError(f"joint scoring returned no {name}")
#         expected = (processed.size(0), len(base_classes))
#         if tuple(value.shape) != expected:
#             raise RuntimeError(
#                 f"{name} shape mismatch: expected {expected}, got {tuple(value.shape)}"
#             )
#         if not torch.isfinite(value).all():
#             raise RuntimeError(f"{name} contains NaN/Inf")
#     return {
#         **dict(scored),
#         "features": features,
#         "raw_center_spectra": spectra,
#     }


# # -----------------------------------------------------------------------------
# # Publication maps
# # -----------------------------------------------------------------------------


# def _build_cmap(
#     class_count: int,
#     *,
#     cmap_name: str = "nipy_spectral",
#     background_color: str = "#20252B",
#     class_ids: Optional[Sequence[int]] = None,
#     total_class_count: Optional[int] = None,
# ) -> ListedColormap:
#     class_count = int(class_count)
#     if class_count <= 0:
#         raise ValueError("class_count must be positive")
#     ids = list(range(class_count)) if class_ids is None else [int(v) for v in class_ids]
#     if len(ids) != class_count or len(set(ids)) != class_count:
#         raise ValueError("class_ids must contain class_count unique IDs")
#     full_count = int(
#         total_class_count
#         if total_class_count is not None
#         else max(class_count, max(ids, default=-1) + 1)
#     )
#     if full_count <= max(ids, default=-1):
#         raise ValueError("total_class_count does not cover class_ids")
#     base = plt.get_cmap(str(cmap_name or "nipy_spectral"))
#     qualitative = {
#         "tab10", "tab20", "tab20b", "tab20c", "set1", "set2", "set3", "paired"
#     }
#     if str(cmap_name).lower() in qualitative:
#         full_colors = [base(index % int(base.N)) for index in range(full_count)]
#     else:
#         full_colors = [base(float(v)) for v in np.linspace(0.08, 0.92, full_count)]
#     return ListedColormap([background_color, *[full_colors[class_id] for class_id in ids]])


# def _legend_handles(
#     cmap: ListedColormap,
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     *,
#     background_label: str,
# ) -> List[mpatches.Patch]:
#     handles = [
#         mpatches.Patch(
#             facecolor=cmap.colors[0],
#             edgecolor="none",
#             label=str(background_label),
#         )
#     ]
#     for display_id, class_id in enumerate(class_ids, start=1):
#         handles.append(
#             mpatches.Patch(
#                 facecolor=cmap.colors[display_id],
#                 edgecolor="none",
#                 label=f"{class_id}: {_safe_name(target_names, class_id)}",
#             )
#         )
#     return handles


# def _save_map(
#     class_map: np.ndarray,
#     *,
#     cmap: ListedColormap,
#     norm: BoundaryNorm,
#     title: str,
#     save_path: str,
#     legend_handles: Sequence[mpatches.Patch],
#     metrics: Optional[Mapping[str, float]] = None,
#     dpi: int = 300,
# ) -> str:
#     class_map = np.asarray(class_map, dtype=np.int32)
#     if class_map.ndim != 2:
#         raise ValueError("class_map must be a 2-D array")
#     height, width = class_map.shape
#     aspect = width / max(height, 1)
#     figure_width = min(max(7.5, 7.2 * aspect), 12.0)
#     legend_columns = min(4, max(1, len(legend_handles)))
#     legend_rows = int(np.ceil(len(legend_handles) / float(legend_columns)))
#     figure_height = 8.4 + 0.30 * max(0, legend_rows - 2)
#     legend_bottom = min(0.36, 0.10 + 0.038 * legend_rows)

#     fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="white")
#     ax.imshow(
#         class_map,
#         cmap=cmap,
#         norm=norm,
#         interpolation="nearest",
#         resample=False,
#         aspect="equal",
#     )
#     ax.set_title(str(title), fontsize=21, fontweight="bold", pad=16)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     if metrics:
#         lines = [
#             f"{key}: {int(round(float(value)))}"
#             if str(key).lower().startswith("visible")
#             else f"{key}: {float(value):.2f}%"
#             for key, value in metrics.items()
#         ]
#         ax.text(
#             0.985,
#             0.02,
#             "\n".join(lines),
#             transform=ax.transAxes,
#             ha="right",
#             va="bottom",
#             fontsize=10.5,
#             color="#111111",
#             bbox={
#                 "boxstyle": "round,pad=0.45",
#                 "facecolor": "white",
#                 "edgecolor": "#D0D0D0",
#                 "alpha": 0.92,
#             },
#         )

#     fig.legend(
#         handles=list(legend_handles),
#         loc="lower center",
#         ncol=legend_columns,
#         bbox_to_anchor=(0.5, 0.012),
#         frameon=False,
#         fontsize=8.8 if legend_rows >= 4 else 9.2,
#         handlelength=1.1,
#         columnspacing=1.25,
#         labelspacing=0.55,
#     )
#     os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#     plt.subplots_adjust(left=0.025, right=0.975, top=0.92, bottom=legend_bottom)
#     fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     return save_path


# @torch.no_grad()
# def predict_base_grid(
#     model: torch.nn.Module,
#     dataset_manager: Any,
#     target_names: Optional[Sequence[str]],
#     *,
#     save_dir: str,
#     device: str = "cuda:0",
#     batch_size: int = 256,
#     split: str = "test",
#     class_cmap: str = "nipy_spectral",
#     background_color: str = "#20252B",
#     save_numpy: bool = True,
#     return_outputs: bool = True,
#     dpi: int = 300,
# ) -> Any:
#     if int(batch_size) <= 0:
#         raise ValueError("batch_size must be positive")
#     requested_device = torch.device(str(device))
#     if requested_device.type == "cuda" and not torch.cuda.is_available():
#         raise RuntimeError("CUDA was requested for visualization but is unavailable")

#     class_ids = _base_classes(dataset_manager)
#     class_set = set(class_ids)
#     class_tensor = torch.tensor(class_ids, device=requested_device, dtype=torch.long)
#     display_by_class = {
#         class_id: display_id for display_id, class_id in enumerate(class_ids, start=1)
#     }
#     gt_shape = tuple(int(v) for v in getattr(dataset_manager, "gt_shape"))
#     if len(gt_shape) != 2:
#         raise RuntimeError(f"gt_shape must be [H,W], got {gt_shape}")

#     gt_display = np.zeros(gt_shape, dtype=np.int32)
#     pred_display = np.zeros(gt_shape, dtype=np.int32)
#     gt_semantic = np.full(gt_shape, -1, dtype=np.int32)
#     pred_semantic = np.full(gt_shape, -1, dtype=np.int32)
#     loader = _base_loader(
#         dataset_manager,
#         split=str(split),
#         batch_size=int(batch_size),
#     )

#     previous_training = bool(model.training)
#     model.eval()
#     labels_all: List[torch.Tensor] = []
#     predictions_all: List[torch.Tensor] = []
#     visited: set[Tuple[int, int]] = set()
#     try:
#         for raw_batch in loader:
#             processed, raw, labels, coords = _unpack_hsi_batch(raw_batch)
#             processed = processed.to(requested_device, dtype=torch.float32, non_blocking=True)
#             raw = raw.to(requested_device, dtype=torch.float32, non_blocking=True)
#             labels = labels.to(requested_device, dtype=torch.long)
#             unknown = sorted(set(int(v) for v in labels.detach().cpu().tolist()) - class_set)
#             if unknown:
#                 raise RuntimeError(f"visualization loader contains non-base classes {unknown}")
#             scored = _joint_scores(model, processed, raw, class_ids)
#             prediction_local = scored["joint_energy"].argmin(dim=1)
#             prediction_global = class_tensor.index_select(0, prediction_local)
#             labels_all.append(labels.detach().cpu())
#             predictions_all.append(prediction_global.detach().cpu())

#             coords_np = coords.detach().cpu().numpy().astype(np.int64)
#             labels_np = labels.detach().cpu().numpy().astype(np.int64)
#             pred_np = prediction_global.detach().cpu().numpy().astype(np.int64)
#             for index, (row, column) in enumerate(coords_np):
#                 row, column = int(row), int(column)
#                 if not (0 <= row < gt_shape[0] and 0 <= column < gt_shape[1]):
#                     raise RuntimeError(f"coordinate {(row, column)} is outside {gt_shape}")
#                 if (row, column) in visited:
#                     raise RuntimeError(f"duplicate coordinate {(row, column)}")
#                 visited.add((row, column))
#                 true_class = int(labels_np[index])
#                 pred_class = int(pred_np[index])
#                 gt_semantic[row, column] = true_class
#                 pred_semantic[row, column] = pred_class
#                 gt_display[row, column] = display_by_class[true_class]
#                 pred_display[row, column] = display_by_class[pred_class]
#     finally:
#         model.train(previous_training)

#     if not labels_all:
#         raise RuntimeError(f"base visualization loader for split={split!r} is empty")
#     y_true = torch.cat(labels_all)
#     y_pred = torch.cat(predictions_all)
#     overall_accuracy = 100.0 * float(y_pred.eq(y_true).float().mean().item())
#     per_class_accuracy: Dict[int, float] = {}
#     for class_id in class_ids:
#         mask = y_true.eq(class_id)
#         per_class_accuracy[class_id] = (
#             100.0 * float(y_pred[mask].eq(y_true[mask]).float().mean().item())
#             if bool(mask.any().item()) else float("nan")
#         )
#     finite = [v for v in per_class_accuracy.values() if np.isfinite(v)]
#     average_accuracy = float(np.mean(finite)) if finite else 0.0

#     split_token = str(split).strip().lower()
#     aliases = {"all_labeled": "all", "full": "all", "validation": "val"}
#     split_token = aliases.get(split_token, split_token)
#     if split_token not in {"train", "val", "test", "all"}:
#         raise ValueError(f"unsupported visualization split {split!r}")
#     qualitative = split_token == "all"

#     total_class_count = max(
#         max(class_ids, default=-1) + 1,
#         len(target_names) if target_names is not None else 0,
#     )
#     cmap = _build_cmap(
#         len(class_ids),
#         cmap_name=class_cmap,
#         background_color=background_color,
#         class_ids=class_ids,
#         total_class_count=total_class_count,
#     )
#     norm = BoundaryNorm(np.arange(-0.5, len(class_ids) + 1.5, 1.0), cmap.N)
#     handles = _legend_handles(
#         cmap,
#         class_ids,
#         target_names,
#         background_label=(
#             "Background / non-base classes" if qualitative else "Background / not in split"
#         ),
#     )
#     os.makedirs(save_dir, exist_ok=True)

#     if qualitative:
#         gt_path = os.path.join(save_dir, "base_all_ground_truth.png")
#         pred_path = os.path.join(save_dir, "base_all_predicted_gt.png")
#         gt_title = "Base Phase Ground Truth — All Labeled Base Pixels"
#         pred_title = "Base Phase Predicted GT — All Labeled Base Pixels"
#         metric_payload = {
#             "Qualitative accuracy": overall_accuracy,
#             "Visible base pixels": float(y_true.numel()),
#         }
#         prefix = "base_all"
#     else:
#         gt_path = os.path.join(save_dir, f"base_{split_token}_ground_truth.png")
#         pred_path = os.path.join(save_dir, f"base_{split_token}_predicted_gt.png")
#         gt_title = f"Base Phase Ground Truth — {split_token.title()} Split"
#         pred_title = f"Base Phase Predicted GT — {split_token.title()} Split"
#         metric_payload = {"OA": overall_accuracy, "AA": average_accuracy}
#         prefix = f"base_{split_token}"

#     _save_map(
#         gt_display,
#         cmap=cmap,
#         norm=norm,
#         title=gt_title,
#         save_path=gt_path,
#         legend_handles=handles,
#         metrics=None,
#         dpi=dpi,
#     )
#     _save_map(
#         pred_display,
#         cmap=cmap,
#         norm=norm,
#         title=pred_title,
#         save_path=pred_path,
#         legend_handles=handles,
#         metrics=metric_payload,
#         dpi=dpi,
#     )

#     numpy_paths: Dict[str, str] = {}
#     if save_numpy:
#         arrays = {
#             "ground_truth_display": gt_display,
#             "predicted_gt_display": pred_display,
#             "ground_truth_semantic": gt_semantic,
#             "predicted_gt_semantic": pred_semantic,
#         }
#         for name, array in arrays.items():
#             path = os.path.join(save_dir, f"{prefix}_{name}.npy")
#             np.save(path, array)
#             numpy_paths[name] = path

#     outputs: Dict[str, Any] = {
#         "ground_truth_path": gt_path,
#         "prediction_path": pred_path,
#         "ground_truth_display_map": gt_display,
#         "prediction_display_map": pred_display,
#         "ground_truth_semantic_map": gt_semantic,
#         "prediction_semantic_map": pred_semantic,
#         "numpy_paths": numpy_paths,
#         "metrics": {
#             "overall_accuracy": overall_accuracy,
#             "average_accuracy": average_accuracy,
#             "per_class_accuracy": per_class_accuracy,
#             "visible_pixels": int(y_true.numel()),
#             "base_classes": list(class_ids),
#             "split": split_token,
#             "official_test_evidence": split_token == "test",
#             "qualitative_only": qualitative,
#         },
#     }
#     print(f"[Map] Ground truth saved separately: {gt_path}")
#     print(f"[Map] Predicted GT saved separately: {pred_path}")
#     return outputs if return_outputs else pred_path


# def predict_phase_grid(
#     model: torch.nn.Module,
#     dataset_manager: Any,
#     phase: int,
#     target_names: Optional[Sequence[str]],
#     save_dir: str = "./results/phase_0/figures",
#     device: str = "cuda:0",
#     patch_size: Optional[int] = None,
#     chunk_size: int = 256,
#     split: str = "test",
#     classifier_mode: Optional[str] = None,
#     semantic_mode: Optional[str] = None,
#     save_numpy: bool = True,
#     return_outputs: bool = False,
#     class_cmap: str = "nipy_spectral",
#     background_color: str = "#20252B",
#     dpi: int = 300,
#     **_: Any,
# ) -> Any:
#     del patch_size
#     if int(phase) != 0:
#         raise RuntimeError("this visualization module is base-phase-only")
#     normalized_mode = str(classifier_mode or "spectral_conditioned_joint").strip().lower().replace("-", "_")
#     if normalized_mode not in {
#         "spectral_conditioned_joint", "geometry", "geometry_only", "joint", "joint_energy"
#     }:
#         raise RuntimeError("visualization supports only spectral_conditioned_joint")
#     if str(semantic_mode or "identity").strip().lower() != "identity":
#         raise RuntimeError("semantic_mode must be identity")
#     return predict_base_grid(
#         model,
#         dataset_manager,
#         target_names,
#         save_dir=save_dir,
#         device=device,
#         batch_size=chunk_size,
#         split=split,
#         class_cmap=class_cmap,
#         background_color=background_color,
#         save_numpy=save_numpy,
#         return_outputs=return_outputs,
#         dpi=dpi,
#     )


# # -----------------------------------------------------------------------------
# # Training plots
# # -----------------------------------------------------------------------------


# def _history_rows(history: Mapping[str, Any], name: str) -> List[Mapping[str, Any]]:
#     values = history.get(name, [])
#     if not isinstance(values, (list, tuple)):
#         return []
#     return [row for row in values if isinstance(row, Mapping)]


# def _row_series(
#     rows: Sequence[Mapping[str, Any]],
#     key: str,
#     *,
#     scale: float = 1.0,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     epochs: List[int] = []
#     values: List[float] = []
#     for row in rows:
#         value = row.get(key)
#         if value is None:
#             continue
#         try:
#             scalar = float(value)
#         except (TypeError, ValueError):
#             continue
#         if not np.isfinite(scalar):
#             continue
#         epochs.append(int(row.get("epoch", len(epochs) + 1)))
#         values.append(scalar * float(scale))
#     return np.asarray(epochs), np.asarray(values, dtype=np.float64)


# def _save_history_plot(
#     *,
#     train_rows: Sequence[Mapping[str, Any]],
#     validation_rows: Sequence[Mapping[str, Any]],
#     specs: Sequence[Tuple[str, str, str, float]],
#     title: str,
#     ylabel: str,
#     save_path: str,
#     dpi: int,
#     zero_line: bool = False,
#     ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
# ) -> Optional[str]:
#     fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor="white")
#     plotted = 0
#     for source, key, label, scale in specs:
#         rows = train_rows if source == "train" else validation_rows
#         epochs, values = _row_series(rows, key, scale=scale)
#         if values.size == 0:
#             continue
#         ax.plot(epochs, values, linewidth=2.0, label=label)
#         plotted += 1
#     if plotted == 0:
#         plt.close(fig)
#         return None
#     if zero_line:
#         ax.axhline(0.0, linewidth=1.0, alpha=0.45)
#     ax.set_title(title, fontsize=15, fontweight="bold")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel(ylabel)
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     ax.grid(True, alpha=0.22)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.legend(frameon=False, fontsize=9)
#     os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     return save_path


# def plot_base_training_dynamics(
#     history: Mapping[str, Any],
#     save_path: str,
#     *,
#     dpi: int = 240,
#     save_separate: bool = True,
# ) -> Dict[str, str]:
#     train_rows = _history_rows(history, "train")
#     validation_rows = _history_rows(history, "validation")
#     if not train_rows:
#         raise ValueError("history must contain non-empty train rows")
#     root, extension = os.path.splitext(save_path)
#     extension = extension or ".png"
#     output_dir = root + "_separate"
#     os.makedirs(output_dir, exist_ok=True)

#     plots = (
#         (
#             "objective",
#             (
#                 ("train", "total", "Total objective", 1.0),
#                 ("train", "ce", "Temporary CE", 1.0),
#                 ("train", "joint_geometry", "Deployed joint margin", 1.0),
#                 ("train", "feature_separation", "Conditional-feature margin", 1.0),
#             ),
#             "Base objective decomposition",
#             "Loss",
#             False,
#             None,
#         ),
#         (
#             "accuracy",
#             (
#                 ("train", "head_accuracy", "Temporary-head accuracy", 100.0),
#                 ("train", "joint_accuracy", "Episode joint accuracy", 100.0),
#                 ("train", "feature_accuracy", "Conditional-feature accuracy", 100.0),
#                 ("train", "spectral_accuracy", "Spectral-only accuracy", 100.0),
#                 ("validation", "accuracy", "Validation joint accuracy", 100.0),
#                 ("validation", "conditional_feature_accuracy", "Validation feature accuracy", 100.0),
#                 ("validation", "minimum_per_class_accuracy", "Validation minimum class", 100.0),
#             ),
#             "Representation and joint-geometry accuracy",
#             "Accuracy (%)",
#             False,
#             (0.0, 101.0),
#         ),
#         (
#             "energy_gap",
#             (
#                 ("train", "mean_gap", "Episode joint mean gap", 1.0),
#                 ("train", "q05_gap", "Episode joint q05 gap", 1.0),
#                 ("train", "feature_mean_gap", "Episode feature mean gap", 1.0),
#                 ("train", "feature_q05_gap", "Episode feature q05 gap", 1.0),
#                 ("validation", "mean_gap", "Validation joint mean gap", 1.0),
#                 ("validation", "q05_gap", "Validation joint q05 gap", 1.0),
#                 ("validation", "feature_q05_gap", "Validation feature q05 gap", 1.0),
#             ),
#             "Joint and conditional-feature separation",
#             "Rival minus target energy",
#             True,
#             None,
#         ),
#         (
#             "risk",
#             (
#                 ("train", "violation_rate", "Joint violation", 100.0),
#                 ("train", "feature_violation_rate", "Feature overlap", 100.0),
#                 ("train", "spectral_help_rate", "Spectral help", 100.0),
#                 ("train", "spectral_harm_rate", "Spectral harm", 100.0),
#                 ("validation", "classification_violation_rate", "Validation joint violation", 100.0),
#                 ("validation", "feature_overlap_rate", "Validation feature overlap", 100.0),
#             ),
#             "Boundary risk and spectral contribution",
#             "Rate (%)",
#             False,
#             (0.0, 101.0),
#         ),
#         (
#             "schedule",
#             (
#                 ("train", "ce_weight", "CE weight", 1.0),
#                 ("train", "joint_weight", "Joint margin weight", 1.0),
#                 ("train", "feature_weight", "Feature overlap weight", 1.0),
#                 ("train", "learning_rate", "Learning rate", 1.0),
#             ),
#             "Base optimization schedule",
#             "Value",
#             False,
#             None,
#         ),
#     )

#     outputs: Dict[str, str] = {}
#     if save_separate:
#         for name, specs, title, ylabel, zero_line, ylim in plots:
#             path = os.path.join(output_dir, f"{name}.png")
#             saved = _save_history_plot(
#                 train_rows=train_rows,
#                 validation_rows=validation_rows,
#                 specs=specs,
#                 title=title,
#                 ylabel=ylabel,
#                 save_path=path,
#                 dpi=dpi,
#                 zero_line=zero_line,
#                 ylim=ylim,
#             )
#             if saved:
#                 outputs[name] = saved

#     fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), facecolor="white")
#     for ax, (_, specs, title, ylabel, zero_line, ylim) in zip(axes.flat, plots[:4]):
#         for source, key, label, scale in specs:
#             rows = train_rows if source == "train" else validation_rows
#             epochs, values = _row_series(rows, key, scale=scale)
#             if values.size:
#                 ax.plot(epochs, values, linewidth=1.8, label=label)
#         if zero_line:
#             ax.axhline(0.0, linewidth=1.0, alpha=0.45)
#         ax.set_title(title)
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel(ylabel)
#         if ylim is not None:
#             ax.set_ylim(*ylim)
#         ax.grid(True, alpha=0.22)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         if ax.lines:
#             ax.legend(frameon=False, fontsize=7.2)
#     fig.suptitle(
#         "Base-phase evolving spectral-conditioned geometry diagnostics",
#         fontsize=17,
#         fontweight="bold",
#     )
#     os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#     fig.tight_layout(rect=(0, 0, 1, 0.97))
#     fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     outputs["dashboard"] = save_path
#     return outputs


# def plot_training_history(
#     history: Mapping[str, Any],
#     save_path: str = "./results/base_training_history.png",
#     save_separate: bool = True,
# ) -> Dict[str, str]:
#     return plot_base_training_dynamics(
#         history,
#         save_path,
#         save_separate=save_separate,
#     )


# # -----------------------------------------------------------------------------
# # Classification and geometry diagnostics
# # -----------------------------------------------------------------------------


# def save_classification_diagnostics(
#     *,
#     confusion_matrix: np.ndarray,
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     save_path: str,
#     title: str,
#     dpi: int = 240,
# ) -> str:
#     ids = [int(v) for v in class_ids]
#     matrix = np.asarray(confusion_matrix, dtype=np.int64)
#     if matrix.shape != (len(ids), len(ids)):
#         raise ValueError("confusion matrix shape does not match class_ids")
#     names = [_safe_name(target_names, class_id) for class_id in ids]
#     support = matrix.sum(axis=1)
#     predicted = matrix.sum(axis=0)
#     tp = np.diag(matrix)
#     precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
#     recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
#     f1 = np.divide(
#         2 * precision * recall,
#         precision + recall,
#         out=np.zeros_like(precision),
#         where=(precision + recall) > 0,
#     )

#     height = max(6.0, 0.48 * len(ids) + 2.5)
#     fig, axes = plt.subplots(
#         1,
#         2,
#         figsize=(16.5, height),
#         gridspec_kw={"width_ratios": [1.15, 1.0]},
#         facecolor="white",
#     )
#     fig.suptitle(title, fontsize=17, fontweight="bold")
#     image = axes[0].imshow(matrix, interpolation="nearest", cmap="Blues")
#     axes[0].set_title("Confusion matrix")
#     axes[0].set_xlabel("Predicted class")
#     axes[0].set_ylabel("True class")
#     axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right")
#     axes[0].set_yticks(range(len(names)), names)
#     threshold = matrix.max() / 2.0 if matrix.size else 0.0
#     for row in range(len(ids)):
#         for column in range(len(ids)):
#             axes[0].text(
#                 column,
#                 row,
#                 str(int(matrix[row, column])),
#                 ha="center",
#                 va="center",
#                 fontsize=7,
#                 color="white" if matrix[row, column] > threshold else "black",
#             )
#     fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

#     positions = np.arange(len(ids), dtype=float)
#     width = 0.24
#     axes[1].barh(positions - width, 100 * precision, height=width, label="Precision")
#     axes[1].barh(positions, 100 * recall, height=width, label="Recall")
#     axes[1].barh(positions + width, 100 * f1, height=width, label="F1")
#     axes[1].set_yticks(positions, names)
#     axes[1].set_xlim(0, 101)
#     axes[1].set_xlabel("Score (%)")
#     axes[1].set_title("Class-wise performance")
#     axes[1].grid(True, axis="x", alpha=0.22)
#     axes[1].spines["top"].set_visible(False)
#     axes[1].spines["right"].set_visible(False)
#     axes[1].legend(frameon=False)
#     os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#     fig.tight_layout(rect=(0, 0, 1, 0.97))
#     fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     return save_path


# def _to_numpy(value: Any) -> np.ndarray:
#     if torch.is_tensor(value):
#         return value.detach().cpu().numpy()
#     return np.asarray(value)


# def _row_covariance(row: Mapping[str, torch.Tensor], *, spectral: bool) -> torch.Tensor:
#     prefix = "spectral" if spectral else "feature"
#     basis = row[f"{prefix}_basis"]
#     eigvals = row[f"{prefix}_eigvals"]
#     residual = row[f"{prefix}_res_var"].reshape(()).clamp_min(1e-12)
#     active_rank = int(row[f"{prefix}_active_rank"].item())
#     dimension = int(basis.size(0))
#     covariance = torch.eye(dimension, device=basis.device, dtype=basis.dtype) * residual
#     if active_rank > 0:
#         active_basis = basis[:, :active_rank]
#         increments = (eigvals[:active_rank] - residual).clamp_min(0.0)
#         covariance = covariance + (
#             active_basis * increments.unsqueeze(0)
#         ) @ active_basis.transpose(0, 1)
#     return 0.5 * (covariance + covariance.transpose(0, 1))


# def _subspace_overlap(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> float:
#     rank_i = int(first["feature_active_rank"].item())
#     rank_j = int(second["feature_active_rank"].item())
#     if rank_i == 0 or rank_j == 0:
#         return 0.0
#     ui = first["feature_basis"][:, :rank_i]
#     uj = second["feature_basis"][:, :rank_j]
#     singular = torch.linalg.svdvals(ui.transpose(0, 1) @ uj)
#     return float(singular.square().mean().clamp(0.0, 1.0).item())


# def build_geometry_overlap_review(
#     *,
#     model: torch.nn.Module,
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     directional_invasion_matrix: Any,
# ) -> Dict[str, Any]:
#     ids = [int(v) for v in class_ids]
#     bank = getattr(model, "geometry_bank", None)
#     if bank is None:
#         raise RuntimeError("model must expose geometry_bank")
#     bank.assert_valid(ids, strict=True)
#     rows = {class_id: bank.get_class_row(class_id) for class_id in ids}
#     invasion = _to_numpy(directional_invasion_matrix).astype(np.float64)
#     expected = (len(ids), len(ids))
#     if invasion.shape != expected:
#         raise ValueError(
#             f"directional_invasion_matrix must have shape {expected}, got {invasion.shape}"
#         )

#     count = len(ids)
#     overlap = np.eye(count, dtype=np.float64)
#     clearance = np.full((count, count), np.nan, dtype=np.float64)
#     normalized_distance = np.zeros((count, count), dtype=np.float64)
#     spectral_distance = np.zeros((count, count), dtype=np.float64)
#     coupling_distance = np.zeros((count, count), dtype=np.float64)

#     for i, class_i in enumerate(ids):
#         row_i = rows[class_i]
#         cov_i = _row_covariance(row_i, spectral=False)
#         spec_cov_i = _row_covariance(row_i, spectral=True)
#         for j, class_j in enumerate(ids):
#             if i == j:
#                 clearance[i, j] = 0.0
#                 continue
#             row_j = rows[class_j]
#             cov_j = _row_covariance(row_j, spectral=False)
#             spec_cov_j = _row_covariance(row_j, spectral=True)
#             difference = row_j["feature_mean"] - row_i["feature_mean"]
#             distance = difference.norm().clamp_min(1e-12)
#             direction = difference / distance
#             radius_i = torch.sqrt(direction @ cov_i @ direction).clamp_min(1e-12)
#             radius_j = torch.sqrt(direction @ cov_j @ direction).clamp_min(1e-12)
#             clearance[i, j] = float((distance - 2.0 * (radius_i + radius_j)).item())
#             scale_i = torch.sqrt(torch.trace(cov_i) / max(cov_i.size(0), 1))
#             scale_j = torch.sqrt(torch.trace(cov_j) / max(cov_j.size(0), 1))
#             normalized_distance[i, j] = float(
#                 (distance / (scale_i + scale_j).clamp_min(1e-12)).item()
#             )
#             spec_difference = row_j["spectral_mean"] - row_i["spectral_mean"]
#             spec_scale = torch.sqrt(
#                 torch.trace(spec_cov_i) / max(spec_cov_i.size(0), 1)
#             ) + torch.sqrt(torch.trace(spec_cov_j) / max(spec_cov_j.size(0), 1))
#             spectral_distance[i, j] = float(
#                 (spec_difference.norm() / spec_scale.clamp_min(1e-12)).item()
#             )
#             coupling_distance[i, j] = float(
#                 (row_i["spectral_coupling"] - row_j["spectral_coupling"]).norm().item()
#             )
#             overlap[i, j] = _subspace_overlap(row_i, row_j)

#     effective_dimension = _to_numpy(bank.effective_dimension(ids)).astype(np.float64)
#     coupling_norm = np.asarray(
#         [float(rows[class_id]["spectral_coupling"].norm().item()) for class_id in ids],
#         dtype=np.float64,
#     )
#     coupling_r2 = np.asarray(
#         [
#             float(rows[class_id]["coupling_explained_variance"].item())
#             for class_id in ids
#         ],
#         dtype=np.float64,
#     )

#     pair_rows: List[Dict[str, Any]] = []
#     for first in range(count):
#         for second in range(first + 1, count):
#             left, right = ids[first], ids[second]
#             align = float(overlap[first, second])
#             pair_clearance = float(min(clearance[first, second], clearance[second, first]))
#             inv_lr = float(invasion[first, second])
#             inv_rl = float(invasion[second, first])
#             max_inv = max(inv_lr, inv_rl)
#             intersects = bool(np.isfinite(pair_clearance) and pair_clearance <= 0.0)
#             if intersects and max_inv >= 0.05:
#                 status = "HIGH"
#             elif intersects or max_inv >= 0.05 or (align >= 0.80 and pair_clearance < 0.50):
#                 status = "MODERATE"
#             else:
#                 status = "LOW"
#             reasons: List[str] = []
#             if intersects:
#                 reasons.append("two-sigma feature radii intersect")
#             if align >= 0.80:
#                 reasons.append("high feature-subspace alignment")
#             if max_inv >= 0.05:
#                 reasons.append("at least 5% empirical directional invasion")
#             if spectral_distance[first, second] < 1.0:
#                 reasons.append("weak spectral-center separation")
#             if not reasons:
#                 reasons.append("adequate operational separation")
#             pair_rows.append(
#                 {
#                     "class_i": left,
#                     "class_j": right,
#                     "class_i_name": _safe_name(target_names, left),
#                     "class_j_name": _safe_name(target_names, right),
#                     "subspace_overlap": align,
#                     "normalized_center_distance": float(normalized_distance[first, second]),
#                     "spectral_center_distance": float(spectral_distance[first, second]),
#                     "coupling_distance": float(coupling_distance[first, second]),
#                     "directional_clearance": pair_clearance,
#                     "invasion_i_to_j": inv_lr,
#                     "invasion_j_to_i": inv_rl,
#                     "maximum_bidirectional_invasion": max_inv,
#                     "ellipsoids_intersect": intersects,
#                     "overlap_status": status,
#                     "reason": "; ".join(reasons),
#                 }
#             )
#     order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
#     pair_rows.sort(
#         key=lambda row: (
#             order[row["overlap_status"]],
#             -float(row["maximum_bidirectional_invasion"]),
#             float(row["directional_clearance"]),
#         )
#     )
#     pair_mask = ~np.eye(count, dtype=bool)
#     finite_clearance = clearance[pair_mask & np.isfinite(clearance)]
#     return {
#         "definition": (
#             "Operational feature overlap combines empirical directional invasion, "
#             "two-sigma directional clearance, and feature-subspace alignment. "
#             "Spectral-center distance is reported separately."
#         ),
#         "class_ids": ids,
#         "feature_space_overlap_detected": any(
#             row["overlap_status"] == "HIGH" for row in pair_rows
#         ),
#         "high_risk_pair_count": sum(row["overlap_status"] == "HIGH" for row in pair_rows),
#         "moderate_risk_pair_count": sum(
#             row["overlap_status"] == "MODERATE" for row in pair_rows
#         ),
#         "maximum_subspace_overlap": float(overlap[pair_mask].max()) if pair_mask.any() else 0.0,
#         "minimum_directional_clearance": (
#             float(finite_clearance.min()) if finite_clearance.size else 0.0
#         ),
#         "maximum_directional_invasion": float(invasion[pair_mask].max()) if pair_mask.any() else 0.0,
#         "subspace_overlap_matrix": overlap,
#         "normalized_center_distance_matrix": normalized_distance,
#         "spectral_center_distance_matrix": spectral_distance,
#         "spectral_coupling_distance_matrix": coupling_distance,
#         "directional_clearance_matrix": clearance,
#         "directional_invasion_matrix": invasion,
#         "effective_dimension": effective_dimension,
#         "spectral_coupling_norm": coupling_norm,
#         "spectral_coupling_explained_variance": coupling_r2,
#         "pair_geometry": pair_rows,
#     }


# def save_geometry_pair_diagnostics(
#     *,
#     review: Mapping[str, Any],
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     save_path: str,
#     dpi: int = 240,
# ) -> str:
#     ids = [int(v) for v in class_ids]
#     names = [_safe_name(target_names, class_id) for class_id in ids]
#     count = len(ids)
#     invasion = np.asarray(review["directional_invasion_matrix"], dtype=float)
#     overlap = np.asarray(review["subspace_overlap_matrix"], dtype=float)
#     clearance = np.asarray(review["directional_clearance_matrix"], dtype=float)
#     spectral_distance = np.asarray(review["spectral_center_distance_matrix"], dtype=float)
#     effective_dimension = np.asarray(review["effective_dimension"], dtype=float)
#     coupling_r2 = np.asarray(
#         review["spectral_coupling_explained_variance"], dtype=float
#     )

#     fig, axes = plt.subplots(2, 3, figsize=(20.0, 12.0), facecolor="white")
#     fig.suptitle(
#         "Base spectral-conditioned geometry diagnostics",
#         fontsize=17,
#         fontweight="bold",
#     )
#     matrices = (
#         (100.0 * invasion, "Directional invasion (%)"),
#         (clearance, "Two-sigma directional clearance"),
#         (overlap, "Feature-subspace overlap"),
#         (spectral_distance, "Spectral-center distance"),
#     )
#     for ax, (matrix, title) in zip(axes.flat[:4], matrices):
#         image = ax.imshow(matrix, interpolation="nearest")
#         ax.set_title(title)
#         ax.set_xticks(range(count), names, rotation=45, ha="right")
#         ax.set_yticks(range(count), names)
#         for row in range(count):
#             for column in range(count):
#                 value = matrix[row, column]
#                 if np.isfinite(value):
#                     ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5)
#         fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

#     axes[1, 1].barh(np.arange(count), effective_dimension)
#     axes[1, 1].set_yticks(np.arange(count), names)
#     axes[1, 1].set_xlabel("Effective dimension")
#     axes[1, 1].set_title("Conditional-feature effective dimension")
#     axes[1, 1].grid(True, axis="x", alpha=0.22)
#     axes[1, 1].spines["top"].set_visible(False)
#     axes[1, 1].spines["right"].set_visible(False)

#     axes[1, 2].barh(np.arange(count), 100.0 * coupling_r2)
#     axes[1, 2].set_yticks(np.arange(count), names)
#     axes[1, 2].set_xlabel("Explained feature variance (%)")
#     axes[1, 2].set_title("Spectral-to-feature coupling")
#     axes[1, 2].set_xlim(0.0, 100.0)
#     axes[1, 2].grid(True, axis="x", alpha=0.22)
#     axes[1, 2].spines["top"].set_visible(False)
#     axes[1, 2].spines["right"].set_visible(False)

#     os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#     fig.tight_layout(rect=(0, 0, 1, 0.96))
#     fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     return save_path


# def save_spectral_conditioning_diagnostics(
#     *,
#     model: torch.nn.Module,
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     save_path: str,
#     dpi: int = 240,
# ) -> str:
#     ids = [int(v) for v in class_ids]
#     bank = model.geometry_bank
#     bank.assert_valid(ids, strict=True)
#     names = [_safe_name(target_names, class_id) for class_id in ids]
#     coupling_norm = []
#     coupling_r2 = []
#     spectral_rank = []
#     feature_rank = []
#     reliability = []
#     for class_id in ids:
#         row = bank.get_class_row(class_id)
#         coupling_norm.append(float(row["spectral_coupling"].norm().item()))
#         coupling_r2.append(float(row["coupling_explained_variance"].item()))
#         spectral_rank.append(int(row["spectral_active_rank"].item()))
#         feature_rank.append(int(row["feature_active_rank"].item()))
#         reliability.append(float(row["reliability"].item()))

#     positions = np.arange(len(ids), dtype=float)
#     fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), facecolor="white")
#     fig.suptitle("Physical spectral conditioning audit", fontsize=17, fontweight="bold")

#     axes[0, 0].barh(positions, coupling_norm)
#     axes[0, 0].set_yticks(positions, names)
#     axes[0, 0].set_xlabel("Frobenius norm")
#     axes[0, 0].set_title("Spectral-to-feature coupling magnitude")

#     axes[0, 1].barh(positions, 100.0 * np.asarray(coupling_r2))
#     axes[0, 1].set_yticks(positions, names)
#     axes[0, 1].set_xlim(0.0, 100.0)
#     axes[0, 1].set_xlabel("Explained variance (%)")
#     axes[0, 1].set_title("Feature variance explained by spectrum")

#     width = 0.36
#     axes[1, 0].barh(positions - width / 2, spectral_rank, height=width, label="Spectral rank")
#     axes[1, 0].barh(positions + width / 2, feature_rank, height=width, label="Feature rank")
#     axes[1, 0].set_yticks(positions, names)
#     axes[1, 0].set_xlabel("Active rank")
#     axes[1, 0].set_title("Class-specific active geometry rank")
#     axes[1, 0].legend(frameon=False)

#     axes[1, 1].barh(positions, 100.0 * np.asarray(reliability))
#     axes[1, 1].set_yticks(positions, names)
#     axes[1, 1].set_xlim(0.0, 100.0)
#     axes[1, 1].set_xlabel("Reliability (%)")
#     axes[1, 1].set_title("Aggregate row reliability")

#     for ax in axes.flat:
#         ax.grid(True, axis="x", alpha=0.22)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#     os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
#     fig.tight_layout(rect=(0, 0, 1, 0.96))
#     fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     return save_path
