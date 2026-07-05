"""
Clean visualization utilities for geometry-native NECIL-HSI.

Contract
--------
- Display value 0 is reserved for background / unseen / suppressed pixels.
- Real class id c is visualized as c + 1, so class 0 is never confused with BG.
- Prediction maps mask logits to seen classes before argmax.
- Confidence/error maps are intentionally not part of the core output.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


# ============================================================
# Colormap / label helpers
# ============================================================
def _build_cmap(num_classes_needed: int, cmap_name: str = "nipy_spectral", background_color: str = "#20252B") -> ListedColormap:
    num_classes_needed = int(max(num_classes_needed, 2))
    class_count = num_classes_needed - 1
    if not cmap_name:
        cmap_name = "nipy_spectral"
    base = plt.get_cmap(cmap_name)
    if cmap_name.lower() in {"tab10", "tab20", "tab20b", "tab20c", "set1", "set2", "set3", "paired"}:
        n = getattr(base, "N", class_count)
        class_colors = [base(i % n) for i in range(class_count)]
    else:
        samples = np.linspace(0.06, 0.94, class_count)
        class_colors = [base(float(s)) for s in samples]
    return ListedColormap([background_color] + class_colors)


def _safe_target_name(target_names: Optional[List[str]], cls: int) -> str:
    cls = int(cls)
    if target_names is not None and 0 <= cls < len(target_names):
        return str(target_names[cls])
    return f"Class {cls}"


def _safe_seen_classes(dataset_manager, phase: int) -> List[int]:
    if not hasattr(dataset_manager, "get_classes_up_to_phase"):
        raise AttributeError("dataset_manager must expose get_classes_up_to_phase(phase).")
    seen = sorted(set(int(c) for c in dataset_manager.get_classes_up_to_phase(int(phase))))
    if not seen:
        raise RuntimeError(f"No seen classes resolved for phase {phase}.")
    return seen


def _get_true_labels_and_coords(dataset_manager) -> Tuple[np.ndarray, np.ndarray]:
    labels = getattr(dataset_manager, "remapped_labels", getattr(dataset_manager, "labels", None))
    coords = getattr(dataset_manager, "coords", None)
    if labels is None or coords is None:
        raise AttributeError("dataset_manager must expose labels/remapped_labels and coords.")
    return np.asarray(labels).reshape(-1).astype(np.int64, copy=False), np.asarray(coords)


def _set_model_phase_and_old_count(model, dataset_manager, phase: int) -> int:
    phase = int(phase)
    old_class_count = 0 if phase == 0 else len(dataset_manager.get_classes_up_to_phase(phase - 1))
    if hasattr(model, "set_phase"):
        model.set_phase(phase)
    else:
        model.current_phase = phase
    if hasattr(model, "set_old_class_count"):
        model.set_old_class_count(old_class_count)
    else:
        model.old_class_count = old_class_count
    return int(old_class_count)


def _resolve_viz_modes(phase: int, classifier_mode: Optional[str], semantic_mode: Optional[str]) -> Tuple[str, str]:
    # Clean architecture uses geometry_only for all phases.
    if classifier_mode is None:
        classifier_mode = "geometry_only"
    if semantic_mode is None:
        semantic_mode = "identity"
    return str(classifier_mode).lower(), str(semantic_mode).lower()


def _mask_logits_to_seen(logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
    if logits.dim() != 2:
        raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
    seen = torch.as_tensor([int(c) for c in seen_classes], device=logits.device, dtype=torch.long)
    if seen.numel() == 0:
        raise RuntimeError("seen_classes is empty.")
    if int(seen.max().item()) >= logits.size(1) or int(seen.min().item()) < 0:
        raise RuntimeError(f"seen_classes {seen.detach().cpu().tolist()} incompatible with logits width={logits.size(1)}")
    masked = torch.full_like(logits, -float("inf"))
    masked.index_copy_(1, seen, logits.index_select(1, seen))
    return masked


@torch.no_grad()
def _viz_model_forward(model: torch.nn.Module, dataset_manager, batch: torch.Tensor, phase: int, classifier_mode: Optional[str] = None, semantic_mode: Optional[str] = None):
    _set_model_phase_and_old_count(model, dataset_manager, phase)
    classifier_mode, semantic_mode = _resolve_viz_modes(phase, classifier_mode, semantic_mode)
    try:
        return model(batch, semantic_mode=semantic_mode, classifier_mode=classifier_mode)
    except TypeError:
        return model(batch, classifier_mode=classifier_mode)


def _clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _legend_items(cmap: ListedColormap, seen_classes: List[int], target_names: Optional[List[str]]) -> List[mpatches.Patch]:
    items = [mpatches.Patch(color=cmap.colors[0], label="BG: Background / Unseen / Suppressed")]
    for cls in seen_classes:
        idx = int(cls) + 1
        if idx < len(cmap.colors):
            items.append(mpatches.Patch(color=cmap.colors[idx], label=f"{cls}: {_safe_target_name(target_names, cls)}"))
    return items


def _save_single_map_figure(
    class_map: np.ndarray,
    cmap: ListedColormap,
    title: str,
    save_path: str,
    seen_classes: List[int],
    target_names: Optional[List[str]],
    metric_text: Optional[str] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8.2, 8.2), facecolor="white")
    ax.imshow(class_map, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(cmap.colors) - 1)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    _clean_axis(ax)
    if metric_text:
        ax.text(
            0.985,
            0.02,
            metric_text,
            transform=ax.transAxes,
            fontsize=10,
            color="#111111",
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.86, edgecolor="#DDDDDD", boxstyle="round,pad=0.4"),
        )
    handles = _legend_items(cmap, seen_classes, target_names)
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 0.01), title="Labels", title_fontsize=12, fontsize=9.5, frameon=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.subplots_adjust(bottom=0.17, top=0.92, left=0.03, right=0.97)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


# ============================================================
# Phase map prediction
# ============================================================
@torch.no_grad()
def predict_phase_grid(
    model: torch.nn.Module,
    dataset_manager,
    phase: int,
    target_names: Optional[List[str]],
    save_dir: str = "./results/phase_visuals",
    device: str = "cuda",
    patch_size=None,
    chunk_size: int = 512,
    classifier_mode: Optional[str] = None,
    semantic_mode: Optional[str] = None,
    save_numpy: bool = True,
    return_outputs: bool = False,
    class_cmap: str = "nipy_spectral",
    confidence_cmap: str = "magma",  # compatibility only
    background_color: str = "#20252B",
    save_error_map: bool = False,  # compatibility only
    save_combined_figure: bool = False,
    save_legacy_publication_name: bool = False,
):
    del confidence_cmap, save_error_map
    phase = int(phase)
    model.eval()
    old_class_count = _set_model_phase_and_old_count(model, dataset_manager, phase)

    gt_shape = tuple(dataset_manager.gt_shape)
    true_labels, coords = _get_true_labels_and_coords(dataset_manager)
    patches = dataset_manager.patches
    num_samples = len(patches)
    if num_samples != len(coords) or num_samples != len(true_labels):
        raise ValueError(f"patch/coord/label length mismatch: patches={num_samples}, coords={len(coords)}, labels={len(true_labels)}")

    seen_classes = _safe_seen_classes(dataset_manager, phase)
    seen_set = set(seen_classes)
    classifier_mode, semantic_mode = _resolve_viz_modes(phase, classifier_mode, semantic_mode)

    preds_all: List[np.ndarray] = []
    conf_all: List[np.ndarray] = []
    raw_invalid_all: List[np.ndarray] = []

    for start in range(0, num_samples, int(chunk_size)):
        end = min(start + int(chunk_size), num_samples)
        batch_np = patches[start:end]
        batch = batch_np.to(device).float() if torch.is_tensor(batch_np) else torch.from_numpy(np.asarray(batch_np)).float().to(device)
        out = _viz_model_forward(model, dataset_manager, batch, phase, classifier_mode, semantic_mode)
        logits_raw = out["logits"]
        raw_pred = logits_raw.argmax(dim=1)
        logits = _mask_logits_to_seen(logits_raw, seen_classes)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        seen_t = torch.as_tensor(seen_classes, device=logits_raw.device, dtype=torch.long)
        if hasattr(torch, "isin"):
            raw_invalid = ~torch.isin(raw_pred, seen_t)
        else:
            valid = torch.zeros_like(raw_pred, dtype=torch.bool)
            for c in seen_t:
                valid |= raw_pred == c
            raw_invalid = ~valid
        preds_all.append(pred.detach().cpu().numpy().astype(np.int64))
        conf_all.append(conf.detach().cpu().numpy().astype(np.float32))
        raw_invalid_all.append(raw_invalid.detach().cpu().numpy().astype(bool))

    if not preds_all:
        print(f"[Viz] No predictions generated for phase {phase}.")
        return None

    preds = np.concatenate(preds_all, axis=0)
    conf = np.concatenate(conf_all, axis=0)
    raw_invalid = np.concatenate(raw_invalid_all, axis=0)

    pred_map = np.zeros(gt_shape, dtype=np.int32)
    gt_map = np.zeros(gt_shape, dtype=np.int32)
    correct = 0
    visible = 0
    suppressed_true_unseen = 0
    suppressed_raw_unseen_pred = 0
    old_correct = old_total = 0
    new_correct = new_total = 0

    for i, (r, c) in enumerate(coords):
        y = int(true_labels[i])
        r, c = int(r), int(c)
        if y not in seen_set:
            gt_map[r, c] = 0
            pred_map[r, c] = 0
            suppressed_true_unseen += 1
            continue
        p = int(preds[i])
        gt_map[r, c] = y + 1
        pred_map[r, c] = p + 1 if p in seen_set else 0
        visible += 1
        if bool(raw_invalid[i]):
            suppressed_raw_unseen_pred += 1
        is_correct = p == y
        correct += int(is_correct)
        if y < old_class_count:
            old_total += 1
            old_correct += int(is_correct)
        else:
            new_total += 1
            new_correct += int(is_correct)

    oa = 100.0 * correct / max(visible, 1)
    old_acc = 100.0 * old_correct / max(old_total, 1)
    new_acc = 100.0 * new_correct / max(new_total, 1)
    hm = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
    raw_unseen_rate = 100.0 * suppressed_raw_unseen_pred / max(visible, 1)
    true_unseen_rate = 100.0 * suppressed_true_unseen / max(num_samples, 1)
    mean_conf = 100.0 * float(conf.mean()) if conf.size else 0.0

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"
    num_display = int(max(max(seen_classes) + 2, gt_map.max() + 1, pred_map.max() + 1))
    cmap = _build_cmap(num_display, cmap_name=class_cmap, background_color=background_color)

    stat_text = (
        f"OA: {oa:.2f}%\n"
        f"Old: {old_acc:.2f}% | New: {new_acc:.2f}%\n"
        f"H: {hm:.2f}%\n"
        f"Seen classes: {len(seen_classes)}\n"
        f"Visible pixels: {visible}\n"
        f"Raw unseen pred: {raw_unseen_rate:.2f}%\n"
        f"Mean conf: {mean_conf:.2f}%"
    )

    os.makedirs(save_dir, exist_ok=True)
    p_str = f"_ps{patch_size}" if patch_size else ""
    gt_path = os.path.join(save_dir, f"phase_{phase}{p_str}_ground_truth.png")
    pred_path = os.path.join(save_dir, f"phase_{phase}{p_str}_prediction.png")
    combined_path = os.path.join(save_dir, f"phase_{phase}{p_str}_gt_pred.png")

    _save_single_map_figure(gt_map, cmap, f"Phase {phase} Ground Truth", gt_path, seen_classes, target_names)
    _save_single_map_figure(pred_map, cmap, f"Phase {phase} Prediction", pred_path, seen_classes, target_names, stat_text)

    publication_path = None
    if save_combined_figure or save_legacy_publication_name:
        fig = plt.figure(figsize=(16.5, 8.4), facecolor="white")
        gs = fig.add_gridspec(1, 2, wspace=0.08)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.imshow(gt_map, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(cmap.colors) - 1)
        ax2.imshow(pred_map, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(cmap.colors) - 1)
        ax1.set_title(f"Phase {phase} Ground Truth", fontsize=18, fontweight="bold", pad=12)
        ax2.set_title(f"Phase {phase} Prediction", fontsize=18, fontweight="bold", pad=12)
        _clean_axis(ax1)
        _clean_axis(ax2)
        ax2.text(0.985, 0.02, stat_text, transform=ax2.transAxes, fontsize=10, color="#111111", ha="right", va="bottom", bbox=dict(facecolor="white", alpha=0.86, edgecolor="#DDDDDD", boxstyle="round,pad=0.4"))
        handles = _legend_items(cmap, seen_classes, target_names)
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 0.01), title="Labels", title_fontsize=12, fontsize=9.5, frameon=False)
        plt.subplots_adjust(bottom=0.17, top=0.92, left=0.03, right=0.97)
        plt.savefig(combined_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if save_legacy_publication_name:
            import shutil
            publication_path = os.path.join(save_dir, f"phase_{phase}{p_str}_publication.png")
            shutil.copyfile(combined_path, publication_path)

    if save_numpy:
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_pred_map.npy"), pred_map)
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_gt_map.npy"), gt_map)
        np.save(os.path.join(save_dir, f"phase_{phase}{p_str}_confidence.npy"), conf)

    print(
        f"[Viz] Saved Phase {phase} Ground Truth to: {gt_path}\n"
        f"[Viz] Saved Phase {phase} Prediction to: {pred_path}\n"
        + (f"[Viz] Saved combined GT/Prediction to: {combined_path}\n" if (save_combined_figure or save_legacy_publication_name) else "")
        + f"[Viz] Metrics — OA: {oa:.2f}%, Old: {old_acc:.2f}%, New: {new_acc:.2f}%, H: {hm:.2f}%, RawUnseenPred: {raw_unseen_rate:.2f}%"
    )

    outputs = {
        "pred_map": pred_map,
        "gt_map": gt_map,
        "metrics": {
            "overall_accuracy": oa,
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "harmonic_mean": hm,
            "raw_unseen_prediction_rate": raw_unseen_rate,
            "true_unseen_suppression_rate": true_unseen_rate,
            "visible_pixels": int(visible),
            "seen_classes": seen_classes,
            "classifier_mode": classifier_mode,
            "semantic_mode": semantic_mode,
            "mean_confidence": mean_conf,
        },
        "ground_truth_path": gt_path,
        "prediction_path": pred_path,
        "combined_path": combined_path if (save_combined_figure or save_legacy_publication_name) else None,
        "publication_path": publication_path,
    }
    return outputs if return_outputs else pred_path


# ============================================================
# Training history plots
# ============================================================
def _save_single_history_plot(*, x, series, title: str, xlabel: str, ylabel: str, save_path: str, phase_boundaries=None, ylim=None) -> str:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")
    for name, values in series:
        if values is not None and len(values) > 0:
            ax.plot(x, values, linewidth=2.2, label=name)
    if phase_boundaries is not None:
        for phase_start in phase_boundaries:
            if phase_start > 0:
                ax.axvline(x=phase_start, color="k", linestyle="--", alpha=0.35)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.tight_layout()
    plt.savefig(save_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Viz] Saved {title} to: {save_path}")
    return save_path


def plot_training_history(history: dict, save_path: str = "./results/training_history.png", save_separate: bool = False):
    if not history or "train_loss" not in history:
        raise ValueError("history must contain at least train_loss.")
    n = len(history.get("train_loss", []))
    epochs = list(range(1, n + 1))
    phase_boundaries = history.get("phase_boundaries", [])
    has_old_new = all(k in history for k in ("val_old_acc", "val_new_acc", "val_hm"))
    nrows = 3 if has_old_new else 2

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4.8 * nrows), facecolor="white")
    if nrows == 1:
        axes = [axes]

    axes[0].plot(epochs, history.get("train_loss", []), linewidth=2.0, label="Train Loss")
    if len(history.get("val_loss", [])) == n:
        axes[0].plot(epochs, history.get("val_loss", []), linewidth=2.0, label="Val Loss")
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    if len(history.get("train_acc", [])) == n:
        axes[1].plot(epochs, history.get("train_acc", []), linewidth=2.0, label="Train Acc")
    if len(history.get("val_acc", [])) == n:
        axes[1].plot(epochs, history.get("val_acc", []), linewidth=2.0, label="Val Acc")
    axes[1].set_title("Overall Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)

    if has_old_new:
        if len(history.get("val_old_acc", [])) == n:
            axes[2].plot(epochs, history.get("val_old_acc", []), linewidth=2.0, label="Old Acc")
        if len(history.get("val_new_acc", [])) == n:
            axes[2].plot(epochs, history.get("val_new_acc", []), linewidth=2.0, label="New Acc")
        if len(history.get("val_hm", [])) == n:
            axes[2].plot(epochs, history.get("val_hm", []), linewidth=2.0, label="Harmonic Mean")
        axes[2].set_title("Old/New Stability-Plasticity Balance", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy / H (%)")
        axes[2].set_ylim(0, 100)
        axes[2].legend(frameon=False)
        axes[2].grid(True, alpha=0.25)

    for phase_start in phase_boundaries:
        if phase_start > 0:
            for ax in axes:
                ax.axvline(x=phase_start, color="k", linestyle="--", alpha=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Viz] Saved training history to: {save_path}")

    saved = {"combined": save_path}
    if save_separate:
        root, ext = os.path.splitext(save_path)
        ext = ext or ".png"
        saved["loss"] = _save_single_history_plot(x=epochs, series=[("Train Loss", history.get("train_loss", [])), ("Val Loss", history.get("val_loss", []))], title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss", save_path=f"{root}_loss{ext}", phase_boundaries=phase_boundaries)
        saved["accuracy"] = _save_single_history_plot(x=epochs, series=[("Train Acc", history.get("train_acc", [])), ("Val Acc", history.get("val_acc", []))], title="Overall Accuracy", xlabel="Epoch", ylabel="Accuracy (%)", save_path=f"{root}_accuracy{ext}", phase_boundaries=phase_boundaries, ylim=(0, 100))
        if has_old_new:
            saved["old_new_hm"] = _save_single_history_plot(x=epochs, series=[("Old Acc", history.get("val_old_acc", [])), ("New Acc", history.get("val_new_acc", [])), ("Harmonic Mean", history.get("val_hm", []))], title="Old/New Stability-Plasticity Balance", xlabel="Epoch", ylabel="Accuracy / H (%)", save_path=f"{root}_old_new_hm{ext}", phase_boundaries=phase_boundaries, ylim=(0, 100))
    return saved


def plot_phase_metric_summary(phase_history: Dict[int, Dict], save_path: str = "./results/phase_metric_summary.png") -> str:
    if not phase_history:
        raise ValueError("phase_history is empty.")
    phases = sorted(int(p) for p in phase_history.keys())
    oa = [float(phase_history[p].get("overall_accuracy", 0.0)) for p in phases]
    old = [float(phase_history[p].get("old_accuracy", 0.0)) for p in phases]
    new = [float(phase_history[p].get("new_accuracy", 0.0)) for p in phases]
    hm = [float(phase_history[p].get("harmonic_mean", 0.0)) for p in phases]
    invalid = [float(phase_history[p].get("invalid_prediction_rate", 0.0)) for p in phases]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.plot(phases, oa, marker="o", linewidth=2.0, label="OA")
    ax.plot(phases, old, marker="o", linewidth=2.0, label="Old")
    ax.plot(phases, new, marker="o", linewidth=2.0, label="New")
    ax.plot(phases, hm, marker="o", linewidth=2.0, label="H")
    ax.plot(phases, invalid, marker="o", linewidth=2.0, label="Invalid Pred")
    ax.set_title("NECIL-HSI Phase Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Metric (%)")
    ax.set_xticks(phases)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Viz] Saved phase metric summary to: {save_path}")
    return save_path





