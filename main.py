from __future__ import annotations

"""NECIL-HSI continual experiment driver.

The driver owns only experiment/runtime orchestration:

    protocol + base-fitted preprocessing
        -> phase 0
        -> phase 1
        -> ...
        -> final phase

Architecture:
    spectral-primary HSI backbone
    + persistent pairwise boundary geometry
    + compact spectral variation for future historical evidence
    + parameter-free equal-rule energy classifier

The driver does not implement model mathematics.
"""

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from data.hsi_dataloader_pytorch import (
    ExtractLabeledPixelIndex,
    FitHSIPreprocessorFromProtocol,
    LoadRawHSIData,
    SaveHSIPreprocessor,
)
from data.incremental_dataset import (
    IncrementalHSIDatasetManager,
    build_incremental_protocol,
)
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from utils.eval import geometry_diagnostics
from utils.qualitative_report import (
    generate_base_qualitative_maps,
    generate_phase_qualitative_maps,
)
from utils.visualize import (
    save_geometry_diagnostic_figures,
    save_incremental_geometry_diagnostic_figures,
)


# ---------------------------------------------------------------------------
# Generic IO / reproducibility
# ---------------------------------------------------------------------------

def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value {value!r}")


def _parse_class_order(value: str) -> Optional[list[int]]:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        values = [
            int(item.strip())
            for item in token.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "class_order must be comma-separated integer IDs"
        ) from exc
    if (
        not values
        or len(values) != len(set(values))
        or any(value < 0 for value in values)
    ):
        raise ValueError(
            "class_order must contain unique non-negative IDs"
        )
    return values


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return (
            tensor.item()
            if tensor.numel() == 1
            else tensor.tolist()
        )
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: str, value: Mapping[str, Any]) -> str:
    destination = os.path.abspath(path)
    os.makedirs(
        os.path.dirname(destination) or ".",
        exist_ok=True,
    )
    temporary = destination + ".tmp"
    try:
        with open(
            temporary, "w", encoding="utf-8"
        ) as stream:
            json.dump(
                _json_safe(value),
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return destination


def save_csv(
    path: str,
    rows: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    values = list(rows)
    if not values:
        return None
    destination = os.path.abspath(path)
    os.makedirs(
        os.path.dirname(destination) or ".",
        exist_ok=True,
    )
    fieldnames = list(values[0])
    with open(
        destination,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames
        )
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    name: _json_safe(row.get(name))
                    for name in fieldnames
                }
            )
    return destination


def set_seed(seed: int, deterministic: bool) -> None:
    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        )
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if (
        hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
    ):
        torch.backends.cuda.matmul.allow_tf32 = (
            not deterministic
        )
    if (
        hasattr(torch.backends, "cudnn")
        and hasattr(torch.backends.cudnn, "allow_tf32")
    ):
        torch.backends.cudnn.allow_tf32 = (
            not deterministic
        )
    torch.use_deterministic_algorithms(
        bool(deterministic),
        warn_only=False,
    )


def resolve_device(value: str) -> str:
    token = (
        str(value)
        .strip()
        .lower()
        .replace("gpu", "cuda")
    )
    try:
        requested = torch.device(token)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"invalid device {value!r}"
        ) from exc
    if requested.type == "cpu":
        return "cpu"
    if (
        requested.type != "cuda"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"requested device {value!r} is unavailable"
        )
    index = (
        torch.cuda.current_device()
        if requested.index is None
        else requested.index
    )
    if (
        index is None
        or not 0 <= int(index) < torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device index {index} is unavailable"
        )
    torch.cuda.set_device(int(index))
    return f"cuda:{int(index)}"


def _load_band_positions(
    path: str,
    expected_bands: int,
) -> Optional[list[float]]:
    token = str(path or "").strip()
    if not token:
        return None
    if not os.path.isfile(token):
        raise FileNotFoundError(token)
    values = (
        np.load(token)
        if token.lower().endswith(".npy")
        else np.loadtxt(token)
    )
    positions = np.asarray(
        values, dtype=np.float32
    ).reshape(-1)
    if positions.size != int(expected_bands):
        raise ValueError(
            f"band position file has {positions.size} values; "
            f"expected {expected_bands}"
        )
    if (
        not np.isfinite(positions).all()
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError(
            "band positions must be finite and strictly increasing"
        )
    return positions.tolist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NECIL-HSI pairwise-boundary continual learning "
            "with boundary-conditioned historical spectral variation"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("Experiment")
    experiment.add_argument("--dataset", default="IP")
    experiment.add_argument("--data_dir", default="./datasets")
    experiment.add_argument(
        "--save_dir",
        default="./results_pairwise_geometry",
    )
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--device", default="cuda:0")
    experiment.add_argument(
        "--num_workers", type=int, default=0
    )
    experiment.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
    )
    experiment.add_argument(
        "--run_mode",
        choices=("base", "incremental", "all"),
        default="base",
        help=(
            "base: phase 0 only (recommended first validation run); "
            "incremental: continue from "
            "--resume_checkpoint; all: fresh full run or continue "
            "from --resume_checkpoint"
        ),
    )
    experiment.add_argument(
        "--resume_checkpoint",
        default="",
        help=(
            "Checkpoint produced by the current continual runtime. "
            "No compatibility conversion is performed."
        ),
    )

    protocol = parser.add_argument_group(
        "Class-incremental protocol"
    )
    protocol.add_argument(
        "--base_classes", type=int, default=6
    )
    protocol.add_argument(
        "--increment",
        type=int,
        default=6,
        help=(
            "New classes per incremental phase; the final "
            "phase uses the remainder."
        ),
    )
    protocol.add_argument(
        "--class_order",
        default="",
        help=(
            "Complete zero-based class permutation, "
            "e.g. 0,1,...,15"
        ),
    )
    protocol.add_argument(
        "--shuffle_order",
        type=str2bool,
        default=False,
    )
    protocol.add_argument(
        "--train_ratio", type=float, default=0.20
    )
    protocol.add_argument(
        "--val_ratio", type=float, default=0.10
    )
    protocol.add_argument(
        "--split_strategy",
        choices=("random_pixel", "spatial_block"),
        default="random_pixel",
    )
    protocol.add_argument(
        "--spatial_block_size",
        type=int,
        default=33,
    )
    protocol.add_argument(
        "--require_patch_disjoint",
        type=str2bool,
        default=False,
    )

    preprocessing = parser.add_argument_group(
        "HSI preprocessing"
    )
    preprocessing.add_argument(
        "--patch_size", type=int, default=11
    )
    preprocessing.add_argument(
        "--no_pca", action="store_true"
    )
    preprocessing.add_argument(
        "--pca_components", type=int, default=30
    )
    preprocessing.add_argument(
        "--pca_whiten",
        type=str2bool,
        default=False,
    )
    preprocessing.add_argument(
        "--band_positions_file",
        default="",
    )

    backbone = parser.add_argument_group(
        "Spectral-primary contextual Mamba"
    )
    backbone.add_argument(
        "--d_model", type=int, default=128
    )
    backbone.add_argument(
        "--representation_dim",
        type=int,
        default=16,
        help=(
            "Dimension of the one canonical HSI/geometry space."
        ),
    )
    backbone.add_argument(
        "--d_state", type=int, default=16
    )
    backbone.add_argument(
        "--d_conv", type=int, default=4
    )
    backbone.add_argument(
        "--expand", type=int, default=2
    )
    backbone.add_argument(
        "--num_spectral_layers",
        type=int,
        default=2,
    )
    backbone.add_argument(
        "--num_spatial_layers",
        type=int,
        default=2,
    )
    backbone.add_argument(
        "--stem_norm_groups",
        type=int,
        default=8,
    )
    backbone.add_argument(
        "--mamba_dropout",
        type=float,
        default=0.0,
    )

    base = parser.add_argument_group(
        "Base optimization"
    )
    base.add_argument(
        "--epochs_base", type=int, default=100
    )
    base.add_argument(
        "--batch_size", type=int, default=64
    )
    base.add_argument(
        "--eval_batch_size", type=int, default=256
    )
    base.add_argument(
        "--lr", type=float, default=1e-4
    )
    base.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    base.add_argument(
        "--gradient_clip",
        type=float,
        default=5.0,
    )
    base.add_argument(
        "--base_classification_weight",
        type=float,
        default=1.0,
    )
    base.add_argument(
        "--base_separation_weight",
        type=float,
        default=1.0,
        help=(
            "Weight of pair-balanced distribution separation in the "
            "canonical HSI representation."
        ),
    )

    incremental = parser.add_argument_group(
        "Incremental optimization"
    )
    incremental.add_argument(
        "--epochs_inc",
        type=int,
        default=15,
    )
    incremental.add_argument(
        "--lr_inc",
        type=float,
        default=None,
        help=(
            "Incremental backbone/candidate learning rate. "
            "If omitted, --lr is reused."
        ),
    )
    incremental.add_argument(
        "--incremental_classification_weight",
        type=float,
        default=None,
        help=(
            "Incremental all-seen classification weight. If omitted, "
            "--base_classification_weight is reused."
        ),
    )
    incremental.add_argument(
        "--incremental_separation_weight",
        type=float,
        default=None,
        help=(
            "Incremental old-new/new-new pair-separation weight. If omitted, "
            "--base_separation_weight is reused."
        ),
    )
    incremental.add_argument(
        "--preservation_weight",
        type=float,
        default=1.0,
        help=(
            "Incremental-only weight for preserving historical "
            "class-incident old-boundary responses."
        ),
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "base_classes",
        "increment",
        "patch_size",
        "pca_components",
        "d_model",
        "representation_dim",
        "d_state",
        "d_conv",
        "expand",
        "num_spectral_layers",
        "num_spatial_layers",
        "stem_norm_groups",
        "epochs_base",
        "epochs_inc",
        "batch_size",
        "eval_batch_size",
        "spatial_block_size",
    )
    for name in positive_integer_fields:
        if int(getattr(args, name)) <= 0:
            raise ValueError(
                f"{name} must be positive"
            )

    if int(args.patch_size) % 2 == 0:
        raise ValueError(
            "patch_size must be odd"
        )

    context_stem_width = max(
        int(args.d_model) // 2, 16
    )
    groups = int(args.stem_norm_groups)
    if (
        groups > context_stem_width
        or context_stem_width % groups != 0
    ):
        raise ValueError(
            "stem_norm_groups must divide the context stem "
            f"width ({context_stem_width})"
        )
    if int(args.num_workers) < 0:
        raise ValueError(
            "num_workers must be non-negative"
        )

    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if (
        train_ratio <= 0.0
        or val_ratio <= 0.0
        or train_ratio + val_ratio >= 1.0
    ):
        raise ValueError(
            "require train_ratio>0, val_ratio>0, "
            "and their sum<1"
        )
    if not 0.0 <= float(args.mamba_dropout) < 1.0:
        raise ValueError(
            "mamba_dropout must lie in [0,1)"
        )

    positive_floats = {
        "lr": args.lr,
    }
    if args.lr_inc is not None:
        positive_floats["lr_inc"] = args.lr_inc
    for name, raw in positive_floats.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and positive"
            )

    for name in (
        "weight_decay",
        "gradient_clip",
        "base_classification_weight",
        "base_separation_weight",
        "preservation_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be finite and non-negative"
            )

    for name in (
        "incremental_classification_weight",
        "incremental_separation_weight",
    ):
        raw = getattr(args, name)
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be finite and non-negative when provided"
            )

    if not any(
        float(getattr(args, name)) > 0.0
        for name in (
            "base_classification_weight",
            "base_separation_weight",
        )
    ):
        raise ValueError(
            "at least one base objective weight must be positive"
        )

    resolved_incremental_classification = (
        float(args.base_classification_weight)
        if args.incremental_classification_weight is None
        else float(args.incremental_classification_weight)
    )
    resolved_incremental_separation = (
        float(args.base_separation_weight)
        if args.incremental_separation_weight is None
        else float(args.incremental_separation_weight)
    )
    if (
        args.run_mode != "base"
        and resolved_incremental_classification == 0.0
        and resolved_incremental_separation == 0.0
        and float(args.preservation_weight) == 0.0
    ):
        raise ValueError(
            "incremental execution requires at least one positive "
            "classification, separation, or preservation weight"
        )

    resume = str(
        args.resume_checkpoint or ""
    ).strip()
    if args.run_mode == "incremental" and not resume:
        raise ValueError(
            "run_mode=incremental requires --resume_checkpoint"
        )
    if args.run_mode == "base" and resume:
        raise ValueError(
            "run_mode=base does not accept --resume_checkpoint"
        )

    _parse_class_order(args.class_order)


# ---------------------------------------------------------------------------
# Dataset / protocol construction
# ---------------------------------------------------------------------------

def _build_data(
    args: argparse.Namespace,
) -> tuple[
    IncrementalHSIDatasetManager,
    Dict[str, Any],
    Mapping[str, Any],
]:
    raw_cube, gt, class_count, names, _, label_policy = (
        LoadRawHSIData(
            args.dataset,
            base_dir=args.data_dir,
        )
    )
    labels, coords = ExtractLabeledPixelIndex(
        gt,
        label_policy=label_policy,
    )

    if int(args.base_classes) > int(class_count):
        raise ValueError(
            f"base_classes={args.base_classes} exceeds "
            f"dataset classes={class_count}"
        )

    class_order = _parse_class_order(
        args.class_order
    )
    if (
        class_order is not None
        and set(class_order)
        != set(range(int(class_count)))
    ):
        raise ValueError(
            "class_order must be a permutation of "
            f"0..{int(class_count) - 1}"
        )

    protocol = build_incremental_protocol(
        labels,
        coords,
        gt_shape=gt.shape,
        class_count=int(class_count),
        base_classes=int(args.base_classes),
        increment=int(args.increment),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        shuffle_order=bool(args.shuffle_order),
        class_order=class_order,
        target_names=names,
        split_strategy=str(args.split_strategy),
        spatial_block_size=int(
            args.spatial_block_size
        ),
    )

    (
        processed_cube,
        ordered_cube,
        preprocessing_state,
    ) = FitHSIPreprocessorFromProtocol(
        raw_cube,
        coords=coords,
        protocol=protocol,
        fit_scope="base_train",
        apply_reduction=not bool(args.no_pca),
        reduction_method="PCA",
        n_components=int(args.pca_components),
        whiten=bool(args.pca_whiten),
    )

    args.num_bands = int(
        processed_cube.shape[2]
    )
    args.spectral_bands = int(
        ordered_cube.shape[2]
    )
    args.raw_num_bands = int(
        ordered_cube.shape[2]
    )
    args.band_positions = _load_band_positions(
        args.band_positions_file,
        args.spectral_bands,
    )
    args.band_wavelengths = None

    manager = IncrementalHSIDatasetManager(
        processed_cube=processed_cube,
        ordered_spectral_cube=ordered_cube,
        labels=labels,
        coords=coords,
        protocol=protocol,
        target_names=names,
        gt_shape=gt.shape,
        patch_size=int(args.patch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        seed=int(args.seed),
        require_patch_disjoint=bool(
            args.require_patch_disjoint
        ),
        context_policy="full_scene_reflect",
    )

    expected_phase_sizes = [
        int(args.base_classes)
    ]
    remaining = (
        int(class_count)
        - int(args.base_classes)
    )
    while remaining > 0:
        current = min(
            int(args.increment), remaining
        )
        expected_phase_sizes.append(current)
        remaining -= current

    actual_phase_sizes = [
        len(manager.phase_to_classes[phase])
        for phase in sorted(
            manager.phase_to_classes
        )
    ]
    if actual_phase_sizes != expected_phase_sizes:
        raise RuntimeError(
            "constructed class-incremental schedule is "
            "inconsistent: "
            f"expected={expected_phase_sizes}, "
            f"actual={actual_phase_sizes}"
        )

    data_summary: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "raw_shape": list(raw_cube.shape),
        "gt_shape": list(gt.shape),
        "labeled_sample_count": int(labels.size),
        "class_count": int(class_count),
        "phase_sizes": [
            len(
                protocol["phase_to_classes"][phase]
            )
            for phase in sorted(
                protocol["phase_to_classes"]
            )
        ],
        "phase_schedule": {
            int(phase): [
                int(value) for value in classes
            ]
            for phase, classes
            in protocol["phase_to_classes"].items()
        },
        "base_global_class_ids": list(
            manager.base_classes
        ),
        "class_order_original_ids": list(
            manager.class_order_original_ids
        ),
        "class_names_global_order": list(
            manager.target_names
        ),
        "processed_bands": int(
            processed_cube.shape[2]
        ),
        "raw_spectral_bands": int(
            ordered_cube.shape[2]
        ),
        "preprocessing_fit_pixel_count": int(
            preprocessing_state["fit_pixel_count"]
        ),
        "split_strategy": manager.split_strategy,
        "spatial_partition_mode": (
            manager.spatial_partition_mode
        ),
        "require_patch_disjoint": (
            manager.require_patch_disjoint
        ),
        "context_policy": manager.context_policy,
        "split_counts_by_global_class": {
            int(class_id): {
                split: int(
                    np.asarray(
                        protocol["split_by_class"]
                        [class_id][split]
                    ).size
                )
                for split in (
                    "train",
                    "val",
                    "test",
                )
            }
            for class_id in range(
                int(class_count)
            )
        },
    }
    return (
        manager,
        data_summary,
        preprocessing_state,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _mapping_value(
    mapping: Any,
    class_id: int,
) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    if class_id in mapping:
        return mapping[class_id]
    return mapping.get(str(class_id))


def _metric_value(
    metrics: Optional[Mapping[str, Any]],
    field: str,
    class_id: int,
) -> Any:
    if not isinstance(metrics, Mapping):
        return None
    return _mapping_value(
        metrics.get(field),
        class_id,
    )


def _phase_classwise_rows(
    *,
    phase: int,
    result: Mapping[str, Any],
    seen_ids: Sequence[int],
    new_ids: Sequence[int],
    class_names: Sequence[str],
) -> list[Dict[str, Any]]:
    """One cumulative classwise table per finalized phase.

    In incremental phases real TRAIN/VAL contain current classes only.
    Historical rows therefore intentionally have no real train/validation
    values.  Cumulative TEST contains every seen class.
    """
    phase = int(phase)
    seen = [int(v) for v in seen_ids]
    new = set(int(v) for v in new_ids)

    if phase == 0:
        train_metrics = result.get(
            "geometry_train"
        )
        validation_metrics = result.get(
            "geometry_validation"
        )
    else:
        train_metrics = result.get(
            "current_train_geometry"
        )
        validation_metrics = result.get(
            "current_validation_geometry"
        )
    test_metrics = result.get(
        "geometry_test"
    )
    if not isinstance(test_metrics, Mapping):
        raise RuntimeError(
            "finalized phase result lacks cumulative test metrics"
        )

    metric_fields = (
        ("accuracy", "per_class_accuracy"),
        (
            "own_cell_coverage",
            "per_class_true_cell_coverage",
        ),
        (
            "true_pair_violation_rate",
            "per_class_true_pair_violation_rate",
        ),
        (
            "no_cell_rate",
            "per_class_no_cell_rate",
        ),
        (
            "rival_cell_invasion_rate",
            "per_class_rival_cell_invasion_rate",
        ),
        (
            "minimum_true_pair_margin",
            "per_class_mean_minimum_true_pair_margin",
        ),
        (
            "mean_decision_margin",
            "per_class_mean_decision_margin",
        ),
    )

    rows: list[Dict[str, Any]] = []
    for class_id in seen:
        row: Dict[str, Any] = {
            "phase": phase,
            "class_id_internal": class_id,
            "class_id_display": class_id + 1,
            "class_name": str(
                class_names[class_id]
            ),
            "phase_role": (
                "base"
                if phase == 0
                else (
                    "new"
                    if class_id in new
                    else "old"
                )
            ),
        }

        # Real current TRAIN/VAL exist only for classes introduced now.
        expose_current = (
            phase == 0
            or class_id in new
        )
        for output_name, source_name in metric_fields:
            row[f"train_{output_name}"] = (
                _metric_value(
                    train_metrics,
                    source_name,
                    class_id,
                )
                if expose_current
                else None
            )
            row[f"validation_{output_name}"] = (
                _metric_value(
                    validation_metrics,
                    source_name,
                    class_id,
                )
                if expose_current
                else None
            )
            row[f"test_{output_name}"] = (
                _metric_value(
                    test_metrics,
                    source_name,
                    class_id,
                )
            )
        rows.append(row)
    return rows


def _structural_geometry(
    *,
    model: NECILModel,
    class_ids: Sequence[int],
    target_names: Sequence[str],
) -> Dict[str, Any]:
    ids = [int(v) for v in class_ids]
    structural = geometry_diagnostics(
        model,
        ids,
        target_names=target_names,
    )
    expected_pairs = (
        len(ids) * (len(ids) - 1) // 2
    )
    if int(
        structural.get("pair_count", -1)
    ) != expected_pairs:
        raise RuntimeError(
            "committed pairwise geometry is incomplete: "
            f"expected {expected_pairs} pairs, "
            f"got {structural.get('pair_count')}"
        )
    if (
        structural.get(
            "strict_interior_overlap"
        )
        != "impossible by construction"
    ):
        raise RuntimeError(
            "committed geometry does not expose the "
            "shared-boundary non-overlap invariant"
        )
    return structural


def _percentage(
    metrics: Mapping[str, Any],
    key: str,
) -> float:
    value = metrics.get(key)
    return (
        float("nan")
        if value is None
        else 100.0 * float(value)
    )


def _scalar(
    metrics: Mapping[str, Any],
    key: str,
) -> float:
    value = metrics.get(key)
    return (
        float("nan")
        if value is None
        else float(value)
    )



def _base_geometry_readiness(
    *,
    result: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> Dict[str, Any]:
    """Separate hard architectural checks from empirical geometry quality.

    Pair exposure is taken from the actual shuffled training epochs, not from a
    deterministic evaluation loader whose minibatch class composition is
    irrelevant to whether a pair was trained.
    """
    train = result["geometry_train"]
    validation = result["geometry_validation"]
    test = result["geometry_test"]

    class_count = int(structural["class_count"])
    expected_pairs = class_count * (class_count - 1) // 2
    committed_pairs = int(structural["pair_count"])

    history = list(result.get("history", []))
    if not history:
        raise RuntimeError("base result lacks optimization history")
    epoch_coverages = [
        float(record["pair_coverage"])
        for record in history
    ]
    if not all(math.isfinite(value) for value in epoch_coverages):
        raise RuntimeError("training pair coverage contains NaN/Inf")

    minimum_epoch_pair_coverage = min(epoch_coverages)
    final_epoch_pair_coverage = epoch_coverages[-1]

    hard_checks = {
        "complete_committed_pair_set": committed_pairs == expected_pairs,
        "all_base_pairs_seen_in_every_training_epoch": (
            minimum_epoch_pair_coverage == 1.0
        ),
        "zero_train_strict_cell_conflict": (
            float(train.get("strict_cell_conflict_rate", float("nan"))) == 0.0
        ),
        "zero_validation_strict_cell_conflict": (
            float(validation.get("strict_cell_conflict_rate", float("nan"))) == 0.0
        ),
        "zero_test_strict_cell_conflict": (
            float(test.get("strict_cell_conflict_rate", float("nan"))) == 0.0
        ),
    }
    hard_checks_passed = all(hard_checks.values())

    quality = {
        split: {
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "minimum_class_accuracy": float(metrics["minimum_class_accuracy"]),
            "true_pair_violation_rate": float(
                metrics["true_pair_violation_rate"]
            ),
            "macro_true_pair_violation_rate": float(
                metrics["macro_true_pair_violation_rate"]
            ),
            "no_cell_rate": float(metrics["no_cell_rate"]),
            "true_cell_coverage": float(metrics["true_cell_coverage"]),
            "mean_minimum_true_pair_margin": float(
                metrics["mean_minimum_true_pair_margin"]
            ),
            "macro_mean_minimum_true_pair_margin": float(
                metrics["macro_mean_minimum_true_pair_margin"]
            ),
            "mean_decision_margin": float(metrics["mean_decision_margin"]),
            "macro_mean_decision_margin": float(
                metrics["macro_mean_decision_margin"]
            ),
        }
        for split, metrics in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }

    return {
        "hard_architectural_checks_passed": hard_checks_passed,
        # Backward-compatible field name.  It means structural readiness only;
        # empirical quality must still be judged from the reported metrics.
        "structurally_ready_for_incremental_experiment": hard_checks_passed,
        "hard_checks": hard_checks,
        "training_pair_coverage": {
            "minimum_across_epochs": minimum_epoch_pair_coverage,
            "final_epoch": final_epoch_pair_coverage,
        },
        "quality_metrics": quality,
        "decision_policy": (
            "Hard checks verify executable geometry invariants only. "
            "No empirical accuracy/overlap threshold is hard-coded. "
            "Minimum-class accuracy, classwise pair violations and margins "
            "must still be inspected before starting incremental learning."
        ),
    }

def _report_base_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    trainer: Trainer,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    class_ids = [
        int(value)
        for value in result["class_ids"]
    ]
    phase = 0
    phase_dir = os.path.join(
        args.save_dir, "phase_0"
    )
    os.makedirs(phase_dir, exist_ok=True)

    classwise_path = save_csv(
        os.path.join(
            phase_dir,
            "classwise_metrics.csv",
        ),
        _phase_classwise_rows(
            phase=0,
            result=result,
            seen_ids=class_ids,
            new_ids=class_ids,
            class_names=dataset.target_names,
        ),
    )

    structural = _structural_geometry(
        model=model,
        class_ids=class_ids,
        target_names=dataset.target_names,
    )
    readiness = _base_geometry_readiness(
        result=result,
        structural=structural,
    )

    report = {
        "phase": 0,
        "seen_class_ids": class_ids,
        "representation": {
            "primary_signal": (
                "ordered full-band center spectrum"
            ),
            "context_signal": (
                "center-relative processed HSI patch"
            ),
            "context_role": (
                "residual refinement of the spectral representation"
            ),
            "representation_dim": int(
                model.backbone.representation_dim
            ),
            "class_representation": (
                "intersection of shared pairwise affine half-spaces"
            ),
            "class_score": (
                "E_c(z) = -min rival-oriented signed boundary distance"
            ),
            "decision": "argmin_c E_c(z)",
        },
        "objective": {
            "classification": (
                "class-uniform cross-entropy over -E"
            ),
            "separation": (
                "pair-balanced distribution separation along each "
                "learned base-base boundary direction"
            ),
            "risk_reduction": (
                "class-uniform CE; each separation pair balances its two sides"
            ),
        },
        "train": result["geometry_train"],
        "validation": result[
            "geometry_validation"
        ],
        "test": result["geometry_test"],
        "geometry_state": result[
            "geometry_state"
        ],
        "spectral_variation_state": result.get(
            "spectral_variation_state",
            result.get("spectral_replay_state"),
        ),
        "structural_geometry": structural,
        "base_geometry_readiness": readiness,
        "optimization": result.get(
            "geometry_summary", {}
        ),
        "artifacts": {
            "classwise_metrics": classwise_path,
        },
    }

    report_path = save_json(
        os.path.join(
            phase_dir,
            "geometry_report.json",
        ),
        report,
    )

    # These diagnostic figures require complete real TRAIN/VAL coverage of
    # every class and therefore belong to base phase only.
    figure_dir = os.path.join(
        phase_dir, "figures"
    )
    geometry_figures = (
        save_geometry_diagnostic_figures(
            output_dir=os.path.join(
                figure_dir, "geometry"
            ),
            train=result["geometry_train"],
            validation=result[
                "geometry_validation"
            ],
            test=result["geometry_test"],
            structural_geometry=structural,
            class_ids=class_ids,
            target_names=dataset.target_names,
        )
    )

    qualitative_maps = (
        generate_base_qualitative_maps(
            model=model,
            dataset=dataset,
            output_dir=os.path.join(
                figure_dir, "qualitative"
            ),
            device=args.device,
            batch_size=int(
                args.eval_batch_size
            ),
        )
    )

    report["artifacts"].update(
        {
            "geometry_figures": geometry_figures,
            "qualitative_maps": qualitative_maps,
        }
    )
    save_json(report_path, report)

    summary = {
        "phase": 0,
        "seen_class_ids": class_ids,
        "train": result["geometry_train"],
        "validation": result[
            "geometry_validation"
        ],
        "test": result["geometry_test"],
        "base_geometry_readiness": readiness,
        "checkpoint": result["checkpoint"],
        "trainer_report": result["report"],
        "geometry_report": report_path,
        "classwise_metrics": classwise_path,
        "geometry_figures": geometry_figures,
        "qualitative_maps": qualitative_maps,
    }
    summary_path = save_json(
        os.path.join(
            phase_dir,
            "phase_summary.json",
        ),
        summary,
    )
    summary["phase_summary"] = summary_path

    validation = result[
        "geometry_validation"
    ]
    test = result["geometry_test"]
    print("\nPhase 0 finalized.")
    for name, metrics in (
        ("Validation", validation),
        ("Test", test),
    ):
        print(
            f"{name:<10} | "
            f"OA={_percentage(metrics, 'accuracy'):.2f}% | "
            f"BA={_percentage(metrics, 'balanced_accuracy'):.2f}% | "
            f"MinClass={_percentage(metrics, 'minimum_class_accuracy'):.2f}% | "
            f"CellCov={_percentage(metrics, 'true_cell_coverage'):.2f}% | "
            f"MacroCellCov={_percentage(metrics, 'macro_true_cell_coverage'):.2f}% | "
            f"PairViol={_percentage(metrics, 'true_pair_violation_rate'):.2f}% | "
            f"NoCell={_percentage(metrics, 'no_cell_rate'):.2f}% | "
            f"RivalInv={_percentage(metrics, 'rival_cell_invasion_rate'):.2f}% | "
            f"MinPairM={_scalar(metrics, 'mean_minimum_true_pair_margin'):.4f} | "
            f"Margin={_scalar(metrics, 'mean_decision_margin'):.4f}"
        )
    print(
        "Base readiness  : "
        + (
            "STRUCTURALLY READY"
            if readiness["structurally_ready_for_incremental_experiment"]
            else "NOT READY"
        )
    )
    if not readiness["structurally_ready_for_incremental_experiment"]:
        failed = [
            name
            for name, passed in readiness["hard_checks"].items()
            if not passed
        ]
        print(f"Failed checks    : {failed}")
    print(
        f"Checkpoint      : {result['checkpoint']}"
    )
    print(
        f"Geometry report : {report_path}"
    )
    return summary



def _incremental_geometry_contract(
    *,
    result: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verify phase-t combinatorics without hardcoding any class count."""
    old_ids = [int(value) for value in result["old_class_ids"]]
    new_ids = [int(value) for value in result["new_class_ids"]]
    seen_ids = [int(value) for value in result["seen_class_ids"]]

    old_count = len(old_ids)
    new_count = len(new_ids)
    seen_count = len(seen_ids)

    expected_old_pairs = old_count * (old_count - 1) // 2
    expected_candidate_pairs = (
        old_count * new_count
        + new_count * (new_count - 1) // 2
    )
    expected_seen_pairs = seen_count * (seen_count - 1) // 2
    expected_old_new_relations = old_count * new_count
    expected_response_width = old_count - 1

    final_epoch = result.get("final_epoch_report", {})
    final_candidate_count = int(
        final_epoch.get("candidate_pair_count", -1)
    )
    final_covered_count = int(
        final_epoch.get("covered_candidate_pair_count", -1)
    )
    final_pair_coverage = float(
        final_epoch.get("pair_coverage", float("nan"))
    )

    replay = result.get("replay_diagnostics", {})
    boundary_selection = (
        replay.get("boundary_selection", {})
        if isinstance(replay, Mapping)
        else {}
    )
    reported_response_width = (
        boundary_selection.get("historical_response_width")
        if isinstance(boundary_selection, Mapping)
        else None
    )
    reported_old_new_relations = (
        boundary_selection.get("old_new_pair_count")
        if isinstance(boundary_selection, Mapping)
        else None
    )

    actual_seen_pairs = int(structural.get("pair_count", -1))

    hard_checks = {
        "seen_partition_is_old_plus_new": seen_ids == old_ids + new_ids,
        "complete_committed_seen_pair_set": (
            actual_seen_pairs == expected_seen_pairs
        ),
        "correct_candidate_pair_count": (
            final_candidate_count == expected_candidate_pairs
        ),
        "all_candidate_pairs_exercised": (
            final_covered_count == expected_candidate_pairs
            and math.isfinite(final_pair_coverage)
            and final_pair_coverage == 1.0
        ),
        "historical_response_width_matches_old_rivals": (
            reported_response_width is None
            or int(reported_response_width) == expected_response_width
        ),
        "boundary_selection_covers_old_new_relations": (
            reported_old_new_relations is None
            or int(reported_old_new_relations) == expected_old_new_relations
        ),
    }

    return {
        "hard_architectural_checks_passed": all(hard_checks.values()),
        "hard_checks": hard_checks,
        "class_counts": {
            "old": old_count,
            "new": new_count,
            "seen": seen_count,
        },
        "pair_count_contract": {
            "old_committed_before_phase": expected_old_pairs,
            "candidate_old_new_plus_new_new": expected_candidate_pairs,
            "old_new_relations": expected_old_new_relations,
            "committed_after_phase": expected_seen_pairs,
        },
        "historical_response_width": {
            "expected": expected_response_width,
            "reported": reported_response_width,
        },
        "final_training_pair_coverage": {
            "covered": final_covered_count,
            "candidate": final_candidate_count,
            "coverage": final_pair_coverage,
        },
    }



def _report_incremental_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    phase = int(result["phase"])
    old_ids = [
        int(v)
        for v in result["old_class_ids"]
    ]
    new_ids = [
        int(v)
        for v in result["new_class_ids"]
    ]
    seen_ids = [
        int(v)
        for v in result["seen_class_ids"]
    ]
    if seen_ids != old_ids + new_ids:
        raise RuntimeError(
            "incremental result contains an invalid class partition"
        )

    phase_dir = os.path.join(
        args.save_dir, f"phase_{phase}"
    )
    os.makedirs(phase_dir, exist_ok=True)

    classwise_path = save_csv(
        os.path.join(
            phase_dir,
            "classwise_metrics.csv",
        ),
        _phase_classwise_rows(
            phase=phase,
            result=result,
            seen_ids=seen_ids,
            new_ids=new_ids,
            class_names=dataset.target_names,
        ),
    )

    structural = _structural_geometry(
        model=model,
        class_ids=seen_ids,
        target_names=dataset.target_names,
    )
    incremental_contract = _incremental_geometry_contract(
        result=result,
        structural=structural,
    )
    if not incremental_contract["hard_architectural_checks_passed"]:
        failed = [
            name
            for name, passed
            in incremental_contract["hard_checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "finalized incremental phase violates its geometry contract: "
            f"{failed}"
        )

    figure_dir = os.path.join(
        phase_dir,
        "figures",
    )
    incremental_geometry_figures = (
        save_incremental_geometry_diagnostic_figures(
            output_dir=os.path.join(
                figure_dir,
                "geometry",
            ),
            test=result["geometry_test"],
            old_test=result["old_test"],
            new_test=result["new_test"],
            boundary_preservation=result[
                "boundary_preservation"
            ],
            class_ids=seen_ids,
            old_class_ids=old_ids,
            new_class_ids=new_ids,
            target_names=dataset.target_names,
        )
    )

    qualitative_maps = generate_phase_qualitative_maps(
        model=model,
        dataset=dataset,
        phase=phase,
        output_dir=os.path.join(
            figure_dir,
            "qualitative",
        ),
        device=args.device,
        batch_size=int(args.eval_batch_size),
    )

    report = {
        "phase": phase,
        "old_class_ids": old_ids,
        "new_class_ids": new_ids,
        "seen_class_ids": seen_ids,
        "continual_roles": {
            "representation_drift": (
                "selected historical HSI supports preserve their "
                "class-incident old boundary responses as the backbone evolves"
            ),
            "boundary_preservation": (
                "old-old pairwise boundaries remain committed and their "
                "historical semantic responses are preserved"
            ),
            "boundary_extension": (
                "only old-new and new-new boundaries are learned"
            ),
            "feature_space_overlap": (
                "real-new and boundary-selected replay-old distributions "
                "are explicitly separated along candidate pair directions"
            ),
            "classifier_balance": (
                "all seen classes use the same parameter-free energy rule; "
                "pair separation balances both class sides"
            ),
        },
        "replay": result[
            "replay_diagnostics"
        ],
        "replay_start_geometry": result.get(
            "replay_start_geometry"
        ),
        "old_replay_after_training_against_old_geometry": (
            result["old_replay_old_geometry"]
        ),
        "old_replay_after_training_against_seen_geometry": (
            result["old_replay_seen_geometry"]
        ),
        "real_new_train": result[
            "current_train_geometry"
        ],
        "real_new_validation": result[
            "current_validation_geometry"
        ],
        "cumulative_test": result[
            "geometry_test"
        ],
        "old_test": result["old_test"],
        "new_test": result["new_test"],
        "harmonic_old_new_accuracy": result[
            "harmonic_old_new_accuracy"
        ],
        "boundary_preservation": result[
            "boundary_preservation"
        ],
        "geometry_state": result[
            "geometry_state"
        ],
        "spectral_variation_state": result.get(
            "spectral_variation_state",
            result.get("spectral_replay_state"),
        ),
        "structural_geometry": structural,
        "incremental_geometry_contract": incremental_contract,
        "optimization": result.get(
            "phase_summary", {}
        ),
        "artifacts": {
            "classwise_metrics": classwise_path,
            "geometry_figures": incremental_geometry_figures,
            "qualitative_maps": qualitative_maps,
        },
    }

    report_path = save_json(
        os.path.join(
            phase_dir,
            "geometry_report.json",
        ),
        report,
    )

    summary = {
        "phase": phase,
        "old_class_ids": old_ids,
        "new_class_ids": new_ids,
        "seen_class_ids": seen_ids,
        "test": result["geometry_test"],
        "old_test": result["old_test"],
        "new_test": result["new_test"],
        "harmonic_old_new_accuracy": result[
            "harmonic_old_new_accuracy"
        ],
        "boundary_preservation": result[
            "boundary_preservation"
        ],
        "incremental_geometry_contract": incremental_contract,
        "replay": result[
            "replay_diagnostics"
        ],
        "checkpoint": result["checkpoint"],
        "trainer_report": result["report"],
        "geometry_report": report_path,
        "classwise_metrics": classwise_path,
        "geometry_figures": incremental_geometry_figures,
        "qualitative_maps": qualitative_maps,
    }
    summary_path = save_json(
        os.path.join(
            phase_dir,
            "phase_summary.json",
        ),
        summary,
    )
    summary["phase_summary"] = summary_path

    test = result["geometry_test"]
    old_test = result["old_test"]
    new_test = result["new_test"]

    print(f"\nPhase {phase} finalized.")
    print(
        f"Cumulative | "
        f"OA={_percentage(test, 'accuracy'):.2f}% | "
        f"BA={_percentage(test, 'balanced_accuracy'):.2f}% | "
        f"MinClass={_percentage(test, 'minimum_class_accuracy'):.2f}% | "
        f"CellCov={_percentage(test, 'true_cell_coverage'):.2f}% | "
        f"PairViol={_percentage(test, 'macro_true_pair_violation_rate'):.2f}% | "
        f"NoCell={_percentage(test, 'macro_no_cell_rate'):.2f}% | "
        f"MacroInv={_percentage(test, 'macro_rival_cell_invasion_rate'):.2f}% | "
        f"Margin={_scalar(test, 'macro_mean_decision_margin'):.4f}"
    )
    print(
        f"Old/New   | "
        f"OldBA={100.0 * float(old_test['balanced_accuracy']):.2f}% | "
        f"NewBA={100.0 * float(new_test['balanced_accuracy']):.2f}% | "
        f"H={100.0 * float(result['harmonic_old_new_accuracy']):.2f}%"
    )
    preservation = result[
        "boundary_preservation"
    ]
    print(
        f"Old Δ     | "
        f"BA={100.0 * float(preservation['old_balanced_accuracy_delta']):+.2f} pp | "
        f"CellCov={100.0 * float(preservation['old_cell_coverage_delta']):+.2f} pp | "
        f"PairViol={100.0 * float(preservation['old_pair_violation_delta']):+.2f} pp | "
        f"NoCell={100.0 * float(preservation['old_no_cell_rate_delta']):+.2f} pp | "
        f"RivalInv={100.0 * float(preservation['old_rival_invasion_delta']):+.2f} pp | "
        f"Margin={float(preservation['old_decision_margin_delta']):+.4f}"
    )

    drift = preservation.get(
        "selected_replay_historical_response_drift", {}
    )
    if isinstance(drift, Mapping):
        mean_drift = drift.get("mean_absolute_drift")
        max_drift = drift.get("max_absolute_drift")
        if mean_drift is not None and max_drift is not None:
            print(
                f"Hist drift | "
                f"MeanAbs={float(mean_drift):.6f} | "
                f"MaxAbs={float(max_drift):.6f}"
            )

    pair_contract = incremental_contract["pair_count_contract"]
    coverage = incremental_contract["final_training_pair_coverage"]
    print(
        f"Geometry   | "
        f"OldPairs={pair_contract['old_committed_before_phase']} | "
        f"CandidatePairs={pair_contract['candidate_old_new_plus_new_new']} | "
        f"OldNew={pair_contract['old_new_relations']} | "
        f"SeenPairs={pair_contract['committed_after_phase']} | "
        f"Pairs={coverage['covered']}/{coverage['candidate']} | "
        f"ResponseWidth={incremental_contract['historical_response_width']['expected']}"
    )
    print(
        f"Checkpoint      : {result['checkpoint']}"
    )
    print(
        f"Geometry report : {report_path}"
    )
    print(
        f"Figures         : {figure_dir}"
    )
    return summary


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    os.makedirs(
        args.save_dir, exist_ok=True
    )

    (
        dataset,
        data_summary,
        preprocessing_state,
    ) = _build_data(args)

    data_protocol_path = save_json(
        os.path.join(
            args.save_dir,
            "data_protocol.json",
        ),
        data_summary,
    )
    preprocessor_path = os.path.join(
        args.save_dir,
        "phase_0_preprocessor.npz",
    )
    SaveHSIPreprocessor(
        preprocessor_path,
        preprocessing_state,
    )
    # Trainer loads this exact persisted base preprocessing state.  Do not
    # pass preprocessing_state into Trainer; its runtime constructor is
    # intentionally Trainer(model, dataset, args).
    experiment_config_path = save_json(
        os.path.join(
            args.save_dir,
            "experiment_config.json",
        ),
        vars(args),
    )

    model = NECILModel(args)
    trainer = Trainer(
        model,
        dataset,
        args,
    )

    resume = str(
        args.resume_checkpoint or ""
    ).strip()
    restored_phase: Optional[int] = None
    if resume:
        payload = trainer.load_checkpoint(
            resume
        )
        restored_phase = int(
            payload["phase"]
        )

    total_phases = len(
        trainer.phase_schedule
    )
    if total_phases <= 0:
        raise RuntimeError(
            "continual protocol contains no phases"
        )

    phase_artifacts: Dict[int, Dict[str, Any]] = {}

    if args.run_mode == "base":
        result = trainer.run_base_only(
            epochs=int(args.epochs_base),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
        )
        phase_artifacts[0] = (
            _report_base_phase(
                args=args,
                dataset=dataset,
                model=model,
                trainer=trainer,
                result=result,
            )
        )

    else:
        # Fresh "all" run starts with base.  Incremental mode and resumed "all"
        # begin strictly after the checkpoint's finalized phase.
        if restored_phase is None:
            if args.run_mode != "all":
                raise RuntimeError(
                    "incremental execution requires a restored checkpoint"
                )
            base_result = trainer.run_base_only(
                epochs=int(args.epochs_base),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
            )
            phase_artifacts[0] = (
                _report_base_phase(
                    args=args,
                    dataset=dataset,
                    model=model,
                    trainer=trainer,
                    result=base_result,
                )
            )
            first_incremental_phase = 1
        else:
            first_incremental_phase = (
                restored_phase + 1
            )

        incremental_lr = (
            float(args.lr)
            if args.lr_inc is None
            else float(args.lr_inc)
        )

        if first_incremental_phase >= total_phases:
            raise RuntimeError(
                "checkpoint already represents the final protocol phase; "
                "there is no incremental phase left to run"
            )

        if args.run_mode == "incremental":
            phases_to_run = [first_incremental_phase]
        elif args.run_mode == "all":
            phases_to_run = list(
                range(first_incremental_phase, total_phases)
            )
        else:
            raise RuntimeError(
                f"unexpected non-base run mode {args.run_mode!r}"
            )

        for phase in phases_to_run:
            old_ids = [
                int(value)
                for value in dataset.get_old_classes(phase)
            ]
            new_ids = [
                int(value)
                for value in dataset.get_new_classes(phase)
            ]
            old_pairs = len(old_ids) * (len(old_ids) - 1) // 2
            candidate_pairs = (
                len(old_ids) * len(new_ids)
                + len(new_ids) * (len(new_ids) - 1) // 2
            )
            seen_count = len(old_ids) + len(new_ids)
            seen_pairs = seen_count * (seen_count - 1) // 2
            print(
                f"\nStarting incremental phase {phase}: "
                f"old={len(old_ids)}, new={len(new_ids)}, "
                f"old_pairs={old_pairs}, "
                f"candidate_pairs={candidate_pairs}, "
                f"expected_seen_pairs={seen_pairs}"
            )

            result = trainer.run_incremental_phase(
                phase=phase,
                epochs=int(args.epochs_inc),
                batch_size=int(args.batch_size),
                lr=incremental_lr,
            )
            phase_artifacts[phase] = _report_incremental_phase(
                args=args,
                dataset=dataset,
                model=model,
                result=result,
            )

    finalized_phases = [
        int(value)
        for value in dataset.finalized_phases
    ]
    if finalized_phases:
        last_phase = finalized_phases[-1]
        final_seen_ids = [
            int(v)
            for v in dataset.get_seen_classes(
                last_phase
            )
        ]
    else:
        last_phase = None
        final_seen_ids = []

    run_summary = {
        "seed": int(args.seed),
        "dataset": str(args.dataset),
        "run_mode": str(args.run_mode),
        "run_mode_semantics": {
            "base": "phase 0 only",
            "incremental": "exactly the next unfinalized phase",
            "all": "every remaining phase",
        },
        "resolved_incremental_objective_weights": {
            "classification": (
                float(args.base_classification_weight)
                if args.incremental_classification_weight is None
                else float(args.incremental_classification_weight)
            ),
            "separation": (
                float(args.base_separation_weight)
                if args.incremental_separation_weight is None
                else float(args.incremental_separation_weight)
            ),
            "preservation": float(args.preservation_weight),
        },
        "resume_checkpoint": (
            None if not resume else os.path.abspath(resume)
        ),
        "restored_phase": restored_phase,
        "phase_schedule": {
            int(phase): [
                int(v) for v in classes
            ]
            for phase, classes
            in trainer.phase_schedule.items()
        },
        "finalized_phases": finalized_phases,
        "final_seen_class_ids": final_seen_ids,
        "all_protocol_phases_completed": (
            finalized_phases
            == list(range(total_phases))
        ),
        "data_protocol": data_protocol_path,
        "preprocessor": os.path.abspath(
            preprocessor_path
        ),
        "experiment_config": (
            experiment_config_path
        ),
        "new_phase_artifacts": phase_artifacts,
        "history": trainer.history,
        "final_geometry_state": (
            trainer.geometry_state_summary()
            if finalized_phases else None
        ),
        "final_spectral_variation_state": (
            trainer.spectral_variation_bank.summary()
            if finalized_phases else None
        ),
    }
    summary_path = save_json(
        os.path.join(
            args.save_dir,
            "run_summary.json",
        ),
        run_summary,
    )
    run_summary["run_summary"] = summary_path

    print("\nRun complete.")
    print(
        f"Finalized phases : {finalized_phases}"
    )
    print(
        f"Seen classes     : {final_seen_ids}"
    )
    print(
        f"Run summary      : {summary_path}"
    )
    return run_summary


def main(
    argv: Optional[Sequence[str]] = None,
) -> None:
    args = build_parser().parse_args(
        argv
    )
    validate_args(args)
    set_seed(
        args.seed,
        args.deterministic,
    )
    args.device = resolve_device(
        args.device
    )
    args.save_dir = os.path.abspath(
        args.save_dir
    )
    run_experiment(args)


if __name__ == "__main__":
    main()
