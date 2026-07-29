from __future__ import annotations

"""Main entry point for transport-verified factor-geometry NECIL-HSI.

This driver executes the complete class-incremental protocol:

    phase 0:
        CE warm-up + cross-fitted factor-geometry shaping

    phase t > 0:
        controlled late-layer plasticity
        + analytical branchwise coordinate registration
        + exact old-row pushforward
        + current-query risk-guided factor-energy separation

After every accepted phase it saves:
* cumulative held-out test metrics;
* separate GT and predicted-GT maps;
* qualitative all-seen-labeled GT and prediction maps;
* confusion matrices and class-wise metrics;
* geometry pair-risk/overlap diagnostics;
* spectral-spatial factor-bank diagnostics;
* base or incremental training dynamics;
* cumulative NECIL forgetting/BWT summaries.

Preprocessing is fitted only on phase-0 training pixels and then frozen.
Incremental training loaders expose only current-phase classes. Cumulative
validation/test loaders expose only classes seen at the requested phase.
"""

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.hsi_dataloader_pytorch import (
    ExtractLabeledPixelIndex,
    FitHSIPreprocessor,
    LoadRawHSIData,
    SaveHSIPreprocessor,
)
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from utils.eval import (
        NECILEvaluator,
        evaluate_factor_geometry_loader,
        geometry_bank_diagnostics,
        geometry_sampling_health,
        save_classification_report,
        save_json,
    )

from utils.visualize import (
        build_geometry_overlap_review,
        plot_necil_phase_summary,
        plot_phase_training_dynamics,
        predict_phase_grid,
        save_classification_diagnostics,
        save_geometry_pair_diagnostics,
        save_spectral_spatial_geometry_diagnostics,
        save_transport_diagnostics,
    )


STACK_BUILD_ID = "HSI-FACTOR-GEOMETRY-NECIL-V2"
METHOD_NAME = "Transport-verified spectral-spatial factor geometry"
CLASSIFICATION_FACTORIZATION = "p(z|c)"
SPECTRAL_RELATION_FACTORIZATION = "p(h|c)"

DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "IP": {"name": "Indian Pines", "bands": 200, "classes": 16},
    "SA": {"name": "Salinas", "bands": 204, "classes": 16},
    "PU": {"name": "Pavia University", "bands": 103, "classes": 9},
    "PC": {"name": "Pavia Centre", "bands": 102, "classes": 9},
    "BS": {"name": "Botswana", "bands": 145, "classes": 14},
    "LK": {"name": "LongKou", "bands": 270, "classes": 9},
    "HH": {"name": "HongHu", "bands": 270, "classes": 22},
    "HC": {"name": "HanChuan", "bands": 274, "classes": 16},
    "UH13": {"name": "Houston 2013", "bands": 144, "classes": 15},
    "QUH": {"name": "QUH-Qingyun", "bands": 270, "classes": 6},
    "PI": {"name": "QUH-Pingan", "bands": 270, "classes": 10},
    "TH": {"name": "QUH-Tangdaowan", "bands": 270, "classes": 18},
}


# =============================================================================
# Generic configuration helpers
# =============================================================================


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on", "t"}:
        return True
    if token in {"0", "false", "no", "n", "off", "f", ""}:
        return False
    raise argparse.ArgumentTypeError(
        f"invalid boolean value: {value!r}"
    )


def parse_int_list(raw: Optional[str], *, name: str) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    values = [
        int(token.strip())
        for token in str(raw).split(",")
        if token.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates: {values}")
    return values


def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return copy.deepcopy(vars(args))


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json(path: str, value: Any) -> str:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(
            json_safe(value),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, absolute)
    return absolute


def set_seed(seed: int, deterministic: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def resolve_device(value: Any) -> str:
    token = str(value).strip().lower()
    if token == "gpu":
        token = "cuda"
    requested = torch.device(token)
    if requested.type == "cpu":
        return "cpu"
    if requested.type != "cuda":
        raise RuntimeError(
            f"unsupported device type {requested.type!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={value!r} requests CUDA, but CUDA is unavailable"
        )
    index = (
        torch.cuda.current_device()
        if requested.index is None
        else int(requested.index)
    )
    if not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(
            f"requested cuda:{index}, but only "
            f"{torch.cuda.device_count()} devices are visible"
        )
    torch.cuda.set_device(index)
    return f"cuda:{index}"


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict exemplar-free HSI class-incremental learning with "
            "transport-verified spectral-spatial factor geometry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("A. Incremental experiment")
    experiment.add_argument(
        "--dataset",
        type=str,
        default="IP",
        choices=sorted(DATASET_INFO),
    )
    experiment.add_argument("--data_dir", type=str, default="./datasets")
    experiment.add_argument(
        "--save_dir",
        type=str,
        default="./results_factor_geometry_necil",
    )
    experiment.add_argument("--patch_size", type=int, default=11)
    experiment.add_argument("--train_ratio", type=float, default=0.20)
    experiment.add_argument("--val_ratio", type=float, default=0.10)
    experiment.add_argument("--min_train_per_class", type=int, default=20)
    experiment.add_argument("--base_classes", type=int, default=6)
    experiment.add_argument("--increment", type=int, default=5)
    experiment.add_argument(
        "--class_order",
        type=str,
        default="",
        help=(
            "Comma-separated complete original class-ID permutation. "
            "Empty uses the natural order."
        ),
    )
    experiment.add_argument("--shuffle_order", type=str2bool, default=False)
    experiment.add_argument("--base_only", type=str2bool, default=False)
    experiment.add_argument(
        "--maximum_phase",
        type=int,
        default=-1,
        help="-1 executes every phase.",
    )
    experiment.add_argument("--epochs_base", type=int, default=80)
    experiment.add_argument("--epochs_inc", type=int, default=15)
    experiment.add_argument("--batch_size", type=int, default=64)
    experiment.add_argument("--eval_batch_size", type=int, default=128)
    experiment.add_argument("--lr", type=float, default=1e-4)
    experiment.add_argument("--lr_inc", type=float, default=1e-4)
    experiment.add_argument("--weight_decay", type=float, default=1e-4)
    experiment.add_argument("--resume_checkpoint", type=str, default="")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--num_runs", type=int, default=1)
    experiment.add_argument("--seed_list", type=str, default="")
    experiment.add_argument("--device", type=str, default="cuda:0")
    experiment.add_argument("--num_workers", type=int, default=0)
    experiment.add_argument("--deterministic", type=str2bool, default=False)

    preprocessing = parser.add_argument_group("B. Leakage-safe preprocessing")
    preprocessing.add_argument("--no_pca", action="store_true")
    preprocessing.add_argument("--reduction_method", type=str, default="PCA")
    preprocessing.add_argument("--pca_components", type=int, default=30)
    preprocessing.add_argument("--pca_whiten", type=str2bool, default=False)

    backbone = parser.add_argument_group("C. Dual-branch HSI backbone")
    backbone.add_argument("--d_model", type=int, default=128)
    backbone.add_argument("--spectral_feature_dim", type=int, default=32)
    backbone.add_argument("--spatial_feature_dim", type=int, default=32)
    backbone.add_argument("--d_state", type=int, default=16)
    backbone.add_argument("--d_conv", type=int, default=3)
    backbone.add_argument("--expand", type=int, default=2)
    backbone.add_argument("--num_layers", type=int, default=3)
    backbone.add_argument("--num_spectral_layers", type=int, default=3)
    backbone.add_argument("--num_spatial_layers", type=int, default=3)
    backbone.add_argument("--dropout", type=float, default=0.10)
    backbone.add_argument("--projection_dropout", type=float, default=0.0)
    backbone.add_argument("--backbone_norm", type=str, default="rms")
    backbone.add_argument("--stem_norm_groups", type=int, default=8)
    backbone.add_argument("--ssm_residual_scale_init", type=float, default=0.50)
    backbone.add_argument("--ssm_kernel_dropout", type=float, default=0.0)

    geometry = parser.add_argument_group("D. Factor GeometryBank")
    geometry.add_argument("--subspace_rank", type=int, default=5)
    geometry.add_argument("--spectral_resample_length", type=int, default=32)
    geometry.add_argument(
        "--spectral_derivative_weight",
        type=float,
        default=0.50,
    )
    geometry.add_argument("--geom_var_floor", type=float, default=1e-4)
    geometry.add_argument(
        "--geometry_variance_floor_relative",
        type=float,
        default=1e-3,
    )
    geometry.add_argument(
        "--geometry_minimum_variance_shrinkage",
        type=float,
        default=0.10,
    )
    geometry.add_argument(
        "--geometry_extra_rare_class_shrinkage",
        type=float,
        default=0.35,
    )
    geometry.add_argument("--geometry_shrinkage_tau", type=float, default=20.0)
    geometry.add_argument("--factor_fit_iterations", type=int, default=3)
    geometry.add_argument(
        "--whitened_eigen_excess_threshold",
        type=float,
        default=0.50,
    )
    geometry.add_argument(
        "--rank_energy_threshold",
        type=float,
        default=0.95,
    )
    geometry.add_argument(
        "--rank_eigen_ratio_threshold",
        type=float,
        default=1e-3,
    )
    geometry.add_argument(
        "--geometry_volume_weight",
        type=float,
        default=0.50,
    )
    geometry.add_argument("--energy_temperature", type=float, default=1.0)
    geometry.add_argument(
        "--robust_geometry_iterations",
        type=int,
        default=3,
    )
    geometry.add_argument(
        "--robust_geometry_huber_delta",
        type=float,
        default=2.5,
    )
    geometry.add_argument("--spatial_purity_power", type=float, default=1.0)
    geometry.add_argument(
        "--geometry_reliability_sample_strength",
        type=float,
        default=20.0,
    )
    geometry.add_argument(
        "--maximum_geometry_reconstruction_error",
        type=float,
        default=0.75,
    )
    geometry.add_argument(
        "--minimum_geometry_effective_dimension",
        type=float,
        default=1.25,
    )

    base = parser.add_argument_group("E. Base factor-geometry training")
    base.add_argument("--base_ce_warmup_epochs", type=int, default=10)
    base.add_argument("--base_geometry_ramp_epochs", type=int, default=10)
    base.add_argument("--base_geometry_weight", type=float, default=1.0)
    base.add_argument("--base_geometry_margin", type=float, default=0.50)
    base.add_argument("--base_risk_strength", type=float, default=0.50)
    base.add_argument(
        "--base_geometry_temperature",
        type=float,
        default=0.50,
    )
    base.add_argument(
        "--base_maximum_pair_margin",
        type=float,
        default=None,
    )
    base.add_argument("--geometry_batch_classes", type=int, default=0)
    base.add_argument("--geometry_samples_per_class", type=int, default=8)
    base.add_argument("--base_steps_per_epoch", type=int, default=0)
    base.add_argument(
        "--base_min_samples_per_class_in_batch",
        type=int,
        default=6,
    )
    base.add_argument("--base_oof_folds", type=int, default=3)
    base.add_argument("--base_eval_every", type=int, default=1)
    base.add_argument("--base_early_stop_patience", type=int, default=0)
    base.add_argument(
        "--base_use_geometry_balanced_sampler",
        type=str2bool,
        default=True,
    )
    base.add_argument("--base_class_balance", type=str2bool, default=True)
    base.add_argument(
        "--base_use_cosine_scheduler",
        type=str2bool,
        default=True,
    )
    base.add_argument("--base_min_lr_ratio", type=float, default=0.05)
    base.add_argument("--label_smoothing", type=float, default=0.0)
    base.add_argument("--grad_clip_base", type=float, default=5.0)
    base.add_argument(
        "--base_diagnostic_samples_per_class",
        type=int,
        default=32,
    )
    base.add_argument("--base_admission_enforce", type=str2bool, default=False)
    base.add_argument(
        "--base_min_factor_gain_over_prototype",
        type=float,
        default=0.0,
    )
    base.add_argument(
        "--base_min_factor_gain_over_diagonal",
        type=float,
        default=0.0,
    )
    base.add_argument(
        "--base_admission_min_class_accuracy",
        type=float,
        default=0.0,
    )
    base.add_argument(
        "--base_admission_max_classification_violation",
        type=float,
        default=1.0,
    )

    incremental = parser.add_argument_group(
        "F. Incremental geometry transport"
    )
    incremental.add_argument(
        "--incremental_controlled_plasticity",
        type=str2bool,
        default=True,
    )
    incremental.add_argument(
        "--incremental_crossfit_folds",
        type=int,
        default=3,
    )
    incremental.add_argument(
        "--incremental_steps_per_epoch",
        type=int,
        default=3,
    )
    incremental.add_argument(
        "--incremental_geometry_weight",
        type=float,
        default=1.0,
    )
    incremental.add_argument(
        "--incremental_coordinate_weight",
        type=float,
        default=0.10,
    )
    incremental.add_argument(
        "--incremental_parameter_trust_weight",
        type=float,
        default=0.0,
    )
    incremental.add_argument(
        "--incremental_base_margin",
        type=float,
        default=0.50,
    )
    incremental.add_argument(
        "--incremental_risk_strength",
        type=float,
        default=0.50,
    )
    incremental.add_argument(
        "--incremental_geometry_temperature",
        type=float,
        default=0.50,
    )
    incremental.add_argument(
        "--incremental_maximum_pair_margin",
        type=float,
        default=None,
    )
    incremental.add_argument(
        "--incremental_weight_decay",
        type=float,
        default=1e-4,
    )
    incremental.add_argument(
        "--incremental_grad_clip",
        type=float,
        default=5.0,
    )
    incremental.add_argument(
        "--incremental_use_cosine_scheduler",
        type=str2bool,
        default=True,
    )
    incremental.add_argument(
        "--incremental_min_lr_ratio",
        type=float,
        default=0.10,
    )
    incremental.add_argument(
        "--incremental_admission_enforce",
        type=str2bool,
        default=False,
    )
    incremental.add_argument(
        "--incremental_max_transport_closure_error",
        type=float,
        default=1e-5,
    )
    incremental.add_argument(
        "--incremental_max_transport_normalized_rmse",
        type=float,
        default=1.50,
    )

    transport = parser.add_argument_group(
        "G. Evidence-gated branch registration"
    )
    transport.add_argument(
        "--transport_minimum_scale_samples",
        type=int,
        default=8,
    )
    transport.add_argument(
        "--transport_minimum_rotation_samples",
        type=int,
        default=None,
    )
    transport.add_argument(
        "--transport_minimum_rank_fraction",
        type=float,
        default=0.50,
    )
    transport.add_argument(
        "--transport_rank_tolerance",
        type=float,
        default=1e-4,
    )
    transport.add_argument(
        "--transport_minimum_relative_improvement",
        type=float,
        default=0.02,
    )
    transport.add_argument(
        "--transport_maximum_normalized_rmse",
        type=float,
        default=1.50,
    )
    transport.add_argument(
        "--transport_minimum_scale",
        type=float,
        default=0.70,
    )
    transport.add_argument(
        "--transport_maximum_scale",
        type=float,
        default=1.30,
    )

    output = parser.add_argument_group("H. Evaluation and visualization")
    output.add_argument("--energy_margin", type=float, default=0.0)
    output.add_argument("--save_raw_arrays", type=str2bool, default=False)
    output.add_argument("--save_model_summary", type=str2bool, default=True)
    output.add_argument("--save_visualizations", type=str2bool, default=True)
    output.add_argument("--save_test_maps", type=str2bool, default=True)
    output.add_argument("--save_all_labeled_maps", type=str2bool, default=True)
    output.add_argument("--save_combined_maps", type=str2bool, default=True)
    output.add_argument("--save_map_numpy", type=str2bool, default=True)
    output.add_argument("--visualization_batch_size", type=int, default=256)
    output.add_argument("--visualization_dpi", type=int, default=300)
    output.add_argument("--viz_class_cmap", type=str, default="nipy_spectral")
    output.add_argument("--viz_background_color", type=str, default="#20252B")
    output.add_argument("--debug_verbose", type=str2bool, default=False)
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    resolved = argparse.Namespace(**namespace_to_dict(args))
    resolved.device = resolve_device(resolved.device)
    resolved.strict_non_exemplar = True
    return resolved


def validate_config(
    args: argparse.Namespace,
    *,
    num_classes: Optional[int] = None,
) -> None:
    if int(args.patch_size) <= 0 or int(args.patch_size) % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    if not 0.0 < float(args.train_ratio) < 1.0:
        raise ValueError("train_ratio must lie in (0,1)")
    if not 0.0 < float(args.val_ratio) < 1.0:
        raise ValueError("val_ratio must lie in (0,1)")
    if float(args.train_ratio) + float(args.val_ratio) >= 1.0:
        raise ValueError(
            "train_ratio + val_ratio must be smaller than one"
        )
    if int(args.base_classes) < 2:
        raise ValueError("base_classes must be at least two")
    if int(args.increment) <= 0:
        raise ValueError("increment must be positive")
    if int(args.epochs_base) <= 0 or int(args.epochs_inc) <= 0:
        raise ValueError("base and incremental epochs must be positive")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if float(args.lr) <= 0.0 or float(args.lr_inc) <= 0.0:
        raise ValueError("learning rates must be positive")
    if int(args.spectral_feature_dim) <= 0:
        raise ValueError("spectral_feature_dim must be positive")
    if int(args.spatial_feature_dim) <= 0:
        raise ValueError("spatial_feature_dim must be positive")
    feature_dim = (
        int(args.spectral_feature_dim)
        + int(args.spatial_feature_dim)
    )
    if not 0 <= int(args.subspace_rank) <= feature_dim:
        raise ValueError(
            f"subspace_rank must lie in [0,{feature_dim}]"
        )
    if (
        int(args.geometry_samples_per_class) < 6
        or int(args.geometry_samples_per_class) % 2 != 0
    ):
        raise ValueError(
            "geometry_samples_per_class must be even and at least six"
        )
    if int(args.incremental_crossfit_folds) < 3:
        raise ValueError(
            "incremental_crossfit_folds must be at least three"
        )
    if int(args.min_train_per_class) < (
        3 * int(args.incremental_crossfit_folds)
    ):
        raise ValueError(
            "min_train_per_class must provide at least three samples "
            "per incremental cross-fit role"
        )
    if not 0.0 <= float(args.label_smoothing) < 1.0:
        raise ValueError("label_smoothing must lie in [0,1)")
    if int(args.visualization_batch_size) <= 0:
        raise ValueError("visualization_batch_size must be positive")
    if int(args.visualization_dpi) <= 0:
        raise ValueError("visualization_dpi must be positive")
    if num_classes is not None:
        if not 2 <= int(args.base_classes) <= int(num_classes):
            raise ValueError(
                f"base_classes must lie in [2,{num_classes}]"
            )
        order = parse_int_list(
            args.class_order,
            name="class_order",
        )
        if order is not None:
            if len(order) != int(num_classes):
                raise ValueError(
                    "class_order must contain a complete class permutation"
                )
            if set(order) != set(range(int(num_classes))):
                raise ValueError(
                    "class_order must be a permutation of original IDs"
                )
    seeds = parse_int_list(args.seed_list, name="seed_list")
    if seeds is not None and len(seeds) != int(args.num_runs):
        raise ValueError(
            "seed_list length must equal num_runs"
        )
    if str(args.resume_checkpoint).strip() and int(args.num_runs) != 1:
        raise ValueError(
            "resume_checkpoint is supported only for one run"
        )


# =============================================================================
# Incremental protocol and leakage-safe lazy patch manager
# =============================================================================


def _split_one_class(
    indices: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    minimum_train: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size < 3:
        raise RuntimeError(
            "each class requires train, validation, and test samples"
        )
    rng = np.random.RandomState(int(seed))
    values = rng.permutation(values)
    count = int(values.size)
    validation_count = max(
        1,
        int(round(count * float(val_ratio))),
    )
    train_count = max(
        1,
        int(minimum_train),
        int(round(count * float(train_ratio))),
    )
    train_count = min(
        train_count,
        count - validation_count - 1,
    )
    if train_count < 1:
        raise RuntimeError(
            f"class split cannot allocate training data: n={count}"
        )
    remaining = count - train_count
    validation_count = min(
        validation_count,
        remaining - 1,
    )
    test_count = count - train_count - validation_count
    if validation_count < 1 or test_count < 1:
        raise RuntimeError(
            "class split must retain validation and test samples"
        )
    return (
        values[:train_count].copy(),
        values[train_count:train_count + validation_count].copy(),
        values[train_count + validation_count:].copy(),
    )


def build_incremental_protocol(
    input_labels: np.ndarray,
    *,
    class_count: int,
    base_classes: int,
    increment: int,
    train_ratio: float,
    val_ratio: float,
    minimum_train: int,
    seed: int,
    shuffle_order: bool,
    class_order: Optional[Sequence[int]],
) -> Dict[str, Any]:
    labels = np.asarray(input_labels, dtype=np.int64).reshape(-1)
    original_ids = sorted(
        int(value) for value in np.unique(labels).tolist()
    )
    if original_ids != list(range(int(class_count))):
        raise RuntimeError(
            "ExtractLabeledPixelIndex must return contiguous class IDs"
        )

    if class_order is not None:
        order = [int(value) for value in class_order]
        if len(order) != class_count or set(order) != set(original_ids):
            raise ValueError(
                "class_order must be a complete original-ID permutation"
            )
    elif shuffle_order:
        rng = np.random.RandomState(int(seed))
        order = [
            int(value)
            for value in rng.permutation(original_ids).tolist()
        ]
    else:
        order = list(original_ids)

    original_to_global = {
        original_id: global_id
        for global_id, original_id in enumerate(order)
    }
    global_labels = np.asarray(
        [original_to_global[int(value)] for value in labels],
        dtype=np.int64,
    )

    phase_to_classes: Dict[int, List[int]] = {
        0: list(range(int(base_classes)))
    }
    cursor = int(base_classes)
    phase = 1
    while cursor < class_count:
        phase_to_classes[phase] = list(
            range(
                cursor,
                min(cursor + int(increment), class_count),
            )
        )
        cursor += int(increment)
        phase += 1

    split_by_class: Dict[int, Dict[str, np.ndarray]] = {}
    class_counts: Dict[int, Dict[str, int]] = {}
    for global_id in range(class_count):
        indices = np.flatnonzero(
            global_labels == global_id
        ).astype(np.int64)
        train, validation, test = _split_one_class(
            indices,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            minimum_train=minimum_train,
            seed=int(seed) + 1009 * global_id,
        )
        split_by_class[global_id] = {
            "train": train,
            "val": validation,
            "test": test,
            "all": indices,
        }
        class_counts[global_id] = {
            "all": int(indices.size),
            "train": int(train.size),
            "val": int(validation.size),
            "test": int(test.size),
            "original_class_id": int(order[global_id]),
            "introduction_phase": next(
                phase_id
                for phase_id, class_ids
                in phase_to_classes.items()
                if global_id in class_ids
            ),
        }

    base_train_indices = np.concatenate(
        [
            split_by_class[class_id]["train"]
            for class_id in phase_to_classes[0]
        ]
    ).astype(np.int64, copy=False)
    return {
        "class_order_original_ids": order,
        "original_to_global": original_to_global,
        "global_to_original": {
            global_id: original_id
            for global_id, original_id in enumerate(order)
        },
        "global_labels": global_labels,
        "phase_to_classes": phase_to_classes,
        "num_phases": len(phase_to_classes),
        "split_by_class": split_by_class,
        "class_counts": class_counts,
        "base_train_indices": base_train_indices,
    }


class LazyHSIPatchDataset(Dataset):
    def __init__(
        self,
        manager: "IncrementalHSIDatasetManager",
        indices: Sequence[int],
    ) -> None:
        self.manager = manager
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.labels = manager.labels[self.indices]

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        sample_index = int(self.indices[int(item)])
        row, column = self.manager.coords[sample_index]
        radius = self.manager.patch_size // 2
        row_padded = int(row) + radius
        column_padded = int(column) + radius
        processed = self.manager.processed_padded[
            row_padded - radius:row_padded + radius + 1,
            column_padded - radius:column_padded + radius + 1,
            :,
        ]
        raw = self.manager.raw_padded[
            row_padded - radius:row_padded + radius + 1,
            column_padded - radius:column_padded + radius + 1,
            :,
        ]
        return {
            "image": torch.from_numpy(
                np.moveaxis(processed, -1, 0).copy()
            ).float(),
            "raw_spectral_patch": torch.from_numpy(
                np.moveaxis(raw, -1, 0).copy()
            ).float(),
            "label": torch.tensor(
                int(self.manager.labels[sample_index]),
                dtype=torch.long,
            ),
            "coord": torch.tensor(
                self.manager.coords[sample_index],
                dtype=torch.long,
            ),
            "sample_index": torch.tensor(
                sample_index,
                dtype=torch.long,
            ),
        }


class IncrementalHSIDatasetManager:
    """Phase-aware HSI dataset with strict training access.

    Current-phase train and validation loaders contain only current classes.
    Cumulative validation/test/all loaders contain only classes seen by the
    requested phase. No trainer API exposes old training samples during an
    incremental phase.
    """

    def __init__(
        self,
        *,
        processed_cube: np.ndarray,
        raw_physical_cube: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        protocol: Mapping[str, Any],
        target_names: Sequence[str],
        gt_shape: Sequence[int],
        patch_size: int,
        num_workers: int,
        device: str,
    ) -> None:
        processed = np.ascontiguousarray(
            processed_cube,
            dtype=np.float32,
        )
        raw = np.ascontiguousarray(
            raw_physical_cube,
            dtype=np.float32,
        )
        if processed.ndim != 3 or raw.ndim != 3:
            raise ValueError("HSI cubes must be [H,W,C]")
        if processed.shape[:2] != raw.shape[:2]:
            raise ValueError("processed and raw cube shapes differ")
        self.patch_size = int(patch_size)
        if self.patch_size <= 0 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be positive and odd")
        radius = self.patch_size // 2
        self.processed_padded = np.pad(
            processed,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="reflect",
        )
        self.raw_padded = np.pad(
            raw,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="reflect",
        )
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.coords = np.asarray(coords, dtype=np.int64)
        if self.coords.shape != (self.labels.size, 2):
            raise ValueError("coords must align with labels")
        self.phase_to_classes = {
            int(phase): [int(value) for value in class_ids]
            for phase, class_ids
            in protocol["phase_to_classes"].items()
        }
        self.num_phases = len(self.phase_to_classes)
        self.class_to_phase = {
            class_id: phase
            for phase, class_ids in self.phase_to_classes.items()
            for class_id in class_ids
        }
        self.split_by_class = {
            int(class_id): {
                str(split): np.asarray(indices, dtype=np.int64)
                for split, indices in values.items()
            }
            for class_id, values
            in protocol["split_by_class"].items()
        }
        self.target_names = [str(value) for value in target_names]
        self.total_class_count = len(self.target_names)
        self.gt_shape = tuple(int(value) for value in gt_shape)
        self.num_workers = int(num_workers)
        self.pin_memory = str(device).startswith("cuda")
        self.current_phase = 0
        self.finalized_phases: List[int] = []

    def start_phase(self, phase: int) -> None:
        phase = int(phase)
        if phase not in self.phase_to_classes:
            raise RuntimeError(f"phase {phase} is outside the schedule")
        accepted = max(self.finalized_phases, default=-1)
        if phase > 0 and accepted != phase - 1:
            raise RuntimeError(
                f"phase {phase} requires finalized phase {phase - 1}; "
                f"accepted={accepted}"
            )
        self.current_phase = phase

    def finalize_phase(self, phase: int) -> None:
        phase = int(phase)
        if phase != self.current_phase:
            raise RuntimeError(
                "cannot finalize a phase that is not active"
            )
        if phase not in self.finalized_phases:
            self.finalized_phases.append(phase)
            self.finalized_phases.sort()

    @contextmanager
    def memory_build_context(self, phase: int) -> Iterator[None]:
        if int(phase) != self.current_phase:
            raise RuntimeError(
                "memory context phase does not match active phase"
            )
        yield

    def get_seen_classes(self, phase: int) -> List[int]:
        phase = int(phase)
        if phase not in self.phase_to_classes:
            raise RuntimeError(f"phase {phase} is outside the schedule")
        output: List[int] = []
        for index in range(phase + 1):
            output.extend(self.phase_to_classes[index])
        return output

    def get_old_classes(self, phase: int) -> List[int]:
        if int(phase) == 0:
            return []
        new_set = set(self.phase_to_classes[int(phase)])
        return [
            class_id
            for class_id in self.get_seen_classes(int(phase))
            if class_id not in new_set
        ]

    def get_new_classes(self, phase: int) -> List[int]:
        return list(self.phase_to_classes[int(phase)])

    def _indices(
        self,
        *,
        class_ids: Sequence[int],
        split: str,
    ) -> np.ndarray:
        token = str(split).strip().lower()
        aliases = {
            "validation": "val",
            "all_labeled": "all",
            "full": "all",
        }
        token = aliases.get(token, token)
        if token not in {"train", "val", "test", "all"}:
            raise ValueError(f"unsupported split {split!r}")
        parts = [
            self.split_by_class[int(class_id)][token]
            for class_id in class_ids
        ]
        return np.concatenate(parts).astype(np.int64, copy=False)

    def _loader(
        self,
        indices: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset = LazyHSIPatchDataset(self, indices)
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_phase_dataloader(
        self,
        phase: int,
        *,
        split: str,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        phase = int(phase)
        class_ids = self.phase_to_classes[phase]
        return self._loader(
            self._indices(class_ids=class_ids, split=split),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_cumulative_dataloader(
        self,
        phase: int,
        *,
        split: str,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return self._loader(
            self._indices(
                class_ids=self.get_seen_classes(int(phase)),
                split=split,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )


def load_hsi_data(args: argparse.Namespace) -> Dict[str, Any]:
    (
        raw_hsi,
        gt,
        num_classes,
        raw_target_names,
        _,
        label_policy,
    ) = LoadRawHSIData(
        method=args.dataset,
        base_dir=args.data_dir,
    )
    validate_config(args, num_classes=int(num_classes))
    raw_hsi = np.asarray(raw_hsi, dtype=np.float32)
    gt = np.asarray(gt)
    input_labels, input_coords = ExtractLabeledPixelIndex(
        gt,
        label_policy=label_policy,
    )
    protocol = build_incremental_protocol(
        input_labels,
        class_count=int(num_classes),
        base_classes=int(args.base_classes),
        increment=int(args.increment),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        minimum_train=int(args.min_train_per_class),
        seed=int(args.seed),
        shuffle_order=bool(args.shuffle_order),
        class_order=parse_int_list(
            args.class_order,
            name="class_order",
        ),
    )

    coords = np.asarray(input_coords, dtype=np.int64)
    fit_mask = np.zeros(gt.shape[:2], dtype=bool)
    fit_coords = coords[protocol["base_train_indices"]]
    fit_mask[fit_coords[:, 0], fit_coords[:, 1]] = True
    if int(fit_mask.sum()) != int(
        protocol["base_train_indices"].size
    ):
        raise RuntimeError(
            "preprocessing fit mask does not match base training samples"
        )

    use_reduction = (
        not bool(args.no_pca)
        and str(args.reduction_method).strip().lower()
        not in {"", "none", "identity"}
    )
    (
        processed_cube,
        normalized_physical_cube,
        preprocessing_state,
    ) = FitHSIPreprocessor(
        raw_hsi,
        fit_mask=fit_mask,
        apply_reduction=use_reduction,
        reduction_method=str(args.reduction_method),
        n_components=int(args.pca_components),
        whiten=bool(args.pca_whiten),
        fit_scope="phase0_train_only",
    )
    preprocessing_path = SaveHSIPreprocessor(
        os.path.join(
            os.path.abspath(str(args.save_dir)),
            "preprocessing_state.npz",
        ),
        preprocessing_state,
    )

    args.num_bands = int(processed_cube.shape[2])
    args.raw_num_bands = int(normalized_physical_cube.shape[2])
    args.max_classes = int(num_classes)
    target_names = [
        str(raw_target_names[original_id])
        for original_id in protocol["class_order_original_ids"]
    ]

    summary = {
        "stack_build_id": STACK_BUILD_ID,
        "method": METHOD_NAME,
        "classification_factorization": CLASSIFICATION_FACTORIZATION,
        "spectral_relation_factorization": (
            SPECTRAL_RELATION_FACTORIZATION
        ),
        "dataset": str(args.dataset),
        "dataset_name": DATASET_INFO[str(args.dataset)]["name"],
        "class_count": int(num_classes),
        "phase_to_classes": protocol["phase_to_classes"],
        "class_order_original_ids": protocol[
            "class_order_original_ids"
        ],
        "class_counts": protocol["class_counts"],
        "raw_sensor_bands": int(normalized_physical_cube.shape[2]),
        "model_input_channels": int(processed_cube.shape[2]),
        "patch_size": int(args.patch_size),
        "pca_active": bool(use_reduction),
        "preprocessing_fit_scope": "phase0_train_only",
        "preprocessing_fit_pixel_count": int(fit_mask.sum()),
        "preprocessing_state_path": preprocessing_path,
        "training_access_policy": (
            "phase train loader exposes current classes only"
        ),
        "cumulative_evaluation_policy": (
            "validation/test/all expose seen classes only"
        ),
        "stores_old_training_exemplars_in_geometry_memory": False,
    }
    del raw_hsi
    gc.collect()
    return {
        "processed_cube": np.ascontiguousarray(
            processed_cube,
            dtype=np.float32,
        ),
        "raw_physical_cube": np.ascontiguousarray(
            normalized_physical_cube,
            dtype=np.float32,
        ),
        "labels": protocol["global_labels"],
        "coords": coords,
        "protocol": protocol,
        "target_names": target_names,
        "gt_shape": tuple(gt.shape[:2]),
        "summary": summary,
    }


def build_dataset(
    args: argparse.Namespace,
    data: Mapping[str, Any],
) -> IncrementalHSIDatasetManager:
    return IncrementalHSIDatasetManager(
        processed_cube=data["processed_cube"],
        raw_physical_cube=data["raw_physical_cube"],
        labels=data["labels"],
        coords=data["coords"],
        protocol=data["protocol"],
        target_names=data["target_names"],
        gt_shape=data["gt_shape"],
        patch_size=int(args.patch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
    )


# =============================================================================
# Model and phase reports
# =============================================================================


def build_model(args: argparse.Namespace) -> NECILModel:
    model = NECILModel(args).to(torch.device(args.device))
    model.assert_architecture_contract()
    summary = model.architecture_summary()
    if summary.get(
        "classification_factorization"
    ) != CLASSIFICATION_FACTORIZATION:
        raise RuntimeError(
            "model classification factorization is inconsistent"
        )
    if summary.get(
        "spectral_relation_factorization"
    ) != SPECTRAL_RELATION_FACTORIZATION:
        raise RuntimeError(
            "model spectral relation is inconsistent"
        )
    if summary.get("joint_feature") != "direct_[z_s;z_p]":
        raise RuntimeError(
            "model does not expose direct spectral-spatial coordinates"
        )
    if bool(summary.get("trainable_transport_network", True)):
        raise RuntimeError(
            "main driver rejects trainable transport networks"
        )
    if bool(summary.get("uses_geometry_replay_for_training", True)):
        raise RuntimeError(
            "main driver rejects geometry replay training"
        )
    if any(
        parameter.requires_grad
        for parameter in model.classifier.parameters()
    ):
        raise RuntimeError(
            "deployed geometry classifier must be parameter-free"
        )
    return model


def save_model_summary(
    path: str,
    model: NECILModel,
) -> str:
    total = sum(
        int(parameter.numel())
        for parameter in model.parameters()
    )
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    lines = [
        f"Model: {model.__class__.__name__}",
        f"Stack: {STACK_BUILD_ID}",
        f"Method: {METHOD_NAME}",
        f"Classification: {CLASSIFICATION_FACTORIZATION}",
        (
            "Spectral relation: "
            f"{SPECTRAL_RELATION_FACTORIZATION} (pair risk only)"
        ),
        "=" * 100,
        f"Total parameters: {total:,}",
        f"Trainable parameters at save time: {trainable:,}",
        "",
        "Architecture contract",
        "-" * 100,
    ]
    for key, value in sorted(model.architecture_summary().items()):
        lines.append(f"{key}: {value}")
    lines.extend(["", "GeometryBank", "-" * 100])
    for key, value in sorted(
        model.geometry_bank.memory_cost_summary().items()
    ):
        lines.append(f"{key}: {value}")
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, absolute)
    return absolute


def write_phase_report(
    path: str,
    *,
    phase: int,
    metrics: Mapping[str, Any],
    sampling: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    lines = [
        (
            "Transport-Verified Spectral-Spatial Factor Geometry "
            f"— Phase {phase}"
        ),
        "=" * 112,
        "",
        "Method contract",
        "-" * 112,
        f"Classification: {CLASSIFICATION_FACTORIZATION}",
        (
            "Spectral relation: "
            f"{SPECTRAL_RELATION_FACTORIZATION} (training pair risk only)"
        ),
        "Feature: direct unnormalized z=[z_s;z_p]",
        "Classifier: parameter-free equal-prior factor energy",
        "Memory: aggregate class rows only; no old training samples",
        "",
        "Cumulative held-out test",
        "-" * 112,
        (
            f"OA={float(metrics['overall_accuracy']):.2f}% | "
            f"AA={float(metrics['average_accuracy']):.2f}% | "
            f"MinClass={float(metrics['minimum_per_class_accuracy']):.2f}% | "
            f"Kappa={float(metrics['kappa']):.4f} | "
            f"MacroF1={float(metrics['f1_macro']):.2f}%"
        ),
        (
            f"Old={float(metrics['old_accuracy']):.2f}% | "
            f"New={float(metrics['new_accuracy']):.2f}% | "
            f"H={float(metrics['harmonic_mean']):.2f}% | "
            f"Old->New={float(metrics['old_to_new_rate']):.2f}% | "
            f"New->Old={float(metrics['new_to_old_rate']):.2f}%"
        ),
        (
            f"MeanGap={float(metrics['geometry_margin_mean']):.6f} | "
            f"Q05Gap={float(metrics['geometry_margin_q05']):.6f} | "
            f"MinGap={float(metrics['geometry_margin_min']):.6f} | "
            f"Violation={float(metrics['geometry_error_rate']):.2f}%"
        ),
        "",
        "Sampling diagnostic only",
        "-" * 112,
        (
            f"Accuracy={float(sampling['accuracy']):.2f}% | "
            f"Q05={float(sampling['q05_gap']):.6f} | "
            f"Violation={float(sampling['violation_rate']):.2f}% | "
            f"TrainingUse={sampling['training_use']}"
        ),
        "",
        "GeometryBank",
        "-" * 112,
        f"Memory: {json_safe(diagnostics['memory_cost'])}",
        f"Admission: {json_safe(diagnostics['admission'])}",
        "",
        "Class-wise results",
        "-" * 112,
        "id  class                              support  accuracy  q05-gap  violation",
    ]
    for class_id in metrics["classes"]:
        geometry = metrics["per_class_geometry"][class_id]
        lines.append(
            f"{class_id:2d}  "
            f"{str(class_id):32s} "
            f"{int(metrics['support'][class_id]):7d} "
            f"{float(metrics['per_class_accuracy'][class_id]):8.2f}% "
            f"{float(geometry['q05_gap']):8.4f} "
            f"{float(geometry['classification_violation_rate']):8.2f}%"
        )
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, absolute)
    return absolute


# =============================================================================
# Phase evaluation and visualization orchestration
# =============================================================================


def _guarded_artifact(
    artifacts: Dict[str, Any],
    name: str,
    operation: Any,
) -> None:
    try:
        artifacts[name] = operation()
    except Exception as error:
        artifacts[name] = {
            "status": "SKIPPED",
            "reason": str(error),
        }
        print(f"[Artifact skipped: {name}] {error}")


def evaluate_and_visualize_phase(
    *,
    args: argparse.Namespace,
    run_dir: str,
    phase: int,
    model: NECILModel,
    dataset: IncrementalHSIDatasetManager,
    history: Mapping[str, Any],
    evaluator: NECILEvaluator,
) -> Dict[str, Any]:
    old_classes = dataset.get_old_classes(phase)
    new_classes = dataset.get_new_classes(phase)
    seen_classes = dataset.get_seen_classes(phase)
    phase_dir = os.path.join(run_dir, f"phase_{phase}")
    reports_dir = os.path.join(phase_dir, "reports")
    figures_dir = os.path.join(phase_dir, "figures")
    arrays_dir = os.path.join(phase_dir, "raw_arrays")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    test_loader = dataset.get_cumulative_dataloader(
        phase,
        split="test",
        batch_size=int(args.eval_batch_size),
        shuffle=False,
    )
    all_loader = dataset.get_cumulative_dataloader(
        phase,
        split="all",
        batch_size=int(args.eval_batch_size),
        shuffle=False,
    )
    test_output = evaluate_factor_geometry_loader(
        model,
        test_loader,
        seen_classes=seen_classes,
        old_classes=old_classes,
        new_classes=new_classes,
        device=args.device,
        energy_margin=float(args.energy_margin),
        return_arrays=True,
    )
    all_output = evaluate_factor_geometry_loader(
        model,
        all_loader,
        seen_classes=seen_classes,
        old_classes=old_classes,
        new_classes=new_classes,
        device=args.device,
        energy_margin=float(args.energy_margin),
        return_arrays=False,
    )
    metrics = test_output["metrics"]
    evaluator.update(
        phase,
        test_output["y_true"],
        test_output["y_pred"],
        seen_classes=seen_classes,
        old_classes=old_classes,
        new_classes=new_classes,
        energy=test_output["energy"],
        energy_margin=float(args.energy_margin),
    )

    classification_report = save_classification_report(
        test_output["y_true"],
        test_output["y_pred"],
        target_names=dataset.target_names,
        save_dir=reports_dir,
        phase=phase,
        seen_classes=seen_classes,
        old_classes=old_classes,
        new_classes=new_classes,
        energy=test_output["energy"],
        energy_margin=float(args.energy_margin),
    )
    sampling = geometry_sampling_health(
        model,
        seen_classes,
        samples_per_class=int(
            args.base_diagnostic_samples_per_class
        ),
    )
    diagnostics = geometry_bank_diagnostics(
        model,
        seen_classes,
    )
    atomic_json(
        os.path.join(reports_dir, "training_history.json"),
        history,
    )
    atomic_json(
        os.path.join(reports_dir, "cumulative_test_metrics.json"),
        metrics,
    )
    atomic_json(
        os.path.join(reports_dir, "all_seen_labeled_metrics.json"),
        all_output["metrics"],
    )
    atomic_json(
        os.path.join(reports_dir, "geometry_sampling_health.json"),
        sampling,
    )
    atomic_json(
        os.path.join(reports_dir, "geometry_bank_diagnostics.json"),
        diagnostics,
    )
    text_report = write_phase_report(
        os.path.join(reports_dir, f"phase_{phase}_geometry_report.txt"),
        phase=phase,
        metrics=metrics,
        sampling=sampling,
        diagnostics=diagnostics,
    )

    artifacts: Dict[str, Any] = {}
    if bool(args.save_visualizations):
        _guarded_artifact(
            artifacts,
            "training_dynamics",
            lambda: plot_phase_training_dynamics(
                history,
                os.path.join(
                    figures_dir,
                    f"phase_{phase}_training_dashboard.png",
                ),
                dpi=int(args.visualization_dpi),
                save_separate=True,
            ),
        )
        _guarded_artifact(
            artifacts,
            "classification_diagnostics",
            lambda: save_classification_diagnostics(
                confusion_matrix=np.asarray(
                    metrics["confusion_matrix"],
                    dtype=np.int64,
                ),
                class_ids=seen_classes,
                target_names=dataset.target_names,
                save_path=os.path.join(
                    figures_dir,
                    f"phase_{phase}_classification_diagnostics.png",
                ),
                title=(
                    f"Phase {phase} Cumulative Test Classification"
                ),
                dpi=int(args.visualization_dpi),
            ),
        )
        if bool(args.save_test_maps):
            _guarded_artifact(
                artifacts,
                "test_maps",
                lambda: predict_phase_grid(
                    model,
                    dataset,
                    phase,
                    dataset.target_names,
                    save_dir=figures_dir,
                    device=str(args.device),
                    batch_size=int(args.visualization_batch_size),
                    split="test",
                    class_cmap=str(args.viz_class_cmap),
                    background_color=str(args.viz_background_color),
                    save_numpy=bool(args.save_map_numpy),
                    save_combined=bool(args.save_combined_maps),
                    return_outputs=True,
                    dpi=int(args.visualization_dpi),
                ),
            )
        if bool(args.save_all_labeled_maps):
            _guarded_artifact(
                artifacts,
                "all_seen_labeled_maps",
                lambda: predict_phase_grid(
                    model,
                    dataset,
                    phase,
                    dataset.target_names,
                    save_dir=figures_dir,
                    device=str(args.device),
                    batch_size=int(args.visualization_batch_size),
                    split="all",
                    class_cmap=str(args.viz_class_cmap),
                    background_color=str(args.viz_background_color),
                    save_numpy=bool(args.save_map_numpy),
                    save_combined=bool(args.save_combined_maps),
                    return_outputs=True,
                    dpi=int(args.visualization_dpi),
                ),
            )

        def overlap_artifact() -> Dict[str, Any]:
            review = build_geometry_overlap_review(
                model=model,
                class_ids=seen_classes,
                target_names=dataset.target_names,
                directional_invasion_matrix=metrics[
                    "directional_invasion_matrix"
                ],
            )
            review_path = atomic_json(
                os.path.join(
                    reports_dir,
                    "geometry_overlap_review.json",
                ),
                review,
            )
            figure_path = save_geometry_pair_diagnostics(
                review=review,
                class_ids=seen_classes,
                target_names=dataset.target_names,
                save_path=os.path.join(
                    figures_dir,
                    f"phase_{phase}_geometry_pair_diagnostics.png",
                ),
                dpi=int(args.visualization_dpi),
            )
            return {
                "review_path": review_path,
                "figure_path": figure_path,
                "high_risk_pair_count": int(
                    review["high_risk_pair_count"]
                ),
                "moderate_risk_pair_count": int(
                    review["moderate_risk_pair_count"]
                ),
            }

        _guarded_artifact(
            artifacts,
            "geometry_overlap",
            overlap_artifact,
        )
        _guarded_artifact(
            artifacts,
            "spectral_spatial_geometry",
            lambda: save_spectral_spatial_geometry_diagnostics(
                model=model,
                class_ids=seen_classes,
                target_names=dataset.target_names,
                save_path=os.path.join(
                    figures_dir,
                    f"phase_{phase}_spectral_spatial_geometry.png",
                ),
                dpi=int(args.visualization_dpi),
            ),
        )
        if phase > 0:
            _guarded_artifact(
                artifacts,
                "transport_diagnostics",
                lambda: save_transport_diagnostics(
                    history=history,
                    save_path=os.path.join(
                        figures_dir,
                        f"phase_{phase}_transport_diagnostics.png",
                    ),
                    dpi=int(args.visualization_dpi),
                ),
            )

    if bool(args.save_raw_arrays):
        os.makedirs(arrays_dir, exist_ok=True)
        np.save(
            os.path.join(arrays_dir, "true_labels.npy"),
            test_output["y_true"],
        )
        np.save(
            os.path.join(arrays_dir, "predicted_labels.npy"),
            test_output["y_pred"],
        )
        if test_output["coords"] is not None:
            np.save(
                os.path.join(arrays_dir, "coords.npy"),
                test_output["coords"],
            )
        torch.save(
            test_output["energy"],
            os.path.join(arrays_dir, "factor_energy.pt"),
        )
        torch.save(
            test_output["quadratic"],
            os.path.join(arrays_dir, "quadratic_energy.pt"),
        )
        torch.save(
            test_output["volume"],
            os.path.join(arrays_dir, "volume_energy.pt"),
        )
        np.save(
            os.path.join(arrays_dir, "confusion_matrix.npy"),
            np.asarray(metrics["confusion_matrix"], dtype=np.int64),
        )

    atomic_json(
        os.path.join(reports_dir, "visualization_artifacts.json"),
        artifacts,
    )
    phase_payload = {
        "phase": phase,
        "old_classes": old_classes,
        "new_classes": new_classes,
        "seen_classes": seen_classes,
        "test_metrics": metrics,
        "all_seen_labeled_metrics": all_output["metrics"],
        "sampling_health": sampling,
        "geometry_bank_diagnostics": diagnostics,
        "classification_report": classification_report,
        "text_report": text_report,
        "visualization_artifacts": artifacts,
    }
    atomic_json(
        os.path.join(phase_dir, "phase_summary.json"),
        phase_payload,
    )
    print(
        f"[Phase {phase} Test] "
        f"OA={metrics['overall_accuracy']:.2f}% | "
        f"AA={metrics['average_accuracy']:.2f}% | "
        f"Old={metrics['old_accuracy']:.2f}% | "
        f"New={metrics['new_accuracy']:.2f}% | "
        f"H={metrics['harmonic_mean']:.2f}% | "
        f"Q05={metrics['geometry_margin_q05']:.4f}"
    )
    return phase_payload


# =============================================================================
# Single and multi-run execution
# =============================================================================


def _execution_phases(
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
) -> List[int]:
    phases = sorted(dataset.phase_to_classes)
    if bool(args.base_only):
        return [0]
    maximum = int(args.maximum_phase)
    if maximum >= 0:
        phases = [phase for phase in phases if phase <= maximum]
    return phases


def run_single_experiment(
    base_args: argparse.Namespace,
    *,
    run_index: int,
    seed: int,
) -> Dict[str, Any]:
    requested = argparse.Namespace(**namespace_to_dict(base_args))
    requested.seed = int(seed)
    args = resolve_config(requested)
    validate_config(args)
    set_seed(int(args.seed), bool(args.deterministic))

    run_dir = os.path.join(
        os.path.abspath(str(args.save_dir)),
        str(args.dataset),
        (
            f"base_{int(args.base_classes)}_"
            f"inc_{int(args.increment)}"
        ),
        f"run_{run_index + 1}_seed_{int(args.seed)}",
    )
    os.makedirs(run_dir, exist_ok=True)
    args.save_dir = run_dir
    atomic_json(
        os.path.join(run_dir, "requested_configuration.json"),
        namespace_to_dict(requested),
    )
    atomic_json(
        os.path.join(run_dir, "resolved_configuration.json"),
        namespace_to_dict(args),
    )

    print("\n" + "=" * 112)
    print("TRANSPORT-VERIFIED SPECTRAL-SPATIAL FACTOR GEOMETRY")
    print("=" * 112)
    print(
        f"[Run] dataset={args.dataset} | seed={args.seed} | "
        f"device={args.device} | base={args.base_classes} | "
        f"increment={args.increment}"
    )
    print(
        f"[Contract] {CLASSIFICATION_FACTORIZATION}; "
        f"{SPECTRAL_RELATION_FACTORIZATION} supplies pair risk only"
    )

    load_start = time.time()
    data = load_hsi_data(args)
    loading_time = time.time() - load_start
    atomic_json(
        os.path.join(run_dir, "dataset_summary.json"),
        data["summary"],
    )
    atomic_json(
        os.path.join(run_dir, "incremental_protocol.json"),
        {
            "phase_to_classes": data["protocol"]["phase_to_classes"],
            "class_order_original_ids": data["protocol"][
                "class_order_original_ids"
            ],
            "class_counts": data["protocol"]["class_counts"],
        },
    )
    dataset = build_dataset(args, data)
    model = build_model(args)
    if bool(args.save_model_summary):
        save_model_summary(
            os.path.join(run_dir, "model_summary_initial.txt"),
            model,
        )
    trainer = Trainer(model, dataset, args)
    evaluator = NECILEvaluator()
    histories: Dict[int, Dict[str, Any]] = {}
    phase_results: Dict[int, Dict[str, Any]] = {}

    resume = str(args.resume_checkpoint).strip()
    accepted_phase = -1
    if resume:
        checkpoint = trainer.load_checkpoint(
            resume,
            strict=True,
            restore_rng=True,
        )
        previous_run_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(resume))
        )
        previous_evaluation = os.path.join(
            previous_run_dir,
            "evaluation",
            "necil_evaluation_summary.json",
        )
        if os.path.isfile(previous_evaluation):
            with open(previous_evaluation, "r", encoding="utf-8") as stream:
                evaluator.restore(json.load(stream))
            print(
                "[Resume] restored prior phase evaluation history from "
                f"{previous_evaluation}"
            )
        accepted_phase = int(checkpoint["phase"])
        history = dict(checkpoint.get("history", {}))
        histories[accepted_phase] = history
        for phase in range(accepted_phase + 1):
            if phase not in dataset.finalized_phases:
                dataset.finalized_phases.append(phase)
        dataset.current_phase = accepted_phase
        print(
            f"[Resume] accepted phase {accepted_phase} loaded from {resume}"
        )
        phase_results[accepted_phase] = evaluate_and_visualize_phase(
            args=args,
            run_dir=run_dir,
            phase=accepted_phase,
            model=model,
            dataset=dataset,
            history=history,
            evaluator=evaluator,
        )

    phases = _execution_phases(args, dataset)
    for phase in phases:
        if phase <= accepted_phase:
            continue
        history = trainer.train_phase(
            phase=phase,
            epochs=(
                int(args.epochs_base)
                if phase == 0
                else int(args.epochs_inc)
            ),
            batch_size=int(args.batch_size),
            lr=(
                float(args.lr)
                if phase == 0
                else float(args.lr_inc)
            ),
        )
        histories[phase] = history
        if phase > 0 and not bool(history.get("committed", False)):
            print(
                f"[Phase {phase}] rejected; stopping after accepted phase "
                f"{phase - 1}"
            )
            break
        phase_results[phase] = evaluate_and_visualize_phase(
            args=args,
            run_dir=run_dir,
            phase=phase,
            model=model,
            dataset=dataset,
            history=history,
            evaluator=evaluator,
        )
        evaluator.save(os.path.join(run_dir, "evaluation"))
        if bool(args.save_visualizations):
            plot_necil_phase_summary(
                evaluator,
                os.path.join(
                    run_dir,
                    "evaluation",
                    "necil_phase_summary.png",
                ),
                dpi=int(args.visualization_dpi),
            )

    if not phase_results:
        raise RuntimeError("no accepted phase was evaluated")
    evaluator_paths = evaluator.save(
        os.path.join(run_dir, "evaluation")
    )
    if bool(args.save_visualizations):
        phase_summary_figure = plot_necil_phase_summary(
            evaluator,
            os.path.join(
                run_dir,
                "evaluation",
                "necil_phase_summary.png",
            ),
            dpi=int(args.visualization_dpi),
        )
    else:
        phase_summary_figure = None

    if bool(args.save_model_summary):
        save_model_summary(
            os.path.join(run_dir, "model_summary_final.txt"),
            model,
        )

    final_phase = max(phase_results)
    final_metrics = phase_results[final_phase]["test_metrics"]
    payload = {
        "stack_build_id": STACK_BUILD_ID,
        "method": METHOD_NAME,
        "classification_factorization": CLASSIFICATION_FACTORIZATION,
        "spectral_relation_factorization": (
            SPECTRAL_RELATION_FACTORIZATION
        ),
        "run_index": int(run_index),
        "seed": int(args.seed),
        "run_dir": run_dir,
        "data_loading_time_sec": float(loading_time),
        "accepted_phases": sorted(phase_results),
        "final_phase": int(final_phase),
        "phase_results": phase_results,
        "necil_metrics": evaluator.to_dict(),
        "evaluator_paths": evaluator_paths,
        "phase_summary_figure": phase_summary_figure,
        "final_metrics": final_metrics,
    }
    atomic_json(
        os.path.join(run_dir, "final_summary.json"),
        payload,
    )
    evaluator.print_summary()
    return payload


def aggregate_runs(
    results: Sequence[Mapping[str, Any]],
    root: str,
) -> Dict[str, Any]:
    if not results:
        raise ValueError("results is empty")
    rows: List[Dict[str, Any]] = []
    for result in results:
        final = result["final_metrics"]
        standard = result["necil_metrics"]["standard_metrics"]
        rows.append(
            {
                "run_index": int(result["run_index"]),
                "seed": int(result["seed"]),
                "final_phase": int(result["final_phase"]),
                "OA": float(final["overall_accuracy"]),
                "AA": float(final["average_accuracy"]),
                "Kappa": float(final["kappa"]),
                "MacroF1": float(final["f1_macro"]),
                "Old": float(final["old_accuracy"]),
                "New": float(final["new_accuracy"]),
                "H": float(final["harmonic_mean"]),
                "F_avg": float(standard.get("F_avg", 0.0)),
                "BWT": float(standard.get("BWT", 0.0)),
                "OldToNew": float(final["old_to_new_rate"]),
                "NewToOld": float(final["new_to_old_rate"]),
                "GeometryViolation": float(
                    final["geometry_error_rate"]
                ),
                "Q05Gap": float(final["geometry_margin_q05"]),
                "run_dir": str(result["run_dir"]),
            }
        )

    numeric_keys = [
        key
        for key in rows[0]
        if key not in {"run_index", "seed", "final_phase", "run_dir"}
    ]
    mean_std = {
        key: (
            float(
                np.mean([float(row[key]) for row in rows])
            ),
            float(
                np.std([float(row[key]) for row in rows], ddof=0)
            ),
        )
        for key in numeric_keys
    }
    summary = {
        "stack_build_id": STACK_BUILD_ID,
        "method": METHOD_NAME,
        "num_runs": len(rows),
        "runs": rows,
        "mean_std": mean_std,
    }
    os.makedirs(root, exist_ok=True)
    atomic_json(
        os.path.join(root, "multi_run_summary.json"),
        summary,
    )
    csv_path = os.path.join(root, "multi_run_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
    summary["csv_path"] = csv_path
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parsed = parse_args(argv)
    preview = resolve_config(parsed)
    validate_config(preview)
    seeds = parse_int_list(preview.seed_list, name="seed_list")
    if seeds is None:
        seeds = [
            int(preview.seed) + index
            for index in range(int(preview.num_runs))
        ]
    results = [
        run_single_experiment(
            parsed,
            run_index=index,
            seed=seed,
        )
        for index, seed in enumerate(seeds)
    ]
    root = os.path.join(
        os.path.abspath(str(preview.save_dir)),
        str(preview.dataset),
        (
            f"base_{int(preview.base_classes)}_"
            f"inc_{int(preview.increment)}"
        ),
    )
    summary = aggregate_runs(results, root)
    oa_mean, oa_std = summary["mean_std"]["OA"]
    h_mean, h_std = summary["mean_std"]["H"]
    forgetting_mean, forgetting_std = summary["mean_std"]["F_avg"]
    print("\n" + "=" * 112)
    print("FACTOR-GEOMETRY NECIL MULTI-RUN SUMMARY")
    print("=" * 112)
    print(
        f"runs={summary['num_runs']} | "
        f"OA={oa_mean:.2f}±{oa_std:.2f} | "
        f"H={h_mean:.2f}±{h_std:.2f} | "
        f"F_avg={forgetting_mean:.2f}±{forgetting_std:.2f}"
    )
    print(
        f"[Saved] {os.path.join(root, 'multi_run_summary.json')}"
    )


if __name__ == "__main__":
    main()










# from __future__ import annotations

# import argparse
# import copy
# import csv
# import gc
# import json
# import math
# import os
# import random
# import sys
# import time
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from data.hsi_dataloader_pytorch import (
#     ApplyHSIPreprocessor,
#     BuildPCSIRGInterventionCentersFromCube,
#     BuildPreprocessFitMask,
#     ExtractLabeledPixelIndex,
#     FitHSIPreprocessor,
#     ImageCubes,
#     LoadHSIPreprocessor,
#     LoadRawHSIData,
#     SaveHSIPreprocessor,
#     ValidatePCSIRGCenterAlignment,
# )
# from data.incremental_dataset import HSIPatchDataset, IncrementalHSIDataset
# from models.necil_model import NECILModel
# from trainers.trainer import Trainer
# from utils.visualize import (
#     build_geometry_overlap_review,
#     plot_base_training_dynamics,
#     predict_phase_grid,
#     save_classification_diagnostics,
#     save_geometry_pair_diagnostics,
# )



# BuildPCSTGBInterventionCentersFromCube = BuildPCSIRGInterventionCentersFromCube
# ValidatePCSTGBCenterAlignment = ValidatePCSIRGCenterAlignment

# STACK_BUILD_ID = "PC-STGB-CONDITIONAL-JOINT-NECIL-V5-2026-07-23"
# METHOD_NAME = "PC-STGB"
# BANK_SCHEMA_VERSION = 5
# JOINT_FACTORIZATION = "p(z|c)prod_k p(g_k|z,c)"
# VISUALIZATION_MODE_ALIAS = "pc_stgb"

# DATASET_INFO: Dict[str, Dict[str, Any]] = {
#     "IP": {"name": "Indian Pines", "bands": 200, "classes": 16},
#     "SA": {"name": "Salinas", "bands": 204, "classes": 16},
#     "PU": {"name": "Pavia University", "bands": 103, "classes": 9},
#     "PC": {"name": "Pavia Centre", "bands": 102, "classes": 9},
#     "BS": {"name": "Botswana", "bands": 145, "classes": 14},
#     "LK": {"name": "LongKou", "bands": 270, "classes": 9},
#     "HH": {"name": "HongHu", "bands": 270, "classes": 22},
#     "HC": {"name": "HanChuan", "bands": 274, "classes": 16},
#     "UH13": {"name": "Houston 2013", "bands": 144, "classes": 15},
#     "QUH": {"name": "QUH-Qingyun", "bands": 270, "classes": 6},
#     "PI": {"name": "QUH-Pingan", "bands": 270, "classes": 10},
#     "TH": {"name": "QUH-Tangdaowan", "bands": 270, "classes": 18},
# }


# # =============================================================================
# # Generic utilities
# # =============================================================================


# def str2bool(value: Any) -> bool:
#     if isinstance(value, bool):
#         return value
#     if value is None:
#         return False
#     token = str(value).strip().lower()
#     if token in {"1", "true", "yes", "y", "on", "t"}:
#         return True
#     if token in {"0", "false", "no", "n", "off", "none", "null", "", "f"}:
#         return False
#     raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


# def parse_seed_list(raw: Optional[str]) -> Optional[List[int]]:
#     if raw is None or not str(raw).strip():
#         return None
#     result = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
#     if len(result) != len(set(result)):
#         raise ValueError(f"seed_list contains duplicates: {result}")
#     return result


# def parse_class_order(raw: Optional[str]) -> Optional[List[int]]:
#     if raw is None or not str(raw).strip():
#         return None
#     result = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
#     if len(result) != len(set(result)):
#         raise ValueError(f"class_order contains duplicates: {result}")
#     return result


# def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
#     return copy.deepcopy(vars(args))


# def json_safe(value: Any) -> Any:
#     if torch.is_tensor(value):
#         tensor = value.detach().cpu()
#         return tensor.item() if tensor.numel() == 1 else tensor.tolist()
#     if isinstance(value, np.ndarray):
#         return value.tolist()
#     if isinstance(value, (np.integer, np.floating)):
#         return value.item()
#     if isinstance(value, Mapping):
#         return {str(key): json_safe(item) for key, item in value.items()}
#     if isinstance(value, (tuple, list)):
#         return [json_safe(item) for item in value]
#     if isinstance(value, set):
#         return [json_safe(item) for item in sorted(value, key=str)]
#     if isinstance(value, (str, int, float, bool)) or value is None:
#         return value
#     return str(value)


# def save_json(path: str, value: Any) -> str:
#     absolute = os.path.abspath(path)
#     os.makedirs(os.path.dirname(absolute), exist_ok=True)
#     temporary = absolute + ".tmp"
#     with open(temporary, "w", encoding="utf-8") as stream:
#         json.dump(json_safe(value), stream, indent=2, sort_keys=True)
#         stream.flush()
#         os.fsync(stream.fileno())
#     os.replace(temporary, absolute)
#     return absolute


# def set_seed(seed: int, deterministic: bool = False) -> None:
#     seed = int(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     if deterministic:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.use_deterministic_algorithms(True, warn_only=True)
#     elif torch.cuda.is_available():
#         torch.backends.cudnn.benchmark = True


# def resolve_device(value: Any) -> str:
#     token = str(value).strip().lower()
#     if token == "gpu":
#         token = "cuda"
#     try:
#         requested = torch.device(token)
#     except (TypeError, RuntimeError, ValueError) as error:
#         raise RuntimeError(
#             f"Invalid device {value!r}; use cpu, cuda, or cuda:<index>"
#         ) from error
#     if requested.type == "cpu":
#         return "cpu"
#     if requested.type != "cuda":
#         raise RuntimeError(f"Unsupported device type {requested.type!r}")
#     if not torch.cuda.is_available():
#         raise RuntimeError(f"device={value!r} requests CUDA, but CUDA is unavailable")
#     count = int(torch.cuda.device_count())
#     index = (
#         int(torch.cuda.current_device())
#         if requested.index is None
#         else int(requested.index)
#     )
#     if index < 0 or index >= count:
#         raise RuntimeError(
#             f"Requested cuda:{index}, but only {count} device(s) are visible"
#         )
#     torch.cuda.set_device(index)
#     return f"cuda:{index}"


# def to_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
#     if torch.is_tensor(value):
#         array = value.detach().cpu().numpy()
#     else:
#         array = np.asarray(value)
#     return np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)


# def _finite_float(value: Any, name: str) -> float:
#     result = float(value)
#     if not math.isfinite(result):
#         raise ValueError(f"{name} must be finite")
#     return result


# def _class_name(target_names: Optional[Sequence[str]], class_id: int) -> str:
#     if target_names is not None and 0 <= int(class_id) < len(target_names):
#         return str(target_names[int(class_id)])
#     return f"Class-{int(class_id)}"


# # =============================================================================
# # Configuration
# # =============================================================================


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Strict non-exemplar HSI class-incremental learning with PC-STGB: "
#             "low-rank occupancy geometry, occupancy-conditioned spectral tangents, "
#             "coupled aggregate replay, and digest-protected atomic admission."
#         ),
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     exp = parser.add_argument_group("A. Experiment and phase protocol")
#     exp.add_argument("--dataset", type=str, default="IP", choices=sorted(DATASET_INFO))
#     exp.add_argument("--data_dir", type=str, default="./datasets")
#     exp.add_argument("--save_dir", type=str, default="./results_pc_stgb_v5")
#     exp.add_argument("--patch_size", type=int, default=11)
#     exp.add_argument("--train_ratio", type=float, default=0.20)
#     exp.add_argument("--val_ratio", type=float, default=0.10)
#     exp.add_argument("--min_train_per_class", type=int, default=20)
#     exp.add_argument("--base_classes", type=int, default=6)
#     exp.add_argument("--increment", type=int, default=3)
#     exp.add_argument("--shuffle_order", type=str2bool, default=False)
#     exp.add_argument("--class_order", type=str, default="")
#     exp.add_argument("--epochs_base", type=int, default=80)
#     exp.add_argument("--epochs_inc", type=int, default=20)
#     exp.add_argument("--batch_size", type=int, default=64)
#     exp.add_argument("--eval_batch_size", type=int, default=128)
#     exp.add_argument("--lr", type=float, default=1e-4)
#     exp.add_argument("--lr_inc", type=float, default=0.0)
#     exp.add_argument("--weight_decay", type=float, default=1e-4)
#     exp.add_argument("--base_only", type=str2bool, default=False)
#     exp.add_argument("--disable_incremental_training", type=str2bool, default=False)
#     exp.add_argument(
#         "--max_phase",
#         type=int,
#         default=-1,
#         help="-1 runs every protocol phase; otherwise stop after this phase index.",
#     )
#     exp.add_argument("--resume_checkpoint", type=str, default="")
#     exp.add_argument("--seed", type=int, default=42)
#     exp.add_argument("--num_runs", type=int, default=1)
#     exp.add_argument("--seed_list", type=str, default="")
#     exp.add_argument("--device", type=str, default="cuda:0")
#     exp.add_argument("--num_workers", type=int, default=0)
#     exp.add_argument("--deterministic", type=str2bool, default=False)
#     exp.add_argument("--strict_non_exemplar", type=str2bool, default=True)

#     preprocessing = parser.add_argument_group("B. Leakage-safe preprocessing")
#     preprocessing.add_argument("--no_pca", action="store_true")
#     preprocessing.add_argument("--reduction_method", type=str, default="PCA")
#     preprocessing.add_argument("--pca_components", type=int, default=30)
#     preprocessing.add_argument("--preprocess_response_chunk_size", type=int, default=4096)

#     backbone = parser.add_argument_group("C. Frozen canonical representation")
#     backbone.add_argument("--d_model", type=int, default=128)
#     backbone.add_argument("--d_state", type=int, default=16)
#     backbone.add_argument("--d_conv", type=int, default=4)
#     backbone.add_argument("--expand", type=int, default=2)
#     backbone.add_argument("--num_spectral_layers", type=int, default=3)
#     backbone.add_argument("--num_layers", type=int, default=3)
#     backbone.add_argument("--dropout", type=float, default=0.10)
#     backbone.add_argument("--projection_dropout", type=float, default=0.10)
#     backbone.add_argument("--backbone_norm", type=str, default="layer", choices=["layer", "rms"])
#     backbone.add_argument("--stem_norm_groups", type=int, default=8)
#     backbone.add_argument("--ssm_residual_scale_init", type=float, default=0.70)
#     backbone.add_argument("--fusion_residual_scale", type=float, default=0.30)
#     backbone.add_argument("--backbone_output_dropout", type=float, default=0.0)
#     backbone.add_argument("--projection_residual_scale_init", type=float, default=0.10)
#     backbone.add_argument("--zero_init_projection_output", type=str2bool, default=True)
#     backbone.add_argument("--normalize_geometry_features", type=str2bool, default=False)
#     backbone.add_argument("--geometry_feature_scale", type=float, default=1.0)
#     backbone.add_argument("--geometry_feature_clamp", type=float, default=0.0)

#     occupancy = parser.add_argument_group("D. Occupancy geometry")
#     occupancy.add_argument("--subspace_rank", type=int, default=5)
#     occupancy.add_argument("--geom_var_floor", type=float, default=5e-4)
#     occupancy.add_argument("--geometry_variance_shrinkage", type=float, default=0.25)
#     occupancy.add_argument("--geometry_max_variance_ratio", type=float, default=50.0)
#     occupancy.add_argument("--geometry_min_reliability", type=float, default=0.05)
#     occupancy.add_argument("--reliability_sample_alpha", type=float, default=20.0)
#     occupancy.add_argument("--rank_energy_threshold", type=float, default=0.90)
#     occupancy.add_argument("--rank_eigen_ratio_threshold", type=float, default=1e-2)
#     occupancy.add_argument("--rank_noise_ratio_threshold", type=float, default=1.50)
#     occupancy.add_argument("--min_active_rank", type=int, default=1)
#     occupancy.add_argument("--small_class_rank_threshold_1", type=int, default=30)
#     occupancy.add_argument("--small_class_rank_threshold_2", type=int, default=80)
#     occupancy.add_argument("--small_class_rank_threshold_3", type=int, default=150)
#     occupancy.add_argument("--small_class_rank_cap_1", type=int, default=1)
#     occupancy.add_argument("--small_class_rank_cap_2", type=int, default=3)
#     occupancy.add_argument("--small_class_rank_cap_3", type=int, default=4)
#     occupancy.add_argument("--small_class_extra_shrinkage", type=float, default=0.35)
#     occupancy.add_argument("--bootstrap_resamples", type=int, default=12)
#     occupancy.add_argument("--bootstrap_min_samples", type=int, default=8)
#     occupancy.add_argument("--bootstrap_max_samples", type=int, default=512)
#     occupancy.add_argument("--energy_logdet_weight", type=float, default=1.0)
#     occupancy.add_argument("--logit_scale", type=float, default=8.0)
#     occupancy.add_argument("--invalid_class_energy", type=float, default=1e6)
#     occupancy.add_argument("--geometry_logit_clip", type=float, default=0.0)

#     tangent = parser.add_argument_group("E. Conditional spectral tangent geometry")
#     tangent.add_argument("--num_spectral_interventions", type=int, default=2)
#     tangent.add_argument("--spectral_response_rank", type=int, default=2)
#     tangent.add_argument("--spectral_response_weight", type=float, default=0.20)
#     tangent.add_argument("--spectral_response_variance_floor", type=float, default=5e-4)
#     tangent.add_argument("--spectral_response_variance_shrinkage_tau", type=float, default=20.0)
#     tangent.add_argument("--spectral_response_mean_shrinkage_tau", type=float, default=5.0)
#     tangent.add_argument("--spectral_response_rank_energy_threshold", type=float, default=0.90)
#     tangent.add_argument("--spectral_response_rank_eigen_ratio_threshold", type=float, default=1e-2)
#     tangent.add_argument("--spectral_response_rank_noise_ratio_threshold", type=float, default=1.25)
#     tangent.add_argument("--response_logdet_weight", type=float, default=1.0)
#     tangent.add_argument("--spectral_tangent_coupling_ridge", type=float, default=1e-2)
#     tangent.add_argument("--spectral_tangent_coupling_shrinkage_tau", type=float, default=20.0)
#     tangent.add_argument("--spectral_tangent_coupling_max_norm", type=float, default=8.0)
#     tangent.add_argument("--intervention_definition_version", type=int, default=1)
#     tangent.add_argument("--spectral_gain_step", type=float, default=0.03)
#     tangent.add_argument("--spectral_tilt_step", type=float, default=0.03)
#     tangent.add_argument("--base_require_response_views", type=str2bool, default=True)

#     base = parser.add_argument_group("F. Base representation objective")
#     base.add_argument("--base_ce_weight", type=float, default=1.0)
#     base.add_argument("--pc_stgb_weight", type=float, default=0.10)
#     base.add_argument("--pc_margin", type=float, default=0.30)
#     base.add_argument("--pc_temperature", type=float, default=0.20)
#     base.add_argument("--boundary_certificate_weight", type=float, default=0.05)
#     base.add_argument("--boundary_certificate_margin", type=float, default=0.0)
#     base.add_argument("--boundary_certificate_temperature", type=float, default=0.20)
#     base.add_argument("--boundary_confidence_multiplier", type=float, default=1.0)
#     base.add_argument("--tangent_consistency_weight", type=float, default=0.0)
#     base.add_argument("--tangent_consistency_magnitude_weight", type=float, default=0.5)
#     base.add_argument("--tangent_consistency_direction_weight", type=float, default=0.5)
#     base.add_argument("--tangent_consistency_minimum_norm", type=float, default=1e-6)
#     base.add_argument("--base_class_balance", type=str2bool, default=True)
#     base.add_argument("--base_use_geometry_balanced_sampler", type=str2bool, default=True)
#     base.add_argument("--geometry_batch_classes", type=int, default=0)
#     base.add_argument("--geometry_samples_per_class", type=int, default=16)
#     base.add_argument("--base_min_samples_per_class_in_batch", type=int, default=16)
#     base.add_argument("--label_smoothing", type=float, default=0.0)
#     base.add_argument("--grad_clip_base", type=float, default=5.0)
#     base.add_argument("--base_admission_enforce", type=str2bool, default=False)
#     base.add_argument("--base_admission_min_oof_accuracy", type=float, default=0.0)
#     base.add_argument("--base_admission_min_class_accuracy", type=float, default=0.0)
#     base.add_argument("--base_admission_max_classification_violation", type=float, default=1.0)
#     base.add_argument("--base_admission_max_certificate_violation", type=float, default=1.0)
#     base.add_argument("--base_admission_max_invasion", type=float, default=1.0)
#     base.add_argument("--base_admission_require_tangent_help", type=str2bool, default=False)

#     inc = parser.add_argument_group("G. Incremental conditional admission")
#     inc.add_argument("--descriptor_lr", type=float, default=1e-3)
#     inc.add_argument("--incremental_steps_per_epoch", type=int, default=10)
#     inc.add_argument("--incremental_crossfit_folds", type=int, default=3)
#     inc.add_argument("--incremental_old_replay_samples_per_class", type=int, default=32)
#     inc.add_argument("--incremental_replay_reliability_gated", type=str2bool, default=True)
#     inc.add_argument("--incremental_margin", type=float, default=0.30)
#     inc.add_argument("--incremental_temperature", type=float, default=0.20)
#     inc.add_argument("--incremental_new_weight", type=float, default=1.0)
#     inc.add_argument("--incremental_old_replay_weight", type=float, default=1.0)
#     inc.add_argument("--incremental_crossfit_weight", type=float, default=1.0)
#     inc.add_argument("--incremental_certificate_weight", type=float, default=0.10)
#     inc.add_argument("--incremental_certificate_margin", type=float, default=0.0)
#     inc.add_argument("--incremental_certificate_temperature", type=float, default=0.20)
#     inc.add_argument("--incremental_certificate_kappa", type=float, default=1.0)
#     inc.add_argument("--incremental_invasion_weight", type=float, default=1.0)
#     inc.add_argument("--incremental_invasion_margin", type=float, default=0.30)
#     inc.add_argument("--incremental_invasion_temperature", type=float, default=0.20)
#     inc.add_argument("--incremental_trust_weight", type=float, default=0.05)
#     inc.add_argument("--incremental_max_mean_shift", type=float, default=0.25)
#     inc.add_argument("--incremental_max_log_eigval_shift", type=float, default=0.35)
#     inc.add_argument("--incremental_max_log_residual_shift", type=float, default=0.35)
#     inc.add_argument("--incremental_max_response_mean_shift", type=float, default=0.25)
#     inc.add_argument("--incremental_max_response_log_eigval_shift", type=float, default=0.35)
#     inc.add_argument("--incremental_max_response_log_residual_shift", type=float, default=0.35)
#     inc.add_argument("--incremental_grad_clip", type=float, default=5.0)
#     inc.add_argument("--incremental_admission_enforce", type=str2bool, default=False)
#     inc.add_argument("--incremental_min_crossfit_accuracy", type=float, default=0.0)
#     inc.add_argument("--incremental_min_crossfit_min_class_accuracy", type=float, default=0.0)
#     inc.add_argument("--incremental_max_crossfit_classification_violation", type=float, default=1.0)
#     inc.add_argument("--incremental_max_crossfit_certificate_violation", type=float, default=1.0)
#     inc.add_argument("--incremental_min_old_new_harmonic_mean", type=float, default=0.0)
#     inc.add_argument("--incremental_max_old_to_new_invasion", type=float, default=1.0)
#     inc.add_argument("--incremental_max_new_to_old_invasion", type=float, default=1.0)

#     runtime = parser.add_argument_group("H. Runtime identity and diagnostics")
#     runtime.add_argument("--classifier_mode", type=str, default="pc_stgb")
#     runtime.add_argument("--base_classifier_mode", type=str, default="base_ce")
#     runtime.add_argument("--eval_classifier_mode", type=str, default="pc_stgb")
#     runtime.add_argument("--incremental_update_mode", type=str, default="pc_stgb_row_replay")
#     runtime.add_argument("--base_diagnostic_replay_samples_per_class", type=int, default=32)
#     runtime.add_argument("--checkpoint_requires_dataset_state", type=str2bool, default=True)
#     runtime.add_argument("--save_raw_arrays", type=str2bool, default=False)
#     runtime.add_argument("--save_model_summary", type=str2bool, default=True)
#     runtime.add_argument("--save_visualizations", type=str2bool, default=True)
#     runtime.add_argument("--skip_phase_maps", type=str2bool, default=False)
#     runtime.add_argument("--save_qualitative_all_labeled_map", type=str2bool, default=True)
#     runtime.add_argument(
#         "--qualitative_map_reload_from_source",
#         type=str2bool,
#         default=True,
#         help=(
#             "Reload the raw scene after each committed phase only for no-gradient "
#             "qualitative map generation. Reloaded samples are never exposed to the "
#             "trainer and are destroyed immediately after rendering."
#         ),
#     )
#     runtime.add_argument("--visualization_dpi", type=int, default=300)
#     runtime.add_argument("--visualization_batch_size", type=int, default=128)
#     runtime.add_argument("--viz_class_cmap", type=str, default="nipy_spectral")
#     runtime.add_argument("--viz_background_color", type=str, default="#20252B")
#     runtime.add_argument("--debug_verbose", type=str2bool, default=False)

#     return parser


# def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
#     return build_parser().parse_args(argv)


# def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
#     resolved = argparse.Namespace(**namespace_to_dict(args))
#     resolved.device = resolve_device(resolved.device)
#     resolved.base_only = bool(resolved.base_only or resolved.disable_incremental_training)
#     resolved.disable_incremental_training = bool(resolved.base_only)
#     if resolved.base_only:
#         resolved.epochs_inc = 0

#     # Model/classifier compatibility fields. They describe the same deployed
#     # conditional joint score; no fallback classifier or calibration is enabled.
#     resolved.loss_scale = float(resolved.logit_scale)
#     resolved.energy_normalize_by_dim = True
#     resolved.normalize_energy_by_dim = True
#     resolved.use_logdet_energy = True
#     resolved.logdet_energy_weight = float(resolved.energy_logdet_weight)
#     resolved.ce_logit_clip = 0.0
#     resolved.base_checkpoint_requires_dataset_state = bool(
#         resolved.checkpoint_requires_dataset_state
#     )

#     # Temporary aliases consumed by older reporting hooks. The revised base
#     # trainer prefers the PC-STGB names above.
#     resolved.sgc_weight = float(resolved.pc_stgb_weight)
#     resolved.sgc_margin = float(resolved.pc_margin)
#     resolved.sgc_temperature = float(resolved.pc_temperature)
#     resolved.sgc_boundary_weight = float(resolved.boundary_certificate_weight)
#     resolved.sgc_boundary_margin = float(resolved.boundary_certificate_margin)

#     # Contradictory branches are fixed off so saved configuration files state
#     # the actual architecture unambiguously.
#     for name in (
#         "use_incremental_adapter",
#         "use_geometry_gated_adapter",
#         "use_geometry_transport",
#         "use_sglat_transport",
#         "allow_old_model_transport",
#         "use_energy_calibrator",
#         "use_geometry_calibrator",
#         "use_adaptive_boundary",
#         "allow_incremental_projection_training",
#         "use_raw_spectral_gaussian",
#         "use_spectral_feature_coupling",
#         "use_independent_geometry_replay",
#         "use_independent_response_replay",
#     ):
#         setattr(resolved, name, False)
#     return resolved


# def validate_config(args: argparse.Namespace, num_classes: Optional[int] = None) -> None:
#     if not bool(args.strict_non_exemplar):
#         raise ValueError("strict_non_exemplar must be true")
#     if str(args.incremental_update_mode).strip().lower().replace("-", "_") != "pc_stgb_row_replay":
#         raise ValueError("incremental_update_mode must be pc_stgb_row_replay")
#     if str(args.classifier_mode).strip().lower().replace("-", "_") != "pc_stgb":
#         raise ValueError("classifier_mode must be pc_stgb")
#     if str(args.eval_classifier_mode).strip().lower().replace("-", "_") != "pc_stgb":
#         raise ValueError("eval_classifier_mode must be pc_stgb")
#     if str(args.base_classifier_mode).strip().lower().replace("-", "_") != "base_ce":
#         raise ValueError("base_classifier_mode must be base_ce")

#     if int(args.patch_size) <= 0 or int(args.patch_size) % 2 == 0:
#         raise ValueError("patch_size must be a positive odd integer")
#     if not 0.0 < float(args.train_ratio) < 1.0:
#         raise ValueError("train_ratio must be in (0,1)")
#     if not 0.0 <= float(args.val_ratio) < 1.0:
#         raise ValueError("val_ratio must be in [0,1)")
#     if float(args.train_ratio) + float(args.val_ratio) >= 1.0:
#         raise ValueError("train_ratio + val_ratio must be < 1")
#     if int(args.epochs_base) <= 0 or int(args.batch_size) <= 0:
#         raise ValueError("epochs_base and batch_size must be positive")
#     if int(args.eval_batch_size) <= 0:
#         raise ValueError("eval_batch_size must be positive")
#     if int(args.epochs_inc) < 0:
#         raise ValueError("epochs_inc must be non-negative")
#     if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
#         raise ValueError("lr must be positive and weight_decay non-negative")
#     if not math.isfinite(float(args.lr_inc)):
#         raise ValueError("lr_inc must be finite")
#     if int(args.num_runs) <= 0:
#         raise ValueError("num_runs must be positive")
#     if int(args.max_phase) < -1:
#         raise ValueError("max_phase must be -1 or a non-negative phase index")

#     if int(args.d_model) <= 0 or not 0 < int(args.subspace_rank) < int(args.d_model):
#         raise ValueError("Require d_model > 0 and 0 < subspace_rank < d_model")
#     if bool(args.normalize_geometry_features):
#         raise ValueError("normalize_geometry_features must be false")
#     if abs(float(args.geometry_feature_scale) - 1.0) > 1e-12:
#         raise ValueError("geometry_feature_scale must equal 1.0")
#     if abs(float(args.geometry_feature_clamp)) > 1e-12:
#         raise ValueError("geometry_feature_clamp must equal 0.0")
#     if float(args.geom_var_floor) <= 0.0 or float(args.energy_logdet_weight) <= 0.0:
#         raise ValueError("Geometry variance floor and log-volume weight must be positive")
#     if abs(float(args.geometry_logit_clip)) > 1e-12:
#         raise ValueError("geometry_logit_clip must equal zero")

#     if int(args.num_spectral_interventions) != 2:
#         raise ValueError(
#             "intervention definition v1 contains exactly two physical directions: "
#             "multiplicative gain and smooth wavelength tilt"
#         )
#     if int(args.intervention_definition_version) != 1:
#         raise ValueError("intervention_definition_version must equal 1")
#     if not 0 < int(args.spectral_response_rank) <= int(args.d_model):
#         raise ValueError("spectral_response_rank must lie in [1,d_model]")
#     positive_tangent = (
#         "spectral_response_weight",
#         "spectral_response_variance_floor",
#         "response_logdet_weight",
#         "spectral_tangent_coupling_ridge",
#         "spectral_tangent_coupling_shrinkage_tau",
#         "spectral_tangent_coupling_max_norm",
#         "spectral_gain_step",
#         "spectral_tilt_step",
#     )
#     for name in positive_tangent:
#         if _finite_float(getattr(args, name), name) <= 0.0:
#             raise ValueError(f"{name} must be positive")
#     if max(float(args.spectral_gain_step), float(args.spectral_tilt_step)) >= 0.25:
#         raise ValueError("spectral intervention steps are too large for a local tangent")
#     if not bool(args.base_require_response_views):
#         raise ValueError("base_require_response_views must be true")

#     if float(args.base_ce_weight) <= 0.0:
#         raise ValueError("base_ce_weight must be positive")
#     if float(args.pc_stgb_weight) <= 0.0 or float(args.pc_temperature) <= 0.0:
#         raise ValueError("pc_stgb_weight and pc_temperature must be positive")
#     if float(args.pc_margin) < 0.0:
#         raise ValueError("pc_margin must be non-negative")
#     if float(args.boundary_certificate_weight) < 0.0:
#         raise ValueError("boundary_certificate_weight must be non-negative")
#     if float(args.boundary_certificate_margin) < 0.0:
#         raise ValueError("boundary_certificate_margin must be non-negative")
#     if float(args.boundary_certificate_temperature) <= 0.0:
#         raise ValueError("boundary_certificate_temperature must be positive")
#     if float(args.boundary_confidence_multiplier) < 0.0:
#         raise ValueError("boundary_confidence_multiplier must be non-negative")
#     if float(args.tangent_consistency_weight) != 0.0:
#         raise ValueError(
#             "The current IncrementalHSIDataset does not provide true h/2 intervention "
#             "views. Keep tangent_consistency_weight=0 until those views are added; "
#             "do not fabricate them by interpolation."
#         )
#     if int(args.geometry_samples_per_class) < 6 or int(args.geometry_samples_per_class) % 2:
#         raise ValueError("geometry_samples_per_class must be even and at least 6")
#     if int(args.base_min_samples_per_class_in_batch) < 6:
#         raise ValueError("base_min_samples_per_class_in_batch must be at least 6")
#     if int(args.geometry_batch_classes) == 1 or int(args.geometry_batch_classes) < 0:
#         raise ValueError("geometry_batch_classes must be 0 or at least 2")

#     if not bool(args.base_only):
#         if float(args.descriptor_lr) <= 0.0:
#             raise ValueError("descriptor_lr must be positive")
#         if int(args.incremental_steps_per_epoch) <= 0 and int(args.epochs_inc) > 0:
#             raise ValueError("incremental_steps_per_epoch must be positive")
#         if int(args.incremental_crossfit_folds) < 2:
#             raise ValueError("incremental_crossfit_folds must be at least 2")
#         if int(args.incremental_old_replay_samples_per_class) <= 0:
#             raise ValueError("incremental_old_replay_samples_per_class must be positive")
#         for name in (
#             "incremental_temperature",
#             "incremental_certificate_temperature",
#             "incremental_invasion_temperature",
#             "incremental_grad_clip",
#             "incremental_max_mean_shift",
#             "incremental_max_log_eigval_shift",
#             "incremental_max_log_residual_shift",
#             "incremental_max_response_mean_shift",
#             "incremental_max_response_log_eigval_shift",
#             "incremental_max_response_log_residual_shift",
#         ):
#             if _finite_float(getattr(args, name), name) <= 0.0:
#                 raise ValueError(f"{name} must be positive")
#         for name in (
#             "incremental_margin",
#             "incremental_new_weight",
#             "incremental_old_replay_weight",
#             "incremental_crossfit_weight",
#             "incremental_certificate_weight",
#             "incremental_certificate_margin",
#             "incremental_certificate_kappa",
#             "incremental_invasion_weight",
#             "incremental_invasion_margin",
#             "incremental_trust_weight",
#         ):
#             if _finite_float(getattr(args, name), name) < 0.0:
#                 raise ValueError(f"{name} must be non-negative")

#     if int(args.preprocess_response_chunk_size) <= 0:
#         raise ValueError("preprocess_response_chunk_size must be positive")
#     if int(args.visualization_batch_size) <= 0:
#         raise ValueError("visualization_batch_size must be positive")

#     seeds = parse_seed_list(args.seed_list)
#     if seeds is not None and len(seeds) != int(args.num_runs):
#         raise ValueError("seed_list length must equal num_runs")
#     if str(args.resume_checkpoint).strip() and int(args.num_runs) != 1:
#         raise ValueError("resume_checkpoint is supported only when num_runs=1")

#     if num_classes is not None:
#         class_count = int(num_classes)
#         if not 2 <= int(args.base_classes) <= class_count:
#             raise ValueError(f"base_classes must lie in [2,{class_count}]")
#         if int(args.increment) <= 0:
#             raise ValueError("increment must be positive")
#         if int(args.geometry_batch_classes) > int(args.base_classes):
#             raise ValueError("geometry_batch_classes cannot exceed base_classes")
#         order = parse_class_order(args.class_order)
#         if order is not None and set(order) != set(range(class_count)):
#             raise ValueError("class_order must be a complete sequential-class permutation")


# # =============================================================================
# # Data and strict non-exemplar ownership
# # =============================================================================


# def load_hsi_data(args: argparse.Namespace) -> Dict[str, Any]:
#     raw_hsi, gt, num_classes, raw_target_names, has_background, label_policy = LoadRawHSIData(
#         method=args.dataset,
#         base_dir=args.data_dir,
#     )
#     validate_config(args, num_classes=int(num_classes))
#     raw_hsi = np.asarray(raw_hsi, dtype=np.float32)
#     raw_band_count = int(raw_hsi.shape[-1])
#     gt = np.asarray(gt)
#     index_labels, index_coords = ExtractLabeledPixelIndex(gt, label_policy=label_policy)
#     protocol = IncrementalHSIDataset.build_protocol_plan(
#         index_labels,
#         base_classes=int(args.base_classes),
#         increment=int(args.increment),
#         train_ratio=float(args.train_ratio),
#         val_ratio=float(args.val_ratio),
#         min_train_per_class=int(args.min_train_per_class),
#         seed=int(args.seed),
#         shuffle_order=bool(args.shuffle_order),
#         class_order=parse_class_order(args.class_order),
#     )

#     fit_mask = BuildPreprocessFitMask(
#         gt.shape,
#         index_coords,
#         protocol["base_train_indices"],
#         labels=protocol["remapped_labels"],
#         allowed_classes=protocol["phase_to_classes"][0],
#     )
#     apply_reduction = (
#         not bool(args.no_pca)
#         and str(args.reduction_method).strip().lower() not in {"", "none", "identity"}
#     )
#     model_hsi, _, preprocessing_state = FitHSIPreprocessor(
#         raw_hsi,
#         fit_mask=fit_mask,
#         apply_reduction=apply_reduction,
#         reduction_method=str(args.reduction_method),
#         n_components=int(args.pca_components),
#         fit_scope="phase0_train_only",
#     )
#     preprocessing_path = SaveHSIPreprocessor(
#         os.path.join(os.path.abspath(str(args.save_dir)), "preprocessing_state.npz"),
#         preprocessing_state,
#     )

#     patches, labels_from_cubes, coords = ImageCubes(
#         model_hsi,
#         gt,
#         WS=int(args.patch_size),
#         removeZeroLabels=True,
#         has_background=bool(has_background),
#         num_classes=int(num_classes),
#         pytorch_format=True,
#         label_policy=label_policy,
#         return_center_spectra=False,
#     )
#     patches = np.ascontiguousarray(to_numpy(patches), dtype=np.float32)
#     labels_from_cubes = np.ascontiguousarray(
#         to_numpy(labels_from_cubes, dtype=np.int64).reshape(-1)
#     )
#     coords = np.ascontiguousarray(to_numpy(coords, dtype=np.int64))
#     if not np.array_equal(labels_from_cubes, np.asarray(index_labels, dtype=np.int64)):
#         raise RuntimeError("ImageCubes changed the labeled-pixel order")
#     if not np.array_equal(coords, np.asarray(index_coords, dtype=np.int64)):
#         raise RuntimeError("ImageCubes changed the coordinate order")

#     positive_centers, negative_centers, step_sizes, intervention_metadata = (
#         BuildPCSTGBInterventionCentersFromCube(
#             raw_hsi,
#             coords,
#             preprocessing_state,
#             gain_step=float(args.spectral_gain_step),
#             tilt_step=float(args.spectral_tilt_step),
#             chunk_size=int(args.preprocess_response_chunk_size),
#             definition_version=int(args.intervention_definition_version),
#         )
#     )
#     intervention_metadata = dict(intervention_metadata)
#     intervention_metadata.update(
#         {
#             "method": METHOD_NAME,
#             "spectral_object": "central_finite_difference_canonical_feature_tangent",
#             "joint_factorization": JOINT_FACTORIZATION,
#             "persistent_raw_spectra": False,
#         }
#     )

#     raw_centers = np.ascontiguousarray(
#         raw_hsi[coords[:, 0], coords[:, 1], :], dtype=np.float32
#     )
#     maximum_center_error = ValidatePCSTGBCenterAlignment(
#         patches,
#         raw_centers,
#         preprocessing_state,
#         tolerance=5e-4,
#         chunk_size=int(args.preprocess_response_chunk_size),
#     )
#     del raw_centers, model_hsi, raw_hsi
#     gc.collect()

#     args.num_bands = int(patches.shape[1])
#     args.max_classes = int(num_classes)
#     args.raw_num_bands = int(raw_band_count)
#     target_names = [
#         str(raw_target_names[int(protocol["inv_label_map"][class_id])])
#         for class_id in range(int(num_classes))
#     ]
#     summary = {
#         "stack_build_id": STACK_BUILD_ID,
#         "method": METHOD_NAME,
#         "schema_version": BANK_SCHEMA_VERSION,
#         "joint_factorization": JOINT_FACTORIZATION,
#         "dataset": str(args.dataset),
#         "dataset_name": DATASET_INFO[str(args.dataset)]["name"],
#         "num_classes": int(num_classes),
#         "num_phases": int(protocol["num_phases"]),
#         "phase_to_classes": protocol["phase_to_classes"],
#         "class_order_input_ids": list(protocol["class_order"]),
#         "labeled_samples": int(protocol["remapped_labels"].size),
#         "base_train_samples": int(protocol["base_train_indices"].size),
#         "raw_sensor_bands": int(raw_band_count),
#         "model_input_channels": int(patches.shape[1]),
#         "patch_size": int(args.patch_size),
#         "pca_active": bool(apply_reduction),
#         "preprocessing_fit_scope": "phase0_train_only",
#         "preprocessing_fit_pixel_count": int(np.asarray(fit_mask).sum()),
#         "preprocessing_state_path": preprocessing_path,
#         "intervention_metadata": intervention_metadata,
#         "maximum_center_preprocessing_error": maximum_center_error,
#         "persistent_memory": (
#             "aggregate occupancy rows, occupancy-tangent coupling, conditional "
#             "tangent residual rows, frozen global tangent prior, and row statistics"
#         ),
#         "strict_non_exemplar": True,
#     }
#     return {
#         "gt": gt,
#         "patches": patches,
#         "input_labels": labels_from_cubes,
#         "coords": coords,
#         "positive_centers": positive_centers,
#         "negative_centers": negative_centers,
#         "step_sizes": step_sizes,
#         "protocol": protocol,
#         "target_names": target_names,
#         "raw_target_names": [str(value) for value in raw_target_names],
#         "label_policy": dict(label_policy),
#         "num_classes": int(num_classes),
#         "summary": summary,
#     }


# def build_dataset(
#     args: argparse.Namespace, data: Mapping[str, Any]
# ) -> IncrementalHSIDataset:
#     dataset = IncrementalHSIDataset(
#         patches=data["patches"],
#         labels=np.asarray(data["input_labels"], dtype=np.int64),
#         coords=data["coords"],
#         gt_shape=tuple(np.asarray(data["gt"]).shape[:2]),
#         GT=data["gt"],
#         base_classes=int(args.base_classes),
#         increment=int(args.increment),
#         train_ratio=float(args.train_ratio),
#         val_ratio=float(args.val_ratio),
#         seed=int(args.seed),
#         shuffle_order=bool(args.shuffle_order),
#         min_train_per_class=int(args.min_train_per_class),
#         num_workers=int(args.num_workers),
#         device=str(args.device),
#         strict_non_exemplar=bool(args.strict_non_exemplar),
#         target_names=list(data["raw_target_names"]),
#         label_policy=dict(data["label_policy"]),
#         class_balanced_train_batches=bool(args.base_class_balance),
#         strict_unique_balanced_batches=True,
#         geometry_batch_classes=int(args.geometry_batch_classes),
#         geometry_samples_per_class=int(args.geometry_samples_per_class),
#         spectral_positive_centers=data["positive_centers"],
#         spectral_negative_centers=data["negative_centers"],
#         spectral_step_sizes=data["step_sizes"],
#         intervention_metadata=data["summary"]["intervention_metadata"],
#         require_response_views=True,
#         purge_finalized_train_payload=True,
#         protocol_plan=data["protocol"],
#     )
#     response_contract = dataset.get_response_contract()
#     if not bool(response_contract.get("available", False)):
#         raise RuntimeError("Dataset did not install paired PC-STGB intervention views")
#     if int(response_contract.get("definition_version", -1)) != int(
#         args.intervention_definition_version
#     ):
#         raise RuntimeError("Dataset/model intervention-definition version mismatch")
#     if int(response_contract.get("num_interventions", -1)) != int(
#         args.num_spectral_interventions
#     ):
#         raise RuntimeError("Dataset/model intervention-count mismatch")
#     return dataset


# def release_external_sample_payload(data: Dict[str, Any]) -> Dict[str, Any]:
#     """Drop duplicate sample arrays after IncrementalHSIDataset takes ownership.

#     IncrementalHSIDataset copies all arrays. Retaining the original arrays in
#     ``main.py`` would leave a second, unpurged copy of old training patches and
#     intervention centers after phase finalization, violating the strict physical
#     non-exemplar contract even though dataset access control is correct.
#     """
#     retained = {
#         "protocol": data.get("protocol"),
#         "summary": data.get("summary"),
#         "target_names": data.get("target_names"),
#         "raw_target_names": data.get("raw_target_names"),
#         "label_policy": data.get("label_policy"),
#         "num_classes": data.get("num_classes"),
#     }
#     data.clear()
#     gc.collect()
#     return retained


# # =============================================================================
# # Model construction and evaluation
# # =============================================================================


# def build_model(args: argparse.Namespace) -> NECILModel:
#     model = NECILModel(args).to(torch.device(args.device))
#     model.assert_architecture_contract()
#     contract = model.feature_contract()
#     expected = {
#         "method": METHOD_NAME,
#         "geometry_bank_schema_version": BANK_SCHEMA_VERSION,
#         "joint_factorization": JOINT_FACTORIZATION,
#         "classifier_contract": "dimension_normalized_conditional_joint_energy",
#         "inference_geometry": "occupancy_conditioned_spectral_tangent_geometry",
#         "occupancy_tangent_coupling": True,
#         "independent_response_factorization": False,
#         "raw_spectral_gaussian": False,
#     }
#     bad = {
#         key: (contract.get(key), value)
#         for key, value in expected.items()
#         if contract.get(key) != value
#     }
#     if bad:
#         raise RuntimeError(f"Model architecture contract mismatch: {bad}")
#     classifier_contract = model.classifier.classifier_contract()
#     if not bool(classifier_contract.get("uses_coupling_inference_score", False)):
#         raise RuntimeError("Classifier does not use occupancy-tangent coupling")
#     return model


# def _batch_tensor(batch: Mapping[str, Any], names: Sequence[str], name: str) -> torch.Tensor:
#     for key in names:
#         value = batch.get(key)
#         if torch.is_tensor(value):
#             return value
#     raise RuntimeError(f"Evaluation batch is missing {name}")


# def _joint_batch(
#     batch: Any, device: torch.device
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     if not isinstance(batch, Mapping):
#         raise RuntimeError("PC-STGB evaluation requires mapping batches")
#     x = _batch_tensor(batch, ("image", "patch", "patches"), "image")
#     labels = _batch_tensor(batch, ("label", "labels", "target"), "label")
#     positive = _batch_tensor(
#         batch,
#         ("spectral_positive_patches", "positive_intervention_patches"),
#         "positive intervention views",
#     )
#     negative = _batch_tensor(
#         batch,
#         ("spectral_negative_patches", "negative_intervention_patches"),
#         "negative intervention views",
#     )
#     steps = _batch_tensor(
#         batch,
#         ("spectral_step_sizes", "intervention_step_sizes"),
#         "intervention step sizes",
#     )
#     return (
#         x.to(device=device, dtype=torch.float32, non_blocking=True),
#         labels.to(device=device, dtype=torch.long, non_blocking=True).flatten(),
#         positive.to(device=device, dtype=torch.float32, non_blocking=True),
#         negative.to(device=device, dtype=torch.float32, non_blocking=True),
#         steps.to(device=device, dtype=torch.float32, non_blocking=True),
#     )


# def _seen_classes(dataset: IncrementalHSIDataset, phase: int) -> List[int]:
#     if hasattr(dataset, "get_seen_classes"):
#         return [int(value) for value in dataset.get_seen_classes(int(phase))]
#     values: List[int] = []
#     for index in range(int(phase) + 1):
#         values.extend(int(value) for value in dataset.phase_to_classes[index])
#     return values


# def _phase_classes(dataset: IncrementalHSIDataset, phase: int) -> List[int]:
#     return [int(value) for value in dataset.phase_to_classes[int(phase)]]


# @torch.no_grad()
# def evaluate_phase_geometry(
#     model: NECILModel,
#     dataset: IncrementalHSIDataset,
#     phase: int,
#     *,
#     batch_size: int,
#     device: torch.device,
#     split: str = "test",
# ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, torch.Tensor]:
#     phase = int(phase)
#     seen_classes = _seen_classes(dataset, phase)
#     new_classes = _phase_classes(dataset, phase)
#     old_classes = seen_classes[: len(seen_classes) - len(new_classes)]
#     loader = dataset.get_cumulative_dataloader(
#         phase,
#         split=str(split),
#         batch_size=int(batch_size),
#         shuffle=False,
#     )

#     joint_parts: List[torch.Tensor] = []
#     occupancy_parts: List[torch.Tensor] = []
#     tangent_parts: List[torch.Tensor] = []
#     labels_parts: List[torch.Tensor] = []
#     model.eval()
#     for batch in loader:
#         x, labels, positive, negative, steps = _joint_batch(batch, device)
#         joint_tuple = model.extract_joint_geometry_tuple(
#             x,
#             positive,
#             negative,
#             step_sizes=steps,
#             deterministic=True,
#             return_view_features=False,
#         )
#         if joint_tuple.get("joint_factorization") != JOINT_FACTORIZATION:
#             raise RuntimeError("Model returned the wrong joint tuple factorization")
#         scored = model.compute_logits_from_features(
#             joint_tuple["features"],
#             spectral_responses=joint_tuple["spectral_responses"],
#             seen_classes=seen_classes,
#             mode="pc_stgb",
#             return_energy=True,
#             return_parts=True,
#         )
#         if scored.get("joint_factorization") != JOINT_FACTORIZATION:
#             raise RuntimeError("Classifier returned the wrong joint factorization")
#         if not bool(scored.get("uses_coupling_inference_score", False)):
#             raise RuntimeError("Evaluation score does not use occupancy-tangent coupling")
#         joint_parts.append(scored["joint_energy"].detach().cpu())
#         occupancy_parts.append(scored["occupancy_energy"].detach().cpu())
#         tangent_parts.append(scored["conditional_tangent_energy"].detach().cpu())
#         labels_parts.append(labels.detach().cpu())

#     if not joint_parts:
#         raise RuntimeError(f"Phase {phase} {split} loader is empty")
#     joint = torch.cat(joint_parts, dim=0)
#     occupancy = torch.cat(occupancy_parts, dim=0)
#     tangent = torch.cat(tangent_parts, dim=0)
#     labels = torch.cat(labels_parts, dim=0)

#     class_ids = list(seen_classes)
#     local_map = {class_id: column for column, class_id in enumerate(class_ids)}
#     try:
#         local = torch.tensor(
#             [local_map[int(value)] for value in labels.tolist()], dtype=torch.long
#         )
#     except KeyError as error:
#         raise RuntimeError(f"Evaluation labels contain an unseen class: {error}") from error

#     joint_pred_local = joint.argmin(dim=1)
#     occupancy_pred_local = occupancy.argmin(dim=1)
#     tangent_pred_local = tangent.argmin(dim=1)
#     class_tensor = torch.tensor(class_ids, dtype=torch.long)
#     joint_pred = class_tensor.index_select(0, joint_pred_local)

#     true_energy = joint.gather(1, local[:, None]).squeeze(1)
#     rivals = joint.clone()
#     rivals.scatter_(1, local[:, None], float("inf"))
#     gaps = rivals.min(dim=1).values - true_energy
#     correct = joint_pred.eq(labels)
#     occupancy_correct = occupancy_pred_local.eq(local)
#     tangent_correct = tangent_pred_local.eq(local)

#     class_count = len(class_ids)
#     confusion = torch.zeros((class_count, class_count), dtype=torch.long)
#     for true_id, pred_id in zip(local.tolist(), joint_pred_local.tolist()):
#         confusion[true_id, pred_id] += 1
#     support = confusion.sum(1).double()
#     predicted = confusion.sum(0).double()
#     true_positive = confusion.diag().double()
#     precision = torch.where(
#         predicted > 0, true_positive / predicted, torch.zeros_like(true_positive)
#     )
#     recall = torch.where(
#         support > 0, true_positive / support, torch.zeros_like(true_positive)
#     )
#     f1 = torch.where(
#         precision + recall > 0,
#         2 * precision * recall / (precision + recall),
#         torch.zeros_like(precision),
#     )

#     invasion = torch.zeros((class_count, class_count), dtype=torch.float32)
#     per_class: Dict[int, Dict[str, Any]] = {}
#     for local_id, class_id in enumerate(class_ids):
#         mask = local.eq(local_id)
#         class_gaps = gaps[mask]
#         for rival_id in range(class_count):
#             if rival_id != local_id:
#                 invasion[local_id, rival_id] = (
#                     joint[mask, rival_id] <= joint[mask, local_id]
#                 ).float().mean()
#         per_class[class_id] = {
#             "class_id": int(class_id),
#             "class_name": _class_name(dataset.target_names, class_id),
#             "phase_created": int(dataset.class_to_phase[class_id]),
#             "support": int(mask.sum().item()),
#             "correct": int(correct[mask].sum().item()),
#             "accuracy": 100.0 * float(correct[mask].float().mean().item()),
#             "precision": float(precision[local_id].item()),
#             "recall": float(recall[local_id].item()),
#             "f1": float(f1[local_id].item()),
#             "mean_gap": float(class_gaps.mean().item()),
#             "q05_gap": float(torch.quantile(class_gaps, 0.05).item()),
#             "minimum_gap": float(class_gaps.min().item()),
#             "classification_violation_rate": float(
#                 class_gaps.le(0.0).float().mean().item()
#             ),
#         }

#     old_set = set(old_classes)
#     new_set = set(new_classes)
#     old_mask = torch.tensor([int(value) in old_set for value in labels.tolist()])
#     new_mask = torch.tensor([int(value) in new_set for value in labels.tolist()])
#     old_accuracy = (
#         100.0 * float(correct[old_mask].float().mean().item())
#         if bool(old_mask.any())
#         else 0.0
#     )
#     new_accuracy = (
#         100.0 * float(correct[new_mask].float().mean().item())
#         if bool(new_mask.any())
#         else 100.0 * float(correct.float().mean().item())
#     )
#     harmonic = (
#         0.0
#         if phase > 0 and old_accuracy + new_accuracy <= 0.0
#         else (
#             new_accuracy
#             if phase == 0
#             else 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)
#         )
#     )
#     old_to_new = (
#         float(
#             torch.tensor(
#                 [int(value) in new_set for value in joint_pred[old_mask].tolist()],
#                 dtype=torch.float32,
#             ).mean().item()
#         )
#         if bool(old_mask.any())
#         else 0.0
#     )
#     new_to_old = (
#         float(
#             torch.tensor(
#                 [int(value) in old_set for value in joint_pred[new_mask].tolist()],
#                 dtype=torch.float32,
#             ).mean().item()
#         )
#         if bool(new_mask.any()) and old_set
#         else 0.0
#     )

#     per_class_accuracies = [float(row["accuracy"]) for row in per_class.values()]
#     metrics: Dict[str, Any] = {
#         "phase": phase,
#         "method": METHOD_NAME,
#         "schema_version": BANK_SCHEMA_VERSION,
#         "joint_factorization": JOINT_FACTORIZATION,
#         "split": str(split),
#         "seen_classes": class_ids,
#         "old_classes": old_classes,
#         "new_classes": new_classes,
#         "overall_accuracy": 100.0 * float(correct.float().mean().item()),
#         "average_accuracy": float(np.mean(per_class_accuracies)),
#         "minimum_per_class_accuracy": float(min(per_class_accuracies)),
#         "old_accuracy": old_accuracy,
#         "new_accuracy": new_accuracy,
#         "old_new_harmonic_mean": harmonic,
#         "old_to_new_invasion_rate": old_to_new,
#         "new_to_old_invasion_rate": new_to_old,
#         "joint_accuracy": 100.0 * float(correct.float().mean().item()),
#         "occupancy_only_accuracy": 100.0 * float(
#             occupancy_correct.float().mean().item()
#         ),
#         "conditional_tangent_accuracy": 100.0 * float(
#             tangent_correct.float().mean().item()
#         ),
#         "conditional_tangent_help_rate": float(
#             ((~occupancy_correct) & correct).float().mean().item()
#         ),
#         "conditional_tangent_harm_rate": float(
#             (occupancy_correct & (~correct)).float().mean().item()
#         ),
#         "mean_energy_gap": float(gaps.mean().item()),
#         "q05_energy_gap": float(torch.quantile(gaps, 0.05).item()),
#         "minimum_energy_gap": float(gaps.min().item()),
#         "classification_violation_rate": float(gaps.le(0.0).float().mean().item()),
#         "maximum_directional_invasion": float(invasion.max().item()),
#         "mean_directional_invasion": float(
#             invasion[~torch.eye(class_count, dtype=torch.bool)].mean().item()
#             if class_count > 1
#             else 0.0
#         ),
#         "directional_invasion_matrix": invasion,
#         "confusion_matrix": confusion,
#         "macro_precision": float(precision.mean().item()),
#         "macro_recall": float(recall.mean().item()),
#         "macro_f1": float(f1.mean().item()),
#         "per_class_metrics": per_class,
#         "sample_count": int(labels.numel()),
#         "geometry_bank_contract_digest": model.geometry_bank.contract_digest(),
#         "classifier_bound_contract_digest": model.classifier.bound_contract_digest,
#         "model_contract_digest": model.model_contract_digest(),
#     }
#     # Compatibility names for old analysis scripts. The term is conditional.
#     metrics["response_only_accuracy"] = metrics["conditional_tangent_accuracy"]
#     metrics["response_help_rate"] = metrics["conditional_tangent_help_rate"]
#     metrics["response_harm_rate"] = metrics["conditional_tangent_harm_rate"]
#     return metrics, labels.numpy(), joint_pred.numpy(), joint


# @torch.no_grad()
# def replay_health(
#     model: NECILModel,
#     class_ids: Sequence[int],
#     *,
#     samples_per_class: int,
# ) -> Dict[str, Any]:
#     ids = [int(value) for value in class_ids]
#     replay = model.sample_geometry_replay(
#         ids,
#         samples_per_class=int(samples_per_class),
#         seen_classes=ids,
#         reliability_gated=True,
#     )
#     features = replay.get("features")
#     responses = replay.get("spectral_responses", replay.get("responses"))
#     labels = replay.get("global_labels")
#     if not torch.is_tensor(features) or features.numel() == 0:
#         raise RuntimeError("Geometry replay produced no samples")
#     if not torch.is_tensor(responses) or responses.size(0) != features.size(0):
#         raise RuntimeError("Geometry replay did not return aligned conditional tangents")
#     if not torch.is_tensor(labels) or labels.numel() != features.size(0):
#         raise RuntimeError("Geometry replay labels are missing or misaligned")
#     scored = model.compute_logits_from_features(
#         features,
#         spectral_responses=responses,
#         seen_classes=ids,
#         mode="pc_stgb",
#         return_energy=True,
#         return_parts=True,
#     )
#     if scored.get("joint_factorization") != JOINT_FACTORIZATION:
#         raise RuntimeError("Replay scoring used the wrong joint factorization")
#     energy = scored["joint_energy"]
#     local_map = {class_id: index for index, class_id in enumerate(ids)}
#     local = torch.tensor(
#         [local_map[int(value)] for value in labels.detach().cpu().tolist()],
#         device=energy.device,
#         dtype=torch.long,
#     )
#     true = energy.gather(1, local[:, None]).squeeze(1)
#     rival = energy.clone()
#     rival.scatter_(1, local[:, None], float("inf"))
#     gap = rival.min(dim=1).values - true
#     prediction = energy.argmin(dim=1)
#     return {
#         "replay_factorization": "z_then_g_given_z_c",
#         "joint_factorization": JOINT_FACTORIZATION,
#         "samples": int(local.numel()),
#         "accuracy": 100.0 * float(prediction.eq(local).float().mean().item()),
#         "mean_gap": float(gap.mean().item()),
#         "q05_gap": float(torch.quantile(gap, 0.05).item()),
#         "minimum_gap": float(gap.min().item()),
#         "violation_rate": float(gap.le(0.0).float().mean().item()),
#     }


# def save_model_summary(path: str, model: torch.nn.Module) -> str:
#     total = sum(int(parameter.numel()) for parameter in model.parameters())
#     trainable = sum(
#         int(parameter.numel())
#         for parameter in model.parameters()
#         if parameter.requires_grad
#     )
#     lines = [
#         f"Model: {model.__class__.__name__}",
#         f"Stack: {STACK_BUILD_ID}",
#         f"Method: {METHOD_NAME}",
#         f"Joint factorization: {JOINT_FACTORIZATION}",
#         "=" * 104,
#         f"Total parameters: {total:,}",
#         f"Trainable parameters at save time: {trainable:,}",
#         "",
#         "Feature contract",
#         "-" * 104,
#     ]
#     for key, value in sorted(model.feature_contract().items()):
#         lines.append(f"{key}: {value}")
#     absolute = os.path.abspath(path)
#     os.makedirs(os.path.dirname(absolute), exist_ok=True)
#     temporary = absolute + ".tmp"
#     with open(temporary, "w", encoding="utf-8") as stream:
#         stream.write("\n".join(lines) + "\n")
#         stream.flush()
#         os.fsync(stream.fileno())
#     os.replace(temporary, absolute)
#     return absolute


# # =============================================================================
# # Phase reports and continual metrics
# # =============================================================================


# def _phase_report_text(
#     phase: int,
#     metrics: Mapping[str, Any],
#     diagnostics: Mapping[str, Any],
#     replay: Mapping[str, Any],
#     training: Mapping[str, Any],
# ) -> str:
#     lines = [
#         f"PC-STGB Phase {int(phase)} Geometry Report",
#         "=" * 112,
#         "",
#         "Method contract",
#         "-" * 112,
#         f"Factorization: {JOINT_FACTORIZATION}",
#         "Inference: dimension-normalized occupancy + occupancy-conditioned tangent energy",
#         "Replay: sample z first, then sample g conditioned on z and class",
#         "Memory: aggregate rows only; old rows and global prior are immutable",
#         "",
#         "Official cumulative test geometry",
#         "-" * 112,
#         (
#             f"OA={float(metrics['overall_accuracy']):.2f}% | "
#             f"AA={float(metrics['average_accuracy']):.2f}% | "
#             f"MinClass={float(metrics['minimum_per_class_accuracy']):.2f}% | "
#             f"Old={float(metrics['old_accuracy']):.2f}% | "
#             f"New={float(metrics['new_accuracy']):.2f}% | "
#             f"H={float(metrics['old_new_harmonic_mean']):.2f}%"
#         ),
#         (
#             f"Occupancy={float(metrics['occupancy_only_accuracy']):.2f}% | "
#             f"ConditionalTangent={float(metrics['conditional_tangent_accuracy']):.2f}% | "
#             f"TangentHelp={float(metrics['conditional_tangent_help_rate']):.4f} | "
#             f"TangentHarm={float(metrics['conditional_tangent_harm_rate']):.4f}"
#         ),
#         (
#             f"MeanGap={float(metrics['mean_energy_gap']):.6f} | "
#             f"Q05Gap={float(metrics['q05_energy_gap']):.6f} | "
#             f"MinGap={float(metrics['minimum_energy_gap']):.6f} | "
#             f"Violation={100.0 * float(metrics['classification_violation_rate']):.2f}%"
#         ),
#         (
#             f"Old->New={100.0 * float(metrics['old_to_new_invasion_rate']):.2f}% | "
#             f"New->Old={100.0 * float(metrics['new_to_old_invasion_rate']):.2f}% | "
#             f"MaxDirectionalInvasion={float(metrics['maximum_directional_invasion']):.4f}"
#         ),
#         "",
#         "Replay fidelity",
#         "-" * 112,
#         (
#             f"Accuracy={float(replay['accuracy']):.2f}% | "
#             f"MeanGap={float(replay['mean_gap']):.6f} | "
#             f"Q05Gap={float(replay['q05_gap']):.6f} | "
#             f"Violation={100.0 * float(replay['violation_rate']):.2f}%"
#         ),
#         "",
#         "Geometry readiness",
#         "-" * 112,
#         (
#             f"response_ready_rate={float(diagnostics.get('response_ready_rate', 0.0)):.4f} | "
#             f"coupling_ready_rate={float(diagnostics.get('coupling_ready_rate', 0.0)):.4f} | "
#             f"coupling_reliability_mean={float(diagnostics.get('coupling_reliability_mean', 0.0)):.4f} | "
#             f"coupling_explained_variance_mean={float(diagnostics.get('coupling_explained_variance_mean', 0.0)):.4f}"
#         ),
#         "",
#         "Training/admission state",
#         "-" * 112,
#         f"status={training.get('status', training.get('state', 'COMPLETE'))}",
#         f"committed={training.get('committed', phase == 0)}",
#         f"protocol_stop={training.get('protocol_stop', False)}",
#         "",
#         "Class-wise cumulative test results",
#         "-" * 112,
#         "id  introduced  class                              support  accuracy    q05-gap     violation",
#     ]
#     per_class = metrics["per_class_metrics"]
#     for class_id in metrics["seen_classes"]:
#         row = per_class[class_id]
#         lines.append(
#             f"{int(class_id):2d}  {int(row['phase_created']):10d}  "
#             f"{str(row['class_name'])[:32]:32s} {int(row['support']):7d}  "
#             f"{float(row['accuracy']):8.2f}%  {float(row['q05_gap']):10.5f}  "
#             f"{100.0 * float(row['classification_violation_rate']):8.2f}%"
#         )
#     return "\n".join(lines) + "\n"


# def write_phase_report(
#     path: str,
#     *,
#     phase: int,
#     metrics: Mapping[str, Any],
#     diagnostics: Mapping[str, Any],
#     replay: Mapping[str, Any],
#     training: Mapping[str, Any],
# ) -> str:
#     absolute = os.path.abspath(path)
#     os.makedirs(os.path.dirname(absolute), exist_ok=True)
#     temporary = absolute + ".tmp"
#     with open(temporary, "w", encoding="utf-8") as stream:
#         stream.write(
#             _phase_report_text(
#                 phase,
#                 metrics,
#                 diagnostics,
#                 replay,
#                 training,
#             )
#         )
#         stream.flush()
#         os.fsync(stream.fileno())
#     os.replace(temporary, absolute)
#     return absolute


# def build_continual_summary(
#     phase_results: Sequence[Mapping[str, Any]],
# ) -> Dict[str, Any]:
#     if not phase_results:
#         raise ValueError("phase_results is empty")
#     evaluated = [item for item in phase_results if isinstance(item.get("test_metrics"), Mapping)]
#     if not evaluated:
#         raise ValueError("phase_results contains no evaluated committed phase")
#     per_class_history: Dict[int, List[Tuple[int, float]]] = {}
#     introduction_accuracy: Dict[int, float] = {}
#     phase_rows: List[Dict[str, Any]] = []
#     for item in evaluated:
#         phase = int(item["phase"])
#         metrics = item["test_metrics"]
#         for class_id, row in metrics["per_class_metrics"].items():
#             class_id = int(class_id)
#             accuracy = float(row["accuracy"])
#             per_class_history.setdefault(class_id, []).append((phase, accuracy))
#             if int(row["phase_created"]) == phase:
#                 introduction_accuracy[class_id] = accuracy
#         phase_rows.append(
#             {
#                 "phase": phase,
#                 "seen_classes": list(metrics["seen_classes"]),
#                 "overall_accuracy": float(metrics["overall_accuracy"]),
#                 "average_accuracy": float(metrics["average_accuracy"]),
#                 "minimum_per_class_accuracy": float(
#                     metrics["minimum_per_class_accuracy"]
#                 ),
#                 "old_accuracy": float(metrics["old_accuracy"]),
#                 "new_accuracy": float(metrics["new_accuracy"]),
#                 "old_new_harmonic_mean": float(metrics["old_new_harmonic_mean"]),
#                 "old_to_new_invasion_rate": float(
#                     metrics["old_to_new_invasion_rate"]
#                 ),
#                 "new_to_old_invasion_rate": float(
#                     metrics["new_to_old_invasion_rate"]
#                 ),
#                 "q05_energy_gap": float(metrics["q05_energy_gap"]),
#                 "checkpoint": item.get("checkpoint"),
#                 "status": item.get("status", "COMPLETE"),
#             }
#         )

#     final_phase = int(phase_rows[-1]["phase"])
#     final_metrics = evaluated[-1]["test_metrics"]
#     forgetting: Dict[int, float] = {}
#     backward_transfer: Dict[int, float] = {}
#     for class_id, trajectory in per_class_history.items():
#         final_accuracy = trajectory[-1][1]
#         previous = [accuracy for phase, accuracy in trajectory if phase < final_phase]
#         forgetting[class_id] = (
#             max(previous) - final_accuracy if previous else 0.0
#         )
#         if class_id in introduction_accuracy:
#             backward_transfer[class_id] = (
#                 final_accuracy - introduction_accuracy[class_id]
#             )

#     old_class_ids = [
#         int(class_id)
#         for class_id in final_metrics["seen_classes"]
#         if int(final_metrics["per_class_metrics"][class_id]["phase_created"]) < final_phase
#     ]
#     average_forgetting = (
#         float(np.mean([forgetting[class_id] for class_id in old_class_ids]))
#         if old_class_ids
#         else 0.0
#     )
#     average_bwt = (
#         float(np.mean([backward_transfer[class_id] for class_id in old_class_ids]))
#         if old_class_ids
#         else 0.0
#     )
#     return {
#         "method": METHOD_NAME,
#         "stack_build_id": STACK_BUILD_ID,
#         "joint_factorization": JOINT_FACTORIZATION,
#         "completed_phases": [row["phase"] for row in phase_rows],
#         "phase_results": phase_rows,
#         "final_phase": final_phase,
#         "final_overall_accuracy": float(final_metrics["overall_accuracy"]),
#         "final_average_accuracy": float(final_metrics["average_accuracy"]),
#         "final_minimum_per_class_accuracy": float(
#             final_metrics["minimum_per_class_accuracy"]
#         ),
#         "average_forgetting": average_forgetting,
#         "average_backward_transfer": average_bwt,
#         "per_class_forgetting": forgetting,
#         "per_class_backward_transfer": backward_transfer,
#         "per_class_accuracy_history": per_class_history,
#         "protocol_stopped": bool(
#             any(
#                 item.get("status") == "REJECTED"
#                 or item.get("protocol_stop", False)
#                 for item in phase_results
#             )
#         ),
#         "summary_scope": (
#             f"phases_{phase_rows[0]['phase']}_through_{phase_rows[-1]['phase']}"
#         ),
#     }


# # =============================================================================
# # Evaluation-only qualitative scene reconstruction
# # =============================================================================


# class _TransientQualitativePhaseManager:
#     """A short-lived, evaluation-only loader for dense labeled-pixel maps.

#     The training dataset physically purges finalized training pixels.  A dense
#     qualitative map for a later phase therefore cannot be produced from the
#     training dataset without violating its access controls.  This manager is
#     reconstructed from the immutable source scene *after* a phase has committed,
#     is never attached to Trainer, runs only under ``torch.no_grad()``, and is
#     explicitly destroyed after rendering.
#     """

#     def __init__(
#         self,
#         *,
#         patch_dataset: HSIPatchDataset,
#         phase_to_classes: Mapping[int, Sequence[int]],
#         phase: int,
#         gt_shape: Sequence[int],
#         num_workers: int,
#         pin_memory: bool,
#     ) -> None:
#         self._patch_dataset = patch_dataset
#         self.phase_to_classes = {
#             int(key): [int(value) for value in values]
#             for key, values in phase_to_classes.items()
#         }
#         self.phase = int(phase)
#         self.gt_shape = tuple(int(value) for value in gt_shape)
#         self.num_workers = int(num_workers)
#         self.pin_memory = bool(pin_memory)
#         self.released = False

#     def get_classes_up_to_phase(self, phase: int) -> List[int]:
#         phase = int(phase)
#         result: List[int] = []
#         for index in range(phase + 1):
#             result.extend(self.phase_to_classes[index])
#         return result

#     def get_cumulative_dataloader(
#         self,
#         phase: Optional[int] = None,
#         *,
#         up_to_phase: Optional[int] = None,
#         split: str = "all",
#         batch_size: int = 128,
#         shuffle: bool = False,
#     ) -> DataLoader:
#         resolved = self.phase if phase is None else int(phase)
#         if up_to_phase is not None:
#             resolved = int(up_to_phase)
#         if resolved != self.phase:
#             raise RuntimeError(
#                 f"Transient map manager was built for phase {self.phase}, got {resolved}."
#             )
#         if str(split).strip().lower() not in {
#             "all", "all_labeled", "full", "labeled"
#         }:
#             raise RuntimeError(
#                 "Transient qualitative manager supports only split='all'."
#             )
#         if bool(shuffle):
#             raise RuntimeError("Qualitative visualization must not shuffle coordinates.")
#         if self.released or self._patch_dataset is None:
#             raise RuntimeError("Transient qualitative manager has been released.")
#         return DataLoader(
#             self._patch_dataset,
#             batch_size=int(batch_size),
#             shuffle=False,
#             drop_last=False,
#             pin_memory=self.pin_memory,
#             num_workers=self.num_workers,
#         )

#     def release(self) -> None:
#         if self.released:
#             return
#         dataset = self._patch_dataset
#         if dataset is not None:
#             for name in (
#                 "patches",
#                 "spectral_positive_centers",
#                 "spectral_negative_centers",
#                 "spectral_step_sizes",
#             ):
#                 value = getattr(dataset, name, None)
#                 if torch.is_tensor(value):
#                     value.zero_()
#         self._patch_dataset = None
#         self.released = True
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


# def _build_transient_qualitative_phase_manager(
#     *,
#     args: argparse.Namespace,
#     retained_data: Mapping[str, Any],
#     phase: int,
# ) -> _TransientQualitativePhaseManager:
#     """Reconstruct all labeled pixels from source for one committed phase map.

#     This routine is deliberately outside ``IncrementalHSIDataset`` and Trainer.
#     It does not restore train access, does not alter the model, and does not
#     contribute any metric used for checkpoint selection or phase admission.
#     """
#     phase = int(phase)
#     protocol = retained_data.get("protocol")
#     summary = retained_data.get("summary")
#     if not isinstance(protocol, Mapping) or not isinstance(summary, Mapping):
#         raise RuntimeError("Qualitative reconstruction metadata are incomplete.")
#     phase_to_classes = {
#         int(key): [int(value) for value in values]
#         for key, values in dict(protocol["phase_to_classes"]).items()
#     }
#     seen_classes: List[int] = []
#     for phase_index in range(phase + 1):
#         seen_classes.extend(phase_to_classes[phase_index])
#     seen_classes = list(dict.fromkeys(seen_classes))

#     preprocessing_path = str(summary.get("preprocessing_state_path", ""))
#     if not preprocessing_path or not os.path.isfile(preprocessing_path):
#         raise RuntimeError(
#             f"Saved preprocessing state is unavailable: {preprocessing_path!r}"
#         )

#     raw_hsi, gt, num_classes, _, has_background, label_policy = LoadRawHSIData(
#         method=args.dataset,
#         base_dir=args.data_dir,
#     )
#     if int(num_classes) != int(retained_data["num_classes"]):
#         raise RuntimeError("Reloaded qualitative scene has a different class count.")
#     raw_hsi = np.asarray(raw_hsi, dtype=np.float32)
#     gt = np.asarray(gt)
#     preprocessing_state = LoadHSIPreprocessor(preprocessing_path)
#     model_hsi, _ = ApplyHSIPreprocessor(raw_hsi, preprocessing_state)

#     patches, input_labels, coords = ImageCubes(
#         model_hsi,
#         gt,
#         WS=int(args.patch_size),
#         removeZeroLabels=True,
#         has_background=bool(has_background),
#         num_classes=int(num_classes),
#         pytorch_format=True,
#         label_policy=label_policy,
#         return_center_spectra=False,
#     )
#     patches = np.ascontiguousarray(to_numpy(patches), dtype=np.float32)
#     input_labels = np.ascontiguousarray(
#         to_numpy(input_labels, dtype=np.int64).reshape(-1)
#     )
#     coords = np.ascontiguousarray(to_numpy(coords, dtype=np.int64))

#     index_labels, index_coords = ExtractLabeledPixelIndex(
#         gt, label_policy=label_policy
#     )
#     if not np.array_equal(input_labels, np.asarray(index_labels, dtype=np.int64)):
#         raise RuntimeError("Reloaded qualitative labels changed sample order.")
#     if not np.array_equal(coords, np.asarray(index_coords, dtype=np.int64)):
#         raise RuntimeError("Reloaded qualitative coordinates changed sample order.")

#     remapped_labels = np.asarray(protocol["remapped_labels"], dtype=np.int64)
#     if remapped_labels.shape != input_labels.shape:
#         raise RuntimeError("Reloaded qualitative sample count differs from protocol.")
#     seen_mask = np.isin(remapped_labels, np.asarray(seen_classes, dtype=np.int64))
#     sample_indices = np.flatnonzero(seen_mask).astype(np.int64)
#     if sample_indices.size == 0:
#         raise RuntimeError(f"Phase {phase} qualitative scene contains no seen pixels.")

#     selected_coords = coords[sample_indices]
#     positive, negative, steps, intervention_metadata = (
#         BuildPCSTGBInterventionCentersFromCube(
#             raw_hsi,
#             selected_coords,
#             preprocessing_state,
#             gain_step=float(args.spectral_gain_step),
#             tilt_step=float(args.spectral_tilt_step),
#             chunk_size=int(args.preprocess_response_chunk_size),
#             definition_version=int(args.intervention_definition_version),
#         )
#     )
#     if int(intervention_metadata.get("num_interventions", -1)) != int(
#         args.num_spectral_interventions
#     ):
#         raise RuntimeError("Qualitative intervention-count mismatch.")

#     patch_dataset = HSIPatchDataset(
#         patches[sample_indices],
#         remapped_labels[sample_indices],
#         coords=selected_coords,
#         sample_indices=sample_indices,
#         spectral_positive_centers=positive,
#         spectral_negative_centers=negative,
#         spectral_step_sizes=steps,
#         intervention_definition_version=int(args.intervention_definition_version),
#     )

#     # The patch dataset owns copied torch tensors.  Remove every reconstruction
#     # array before returning so only the short-lived evaluation dataset survives.
#     del (
#         patches,
#         input_labels,
#         coords,
#         index_labels,
#         index_coords,
#         remapped_labels,
#         selected_coords,
#         positive,
#         negative,
#         steps,
#         model_hsi,
#         raw_hsi,
#         gt,
#     )
#     gc.collect()

#     return _TransientQualitativePhaseManager(
#         patch_dataset=patch_dataset,
#         phase_to_classes=phase_to_classes,
#         phase=phase,
#         gt_shape=preprocessing_state["spatial_shape"],
#         num_workers=int(args.num_workers),
#         pin_memory=str(args.device).startswith("cuda"),
#     )


# def _save_transient_qualitative_phase_map(
#     *,
#     args: argparse.Namespace,
#     model: NECILModel,
#     dataset: IncrementalHSIDataset,
#     retained_data: Mapping[str, Any],
#     phase: int,
#     figures_dir: str,
# ) -> Dict[str, Any]:
#     if not bool(args.qualitative_map_reload_from_source):
#         raise RuntimeError(
#             "All-phase dense maps require qualitative_map_reload_from_source=true."
#         )
#     manager = _build_transient_qualitative_phase_manager(
#         args=args,
#         retained_data=retained_data,
#         phase=int(phase),
#     )
#     try:
#         outputs = predict_phase_grid(
#             model=model,
#             dataset_manager=manager,
#             phase=int(phase),
#             target_names=dataset.target_names,
#             save_dir=figures_dir,
#             device=str(args.device),
#             patch_size=int(args.patch_size),
#             chunk_size=int(args.visualization_batch_size),
#             split="all",
#             classifier_mode=VISUALIZATION_MODE_ALIAS,
#             semantic_mode="identity",
#             save_numpy=bool(args.save_raw_arrays),
#             return_outputs=True,
#             class_cmap=str(args.viz_class_cmap),
#             background_color=str(args.viz_background_color),
#             save_combined_figure=False,
#             save_qualitative_all_labeled=False,
#             dpi=int(args.visualization_dpi),
#         )
#         outputs["source_reloaded_after_commit"] = True
#         outputs["used_for_training_or_admission"] = False
#         outputs["official_test_evidence"] = False
#         return outputs
#     finally:
#         manager.release()


# # =============================================================================
# # Visualization helpers
# # =============================================================================


# def _safe_visualization(
#     artifacts: Dict[str, Any],
#     name: str,
#     function: Any,
#     *args: Any,
#     **kwargs: Any,
# ) -> None:
#     try:
#         artifacts[name] = function(*args, **kwargs)
#     except Exception as error:
#         artifacts[name] = {"status": "SKIPPED", "reason": str(error)}
#         print(f"[Visualization skipped: {name}] {error}")


# def save_phase_visualizations(
#     *,
#     args: argparse.Namespace,
#     model: NECILModel,
#     dataset: IncrementalHSIDataset,
#     retained_data: Mapping[str, Any],
#     phase: int,
#     history: Mapping[str, Any],
#     metrics: Mapping[str, Any],
#     run_dir: str,
# ) -> Dict[str, Any]:
#     artifacts: Dict[str, Any] = {}
#     if not bool(args.save_visualizations):
#         return artifacts
#     figures_dir = os.path.join(run_dir, f"phase_{phase}", "figures")
#     os.makedirs(figures_dir, exist_ok=True)

#     if phase == 0:
#         _safe_visualization(
#             artifacts,
#             "training",
#             plot_base_training_dynamics,
#             history,
#             os.path.join(figures_dir, "base_training_dashboard.png"),
#             dpi=int(args.visualization_dpi),
#             save_separate=True,
#         )

#     _safe_visualization(
#         artifacts,
#         "classification",
#         save_classification_diagnostics,
#         confusion_matrix=to_numpy(metrics["confusion_matrix"], dtype=np.int64),
#         class_ids=metrics["seen_classes"],
#         target_names=dataset.target_names,
#         save_path=os.path.join(figures_dir, "classification_diagnostics.png"),
#         title=f"Phase {phase} PC-STGB cumulative test classification",
#         dpi=int(args.visualization_dpi),
#     )

#     if not bool(args.skip_phase_maps):
#         _safe_visualization(
#             artifacts,
#             "maps",
#             predict_phase_grid,
#             model=model,
#             dataset_manager=dataset,
#             phase=int(phase),
#             target_names=dataset.target_names,
#             save_dir=figures_dir,
#             device=str(args.device),
#             patch_size=int(args.patch_size),
#             chunk_size=int(args.visualization_batch_size),
#             split="test",
#             classifier_mode=VISUALIZATION_MODE_ALIAS,
#             semantic_mode="identity",
#             save_numpy=bool(args.save_raw_arrays),
#             return_outputs=True,
#             class_cmap=str(args.viz_class_cmap),
#             background_color=str(args.viz_background_color),
#             save_combined_figure=False,
#             save_qualitative_all_labeled=False,
#             dpi=int(args.visualization_dpi),
#         )

#         if bool(args.save_qualitative_all_labeled_map):
#             _safe_visualization(
#                 artifacts,
#                 "qualitative_all_labeled",
#                 _save_transient_qualitative_phase_map,
#                 args=args,
#                 model=model,
#                 dataset=dataset,
#                 retained_data=retained_data,
#                 phase=int(phase),
#                 figures_dir=figures_dir,
#             )
#     return artifacts


# # =============================================================================
# # Run orchestration
# # =============================================================================


# def _phase_limit(args: argparse.Namespace, dataset: IncrementalHSIDataset) -> int:
#     final = int(dataset.num_phases) - 1
#     if bool(args.base_only):
#         return 0
#     if int(args.max_phase) >= 0:
#         final = min(final, int(args.max_phase))
#     return final


# def _restore_checkpoint_dataset_purge(
#     trainer: Trainer,
#     dataset: IncrementalHSIDataset,
#     checkpoint_path: str,
# ) -> int:
#     checkpoint = trainer.load_checkpoint(
#         checkpoint_path,
#         strict=True,
#         verify_checksum=True,
#         restore_rng=True,
#     )
#     state = checkpoint.get("dataset_runtime_state")
#     if not isinstance(state, Mapping):
#         raise RuntimeError(
#             "Resume requires dataset_runtime_state so finalized old training payloads "
#             "can be physically purged in the reconstructed process."
#         )
#     # Trainer restoration intentionally avoids a second purge during internal
#     # loading. Main is the data owner, so it immediately reapplies the state with
#     # physical purge enabled before any next-phase loader is created.
#     dataset.restore_runtime_state(state, purge_finalized_train_payload=True)
#     dataset.assert_non_exemplar()
#     return int(checkpoint["phase"])


# def _save_overlap_review(
#     *,
#     model: NECILModel,
#     dataset: IncrementalHSIDataset,
#     metrics: Mapping[str, Any],
#     reports_dir: str,
# ) -> Dict[str, Any]:
#     try:
#         review = build_geometry_overlap_review(
#             model=model,
#             class_ids=metrics["seen_classes"],
#             target_names=dataset.target_names,
#             directional_invasion_matrix=metrics["directional_invasion_matrix"],
#         )
#     except Exception as error:
#         review = {
#             "status": "UNAVAILABLE",
#             "reason": str(error),
#             "definition": (
#                 "Operational overlap should be interpreted from the deployed "
#                 "conditional joint-energy invasion matrix."
#             ),
#         }
#     save_json(os.path.join(reports_dir, "geometry_overlap_review.json"), review)
#     rows = list(review.get("pair_geometry", []))
#     if rows:
#         csv_path = os.path.join(reports_dir, "geometry_overlap_pairs.csv")
#         with open(csv_path, "w", newline="", encoding="utf-8") as stream:
#             writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
#             writer.writeheader()
#             writer.writerows(rows)
#     return review


# def run_single_experiment(
#     base_args: argparse.Namespace,
#     *,
#     run_index: int,
#     seed: int,
# ) -> Dict[str, Any]:
#     requested = argparse.Namespace(**namespace_to_dict(base_args))
#     requested.seed = int(seed)
#     args = resolve_config(requested)
#     validate_config(args)
#     set_seed(int(args.seed), bool(args.deterministic))

#     run_dir = os.path.join(
#         os.path.abspath(str(args.save_dir)),
#         str(args.dataset),
#         f"base_{int(args.base_classes)}_inc_{int(args.increment)}",
#         f"run_{run_index + 1}_seed_{int(args.seed)}",
#     )
#     os.makedirs(run_dir, exist_ok=True)
#     args.save_dir = run_dir
#     save_json(
#         os.path.join(run_dir, "requested_configuration.json"),
#         namespace_to_dict(requested),
#     )
#     save_json(
#         os.path.join(run_dir, "resolved_configuration.json"),
#         namespace_to_dict(args),
#     )

#     print("\n" + "=" * 112)
#     print("PC-STGB STRICT NON-EXEMPLAR HSI CLASS-INCREMENTAL EXPERIMENT")
#     print("=" * 112)
#     print(
#         f"[Run] dataset={args.dataset} | seed={args.seed} | device={args.device} | "
#         f"base={args.base_classes} | increment={args.increment}"
#     )
#     print(
#         f"[Geometry] occupancy_rank={args.subspace_rank} | tangent_rank={args.spectral_response_rank} | "
#         f"K={args.num_spectral_interventions} | beta_T={args.spectral_response_weight:.3f}"
#     )
#     print(f"[Factorization] {JOINT_FACTORIZATION}")

#     load_start = time.time()
#     data = load_hsi_data(args)
#     load_time = time.time() - load_start
#     save_json(os.path.join(run_dir, "dataset_summary.json"), data["summary"])
#     dataset = build_dataset(args, data)
#     protocol = data["protocol"]
#     save_json(
#         os.path.join(run_dir, "class_protocol.json"),
#         {
#             "phase_to_classes": protocol["phase_to_classes"],
#             "class_order_input_ids": protocol["class_order"],
#             "num_phases": int(protocol["num_phases"]),
#             "execution": "base_plus_incremental" if not args.base_only else "base_only",
#         },
#     )
#     retained_data = release_external_sample_payload(data)
#     del protocol

#     model = build_model(args)
#     if bool(args.save_model_summary):
#         save_model_summary(os.path.join(run_dir, "model_summary_initial.txt"), model)
#     trainer = Trainer(model, dataset, args)

#     start_phase = 0
#     resume = str(args.resume_checkpoint).strip()
#     if resume:
#         restored_phase = _restore_checkpoint_dataset_purge(trainer, dataset, resume)
#         start_phase = restored_phase + 1
#         print(f"[Resume] restored phase {restored_phase}; next phase={start_phase}")

#     final_phase = _phase_limit(args, dataset)
#     if start_phase > final_phase:
#         raise RuntimeError(
#             f"Nothing to run: start_phase={start_phase}, final_phase={final_phase}"
#         )

#     phase_results: List[Dict[str, Any]] = []

#     for phase in range(start_phase, final_phase + 1):
#         phase_start = time.time()
#         epochs = int(args.epochs_base if phase == 0 else args.epochs_inc)
#         learning_rate = float(args.lr if phase == 0 else args.lr_inc)
#         history = trainer.train_phase(
#             phase=phase,
#             epochs=epochs,
#             batch_size=int(args.batch_size),
#             lr=learning_rate,
#         )
#         phase_training_time = time.time() - phase_start

#         status = str(history.get("status", "COMMITTED" if phase > 0 else "COMPLETE"))
#         protocol_stop = bool(history.get("protocol_stop", False))
#         committed = bool(history.get("committed", phase == 0))
#         if phase > 0 and (status == "REJECTED" or protocol_stop or not committed):
#             phase_result = {
#                 "phase": phase,
#                 "status": "REJECTED",
#                 "protocol_stop": True,
#                 "training_time_sec": phase_training_time,
#                 "history": history,
#                 "checkpoint": None,
#             }
#             phase_results.append(phase_result)
#             save_json(
#                 os.path.join(run_dir, f"phase_{phase}", "reports", "phase_result.json"),
#                 phase_result,
#             )
#             print(f"[Protocol stop] phase {phase} candidate was not committed")
#             break

#         seen_classes = _seen_classes(dataset, phase)
#         dataset.assert_non_exemplar()
#         eval_start = time.time()
#         metrics, y_true, y_pred, energy = evaluate_phase_geometry(
#             model,
#             dataset,
#             phase,
#             batch_size=int(args.eval_batch_size),
#             device=torch.device(args.device),
#             split="test",
#         )
#         evaluation_time = time.time() - eval_start
#         replay = replay_health(
#             model,
#             seen_classes,
#             samples_per_class=int(args.base_diagnostic_replay_samples_per_class),
#         )
#         diagnostics = model.geometry_diagnostics(seen_classes)
#         phase_certificate = trainer.geometry_phase_certificate(
#             phase,
#             seen_classes,
#             require_statistics=True,
#             require_response=True,
#             require_frozen=True,
#         )

#         reports_dir = os.path.join(run_dir, f"phase_{phase}", "reports")
#         os.makedirs(reports_dir, exist_ok=True)
#         save_json(os.path.join(reports_dir, "training_history.json"), history)
#         save_json(os.path.join(reports_dir, "test_metrics.json"), metrics)
#         save_json(os.path.join(reports_dir, "replay_health.json"), replay)
#         save_json(os.path.join(reports_dir, "geometry_bank_diagnostics.json"), diagnostics)
#         save_json(os.path.join(reports_dir, "phase_geometry_certificate.json"), phase_certificate)
#         overlap_review = _save_overlap_review(
#             model=model,
#             dataset=dataset,
#             metrics=metrics,
#             reports_dir=reports_dir,
#         )
#         report_text = write_phase_report(
#             os.path.join(reports_dir, "phase_geometry_report.txt"),
#             phase=phase,
#             metrics=metrics,
#             diagnostics=diagnostics,
#             replay=replay,
#             training=history,
#         )

#         visualization_artifacts = save_phase_visualizations(
#             args=args,
#             model=model,
#             dataset=dataset,
#             retained_data=retained_data,
#             phase=phase,
#             history=history,
#             metrics=metrics,
#             run_dir=run_dir,
#         )
#         if bool(args.save_visualizations):
#             figures_dir = os.path.join(run_dir, f"phase_{phase}", "figures")
#             _safe_visualization(
#                 visualization_artifacts,
#                 "geometry_overlap",
#                 save_geometry_pair_diagnostics,
#                 review=overlap_review,
#                 class_ids=seen_classes,
#                 target_names=dataset.target_names,
#                 save_path=os.path.join(figures_dir, "geometry_overlap_diagnostics.png"),
#                 dpi=int(args.visualization_dpi),
#             )

#         if bool(args.save_raw_arrays):
#             raw_dir = os.path.join(run_dir, f"phase_{phase}", "raw_arrays")
#             os.makedirs(raw_dir, exist_ok=True)
#             np.save(os.path.join(raw_dir, "test_true_labels.npy"), y_true)
#             np.save(os.path.join(raw_dir, "test_predicted_labels.npy"), y_pred)
#             torch.save(energy, os.path.join(raw_dir, "test_joint_energy.pt"))
#             np.save(
#                 os.path.join(raw_dir, "test_confusion_matrix.npy"),
#                 to_numpy(metrics["confusion_matrix"], dtype=np.int64),
#             )

#         checkpoint = str(
#             history.get(
#                 "checkpoint_path",
#                 os.path.join(run_dir, f"phase_{phase}", "checkpoint.pth"),
#             )
#         )
#         if not os.path.isfile(checkpoint):
#             raise RuntimeError(f"Phase {phase} checkpoint was not created: {checkpoint}")

#         phase_result = {
#             "phase": phase,
#             "status": status,
#             "protocol_stop": False,
#             "committed": True,
#             "seen_classes": seen_classes,
#             "training_time_sec": phase_training_time,
#             "evaluation_time_sec": evaluation_time,
#             "test_metrics": metrics,
#             "replay_health": replay,
#             "geometry_bank_diagnostics": diagnostics,
#             "phase_geometry_certificate": phase_certificate,
#             "geometry_overlap_review": overlap_review,
#             "checkpoint": checkpoint,
#             "report_text": report_text,
#             "visualization_artifacts": visualization_artifacts,
#         }
#         phase_results.append(phase_result)
#         save_json(
#             os.path.join(reports_dir, "phase_result.json"),
#             phase_result,
#         )

#         print(
#             f"[Phase {phase} Test] OA={metrics['overall_accuracy']:.2f}% | "
#             f"AA={metrics['average_accuracy']:.2f}% | "
#             f"Old={metrics['old_accuracy']:.2f}% | New={metrics['new_accuracy']:.2f}% | "
#             f"H={metrics['old_new_harmonic_mean']:.2f}% | "
#             f"Q05={metrics['q05_energy_gap']:.4f}"
#         )
#         print(
#             f"               Old->New={100.0 * metrics['old_to_new_invasion_rate']:.2f}% | "
#             f"New->Old={100.0 * metrics['new_to_old_invasion_rate']:.2f}% | "
#             f"TangentHelp={metrics['conditional_tangent_help_rate']:.4f} | "
#             f"TangentHarm={metrics['conditional_tangent_harm_rate']:.4f}"
#         )

#     completed = [item for item in phase_results if "test_metrics" in item]
#     if not completed:
#         raise RuntimeError("The run completed no committed/evaluated phase")
#     continual = build_continual_summary(phase_results)
#     save_json(os.path.join(run_dir, "continual_summary.json"), continual)
#     if bool(args.save_model_summary):
#         save_model_summary(os.path.join(run_dir, "model_summary_final.txt"), model)

#     payload = {
#         "stack_build_id": STACK_BUILD_ID,
#         "method": METHOD_NAME,
#         "schema_version": BANK_SCHEMA_VERSION,
#         "joint_factorization": JOINT_FACTORIZATION,
#         "run_index": int(run_index),
#         "seed": int(args.seed),
#         "run_dir": run_dir,
#         "data_loading_time_sec": float(load_time),
#         "phase_results": phase_results,
#         "continual_summary": continual,
#         "final_checkpoint": completed[-1]["checkpoint"],
#         "strict_non_exemplar_verified": bool(dataset.assert_non_exemplar()),
#         "external_sample_payload_released": set(retained_data) == {
#             "protocol",
#             "summary",
#             "target_names",
#             "raw_target_names",
#             "label_policy",
#             "num_classes",
#         },
#     }
#     save_json(os.path.join(run_dir, "final_run_summary.json"), payload)
#     print(
#         f"[Run complete] final_phase={continual['final_phase']} | "
#         f"FinalOA={continual['final_overall_accuracy']:.2f}% | "
#         f"Forgetting={continual['average_forgetting']:.2f} | "
#         f"BWT={continual['average_backward_transfer']:.2f}"
#     )
#     return payload


# def aggregate_runs(results: Sequence[Mapping[str, Any]], root: str) -> Dict[str, Any]:
#     if not results:
#         raise ValueError("results is empty")
#     rows: List[Dict[str, Any]] = []
#     for result in results:
#         summary = result["continual_summary"]
#         rows.append(
#             {
#                 "run_index": int(result["run_index"]),
#                 "seed": int(result["seed"]),
#                 "final_phase": int(summary["final_phase"]),
#                 "final_overall_accuracy": float(summary["final_overall_accuracy"]),
#                 "final_average_accuracy": float(summary["final_average_accuracy"]),
#                 "final_minimum_per_class_accuracy": float(
#                     summary["final_minimum_per_class_accuracy"]
#                 ),
#                 "average_forgetting": float(summary["average_forgetting"]),
#                 "average_backward_transfer": float(
#                     summary["average_backward_transfer"]
#                 ),
#                 "protocol_stopped": bool(summary["protocol_stopped"]),
#                 "run_dir": str(result["run_dir"]),
#             }
#         )

#     def mean_std(key: str) -> Tuple[float, float]:
#         values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
#         return float(values.mean()), float(values.std(ddof=0))

#     summary = {
#         "stack_build_id": STACK_BUILD_ID,
#         "method": METHOD_NAME,
#         "joint_factorization": JOINT_FACTORIZATION,
#         "num_runs": len(rows),
#         "runs": rows,
#         "final_overall_accuracy_mean_std": mean_std("final_overall_accuracy"),
#         "final_average_accuracy_mean_std": mean_std("final_average_accuracy"),
#         "final_minimum_class_accuracy_mean_std": mean_std(
#             "final_minimum_per_class_accuracy"
#         ),
#         "average_forgetting_mean_std": mean_std("average_forgetting"),
#         "average_backward_transfer_mean_std": mean_std(
#             "average_backward_transfer"
#         ),
#     }
#     os.makedirs(root, exist_ok=True)
#     save_json(os.path.join(root, "multi_run_summary.json"), summary)
#     csv_path = os.path.join(root, "multi_run_summary.csv")
#     with open(csv_path, "w", newline="", encoding="utf-8") as stream:
#         writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)
#     summary["csv_path"] = csv_path
#     return summary


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     parsed = parse_args(argv)
#     preview = resolve_config(parsed)
#     validate_config(preview)
#     seeds = parse_seed_list(preview.seed_list)
#     if seeds is None:
#         seeds = [int(preview.seed) + index for index in range(int(preview.num_runs))]
#     results = [
#         run_single_experiment(parsed, run_index=index, seed=seed)
#         for index, seed in enumerate(seeds)
#     ]
#     root = os.path.join(
#         os.path.abspath(str(preview.save_dir)),
#         str(preview.dataset),
#         f"base_{int(preview.base_classes)}_inc_{int(preview.increment)}",
#     )
#     summary = aggregate_runs(results, root)
#     final_mean, final_std = summary["final_overall_accuracy_mean_std"]
#     aa_mean, aa_std = summary["final_average_accuracy_mean_std"]
#     forgetting_mean, forgetting_std = summary["average_forgetting_mean_std"]
#     print("\n" + "=" * 104)
#     print("PC-STGB STRICT NECIL MULTI-RUN SUMMARY")
#     print("=" * 104)
#     print(
#         f"runs={summary['num_runs']} | FinalOA={final_mean:.2f}±{final_std:.2f} | "
#         f"FinalAA={aa_mean:.2f}±{aa_std:.2f} | "
#         f"Forgetting={forgetting_mean:.2f}±{forgetting_std:.2f}"
#     )
#     print(f"[Saved] {os.path.join(root, 'multi_run_summary.json')}")


# if __name__ == "__main__":
#     main()


