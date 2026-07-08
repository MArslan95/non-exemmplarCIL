from __future__ import annotations
import argparse
import copy
import csv
import inspect
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.hsi_dataloader_pytorch import ImageCubes, LoadHSIData
from data.incremental_dataset import IncrementalHSIDataset
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from utils.eval import (
    NECILEvaluator,
    calculate_metrics_torch,
    make_json_serializable,
    predictions_from_seen_local_scores,
    save_classification_report,
    validate_seen_local_outputs,
)
from utils.visualize import plot_training_history, predict_phase_grid


STACK_BUILD_ID = "SCTGR-ENERGY-CONTRACT-2026-07-07-R2"

DATASET_INFO = {
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


# -----------------------------------------------------------------------------
# Robust utilities
# -----------------------------------------------------------------------------


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t", "on"}:
        return True
    if s in {"false", "0", "no", "n", "f", "off", "none", "null", ""}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def parse_seed_list(seed_list_str: Optional[str]) -> Optional[List[int]]:
    if seed_list_str is None or str(seed_list_str).strip() == "":
        return None
    return [int(s.strip()) for s in str(seed_list_str).split(",") if s.strip()]


def _json_safe(obj: Any) -> Any:
    try:
        return make_json_serializable(obj)
    except Exception:
        pass
    if torch.is_tensor(obj):
        x = obj.detach().cpu()
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(path: str, data: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, sort_keys=True)
    return path


def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return copy.deepcopy(vars(args))


def _set_resolved(args: argparse.Namespace, name: str, value: Any, reasons: Dict[str, str], reason: str) -> None:
    old = getattr(args, name, None)
    if old != value or not hasattr(args, name):
        setattr(args, name, value)
        reasons[name] = reason


def compute_config_diff(
    original: Dict[str, Any],
    resolved: Dict[str, Any],
    reasons: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    reasons = reasons or {}
    rows: List[Dict[str, Any]] = []
    for k in sorted(set(original.keys()) | set(resolved.keys())):
        if original.get(k) != resolved.get(k):
            rows.append({
                "name": k,
                "original": _json_safe(original.get(k)),
                "resolved": _json_safe(resolved.get(k)),
                "reason": reasons.get(k, "resolved_config"),
            })
    return rows


def normalize_classifier_mode(mode: Optional[str]) -> str:
    """Normalize every public geometry alias to the strict classifier token.

    The repaired classifier/model/trainer use ``geometry_only`` as the explicit
    method-contract string. ``geometry`` is accepted from older commands, but it
    is semantically identical and must be normalized before the trainer identity
    check. Otherwise main.py incorrectly reports a method-identity mutation even
    though the classifier path did not change.
    """
    m = str(mode or "geometry_only").lower().strip()
    aliases = {
        "": "geometry_only",
        "none": "geometry_only",
        "geo": "geometry_only",
        "geometry": "geometry_only",
        "geometry_only": "geometry_only",
        "geometry-only": "geometry_only",
        "feature_geometry": "geometry_only",
        "low_rank_geometry": "geometry_only",
        "srgp": "geometry_only",
        "srgp_geometry": "geometry_only",
        "spectral_geometry": "geometry_only",
        "spectral_residual_geometry": "geometry_only",
        "calibrated": "geometry_only",
        "calibrated_geometry": "geometry_only",
    }
    m = aliases.get(m, m)
    if m != "geometry_only":
        raise ValueError(f"Unsupported classifier mode {mode!r}. Use geometry_only/geometry.")
    return m


def normalize_incremental_update_mode(mode: Optional[str]) -> str:
    """Normalize the only valid incremental architecture token."""
    raw = str(mode or "spectral_coupled_geometry_replay").lower().strip()
    aliases = {
        "": "spectral_coupled_geometry_replay",
        "none": "spectral_coupled_geometry_replay",
        "main": "spectral_coupled_geometry_replay",
        "clean": "spectral_coupled_geometry_replay",
        "descriptor": "spectral_coupled_geometry_replay",
        "descriptor_only": "spectral_coupled_geometry_replay",
        "descriptor_refinement": "spectral_coupled_geometry_replay",
        "scbgr": "spectral_coupled_geometry_replay",
        "sctgr": "spectral_coupled_geometry_replay",
        "spectral_coupled": "spectral_coupled_geometry_replay",
        "spectral_coupled_replay": "spectral_coupled_geometry_replay",
        "spectral_coupled_geometry_replay": "spectral_coupled_geometry_replay",
    }
    forbidden = {
        "adapter", "gated_adapter", "geometry_adapter", "geometry_gated_adapter",
        "g2rpa", "g2-rpa", "transport", "geometry_transport",
    }
    if raw in forbidden:
        raise ValueError(
            f"incremental_update_mode={raw!r} selects a removed architecture. "
            "Use spectral_coupled_geometry_replay."
        )
    out = aliases.get(raw, raw)
    if out != "spectral_coupled_geometry_replay":
        raise ValueError(
            f"Unsupported --incremental_update_mode={mode!r}. "
            "Use spectral_coupled_geometry_replay."
        )
    return out

# -----------------------------------------------------------------------------
# Parser and resolved configuration
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spectral-Coupled Tangent Geometry Replay with new-row descriptor adaptation for strict non-exemplar HSI class-incremental learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    main = parser.add_argument_group("A. Core experiment")
    main.add_argument("--dataset", type=str, default="IP", choices=DATASET_INFO.keys())
    main.add_argument("--data_dir", type=str, default="./datasets")
    main.add_argument("--save_dir", type=str, default="./results_sctgr")
    main.add_argument("--patch_size", type=int, default=11)
    main.add_argument("--train_ratio", type=float, default=0.2)
    main.add_argument("--val_ratio", type=float, default=0.1)
    main.add_argument("--min_train_per_class", type=int, default=20)
    main.add_argument("--no_pca", action="store_true")
    main.add_argument("--pca_components", type=int, default=30)
    main.add_argument("--reduction_method", type=str, default="PCA")
    main.add_argument("--base_classes", type=int, default=None)
    main.add_argument("--increment", type=int, default=None)
    main.add_argument("--epochs_base", type=int, default=80)
    main.add_argument("--epochs_inc", type=int, default=30)
    main.add_argument("--batch_size", type=int, default=64)
    main.add_argument("--lr", type=float, default=1e-4)
    main.add_argument("--lr_inc", type=float, default=1e-4)
    main.add_argument("--weight_decay", type=float, default=1e-4)
    main.add_argument("--seed", type=int, default=42)
    main.add_argument("--num_runs", type=int, default=1)
    main.add_argument("--seed_list", type=str, default="")
    main.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    main.add_argument("--num_workers", type=int, default=0)
    main.add_argument("--base_only", type=str2bool, default=False)
    main.add_argument("--max_train_phase", type=int, default=-1)
    main.add_argument("--max_phases", type=int, default=0)

    model = parser.add_argument_group("B. Backbone and GeometryBank")
    model.add_argument("--d_model", type=int, default=128)
    model.add_argument("--d_state", type=int, default=16)
    model.add_argument("--d_conv", type=int, default=4)
    model.add_argument("--expand", type=int, default=2)
    model.add_argument("--num_spectral_layers", type=int, default=3)
    model.add_argument("--num_layers", type=int, default=3)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--projection_dropout", type=float, default=0.1)
    model.add_argument("--backbone_norm", type=str, default="layer", choices=["layer", "rms"])
    model.add_argument("--stem_norm_groups", type=int, default=8)
    model.add_argument("--ssm_residual_scale_init", type=float, default=0.7)
    model.add_argument("--fusion_residual_scale", type=float, default=0.3)
    model.add_argument("--backbone_output_dropout", type=float, default=0.0)
    model.add_argument("--subspace_rank", type=int, default=5)
    model.add_argument("--geom_var_floor", type=float, default=5e-4)
    model.add_argument("--geometry_variance_shrinkage", type=float, default=0.25)
    model.add_argument("--geometry_max_variance_ratio", type=float, default=50.0)
    model.add_argument("--geometry_min_reliability", type=float, default=0.05)
    model.add_argument("--rank_energy_threshold", type=float, default=0.90)
    model.add_argument("--rank_eigen_ratio_threshold", type=float, default=1e-2)
    model.add_argument("--min_active_rank", type=int, default=1)
    model.add_argument("--normalize_geometry_features", type=str2bool, default=True)
    model.add_argument("--geometry_feature_scale", type=float, default=0.0)
    model.add_argument("--geometry_feature_clamp", type=float, default=0.0)
    model.add_argument("--subspace_extract_batch_size", type=int, default=256)
    model.add_argument("--spectral_geometry_rank", type=int, default=5)
    model.add_argument("--spectral_rank_energy_threshold", type=float, default=0.95)
    model.add_argument("--spectral_rank_eigen_ratio_threshold", type=float, default=1e-3)
    model.add_argument("--spectral_variance_floor", type=float, default=1e-6)
    model.add_argument("--coupling_ridge", type=float, default=1e-3)
    model.add_argument("--coupling_min_reliability", type=float, default=0.20)
    model.add_argument("--spectral_tangent_clip", type=float, default=2.5)
    model.add_argument("--replay_candidate_multiplier", type=int, default=4)

    base = parser.add_argument_group("C. Mandatory base objective")
    base.add_argument("--base_ce_weight", type=float, default=1.0)
    base.add_argument("--base_srpgr_weight", type=float, default=1.0)
    base.add_argument("--base_gics_weight", type=float, default=0.20)
    base.add_argument("--base_gics_temperature", type=float, default=0.07)
    base.add_argument("--base_energy_margin_weight", type=float, default=0.15)
    base.add_argument("--base_energy_margin", type=float, default=0.25)
    base.add_argument("--base_class_balance", type=str2bool, default=True)
    base.add_argument("--base_gics_key_noise_std", type=float, default=0.0)
    base.add_argument("--base_gics_key_scale_jitter", type=float, default=0.0)
    base.add_argument("--base_gics_key_band_drop", type=float, default=0.0)
    base.add_argument("--base_gics_key_spatial_drop", type=float, default=0.0)
    base.add_argument("--pgr_weight", type=float, default=0.10)
    base.add_argument("--pgr_compact_weight", type=float, default=0.15)
    base.add_argument("--pgr_center_weight", type=float, default=0.25)
    base.add_argument("--pgr_subspace_weight", type=float, default=0.15)
    base.add_argument("--pgr_band_weight", type=float, default=0.05)
    base.add_argument("--pgr_volume_weight", type=float, default=0.05)
    base.add_argument("--pgr_center_margin", type=float, default=1.10)
    base.add_argument("--pgr_band_overlap_max", type=float, default=0.65)
    base.add_argument("--pgr_max_subspace_overlap", type=float, default=0.55)
    base.add_argument("--pgr_min_class_variance", type=float, default=0.015)
    base.add_argument("--pgr_max_class_variance", type=float, default=0.75)
    base.add_argument("--pgr_min_class_samples", type=int, default=3)
    base.add_argument("--pgr_subspace_min_samples", type=int, default=6)
    base.add_argument("--pgr_subspace_rank", type=int, default=3)
    base.add_argument("--base_spectral_shape_weight", type=float, default=0.05)
    base.add_argument("--base_max_spectral_shape_similarity", type=float, default=0.75)
    base.add_argument("--base_spectral_shape_risk_weight", type=float, default=1.0)
    base.add_argument("--base_require_physical_spectral_shape", type=str2bool, default=True)
    base.add_argument("--strict_base_component_coverage", type=str2bool, default=True)

    inc = parser.add_argument_group("D. Spectral-coupled replay and new-row descriptor adaptation")
    inc.add_argument("--incremental_update_mode", type=str, default="spectral_coupled_geometry_replay")
    inc.add_argument("--use_spectral_coupled_replay", type=str2bool, default=True)
    inc.add_argument("--gfa_weight", type=float, default=1.0)
    inc.add_argument("--gfa_samples_per_class", type=int, default=48)
    inc.add_argument("--gfa_reliability_gated", type=str2bool, default=True)
    inc.add_argument("--gfa_parallel_scale", type=float, default=0.95)
    inc.add_argument("--gfa_residual_scale", type=float, default=0.25)
    inc.add_argument("--replay_min_per_class", type=int, default=24)
    inc.add_argument("--replay_max_per_class", type=int, default=64)
    inc.add_argument("--core_replay_ratio", type=float, default=0.85)
    inc.add_argument("--directed_replay_min_ratio", type=float, default=0.10)
    inc.add_argument("--directed_replay_max_ratio", type=float, default=0.40)
    inc.add_argument("--pair_risk_topk", type=int, default=3)
    inc.add_argument("--pair_risk_temperature", type=float, default=0.75)
    inc.add_argument("--replay_energy_filter", type=str2bool, default=True)
    inc.add_argument("--replay_energy_filter_multiplier", type=int, default=3)
    inc.add_argument("--replay_resample_rounds", type=int, default=4)
    inc.add_argument("--replay_core_accept_margin", type=float, default=0.0)
    inc.add_argument("--replay_directed_max_margin", type=float, default=1.0e9)
    inc.add_argument("--replay_risk_weight", type=float, default=0.75)
    inc.add_argument("--replay_unreliability_weight", type=float, default=0.50)
    inc.add_argument("--joint_old_new_ce_weight", type=float, default=1.0)
    inc.add_argument("--geometry_energy_margin_weight", type=float, default=0.30)
    inc.add_argument("--geometry_energy_margin", type=float, default=0.30)
    inc.add_argument("--old_new_invasion_weight", type=float, default=0.50)
    inc.add_argument("--old_new_geometry_margin", type=float, default=0.35)
    inc.add_argument("--refine_new_descriptors", type=str2bool, default=True)
    inc.add_argument("--use_descriptor_refinement", type=str2bool, default=True)
    inc.add_argument("--descriptor_refine_steps", type=int, default=20)
    inc.add_argument("--descriptor_refine_steps_per_epoch", type=int, default=None)
    inc.add_argument("--descriptor_refine_lr", type=float, default=1e-3)
    inc.add_argument("--descriptor_refine_grad_clip", type=float, default=1.0)
    inc.add_argument("--descriptor_trust_weight", type=float, default=0.80)
    inc.add_argument("--descriptor_refine_max_mean_shift", type=float, default=0.30)
    inc.add_argument("--descriptor_refine_max_logvar_shift", type=float, default=0.50)
    inc.add_argument("--descriptor_subspace_collision_weight", type=float, default=0.10)
    inc.add_argument("--descriptor_subspace_overlap_max", type=float, default=0.35)
    inc.add_argument("--descriptor_center_margin_weight", type=float, default=0.05)
    inc.add_argument("--descriptor_center_collision_weight", type=float, default=0.05)
    inc.add_argument("--descriptor_center_margin", type=float, default=0.50)
    inc.add_argument("--descriptor_volume_weight", type=float, default=0.03)
    inc.add_argument("--descriptor_volume_control_weight", type=float, default=0.03)
    inc.add_argument("--descriptor_volume_margin", type=float, default=0.0)

    clf = parser.add_argument_group("E. Geometry classifier/evaluation")
    clf.add_argument("--classifier_mode", type=str, default="geometry")
    clf.add_argument("--base_classifier_mode", type=str, default=None)
    clf.add_argument("--incremental_classifier_mode", type=str, default=None)
    clf.add_argument("--eval_classifier_mode", type=str, default="geometry")
    clf.add_argument("--logit_scale", type=float, default=8.0)
    clf.add_argument("--loss_scale", type=float, default=None)
    clf.add_argument("--residual_variance_scale", type=float, default=1.0)
    clf.add_argument("--energy_normalize_by_dim", type=str2bool, default=True)
    clf.add_argument("--use_logdet_energy", type=str2bool, default=False)
    clf.add_argument("--logdet_energy_weight", type=float, default=0.0)
    clf.add_argument("--center_logdet_energy", type=str2bool, default=False)
    clf.add_argument("--use_reliability_penalty", type=str2bool, default=False)
    clf.add_argument("--reliability_energy_weight", type=float, default=0.0)
    clf.add_argument("--geometry_logit_clip", type=float, default=0.0)
    clf.add_argument("--best_state_metric", type=str, default="geometry_score")
    clf.add_argument("--label_smoothing", type=float, default=0.0)
    clf.add_argument("--ce_logit_clip", type=float, default=50.0)
    clf.add_argument("--grad_clip_base", type=float, default=1.0)
    clf.add_argument("--grad_clip_inc", type=float, default=0.5)

    spec = parser.add_argument_group("F. HSI spectral metadata")
    spec.add_argument("--spectral_summary_mode", type=str, default="center", choices=["center", "mean"])
    spec.add_argument("--spectral_summary_is_physical", type=str2bool, default=False)
    spec.add_argument("--raw_spectral_summary_is_physical", type=str2bool, default=True)
    spec.add_argument("--external_spectra_are_physical", type=str2bool, default=True)
    spec.add_argument("--allow_nonphysical_spectral_summary", type=str2bool, default=False)
    spec.add_argument("--spectral_require_physical_summary", type=str2bool, default=True)
    spec.add_argument("--use_spectral_geometry", type=str2bool, default=False)
    spec.add_argument("--spectral_energy_weight", type=float, default=0.0)
    spec.add_argument("--spectral_derivative_weight", type=float, default=0.50)
    spec.add_argument("--spectral_second_derivative_weight", type=float, default=0.25)
    spec.add_argument("--band_energy_weight", type=float, default=0.0)
    spec.add_argument("--require_raw_spectral_metadata", type=str2bool, default=True)

    safety = parser.add_argument_group("G. Safety, diagnostics, visualization")
    safety.add_argument("--strict_non_exemplar", type=str2bool, default=True)
    safety.add_argument("--strict_feature_contract", type=str2bool, default=True)
    safety.add_argument("--strict_updated_stack", type=str2bool, default=True)
    safety.add_argument("--freeze_projection_during_incremental", type=str2bool, default=True)
    safety.add_argument("--allow_incremental_projection_training", type=str2bool, default=False)
    safety.add_argument("--freeze_classifier_during_incremental", type=str2bool, default=True)
    safety.add_argument("--save_geometry_diagnostics", type=str2bool, default=True)
    safety.add_argument("--save_classification_report", type=str2bool, default=True)
    safety.add_argument("--save_final_classification_report", type=str2bool, default=True)
    safety.add_argument("--skip_phase_maps", type=str2bool, default=False)
    safety.add_argument("--viz_class_cmap", type=str, default="nipy_spectral")
    safety.add_argument("--viz_background_color", type=str, default="#20252B")
    safety.add_argument("--viz_save_numpy", type=str2bool, default=True)
    safety.add_argument("--deterministic", type=str2bool, default=False)
    safety.add_argument("--debug_verbose", type=str2bool, default=False)
    safety.add_argument("--refresh_before_validation", type=str2bool, default=True)
    safety.add_argument("--validation_refresh_every", type=int, default=1)
    safety.add_argument("--base_geometry_refresh_every", type=int, default=1)
    safety.add_argument("--print_base_geometry_diagnostics", type=str2bool, default=True)
    safety.add_argument("--geometry_diag_anchors_per_class", type=int, default=64)
    safety.add_argument("--geometry_diag_topk_pairs", type=int, default=20)
    safety.add_argument("--geometry_diag_topk_bands", type=int, default=5)
    safety.add_argument("--base_cert_min_geom_acc", type=float, default=95.0)
    safety.add_argument("--base_cert_min_reliability", type=float, default=0.15)
    safety.add_argument("--base_cert_min_mean_reliability", type=float, default=0.35)
    safety.add_argument("--base_cert_max_subspace_overlap", type=float, default=0.55)
    safety.add_argument("--base_cert_subspace_warn_overlap", type=float, default=0.72)
    safety.add_argument("--base_cert_max_geometry_conflict", type=float, default=1.35)
    safety.add_argument("--base_cert_max_geometry_conflict_soft", type=float, default=1.40)
    safety.add_argument("--base_cert_max_guided_geometry_conflict", type=float, default=0.18)
    safety.add_argument("--base_cert_max_band_similarity", type=float, default=0.90)
    safety.add_argument("--base_cert_max_spectral_shape_similarity", type=float, default=0.85)

    legacy = parser.add_argument_group("H. Legacy flags accepted but disabled")
    legacy.add_argument("--use_geometry_transport", type=str2bool, default=False)
    legacy.add_argument("--use_sglat_transport", type=str2bool, default=False)
    legacy.add_argument("--transport_mode", type=str, default="new_row_only")
    legacy.add_argument("--allow_old_model_transport", type=str2bool, default=False)
    legacy.add_argument("--allow_transport_without_adapter", type=str2bool, default=False)
    legacy.add_argument("--use_energy_calibrator", type=str2bool, default=False)
    legacy.add_argument("--energy_calibrator_type", type=str, default="none")
    legacy.add_argument("--energy_calibration_weight", type=float, default=0.0)
    legacy.add_argument("--use_adaptive_boundary", type=str2bool, default=False)
    legacy.add_argument("--use_incremental_adapter", type=str2bool, default=False)
    legacy.add_argument("--adapter_bottleneck", type=int, default=32)
    legacy.add_argument("--adapter_max_scale", type=float, default=0.0)
    legacy.add_argument("--adapter_dropout", type=float, default=0.0)
    legacy.add_argument("--adapter_gate_bias_init", type=float, default=-3.0)
    legacy.add_argument("--adapter_lr", type=float, default=0.0)
    legacy.add_argument("--adapter_weight_decay", type=float, default=0.0)
    legacy.add_argument("--g2rpa_adapter_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_old_delta_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_old_gate_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_old_energy_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_old_margin_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_delta_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_new_gate_weight", type=float, default=0.0)
    legacy.add_argument("--adapter_new_gate_target", type=float, default=0.0)
    legacy.add_argument("--adapter_new_gate_max_target", type=float, default=0.0)
    legacy.add_argument("--disable_incremental_adapter", type=str2bool, default=False)
    legacy.add_argument("--use_geometry_calibrator", type=str2bool, default=False)
    legacy.add_argument("--use_bicyc_geometry_cycle", type=str2bool, default=False)
    legacy.add_argument("--bss_weight", type=float, default=0.0)
    legacy.add_argument("--sym_bss_weight", type=float, default=0.0)
    legacy.add_argument("--gdr_weight", type=float, default=0.0)
    legacy.add_argument("--anchor_consistency_weight", type=float, default=0.0)
    legacy.add_argument("--use_mssl_loss", type=str2bool, default=False)
    legacy.add_argument("--unsafe_ablation_use_mssl_loss", type=str2bool, default=False)
    legacy.add_argument("--mssl_weight", type=float, default=0.0)
    legacy.add_argument("--mssl_inc_weight", type=float, default=0.0)
    legacy.add_argument("--bank_refresh_every", type=int, default=0)
    legacy.add_argument("--early_stop_patience", type=int, default=0)
    legacy.add_argument("--base_early_stop_patience", type=int, default=0)
    legacy.add_argument("--incremental_early_stop_patience", type=int, default=0)
    legacy.add_argument("--eval_semantic_mode", type=str, default="identity")
    legacy.add_argument("--use_pretrain_incremental_baseline", type=str2bool, default=False)
    legacy.add_argument("--allow_unknown_legacy_args", type=str2bool, default=False)

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    args._unknown_args = unknown
    return args


def resolve_experiment_config(args: argparse.Namespace) -> Tuple[argparse.Namespace, List[Dict[str, Any]], Dict[str, Any]]:
    original = namespace_to_dict(args)
    resolved = argparse.Namespace(**copy.deepcopy(original))
    reasons: Dict[str, str] = {}

    mode = normalize_incremental_update_mode(
        getattr(resolved, "incremental_update_mode", "spectral_coupled_geometry_replay")
    )
    _set_resolved(resolved, "incremental_update_mode", mode, reasons, "sctgr_method_identity")

    for key, value in {
        "classifier_mode": "geometry_only",
        "base_classifier_mode": "geometry_only",
        "incremental_classifier_mode": "geometry_only",
        "eval_classifier_mode": "geometry_only",
    }.items():
        _set_resolved(resolved, key, normalize_classifier_mode(value), reasons, "seen_local_geometry_classifier")

    forced_true = {
        "strict_non_exemplar": True,
        "strict_feature_contract": True,
        "strict_updated_stack": True,
        "freeze_projection_during_incremental": True,
        "freeze_classifier_during_incremental": True,
        "base_class_balance": True,
        "strict_base_component_coverage": True,
        "use_spectral_coupled_replay": True,
        "replay_energy_filter": True,
        "refine_new_descriptors": True,
        "use_descriptor_refinement": True,
        "require_raw_spectral_metadata": True,
        "base_require_physical_spectral_shape": True,
    }
    for k, v in forced_true.items():
        _set_resolved(resolved, k, v, reasons, "sctgr_contract")

    forced_false = {
        "allow_incremental_projection_training": False,
        "use_geometry_transport": False,
        "use_sglat_transport": False,
        "allow_old_model_transport": False,
        "allow_transport_without_adapter": False,
        "use_energy_calibrator": False,
        "use_adaptive_boundary": False,
        "use_incremental_adapter": False,
        "disable_incremental_adapter": True,
        "use_geometry_calibrator": False,
        "use_bicyc_geometry_cycle": False,
        "use_mssl_loss": False,
        "use_pretrain_incremental_baseline": False,
        "use_spectral_geometry": False,
        "geometry_normalize_logits": False,
        "allow_nonphysical_spectral_summary": False,
        "use_logdet_energy": False,
        "center_logdet_energy": False,
        "use_reliability_penalty": False,
    }
    for k, v in forced_false.items():
        _set_resolved(resolved, k, v, reasons, "removed_or_inconsistent_branch")

    forced_values = {
        "residual_variance_scale": 1.0,
        "energy_normalize_by_dim": True,
        "logdet_energy_weight": 0.0,
        "reliability_energy_weight": 0.0,
        "spectral_energy_weight": 0.0,
        "band_energy_weight": 0.0,
        "energy_calibration_weight": 0.0,
        "bss_weight": 0.0,
        "sym_bss_weight": 0.0,
        "gdr_weight": 0.0,
        "anchor_consistency_weight": 0.0,
        "mssl_weight": 0.0,
        "mssl_inc_weight": 0.0,
        "bank_refresh_every": 0,
        "early_stop_patience": 0,
        "base_early_stop_patience": 0,
        "incremental_early_stop_patience": 0,
        "adapter_max_scale": 0.0,
        "adapter_lr": 0.0,
        "adapter_weight_decay": 0.0,
        "g2rpa_adapter_weight": 0.0,
        "adapter_old_delta_weight": 0.0,
        "adapter_old_gate_weight": 0.0,
        "adapter_old_energy_weight": 0.0,
        "adapter_old_margin_weight": 0.0,
        "adapter_delta_weight": 0.0,
        "adapter_new_gate_weight": 0.0,
        "adapter_new_gate_target": 0.0,
        "adapter_new_gate_max_target": 0.0,
    }
    for k, v in forced_values.items():
        _set_resolved(resolved, k, v, reasons, "strict_energy_or_removed_branch")

    required_positive = {
        "base_ce_weight": 1.0,
        "base_srpgr_weight": 1.0,
        "base_gics_weight": 0.20,
        "base_energy_margin_weight": 0.15,
        "base_energy_margin": 0.25,
        "pgr_weight": 0.10,
        "pgr_compact_weight": 0.15,
        "pgr_center_weight": 0.25,
        "pgr_subspace_weight": 0.15,
        "pgr_band_weight": 0.05,
        "pgr_volume_weight": 0.05,
        "pgr_max_subspace_overlap": 0.55,
        "pgr_min_class_variance": 0.015,
        "spectral_geometry_rank": 5,
        "coupling_ridge": 1e-3,
        "coupling_min_reliability": 0.20,
        "replay_candidate_multiplier": 4,
        "replay_min_per_class": 24,
        "replay_max_per_class": 64,
        "pair_risk_topk": 3,
    }
    for k, fallback in required_positive.items():
        if float(getattr(resolved, k, fallback)) <= 0.0:
            _set_resolved(resolved, k, fallback, reasons, "mandatory_sctgr_component")

    pca_active = (not bool(getattr(resolved, "no_pca", False))) and int(getattr(resolved, "pca_components", 0) or 0) > 0
    if pca_active:
        _set_resolved(resolved, "spectral_summary_is_physical", False, reasons, "pca_features_are_not_physical_spectra")
    _set_resolved(resolved, "raw_spectral_summary_is_physical", True, reasons, "raw_center_spectrum_contract")
    _set_resolved(resolved, "external_spectra_are_physical", True, reasons, "raw_center_spectrum_contract")

    if getattr(resolved, "descriptor_refine_steps_per_epoch", None) is None:
        _set_resolved(
            resolved,
            "descriptor_refine_steps_per_epoch",
            int(getattr(resolved, "descriptor_refine_steps", 20)),
            reasons,
            "default_filled",
        )
    if getattr(resolved, "loss_scale", None) is None:
        _set_resolved(resolved, "loss_scale", float(getattr(resolved, "logit_scale", 8.0)), reasons, "classifier_scale_alias")

    if bool(getattr(resolved, "base_only", False)):
        _set_resolved(resolved, "epochs_inc", 0, reasons, "base_only")
        _set_resolved(resolved, "lr_inc", 0.0, reasons, "base_only")
        _set_resolved(resolved, "best_state_metric", "geometry_score", reasons, "base_only")

    method_identity = build_method_identity(resolved)
    diff = compute_config_diff(original, namespace_to_dict(resolved), reasons)
    return resolved, diff, method_identity

def build_method_identity(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "method_name": "Spectral-Coupled Tangent Geometry Replay with New-Row Descriptor Adaptation for HSI NECIL",
        "short_name": "SCTGR-HSI",
        "main_path": True,
        "incremental_update_mode": "spectral_coupled_geometry_replay",
        "base": {
            "temporary_ce_head": True,
            "mandatory_balanced_ce": True,
            "mandatory_gics": True,
            "mandatory_pgr": ["compact", "center", "subspace", "band", "volume"],
            "physical_spectral_shape_guidance": True,
            "geometry_bank_space": "canonical_projected_z",
            "paired_raw_spectral_geometry": True,
            "spectral_to_feature_coupling": True,
            "base_geometry_energy_margin": True,
        },
        "incremental": {
            "frozen_backbone": True,
            "frozen_projection": True,
            "frozen_classifier": True,
            "frozen_old_geometry_rows": True,
            "spectral_consistent_core_replay": True,
            "risk_directed_tangent_replay": True,
            "energy_filtered_replay": True,
            "new_row_descriptor_refinement_only": True,
            "seen_local_geometry_classifier": True,
            "joint_class_balanced_ce": True,
            "geometry_energy_margin": True,
            "bidirectional_old_new_invasion": True,
        },
        "energy_contract": {
            "rank_normalized_parallel": True,
            "residual_dimension_normalized": True,
            "residual_variance_scale": 1.0,
            "logdet_score_bias": False,
            "centered_logdet_score_bias": False,
            "reliability_score_bias": False,
            "classifier_replay_loss_energy_identical": True,
        },
        "forbidden": {
            "raw_exemplars": False,
            "stored_old_features": False,
            "kd_teacher": False,
            "prototype_classifier": False,
            "feature_adapter": False,
            "old_row_transport": False,
            "score_calibrator": False,
            "adaptive_boundary": False,
            "projection_plasticity": False,
        },
        "label_convention": {
            "dataset_labels": "global_class_ids",
            "geometry_bank_rows": "global_class_ids",
            "classifier_logits": "seen_local_column_order",
            "ce_targets": "seen_local_labels",
            "evaluation_predictions": "mapped_to_global_class_ids",
            "old_new_partition": "explicit_global_class_lists",
        },
        "raw_spectral_metadata_required": bool(getattr(args, "require_raw_spectral_metadata", True)),
    }

def validate_config(args: argparse.Namespace, *, num_classes: Optional[int] = None) -> None:
    unknown_args = list(getattr(args, "_unknown_args", []) or [])
    if unknown_args and not bool(getattr(args, "allow_unknown_legacy_args", False)):
        raise ValueError(
            "Unknown CLI arguments are forbidden because they can silently select a different method: "
            + str(unknown_args)
        )
    if args.dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {args.dataset!r}.")
    if not bool(args.strict_non_exemplar):
        raise ValueError("SCTGR requires --strict_non_exemplar true.")
    if int(args.patch_size) <= 0 or int(args.patch_size) % 2 == 0:
        raise ValueError("--patch_size must be a positive odd integer.")
    if float(args.train_ratio) <= 0 or float(args.val_ratio) < 0 or float(args.train_ratio) + float(args.val_ratio) >= 1.0:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")
    if int(args.epochs_base) <= 0 or int(args.epochs_inc) < 0:
        raise ValueError("epochs_base must be > 0 and epochs_inc must be >= 0.")
    if int(args.batch_size) <= 0 or int(args.d_model) <= 0:
        raise ValueError("batch_size and d_model must be positive.")
    if int(args.subspace_rank) <= 0 or int(args.subspace_rank) >= int(args.d_model):
        raise ValueError("Require 0 < subspace_rank < d_model.")
    if int(args.spectral_geometry_rank) <= 0:
        raise ValueError("spectral_geometry_rank must be positive.")
    if not bool(args.no_pca) and int(args.pca_components) <= 0:
        raise ValueError("pca_components must be positive unless --no_pca is used.")
    if float(args.geom_var_floor) <= 0 or float(args.spectral_variance_floor) <= 0:
        raise ValueError("feature and spectral variance floors must be positive.")
    if float(args.coupling_ridge) <= 0:
        raise ValueError("coupling_ridge must be positive.")
    if not (0.0 <= float(args.coupling_min_reliability) <= 1.0):
        raise ValueError("coupling_min_reliability must be in [0,1].")
    if normalize_incremental_update_mode(args.incremental_update_mode) != "spectral_coupled_geometry_replay":
        raise ValueError("incremental_update_mode must be spectral_coupled_geometry_replay.")
    if not bool(args.use_spectral_coupled_replay):
        raise ValueError("SCTGR requires use_spectral_coupled_replay=true.")
    if bool(args.allow_incremental_projection_training) or not bool(args.freeze_projection_during_incremental):
        raise ValueError("Incremental projection training invalidates the frozen GeometryBank coordinate system.")
    if not bool(args.freeze_classifier_during_incremental):
        raise ValueError("The geometry classifier has no incremental trainable parameters and must remain frozen.")

    forbidden_bool = [
        "use_geometry_transport", "use_sglat_transport", "allow_old_model_transport",
        "use_energy_calibrator", "use_adaptive_boundary", "use_incremental_adapter",
        "use_geometry_calibrator", "use_bicyc_geometry_cycle", "use_spectral_geometry",
        "use_mssl_loss",
    ]
    bad = [k for k in forbidden_bool if bool(getattr(args, k, False))]
    if bad:
        raise ValueError(f"Removed architecture branches must be false: {bad}")
    if not bool(getattr(args, "disable_incremental_adapter", True)):
        raise ValueError("disable_incremental_adapter must remain true.")

    if abs(float(args.residual_variance_scale) - 1.0) > 1e-12:
        raise ValueError("residual_variance_scale must be exactly 1.0 for classifier/replay/loss consistency.")
    if not bool(args.energy_normalize_by_dim):
        raise ValueError("energy_normalize_by_dim must remain true for the rank/residual normalized energy contract.")
    if (
        bool(args.use_logdet_energy)
        or float(args.logdet_energy_weight) != 0.0
        or bool(args.center_logdet_energy)
    ):
        raise ValueError(
            "Log-determinant score bias is removed from the classifier. "
            "Require use_logdet_energy=false, logdet_energy_weight=0.0, "
            "and center_logdet_energy=false."
        )
    if bool(args.use_reliability_penalty) or float(args.reliability_energy_weight) != 0.0:
        raise ValueError("Reliability controls replay/trust, not classifier logits.")
    if bool(args.spectral_summary_is_physical) and (not bool(args.no_pca)) and int(args.pca_components) > 0:
        raise ValueError("PCA feature channels cannot be marked as physical wavelengths.")
    if not bool(args.raw_spectral_summary_is_physical):
        raise ValueError("Raw center spectra must be marked physical for spectral-coupled replay.")

    for key in (
        "base_ce_weight", "base_srpgr_weight", "base_gics_weight",
        "base_energy_margin_weight", "base_energy_margin", "pgr_weight",
        "pgr_compact_weight", "pgr_center_weight", "pgr_subspace_weight",
        "pgr_band_weight", "pgr_volume_weight", "geometry_energy_margin_weight",
        "geometry_energy_margin", "old_new_invasion_weight", "old_new_geometry_margin",
        "descriptor_refine_steps", "descriptor_refine_lr", "descriptor_trust_weight",
        "replay_min_per_class", "replay_max_per_class", "replay_candidate_multiplier",
    ):
        if float(getattr(args, key)) <= 0.0:
            raise ValueError(f"{key} must be > 0 in the SCTGR main path.")
    if int(args.replay_min_per_class) > int(args.replay_max_per_class):
        raise ValueError("replay_min_per_class cannot exceed replay_max_per_class.")
    if float(args.gfa_parallel_scale) <= 0.0 or float(args.gfa_residual_scale) < 0.0:
        raise ValueError("gfa_parallel_scale must be > 0 and gfa_residual_scale must be >= 0.")
    if int(args.replay_energy_filter_multiplier) <= 0 or int(args.replay_resample_rounds) <= 0:
        raise ValueError("replay candidate multiplier and resample rounds must be positive.")
    if not (0.0 <= float(args.directed_replay_min_ratio) <= float(args.directed_replay_max_ratio) <= 1.0):
        raise ValueError("Require 0 <= directed_replay_min_ratio <= directed_replay_max_ratio <= 1.")
    if not (0.0 <= float(args.core_replay_ratio) <= 1.0):
        raise ValueError("core_replay_ratio must be in [0,1].")
    if not (0.0 < float(args.pgr_max_subspace_overlap) <= 1.0):
        raise ValueError("pgr_max_subspace_overlap must be in (0,1].")
    if float(args.pgr_min_class_variance) >= float(args.pgr_max_class_variance):
        raise ValueError("pgr_min_class_variance must be smaller than pgr_max_class_variance.")

    if num_classes is not None:
        if int(num_classes) != int(DATASET_INFO[args.dataset]["classes"]):
            raise ValueError(
                f"Loaded class count {num_classes} does not match DATASET_INFO[{args.dataset}]="
                f"{DATASET_INFO[args.dataset]['classes']}."
            )
        if args.base_classes is not None and (int(args.base_classes) <= 0 or int(args.base_classes) >= int(num_classes)):
            raise ValueError(f"base_classes={args.base_classes} must be in [1,{num_classes - 1}].")
        if args.increment is not None and int(args.increment) <= 0:
            raise ValueError("increment must be positive.")
    seeds = parse_seed_list(getattr(args, "seed_list", ""))
    if seeds is not None and len(seeds) != int(args.num_runs):
        raise ValueError(f"seed_list has {len(seeds)} seeds but num_runs={args.num_runs}.")
    if int(args.num_runs) <= 0:
        raise ValueError("num_runs must be >= 1.")

def save_config_files(
    save_root: str,
    original: Dict[str, Any],
    resolved: argparse.Namespace,
    diff: List[Dict[str, Any]],
    method_identity: Dict[str, Any],
) -> Dict[str, str]:
    os.makedirs(save_root, exist_ok=True)
    paths = {
        "config_original": save_json(os.path.join(save_root, "config_original.json"), original),
        "config_resolved": save_json(os.path.join(save_root, "config_resolved.json"), namespace_to_dict(resolved)),
        "config_diff": save_json(os.path.join(save_root, "config_diff.json"), diff),
        "method_identity": save_json(os.path.join(save_root, "method_identity.json"), method_identity),
    }
    if getattr(resolved, "_unknown_args", None):
        paths["unknown_args"] = save_json(os.path.join(save_root, "unknown_args.json"), {"unknown_args": list(resolved._unknown_args)})
    return paths


def print_config_summary(args: argparse.Namespace, diff: List[Dict[str, Any]], method_identity: Dict[str, Any]) -> None:
    print("[Method]", method_identity["method_name"])
    print(f"[Method] short_name={method_identity['short_name']} | update_mode={args.incremental_update_mode}")
    print(f"[Classifier] base={args.base_classifier_mode} | incremental={args.incremental_classifier_mode} | eval={args.eval_classifier_mode}")
    print(
        f"[Energy] residual_scale={args.residual_variance_scale} | rank/dim normalized=True | "
        f"logdet=False | centered_logdet=False | reliability_logit_bias=False"
    )
    print(
        f"[Base] CE={args.base_ce_weight} | GICS={args.base_gics_weight} | PGR={args.pgr_weight} | "
        f"margin={args.base_energy_margin_weight}@{args.base_energy_margin}"
    )
    print(
        f"[Replay] target/class={args.gfa_samples_per_class} | range={args.replay_min_per_class}-{args.replay_max_per_class} | "
        f"core={args.core_replay_ratio} | directed={args.directed_replay_min_ratio}-{args.directed_replay_max_ratio} | "
        f"risk_topk={args.pair_risk_topk}"
    )
    print(
        f"[Coupling] spectral_rank={args.spectral_geometry_rank} | ridge={args.coupling_ridge} | "
        f"min_reliability={args.coupling_min_reliability} | tangent_clip={args.spectral_tangent_clip}"
    )
    print(
        f"[Refinement] steps/epoch={args.descriptor_refine_steps_per_epoch} | lr={args.descriptor_refine_lr} | "
        f"trust={args.descriptor_trust_weight}"
    )
    print("[Forbidden OFF] adapters | transport | calibrator | adaptive boundary | KD | raw exemplars | spectral classifier")
    if getattr(args, "_unknown_args", None):
        mode = "IGNORED BY USER OVERRIDE" if bool(getattr(args, "allow_unknown_legacy_args", False)) else "ERROR"
        print(f"[UNKNOWN ARGS {mode}] {args._unknown_args}")
    if diff:
        print("[Config Diff] resolved changes:")
        for row in diff[:40]:
            print(f"  - {row['name']}: {row['original']} -> {row['resolved']} ({row['reason']})")
        if len(diff) > 40:
            print(f"  ... {len(diff) - 40} more changes saved in config_diff.json")

# -----------------------------------------------------------------------------
# Reproducibility and data loading
# -----------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def load_hsi_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    apply_reduction = (not bool(args.no_pca)) and str(args.reduction_method).lower() != "none"
    raw_hsi_physical = None
    label_policy = None
    try:
        load_out = LoadHSIData(
            method=args.dataset,
            base_dir=args.data_dir,
            apply_reduction=apply_reduction,
            n_components=args.pca_components,
            reduction_method=args.reduction_method,
            return_label_policy=True,
            return_raw_hsi=True,
        )
        if len(load_out) == 7:
            hsi, gt, num_classes, target_names, has_bg, label_policy, raw_hsi_physical = load_out
        else:
            hsi, gt, num_classes, target_names, has_bg, label_policy = load_out
    except TypeError:
        hsi, gt, num_classes, target_names, has_bg = LoadHSIData(
            method=args.dataset,
            base_dir=args.data_dir,
            apply_reduction=apply_reduction,
            n_components=args.pca_components,
            reduction_method=args.reduction_method,
        )

    validate_config(args, num_classes=int(num_classes))

    try:
        cube_out = ImageCubes(
            HSI=hsi,
            GT=gt,
            WS=args.patch_size,
            removeZeroLabels=True,
            has_background=has_bg,
            num_classes=num_classes,
            pytorch_format=True,
            label_policy=label_policy,
            return_center_spectra=True,
            raw_hsi_for_spectra=raw_hsi_physical,
        )
        if len(cube_out) == 4:
            patches, labels, coords, raw_center_spectra = cube_out
        else:
            patches, labels, coords = cube_out
            raw_center_spectra = None
    except TypeError:
        patches, labels, coords = ImageCubes(
            HSI=hsi,
            GT=gt,
            WS=args.patch_size,
            removeZeroLabels=True,
            has_background=has_bg,
            num_classes=num_classes,
            pytorch_format=True,
        )
        raw_center_spectra = None

    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    if np.any(labels_np < 0):
        raise RuntimeError("Dataset labels must be non-negative global class IDs after background removal.")

    args.num_bands = int(patches.shape[1])
    args.max_classes = int(num_classes)
    args.raw_spectral_summary_is_physical = bool(raw_center_spectra is not None)
    if bool(getattr(args, "require_raw_spectral_metadata", True)) and raw_center_spectra is None:
        raise RuntimeError(
            "SCTGR requires aligned raw physical center spectra. The loader returned none. "
            "Update LoadHSIData/ImageCubes so PCA patches and raw center spectra are returned together."
        )
    if raw_center_spectra is not None:
        raw_arr = raw_center_spectra.detach().cpu().numpy() if torch.is_tensor(raw_center_spectra) else np.asarray(raw_center_spectra)
        if raw_arr.ndim != 2 or raw_arr.shape[0] != labels_np.size:
            raise RuntimeError(
                f"Raw center spectra must be [N,S] aligned with labels; got {raw_arr.shape}, N={labels_np.size}."
            )
        if not np.isfinite(raw_arr).all():
            raise RuntimeError("Raw center spectra contain NaN/Inf values.")

    dataset_summary = {
        "dataset": args.dataset,
        "name": DATASET_INFO[args.dataset]["name"],
        "expected_raw_bands": DATASET_INFO[args.dataset]["bands"],
        "used_channels": int(patches.shape[1]),
        "pca_active": bool(apply_reduction),
        "pca_components": int(args.pca_components) if apply_reduction else 0,
        "num_classes": int(num_classes),
        "has_background": bool(has_bg),
        "labeled_samples": int(labels_np.size),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "raw_physical_center_spectra_available": bool(raw_center_spectra is not None),
        "raw_center_spectra_shape": list(raw_center_spectra.shape) if raw_center_spectra is not None and hasattr(raw_center_spectra, "shape") else None,
        "label_policy": label_policy,
    }
    print("[Dataset]")
    print(f"  name={dataset_summary['name']} | classes={num_classes} | used_channels={patches.shape[1]} | PCA={apply_reduction}")
    print(f"  samples={dataset_summary['labeled_samples']} | raw_spectra={dataset_summary['raw_physical_center_spectra_available']}")
    return {
        "hsi": hsi,
        "gt": gt,
        "num_classes": int(num_classes),
        "target_names": list(target_names),
        "has_background": bool(has_bg),
        "label_policy": label_policy,
        "patches": patches,
        "labels": labels,
        "coords": coords,
        "raw_center_spectra": raw_center_spectra,
        "summary": dataset_summary,
    }


def build_incremental_dataset(args: argparse.Namespace, data: Dict[str, Any]) -> IncrementalHSIDataset:
    if args.base_classes is None:
        args.base_classes = 6 if args.dataset in {"IP", "SA", "HC"} else max(2, data["num_classes"] // 2)
    if args.increment is None:
        remaining = max(1, data["num_classes"] - int(args.base_classes))
        args.increment = 3 if remaining >= 3 else 1
    validate_config(args, num_classes=data["num_classes"])

    kwargs = dict(
        patches=data["patches"],
        labels=data["labels"],
        coords=data["coords"],
        gt_shape=data["gt"].shape,
        GT=data["gt"].copy().astype(np.int64),
        base_classes=int(args.base_classes),
        increment=int(args.increment),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        device=str(args.device),
        min_train_per_class=int(args.min_train_per_class),
        strict_non_exemplar=bool(args.strict_non_exemplar),
    )
    optional = {
        "num_workers": int(args.num_workers),
        "target_names": data["target_names"],
        "label_policy": data.get("label_policy"),
        "return_metadata": True,
        "include_metadata": True,
        "raw_spectra": data.get("raw_center_spectra"),
        "center_spectra": data.get("raw_center_spectra"),
        "spectra_are_physical": bool(data.get("raw_center_spectra") is not None and args.raw_spectral_summary_is_physical),
    }
    sig = inspect.signature(IncrementalHSIDataset.__init__)
    for k, v in optional.items():
        if k in sig.parameters:
            kwargs[k] = v
    dataset = IncrementalHSIDataset(**kwargs)
    validate_incremental_dataset(dataset, args, data["num_classes"])
    print_incremental_protocol(dataset)
    return dataset


def phase_to_classes_as_list(dataset: Any) -> List[List[int]]:
    if not hasattr(dataset, "phase_to_classes"):
        raise RuntimeError("Incremental dataset must expose phase_to_classes.")
    ptc = getattr(dataset, "phase_to_classes")
    if isinstance(ptc, dict):
        phases = sorted(int(k) for k in ptc.keys())
        expected = list(range(len(phases)))
        if phases != expected:
            raise RuntimeError(f"phase_to_classes keys must be contiguous 0..P-1, got {phases}")
        return [[int(c) for c in list(ptc[p])] for p in phases]
    if isinstance(ptc, (list, tuple)):
        return [[int(c) for c in list(v)] for v in ptc]
    raise RuntimeError(f"Unsupported phase_to_classes type: {type(ptc)}")


def validate_incremental_dataset(dataset: Any, args: argparse.Namespace, num_classes: int) -> None:
    phases = phase_to_classes_as_list(dataset)
    if not phases:
        raise RuntimeError("phase_to_classes is empty.")
    all_classes: List[int] = []
    for p_idx, cls_list in enumerate(phases):
        if not cls_list:
            raise RuntimeError(f"Phase {p_idx} has no classes.")
        all_classes.extend(cls_list)
    if len(all_classes) != len(set(all_classes)):
        dup = sorted(c for c in set(all_classes) if all_classes.count(c) > 1)
        raise RuntimeError(f"A class appears in more than one phase: {dup}")
    expected = set(range(int(num_classes)))
    actual = set(all_classes)
    if actual != expected:
        raise RuntimeError(f"Phase classes must cover exactly 0..{num_classes - 1}; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if hasattr(dataset, "assert_non_exemplar"):
        dataset.assert_non_exemplar()
    if not bool(args.strict_non_exemplar):
        raise RuntimeError("strict_non_exemplar must be true.")
    for a, b in [("train_indices", "val_indices"), ("train_indices", "test_indices"), ("val_indices", "test_indices")]:
        if hasattr(dataset, a) and hasattr(dataset, b):
            overlap = sorted(set(map(int, getattr(dataset, a))) & set(map(int, getattr(dataset, b))))
            if overlap:
                raise RuntimeError(f"Dataset split leakage: {a} and {b} overlap. First overlaps={overlap[:20]}")


def print_incremental_protocol(dataset: Any) -> None:
    print("[Incremental Protocol]")
    for p_idx, cls_list in enumerate(phase_to_classes_as_list(dataset)):
        print(f"  phase_{p_idx}_classes={list(map(int, cls_list))}")


def resolve_target_names(dataset: Any, raw_target_names: List[str]) -> List[str]:
    if hasattr(dataset, "inv_label_map"):
        names = []
        for sid in range(int(getattr(dataset, "num_classes", len(raw_target_names)))):
            input_label = int(dataset.inv_label_map[sid])
            names.append(raw_target_names[input_label] if input_label < len(raw_target_names) else f"Class {sid}")
        return names
    return list(raw_target_names)


# -----------------------------------------------------------------------------
# Model / trainer / evaluator
# -----------------------------------------------------------------------------



def assert_strict_energy_contract(args: argparse.Namespace, *, context: str) -> None:
    """Fail fast if any component can select a different geometry energy."""
    failures: List[str] = []
    if abs(float(getattr(args, "residual_variance_scale", 1.0)) - 1.0) > 1e-12:
        failures.append(
            f"residual_variance_scale={getattr(args, 'residual_variance_scale', None)!r}"
        )
    if not bool(getattr(args, "energy_normalize_by_dim", True)):
        failures.append("energy_normalize_by_dim=false")
    if bool(getattr(args, "use_logdet_energy", False)):
        failures.append("use_logdet_energy=true")
    if abs(float(getattr(args, "logdet_energy_weight", 0.0))) > 0.0:
        failures.append(
            f"logdet_energy_weight={getattr(args, 'logdet_energy_weight', None)!r}"
        )
    if bool(getattr(args, "center_logdet_energy", False)):
        failures.append("center_logdet_energy=true")
    if bool(getattr(args, "use_reliability_penalty", False)):
        failures.append("use_reliability_penalty=true")
    if abs(float(getattr(args, "reliability_energy_weight", 0.0))) > 0.0:
        failures.append(
            f"reliability_energy_weight={getattr(args, 'reliability_energy_weight', None)!r}"
        )
    if failures:
        raise RuntimeError(
            f"{context}: strict SCTGR energy contract violated: " + ", ".join(failures)
        )


def _read_energy_contract(obj: Any) -> Dict[str, Any]:
    return {
        "residual_variance_scale": float(getattr(obj, "residual_variance_scale", 1.0)),
        "energy_normalize_by_dim": bool(
            getattr(obj, "energy_normalize_by_dim", getattr(obj, "normalize_energy_by_dim", True))
        ),
        "use_logdet_energy": bool(getattr(obj, "use_logdet_energy", False)),
        "logdet_energy_weight": float(getattr(obj, "logdet_energy_weight", 0.0)),
        "center_logdet_energy": bool(getattr(obj, "center_logdet_energy", False)),
        "use_reliability_penalty": bool(getattr(obj, "use_reliability_penalty", False)),
        "reliability_energy_weight": float(getattr(obj, "reliability_energy_weight", 0.0)),
    }


def assert_model_energy_contract(model: torch.nn.Module) -> None:
    """Verify that model and classifier use the same strict replay energy."""
    expected = {
        "residual_variance_scale": 1.0,
        "energy_normalize_by_dim": True,
        "use_logdet_energy": False,
        "logdet_energy_weight": 0.0,
        "center_logdet_energy": False,
        "use_reliability_penalty": False,
        "reliability_energy_weight": 0.0,
    }
    failures: List[str] = []
    for owner_name, owner in (
        ("model", model),
        ("classifier", getattr(model, "classifier", None)),
    ):
        if owner is None:
            failures.append(f"{owner_name}=missing")
            continue
        state = _read_energy_contract(owner)
        for key, wanted in expected.items():
            got = state[key]
            if isinstance(wanted, float):
                if abs(float(got) - wanted) > 1e-12:
                    failures.append(f"{owner_name}.{key}={got!r}, expected {wanted!r}")
            elif bool(got) != bool(wanted):
                failures.append(f"{owner_name}.{key}={got!r}, expected {wanted!r}")
    if failures:
        raise RuntimeError(
            "Model/classifier energy contract mismatch: " + "; ".join(failures)
        )


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    assert_strict_energy_contract(args, context="build_model(args)")
    model = NECILModel(args).to(device)
    assert_model_energy_contract(model)
    if int(getattr(model, "d_model", args.d_model)) != int(args.d_model):
        raise RuntimeError(f"model.d_model={getattr(model, 'd_model', None)} != args.d_model={args.d_model}")
    gb = getattr(model, "geometry_bank", None)
    if gb is None:
        raise RuntimeError("Model must expose geometry_bank.")
    bank_rank = getattr(gb, "rank", getattr(gb, "subspace_rank", None))
    if bank_rank is not None and int(bank_rank) != int(args.subspace_rank):
        raise RuntimeError(f"GeometryBank rank {bank_rank} != subspace_rank {args.subspace_rank}")
    if getattr(model, "classifier", None) is None:
        raise RuntimeError("Model must expose classifier.")

    required_model = (
        "assert_method_identity", "extract_projected_features", "compute_logits_from_features",
        "sample_geometry_replay", "set_incremental_mode", "export_memory_snapshot",
    )
    missing_model = [name for name in required_model if not hasattr(model, name)]
    if missing_model:
        raise RuntimeError(f"Model is not the updated SCTGR stack; missing APIs: {missing_model}")
    required_bank = (
        "build_candidate_geometry_rows", "commit_candidate_geometry_rows", "sample_replay",
        "assert_bank_valid", "get_valid_mask",
    )
    missing_bank = [name for name in required_bank if not hasattr(gb, name)]
    if missing_bank:
        raise RuntimeError(f"GeometryBank is not replay-ready; missing APIs: {missing_bank}")
    if getattr(model, "geometry_plastic_adapter", None) is not None:
        raise RuntimeError("Feature adapter module is forbidden in SCTGR.")

    model.incremental_update_mode = "spectral_coupled_geometry_replay"
    for name, value in {
        "use_geometry_gated_adapter": False,
        "use_incremental_adapter": False,
        "use_geometry_calibrator": False,
        "use_energy_calibrator": False,
        "use_geometry_transport": False,
        "use_sglat_transport": False,
        "use_adaptive_boundary": False,
    }.items():
        if hasattr(model, name):
            setattr(model, name, value)
    model.assert_method_identity()
    print("[Model]")
    print(f"  feature_dim={args.d_model} | feature_rank={args.subspace_rank} | spectral_rank={args.spectral_geometry_rank}")
    print(f"  classifier={type(model.classifier).__name__} | update_mode=spectral_coupled_geometry_replay")
    print("  incremental_trainability=new_descriptor_tensors_only | backbone/projection/classifier frozen")
    return model

def _canonical_method_value(name: str, value: Any) -> Any:
    """Canonicalize semantically equivalent method values for identity checks."""
    if name in {"base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "classifier_mode"}:
        return normalize_classifier_mode(value)
    if name == "incremental_update_mode":
        return normalize_incremental_update_mode(value)
    if name in {"use_geometry_gated_adapter", "use_spectral_coupled_replay"}:
        return bool(value)
    return value


def build_trainer(model: torch.nn.Module, dataset: Any, args: argparse.Namespace, run_dir: str) -> Trainer:
    assert_strict_energy_contract(args, context="build_trainer(args-before)")
    assert_model_energy_contract(model)
    before = namespace_to_dict(args)
    trainer = Trainer(model, dataset, args)
    assert_strict_energy_contract(args, context="build_trainer(args-after)")
    assert_model_energy_contract(model)
    after = namespace_to_dict(args)
    diff = compute_config_diff(before, after, {})
    save_json(os.path.join(run_dir, "config_diff_after_trainer.json"), diff)
    critical = {"incremental_update_mode", "base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "use_spectral_coupled_replay", "use_geometry_gated_adapter"}
    changed_critical = []
    for d in diff:
        name = d["name"]
        if name not in critical:
            continue
        before_v = _canonical_method_value(name, d.get("original"))
        after_v = _canonical_method_value(name, d.get("resolved"))
        if before_v != after_v:
            changed_critical.append(d)
    if changed_critical:
        raise RuntimeError(f"Trainer changed method identity after construction: {changed_critical}")
    if hasattr(trainer, "assert_method_identity"):
        trainer.assert_method_identity()
    return trainer


def build_evaluator() -> NECILEvaluator:
    return NECILEvaluator()


# -----------------------------------------------------------------------------
# Phase metadata and evaluation
# -----------------------------------------------------------------------------


def get_phase_info(dataset: Any, phase: int) -> Dict[str, Any]:
    phase = int(phase)
    if hasattr(dataset, "get_phase_info"):
        info = dataset.get_phase_info(phase)
        if isinstance(info, dict):
            old_classes = [int(c) for c in info.get("old_classes", [])]
            new_classes = [int(c) for c in info.get("new_classes", [])]
            seen_classes = [int(c) for c in info.get("seen_classes", old_classes + new_classes)]
            if not seen_classes:
                seen_classes = old_classes + new_classes
            if len(seen_classes) != len(set(seen_classes)):
                raise RuntimeError(f"Duplicate class in dataset.get_phase_info({phase}) seen_classes={seen_classes}")
            return {
                "phase": int(info.get("phase", phase)),
                "old_classes": old_classes,
                "new_classes": new_classes,
                "seen_classes": seen_classes,
                "old_class_count": int(info.get("old_class_count", len(old_classes))),
            }

    phases = phase_to_classes_as_list(dataset)
    if phase < 0 or phase >= len(phases):
        raise RuntimeError(f"Invalid phase {phase}; available phases are 0..{len(phases) - 1}.")
    new_classes = [int(c) for c in phases[phase]]
    old_classes: List[int] = []
    for p_idx in range(phase):
        old_classes.extend(int(c) for c in phases[p_idx])
    old_classes = list(dict.fromkeys(old_classes))
    seen_classes = old_classes + new_classes
    if len(seen_classes) != len(set(seen_classes)):
        raise RuntimeError(f"Duplicate class in seen_classes at phase {phase}: {seen_classes}")
    return {"phase": phase, "old_classes": old_classes, "new_classes": new_classes, "seen_classes": seen_classes, "old_class_count": len(old_classes)}


def set_model_phase(model: torch.nn.Module, phase_info: Dict[str, Any]) -> None:
    """Set phase state without enabling any trainable incremental network branch."""
    phase = int(phase_info["phase"])
    old_classes = [int(c) for c in phase_info.get("old_classes", [])]
    seen_classes = [int(c) for c in phase_info.get("seen_classes", [])]
    if phase == 0:
        if hasattr(model, "set_base_mode"):
            try:
                model.set_base_mode(train_backbone=False, train_projection=False)
            except TypeError:
                model.set_base_mode()
        elif hasattr(model, "set_phase"):
            model.set_phase(phase)
        else:
            model.current_phase = phase
    else:
        if not hasattr(model, "set_incremental_mode"):
            raise RuntimeError("Updated model must expose set_incremental_mode().")
        model.set_incremental_mode(
            phase=phase,
            old_classes=old_classes,
            old_class_count=len(old_classes),
            train_classifier_calibration=False,
            train_geometry_adapter=False,
        )
    if hasattr(model, "current_phase"):
        model.current_phase = phase
    if hasattr(model, "old_classes"):
        model.old_classes = list(old_classes)
    if hasattr(model, "old_class_count"):
        model.old_class_count = len(old_classes)
    if hasattr(model, "seen_classes"):
        model.seen_classes = list(seen_classes)
    if hasattr(model, "current_num_classes"):
        model.current_num_classes = len(seen_classes)

def unpack_eval_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
    if isinstance(batch, dict):
        patches = batch.get("image", batch.get("patch", batch.get("patches", None)))
        labels = batch.get("label", batch.get("labels", None))
        spectra = batch.get("spectrum", batch.get("spectra", None))
        coords = batch.get("coord", batch.get("coords", None))
        if patches is None or labels is None:
            raise RuntimeError(f"Unsupported eval batch dict keys: {list(batch.keys())}")
        return patches, labels, spectra, coords
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        spectra = batch[2] if len(batch) >= 3 else None
        coords = batch[3] if len(batch) >= 4 else None
        return batch[0], batch[1], spectra, coords
    raise RuntimeError(f"Unsupported eval batch type: {type(batch)}")


def prepare_eval_spectra(patches: torch.Tensor, spectra: Optional[torch.Tensor], args: argparse.Namespace) -> Tuple[Optional[torch.Tensor], bool, Dict[str, Any]]:
    if torch.is_tensor(spectra) and spectra.numel() > 0:
        s = spectra.to(device=patches.device, dtype=patches.dtype)
        if s.dim() == 4:
            s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
        elif s.dim() == 3:
            if s.size(0) == patches.size(0) and s.size(1) > 0 and s.size(2) > 1:
                s = s[:, :, s.size(-1) // 2]
            else:
                s = s.reshape(patches.size(0), -1)
        elif s.dim() == 1:
            if s.numel() % max(int(patches.size(0)), 1) != 0:
                raise RuntimeError("1-D spectra cannot be reshaped to batch.")
            s = s.reshape(patches.size(0), -1)
        elif s.dim() != 2:
            s = s.reshape(patches.size(0), -1)
        if s.size(0) != patches.size(0):
            raise RuntimeError("Spectral metadata batch mismatch during evaluation.")
        physical = bool(args.raw_spectral_summary_is_physical)
        source = "batch_metadata"
    else:
        s = None
        physical = False
        source = "none"
    pca_active = (not bool(args.no_pca)) and int(args.pca_components) > 0
    if pca_active and s is not None and s.size(1) <= int(args.pca_components):
        physical = False
    return s, bool(physical), {"source": source, "physical": bool(physical), "pca_active": bool(pca_active), "spectral_dim": int(s.size(1)) if s is not None else 0}


def forward_eval_batch(
    model: torch.nn.Module,
    patches: torch.Tensor,
    spectra: Optional[torch.Tensor],
    args: argparse.Namespace,
    seen_classes: List[int],
    *,
    old_classes: Optional[List[int]] = None,
    new_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    # Raw spectra are bank-construction/replay guidance, not a classifier branch.
    _, spectral_is_physical, spec_diag = prepare_eval_spectra(patches, spectra, args)
    out = model(
        patches,
        seen_classes=[int(c) for c in seen_classes],
        old_classes=[int(c) for c in (old_classes or [])],
        new_classes=[int(c) for c in (new_classes or [])],
        classifier_mode=normalize_classifier_mode(args.eval_classifier_mode),
        return_energy=True,
        return_parts=False,
        return_diagnostics=False,
    )
    if not isinstance(out, dict) or "logits" not in out or "energy" not in out:
        raise RuntimeError("Evaluation requires model output dict containing both logits and energy.")
    validate_seen_local_outputs(
        seen_classes=seen_classes,
        logits=out["logits"],
        energy=out["energy"],
        batch_size=int(patches.size(0)),
    )
    out["spectral_diagnostics"] = {
        **spec_diag,
        "used_in_classifier": False,
        "physical_metadata_available": bool(spectral_is_physical),
    }
    return out

def logits_to_global_predictions(logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
    return predictions_from_seen_local_scores(logits, seen_classes, lower_is_better=False)

@torch.no_grad()
def get_phase_predictions(
    model: torch.nn.Module,
    dataset: Any,
    phase_info: Dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    model.eval()
    set_model_phase(model, phase_info)
    phase = int(phase_info["phase"])
    seen_classes = [int(c) for c in phase_info["seen_classes"]]
    old_classes = [int(c) for c in phase_info.get("old_classes", [])]
    new_classes = [int(c) for c in phase_info.get("new_classes", [])]
    loader = dataset.get_cumulative_dataloader(phase, split="test", batch_size=batch_size, shuffle=False)
    preds: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    energies: List[torch.Tensor] = []
    pred_hist: Dict[int, int] = {}
    spectral_diag: Optional[Dict[str, Any]] = None
    for batch in loader:
        patches, labels, spectra, _ = unpack_eval_batch(batch)
        patches = patches.to(device, non_blocking=True).float()
        if torch.is_tensor(spectra):
            spectra = spectra.to(device, non_blocking=True)
        labels_t = labels.to(device).long().view(-1) if torch.is_tensor(labels) else torch.as_tensor(labels, device=device).long().view(-1)
        bad = sorted(set(labels_t.detach().cpu().tolist()) - set(seen_classes))
        if bad:
            raise RuntimeError(f"Evaluation labels outside seen classes at phase {phase}: {bad}")
        out = forward_eval_batch(
            model,
            patches,
            spectra,
            args,
            seen_classes,
            old_classes=old_classes,
            new_classes=new_classes,
        )
        pred_global = logits_to_global_predictions(out["logits"], seen_classes)
        energy = out["energy"].detach().float().cpu()
        for p in pred_global.detach().cpu().tolist():
            pred_hist[int(p)] = pred_hist.get(int(p), 0) + 1
        preds.append(pred_global.detach().cpu().numpy())
        labels_all.append(labels_t.detach().cpu().numpy())
        energies.append(energy)
        spectral_diag = out.get("spectral_diagnostics", spectral_diag)
    if not preds:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {
            "prediction_histogram": {},
            "energy": torch.empty((0, len(seen_classes))),
        }
    energy_all = torch.cat(energies, dim=0)
    return np.concatenate(preds), np.concatenate(labels_all), {
        "prediction_histogram": pred_hist,
        "energy": energy_all,
        "spectral_diagnostics": spectral_diag or {},
    }

def compute_phase_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phase_info: Dict[str, Any],
    *,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Dict[str, Any]:
    if y_true.size == 0:
        return {
            "overall_accuracy": 0.0,
            "per_class_accuracy": {},
            "old_accuracy": 0.0,
            "new_accuracy": 0.0,
            "harmonic_mean": 0.0,
            "hm": 0.0,
        }
    metrics = calculate_metrics_torch(
        y_true=y_true,
        y_pred=y_pred,
        seen_classes=[int(c) for c in phase_info["seen_classes"]],
        old_classes=[int(c) for c in phase_info.get("old_classes", [])],
        new_classes=[int(c) for c in phase_info.get("new_classes", [])],
        old_class_count=len(phase_info.get("old_classes", [])),
        energy=energy,
        energy_margin=float(energy_margin),
        device="cpu",
    )
    metrics["hm"] = float(metrics.get("harmonic_mean", metrics.get("hm", 0.0)))
    return metrics

def evaluator_update_compat(
    evaluator: Any,
    phase: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phase_info: Dict[str, Any],
    *,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> None:
    sig = inspect.signature(evaluator.update)
    candidates = {
        "old_class_count": len(phase_info.get("old_classes", [])),
        "seen_classes": [int(c) for c in phase_info.get("seen_classes", [])],
        "old_classes": [int(c) for c in phase_info.get("old_classes", [])],
        "new_classes": [int(c) for c in phase_info.get("new_classes", [])],
        "energy": energy,
        "energy_margin": float(energy_margin),
    }
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    evaluator.update(int(phase), y_true, y_pred, **kwargs)

def save_classification_report_compat(
    evaluator: Any,
    phase: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    phase_dir: str,
    phase_info: Dict[str, Any],
    enabled: bool,
    tr_time: float,
    te_time: float,
    *,
    energy: Optional[torch.Tensor] = None,
    energy_margin: float = 0.0,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    os.makedirs(phase_dir, exist_ok=True)
    common = {
        "phase": int(phase),
        "y_true": y_true,
        "y_pred": y_pred,
        "target_names": target_names,
        "save_dir": phase_dir,
        "seen_classes": [int(c) for c in phase_info.get("seen_classes", [])],
        "old_class_count": len(phase_info.get("old_classes", [])),
        "old_classes": [int(c) for c in phase_info.get("old_classes", [])],
        "new_classes": [int(c) for c in phase_info.get("new_classes", [])],
        "energy": energy,
        "energy_margin": float(energy_margin),
        "tr_time": tr_time,
        "te_time": te_time,
        "dl_time": 0.0,
    }
    if hasattr(evaluator, "save_phase_report"):
        sig = inspect.signature(evaluator.save_phase_report)
        return evaluator.save_phase_report(**{k: v for k, v in common.items() if k in sig.parameters})
    return save_classification_report(
        **common,
        save_hsi_style=True,
        save_structured=True,
    )

# -----------------------------------------------------------------------------
# Checkpoint, artifacts, phase loop
# -----------------------------------------------------------------------------


def geometry_bank_state(model: torch.nn.Module) -> Any:
    gb = getattr(model, "geometry_bank", None)
    if gb is None:
        return None
    if hasattr(gb, "state_dict"):
        try:
            return gb.state_dict()
        except Exception:
            pass
    if hasattr(model, "export_memory_snapshot"):
        return model.export_memory_snapshot()
    if hasattr(gb, "export_state"):
        return gb.export_state()
    return None


def runtime_contract(args: argparse.Namespace, phase_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    phase = int(phase_info["phase"]) if phase_info else None
    return {
        "method": build_method_identity(args),
        "phase": phase,
        "feature_space": "single frozen canonical projected-z space",
        "classifier_mode": args.incremental_classifier_mode if phase and phase > 0 else args.base_classifier_mode,
        "eval_mode": args.eval_classifier_mode,
        "trainable_policy": "base_backbone_projection_ce_head" if phase in {None, 0} else "temporary_new_descriptor_tensors_only",
        "strict_non_exemplar": bool(args.strict_non_exemplar),
        "raw_exemplars_stored": False,
        "old_features_stored": False,
        "raw_old_spectra_stored": False,
        "aggregate_spectral_geometry_stored": True,
        "spectral_to_feature_coupling_stored": True,
        "kd_teacher_used": False,
        "projection_trainable_during_incremental": False,
        "classifier_trainable_during_incremental": False,
        "old_geometry_bank_frozen": True if phase and phase > 0 else None,
        "replay": "spectral-consistent core plus risk-directed tangent replay",
        "replay_energy_filter": bool(args.replay_energy_filter),
        "label_convention": "dataset/global, logits/seen-local, CE/seen-local, predictions/global, explicit old/new lists",
        "transport_enabled": False,
        "feature_adapter_enabled": False,
        "energy_calibrator_enabled": False,
        "adaptive_boundary_enabled": False,
    }

def checkpoint_payload(
    model: torch.nn.Module,
    args: argparse.Namespace,
    phase_info: Dict[str, Any],
    history: Any,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
    method_identity: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "phase": int(phase_info["phase"]),
        "model_state": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "geometry_bank_state": geometry_bank_state(model),
        "memory_snapshot": model.export_memory_snapshot() if hasattr(model, "export_memory_snapshot") else None,
        "seen_classes": [int(c) for c in phase_info["seen_classes"]],
        "old_classes": [int(c) for c in phase_info["old_classes"]],
        "new_classes": [int(c) for c in phase_info["new_classes"]],
        "class_mappings": {
            "seen_global_to_local": {str(c): i for i, c in enumerate(phase_info["seen_classes"])},
            "seen_local_to_global": {str(i): int(c) for i, c in enumerate(phase_info["seen_classes"])},
        },
        "base_geometry_certificate": getattr(model, "base_geometry_certificate", None),
        "base_handoff": getattr(model, "base_handoff", None),
        "runtime_contract": runtime_contract(args, phase_info),
        "method_identity": method_identity,
        "args_resolved": namespace_to_dict(args),
        "metrics": metrics,
        "history": history,
        "diagnostics": diagnostics,
    }
    required = ["phase", "model_state", "geometry_bank_state", "seen_classes", "class_mappings", "runtime_contract", "args_resolved", "metrics"]
    missing = [k for k in required if k not in payload or payload[k] is None]
    if missing:
        raise RuntimeError(f"Checkpoint missing critical fields: {missing}")
    return payload


def save_phase_artifacts(
    phase_dir: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
    phase_info: Dict[str, Any],
    history: Any,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
    method_identity: Dict[str, Any],
    classification_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    os.makedirs(phase_dir, exist_ok=True)
    paths = {
        "metrics": save_json(os.path.join(phase_dir, "metrics.json"), metrics),
        "diagnostics": save_json(os.path.join(phase_dir, "diagnostics.json"), diagnostics),
        "runtime_contract": save_json(os.path.join(phase_dir, "runtime_contract.json"), runtime_contract(args, phase_info)),
    }
    if classification_report is not None:
        paths["classification_report_info"] = save_json(os.path.join(phase_dir, "classification_report_info.json"), classification_report)
    if int(phase_info["phase"]) == 0:
        cert = getattr(model, "base_geometry_certificate", None)
        if cert is not None:
            paths["geometry_certificate"] = save_json(os.path.join(phase_dir, "geometry_certificate.json"), cert)
    cm = metrics.get("confusion_matrix", None)
    if torch.is_tensor(cm):
        cm = cm.detach().cpu().numpy()
    if isinstance(cm, np.ndarray):
        cm_path = os.path.join(phase_dir, "confusion_matrix.npy")
        np.save(cm_path, cm)
        paths["confusion_matrix"] = cm_path
    ckpt = checkpoint_payload(model, args, phase_info, history, metrics, diagnostics, method_identity)
    ckpt_path = os.path.join(phase_dir, f"phase_{int(phase_info['phase'])}_checkpoint.pt")
    torch.save(ckpt, ckpt_path)
    paths["checkpoint"] = ckpt_path
    return paths


def collect_trainer_diagnostics(trainer: Any, phase: int, phase_dir: str) -> Dict[str, Any]:
    candidates = [
        f"_last_phase_{int(phase)}_geometry_diagnostics",
        "_last_incremental_geometry_diagnostics",
        "_last_phase_geometry_diagnostics",
        "_last_geometry_diagnostics",
    ]
    if int(phase) == 0:
        candidates.insert(0, "_last_base_geometry_diagnostics")
    for attr in candidates:
        diag = getattr(trainer, attr, None)
        if isinstance(diag, dict) and diag:
            return _json_safe(diag)
    json_path = os.path.join(phase_dir, f"phase_{int(phase)}_geometry_diagnostics.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def call_phase_map_compat(model: torch.nn.Module, dataset: Any, phase: int, target_names: List[str], phase_dir: str, args: argparse.Namespace) -> None:
    if bool(args.skip_phase_maps):
        return
    try:
        sig = inspect.signature(predict_phase_grid)
        kwargs = {
            "model": model,
            "dataset_manager": dataset,
            "phase": int(phase),
            "target_names": target_names,
            "save_dir": os.path.join(phase_dir, "maps"),
            "device": args.device,
            "patch_size": args.patch_size,
            "classifier_mode": args.eval_classifier_mode,
            "semantic_mode": "identity",
            "class_cmap": args.viz_class_cmap,
            "background_color": args.viz_background_color,
            "save_numpy": args.viz_save_numpy,
        }
        predict_phase_grid(**{k: v for k, v in kwargs.items() if k in sig.parameters})
    except Exception as exc:
        print(f"[WARN] phase map generation failed: {exc}")


def assert_base_handoff_ready(model: torch.nn.Module, trainer: Any) -> None:
    """Verify phase-0 compact geometry rows without a hard conflict certificate.

    This is intentionally not a semantic handoff gate.  HSI classes can be
    physically similar, so overlap/risk diagnostics must not stop the run.  The
    only hard requirements here are: base rows exist, are finite, and can be
    frozen for future replay/adaptation.
    """
    base_ids = [int(c) for c in phase_to_classes_as_list(trainer.dataset)[0]]
    bank = getattr(model, "geometry_bank", None)
    if bank is None:
        raise RuntimeError("Base phase completed without GeometryBank.")

    if hasattr(bank, "phase_geometry_state_report"):
        report = bank.phase_geometry_state_report(base_ids, freeze=True)
        if isinstance(report, dict) and not bool(report.get("ok", True)):
            raise RuntimeError(f"Base GeometryBank rows are not ready: {report}")
    elif hasattr(bank, "assert_bank_valid"):
        try:
            bank.assert_bank_valid(seen_classes=base_ids, strict=True)
        except TypeError:
            bank.assert_bank_valid(seen_classes=base_ids)
        if hasattr(bank, "freeze_classes"):
            bank.freeze_classes(base_ids)
        elif hasattr(bank, "freeze_classes_up_to") and base_ids == list(range(max(base_ids) + 1)):
            bank.freeze_classes_up_to(max(base_ids) + 1)
    elif hasattr(model, "assert_base_handoff_ready"):
        # Compatibility only.  Prefer non-strict mode because old implementations
        # sometimes used band/spectral overlap as a hard failure.
        try:
            model.assert_base_handoff_ready(base_ids, freeze=True, strict=False)
        except TypeError:
            model.assert_base_handoff_ready()
    else:
        raise RuntimeError("No GeometryBank validation API available after base phase.")


def assert_incremental_phase_complete(model: torch.nn.Module, phase_info: Dict[str, Any]) -> None:
    if geometry_bank_state(model) is None:
        raise RuntimeError("Incremental phase ended without GeometryBank state.")
    if int(phase_info["phase"]) > 0 and not phase_info.get("old_classes"):
        raise RuntimeError("Incremental phase missing old classes in phase_info.")
    gb = getattr(model, "geometry_bank", None)
    if gb is not None and hasattr(gb, "assert_bank_valid"):
        try:
            gb.assert_bank_valid(seen_classes=[int(c) for c in phase_info["seen_classes"]], strict=True)
        except TypeError:
            gb.assert_bank_valid(seen_classes=[int(c) for c in phase_info["seen_classes"]])


def run_phase(
    *,
    trainer: Trainer,
    model: torch.nn.Module,
    dataset: Any,
    evaluator: Any,
    args: argparse.Namespace,
    phase_info: Dict[str, Any],
    target_names: List[str],
    run_dir: str,
    method_identity: Dict[str, Any],
) -> Dict[str, Any]:
    phase = int(phase_info["phase"])
    phase_dir = os.path.join(run_dir, f"phase_{phase}")
    os.makedirs(phase_dir, exist_ok=True)
    if hasattr(dataset, "start_phase"):
        dataset.start_phase(phase)
    set_model_phase(model, phase_info)
    print("\n" + "=" * 88)
    print(f"[Phase {phase}] old={phase_info['old_classes']} | new={phase_info['new_classes']} | seen={phase_info['seen_classes']}")
    print("=" * 88)

    epochs = int(args.epochs_base if phase == 0 else args.epochs_inc)
    lr = float(args.lr if phase == 0 else (args.lr_inc if args.lr_inc > 0 else args.lr))

    if phase > 0 and hasattr(trainer, "_assert_incremental_preflight"):
        trainer._assert_incremental_preflight(
            phase,
            old_classes=phase_info["old_classes"],
            new_classes=phase_info["new_classes"],
            seen_classes=phase_info["seen_classes"],
        )
        print(f"[Incremental Preflight PASS] phase={phase}")

    t0 = time.time()
    # Always use Trainer.train_phase() so phase-specific trainability, cleaned
    # argument normalization, old/new class resolution, and incremental preflight
    # are applied consistently.  Calling train_base_phase()/train_incremental_phase()
    # directly bypasses that contract.
    history = trainer.train_phase(phase=phase, epochs=epochs, batch_size=args.batch_size, lr=lr)
    if phase == 0:
        assert_base_handoff_ready(model, trainer)
    else:
        assert_incremental_phase_complete(model, phase_info)
    train_time = time.time() - t0

    print(f"[Eval] phase={phase} cumulative seen-class evaluation")
    e0 = time.time()
    y_pred, y_true, pred_diag = get_phase_predictions(model, dataset, phase_info, torch.device(args.device), args, batch_size=args.batch_size)
    eval_time = time.time() - e0
    energy = pred_diag.get("energy", None)

    metrics = compute_phase_metrics(
        y_true,
        y_pred,
        phase_info,
        energy=energy,
        energy_margin=float(args.geometry_energy_margin if phase > 0 else args.base_energy_margin),
    )
    metrics["train_time_sec"] = train_time
    metrics["eval_time_sec"] = eval_time
    metrics["prediction_histogram"] = pred_diag.get("prediction_histogram", {})
    evaluator_update_compat(
        evaluator,
        phase,
        y_true,
        y_pred,
        phase_info,
        energy=energy,
        energy_margin=float(args.geometry_energy_margin if phase > 0 else args.base_energy_margin),
    )
    if hasattr(evaluator, "print_summary"):
        evaluator.print_summary()
    report = save_classification_report_compat(
        evaluator=evaluator,
        phase=phase,
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        phase_dir=phase_dir,
        phase_info=phase_info,
        enabled=bool(args.save_classification_report),
        tr_time=train_time,
        te_time=eval_time,
        energy=energy,
        energy_margin=float(args.geometry_energy_margin if phase > 0 else args.base_energy_margin),
    )
    diagnostics = collect_trainer_diagnostics(trainer, phase, phase_dir)
    diagnostics["prediction_histogram"] = pred_diag.get("prediction_histogram", {})
    diagnostics["evaluation_spectral_metadata"] = pred_diag.get("spectral_diagnostics", {})
    diagnostics["phase_info"] = phase_info
    paths = save_phase_artifacts(phase_dir, model, args, phase_info, history, metrics, diagnostics, method_identity, classification_report=report)
    call_phase_map_compat(model, dataset, phase, target_names, phase_dir, args)
    return {
        "phase": phase,
        "phase_dir": phase_dir,
        "phase_info": phase_info,
        "history": history,
        "metrics": _json_safe(metrics),
        "diagnostics": diagnostics,
        "artifact_paths": paths,
        "classification_report": report,
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
    }


def determine_total_phases(dataset: Any, args: argparse.Namespace) -> int:
    phases = phase_to_classes_as_list(dataset)
    dataset_total = int(getattr(dataset, "num_phases", len(phases)))
    if dataset_total != len(phases):
        print(f"[WARN] dataset.num_phases={dataset_total} but len(phase_to_classes)={len(phases)}. Using phase_to_classes.")
        dataset_total = len(phases)
    if bool(args.base_only):
        return 1
    total = dataset_total
    if int(args.max_phases or 0) > 0:
        total = min(total, int(args.max_phases))
    elif int(args.max_train_phase or -1) >= 0:
        total = min(total, int(args.max_train_phase) + 1)
    return max(1, total)


def save_dataset_protocol_files(run_dir: str, data: Dict[str, Any], dataset: Any) -> Dict[str, str]:
    phases = phase_to_classes_as_list(dataset)
    phase_splits = {
        "num_phases": int(len(phases)),
        "class_order": _json_safe(getattr(dataset, "class_order", None)),
        "phase_to_classes": _json_safe(phases),
    }
    return {
        "dataset_summary": save_json(os.path.join(run_dir, "dataset_summary.json"), data["summary"]),
        "phase_splits": save_json(os.path.join(run_dir, "phase_splits.json"), phase_splits),
    }


def run_single_experiment(base_args: argparse.Namespace, run_idx: int, run_seed: int) -> Dict[str, Any]:
    raw_args = argparse.Namespace(**namespace_to_dict(base_args))
    raw_args.seed = int(run_seed)
    original = namespace_to_dict(raw_args)
    resolved, diff, method_identity = resolve_experiment_config(raw_args)
    validate_config(resolved)
    assert_strict_energy_contract(resolved, context="run_single_experiment(resolved)")
    set_seed(resolved.seed, deterministic=bool(resolved.deterministic))

    run_tag = "base_only" if bool(resolved.base_only) else "sctgr"
    run_dir = os.path.join(
        resolved.save_dir,
        resolved.dataset,
        f"patch_{resolved.patch_size}",
        f"{run_tag}_run_{run_idx + 1}_seed_{resolved.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)
    resolved.run_dir = run_dir
    resolved.save_dir = run_dir

    save_config_files(run_dir, original, resolved, diff, method_identity)
    print("\n=== NECIL-HSI RUN ===")
    print(f"[Build] {STACK_BUILD_ID}")
    print(f"Run {run_idx + 1}/{base_args.num_runs} | seed={resolved.seed} | device={resolved.device}")
    print_config_summary(resolved, diff, method_identity)

    data = load_hsi_dataset(resolved)
    dataset = build_incremental_dataset(resolved, data)
    target_names = resolve_target_names(dataset, data["target_names"])
    if hasattr(dataset, "target_names"):
        dataset.target_names = target_names
    save_dataset_protocol_files(run_dir, data, dataset)

    device = torch.device(resolved.device)
    model = build_model(resolved, device)
    trainer = build_trainer(model, dataset, resolved, run_dir)
    evaluator = build_evaluator()

    phase_results: Dict[int, Dict[str, Any]] = {}
    total_phases = determine_total_phases(dataset, resolved)
    print(f"[Run] phases=0..{total_phases - 1} of dataset phases={getattr(dataset, 'num_phases', total_phases)}")
    start = time.time()
    for phase in range(total_phases):
        phase_info = get_phase_info(dataset, phase)
        phase_results[phase] = run_phase(
            trainer=trainer,
            model=model,
            dataset=dataset,
            evaluator=evaluator,
            args=resolved,
            phase_info=phase_info,
            target_names=target_names,
            run_dir=run_dir,
            method_identity=method_identity,
        )

    elapsed = time.time() - start
    final_phase = max(phase_results)
    final_metrics = phase_results[final_phase]["metrics"]
    final_phase_info = get_phase_info(dataset, final_phase)
    final_results = {
        "run_idx": run_idx,
        "seed": int(resolved.seed),
        "run_dir": run_dir,
        "elapsed_sec": elapsed,
        "final_phase": int(final_phase),
        "final_metrics": final_metrics,
        "phase_results": _json_safe(phase_results),
        "method_identity": method_identity,
        "runtime_contract": runtime_contract(resolved, final_phase_info),
        "evaluator": evaluator.to_dict() if hasattr(evaluator, "to_dict") else None,
    }
    save_json(os.path.join(run_dir, "final_results.json"), final_results)
    torch.save(
        checkpoint_payload(model, resolved, final_phase_info, None, final_metrics, {}, method_identity),
        os.path.join(run_dir, "final_model.pt"),
    )
    try:
        history = {}
        for pr in phase_results.values():
            h = pr.get("history", {})
            if isinstance(h, dict):
                for k, v in h.items():
                    if isinstance(v, list):
                        history.setdefault(k, []).extend(v)
        if history:
            plot_training_history(history, os.path.join(run_dir, "training_history.png"))
    except Exception as exc:
        print(f"[WARN] Could not plot training history: {exc}")
    write_run_report(os.path.join(run_dir, "SCTGR_HSI_RUN_REPORT.txt"), resolved, final_results)
    print(f"[Done] run_dir={run_dir} | final_OA={float(final_metrics.get('overall_accuracy', 0.0)):.2f} | final_HM={float(final_metrics.get('hm', 0.0)):.2f}")
    return final_results


def write_run_report(path: str, args: argparse.Namespace, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SCTGR-HSI Run Report - {args.dataset}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(json.dumps(_json_safe(result["method_identity"]), indent=2) + "\n\n")
        f.write("Phase metrics\n")
        f.write("-" * 80 + "\n")
        for p, pr in sorted((result.get("phase_results", {}) or {}).items(), key=lambda kv: int(kv[0])):
            m = pr.get("metrics", {}) or {}
            info = pr.get("phase_info", {}) or {}
            f.write(
                f"Phase {p}: OA={float(m.get('overall_accuracy', 0.0)):.2f} | "
                f"Old={float(m.get('old_accuracy', 0.0)):.2f} | New={float(m.get('new_accuracy', 0.0)):.2f} | "
                f"HM={float(m.get('hm', 0.0)):.2f} | seen={info.get('seen_classes', [])}\n"
            )
        f.write("\nFinal metrics\n")
        f.write(json.dumps(_json_safe(result.get("final_metrics", {})), indent=2) + "\n")
    print(f"[Report] {path}")


def aggregate_runs(results: List[Dict[str, Any]], root_dir: str) -> Dict[str, Any]:
    os.makedirs(root_dir, exist_ok=True)
    rows = []
    for r in results:
        m = r.get("final_metrics", {}) or {}
        rows.append({
            "run_idx": int(r.get("run_idx", 0)),
            "seed": int(r.get("seed", 0)),
            "run_dir": r.get("run_dir", ""),
            "final_phase": int(r.get("final_phase", 0)),
            "overall_accuracy": float(m.get("overall_accuracy", 0.0)),
            "old_accuracy": float(m.get("old_accuracy", 0.0)),
            "new_accuracy": float(m.get("new_accuracy", 0.0)),
            "hm": float(m.get("hm", 0.0)),
        })

    def mean_std(key: str) -> Tuple[float, float]:
        arr = np.asarray([row[key] for row in rows], dtype=np.float64)
        return (float(arr.mean()), float(arr.std(ddof=0))) if arr.size else (0.0, 0.0)

    summary = {
        "num_runs": len(results),
        "rows": rows,
        "overall_accuracy_mean_std": mean_std("overall_accuracy"),
        "old_accuracy_mean_std": mean_std("old_accuracy"),
        "new_accuracy_mean_std": mean_std("new_accuracy"),
        "hm_mean_std": mean_std("hm"),
    }
    save_json(os.path.join(root_dir, "runs_summary.json"), summary)
    csv_path = os.path.join(root_dir, "runs_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["run_idx", "seed", "run_dir", "overall_accuracy", "old_accuracy", "new_accuracy", "hm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary["runs_summary_csv"] = csv_path
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    preview_args = argparse.Namespace(**namespace_to_dict(args))
    resolved, _, _ = resolve_experiment_config(preview_args)
    seeds = parse_seed_list(getattr(resolved, "seed_list", "")) or [int(resolved.seed) + i for i in range(int(resolved.num_runs))]
    if len(seeds) != int(resolved.num_runs):
        raise ValueError("seed_list length must match num_runs.")

    results: List[Dict[str, Any]] = []
    for run_idx, seed in enumerate(seeds):
        run_args = argparse.Namespace(**namespace_to_dict(args))
        run_args.seed = int(seed)
        results.append(run_single_experiment(run_args, run_idx, int(seed)))

    root_dir = os.path.join(resolved.save_dir, resolved.dataset, f"patch_{resolved.patch_size}")
    summary = aggregate_runs(results, root_dir)
    print("\n=== MULTI-RUN SUMMARY ===")
    print(f"runs={summary['num_runs']} | OA={summary['overall_accuracy_mean_std'][0]:.2f}±{summary['overall_accuracy_mean_std'][1]:.2f} | HM={summary['hm_mean_std'][0]:.2f}±{summary['hm_mean_std'][1]:.2f}")
    print(f"Saved: {os.path.join(root_dir, 'runs_summary.json')}")
    print(f"Saved: {summary['runs_summary_csv']}")


if __name__ == "__main__":
    main()

















# from __future__ import annotations

# import argparse
# import copy
# import csv
# import inspect
# import json
# import os
# import random
# import sys
# import time
# from datetime import datetime
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import numpy as np
# import torch

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from data.hsi_dataloader_pytorch import ImageCubes, LoadHSIData
# from data.incremental_dataset import IncrementalHSIDataset
# from models.necil_model import NECILModel
# from trainers.trainer import Trainer
# from utils.eval import NECILEvaluator, make_json_serializable, save_classification_report
# from utils.visualize import plot_training_history, predict_phase_grid


# DATASET_INFO = {
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


# # -----------------------------------------------------------------------------
# # Robust utilities
# # -----------------------------------------------------------------------------


# def str2bool(v: Any) -> bool:
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return False
#     s = str(v).strip().lower()
#     if s in {"true", "1", "yes", "y", "t", "on"}:
#         return True
#     if s in {"false", "0", "no", "n", "f", "off", "none", "null", ""}:
#         return False
#     raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


# def parse_seed_list(seed_list_str: Optional[str]) -> Optional[List[int]]:
#     if seed_list_str is None or str(seed_list_str).strip() == "":
#         return None
#     return [int(s.strip()) for s in str(seed_list_str).split(",") if s.strip()]


# def _json_safe(obj: Any) -> Any:
#     try:
#         return make_json_serializable(obj)
#     except Exception:
#         pass
#     if torch.is_tensor(obj):
#         x = obj.detach().cpu()
#         return x.item() if x.numel() == 1 else x.tolist()
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (np.integer, np.floating)):
#         return obj.item()
#     if isinstance(obj, dict):
#         return {str(k): _json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_json_safe(v) for v in obj]
#     if isinstance(obj, (str, int, float, bool)) or obj is None:
#         return obj
#     return str(obj)


# def save_json(path: str, data: Any) -> str:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(_json_safe(data), f, indent=2, sort_keys=True)
#     return path


# def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
#     return copy.deepcopy(vars(args))


# def _set_resolved(args: argparse.Namespace, name: str, value: Any, reasons: Dict[str, str], reason: str) -> None:
#     old = getattr(args, name, None)
#     if old != value or not hasattr(args, name):
#         setattr(args, name, value)
#         reasons[name] = reason


# def compute_config_diff(
#     original: Dict[str, Any],
#     resolved: Dict[str, Any],
#     reasons: Optional[Dict[str, str]] = None,
# ) -> List[Dict[str, Any]]:
#     reasons = reasons or {}
#     rows: List[Dict[str, Any]] = []
#     for k in sorted(set(original.keys()) | set(resolved.keys())):
#         if original.get(k) != resolved.get(k):
#             rows.append({
#                 "name": k,
#                 "original": _json_safe(original.get(k)),
#                 "resolved": _json_safe(resolved.get(k)),
#                 "reason": reasons.get(k, "resolved_config"),
#             })
#     return rows


# def normalize_classifier_mode(mode: Optional[str]) -> str:
#     """Normalize every public geometry alias to the strict classifier token.

#     The repaired classifier/model/trainer use ``geometry_only`` as the explicit
#     method-contract string. ``geometry`` is accepted from older commands, but it
#     is semantically identical and must be normalized before the trainer identity
#     check. Otherwise main.py incorrectly reports a method-identity mutation even
#     though the classifier path did not change.
#     """
#     m = str(mode or "geometry_only").lower().strip()
#     aliases = {
#         "": "geometry_only",
#         "none": "geometry_only",
#         "geo": "geometry_only",
#         "geometry": "geometry_only",
#         "geometry_only": "geometry_only",
#         "geometry-only": "geometry_only",
#         "feature_geometry": "geometry_only",
#         "low_rank_geometry": "geometry_only",
#         "srgp": "geometry_only",
#         "srgp_geometry": "geometry_only",
#         "spectral_geometry": "geometry_only",
#         "spectral_residual_geometry": "geometry_only",
#         "calibrated": "geometry_only",
#         "calibrated_geometry": "geometry_only",
#     }
#     m = aliases.get(m, m)
#     if m != "geometry_only":
#         raise ValueError(f"Unsupported classifier mode {mode!r}. Use geometry_only/geometry.")
#     return m


# def normalize_incremental_update_mode(mode: Optional[str]) -> str:
#     """PG-RGA is the main method.

#     Old names are accepted only so old commands do not break. They all resolve to
#     geometry_gated_adapter because PG-RGA uses the bounded geometry residual
#     adapter as the only incremental model plasticity.
#     """
#     m = str(mode or "geometry_gated_adapter").lower().strip()
#     aliases = {
#         "": "geometry_gated_adapter",
#         "none": "geometry_gated_adapter",
#         "clean": "geometry_gated_adapter",
#         "main": "geometry_gated_adapter",
#         "pg_rga": "geometry_gated_adapter",
#         "pg-rga": "geometry_gated_adapter",
#         "pgrga": "geometry_gated_adapter",
#         "geometry_gated_adapter": "geometry_gated_adapter",
#         "g2rpa": "geometry_gated_adapter",
#         "g2-rpa": "geometry_gated_adapter",
#         "g²rpa": "geometry_gated_adapter",
#         "gated_adapter": "geometry_gated_adapter",
#         "geometry_adapter": "geometry_gated_adapter",
#         "adapter": "geometry_gated_adapter",
#         # legacy aliases from earlier drafts
#         "scbgr": "geometry_gated_adapter",
#         "scb-gr": "geometry_gated_adapter",
#         "descriptor": "geometry_gated_adapter",
#         "descriptor_only": "geometry_gated_adapter",
#         "rsgi": "geometry_gated_adapter",
#         "geometry_state_admission": "geometry_gated_adapter",
#         "spectral_risk_boundary": "geometry_gated_adapter",
#         "boundary_geometry": "geometry_gated_adapter",
#     }
#     if m not in aliases:
#         raise ValueError("Unsupported --incremental_update_mode. Use geometry_gated_adapter / pg_rga.")
#     return aliases[m]


# # -----------------------------------------------------------------------------
# # Parser and resolved configuration
# # -----------------------------------------------------------------------------


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description="PG-RGA-HSI: low-rank GeometryBank descriptor-based exemplar-free HSI class-incremental classification",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     main = parser.add_argument_group("A. Core experiment")
#     main.add_argument("--dataset", type=str, default="IP", choices=DATASET_INFO.keys())
#     main.add_argument("--data_dir", type=str, default="./datasets")
#     main.add_argument("--save_dir", type=str, default="./results_pg_rga")
#     main.add_argument("--patch_size", type=int, default=11)
#     main.add_argument("--train_ratio", type=float, default=0.2)
#     main.add_argument("--val_ratio", type=float, default=0.1)
#     main.add_argument("--min_train_per_class", type=int, default=20)
#     main.add_argument("--no_pca", action="store_true")
#     main.add_argument("--pca_components", type=int, default=30)
#     main.add_argument("--reduction_method", type=str, default="PCA")
#     main.add_argument("--base_classes", type=int, default=None)
#     main.add_argument("--increment", type=int, default=None)
#     main.add_argument("--epochs_base", type=int, default=80)
#     main.add_argument("--epochs_inc", type=int, default=30)
#     main.add_argument("--batch_size", type=int, default=64)
#     main.add_argument("--lr", type=float, default=1e-4)
#     main.add_argument("--lr_inc", type=float, default=1e-4)
#     main.add_argument("--weight_decay", type=float, default=1e-4)
#     main.add_argument("--seed", type=int, default=42)
#     main.add_argument("--num_runs", type=int, default=1)
#     main.add_argument("--seed_list", type=str, default="")
#     main.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     main.add_argument("--num_workers", type=int, default=0)
#     main.add_argument("--base_only", type=str2bool, default=False)
#     main.add_argument("--max_train_phase", type=int, default=-1)
#     main.add_argument("--max_phases", type=int, default=0)

#     model = parser.add_argument_group("B. Backbone and GeometryBank")
#     model.add_argument("--d_model", type=int, default=128)
#     model.add_argument("--d_state", type=int, default=16)
#     model.add_argument("--d_conv", type=int, default=4)
#     model.add_argument("--expand", type=int, default=2)
#     model.add_argument("--num_spectral_layers", type=int, default=3)
#     model.add_argument("--num_layers", type=int, default=3)
#     model.add_argument("--dropout", type=float, default=0.1)
#     model.add_argument("--projection_dropout", type=float, default=0.1)
#     model.add_argument("--backbone_norm", type=str, default="layer", choices=["layer", "rms"])
#     model.add_argument("--stem_norm_groups", type=int, default=8)
#     model.add_argument("--ssm_residual_scale_init", type=float, default=0.7)
#     model.add_argument("--fusion_residual_scale", type=float, default=0.3)
#     model.add_argument("--backbone_output_dropout", type=float, default=0.0)
#     model.add_argument("--subspace_rank", type=int, default=5)
#     model.add_argument("--geom_var_floor", type=float, default=5e-4)
#     model.add_argument("--geometry_variance_shrinkage", type=float, default=0.25)
#     model.add_argument("--geometry_max_variance_ratio", type=float, default=50.0)
#     model.add_argument("--geometry_min_reliability", type=float, default=0.05)
#     model.add_argument("--rank_energy_threshold", type=float, default=0.90)
#     model.add_argument("--rank_eigen_ratio_threshold", type=float, default=1e-2)
#     model.add_argument("--min_active_rank", type=int, default=1)
#     model.add_argument("--normalize_geometry_features", type=str2bool, default=True)
#     model.add_argument("--geometry_feature_scale", type=float, default=0.0)
#     model.add_argument("--geometry_feature_clamp", type=float, default=0.0)
#     model.add_argument("--subspace_extract_batch_size", type=int, default=256)

#     base = parser.add_argument_group("C. Mandatory base objective")
#     base.add_argument("--base_ce_weight", type=float, default=1.0)
#     base.add_argument("--base_srpgr_weight", type=float, default=1.0)
#     base.add_argument("--base_gics_weight", type=float, default=0.20)
#     base.add_argument("--base_gics_temperature", type=float, default=0.07)
#     base.add_argument("--base_class_balance", type=str2bool, default=True)
#     base.add_argument("--base_gics_key_noise_std", type=float, default=0.0)
#     base.add_argument("--base_gics_key_scale_jitter", type=float, default=0.0)
#     base.add_argument("--base_gics_key_band_drop", type=float, default=0.0)
#     base.add_argument("--base_gics_key_spatial_drop", type=float, default=0.0)
#     base.add_argument("--pgr_weight", type=float, default=0.10)
#     base.add_argument("--pgr_compact_weight", type=float, default=0.15)
#     base.add_argument("--pgr_center_weight", type=float, default=0.25)
#     base.add_argument("--pgr_subspace_weight", type=float, default=0.15)
#     base.add_argument("--pgr_band_weight", type=float, default=0.05)
#     base.add_argument("--pgr_volume_weight", type=float, default=0.05)
#     base.add_argument("--pgr_center_margin", type=float, default=1.10)
#     base.add_argument("--pgr_band_overlap_max", type=float, default=0.65)
#     base.add_argument("--pgr_max_subspace_overlap", type=float, default=0.55)
#     base.add_argument("--pgr_min_class_variance", type=float, default=0.015)
#     base.add_argument("--pgr_max_class_variance", type=float, default=0.75)
#     base.add_argument("--pgr_min_class_samples", type=int, default=3)
#     base.add_argument("--pgr_subspace_min_samples", type=int, default=6)
#     base.add_argument("--pgr_subspace_rank", type=int, default=3)
#     base.add_argument("--base_spectral_shape_weight", type=float, default=0.05)
#     base.add_argument("--base_max_spectral_shape_similarity", type=float, default=0.75)
#     base.add_argument("--base_spectral_shape_risk_weight", type=float, default=1.0)
#     base.add_argument("--base_require_physical_spectral_shape", type=str2bool, default=False)
#     base.add_argument("--strict_base_component_coverage", type=str2bool, default=True)

#     inc = parser.add_argument_group("D. PG-RGA incremental objective")
#     inc.add_argument("--incremental_update_mode", type=str, default="geometry_gated_adapter")
#     inc.add_argument("--gfa_weight", type=float, default=1.0)
#     inc.add_argument("--gfa_samples_per_class", type=int, default=48)
#     inc.add_argument("--gfa_parallel_scale", type=float, default=1.0)
#     inc.add_argument("--gfa_residual_scale", type=float, default=0.25)
#     inc.add_argument("--joint_old_new_ce_weight", type=float, default=1.0)
#     inc.add_argument("--geometry_energy_margin_weight", type=float, default=0.30)
#     inc.add_argument("--geometry_energy_margin", type=float, default=0.30)
#     inc.add_argument("--old_new_invasion_weight", type=float, default=0.50)
#     inc.add_argument("--old_new_geometry_margin", type=float, default=0.35)
#     inc.add_argument("--refine_new_descriptors", type=str2bool, default=True)
#     inc.add_argument("--descriptor_refine_steps", type=int, default=20)
#     inc.add_argument("--descriptor_refine_steps_per_epoch", type=int, default=None)
#     inc.add_argument("--descriptor_refine_lr", type=float, default=1e-3)
#     inc.add_argument("--descriptor_trust_weight", type=float, default=0.8)
#     inc.add_argument("--descriptor_refine_max_mean_shift", type=float, default=0.30)
#     inc.add_argument("--descriptor_refine_max_logvar_shift", type=float, default=0.50)
#     inc.add_argument("--adapter_bottleneck", type=int, default=32)
#     inc.add_argument("--adapter_max_scale", type=float, default=0.35)
#     inc.add_argument("--adapter_dropout", type=float, default=0.0)
#     inc.add_argument("--adapter_gate_bias_init", type=float, default=-3.0)
#     inc.add_argument("--adapter_lr", type=float, default=5e-4)
#     inc.add_argument("--adapter_weight_decay", type=float, default=0.0)
#     inc.add_argument("--g2rpa_adapter_weight", type=float, default=1.0)
#     inc.add_argument("--adapter_old_delta_weight", type=float, default=1.0)
#     inc.add_argument("--adapter_old_gate_weight", type=float, default=0.75)
#     inc.add_argument("--adapter_old_energy_weight", type=float, default=0.25)
#     inc.add_argument("--adapter_old_margin_weight", type=float, default=0.25)
#     inc.add_argument("--adapter_delta_weight", type=float, default=0.10)
#     inc.add_argument("--adapter_new_gate_weight", type=float, default=0.05)
#     inc.add_argument("--adapter_new_gate_target", type=float, default=0.25)
#     inc.add_argument("--adapter_new_gate_max_target", type=float, default=0.75)

#     clf = parser.add_argument_group("E. Geometry classifier/evaluation")
#     clf.add_argument("--classifier_mode", type=str, default="geometry")
#     clf.add_argument("--base_classifier_mode", type=str, default=None)
#     clf.add_argument("--incremental_classifier_mode", type=str, default=None)
#     clf.add_argument("--eval_classifier_mode", type=str, default="geometry")
#     clf.add_argument("--logit_scale", type=float, default=8.0)
#     clf.add_argument("--loss_scale", type=float, default=None)
#     clf.add_argument("--residual_variance_scale", type=float, default=0.75)
#     clf.add_argument("--energy_normalize_by_dim", type=str2bool, default=True)
#     clf.add_argument("--use_logdet_energy", type=str2bool, default=True)
#     clf.add_argument("--logdet_energy_weight", type=float, default=0.05)
#     clf.add_argument("--use_reliability_penalty", type=str2bool, default=True)
#     clf.add_argument("--reliability_energy_weight", type=float, default=0.03)
#     clf.add_argument("--geometry_logit_clip", type=float, default=0.0)
#     clf.add_argument("--best_state_metric", type=str, default="hm")
#     clf.add_argument("--label_smoothing", type=float, default=0.0)
#     clf.add_argument("--ce_logit_clip", type=float, default=50.0)
#     clf.add_argument("--grad_clip_base", type=float, default=1.0)
#     clf.add_argument("--grad_clip_inc", type=float, default=0.5)

#     spec = parser.add_argument_group("F. HSI spectral metadata")
#     spec.add_argument("--spectral_summary_mode", type=str, default="center", choices=["center", "mean"])
#     spec.add_argument("--spectral_summary_is_physical", type=str2bool, default=False)
#     spec.add_argument("--raw_spectral_summary_is_physical", type=str2bool, default=True)
#     spec.add_argument("--external_spectra_are_physical", type=str2bool, default=True)
#     spec.add_argument("--allow_nonphysical_spectral_summary", type=str2bool, default=False)
#     spec.add_argument("--spectral_require_physical_summary", type=str2bool, default=True)
#     spec.add_argument("--use_spectral_geometry", type=str2bool, default=True)
#     spec.add_argument("--spectral_energy_weight", type=float, default=0.0)
#     spec.add_argument("--spectral_derivative_weight", type=float, default=0.50)
#     spec.add_argument("--spectral_second_derivative_weight", type=float, default=0.25)
#     spec.add_argument("--band_energy_weight", type=float, default=0.0)

#     safety = parser.add_argument_group("G. Safety, diagnostics, visualization")
#     safety.add_argument("--strict_non_exemplar", type=str2bool, default=True)
#     safety.add_argument("--strict_feature_contract", type=str2bool, default=True)
#     safety.add_argument("--strict_updated_stack", type=str2bool, default=True)
#     safety.add_argument("--freeze_projection_during_incremental", type=str2bool, default=True)
#     safety.add_argument("--allow_incremental_projection_training", type=str2bool, default=False)
#     safety.add_argument("--freeze_classifier_during_incremental", type=str2bool, default=True)
#     safety.add_argument("--save_geometry_diagnostics", type=str2bool, default=True)
#     safety.add_argument("--save_classification_report", type=str2bool, default=True)
#     safety.add_argument("--save_final_classification_report", type=str2bool, default=True)
#     safety.add_argument("--skip_phase_maps", type=str2bool, default=False)
#     safety.add_argument("--viz_class_cmap", type=str, default="nipy_spectral")
#     safety.add_argument("--viz_background_color", type=str, default="#20252B")
#     safety.add_argument("--viz_save_numpy", type=str2bool, default=True)
#     safety.add_argument("--deterministic", type=str2bool, default=False)
#     safety.add_argument("--debug_verbose", type=str2bool, default=False)
#     safety.add_argument("--refresh_before_validation", type=str2bool, default=True)
#     safety.add_argument("--validation_refresh_every", type=int, default=1)
#     safety.add_argument("--base_geometry_refresh_every", type=int, default=1)
#     safety.add_argument("--print_base_geometry_diagnostics", type=str2bool, default=True)
#     safety.add_argument("--geometry_diag_anchors_per_class", type=int, default=64)
#     safety.add_argument("--geometry_diag_topk_pairs", type=int, default=20)
#     safety.add_argument("--geometry_diag_topk_bands", type=int, default=5)
#     safety.add_argument("--base_cert_min_geom_acc", type=float, default=95.0)
#     safety.add_argument("--base_cert_min_reliability", type=float, default=0.15)
#     safety.add_argument("--base_cert_min_mean_reliability", type=float, default=0.35)
#     safety.add_argument("--base_cert_max_subspace_overlap", type=float, default=0.55)
#     safety.add_argument("--base_cert_subspace_warn_overlap", type=float, default=0.72)
#     safety.add_argument("--base_cert_max_geometry_conflict", type=float, default=1.35)
#     safety.add_argument("--base_cert_max_geometry_conflict_soft", type=float, default=1.40)
#     safety.add_argument("--base_cert_max_guided_geometry_conflict", type=float, default=0.18)
#     safety.add_argument("--base_cert_max_band_similarity", type=float, default=0.90)
#     safety.add_argument("--base_cert_max_spectral_shape_similarity", type=float, default=0.85)

#     legacy = parser.add_argument_group("H. Legacy flags accepted but disabled")
#     legacy.add_argument("--use_geometry_transport", type=str2bool, default=False)
#     legacy.add_argument("--use_sglat_transport", type=str2bool, default=False)
#     legacy.add_argument("--transport_mode", type=str, default="new_row_only")
#     legacy.add_argument("--allow_old_model_transport", type=str2bool, default=False)
#     legacy.add_argument("--allow_transport_without_adapter", type=str2bool, default=False)
#     legacy.add_argument("--use_energy_calibrator", type=str2bool, default=False)
#     legacy.add_argument("--energy_calibrator_type", type=str, default="none")
#     legacy.add_argument("--energy_calibration_weight", type=float, default=0.0)
#     legacy.add_argument("--use_adaptive_boundary", type=str2bool, default=False)
#     legacy.add_argument("--use_incremental_adapter", type=str2bool, default=False)
#     legacy.add_argument("--disable_incremental_adapter", type=str2bool, default=True)
#     legacy.add_argument("--use_geometry_calibrator", type=str2bool, default=False)
#     legacy.add_argument("--use_bicyc_geometry_cycle", type=str2bool, default=False)
#     legacy.add_argument("--bss_weight", type=float, default=0.0)
#     legacy.add_argument("--sym_bss_weight", type=float, default=0.0)
#     legacy.add_argument("--gdr_weight", type=float, default=0.0)
#     legacy.add_argument("--anchor_consistency_weight", type=float, default=0.0)
#     legacy.add_argument("--use_mssl_loss", type=str2bool, default=False)
#     legacy.add_argument("--unsafe_ablation_use_mssl_loss", type=str2bool, default=False)
#     legacy.add_argument("--mssl_weight", type=float, default=0.0)
#     legacy.add_argument("--mssl_inc_weight", type=float, default=0.0)
#     legacy.add_argument("--bank_refresh_every", type=int, default=0)
#     legacy.add_argument("--early_stop_patience", type=int, default=0)
#     legacy.add_argument("--base_early_stop_patience", type=int, default=0)
#     legacy.add_argument("--incremental_early_stop_patience", type=int, default=0)
#     legacy.add_argument("--eval_semantic_mode", type=str, default="identity")
#     legacy.add_argument("--use_pretrain_incremental_baseline", type=str2bool, default=False)
#     legacy.add_argument("--allow_unknown_legacy_args", type=str2bool, default=False)

#     return parser


# def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
#     parser = build_parser()
#     args, unknown = parser.parse_known_args(argv)
#     args._unknown_args = unknown
#     return args


# def resolve_experiment_config(args: argparse.Namespace) -> Tuple[argparse.Namespace, List[Dict[str, Any]], Dict[str, Any]]:
#     original = namespace_to_dict(args)
#     resolved = argparse.Namespace(**copy.deepcopy(original))
#     reasons: Dict[str, str] = {}

#     mode = normalize_incremental_update_mode(getattr(resolved, "incremental_update_mode", "geometry_gated_adapter"))
#     _set_resolved(resolved, "incremental_update_mode", mode, reasons, "pg_rga_method_identity")
#     _set_resolved(resolved, "use_geometry_gated_adapter", True, reasons, "pg_rga_adapter_required")

#     base_mode = normalize_classifier_mode(getattr(resolved, "base_classifier_mode", None) or "geometry")
#     inc_mode = normalize_classifier_mode(getattr(resolved, "incremental_classifier_mode", None) or "geometry")
#     eval_mode = normalize_classifier_mode(getattr(resolved, "eval_classifier_mode", "geometry"))
#     for key, value in {
#         "classifier_mode": eval_mode,
#         "base_classifier_mode": base_mode,
#         "incremental_classifier_mode": inc_mode,
#         "eval_classifier_mode": eval_mode,
#     }.items():
#         _set_resolved(resolved, key, value, reasons, "seen_local_geometry_classifier")

#     forced_true = {
#         "strict_non_exemplar": True,
#         "strict_feature_contract": True,
#         "strict_updated_stack": True,
#         "freeze_projection_during_incremental": True,
#         "freeze_classifier_during_incremental": True,
#         "disable_incremental_adapter": True,
#         "base_class_balance": True,
#         "strict_base_component_coverage": True,
#         "use_spectral_geometry": True,
#     }
#     for k, v in forced_true.items():
#         _set_resolved(resolved, k, v, reasons, "pg_rga_contract")

#     forced_false = {
#         "allow_incremental_projection_training": False,
#         "use_geometry_transport": False,
#         "use_sglat_transport": False,
#         "allow_old_model_transport": False,
#         "allow_transport_without_adapter": False,
#         "use_energy_calibrator": False,
#         "use_adaptive_boundary": False,
#         "use_incremental_adapter": False,
#         "use_geometry_calibrator": False,
#         "use_bicyc_geometry_cycle": False,
#         "use_mssl_loss": False,
#         "use_pretrain_incremental_baseline": False,
#         "geometry_normalize_logits": False,
#         "allow_nonphysical_spectral_summary": False,
#     }
#     for k, v in forced_false.items():
#         _set_resolved(resolved, k, v, reasons, "legacy_or_ablation_disabled")

#     forced_zero = [
#         "energy_calibration_weight",
#         "bss_weight",
#         "sym_bss_weight",
#         "gdr_weight",
#         "anchor_consistency_weight",
#         "mssl_weight",
#         "mssl_inc_weight",
#         "bank_refresh_every",
#         "early_stop_patience",
#         "base_early_stop_patience",
#         "incremental_early_stop_patience",
#         "spectral_energy_weight",
#         "band_energy_weight",
#     ]
#     for k in forced_zero:
#         _set_resolved(resolved, k, 0.0 if "weight" in k or "energy" in k else 0, reasons, "not_used_in_pg_rga_main_path")

#     # Required non-zero base components.
#     required_positive = {
#         "base_ce_weight": 1.0,
#         "base_srpgr_weight": 1.0,
#         "base_gics_weight": 0.20,
#         "pgr_weight": 0.10,
#         "pgr_compact_weight": 0.15,
#         "pgr_center_weight": 0.25,
#         "pgr_subspace_weight": 0.15,
#         "pgr_band_weight": 0.05,
#         "pgr_volume_weight": 0.05,
#         "pgr_max_subspace_overlap": 0.55,
#         "pgr_min_class_variance": 0.015,
#     }
#     for k, fallback in required_positive.items():
#         if float(getattr(resolved, k, fallback)) <= 0.0:
#             _set_resolved(resolved, k, fallback, reasons, "mandatory_base_component")

#     pca_active = (not bool(getattr(resolved, "no_pca", False))) and int(getattr(resolved, "pca_components", 0) or 0) > 0
#     if pca_active:
#         _set_resolved(resolved, "spectral_summary_is_physical", False, reasons, "pca_components_are_not_physical_wavelengths")
#     if getattr(resolved, "descriptor_refine_steps_per_epoch", None) is None:
#         _set_resolved(
#             resolved,
#             "descriptor_refine_steps_per_epoch",
#             int(getattr(resolved, "descriptor_refine_steps", 20)),
#             reasons,
#             "default_filled",
#         )
#     if getattr(resolved, "loss_scale", None) is None:
#         _set_resolved(resolved, "loss_scale", float(getattr(resolved, "logit_scale", 8.0)), reasons, "classifier_scale_alias")

#     if bool(getattr(resolved, "base_only", False)):
#         _set_resolved(resolved, "epochs_inc", 0, reasons, "base_only")
#         _set_resolved(resolved, "lr_inc", 0.0, reasons, "base_only")
#         _set_resolved(resolved, "best_state_metric", "geometry_score", reasons, "base_only")

#     method_identity = build_method_identity(resolved)
#     diff = compute_config_diff(original, namespace_to_dict(resolved), reasons)
#     return resolved, diff, method_identity


# def build_method_identity(args: argparse.Namespace) -> Dict[str, Any]:
#     return {
#         "method_name": "Low-Rank Geometry Replay and Residual Geometry Adaptation for HSI NECIL",
#         "short_name": "LRGRR-HSI",
#         "main_path": True,
#         "incremental_update_mode": "geometry_gated_adapter",
#         "base": {
#             "temporary_ce_head": True,
#             "mandatory_balanced_ce": True,
#             "mandatory_gics": True,
#             "mandatory_pgr": ["compact", "center", "subspace", "band", "volume"],
#             "physical_spectral_shape_only_when_raw_spectra_exist": True,
#             "geometry_bank_space": "canonical_projected_z",
#             "base_handoff_certificate": True,
#         },
#         "incremental": {
#             "frozen_backbone": True,
#             "frozen_projection": True,
#             "frozen_old_geometry_bank_rows": True,
#             "new_class_geometry_insertion": True,
#             "synthetic_old_geometry_replay": True,
#             "geometry_plastic_adapter": True,
#             "seen_local_geometry_classifier": True,
#             "joint_old_new_ce": True,
#             "old_new_energy_margin": True,
#             "descriptor_refinement_new_rows_only": bool(getattr(args, "refine_new_descriptors", True)),
#         },
#         "forbidden": {
#             "raw_exemplars": False,
#             "stored_old_features": False,
#             "kd_teacher": False,
#             "centroid_classifier": False,
#             "old_row_transport": False,
#             "score_calibrator": False,
#             "adaptive_boundary": False,
#             "bicyc_cycle": False,
#             "projection_plasticity": False,
#         },
#         "label_convention": {
#             "dataset_labels": "global_class_ids",
#             "geometry_bank_rows": "global_class_ids",
#             "classifier_logits": "seen_local_column_order",
#             "ce_targets": "seen_local_labels",
#             "evaluation_predictions": "mapped_to_global_class_ids",
#         },
#     }


# def validate_config(args: argparse.Namespace, *, num_classes: Optional[int] = None) -> None:
#     unknown_args = list(getattr(args, "_unknown_args", []) or [])
#     if unknown_args and not bool(getattr(args, "allow_unknown_legacy_args", False)):
#         raise ValueError(
#             "Unknown CLI arguments are not allowed in the PG-RGA main path because they can silently disable a required component: "
#             + str(unknown_args)
#             + ". Add the flag to main.py if it is part of the architecture, or remove it from the command."
#         )
#     if args.dataset not in DATASET_INFO:
#         raise ValueError(f"Unknown dataset {args.dataset!r}.")
#     if not bool(args.strict_non_exemplar):
#         raise ValueError("PG-RGA requires --strict_non_exemplar true.")
#     if int(args.patch_size) <= 0 or int(args.patch_size) % 2 == 0:
#         raise ValueError("--patch_size must be a positive odd integer.")
#     if float(args.train_ratio) <= 0 or float(args.val_ratio) < 0 or float(args.train_ratio) + float(args.val_ratio) >= 1.0:
#         raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")
#     if int(args.epochs_base) <= 0:
#         raise ValueError("--epochs_base must be positive.")
#     if int(args.epochs_inc) < 0:
#         raise ValueError("--epochs_inc must be >= 0.")
#     if int(args.batch_size) <= 0:
#         raise ValueError("--batch_size must be positive.")
#     if int(args.d_model) <= 0:
#         raise ValueError("--d_model must be positive.")
#     if int(args.subspace_rank) <= 0 or int(args.subspace_rank) >= int(args.d_model):
#         raise ValueError("Require 0 < subspace_rank < d_model.")
#     if not bool(args.no_pca) and int(args.pca_components) <= 0:
#         raise ValueError("--pca_components must be positive unless --no_pca is used.")
#     if float(args.geom_var_floor) <= 0:
#         raise ValueError("--geom_var_floor must be > 0.")
#     if normalize_incremental_update_mode(args.incremental_update_mode) != "geometry_gated_adapter":
#         raise ValueError("PG-RGA requires incremental_update_mode=geometry_gated_adapter.")
#     if bool(args.allow_incremental_projection_training) or not bool(args.freeze_projection_during_incremental):
#         raise ValueError("Incremental projection training invalidates frozen GeometryBank coordinates.")
#     forbidden_bool = [
#         "use_geometry_transport",
#         "use_sglat_transport",
#         "allow_old_model_transport",
#         "use_energy_calibrator",
#         "use_adaptive_boundary",
#         "use_incremental_adapter",
#         "use_geometry_calibrator",
#         "use_bicyc_geometry_cycle",
#         "use_mssl_loss",
#     ]
#     bad = [k for k in forbidden_bool if bool(getattr(args, k, False))]
#     if bad:
#         raise ValueError(f"These flags are not part of PG-RGA main path and must be false: {bad}")
#     if bool(args.spectral_summary_is_physical) and (not bool(args.no_pca)) and int(args.pca_components) > 0:
#         raise ValueError("PCA summaries cannot be marked physical.")
#     for key in (
#         "base_ce_weight", "base_srpgr_weight", "base_gics_weight",
#         "pgr_weight", "pgr_compact_weight", "pgr_center_weight",
#         "pgr_subspace_weight", "pgr_band_weight", "pgr_volume_weight",
#         "pgr_max_subspace_overlap", "pgr_min_class_variance",
#     ):
#         if float(getattr(args, key)) <= 0.0:
#             raise ValueError(f"{key} must be > 0 in mandatory base phase.")
#     if not (0.0 < float(args.pgr_max_subspace_overlap) <= 1.0):
#         raise ValueError("--pgr_max_subspace_overlap must be in (0, 1].")
#     if not (0.0 < float(args.pgr_band_overlap_max) <= 1.0):
#         raise ValueError("--pgr_band_overlap_max must be in (0, 1].")
#     if float(args.pgr_min_class_variance) >= float(args.pgr_max_class_variance):
#         raise ValueError("--pgr_min_class_variance must be smaller than --pgr_max_class_variance.")
#     if float(args.base_cert_max_geometry_conflict) <= 0.0:
#         raise ValueError("--base_cert_max_geometry_conflict must be > 0.")
#     if float(args.base_cert_max_geometry_conflict_soft) < float(args.base_cert_max_geometry_conflict):
#         raise ValueError("--base_cert_max_geometry_conflict_soft must be >= --base_cert_max_geometry_conflict.")
#     if float(args.base_cert_max_guided_geometry_conflict) <= 0.0:
#         raise ValueError("--base_cert_max_guided_geometry_conflict must be > 0.")
#     if float(args.base_cert_subspace_warn_overlap) < float(args.base_cert_max_subspace_overlap):
#         raise ValueError("--base_cert_subspace_warn_overlap must be >= --base_cert_max_subspace_overlap.")
#     if float(args.adapter_max_scale) <= 0.0:
#         raise ValueError("PG-RGA requires --adapter_max_scale > 0.")
#     if num_classes is not None:
#         if int(num_classes) != int(DATASET_INFO[args.dataset]["classes"]):
#             raise ValueError(f"Loaded class count {num_classes} does not match DATASET_INFO[{args.dataset}]={DATASET_INFO[args.dataset]['classes']}.")
#         if args.base_classes is not None and (int(args.base_classes) <= 0 or int(args.base_classes) >= int(num_classes)):
#             raise ValueError(f"base_classes={args.base_classes} must be in [1,{num_classes - 1}].")
#         if args.increment is not None and int(args.increment) <= 0:
#             raise ValueError("--increment must be positive.")
#     seeds = parse_seed_list(getattr(args, "seed_list", ""))
#     if seeds is not None and len(seeds) != int(args.num_runs):
#         raise ValueError(f"--seed_list has {len(seeds)} seeds but --num_runs={args.num_runs}.")
#     if int(args.num_runs) <= 0:
#         raise ValueError("--num_runs must be >= 1.")


# def save_config_files(
#     save_root: str,
#     original: Dict[str, Any],
#     resolved: argparse.Namespace,
#     diff: List[Dict[str, Any]],
#     method_identity: Dict[str, Any],
# ) -> Dict[str, str]:
#     os.makedirs(save_root, exist_ok=True)
#     paths = {
#         "config_original": save_json(os.path.join(save_root, "config_original.json"), original),
#         "config_resolved": save_json(os.path.join(save_root, "config_resolved.json"), namespace_to_dict(resolved)),
#         "config_diff": save_json(os.path.join(save_root, "config_diff.json"), diff),
#         "method_identity": save_json(os.path.join(save_root, "method_identity.json"), method_identity),
#     }
#     if getattr(resolved, "_unknown_args", None):
#         paths["unknown_args"] = save_json(os.path.join(save_root, "unknown_args.json"), {"unknown_args": list(resolved._unknown_args)})
#     return paths


# def print_config_summary(args: argparse.Namespace, diff: List[Dict[str, Any]], method_identity: Dict[str, Any]) -> None:
#     print("[Method]", method_identity["method_name"])
#     print(f"[Method] short_name={method_identity['short_name']} | incremental_update_mode={args.incremental_update_mode}")
#     print(f"[Classifier] base={args.base_classifier_mode} | incremental={args.incremental_classifier_mode} | eval={args.eval_classifier_mode}")
#     print(f"[Base] CE={args.base_ce_weight} | GICS={args.base_gics_weight} | PGR={args.pgr_weight} | class_balance={args.base_class_balance}")
#     print(
#         f"[Base Certificate] guided_conflict<={args.base_cert_max_guided_geometry_conflict} | "
#         f"subspace_warn<={args.base_cert_subspace_warn_overlap} | "
#         f"energy_conflict<={args.base_cert_max_geometry_conflict} "
#         f"(soft<={args.base_cert_max_geometry_conflict_soft})"
#     )
#     print(f"[Incremental] replay/class={args.gfa_samples_per_class} | adapter_max_scale={args.adapter_max_scale} | old_new_margin={args.old_new_geometry_margin}")
#     # print("[Forbidden OFF] transport=False | calibrator=False | adaptive_boundary=False | KD=False | raw_exemplars=False")
#     if getattr(args, "_unknown_args", None):
#         mode = "IGNORED BY USER OVERRIDE" if bool(getattr(args, "allow_unknown_legacy_args", False)) else "ERROR"
#         print(f"[UNKNOWN ARGS {mode}] {args._unknown_args}")
#     if diff:
#         print("[Config Diff] resolved changes:")
#         for row in diff[:40]:
#             print(f"  - {row['name']}: {row['original']} -> {row['resolved']} ({row['reason']})")
#         if len(diff) > 40:
#             print(f"  ... {len(diff) - 40} more changes saved in config_diff.json")


# # -----------------------------------------------------------------------------
# # Reproducibility and data loading
# # -----------------------------------------------------------------------------


# def set_seed(seed: int, deterministic: bool = False) -> None:
#     random.seed(int(seed))
#     np.random.seed(int(seed))
#     torch.manual_seed(int(seed))
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(int(seed))
#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         try:
#             torch.use_deterministic_algorithms(True, warn_only=True)
#         except Exception:
#             pass
#     else:
#         torch.backends.cudnn.benchmark = True


# def load_hsi_dataset(args: argparse.Namespace) -> Dict[str, Any]:
#     apply_reduction = (not bool(args.no_pca)) and str(args.reduction_method).lower() != "none"
#     raw_hsi_physical = None
#     label_policy = None
#     try:
#         load_out = LoadHSIData(
#             method=args.dataset,
#             base_dir=args.data_dir,
#             apply_reduction=apply_reduction,
#             n_components=args.pca_components,
#             reduction_method=args.reduction_method,
#             return_label_policy=True,
#             return_raw_hsi=True,
#         )
#         if len(load_out) == 7:
#             hsi, gt, num_classes, target_names, has_bg, label_policy, raw_hsi_physical = load_out
#         else:
#             hsi, gt, num_classes, target_names, has_bg, label_policy = load_out
#     except TypeError:
#         hsi, gt, num_classes, target_names, has_bg = LoadHSIData(
#             method=args.dataset,
#             base_dir=args.data_dir,
#             apply_reduction=apply_reduction,
#             n_components=args.pca_components,
#             reduction_method=args.reduction_method,
#         )

#     validate_config(args, num_classes=int(num_classes))

#     try:
#         cube_out = ImageCubes(
#             HSI=hsi,
#             GT=gt,
#             WS=args.patch_size,
#             removeZeroLabels=True,
#             has_background=has_bg,
#             num_classes=num_classes,
#             pytorch_format=True,
#             label_policy=label_policy,
#             return_center_spectra=True,
#             raw_hsi_for_spectra=raw_hsi_physical,
#         )
#         if len(cube_out) == 4:
#             patches, labels, coords, raw_center_spectra = cube_out
#         else:
#             patches, labels, coords = cube_out
#             raw_center_spectra = None
#     except TypeError:
#         patches, labels, coords = ImageCubes(
#             HSI=hsi,
#             GT=gt,
#             WS=args.patch_size,
#             removeZeroLabels=True,
#             has_background=has_bg,
#             num_classes=num_classes,
#             pytorch_format=True,
#         )
#         raw_center_spectra = None

#     labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
#     if np.any(labels_np < 0):
#         raise RuntimeError("Dataset labels must be non-negative global class IDs after background removal.")

#     args.num_bands = int(patches.shape[1])
#     args.max_classes = int(num_classes)
#     args.raw_spectral_summary_is_physical = bool(raw_center_spectra is not None)

#     dataset_summary = {
#         "dataset": args.dataset,
#         "name": DATASET_INFO[args.dataset]["name"],
#         "expected_raw_bands": DATASET_INFO[args.dataset]["bands"],
#         "used_channels": int(patches.shape[1]),
#         "pca_active": bool(apply_reduction),
#         "pca_components": int(args.pca_components) if apply_reduction else 0,
#         "num_classes": int(num_classes),
#         "has_background": bool(has_bg),
#         "labeled_samples": int(labels_np.size),
#         "train_ratio": float(args.train_ratio),
#         "val_ratio": float(args.val_ratio),
#         "raw_physical_center_spectra_available": bool(raw_center_spectra is not None),
#         "raw_center_spectra_shape": list(raw_center_spectra.shape) if raw_center_spectra is not None and hasattr(raw_center_spectra, "shape") else None,
#         "label_policy": label_policy,
#     }
#     print("[Dataset]")
#     print(f"  name={dataset_summary['name']} | classes={num_classes} | used_channels={patches.shape[1]} | PCA={apply_reduction}")
#     print(f"  samples={dataset_summary['labeled_samples']} | raw_spectra={dataset_summary['raw_physical_center_spectra_available']}")
#     return {
#         "hsi": hsi,
#         "gt": gt,
#         "num_classes": int(num_classes),
#         "target_names": list(target_names),
#         "has_background": bool(has_bg),
#         "label_policy": label_policy,
#         "patches": patches,
#         "labels": labels,
#         "coords": coords,
#         "raw_center_spectra": raw_center_spectra,
#         "summary": dataset_summary,
#     }


# def build_incremental_dataset(args: argparse.Namespace, data: Dict[str, Any]) -> IncrementalHSIDataset:
#     if args.base_classes is None:
#         args.base_classes = 6 if args.dataset in {"IP", "SA", "HC"} else max(2, data["num_classes"] // 2)
#     if args.increment is None:
#         remaining = max(1, data["num_classes"] - int(args.base_classes))
#         args.increment = 3 if remaining >= 3 else 1
#     validate_config(args, num_classes=data["num_classes"])

#     kwargs = dict(
#         patches=data["patches"],
#         labels=data["labels"],
#         coords=data["coords"],
#         gt_shape=data["gt"].shape,
#         GT=data["gt"].copy().astype(np.int64),
#         base_classes=int(args.base_classes),
#         increment=int(args.increment),
#         train_ratio=float(args.train_ratio),
#         val_ratio=float(args.val_ratio),
#         seed=int(args.seed),
#         device=str(args.device),
#         min_train_per_class=int(args.min_train_per_class),
#         strict_non_exemplar=bool(args.strict_non_exemplar),
#     )
#     optional = {
#         "num_workers": int(args.num_workers),
#         "target_names": data["target_names"],
#         "label_policy": data.get("label_policy"),
#         "return_metadata": True,
#         "include_metadata": True,
#         "raw_spectra": data.get("raw_center_spectra"),
#         "center_spectra": data.get("raw_center_spectra"),
#         "spectra_are_physical": bool(data.get("raw_center_spectra") is not None and args.raw_spectral_summary_is_physical),
#     }
#     sig = inspect.signature(IncrementalHSIDataset.__init__)
#     for k, v in optional.items():
#         if k in sig.parameters:
#             kwargs[k] = v
#     dataset = IncrementalHSIDataset(**kwargs)
#     validate_incremental_dataset(dataset, args, data["num_classes"])
#     print_incremental_protocol(dataset)
#     return dataset


# def phase_to_classes_as_list(dataset: Any) -> List[List[int]]:
#     if not hasattr(dataset, "phase_to_classes"):
#         raise RuntimeError("Incremental dataset must expose phase_to_classes.")
#     ptc = getattr(dataset, "phase_to_classes")
#     if isinstance(ptc, dict):
#         phases = sorted(int(k) for k in ptc.keys())
#         expected = list(range(len(phases)))
#         if phases != expected:
#             raise RuntimeError(f"phase_to_classes keys must be contiguous 0..P-1, got {phases}")
#         return [[int(c) for c in list(ptc[p])] for p in phases]
#     if isinstance(ptc, (list, tuple)):
#         return [[int(c) for c in list(v)] for v in ptc]
#     raise RuntimeError(f"Unsupported phase_to_classes type: {type(ptc)}")


# def validate_incremental_dataset(dataset: Any, args: argparse.Namespace, num_classes: int) -> None:
#     phases = phase_to_classes_as_list(dataset)
#     if not phases:
#         raise RuntimeError("phase_to_classes is empty.")
#     all_classes: List[int] = []
#     for p_idx, cls_list in enumerate(phases):
#         if not cls_list:
#             raise RuntimeError(f"Phase {p_idx} has no classes.")
#         all_classes.extend(cls_list)
#     if len(all_classes) != len(set(all_classes)):
#         dup = sorted(c for c in set(all_classes) if all_classes.count(c) > 1)
#         raise RuntimeError(f"A class appears in more than one phase: {dup}")
#     expected = set(range(int(num_classes)))
#     actual = set(all_classes)
#     if actual != expected:
#         raise RuntimeError(f"Phase classes must cover exactly 0..{num_classes - 1}; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
#     if hasattr(dataset, "assert_non_exemplar"):
#         dataset.assert_non_exemplar()
#     if not bool(args.strict_non_exemplar):
#         raise RuntimeError("strict_non_exemplar must be true.")
#     for a, b in [("train_indices", "val_indices"), ("train_indices", "test_indices"), ("val_indices", "test_indices")]:
#         if hasattr(dataset, a) and hasattr(dataset, b):
#             overlap = sorted(set(map(int, getattr(dataset, a))) & set(map(int, getattr(dataset, b))))
#             if overlap:
#                 raise RuntimeError(f"Dataset split leakage: {a} and {b} overlap. First overlaps={overlap[:20]}")


# def print_incremental_protocol(dataset: Any) -> None:
#     print("[Incremental Protocol]")
#     for p_idx, cls_list in enumerate(phase_to_classes_as_list(dataset)):
#         print(f"  phase_{p_idx}_classes={list(map(int, cls_list))}")


# def resolve_target_names(dataset: Any, raw_target_names: List[str]) -> List[str]:
#     if hasattr(dataset, "inv_label_map"):
#         names = []
#         for sid in range(int(getattr(dataset, "num_classes", len(raw_target_names)))):
#             input_label = int(dataset.inv_label_map[sid])
#             names.append(raw_target_names[input_label] if input_label < len(raw_target_names) else f"Class {sid}")
#         return names
#     return list(raw_target_names)


# # -----------------------------------------------------------------------------
# # Model / trainer / evaluator
# # -----------------------------------------------------------------------------


# def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
#     model = NECILModel(args).to(device)
#     if int(getattr(model, "d_model", args.d_model)) != int(args.d_model):
#         raise RuntimeError(f"model.d_model={getattr(model, 'd_model', None)} != args.d_model={args.d_model}")
#     gb = getattr(model, "geometry_bank", None)
#     if gb is None:
#         raise RuntimeError("Model must expose geometry_bank.")
#     bank_rank = getattr(gb, "rank", getattr(gb, "subspace_rank", None))
#     if bank_rank is not None and int(bank_rank) != int(args.subspace_rank):
#         raise RuntimeError(f"GeometryBank rank {bank_rank} != subspace_rank {args.subspace_rank}")
#     if getattr(model, "classifier", None) is None:
#         raise RuntimeError("Model must expose classifier.")
#     for required in ("extract_projected_features", "get_subspace_bank"):
#         if not hasattr(model, required):
#             raise RuntimeError(f"Model missing required API: {required}")
#     if hasattr(model, "incremental_update_mode"):
#         model.incremental_update_mode = "geometry_gated_adapter"
#     if hasattr(model, "use_geometry_gated_adapter"):
#         model.use_geometry_gated_adapter = True
#     if hasattr(model, "use_geometry_calibrator"):
#         model.use_geometry_calibrator = False
#     if hasattr(model, "use_incremental_adapter"):
#         model.use_incremental_adapter = False
#     print("[Model]")
#     print(f"  feature_dim={args.d_model} | subspace_rank={args.subspace_rank} | classifier={type(model.classifier).__name__}")
#     print("  incremental_plasticity=geometry_plastic_adapter | transport=False | calibrator=False")
#     return model


# def _canonical_method_value(name: str, value: Any) -> Any:
#     """Canonicalize semantically equivalent method values for identity checks."""
#     if name in {"base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "classifier_mode"}:
#         return normalize_classifier_mode(value)
#     if name == "incremental_update_mode":
#         return normalize_incremental_update_mode(value)
#     if name == "use_geometry_gated_adapter":
#         return bool(value)
#     return value


# def build_trainer(model: torch.nn.Module, dataset: Any, args: argparse.Namespace, run_dir: str) -> Trainer:
#     before = namespace_to_dict(args)
#     trainer = Trainer(model, dataset, args)
#     after = namespace_to_dict(args)
#     diff = compute_config_diff(before, after, {})
#     save_json(os.path.join(run_dir, "config_diff_after_trainer.json"), diff)
#     critical = {"incremental_update_mode", "base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "use_geometry_gated_adapter"}
#     changed_critical = []
#     for d in diff:
#         name = d["name"]
#         if name not in critical:
#             continue
#         before_v = _canonical_method_value(name, d.get("original"))
#         after_v = _canonical_method_value(name, d.get("resolved"))
#         if before_v != after_v:
#             changed_critical.append(d)
#     if changed_critical:
#         raise RuntimeError(f"Trainer changed method identity after construction: {changed_critical}")
#     if hasattr(trainer, "assert_method_identity"):
#         trainer.assert_method_identity()
#     return trainer


# def build_evaluator() -> NECILEvaluator:
#     return NECILEvaluator()


# # -----------------------------------------------------------------------------
# # Phase metadata and evaluation
# # -----------------------------------------------------------------------------


# def get_phase_info(dataset: Any, phase: int) -> Dict[str, Any]:
#     phase = int(phase)
#     if hasattr(dataset, "get_phase_info"):
#         info = dataset.get_phase_info(phase)
#         if isinstance(info, dict):
#             old_classes = [int(c) for c in info.get("old_classes", [])]
#             new_classes = [int(c) for c in info.get("new_classes", [])]
#             seen_classes = [int(c) for c in info.get("seen_classes", old_classes + new_classes)]
#             if not seen_classes:
#                 seen_classes = old_classes + new_classes
#             if len(seen_classes) != len(set(seen_classes)):
#                 raise RuntimeError(f"Duplicate class in dataset.get_phase_info({phase}) seen_classes={seen_classes}")
#             return {
#                 "phase": int(info.get("phase", phase)),
#                 "old_classes": old_classes,
#                 "new_classes": new_classes,
#                 "seen_classes": seen_classes,
#                 "old_class_count": int(info.get("old_class_count", len(old_classes))),
#             }

#     phases = phase_to_classes_as_list(dataset)
#     if phase < 0 or phase >= len(phases):
#         raise RuntimeError(f"Invalid phase {phase}; available phases are 0..{len(phases) - 1}.")
#     new_classes = [int(c) for c in phases[phase]]
#     old_classes: List[int] = []
#     for p_idx in range(phase):
#         old_classes.extend(int(c) for c in phases[p_idx])
#     old_classes = sorted(set(old_classes))
#     seen_classes = old_classes + new_classes
#     if len(seen_classes) != len(set(seen_classes)):
#         raise RuntimeError(f"Duplicate class in seen_classes at phase {phase}: {seen_classes}")
#     return {"phase": phase, "old_classes": old_classes, "new_classes": new_classes, "seen_classes": seen_classes, "old_class_count": len(old_classes)}


# def set_model_phase(model: torch.nn.Module, phase_info: Dict[str, Any]) -> None:
#     phase = int(phase_info["phase"])
#     old_count = int(len(phase_info.get("old_classes", [])))
#     if hasattr(model, "set_phase"):
#         model.set_phase(phase)
#     else:
#         model.current_phase = phase
#     if hasattr(model, "set_old_class_count"):
#         model.set_old_class_count(old_count)
#     else:
#         model.old_class_count = old_count


# def unpack_eval_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
#     if isinstance(batch, dict):
#         patches = batch.get("image", batch.get("patch", batch.get("patches", None)))
#         labels = batch.get("label", batch.get("labels", None))
#         spectra = batch.get("spectrum", batch.get("spectra", None))
#         coords = batch.get("coord", batch.get("coords", None))
#         if patches is None or labels is None:
#             raise RuntimeError(f"Unsupported eval batch dict keys: {list(batch.keys())}")
#         return patches, labels, spectra, coords
#     if isinstance(batch, (tuple, list)) and len(batch) >= 2:
#         spectra = batch[2] if len(batch) >= 3 else None
#         coords = batch[3] if len(batch) >= 4 else None
#         return batch[0], batch[1], spectra, coords
#     raise RuntimeError(f"Unsupported eval batch type: {type(batch)}")


# def prepare_eval_spectra(patches: torch.Tensor, spectra: Optional[torch.Tensor], args: argparse.Namespace) -> Tuple[Optional[torch.Tensor], bool, Dict[str, Any]]:
#     if torch.is_tensor(spectra) and spectra.numel() > 0:
#         s = spectra.to(device=patches.device, dtype=patches.dtype)
#         if s.dim() == 4:
#             s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#         elif s.dim() == 3:
#             if s.size(0) == patches.size(0) and s.size(1) > 0 and s.size(2) > 1:
#                 s = s[:, :, s.size(-1) // 2]
#             else:
#                 s = s.reshape(patches.size(0), -1)
#         elif s.dim() == 1:
#             if s.numel() % max(int(patches.size(0)), 1) != 0:
#                 raise RuntimeError("1-D spectra cannot be reshaped to batch.")
#             s = s.reshape(patches.size(0), -1)
#         elif s.dim() != 2:
#             s = s.reshape(patches.size(0), -1)
#         if s.size(0) != patches.size(0):
#             raise RuntimeError("Spectral metadata batch mismatch during evaluation.")
#         physical = bool(args.raw_spectral_summary_is_physical)
#         source = "batch_metadata"
#     else:
#         s = None
#         physical = False
#         source = "none"
#     pca_active = (not bool(args.no_pca)) and int(args.pca_components) > 0
#     if pca_active and s is not None and s.size(1) <= int(args.pca_components):
#         physical = False
#     return s, bool(physical), {"source": source, "physical": bool(physical), "pca_active": bool(pca_active), "spectral_dim": int(s.size(1)) if s is not None else 0}


# def forward_eval_batch(model: torch.nn.Module, patches: torch.Tensor, spectra: Optional[torch.Tensor], args: argparse.Namespace, seen_classes: List[int]) -> Dict[str, Any]:
#     spectral_summary, spectral_is_physical, spec_diag = prepare_eval_spectra(patches, spectra, args)
#     kwargs = dict(
#         seen_classes=[int(c) for c in seen_classes],
#         classifier_mode=normalize_classifier_mode(args.eval_classifier_mode),
#         return_energy=True,
#         spectral_summary=spectral_summary,
#         spectral_summary_is_physical=bool(spectral_is_physical),
#     )
#     try:
#         out = model(patches, **kwargs)
#     except TypeError:
#         kwargs.pop("spectral_summary", None)
#         kwargs.pop("spectral_summary_is_physical", None)
#         out = model(patches, **kwargs)
#     if not isinstance(out, dict):
#         out = {"logits": out}
#     out["spectral_diagnostics"] = spec_diag
#     return out


# def logits_to_global_predictions(logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
#     seen = torch.as_tensor([int(c) for c in seen_classes], device=logits.device, dtype=torch.long)
#     if logits.dim() != 2:
#         raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
#     if logits.size(1) == seen.numel():
#         pred_local = logits.argmax(dim=1)
#         return seen[pred_local]
#     if seen.numel() > 0 and int(seen.max().item()) < logits.size(1):
#         logits_seen = logits.index_select(1, seen)
#         pred_local = logits_seen.argmax(dim=1)
#         return seen[pred_local]
#     raise RuntimeError(f"Cannot map logits width={logits.size(1)} to seen_classes={seen.detach().cpu().tolist()}")


# @torch.no_grad()
# def get_phase_predictions(
#     model: torch.nn.Module,
#     dataset: Any,
#     phase_info: Dict[str, Any],
#     device: torch.device,
#     args: argparse.Namespace,
#     batch_size: int,
# ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
#     model.eval()
#     set_model_phase(model, phase_info)
#     phase = int(phase_info["phase"])
#     seen_classes = [int(c) for c in phase_info["seen_classes"]]
#     loader = dataset.get_cumulative_dataloader(phase, split="test", batch_size=batch_size, shuffle=False)
#     preds: List[np.ndarray] = []
#     labels_all: List[np.ndarray] = []
#     pred_hist: Dict[int, int] = {}
#     for batch in loader:
#         patches, labels, spectra, _ = unpack_eval_batch(batch)
#         patches = patches.to(device, non_blocking=True).float()
#         if torch.is_tensor(spectra):
#             spectra = spectra.to(device, non_blocking=True)
#         labels_t = labels.to(device).long().view(-1) if torch.is_tensor(labels) else torch.as_tensor(labels, device=device).long().view(-1)
#         if not set(labels_t.detach().cpu().tolist()).issubset(set(seen_classes)):
#             bad = sorted(set(labels_t.detach().cpu().tolist()) - set(seen_classes))
#             raise RuntimeError(f"Evaluation labels outside seen classes at phase {phase}: {bad}")
#         out = forward_eval_batch(model, patches, spectra, args, seen_classes)
#         pred_global = logits_to_global_predictions(out["logits"], seen_classes)
#         if not set(pred_global.detach().cpu().tolist()).issubset(set(seen_classes)):
#             raise RuntimeError("Evaluation produced unseen predictions after seen-class mapping.")
#         for p in pred_global.detach().cpu().tolist():
#             pred_hist[int(p)] = pred_hist.get(int(p), 0) + 1
#         preds.append(pred_global.detach().cpu().numpy())
#         labels_all.append(labels_t.detach().cpu().numpy())
#     if not preds:
#         return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {"prediction_histogram": {}}
#     return np.concatenate(preds), np.concatenate(labels_all), {"prediction_histogram": pred_hist}


# def compute_phase_metrics(y_true: np.ndarray, y_pred: np.ndarray, phase_info: Dict[str, Any]) -> Dict[str, Any]:
#     if y_true.size == 0:
#         return {"overall_accuracy": 0.0, "per_class_accuracy": {}, "old_accuracy": 0.0, "new_accuracy": 0.0, "hm": 0.0}
#     seen = [int(c) for c in phase_info["seen_classes"]]
#     old = [int(c) for c in phase_info["old_classes"]]
#     new = [int(c) for c in phase_info["new_classes"]]
#     if not set(y_true.tolist()).issubset(set(seen)) or not set(y_pred.tolist()).issubset(set(seen)):
#         raise RuntimeError("Metric labels/predictions outside seen classes.")
#     overall = 100.0 * float((y_true == y_pred).mean())
#     per_class: Dict[int, float] = {}
#     for c in seen:
#         mask = y_true == int(c)
#         per_class[int(c)] = 100.0 * float((y_pred[mask] == c).mean()) if mask.any() else 0.0
#     if old and new:
#         old_mask = np.isin(y_true, np.asarray(old))
#         new_mask = np.isin(y_true, np.asarray(new))
#         old_acc = 100.0 * float((y_pred[old_mask] == y_true[old_mask]).mean()) if old_mask.any() else 0.0
#         new_acc = 100.0 * float((y_pred[new_mask] == y_true[new_mask]).mean()) if new_mask.any() else 0.0
#         hm = 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-8)
#     else:
#         old_acc = 0.0
#         new_acc = overall
#         hm = overall
#     cm = np.zeros((len(seen), len(seen)), dtype=np.int64)
#     pos = {c: i for i, c in enumerate(seen)}
#     for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
#         cm[pos[int(yt)], pos[int(yp)]] += 1
#     return {
#         "overall_accuracy": overall,
#         "old_accuracy": old_acc,
#         "new_accuracy": new_acc,
#         "hm": hm,
#         "per_class_accuracy": per_class,
#         "seen_classes": seen,
#         "old_classes": old,
#         "new_classes": new,
#         "confusion_matrix": cm,
#     }


# def evaluator_update_compat(evaluator: Any, phase: int, y_true: np.ndarray, y_pred: np.ndarray, old_class_count: int, seen_classes: List[int]) -> None:
#     sig = inspect.signature(evaluator.update)
#     kwargs = {}
#     if "old_class_count" in sig.parameters:
#         kwargs["old_class_count"] = int(old_class_count)
#     if "seen_classes" in sig.parameters:
#         kwargs["seen_classes"] = seen_classes
#     evaluator.update(int(phase), y_true, y_pred, **kwargs)


# def save_classification_report_compat(
#     evaluator: Any,
#     phase: int,
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     target_names: List[str],
#     phase_dir: str,
#     seen_classes: List[int],
#     old_class_count: int,
#     enabled: bool,
#     tr_time: float,
#     te_time: float,
# ) -> Optional[Dict[str, Any]]:
#     if not enabled:
#         return None
#     os.makedirs(phase_dir, exist_ok=True)
#     if hasattr(evaluator, "save_phase_report"):
#         return evaluator.save_phase_report(
#             phase=int(phase),
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names,
#             save_dir=phase_dir,
#             seen_classes=seen_classes,
#             old_class_count=int(old_class_count),
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=0.0,
#         )
#     return save_classification_report(
#         y_true=y_true,
#         y_pred=y_pred,
#         target_names=target_names,
#         save_dir=phase_dir,
#         phase=int(phase),
#         seen_classes=seen_classes,
#         old_class_count=int(old_class_count),
#         tr_time=tr_time,
#         te_time=te_time,
#         dl_time=0.0,
#         save_hsi_style=True,
#         save_structured=True,
#     )


# # -----------------------------------------------------------------------------
# # Checkpoint, artifacts, phase loop
# # -----------------------------------------------------------------------------


# def geometry_bank_state(model: torch.nn.Module) -> Any:
#     gb = getattr(model, "geometry_bank", None)
#     if gb is None:
#         return None
#     if hasattr(gb, "state_dict"):
#         try:
#             return gb.state_dict()
#         except Exception:
#             pass
#     if hasattr(model, "export_memory_snapshot"):
#         return model.export_memory_snapshot()
#     if hasattr(gb, "export_state"):
#         return gb.export_state()
#     return None


# def runtime_contract(args: argparse.Namespace, phase_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     phase = int(phase_info["phase"]) if phase_info else None
#     return {
#         "method": build_method_identity(args),
#         "phase": phase,
#         "feature_space": "canonical_projected_z in base; adapted_z only through geometry_plastic_adapter in incremental",
#         "classifier_mode": args.incremental_classifier_mode if phase and phase > 0 else args.base_classifier_mode,
#         "eval_mode": args.eval_classifier_mode,
#         "trainable_policy": "base_backbone_projection_ce_head" if phase in {None, 0} else "geometry_plastic_adapter_only",
#         "strict_non_exemplar": bool(args.strict_non_exemplar),
#         "raw_exemplars_stored": False,
#         "old_features_stored": False,
#         "kd_teacher_used": False,
#         "projection_trainable_during_incremental": False,
#         "old_geometry_bank_frozen": True if phase and phase > 0 else None,
#         "label_convention": "dataset/global, logits/seen-local, CE/seen-local, predictions/global",
#         "transport_enabled": False,
#         "old_row_transport_enabled": False,
#         "adapter_enabled": True,
#         "energy_calibrator_enabled": False,
#         "adaptive_boundary_enabled": False,
#     }


# def checkpoint_payload(
#     model: torch.nn.Module,
#     args: argparse.Namespace,
#     phase_info: Dict[str, Any],
#     history: Any,
#     metrics: Dict[str, Any],
#     diagnostics: Dict[str, Any],
#     method_identity: Dict[str, Any],
# ) -> Dict[str, Any]:
#     payload = {
#         "phase": int(phase_info["phase"]),
#         "model_state": model.state_dict(),
#         "model_state_dict": model.state_dict(),
#         "geometry_bank_state": geometry_bank_state(model),
#         "memory_snapshot": model.export_memory_snapshot() if hasattr(model, "export_memory_snapshot") else None,
#         "seen_classes": [int(c) for c in phase_info["seen_classes"]],
#         "old_classes": [int(c) for c in phase_info["old_classes"]],
#         "new_classes": [int(c) for c in phase_info["new_classes"]],
#         "class_mappings": {
#             "seen_global_to_local": {str(c): i for i, c in enumerate(phase_info["seen_classes"])},
#             "seen_local_to_global": {str(i): int(c) for i, c in enumerate(phase_info["seen_classes"])},
#         },
#         "base_geometry_certificate": getattr(model, "base_geometry_certificate", None),
#         "base_handoff": getattr(model, "base_handoff", None),
#         "runtime_contract": runtime_contract(args, phase_info),
#         "method_identity": method_identity,
#         "args_resolved": namespace_to_dict(args),
#         "metrics": metrics,
#         "history": history,
#         "diagnostics": diagnostics,
#     }
#     required = ["phase", "model_state", "geometry_bank_state", "seen_classes", "class_mappings", "runtime_contract", "args_resolved", "metrics"]
#     missing = [k for k in required if k not in payload or payload[k] is None]
#     if missing:
#         raise RuntimeError(f"Checkpoint missing critical fields: {missing}")
#     return payload


# def save_phase_artifacts(
#     phase_dir: str,
#     model: torch.nn.Module,
#     args: argparse.Namespace,
#     phase_info: Dict[str, Any],
#     history: Any,
#     metrics: Dict[str, Any],
#     diagnostics: Dict[str, Any],
#     method_identity: Dict[str, Any],
#     classification_report: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, str]:
#     os.makedirs(phase_dir, exist_ok=True)
#     paths = {
#         "metrics": save_json(os.path.join(phase_dir, "metrics.json"), metrics),
#         "diagnostics": save_json(os.path.join(phase_dir, "diagnostics.json"), diagnostics),
#         "runtime_contract": save_json(os.path.join(phase_dir, "runtime_contract.json"), runtime_contract(args, phase_info)),
#     }
#     if classification_report is not None:
#         paths["classification_report_info"] = save_json(os.path.join(phase_dir, "classification_report_info.json"), classification_report)
#     if int(phase_info["phase"]) == 0:
#         cert = getattr(model, "base_geometry_certificate", None)
#         if cert is not None:
#             paths["geometry_certificate"] = save_json(os.path.join(phase_dir, "geometry_certificate.json"), cert)
#     cm = metrics.get("confusion_matrix", None)
#     if isinstance(cm, np.ndarray):
#         cm_path = os.path.join(phase_dir, "confusion_matrix.npy")
#         np.save(cm_path, cm)
#         paths["confusion_matrix"] = cm_path
#     ckpt = checkpoint_payload(model, args, phase_info, history, metrics, diagnostics, method_identity)
#     ckpt_path = os.path.join(phase_dir, f"phase_{int(phase_info['phase'])}_checkpoint.pt")
#     torch.save(ckpt, ckpt_path)
#     paths["checkpoint"] = ckpt_path
#     return paths


# def collect_trainer_diagnostics(trainer: Any, phase: int, phase_dir: str) -> Dict[str, Any]:
#     candidates = [
#         f"_last_phase_{int(phase)}_geometry_diagnostics",
#         "_last_incremental_geometry_diagnostics",
#         "_last_phase_geometry_diagnostics",
#         "_last_geometry_diagnostics",
#     ]
#     if int(phase) == 0:
#         candidates.insert(0, "_last_base_geometry_diagnostics")
#     for attr in candidates:
#         diag = getattr(trainer, attr, None)
#         if isinstance(diag, dict) and diag:
#             return _json_safe(diag)
#     json_path = os.path.join(phase_dir, f"phase_{int(phase)}_geometry_diagnostics.json")
#     if os.path.exists(json_path):
#         with open(json_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}


# def call_phase_map_compat(model: torch.nn.Module, dataset: Any, phase: int, target_names: List[str], phase_dir: str, args: argparse.Namespace) -> None:
#     if bool(args.skip_phase_maps):
#         return
#     try:
#         sig = inspect.signature(predict_phase_grid)
#         kwargs = {
#             "model": model,
#             "dataset_manager": dataset,
#             "phase": int(phase),
#             "target_names": target_names,
#             "save_dir": os.path.join(phase_dir, "maps"),
#             "device": args.device,
#             "patch_size": args.patch_size,
#             "classifier_mode": args.eval_classifier_mode,
#             "semantic_mode": "identity",
#             "class_cmap": args.viz_class_cmap,
#             "background_color": args.viz_background_color,
#             "save_numpy": args.viz_save_numpy,
#         }
#         predict_phase_grid(**{k: v for k, v in kwargs.items() if k in sig.parameters})
#     except Exception as exc:
#         print(f"[WARN] phase map generation failed: {exc}")


# def assert_base_handoff_ready(model: torch.nn.Module, trainer: Any) -> None:
#     if hasattr(model, "assert_base_handoff_ready"):
#         phases = getattr(trainer.dataset, "phase_to_classes", None)
#         try:
#             base_ids = phase_to_classes_as_list(trainer.dataset)[0]
#             model.assert_base_handoff_ready(base_ids, freeze=True, strict=True)
#             return
#         except Exception:
#             pass
#     handoff = getattr(model, "base_handoff", None) or getattr(trainer, "base_handoff", None) or getattr(trainer, "_last_base_handoff", None)
#     cert = getattr(model, "base_geometry_certificate", None) or getattr(trainer, "_last_base_geometry_certificate", None)
#     if handoff is None and cert is None:
#         raise RuntimeError("Base phase completed without base handoff/certificate. Incremental phase cannot be trusted.")


# def assert_incremental_phase_complete(model: torch.nn.Module, phase_info: Dict[str, Any]) -> None:
#     if geometry_bank_state(model) is None:
#         raise RuntimeError("Incremental phase ended without GeometryBank state.")
#     if int(phase_info["phase"]) > 0 and not phase_info.get("old_classes"):
#         raise RuntimeError("Incremental phase missing old classes in phase_info.")


# def run_phase(
#     *,
#     trainer: Trainer,
#     model: torch.nn.Module,
#     dataset: Any,
#     evaluator: Any,
#     args: argparse.Namespace,
#     phase_info: Dict[str, Any],
#     target_names: List[str],
#     run_dir: str,
#     method_identity: Dict[str, Any],
# ) -> Dict[str, Any]:
#     phase = int(phase_info["phase"])
#     phase_dir = os.path.join(run_dir, f"phase_{phase}")
#     os.makedirs(phase_dir, exist_ok=True)
#     if hasattr(dataset, "start_phase"):
#         dataset.start_phase(phase)
#     set_model_phase(model, phase_info)
#     print("\n" + "=" * 88)
#     print(f"[Phase {phase}] old={phase_info['old_classes']} | new={phase_info['new_classes']} | seen={phase_info['seen_classes']}")
#     print("=" * 88)

#     epochs = int(args.epochs_base if phase == 0 else args.epochs_inc)
#     lr = float(args.lr if phase == 0 else (args.lr_inc if args.lr_inc > 0 else args.lr))

#     if phase > 0 and hasattr(trainer, "_assert_incremental_preflight"):
#         trainer._assert_incremental_preflight(phase, int(len(phase_info["old_classes"])))
#         print(f"[Incremental Preflight PASS] phase={phase}")

#     t0 = time.time()
#     if phase == 0 and hasattr(trainer, "train_base_phase"):
#         history = trainer.train_base_phase(phase=0, epochs=epochs, batch_size=args.batch_size, lr=lr)
#         assert_base_handoff_ready(model, trainer)
#     elif phase > 0 and hasattr(trainer, "train_incremental_phase"):
#         history = trainer.train_incremental_phase(phase=phase, epochs=epochs, batch_size=args.batch_size, lr=lr)
#         assert_incremental_phase_complete(model, phase_info)
#     else:
#         history = trainer.train_phase(phase=phase, epochs=epochs, batch_size=args.batch_size, lr=lr)
#     train_time = time.time() - t0

#     print(f"[Eval] phase={phase} cumulative seen-class evaluation")
#     e0 = time.time()
#     y_pred, y_true, pred_diag = get_phase_predictions(model, dataset, phase_info, torch.device(args.device), args, batch_size=args.batch_size)
#     eval_time = time.time() - e0

#     metrics = compute_phase_metrics(y_true, y_pred, phase_info)
#     metrics["train_time_sec"] = train_time
#     metrics["eval_time_sec"] = eval_time
#     metrics["prediction_histogram"] = pred_diag.get("prediction_histogram", {})
#     evaluator_update_compat(evaluator, phase, y_true, y_pred, int(len(phase_info["old_classes"])), phase_info["seen_classes"])
#     if hasattr(evaluator, "print_summary"):
#         evaluator.print_summary()
#     report = save_classification_report_compat(
#         evaluator=evaluator,
#         phase=phase,
#         y_true=y_true,
#         y_pred=y_pred,
#         target_names=target_names,
#         phase_dir=phase_dir,
#         seen_classes=phase_info["seen_classes"],
#         old_class_count=int(len(phase_info["old_classes"])),
#         enabled=bool(args.save_classification_report),
#         tr_time=train_time,
#         te_time=eval_time,
#     )
#     diagnostics = collect_trainer_diagnostics(trainer, phase, phase_dir)
#     diagnostics["prediction_histogram"] = pred_diag.get("prediction_histogram", {})
#     diagnostics["phase_info"] = phase_info
#     paths = save_phase_artifacts(phase_dir, model, args, phase_info, history, metrics, diagnostics, method_identity, classification_report=report)
#     call_phase_map_compat(model, dataset, phase, target_names, phase_dir, args)
#     return {
#         "phase": phase,
#         "phase_dir": phase_dir,
#         "phase_info": phase_info,
#         "history": history,
#         "metrics": _json_safe(metrics),
#         "diagnostics": diagnostics,
#         "artifact_paths": paths,
#         "classification_report": report,
#         "train_time_sec": train_time,
#         "eval_time_sec": eval_time,
#     }


# def determine_total_phases(dataset: Any, args: argparse.Namespace) -> int:
#     phases = phase_to_classes_as_list(dataset)
#     dataset_total = int(getattr(dataset, "num_phases", len(phases)))
#     if dataset_total != len(phases):
#         print(f"[WARN] dataset.num_phases={dataset_total} but len(phase_to_classes)={len(phases)}. Using phase_to_classes.")
#         dataset_total = len(phases)
#     if bool(args.base_only):
#         return 1
#     total = dataset_total
#     if int(args.max_phases or 0) > 0:
#         total = min(total, int(args.max_phases))
#     elif int(args.max_train_phase or -1) >= 0:
#         total = min(total, int(args.max_train_phase) + 1)
#     return max(1, total)


# def save_dataset_protocol_files(run_dir: str, data: Dict[str, Any], dataset: Any) -> Dict[str, str]:
#     phases = phase_to_classes_as_list(dataset)
#     phase_splits = {
#         "num_phases": int(len(phases)),
#         "class_order": _json_safe(getattr(dataset, "class_order", None)),
#         "phase_to_classes": _json_safe(phases),
#     }
#     return {
#         "dataset_summary": save_json(os.path.join(run_dir, "dataset_summary.json"), data["summary"]),
#         "phase_splits": save_json(os.path.join(run_dir, "phase_splits.json"), phase_splits),
#     }


# def run_single_experiment(base_args: argparse.Namespace, run_idx: int, run_seed: int) -> Dict[str, Any]:
#     raw_args = argparse.Namespace(**namespace_to_dict(base_args))
#     raw_args.seed = int(run_seed)
#     original = namespace_to_dict(raw_args)
#     resolved, diff, method_identity = resolve_experiment_config(raw_args)
#     validate_config(resolved)
#     set_seed(resolved.seed, deterministic=bool(resolved.deterministic))

#     run_tag = "base_only" if bool(resolved.base_only) else "pg_rga"
#     run_dir = os.path.join(
#         resolved.save_dir,
#         resolved.dataset,
#         f"patch_{resolved.patch_size}",
#         f"{run_tag}_run_{run_idx + 1}_seed_{resolved.seed}",
#     )
#     os.makedirs(run_dir, exist_ok=True)
#     resolved.run_dir = run_dir
#     resolved.save_dir = run_dir

#     save_config_files(run_dir, original, resolved, diff, method_identity)
#     print("\n=== NECIL-HSI RUN ===")
#     print(f"Run {run_idx + 1}/{base_args.num_runs} | seed={resolved.seed} | device={resolved.device}")
#     print_config_summary(resolved, diff, method_identity)

#     data = load_hsi_dataset(resolved)
#     dataset = build_incremental_dataset(resolved, data)
#     target_names = resolve_target_names(dataset, data["target_names"])
#     if hasattr(dataset, "target_names"):
#         dataset.target_names = target_names
#     save_dataset_protocol_files(run_dir, data, dataset)

#     device = torch.device(resolved.device)
#     model = build_model(resolved, device)
#     trainer = build_trainer(model, dataset, resolved, run_dir)
#     evaluator = build_evaluator()

#     phase_results: Dict[int, Dict[str, Any]] = {}
#     total_phases = determine_total_phases(dataset, resolved)
#     print(f"[Run] phases=0..{total_phases - 1} of dataset phases={getattr(dataset, 'num_phases', total_phases)}")
#     start = time.time()
#     for phase in range(total_phases):
#         phase_info = get_phase_info(dataset, phase)
#         phase_results[phase] = run_phase(
#             trainer=trainer,
#             model=model,
#             dataset=dataset,
#             evaluator=evaluator,
#             args=resolved,
#             phase_info=phase_info,
#             target_names=target_names,
#             run_dir=run_dir,
#             method_identity=method_identity,
#         )

#     elapsed = time.time() - start
#     final_phase = max(phase_results)
#     final_metrics = phase_results[final_phase]["metrics"]
#     final_phase_info = get_phase_info(dataset, final_phase)
#     final_results = {
#         "run_idx": run_idx,
#         "seed": int(resolved.seed),
#         "run_dir": run_dir,
#         "elapsed_sec": elapsed,
#         "final_phase": int(final_phase),
#         "final_metrics": final_metrics,
#         "phase_results": _json_safe(phase_results),
#         "method_identity": method_identity,
#         "runtime_contract": runtime_contract(resolved, final_phase_info),
#         "evaluator": evaluator.to_dict() if hasattr(evaluator, "to_dict") else None,
#     }
#     save_json(os.path.join(run_dir, "final_results.json"), final_results)
#     torch.save(
#         checkpoint_payload(model, resolved, final_phase_info, None, final_metrics, {}, method_identity),
#         os.path.join(run_dir, "final_model.pt"),
#     )
#     try:
#         history = {}
#         for pr in phase_results.values():
#             h = pr.get("history", {})
#             if isinstance(h, dict):
#                 for k, v in h.items():
#                     if isinstance(v, list):
#                         history.setdefault(k, []).extend(v)
#         if history:
#             plot_training_history(history, os.path.join(run_dir, "training_history.png"))
#     except Exception as exc:
#         print(f"[WARN] Could not plot training history: {exc}")
#     write_run_report(os.path.join(run_dir, "PG_RGA_HSI_RUN_REPORT.txt"), resolved, final_results)
#     print(f"[Done] run_dir={run_dir} | final_OA={float(final_metrics.get('overall_accuracy', 0.0)):.2f} | final_HM={float(final_metrics.get('hm', 0.0)):.2f}")
#     return final_results


# def write_run_report(path: str, args: argparse.Namespace, result: Dict[str, Any]) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(f"NECIL-HSI Run Report - {args.dataset}\n")
#         f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 80 + "\n")
#         f.write(json.dumps(_json_safe(result["method_identity"]), indent=2) + "\n\n")
#         f.write("Phase metrics\n")
#         f.write("-" * 80 + "\n")
#         for p, pr in sorted((result.get("phase_results", {}) or {}).items(), key=lambda kv: int(kv[0])):
#             m = pr.get("metrics", {}) or {}
#             info = pr.get("phase_info", {}) or {}
#             f.write(
#                 f"Phase {p}: OA={float(m.get('overall_accuracy', 0.0)):.2f} | "
#                 f"Old={float(m.get('old_accuracy', 0.0)):.2f} | New={float(m.get('new_accuracy', 0.0)):.2f} | "
#                 f"HM={float(m.get('hm', 0.0)):.2f} | seen={info.get('seen_classes', [])}\n"
#             )
#         f.write("\nFinal metrics\n")
#         f.write(json.dumps(_json_safe(result.get("final_metrics", {})), indent=2) + "\n")
#     print(f"[Report] {path}")


# def aggregate_runs(results: List[Dict[str, Any]], root_dir: str) -> Dict[str, Any]:
#     os.makedirs(root_dir, exist_ok=True)
#     rows = []
#     for r in results:
#         m = r.get("final_metrics", {}) or {}
#         rows.append({
#             "run_idx": int(r.get("run_idx", 0)),
#             "seed": int(r.get("seed", 0)),
#             "run_dir": r.get("run_dir", ""),
#             "final_phase": int(r.get("final_phase", 0)),
#             "overall_accuracy": float(m.get("overall_accuracy", 0.0)),
#             "old_accuracy": float(m.get("old_accuracy", 0.0)),
#             "new_accuracy": float(m.get("new_accuracy", 0.0)),
#             "hm": float(m.get("hm", 0.0)),
#         })

#     def mean_std(key: str) -> Tuple[float, float]:
#         arr = np.asarray([row[key] for row in rows], dtype=np.float64)
#         return (float(arr.mean()), float(arr.std(ddof=0))) if arr.size else (0.0, 0.0)

#     summary = {
#         "num_runs": len(results),
#         "rows": rows,
#         "overall_accuracy_mean_std": mean_std("overall_accuracy"),
#         "old_accuracy_mean_std": mean_std("old_accuracy"),
#         "new_accuracy_mean_std": mean_std("new_accuracy"),
#         "hm_mean_std": mean_std("hm"),
#     }
#     save_json(os.path.join(root_dir, "runs_summary.json"), summary)
#     csv_path = os.path.join(root_dir, "runs_summary.csv")
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         fieldnames = list(rows[0].keys()) if rows else ["run_idx", "seed", "run_dir", "overall_accuracy", "old_accuracy", "new_accuracy", "hm"]
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in rows:
#             writer.writerow(row)
#     summary["runs_summary_csv"] = csv_path
#     return summary


# def main(argv: Optional[List[str]] = None) -> None:
#     args = parse_args(argv)
#     preview_args = argparse.Namespace(**namespace_to_dict(args))
#     resolved, _, _ = resolve_experiment_config(preview_args)
#     seeds = parse_seed_list(getattr(resolved, "seed_list", "")) or [int(resolved.seed) + i for i in range(int(resolved.num_runs))]
#     if len(seeds) != int(resolved.num_runs):
#         raise ValueError("seed_list length must match num_runs.")

#     results: List[Dict[str, Any]] = []
#     for run_idx, seed in enumerate(seeds):
#         run_args = argparse.Namespace(**namespace_to_dict(args))
#         run_args.seed = int(seed)
#         results.append(run_single_experiment(run_args, run_idx, int(seed)))

#     root_dir = os.path.join(resolved.save_dir, resolved.dataset, f"patch_{resolved.patch_size}")
#     summary = aggregate_runs(results, root_dir)
#     print("\n=== MULTI-RUN SUMMARY ===")
#     print(f"runs={summary['num_runs']} | OA={summary['overall_accuracy_mean_std'][0]:.2f}±{summary['overall_accuracy_mean_std'][1]:.2f} | HM={summary['hm_mean_std'][0]:.2f}±{summary['hm_mean_std'][1]:.2f}")
#     print(f"Saved: {os.path.join(root_dir, 'runs_summary.json')}")
#     print(f"Saved: {summary['runs_summary_csv']}")


# if __name__ == "__main__":
#     main()

