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
from utils.eval import NECILEvaluator, make_json_serializable, save_classification_report
from utils.visualize import plot_training_history, predict_phase_grid


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
    """PG-RGA is the main method.

    Old names are accepted only so old commands do not break. They all resolve to
    geometry_gated_adapter because PG-RGA uses the bounded geometry residual
    adapter as the only incremental model plasticity.
    """
    m = str(mode or "geometry_gated_adapter").lower().strip()
    aliases = {
        "": "geometry_gated_adapter",
        "none": "geometry_gated_adapter",
        "clean": "geometry_gated_adapter",
        "main": "geometry_gated_adapter",
        "pg_rga": "geometry_gated_adapter",
        "pg-rga": "geometry_gated_adapter",
        "pgrga": "geometry_gated_adapter",
        "geometry_gated_adapter": "geometry_gated_adapter",
        "g2rpa": "geometry_gated_adapter",
        "g2-rpa": "geometry_gated_adapter",
        "g²rpa": "geometry_gated_adapter",
        "gated_adapter": "geometry_gated_adapter",
        "geometry_adapter": "geometry_gated_adapter",
        "adapter": "geometry_gated_adapter",
        # legacy aliases from earlier drafts
        "scbgr": "geometry_gated_adapter",
        "scb-gr": "geometry_gated_adapter",
        "descriptor": "geometry_gated_adapter",
        "descriptor_only": "geometry_gated_adapter",
        "rsgi": "geometry_gated_adapter",
        "geometry_state_admission": "geometry_gated_adapter",
        "spectral_risk_boundary": "geometry_gated_adapter",
        "boundary_geometry": "geometry_gated_adapter",
    }
    if m not in aliases:
        raise ValueError("Unsupported --incremental_update_mode. Use geometry_gated_adapter / pg_rga.")
    return aliases[m]


# -----------------------------------------------------------------------------
# Parser and resolved configuration
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PG-RGA-HSI: GeometryBank-based exemplar-free class-incremental HSI classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    main = parser.add_argument_group("A. Core experiment")
    main.add_argument("--dataset", type=str, default="IP", choices=DATASET_INFO.keys())
    main.add_argument("--data_dir", type=str, default="./datasets")
    main.add_argument("--save_dir", type=str, default="./results_pg_rga")
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

    base = parser.add_argument_group("C. Mandatory base objective")
    base.add_argument("--base_ce_weight", type=float, default=1.0)
    base.add_argument("--base_srpgr_weight", type=float, default=1.0)
    base.add_argument("--base_gics_weight", type=float, default=0.20)
    base.add_argument("--base_gics_temperature", type=float, default=0.07)
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
    base.add_argument("--pgr_max_class_variance", type=float, default=0.75)
    base.add_argument("--pgr_min_class_samples", type=int, default=3)
    base.add_argument("--pgr_subspace_min_samples", type=int, default=6)
    base.add_argument("--pgr_subspace_rank", type=int, default=3)
    base.add_argument("--base_spectral_shape_weight", type=float, default=0.05)
    base.add_argument("--base_require_physical_spectral_shape", type=str2bool, default=False)
    base.add_argument("--strict_base_component_coverage", type=str2bool, default=True)

    inc = parser.add_argument_group("D. PG-RGA incremental objective")
    inc.add_argument("--incremental_update_mode", type=str, default="geometry_gated_adapter")
    inc.add_argument("--gfa_weight", type=float, default=1.0)
    inc.add_argument("--gfa_samples_per_class", type=int, default=48)
    inc.add_argument("--gfa_parallel_scale", type=float, default=1.0)
    inc.add_argument("--gfa_residual_scale", type=float, default=0.25)
    inc.add_argument("--joint_old_new_ce_weight", type=float, default=1.0)
    inc.add_argument("--geometry_energy_margin_weight", type=float, default=0.30)
    inc.add_argument("--geometry_energy_margin", type=float, default=0.30)
    inc.add_argument("--old_new_invasion_weight", type=float, default=0.50)
    inc.add_argument("--old_new_geometry_margin", type=float, default=0.35)
    inc.add_argument("--refine_new_descriptors", type=str2bool, default=True)
    inc.add_argument("--descriptor_refine_steps", type=int, default=20)
    inc.add_argument("--descriptor_refine_steps_per_epoch", type=int, default=None)
    inc.add_argument("--descriptor_refine_lr", type=float, default=1e-3)
    inc.add_argument("--descriptor_trust_weight", type=float, default=0.8)
    inc.add_argument("--descriptor_refine_max_mean_shift", type=float, default=0.30)
    inc.add_argument("--descriptor_refine_max_logvar_shift", type=float, default=0.50)
    inc.add_argument("--adapter_bottleneck", type=int, default=32)
    inc.add_argument("--adapter_max_scale", type=float, default=0.35)
    inc.add_argument("--adapter_dropout", type=float, default=0.0)
    inc.add_argument("--adapter_gate_bias_init", type=float, default=-3.0)
    inc.add_argument("--adapter_lr", type=float, default=5e-4)
    inc.add_argument("--adapter_weight_decay", type=float, default=0.0)
    inc.add_argument("--g2rpa_adapter_weight", type=float, default=1.0)
    inc.add_argument("--adapter_old_delta_weight", type=float, default=1.0)
    inc.add_argument("--adapter_old_gate_weight", type=float, default=0.75)
    inc.add_argument("--adapter_old_energy_weight", type=float, default=0.25)
    inc.add_argument("--adapter_old_margin_weight", type=float, default=0.25)
    inc.add_argument("--adapter_delta_weight", type=float, default=0.10)
    inc.add_argument("--adapter_new_gate_weight", type=float, default=0.05)
    inc.add_argument("--adapter_new_gate_target", type=float, default=0.25)
    inc.add_argument("--adapter_new_gate_max_target", type=float, default=0.75)

    clf = parser.add_argument_group("E. Geometry classifier/evaluation")
    clf.add_argument("--classifier_mode", type=str, default="geometry")
    clf.add_argument("--base_classifier_mode", type=str, default=None)
    clf.add_argument("--incremental_classifier_mode", type=str, default=None)
    clf.add_argument("--eval_classifier_mode", type=str, default="geometry")
    clf.add_argument("--logit_scale", type=float, default=8.0)
    clf.add_argument("--loss_scale", type=float, default=None)
    clf.add_argument("--residual_variance_scale", type=float, default=0.75)
    clf.add_argument("--energy_normalize_by_dim", type=str2bool, default=True)
    clf.add_argument("--use_logdet_energy", type=str2bool, default=True)
    clf.add_argument("--logdet_energy_weight", type=float, default=0.05)
    clf.add_argument("--use_reliability_penalty", type=str2bool, default=True)
    clf.add_argument("--reliability_energy_weight", type=float, default=0.03)
    clf.add_argument("--geometry_logit_clip", type=float, default=0.0)
    clf.add_argument("--best_state_metric", type=str, default="hm")
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
    spec.add_argument("--use_spectral_geometry", type=str2bool, default=True)
    spec.add_argument("--spectral_energy_weight", type=float, default=0.0)
    spec.add_argument("--spectral_derivative_weight", type=float, default=0.50)
    spec.add_argument("--spectral_second_derivative_weight", type=float, default=0.25)
    spec.add_argument("--band_energy_weight", type=float, default=0.0)

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
    legacy.add_argument("--disable_incremental_adapter", type=str2bool, default=True)
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

    mode = normalize_incremental_update_mode(getattr(resolved, "incremental_update_mode", "geometry_gated_adapter"))
    _set_resolved(resolved, "incremental_update_mode", mode, reasons, "pg_rga_method_identity")
    _set_resolved(resolved, "use_geometry_gated_adapter", True, reasons, "pg_rga_adapter_required")

    base_mode = normalize_classifier_mode(getattr(resolved, "base_classifier_mode", None) or "geometry")
    inc_mode = normalize_classifier_mode(getattr(resolved, "incremental_classifier_mode", None) or "geometry")
    eval_mode = normalize_classifier_mode(getattr(resolved, "eval_classifier_mode", "geometry"))
    for key, value in {
        "classifier_mode": eval_mode,
        "base_classifier_mode": base_mode,
        "incremental_classifier_mode": inc_mode,
        "eval_classifier_mode": eval_mode,
    }.items():
        _set_resolved(resolved, key, value, reasons, "seen_local_geometry_classifier")

    forced_true = {
        "strict_non_exemplar": True,
        "strict_feature_contract": True,
        "strict_updated_stack": True,
        "freeze_projection_during_incremental": True,
        "freeze_classifier_during_incremental": True,
        "disable_incremental_adapter": True,
        "base_class_balance": True,
        "strict_base_component_coverage": True,
        "use_spectral_geometry": True,
    }
    for k, v in forced_true.items():
        _set_resolved(resolved, k, v, reasons, "pg_rga_contract")

    forced_false = {
        "allow_incremental_projection_training": False,
        "use_geometry_transport": False,
        "use_sglat_transport": False,
        "allow_old_model_transport": False,
        "allow_transport_without_adapter": False,
        "use_energy_calibrator": False,
        "use_adaptive_boundary": False,
        "use_incremental_adapter": False,
        "use_geometry_calibrator": False,
        "use_bicyc_geometry_cycle": False,
        "use_mssl_loss": False,
        "use_pretrain_incremental_baseline": False,
        "geometry_normalize_logits": False,
        "allow_nonphysical_spectral_summary": False,
    }
    for k, v in forced_false.items():
        _set_resolved(resolved, k, v, reasons, "legacy_or_ablation_disabled")

    forced_zero = [
        "energy_calibration_weight",
        "bss_weight",
        "sym_bss_weight",
        "gdr_weight",
        "anchor_consistency_weight",
        "mssl_weight",
        "mssl_inc_weight",
        "bank_refresh_every",
        "early_stop_patience",
        "base_early_stop_patience",
        "incremental_early_stop_patience",
        "spectral_energy_weight",
        "band_energy_weight",
    ]
    for k in forced_zero:
        _set_resolved(resolved, k, 0.0 if "weight" in k or "energy" in k else 0, reasons, "not_used_in_pg_rga_main_path")

    # Required non-zero base components.
    required_positive = {
        "base_ce_weight": 1.0,
        "base_srpgr_weight": 1.0,
        "base_gics_weight": 0.20,
        "pgr_weight": 0.10,
        "pgr_compact_weight": 0.15,
        "pgr_center_weight": 0.25,
        "pgr_subspace_weight": 0.15,
        "pgr_band_weight": 0.05,
        "pgr_volume_weight": 0.05,
    }
    for k, fallback in required_positive.items():
        if float(getattr(resolved, k, fallback)) <= 0.0:
            _set_resolved(resolved, k, fallback, reasons, "mandatory_base_component")

    pca_active = (not bool(getattr(resolved, "no_pca", False))) and int(getattr(resolved, "pca_components", 0) or 0) > 0
    if pca_active:
        _set_resolved(resolved, "spectral_summary_is_physical", False, reasons, "pca_components_are_not_physical_wavelengths")
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
        "method_name": "Prototype-Free Geometry Replay and Residual Geometry Adaptation for HSI NECIL",
        "short_name": "PG-RGA-HSI",
        "main_path": True,
        "incremental_update_mode": "geometry_gated_adapter",
        "base": {
            "temporary_ce_head": True,
            "mandatory_balanced_ce": True,
            "mandatory_gics": True,
            "mandatory_pgr": ["compact", "center", "subspace", "band", "volume"],
            "physical_spectral_shape_only_when_raw_spectra_exist": True,
            "geometry_bank_space": "canonical_projected_z",
            "base_handoff_certificate": True,
        },
        "incremental": {
            "frozen_backbone": True,
            "frozen_projection": True,
            "frozen_old_geometry_bank_rows": True,
            "new_class_geometry_insertion": True,
            "synthetic_old_geometry_replay": True,
            "geometry_plastic_adapter": True,
            "seen_local_geometry_classifier": True,
            "joint_old_new_ce": True,
            "old_new_energy_margin": True,
            "descriptor_refinement_new_rows_only": bool(getattr(args, "refine_new_descriptors", True)),
        },
        "forbidden": {
            "raw_exemplars": False,
            "stored_old_features": False,
            "kd_teacher": False,
            "prototypes_as_classifier": False,
            "old_row_transport": False,
            "score_calibrator": False,
            "adaptive_boundary": False,
            "bicyc_cycle": False,
            "projection_plasticity": False,
        },
        "label_convention": {
            "dataset_labels": "global_class_ids",
            "geometry_bank_rows": "global_class_ids",
            "classifier_logits": "seen_local_column_order",
            "ce_targets": "seen_local_labels",
            "evaluation_predictions": "mapped_to_global_class_ids",
        },
    }


def validate_config(args: argparse.Namespace, *, num_classes: Optional[int] = None) -> None:
    if args.dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {args.dataset!r}.")
    if not bool(args.strict_non_exemplar):
        raise ValueError("PG-RGA requires --strict_non_exemplar true.")
    if int(args.patch_size) <= 0 or int(args.patch_size) % 2 == 0:
        raise ValueError("--patch_size must be a positive odd integer.")
    if float(args.train_ratio) <= 0 or float(args.val_ratio) < 0 or float(args.train_ratio) + float(args.val_ratio) >= 1.0:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")
    if int(args.epochs_base) <= 0:
        raise ValueError("--epochs_base must be positive.")
    if int(args.epochs_inc) < 0:
        raise ValueError("--epochs_inc must be >= 0.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive.")
    if int(args.d_model) <= 0:
        raise ValueError("--d_model must be positive.")
    if int(args.subspace_rank) <= 0 or int(args.subspace_rank) >= int(args.d_model):
        raise ValueError("Require 0 < subspace_rank < d_model.")
    if not bool(args.no_pca) and int(args.pca_components) <= 0:
        raise ValueError("--pca_components must be positive unless --no_pca is used.")
    if float(args.geom_var_floor) <= 0:
        raise ValueError("--geom_var_floor must be > 0.")
    if normalize_incremental_update_mode(args.incremental_update_mode) != "geometry_gated_adapter":
        raise ValueError("PG-RGA requires incremental_update_mode=geometry_gated_adapter.")
    if bool(args.allow_incremental_projection_training) or not bool(args.freeze_projection_during_incremental):
        raise ValueError("Incremental projection training invalidates frozen GeometryBank coordinates.")
    forbidden_bool = [
        "use_geometry_transport",
        "use_sglat_transport",
        "allow_old_model_transport",
        "use_energy_calibrator",
        "use_adaptive_boundary",
        "use_incremental_adapter",
        "use_geometry_calibrator",
        "use_bicyc_geometry_cycle",
        "use_mssl_loss",
    ]
    bad = [k for k in forbidden_bool if bool(getattr(args, k, False))]
    if bad:
        raise ValueError(f"These flags are not part of PG-RGA main path and must be false: {bad}")
    if bool(args.spectral_summary_is_physical) and (not bool(args.no_pca)) and int(args.pca_components) > 0:
        raise ValueError("PCA summaries cannot be marked physical.")
    for key in ("base_ce_weight", "base_srpgr_weight", "base_gics_weight", "pgr_weight", "pgr_compact_weight", "pgr_center_weight", "pgr_subspace_weight", "pgr_band_weight", "pgr_volume_weight"):
        if float(getattr(args, key)) <= 0.0:
            raise ValueError(f"{key} must be > 0 in mandatory base phase.")
    if float(args.adapter_max_scale) <= 0.0:
        raise ValueError("PG-RGA requires --adapter_max_scale > 0.")
    if num_classes is not None:
        if int(num_classes) != int(DATASET_INFO[args.dataset]["classes"]):
            raise ValueError(f"Loaded class count {num_classes} does not match DATASET_INFO[{args.dataset}]={DATASET_INFO[args.dataset]['classes']}.")
        if args.base_classes is not None and (int(args.base_classes) <= 0 or int(args.base_classes) >= int(num_classes)):
            raise ValueError(f"base_classes={args.base_classes} must be in [1,{num_classes - 1}].")
        if args.increment is not None and int(args.increment) <= 0:
            raise ValueError("--increment must be positive.")
    seeds = parse_seed_list(getattr(args, "seed_list", ""))
    if seeds is not None and len(seeds) != int(args.num_runs):
        raise ValueError(f"--seed_list has {len(seeds)} seeds but --num_runs={args.num_runs}.")
    if int(args.num_runs) <= 0:
        raise ValueError("--num_runs must be >= 1.")


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
        paths["unknown_args"] = save_json(os.path.join(save_root, "legacy_unknown_args_ignored.json"), {"unknown_args": list(resolved._unknown_args)})
    return paths


def print_config_summary(args: argparse.Namespace, diff: List[Dict[str, Any]], method_identity: Dict[str, Any]) -> None:
    print("[Method]", method_identity["method_name"])
    print(f"[Method] short_name={method_identity['short_name']} | incremental_update_mode={args.incremental_update_mode}")
    print(f"[Classifier] base={args.base_classifier_mode} | incremental={args.incremental_classifier_mode} | eval={args.eval_classifier_mode}")
    print(f"[Base] CE={args.base_ce_weight} | GICS={args.base_gics_weight} | PGR={args.pgr_weight} | class_balance={args.base_class_balance}")
    print(f"[Incremental] replay/class={args.gfa_samples_per_class} | adapter_max_scale={args.adapter_max_scale} | old_new_margin={args.old_new_geometry_margin}")
    print("[Forbidden OFF] transport=False | calibrator=False | adaptive_boundary=False | KD=False | raw_exemplars=False")
    if getattr(args, "_unknown_args", None):
        print(f"[LEGACY IGNORED] unknown CLI args: {args._unknown_args}")
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


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = NECILModel(args).to(device)
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
    for required in ("extract_projected_features", "get_subspace_bank"):
        if not hasattr(model, required):
            raise RuntimeError(f"Model missing required API: {required}")
    if hasattr(model, "incremental_update_mode"):
        model.incremental_update_mode = "geometry_gated_adapter"
    if hasattr(model, "use_geometry_gated_adapter"):
        model.use_geometry_gated_adapter = True
    if hasattr(model, "use_geometry_calibrator"):
        model.use_geometry_calibrator = False
    if hasattr(model, "use_incremental_adapter"):
        model.use_incremental_adapter = False
    print("[Model]")
    print(f"  feature_dim={args.d_model} | subspace_rank={args.subspace_rank} | classifier={type(model.classifier).__name__}")
    print("  incremental_plasticity=geometry_plastic_adapter | transport=False | calibrator=False")
    return model


def _canonical_method_value(name: str, value: Any) -> Any:
    """Canonicalize semantically equivalent method values for identity checks."""
    if name in {"base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "classifier_mode"}:
        return normalize_classifier_mode(value)
    if name == "incremental_update_mode":
        return normalize_incremental_update_mode(value)
    if name == "use_geometry_gated_adapter":
        return bool(value)
    return value


def build_trainer(model: torch.nn.Module, dataset: Any, args: argparse.Namespace, run_dir: str) -> Trainer:
    before = namespace_to_dict(args)
    trainer = Trainer(model, dataset, args)
    after = namespace_to_dict(args)
    diff = compute_config_diff(before, after, {})
    save_json(os.path.join(run_dir, "config_diff_after_trainer.json"), diff)
    critical = {"incremental_update_mode", "base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "use_geometry_gated_adapter"}
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
    old_classes = sorted(set(old_classes))
    seen_classes = old_classes + new_classes
    if len(seen_classes) != len(set(seen_classes)):
        raise RuntimeError(f"Duplicate class in seen_classes at phase {phase}: {seen_classes}")
    return {"phase": phase, "old_classes": old_classes, "new_classes": new_classes, "seen_classes": seen_classes, "old_class_count": len(old_classes)}


def set_model_phase(model: torch.nn.Module, phase_info: Dict[str, Any]) -> None:
    phase = int(phase_info["phase"])
    old_count = int(len(phase_info.get("old_classes", [])))
    if hasattr(model, "set_phase"):
        model.set_phase(phase)
    else:
        model.current_phase = phase
    if hasattr(model, "set_old_class_count"):
        model.set_old_class_count(old_count)
    else:
        model.old_class_count = old_count


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


def forward_eval_batch(model: torch.nn.Module, patches: torch.Tensor, spectra: Optional[torch.Tensor], args: argparse.Namespace, seen_classes: List[int]) -> Dict[str, Any]:
    spectral_summary, spectral_is_physical, spec_diag = prepare_eval_spectra(patches, spectra, args)
    kwargs = dict(
        seen_classes=[int(c) for c in seen_classes],
        classifier_mode=normalize_classifier_mode(args.eval_classifier_mode),
        return_energy=True,
        spectral_summary=spectral_summary,
        spectral_summary_is_physical=bool(spectral_is_physical),
    )
    try:
        out = model(patches, **kwargs)
    except TypeError:
        kwargs.pop("spectral_summary", None)
        kwargs.pop("spectral_summary_is_physical", None)
        out = model(patches, **kwargs)
    if not isinstance(out, dict):
        out = {"logits": out}
    out["spectral_diagnostics"] = spec_diag
    return out


def logits_to_global_predictions(logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
    seen = torch.as_tensor([int(c) for c in seen_classes], device=logits.device, dtype=torch.long)
    if logits.dim() != 2:
        raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
    if logits.size(1) == seen.numel():
        pred_local = logits.argmax(dim=1)
        return seen[pred_local]
    if seen.numel() > 0 and int(seen.max().item()) < logits.size(1):
        logits_seen = logits.index_select(1, seen)
        pred_local = logits_seen.argmax(dim=1)
        return seen[pred_local]
    raise RuntimeError(f"Cannot map logits width={logits.size(1)} to seen_classes={seen.detach().cpu().tolist()}")


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
    loader = dataset.get_cumulative_dataloader(phase, split="test", batch_size=batch_size, shuffle=False)
    preds: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    pred_hist: Dict[int, int] = {}
    for batch in loader:
        patches, labels, spectra, _ = unpack_eval_batch(batch)
        patches = patches.to(device, non_blocking=True).float()
        if torch.is_tensor(spectra):
            spectra = spectra.to(device, non_blocking=True)
        labels_t = labels.to(device).long().view(-1) if torch.is_tensor(labels) else torch.as_tensor(labels, device=device).long().view(-1)
        if not set(labels_t.detach().cpu().tolist()).issubset(set(seen_classes)):
            bad = sorted(set(labels_t.detach().cpu().tolist()) - set(seen_classes))
            raise RuntimeError(f"Evaluation labels outside seen classes at phase {phase}: {bad}")
        out = forward_eval_batch(model, patches, spectra, args, seen_classes)
        pred_global = logits_to_global_predictions(out["logits"], seen_classes)
        if not set(pred_global.detach().cpu().tolist()).issubset(set(seen_classes)):
            raise RuntimeError("Evaluation produced unseen predictions after seen-class mapping.")
        for p in pred_global.detach().cpu().tolist():
            pred_hist[int(p)] = pred_hist.get(int(p), 0) + 1
        preds.append(pred_global.detach().cpu().numpy())
        labels_all.append(labels_t.detach().cpu().numpy())
    if not preds:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {"prediction_histogram": {}}
    return np.concatenate(preds), np.concatenate(labels_all), {"prediction_histogram": pred_hist}


def compute_phase_metrics(y_true: np.ndarray, y_pred: np.ndarray, phase_info: Dict[str, Any]) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"overall_accuracy": 0.0, "per_class_accuracy": {}, "old_accuracy": 0.0, "new_accuracy": 0.0, "hm": 0.0}
    seen = [int(c) for c in phase_info["seen_classes"]]
    old = [int(c) for c in phase_info["old_classes"]]
    new = [int(c) for c in phase_info["new_classes"]]
    if not set(y_true.tolist()).issubset(set(seen)) or not set(y_pred.tolist()).issubset(set(seen)):
        raise RuntimeError("Metric labels/predictions outside seen classes.")
    overall = 100.0 * float((y_true == y_pred).mean())
    per_class: Dict[int, float] = {}
    for c in seen:
        mask = y_true == int(c)
        per_class[int(c)] = 100.0 * float((y_pred[mask] == c).mean()) if mask.any() else 0.0
    if old and new:
        old_mask = np.isin(y_true, np.asarray(old))
        new_mask = np.isin(y_true, np.asarray(new))
        old_acc = 100.0 * float((y_pred[old_mask] == y_true[old_mask]).mean()) if old_mask.any() else 0.0
        new_acc = 100.0 * float((y_pred[new_mask] == y_true[new_mask]).mean()) if new_mask.any() else 0.0
        hm = 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-8)
    else:
        old_acc = 0.0
        new_acc = overall
        hm = overall
    cm = np.zeros((len(seen), len(seen)), dtype=np.int64)
    pos = {c: i for i, c in enumerate(seen)}
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        cm[pos[int(yt)], pos[int(yp)]] += 1
    return {
        "overall_accuracy": overall,
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "hm": hm,
        "per_class_accuracy": per_class,
        "seen_classes": seen,
        "old_classes": old,
        "new_classes": new,
        "confusion_matrix": cm,
    }


def evaluator_update_compat(evaluator: Any, phase: int, y_true: np.ndarray, y_pred: np.ndarray, old_class_count: int, seen_classes: List[int]) -> None:
    sig = inspect.signature(evaluator.update)
    kwargs = {}
    if "old_class_count" in sig.parameters:
        kwargs["old_class_count"] = int(old_class_count)
    if "seen_classes" in sig.parameters:
        kwargs["seen_classes"] = seen_classes
    evaluator.update(int(phase), y_true, y_pred, **kwargs)


def save_classification_report_compat(
    evaluator: Any,
    phase: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    phase_dir: str,
    seen_classes: List[int],
    old_class_count: int,
    enabled: bool,
    tr_time: float,
    te_time: float,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    os.makedirs(phase_dir, exist_ok=True)
    if hasattr(evaluator, "save_phase_report"):
        return evaluator.save_phase_report(
            phase=int(phase),
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            save_dir=phase_dir,
            seen_classes=seen_classes,
            old_class_count=int(old_class_count),
            tr_time=tr_time,
            te_time=te_time,
            dl_time=0.0,
        )
    return save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        save_dir=phase_dir,
        phase=int(phase),
        seen_classes=seen_classes,
        old_class_count=int(old_class_count),
        tr_time=tr_time,
        te_time=te_time,
        dl_time=0.0,
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
        "feature_space": "canonical_projected_z in base; adapted_z only through geometry_plastic_adapter in incremental",
        "classifier_mode": args.incremental_classifier_mode if phase and phase > 0 else args.base_classifier_mode,
        "eval_mode": args.eval_classifier_mode,
        "trainable_policy": "base_backbone_projection_ce_head" if phase in {None, 0} else "geometry_plastic_adapter_only",
        "strict_non_exemplar": bool(args.strict_non_exemplar),
        "raw_exemplars_stored": False,
        "old_features_stored": False,
        "kd_teacher_used": False,
        "projection_trainable_during_incremental": False,
        "old_geometry_bank_frozen": True if phase and phase > 0 else None,
        "label_convention": "dataset/global, logits/seen-local, CE/seen-local, predictions/global",
        "transport_enabled": False,
        "old_row_transport_enabled": False,
        "adapter_enabled": True,
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
    if hasattr(model, "assert_base_handoff_ready"):
        phases = getattr(trainer.dataset, "phase_to_classes", None)
        try:
            base_ids = phase_to_classes_as_list(trainer.dataset)[0]
            model.assert_base_handoff_ready(base_ids, freeze=True, strict=True)
            return
        except Exception:
            pass
    handoff = getattr(model, "base_handoff", None) or getattr(trainer, "base_handoff", None) or getattr(trainer, "_last_base_handoff", None)
    cert = getattr(model, "base_geometry_certificate", None) or getattr(trainer, "_last_base_geometry_certificate", None)
    if handoff is None and cert is None:
        raise RuntimeError("Base phase completed without base handoff/certificate. Incremental phase cannot be trusted.")


def assert_incremental_phase_complete(model: torch.nn.Module, phase_info: Dict[str, Any]) -> None:
    if geometry_bank_state(model) is None:
        raise RuntimeError("Incremental phase ended without GeometryBank state.")
    if int(phase_info["phase"]) > 0 and not phase_info.get("old_classes"):
        raise RuntimeError("Incremental phase missing old classes in phase_info.")


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
        trainer._assert_incremental_preflight(phase, int(len(phase_info["old_classes"])))
        print(f"[Incremental Preflight PASS] phase={phase}")

    t0 = time.time()
    if phase == 0 and hasattr(trainer, "train_base_phase"):
        history = trainer.train_base_phase(phase=0, epochs=epochs, batch_size=args.batch_size, lr=lr)
        assert_base_handoff_ready(model, trainer)
    elif phase > 0 and hasattr(trainer, "train_incremental_phase"):
        history = trainer.train_incremental_phase(phase=phase, epochs=epochs, batch_size=args.batch_size, lr=lr)
        assert_incremental_phase_complete(model, phase_info)
    else:
        history = trainer.train_phase(phase=phase, epochs=epochs, batch_size=args.batch_size, lr=lr)
    train_time = time.time() - t0

    print(f"[Eval] phase={phase} cumulative seen-class evaluation")
    e0 = time.time()
    y_pred, y_true, pred_diag = get_phase_predictions(model, dataset, phase_info, torch.device(args.device), args, batch_size=args.batch_size)
    eval_time = time.time() - e0

    metrics = compute_phase_metrics(y_true, y_pred, phase_info)
    metrics["train_time_sec"] = train_time
    metrics["eval_time_sec"] = eval_time
    metrics["prediction_histogram"] = pred_diag.get("prediction_histogram", {})
    evaluator_update_compat(evaluator, phase, y_true, y_pred, int(len(phase_info["old_classes"])), phase_info["seen_classes"])
    if hasattr(evaluator, "print_summary"):
        evaluator.print_summary()
    report = save_classification_report_compat(
        evaluator=evaluator,
        phase=phase,
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        phase_dir=phase_dir,
        seen_classes=phase_info["seen_classes"],
        old_class_count=int(len(phase_info["old_classes"])),
        enabled=bool(args.save_classification_report),
        tr_time=train_time,
        te_time=eval_time,
    )
    diagnostics = collect_trainer_diagnostics(trainer, phase, phase_dir)
    diagnostics["prediction_histogram"] = pred_diag.get("prediction_histogram", {})
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
    set_seed(resolved.seed, deterministic=bool(resolved.deterministic))

    run_tag = "base_only" if bool(resolved.base_only) else "pg_rga"
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
    print("\n=== PG-RGA-HSI RUN ===")
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
    write_run_report(os.path.join(run_dir, "PG_RGA_HSI_RUN_REPORT.txt"), resolved, final_results)
    print(f"[Done] run_dir={run_dir} | final_OA={float(final_metrics.get('overall_accuracy', 0.0)):.2f} | final_HM={float(final_metrics.get('hm', 0.0)):.2f}")
    return final_results


def write_run_report(path: str, args: argparse.Namespace, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PG-RGA-HSI Run Report - {args.dataset}\n")
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
# import csv
# import inspect
# import json
# import os
# import random
# import sys
# import time
# from datetime import datetime
# from typing import Any, Dict, List, Optional

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


# def str2bool(v):
#     """Robust argparse bool parser.

#     Critical detail: bool("false") is True in Python. Do not use bool(...) for
#     CLI flags in this project; it silently activates forbidden incremental paths.
#     """
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


# def parse_seed_list(seed_list_str: str):
#     if seed_list_str is None or str(seed_list_str).strip() == "":
#         return None
#     return [int(s.strip()) for s in str(seed_list_str).split(",") if s.strip()]



# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=(
#             "SGLAT-HSI training: phase 0 builds spectral-residual prospective geometry (SRPGR); incremental phases use spectral-guided low-rank affine transport, risk-gated candidate descriptor admission, and boundary-preserving descriptor optimization."
#         )
#     )

#     # Dataset.
#     parser.add_argument("--dataset", type=str, default="IP", choices=DATASET_INFO.keys())
#     parser.add_argument("--data_dir", type=str, default="./datasets")
#     parser.add_argument("--save_dir", type=str, default="./checkpoints")
#     parser.add_argument("--patch_size", type=int, default=11)
#     parser.add_argument("--train_ratio", type=float, default=0.2)
#     parser.add_argument("--val_ratio", type=float, default=0.1)
#     parser.add_argument("--min_train_per_class", type=int, default=20)

#     # Protocol.
#     parser.add_argument("--base_classes", type=int, default=None)
#     parser.add_argument("--increment", type=int, default=None)
#     parser.add_argument("--strict_non_exemplar", type=str2bool, default=True)
#     parser.add_argument("--base_only", type=str2bool, default=False)
#     parser.add_argument(
#         "--max_train_phase",
#         type=int,
#         default=-1,
#         help="Stop after this phase index. Example: 2 runs phases 0,1,2 only.",
#     )
#     parser.add_argument(
#         "--max_phases",
#         type=int,
#         default=0,
#         help="Number of phases to run. Example: 3 runs phases 0,1,2 only.",
#     )

#     # Preprocessing.
#     parser.add_argument("--no_pca", action="store_true")
#     parser.add_argument("--pca_components", type=int, default=30)
#     parser.add_argument("--reduction_method", type=str, default="PCA")

#     # Backbone/model.
#     parser.add_argument("--d_model", type=int, default=128)
#     parser.add_argument("--d_state", type=int, default=16)
#     parser.add_argument("--d_conv", type=int, default=4)
#     parser.add_argument("--expand", type=int, default=2)
#     parser.add_argument("--num_spectral_layers", type=int, default=3)
#     parser.add_argument("--num_layers", type=int, default=3)
#     parser.add_argument("--dropout", type=float, default=0.1)
#     parser.add_argument("--projection_dropout", type=float, default=0.1)
#     parser.add_argument("--backbone_norm", type=str, default="layer", choices=["layer", "rms"])
#     parser.add_argument("--stem_norm_groups", type=int, default=8)
#     parser.add_argument("--ssm_residual_scale_init", type=float, default=0.7)
#     parser.add_argument("--fusion_residual_scale", type=float, default=0.3)
#     parser.add_argument("--backbone_output_dropout", type=float, default=0.0)

#     # Clean low-rank GeometryBank / classifier.
#     parser.add_argument("--subspace_rank", type=int, default=5)
#     parser.add_argument("--geom_var_floor", type=float, default=5e-4)
#     parser.add_argument("--geometry_variance_shrinkage", type=float, default=0.25)
#     parser.add_argument("--geometry_max_variance_ratio", type=float, default=50.0)
#     parser.add_argument("--geometry_min_reliability", type=float, default=0.05)
#     parser.add_argument("--loss_scale", type=float, default=8.0)
#     parser.add_argument("--energy_normalize_by_dim", type=str2bool, default=True)
#     parser.add_argument("--reliability_energy_weight", type=float, default=0.05)
#     parser.add_argument("--residual_variance_scale", type=float, default=1.10)
#     parser.add_argument("--invalid_class_energy", type=float, default=1e6)
#     parser.add_argument("--geometry_normalize_logits", type=str2bool, default=False)
#     parser.add_argument("--geometry_logit_clip", type=float, default=0.0)
#     # Canonical z-space and spectral consistency contract.
#     parser.add_argument("--normalize_geometry_features", type=str2bool, default=True)
#     parser.add_argument("--geometry_feature_scale", type=float, default=0.0)
#     parser.add_argument("--geometry_feature_clamp", type=float, default=0.0)
#     parser.add_argument("--strict_feature_contract", type=str2bool, default=True)
#     parser.add_argument("--spectral_summary_mode", type=str, default="center", choices=["center", "mean"])
#     parser.add_argument("--min_band_mass", type=float, default=1e-8)
#     parser.add_argument("--center_reliability_energy", type=str2bool, default=True)
#     parser.add_argument("--reliability_band_weight", type=float, default=0.20)

#     # Covariance-volume/logdet energy used by classifier, loss, replay and diagnostics.
#     parser.add_argument("--use_logdet_energy", type=str2bool, default=True)
#     parser.add_argument("--logdet_energy_weight", type=float, default=0.05)
#     parser.add_argument("--logdet_normalize_by_dim", type=str2bool, default=True)
#     parser.add_argument("--center_logdet_energy", type=str2bool, default=True)


#     # Geometry reliability/rank.
#     parser.add_argument("--reliability_sample_alpha", type=float, default=20.0)
#     parser.add_argument("--reliability_sample_weight", type=float, default=0.30)
#     parser.add_argument("--reliability_rank_weight", type=float, default=0.20)
#     parser.add_argument("--reliability_compact_weight", type=float, default=0.50)
#     parser.add_argument("--rank_energy_threshold", type=float, default=0.90)
#     parser.add_argument("--rank_eigen_ratio_threshold", type=float, default=1e-2)
#     parser.add_argument("--min_active_rank", type=int, default=1)
#     parser.add_argument("--residual_fraction_floor", type=float, default=1e-6)

#     # SRGP spectral-residual geometry.  The spectral branch is active only when
#     # physical wavelength-ordered spectra are available; PCA channels are gated off.
#     parser.add_argument("--use_spectral_geometry", type=str2bool, default=True)
#     parser.add_argument("--max_charts_per_class", type=int, default=1)
#     parser.add_argument("--spectral_shape_weight", type=float, default=0.25)
#     parser.add_argument("--spectral_rank", type=int, default=None)
#     parser.add_argument("--spectral_variance_floor", type=float, default=1e-5)
#     parser.add_argument("--spectral_variance_shrinkage", type=float, default=0.05)
#     parser.add_argument("--spectral_max_variance_ratio", type=float, default=100.0)
#     parser.add_argument("--spectral_rank_energy_threshold", type=float, default=0.98)
#     parser.add_argument("--spectral_rank_eigen_ratio_threshold", type=float, default=1e-4)
#     parser.add_argument("--spectral_min_active_rank", type=int, default=1)
#     parser.add_argument("--spectral_energy_weight", type=float, default=0.05)
#     parser.add_argument("--spectral_derivative_weight", type=float, default=0.50)
#     parser.add_argument("--spectral_second_derivative_weight", type=float, default=0.25)
#     parser.add_argument("--spectral_require_physical_summary", type=str2bool, default=True)
#     parser.add_argument("--spectral_summary_is_physical", type=str2bool, default=False)
#     parser.add_argument("--raw_spectral_summary_is_physical", type=str2bool, default=True)
#     parser.add_argument("--allow_nonphysical_spectral_summary", type=str2bool, default=True)
#     parser.add_argument("--band_energy_weight", type=float, default=0.0)
#     parser.add_argument("--spectral_reliability_energy_weight", type=float, default=0.0)
#     parser.add_argument("--spectral_residual_variance_scale", type=float, default=1.0)
#     parser.add_argument("--band_normalize_by_dim", type=str2bool, default=True)
#     parser.add_argument("--require_spectral_for_dual", type=str2bool, default=False)

#     # Final dual reliability mixture.
#     parser.add_argument("--final_feature_reliability_weight", type=float, default=0.45)
#     parser.add_argument("--final_spectral_reliability_weight", type=float, default=0.35)
#     parser.add_argument("--final_band_reliability_weight", type=float, default=0.10)
#     parser.add_argument("--final_sample_reliability_weight", type=float, default=0.10)

#     # Base training.
#     parser.add_argument("--lr", type=float, default=5e-5)
#     parser.add_argument("--epochs_base", type=int, default=120)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--label_smoothing", type=float, default=0.0)
#     parser.add_argument("--ce_logit_clip", type=float, default=50.0)
#     parser.add_argument("--grad_clip_base", type=float, default=1.0)
#     parser.add_argument("--weight_decay", type=float, default=1e-4)

#     # Incremental phase.
#     # Clean incremental does not train backbone/projection. Epochs are retained for
#     # compatibility; descriptor refinement is the actual plasticity path.
#     parser.add_argument("--epochs_inc", type=int, default=30)
#     parser.add_argument("--lr_inc", type=float, default=0.0)

#     # Base memory/evaluation schedule.
#     parser.add_argument("--base_geometry_refresh_every", type=int, default=1)
#     parser.add_argument("--print_base_geometry_diagnostics", type=str2bool, default=True)
#     parser.add_argument("--save_geometry_diagnostics", type=str2bool, default=True)
#     parser.add_argument("--geometry_diag_anchors_per_class", type=int, default=64)
#     parser.add_argument("--geometry_diag_topk_pairs", type=int, default=20)
#     parser.add_argument("--geometry_diag_topk_bands", type=int, default=5)

#     # Base CE + Geometry-Involved Contrastive Separation.
#     parser.add_argument("--base_ce_weight", type=float, default=1.0)
#     parser.add_argument("--base_gics_weight", type=float, default=0.20)
#     parser.add_argument("--base_gics_temperature", type=float, default=0.07)
#     # Legacy GICS spectral/band knobs are accepted but forced off. In the clean
#     # method, GICS is feature-only; PGR handles real band-shaping.
#     parser.add_argument("--base_gics_spectral_temperature", type=float, default=0.20)
#     parser.add_argument("--base_gics_band_temperature", type=float, default=0.20)
#     parser.add_argument("--base_gics_feature_weight", type=float, default=1.0)
#     parser.add_argument("--base_gics_spectral_weight", type=float, default=0.0)
#     parser.add_argument("--base_gics_band_weight", type=float, default=0.0)
#     parser.add_argument("--base_gics_same_class_positive", type=str2bool, default=True)
#     parser.add_argument("--base_gics_class_balanced", type=str2bool, default=True)
#     parser.add_argument("--base_gics_normalize", type=str2bool, default=True)
#     parser.add_argument("--base_gics_key_noise_std", type=float, default=0.0)
#     parser.add_argument("--base_gics_key_band_drop", type=float, default=0.0)
#     parser.add_argument("--base_gics_key_spatial_drop", type=float, default=0.0)
#     parser.add_argument("--base_gics_key_scale_jitter", type=float, default=0.0)

#     # MSSL-inspired Spatial-Spectral Manifold Regularization (SSMR).
#     # This is not the full MSSL paper objective. It uses the paper's HSI-aware
#     # spatial-spectral neighbor construction as a bounded NECIL-safe regularizer
#     # over the canonical projected geometry feature z.
#     parser.add_argument("--use_mssl_loss", type=str2bool, default=False)
#     parser.add_argument("--unsafe_ablation_use_mssl_loss", type=str2bool, default=False)
#     parser.add_argument("--mssl_loss_type", type=str, default="margin", choices=["margin", "signed"])
#     parser.add_argument("--mssl_weight", type=float, default=0.0)
#     parser.add_argument("--mssl_inc_weight", type=float, default=0.0)
#     parser.add_argument("--mssl_margin", type=float, default=1.0)
#     parser.add_argument("--mssl_temperature", type=float, default=0.20)
#     parser.add_argument("--mssl_neg_k", type=int, default=4)
#     parser.add_argument("--mssl_spatial_radius", type=float, default=2.0)
#     parser.add_argument("--mssl_same_label_positive", type=str2bool, default=True)
#     parser.add_argument("--mssl_use_labels_for_negatives", type=str2bool, default=True)
#     parser.add_argument("--mssl_signed_neg_weight", type=float, default=0.05)


#     # Prospective Geometry Reserve (PGR): base-space preparation for future HSI classes.
#     parser.add_argument("--use_prospective_geometry_reserve", type=str2bool, default=True)
#     parser.add_argument("--pgr_weight", type=float, default=0.10)
#     parser.add_argument("--pgr_compact_weight", type=float, default=0.15)
#     parser.add_argument("--pgr_center_weight", type=float, default=0.20)
#     parser.add_argument("--pgr_subspace_weight", type=float, default=0.10)
#     parser.add_argument("--pgr_spectral_weight", type=float, default=0.0)
#     parser.add_argument("--pgr_band_weight", type=float, default=0.05)
#     parser.add_argument("--pgr_volume_weight", type=float, default=0.05)
#     parser.add_argument("--base_srpgr_weight", type=float, default=1.0)
#     parser.add_argument("--base_spectral_shape_weight", type=float, default=0.05)
#     parser.add_argument("--base_spectral_shape_overlap_max", type=float, default=0.75)
#     parser.add_argument("--base_spectral_shape_require_physical", type=str2bool, default=True)
#     parser.add_argument("--base_spectral_shape_risk_weight", type=float, default=1.0)
#     parser.add_argument("--pgr_center_margin", type=float, default=1.05)
#     parser.add_argument("--pgr_spectral_margin", type=float, default=0.75)
#     parser.add_argument("--pgr_band_overlap_max", type=float, default=0.65)
#     parser.add_argument("--pgr_min_class_samples", type=int, default=3)
#     parser.add_argument("--pgr_subspace_min_samples", type=int, default=6)
#     parser.add_argument("--pgr_subspace_rank", type=int, default=3)
#     parser.add_argument("--pgr_max_class_variance", type=float, default=0.75)
#     parser.add_argument("--pgr_normalize_features", type=str2bool, default=True)

#     # Classifier modes for SRGP. Synthetic replay still uses geometry_only inside
#     # incremental_phase_trainer.py because replay features do not carry spectra.
#     parser.add_argument("--base_classifier_mode", type=str, default="srgp")
#     parser.add_argument("--incremental_classifier_mode", type=str, default="geometry_only")
#     parser.add_argument("--eval_classifier_mode", type=str, default="geometry_only")
#     parser.add_argument("--use_geometry_calibrator", type=str2bool, default=False)
#     parser.add_argument("--geometry_calibration_hidden_dim", type=int, default=128)
#     parser.add_argument("--geometry_calibration_dropout", type=float, default=0.0)
#     parser.add_argument("--geometry_calibration_weight", type=float, default=0.0)
#     parser.add_argument("--geometry_max_mean_scale", type=float, default=0.08)
#     parser.add_argument("--geometry_max_var_scale", type=float, default=0.08)

#     # Validation/checkpoint.
#     parser.add_argument("--refresh_before_validation", type=str2bool, default=True)
#     parser.add_argument("--validation_refresh_every", type=int, default=1)
#     parser.add_argument("--bank_refresh_every", type=int, default=1)
#     parser.add_argument(
#         "--best_state_metric",
#         type=str,
#         default="geometry_score",
#         choices=["acc", "oa", "hm", "h", "harmonic", "geometry_score", "geo", "geo_score"],
#     )
#     # Accepted only for backward-compatible commands. The clean NECIL path never
#     # early-stops: base GeometryBank quality must be learned for the full budget,
#     # and incremental phases must report their full trajectory for diagnosis.
#     parser.add_argument("--early_stop_patience", type=int, default=0)

#     # Base geometry certificate: base accuracy alone is not enough for NECIL.
#     parser.add_argument("--enforce_base_geometry_certificate", type=str2bool, default=False)
#     parser.add_argument("--base_cert_min_geom_acc", type=float, default=90.0)
#     parser.add_argument("--base_cert_min_reliability", type=float, default=0.15)
#     parser.add_argument("--base_cert_min_mean_reliability", type=float, default=0.35)
#     parser.add_argument("--base_cert_max_subspace_overlap", type=float, default=0.65)
#     parser.add_argument("--base_cert_max_geometry_conflict", type=float, default=2.0)
#     parser.add_argument("--base_cert_max_band_similarity", type=float, default=0.98)
#     parser.add_argument("--base_cert_max_spectral_shape_similarity", type=float, default=0.90)


#     # Compatibility fields expected by older model/trainer modules.
#     parser.add_argument("--num_concepts_per_class", type=int, default=1)
#     parser.add_argument("--semantic_dropout", type=float, default=0.0)
#     parser.add_argument("--eval_semantic_mode", type=str, default="identity")
#     parser.add_argument("--freeze_semantic_encoder_during_incremental", type=str2bool, default=True)
#     parser.add_argument("--disable_semantic_in_incremental", type=str2bool, default=True)
#     parser.add_argument("--freeze_classifier_during_incremental", type=str2bool, default=True)
#     parser.add_argument("--allow_legacy_classifier_modes", type=str2bool, default=False)
#     parser.add_argument("--use_adaptive_fusion", type=str2bool, default=False)
#     parser.add_argument("--allow_incremental_projection_training", type=str2bool, default=False)
#     parser.add_argument("--freeze_projection_during_incremental", type=str2bool, default=True)
#     parser.add_argument("--unfreeze_last_backbone_during_incremental", type=str2bool, default=False)
#     parser.add_argument("--grad_clip_inc", type=float, default=0.5)
#     # Legacy adapter flags are still accepted for old commands, but the approved
#     # architecture-level plasticity path is controlled by --incremental_update_mode.
#     parser.add_argument("--use_incremental_adapter", type=str2bool, default=False)
#     parser.add_argument("--incremental_adapter_scale", type=float, default=0.0)
#     parser.add_argument("--incremental_adapter_normalize", type=str2bool, default=False)
#     parser.add_argument("--anchor_consistency_weight", type=float, default=0.0)

#     # G²RPA: Geometry-Gated Residual Plastic Adaptation. This is the only
#     # approved feature-plasticity path after base training. It trains a small
#     # residual adapter after canonical z, while backbone/projection/classifier
#     # and frozen old GeometryBank rows stay fixed.
#     parser.add_argument(
#         "--incremental_update_mode",
#         type=str,
#         default="geometry_gated_adapter",
#         choices=[
#             "scbgr", "scb-gr", "spectral_risk_boundary", "geometry_state_admission",
#             "descriptor_only", "rsgi", "clean",
#             "geometry_gated_adapter", "g2rpa", "gated_adapter", "geometry_adapter", "adapter",
#         ],
#     )
#     parser.add_argument("--adapter_bottleneck", type=int, default=32)
#     parser.add_argument("--adapter_max_scale", type=float, default=0.35)
#     parser.add_argument("--adapter_dropout", type=float, default=0.0)
#     parser.add_argument("--adapter_gate_bias_init", type=float, default=-3.0)
#     parser.add_argument("--adapter_lr", type=float, default=5e-4)
#     parser.add_argument("--adapter_weight_decay", type=float, default=0.0)
#     parser.add_argument("--g2rpa_adapter_weight", type=float, default=1.0)
#     parser.add_argument("--adapter_old_delta_weight", type=float, default=1.0)
#     parser.add_argument("--adapter_old_gate_weight", type=float, default=0.75)
#     parser.add_argument("--adapter_old_energy_weight", type=float, default=0.25)
#     parser.add_argument("--adapter_old_margin_weight", type=float, default=0.25)
#     parser.add_argument("--adapter_delta_weight", type=float, default=0.10)
#     parser.add_argument("--adapter_new_gate_weight", type=float, default=0.05)
#     parser.add_argument("--adapter_new_gate_target", type=float, default=0.25)
#     parser.add_argument("--adapter_new_gate_max_target", type=float, default=0.75)
#     parser.add_argument("--use_full_incremental_loss_stack", type=str2bool, default=False)
#     parser.add_argument("--allow_fixed_geometry_incremental", type=str2bool, default=True)
#     parser.add_argument("--use_descriptor_refinement", type=str2bool, default=False)

#     # Clean descriptor-only incremental refinement. These are the only main-path
#     # plasticity knobs after base: they update new GeometryBank rows, not network weights.
#     parser.add_argument("--refine_new_descriptors", type=str2bool, default=True)
#     parser.add_argument("--descriptor_refine_steps", type=int, default=50)
#     parser.add_argument("--descriptor_refine_lr", type=float, default=1e-3)
#     parser.add_argument("--descriptor_trust_weight", type=float, default=1.0)
#     parser.add_argument("--descriptor_refine_max_mean_shift", type=float, default=0.35)
#     parser.add_argument("--descriptor_refine_max_logvar_shift", type=float, default=0.70)
#     parser.add_argument("--descriptor_refine_grad_clip", type=float, default=1.0)

#     # Incremental objective knobs for clean geometry replay.
#     parser.add_argument("--gfa_weight", type=float, default=1.0)
#     parser.add_argument("--synthetic_replay_weight", type=float, default=0.80)
#     parser.add_argument("--gfa_samples_per_class", type=int, default=32)
#     parser.add_argument("--synthetic_replay_per_class", type=int, default=16)
#     parser.add_argument("--gfa_parallel_scale", type=float, default=1.0)
#     parser.add_argument("--gfa_residual_scale", type=float, default=0.30)
#     parser.add_argument("--joint_old_new_ce_weight", type=float, default=1.0)
#     parser.add_argument("--bss_weight", type=float, default=0.0)
#     parser.add_argument("--sym_bss_weight", type=float, default=0.0)
#     parser.add_argument("--gdr_weight", type=float, default=0.0)
#     parser.add_argument("--bss_margin", type=float, default=5.0)
#     parser.add_argument("--risk_margin_scale", type=float, default=3.0)
#     parser.add_argument("--bss_reliability_weighted", type=str2bool, default=True)
#     parser.add_argument("--sym_bss_old_anchor_weight", type=float, default=1.0)
#     parser.add_argument("--sym_bss_new_sample_weight", type=float, default=1.0)
#     parser.add_argument("--gdr_mean_weight", type=float, default=1.0)
#     parser.add_argument("--gdr_basis_weight", type=float, default=1.0)
#     parser.add_argument("--gdr_variance_weight", type=float, default=1.0)
#     parser.add_argument("--gdr_reliability_weighted", type=str2bool, default=True)
#     parser.add_argument("--center_risk_weight", type=float, default=0.35)
#     parser.add_argument("--subspace_risk_weight", type=float, default=0.35)
#     parser.add_argument("--spectral_center_risk_weight", type=float, default=0.15)
#     parser.add_argument("--spectral_subspace_risk_weight", type=float, default=0.10)
#     parser.add_argument("--band_risk_weight", type=float, default=0.05)
#     parser.add_argument("--use_pretrain_incremental_baseline", type=str2bool, default=True)

#     # Clean incremental geometry-energy losses.
#     parser.add_argument("--geometry_energy_margin_weight", type=float, default=0.25)
#     parser.add_argument("--geometry_energy_margin", type=float, default=0.25)
#     parser.add_argument("--old_new_invasion_weight", type=float, default=0.35)
#     parser.add_argument("--invasion_weight", type=float, default=0.10)
#     parser.add_argument("--old_new_geometry_margin", type=float, default=0.30)
#     parser.add_argument("--incremental_weight_anchor", type=float, default=0.0)

#     # SGLAT boundary refinement + candidate-admission controls.
#     parser.add_argument("--use_boundary_geometry_replay", type=str2bool, default=True)
#     parser.add_argument("--boundary_replay_risk_threshold", type=float, default=0.35)
#     parser.add_argument("--boundary_replay_overlap_threshold", type=float, default=0.30)
#     parser.add_argument("--boundary_replay_samples_per_pair", type=int, default=12)
#     parser.add_argument("--boundary_replay_max_pairs", type=int, default=24)
#     parser.add_argument("--boundary_replay_parallel_scale", type=float, default=0.15)
#     parser.add_argument("--boundary_replay_residual_scale", type=float, default=0.05)
#     parser.add_argument("--scbgr_commit_only_if_safe", type=str2bool, default=True)
#     parser.add_argument("--unified_loss_weight", type=float, default=1.0)
#     # Backward-compatible aliases consumed by IncrementalPhaseTrainer.  These
#     # were used in the boundary-valid command; without parser support argparse
#     # exits before training.  They are real loss weights, not random flags.
#     parser.add_argument("--unified_admission_weight", type=float, default=0.70)
#     parser.add_argument("--unified_subspace_weight", type=float, default=0.40)
#     parser.add_argument("--unified_rank_weight", type=float, default=0.25)
#     parser.add_argument("--unified_volume_weight", type=float, default=0.03)
#     parser.add_argument("--unified_trust_weight", type=float, default=1.0)

#     # Fixed SGLAT incremental controls: transport, candidate admission, and safe new-state commit.
#     parser.add_argument("--strict_updated_stack", type=str2bool, default=True)
#     parser.add_argument("--use_risk_weighted_replay", type=str2bool, default=False)
#     parser.add_argument("--risk_replay_min_samples", type=int, default=4)
#     parser.add_argument("--risk_replay_max_multiplier", type=float, default=3.0)
#     parser.add_argument("--risk_replay_reliability_gated", type=str2bool, default=True)
#     parser.add_argument("--risk_replay_reliability_weighted", type=str2bool, default=True)
#     parser.add_argument("--risk_center_margin", type=float, default=1.0)
#     parser.add_argument("--risk_subspace_weight", type=float, default=1.0)
#     parser.add_argument("--risk_band_weight", type=float, default=0.25)
#     parser.add_argument("--risk_spectral_shape_weight", type=float, default=0.25)
#     parser.add_argument("--old_new_risk_spectral_shape_weight", type=float, default=0.25)
#     parser.add_argument("--risk_chart_weight", type=float, default=0.0)
#     parser.add_argument("--gfa_reliability_gated", type=str2bool, default=True)

#     parser.add_argument("--reliability_gated_admission", type=str2bool, default=True)
#     parser.add_argument("--admission_min_gate", type=float, default=0.35)
#     parser.add_argument("--admission_shrink_floor", type=float, default=0.15)
#     parser.add_argument("--admission_low_rank_cap", type=int, default=2)

#     # RSGI active high-risk descriptor correction. These names match the updated
#     # incremental_phase_trainer.py implementation.
#     parser.add_argument("--risk_aware_descriptor_correction", type=str2bool, default=True)
#     parser.add_argument("--descriptor_correction_risk_threshold", type=float, default=0.35)
#     parser.add_argument("--descriptor_correction_overlap_threshold", type=float, default=0.30)
#     parser.add_argument("--descriptor_correction_basis_strength", type=float, default=0.85)
#     parser.add_argument("--descriptor_correction_mean_push", type=float, default=0.20)
#     parser.add_argument("--descriptor_correction_var_shrink", type=float, default=0.15)
#     parser.add_argument("--descriptor_correction_topk_old", type=int, default=3)
#     # Backward-compatible aliases accepted by older commands/trainer variants.
#     parser.add_argument("--descriptor_correction_subspace_eta", type=float, default=0.85)
#     parser.add_argument("--descriptor_correction_center_step", type=float, default=0.20)
#     parser.add_argument("--descriptor_correction_variance_shrink", type=float, default=0.15)
#     parser.add_argument("--risk_sep_weight", type=float, default=0.30)
#     parser.add_argument("--risk_sep_overlap_target", type=float, default=0.35)
#     parser.add_argument("--risk_sep_active_threshold", type=float, default=0.50)

#     parser.add_argument("--descriptor_refine_steps_per_epoch", type=int, default=None)
#     parser.add_argument("--descriptor_subspace_collision_weight", type=float, default=0.20)
#     parser.add_argument("--descriptor_overlap_target", type=float, default=0.35)
#     parser.add_argument("--descriptor_subspace_overlap_max", type=float, default=0.35)
#     parser.add_argument("--descriptor_center_collision_weight", type=float, default=0.05)
#     parser.add_argument("--descriptor_center_margin_weight", type=float, default=0.05)
#     parser.add_argument("--descriptor_center_margin", type=float, default=1.0)
#     parser.add_argument("--descriptor_volume_control_weight", type=float, default=0.03)
#     parser.add_argument("--descriptor_volume_weight", type=float, default=0.03)

#     # Boundary-preserving descriptor optimization. These flags are consumed by
#     # IncrementalPhaseTrainer._old_new_boundary_preservation_loss() and
#     # _project_new_descriptor_params_out_of_old_tangent_space().  This is the
#     # valid all-phase fix: constrain the new descriptor while it is being learned,
#     # instead of only warning after old/new overlap has already appeared.
#     parser.add_argument("--boundary_preserve_weight", type=float, default=0.35)
#     parser.add_argument("--boundary_preserve_overlap_weight", type=float, default=1.0)
#     parser.add_argument("--boundary_preserve_center_weight", type=float, default=0.50)
#     parser.add_argument("--boundary_preserve_volume_weight", type=float, default=0.25)
#     parser.add_argument("--boundary_preserve_band_weight", type=float, default=0.10)
#     parser.add_argument("--max_old_new_risk", type=float, default=0.60)
#     parser.add_argument("--max_old_new_overlap", type=float, default=0.65)
#     parser.add_argument("--use_boundary_projection", type=str2bool, default=True)
#     parser.add_argument("--boundary_projection_strength", type=float, default=0.35)
#     parser.add_argument("--boundary_projection_mean_push", type=float, default=0.05)
#     parser.add_argument("--boundary_projection_var_shrink", type=float, default=0.05)
#     parser.add_argument("--boundary_projection_overlap_threshold", type=float, default=0.65)
#     parser.add_argument("--boundary_projection_topk_old", type=int, default=2)


#     # Removed BiCyc/feature-cycle path. Args are accepted for old commands but
#     # forced off by normalize_args(); use a separate ablation script if needed.
#     parser.add_argument("--use_bicyc_geometry_cycle", type=str2bool, default=False)
#     parser.add_argument("--bicyc_cycle_weight", type=float, default=0.0)
#     parser.add_argument("--bicyc_reg_weight", type=float, default=0.0)
#     parser.add_argument("--bicyc_hidden_ratio", type=float, default=0.5)
#     parser.add_argument("--bicyc_dropout", type=float, default=0.0)
#     parser.add_argument("--bicyc_max_delta_scale", type=float, default=0.10)
#     parser.add_argument("--bicyc_cycle_updates_projection", type=str2bool, default=False)

#     # Optional bounded score calibration; off for the main clean path unless explicitly enabled.
#     parser.add_argument("--use_energy_calibrator", type=str2bool, default=False)
#     parser.add_argument("--energy_calibrator_type", type=str, default="none", choices=["old_new", "per_class", "none"])
#     parser.add_argument("--energy_calibrator_max_log_scale", type=float, default=0.50)
#     parser.add_argument("--energy_calibrator_max_bias", type=float, default=2.0)
#     parser.add_argument("--energy_calibrator_max_classes", type=int, default=0)
#     parser.add_argument("--energy_calibration_weight", type=float, default=1e-3)
#     parser.add_argument("--use_measured_energy_calibration", type=str2bool, default=False)

#     # Adaptive decision boundary for GeometryBank energy.
#     # This is the ADBS-style component converted to HSI low-rank energy: each
#     # class owns a learnable decision radius rho_c; old radii are frozen and
#     # new radii adapt under old/new geometry-risk constraints.
#     parser.add_argument("--use_adaptive_boundary", type=str2bool, default=True)
#     parser.add_argument("--boundary_radius_min", type=float, default=0.50)
#     parser.add_argument("--boundary_radius_max", type=float, default=2.00)
#     parser.add_argument("--boundary_init_radius", type=float, default=1.00)
#     parser.add_argument("--boundary_radius_reg_weight", type=float, default=0.01)
#     parser.add_argument("--boundary_old_new_constraint_weight", type=float, default=0.20)
#     parser.add_argument("--boundary_old_new_margin_base", type=float, default=0.05)
#     parser.add_argument("--boundary_old_new_margin_scale", type=float, default=0.25)
#     parser.add_argument("--adaptive_boundary_loss_weight", type=float, default=1.00)
#     parser.add_argument("--adaptive_boundary_lr", type=float, default=1e-4)
#     parser.add_argument("--freeze_old_boundaries", type=str2bool, default=True)

#     # SGLAT-HSI transport. This is the main old-geometry calibration path.
#     # It estimates z_new ≈ z_old @ A + b from current-phase samples, then
#     # GeometryBank.transport_frozen_geometry() moves only compact old descriptors.
#     parser.add_argument("--use_sglat_transport", type=str2bool, default=False)
#     parser.add_argument("--use_geometry_transport", type=str2bool, default=False)
#     parser.add_argument("--allow_old_model_transport", type=str2bool, default=False)
#     parser.add_argument("--allow_transport_without_adapter", type=str2bool, default=False)
#     parser.add_argument("--transport_type", type=str, default="ridge", choices=["ridge", "gls"])
#     parser.add_argument("--transport_ridge", type=float, default=1e-3)
#     parser.add_argument("--transport_ema", type=float, default=0.97)
#     parser.add_argument("--transport_batches", type=int, default=20)
#     parser.add_argument("--transport_identity_blend", type=float, default=0.75)
#     parser.add_argument("--transport_low_rank", type=int, default=4)
#     parser.add_argument("--transport_after_adapter_epoch", type=int, default=3)
#     parser.add_argument("--transport_spectral_reliability_gate", type=str2bool, default=True)
#     parser.add_argument("--transport_min_reliability_gate", type=float, default=0.30)
#     parser.add_argument("--transport_max_a_minus_i_fro", type=float, default=1.5)
#     parser.add_argument("--transport_max_b_norm", type=float, default=0.75)
#     parser.add_argument("--transport_residual_scale", type=float, default=0.50)
#     parser.add_argument("--transport_min_rmse_gain", type=float, default=1e-5)
#     parser.add_argument("--transport_max_rmse_ratio", type=float, default=0.98)
#     parser.add_argument("--transport_min_old_anchor_acc", type=float, default=95.0)
#     parser.add_argument("--save_transport_diagnostics", type=str2bool, default=True)
#     parser.add_argument("--candidate_admission_mode", type=str, default="provisional", choices=["provisional", "shadow"])
#     parser.add_argument("--disable_incremental_adapter", type=str2bool, default=True)
#     parser.add_argument("--geometry_bank_canonical_space", type=str2bool, default=True)

#     # Reporting/maps.
#     parser.add_argument("--skip_phase_maps", type=str2bool, default=False)
#     parser.add_argument("--save_classification_report", type=str2bool, default=True)
#     parser.add_argument("--save_final_classification_report", type=str2bool, default=True)
#     parser.add_argument("--viz_class_cmap", type=str, default="nipy_spectral")
#     parser.add_argument("--viz_background_color", type=str, default="#20252B")
#     parser.add_argument("--viz_save_numpy", type=str2bool, default=True)

#     # Reproducibility/system.
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num_runs", type=int, default=1)
#     parser.add_argument("--seed_list", type=str, default="")
#     parser.add_argument("--deterministic", type=str2bool, default=False)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--subspace_extract_batch_size", type=int, default=256)
#     parser.add_argument("--debug_verbose", type=str2bool, default=False)

#     return parser.parse_args()


# def normalize_args(args):
#     """Normalize flags for the coherent SGLAT-HSI path.

#     Valid runtime contract:
#         Base:
#             temporary CE head + SRPGR over canonical projected z. GICS/PGR are
#             internal compact/reserve terms; physical spectral-shape discrimination
#             is active only when raw wavelength-ordered spectra are available.
#         Memory:
#             GeometryBank stores low-rank feature geometry, residual uncertainty,
#             reliability/count, band signature, and optional physical spectral-shape
#             descriptors.
#         Incremental:
#             freeze backbone/projection/old rows; snapshot the previous model for
#             current-sample old→new feature transport; transport compact old rows;
#             estimate provisional new descriptors; risk-correct/admit candidates;
#             then use boundary geometry refinement.

#     The main path has no BiCyc transport, no projection plasticity, no teacher/KD,
#     and no raw old exemplars. Synthetic replay never receives fake spectra.
#     """
#     args.base_only = bool(getattr(args, "base_only", False))
#     args.disable_incremental_training = bool(args.base_only)

#     # Classifier/memory contract. Real samples use SRGP; synthetic replay uses
#     # geometry_only inside the incremental trainer because replay features have no spectra.
#     args.base_classifier_mode = "srgp"
#     args.incremental_classifier_mode = "geometry_only"
#     args.eval_classifier_mode = "geometry_only"
#     args.geometry_bank_canonical_space = True
#     args.strict_non_exemplar = bool(getattr(args, "strict_non_exemplar", True))

#     # Incremental update policy. SGLAT-HSI is the main path. The recommended
#     # runtime is geometry_gated_adapter because transport is meaningful only when
#     # the current z-space has controlled plasticity. Legacy SCB-GR names remain aliases.
#     args.incremental_update_mode = _normalize_incremental_update_mode(
#         getattr(args, "incremental_update_mode", "scbgr")
#     )
#     adapter_mode = args.incremental_update_mode == "geometry_gated_adapter"

#     # SGLAT uses transport first, then candidate admission, then boundary refinement.
#     args.use_boundary_geometry_replay = bool(getattr(args, "use_boundary_geometry_replay", True))
#     args.boundary_replay_risk_threshold = float(getattr(args, "boundary_replay_risk_threshold", 0.35))
#     args.boundary_replay_overlap_threshold = float(getattr(args, "boundary_replay_overlap_threshold", 0.30))
#     args.boundary_replay_samples_per_pair = int(getattr(args, "boundary_replay_samples_per_pair", 12))
#     args.boundary_replay_max_pairs = int(getattr(args, "boundary_replay_max_pairs", 24))
#     args.boundary_replay_parallel_scale = float(getattr(args, "boundary_replay_parallel_scale", 0.15))
#     args.boundary_replay_residual_scale = float(getattr(args, "boundary_replay_residual_scale", 0.05))
#     args.scbgr_commit_only_if_safe = bool(getattr(args, "scbgr_commit_only_if_safe", True))
#     args.unified_loss_weight = float(getattr(args, "unified_loss_weight", 1.0))
#     # Unified incremental geometry-loss weights consumed by
#     # IncrementalPhaseTrainer._descriptor_refinement_epoch(). Keep these explicit
#     # so the command line, report, and trainer objective cannot silently diverge.
#     args.unified_admission_weight = float(getattr(args, "unified_admission_weight", 0.70))
#     args.unified_subspace_weight = float(getattr(args, "unified_subspace_weight", 0.40))
#     args.unified_rank_weight = float(getattr(args, "unified_rank_weight", getattr(args, "geometry_energy_margin_weight", 0.25)))
#     args.unified_volume_weight = float(getattr(args, "unified_volume_weight", getattr(args, "descriptor_volume_control_weight", 0.03)))
#     args.unified_trust_weight = float(getattr(args, "unified_trust_weight", getattr(args, "descriptor_trust_weight", 1.0)))

#     # SGLAT transport/admission defaults. These are architectural controls, not losses.
#     args.use_sglat_transport = bool(getattr(args, "use_sglat_transport", False))
#     args.use_geometry_transport = bool(getattr(args, "use_geometry_transport", False))
#     args.allow_old_model_transport = bool(getattr(args, "allow_old_model_transport", False))
#     args.allow_transport_without_adapter = bool(getattr(args, "allow_transport_without_adapter", False))
#     if args.use_sglat_transport:
#         args.use_geometry_transport = True
#         args.allow_old_model_transport = True
#     args.transport_type = str(getattr(args, "transport_type", "ridge")).lower().strip()
#     if args.transport_type not in {"ridge", "gls"}:
#         raise ValueError("--transport_type must be ridge or gls.")
#     args.transport_ridge = float(getattr(args, "transport_ridge", 1e-3))
#     args.transport_ema = float(getattr(args, "transport_ema", 0.97))
#     args.transport_batches = int(getattr(args, "transport_batches", 20))
#     args.transport_identity_blend = float(getattr(args, "transport_identity_blend", 0.75))
#     args.transport_low_rank = int(getattr(args, "transport_low_rank", 4))
#     args.transport_after_adapter_epoch = int(getattr(args, "transport_after_adapter_epoch", 3))
#     args.transport_spectral_reliability_gate = bool(getattr(args, "transport_spectral_reliability_gate", True))
#     args.transport_min_reliability_gate = float(getattr(args, "transport_min_reliability_gate", 0.30))
#     args.transport_max_a_minus_i_fro = float(getattr(args, "transport_max_a_minus_i_fro", 1.5))
#     args.transport_max_b_norm = float(getattr(args, "transport_max_b_norm", 0.75))
#     args.transport_residual_scale = float(getattr(args, "transport_residual_scale", 0.50))
#     args.transport_min_rmse_gain = float(getattr(args, "transport_min_rmse_gain", 1e-5))
#     args.transport_max_rmse_ratio = float(getattr(args, "transport_max_rmse_ratio", 0.98))
#     args.transport_min_old_anchor_acc = float(getattr(args, "transport_min_old_anchor_acc", 95.0))
#     args.save_transport_diagnostics = bool(getattr(args, "save_transport_diagnostics", True))
#     args.candidate_admission_mode = str(getattr(args, "candidate_admission_mode", "provisional")).lower().strip()
#     if args.candidate_admission_mode not in {"provisional", "shadow"}:
#         raise ValueError("--candidate_admission_mode must be provisional or shadow.")

#     # Canonical feature/spectral contract used by the updated model and bank.
#     args.normalize_geometry_features = bool(getattr(args, "normalize_geometry_features", True))
#     args.geometry_feature_scale = float(getattr(args, "geometry_feature_scale", 0.0))
#     args.geometry_feature_clamp = float(getattr(args, "geometry_feature_clamp", 0.0))
#     args.strict_feature_contract = bool(getattr(args, "strict_feature_contract", True))
#     args.spectral_summary_mode = str(getattr(args, "spectral_summary_mode", "center")).lower().strip()
#     if args.spectral_summary_mode not in {"center", "mean"}:
#         raise ValueError("--spectral_summary_mode must be 'center' or 'mean'.")
#     args.min_band_mass = float(getattr(args, "min_band_mass", 1e-8))
#     args.center_reliability_energy = bool(getattr(args, "center_reliability_energy", True))
#     args.reliability_band_weight = float(getattr(args, "reliability_band_weight", 0.20))

#     # The updated classifier/loss use covariance-consistent low-rank Gaussian energy.
#     args.use_logdet_energy = bool(getattr(args, "use_logdet_energy", True))
#     args.logdet_energy_weight = float(getattr(args, "logdet_energy_weight", 0.05))
#     args.logdet_normalize_by_dim = bool(getattr(args, "logdet_normalize_by_dim", True))
#     args.center_logdet_energy = bool(getattr(args, "center_logdet_energy", True))
#     args.geometry_normalize_logits = False
#     args.strict_updated_stack = bool(getattr(args, "strict_updated_stack", True))

#     # Adaptive decision boundary.  This is the actual ADBS-style mechanism for
#     # our GeometryBank classifier: adaptive class-wise energy radii, not cosine
#     # prototype margins.  It must remain enabled in the main path; old radii are
#     # frozen by the trainer and only new radii are trainable in incremental phases.
#     args.use_adaptive_boundary = bool(getattr(args, "use_adaptive_boundary", True))
#     args.boundary_radius_min = float(getattr(args, "boundary_radius_min", 0.50))
#     args.boundary_radius_max = float(getattr(args, "boundary_radius_max", 2.00))
#     args.boundary_init_radius = float(getattr(args, "boundary_init_radius", 1.00))
#     args.boundary_radius_reg_weight = float(getattr(args, "boundary_radius_reg_weight", 0.01))
#     args.boundary_old_new_constraint_weight = float(getattr(args, "boundary_old_new_constraint_weight", 0.20))
#     args.boundary_old_new_margin_base = float(getattr(args, "boundary_old_new_margin_base", 0.05))
#     args.boundary_old_new_margin_scale = float(getattr(args, "boundary_old_new_margin_scale", 0.25))
#     args.adaptive_boundary_loss_weight = float(getattr(args, "adaptive_boundary_loss_weight", 1.00))
#     args.adaptive_boundary_lr = float(getattr(args, "adaptive_boundary_lr", 1e-4))
#     args.freeze_old_boundaries = bool(getattr(args, "freeze_old_boundaries", True))

#     # SRGP spectral residual branch. It is safe with PCA because every trainer/model
#     # call gates the spectral term behind spectral_summary_is_physical=True.
#     args.use_spectral_geometry = True
#     args.spectral_energy_weight = float(getattr(args, "spectral_energy_weight", 0.05))
#     args.spectral_derivative_weight = float(getattr(args, "spectral_derivative_weight", 0.50))
#     args.spectral_second_derivative_weight = float(getattr(args, "spectral_second_derivative_weight", 0.25))
#     args.spectral_require_physical_summary = bool(getattr(args, "spectral_require_physical_summary", True))
#     args.spectral_summary_is_physical = bool(getattr(args, "spectral_summary_is_physical", False))
#     args.raw_spectral_summary_is_physical = bool(getattr(args, "raw_spectral_summary_is_physical", True))
#     args.allow_nonphysical_spectral_summary = bool(getattr(args, "allow_nonphysical_spectral_summary", True))
#     args.risk_spectral_shape_weight = float(getattr(args, "risk_spectral_shape_weight", 0.25))
#     args.old_new_risk_spectral_shape_weight = float(getattr(args, "old_new_risk_spectral_shape_weight", args.risk_spectral_shape_weight))
#     args.max_charts_per_class = int(getattr(args, "max_charts_per_class", 1))
#     args.spectral_shape_weight = float(getattr(args, "spectral_shape_weight", 0.25))

#     # Removed or non-core mechanisms: force off regardless of stale command line.
#     # This does not disable G²RPA; that path is controlled by incremental_update_mode
#     # and model.use_geometry_gated_adapter, not by the legacy use_incremental_adapter flag.
#     forced_false = [
#         "use_geometry_calibrator",
#         "use_incremental_adapter",
#         "incremental_adapter_normalize",
#         "use_full_incremental_loss_stack",
#         "use_bicyc_geometry_cycle",
#         "bicyc_cycle_updates_projection",
#         "use_descriptor_refinement",
#         "use_measured_energy_calibration",
#         "allow_incremental_projection_training",
#         "unfreeze_last_backbone_during_incremental",
#         "allow_legacy_classifier_modes",
#         "use_adaptive_fusion",
#     ]
#     for name in forced_false:
#         if hasattr(args, name):
#             setattr(args, name, False)

#     forced_zero = [
#         "geometry_calibration_weight",
#         "incremental_adapter_scale",
#         "anchor_consistency_weight",
#         "bss_weight",
#         "sym_bss_weight",
#         "gdr_weight",
#         "bicyc_cycle_weight",
#         "bicyc_reg_weight",
#         "band_energy_weight",
#         "spectral_reliability_energy_weight",
#         "pgr_spectral_weight",
#         "base_gics_spectral_weight",
#         "base_gics_band_weight",
#     ]
#     for name in forced_zero:
#         if hasattr(args, name):
#             setattr(args, name, 0.0)

#     # Legacy adapter flag: false in descriptor-only mode; not used by G²RPA.
#     # In G²RPA mode keep disable_incremental_adapter=False so stale model hooks do
#     # not accidentally switch off the approved geometry_plastic_adapter path.
#     args.disable_incremental_adapter = not adapter_mode
#     args.use_geometry_gated_adapter = bool(adapter_mode)
#     args.adapter_bottleneck = int(getattr(args, "adapter_bottleneck", 32))
#     args.adapter_max_scale = float(getattr(args, "adapter_max_scale", 0.35))
#     args.adapter_dropout = float(getattr(args, "adapter_dropout", 0.0))
#     args.adapter_gate_bias_init = float(getattr(args, "adapter_gate_bias_init", -3.0))
#     args.adapter_lr = float(getattr(args, "adapter_lr", 5e-4))
#     args.adapter_weight_decay = float(getattr(args, "adapter_weight_decay", 0.0))
#     args.g2rpa_adapter_weight = float(getattr(args, "g2rpa_adapter_weight", 1.0))
#     args.adapter_old_delta_weight = float(getattr(args, "adapter_old_delta_weight", 1.0))
#     args.adapter_old_gate_weight = float(getattr(args, "adapter_old_gate_weight", 0.75))
#     args.adapter_old_energy_weight = float(getattr(args, "adapter_old_energy_weight", 0.25))
#     args.adapter_old_margin_weight = float(getattr(args, "adapter_old_margin_weight", 0.25))
#     args.adapter_delta_weight = float(getattr(args, "adapter_delta_weight", 0.10 if adapter_mode else 0.0))
#     args.adapter_new_gate_weight = float(getattr(args, "adapter_new_gate_weight", 0.05))
#     args.adapter_new_gate_target = float(getattr(args, "adapter_new_gate_target", 0.25))
#     args.adapter_new_gate_max_target = float(getattr(args, "adapter_new_gate_max_target", 0.75))
#     args.freeze_projection_during_incremental = True
#     args.freeze_semantic_encoder_during_incremental = True
#     args.disable_semantic_in_incremental = True
#     args.eval_semantic_mode = "identity"
#     args.freeze_classifier_during_incremental = True
#     args.require_spectral_for_dual = False
#     args.spectral_residual_variance_scale = 1.0
#     if getattr(args, "spectral_rank", None) is None:
#         args.spectral_rank = int(args.subspace_rank)

#     # Clean base objective.
#     args.base_ce_weight = float(getattr(args, "base_ce_weight", 1.0))
#     args.base_gics_weight = float(getattr(args, "base_gics_weight", 0.20))
#     args.base_gics_feature_weight = 1.0

#     # MSSL is not part of the clean method. Permit only explicit unsafe ablation.
#     args.unsafe_ablation_use_mssl_loss = bool(getattr(args, "unsafe_ablation_use_mssl_loss", False))
#     args.use_mssl_loss = bool(getattr(args, "use_mssl_loss", False))
#     if args.use_mssl_loss and not args.unsafe_ablation_use_mssl_loss:
#         raise ValueError(
#             "--use_mssl_loss is not part of the clean descriptor-only method. "
#             "Use --unsafe_ablation_use_mssl_loss true only for a separate ablation run."
#         )
#     if not args.use_mssl_loss:
#         args.mssl_weight = 0.0
#         args.mssl_inc_weight = 0.0
#     args.mssl_loss_type = str(getattr(args, "mssl_loss_type", "margin")).lower().strip()
#     if args.mssl_loss_type not in {"margin", "signed"}:
#         args.mssl_loss_type = "margin"

#     # PGR remains the HSI-specific base-space reserve/shaping loss.
#     args.use_prospective_geometry_reserve = bool(getattr(args, "use_prospective_geometry_reserve", True))
#     args.pgr_weight = float(getattr(args, "pgr_weight", 0.10)) if args.use_prospective_geometry_reserve else 0.0
#     args.pgr_compact_weight = float(getattr(args, "pgr_compact_weight", 0.15))
#     args.pgr_center_weight = float(getattr(args, "pgr_center_weight", 0.20))
#     args.pgr_subspace_weight = float(getattr(args, "pgr_subspace_weight", 0.10))
#     args.pgr_band_weight = float(getattr(args, "pgr_band_weight", 0.05))
#     args.pgr_volume_weight = float(getattr(args, "pgr_volume_weight", 0.05))
#     args.base_srpgr_weight = float(getattr(args, "base_srpgr_weight", 1.0))
#     args.base_spectral_shape_weight = float(getattr(args, "base_spectral_shape_weight", 0.05))
#     args.base_spectral_shape_overlap_max = float(getattr(args, "base_spectral_shape_overlap_max", 0.75))
#     args.base_spectral_shape_require_physical = bool(getattr(args, "base_spectral_shape_require_physical", True))
#     args.base_spectral_shape_risk_weight = float(getattr(args, "base_spectral_shape_risk_weight", 1.0))
#     args.pgr_center_margin = float(getattr(args, "pgr_center_margin", 1.05))
#     args.pgr_spectral_margin = float(getattr(args, "pgr_spectral_margin", 0.75))
#     args.pgr_band_overlap_max = float(getattr(args, "pgr_band_overlap_max", 0.65))
#     args.pgr_min_class_samples = int(getattr(args, "pgr_min_class_samples", 3))
#     args.pgr_subspace_min_samples = int(getattr(args, "pgr_subspace_min_samples", 6))
#     args.pgr_subspace_rank = int(getattr(args, "pgr_subspace_rank", 3))
#     args.pgr_max_class_variance = float(getattr(args, "pgr_max_class_variance", 0.75))
#     args.pgr_normalize_features = bool(getattr(args, "pgr_normalize_features", True))

#     # Incremental descriptor refinement remains active in both modes: descriptor_only uses it as the
#     # only plasticity path; G²RPA uses it after adapter updates to rebuild new rows in adapted z-space.
#     args.refine_new_descriptors = bool(getattr(args, "refine_new_descriptors", True))
#     args.descriptor_refine_steps = int(getattr(args, "descriptor_refine_steps", 50))
#     args.descriptor_refine_lr = float(getattr(args, "descriptor_refine_lr", 1e-3))
#     args.descriptor_trust_weight = float(getattr(args, "descriptor_trust_weight", 1.0))
#     args.descriptor_refine_max_mean_shift = float(getattr(args, "descriptor_refine_max_mean_shift", 0.35))
#     args.descriptor_refine_max_logvar_shift = float(getattr(args, "descriptor_refine_max_logvar_shift", 0.70))
#     args.descriptor_refine_grad_clip = float(getattr(args, "descriptor_refine_grad_clip", 1.0))
#     args.allow_fixed_geometry_incremental = bool(getattr(args, "allow_fixed_geometry_incremental", True))

#     args.synthetic_replay_weight = float(getattr(args, "synthetic_replay_weight", getattr(args, "gfa_weight", 0.80)))
#     args.synthetic_replay_per_class = int(getattr(args, "synthetic_replay_per_class", getattr(args, "gfa_samples_per_class", 16)))
#     args.gfa_weight = float(getattr(args, "gfa_weight", args.synthetic_replay_weight))
#     args.gfa_samples_per_class = int(getattr(args, "gfa_samples_per_class", args.synthetic_replay_per_class))
#     # Keep legacy and SRGP/RSGI naming synchronized.
#     args.gfa_weight = float(args.synthetic_replay_weight)
#     args.gfa_samples_per_class = int(args.synthetic_replay_per_class)
#     args.gfa_parallel_scale = float(getattr(args, "gfa_parallel_scale", 1.0))
#     args.gfa_residual_scale = float(getattr(args, "gfa_residual_scale", 0.30))
#     args.joint_old_new_ce_weight = float(getattr(args, "joint_old_new_ce_weight", 1.0))
#     args.geometry_energy_margin_weight = float(getattr(args, "geometry_energy_margin_weight", 0.25))
#     args.geometry_energy_margin = float(getattr(args, "geometry_energy_margin", 0.25))
#     args.invasion_weight = float(getattr(args, "invasion_weight", getattr(args, "old_new_invasion_weight", 0.10)))
#     args.old_new_invasion_weight = float(getattr(args, "old_new_invasion_weight", args.invasion_weight))
#     args.old_new_invasion_weight = float(args.invasion_weight)
#     args.old_new_geometry_margin = float(getattr(args, "old_new_geometry_margin", 0.30))
#     args.incremental_weight_anchor = 0.0


#     # Generic risk-weighted replay is kept only as fallback. The main path uses
#     # deterministic SCB-GR boundary anchors from losses.loss.
#     args.use_risk_weighted_replay = bool(getattr(args, "use_risk_weighted_replay", False))
#     args.risk_replay_min_samples = int(getattr(args, "risk_replay_min_samples", 4))
#     args.risk_replay_max_multiplier = float(getattr(args, "risk_replay_max_multiplier", 3.0))
#     args.risk_replay_reliability_gated = bool(getattr(args, "risk_replay_reliability_gated", True))
#     args.risk_replay_reliability_weighted = bool(getattr(args, "risk_replay_reliability_weighted", True))
#     args.risk_center_margin = float(getattr(args, "risk_center_margin", 1.0))
#     args.risk_subspace_weight = float(getattr(args, "risk_subspace_weight", 1.0))
#     args.risk_band_weight = float(getattr(args, "risk_band_weight", 0.25))
#     args.gfa_reliability_gated = bool(getattr(args, "gfa_reliability_gated", True))

#     args.reliability_gated_admission = bool(getattr(args, "reliability_gated_admission", True))
#     args.admission_min_gate = float(getattr(args, "admission_min_gate", 0.35))
#     args.admission_shrink_floor = float(getattr(args, "admission_shrink_floor", 0.15))
#     args.admission_low_rank_cap = int(getattr(args, "admission_low_rank_cap", 2))

#     # Candidate correction is part of SGLAT admission. It runs before safe commit,
#     # not as a blind post-commit heuristic.
#     args.risk_aware_descriptor_correction = bool(getattr(args, "risk_aware_descriptor_correction", True))
#     args.descriptor_correction_risk_threshold = float(getattr(args, "descriptor_correction_risk_threshold", 0.35))
#     args.descriptor_correction_overlap_threshold = float(getattr(args, "descriptor_correction_overlap_threshold", 0.30))
#     args.descriptor_correction_basis_strength = float(getattr(args, "descriptor_correction_basis_strength", getattr(args, "descriptor_correction_subspace_eta", 0.85)))
#     args.descriptor_correction_mean_push = float(getattr(args, "descriptor_correction_mean_push", getattr(args, "descriptor_correction_center_step", 0.20)))
#     # Important: this coefficient is a shrink fraction in var *= (1 - coeff*gate), not a keep ratio.
#     args.descriptor_correction_var_shrink = float(getattr(args, "descriptor_correction_var_shrink", getattr(args, "descriptor_correction_variance_shrink", 0.15)))
#     args.descriptor_correction_topk_old = int(getattr(args, "descriptor_correction_topk_old", 3))
#     args.descriptor_correction_subspace_eta = args.descriptor_correction_basis_strength
#     args.descriptor_correction_center_step = args.descriptor_correction_mean_push
#     args.descriptor_correction_variance_shrink = args.descriptor_correction_var_shrink
#     args.risk_sep_weight = float(getattr(args, "risk_sep_weight", getattr(args, "descriptor_risk_sep_weight", 0.30)))
#     args.risk_sep_overlap_target = float(getattr(args, "risk_sep_overlap_target", getattr(args, "descriptor_overlap_target", 0.25)))
#     args.risk_sep_active_threshold = float(getattr(args, "risk_sep_active_threshold", 0.50))

#     # Descriptor collision controls; keep aliases synchronized for old/new trainer variants.
#     args.descriptor_refine_steps_per_epoch = int(
#         getattr(args, "descriptor_refine_steps_per_epoch", None) or getattr(args, "descriptor_refine_steps", 50)
#     )
#     args.descriptor_subspace_collision_weight = float(getattr(args, "descriptor_subspace_collision_weight", 0.20))
#     args.descriptor_overlap_target = float(getattr(args, "descriptor_overlap_target", getattr(args, "descriptor_subspace_overlap_max", 0.25)))
#     args.descriptor_subspace_overlap_max = float(getattr(args, "descriptor_subspace_overlap_max", args.descriptor_overlap_target))
#     args.descriptor_center_collision_weight = float(getattr(args, "descriptor_center_collision_weight", getattr(args, "descriptor_center_margin_weight", 0.05)))
#     args.descriptor_center_margin_weight = float(getattr(args, "descriptor_center_margin_weight", args.descriptor_center_collision_weight))
#     args.descriptor_center_margin = float(getattr(args, "descriptor_center_margin", 1.0))
#     args.descriptor_volume_control_weight = float(getattr(args, "descriptor_volume_control_weight", getattr(args, "descriptor_volume_weight", 0.03)))
#     args.descriptor_volume_weight = float(getattr(args, "descriptor_volume_weight", args.descriptor_volume_control_weight))

#     # Boundary-preserving descriptor optimization. This is the permanent all-phase
#     # old/new protection used by the updated incremental trainer.
#     args.boundary_preserve_weight = float(getattr(args, "boundary_preserve_weight", 0.35))
#     args.boundary_preserve_overlap_weight = float(getattr(args, "boundary_preserve_overlap_weight", 1.0))
#     args.boundary_preserve_center_weight = float(getattr(args, "boundary_preserve_center_weight", 0.50))
#     args.boundary_preserve_volume_weight = float(getattr(args, "boundary_preserve_volume_weight", 0.25))
#     args.boundary_preserve_band_weight = float(getattr(args, "boundary_preserve_band_weight", 0.10))
#     args.max_old_new_risk = float(getattr(args, "max_old_new_risk", 0.60))
#     args.max_old_new_overlap = float(getattr(args, "max_old_new_overlap", 0.65))
#     args.use_boundary_projection = bool(getattr(args, "use_boundary_projection", True))
#     args.boundary_projection_strength = float(getattr(args, "boundary_projection_strength", 0.35))
#     args.boundary_projection_mean_push = float(getattr(args, "boundary_projection_mean_push", 0.05))
#     args.boundary_projection_var_shrink = float(getattr(args, "boundary_projection_var_shrink", 0.05))
#     args.boundary_projection_overlap_threshold = float(getattr(args, "boundary_projection_overlap_threshold", args.max_old_new_overlap))
#     args.boundary_projection_topk_old = int(getattr(args, "boundary_projection_topk_old", 2))

#     # Optional bounded old/new score calibration is allowed only as a secondary ablation.
#     args.use_energy_calibrator = bool(getattr(args, "use_energy_calibrator", False))
#     args.energy_calibrator_type = str(getattr(args, "energy_calibrator_type", "none")).lower().strip()
#     if not args.use_energy_calibrator or args.energy_calibrator_type in {"", "none", "false", "off"}:
#         args.use_energy_calibrator = False
#         args.energy_calibrator_type = "none"
#         args.energy_calibration_weight = 0.0
#     elif args.energy_calibrator_type != "old_new":
#         raise ValueError("Clean method supports only --energy_calibrator_type old_new as an ablation; per_class is disabled.")
#     else:
#         args.energy_calibration_weight = float(getattr(args, "energy_calibration_weight", 1e-3))

#     # Base certificate thresholds. The default is warning, not hard stop.
#     args.enforce_base_geometry_certificate = bool(getattr(args, "enforce_base_geometry_certificate", False))
#     args.base_cert_min_geom_acc = float(getattr(args, "base_cert_min_geom_acc", 90.0))
#     args.base_cert_min_reliability = float(getattr(args, "base_cert_min_reliability", 0.15))
#     args.base_cert_min_mean_reliability = float(getattr(args, "base_cert_min_mean_reliability", 0.35))
#     args.base_cert_max_subspace_overlap = float(getattr(args, "base_cert_max_subspace_overlap", 0.65))
#     args.base_cert_max_geometry_conflict = float(getattr(args, "base_cert_max_geometry_conflict", 2.0))
#     args.base_cert_max_band_similarity = float(getattr(args, "base_cert_max_band_similarity", 0.98))
#     args.base_cert_max_spectral_shape_similarity = float(getattr(args, "base_cert_max_spectral_shape_similarity", 0.90))

#     # Validation/checkpoint protocol. Early stopping is deliberately disabled in
#     # this project. Previous runs proved that stopping phase 0 early builds a weak
#     # GeometryBank and then every incremental phase fails for the wrong reason.
#     args.early_stop_patience = 0
#     args.base_early_stop_patience = 0
#     args.incremental_early_stop_patience = 0
#     args.refresh_before_validation = bool(getattr(args, "refresh_before_validation", True))
#     args.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))
#     args.bank_refresh_every = 0  # no epoch-wise old-bank refresh; descriptor refinement commits rows explicitly
#     args.best_state_metric = str(getattr(args, "best_state_metric", "hm")).lower()
#     if args.base_only and args.best_state_metric in {"hm", "h", "harmonic"}:
#         args.best_state_metric = "geometry_score"
#     elif not args.base_only and args.best_state_metric in {"geometry_score", "geo", "geo_score"}:
#         args.best_state_metric = "hm"

#     if args.base_only:
#         args.epochs_inc = 0
#         args.lr_inc = 0.0
#     else:
#         args.epochs_inc = max(1, int(getattr(args, "epochs_inc", 1)))
#         args.lr_inc = float(getattr(args, "lr_inc", 0.0))

#     return args
# def validate_args(args):
#     args = normalize_args(args)

#     # Incremental update path must be explicit. descriptor_only needs descriptor
#     # refinement or optional score calibration; G²RPA can train the adapter even
#     # when descriptor refinement is disabled, although descriptor refinement is
#     # recommended for final new-row alignment.
#     adapter_mode = _normalize_incremental_update_mode(getattr(args, "incremental_update_mode", "scbgr")) == "geometry_gated_adapter"
#     if (not args.base_only) and int(getattr(args, "epochs_inc", 0)) > 0:
#         if (not adapter_mode) and not bool(getattr(args, "refine_new_descriptors", True)) and not bool(getattr(args, "use_energy_calibrator", False)):
#             raise ValueError(
#                 "Incremental phase has no update path. Use --incremental_update_mode scbgr "
#                 "with --refine_new_descriptors true, or use --incremental_update_mode geometry_gated_adapter "
#                 "for the G²RPA ablation."
#             )

#     # PGR validation.
#     for name in [
#         "pgr_weight", "pgr_compact_weight", "pgr_center_weight", "pgr_subspace_weight",
#         "pgr_spectral_weight", "pgr_band_weight", "pgr_volume_weight", "pgr_center_margin",
#         "pgr_spectral_margin", "pgr_band_overlap_max", "pgr_max_class_variance",
#         "geometry_energy_margin_weight", "geometry_energy_margin", "old_new_invasion_weight",
#         "old_new_geometry_margin", "incremental_weight_anchor", "energy_calibration_weight",
#         "mssl_weight", "mssl_inc_weight", "mssl_margin", "mssl_temperature",
#         "mssl_spatial_radius", "mssl_signed_neg_weight", "descriptor_refine_lr",
#         "descriptor_trust_weight", "descriptor_refine_max_mean_shift",
#         "descriptor_refine_max_logvar_shift", "descriptor_refine_grad_clip",
#         "use_logdet_energy", "logdet_energy_weight", "geometry_feature_scale", "geometry_feature_clamp",
#         "base_cert_min_geom_acc", "base_cert_min_reliability", "base_cert_min_mean_reliability",
#         "base_cert_max_subspace_overlap", "base_cert_max_geometry_conflict", "base_cert_max_band_similarity",
#         "risk_replay_min_samples", "risk_replay_max_multiplier", "risk_center_margin",
#         "risk_subspace_weight", "risk_band_weight", "admission_min_gate", "admission_shrink_floor",
#         "descriptor_subspace_collision_weight", "descriptor_overlap_target", "descriptor_subspace_overlap_max",
#         "descriptor_center_collision_weight", "descriptor_center_margin_weight", "descriptor_center_margin",
#         "descriptor_volume_control_weight", "descriptor_volume_weight",
#         "boundary_preserve_weight", "boundary_preserve_overlap_weight", "boundary_preserve_center_weight",
#         "boundary_preserve_volume_weight", "boundary_preserve_band_weight", "max_old_new_risk",
#         "max_old_new_overlap", "boundary_projection_strength", "boundary_projection_mean_push",
#         "boundary_projection_var_shrink", "boundary_projection_overlap_threshold",
#         "boundary_radius_min", "boundary_radius_max", "boundary_init_radius",
#         "boundary_radius_reg_weight", "boundary_old_new_constraint_weight",
#         "boundary_old_new_margin_base", "boundary_old_new_margin_scale",
#         "adaptive_boundary_loss_weight", "adaptive_boundary_lr",
#         "spectral_energy_weight", "spectral_derivative_weight", "spectral_second_derivative_weight",
#         "max_charts_per_class", "spectral_shape_weight", "synthetic_replay_weight", "synthetic_replay_per_class", "invasion_weight",
#         "base_srpgr_weight", "base_spectral_shape_weight", "base_spectral_shape_overlap_max",
#         "base_spectral_shape_risk_weight", "risk_spectral_shape_weight", "old_new_risk_spectral_shape_weight",
#         "descriptor_correction_risk_threshold", "descriptor_correction_overlap_threshold",
#         "descriptor_correction_basis_strength", "descriptor_correction_mean_push", "descriptor_correction_var_shrink",
#         "adapter_max_scale", "adapter_dropout", "adapter_lr", "adapter_weight_decay",
#         "g2rpa_adapter_weight", "adapter_old_delta_weight", "adapter_old_gate_weight",
#         "adapter_old_energy_weight", "adapter_old_margin_weight", "adapter_delta_weight",
#         "adapter_new_gate_weight", "adapter_new_gate_target", "adapter_new_gate_max_target",
#         "risk_sep_weight", "risk_sep_overlap_target", "risk_sep_active_threshold",
#         "boundary_replay_risk_threshold", "boundary_replay_overlap_threshold",
#         "boundary_replay_parallel_scale", "boundary_replay_residual_scale", "unified_loss_weight",
#         "unified_admission_weight", "unified_subspace_weight", "unified_rank_weight",
#         "unified_volume_weight", "unified_trust_weight",
#         "transport_ridge", "transport_ema", "transport_identity_blend",
#         "transport_min_reliability_gate", "transport_max_a_minus_i_fro", "transport_max_b_norm",
#         "transport_residual_scale", "transport_min_rmse_gain", "transport_max_rmse_ratio", "transport_min_old_anchor_acc",
#     ]:
#         if float(getattr(args, name)) < 0:
#             raise ValueError(f"--{name} must be non-negative.")
#     for name in ["pgr_min_class_samples", "pgr_subspace_min_samples", "pgr_subspace_rank", "gfa_samples_per_class", "synthetic_replay_per_class", "max_charts_per_class", "mssl_neg_k", "risk_replay_min_samples", "admission_low_rank_cap", "descriptor_refine_steps_per_epoch", "descriptor_correction_topk_old", "boundary_projection_topk_old", "adapter_bottleneck", "boundary_replay_samples_per_pair", "boundary_replay_max_pairs", "transport_batches", "transport_low_rank", "transport_after_adapter_epoch"]:
#         if int(getattr(args, name)) <= 0:
#             raise ValueError(f"--{name} must be positive.")

#     if args.base_ce_weight < 0:
#         raise ValueError("--base_ce_weight must be non-negative.")
#     if args.base_gics_weight < 0:
#         raise ValueError("--base_gics_weight must be non-negative.")
#     if args.base_gics_temperature <= 0:
#         raise ValueError("--base_gics_temperature must be > 0.")
#     if args.base_gics_spectral_temperature <= 0:
#         raise ValueError("--base_gics_spectral_temperature must be > 0.")
#     if args.base_gics_band_temperature <= 0:
#         raise ValueError("--base_gics_band_temperature must be > 0.")
#     if args.base_gics_feature_weight < 0:
#         raise ValueError("--base_gics_feature_weight must be non-negative.")
#     if args.base_gics_spectral_weight < 0:
#         raise ValueError("--base_gics_spectral_weight must be non-negative.")
#     if args.base_gics_band_weight < 0:
#         raise ValueError("--base_gics_band_weight must be non-negative.")
#     if args.base_gics_key_noise_std < 0:
#         raise ValueError("--base_gics_key_noise_std must be non-negative.")
#     if not (0.0 <= args.base_gics_key_band_drop < 1.0):
#         raise ValueError("--base_gics_key_band_drop must be in [0, 1).")
#     if not (0.0 <= args.base_gics_key_spatial_drop < 1.0):
#         raise ValueError("--base_gics_key_spatial_drop must be in [0, 1).")
#     if args.base_gics_key_scale_jitter < 0:
#         raise ValueError("--base_gics_key_scale_jitter must be non-negative.")
#     if args.use_mssl_loss:
#         if not bool(getattr(args, "unsafe_ablation_use_mssl_loss", False)):
#             raise ValueError("MSSL is disabled in the clean method. Use --unsafe_ablation_use_mssl_loss true only for ablation.")
#         if args.mssl_temperature <= 0:
#             raise ValueError("--mssl_temperature must be > 0.")
#         if args.mssl_margin <= 0:
#             raise ValueError("--mssl_margin must be > 0.")
#         if args.mssl_weight <= 0 and args.mssl_inc_weight <= 0:
#             raise ValueError("--use_mssl_loss true requires --mssl_weight or --mssl_inc_weight > 0.")
#     if bool(getattr(args, "refine_new_descriptors", True)):
#         if int(args.descriptor_refine_steps) <= 0:
#             raise ValueError("--descriptor_refine_steps must be positive when descriptor refinement is enabled.")
#         if int(args.descriptor_refine_steps_per_epoch) <= 0:
#             raise ValueError("--descriptor_refine_steps_per_epoch must be positive when descriptor refinement is enabled.")
#         if float(args.descriptor_refine_lr) <= 0:
#             raise ValueError("--descriptor_refine_lr must be > 0 when descriptor refinement is enabled.")
#     if not (0.0 <= float(args.descriptor_overlap_target) <= 1.0):
#         raise ValueError("--descriptor_overlap_target must be in [0,1].")
#     if not (0.0 <= float(args.descriptor_subspace_overlap_max) <= 1.0):
#         raise ValueError("--descriptor_subspace_overlap_max must be in [0,1].")
#     if not (0.0 <= float(args.max_old_new_overlap) <= 1.0):
#         raise ValueError("--max_old_new_overlap must be in [0,1].")
#     if not (0.0 <= float(args.boundary_projection_overlap_threshold) <= 1.0):
#         raise ValueError("--boundary_projection_overlap_threshold must be in [0,1].")
#     if not bool(getattr(args, "use_adaptive_boundary", True)):
#         raise ValueError("--use_adaptive_boundary false disables the adaptive decision-boundary solver; use only for ablation, not main runs.")
#     if float(args.boundary_radius_min) <= 0.0:
#         raise ValueError("--boundary_radius_min must be > 0.")
#     if float(args.boundary_radius_max) <= float(args.boundary_radius_min):
#         raise ValueError("--boundary_radius_max must be greater than --boundary_radius_min.")
#     if not (float(args.boundary_radius_min) <= float(args.boundary_init_radius) <= float(args.boundary_radius_max)):
#         raise ValueError("--boundary_init_radius must lie within [boundary_radius_min, boundary_radius_max].")
#     if not (0.0 <= float(args.adapter_new_gate_target) <= 1.0):
#         raise ValueError("--adapter_new_gate_target must be in [0,1].")
#     if not (0.0 <= float(args.adapter_new_gate_max_target) <= 1.0):
#         raise ValueError("--adapter_new_gate_max_target must be in [0,1].")
#     if float(args.adapter_new_gate_target) > float(args.adapter_new_gate_max_target):
#         raise ValueError("--adapter_new_gate_target must be <= --adapter_new_gate_max_target.")
#     if bool(getattr(args, "use_geometry_gated_adapter", False)) and float(args.adapter_max_scale) <= 0.0:
#         raise ValueError("G²RPA mode requires --adapter_max_scale > 0.")
#     if float(args.logdet_energy_weight) < 0.0:
#         raise ValueError("--logdet_energy_weight must be non-negative.")
#     if str(args.spectral_summary_mode).lower().strip() not in {"center", "mean"}:
#         raise ValueError("--spectral_summary_mode must be center or mean.")
#     if args.base_classes is not None and args.base_classes <= 0:
#         raise ValueError("--base_classes must be positive.")
#     if args.increment is not None and args.increment <= 0:
#         raise ValueError("--increment must be positive.")
#     if args.pca_components <= 0 and not args.no_pca:
#         raise ValueError("--pca_components must be positive when PCA is enabled.")
#     if args.subspace_rank <= 0:
#         raise ValueError("--subspace_rank must be positive.")
#     if args.spectral_rank <= 0:
#         raise ValueError("--spectral_rank must be positive.")
#     if args.batch_size <= 0:
#         raise ValueError("--batch_size must be positive.")
#     if args.epochs_base <= 0:
#         raise ValueError("--epochs_base must be positive.")
#     if args.base_geometry_refresh_every <= 0:
#         raise ValueError("--base_geometry_refresh_every must be > 0.")
#     if args.geometry_diag_anchors_per_class <= 0:
#         raise ValueError("--geometry_diag_anchors_per_class must be > 0.")
#     if args.geometry_diag_topk_pairs <= 0:
#         raise ValueError("--geometry_diag_topk_pairs must be > 0.")
#     if args.geometry_diag_topk_bands <= 0:
#         raise ValueError("--geometry_diag_topk_bands must be > 0.")
#     if args.train_ratio + args.val_ratio >= 1.0:
#         raise ValueError("--train_ratio + --val_ratio must be < 1.0.")
#     if args.residual_variance_scale <= 0:
#         raise ValueError("--residual_variance_scale must be > 0.")
#     if args.spectral_residual_variance_scale <= 0:
#         raise ValueError("--spectral_residual_variance_scale must be > 0.")
#     if args.invalid_class_energy <= 0:
#         raise ValueError("--invalid_class_energy must be > 0.")
#     if args.num_runs <= 0:
#         raise ValueError("--num_runs must be >= 1.")

#     for name in [
#         "geom_var_floor",
#         "spectral_variance_floor",
#         "base_gics_weight",
#         "base_gics_feature_weight",
#         "base_gics_spectral_weight",
#         "base_gics_band_weight",
#         "base_gics_key_noise_std",
#         "base_gics_key_band_drop",
#         "base_gics_key_spatial_drop",
#         "base_gics_key_scale_jitter",
#         "spectral_energy_weight",
#         "band_energy_weight",
#         "spectral_reliability_energy_weight",
#         "lr_inc",
#         "gfa_weight", "gfa_parallel_scale", "gfa_residual_scale",
#         "joint_old_new_ce_weight", "bss_weight", "sym_bss_weight", "gdr_weight",
#         "bss_margin", "risk_margin_scale", "geometry_calibration_weight",
#         "gdr_mean_weight", "gdr_basis_weight", "gdr_variance_weight",
#         "incremental_adapter_scale", "adapter_delta_weight", "anchor_consistency_weight",
#     ]:
#         if float(getattr(args, name)) < 0:
#             raise ValueError(f"--{name} must be non-negative.")

#     seed_list = parse_seed_list(args.seed_list)
#     if seed_list and len(seed_list) != args.num_runs:
#         raise ValueError(f"--seed_list has {len(seed_list)} seeds but --num_runs={args.num_runs}.")

#     return args


# def set_seed(seed: int, deterministic: bool = False):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         try:
#             torch.use_deterministic_algorithms(True, warn_only=True)
#         except Exception:
#             pass
#     else:
#         torch.backends.cudnn.benchmark = True


# def _set_model_phase_and_old_count(model, phase: int):
#     if hasattr(model, "set_phase"):
#         model.set_phase(int(phase))
#     else:
#         model.current_phase = int(phase)
#     if hasattr(model, "set_old_class_count"):
#         model.set_old_class_count(0)
#     else:
#         model.old_class_count = 0


# def _normalize_classifier_mode(mode: str) -> str:
#     m = str(mode or "srgp").lower().strip()
#     if m in {"srgp", "srgp_geometry", "spectral_residual_geometry"}:
#         return "srgp"
#     if m in {"geometry", "geo", "geometry_only"}:
#         return "geometry_only"
#     return m


# def _normalize_incremental_update_mode(mode: str) -> str:
#     """Normalize incremental update-mode aliases.

#     scbgr is the main clean path: frozen old GeometryBank rows, deterministic
#     old/new boundary anchors, and unified geometry-state admission loss.
#     Legacy names such as descriptor_only/rsgi/clean are accepted as aliases to
#     avoid stale commands silently selecting the old objective.  G²RPA remains a
#     separate optional ablation.
#     """
#     m = str(mode or "scbgr").lower().strip()
#     aliases = {
#         "": "scbgr",
#         "none": "scbgr",
#         "clean": "scbgr",
#         "rsgi": "scbgr",
#         "descriptor": "scbgr",
#         "descriptor_only": "scbgr",
#         "scbgr": "scbgr",
#         "scb-gr": "scbgr",
#         "boundary": "scbgr",
#         "spectral_risk_boundary": "scbgr",
#         "geometry_state_admission": "scbgr",
#         "geometry_gated_adapter": "geometry_gated_adapter",
#         "g2rpa": "geometry_gated_adapter",
#         "g2-rpa": "geometry_gated_adapter",
#         "g²rpa": "geometry_gated_adapter",
#         "gated_adapter": "geometry_gated_adapter",
#         "geometry_adapter": "geometry_gated_adapter",
#         "adapter": "geometry_gated_adapter",
#     }
#     if m not in aliases:
#         raise ValueError(
#             f"Unsupported --incremental_update_mode {mode!r}. "
#             "Use scbgr or geometry_gated_adapter."
#         )
#     return aliases[m]


# def _spectral_summary_is_physical_for_eval(args, spectra) -> bool:
#     if spectra is not None:
#         return bool(getattr(args, "raw_spectral_summary_is_physical", True))
#     if bool(getattr(args, "spectral_summary_is_physical", False)):
#         return True
#     if int(getattr(args, "pca_components", 0) or 0) > 0 and not bool(getattr(args, "no_pca", False)):
#         return False
#     return bool(getattr(args, "allow_nonphysical_spectral_summary", False))


# def _prepare_eval_spectra(patches: torch.Tensor, spectra, args):
#     if spectra is not None and torch.is_tensor(spectra) and spectra.numel() > 0:
#         s = spectra.to(device=patches.device, dtype=patches.dtype)
#         if s.dim() == 4:
#             s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#         elif s.dim() == 3:
#             s = s[:, :, s.size(-1) // 2]
#         elif s.dim() != 2:
#             s = s.flatten(1)
#         if s.size(0) == patches.size(0):
#             return s, _spectral_summary_is_physical_for_eval(args, spectra)
#     # Fallback is the center of the model input. With PCA this is explicitly non-physical.
#     if patches.dim() == 4:
#         s = patches[:, :, patches.size(-2) // 2, patches.size(-1) // 2]
#     elif patches.dim() == 2:
#         s = patches
#     else:
#         s = patches.flatten(1)
#     return s, _spectral_summary_is_physical_for_eval(args, None)


# def _model_forward(model, patches, args, spectra=None, phase: int = 0):
#     # Do not reset old_class_count here for incremental phases. The caller sets
#     # model.current_phase and model.old_class_count before evaluation.
#     if int(phase) == 0:
#         _set_model_phase_and_old_count(model, 0)
#     mode = _phase_classifier_mode(args, phase)
#     kwargs = {"classifier_mode": mode, "semantic_mode": "identity"}
#     spectral_summary, spec_is_physical = _prepare_eval_spectra(patches, spectra, args)
#     kwargs["spectral_summary"] = spectral_summary
#     kwargs["spectral_summary_is_physical"] = bool(spec_is_physical)
#     try:
#         return model(patches, **kwargs)
#     except TypeError:
#         kwargs.pop("spectral_summary", None)
#         kwargs.pop("spectral_summary_is_physical", None)
#         try:
#             return model(patches, **kwargs)
#         except TypeError:
#             kwargs.pop("semantic_mode", None)
#             return model(patches, **kwargs)


# def _unpack_eval_batch(batch):
#     """Accept legacy (patch,label) and metadata batches without breaking evaluation."""
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


# def runtime_contract_summary(args) -> Dict[str, Any]:
#     """Compact run contract saved with configs/checkpoints/reports."""
#     inc_mode = _normalize_incremental_update_mode(getattr(args, "incremental_update_mode", "scbgr"))
#     adapter_mode = inc_mode == "geometry_gated_adapter"
#     return {
#         "method": "sglat_hsi",
#         "classifier_mode": _normalize_classifier_mode(getattr(args, "eval_classifier_mode", "srgp")),
#         "incremental_update_mode": inc_mode,
#         "canonical_z": {
#             "normalize_geometry_features": bool(getattr(args, "normalize_geometry_features", True)),
#             "geometry_feature_scale": float(getattr(args, "geometry_feature_scale", 0.0)),
#             "strict_feature_contract": bool(getattr(args, "strict_feature_contract", True)),
#             "spectral_summary_mode": str(getattr(args, "spectral_summary_mode", "center")),
#         },
#         "energy": {
#             "use_logdet_energy": bool(getattr(args, "use_logdet_energy", True)),
#             "logdet_energy_weight": float(getattr(args, "logdet_energy_weight", 0.05)),
#             "energy_normalize_by_dim": bool(getattr(args, "energy_normalize_by_dim", True)),
#             "geometry_normalize_logits": bool(getattr(args, "geometry_normalize_logits", False)),
#             "reliability_energy_weight": float(getattr(args, "reliability_energy_weight", 0.05)),
#         },
#         "adaptive_boundary": {
#             "enabled": bool(getattr(args, "use_adaptive_boundary", True)),
#             "radius_min": float(getattr(args, "boundary_radius_min", 0.50)),
#             "radius_max": float(getattr(args, "boundary_radius_max", 2.00)),
#             "init_radius": float(getattr(args, "boundary_init_radius", 1.00)),
#             "radius_reg_weight": float(getattr(args, "boundary_radius_reg_weight", 0.01)),
#             "old_new_constraint_weight": float(getattr(args, "boundary_old_new_constraint_weight", 0.20)),
#             "old_new_margin_base": float(getattr(args, "boundary_old_new_margin_base", 0.05)),
#             "old_new_margin_scale": float(getattr(args, "boundary_old_new_margin_scale", 0.25)),
#             "loss_weight": float(getattr(args, "adaptive_boundary_loss_weight", 1.00)),
#             "lr": float(getattr(args, "adaptive_boundary_lr", 1e-4)),
#             "freeze_old_boundaries": bool(getattr(args, "freeze_old_boundaries", True)),
#         },
#         "base": {
#             "objective": "CE + SRPGR",
#             "early_stop": False,
#             "pgr_weight": float(getattr(args, "pgr_weight", 0.10)),
#             "pgr_band_weight": float(getattr(args, "pgr_band_weight", 0.05)),
#             "base_spectral_shape_weight": float(getattr(args, "base_spectral_shape_weight", 0.05)),
#             "enforce_base_geometry_certificate": bool(getattr(args, "enforce_base_geometry_certificate", False)),
#         },
#         "incremental": {
#             "update_mode": inc_mode,
#             "adaptive_boundary": bool(getattr(args, "use_adaptive_boundary", True)),
#             "adaptive_boundary_loss_weight": float(getattr(args, "adaptive_boundary_loss_weight", 1.00)),
#             "adaptive_boundary_lr": float(getattr(args, "adaptive_boundary_lr", 1e-4)),
#             "freeze_old_boundaries": bool(getattr(args, "freeze_old_boundaries", True)),
#             "early_stop": False,
#             "feature_plasticity_after_base": bool(adapter_mode),
#             "feature_plasticity_scope": "geometry_plastic_adapter_only" if adapter_mode else "none",
#             "refine_new_descriptors": bool(getattr(args, "refine_new_descriptors", True)),
#             "descriptor_refine_steps": int(getattr(args, "descriptor_refine_steps", 50)),
#             "descriptor_refine_steps_per_epoch": int(getattr(args, "descriptor_refine_steps_per_epoch", getattr(args, "descriptor_refine_steps", 50))),
#             "use_boundary_geometry_replay": bool(getattr(args, "use_boundary_geometry_replay", True)),
#             "boundary_replay_risk_threshold": float(getattr(args, "boundary_replay_risk_threshold", 0.35)),
#             "boundary_replay_overlap_threshold": float(getattr(args, "boundary_replay_overlap_threshold", 0.30)),
#             "boundary_replay_samples_per_pair": int(getattr(args, "boundary_replay_samples_per_pair", 12)),
#             "boundary_replay_max_pairs": int(getattr(args, "boundary_replay_max_pairs", 24)),
#             "scbgr_commit_only_if_safe": bool(getattr(args, "scbgr_commit_only_if_safe", True)),
#             "unified_loss": "unified_spectral_geometry_loss(phase='incremental')",
#             "unified_admission_weight": float(getattr(args, "unified_admission_weight", 0.70)),
#             "unified_subspace_weight": float(getattr(args, "unified_subspace_weight", 0.40)),
#             "unified_rank_weight": float(getattr(args, "unified_rank_weight", 0.25)),
#             "unified_volume_weight": float(getattr(args, "unified_volume_weight", 0.03)),
#             "unified_trust_weight": float(getattr(args, "unified_trust_weight", 1.0)),
#             "use_risk_weighted_replay": bool(getattr(args, "use_risk_weighted_replay", False)),
#             "reliability_gated_admission": bool(getattr(args, "reliability_gated_admission", True)),
#             "risk_aware_descriptor_correction": bool(getattr(args, "risk_aware_descriptor_correction", True)),
#             "use_sglat_transport": bool(getattr(args, "use_sglat_transport", False)),
#             "use_geometry_transport": bool(getattr(args, "use_geometry_transport", False)),
#             "allow_old_model_transport": bool(getattr(args, "allow_old_model_transport", False)),
#             "allow_transport_without_adapter": bool(getattr(args, "allow_transport_without_adapter", False)),
#             "transport_type": str(getattr(args, "transport_type", "ridge")),
#             "transport_ema": float(getattr(args, "transport_ema", 0.97)),
#             "transport_identity_blend": float(getattr(args, "transport_identity_blend", 0.75)),
#             "transport_batches": int(getattr(args, "transport_batches", 20)),
#             "transport_low_rank": int(getattr(args, "transport_low_rank", 4)),
#             "transport_min_reliability_gate": float(getattr(args, "transport_min_reliability_gate", 0.30)),
#             "transport_max_a_minus_i_fro": float(getattr(args, "transport_max_a_minus_i_fro", 1.5)),
#             "transport_max_b_norm": float(getattr(args, "transport_max_b_norm", 0.75)),
#             "transport_residual_scale": float(getattr(args, "transport_residual_scale", 0.50)),
#             "transport_min_rmse_gain": float(getattr(args, "transport_min_rmse_gain", 1e-5)),
#             "transport_max_rmse_ratio": float(getattr(args, "transport_max_rmse_ratio", 0.98)),
#             "transport_min_old_anchor_acc": float(getattr(args, "transport_min_old_anchor_acc", 95.0)),
#             "candidate_admission_mode": str(getattr(args, "candidate_admission_mode", "provisional")),
#             "descriptor_subspace_collision_weight": float(getattr(args, "descriptor_subspace_collision_weight", 0.20)),
#             "descriptor_center_collision_weight": float(getattr(args, "descriptor_center_collision_weight", 0.05)),
#             "descriptor_volume_control_weight": float(getattr(args, "descriptor_volume_control_weight", 0.03)),
#             "boundary_preserve_weight": float(getattr(args, "boundary_preserve_weight", 0.35)),
#             "boundary_preserve_overlap_weight": float(getattr(args, "boundary_preserve_overlap_weight", 1.0)),
#             "boundary_preserve_center_weight": float(getattr(args, "boundary_preserve_center_weight", 0.50)),
#             "boundary_preserve_volume_weight": float(getattr(args, "boundary_preserve_volume_weight", 0.25)),
#             "boundary_preserve_band_weight": float(getattr(args, "boundary_preserve_band_weight", 0.10)),
#             "max_old_new_risk": float(getattr(args, "max_old_new_risk", 0.60)),
#             "max_old_new_overlap": float(getattr(args, "max_old_new_overlap", 0.65)),
#             "use_boundary_projection": bool(getattr(args, "use_boundary_projection", True)),
#             "boundary_projection_strength": float(getattr(args, "boundary_projection_strength", 0.35)),
#             "boundary_projection_mean_push": float(getattr(args, "boundary_projection_mean_push", 0.05)),
#             "boundary_projection_var_shrink": float(getattr(args, "boundary_projection_var_shrink", 0.05)),
#             "boundary_projection_overlap_threshold": float(getattr(args, "boundary_projection_overlap_threshold", 0.65)),
#             "boundary_projection_topk_old": int(getattr(args, "boundary_projection_topk_old", 2)),
#         },
#         "g2rpa": {
#             "enabled": bool(adapter_mode),
#             "adapter_bottleneck": int(getattr(args, "adapter_bottleneck", 32)),
#             "adapter_max_scale": float(getattr(args, "adapter_max_scale", 0.35)),
#             "adapter_lr": float(getattr(args, "adapter_lr", 5e-4)),
#             "adapter_weight": float(getattr(args, "g2rpa_adapter_weight", 1.0)),
#             "old_delta_weight": float(getattr(args, "adapter_old_delta_weight", 1.0)),
#             "old_gate_weight": float(getattr(args, "adapter_old_gate_weight", 0.75)),
#             "new_delta_weight": float(getattr(args, "adapter_delta_weight", 0.10)),
#             "new_gate_weight": float(getattr(args, "adapter_new_gate_weight", 0.05)),
#             "new_gate_target": float(getattr(args, "adapter_new_gate_target", 0.25)),
#         },
#         "forbidden": {
#             "kd": False,
#             "raw_exemplars": False,
#             "projection_plasticity": False,
#             "backbone_plasticity": False,
#             "legacy_incremental_adapter": False,
#             "spectral_classifier_branch": "physical-summary-gated SRGP residual energy",
#             "bicyc_transport": False,
#         },
#     }


# def print_runtime_contract(args) -> None:
#     c = runtime_contract_summary(args)
#     print("[Runtime Contract] SGLAT-HSI")
#     print(
#         f"  z: normalize={c['canonical_z']['normalize_geometry_features']} | "
#         f"spectral_summary={c['canonical_z']['spectral_summary_mode']} | "
#         f"strict={c['canonical_z']['strict_feature_contract']}"
#     )
#     print(
#         f"  energy: logdet={c['energy']['use_logdet_energy']} "
#         f"(w={c['energy']['logdet_energy_weight']}) | "
#         f"row_norm_logits={c['energy']['geometry_normalize_logits']}"
#     )
#     print(
#         f"  adaptive boundary: enabled={c['adaptive_boundary']['enabled']} | "
#         f"rho=[{c['adaptive_boundary']['radius_min']}, {c['adaptive_boundary']['radius_max']}] | "
#         f"loss_w={c['adaptive_boundary']['loss_weight']} | "
#         f"lr={c['adaptive_boundary']['lr']} | "
#         f"freeze_old={c['adaptive_boundary']['freeze_old_boundaries']}"
#     )
#     print("  early_stop: disabled for base and incremental phases")
#     print(
#         f"  incremental: mode={c['incremental']['update_mode']} | "
#         f"descriptor_refine={c['incremental']['refine_new_descriptors']} | "
#         f"steps/epoch={c['incremental']['descriptor_refine_steps_per_epoch']} | "
#         f"risk_replay={c['incremental']['use_risk_weighted_replay']} | "
#         f"admission_gate={c['incremental']['reliability_gated_admission']}"
#     )
#     print(
#         f"  boundary: preserve_w={c['incremental']['boundary_preserve_weight']} | "
#         f"max_risk={c['incremental']['max_old_new_risk']} | "
#         f"max_overlap={c['incremental']['max_old_new_overlap']} | "
#         f"projection={c['incremental']['use_boundary_projection']} | "
#         f"proj_thr={c['incremental']['boundary_projection_overlap_threshold']}"
#     )
#     print(
#         f"  unified loss: admission_w={c['incremental']['unified_admission_weight']} | "
#         f"subspace_w={c['incremental']['unified_subspace_weight']} | "
#         f"rank_w={c['incremental']['unified_rank_weight']} | "
#         f"volume_w={c['incremental']['unified_volume_weight']}"
#     )
#     print(
#         f"  SGLAT: transport={c['incremental']['use_sglat_transport']} | "
#         f"type={c['incremental']['transport_type']} | "
#         f"ema={c['incremental']['transport_ema']} | "
#         f"identity_blend={c['incremental']['transport_identity_blend']} | "
#         f"candidate={c['incremental']['candidate_admission_mode']}"
#     )
#     if c.get("g2rpa", {}).get("enabled", False):
#         print(
#             f"  G2RPA: adapter_max_scale={c['g2rpa']['adapter_max_scale']} | "
#             f"adapter_lr={c['g2rpa']['adapter_lr']} | "
#             f"old_gate_w={c['g2rpa']['old_gate_weight']} | "
#             f"new_gate_target={c['g2rpa']['new_gate_target']}"
#         )


# def assert_clean_runtime_stack(model, trainer, args) -> None:
#     """Fail early if main.py is paired with stale model/bank/classifier/trainer files."""
#     if not bool(getattr(args, "strict_updated_stack", True)):
#         return
#     missing: List[str] = []
#     gb = getattr(model, "geometry_bank", None)
#     clf = getattr(model, "classifier", None)
#     for obj, label, attrs in [
#         (model, "NECILModel", ["extract_projected_features", "extract_geometry_features", "clone_frozen_for_transport", "get_subspace_bank", "transport_frozen_geometry"]),
#         (gb, "GeometryBank", ["validate_consistency", "update_class_geometry", "transport_frozen_geometry", "build_candidate_geometry_rows", "commit_candidate_geometry_rows"]),
#         (clf, "GeometryEnergyClassifier", ["geometry_energy", "geometry_energy_from_bank", "sglat_candidate_admission_report"]),
#         (trainer, "Trainer", [
#             "_assert_updated_stack_contract",
#             "_assert_incremental_preflight",
#             "_old_new_boundary_preservation_loss",
#             "_project_new_descriptor_params_out_of_old_tangent_space",
#         ]),
#     ]:
#         if obj is None:
#             missing.append(label)
#             continue
#         for attr in attrs:
#             if not hasattr(obj, attr):
#                 missing.append(f"{label}.{attr}")
#     if gb is not None and not (hasattr(gb, "apply_refined_feature_rows") or hasattr(model, "refresh_class_subspace")):
#         missing.append("GeometryBank.apply_refined_feature_rows or NECILModel.refresh_class_subspace")
#     # SGLAT transport/candidate admission is bank-native; boundary replay remains in losses.loss.
#     if clf is not None and not (hasattr(clf, "geometry_logits_from_bank") or hasattr(model, "compute_logits_from_features")):
#         missing.append("GeometryEnergyClassifier.geometry_logits_from_bank or NECILModel.compute_logits_from_features")
#     if bool(getattr(args, "use_spectral_geometry", True)) and clf is not None:
#         if not (hasattr(clf, "spectral_residual_energy") or hasattr(clf, "geometry_energy")):
#             missing.append("classifier spectral-residual SRGP support")
#     try:
#         from losses import loss as _loss_mod
#         if not hasattr(_loss_mod, "unified_spectral_geometry_loss"):
#             missing.append("losses.loss.unified_spectral_geometry_loss")
#         if bool(getattr(args, "use_boundary_geometry_replay", True)) and not hasattr(_loss_mod, "sample_boundary_geometry_features"):
#             missing.append("losses.loss.sample_boundary_geometry_features")
#     except Exception as exc:
#         missing.append(f"losses.loss import failed: {exc}")
#     if bool(getattr(args, "use_adaptive_boundary", True)):
#         for attr in ("adaptive_boundary_enabled", "ensure_adaptive_boundary_capacity", "adaptive_boundary_parameters", "freeze_old_boundary_radii", "adaptive_boundary_loss", "adaptive_boundary_state"):
#             if not hasattr(model, attr):
#                 missing.append(f"NECILModel.{attr}")
#         if clf is not None:
#             for attr in ("boundary_parameters", "adaptive_boundary_loss", "adaptive_boundary_state"):
#                 if not hasattr(clf, attr):
#                     missing.append(f"GeometryEnergyClassifier.{attr}")
#         for attr in ("_adaptive_boundary_enabled", "_adaptive_boundary_state", "_adaptive_boundary_trainable_params"):
#             if not hasattr(trainer, attr):
#                 missing.append(f"Trainer.{attr}")
#         if not hasattr(trainer, "_adaptive_boundary_loss_from_current_bank"):
#             missing.append("IncrementalPhaseTrainer._adaptive_boundary_loss_from_current_bank")

#     if _normalize_incremental_update_mode(getattr(args, "incremental_update_mode", "scbgr")) == "geometry_gated_adapter":
#         for attr in ("geometry_plastic_adapter", "adapt_projected_features", "compute_old_geometry_risk_features"):
#             if not hasattr(model, attr):
#                 missing.append(f"NECILModel.{attr}")
#         if gb is not None and not hasattr(gb, "old_geometry_risk_features"):
#             missing.append("GeometryBank.old_geometry_risk_features")
#         if not hasattr(trainer, "_adapter_mode_enabled"):
#             missing.append("Trainer/IncrementalPhaseTrainer._adapter_mode_enabled")
#     if missing:
#         raise RuntimeError(
#             "Updated SRGP/RSGI stack is incomplete. Replace stale files before running. Missing: "
#             + ", ".join(missing[:30])
#         )
#     if hasattr(trainer, "_assert_updated_stack_contract"):
#         trainer._assert_updated_stack_contract(phase=0)


# def _build_checkpoint_payload(model, args, extra: Optional[dict] = None) -> dict:
#     payload = {
#         "model_state_dict": model.state_dict(),
#         "memory_snapshot": model.export_memory_snapshot() if hasattr(model, "export_memory_snapshot") else None,
#         "args": vars(args),
#         "current_num_classes": int(getattr(model, "current_num_classes", 0)),
#         "old_class_count": int(getattr(model, "old_class_count", 0)),
#         "current_phase": int(getattr(model, "current_phase", 0)),
#         "runtime_contract": runtime_contract_summary(args),
#         "adaptive_boundary_state": model.adaptive_boundary_state(int(getattr(model, "old_class_count", 0))) if hasattr(model, "adaptive_boundary_state") else None,
#     }
#     if extra:
#         payload.update(extra)
#     return payload


# @torch.no_grad()
# def get_base_predictions(model, dataset, device, args, batch_size=128):
#     model.eval()
#     _set_model_phase_and_old_count(model, 0)
#     loader = dataset.get_cumulative_dataloader(0, split="test", batch_size=batch_size, shuffle=False)
#     all_preds, all_labels = [], []
#     for batch in loader:
#         patches, labels, spectra, _ = _unpack_eval_batch(batch)
#         patches = patches.to(device).float()
#         if torch.is_tensor(spectra):
#             spectra = spectra.to(device)
#         out = _model_forward(model, patches, args, spectra=spectra, phase=0)
#         logits = out["logits"]
#         labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
#         seen_classes = dataset.get_classes_up_to_phase(0) if hasattr(dataset, "get_classes_up_to_phase") else sorted(np.unique(labels_np).tolist())
#         seen_t = torch.tensor([int(c) for c in seen_classes], device=logits.device, dtype=torch.long)
#         masked = torch.full_like(logits, -1e9)
#         masked.index_copy_(1, seen_t, logits.index_select(1, seen_t))
#         all_preds.append(masked.argmax(dim=1).cpu().numpy())
#         all_labels.append(labels_np)
#     if not all_preds:
#         return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
#     return np.concatenate(all_preds), np.concatenate(all_labels)


# @torch.no_grad()
# def evaluate_base_model(model, dataset, device, args, batch_size=128):
#     y_pred, y_true = get_base_predictions(model, dataset, device, args, batch_size=batch_size)
#     if y_pred.size == 0:
#         return {"overall_accuracy": 0.0, "per_class_accuracy": {}}
#     overall = 100.0 * float((y_pred == y_true).mean())
#     per_class = {}
#     for cls in np.unique(y_true):
#         mask = y_true == cls
#         per_class[int(cls)] = 100.0 * float((y_pred[mask] == cls).mean()) if mask.sum() > 0 else 0.0
#     return {"overall_accuracy": overall, "per_class_accuracy": per_class}


# def _call_predict_phase_grid_compat(**kwargs):
#     sig = inspect.signature(predict_phase_grid)
#     filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
#     return predict_phase_grid(**filtered)


# def build_incremental_dataset(args, patches, labels, coords, gt_shape, gt_map, target_names=None, label_policy=None, raw_spectra=None):
#     kwargs = dict(
#         patches=patches,
#         labels=labels,
#         coords=coords,
#         gt_shape=gt_shape,
#         GT=gt_map,
#         base_classes=args.base_classes,
#         increment=args.increment,
#         train_ratio=args.train_ratio,
#         val_ratio=args.val_ratio,
#         seed=args.seed,
#         device=args.device,
#         min_train_per_class=args.min_train_per_class,
#         strict_non_exemplar=args.strict_non_exemplar,
#     )
#     sig = inspect.signature(IncrementalHSIDataset.__init__)
#     optional = {
#         "num_workers": args.num_workers,
#         "target_names": target_names,
#         "label_policy": label_policy,
#         # Used by the updated IncrementalHSIDataset to preserve center spectra
#         # and pixel coordinates for SSMR/MSSL-style loss. Ignored by older dataset code.
#         # Metadata is useful for SRGP physical center spectra and diagnostics. Older
#         # IncrementalHSIDataset implementations ignore these optional keys.
#         "return_metadata": True,
#         "include_metadata": True,
#         "raw_spectra": raw_spectra,
#         "center_spectra": raw_spectra,
#         "spectra_are_physical": bool(raw_spectra is not None and getattr(args, "raw_spectral_summary_is_physical", True)),
#     }
#     for k, v in optional.items():
#         if k in sig.parameters:
#             kwargs[k] = v
#     return IncrementalHSIDataset(**kwargs)


# def evaluator_update_compat(evaluator, y_true, y_pred, seen_classes=None):
#     sig = inspect.signature(evaluator.update)
#     kwargs = {}
#     if "old_class_count" in sig.parameters:
#         kwargs["old_class_count"] = 0
#     if "seen_classes" in sig.parameters:
#         kwargs["seen_classes"] = seen_classes
#     evaluator.update(0, y_true, y_pred, **kwargs)


# def save_run_config(args, save_root, model=None):
#     """Save run configuration without assuming the model has already been built.

#     The previous version referenced a free variable named `model`, but
#     `save_run_config(local_args, run_dir)` is called before `model = NECILModel(...)`
#     in run_single_experiment(). That causes a NameError at runtime.
#     Pass model explicitly when adaptive-boundary state is available.
#     """
#     os.makedirs(save_root, exist_ok=True)
#     path = os.path.join(save_root, "run_config.json")

#     adaptive_state = None
#     if model is not None and hasattr(model, "adaptive_boundary_state"):
#         try:
#             adaptive_state = model.adaptive_boundary_state(
#                 int(getattr(model, "old_class_count", 0))
#             )
#         except Exception as exc:
#             adaptive_state = {"error": f"adaptive_boundary_state unavailable: {exc}"}

#     payload = {
#         "args": vars(args),
#         "runtime_contract": runtime_contract_summary(args),
#         "adaptive_boundary_state": adaptive_state,
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(payload), f, indent=2)
#     return path


# def aggregate_metric(metric_list):
#     arr = np.asarray(metric_list, dtype=np.float64)
#     return float(arr.mean()), float(arr.std(ddof=0))


# def _metric_get(metrics: Dict[str, Any], *keys, default=0.0):
#     for k in keys:
#         if k in metrics:
#             return metrics[k]
#     return default


# def save_base_classification_report(
#     *,
#     evaluator,
#     y_true,
#     y_pred,
#     phase_dir,
#     target_names_seq,
#     seen_classes,
#     enabled=True,
#     tr_time=None,
#     te_time=None,
# ):
#     if not enabled:
#         return None
#     os.makedirs(phase_dir, exist_ok=True)
#     if hasattr(evaluator, "save_phase_report"):
#         return evaluator.save_phase_report(
#             phase=0,
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names_seq,
#             save_dir=phase_dir,
#             seen_classes=seen_classes,
#             old_class_count=0,
#             tr_time=tr_time,
#             te_time=te_time,
#             dl_time=0.0,
#         )
#     return save_classification_report(
#         y_true=y_true,
#         y_pred=y_pred,
#         target_names=target_names_seq,
#         save_dir=phase_dir,
#         phase=0,
#         seen_classes=seen_classes,
#         old_class_count=0,
#         tr_time=tr_time,
#         te_time=te_time,
#         dl_time=0.0,
#         save_hsi_style=True,
#         save_structured=True,
#     )



# def _load_json_if_exists(path: Optional[str]) -> Optional[Dict[str, Any]]:
#     if not path or not os.path.exists(path):
#         return None
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             obj = json.load(f)
#         return obj if isinstance(obj, dict) else {"value": obj}
#     except Exception as exc:
#         return {"error": f"Failed to load {path}: {exc}"}


# def _default_geometry_diagnostic_paths(phase_dir: str, phase: int = 0) -> Dict[str, str]:
#     phase = int(phase)
#     return {
#         "json": os.path.join(phase_dir, f"phase_{phase}_geometry_diagnostics.json"),
#         "class_csv": os.path.join(phase_dir, f"phase_{phase}_geometry_class_stats.csv"),
#         "energy_csv": os.path.join(phase_dir, f"phase_{phase}_geometry_energy_margins.csv"),
#         "subspace_csv": os.path.join(phase_dir, f"phase_{phase}_geometry_subspace_pairs.csv"),
#         "anchor_csv": os.path.join(phase_dir, f"phase_{phase}_geometry_anchor_stats.csv"),
#         "txt": os.path.join(phase_dir, f"phase_{phase}_geometry_diagnostics.txt"),
#     }


# def _collect_geometry_diagnostics(trainer, phase_dir: str, phase: int = 0) -> tuple[Optional[Dict[str, Any]], Dict[str, str]]:
#     """
#     Collect the persistent geometry-health report produced by TrainerHelper/BasePhaseTrainer.

#     This is intentionally separate from the classification report. Base OA can be high while
#     geometry memory is weak; this function loads the actual memory diagnostics: active ranks,
#     reliability, residual variance, energy margins, subspace-risk pairs, and anchor replay.
#     """
#     paths = _default_geometry_diagnostic_paths(phase_dir, phase=phase)

#     # Prefer in-memory diagnostics produced by the updated base trainer.
#     diag = getattr(trainer, "_last_base_geometry_diagnostics", None)
#     if isinstance(diag, dict) and diag:
#         return make_json_serializable(diag), paths

#     # Fallback: load the JSON written by _save_geometry_diagnostics_to_files().
#     loaded = _load_json_if_exists(paths.get("json"))
#     return loaded, paths


# def _write_geometry_diagnostics_section(
#     f,
#     geometry_diagnostics: Optional[Dict[str, Any]],
#     geometry_diagnostics_paths: Optional[Dict[str, str]] = None,
#     max_rows: int = 25,
# ) -> None:
#     f.write("\nGeometry Memory Health Diagnostics\n")
#     f.write("=" * 70 + "\n")

#     if geometry_diagnostics_paths:
#         f.write("Diagnostic files:\n")
#         for key, path in geometry_diagnostics_paths.items():
#             exists = os.path.exists(path) if isinstance(path, str) else False
#             f.write(f"  {key}: {path} {'[OK]' if exists else '[missing]'}\n")
#         f.write("\n")

#     if not geometry_diagnostics:
#         f.write(
#             "No geometry diagnostics were found. This report is incomplete: it can show base OA, "
#             "but it cannot prove the GeometryBank is valid for non-exemplar incremental learning.\n"
#         )
#         return

#     if "error" in geometry_diagnostics:
#         f.write(f"Diagnostics error: {geometry_diagnostics['error']}\n")
#         return

#     alerts = geometry_diagnostics.get("alerts", []) or []
#     f.write("Alerts:\n")
#     if alerts:
#         for alert in alerts:
#             f.write(f"  [WARN] {alert}\n")
#     else:
#         f.write("  No major geometry alarms triggered by current thresholds.\n")

#     # Compact overall status.
#     energy_overall = (geometry_diagnostics.get("energy_margin", {}) or {}).get("overall", {}) or {}
#     anchor_overall = (geometry_diagnostics.get("anchor_replay", {}) or {}).get("overall", {}) or {}
#     f.write("\nOverall geometry boundary checks:\n")
#     f.write(
#         "  Energy margin: "
#         f"acc={float(energy_overall.get('accuracy', 0.0)):.2f}% | "
#         f"mean_margin={float(energy_overall.get('mean_margin', 0.0)):.6f} | "
#         f"min_margin={float(energy_overall.get('min_margin', 0.0)):.6f} | "
#         f"viol={float(energy_overall.get('violation_rate', 0.0)):.2f}%\n"
#     )
#     f.write(
#         "  Anchor replay: "
#         f"acc={float(anchor_overall.get('accuracy', 0.0)):.2f}% | "
#         f"mean_margin={float(anchor_overall.get('mean_margin', 0.0)):.6f} | "
#         f"min_margin={float(anchor_overall.get('min_margin', 0.0)):.6f} | "
#         f"viol={float(anchor_overall.get('violation_rate', 0.0)):.2f}%\n"
#     )

#     class_rows = geometry_diagnostics.get("class_geometry", []) or []
#     f.write("\nClass geometry rows:\n")
#     f.write("  cls name                 n     rank rel    resvar      band-H  band-max\n")
#     for r in class_rows[:max_rows]:
#         f.write(
#             f"  {int(r.get('class_id', -1)):3d} {str(r.get('class_name', ''))[:20]:20s} "
#             f"{float(r.get('sample_count', -1.0)):5.0f} "
#             f"{int(r.get('feature_active_rank', r.get('active_rank', -1))):4d} "
#             f"{float(r.get('final_reliability', r.get('feature_reliability', r.get('reliability', -1.0)))):6.3f} "
#             f"{float(r.get('feature_residual_var', r.get('residual_var', -1.0))):9.5f} "
#             f"{float(r.get('band_entropy', -1.0)):7.3f} "
#             f"{float(r.get('band_max_weight', -1.0)):8.4f}\n"
#         )

#     margin_rows = (geometry_diagnostics.get("energy_margin", {}) or {}).get("per_class", []) or []
#     f.write("\nEnergy margin per class:\n")
#     f.write("  cls name                 n     acc     mean_margin   min_margin    viol\n")
#     for r in margin_rows[:max_rows]:
#         f.write(
#             f"  {int(r.get('class_id', -1)):3d} {str(r.get('class_name', ''))[:20]:20s} "
#             f"{int(r.get('n', 0)):5d} "
#             f"{float(r.get('accuracy', 0.0)):7.2f} "
#             f"{float(r.get('mean_margin', 0.0)):13.6f} "
#             f"{float(r.get('min_margin', 0.0)):12.6f} "
#             f"{float(r.get('violation_rate', 0.0)):7.2f}%\n"
#         )

#     pair_rows = geometry_diagnostics.get("subspace_risk_pairs", []) or []
#     f.write("\nTop subspace risk pairs:\n")
#     f.write("  pair                                      overlap    distance  risk\n")
#     for r in pair_rows[:min(max_rows, 15)]:
#         pair = f"{r.get('name_i', r.get('class_i', '?'))} / {r.get('name_j', r.get('class_j', '?'))}"
#         f.write(
#             f"  {pair[:40]:40s} "
#             f"{float(r.get('feature_overlap', 0.0)):9.4f} "
#             f"{float(r.get('spectral_overlap', 0.0)):9.4f} "
#             f"{float(r.get('feature_center_distance', 0.0)):8.4f} "
#             f"{float(r.get('spectral_center_distance', 0.0)):8.4f} "
#             f"{float(r.get('risk_score', 0.0)):8.4f}\n"
#         )

#     anchor_rows = (geometry_diagnostics.get("anchor_replay", {}) or {}).get("per_class", []) or []
#     f.write("\nAnchor replay per class:\n")
#     f.write("  cls name                 n     acc     mean_margin   min_margin    viol\n")
#     for r in anchor_rows[:max_rows]:
#         f.write(
#             f"  {int(r.get('class_id', -1)):3d} {str(r.get('class_name', ''))[:20]:20s} "
#             f"{int(r.get('n', 0)):5d} "
#             f"{float(r.get('anchor_accuracy', 0.0)):7.2f} "
#             f"{float(r.get('anchor_mean_margin', 0.0)):13.6f} "
#             f"{float(r.get('anchor_min_margin', 0.0)):12.6f} "
#             f"{float(r.get('anchor_violation_rate', 0.0)):7.2f}%\n"
#         )



# # ============================================================
# # Full NECIL phase evaluation/reporting
# # ============================================================


# def _set_model_phase_and_old_count_full(model, phase: int, old_class_count: int) -> None:
#     if hasattr(model, "set_phase"):
#         model.set_phase(int(phase))
#     else:
#         model.current_phase = int(phase)
#     if hasattr(model, "set_old_class_count"):
#         model.set_old_class_count(int(old_class_count))
#     else:
#         model.old_class_count = int(old_class_count)


# def _phase_classifier_mode(args, phase: int) -> str:
#     del phase
#     return _normalize_classifier_mode(getattr(args, "eval_classifier_mode", "srgp"))

# @torch.no_grad()
# def get_phase_predictions(model, dataset, phase: int, device, args, batch_size=128):
#     phase = int(phase)
#     model.eval()
#     old_class_count = 0 if phase == 0 else len(dataset.get_classes_up_to_phase(phase - 1))
#     _set_model_phase_and_old_count_full(model, phase, old_class_count)
#     loader = dataset.get_cumulative_dataloader(phase, split="test", batch_size=batch_size, shuffle=False)
#     all_preds, all_labels = [], []
#     classifier_mode = _phase_classifier_mode(args, phase)
#     for batch in loader:
#         patches, labels, spectra, _ = _unpack_eval_batch(batch)
#         patches = patches.to(device).float()
#         if torch.is_tensor(spectra):
#             spectra = spectra.to(device)
#         out = _model_forward(model, patches, args, spectra=spectra, phase=phase)
#         logits = out["logits"]
#         labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
#         seen_classes = dataset.get_classes_up_to_phase(phase) if hasattr(dataset, "get_classes_up_to_phase") else sorted(np.unique(labels_np).tolist())
#         seen_t = torch.tensor([int(c) for c in seen_classes], device=logits.device, dtype=torch.long)
#         masked = torch.full_like(logits, -1e9)
#         masked.index_copy_(1, seen_t, logits.index_select(1, seen_t))
#         all_preds.append(masked.argmax(dim=1).cpu().numpy())
#         all_labels.append(labels_np)
#     if not all_preds:
#         return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
#     return np.concatenate(all_preds), np.concatenate(all_labels)


# @torch.no_grad()
# def evaluate_phase_model(model, dataset, phase: int, device, args, batch_size=128):
#     y_pred, y_true = get_phase_predictions(model, dataset, phase, device, args, batch_size=batch_size)
#     if y_pred.size == 0:
#         return {"overall_accuracy": 0.0, "per_class_accuracy": {}}
#     overall = 100.0 * float((y_pred == y_true).mean())
#     per_class = {}
#     for cls in np.unique(y_true):
#         mask = y_true == cls
#         per_class[int(cls)] = 100.0 * float((y_pred[mask] == cls).mean()) if mask.sum() > 0 else 0.0
#     return {"overall_accuracy": overall, "per_class_accuracy": per_class}


# def evaluator_update_phase_compat(evaluator, phase: int, y_true, y_pred, old_class_count: int, seen_classes=None):
#     sig = inspect.signature(evaluator.update)
#     kwargs = {}
#     if "old_class_count" in sig.parameters:
#         kwargs["old_class_count"] = int(old_class_count)
#     if "seen_classes" in sig.parameters:
#         kwargs["seen_classes"] = seen_classes
#     evaluator.update(int(phase), y_true, y_pred, **kwargs)


# def save_phase_classification_report(
#     *,
#     evaluator,
#     phase: int,
#     y_true,
#     y_pred,
#     phase_dir,
#     target_names_seq,
#     seen_classes,
#     old_class_count: int,
#     enabled=True,
#     tr_time=None,
#     te_time=None,
# ):
#     if not enabled:
#         return None
#     os.makedirs(phase_dir, exist_ok=True)
#     if hasattr(evaluator, "save_phase_report"):
#         return evaluator.save_phase_report(
#             phase=int(phase),
#             y_true=y_true,
#             y_pred=y_pred,
#             target_names=target_names_seq,
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
#         target_names=target_names_seq,
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


# def _collect_geometry_diagnostics_for_phase(trainer, phase_dir: str, phase: int) -> tuple[Optional[Dict[str, Any]], Dict[str, str]]:
#     paths = _default_geometry_diagnostic_paths(phase_dir, phase=phase)
#     # Base trainer uses _last_base_geometry_diagnostics. Incremental-safe trainer may use generic/phase attrs.
#     attr_candidates = [
#         f"_last_phase_{int(phase)}_geometry_diagnostics",
#         "_last_incremental_geometry_diagnostics",
#         "_last_phase_geometry_diagnostics",
#         "_last_geometry_diagnostics",
#     ]
#     if int(phase) == 0:
#         attr_candidates.insert(0, "_last_base_geometry_diagnostics")
#     for attr in attr_candidates:
#         diag = getattr(trainer, attr, None)
#         if isinstance(diag, dict) and diag:
#             return make_json_serializable(diag), paths
#     loaded = _load_json_if_exists(paths.get("json"))
#     return loaded, paths


# def _phase_history_extend(global_history: Dict[str, Any], phase_history: Optional[Dict[str, Any]], phase: int) -> None:
#     if not isinstance(phase_history, dict):
#         return
#     if "phase_boundaries" not in global_history:
#         global_history["phase_boundaries"] = []
#     current_len = len(global_history.get("train_loss", []))
#     global_history["phase_boundaries"].append(current_len)
#     for key, values in phase_history.items():
#         if isinstance(values, list):
#             global_history.setdefault(key, [])
#             global_history[key].extend(values)
#     global_history.setdefault("phase_ids", [])
#     add_len = 0
#     for values in phase_history.values():
#         if isinstance(values, list):
#             add_len = max(add_len, len(values))
#     global_history["phase_ids"].extend([int(phase)] * add_len)


# def _maybe_drop_base_head_after_phase0(model) -> None:
#     # The temporary base CE head is discarded once base memory is finalized.
#     if hasattr(model, "drop_base_prl_head"):
#         model.drop_base_prl_head()


# def run_single_experiment(args, run_idx: int, run_seed: int):
#     local_args = argparse.Namespace(**vars(args))
#     local_args.seed = int(run_seed)
#     local_args = validate_args(local_args)

#     inc_mode = _normalize_incremental_update_mode(getattr(local_args, "incremental_update_mode", "scbgr"))
#     has_clean_descriptor_update = bool(getattr(local_args, "refine_new_descriptors", True)) and int(getattr(local_args, "descriptor_refine_steps", 0)) > 0
#     if (not has_clean_descriptor_update) and int(getattr(local_args, "epochs_inc", 0)) > 0 and inc_mode != "geometry_gated_adapter":
#         print("[WARN] New-row refinement is disabled; SGLAT incremental phases will rely only on transport/admission diagnostics.")
#     if inc_mode == "geometry_gated_adapter" and not has_clean_descriptor_update:
#         print("[WARN] G²RPA adapter will train, but descriptor refinement is disabled; new GeometryBank rows may stay under-aligned.")

#     set_seed(local_args.seed, deterministic=local_args.deterministic)
#     device = torch.device(local_args.device)

#     print("\n=== SGLAT-HSI FULL RUN ===")
#     print(f"Run: {run_idx + 1}/{args.num_runs} | Seed: {local_args.seed}")
#     print(f"Device: {device} | Dataset: {local_args.dataset}")
#     print(f"Protocol: {'base only' if local_args.base_only else 'full incremental'} | strict non-exemplar memory build")
#     print(f"Geometry: z_rank={local_args.subspace_rank}, SRGP energy={local_args.use_spectral_geometry}, spectral_w={local_args.spectral_energy_weight}, physical_spectral_required={local_args.spectral_require_physical_summary}")
#     print(f"Adaptive boundary: enabled={local_args.use_adaptive_boundary}, rho=[{local_args.boundary_radius_min}, {local_args.boundary_radius_max}], loss_w={local_args.adaptive_boundary_loss_weight}, freeze_old={local_args.freeze_old_boundaries}")
#     if bool(getattr(local_args, "use_mssl_loss", False)):
#         print(
#             f"UNSAFE ABLATION SSMR/MSSL regularizer: type={local_args.mssl_loss_type}, "
#             f"base_w={local_args.mssl_weight}, inc_w={local_args.mssl_inc_weight}, "
#             f"radius={local_args.mssl_spatial_radius}, neg_k={local_args.mssl_neg_k}"
#         )
#     print("Base objective: unified_spectral_geometry_loss(phase='base') = CE + SRPGR")
#     print("Incremental objective: SGLAT transport + geometry-gated adapter + candidate admission + boundary refinement" if inc_mode == "geometry_gated_adapter" else "Incremental objective: SGLAT descriptor transport + candidate admission + boundary refinement")
#     print("Clean constraints: no KD, no raw old patches, no stored old features, no BiCyc/projection/backbone plasticity; old rows move only by compact SGLAT descriptor transport")
#     print("=======================================================\n")
#     print_runtime_contract(local_args)
#     print("")

#     apply_reduction = (not local_args.no_pca) and (local_args.reduction_method.lower() != "none")

#     raw_hsi_physical = None
#     raw_center_spectra = None
#     try:
#         load_out = LoadHSIData(
#             method=local_args.dataset,
#             base_dir=local_args.data_dir,
#             apply_reduction=apply_reduction,
#             n_components=local_args.pca_components,
#             reduction_method=local_args.reduction_method,
#             return_label_policy=True,
#             return_raw_hsi=True,
#         )
#         if len(load_out) == 7:
#             hsi, gt, num_classes, target_names, has_bg, label_policy, raw_hsi_physical = load_out
#         else:
#             hsi, gt, num_classes, target_names, has_bg, label_policy = load_out
#     except TypeError:
#         hsi, gt, num_classes, target_names, has_bg = LoadHSIData(
#             method=local_args.dataset,
#             base_dir=local_args.data_dir,
#             apply_reduction=apply_reduction,
#             n_components=local_args.pca_components,
#             reduction_method=local_args.reduction_method,
#         )
#         label_policy = None

#     try:
#         cube_out = ImageCubes(
#             HSI=hsi,
#             GT=gt,
#             WS=local_args.patch_size,
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
#     except TypeError:
#         patches, labels, coords = ImageCubes(
#             HSI=hsi,
#             GT=gt,
#             WS=local_args.patch_size,
#             removeZeroLabels=True,
#             has_background=has_bg,
#             num_classes=num_classes,
#             pytorch_format=True,
#         )
#         raw_center_spectra = None

#     # Only raw_hsi_physical before PCA is allowed to activate spectral-shape SRGP.
#     if raw_center_spectra is None:
#         local_args.raw_spectral_summary_is_physical = False
#         print("[SRGP] Raw physical center spectra unavailable; spectral-shape energy will be gated off.")
#     else:
#         local_args.raw_spectral_summary_is_physical = True
#         print(f"[SRGP] Raw physical center spectra available: {raw_center_spectra.shape}")

#     local_args.num_bands = int(patches.shape[1])
#     local_args.max_classes = int(num_classes)

#     if local_args.base_classes is None:
#         local_args.base_classes = 6 if local_args.dataset in {"IP", "SA", "HC"} else max(2, num_classes // 2)
#     if local_args.increment is None:
#         remaining = max(1, num_classes - local_args.base_classes)
#         local_args.increment = 3 if remaining >= 3 else 1
#     if local_args.base_classes >= num_classes:
#         raise ValueError(f"base_classes={local_args.base_classes} must be < total classes={num_classes}")

#     inc_dataset = build_incremental_dataset(
#         local_args,
#         patches,
#         labels,
#         coords,
#         gt.shape,
#         gt.copy().astype(np.int64),
#         target_names=target_names,
#         label_policy=label_policy,
#         raw_spectra=raw_center_spectra,
#     )

#     if hasattr(inc_dataset, "inv_label_map"):
#         target_names_seq = []
#         for sid in range(inc_dataset.num_classes):
#             input_label = inc_dataset.inv_label_map[sid]
#             target_names_seq.append(target_names[int(input_label)] if int(input_label) < len(target_names) else f"Class {sid}")
#     else:
#         target_names_seq = list(target_names)
#     inc_dataset.target_names = target_names_seq

#     if label_policy is not None and not bool(label_policy.get("has_background", True)):
#         if 0 in label_policy.get("raw_class_values", []) and 0 not in np.unique(labels):
#             raise RuntimeError(
#                 "Label policy says raw class 0 is real, but label 0 is missing after ImageCubes. "
#                 "The loader is still treating class 0 as background."
#             )

#     run_tag = "base_only" if local_args.base_only else "full_necil"
#     run_dir = os.path.join(
#         local_args.save_dir,
#         local_args.dataset,
#         f"patch_{local_args.patch_size}",
#         f"{run_tag}_run_{run_idx + 1}_seed_{local_args.seed}",
#     )
#     os.makedirs(run_dir, exist_ok=True)
#     local_args.run_dir = run_dir
#     local_args.save_dir = run_dir
#     # Save an initial config before model construction. Adaptive-boundary state is
#     # written again immediately after model/trainer construction below.
#     save_run_config(local_args, run_dir)
#     print(f"Run directory: {run_dir}")
#     print(f"Phases: {getattr(inc_dataset, 'num_phases', 'unknown')} | class_order={getattr(inc_dataset, 'class_order', None)}")

#     model = NECILModel(local_args).to(device)
#     trainer = Trainer(model, inc_dataset, local_args)
#     assert_clean_runtime_stack(model, trainer, local_args)
#     save_run_config(local_args, run_dir, model=model)
#     evaluator = NECILEvaluator()

#     history: Dict[str, Any] = {
#         "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
#         "val_old_acc": [], "val_new_acc": [], "val_hm": [],
#         "phase_boundaries": [], "phase_ids": [],
#     }
#     phase_results: Dict[int, Dict[str, Any]] = {}
#     start_time = time.time()

#     dataset_total_phases = int(getattr(inc_dataset, "num_phases", 1))
#     if local_args.base_only:
#         total_phases = 1
#     else:
#         total_phases = dataset_total_phases
#         max_phases = int(getattr(local_args, "max_phases", 0) or 0)
#         max_train_phase = int(getattr(local_args, "max_train_phase", -1) or -1)
#         if max_phases > 0:
#             total_phases = min(total_phases, max_phases)
#         elif max_train_phase >= 0:
#             total_phases = min(total_phases, max_train_phase + 1)
#     print(f"Training phases: 0..{max(total_phases - 1, 0)} of dataset phases={dataset_total_phases}")

#     for phase in range(total_phases):
#         old_class_count = 0 if phase == 0 else len(inc_dataset.get_classes_up_to_phase(phase - 1))
#         seen_classes = inc_dataset.get_classes_up_to_phase(phase)
#         phase_dir = os.path.join(run_dir, f"phase_{phase}")
#         os.makedirs(phase_dir, exist_ok=True)

#         if phase == 0:
#             epochs = int(local_args.epochs_base)
#             lr = float(local_args.lr)
#         else:
#             epochs = int(local_args.epochs_inc)
#             lr = float(local_args.lr_inc if float(local_args.lr_inc) > 0 else local_args.lr)

#         print("\n" + "=" * 80)
#         print(f"[Phase {phase}] old_class_count={old_class_count} | seen={seen_classes} | epochs={epochs} | lr={lr}")
#         print("=" * 80)

#         phase_train_start = time.time()
#         phase_history = trainer.train_phase(
#             phase=phase,
#             epochs=epochs,
#             batch_size=local_args.batch_size,
#             lr=lr,
#         )
#         phase_train_time = time.time() - phase_train_start
#         _phase_history_extend(history, phase_history, phase)

#         if phase == 0:
#             _maybe_drop_base_head_after_phase0(model)

#         print(f"\n[Eval] Phase {phase} | cumulative geometry evaluation")
#         phase_eval_start = time.time()
#         y_pred, y_true = get_phase_predictions(model, inc_dataset, phase, device, local_args, batch_size=local_args.batch_size)
#         phase_eval_time = time.time() - phase_eval_start

#         evaluator_update_phase_compat(
#             evaluator,
#             phase=phase,
#             y_true=y_true,
#             y_pred=y_pred,
#             old_class_count=old_class_count,
#             seen_classes=seen_classes,
#         )
#         evaluator.print_summary()

#         geometry_diagnostics, geometry_diagnostics_paths = _collect_geometry_diagnostics_for_phase(
#             trainer=trainer,
#             phase_dir=phase_dir,
#             phase=phase,
#         )

#         report_info = save_phase_classification_report(
#             evaluator=evaluator,
#             phase=phase,
#             y_true=y_true,
#             y_pred=y_pred,
#             phase_dir=phase_dir,
#             target_names_seq=target_names_seq,
#             seen_classes=seen_classes,
#             old_class_count=old_class_count,
#             enabled=bool(local_args.save_classification_report),
#             tr_time=phase_train_time,
#             te_time=phase_eval_time,
#         )

#         metrics = evaluator.phase_history.get(phase, {})
#         phase_eval_results = evaluate_phase_model(model, inc_dataset, phase, device, local_args, batch_size=local_args.batch_size)

#         phase_results[phase] = {
#             "phase": phase,
#             "old_class_count": old_class_count,
#             "seen_classes": seen_classes,
#             "metrics": metrics,
#             "eval_results": phase_eval_results,
#             "classification_report": report_info,
#             "geometry_diagnostics": geometry_diagnostics,
#             "geometry_diagnostics_paths": geometry_diagnostics_paths,
#             "train_time_sec": phase_train_time,
#             "runtime_contract": runtime_contract_summary(local_args),
#             "eval_time_sec": phase_eval_time,
#         }

#         torch.save(
#             _build_checkpoint_payload(
#                 model=model,
#                 args=local_args,
#                 extra={
#                     "phase": phase,
#                     "metrics": metrics,
#                     "history": phase_history if isinstance(phase_history, dict) else None,
#                     "classification_report": report_info,
#                     "geometry_diagnostics": geometry_diagnostics,
#                     "geometry_diagnostics_paths": geometry_diagnostics_paths,
#                     "target_names_seq": target_names_seq,
#                     "target_names_raw": target_names,
#                     "class_order": getattr(inc_dataset, "class_order", None),
#                     "label_map": getattr(inc_dataset, "label_map", None),
#                     "inv_label_map": getattr(inc_dataset, "inv_label_map", None),
#                     "label_policy": label_policy,
#                 },
#             ),
#             os.path.join(phase_dir, "checkpoint.pth"),
#         )

#         if not local_args.skip_phase_maps:
#             _call_predict_phase_grid_compat(
#                 model=model,
#                 dataset_manager=inc_dataset,
#                 phase=phase,
#                 target_names=target_names_seq,
#                 save_dir=phase_dir,
#                 device=local_args.device,
#                 patch_size=local_args.patch_size,
#                 classifier_mode=_phase_classifier_mode(local_args, phase),
#                 semantic_mode="identity",
#                 class_cmap=local_args.viz_class_cmap,
#                 background_color=local_args.viz_background_color,
#                 save_numpy=local_args.viz_save_numpy,
#             )

#     elapsed_min = (time.time() - start_time) / 60.0
#     print(f"Full run done. Time: {elapsed_min:.1f} min")

#     final_phase = max(phase_results.keys())
#     final_metrics = evaluator.get_standard_metrics()
#     final_eval_results = evaluate_phase_model(model, inc_dataset, final_phase, device, local_args, batch_size=local_args.batch_size)

#     final_report_info = None
#     if bool(local_args.save_final_classification_report):
#         final_y_pred, final_y_true = get_phase_predictions(model, inc_dataset, final_phase, device, local_args, batch_size=local_args.batch_size)
#         final_report_info = save_phase_classification_report(
#             evaluator=evaluator,
#             phase=final_phase,
#             y_true=final_y_true,
#             y_pred=final_y_pred,
#             phase_dir=run_dir,
#             target_names_seq=target_names_seq,
#             seen_classes=inc_dataset.get_classes_up_to_phase(final_phase),
#             old_class_count=0 if final_phase == 0 else len(inc_dataset.get_classes_up_to_phase(final_phase - 1)),
#             enabled=True,
#             tr_time=elapsed_min * 60.0,
#             te_time=phase_results[final_phase].get("eval_time_sec", 0.0),
#         )

#     try:
#         plot_training_history(history, os.path.join(run_dir, "training_history.png"))
#     except Exception as exc:
#         print(f"[WARN] Could not plot training history: {exc}")

#     torch.save(
#         _build_checkpoint_payload(
#             model=model,
#             args=local_args,
#             extra={
#                 "final_phase": final_phase,
#                 "phase_results": phase_results,
#                 "eval_results": final_eval_results,
#                 "final_metrics": final_metrics,
#                 "history": history,
#                 "final_classification_report": final_report_info,
#                 "target_names_seq": target_names_seq,
#                 "target_names_raw": target_names,
#                 "class_order": getattr(inc_dataset, "class_order", None),
#                 "label_map": getattr(inc_dataset, "label_map", None),
#                 "inv_label_map": getattr(inc_dataset, "inv_label_map", None),
#                 "label_policy": label_policy,
#                 "evaluator": evaluator.to_dict() if hasattr(evaluator, "to_dict") else None,
#             },
#         ),
#         os.path.join(run_dir, "final_model.pth"),
#     )

#     write_full_run_report(
#         report_path=os.path.join(run_dir, f"patch{local_args.patch_size}_NECIL_GEOMETRY_REPORT.txt"),
#         local_args=local_args,
#         args=args,
#         run_idx=run_idx,
#         final_metrics=final_metrics,
#         final_eval_results=final_eval_results,
#         evaluator=evaluator,
#         target_names_seq=target_names_seq,
#         label_policy=label_policy,
#         phase_results=phase_results,
#         final_report_info=final_report_info,
#     )

#     return {
#         "run_idx": run_idx,
#         "seed": local_args.seed,
#         "run_dir": run_dir,
#         "final_phase": final_phase,
#         "final_metrics": final_metrics,
#         "eval_results": final_eval_results,
#         "phase_results": phase_results,
#         "final_classification_report": final_report_info,
#     }


# def write_full_run_report(
#     report_path: str,
#     local_args,
#     args,
#     run_idx: int,
#     final_metrics: Dict[str, Any],
#     final_eval_results: Dict[str, Any],
#     evaluator,
#     target_names_seq: List[str],
#     label_policy: Optional[Dict[str, Any]] = None,
#     phase_results: Optional[Dict[int, Dict[str, Any]]] = None,
#     final_report_info: Optional[Dict[str, Any]] = None,
# ):
#     os.makedirs(os.path.dirname(report_path), exist_ok=True)
#     phase_results = phase_results or {}

#     a_last = _metric_get(final_metrics, "A_last (Final Accuracy)", default=final_eval_results.get("overall_accuracy", 0.0))
#     a_avg = _metric_get(final_metrics, "A_avg (Avg Accuracy)", "A_avg (Avg Inc Accuracy)", default=a_last)

#     report_keys = [
#         "dataset", "patch_size", "pca_components", "base_classes", "increment", "base_only",
#         "base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode", "incremental_update_mode",
#         "use_adaptive_boundary", "boundary_radius_min", "boundary_radius_max", "boundary_init_radius",
#         "boundary_radius_reg_weight", "boundary_old_new_constraint_weight",
#         "boundary_old_new_margin_base", "boundary_old_new_margin_scale",
#         "adaptive_boundary_loss_weight", "adaptive_boundary_lr", "freeze_old_boundaries",
#         "use_boundary_geometry_replay", "boundary_replay_risk_threshold", "boundary_replay_overlap_threshold",
#         "boundary_replay_samples_per_pair", "boundary_replay_max_pairs", "scbgr_commit_only_if_safe", "unified_loss_weight",
#         "unified_admission_weight", "unified_subspace_weight", "unified_rank_weight", "unified_volume_weight", "unified_trust_weight",
#         "adapter_bottleneck", "adapter_max_scale", "adapter_lr", "adapter_weight_decay", "g2rpa_adapter_weight",
#         "adapter_old_delta_weight", "adapter_old_gate_weight", "adapter_old_energy_weight", "adapter_old_margin_weight",
#         "adapter_delta_weight", "adapter_new_gate_weight", "adapter_new_gate_target", "adapter_new_gate_max_target",
#         "d_model", "subspace_rank",
#         "refine_new_descriptors", "descriptor_refine_steps", "descriptor_refine_lr", "descriptor_trust_weight",
#         "descriptor_refine_max_mean_shift", "descriptor_refine_max_logvar_shift",
#         "boundary_preserve_weight", "boundary_preserve_overlap_weight", "boundary_preserve_center_weight",
#         "boundary_preserve_volume_weight", "boundary_preserve_band_weight", "max_old_new_risk",
#         "max_old_new_overlap", "use_boundary_projection", "boundary_projection_strength",
#         "boundary_projection_mean_push", "boundary_projection_var_shrink", "boundary_projection_overlap_threshold",
#         "boundary_projection_topk_old",
#         "use_incremental_adapter", "incremental_adapter_scale", "use_bicyc_geometry_cycle",
#         "use_spectral_geometry", "spectral_rank", "spectral_energy_weight", "spectral_derivative_weight", "spectral_second_derivative_weight", "spectral_require_physical_summary", "spectral_summary_is_physical", "raw_spectral_summary_is_physical", "band_energy_weight", "base_spectral_shape_weight", "risk_spectral_shape_weight", "residual_variance_scale", "invalid_class_energy",
#         "rank_energy_threshold", "spectral_rank_energy_threshold",
#         "base_ce_weight", "base_gics_weight", "base_gics_temperature",
#         "base_gics_spectral_temperature", "base_gics_band_temperature",
#         "base_gics_feature_weight", "base_gics_spectral_weight", "base_gics_band_weight",
#         "base_gics_same_class_positive", "base_gics_class_balanced", "base_gics_normalize",
#         "base_gics_key_noise_std", "base_gics_key_band_drop",
#         "base_gics_key_spatial_drop", "base_gics_key_scale_jitter",
#         "use_mssl_loss", "mssl_loss_type", "mssl_weight", "mssl_inc_weight",
#         "mssl_margin", "mssl_temperature", "mssl_neg_k", "mssl_spatial_radius",
#         "mssl_same_label_positive", "mssl_use_labels_for_negatives", "mssl_signed_neg_weight",
#         "refresh_before_validation", "validation_refresh_every", "bank_refresh_every",
#         "use_sglat_transport", "use_geometry_transport", "allow_old_model_transport", "allow_transport_without_adapter",
#         "transport_type", "transport_ema", "transport_identity_blend", "transport_batches", "transport_low_rank",
#         "transport_after_adapter_epoch", "transport_min_reliability_gate", "transport_max_a_minus_i_fro",
#         "transport_max_b_norm", "transport_residual_scale", "transport_min_rmse_gain",
#         "transport_max_rmse_ratio", "transport_min_old_anchor_acc",
#         "candidate_admission_mode", "risk_aware_descriptor_correction", "descriptor_correction_risk_threshold", "descriptor_correction_overlap_threshold", "descriptor_correction_basis_strength", "descriptor_correction_mean_push", "descriptor_correction_var_shrink", "risk_sep_weight", "risk_sep_overlap_target",
#         "best_state_metric", "strict_non_exemplar", "early_stop_patience", "base_early_stop_patience", "incremental_early_stop_patience", "epochs_base", "epochs_inc", "lr", "lr_inc", "gfa_weight", "gfa_samples_per_class", "joint_old_new_ce_weight", "batch_size", "seed", "max_train_phase", "max_phases",
#     ]

#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(f"SGLAT-HSI Report - {local_args.dataset}\n")
#         f.write(f"Run: {run_idx + 1}/{args.num_runs} | Seed: {local_args.seed}\n")
#         f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 70 + "\n")
#         f.write(f"Final phase: {max(phase_results.keys()) if phase_results else 0}\n")
#         f.write(f"A_last/OA: {a_last:.2f}%\n")
#         f.write(f"A_avg: {a_avg:.2f}%\n")
#         f.write("Architecture: phase 0 uses unified_spectral_geometry_loss(phase='base') to build SRPGR low-rank GeometryBank states. Incremental phases use SGLAT-HSI: old-model current-sample paired features estimate bounded low-rank residual transport, GeometryBank transports frozen old descriptors, new rows are admitted as provisional candidate descriptors with risk correction, and boundary-preserving descriptor optimization constrains new means/bases/variances with old/new tangent projection and boundary replay. Early stopping is disabled for both base and incremental phases. No KD, no raw old patches, no stored old features, no BiCyc/projection/backbone plasticity.\n")
#         f.write("=" * 70 + "\n\n")

#         if label_policy is not None:
#             f.write("Label Policy:\n")
#             f.write(json.dumps(make_json_serializable(label_policy), indent=2) + "\n\n")

#         f.write("Configuration:\n")
#         for key in report_keys:
#             if hasattr(local_args, key):
#                 f.write(f"{key}: {getattr(local_args, key)}\n")

#         f.write("\nPhase Metrics:\n")
#         for p in sorted(phase_results.keys()):
#             r = phase_results[p]
#             m = r.get("metrics", {}) or {}
#             f.write(
#                 f"Phase {p}: OA={m.get('overall_accuracy', m.get('acc', 0)):.2f}%, "
#                 f"AA={m.get('average_accuracy', 0):.2f}%, "
#                 f"Kappa={m.get('kappa', 0):.2f}%, "
#                 f"F1={m.get('f1_macro', 0):.2f}%, "
#                 f"old_class_count={r.get('old_class_count', 0)}, seen={r.get('seen_classes', [])}\n"
#             )
#             if r.get("classification_report"):
#                 f.write("  Classification report files: " + json.dumps(make_json_serializable(r.get("classification_report"))) + "\n")
#             _write_geometry_diagnostics_section(
#                 f,
#                 geometry_diagnostics=r.get("geometry_diagnostics"),
#                 geometry_diagnostics_paths=r.get("geometry_diagnostics_paths"),
#                 max_rows=30,
#             )
#             f.write("\n" + "-" * 70 + "\n")

#         if final_report_info:
#             f.write("\nFinal Classification Report Files:\n")
#             f.write(json.dumps(make_json_serializable(final_report_info), indent=2) + "\n")

#         f.write("\nFinal Per-Class Acc:\n")
#         for cls, acc in final_eval_results.get("per_class_accuracy", {}).items():
#             name = target_names_seq[int(cls)] if int(cls) < len(target_names_seq) else f"Class {cls}"
#             f.write(f"  {cls} ({name}): {acc:.2f}%\n")

#     print(f"[Report] Saved full NECIL geometry report to: {report_path}")




# def _phase_metric_float(phase_result: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
#     """Robust metric getter for evaluator variants and saved eval_results."""
#     if not isinstance(phase_result, dict):
#         return float(default)
#     metrics = phase_result.get("metrics", {}) or {}
#     eval_results = phase_result.get("eval_results", {}) or {}
#     for src in (metrics, eval_results):
#         if not isinstance(src, dict):
#             continue
#         for key in keys:
#             if key in src:
#                 try:
#                     return float(src[key])
#                 except Exception:
#                     pass
#     return float(default)


# def _phase_result_compact(phase: int, phase_result: Dict[str, Any]) -> Dict[str, Any]:
#     """Compact per-phase summary: base and every incremental phase."""
#     phase = int(phase)
#     old_class_count = int(phase_result.get("old_class_count", 0)) if isinstance(phase_result, dict) else 0
#     seen_classes = phase_result.get("seen_classes", []) if isinstance(phase_result, dict) else []
#     return {
#         "phase": phase,
#         "old_class_count": old_class_count,
#         "seen_classes": seen_classes,
#         "oa": _phase_metric_float(phase_result, "overall_accuracy", "OA", "oa", "acc", default=0.0),
#         "aa": _phase_metric_float(phase_result, "average_accuracy", "AA", "aa", default=0.0),
#         "kappa": _phase_metric_float(phase_result, "kappa", "Kappa", default=0.0),
#         "f1_macro": _phase_metric_float(phase_result, "f1_macro", "macro_f1", "Macro-F1", default=0.0),
#         "old_acc": _phase_metric_float(phase_result, "old_accuracy", "old_acc", "Old Accuracy", "old", default=0.0),
#         "new_acc": _phase_metric_float(phase_result, "new_accuracy", "new_acc", "New Accuracy", "new", default=0.0),
#         "hm": _phase_metric_float(phase_result, "hm", "harmonic_mean", "H", "h", default=0.0),
#         "train_time_sec": float(phase_result.get("train_time_sec", 0.0) or 0.0),
#         "eval_time_sec": float(phase_result.get("eval_time_sec", 0.0) or 0.0),
#     }


# def build_phase_summary(all_run_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Aggregate base + incremental phase metrics across runs."""
#     phase_ids = sorted({
#         int(p)
#         for r in all_run_results
#         for p in (r.get("phase_results", {}) or {}).keys()
#     })
#     rows: List[Dict[str, Any]] = []
#     for phase in phase_ids:
#         compact_rows = []
#         for r in all_run_results:
#             phase_results = r.get("phase_results", {}) or {}
#             pr = phase_results.get(phase, phase_results.get(str(phase), None))
#             if isinstance(pr, dict):
#                 compact_rows.append(_phase_result_compact(phase, pr))
#         if not compact_rows:
#             continue

#         def ms(metric: str):
#             vals = [float(x.get(metric, 0.0)) for x in compact_rows]
#             return aggregate_metric(vals)

#         oa_m, oa_s = ms("oa")
#         aa_m, aa_s = ms("aa")
#         k_m, k_s = ms("kappa")
#         f1_m, f1_s = ms("f1_macro")
#         old_m, old_s = ms("old_acc")
#         new_m, new_s = ms("new_acc")
#         hm_m, hm_s = ms("hm")

#         rows.append({
#             "phase": phase,
#             "num_runs": len(compact_rows),
#             "old_class_count": int(compact_rows[0].get("old_class_count", 0)),
#             "seen_classes": compact_rows[0].get("seen_classes", []),
#             "oa_mean": oa_m, "oa_std": oa_s,
#             "aa_mean": aa_m, "aa_std": aa_s,
#             "kappa_mean": k_m, "kappa_std": k_s,
#             "f1_macro_mean": f1_m, "f1_macro_std": f1_s,
#             "old_acc_mean": old_m, "old_acc_std": old_s,
#             "new_acc_mean": new_m, "new_acc_std": new_s,
#             "hm_mean": hm_m, "hm_std": hm_s,
#         })
#     return rows


# def save_phase_summary_csv(path: str, phase_summary: List[Dict[str, Any]]) -> str:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     fieldnames = [
#         "phase", "num_runs", "old_class_count", "seen_classes",
#         "oa_mean", "oa_std", "aa_mean", "aa_std", "kappa_mean", "kappa_std",
#         "f1_macro_mean", "f1_macro_std", "old_acc_mean", "old_acc_std",
#         "new_acc_mean", "new_acc_std", "hm_mean", "hm_std",
#     ]
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for row in phase_summary:
#             r = dict(row)
#             r["seen_classes"] = json.dumps(make_json_serializable(r.get("seen_classes", [])))
#             w.writerow({k: r.get(k, "") for k in fieldnames})
#     return path

# def main():
#     args = parse_args()
#     args = validate_args(args)
#     seed_list = parse_seed_list(args.seed_list) or [args.seed + i for i in range(args.num_runs)]

#     all_run_results = []
#     for run_idx in range(args.num_runs):
#         all_run_results.append(run_single_experiment(args, run_idx, seed_list[run_idx]))

#     root_dir = os.path.join(args.save_dir, args.dataset, f"patch_{args.patch_size}")
#     os.makedirs(root_dir, exist_ok=True)

#     final_oa = [r["eval_results"].get("overall_accuracy", 0.0) for r in all_run_results]

#     # Base + incremental phase summary.
#     phase_summary = build_phase_summary(all_run_results)
#     base_phase_rows = [r for r in phase_summary if int(r.get("phase", -1)) == 0]
#     base_oa_mean = float(base_phase_rows[0]["oa_mean"]) if base_phase_rows else 0.0
#     base_oa_std = float(base_phase_rows[0]["oa_std"]) if base_phase_rows else 0.0

#     phase_summary_csv = save_phase_summary_csv(
#         os.path.join(root_dir, "multi_run_phase_summary.csv"),
#         phase_summary,
#     )

#     summary = {
#         "method": ("SGLAT-HSI" if _normalize_incremental_update_mode(getattr(args, "incremental_update_mode", "scbgr")) == "geometry_gated_adapter" else ("SGLAT-HSI-SSMR-ABLATION" if bool(getattr(args, "use_mssl_loss", False)) else "SGLAT-HSI")),
#         "num_runs": args.num_runs,
#         "seeds": seed_list,
#         "base_only": bool(args.base_only),
#         "base_OA_mean": base_oa_mean,
#         "base_OA_std": base_oa_std,
#         "final_OA_mean": aggregate_metric(final_oa)[0],
#         "final_OA_std": aggregate_metric(final_oa)[1],
#         "phase_summary": phase_summary,
#         "phase_summary_csv": phase_summary_csv,
#         "runs": all_run_results,
#     }

#     with open(os.path.join(root_dir, "multi_run_summary.json"), "w", encoding="utf-8") as f:
#         json.dump(make_json_serializable(summary), f, indent=2)

#     report_path = os.path.join(root_dir, "MULTI_RUN_REPORT.txt")
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(f"SGLAT-HSI Multi-Run Report - {args.dataset}\n")
#         f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 96 + "\n")
#         f.write(f"Runs: {args.num_runs}\n")
#         f.write(f"Seeds: {seed_list}\n")
#         f.write(f"base_only: {args.base_only}\n")
#         f.write(f"Base OA:  {summary['base_OA_mean']:.2f} ± {summary['base_OA_std']:.2f}\n")
#         f.write(f"Final OA: {summary['final_OA_mean']:.2f} ± {summary['final_OA_std']:.2f}\n")
#         f.write(f"Phase summary CSV: {phase_summary_csv}\n")
#         f.write("=" * 96 + "\n\n")

#         f.write("Phase-wise NECIL Summary\n")
#         f.write("-" * 96 + "\n")
#         f.write(
#             "phase | old_count | seen_classes        | "
#             "OA(mean±std)     AA(mean±std)     Kappa(mean±std)  "
#             "F1(mean±std)     Old(mean±std)    New(mean±std)    HM(mean±std)\n"
#         )
#         for row in phase_summary:
#             f.write(
#                 f"{int(row['phase']):5d} | "
#                 f"{int(row.get('old_class_count', 0)):9d} | "
#                 f"{str(row.get('seen_classes', []))[:19]:19s} | "
#                 f"{row['oa_mean']:6.2f}±{row['oa_std']:<6.2f} "
#                 f"{row['aa_mean']:6.2f}±{row['aa_std']:<6.2f} "
#                 f"{row['kappa_mean']:6.2f}±{row['kappa_std']:<6.2f} "
#                 f"{row['f1_macro_mean']:6.2f}±{row['f1_macro_std']:<6.2f} "
#                 f"{row['old_acc_mean']:6.2f}±{row['old_acc_std']:<6.2f} "
#                 f"{row['new_acc_mean']:6.2f}±{row['new_acc_std']:<6.2f} "
#                 f"{row['hm_mean']:6.2f}±{row['hm_std']:<6.2f}\n"
#             )

#         f.write("\nPer-run phase details\n")
#         f.write("-" * 96 + "\n")
#         for r in all_run_results:
#             f.write(
#                 f"Run {r['run_idx'] + 1} | Seed {r['seed']} | "
#                 f"FinalPhase={r.get('final_phase', 0)} | "
#                 f"FinalOA={r['eval_results'].get('overall_accuracy', 0.0):.2f} | "
#                 f"RunDir={r.get('run_dir', '')}\n"
#             )
#             phase_results = r.get("phase_results", {}) or {}
#             for phase in sorted(int(p) for p in phase_results.keys()):
#                 pr = phase_results.get(phase, phase_results.get(str(phase), {}))
#                 c = _phase_result_compact(phase, pr)
#                 f.write(
#                     f"  Phase {phase}: "
#                     f"OA={c['oa']:.2f}, AA={c['aa']:.2f}, Kappa={c['kappa']:.2f}, "
#                     f"F1={c['f1_macro']:.2f}, Old={c['old_acc']:.2f}, "
#                     f"New={c['new_acc']:.2f}, HM={c['hm']:.2f}, "
#                     f"old_count={c['old_class_count']}, seen={c['seen_classes']}\n"
#                 )
#             f.write("\n")

#     print("\n=== SUMMARY ===")
#     print(f"base_only: {args.base_only}")
#     print(f"Base OA:  {summary['base_OA_mean']:.2f} ± {summary['base_OA_std']:.2f}")
#     print(f"Final OA: {summary['final_OA_mean']:.2f} ± {summary['final_OA_std']:.2f}")
#     print("\nPhase-wise summary:")
#     for row in phase_summary:
#         print(
#             f"  Phase {int(row['phase'])}: "
#             f"OA={row['oa_mean']:.2f}±{row['oa_std']:.2f} | "
#             f"Old={row['old_acc_mean']:.2f}±{row['old_acc_std']:.2f} | "
#             f"New={row['new_acc_mean']:.2f}±{row['new_acc_std']:.2f} | "
#             f"HM={row['hm_mean']:.2f}±{row['hm_std']:.2f} | "
#             f"seen={row.get('seen_classes', [])}"
#         )
#     print(f"\nSaved: {os.path.join(root_dir, 'multi_run_summary.json')}")
#     print(f"Saved: {phase_summary_csv}")
#     print(f"Saved: {report_path}")
#     print("================================\n")



# if __name__ == "__main__":
#     main()
