from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from trainers.trainer_helpers import TrainerHelper
from trainers.base_phase_trainer import BasePhaseTrainer
from trainers.incremental_phase_trainer import IncrementalPhaseTrainer

try:
    from losses.loss import unified_spectral_geometry_loss
except Exception:  # pragma: no cover
    unified_spectral_geometry_loss = None


class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
    """PG-RGA trainer for strict non-exemplar HSI class-incremental learning.

    Active architecture:
        Base phase:
            Balanced CE + GICS + PGR in canonical projected z-space.

        Incremental phase:
            Frozen old GeometryBank + new rows + synthetic GeometryBank replay
            + geometry-gated residual adapter + joint old/new CE + geometry
            energy margins.

    Deliberately inactive in the main path:
        KD/teacher, prototypes, raw exemplars, transport, adaptive boundary,
        measured energy calibration, BiCyc/BSS/GDR, prompt/token memory.
    """

    # ------------------------------------------------------------------
    # Small config helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off", "none", "null", ""}:
                return False
        return bool(value)

    def _arg_bool(self, name: str, default: bool = False) -> bool:
        return self._as_bool(getattr(self.args, name, default), default=default)

    def _set_arg_default(self, name: str, value: Any) -> None:
        if not hasattr(self.args, name) or getattr(self.args, name) is None:
            try:
                setattr(self.args, name, value)
            except Exception:
                pass

    @staticmethod
    def _freeze_module_if_present(module: Optional[torch.nn.Module]) -> None:
        if module is None:
            return
        try:
            for p in module.parameters():
                p.requires_grad = False
        except Exception:
            pass
        try:
            module.eval()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Runtime defaults and architecture normalization
    # ------------------------------------------------------------------
    def _install_clean_runtime_defaults(self) -> None:
        """Install only defaults needed by the PG-RGA stack.

        Defaults are set only when the user/main.py did not provide a value.
        Forbidden legacy branches are then force-disabled in
        _force_clean_main_path_args().
        """
        defaults = {
            # Classifier energy contract.
            "base_classifier_mode": "geometry_only",
            "incremental_classifier_mode": "geometry_only",
            "eval_classifier_mode": "geometry_only",
            "use_logdet_energy": True,
            "logdet_energy_weight": 0.05,
            "logdet_normalize_by_dim": True,
            "center_logdet_energy": True,
            "energy_normalize_by_dim": True,
            "geometry_normalize_logits": False,
            "residual_variance_scale": 0.75,
            "reliability_energy_weight": 0.03,
            "invalid_class_energy": 1e6,

            # Mandatory base objective.
            "base_class_balance": True,
            "base_ce_weight": 1.0,
            "base_srpgr_weight": 1.0,
            "base_gics_weight": 0.20,
            "base_gics_temperature": 0.07,
            "pgr_weight": 0.10,
            "pgr_compact_weight": 0.15,
            "pgr_center_weight": 0.25,
            "pgr_subspace_weight": 0.15,
            "pgr_band_weight": 0.05,
            "pgr_volume_weight": 0.05,
            "pgr_center_margin": 1.10,
            "pgr_min_class_samples": 3,
            "pgr_subspace_min_samples": 6,
            "pgr_subspace_rank": 3,
            "pgr_max_class_variance": 0.75,
            "pgr_band_overlap_max": 0.65,
            "base_spectral_shape_weight": 0.0,
            "base_require_physical_spectral_shape": False,
            "strict_base_component_coverage": True,

            # GeometryBank extraction.
            "rank_energy_threshold": 0.90,
            "rank_eigen_ratio_threshold": 1e-2,
            "min_active_rank": 1,
            "geometry_variance_shrinkage": 0.25,
            "geom_var_floor": 5e-4,
            "geometry_bank_feature_space": "canonical",

            # Incremental PG-RGA path.
            "incremental_update_mode": "geometry_gated_adapter",
            "gfa_weight": 1.0,
            "gfa_samples_per_class": 48,
            "gfa_parallel_scale": 0.95,
            "gfa_residual_scale": 0.25,
            "gfa_reliability_gated": True,
            "joint_old_new_ce_weight": 1.0,
            "geometry_energy_margin_weight": 0.30,
            "geometry_energy_margin": 0.30,
            "old_new_invasion_weight": 0.50,
            "old_new_geometry_margin": 0.35,
            "adapter_lr": 1e-4,
            "adapter_weight_decay": 0.0,
            "adapter_bottleneck": 32,
            "adapter_max_scale": 0.10,
            "adapter_dropout": 0.0,
            "adapter_gate_bias_init": -3.0,
            "g2rpa_adapter_weight": 1.0,
            "adapter_old_delta_weight": 1.0,
            "adapter_old_gate_weight": 0.75,
            "adapter_old_energy_weight": 0.25,
            "adapter_old_margin_weight": 0.25,
            "adapter_delta_weight": 0.10,
            "adapter_new_gate_weight": 0.05,
            "adapter_new_gate_target": 0.25,
            "adapter_new_gate_max_target": 0.75,

            # Descriptor/new-row refinement kept for incremental row quality.
            "refine_new_descriptors": True,
            "use_descriptor_refinement": True,
            "descriptor_refine_steps": 5,
            "descriptor_refine_lr": 1e-3,
            "descriptor_trust_weight": 0.80,
            "descriptor_refine_max_mean_shift": 0.30,
            "descriptor_refine_max_logvar_shift": 0.50,
            "descriptor_subspace_collision_weight": 0.10,
            "descriptor_subspace_overlap_max": 0.35,
            "descriptor_center_margin_weight": 0.05,
            "descriptor_center_collision_weight": 0.05,
            "descriptor_center_margin": 0.50,
            "descriptor_volume_weight": 0.03,
            "descriptor_volume_control_weight": 0.03,
            "descriptor_volume_margin": 0.0,

            # Spectral summaries can shape base loss/bank but classifier remains geometry-only.
            "use_spectral_geometry": False,
            "spectral_energy_weight": 0.0,
            "band_energy_weight": 0.0,
            "spectral_require_physical_summary": True,
            "spectral_summary_is_physical": False,
            "raw_spectral_summary_is_physical": True,
            "external_spectra_are_physical": True,

            # Checkpoint/validation.
            "best_state_metric": "geometry_score",
            "refresh_before_validation": True,
            "validation_refresh_every": 1,
            "enforce_base_geometry_certificate": False,
            "base_cert_min_geom_acc": 95.0,
            "base_cert_min_reliability": 0.15,
            "base_cert_min_mean_reliability": 0.35,
            "base_cert_max_subspace_overlap": 0.55,
            "base_cert_max_geometry_conflict": 1.35,
            "base_cert_max_band_similarity": 0.90,
            "strict_updated_stack": True,
        }
        for k, v in defaults.items():
            self._set_arg_default(k, v)

    def _incremental_update_mode(self) -> str:
        raw = str(getattr(self.args, "incremental_update_mode", "geometry_gated_adapter")).lower().strip()
        aliases = {
            "": "geometry_gated_adapter",
            "none": "geometry_gated_adapter",
            "clean": "geometry_gated_adapter",
            "pg_rga": "geometry_gated_adapter",
            "pg-rga": "geometry_gated_adapter",
            "g2rpa": "geometry_gated_adapter",
            "g²rpa": "geometry_gated_adapter",
            "adapter": "geometry_gated_adapter",
            "gated_adapter": "geometry_gated_adapter",
            "geometry_adapter": "geometry_gated_adapter",
            "geometry_gated_adapter": "geometry_gated_adapter",
            # Legacy names are accepted but mapped to the actual PG-RGA update.
            "scbgr": "geometry_gated_adapter",
            "scb-gr": "geometry_gated_adapter",
            "rsgi": "geometry_gated_adapter",
            "descriptor": "geometry_gated_adapter",
            "descriptor_only": "geometry_gated_adapter",
        }
        mode = aliases.get(raw, raw)
        if mode != "geometry_gated_adapter":
            raise RuntimeError(
                f"Unsupported incremental_update_mode={raw!r}. The cleaned trainer uses "
                "incremental_update_mode=geometry_gated_adapter."
            )
        try:
            setattr(self.args, "incremental_update_mode", mode)
        except Exception:
            pass
        if hasattr(self.model, "incremental_update_mode"):
            self.model.incremental_update_mode = mode
        return mode

    def _adapter_mode_enabled(self) -> bool:
        return self._incremental_update_mode() == "geometry_gated_adapter"

    def _normalize_classifier_mode(self, mode: Optional[str], *, context: str = "runtime") -> str:
        raw = str(mode or "geometry_only").lower().strip()
        aliases = {
            "": "geometry_only",
            "none": "geometry_only",
            "geo": "geometry_only",
            "geometry": "geometry_only",
            "geometry-only": "geometry_only",
            "feature_geometry": "geometry_only",
            "low_rank_geometry": "geometry_only",
            "replay": "geometry_only",
            "synthetic_replay": "geometry_only",
            "srgp": "geometry_only",
            "srgp_geometry": "geometry_only",
            "spectral_geometry": "geometry_only",
            "spectral_residual": "geometry_only",
            "calibrated_geometry": "geometry_only",
            "topology_calibrated_geometry": "geometry_only",
        }
        normalized = aliases.get(raw, raw)
        if normalized not in {"geometry_only", "base_ce"}:
            raise RuntimeError(f"{context}: unsupported classifier mode={raw!r}; use geometry_only or base_ce.")
        return normalized

    def _force_clean_main_path_args(self) -> None:
        """Disable unused/forbidden branches and align public args."""
        mode = self._incremental_update_mode()
        forced = {
            "incremental_update_mode": mode,
            "base_classifier_mode": "geometry_only",
            "incremental_classifier_mode": "geometry_only",
            "eval_classifier_mode": "geometry_only",
            "geometry_normalize_logits": False,
            "use_incremental_adapter": False,   # legacy flag; PG-RGA uses geometry_plastic_adapter.
            "disable_incremental_adapter": False,
            "incremental_adapter_normalize": False,
            "allow_incremental_projection_training": False,
            "freeze_projection_during_incremental": True,
            "use_geometry_calibrator": False,
            "geometry_calibration_weight": 0.0,
            "use_energy_calibrator": False,
            "energy_calibration_weight": 0.0,
            "use_bicyc_geometry_cycle": False,
            "bicyc_geometry_cycle_weight": 0.0,
            "bicyc_cycle_weight": 0.0,
            "bss_weight": 0.0,
            "sym_bss_weight": 0.0,
            "gdr_weight": 0.0,
            "anchor_consistency_weight": 0.0,
            "use_sglat_transport": False,
            "use_geometry_transport": False,
            "allow_old_model_transport": False,
            "allow_transport_without_adapter": False,
            "use_boundary_geometry_replay": False,
            "use_adaptive_boundary": False,
            "use_boundary_projection": False,
            "boundary_preserve_weight": 0.0,
            "use_spectral_geometry": False,
            "spectral_energy_weight": 0.0,
            "band_energy_weight": 0.0,
            "early_stop_patience": 0,
            "base_early_stop_patience": 0,
            "incremental_early_stop_patience": 0,
        }
        for k, v in forced.items():
            try:
                setattr(self.args, k, v)
            except Exception:
                pass

        self.incremental_update_mode = mode
        self.use_geometry_transport = False
        self.use_sglat_transport = False
        self.use_boundary_geometry_replay = False
        self.use_adaptive_boundary = False
        self.use_energy_calibrator = False
        self.use_spectral_geometry = False
        self.spectral_energy_weight = 0.0
        self.band_energy_weight = 0.0

        if hasattr(self.model, "use_geometry_gated_adapter"):
            self.model.use_geometry_gated_adapter = bool(mode == "geometry_gated_adapter")
        if hasattr(self.model, "incremental_update_mode"):
            self.model.incremental_update_mode = mode
        for attr in ("use_geometry_transport", "use_sglat_transport", "use_geometry_calibrator", "use_bicyc_geometry_cycle", "use_incremental_adapter"):
            if hasattr(self.model, attr):
                setattr(self.model, attr, False)
        if hasattr(self.model, "freeze_geometry_calibrator"):
            self.model.freeze_geometry_calibrator()
        if hasattr(self.model, "freeze_energy_calibrator"):
            self.model.freeze_energy_calibrator()

    def _propagate_clean_energy_config_to_model(self) -> None:
        clf = getattr(self.model, "classifier", None)
        if clf is not None:
            pairs = {
                "use_logdet_energy": bool(self.use_logdet_energy),
                "logdet_energy_weight": float(self.logdet_energy_weight),
                "logdet_normalize_by_dim": bool(self.logdet_normalize_by_dim),
                "center_logdet_energy": bool(self.center_logdet_energy),
                "energy_normalize_by_dim": bool(self.energy_normalize_by_dim),
                "normalize_energy_by_dim": bool(self.energy_normalize_by_dim),
                "reliability_energy_weight": float(self.reliability_energy_weight),
                "residual_variance_scale": float(self.residual_variance_scale),
                "invalid_class_energy": float(self.invalid_class_energy),
                "use_spectral_geometry": False,
                "spectral_energy_weight": 0.0,
                "band_energy_weight": 0.0,
                "use_adaptive_boundary": False,
            }
            for k, v in pairs.items():
                if hasattr(clf, k):
                    try:
                        setattr(clf, k, v)
                    except Exception:
                        pass
            if hasattr(clf, "normalize_logits"):
                clf.normalize_logits = False
            if hasattr(clf, "freeze_all_adaptation"):
                clf.freeze_all_adaptation()
        if hasattr(self.model, "use_adaptive_boundary"):
            self.model.use_adaptive_boundary = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, model, dataset, args) -> None:
        self.args = args
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.model = model.to(self.device)
        self.dataset = dataset
        self.save_dir = str(getattr(args, "save_dir", "./checkpoints"))
        os.makedirs(self.save_dir, exist_ok=True)

        self.debug = self._arg_bool("debug_verbose", False) or os.environ.get("NECIL_DEBUG", "0") == "1"
        self.base_only = self._arg_bool("base_only", False)
        self.disable_incremental_training = self._arg_bool("disable_incremental_training", False)

        self._install_clean_runtime_defaults()
        self.incremental_update_mode = self._incremental_update_mode()

        # Core geometry/classifier settings.
        self.subspace_rank = int(getattr(args, "subspace_rank", 5))
        self.geom_var_floor = float(getattr(args, "geom_var_floor", 5e-4))
        self.reliability_energy_weight = float(getattr(args, "reliability_energy_weight", 0.03))
        self.energy_normalize_by_dim = self._arg_bool("energy_normalize_by_dim", True)
        self.residual_variance_scale = float(getattr(args, "residual_variance_scale", 0.75))
        self.invalid_class_energy = float(getattr(args, "invalid_class_energy", 1e6))
        self.use_logdet_energy = self._arg_bool("use_logdet_energy", True)
        self.logdet_energy_weight = float(getattr(args, "logdet_energy_weight", 0.05))
        self.logdet_normalize_by_dim = self._arg_bool("logdet_normalize_by_dim", True)
        self.center_logdet_energy = self._arg_bool("center_logdet_energy", True)

        # Mandatory base phase.
        self.base_ce_weight = float(getattr(args, "base_ce_weight", 1.0))
        self.base_srpgr_weight = float(getattr(args, "base_srpgr_weight", 1.0))
        self.base_gics_weight = float(getattr(args, "base_gics_weight", 0.20))
        self.pgr_weight = float(getattr(args, "pgr_weight", 0.10))
        self.pgr_compact_weight = float(getattr(args, "pgr_compact_weight", 0.15))
        self.pgr_center_weight = float(getattr(args, "pgr_center_weight", 0.25))
        self.pgr_subspace_weight = float(getattr(args, "pgr_subspace_weight", 0.15))
        self.pgr_band_weight = float(getattr(args, "pgr_band_weight", 0.05))
        self.pgr_volume_weight = float(getattr(args, "pgr_volume_weight", 0.05))
        self.pgr_center_margin = float(getattr(args, "pgr_center_margin", 1.10))
        self.pgr_min_class_samples = int(getattr(args, "pgr_min_class_samples", 3))
        self.pgr_subspace_min_samples = int(getattr(args, "pgr_subspace_min_samples", 6))
        self.pgr_subspace_rank = int(getattr(args, "pgr_subspace_rank", 3))
        self.pgr_max_class_variance = float(getattr(args, "pgr_max_class_variance", 0.75))
        self.pgr_band_overlap_max = float(getattr(args, "pgr_band_overlap_max", 0.65))
        self.pgr_normalize_features = self._arg_bool("pgr_normalize_features", True)

        # GeometryBank extraction/rank.
        self.rank_energy_threshold = float(getattr(args, "rank_energy_threshold", 0.90))
        self.rank_eigen_ratio_threshold = float(getattr(args, "rank_eigen_ratio_threshold", 1e-2))
        self.min_active_rank = int(getattr(args, "min_active_rank", 1))
        self.geometry_variance_shrinkage = float(getattr(args, "geometry_variance_shrinkage", 0.25))

        # Optimizer/checkpoint policy.
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))
        self.ce_logit_clip = float(getattr(args, "ce_logit_clip", 50.0))
        self.grad_clip_base = float(getattr(args, "grad_clip_base", 1.0))
        self.grad_clip_inc = float(getattr(args, "grad_clip_inc", 0.5))
        self.refresh_before_validation = self._arg_bool("refresh_before_validation", True)
        self.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))
        self.bank_refresh_every = int(getattr(args, "bank_refresh_every", 0))
        self.best_state_metric = str(getattr(args, "best_state_metric", "geometry_score")).lower().strip() or "geometry_score"
        self.early_stop_patience = 0

        # Incremental PG-RGA objective.
        self.gfa_weight = float(getattr(args, "gfa_weight", 1.0))
        self.gfa_samples_per_class = int(getattr(args, "gfa_samples_per_class", 48))
        self.gfa_parallel_scale = float(getattr(args, "gfa_parallel_scale", 0.95))
        self.gfa_residual_scale = float(getattr(args, "gfa_residual_scale", 0.25))
        self.gfa_reliability_gated = self._arg_bool("gfa_reliability_gated", True)
        self.joint_old_new_ce_weight = float(getattr(args, "joint_old_new_ce_weight", 1.0))
        self.geometry_energy_margin_weight = float(getattr(args, "geometry_energy_margin_weight", 0.30))
        self.geometry_energy_margin = float(getattr(args, "geometry_energy_margin", 0.30))
        self.old_new_invasion_weight = float(getattr(args, "old_new_invasion_weight", 0.50))
        self.old_new_geometry_margin = float(getattr(args, "old_new_geometry_margin", 0.35))
        self.use_pretrain_incremental_baseline = self._arg_bool("use_pretrain_incremental_baseline", True)

        # Adapter controls.
        self.adapter_lr = float(getattr(args, "adapter_lr", 1e-4))
        self.adapter_weight_decay = float(getattr(args, "adapter_weight_decay", 0.0))
        self.g2rpa_adapter_weight = float(getattr(args, "g2rpa_adapter_weight", 1.0))
        self.adapter_old_delta_weight = float(getattr(args, "adapter_old_delta_weight", 1.0))
        self.adapter_old_gate_weight = float(getattr(args, "adapter_old_gate_weight", 0.75))
        self.adapter_old_energy_weight = float(getattr(args, "adapter_old_energy_weight", 0.25))
        self.adapter_old_margin_weight = float(getattr(args, "adapter_old_margin_weight", 0.25))
        self.adapter_delta_weight = float(getattr(args, "adapter_delta_weight", 0.10))
        self.adapter_new_gate_weight = float(getattr(args, "adapter_new_gate_weight", 0.05))
        self.adapter_new_gate_target = float(getattr(args, "adapter_new_gate_target", 0.25))
        self.adapter_new_gate_max_target = float(getattr(args, "adapter_new_gate_max_target", 0.75))

        # New descriptor row refinement knobs consumed by IncrementalPhaseTrainer.
        self.refine_new_descriptors = self._arg_bool("refine_new_descriptors", True)
        self.use_descriptor_refinement = self._arg_bool("use_descriptor_refinement", self.refine_new_descriptors)
        self.descriptor_refine_steps = int(getattr(args, "descriptor_refine_steps", 5))
        self.descriptor_refine_lr = float(getattr(args, "descriptor_refine_lr", 1e-3))
        self.descriptor_trust_weight = float(getattr(args, "descriptor_trust_weight", 0.80))
        self.descriptor_refine_max_mean_shift = float(getattr(args, "descriptor_refine_max_mean_shift", 0.30))
        self.descriptor_refine_max_logvar_shift = float(getattr(args, "descriptor_refine_max_logvar_shift", 0.50))
        self.descriptor_refine_steps_per_epoch = int(getattr(args, "descriptor_refine_steps_per_epoch", 0))
        self.descriptor_refine_grad_clip = float(getattr(args, "descriptor_refine_grad_clip", 1.0))
        self.descriptor_subspace_collision_weight = float(getattr(args, "descriptor_subspace_collision_weight", 0.10))
        self.descriptor_subspace_overlap_max = float(getattr(args, "descriptor_subspace_overlap_max", 0.35))
        self.descriptor_center_margin_weight = float(getattr(args, "descriptor_center_margin_weight", 0.05))
        self.descriptor_center_collision_weight = float(getattr(args, "descriptor_center_collision_weight", self.descriptor_center_margin_weight))
        self.descriptor_center_margin = float(getattr(args, "descriptor_center_margin", 0.50))
        self.descriptor_volume_weight = float(getattr(args, "descriptor_volume_weight", 0.03))
        self.descriptor_volume_control_weight = float(getattr(args, "descriptor_volume_control_weight", self.descriptor_volume_weight))
        self.descriptor_volume_margin = float(getattr(args, "descriptor_volume_margin", 0.0))

        # Explicitly inactive legacy branches. Attributes remain for compatibility only.
        self.use_boundary_geometry_replay = False
        self.use_risk_weighted_replay = False
        self.use_energy_calibrator = False
        self.use_adaptive_boundary = False
        self.use_geometry_transport = False
        self.use_sglat_transport = False
        self.allow_incremental_projection_training = False
        self.freeze_projection_during_incremental = True
        self.use_spectral_geometry = False
        self.spectral_energy_weight = 0.0
        self.band_energy_weight = 0.0
        self.bss_weight = 0.0
        self.sym_bss_weight = 0.0
        self.gdr_weight = 0.0
        self.anchor_consistency_weight = 0.0
        self.geometry_calibration_weight = 0.0
        self.energy_calibration_weight = 0.0
        self.boundary_preserve_weight = 0.0
        self.use_boundary_projection = False

        self._force_clean_main_path_args()
        self._propagate_clean_energy_config_to_model()
        self._assert_global_architecture_contract()
        self._assert_updated_stack_contract(phase=0)
        self._set_base_trainable_params()

    # ------------------------------------------------------------------
    # Stack contract and trainability
    # ------------------------------------------------------------------
    def _assert_updated_stack_contract(self, phase: Optional[int] = None) -> None:
        strict = self._arg_bool("strict_updated_stack", True)
        phase_i = 0 if phase is None else int(phase)
        missing: List[str] = []

        for attr in ("extract_projected_features", "get_subspace_bank"):
            if not hasattr(self.model, attr):
                missing.append(f"model.{attr}")
        if phase_i > 0:
            for attr in ("geometry_plastic_adapter", "adapt_projected_features", "compute_logits_from_features"):
                if not hasattr(self.model, attr):
                    missing.append(f"model.{attr}")

        clf = getattr(self.model, "classifier", None)
        if clf is None or not (hasattr(clf, "forward") or hasattr(clf, "compute_geometry_energy") or hasattr(clf, "geometry_energy_from_bank")):
            missing.append("model.classifier strict GeometryEnergyClassifier")
        gb = getattr(self.model, "geometry_bank", None)
        if gb is None:
            missing.append("model.geometry_bank")
        else:
            for attr in ("get_valid_mask", "assert_bank_valid"):
                if not hasattr(gb, attr):
                    missing.append(f"geometry_bank.{attr}")
            if phase_i > 0 and not any(hasattr(gb, a) for a in ("sample_replay", "sample_synthetic_features")):
                missing.append("geometry_bank.sample_replay/sample_synthetic_features")

        for attr in ("_safe_get_subspace_bank", "global_to_seen_local", "assert_bank_ready_for_seen_classes"):
            if not hasattr(self, attr):
                missing.append(f"TrainerHelper.{attr}")
        if unified_spectral_geometry_loss is None:
            missing.append("losses.loss.unified_spectral_geometry_loss")

        if missing:
            msg = "PG-RGA stack contract failed; stale/missing component(s): " + ", ".join(missing)
            if strict:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}")

    def _assert_global_architecture_contract(self) -> None:
        mode = self._incremental_update_mode()
        for key in ("base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode"):
            self._normalize_classifier_mode(getattr(self.args, key, "geometry_only"), context=key)

        forbidden_true = [
            "use_geometry_calibrator", "use_incremental_adapter", "use_bicyc_geometry_cycle",
            "allow_incremental_projection_training", "use_sglat_transport", "use_geometry_transport",
            "allow_old_model_transport", "use_boundary_geometry_replay", "use_adaptive_boundary",
            "use_energy_calibrator", "geometry_normalize_logits",
        ]
        bad = [k for k in forbidden_true if self._arg_bool(k, False)]
        if bad:
            raise RuntimeError(f"Forbidden PG-RGA switches are active: {bad}")
        for key in ("bss_weight", "sym_bss_weight", "gdr_weight", "anchor_consistency_weight", "geometry_calibration_weight", "energy_calibration_weight"):
            if abs(float(getattr(self.args, key, 0.0))) > 0.0:
                raise RuntimeError(f"{key} must be 0.0 in the PG-RGA architecture path.")
        if mode == "geometry_gated_adapter":
            if not hasattr(self.model, "geometry_plastic_adapter"):
                raise RuntimeError("PG-RGA requires NECILModel.geometry_plastic_adapter.")
            max_scale = float(getattr(getattr(self.model, "geometry_plastic_adapter", None), "max_scale", getattr(self.args, "adapter_max_scale", 0.0)) or 0.0)
            if max_scale <= 0.0:
                raise RuntimeError("geometry_gated_adapter selected but adapter_max_scale <= 0.")

    def _set_base_trainable_params(self) -> None:
        self._force_clean_main_path_args()
        self._propagate_clean_energy_config_to_model()
        for name, p in self.model.named_parameters():
            blocked = (
                name.startswith("classifier.") or name.startswith("geometry_bank.")
                or name.startswith("geometry_calibrator.") or name.startswith("geometry_cycle_calibrator.")
                or name.startswith("incremental_adapter.") or name.startswith("geometry_plastic_adapter.")
                or name.startswith("base_ce_head.")
            )
            p.requires_grad = not blocked
        if hasattr(self.model, "freeze_energy_calibrator"):
            self.model.freeze_energy_calibrator()
        if hasattr(self.model, "freeze_geometry_calibrator"):
            self.model.freeze_geometry_calibrator()
        if hasattr(self.model, "freeze_geometry_plastic_adapter"):
            self.model.freeze_geometry_plastic_adapter()

    def _set_incremental_trainable_params(self, old_class_count: int = 0) -> List[torch.nn.Parameter]:
        old_class_count = int(old_class_count)
        self._force_clean_main_path_args()
        self._propagate_clean_energy_config_to_model()
        self._incremental_update_mode()

        for _, p in self.model.named_parameters():
            p.requires_grad = False
        if hasattr(self.model, "set_incremental_mode"):
            try:
                self.model.set_incremental_mode(
                    phase=int(getattr(self.model, "current_phase", 1)),
                    old_class_count=old_class_count,
                    train_classifier_calibration=False,
                    train_geometry_adapter=True,
                )
            except TypeError:
                self.model.set_incremental_mode(phase=int(getattr(self.model, "current_phase", 1)), old_class_count=old_class_count)
        if hasattr(self.model, "enable_incremental_adapter"):
            self.model.enable_incremental_adapter()
        if hasattr(self.model, "unfreeze_geometry_plastic_adapter"):
            self.model.unfreeze_geometry_plastic_adapter()
        elif hasattr(self.model, "geometry_plastic_adapter"):
            for p in self.model.geometry_plastic_adapter.parameters():
                p.requires_grad = True

        if hasattr(self.model, "freeze_backbone_only"):
            self.model.freeze_backbone_only()
        if hasattr(self.model, "freeze_projection_head"):
            self.model.freeze_projection_head()
        if hasattr(self.model, "freeze_classifier"):
            self.model.freeze_classifier()
        if hasattr(self.model, "freeze_energy_calibrator"):
            self.model.freeze_energy_calibrator()
        if hasattr(self.model, "freeze_geometry_calibrator"):
            self.model.freeze_geometry_calibrator()

        allowed = ("geometry_plastic_adapter",)
        bad = [name for name, p in self.model.named_parameters() if p.requires_grad and not any(a in name for a in allowed)]
        if bad:
            raise RuntimeError(f"Incremental PG-RGA allows only geometry_plastic_adapter parameters, got: {bad[:30]}")
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("Incremental PG-RGA selected but no geometry_plastic_adapter parameters are trainable.")
        return params

    def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
        return self._set_incremental_trainable_params(old_class_count)

    def _has_feature_plasticity(self) -> bool:
        return bool(self._adapter_mode_enabled())

    def _has_descriptor_plasticity(self) -> bool:
        return bool(getattr(self, "refine_new_descriptors", False))

    def _has_energy_calibration_plasticity(self) -> bool:
        return False

    def _has_adaptive_boundary_plasticity(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Phase/mode helpers
    # ------------------------------------------------------------------
    def _base_classifier_mode(self) -> str:
        return "geometry_only"

    def _inc_classifier_mode(self) -> str:
        return "geometry_only"

    def _eval_classifier_mode(self) -> str:
        return "geometry_only"

    def _set_model_phase_and_old_count(self, phase: int, old_class_count: int) -> None:
        phase = int(phase)
        old_class_count = int(old_class_count)
        if phase == 0 and hasattr(self.model, "set_base_mode"):
            try:
                self.model.set_base_mode(train_backbone=True, train_projection=True)
            except TypeError:
                self.model.set_base_mode()
        elif phase > 0 and hasattr(self.model, "set_incremental_mode"):
            try:
                self.model.set_incremental_mode(phase=phase, old_class_count=old_class_count, train_geometry_adapter=True)
            except TypeError:
                self.model.set_incremental_mode(phase=phase, old_class_count=old_class_count)
        else:
            self.model.current_phase = phase
            self.model.old_class_count = old_class_count
        if hasattr(self.model, "current_phase"):
            self.model.current_phase = phase
        if hasattr(self.model, "old_class_count"):
            self.model.old_class_count = old_class_count

    # ------------------------------------------------------------------
    # Validation/scoring helpers
    # ------------------------------------------------------------------
    def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
            return self._zero(logits)
        labels = labels.long().view(-1).to(logits.device)
        if labels.numel() != logits.size(0):
            raise RuntimeError(f"CE batch mismatch: logits={logits.size(0)}, labels={labels.numel()}")
        if labels.numel() == 0:
            return logits.sum() * 0.0
        lo = int(labels.min().detach().item())
        hi = int(labels.max().detach().item())
        if lo < 0 or hi >= logits.size(1):
            raise RuntimeError(f"CE label range [{lo},{hi}] incompatible with logits width={logits.size(1)}")
        clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
        smoothing = float(getattr(self, "label_smoothing", getattr(self.args, "label_smoothing", 0.0)))
        return F.cross_entropy(logits.clamp(-clip, clip), labels, label_smoothing=smoothing)

    def _classes_tensor(self, class_ids: Iterable[int], *, device=None) -> torch.Tensor:
        ids = [int(c) for c in class_ids]
        if not ids:
            raise RuntimeError("class_ids must be non-empty.")
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"class_ids contains duplicates: {ids}")
        if min(ids) < 0:
            raise RuntimeError(f"class_ids must be non-negative global IDs, got {ids}")
        return torch.as_tensor(ids, device=device if device is not None else self.device, dtype=torch.long)

    def _seen_classes_for_phase(self, phase: int, fallback_labels: Optional[torch.Tensor] = None) -> List[int]:
        if hasattr(self.dataset, "get_classes_up_to_phase"):
            try:
                seen = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase))]
                if seen:
                    return list(dict.fromkeys(seen))
            except Exception:
                pass
        seen: List[int] = []
        if hasattr(self.dataset, "phase_to_classes"):
            for p in range(int(phase) + 1):
                try:
                    seen.extend(int(c) for c in self.dataset.phase_to_classes[p])
                except Exception:
                    pass
        if not seen and torch.is_tensor(fallback_labels) and fallback_labels.numel() > 0:
            seen = [int(c) for c in fallback_labels.detach().cpu().unique(sorted=True).tolist()]
        if not seen:
            raise RuntimeError(f"Cannot resolve seen classes for phase={phase}")
        return list(dict.fromkeys(seen))

    def _assert_labels_in_seen_classes(self, labels: torch.Tensor, seen_classes: Iterable[int], *, context: str) -> None:
        if hasattr(self, "assert_global_labels_in_set"):
            return self.assert_global_labels_in_set(labels, seen_classes, context)
        y = labels.to(self.device).long().view(-1)
        seen = set(int(c) for c in seen_classes)
        bad = sorted(set(int(v) for v in y.detach().cpu().tolist()) - seen)
        if bad:
            raise RuntimeError(f"{context}: labels outside seen classes. bad={bad}, seen={sorted(seen)}")

    def _global_to_seen_local_safe(self, labels_global: torch.Tensor, seen_classes: Iterable[int], *, context: str) -> torch.Tensor:
        if hasattr(self, "global_to_seen_local"):
            return self.global_to_seen_local(labels_global, seen_classes, context=context)
        y = labels_global.to(self.device).long().view(-1)
        seen = [int(c) for c in seen_classes]
        mapping = {c: i for i, c in enumerate(seen)}
        local = torch.full_like(y, -1)
        for c, i in mapping.items():
            local[y == int(c)] = int(i)
        if bool((local < 0).any().item()):
            bad = sorted(set(int(v) for v in y[local < 0].detach().cpu().tolist()))
            raise RuntimeError(f"{context}: labels not in seen_classes: {bad}; seen={seen}")
        return local

    def _seen_local_to_global_safe(self, preds_local: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
        if hasattr(self, "seen_local_to_global"):
            return self.seen_local_to_global(preds_local, seen_classes, context="seen_local_to_global")
        seen = torch.as_tensor([int(c) for c in seen_classes], device=preds_local.device, dtype=torch.long)
        return seen.index_select(0, preds_local.long().view(-1))

    def _mask_logits_to_seen_classes(self, logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
        """Return seen-local logits.

        New classifier outputs [B, len(seen_classes)].  Legacy full-global logits
        are sliced explicitly.
        """
        if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError(f"logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")
        seen = [int(c) for c in seen_classes]
        if logits.size(1) == len(seen):
            return logits
        seen_t = torch.as_tensor(seen, device=logits.device, dtype=torch.long)
        if int(seen_t.max().item()) >= logits.size(1):
            raise RuntimeError(f"Cannot slice global logits width={logits.size(1)} by seen classes={seen}")
        return logits.index_select(1, seen_t)

    def _old_new_classes_for_validation(self, phase: int, old_class_count: int, seen_classes: Iterable[int]) -> Tuple[List[int], List[int]]:
        seen = [int(c) for c in seen_classes]
        old: List[int] = []
        if int(old_class_count) > 0 and int(phase) > 0 and hasattr(self.dataset, "get_classes_up_to_phase"):
            try:
                old = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase) - 1)]
            except Exception:
                old = []
        if not old and int(old_class_count) > 0:
            old = seen[: int(old_class_count)]
        old_set = set(old)
        new = [c for c in seen if c not in old_set]
        return [c for c in seen if c in old_set], new

    def _label_membership_mask(self, labels: torch.Tensor, class_ids: Iterable[int]) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in [int(x) for x in class_ids]:
            mask |= labels.eq(int(c))
        return mask

    def _prepare_batch_spectral_summary(self, batch_spectra: Optional[torch.Tensor], x: torch.Tensor) -> Tuple[Optional[torch.Tensor], bool]:
        if torch.is_tensor(batch_spectra) and batch_spectra.numel() > 0:
            s = batch_spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
            if s.dim() == 4:
                s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
            elif s.dim() == 3:
                s = s[:, :, s.size(-1) // 2] if s.size(0) == x.size(0) and s.size(2) > 1 else s.flatten(1)
            elif s.dim() == 1:
                s = s.view(x.size(0), -1)
            elif s.dim() > 4:
                s = s.flatten(1)
            if s.size(0) != x.size(0):
                raise RuntimeError(f"Batch spectral summary mismatch: {tuple(s.shape)} vs input {tuple(x.shape)}")
            return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), bool(getattr(self.args, "raw_spectral_summary_is_physical", True))
        return None, False

    def _forward_real_batch(
        self,
        x: torch.Tensor,
        batch_spectra: Optional[torch.Tensor] = None,
        *,
        classifier_mode: Optional[str] = None,
        seen_classes: Optional[Iterable[int]] = None,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        return_energy: bool = True,
        return_parts: bool = False,
    ) -> Dict[str, Any]:
        mode = self._normalize_classifier_mode(classifier_mode or self._eval_classifier_mode(), context="real_batch_forward")
        spectral_summary, is_physical = self._prepare_batch_spectral_summary(batch_spectra, x)
        kwargs = dict(
            classifier_mode=mode,
            mode=mode,
            seen_classes=list(seen_classes) if seen_classes is not None else None,
            old_classes=list(old_classes) if old_classes is not None else None,
            new_classes=list(new_classes) if new_classes is not None else None,
            return_energy=return_energy,
            return_parts=return_parts,
            spectral_summary=spectral_summary,
            spectral_summary_is_physical=bool(is_physical),
        )
        try:
            out = self.model(x, **kwargs)
        except TypeError:
            # Compatibility fallback for older model.forward; strict classifier/model should not hit this.
            fallback = {k: v for k, v in kwargs.items() if k in {"classifier_mode", "return_energy", "return_parts"}}
            out = self.model(x, **fallback)
        if not isinstance(out, dict):
            out = {"logits": out}
        out["spectral_summary"] = spectral_summary
        out["spectral_summary_is_physical"] = bool(is_physical)
        return out

    @torch.no_grad()
    def _validate_split_metrics(self, loader, old_class_count: int) -> Dict[str, Any]:
        self._assert_global_architecture_contract()
        self.model.eval()
        old_class_count = int(old_class_count)
        phase = int(getattr(self.model, "current_phase", 0))
        seen_classes = self._seen_classes_for_phase(phase)
        old_classes, new_classes = self._old_new_classes_for_validation(phase, old_class_count, seen_classes)
        old_pos = [seen_classes.index(c) for c in old_classes if c in seen_classes]
        new_pos = [seen_classes.index(c) for c in new_classes if c in seen_classes]

        total_loss = total_correct = total = batches = 0
        old_correct = old_total = new_correct = new_total = 0
        new_into_old_sum = old_into_new_sum = old_new_gap_sum = 0.0
        old_new_diag_batches = 0

        for batch in loader:
            x, y, batch_spectra, _ = self._unpack_hsi_batch(batch)
            x = x.to(self.device, non_blocking=True).float()
            y = y.to(self.device, non_blocking=True).long().view(-1)
            self._assert_labels_in_seen_classes(y, seen_classes, context=f"phase_{phase}_validation")
            y_local = self._global_to_seen_local_safe(y, seen_classes, context=f"phase_{phase}_validation")
            out = self._forward_real_batch(
                x,
                batch_spectra,
                classifier_mode="geometry_only",
                seen_classes=seen_classes,
                old_classes=old_classes,
                new_classes=new_classes,
                return_energy=True,
                return_parts=False,
            )
            logits_seen = self._mask_logits_to_seen_classes(out["logits"], seen_classes)
            if logits_seen.size(0) != y_local.numel():
                raise RuntimeError(f"Validation logits/labels mismatch: {logits_seen.size(0)} vs {y_local.numel()}")
            loss = self._stable_ce(logits_seen, y_local)
            pred_local = logits_seen.argmax(dim=1)
            pred_global = self._seen_local_to_global_safe(pred_local, seen_classes)
            correct = pred_global.eq(y)

            total_loss += float(loss.detach().item())
            total_correct += int(correct.sum().item())
            total += int(y.numel())
            batches += 1

            if old_class_count > 0 and old_classes and new_classes:
                old_mask = self._label_membership_mask(y, old_classes)
                new_mask = self._label_membership_mask(y, new_classes)
                if bool(old_mask.any().item()):
                    old_correct += int(correct[old_mask].sum().item())
                    old_total += int(old_mask.sum().item())
                if bool(new_mask.any().item()):
                    new_correct += int(correct[new_mask].sum().item())
                    new_total += int(new_mask.sum().item())

            energy = out.get("energy", None) if isinstance(out, dict) else None
            if torch.is_tensor(energy) and energy.dim() == 2 and old_pos and new_pos:
                e_seen = self._mask_logits_to_seen_classes(-energy, seen_classes) if energy.size(1) != len(seen_classes) else energy
                # e_seen is energy if already seen-local.  If legacy conversion was used above, negate back.
                if energy.size(1) != len(seen_classes):
                    e_seen = -e_seen
                old_idx = torch.as_tensor(old_pos, device=e_seen.device, dtype=torch.long)
                new_idx = torch.as_tensor(new_pos, device=e_seen.device, dtype=torch.long)
                old_min = e_seen.index_select(1, old_idx).min(dim=1).values
                new_min = e_seen.index_select(1, new_idx).min(dim=1).values
                old_mask_e = self._label_membership_mask(y, old_classes)
                new_mask_e = self._label_membership_mask(y, new_classes)
                if bool(new_mask_e.any().item()):
                    new_into_old_sum += float((old_min[new_mask_e] < new_min[new_mask_e]).float().mean().detach().item())
                if bool(old_mask_e.any().item()):
                    old_into_new_sum += float((new_min[old_mask_e] < old_min[old_mask_e]).float().mean().detach().item())
                old_new_gap_sum += float((new_min - old_min).mean().detach().item())
                old_new_diag_batches += 1

        acc = 100.0 * total_correct / max(total, 1)
        split = old_class_count > 0 and bool(old_classes) and bool(new_classes)
        old_acc = 100.0 * old_correct / max(old_total, 1) if split else 0.0
        new_acc = 100.0 * new_correct / max(new_total, 1) if split else acc
        hm = 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-8) if split else acc
        return {
            "loss": total_loss / max(batches, 1),
            "acc": acc,
            "old_acc": old_acc,
            "new_acc": new_acc,
            "hm": hm,
            "old_new_split_available": bool(split),
            "predicted_unseen": 0.0,
            "new_into_old_rate": float(new_into_old_sum / max(old_new_diag_batches, 1)),
            "old_into_new_rate": float(old_into_new_sum / max(old_new_diag_batches, 1)),
            "mean_old_new_energy_gap": float(old_new_gap_sum / max(old_new_diag_batches, 1)),
            "seen_classes": seen_classes,
            "old_classes": old_classes,
            "new_classes": new_classes,
        }

    # ------------------------------------------------------------------
    # Scoring/checkpoint helpers
    # ------------------------------------------------------------------
    def _base_geometry_global_metrics(self) -> Dict[str, float]:
        gb = getattr(self.model, "geometry_bank", None)
        if gb is None:
            return {}
        diag: Dict[str, Any] = {}
        try:
            if hasattr(gb, "compute_geometry_diagnostics"):
                diag = gb.compute_geometry_diagnostics()
            elif hasattr(gb, "geometry_diagnostics"):
                diag = gb.geometry_diagnostics()
        except Exception:
            return {}
        out: Dict[str, float] = {}
        for k, v in diag.items() if isinstance(diag, dict) else []:
            if torch.is_tensor(v) and v.numel() == 1:
                out[str(k)] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out

    def _select_score(self, val_stats: Dict[str, float], phase: int) -> float:
        if int(phase) > 0:
            metric = str(getattr(self, "best_state_metric", getattr(self.args, "best_state_metric", "hm"))).lower().strip()
            if metric in {"acc", "oa"}:
                return float(val_stats.get("acc", 0.0))
            if metric in {"old_new_min", "min"}:
                return min(float(val_stats.get("old_acc", 0.0)), float(val_stats.get("new_acc", 0.0)))
            if metric in {"loss", "val_loss"}:
                return -float(val_stats.get("loss", 0.0))
            return float(val_stats.get("hm", 0.0))
        return self._select_base_checkpoint_score(val_stats, getattr(self, "_last_base_geom_stats", None))

    def _select_base_checkpoint_score(self, val_stats: Dict[str, float], geom_stats: Optional[Dict[str, float]] = None) -> float:
        metric = str(getattr(self, "best_state_metric", getattr(self.args, "best_state_metric", "geometry_score"))).lower().strip()
        acc = float(val_stats.get("acc", val_stats.get("oa", 0.0)))
        if metric in {"acc", "oa", "hm", "h", "harmonic"}:
            return acc
        if metric in {"loss", "val_loss"}:
            return -float(val_stats.get("loss", 0.0))
        geom = geom_stats if isinstance(geom_stats, dict) else {}
        reserve = float(geom.get("geometry_reserve_score", geom.get("reserve_score", 0.0)))
        overlap = float(geom.get("feature_subspace_overlap", geom.get("subspace_overlap_mean", 0.0)))
        conflict = float(geom.get("geometry_conflict_max", 0.0))
        return acc + reserve - 10.0 * overlap - conflict

    def _capture_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _print_trainable_summary(self, phase: int) -> None:
        trainable = [(n, int(p.numel())) for n, p in self.model.named_parameters() if p.requires_grad]
        total = sum(n for _, n in trainable)
        if int(phase) == 0:
            print(f"[Trainable] Base phase: {total:,} params | backbone/projection + temporary CE head")
            print("[Base Objective] Balanced CE + GICS + PGR(compact,center,subspace,band,volume)")
        else:
            print(f"[Trainable] Incremental phase {phase}: {total:,} params | geometry_plastic_adapter only")
            print("[Incremental Objective] GeometryBank replay + adapted new samples + joint CE + old/new energy margins")
        if self.debug:
            for name, count in trainable[:150]:
                print(f"  {name}: {count:,}")

    def _current_runtime_contract(self) -> Dict[str, object]:
        adapter_trainable = any(p.requires_grad and "geometry_plastic_adapter" in n for n, p in self.model.named_parameters())
        return {
            "method": "PG-RGA-HSI",
            "feature_space": "canonical projected z; incremental scoring uses bounded geometry residual adapter",
            "classifier": "strict seen-class low-rank GeometryBank energy",
            "classifier_output": "[B, len(seen_classes)]",
            "incremental_update_mode": self._incremental_update_mode(),
            "geometry_gated_adapter_present": bool(hasattr(self.model, "geometry_plastic_adapter")),
            "geometry_gated_adapter_trainable": bool(adapter_trainable),
            "old_memory": "frozen GeometryBank statistics only",
            "raw_exemplars": False,
            "kd_teacher": False,
            "transport": False,
            "adaptive_boundary": False,
            "energy_calibrator": False,
            "spectral_classifier_branch": False,
            "base_objective": "Balanced CE + GICS + PGR",
            "incremental_objective": "GeometryBank replay + PG-RGA adapter + joint CE + old/new energy margin",
            "uses_logdet_energy": bool(self.use_logdet_energy),
            "logdet_energy_weight": float(self.logdet_energy_weight),
            "row_energy_standardization": False,
        }

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------
    def _assert_incremental_preflight(self, phase: int, old_class_count: int) -> None:
        phase = int(phase)
        old_class_count = int(old_class_count)
        self._assert_updated_stack_contract(phase=phase)
        if old_class_count <= 0:
            raise RuntimeError(f"Incremental phase {phase} requires old_class_count > 0.")
        if not hasattr(self.model, "geometry_plastic_adapter"):
            raise RuntimeError("Incremental PG-RGA requires model.geometry_plastic_adapter.")
        bank = self._safe_get_subspace_bank(require_ready=True)
        old_ids = list(range(old_class_count))
        if hasattr(self, "assert_bank_ready_for_seen_classes"):
            self.assert_bank_ready_for_seen_classes(bank, old_ids)
        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None:
            if hasattr(gb, "freeze_classes"):
                gb.freeze_classes(old_ids)
            elif hasattr(gb, "freeze_classes_up_to"):
                gb.freeze_classes_up_to(old_class_count)
            if hasattr(gb, "assert_bank_valid"):
                gb.assert_bank_valid(seen_classes=old_ids, strict=True)

    def train_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4):
        phase = int(phase)
        self.early_stop_patience = 0
        for k in ("early_stop_patience", "base_early_stop_patience", "incremental_early_stop_patience"):
            try:
                setattr(self.args, k, 0)
            except Exception:
                pass

        self._force_clean_main_path_args()
        self._propagate_clean_energy_config_to_model()
        self._assert_global_architecture_contract()

        if phase == 0:
            self._set_model_phase_and_old_count(0, 0)
            self._set_base_trainable_params()
            self._assert_updated_stack_contract(phase=0)
            self._print_trainable_summary(phase=0)
            return self.train_base_phase(phase=0, epochs=epochs, batch_size=batch_size, lr=lr)

        if self.base_only or self.disable_incremental_training:
            raise RuntimeError("Incremental training is disabled. Set --base_only false and --disable_incremental_training false.")

        old_class_count = len(self.dataset.get_classes_up_to_phase(phase - 1))
        self._set_model_phase_and_old_count(phase, old_class_count)
        self._set_incremental_trainable_params(old_class_count)
        self._assert_incremental_preflight(phase, old_class_count)
        self._print_trainable_summary(phase=phase)
        return self.train_incremental_phase(phase=phase, epochs=epochs, batch_size=batch_size, lr=lr)

    # ------------------------------------------------------------------
    # Hard-disabled compatibility fallbacks for stale main.py checks
    # ------------------------------------------------------------------
    def _old_new_boundary_preservation_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ref = next((v for v in list(args) + list(kwargs.values()) if torch.is_tensor(v)), None)
        z = self._zero(ref)
        return {"total": z, "overlap": z.detach(), "center": z.detach(), "volume": z.detach(), "band": z.detach()}

    def _project_new_descriptor_params_out_of_old_tangent_space(self, *args, **kwargs):
        return args[0] if len(args) == 1 else (args if args else kwargs.get("new_descriptor", None))

    def _adaptive_boundary_loss_from_current_bank(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ref = next((v for v in list(args) + list(kwargs.values()) if torch.is_tensor(v)), None)
        z = self._zero(ref)
        return {"total": z, "boundary": z.detach(), "old_new": z.detach(), "radius_reg": z.detach()}

    def _adaptive_boundary_enabled(self) -> bool:
        return False

    def _adaptive_boundary_state(self, old_class_count: int = 0) -> Dict[str, float]:
        return {"adaptive_boundary_enabled": 0.0, "boundary_radius_mean": 0.0, "old_boundary_radius_mean": 0.0, "new_boundary_radius_mean": 0.0}

    def _adaptive_boundary_trainable_params(self) -> List[torch.nn.Parameter]:
        return []

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, phase, history, evaluator_metrics: Optional[Dict] = None) -> None:
        phase = int(phase)
        phase_dir = os.path.join(self.save_dir, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)
        ckpt = {
            "phase": phase,
            "base_only": self._as_bool(self.base_only),
            "incremental_enabled": not self._as_bool(self.disable_incremental_training),
            "base_objective": "Balanced CE + GICS + PGR",
            "incremental_objective": "GeometryBank replay + PG-RGA adapter + joint CE + old/new geometry energy margins",
            "runtime_contract": self._current_runtime_contract(),
            "architecture_contract": {
                "method": "PG-RGA-HSI",
                "classifier_mode": "geometry_only",
                "bank": "mean,basis,eigvals,residual_variance,reliability,sample_count,band_signature,spectral_shape",
                "old_memory": "frozen GeometryBank statistics only",
                "replay": "synthetic GeometryBank replay only; no raw old patches/features",
                "incremental_update_mode": self._incremental_update_mode(),
                "adapter": "geometry_plastic_adapter_only",
                "forbidden_components": ["KD", "teacher", "prototypes", "transport", "adaptive_boundary", "measured_calibration", "BSS/GDR"],
            },
            "model_state_dict": self.model.state_dict(),
            "memory_snapshot": self.model.export_memory_snapshot() if hasattr(self.model, "export_memory_snapshot") else None,
            "current_num_classes": int(getattr(self.model, "current_num_classes", 0)),
            "old_class_count": int(getattr(self.model, "old_class_count", 0)),
            "history": history,
            "base_geometry_certificate": getattr(self, "_last_base_geometry_certificate", getattr(self.model, "base_geometry_certificate", None)),
            "args": vars(self.args) if hasattr(self.args, "__dict__") else {},
        }
        diag = getattr(self, f"_last_phase_{phase}_geometry_diagnostics", None)
        if diag is None:
            diag = getattr(self, "_last_base_geometry_diagnostics", None)
        if diag is not None:
            ckpt["geometry_diagnostics"] = diag
        if evaluator_metrics is not None:
            ckpt["evaluator_metrics"] = evaluator_metrics
        path = os.path.join(phase_dir, "checkpoint.pth")
        torch.save(ckpt, path)
        print(f"[Saved] {path}")











# from __future__ import annotations

# import os
# from typing import Any, Dict, Iterable, List, Optional, Tuple
# import torch
# import torch.nn.functional as F
# from trainers.trainer_helpers import TrainerHelper
# from trainers.base_phase_trainer import BasePhaseTrainer
# from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
# try:
#     from losses.loss import unified_spectral_geometry_loss, sample_boundary_geometry_features
# except Exception:  # pragma: no cover - strict contract reports this later
#     unified_spectral_geometry_loss = None
#     sample_boundary_geometry_features = None



# class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):

#     """Unified SGLAT-HSI trainer: SRPGR base construction + boundary-admitted geometry insertion.

#     Public training contract:
#         base phase        -> unified_spectral_geometry_loss(phase="base") via BasePhaseTrainer
#         incremental phase -> unified_spectral_geometry_loss(phase="incremental") via IncrementalPhaseTrainer

#     """



#     @staticmethod

#     def _as_bool(value, default: bool = False) -> bool:

#         """Parse argparse/config booleans safely.



#         Python's bool("false") is True, which can silently enable forbidden

#         incremental paths. This parser treats common string values explicitly.

#         """

#         if value is None:

#             return bool(default)

#         if isinstance(value, bool):

#             return value

#         if isinstance(value, (int, float)):

#             return bool(value)

#         if isinstance(value, str):

#             v = value.strip().lower()

#             if v in {"1", "true", "yes", "y", "on"}:

#                 return True

#             if v in {"0", "false", "no", "n", "off", "none", "null", ""}:

#                 return False

#         return bool(value)



#     def _arg_bool(self, name: str, default: bool = False) -> bool:

#         return self._as_bool(getattr(self.args, name, default), default=default)



#     def _set_arg_default(self, name: str, value: Any) -> None:

#         """Set a runtime default without overwriting explicit CLI/config values."""

#         if not hasattr(self.args, name) or getattr(self.args, name) is None:

#             try:

#                 setattr(self.args, name, value)

#             except Exception:

#                 pass



#     def _install_clean_runtime_defaults(self) -> None:

#         """Centralize the defaults required by the fixed bank/classifier/loss stack.



#         These are not new architectural branches. They ensure the orchestrator

#         calls the same low-rank Gaussian geometry path everywhere: validation,

#         replay, descriptor refinement, and checkpoint metadata.

#         """

#         defaults = {

#             # Fixed classifier/loss energy. Do not row-standardize geometry logits.

#             "use_logdet_energy": True,

#             "logdet_energy_weight": 0.05,

#             "logdet_normalize_by_dim": True,

#             "center_logdet_energy": True,

#             "geometry_normalize_logits": False,

#             # SRGP classifier/spectral-shape defaults. Spectral residual energy is used only

#             # when the batch carries physical wavelength-ordered spectra; PCA channels stay gated off.

#             "use_spectral_geometry": True,

#             "spectral_energy_weight": 0.05,

#             "spectral_derivative_weight": 0.50,

#             "spectral_second_derivative_weight": 0.25,

#             "spectral_require_physical_summary": True,

#             "spectral_summary_is_physical": False,

#             "raw_spectral_summary_is_physical": True,

#             "base_classifier_mode": "srgp",

#             "incremental_classifier_mode": "geometry_only",

#             "eval_classifier_mode": "geometry_only",

#             "base_spectral_shape_weight": 0.05,

#             "base_spectral_shape_overlap_max": 0.75,

#             "base_max_spectral_shape_similarity": 0.75,

#             "base_spectral_shape_require_physical": True,

#             "spectral_require_physical_summary": True,

#             "external_spectra_are_physical": True,

#             "spectral_shape_weight": 0.25,

#             "risk_spectral_shape_weight": 0.25,

#             "old_new_risk_spectral_shape_weight": 0.25,

#             # Base-to-incremental certificate/gate. Warning by default; hard fail when enabled.

#             "enforce_base_geometry_certificate": False,

#             "base_cert_min_geom_acc": 95.0,

#             "base_cert_min_reliability": 0.15,

#             "base_cert_min_mean_reliability": 0.35,

#             "base_cert_max_subspace_overlap": 0.55,

#             "base_cert_max_geometry_conflict": 1.35,

#             "base_cert_max_band_similarity": 0.90,

#             # SGLAT-HSI boundary replay from frozen old low-rank geometry.
#             # This is the main old-class protection mechanism. Generic replay
#             # can remain as fallback, but it must not be the method identity.
#             "use_boundary_geometry_replay": True,
#             "boundary_replay_risk_threshold": 0.35,
#             "boundary_replay_overlap_threshold": 0.30,
#             "boundary_replay_samples_per_pair": 12,
#             "boundary_replay_max_pairs": 24,
#             "boundary_replay_parallel_scale": 0.15,
#             "boundary_replay_residual_scale": 0.05,
#             "scbgr_commit_only_if_safe": True,
#             "use_sglat_transport": bool(self._arg_bool("use_sglat_transport", False)),
#             "use_geometry_transport": bool(self._arg_bool("use_geometry_transport", False)),
#             "allow_old_model_transport": bool(self._arg_bool("allow_old_model_transport", False)),
#             "allow_transport_without_adapter": bool(self._arg_bool("allow_transport_without_adapter", False)),
#             "transport_type": str(getattr(self.args, "transport_type", "ridge")),
#             "transport_ridge": float(getattr(self.args, "transport_ridge", 1e-3)),
#             "transport_ema": float(getattr(self.args, "transport_ema", 0.97)),
#             "transport_batches": int(getattr(self.args, "transport_batches", 20)),
#             "transport_identity_blend": float(getattr(self.args, "transport_identity_blend", 0.75)),
#             "transport_low_rank": int(getattr(self.args, "transport_low_rank", 4)),
#             "transport_min_reliability_gate": float(getattr(self.args, "transport_min_reliability_gate", 0.30)),
#             "transport_max_a_minus_i_fro": float(getattr(self.args, "transport_max_a_minus_i_fro", 1.5)),
#             "transport_max_b_norm": float(getattr(self.args, "transport_max_b_norm", 0.75)),
#             "transport_after_adapter_epoch": int(getattr(self.args, "transport_after_adapter_epoch", 3)),
#             "admission_margin": 0.25,
#             "max_new_admission_violation": 0.25,
#             "max_old_boundary_violation": 0.25,

#             # Generic/risk-weighted old replay from GeometryBank is only a fallback.

#             "use_risk_weighted_replay": True,

#             "risk_center_margin": 1.0,

#             "risk_subspace_weight": 1.0,

#             "risk_band_weight": 0.25,

#             "risk_reliability_weighted": True,

#             "risk_replay_reliability_weighted": True,

#             "risk_replay_reliability_gated": True,

#             "gfa_reliability_gated": True,

#             "risk_replay_min_samples": 4,

#             "risk_replay_max_multiplier": 3.0,

#             # Reliability-gated and risk-aware new-row admission/correction.

#             # Do not mutate new rows with heuristic pre-correction by default.
#             # New geometry safety is enforced by the unified incremental loss
#             # and boundary-admission diagnostics.
#             "reliability_gated_admission": True,

#             "risk_aware_descriptor_correction": True,

#             "descriptor_correction_risk_threshold": 0.35,

#             "descriptor_correction_overlap_threshold": 0.30,

#             # Canonical names consumed by IncrementalPhaseTrainer.

#             "descriptor_correction_basis_strength": 0.85,

#             "descriptor_correction_mean_push": 0.20,

#             "descriptor_correction_var_shrink": 0.15,

#             "descriptor_correction_topk_old": 3,

#             "risk_sep_weight": 0.30,

#             "risk_sep_overlap_target": 0.20,

#             "risk_sep_active_threshold": 0.25,

#             "descriptor_overlap_target": 0.20,

#             # Backward-compatible aliases from earlier drafts.  Keep them

#             # synchronized in _force_clean_main_path_args().

#             "descriptor_correction_center_step": 0.20,

#             "descriptor_correction_subspace_eta": 0.85,

#             "descriptor_correction_variance_shrink": 0.15,

#             "descriptor_risk_sep_weight": 0.20,

#             "descriptor_risk_sep_target": 0.35,

#             "admission_min_gate": 0.35,

#             "admission_shrink_floor": 0.15,

#             "admission_low_rank_cap": 2,

#             # Descriptor-only old/new collision controls.

#             "descriptor_subspace_collision_weight": 0.20,

#             "descriptor_subspace_overlap_max": 0.35,

#             "descriptor_center_margin_weight": 0.10,

#             "descriptor_center_collision_weight": 0.10,

#             "descriptor_center_margin": 0.50,

#             "descriptor_volume_weight": 0.05,

#             "descriptor_volume_control_weight": 0.05,

#             "descriptor_volume_margin": 0.0,

#             "descriptor_refine_steps_per_epoch": 50,

#             "descriptor_refine_grad_clip": 1.0,

#             "descriptor_refine_max_mean_shift": 0.35,

#             "descriptor_refine_max_logvar_shift": 0.75,

#             # Architecture-level incremental update mode.

#             "incremental_update_mode": "scbgr",

#             "adapter_bottleneck": 32,

#             "adapter_max_scale": 0.35,

#             "adapter_dropout": 0.0,

#             "adapter_gate_bias_init": -3.0,

#             "adapter_lr": 5e-4,

#             "adapter_weight_decay": 0.0,

#             "g2rpa_adapter_weight": 1.0,

#             "adapter_old_delta_weight": 1.0,

#             "adapter_old_gate_weight": 0.75,

#             "adapter_old_energy_weight": 0.25,

#             "adapter_old_margin_weight": 0.25,

#             "adapter_delta_weight": 0.10,

#             "adapter_new_gate_weight": 0.05,

#             "adapter_new_gate_target": 0.25,

#             "adapter_new_gate_max_target": 0.75,

#             # Strict stack guards. Disable only for legacy ablation/debug.

#             "strict_updated_stack": True,

#         }

#         for k, v in defaults.items():

#             self._set_arg_default(k, v)



#     @staticmethod

#     def _freeze_module_if_present(module) -> None:

#         if module is None:

#             return

#         try:

#             for p in module.parameters():

#                 p.requires_grad = False

#         except Exception:

#             pass



#     def _incremental_update_mode(self) -> str:
#         """Normalize the selected incremental update policy.

#         ``scbgr`` is the clean main method: transport-guided candidate admission plus boundary geometry replay and the
#         unified incremental geometry-admission loss. Legacy names such as
#         ``descriptor_only`` and ``rsgi`` are accepted only as aliases so old
#         commands do not silently select the weaker replay-only path.

#         ``geometry_gated_adapter`` remains an explicit ablation.
#         """
#         raw = str(getattr(self.args, "incremental_update_mode", "scbgr")).lower().strip()

#         aliases = {
#             "": "scbgr",
#             "none": "scbgr",
#             "clean": "scbgr",
#             "rsgi": "scbgr",
#             "descriptor": "scbgr",
#             "descriptor_only": "scbgr",
#             "geometry_state_admission": "scbgr",
#             "spectral_risk_boundary": "scbgr",
#             "spectral_boundary": "scbgr",
#             "boundary_geometry": "scbgr",
#             "scbgr": "scbgr",
#             "scb-gr": "scbgr",
#             "bage": "scbgr",
#             "g2rpa": "geometry_gated_adapter",
#             "g²rpa": "geometry_gated_adapter",
#             "adapter": "geometry_gated_adapter",
#             "gated_adapter": "geometry_gated_adapter",
#             "geometry_adapter": "geometry_gated_adapter",
#             "geometry_gated_adapter": "geometry_gated_adapter",
#         }

#         mode = aliases.get(raw, raw)

#         if mode not in {"scbgr", "geometry_gated_adapter"}:

#             raise RuntimeError(

#                 f"Unsupported incremental_update_mode={raw!r}. "

#                 "Allowed: scbgr, geometry_gated_adapter."

#             )

#         try:

#             setattr(self.args, "incremental_update_mode", mode)

#         except Exception:

#             pass

#         if hasattr(self.model, "incremental_update_mode"):

#             try:

#                 self.model.incremental_update_mode = mode

#             except Exception:

#                 pass

#         return mode


#     def _adapter_mode_enabled(self) -> bool:

#         return self._incremental_update_mode() == "geometry_gated_adapter"



#     def _classifier_supports_srgp(self) -> bool:

#         """Return True when the classifier exposes the SRGP spectral-residual API.



#         Geometry-only classifiers can still be used in explicit ablations, but the

#         main SRGP path should not silently request an unsupported classifier mode.

#         """

#         clf = getattr(self.model, "classifier", None)

#         if clf is None:

#             return False

#         return any(

#             hasattr(clf, name)

#             for name in (

#                 "spectral_residual_energy",

#                 "geometry_logits_from_bank",

#                 "geometry_energy_from_bank",

#             )

#         )



#     def _normalize_classifier_mode(self, mode: Optional[str], *, context: str = "runtime") -> str:

#         """Normalize classifier-mode aliases and fail cleanly on stale stacks."""

#         raw = str(mode or "srgp").lower().strip()

#         aliases = {

#             "spectral_residual": "srgp",

#             "spectral_residual_geometry": "srgp",

#             "srgp_geometry": "srgp",

#             "srpg": "srgp",

#             "geo": "geometry_only",

#             "geometry": "geometry_only",

#         }

#         mode = aliases.get(raw, raw)

#         allowed = {"srgp", "geometry_only"}

#         if mode not in allowed:

#             raise RuntimeError(f"{context}: unsupported classifier mode {raw!r}. Allowed: {sorted(allowed)}")

#         if mode == "srgp" and not self._classifier_supports_srgp():

#             msg = (

#                 f"{context}: requested SRGP classifier mode, but the loaded classifier does not expose "

#                 "the spectral-residual geometry API. Replace models/classifier.py with the SRGP version, "

#                 "or set strict_updated_stack=false only for a geometry_only ablation."

#             )

#             if self._arg_bool("strict_updated_stack", True):

#                 raise RuntimeError(msg)

#             print(f"[WARN] {msg} Falling back to geometry_only.")

#             return "geometry_only"

#         return mode



#     def _spectral_batch_is_physical(self, batch_spectra: Optional[torch.Tensor]) -> bool:

#         if torch.is_tensor(batch_spectra) and batch_spectra.numel() > 0:

#             return self._arg_bool("raw_spectral_summary_is_physical", True)

#         return self._arg_bool("spectral_summary_is_physical", False)



#     def _prepare_batch_spectral_summary(

#         self,

#         batch_spectra: Optional[torch.Tensor],

#         x: torch.Tensor,

#     ) -> Tuple[Optional[torch.Tensor], bool]:

#         """Prepare raw physical spectral summaries for real HSI batches.



#         The trainer never treats PCA channels as physical wavelength-ordered spectra.

#         If the dataloader does not return raw spectra, the spectral branch is gated off.

#         """

#         if torch.is_tensor(batch_spectra) and batch_spectra.numel() > 0:

#             s = batch_spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)

#             # Raw HSI metadata may arrive as [B,S], [B,S,H,W], or other

#             # dataset-specific shapes.  For patch cubes, use the center pixel;

#             # flatten only metadata tensors that are not spatial cubes.

#             if s.dim() == 4:

#                 s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]

#             elif s.dim() == 3:

#                 # [B,S,L] metadata: use center spectrum, not full patch flatten.
#                 if s.size(0) == x.size(0) and s.size(1) > 0 and s.size(2) > 1:
#                     s = s[:, :, s.size(-1) // 2]
#                 else:
#                     s = s.flatten(1)

#             elif s.dim() == 1:

#                 if s.numel() % max(int(x.size(0)), 1) != 0:

#                     raise RuntimeError(

#                         f"1-D spectral metadata cannot be reshaped to batch size {x.size(0)}: {tuple(s.shape)}"

#                     )

#                 s = s.view(x.size(0), -1)

#             elif s.dim() > 4:

#                 s = s.flatten(1)

#             if s.size(0) != x.size(0):

#                 raise RuntimeError(

#                     f"Batch spectral summary size mismatch: spectra={tuple(s.shape)}, input={tuple(x.shape)}"

#                 )

#             s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

#             return s, self._spectral_batch_is_physical(s)

#         return None, False



#     def _forward_real_batch(

#         self,

#         x: torch.Tensor,

#         batch_spectra: Optional[torch.Tensor] = None,

#         *,

#         classifier_mode: Optional[str] = None,

#         return_energy: bool = True,

#         return_parts: bool = False,

#     ) -> Dict[str, torch.Tensor]:

#         """Forward real HSI samples through SRGP with safe spectral gating.



#         Synthetic replay must not use this helper because replay features do not

#         have physical spectra. It is intentionally limited to real image batches.

#         """

#         mode = self._normalize_classifier_mode(classifier_mode or self._eval_classifier_mode(), context="real_batch_forward")

#         spectral_summary, is_physical = self._prepare_batch_spectral_summary(batch_spectra, x)

#         try:

#             out = self.model(

#                 x,

#                 classifier_mode=mode,

#                 return_energy=return_energy,

#                 return_parts=return_parts,

#                 spectral_summary=spectral_summary,

#                 spectral_summary_is_physical=bool(is_physical),

#             )

#         except TypeError:

#             # Compatibility path for older model.forward signatures. If strict is

#             # enabled and spectral summaries are present, failing loudly is safer

#             # than silently discarding the HSI-specific branch.

#             if spectral_summary is not None and bool(is_physical) and self._arg_bool("strict_updated_stack", True):

#                 raise

#             try:

#                 out = self.model(x, classifier_mode=mode, return_energy=return_energy, return_parts=return_parts)

#             except TypeError:

#                 out = self.model(x)

#         if not isinstance(out, dict):

#             out = {"logits": out}

#         out["spectral_summary"] = spectral_summary

#         out["spectral_summary_is_physical"] = bool(is_physical)

#         return out



#     def __init__(self, model, dataset, args) -> None:

#         self.args = args

#         self.device = torch.device(args.device)

#         self.model = model.to(self.device)

#         self.dataset = dataset

#         self.save_dir = str(getattr(args, "save_dir", "./checkpoints"))



#         self.debug = self._arg_bool("debug_verbose", False) or os.environ.get("NECIL_DEBUG", "0") == "1"

#         self.base_only = self._arg_bool("base_only", False)

#         self.disable_incremental_training = self._arg_bool("disable_incremental_training", False)

#         self._install_clean_runtime_defaults()

#         self.incremental_update_mode = self._incremental_update_mode()



#         # Geometry/classifier contract. SRGP uses feature low-rank geometry plus optional

#         # physical spectral-shape residual energy for real HSI samples.

#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))

#         self.geom_var_floor = float(getattr(args, "geom_var_floor", 1e-4))

#         self.reliability_energy_weight = float(getattr(args, "reliability_energy_weight", 0.05))

#         self.energy_normalize_by_dim = self._arg_bool("energy_normalize_by_dim", True)

#         self.residual_variance_scale = float(getattr(args, "residual_variance_scale", 1.0))

#         self.invalid_class_energy = float(getattr(args, "invalid_class_energy", 1e6))

#         self.use_logdet_energy = self._arg_bool("use_logdet_energy", True)

#         self.logdet_energy_weight = float(getattr(args, "logdet_energy_weight", 0.05))

#         self.logdet_normalize_by_dim = self._arg_bool("logdet_normalize_by_dim", True)

#         self.center_logdet_energy = self._arg_bool("center_logdet_energy", True)



#         # Base objective is exposed through unified_spectral_geometry_loss(phase="base").

#         # CE/GICS/PGR/spectral-shape are internal parts of that single public objective.

#         self.base_ce_weight = float(getattr(args, "base_ce_weight", 1.0))

#         self.base_gics_weight = float(getattr(args, "base_gics_weight", 0.20))

#         self.use_prospective_geometry_reserve = self._arg_bool("use_prospective_geometry_reserve", True)

#         self.pgr_weight = float(getattr(args, "pgr_weight", 0.10))

#         self.pgr_compact_weight = float(getattr(args, "pgr_compact_weight", 0.15))

#         self.pgr_center_weight = float(getattr(args, "pgr_center_weight", 0.20))

#         self.pgr_subspace_weight = float(getattr(args, "pgr_subspace_weight", 0.10))

#         self.pgr_band_weight = float(getattr(args, "pgr_band_weight", 0.05))

#         self.pgr_volume_weight = float(getattr(args, "pgr_volume_weight", 0.05))

#         self.pgr_center_margin = float(getattr(args, "pgr_center_margin", 1.05))

#         self.pgr_min_class_samples = int(getattr(args, "pgr_min_class_samples", 3))

#         self.pgr_subspace_min_samples = int(getattr(args, "pgr_subspace_min_samples", 6))

#         self.pgr_subspace_rank = int(getattr(args, "pgr_subspace_rank", 3))

#         self.pgr_max_class_variance = float(getattr(args, "pgr_max_class_variance", 0.75))

#         self.pgr_band_overlap_max = float(getattr(args, "pgr_band_overlap_max", 0.75))

#         self.pgr_normalize_features = self._arg_bool("pgr_normalize_features", True)



#         # GeometryBank extraction/rank settings.

#         self.rank_energy_threshold = float(getattr(args, "rank_energy_threshold", 0.95))

#         self.rank_eigen_ratio_threshold = float(getattr(args, "rank_eigen_ratio_threshold", 1e-3))

#         self.min_active_rank = int(getattr(args, "min_active_rank", 1))

#         self.geometry_variance_shrinkage = float(getattr(args, "geometry_variance_shrinkage", 0.10))



#         # Optimization/checkpoint policy.

#         self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))

#         self.ce_logit_clip = float(getattr(args, "ce_logit_clip", 50.0))

#         self.grad_clip_base = float(getattr(args, "grad_clip_base", 1.0))

#         self.grad_clip_inc = float(getattr(args, "grad_clip_inc", 0.5))

#         self.refresh_before_validation = self._arg_bool("refresh_before_validation", True)

#         self.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))

#         self.bank_refresh_every = int(getattr(args, "bank_refresh_every", 0))

#         self.best_state_metric = str(getattr(args, "best_state_metric", "geometry_score")).lower().strip() or "geometry_score"

#         self.early_stop_metric = str(getattr(args, "early_stop_metric", self.best_state_metric)).lower().strip()

#         self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))



#         # Incremental objective: SGLAT-HSI = boundary geometry replay + unified geometry-admission loss.

#         # Generic old replay and score calibration are fallback/ablation only.

#         self.gfa_weight = float(getattr(args, "gfa_weight", 1.0))

#         self.gfa_samples_per_class = int(getattr(args, "gfa_samples_per_class", getattr(args, "component_replay_per_class", 64)))

#         self.gfa_parallel_scale = float(getattr(args, "gfa_parallel_scale", 1.0))

#         self.gfa_residual_scale = float(getattr(args, "gfa_residual_scale", 0.30))

#         self.use_boundary_geometry_replay = self._arg_bool("use_boundary_geometry_replay", True)

#         self.boundary_replay_risk_threshold = float(getattr(args, "boundary_replay_risk_threshold", 0.35))

#         self.boundary_replay_overlap_threshold = float(getattr(args, "boundary_replay_overlap_threshold", 0.30))

#         self.boundary_replay_samples_per_pair = int(getattr(args, "boundary_replay_samples_per_pair", 12))

#         self.boundary_replay_max_pairs = int(getattr(args, "boundary_replay_max_pairs", 24))

#         self.boundary_replay_parallel_scale = float(getattr(args, "boundary_replay_parallel_scale", 0.15))

#         self.boundary_replay_residual_scale = float(getattr(args, "boundary_replay_residual_scale", 0.05))

#         self.joint_old_new_ce_weight = float(getattr(args, "joint_old_new_ce_weight", 1.0))

#         self.geometry_energy_margin_weight = float(getattr(args, "geometry_energy_margin_weight", 0.25))

#         self.old_new_invasion_weight = float(getattr(args, "old_new_invasion_weight", 0.25))

#         self.use_pretrain_incremental_baseline = self._arg_bool("use_pretrain_incremental_baseline", True)

#         self.use_risk_weighted_replay = self._arg_bool("use_risk_weighted_replay", True)

#         self.risk_replay_min_samples = int(getattr(args, "risk_replay_min_samples", 4))

#         self.risk_replay_max_multiplier = float(getattr(args, "risk_replay_max_multiplier", 3.0))

#         self.risk_center_margin = float(getattr(args, "risk_center_margin", 1.0))

#         self.risk_subspace_weight = float(getattr(args, "risk_subspace_weight", 1.0))

#         self.risk_band_weight = float(getattr(args, "risk_band_weight", 0.25))

#         self.risk_reliability_weighted = self._arg_bool("risk_reliability_weighted", True)

#         self.reliability_gated_admission = self._arg_bool("reliability_gated_admission", True)

#         self.admission_min_gate = float(getattr(args, "admission_min_gate", 0.35))

#         self.admission_shrink_floor = float(getattr(args, "admission_shrink_floor", 0.15))

#         self.admission_low_rank_cap = int(getattr(args, "admission_low_rank_cap", 2))

#         self.risk_aware_descriptor_correction = self._arg_bool("risk_aware_descriptor_correction", True)

#         self.descriptor_correction_risk_threshold = float(getattr(args, "descriptor_correction_risk_threshold", 0.75))

#         self.descriptor_correction_overlap_threshold = float(getattr(args, "descriptor_correction_overlap_threshold", 0.60))

#         self.descriptor_correction_center_step = float(getattr(args, "descriptor_correction_center_step", 0.20))

#         self.descriptor_correction_subspace_eta = float(getattr(args, "descriptor_correction_subspace_eta", 0.85))

#         self.descriptor_correction_variance_shrink = float(getattr(args, "descriptor_correction_variance_shrink", 0.15))

#         self.descriptor_correction_basis_strength = float(getattr(args, "descriptor_correction_basis_strength", self.descriptor_correction_subspace_eta))

#         self.descriptor_correction_mean_push = float(getattr(args, "descriptor_correction_mean_push", self.descriptor_correction_center_step))

#         self.descriptor_correction_var_shrink = float(getattr(args, "descriptor_correction_var_shrink", 0.15))

#         self.descriptor_correction_topk_old = int(getattr(args, "descriptor_correction_topk_old", 3))

#         self.descriptor_risk_sep_weight = float(getattr(args, "descriptor_risk_sep_weight", 0.20))

#         self.descriptor_risk_sep_target = float(getattr(args, "descriptor_risk_sep_target", 0.35))

#         self.risk_sep_weight = float(getattr(args, "risk_sep_weight", self.descriptor_risk_sep_weight))

#         self.risk_sep_overlap_target = float(getattr(args, "risk_sep_overlap_target", self.descriptor_risk_sep_target))

#         self.risk_sep_active_threshold = float(getattr(args, "risk_sep_active_threshold", 0.50))



#         self.refine_new_descriptors = self._arg_bool("refine_new_descriptors", True)

#         self.descriptor_refine_steps = int(getattr(args, "descriptor_refine_steps", 50))

#         self.descriptor_refine_lr = float(getattr(args, "descriptor_refine_lr", 1e-3))

#         self.descriptor_trust_weight = float(getattr(args, "descriptor_trust_weight", 1.0))

#         self.descriptor_subspace_collision_weight = float(getattr(args, "descriptor_subspace_collision_weight", 0.20))

#         self.descriptor_subspace_overlap_max = float(getattr(args, "descriptor_subspace_overlap_max", 0.35))

#         self.descriptor_center_margin_weight = float(getattr(args, "descriptor_center_margin_weight", 0.10))

#         self.descriptor_center_collision_weight = float(getattr(args, "descriptor_center_collision_weight", self.descriptor_center_margin_weight))

#         self.descriptor_center_margin = float(getattr(args, "descriptor_center_margin", 0.50))

#         self.descriptor_volume_weight = float(getattr(args, "descriptor_volume_weight", 0.05))

#         self.descriptor_volume_control_weight = float(getattr(args, "descriptor_volume_control_weight", self.descriptor_volume_weight))

#         self.descriptor_volume_margin = float(getattr(args, "descriptor_volume_margin", 0.0))


#         self.use_energy_calibrator = self._arg_bool("use_energy_calibrator", False)

#         self.energy_calibrator_type = str(getattr(args, "energy_calibrator_type", "old_new")).lower().strip()

#         self.energy_calibration_weight = float(getattr(args, "energy_calibration_weight", 0.0 if not self.use_energy_calibrator else 1e-3))


#         # Legacy neural/plasticity paths stay disabled; SRGP spectral geometry remains enabled.

#         self.use_descriptor_refinement = False

#         self.use_geometry_transport = bool(self._arg_bool("use_sglat_transport", False) or self._arg_bool("use_geometry_transport", False))

#         self.use_spectral_geometry = self._arg_bool("use_spectral_geometry", True)

#         self.use_bicyc_geometry_cycle = False

#         self.allow_incremental_projection_training = False

#         self.freeze_projection_during_incremental = True

#         self.spectral_rank = 0

#         self.spectral_variance_floor = float(getattr(args, "spectral_variance_floor", 1e-5))

#         self.spectral_energy_weight = float(getattr(args, "spectral_energy_weight", 0.05))

#         self.spectral_derivative_weight = float(getattr(args, "spectral_derivative_weight", 0.50))

#         self.spectral_second_derivative_weight = float(getattr(args, "spectral_second_derivative_weight", 0.25))

#         self.spectral_require_physical_summary = self._arg_bool("spectral_require_physical_summary", True)

#         self.band_energy_weight = float(getattr(args, "band_energy_weight", 0.0))

#         self.spectral_reliability_energy_weight = float(getattr(args, "spectral_reliability_energy_weight", 0.0))

#         self.spectral_residual_variance_scale = float(getattr(args, "spectral_residual_variance_scale", 1.0))

#         # G²RPA adapter controls. These are active only when

#         # --incremental_update_mode geometry_gated_adapter is selected.

#         self.adapter_lr = float(getattr(args, "adapter_lr", 5e-4))

#         self.adapter_weight_decay = float(getattr(args, "adapter_weight_decay", 0.0))

#         self.g2rpa_adapter_weight = float(getattr(args, "g2rpa_adapter_weight", 1.0))

#         self.adapter_old_delta_weight = float(getattr(args, "adapter_old_delta_weight", 1.0))

#         self.adapter_old_gate_weight = float(getattr(args, "adapter_old_gate_weight", 0.75))

#         self.adapter_old_energy_weight = float(getattr(args, "adapter_old_energy_weight", 0.25))

#         self.adapter_old_margin_weight = float(getattr(args, "adapter_old_margin_weight", 0.25))

#         self.adapter_delta_weight = float(getattr(args, "adapter_delta_weight", 0.10))

#         self.adapter_new_gate_weight = float(getattr(args, "adapter_new_gate_weight", 0.05))

#         self.adapter_new_gate_target = float(getattr(args, "adapter_new_gate_target", 0.25))

#         self.adapter_new_gate_max_target = float(getattr(args, "adapter_new_gate_max_target", 0.75))

#         self.bss_weight = 0.0

#         self.sym_bss_weight = 0.0

#         self.gdr_weight = 0.0

#         self.anchor_consistency_weight = 0.0

#         self.geometry_calibration_weight = 0.0
        
#         self._force_clean_main_path_args()

#         self._propagate_clean_energy_config_to_model()

#         self._assert_global_architecture_contract()

#         self._assert_updated_stack_contract(phase=0)

#         self._set_base_trainable_params()

#     def _propagate_clean_energy_config_to_model(self) -> None:

#         """Keep model/classifier runtime attributes aligned with fixed loss defaults."""

#         clf = getattr(self.model, "classifier", None)

#         if clf is not None:

#             for key in (

#                 "use_logdet_energy",

#                 "logdet_energy_weight",

#                 "logdet_normalize_by_dim",

#                 "center_logdet_energy",

#                 "energy_normalize_by_dim",

#                 "reliability_energy_weight",

#                 "residual_variance_scale",

#                 "invalid_class_energy",

#             ):

#                 if hasattr(clf, key):

#                     try:

#                         if hasattr(self, key):

#                             value = getattr(self, key)

#                         else:

#                             value = getattr(self.args, key)

#                         setattr(clf, key, value)

#                     except Exception:

#                         pass

#             if hasattr(clf, "normalize_logits"):

#                 clf.normalize_logits = False

#             if hasattr(clf, "use_spectral_geometry"):

#                 clf.use_spectral_geometry = bool(self.use_spectral_geometry)

#             for key in (

#                 "spectral_energy_weight",

#                 "spectral_derivative_weight",

#                 "spectral_second_derivative_weight",

#                 "spectral_require_physical_summary",

#                 "spectral_residual_variance_scale",

#                 "spectral_reliability_energy_weight",

#                 "band_energy_weight",

#             ):

#                 if hasattr(clf, key) and hasattr(self, key):

#                     try:

#                         setattr(clf, key, getattr(self, key))

#                     except Exception:

#                         pass



#     def _assert_updated_stack_contract(self, phase: Optional[int] = None) -> None:

#         """Fail early when the orchestrator is used with stale components.



#         This catches the common failure mode where trainer.py is updated but the

#         run still imports an old GeometryBank/loss/classifier that bypasses

#         risk-aware replay, reliability-gated admission, or logdet energy.

#         """

#         strict = self._arg_bool("strict_updated_stack", True)

#         missing: List[str] = []



#         if not hasattr(self.model, "extract_projected_features"):

#             missing.append("model.extract_projected_features")

#         if not hasattr(self.model, "get_subspace_bank"):

#             missing.append("model.get_subspace_bank")



#         clf = getattr(self.model, "classifier", None)

#         if clf is None or not hasattr(clf, "geometry_energy"):

#             missing.append("model.classifier.geometry_energy")

#         elif hasattr(clf, "method_summary"):

#             try:

#                 summary = clf.method_summary()

#                 # SRGP spectral energy is allowed, but it must be gated by physical spectral summaries.

#                 if bool(summary.get("uses_spectral_energy", False)) and not hasattr(clf, "spectral_residual_energy"):

#                     missing.append("classifier.spectral_residual_energy")

#             except Exception:

#                 pass



#         gb = getattr(self.model, "geometry_bank", None)

#         if unified_spectral_geometry_loss is None:

#             missing.append("losses.loss.unified_spectral_geometry_loss")

#         if sample_boundary_geometry_features is None:

#             missing.append("losses.loss.sample_boundary_geometry_features")

#         if gb is None:

#             missing.append("model.geometry_bank")

#         else:

#             for name in ("validate_consistency", "freeze_classes_up_to", "get_valid_mask"):

#                 if not hasattr(gb, name):

#                     missing.append(f"geometry_bank.{name}")

#             if phase is not None and int(phase) > 0:

#                 # RSGI needs pairwise risk diagnostics and a way to update only new rows.

#                 for name in ("pairwise_subspace_overlap", "geometry_conflict_matrix"):

#                     if not hasattr(gb, name):

#                         missing.append(f"geometry_bank.{name}")

#                 if not (hasattr(gb, "apply_refined_feature_rows") or hasattr(gb, "update_class_geometry") or hasattr(gb, "update_class")):

#                     missing.append("geometry_bank.apply_refined_feature_rows/update_class_geometry")

#                 # SGLAT-HSI boundary replay is provided by losses.loss.sample_boundary_geometry_features

#                 # and the current GeometryBank tensors. A bank-native risk sampler is optional fallback,

#                 # not a hard dependency.



#         for name in ("_safe_get_subspace_bank", "_snapshot_old_bank", "_old_bank_integrity_snapshot", "_assert_old_bank_integrity"):

#             if not hasattr(self, name):

#                 missing.append(f"TrainerHelper.{name}")



#         if missing:

#             msg = "Updated GEO-NECIL-HSI stack contract failed; stale component(s): " + ", ".join(missing)

#             if strict:

#                 raise RuntimeError(msg)

#             print(f"[WARN] {msg}")



#     def _assert_incremental_preflight(self, phase: int, old_class_count: int) -> None:

#         """Verify that incremental training starts from a frozen, valid old bank."""

#         phase = int(phase)

#         old_class_count = int(old_class_count)

#         self._assert_updated_stack_contract(phase=phase)

#         if self._adapter_mode_enabled():

#             if not hasattr(self.model, "geometry_plastic_adapter"):

#                 raise RuntimeError("G²RPA incremental preflight failed: model.geometry_plastic_adapter is missing.")

#             if hasattr(self.model, "adapter_enabled") and not bool(self.model.adapter_enabled()):

#                 # The trainability step should enable it.  Fail early if the model

#                 # still reports descriptor-only behavior.

#                 raise RuntimeError("G²RPA selected but model.adapter_enabled() is False. Check necil_model.py and trainer args.")

#         if old_class_count <= 0:

#             raise RuntimeError(f"Incremental phase {phase} requires old_class_count > 0.")



#         cert = getattr(self, "_last_base_geometry_certificate", None)

#         if cert is None:

#             cert = getattr(self.model, "base_geometry_certificate", None)

#         if self._arg_bool("enforce_base_geometry_certificate", False):

#             if not isinstance(cert, dict) or not bool(cert.get("ok", cert.get("valid", False))):

#                 raise RuntimeError(

#                     "Base geometry certificate is missing or failed. Do not start incremental insertion from an uncertified base bank."

#                 )



#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()

#         if hasattr(self, "_canonicalize_bank"):

#             bank = self._canonicalize_bank(bank)

#         old_ids = list(range(old_class_count))

#         if hasattr(self, "_validate_bank_has_classes"):

#             self._validate_bank_has_classes(bank, old_ids)

#         gb = getattr(self.model, "geometry_bank", None)

#         if gb is not None and hasattr(gb, "validate_consistency"):

#             gb.validate_consistency(strict=True)

#         if gb is not None and hasattr(gb, "freeze_classes_up_to"):

#             gb.freeze_classes_up_to(old_class_count)



#     def _current_runtime_contract(self) -> Dict[str, object]:

#         """Small audit payload stored in checkpoints."""

#         adapter_mode = self._adapter_mode_enabled()

#         adapter_trainable = any(

#             p.requires_grad and "geometry_plastic_adapter" in name

#             for name, p in self.model.named_parameters()

#         )

#         return {

#             "feature_space": "canonical projected z" if not adapter_mode else "canonical z plus geometry-gated residual adapter",

#             "classifier": "SRGP spectral-residual low-rank Gaussian geometry energy",

#             "incremental_update_mode": self._incremental_update_mode(),

#             "feature_plasticity_after_base": bool(adapter_mode),

#             "feature_plasticity_scope": "geometry_plastic_adapter_only" if adapter_mode else "none",

#             "geometry_gated_adapter_present": bool(hasattr(self.model, "geometry_plastic_adapter")),

#             "geometry_gated_adapter_trainable": bool(adapter_trainable),

#             "adapter_max_scale": float(getattr(getattr(self.model, "geometry_plastic_adapter", None), "max_scale", getattr(self.args, "adapter_max_scale", 0.0)) or 0.0),

#             "uses_logdet_energy": bool(self.use_logdet_energy),

#             "logdet_energy_weight": float(self.logdet_energy_weight),

#             "row_energy_standardization": False,

#             "boundary_geometry_replay": bool(getattr(self.args, "use_boundary_geometry_replay", True)),
#             "boundary_replay_risk_threshold": float(getattr(self.args, "boundary_replay_risk_threshold", 0.35)),
#             "boundary_replay_overlap_threshold": float(getattr(self.args, "boundary_replay_overlap_threshold", 0.30)),
#             "use_sglat_transport": bool(getattr(self.args, "use_sglat_transport", False)),
#             "transport_identity_blend": float(getattr(self.args, "transport_identity_blend", 0.75)),
#             "transport_low_rank": int(getattr(self.args, "transport_low_rank", 4)),
#             "transport_ema": float(getattr(self.args, "transport_ema", 0.97)),
#             "transport_after_adapter_epoch": int(getattr(self.args, "transport_after_adapter_epoch", 3)),
#             "risk_weighted_replay_fallback": bool(self.use_risk_weighted_replay),

#             "heuristic_reliability_gated_admission": bool(self.reliability_gated_admission),

#             "heuristic_risk_aware_descriptor_correction": bool(getattr(self, "risk_aware_descriptor_correction", getattr(self.args, "risk_aware_descriptor_correction", False))),

#             "unified_incremental_loss": "Boundary-Admitted Geometry Energy / unified_spectral_geometry_loss",

#             "uses_spectral_geometry": bool(self.use_spectral_geometry),

#             "spectral_energy_weight": float(getattr(self, "spectral_energy_weight", 0.0)),

#             "geometry_state_refinement": bool(self.refine_new_descriptors),

#             "descriptor_subspace_collision_weight": float(self.descriptor_subspace_collision_weight),

#             "descriptor_center_margin_weight": float(self.descriptor_center_margin_weight),

#             "descriptor_volume_weight": float(self.descriptor_volume_weight),

#             "raw_exemplars": False,

#             "kd_teacher": False,

#             "spectral_classifier_branch": "physical-summary-gated SRGP residual energy",

#         }



#     # ------------------------------------------------------------------

#     # Hard architecture contract

#     # ------------------------------------------------------------------

#     def _force_clean_main_path_args(self) -> None:

#         """Normalize dangerous legacy arguments while preserving the selected G²RPA mode.



#         This trainer still forbids old transport/calibrator/prompt-style paths.

#         The geometry-gated adapter is allowed only as a bounded residual module

#         after canonical z and is trained with old synthetic replay invariance.

#         """

#         mode = self._incremental_update_mode()

#         forced = {

#             "base_classifier_mode": "srgp",

#             "incremental_classifier_mode": "geometry_only",

#             "eval_classifier_mode": "geometry_only",

#             "incremental_update_mode": mode,

#             "use_boundary_geometry_replay": True,

#             "boundary_replay_risk_threshold": float(getattr(self.args, "boundary_replay_risk_threshold", 0.35)),

#             "boundary_replay_overlap_threshold": float(getattr(self.args, "boundary_replay_overlap_threshold", 0.30)),

#             "boundary_replay_samples_per_pair": int(getattr(self.args, "boundary_replay_samples_per_pair", 12)),

#             "boundary_replay_max_pairs": int(getattr(self.args, "boundary_replay_max_pairs", 24)),

#             "boundary_replay_parallel_scale": float(getattr(self.args, "boundary_replay_parallel_scale", 0.15)),

#             "boundary_replay_residual_scale": float(getattr(self.args, "boundary_replay_residual_scale", 0.05)),

#             "scbgr_commit_only_if_safe": True,
#             "use_sglat_transport": bool(self._arg_bool("use_sglat_transport", False)),
#             "use_geometry_transport": bool(self._arg_bool("use_geometry_transport", False)),
#             "allow_old_model_transport": bool(self._arg_bool("allow_old_model_transport", False)),
#             "allow_transport_without_adapter": bool(self._arg_bool("allow_transport_without_adapter", False)),
#             "transport_type": str(getattr(self.args, "transport_type", "ridge")),
#             "transport_ridge": float(getattr(self.args, "transport_ridge", 1e-3)),
#             "transport_ema": float(getattr(self.args, "transport_ema", 0.97)),
#             "transport_batches": int(getattr(self.args, "transport_batches", 20)),
#             "transport_identity_blend": float(getattr(self.args, "transport_identity_blend", 0.75)),
#             "transport_low_rank": int(getattr(self.args, "transport_low_rank", 4)),
#             "transport_min_reliability_gate": float(getattr(self.args, "transport_min_reliability_gate", 0.30)),
#             "transport_max_a_minus_i_fro": float(getattr(self.args, "transport_max_a_minus_i_fro", 1.5)),
#             "transport_max_b_norm": float(getattr(self.args, "transport_max_b_norm", 0.75)),
#             "transport_after_adapter_epoch": int(getattr(self.args, "transport_after_adapter_epoch", 3)),

#             "use_descriptor_refinement": False,

#             "use_geometry_calibrator": False,

#             "geometry_calibration_weight": 0.0,

#             # Legacy adapter flag stays false.  G²RPA uses model.use_geometry_gated_adapter.

#             "use_incremental_adapter": False,

#             "disable_incremental_adapter": mode != "geometry_gated_adapter",

#             "incremental_adapter_normalize": False,

#             "use_bicyc_geometry_cycle": False,

#             "bicyc_geometry_cycle_weight": 0.0,

#             "bicyc_cycle_weight": 0.0,

#             "allow_incremental_projection_training": False,

#             "freeze_projection_during_incremental": True,

#             "refresh_incremental_geometry_after_epoch": False,

#             "incremental_weight_anchor": 0.0,

#             "use_spectral_geometry": True,

#             "spectral_energy_weight": float(getattr(self.args, "spectral_energy_weight", 0.05)),

#             "spectral_derivative_weight": float(getattr(self.args, "spectral_derivative_weight", 0.50)),

#             "spectral_second_derivative_weight": float(getattr(self.args, "spectral_second_derivative_weight", 0.25)),

#             "band_energy_weight": float(getattr(self.args, "band_energy_weight", 0.0)),

#             # Align arg names consumed by the RSGI/G²RPA incremental trainer.

#             "reliability_gated_admission": bool(self._arg_bool("reliability_gated_admission", True)),

#             "risk_aware_descriptor_correction": bool(self._arg_bool("risk_aware_descriptor_correction", True)),

#             "descriptor_correction_basis_strength": float(getattr(self.args, "descriptor_correction_basis_strength", getattr(self.args, "descriptor_correction_subspace_eta", 0.85))),

#             "descriptor_correction_mean_push": float(getattr(self.args, "descriptor_correction_mean_push", getattr(self.args, "descriptor_correction_center_step", 0.20))),

#             "descriptor_correction_var_shrink": float(getattr(self.args, "descriptor_correction_var_shrink", 0.15)),

#             "risk_sep_weight": float(getattr(self.args, "risk_sep_weight", getattr(self.args, "descriptor_risk_sep_weight", 0.20))),

#             "risk_sep_overlap_target": float(getattr(self.args, "risk_sep_overlap_target", getattr(self.args, "descriptor_risk_sep_target", 0.35))),

#             "descriptor_overlap_target": float(getattr(self.args, "descriptor_overlap_target", 0.35)),

#             "descriptor_center_collision_weight": float(getattr(self.args, "descriptor_center_collision_weight", getattr(self.args, "descriptor_center_margin_weight", 0.10))),

#             "descriptor_volume_control_weight": float(getattr(self.args, "descriptor_volume_control_weight", getattr(self.args, "descriptor_volume_weight", 0.05))),

#             "risk_replay_reliability_weighted": bool(self._arg_bool("risk_replay_reliability_weighted", self._arg_bool("risk_reliability_weighted", True))),

#             "geometry_normalize_logits": False,

#             "bss_weight": 0.0,

#             "sym_bss_weight": 0.0,

#             "gdr_weight": 0.0,

#             "anchor_consistency_weight": 0.0,

#         }

#         if not self._arg_bool("use_energy_calibrator", False):

#             forced["energy_calibration_weight"] = 0.0

#         for key, value in forced.items():

#             try:

#                 setattr(self.args, key, value)

#             except Exception:

#                 pass



#         # Keep runtime attributes synchronized after the forced SRGP defaults.

#         self.incremental_update_mode = mode

#         self.use_spectral_geometry = True

#         self.spectral_energy_weight = float(getattr(self.args, "spectral_energy_weight", 0.05))

#         self.spectral_derivative_weight = float(getattr(self.args, "spectral_derivative_weight", 0.50))

#         self.spectral_second_derivative_weight = float(getattr(self.args, "spectral_second_derivative_weight", 0.25))

#         self.spectral_require_physical_summary = self._arg_bool("spectral_require_physical_summary", True)

#         self.band_energy_weight = float(getattr(self.args, "band_energy_weight", 0.0))

#         self.descriptor_correction_basis_strength = float(getattr(self.args, "descriptor_correction_basis_strength", 0.85))

#         self.descriptor_correction_mean_push = float(getattr(self.args, "descriptor_correction_mean_push", 0.20))

#         self.descriptor_correction_var_shrink = float(getattr(self.args, "descriptor_correction_var_shrink", 0.15))

#         self.risk_sep_weight = float(getattr(self.args, "risk_sep_weight", 0.20))

#         self.risk_sep_overlap_target = float(getattr(self.args, "risk_sep_overlap_target", 0.35))

#         self.descriptor_center_collision_weight = float(getattr(self.args, "descriptor_center_collision_weight", getattr(self.args, "descriptor_center_margin_weight", 0.10)))

#         self.descriptor_volume_control_weight = float(getattr(self.args, "descriptor_volume_control_weight", getattr(self.args, "descriptor_volume_weight", 0.05)))



#         if hasattr(self.model, "use_geometry_gated_adapter"):

#             self.model.use_geometry_gated_adapter = bool(mode == "geometry_gated_adapter")

#         if hasattr(self.model, "incremental_update_mode"):

#             self.model.incremental_update_mode = mode

#         if hasattr(self.model, "use_geometry_calibrator"):

#             self.model.use_geometry_calibrator = False

#         if hasattr(self.model, "use_bicyc_geometry_cycle"):

#             self.model.use_bicyc_geometry_cycle = False

#         if hasattr(self.model, "use_incremental_adapter"):

#             self.model.use_incremental_adapter = False



#         # Freeze stale paths.  In adapter mode, do not call disable_incremental_adapter(),

#         # because the updated NECILModel maps that legacy hook to freezing the

#         # geometry_plastic_adapter.  The incremental trainability function will

#         # unfreeze only geometry_plastic_adapter at the correct time.

#         if mode != "geometry_gated_adapter" and hasattr(self.model, "disable_incremental_adapter"):

#             self.model.disable_incremental_adapter()

#         if hasattr(self.model, "incremental_adapter") and hasattr(self.model.incremental_adapter, "normalize_output"):

#             self.model.incremental_adapter.normalize_output = False

#         if mode != "geometry_gated_adapter" and hasattr(self.model, "freeze_incremental_adapter"):

#             self.model.freeze_incremental_adapter()

#         if mode == "geometry_gated_adapter" and hasattr(self.model, "freeze_geometry_plastic_adapter"):

#             self.model.freeze_geometry_plastic_adapter()

#         if hasattr(self.model, "freeze_geometry_calibrator"):

#             self.model.freeze_geometry_calibrator()

#         if hasattr(self.model, "geometry_cycle_calibrator"):

#             self._freeze_module_if_present(self.model.geometry_cycle_calibrator)

#         if hasattr(self.model, "classifier"):

#             clf = self.model.classifier

#             if hasattr(clf, "use_spectral_geometry"):

#                 clf.use_spectral_geometry = bool(getattr(self.args, "use_spectral_geometry", True))

#             if hasattr(clf, "spectral_energy_weight"):

#                 clf.spectral_energy_weight = float(getattr(self.args, "spectral_energy_weight", 0.05))

#             if hasattr(clf, "band_energy_weight"):

#                 clf.band_energy_weight = float(getattr(self.args, "band_energy_weight", 0.0))



#     def _assert_global_architecture_contract(self) -> None:

#         mode = self._incremental_update_mode()

#         allowed_modes = {"srgp", "srgp_geometry", "spectral_residual_geometry", "geometry_only", "geometry", "geo"}

#         for key in ("base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode"):

#             raw = str(getattr(self.args, key, "srgp")).lower().strip()

#             if raw not in allowed_modes:

#                 raise RuntimeError(f"{key} must be one of {sorted(allowed_modes)} in SRGP trainer, got {raw!r}.")

#             self._normalize_classifier_mode(raw, context=key)



#         forbidden_true = [

#             "use_descriptor_refinement",

#             "use_geometry_calibrator",

#             # Legacy incremental_adapter flag is still forbidden; G²RPA is selected by

#             # incremental_update_mode=geometry_gated_adapter instead.

#             "use_incremental_adapter",

#             "use_bicyc_geometry_cycle",

#             "allow_incremental_projection_training",

#             "unsafe_ablation_bicyc_geometry_cycle",

#             "unsafe_ablation_projection_plasticity",

#             "unsafe_ablation_backbone_plasticity",

#         ]

#         bad = [k for k in forbidden_true if self._arg_bool(k, False)]

#         if bad:

#             raise RuntimeError(

#                 "Unified SGLAT-HSI/G²RPA trainer forbids these active switches: "

#                 f"{bad}. Use incremental_update_mode=scbgr for the clean method or geometry_gated_adapter for the adapter ablation."

#             )



#         zero_weight_keys = (

#             "bss_weight",

#             "sym_bss_weight",

#             "gdr_weight",

#             "anchor_consistency_weight",

#             "geometry_calibration_weight",

#             "bicyc_geometry_cycle_weight",

#             "bicyc_cycle_weight",

#         )

#         for key in zero_weight_keys:

#             if float(getattr(self.args, key, 0.0)) != 0.0:

#                 raise RuntimeError(f"{key} must be 0.0 in the SRGP/G²RPA architecture path.")



#         if not self._arg_bool("freeze_projection_during_incremental", True):

#             raise RuntimeError("SRGP/G²RPA requires freeze_projection_during_incremental=True.")

#         if self._arg_bool("geometry_normalize_logits", False):

#             raise RuntimeError("geometry_normalize_logits must be False; row-wise logit normalization hides old/new energy bias.")



#         if mode == "geometry_gated_adapter":

#             if not hasattr(self.model, "geometry_plastic_adapter"):

#                 raise RuntimeError(

#                     "incremental_update_mode=geometry_gated_adapter requires the updated NECILModel "

#                     "with model.geometry_plastic_adapter. Replace necil_model.py first."

#                 )

#             max_scale = float(getattr(getattr(self.model, "geometry_plastic_adapter", None), "max_scale", getattr(self.args, "adapter_max_scale", 0.0)) or 0.0)

#             if max_scale <= 0.0:

#                 raise RuntimeError("geometry_gated_adapter selected but adapter_max_scale <= 0; no feature plasticity is possible.")



#         if hasattr(self.model, "incremental_adapter") and bool(getattr(self.model.incremental_adapter, "normalize_output", False)):

#             raise RuntimeError("incremental_adapter.normalize_output=True corrupts GeometryBank coordinates.")

#         if hasattr(self.model, "use_bicyc_geometry_cycle") and self._as_bool(getattr(self.model, "use_bicyc_geometry_cycle", False)):

#             raise RuntimeError("model.use_bicyc_geometry_cycle=True is forbidden in this trainer.")

#         if hasattr(self.model, "use_geometry_calibrator") and self._as_bool(getattr(self.model, "use_geometry_calibrator", False)):

#             raise RuntimeError("model.use_geometry_calibrator=True is forbidden in this trainer.")

#         if hasattr(self.model, "use_incremental_adapter") and self._as_bool(getattr(self.model, "use_incremental_adapter", False)):

#             raise RuntimeError("model.use_incremental_adapter=True is legacy/stale. Use model.use_geometry_gated_adapter for G²RPA.")



#     def _set_base_trainable_params(self) -> None:

#         """Base trains backbone/projection/norm; bank/classifier/adapters remain frozen."""

#         self._force_clean_main_path_args()

#         self._propagate_clean_energy_config_to_model()

#         for name, p in self.model.named_parameters():

#             blocked = (

#                 name.startswith("classifier.")

#                 or name.startswith("geometry_bank.")

#                 or name.startswith("geometry_calibrator.")

#                 or name.startswith("geometry_cycle_calibrator.")

#                 or name.startswith("incremental_adapter.")

#                 or name.startswith("geometry_plastic_adapter.")

#                 or name.startswith("base_ce_head.")

#             )

#             p.requires_grad = not blocked

#         if hasattr(self.model, "freeze_energy_calibrator"):

#             self.model.freeze_energy_calibrator()

#         if hasattr(self.model, "freeze_geometry_calibrator"):

#             self.model.freeze_geometry_calibrator()

#         if hasattr(self.model, "freeze_incremental_adapter"):

#             self.model.freeze_incremental_adapter()

#         if hasattr(self.model, "freeze_geometry_plastic_adapter"):

#             self.model.freeze_geometry_plastic_adapter()



#     def _set_incremental_trainable_params(self, old_class_count: int = 0) -> List[torch.nn.Parameter]:

#         """Set incremental trainability for descriptor-only RSGI or G²RPA.



#         descriptor_only: no model parameters except optional energy calibrator.

#         geometry_gated_adapter: only model.geometry_plastic_adapter is trainable;

#         backbone, projection, classifier, and old GeometryBank coordinates stay frozen.

#         """

#         del old_class_count

#         self._force_clean_main_path_args()

#         self._propagate_clean_energy_config_to_model()

#         mode = self._incremental_update_mode()



#         for _, p in self.model.named_parameters():

#             p.requires_grad = False



#         if hasattr(self.model, "freeze_projection_head"):

#             self.model.freeze_projection_head()

#         if hasattr(self.model, "freeze_backbone_only"):

#             self.model.freeze_backbone_only()

#         if hasattr(self.model, "freeze_geometry_calibrator"):

#             self.model.freeze_geometry_calibrator()

#         if hasattr(self.model, "geometry_cycle_calibrator"):

#             self._freeze_module_if_present(self.model.geometry_cycle_calibrator)

#         if hasattr(self.model, "use_bicyc_geometry_cycle"):

#             self.model.use_bicyc_geometry_cycle = False

#         if hasattr(self.model, "use_geometry_calibrator"):

#             self.model.use_geometry_calibrator = False

#         if hasattr(self.model, "use_incremental_adapter"):

#             self.model.use_incremental_adapter = False



#         if mode == "geometry_gated_adapter":

#             if not hasattr(self.model, "geometry_plastic_adapter"):

#                 raise RuntimeError(

#                     "G²RPA mode selected but model.geometry_plastic_adapter is missing. "

#                     "Use the updated necil_model.py."

#                 )

#             if hasattr(self.model, "use_geometry_gated_adapter"):

#                 self.model.use_geometry_gated_adapter = True

#             if hasattr(self.model, "incremental_update_mode"):

#                 self.model.incremental_update_mode = mode

#             if hasattr(self.model, "enable_incremental_adapter"):

#                 # Updated NECILModel maps this legacy hook to the geometry-gated adapter only.

#                 self.model.enable_incremental_adapter()

#             if hasattr(self.model, "unfreeze_geometry_plastic_adapter"):

#                 self.model.unfreeze_geometry_plastic_adapter()

#             else:

#                 for p in self.model.geometry_plastic_adapter.parameters():

#                     p.requires_grad = True



#             # Energy calibration is deliberately disabled in the main G²RPA path.

#             # Otherwise we cannot tell whether improvement came from real feature

#             # plasticity or score rescaling. Keep it as a separate ablation.

#             self.use_energy_calibrator = False

#             if hasattr(self.model, "freeze_energy_calibrator"):

#                 self.model.freeze_energy_calibrator()



#             bad = [

#                 name for name, p in self.model.named_parameters()

#                 if p.requires_grad and "geometry_plastic_adapter" not in name

#             ]

#             if bad:

#                 raise RuntimeError(f"G²RPA mode allows only geometry_plastic_adapter params, got: {bad[:30]}")

#             params = [p for p in self.model.parameters() if p.requires_grad]

#             if not params:

#                 raise RuntimeError("G²RPA mode selected but no adapter parameters are trainable.")

#             return params



#         # scbgr path: no model parameters except optional score calibrator.
#         # Geometry-state parameters are optimized inside IncrementalPhaseTrainer.

#         if hasattr(self.model, "disable_incremental_adapter"):

#             self.model.disable_incremental_adapter()

#         if hasattr(self.model, "freeze_incremental_adapter"):

#             self.model.freeze_incremental_adapter()

#         if hasattr(self.model, "freeze_geometry_plastic_adapter"):

#             self.model.freeze_geometry_plastic_adapter()

#         if hasattr(self.model, "use_geometry_gated_adapter"):

#             self.model.use_geometry_gated_adapter = False



#         use_cal = self._arg_bool("use_energy_calibrator", False)

#         self.use_energy_calibrator = use_cal

#         if use_cal:

#             if hasattr(self.model, "enable_energy_calibration"):

#                 self.model.enable_energy_calibration(True, calibrator_type=self.energy_calibrator_type)

#             if hasattr(self.model, "unfreeze_energy_calibrator"):

#                 self.model.unfreeze_energy_calibrator()

#             else:

#                 for name, p in self.model.named_parameters():

#                     if any(k in name for k in (

#                         "energy_calibrator",

#                         "old_log_scale",

#                         "new_log_scale",

#                         "old_bias",

#                         "new_bias",

#                         "log_scale_raw",

#                         "bias_raw",

#                     )):

#                         p.requires_grad = True

#         elif hasattr(self.model, "freeze_energy_calibrator"):

#             self.model.freeze_energy_calibrator()



#         allowed = (

#             "energy_calibrator",

#             "old_log_scale",

#             "new_log_scale",

#             "old_bias",

#             "new_bias",

#             "log_scale_raw",

#             "bias_raw",

#         )

#         bad = [name for name, p in self.model.named_parameters() if p.requires_grad and not any(k in name for k in allowed)]

#         if bad:

#             raise RuntimeError(f"Invalid incremental trainable parameters in descriptor-only path: {bad[:30]}")



#         params = [p for p in self.model.parameters() if p.requires_grad]

#         return params



#     def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:

#         """Compatibility hook used by IncrementalPhaseTrainer."""

#         return self._set_incremental_trainable_params(old_class_count)



#     def _has_feature_plasticity(self) -> bool:

#         """True only for the approved G²RPA residual adapter, never backbone/projection drift."""

#         return self._adapter_mode_enabled()



#     def _has_descriptor_plasticity(self) -> bool:

#         return bool(getattr(self, "refine_new_descriptors", False))



#     def _has_energy_calibration_plasticity(self) -> bool:

#         return any(

#             p.requires_grad and any(k in name for k in ("energy_calibrator", "old_log_scale", "new_log_scale", "old_bias", "new_bias", "log_scale_raw", "bias_raw"))

#             for name, p in self.model.named_parameters()

#         )



#     # ------------------------------------------------------------------

#     # Classifier modes and phase state

#     # ------------------------------------------------------------------

#     def _base_classifier_mode(self) -> str:

#         return self._normalize_classifier_mode(getattr(self.args, "base_classifier_mode", "srgp"), context="base_classifier_mode")



#     def _inc_classifier_mode(self) -> str:

#         return self._normalize_classifier_mode(getattr(self.args, "incremental_classifier_mode", "srgp"), context="incremental_classifier_mode")



#     def _eval_classifier_mode(self) -> str:

#         return self._normalize_classifier_mode(getattr(self.args, "eval_classifier_mode", "srgp"), context="eval_classifier_mode")



#     def _set_model_phase_and_old_count(self, phase: int, old_class_count: int) -> None:

#         phase = int(phase)

#         old_class_count = int(old_class_count)

#         if hasattr(self.model, "set_phase"):

#             self.model.set_phase(phase)

#         else:

#             self.model.current_phase = phase

#         if hasattr(self.model, "set_old_class_count"):

#             self.model.set_old_class_count(old_class_count)

#         else:

#             self.model.old_class_count = old_class_count



#     # ------------------------------------------------------------------

#     # Scoring / state helpers

#     # ------------------------------------------------------------------

#     def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

#         if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:

#             return self._zero(logits)

#         labels = labels.long().view(-1).to(logits.device)

#         if labels.numel() != logits.size(0):

#             raise RuntimeError(f"CE batch mismatch: logits={logits.size(0)}, labels={labels.numel()}")

#         min_label = int(labels.min().detach().item())

#         max_label = int(labels.max().detach().item())

#         if min_label < 0 or max_label >= logits.size(1):

#             raise RuntimeError(f"CE label range [{min_label},{max_label}] incompatible with logits width={logits.size(1)}")

#         logits = logits.clamp(min=-float(self.ce_logit_clip), max=float(self.ce_logit_clip))

#         return F.cross_entropy(logits, labels, label_smoothing=float(self.label_smoothing))



#     def _base_geometry_global_metrics(self) -> Dict[str, float]:

#         if not hasattr(self.model, "geometry_bank"):

#             return {}

#         gb = self.model.geometry_bank

#         if hasattr(gb, "geometry_diagnostics"):

#             try:

#                 diag = gb.geometry_diagnostics()

#             except Exception:

#                 return {}

#         elif hasattr(gb, "geometry_health_summary"):

#             try:

#                 diag = gb.geometry_health_summary()

#             except Exception:

#                 return {}

#         else:

#             return {}

#         out: Dict[str, float] = {}

#         for key, value in diag.items() if isinstance(diag, dict) else []:

#             if torch.is_tensor(value) and value.numel() == 1:

#                 out[str(key)] = float(value.detach().cpu().item())

#             elif isinstance(value, (int, float)):

#                 out[str(key)] = float(value)

#         return out



#     def _select_score(self, val_stats: Dict[str, float], phase: int) -> float:

#         metric = str(getattr(self, "best_state_metric", getattr(self.args, "best_state_metric", "hm"))).lower().strip()

#         if int(phase) > 0:

#             if metric in {"acc", "oa"}:

#                 return float(val_stats.get("acc", 0.0))

#             if metric in {"old_new_min", "min"}:

#                 return min(float(val_stats.get("old_acc", 0.0)), float(val_stats.get("new_acc", 0.0)))

#             if metric in {"loss", "val_loss"}:

#                 return -float(val_stats.get("loss", 0.0))

#             return float(val_stats.get("hm", 0.0))

#         return self._select_base_checkpoint_score(val_stats, getattr(self, "_last_base_geom_stats", None))



#     def _select_base_checkpoint_score(self, val_stats: Dict[str, float], geom_stats: Optional[Dict[str, float]] = None) -> float:

#         metric = str(getattr(self, "best_state_metric", getattr(self.args, "best_state_metric", "geometry_score"))).lower().strip()

#         acc = float(val_stats.get("acc", val_stats.get("oa", 0.0)))

#         if metric in {"acc", "oa", "hm", "h", "harmonic"}:

#             return acc

#         if metric in {"loss", "val_loss"}:

#             return -float(val_stats.get("loss", 0.0))

#         geom = geom_stats if isinstance(geom_stats, dict) else {}

#         reserve = float(geom.get("geometry_reserve_score", geom.get("pgr_reserve_score", geom.get("reserve_score", 0.0))))

#         feature_overlap = float(geom.get("feature_subspace_overlap", geom.get("raw_feature_subspace_overlap", 0.0)))

#         band_overlap = float(geom.get("band_overlap", 0.0))

#         spectral_shape_overlap = float(geom.get("spectral_shape_overlap", geom.get("max_spectral_shape_similarity", 0.0)))

#         conflict_mean = float(geom.get("geometry_conflict_mean", 0.0))

#         conflict_max = float(geom.get("geometry_conflict_max", 0.0))

#         return acc + reserve - 10.0 * feature_overlap - 1.0 * band_overlap - 1.0 * spectral_shape_overlap - 4.0 * conflict_mean - 1.0 * conflict_max



#     def _capture_state(self) -> Dict[str, torch.Tensor]:

#         return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}



#     def _print_trainable_summary(self, phase: int) -> None:

#         trainable = [(n, int(p.numel())) for n, p in self.model.named_parameters() if p.requires_grad]

#         total = sum(n for _, n in trainable)

#         if int(phase) == 0:

#             print(f"[Trainable] Base phase {phase}: {total:,} parameters in {len(trainable)} tensors")

#             print("[Base Objective] CE + SRPGR | compact-separation + reserve + spectral-shape bank construction")

#         else:

#             mode = self._incremental_update_mode()

#             print(f"[Trainable] Incremental phase {phase}: {total:,} parameters in {len(trainable)} tensors | mode={mode}")

#             if mode == "geometry_gated_adapter":

#                 print("[Incremental Objective] G²RPA + SGLAT-HSI: train only geometry_plastic_adapter; boundary replay and unified admission still protect old geometry")

#             elif total == 0:

#                 print("[Incremental Objective] SGLAT-HSI: boundary replay + unified geometry-state admission; no model-parameter drift")

#             else:

#                 print("[Incremental Objective] SGLAT-HSI + optional bounded score calibration only; backbone/projection/old bank frozen")

#         if self.debug:

#             for name, count in trainable[:150]:

#                 print(f"  {name}: {count:,}")



#     # ------------------------------------------------------------------

#     # Validation helpers

#     # ------------------------------------------------------------------

#     def _seen_classes_for_phase(self, phase: int, fallback_labels: Optional[torch.Tensor] = None) -> List[int]:

#         if hasattr(self.dataset, "get_classes_up_to_phase"):

#             try:

#                 seen = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase))]

#                 if seen:

#                     return sorted(set(seen))

#             except Exception:

#                 pass

#         seen: List[int] = []

#         if hasattr(self.dataset, "phase_to_classes"):

#             for p in range(int(phase) + 1):

#                 try:

#                     seen.extend(int(c) for c in self.dataset.phase_to_classes[p])

#                 except Exception:

#                     pass

#         if not seen and torch.is_tensor(fallback_labels) and fallback_labels.numel() > 0:

#             seen = [int(c) for c in fallback_labels.detach().cpu().unique(sorted=True).tolist()]

#         if not seen:

#             raise RuntimeError(f"Cannot resolve seen classes for phase={phase}")

#         return sorted(set(seen))



#     def _assert_labels_in_seen_classes(self, labels: torch.Tensor, seen_classes: Iterable[int], *, context: str) -> None:

#         labels = labels.long().view(-1)

#         if labels.numel() == 0:

#             return

#         seen = torch.as_tensor([int(c) for c in seen_classes], device=labels.device, dtype=torch.long)

#         if hasattr(torch, "isin"):

#             ok = torch.isin(labels, seen)

#         else:

#             ok = torch.zeros_like(labels, dtype=torch.bool)

#             for c in seen:

#                 ok |= labels.eq(c)

#         if not bool(ok.all().item()):

#             bad = labels[~ok].detach().cpu().unique(sorted=True).tolist()

#             raise RuntimeError(f"{context}: labels outside seen classes. bad={bad}, seen={seen.detach().cpu().tolist()}")



#     def _mask_logits_to_seen_classes(self, logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:

#         if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:

#             raise RuntimeError(f"logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")

#         seen = torch.as_tensor([int(c) for c in seen_classes], device=logits.device, dtype=torch.long)

#         if seen.numel() == 0:

#             raise RuntimeError("Cannot mask logits with empty seen classes")

#         if int(seen.min().item()) < 0 or int(seen.max().item()) >= logits.size(1):

#             raise RuntimeError(f"Seen classes incompatible with logits width={logits.size(1)}: {seen.detach().cpu().tolist()}")

#         masked = torch.full_like(logits, -1e9)

#         masked.index_copy_(1, seen, logits.index_select(1, seen))

#         return masked



#     def _old_new_classes_for_validation(self, phase: int, old_class_count: int, seen_classes: Iterable[int]) -> Tuple[List[int], List[int]]:

#         """Resolve old/new class ids without assuming non-contiguous labels."""

#         seen = [int(c) for c in seen_classes]

#         old: List[int] = []

#         if int(old_class_count) > 0:

#             if hasattr(self.dataset, "get_classes_up_to_phase") and int(phase) > 0:

#                 try:

#                     old = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase) - 1)]

#                 except Exception:

#                     old = []

#             if not old:

#                 old = [int(c) for c in seen if int(c) < int(old_class_count)]

#         old_set = set(old)

#         new = [int(c) for c in seen if int(c) not in old_set]

#         return sorted(old_set), sorted(new)



#     def _label_membership_mask(self, labels: torch.Tensor, class_ids: Iterable[int]) -> torch.Tensor:

#         ids = [int(c) for c in class_ids]

#         mask = torch.zeros_like(labels, dtype=torch.bool)

#         for c in ids:

#             mask |= labels.eq(int(c))

#         return mask



#     @torch.no_grad()

#     def _validate_split_metrics(self, loader, old_class_count: int) -> Dict[str, float]:

#         self._assert_global_architecture_contract()

#         self.model.eval()

#         old_class_count = int(old_class_count)

#         phase = int(getattr(self.model, "current_phase", 0))

#         seen_classes = self._seen_classes_for_phase(phase)

#         mode = self._eval_classifier_mode()

#         old_classes, new_classes = self._old_new_classes_for_validation(phase, old_class_count, seen_classes)



#         total_loss = 0.0

#         total_correct = 0

#         total = 0

#         batches = 0

#         old_correct = old_total = 0

#         new_correct = new_total = 0

#         predicted_unseen = 0

#         new_into_old_sum = 0.0

#         old_into_new_sum = 0.0

#         old_new_gap_sum = 0.0

#         old_new_diag_batches = 0



#         for batch in loader:

#             x, y, batch_spectra, _ = self._unpack_hsi_batch(batch)

#             x = x.to(self.device, non_blocking=True).float()

#             y = y.to(self.device, non_blocking=True).long().view(-1)

#             self._assert_labels_in_seen_classes(y, seen_classes, context=f"phase_{phase}_validation")

#             if hasattr(self, "_forward_real_batch"):

#                 out = self._forward_real_batch(x, batch_spectra, classifier_mode=mode, return_energy=True)

#             else:

#                 try:

#                     out = self.model(x, classifier_mode=mode, return_energy=True)

#                 except TypeError:

#                     out = self.model(x)

#             logits_raw = out["logits"] if isinstance(out, dict) else out

#             if logits_raw.dim() != 2 or logits_raw.size(0) != y.numel():

#                 raise RuntimeError(f"Validation logits must be [B,C], got {tuple(logits_raw.shape)} for labels {tuple(y.shape)}")

#             raw_pred = logits_raw.argmax(dim=1)

#             if hasattr(torch, "isin"):

#                 seen_t = torch.as_tensor(seen_classes, device=raw_pred.device, dtype=torch.long)

#                 predicted_unseen += int((~torch.isin(raw_pred, seen_t)).sum().item())

#             logits = self._mask_logits_to_seen_classes(logits_raw, seen_classes)

#             loss = self._stable_ce(logits, y)

#             pred = logits.argmax(dim=1)

#             correct = pred.eq(y)

#             total_loss += float(loss.detach().item())

#             total_correct += int(correct.sum().item())

#             total += int(y.numel())

#             batches += 1

#             if old_class_count > 0:

#                 old_mask = self._label_membership_mask(y, old_classes)

#                 new_mask = self._label_membership_mask(y, new_classes)

#                 if bool(old_mask.any().item()):

#                     old_correct += int(correct[old_mask].sum().item())

#                     old_total += int(old_mask.sum().item())

#                 if bool(new_mask.any().item()):

#                     new_correct += int(correct[new_mask].sum().item())

#                     new_total += int(new_mask.sum().item())

#             if isinstance(out, dict) and torch.is_tensor(out.get("energy", None)) and old_class_count > 0:

#                 energy = out["energy"]

#                 if energy.dim() == 2 and old_classes and new_classes:

#                     old_idx = torch.as_tensor(old_classes, device=energy.device, dtype=torch.long)

#                     new_idx = torch.as_tensor(new_classes, device=energy.device, dtype=torch.long)

#                     if int(old_idx.max().item()) < energy.size(1) and int(new_idx.max().item()) < energy.size(1):

#                         old_min = energy.index_select(1, old_idx).min(dim=1).values

#                         new_min = energy.index_select(1, new_idx).min(dim=1).values

#                         old_mask_e = self._label_membership_mask(y, old_classes)

#                         new_mask_e = self._label_membership_mask(y, new_classes)

#                         if bool(new_mask_e.any().item()):

#                             new_into_old_sum += float((old_min[new_mask_e] < new_min[new_mask_e]).float().mean().detach().item())

#                         if bool(old_mask_e.any().item()):

#                             old_into_new_sum += float((new_min[old_mask_e] < old_min[old_mask_e]).float().mean().detach().item())

#                         old_new_gap_sum += float((new_min - old_min).mean().detach().item())

#                         old_new_diag_batches += 1



#         acc = 100.0 * total_correct / max(total, 1)

#         old_new_split_available = old_class_count > 0 and bool(old_classes) and bool(new_classes)

#         old_acc = 100.0 * old_correct / max(old_total, 1) if old_new_split_available else 0.0

#         new_acc = 100.0 * new_correct / max(new_total, 1) if old_new_split_available else acc

#         hm = (2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-8)) if old_new_split_available else acc

#         return {

#             "loss": total_loss / max(batches, 1),

#             "acc": acc,

#             "old_acc": old_acc,

#             "new_acc": new_acc,

#             "hm": hm,

#             "old_new_split_available": bool(old_new_split_available),

#             "predicted_unseen": float(predicted_unseen),

#             "new_into_old_rate": float(new_into_old_sum / max(old_new_diag_batches, 1)),

#             "old_into_new_rate": float(old_into_new_sum / max(old_new_diag_batches, 1)),

#             "mean_old_new_energy_gap": float(old_new_gap_sum / max(old_new_diag_batches, 1)),

#             "seen_classes": seen_classes,

#             "old_classes": old_classes,

#             "new_classes": new_classes,

#         }



#     # ------------------------------------------------------------------

#     # Phase dispatch

#     # ------------------------------------------------------------------

#     def train_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4):

#         phase = int(phase)

#         self._force_clean_main_path_args()

#         self._propagate_clean_energy_config_to_model()

#         self._assert_global_architecture_contract()

#         if phase == 0:

#             self._set_base_trainable_params()

#             self._set_model_phase_and_old_count(0, 0)

#             self._assert_updated_stack_contract(phase=0)

#             return self.train_base_phase(phase=0, epochs=epochs, batch_size=batch_size, lr=lr)

#         if self.base_only or self.disable_incremental_training:

#             raise RuntimeError("Incremental training is disabled. Set --base_only false and --disable_incremental_training false.")

#         old_class_count = len(self.dataset.get_classes_up_to_phase(phase - 1))

#         self._set_model_phase_and_old_count(phase, old_class_count)

#         self._set_incremental_trainable_params(old_class_count)

#         self._assert_global_architecture_contract()

#         self._assert_incremental_preflight(phase, old_class_count)

#         return self.train_incremental_phase(phase=phase, epochs=epochs, batch_size=batch_size, lr=lr)



#     # ------------------------------------------------------------------

#     # Checkpointing

#     # ------------------------------------------------------------------

#     def save_checkpoint(self, phase, history, evaluator_metrics: Optional[Dict] = None) -> None:

#         phase = int(phase)

#         phase_dir = os.path.join(self.save_dir, f"phase_{phase}")

#         os.makedirs(phase_dir, exist_ok=True)

#         ckpt = {

#             "phase": phase,

#             "base_only": self._as_bool(self.base_only),

#             "incremental_enabled": not self._as_bool(self.disable_incremental_training),

#             "base_objective": "CE+SRPGR",

#             "incremental_objective": (

#                 "SGLAT-HSI: boundary replay + unified geometry-state admission" if not self._adapter_mode_enabled() else

#                 "G²RPA+SGLAT-HSI: GeometryGatedResidualAdapter + boundary replay + unified admission"

#             ),

#             "runtime_contract": self._current_runtime_contract(),

#             "architecture_contract": {

#                 "classifier_mode": "srgp",

#                 "bank": "mean,basis,eigvals,residual_variance,reliability,sample_count,band_signature,spectral_shape",

#                 "old_memory": "frozen GeometryBank statistics only",

#                 "replay": "SGLAT-HSI old-boundary anchors from frozen low-rank geometry; generic replay only as fallback; no fake spectra",

#                 "incremental_update_mode": self._incremental_update_mode(),

#                 "adapter": "geometry_plastic_adapter_only" if self._adapter_mode_enabled() else "none",

#             },

#             "model_state_dict": self.model.state_dict(),

#             "memory_snapshot": self.model.export_memory_snapshot() if hasattr(self.model, "export_memory_snapshot") else None,

#             "current_num_classes": int(getattr(self.model, "current_num_classes", 0)),

#             "old_class_count": int(getattr(self.model, "old_class_count", 0)),

#             "history": history,

#             "base_geometry_certificate": getattr(self, "_last_base_geometry_certificate", getattr(self.model, "base_geometry_certificate", None)),

#             "args": vars(self.args) if hasattr(self.args, "__dict__") else {},

#         }

#         diag = getattr(self, f"_last_phase_{phase}_geometry_diagnostics", None)

#         if diag is None:

#             diag = getattr(self, "_last_base_geometry_diagnostics", None)

#         if diag is not None:

#             ckpt["geometry_diagnostics"] = diag

#         if evaluator_metrics is not None:

#             ckpt["evaluator_metrics"] = evaluator_metrics

#         path = os.path.join(phase_dir, "checkpoint.pth")

#         torch.save(ckpt, path)

#         print(f"[Saved] {path}")

