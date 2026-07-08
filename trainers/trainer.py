from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from trainers.trainer_helpers import TrainerHelper
from trainers.base_phase_trainer import BasePhaseTrainer
from trainers.incremental_phase_trainer import IncrementalPhaseTrainer


STACK_BUILD_ID = "SCTGR-STACK-CONTRACT-2026-07-07-R6"


class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
    STACK_BUILD_ID = STACK_BUILD_ID

    """Strict exemplar-free HSI class-incremental trainer.

    Base phase learns one canonical projected feature space and a replay-ready
    GeometryBank. Incremental phases freeze the encoder, projection, classifier,
    and every old bank row. Old support is reconstructed only through
    spectral-coupled tangent geometry replay; temporary residual parameters refine
    current new rows and are committed after validation.

    The main path contains no feature adapter, teacher/KD, raw exemplar memory,
    prototype classifier, transport, score calibration, or adaptive-boundary head.
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
        """Install defaults for the single SCTGR-RGA architecture."""
        defaults = {
            # Geometry-only classifier.
            "base_classifier_mode": "geometry_only",
            "incremental_classifier_mode": "geometry_only",
            "eval_classifier_mode": "geometry_only",
            "use_logdet_energy": False,
            "logdet_energy_weight": 0.0,
            "logdet_normalize_by_dim": False,
            "center_logdet_energy": False,
            "energy_normalize_by_dim": True,
            "geometry_normalize_logits": False,
            "residual_variance_scale": 1.0,
            "use_reliability_penalty": False,
            "reliability_energy_weight": 0.0,
            "invalid_class_energy": 1e6,

            # Base objective.
            "base_class_balance": True,
            "base_ce_weight": 1.0,
            "base_srpgr_weight": 1.0,
            "base_energy_margin_weight": 0.15,
            "base_energy_margin": 0.25,
            "base_gics_weight": 0.20,
            "base_gics_temperature": 0.07,
            "pgr_weight": 0.10,
            "pgr_compact_weight": 0.15,
            "pgr_center_weight": 0.25,
            "pgr_subspace_weight": 0.20,
            "pgr_band_weight": 0.10,
            "pgr_volume_weight": 0.05,
            "pgr_center_margin": 1.10,
            "pgr_min_class_samples": 3,
            "pgr_subspace_min_samples": 6,
            "pgr_subspace_rank": 3,
            "pgr_max_class_variance": 0.75,
            "pgr_band_overlap_max": 0.60,
            "base_spectral_shape_weight": 0.10,
            "base_require_physical_spectral_shape": False,
            "strict_base_component_coverage": True,

            # Replay-ready GeometryBank.
            "rank_energy_threshold": 0.90,
            "rank_eigen_ratio_threshold": 1e-2,
            "min_active_rank": 1,
            "geometry_variance_shrinkage": 0.25,
            "geom_var_floor": 5e-4,
            "geometry_bank_feature_space": "canonical",
            "spectral_geometry_rank": 5,
            "spectral_rank_energy_threshold": 0.95,
            "spectral_rank_eigen_ratio_threshold": 1e-3,
            "spectral_variance_floor": 1e-6,
            "coupling_ridge": 1e-3,
            "coupling_min_reliability": 0.20,
            "spectral_tangent_clip": 2.5,
            "replay_candidate_multiplier": 4,

            # Incremental architecture and replay policy.
            "incremental_update_mode": "spectral_coupled_geometry_replay",
            "use_spectral_coupled_replay": True,
            "gfa_weight": 1.0,
            "gfa_samples_per_class": 48,
            "gfa_reliability_gated": True,
            "replay_min_per_class": 24,
            "replay_max_per_class": 64,
            "core_replay_ratio": 0.85,
            "directed_replay_min_ratio": 0.10,
            "directed_replay_max_ratio": 0.40,
            "pair_risk_topk": 3,
            "pair_risk_temperature": 0.75,
            "replay_energy_filter": True,
            "joint_old_new_ce_weight": 1.0,
            "geometry_energy_margin_weight": 0.30,
            "geometry_energy_margin": 0.30,
            "old_new_invasion_weight": 0.50,
            "old_new_geometry_margin": 0.35,

            # New-row residual refinement.
            "refine_new_descriptors": True,
            "use_descriptor_refinement": True,
            "descriptor_refine_steps": 20,
            "descriptor_refine_steps_per_epoch": 20,
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

            # Physical spectra guide replay but never classifier logits.
            "use_spectral_geometry": False,
            "spectral_energy_weight": 0.0,
            "band_energy_weight": 0.0,
            "spectral_require_physical_summary": True,
            "spectral_summary_is_physical": False,
            "raw_spectral_summary_is_physical": True,
            "external_spectra_are_physical": True,

            # Validation and checkpointing.
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
        for key, value in defaults.items():
            self._set_arg_default(key, value)

    def _incremental_update_mode(self) -> str:
        raw = str(getattr(self.args, "incremental_update_mode", "spectral_coupled_geometry_replay")).lower().strip()
        aliases = {
            "": "spectral_coupled_geometry_replay",
            "none": "spectral_coupled_geometry_replay",
            "clean": "spectral_coupled_geometry_replay",
            "main": "spectral_coupled_geometry_replay",
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
            raise RuntimeError(
                f"incremental_update_mode={raw!r} selects a removed architecture. "
                "Use spectral_coupled_geometry_replay."
            )
        mode = aliases.get(raw, raw)
        if mode != "spectral_coupled_geometry_replay":
            raise RuntimeError(f"Unsupported incremental_update_mode={raw!r}.")
        setattr(self.args, "incremental_update_mode", mode)
        if hasattr(self.model, "incremental_update_mode"):
            self.model.incremental_update_mode = mode
        return mode

    def _adapter_mode_enabled(self) -> bool:
        """Feature adapters are not part of SCTGR-RGA."""
        return False

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
        }
        forbidden_aliases = {"spectral_geometry", "spectral_residual", "calibrated_geometry", "topology_calibrated_geometry", "base_ce"}
        if raw in forbidden_aliases:
            raise RuntimeError(
                f"{context}: classifier mode {raw!r} is not allowed. The trainer has one inference path: geometry_only."
            )
        normalized = aliases.get(raw, raw)
        if normalized != "geometry_only":
            raise RuntimeError(f"{context}: unsupported classifier mode={raw!r}; use geometry_only.")
        return normalized

    def _force_clean_main_path_args(self) -> None:
        """Force one coherent architecture and disable contradictory branches."""
        mode = self._incremental_update_mode()
        forced = {
            "incremental_update_mode": mode,
            "base_classifier_mode": "geometry_only",
            "incremental_classifier_mode": "geometry_only",
            "eval_classifier_mode": "geometry_only",
            "geometry_normalize_logits": False,
            "energy_normalize_by_dim": True,
            "residual_variance_scale": 1.0,
            "use_logdet_energy": False,
            "logdet_energy_weight": 0.0,
            "logdet_normalize_by_dim": False,
            "center_logdet_energy": False,
            "use_reliability_penalty": False,
            "reliability_energy_weight": 0.0,
            "use_spectral_coupled_replay": True,
            "use_incremental_adapter": False,
            "use_geometry_gated_adapter": False,
            "disable_incremental_adapter": True,
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
        for key, value in forced.items():
            setattr(self.args, key, value)

        self.incremental_update_mode = mode
        self.use_spectral_coupled_replay = True
        self.energy_normalize_by_dim = True
        self.residual_variance_scale = 1.0
        self.use_logdet_energy = False
        self.logdet_energy_weight = 0.0
        self.logdet_normalize_by_dim = False
        self.center_logdet_energy = False
        self.reliability_energy_weight = 0.0
        self.use_geometry_transport = False
        self.use_sglat_transport = False
        self.use_boundary_geometry_replay = False
        self.use_adaptive_boundary = False
        self.use_energy_calibrator = False
        self.use_spectral_geometry = False
        self.spectral_energy_weight = 0.0
        self.band_energy_weight = 0.0

        for attr, value in (
            ("incremental_update_mode", mode),
            ("use_incremental_adapter", False),
            ("use_geometry_gated_adapter", False),
            ("use_geometry_transport", False),
            ("use_sglat_transport", False),
            ("use_geometry_calibrator", False),
            ("use_energy_calibrator", False),
            ("use_adaptive_boundary", False),
            ("use_bicyc_geometry_cycle", False),
        ):
            if hasattr(self.model, attr):
                setattr(self.model, attr, value)
        for name in ("disable_incremental_adapter", "freeze_geometry_calibrator", "freeze_energy_calibrator"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                fn()

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
                "use_reliability_penalty": False,
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

    def _assert_strict_energy_contract(self) -> None:
        """Fail immediately if any component reactivates obsolete scoring terms."""
        errors: List[str] = []
        checks = {
            "energy_normalize_by_dim": bool(self.energy_normalize_by_dim),
            "residual_variance_scale": abs(float(self.residual_variance_scale) - 1.0) <= 1e-8,
            "use_logdet_energy": not bool(self.use_logdet_energy),
            "logdet_energy_weight": abs(float(self.logdet_energy_weight)) <= 1e-12,
            "center_logdet_energy": not bool(self.center_logdet_energy),
            "reliability_energy_weight": abs(float(self.reliability_energy_weight)) <= 1e-12,
        }
        errors.extend(name for name, ok in checks.items() if not ok)

        clf = getattr(self.model, "classifier", None)
        if clf is not None:
            classifier_checks = {
                "classifier.normalize_energy_by_dim": bool(
                    getattr(clf, "normalize_energy_by_dim", getattr(clf, "energy_normalize_by_dim", True))
                ),
                "classifier.residual_variance_scale": abs(
                    float(getattr(clf, "residual_variance_scale", 1.0)) - 1.0
                ) <= 1e-8,
                "classifier.use_logdet_energy": not bool(getattr(clf, "use_logdet_energy", False)),
                "classifier.logdet_energy_weight": abs(
                    float(getattr(clf, "logdet_energy_weight", 0.0))
                ) <= 1e-12,
                "classifier.center_logdet_energy": not bool(
                    getattr(clf, "center_logdet_energy", False)
                ),
                "classifier.reliability_energy_weight": abs(
                    float(getattr(clf, "reliability_energy_weight", 0.0))
                ) <= 1e-12,
            }
            errors.extend(name for name, ok in classifier_checks.items() if not ok)

        if errors:
            raise RuntimeError(
                "Strict SCTGR geometry-energy contract failed: "
                + ", ".join(errors)
                + ". Required: rank/dimension normalized energy, residual scale 1.0, "
                  "no logdet, no centered logdet, and no reliability logit bias."
            )

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
        self._last_incremental_preflight_signature: Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = None

        self._install_clean_runtime_defaults()
        self.incremental_update_mode = self._incremental_update_mode()

        # Core geometry/classifier settings.
        self.subspace_rank = int(getattr(args, "subspace_rank", 5))
        self.geom_var_floor = float(getattr(args, "geom_var_floor", 5e-4))
        self.reliability_energy_weight = 0.0
        self.energy_normalize_by_dim = self._arg_bool("energy_normalize_by_dim", True)
        self.residual_variance_scale = 1.0
        self.invalid_class_energy = float(getattr(args, "invalid_class_energy", 1e6))
        self.use_logdet_energy = False
        self.logdet_energy_weight = 0.0
        self.logdet_normalize_by_dim = self._arg_bool("logdet_normalize_by_dim", True)
        self.center_logdet_energy = False

        # Mandatory base phase.
        self.base_ce_weight = float(getattr(args, "base_ce_weight", 1.0))
        self.base_srpgr_weight = float(getattr(args, "base_srpgr_weight", 1.0))
        self.base_energy_margin_weight = float(getattr(args, "base_energy_margin_weight", 0.15))
        self.base_energy_margin = float(getattr(args, "base_energy_margin", 0.25))
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

        # Incremental Low-Rank Geometry Replay / Residual Geometry Adaptation objective.
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
        self.use_pretrain_incremental_baseline = False

        # Spectral-coupled replay policy.
        self.use_spectral_coupled_replay = self._arg_bool("use_spectral_coupled_replay", True)
        self.replay_min_per_class = int(getattr(args, "replay_min_per_class", 24))
        self.replay_max_per_class = int(getattr(args, "replay_max_per_class", 64))
        self.core_replay_ratio = float(getattr(args, "core_replay_ratio", 0.85))
        self.directed_replay_min_ratio = float(getattr(args, "directed_replay_min_ratio", 0.10))
        self.directed_replay_max_ratio = float(getattr(args, "directed_replay_max_ratio", 0.40))
        self.pair_risk_topk = int(getattr(args, "pair_risk_topk", 3))
        self.pair_risk_temperature = float(getattr(args, "pair_risk_temperature", 0.75))
        self.spectral_tangent_clip = float(getattr(args, "spectral_tangent_clip", 2.5))
        self.replay_candidate_multiplier = int(getattr(args, "replay_candidate_multiplier", 4))
        self.coupling_min_reliability = float(getattr(args, "coupling_min_reliability", 0.20))

        # New descriptor row refinement knobs consumed by IncrementalPhaseTrainer.
        self.refine_new_descriptors = self._arg_bool("refine_new_descriptors", True)
        self.use_descriptor_refinement = self._arg_bool("use_descriptor_refinement", self.refine_new_descriptors)
        self.descriptor_refine_steps = int(getattr(args, "descriptor_refine_steps", 5))
        self.descriptor_refine_lr = float(getattr(args, "descriptor_refine_lr", 1e-3))
        self.descriptor_trust_weight = float(getattr(args, "descriptor_trust_weight", 0.80))
        self.descriptor_refine_max_mean_shift = float(getattr(args, "descriptor_refine_max_mean_shift", 0.30))
        self.descriptor_refine_max_logvar_shift = float(getattr(args, "descriptor_refine_max_logvar_shift", 0.50))
        self.descriptor_refine_steps_per_epoch = int(getattr(args, "descriptor_refine_steps_per_epoch", self.descriptor_refine_steps))
        if self.descriptor_refine_steps_per_epoch <= 0:
            self.descriptor_refine_steps_per_epoch = self.descriptor_refine_steps
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
        self.use_risk_weighted_replay = True
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
        self._assert_strict_energy_contract()
        self._assert_global_architecture_contract()
        self._assert_updated_stack_contract(phase=0)
        self._set_base_trainable_params()

    # ------------------------------------------------------------------
    # Stack contract and trainability
    # ------------------------------------------------------------------
    def _assert_updated_stack_contract(self, phase: Optional[int] = None) -> None:
        """Validate the exact APIs required by the current phase."""
        strict = self._arg_bool("strict_updated_stack", True)
        phase_i = 0 if phase is None else int(phase)
        missing: List[str] = []

        expected_incremental_build = "SCTGR-DIRECT-INVASION-2026-07-07-R5"
        actual_incremental_build = getattr(IncrementalPhaseTrainer, "STACK_BUILD_ID", None)
        if actual_incremental_build != expected_incremental_build:
            missing.append(
                "incremental_phase_trainer build mismatch "
                f"(expected={expected_incremental_build}, actual={actual_incremental_build})"
            )
        if getattr(IncrementalPhaseTrainer, "USES_LEGACY_BOUNDARY_HELPER", None) is not False:
            missing.append("IncrementalPhaseTrainer.USES_LEGACY_BOUNDARY_HELPER must be False")
        if getattr(IncrementalPhaseTrainer, "REFINES_NEW_BASES", None) is not False:
            missing.append("IncrementalPhaseTrainer.REFINES_NEW_BASES must be False")

        for attr in ("extract_projected_features", "compute_logits_from_features", "get_subspace_bank"):
            if not hasattr(self.model, attr):
                missing.append(f"model.{attr}")
        if not hasattr(self.model, "assert_method_identity"):
            missing.append("model.assert_method_identity")

        classifier = getattr(self.model, "classifier", None)
        if classifier is None or not hasattr(classifier, "forward"):
            missing.append("model.classifier")

        bank = getattr(self.model, "geometry_bank", None)
        if bank is None:
            missing.append("model.geometry_bank")
        else:
            for attr in ("get_valid_mask", "assert_bank_valid", "build_candidate_geometry_rows", "commit_candidate_geometry_rows"):
                if not hasattr(bank, attr):
                    missing.append(f"geometry_bank.{attr}")
            if phase_i > 0 and not hasattr(bank, "sample_replay"):
                missing.append("geometry_bank.sample_replay")

        for attr in (
            "_safe_get_subspace_bank", "global_to_seen_local", "assert_bank_ready_for_seen_classes",
            "snapshot_bank_rows", "assert_bank_rows_unchanged",
        ):
            if not hasattr(self, attr):
                missing.append(f"TrainerHelper.{attr}")

        if missing:
            message = "Strict SCTGR-RGA stack contract failed: " + ", ".join(missing)
            if strict:
                raise RuntimeError(message)
            print(f"[WARN] {message}")
        if hasattr(self.model, "assert_method_identity"):
            self.model.assert_method_identity()

    def _assert_global_architecture_contract(self) -> None:
        mode = self._incremental_update_mode()
        if mode != "spectral_coupled_geometry_replay":
            raise RuntimeError(f"Unexpected incremental mode {mode!r}.")
        for key in ("base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode"):
            self._normalize_classifier_mode(getattr(self.args, key, "geometry_only"), context=key)

        forbidden_true = (
            "use_geometry_calibrator", "use_incremental_adapter", "use_geometry_gated_adapter",
            "use_bicyc_geometry_cycle", "allow_incremental_projection_training",
            "use_sglat_transport", "use_geometry_transport", "allow_old_model_transport",
            "use_boundary_geometry_replay", "use_adaptive_boundary", "use_energy_calibrator",
            "geometry_normalize_logits",
        )
        bad = [key for key in forbidden_true if self._arg_bool(key, False)]
        if bad:
            raise RuntimeError(f"Forbidden architecture switches are active: {bad}")
        for key in (
            "bss_weight", "sym_bss_weight", "gdr_weight", "anchor_consistency_weight",
            "geometry_calibration_weight", "energy_calibration_weight",
        ):
            if abs(float(getattr(self.args, key, 0.0))) > 0.0:
                raise RuntimeError(f"{key} must be 0.0 in SCTGR-RGA.")
        if not self._arg_bool("use_spectral_coupled_replay", True):
            raise RuntimeError("use_spectral_coupled_replay must be true.")
        self._assert_strict_energy_contract()
        if hasattr(self.model, "assert_method_identity"):
            self.model.assert_method_identity()

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

    def _set_incremental_trainable_params(
        self,
        old_classes: Iterable[int] | int = 0,
        new_classes: Optional[Iterable[int]] = None,
    ) -> List[torch.nn.Parameter]:
        """Freeze every model parameter; refinement uses temporary descriptor tensors."""
        self._force_clean_main_path_args()
        self._propagate_clean_energy_config_to_model()
        if isinstance(old_classes, int):
            old_ids = list(range(max(int(old_classes), 0)))
        else:
            old_ids = [int(c) for c in old_classes]
        new_ids = [int(c) for c in (new_classes or [])]
        for _, parameter in self.model.named_parameters():
            parameter.requires_grad = False
        if hasattr(self.model, "set_incremental_mode"):
            self.model.set_incremental_mode(
                phase=int(getattr(self.model, "current_phase", 1)),
                old_classes=old_ids,
                old_class_count=len(old_ids),
                train_classifier_calibration=False,
                train_geometry_adapter=False,
            )
        for name in (
            "freeze_backbone_only", "freeze_projection_head", "freeze_classifier",
            "freeze_energy_calibrator", "freeze_geometry_calibrator", "disable_incremental_adapter",
        ):
            fn = getattr(self.model, name, None)
            if callable(fn):
                fn()
        bad = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if bad:
            raise RuntimeError(f"No model parameters may be trainable incrementally: {bad[:30]}")
        return []

    def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
        return self._set_incremental_trainable_params(int(old_class_count), [])

    def _has_feature_plasticity(self) -> bool:
        return False

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
        """Set phase state before training or after committing a phase.

        Before phase ``t`` the old set is classes through ``t-1``. After phase
        ``t`` is committed, callers promote all classes through ``t`` to the old
        set for the next phase. Both transitions are valid and explicit.
        """
        phase = int(phase)
        old_class_count = int(old_class_count)
        if phase == 0:
            if old_class_count != 0:
                raise RuntimeError("Base phase old_class_count must be zero.")
            if hasattr(self.model, "set_base_mode"):
                self.model.set_base_mode(train_backbone=True, train_projection=True)
            self.model.current_phase = 0
            self.model.old_class_count = 0
            return

        before_ids: List[int] = []
        through_ids: List[int] = []
        if hasattr(self.dataset, "get_classes_up_to_phase"):
            try:
                before_ids = [int(c) for c in self.dataset.get_classes_up_to_phase(phase - 1)]
                through_ids = [int(c) for c in self.dataset.get_classes_up_to_phase(phase)]
            except Exception:
                before_ids, through_ids = [], []
        if not before_ids:
            before_ids = self._seen_classes_for_phase(phase - 1)
        if not through_ids:
            through_ids = self._seen_classes_for_phase(phase)

        if old_class_count == len(before_ids):
            old_ids = before_ids
        elif old_class_count == len(through_ids):
            old_ids = through_ids
        else:
            raise RuntimeError(
                f"old_class_count={old_class_count} matches neither pre-phase ({len(before_ids)}) "
                f"nor post-phase ({len(through_ids)}) state for phase {phase}."
            )
        if hasattr(self.model, "set_incremental_mode"):
            self.model.set_incremental_mode(
                phase=phase,
                old_class_count=len(old_ids),
                old_classes=old_ids,
                train_classifier_calibration=False,
                train_geometry_adapter=False,
            )
        self.model.current_phase = phase
        self.model.old_class_count = len(old_ids)

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
        """Validate and return strict seen-local logits.

        The cleaned classifier has one output contract: [B, len(seen_classes)].
        Full-global logits are not accepted because silently slicing them hides
        class-order bugs during incremental learning.
        """
        if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError(f"logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")
        seen = [int(c) for c in seen_classes]
        if logits.size(1) != len(seen):
            raise RuntimeError(
                f"Strict classifier contract violation: logits width={logits.size(1)} but len(seen_classes)={len(seen)}. "
                "Return seen-local logits from the model/classifier instead of global-full logits."
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError("Strict classifier contract violation: logits contain NaN/Inf.")
        return logits

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
                if energy.size(1) != len(seen_classes):
                    raise RuntimeError(
                        f"Strict energy contract violation: energy width={energy.size(1)} but len(seen_classes)={len(seen_classes)}."
                    )
                e_seen = energy
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
        """Use the base trainer's cleaned geometry-state aliases.

        Keeping a second local implementation here would override the stricter
        base-phase scoring logic and lose the base geometry-energy margin fields.
        """
        return BasePhaseTrainer._base_geometry_global_metrics(self)

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
        """Delegate to BasePhaseTrainer so base energy-margin health affects checkpoint selection."""
        return BasePhaseTrainer._select_base_checkpoint_score(self, val_stats, geom_stats)

    def _capture_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _print_trainable_summary(self, phase: int) -> None:
        trainable = [(name, int(parameter.numel())) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        total = sum(count for _, count in trainable)
        if int(phase) == 0:
            print(f"[Trainable] Base phase: {total:,} params | backbone/projection + temporary CE head")
            print("[Base Objective] balanced CE + GICS + geometry reserve + base geometry-energy margin")
        else:
            if total != 0:
                raise RuntimeError(f"Incremental model parameters must be frozen, found {total:,} trainable.")
            print(f"[Trainable] Incremental phase {phase}: 0 model params | temporary new-row residuals only")
            print("[Incremental Objective] spectral-coupled core/directed replay + class-balanced CE + energy margin + bidirectional invasion + trust")
        if self.debug:
            for name, count in trainable[:150]:
                print(f"  {name}: {count:,}")

    def _current_runtime_contract(self) -> Dict[str, object]:
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        return {
            "method": "Spectral-Coupled Tangent Geometry Replay and New-Row Residual Adaptation for HSI NECIL",
            "feature_space": "one frozen canonical projected z-space",
            "classifier": "strict seen-local low-rank GeometryBank energy",
            "classifier_output": "[B, len(seen_classes)]",
            "incremental_update_mode": self._incremental_update_mode(),
            "model_trainable_parameters": trainable,
            "descriptor_plasticity": "temporary new mean/eigenvalue/residual-variance residuals",
            "old_memory": "frozen feature geometry + compact spectral tangent/coupling statistics",
            "replay": "spectral-consistent core + risk-directed spectral tangent replay",
            "shell_replay": False,
            "raw_exemplars": False,
            "kd_teacher": False,
            "feature_adapter": False,
            "transport": False,
            "adaptive_boundary": False,
            "energy_calibrator": False,
            "spectral_classifier_branch": False,
            "uses_logdet_energy": bool(self.use_logdet_energy),
            "logdet_energy_weight": float(self.logdet_energy_weight),
        }

    def _assert_incremental_preflight(
        self,
        phase: int,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        seen_classes: Optional[Iterable[int]] = None,
    ) -> None:
        phase = int(phase)
        if phase <= 0:
            raise RuntimeError("Incremental preflight requires phase > 0.")
        self._assert_updated_stack_contract(phase=phase)
        if old_classes is None or new_classes is None or seen_classes is None:
            old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
        old_ids = [int(c) for c in old_classes]
        new_ids = [int(c) for c in new_classes]
        seen_ids = [int(c) for c in seen_classes]
        signature = (phase, tuple(old_ids), tuple(new_ids))
        if self._last_incremental_preflight_signature == signature:
            return
        if not old_ids or not new_ids:
            raise RuntimeError(f"Invalid phase split: old={old_ids}, new={new_ids}")
        if set(old_ids).intersection(new_ids):
            raise RuntimeError(f"Old/new overlap: {sorted(set(old_ids).intersection(new_ids))}")
        if seen_ids != list(dict.fromkeys([*old_ids, *new_ids])):
            raise RuntimeError("seen_classes must equal old_classes + new_classes in order.")
        self.assert_clean_incremental_contract(
            phase=phase, old_classes=old_ids, new_classes=new_ids, seen_classes=seen_ids
        )
        bank = self._safe_get_subspace_bank(require_ready=True)
        self.assert_bank_ready_for_seen_classes(bank, old_ids)
        self.assert_bank_has_only_allowed_valid_rows(bank, old_ids)
        gb = getattr(self.model, "geometry_bank", None)
        if gb is None or not hasattr(gb, "sample_replay"):
            raise RuntimeError("Updated GeometryBank.sample_replay is required.")
        if hasattr(gb, "freeze_classes"):
            gb.freeze_classes(old_ids)
        if hasattr(gb, "assert_bank_valid"):
            gb.assert_bank_valid(seen_classes=old_ids, strict=True)
        self._set_incremental_trainable_params(old_ids, new_ids)
        self._last_incremental_preflight_signature = signature
        print(
            f"[Incremental Preflight] phase={phase} | old={old_ids} | new={new_ids} | "
            f"seen={seen_ids} | model_frozen=True | replay=spectral_coupled"
        )

    def train_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4):
        """Dispatch a phase under the single SCTGR-RGA contract."""
        phase = int(phase)
        self.early_stop_patience = 0
        for key in ("early_stop_patience", "base_early_stop_patience", "incremental_early_stop_patience"):
            setattr(self.args, key, 0)
        self._force_clean_main_path_args()
        self._propagate_clean_energy_config_to_model()
        self._assert_strict_energy_contract()
        self._assert_global_architecture_contract()

        if phase == 0:
            self._set_model_phase_and_old_count(0, 0)
            self._set_base_trainable_params()
            self._assert_updated_stack_contract(phase=0)
            self._print_trainable_summary(phase=0)
            return self.train_base_phase(phase=0, epochs=epochs, batch_size=batch_size, lr=lr)

        if self.base_only or self.disable_incremental_training:
            raise RuntimeError("Incremental training is disabled by configuration.")
        old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
        self._set_model_phase_and_old_count(phase, len(old_classes))
        self._assert_incremental_preflight(
            phase, old_classes=old_classes, new_classes=new_classes, seen_classes=seen_classes
        )
        self._print_trainable_summary(phase=phase)
        return self.train_incremental_phase(
            phase=phase, epochs=epochs, batch_size=batch_size, lr=lr
        )

    def _adaptive_boundary_loss_from_current_bank(self, *args, **kwargs):
        """Adaptive-boundary remains disabled through IncrementalPhaseTrainer."""
        return IncrementalPhaseTrainer._adaptive_boundary_loss_from_current_bank(self, *args, **kwargs)

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
            "base_objective": "balanced CE + GICS + geometry reserve + base geometry-energy margin",
            "incremental_objective": "spectral-coupled core/directed replay + class-balanced CE + energy margin + bidirectional invasion + descriptor trust",
            "runtime_contract": self._current_runtime_contract(),
            "architecture_contract": {
                "method": "Spectral-Coupled Tangent Geometry Replay and New-Row Residual Adaptation for HSI NECIL",
                "classifier_mode": "geometry_only",
                "bank": "feature geometry + physical spectral tangent + spectral-to-feature coupling + reliability/quantiles",
                "old_memory": "frozen GeometryBank statistics only",
                "replay": "spectral-consistent core + risk-directed spectral tangent replay; no raw old patches/features",
                "incremental_update_mode": self._incremental_update_mode(),
                "residual_adaptation": "temporary new-row mean/eigenvalue/residual-variance residuals only",
                "forbidden_components": ["KD", "teacher", "prototypes", "feature_adapter", "shell_replay", "transport", "adaptive_boundary", "measured_calibration", "BSS/GDR"],
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
#     from losses.loss import unified_spectral_geometry_loss
# except Exception:  # pragma: no cover
#     unified_spectral_geometry_loss = None


# class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
#     """Trainer for strict non-exemplar HSI class-incremental learning with Low-Rank Geometry Replay and Residual Geometry Adaptation.

#     Active architecture:
#         Base phase:
#             Balanced CE + GICS + geometry reserve + base geometry-energy margin
#             in canonical projected z-space.

#         Incremental phase:
#             Frozen old GeometryBank + new rows + synthetic GeometryBank replay
#             + bounded residual geometry adaptation + joint old/new CE
#             + geometry energy margins.

#     Deliberately inactive in the main path:
#         KD/teacher, prototypes, raw exemplars, transport, adaptive boundary,
#         measured energy calibration, BiCyc/BSS/GDR, prompt/token memory.
#     """

#     # ------------------------------------------------------------------
#     # Small config helpers
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _as_bool(value: Any, default: bool = False) -> bool:
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
#         if not hasattr(self.args, name) or getattr(self.args, name) is None:
#             try:
#                 setattr(self.args, name, value)
#             except Exception:
#                 pass

#     @staticmethod
#     def _freeze_module_if_present(module: Optional[torch.nn.Module]) -> None:
#         if module is None:
#             return
#         try:
#             for p in module.parameters():
#                 p.requires_grad = False
#         except Exception:
#             pass
#         try:
#             module.eval()
#         except Exception:
#             pass

#     # ------------------------------------------------------------------
#     # Runtime defaults and architecture normalization
#     # ------------------------------------------------------------------
#     def _install_clean_runtime_defaults(self) -> None:
#         """Install only defaults needed by the Low-Rank Geometry Replay / Residual Geometry Adaptation stack.

#         Defaults are set only when the user/main.py did not provide a value.
#         Forbidden legacy branches are then force-disabled in
#         _force_clean_main_path_args().
#         """
#         defaults = {
#             # Classifier energy contract.
#             "base_classifier_mode": "geometry_only",
#             "incremental_classifier_mode": "geometry_only",
#             "eval_classifier_mode": "geometry_only",
#             "use_logdet_energy": True,
#             "logdet_energy_weight": 0.05,
#             "logdet_normalize_by_dim": True,
#             "center_logdet_energy": True,
#             "energy_normalize_by_dim": True,
#             "geometry_normalize_logits": False,
#             "residual_variance_scale": 0.75,
#             "reliability_energy_weight": 0.03,
#             "invalid_class_energy": 1e6,

#             # Mandatory base objective.
#             "base_class_balance": True,
#             "base_ce_weight": 1.0,
#             "base_srpgr_weight": 1.0,
#             "base_energy_margin_weight": 0.15,
#             "base_energy_margin": 0.25,
#             "base_gics_weight": 0.20,
#             "base_gics_temperature": 0.07,
#             "pgr_weight": 0.10,
#             "pgr_compact_weight": 0.15,
#             "pgr_center_weight": 0.25,
#             "pgr_subspace_weight": 0.20,
#             "pgr_band_weight": 0.10,
#             "pgr_volume_weight": 0.05,
#             "pgr_center_margin": 1.10,
#             "pgr_min_class_samples": 3,
#             "pgr_subspace_min_samples": 6,
#             "pgr_subspace_rank": 3,
#             "pgr_max_class_variance": 0.75,
#             "pgr_band_overlap_max": 0.60,
#             "base_spectral_shape_weight": 0.10,
#             "base_require_physical_spectral_shape": False,
#             "strict_base_component_coverage": True,

#             # GeometryBank extraction.
#             "rank_energy_threshold": 0.90,
#             "rank_eigen_ratio_threshold": 1e-2,
#             "min_active_rank": 1,
#             "geometry_variance_shrinkage": 0.25,
#             "geom_var_floor": 5e-4,
#             "geometry_bank_feature_space": "canonical",

#             # Incremental Low-Rank Geometry Replay + Residual Geometry Adaptation path.
#             "incremental_update_mode": "geometry_gated_adapter",
#             "gfa_weight": 1.0,
#             "gfa_samples_per_class": 48,
#             "gfa_parallel_scale": 1.0,
#             "gfa_residual_scale": 0.25,
#             "gfa_reliability_gated": True,
#             "joint_old_new_ce_weight": 1.0,
#             "geometry_energy_margin_weight": 0.30,
#             "geometry_energy_margin": 0.30,
#             "old_new_invasion_weight": 0.50,
#             "old_new_geometry_margin": 0.35,
#             "adapter_lr": 1e-4,
#             "adapter_weight_decay": 0.0,
#             "adapter_bottleneck": 32,
#             "adapter_max_scale": 0.35,
#             "adapter_dropout": 0.0,
#             "adapter_gate_bias_init": -3.0,
#             "residual_geometry_adapter_weight": 1.0,
#             "g2rpa_adapter_weight": 1.0,  # compatibility alias consumed by older incremental trainer code
#             "adapter_old_delta_weight": 1.0,
#             "adapter_old_gate_weight": 0.75,
#             "adapter_old_energy_weight": 0.25,
#             "adapter_old_margin_weight": 0.25,
#             "adapter_delta_weight": 0.10,
#             "adapter_new_gate_weight": 0.05,
#             "adapter_new_gate_target": 0.25,
#             "adapter_new_gate_max_target": 0.75,

#             # Descriptor/new-row refinement kept for incremental row quality.
#             "refine_new_descriptors": True,
#             "use_descriptor_refinement": True,
#             "descriptor_refine_steps": 5,
#             "descriptor_refine_lr": 1e-3,
#             "descriptor_trust_weight": 0.80,
#             "descriptor_refine_max_mean_shift": 0.30,
#             "descriptor_refine_max_logvar_shift": 0.50,
#             "descriptor_subspace_collision_weight": 0.10,
#             "descriptor_subspace_overlap_max": 0.35,
#             "descriptor_center_margin_weight": 0.05,
#             "descriptor_center_collision_weight": 0.05,
#             "descriptor_center_margin": 0.50,
#             "descriptor_volume_weight": 0.03,
#             "descriptor_volume_control_weight": 0.03,
#             "descriptor_volume_margin": 0.0,

#             # Spectral summaries can shape base loss/bank but classifier remains geometry-only.
#             "use_spectral_geometry": False,
#             "spectral_energy_weight": 0.0,
#             "band_energy_weight": 0.0,
#             "spectral_require_physical_summary": True,
#             "spectral_summary_is_physical": False,
#             "raw_spectral_summary_is_physical": True,
#             "external_spectra_are_physical": True,

#             # Checkpoint/validation.
#             "best_state_metric": "geometry_score",
#             "refresh_before_validation": True,
#             "validation_refresh_every": 1,
#             "enforce_base_geometry_certificate": False,
#             "base_cert_min_geom_acc": 95.0,
#             "base_cert_min_reliability": 0.15,
#             "base_cert_min_mean_reliability": 0.35,
#             "base_cert_max_subspace_overlap": 0.55,
#             "base_cert_max_geometry_conflict": 1.35,
#             "base_cert_max_band_similarity": 0.90,
#             "strict_updated_stack": True,
#         }
#         for k, v in defaults.items():
#             self._set_arg_default(k, v)

#     def _incremental_update_mode(self) -> str:
#         raw = str(getattr(self.args, "incremental_update_mode", "geometry_gated_adapter")).lower().strip()
#         aliases = {
#             "": "geometry_gated_adapter",
#             "none": "geometry_gated_adapter",
#             "clean": "geometry_gated_adapter",
#             "residual_geometry_adaptation": "geometry_gated_adapter",
#             # Backward-compatible CLI aliases are accepted, but they all resolve to the single residual adaptation path.
#             "adapter": "geometry_gated_adapter",
#             "gated_adapter": "geometry_gated_adapter",
#             "geometry_adapter": "geometry_gated_adapter",
#             "geometry_gated_adapter": "geometry_gated_adapter",
#             "descriptor": "geometry_gated_adapter",
#             "descriptor_only": "geometry_gated_adapter",
#         }
#         mode = aliases.get(raw, raw)
#         if mode != "geometry_gated_adapter":
#             raise RuntimeError(
#                 f"Unsupported incremental_update_mode={raw!r}. The trainer has one incremental path: "
#                 "Low-Rank Geometry Replay + Residual Geometry Adaptation."
#             )
#         try:
#             setattr(self.args, "incremental_update_mode", mode)
#         except Exception:
#             pass
#         if hasattr(self.model, "incremental_update_mode"):
#             self.model.incremental_update_mode = mode
#         return mode

#     def _adapter_mode_enabled(self) -> bool:
#         return self._incremental_update_mode() == "geometry_gated_adapter"

#     def _normalize_classifier_mode(self, mode: Optional[str], *, context: str = "runtime") -> str:
#         raw = str(mode or "geometry_only").lower().strip()
#         aliases = {
#             "": "geometry_only",
#             "none": "geometry_only",
#             "geo": "geometry_only",
#             "geometry": "geometry_only",
#             "geometry-only": "geometry_only",
#             "feature_geometry": "geometry_only",
#             "low_rank_geometry": "geometry_only",
#             "replay": "geometry_only",
#             "synthetic_replay": "geometry_only",
#             "srgp": "geometry_only",
#             "srgp_geometry": "geometry_only",
#         }
#         forbidden_aliases = {"spectral_geometry", "spectral_residual", "calibrated_geometry", "topology_calibrated_geometry", "base_ce"}
#         if raw in forbidden_aliases:
#             raise RuntimeError(
#                 f"{context}: classifier mode {raw!r} is not allowed. The trainer has one inference path: geometry_only."
#             )
#         normalized = aliases.get(raw, raw)
#         if normalized != "geometry_only":
#             raise RuntimeError(f"{context}: unsupported classifier mode={raw!r}; use geometry_only.")
#         return normalized

#     def _force_clean_main_path_args(self) -> None:
#         """Disable unused/forbidden branches and align public args."""
#         mode = self._incremental_update_mode()
#         forced = {
#             "incremental_update_mode": mode,
#             "base_classifier_mode": "geometry_only",
#             "incremental_classifier_mode": "geometry_only",
#             "eval_classifier_mode": "geometry_only",
#             "geometry_normalize_logits": False,
#             "use_incremental_adapter": False,   # legacy flag; Residual Geometry Adaptation uses geometry_plastic_adapter as the bounded residual module.
#             "disable_incremental_adapter": False,
#             "incremental_adapter_normalize": False,
#             "allow_incremental_projection_training": False,
#             "freeze_projection_during_incremental": True,
#             "use_geometry_calibrator": False,
#             "geometry_calibration_weight": 0.0,
#             "use_energy_calibrator": False,
#             "energy_calibration_weight": 0.0,
#             "use_bicyc_geometry_cycle": False,
#             "bicyc_geometry_cycle_weight": 0.0,
#             "bicyc_cycle_weight": 0.0,
#             "bss_weight": 0.0,
#             "sym_bss_weight": 0.0,
#             "gdr_weight": 0.0,
#             "anchor_consistency_weight": 0.0,
#             "use_sglat_transport": False,
#             "use_geometry_transport": False,
#             "allow_old_model_transport": False,
#             "allow_transport_without_adapter": False,
#             "use_boundary_geometry_replay": False,
#             "use_adaptive_boundary": False,
#             "use_boundary_projection": False,
#             "boundary_preserve_weight": 0.0,
#             "use_spectral_geometry": False,
#             "spectral_energy_weight": 0.0,
#             "band_energy_weight": 0.0,
#             "early_stop_patience": 0,
#             "base_early_stop_patience": 0,
#             "incremental_early_stop_patience": 0,
#         }
#         for k, v in forced.items():
#             try:
#                 setattr(self.args, k, v)
#             except Exception:
#                 pass

#         self.incremental_update_mode = mode
#         self.use_geometry_transport = False
#         self.use_sglat_transport = False
#         self.use_boundary_geometry_replay = False
#         self.use_adaptive_boundary = False
#         self.use_energy_calibrator = False
#         self.use_spectral_geometry = False
#         self.spectral_energy_weight = 0.0
#         self.band_energy_weight = 0.0

#         if hasattr(self.model, "use_geometry_gated_adapter"):
#             self.model.use_geometry_gated_adapter = bool(mode == "geometry_gated_adapter")
#         if hasattr(self.model, "incremental_update_mode"):
#             self.model.incremental_update_mode = mode
#         for attr in ("use_geometry_transport", "use_sglat_transport", "use_geometry_calibrator", "use_bicyc_geometry_cycle", "use_incremental_adapter"):
#             if hasattr(self.model, attr):
#                 setattr(self.model, attr, False)
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()

#     def _propagate_clean_energy_config_to_model(self) -> None:
#         clf = getattr(self.model, "classifier", None)
#         if clf is not None:
#             pairs = {
#                 "use_logdet_energy": bool(self.use_logdet_energy),
#                 "logdet_energy_weight": float(self.logdet_energy_weight),
#                 "logdet_normalize_by_dim": bool(self.logdet_normalize_by_dim),
#                 "center_logdet_energy": bool(self.center_logdet_energy),
#                 "energy_normalize_by_dim": bool(self.energy_normalize_by_dim),
#                 "normalize_energy_by_dim": bool(self.energy_normalize_by_dim),
#                 "reliability_energy_weight": float(self.reliability_energy_weight),
#                 "residual_variance_scale": float(self.residual_variance_scale),
#                 "invalid_class_energy": float(self.invalid_class_energy),
#                 "use_spectral_geometry": False,
#                 "spectral_energy_weight": 0.0,
#                 "band_energy_weight": 0.0,
#                 "use_adaptive_boundary": False,
#             }
#             for k, v in pairs.items():
#                 if hasattr(clf, k):
#                     try:
#                         setattr(clf, k, v)
#                     except Exception:
#                         pass
#             if hasattr(clf, "normalize_logits"):
#                 clf.normalize_logits = False
#             if hasattr(clf, "freeze_all_adaptation"):
#                 clf.freeze_all_adaptation()
#         if hasattr(self.model, "use_adaptive_boundary"):
#             self.model.use_adaptive_boundary = False

#     # ------------------------------------------------------------------
#     # Construction
#     # ------------------------------------------------------------------
#     def __init__(self, model, dataset, args) -> None:
#         self.args = args
#         self.device = torch.device(getattr(args, "device", "cpu"))
#         self.model = model.to(self.device)
#         self.dataset = dataset
#         self.save_dir = str(getattr(args, "save_dir", "./checkpoints"))
#         os.makedirs(self.save_dir, exist_ok=True)

#         self.debug = self._arg_bool("debug_verbose", False) or os.environ.get("NECIL_DEBUG", "0") == "1"
#         self.base_only = self._arg_bool("base_only", False)
#         self.disable_incremental_training = self._arg_bool("disable_incremental_training", False)

#         self._install_clean_runtime_defaults()
#         self.incremental_update_mode = self._incremental_update_mode()

#         # Core geometry/classifier settings.
#         self.subspace_rank = int(getattr(args, "subspace_rank", 5))
#         self.geom_var_floor = float(getattr(args, "geom_var_floor", 5e-4))
#         self.reliability_energy_weight = float(getattr(args, "reliability_energy_weight", 0.03))
#         self.energy_normalize_by_dim = self._arg_bool("energy_normalize_by_dim", True)
#         self.residual_variance_scale = float(getattr(args, "residual_variance_scale", 0.75))
#         self.invalid_class_energy = float(getattr(args, "invalid_class_energy", 1e6))
#         self.use_logdet_energy = self._arg_bool("use_logdet_energy", True)
#         self.logdet_energy_weight = float(getattr(args, "logdet_energy_weight", 0.05))
#         self.logdet_normalize_by_dim = self._arg_bool("logdet_normalize_by_dim", True)
#         self.center_logdet_energy = self._arg_bool("center_logdet_energy", True)

#         # Mandatory base phase.
#         self.base_ce_weight = float(getattr(args, "base_ce_weight", 1.0))
#         self.base_srpgr_weight = float(getattr(args, "base_srpgr_weight", 1.0))
#         self.base_energy_margin_weight = float(getattr(args, "base_energy_margin_weight", 0.15))
#         self.base_energy_margin = float(getattr(args, "base_energy_margin", 0.25))
#         self.base_gics_weight = float(getattr(args, "base_gics_weight", 0.20))
#         self.pgr_weight = float(getattr(args, "pgr_weight", 0.10))
#         self.pgr_compact_weight = float(getattr(args, "pgr_compact_weight", 0.15))
#         self.pgr_center_weight = float(getattr(args, "pgr_center_weight", 0.25))
#         self.pgr_subspace_weight = float(getattr(args, "pgr_subspace_weight", 0.15))
#         self.pgr_band_weight = float(getattr(args, "pgr_band_weight", 0.05))
#         self.pgr_volume_weight = float(getattr(args, "pgr_volume_weight", 0.05))
#         self.pgr_center_margin = float(getattr(args, "pgr_center_margin", 1.10))
#         self.pgr_min_class_samples = int(getattr(args, "pgr_min_class_samples", 3))
#         self.pgr_subspace_min_samples = int(getattr(args, "pgr_subspace_min_samples", 6))
#         self.pgr_subspace_rank = int(getattr(args, "pgr_subspace_rank", 3))
#         self.pgr_max_class_variance = float(getattr(args, "pgr_max_class_variance", 0.75))
#         self.pgr_band_overlap_max = float(getattr(args, "pgr_band_overlap_max", 0.65))
#         self.pgr_normalize_features = self._arg_bool("pgr_normalize_features", True)

#         # GeometryBank extraction/rank.
#         self.rank_energy_threshold = float(getattr(args, "rank_energy_threshold", 0.90))
#         self.rank_eigen_ratio_threshold = float(getattr(args, "rank_eigen_ratio_threshold", 1e-2))
#         self.min_active_rank = int(getattr(args, "min_active_rank", 1))
#         self.geometry_variance_shrinkage = float(getattr(args, "geometry_variance_shrinkage", 0.25))

#         # Optimizer/checkpoint policy.
#         self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))
#         self.ce_logit_clip = float(getattr(args, "ce_logit_clip", 50.0))
#         self.grad_clip_base = float(getattr(args, "grad_clip_base", 1.0))
#         self.grad_clip_inc = float(getattr(args, "grad_clip_inc", 0.5))
#         self.refresh_before_validation = self._arg_bool("refresh_before_validation", True)
#         self.validation_refresh_every = int(getattr(args, "validation_refresh_every", 1))
#         self.bank_refresh_every = int(getattr(args, "bank_refresh_every", 0))
#         self.best_state_metric = str(getattr(args, "best_state_metric", "geometry_score")).lower().strip() or "geometry_score"
#         self.early_stop_patience = 0

#         # Incremental Low-Rank Geometry Replay / Residual Geometry Adaptation objective.
#         self.gfa_weight = float(getattr(args, "gfa_weight", 1.0))
#         self.gfa_samples_per_class = int(getattr(args, "gfa_samples_per_class", 48))
#         self.gfa_parallel_scale = float(getattr(args, "gfa_parallel_scale", 0.95))
#         self.gfa_residual_scale = float(getattr(args, "gfa_residual_scale", 0.25))
#         self.gfa_reliability_gated = self._arg_bool("gfa_reliability_gated", True)
#         self.joint_old_new_ce_weight = float(getattr(args, "joint_old_new_ce_weight", 1.0))
#         self.geometry_energy_margin_weight = float(getattr(args, "geometry_energy_margin_weight", 0.30))
#         self.geometry_energy_margin = float(getattr(args, "geometry_energy_margin", 0.30))
#         self.old_new_invasion_weight = float(getattr(args, "old_new_invasion_weight", 0.50))
#         self.old_new_geometry_margin = float(getattr(args, "old_new_geometry_margin", 0.35))
#         self.use_pretrain_incremental_baseline = self._arg_bool("use_pretrain_incremental_baseline", True)

#         # Adapter controls.
#         self.adapter_lr = float(getattr(args, "adapter_lr", 1e-4))
#         self.adapter_weight_decay = float(getattr(args, "adapter_weight_decay", 0.0))
#         self.residual_geometry_adapter_weight = float(getattr(args, "residual_geometry_adapter_weight", getattr(args, "g2rpa_adapter_weight", 1.0)))
#         self.g2rpa_adapter_weight = self.residual_geometry_adapter_weight  # compatibility alias for older incremental trainer code
#         self.adapter_old_delta_weight = float(getattr(args, "adapter_old_delta_weight", 1.0))
#         self.adapter_old_gate_weight = float(getattr(args, "adapter_old_gate_weight", 0.75))
#         self.adapter_old_energy_weight = float(getattr(args, "adapter_old_energy_weight", 0.25))
#         self.adapter_old_margin_weight = float(getattr(args, "adapter_old_margin_weight", 0.25))
#         self.adapter_delta_weight = float(getattr(args, "adapter_delta_weight", 0.10))
#         self.adapter_new_gate_weight = float(getattr(args, "adapter_new_gate_weight", 0.05))
#         self.adapter_new_gate_target = float(getattr(args, "adapter_new_gate_target", 0.25))
#         self.adapter_new_gate_max_target = float(getattr(args, "adapter_new_gate_max_target", 0.75))

#         # New descriptor row refinement knobs consumed by IncrementalPhaseTrainer.
#         self.refine_new_descriptors = self._arg_bool("refine_new_descriptors", True)
#         self.use_descriptor_refinement = self._arg_bool("use_descriptor_refinement", self.refine_new_descriptors)
#         self.descriptor_refine_steps = int(getattr(args, "descriptor_refine_steps", 5))
#         self.descriptor_refine_lr = float(getattr(args, "descriptor_refine_lr", 1e-3))
#         self.descriptor_trust_weight = float(getattr(args, "descriptor_trust_weight", 0.80))
#         self.descriptor_refine_max_mean_shift = float(getattr(args, "descriptor_refine_max_mean_shift", 0.30))
#         self.descriptor_refine_max_logvar_shift = float(getattr(args, "descriptor_refine_max_logvar_shift", 0.50))
#         self.descriptor_refine_steps_per_epoch = int(getattr(args, "descriptor_refine_steps_per_epoch", 0))
#         self.descriptor_refine_grad_clip = float(getattr(args, "descriptor_refine_grad_clip", 1.0))
#         self.descriptor_subspace_collision_weight = float(getattr(args, "descriptor_subspace_collision_weight", 0.10))
#         self.descriptor_subspace_overlap_max = float(getattr(args, "descriptor_subspace_overlap_max", 0.35))
#         self.descriptor_center_margin_weight = float(getattr(args, "descriptor_center_margin_weight", 0.05))
#         self.descriptor_center_collision_weight = float(getattr(args, "descriptor_center_collision_weight", self.descriptor_center_margin_weight))
#         self.descriptor_center_margin = float(getattr(args, "descriptor_center_margin", 0.50))
#         self.descriptor_volume_weight = float(getattr(args, "descriptor_volume_weight", 0.03))
#         self.descriptor_volume_control_weight = float(getattr(args, "descriptor_volume_control_weight", self.descriptor_volume_weight))
#         self.descriptor_volume_margin = float(getattr(args, "descriptor_volume_margin", 0.0))

#         # Explicitly inactive legacy branches. Attributes remain for compatibility only.
#         self.use_boundary_geometry_replay = False
#         self.use_risk_weighted_replay = False
#         self.use_energy_calibrator = False
#         self.use_adaptive_boundary = False
#         self.use_geometry_transport = False
#         self.use_sglat_transport = False
#         self.allow_incremental_projection_training = False
#         self.freeze_projection_during_incremental = True
#         self.use_spectral_geometry = False
#         self.spectral_energy_weight = 0.0
#         self.band_energy_weight = 0.0
#         self.bss_weight = 0.0
#         self.sym_bss_weight = 0.0
#         self.gdr_weight = 0.0
#         self.anchor_consistency_weight = 0.0
#         self.geometry_calibration_weight = 0.0
#         self.energy_calibration_weight = 0.0
#         self.boundary_preserve_weight = 0.0
#         self.use_boundary_projection = False

#         self._force_clean_main_path_args()
#         self._propagate_clean_energy_config_to_model()
#         self._assert_global_architecture_contract()
#         self._assert_updated_stack_contract(phase=0)
#         self._set_base_trainable_params()

#     # ------------------------------------------------------------------
#     # Stack contract and trainability
#     # ------------------------------------------------------------------
#     def _assert_updated_stack_contract(self, phase: Optional[int] = None) -> None:
#         """Validate only the components required by the clean Low-Rank Geometry Replay / Residual Geometry Adaptation path.

#         This check intentionally does not require transport, calibration,
#         adaptive-boundary, or GeometryBank-native replay APIs. The cleaned
#         incremental trainer samples replay from frozen bank snapshots, so
#         requiring ``geometry_bank.sample_replay`` here would create a false
#         failure on a valid implementation.
#         """
#         strict = self._arg_bool("strict_updated_stack", True)
#         phase_i = 0 if phase is None else int(phase)
#         missing: List[str] = []

#         for attr in ("extract_projected_features", "get_subspace_bank"):
#             if not hasattr(self.model, attr):
#                 missing.append(f"model.{attr}")

#         if phase_i > 0:
#             if not hasattr(self.model, "compute_logits_from_features") and not hasattr(self.model, "classifier"):
#                 missing.append("model.compute_logits_from_features or model.classifier")
#             if self._incremental_update_mode() == "geometry_gated_adapter":
#                 if not hasattr(self.model, "geometry_plastic_adapter"):
#                     missing.append("model.geometry_plastic_adapter")
#                 if not hasattr(self.model, "adapt_projected_features"):
#                     # The adapter trainer has a safe identity fallback for replay,
#                     # but real new-sample adapter training needs this method to
#                     # expose adapted features. Keep this as a strict contract.
#                     missing.append("model.adapt_projected_features")

#         clf = getattr(self.model, "classifier", None)
#         if clf is None or not (
#             hasattr(clf, "forward")
#             or hasattr(clf, "compute_geometry_energy")
#             or hasattr(clf, "geometry_energy_from_bank")
#         ):
#             missing.append("model.classifier strict GeometryEnergyClassifier")

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             missing.append("model.geometry_bank")
#         else:
#             for attr in ("get_valid_mask", "assert_bank_valid"):
#                 if not hasattr(gb, attr):
#                     missing.append(f"geometry_bank.{attr}")
#             if phase_i > 0 and not any(hasattr(gb, a) for a in ("add_or_update_class_geometry", "update_class_geometry", "update_class")):
#                 missing.append("geometry_bank.add_or_update_class_geometry/update_class_geometry/update_class")

#         for attr in ("_safe_get_subspace_bank", "global_to_seen_local", "assert_bank_ready_for_seen_classes"):
#             if not hasattr(self, attr):
#                 missing.append(f"TrainerHelper.{attr}")

#         if missing:
#             msg = "Strict architecture stack contract failed; stale/missing component(s): " + ", ".join(missing)
#             if strict:
#                 raise RuntimeError(msg)
#             print(f"[WARN] {msg}")

#     def _assert_global_architecture_contract(self) -> None:
#         mode = self._incremental_update_mode()
#         for key in ("base_classifier_mode", "incremental_classifier_mode", "eval_classifier_mode"):
#             self._normalize_classifier_mode(getattr(self.args, key, "geometry_only"), context=key)

#         forbidden_true = [
#             "use_geometry_calibrator", "use_incremental_adapter", "use_bicyc_geometry_cycle",
#             "allow_incremental_projection_training", "use_sglat_transport", "use_geometry_transport",
#             "allow_old_model_transport", "use_boundary_geometry_replay", "use_adaptive_boundary",
#             "use_energy_calibrator", "geometry_normalize_logits",
#         ]
#         bad = [k for k in forbidden_true if self._arg_bool(k, False)]
#         if bad:
#             raise RuntimeError(f"Forbidden architecture switches are active: {bad}")
#         for key in ("bss_weight", "sym_bss_weight", "gdr_weight", "anchor_consistency_weight", "geometry_calibration_weight", "energy_calibration_weight"):
#             if abs(float(getattr(self.args, key, 0.0))) > 0.0:
#                 raise RuntimeError(f"{key} must be 0.0 in the strict geometry-replay/adaptation architecture path.")
#         if mode == "geometry_gated_adapter":
#             if not hasattr(self.model, "geometry_plastic_adapter"):
#                 raise RuntimeError("Residual Geometry Adaptation requires NECILModel.geometry_plastic_adapter.")
#             max_scale = float(getattr(getattr(self.model, "geometry_plastic_adapter", None), "max_scale", getattr(self.args, "adapter_max_scale", 0.0)) or 0.0)
#             if max_scale <= 0.0:
#                 raise RuntimeError("geometry_gated_adapter selected but adapter_max_scale <= 0.")

#     def _set_base_trainable_params(self) -> None:
#         self._force_clean_main_path_args()
#         self._propagate_clean_energy_config_to_model()
#         for name, p in self.model.named_parameters():
#             blocked = (
#                 name.startswith("classifier.") or name.startswith("geometry_bank.")
#                 or name.startswith("geometry_calibrator.") or name.startswith("geometry_cycle_calibrator.")
#                 or name.startswith("incremental_adapter.") or name.startswith("geometry_plastic_adapter.")
#                 or name.startswith("base_ce_head.")
#             )
#             p.requires_grad = not blocked
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()
#         if hasattr(self.model, "freeze_geometry_plastic_adapter"):
#             self.model.freeze_geometry_plastic_adapter()

#     def _set_incremental_trainable_params(self, old_class_count: int = 0) -> List[torch.nn.Parameter]:
#         old_class_count = int(old_class_count)
#         self._force_clean_main_path_args()
#         self._propagate_clean_energy_config_to_model()
#         self._incremental_update_mode()

#         for _, p in self.model.named_parameters():
#             p.requires_grad = False
#         if hasattr(self.model, "set_incremental_mode"):
#             try:
#                 self.model.set_incremental_mode(
#                     phase=int(getattr(self.model, "current_phase", 1)),
#                     old_class_count=old_class_count,
#                     train_classifier_calibration=False,
#                     train_geometry_adapter=True,
#                 )
#             except TypeError:
#                 self.model.set_incremental_mode(phase=int(getattr(self.model, "current_phase", 1)), old_class_count=old_class_count)
#         if hasattr(self.model, "enable_incremental_adapter"):
#             self.model.enable_incremental_adapter()
#         if hasattr(self.model, "unfreeze_geometry_plastic_adapter"):
#             self.model.unfreeze_geometry_plastic_adapter()
#         elif hasattr(self.model, "geometry_plastic_adapter"):
#             for p in self.model.geometry_plastic_adapter.parameters():
#                 p.requires_grad = True

#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()
#         if hasattr(self.model, "freeze_projection_head"):
#             self.model.freeze_projection_head()
#         if hasattr(self.model, "freeze_classifier"):
#             self.model.freeze_classifier()
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()

#         allowed = ("geometry_plastic_adapter",)
#         bad = [name for name, p in self.model.named_parameters() if p.requires_grad and not any(a in name for a in allowed)]
#         if bad:
#             raise RuntimeError(f"Incremental residual geometry adaptation allows only geometry_plastic_adapter parameters, got: {bad[:30]}")
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         if not params:
#             raise RuntimeError("Incremental residual geometry adaptation selected but no geometry_plastic_adapter parameters are trainable.")
#         return params

#     def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
#         return self._set_incremental_trainable_params(old_class_count)

#     def _has_feature_plasticity(self) -> bool:
#         return bool(self._adapter_mode_enabled())

#     def _has_descriptor_plasticity(self) -> bool:
#         return bool(getattr(self, "refine_new_descriptors", False))

#     def _has_energy_calibration_plasticity(self) -> bool:
#         return False

#     def _has_adaptive_boundary_plasticity(self) -> bool:
#         return False

#     # ------------------------------------------------------------------
#     # Phase/mode helpers
#     # ------------------------------------------------------------------
#     def _base_classifier_mode(self) -> str:
#         return "geometry_only"

#     def _inc_classifier_mode(self) -> str:
#         return "geometry_only"

#     def _eval_classifier_mode(self) -> str:
#         return "geometry_only"

#     def _set_model_phase_and_old_count(self, phase: int, old_class_count: int) -> None:
#         phase = int(phase)
#         old_class_count = int(old_class_count)
#         if phase == 0 and hasattr(self.model, "set_base_mode"):
#             try:
#                 self.model.set_base_mode(train_backbone=True, train_projection=True)
#             except TypeError:
#                 self.model.set_base_mode()
#         elif phase > 0 and hasattr(self.model, "set_incremental_mode"):
#             try:
#                 self.model.set_incremental_mode(phase=phase, old_class_count=old_class_count, train_geometry_adapter=True)
#             except TypeError:
#                 self.model.set_incremental_mode(phase=phase, old_class_count=old_class_count)
#         else:
#             self.model.current_phase = phase
#             self.model.old_class_count = old_class_count
#         if hasattr(self.model, "current_phase"):
#             self.model.current_phase = phase
#         if hasattr(self.model, "old_class_count"):
#             self.model.old_class_count = old_class_count

#     # ------------------------------------------------------------------
#     # Validation/scoring helpers
#     # ------------------------------------------------------------------
#     def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
#             return self._zero(logits)
#         labels = labels.long().view(-1).to(logits.device)
#         if labels.numel() != logits.size(0):
#             raise RuntimeError(f"CE batch mismatch: logits={logits.size(0)}, labels={labels.numel()}")
#         if labels.numel() == 0:
#             return logits.sum() * 0.0
#         lo = int(labels.min().detach().item())
#         hi = int(labels.max().detach().item())
#         if lo < 0 or hi >= logits.size(1):
#             raise RuntimeError(f"CE label range [{lo},{hi}] incompatible with logits width={logits.size(1)}")
#         clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
#         smoothing = float(getattr(self, "label_smoothing", getattr(self.args, "label_smoothing", 0.0)))
#         return F.cross_entropy(logits.clamp(-clip, clip), labels, label_smoothing=smoothing)

#     def _classes_tensor(self, class_ids: Iterable[int], *, device=None) -> torch.Tensor:
#         ids = [int(c) for c in class_ids]
#         if not ids:
#             raise RuntimeError("class_ids must be non-empty.")
#         if len(set(ids)) != len(ids):
#             raise RuntimeError(f"class_ids contains duplicates: {ids}")
#         if min(ids) < 0:
#             raise RuntimeError(f"class_ids must be non-negative global IDs, got {ids}")
#         return torch.as_tensor(ids, device=device if device is not None else self.device, dtype=torch.long)

#     def _seen_classes_for_phase(self, phase: int, fallback_labels: Optional[torch.Tensor] = None) -> List[int]:
#         if hasattr(self.dataset, "get_classes_up_to_phase"):
#             try:
#                 seen = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase))]
#                 if seen:
#                     return list(dict.fromkeys(seen))
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
#         return list(dict.fromkeys(seen))

#     def _assert_labels_in_seen_classes(self, labels: torch.Tensor, seen_classes: Iterable[int], *, context: str) -> None:
#         if hasattr(self, "assert_global_labels_in_set"):
#             return self.assert_global_labels_in_set(labels, seen_classes, context)
#         y = labels.to(self.device).long().view(-1)
#         seen = set(int(c) for c in seen_classes)
#         bad = sorted(set(int(v) for v in y.detach().cpu().tolist()) - seen)
#         if bad:
#             raise RuntimeError(f"{context}: labels outside seen classes. bad={bad}, seen={sorted(seen)}")

#     def _global_to_seen_local_safe(self, labels_global: torch.Tensor, seen_classes: Iterable[int], *, context: str) -> torch.Tensor:
#         if hasattr(self, "global_to_seen_local"):
#             return self.global_to_seen_local(labels_global, seen_classes, context=context)
#         y = labels_global.to(self.device).long().view(-1)
#         seen = [int(c) for c in seen_classes]
#         mapping = {c: i for i, c in enumerate(seen)}
#         local = torch.full_like(y, -1)
#         for c, i in mapping.items():
#             local[y == int(c)] = int(i)
#         if bool((local < 0).any().item()):
#             bad = sorted(set(int(v) for v in y[local < 0].detach().cpu().tolist()))
#             raise RuntimeError(f"{context}: labels not in seen_classes: {bad}; seen={seen}")
#         return local

#     def _seen_local_to_global_safe(self, preds_local: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
#         if hasattr(self, "seen_local_to_global"):
#             return self.seen_local_to_global(preds_local, seen_classes, context="seen_local_to_global")
#         seen = torch.as_tensor([int(c) for c in seen_classes], device=preds_local.device, dtype=torch.long)
#         return seen.index_select(0, preds_local.long().view(-1))

#     def _mask_logits_to_seen_classes(self, logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
#         """Validate and return strict seen-local logits.

#         The cleaned classifier has one output contract: [B, len(seen_classes)].
#         Full-global logits are not accepted because silently slicing them hides
#         class-order bugs during incremental learning.
#         """
#         if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError(f"logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")
#         seen = [int(c) for c in seen_classes]
#         if logits.size(1) != len(seen):
#             raise RuntimeError(
#                 f"Strict classifier contract violation: logits width={logits.size(1)} but len(seen_classes)={len(seen)}. "
#                 "Return seen-local logits from the model/classifier instead of global-full logits."
#             )
#         if not torch.isfinite(logits).all():
#             raise RuntimeError("Strict classifier contract violation: logits contain NaN/Inf.")
#         return logits

#     def _old_new_classes_for_validation(self, phase: int, old_class_count: int, seen_classes: Iterable[int]) -> Tuple[List[int], List[int]]:
#         seen = [int(c) for c in seen_classes]
#         old: List[int] = []
#         if int(old_class_count) > 0 and int(phase) > 0 and hasattr(self.dataset, "get_classes_up_to_phase"):
#             try:
#                 old = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase) - 1)]
#             except Exception:
#                 old = []
#         if not old and int(old_class_count) > 0:
#             old = seen[: int(old_class_count)]
#         old_set = set(old)
#         new = [c for c in seen if c not in old_set]
#         return [c for c in seen if c in old_set], new

#     def _label_membership_mask(self, labels: torch.Tensor, class_ids: Iterable[int]) -> torch.Tensor:
#         mask = torch.zeros_like(labels, dtype=torch.bool)
#         for c in [int(x) for x in class_ids]:
#             mask |= labels.eq(int(c))
#         return mask

#     def _prepare_batch_spectral_summary(self, batch_spectra: Optional[torch.Tensor], x: torch.Tensor) -> Tuple[Optional[torch.Tensor], bool]:
#         if torch.is_tensor(batch_spectra) and batch_spectra.numel() > 0:
#             s = batch_spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
#             if s.dim() == 4:
#                 s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#             elif s.dim() == 3:
#                 s = s[:, :, s.size(-1) // 2] if s.size(0) == x.size(0) and s.size(2) > 1 else s.flatten(1)
#             elif s.dim() == 1:
#                 s = s.view(x.size(0), -1)
#             elif s.dim() > 4:
#                 s = s.flatten(1)
#             if s.size(0) != x.size(0):
#                 raise RuntimeError(f"Batch spectral summary mismatch: {tuple(s.shape)} vs input {tuple(x.shape)}")
#             return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), bool(getattr(self.args, "raw_spectral_summary_is_physical", True))
#         return None, False

#     def _forward_real_batch(
#         self,
#         x: torch.Tensor,
#         batch_spectra: Optional[torch.Tensor] = None,
#         *,
#         classifier_mode: Optional[str] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_energy: bool = True,
#         return_parts: bool = False,
#     ) -> Dict[str, Any]:
#         mode = self._normalize_classifier_mode(classifier_mode or self._eval_classifier_mode(), context="real_batch_forward")
#         spectral_summary, is_physical = self._prepare_batch_spectral_summary(batch_spectra, x)
#         kwargs = dict(
#             classifier_mode=mode,
#             mode=mode,
#             seen_classes=list(seen_classes) if seen_classes is not None else None,
#             old_classes=list(old_classes) if old_classes is not None else None,
#             new_classes=list(new_classes) if new_classes is not None else None,
#             return_energy=return_energy,
#             return_parts=return_parts,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=bool(is_physical),
#         )
#         try:
#             out = self.model(x, **kwargs)
#         except TypeError:
#             # Compatibility fallback for older model.forward; strict classifier/model should not hit this.
#             fallback = {k: v for k, v in kwargs.items() if k in {"classifier_mode", "return_energy", "return_parts"}}
#             out = self.model(x, **fallback)
#         if not isinstance(out, dict):
#             out = {"logits": out}
#         out["spectral_summary"] = spectral_summary
#         out["spectral_summary_is_physical"] = bool(is_physical)
#         return out

#     @torch.no_grad()
#     def _validate_split_metrics(self, loader, old_class_count: int) -> Dict[str, Any]:
#         self._assert_global_architecture_contract()
#         self.model.eval()
#         old_class_count = int(old_class_count)
#         phase = int(getattr(self.model, "current_phase", 0))
#         seen_classes = self._seen_classes_for_phase(phase)
#         old_classes, new_classes = self._old_new_classes_for_validation(phase, old_class_count, seen_classes)
#         old_pos = [seen_classes.index(c) for c in old_classes if c in seen_classes]
#         new_pos = [seen_classes.index(c) for c in new_classes if c in seen_classes]

#         total_loss = total_correct = total = batches = 0
#         old_correct = old_total = new_correct = new_total = 0
#         new_into_old_sum = old_into_new_sum = old_new_gap_sum = 0.0
#         old_new_diag_batches = 0

#         for batch in loader:
#             x, y, batch_spectra, _ = self._unpack_hsi_batch(batch)
#             x = x.to(self.device, non_blocking=True).float()
#             y = y.to(self.device, non_blocking=True).long().view(-1)
#             self._assert_labels_in_seen_classes(y, seen_classes, context=f"phase_{phase}_validation")
#             y_local = self._global_to_seen_local_safe(y, seen_classes, context=f"phase_{phase}_validation")
#             out = self._forward_real_batch(
#                 x,
#                 batch_spectra,
#                 classifier_mode="geometry_only",
#                 seen_classes=seen_classes,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#                 return_energy=True,
#                 return_parts=False,
#             )
#             logits_seen = self._mask_logits_to_seen_classes(out["logits"], seen_classes)
#             if logits_seen.size(0) != y_local.numel():
#                 raise RuntimeError(f"Validation logits/labels mismatch: {logits_seen.size(0)} vs {y_local.numel()}")
#             loss = self._stable_ce(logits_seen, y_local)
#             pred_local = logits_seen.argmax(dim=1)
#             pred_global = self._seen_local_to_global_safe(pred_local, seen_classes)
#             correct = pred_global.eq(y)

#             total_loss += float(loss.detach().item())
#             total_correct += int(correct.sum().item())
#             total += int(y.numel())
#             batches += 1

#             if old_class_count > 0 and old_classes and new_classes:
#                 old_mask = self._label_membership_mask(y, old_classes)
#                 new_mask = self._label_membership_mask(y, new_classes)
#                 if bool(old_mask.any().item()):
#                     old_correct += int(correct[old_mask].sum().item())
#                     old_total += int(old_mask.sum().item())
#                 if bool(new_mask.any().item()):
#                     new_correct += int(correct[new_mask].sum().item())
#                     new_total += int(new_mask.sum().item())

#             energy = out.get("energy", None) if isinstance(out, dict) else None
#             if torch.is_tensor(energy) and energy.dim() == 2 and old_pos and new_pos:
#                 if energy.size(1) != len(seen_classes):
#                     raise RuntimeError(
#                         f"Strict energy contract violation: energy width={energy.size(1)} but len(seen_classes)={len(seen_classes)}."
#                     )
#                 e_seen = energy
#                 old_idx = torch.as_tensor(old_pos, device=e_seen.device, dtype=torch.long)
#                 new_idx = torch.as_tensor(new_pos, device=e_seen.device, dtype=torch.long)
#                 old_min = e_seen.index_select(1, old_idx).min(dim=1).values
#                 new_min = e_seen.index_select(1, new_idx).min(dim=1).values
#                 old_mask_e = self._label_membership_mask(y, old_classes)
#                 new_mask_e = self._label_membership_mask(y, new_classes)
#                 if bool(new_mask_e.any().item()):
#                     new_into_old_sum += float((old_min[new_mask_e] < new_min[new_mask_e]).float().mean().detach().item())
#                 if bool(old_mask_e.any().item()):
#                     old_into_new_sum += float((new_min[old_mask_e] < old_min[old_mask_e]).float().mean().detach().item())
#                 old_new_gap_sum += float((new_min - old_min).mean().detach().item())
#                 old_new_diag_batches += 1

#         acc = 100.0 * total_correct / max(total, 1)
#         split = old_class_count > 0 and bool(old_classes) and bool(new_classes)
#         old_acc = 100.0 * old_correct / max(old_total, 1) if split else 0.0
#         new_acc = 100.0 * new_correct / max(new_total, 1) if split else acc
#         hm = 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-8) if split else acc
#         return {
#             "loss": total_loss / max(batches, 1),
#             "acc": acc,
#             "old_acc": old_acc,
#             "new_acc": new_acc,
#             "hm": hm,
#             "old_new_split_available": bool(split),
#             "predicted_unseen": 0.0,
#             "new_into_old_rate": float(new_into_old_sum / max(old_new_diag_batches, 1)),
#             "old_into_new_rate": float(old_into_new_sum / max(old_new_diag_batches, 1)),
#             "mean_old_new_energy_gap": float(old_new_gap_sum / max(old_new_diag_batches, 1)),
#             "seen_classes": seen_classes,
#             "old_classes": old_classes,
#             "new_classes": new_classes,
#         }

#     # ------------------------------------------------------------------
#     # Scoring/checkpoint helpers
#     # ------------------------------------------------------------------
#     def _base_geometry_global_metrics(self) -> Dict[str, float]:
#         """Use the base trainer's cleaned geometry-state aliases.

#         Keeping a second local implementation here would override the stricter
#         base-phase scoring logic and lose the base geometry-energy margin fields.
#         """
#         return BasePhaseTrainer._base_geometry_global_metrics(self)

#     def _select_score(self, val_stats: Dict[str, float], phase: int) -> float:
#         if int(phase) > 0:
#             metric = str(getattr(self, "best_state_metric", getattr(self.args, "best_state_metric", "hm"))).lower().strip()
#             if metric in {"acc", "oa"}:
#                 return float(val_stats.get("acc", 0.0))
#             if metric in {"old_new_min", "min"}:
#                 return min(float(val_stats.get("old_acc", 0.0)), float(val_stats.get("new_acc", 0.0)))
#             if metric in {"loss", "val_loss"}:
#                 return -float(val_stats.get("loss", 0.0))
#             return float(val_stats.get("hm", 0.0))
#         return self._select_base_checkpoint_score(val_stats, getattr(self, "_last_base_geom_stats", None))

#     def _select_base_checkpoint_score(self, val_stats: Dict[str, float], geom_stats: Optional[Dict[str, float]] = None) -> float:
#         """Delegate to BasePhaseTrainer so base energy-margin health affects checkpoint selection."""
#         return BasePhaseTrainer._select_base_checkpoint_score(self, val_stats, geom_stats)

#     def _capture_state(self) -> Dict[str, torch.Tensor]:
#         return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

#     def _print_trainable_summary(self, phase: int) -> None:
#         trainable = [(n, int(p.numel())) for n, p in self.model.named_parameters() if p.requires_grad]
#         total = sum(n for _, n in trainable)
#         if int(phase) == 0:
#             print(f"[Trainable] Base phase: {total:,} params | backbone/projection + temporary CE head")
#             print("[Base Objective] balanced CE + GICS + geometry reserve(compact,center,subspace,band,volume) + base geometry-energy margin")
#         else:
#             print(f"[Trainable] Incremental phase {phase}: {total:,} params | residual geometry adapter only")
#             print("[Incremental Objective] low-rank GeometryBank replay + residual geometry adaptation + joint CE + old/new energy margins")
#         if self.debug:
#             for name, count in trainable[:150]:
#                 print(f"  {name}: {count:,}")

#     def _current_runtime_contract(self) -> Dict[str, object]:
#         adapter_trainable = any(p.requires_grad and "geometry_plastic_adapter" in n for n, p in self.model.named_parameters())
#         return {
#             "method": "Low-Rank Geometry Replay and Residual Geometry Adaptation for HSI NECIL",
#             "feature_space": "canonical projected z; incremental residual adaptation is bounded and old rows stay frozen",
#             "classifier": "strict seen-class low-rank GeometryBank energy",
#             "classifier_output": "[B, len(seen_classes)]",
#             "incremental_update_mode": self._incremental_update_mode(),
#             "residual_geometry_adapter_present": bool(hasattr(self.model, "geometry_plastic_adapter")),
#             "residual_geometry_adapter_trainable": bool(adapter_trainable),
#             "old_memory": "frozen GeometryBank statistics only",
#             "raw_exemplars": False,
#             "kd_teacher": False,
#             "transport": False,
#             "adaptive_boundary": False,
#             "energy_calibrator": False,
#             "spectral_classifier_branch": False,
#             "base_objective": "balanced CE + GICS + geometry reserve + base geometry-energy margin",
#             "incremental_objective": "low-rank geometry replay + residual geometry adaptation + joint CE + old/new energy margin",
#             "uses_logdet_energy": bool(self.use_logdet_energy),
#             "logdet_energy_weight": float(self.logdet_energy_weight),
#             "row_energy_standardization": False,
#         }

#     # ------------------------------------------------------------------
#     # Phase dispatch
#     # ------------------------------------------------------------------
#     def _assert_incremental_preflight(
#         self,
#         phase: int,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#     ) -> None:
#         """Preflight for the clean incremental row-insertion path.

#         The previous version used ``range(old_class_count)``. That works only
#         for sequential class ids and can silently freeze/check the wrong rows.
#         This version resolves old/new/seen classes from the dataset protocol and
#         validates the exact global GeometryBank rows that will be used.
#         """
#         phase = int(phase)
#         if phase <= 0:
#             raise RuntimeError("Incremental preflight requires phase > 0.")
#         self._assert_updated_stack_contract(phase=phase)

#         if old_classes is None or new_classes is None or seen_classes is None:
#             old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
#         old_ids = [int(c) for c in old_classes]
#         new_ids = [int(c) for c in new_classes]
#         seen_ids = [int(c) for c in seen_classes]

#         if not old_ids:
#             raise RuntimeError(f"Incremental phase {phase} has no old classes.")
#         if not new_ids:
#             raise RuntimeError(f"Incremental phase {phase} has no new classes.")
#         if set(old_ids).intersection(new_ids):
#             raise RuntimeError(f"Incremental phase {phase} old/new overlap: {sorted(set(old_ids).intersection(new_ids))}")
#         if seen_ids != list(dict.fromkeys([*old_ids, *new_ids])):
#             raise RuntimeError(f"Incremental phase {phase} seen_classes must equal old+new. old={old_ids}, new={new_ids}, seen={seen_ids}")

#         # Central contract check added in the cleaned helper; fall back to local
#         # checks when older helpers are still present.
#         fn = getattr(self, "assert_clean_incremental_contract", None)
#         if callable(fn):
#             fn(phase=phase, old_classes=old_ids, new_classes=new_ids, seen_classes=seen_ids)

#         if self._incremental_update_mode() == "geometry_gated_adapter" and not hasattr(self.model, "geometry_plastic_adapter"):
#             raise RuntimeError("Incremental Residual Geometry Adaptation requires model.geometry_plastic_adapter.")

#         bank = self._safe_get_subspace_bank(require_ready=True)
#         self.assert_bank_ready_for_seen_classes(bank, old_ids)
#         # Before inserting new rows, future rows must not already be valid.
#         if hasattr(self, "assert_bank_has_only_allowed_valid_rows"):
#             self.assert_bank_has_only_allowed_valid_rows(bank, old_ids)

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None:
#             if hasattr(gb, "freeze_classes"):
#                 gb.freeze_classes(old_ids)
#             elif hasattr(gb, "freeze_classes_up_to") and old_ids == list(range(max(old_ids) + 1)):
#                 gb.freeze_classes_up_to(max(old_ids) + 1)
#             elif hasattr(gb, "frozen_class_mask"):
#                 gb.frozen_class_mask[torch.as_tensor(old_ids, device=gb.frozen_class_mask.device)] = True
#             if hasattr(gb, "assert_bank_valid"):
#                 try:
#                     gb.assert_bank_valid(seen_classes=old_ids, strict=True)
#                 except TypeError:
#                     gb.assert_bank_valid(seen_classes=old_ids)

#         print(f"[Incremental Preflight] phase={phase} | old={old_ids} | new={new_ids} | seen={seen_ids} | old_rows_frozen=True")

#     def train_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4):
#         """Dispatch one phase using the strict geometry-replay/adaptation contract.

#         Phase 0 delegates to the certificate-aware base trainer. Incremental
#         phases resolve old/new/seen classes from the dataset protocol, freeze the
#         certified old rows, configure only permitted trainable parameters, and
#         then delegate to ``train_incremental_phase``.
#         """
#         phase = int(phase)
#         self.early_stop_patience = 0
#         for k in ("early_stop_patience", "base_early_stop_patience", "incremental_early_stop_patience"):
#             try:
#                 setattr(self.args, k, 0)
#             except Exception:
#                 pass

#         self._force_clean_main_path_args()
#         self._propagate_clean_energy_config_to_model()
#         self._assert_global_architecture_contract()

#         if phase == 0:
#             self._set_model_phase_and_old_count(0, 0)
#             self._set_base_trainable_params()
#             self._assert_updated_stack_contract(phase=0)
#             self._print_trainable_summary(phase=0)
#             return self.train_base_phase(phase=0, epochs=epochs, batch_size=batch_size, lr=lr)

#         if self.base_only or self.disable_incremental_training:
#             raise RuntimeError(
#                 "Incremental training is disabled. Set --base_only false and "
#                 "--disable_incremental_training false."
#             )

#         old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
#         old_class_count = len(old_classes)
#         self._set_model_phase_and_old_count(phase, old_class_count)
#         self._set_incremental_trainable_params(old_class_count)
#         self._assert_incremental_preflight(
#             phase,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             seen_classes=seen_classes,
#         )
#         self._print_trainable_summary(phase=phase)
#         return self.train_incremental_phase(phase=phase, epochs=epochs, batch_size=batch_size, lr=lr)

#     def _old_new_boundary_preservation_loss(self, *args, **kwargs):
#         """Delegate to the real IncrementalPhaseTrainer implementation.

#         This method used to return a zero stub from Trainer, which silently
#         disabled the old/new invasion loss during descriptor refinement because
#         Trainer appears before IncrementalPhaseTrainer in the MRO.
#         """
#         return IncrementalPhaseTrainer._old_new_boundary_preservation_loss(self, *args, **kwargs)

#     def _project_new_descriptor_params_out_of_old_tangent_space(self, *args, **kwargs):
#         """Delegate to the real new-row-only projection implementation."""
#         return IncrementalPhaseTrainer._project_new_descriptor_params_out_of_old_tangent_space(self, *args, **kwargs)

#     def _adaptive_boundary_loss_from_current_bank(self, *args, **kwargs):
#         """Adaptive-boundary remains disabled through IncrementalPhaseTrainer."""
#         return IncrementalPhaseTrainer._adaptive_boundary_loss_from_current_bank(self, *args, **kwargs)

#     def _adaptive_boundary_enabled(self) -> bool:
#         return False

#     def _adaptive_boundary_state(self, old_class_count: int = 0) -> Dict[str, float]:
#         return {"adaptive_boundary_enabled": 0.0, "boundary_radius_mean": 0.0, "old_boundary_radius_mean": 0.0, "new_boundary_radius_mean": 0.0}

#     def _adaptive_boundary_trainable_params(self) -> List[torch.nn.Parameter]:
#         return []

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
#             "base_objective": "balanced CE + GICS + geometry reserve + base geometry-energy margin",
#             "incremental_objective": "low-rank geometry replay + residual geometry adaptation + joint CE + old/new geometry energy margins",
#             "runtime_contract": self._current_runtime_contract(),
#             "architecture_contract": {
#                 "method": "Low-Rank Geometry Replay and Residual Geometry Adaptation for HSI NECIL",
#                 "classifier_mode": "geometry_only",
#                 "bank": "mean,basis,eigvals,residual_variance,reliability,sample_count,band_signature,spectral_shape",
#                 "old_memory": "frozen GeometryBank statistics only",
#                 "replay": "synthetic GeometryBank replay only; no raw old patches/features",
#                 "incremental_update_mode": self._incremental_update_mode(),
#                 "residual_adaptation": "geometry_plastic_adapter_only",
#                 "forbidden_components": ["KD", "teacher", "prototypes", "transport", "adaptive_boundary", "measured_calibration", "BSS/GDR"],
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

