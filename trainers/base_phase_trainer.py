from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, List

import copy
import json
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from losses.loss import (
    unified_spectral_geometry_loss,
    base_center_overlap_diagnostics,
    spectral_shape_discrimination_loss,
    geometry_energy_matrix,
)




class BasePhaseTrainer:
    # ============================================================
    # Config helpers
    # ============================================================
    def _base_cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self, name, getattr(self.args, name, default)))

    def _base_cfg_int(self, name: str, default: int) -> int:
        return int(getattr(self, name, getattr(self.args, name, default)))

    def _base_cfg_bool(self, name: str, default: bool) -> bool:
        v = getattr(self, name, getattr(self.args, name, default))
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(v)

    def _zero(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(ref, torch.Tensor):
            return ref.sum() * 0.0
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def _enforce_base_contract(self) -> None:
        """Force a base-only training contract without poisoning incremental args.

        The previous version mutated shared ``args`` fields such as
        ``use_geometry_transport`` and ``eval_classifier_mode`` during base
        training. That is a silent architecture bug: after phase 0, the
        incremental trainer receives already-disabled transport/refinement flags.

        This method now applies only runtime model restrictions for phase 0. It
        keeps the command-line/incremental configuration intact, while ensuring
        that the base phase trains only backbone/projection + temporary CE head
        and builds a clean PG-RGA/GeometryBank handoff.
        """
        # Keep the method coherent: GICS/PGR/SpectralShape are internal parts of
        # SRPGR. Do not mutate shared args here. Legacy spectral-GICS knobs are
        # ignored by the cleaned base objective and must be controlled by the
        # loss-call kwargs, not by rewriting the global runtime configuration.

        if self._base_cfg_bool("use_mssl_loss", False) and self._base_cfg_float("mssl_weight", 0.0) > 0.0:
            if not self._base_cfg_bool("unsafe_ablation_use_mssl_loss", False):
                raise RuntimeError(
                    "PG-RGA base trainer uses one coherent objective: CE + SRPGR. "
                    "MSSL must be a separate ablation, not mixed into the main method."
                )

        # Runtime-only base restrictions. Do not mutate shared args here.
        for name, value in (
            ("use_incremental_adapter", False),
            ("use_bicyc_geometry_cycle", False),
            ("use_geometry_calibrator", False),
            ("use_sglat_transport", False),
            ("use_geometry_transport", False),
            ("allow_old_model_transport", False),
        ):
            if hasattr(self.model, name):
                setattr(self.model, name, value)

        # Base representation learning uses CE + GICS + PGR (+ optional physical
        # spectral-shape loss inside losses.loss).  The classifier/certificate
        # must judge feature GeometryBank quality only; no PG-RGA/spectral scoring
        # branch is allowed in the main classifier path.
        for attr in (
            "default_base_classifier_mode",
            "default_incremental_classifier_mode",
            "default_eval_classifier_mode",
        ):
            if hasattr(self.model, attr):
                setattr(self.model, attr, "geometry_only")

        # Freeze adapters/calibrators for base without destroying the user's
        # incremental configuration.  The snapshot/restore guard below restores
        # runtime flags after phase 0.
        if hasattr(self.model, "freeze_geometry_plastic_adapter"):
            self.model.freeze_geometry_plastic_adapter()
        elif hasattr(self.model, "disable_incremental_adapter"):
            self.model.disable_incremental_adapter()
        if hasattr(self.model, "freeze_geometry_calibrator"):
            self.model.freeze_geometry_calibrator()
        if hasattr(self.model, "freeze_energy_calibrator"):
            self.model.freeze_energy_calibrator()

        self._assert_mandatory_base_objective_config()


    def _assert_mandatory_base_objective_config(self) -> None:
        """Base phase has no optional architecture components.

        Phase 0 must always optimize the same canonical objective:
            balanced CE + GICS + PGR(compact, center, subspace, band, volume)
            + physical spectral-shape reserve when raw wavelength spectra exist.

        Spectral-shape derivatives cannot be forced when only PCA/reduced
        components are available. In that case the mandatory HSI reserve is the
        band-summary term. All other base terms are hard-required.
        """
        required_positive = {
            "base_ce_weight": self._base_cfg_float("base_ce_weight", 1.0),
            "base_srpgr_weight": self._base_cfg_float("base_srpgr_weight", 1.0),
            "base_gics_weight": self._base_cfg_float("base_gics_weight", 0.20),
            "pgr_weight": self._base_cfg_float("pgr_weight", 0.10),
            "pgr_compact_weight": self._base_cfg_float("pgr_compact_weight", 0.15),
            "pgr_center_weight": self._base_cfg_float("pgr_center_weight", 0.20),
            "pgr_subspace_weight": self._base_cfg_float("pgr_subspace_weight", 0.10),
            "pgr_band_weight": self._base_cfg_float("pgr_band_weight", 0.05),
            "pgr_volume_weight": self._base_cfg_float("pgr_volume_weight", 0.05),
            "base_gics_temperature": self._base_cfg_float("base_gics_temperature", 0.07),
            "pgr_center_margin": self._base_cfg_float("pgr_center_margin", 1.05),
            "pgr_max_class_variance": self._base_cfg_float("pgr_max_class_variance", 0.75),
            "pgr_min_class_variance": self._base_cfg_float("pgr_min_class_variance", 0.015),
            "pgr_max_subspace_overlap": self._base_cfg_float("pgr_max_subspace_overlap", self._base_cfg_float("base_cert_max_subspace_overlap", 0.55)),
        }
        bad = [f"{k}={v}" for k, v in required_positive.items() if float(v) <= 0.0]
        if bad:
            raise RuntimeError(
                "Base phase has no optional core components. Required base weights/values must be > 0: "
                + ", ".join(bad)
            )
        if self._base_cfg_int("pgr_subspace_rank", 3) <= 0:
            raise RuntimeError("pgr_subspace_rank must be > 0 because subspace reserve is mandatory.")
        if self._base_cfg_int("pgr_min_class_samples", 3) < 2:
            raise RuntimeError("pgr_min_class_samples must be >= 2 for compact/center reserve.")
        if self._base_cfg_int("pgr_subspace_min_samples", 6) < 3:
            raise RuntimeError("pgr_subspace_min_samples must be >= 3 for low-rank subspace reserve.")
        if not self._base_cfg_bool("base_class_balance", True):
            raise RuntimeError("base_class_balance must be true in the main PG-RGA base phase.")
        if unified_spectral_geometry_loss is None:
            raise RuntimeError("unified_spectral_geometry_loss is required for mandatory CE+GICS+PGR base training.")

    def _assert_mandatory_base_batch_inputs(
        self,
        *,
        features: torch.Tensor,
        labels_local: torch.Tensor,
        key_features: Optional[torch.Tensor],
        spectral_summary: Optional[torch.Tensor],
        band_summary: Optional[torch.Tensor],
    ) -> None:
        """Fail fast when a required base component cannot be computed."""
        if not torch.is_tensor(key_features) or key_features.shape != features.shape:
            raise RuntimeError(
                "Mandatory GICS requires detached key_features with the same shape as query features. "
                f"query={tuple(features.shape)}, key={None if key_features is None else tuple(key_features.shape)}"
            )
        if bool(key_features.requires_grad):
            raise RuntimeError("Mandatory GICS key_features must be detached; otherwise GICS becomes a second train path.")
        if not torch.is_tensor(spectral_summary) or spectral_summary.dim() != 2 or spectral_summary.size(0) != features.size(0):
            raise RuntimeError(
                "Mandatory base HSI reserve requires spectral_summary [B,S] for band/PGR construction. "
                f"got {None if spectral_summary is None else tuple(spectral_summary.shape)}"
            )
        if not torch.is_tensor(band_summary) or band_summary.dim() != 2 or band_summary.size(0) != features.size(0):
            raise RuntimeError(
                "Mandatory PGR band reserve requires band_summary [B,S]. "
                f"got {None if band_summary is None else tuple(band_summary.shape)}"
            )
        if band_summary.size(1) <= 0:
            raise RuntimeError("Mandatory PGR band reserve received zero spectral/band dimension.")
        if not torch.isfinite(spectral_summary).all() or not torch.isfinite(band_summary).all():
            raise RuntimeError("spectral_summary/band_summary contains NaN/Inf in mandatory base phase.")
        if labels_local.numel() != features.size(0):
            raise RuntimeError("labels/features batch mismatch in mandatory base objective.")
        if int(torch.unique(labels_local).numel()) < 2:
            raise RuntimeError(
                "Mandatory GICS/PGR center reserve requires at least two classes in each base batch. "
                "Use a class-balanced base sampler/batch construction."
            )

    def _assert_mandatory_base_loss_parts(self, loss_out: Dict[str, Any], *, spectral_summary_is_physical: bool) -> None:
        """Ensure the unified loss exposes every required base component."""
        required = (
            "total", "base_gics", "base_gics_weighted", "base_gics_anchors", "base_gics_pos",
            "base_pgr", "base_pgr_weighted", "base_compact", "base_center", "base_subspace",
            "base_band", "base_volume", "base_pgr_valid_class_count", "base_pgr_subspace_pair_count",
            "base_pgr_band_pair_count", "base_pgr_volume_factor", "base_spectral_shape_active",
            "base_pgr_subspace_max_overlap", "base_pgr_band_max_similarity",
            "base_spectral_shape_raw", "base_spectral_shape_pair_count",
        )
        missing = [k for k in required if k not in loss_out]
        if missing:
            raise RuntimeError("unified_spectral_geometry_loss did not return mandatory base keys: " + ", ".join(missing))
        for k in required:
            v = loss_out[k]
            if torch.is_tensor(v) and v.numel() > 0 and not torch.isfinite(v).all():
                raise RuntimeError(f"mandatory base loss component {k} contains NaN/Inf.")
        if self._scalar(loss_out.get("base_gics_anchors", 0.0)) <= 0.0:
            raise RuntimeError("Mandatory GICS has zero valid anchors. Fix base batch construction/key view.")
        if self._scalar(loss_out.get("base_pgr_valid_class_count", 0.0)) <= 0.0:
            raise RuntimeError("Mandatory PGR has zero valid class groups in this batch.")
        if self._base_cfg_bool("base_require_physical_spectral_shape", False) and not bool(spectral_summary_is_physical):
            raise RuntimeError(
                "base_require_physical_spectral_shape=true but this batch is not marked physical. "
                "Provide raw wavelength-ordered center spectra or disable this strict raw-spectrum requirement."
            )

    def _assert_mandatory_base_epoch_coverage(self, stats: Dict[str, float]) -> None:
        """Check that required base terms were structurally active over the epoch."""
        if not self._base_cfg_bool("strict_base_component_coverage", True):
            return
        failures: List[str] = []
        if float(stats.get("gics_anchors", 0.0)) <= 0.0:
            failures.append("GICS valid anchors were zero")
        if float(stats.get("pgr_valid_class_count", 0.0)) <= 0.0:
            failures.append("PGR valid class count was zero")
        if float(stats.get("pgr_subspace_pair_count", 0.0)) <= 0.0:
            failures.append("PGR subspace pair count was zero; subspace reserve was not exercised")
        if float(stats.get("pgr_band_pair_count", 0.0)) <= 0.0:
            failures.append("PGR band pair count was zero; band reserve was not exercised")
        if float(stats.get("pgr_volume_factor", 0.0)) <= 0.0:
            failures.append("PGR volume factor was zero; volume reserve was not exercised")
        if self._base_cfg_bool("base_require_physical_spectral_shape", False) and float(stats.get("spectral_active", 0.0)) <= 0.0:
            failures.append("physical spectral-shape reserve was required but never active")
        if failures:
            raise RuntimeError(
                "Mandatory base objective coverage failed. This usually means the base sampler/batch composition "
                "does not provide enough same-class and cross-class evidence. " + "; ".join(failures)
            )

    def _capture_incremental_runtime_flags(self) -> Dict[str, Any]:
        """Snapshot args/model fields that base training must not destroy."""
        keys = (
            "use_sglat_transport",
            "use_geometry_transport",
            "allow_old_model_transport",
            "eval_classifier_mode",
            "incremental_classifier_mode",
            "base_classifier_mode",
            "use_incremental_adapter",
            "use_bicyc_geometry_cycle",
            "use_geometry_calibrator",
            "use_energy_calibrator",
            "use_geometry_gated_adapter",
            "incremental_update_mode",
            "base_gics_spectral_weight",
            "base_gics_band_weight",
            "base_gics_spectral_temperature",
            "base_gics_band_temperature",
        )
        state: Dict[str, Any] = {"args": {}, "model": {}}
        for key in keys:
            if hasattr(self.args, key):
                state["args"][key] = getattr(self.args, key)
            if hasattr(self.model, key):
                state["model"][key] = getattr(self.model, key)
        for key in ("default_eval_classifier_mode", "default_incremental_classifier_mode", "default_base_classifier_mode"):
            if hasattr(self.model, key):
                state["model"][key] = getattr(self.model, key)
        return state

    def _restore_incremental_runtime_flags(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore user-requested incremental flags after base phase."""
        if not isinstance(state, dict):
            return
        for key, value in state.get("args", {}).items():
            try:
                setattr(self.args, key, value)
            except Exception:
                pass
        for key, value in state.get("model", {}).items():
            try:
                setattr(self.model, key, value)
            except Exception:
                pass
        # NECIL-HSI incremental/replay scoring should remain feature-only unless
        # the user explicitly configured otherwise.
        if not hasattr(self.args, "incremental_classifier_mode") and hasattr(self.model, "default_incremental_classifier_mode"):
            self.model.default_incremental_classifier_mode = "geometry_only"
        if not hasattr(self.args, "eval_classifier_mode") and hasattr(self.model, "default_eval_classifier_mode"):
            self.model.default_eval_classifier_mode = "geometry_only"
        self._assert_runtime_state_restored(state)

    def _force_base_validation_geometry_only(self) -> Dict[str, Any]:
        """Temporarily make validation/certificate measure feature geometry.

        Base CE/SRPGR can use physical spectra, but the incremental phase consumes
        GeometryBank rows mostly through geometry-only replay/admission. A base
        certificate based on PG-RGA spectral scoring can hide a dirty feature
        geometry field. This guard makes validation honest.
        """
        state: Dict[str, Any] = {"args_eval": None, "model_eval": None}
        if hasattr(self.args, "eval_classifier_mode"):
            state["args_eval"] = getattr(self.args, "eval_classifier_mode")
            setattr(self.args, "eval_classifier_mode", "geometry_only")
        if hasattr(self.model, "default_eval_classifier_mode"):
            state["model_eval"] = getattr(self.model, "default_eval_classifier_mode")
            self.model.default_eval_classifier_mode = "geometry_only"
        return state

    def _restore_base_validation_mode(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return
        if state.get("args_eval", None) is not None and hasattr(self.args, "eval_classifier_mode"):
            setattr(self.args, "eval_classifier_mode", state["args_eval"])
        if state.get("model_eval", None) is not None and hasattr(self.model, "default_eval_classifier_mode"):
            self.model.default_eval_classifier_mode = state["model_eval"]

    def _set_base_trainability(self, base_head: nn.Module) -> None:
        """
        Base phase should train representation + temporary CE head only.

        It must not train geometry classifier/calibrator/adapters. The bank is
        rebuilt from z after updates; it is not a trainable memory module.
        """
        self._enforce_base_contract()
        blocked_prefixes = (
            "classifier.",
            "geometry_bank.",
            "geometry_calibrator.",
            "geometry_cycle_calibrator.",
            "incremental_adapter.",
            "geometry_plastic_adapter.",
            "base_ce_head.",
        )
        blocked_keywords = ("bicyc", "transport", "calibrator")
        for name, p in self.model.named_parameters():
            lname = name.lower()
            blocked = name.startswith(blocked_prefixes) or any(k in lname for k in blocked_keywords)
            p.requires_grad = not blocked

        for p in base_head.parameters():
            p.requires_grad = True

        if hasattr(self.model, "freeze_geometry_plastic_adapter"):
            self.model.freeze_geometry_plastic_adapter()
        elif hasattr(self.model, "disable_incremental_adapter"):
            self.model.disable_incremental_adapter()
        if hasattr(self.model, "freeze_energy_calibrator"):
            self.model.freeze_energy_calibrator()
        if hasattr(self.model, "freeze_geometry_calibrator"):
            self.model.freeze_geometry_calibrator()

    def _base_trainable_parameters(self, base_head: nn.Module) -> list[nn.Parameter]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        params.extend([p for p in base_head.parameters() if p.requires_grad])
        # deduplicate in case a future model stores the same head internally
        unique: list[nn.Parameter] = []
        seen = set()
        for p in params:
            pid = id(p)
            if pid not in seen:
                unique.append(p)
                seen.add(pid)
        if not unique:
            raise RuntimeError("No trainable parameters found for base phase.")
        return unique

    # ============================================================
    # Label mapping / base head
    # ============================================================
    def _build_base_label_map(self, phase_class_ids: Iterable[int]) -> Dict[int, int]:
        return {int(c): i for i, c in enumerate([int(x) for x in phase_class_ids])}

    def _labels_to_local(self, labels: torch.Tensor, label_map: Dict[int, int]) -> torch.Tensor:
        labels = labels.long().view(-1)
        local = torch.full_like(labels, -1)
        for global_cls, local_cls in label_map.items():
            local[labels == int(global_cls)] = int(local_cls)
        if (local < 0).any():
            bad = labels[local < 0].detach().cpu().unique().tolist()
            raise RuntimeError(f"Base batch contains labels outside phase-0 classes: {bad}")
        return local

    def _extract_feature_dim(self, train_loader) -> int:
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            for batch in train_loader:
                x, _, _, _ = self._unpack_hsi_batch(batch)
                x = x.float().to(self.device, non_blocking=True)
                out = self._call_extract_projected_features(x)
                features = out["features"]
                if features.dim() != 2:
                    raise RuntimeError(f"Projected features must be [B,D], got {tuple(features.shape)}")
                if was_training:
                    self.model.train()
                return int(features.size(1))
        if was_training:
            self.model.train()
        raise RuntimeError("Cannot infer projected feature dimension: train_loader is empty.")

    def _make_base_ce_head(self, feature_dim: int, num_base_classes: int) -> nn.Module:
        head = nn.Linear(int(feature_dim), int(num_base_classes), bias=True).to(self.device)
        nn.init.normal_(head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(head.bias)
        return head

    def _batch_balanced_ce(self, logits: torch.Tensor, labels_local: torch.Tensor, num_classes: int) -> torch.Tensor:
        label_smoothing = self._base_cfg_float("label_smoothing", 0.0)
        use_balance = self._base_cfg_bool("base_class_balance", True)
        if not use_balance:
            return F.cross_entropy(logits, labels_local, label_smoothing=label_smoothing)

        counts = torch.bincount(labels_local, minlength=int(num_classes)).float().to(logits.device)
        weights = torch.zeros_like(counts)
        valid = counts > 0
        weights[valid] = 1.0 / counts[valid].sqrt().clamp_min(1.0)
        weights = weights / weights[valid].mean().clamp_min(1e-6)
        return F.cross_entropy(logits, labels_local, weight=weights, label_smoothing=label_smoothing)

    # ============================================================
    # GICS key view
    # ============================================================
    def _augment_gics_key_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mild HSI-safe augmentation for the detached key branch.

        Default is identity. Do not use aggressive band dropping; PCA/band axes
        carry material information and corrupting them makes GICS learn nonsense.
        """
        if x is None or not torch.is_tensor(x) or x.numel() == 0:
            return x

        x_key = x.clone()
        noise_std = self._base_cfg_float("base_gics_key_noise_std", 0.0)
        scale_jitter = self._base_cfg_float("base_gics_key_scale_jitter", 0.0)
        band_drop = self._base_cfg_float("base_gics_key_band_drop", 0.0)
        spatial_drop = self._base_cfg_float("base_gics_key_spatial_drop", 0.0)

        if noise_std > 0.0:
            x_key = x_key + torch.randn_like(x_key) * float(noise_std)

        if scale_jitter > 0.0:
            shape = [x_key.size(0)] + [1] * (x_key.dim() - 1)
            scale = 1.0 + float(scale_jitter) * torch.randn(shape, device=x_key.device, dtype=x_key.dtype)
            x_key = x_key * scale

        if band_drop > 0.0 and x_key.dim() >= 4 and x_key.size(1) > 1:
            keep = (torch.rand(x_key.size(0), x_key.size(1), 1, 1, device=x_key.device) > float(band_drop)).to(x_key.dtype)
            all_zero = keep.flatten(1).sum(dim=1) <= 0
            if bool(all_zero.any().item()):
                keep[all_zero, 0, 0, 0] = 1.0
            x_key = x_key * keep

        if spatial_drop > 0.0 and x_key.dim() >= 4 and x_key.size(-2) > 1 and x_key.size(-1) > 1:
            smask = (torch.rand(x_key.size(0), 1, x_key.size(-2), x_key.size(-1), device=x_key.device) > float(spatial_drop)).to(x_key.dtype)
            x_key = x_key * smask

        return x_key

    @staticmethod
    def _band_summary_from_spectral(spectral_summary: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Fallback band signature used only when the model did not expose band_summary.

        The updated NECILModel should normally provide center-pixel band_summary.
        This fallback is intentionally conservative: signed/PCA-like vectors are
        converted by softmax, non-negative rows are sum-normalized, and degenerate
        rows become uniform.  It is a reliability/geometry-conflict signal, not classifier input.
        """
        if spectral_summary is None or not torch.is_tensor(spectral_summary) or spectral_summary.numel() == 0:
            return None
        if spectral_summary.dim() != 2 or spectral_summary.size(1) <= 0:
            return None
        s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
        if bool((s < 0).any().item()):
            return torch.softmax(s, dim=1)
        b = s.clamp_min(0.0)
        denom = b.sum(dim=1, keepdim=True)
        uniform = torch.full_like(b, 1.0 / float(max(int(b.size(1)), 1)))
        return torch.where(denom > 1e-8, b / denom.clamp_min(1e-8), uniform)


    def _select_srpgr_band_summary(
        self,
        *,
        spectral_summary: Optional[torch.Tensor],
        band_summary: Optional[torch.Tensor],
        spectral_summary_is_physical: bool,
        x: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Select the band descriptor used by PGR.

        In reduced-band experiments, the model input is PCA/reduced channels
        while the dataloader may also provide raw wavelength-ordered spectra.
        The base GeometryBank certificate checks physical band/spectral overlap,
        so the PGR band reserve must operate on the physical spectrum whenever it
        is available.  A 30-D PCA vector must not compete with a 200-D physical
        descriptor.
        """
        if (
            torch.is_tensor(spectral_summary)
            and spectral_summary.dim() == 2
            and spectral_summary.numel() > 0
            and bool(spectral_summary_is_physical)
        ):
            return spectral_summary

        # If band_summary and spectral_summary disagree in dimensionality, prefer
        # the one that matches the physical/reliability source.  With PCA input
        # this avoids passing a reduced component vector into a physical-band
        # reserve by accident.
        if (
            torch.is_tensor(band_summary)
            and torch.is_tensor(spectral_summary)
            and band_summary.dim() == 2
            and spectral_summary.dim() == 2
            and band_summary.size(0) == spectral_summary.size(0)
            and band_summary.size(1) != spectral_summary.size(1)
        ):
            pca_components = int(getattr(self.args, "pca_components", 0) or 0)
            uses_reduction = pca_components > 0 and not bool(getattr(self.args, "no_pca", False))
            input_channels = int(x.size(1)) if torch.is_tensor(x) and x.dim() >= 2 else 0
            if uses_reduction and input_channels > 0 and band_summary.size(1) == input_channels:
                return spectral_summary

        if torch.is_tensor(band_summary) and band_summary.dim() == 2 and band_summary.numel() > 0:
            return band_summary
        return self._band_summary_from_spectral(spectral_summary)

    def _base_spectral_summary_is_physical(self, explicit: Optional[bool] = None) -> bool:
        """Whether spectral_summary is wavelength-ordered raw HSI, not PCA.

        Spectral derivatives are physically meaningful only for ordered bands.
        If the pipeline uses PCA and no raw center spectra are supplied, this
        returns False and the spectral-shape part of SRPGR becomes a safe zero.
        """
        if explicit is not None:
            return bool(explicit)
        if hasattr(self.args, "base_spectral_summary_is_physical"):
            return self._base_cfg_bool("base_spectral_summary_is_physical", False)
        if hasattr(self.args, "spectral_summary_is_physical"):
            return self._base_cfg_bool("spectral_summary_is_physical", False)
        pca = int(getattr(self.args, "pca_components", 0) or 0)
        allow_nonphysical = self._base_cfg_bool("allow_nonphysical_spectral_summary", False)
        if pca > 0 and not allow_nonphysical:
            return False
        return self._base_cfg_bool("assume_input_spectral_order_is_physical", False)

    def _normalize_external_spectra(self, spectra: Optional[torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return external spectra as [B,S] when the loader provides raw center spectra."""
        if spectra is None or not torch.is_tensor(spectra) or spectra.numel() == 0:
            return None
        s = spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
        if s.dim() == 4:
            # [B,S,H,W] raw cube: center pixel only.
            s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
        elif s.dim() == 3:
            # [B,S,L] raw/reduced spectral metadata: center token only. Flattening
            # would mix neighboring pixels into the class spectrum and poison
            # spectral-shape regularization.
            if s.size(0) == x.size(0) and s.size(1) > 0 and s.size(2) > 1:
                s = s[:, :, s.size(-1) // 2]
            elif s.size(0) == x.size(0):
                s = s.reshape(s.size(0), -1)
        elif s.dim() == 1:
            s = s.view(x.size(0), -1)
        elif s.dim() != 2:
            s = s.flatten(1)
        if s.size(0) != x.size(0):
            raise RuntimeError(f"External spectral summary batch mismatch: spectra={tuple(s.shape)}, x={tuple(x.shape)}")
        return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _center_spectrum_from_input(x: torch.Tensor) -> torch.Tensor:
        """Fallback center spectrum from model input. Usually PCA, so not physical by default."""
        if x.dim() == 4:
            return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
        if x.dim() == 3:
            return x.flatten(1)
        if x.dim() == 2:
            return x
        return x.flatten(1)

    def _call_extract_projected_features(
        self,
        x: torch.Tensor,
        *,
        spectral_summary: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return canonical pre-adapter projected z for base phase.

        Base GeometryBank rows must be built from the same canonical z-space used
        by CE/GICS/PGR.  If NECILModel exposes an explicit canonical extractor,
        prefer it so the geometry-plastic adapter can never leak into phase 0.
        """
        physical = self._base_spectral_summary_is_physical(spectral_summary_is_physical)
        if hasattr(self.model, "extract_canonical_projected_features"):
            return self.model.extract_canonical_projected_features(
                x,
                spectral_summary=spectral_summary,
                spectral_summary_is_physical=physical,
            )
        try:
            out = self.model.extract_projected_features(
                x,
                spectral_summary=spectral_summary,
                spectral_summary_is_physical=physical,
            )
        except TypeError:
            out = self.model.extract_projected_features(x)
        # Backward compatibility with models that return both canonical and
        # adapted features: force the canonical tensors into the public keys.
        if isinstance(out, dict) and torch.is_tensor(out.get("canonical_projected_features", None)):
            out = dict(out)
            z = out["canonical_projected_features"]
            out["features"] = z
            out["projected_features"] = z
        elif isinstance(out, dict) and torch.is_tensor(out.get("canonical_features", None)):
            out = dict(out)
            z = out["canonical_features"]
            out["features"] = z
            out["projected_features"] = z
        return out

    def _base_spectral_shape_regularizer(
        self,
        *,
        features: torch.Tensor,
        labels: torch.Tensor,
        spectral_summary: Optional[torch.Tensor],
        spectral_summary_is_physical: bool,
    ) -> Dict[str, torch.Tensor]:
        """Spectral-shape component of SRPGR.

        This is a safe no-op unless physical raw spectra are available and the
        PG-RGA loss file exposes spectral_shape_discrimination_loss.
        """
        if spectral_shape_discrimination_loss is None:
            z = self._zero(features)
            return {"total": z, "spectral_shape": z, "mean_similarity": z, "pair_count": z, "valid_class_count": z}
        weight = self._base_cfg_float("base_spectral_shape_weight", 0.05)
        if weight <= 0.0:
            z = self._zero(features)
            return {"total": z, "spectral_shape": z, "mean_similarity": z, "pair_count": z, "valid_class_count": z}
        parts = spectral_shape_discrimination_loss(
            spectral_summary=spectral_summary,
            labels=labels,
            features=features,
            spectral_summary_is_physical=bool(spectral_summary_is_physical),
            require_physical_summary=self._base_cfg_bool("spectral_require_physical_summary", True),
            min_samples=self._base_cfg_int("pgr_min_class_samples", 3),
            max_shape_similarity=self._base_cfg_float("base_max_spectral_shape_similarity", 0.75),
            risk_center_margin=self._base_cfg_float("pgr_center_margin", 1.05),
            risk_weight=self._base_cfg_float("base_spectral_shape_risk_weight", 1.0),
            return_parts=True,
        )
        if not isinstance(parts, dict):
            parts = {"total": parts, "spectral_shape": parts.detach()}
        raw_total = parts.get("total", self._zero(features))
        parts["raw_total"] = raw_total.detach()
        parts["total"] = float(weight) * raw_total
        return parts

    def _extract_base_views(
        self,
        x: torch.Tensor,
        external_spectra: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        bool,
    ]:
        """Return query/key views for CE + unified SRPGR.

        The spectral summary is physical only when raw wavelength-ordered spectra
        are supplied by the dataloader. If the model input is PCA, the fallback
        center vector may still be used for band diagnostics, but not for
        spectral-shape derivatives.
        """
        ext_s = self._normalize_external_spectra(external_spectra, x)
        if ext_s is not None:
            input_s = ext_s

            # Reduced-band safety rule.  If metadata spectra have the same
            # channel count as the model input while PCA/iPCA is active, they are
            # reduced components, not physical wavelength-ordered spectra.
            # Treat them as non-physical so SRPGR spectral-derivative terms are
            # gated off and the model never mixes 30-D band weights with 200-D
            # raw spectra.
            pca_components = int(getattr(self.args, "pca_components", 0) or 0)
            uses_reduction = pca_components > 0 and not bool(getattr(self.args, "no_pca", False))
            input_channels = int(x.size(1)) if torch.is_tensor(x) and x.dim() >= 2 else 0
            spectra_dim = int(input_s.size(1)) if torch.is_tensor(input_s) and input_s.dim() == 2 else 0
            reduced_metadata = bool(uses_reduction and input_channels > 0 and spectra_dim == input_channels)

            if reduced_metadata:
                input_s_is_physical = False
            else:
                input_s_is_physical = self._base_spectral_summary_is_physical(
                    self._base_cfg_bool("external_spectra_are_physical", True)
                )
        else:
            input_s = self._center_spectrum_from_input(x)
            input_s_is_physical = self._base_spectral_summary_is_physical(None)

        out_q = self._call_extract_projected_features(
            x,
            spectral_summary=input_s,
            spectral_summary_is_physical=input_s_is_physical,
        )
        z_q = out_q["features"]
        if z_q.dim() != 2 or not torch.isfinite(z_q).all():
            raise RuntimeError(f"Invalid projected query features: {tuple(z_q.shape)}")
        if "projected_features" in out_q and torch.is_tensor(out_q["projected_features"]):
            if not torch.allclose(z_q, out_q["projected_features"], atol=1e-5, rtol=1e-4):
                raise RuntimeError(
                    "Base feature-space mismatch: out['features'] and out['projected_features'] differ. "
                    "SRPGR/GeometryBank must all use the same canonical projected z-space."
                )

        s_q = out_q.get("spectral_summary", None)
        if not (torch.is_tensor(s_q) and s_q.dim() == 2 and s_q.size(0) == x.size(0)):
            s_q = input_s
        spectral_flag = out_q.get("spectral_summary_is_physical", input_s_is_physical)
        if torch.is_tensor(spectral_flag):
            spectral_is_physical = bool(spectral_flag.detach().cpu().item())
        else:
            spectral_is_physical = bool(spectral_flag)
        b_q = out_q.get("band_summary", out_q.get("band_importance", None))
        if b_q is None:
            b_q = self._band_summary_from_spectral(s_q)

        x_key = self._augment_gics_key_view(x)
        # The key view uses the same spectral summary because physical spectral
        # identity belongs to the center pixel/label, not to the mild feature-view
        # augmentation. This avoids corrupting spectral-shape targets.
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            out_k = self._call_extract_projected_features(
                x_key,
                spectral_summary=input_s,
                spectral_summary_is_physical=spectral_is_physical,
            )
        if was_training:
            self.model.train()

        z_k = out_k["features"].detach()
        if "projected_features" in out_k and torch.is_tensor(out_k["projected_features"]):
            if not torch.allclose(out_k["features"], out_k["projected_features"], atol=1e-5, rtol=1e-4):
                raise RuntimeError("SRPGR key branch is not in canonical projected z-space.")
        s_k = out_k.get("spectral_summary", None)
        if not (torch.is_tensor(s_k) and s_k.dim() == 2 and s_k.size(0) == x.size(0)):
            s_k = s_q
        b_k = out_k.get("band_summary", out_k.get("band_importance", None))
        if b_k is None:
            b_k = self._band_summary_from_spectral(s_k)
        if torch.is_tensor(s_k):
            s_k = s_k.detach()
        if torch.is_tensor(b_k):
            b_k = b_k.detach()

        if z_k.shape != z_q.shape:
            raise RuntimeError(f"SRPGR key/query mismatch: query={tuple(z_q.shape)}, key={tuple(z_k.shape)}")
        if not torch.isfinite(z_k).all():
            raise RuntimeError("Projected key features contain NaN/Inf.")
        return z_q, s_q, b_q, z_k, s_k, b_k, spectral_is_physical

    @torch.no_grad()
    def _rebuild_base_geometry_bank_for_validation(
        self,
        phase: int,
        phase_class_ids: Iterable[int],
        *,
        split: str = "train",
    ) -> None:
        """Rebuild base GeometryBank rows from canonical z before geometry validation.

        Base validation uses the geometry-energy classifier, so the bank must be
        synchronized with the current representation every epoch. This method
        bypasses refresh_before_validation because base geometry validation is
        meaningless with stale or unbuilt rows.
        """
        if not hasattr(self, "_build_class_memory_from_current_phase"):
            raise AttributeError("TrainerHelper._build_class_memory_from_current_phase() is required.")
        old_training_state = bool(self.model.training)
        ctx = self.dataset.memory_build_context(int(phase)) if hasattr(self.dataset, "memory_build_context") else None
        if ctx is None:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            for cls in [int(c) for c in phase_class_ids]:
                self._build_class_memory_from_current_phase(cls, split=split)
        self.model.train(old_training_state)

    # ============================================================
    # Diagnostics
    # ============================================================
    @torch.no_grad()
    def _base_geometry_global_metrics(self) -> Dict[str, float]:
        """Bank-level diagnostics for checkpoint scoring.

        Supports both the old bank name ``geometry_diagnostics`` and the cleaned
        descriptor bank name ``compute_geometry_diagnostics``.
        """
        gb = getattr(self.model, "geometry_bank", None)
        if gb is None:
            return {}
        try:
            if hasattr(gb, "compute_geometry_diagnostics"):
                diag = gb.compute_geometry_diagnostics()
            elif hasattr(gb, "geometry_diagnostics"):
                diag = gb.geometry_diagnostics()
            else:
                diag = {}
        except Exception:
            diag = {}
        out: Dict[str, float] = {}
        aliases = {
            "feature_subspace_overlap": ("feature_subspace_overlap", "mean_subspace_overlap", "subspace_overlap_mean"),
            "band_overlap": ("band_overlap", "mean_band_similarity", "band_similarity_mean"),
            "spectral_shape_overlap": ("spectral_shape_overlap", "mean_spectral_similarity", "spectral_similarity_mean"),
            "spectral_shape_similarity_mean": ("spectral_shape_similarity_mean", "mean_spectral_similarity", "spectral_similarity_mean"),
            "spectral_shape_similarity_max": ("spectral_shape_similarity_max", "max_spectral_similarity", "spectral_similarity_max"),
            "geometry_conflict_mean": ("geometry_conflict_mean", "mean_geometry_conflict", "conflict_mean"),
            "geometry_conflict_max": ("geometry_conflict_max", "max_geometry_conflict", "conflict_max"),
            "geometry_reserve_score": ("geometry_reserve_score", "reserve_score"),
            "feature_rank_usage": ("feature_rank_usage", "rank_usage"),
        }
        for public_key, candidates in aliases.items():
            for key in candidates:
                value = diag.get(key, None) if isinstance(diag, dict) else None
                if torch.is_tensor(value) and value.numel() == 1:
                    out[public_key] = float(value.detach().cpu().item())
                    break
                if isinstance(value, (int, float)):
                    out[public_key] = float(value)
                    break
        return out

    @staticmethod
    def _scalar(value: Any, default: float = 0.0) -> float:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().float().mean().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    @torch.no_grad()
    def _build_base_geometry_certificate(
        self,
        phase_class_ids: Iterable[int],
        *,
        val_stats: Optional[Dict[str, Any]] = None,
        train_stats: Optional[Dict[str, Any]] = None,
        head_train_acc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create the real base-to-incremental geometry certificate.

        This is not a print-only artifact. It is JSON/PT serializable and becomes
        part of the phase-0 checkpoint/handoff consumed by the incremental
        trainer for replay strength, insertion margin, and transport activation.
        """
        classes = [int(c) for c in phase_class_ids]
        cert: Dict[str, Any] = {
            "phase": 0,
            "class_ids": classes,
            "base_class_ids": classes,
            "num_base_classes": len(classes),
            "valid": False,
            "errors": [],
            "warnings": [],
        }
        if not classes:
            cert["errors"].append("no base classes")
            return cert

        try:
            self._assert_base_bank_valid(classes)
            bank = self._get_valid_base_bank(max(classes))
        except Exception as exc:
            cert["errors"].append(f"bank_invalid: {exc}")
            return cert

        idx = torch.as_tensor(classes, device=self.device, dtype=torch.long)
        gb = getattr(self.model, "geometry_bank", None)
        counts = bank.get("sample_counts")
        ranks = bank.get("active_ranks")
        rel = bank.get("reliability")
        variances = bank.get("variances")
        means = bank.get("means")

        def _take(t: Optional[torch.Tensor], fill: float = 0.0) -> torch.Tensor:
            if not torch.is_tensor(t) or t.numel() <= int(idx.max().item()):
                return torch.full((len(classes),), float(fill), device=self.device, dtype=torch.float32)
            return t.to(device=self.device).flatten().index_select(0, idx).float()

        count_v = _take(counts, 0.0)
        rank_v = _take(ranks, 0.0)
        rel_v = _take(rel, 0.0)
        if torch.is_tensor(variances) and variances.dim() == 2 and variances.size(0) > int(idx.max().item()):
            res_v = variances.to(device=self.device).index_select(0, idx)[:, -1].float()
            eig_v = variances.to(device=self.device).index_select(0, idx)[:, :-1].float()
        else:
            res_v = torch.zeros((len(classes),), device=self.device)
            eig_v = torch.zeros((len(classes), 0), device=self.device)

        center = self._pairwise_matrix_or_zeros(gb, ("pairwise_center_distance",), idx) if gb is not None else torch.zeros((len(classes), len(classes)), device=self.device)
        sub = self._pairwise_matrix_or_zeros(gb, ("pairwise_subspace_overlap",), idx) if gb is not None else torch.zeros_like(center)
        band = self._pairwise_matrix_or_zeros(gb, ("pairwise_band_similarity",), idx) if gb is not None else torch.zeros_like(center)
        spec = self._pairwise_matrix_or_zeros(gb, ("pairwise_spectral_similarity", "pairwise_spectral_shape_similarity"), idx) if gb is not None else torch.zeros_like(center)
        conflict = self._pairwise_matrix_or_zeros(gb, ("geometry_conflict_matrix",), idx) if gb is not None else torch.zeros_like(center)

        eye = torch.eye(len(classes), device=self.device, dtype=torch.bool)
        def _pair_mean_max(mat: torch.Tensor) -> Tuple[float, float, int]:
            if mat.numel() == 0 or mat.size(0) < 2:
                return 0.0, 0.0, 0
            pair = mat[~eye]
            pair = pair[torch.isfinite(pair)]
            if pair.numel() == 0:
                return 0.0, 0.0, 0
            return float(pair.mean().cpu().item()), float(pair.max().cpu().item()), int(pair.numel())

        center_mean, center_min, _ = 0.0, 0.0, 0
        if center.numel() > 0 and center.size(0) >= 2:
            pair_center = center[~eye]
            if pair_center.numel() > 0:
                center_mean = float(pair_center.mean().cpu().item())
                center_min = float(pair_center.min().cpu().item())
        sub_mean, sub_max, pair_count = _pair_mean_max(sub)
        band_mean, band_max, _ = _pair_mean_max(band)
        spec_mean, spec_max, spec_pair_count = _pair_mean_max(spec)
        conflict_mean, conflict_max, _ = _pair_mean_max(conflict)
        geom = self._base_geometry_global_metrics()

        cert.update({
            "feature_dim": int(means.size(1)) if torch.is_tensor(means) and means.dim() == 2 else int(getattr(self.model, "d_model", 0)),
            "geom_val_acc": float((val_stats or {}).get("acc", 0.0)),
            "geom_val_loss": float((val_stats or {}).get("loss", 0.0)),
            "geom_train_acc": float((train_stats or {}).get("acc", 0.0)),
            "head_train_acc": float(head_train_acc) if head_train_acc is not None else 0.0,
            "valid_geometry_rows": classes,
            "valid_row_count": int((count_v > 0).sum().detach().cpu().item()),
            "sample_counts": {int(c): float(count_v[i].detach().cpu().item()) for i, c in enumerate(classes)},
            "reliability": {int(c): float(rel_v[i].detach().cpu().item()) for i, c in enumerate(classes)},
            "active_rank": {int(c): int(rank_v[i].detach().cpu().item()) for i, c in enumerate(classes)},
            "residual_variance": {int(c): float(res_v[i].detach().cpu().item()) for i, c in enumerate(classes)},
            "eigenvalue_mean": {int(c): float(eig_v[i].mean().detach().cpu().item()) if eig_v.numel() else 0.0 for i, c in enumerate(classes)},
            "min_sample_count": float(count_v.min().detach().cpu().item()) if count_v.numel() else 0.0,
            "mean_sample_count": float(count_v.mean().detach().cpu().item()) if count_v.numel() else 0.0,
            "min_reliability": float(rel_v.min().detach().cpu().item()) if rel_v.numel() else 0.0,
            "mean_reliability": float(rel_v.mean().detach().cpu().item()) if rel_v.numel() else 0.0,
            "mean_active_rank": float(rank_v.mean().detach().cpu().item()) if rank_v.numel() else 0.0,
            "max_active_rank": float(rank_v.max().detach().cpu().item()) if rank_v.numel() else 0.0,
            "mean_res_var": float(res_v.mean().detach().cpu().item()) if res_v.numel() else 0.0,
            "max_res_var": float(res_v.max().detach().cpu().item()) if res_v.numel() else 0.0,
            "pair_count": pair_count,
            "mean_center_distance": center_mean,
            "min_center_distance": center_min,
            "mean_subspace_overlap": sub_mean,
            "max_subspace_overlap": sub_max,
            "mean_band_similarity": band_mean,
            "max_band_similarity": band_max,
            "mean_spectral_shape_similarity": spec_mean,
            "max_spectral_shape_similarity": spec_max,
            "spectral_shape_pair_count": spec_pair_count,
            "mean_geometry_conflict": conflict_mean,
            "max_geometry_conflict": conflict_max,
            "geometry_reserve_score": float(geom.get("geometry_reserve_score", max(0.0, 1.0 - conflict_max - sub_max))),
            "feature_rank_usage": float(geom.get("feature_rank_usage", 0.0)),
            "center_distance_matrix": center.detach().cpu().tolist(),
            "subspace_overlap_matrix": sub.detach().cpu().tolist(),
            "band_similarity_matrix": band.detach().cpu().tolist(),
            "spectral_similarity_matrix": spec.detach().cpu().tolist(),
            "geometry_conflict_matrix": conflict.detach().cpu().tolist(),
        })

        thr = {
            "min_geom_acc": self._base_cfg_float("base_cert_min_geom_acc", 95.0),
            "min_reliability": self._base_cfg_float("base_cert_min_reliability", 0.15),
            "min_mean_reliability": self._base_cfg_float("base_cert_min_mean_reliability", 0.35),
            "max_subspace_overlap": self._base_cfg_float("base_cert_max_subspace_overlap", 0.55),
            "max_geometry_conflict": self._base_cfg_float("base_cert_max_geometry_conflict", 1.35),
            "max_band_similarity": self._base_cfg_float("base_cert_max_band_similarity", 0.90),
            "max_spectral_shape_similarity": self._base_cfg_float("base_cert_max_spectral_shape_similarity", 0.85),
        }
        cert["thresholds"] = thr
        self._compute_pair_risks(cert, center, sub, conflict, band, spec)
        self._add_handoff_recommendations(cert)
        checks = {
            "all_base_rows_valid": cert["valid_row_count"] == len(classes),
            "geom_acc_ok": cert["geom_val_acc"] >= thr["min_geom_acc"],
            "min_reliability_ok": cert["min_reliability"] >= thr["min_reliability"],
            "mean_reliability_ok": cert["mean_reliability"] >= thr["min_mean_reliability"],
            "subspace_overlap_ok": cert["max_subspace_overlap"] <= thr["max_subspace_overlap"],
            "geometry_conflict_ok": cert["max_geometry_conflict"] <= thr["max_geometry_conflict"],
            "band_similarity_ok": cert["max_band_similarity"] <= thr["max_band_similarity"],
            "spectral_shape_similarity_ok": (
                int(cert.get("spectral_shape_pair_count", 0)) == 0
                or cert["max_spectral_shape_similarity"] <= thr["max_spectral_shape_similarity"]
            ),
        }
        cert["checks"] = checks
        cert["valid"] = bool(all(checks.values()) and len(cert["errors"]) == 0)
        if not cert["valid"]:
            cert["errors"].extend([k for k, ok in checks.items() if not ok])
        return cert

    def _print_base_geometry_certificate(self, certificate: Dict[str, Any]) -> None:
        print("[Base Geometry Certificate]")
        if not certificate:
            print("  unavailable")
            return
        status = "PASS" if bool(certificate.get("valid", False)) else "WARN/FAIL"
        print(f"  status={status} | classes={certificate.get('class_ids', [])}")
        print(
            "  "
            f"GeomValAcc={float(certificate.get('geom_val_acc', 0.0)):.2f}% | "
            f"valid_rows={int(certificate.get('valid_row_count', 0))}/{int(certificate.get('num_base_classes', 0))} | "
            f"rel(min/mean)={float(certificate.get('min_reliability', 0.0)):.3f}/{float(certificate.get('mean_reliability', 0.0)):.3f} | "
            f"subspace(max/mean)={float(certificate.get('max_subspace_overlap', 0.0)):.3f}/{float(certificate.get('mean_subspace_overlap', 0.0)):.3f} | "
            f"band(max/mean)={float(certificate.get('max_band_similarity', 0.0)):.3f}/{float(certificate.get('mean_band_similarity', 0.0)):.3f} | "
            f"spec(max/mean)={float(certificate.get('max_spectral_shape_similarity', 0.0)):.3f}/{float(certificate.get('mean_spectral_shape_similarity', 0.0)):.3f} | "
            f"conflict(max/mean)={float(certificate.get('max_geometry_conflict', 0.0)):.3f}/{float(certificate.get('mean_geometry_conflict', 0.0)):.3f}"
        )
        if certificate.get("errors"):
            print(f"  failed_checks={certificate.get('errors')}")
        if certificate.get("warnings") and self.debug:
            print(f"  warnings={certificate.get('warnings')}")

    def _enforce_base_geometry_certificate(self, certificate: Dict[str, Any]) -> None:
        """Warn by default; optionally hard-stop if the user requests strict gating."""
        self._last_base_geometry_certificate = certificate
        try:
            setattr(self.model, "base_geometry_certificate", certificate)
        except Exception:
            pass
        self._print_base_geometry_certificate(certificate)
        if bool(certificate.get("valid", False)):
            return
        msg = (
            "Base geometry certificate failed. Incremental training will be high-risk because "
            "base geometry is not clean enough for descriptor insertion. Failed checks: "
            f"{certificate.get('errors', [])}"
        )
        if self._base_cfg_bool("enforce_base_geometry_certificate", False):
            raise RuntimeError(msg)
        print(f"[Base Geometry Certificate WARN] {msg}")

    def _get_valid_base_bank(self, max_class_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "_safe_get_subspace_bank"):
            raise AttributeError("TrainerHelper._safe_get_subspace_bank() is required.")
        bank = self._safe_get_subspace_bank()
        required = ("means", "bases", "variances", "sample_counts")
        missing = [k for k in required if k not in bank or not torch.is_tensor(bank[k]) or bank[k].numel() == 0]
        if missing:
            raise RuntimeError(f"Base GeometryBank is not ready; missing/empty keys: {missing}")
        if max_class_id is not None and int(max_class_id) >= bank["means"].size(0):
            raise RuntimeError(f"Bank has {bank['means'].size(0)} rows but needs class id {int(max_class_id)}")
        return bank

    @torch.no_grad()
    def _print_base_geometry_diagnostics(self, phase_class_ids: Iterable[int]) -> None:
        if not self.debug and not bool(getattr(self.args, "print_base_geometry_diagnostics", True)):
            return
        try:
            bank = self._get_valid_base_bank()
        except Exception as exc:
            print(f"[Base Geometry Diagnostics] unavailable: {exc}")
            return

        counts = bank.get("sample_counts", None)
        active_ranks = bank.get("active_ranks", None)
        reliability = bank.get("reliability", None)
        variances = bank.get("variances", None)
        means = bank.get("means", None)
        bands = bank.get("band_importances", bank.get("band_importance", None))
        spec_rel = bank.get("spectral_shape_reliability", None)

        print("[Base Geometry Diagnostics]")
        print("  cls | count | rank | rel  | spec-rel | resvar   | mean_norm | band_entropy")
        for cls in [int(c) for c in phase_class_ids]:
            count = float(counts[cls].detach().item()) if torch.is_tensor(counts) and counts.numel() > cls else -1.0
            rank = int(active_ranks[cls].detach().item()) if torch.is_tensor(active_ranks) and active_ranks.numel() > cls else -1
            rel = float(reliability[cls].detach().item()) if torch.is_tensor(reliability) and reliability.numel() > cls else -1.0
            srel = float(spec_rel[cls].detach().item()) if torch.is_tensor(spec_rel) and spec_rel.numel() > cls else -1.0
            rv = float(variances[cls, -1].detach().item()) if torch.is_tensor(variances) and variances.size(0) > cls else -1.0
            mn = float(means[cls].norm().detach().item()) if torch.is_tensor(means) and means.size(0) > cls else -1.0
            bent = -1.0
            if torch.is_tensor(bands) and bands.dim() == 2 and bands.size(0) > cls:
                b = bands[cls].detach().float().clamp_min(0.0)
                b = b / b.sum().clamp_min(1e-8)
                bent = float((-(b * torch.log(b.clamp_min(1e-8))).sum()).cpu().item())
            print(f"  {cls:3d} | {count:5.0f} | {rank:4d} | {rel:4.2f} | {srel:8.3f} | {rv:8.5f} | {mn:9.4f} | {bent:12.4f}")

        geom = self._base_geometry_global_metrics()
        if geom:
            print("  " + " | ".join(f"{k}={v:.4f}" for k, v in geom.items()))

    @torch.no_grad()
    def _batch_base_overlap_diagnostics(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        try:
            diag = base_center_overlap_diagnostics(
                features,
                labels,
                normalize=True,
                min_samples=self._base_cfg_int("pgr_min_class_samples", 3),
            )
        except Exception:
            return {"batch_compact": 0.0, "batch_center_margin": 0.0, "batch_min_center_margin": 0.0}
        return {
            "batch_compact": float(diag.get("compact", self._zero(features)).detach().cpu().item()),
            "batch_center_margin": float(diag.get("mean_center_margin", self._zero(features)).detach().cpu().item()),
            "batch_min_center_margin": float(diag.get("min_center_margin", self._zero(features)).detach().cpu().item()),
        }

    # ============================================================
    # Clean PRL-style public structure / assertions / handoff
    # ============================================================
    def configure_base_contract(self, phase: int = 0) -> None:
        if int(phase) != 0:
            raise ValueError("BasePhaseTrainer.configure_base_contract() is phase-0 only.")
        self._enforce_base_contract()

    def snapshot_runtime_state(self) -> Dict[str, Any]:
        return self._capture_incremental_runtime_flags()

    def restore_runtime_state(self, state: Optional[Dict[str, Any]]) -> None:
        self._restore_incremental_runtime_flags(state)

    def build_base_label_map(self, phase_class_ids: Iterable[int]) -> Dict[int, int]:
        return self._build_base_label_map(phase_class_ids)

    def configure_base_trainability(self, base_head: nn.Module) -> None:
        self._set_base_trainability(base_head)
        self._assert_forbidden_modules_frozen()
    def build_temporary_ce_head(self, feature_dim: int, num_base_classes: int) -> nn.Module:
        return self._make_base_ce_head(feature_dim, num_base_classes)

    def extract_base_features(self, x: torch.Tensor, external_spectra: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        z, s, b, zk, sk, bk, physical = self._extract_base_views(x, external_spectra)
        self._assert_feature_space_consistency({"features": z, "projected_features": z})
        return {
            "features": z,
            "projected_features": z,
            "spectral_summary": s,
            "band_summary": b,
            "key_features": zk,
            "key_spectral_summary": sk,
            "key_band_summary": bk,
            "spectral_summary_is_physical": bool(physical),
        }

    def train_one_base_epoch(self, *args, **kwargs) -> Tuple[float, float]:
        return self._train_epoch_base_geometry(*args, **kwargs)

    @torch.no_grad()
    def rebuild_base_geometry_bank(self, phase: int, phase_class_ids: Iterable[int], *, split: str = "train") -> None:
        self._rebuild_base_geometry_bank_for_validation(int(phase), phase_class_ids, split=split)
        self._assert_base_bank_valid([int(c) for c in phase_class_ids])

    @torch.no_grad()
    def validate_geometry_only(self, loader, *, old_class_count: int = 0) -> Dict[str, Any]:
        state = self._force_base_validation_geometry_only()
        try:
            return self._validate_split_metrics(loader, old_class_count=int(old_class_count))
        finally:
            self._restore_base_validation_mode(state)

    @torch.no_grad()
    def compute_base_certificate(self, phase_class_ids: Iterable[int], **kwargs: Any) -> Dict[str, Any]:
        return self._build_base_geometry_certificate(phase_class_ids, **kwargs)

    def compute_checkpoint_score(self, val_stats: Dict[str, Any], geom_stats: Optional[Dict[str, Any]] = None) -> float:
        return self._select_base_checkpoint_score(val_stats, geom_stats)

    def _jsonify(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): self._jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonify(v) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _diagnostic_output_dir(self) -> str:
        for name in ("save_dir", "output_dir", "results_dir", "log_dir"):
            value = getattr(self.args, name, None)
            if value:
                path = os.path.abspath(str(value))
                os.makedirs(path, exist_ok=True)
                return path
        path = os.path.abspath("./results")
        os.makedirs(path, exist_ok=True)
        return path

    def _assert_runtime_state_restored(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return
        failures: List[str] = []
        for key, value in state.get("args", {}).items():
            if hasattr(self.args, key) and getattr(self.args, key) != value:
                failures.append(f"args.{key}: expected {value!r}, got {getattr(self.args, key)!r}")
        for key, value in state.get("model", {}).items():
            if hasattr(self.model, key) and getattr(self.model, key) != value:
                failures.append(f"model.{key}: expected {value!r}, got {getattr(self.model, key)!r}")
        if failures:
            raise RuntimeError("Base phase failed to restore runtime state: " + "; ".join(failures))

    def _assert_feature_space_consistency(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "features" not in out:
            raise RuntimeError("Feature output must contain 'features'.")
        features = out["features"]
        projected = out.get("projected_features", features)
        if not torch.is_tensor(features) or features.dim() != 2:
            raise RuntimeError(f"features must be [B,D], got {tuple(features.shape) if torch.is_tensor(features) else type(features)}")
        if not torch.is_tensor(projected) or projected.shape != features.shape:
            raise RuntimeError("projected_features must exist and match features shape.")
        if not torch.allclose(features, projected, atol=1e-5, rtol=1e-4):
            raise RuntimeError("features and projected_features are not the same canonical z tensor.")
        if not torch.isfinite(features).all():
            raise RuntimeError("canonical base features contain NaN/Inf.")
        gb = getattr(self.model, "geometry_bank", None)
        bank_dim = int(getattr(gb, "feature_dim", getattr(gb, "d_model", features.size(1)))) if gb is not None else features.size(1)
        if int(features.size(1)) != int(bank_dim):
            raise RuntimeError(f"feature dim {features.size(1)} does not match GeometryBank feature_dim={bank_dim}.")
        return features

    def _assert_base_batch_labels(self, labels_global: torch.Tensor, labels_local: torch.Tensor, phase_class_ids: Iterable[int], num_base: int) -> None:
        if labels_global.numel() == 0:
            raise RuntimeError("Base batch is empty.")
        allowed = {int(c) for c in phase_class_ids}
        bad = sorted(set(int(x) for x in labels_global.detach().cpu().view(-1).tolist()).difference(allowed))
        if bad:
            raise RuntimeError(f"Base batch contains non-base/future/background labels: {bad}; allowed={sorted(allowed)}")
        if labels_local.numel() != labels_global.numel():
            raise RuntimeError("Local/global label count mismatch.")
        if int(labels_local.min().item()) < 0 or int(labels_local.max().item()) >= int(num_base):
            raise RuntimeError(f"Local CE labels out of range [0,{int(num_base)-1}].")

    def _assert_base_ce_logits(self, logits: torch.Tensor, labels_local: torch.Tensor, num_base: int) -> None:
        if not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError(f"Base CE logits must be [B,C], got {tuple(logits.shape) if torch.is_tensor(logits) else type(logits)}")
        if logits.size(0) != labels_local.numel() or logits.size(1) != int(num_base):
            raise RuntimeError(f"Base CE logits shape {tuple(logits.shape)} incompatible with labels={labels_local.numel()} and num_base={num_base}.")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Base CE logits contain NaN/Inf.")

    def _assert_finite_loss(self, value: torch.Tensor, name: str) -> None:
        if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
            raise RuntimeError(f"{name} must be a finite scalar tensor.")

    @torch.no_grad()
    def _assert_base_bank_valid(self, phase_class_ids: Iterable[int]) -> None:
        classes = [int(c) for c in phase_class_ids]
        if not classes:
            raise RuntimeError("No base classes for GeometryBank validation.")
        gb = getattr(self.model, "geometry_bank", None)
        if gb is None:
            raise RuntimeError("Model has no GeometryBank.")
        if hasattr(gb, "assert_bank_valid"):
            gb.assert_bank_valid(seen_classes=classes, strict=True)
        bank = self._get_valid_base_bank(max(classes))
        idx = torch.as_tensor(classes, device=self.device, dtype=torch.long)
        means = bank["means"].index_select(0, idx)
        bases = bank["bases"].index_select(0, idx)
        variances = bank["variances"].index_select(0, idx)
        counts = bank["sample_counts"].to(device=self.device).flatten().index_select(0, idx)
        if not torch.isfinite(means).all():
            raise RuntimeError("Base GeometryBank means contain NaN/Inf.")
        if not torch.isfinite(bases).all():
            raise RuntimeError("Base GeometryBank bases contain NaN/Inf.")
        if not torch.isfinite(variances).all():
            raise RuntimeError("Base GeometryBank variances contain NaN/Inf.")
        if not bool((counts > 0).all().item()):
            raise RuntimeError("Every base class must have positive GeometryBank sample_count.")
        if bool((variances < -1e-8).any().item()):
            raise RuntimeError("GeometryBank eigen/residual variances must be non-negative.")
        # Orthonormality check for active columns. Zero-padded inactive columns are allowed.
        ranks = bank.get("active_ranks", None)
        if torch.is_tensor(ranks):
            ranks = ranks.to(device=self.device).long().flatten().index_select(0, idx)
        else:
            ranks = torch.full((len(classes),), bases.size(2), device=self.device, dtype=torch.long)
        eye_cache: Dict[int, torch.Tensor] = {}
        for row, r_t in enumerate(ranks):
            r = int(r_t.item())
            if r <= 0:
                continue
            U = bases[row, :, :r]
            eye = eye_cache.setdefault(r, torch.eye(r, device=U.device, dtype=U.dtype))
            err = (U.t() @ U - eye).abs().max()
            if float(err.detach().cpu().item()) > 5e-3:
                raise RuntimeError(f"Base GeometryBank basis for class {classes[row]} is not orthonormal; max_err={float(err):.4e}")
        max_cls = max(classes)
        counts_full = bank["sample_counts"].to(device=self.device).flatten()
        if counts_full.numel() > max_cls + 1:
            future_valid = counts_full[max_cls + 1:] > 0
            if bool(future_valid.any().item()):
                bad = (torch.nonzero(future_valid, as_tuple=False).flatten() + max_cls + 1).detach().cpu().tolist()
                raise RuntimeError(f"Future/non-base classes have valid geometry after base phase: {bad}")

    def _assert_forbidden_modules_frozen(self) -> None:
        forbidden = ("classifier.", "geometry_bank.", "geometry_calibrator", "calibrator", "adapter", "transport", "cycle")
        offenders = []
        for name, p in self.model.named_parameters():
            lname = name.lower()
            if any(tok in lname for tok in forbidden) and p.requires_grad:
                # Backbone/projection names may contain no forbidden token. Classifier/adapters must be frozen.
                offenders.append(name)
        if offenders:
            raise RuntimeError("Forbidden base-phase modules are trainable: " + ", ".join(offenders[:20]))

    def _assert_base_ce_head_discarded(self) -> None:
        if getattr(self, "_base_ce_head", None) is not None:
            raise RuntimeError("Temporary base CE head is still attached to trainer after base finalization.")
        if hasattr(self.model, "base_ce_head") and getattr(self.model, "base_ce_head") is not None:
            raise RuntimeError("Temporary model base_ce_head was not dropped after base finalization.")

    def _pairwise_matrix_or_zeros(self, gb: Any, method_names: Iterable[str], idx: torch.Tensor) -> torch.Tensor:
        for name in method_names:
            if hasattr(gb, name):
                mat = getattr(gb, name)()
                if torch.is_tensor(mat) and mat.dim() == 2 and mat.size(0) > int(idx.max().item()) and mat.size(1) > int(idx.max().item()):
                    return mat.to(device=self.device).index_select(0, idx).index_select(1, idx).float()
        n = int(idx.numel())
        return torch.zeros((n, n), device=self.device, dtype=torch.float32)

    def _compute_pair_risks(self, cert: Dict[str, Any], center: torch.Tensor, sub: torch.Tensor, conflict: torch.Tensor, band: torch.Tensor, spec: torch.Tensor) -> None:
        classes = [int(c) for c in cert.get("class_ids", [])]
        n = len(classes)
        thr = cert.get("thresholds", {})
        unsafe_pairs = []
        risk_by_class = {int(c): 0.0 for c in classes}
        for i in range(n):
            for j in range(i + 1, n):
                risk = 0.0
                reasons: List[str] = []
                sub_ij = float(max(sub[i, j].item(), sub[j, i].item())) if sub.numel() else 0.0
                conf_ij = float(max(conflict[i, j].item(), conflict[j, i].item())) if conflict.numel() else 0.0
                band_ij = float(max(band[i, j].item(), band[j, i].item())) if band.numel() else 0.0
                spec_ij = float(max(spec[i, j].item(), spec[j, i].item())) if spec.numel() else 0.0
                dist_ij = float(center[i, j].item()) if center.numel() else 0.0
                if sub_ij > float(thr.get("max_subspace_overlap", 0.55)):
                    risk += sub_ij; reasons.append("subspace")
                if conf_ij > float(thr.get("max_geometry_conflict", 1.35)):
                    risk += conf_ij; reasons.append("geometry_conflict")
                if band_ij > float(thr.get("max_band_similarity", 0.90)):
                    risk += 0.5 * band_ij; reasons.append("band")
                if spec_ij > float(thr.get("max_spectral_shape_similarity", 0.85)):
                    risk += 0.5 * spec_ij; reasons.append("spectral_shape")
                if reasons:
                    item = {
                        "class_i": classes[i],
                        "class_j": classes[j],
                        "risk": float(risk),
                        "reasons": reasons,
                        "center_distance": dist_ij,
                        "subspace_overlap": sub_ij,
                        "geometry_conflict": conf_ij,
                        "band_similarity": band_ij,
                        "spectral_similarity": spec_ij,
                    }
                    unsafe_pairs.append(item)
                    risk_by_class[classes[i]] = max(float(risk_by_class[classes[i]]), float(risk))
                    risk_by_class[classes[j]] = max(float(risk_by_class[classes[j]]), float(risk))
        unsafe_pairs = sorted(unsafe_pairs, key=lambda x: x["risk"], reverse=True)
        for item in unsafe_pairs:
            item["geometry_conflict_score"] = float(item.get("risk", 0.0))
        cert["unsafe_class_pairs"] = unsafe_pairs

        # New PG-RGA naming.  Keep risk_by_class as a backward-compatible alias
        # for older handoff readers, but all new reports should use conflict.
        geometry_conflict_by_class = {int(k): float(v) for k, v in risk_by_class.items()}
        cert["geometry_conflict_by_class"] = geometry_conflict_by_class
        cert["risk_by_class"] = geometry_conflict_by_class
        cert["high_conflict_old_classes"] = [int(c) for c, v in geometry_conflict_by_class.items() if float(v) > 0.0]
        cert["high_risk_old_classes"] = cert["high_conflict_old_classes"]

    def _add_handoff_recommendations(self, cert: Dict[str, Any]) -> None:
        conflict_by_class = {int(k): float(v) for k, v in cert.get("geometry_conflict_by_class", cert.get("risk_by_class", {})).items()}
        base_replay = self._base_cfg_int("base_recommend_replay_per_class", 16)
        max_replay = self._base_cfg_int("base_recommend_max_replay_per_class", 64)
        replay_plan = {}
        for cls in cert.get("class_ids", []):
            conflict = conflict_by_class.get(int(cls), 0.0)
            replay_plan[int(cls)] = int(min(max_replay, max(base_replay, round(base_replay * (1.0 + conflict)))))
        max_conflict = max(conflict_by_class.values()) if conflict_by_class else 0.0
        margin_base = self._base_cfg_float("base_recommend_insertion_margin", 1.0)
        cert["recommended_replay_per_class"] = replay_plan
        cert["recommended_insertion_margin"] = float(margin_base * (1.0 + 0.25 * max_conflict))
        # Main PG-RGA path does not use old-row transport.  The handoff should
        # tell the incremental trainer how strong replay/alignment margins should
        # be, not activate a transport module.
        cert["recommended_transport_activation"] = False
        cert["transport_required"] = False
        cert["recommended_reserved_alignment"] = bool(
            max_conflict > self._base_cfg_float("base_reserved_alignment_conflict_threshold", self._base_cfg_float("base_reserved_alignment_risk_threshold", 0.50))
            or float(cert.get("max_geometry_conflict", 0.0)) > self._base_cfg_float("base_cert_max_geometry_conflict", 1.35)
            or float(cert.get("max_subspace_overlap", 0.0)) > self._base_cfg_float("base_cert_max_subspace_overlap", 0.55)
        )
        cert["recommended_old_new_energy_margin"] = float(
            self._base_cfg_float("base_recommend_old_new_energy_margin", 0.30) * (1.0 + 0.25 * max_conflict)
        )

    @torch.no_grad()
    def finalize_base_handoff(self, phase_class_ids: Iterable[int], certificate: Dict[str, Any], history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        classes = [int(c) for c in phase_class_ids]
        feature_dim = int(getattr(getattr(self.model, "geometry_bank", None), "feature_dim", getattr(self.model, "d_model", 0)))
        handoff = {
            "phase": 0,
            "base_classes": classes,
            "feature_dim": feature_dim,
            "geometry_bank_ready": bool(certificate.get("valid_row_count", 0) == len(classes)),
            "geometry_certificate": certificate,
            "geometry_conflict_by_class": certificate.get("geometry_conflict_by_class", certificate.get("risk_by_class", {})),
            "risk_by_class": certificate.get("risk_by_class", {}),  # backward-compatible alias
            "unsafe_class_pairs": certificate.get("unsafe_class_pairs", []),
            "recommended_replay_per_class": certificate.get("recommended_replay_per_class", {}),
            "recommended_insertion_margin": certificate.get("recommended_insertion_margin", 1.0),
            "transport_required": False,
            "reserved_alignment_required": bool(certificate.get("recommended_reserved_alignment", False)),
            "recommended_old_new_energy_margin": certificate.get("recommended_old_new_energy_margin", 0.30),
            "high_conflict_old_classes": certificate.get("high_conflict_old_classes", certificate.get("high_risk_old_classes", [])),
            "high_risk_old_classes": certificate.get("high_risk_old_classes", []),  # backward-compatible alias
        }
        out_dir = self._diagnostic_output_dir()
        json_path = os.path.join(out_dir, "phase_0_base_handoff.json")
        pt_path = os.path.join(out_dir, "phase_0_base_handoff.pt")
        cert_path = os.path.join(out_dir, "phase_0_base_geometry_certificate.json")
        diag_path = os.path.join(out_dir, "phase_0_base_diagnostics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._jsonify(handoff), f, indent=2)
        with open(cert_path, "w", encoding="utf-8") as f:
            json.dump(self._jsonify(certificate), f, indent=2)
        if history is not None:
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(self._jsonify(history), f, indent=2)
        torch.save(self._jsonify(handoff), pt_path)
        handoff["handoff_json_path"] = json_path
        handoff["handoff_pt_path"] = pt_path
        handoff["certificate_json_path"] = cert_path
        handoff["diagnostics_json_path"] = diag_path if history is not None else None
        try:
            setattr(self.model, "base_handoff", handoff)
            setattr(self.model, "base_geometry_certificate", certificate)
        except Exception:
            pass
        return handoff

    # ============================================================
    # Base objective: unified_spectral_geometry_loss(phase="base")
    # ============================================================
    def _train_epoch_base_geometry(
        self,
        loader,
        optimizer,
        base_head: nn.Module,
        phase_class_ids: Iterable[int],
        label_map: Dict[int, int],
        trainable_params: list[nn.Parameter],
    ) -> Tuple[float, float]:
        self._assert_mandatory_base_objective_config()
        self.model.train()
        base_head.train()

        num_base = len([int(c) for c in phase_class_ids])
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        stat_steps = 0
        stat_sums = {
            "ce": 0.0,
            "srpgr": 0.0,
            "compact_sep": 0.0,
            "gics": 0.0,
            "weighted_gics": 0.0,
            "gics_anchors": 0.0,
            "gics_pos": 0.0,
            "pgr": 0.0,
            "pgr_unweighted": 0.0,
            "pgr_compact": 0.0,
            "pgr_center": 0.0,
            "pgr_subspace": 0.0,
            "pgr_band": 0.0,
            "pgr_volume": 0.0,
            "pgr_valid_class_count": 0.0,
            "pgr_subspace_pair_count": 0.0,
            "pgr_band_pair_count": 0.0,
            "pgr_volume_factor": 0.0,
            "pgr_subspace_max_overlap": 0.0,
            "pgr_band_max_similarity": 0.0,
            "spectral_shape": 0.0,
            "spectral_shape_raw": 0.0,
            "spectral_shape_mean_similarity": 0.0,
            "spectral_shape_pairs": 0.0,
            "spectral_active": 0.0,
            "batch_compact": 0.0,
            "batch_center_margin": 0.0,
            "batch_min_center_margin": 0.0,
        }

        for batch in loader:
            x, y, spectra, _ = self._unpack_hsi_batch(batch)
            x = x.float().to(self.device, non_blocking=True)
            y = y.long().to(self.device, non_blocking=True)
            y_local = self._labels_to_local(y, label_map)
            self._assert_base_batch_labels(y, y_local, phase_class_ids, num_base)

            optimizer.zero_grad(set_to_none=True)

            features, spectral_summary, band_summary, key_features, _, _, spectral_summary_is_physical = self._extract_base_views(x, spectra)
            self._assert_feature_space_consistency({"features": features, "projected_features": features})

            # PGR band reserve must use the physical spectrum when raw spectra are
            # available.  This is the fix for the previous 200-D raw spectrum vs
            # 30-D PCA band-vector mismatch and for weak band/spectral certificates.
            band_for_srpgr = self._select_srpgr_band_summary(
                spectral_summary=spectral_summary,
                band_summary=band_summary,
                spectral_summary_is_physical=bool(spectral_summary_is_physical),
                x=x,
            )

            self._assert_mandatory_base_batch_inputs(
                features=features,
                labels_local=y_local,
                key_features=key_features,
                spectral_summary=spectral_summary,
                band_summary=band_for_srpgr,
            )
            if torch.is_tensor(key_features) and key_features.requires_grad:
                raise RuntimeError("SRPGR/GICS key_features must be detached; otherwise the key branch becomes a second train path.")
            logits = base_head(features)
            self._assert_base_ce_logits(logits, y_local, num_base)

            # Architecture fix: base CE must be class-balanced independently of
            # SRPGR. Indian Pines style base phases contain tiny classes; plain
            # CE can learn a high-accuracy head while leaving minority geometry
            # dirty. The unified loss still owns SRPGR; CE is computed here so
            # class-balanced CE can be used without changing the public loss API.
            loss_out = unified_spectral_geometry_loss(
                phase="base",
                logits=logits,
                features=features,
                labels=y_local,
                key_features=key_features,
                band_summary=band_for_srpgr,
                spectral_summary=spectral_summary,
                spectral_summary_is_physical=bool(spectral_summary_is_physical),
                ce_weight=0.0,
                base_geometry_weight=self._base_cfg_float("base_srpgr_weight", 1.0),
                label_smoothing=self._base_cfg_float("label_smoothing", 0.0),
                gics_weight=self._base_cfg_float("base_gics_weight", 0.20),
                pgr_weight=self._base_cfg_float("pgr_weight", 0.10),
                spectral_shape_weight=self._base_cfg_float("base_spectral_shape_weight", 0.05),
                gics_temperature=self._base_cfg_float("base_gics_temperature", 0.07),
                pgr_compact_weight=self._base_cfg_float("pgr_compact_weight", 0.15),
                pgr_center_weight=self._base_cfg_float("pgr_center_weight", 0.20),
                pgr_subspace_weight=self._base_cfg_float("pgr_subspace_weight", 0.10),
                pgr_band_weight=self._base_cfg_float("pgr_band_weight", 0.05),
                pgr_volume_weight=self._base_cfg_float("pgr_volume_weight", 0.05),
                pgr_center_margin=self._base_cfg_float("pgr_center_margin", 1.05),
                pgr_max_band_similarity=self._base_cfg_float("pgr_band_overlap_max", 0.75),
                pgr_max_class_variance=self._base_cfg_float("pgr_max_class_variance", 0.75),
                pgr_min_class_variance=self._base_cfg_float("pgr_min_class_variance", 0.015),
                pgr_max_subspace_overlap=self._base_cfg_float("pgr_max_subspace_overlap", self._base_cfg_float("base_cert_max_subspace_overlap", 0.55)),
                subspace_overlap_max=self._base_cfg_float("pgr_max_subspace_overlap", self._base_cfg_float("base_cert_max_subspace_overlap", 0.55)),
                subspace_rank=self._base_cfg_int("pgr_subspace_rank", 3),
                min_class_samples=self._base_cfg_int("pgr_min_class_samples", 3),
                subspace_min_samples=self._base_cfg_int("pgr_subspace_min_samples", 6),
                max_spectral_shape_similarity=self._base_cfg_float("base_max_spectral_shape_similarity", 0.75),
                spectral_shape_risk_weight=self._base_cfg_float("base_spectral_shape_risk_weight", 1.0),
                require_physical_summary=self._base_cfg_bool("spectral_require_physical_summary", True),
                return_parts=True,
            )
            if not isinstance(loss_out, dict) or "total" not in loss_out:
                raise RuntimeError("unified_spectral_geometry_loss(phase='base') must return a dict containing 'total'.")
            self._assert_mandatory_base_loss_parts(
                loss_out,
                spectral_summary_is_physical=bool(spectral_summary_is_physical),
            )
            ce = self._batch_balanced_ce(logits, y_local, num_base)
            # Important: the cleaned loss returns base_geometry as a detached
            # logging value.  The differentiable SRPGR/GICS/PGR graph is in
            # loss_out["total"] because ce_weight=0.0 above.
            srpgr_total = loss_out.get("total", self._zero(features))
            if not torch.is_tensor(srpgr_total):
                srpgr_total = torch.as_tensor(float(srpgr_total), device=features.device, dtype=features.dtype)
            loss = self._base_cfg_float("base_ce_weight", 1.0) * ce + srpgr_total
            loss_out["ce"] = ce.detach()
            loss_out["base_geometry_train"] = srpgr_total.detach()
            loss_out["total"] = loss
            self._assert_finite_loss(ce, "base_ce")
            self._assert_finite_loss(srpgr_total, "base_srpgr")
            self._assert_finite_loss(loss, "base_total_loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(self.grad_clip_base))
            optimizer.step()

            pred = logits.argmax(dim=1)
            total_loss += float(loss.detach().item())
            total_correct += int((pred == y_local).sum().item())
            total_count += int(y_local.numel())
            stat_steps += 1

            batch_diag = self._batch_base_overlap_diagnostics(features.detach(), y.detach())

            def _lstat(key: str, default: float = 0.0) -> float:
                value = loss_out.get(key, None)
                if torch.is_tensor(value):
                    return float(value.detach().float().mean().cpu().item())
                if isinstance(value, (int, float)):
                    return float(value)
                return float(default)

            stat_sums["ce"] += _lstat("ce")
            stat_sums["srpgr"] += _lstat("base_geometry_train", _lstat("base_geometry"))
            stat_sums["compact_sep"] += _lstat("base_gics")
            stat_sums["gics"] += _lstat("base_gics")
            stat_sums["weighted_gics"] += _lstat("base_gics_weighted", _lstat("base_gics"))
            stat_sums["gics_anchors"] += _lstat("base_gics_anchors")
            stat_sums["gics_pos"] += _lstat("base_gics_pos")
            stat_sums["pgr"] += _lstat("base_pgr_weighted", _lstat("base_pgr"))
            stat_sums["pgr_unweighted"] += _lstat("base_pgr")
            stat_sums["pgr_compact"] += _lstat("base_compact")
            stat_sums["pgr_center"] += _lstat("base_center")
            stat_sums["pgr_subspace"] += _lstat("base_subspace")
            stat_sums["pgr_band"] += _lstat("base_band")
            stat_sums["pgr_volume"] += _lstat("base_volume")
            stat_sums["pgr_valid_class_count"] += _lstat("base_pgr_valid_class_count")
            stat_sums["pgr_subspace_pair_count"] += _lstat("base_pgr_subspace_pair_count")
            stat_sums["pgr_band_pair_count"] += _lstat("base_pgr_band_pair_count")
            stat_sums["pgr_volume_factor"] += _lstat("base_pgr_volume_factor")
            stat_sums["pgr_subspace_max_overlap"] += _lstat("base_pgr_subspace_max_overlap")
            stat_sums["pgr_band_max_similarity"] += _lstat("base_pgr_band_max_similarity")
            stat_sums["spectral_shape"] += _lstat("base_spectral_shape")
            stat_sums["spectral_shape_raw"] += _lstat("base_spectral_shape_raw", _lstat("base_spectral_shape"))
            stat_sums["spectral_shape_mean_similarity"] += _lstat("base_spectral_shape_mean_similarity")
            stat_sums["spectral_shape_pairs"] += _lstat("base_spectral_shape_pair_count")
            stat_sums["spectral_active"] += _lstat("base_spectral_shape_active", 1.0 if bool(spectral_summary_is_physical) else 0.0)
            for k, v in batch_diag.items():
                stat_sums[k] += float(v)

        self._last_base_stats = {k: v / max(stat_steps, 1) for k, v in stat_sums.items()}
        self._assert_mandatory_base_epoch_coverage(self._last_base_stats)
        return total_loss / max(stat_steps, 1), 100.0 * total_correct / max(total_count, 1)

    # backward-compatible alias for trainer.py calls that still use the old name
    def _train_epoch_base_gics(self, *args, **kwargs):
        return self._train_epoch_base_geometry(*args, **kwargs)

    # ============================================================
    # Checkpoint score
    # ============================================================
    def _select_base_checkpoint_score(self, val_stats: Dict, geom_stats: Optional[Dict] = None) -> float:
        metric = str(getattr(self.args, "best_state_metric", "geometry_score")).lower()
        geom_stats = geom_stats or {}
        if metric in {"geometry_score", "geo", "reserve"}:
            acc = float(val_stats.get("acc", 0.0))
            reserve = float(geom_stats.get("geometry_reserve_score", val_stats.get("geometry_score", 0.0)))
            conflict_mean = float(geom_stats.get("geometry_conflict_mean", 0.0))
            conflict_max = float(val_stats.get("cert_max_geometry_conflict", geom_stats.get("geometry_conflict_max", conflict_mean)))
            overlap_mean = float(geom_stats.get("feature_subspace_overlap", 0.0))
            overlap_max = float(val_stats.get("cert_max_subspace_overlap", overlap_mean))
            band_max = float(val_stats.get("cert_max_band_similarity", geom_stats.get("band_overlap", 0.0)))
            spec_max = float(val_stats.get("cert_max_spectral_shape_similarity", geom_stats.get("spectral_shape_similarity_max", geom_stats.get("spectral_shape_overlap", 0.0))))
            cert_valid = 1.0 if bool(val_stats.get("cert_valid", False)) else 0.0
            a = self._base_cfg_float("base_score_reserve_alpha", 8.0)
            b_conf_max = self._base_cfg_float("base_score_conflict_max_beta", 7.0)
            b_conf_mean = self._base_cfg_float("base_score_conflict_mean_beta", 3.0)
            g_overlap_max = self._base_cfg_float("base_score_overlap_max_gamma", 6.0)
            g_overlap_mean = self._base_cfg_float("base_score_overlap_mean_gamma", 2.0)
            d_band = self._base_cfg_float("base_score_band_delta", 2.0)
            d_spec = self._base_cfg_float("base_score_spectral_delta", 1.5)
            cert_bonus = self._base_cfg_float("base_score_cert_bonus", 3.0)
            return (
                acc
                + a * reserve
                + cert_bonus * cert_valid
                - b_conf_max * conflict_max
                - b_conf_mean * conflict_mean
                - g_overlap_max * overlap_max
                - g_overlap_mean * overlap_mean
                - d_band * band_max
                - d_spec * spec_max
            )
        if metric in {"acc", "accuracy", "oa", "val_acc"}:
            return float(val_stats.get("acc", 0.0))
        if metric in {"loss", "val_loss"}:
            return -float(val_stats.get("loss", 1e9))
        if metric in {"hm", "harmonic"}:
            return float(val_stats.get("hm", val_stats.get("acc", 0.0)))
        return float(val_stats.get("acc", 0.0))

    def train_base_phase(self, phase, epochs, batch_size=64, lr=1e-4) -> Dict:
        phase = int(phase)
        if phase != 0:
            raise ValueError("train_base_phase() must only be called for phase 0.")

        incremental_runtime_state = self.snapshot_runtime_state()
        base_eval_state = None
        base_head: Optional[nn.Module] = None
        history: Dict[str, Any] = {}
        phase_class_ids: List[int] = []
        try:
            base_eval_state = self._force_base_validation_geometry_only()
            self.configure_base_contract(phase)

            print("==== Base Phase Training | PRL-style Geometry Preparation | Prospective Geometry Reserve ====")
            self.dataset.start_phase(phase)
            phase_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
            if not phase_class_ids:
                raise RuntimeError("Phase 0 has no base classes.")
            label_map = self.build_base_label_map(phase_class_ids)
            self._set_model_phase_and_old_count(phase, 0)

            needed_classes = max(phase_class_ids) + 1
            if hasattr(self.model, "ensure_class_capacity"):
                self.model.ensure_class_capacity(needed_classes)

            train_loader = self.dataset.get_phase_dataloader(phase, split="train", batch_size=batch_size, shuffle=True)
            val_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)

            feature_dim = self._extract_feature_dim(train_loader)
            base_head = self.build_temporary_ce_head(feature_dim, len(phase_class_ids))
            self._base_ce_head = base_head
            self.configure_base_trainability(base_head)
            trainable_params = self._base_trainable_parameters(base_head)

            optimizer = optim.Adam(
                trainable_params,
                lr=float(lr),
                weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

            history = {
                "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                "val_old_acc": [], "val_new_acc": [], "val_hm": [], "checkpoint_score": [],
                "base_ce": [], "base_srpgr": [], "base_compact_sep": [], "base_gics": [], "base_weighted_gics": [],
                "base_gics_anchors": [], "base_gics_pos": [],
                "base_pgr": [], "base_pgr_unweighted": [], "base_pgr_compact": [],
                "base_pgr_center": [], "base_pgr_subspace": [], "base_pgr_band": [], "base_pgr_volume": [],
                "base_pgr_valid_class_count": [], "base_pgr_subspace_pair_count": [], "base_pgr_band_pair_count": [], "base_pgr_volume_factor": [],
                "base_pgr_subspace_max_overlap": [], "base_pgr_band_max_similarity": [],
                "base_spectral_shape": [], "base_spectral_shape_raw": [], "base_spectral_shape_mean_similarity": [],
                "base_spectral_shape_pairs": [], "base_spectral_active": [],
                "batch_compact": [], "batch_center_margin": [], "batch_min_center_margin": [],
                "feature_subspace_overlap": [], "band_overlap": [], "spectral_shape_overlap": [],
                "spectral_shape_similarity_mean": [], "spectral_shape_similarity_max": [],
                "geometry_conflict_mean": [], "geometry_conflict_max": [], "geometry_reserve_score": [],
                "base_cert_geom_acc": [], "base_cert_min_reliability": [],
                "base_cert_max_subspace_overlap": [], "base_cert_max_band_similarity": [], "base_cert_max_spectral_shape_similarity": [],
                "base_cert_max_geometry_conflict": [], "base_cert_valid": [],
            }

            best_score = -1e18
            best_state = None
            no_improve = 0
            epochs = int(epochs)

            if hasattr(self, "_print_trainable_summary"):
                self._print_trainable_summary(phase)
            print(f"[Base] Temporary CE head: feature_dim={feature_dim}, classes={len(phase_class_ids)}. It will be discarded after geometry handoff.")
            print("[Base Objective] MANDATORY: balanced CE + GICS + PGR(compact, center, subspace, band, volume) on canonical projected z-space.")

            for epoch in range(epochs):
                self._base_epoch = int(epoch)
                self._set_model_phase_and_old_count(phase, 0)

                tr_loss, head_train_acc = self.train_one_base_epoch(
                    train_loader,
                    optimizer,
                    base_head,
                    phase_class_ids,
                    label_map,
                    trainable_params,
                )

                self.rebuild_base_geometry_bank(phase, phase_class_ids, split="train")

                train_eval_stats = self.validate_geometry_only(train_loader, old_class_count=0)
                val_stats = self.validate_geometry_only(val_loader, old_class_count=0)
                scheduler.step()

                base_stats = getattr(self, "_last_base_stats", {})
                geom_stats = self._base_geometry_global_metrics()
                epoch_cert = self.compute_base_certificate(
                    phase_class_ids,
                    val_stats=val_stats,
                    train_stats=train_eval_stats,
                    head_train_acc=head_train_acc,
                )

                history["train_loss"].append(float(tr_loss))
                history["train_acc"].append(float(train_eval_stats["acc"]))
                history["val_loss"].append(float(val_stats["loss"]))
                history["val_acc"].append(float(val_stats["acc"]))
                history["val_old_acc"].append(float(val_stats.get("old_acc", 0.0)))
                history["val_new_acc"].append(float(val_stats.get("new_acc", 0.0)))
                history["val_hm"].append(float(val_stats.get("hm", 0.0)))

                for k in (
                    "ce", "srpgr", "compact_sep", "gics", "weighted_gics", "gics_anchors", "gics_pos", "pgr", "pgr_unweighted",
                    "pgr_compact", "pgr_center", "pgr_subspace", "pgr_band", "pgr_volume",
                    "pgr_valid_class_count", "pgr_subspace_pair_count", "pgr_band_pair_count", "pgr_volume_factor",
                    "pgr_subspace_max_overlap", "pgr_band_max_similarity",
                    "spectral_shape", "spectral_shape_raw", "spectral_shape_mean_similarity", "spectral_shape_pairs", "spectral_active",
                    "batch_compact", "batch_center_margin", "batch_min_center_margin",
                ):
                    history_key = f"base_{k}" if k not in {"batch_compact", "batch_center_margin", "batch_min_center_margin"} else k
                    if history_key in history:
                        history[history_key].append(float(base_stats.get(k, 0.0)))

                for k in ("feature_subspace_overlap", "band_overlap", "spectral_shape_overlap", "spectral_shape_similarity_mean", "spectral_shape_similarity_max", "geometry_conflict_mean", "geometry_conflict_max", "geometry_reserve_score"):
                    history[k].append(float(geom_stats.get(k, 0.0)))

                history["base_cert_geom_acc"].append(float(epoch_cert.get("geom_val_acc", 0.0)))
                history["base_cert_min_reliability"].append(float(epoch_cert.get("min_reliability", 0.0)))
                history["base_cert_max_subspace_overlap"].append(float(epoch_cert.get("max_subspace_overlap", 0.0)))
                history["base_cert_max_band_similarity"].append(float(epoch_cert.get("max_band_similarity", 0.0)))
                history["base_cert_max_spectral_shape_similarity"].append(float(epoch_cert.get("max_spectral_shape_similarity", 0.0)))
                history["base_cert_max_geometry_conflict"].append(float(epoch_cert.get("max_geometry_conflict", 0.0)))
                history["base_cert_valid"].append(1.0 if bool(epoch_cert.get("valid", False)) else 0.0)

                score_stats = dict(val_stats)
                score_stats.update(geom_stats)
                score_stats.update({f"cert_{k}": v for k, v in epoch_cert.items() if isinstance(v, (int, float, bool))})
                score_stats["geometry_score"] = float(geom_stats.get("geometry_reserve_score", 0.0))
                score = self.compute_checkpoint_score(score_stats, geom_stats)
                history["checkpoint_score"].append(float(score))

                print(
                    f"[Base Prep] Epoch {epoch + 1:03d}/{epochs} | "
                    f"Loss={tr_loss:.4f} | HeadAcc={head_train_acc:.2f}% | "
                    f"GeomTrain={train_eval_stats['acc']:.2f}% | GeomVal={val_stats['acc']:.2f}% | "
                    f"CE={float(base_stats.get('ce', 0.0)):.4f} | "
                    f"Compact={float(base_stats.get('pgr_compact', 0.0)):.4f} | "
                    f"Sep={float(base_stats.get('pgr_center', 0.0)):.4f} | "
                    f"Subspace={float(base_stats.get('pgr_subspace', 0.0)):.4f} | "
                    f"Band={float(base_stats.get('pgr_band', 0.0)):.4f} | "
                    f"Volume={float(base_stats.get('pgr_volume', 0.0)):.4f} | "
                    f"SpecShape={float(base_stats.get('spectral_shape', 0.0)):.4f} | "
                    f"PGRSubMax={float(base_stats.get('pgr_subspace_max_overlap', 0.0)):.4f} | "
                    f"PGRBandMax={float(base_stats.get('pgr_band_max_similarity', 0.0)):.4f} | "
                    f"OverlapMax={float(epoch_cert.get('max_subspace_overlap', 0.0)):.4f} | "
                    f"ConflictMax={float(epoch_cert.get('max_geometry_conflict', 0.0)):.4f} | "
                    f"Reserve={float(epoch_cert.get('geometry_reserve_score', 0.0)):.4f} | "
                    f"Score={float(score):.4f} | Cert={'PASS' if bool(epoch_cert.get('valid', False)) else 'CONFLICT'}"
                )

                if score > best_score:
                    best_score = score
                    best_state = self._capture_state()
                    no_improve = 0
                else:
                    no_improve += 1

                if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
                    print(f"[EarlyStop] Base phase: no improvement for {no_improve} epochs.")
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)
                self._set_model_phase_and_old_count(phase, 0)

            print("[Base] Final GeometryBank rebuild from restored best canonical z-space.")
            self._finalize_phase_memory(phase, split="train")
            self._set_model_phase_and_old_count(phase, len(phase_class_ids))
            self._assert_base_bank_valid(phase_class_ids)

            # Freeze only after final rebuild and make the base handoff contract
            # explicit: base rows valid, future rows invalid, base rows frozen.
            if hasattr(self.model, "assert_base_handoff_ready"):
                self.model.assert_base_handoff_ready(phase_class_ids, freeze=True, strict=True)
            elif hasattr(self.model, "geometry_bank"):
                gb = self.model.geometry_bank
                if hasattr(gb, "assert_phase0_base_handoff_ready"):
                    gb.assert_phase0_base_handoff_ready(phase_class_ids, freeze=True, strict=True)
                elif hasattr(gb, "freeze_classes"):
                    gb.freeze_classes(phase_class_ids)
                elif hasattr(gb, "freeze_classes_up_to"):
                    gb.freeze_classes_up_to(len(phase_class_ids))
            self._assert_base_bank_valid(phase_class_ids)
            self._print_base_geometry_diagnostics(phase_class_ids)

            final_train_stats = self.validate_geometry_only(train_loader, old_class_count=0)
            final_val_stats = self.validate_geometry_only(val_loader, old_class_count=0)
            final_cert = self.compute_base_certificate(
                phase_class_ids,
                val_stats=final_val_stats,
                train_stats=final_train_stats,
                head_train_acc=history["train_acc"][-1] if history.get("train_acc") else None,
            )
            history["base_geometry_certificate"] = final_cert
            self._enforce_base_geometry_certificate(final_cert)
            base_handoff = self.finalize_base_handoff(phase_class_ids, final_cert, history)
            history["base_handoff"] = base_handoff

            if hasattr(self, "diagnose_full_base_geometry"):
                try:
                    self._last_base_geometry_diagnostics = self.diagnose_full_base_geometry(
                        loader=val_loader,
                        phase_class_ids=phase_class_ids,
                        anchors_per_class=int(getattr(self.args, "geometry_diag_anchors_per_class", 64)),
                        topk_pairs=int(getattr(self.args, "geometry_diag_topk_pairs", 20)),
                        topk_bands=int(getattr(self.args, "geometry_diag_topk_bands", 5)),
                    )
                    if hasattr(self, "_print_geometry_diagnostics_summary"):
                        self._print_geometry_diagnostics_summary(self._last_base_geometry_diagnostics)
                    if hasattr(self, "_save_geometry_diagnostics_to_files"):
                        saved_paths = self._save_geometry_diagnostics_to_files(self._last_base_geometry_diagnostics, phase=phase)
                        print(f"[Geometry Health] saved diagnostics: {saved_paths.get('json', '')}")
                except Exception as exc:
                    print(f"[Geometry Health WARN] could not create persistent diagnostics: {exc}")

            self._base_ce_head = None
            if hasattr(self.model, "drop_base_ce_head"):
                self.model.drop_base_ce_head()
            self._assert_base_ce_head_discarded()
            if hasattr(self, "save_checkpoint"):
                self.save_checkpoint(phase, history)
            print(
                f"[Base Handoff] saved: {base_handoff.get('handoff_json_path')} | "
                f"reserved_alignment_required={base_handoff.get('reserved_alignment_required')} | "
                f"recommended_margin={float(base_handoff.get('recommended_insertion_margin', 0.0)):.3f}"
            )
            return history
        finally:
            if base_eval_state is not None:
                self._restore_base_validation_mode(base_eval_state)
            self.restore_runtime_state(incremental_runtime_state)













# from __future__ import annotations

# from typing import Any, Dict, Iterable, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from losses.loss import (
#     base_geometry_involved_contrastive_loss,
#     prospective_geometry_reserve_loss,
#     base_center_overlap_diagnostics,
# )

# try:
#     from losses.loss import spectral_shape_discrimination_loss
# except Exception:  # pragma: no cover - backward compatibility with older loss files
#     spectral_shape_discrimination_loss = None


# class BasePhaseTrainer:
#     # ============================================================
#     # Config helpers
#     # ============================================================
#     def _base_cfg_float(self, name: str, default: float) -> float:
#         return float(getattr(self, name, getattr(self.args, name, default)))

#     def _base_cfg_int(self, name: str, default: int) -> int:
#         return int(getattr(self, name, getattr(self.args, name, default)))

#     def _base_cfg_bool(self, name: str, default: bool) -> bool:
#         v = getattr(self, name, getattr(self.args, name, default))
#         if isinstance(v, str):
#             return v.strip().lower() in {"1", "true", "yes", "y", "on"}
#         return bool(v)

#     def _zero(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if isinstance(ref, torch.Tensor):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     def _enforce_base_contract(self) -> None:
#         """Force the base phase to the SRGP geometry-construction path.

#         Base training is allowed to update the backbone/projection and a
#         temporary CE head. It is not allowed to train incremental adapters,
#         geometry transport modules, or calibration modules. Unlike the previous
#         cleaned trainer, this method does NOT disable spectral geometry. The
#         base phase must prepare the exact SRGP memory consumed later by the
#         incremental phase.
#         """
#         forced_false = (
#             "use_incremental_adapter",
#             "use_geometry_transport",
#             "use_geometry_calibrator",
#             "use_bicyc_geometry_cycle",
#             "use_descriptor_refinement",
#             "use_measured_energy_calibration",
#             "allow_incremental_projection_training",
#         )
#         for key in forced_false:
#             if hasattr(self.args, key):
#                 setattr(self.args, key, False)

#         # Keep the method coherent: GICS/PGR/SpectralShape are internal parts of
#         # SRPGR. Do not expose legacy spectral-GICS knobs as independent losses.
#         for key in (
#             "base_gics_spectral_weight",
#             "base_gics_band_weight",
#             "base_gics_spectral_temperature",
#             "base_gics_band_temperature",
#         ):
#             if hasattr(self.args, key):
#                 setattr(self.args, key, 0.0)

#         if hasattr(self.args, "use_spectral_geometry"):
#             setattr(self.args, "use_spectral_geometry", True)
#         if hasattr(self.args, "base_classifier_mode"):
#             setattr(self.args, "base_classifier_mode", "srgp")
#         if hasattr(self.args, "eval_classifier_mode"):
#             setattr(self.args, "eval_classifier_mode", "srgp")

#         if self._base_cfg_bool("use_mssl_loss", False) and self._base_cfg_float("mssl_weight", 0.0) > 0.0:
#             if not self._base_cfg_bool("unsafe_ablation_use_mssl_loss", False):
#                 raise RuntimeError(
#                     "SRGP base trainer uses one coherent objective: CE + SRPGR. "
#                     "MSSL must be a separate ablation, not mixed into the main method."
#                 )

#         if hasattr(self.model, "use_incremental_adapter"):
#             self.model.use_incremental_adapter = False
#         if hasattr(self.model, "use_bicyc_geometry_cycle"):
#             self.model.use_bicyc_geometry_cycle = False
#         if hasattr(self.model, "use_geometry_calibrator"):
#             self.model.use_geometry_calibrator = False
#         if hasattr(self.model, "use_spectral_geometry"):
#             self.model.use_spectral_geometry = True
#         if hasattr(self.model, "disable_incremental_adapter"):
#             self.model.disable_incremental_adapter()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()

#     # ============================================================
#     # Trainability
#     # ============================================================
#     def _set_base_trainability(self, base_head: nn.Module) -> None:
#         """
#         Base phase should train representation + temporary CE head only.

#         It must not train geometry classifier/calibrator/adapters. The bank is
#         rebuilt from z after updates; it is not a trainable memory module.
#         """
#         self._enforce_base_contract()
#         blocked_prefixes = (
#             "classifier.",
#             "geometry_bank.",
#             "geometry_calibrator.",
#             "geometry_cycle_calibrator.",
#             "incremental_adapter.",
#             "geometry_plastic_adapter.",
#             "base_ce_head.",
#         )
#         blocked_keywords = ("bicyc", "transport", "calibrator")
#         for name, p in self.model.named_parameters():
#             lname = name.lower()
#             blocked = name.startswith(blocked_prefixes) or any(k in lname for k in blocked_keywords)
#             p.requires_grad = not blocked

#         for p in base_head.parameters():
#             p.requires_grad = True

#         if hasattr(self.model, "disable_incremental_adapter"):
#             self.model.disable_incremental_adapter()
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()

#     def _base_trainable_parameters(self, base_head: nn.Module) -> list[nn.Parameter]:
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         params.extend([p for p in base_head.parameters() if p.requires_grad])
#         # deduplicate in case a future model stores the same head internally
#         unique: list[nn.Parameter] = []
#         seen = set()
#         for p in params:
#             pid = id(p)
#             if pid not in seen:
#                 unique.append(p)
#                 seen.add(pid)
#         if not unique:
#             raise RuntimeError("No trainable parameters found for base phase.")
#         return unique

#     # ============================================================
#     # Label mapping / base head
#     # ============================================================
#     def _build_base_label_map(self, phase_class_ids: Iterable[int]) -> Dict[int, int]:
#         return {int(c): i for i, c in enumerate([int(x) for x in phase_class_ids])}

#     def _labels_to_local(self, labels: torch.Tensor, label_map: Dict[int, int]) -> torch.Tensor:
#         labels = labels.long().view(-1)
#         local = torch.full_like(labels, -1)
#         for global_cls, local_cls in label_map.items():
#             local[labels == int(global_cls)] = int(local_cls)
#         if (local < 0).any():
#             bad = labels[local < 0].detach().cpu().unique().tolist()
#             raise RuntimeError(f"Base batch contains labels outside phase-0 classes: {bad}")
#         return local

#     def _extract_feature_dim(self, train_loader) -> int:
#         was_training = bool(self.model.training)
#         self.model.eval()
#         with torch.no_grad():
#             for batch in train_loader:
#                 x, _, _, _ = self._unpack_hsi_batch(batch)
#                 x = x.float().to(self.device, non_blocking=True)
#                 out = self.model.extract_projected_features(x)
#                 features = out["features"]
#                 if features.dim() != 2:
#                     raise RuntimeError(f"Projected features must be [B,D], got {tuple(features.shape)}")
#                 if was_training:
#                     self.model.train()
#                 return int(features.size(1))
#         if was_training:
#             self.model.train()
#         raise RuntimeError("Cannot infer projected feature dimension: train_loader is empty.")

#     def _make_base_ce_head(self, feature_dim: int, num_base_classes: int) -> nn.Module:
#         head = nn.Linear(int(feature_dim), int(num_base_classes), bias=True).to(self.device)
#         nn.init.normal_(head.weight, mean=0.0, std=0.01)
#         nn.init.zeros_(head.bias)
#         return head

#     def _batch_balanced_ce(self, logits: torch.Tensor, labels_local: torch.Tensor, num_classes: int) -> torch.Tensor:
#         label_smoothing = self._base_cfg_float("label_smoothing", 0.0)
#         use_balance = self._base_cfg_bool("base_class_balance", False)
#         if not use_balance:
#             return F.cross_entropy(logits, labels_local, label_smoothing=label_smoothing)

#         counts = torch.bincount(labels_local, minlength=int(num_classes)).float().to(logits.device)
#         weights = torch.zeros_like(counts)
#         valid = counts > 0
#         weights[valid] = 1.0 / counts[valid].sqrt().clamp_min(1.0)
#         weights = weights / weights[valid].mean().clamp_min(1e-6)
#         return F.cross_entropy(logits, labels_local, weight=weights, label_smoothing=label_smoothing)

#     # ============================================================
#     # GICS key view
#     # ============================================================
#     def _augment_gics_key_view(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Mild HSI-safe augmentation for the detached key branch.

#         Default is identity. Do not use aggressive band dropping; PCA/band axes
#         carry material information and corrupting them makes GICS learn nonsense.
#         """
#         if x is None or not torch.is_tensor(x) or x.numel() == 0:
#             return x

#         x_key = x.clone()
#         noise_std = self._base_cfg_float("base_gics_key_noise_std", 0.0)
#         scale_jitter = self._base_cfg_float("base_gics_key_scale_jitter", 0.0)
#         band_drop = self._base_cfg_float("base_gics_key_band_drop", 0.0)
#         spatial_drop = self._base_cfg_float("base_gics_key_spatial_drop", 0.0)

#         if noise_std > 0.0:
#             x_key = x_key + torch.randn_like(x_key) * float(noise_std)

#         if scale_jitter > 0.0:
#             shape = [x_key.size(0)] + [1] * (x_key.dim() - 1)
#             scale = 1.0 + float(scale_jitter) * torch.randn(shape, device=x_key.device, dtype=x_key.dtype)
#             x_key = x_key * scale

#         if band_drop > 0.0 and x_key.dim() >= 4 and x_key.size(1) > 1:
#             keep = (torch.rand(x_key.size(0), x_key.size(1), 1, 1, device=x_key.device) > float(band_drop)).to(x_key.dtype)
#             all_zero = keep.flatten(1).sum(dim=1) <= 0
#             if bool(all_zero.any().item()):
#                 keep[all_zero, 0, 0, 0] = 1.0
#             x_key = x_key * keep

#         if spatial_drop > 0.0 and x_key.dim() >= 4 and x_key.size(-2) > 1 and x_key.size(-1) > 1:
#             smask = (torch.rand(x_key.size(0), 1, x_key.size(-2), x_key.size(-1), device=x_key.device) > float(spatial_drop)).to(x_key.dtype)
#             x_key = x_key * smask

#         return x_key

#     @staticmethod
#     def _band_summary_from_spectral(spectral_summary: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#         """Fallback band signature used only when the model did not expose band_summary.

#         The updated NECILModel should normally provide center-pixel band_summary.
#         This fallback is intentionally conservative: signed/PCA-like vectors are
#         converted by softmax, non-negative rows are sum-normalized, and degenerate
#         rows become uniform.  It is a reliability/risk signal, not classifier input.
#         """
#         if spectral_summary is None or not torch.is_tensor(spectral_summary) or spectral_summary.numel() == 0:
#             return None
#         if spectral_summary.dim() != 2 or spectral_summary.size(1) <= 0:
#             return None
#         s = torch.nan_to_num(spectral_summary, nan=0.0, posinf=0.0, neginf=0.0)
#         if bool((s < 0).any().item()):
#             return torch.softmax(s, dim=1)
#         b = s.clamp_min(0.0)
#         denom = b.sum(dim=1, keepdim=True)
#         uniform = torch.full_like(b, 1.0 / float(max(int(b.size(1)), 1)))
#         return torch.where(denom > 1e-8, b / denom.clamp_min(1e-8), uniform)

#     def _base_spectral_summary_is_physical(self, explicit: Optional[bool] = None) -> bool:
#         """Whether spectral_summary is wavelength-ordered raw HSI, not PCA.

#         Spectral derivatives are physically meaningful only for ordered bands.
#         If the pipeline uses PCA and no raw center spectra are supplied, this
#         returns False and the spectral-shape part of SRPGR becomes a safe zero.
#         """
#         if explicit is not None:
#             return bool(explicit)
#         if hasattr(self.args, "base_spectral_summary_is_physical"):
#             return self._base_cfg_bool("base_spectral_summary_is_physical", False)
#         if hasattr(self.args, "spectral_summary_is_physical"):
#             return self._base_cfg_bool("spectral_summary_is_physical", False)
#         pca = int(getattr(self.args, "pca_components", 0) or 0)
#         allow_nonphysical = self._base_cfg_bool("allow_nonphysical_spectral_summary", False)
#         if pca > 0 and not allow_nonphysical:
#             return False
#         return self._base_cfg_bool("assume_input_spectral_order_is_physical", False)

#     def _normalize_external_spectra(self, spectra: Optional[torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
#         """Return external spectra as [B,S] when the loader provides raw center spectra."""
#         if spectra is None or not torch.is_tensor(spectra) or spectra.numel() == 0:
#             return None
#         s = spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
#         if s.dim() == 4:
#             # [B,S,H,W] raw cube: center pixel only.
#             s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#         elif s.dim() == 3:
#             # [B,S,L] or [B,H,W]-like metadata: keep spectral-like axis conservative.
#             if s.size(0) == x.size(0):
#                 s = s.flatten(1)
#         elif s.dim() == 1:
#             s = s.view(x.size(0), -1)
#         elif s.dim() != 2:
#             s = s.flatten(1)
#         if s.size(0) != x.size(0):
#             raise RuntimeError(f"External spectral summary batch mismatch: spectra={tuple(s.shape)}, x={tuple(x.shape)}")
#         return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

#     @staticmethod
#     def _center_spectrum_from_input(x: torch.Tensor) -> torch.Tensor:
#         """Fallback center spectrum from model input. Usually PCA, so not physical by default."""
#         if x.dim() == 4:
#             return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
#         if x.dim() == 3:
#             return x.flatten(1)
#         if x.dim() == 2:
#             return x
#         return x.flatten(1)

#     def _call_extract_projected_features(
#         self,
#         x: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: Optional[bool] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Call updated NECILModel when available, with backward compatibility."""
#         try:
#             return self.model.extract_projected_features(
#                 x,
#                 spectral_summary=spectral_summary,
#                 spectral_summary_is_physical=self._base_spectral_summary_is_physical(spectral_summary_is_physical),
#             )
#         except TypeError:
#             return self.model.extract_projected_features(x)

#     def _base_spectral_shape_regularizer(
#         self,
#         *,
#         features: torch.Tensor,
#         labels: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor],
#         spectral_summary_is_physical: bool,
#     ) -> Dict[str, torch.Tensor]:
#         """Spectral-shape component of SRPGR.

#         This is a safe no-op unless physical raw spectra are available and the
#         SRGP loss file exposes spectral_shape_discrimination_loss.
#         """
#         if spectral_shape_discrimination_loss is None:
#             z = self._zero(features)
#             return {"total": z, "spectral_shape": z, "mean_similarity": z, "pair_count": z, "valid_class_count": z}
#         weight = self._base_cfg_float("base_spectral_shape_weight", 0.05)
#         if weight <= 0.0:
#             z = self._zero(features)
#             return {"total": z, "spectral_shape": z, "mean_similarity": z, "pair_count": z, "valid_class_count": z}
#         parts = spectral_shape_discrimination_loss(
#             spectral_summary=spectral_summary,
#             labels=labels,
#             features=features,
#             spectral_summary_is_physical=bool(spectral_summary_is_physical),
#             require_physical_summary=self._base_cfg_bool("spectral_require_physical_summary", True),
#             min_samples=self._base_cfg_int("pgr_min_class_samples", 3),
#             max_shape_similarity=self._base_cfg_float("base_max_spectral_shape_similarity", 0.75),
#             risk_center_margin=self._base_cfg_float("pgr_center_margin", 1.05),
#             risk_weight=self._base_cfg_float("base_spectral_shape_risk_weight", 1.0),
#             return_parts=True,
#         )
#         if not isinstance(parts, dict):
#             parts = {"total": parts, "spectral_shape": parts.detach()}
#         raw_total = parts.get("total", self._zero(features))
#         parts["raw_total"] = raw_total.detach()
#         parts["total"] = float(weight) * raw_total
#         return parts

#     def _extract_base_views(
#         self,
#         x: torch.Tensor,
#         external_spectra: Optional[torch.Tensor] = None,
#     ) -> Tuple[
#         torch.Tensor,
#         Optional[torch.Tensor],
#         Optional[torch.Tensor],
#         torch.Tensor,
#         Optional[torch.Tensor],
#         Optional[torch.Tensor],
#         bool,
#     ]:
#         """Return query/key views for CE + unified SRPGR.

#         The spectral summary is physical only when raw wavelength-ordered spectra
#         are supplied by the dataloader. If the model input is PCA, the fallback
#         center vector may still be used for band diagnostics, but not for
#         spectral-shape derivatives.
#         """
#         ext_s = self._normalize_external_spectra(external_spectra, x)
#         if ext_s is not None:
#             input_s = ext_s

#             # Reduced-band safety rule.  If metadata spectra have the same
#             # channel count as the model input while PCA/iPCA is active, they are
#             # reduced components, not physical wavelength-ordered spectra.
#             # Treat them as non-physical so SRPGR spectral-derivative terms are
#             # gated off and the model never mixes 30-D band weights with 200-D
#             # raw spectra.
#             pca_components = int(getattr(self.args, "pca_components", 0) or 0)
#             uses_reduction = pca_components > 0 and not bool(getattr(self.args, "no_pca", False))
#             input_channels = int(x.size(1)) if torch.is_tensor(x) and x.dim() >= 2 else 0
#             spectra_dim = int(input_s.size(1)) if torch.is_tensor(input_s) and input_s.dim() == 2 else 0
#             reduced_metadata = bool(uses_reduction and input_channels > 0 and spectra_dim == input_channels)

#             if reduced_metadata:
#                 input_s_is_physical = False
#             else:
#                 input_s_is_physical = self._base_spectral_summary_is_physical(
#                     self._base_cfg_bool("external_spectra_are_physical", True)
#                 )
#         else:
#             input_s = self._center_spectrum_from_input(x)
#             input_s_is_physical = self._base_spectral_summary_is_physical(None)

#         out_q = self._call_extract_projected_features(
#             x,
#             spectral_summary=input_s,
#             spectral_summary_is_physical=input_s_is_physical,
#         )
#         z_q = out_q["features"]
#         if z_q.dim() != 2 or not torch.isfinite(z_q).all():
#             raise RuntimeError(f"Invalid projected query features: {tuple(z_q.shape)}")
#         if "projected_features" in out_q and torch.is_tensor(out_q["projected_features"]):
#             if not torch.allclose(z_q, out_q["projected_features"], atol=1e-5, rtol=1e-4):
#                 raise RuntimeError(
#                     "Base feature-space mismatch: out['features'] and out['projected_features'] differ. "
#                     "SRPGR/GeometryBank must all use the same canonical projected z-space."
#                 )

#         s_q = out_q.get("spectral_summary", None)
#         if not (torch.is_tensor(s_q) and s_q.dim() == 2 and s_q.size(0) == x.size(0)):
#             s_q = input_s
#         spectral_is_physical = bool(out_q.get("spectral_summary_is_physical", input_s_is_physical))
#         b_q = out_q.get("band_summary", out_q.get("band_importance", None))
#         if b_q is None:
#             b_q = self._band_summary_from_spectral(s_q)

#         x_key = self._augment_gics_key_view(x)
#         # The key view uses the same spectral summary because physical spectral
#         # identity belongs to the center pixel/label, not to the mild feature-view
#         # augmentation. This avoids corrupting spectral-shape targets.
#         was_training = bool(self.model.training)
#         self.model.eval()
#         with torch.no_grad():
#             out_k = self._call_extract_projected_features(
#                 x_key,
#                 spectral_summary=input_s,
#                 spectral_summary_is_physical=spectral_is_physical,
#             )
#         if was_training:
#             self.model.train()

#         z_k = out_k["features"].detach()
#         if "projected_features" in out_k and torch.is_tensor(out_k["projected_features"]):
#             if not torch.allclose(out_k["features"], out_k["projected_features"], atol=1e-5, rtol=1e-4):
#                 raise RuntimeError("SRPGR key branch is not in canonical projected z-space.")
#         s_k = out_k.get("spectral_summary", None)
#         if not (torch.is_tensor(s_k) and s_k.dim() == 2 and s_k.size(0) == x.size(0)):
#             s_k = s_q
#         b_k = out_k.get("band_summary", out_k.get("band_importance", None))
#         if b_k is None:
#             b_k = self._band_summary_from_spectral(s_k)
#         if torch.is_tensor(s_k):
#             s_k = s_k.detach()
#         if torch.is_tensor(b_k):
#             b_k = b_k.detach()

#         if z_k.shape != z_q.shape:
#             raise RuntimeError(f"SRPGR key/query mismatch: query={tuple(z_q.shape)}, key={tuple(z_k.shape)}")
#         if not torch.isfinite(z_k).all():
#             raise RuntimeError("Projected key features contain NaN/Inf.")
#         return z_q, s_q, b_q, z_k, s_k, b_k, spectral_is_physical

#     @torch.no_grad()
#     def _rebuild_base_geometry_bank_for_validation(
#         self,
#         phase: int,
#         phase_class_ids: Iterable[int],
#         *,
#         split: str = "train",
#     ) -> None:
#         """Rebuild base GeometryBank rows from canonical z before geometry validation.

#         Base validation uses the geometry-energy classifier, so the bank must be
#         synchronized with the current representation every epoch. This method
#         bypasses refresh_before_validation because base geometry validation is
#         meaningless with stale or unbuilt rows.
#         """
#         if not hasattr(self, "_build_class_memory_from_current_phase"):
#             raise AttributeError("TrainerHelper._build_class_memory_from_current_phase() is required.")
#         old_training_state = bool(self.model.training)
#         ctx = self.dataset.memory_build_context(int(phase)) if hasattr(self.dataset, "memory_build_context") else None
#         if ctx is None:
#             from contextlib import nullcontext
#             ctx = nullcontext()
#         with ctx:
#             for cls in [int(c) for c in phase_class_ids]:
#                 self._build_class_memory_from_current_phase(cls, split=split)
#         self.model.train(old_training_state)

#     # ============================================================
#     # Diagnostics
#     # ============================================================
#     @torch.no_grad()
#     def _base_geometry_global_metrics(self) -> Dict[str, float]:
#         """Diagnostics compatible with the cleaned bank only."""
#         if not hasattr(self.model, "geometry_bank") or not hasattr(self.model.geometry_bank, "geometry_diagnostics"):
#             return {}
#         try:
#             diag = self.model.geometry_bank.geometry_diagnostics()
#         except Exception:
#             return {}
#         out: Dict[str, float] = {}
#         for key in (
#             "feature_subspace_overlap",
#             "band_overlap",
#             "spectral_shape_overlap",
#             "spectral_shape_similarity_mean",
#             "spectral_shape_similarity_max",
#             "geometry_conflict_mean",
#             "geometry_conflict_max",
#             "geometry_reserve_score",
#             "feature_rank_usage",
#         ):
#             value = diag.get(key, None) if isinstance(diag, dict) else None
#             if torch.is_tensor(value) and value.numel() == 1:
#                 out[key] = float(value.detach().cpu().item())
#             elif isinstance(value, (int, float)):
#                 out[key] = float(value)
#         return out

#     @staticmethod
#     def _scalar(value: Any, default: float = 0.0) -> float:
#         if torch.is_tensor(value):
#             if value.numel() == 0:
#                 return float(default)
#             return float(value.detach().float().mean().cpu().item())
#         if isinstance(value, (int, float)):
#             return float(value)
#         return float(default)

#     @torch.no_grad()
#     def _build_base_geometry_certificate(
#         self,
#         phase_class_ids: Iterable[int],
#         *,
#         val_stats: Optional[Dict[str, Any]] = None,
#         train_stats: Optional[Dict[str, Any]] = None,
#         head_train_acc: Optional[float] = None,
#     ) -> Dict[str, Any]:
#         """Create the base-to-incremental geometry certificate.

#         This is the explicit PRL-style handoff: base phase prepares a canonical
#         geometry field; incremental phase should consume this certificate to
#         decide risk-aware replay, descriptor admission, and old/new margins.
#         """
#         classes = [int(c) for c in phase_class_ids]
#         cert: Dict[str, Any] = {
#             "phase": 0,
#             "class_ids": classes,
#             "num_base_classes": len(classes),
#             "valid": False,
#             "errors": [],
#             "warnings": [],
#         }
#         if not classes:
#             cert["errors"].append("no base classes")
#             return cert

#         try:
#             bank = self._get_valid_base_bank(max(classes))
#         except Exception as exc:
#             cert["errors"].append(f"bank_unavailable: {exc}")
#             return cert

#         idx = torch.as_tensor(classes, device=self.device, dtype=torch.long)
#         counts = bank.get("sample_counts")
#         ranks = bank.get("active_ranks")
#         rel = bank.get("reliability")
#         feat_rel = bank.get("feature_reliability", rel)
#         band_rel = bank.get("band_reliability", None)
#         variances = bank.get("variances")
#         valid_mask = bank.get("valid_mask", None)
#         if valid_mask is None or not torch.is_tensor(valid_mask) or valid_mask.numel() <= int(idx.max().item()):
#             valid_mask = counts.flatten() > 0 if torch.is_tensor(counts) else torch.zeros(max(classes) + 1, device=self.device, dtype=torch.bool)
#         valid_base = valid_mask.to(device=self.device).bool().index_select(0, idx)

#         def _take(t: Optional[torch.Tensor], fill: float = 0.0) -> torch.Tensor:
#             if not torch.is_tensor(t) or t.numel() <= int(idx.max().item()):
#                 return torch.full((len(classes),), float(fill), device=self.device, dtype=torch.float32)
#             return t.to(device=self.device).flatten().index_select(0, idx).float()

#         count_v = _take(counts, 0.0)
#         rank_v = _take(ranks, 0.0)
#         rel_v = _take(rel, 0.0)
#         feat_rel_v = _take(feat_rel, 0.0)
#         band_rel_v = _take(band_rel, 0.0) if torch.is_tensor(band_rel) else torch.full_like(rel_v, 0.0)
#         if torch.is_tensor(variances) and variances.dim() == 2 and variances.size(0) > int(idx.max().item()):
#             res_v = variances.to(device=self.device).index_select(0, idx)[:, -1].float()
#         else:
#             res_v = torch.full((len(classes),), 0.0, device=self.device)

#         sub_mean = sub_max = 0.0
#         band_mean = band_max = 0.0
#         spectral_mean = spectral_max = 0.0
#         spectral_pair_count = 0
#         conflict_mean = conflict_max = 0.0
#         pair_count = 0
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None:
#             try:
#                 sub = gb.pairwise_subspace_overlap().to(device=self.device).index_select(0, idx).index_select(1, idx)
#                 eye = torch.eye(sub.size(0), device=sub.device, dtype=torch.bool)
#                 pair = sub[~eye]
#                 if pair.numel() > 0:
#                     sub_mean = float(pair.mean().detach().cpu().item())
#                     sub_max = float(pair.max().detach().cpu().item())
#                     pair_count = int(pair.numel())
#             except Exception as exc:
#                 cert["warnings"].append(f"subspace_pairwise_unavailable: {exc}")
#             try:
#                 band = gb.pairwise_band_similarity().to(device=self.device).index_select(0, idx).index_select(1, idx)
#                 eye = torch.eye(band.size(0), device=band.device, dtype=torch.bool)
#                 pair = band[~eye]
#                 if pair.numel() > 0:
#                     band_mean = float(pair.mean().detach().cpu().item())
#                     band_max = float(pair.max().detach().cpu().item())
#             except Exception as exc:
#                 cert["warnings"].append(f"band_pairwise_unavailable: {exc}")
#             try:
#                 if hasattr(gb, "pairwise_spectral_shape_similarity"):
#                     spec = gb.pairwise_spectral_shape_similarity().to(device=self.device).index_select(0, idx).index_select(1, idx)
#                     eye = torch.eye(spec.size(0), device=spec.device, dtype=torch.bool)
#                     pair = spec[~eye]
#                     pair = pair[torch.isfinite(pair)]
#                     if pair.numel() > 0:
#                         spectral_mean = float(pair.mean().detach().cpu().item())
#                         spectral_max = float(pair.max().detach().cpu().item())
#                         spectral_pair_count = int(pair.numel())
#             except Exception as exc:
#                 cert["warnings"].append(f"spectral_shape_pairwise_unavailable: {exc}")
#             try:
#                 conflict = gb.geometry_conflict_matrix().to(device=self.device).index_select(0, idx).index_select(1, idx)
#                 eye = torch.eye(conflict.size(0), device=conflict.device, dtype=torch.bool)
#                 pair = conflict[~eye]
#                 if pair.numel() > 0:
#                     conflict_mean = float(pair.mean().detach().cpu().item())
#                     conflict_max = float(pair.max().detach().cpu().item())
#             except Exception as exc:
#                 cert["warnings"].append(f"conflict_matrix_unavailable: {exc}")

#         geom = self._base_geometry_global_metrics()
#         cert.update({
#             "geom_val_acc": float((val_stats or {}).get("acc", 0.0)),
#             "geom_val_loss": float((val_stats or {}).get("loss", 0.0)),
#             "geom_train_acc": float((train_stats or {}).get("acc", 0.0)),
#             "head_train_acc": float(head_train_acc) if head_train_acc is not None else 0.0,
#             "valid_row_count": int(valid_base.sum().detach().cpu().item()),
#             "min_sample_count": float(count_v.min().detach().cpu().item()) if count_v.numel() else 0.0,
#             "mean_sample_count": float(count_v.mean().detach().cpu().item()) if count_v.numel() else 0.0,
#             "min_reliability": float(rel_v.min().detach().cpu().item()) if rel_v.numel() else 0.0,
#             "mean_reliability": float(rel_v.mean().detach().cpu().item()) if rel_v.numel() else 0.0,
#             "min_feature_reliability": float(feat_rel_v.min().detach().cpu().item()) if feat_rel_v.numel() else 0.0,
#             "mean_feature_reliability": float(feat_rel_v.mean().detach().cpu().item()) if feat_rel_v.numel() else 0.0,
#             "min_band_reliability": float(band_rel_v.min().detach().cpu().item()) if band_rel_v.numel() else 0.0,
#             "mean_band_reliability": float(band_rel_v.mean().detach().cpu().item()) if band_rel_v.numel() else 0.0,
#             "mean_active_rank": float(rank_v.mean().detach().cpu().item()) if rank_v.numel() else 0.0,
#             "max_active_rank": float(rank_v.max().detach().cpu().item()) if rank_v.numel() else 0.0,
#             "mean_res_var": float(res_v.mean().detach().cpu().item()) if res_v.numel() else 0.0,
#             "max_res_var": float(res_v.max().detach().cpu().item()) if res_v.numel() else 0.0,
#             "pair_count": pair_count,
#             "mean_subspace_overlap": sub_mean,
#             "max_subspace_overlap": sub_max,
#             "mean_band_similarity": band_mean,
#             "max_band_similarity": band_max,
#             "mean_spectral_shape_similarity": spectral_mean,
#             "max_spectral_shape_similarity": spectral_max,
#             "spectral_shape_pair_count": spectral_pair_count,
#             "mean_geometry_conflict": conflict_mean,
#             "max_geometry_conflict": conflict_max,
#             "geometry_reserve_score": float(geom.get("geometry_reserve_score", 0.0)),
#             "feature_rank_usage": float(geom.get("feature_rank_usage", 0.0)),
#         })

#         thr = {
#             "min_geom_acc": self._base_cfg_float("base_cert_min_geom_acc", 90.0),
#             "min_reliability": self._base_cfg_float("base_cert_min_reliability", 0.15),
#             "min_mean_reliability": self._base_cfg_float("base_cert_min_mean_reliability", 0.35),
#             "max_subspace_overlap": self._base_cfg_float("base_cert_max_subspace_overlap", 0.65),
#             "max_geometry_conflict": self._base_cfg_float("base_cert_max_geometry_conflict", 2.0),
#             "max_band_similarity": self._base_cfg_float("base_cert_max_band_similarity", 0.98),
#             "max_spectral_shape_similarity": self._base_cfg_float("base_cert_max_spectral_shape_similarity", 0.90),
#         }
#         cert["thresholds"] = thr
#         checks = {
#             "all_base_rows_valid": cert["valid_row_count"] == len(classes),
#             "geom_acc_ok": cert["geom_val_acc"] >= thr["min_geom_acc"],
#             "min_reliability_ok": cert["min_reliability"] >= thr["min_reliability"],
#             "mean_reliability_ok": cert["mean_reliability"] >= thr["min_mean_reliability"],
#             "subspace_overlap_ok": cert["max_subspace_overlap"] <= thr["max_subspace_overlap"],
#             "geometry_conflict_ok": cert["max_geometry_conflict"] <= thr["max_geometry_conflict"],
#             "band_similarity_ok": cert["max_band_similarity"] <= thr["max_band_similarity"],
#             "spectral_shape_similarity_ok": (
#                 int(cert.get("spectral_shape_pair_count", 0)) == 0
#                 or cert["max_spectral_shape_similarity"] <= thr["max_spectral_shape_similarity"]
#             ),
#         }
#         cert["checks"] = checks
#         cert["valid"] = bool(all(checks.values()) and len(cert["errors"]) == 0)
#         if not cert["valid"]:
#             failed = [k for k, ok in checks.items() if not ok]
#             cert["errors"].extend(failed)
#         return cert

#     def _print_base_geometry_certificate(self, certificate: Dict[str, Any]) -> None:
#         print("[Base Geometry Certificate]")
#         if not certificate:
#             print("  unavailable")
#             return
#         status = "PASS" if bool(certificate.get("valid", False)) else "WARN/FAIL"
#         print(f"  status={status} | classes={certificate.get('class_ids', [])}")
#         print(
#             "  "
#             f"GeomValAcc={float(certificate.get('geom_val_acc', 0.0)):.2f}% | "
#             f"valid_rows={int(certificate.get('valid_row_count', 0))}/{int(certificate.get('num_base_classes', 0))} | "
#             f"rel(min/mean)={float(certificate.get('min_reliability', 0.0)):.3f}/{float(certificate.get('mean_reliability', 0.0)):.3f} | "
#             f"subspace(max/mean)={float(certificate.get('max_subspace_overlap', 0.0)):.3f}/{float(certificate.get('mean_subspace_overlap', 0.0)):.3f} | "
#             f"band(max/mean)={float(certificate.get('max_band_similarity', 0.0)):.3f}/{float(certificate.get('mean_band_similarity', 0.0)):.3f} | "
#             f"spec(max/mean)={float(certificate.get('max_spectral_shape_similarity', 0.0)):.3f}/{float(certificate.get('mean_spectral_shape_similarity', 0.0)):.3f} | "
#             f"conflict(max/mean)={float(certificate.get('max_geometry_conflict', 0.0)):.3f}/{float(certificate.get('mean_geometry_conflict', 0.0)):.3f}"
#         )
#         if certificate.get("errors"):
#             print(f"  failed_checks={certificate.get('errors')}")
#         if certificate.get("warnings") and self.debug:
#             print(f"  warnings={certificate.get('warnings')}")

#     def _enforce_base_geometry_certificate(self, certificate: Dict[str, Any]) -> None:
#         """Warn by default; optionally hard-stop if the user requests strict gating."""
#         self._last_base_geometry_certificate = certificate
#         try:
#             setattr(self.model, "base_geometry_certificate", certificate)
#         except Exception:
#             pass
#         self._print_base_geometry_certificate(certificate)
#         if bool(certificate.get("valid", False)):
#             return
#         msg = (
#             "Base geometry certificate failed. Incremental training will be high-risk because "
#             "base geometry is not clean enough for descriptor insertion. Failed checks: "
#             f"{certificate.get('errors', [])}"
#         )
#         if self._base_cfg_bool("enforce_base_geometry_certificate", False):
#             raise RuntimeError(msg)
#         print(f"[Base Geometry Certificate WARN] {msg}")

#     def _get_valid_base_bank(self, max_class_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
#         if not hasattr(self, "_safe_get_subspace_bank"):
#             raise AttributeError("TrainerHelper._safe_get_subspace_bank() is required.")
#         bank = self._safe_get_subspace_bank()
#         required = ("means", "bases", "variances", "sample_counts")
#         missing = [k for k in required if k not in bank or not torch.is_tensor(bank[k]) or bank[k].numel() == 0]
#         if missing:
#             raise RuntimeError(f"Base GeometryBank is not ready; missing/empty keys: {missing}")
#         if max_class_id is not None and int(max_class_id) >= bank["means"].size(0):
#             raise RuntimeError(f"Bank has {bank['means'].size(0)} rows but needs class id {int(max_class_id)}")
#         return bank

#     @torch.no_grad()
#     def _print_base_geometry_diagnostics(self, phase_class_ids: Iterable[int]) -> None:
#         if not self.debug and not bool(getattr(self.args, "print_base_geometry_diagnostics", True)):
#             return
#         try:
#             bank = self._get_valid_base_bank()
#         except Exception as exc:
#             print(f"[Base Geometry Diagnostics] unavailable: {exc}")
#             return

#         counts = bank.get("sample_counts", None)
#         active_ranks = bank.get("active_ranks", None)
#         reliability = bank.get("reliability", None)
#         variances = bank.get("variances", None)
#         means = bank.get("means", None)
#         bands = bank.get("band_importances", bank.get("band_importance", None))
#         spec_rel = bank.get("spectral_shape_reliability", None)

#         print("[Base Geometry Diagnostics]")
#         print("  cls | count | rank | rel  | spec-rel | resvar   | mean_norm | band_entropy")
#         for cls in [int(c) for c in phase_class_ids]:
#             count = float(counts[cls].detach().item()) if torch.is_tensor(counts) and counts.numel() > cls else -1.0
#             rank = int(active_ranks[cls].detach().item()) if torch.is_tensor(active_ranks) and active_ranks.numel() > cls else -1
#             rel = float(reliability[cls].detach().item()) if torch.is_tensor(reliability) and reliability.numel() > cls else -1.0
#             srel = float(spec_rel[cls].detach().item()) if torch.is_tensor(spec_rel) and spec_rel.numel() > cls else -1.0
#             rv = float(variances[cls, -1].detach().item()) if torch.is_tensor(variances) and variances.size(0) > cls else -1.0
#             mn = float(means[cls].norm().detach().item()) if torch.is_tensor(means) and means.size(0) > cls else -1.0
#             bent = -1.0
#             if torch.is_tensor(bands) and bands.dim() == 2 and bands.size(0) > cls:
#                 b = bands[cls].detach().float().clamp_min(0.0)
#                 b = b / b.sum().clamp_min(1e-8)
#                 bent = float((-(b * torch.log(b.clamp_min(1e-8))).sum()).cpu().item())
#             print(f"  {cls:3d} | {count:5.0f} | {rank:4d} | {rel:4.2f} | {srel:8.3f} | {rv:8.5f} | {mn:9.4f} | {bent:12.4f}")

#         geom = self._base_geometry_global_metrics()
#         if geom:
#             print("  " + " | ".join(f"{k}={v:.4f}" for k, v in geom.items()))

#     @torch.no_grad()
#     def _batch_base_overlap_diagnostics(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
#         try:
#             diag = base_center_overlap_diagnostics(
#                 features,
#                 labels,
#                 normalize=True,
#                 min_samples=self._base_cfg_int("pgr_min_class_samples", 3),
#             )
#         except Exception:
#             return {"batch_compact": 0.0, "batch_center_margin": 0.0, "batch_min_center_margin": 0.0}
#         return {
#             "batch_compact": float(diag.get("compact", self._zero(features)).detach().cpu().item()),
#             "batch_center_margin": float(diag.get("mean_center_margin", self._zero(features)).detach().cpu().item()),
#             "batch_min_center_margin": float(diag.get("min_center_margin", self._zero(features)).detach().cpu().item()),
#         }

#     # ============================================================
#     # Base objective: CE + GICS + PGR
#     # ============================================================
#     def _train_epoch_base_geometry(
#         self,
#         loader,
#         optimizer,
#         base_head: nn.Module,
#         phase_class_ids: Iterable[int],
#         label_map: Dict[int, int],
#         trainable_params: list[nn.Parameter],
#     ) -> Tuple[float, float]:
#         self.model.train()
#         base_head.train()

#         num_base = len([int(c) for c in phase_class_ids])
#         total_loss = 0.0
#         total_correct = 0
#         total_count = 0
#         stat_steps = 0
#         stat_sums = {
#             "ce": 0.0,
#             "srpgr": 0.0,
#             "compact_sep": 0.0,
#             "gics": 0.0,
#             "weighted_gics": 0.0,
#             "gics_anchors": 0.0,
#             "gics_pos": 0.0,
#             "pgr": 0.0,
#             "pgr_unweighted": 0.0,
#             "pgr_compact": 0.0,
#             "pgr_center": 0.0,
#             "pgr_subspace": 0.0,
#             "pgr_band": 0.0,
#             "pgr_volume": 0.0,
#             "spectral_shape": 0.0,
#             "spectral_shape_raw": 0.0,
#             "spectral_shape_mean_similarity": 0.0,
#             "spectral_shape_pairs": 0.0,
#             "spectral_active": 0.0,
#             "batch_compact": 0.0,
#             "batch_center_margin": 0.0,
#             "batch_min_center_margin": 0.0,
#         }

#         for batch in loader:
#             x, y, spectra, _ = self._unpack_hsi_batch(batch)
#             x = x.float().to(self.device, non_blocking=True)
#             y = y.long().to(self.device, non_blocking=True)
#             y_local = self._labels_to_local(y, label_map)

#             optimizer.zero_grad(set_to_none=True)

#             features, spectral_summary, band_summary, key_features, _, _, spectral_summary_is_physical = self._extract_base_views(x, spectra)
#             logits = base_head(features)
#             ce = self._batch_balanced_ce(logits, y_local, num_base)

#             gics = base_geometry_involved_contrastive_loss(
#                 features,
#                 y,
#                 key_features=key_features,
#                 weight=self._base_cfg_float("base_gics_weight", 0.20),
#                 temperature=self._base_cfg_float("base_gics_temperature", 0.07),
#                 same_class_positive=self._base_cfg_bool("base_gics_same_class_positive", True),
#                 class_balanced=self._base_cfg_bool("base_gics_class_balanced", True),
#                 detach_key=True,
#                 normalize=self._base_cfg_bool("base_gics_normalize", True),
#                 return_parts=True,
#             )

#             pgr = prospective_geometry_reserve_loss(
#                 features,
#                 y,
#                 band_summary=band_summary,
#                 weight=self._base_cfg_float("pgr_weight", 0.10),
#                 compact_weight=self._base_cfg_float("pgr_compact_weight", 0.15),
#                 center_weight=self._base_cfg_float("pgr_center_weight", 0.20),
#                 subspace_weight=self._base_cfg_float("pgr_subspace_weight", 0.10),
#                 band_weight=self._base_cfg_float("pgr_band_weight", 0.05),
#                 volume_weight=self._base_cfg_float("pgr_volume_weight", 0.05),
#                 center_margin=self._base_cfg_float("pgr_center_margin", 1.05),
#                 min_class_samples=self._base_cfg_int("pgr_min_class_samples", 3),
#                 subspace_min_samples=self._base_cfg_int("pgr_subspace_min_samples", 6),
#                 subspace_rank=self._base_cfg_int("pgr_subspace_rank", 3),
#                 max_band_similarity=self._base_cfg_float("pgr_band_overlap_max", 0.75),
#                 max_class_variance=self._base_cfg_float("pgr_max_class_variance", 0.75),
#                 normalize_features=self._base_cfg_bool("pgr_normalize_features", True),
#                 return_parts=True,
#             )

#             spectral_shape = self._base_spectral_shape_regularizer(
#                 features=features,
#                 labels=y,
#                 spectral_summary=spectral_summary,
#                 spectral_summary_is_physical=spectral_summary_is_physical,
#             )

#             srpgr_total = (
#                 gics["total"]
#                 + pgr["total"]
#                 + spectral_shape["total"]
#             )
#             loss = (
#                 self._base_cfg_float("base_ce_weight", 1.0) * ce
#                 + self._base_cfg_float("base_srpgr_weight", 1.0) * srpgr_total
#             )
#             if not torch.isfinite(loss):
#                 if self.debug:
#                     print("[WARN] Non-finite base loss skipped.")
#                 continue

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(trainable_params, float(self.grad_clip_base))
#             optimizer.step()

#             pred = logits.argmax(dim=1)
#             total_loss += float(loss.detach().item())
#             total_correct += int((pred == y_local).sum().item())
#             total_count += int(y_local.numel())
#             stat_steps += 1

#             batch_diag = self._batch_base_overlap_diagnostics(features.detach(), y.detach())
#             stat_sums["ce"] += float(ce.detach().item())
#             stat_sums["srpgr"] += float(srpgr_total.detach().item())
#             stat_sums["compact_sep"] += float(gics["total"].detach().item())
#             stat_sums["gics"] += float(gics.get("loss", gics.get("gics", self._zero(features))).detach().item())
#             stat_sums["weighted_gics"] += float(gics["total"].detach().item())
#             stat_sums["gics_anchors"] += float(gics.get("num_anchors", self._zero(features)).detach().item())
#             stat_sums["gics_pos"] += float(gics.get("mean_positive_count", self._zero(features)).detach().item())
#             stat_sums["pgr"] += float(pgr["total"].detach().item())
#             stat_sums["pgr_unweighted"] += float(pgr.get("pgr", self._zero(features)).detach().item())
#             stat_sums["pgr_compact"] += float(pgr["compact"].detach().item())
#             stat_sums["pgr_center"] += float(pgr["center"].detach().item())
#             stat_sums["pgr_subspace"] += float(pgr["subspace"].detach().item())
#             stat_sums["pgr_band"] += float(pgr["band"].detach().item())
#             stat_sums["pgr_volume"] += float(pgr["volume"].detach().item())
#             stat_sums["spectral_shape"] += float(spectral_shape["total"].detach().item())
#             stat_sums["spectral_shape_raw"] += float(spectral_shape.get("raw_total", self._zero(features)).detach().item())
#             stat_sums["spectral_shape_mean_similarity"] += float(spectral_shape.get("mean_similarity", self._zero(features)).detach().item())
#             stat_sums["spectral_shape_pairs"] += float(spectral_shape.get("pair_count", self._zero(features)).detach().item())
#             stat_sums["spectral_active"] += 1.0 if bool(spectral_summary_is_physical) else 0.0
#             for k, v in batch_diag.items():
#                 stat_sums[k] += float(v)

#         self._last_base_stats = {k: v / max(stat_steps, 1) for k, v in stat_sums.items()}
#         return total_loss / max(stat_steps, 1), 100.0 * total_correct / max(total_count, 1)

#     # backward-compatible alias for trainer.py calls that still use the old name
#     def _train_epoch_base_gics(self, *args, **kwargs):
#         return self._train_epoch_base_geometry(*args, **kwargs)

#     # ============================================================
#     # Checkpoint score
#     # ============================================================
#     def _select_base_checkpoint_score(self, val_stats: Dict, geom_stats: Optional[Dict] = None) -> float:
#         metric = str(getattr(self.args, "best_state_metric", "geometry_score")).lower()
#         geom_stats = geom_stats or {}
#         if metric in {"geometry_score", "geo", "reserve"}:
#             # Accuracy is necessary, but base feature geometry must also be clean.
#             acc = float(val_stats.get("acc", 0.0))
#             reserve = float(geom_stats.get("geometry_reserve_score", val_stats.get("geometry_score", 0.0)))
#             conflict = float(geom_stats.get("geometry_conflict_mean", 0.0))
#             overlap = float(geom_stats.get("feature_subspace_overlap", 0.0))
#             spec_overlap = float(geom_stats.get("spectral_shape_similarity_max", geom_stats.get("spectral_shape_overlap", 0.0)))
#             return acc + 10.0 * reserve - 5.0 * conflict - 2.0 * overlap - 1.0 * spec_overlap
#         if metric in {"acc", "accuracy", "oa", "val_acc"}:
#             return float(val_stats.get("acc", 0.0))
#         if metric in {"loss", "val_loss"}:
#             return -float(val_stats.get("loss", 1e9))
#         if metric in {"hm", "harmonic"}:
#             return float(val_stats.get("hm", val_stats.get("acc", 0.0)))
#         return float(val_stats.get("acc", 0.0))

#     # ============================================================
#     # Main base phase
#     # ============================================================
#     def train_base_phase(self, phase, epochs, batch_size=64, lr=1e-4) -> Dict:
#         phase = int(phase)
#         if phase != 0:
#             raise ValueError("train_base_phase() must only be called for phase 0.")

#         self._enforce_base_contract()

#         print("==== Base Phase Training | CE + SRPGR | Spectral-Residual GeometryBank ====")
#         self.dataset.start_phase(phase)
#         phase_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         label_map = self._build_base_label_map(phase_class_ids)
#         self._set_model_phase_and_old_count(phase, 0)

#         needed_classes = max(phase_class_ids) + 1
#         if hasattr(self.model, "ensure_class_capacity"):
#             self.model.ensure_class_capacity(needed_classes)

#         train_loader = self.dataset.get_phase_dataloader(phase, split="train", batch_size=batch_size, shuffle=True)
#         val_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)

#         feature_dim = self._extract_feature_dim(train_loader)
#         base_head = self._make_base_ce_head(feature_dim, len(phase_class_ids))
#         self._base_ce_head = base_head
#         self._set_base_trainability(base_head)
#         trainable_params = self._base_trainable_parameters(base_head)

#         optimizer = optim.Adam(
#             trainable_params,
#             lr=float(lr),
#             weight_decay=float(getattr(self.args, "weight_decay", 1e-4)),
#         )
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

#         history = {
#             "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
#             "val_old_acc": [], "val_new_acc": [], "val_hm": [],
#             "base_ce": [], "base_srpgr": [], "base_compact_sep": [], "base_gics": [], "base_weighted_gics": [],
#             "base_gics_anchors": [], "base_gics_pos": [],
#             "base_pgr": [], "base_pgr_unweighted": [], "base_pgr_compact": [],
#             "base_pgr_center": [], "base_pgr_subspace": [], "base_pgr_band": [], "base_pgr_volume": [],
#             "base_spectral_shape": [], "base_spectral_shape_raw": [], "base_spectral_shape_mean_similarity": [],
#             "base_spectral_shape_pairs": [], "base_spectral_active": [],
#             "batch_compact": [], "batch_center_margin": [], "batch_min_center_margin": [],
#             "feature_subspace_overlap": [], "band_overlap": [], "spectral_shape_overlap": [],
#             "spectral_shape_similarity_mean": [], "spectral_shape_similarity_max": [],
#             "geometry_conflict_mean": [], "geometry_conflict_max": [], "geometry_reserve_score": [],
#             "base_cert_geom_acc": [], "base_cert_min_reliability": [],
#             "base_cert_max_subspace_overlap": [], "base_cert_max_band_similarity": [], "base_cert_max_spectral_shape_similarity": [],
#             "base_cert_max_geometry_conflict": [], "base_cert_valid": [],
#         }

#         best_score = -1e18
#         best_state = None
#         no_improve = 0
#         epochs = int(epochs)

#         if hasattr(self, "_print_trainable_summary"):
#             self._print_trainable_summary(phase)
#         print(f"[Base] Temporary CE head: feature_dim={feature_dim}, classes={len(phase_class_ids)}. Discarded after base training.")
#         print("[Base Objective] CE + SRPGR. SRPGR = compact-sep + subspace reserve + volume control + spectral-shape reserve.")

#         for epoch in range(epochs):
#             self._base_epoch = int(epoch)
#             self._set_model_phase_and_old_count(phase, 0)

#             tr_loss, head_train_acc = self._train_epoch_base_geometry(
#                 train_loader,
#                 optimizer,
#                 base_head,
#                 phase_class_ids,
#                 label_map,
#                 trainable_params,
#             )

#             self._rebuild_base_geometry_bank_for_validation(phase, phase_class_ids, split="train")

#             train_eval_stats = self._validate_split_metrics(train_loader, old_class_count=0)
#             val_stats = self._validate_split_metrics(val_loader, old_class_count=0)
#             scheduler.step()

#             base_stats = getattr(self, "_last_base_stats", {})
#             geom_stats = self._base_geometry_global_metrics()
#             epoch_cert = self._build_base_geometry_certificate(
#                 phase_class_ids,
#                 val_stats=val_stats,
#                 train_stats=train_eval_stats,
#                 head_train_acc=head_train_acc,
#             )

#             history["train_loss"].append(float(tr_loss))
#             history["train_acc"].append(float(train_eval_stats["acc"]))
#             history["val_loss"].append(float(val_stats["loss"]))
#             history["val_acc"].append(float(val_stats["acc"]))
#             history["val_old_acc"].append(float(val_stats.get("old_acc", 0.0)))
#             history["val_new_acc"].append(float(val_stats.get("new_acc", 0.0)))
#             history["val_hm"].append(float(val_stats.get("hm", 0.0)))

#             for k in (
#                 "ce", "srpgr", "compact_sep", "gics", "weighted_gics", "gics_anchors", "gics_pos", "pgr", "pgr_unweighted",
#                 "pgr_compact", "pgr_center", "pgr_subspace", "pgr_band", "pgr_volume",
#                 "spectral_shape", "spectral_shape_raw", "spectral_shape_mean_similarity", "spectral_shape_pairs", "spectral_active",
#                 "batch_compact", "batch_center_margin", "batch_min_center_margin",
#             ):
#                 history_key = f"base_{k}" if k not in {"batch_compact", "batch_center_margin", "batch_min_center_margin"} else k
#                 if history_key in history:
#                     history[history_key].append(float(base_stats.get(k, 0.0)))

#             for k in ("feature_subspace_overlap", "band_overlap", "spectral_shape_overlap", "spectral_shape_similarity_mean", "spectral_shape_similarity_max", "geometry_conflict_mean", "geometry_conflict_max", "geometry_reserve_score"):
#                 history[k].append(float(geom_stats.get(k, 0.0)))

#             history["base_cert_geom_acc"].append(float(epoch_cert.get("geom_val_acc", 0.0)))
#             history["base_cert_min_reliability"].append(float(epoch_cert.get("min_reliability", 0.0)))
#             history["base_cert_max_subspace_overlap"].append(float(epoch_cert.get("max_subspace_overlap", 0.0)))
#             history["base_cert_max_band_similarity"].append(float(epoch_cert.get("max_band_similarity", 0.0)))
#             history["base_cert_max_spectral_shape_similarity"].append(float(epoch_cert.get("max_spectral_shape_similarity", 0.0)))
#             history["base_cert_max_geometry_conflict"].append(float(epoch_cert.get("max_geometry_conflict", 0.0)))
#             history["base_cert_valid"].append(1.0 if bool(epoch_cert.get("valid", False)) else 0.0)

#             print(
#                 f"[Base CE+SRPGR] Epoch {epoch + 1:03d}/{epochs} | "
#                 f"Loss: {tr_loss:.4f} | HeadAcc: {head_train_acc:.2f}% | "
#                 f"GeomTrainAcc: {train_eval_stats['acc']:.2f}% | "
#                 f"GeomValAcc: {val_stats['acc']:.2f}% | GeomValLoss: {val_stats['loss']:.4f} | "
#                 f"CE: {float(base_stats.get('ce', 0.0)):.4f} | "
#                 f"SRPGR: {float(base_stats.get('srpgr', 0.0)):.4f} | "
#                 f"CompactSep: {float(base_stats.get('compact_sep', 0.0)):.4f} | "
#                 f"Reserve: {float(base_stats.get('pgr', 0.0)):.4f} | "
#                 f"Compact: {float(base_stats.get('pgr_compact', 0.0)):.4f} | "
#                 f"Subspace: {float(base_stats.get('pgr_subspace', 0.0)):.4f} | "
#                 f"Band: {float(base_stats.get('pgr_band', 0.0)):.4f} | "
#                 f"SpecShape: {float(base_stats.get('spectral_shape', 0.0)):.4f} | "
#                 f"Overlap: {float(geom_stats.get('feature_subspace_overlap', 0.0)):.4f} | "
#                 f"ConflictMax: {float(epoch_cert.get('max_geometry_conflict', 0.0)):.4f} | "
#                 f"GeoReserve: {float(geom_stats.get('geometry_reserve_score', 0.0)):.4f} | "
#                 f"Cert: {'OK' if bool(epoch_cert.get('valid', False)) else 'RISK'}"
#             )

#             score_stats = dict(val_stats)
#             score_stats.update(geom_stats)
#             score_stats.update({f"cert_{k}": v for k, v in epoch_cert.items() if isinstance(v, (int, float, bool))})
#             score_stats["geometry_score"] = float(geom_stats.get("geometry_reserve_score", 0.0))
#             score = self._select_base_checkpoint_score(score_stats, geom_stats)
#             if score > best_score:
#                 best_score = score
#                 best_state = self._capture_state()
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             if self.early_stop_patience > 0 and no_improve >= self.early_stop_patience:
#                 print(f"[EarlyStop] Base phase: no improvement for {no_improve} epochs.")
#                 break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, 0)

#         print("[Base] Final SRGP GeometryBank rebuild from CE+SRPGR projected features.")
#         self._finalize_phase_memory(phase, split="train")
#         self._set_model_phase_and_old_count(phase, len(phase_class_ids))
#         self._print_base_geometry_diagnostics(phase_class_ids)

#         final_train_stats = self._validate_split_metrics(train_loader, old_class_count=0)
#         final_val_stats = self._validate_split_metrics(val_loader, old_class_count=0)
#         final_cert = self._build_base_geometry_certificate(
#             phase_class_ids,
#             val_stats=final_val_stats,
#             train_stats=final_train_stats,
#             head_train_acc=history["train_acc"][-1] if history.get("train_acc") else None,
#         )
#         history["base_geometry_certificate"] = final_cert
#         self._enforce_base_geometry_certificate(final_cert)

#         if hasattr(self, "diagnose_full_base_geometry"):
#             try:
#                 self._last_base_geometry_diagnostics = self.diagnose_full_base_geometry(
#                     loader=val_loader,
#                     phase_class_ids=phase_class_ids,
#                     anchors_per_class=int(getattr(self.args, "geometry_diag_anchors_per_class", 64)),
#                     topk_pairs=int(getattr(self.args, "geometry_diag_topk_pairs", 20)),
#                     topk_bands=int(getattr(self.args, "geometry_diag_topk_bands", 5)),
#                 )
#                 if hasattr(self, "_print_geometry_diagnostics_summary"):
#                     self._print_geometry_diagnostics_summary(self._last_base_geometry_diagnostics)
#                 if hasattr(self, "_save_geometry_diagnostics_to_files"):
#                     saved_paths = self._save_geometry_diagnostics_to_files(self._last_base_geometry_diagnostics, phase=phase)
#                     print(f"[Geometry Health] saved diagnostics: {saved_paths.get('json', '')}")
#             except Exception as exc:
#                 print(f"[Geometry Health WARN] could not create persistent diagnostics: {exc}")

#         self._base_ce_head = None
#         if hasattr(self.model, "drop_base_ce_head"):
#             self.model.drop_base_ce_head()
#         if hasattr(self, "save_checkpoint"):
#             self.save_checkpoint(phase, history)
#         return history

