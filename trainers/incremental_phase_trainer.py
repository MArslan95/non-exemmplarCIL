from __future__ import annotations

import copy
import json
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


_EPS = 1e-8


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _finite(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a tensor.")
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
        raise RuntimeError(f"{name} contains {bad} NaN/Inf values.")
    return x


class IncrementalPhaseTrainer:
    """Spectral-coupled descriptor-only incremental trainer for NECIL-HSI.

    Main-path contract
    ------------------
    * Dataset labels are global class ids.
    * GeometryBank rows are global class ids.
    * Classifier/logit columns are compact seen-class indices in ``seen_classes`` order.
    * CE labels are always seen-local indices.
    * Old GeometryBank rows are frozen and checked after every mutating operation.
    * New rows may be inserted/refined; old rows may not move.

    The SCTGR-RGA path keeps the certified base z-space fixed, builds paired
    physical-spectral/feature geometry for every new class, replays old classes
    through spectral-consistent core and risk-directed spectral tangent samples,
    and refines only bounded new descriptor parameters. Feature adapters,
    transport, calibrators, KD, shell replay, and raw exemplars are forbidden.
    """

    # ------------------------------------------------------------------
    # Basic config helpers
    # ------------------------------------------------------------------
    def _zero_like_ref(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        if torch.is_tensor(ref):
            return ref.sum() * 0.0
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def _inc_cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self, name, getattr(self.args, name, default)))

    def _inc_cfg_int(self, name: str, default: int) -> int:
        return int(getattr(self, name, getattr(self.args, name, default)))

    def _inc_cfg_bool(self, name: str, default: bool) -> bool:
        return _as_bool(getattr(self, name, getattr(self.args, name, default)))

    def _classifier_mode(self) -> str:
        """Clean incremental path uses geometry-only scoring.

        Calibrated/topology/anchor aliases are intentionally rejected here so a
        command cannot silently switch the classifier away from the GeometryBank
        energy field produced by the certified base phase.
        """
        mode = str(getattr(self.args, "incremental_classifier_mode", "geometry_only")).lower().strip()
        aliases = {
            "": "geometry",
            "geometry_only": "geometry",
            "geometry": "geometry",
        }
        mode = aliases.get(mode, mode)
        if mode != "geometry":
            raise RuntimeError(
                f"Unsupported incremental_classifier_mode={mode!r}. "
                "Use geometry_only/geometry for the clean SCTGR-RGA incremental path."
            )
        return "geometry"
    def _ordered_unique(self, values: Iterable[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for v in values:
            iv = int(v)
            if iv not in seen:
                out.append(iv)
                seen.add(iv)
        return out

    def resolve_phase_classes(self, phase: int) -> Tuple[List[int], List[int], List[int]]:
        phase = int(phase)
        if phase <= 0:
            raise ValueError("Incremental phase must be > 0.")
        if not hasattr(self.dataset, "phase_to_classes"):
            raise AttributeError("dataset.phase_to_classes is required.")
        new_classes = self._ordered_unique(int(c) for c in self.dataset.phase_to_classes[phase])
        old_classes: List[int] = []
        if hasattr(self.dataset, "get_classes_up_to_phase"):
            old_classes = self._ordered_unique(int(c) for c in self.dataset.get_classes_up_to_phase(phase - 1))
        else:
            for p in range(phase):
                old_classes.extend(int(c) for c in self.dataset.phase_to_classes[p])
            old_classes = self._ordered_unique(old_classes)
        seen_classes = self._ordered_unique([*old_classes, *new_classes])
        if not old_classes:
            raise RuntimeError("Incremental phase has no old classes. Did phase 0 finalize correctly?")
        if not new_classes:
            raise RuntimeError(f"Phase {phase} has no new classes.")
        if len(seen_classes) != len(old_classes) + len(new_classes):
            overlap = sorted(set(old_classes).intersection(new_classes))
            raise RuntimeError(f"Old/new class overlap in phase {phase}: {overlap}")
        return old_classes, new_classes, seen_classes

    # Backward-compatible name used by older trainer code.
    def _seen_classes_for_phase(self, phase: int) -> List[int]:
        return self.resolve_phase_classes(int(phase))[2] if int(phase) > 0 else self._ordered_unique(self.dataset.phase_to_classes[0])

    def global_to_seen_local(self, labels_global: torch.Tensor, seen_classes: Sequence[int]) -> torch.Tensor:
        labels_global = labels_global.long().view(-1)
        mapping = {int(c): i for i, c in enumerate([int(x) for x in seen_classes])}
        local = torch.full_like(labels_global, -1)
        for global_id, local_id in mapping.items():
            local[labels_global == int(global_id)] = int(local_id)
        if (local < 0).any():
            bad = labels_global[local < 0].detach().cpu().unique().tolist()
            raise RuntimeError(f"Labels not in seen_classes. bad={bad}, seen={list(map(int, seen_classes))}")
        return local

    def seen_local_to_global(self, preds_local: torch.Tensor, seen_classes: Sequence[int]) -> torch.Tensor:
        preds_local = preds_local.long().view(-1)
        seen = torch.as_tensor([int(c) for c in seen_classes], device=preds_local.device, dtype=torch.long)
        if preds_local.numel() == 0:
            return preds_local
        if int(preds_local.min().item()) < 0 or int(preds_local.max().item()) >= int(seen.numel()):
            raise RuntimeError(
                f"Local predictions [{int(preds_local.min())},{int(preds_local.max())}] incompatible with {int(seen.numel())} seen classes."
            )
        return seen.index_select(0, preds_local)

    def assert_valid_seen_targets(self, targets_local: torch.Tensor, num_seen: int, context: str = "CE") -> None:
        targets_local = targets_local.long().view(-1)
        if targets_local.numel() == 0:
            raise RuntimeError(f"{context}: empty CE target tensor.")
        if int(targets_local.min().item()) < 0 or int(targets_local.max().item()) >= int(num_seen):
            raise RuntimeError(
                f"{context}: local targets [{int(targets_local.min())},{int(targets_local.max())}] outside [0,{int(num_seen)-1}]."
            )

    def assert_global_labels_in_set(self, labels_global: torch.Tensor, allowed_classes: Iterable[int], context: str) -> None:
        labels_global = labels_global.long().view(-1)
        allowed = torch.as_tensor([int(c) for c in allowed_classes], device=labels_global.device, dtype=torch.long)
        if labels_global.numel() == 0:
            raise RuntimeError(f"{context}: empty label tensor.")
        if allowed.numel() == 0:
            raise RuntimeError(f"{context}: empty allowed class set.")
        if hasattr(torch, "isin"):
            ok = torch.isin(labels_global, allowed).all()
        else:
            mask = torch.zeros_like(labels_global, dtype=torch.bool)
            for c in allowed:
                mask |= labels_global == int(c)
            ok = mask.all()
        if not bool(ok.item()):
            bad = labels_global[~torch.isin(labels_global, allowed)].detach().cpu().unique().tolist() if hasattr(torch, "isin") else labels_global.detach().cpu().unique().tolist()
            raise RuntimeError(f"{context}: labels outside allowed set. bad={bad}, allowed={allowed.detach().cpu().tolist()}")

    # Older name in the uploaded file.
    def _assert_batch_labels_in_classes(self, y: torch.Tensor, class_ids: Iterable[int], context: str) -> None:
        self.assert_global_labels_in_set(y, class_ids, context)

    # ------------------------------------------------------------------
    # Canonical feature extraction and classifier scoring
    # ------------------------------------------------------------------
    def _prepare_real_spectral_summary(
        self,
        x: torch.Tensor,
        spectra: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Return aligned physical centre spectra for spectral-coupled replay.

        PCA affects the model input only. Raw centre spectra with a different width
        remain physical metadata and are required to fit the spectral tangent and
        spectral-to-feature coupling. A same-width tensor under PCA is treated as
        non-physical unless explicitly allowed.
        """
        if not torch.is_tensor(spectra) or spectra.numel() == 0:
            return None, False
        s = spectra.to(device=x.device, dtype=torch.float32, non_blocking=True)
        if s.dim() == 4:
            s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
        elif s.dim() == 3:
            if s.size(0) == x.size(0) and s.size(1) > 0:
                s = s[:, :, s.size(-1) // 2]
            else:
                s = s.flatten(1)
        elif s.dim() == 1:
            if s.numel() % max(int(x.size(0)), 1) != 0:
                return None, False
            s = s.view(x.size(0), -1)
        elif s.dim() > 4:
            s = s.flatten(1)
        if s.dim() != 2 or s.size(0) != x.size(0):
            return None, False
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        input_channels = int(x.size(1)) if x.dim() >= 2 else 0
        pca_active = int(getattr(self.args, "pca_components", 0) or 0) > 0 and not bool(getattr(self.args, "no_pca", False))
        explicit = getattr(self.args, "incremental_spectral_summary_is_physical", None)
        if explicit is None:
            explicit = getattr(self.args, "raw_spectral_summary_is_physical", None)
        if explicit is None:
            physical = not (pca_active and input_channels > 0 and int(s.size(1)) == input_channels)
        else:
            physical = _as_bool(explicit)
        if self._inc_cfg_bool("force_nonphysical_spectral_summary", False):
            physical = False
        if pca_active and input_channels > 0 and int(s.size(1)) == input_channels and not self._inc_cfg_bool("allow_nonphysical_spectral_summary", False):
            physical = False
        return s, bool(physical)

    def extract_incremental_features(self, x: torch.Tensor, spectra: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = x.float().to(self.device, non_blocking=True)
        spectral_summary, spectral_is_physical = self._prepare_real_spectral_summary(x, spectra)
        fn_names = ("forward_features", "extract_features", "extract_projected_features", "extract_geometry_features")
        for name in fn_names:
            fn = getattr(self.model, name, None)
            if callable(fn):
                try:
                    out = fn(x, spectral_summary=spectral_summary, spectral_summary_is_physical=spectral_is_physical)
                except TypeError:
                    try:
                        out = fn(x)
                    except TypeError:
                        continue
                if isinstance(out, dict):
                    z = out.get("features", out.get("projected_features", out.get("z", None)))
                    if torch.is_tensor(z):
                        z = _finite(z.float(), "incremental features")
                        if z.dim() != 2:
                            raise RuntimeError(f"Canonical incremental features must be [B,D], got {tuple(z.shape)}")
                        out = dict(out)
                        out["features"] = z
                        out["projected_features"] = z
                        out["spectral_summary"] = spectral_summary if spectral_summary is not None else out.get("spectral_summary", None)
                        out["spectral_summary_is_physical"] = bool(spectral_is_physical)
                        self._assert_feature_dim_matches_bank(z)
                        return out
                elif torch.is_tensor(out):
                    z = _finite(out.float(), "incremental features")
                    if z.dim() != 2:
                        raise RuntimeError(f"Canonical incremental features must be [B,D], got {tuple(z.shape)}")
                    self._assert_feature_dim_matches_bank(z)
                    return {"features": z, "projected_features": z, "spectral_summary": spectral_summary, "spectral_summary_is_physical": bool(spectral_is_physical)}
        # Fallback through model forward.
        try:
            out = self.model(x, seen_classes=getattr(self, "_active_seen_classes", None), mode="geometry")
        except TypeError:
            out = self.model(x)
        if not isinstance(out, dict) or not torch.is_tensor(out.get("features", None)):
            raise RuntimeError("Model must expose forward_features/extract_features/extract_projected_features returning canonical z.")
        z = _finite(out["features"].float(), "incremental features")
        if z.dim() != 2:
            raise RuntimeError(f"Canonical incremental features must be [B,D], got {tuple(z.shape)}")
        out = dict(out)
        out["features"] = z
        out["projected_features"] = z
        self._assert_feature_dim_matches_bank(z)
        return out

    def _assert_feature_dim_matches_bank(self, features: torch.Tensor) -> None:
        gb = getattr(self.model, "geometry_bank", None)
        dim = getattr(gb, "feature_dim", None)
        if dim is not None and int(dim) > 0 and int(features.size(1)) != int(dim):
            raise RuntimeError(f"Feature dim {int(features.size(1))} != GeometryBank feature_dim {int(dim)}")

    def compute_seen_logits(
        self,
        features: torch.Tensor,
        seen_classes: Sequence[int],
        *,
        mode: Optional[str] = None,
        return_diagnostics: bool = False,
        old_classes: Optional[Sequence[int]] = None,
        new_classes: Optional[Sequence[int]] = None,
        targets_local: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, float]]:
        """Score exactly the requested seen classes; global-width fallbacks are forbidden."""
        features = _finite(features.float(), "features for seen logits")
        if features.dim() != 2:
            raise RuntimeError(f"features must be [B,D], got {tuple(features.shape)}")
        seen = [int(c) for c in seen_classes]
        if not seen:
            raise RuntimeError("seen_classes is empty.")
        self.assert_geometry_exists(seen, context="compute_seen_logits")
        mode = str(mode or self._classifier_mode()).lower().strip()
        mode = "geometry" if mode == "geometry_only" else mode
        if mode != "geometry":
            raise RuntimeError(f"Only geometry scoring is supported, got {mode!r}.")
        if hasattr(self.model, "compute_logits_from_features"):
            out = self.model.compute_logits_from_features(
                features,
                seen_classes=seen,
                geometry_bank=self._bank_object(),
                mode="geometry",
                old_classes=list(old_classes or []),
                new_classes=list(new_classes or []),
                targets=targets_local,
                return_diagnostics=return_diagnostics,
            )
        elif hasattr(self.model, "classifier"):
            out = self.model.classifier(
                features,
                seen_classes=seen,
                geometry_bank=self._bank_object(),
                mode="geometry",
                old_classes=list(old_classes or []),
                new_classes=list(new_classes or []),
                targets=targets_local,
                return_diagnostics=return_diagnostics,
            )
        else:
            raise AttributeError("model must expose compute_logits_from_features() or classifier().")
        result = dict(out) if isinstance(out, dict) else {"logits": out}
        logits = result.get("logits", None)
        if not torch.is_tensor(logits):
            raise RuntimeError("Classifier output does not contain tensor logits.")
        logits = _finite(logits.float(), "seen logits")
        expected = (features.size(0), len(seen))
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"Geometry classifier must return exact seen-local logits {expected}; got {tuple(logits.shape)}. "
                "Do not return global logits and slice them in the trainer."
            )
        result["logits"] = logits
        if targets_local is not None:
            self.assert_valid_seen_targets(targets_local.to(logits.device), len(seen), context="seen logits CE")
        return result

    def _stable_ce_seen(
        self,
        logits_seen: torch.Tensor,
        labels_global: torch.Tensor,
        seen_classes: Sequence[int],
        context: str,
    ) -> torch.Tensor:
        """Class-balanced CE so class count, replay count, and imbalance cannot bias training."""
        logits_seen = _finite(logits_seen.float(), f"{context} logits")
        if logits_seen.dim() != 2 or logits_seen.size(1) != len(seen_classes):
            raise RuntimeError(f"{context}: logits must be [B,{len(seen_classes)}], got {tuple(logits_seen.shape)}")
        labels_local = self.global_to_seen_local(labels_global.to(logits_seen.device), seen_classes)
        self.assert_valid_seen_targets(labels_local, len(seen_classes), context=context)
        if labels_local.numel() != logits_seen.size(0):
            raise RuntimeError(f"{context}: label/logit batch mismatch {labels_local.numel()} vs {logits_seen.size(0)}")
        clip = self._inc_cfg_float("ce_logit_clip", 50.0)
        per_sample = F.cross_entropy(
            logits_seen.clamp(-clip, clip), labels_local, reduction="none",
            label_smoothing=self._inc_cfg_float("label_smoothing", 0.0),
        )
        class_losses = [per_sample[labels_local == c].mean() for c in torch.unique(labels_local, sorted=True)]
        return torch.stack(class_losses).mean()

    # Backward-compatible old helper: expects local labels for seen-width logits.
    def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(device=logits.device).long().view(-1)
        self.assert_valid_seen_targets(labels, logits.size(1), context="legacy CE")
        clip = self._inc_cfg_float("ce_logit_clip", 50.0)
        per_sample = F.cross_entropy(
            logits.clamp(-clip, clip), labels, reduction="none",
            label_smoothing=self._inc_cfg_float("label_smoothing", 0.0),
        )
        return torch.stack([per_sample[labels == c].mean() for c in torch.unique(labels, sorted=True)]).mean()

    # ------------------------------------------------------------------
    # GeometryBank access and immutability checks
    # ------------------------------------------------------------------
    def _bank_object(self):
        gb = getattr(self.model, "geometry_bank", None)
        if gb is None:
            raise AttributeError("model.geometry_bank is required.")
        return gb

    def _bank_dict(self) -> Dict[str, torch.Tensor]:
        gb = self._bank_object()
        if hasattr(gb, "get_bank"):
            bank = gb.get_bank()
        elif hasattr(gb, "get_subspace_bank"):
            bank = gb.get_subspace_bank()
        elif hasattr(self.model, "get_subspace_bank"):
            bank = self.model.get_subspace_bank()
        else:
            names = (
                "means", "bases", "eigvals", "res_vars", "sample_counts", "active_ranks",
                "reliability", "spectral_prototypes", "band_importances",
            )
            bank = {name: getattr(gb, name) for name in names if torch.is_tensor(getattr(gb, name, None))}
        if not isinstance(bank, dict):
            raise RuntimeError("GeometryBank export must be a dict.")
        return self._canonical_bank_dict(bank)

    def _canonical_bank_dict(self, bank: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        aliases = {
            "means": ("means", "mu"),
            "bases": ("bases", "basis", "U"),
            "eigvals": ("eigvals", "eigenvalues", "lambdas"),
            "res_vars": ("res_vars", "resvars", "residual_vars", "sigma_c2"),
            "sample_counts": ("sample_counts", "counts", "n"),
            "active_ranks": ("active_ranks", "ranks"),
            "reliability": ("reliability", "feature_reliability"),
            "feature_reliability": ("feature_reliability",),
            "spectral_prototypes": ("spectral_prototypes", "spectral_prototype", "spectral_means"),
            "band_importance": ("band_importance", "band_importances", "band"),
            "band_reliability": ("band_reliability",),
            "spectral_reliability": ("spectral_reliability",),
            "spectral_bases": ("spectral_bases",),
            "spectral_eigvals": ("spectral_eigvals",),
            "spectral_res_vars": ("spectral_res_vars",),
            "spectral_active_ranks": ("spectral_active_ranks",),
            "spectral_to_feature": ("spectral_to_feature",),
            "coupling_residual_vars": ("coupling_residual_vars",),
            "coupling_reliability": ("coupling_reliability",),
            "spectral_sam_limits": ("spectral_sam_limits",),
            "spectral_d1_limits": ("spectral_d1_limits",),
            "spectral_d2_limits": ("spectral_d2_limits",),
            "energy_quantiles": ("energy_quantiles",),
            "margin_quantiles": ("margin_quantiles",),
            "phase_created": ("phase_created",),
            "frozen_class_mask": ("frozen_class_mask",),
        }
        for key, names in aliases.items():
            for name in names:
                value = bank.get(name, None)
                if torch.is_tensor(value):
                    out[key] = value.to(self.device)
                    break
        if "eigvals" not in out and torch.is_tensor(bank.get("variances", None)):
            out["eigvals"] = bank["variances"].to(self.device)[:, :-1]
            out["res_vars"] = bank["variances"].to(self.device)[:, -1]
        required = ("means", "bases", "eigvals", "res_vars", "sample_counts")
        missing = [k for k in required if k not in out]
        if missing:
            raise RuntimeError(f"GeometryBank missing required tensors: {missing}")
        if "active_ranks" not in out:
            out["active_ranks"] = torch.full((out["means"].size(0),), out["bases"].size(-1), device=self.device, dtype=torch.long)
        return out

    def assert_geometry_exists(self, class_ids: Iterable[int], context: str) -> None:
        ids = [int(c) for c in class_ids]
        if not ids:
            raise RuntimeError(f"{context}: empty class id list.")
        gb = self._bank_object()
        if hasattr(gb, "assert_bank_valid"):
            try:
                gb.assert_bank_valid(seen_classes=ids)
                return
            except TypeError:
                gb.assert_bank_valid()
        bank = self._bank_dict()
        counts = bank["sample_counts"].flatten()
        max_id = max(ids)
        if max_id >= counts.numel():
            raise RuntimeError(f"{context}: bank has {counts.numel()} rows but needs class {max_id}")
        bad = [c for c in ids if float(counts[c].detach().cpu().item()) <= 0]
        if bad:
            raise RuntimeError(f"{context}: missing GeometryBank rows for classes {bad}")

    def freeze_old_geometry(self, old_classes: Sequence[int]) -> None:
        gb = self._bank_object()
        old = [int(c) for c in old_classes]
        if hasattr(gb, "freeze_classes"):
            gb.freeze_classes(old)
        elif hasattr(gb, "freeze_classes_up_to") and old:
            # Safe only for sequential old classes; otherwise fall back to mask if present.
            if old == list(range(max(old) + 1)):
                gb.freeze_classes_up_to(max(old) + 1)
            elif hasattr(gb, "frozen_mask"):
                gb.frozen_mask[torch.as_tensor(old, device=gb.frozen_mask.device)] = True
        elif hasattr(gb, "frozen_class_mask"):
            gb.frozen_class_mask[torch.as_tensor(old, device=gb.frozen_class_mask.device)] = True
        if hasattr(gb, "assert_bank_valid"):
            try:
                gb.assert_bank_valid(seen_classes=old)
            except TypeError:
                gb.assert_bank_valid()

    def snapshot_old_geometry(self, old_classes: Sequence[int]) -> Dict[str, torch.Tensor]:
        old = [int(c) for c in old_classes]
        self.assert_geometry_exists(old, context="snapshot_old_geometry")
        gb = self._bank_object()
        if hasattr(gb, "snapshot_rows"):
            return gb.snapshot_rows(old)
        bank = self._bank_dict()
        ids = torch.as_tensor(old, device=self.device, dtype=torch.long)
        snap: Dict[str, torch.Tensor] = {"class_ids": ids.detach().clone()}
        for key, value in bank.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) > int(ids.max().item()):
                snap[key] = value.index_select(0, ids).detach().clone()
        return snap

    # Compatibility name used by uploaded code.
    def _snapshot_old_bank_clean(self, old_class_count: int) -> Dict[str, torch.Tensor]:
        return self.snapshot_old_geometry(list(range(int(old_class_count))))

    def assert_old_geometry_unchanged(
        self,
        snapshot: Dict[str, torch.Tensor],
        context: str,
        atol: float = 1e-6,
    ) -> Dict[str, float]:
        gb = self._bank_object()
        if hasattr(gb, "assert_rows_unchanged"):
            gb.assert_rows_unchanged(snapshot, context=context, atol=atol, rtol=0.0)
        ids = snapshot["class_ids"].to(self.device).long()
        bank = self._bank_dict()
        drift: Dict[str, float] = {}
        for key in ("means", "bases", "eigvals", "res_vars", "spectral_bases", "spectral_to_feature"):
            if key not in snapshot or key not in bank:
                continue
            cur = bank[key].index_select(0, ids).detach()
            ref = snapshot[key].to(cur.device, cur.dtype)
            delta = (cur - ref).abs().max() if cur.numel() else torch.tensor(0.0, device=self.device)
            drift[f"old_{key}_max_abs_drift"] = float(delta.cpu().item())
            if float(delta.cpu().item()) > float(atol):
                raise RuntimeError(f"{context}: frozen old geometry changed for {key}; max_abs={float(delta):.6g}")
        return drift

    # ------------------------------------------------------------------
    # New geometry construction and safe admission
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch):
        if hasattr(self, "_unpack_hsi_batch"):
            return self._unpack_hsi_batch(batch)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
                return x, y, None, None
            if len(batch) == 3:
                x, y, spectra = batch
                return x, y, spectra, None
            return batch[0], batch[1], batch[2] if len(batch) > 2 else None, batch[3] if len(batch) > 3 else None
        raise RuntimeError("Cannot unpack HSI batch.")

    @torch.no_grad()
    def collect_current_phase_features(
        self,
        loader,
        new_classes: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.model.eval()
        feats: List[torch.Tensor] = []
        labs: List[torch.Tensor] = []
        spectra_rows: List[torch.Tensor] = []
        bands: List[torch.Tensor] = []
        physical_flags: List[bool] = []
        for batch in loader:
            x, y, spectra, _ = self._unpack_batch(batch)
            x = x.float().to(self.device, non_blocking=True)
            y = y.long().to(self.device, non_blocking=True).view(-1)
            self.assert_global_labels_in_set(y, new_classes, "current incremental train loader")
            out = self.extract_incremental_features(x, spectra)
            z = out["features"].detach()
            feats.append(z)
            labs.append(y.detach())
            s = out.get("spectral_summary", None)
            if torch.is_tensor(s) and s.dim() == 2 and s.size(0) == z.size(0):
                spectra_rows.append(s.detach().float())
                physical_flags.append(bool(out.get("spectral_summary_is_physical", False)))
            b = out.get("band_summary", out.get("band_importance", None))
            if torch.is_tensor(b) and b.dim() == 2 and b.size(0) == z.size(0):
                bands.append(b.detach().float())
        if not feats:
            raise RuntimeError("Current phase train loader produced no features.")
        z_all = torch.cat(feats, dim=0)
        y_all = torch.cat(labs, dim=0)
        s_all = torch.cat(spectra_rows, dim=0) if spectra_rows and sum(s.size(0) for s in spectra_rows) == z_all.size(0) else None
        b_all = torch.cat(bands, dim=0) if bands and sum(b.size(0) for b in bands) == z_all.size(0) else None
        physical = bool(s_all is not None and physical_flags and all(physical_flags))
        self._current_spectral_summary_is_physical = physical
        if self._inc_cfg_bool("use_spectral_coupled_replay", True) and s_all is None:
            print("[SCTGR WARN] Raw centre spectra were unavailable; new rows will use conservative feature-geometry fallback.")
        elif self._inc_cfg_bool("use_spectral_coupled_replay", True) and not physical:
            print("[SCTGR WARN] Spectral metadata is non-physical; coupling is disabled for these new rows.")
        return z_all, y_all, s_all, b_all

    def _estimate_geometry_from_features(
        self,
        features: torch.Tensor,
        labels_global: torch.Tensor,
        class_ids: Sequence[int],
        spectral_summary: Optional[torch.Tensor] = None,
        band_summary: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Delegate all feature/spectral/coupling estimation to GeometryBank."""
        features = _finite(features.float(), "new descriptor features")
        labels_global = labels_global.long().view(-1).to(features.device)
        gb = self._bank_object()
        builder = getattr(gb, "build_candidate_geometry_rows", getattr(gb, "extract_geometry", None))
        if not callable(builder):
            raise AttributeError("GeometryBank must expose build_candidate_geometry_rows/extract_geometry.")
        rows = builder(
            features,
            labels_global,
            spectral_summary=spectral_summary,
            band_weights=band_summary,
            spectral_summary_is_physical=bool(getattr(self, "_current_spectral_summary_is_physical", False)),
            class_ids=class_ids,
        )
        ids = [int(c) for c in class_ids]
        missing = [c for c in ids if c not in rows]
        if missing:
            raise RuntimeError(f"GeometryBank failed to build new rows for classes {missing}.")
        singular_to_block = {
            "mean": "means", "basis": "bases", "eigvals": "eigvals", "res_var": "res_vars",
            "active_rank": "active_ranks", "sample_count": "sample_counts",
            "reliability": "reliability", "feature_reliability": "feature_reliability",
            "band_importance": "band_importance", "band_reliability": "band_reliability",
            "spectral_prototype": "spectral_prototypes", "spectral_reliability": "spectral_reliability",
            "spectral_basis": "spectral_bases", "spectral_eigvals": "spectral_eigvals",
            "spectral_res_var": "spectral_res_vars", "spectral_active_rank": "spectral_active_ranks",
            "spectral_to_feature": "spectral_to_feature", "coupling_residual_vars": "coupling_residual_vars",
            "coupling_reliability": "coupling_reliability", "spectral_sam_limit": "spectral_sam_limits",
            "spectral_d1_limit": "spectral_d1_limits", "spectral_d2_limit": "spectral_d2_limits",
            "energy_quantiles": "energy_quantiles", "margin_quantiles": "margin_quantiles",
        }
        result: Dict[str, torch.Tensor] = {
            "class_ids": torch.as_tensor(ids, device=features.device, dtype=torch.long)
        }
        for singular, plural in singular_to_block.items():
            if all(torch.is_tensor(rows[c].get(singular, None)) for c in ids):
                result[plural] = torch.stack([rows[c][singular].to(features.device) for c in ids], dim=0)
        self._assert_descriptor_block_valid(result, context="estimated replay-ready new geometry")
        return result

    def _assert_descriptor_block_valid(self, desc: Dict[str, torch.Tensor], context: str) -> None:
        means = desc["means"]
        bases = desc["bases"]
        eig = desc["eigvals"]
        res = desc["res_vars"]
        if means.dim() != 2:
            raise RuntimeError(f"{context}: means must be [K,D], got {tuple(means.shape)}")
        if bases.dim() != 3 or bases.size(0) != means.size(0) or bases.size(1) != means.size(1):
            raise RuntimeError(f"{context}: bases must be [K,D,R], got {tuple(bases.shape)} with means {tuple(means.shape)}")
        if eig.shape != (means.size(0), bases.size(2)):
            raise RuntimeError(f"{context}: eigvals shape {tuple(eig.shape)} incompatible with bases {tuple(bases.shape)}")
        if res.numel() != means.size(0):
            raise RuntimeError(f"{context}: res_vars length mismatch")
        for name, t in (("means", means), ("bases", bases), ("eigvals", eig), ("res_vars", res)):
            _finite(t, f"{context}.{name}")
        if bool((eig < 0).any().item()) or bool((res < 0).any().item()):
            raise RuntimeError(f"{context}: negative variances/eigenvalues.")
        gram = torch.matmul(bases.transpose(1, 2), bases)
        eye = torch.eye(bases.size(2), device=bases.device, dtype=bases.dtype).unsqueeze(0)
        err = (gram - eye).abs().max()
        if float(err.detach().cpu().item()) > 5e-3:
            raise RuntimeError(f"{context}: bases are not orthonormal; max gram error={float(err):.6f}")

    def _commit_new_descriptors(self, desc: Dict[str, torch.Tensor], phase: int, *, freeze: bool = False) -> None:
        gb = self._bank_object()
        ids = [int(c) for c in desc["class_ids"].detach().cpu().tolist()]
        block_to_singular = {
            "means": "mean", "bases": "basis", "eigvals": "eigvals", "res_vars": "res_var",
            "active_ranks": "active_rank", "sample_counts": "sample_count",
            "reliability": "reliability", "feature_reliability": "feature_reliability",
            "band_importance": "band_importance", "band_reliability": "band_reliability",
            "spectral_prototypes": "spectral_prototype", "spectral_reliability": "spectral_reliability",
            "spectral_bases": "spectral_basis", "spectral_eigvals": "spectral_eigvals",
            "spectral_res_vars": "spectral_res_var", "spectral_active_ranks": "spectral_active_rank",
            "spectral_to_feature": "spectral_to_feature", "coupling_residual_vars": "coupling_residual_vars",
            "coupling_reliability": "coupling_reliability", "spectral_sam_limits": "spectral_sam_limit",
            "spectral_d1_limits": "spectral_d1_limit", "spectral_d2_limits": "spectral_d2_limit",
            "energy_quantiles": "energy_quantiles", "margin_quantiles": "margin_quantiles",
        }
        rows: Dict[int, Dict[str, torch.Tensor]] = {}
        for row_idx, cls in enumerate(ids):
            row: Dict[str, torch.Tensor] = {}
            for block_key, singular in block_to_singular.items():
                value = desc.get(block_key, None)
                if torch.is_tensor(value) and value.size(0) == len(ids):
                    row[singular] = value[row_idx].detach()
            rows[cls] = row
        if hasattr(gb, "commit_candidate_geometry_rows"):
            gb.commit_candidate_geometry_rows(
                rows, allow_frozen_update=False, phase_created=int(phase), freeze=bool(freeze),
                context=f"phase_{int(phase)}_new_row_commit",
            )
        else:
            for cls, row in rows.items():
                gb.add_or_update_class_geometry(class_id=cls, phase_created=int(phase), freeze=bool(freeze), **row)
        gb.assert_bank_valid(seen_classes=ids)

    def _risk_matrix_from_descriptors(
        self,
        old_snapshot: Dict[str, torch.Tensor],
        new_desc: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Differentiable old/new geometry risk using active ranks and normalized energy."""
        old_mu = old_snapshot["means"].to(self.device).float().detach()
        old_U = old_snapshot["bases"].to(self.device).float().detach()
        old_e = old_snapshot["eigvals"].to(self.device).float().detach().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        old_rv = old_snapshot["res_vars"].to(self.device).float().detach().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        old_r = old_snapshot.get("active_ranks", torch.full((old_mu.size(0),), old_U.size(-1), device=self.device, dtype=torch.long)).to(self.device).long()
        new_mu = new_desc["means"].to(self.device).float()
        new_U = new_desc["bases"].to(self.device).float()
        new_e = new_desc["eigvals"].to(self.device).float().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        new_rv = new_desc["res_vars"].to(self.device).float().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        new_r = new_desc.get("active_ranks", torch.full((new_mu.size(0),), new_U.size(-1), device=self.device, dtype=torch.long)).to(self.device).long()

        dist = torch.cdist(old_mu, new_mu, p=2)
        old_scale = torch.stack([old_e[i, :max(int(old_r[i].item()), 1)].mean() for i in range(old_mu.size(0))]).view(-1, 1)
        new_scale = torch.stack([new_e[j, :max(int(new_r[j].item()), 1)].mean() for j in range(new_mu.size(0))]).view(1, -1)
        norm_dist = dist / (old_scale + new_scale).sqrt().clamp_min(_EPS)
        center = torch.exp(-norm_dist / max(self._inc_cfg_float("risk_center_temperature", 3.0), 1e-6)).clamp(0.0, 1.0)

        energy_rows: List[torch.Tensor] = []
        overlap_rows: List[torch.Tensor] = []
        for i in range(old_mu.size(0)):
            ri = max(0, min(int(old_r[i].item()), old_U.size(-1)))
            diff = new_mu - old_mu[i].view(1, -1)
            if ri > 0:
                Ui = old_U[i, :, :ri]
                coord = diff.matmul(Ui)
                parallel = (coord.pow(2) / old_e[i, :ri].view(1, -1)).sum(dim=1) / float(ri)
                residual = diff - coord.matmul(Ui.t())
            else:
                parallel = torch.zeros((new_mu.size(0),), device=self.device, dtype=new_mu.dtype)
                residual = diff
            residual_energy = residual.pow(2).sum(dim=1) / (float(max(old_mu.size(1) - ri, 1)) * old_rv[i])
            energy_rows.append(parallel + residual_energy)
            pair_ov: List[torch.Tensor] = []
            for j in range(new_mu.size(0)):
                rj = max(0, min(int(new_r[j].item()), new_U.size(-1)))
                if ri <= 0 or rj <= 0:
                    pair_ov.append(new_mu.new_tensor(0.0))
                else:
                    pair_ov.append(old_U[i, :, :ri].t().matmul(new_U[j, :, :rj]).pow(2).sum() / float(max(min(ri, rj), 1)))
            overlap_rows.append(torch.stack(pair_ov))
        energy = torch.stack(energy_rows, dim=0)
        subspace = torch.stack(overlap_rows, dim=0).clamp(0.0, 1.0)
        mahal = torch.exp(-energy / max(self._inc_cfg_float("risk_energy_temperature", 1.0), 1e-6)).clamp(0.0, 1.0)

        band = torch.zeros_like(subspace)
        if torch.is_tensor(old_snapshot.get("band_importances", old_snapshot.get("band_importance", None))) and torch.is_tensor(new_desc.get("band_importance", None)):
            ob = old_snapshot.get("band_importances", old_snapshot.get("band_importance")).to(self.device).float()
            nb = new_desc["band_importance"].to(self.device).float()
            if ob.dim() == 2 and nb.dim() == 2 and ob.size(1) == nb.size(1):
                band = F.normalize(ob, p=2, dim=1).matmul(F.normalize(nb, p=2, dim=1).t()).clamp(0.0, 1.0)
        spectral = torch.zeros_like(subspace)
        if torch.is_tensor(old_snapshot.get("spectral_prototypes", None)) and torch.is_tensor(new_desc.get("spectral_prototypes", None)):
            os = old_snapshot["spectral_prototypes"].to(self.device).float()
            ns = new_desc["spectral_prototypes"].to(self.device).float()
            if os.dim() == 2 and ns.dim() == 2 and os.size(1) == ns.size(1):
                os = os - os.mean(dim=1, keepdim=True)
                ns = ns - ns.mean(dim=1, keepdim=True)
                spectral = F.normalize(os, p=2, dim=1).matmul(F.normalize(ns, p=2, dim=1).t()).clamp(0.0, 1.0)
        old_rel = old_snapshot.get("reliability", torch.ones((old_mu.size(0),), device=self.device)).to(self.device).float().view(-1, 1)
        uncertainty = (1.0 - old_rel).clamp(0.0, 1.0)
        risk = (
            self._inc_cfg_float("risk_center_weight", 0.35) * center
            + self._inc_cfg_float("risk_mahal_weight", 0.45) * mahal
            + self._inc_cfg_float("risk_subspace_weight", 0.20) * subspace
            + self._inc_cfg_float("risk_band_weight", 0.05) * band * (center + mahal).clamp(max=1.0)
            + self._inc_cfg_float("risk_spectral_shape_weight", 0.05) * spectral * (center + mahal).clamp(max=1.0)
        ) * (1.0 + 0.25 * uncertainty)
        return {
            "risk": risk, "center": center, "mahal": mahal, "subspace": subspace,
            "band": band, "spectral": spectral, "dist": dist, "normalized_dist": norm_dist, "energy": energy,
        }

    def admit_new_geometry(
        self,
        new_desc: Dict[str, torch.Tensor],
        old_snapshot: Dict[str, torch.Tensor],
        new_classes: Sequence[int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Safe new-row admission for the clean SCTGR-RGA path.

        No geometry transport, no old-row transport, and no hidden ablation is
        allowed here. The only permitted operation is a bounded correction of
        the *new* descriptors relative to a detached snapshot of old rows.
        """
        if self._inc_cfg_bool("use_geometry_transport", False) or self._inc_cfg_bool("use_sglat_transport", False) or self._inc_cfg_bool("allow_old_model_transport", False):
            raise RuntimeError(
                "Transport is disabled in the clean incremental path. "
                "Keep old GeometryBank rows frozen and use deterministic new-row admission only."
            )

        risk_before = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
        corrected = self._fallback_safe_new_row_correction(new_desc, old_snapshot, risk_before)
        self._assert_descriptor_block_valid(corrected, context="admitted new geometry")

        risk_after = self._risk_matrix_from_descriptors(old_snapshot, corrected)
        stats = {
            "risk_before_max": float(risk_before["risk"].max().detach().cpu().item()) if risk_before["risk"].numel() else 0.0,
            "risk_before_mean": float(risk_before["risk"].mean().detach().cpu().item()) if risk_before["risk"].numel() else 0.0,
            "risk_after_max": float(risk_after["risk"].max().detach().cpu().item()) if risk_after["risk"].numel() else 0.0,
            "risk_after_mean": float(risk_after["risk"].mean().detach().cpu().item()) if risk_after["risk"].numel() else 0.0,
            "overlap_before_max": float(risk_before["subspace"].max().detach().cpu().item()) if risk_before["subspace"].numel() else 0.0,
            "overlap_after_max": float(risk_after["subspace"].max().detach().cpu().item()) if risk_after["subspace"].numel() else 0.0,
            "transport_active": 0.0,
        }
        return corrected, stats
    def _fallback_safe_new_row_correction(
        self,
        desc: Dict[str, torch.Tensor],
        old_snapshot: Dict[str, torch.Tensor],
        risk_parts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Bounded new-row admission without rotating the coupled feature basis.

        Spectral-to-feature coupling is expressed in the new feature basis. Rotating
        that basis after coupling fit would silently invalidate the replay model.
        Admission may therefore adjust only the new centre and variance scale; the
        feature/spectral bases and coupling matrix remain exactly as estimated.
        """
        out = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in desc.items()}
        risk = risk_parts["risk"].to(self.device)
        if risk.numel() == 0:
            return out
        old_mu = old_snapshot["means"].to(self.device).float().detach()
        means = out["means"].to(self.device).float()
        eig = out["eigvals"].to(self.device).float()
        res = out["res_vars"].to(self.device).float()
        risk_thr = self._inc_cfg_float("descriptor_correction_risk_threshold", 0.60)
        max_shift = self._inc_cfg_float("descriptor_admission_max_mean_shift", 0.15)
        max_log_shrink = self._inc_cfg_float("descriptor_admission_max_logvar_shrink", 0.10)
        floor = self._inc_cfg_float("geom_var_floor", 1e-4)
        for j in range(means.size(0)):
            col = risk[:, j].clamp_min(0.0)
            peak = float(col.max().detach().cpu().item())
            if peak <= risk_thr:
                continue
            w = torch.softmax(col / max(self._inc_cfg_float("pair_risk_temperature", 0.75), 1e-6), dim=0)
            push = torch.zeros_like(means[j])
            for i in range(old_mu.size(0)):
                direction = means[j] - old_mu[i]
                push = push + w[i] * direction / direction.norm().clamp_min(_EPS)
            severity = min(1.0, max(0.0, (peak - risk_thr) / max(1.0 - risk_thr, 1e-6)))
            if push.norm() > _EPS:
                means[j] = means[j] + max_shift * severity * push / push.norm().clamp_min(_EPS)
            shrink = math.exp(-max_log_shrink * severity)
            eig[j] = (eig[j] * shrink).clamp_min(floor)
            res[j] = (res[j] * math.sqrt(shrink)).clamp_min(floor)
        out["means"], out["eigvals"], out["res_vars"] = means, eig, res
        return out

    # ------------------------------------------------------------------
    # Replay from frozen old geometry
    # ------------------------------------------------------------------
    def sample_old_replay(
        self,
        old_snapshot: Dict[str, torch.Tensor],
        seen_classes: Sequence[int],
        new_desc: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, Any]]:
        """Generate geometry-valid core and directed spectral-tangent replay.

        Core and directed quotas are selected independently. This prevents easy core
        candidates from silently consuming the directed replay budget.
        """
        gb = self._bank_object()
        old_classes = [int(c) for c in old_snapshot["class_ids"].detach().cpu().tolist()]
        new_classes = [int(c) for c in new_desc["class_ids"].detach().cpu().tolist()] if isinstance(new_desc, dict) and torch.is_tensor(new_desc.get("class_ids", None)) else [c for c in seen_classes if c not in old_classes]
        base = self._inc_cfg_int("gfa_samples_per_class", self._inc_cfg_int("synthetic_replay_per_class", 48))
        min_count = self._inc_cfg_int("gfa_min_samples_per_class", max(8, base // 2))
        max_count = self._inc_cfg_int("gfa_max_samples_per_class", max(base, int(round(base * self._inc_cfg_float("gfa_max_replay_multiplier", 2.0)))))
        if hasattr(gb, "adaptive_replay_sample_counts"):
            target_counts = gb.adaptive_replay_sample_counts(
                old_classes, new_classes, base_samples_per_class=base,
                min_samples_per_class=min_count, max_samples_per_class=max_count,
                risk_weight=self._inc_cfg_float("replay_risk_weight", 0.75),
                unreliability_weight=self._inc_cfg_float("replay_unreliability_weight", 0.50),
            )
        else:
            target_counts = {c: base for c in old_classes}
        if hasattr(gb, "adaptive_directed_replay_ratios"):
            directed_ratios = gb.adaptive_directed_replay_ratios(
                old_classes, new_classes,
                min_ratio=self._inc_cfg_float("directed_replay_min_ratio", 0.10),
                max_ratio=self._inc_cfg_float("directed_replay_max_ratio", 0.40),
            )
        else:
            directed_ratios = {c: self._inc_cfg_float("directed_replay_ratio", 0.15) for c in old_classes}
        target_directed = {
            c: (int(round(target_counts[c] * directed_ratios[c])) if new_classes else 0)
            for c in old_classes
        }
        target_core = {c: target_counts[c] - target_directed[c] for c in old_classes}
        pair_risk = gb.adaptive_geometry_risk_matrix() if hasattr(gb, "adaptive_geometry_risk_matrix") and new_classes else None
        multiplier = max(1, self._inc_cfg_int("replay_energy_filter_multiplier", 3))
        max_rounds = max(1, self._inc_cfg_int("replay_resample_rounds", 4))
        accepted_x: Dict[Tuple[int, int], List[torch.Tensor]] = {(c, p): [] for c in old_classes for p in (0, 1)}
        generated = accepted = 0
        pool_generated = {0: 0, 1: 0}; pool_accepted = {0: 0, 1: 0}
        coupling_values: List[float] = []

        def have(c: int, pool: int) -> int:
            return sum(t.size(0) for t in accepted_x[(c, pool)])

        for _ in range(max_rounds):
            deficits = {
                c: (max(0, target_core[c] - have(c, 0)), max(0, target_directed[c] - have(c, 1)))
                for c in old_classes
            }
            if all(a == 0 and b == 0 for a, b in deficits.values()):
                break
            request: Dict[int, int] = {}
            round_ratio: Dict[int, float] = {}
            for c in old_classes:
                core_need, dir_need = deficits[c]
                total_need = core_need + dir_need
                if total_need <= 0:
                    request[c] = 0; round_ratio[c] = directed_ratios[c]
                    continue
                ratio = directed_ratios[c] if new_classes else 0.0
                if dir_need > 0 and core_need == 0:
                    ratio = max(ratio, 0.70)
                elif core_need > 0 and dir_need == 0:
                    ratio = 0.0
                round_ratio[c] = min(max(ratio, 0.0), 0.80)
                request[c] = max(total_need * multiplier, total_need)
            replay = gb.sample_replay(
                old_classes, samples_per_class=request, seen_classes=seen_classes,
                new_class_ids=new_classes, pair_risk=pair_risk, directed_ratio=round_ratio,
                pair_risk_topk=self._inc_cfg_int("pair_risk_topk", 3),
                residual_scale=self._inc_cfg_float("gfa_residual_scale", 0.25),
                tangent_clip=self._inc_cfg_float("spectral_tangent_clip", 2.5),
                use_spectral_coupled=self._inc_cfg_bool("use_spectral_coupled_replay", True),
            )
            x = replay["features"]; y = replay["global_labels"]
            pools = replay.get("pool_types", torch.zeros_like(y))
            generated += int(y.numel()); accepted_diag = replay.get("diagnostics", {})
            for pool in (0, 1): pool_generated[pool] += int((pools == pool).sum().item())
            filt = gb.filter_replay_by_geometry_energy(
                x, y, seen_classes=seen_classes, pool_types=pools,
                core_margin=self._inc_cfg_float("replay_core_accept_margin", 0.0),
                directed_max_margin=(self._inc_cfg_float("replay_directed_max_margin", 1.0e9) if self._inc_cfg_float("replay_directed_max_margin", 1.0e9) < 1.0e8 else None),
            )
            mask = filt["accept_mask"]
            accepted += int(mask.sum().item())
            for pool in (0, 1): pool_accepted[pool] += int((mask & (pools == pool)).sum().item())
            for c in old_classes:
                for pool, target in ((0, target_core[c]), (1, target_directed[c])):
                    room = target - have(c, pool)
                    if room <= 0: continue
                    idx = torch.nonzero(mask & (y == c) & (pools == pool), as_tuple=False).flatten()[:room]
                    if idx.numel() > 0: accepted_x[(c, pool)].append(x.index_select(0, idx).detach())
            for diag in accepted_diag.values():
                if isinstance(diag, dict) and "coupling_reliability" in diag:
                    coupling_values.append(float(diag["coupling_reliability"]))

        missing = {
            c: {"core": target_core[c] - have(c, 0), "directed": target_directed[c] - have(c, 1)}
            for c in old_classes
        }
        if any(v["core"] > 0 or v["directed"] > 0 for v in missing.values()):
            raise RuntimeError(
                "SCTGR replay could not satisfy geometry-valid core/directed quotas. "
                f"missing={missing}. Inspect spectral coupling or increase candidate/resample settings."
            )
        feat_parts: List[torch.Tensor] = []; pool_parts: List[torch.Tensor] = []; label_parts: List[torch.Tensor] = []
        for c in old_classes:
            for pool, target in ((0, target_core[c]), (1, target_directed[c])):
                if target <= 0: continue
                part = torch.cat(accepted_x[(c, pool)], dim=0)[:target]
                feat_parts.append(part)
                pool_parts.append(torch.full((target,), pool, device=self.device, dtype=torch.long))
                label_parts.append(torch.full((target,), c, device=self.device, dtype=torch.long))
        feats = torch.cat(feat_parts, dim=0); pools = torch.cat(pool_parts, dim=0); labels = torch.cat(label_parts, dim=0)
        local = self.global_to_seen_local(labels, seen_classes)
        stats: Dict[str, Any] = {
            "target_counts": target_counts, "target_core_counts": target_core,
            "target_directed_counts": target_directed, "directed_ratios": directed_ratios,
            "replay_count": int(labels.numel()), "generated_count": generated,
            "acceptance_rate": float(accepted) / float(max(generated, 1)),
            "core_count": int((pools == 0).sum().item()), "directed_count": int((pools == 1).sum().item()),
            "core_acceptance_rate": float(pool_accepted[0]) / float(max(pool_generated[0], 1)),
            "directed_acceptance_rate": float(pool_accepted[1]) / float(max(pool_generated[1], 1)),
            "coupling_reliability_mean": sum(coupling_values) / max(len(coupling_values), 1),
        }
        return {
            "features": _finite(feats, "accepted SCTGR replay").detach(),
            "global_labels": labels.detach(), "local_labels": local.detach(),
            "pool_types": pools.detach(), "stats": stats,
        }
    def _sample_old_anchor_batch(self, old_bank_snapshot: Dict[str, torch.Tensor], old_class_count: int, new_class_ids: Optional[Iterable[int]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Convert old contiguous snapshot from old code to our canonical snapshot if needed.
        snap = old_bank_snapshot
        if "class_ids" not in snap:
            snap = dict(snap)
            snap["class_ids"] = torch.arange(int(old_class_count), device=self.device, dtype=torch.long)
        if "eigvals" not in snap and torch.is_tensor(snap.get("variances", None)):
            snap["eigvals"] = snap["variances"][:, :-1]
            snap["res_vars"] = snap["variances"][:, -1]
        replay = self.sample_old_replay(snap, list(range(int(old_class_count))) + [int(c) for c in (new_class_ids or [])])
        return replay["features"], replay["global_labels"]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Descriptor refinement: optimize only current new rows, old snapshot detached
    # ------------------------------------------------------------------
    def _compose_bank_for_descriptor_params(self, base_bank: Dict[str, torch.Tensor], new_classes: Sequence[int], mu: torch.Tensor, bases: torch.Tensor, eig: torch.Tensor, res: torch.Tensor) -> Dict[str, torch.Tensor]:
        bank = {k: (v.detach().clone().to(self.device) if torch.is_tensor(v) else v) for k, v in base_bank.items()}
        ids = torch.as_tensor([int(c) for c in new_classes], device=self.device, dtype=torch.long)
        bank["means"][ids] = mu
        bank["bases"][ids] = bases
        bank["eigvals"][ids] = eig
        bank["res_vars"][ids] = res
        return bank  # type: ignore[return-value]

    def _orthonormalize_bases(self, raw: torch.Tensor, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        outs = []
        R = raw.size(-1)
        for i in range(raw.size(0)):
            q, _ = torch.linalg.qr(raw[i], mode="reduced")
            q = q[:, :R]
            if torch.is_tensor(reference) and reference.shape == raw.shape:
                sign = torch.where((q * reference[i].to(q.device, q.dtype)).sum(dim=0, keepdim=True) < 0, -torch.ones(1, R, device=q.device, dtype=q.dtype), torch.ones(1, R, device=q.device, dtype=q.dtype))
                q = q * sign
            outs.append(q)
        return torch.stack(outs, dim=0)

    def _geometry_energy_from_bank(
        self,
        features: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        seen_classes: Sequence[int],
    ) -> torch.Tensor:
        ids = torch.as_tensor([int(c) for c in seen_classes], device=features.device, dtype=torch.long)
        mu = bank["means"].to(features.device).index_select(0, ids)
        Uall = bank["bases"].to(features.device).index_select(0, ids)
        eigall = bank["eigvals"].to(features.device).index_select(0, ids).clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        rv = bank["res_vars"].to(features.device).flatten().index_select(0, ids).clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        ranks = bank.get("active_ranks", torch.full((bank["means"].size(0),), Uall.size(-1), device=features.device, dtype=torch.long)).to(features.device).index_select(0, ids).long()
        cols: List[torch.Tensor] = []
        for j in range(len(seen_classes)):
            r = max(0, min(int(ranks[j].item()), Uall.size(-1)))
            diff = features - mu[j].view(1, -1)
            if r > 0:
                U = Uall[j, :, :r]
                coord = diff.matmul(U)
                parallel = (coord.pow(2) / eigall[j, :r].view(1, -1)).sum(dim=1) / float(r)
                residual = diff - coord.matmul(U.t())
            else:
                parallel = torch.zeros((features.size(0),), device=features.device, dtype=features.dtype)
                residual = diff
            residual_energy = residual.pow(2).sum(dim=1) / (float(max(features.size(1) - r, 1)) * rv[j])
            energy = parallel + residual_energy
            if self._inc_cfg_bool("use_logdet_energy", False):
                logdet = torch.log(eigall[j, :r]).mean() if r > 0 else features.new_tensor(0.0)
                logdet = logdet + torch.log(rv[j])
                energy = energy + self._inc_cfg_float("logdet_energy_weight", 0.02) * logdet
            cols.append(energy)
        return torch.stack(cols, dim=1)

    def _geometry_logits_from_bank(self, features: torch.Tensor, bank: Dict[str, torch.Tensor], seen_classes: Sequence[int]) -> torch.Tensor:
        energy = self._geometry_energy_from_bank(features, bank, seen_classes)
        scale = self._inc_cfg_float("loss_scale", 8.0)
        logits = -scale * energy
        return logits.clamp(-self._inc_cfg_float("geometry_logit_clip", 80.0), self._inc_cfg_float("geometry_logit_clip", 80.0))

    def _descriptor_margin_loss(self, old_snapshot: Dict[str, torch.Tensor], new_desc: Dict[str, torch.Tensor]) -> torch.Tensor:
        risk = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
        pair_weight = risk["risk"].detach().clamp_min(0.0)
        if pair_weight.numel() == 0:
            return self._zero_like_ref(new_desc["means"])
        pair_weight = pair_weight / pair_weight.mean().clamp_min(_EPS)
        overlap_target = self._inc_cfg_float("descriptor_subspace_overlap_max", 0.55)
        center_target = self._inc_cfg_float("descriptor_center_risk_max", 0.50)
        sub = (pair_weight * F.relu(risk["subspace"] - overlap_target).pow(2)).mean()
        center = (pair_weight * F.relu(risk["center"] - center_target).pow(2)).mean()
        return self._inc_cfg_float("lambda_subspace", 0.10) * sub + self._inc_cfg_float("lambda_center_collision", 0.05) * center



    def _refine_new_descriptors_impl(
        self,
        *,
        z_new: torch.Tensor,
        y_new: torch.Tensor,
        seen_classes: Sequence[int],
        old_classes: Sequence[int],
        new_classes: Sequence[int],
        old_snapshot: Dict[str, torch.Tensor],
        init_desc: Dict[str, torch.Tensor],
        steps: int,
    ) -> Dict[str, Any]:
        """Refine only new means/eigenvalues/residual variances in canonical space.

        The feature and spectral bases stay fixed so the fitted spectral-to-feature
        coupling remains valid. Old rows are detached and immutable.
        """
        if steps <= 0 or not self._inc_cfg_bool("refine_new_descriptors", True):
            return {"desc": init_desc, "stats": {"loss": 0.0, "ce_new": 0.0, "ce_replay": 0.0, "steps": 0.0}}
        base_bank = self._bank_dict()
        mu0 = init_desc["means"].detach().clone()
        U0 = init_desc["bases"].detach().clone()
        eig0 = init_desc["eigvals"].detach().clone().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        res0 = init_desc["res_vars"].detach().clone().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
        mu = nn.Parameter(mu0.clone())
        log_eig = nn.Parameter(eig0.log())
        log_res = nn.Parameter(res0.log())
        opt = optim.Adam([mu, log_eig, log_res], lr=self._inc_cfg_float("descriptor_refine_lr", 1e-3), weight_decay=0.0)
        max_mean_shift = self._inc_cfg_float("descriptor_refine_max_mean_shift", 0.30)
        max_logvar_shift = self._inc_cfg_float("descriptor_refine_max_logvar_shift", 0.50)
        margin_energy = self._inc_cfg_float("geometry_energy_margin", 0.30)
        invasion_energy = self._inc_cfg_float("old_new_geometry_margin", 0.35)
        scale = self._inc_cfg_float("loss_scale", 8.0)
        new_local = self.global_to_seen_local(y_new, seen_classes)
        old_local_ids = torch.as_tensor([seen_classes.index(c) for c in old_classes], device=self.device, dtype=torch.long)
        new_local_ids = torch.as_tensor([seen_classes.index(c) for c in new_classes], device=self.device, dtype=torch.long)
        counts = init_desc.get("sample_counts", torch.ones((len(new_classes),), device=self.device)).float().clamp_min(1.0)
        rel = init_desc.get("reliability", torch.ones_like(counts)).float().clamp(0.05, 1.0)
        trust_class = 1.0 + self._inc_cfg_float("descriptor_trust_small_class_weight", 1.0) / counts.sqrt() + self._inc_cfg_float("descriptor_trust_unreliable_weight", 1.0) * (1.0 - rel)

        stats = {k: 0.0 for k in ("loss", "ce_new", "ce_replay", "margin", "invasion", "fit", "collision", "trust", "replay_acc", "old_to_new", "new_to_old", "steps")}
        last_replay_stats: Dict[str, Any] = {}
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            eig = log_eig.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
            res = log_res.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
            cur_desc = dict(init_desc)
            cur_desc.update({"means": mu, "bases": U0, "eigvals": eig, "res_vars": res})
            tmp_bank = self._compose_bank_for_descriptor_params(base_bank, new_classes, mu, U0, eig, res)

            replay = self.sample_old_replay(old_snapshot, seen_classes, new_desc=cur_desc)
            z_old = replay["features"].to(self.device)  # type: ignore[index]
            y_old = replay["global_labels"].to(self.device).long()  # type: ignore[index]
            y_old_local = self.global_to_seen_local(y_old, seen_classes)
            last_replay_stats = dict(replay.get("stats", {}))  # type: ignore[arg-type]

            energy_new = self._geometry_energy_from_bank(z_new, tmp_bank, seen_classes)
            energy_old = self._geometry_energy_from_bank(z_old, tmp_bank, seen_classes)
            logits_new = -scale * energy_new
            logits_old = -scale * energy_old
            ce_new = self._stable_ce_seen(logits_new, y_new, seen_classes, "SCTGR new CE")
            ce_replay = self._stable_ce_seen(logits_old, y_old, seen_classes, "SCTGR replay CE")
            joint_ce = self._stable_ce_seen(
                torch.cat([logits_old, logits_new], dim=0),
                torch.cat([y_old, y_new], dim=0),
                seen_classes,
                "SCTGR all-seen class-balanced CE",
            )

            def margin_loss(energy: torch.Tensor, targets: torch.Tensor, margin: float) -> Tuple[torch.Tensor, torch.Tensor]:
                true = energy.gather(1, targets.view(-1, 1)).squeeze(1)
                rival_m = energy.clone()
                rival_m.scatter_(1, targets.view(-1, 1), float("inf"))
                rival = rival_m.min(dim=1).values
                gap = rival - true
                return F.relu(float(margin) - gap).mean(), gap

            m_new, gap_new = margin_loss(energy_new, new_local, margin_energy)
            m_old, gap_old = margin_loss(energy_old, y_old_local, margin_energy)
            margin = 0.5 * (m_new + m_old)

            true_new = energy_new.gather(1, new_local.view(-1, 1)).squeeze(1)
            best_old = energy_new.index_select(1, old_local_ids).min(dim=1).values
            new_to_old = F.relu(invasion_energy - (best_old - true_new)).mean()
            true_old = energy_old.gather(1, y_old_local.view(-1, 1)).squeeze(1)
            best_new = energy_old.index_select(1, new_local_ids).min(dim=1).values
            old_to_new = F.relu(invasion_energy - (best_new - true_old)).mean()
            invasion = new_to_old + old_to_new

            fit = torch.stack([true_new[y_new == int(c)].mean() for c in new_classes]).mean()
            collision = self._descriptor_margin_loss(old_snapshot, cur_desc)
            mean_shift = (mu - mu0).pow(2).mean(dim=1)
            eig_shift = (log_eig - eig0.log()).pow(2).mean(dim=1)
            res_shift = (log_res - res0.log()).pow(2)
            trust = (trust_class * (mean_shift + eig_shift + res_shift)).mean()

            loss = (
                self._inc_cfg_float("joint_old_new_ce_weight", 1.0) * joint_ce
                + self._inc_cfg_float("geometry_energy_margin_weight", 0.30) * margin
                + self._inc_cfg_float("old_new_invasion_weight", 0.50) * invasion
                + self._inc_cfg_float("new_descriptor_fit_weight", 0.10) * fit
                + collision
                + self._inc_cfg_float("descriptor_trust_weight", 0.80) * trust
            )
            _finite(loss, "SCTGR descriptor refinement loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mu, log_eig, log_res], self._inc_cfg_float("descriptor_refine_grad_clip", 1.0))
            opt.step()
            with torch.no_grad():
                delta = mu - mu0
                norm = delta.norm(dim=1, keepdim=True).clamp_min(_EPS)
                mu.copy_(mu0 + delta * (max_mean_shift / norm).clamp(max=1.0))
                log_eig.copy_(torch.max(torch.min(log_eig, eig0.log() + max_logvar_shift), eig0.log() - max_logvar_shift))
                log_res.copy_(torch.max(torch.min(log_res, res0.log() + max_logvar_shift), res0.log() - max_logvar_shift))
            stats["loss"] += float(loss.detach().cpu().item())
            stats["ce_new"] += float(ce_new.detach().cpu().item())
            stats["ce_replay"] += float(ce_replay.detach().cpu().item())
            stats["margin"] += float(margin.detach().cpu().item())
            stats["invasion"] += float(invasion.detach().cpu().item())
            stats["fit"] += float(fit.detach().cpu().item())
            stats["collision"] += float(collision.detach().cpu().item())
            stats["trust"] += float(trust.detach().cpu().item())
            stats["replay_acc"] += float((energy_old.argmin(dim=1) == y_old_local).float().mean().cpu().item() * 100.0)
            stats["old_to_new"] += float((energy_old.index_select(1, new_local_ids).min(dim=1).values <= true_old).float().mean().cpu().item() * 100.0)
            stats["new_to_old"] += float((energy_new.index_select(1, old_local_ids).min(dim=1).values <= true_new).float().mean().cpu().item() * 100.0)
            stats["steps"] += 1.0

        with torch.no_grad():
            final_desc = dict(init_desc)
            final_desc.update({
                "means": mu.detach(), "bases": U0.detach(),
                "eigvals": log_eig.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4)).detach(),
                "res_vars": log_res.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4)).detach(),
            })
            final_bank = self._compose_bank_for_descriptor_params(base_bank, new_classes, final_desc["means"], U0, final_desc["eigvals"], final_desc["res_vars"])
            final_energy = self._geometry_energy_from_bank(z_new, final_bank, seen_classes)
            q_e, q_m = [], []
            for c in new_classes:
                mask = y_new == int(c)
                target = torch.full((int(mask.sum().item()),), seen_classes.index(c), device=self.device, dtype=torch.long)
                e = final_energy[mask]
                true = e.gather(1, target.view(-1, 1)).squeeze(1)
                rival_m = e.clone(); rival_m.scatter_(1, target.view(-1, 1), float("inf"))
                gap = rival_m.min(dim=1).values - true
                q_e.append(torch.quantile(true, torch.tensor([0.50, 0.75, 0.90, 0.95], device=self.device)))
                q_m.append(torch.quantile(gap, torch.tensor([0.05, 0.10], device=self.device)))
            final_desc["energy_quantiles"] = torch.stack(q_e)
            final_desc["margin_quantiles"] = torch.stack(q_m)
            self._assert_descriptor_block_valid(final_desc, context="refined SCTGR new descriptors")
        denom = max(stats["steps"], 1.0)
        for key in stats:
            if key != "steps":
                stats[key] /= denom
        stats["replay"] = last_replay_stats
        return {"desc": final_desc, "stats": stats}

    # ------------------------------------------------------------------
    # Trainability: frozen base space + optional bounded residual adapter
    # ------------------------------------------------------------------
    def _incremental_update_mode(self) -> str:
        mode = str(getattr(self.args, "incremental_update_mode", "spectral_coupled_geometry_replay")).lower().strip()
        aliases = {
            "": "spectral_coupled_geometry_replay",
            "none": "spectral_coupled_geometry_replay",
            "clean": "spectral_coupled_geometry_replay",
            "descriptor": "spectral_coupled_geometry_replay",
            "descriptor_only": "spectral_coupled_geometry_replay",
            "scbgr": "spectral_coupled_geometry_replay",
            "sctgr": "spectral_coupled_geometry_replay",
            "spectral_coupled": "spectral_coupled_geometry_replay",
            "spectral_coupled_geometry_replay": "spectral_coupled_geometry_replay",
        }
        forbidden = {"geometry_gated_adapter", "adapter", "gated_adapter", "g2rpa", "g2-rpa"}
        if mode in forbidden:
            raise RuntimeError("Feature adapters are forbidden. Use incremental_update_mode=spectral_coupled_geometry_replay.")
        mode = aliases.get(mode, mode)
        if mode != "spectral_coupled_geometry_replay":
            raise RuntimeError(f"Unsupported incremental_update_mode={mode!r}.")
        setattr(self.args, "incremental_update_mode", mode)
        return mode

    def _adapter_mode_enabled(self) -> bool:
        return False

    def configure_incremental_trainability(self, old_classes: Sequence[int], new_classes: Sequence[int]) -> List[nn.Parameter]:
        """Freeze the canonical model. Only temporary descriptor tensors are optimized."""
        if any(self._inc_cfg_bool(name, False) for name in (
            "use_sglat_transport", "allow_old_model_transport", "use_geometry_transport",
            "use_energy_calibrator", "use_adaptive_boundary",
        )):
            raise RuntimeError("Transport, calibration, and adaptive-boundary modules are forbidden in SCTGR-RGA.")
        self._incremental_update_mode()
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        for attr in (
            "use_incremental_adapter", "use_geometry_gated_adapter", "use_bicyc_geometry_cycle",
            "use_geometry_calibrator", "use_geometry_transport", "use_energy_calibrator", "use_adaptive_boundary",
        ):
            if hasattr(self.model, attr):
                setattr(self.model, attr, False)
        for name in (
            "freeze_backbone_except_allowed", "freeze_semantic_encoder", "freeze_classifier",
            "freeze_projection_head", "freeze_backbone_only", "freeze_energy_calibrator",
            "freeze_geometry_calibrator", "disable_incremental_adapter",
        ):
            fn = getattr(self.model, name, None)
            if callable(fn):
                try:
                    fn()
                except TypeError:
                    try: fn(allow_last_block=False)
                    except TypeError: pass
        bad = [name for name, p in self.model.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"No model parameters may be trainable in SCTGR-RGA: {bad[:20]}")
        print("[Incremental Trainability] frozen backbone/projection/classifier; temporary new descriptor residuals only")
        return []
    def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[nn.Parameter]:
        return self.configure_incremental_trainability(list(range(int(old_class_count))), [])

    def _set_incremental_trainable_params(self, old_class_count: int) -> List[nn.Parameter]:
        return self._set_clean_incremental_trainable_params(old_class_count)

    # ------------------------------------------------------------------


    @torch.no_grad()
    def validate_incremental_phase(
        self,
        loader,
        old_classes: Sequence[int],
        new_classes: Sequence[int],
        seen_classes: Sequence[int],
    ) -> Dict[str, Any]:
        self.model.eval()
        total_loss = total = correct = invalid = 0
        old_to_new = new_to_old = 0
        old_total = new_total = 0
        margin_viol = 0
        per_total = {int(c): 0 for c in seen_classes}
        per_correct = {int(c): 0 for c in seen_classes}
        pred_hist = {int(c): 0 for c in seen_classes}
        old_set, new_set = set(map(int, old_classes)), set(map(int, new_classes))
        old_idx = torch.as_tensor([seen_classes.index(c) for c in old_classes], device=self.device, dtype=torch.long)
        new_idx = torch.as_tensor([seen_classes.index(c) for c in new_classes], device=self.device, dtype=torch.long)
        gaps: List[torch.Tensor] = []
        for batch in loader:
            x, y, spectra, _ = self._unpack_batch(batch)
            x = x.float().to(self.device, non_blocking=True)
            y = y.long().to(self.device, non_blocking=True).view(-1)
            self.assert_global_labels_in_set(y, seen_classes, "cumulative validation batch")
            z = self.extract_incremental_features(x, spectra)["features"]
            logits = self.compute_seen_logits(z, seen_classes, mode="geometry", old_classes=old_classes, new_classes=new_classes)["logits"]  # type: ignore[index]
            y_local = self.global_to_seen_local(y, seen_classes)
            loss = self._stable_ce(logits, y_local)
            pred_local = logits.argmax(dim=1)
            pred_global = self.seen_local_to_global(pred_local, seen_classes)
            true = logits.gather(1, y_local.view(-1, 1)).squeeze(1)
            rival_m = logits.clone(); rival_m.scatter_(1, y_local.view(-1, 1), float("-inf"))
            gap = true - rival_m.max(dim=1).values
            gaps.append(gap.detach())
            margin_viol += int((gap < self._inc_cfg_float("validation_logit_margin", 0.0)).sum().item())
            total_loss += float(loss.cpu().item()) * int(y.numel())
            total += int(y.numel()); correct += int((pred_global == y).sum().item())
            for cls in seen_classes:
                mask = y == int(cls); n = int(mask.sum().item())
                if n:
                    per_total[int(cls)] += n
                    per_correct[int(cls)] += int((pred_global[mask] == int(cls)).sum().item())
            for pg, yt in zip(pred_global.detach().cpu().tolist(), y.detach().cpu().tolist()):
                if int(pg) in pred_hist: pred_hist[int(pg)] += 1
                else: invalid += 1
                if int(yt) in old_set:
                    old_total += 1
                    if int(pg) in new_set: old_to_new += 1
                elif int(yt) in new_set:
                    new_total += 1
                    if int(pg) in old_set: new_to_old += 1
        acc = 100.0 * correct / max(total, 1)
        old_correct = sum(per_correct[c] for c in old_classes)
        new_correct = sum(per_correct[c] for c in new_classes)
        old_acc = 100.0 * old_correct / max(sum(per_total[c] for c in old_classes), 1)
        new_acc = 100.0 * new_correct / max(sum(per_total[c] for c in new_classes), 1)
        hm = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
        per_class_acc = {int(c): 100.0 * per_correct[int(c)] / max(per_total[int(c)], 1) for c in seen_classes}
        all_gaps = torch.cat(gaps) if gaps else torch.zeros((1,), device=self.device)
        return {
            "loss": total_loss / max(total, 1), "acc": acc, "old_acc": old_acc, "new_acc": new_acc,
            "hm": hm, "aa": sum(per_class_acc.values()) / max(len(per_class_acc), 1),
            "per_class_accuracy": per_class_acc, "prediction_histogram": pred_hist,
            "invalid_prediction_rate": float(invalid) / float(max(total, 1)),
            "old_to_new_error_rate": 100.0 * old_to_new / max(old_total, 1),
            "new_to_old_error_rate": 100.0 * new_to_old / max(new_total, 1),
            "margin_violation_rate": 100.0 * margin_viol / max(total, 1),
            "mean_logit_margin": float(all_gaps.mean().cpu().item()),
            "min_logit_margin": float(all_gaps.min().cpu().item()),
            "old_new_logit_gap": 0.0,
        }

    def _compute_old_new_overlap_stats(self, old_snapshot: Dict[str, torch.Tensor], new_desc: Dict[str, torch.Tensor]) -> Dict[str, float]:
        risk = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
        return {
            "old_new_risk_max": float(risk["risk"].max().detach().cpu().item()) if risk["risk"].numel() else 0.0,
            "old_new_risk_mean": float(risk["risk"].mean().detach().cpu().item()) if risk["risk"].numel() else 0.0,
            "old_new_subspace_overlap_max": float(risk["subspace"].max().detach().cpu().item()) if risk["subspace"].numel() else 0.0,
            "old_new_center_distance_min": float(risk["dist"].min().detach().cpu().item()) if risk["dist"].numel() else 0.0,
            "old_new_ellipsoid_energy_min": float(risk["energy"].min().detach().cpu().item()) if risk["energy"].numel() else 0.0,
        }

    def select_best_incremental_checkpoint(self, val_stats: Dict[str, Any], drift_stats: Dict[str, float], overlap_stats: Dict[str, float]) -> float:
        """Validation-first checkpoint score for descriptor-only NECIL.

        The previous score subtracted the raw old/new logit gap.  Geometry-energy
        logits in this code can be thousands of units apart, so that term drowned
        the actual validation harmonic mean and selected worse descriptors.

        Correct policy:
        - primary: old/new harmonic mean on real validation data
        - secondary: keep both old and new non-collapsed
        - small penalties: frozen-row drift and old/new geometry risk
        - optional normalized logit-gap penalty only after tanh scaling
        """
        hm = float(val_stats.get("hm", 0.0))
        acc = float(val_stats.get("acc", val_stats.get("overall_accuracy", 0.0)))
        old_acc = float(val_stats.get("old_acc", 0.0))
        new_acc = float(val_stats.get("new_acc", 0.0))
        balance = min(old_acc, new_acc)
        drift = max(float(v) for v in drift_stats.values()) if drift_stats else 0.0
        overlap = float(overlap_stats.get("old_new_risk_max", 0.0))

        raw_gap = abs(float(val_stats.get("old_new_logit_gap", 0.0)))
        gap_scale = max(self._inc_cfg_float("ckpt_logit_gap_scale", 1000.0), 1e-6)
        gap_penalty = math.tanh(raw_gap / gap_scale)

        return (
            hm
            + self._inc_cfg_float("ckpt_balance_weight", 0.20) * balance
            + self._inc_cfg_float("ckpt_acc_weight", 0.05) * acc
            - self._inc_cfg_float("ckpt_logit_gap_weight", 0.0) * gap_penalty
            - self._inc_cfg_float("ckpt_geometry_drift_weight", 50.0) * drift
            - self._inc_cfg_float("ckpt_overlap_weight", 2.0) * overlap
        )

    # Compatibility with outer Trainer.
    def _select_score(self, val_stats: Dict[str, Any], phase: int) -> float:
        return float(val_stats.get("hm", val_stats.get("acc", 0.0)))

    def _capture_state(self):
        return copy.deepcopy(self.model.state_dict())

    def _restore_state(self, state):
        if state is not None:
            self.model.load_state_dict(state)

    def _save_phase_artifacts(self, phase: int, history: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
        save_dir = str(getattr(self, "save_dir", getattr(self.args, "save_dir", "./results")))
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f"phase_{int(phase)}_incremental_diagnostics.json")
        pt_path = os.path.join(save_dir, f"phase_{int(phase)}_incremental_handoff.pt")
        def _jsonable(v: Any):
            if torch.is_tensor(v):
                return v.detach().cpu().tolist()
            if isinstance(v, dict):
                return {str(k): _jsonable(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_jsonable(x) for x in v]
            if isinstance(v, (int, float, str, bool)) or v is None:
                return v
            return str(v)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable({"history": history, "diagnostics": diagnostics}), f, indent=2)
        torch.save({"history": history, "diagnostics": diagnostics}, pt_path)
        print(f"[Incremental Diagnostics] saved json={json_path} | pt={pt_path}")


    def load_base_handoff(self, phase: int) -> Dict[str, Any]:
        """Load optional base geometry recommendations without enabling forbidden modules."""
        if int(phase) <= 0:
            return {}
        save_dir = str(getattr(self, "save_dir", getattr(self.args, "save_dir", "./results")))
        candidates = [os.path.join(save_dir, "phase_0_base_handoff.pt"), os.path.join(save_dir, "phase_0_base_handoff.json")]
        handoff: Dict[str, Any] = {}
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                if path.endswith(".pt"):
                    obj = torch.load(path, map_location=self.device)
                else:
                    with open(path, "r", encoding="utf-8") as f: obj = json.load(f)
                handoff = obj if isinstance(obj, dict) else {}
                if handoff:
                    print(f"[BaseHandoff] loaded {path}")
                    break
            except Exception as exc:
                print(f"[BaseHandoff WARN] could not load {path}: {exc}")
        if not handoff:
            return {}
        margin = handoff.get("recommended_insertion_margin", handoff.get("recommended_margin", None))
        if isinstance(margin, (int, float)) and float(margin) > 0:
            setattr(self, "old_new_geometry_margin", float(min(float(margin), self._inc_cfg_float("max_adaptive_geometry_margin", 0.60))))
        replay = handoff.get("recommended_replay_per_class", None)
        if isinstance(replay, dict):
            vals = [int(v) for v in replay.values() if isinstance(v, (int, float))]
            if vals: setattr(self, "synthetic_replay_per_class", max(vals))
        self._base_handoff = handoff
        return handoff

    # ------------------------------------------------------------------
    # Main epoch/phase flow
    # ------------------------------------------------------------------
    def assert_incremental_contract(self, phase: int, old_classes: Sequence[int], new_classes: Sequence[int], seen_classes: Sequence[int]) -> None:
        if int(phase) <= 0:
            raise RuntimeError("phase must be > 0 for incremental training.")
        if seen_classes != list(old_classes) + list(new_classes):
            raise RuntimeError(f"seen_classes must equal old+new order. old={old_classes}, new={new_classes}, seen={seen_classes}")
        self.assert_geometry_exists(old_classes, context="incremental old GeometryBank")
        self.freeze_old_geometry(old_classes)
        if hasattr(self.model, "set_incremental_mode"):
            self.model.set_incremental_mode(phase=int(phase), old_class_count=len(old_classes))
        if hasattr(self, "_set_model_phase_and_old_count"):
            self._set_model_phase_and_old_count(int(phase), len(old_classes))

    def train_incremental_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4) -> Dict[str, Any]:
        phase = int(phase)
        old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
        self._active_seen_classes = list(seen_classes)
        print(f"==== Incremental Phase {phase} | Spectral-Coupled Tangent Geometry Replay ====")
        print(f"[Classes] old={old_classes} | new={new_classes} | seen={seen_classes}")
        self.dataset.start_phase(phase)
        self.assert_incremental_contract(phase, old_classes, new_classes, seen_classes)
        base_handoff = self.load_base_handoff(phase)
        if hasattr(self.model, "ensure_class_capacity"):
            self.model.ensure_class_capacity(max(seen_classes) + 1)
        train_loader = self.dataset.get_phase_dataloader(phase, split="train", batch_size=batch_size, shuffle=True)
        val_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)

        old_snapshot = self.snapshot_old_geometry(old_classes)
        z_new, y_new, s_new, b_new = self.collect_current_phase_features(train_loader, new_classes)
        raw_desc = self._estimate_geometry_from_features(z_new, y_new, new_classes, spectral_summary=s_new, band_summary=b_new)
        admitted_desc, admission_stats = self.admit_new_geometry(raw_desc, old_snapshot, new_classes)
        self.assert_old_geometry_unchanged(old_snapshot, "post_new_admission")
        self._commit_new_descriptors(admitted_desc, phase=phase, freeze=False)
        self.assert_geometry_exists(seen_classes, context="post provisional new-row insertion")
        self.assert_old_geometry_unchanged(old_snapshot, "post_provisional_new_commit")
        self.configure_incremental_trainability(old_classes, new_classes)

        history: Dict[str, Any] = {
            "val_acc": [], "val_old_acc": [], "val_new_acc": [], "val_hm": [], "val_aa": [],
            "old_to_new_error": [], "new_to_old_error": [], "margin_violation": [],
            "ce_new": [], "ce_replay": [], "desc_margin": [], "desc_loss": [], "old_replay_acc": [],
            "old_new_risk_max": [], "old_new_overlap_max": [], "desc_invasion": [], "desc_fit": [],
            "desc_collision": [], "desc_trust": [], "replay_acceptance": [], "directed_replay_count": [],
            "coupling_reliability": [], "checkpoint_score": [],
        }
        diagnostics: Dict[str, Any] = {
            "phase": phase, "old_classes": old_classes, "new_classes": new_classes, "seen_classes": seen_classes,
            "method": "spectral_coupled_tangent_geometry_replay", "base_handoff_loaded": bool(base_handoff),
            "base_handoff": base_handoff, "admission": admission_stats, "old_rows_immutable": True,
        }

        init_val = self.validate_incremental_phase(val_loader, old_classes, new_classes, seen_classes)
        init_drift = self.assert_old_geometry_unchanged(old_snapshot, f"phase{phase}_init")
        init_overlap = self._compute_old_new_overlap_stats(old_snapshot, admitted_desc)
        best_score = self.select_best_incremental_checkpoint(init_val, init_drift, init_overlap)
        best_state = self._capture_state()
        best_desc = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in admitted_desc.items()}
        print(f"[InitVal] Val={init_val['acc']:.2f}% | Old={init_val['old_acc']:.2f}% | New={init_val['new_acc']:.2f}% | HM={init_val['hm']:.2f}% | O→N={init_val['old_to_new_error_rate']:.2f}% | N→O={init_val['new_to_old_error_rate']:.2f}%")

        epochs = int(max(0, epochs))
        steps = self._inc_cfg_int("descriptor_refine_steps_per_epoch", self._inc_cfg_int("descriptor_refine_steps", 20))
        for epoch in range(epochs):
            bank = self._bank_dict()
            ids = torch.as_tensor(new_classes, device=self.device, dtype=torch.long)
            keys = (
                "means", "bases", "eigvals", "res_vars", "active_ranks", "sample_counts", "reliability",
                "feature_reliability", "band_importance", "band_reliability", "spectral_prototypes",
                "spectral_reliability", "spectral_bases", "spectral_eigvals", "spectral_res_vars",
                "spectral_active_ranks", "spectral_to_feature", "coupling_residual_vars",
                "coupling_reliability", "spectral_sam_limits", "spectral_d1_limits", "spectral_d2_limits",
                "energy_quantiles", "margin_quantiles",
            )
            current_desc: Dict[str, torch.Tensor] = {"class_ids": ids}
            for key in keys:
                value = bank.get(key, None)
                if torch.is_tensor(value) and value.dim() > 0 and value.size(0) > int(ids.max().item()):
                    current_desc[key] = value.index_select(0, ids).detach()
            refined = self._refine_new_descriptors_impl(
                z_new=z_new, y_new=y_new, seen_classes=seen_classes, old_classes=old_classes,
                new_classes=new_classes, old_snapshot=old_snapshot, init_desc=current_desc, steps=steps,
            )
            desc, ds = refined["desc"], refined["stats"]
            self._commit_new_descriptors(desc, phase=phase, freeze=False)
            drift = self.assert_old_geometry_unchanged(old_snapshot, f"phase{phase}_epoch{epoch+1}")
            val = self.validate_incremental_phase(val_loader, old_classes, new_classes, seen_classes)
            overlap = self._compute_old_new_overlap_stats(old_snapshot, desc)
            score = self.select_best_incremental_checkpoint(val, drift, overlap)
            if score > best_score:
                best_score = score; best_state = self._capture_state()
                best_desc = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in desc.items()}
            rp = ds.get("replay", {}) if isinstance(ds.get("replay", {}), dict) else {}
            history["val_acc"].append(float(val["acc"])); history["val_old_acc"].append(float(val["old_acc"])); history["val_new_acc"].append(float(val["new_acc"])); history["val_hm"].append(float(val["hm"])); history["val_aa"].append(float(val["aa"]))
            history["old_to_new_error"].append(float(val["old_to_new_error_rate"])); history["new_to_old_error"].append(float(val["new_to_old_error_rate"])); history["margin_violation"].append(float(val["margin_violation_rate"]))
            history["ce_new"].append(float(ds.get("ce_new", 0.0))); history["ce_replay"].append(float(ds.get("ce_replay", 0.0))); history["desc_margin"].append(float(ds.get("margin", 0.0))); history["desc_loss"].append(float(ds.get("loss", 0.0))); history["old_replay_acc"].append(float(ds.get("replay_acc", 0.0)))
            history["old_new_risk_max"].append(float(overlap["old_new_risk_max"])); history["old_new_overlap_max"].append(float(overlap["old_new_subspace_overlap_max"])); history["desc_invasion"].append(float(ds.get("invasion", 0.0))); history["desc_fit"].append(float(ds.get("fit", 0.0))); history["desc_collision"].append(float(ds.get("collision", 0.0))); history["desc_trust"].append(float(ds.get("trust", 0.0)))
            history["replay_acceptance"].append(float(rp.get("acceptance_rate", 0.0))); history["directed_replay_count"].append(float(rp.get("directed_count", 0.0))); history["coupling_reliability"].append(float(rp.get("coupling_reliability_mean", 0.0))); history["checkpoint_score"].append(float(score))
            print(
                f"[IncEpoch] P{phase} E{epoch+1:03d}/{epochs} | Loss={ds.get('loss',0.0):.4f} | "
                f"CEnew={ds.get('ce_new',0.0):.4f} | CEold={ds.get('ce_replay',0.0):.4f} | Replay={ds.get('replay_acc',0.0):.2f}% | "
                f"Accept={rp.get('acceptance_rate',0.0)*100.0:.1f}% | Directed={rp.get('directed_count',0)} | "
                f"Val={val['acc']:.2f}% | Old={val['old_acc']:.2f}% | New={val['new_acc']:.2f}% | HM={val['hm']:.2f}% | "
                f"O→N={val['old_to_new_error_rate']:.2f}% | N→O={val['new_to_old_error_rate']:.2f}% | Risk={overlap['old_new_risk_max']:.4f} | Score={score:.4f}"
            )

        self._restore_state(best_state)
        self._commit_new_descriptors(best_desc, phase=phase, freeze=True)
        self.assert_old_geometry_unchanged(old_snapshot, f"phase{phase}_best_restore")
        self.freeze_old_geometry(seen_classes)
        if hasattr(self.dataset, "finalize_phase"): self.dataset.finalize_phase(phase)
        if hasattr(self, "_set_model_phase_and_old_count"): self._set_model_phase_and_old_count(phase, len(seen_classes))
        final_val = self.validate_incremental_phase(val_loader, old_classes, new_classes, seen_classes)
        final_overlap = self._compute_old_new_overlap_stats(old_snapshot, best_desc)
        final_replay = self.sample_old_replay(old_snapshot, seen_classes, new_desc=best_desc)
        diagnostics.update({"final_val": final_val, "final_overlap": final_overlap, "final_replay": final_replay.get("stats", {}), "best_checkpoint_score": best_score})
        history["final_val"] = final_val
        if hasattr(self, "save_checkpoint"): self.save_checkpoint(phase, history)
        self._save_phase_artifacts(phase, history, diagnostics)
        print(f"[PhaseDone] P{phase} | Val={final_val['acc']:.2f}% | Old={final_val['old_acc']:.2f}% | New={final_val['new_acc']:.2f}% | HM={final_val['hm']:.2f}% | old_rows_unchanged=True")
        return history
















# from __future__ import annotations

# import copy
# import json
# import math
# import os
# from contextlib import nullcontext
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


# _EPS = 1e-8


# def _as_bool(value: Any) -> bool:
#     if isinstance(value, str):
#         return value.strip().lower() in {"1", "true", "yes", "y", "on"}
#     return bool(value)


# def _finite(x: torch.Tensor, name: str) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a tensor.")
#     if not torch.isfinite(x).all():
#         bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
#         raise RuntimeError(f"{name} contains {bad} NaN/Inf values.")
#     return x


# class IncrementalPhaseTrainer:
#     """Clean descriptor-only incremental trainer for NECIL-HSI.

#     Main-path contract
#     ------------------
#     * Dataset labels are global class ids.
#     * GeometryBank rows are global class ids.
#     * Classifier/logit columns are compact seen-class indices in ``seen_classes`` order.
#     * CE labels are always seen-local indices.
#     * Old GeometryBank rows are frozen and checked after every mutating operation.
#     * New rows may be inserted/refined; old rows may not move.

#     The default path is descriptor-only / SCBGR.  A bounded geometry-gated adapter
#     is left as an explicit ablation, but old-row affine/SGLAT transport is disabled
#     in the clean path because it mutates frozen memory.
#     """

#     # ------------------------------------------------------------------
#     # Basic config helpers
#     # ------------------------------------------------------------------
#     def _zero_like_ref(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if torch.is_tensor(ref):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     def _inc_cfg_float(self, name: str, default: float) -> float:
#         return float(getattr(self, name, getattr(self.args, name, default)))

#     def _inc_cfg_int(self, name: str, default: int) -> int:
#         return int(getattr(self, name, getattr(self.args, name, default)))

#     def _inc_cfg_bool(self, name: str, default: bool) -> bool:
#         return _as_bool(getattr(self, name, getattr(self.args, name, default)))

#     def _classifier_mode(self) -> str:
#         mode = str(getattr(self.args, "incremental_classifier_mode", "geometry")).lower().strip()
#         aliases = {
#             "geometry_only": "geometry",
#             "calibrated_geometry": "calibrated_geometry",
#             "topology_calibrated_geometry": "calibrated_geometry",
#             "anchor": "geometry",
#             "anchor_concept": "geometry",
#             "srgp": "geometry",
#         }
#         return aliases.get(mode, mode)

#     # ------------------------------------------------------------------
#     # Phase/class resolution and label mapping
#     # ------------------------------------------------------------------
#     def _ordered_unique(self, values: Iterable[int]) -> List[int]:
#         out: List[int] = []
#         seen = set()
#         for v in values:
#             iv = int(v)
#             if iv not in seen:
#                 out.append(iv)
#                 seen.add(iv)
#         return out

#     def resolve_phase_classes(self, phase: int) -> Tuple[List[int], List[int], List[int]]:
#         phase = int(phase)
#         if phase <= 0:
#             raise ValueError("Incremental phase must be > 0.")
#         if not hasattr(self.dataset, "phase_to_classes"):
#             raise AttributeError("dataset.phase_to_classes is required.")
#         new_classes = self._ordered_unique(int(c) for c in self.dataset.phase_to_classes[phase])
#         old_classes: List[int] = []
#         if hasattr(self.dataset, "get_classes_up_to_phase"):
#             old_classes = self._ordered_unique(int(c) for c in self.dataset.get_classes_up_to_phase(phase - 1))
#         else:
#             for p in range(phase):
#                 old_classes.extend(int(c) for c in self.dataset.phase_to_classes[p])
#             old_classes = self._ordered_unique(old_classes)
#         seen_classes = self._ordered_unique([*old_classes, *new_classes])
#         if not old_classes:
#             raise RuntimeError("Incremental phase has no old classes. Did phase 0 finalize correctly?")
#         if not new_classes:
#             raise RuntimeError(f"Phase {phase} has no new classes.")
#         if len(seen_classes) != len(old_classes) + len(new_classes):
#             overlap = sorted(set(old_classes).intersection(new_classes))
#             raise RuntimeError(f"Old/new class overlap in phase {phase}: {overlap}")
#         return old_classes, new_classes, seen_classes

#     # Backward-compatible name used by older trainer code.
#     def _seen_classes_for_phase(self, phase: int) -> List[int]:
#         return self.resolve_phase_classes(int(phase))[2] if int(phase) > 0 else self._ordered_unique(self.dataset.phase_to_classes[0])

#     def global_to_seen_local(self, labels_global: torch.Tensor, seen_classes: Sequence[int]) -> torch.Tensor:
#         labels_global = labels_global.long().view(-1)
#         mapping = {int(c): i for i, c in enumerate([int(x) for x in seen_classes])}
#         local = torch.full_like(labels_global, -1)
#         for global_id, local_id in mapping.items():
#             local[labels_global == int(global_id)] = int(local_id)
#         if (local < 0).any():
#             bad = labels_global[local < 0].detach().cpu().unique().tolist()
#             raise RuntimeError(f"Labels not in seen_classes. bad={bad}, seen={list(map(int, seen_classes))}")
#         return local

#     def seen_local_to_global(self, preds_local: torch.Tensor, seen_classes: Sequence[int]) -> torch.Tensor:
#         preds_local = preds_local.long().view(-1)
#         seen = torch.as_tensor([int(c) for c in seen_classes], device=preds_local.device, dtype=torch.long)
#         if preds_local.numel() == 0:
#             return preds_local
#         if int(preds_local.min().item()) < 0 or int(preds_local.max().item()) >= int(seen.numel()):
#             raise RuntimeError(
#                 f"Local predictions [{int(preds_local.min())},{int(preds_local.max())}] incompatible with {int(seen.numel())} seen classes."
#             )
#         return seen.index_select(0, preds_local)

#     def assert_valid_seen_targets(self, targets_local: torch.Tensor, num_seen: int, context: str = "CE") -> None:
#         targets_local = targets_local.long().view(-1)
#         if targets_local.numel() == 0:
#             raise RuntimeError(f"{context}: empty CE target tensor.")
#         if int(targets_local.min().item()) < 0 or int(targets_local.max().item()) >= int(num_seen):
#             raise RuntimeError(
#                 f"{context}: local targets [{int(targets_local.min())},{int(targets_local.max())}] outside [0,{int(num_seen)-1}]."
#             )

#     def assert_global_labels_in_set(self, labels_global: torch.Tensor, allowed_classes: Iterable[int], context: str) -> None:
#         labels_global = labels_global.long().view(-1)
#         allowed = torch.as_tensor([int(c) for c in allowed_classes], device=labels_global.device, dtype=torch.long)
#         if labels_global.numel() == 0:
#             raise RuntimeError(f"{context}: empty label tensor.")
#         if allowed.numel() == 0:
#             raise RuntimeError(f"{context}: empty allowed class set.")
#         if hasattr(torch, "isin"):
#             ok = torch.isin(labels_global, allowed).all()
#         else:
#             mask = torch.zeros_like(labels_global, dtype=torch.bool)
#             for c in allowed:
#                 mask |= labels_global == int(c)
#             ok = mask.all()
#         if not bool(ok.item()):
#             bad = labels_global[~torch.isin(labels_global, allowed)].detach().cpu().unique().tolist() if hasattr(torch, "isin") else labels_global.detach().cpu().unique().tolist()
#             raise RuntimeError(f"{context}: labels outside allowed set. bad={bad}, allowed={allowed.detach().cpu().tolist()}")

#     # Older name in the uploaded file.
#     def _assert_batch_labels_in_classes(self, y: torch.Tensor, class_ids: Iterable[int], context: str) -> None:
#         self.assert_global_labels_in_set(y, class_ids, context)

#     # ------------------------------------------------------------------
#     # Canonical feature extraction and classifier scoring
#     # ------------------------------------------------------------------
#     def _prepare_real_spectral_summary(self, x: torch.Tensor, spectra: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], bool]:
#         if not torch.is_tensor(spectra) or spectra.numel() == 0:
#             return None, False
#         s = spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
#         if s.dim() == 4:
#             s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#         elif s.dim() == 3:
#             # [B,S,L] metadata: use the center token/spectrum.
#             # Flattening would mix neighboring pixels into the spectral summary
#             # and corrupt spectral-shape/risk calculations.
#             if s.size(0) == x.size(0) and s.size(1) > 0 and s.size(2) > 1:
#                 s = s[:, :, s.size(-1) // 2]
#             else:
#                 s = s.reshape(s.size(0), -1)
#         elif s.dim() == 1:
#             if s.numel() % max(int(x.size(0)), 1) == 0:
#                 s = s.view(x.size(0), -1)
#             else:
#                 return None, False
#         elif s.dim() > 4:
#             s = s.flatten(1)
#         if s.size(0) != x.size(0):
#             return None, False
#         # Raw metadata from the dataloader can stay physical even when model
#         # input uses PCA. PCA channels themselves are never physical spectra.
#         physical = self._inc_cfg_bool(
#             "incremental_spectral_summary_is_physical",
#             self._inc_cfg_bool("raw_spectral_summary_is_physical", False),
#         )
#         input_channels = int(x.size(1)) if torch.is_tensor(x) and x.dim() >= 2 else 0
#         pca_active = int(getattr(self.args, "pca_components", 0) or 0) > 0 and not bool(getattr(self.args, "no_pca", False))
#         if pca_active and input_channels > 0 and int(s.size(1)) == input_channels and not self._inc_cfg_bool("allow_nonphysical_spectral_summary", False):
#             physical = False
#         return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), bool(physical)

#     def extract_incremental_features(self, x: torch.Tensor, spectra: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         x = x.float().to(self.device, non_blocking=True)
#         spectral_summary, spectral_is_physical = self._prepare_real_spectral_summary(x, spectra)
#         fn_names = ("forward_features", "extract_features", "extract_projected_features", "extract_geometry_features")
#         for name in fn_names:
#             fn = getattr(self.model, name, None)
#             if callable(fn):
#                 try:
#                     out = fn(x, spectral_summary=spectral_summary, spectral_summary_is_physical=spectral_is_physical)
#                 except TypeError:
#                     try:
#                         out = fn(x)
#                     except TypeError:
#                         continue
#                 if isinstance(out, dict):
#                     z = out.get("features", out.get("projected_features", out.get("z", None)))
#                     if torch.is_tensor(z):
#                         z = _finite(z.float(), "incremental features")
#                         if z.dim() != 2:
#                             raise RuntimeError(f"Canonical incremental features must be [B,D], got {tuple(z.shape)}")
#                         out = dict(out)
#                         out["features"] = z
#                         out["projected_features"] = z
#                         out["spectral_summary"] = spectral_summary if spectral_summary is not None else out.get("spectral_summary", None)
#                         out["spectral_summary_is_physical"] = bool(spectral_is_physical)
#                         self._assert_feature_dim_matches_bank(z)
#                         return out
#                 elif torch.is_tensor(out):
#                     z = _finite(out.float(), "incremental features")
#                     if z.dim() != 2:
#                         raise RuntimeError(f"Canonical incremental features must be [B,D], got {tuple(z.shape)}")
#                     self._assert_feature_dim_matches_bank(z)
#                     return {"features": z, "projected_features": z, "spectral_summary": spectral_summary, "spectral_summary_is_physical": bool(spectral_is_physical)}
#         # Fallback through model forward.
#         try:
#             out = self.model(x, seen_classes=getattr(self, "_active_seen_classes", None), mode="geometry")
#         except TypeError:
#             out = self.model(x)
#         if not isinstance(out, dict) or not torch.is_tensor(out.get("features", None)):
#             raise RuntimeError("Model must expose forward_features/extract_features/extract_projected_features returning canonical z.")
#         z = _finite(out["features"].float(), "incremental features")
#         if z.dim() != 2:
#             raise RuntimeError(f"Canonical incremental features must be [B,D], got {tuple(z.shape)}")
#         out = dict(out)
#         out["features"] = z
#         out["projected_features"] = z
#         self._assert_feature_dim_matches_bank(z)
#         return out

#     def _assert_feature_dim_matches_bank(self, features: torch.Tensor) -> None:
#         gb = getattr(self.model, "geometry_bank", None)
#         dim = getattr(gb, "feature_dim", None)
#         if dim is not None and int(dim) > 0 and int(features.size(1)) != int(dim):
#             raise RuntimeError(f"Feature dim {int(features.size(1))} != GeometryBank feature_dim {int(dim)}")

#     def compute_seen_logits(
#         self,
#         features: torch.Tensor,
#         seen_classes: Sequence[int],
#         *,
#         mode: Optional[str] = None,
#         return_diagnostics: bool = False,
#         old_classes: Optional[Sequence[int]] = None,
#         new_classes: Optional[Sequence[int]] = None,
#         targets_local: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor | Dict[str, float]]:
#         features = _finite(features.float(), "features for seen logits")
#         if features.dim() != 2:
#             raise RuntimeError(f"features must be [B,D], got {tuple(features.shape)}")
#         seen = [int(c) for c in seen_classes]
#         if not seen:
#             raise RuntimeError("seen_classes is empty.")
#         self.assert_geometry_exists(seen, context="compute_seen_logits")
#         mode = str(mode or self._classifier_mode()).lower().strip()
#         mode = "geometry" if mode == "geometry_only" else mode
#         out: Any
#         if hasattr(self.model, "compute_logits_from_features"):
#             try:
#                 out = self.model.compute_logits_from_features(
#                     features,
#                     seen_classes=seen,
#                     geometry_bank=getattr(self.model, "geometry_bank", None),
#                     mode=mode,
#                     old_classes=list(old_classes or []),
#                     new_classes=list(new_classes or []),
#                     targets=targets_local,
#                     return_diagnostics=return_diagnostics,
#                 )
#             except TypeError:
#                 out = self.model.compute_logits_from_features(features, classifier_mode=mode)
#         elif hasattr(self.model, "classifier"):
#             out = self.model.classifier(
#                 features,
#                 seen_classes=seen,
#                 geometry_bank=getattr(self.model, "geometry_bank", None),
#                 mode=mode,
#                 old_classes=list(old_classes or []),
#                 new_classes=list(new_classes or []),
#                 targets=targets_local,
#                 return_diagnostics=return_diagnostics,
#             )
#         else:
#             raise AttributeError("model must expose compute_logits_from_features() or classifier().")

#         if isinstance(out, dict):
#             logits = out.get("logits", None)
#             result = dict(out)
#         else:
#             logits = out
#             result = {"logits": logits}
#         if not torch.is_tensor(logits):
#             raise RuntimeError("Classifier output does not contain tensor logits.")
#         logits = _finite(logits.float(), "seen logits")
#         # Clean classifier returns [B, len(seen)].  Legacy classifier may return full global width; convert once here.
#         if logits.dim() != 2:
#             raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
#         if logits.size(0) != features.size(0):
#             raise RuntimeError(f"logit batch {logits.size(0)} != feature batch {features.size(0)}")
#         if logits.size(1) == len(seen):
#             seen_logits = logits
#         elif logits.size(1) > max(seen):
#             idx = torch.as_tensor(seen, device=logits.device, dtype=torch.long)
#             seen_logits = logits.index_select(1, idx)
#         else:
#             raise RuntimeError(
#                 f"Classifier logits width={logits.size(1)} cannot represent seen_classes={seen}. "
#                 "Use repaired classifier with explicit seen_classes."
#             )
#         if seen_logits.size(1) != len(seen):
#             raise RuntimeError("Internal error: seen logits width mismatch.")
#         result["logits"] = _finite(seen_logits, "seen logits")
#         if targets_local is not None:
#             self.assert_valid_seen_targets(targets_local.to(seen_logits.device), len(seen), context="seen logits CE")
#         return result

#     def _stable_ce_seen(self, logits_seen: torch.Tensor, labels_global: torch.Tensor, seen_classes: Sequence[int], context: str) -> torch.Tensor:
#         logits_seen = _finite(logits_seen.float(), f"{context} logits")
#         if logits_seen.dim() != 2 or logits_seen.size(1) != len(seen_classes):
#             raise RuntimeError(f"{context}: logits must be [B,{len(seen_classes)}], got {tuple(logits_seen.shape)}")
#         labels_local = self.global_to_seen_local(labels_global.to(logits_seen.device), seen_classes)
#         self.assert_valid_seen_targets(labels_local, len(seen_classes), context=context)
#         if labels_local.numel() != logits_seen.size(0):
#             raise RuntimeError(f"{context}: label/logit batch mismatch {labels_local.numel()} vs {logits_seen.size(0)}")
#         clip = self._inc_cfg_float("ce_logit_clip", 50.0)
#         return F.cross_entropy(
#             logits_seen.clamp(-clip, clip),
#             labels_local,
#             label_smoothing=self._inc_cfg_float("label_smoothing", 0.0),
#         )

#     # Backward-compatible old helper: expects local labels for seen-width logits.
#     def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         labels = labels.to(device=logits.device).long().view(-1)
#         self.assert_valid_seen_targets(labels, logits.size(1), context="legacy CE")
#         clip = self._inc_cfg_float("ce_logit_clip", 50.0)
#         return F.cross_entropy(logits.clamp(-clip, clip), labels, label_smoothing=self._inc_cfg_float("label_smoothing", 0.0))

#     # ------------------------------------------------------------------
#     # GeometryBank access and immutability checks
#     # ------------------------------------------------------------------
#     def _bank_object(self):
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             raise AttributeError("model.geometry_bank is required.")
#         return gb

#     def _bank_dict(self) -> Dict[str, torch.Tensor]:
#         gb = self._bank_object()
#         if hasattr(gb, "get_seen_class_bank"):
#             # Use all currently allocated rows when possible.
#             try:
#                 counts = getattr(gb, "sample_counts", None)
#                 if torch.is_tensor(counts):
#                     ids = list(range(int(counts.numel())))
#                     return gb.get_seen_class_bank(ids)
#             except Exception:
#                 pass
#         if hasattr(gb, "get_subspace_bank"):
#             bank = gb.get_subspace_bank()
#         elif hasattr(self.model, "get_subspace_bank"):
#             bank = self.model.get_subspace_bank()
#         else:
#             bank = {name: getattr(gb, name) for name in ("means", "bases", "eigvals", "res_vars", "sample_counts", "active_ranks") if torch.is_tensor(getattr(gb, name, None))}
#         if not isinstance(bank, dict):
#             raise RuntimeError("GeometryBank export must be a dict.")
#         return self._canonical_bank_dict(bank)

#     def _canonical_bank_dict(self, bank: Dict[str, Any]) -> Dict[str, torch.Tensor]:
#         out: Dict[str, torch.Tensor] = {}
#         aliases = {
#             "means": ("means", "mu"),
#             "bases": ("bases", "basis", "U"),
#             "eigvals": ("eigvals", "eigenvalues", "lambdas"),
#             "res_vars": ("res_vars", "resvars", "residual_vars", "sigma_c2"),
#             "sample_counts": ("sample_counts", "counts", "n"),
#             "active_ranks": ("active_ranks", "ranks"),
#             "reliability": ("reliability", "feature_reliability"),
#             "spectral_prototypes": ("spectral_prototypes", "spectral_prototype", "spectral"),
#             "band_importance": ("band_importance", "band_importances", "band"),
#         }
#         for key, names in aliases.items():
#             for name in names:
#                 value = bank.get(name, None)
#                 if torch.is_tensor(value):
#                     out[key] = value.to(self.device)
#                     break
#         if "eigvals" not in out and torch.is_tensor(bank.get("variances", None)):
#             out["eigvals"] = bank["variances"].to(self.device)[:, :-1]
#             out["res_vars"] = bank["variances"].to(self.device)[:, -1]
#         if "res_vars" not in out and torch.is_tensor(bank.get("variances", None)):
#             out["res_vars"] = bank["variances"].to(self.device)[:, -1]
#         required = ("means", "bases", "eigvals", "res_vars", "sample_counts")
#         missing = [k for k in required if k not in out]
#         if missing:
#             raise RuntimeError(f"GeometryBank missing required tensors: {missing}")
#         if "active_ranks" not in out:
#             out["active_ranks"] = torch.full((out["means"].size(0),), out["bases"].size(-1), device=self.device, dtype=torch.long)
#         return out

#     def assert_geometry_exists(self, class_ids: Iterable[int], context: str) -> None:
#         ids = [int(c) for c in class_ids]
#         if not ids:
#             raise RuntimeError(f"{context}: empty class id list.")
#         gb = self._bank_object()
#         if hasattr(gb, "assert_bank_valid"):
#             try:
#                 gb.assert_bank_valid(seen_classes=ids)
#                 return
#             except TypeError:
#                 gb.assert_bank_valid()
#         bank = self._bank_dict()
#         counts = bank["sample_counts"].flatten()
#         max_id = max(ids)
#         if max_id >= counts.numel():
#             raise RuntimeError(f"{context}: bank has {counts.numel()} rows but needs class {max_id}")
#         bad = [c for c in ids if float(counts[c].detach().cpu().item()) <= 0]
#         if bad:
#             raise RuntimeError(f"{context}: missing GeometryBank rows for classes {bad}")

#     def freeze_old_geometry(self, old_classes: Sequence[int]) -> None:
#         gb = self._bank_object()
#         old = [int(c) for c in old_classes]
#         if hasattr(gb, "freeze_classes"):
#             gb.freeze_classes(old)
#         elif hasattr(gb, "freeze_classes_up_to") and old:
#             # Safe only for sequential old classes; otherwise fall back to mask if present.
#             if old == list(range(max(old) + 1)):
#                 gb.freeze_classes_up_to(max(old) + 1)
#             elif hasattr(gb, "frozen_mask"):
#                 gb.frozen_mask[torch.as_tensor(old, device=gb.frozen_mask.device)] = True
#         elif hasattr(gb, "frozen_class_mask"):
#             gb.frozen_class_mask[torch.as_tensor(old, device=gb.frozen_class_mask.device)] = True
#         if hasattr(gb, "assert_bank_valid"):
#             try:
#                 gb.assert_bank_valid(seen_classes=old)
#             except TypeError:
#                 gb.assert_bank_valid()

#     def snapshot_old_geometry(self, old_classes: Sequence[int]) -> Dict[str, torch.Tensor]:
#         old = [int(c) for c in old_classes]
#         self.assert_geometry_exists(old, context="snapshot_old_geometry")
#         bank = self._bank_dict()
#         ids = torch.as_tensor(old, device=self.device, dtype=torch.long)
#         snap = {
#             "class_ids": ids.detach().clone(),
#             "means": bank["means"].index_select(0, ids).detach().clone(),
#             "bases": bank["bases"].index_select(0, ids).detach().clone(),
#             "eigvals": bank["eigvals"].index_select(0, ids).detach().clone(),
#             "res_vars": bank["res_vars"].index_select(0, ids).detach().clone(),
#             "active_ranks": bank["active_ranks"].index_select(0, ids).detach().clone(),
#             "sample_counts": bank["sample_counts"].flatten().index_select(0, ids).detach().clone(),
#         }
#         for k in ("reliability", "spectral_prototypes", "band_importance"):
#             v = bank.get(k, None)
#             if torch.is_tensor(v) and v.size(0) > int(ids.max().item()):
#                 snap[k] = v.index_select(0, ids).detach().clone()
#         if bool((snap["sample_counts"] <= 0).any().item()):
#             bad = ids[snap["sample_counts"] <= 0].detach().cpu().tolist()
#             raise RuntimeError(f"Old GeometryBank has invalid old rows: {bad}")
#         return snap

#     # Compatibility name used by uploaded code.
#     def _snapshot_old_bank_clean(self, old_class_count: int) -> Dict[str, torch.Tensor]:
#         return self.snapshot_old_geometry(list(range(int(old_class_count))))

#     def assert_old_geometry_unchanged(self, snapshot: Dict[str, torch.Tensor], context: str, atol: float = 1e-6) -> Dict[str, float]:
#         ids = snapshot["class_ids"].to(self.device).long()
#         bank = self._bank_dict()
#         drift: Dict[str, float] = {}
#         for key, bank_key in (("means", "means"), ("bases", "bases"), ("eigvals", "eigvals"), ("res_vars", "res_vars")):
#             cur = bank[bank_key].index_select(0, ids).detach()
#             ref = snapshot[key].to(cur.device, cur.dtype)
#             if cur.shape != ref.shape:
#                 raise RuntimeError(f"{context}: old {key} shape changed {tuple(ref.shape)} -> {tuple(cur.shape)}")
#             delta = (cur - ref).abs().max()
#             drift[f"old_{key}_max_abs_drift"] = float(delta.detach().cpu().item())
#             if float(delta.detach().cpu().item()) > float(atol):
#                 raise RuntimeError(f"{context}: frozen old geometry changed for {key}. max_abs={float(delta):.6g}")
#         return drift

#     # ------------------------------------------------------------------
#     # New geometry construction and safe admission
#     # ------------------------------------------------------------------
#     def _unpack_batch(self, batch):
#         if hasattr(self, "_unpack_hsi_batch"):
#             return self._unpack_hsi_batch(batch)
#         if isinstance(batch, (list, tuple)):
#             if len(batch) == 2:
#                 x, y = batch
#                 return x, y, None, None
#             if len(batch) == 3:
#                 x, y, spectra = batch
#                 return x, y, spectra, None
#             return batch[0], batch[1], batch[2] if len(batch) > 2 else None, batch[3] if len(batch) > 3 else None
#         raise RuntimeError("Cannot unpack HSI batch.")

#     @torch.no_grad()
#     def collect_current_phase_features(self, loader, new_classes: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
#         self.model.eval()
#         feats: List[torch.Tensor] = []
#         labs: List[torch.Tensor] = []
#         spectra_rows: List[torch.Tensor] = []
#         bands: List[torch.Tensor] = []
#         for batch in loader:
#             x, y, spectra, _ = self._unpack_batch(batch)
#             x = x.float().to(self.device, non_blocking=True)
#             y = y.long().to(self.device, non_blocking=True).view(-1)
#             self.assert_global_labels_in_set(y, new_classes, "current incremental train loader")
#             out = self.extract_incremental_features(x, spectra)
#             z = out["features"].detach()
#             feats.append(z)
#             labs.append(y.detach())
#             s = out.get("spectral_summary", None)
#             if torch.is_tensor(s) and s.size(0) == z.size(0):
#                 spectra_rows.append(s.detach().float())
#             b = out.get("band_summary", out.get("band_importance", None))
#             if torch.is_tensor(b) and b.size(0) == z.size(0):
#                 bands.append(b.detach().float())
#         if not feats:
#             raise RuntimeError("Current phase train loader produced no features.")
#         z_all = torch.cat(feats, dim=0)
#         y_all = torch.cat(labs, dim=0)
#         s_all = torch.cat(spectra_rows, dim=0) if spectra_rows else None
#         b_all = torch.cat(bands, dim=0) if bands else None
#         return z_all, y_all, s_all, b_all

#     def _estimate_geometry_from_features(
#         self,
#         features: torch.Tensor,
#         labels_global: torch.Tensor,
#         class_ids: Sequence[int],
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_summary: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         features = _finite(features.float(), "new descriptor features")
#         labels_global = labels_global.long().view(-1).to(features.device)
#         D = int(features.size(1))
#         rank = int(getattr(getattr(self.model, "geometry_bank", None), "rank", getattr(self.args, "subspace_rank", 5)))
#         rank = max(1, min(rank, D))
#         means: List[torch.Tensor] = []
#         bases: List[torch.Tensor] = []
#         eigvals: List[torch.Tensor] = []
#         res_vars: List[torch.Tensor] = []
#         active: List[int] = []
#         counts: List[int] = []
#         reliability: List[torch.Tensor] = []
#         spectral_proto: List[torch.Tensor] = []
#         band_proto: List[torch.Tensor] = []
#         var_floor = self._inc_cfg_float("geom_var_floor", 1e-4)
#         for cls in [int(c) for c in class_ids]:
#             mask = labels_global == int(cls)
#             if not bool(mask.any().item()):
#                 raise RuntimeError(f"Cannot estimate new geometry for class {cls}: no samples.")
#             zc = features[mask]
#             n = int(zc.size(0))
#             mu = zc.mean(dim=0)
#             xc = zc - mu
#             denom = float(max(n - 1, 1))
#             if n >= 2:
#                 try:
#                     U, S, _ = torch.linalg.svd(xc / math.sqrt(denom), full_matrices=False)
#                     # For data matrix [N,D], right singular vectors are Vh.T. Recompute from covariance for clarity.
#                     cov = xc.t().matmul(xc) / denom
#                     evals, evecs = torch.linalg.eigh(cov)
#                     order = torch.argsort(evals, descending=True)
#                     evals = evals.index_select(0, order).clamp_min(var_floor)
#                     evecs = evecs.index_select(1, order)
#                     r_eff = min(rank, int((evals > var_floor * 1.01).sum().detach().cpu().item()), D)
#                     r_eff = max(1, r_eff)
#                     Uc = evecs[:, :rank]
#                     if Uc.size(1) < rank:
#                         pad = torch.eye(D, device=features.device, dtype=features.dtype)[:, : rank - Uc.size(1)]
#                         Uc = torch.cat([Uc, pad], dim=1)
#                     q, _ = torch.linalg.qr(Uc, mode="reduced")
#                     Uc = q[:, :rank]
#                     lam = evals[:rank]
#                     if lam.numel() < rank:
#                         lam = torch.cat([lam, lam.new_full((rank - lam.numel(),), var_floor)])
#                     total_var = torch.diag(cov).sum().clamp_min(var_floor)
#                     res = ((total_var - lam[:r_eff].sum()).clamp_min(var_floor) / float(max(D - r_eff, 1))).clamp_min(var_floor)
#                 except RuntimeError:
#                     Uc = torch.eye(D, device=features.device, dtype=features.dtype)[:, :rank]
#                     lam = torch.full((rank,), var_floor, device=features.device, dtype=features.dtype)
#                     res = torch.tensor(var_floor, device=features.device, dtype=features.dtype)
#                     r_eff = 1
#             else:
#                 Uc = torch.eye(D, device=features.device, dtype=features.dtype)[:, :rank]
#                 lam = torch.full((rank,), var_floor, device=features.device, dtype=features.dtype)
#                 res = torch.tensor(var_floor, device=features.device, dtype=features.dtype)
#                 r_eff = 1
#             means.append(mu)
#             bases.append(Uc)
#             eigvals.append(lam.clamp_min(var_floor))
#             res_vars.append(res.clamp_min(var_floor).view(()))
#             active.append(int(r_eff))
#             counts.append(n)
#             rel = torch.tensor(min(1.0, math.log1p(n) / math.log1p(64.0)), device=features.device, dtype=features.dtype)
#             reliability.append(rel)
#             if torch.is_tensor(spectral_summary) and spectral_summary.size(0) == labels_global.numel():
#                 spectral_proto.append(spectral_summary.to(features.device).float()[mask].mean(dim=0))
#             if torch.is_tensor(band_summary) and band_summary.size(0) == labels_global.numel():
#                 b = band_summary.to(features.device).float()[mask].mean(dim=0)
#                 b = b.clamp_min(0.0)
#                 b = b / b.sum().clamp_min(_EPS)
#                 band_proto.append(b)
#         result: Dict[str, torch.Tensor] = {
#             "class_ids": torch.as_tensor([int(c) for c in class_ids], device=features.device, dtype=torch.long),
#             "means": torch.stack(means, dim=0),
#             "bases": torch.stack(bases, dim=0),
#             "eigvals": torch.stack(eigvals, dim=0),
#             "res_vars": torch.stack(res_vars, dim=0),
#             "active_ranks": torch.as_tensor(active, device=features.device, dtype=torch.long),
#             "sample_counts": torch.as_tensor(counts, device=features.device, dtype=torch.float32),
#             "reliability": torch.stack(reliability, dim=0),
#         }
#         if spectral_proto:
#             result["spectral_prototypes"] = torch.stack(spectral_proto, dim=0)
#         if band_proto:
#             result["band_importance"] = torch.stack(band_proto, dim=0)
#         self._assert_descriptor_block_valid(result, context="estimated new geometry")
#         return result

#     def _assert_descriptor_block_valid(self, desc: Dict[str, torch.Tensor], context: str) -> None:
#         means = desc["means"]
#         bases = desc["bases"]
#         eig = desc["eigvals"]
#         res = desc["res_vars"]
#         if means.dim() != 2:
#             raise RuntimeError(f"{context}: means must be [K,D], got {tuple(means.shape)}")
#         if bases.dim() != 3 or bases.size(0) != means.size(0) or bases.size(1) != means.size(1):
#             raise RuntimeError(f"{context}: bases must be [K,D,R], got {tuple(bases.shape)} with means {tuple(means.shape)}")
#         if eig.shape != (means.size(0), bases.size(2)):
#             raise RuntimeError(f"{context}: eigvals shape {tuple(eig.shape)} incompatible with bases {tuple(bases.shape)}")
#         if res.numel() != means.size(0):
#             raise RuntimeError(f"{context}: res_vars length mismatch")
#         for name, t in (("means", means), ("bases", bases), ("eigvals", eig), ("res_vars", res)):
#             _finite(t, f"{context}.{name}")
#         if bool((eig < 0).any().item()) or bool((res < 0).any().item()):
#             raise RuntimeError(f"{context}: negative variances/eigenvalues.")
#         gram = torch.matmul(bases.transpose(1, 2), bases)
#         eye = torch.eye(bases.size(2), device=bases.device, dtype=bases.dtype).unsqueeze(0)
#         err = (gram - eye).abs().max()
#         if float(err.detach().cpu().item()) > 5e-3:
#             raise RuntimeError(f"{context}: bases are not orthonormal; max gram error={float(err):.6f}")

#     def _commit_new_descriptors(self, desc: Dict[str, torch.Tensor], phase: int) -> None:
#         gb = self._bank_object()
#         ids = desc["class_ids"].detach().cpu().tolist()
#         for row, cls in enumerate(ids):
#             kwargs = dict(
#                 class_id=int(cls),
#                 mean=desc["means"][row].detach(),
#                 basis=desc["bases"][row].detach(),
#                 eigvals=desc["eigvals"][row].detach(),
#                 res_var=desc["res_vars"][row].detach(),
#                 sample_count=desc["sample_counts"][row].detach(),
#                 active_rank=desc["active_ranks"][row].detach(),
#                 reliability=desc.get("reliability", torch.ones_like(desc["sample_counts"]))[row].detach(),
#                 spectral_prototype=desc.get("spectral_prototypes", None)[row].detach() if torch.is_tensor(desc.get("spectral_prototypes", None)) else None,
#                 band_importance=desc.get("band_importance", None)[row].detach() if torch.is_tensor(desc.get("band_importance", None)) else None,
#                 phase_created=int(phase),
#             )
#             if hasattr(gb, "add_or_update_class_geometry"):
#                 gb.add_or_update_class_geometry(**kwargs)
#             elif hasattr(gb, "update_class_geometry"):
#                 gb.update_class_geometry(allow_frozen_update=False, **{k: v for k, v in kwargs.items() if k != "phase_created"})
#             elif hasattr(gb, "update_class"):
#                 kwargs["cls_id"] = kwargs.pop("class_id")
#                 gb.update_class(**{k: v for k, v in kwargs.items() if k != "phase_created"})
#             else:
#                 raise AttributeError("GeometryBank must expose add_or_update_class_geometry/update_class_geometry/update_class.")
#         if hasattr(gb, "assert_bank_valid"):
#             try:
#                 gb.assert_bank_valid(seen_classes=ids)
#             except TypeError:
#                 gb.assert_bank_valid()

#     def _risk_matrix_from_descriptors(
#         self,
#         old_snapshot: Dict[str, torch.Tensor],
#         new_desc: Dict[str, torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         old_mu = old_snapshot["means"].to(self.device).float()
#         old_U = old_snapshot["bases"].to(self.device).float()
#         old_e = old_snapshot["eigvals"].to(self.device).float().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         old_rv = old_snapshot["res_vars"].to(self.device).float().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         new_mu = new_desc["means"].to(self.device).float()
#         new_U = new_desc["bases"].to(self.device).float()
#         # Center proximity
#         dist = torch.cdist(old_mu, new_mu, p=2)
#         center_margin = max(self._inc_cfg_float("risk_center_margin", 1.0), 1e-6)
#         center = torch.exp(-dist / center_margin).clamp(0.0, 1.0)
#         # Old ellipsoid invasion energy of new means.
#         diff = new_mu.unsqueeze(0) - old_mu.unsqueeze(1)  # [O,N,D]
#         proj = torch.einsum("ond,odr->onr", diff, old_U)
#         low = (proj.pow(2) / old_e.unsqueeze(1).clamp_min(_EPS)).sum(dim=-1)
#         rec = torch.einsum("onr,odr->ond", proj, old_U)
#         residual = ((diff - rec).pow(2).sum(dim=-1) / old_rv.view(-1, 1).clamp_min(_EPS))
#         energy = (low + residual) / float(max(old_mu.size(1), 1))
#         mahal_margin = max(self._inc_cfg_float("old_new_geometry_margin", 0.30), 1e-6)
#         mahal = F.relu(mahal_margin - energy) / mahal_margin
#         # Subspace overlap.
#         overlap_rows: List[torch.Tensor] = []
#         for i in range(old_U.size(0)):
#             vals = []
#             for j in range(new_U.size(0)):
#                 m = old_U[i].t().matmul(new_U[j])
#                 vals.append(m.pow(2).sum() / float(max(1, min(old_U.size(-1), new_U.size(-1)))))
#             overlap_rows.append(torch.stack(vals))
#         subspace = torch.stack(overlap_rows, dim=0).clamp(0.0, 1.0)
#         band = torch.zeros_like(subspace)
#         if torch.is_tensor(old_snapshot.get("band_importance", None)) and torch.is_tensor(new_desc.get("band_importance", None)):
#             ob = F.normalize(old_snapshot["band_importance"].to(self.device).float(), p=2, dim=1)
#             nb = F.normalize(new_desc["band_importance"].to(self.device).float(), p=2, dim=1)
#             if ob.size(1) == nb.size(1):
#                 band = ob.matmul(nb.t()).clamp(0.0, 1.0) * center
#         spectral = torch.zeros_like(subspace)
#         if torch.is_tensor(old_snapshot.get("spectral_prototypes", None)) and torch.is_tensor(new_desc.get("spectral_prototypes", None)):
#             os = old_snapshot["spectral_prototypes"].to(self.device).float()
#             ns = new_desc["spectral_prototypes"].to(self.device).float()
#             if os.dim() == 2 and ns.dim() == 2 and os.size(1) == ns.size(1):
#                 spectral = F.normalize(os, p=2, dim=1).matmul(F.normalize(ns, p=2, dim=1).t()).clamp(0.0, 1.0) * center
#         risk = (
#             self._inc_cfg_float("risk_center_weight", 0.50) * center
#             + self._inc_cfg_float("risk_mahal_weight", 1.00) * mahal
#             + self._inc_cfg_float("risk_subspace_weight", 1.00) * subspace
#             + self._inc_cfg_float("risk_band_weight", 0.15) * band
#             + self._inc_cfg_float("risk_spectral_shape_weight", 0.25) * spectral
#         )
#         return {"risk": risk, "center": center, "mahal": mahal, "subspace": subspace, "band": band, "spectral": spectral, "dist": dist, "energy": energy}

#     def admit_new_geometry(
#         self,
#         new_desc: Dict[str, torch.Tensor],
#         old_snapshot: Dict[str, torch.Tensor],
#         new_classes: Sequence[int],
#     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
#         """Apply new-row-only spectral-guided safe insertion.

#         This calls the repaired geometry_transport.safe_insert_new_geometry when available.
#         Fallback is a deterministic center push + basis projection.  In both cases old rows are never modified.
#         """
#         risk_before = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
#         corrected = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in new_desc.items()}
#         # Main SCBGR run must not secretly activate transport.
#         # Safe new-row transport is an explicit ablation: --use_geometry_transport true.
#         use_transport = self._inc_cfg_bool("use_geometry_transport", False) or self._inc_cfg_bool("use_new_row_transport", False)
#         if use_transport:
#             try:
#                 from models.geometry_transport import safe_insert_new_geometry  # type: ignore

#                 out = safe_insert_new_geometry(
#                     new_class_ids=[int(c) for c in new_classes],
#                     old_class_ids=old_snapshot["class_ids"].detach().cpu().tolist(),
#                     new_means=new_desc["means"],
#                     new_bases=new_desc["bases"],
#                     new_eigvals=new_desc["eigvals"],
#                     new_res_vars=new_desc["res_vars"],
#                     old_means=old_snapshot["means"],
#                     old_bases=old_snapshot["bases"],
#                     old_eigvals=old_snapshot["eigvals"],
#                     old_res_vars=old_snapshot["res_vars"],
#                     new_active_ranks=new_desc.get("active_ranks", None),
#                     old_active_ranks=old_snapshot.get("active_ranks", None),
#                     new_spectral=new_desc.get("spectral_prototypes", None),
#                     old_spectral=old_snapshot.get("spectral_prototypes", None),
#                     new_band=new_desc.get("band_importance", None),
#                     old_band=old_snapshot.get("band_importance", None),
#                     center_margin=self._inc_cfg_float("risk_center_margin", 1.0),
#                     ellipsoid_margin=self._inc_cfg_float("old_new_geometry_margin", 0.30),
#                     max_mean_shift=self._inc_cfg_float("descriptor_refine_max_mean_shift", 0.35),
#                 )
#                 if isinstance(out, dict) and torch.is_tensor(out.get("means", None)):
#                     corrected["means"] = out["means"].to(self.device)
#                     corrected["bases"] = out["bases"].to(self.device)
#                     corrected["eigvals"] = out["eigvals"].to(self.device)
#                     corrected["res_vars"] = out["res_vars"].to(self.device)
#                     if torch.is_tensor(out.get("active_ranks", None)):
#                         corrected["active_ranks"] = out["active_ranks"].to(self.device)
#             except Exception as exc:
#                 if bool(getattr(self, "debug", False)):
#                     print(f"[NewRowTransport WARN] using fallback safe insertion: {exc}")
#                 corrected = self._fallback_safe_new_row_correction(corrected, old_snapshot, risk_before)
#         else:
#             corrected = self._fallback_safe_new_row_correction(corrected, old_snapshot, risk_before)
#         self._assert_descriptor_block_valid(corrected, context="admitted new geometry")
#         risk_after = self._risk_matrix_from_descriptors(old_snapshot, corrected)
#         stats = {
#             "risk_before_max": float(risk_before["risk"].max().detach().cpu().item()) if risk_before["risk"].numel() else 0.0,
#             "risk_before_mean": float(risk_before["risk"].mean().detach().cpu().item()) if risk_before["risk"].numel() else 0.0,
#             "risk_after_max": float(risk_after["risk"].max().detach().cpu().item()) if risk_after["risk"].numel() else 0.0,
#             "risk_after_mean": float(risk_after["risk"].mean().detach().cpu().item()) if risk_after["risk"].numel() else 0.0,
#             "overlap_before_max": float(risk_before["subspace"].max().detach().cpu().item()) if risk_before["subspace"].numel() else 0.0,
#             "overlap_after_max": float(risk_after["subspace"].max().detach().cpu().item()) if risk_after["subspace"].numel() else 0.0,
#             "transport_active": float(1.0 if use_transport else 0.0),
#         }
#         if stats["risk_after_max"] > stats["risk_before_max"] + 1e-5:
#             print(f"[NewRowAdmission WARN] max risk increased {stats['risk_before_max']:.4f}->{stats['risk_after_max']:.4f}; keeping correction because identity clamp may still improve subspace/boundary.")
#         return corrected, stats

#     def _fallback_safe_new_row_correction(self, desc: Dict[str, torch.Tensor], old_snapshot: Dict[str, torch.Tensor], risk_parts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         out = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in desc.items()}
#         risk = risk_parts["risk"].to(self.device)
#         if risk.numel() == 0:
#             return out
#         old_mu = old_snapshot["means"].to(self.device).float()
#         old_U = old_snapshot["bases"].to(self.device).float()
#         means = out["means"].to(self.device).float()
#         bases = out["bases"].to(self.device).float()
#         eig = out["eigvals"].to(self.device).float()
#         res = out["res_vars"].to(self.device).float()
#         risk_thr = self._inc_cfg_float("descriptor_correction_risk_threshold", 0.35)
#         max_shift = self._inc_cfg_float("descriptor_refine_max_mean_shift", 0.35)
#         basis_strength = self._inc_cfg_float("descriptor_correction_basis_strength", 0.50)
#         var_shrink = self._inc_cfg_float("descriptor_correction_var_shrink", 0.10)
#         var_floor = self._inc_cfg_float("geom_var_floor", 1e-4)
#         for j in range(means.size(0)):
#             col = risk[:, j].clamp_min(0.0)
#             if float(col.max().detach().cpu().item()) < risk_thr:
#                 continue
#             w = col / col.sum().clamp_min(_EPS)
#             push = torch.zeros_like(means[j])
#             P = torch.zeros((bases.size(1), bases.size(1)), device=self.device, dtype=bases.dtype)
#             for i in range(old_mu.size(0)):
#                 direction = means[j] - old_mu[i]
#                 direction = direction / direction.norm().clamp_min(_EPS)
#                 push = push + w[i] * direction
#                 P = P + w[i] * old_U[i].matmul(old_U[i].t())
#             gate = float(min(1.0, max(0.0, (float(col.max().detach().cpu().item()) - risk_thr) / max(1e-6, 1.5 - risk_thr))))
#             if push.norm() > _EPS:
#                 delta = max_shift * gate * push / push.norm().clamp_min(_EPS)
#                 means[j] = means[j] + delta
#             Ucorr = bases[j] - basis_strength * gate * P.matmul(bases[j])
#             q, _ = torch.linalg.qr(Ucorr, mode="reduced")
#             q = q[:, : bases.size(2)]
#             # sign-stabilize
#             sign = torch.where((q * bases[j]).sum(dim=0, keepdim=True) < 0, -torch.ones(1, bases.size(2), device=q.device), torch.ones(1, bases.size(2), device=q.device))
#             bases[j] = q * sign
#             eig[j] = (eig[j] * (1.0 - var_shrink * gate)).clamp_min(var_floor)
#             res[j] = (res[j] * (1.0 - 0.5 * var_shrink * gate)).clamp_min(var_floor)
#         out["means"], out["bases"], out["eigvals"], out["res_vars"] = means, bases, eig, res
#         return out

#     # ------------------------------------------------------------------
#     # Replay from frozen old geometry
#     # ------------------------------------------------------------------
#     def _sample_from_snapshot(self, snapshot: Dict[str, torch.Tensor], samples_per_class: int, boundary: bool = False, new_desc: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
#         means = snapshot["means"].to(self.device).float()
#         bases = snapshot["bases"].to(self.device).float()
#         eig = snapshot["eigvals"].to(self.device).float().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         res = snapshot["res_vars"].to(self.device).float().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         ids = snapshot["class_ids"].to(self.device).long()
#         active = snapshot.get("active_ranks", torch.full((means.size(0),), bases.size(2), device=self.device, dtype=torch.long)).to(self.device).long()
#         per = max(1, int(samples_per_class))
#         xs: List[torch.Tensor] = []
#         ys: List[torch.Tensor] = []
#         boundary_count = 0
#         # Boundary replay is old-side only: sample near risky old means in directions toward new centers but keep old labels.
#         risk_pairs = None
#         if boundary and new_desc is not None:
#             risk = self._risk_matrix_from_descriptors(snapshot, new_desc)["risk"]
#             if risk.numel() > 0:
#                 thr = self._inc_cfg_float("boundary_replay_risk_threshold", 0.35)
#                 coords = (risk >= thr).nonzero(as_tuple=False)
#                 if coords.numel() == 0:
#                     # fallback to top dangerous pairs, but log fallback.
#                     k = min(self._inc_cfg_int("boundary_replay_max_pairs", 24), int(risk.numel()))
#                     _, flat_idx = torch.topk(risk.flatten(), k=k, largest=True)
#                     coords = torch.stack([flat_idx // risk.size(1), flat_idx % risk.size(1)], dim=1)
#                 risk_pairs = coords
#         for i in range(means.size(0)):
#             r = int(active[i].detach().cpu().item())
#             r = max(0, min(r, bases.size(2)))
#             eps = torch.randn(per, max(r, 1), device=self.device, dtype=means.dtype)
#             if r > 0:
#                 low = eps[:, :r].matmul((bases[i, :, :r] * eig[i, :r].sqrt().view(1, -1)).t())
#             else:
#                 low = torch.zeros(per, means.size(1), device=self.device, dtype=means.dtype)
#             residual = torch.randn(per, means.size(1), device=self.device, dtype=means.dtype) * res[i].sqrt() * self._inc_cfg_float("replay_residual_scale", 0.15)
#             z = means[i].view(1, -1) + self._inc_cfg_float("replay_parallel_scale", 0.35) * low + residual
#             xs.append(z)
#             ys.append(torch.full((per,), int(ids[i].detach().cpu().item()), device=self.device, dtype=torch.long))
#         if risk_pairs is not None and risk_pairs.numel() > 0 and new_desc is not None:
#             samples_per_pair = max(1, self._inc_cfg_int("boundary_replay_samples_per_pair", 4))
#             new_mu = new_desc["means"].to(self.device).float()
#             for oi, nj in risk_pairs.tolist():
#                 direction = new_mu[int(nj)] - means[int(oi)]
#                 if direction.norm() <= _EPS:
#                     continue
#                 direction = direction / direction.norm().clamp_min(_EPS)
#                 radius = eig[int(oi)].mean().sqrt().clamp_min(1e-3) * self._inc_cfg_float("boundary_replay_radius", 0.75)
#                 noise = torch.randn(samples_per_pair, means.size(1), device=self.device, dtype=means.dtype) * res[int(oi)].sqrt() * 0.05
#                 z = means[int(oi)].view(1, -1) + radius * direction.view(1, -1) + noise
#                 xs.append(z)
#                 ys.append(torch.full((samples_per_pair,), int(ids[int(oi)].detach().cpu().item()), device=self.device, dtype=torch.long))
#                 boundary_count += samples_per_pair
#         x_old = torch.cat(xs, dim=0)
#         y_old = torch.cat(ys, dim=0)
#         _finite(x_old, "old replay features")
#         stats = {
#             "replay_count": float(y_old.numel()),
#             "boundary_replay_count": float(boundary_count),
#             "boundary_replay_fallback": float(1.0 if boundary and boundary_count == 0 else 0.0),
#         }
#         return x_old, y_old, stats

#     def sample_old_replay(self, old_snapshot: Dict[str, torch.Tensor], seen_classes: Sequence[int], new_desc: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor | Dict[str, float]]:
#         old_classes = old_snapshot["class_ids"].detach().cpu().tolist()
#         samples_per_class = self._inc_cfg_int("synthetic_replay_per_class", self._inc_cfg_int("component_replay_per_class", 32))
#         use_boundary = self._inc_cfg_bool("use_boundary_geometry_replay", True)
#         x_old, y_old, stats = self._sample_from_snapshot(old_snapshot, samples_per_class, boundary=use_boundary, new_desc=new_desc)
#         if x_old.dim() != 2:
#             raise RuntimeError(f"Replay features must be [B,D], got {tuple(x_old.shape)}")
#         self.assert_global_labels_in_set(y_old, old_classes, "old replay labels")
#         bad_new = sorted(set(y_old.detach().cpu().tolist()).difference(set(old_classes)))
#         if bad_new:
#             raise RuntimeError(f"Replay labels include non-old classes: {bad_new}")
#         local = self.global_to_seen_local(y_old, seen_classes)
#         self.assert_valid_seen_targets(local, len(seen_classes), context="old replay local labels")
#         return {"features": x_old.detach(), "global_labels": y_old.detach(), "local_labels": local.detach(), "stats": stats}

#     # Compatibility old API.
#     def _sample_old_anchor_batch(self, old_bank_snapshot: Dict[str, torch.Tensor], old_class_count: int, new_class_ids: Optional[Iterable[int]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
#         # Convert old contiguous snapshot from old code to our canonical snapshot if needed.
#         snap = old_bank_snapshot
#         if "class_ids" not in snap:
#             snap = dict(snap)
#             snap["class_ids"] = torch.arange(int(old_class_count), device=self.device, dtype=torch.long)
#         if "eigvals" not in snap and torch.is_tensor(snap.get("variances", None)):
#             snap["eigvals"] = snap["variances"][:, :-1]
#             snap["res_vars"] = snap["variances"][:, -1]
#         replay = self.sample_old_replay(snap, list(range(int(old_class_count))) + [int(c) for c in (new_class_ids or [])])
#         return replay["features"], replay["global_labels"]  # type: ignore[return-value]

#     # ------------------------------------------------------------------
#     # Descriptor refinement: optimize only current new rows, old snapshot detached
#     # ------------------------------------------------------------------
#     def _compose_bank_for_descriptor_params(self, base_bank: Dict[str, torch.Tensor], new_classes: Sequence[int], mu: torch.Tensor, bases: torch.Tensor, eig: torch.Tensor, res: torch.Tensor) -> Dict[str, torch.Tensor]:
#         bank = {k: (v.detach().clone().to(self.device) if torch.is_tensor(v) else v) for k, v in base_bank.items()}
#         ids = torch.as_tensor([int(c) for c in new_classes], device=self.device, dtype=torch.long)
#         bank["means"][ids] = mu
#         bank["bases"][ids] = bases
#         bank["eigvals"][ids] = eig
#         bank["res_vars"][ids] = res
#         return bank  # type: ignore[return-value]

#     def _orthonormalize_bases(self, raw: torch.Tensor, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
#         outs = []
#         R = raw.size(-1)
#         for i in range(raw.size(0)):
#             q, _ = torch.linalg.qr(raw[i], mode="reduced")
#             q = q[:, :R]
#             if torch.is_tensor(reference) and reference.shape == raw.shape:
#                 sign = torch.where((q * reference[i].to(q.device, q.dtype)).sum(dim=0, keepdim=True) < 0, -torch.ones(1, R, device=q.device, dtype=q.dtype), torch.ones(1, R, device=q.device, dtype=q.dtype))
#                 q = q * sign
#             outs.append(q)
#         return torch.stack(outs, dim=0)

#     def _geometry_logits_from_bank(self, features: torch.Tensor, bank: Dict[str, torch.Tensor], seen_classes: Sequence[int]) -> torch.Tensor:
#         ids = torch.as_tensor([int(c) for c in seen_classes], device=features.device, dtype=torch.long)
#         mu = bank["means"].to(features.device).index_select(0, ids)
#         U = bank["bases"].to(features.device).index_select(0, ids)
#         eig = bank["eigvals"].to(features.device).index_select(0, ids).clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         rv = bank["res_vars"].to(features.device).flatten().index_select(0, ids).clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         diff = features.unsqueeze(1) - mu.unsqueeze(0)
#         proj = torch.einsum("bsd,sdr->bsr", diff, U)
#         low = (proj.pow(2) / eig.unsqueeze(0).clamp_min(_EPS)).sum(dim=-1)
#         rec = torch.einsum("bsr,sdr->bsd", proj, U)
#         residual = ((diff - rec).pow(2).sum(dim=-1) / rv.view(1, -1).clamp_min(_EPS))
#         energy = (low + residual) / float(max(features.size(1), 1))
#         if self._inc_cfg_bool("use_logdet_energy", True):
#             logdet = torch.log(eig).sum(dim=-1) + torch.log(rv).view(-1) * float(max(features.size(1) - U.size(-1), 0))
#             energy = energy + self._inc_cfg_float("logdet_energy_weight", 0.02) * logdet.view(1, -1)
#         energy = energy - energy.min(dim=1, keepdim=True).values.detach()
#         return -self._inc_cfg_float("loss_scale", 8.0) * energy

#     def _descriptor_margin_loss(self, old_snapshot: Dict[str, torch.Tensor], new_desc: Dict[str, torch.Tensor]) -> torch.Tensor:
#         risk = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
#         margin = self._inc_cfg_float("descriptor_overlap_target", 0.35)
#         sub_loss = F.relu(risk["subspace"] - margin).pow(2).mean() if risk["subspace"].numel() else self._zero_like_ref(new_desc["means"])
#         invasion = risk["mahal"].pow(2).mean() if risk["mahal"].numel() else self._zero_like_ref(new_desc["means"])
#         return self._inc_cfg_float("lambda_subspace", 0.20) * sub_loss + self._inc_cfg_float("lambda_insertion", 0.35) * invasion

#     def _old_new_boundary_preservation_loss(
#         self,
#         old_snapshot: Dict[str, torch.Tensor],
#         new_desc: Dict[str, torch.Tensor],
#         *,
#         return_parts: bool = False,
#     ) -> torch.Tensor | Dict[str, torch.Tensor]:
#         """Differentiable old/new boundary preservation for new descriptors only.

#         This is the method the trainer contract was correctly looking for. It is
#         not a dummy/stub: it penalizes new rows that enter old ellipsoids, reuse
#         old tangent directions, become too broad, or share high band signatures
#         with risky old rows. Old tensors are detached by construction.
#         """
#         ref = new_desc.get("means", None)
#         if not torch.is_tensor(ref) or ref.numel() == 0:
#             z = self._zero_like_ref(ref)
#             return {"total": z, "risk": z, "overlap": z, "volume": z, "band": z} if return_parts else z

#         risk = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
#         risk_mat = risk["risk"]
#         sub = risk["subspace"]
#         band = risk.get("band", torch.zeros_like(sub))

#         risk_target = self._inc_cfg_float("max_old_new_risk", 0.60)
#         overlap_target = self._inc_cfg_float("max_old_new_overlap", self._inc_cfg_float("descriptor_subspace_overlap_max", 0.35))
#         risk_loss = F.relu(risk_mat - risk_target).pow(2).mean() if risk_mat.numel() else self._zero_like_ref(ref)
#         overlap_loss = F.relu(sub - overlap_target).pow(2).mean() if sub.numel() else self._zero_like_ref(ref)
#         band_loss = F.relu(band - self._inc_cfg_float("pgr_band_overlap_max", 0.75)).pow(2).mean() if band.numel() else self._zero_like_ref(ref)

#         eig = new_desc.get("eigvals", None)
#         res = new_desc.get("res_vars", None)
#         volume_loss = self._zero_like_ref(ref)
#         if torch.is_tensor(eig) and torch.is_tensor(res) and eig.numel() > 0 and res.numel() > 0:
#             new_volume = torch.log(eig.clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))).mean(dim=1) + torch.log(res.clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))).view(-1)
#             old_e = old_snapshot.get("eigvals", None)
#             old_r = old_snapshot.get("res_vars", None)
#             if torch.is_tensor(old_e) and torch.is_tensor(old_r) and old_e.numel() > 0 and old_r.numel() > 0:
#                 old_volume = torch.log(old_e.detach().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))).mean(dim=1) + torch.log(old_r.detach().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))).view(-1)
#                 cap = old_volume.mean().detach() + old_volume.std(unbiased=False).detach()
#                 volume_loss = F.relu(new_volume - cap).pow(2).mean()

#         total = (
#             self._inc_cfg_float("boundary_preserve_center_weight", 0.50) * risk_loss
#             + self._inc_cfg_float("boundary_preserve_overlap_weight", 1.00) * overlap_loss
#             + self._inc_cfg_float("boundary_preserve_volume_weight", 0.25) * volume_loss
#             + self._inc_cfg_float("boundary_preserve_band_weight", 0.10) * band_loss
#         )
#         _finite(total, "old_new_boundary_preservation_loss")
#         if return_parts:
#             return {
#                 "total": total,
#                 "risk": risk_loss.detach(),
#                 "overlap": overlap_loss.detach(),
#                 "volume": volume_loss.detach(),
#                 "band": band_loss.detach(),
#                 "risk_max": risk_mat.detach().max() if risk_mat.numel() else self._zero_like_ref(ref).detach(),
#                 "overlap_max": sub.detach().max() if sub.numel() else self._zero_like_ref(ref).detach(),
#             }
#         return total

#     @torch.no_grad()
#     def _project_new_descriptor_params_out_of_old_tangent_space(
#         self,
#         desc: Dict[str, torch.Tensor],
#         old_snapshot: Dict[str, torch.Tensor],
#         *,
#         risk_parts: Optional[Dict[str, torch.Tensor]] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Hard projection step for new rows only.

#         It removes components of new bases that lie in risky old tangent spaces
#         and applies a small mean push/variance shrink. This never edits old rows.
#         """
#         parts = risk_parts if isinstance(risk_parts, dict) else self._risk_matrix_from_descriptors(old_snapshot, desc)
#         out = self._fallback_safe_new_row_correction(desc, old_snapshot, parts)
#         self._assert_descriptor_block_valid(out, context="boundary-projected new descriptors")
#         return out

#     def _adaptive_boundary_loss_from_current_bank(
#         self,
#         logits: Optional[torch.Tensor] = None,
#         labels_local: Optional[torch.Tensor] = None,
#         *,
#         old_class_count: int = 0,
#         seen_classes: Optional[Sequence[int]] = None,
#     ) -> torch.Tensor:
#         """Optional adaptive-boundary loss, gated by --use_adaptive_boundary.

#         The clean command uses --use_adaptive_boundary false, so this returns a
#         safe zero. When a repaired classifier exposes adaptive_boundary_loss, this
#         delegates to it without making adaptive boundary a hidden main-path module.
#         """
#         ref = logits if torch.is_tensor(logits) else None
#         if not self._inc_cfg_bool("use_adaptive_boundary", False):
#             return self._zero_like_ref(ref)
#         clf = getattr(self.model, "classifier", None)
#         if clf is None or not hasattr(clf, "adaptive_boundary_loss"):
#             return self._zero_like_ref(ref)
#         try:
#             loss = clf.adaptive_boundary_loss(
#                 logits=logits,
#                 labels=labels_local,
#                 old_class_count=int(old_class_count),
#                 seen_classes=list(seen_classes or []),
#             )
#         except TypeError:
#             try:
#                 loss = clf.adaptive_boundary_loss(logits, labels_local, int(old_class_count))
#             except TypeError:
#                 return self._zero_like_ref(ref)
#         if isinstance(loss, dict):
#             loss = loss.get("total", self._zero_like_ref(ref))
#         if not torch.is_tensor(loss):
#             return self._zero_like_ref(ref)
#         return _finite(loss, "adaptive_boundary_loss")

#     def _refine_new_descriptors_impl(
#         self,
#         *,
#         z_new: torch.Tensor,
#         y_new: torch.Tensor,
#         seen_classes: Sequence[int],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         old_snapshot: Dict[str, torch.Tensor],
#         init_desc: Dict[str, torch.Tensor],
#         steps: int,
#     ) -> Dict[str, Any]:
#         if steps <= 0 or not self._inc_cfg_bool("refine_new_descriptors", True):
#             return {"desc": init_desc, "stats": {"loss": 0.0, "ce_new": 0.0, "ce_replay": 0.0, "margin": 0.0, "steps": 0.0}}
#         base_bank = self._bank_dict()
#         ids = torch.as_tensor([int(c) for c in new_classes], device=self.device, dtype=torch.long)
#         mu0 = init_desc["means"].detach().clone()
#         U0 = init_desc["bases"].detach().clone()
#         eig0 = init_desc["eigvals"].detach().clone().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         res0 = init_desc["res_vars"].detach().clone().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#         mu = nn.Parameter(mu0.clone())
#         raw_U = nn.Parameter(U0.clone())
#         log_eig = nn.Parameter(eig0.log())
#         log_res = nn.Parameter(res0.log())
#         opt = optim.Adam([mu, raw_U, log_eig, log_res], lr=self._inc_cfg_float("descriptor_refine_lr", 1e-3), weight_decay=0.0)
#         max_mean_shift = self._inc_cfg_float("descriptor_refine_max_mean_shift", 0.35)
#         max_logvar_shift = self._inc_cfg_float("descriptor_refine_max_logvar_shift", 0.75)
#         replay_weight = self._inc_cfg_float("lambda_replay", self._inc_cfg_float("synthetic_replay_weight", 1.0))
#         new_weight = self._inc_cfg_float("lambda_new", 1.0)
#         margin_weight = self._inc_cfg_float("lambda_margin", 1.0)
#         stats = {"loss": 0.0, "ce_new": 0.0, "ce_replay": 0.0, "margin": 0.0, "replay_acc": 0.0, "steps": 0.0}
#         for _ in range(int(steps)):
#             opt.zero_grad(set_to_none=True)
#             U = self._orthonormalize_bases(raw_U, U0)
#             eig = log_eig.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#             res = log_res.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#             tmp_bank = self._compose_bank_for_descriptor_params(base_bank, new_classes, mu, U, eig, res)
#             logits_new = self._geometry_logits_from_bank(z_new, tmp_bank, seen_classes)
#             ce_new = self._stable_ce_seen(logits_new, y_new, seen_classes, "descriptor new CE")
#             cur_desc = dict(init_desc)
#             cur_desc.update({"means": mu, "bases": U, "eigvals": eig, "res_vars": res})
#             replay = self.sample_old_replay(old_snapshot, seen_classes, new_desc=cur_desc)
#             z_old = replay["features"].to(self.device)  # type: ignore[index]
#             y_old = replay["global_labels"].to(self.device).long()  # type: ignore[index]
#             logits_old = self._geometry_logits_from_bank(z_old, tmp_bank, seen_classes)
#             ce_replay = self._stable_ce_seen(logits_old, y_old, seen_classes, "descriptor replay CE")
#             # Boundary preservation is the real old/new protection.
#             # It includes ellipsoid-invasion, tangent-space overlap, volume, and band-risk terms.
#             boundary_parts = self._old_new_boundary_preservation_loss(old_snapshot, cur_desc, return_parts=True)
#             margin = self._descriptor_margin_loss(old_snapshot, cur_desc) + self._inc_cfg_float("boundary_preserve_weight", 0.35) * boundary_parts["total"]
#             trust = (mu - mu0).pow(2).mean() + (U - U0).pow(2).mean() + (log_eig - eig0.log()).pow(2).mean() + (log_res - res0.log()).pow(2).mean()
#             loss = new_weight * ce_new + replay_weight * ce_replay + margin_weight * margin + self._inc_cfg_float("lambda_trust", 0.05) * trust
#             _finite(loss, "descriptor refinement loss")
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_([mu, raw_U, log_eig, log_res], self._inc_cfg_float("descriptor_refine_grad_clip", 1.0))
#             opt.step()
#             with torch.no_grad():
#                 # Hard identity preservation around admitted descriptor.
#                 delta = mu - mu0
#                 norm = delta.norm(dim=1, keepdim=True).clamp_min(_EPS)
#                 scale = (max_mean_shift / norm).clamp(max=1.0)
#                 mu.copy_(mu0 + delta * scale)
#                 log_eig.copy_(torch.max(torch.min(log_eig, eig0.log() + max_logvar_shift), eig0.log() - max_logvar_shift))
#                 log_res.copy_(torch.max(torch.min(log_res, res0.log() + max_logvar_shift), res0.log() - max_logvar_shift))
#             pred_old = logits_old.detach().argmax(dim=1)
#             y_old_local = self.global_to_seen_local(y_old, seen_classes)
#             stats["loss"] += float(loss.detach().cpu().item())
#             stats["ce_new"] += float(ce_new.detach().cpu().item())
#             stats["ce_replay"] += float(ce_replay.detach().cpu().item())
#             stats["margin"] += float(margin.detach().cpu().item())
#             stats["replay_acc"] += float((pred_old == y_old_local).float().mean().detach().cpu().item() * 100.0)
#             stats["steps"] += 1.0
#         with torch.no_grad():
#             U = self._orthonormalize_bases(raw_U, U0)
#             eig = log_eig.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#             res = log_res.exp().clamp_min(self._inc_cfg_float("geom_var_floor", 1e-4))
#             final_desc = dict(init_desc)
#             final_desc.update({"means": mu.detach(), "bases": U.detach(), "eigvals": eig.detach(), "res_vars": res.detach()})
#             if self._inc_cfg_bool("use_boundary_projection", False):
#                 final_desc = self._project_new_descriptor_params_out_of_old_tangent_space(final_desc, old_snapshot)
#             self._assert_descriptor_block_valid(final_desc, context="refined new descriptors")
#         denom = max(stats["steps"], 1.0)
#         for k in ("loss", "ce_new", "ce_replay", "margin", "replay_acc"):
#             stats[k] /= denom
#         return {"desc": final_desc, "stats": stats}

#     # ------------------------------------------------------------------
#     # Trainability: descriptor-only default; adapter optional ablation
#     # ------------------------------------------------------------------
#     def _incremental_update_mode(self) -> str:
#         mode = str(getattr(self.args, "incremental_update_mode", "scbgr")).lower().strip()
#         aliases = {
#             "": "scbgr",
#             "none": "scbgr",
#             "clean": "scbgr",
#             "descriptor": "scbgr",
#             "descriptor_only": "scbgr",
#             "rsgi": "scbgr",
#             "scbgr": "scbgr",
#             "scb-gr": "scbgr",
#             "spectral_risk_boundary": "scbgr",
#             "geometry_state_admission": "scbgr",
#             "g2rpa": "geometry_gated_adapter",
#             "g2-rpa": "geometry_gated_adapter",
#             "adapter": "geometry_gated_adapter",
#             "gated_adapter": "geometry_gated_adapter",
#             "geometry_adapter": "geometry_gated_adapter",
#         }
#         mode = aliases.get(mode, mode)
#         if mode not in {"scbgr", "geometry_gated_adapter"}:
#             raise RuntimeError(f"Unsupported incremental_update_mode={mode!r}. Use scbgr or geometry_gated_adapter.")
#         try:
#             setattr(self.args, "incremental_update_mode", mode)
#         except Exception:
#             pass
#         return mode

#     def _adapter_mode_enabled(self) -> bool:
#         return self._incremental_update_mode() == "geometry_gated_adapter"

#     def configure_incremental_trainability(self, old_classes: Sequence[int], new_classes: Sequence[int]) -> List[nn.Parameter]:
#         # Hard-disable old-row transport and legacy paths unless explicit unsafe ablation.
#         if self._inc_cfg_bool("use_sglat_transport", False) or self._inc_cfg_bool("allow_old_model_transport", False):
#             if not self._inc_cfg_bool("unsafe_allow_old_row_transport", False):
#                 raise RuntimeError(
#                     "Old-row SGLAT/affine transport is disabled in the clean NECIL-HSI path. "
#                     "Use new-row safe insertion transport only."
#                 )
#         for _, p in self.model.named_parameters():
#             p.requires_grad = False
#         for attr, value in (
#             ("use_incremental_adapter", False),
#             ("use_bicyc_geometry_cycle", False),
#             ("use_geometry_calibrator", False),
#             ("use_geometry_transport", False),  # model-side old-row transport off; trainer calls new-row transport explicitly.
#         ):
#             if hasattr(self.model, attr):
#                 setattr(self.model, attr, value)
#         for name in ("freeze_backbone_except_allowed", "freeze_semantic_encoder", "freeze_classifier", "freeze_projection_head", "freeze_backbone_only", "disable_incremental_adapter"):
#             fn = getattr(self.model, name, None)
#             if callable(fn):
#                 try:
#                     fn()
#                 except TypeError:
#                     try:
#                         fn(allow_last_block=False)
#                     except TypeError:
#                         pass
#         params: List[nn.Parameter] = []
#         if self._adapter_mode_enabled():
#             adapter = getattr(self.model, "geometry_plastic_adapter", None)
#             if adapter is None:
#                 raise RuntimeError("geometry_gated_adapter ablation requires model.geometry_plastic_adapter.")
#             if hasattr(self.model, "use_geometry_gated_adapter"):
#                 self.model.use_geometry_gated_adapter = True
#             for p in adapter.parameters():
#                 p.requires_grad = True
#                 params.append(p)
#         if self._inc_cfg_bool("use_energy_calibrator", False):
#             if hasattr(self.model, "unfreeze_energy_calibrator"):
#                 self.model.unfreeze_energy_calibrator()
#             for name, p in self.model.named_parameters():
#                 if "energy_calibrator" in name or "old_bias" in name or "new_bias" in name or "old_log_scale" in name or "new_log_scale" in name:
#                     p.requires_grad = True
#                     params.append(p)
#         if self._inc_cfg_bool("use_adaptive_boundary", False):
#             clf = getattr(self.model, "classifier", None)
#             if clf is not None and hasattr(clf, "boundary_parameters"):
#                 if hasattr(clf, "freeze_old_boundary_radii"):
#                     clf.freeze_old_boundary_radii(len(old_classes))
#                 for p in clf.boundary_parameters():
#                     if p.requires_grad:
#                         params.append(p)
#         bad = []
#         allowed = ("geometry_plastic_adapter", "energy_calibrator", "old_bias", "new_bias", "old_log_scale", "new_log_scale", "boundary")
#         for name, p in self.model.named_parameters():
#             if p.requires_grad and not any(a in name for a in allowed):
#                 bad.append(name)
#         if bad:
#             raise RuntimeError(f"Forbidden trainable incremental parameters: {bad[:20]}")
#         names = [name for name, p in self.model.named_parameters() if p.requires_grad]
#         print(f"[Incremental Trainability] mode={self._incremental_update_mode()} | trainable={names if names else 'descriptor-only (no model weights)'}")
#         return list(dict.fromkeys(params))

#     # Uploaded-file compatibility names.
#     def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[nn.Parameter]:
#         return self.configure_incremental_trainability(list(range(int(old_class_count))), [])

#     def _set_incremental_trainable_params(self, old_class_count: int) -> List[nn.Parameter]:
#         return self._set_clean_incremental_trainable_params(old_class_count)

#     # ------------------------------------------------------------------
#     # Optional adapter ablation training (bounded, not default)
#     # ------------------------------------------------------------------
#     def _adapt_replay_features_for_adapter_training(self, z_old: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """Apply the geometry-gated adapter to synthetic old z-space samples.

#         Synthetic replay samples already live in GeometryBank z-space, so they
#         cannot pass through the backbone/projection. But when the adapter is
#         trainable, old replay must still pass through the adapter directly;
#         otherwise CE_replay has no gradient path into the adapter and cannot
#         teach gate≈0 in old basins.
#         """
#         z_old = _finite(z_old.float(), "adapter old replay base features")
#         fn = getattr(self.model, "adapt_projected_features", None)
#         if callable(fn):
#             try:
#                 out = fn(z_old, force=True, return_delta=True)
#             except TypeError:
#                 out = fn(z_old)
#             if isinstance(out, dict):
#                 z = out.get("features", out.get("projected_features", None))
#                 if not torch.is_tensor(z):
#                     raise RuntimeError("adapt_projected_features(return_delta=True) did not return features.")
#                 delta = out.get("delta", z - z_old)
#                 gate = out.get("gate", torch.zeros((z.size(0), 1), device=z.device, dtype=z.dtype))
#                 return {"features": _finite(z.float(), "adapted old replay features"), "delta": delta, "gate": gate}
#             if torch.is_tensor(out):
#                 return {"features": _finite(out.float(), "adapted old replay features"), "delta": out - z_old, "gate": torch.zeros((out.size(0), 1), device=out.device, dtype=out.dtype)}
#         return {"features": z_old, "delta": torch.zeros_like(z_old), "gate": torch.zeros((z_old.size(0), 1), device=z_old.device, dtype=z_old.dtype)}

#     def _adapter_regularization_loss(
#         self,
#         *,
#         new_out: Dict[str, torch.Tensor],
#         old_adapt: Dict[str, torch.Tensor],
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         """Bounded plasticity regularizer for G²RPA.

#         Old synthetic replay must remain nearly unchanged. New real samples may
#         move, but only inside a small trust region. This prevents the adapter
#         from becoming an uncontrolled incremental classifier while still giving
#         enough plasticity to fix descriptor-only old/new bias.
#         """
#         ref = old_adapt["features"]
#         old_delta = old_adapt.get("delta", torch.zeros_like(ref))
#         old_gate = old_adapt.get("gate", torch.zeros((ref.size(0), 1), device=ref.device, dtype=ref.dtype))
#         new_delta = new_out.get("adapter_delta", torch.zeros_like(new_out["features"]))
#         new_gate = new_out.get("adapter_gate", torch.zeros((new_out["features"].size(0), 1), device=new_out["features"].device, dtype=new_out["features"].dtype))

#         old_delta_loss = old_delta.pow(2).mean()
#         old_gate_loss = old_gate.clamp_min(0.0).mean()
#         max_new_delta = self._inc_cfg_float("adapter_new_delta_max", self._inc_cfg_float("adapter_max_scale", 0.10))
#         new_norm = new_delta.norm(dim=1)
#         new_delta_loss = F.relu(new_norm - max_new_delta).pow(2).mean()
#         new_gate_target = self._inc_cfg_float("adapter_new_gate_target", 0.15)
#         new_gate_max = self._inc_cfg_float("adapter_new_gate_max_target", self._inc_cfg_float("adapter_new_gate_max", 0.35))
#         new_gate_mean = new_gate.clamp_min(0.0).mean()
#         new_gate_loss = F.relu(new_gate_mean - new_gate_max).pow(2) + 0.10 * (new_gate_mean - new_gate_target).pow(2)

#         loss = (
#             self._inc_cfg_float("adapter_old_delta_weight", 1.00) * old_delta_loss
#             + self._inc_cfg_float("adapter_old_gate_weight", 0.75) * old_gate_loss
#             + self._inc_cfg_float("adapter_new_delta_weight", 0.25) * new_delta_loss
#             + self._inc_cfg_float("adapter_new_gate_weight", 0.10) * new_gate_loss
#         )
#         stats = {
#             "gate_old": float(old_gate.detach().mean().cpu().item()) if old_gate.numel() else 0.0,
#             "gate_new": float(new_gate.detach().mean().cpu().item()) if new_gate.numel() else 0.0,
#             "delta_old": float(old_delta.detach().norm(dim=1).mean().cpu().item()) if old_delta.numel() else 0.0,
#             "delta_new": float(new_delta.detach().norm(dim=1).mean().cpu().item()) if new_delta.numel() else 0.0,
#             "adapter_reg": float(loss.detach().cpu().item()),
#         }
#         return _finite(loss, "adapter_regularization_loss"), stats

#     def train_one_adapter_epoch(
#         self,
#         loader,
#         optimizer: optim.Optimizer,
#         *,
#         old_snapshot: Dict[str, torch.Tensor],
#         seen_classes: Sequence[int],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         new_desc: Dict[str, torch.Tensor],
#     ) -> Dict[str, float]:
#         if optimizer is None:
#             return {"adapter_loss": 0.0, "ce_new": 0.0, "ce_replay": 0.0, "steps": 0.0}
#         self.model.train()
#         stats = {
#             "adapter_loss": 0.0, "ce_new": 0.0, "ce_replay": 0.0, "adaptive_boundary": 0.0,
#             "adapter_reg": 0.0, "gate_old": 0.0, "gate_new": 0.0, "delta_old": 0.0, "delta_new": 0.0,
#             "steps": 0.0, "old_replay_acc": 0.0,
#         }
#         old_ref = self.snapshot_old_geometry(old_classes)
#         for batch in loader:
#             x, y, spectra, _ = self._unpack_batch(batch)
#             x = x.float().to(self.device, non_blocking=True)
#             y = y.long().to(self.device, non_blocking=True).view(-1)
#             self.assert_global_labels_in_set(y, new_classes, "adapter real new batch")
#             optimizer.zero_grad(set_to_none=True)

#             # Real new samples go through the model path, therefore through the
#             # adapter when geometry_gated_adapter is enabled.
#             out = self.extract_incremental_features(x, spectra)
#             z_new = out["features"]
#             logits_new = self.compute_seen_logits(z_new, seen_classes, mode="geometry", old_classes=old_classes, new_classes=new_classes)["logits"]  # type: ignore[index]
#             ce_new = self._stable_ce_seen(logits_new, y, seen_classes, "adapter CE_new")

#             # Synthetic old replay is already z-space. It must pass through the
#             # adapter directly; otherwise CE_replay cannot train the adapter and
#             # old-region gates never learn to close.
#             replay = self.sample_old_replay(old_snapshot, seen_classes, new_desc=new_desc)
#             z_old_base = replay["features"].to(self.device)  # type: ignore[index]
#             y_old = replay["global_labels"].to(self.device).long()  # type: ignore[index]
#             old_adapt = self._adapt_replay_features_for_adapter_training(z_old_base)
#             z_old = old_adapt["features"]
#             logits_old = self.compute_seen_logits(z_old, seen_classes, mode="geometry", old_classes=old_classes, new_classes=new_classes)["logits"]  # type: ignore[index]
#             ce_replay = self._stable_ce_seen(logits_old, y_old, seen_classes, "adapter CE_replay")

#             y_new_local = self.global_to_seen_local(y, seen_classes)
#             y_old_local = self.global_to_seen_local(y_old, seen_classes)
#             logits_all = torch.cat([logits_new, logits_old], dim=0)
#             labels_all = torch.cat([y_new_local, y_old_local], dim=0)
#             adaptive_boundary = self._adaptive_boundary_loss_from_current_bank(
#                 logits_all,
#                 labels_all,
#                 old_class_count=len(old_classes),
#                 seen_classes=seen_classes,
#             )
#             adapter_reg, reg_stats = self._adapter_regularization_loss(new_out=out, old_adapt=old_adapt)

#             loss = (
#                 self._inc_cfg_float("adapter_new_ce_weight", 1.00) * ce_new
#                 + self._inc_cfg_float("lambda_replay", self._inc_cfg_float("adapter_replay_weight", 1.00)) * ce_replay
#                 + self._inc_cfg_float("adaptive_boundary_loss_weight", 1.00) * adaptive_boundary
#                 + self._inc_cfg_float("adapter_regularization_weight", 1.00) * adapter_reg
#             )
#             _finite(loss, "adapter incremental loss")
#             loss.backward()
#             trainable = [p for p in self.model.parameters() if p.requires_grad]
#             if trainable:
#                 torch.nn.utils.clip_grad_norm_(trainable, self._inc_cfg_float("grad_clip_inc", 0.5))
#             optimizer.step()
#             self.assert_old_geometry_unchanged(old_ref, "adapter_epoch", atol=1e-6)

#             stats["adapter_loss"] += float(loss.detach().cpu().item())
#             stats["ce_new"] += float(ce_new.detach().cpu().item())
#             stats["ce_replay"] += float(ce_replay.detach().cpu().item())
#             stats["adaptive_boundary"] += float(adaptive_boundary.detach().cpu().item()) if torch.is_tensor(adaptive_boundary) else 0.0
#             stats["adapter_reg"] += float(adapter_reg.detach().cpu().item())
#             stats["gate_old"] += reg_stats["gate_old"]
#             stats["gate_new"] += reg_stats["gate_new"]
#             stats["delta_old"] += reg_stats["delta_old"]
#             stats["delta_new"] += reg_stats["delta_new"]
#             stats["old_replay_acc"] += float((logits_old.detach().argmax(dim=1) == y_old_local).float().mean().cpu().item() * 100.0)
#             stats["steps"] += 1.0
#         denom = max(stats["steps"], 1.0)
#         for k in ("adapter_loss", "ce_new", "ce_replay", "adaptive_boundary", "adapter_reg", "gate_old", "gate_new", "delta_old", "delta_new", "old_replay_acc"):
#             stats[k] /= denom
#         return stats


#     # ------------------------------------------------------------------
#     # Validation/diagnostics/checkpoint selection
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def validate_incremental_phase(self, loader, old_classes: Sequence[int], new_classes: Sequence[int], seen_classes: Sequence[int]) -> Dict[str, Any]:
#         self.model.eval()
#         total_loss = 0.0
#         total = 0
#         correct = 0
#         per_total = {int(c): 0 for c in seen_classes}
#         per_correct = {int(c): 0 for c in seen_classes}
#         pred_hist = {int(c): 0 for c in seen_classes}
#         old_logits: List[torch.Tensor] = []
#         new_logits: List[torch.Tensor] = []
#         invalid = 0
#         for batch in loader:
#             x, y, spectra, _ = self._unpack_batch(batch)
#             x = x.float().to(self.device, non_blocking=True)
#             y = y.long().to(self.device, non_blocking=True).view(-1)
#             self.assert_global_labels_in_set(y, seen_classes, "cumulative validation batch")
#             out = self.extract_incremental_features(x, spectra)
#             logits = self.compute_seen_logits(out["features"], seen_classes, mode="geometry", old_classes=old_classes, new_classes=new_classes)["logits"]  # type: ignore[index]
#             y_local = self.global_to_seen_local(y, seen_classes)
#             loss = self._stable_ce(logits, y_local)
#             pred_local = logits.argmax(dim=1)
#             pred_global = self.seen_local_to_global(pred_local, seen_classes)
#             total_loss += float(loss.detach().cpu().item()) * int(y.numel())
#             total += int(y.numel())
#             correct += int((pred_global == y).sum().item())
#             for cls in seen_classes:
#                 mask = y == int(cls)
#                 n = int(mask.sum().item())
#                 if n > 0:
#                     per_total[int(cls)] += n
#                     per_correct[int(cls)] += int((pred_global[mask] == int(cls)).sum().item())
#             for cls in pred_global.detach().cpu().tolist():
#                 if int(cls) in pred_hist:
#                     pred_hist[int(cls)] += 1
#                 else:
#                     invalid += 1
#             old_idx = [seen_classes.index(c) for c in old_classes if c in seen_classes]
#             new_idx = [seen_classes.index(c) for c in new_classes if c in seen_classes]
#             if old_idx:
#                 old_logits.append(logits[:, old_idx].detach().mean(dim=1))
#             if new_idx:
#                 new_logits.append(logits[:, new_idx].detach().mean(dim=1))
#         acc = 100.0 * correct / max(total, 1)
#         old_total = sum(per_total[c] for c in old_classes if c in per_total)
#         old_correct = sum(per_correct[c] for c in old_classes if c in per_correct)
#         new_total = sum(per_total[c] for c in new_classes if c in per_total)
#         new_correct = sum(per_correct[c] for c in new_classes if c in per_correct)
#         old_acc = 100.0 * old_correct / max(old_total, 1)
#         new_acc = 100.0 * new_correct / max(new_total, 1)
#         hm = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
#         per_class_acc = {int(c): 100.0 * per_correct[int(c)] / max(per_total[int(c)], 1) for c in seen_classes}
#         avg_acc = sum(per_class_acc.values()) / max(len(per_class_acc), 1)
#         old_mean = float(torch.cat(old_logits).mean().cpu().item()) if old_logits else 0.0
#         new_mean = float(torch.cat(new_logits).mean().cpu().item()) if new_logits else 0.0
#         return {
#             "loss": total_loss / max(total, 1),
#             "acc": acc,
#             "old_acc": old_acc,
#             "new_acc": new_acc,
#             "hm": hm,
#             "aa": avg_acc,
#             "per_class_accuracy": per_class_acc,
#             "prediction_histogram": pred_hist,
#             "invalid_prediction_rate": float(invalid) / float(max(total, 1)),
#             "old_logit_mean": old_mean,
#             "new_logit_mean": new_mean,
#             "old_new_logit_gap": old_mean - new_mean,
#         }

#     def _compute_old_new_overlap_stats(self, old_snapshot: Dict[str, torch.Tensor], new_desc: Dict[str, torch.Tensor]) -> Dict[str, float]:
#         risk = self._risk_matrix_from_descriptors(old_snapshot, new_desc)
#         return {
#             "old_new_risk_max": float(risk["risk"].max().detach().cpu().item()) if risk["risk"].numel() else 0.0,
#             "old_new_risk_mean": float(risk["risk"].mean().detach().cpu().item()) if risk["risk"].numel() else 0.0,
#             "old_new_subspace_overlap_max": float(risk["subspace"].max().detach().cpu().item()) if risk["subspace"].numel() else 0.0,
#             "old_new_center_distance_min": float(risk["dist"].min().detach().cpu().item()) if risk["dist"].numel() else 0.0,
#             "old_new_ellipsoid_energy_min": float(risk["energy"].min().detach().cpu().item()) if risk["energy"].numel() else 0.0,
#         }

#     def select_best_incremental_checkpoint(self, val_stats: Dict[str, Any], drift_stats: Dict[str, float], overlap_stats: Dict[str, float]) -> float:
#         """Validation-first checkpoint score for descriptor-only NECIL.

#         The previous score subtracted the raw old/new logit gap.  Geometry-energy
#         logits in this code can be thousands of units apart, so that term drowned
#         the actual validation harmonic mean and selected worse descriptors.

#         Correct policy:
#         - primary: old/new harmonic mean on real validation data
#         - secondary: keep both old and new non-collapsed
#         - small penalties: frozen-row drift and old/new geometry risk
#         - optional normalized logit-gap penalty only after tanh scaling
#         """
#         hm = float(val_stats.get("hm", 0.0))
#         acc = float(val_stats.get("acc", val_stats.get("overall_accuracy", 0.0)))
#         old_acc = float(val_stats.get("old_acc", 0.0))
#         new_acc = float(val_stats.get("new_acc", 0.0))
#         balance = min(old_acc, new_acc)
#         drift = max(float(v) for v in drift_stats.values()) if drift_stats else 0.0
#         overlap = float(overlap_stats.get("old_new_risk_max", 0.0))

#         raw_gap = abs(float(val_stats.get("old_new_logit_gap", 0.0)))
#         gap_scale = max(self._inc_cfg_float("ckpt_logit_gap_scale", 1000.0), 1e-6)
#         gap_penalty = math.tanh(raw_gap / gap_scale)

#         return (
#             hm
#             + self._inc_cfg_float("ckpt_balance_weight", 0.20) * balance
#             + self._inc_cfg_float("ckpt_acc_weight", 0.05) * acc
#             - self._inc_cfg_float("ckpt_logit_gap_weight", 0.0) * gap_penalty
#             - self._inc_cfg_float("ckpt_geometry_drift_weight", 50.0) * drift
#             - self._inc_cfg_float("ckpt_overlap_weight", 2.0) * overlap
#         )

#     # Compatibility with outer Trainer.
#     def _select_score(self, val_stats: Dict[str, Any], phase: int) -> float:
#         return float(val_stats.get("hm", val_stats.get("acc", 0.0)))

#     def _capture_state(self):
#         return copy.deepcopy(self.model.state_dict())

#     def _restore_state(self, state):
#         if state is not None:
#             self.model.load_state_dict(state)

#     def _save_phase_artifacts(self, phase: int, history: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
#         save_dir = str(getattr(self, "save_dir", getattr(self.args, "save_dir", "./results")))
#         os.makedirs(save_dir, exist_ok=True)
#         json_path = os.path.join(save_dir, f"phase_{int(phase)}_incremental_diagnostics.json")
#         pt_path = os.path.join(save_dir, f"phase_{int(phase)}_incremental_handoff.pt")
#         def _jsonable(v: Any):
#             if torch.is_tensor(v):
#                 return v.detach().cpu().tolist()
#             if isinstance(v, dict):
#                 return {str(k): _jsonable(val) for k, val in v.items()}
#             if isinstance(v, (list, tuple)):
#                 return [_jsonable(x) for x in v]
#             if isinstance(v, (int, float, str, bool)) or v is None:
#                 return v
#             return str(v)
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(_jsonable({"history": history, "diagnostics": diagnostics}), f, indent=2)
#         torch.save({"history": history, "diagnostics": diagnostics}, pt_path)
#         print(f"[Incremental Diagnostics] saved json={json_path} | pt={pt_path}")


#     def load_base_handoff(self, phase: int) -> Dict[str, Any]:
#         """Load phase-0 PRL-style geometry handoff when available.

#         The handoff is not required for correctness, but when present it makes
#         the base preparation actionable: replay strength, insertion margin, and
#         new-row transport activation are initialized from the phase-0 geometry
#         certificate instead of guessed again inside the incremental trainer.
#         """
#         if int(phase) <= 0:
#             return {}
#         save_dir = str(getattr(self, "save_dir", getattr(self.args, "save_dir", "./results")))
#         candidates = [
#             os.path.join(save_dir, "phase_0_base_handoff.pt"),
#             os.path.join(save_dir, "phase_0_base_handoff.json"),
#         ]
#         handoff: Dict[str, Any] = {}
#         for path in candidates:
#             if not os.path.exists(path):
#                 continue
#             try:
#                 if path.endswith(".pt"):
#                     obj = torch.load(path, map_location=self.device)
#                     handoff = obj if isinstance(obj, dict) else {}
#                 else:
#                     with open(path, "r", encoding="utf-8") as f:
#                         obj = json.load(f)
#                     handoff = obj if isinstance(obj, dict) else {}
#                 if handoff:
#                     print(f"[BaseHandoff] loaded {path}")
#                     break
#             except Exception as exc:
#                 print(f"[BaseHandoff WARN] could not load {path}: {exc}")
#         if not handoff:
#             return {}

#         # Make certificate recommendations operational. These are runtime fields,
#         # not permanent argparse mutations. _inc_cfg_* checks self first.
#         margin = handoff.get("recommended_insertion_margin", None)
#         if isinstance(margin, (int, float)) and float(margin) > 0:
#             setattr(self, "old_new_geometry_margin", float(margin))

#         replay = handoff.get("recommended_replay_per_class", None)
#         if isinstance(replay, dict) and replay:
#             vals = []
#             for v in replay.values():
#                 if isinstance(v, (int, float)):
#                     vals.append(int(v))
#             if vals:
#                 setattr(self, "synthetic_replay_per_class", int(max(vals)))

#         # Do not silently enable transport from the handoff.  The certificate may
#         # recommend transport, but the CLI contract decides whether the run is
#         # descriptor-only or a transport ablation.  Secretly setting
#         # use_new_row_transport=True makes results non-auditable.
#         if bool(handoff.get("transport_required", False)):
#             setattr(self, "handoff_transport_recommended", True)
#             if self._inc_cfg_bool("use_geometry_transport", False) or self._inc_cfg_bool("use_new_row_transport", False):
#                 setattr(self, "use_new_row_transport", True)
#         self._base_handoff = handoff
#         return handoff

#     # ------------------------------------------------------------------
#     # Main epoch/phase flow
#     # ------------------------------------------------------------------
#     def assert_incremental_contract(self, phase: int, old_classes: Sequence[int], new_classes: Sequence[int], seen_classes: Sequence[int]) -> None:
#         if int(phase) <= 0:
#             raise RuntimeError("phase must be > 0 for incremental training.")
#         if seen_classes != list(old_classes) + list(new_classes):
#             raise RuntimeError(f"seen_classes must equal old+new order. old={old_classes}, new={new_classes}, seen={seen_classes}")
#         self.assert_geometry_exists(old_classes, context="incremental old GeometryBank")
#         self.freeze_old_geometry(old_classes)
#         if hasattr(self.model, "set_incremental_mode"):
#             self.model.set_incremental_mode(phase=int(phase), old_class_count=len(old_classes))
#         if hasattr(self, "_set_model_phase_and_old_count"):
#             self._set_model_phase_and_old_count(int(phase), len(old_classes))

#     def train_incremental_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4) -> Dict[str, Any]:
#         phase = int(phase)
#         old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
#         self._active_seen_classes = list(seen_classes)
#         print(f"==== Incremental Phase {phase} | SCBGR Descriptor-Only NECIL-HSI ====")
#         print(f"[Classes] old={old_classes} | new={new_classes} | seen={seen_classes}")
#         self.dataset.start_phase(phase)
#         self.assert_incremental_contract(phase, old_classes, new_classes, seen_classes)
#         base_handoff = self.load_base_handoff(phase)

#         if hasattr(self.model, "ensure_class_capacity"):
#             self.model.ensure_class_capacity(max(seen_classes) + 1)

#         train_loader = self.dataset.get_phase_dataloader(phase, split="train", batch_size=batch_size, shuffle=True)
#         val_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)

#         old_snapshot0 = self.snapshot_old_geometry(old_classes)
#         z_new, y_new, s_new, b_new = self.collect_current_phase_features(train_loader, new_classes)
#         raw_new_desc = self._estimate_geometry_from_features(z_new, y_new, new_classes, spectral_summary=s_new, band_summary=b_new)
#         admitted_desc, admission_stats = self.admit_new_geometry(raw_new_desc, old_snapshot0, new_classes)
#         self.assert_old_geometry_unchanged(old_snapshot0, "post_new_admission")
#         self._commit_new_descriptors(admitted_desc, phase=phase)
#         self.assert_geometry_exists(seen_classes, context="post new admission")
#         self.assert_old_geometry_unchanged(old_snapshot0, "post_new_commit")

#         trainable_params = self.configure_incremental_trainability(old_classes, new_classes)
#         optimizer = None
#         if trainable_params:
#             opt_lr = self._inc_cfg_float("adapter_lr", float(lr)) if self._adapter_mode_enabled() else float(lr)
#             opt_wd = self._inc_cfg_float("adapter_weight_decay", 0.0) if self._adapter_mode_enabled() else self._inc_cfg_float("weight_decay", 0.0)
#             optimizer = optim.AdamW(trainable_params, lr=float(opt_lr), weight_decay=float(opt_wd))
#             print(f"[Incremental Optimizer] lr={float(opt_lr):.3g} | weight_decay={float(opt_wd):.3g} | params={sum(p.numel() for p in trainable_params):,}")

#         history: Dict[str, List[float]] = {
#             "val_acc": [], "val_old_acc": [], "val_new_acc": [], "val_hm": [], "val_aa": [],
#             "ce_new": [], "ce_replay": [], "desc_margin": [], "desc_loss": [], "old_replay_acc": [],
#             "old_new_logit_gap": [], "old_new_risk_max": [], "old_new_overlap_max": [],
#             "old_mean_drift": [], "old_basis_drift": [], "old_eigval_drift": [], "old_resvar_drift": [],
#             "boundary_replay_count": [], "boundary_replay_fallback": [],
#             "adapter_loss": [], "adapter_reg": [], "adaptive_boundary": [], "gate_old": [], "gate_new": [], "delta_old": [], "delta_new": [], "checkpoint_score": [],
#         }
#         phase_diagnostics: Dict[str, Any] = {
#             "phase": phase,
#             "old_classes": old_classes,
#             "new_classes": new_classes,
#             "seen_classes": seen_classes,
#             "base_handoff_loaded": bool(base_handoff),
#             "base_handoff": base_handoff,
#             "admission": admission_stats,
#             "trainable_parameter_names": [n for n, p in self.model.named_parameters() if p.requires_grad],
#         }

#         best_state = self._capture_state()
#         best_score = -1e18
#         best_desc = admitted_desc
#         epochs = int(max(0, epochs))
#         steps_per_epoch = self._inc_cfg_int("descriptor_refine_steps_per_epoch", self._inc_cfg_int("descriptor_refine_steps", 20))
#         old_snapshot_phase = self.snapshot_old_geometry(old_classes)

#         # Initial validation before descriptor refinement.
#         # This is a real checkpoint candidate. In the failed run, InitVal was
#         # consistently better than all refined states, but the old code ignored it
#         # and was forced to pick a degraded epoch.
#         init_val = self.validate_incremental_phase(val_loader, old_classes, new_classes, seen_classes)
#         init_drift = self.assert_old_geometry_unchanged(old_snapshot_phase, f"phase{phase}_init_validation_drift_check")
#         init_overlap = self._compute_old_new_overlap_stats(old_snapshot_phase, admitted_desc)
#         init_score = self.select_best_incremental_checkpoint(init_val, init_drift, init_overlap)
#         best_score = init_score
#         best_state = self._capture_state()
#         best_desc = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in admitted_desc.items()}
#         no_improve_epochs = 0
#         refine_patience = self._inc_cfg_int("descriptor_refine_patience", 3)
#         min_new_drop = self._inc_cfg_float("descriptor_refine_max_new_drop", 1.0)
#         print(
#             f"[InitVal] Phase {phase} | ValAcc={init_val['acc']:.2f}% | Old={init_val['old_acc']:.2f}% | "
#             f"New={init_val['new_acc']:.2f}% | HM={init_val['hm']:.2f}% | Gap={init_val['old_new_logit_gap']:.4f} | Score={init_score:.4f}"
#         )

#         for epoch in range(epochs):
#             # Recollect features because optional adapter ablation may change scoring z.
#             z_new, y_new, s_new, b_new = self.collect_current_phase_features(train_loader, new_classes)
#             current_bank = self._bank_dict()
#             ids = torch.as_tensor(new_classes, device=self.device, dtype=torch.long)
#             current_desc = {
#                 "class_ids": ids,
#                 "means": current_bank["means"].index_select(0, ids).detach(),
#                 "bases": current_bank["bases"].index_select(0, ids).detach(),
#                 "eigvals": current_bank["eigvals"].index_select(0, ids).detach(),
#                 "res_vars": current_bank["res_vars"].index_select(0, ids).detach(),
#                 "active_ranks": current_bank["active_ranks"].index_select(0, ids).detach(),
#                 "sample_counts": current_bank["sample_counts"].flatten().index_select(0, ids).detach(),
#                 "reliability": current_bank.get("reliability", torch.ones_like(current_bank["sample_counts"].float())).flatten().index_select(0, ids).detach() if torch.is_tensor(current_bank.get("reliability", None)) else torch.ones(len(new_classes), device=self.device),
#             }
#             if torch.is_tensor(current_bank.get("band_importance", None)) and current_bank["band_importance"].size(0) > int(ids.max().item()):
#                 current_desc["band_importance"] = current_bank["band_importance"].index_select(0, ids).detach()
#             # Use the implementation name, not self.refine_new_descriptors.
#             # Trainer/config code may store the CLI boolean flag under
#             # self.refine_new_descriptors, which shadows any method with that
#             # name and causes: TypeError: 'bool' object is not callable.
#             refined = self._refine_new_descriptors_impl(
#                 z_new=z_new,
#                 y_new=y_new,
#                 seen_classes=seen_classes,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#                 old_snapshot=old_snapshot_phase,
#                 init_desc=current_desc,
#                 steps=steps_per_epoch,
#             )
#             desc = refined["desc"]
#             desc_stats = refined["stats"]
#             self._commit_new_descriptors(desc, phase=phase)
#             self.assert_old_geometry_unchanged(old_snapshot_phase, f"phase{phase}_epoch{epoch+1}_post_descriptor_refine")

#             adapter_stats = {"adapter_loss": 0.0, "ce_new": 0.0, "ce_replay": 0.0, "old_replay_acc": 0.0}
#             if optimizer is not None:
#                 adapter_stats = self.train_one_adapter_epoch(train_loader, optimizer, old_snapshot=old_snapshot_phase, seen_classes=seen_classes, old_classes=old_classes, new_classes=new_classes, new_desc=desc)
#                 self.assert_old_geometry_unchanged(old_snapshot_phase, f"phase{phase}_epoch{epoch+1}_post_adapter")

#             val = self.validate_incremental_phase(val_loader, old_classes, new_classes, seen_classes)
#             drift = self.assert_old_geometry_unchanged(old_snapshot_phase, f"phase{phase}_epoch{epoch+1}_validation_drift_check")
#             overlap = self._compute_old_new_overlap_stats(old_snapshot_phase, desc)
#             score = self.select_best_incremental_checkpoint(val, drift, overlap)
#             if score > best_score:
#                 best_score = score
#                 best_state = self._capture_state()
#                 best_desc = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in desc.items()}
#                 no_improve_epochs = 0
#             else:
#                 no_improve_epochs += 1
#             replay_probe = self.sample_old_replay(old_snapshot_phase, seen_classes, new_desc=desc)
#             replay_stats = replay_probe["stats"]  # type: ignore[index]
#             history["val_acc"].append(float(val["acc"]))
#             history["val_old_acc"].append(float(val["old_acc"]))
#             history["val_new_acc"].append(float(val["new_acc"]))
#             history["val_hm"].append(float(val["hm"]))
#             history["val_aa"].append(float(val["aa"]))
#             history["ce_new"].append(float(desc_stats.get("ce_new", adapter_stats.get("ce_new", 0.0))))
#             history["ce_replay"].append(float(desc_stats.get("ce_replay", adapter_stats.get("ce_replay", 0.0))))
#             history["desc_margin"].append(float(desc_stats.get("margin", 0.0)))
#             history["desc_loss"].append(float(desc_stats.get("loss", 0.0)))
#             history["old_replay_acc"].append(float(desc_stats.get("replay_acc", adapter_stats.get("old_replay_acc", 0.0))))
#             history["old_new_logit_gap"].append(float(val["old_new_logit_gap"]))
#             history["old_new_risk_max"].append(float(overlap["old_new_risk_max"]))
#             history["old_new_overlap_max"].append(float(overlap["old_new_subspace_overlap_max"]))
#             history["old_mean_drift"].append(float(drift.get("old_means_max_abs_drift", 0.0)))
#             history["old_basis_drift"].append(float(drift.get("old_bases_max_abs_drift", 0.0)))
#             history["old_eigval_drift"].append(float(drift.get("old_eigvals_max_abs_drift", 0.0)))
#             history["old_resvar_drift"].append(float(drift.get("old_res_vars_max_abs_drift", 0.0)))
#             history["boundary_replay_count"].append(float(replay_stats.get("boundary_replay_count", 0.0)))
#             history["boundary_replay_fallback"].append(float(replay_stats.get("boundary_replay_fallback", 0.0)))
#             history["adapter_loss"].append(float(adapter_stats.get("adapter_loss", 0.0)))
#             history["adapter_reg"].append(float(adapter_stats.get("adapter_reg", 0.0)))
#             history["adaptive_boundary"].append(float(adapter_stats.get("adaptive_boundary", 0.0)))
#             history["gate_old"].append(float(adapter_stats.get("gate_old", 0.0)))
#             history["gate_new"].append(float(adapter_stats.get("gate_new", 0.0)))
#             history["delta_old"].append(float(adapter_stats.get("delta_old", 0.0)))
#             history["delta_new"].append(float(adapter_stats.get("delta_new", 0.0)))
#             history["checkpoint_score"].append(float(score))
#             print(
#                 f"[IncEpoch] Phase {phase} Ep {epoch+1:03d}/{epochs} | "
#                 f"DescLoss={desc_stats.get('loss', 0.0):.4f} | CEnew={history['ce_new'][-1]:.4f} | "
#                 f"CEold={history['ce_replay'][-1]:.4f} | ReplayAcc={history['old_replay_acc'][-1]:.2f}% | "
#                 f"Val={val['acc']:.2f}% | Old={val['old_acc']:.2f}% | New={val['new_acc']:.2f}% | HM={val['hm']:.2f}% | "
#                 f"Gap={val['old_new_logit_gap']:.4f} | RiskMax={overlap['old_new_risk_max']:.4f} | "
#                 f"OverlapMax={overlap['old_new_subspace_overlap_max']:.4f} | "
#                 f"GateOld={adapter_stats.get('gate_old', 0.0):.4f} | GateNew={adapter_stats.get('gate_new', 0.0):.4f} | "
#                 f"ABnd={adapter_stats.get('adaptive_boundary', 0.0):.4f} | OldDrift={max(drift.values()) if drift else 0.0:.2e} | Score={score:.4f}"
#             )
#             if no_improve_epochs >= refine_patience or float(val.get("new_acc", 0.0)) < float(init_val.get("new_acc", 0.0)) - min_new_drop:
#                 print(
#                     f"[DescriptorRefine STOP] Phase {phase} stopped at epoch {epoch+1}: "
#                     f"best_score={best_score:.4f}, current_score={score:.4f}, "
#                     f"init_new={float(init_val.get('new_acc', 0.0)):.2f}, current_new={float(val.get('new_acc', 0.0)):.2f}. "
#                     "Restoring best descriptor checkpoint."
#                 )
#                 break

#         self._restore_state(best_state)
#         self._commit_new_descriptors(best_desc, phase=phase)
#         self.assert_old_geometry_unchanged(old_snapshot_phase, f"phase{phase}_post_best_restore")
#         # Freeze all seen rows after the phase. New rows become old for next phase.
#         self.freeze_old_geometry(seen_classes)
#         if hasattr(self.dataset, "finalize_phase"):
#             self.dataset.finalize_phase(phase)
#         if hasattr(self, "_set_model_phase_and_old_count"):
#             self._set_model_phase_and_old_count(phase, len(seen_classes))
#         final_val = self.validate_incremental_phase(val_loader, old_classes, new_classes, seen_classes)
#         final_overlap = self._compute_old_new_overlap_stats(old_snapshot_phase, best_desc)
#         phase_diagnostics.update({
#             "final_val": final_val,
#             "final_overlap": final_overlap,
#             "best_checkpoint_score": best_score,
#             "best_descriptor_classes": [int(c) for c in best_desc["class_ids"].detach().cpu().tolist()],
#         })
#         history["final_val"] = final_val  # type: ignore[assignment]
#         if hasattr(self, "save_checkpoint"):
#             self.save_checkpoint(phase, history)
#         self._save_phase_artifacts(phase, history, phase_diagnostics)
#         print(
#             f"[PhaseDone] Phase {phase} | Final Val={final_val['acc']:.2f}% | Old={final_val['old_acc']:.2f}% | "
#             f"New={final_val['new_acc']:.2f}% | HM={final_val['hm']:.2f}% | old_geometry_frozen=True"
#         )
#         return history







# from __future__ import annotations

# import copy
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import torch
# import torch.nn.functional as F
# import torch.optim as optim

# from losses.loss import (
#     unified_spectral_geometry_loss,
#     sample_geometry_features,
#     sample_boundary_geometry_features,
#     descriptor_subspace_collision_loss,
#     center_to_old_ellipsoid_loss,
#     descriptor_volume_control_loss,
#     descriptor_trust_region_loss,
#     GeometryGatedAdapterLoss,
# )
# from models.classifier import geometry_energy_margin_loss, old_new_invasion_loss


# class IncrementalPhaseTrainer:
#     # ------------------------------------------------------------------
#     # Basic config / tensor helpers
#     # ------------------------------------------------------------------
#     def _zero_like_ref(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if hasattr(self, "_zero"):
#             return self._zero(ref)
#         if torch.is_tensor(ref):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     def _inc_cfg_float(self, name: str, default: float) -> float:
#         return float(getattr(self, name, getattr(self.args, name, default)))

#     def _inc_cfg_int(self, name: str, default: int) -> int:
#         return int(getattr(self, name, getattr(self.args, name, default)))

#     def _inc_cfg_bool(self, name: str, default: bool) -> bool:
#         value = getattr(self, name, getattr(self.args, name, default))
#         if isinstance(value, str):
#             return value.strip().lower() in {"1", "true", "yes", "y", "on"}
#         return bool(value)

#     def _classifier_mode(self) -> str:
#         if hasattr(self, "_inc_classifier_mode"):
#             return str(self._inc_classifier_mode()).lower().strip()
#         return str(getattr(self.args, "incremental_classifier_mode", "geometry_only")).lower().strip()

#     def _seen_classes_for_phase(self, phase: int) -> List[int]:
#         if hasattr(self.dataset, "get_classes_up_to_phase"):
#             seen = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase))]
#             if seen:
#                 return sorted(set(seen))
#         classes: List[int] = []
#         for p in range(int(phase) + 1):
#             classes.extend(int(c) for c in self.dataset.phase_to_classes[p])
#         return sorted(set(classes))

#     def _mask_logits_to_seen_classes(self, logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
#         if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError(f"logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")
#         seen_list = [int(c) for c in seen_classes]
#         if not seen_list:
#             raise RuntimeError("seen_classes is empty.")
#         seen = torch.as_tensor(seen_list, device=logits.device, dtype=torch.long)
#         if int(seen.min().item()) < 0 or int(seen.max().item()) >= logits.size(1):
#             raise RuntimeError(
#                 f"seen class range [{int(seen.min())},{int(seen.max())}] incompatible with logits width={logits.size(1)}"
#             )
#         masked = torch.full_like(logits, -1e9)
#         masked.index_copy_(1, seen, logits.index_select(1, seen))
#         return masked

#     def _assert_batch_labels_in_classes(self, y: torch.Tensor, class_ids: Iterable[int], context: str) -> None:
#         y = y.long().view(-1)
#         allowed = torch.as_tensor([int(c) for c in class_ids], device=y.device, dtype=torch.long)
#         if y.numel() == 0:
#             raise RuntimeError(f"{context}: empty label tensor.")
#         if allowed.numel() == 0:
#             raise RuntimeError(f"{context}: empty allowed class set.")
#         if hasattr(torch, "isin"):
#             ok = torch.isin(y, allowed).all()
#         else:
#             valid = torch.zeros_like(y, dtype=torch.bool)
#             for c in allowed:
#                 valid |= y == int(c)
#             ok = valid.all()
#         if not bool(ok.item()):
#             raise RuntimeError(
#                 f"{context}: labels are not expected global sequential ids. "
#                 f"unique={torch.unique(y).detach().cpu().tolist()}, allowed={allowed.detach().cpu().tolist()}"
#             )

#     def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
#             return self._zero_like_ref(logits)
#         labels = labels.to(device=logits.device).long().view(-1)
#         if labels.numel() == 0:
#             return logits.sum() * 0.0
#         if labels.numel() != logits.size(0):
#             raise RuntimeError(f"CE label/logit batch mismatch: {labels.numel()} vs {logits.size(0)}")
#         min_label = int(labels.min().detach().item())
#         max_label = int(labels.max().detach().item())
#         if min_label < 0 or max_label >= logits.size(1):
#             raise RuntimeError(f"CE labels [{min_label},{max_label}] incompatible with logits width={logits.size(1)}")
#         clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
#         return F.cross_entropy(
#             logits.clamp(-clip, clip),
#             labels,
#             label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#         )

#     def _incremental_accuracy_with_count(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         class_ids: Iterable[int],
#     ) -> Tuple[int, int]:
#         labels = labels.to(device=logits.device).long().view(-1)
#         valid = torch.zeros_like(labels, dtype=torch.bool)
#         for c in [int(x) for x in class_ids]:
#             valid |= labels == int(c)
#         if not bool(valid.any().item()):
#             return 0, 0
#         pred = logits[valid].argmax(dim=1)
#         return int((pred == labels[valid]).sum().item()), int(valid.sum().item())


#     # ------------------------------------------------------------------
#     # SRGP spectral-summary handling for real samples only
#     # ------------------------------------------------------------------
#     def _inc_spectral_summary_is_physical(self, explicit: Optional[bool] = None) -> bool:
#         """Return whether a batch spectral summary is physically wavelength ordered.

#         SRGP spectral residual energy is valid only for raw HSI spectra.  PCA
#         components must not be treated as wavelengths.  Synthetic replay never
#         receives spectral summaries.
#         """
#         if explicit is not None:
#             return bool(explicit)
#         for key in (
#             "spectral_summary_is_physical",
#             "raw_spectral_summary_is_physical",
#             "incremental_spectral_summary_is_physical",
#         ):
#             if hasattr(self.args, key):
#                 return self._inc_cfg_bool(key, False)
#         if hasattr(self.args, "pca_components") and int(getattr(self.args, "pca_components", 0)) > 0:
#             return False
#         return False

#     @staticmethod
#     def _center_spectrum_from_input(x: torch.Tensor) -> Optional[torch.Tensor]:
#         if not torch.is_tensor(x) or x.dim() != 4 or x.size(1) <= 0:
#             return None
#         h = int(x.size(-2)) // 2
#         w = int(x.size(-1)) // 2
#         return x[:, :, h, w].contiguous()

#     def _prepare_real_spectral_summary(
#         self,
#         x: torch.Tensor,
#         spectra: Optional[torch.Tensor] = None,
#     ) -> Tuple[Optional[torch.Tensor], bool]:
#         """Prepare real-sample spectral summaries for SRGP scoring.

#         Priority is raw spectra supplied by the dataloader/helper.  If absent,
#         the input patch center is used only as a non-physical summary unless the
#         user explicitly marks it as physical.  This avoids fake derivative losses
#         over PCA components.
#         """
#         spectral_summary = None
#         is_physical = False

#         if torch.is_tensor(spectra) and spectra.numel() > 0:
#             s = spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
#             # HSI labels belong to the center pixel.  Do not flatten [B,S,H,W]
#             # raw metadata into [B,S*H*W], because that poisons spectral-shape
#             # descriptors and SRGP residual energy.
#             if s.dim() == 4:
#                 s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#             elif s.dim() == 3:
#                 if s.size(0) == x.size(0) and s.size(2) > 1:
#                     s = s[:, :, s.size(-1) // 2]
#                 else:
#                     s = s.reshape(s.size(0), -1)
#             elif s.dim() == 1:
#                 s = s.reshape(x.size(0), -1) if s.numel() % max(int(x.size(0)), 1) == 0 else s.reshape(1, -1)
#             elif s.dim() > 4:
#                 s = s.reshape(s.size(0), -1)
#             spectral_summary = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#             is_physical = self._inc_spectral_summary_is_physical(True)
#         else:
#             spectral_summary = self._center_spectrum_from_input(x)
#             is_physical = self._inc_spectral_summary_is_physical(None)

#         if spectral_summary is not None and spectral_summary.size(0) != x.size(0):
#             spectral_summary = None
#             is_physical = False
#         return spectral_summary, bool(is_physical)

#     def _forward_real_batch(
#         self,
#         x: torch.Tensor,
#         spectra: Optional[torch.Tensor],
#         *,
#         classifier_mode: str,
#         return_energy: bool = True,
#         return_parts: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         """Forward real HSI samples with SRGP spectral information when valid."""
#         spectral_summary, spec_is_physical = self._prepare_real_spectral_summary(x, spectra)
#         kwargs = dict(classifier_mode=classifier_mode, return_energy=return_energy)
#         if return_parts:
#             kwargs["return_parts"] = True
#         if spectral_summary is not None:
#             kwargs["spectral_summary"] = spectral_summary
#             kwargs["spectral_summary_is_physical"] = spec_is_physical
#         try:
#             out = self.model(x, **kwargs)
#         except TypeError:
#             # Compatibility with older NECILModel signatures.
#             kwargs.pop("spectral_summary", None)
#             kwargs.pop("spectral_summary_is_physical", None)
#             kwargs.pop("return_parts", None)
#             out = self.model(x, **kwargs)
#         if not isinstance(out, dict):
#             raise RuntimeError("Model forward must return a dict in incremental phase.")
#         out["spectral_summary"] = spectral_summary
#         out["spectral_summary_is_physical"] = spec_is_physical
#         return out

#     # ------------------------------------------------------------------
#     # Incremental trainability
#     # ------------------------------------------------------------------
#     def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
#         """Freeze representation and bank; optionally expose bounded energy calibration only."""
#         del old_class_count
#         for _, p in self.model.named_parameters():
#             p.requires_grad = False

#         # Hard-disable stale paths. The trainer orchestrator should also force these off,
#         # but this mixin is defensive because argparse string booleans can be poisonous.
#         for attr, value in (
#             ("use_bicyc_geometry_cycle", False),
#             ("use_geometry_calibrator", False),
#             ("use_incremental_adapter", False),
#         ):
#             if hasattr(self.model, attr):
#                 setattr(self.model, attr, value)
#         if hasattr(self.model, "disable_incremental_adapter"):
#             self.model.disable_incremental_adapter()
#         if hasattr(self.model, "freeze_incremental_adapter"):
#             self.model.freeze_incremental_adapter()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()
#         if hasattr(self.model, "freeze_projection_head"):
#             self.model.freeze_projection_head()
#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()

#         if self._inc_cfg_bool("allow_incremental_projection_training", False):
#             raise RuntimeError(
#                 "Clean incremental trainer forbids projection/backbone plasticity. "
#                 "Use a separate unsafe ablation if you want to move z-space."
#             )
#         if self._inc_cfg_bool("use_bicyc_geometry_cycle", False):
#             raise RuntimeError("Clean incremental trainer forbids BiCyc/geometry-cycle transport.")
#         if self._inc_cfg_bool("use_mssl_loss", False) and self._inc_cfg_float("mssl_inc_weight", 0.0) > 0.0:
#             raise RuntimeError("Clean incremental trainer forbids MSSL as an incremental solver. Use it only as a base ablation.")

#         use_cal = self._inc_cfg_bool("use_energy_calibrator", False)
#         if hasattr(self.model, "enable_energy_calibration"):
#             self.model.enable_energy_calibration(use_cal, calibrator_type=str(getattr(self.args, "energy_calibrator_type", "old_new")))
#         if use_cal and hasattr(self.model, "unfreeze_energy_calibrator"):
#             self.model.unfreeze_energy_calibrator()

#         params = [p for p in self.model.parameters() if p.requires_grad]
#         allowed = (
#             "energy_calibrator", "old_log_scale", "new_log_scale", "old_bias", "new_bias",
#             "log_scale_raw", "bias_raw",
#         )
#         bad = [name for name, p in self.model.named_parameters() if p.requires_grad and not any(k in name for k in allowed)]
#         if bad:
#             raise RuntimeError(f"Invalid incremental trainable parameters in clean path: {bad[:30]}")
#         return params

#     def _incremental_update_mode(self) -> str:
#         """Return the requested incremental update architecture.

#         ``descriptor_only`` keeps the old clean SRGP/RSGI behavior.
#         ``geometry_gated_adapter`` enables G²RPA: a small residual adapter after
#         canonical z, trained with new real samples and old synthetic replay.
#         """
#         mode = str(getattr(self.args, "incremental_update_mode", "scbgr")).lower().strip()
#         aliases = {
#             "g2rpa": "geometry_gated_adapter",
#             "g2-rpa": "geometry_gated_adapter",
#             "gated_adapter": "geometry_gated_adapter",
#             "geometry_adapter": "geometry_gated_adapter",
#             "adapter": "geometry_gated_adapter",
#             "clean": "scbgr",
#             "rsgi": "scbgr",
#             "descriptor_only": "scbgr",
#             "geometry_state_admission": "scbgr",
#             "spectral_risk_boundary": "scbgr",
#         }
#         return aliases.get(mode, mode)

#     def _adapter_mode_enabled(self) -> bool:
#         return self._incremental_update_mode() == "geometry_gated_adapter"

#     def _set_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
#         """Set trainable parameters for the selected incremental architecture.

#         Descriptor-only mode keeps the original strict path.  G²RPA mode freezes
#         backbone/projection/classifier and trains only ``geometry_plastic_adapter``.
#         Old GeometryBank rows are still frozen by the phase entry code.
#         """
#         if not self._adapter_mode_enabled():
#             return self._set_clean_incremental_trainable_params(old_class_count)

#         del old_class_count
#         for _, p in self.model.named_parameters():
#             p.requires_grad = False

#         # Do not enable legacy/stale transport paths.  G²RPA is the only allowed
#         # feature-space plasticity path.
#         for attr, value in (
#             ("use_bicyc_geometry_cycle", False),
#             ("use_geometry_calibrator", False),
#             ("use_incremental_adapter", False),
#         ):
#             if hasattr(self.model, attr):
#                 setattr(self.model, attr, value)
#         if hasattr(self.model, "freeze_projection_head"):
#             self.model.freeze_projection_head()
#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()

#         # The updated NECILModel exposes use_geometry_gated_adapter and
#         # geometry_plastic_adapter.  Fail loudly if the model file was not
#         # updated; otherwise this trainer would silently fall back to no-op
#         # descriptor-only behavior.
#         if not hasattr(self.model, "geometry_plastic_adapter"):
#             raise RuntimeError(
#                 "incremental_update_mode=geometry_gated_adapter requires the updated "
#                 "NECILModel with model.geometry_plastic_adapter."
#             )
#         if hasattr(self.model, "use_geometry_gated_adapter"):
#             self.model.use_geometry_gated_adapter = True
#         if hasattr(self.model, "unfreeze_geometry_plastic_adapter"):
#             self.model.unfreeze_geometry_plastic_adapter()
#         else:
#             for p in self.model.geometry_plastic_adapter.parameters():
#                 p.requires_grad = True

#         params = [p for p in self.model.parameters() if p.requires_grad]
#         bad = [
#             name for name, p in self.model.named_parameters()
#             if p.requires_grad and "geometry_plastic_adapter" not in name
#         ]
#         if bad:
#             raise RuntimeError(f"G²RPA mode allows only geometry_plastic_adapter params, got: {bad[:30]}")
#         if not params:
#             raise RuntimeError("G²RPA mode selected but no adapter parameters are trainable.")
#         return params

#     def _make_g2rpa_loss(self) -> GeometryGatedAdapterLoss:
#         return GeometryGatedAdapterLoss(
#             old_delta_weight=self._inc_cfg_float("adapter_old_delta_weight", 1.0),
#             old_gate_weight=self._inc_cfg_float("adapter_old_gate_weight", 0.75),
#             old_energy_weight=self._inc_cfg_float("adapter_old_energy_weight", 0.25),
#             old_margin_weight=self._inc_cfg_float("adapter_old_margin_weight", 0.25),
#             new_delta_weight=self._inc_cfg_float("adapter_delta_weight", 0.10),
#             new_gate_weight=self._inc_cfg_float("adapter_new_gate_weight", 0.05),
#             new_gate_target=self._inc_cfg_float("adapter_new_gate_target", 0.25),
#             new_gate_max_target=self._inc_cfg_float("adapter_new_gate_max_target", 0.75),
#             margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#             variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
#             reliability_energy_weight=float(getattr(self.args, "reliability_energy_weight", 0.03)),
#             residual_variance_scale=float(getattr(self.args, "residual_variance_scale", 0.75)),
#             normalize_by_dim=bool(getattr(self.args, "energy_normalize_by_dim", True)),
#             use_logdet_energy=bool(getattr(self.args, "use_logdet_energy", True)),
#             logdet_energy_weight=float(getattr(self.args, "logdet_energy_weight", 0.05)),
#             logit_scale=float(getattr(self.args, "loss_scale", 8.0)),
#         )

#     def _compute_g2rpa_adapter_loss(
#         self,
#         *,
#         real_out: Dict[str, torch.Tensor],
#         old_z_base: Optional[torch.Tensor],
#         old_z_adapt: Optional[torch.Tensor],
#         old_y: Optional[torch.Tensor],
#         gate_old: Optional[torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         """Adapter safety/plasticity loss for one incremental batch."""
#         ref = real_out.get("features", None)
#         if not self._adapter_mode_enabled():
#             z = self._zero_like_ref(ref)
#             return {
#                 "total": z, "old_total": z.detach(), "old_delta": z.detach(),
#                 "old_energy": z.detach(), "old_margin": z.detach(), "old_gate": z.detach(),
#                 "old_mean_gate": z.detach(), "old_adapter_acc": z.detach(),
#                 "new_delta": z.detach(), "new_gate": z.detach(), "new_mean_gate": z.detach(),
#             }

#         z_new_adapt = real_out.get("features", None)
#         z_new_base = real_out.get("base_features", real_out.get("pre_adapter_features", None))
#         gate_new = real_out.get("adapter_gate", None)
#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         loss_fn = self._make_g2rpa_loss()
#         return loss_fn(
#             z_old_base=old_z_base,
#             z_old_adapt=old_z_adapt,
#             y_old=old_y,
#             gate_old=gate_old,
#             z_new_base=z_new_base,
#             z_new_adapt=z_new_adapt,
#             gate_new=gate_new,
#             means=bank.get("means", None),
#             bases=bank.get("bases", None),
#             variances=bank.get("variances", None),
#             active_ranks=bank.get("active_ranks", None),
#             reliability=bank.get("reliability", None),
#             sample_counts=bank.get("sample_counts", None),
#         )

#     def _adapt_old_replay_if_needed(
#         self,
#         old_z: Optional[torch.Tensor],
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         """Pass old synthetic replay through adapter in G²RPA mode.

#         Returns (old_z_base, old_z_for_scoring, gate_old).  In descriptor-only
#         mode, old_z_for_scoring is the original replay feature and gate is None.
#         """
#         if old_z is None or not torch.is_tensor(old_z) or old_z.numel() == 0:
#             return old_z, old_z, None
#         old_z_base = old_z.detach()
#         if not self._adapter_mode_enabled():
#             return old_z_base, old_z_base, None
#         if not hasattr(self.model, "adapt_projected_features"):
#             raise RuntimeError("G²RPA mode requires model.adapt_projected_features().")
#         adapted = self.model.adapt_projected_features(old_z_base, force=True, return_delta=True)
#         if isinstance(adapted, dict):
#             old_z_adapt = adapted.get("features", old_z_base)
#             gate_old = adapted.get("gate", None)
#         else:
#             old_z_adapt = adapted
#             gate_old = None
#         if not torch.is_tensor(old_z_adapt) or old_z_adapt.shape != old_z_base.shape:
#             raise RuntimeError("adapt_projected_features must return adapted old replay features with the same shape.")
#         return old_z_base, old_z_adapt, gate_old

#     def _capture_trainable_anchor(self) -> Dict[str, torch.Tensor]:
#         return {name: p.detach().clone() for name, p in self.model.named_parameters() if p.requires_grad}

#     def _trainable_anchor_loss(self, anchor: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
#         weight = float(getattr(self.args, "incremental_weight_anchor", 1e-4))
#         if weight <= 0.0 or not anchor:
#             return self._zero_like_ref(ref)
#         loss = self._zero_like_ref(ref)
#         n = 0
#         for name, p in self.model.named_parameters():
#             if p.requires_grad and name in anchor:
#                 loss = loss + (p - anchor[name].to(p.device)).pow(2).mean()
#                 n += 1
#         return self._zero_like_ref(ref) if n == 0 else weight * loss / float(n)

#     # ------------------------------------------------------------------
#     # Old geometry replay
#     # ------------------------------------------------------------------
#     def _snapshot_old_bank_clean(self, old_class_count: int) -> Dict[str, torch.Tensor]:
#         old_class_count = int(old_class_count)
#         if old_class_count <= 0:
#             return {}
#         if hasattr(self, "_snapshot_old_bank"):
#             snap = self._snapshot_old_bank(old_class_count)
#         elif hasattr(self.model, "get_old_subspace_bank"):
#             snap = self.model.get_old_subspace_bank(old_class_count)
#         else:
#             bank = self.model.get_subspace_bank()
#             snap = {k: v[:old_class_count].detach().clone() for k, v in bank.items() if torch.is_tensor(v) and v.dim() > 0}
#         if hasattr(self, "_canonicalize_bank"):
#             snap = self._canonicalize_bank(snap)

#         for key in ("means", "bases", "variances", "sample_counts"):
#             if key not in snap or not torch.is_tensor(snap[key]) or snap[key].numel() == 0:
#                 raise RuntimeError(f"Old GeometryBank snapshot missing required key '{key}'.")
#         counts = snap["sample_counts"][:old_class_count].to(self.device)
#         if bool((counts <= 0).any().item()):
#             bad = (counts <= 0).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
#             raise RuntimeError(f"Old-bank snapshot has invalid old rows: {bad}")
#         return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in snap.items()}

#     def _select_scbgr_boundary_pairs(
#         self,
#         *,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         """Select risky old/new pairs for SGLAT-HSI boundary replay.

#         Pair format is [old_row, new_local_row].  The boundary sampler consumes
#         new-local indices because ``new_means`` is already sliced to the current
#         phase classes.
#         """
#         old_class_count = int(old_class_count)
#         new_ids = [int(c) for c in new_class_ids]
#         device = self.device
#         empty = torch.empty((0, 2), device=device, dtype=torch.long)
#         if old_class_count <= 0 or not new_ids:
#             return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}

#         try:
#             risk_parts = self._old_new_descriptor_risk_matrix(bank, old_class_count, new_ids)
#         except Exception as exc:
#             if bool(getattr(self, "debug", False)):
#                 print(f"[SGLAT-HSI WARN] could not mine old/new risk pairs: {exc}")
#             return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}

#         risk = risk_parts.get("risk", empty.new_zeros((0, 0)))
#         overlap = risk_parts.get("subspace", torch.zeros_like(risk))
#         if not torch.is_tensor(risk) or risk.numel() == 0:
#             return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}
#         risk = torch.nan_to_num(risk.to(device=device).float(), nan=0.0, posinf=1e6, neginf=0.0)
#         overlap = torch.nan_to_num(overlap.to(device=device).float(), nan=0.0, posinf=1e6, neginf=0.0)

#         risk_thr = self._inc_cfg_float("boundary_replay_risk_threshold", self._inc_cfg_float("descriptor_correction_risk_threshold", 0.35))
#         overlap_thr = self._inc_cfg_float("boundary_replay_overlap_threshold", self._inc_cfg_float("descriptor_correction_overlap_threshold", 0.30))
#         max_pairs = self._inc_cfg_int("boundary_replay_max_pairs", 24)
#         max_pairs = int(max(1, max_pairs))

#         active = (risk >= float(risk_thr)) | (overlap >= float(overlap_thr))
#         coords = active.nonzero(as_tuple=False)
#         if coords.numel() == 0:
#             # Still sample the most dangerous pairs.  If we require thresholds
#             # only, the method can silently turn off exactly when the thresholds
#             # are miscalibrated.
#             flat = risk.flatten()
#             k = min(max_pairs, int(flat.numel()))
#             if k <= 0:
#                 return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}
#             _, idx = torch.topk(flat, k=k, largest=True)
#             coords = torch.stack([idx // risk.size(1), idx % risk.size(1)], dim=1)
#         else:
#             score = risk[coords[:, 0], coords[:, 1]] + overlap[coords[:, 0], coords[:, 1]]
#             k = min(max_pairs, int(score.numel()))
#             _, order = torch.topk(score, k=k, largest=True)
#             coords = coords.index_select(0, order)

#         stats = {
#             "boundary_pair_count": float(coords.size(0)),
#             "boundary_risk_max": float(risk.max().detach().cpu().item()),
#             "boundary_overlap_max": float(overlap.max().detach().cpu().item()),
#         }
#         return coords.long(), stats

#     def _sample_old_anchor_batch(
#         self,
#         old_bank_snapshot: Dict[str, torch.Tensor],
#         old_class_count: int,
#         new_class_ids: Optional[Iterable[int]] = None,
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
#         """Sample old anchors for incremental training.

#         SGLAT-HSI must prefer boundary anchors, not generic old replay.  Generic
#         replay protects old class centers; boundary replay protects the old/new
#         interface where your collapse actually happens.
#         """
#         old_class_count = int(old_class_count)
#         if old_class_count <= 0 or not old_bank_snapshot:
#             self._last_boundary_replay_stats = {"boundary_anchor_count": 0.0, "boundary_pair_count": 0.0}
#             return None, None

#         use_boundary = self._inc_cfg_bool("use_boundary_geometry_replay", True)
#         new_ids = [int(c) for c in (new_class_ids or [])]
#         samples_per_pair = self._inc_cfg_int(
#             "boundary_replay_samples_per_pair",
#             self._inc_cfg_int("gfa_samples_per_class", self._inc_cfg_int("component_replay_per_class", 64)),
#         )
#         samples_per_class = self._inc_cfg_int("gfa_samples_per_class", self._inc_cfg_int("component_replay_per_class", 64))
#         var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))

#         means = old_bank_snapshot["means"][:old_class_count].to(self.device)
#         bases = old_bank_snapshot["bases"][:old_class_count].to(self.device)
#         variances = old_bank_snapshot["variances"][:old_class_count].to(self.device)
#         active = old_bank_snapshot.get("active_ranks", None)
#         rel = old_bank_snapshot.get("reliability", None)
#         counts = old_bank_snapshot.get("sample_counts", None)
#         if torch.is_tensor(active):
#             active = active[:old_class_count].to(self.device)
#         if torch.is_tensor(rel):
#             rel = rel[:old_class_count].to(self.device)
#         if torch.is_tensor(counts):
#             counts = counts[:old_class_count].to(self.device)

#         if use_boundary and new_ids:
#             try:
#                 bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#                 if hasattr(self, "_canonicalize_bank"):
#                     bank = self._canonicalize_bank(bank)
#                 ids_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
#                 new_means = bank["means"].to(self.device).index_select(0, ids_t)
#                 new_bases = bank["bases"].to(self.device).index_select(0, ids_t) if torch.is_tensor(bank.get("bases", None)) else None
#                 pairs, pair_stats = self._select_scbgr_boundary_pairs(
#                     bank=bank,
#                     old_class_count=old_class_count,
#                     new_class_ids=new_ids,
#                 )
#                 x_old, y_old, meta = sample_boundary_geometry_features(
#                     means,
#                     bases,
#                     variances,
#                     new_means=new_means,
#                     new_bases=new_bases,
#                     risk_pairs=pairs,
#                     old_active_ranks=active,
#                     old_reliability=rel,
#                     old_sample_counts=counts,
#                     old_class_ids=list(range(old_class_count)),
#                     samples_per_pair=max(1, int(samples_per_pair)),
#                     variance_floor=var_floor,
#                     parallel_scale=self._inc_cfg_float("boundary_replay_parallel_scale", 0.15),
#                     residual_scale=self._inc_cfg_float("boundary_replay_residual_scale", 0.05),
#                     fallback_samples_per_class=max(1, int(samples_per_class)),
#                     return_metadata=True,
#                 )
#                 self._last_boundary_replay_stats = {
#                     **pair_stats,
#                     "boundary_anchor_count": float(meta.get("boundary_anchor_count", torch.tensor(0.0)).detach().cpu().item()) if isinstance(meta, dict) else float(x_old.size(0)),
#                 }
#                 if torch.is_tensor(x_old) and x_old.numel() > 0:
#                     self._last_risk_replay_counts = {int(c): int(samples_per_pair) for c in range(old_class_count)}
#                     return x_old.to(self.device), y_old.to(self.device).long()
#             except Exception as exc:
#                 if bool(getattr(self, "debug", False)):
#                     print(f"[SGLAT-HSI WARN] boundary replay failed; falling back to generic geometry replay: {exc}")

#         # Compatibility fallback.  This is not the preferred SGLAT-HSI path.
#         x_old, y_old = sample_geometry_features(
#             means,
#             bases,
#             variances,
#             active_ranks=active,
#             reliability=rel,
#             sample_counts=counts,
#             samples_per_class=max(1, int(samples_per_class)),
#             variance_floor=var_floor,
#             parallel_scale=float(getattr(self.args, "gfa_parallel_scale", 1.0)),
#             residual_scale=float(getattr(self.args, "gfa_residual_scale", 0.25)),
#             reliability_gated=self._inc_cfg_bool("gfa_reliability_gated", True),
#             skip_invalid_classes=True,
#         )
#         self._last_boundary_replay_stats = {
#             "boundary_pair_count": 0.0,
#             "boundary_anchor_count": float(x_old.size(0)) if torch.is_tensor(x_old) else 0.0,
#             "boundary_risk_max": 0.0,
#             "boundary_overlap_max": 0.0,
#         }
#         self._last_risk_replay_counts = {int(c): samples_per_class for c in range(old_class_count)}
#         if not torch.is_tensor(x_old) or x_old.numel() == 0:
#             return None, None
#         return x_old.to(self.device), y_old.to(self.device).long()

#     @torch.no_grad()
#     def _apply_reliability_gated_admission_to_new_rows(self, new_class_ids: Iterable[int]) -> None:
#         """Shrink/cap newly bootstrapped descriptors before refinement.

#         This uses the fixed GeometryBank admission rule and never touches frozen
#         old rows. It is the code-level link between the base-prepared geometry
#         field and incremental new-row insertion.
#         """
#         if not self._inc_cfg_bool("reliability_gated_admission", True):
#             return
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None or not hasattr(gb, "update_class_geometry"):
#             return
#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         band_importances = bank.get("band_importances", bank.get("band_importance", None))
#         band_reliability = bank.get("band_reliability", None)
#         feature_reliability = bank.get("feature_reliability", bank.get("reliability", None))
#         for cls in [int(c) for c in new_class_ids]:
#             kwargs = dict(
#                 class_id=cls,
#                 mean=bank["means"][cls].detach(),
#                 basis=bank["bases"][cls].detach(),
#                 eigvals=bank["variances"][cls, :-1].detach(),
#                 res_var=bank["variances"][cls, -1].detach(),
#                 reliability=bank.get("reliability", None)[cls].detach() if torch.is_tensor(bank.get("reliability", None)) else None,
#                 active_rank=bank.get("active_ranks", None)[cls].detach() if torch.is_tensor(bank.get("active_ranks", None)) else None,
#                 sample_count=bank.get("sample_counts", None)[cls].detach() if torch.is_tensor(bank.get("sample_counts", None)) else None,
#                 feature_reliability=feature_reliability[cls].detach() if torch.is_tensor(feature_reliability) and feature_reliability.numel() > cls else None,
#                 band_importance=band_importances[cls].detach() if torch.is_tensor(band_importances) and band_importances.dim() == 2 and band_importances.size(0) > cls else None,
#                 band_reliability=band_reliability[cls].detach() if torch.is_tensor(band_reliability) and band_reliability.numel() > cls else None,
#                 allow_frozen_update=False,
#                 reliability_gated_admission=True,
#                 admission_min_gate=self._inc_cfg_float("admission_min_gate", 0.35),
#                 admission_shrink_floor=self._inc_cfg_float("admission_shrink_floor", 0.15),
#                 admission_low_rank_cap=self._inc_cfg_int("admission_low_rank_cap", 2),
#             )
#             gb.update_class_geometry(**kwargs)
#         if hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)


#     def _safe_update_new_descriptor_row(
#         self,
#         cls: int,
#         *,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         variances: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#     ) -> None:
#         """Commit one corrected new descriptor row while preserving SRGP fields."""
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             raise AttributeError("model.geometry_bank is required for descriptor correction.")
#         cls = int(cls)
#         rel = bank.get("reliability", None)
#         feat_rel = bank.get("feature_reliability", rel)
#         active = bank.get("active_ranks", None)
#         counts = bank.get("sample_counts", None)
#         band = bank.get("band_importances", bank.get("band_importance", None))
#         band_rel = bank.get("band_reliability", None)
#         spectral_kwargs: Dict[str, torch.Tensor] = {}
#         for key in (
#             "spectral_curve_means",
#             "spectral_curve_vars",
#             "spectral_curve_d1",
#             "spectral_curve_d2",
#             "spectral_shape_reliability",
#         ):
#             value = bank.get(key, None)
#             if torch.is_tensor(value) and value.size(0) > cls:
#                 spectral_kwargs[key] = value[cls].detach()

#         if hasattr(gb, "apply_refined_feature_rows"):
#             kwargs = dict(
#                 class_ids=[cls],
#                 means=mean.detach().unsqueeze(0),
#                 bases=basis.detach().unsqueeze(0),
#                 eigvals=variances[:-1].detach().unsqueeze(0),
#                 res_vars=variances[-1].detach().view(1),
#                 reliability=rel[cls].detach().view(1) if torch.is_tensor(rel) and rel.numel() > cls else None,
#                 feature_reliability=feat_rel[cls].detach().view(1) if torch.is_tensor(feat_rel) and feat_rel.numel() > cls else None,
#                 active_ranks=active[cls].detach().view(1) if torch.is_tensor(active) and active.numel() > cls else None,
#                 allow_frozen_update=False,
#             )
#             # New GeometryBank versions accept SRGP spectral rows.  Older ones do not.
#             try:
#                 kwargs.update({k: v.unsqueeze(0) if v.dim() > 0 else v.view(1) for k, v in spectral_kwargs.items()})
#                 gb.apply_refined_feature_rows(**kwargs)
#             except TypeError:
#                 for k in list(spectral_kwargs.keys()):
#                     kwargs.pop(k, None)
#                 gb.apply_refined_feature_rows(**kwargs)
#             return

#         if hasattr(gb, "update_class_geometry"):
#             kwargs = dict(
#                 class_id=cls,
#                 mean=mean.detach(),
#                 basis=basis.detach(),
#                 eigvals=variances[:-1].detach(),
#                 res_var=variances[-1].detach(),
#                 reliability=rel[cls].detach() if torch.is_tensor(rel) and rel.numel() > cls else None,
#                 active_rank=active[cls].detach() if torch.is_tensor(active) and active.numel() > cls else None,
#                 sample_count=counts[cls].detach() if torch.is_tensor(counts) and counts.numel() > cls else None,
#                 feature_reliability=feat_rel[cls].detach() if torch.is_tensor(feat_rel) and feat_rel.numel() > cls else None,
#                 band_importance=band[cls].detach() if torch.is_tensor(band) and band.dim() == 2 and band.size(0) > cls else None,
#                 band_reliability=band_rel[cls].detach() if torch.is_tensor(band_rel) and band_rel.numel() > cls else None,
#                 allow_frozen_update=False,
#             )
#             try:
#                 # Map plural bank names to update_class_geometry names when supported.
#                 if "spectral_curve_means" in spectral_kwargs:
#                     kwargs["spectral_curve_mean"] = spectral_kwargs["spectral_curve_means"]
#                 if "spectral_curve_vars" in spectral_kwargs:
#                     kwargs["spectral_curve_var"] = spectral_kwargs["spectral_curve_vars"]
#                 if "spectral_curve_d1" in spectral_kwargs:
#                     kwargs["spectral_curve_d1"] = spectral_kwargs["spectral_curve_d1"]
#                 if "spectral_curve_d2" in spectral_kwargs:
#                     kwargs["spectral_curve_d2"] = spectral_kwargs["spectral_curve_d2"]
#                 if "spectral_shape_reliability" in spectral_kwargs:
#                     kwargs["spectral_shape_reliability"] = spectral_kwargs["spectral_shape_reliability"]
#                 gb.update_class_geometry(**kwargs)
#             except TypeError:
#                 for key in (
#                     "spectral_curve_mean",
#                     "spectral_curve_var",
#                     "spectral_curve_d1",
#                     "spectral_curve_d2",
#                     "spectral_shape_reliability",
#                 ):
#                     kwargs.pop(key, None)
#                 gb.update_class_geometry(**kwargs)
#             return
#         raise AttributeError("GeometryBank must expose apply_refined_feature_rows() or update_class_geometry().")

#     @staticmethod
#     def _basis_overlap_matrix(old_bases: torch.Tensor, new_bases: torch.Tensor) -> torch.Tensor:
#         if old_bases.numel() == 0 or new_bases.numel() == 0:
#             return old_bases.new_zeros((old_bases.size(0), new_bases.size(0)))
#         prod = torch.einsum("odr,ndr->onr", old_bases, new_bases)
#         # The einsum above only compares matching rank indices.  Use full matrix overlap instead.
#         vals = []
#         for i in range(old_bases.size(0)):
#             row = []
#             for j in range(new_bases.size(0)):
#                 m = old_bases[i].transpose(0, 1).matmul(new_bases[j])
#                 denom = float(max(1, min(old_bases.size(-1), new_bases.size(-1))))
#                 row.append(m.pow(2).sum() / denom)
#             vals.append(torch.stack(row))
#         return torch.stack(vals, dim=0).clamp_min(0.0)

#     def _old_new_descriptor_risk_matrix(
#         self,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#     ) -> Dict[str, torch.Tensor]:
#         """Compute old/new conflict using the same SRGP descriptors used by diagnostics."""
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         old_class_count = int(old_class_count)
#         new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=self.device, dtype=torch.long)
#         if old_class_count <= 0 or new_ids.numel() == 0:
#             empty = torch.zeros((0, 0), device=self.device)
#             return {"risk": empty, "subspace": empty, "center": empty, "band": empty, "spectral": empty}

#         means = bank["means"].to(self.device)
#         bases = bank["bases"].to(self.device)
#         old_means = means[:old_class_count]
#         new_means = means.index_select(0, new_ids)
#         old_bases = bases[:old_class_count]
#         new_bases = bases.index_select(0, new_ids)

#         sub = self._basis_overlap_matrix(old_bases, new_bases)
#         dist = torch.cdist(old_means, new_means, p=2)
#         center_margin = self._inc_cfg_float("risk_center_margin", 1.0)
#         center = torch.exp(-dist / max(center_margin, 1e-6))

#         band = torch.zeros_like(sub)
#         band_all = bank.get("band_importances", bank.get("band_importance", None))
#         if torch.is_tensor(band_all) and band_all.dim() == 2 and band_all.size(0) > int(new_ids.max().item()):
#             old_band = F.normalize(band_all[:old_class_count].to(self.device).float(), p=2, dim=1)
#             new_band = F.normalize(band_all.index_select(0, new_ids).to(self.device).float(), p=2, dim=1)
#             band = old_band.matmul(new_band.t()).clamp(0.0, 1.0)

#         spectral = torch.zeros_like(sub)
#         # Prefer bank method if available; otherwise use derivative rows directly.
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "pairwise_spectral_shape_similarity"):
#             try:
#                 spec_all = gb.pairwise_spectral_shape_similarity().to(self.device)
#                 spectral = spec_all[:old_class_count].index_select(1, new_ids).clamp(0.0, 1.0)
#             except Exception:
#                 spectral = torch.zeros_like(sub)
#         elif torch.is_tensor(bank.get("spectral_curve_d1", None)):
#             d1 = bank["spectral_curve_d1"].to(self.device).float()
#             if d1.dim() == 2 and d1.size(0) > int(new_ids.max().item()):
#                 old_d1 = F.normalize(d1[:old_class_count], p=2, dim=1)
#                 new_d1 = F.normalize(d1.index_select(0, new_ids), p=2, dim=1)
#                 spectral = old_d1.matmul(new_d1.t()).clamp(0.0, 1.0)

#         risk = (
#             self._inc_cfg_float("risk_subspace_weight", 1.0) * sub
#             + self._inc_cfg_float("risk_center_weight", 0.50) * center
#             + self._inc_cfg_float("risk_band_weight", 0.15) * band
#             + self._inc_cfg_float("risk_spectral_shape_weight", 0.25) * spectral
#         )
#         return {"risk": risk, "subspace": sub, "center": center, "band": band, "spectral": spectral, "dist": dist}

#     @torch.no_grad()
#     def _apply_risk_aware_descriptor_correction_to_new_rows(
#         self,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#     ) -> Dict[str, float]:
#         """Actively correct high-risk new descriptors before refinement.

#         This is the missing RSGI step: if a new class uses old-class tangent
#         directions, remove those directions from the new basis and push the new
#         center away from the most dangerous old centers.  Old rows are never
#         modified.
#         """
#         if not self._inc_cfg_bool("risk_aware_descriptor_correction", True):
#             return {"active": 0.0, "max_risk_before": 0.0, "max_overlap_before": 0.0}
#         old_class_count = int(old_class_count)
#         new_ids = [int(c) for c in new_class_ids]
#         if old_class_count <= 0 or not new_ids:
#             return {"active": 0.0, "max_risk_before": 0.0, "max_overlap_before": 0.0}

#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         risk_parts = self._old_new_descriptor_risk_matrix(bank, old_class_count, new_ids)
#         risk = risk_parts["risk"]
#         sub = risk_parts["subspace"]
#         if risk.numel() == 0:
#             return {"active": 0.0, "max_risk_before": 0.0, "max_overlap_before": 0.0}

#         risk_thr = self._inc_cfg_float("descriptor_correction_risk_threshold", 0.75)
#         overlap_thr = self._inc_cfg_float("descriptor_correction_overlap_threshold", 0.60)
#         basis_strength = self._inc_cfg_float("descriptor_correction_basis_strength", 0.85)
#         mean_push = self._inc_cfg_float("descriptor_correction_mean_push", 0.20)
#         var_shrink = self._inc_cfg_float("descriptor_correction_var_shrink", 0.15)
#         topk = max(1, self._inc_cfg_int("descriptor_correction_topk_old", 3))
#         var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))

#         means = bank["means"].to(self.device)
#         bases = bank["bases"].to(self.device)
#         variances = bank["variances"].to(self.device).clamp_min(var_floor)
#         corrected = 0
#         max_risk_before = float(risk.max().detach().cpu().item())
#         max_overlap_before = float(sub.max().detach().cpu().item())
#         max_risk_after = max_risk_before
#         max_overlap_after = max_overlap_before

#         for j, cls in enumerate(new_ids):
#             col_risk = risk[:, j]
#             col_sub = sub[:, j]
#             do_correct = bool(((col_risk > risk_thr) | (col_sub > overlap_thr)).any().item())
#             if not do_correct:
#                 continue
#             corrected += 1
#             k = min(topk, int(col_risk.numel()))
#             vals, old_idx = torch.topk(col_risk, k=k, largest=True)
#             weights = vals.clamp_min(0.0)
#             if float(weights.sum().detach().item()) <= 1e-12:
#                 weights = torch.ones_like(weights)
#             weights = weights / weights.sum().clamp_min(1e-12)

#             mu = means[cls].detach().clone()
#             U = bases[cls].detach().clone()
#             var = variances[cls].detach().clone()
#             D = int(U.size(0))
#             P = torch.zeros((D, D), device=self.device, dtype=U.dtype)
#             push = torch.zeros((D,), device=self.device, dtype=mu.dtype)
#             for w, oi in zip(weights, old_idx):
#                 Uo = bases[int(oi)].to(self.device)
#                 P = P + w * Uo.matmul(Uo.transpose(0, 1))
#                 direction = mu - means[int(oi)].to(self.device)
#                 direction = direction / direction.norm().clamp_min(1e-12)
#                 push = push + w * direction

#             gate = (float(col_risk.max().detach().item()) - risk_thr) / max(1e-6, 1.5 - risk_thr)
#             gate = float(max(0.0, min(1.0, gate)))
#             U_corr = U - float(basis_strength) * gate * P.matmul(U)
#             q, _ = torch.linalg.qr(U_corr, mode="reduced")
#             q = q[:, : U.size(1)]
#             # Sign-stabilize relative to original inserted basis.
#             sign = torch.where((q * U).sum(dim=0, keepdim=True) < 0, -torch.ones(1, U.size(1), device=q.device), torch.ones(1, U.size(1), device=q.device))
#             q = q * sign
#             mu_corr = mu + float(mean_push) * gate * push / push.norm().clamp_min(1e-12)
#             var_corr = var.clone()
#             var_corr[:-1] = (var_corr[:-1] * (1.0 - float(var_shrink) * gate)).clamp_min(var_floor)
#             var_corr[-1] = (var_corr[-1] * (1.0 - 0.5 * float(var_shrink) * gate)).clamp_min(var_floor)
#             self._safe_update_new_descriptor_row(cls, mean=mu_corr, basis=q, variances=var_corr, bank=bank)

#         if corrected > 0:
#             bank_after = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#             if hasattr(self, "_canonicalize_bank"):
#                 bank_after = self._canonicalize_bank(bank_after)
#             after = self._old_new_descriptor_risk_matrix(bank_after, old_class_count, new_ids)
#             if after["risk"].numel() > 0:
#                 max_risk_after = float(after["risk"].max().detach().cpu().item())
#                 max_overlap_after = float(after["subspace"].max().detach().cpu().item())
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)
#         stats = {
#             "active": float(corrected),
#             "max_risk_before": max_risk_before,
#             "max_overlap_before": max_overlap_before,
#             "max_risk_after": max_risk_after,
#             "max_overlap_after": max_overlap_after,
#         }
#         self._last_descriptor_correction_stats = stats
#         if corrected > 0:
#             print(
#                 "[RSGI Descriptor Correction] "
#                 f"active={corrected} | risk {max_risk_before:.4f}->{max_risk_after:.4f} | "
#                 f"overlap {max_overlap_before:.4f}->{max_overlap_after:.4f}"
#             )
#         return stats

#     @torch.no_grad()
#     def _incremental_risk_report(self, old_class_count: int, new_class_ids: Iterable[int]) -> Dict[str, float]:
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             return {}
#         report: Dict[str, float] = {}
#         try:
#             if hasattr(gb, "geometry_conflict_matrix"):
#                 risk_kwargs = dict(
#                     center_margin=self._inc_cfg_float("risk_center_margin", 1.0),
#                     subspace_weight=self._inc_cfg_float("risk_subspace_weight", 1.0),
#                     band_weight=self._inc_cfg_float("risk_band_weight", 0.15),
#                     spectral_shape_weight=self._inc_cfg_float("risk_spectral_shape_weight", 0.25),
#                     chart_weight=self._inc_cfg_float("risk_chart_weight", 0.0),
#                     reliability_weighted=self._inc_cfg_bool("risk_replay_reliability_weighted", True),
#                 )
#                 try:
#                     risk = gb.geometry_conflict_matrix(**risk_kwargs)
#                 except TypeError:
#                     risk_kwargs.pop("spectral_shape_weight", None)
#                     risk_kwargs.pop("chart_weight", None)
#                     risk = gb.geometry_conflict_matrix(**risk_kwargs)
#                 new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=risk.device, dtype=torch.long)
#                 if risk.numel() > 0 and old_class_count > 0 and new_ids.numel() > 0:
#                     old_new = risk[:int(old_class_count)].index_select(1, new_ids)
#                     report["risk_old_new_mean"] = float(old_new.mean().detach().cpu().item())
#                     report["risk_old_new_max"] = float(old_new.max().detach().cpu().item())
#             if hasattr(gb, "pairwise_subspace_overlap"):
#                 sub = gb.pairwise_subspace_overlap()
#                 new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=sub.device, dtype=torch.long)
#                 if sub.numel() > 0 and old_class_count > 0 and new_ids.numel() > 0:
#                     vals = sub[:int(old_class_count)].index_select(1, new_ids)
#                     report["old_new_subspace_overlap_max"] = float(vals.max().detach().cpu().item())
#                     report["old_new_subspace_overlap_mean"] = float(vals.mean().detach().cpu().item())
#             if hasattr(gb, "pairwise_band_similarity"):
#                 band = gb.pairwise_band_similarity()
#                 new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=band.device, dtype=torch.long)
#                 if band.numel() > 0 and old_class_count > 0 and new_ids.numel() > 0:
#                     vals = band[:int(old_class_count)].index_select(1, new_ids)
#                     report["old_new_band_similarity_max"] = float(vals.max().detach().cpu().item())
#                     report["old_new_band_similarity_mean"] = float(vals.mean().detach().cpu().item())
#         except Exception as exc:
#             if bool(getattr(self, "debug", False)):
#                 print(f"[WARN] incremental risk report failed: {exc}")
#         return report

#     # ------------------------------------------------------------------
#     # Descriptor-only refinement
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _extract_current_phase_feature_cache(self, class_ids: Iterable[int], split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
#         feats: List[torch.Tensor] = []
#         labs: List[torch.Tensor] = []
#         for cls in [int(c) for c in class_ids]:
#             if not hasattr(self, "_extract_backbone_outputs_for_class"):
#                 raise AttributeError("TrainerHelper._extract_backbone_outputs_for_class() is required for descriptor refinement.")
#             out = self._extract_backbone_outputs_for_class(cls, split=split)
#             if not isinstance(out, dict) or "features" not in out:
#                 raise RuntimeError("_extract_backbone_outputs_for_class() must return {'features': tensor}.")
#             z = out["features"].detach().to(self.device)
#             if z.dim() != 2 or z.numel() == 0:
#                 raise RuntimeError(f"Invalid projected features for class {cls}: {tuple(z.shape)}")
#             if not torch.isfinite(z).all():
#                 raise RuntimeError(f"Non-finite projected features for class {cls}.")
#             feats.append(z)
#             labs.append(torch.full((z.size(0),), cls, device=self.device, dtype=torch.long))
#         if not feats:
#             raise RuntimeError("No current-phase features available for descriptor refinement.")
#         return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

#     def _orthonormalize_descriptor_bases(self, raw_basis: torch.Tensor, reference_basis: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if raw_basis.dim() != 3:
#             raise RuntimeError(f"raw_basis must be [K,D,R], got {tuple(raw_basis.shape)}")
#         bases: List[torch.Tensor] = []
#         R = int(raw_basis.size(-1))
#         for k in range(raw_basis.size(0)):
#             q, _ = torch.linalg.qr(raw_basis[k], mode="reduced")
#             q = q[:, :R]
#             # Stabilize sign relative to the inserted descriptor to reduce artificial trust loss.
#             if torch.is_tensor(reference_basis) and reference_basis.shape == raw_basis.shape:
#                 dots = (q * reference_basis[k].to(device=q.device, dtype=q.dtype)).sum(dim=0, keepdim=True)
#                 signs = torch.where(dots < 0, torch.full_like(dots, -1.0), torch.ones_like(dots))
#                 q = q * signs
#             bases.append(q)
#         return torch.stack(bases, dim=0)

#     def _make_refinement_bank(
#         self,
#         base_bank: Dict[str, torch.Tensor],
#         class_ids: List[int],
#         means_new: torch.Tensor,
#         bases_new: torch.Tensor,
#         variances_new: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         bank = {}
#         for key, value in base_bank.items():
#             if torch.is_tensor(value):
#                 bank[key] = value.detach().clone().to(self.device)
#             else:
#                 bank[key] = value
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         ids = torch.as_tensor(class_ids, device=self.device, dtype=torch.long)
#         bank["means"][ids] = means_new
#         bank["bases"][ids] = bases_new
#         bank["variances"][ids] = variances_new
#         bank["eigvals"] = bank["variances"][:, :-1]
#         bank["res_vars"] = bank["variances"][:, -1]
#         bank["resvars"] = bank["res_vars"]
#         return bank

#     def _score_features_with_bank(
#         self,
#         features: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#         *,
#         return_parts: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.model, "classifier"):
#             raise AttributeError("Model must expose GeometryEnergyClassifier as model.classifier.")
#         out = self.model.classifier(
#             features,
#             geometry_bank=bank,
#             mode="geometry_only",
#             old_class_count=int(old_class_count),
#             return_energy=True,
#             return_parts=return_parts,
#         )
#         if not isinstance(out, dict) or "logits" not in out or "energy" not in out:
#             raise RuntimeError("GeometryEnergyClassifier must return dict with logits and energy for descriptor refinement.")
#         return out

#     @torch.no_grad()
#     def _commit_refined_descriptors(
#         self,
#         class_ids: List[int],
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         init_bank: Dict[str, torch.Tensor],
#     ) -> None:
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             raise AttributeError("model.geometry_bank is required to commit refined descriptors.")
#         class_ids = [int(c) for c in class_ids]
#         ids = torch.as_tensor(class_ids, device=self.device, dtype=torch.long)
#         reliability = init_bank.get("reliability", None)
#         feature_reliability = init_bank.get("feature_reliability", reliability)
#         active_ranks = init_bank.get("active_ranks", None)

#         if hasattr(gb, "apply_refined_feature_rows"):
#             gb.apply_refined_feature_rows(
#                 class_ids,
#                 means=means.detach(),
#                 bases=bases.detach(),
#                 eigvals=variances[:, :-1].detach(),
#                 res_vars=variances[:, -1].detach(),
#                 reliability=reliability[ids].detach() if torch.is_tensor(reliability) else None,
#                 feature_reliability=feature_reliability[ids].detach() if torch.is_tensor(feature_reliability) else None,
#                 active_ranks=active_ranks[ids].detach() if torch.is_tensor(active_ranks) else None,
#                 allow_frozen_update=False,
#             )
#             return

#         # Fallback for older GeometryBank versions.
#         band_importances = init_bank.get("band_importances", init_bank.get("band_importance", None))
#         sample_counts = init_bank.get("sample_counts", None)
#         for i, cls in enumerate(class_ids):
#             kwargs = dict(
#                 class_id=cls,
#                 mean=means[i].detach(),
#                 basis=bases[i].detach(),
#                 eigvals=variances[i, :-1].detach(),
#                 res_var=variances[i, -1].detach(),
#                 reliability=reliability[cls].detach() if torch.is_tensor(reliability) and reliability.numel() > cls else None,
#                 active_rank=active_ranks[cls].detach() if torch.is_tensor(active_ranks) and active_ranks.numel() > cls else None,
#                 sample_count=sample_counts[cls].detach() if torch.is_tensor(sample_counts) and sample_counts.numel() > cls else None,
#                 feature_reliability=feature_reliability[cls].detach() if torch.is_tensor(feature_reliability) and feature_reliability.numel() > cls else None,
#                 band_importance=band_importances[cls].detach() if torch.is_tensor(band_importances) and band_importances.dim() > 1 and band_importances.size(0) > cls else None,
#                 allow_frozen_update=False,
#             )
#             if hasattr(gb, "update_class_geometry"):
#                 gb.update_class_geometry(**kwargs)
#             elif hasattr(gb, "update_class"):
#                 kwargs["cls_id"] = kwargs.pop("class_id")
#                 gb.update_class(**kwargs)
#             else:
#                 raise AttributeError("GeometryBank must expose update_class_geometry(), update_class(), or apply_refined_feature_rows().")


#     def _risk_weighted_subspace_separation_loss(
#         self,
#         *,
#         old_means: torch.Tensor,
#         old_bases: torch.Tensor,
#         new_means: torch.Tensor,
#         new_bases: torch.Tensor,
#         old_reliability: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Differentiable high-risk old/new subspace separation.

#         This directly attacks failures such as Gray roof -> Bare soil where a new
#         class reuses old tangent directions.  Risk is detached as a weighting
#         signal; the overlap term remains differentiable w.r.t. new bases/means.
#         """
#         if old_bases.numel() == 0 or new_bases.numel() == 0:
#             z = self._zero_like_ref(new_bases)
#             return z, z
#         rows = []
#         for i in range(old_bases.size(0)):
#             vals = []
#             for j in range(new_bases.size(0)):
#                 m = old_bases[i].transpose(0, 1).matmul(new_bases[j])
#                 denom = float(max(1, min(old_bases.size(-1), new_bases.size(-1))))
#                 vals.append(m.pow(2).sum() / denom)
#             rows.append(torch.stack(vals))
#         overlap = torch.stack(rows, dim=0).clamp_min(0.0)
#         center = torch.exp(-torch.cdist(old_means, new_means, p=2) / max(self._inc_cfg_float("risk_center_margin", 1.0), 1e-6))
#         risk = (self._inc_cfg_float("risk_subspace_weight", 1.0) * overlap.detach()
#                 + self._inc_cfg_float("risk_center_weight", 0.50) * center.detach())
#         if torch.is_tensor(old_reliability) and old_reliability.numel() == old_bases.size(0):
#             risk = risk * old_reliability.view(-1, 1).to(risk.device).clamp(0.05, 1.0)
#         target = self._inc_cfg_float("risk_sep_overlap_target", self._inc_cfg_float("descriptor_overlap_target", 0.35))
#         active = (risk > self._inc_cfg_float("risk_sep_active_threshold", 0.50)).float()
#         weights = (risk * active).detach()
#         if float(weights.sum().detach().item()) <= 1e-12:
#             return self._zero_like_ref(new_bases), active.sum()
#         loss = (weights * F.relu(overlap - target).pow(2)).sum() / weights.sum().clamp_min(1e-12)
#         return loss, active.sum()

#     def _empty_descriptor_stats(self) -> Dict[str, float]:
#         return {
#             "loss": 0.0,
#             "ce": 0.0,
#             "margin": 0.0,
#             "invasion": 0.0,
#             "trust": 0.0,
#             "subspace_collision": 0.0,
#             "center_collision": 0.0,
#             "volume": 0.0,
#             "risk_sep": 0.0,
#             "risk_active_pairs": 0.0,
#             "risk_old_new_max": 0.0,
#             "admission": 0.0,
#             "admission_safe": 0.0,
#             "admission_new_violation_rate": 0.0,
#             "admission_old_boundary_violation_rate": 0.0,
#             "boundary_anchor_count": 0.0,
#             "boundary_pair_count": 0.0,
#             "mean_shift": 0.0,
#             "basis_shift": 0.0,
#             "logvar_shift": 0.0,
#             "anchor_count": 0.0,
#             "steps": 0.0,
#         }

#     def _prepare_descriptor_refinement_state(
#         self,
#         *,
#         new_class_ids: Iterable[int],
#         seen_classes: Iterable[int],
#     ) -> Optional[Dict[str, torch.Tensor | List[int] | optim.Optimizer]]:
#         """Create persistent descriptor parameters for epoch-driven incremental training.

#         The inserted new rows define the trust-region origin. The same descriptor
#         parameters are optimized across epochs, so ``epochs_inc`` now corresponds
#         to real incremental optimization epochs rather than a one-shot pre-loop
#         refinement block.
#         """
#         if not self._inc_cfg_bool("refine_new_descriptors", True):
#             return None

#         new_class_ids = [int(c) for c in new_class_ids]
#         seen_classes = [int(c) for c in seen_classes]
#         if not new_class_ids:
#             return None

#         bank0 = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank0 = self._canonicalize_bank(bank0)
#         if hasattr(self, "_validate_bank_has_classes"):
#             self._validate_bank_has_classes(bank0, seen_classes)

#         ids = torch.as_tensor(new_class_ids, device=self.device, dtype=torch.long)
#         means0 = bank0["means"].detach().to(self.device)
#         bases0 = bank0["bases"].detach().to(self.device)
#         vars0 = bank0["variances"].detach().to(self.device).clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4)))
#         counts0 = bank0["sample_counts"].detach().to(self.device)
#         if bool((counts0[ids] <= 0).any().item()):
#             bad = ids[counts0[ids] <= 0].detach().cpu().tolist()
#             raise RuntimeError(f"Cannot refine unbuilt new GeometryBank rows: {bad}")

#         z_new, y_new = self._extract_current_phase_feature_cache(new_class_ids, split="train")
#         z_new = z_new.detach()
#         y_new = y_new.detach()

#         init_means = means0[ids].detach().clone()
#         init_bases = bases0[ids].detach().clone()
#         init_vars = vars0[ids].detach().clone()
#         init_log_vars = init_vars.log()

#         mu = torch.nn.Parameter(init_means.clone())
#         raw_basis = torch.nn.Parameter(init_bases.clone())
#         log_vars = torch.nn.Parameter(init_log_vars.clone())
#         optimizer = optim.Adam(
#             [mu, raw_basis, log_vars],
#             lr=self._inc_cfg_float("descriptor_refine_lr", 1e-3),
#             weight_decay=0.0,
#         )

#         return {
#             "new_class_ids": new_class_ids,
#             "seen_classes": seen_classes,
#             "bank0": bank0,
#             "valid_mask": counts0 > 0,
#             "z_new": z_new,
#             "y_new": y_new,
#             "mu": mu,
#             "raw_basis": raw_basis,
#             "log_vars": log_vars,
#             "init_means": init_means,
#             "init_bases": init_bases,
#             "init_log_vars": init_log_vars,
#             "optimizer": optimizer,
#         }

#     def _descriptor_refinement_epoch(
#         self,
#         *,
#         state: Optional[Dict[str, object]],
#         phase: int,
#         old_class_count: int,
#         old_bank_snapshot: Dict[str, torch.Tensor],
#         steps_per_epoch: int,
#     ) -> Dict[str, float]:
#         """Run one real incremental epoch over descriptor parameters only."""
#         if state is None:
#             return self._empty_descriptor_stats()

#         steps = int(max(steps_per_epoch, 0))
#         if steps <= 0:
#             return self._empty_descriptor_stats()

#         old_ids = list(range(int(old_class_count)))
#         old_integrity = self._old_bank_integrity_snapshot(old_ids) if hasattr(self, "_old_bank_integrity_snapshot") else None

#         new_class_ids = [int(c) for c in state["new_class_ids"]]  # type: ignore[index]
#         seen_classes = [int(c) for c in state["seen_classes"]]  # type: ignore[index]
#         bank0 = state["bank0"]  # type: ignore[assignment]
#         valid_mask = state["valid_mask"]  # type: ignore[assignment]
#         z_new = state["z_new"]  # type: ignore[assignment]
#         y_new = state["y_new"]  # type: ignore[assignment]
#         mu = state["mu"]  # type: ignore[assignment]
#         raw_basis = state["raw_basis"]  # type: ignore[assignment]
#         log_vars = state["log_vars"]  # type: ignore[assignment]
#         init_means = state["init_means"]  # type: ignore[assignment]
#         init_bases = state["init_bases"]  # type: ignore[assignment]
#         init_log_vars = state["init_log_vars"]  # type: ignore[assignment]
#         optimizer = state["optimizer"]  # type: ignore[assignment]

#         assert isinstance(bank0, dict)
#         assert torch.is_tensor(valid_mask)
#         assert torch.is_tensor(z_new) and torch.is_tensor(y_new)
#         assert isinstance(mu, torch.nn.Parameter)
#         assert isinstance(raw_basis, torch.nn.Parameter)
#         assert isinstance(log_vars, torch.nn.Parameter)
#         assert torch.is_tensor(init_means) and torch.is_tensor(init_bases) and torch.is_tensor(init_log_vars)
#         assert isinstance(optimizer, optim.Optimizer)

#         trust_w = self._inc_cfg_float("descriptor_trust_weight", 1.0)
#         margin_w = self._inc_cfg_float("geometry_energy_margin_weight", float(getattr(self.args, "geometry_energy_margin_weight", 0.25)))
#         invasion_w = self._inc_cfg_float("old_new_invasion_weight", float(getattr(self.args, "old_new_invasion_weight", 0.35)))
#         subspace_w = self._inc_cfg_float("descriptor_subspace_collision_weight", 0.20)
#         center_w = self._inc_cfg_float("descriptor_center_collision_weight", 0.05)
#         volume_w = self._inc_cfg_float("descriptor_volume_control_weight", 0.03)
#         risk_sep_w = self._inc_cfg_float("risk_sep_weight", 0.30)
#         max_mean_shift = self._inc_cfg_float("descriptor_refine_max_mean_shift", 0.35)
#         max_logvar_shift = self._inc_cfg_float("descriptor_refine_max_logvar_shift", 0.75)
#         var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))

#         stat = self._empty_descriptor_stats()
#         for _ in range(steps):
#             optimizer.zero_grad(set_to_none=True)

#             bases_new = self._orthonormalize_descriptor_bases(raw_basis, init_bases)
#             vars_new = log_vars.exp().clamp_min(var_floor)
#             tmp_bank = self._make_refinement_bank(bank0, new_class_ids, mu, bases_new, vars_new)

#             old_z, old_y = self._sample_old_anchor_batch(old_bank_snapshot, old_class_count, new_class_ids)
#             if old_z is not None and old_y is not None and old_z.numel() > 0:
#                 z_joint = torch.cat([z_new, old_z.detach()], dim=0)
#                 y_joint = torch.cat([y_new, old_y.detach()], dim=0)
#                 anchor_count = float(old_y.numel())
#             else:
#                 z_joint = z_new
#                 y_joint = y_new
#                 anchor_count = 0.0

#             out = self._score_features_with_bank(z_joint, tmp_bank, old_class_count, return_parts=False)
#             energy = out["energy"]

#             role_new = torch.zeros((z_new.size(0),), device=self.device, dtype=torch.long)
#             if old_z is not None and old_y is not None and old_z.numel() > 0:
#                 role_old = torch.ones((old_y.numel(),), device=self.device, dtype=torch.long)
#                 batch_role = torch.cat([role_new, role_old], dim=0)
#             else:
#                 batch_role = role_new

#             old_bases = bank0["bases"][:int(old_class_count)].to(self.device)
#             old_vars = bank0["variances"][:int(old_class_count)].to(self.device)
#             old_active = bank0.get("active_ranks", None)
#             new_active = bank0.get("active_ranks", None)
#             old_reliability = bank0.get("reliability", None)
#             if torch.is_tensor(old_active):
#                 old_active = old_active[:int(old_class_count)].to(self.device)
#             if torch.is_tensor(new_active):
#                 ids_t = torch.as_tensor(new_class_ids, device=self.device, dtype=torch.long)
#                 new_active = new_active.index_select(0, ids_t).to(self.device)
#             if torch.is_tensor(old_reliability):
#                 old_reliability = old_reliability[:int(old_class_count)].to(self.device)

#             sample_counts = tmp_bank.get("sample_counts", valid_mask)
#             loss_out = unified_spectral_geometry_loss(
#                 phase="incremental",
#                 energy=energy,
#                 labels=y_joint,
#                 sample_counts=sample_counts,
#                 old_class_count=int(old_class_count),
#                 batch_role=batch_role,
#                 features=z_joint,
#                 old_bases=old_bases,
#                 new_bases=bases_new,
#                 old_active_ranks=old_active,
#                 new_active_ranks=new_active,
#                 reliability=old_reliability,
#                 new_means=mu,
#                 new_variances=vars_new,
#                 init_new_means=init_means,
#                 init_new_bases=init_bases,
#                 init_new_variances=init_log_vars.exp().clamp_min(var_floor),
#                 reference_old_variances=old_vars,
#                 reference_old_active_ranks=old_active,
#                 ce_weight=1.0,
#                 rank_weight=self._inc_cfg_float("unified_rank_weight", margin_w),
#                 admission_weight=self._inc_cfg_float("unified_admission_weight", invasion_w),
#                 subspace_weight=self._inc_cfg_float("unified_subspace_weight", max(subspace_w, risk_sep_w)),
#                 volume_weight=self._inc_cfg_float("unified_volume_weight", volume_w),
#                 trust_weight=self._inc_cfg_float("unified_trust_weight", trust_w),
#                 logit_scale=float(getattr(self.args, "loss_scale", 8.0)),
#                 label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#                 rank_margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                 admission_margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                 target_overlap=self._inc_cfg_float("descriptor_overlap_target", 0.35),
#                 spectral_margin_strength=self._inc_cfg_float("spectral_margin_strength", 0.20),
#                 return_parts=True,
#             )

#             loss = loss_out["total"]
#             ce = loss_out.get("ce", self._zero_like_ref(loss))
#             margin = loss_out.get("rank", self._zero_like_ref(loss))
#             invasion = loss_out.get("admission", self._zero_like_ref(loss))
#             trust = loss_out.get("trust", self._zero_like_ref(loss))
#             subspace_collision = loss_out.get("subspace", self._zero_like_ref(loss))
#             center_collision = self._zero_like_ref(loss)
#             volume = loss_out.get("volume", self._zero_like_ref(loss))
#             risk_sep = subspace_collision
#             risk_active_pairs = loss_out.get("subspace_pair_count", self._zero_like_ref(loss))

#             if not torch.isfinite(loss):
#                 raise RuntimeError("Descriptor epoch produced non-finite loss.")
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_([mu, raw_basis, log_vars], self._inc_cfg_float("descriptor_refine_grad_clip", 1.0))
#             optimizer.step()

#             # Hard global trust-region projection around the originally inserted
#             # descriptor rows, not around the previous epoch. This avoids gradual
#             # descriptor drift across many epochs.
#             with torch.no_grad():
#                 if max_mean_shift > 0.0:
#                     delta = mu - init_means
#                     norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
#                     scale = (float(max_mean_shift) / norm).clamp(max=1.0)
#                     mu.copy_(init_means + delta * scale)
#                 if max_logvar_shift > 0.0:
#                     log_vars.copy_(torch.max(torch.min(log_vars, init_log_vars + max_logvar_shift), init_log_vars - max_logvar_shift))

#             stat["loss"] += float(loss.detach().item())
#             stat["ce"] += float(ce.detach().item())
#             stat["margin"] += float(margin.detach().item())
#             stat["invasion"] += float(invasion.detach().item())
#             stat["admission"] += float(invasion.detach().item())
#             stat["trust"] += float(trust.detach().item())
#             stat["subspace_collision"] += float(subspace_collision.detach().item())
#             stat["center_collision"] += float(center_collision.detach().item())
#             stat["volume"] += float(volume.detach().item())
#             stat["risk_sep"] += float(risk_sep.detach().item())
#             stat["risk_active_pairs"] += float(risk_active_pairs.detach().item())
#             stat["anchor_count"] += anchor_count
#             stat["boundary_anchor_count"] += float(getattr(self, "_last_boundary_replay_stats", {}).get("boundary_anchor_count", anchor_count))
#             stat["boundary_pair_count"] += float(getattr(self, "_last_boundary_replay_stats", {}).get("boundary_pair_count", 0.0))
#             if isinstance(loss_out, dict):
#                 def _loss_float(key: str, default: float = 0.0) -> float:
#                     v = loss_out.get(key, None)
#                     if torch.is_tensor(v):
#                         return float(v.detach().float().mean().cpu().item())
#                     return float(default)
#                 stat["admission_safe"] += _loss_float("admission_safe", 0.0)
#                 stat["admission_new_violation_rate"] += _loss_float("admission_new_violation_rate", 0.0)
#                 stat["admission_old_boundary_violation_rate"] += _loss_float("admission_old_boundary_violation_rate", 0.0)
#             stat["steps"] += 1.0

#         with torch.no_grad():
#             bases_final = self._orthonormalize_descriptor_bases(raw_basis, init_bases)
#             vars_final = log_vars.exp().clamp_min(var_floor)
#             self._commit_refined_descriptors(new_class_ids, mu.detach(), bases_final.detach(), vars_final.detach(), bank0)

#             denom = max(stat["steps"], 1.0)
#             for k in (
#                 "loss", "ce", "margin", "invasion", "admission", "trust", "subspace_collision", "center_collision", "volume",
#                 "risk_sep", "risk_active_pairs", "anchor_count", "boundary_anchor_count", "boundary_pair_count",
#                 "admission_safe", "admission_new_violation_rate", "admission_old_boundary_violation_rate",
#             ):
#                 stat[k] /= denom
#             stat["mean_shift"] = float((mu.detach() - init_means).norm(dim=1).mean().cpu().item())
#             stat["basis_shift"] = float((bases_final.detach() - init_bases).pow(2).mean().sqrt().cpu().item())
#             stat["logvar_shift"] = float((log_vars.detach() - init_log_vars).abs().mean().cpu().item())
#             risk_report = self._incremental_risk_report(int(old_class_count), new_class_ids)
#             stat["risk_old_new_max"] = float(risk_report.get("risk_old_new_max", 0.0))

#         if old_integrity is not None and hasattr(self, "_assert_old_bank_integrity"):
#             self._assert_old_bank_integrity(old_ids, old_integrity, context=f"phase_{phase}_descriptor_epoch")
#         if hasattr(self.model.geometry_bank, "validate_consistency"):
#             self.model.geometry_bank.validate_consistency(strict=True)
#         return stat

#     def _refine_current_phase_descriptors(
#         self,
#         *,
#         phase: int,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#         seen_classes: Iterable[int],
#         old_bank_snapshot: Dict[str, torch.Tensor],
#     ) -> Dict[str, float]:
#         """Backward-compatible one-shot refinement wrapper."""
#         state = self._prepare_descriptor_refinement_state(new_class_ids=new_class_ids, seen_classes=seen_classes)
#         return self._descriptor_refinement_epoch(
#             state=state,
#             phase=int(phase),
#             old_class_count=int(old_class_count),
#             old_bank_snapshot=old_bank_snapshot,
#             steps_per_epoch=self._inc_cfg_int("descriptor_refine_steps", 50),
#         )

#     # ------------------------------------------------------------------
#     # Optional energy-calibration epoch; no BiCyc/MSSL/projection paths
#     # ------------------------------------------------------------------
#     def _compute_logits_energy_from_features(
#         self,
#         features: torch.Tensor,
#         *,
#         classifier_mode: str,
#         return_parts: bool = False,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         kwargs = dict(
#             classifier_mode=classifier_mode,
#             return_parts=return_parts,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         )
#         if hasattr(self.model, "compute_logits_and_energy_from_features"):
#             try:
#                 out = self.model.compute_logits_and_energy_from_features(features, **kwargs)
#             except TypeError:
#                 kwargs.pop("spectral_summary_is_physical", None)
#                 try:
#                     out = self.model.compute_logits_and_energy_from_features(features, **kwargs)
#                 except TypeError:
#                     kwargs.pop("spectral_summary", None)
#                     out = self.model.compute_logits_and_energy_from_features(features, **kwargs)
#         else:
#             kwargs["return_energy"] = True
#             try:
#                 out = self.model.compute_logits_from_features(features, **kwargs)
#             except TypeError:
#                 kwargs.pop("spectral_summary_is_physical", None)
#                 try:
#                     out = self.model.compute_logits_from_features(features, **kwargs)
#                 except TypeError:
#                     kwargs.pop("spectral_summary", None)
#                     out = self.model.compute_logits_from_features(features, **kwargs)
#         if not isinstance(out, dict):
#             return {"logits": out}
#         return out

#     def _energy_calibration_reg(self, ref: torch.Tensor) -> torch.Tensor:
#         if hasattr(self.model, "energy_calibration_regularization_loss"):
#             reg = self.model.energy_calibration_regularization_loss()
#             if torch.is_tensor(reg):
#                 return reg.to(ref.device)
#         return self._zero_like_ref(ref)

#     def _current_valid_mask_from_bank(self) -> Optional[torch.Tensor]:
#         try:
#             bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#             if hasattr(self, "_canonicalize_bank"):
#                 bank = self._canonicalize_bank(bank)
#         except Exception:
#             return None
#         counts = bank.get("sample_counts", None)
#         if torch.is_tensor(counts) and counts.numel() > 0:
#             return counts.to(self.device).flatten() > 0
#         return None

#     def _train_epoch_incremental(
#         self,
#         loader,
#         optimizer,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#         old_bank_snapshot: Dict[str, torch.Tensor],
#         seen_classes: Iterable[int],
#         trainable_anchor: Optional[Dict[str, torch.Tensor]] = None,
#     ) -> Tuple[float, float]:
#         """Train optional score calibration only. Descriptor refinement is handled separately."""
#         self.model.train()
#         self._set_model_phase_and_old_count(getattr(self.model, "current_phase", 0), old_class_count)
#         classifier_mode = self._classifier_mode()
#         new_class_ids = [int(c) for c in new_class_ids]
#         seen_classes = [int(c) for c in seen_classes]

#         total_loss = 0.0
#         total_correct = 0
#         total_count = 0
#         stat_steps = 0
#         stat_sums = {
#             "ce_new": 0.0,
#             "ce_replay": 0.0,
#             "joint_ce": 0.0,
#             "geom_margin": 0.0,
#             "old_new_invasion": 0.0,
#             "energy_calib_reg": 0.0,
#             "weight_anchor": 0.0,
#             "anchor_count": 0.0,
#             "g2rpa_adapter": 0.0,
#             "g2rpa_old_delta": 0.0,
#             "g2rpa_old_gate": 0.0,
#             "g2rpa_old_mean_gate": 0.0,
#             "g2rpa_old_adapter_acc": 0.0,
#             "g2rpa_new_delta": 0.0,
#             "g2rpa_new_mean_gate": 0.0,
#         }

#         for batch in loader:
#             x, y, spectra, _ = self._unpack_hsi_batch(batch)
#             x = x.float().to(self.device, non_blocking=True)
#             y = y.long().to(self.device, non_blocking=True).view(-1)
#             self._assert_batch_labels_in_classes(y, new_class_ids, f"phase_{getattr(self.model, 'current_phase', -1)}_incremental_train")

#             if optimizer is not None:
#                 optimizer.zero_grad(set_to_none=True)

#             out = self._forward_real_batch(x, spectra, classifier_mode=classifier_mode, return_energy=True)
#             logits_new = self._mask_logits_to_seen_classes(out["logits"], seen_classes)
#             features = out["features"]
#             ce_new = self._stable_ce(logits_new, y)

#             old_z, old_y = self._sample_old_anchor_batch(old_bank_snapshot, old_class_count, new_class_ids)
#             ce_replay = self._zero_like_ref(logits_new)
#             joint_ce = self._zero_like_ref(logits_new)
#             margin = self._zero_like_ref(logits_new)
#             invasion = self._zero_like_ref(logits_new)
#             adapter_parts = self._compute_g2rpa_adapter_loss(
#                 real_out=out, old_z_base=None, old_z_adapt=None, old_y=None, gate_old=None
#             )
#             adapter_loss = adapter_parts["total"]
#             anchor_count = 0
#             valid_mask = self._current_valid_mask_from_bank()
#             unified_energy = out.get("energy", None)
#             unified_labels = y
#             unified_role = torch.zeros_like(y, dtype=torch.long, device=self.device)

#             if old_z is not None and old_y is not None and old_z.numel() > 0:
#                 old_z_base, old_z_score, gate_old = self._adapt_old_replay_if_needed(old_z)
#                 anchor_out = self._compute_logits_energy_from_features(old_z_score, classifier_mode="geometry_only")
#                 anchor_logits = self._mask_logits_to_seen_classes(anchor_out["logits"], seen_classes)
#                 ce_replay = self._stable_ce(anchor_logits, old_y.detach())
#                 joint_features = torch.cat([features, old_z_score], dim=0)
#                 joint_labels = torch.cat([y, old_y.detach()], dim=0)
#                 joint_out = self._compute_logits_energy_from_features(joint_features, classifier_mode="geometry_only")
#                 joint_logits = self._mask_logits_to_seen_classes(joint_out["logits"], seen_classes)
#                 joint_ce = self._stable_ce(joint_logits, joint_labels)
#                 unified_energy = joint_out.get("energy", None)
#                 unified_labels = joint_labels
#                 unified_role = torch.cat([
#                     torch.zeros((features.size(0),), device=self.device, dtype=torch.long),
#                     torch.ones((old_y.numel(),), device=self.device, dtype=torch.long),
#                 ], dim=0)
#                 if torch.is_tensor(joint_out.get("energy", None)):
#                     margin = geometry_energy_margin_loss(
#                         joint_out["energy"],
#                         joint_labels,
#                         margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                         valid_mask=valid_mask,
#                     )
#                     invasion = old_new_invasion_loss(
#                         joint_out["energy"],
#                         joint_labels,
#                         old_class_count=int(old_class_count),
#                         margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                         valid_mask=valid_mask,
#                     )
#                 anchor_count = int(old_y.numel())
#                 adapter_parts = self._compute_g2rpa_adapter_loss(
#                     real_out=out,
#                     old_z_base=old_z_base,
#                     old_z_adapt=old_z_score,
#                     old_y=old_y.detach(),
#                     gate_old=gate_old,
#                 )
#                 adapter_loss = adapter_parts["total"]
#             elif torch.is_tensor(out.get("energy", None)):
#                 margin = geometry_energy_margin_loss(
#                     out["energy"],
#                     y,
#                     margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                     valid_mask=valid_mask,
#                 )
#                 invasion = old_new_invasion_loss(
#                     out["energy"],
#                     y,
#                     old_class_count=int(old_class_count),
#                     margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                     valid_mask=valid_mask,
#                 )

#             unified_loss = ce_new + margin + invasion
#             if torch.is_tensor(unified_energy) and unified_energy.numel() > 0:
#                 try:
#                     cur_bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#                     if hasattr(self, "_canonicalize_bank"):
#                         cur_bank = self._canonicalize_bank(cur_bank)
#                     ids_t = torch.as_tensor(new_class_ids, device=self.device, dtype=torch.long)
#                     old_bases_for_loss = cur_bank["bases"][:int(old_class_count)].to(self.device)
#                     new_bases_for_loss = cur_bank["bases"].to(self.device).index_select(0, ids_t)
#                     old_active_for_loss = cur_bank.get("active_ranks", None)
#                     new_active_for_loss = cur_bank.get("active_ranks", None)
#                     old_rel_for_loss = cur_bank.get("reliability", None)
#                     if torch.is_tensor(old_active_for_loss):
#                         old_active_for_loss = old_active_for_loss[:int(old_class_count)].to(self.device)
#                     if torch.is_tensor(new_active_for_loss):
#                         new_active_for_loss = new_active_for_loss.to(self.device).index_select(0, ids_t)
#                     if torch.is_tensor(old_rel_for_loss):
#                         old_rel_for_loss = old_rel_for_loss[:int(old_class_count)].to(self.device)
#                 except Exception:
#                     cur_bank = {}
#                     old_bases_for_loss = None
#                     new_bases_for_loss = None
#                     old_active_for_loss = None
#                     new_active_for_loss = None
#                     old_rel_for_loss = None

#                 inc_loss_out = unified_spectral_geometry_loss(
#                     phase="incremental",
#                     energy=unified_energy,
#                     labels=unified_labels,
#                     sample_counts=valid_mask,
#                     old_class_count=int(old_class_count),
#                     batch_role=unified_role,
#                     old_bases=old_bases_for_loss,
#                     new_bases=new_bases_for_loss,
#                     old_active_ranks=old_active_for_loss,
#                     new_active_ranks=new_active_for_loss,
#                     reliability=old_rel_for_loss,
#                     ce_weight=1.0,
#                     rank_weight=self._inc_cfg_float("unified_rank_weight", self._inc_cfg_float("geometry_energy_margin_weight", 0.25)),
#                     admission_weight=self._inc_cfg_float("unified_admission_weight", self._inc_cfg_float("old_new_invasion_weight", 0.35)),
#                     subspace_weight=self._inc_cfg_float("unified_subspace_weight", self._inc_cfg_float("descriptor_subspace_collision_weight", 0.20)),
#                     volume_weight=0.0,
#                     trust_weight=0.0,
#                     logit_scale=float(getattr(self.args, "loss_scale", 8.0)),
#                     label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#                     rank_margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                     admission_margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                     target_overlap=self._inc_cfg_float("descriptor_overlap_target", 0.35),
#                     return_parts=True,
#                 )
#                 unified_loss = inc_loss_out["total"]
#                 joint_ce = inc_loss_out.get("ce", joint_ce)
#                 margin = inc_loss_out.get("rank", margin)
#                 invasion = inc_loss_out.get("admission", invasion)

#             calib_reg = self._energy_calibration_reg(logits_new)
#             weight_anchor = self._trainable_anchor_loss(trainable_anchor or {}, logits_new)
#             loss = (
#                 unified_loss
#                 + self._inc_cfg_float("energy_calibration_weight", 1e-3) * calib_reg
#                 + self._inc_cfg_float("g2rpa_adapter_weight", 1.0) * adapter_loss
#                 + weight_anchor
#             )
#             if not torch.isfinite(loss):
#                 raise RuntimeError("Non-finite incremental calibration loss.")

#             if optimizer is not None:
#                 loss.backward()
#                 trainable = [p for p in self.model.parameters() if p.requires_grad]
#                 if trainable:
#                     torch.nn.utils.clip_grad_norm_(trainable, float(getattr(self.args, "grad_clip_inc", 0.5)))
#                 optimizer.step()

#             total_loss += float(loss.detach().item())
#             c, n = self._incremental_accuracy_with_count(logits_new.detach(), y.detach(), new_class_ids)
#             total_correct += c
#             total_count += n
#             stat_sums["ce_new"] += float(ce_new.detach().item())
#             stat_sums["ce_replay"] += float(ce_replay.detach().item())
#             stat_sums["joint_ce"] += float(joint_ce.detach().item())
#             stat_sums["geom_margin"] += float(margin.detach().item())
#             stat_sums["old_new_invasion"] += float(invasion.detach().item())
#             stat_sums["energy_calib_reg"] += float(calib_reg.detach().item())
#             stat_sums["weight_anchor"] += float(weight_anchor.detach().item())
#             stat_sums["anchor_count"] += float(anchor_count)
#             stat_sums["g2rpa_adapter"] += float(adapter_loss.detach().item())
#             for key, stat_key in (
#                 ("old_delta", "g2rpa_old_delta"),
#                 ("old_gate", "g2rpa_old_gate"),
#                 ("old_mean_gate", "g2rpa_old_mean_gate"),
#                 ("old_adapter_acc", "g2rpa_old_adapter_acc"),
#                 ("new_delta", "g2rpa_new_delta"),
#                 ("new_mean_gate", "g2rpa_new_mean_gate"),
#             ):
#                 v = adapter_parts.get(key, None) if isinstance(adapter_parts, dict) else None
#                 if torch.is_tensor(v):
#                     stat_sums[stat_key] += float(v.detach().item())
#             stat_steps += 1

#         self._last_incremental_loss_stats = {k: v / max(stat_steps, 1) for k, v in stat_sums.items()}
#         return total_loss / max(stat_steps, 1), 100.0 * total_correct / max(total_count, 1)

#     # ------------------------------------------------------------------
#     # Main phase entry
#     # ------------------------------------------------------------------

#     # ------------------------------------------------------------------
#     # SGLAT-HSI transport and candidate admission
#     # ------------------------------------------------------------------

#     # ------------------------------------------------------------------
#     # HSI-safe SGLAT transport gates
#     # ------------------------------------------------------------------
#     def _safe_transport_float(self, value: object, default: float = 0.0) -> float:
#         if torch.is_tensor(value):
#             if value.numel() == 0:
#                 return float(default)
#             return float(value.detach().float().mean().cpu().item())
#         try:
#             return float(value)  # type: ignore[arg-type]
#         except Exception:
#             return float(default)

#     @torch.no_grad()
#     def _project_transport_to_hsi_safe_residual(
#         self,
#         A: torch.Tensor,
#         *,
#         low_rank: Optional[int] = None,
#         max_delta_fro: Optional[float] = None,
#     ) -> torch.Tensor:
#         """Project a full affine map to a small low-rank residual around identity.

#         HSI incremental phases often contain tiny classes.  A full D x D map can
#         overfit current new-class samples and poison old GeometryBank rows.  The
#         only safe old-row correction is A = I + Delta where Delta is low-rank
#         and norm-bounded.
#         """
#         if not torch.is_tensor(A) or A.dim() != 2 or A.size(0) != A.size(1):
#             raise RuntimeError(f"Transport A must be square [D,D], got {None if A is None else tuple(A.shape)}")
#         D = int(A.size(0))
#         eye = torch.eye(D, device=A.device, dtype=A.dtype)
#         delta = torch.nan_to_num(A - eye, nan=0.0, posinf=0.0, neginf=0.0)

#         r = int(self._inc_cfg_int("transport_low_rank", 4) if low_rank is None else low_rank)
#         r = max(0, min(r, D))
#         if r <= 0:
#             delta = torch.zeros_like(delta)
#         else:
#             try:
#                 U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
#                 delta = (U[:, :r] * S[:r].view(1, -1)).matmul(Vh[:r])
#             except RuntimeError:
#                 delta = torch.zeros_like(delta)

#         max_fro = float(self._inc_cfg_float("transport_max_a_minus_i_fro", 1.5) if max_delta_fro is None else max_delta_fro)
#         max_fro = max(max_fro, 1e-8)
#         fro = delta.norm()
#         if torch.isfinite(fro) and float(fro.detach().cpu().item()) > max_fro:
#             delta = delta * (max_fro / fro.clamp_min(1e-8))
#         return eye + delta

#     @torch.no_grad()
#     def _clamp_transport_bias(self, b: torch.Tensor, *, max_norm: Optional[float] = None) -> torch.Tensor:
#         if not torch.is_tensor(b):
#             raise RuntimeError("Transport b must be a tensor.")
#         b = torch.nan_to_num(b.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
#         max_b = float(self._inc_cfg_float("transport_max_b_norm", 0.75) if max_norm is None else max_norm)
#         max_b = max(max_b, 1e-8)
#         bn = b.norm()
#         if torch.isfinite(bn) and float(bn.detach().cpu().item()) > max_b:
#             b = b * (max_b / bn.clamp_min(1e-8))
#         return b

#     @torch.no_grad()
#     def _snapshot_transport_mutable_bank_rows(self) -> Dict[str, torch.Tensor]:
#         """Snapshot only buffers that transport is allowed to mutate."""
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             return {}
#         names = (
#             "means", "bases", "eigvals", "res_vars", "resvars", "active_ranks",
#             "chart_means", "chart_bases", "chart_eigvals", "chart_res_vars",
#             "chart_active_ranks", "chart_reliability", "chart_weights", "chart_valid_mask",
#         )
#         snap: Dict[str, torch.Tensor] = {}
#         for name in names:
#             value = getattr(gb, name, None)
#             if torch.is_tensor(value):
#                 snap[name] = value.detach().clone()
#         return snap

#     @torch.no_grad()
#     def _restore_transport_mutable_bank_rows(self, snap: Dict[str, torch.Tensor]) -> None:
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None or not snap:
#             return
#         for name, value in snap.items():
#             cur = getattr(gb, name, None)
#             if torch.is_tensor(cur) and tuple(cur.shape) == tuple(value.shape):
#                 cur.copy_(value.to(device=cur.device, dtype=cur.dtype))
#             else:
#                 try:
#                     setattr(gb, name, value.to(device=gb.device if hasattr(gb, "device") else value.device))
#                 except Exception:
#                     pass
#         if hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)

#     @torch.no_grad()
#     def _old_anchor_safety_after_transport(
#         self,
#         old_class_count: int,
#         seen_classes: Iterable[int],
#         *,
#         samples_per_class: Optional[int] = None,
#     ) -> Dict[str, float]:
#         """Check that transported old rows still classify old synthetic anchors as old."""
#         old_class_count = int(old_class_count)
#         if old_class_count <= 0:
#             return {"old_anchor_acc": 100.0, "old_anchor_violation": 0.0, "old_anchor_count": 0.0}
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None or not hasattr(gb, "sample_synthetic_features"):
#             return {"old_anchor_acc": 100.0, "old_anchor_violation": 0.0, "old_anchor_count": 0.0}
#         n = int(samples_per_class or self._inc_cfg_int("transport_safety_samples_per_class", 16))
#         x_old, y_old = gb.sample_synthetic_features(
#             class_ids=list(range(old_class_count)),
#             samples_per_class=max(1, n),
#             parallel_scale=self._inc_cfg_float("boundary_replay_parallel_scale", 0.15),
#             residual_scale=self._inc_cfg_float("boundary_replay_residual_scale", 0.05),
#             reliability_gated=True,
#             reliability_shrink_floor=self._inc_cfg_float("transport_safety_reliability_shrink", 0.25),
#         )
#         if not torch.is_tensor(x_old) or x_old.numel() == 0:
#             return {"old_anchor_acc": 0.0, "old_anchor_violation": 1.0, "old_anchor_count": 0.0}
#         x_old = x_old.to(self.device)
#         y_old = y_old.to(self.device).long()
#         out = self._compute_logits_energy_from_features(x_old, classifier_mode="geometry_only")
#         logits = self._mask_logits_to_seen_classes(out["logits"], seen_classes)
#         pred = logits.argmax(dim=1)
#         acc = float((pred == y_old).float().mean().detach().cpu().item())
#         return {
#             "old_anchor_acc": 100.0 * acc,
#             "old_anchor_violation": 1.0 - acc,
#             "old_anchor_count": float(y_old.numel()),
#         }

#     def _sglat_enabled(self) -> bool:
#         enabled = self._inc_cfg_bool(
#             "use_sglat_transport",
#             self._inc_cfg_bool("use_geometry_transport", False),
#         )
#         if not enabled:
#             return False

#         # Transport is meaningful only when the current canonical z-space has
#         # moved.  In SCBGR/descriptor-only mode the backbone/projection/adapter
#         # are frozen, so moving old rows is corruption, not correction.
#         if not self._adapter_mode_enabled() and not self._inc_cfg_bool("allow_transport_without_adapter", False):
#             return False
#         return True

#     def _sglat_transport_type(self) -> str:
#         return str(getattr(self.args, "transport_type", "ridge")).lower().strip()

#     def _snapshot_old_model_for_transport(self, phase: int):
#         """Frozen previous-phase model used only for z_old=f_{t-1}(x_new).

#         This is not KD: no logits are matched and no old samples/features are
#         stored.  The snapshot gives SGLAT a stable old coordinate system for
#         estimating old→new feature drift from current-phase samples only.
#         """
#         if int(phase) <= 0 or not self._sglat_enabled():
#             return None
#         if not self._inc_cfg_bool("allow_old_model_transport", True):
#             raise RuntimeError(
#                 "SGLAT requires allow_old_model_transport=True. The snapshot is "
#                 "used only to estimate feature-coordinate drift on current data."
#             )
#         old_model = copy.deepcopy(self.model).to(self.device).eval()
#         for p in old_model.parameters():
#             p.requires_grad = False
#         return old_model

#     @torch.no_grad()
#     def _extract_sglat_geometry_z(self, model, x: torch.Tensor) -> torch.Tensor:
#         """Extract the exact projected/canonical z-space used by GeometryBank."""
#         model.eval()
#         # Preferred API for the updated NECILModel. Transport pairs must be
#         # collected in the same space used by classifier scoring/bank decisions.
#         # In G²RPA mode that is the adapted scoring space, not the pre-adapter
#         # canonical space. Older models without the ``space`` argument fall back
#         # to their default extraction path.
#         if hasattr(model, "extract_geometry_features"):
#             try:
#                 out = model.extract_geometry_features(x, space="scoring", return_dict=True)
#             except TypeError:
#                 try:
#                     out = model.extract_geometry_features(x, space="scoring")
#                 except TypeError:
#                     out = model.extract_geometry_features(x)
#             if isinstance(out, dict):
#                 z = out.get("features", out.get("scoring_features", out.get("z", None)))
#             else:
#                 z = out
#             if torch.is_tensor(z):
#                 return z.detach()

#         # Existing project APIs.
#         for name in ("extract_projected_features", "extract_features"):
#             fn = getattr(model, name, None)
#             if callable(fn):
#                 out = fn(x)
#                 if isinstance(out, dict):
#                     for key in ("features", "projected", "z", "embedding"):
#                         if torch.is_tensor(out.get(key, None)):
#                             return out[key].detach()
#                 if torch.is_tensor(out):
#                     return out.detach()

#         # Last fallback: use model forward and read its feature field.
#         try:
#             out = model(x, classifier_mode="geometry_only", return_energy=True)
#         except TypeError:
#             out = model(x)
#         if isinstance(out, dict):
#             for key in ("features", "pre_adapter_features", "base_features", "z"):
#                 if torch.is_tensor(out.get(key, None)):
#                     return out[key].detach()
#         raise RuntimeError(
#             "Cannot extract SGLAT geometry features. Add NECILModel.extract_geometry_features() "
#             "that returns the same z-space used by GeometryBank."
#         )

#     @torch.no_grad()
#     def _collect_sglat_transport_pairs(self, loader, old_model, *, max_batches: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         if old_model is None:
#             raise RuntimeError("old_model snapshot is missing; cannot collect SGLAT pairs.")
#         max_batches = int(max_batches or getattr(self.args, "transport_batches", 20))
#         z_old_all, z_new_all = [], []
#         old_was_training = bool(old_model.training)
#         cur_was_training = bool(self.model.training)
#         old_model.eval()
#         self.model.eval()
#         try:
#             for bi, batch in enumerate(loader):
#                 x, _, _, _ = self._unpack_hsi_batch(batch)
#                 x = x.to(self.device, non_blocking=True).float()
#                 z_old = self._extract_sglat_geometry_z(old_model, x)
#                 z_new = self._extract_sglat_geometry_z(self.model, x)
#                 if z_old.shape != z_new.shape:
#                     raise RuntimeError(f"SGLAT feature pair shape mismatch: {tuple(z_old.shape)} vs {tuple(z_new.shape)}")
#                 z_old_all.append(torch.nan_to_num(z_old.float(), nan=0.0, posinf=0.0, neginf=0.0))
#                 z_new_all.append(torch.nan_to_num(z_new.float(), nan=0.0, posinf=0.0, neginf=0.0))
#                 if bi + 1 >= max_batches:
#                     break
#         finally:
#             old_model.train(old_was_training)
#             self.model.train(cur_was_training)

#         if not z_old_all:
#             raise RuntimeError("No SGLAT transport pairs were collected.")
#         z_old_cat = torch.cat(z_old_all, dim=0)
#         z_new_cat = torch.cat(z_new_all, dim=0)
#         if hasattr(self, "_sglat_pair_sanity_check"):
#             self._sglat_pair_sanity_check(z_old_cat, z_new_cat, context="sglat_transport_pairs")
#         return z_old_cat, z_new_cat

#     @torch.no_grad()
#     def _transport_old_rows_fallback(self, old_class_count: int, A: torch.Tensor, b: torch.Tensor, *, ema: float) -> Dict[str, float]:
#         """Fallback low-rank row transport when GeometryBank lacks a native method."""
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             raise RuntimeError("model.geometry_bank is required for SGLAT transport.")
#         required = ("means", "bases")
#         for key in required:
#             if not hasattr(gb, key):
#                 raise RuntimeError(f"GeometryBank missing buffer '{key}' for fallback SGLAT transport.")

#         means = gb.means
#         bases = gb.bases
#         if means.dim() != 2 or bases.dim() != 3:
#             raise RuntimeError("GeometryBank fallback expects means [C,D] and bases [C,D,R].")
#         C, D = int(means.size(0)), int(means.size(1))
#         old_class_count = int(min(max(old_class_count, 0), C))
#         if old_class_count <= 0:
#             return {"rows": 0.0}

#         A = A.to(device=means.device, dtype=means.dtype)
#         b = b.to(device=means.device, dtype=means.dtype).flatten()
#         if A.shape != (D, D) or b.numel() != D:
#             raise RuntimeError(f"SGLAT transport shape mismatch: A={tuple(A.shape)}, b={tuple(b.shape)}, D={D}")

#         # Prefer explicit eig/res buffers; fall back to variances only when exposed as a buffer.
#         eigvals = getattr(gb, "eigvals", None)
#         res_vars = getattr(gb, "res_vars", getattr(gb, "resvars", None))
#         if not torch.is_tensor(eigvals) or not torch.is_tensor(res_vars):
#             variances = getattr(gb, "variances", None)
#             if torch.is_tensor(variances) and variances.dim() == 2 and variances.size(0) == C:
#                 eigvals = variances[:, :-1]
#                 res_vars = variances[:, -1]
#             else:
#                 raise RuntimeError("GeometryBank fallback needs eigvals/res_vars or variances.")
#         active = getattr(gb, "active_ranks", None)
#         if not torch.is_tensor(active):
#             active = torch.full((C,), bases.size(2), device=means.device, dtype=torch.long)

#         alpha = float(max(0.0, min(ema, 0.999)))
#         var_floor = float(getattr(gb, "variance_floor", getattr(self.args, "geom_var_floor", 1e-4)))
#         transported = 0
#         for c in range(old_class_count):
#             r = int(active[c].detach().cpu().item()) if active.numel() > c else int(bases.size(2))
#             r = max(0, min(r, int(bases.size(2))))
#             mu = means[c].detach()
#             U = bases[c].detach()
#             lam = eigvals[c].detach().clamp_min(var_floor)
#             rv = res_vars[c].detach().clamp_min(var_floor)

#             mu_t = mu.matmul(A) + b
#             if r > 0:
#                 B = A.t().matmul(U[:, :r] * lam[:r].sqrt().view(1, -1))
#                 U_svd, S, _ = torch.linalg.svd(B, full_matrices=False)
#                 U_t = U.clone()
#                 U_t[:, :r] = U_svd[:, :r]
#                 lam_t = lam.clone()
#                 lam_t[:r] = S[:r].pow(2).clamp_min(var_floor)
#             else:
#                 U_t = U
#                 lam_t = lam
#             rv_t = (rv * A.pow(2).mean().clamp_min(1e-8)).clamp_min(var_floor)

#             means[c].copy_(alpha * means[c] + (1.0 - alpha) * mu_t)
#             bases[c].copy_(U_t)
#             eigvals[c].copy_(alpha * eigvals[c] + (1.0 - alpha) * lam_t)
#             res_vars[c].copy_(alpha * res_vars[c] + (1.0 - alpha) * rv_t)
#             transported += 1

#         if hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)
#         return {"rows": float(transported), "ema": alpha, "fallback": 1.0}

#     @torch.no_grad()
#     def _apply_sglat_transport(self, train_loader, old_model, old_class_count: int, *, context: str) -> Dict[str, float]:
#         """Estimate old-row transport and apply it only if it passes HSI safety gates.

#         Previous code applied transport before checking whether the affine map was
#         useful.  That is the collapse bug.  This version rejects negative/weak
#         RMSE gain, overly non-identity A, large b, and any map that damages old
#         synthetic-anchor classification.
#         """
#         if not self._sglat_enabled() or int(old_class_count) <= 0:
#             return {"active": 0.0, "rejected": 0.0, "reason": "disabled_or_no_old"}
#         if old_model is None:
#             return {"active": 0.0, "rejected": 1.0, "reason": "missing_old_model_snapshot"}

#         z_old, z_new = self._collect_sglat_transport_pairs(
#             train_loader,
#             old_model,
#             max_batches=int(getattr(self.args, "transport_batches", 20)),
#         )

#         from models.geometry_transport import (
#             estimate_gls_transport,
#             estimate_ridge_transport,
#             transport_diagnostics,
#         )

#         transport_type = self._sglat_transport_type()
#         kwargs = dict(
#             ridge=float(getattr(self.args, "transport_ridge", 1e-3)),
#             identity_blend=float(getattr(self.args, "transport_identity_blend", 0.75)),
#         )
#         if transport_type == "gls":
#             A, b = estimate_gls_transport(
#                 z_old,
#                 z_new,
#                 target_cov=str(getattr(self.args, "transport_gls_target_cov", "diag")),
#                 **kwargs,
#             )
#         elif transport_type == "ridge":
#             A, b = estimate_ridge_transport(z_old, z_new, **kwargs)
#         else:
#             raise RuntimeError(f"Unsupported SGLAT transport_type={transport_type!r}. Use ridge or gls.")

#         # HSI-specific stabilization: full affine maps are too flexible for
#         # small new-class phases, so the actual transported map is low-rank and
#         # close to identity.
#         A = self._project_transport_to_hsi_safe_residual(
#             A,
#             low_rank=int(getattr(self.args, "transport_low_rank", 4)),
#             max_delta_fro=float(getattr(self.args, "transport_max_a_minus_i_fro", 1.5)),
#         )
#         b = self._clamp_transport_bias(
#             b,
#             max_norm=float(getattr(self.args, "transport_max_b_norm", 0.75)),
#         )

#         diag = transport_diagnostics(z_old, z_new, A, b)
#         rmse_before = float(diag.get("rmse_before", 0.0))
#         rmse_after = float(diag.get("rmse_after", 0.0))
#         rmse_gain = float(diag.get("rmse_gain", 0.0))
#         rmse_ratio = rmse_after / max(rmse_before, 1e-12)
#         A_minus_I = float(diag.get("A_minus_I_fro", 0.0))
#         b_norm = float(diag.get("b_norm", 0.0))

#         min_gain = float(getattr(self.args, "transport_min_rmse_gain", 1e-5))
#         max_ratio = float(getattr(self.args, "transport_max_rmse_ratio", 0.98))
#         max_A = float(getattr(self.args, "transport_max_a_minus_i_fro", 1.5))
#         max_b = float(getattr(self.args, "transport_max_b_norm", 0.75))

#         unsafe = (
#             (not torch.isfinite(A).all())
#             or (not torch.isfinite(b).all())
#             or rmse_gain <= min_gain
#             or rmse_ratio >= max_ratio
#             or A_minus_I > max_A
#             or b_norm > max_b
#         )

#         if unsafe:
#             diag.update({
#                 "active": 0.0,
#                 "rejected": 1.0,
#                 "rows": 0.0,
#                 "rmse_ratio": rmse_ratio,
#                 "reason": "pre_apply_transport_gate_failed",
#             })
#             self._last_sglat_transport_stats = diag
#             print(
#                 f"[SGLAT-HSI REJECT] {context} | "
#                 f"rmse={rmse_before:.5f}->{rmse_after:.5f} | "
#                 f"gain={rmse_gain:.5f} | ratio={rmse_ratio:.4f} | "
#                 f"A-I={A_minus_I:.5f} | b={b_norm:.5f}"
#             )
#             if hasattr(self, "_save_sglat_transport_diagnostics"):
#                 self._save_sglat_transport_diagnostics(diag, phase=int(getattr(self.model, "current_phase", 0)))
#             return diag

#         gb = getattr(self.model, "geometry_bank", None)
#         ema = float(getattr(self.args, "transport_ema", 0.97))
#         snap = self._snapshot_transport_mutable_bank_rows()
#         if gb is not None and hasattr(gb, "transport_frozen_geometry"):
#             out = gb.transport_frozen_geometry(
#                 class_ids=list(range(int(old_class_count))),
#                 A=A,
#                 b=b,
#                 ema=ema,
#                 context=context,
#                 spectral_reliability_gate=self._inc_cfg_bool("transport_spectral_reliability_gate", True),
#                 min_reliability_gate=self._inc_cfg_float("transport_min_reliability_gate", 0.30),
#                 require_frozen=True,
#             )
#             rows = float(out.get("active", out.get("rows", 0.0))) if isinstance(out, dict) else float(old_class_count)
#         else:
#             out = self._transport_old_rows_fallback(old_class_count, A, b, ema=ema)
#             rows = float(out.get("rows", 0.0))

#         seen_classes = self._seen_classes_for_phase(int(getattr(self.model, "current_phase", 0)))
#         safety = self._old_anchor_safety_after_transport(old_class_count, seen_classes)
#         min_old_anchor_acc = float(getattr(self.args, "transport_min_old_anchor_acc", 95.0))
#         if safety.get("old_anchor_acc", 0.0) < min_old_anchor_acc:
#             self._restore_transport_mutable_bank_rows(snap)
#             diag.update({
#                 "active": 0.0,
#                 "rejected": 1.0,
#                 "rows": 0.0,
#                 "ema": ema,
#                 "rmse_ratio": rmse_ratio,
#                 "reason": "post_apply_old_anchor_safety_failed",
#                 **safety,
#             })
#             self._last_sglat_transport_stats = diag
#             print(
#                 f"[SGLAT-HSI ROLLBACK] {context} | old_anchor_acc={safety.get('old_anchor_acc', 0.0):.2f}% "
#                 f"< {min_old_anchor_acc:.2f}% | rmse={rmse_before:.5f}->{rmse_after:.5f} | "
#                 f"A-I={A_minus_I:.5f} | b={b_norm:.5f}"
#             )
#             if hasattr(self, "_save_sglat_transport_diagnostics"):
#                 self._save_sglat_transport_diagnostics(diag, phase=int(getattr(self.model, "current_phase", 0)))
#             return diag

#         diag.update({
#             "active": 1.0,
#             "rejected": 0.0,
#             "rows": rows,
#             "ema": ema,
#             "rmse_ratio": rmse_ratio,
#             "type": 0.0 if transport_type == "ridge" else 1.0,
#             **safety,
#         })
#         self._last_sglat_transport_stats = diag
#         print(
#             f"[SGLAT-HSI APPLY] {context} | rows={rows:.0f} | type={transport_type} | "
#             f"rmse={rmse_before:.5f}->{rmse_after:.5f} | gain={rmse_gain:.5f} | "
#             f"ratio={rmse_ratio:.4f} | A-I={A_minus_I:.5f} | b={b_norm:.5f} | "
#             f"old_anchor_acc={safety.get('old_anchor_acc', 0.0):.2f}%"
#         )
#         if hasattr(self, "_save_sglat_transport_diagnostics"):
#             self._save_sglat_transport_diagnostics(diag, phase=int(getattr(self.model, "current_phase", 0)))
#         return diag

#     def _initialize_candidate_new_descriptors(self, phase: int, phase_class_ids: Iterable[int], *, split: str = "train") -> Dict[str, torch.Tensor]:
#         """Create provisional new rows, then admission-shrink them before training.

#         This keeps your existing bootstrap path for compatibility, but it forbids
#         old-row mutation and immediately applies reliability-gated admission to
#         the new rows only.  If your GeometryBank exposes candidate APIs, use them
#         in GeometryBank; this trainer will still validate old-row integrity after
#         this call from train_incremental_phase().
#         """
#         new_ids = [int(c) for c in phase_class_ids]
#         print(f"[SGLAT Candidate] Initializing provisional descriptors for classes {new_ids}.")
#         self._bootstrap_phase_classes(int(phase), split=split, force_rebuild=True)
#         self._apply_reliability_gated_admission_to_new_rows(new_ids)

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)
#         return {}

#     def _risk_gate_and_correct_candidate_descriptors(self, old_class_count: int, phase_class_ids: Iterable[int], *, context: str) -> Dict[str, float]:
#         risk_before = self._incremental_risk_report(old_class_count, phase_class_ids)
#         if risk_before:
#             print(f"[{context} Risk Before] " + " | ".join(f"{k}={v:.4f}" for k, v in sorted(risk_before.items())))
#         corr = self._apply_risk_aware_descriptor_correction_to_new_rows(old_class_count, phase_class_ids)
#         self._last_descriptor_correction_stats = corr
#         risk_after = self._incremental_risk_report(old_class_count, phase_class_ids)
#         if risk_after:
#             print(f"[{context} Risk After] " + " | ".join(f"{k}={v:.4f}" for k, v in sorted(risk_after.items())))
#         return risk_after or {}

#     def _phase_geometry_safe(self, risk_report: Dict[str, float]) -> bool:
#         if not risk_report:
#             return True
#         return (
#             float(risk_report.get("risk_old_new_max", 0.0)) <= float(getattr(self.args, "max_old_new_risk", 1.00))
#             and float(risk_report.get("old_new_subspace_overlap_max", 0.0)) <= float(getattr(self.args, "max_old_new_overlap", 0.55))
#         )

#     def _append_history_snapshot(
#         self,
#         history: Dict[str, List[float]],
#         *,
#         train_stats: Dict[str, float],
#         val_stats: Dict[str, float],
#         desc_stats: Dict[str, float],
#         loss_stats: Optional[Dict[str, float]] = None,
#     ) -> None:
#         loss_stats = loss_stats or {}
#         history["train_loss"].append(float(train_stats.get("loss", 0.0)))
#         history["train_acc"].append(float(train_stats.get("acc", 0.0)))
#         history["val_loss"].append(float(val_stats.get("loss", 0.0)))
#         history["val_acc"].append(float(val_stats.get("acc", 0.0)))
#         history["val_old_acc"].append(float(val_stats.get("old_acc", 0.0)))
#         history["val_new_acc"].append(float(val_stats.get("new_acc", 0.0)))
#         history["val_hm"].append(float(val_stats.get("hm", 0.0)))
#         history["desc_refine_loss"].append(float(desc_stats.get("loss", 0.0)))
#         history["desc_refine_ce"].append(float(desc_stats.get("ce", 0.0)))
#         history["desc_refine_margin"].append(float(desc_stats.get("margin", 0.0)))
#         history["desc_refine_invasion"].append(float(desc_stats.get("invasion", 0.0)))
#         history["desc_refine_trust"].append(float(desc_stats.get("trust", 0.0)))
#         history["desc_subspace_collision"].append(float(desc_stats.get("subspace_collision", 0.0)))
#         history["desc_center_collision"].append(float(desc_stats.get("center_collision", 0.0)))
#         history["desc_volume"].append(float(desc_stats.get("volume", 0.0)))
#         history["desc_risk_sep"].append(float(desc_stats.get("risk_sep", 0.0)))
#         history["desc_risk_active_pairs"].append(float(desc_stats.get("risk_active_pairs", 0.0)))
#         history["risk_old_new_max"].append(float(desc_stats.get("risk_old_new_max", 0.0)))
#         if "desc_admission" in history:
#             history["desc_admission"].append(float(desc_stats.get("admission", desc_stats.get("invasion", 0.0))))
#             history["desc_admission_safe"].append(float(desc_stats.get("admission_safe", 0.0)))
#             history["desc_admission_new_violation_rate"].append(float(desc_stats.get("admission_new_violation_rate", 0.0)))
#             history["desc_admission_old_boundary_violation_rate"].append(float(desc_stats.get("admission_old_boundary_violation_rate", 0.0)))
#             history["boundary_anchor_count"].append(float(desc_stats.get("boundary_anchor_count", desc_stats.get("anchor_count", 0.0))))
#             history["boundary_pair_count"].append(float(desc_stats.get("boundary_pair_count", 0.0)))
#         corr = getattr(self, "_last_descriptor_correction_stats", {}) or {}
#         history["descriptor_corrections"].append(float(corr.get("active", 0.0)))
#         history["correction_risk_before"].append(float(corr.get("max_risk_before", 0.0)))
#         history["correction_risk_after"].append(float(corr.get("max_risk_after", 0.0)))
#         history["correction_overlap_before"].append(float(corr.get("max_overlap_before", 0.0)))
#         history["correction_overlap_after"].append(float(corr.get("max_overlap_after", 0.0)))
#         history["desc_mean_shift"].append(float(desc_stats.get("mean_shift", 0.0)))
#         history["desc_basis_shift"].append(float(desc_stats.get("basis_shift", 0.0)))
#         history["desc_logvar_shift"].append(float(desc_stats.get("logvar_shift", 0.0)))
#         history["inc_ce_new"].append(float(loss_stats.get("ce_new", 0.0)))
#         history["inc_ce_replay"].append(float(loss_stats.get("ce_replay", 0.0)))
#         history["inc_joint_ce"].append(float(loss_stats.get("joint_ce", 0.0)))
#         history["inc_geom_margin"].append(float(loss_stats.get("geom_margin", 0.0)))
#         history["inc_old_new_invasion"].append(float(loss_stats.get("old_new_invasion", 0.0)))
#         history["inc_energy_calib_reg"].append(float(loss_stats.get("energy_calib_reg", 0.0)))
#         history["inc_weight_anchor"].append(float(loss_stats.get("weight_anchor", 0.0)))
#         history["inc_anchor_count"].append(float(loss_stats.get("anchor_count", desc_stats.get("anchor_count", 0.0))))
#         history["g2rpa_adapter"].append(float(loss_stats.get("g2rpa_adapter", 0.0)))
#         history["g2rpa_old_delta"].append(float(loss_stats.get("g2rpa_old_delta", 0.0)))
#         history["g2rpa_old_gate"].append(float(loss_stats.get("g2rpa_old_gate", 0.0)))
#         history["g2rpa_old_mean_gate"].append(float(loss_stats.get("g2rpa_old_mean_gate", 0.0)))
#         history["g2rpa_old_adapter_acc"].append(float(loss_stats.get("g2rpa_old_adapter_acc", 0.0)))
#         history["g2rpa_new_delta"].append(float(loss_stats.get("g2rpa_new_delta", 0.0)))
#         history["g2rpa_new_mean_gate"].append(float(loss_stats.get("g2rpa_new_mean_gate", 0.0)))
#         tstats = getattr(self, "_last_sglat_transport_stats", {}) or {}
#         if "sglat_rmse_before" in history:
#             history["sglat_rmse_before"].append(float(tstats.get("rmse_before", 0.0)))
#             history["sglat_rmse_after"].append(float(tstats.get("rmse_after", 0.0)))
#             history["sglat_A_minus_I"].append(float(tstats.get("A_minus_I_fro", 0.0)))
#             history["sglat_b_norm"].append(float(tstats.get("b_norm", 0.0)))

#     def train_incremental_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4) -> Dict:
#         phase = int(phase)
#         if phase <= 0:
#             raise ValueError("train_incremental_phase() must only be called for phase > 0.")

#         print(f"==== Incremental Phase {phase} | SGLAT-HSI: Transport → Candidate Admission → Boundary Refinement ====")
#         self.dataset.start_phase(phase)
#         old_class_count = len(self.dataset.get_classes_up_to_phase(phase - 1))
#         phase_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         seen_classes = self._seen_classes_for_phase(phase)
#         self._set_model_phase_and_old_count(phase, old_class_count)

#         if hasattr(self.model, "ensure_class_capacity"):
#             self.model.ensure_class_capacity(max(seen_classes) + 1)
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes_up_to"):
#             self.model.geometry_bank.freeze_classes_up_to(old_class_count)

#         train_loader = self.dataset.get_phase_dataloader(phase, split="train", batch_size=batch_size, shuffle=True)
#         val_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)

#         old_ids = list(range(old_class_count))
#         old_integrity = self._old_bank_integrity_snapshot(old_ids) if hasattr(self, "_old_bank_integrity_snapshot") else None
#         old_model_for_transport = self._snapshot_old_model_for_transport(phase)

#         # Candidate descriptor initialization replaces blind committed bootstrap.
#         # The candidate rows are immediately reliability-gated and corrected against
#         # old geometry.  When adapter plasticity moves z-space, SGLAT later
#         # transports old rows and re-runs the candidate correction.
#         self._initialize_candidate_new_descriptors(phase, phase_class_ids, split="train")
#         risk_report = self._risk_gate_and_correct_candidate_descriptors(
#             old_class_count,
#             phase_class_ids,
#             context="CandidateAdmission",
#         )
#         if old_integrity is not None and hasattr(self, "_assert_old_bank_integrity"):
#             self._assert_old_bank_integrity(old_ids, old_integrity, context="post_candidate_admission")

#         old_bank_snapshot = self._snapshot_old_bank_clean(old_class_count)
#         raw_bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_validate_bank_has_classes"):
#             self._validate_bank_has_classes(raw_bank, seen_classes)

#         history = {
#             "train_loss": [],
#             "train_acc": [],
#             "val_loss": [],
#             "val_acc": [],
#             "val_old_acc": [],
#             "val_new_acc": [],
#             "val_hm": [],
#             "desc_refine_loss": [],
#             "desc_refine_ce": [],
#             "desc_refine_margin": [],
#             "desc_refine_invasion": [],
#             "desc_refine_trust": [],
#             "desc_subspace_collision": [],
#             "desc_center_collision": [],
#             "desc_volume": [],
#             "desc_risk_sep": [],
#             "desc_risk_active_pairs": [],
#             "risk_old_new_max": [],
#             "desc_admission": [],
#             "desc_admission_safe": [],
#             "desc_admission_new_violation_rate": [],
#             "desc_admission_old_boundary_violation_rate": [],
#             "boundary_anchor_count": [],
#             "boundary_pair_count": [],
#             "descriptor_corrections": [],
#             "correction_risk_before": [],
#             "correction_risk_after": [],
#             "correction_overlap_before": [],
#             "correction_overlap_after": [],
#             "desc_mean_shift": [],
#             "desc_basis_shift": [],
#             "desc_logvar_shift": [],
#             "inc_ce_new": [],
#             "inc_ce_replay": [],
#             "inc_joint_ce": [],
#             "inc_geom_margin": [],
#             "inc_old_new_invasion": [],
#             "inc_energy_calib_reg": [],
#             "inc_weight_anchor": [],
#             "inc_anchor_count": [],
#             "g2rpa_adapter": [],
#             "g2rpa_old_delta": [],
#             "g2rpa_old_gate": [],
#             "g2rpa_old_mean_gate": [],
#             "g2rpa_old_adapter_acc": [],
#             "g2rpa_new_delta": [],
#             "g2rpa_new_mean_gate": [],
#             "sglat_rmse_before": [],
#             "sglat_rmse_after": [],
#             "sglat_A_minus_I": [],
#             "sglat_b_norm": [],
#         }

#         init_val_stats = self._validate_split_metrics(val_loader, old_class_count)
#         init_train_stats = self._validate_split_metrics(train_loader, old_class_count)
#         min_init_new = float(getattr(self.args, "min_init_new_train_acc", 5.0))
#         if float(init_train_stats.get("new_acc", init_train_stats.get("acc", 0.0))) < min_init_new:
#             msg = (
#                 f"[SGLAT STOP] Candidate descriptors are dead before refinement: "
#                 f"train_new_acc={float(init_train_stats.get('new_acc', init_train_stats.get('acc', 0.0))):.2f}% "
#                 f"< {min_init_new:.2f}%. Fix descriptor insertion/risk correction before training adapter/transport."
#             )
#             if self._inc_cfg_bool("stop_on_dead_candidate", True):
#                 raise RuntimeError(msg)
#             print(msg + " Continuing only because stop_on_dead_candidate=False.")
#         best_score = self._select_score(init_val_stats, phase)
#         best_state = self._capture_state()
#         print(
#             f"[InitVal] Phase {phase} | TrainAcc: {init_train_stats['acc']:.2f}% | "
#             f"ValAcc: {init_val_stats['acc']:.2f}% | Old: {init_val_stats['old_acc']:.2f}% | "
#             f"New: {init_val_stats['new_acc']:.2f}% | HM: {init_val_stats['hm']:.2f}% | "
#             f"Loss: {init_val_stats['loss']:.4f}"
#         )

#         # Freeze model/backbone/projection and validate that no forbidden neural
#         # path is trainable. Epoch training below optimizes descriptor parameters,
#         # not model weights, so an empty trainable-parameter list is valid.
#         trainable_params = self._set_incremental_trainable_params(old_class_count)
#         if hasattr(self, "_print_trainable_summary"):
#             self._print_trainable_summary(phase)

#         steps_per_epoch = self._inc_cfg_int(
#             "descriptor_refine_steps_per_epoch",
#             self._inc_cfg_int("descriptor_refine_steps", 50),
#         )
#         steps_per_epoch = int(max(0, steps_per_epoch))
#         total_epochs = int(max(epochs, 0))
#         adapter_mode = self._adapter_mode_enabled()
#         desc_state = None if adapter_mode else self._prepare_descriptor_refinement_state(
#             new_class_ids=phase_class_ids,
#             seen_classes=seen_classes,
#         )

#         print(
#             f"[SGLAT Incremental] phase={phase} | old_classes={old_class_count} | "
#             f"new_classes={phase_class_ids} | seen={seen_classes} | "
#             f"descriptor_refine={self._inc_cfg_bool('refine_new_descriptors', True)} | "
#             f"epochs={total_epochs} | desc_steps/epoch={steps_per_epoch} | "
#             f"sglat={self._sglat_enabled()} | transport={self._sglat_transport_type()} | "
#             f"boundary_replay={self._inc_cfg_bool('use_boundary_geometry_replay', True)} | "
#             f"boundary_samples/pair={self._inc_cfg_int('boundary_replay_samples_per_pair', 12)} | "
#             f"risk_correction={self._inc_cfg_bool('risk_aware_descriptor_correction', True)} | "
#             f"update_mode={self._incremental_update_mode()} | "
#             f"energy_calibrator={self._inc_cfg_bool('use_energy_calibrator', False)} | "
#             f"model_trainable_params={sum(p.numel() for p in trainable_params):,}"
#         )

#         self._sglat_transport_done_this_phase = False
#         no_improve = 0
#         if total_epochs <= 0:
#             print(f"[SkipTrain] Phase {phase}: epochs_inc <= 0, evaluation only after provisional candidate admission.")
#         else:
#             for epoch in range(total_epochs):
#                 # In G²RPA mode, train the adapter first with real new samples +
#                 # old synthetic replay, then refine new descriptors against the
#                 # current adapted feature space.  Descriptor-only mode keeps the
#                 # original persistent descriptor state.
#                 loss_stats = getattr(self, "_last_incremental_loss_stats", {})
#                 tr_loss = 0.0
#                 if trainable_params:
#                     optimizer = optim.AdamW(
#                         trainable_params,
#                         lr=float(getattr(self.args, "adapter_lr", lr)) if adapter_mode else float(lr),
#                         weight_decay=float(getattr(self.args, "adapter_weight_decay", 0.0)) if adapter_mode else float(getattr(self.args, "weight_decay", 1e-5)),
#                     )
#                     tr_loss, _ = self._train_epoch_incremental(
#                         train_loader,
#                         optimizer,
#                         old_class_count=old_class_count,
#                         new_class_ids=phase_class_ids,
#                         old_bank_snapshot=old_bank_snapshot,
#                         seen_classes=seen_classes,
#                         trainable_anchor=self._capture_trainable_anchor(),
#                     )
#                     loss_stats = getattr(self, "_last_incremental_loss_stats", {})

#                     # SGLAT: after bounded adapter plasticity changes z-space,
#                     # transport old rows and re-admit/correct the provisional new
#                     # descriptors against aligned old geometry.  Apply once per
#                     # phase by default; repeated transport can over-move small
#                     # HSI classes.
#                     if (
#                         self._sglat_enabled()
#                         and old_model_for_transport is not None
#                         and not bool(getattr(self, "_sglat_transport_done_this_phase", False))
#                         and (epoch + 1) >= int(getattr(self.args, "transport_after_adapter_epoch", 3))
#                     ):
#                         transport_stats = self._apply_sglat_transport(
#                             train_loader,
#                             old_model_for_transport,
#                             old_class_count,
#                             context=f"phase{phase}_epoch{epoch + 1}",
#                         )
#                         self._sglat_transport_done_this_phase = True
#                         if float(transport_stats.get("active", 0.0)) > 0.0 and float(transport_stats.get("rejected", 0.0)) <= 0.0:
#                             risk_report = self._risk_gate_and_correct_candidate_descriptors(
#                                 old_class_count,
#                                 phase_class_ids,
#                                 context="PostTransportAdmission",
#                             )
#                             if not self._phase_geometry_safe(risk_report):
#                                 # Descriptor-risk is a diagnostic gate, not a training-kill switch.
#                                 # In Phase-2 IP, physically similar classes can keep high band/risk
#                                 # scores even while Old/New/HM accuracy is improving.  Stopping here
#                                 # turns a useful warning into a false failure.  Keep the accepted
#                                 # transport if it already passed RMSE + old-anchor safety; let the
#                                 # validation HM decide checkpoint quality.
#                                 msg = "[SGLAT WARN] Candidate geometry remains high-risk after accepted transport/correction."
#                                 try:
#                                     self._last_candidate_unsafe_after_transport_stats = dict(risk_report)
#                                 except Exception:
#                                     self._last_candidate_unsafe_after_transport_stats = risk_report
#                                 if self._inc_cfg_bool("strict_stop_on_unsafe_candidate", False):
#                                     raise RuntimeError(f"{msg} risk_report={risk_report}")
#                                 if self._inc_cfg_bool("stop_on_unsafe_candidate", False):
#                                     raise RuntimeError(f"{msg} risk_report={risk_report}")
#                                 print(f"{msg} Continuing because transport passed old-anchor safety and validation will select the best state. risk_report={risk_report}")
#                             old_bank_snapshot = self._snapshot_old_bank_clean(old_class_count)
#                         else:
#                             print(f"[SGLAT-HSI] transport skipped/rejected; keeping old GeometryBank rows unchanged. reason={transport_stats.get('reason', 'unknown')}")

#                 epoch_desc_state = desc_state
#                 if adapter_mode and self._inc_cfg_bool("refine_new_descriptors", True):
#                     # Adapter parameters changed this epoch.  Re-cache current
#                     # phase features so descriptor rows are fitted to z_adapt,
#                     # not stale pre-adapter features.
#                     epoch_desc_state = self._prepare_descriptor_refinement_state(
#                         new_class_ids=phase_class_ids,
#                         seen_classes=seen_classes,
#                     )
#                 desc_stats = self._descriptor_refinement_epoch(
#                     state=epoch_desc_state,
#                     phase=phase,
#                     old_class_count=old_class_count,
#                     old_bank_snapshot=old_bank_snapshot,
#                     steps_per_epoch=steps_per_epoch,
#                 )
#                 if trainable_params and desc_stats.get("steps", 0.0) <= 0.0:
#                     desc_stats["loss"] = float(tr_loss)

#                 train_eval_stats = self._validate_split_metrics(train_loader, old_class_count)
#                 val_stats = self._validate_split_metrics(val_loader, old_class_count)
#                 self._append_history_snapshot(
#                     history,
#                     train_stats=train_eval_stats,
#                     val_stats=val_stats,
#                     desc_stats=desc_stats,
#                     loss_stats=loss_stats,
#                 )

#                 cal_state = self.model.energy_calibration_state() if hasattr(self.model, "energy_calibration_state") else {}
#                 print(
#                     f"[IncEpoch] Phase {phase} Ep {epoch + 1:03d}/{total_epochs} | "
#                     f"DescLoss: {desc_stats['loss']:.4f} | CE: {desc_stats['ce']:.4f} | "
#                     f"Margin: {desc_stats['margin']:.4f} | Admit: {desc_stats.get('admission', desc_stats['invasion']):.4f} | "
#                     f"Safe: {desc_stats.get('admission_safe', 0.0):.2f} | OldBViol: {desc_stats.get('admission_old_boundary_violation_rate', 0.0):.3f} | "
#                     f"Trust: {desc_stats['trust']:.6f} | "
#                     f"SubColl: {desc_stats.get('subspace_collision', 0.0):.4f} | "
#                     f"RiskSep: {desc_stats.get('risk_sep', 0.0):.4f} | "
#                     f"CtrColl: {desc_stats.get('center_collision', 0.0):.4f} | "
#                     f"Vol: {desc_stats.get('volume', 0.0):.4f} | Steps: {desc_stats['steps']:.0f} | "
#                     f"TrainAcc: {train_eval_stats['acc']:.2f}% | ValAcc: {val_stats['acc']:.2f}% | "
#                     f"Old: {val_stats['old_acc']:.2f}% | New: {val_stats['new_acc']:.2f}% | HM: {val_stats['hm']:.2f}% | "
#                     f"MeanShift: {desc_stats['mean_shift']:.4f} | BasisShift: {desc_stats['basis_shift']:.4f} | "
#                     f"LogVarShift: {desc_stats['logvar_shift']:.4f} | "
#                     f"G2RPA: {loss_stats.get('g2rpa_adapter', 0.0):.4f} | "
#                     f"GateOld: {loss_stats.get('g2rpa_old_mean_gate', 0.0):.4f} | "
#                     f"GateNew: {loss_stats.get('g2rpa_new_mean_gate', 0.0):.4f} | "
#                     f"OldScale: {float(cal_state.get('old_scale', 1.0)):.4f} | "
#                     f"NewScale: {float(cal_state.get('new_scale', 1.0)):.4f}"
#                 )

#                 score = self._select_score(val_stats, phase)
#                 if score > best_score:
#                     best_score = score
#                     best_state = self._capture_state()
#                     no_improve = 0
#                 else:
#                     no_improve += 1
#                 patience = self._inc_cfg_int("early_stop_patience", 0)
#                 if patience > 0 and no_improve >= patience:
#                     print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
#                     break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, old_class_count)

#         # Finalize the phase without rebuilding old rows. Rebuilding current rows after descriptor
#         # refinement would overwrite the refined descriptors, so keep finalize_incremental_rebuild off.
#         old_integrity = self._old_bank_integrity_snapshot(old_ids) if hasattr(self, "_old_bank_integrity_snapshot") else None
#         if self._inc_cfg_bool("finalize_incremental_rebuild", False):
#             raise RuntimeError(
#                 "finalize_incremental_rebuild=True would overwrite descriptor-refined rows. "
#                 "Keep it False in the clean descriptor-refinement method."
#             )
#         if hasattr(self.dataset, "finalize_phase"):
#             self.dataset.finalize_phase(phase)
#         else:
#             self._finalize_phase_memory(phase, split="train")
#         if old_integrity is not None and hasattr(self, "_assert_old_bank_integrity"):
#             self._assert_old_bank_integrity(old_ids, old_integrity, context="post_finalize_incremental")

#         new_old_count = len(self.dataset.get_classes_up_to_phase(phase))
#         self._set_model_phase_and_old_count(phase, new_old_count)
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes_up_to"):
#             self.model.geometry_bank.freeze_classes_up_to(new_old_count)

#         if bool(getattr(self.args, "save_geometry_diagnostics", True)) and hasattr(self, "diagnose_full_base_geometry"):
#             try:
#                 cumulative_ids = [int(c) for c in self.dataset.get_classes_up_to_phase(phase)]
#                 diag_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)
#                 phase_diag = self.diagnose_full_base_geometry(
#                     diag_loader,
#                     cumulative_ids,
#                     anchors_per_class=int(getattr(self.args, "geometry_diag_anchors_per_class", 64)),
#                     topk_pairs=int(getattr(self.args, "geometry_diag_topk_pairs", 20)),
#                     topk_bands=int(getattr(self.args, "geometry_diag_topk_bands", 5)),
#                 )
#                 setattr(self, f"_last_phase_{phase}_geometry_diagnostics", phase_diag)
#                 if hasattr(self, "_print_geometry_diagnostics_summary"):
#                     self._print_geometry_diagnostics_summary(phase_diag)
#                 if hasattr(self, "_save_geometry_diagnostics_to_files"):
#                     self._save_geometry_diagnostics_to_files(phase_diag, phase=phase)
#             except Exception as exc:
#                 print(f"[WARN] Phase {phase} geometry diagnostics failed: {exc}")

#         if hasattr(self, "save_checkpoint"):
#             self.save_checkpoint(phase, history)
#         return history






















# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import torch
# import torch.nn.functional as F
# import torch.optim as optim

# from losses.loss import (
#     unified_spectral_geometry_loss,
#     sample_geometry_features,
#     sample_boundary_geometry_features,
#     descriptor_subspace_collision_loss,
#     center_to_old_ellipsoid_loss,
#     descriptor_volume_control_loss,
#     descriptor_trust_region_loss,
#     GeometryGatedAdapterLoss,
# )
# from models.classifier import geometry_energy_margin_loss, old_new_invasion_loss


# class IncrementalPhaseTrainer:
#     # ------------------------------------------------------------------
#     # Basic config / tensor helpers
#     # ------------------------------------------------------------------
#     def _zero_like_ref(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if hasattr(self, "_zero"):
#             return self._zero(ref)
#         if torch.is_tensor(ref):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     def _inc_cfg_float(self, name: str, default: float) -> float:
#         return float(getattr(self, name, getattr(self.args, name, default)))

#     def _inc_cfg_int(self, name: str, default: int) -> int:
#         return int(getattr(self, name, getattr(self.args, name, default)))

#     def _inc_cfg_bool(self, name: str, default: bool) -> bool:
#         value = getattr(self, name, getattr(self.args, name, default))
#         if isinstance(value, str):
#             return value.strip().lower() in {"1", "true", "yes", "y", "on"}
#         return bool(value)

#     def _classifier_mode(self) -> str:
#         if hasattr(self, "_inc_classifier_mode"):
#             return str(self._inc_classifier_mode()).lower().strip()
#         return str(getattr(self.args, "incremental_classifier_mode", "srgp")).lower().strip()

#     def _seen_classes_for_phase(self, phase: int) -> List[int]:
#         if hasattr(self.dataset, "get_classes_up_to_phase"):
#             seen = [int(c) for c in self.dataset.get_classes_up_to_phase(int(phase))]
#             if seen:
#                 return sorted(set(seen))
#         classes: List[int] = []
#         for p in range(int(phase) + 1):
#             classes.extend(int(c) for c in self.dataset.phase_to_classes[p])
#         return sorted(set(classes))

#     def _mask_logits_to_seen_classes(self, logits: torch.Tensor, seen_classes: Iterable[int]) -> torch.Tensor:
#         if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError(f"logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")
#         seen_list = [int(c) for c in seen_classes]
#         if not seen_list:
#             raise RuntimeError("seen_classes is empty.")
#         seen = torch.as_tensor(seen_list, device=logits.device, dtype=torch.long)
#         if int(seen.min().item()) < 0 or int(seen.max().item()) >= logits.size(1):
#             raise RuntimeError(
#                 f"seen class range [{int(seen.min())},{int(seen.max())}] incompatible with logits width={logits.size(1)}"
#             )
#         masked = torch.full_like(logits, -1e9)
#         masked.index_copy_(1, seen, logits.index_select(1, seen))
#         return masked

#     def _assert_batch_labels_in_classes(self, y: torch.Tensor, class_ids: Iterable[int], context: str) -> None:
#         y = y.long().view(-1)
#         allowed = torch.as_tensor([int(c) for c in class_ids], device=y.device, dtype=torch.long)
#         if y.numel() == 0:
#             raise RuntimeError(f"{context}: empty label tensor.")
#         if allowed.numel() == 0:
#             raise RuntimeError(f"{context}: empty allowed class set.")
#         if hasattr(torch, "isin"):
#             ok = torch.isin(y, allowed).all()
#         else:
#             valid = torch.zeros_like(y, dtype=torch.bool)
#             for c in allowed:
#                 valid |= y == int(c)
#             ok = valid.all()
#         if not bool(ok.item()):
#             raise RuntimeError(
#                 f"{context}: labels are not expected global sequential ids. "
#                 f"unique={torch.unique(y).detach().cpu().tolist()}, allowed={allowed.detach().cpu().tolist()}"
#             )

#     def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
#             return self._zero_like_ref(logits)
#         labels = labels.to(device=logits.device).long().view(-1)
#         if labels.numel() == 0:
#             return logits.sum() * 0.0
#         if labels.numel() != logits.size(0):
#             raise RuntimeError(f"CE label/logit batch mismatch: {labels.numel()} vs {logits.size(0)}")
#         min_label = int(labels.min().detach().item())
#         max_label = int(labels.max().detach().item())
#         if min_label < 0 or max_label >= logits.size(1):
#             raise RuntimeError(f"CE labels [{min_label},{max_label}] incompatible with logits width={logits.size(1)}")
#         clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
#         return F.cross_entropy(
#             logits.clamp(-clip, clip),
#             labels,
#             label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#         )

#     def _incremental_accuracy_with_count(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         class_ids: Iterable[int],
#     ) -> Tuple[int, int]:
#         labels = labels.to(device=logits.device).long().view(-1)
#         valid = torch.zeros_like(labels, dtype=torch.bool)
#         for c in [int(x) for x in class_ids]:
#             valid |= labels == int(c)
#         if not bool(valid.any().item()):
#             return 0, 0
#         pred = logits[valid].argmax(dim=1)
#         return int((pred == labels[valid]).sum().item()), int(valid.sum().item())


#     # ------------------------------------------------------------------
#     # SRGP spectral-summary handling for real samples only
#     # ------------------------------------------------------------------
#     def _inc_spectral_summary_is_physical(self, explicit: Optional[bool] = None) -> bool:
#         """Return whether a batch spectral summary is physically wavelength ordered.

#         SRGP spectral residual energy is valid only for raw HSI spectra.  PCA
#         components must not be treated as wavelengths.  Synthetic replay never
#         receives spectral summaries.
#         """
#         if explicit is not None:
#             return bool(explicit)
#         for key in (
#             "spectral_summary_is_physical",
#             "raw_spectral_summary_is_physical",
#             "incremental_spectral_summary_is_physical",
#         ):
#             if hasattr(self.args, key):
#                 return self._inc_cfg_bool(key, False)
#         if hasattr(self.args, "pca_components") and int(getattr(self.args, "pca_components", 0)) > 0:
#             return False
#         return False

#     @staticmethod
#     def _center_spectrum_from_input(x: torch.Tensor) -> Optional[torch.Tensor]:
#         if not torch.is_tensor(x) or x.dim() != 4 or x.size(1) <= 0:
#             return None
#         h = int(x.size(-2)) // 2
#         w = int(x.size(-1)) // 2
#         return x[:, :, h, w].contiguous()

#     def _prepare_real_spectral_summary(
#         self,
#         x: torch.Tensor,
#         spectra: Optional[torch.Tensor] = None,
#     ) -> Tuple[Optional[torch.Tensor], bool]:
#         """Prepare real-sample spectral summaries for SRGP scoring.

#         Priority is raw spectra supplied by the dataloader/helper.  If absent,
#         the input patch center is used only as a non-physical summary unless the
#         user explicitly marks it as physical.  This avoids fake derivative losses
#         over PCA components.
#         """
#         spectral_summary = None
#         is_physical = False

#         if torch.is_tensor(spectra) and spectra.numel() > 0:
#             s = spectra.to(device=x.device, dtype=x.dtype, non_blocking=True)
#             if s.dim() > 2:
#                 s = s.view(s.size(0), -1)
#             spectral_summary = s
#             is_physical = self._inc_spectral_summary_is_physical(True)
#         else:
#             spectral_summary = self._center_spectrum_from_input(x)
#             is_physical = self._inc_spectral_summary_is_physical(None)

#         if spectral_summary is not None and spectral_summary.size(0) != x.size(0):
#             spectral_summary = None
#             is_physical = False
#         return spectral_summary, bool(is_physical)

#     def _forward_real_batch(
#         self,
#         x: torch.Tensor,
#         spectra: Optional[torch.Tensor],
#         *,
#         classifier_mode: str,
#         return_energy: bool = True,
#         return_parts: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         """Forward real HSI samples with SRGP spectral information when valid."""
#         spectral_summary, spec_is_physical = self._prepare_real_spectral_summary(x, spectra)
#         kwargs = dict(classifier_mode=classifier_mode, return_energy=return_energy)
#         if return_parts:
#             kwargs["return_parts"] = True
#         if spectral_summary is not None:
#             kwargs["spectral_summary"] = spectral_summary
#             kwargs["spectral_summary_is_physical"] = spec_is_physical
#         try:
#             out = self.model(x, **kwargs)
#         except TypeError:
#             # Compatibility with older NECILModel signatures.
#             kwargs.pop("spectral_summary", None)
#             kwargs.pop("spectral_summary_is_physical", None)
#             kwargs.pop("return_parts", None)
#             out = self.model(x, **kwargs)
#         if not isinstance(out, dict):
#             raise RuntimeError("Model forward must return a dict in incremental phase.")
#         out["spectral_summary"] = spectral_summary
#         out["spectral_summary_is_physical"] = spec_is_physical
#         return out

#     # ------------------------------------------------------------------
#     # Incremental trainability
#     # ------------------------------------------------------------------
#     def _set_clean_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
#         """Freeze representation and bank; optionally expose bounded energy calibration only."""
#         del old_class_count
#         for _, p in self.model.named_parameters():
#             p.requires_grad = False

#         # Hard-disable stale paths. The trainer orchestrator should also force these off,
#         # but this mixin is defensive because argparse string booleans can be poisonous.
#         for attr, value in (
#             ("use_bicyc_geometry_cycle", False),
#             ("use_geometry_calibrator", False),
#             ("use_incremental_adapter", False),
#         ):
#             if hasattr(self.model, attr):
#                 setattr(self.model, attr, value)
#         if hasattr(self.model, "disable_incremental_adapter"):
#             self.model.disable_incremental_adapter()
#         if hasattr(self.model, "freeze_incremental_adapter"):
#             self.model.freeze_incremental_adapter()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()
#         if hasattr(self.model, "freeze_projection_head"):
#             self.model.freeze_projection_head()
#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()

#         if self._inc_cfg_bool("allow_incremental_projection_training", False):
#             raise RuntimeError(
#                 "Clean incremental trainer forbids projection/backbone plasticity. "
#                 "Use a separate unsafe ablation if you want to move z-space."
#             )
#         if self._inc_cfg_bool("use_bicyc_geometry_cycle", False):
#             raise RuntimeError("Clean incremental trainer forbids BiCyc/geometry-cycle transport.")
#         if self._inc_cfg_bool("use_mssl_loss", False) and self._inc_cfg_float("mssl_inc_weight", 0.0) > 0.0:
#             raise RuntimeError("Clean incremental trainer forbids MSSL as an incremental solver. Use it only as a base ablation.")

#         use_cal = self._inc_cfg_bool("use_energy_calibrator", False)
#         if hasattr(self.model, "enable_energy_calibration"):
#             self.model.enable_energy_calibration(use_cal, calibrator_type=str(getattr(self.args, "energy_calibrator_type", "old_new")))
#         if use_cal and hasattr(self.model, "unfreeze_energy_calibrator"):
#             self.model.unfreeze_energy_calibrator()

#         params = [p for p in self.model.parameters() if p.requires_grad]
#         allowed = (
#             "energy_calibrator", "old_log_scale", "new_log_scale", "old_bias", "new_bias",
#             "log_scale_raw", "bias_raw",
#         )
#         bad = [name for name, p in self.model.named_parameters() if p.requires_grad and not any(k in name for k in allowed)]
#         if bad:
#             raise RuntimeError(f"Invalid incremental trainable parameters in clean path: {bad[:30]}")
#         return params

#     def _incremental_update_mode(self) -> str:
#         """Return the requested incremental update architecture.

#         ``descriptor_only`` keeps the old clean SRGP/RSGI behavior.
#         ``geometry_gated_adapter`` enables G²RPA: a small residual adapter after
#         canonical z, trained with new real samples and old synthetic replay.
#         """
#         mode = str(getattr(self.args, "incremental_update_mode", "scbgr")).lower().strip()
#         aliases = {
#             "g2rpa": "geometry_gated_adapter",
#             "g2-rpa": "geometry_gated_adapter",
#             "gated_adapter": "geometry_gated_adapter",
#             "geometry_adapter": "geometry_gated_adapter",
#             "adapter": "geometry_gated_adapter",
#             "clean": "scbgr",
#             "rsgi": "scbgr",
#             "descriptor_only": "scbgr",
#             "geometry_state_admission": "scbgr",
#             "spectral_risk_boundary": "scbgr",
#         }
#         return aliases.get(mode, mode)

#     def _adapter_mode_enabled(self) -> bool:
#         return self._incremental_update_mode() == "geometry_gated_adapter"

#     def _set_incremental_trainable_params(self, old_class_count: int) -> List[torch.nn.Parameter]:
#         """Set trainable parameters for the selected incremental architecture.

#         Descriptor-only mode keeps the original strict path.  G²RPA mode freezes
#         backbone/projection/classifier and trains only ``geometry_plastic_adapter``.
#         Old GeometryBank rows are still frozen by the phase entry code.
#         """
#         if not self._adapter_mode_enabled():
#             return self._set_clean_incremental_trainable_params(old_class_count)

#         del old_class_count
#         for _, p in self.model.named_parameters():
#             p.requires_grad = False

#         # Do not enable legacy/stale transport paths.  G²RPA is the only allowed
#         # feature-space plasticity path.
#         for attr, value in (
#             ("use_bicyc_geometry_cycle", False),
#             ("use_geometry_calibrator", False),
#             ("use_incremental_adapter", False),
#         ):
#             if hasattr(self.model, attr):
#                 setattr(self.model, attr, value)
#         if hasattr(self.model, "freeze_projection_head"):
#             self.model.freeze_projection_head()
#         if hasattr(self.model, "freeze_backbone_only"):
#             self.model.freeze_backbone_only()
#         if hasattr(self.model, "freeze_energy_calibrator"):
#             self.model.freeze_energy_calibrator()
#         if hasattr(self.model, "freeze_geometry_calibrator"):
#             self.model.freeze_geometry_calibrator()

#         # The updated NECILModel exposes use_geometry_gated_adapter and
#         # geometry_plastic_adapter.  Fail loudly if the model file was not
#         # updated; otherwise this trainer would silently fall back to no-op
#         # descriptor-only behavior.
#         if not hasattr(self.model, "geometry_plastic_adapter"):
#             raise RuntimeError(
#                 "incremental_update_mode=geometry_gated_adapter requires the updated "
#                 "NECILModel with model.geometry_plastic_adapter."
#             )
#         if hasattr(self.model, "use_geometry_gated_adapter"):
#             self.model.use_geometry_gated_adapter = True
#         if hasattr(self.model, "unfreeze_geometry_plastic_adapter"):
#             self.model.unfreeze_geometry_plastic_adapter()
#         else:
#             for p in self.model.geometry_plastic_adapter.parameters():
#                 p.requires_grad = True

#         params = [p for p in self.model.parameters() if p.requires_grad]
#         bad = [
#             name for name, p in self.model.named_parameters()
#             if p.requires_grad and "geometry_plastic_adapter" not in name
#         ]
#         if bad:
#             raise RuntimeError(f"G²RPA mode allows only geometry_plastic_adapter params, got: {bad[:30]}")
#         if not params:
#             raise RuntimeError("G²RPA mode selected but no adapter parameters are trainable.")
#         return params

#     def _make_g2rpa_loss(self) -> GeometryGatedAdapterLoss:
#         return GeometryGatedAdapterLoss(
#             old_delta_weight=self._inc_cfg_float("adapter_old_delta_weight", 1.0),
#             old_gate_weight=self._inc_cfg_float("adapter_old_gate_weight", 0.75),
#             old_energy_weight=self._inc_cfg_float("adapter_old_energy_weight", 0.25),
#             old_margin_weight=self._inc_cfg_float("adapter_old_margin_weight", 0.25),
#             new_delta_weight=self._inc_cfg_float("adapter_delta_weight", 0.10),
#             new_gate_weight=self._inc_cfg_float("adapter_new_gate_weight", 0.05),
#             new_gate_target=self._inc_cfg_float("adapter_new_gate_target", 0.25),
#             new_gate_max_target=self._inc_cfg_float("adapter_new_gate_max_target", 0.75),
#             margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#             variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
#             reliability_energy_weight=float(getattr(self.args, "reliability_energy_weight", 0.03)),
#             residual_variance_scale=float(getattr(self.args, "residual_variance_scale", 0.75)),
#             normalize_by_dim=bool(getattr(self.args, "energy_normalize_by_dim", True)),
#             use_logdet_energy=bool(getattr(self.args, "use_logdet_energy", True)),
#             logdet_energy_weight=float(getattr(self.args, "logdet_energy_weight", 0.05)),
#             logit_scale=float(getattr(self.args, "loss_scale", 8.0)),
#         )

#     def _compute_g2rpa_adapter_loss(
#         self,
#         *,
#         real_out: Dict[str, torch.Tensor],
#         old_z_base: Optional[torch.Tensor],
#         old_z_adapt: Optional[torch.Tensor],
#         old_y: Optional[torch.Tensor],
#         gate_old: Optional[torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         """Adapter safety/plasticity loss for one incremental batch."""
#         ref = real_out.get("features", None)
#         if not self._adapter_mode_enabled():
#             z = self._zero_like_ref(ref)
#             return {
#                 "total": z, "old_total": z.detach(), "old_delta": z.detach(),
#                 "old_energy": z.detach(), "old_margin": z.detach(), "old_gate": z.detach(),
#                 "old_mean_gate": z.detach(), "old_adapter_acc": z.detach(),
#                 "new_delta": z.detach(), "new_gate": z.detach(), "new_mean_gate": z.detach(),
#             }

#         z_new_adapt = real_out.get("features", None)
#         z_new_base = real_out.get("base_features", real_out.get("pre_adapter_features", None))
#         gate_new = real_out.get("adapter_gate", None)
#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         loss_fn = self._make_g2rpa_loss()
#         return loss_fn(
#             z_old_base=old_z_base,
#             z_old_adapt=old_z_adapt,
#             y_old=old_y,
#             gate_old=gate_old,
#             z_new_base=z_new_base,
#             z_new_adapt=z_new_adapt,
#             gate_new=gate_new,
#             means=bank.get("means", None),
#             bases=bank.get("bases", None),
#             variances=bank.get("variances", None),
#             active_ranks=bank.get("active_ranks", None),
#             reliability=bank.get("reliability", None),
#             sample_counts=bank.get("sample_counts", None),
#         )

#     def _adapt_old_replay_if_needed(
#         self,
#         old_z: Optional[torch.Tensor],
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         """Pass old synthetic replay through adapter in G²RPA mode.

#         Returns (old_z_base, old_z_for_scoring, gate_old).  In descriptor-only
#         mode, old_z_for_scoring is the original replay feature and gate is None.
#         """
#         if old_z is None or not torch.is_tensor(old_z) or old_z.numel() == 0:
#             return old_z, old_z, None
#         old_z_base = old_z.detach()
#         if not self._adapter_mode_enabled():
#             return old_z_base, old_z_base, None
#         if not hasattr(self.model, "adapt_projected_features"):
#             raise RuntimeError("G²RPA mode requires model.adapt_projected_features().")
#         adapted = self.model.adapt_projected_features(old_z_base, force=True, return_delta=True)
#         if isinstance(adapted, dict):
#             old_z_adapt = adapted.get("features", old_z_base)
#             gate_old = adapted.get("gate", None)
#         else:
#             old_z_adapt = adapted
#             gate_old = None
#         if not torch.is_tensor(old_z_adapt) or old_z_adapt.shape != old_z_base.shape:
#             raise RuntimeError("adapt_projected_features must return adapted old replay features with the same shape.")
#         return old_z_base, old_z_adapt, gate_old

#     def _capture_trainable_anchor(self) -> Dict[str, torch.Tensor]:
#         return {name: p.detach().clone() for name, p in self.model.named_parameters() if p.requires_grad}

#     def _trainable_anchor_loss(self, anchor: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
#         weight = float(getattr(self.args, "incremental_weight_anchor", 1e-4))
#         if weight <= 0.0 or not anchor:
#             return self._zero_like_ref(ref)
#         loss = self._zero_like_ref(ref)
#         n = 0
#         for name, p in self.model.named_parameters():
#             if p.requires_grad and name in anchor:
#                 loss = loss + (p - anchor[name].to(p.device)).pow(2).mean()
#                 n += 1
#         return self._zero_like_ref(ref) if n == 0 else weight * loss / float(n)

#     # ------------------------------------------------------------------
#     # Old geometry replay
#     # ------------------------------------------------------------------
#     def _snapshot_old_bank_clean(self, old_class_count: int) -> Dict[str, torch.Tensor]:
#         old_class_count = int(old_class_count)
#         if old_class_count <= 0:
#             return {}
#         if hasattr(self, "_snapshot_old_bank"):
#             snap = self._snapshot_old_bank(old_class_count)
#         elif hasattr(self.model, "get_old_subspace_bank"):
#             snap = self.model.get_old_subspace_bank(old_class_count)
#         else:
#             bank = self.model.get_subspace_bank()
#             snap = {k: v[:old_class_count].detach().clone() for k, v in bank.items() if torch.is_tensor(v) and v.dim() > 0}
#         if hasattr(self, "_canonicalize_bank"):
#             snap = self._canonicalize_bank(snap)

#         for key in ("means", "bases", "variances", "sample_counts"):
#             if key not in snap or not torch.is_tensor(snap[key]) or snap[key].numel() == 0:
#                 raise RuntimeError(f"Old GeometryBank snapshot missing required key '{key}'.")
#         counts = snap["sample_counts"][:old_class_count].to(self.device)
#         if bool((counts <= 0).any().item()):
#             bad = (counts <= 0).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
#             raise RuntimeError(f"Old-bank snapshot has invalid old rows: {bad}")
#         return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in snap.items()}

#     def _select_scbgr_boundary_pairs(
#         self,
#         *,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         """Select risky old/new pairs for SCB-GR boundary replay.

#         Pair format is [old_row, new_local_row].  The boundary sampler consumes
#         new-local indices because ``new_means`` is already sliced to the current
#         phase classes.
#         """
#         old_class_count = int(old_class_count)
#         new_ids = [int(c) for c in new_class_ids]
#         device = self.device
#         empty = torch.empty((0, 2), device=device, dtype=torch.long)
#         if old_class_count <= 0 or not new_ids:
#             return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}

#         try:
#             risk_parts = self._old_new_descriptor_risk_matrix(bank, old_class_count, new_ids)
#         except Exception as exc:
#             if bool(getattr(self, "debug", False)):
#                 print(f"[SCB-GR WARN] could not mine old/new risk pairs: {exc}")
#             return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}

#         risk = risk_parts.get("risk", empty.new_zeros((0, 0)))
#         overlap = risk_parts.get("subspace", torch.zeros_like(risk))
#         if not torch.is_tensor(risk) or risk.numel() == 0:
#             return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}
#         risk = torch.nan_to_num(risk.to(device=device).float(), nan=0.0, posinf=1e6, neginf=0.0)
#         overlap = torch.nan_to_num(overlap.to(device=device).float(), nan=0.0, posinf=1e6, neginf=0.0)

#         risk_thr = self._inc_cfg_float("boundary_replay_risk_threshold", self._inc_cfg_float("descriptor_correction_risk_threshold", 0.35))
#         overlap_thr = self._inc_cfg_float("boundary_replay_overlap_threshold", self._inc_cfg_float("descriptor_correction_overlap_threshold", 0.30))
#         max_pairs = self._inc_cfg_int("boundary_replay_max_pairs", 24)
#         max_pairs = int(max(1, max_pairs))

#         active = (risk >= float(risk_thr)) | (overlap >= float(overlap_thr))
#         coords = active.nonzero(as_tuple=False)
#         if coords.numel() == 0:
#             # Still sample the most dangerous pairs.  If we require thresholds
#             # only, the method can silently turn off exactly when the thresholds
#             # are miscalibrated.
#             flat = risk.flatten()
#             k = min(max_pairs, int(flat.numel()))
#             if k <= 0:
#                 return empty, {"boundary_pair_count": 0.0, "boundary_risk_max": 0.0, "boundary_overlap_max": 0.0}
#             _, idx = torch.topk(flat, k=k, largest=True)
#             coords = torch.stack([idx // risk.size(1), idx % risk.size(1)], dim=1)
#         else:
#             score = risk[coords[:, 0], coords[:, 1]] + overlap[coords[:, 0], coords[:, 1]]
#             k = min(max_pairs, int(score.numel()))
#             _, order = torch.topk(score, k=k, largest=True)
#             coords = coords.index_select(0, order)

#         stats = {
#             "boundary_pair_count": float(coords.size(0)),
#             "boundary_risk_max": float(risk.max().detach().cpu().item()),
#             "boundary_overlap_max": float(overlap.max().detach().cpu().item()),
#         }
#         return coords.long(), stats

#     def _sample_old_anchor_batch(
#         self,
#         old_bank_snapshot: Dict[str, torch.Tensor],
#         old_class_count: int,
#         new_class_ids: Optional[Iterable[int]] = None,
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
#         """Sample old anchors for incremental training.

#         SCB-GR must prefer boundary anchors, not generic old replay.  Generic
#         replay protects old class centers; boundary replay protects the old/new
#         interface where your collapse actually happens.
#         """
#         old_class_count = int(old_class_count)
#         if old_class_count <= 0 or not old_bank_snapshot:
#             self._last_boundary_replay_stats = {"boundary_anchor_count": 0.0, "boundary_pair_count": 0.0}
#             return None, None

#         use_boundary = self._inc_cfg_bool("use_boundary_geometry_replay", True)
#         new_ids = [int(c) for c in (new_class_ids or [])]
#         samples_per_pair = self._inc_cfg_int(
#             "boundary_replay_samples_per_pair",
#             self._inc_cfg_int("gfa_samples_per_class", self._inc_cfg_int("component_replay_per_class", 64)),
#         )
#         samples_per_class = self._inc_cfg_int("gfa_samples_per_class", self._inc_cfg_int("component_replay_per_class", 64))
#         var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))

#         means = old_bank_snapshot["means"][:old_class_count].to(self.device)
#         bases = old_bank_snapshot["bases"][:old_class_count].to(self.device)
#         variances = old_bank_snapshot["variances"][:old_class_count].to(self.device)
#         active = old_bank_snapshot.get("active_ranks", None)
#         rel = old_bank_snapshot.get("reliability", None)
#         counts = old_bank_snapshot.get("sample_counts", None)
#         if torch.is_tensor(active):
#             active = active[:old_class_count].to(self.device)
#         if torch.is_tensor(rel):
#             rel = rel[:old_class_count].to(self.device)
#         if torch.is_tensor(counts):
#             counts = counts[:old_class_count].to(self.device)

#         if use_boundary and new_ids:
#             try:
#                 bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#                 if hasattr(self, "_canonicalize_bank"):
#                     bank = self._canonicalize_bank(bank)
#                 ids_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
#                 new_means = bank["means"].to(self.device).index_select(0, ids_t)
#                 new_bases = bank["bases"].to(self.device).index_select(0, ids_t) if torch.is_tensor(bank.get("bases", None)) else None
#                 pairs, pair_stats = self._select_scbgr_boundary_pairs(
#                     bank=bank,
#                     old_class_count=old_class_count,
#                     new_class_ids=new_ids,
#                 )
#                 x_old, y_old, meta = sample_boundary_geometry_features(
#                     means,
#                     bases,
#                     variances,
#                     new_means=new_means,
#                     new_bases=new_bases,
#                     risk_pairs=pairs,
#                     old_active_ranks=active,
#                     old_reliability=rel,
#                     old_sample_counts=counts,
#                     old_class_ids=list(range(old_class_count)),
#                     samples_per_pair=max(1, int(samples_per_pair)),
#                     variance_floor=var_floor,
#                     parallel_scale=self._inc_cfg_float("boundary_replay_parallel_scale", 0.15),
#                     residual_scale=self._inc_cfg_float("boundary_replay_residual_scale", 0.05),
#                     fallback_samples_per_class=max(1, int(samples_per_class)),
#                     return_metadata=True,
#                 )
#                 self._last_boundary_replay_stats = {
#                     **pair_stats,
#                     "boundary_anchor_count": float(meta.get("boundary_anchor_count", torch.tensor(0.0)).detach().cpu().item()) if isinstance(meta, dict) else float(x_old.size(0)),
#                 }
#                 if torch.is_tensor(x_old) and x_old.numel() > 0:
#                     self._last_risk_replay_counts = {int(c): int(samples_per_pair) for c in range(old_class_count)}
#                     return x_old.to(self.device), y_old.to(self.device).long()
#             except Exception as exc:
#                 if bool(getattr(self, "debug", False)):
#                     print(f"[SCB-GR WARN] boundary replay failed; falling back to generic geometry replay: {exc}")

#         # Compatibility fallback.  This is not the preferred SCB-GR path.
#         x_old, y_old = sample_geometry_features(
#             means,
#             bases,
#             variances,
#             active_ranks=active,
#             reliability=rel,
#             sample_counts=counts,
#             samples_per_class=max(1, int(samples_per_class)),
#             variance_floor=var_floor,
#             parallel_scale=float(getattr(self.args, "gfa_parallel_scale", 1.0)),
#             residual_scale=float(getattr(self.args, "gfa_residual_scale", 0.25)),
#             reliability_gated=self._inc_cfg_bool("gfa_reliability_gated", True),
#             skip_invalid_classes=True,
#         )
#         self._last_boundary_replay_stats = {
#             "boundary_pair_count": 0.0,
#             "boundary_anchor_count": float(x_old.size(0)) if torch.is_tensor(x_old) else 0.0,
#             "boundary_risk_max": 0.0,
#             "boundary_overlap_max": 0.0,
#         }
#         self._last_risk_replay_counts = {int(c): samples_per_class for c in range(old_class_count)}
#         if not torch.is_tensor(x_old) or x_old.numel() == 0:
#             return None, None
#         return x_old.to(self.device), y_old.to(self.device).long()

#     @torch.no_grad()
#     def _apply_reliability_gated_admission_to_new_rows(self, new_class_ids: Iterable[int]) -> None:
#         """Shrink/cap newly bootstrapped descriptors before refinement.

#         This uses the fixed GeometryBank admission rule and never touches frozen
#         old rows. It is the code-level link between the base-prepared geometry
#         field and incremental new-row insertion.
#         """
#         if not self._inc_cfg_bool("reliability_gated_admission", True):
#             return
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None or not hasattr(gb, "update_class_geometry"):
#             return
#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         band_importances = bank.get("band_importances", bank.get("band_importance", None))
#         band_reliability = bank.get("band_reliability", None)
#         feature_reliability = bank.get("feature_reliability", bank.get("reliability", None))
#         for cls in [int(c) for c in new_class_ids]:
#             kwargs = dict(
#                 class_id=cls,
#                 mean=bank["means"][cls].detach(),
#                 basis=bank["bases"][cls].detach(),
#                 eigvals=bank["variances"][cls, :-1].detach(),
#                 res_var=bank["variances"][cls, -1].detach(),
#                 reliability=bank.get("reliability", None)[cls].detach() if torch.is_tensor(bank.get("reliability", None)) else None,
#                 active_rank=bank.get("active_ranks", None)[cls].detach() if torch.is_tensor(bank.get("active_ranks", None)) else None,
#                 sample_count=bank.get("sample_counts", None)[cls].detach() if torch.is_tensor(bank.get("sample_counts", None)) else None,
#                 feature_reliability=feature_reliability[cls].detach() if torch.is_tensor(feature_reliability) and feature_reliability.numel() > cls else None,
#                 band_importance=band_importances[cls].detach() if torch.is_tensor(band_importances) and band_importances.dim() == 2 and band_importances.size(0) > cls else None,
#                 band_reliability=band_reliability[cls].detach() if torch.is_tensor(band_reliability) and band_reliability.numel() > cls else None,
#                 allow_frozen_update=False,
#                 reliability_gated_admission=True,
#                 admission_min_gate=self._inc_cfg_float("admission_min_gate", 0.35),
#                 admission_shrink_floor=self._inc_cfg_float("admission_shrink_floor", 0.15),
#                 admission_low_rank_cap=self._inc_cfg_int("admission_low_rank_cap", 2),
#             )
#             gb.update_class_geometry(**kwargs)
#         if hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)


#     def _safe_update_new_descriptor_row(
#         self,
#         cls: int,
#         *,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         variances: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#     ) -> None:
#         """Commit one corrected new descriptor row while preserving SRGP fields."""
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             raise AttributeError("model.geometry_bank is required for descriptor correction.")
#         cls = int(cls)
#         rel = bank.get("reliability", None)
#         feat_rel = bank.get("feature_reliability", rel)
#         active = bank.get("active_ranks", None)
#         counts = bank.get("sample_counts", None)
#         band = bank.get("band_importances", bank.get("band_importance", None))
#         band_rel = bank.get("band_reliability", None)
#         spectral_kwargs: Dict[str, torch.Tensor] = {}
#         for key in (
#             "spectral_curve_means",
#             "spectral_curve_vars",
#             "spectral_curve_d1",
#             "spectral_curve_d2",
#             "spectral_shape_reliability",
#         ):
#             value = bank.get(key, None)
#             if torch.is_tensor(value) and value.size(0) > cls:
#                 spectral_kwargs[key] = value[cls].detach()

#         if hasattr(gb, "apply_refined_feature_rows"):
#             kwargs = dict(
#                 class_ids=[cls],
#                 means=mean.detach().unsqueeze(0),
#                 bases=basis.detach().unsqueeze(0),
#                 eigvals=variances[:-1].detach().unsqueeze(0),
#                 res_vars=variances[-1].detach().view(1),
#                 reliability=rel[cls].detach().view(1) if torch.is_tensor(rel) and rel.numel() > cls else None,
#                 feature_reliability=feat_rel[cls].detach().view(1) if torch.is_tensor(feat_rel) and feat_rel.numel() > cls else None,
#                 active_ranks=active[cls].detach().view(1) if torch.is_tensor(active) and active.numel() > cls else None,
#                 allow_frozen_update=False,
#             )
#             # New GeometryBank versions accept SRGP spectral rows.  Older ones do not.
#             try:
#                 kwargs.update({k: v.unsqueeze(0) if v.dim() > 0 else v.view(1) for k, v in spectral_kwargs.items()})
#                 gb.apply_refined_feature_rows(**kwargs)
#             except TypeError:
#                 for k in list(spectral_kwargs.keys()):
#                     kwargs.pop(k, None)
#                 gb.apply_refined_feature_rows(**kwargs)
#             return

#         if hasattr(gb, "update_class_geometry"):
#             kwargs = dict(
#                 class_id=cls,
#                 mean=mean.detach(),
#                 basis=basis.detach(),
#                 eigvals=variances[:-1].detach(),
#                 res_var=variances[-1].detach(),
#                 reliability=rel[cls].detach() if torch.is_tensor(rel) and rel.numel() > cls else None,
#                 active_rank=active[cls].detach() if torch.is_tensor(active) and active.numel() > cls else None,
#                 sample_count=counts[cls].detach() if torch.is_tensor(counts) and counts.numel() > cls else None,
#                 feature_reliability=feat_rel[cls].detach() if torch.is_tensor(feat_rel) and feat_rel.numel() > cls else None,
#                 band_importance=band[cls].detach() if torch.is_tensor(band) and band.dim() == 2 and band.size(0) > cls else None,
#                 band_reliability=band_rel[cls].detach() if torch.is_tensor(band_rel) and band_rel.numel() > cls else None,
#                 allow_frozen_update=False,
#             )
#             try:
#                 # Map plural bank names to update_class_geometry names when supported.
#                 if "spectral_curve_means" in spectral_kwargs:
#                     kwargs["spectral_curve_mean"] = spectral_kwargs["spectral_curve_means"]
#                 if "spectral_curve_vars" in spectral_kwargs:
#                     kwargs["spectral_curve_var"] = spectral_kwargs["spectral_curve_vars"]
#                 if "spectral_curve_d1" in spectral_kwargs:
#                     kwargs["spectral_curve_d1"] = spectral_kwargs["spectral_curve_d1"]
#                 if "spectral_curve_d2" in spectral_kwargs:
#                     kwargs["spectral_curve_d2"] = spectral_kwargs["spectral_curve_d2"]
#                 if "spectral_shape_reliability" in spectral_kwargs:
#                     kwargs["spectral_shape_reliability"] = spectral_kwargs["spectral_shape_reliability"]
#                 gb.update_class_geometry(**kwargs)
#             except TypeError:
#                 for key in (
#                     "spectral_curve_mean",
#                     "spectral_curve_var",
#                     "spectral_curve_d1",
#                     "spectral_curve_d2",
#                     "spectral_shape_reliability",
#                 ):
#                     kwargs.pop(key, None)
#                 gb.update_class_geometry(**kwargs)
#             return
#         raise AttributeError("GeometryBank must expose apply_refined_feature_rows() or update_class_geometry().")

#     @staticmethod
#     def _basis_overlap_matrix(old_bases: torch.Tensor, new_bases: torch.Tensor) -> torch.Tensor:
#         if old_bases.numel() == 0 or new_bases.numel() == 0:
#             return old_bases.new_zeros((old_bases.size(0), new_bases.size(0)))
#         prod = torch.einsum("odr,ndr->onr", old_bases, new_bases)
#         # The einsum above only compares matching rank indices.  Use full matrix overlap instead.
#         vals = []
#         for i in range(old_bases.size(0)):
#             row = []
#             for j in range(new_bases.size(0)):
#                 m = old_bases[i].transpose(0, 1).matmul(new_bases[j])
#                 denom = float(max(1, min(old_bases.size(-1), new_bases.size(-1))))
#                 row.append(m.pow(2).sum() / denom)
#             vals.append(torch.stack(row))
#         return torch.stack(vals, dim=0).clamp_min(0.0)

#     def _old_new_descriptor_risk_matrix(
#         self,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#     ) -> Dict[str, torch.Tensor]:
#         """Compute old/new conflict using the same SRGP descriptors used by diagnostics."""
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         old_class_count = int(old_class_count)
#         new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=self.device, dtype=torch.long)
#         if old_class_count <= 0 or new_ids.numel() == 0:
#             empty = torch.zeros((0, 0), device=self.device)
#             return {"risk": empty, "subspace": empty, "center": empty, "band": empty, "spectral": empty}

#         means = bank["means"].to(self.device)
#         bases = bank["bases"].to(self.device)
#         old_means = means[:old_class_count]
#         new_means = means.index_select(0, new_ids)
#         old_bases = bases[:old_class_count]
#         new_bases = bases.index_select(0, new_ids)

#         sub = self._basis_overlap_matrix(old_bases, new_bases)
#         dist = torch.cdist(old_means, new_means, p=2)
#         center_margin = self._inc_cfg_float("risk_center_margin", 1.0)
#         center = torch.exp(-dist / max(center_margin, 1e-6))

#         band = torch.zeros_like(sub)
#         band_all = bank.get("band_importances", bank.get("band_importance", None))
#         if torch.is_tensor(band_all) and band_all.dim() == 2 and band_all.size(0) > int(new_ids.max().item()):
#             old_band = F.normalize(band_all[:old_class_count].to(self.device).float(), p=2, dim=1)
#             new_band = F.normalize(band_all.index_select(0, new_ids).to(self.device).float(), p=2, dim=1)
#             band = old_band.matmul(new_band.t()).clamp(0.0, 1.0)

#         spectral = torch.zeros_like(sub)
#         # Prefer bank method if available; otherwise use derivative rows directly.
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "pairwise_spectral_shape_similarity"):
#             try:
#                 spec_all = gb.pairwise_spectral_shape_similarity().to(self.device)
#                 spectral = spec_all[:old_class_count].index_select(1, new_ids).clamp(0.0, 1.0)
#             except Exception:
#                 spectral = torch.zeros_like(sub)
#         elif torch.is_tensor(bank.get("spectral_curve_d1", None)):
#             d1 = bank["spectral_curve_d1"].to(self.device).float()
#             if d1.dim() == 2 and d1.size(0) > int(new_ids.max().item()):
#                 old_d1 = F.normalize(d1[:old_class_count], p=2, dim=1)
#                 new_d1 = F.normalize(d1.index_select(0, new_ids), p=2, dim=1)
#                 spectral = old_d1.matmul(new_d1.t()).clamp(0.0, 1.0)

#         risk = (
#             self._inc_cfg_float("risk_subspace_weight", 1.0) * sub
#             + self._inc_cfg_float("risk_center_weight", 0.50) * center
#             + self._inc_cfg_float("risk_band_weight", 0.15) * band
#             + self._inc_cfg_float("risk_spectral_shape_weight", 0.25) * spectral
#         )
#         return {"risk": risk, "subspace": sub, "center": center, "band": band, "spectral": spectral, "dist": dist}

#     @torch.no_grad()
#     def _apply_risk_aware_descriptor_correction_to_new_rows(
#         self,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#     ) -> Dict[str, float]:
#         """Actively correct high-risk new descriptors before refinement.

#         This is the missing RSGI step: if a new class uses old-class tangent
#         directions, remove those directions from the new basis and push the new
#         center away from the most dangerous old centers.  Old rows are never
#         modified.
#         """
#         if not self._inc_cfg_bool("risk_aware_descriptor_correction", True):
#             return {"active": 0.0, "max_risk_before": 0.0, "max_overlap_before": 0.0}
#         old_class_count = int(old_class_count)
#         new_ids = [int(c) for c in new_class_ids]
#         if old_class_count <= 0 or not new_ids:
#             return {"active": 0.0, "max_risk_before": 0.0, "max_overlap_before": 0.0}

#         bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         risk_parts = self._old_new_descriptor_risk_matrix(bank, old_class_count, new_ids)
#         risk = risk_parts["risk"]
#         sub = risk_parts["subspace"]
#         if risk.numel() == 0:
#             return {"active": 0.0, "max_risk_before": 0.0, "max_overlap_before": 0.0}

#         risk_thr = self._inc_cfg_float("descriptor_correction_risk_threshold", 0.75)
#         overlap_thr = self._inc_cfg_float("descriptor_correction_overlap_threshold", 0.60)
#         basis_strength = self._inc_cfg_float("descriptor_correction_basis_strength", 0.85)
#         mean_push = self._inc_cfg_float("descriptor_correction_mean_push", 0.20)
#         var_shrink = self._inc_cfg_float("descriptor_correction_var_shrink", 0.15)
#         topk = max(1, self._inc_cfg_int("descriptor_correction_topk_old", 3))
#         var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))

#         means = bank["means"].to(self.device)
#         bases = bank["bases"].to(self.device)
#         variances = bank["variances"].to(self.device).clamp_min(var_floor)
#         corrected = 0
#         max_risk_before = float(risk.max().detach().cpu().item())
#         max_overlap_before = float(sub.max().detach().cpu().item())
#         max_risk_after = max_risk_before
#         max_overlap_after = max_overlap_before

#         for j, cls in enumerate(new_ids):
#             col_risk = risk[:, j]
#             col_sub = sub[:, j]
#             do_correct = bool(((col_risk > risk_thr) | (col_sub > overlap_thr)).any().item())
#             if not do_correct:
#                 continue
#             corrected += 1
#             k = min(topk, int(col_risk.numel()))
#             vals, old_idx = torch.topk(col_risk, k=k, largest=True)
#             weights = vals.clamp_min(0.0)
#             if float(weights.sum().detach().item()) <= 1e-12:
#                 weights = torch.ones_like(weights)
#             weights = weights / weights.sum().clamp_min(1e-12)

#             mu = means[cls].detach().clone()
#             U = bases[cls].detach().clone()
#             var = variances[cls].detach().clone()
#             D = int(U.size(0))
#             P = torch.zeros((D, D), device=self.device, dtype=U.dtype)
#             push = torch.zeros((D,), device=self.device, dtype=mu.dtype)
#             for w, oi in zip(weights, old_idx):
#                 Uo = bases[int(oi)].to(self.device)
#                 P = P + w * Uo.matmul(Uo.transpose(0, 1))
#                 direction = mu - means[int(oi)].to(self.device)
#                 direction = direction / direction.norm().clamp_min(1e-12)
#                 push = push + w * direction

#             gate = (float(col_risk.max().detach().item()) - risk_thr) / max(1e-6, 1.5 - risk_thr)
#             gate = float(max(0.0, min(1.0, gate)))
#             U_corr = U - float(basis_strength) * gate * P.matmul(U)
#             q, _ = torch.linalg.qr(U_corr, mode="reduced")
#             q = q[:, : U.size(1)]
#             # Sign-stabilize relative to original inserted basis.
#             sign = torch.where((q * U).sum(dim=0, keepdim=True) < 0, -torch.ones(1, U.size(1), device=q.device), torch.ones(1, U.size(1), device=q.device))
#             q = q * sign
#             mu_corr = mu + float(mean_push) * gate * push / push.norm().clamp_min(1e-12)
#             var_corr = var.clone()
#             var_corr[:-1] = (var_corr[:-1] * (1.0 - float(var_shrink) * gate)).clamp_min(var_floor)
#             var_corr[-1] = (var_corr[-1] * (1.0 - 0.5 * float(var_shrink) * gate)).clamp_min(var_floor)
#             self._safe_update_new_descriptor_row(cls, mean=mu_corr, basis=q, variances=var_corr, bank=bank)

#         if corrected > 0:
#             bank_after = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#             if hasattr(self, "_canonicalize_bank"):
#                 bank_after = self._canonicalize_bank(bank_after)
#             after = self._old_new_descriptor_risk_matrix(bank_after, old_class_count, new_ids)
#             if after["risk"].numel() > 0:
#                 max_risk_after = float(after["risk"].max().detach().cpu().item())
#                 max_overlap_after = float(after["subspace"].max().detach().cpu().item())
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)
#         stats = {
#             "active": float(corrected),
#             "max_risk_before": max_risk_before,
#             "max_overlap_before": max_overlap_before,
#             "max_risk_after": max_risk_after,
#             "max_overlap_after": max_overlap_after,
#         }
#         self._last_descriptor_correction_stats = stats
#         if corrected > 0:
#             print(
#                 "[RSGI Descriptor Correction] "
#                 f"active={corrected} | risk {max_risk_before:.4f}->{max_risk_after:.4f} | "
#                 f"overlap {max_overlap_before:.4f}->{max_overlap_after:.4f}"
#             )
#         return stats

#     @torch.no_grad()
#     def _incremental_risk_report(self, old_class_count: int, new_class_ids: Iterable[int]) -> Dict[str, float]:
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             return {}
#         report: Dict[str, float] = {}
#         try:
#             if hasattr(gb, "geometry_conflict_matrix"):
#                 risk_kwargs = dict(
#                     center_margin=self._inc_cfg_float("risk_center_margin", 1.0),
#                     subspace_weight=self._inc_cfg_float("risk_subspace_weight", 1.0),
#                     band_weight=self._inc_cfg_float("risk_band_weight", 0.15),
#                     spectral_shape_weight=self._inc_cfg_float("risk_spectral_shape_weight", 0.25),
#                     chart_weight=self._inc_cfg_float("risk_chart_weight", 0.0),
#                     reliability_weighted=self._inc_cfg_bool("risk_replay_reliability_weighted", True),
#                 )
#                 try:
#                     risk = gb.geometry_conflict_matrix(**risk_kwargs)
#                 except TypeError:
#                     risk_kwargs.pop("spectral_shape_weight", None)
#                     risk_kwargs.pop("chart_weight", None)
#                     risk = gb.geometry_conflict_matrix(**risk_kwargs)
#                 new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=risk.device, dtype=torch.long)
#                 if risk.numel() > 0 and old_class_count > 0 and new_ids.numel() > 0:
#                     old_new = risk[:int(old_class_count)].index_select(1, new_ids)
#                     report["risk_old_new_mean"] = float(old_new.mean().detach().cpu().item())
#                     report["risk_old_new_max"] = float(old_new.max().detach().cpu().item())
#             if hasattr(gb, "pairwise_subspace_overlap"):
#                 sub = gb.pairwise_subspace_overlap()
#                 new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=sub.device, dtype=torch.long)
#                 if sub.numel() > 0 and old_class_count > 0 and new_ids.numel() > 0:
#                     vals = sub[:int(old_class_count)].index_select(1, new_ids)
#                     report["old_new_subspace_overlap_max"] = float(vals.max().detach().cpu().item())
#                     report["old_new_subspace_overlap_mean"] = float(vals.mean().detach().cpu().item())
#             if hasattr(gb, "pairwise_band_similarity"):
#                 band = gb.pairwise_band_similarity()
#                 new_ids = torch.as_tensor([int(c) for c in new_class_ids], device=band.device, dtype=torch.long)
#                 if band.numel() > 0 and old_class_count > 0 and new_ids.numel() > 0:
#                     vals = band[:int(old_class_count)].index_select(1, new_ids)
#                     report["old_new_band_similarity_max"] = float(vals.max().detach().cpu().item())
#                     report["old_new_band_similarity_mean"] = float(vals.mean().detach().cpu().item())
#         except Exception as exc:
#             if bool(getattr(self, "debug", False)):
#                 print(f"[WARN] incremental risk report failed: {exc}")
#         return report

#     # ------------------------------------------------------------------
#     # Descriptor-only refinement
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _extract_current_phase_feature_cache(self, class_ids: Iterable[int], split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
#         feats: List[torch.Tensor] = []
#         labs: List[torch.Tensor] = []
#         for cls in [int(c) for c in class_ids]:
#             if not hasattr(self, "_extract_backbone_outputs_for_class"):
#                 raise AttributeError("TrainerHelper._extract_backbone_outputs_for_class() is required for descriptor refinement.")
#             out = self._extract_backbone_outputs_for_class(cls, split=split)
#             if not isinstance(out, dict) or "features" not in out:
#                 raise RuntimeError("_extract_backbone_outputs_for_class() must return {'features': tensor}.")
#             z = out["features"].detach().to(self.device)
#             if z.dim() != 2 or z.numel() == 0:
#                 raise RuntimeError(f"Invalid projected features for class {cls}: {tuple(z.shape)}")
#             if not torch.isfinite(z).all():
#                 raise RuntimeError(f"Non-finite projected features for class {cls}.")
#             feats.append(z)
#             labs.append(torch.full((z.size(0),), cls, device=self.device, dtype=torch.long))
#         if not feats:
#             raise RuntimeError("No current-phase features available for descriptor refinement.")
#         return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

#     def _orthonormalize_descriptor_bases(self, raw_basis: torch.Tensor, reference_basis: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if raw_basis.dim() != 3:
#             raise RuntimeError(f"raw_basis must be [K,D,R], got {tuple(raw_basis.shape)}")
#         bases: List[torch.Tensor] = []
#         R = int(raw_basis.size(-1))
#         for k in range(raw_basis.size(0)):
#             q, _ = torch.linalg.qr(raw_basis[k], mode="reduced")
#             q = q[:, :R]
#             # Stabilize sign relative to the inserted descriptor to reduce artificial trust loss.
#             if torch.is_tensor(reference_basis) and reference_basis.shape == raw_basis.shape:
#                 dots = (q * reference_basis[k].to(device=q.device, dtype=q.dtype)).sum(dim=0, keepdim=True)
#                 signs = torch.where(dots < 0, torch.full_like(dots, -1.0), torch.ones_like(dots))
#                 q = q * signs
#             bases.append(q)
#         return torch.stack(bases, dim=0)

#     def _make_refinement_bank(
#         self,
#         base_bank: Dict[str, torch.Tensor],
#         class_ids: List[int],
#         means_new: torch.Tensor,
#         bases_new: torch.Tensor,
#         variances_new: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         bank = {}
#         for key, value in base_bank.items():
#             if torch.is_tensor(value):
#                 bank[key] = value.detach().clone().to(self.device)
#             else:
#                 bank[key] = value
#         if hasattr(self, "_canonicalize_bank"):
#             bank = self._canonicalize_bank(bank)
#         ids = torch.as_tensor(class_ids, device=self.device, dtype=torch.long)
#         bank["means"][ids] = means_new
#         bank["bases"][ids] = bases_new
#         bank["variances"][ids] = variances_new
#         bank["eigvals"] = bank["variances"][:, :-1]
#         bank["res_vars"] = bank["variances"][:, -1]
#         bank["resvars"] = bank["res_vars"]
#         return bank

#     def _score_features_with_bank(
#         self,
#         features: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#         old_class_count: int,
#         *,
#         return_parts: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.model, "classifier"):
#             raise AttributeError("Model must expose GeometryEnergyClassifier as model.classifier.")
#         out = self.model.classifier(
#             features,
#             geometry_bank=bank,
#             mode="geometry_only",
#             old_class_count=int(old_class_count),
#             return_energy=True,
#             return_parts=return_parts,
#         )
#         if not isinstance(out, dict) or "logits" not in out or "energy" not in out:
#             raise RuntimeError("GeometryEnergyClassifier must return dict with logits and energy for descriptor refinement.")
#         return out

#     @torch.no_grad()
#     def _commit_refined_descriptors(
#         self,
#         class_ids: List[int],
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         init_bank: Dict[str, torch.Tensor],
#     ) -> None:
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is None:
#             raise AttributeError("model.geometry_bank is required to commit refined descriptors.")
#         class_ids = [int(c) for c in class_ids]
#         ids = torch.as_tensor(class_ids, device=self.device, dtype=torch.long)
#         reliability = init_bank.get("reliability", None)
#         feature_reliability = init_bank.get("feature_reliability", reliability)
#         active_ranks = init_bank.get("active_ranks", None)

#         if hasattr(gb, "apply_refined_feature_rows"):
#             gb.apply_refined_feature_rows(
#                 class_ids,
#                 means=means.detach(),
#                 bases=bases.detach(),
#                 eigvals=variances[:, :-1].detach(),
#                 res_vars=variances[:, -1].detach(),
#                 reliability=reliability[ids].detach() if torch.is_tensor(reliability) else None,
#                 feature_reliability=feature_reliability[ids].detach() if torch.is_tensor(feature_reliability) else None,
#                 active_ranks=active_ranks[ids].detach() if torch.is_tensor(active_ranks) else None,
#                 allow_frozen_update=False,
#             )
#             return

#         # Fallback for older GeometryBank versions.
#         band_importances = init_bank.get("band_importances", init_bank.get("band_importance", None))
#         sample_counts = init_bank.get("sample_counts", None)
#         for i, cls in enumerate(class_ids):
#             kwargs = dict(
#                 class_id=cls,
#                 mean=means[i].detach(),
#                 basis=bases[i].detach(),
#                 eigvals=variances[i, :-1].detach(),
#                 res_var=variances[i, -1].detach(),
#                 reliability=reliability[cls].detach() if torch.is_tensor(reliability) and reliability.numel() > cls else None,
#                 active_rank=active_ranks[cls].detach() if torch.is_tensor(active_ranks) and active_ranks.numel() > cls else None,
#                 sample_count=sample_counts[cls].detach() if torch.is_tensor(sample_counts) and sample_counts.numel() > cls else None,
#                 feature_reliability=feature_reliability[cls].detach() if torch.is_tensor(feature_reliability) and feature_reliability.numel() > cls else None,
#                 band_importance=band_importances[cls].detach() if torch.is_tensor(band_importances) and band_importances.dim() > 1 and band_importances.size(0) > cls else None,
#                 allow_frozen_update=False,
#             )
#             if hasattr(gb, "update_class_geometry"):
#                 gb.update_class_geometry(**kwargs)
#             elif hasattr(gb, "update_class"):
#                 kwargs["cls_id"] = kwargs.pop("class_id")
#                 gb.update_class(**kwargs)
#             else:
#                 raise AttributeError("GeometryBank must expose update_class_geometry(), update_class(), or apply_refined_feature_rows().")


#     def _risk_weighted_subspace_separation_loss(
#         self,
#         *,
#         old_means: torch.Tensor,
#         old_bases: torch.Tensor,
#         new_means: torch.Tensor,
#         new_bases: torch.Tensor,
#         old_reliability: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Differentiable high-risk old/new subspace separation.

#         This directly attacks failures such as Gray roof -> Bare soil where a new
#         class reuses old tangent directions.  Risk is detached as a weighting
#         signal; the overlap term remains differentiable w.r.t. new bases/means.
#         """
#         if old_bases.numel() == 0 or new_bases.numel() == 0:
#             z = self._zero_like_ref(new_bases)
#             return z, z
#         rows = []
#         for i in range(old_bases.size(0)):
#             vals = []
#             for j in range(new_bases.size(0)):
#                 m = old_bases[i].transpose(0, 1).matmul(new_bases[j])
#                 denom = float(max(1, min(old_bases.size(-1), new_bases.size(-1))))
#                 vals.append(m.pow(2).sum() / denom)
#             rows.append(torch.stack(vals))
#         overlap = torch.stack(rows, dim=0).clamp_min(0.0)
#         center = torch.exp(-torch.cdist(old_means, new_means, p=2) / max(self._inc_cfg_float("risk_center_margin", 1.0), 1e-6))
#         risk = (self._inc_cfg_float("risk_subspace_weight", 1.0) * overlap.detach()
#                 + self._inc_cfg_float("risk_center_weight", 0.50) * center.detach())
#         if torch.is_tensor(old_reliability) and old_reliability.numel() == old_bases.size(0):
#             risk = risk * old_reliability.view(-1, 1).to(risk.device).clamp(0.05, 1.0)
#         target = self._inc_cfg_float("risk_sep_overlap_target", self._inc_cfg_float("descriptor_overlap_target", 0.35))
#         active = (risk > self._inc_cfg_float("risk_sep_active_threshold", 0.50)).float()
#         weights = (risk * active).detach()
#         if float(weights.sum().detach().item()) <= 1e-12:
#             return self._zero_like_ref(new_bases), active.sum()
#         loss = (weights * F.relu(overlap - target).pow(2)).sum() / weights.sum().clamp_min(1e-12)
#         return loss, active.sum()

#     def _empty_descriptor_stats(self) -> Dict[str, float]:
#         return {
#             "loss": 0.0,
#             "ce": 0.0,
#             "margin": 0.0,
#             "invasion": 0.0,
#             "trust": 0.0,
#             "subspace_collision": 0.0,
#             "center_collision": 0.0,
#             "volume": 0.0,
#             "risk_sep": 0.0,
#             "risk_active_pairs": 0.0,
#             "risk_old_new_max": 0.0,
#             "admission": 0.0,
#             "admission_safe": 0.0,
#             "admission_new_violation_rate": 0.0,
#             "admission_old_boundary_violation_rate": 0.0,
#             "boundary_anchor_count": 0.0,
#             "boundary_pair_count": 0.0,
#             "mean_shift": 0.0,
#             "basis_shift": 0.0,
#             "logvar_shift": 0.0,
#             "anchor_count": 0.0,
#             "steps": 0.0,
#         }

#     def _prepare_descriptor_refinement_state(
#         self,
#         *,
#         new_class_ids: Iterable[int],
#         seen_classes: Iterable[int],
#     ) -> Optional[Dict[str, torch.Tensor | List[int] | optim.Optimizer]]:
#         """Create persistent descriptor parameters for epoch-driven incremental training.

#         The inserted new rows define the trust-region origin. The same descriptor
#         parameters are optimized across epochs, so ``epochs_inc`` now corresponds
#         to real incremental optimization epochs rather than a one-shot pre-loop
#         refinement block.
#         """
#         if not self._inc_cfg_bool("refine_new_descriptors", True):
#             return None

#         new_class_ids = [int(c) for c in new_class_ids]
#         seen_classes = [int(c) for c in seen_classes]
#         if not new_class_ids:
#             return None

#         bank0 = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_canonicalize_bank"):
#             bank0 = self._canonicalize_bank(bank0)
#         if hasattr(self, "_validate_bank_has_classes"):
#             self._validate_bank_has_classes(bank0, seen_classes)

#         ids = torch.as_tensor(new_class_ids, device=self.device, dtype=torch.long)
#         means0 = bank0["means"].detach().to(self.device)
#         bases0 = bank0["bases"].detach().to(self.device)
#         vars0 = bank0["variances"].detach().to(self.device).clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4)))
#         counts0 = bank0["sample_counts"].detach().to(self.device)
#         if bool((counts0[ids] <= 0).any().item()):
#             bad = ids[counts0[ids] <= 0].detach().cpu().tolist()
#             raise RuntimeError(f"Cannot refine unbuilt new GeometryBank rows: {bad}")

#         z_new, y_new = self._extract_current_phase_feature_cache(new_class_ids, split="train")
#         z_new = z_new.detach()
#         y_new = y_new.detach()

#         init_means = means0[ids].detach().clone()
#         init_bases = bases0[ids].detach().clone()
#         init_vars = vars0[ids].detach().clone()
#         init_log_vars = init_vars.log()

#         mu = torch.nn.Parameter(init_means.clone())
#         raw_basis = torch.nn.Parameter(init_bases.clone())
#         log_vars = torch.nn.Parameter(init_log_vars.clone())
#         optimizer = optim.Adam(
#             [mu, raw_basis, log_vars],
#             lr=self._inc_cfg_float("descriptor_refine_lr", 1e-3),
#             weight_decay=0.0,
#         )

#         return {
#             "new_class_ids": new_class_ids,
#             "seen_classes": seen_classes,
#             "bank0": bank0,
#             "valid_mask": counts0 > 0,
#             "z_new": z_new,
#             "y_new": y_new,
#             "mu": mu,
#             "raw_basis": raw_basis,
#             "log_vars": log_vars,
#             "init_means": init_means,
#             "init_bases": init_bases,
#             "init_log_vars": init_log_vars,
#             "optimizer": optimizer,
#         }

#     def _descriptor_refinement_epoch(
#         self,
#         *,
#         state: Optional[Dict[str, object]],
#         phase: int,
#         old_class_count: int,
#         old_bank_snapshot: Dict[str, torch.Tensor],
#         steps_per_epoch: int,
#     ) -> Dict[str, float]:
#         """Run one real incremental epoch over descriptor parameters only."""
#         if state is None:
#             return self._empty_descriptor_stats()

#         steps = int(max(steps_per_epoch, 0))
#         if steps <= 0:
#             return self._empty_descriptor_stats()

#         old_ids = list(range(int(old_class_count)))
#         old_integrity = self._old_bank_integrity_snapshot(old_ids) if hasattr(self, "_old_bank_integrity_snapshot") else None

#         new_class_ids = [int(c) for c in state["new_class_ids"]]  # type: ignore[index]
#         seen_classes = [int(c) for c in state["seen_classes"]]  # type: ignore[index]
#         bank0 = state["bank0"]  # type: ignore[assignment]
#         valid_mask = state["valid_mask"]  # type: ignore[assignment]
#         z_new = state["z_new"]  # type: ignore[assignment]
#         y_new = state["y_new"]  # type: ignore[assignment]
#         mu = state["mu"]  # type: ignore[assignment]
#         raw_basis = state["raw_basis"]  # type: ignore[assignment]
#         log_vars = state["log_vars"]  # type: ignore[assignment]
#         init_means = state["init_means"]  # type: ignore[assignment]
#         init_bases = state["init_bases"]  # type: ignore[assignment]
#         init_log_vars = state["init_log_vars"]  # type: ignore[assignment]
#         optimizer = state["optimizer"]  # type: ignore[assignment]

#         assert isinstance(bank0, dict)
#         assert torch.is_tensor(valid_mask)
#         assert torch.is_tensor(z_new) and torch.is_tensor(y_new)
#         assert isinstance(mu, torch.nn.Parameter)
#         assert isinstance(raw_basis, torch.nn.Parameter)
#         assert isinstance(log_vars, torch.nn.Parameter)
#         assert torch.is_tensor(init_means) and torch.is_tensor(init_bases) and torch.is_tensor(init_log_vars)
#         assert isinstance(optimizer, optim.Optimizer)

#         trust_w = self._inc_cfg_float("descriptor_trust_weight", 1.0)
#         margin_w = self._inc_cfg_float("geometry_energy_margin_weight", float(getattr(self.args, "geometry_energy_margin_weight", 0.25)))
#         invasion_w = self._inc_cfg_float("old_new_invasion_weight", float(getattr(self.args, "old_new_invasion_weight", 0.35)))
#         subspace_w = self._inc_cfg_float("descriptor_subspace_collision_weight", 0.20)
#         center_w = self._inc_cfg_float("descriptor_center_collision_weight", 0.05)
#         volume_w = self._inc_cfg_float("descriptor_volume_control_weight", 0.03)
#         risk_sep_w = self._inc_cfg_float("risk_sep_weight", 0.30)
#         max_mean_shift = self._inc_cfg_float("descriptor_refine_max_mean_shift", 0.35)
#         max_logvar_shift = self._inc_cfg_float("descriptor_refine_max_logvar_shift", 0.75)
#         var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))

#         stat = self._empty_descriptor_stats()
#         for _ in range(steps):
#             optimizer.zero_grad(set_to_none=True)

#             bases_new = self._orthonormalize_descriptor_bases(raw_basis, init_bases)
#             vars_new = log_vars.exp().clamp_min(var_floor)
#             tmp_bank = self._make_refinement_bank(bank0, new_class_ids, mu, bases_new, vars_new)

#             old_z, old_y = self._sample_old_anchor_batch(old_bank_snapshot, old_class_count, new_class_ids)
#             if old_z is not None and old_y is not None and old_z.numel() > 0:
#                 z_joint = torch.cat([z_new, old_z.detach()], dim=0)
#                 y_joint = torch.cat([y_new, old_y.detach()], dim=0)
#                 anchor_count = float(old_y.numel())
#             else:
#                 z_joint = z_new
#                 y_joint = y_new
#                 anchor_count = 0.0

#             out = self._score_features_with_bank(z_joint, tmp_bank, old_class_count, return_parts=False)
#             energy = out["energy"]

#             role_new = torch.zeros((z_new.size(0),), device=self.device, dtype=torch.long)
#             if old_z is not None and old_y is not None and old_z.numel() > 0:
#                 role_old = torch.ones((old_y.numel(),), device=self.device, dtype=torch.long)
#                 batch_role = torch.cat([role_new, role_old], dim=0)
#             else:
#                 batch_role = role_new

#             old_bases = bank0["bases"][:int(old_class_count)].to(self.device)
#             old_vars = bank0["variances"][:int(old_class_count)].to(self.device)
#             old_active = bank0.get("active_ranks", None)
#             new_active = bank0.get("active_ranks", None)
#             old_reliability = bank0.get("reliability", None)
#             if torch.is_tensor(old_active):
#                 old_active = old_active[:int(old_class_count)].to(self.device)
#             if torch.is_tensor(new_active):
#                 ids_t = torch.as_tensor(new_class_ids, device=self.device, dtype=torch.long)
#                 new_active = new_active.index_select(0, ids_t).to(self.device)
#             if torch.is_tensor(old_reliability):
#                 old_reliability = old_reliability[:int(old_class_count)].to(self.device)

#             sample_counts = tmp_bank.get("sample_counts", valid_mask)
#             loss_out = unified_spectral_geometry_loss(
#                 phase="incremental",
#                 energy=energy,
#                 labels=y_joint,
#                 sample_counts=sample_counts,
#                 old_class_count=int(old_class_count),
#                 batch_role=batch_role,
#                 features=z_joint,
#                 old_bases=old_bases,
#                 new_bases=bases_new,
#                 old_active_ranks=old_active,
#                 new_active_ranks=new_active,
#                 reliability=old_reliability,
#                 new_means=mu,
#                 new_variances=vars_new,
#                 init_new_means=init_means,
#                 init_new_bases=init_bases,
#                 init_new_variances=init_log_vars.exp().clamp_min(var_floor),
#                 reference_old_variances=old_vars,
#                 reference_old_active_ranks=old_active,
#                 ce_weight=1.0,
#                 rank_weight=self._inc_cfg_float("unified_rank_weight", margin_w),
#                 admission_weight=self._inc_cfg_float("unified_admission_weight", invasion_w),
#                 subspace_weight=self._inc_cfg_float("unified_subspace_weight", max(subspace_w, risk_sep_w)),
#                 volume_weight=self._inc_cfg_float("unified_volume_weight", volume_w),
#                 trust_weight=self._inc_cfg_float("unified_trust_weight", trust_w),
#                 logit_scale=float(getattr(self.args, "loss_scale", 8.0)),
#                 label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#                 rank_margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                 admission_margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                 target_overlap=self._inc_cfg_float("descriptor_overlap_target", 0.35),
#                 spectral_margin_strength=self._inc_cfg_float("spectral_margin_strength", 0.20),
#                 return_parts=True,
#             )

#             loss = loss_out["total"]
#             ce = loss_out.get("ce", self._zero_like_ref(loss))
#             margin = loss_out.get("rank", self._zero_like_ref(loss))
#             invasion = loss_out.get("admission", self._zero_like_ref(loss))
#             trust = loss_out.get("trust", self._zero_like_ref(loss))
#             subspace_collision = loss_out.get("subspace", self._zero_like_ref(loss))
#             center_collision = self._zero_like_ref(loss)
#             volume = loss_out.get("volume", self._zero_like_ref(loss))
#             risk_sep = subspace_collision
#             risk_active_pairs = loss_out.get("subspace_pair_count", self._zero_like_ref(loss))

#             if not torch.isfinite(loss):
#                 raise RuntimeError("Descriptor epoch produced non-finite loss.")
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_([mu, raw_basis, log_vars], self._inc_cfg_float("descriptor_refine_grad_clip", 1.0))
#             optimizer.step()

#             # Hard global trust-region projection around the originally inserted
#             # descriptor rows, not around the previous epoch. This avoids gradual
#             # descriptor drift across many epochs.
#             with torch.no_grad():
#                 if max_mean_shift > 0.0:
#                     delta = mu - init_means
#                     norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
#                     scale = (float(max_mean_shift) / norm).clamp(max=1.0)
#                     mu.copy_(init_means + delta * scale)
#                 if max_logvar_shift > 0.0:
#                     log_vars.copy_(torch.max(torch.min(log_vars, init_log_vars + max_logvar_shift), init_log_vars - max_logvar_shift))

#             stat["loss"] += float(loss.detach().item())
#             stat["ce"] += float(ce.detach().item())
#             stat["margin"] += float(margin.detach().item())
#             stat["invasion"] += float(invasion.detach().item())
#             stat["admission"] += float(invasion.detach().item())
#             stat["trust"] += float(trust.detach().item())
#             stat["subspace_collision"] += float(subspace_collision.detach().item())
#             stat["center_collision"] += float(center_collision.detach().item())
#             stat["volume"] += float(volume.detach().item())
#             stat["risk_sep"] += float(risk_sep.detach().item())
#             stat["risk_active_pairs"] += float(risk_active_pairs.detach().item())
#             stat["anchor_count"] += anchor_count
#             stat["boundary_anchor_count"] += float(getattr(self, "_last_boundary_replay_stats", {}).get("boundary_anchor_count", anchor_count))
#             stat["boundary_pair_count"] += float(getattr(self, "_last_boundary_replay_stats", {}).get("boundary_pair_count", 0.0))
#             if isinstance(loss_out, dict):
#                 def _loss_float(key: str, default: float = 0.0) -> float:
#                     v = loss_out.get(key, None)
#                     if torch.is_tensor(v):
#                         return float(v.detach().float().mean().cpu().item())
#                     return float(default)
#                 stat["admission_safe"] += _loss_float("admission_safe", 0.0)
#                 stat["admission_new_violation_rate"] += _loss_float("admission_new_violation_rate", 0.0)
#                 stat["admission_old_boundary_violation_rate"] += _loss_float("admission_old_boundary_violation_rate", 0.0)
#             stat["steps"] += 1.0

#         with torch.no_grad():
#             bases_final = self._orthonormalize_descriptor_bases(raw_basis, init_bases)
#             vars_final = log_vars.exp().clamp_min(var_floor)
#             self._commit_refined_descriptors(new_class_ids, mu.detach(), bases_final.detach(), vars_final.detach(), bank0)

#             denom = max(stat["steps"], 1.0)
#             for k in (
#                 "loss", "ce", "margin", "invasion", "admission", "trust", "subspace_collision", "center_collision", "volume",
#                 "risk_sep", "risk_active_pairs", "anchor_count", "boundary_anchor_count", "boundary_pair_count",
#                 "admission_safe", "admission_new_violation_rate", "admission_old_boundary_violation_rate",
#             ):
#                 stat[k] /= denom
#             stat["mean_shift"] = float((mu.detach() - init_means).norm(dim=1).mean().cpu().item())
#             stat["basis_shift"] = float((bases_final.detach() - init_bases).pow(2).mean().sqrt().cpu().item())
#             stat["logvar_shift"] = float((log_vars.detach() - init_log_vars).abs().mean().cpu().item())
#             risk_report = self._incremental_risk_report(int(old_class_count), new_class_ids)
#             stat["risk_old_new_max"] = float(risk_report.get("risk_old_new_max", 0.0))

#         if old_integrity is not None and hasattr(self, "_assert_old_bank_integrity"):
#             self._assert_old_bank_integrity(old_ids, old_integrity, context=f"phase_{phase}_descriptor_epoch")
#         if hasattr(self.model.geometry_bank, "validate_consistency"):
#             self.model.geometry_bank.validate_consistency(strict=True)
#         return stat

#     def _refine_current_phase_descriptors(
#         self,
#         *,
#         phase: int,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#         seen_classes: Iterable[int],
#         old_bank_snapshot: Dict[str, torch.Tensor],
#     ) -> Dict[str, float]:
#         """Backward-compatible one-shot refinement wrapper."""
#         state = self._prepare_descriptor_refinement_state(new_class_ids=new_class_ids, seen_classes=seen_classes)
#         return self._descriptor_refinement_epoch(
#             state=state,
#             phase=int(phase),
#             old_class_count=int(old_class_count),
#             old_bank_snapshot=old_bank_snapshot,
#             steps_per_epoch=self._inc_cfg_int("descriptor_refine_steps", 50),
#         )

#     # ------------------------------------------------------------------
#     # Optional energy-calibration epoch; no BiCyc/MSSL/projection paths
#     # ------------------------------------------------------------------
#     def _compute_logits_energy_from_features(
#         self,
#         features: torch.Tensor,
#         *,
#         classifier_mode: str,
#         return_parts: bool = False,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         kwargs = dict(
#             classifier_mode=classifier_mode,
#             return_parts=return_parts,
#             spectral_summary=spectral_summary,
#             spectral_summary_is_physical=bool(spectral_summary_is_physical),
#         )
#         if hasattr(self.model, "compute_logits_and_energy_from_features"):
#             try:
#                 out = self.model.compute_logits_and_energy_from_features(features, **kwargs)
#             except TypeError:
#                 kwargs.pop("spectral_summary_is_physical", None)
#                 try:
#                     out = self.model.compute_logits_and_energy_from_features(features, **kwargs)
#                 except TypeError:
#                     kwargs.pop("spectral_summary", None)
#                     out = self.model.compute_logits_and_energy_from_features(features, **kwargs)
#         else:
#             kwargs["return_energy"] = True
#             try:
#                 out = self.model.compute_logits_from_features(features, **kwargs)
#             except TypeError:
#                 kwargs.pop("spectral_summary_is_physical", None)
#                 try:
#                     out = self.model.compute_logits_from_features(features, **kwargs)
#                 except TypeError:
#                     kwargs.pop("spectral_summary", None)
#                     out = self.model.compute_logits_from_features(features, **kwargs)
#         if not isinstance(out, dict):
#             return {"logits": out}
#         return out

#     def _energy_calibration_reg(self, ref: torch.Tensor) -> torch.Tensor:
#         if hasattr(self.model, "energy_calibration_regularization_loss"):
#             reg = self.model.energy_calibration_regularization_loss()
#             if torch.is_tensor(reg):
#                 return reg.to(ref.device)
#         return self._zero_like_ref(ref)

#     def _current_valid_mask_from_bank(self) -> Optional[torch.Tensor]:
#         try:
#             bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#             if hasattr(self, "_canonicalize_bank"):
#                 bank = self._canonicalize_bank(bank)
#         except Exception:
#             return None
#         counts = bank.get("sample_counts", None)
#         if torch.is_tensor(counts) and counts.numel() > 0:
#             return counts.to(self.device).flatten() > 0
#         return None

#     def _train_epoch_incremental(
#         self,
#         loader,
#         optimizer,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#         old_bank_snapshot: Dict[str, torch.Tensor],
#         seen_classes: Iterable[int],
#         trainable_anchor: Optional[Dict[str, torch.Tensor]] = None,
#     ) -> Tuple[float, float]:
#         """Train optional score calibration only. Descriptor refinement is handled separately."""
#         self.model.train()
#         self._set_model_phase_and_old_count(getattr(self.model, "current_phase", 0), old_class_count)
#         classifier_mode = self._classifier_mode()
#         new_class_ids = [int(c) for c in new_class_ids]
#         seen_classes = [int(c) for c in seen_classes]

#         total_loss = 0.0
#         total_correct = 0
#         total_count = 0
#         stat_steps = 0
#         stat_sums = {
#             "ce_new": 0.0,
#             "ce_replay": 0.0,
#             "joint_ce": 0.0,
#             "geom_margin": 0.0,
#             "old_new_invasion": 0.0,
#             "energy_calib_reg": 0.0,
#             "weight_anchor": 0.0,
#             "anchor_count": 0.0,
#             "g2rpa_adapter": 0.0,
#             "g2rpa_old_delta": 0.0,
#             "g2rpa_old_gate": 0.0,
#             "g2rpa_old_mean_gate": 0.0,
#             "g2rpa_old_adapter_acc": 0.0,
#             "g2rpa_new_delta": 0.0,
#             "g2rpa_new_mean_gate": 0.0,
#         }

#         for batch in loader:
#             x, y, spectra, _ = self._unpack_hsi_batch(batch)
#             x = x.float().to(self.device, non_blocking=True)
#             y = y.long().to(self.device, non_blocking=True).view(-1)
#             self._assert_batch_labels_in_classes(y, new_class_ids, f"phase_{getattr(self.model, 'current_phase', -1)}_incremental_train")

#             if optimizer is not None:
#                 optimizer.zero_grad(set_to_none=True)

#             out = self._forward_real_batch(x, spectra, classifier_mode=classifier_mode, return_energy=True)
#             logits_new = self._mask_logits_to_seen_classes(out["logits"], seen_classes)
#             features = out["features"]
#             ce_new = self._stable_ce(logits_new, y)

#             old_z, old_y = self._sample_old_anchor_batch(old_bank_snapshot, old_class_count, new_class_ids)
#             ce_replay = self._zero_like_ref(logits_new)
#             joint_ce = self._zero_like_ref(logits_new)
#             margin = self._zero_like_ref(logits_new)
#             invasion = self._zero_like_ref(logits_new)
#             adapter_parts = self._compute_g2rpa_adapter_loss(
#                 real_out=out, old_z_base=None, old_z_adapt=None, old_y=None, gate_old=None
#             )
#             adapter_loss = adapter_parts["total"]
#             anchor_count = 0
#             valid_mask = self._current_valid_mask_from_bank()
#             unified_energy = out.get("energy", None)
#             unified_labels = y
#             unified_role = torch.zeros_like(y, dtype=torch.long, device=self.device)

#             if old_z is not None and old_y is not None and old_z.numel() > 0:
#                 old_z_base, old_z_score, gate_old = self._adapt_old_replay_if_needed(old_z)
#                 anchor_out = self._compute_logits_energy_from_features(old_z_score, classifier_mode="geometry_only")
#                 anchor_logits = self._mask_logits_to_seen_classes(anchor_out["logits"], seen_classes)
#                 ce_replay = self._stable_ce(anchor_logits, old_y.detach())
#                 joint_features = torch.cat([features, old_z_score], dim=0)
#                 joint_labels = torch.cat([y, old_y.detach()], dim=0)
#                 joint_out = self._compute_logits_energy_from_features(joint_features, classifier_mode="geometry_only")
#                 joint_logits = self._mask_logits_to_seen_classes(joint_out["logits"], seen_classes)
#                 joint_ce = self._stable_ce(joint_logits, joint_labels)
#                 unified_energy = joint_out.get("energy", None)
#                 unified_labels = joint_labels
#                 unified_role = torch.cat([
#                     torch.zeros((features.size(0),), device=self.device, dtype=torch.long),
#                     torch.ones((old_y.numel(),), device=self.device, dtype=torch.long),
#                 ], dim=0)
#                 if torch.is_tensor(joint_out.get("energy", None)):
#                     margin = geometry_energy_margin_loss(
#                         joint_out["energy"],
#                         joint_labels,
#                         margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                         valid_mask=valid_mask,
#                     )
#                     invasion = old_new_invasion_loss(
#                         joint_out["energy"],
#                         joint_labels,
#                         old_class_count=int(old_class_count),
#                         margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                         valid_mask=valid_mask,
#                     )
#                 anchor_count = int(old_y.numel())
#                 adapter_parts = self._compute_g2rpa_adapter_loss(
#                     real_out=out,
#                     old_z_base=old_z_base,
#                     old_z_adapt=old_z_score,
#                     old_y=old_y.detach(),
#                     gate_old=gate_old,
#                 )
#                 adapter_loss = adapter_parts["total"]
#             elif torch.is_tensor(out.get("energy", None)):
#                 margin = geometry_energy_margin_loss(
#                     out["energy"],
#                     y,
#                     margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                     valid_mask=valid_mask,
#                 )
#                 invasion = old_new_invasion_loss(
#                     out["energy"],
#                     y,
#                     old_class_count=int(old_class_count),
#                     margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                     valid_mask=valid_mask,
#                 )

#             unified_loss = ce_new + margin + invasion
#             if torch.is_tensor(unified_energy) and unified_energy.numel() > 0:
#                 try:
#                     cur_bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#                     if hasattr(self, "_canonicalize_bank"):
#                         cur_bank = self._canonicalize_bank(cur_bank)
#                     ids_t = torch.as_tensor(new_class_ids, device=self.device, dtype=torch.long)
#                     old_bases_for_loss = cur_bank["bases"][:int(old_class_count)].to(self.device)
#                     new_bases_for_loss = cur_bank["bases"].to(self.device).index_select(0, ids_t)
#                     old_active_for_loss = cur_bank.get("active_ranks", None)
#                     new_active_for_loss = cur_bank.get("active_ranks", None)
#                     old_rel_for_loss = cur_bank.get("reliability", None)
#                     if torch.is_tensor(old_active_for_loss):
#                         old_active_for_loss = old_active_for_loss[:int(old_class_count)].to(self.device)
#                     if torch.is_tensor(new_active_for_loss):
#                         new_active_for_loss = new_active_for_loss.to(self.device).index_select(0, ids_t)
#                     if torch.is_tensor(old_rel_for_loss):
#                         old_rel_for_loss = old_rel_for_loss[:int(old_class_count)].to(self.device)
#                 except Exception:
#                     cur_bank = {}
#                     old_bases_for_loss = None
#                     new_bases_for_loss = None
#                     old_active_for_loss = None
#                     new_active_for_loss = None
#                     old_rel_for_loss = None

#                 inc_loss_out = unified_spectral_geometry_loss(
#                     phase="incremental",
#                     energy=unified_energy,
#                     labels=unified_labels,
#                     sample_counts=valid_mask,
#                     old_class_count=int(old_class_count),
#                     batch_role=unified_role,
#                     old_bases=old_bases_for_loss,
#                     new_bases=new_bases_for_loss,
#                     old_active_ranks=old_active_for_loss,
#                     new_active_ranks=new_active_for_loss,
#                     reliability=old_rel_for_loss,
#                     ce_weight=1.0,
#                     rank_weight=self._inc_cfg_float("unified_rank_weight", self._inc_cfg_float("geometry_energy_margin_weight", 0.25)),
#                     admission_weight=self._inc_cfg_float("unified_admission_weight", self._inc_cfg_float("old_new_invasion_weight", 0.35)),
#                     subspace_weight=self._inc_cfg_float("unified_subspace_weight", self._inc_cfg_float("descriptor_subspace_collision_weight", 0.20)),
#                     volume_weight=0.0,
#                     trust_weight=0.0,
#                     logit_scale=float(getattr(self.args, "loss_scale", 8.0)),
#                     label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
#                     rank_margin=float(getattr(self.args, "geometry_energy_margin", 0.25)),
#                     admission_margin=float(getattr(self.args, "old_new_geometry_margin", 0.30)),
#                     target_overlap=self._inc_cfg_float("descriptor_overlap_target", 0.35),
#                     return_parts=True,
#                 )
#                 unified_loss = inc_loss_out["total"]
#                 joint_ce = inc_loss_out.get("ce", joint_ce)
#                 margin = inc_loss_out.get("rank", margin)
#                 invasion = inc_loss_out.get("admission", invasion)

#             calib_reg = self._energy_calibration_reg(logits_new)
#             weight_anchor = self._trainable_anchor_loss(trainable_anchor or {}, logits_new)
#             loss = (
#                 unified_loss
#                 + self._inc_cfg_float("energy_calibration_weight", 1e-3) * calib_reg
#                 + self._inc_cfg_float("g2rpa_adapter_weight", 1.0) * adapter_loss
#                 + weight_anchor
#             )
#             if not torch.isfinite(loss):
#                 raise RuntimeError("Non-finite incremental calibration loss.")

#             if optimizer is not None:
#                 loss.backward()
#                 trainable = [p for p in self.model.parameters() if p.requires_grad]
#                 if trainable:
#                     torch.nn.utils.clip_grad_norm_(trainable, float(getattr(self.args, "grad_clip_inc", 0.5)))
#                 optimizer.step()

#             total_loss += float(loss.detach().item())
#             c, n = self._incremental_accuracy_with_count(logits_new.detach(), y.detach(), new_class_ids)
#             total_correct += c
#             total_count += n
#             stat_sums["ce_new"] += float(ce_new.detach().item())
#             stat_sums["ce_replay"] += float(ce_replay.detach().item())
#             stat_sums["joint_ce"] += float(joint_ce.detach().item())
#             stat_sums["geom_margin"] += float(margin.detach().item())
#             stat_sums["old_new_invasion"] += float(invasion.detach().item())
#             stat_sums["energy_calib_reg"] += float(calib_reg.detach().item())
#             stat_sums["weight_anchor"] += float(weight_anchor.detach().item())
#             stat_sums["anchor_count"] += float(anchor_count)
#             stat_sums["g2rpa_adapter"] += float(adapter_loss.detach().item())
#             for key, stat_key in (
#                 ("old_delta", "g2rpa_old_delta"),
#                 ("old_gate", "g2rpa_old_gate"),
#                 ("old_mean_gate", "g2rpa_old_mean_gate"),
#                 ("old_adapter_acc", "g2rpa_old_adapter_acc"),
#                 ("new_delta", "g2rpa_new_delta"),
#                 ("new_mean_gate", "g2rpa_new_mean_gate"),
#             ):
#                 v = adapter_parts.get(key, None) if isinstance(adapter_parts, dict) else None
#                 if torch.is_tensor(v):
#                     stat_sums[stat_key] += float(v.detach().item())
#             stat_steps += 1

#         self._last_incremental_loss_stats = {k: v / max(stat_steps, 1) for k, v in stat_sums.items()}
#         return total_loss / max(stat_steps, 1), 100.0 * total_correct / max(total_count, 1)

#     # ------------------------------------------------------------------
#     # Main phase entry
#     # ------------------------------------------------------------------
#     def _append_history_snapshot(
#         self,
#         history: Dict[str, List[float]],
#         *,
#         train_stats: Dict[str, float],
#         val_stats: Dict[str, float],
#         desc_stats: Dict[str, float],
#         loss_stats: Optional[Dict[str, float]] = None,
#     ) -> None:
#         loss_stats = loss_stats or {}
#         history["train_loss"].append(float(train_stats.get("loss", 0.0)))
#         history["train_acc"].append(float(train_stats.get("acc", 0.0)))
#         history["val_loss"].append(float(val_stats.get("loss", 0.0)))
#         history["val_acc"].append(float(val_stats.get("acc", 0.0)))
#         history["val_old_acc"].append(float(val_stats.get("old_acc", 0.0)))
#         history["val_new_acc"].append(float(val_stats.get("new_acc", 0.0)))
#         history["val_hm"].append(float(val_stats.get("hm", 0.0)))
#         history["desc_refine_loss"].append(float(desc_stats.get("loss", 0.0)))
#         history["desc_refine_ce"].append(float(desc_stats.get("ce", 0.0)))
#         history["desc_refine_margin"].append(float(desc_stats.get("margin", 0.0)))
#         history["desc_refine_invasion"].append(float(desc_stats.get("invasion", 0.0)))
#         history["desc_refine_trust"].append(float(desc_stats.get("trust", 0.0)))
#         history["desc_subspace_collision"].append(float(desc_stats.get("subspace_collision", 0.0)))
#         history["desc_center_collision"].append(float(desc_stats.get("center_collision", 0.0)))
#         history["desc_volume"].append(float(desc_stats.get("volume", 0.0)))
#         history["desc_risk_sep"].append(float(desc_stats.get("risk_sep", 0.0)))
#         history["desc_risk_active_pairs"].append(float(desc_stats.get("risk_active_pairs", 0.0)))
#         history["risk_old_new_max"].append(float(desc_stats.get("risk_old_new_max", 0.0)))
#         if "desc_admission" in history:
#             history["desc_admission"].append(float(desc_stats.get("admission", desc_stats.get("invasion", 0.0))))
#             history["desc_admission_safe"].append(float(desc_stats.get("admission_safe", 0.0)))
#             history["desc_admission_new_violation_rate"].append(float(desc_stats.get("admission_new_violation_rate", 0.0)))
#             history["desc_admission_old_boundary_violation_rate"].append(float(desc_stats.get("admission_old_boundary_violation_rate", 0.0)))
#             history["boundary_anchor_count"].append(float(desc_stats.get("boundary_anchor_count", desc_stats.get("anchor_count", 0.0))))
#             history["boundary_pair_count"].append(float(desc_stats.get("boundary_pair_count", 0.0)))
#         corr = getattr(self, "_last_descriptor_correction_stats", {}) or {}
#         history["descriptor_corrections"].append(float(corr.get("active", 0.0)))
#         history["correction_risk_before"].append(float(corr.get("max_risk_before", 0.0)))
#         history["correction_risk_after"].append(float(corr.get("max_risk_after", 0.0)))
#         history["correction_overlap_before"].append(float(corr.get("max_overlap_before", 0.0)))
#         history["correction_overlap_after"].append(float(corr.get("max_overlap_after", 0.0)))
#         history["desc_mean_shift"].append(float(desc_stats.get("mean_shift", 0.0)))
#         history["desc_basis_shift"].append(float(desc_stats.get("basis_shift", 0.0)))
#         history["desc_logvar_shift"].append(float(desc_stats.get("logvar_shift", 0.0)))
#         history["inc_ce_new"].append(float(loss_stats.get("ce_new", 0.0)))
#         history["inc_ce_replay"].append(float(loss_stats.get("ce_replay", 0.0)))
#         history["inc_joint_ce"].append(float(loss_stats.get("joint_ce", 0.0)))
#         history["inc_geom_margin"].append(float(loss_stats.get("geom_margin", 0.0)))
#         history["inc_old_new_invasion"].append(float(loss_stats.get("old_new_invasion", 0.0)))
#         history["inc_energy_calib_reg"].append(float(loss_stats.get("energy_calib_reg", 0.0)))
#         history["inc_weight_anchor"].append(float(loss_stats.get("weight_anchor", 0.0)))
#         history["inc_anchor_count"].append(float(loss_stats.get("anchor_count", desc_stats.get("anchor_count", 0.0))))
#         history["g2rpa_adapter"].append(float(loss_stats.get("g2rpa_adapter", 0.0)))
#         history["g2rpa_old_delta"].append(float(loss_stats.get("g2rpa_old_delta", 0.0)))
#         history["g2rpa_old_gate"].append(float(loss_stats.get("g2rpa_old_gate", 0.0)))
#         history["g2rpa_old_mean_gate"].append(float(loss_stats.get("g2rpa_old_mean_gate", 0.0)))
#         history["g2rpa_old_adapter_acc"].append(float(loss_stats.get("g2rpa_old_adapter_acc", 0.0)))
#         history["g2rpa_new_delta"].append(float(loss_stats.get("g2rpa_new_delta", 0.0)))
#         history["g2rpa_new_mean_gate"].append(float(loss_stats.get("g2rpa_new_mean_gate", 0.0)))

#     def train_incremental_phase(self, phase, epochs, batch_size: int = 64, lr: float = 1e-4) -> Dict:
#         phase = int(phase)
#         if phase <= 0:
#             raise ValueError("train_incremental_phase() must only be called for phase > 0.")

#         print(f"==== Incremental Phase {phase} | SCB-GR: Unified Boundary-Admitted Geometry Loss ====")
#         self.dataset.start_phase(phase)
#         old_class_count = len(self.dataset.get_classes_up_to_phase(phase - 1))
#         phase_class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         seen_classes = self._seen_classes_for_phase(phase)
#         self._set_model_phase_and_old_count(phase, old_class_count)

#         if hasattr(self.model, "ensure_class_capacity"):
#             self.model.ensure_class_capacity(max(seen_classes) + 1)
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes_up_to"):
#             self.model.geometry_bank.freeze_classes_up_to(old_class_count)

#         old_ids = list(range(old_class_count))
#         old_integrity = self._old_bank_integrity_snapshot(old_ids) if hasattr(self, "_old_bank_integrity_snapshot") else None
#         self._bootstrap_phase_classes(phase, split="train", force_rebuild=True)
#         self._apply_reliability_gated_admission_to_new_rows(phase_class_ids)
#         risk_before = self._incremental_risk_report(old_class_count, phase_class_ids)
#         if risk_before:
#             print("[Incremental Risk Before Correction] " + " | ".join(f"{k}={v:.4f}" for k, v in sorted(risk_before.items())))
#         self._apply_risk_aware_descriptor_correction_to_new_rows(old_class_count, phase_class_ids)
#         if old_integrity is not None and hasattr(self, "_assert_old_bank_integrity"):
#             self._assert_old_bank_integrity(old_ids, old_integrity, context="post_bootstrap_incremental")

#         risk_report = self._incremental_risk_report(old_class_count, phase_class_ids)
#         if risk_report:
#             print("[Incremental Risk After Correction] " + " | ".join(f"{k}={v:.4f}" for k, v in sorted(risk_report.items())))

#         old_bank_snapshot = self._snapshot_old_bank_clean(old_class_count)
#         raw_bank = self._safe_get_subspace_bank(require_ready=True) if hasattr(self, "_safe_get_subspace_bank") else self.model.get_subspace_bank()
#         if hasattr(self, "_validate_bank_has_classes"):
#             self._validate_bank_has_classes(raw_bank, seen_classes)

#         train_loader = self.dataset.get_phase_dataloader(phase, split="train", batch_size=batch_size, shuffle=True)
#         val_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)

#         history = {
#             "train_loss": [],
#             "train_acc": [],
#             "val_loss": [],
#             "val_acc": [],
#             "val_old_acc": [],
#             "val_new_acc": [],
#             "val_hm": [],
#             "desc_refine_loss": [],
#             "desc_refine_ce": [],
#             "desc_refine_margin": [],
#             "desc_refine_invasion": [],
#             "desc_refine_trust": [],
#             "desc_subspace_collision": [],
#             "desc_center_collision": [],
#             "desc_volume": [],
#             "desc_risk_sep": [],
#             "desc_risk_active_pairs": [],
#             "risk_old_new_max": [],
#             "desc_admission": [],
#             "desc_admission_safe": [],
#             "desc_admission_new_violation_rate": [],
#             "desc_admission_old_boundary_violation_rate": [],
#             "boundary_anchor_count": [],
#             "boundary_pair_count": [],
#             "descriptor_corrections": [],
#             "correction_risk_before": [],
#             "correction_risk_after": [],
#             "correction_overlap_before": [],
#             "correction_overlap_after": [],
#             "desc_mean_shift": [],
#             "desc_basis_shift": [],
#             "desc_logvar_shift": [],
#             "inc_ce_new": [],
#             "inc_ce_replay": [],
#             "inc_joint_ce": [],
#             "inc_geom_margin": [],
#             "inc_old_new_invasion": [],
#             "inc_energy_calib_reg": [],
#             "inc_weight_anchor": [],
#             "inc_anchor_count": [],
#             "g2rpa_adapter": [],
#             "g2rpa_old_delta": [],
#             "g2rpa_old_gate": [],
#             "g2rpa_old_mean_gate": [],
#             "g2rpa_old_adapter_acc": [],
#             "g2rpa_new_delta": [],
#             "g2rpa_new_mean_gate": [],
#         }

#         init_val_stats = self._validate_split_metrics(val_loader, old_class_count)
#         init_train_stats = self._validate_split_metrics(train_loader, old_class_count)
#         best_score = self._select_score(init_val_stats, phase)
#         best_state = self._capture_state()
#         print(
#             f"[InitVal] Phase {phase} | TrainAcc: {init_train_stats['acc']:.2f}% | "
#             f"ValAcc: {init_val_stats['acc']:.2f}% | Old: {init_val_stats['old_acc']:.2f}% | "
#             f"New: {init_val_stats['new_acc']:.2f}% | HM: {init_val_stats['hm']:.2f}% | "
#             f"Loss: {init_val_stats['loss']:.4f}"
#         )

#         # Freeze model/backbone/projection and validate that no forbidden neural
#         # path is trainable. Epoch training below optimizes descriptor parameters,
#         # not model weights, so an empty trainable-parameter list is valid.
#         trainable_params = self._set_incremental_trainable_params(old_class_count)
#         if hasattr(self, "_print_trainable_summary"):
#             self._print_trainable_summary(phase)

#         steps_per_epoch = self._inc_cfg_int(
#             "descriptor_refine_steps_per_epoch",
#             self._inc_cfg_int("descriptor_refine_steps", 50),
#         )
#         steps_per_epoch = int(max(0, steps_per_epoch))
#         total_epochs = int(max(epochs, 0))
#         adapter_mode = self._adapter_mode_enabled()
#         desc_state = None if adapter_mode else self._prepare_descriptor_refinement_state(
#             new_class_ids=phase_class_ids,
#             seen_classes=seen_classes,
#         )

#         print(
#             f"[SCB-GR Incremental] phase={phase} | old_classes={old_class_count} | "
#             f"new_classes={phase_class_ids} | seen={seen_classes} | "
#             f"descriptor_refine={self._inc_cfg_bool('refine_new_descriptors', True)} | "
#             f"epochs={total_epochs} | desc_steps/epoch={steps_per_epoch} | "
#             f"boundary_replay={self._inc_cfg_bool('use_boundary_geometry_replay', True)} | "
#             f"boundary_samples/pair={self._inc_cfg_int('boundary_replay_samples_per_pair', 12)} | "
#             f"risk_correction={self._inc_cfg_bool('risk_aware_descriptor_correction', True)} | "
#             f"update_mode={self._incremental_update_mode()} | "
#             f"energy_calibrator={self._inc_cfg_bool('use_energy_calibrator', False)} | "
#             f"model_trainable_params={sum(p.numel() for p in trainable_params):,}"
#         )

#         no_improve = 0
#         if total_epochs <= 0:
#             print(f"[SkipTrain] Phase {phase}: epochs_inc <= 0, evaluation only after new-row bootstrap.")
#         else:
#             for epoch in range(total_epochs):
#                 # In G²RPA mode, train the adapter first with real new samples +
#                 # old synthetic replay, then refine new descriptors against the
#                 # current adapted feature space.  Descriptor-only mode keeps the
#                 # original persistent descriptor state.
#                 loss_stats = getattr(self, "_last_incremental_loss_stats", {})
#                 tr_loss = 0.0
#                 if trainable_params:
#                     optimizer = optim.AdamW(
#                         trainable_params,
#                         lr=float(getattr(self.args, "adapter_lr", lr)) if adapter_mode else float(lr),
#                         weight_decay=float(getattr(self.args, "adapter_weight_decay", 0.0)) if adapter_mode else float(getattr(self.args, "weight_decay", 1e-5)),
#                     )
#                     tr_loss, _ = self._train_epoch_incremental(
#                         train_loader,
#                         optimizer,
#                         old_class_count=old_class_count,
#                         new_class_ids=phase_class_ids,
#                         old_bank_snapshot=old_bank_snapshot,
#                         seen_classes=seen_classes,
#                         trainable_anchor=self._capture_trainable_anchor(),
#                     )
#                     loss_stats = getattr(self, "_last_incremental_loss_stats", {})

#                 epoch_desc_state = desc_state
#                 if adapter_mode and self._inc_cfg_bool("refine_new_descriptors", True):
#                     # Adapter parameters changed this epoch.  Re-cache current
#                     # phase features so descriptor rows are fitted to z_adapt,
#                     # not stale pre-adapter features.
#                     epoch_desc_state = self._prepare_descriptor_refinement_state(
#                         new_class_ids=phase_class_ids,
#                         seen_classes=seen_classes,
#                     )
#                 desc_stats = self._descriptor_refinement_epoch(
#                     state=epoch_desc_state,
#                     phase=phase,
#                     old_class_count=old_class_count,
#                     old_bank_snapshot=old_bank_snapshot,
#                     steps_per_epoch=steps_per_epoch,
#                 )
#                 if trainable_params and desc_stats.get("steps", 0.0) <= 0.0:
#                     desc_stats["loss"] = float(tr_loss)

#                 train_eval_stats = self._validate_split_metrics(train_loader, old_class_count)
#                 val_stats = self._validate_split_metrics(val_loader, old_class_count)
#                 self._append_history_snapshot(
#                     history,
#                     train_stats=train_eval_stats,
#                     val_stats=val_stats,
#                     desc_stats=desc_stats,
#                     loss_stats=loss_stats,
#                 )

#                 cal_state = self.model.energy_calibration_state() if hasattr(self.model, "energy_calibration_state") else {}
#                 print(
#                     f"[IncEpoch] Phase {phase} Ep {epoch + 1:03d}/{total_epochs} | "
#                     f"DescLoss: {desc_stats['loss']:.4f} | CE: {desc_stats['ce']:.4f} | "
#                     f"Margin: {desc_stats['margin']:.4f} | Admit: {desc_stats.get('admission', desc_stats['invasion']):.4f} | "
#                     f"Safe: {desc_stats.get('admission_safe', 0.0):.2f} | OldBViol: {desc_stats.get('admission_old_boundary_violation_rate', 0.0):.3f} | "
#                     f"Trust: {desc_stats['trust']:.6f} | "
#                     f"SubColl: {desc_stats.get('subspace_collision', 0.0):.4f} | "
#                     f"RiskSep: {desc_stats.get('risk_sep', 0.0):.4f} | "
#                     f"CtrColl: {desc_stats.get('center_collision', 0.0):.4f} | "
#                     f"Vol: {desc_stats.get('volume', 0.0):.4f} | Steps: {desc_stats['steps']:.0f} | "
#                     f"TrainAcc: {train_eval_stats['acc']:.2f}% | ValAcc: {val_stats['acc']:.2f}% | "
#                     f"Old: {val_stats['old_acc']:.2f}% | New: {val_stats['new_acc']:.2f}% | HM: {val_stats['hm']:.2f}% | "
#                     f"MeanShift: {desc_stats['mean_shift']:.4f} | BasisShift: {desc_stats['basis_shift']:.4f} | "
#                     f"LogVarShift: {desc_stats['logvar_shift']:.4f} | "
#                     f"G2RPA: {loss_stats.get('g2rpa_adapter', 0.0):.4f} | "
#                     f"GateOld: {loss_stats.get('g2rpa_old_mean_gate', 0.0):.4f} | "
#                     f"GateNew: {loss_stats.get('g2rpa_new_mean_gate', 0.0):.4f} | "
#                     f"OldScale: {float(cal_state.get('old_scale', 1.0)):.4f} | "
#                     f"NewScale: {float(cal_state.get('new_scale', 1.0)):.4f}"
#                 )

#                 score = self._select_score(val_stats, phase)
#                 if score > best_score:
#                     best_score = score
#                     best_state = self._capture_state()
#                     no_improve = 0
#                 else:
#                     no_improve += 1
#                 patience = self._inc_cfg_int("early_stop_patience", 0)
#                 if patience > 0 and no_improve >= patience:
#                     print(f"[EarlyStop] Phase {phase}: no improvement for {no_improve} epochs.")
#                     break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self._set_model_phase_and_old_count(phase, old_class_count)

#         # Finalize the phase without rebuilding old rows. Rebuilding current rows after descriptor
#         # refinement would overwrite the refined descriptors, so keep finalize_incremental_rebuild off.
#         old_integrity = self._old_bank_integrity_snapshot(old_ids) if hasattr(self, "_old_bank_integrity_snapshot") else None
#         if self._inc_cfg_bool("finalize_incremental_rebuild", False):
#             raise RuntimeError(
#                 "finalize_incremental_rebuild=True would overwrite descriptor-refined rows. "
#                 "Keep it False in the clean descriptor-refinement method."
#             )
#         if hasattr(self.dataset, "finalize_phase"):
#             self.dataset.finalize_phase(phase)
#         else:
#             self._finalize_phase_memory(phase, split="train")
#         if old_integrity is not None and hasattr(self, "_assert_old_bank_integrity"):
#             self._assert_old_bank_integrity(old_ids, old_integrity, context="post_finalize_incremental")

#         new_old_count = len(self.dataset.get_classes_up_to_phase(phase))
#         self._set_model_phase_and_old_count(phase, new_old_count)
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes_up_to"):
#             self.model.geometry_bank.freeze_classes_up_to(new_old_count)

#         if bool(getattr(self.args, "save_geometry_diagnostics", True)) and hasattr(self, "diagnose_full_base_geometry"):
#             try:
#                 cumulative_ids = [int(c) for c in self.dataset.get_classes_up_to_phase(phase)]
#                 diag_loader = self.dataset.get_cumulative_dataloader(phase, split="val", batch_size=batch_size, shuffle=False)
#                 phase_diag = self.diagnose_full_base_geometry(
#                     diag_loader,
#                     cumulative_ids,
#                     anchors_per_class=int(getattr(self.args, "geometry_diag_anchors_per_class", 64)),
#                     topk_pairs=int(getattr(self.args, "geometry_diag_topk_pairs", 20)),
#                     topk_bands=int(getattr(self.args, "geometry_diag_topk_bands", 5)),
#                 )
#                 setattr(self, f"_last_phase_{phase}_geometry_diagnostics", phase_diag)
#                 if hasattr(self, "_print_geometry_diagnostics_summary"):
#                     self._print_geometry_diagnostics_summary(phase_diag)
#                 if hasattr(self, "_save_geometry_diagnostics_to_files"):
#                     self._save_geometry_diagnostics_to_files(phase_diag, phase=phase)
#             except Exception as exc:
#                 print(f"[WARN] Phase {phase} geometry diagnostics failed: {exc}")

#         if hasattr(self, "save_checkpoint"):
#             self.save_checkpoint(phase, history)
#         return history
