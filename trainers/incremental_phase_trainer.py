from __future__ import annotations

"""Incremental training for transport-verified HSI factor geometry.

The deployed class model is

    p(z | c) = N(mu_c, L_c L_c^T + Psi_c),

with z = [z_s ; z_p].  The raw ordered-spectrum relation p(h | c) is used
only to construct detached pair-risk margins.

Incremental protocol
--------------------
1. Keep a temporary frozen phase-start observer.
2. Update only the backbone modules explicitly exposed by controlled
   plasticity (or run the frozen-backbone baseline).
3. On current-class support pairs, fit an analytical branchwise similarity
   transform; select its complexity on disjoint current validation pairs.
4. Exactly push forward all aggregate old class rows.
5. Fit provisional new rows from current support samples.
6. Score real current query samples against the provisional cumulative bank.
7. Optimize risk-guided factor-energy separation plus coordinate consistency.
8. Select a checkpoint using current-class validation only.
9. Atomically commit transported old rows and final new rows.

No old real sample, old feature, old spectrum, knowledge-distillation teacher,
trainable transport network, descriptor correction, or geometry replay is used
for gradient optimization.
"""

import copy
import json
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from losses.loss import (
        coordinate_transport_loss,
        incremental_geometry_objective,
        pairwise_directional_invasion_matrix,
        relative_parameter_trust_loss,
    )
except ImportError:  # Standalone generated-file validation.
    from factor_geometry_loss_v2 import (
        coordinate_transport_loss,
        incremental_geometry_objective,
        pairwise_directional_invasion_matrix,
        relative_parameter_trust_loss,
    )


Tensor = torch.Tensor
Row = Dict[str, Tensor]


class IncrementalPhaseTrainer:
    """Phase-t trainer for analytical geometry transport and separation."""

    METHOD_NAME = "Transport-verified spectral-spatial factor geometry"
    CLASSIFICATION_FACTORIZATION = "p(z|c)"
    SPECTRAL_RELATION_FACTORIZATION = "p(h|c)"

    # ------------------------------------------------------------------
    # Configuration and phase contract
    # ------------------------------------------------------------------

    def _inc_value(self, names: Sequence[str], default: Any) -> Any:
        for name in names:
            local = getattr(self, name, None)
            if local is not None:
                return local
            args = getattr(self, "args", None)
            if args is not None and hasattr(args, name):
                value = getattr(args, name)
                if value is not None:
                    return value
        return default

    def _inc_float(self, *names: str, default: float) -> float:
        value = float(self._inc_value(names, default))
        if not math.isfinite(value):
            raise RuntimeError(f"{names[0]} must be finite")
        return value

    def _inc_int(self, *names: str, default: int) -> int:
        value = self._inc_value(names, default)
        if isinstance(value, bool):
            raise RuntimeError(f"{names[0]} must be an integer, not bool")
        integer = int(value)
        if float(value) != float(integer):
            raise RuntimeError(f"{names[0]} must be an integer")
        return integer

    def _inc_bool(self, *names: str, default: bool) -> bool:
        value = self._inc_value(names, default)
        parser = getattr(self, "_parse_bool", None)
        if callable(parser):
            return bool(parser(value, names[0]))
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y", "on"}:
                return True
            if token in {"0", "false", "no", "n", "off"}:
                return False
        raise RuntimeError(f"{names[0]} must be an explicit boolean")

    def _validate_incremental_configuration(
        self,
        *,
        epochs: int,
        lr: float,
    ) -> None:
        if epochs <= 0:
            raise RuntimeError("incremental epochs must be positive")
        if not math.isfinite(float(lr)) or float(lr) <= 0.0:
            raise RuntimeError("incremental learning rate must be positive")
        if self._inc_int("incremental_crossfit_folds", default=3) < 3:
            raise RuntimeError(
                "incremental_crossfit_folds must be at least three so "
                "transport-fit, transform-validation, and query sets are disjoint"
            )
        if self._inc_int("incremental_steps_per_epoch", default=3) <= 0:
            raise RuntimeError("incremental_steps_per_epoch must be positive")
        if self._inc_float("incremental_geometry_weight", default=1.0) <= 0.0:
            raise RuntimeError("incremental_geometry_weight must be positive")
        for name, default in (
            ("incremental_coordinate_weight", 0.10),
            ("incremental_parameter_trust_weight", 0.0),
            ("incremental_base_margin", 0.50),
            ("incremental_risk_strength", 0.50),
        ):
            if self._inc_float(name, default=default) < 0.0:
                raise RuntimeError(f"{name} must be non-negative")
        if self._inc_float("incremental_geometry_temperature", default=0.50) <= 0.0:
            raise RuntimeError("incremental_geometry_temperature must be positive")
        if self._inc_float("incremental_grad_clip", default=5.0) <= 0.0:
            raise RuntimeError("incremental_grad_clip must be positive")
        if self._inc_float("incremental_weight_decay", default=1e-4) < 0.0:
            raise RuntimeError("incremental_weight_decay must be non-negative")

        retired_flags = (
            "use_geometry_replay",
            "use_old_feature_replay",
            "use_coupled_geometry_replay",
            "use_descriptor_refinement",
            "refine_new_descriptors",
            "use_energy_calibrator",
            "use_incremental_adapter",
            "use_geometry_gated_adapter",
            "use_trainable_transport",
            "use_spectral_conditioned_joint_energy",
            "use_pc_stgb",
        )
        active = [
            name for name in retired_flags
            if self._inc_bool(name, default=False)
        ]
        if active:
            raise RuntimeError(
                "retired incremental mechanisms are enabled: " + ", ".join(active)
            )

    def _resolve_phase_classes(
        self,
        phase: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        schedule = getattr(self, "phase_schedule", None)
        if not isinstance(schedule, Mapping):
            contract = self.assert_dataset_contract()
            schedule = contract["schedule"]
        if phase not in schedule or phase <= 0:
            raise RuntimeError(f"invalid incremental phase {phase}")
        old: List[int] = []
        for prior in range(phase):
            old.extend(int(value) for value in schedule[prior])
        new = [int(value) for value in schedule[phase]]
        seen = [*old, *new]
        if len(seen) != len(set(seen)):
            raise RuntimeError("phase schedule contains duplicate class IDs")
        committed = self.model.infer_seen_classes()
        if set(committed) != set(old):
            raise RuntimeError(
                f"phase {phase} requires committed old classes {old}, "
                f"but bank contains {committed}"
            )
        return old, new, seen

    # ------------------------------------------------------------------
    # Current-phase tensor collection
    # ------------------------------------------------------------------

    def _collect_current_phase_tensors(
        self,
        loader: Any,
        allowed_classes: Sequence[int],
        *,
        context: str,
    ) -> Dict[str, Optional[Tensor]]:
        allowed = set(int(value) for value in allowed_classes)
        patch_parts: List[Tensor] = []
        raw_patch_parts: List[Tensor] = []
        raw_center_parts: List[Tensor] = []
        label_parts: List[Tensor] = []
        raw_mode: Optional[str] = None

        for batch in loader:
            unpacked = self._unpack_hsi_batch(batch)
            patch = torch.as_tensor(unpacked[0]).float()
            labels = torch.as_tensor(unpacked[1]).long().flatten()
            if patch.dim() != 4 or patch.size(0) != labels.numel():
                raise RuntimeError(f"{context}: patch/label batch is misaligned")
            raw_patch, raw_center = self._find_raw_inputs(batch, patch)
            mode = "patch" if raw_patch is not None else "center"
            if raw_mode is None:
                raw_mode = mode
            elif raw_mode != mode:
                raise RuntimeError(
                    f"{context}: raw spectral representation changed between batches"
                )

            observed = set(
                int(value)
                for value in torch.unique(labels).detach().cpu().tolist()
            )
            leaked = sorted(observed - allowed)
            if leaked:
                raise RuntimeError(
                    f"{context}: loader exposed forbidden classes {leaked}"
                )

            patch_parts.append(patch.to(self.device))
            label_parts.append(labels.to(self.device))
            if raw_patch is not None:
                raw_patch_parts.append(torch.as_tensor(raw_patch).float().to(self.device))
            else:
                assert raw_center is not None
                raw_center_parts.append(
                    torch.as_tensor(raw_center).float().to(self.device)
                )

        if not patch_parts:
            raise RuntimeError(f"{context}: loader is empty")
        result: Dict[str, Optional[Tensor]] = {
            "patches": torch.cat(patch_parts, dim=0),
            "labels": torch.cat(label_parts, dim=0),
            "raw_spectral_patch": (
                torch.cat(raw_patch_parts, dim=0)
                if raw_mode == "patch"
                else None
            ),
            "raw_center_spectrum": (
                torch.cat(raw_center_parts, dim=0)
                if raw_mode == "center"
                else None
            ),
        }
        labels = result["labels"]
        assert torch.is_tensor(labels)
        observed = set(
            int(value)
            for value in torch.unique(labels).detach().cpu().tolist()
        )
        if observed != allowed:
            raise RuntimeError(
                f"{context}: observed classes {sorted(observed)} do not match "
                f"expected {sorted(allowed)}"
            )
        return result

    @staticmethod
    def _index_optional(value: Optional[Tensor], index: Tensor) -> Optional[Tensor]:
        return None if value is None else value.index_select(0, index)

    def _current_output(
        self,
        data: Mapping[str, Optional[Tensor]],
        index: Tensor,
        *,
        require_grad: bool,
    ) -> Dict[str, Any]:
        patches = data["patches"]
        assert torch.is_tensor(patches)
        context = nullcontext() if require_grad else torch.no_grad()
        with context:
            return self.model.forward_features(
                patches.index_select(0, index),
                raw_spectral_patch=self._index_optional(
                    data.get("raw_spectral_patch"), index
                ),
                raw_center_spectrum=self._index_optional(
                    data.get("raw_center_spectrum"), index
                ),
                deterministic=True,
            )

    @torch.no_grad()
    def _observer_output(
        self,
        phase_context: Any,
        data: Mapping[str, Optional[Tensor]],
        index: Tensor,
    ) -> Dict[str, Any]:
        patches = data["patches"]
        assert torch.is_tensor(patches)
        return self.model.encode_observer(
            phase_context,
            patches.index_select(0, index),
            raw_spectral_patch=self._index_optional(
                data.get("raw_spectral_patch"), index
            ),
            raw_center_spectrum=self._index_optional(
                data.get("raw_center_spectrum"), index
            ),
        )

    # ------------------------------------------------------------------
    # Three-way class-stratified cross-fitting
    # ------------------------------------------------------------------

    def _incremental_folds(
        self,
        labels: Tensor,
        new_classes: Sequence[int],
    ) -> Dict[int, List[Tensor]]:
        fold_count = self._inc_int("incremental_crossfit_folds", default=3)
        seed = self._inc_int("seed", default=0)
        folds: Dict[int, List[Tensor]] = {}
        for class_id in new_classes:
            indices = torch.nonzero(
                labels.eq(int(class_id)), as_tuple=False
            ).flatten()
            if indices.numel() < 3 * fold_count:
                raise RuntimeError(
                    f"class {class_id} has {indices.numel()} training samples; "
                    f"three-way {fold_count}-fold training requires at least "
                    f"{3 * fold_count}"
                )
            generator = torch.Generator().manual_seed(
                seed + 104729 * int(class_id)
            )
            order = torch.randperm(indices.numel(), generator=generator).to(
                indices.device
            )
            parts = list(
                torch.tensor_split(indices.index_select(0, order), fold_count)
            )
            if any(part.numel() < 3 for part in parts):
                raise RuntimeError(
                    f"class {class_id} produced a fold with fewer than three samples"
                )
            folds[int(class_id)] = parts
        return folds

    @staticmethod
    def _fold_role_indices(
        folds: Mapping[int, Sequence[Tensor]],
        fold_index: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        fold_count = len(next(iter(folds.values())))
        query_fold = int(fold_index) % fold_count
        validation_fold = (query_fold + 1) % fold_count
        support_parts: List[Tensor] = []
        validation_parts: List[Tensor] = []
        query_parts: List[Tensor] = []
        for parts in folds.values():
            query_parts.append(parts[query_fold])
            validation_parts.append(parts[validation_fold])
            support_parts.append(
                torch.cat(
                    [
                        part for index, part in enumerate(parts)
                        if index not in {query_fold, validation_fold}
                    ]
                )
            )
        return (
            torch.cat(support_parts),
            torch.cat(validation_parts),
            torch.cat(query_parts),
        )

    # ------------------------------------------------------------------
    # Provisional transport and objective
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fit_transform_for_indices(
        self,
        phase_context: Any,
        data: Mapping[str, Optional[Tensor]],
        support_index: Tensor,
        validation_index: Tensor,
    ) -> Tuple[Any, Dict[str, Any]]:
        previous_support = self._observer_output(
            phase_context, data, support_index
        )
        previous_validation = self._observer_output(
            phase_context, data, validation_index
        )
        current_support = self._current_output(
            data, support_index, require_grad=False
        )
        current_validation = self._current_output(
            data, validation_index, require_grad=False
        )
        return self.model.fit_phase_transform(
            phase_context,
            support_previous_features=previous_support["joint_feature"],
            support_current_features=current_support["joint_feature"],
            validation_previous_features=previous_validation["joint_feature"],
            validation_current_features=current_validation["joint_feature"],
        )

    @torch.no_grad()
    def _build_rows_for_indices(
        self,
        data: Mapping[str, Optional[Tensor]],
        row_index: Tensor,
        *,
        transform: Any,
    ) -> Tuple[Dict[int, Row], Dict[str, Any]]:
        current = self._current_output(data, row_index, require_grad=False)
        labels = data["labels"]
        assert torch.is_tensor(labels)
        return self.model.build_phase_rows(
            transform=transform,
            new_support_features=current["joint_feature"],
            new_support_raw_spectra=current["raw_center_spectrum"],
            new_support_labels=labels.index_select(0, row_index),
            new_support_weights=None,
        )

    def _training_step(
        self,
        *,
        phase_context: Any,
        train_data: Mapping[str, Optional[Tensor]],
        folds: Mapping[int, Sequence[Tensor]],
        fold_index: int,
        optimizer: optim.Optimizer,
        trainable_parameters: Sequence[nn.Parameter],
    ) -> Dict[str, Any]:
        support_index, validation_index, query_index = self._fold_role_indices(
            folds, fold_index
        )
        transform, transform_report = self._fit_transform_for_indices(
            phase_context,
            train_data,
            support_index,
            validation_index,
        )
        row_index = torch.cat([support_index, validation_index])
        temporary_rows, transport_report = self._build_rows_for_indices(
            train_data,
            row_index,
            transform=transform,
        )
        pair_risk = self.model.phase_pair_risk(temporary_rows)

        previous_query = self._observer_output(
            phase_context, train_data, query_index
        )
        current_query = self._current_output(
            train_data, query_index, require_grad=True
        )
        labels = train_data["labels"]
        assert torch.is_tensor(labels)
        query_labels = labels.index_select(0, query_index)

        scored = self.model.compute_logits_from_features(
            current_query["joint_feature"],
            class_ids=self.model.seen_classes,
            temporary_rows=temporary_rows,
            targets=query_labels,
            targets_are_global=True,
            old_classes=self.model.old_classes,
            new_classes=self.model.new_classes,
            return_parts=True,
            return_diagnostics=True,
        )
        coordinate = coordinate_transport_loss(
            previous_query["joint_feature"],
            current_query["joint_feature"],
            spectral_dim=self.model.spectral_dim,
            transform=transform,
            class_targets_global=query_labels,
            class_balanced=True,
            return_parts=True,
        )

        parameter_weight = self._inc_float(
            "incremental_parameter_trust_weight", default=0.0
        )
        parameter_trust: Optional[Mapping[str, Tensor]] = None
        if parameter_weight > 0.0:
            parameter_trust = relative_parameter_trust_loss(
                self.model.backbone,
                phase_context.parameter_snapshot,
                parameter_names=phase_context.parameter_snapshot.keys(),
                return_parts=True,
            )

        objective = incremental_geometry_objective(
            scored["energy"],
            scored["class_ids"],
            query_labels,
            pair_risk=pair_risk,
            coordinate_loss=coordinate,
            parameter_trust_loss=parameter_trust,
            geometry_weight=self._inc_float(
                "incremental_geometry_weight", default=1.0
            ),
            coordinate_weight=self._inc_float(
                "incremental_coordinate_weight", default=0.10
            ),
            parameter_trust_weight=parameter_weight,
            base_margin=self._inc_float(
                "incremental_base_margin", default=0.50
            ),
            risk_strength=self._inc_float(
                "incremental_risk_strength", default=0.50
            ),
            temperature=self._inc_float(
                "incremental_geometry_temperature", default=0.50
            ),
            maximum_margin=(
                None
                if self._inc_value(("incremental_maximum_pair_margin",), None)
                is None
                else self._inc_float(
                    "incremental_maximum_pair_margin", default=1.0
                )
            ),
            class_balanced=True,
            return_parts=True,
        )

        optimizer.zero_grad(set_to_none=True)
        objective["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(trainable_parameters),
            self._inc_float("incremental_grad_clip", default=5.0),
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise RuntimeError("incremental gradient norm is NaN/Inf")
        optimizer.step()

        return {
            "total": float(objective["total"].detach().item()),
            "geometry": float(objective["geometry"].detach().item()),
            "coordinate": float(objective["coordinate"].detach().item()),
            "parameter_trust": float(objective["parameter_trust"].detach().item()),
            "accuracy": 100.0 * float(objective["accuracy"].detach().item()),
            "mean_gap": float(objective["mean_gap"].detach().item()),
            "q05_gap": float(objective["q05_gap"].detach().item()),
            "classification_violation_rate": float(
                objective["classification_violation_rate"].detach().item()
            ),
            "margin_violation_rate": float(
                objective["margin_violation_rate"].detach().item()
            ),
            "spectral_coordinate_rmse": float(
                coordinate["spectral_rmse"].detach().item()
            ),
            "spatial_coordinate_rmse": float(
                coordinate["spatial_rmse"].detach().item()
            ),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().item()),
            "spectral_transport_level": int(
                transform_report["spectral"]["selected_level"]
            ),
            "spatial_transport_level": int(
                transform_report["spatial"]["selected_level"]
            ),
            "spectral_transport_rmse": float(
                transform_report["spectral"]["selected_normalized_rmse"]
            ),
            "spatial_transport_rmse": float(
                transform_report["spatial"]["selected_normalized_rmse"]
            ),
            "transport_closure_error": float(
                transport_report["maximum_closure_error"]
            ),
        }

    # ------------------------------------------------------------------
    # Candidate construction and evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fit_full_candidate(
        self,
        phase_context: Any,
        train_data: Mapping[str, Optional[Tensor]],
        folds: Mapping[int, Sequence[Tensor]],
    ) -> Tuple[Any, Dict[int, Row], Dict[int, Row], Dict[str, Any]]:
        support_index, validation_index, _ = self._fold_role_indices(folds, 0)
        transform, transform_report = self._fit_transform_for_indices(
            phase_context,
            train_data,
            support_index,
            validation_index,
        )
        labels = train_data["labels"]
        assert torch.is_tensor(labels)
        full_index = torch.arange(labels.numel(), device=labels.device)
        temporary_rows, transport_report = self._build_rows_for_indices(
            train_data,
            full_index,
            transform=transform,
        )
        new_rows = {
            int(class_id): temporary_rows[int(class_id)]
            for class_id in self.model.new_classes
        }
        return transform, temporary_rows, new_rows, {
            "transform": transform_report,
            "transport": transport_report,
        }

    @torch.no_grad()
    def _evaluate_loader_with_rows(
        self,
        loader: Any,
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]],
        class_ids: Sequence[int],
        old_classes: Sequence[int],
        new_classes: Sequence[int],
        allowed_classes: Sequence[int],
        context: str,
    ) -> Dict[str, Any]:
        allowed = set(int(value) for value in allowed_classes)
        energy_parts: List[Tensor] = []
        label_parts: List[Tensor] = []
        prediction_parts: List[Tensor] = []
        class_tensor: Optional[Tensor] = None

        previous_mode = bool(self.model.training)
        self.model.eval()
        try:
            for batch in loader:
                unpacked = self._unpack_hsi_batch(batch)
                patches = torch.as_tensor(unpacked[0]).float()
                labels = torch.as_tensor(unpacked[1]).long().flatten()
                raw_patch, raw_center = self._find_raw_inputs(batch, patches)
                observed = set(
                    int(value)
                    for value in torch.unique(labels).detach().cpu().tolist()
                )
                leaked = sorted(observed - allowed)
                if leaked:
                    raise RuntimeError(
                        f"{context}: loader exposed forbidden classes {leaked}"
                    )
                output = self.model.forward_features(
                    patches.to(self.device),
                    raw_spectral_patch=(
                        None
                        if raw_patch is None
                        else torch.as_tensor(raw_patch).float().to(self.device)
                    ),
                    raw_center_spectrum=(
                        None
                        if raw_center is None
                        else torch.as_tensor(raw_center).float().to(self.device)
                    ),
                    deterministic=True,
                )
                labels = labels.to(self.device)
                scored = self.model.compute_logits_from_features(
                    output["joint_feature"],
                    class_ids=class_ids,
                    temporary_rows=rows,
                    targets=labels,
                    targets_are_global=True,
                    old_classes=old_classes,
                    new_classes=new_classes,
                    return_parts=True,
                    return_diagnostics=True,
                )
                class_tensor = scored["class_ids"]
                prediction = class_tensor.index_select(
                    0, scored["energy"].argmin(dim=1)
                )
                energy_parts.append(scored["energy"].detach())
                label_parts.append(labels.detach())
                prediction_parts.append(prediction.detach())
        finally:
            self.model.train(previous_mode)

        if not energy_parts or class_tensor is None:
            raise RuntimeError(f"{context}: loader is empty")
        energy = torch.cat(energy_parts, dim=0)
        labels = torch.cat(label_parts, dim=0)
        predictions = torch.cat(prediction_parts, dim=0)
        mapping = {int(class_id): index for index, class_id in enumerate(class_ids)}
        targets = torch.tensor(
            [mapping[int(value)] for value in labels.detach().cpu().tolist()],
            device=energy.device,
            dtype=torch.long,
        )
        true = energy.gather(1, targets[:, None]).squeeze(1)
        rivals = energy.clone()
        rivals.scatter_(1, targets[:, None], float("inf"))
        gap = rivals.min(dim=1).values - true

        per_class: Dict[int, float] = {}
        for class_id in allowed_classes:
            selected = labels.eq(int(class_id))
            if bool(selected.any()):
                per_class[int(class_id)] = 100.0 * float(
                    predictions[selected].eq(labels[selected]).float().mean().item()
                )

        old_mask = torch.zeros_like(labels, dtype=torch.bool)
        new_mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in old_classes:
            old_mask |= labels.eq(int(class_id))
        for class_id in new_classes:
            new_mask |= labels.eq(int(class_id))

        def subset_accuracy(mask: Tensor) -> float:
            if not bool(mask.any()):
                return float("nan")
            return 100.0 * float(
                predictions[mask].eq(labels[mask]).float().mean().item()
            )

        old_accuracy = subset_accuracy(old_mask)
        new_accuracy = subset_accuracy(new_mask)
        harmonic = (
            0.0
            if not math.isfinite(old_accuracy)
            or not math.isfinite(new_accuracy)
            or old_accuracy + new_accuracy <= 0.0
            else 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)
        )
        old_set = set(int(value) for value in old_classes)
        new_set = set(int(value) for value in new_classes)
        old_to_new = (
            float(
                torch.tensor(
                    [
                        int(int(value) in new_set)
                        for value in predictions[old_mask].detach().cpu().tolist()
                    ],
                    dtype=torch.float32,
                ).mean().item()
            )
            if bool(old_mask.any())
            else float("nan")
        )
        new_to_old = (
            float(
                torch.tensor(
                    [
                        int(int(value) in old_set)
                        for value in predictions[new_mask].detach().cpu().tolist()
                    ],
                    dtype=torch.float32,
                ).mean().item()
            )
            if bool(new_mask.any())
            else float("nan")
        )
        invasion = pairwise_directional_invasion_matrix(
            energy, class_tensor, labels
        )
        return {
            "accuracy": 100.0 * float(predictions.eq(labels).float().mean().item()),
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "old_new_harmonic_mean": harmonic,
            "minimum_per_class_accuracy": min(per_class.values()) if per_class else 0.0,
            "per_class_accuracy": per_class,
            "classification_violation_rate": float(gap.le(0.0).float().mean().item()),
            "mean_gap": float(gap.mean().item()),
            "q05_gap": float(torch.quantile(gap, 0.05).item()),
            "minimum_gap": float(gap.min().item()),
            "old_to_new_invasion_rate": old_to_new,
            "new_to_old_invasion_rate": new_to_old,
            "maximum_directional_invasion": float(
                invasion["maximum_directional_invasion"].item()
            ),
            "sample_count": int(labels.numel()),
            "class_ids": list(int(value) for value in class_ids),
            "targets_global": labels.detach().cpu(),
            "predictions_global": predictions.detach().cpu(),
        }

    @torch.no_grad()
    def _crossfit_certificate(
        self,
        phase_context: Any,
        train_data: Mapping[str, Optional[Tensor]],
        folds: Mapping[int, Sequence[Tensor]],
        seen_classes: Sequence[int],
        old_classes: Sequence[int],
        new_classes: Sequence[int],
    ) -> Dict[str, Any]:
        energy_parts: List[Tensor] = []
        label_parts: List[Tensor] = []
        transform_reports: List[Dict[str, Any]] = []
        fold_count = len(next(iter(folds.values())))
        for fold_index in range(fold_count):
            support_index, validation_index, query_index = self._fold_role_indices(
                folds, fold_index
            )
            transform, transform_report = self._fit_transform_for_indices(
                phase_context, train_data, support_index, validation_index
            )
            temporary_rows, transport_report = self._build_rows_for_indices(
                train_data,
                torch.cat([support_index, validation_index]),
                transform=transform,
            )
            current_query = self._current_output(
                train_data, query_index, require_grad=False
            )
            labels = train_data["labels"]
            assert torch.is_tensor(labels)
            query_labels = labels.index_select(0, query_index)
            scored = self.model.compute_logits_from_features(
                current_query["joint_feature"],
                class_ids=seen_classes,
                temporary_rows=temporary_rows,
                return_parts=True,
            )
            energy_parts.append(scored["energy"].detach())
            label_parts.append(query_labels.detach())
            transform_reports.append(
                {
                    "transform": transform_report,
                    "transport": transport_report,
                }
            )

        energy = torch.cat(energy_parts, dim=0)
        labels = torch.cat(label_parts, dim=0)
        class_tensor = torch.tensor(
            list(seen_classes), device=energy.device, dtype=torch.long
        )
        mapping = {int(value): index for index, value in enumerate(seen_classes)}
        targets = torch.tensor(
            [mapping[int(value)] for value in labels.detach().cpu().tolist()],
            device=energy.device,
            dtype=torch.long,
        )
        predictions = class_tensor.index_select(0, energy.argmin(dim=1))
        true = energy.gather(1, targets[:, None]).squeeze(1)
        rivals = energy.clone()
        rivals.scatter_(1, targets[:, None], float("inf"))
        gap = rivals.min(dim=1).values - true
        old_set = set(int(value) for value in old_classes)
        new_to_old = float(
            torch.tensor(
                [
                    int(int(value) in old_set)
                    for value in predictions.detach().cpu().tolist()
                ],
                dtype=torch.float32,
            ).mean().item()
        )
        per_class = {
            int(class_id): 100.0 * float(
                predictions[labels.eq(int(class_id))]
                .eq(labels[labels.eq(int(class_id))])
                .float()
                .mean()
                .item()
            )
            for class_id in new_classes
        }
        return {
            "protocol": "three_way_stratified_current_class_crossfit",
            "folds": fold_count,
            "accuracy": 100.0 * float(predictions.eq(labels).float().mean().item()),
            "minimum_per_class_accuracy": min(per_class.values()),
            "per_class_accuracy": per_class,
            "classification_violation_rate": float(gap.le(0.0).float().mean().item()),
            "mean_gap": float(gap.mean().item()),
            "q05_gap": float(torch.quantile(gap, 0.05).item()),
            "minimum_gap": float(gap.min().item()),
            "new_to_old_invasion_rate": new_to_old,
            "query_count": int(labels.numel()),
            "transport_reports": transform_reports,
            "uses_old_validation_for_selection": False,
        }

    # ------------------------------------------------------------------
    # Accepted-state rollback and admission
    # ------------------------------------------------------------------

    def _capture_accepted_state(self) -> Dict[str, Any]:
        return {
            "backbone": copy.deepcopy(self.model.backbone.state_dict()),
            "bank": self.model.geometry_bank.export_snapshot(),
            "classifier": copy.deepcopy(self.model.classifier.state_dict()),
            "current_phase": int(self.model.current_phase),
            "phase_mode": str(self.model.phase_mode),
            "seen_classes": list(self.model.seen_classes),
            "old_classes": list(self.model.old_classes),
            "new_classes": list(self.model.new_classes),
            "phase_old_digest": self.model.phase_old_digest,
            "base_handoff": copy.deepcopy(self.model.base_handoff),
        }

    @torch.no_grad()
    def _restore_accepted_state(self, state: Mapping[str, Any]) -> None:
        self.model.geometry_bank.load_snapshot(state["bank"], strict=True)
        self.model.backbone.load_state_dict(state["backbone"], strict=True)
        self.model.classifier.load_state_dict(state["classifier"], strict=True)
        self.model.current_phase = int(state["current_phase"])
        self.model.phase_mode = str(state["phase_mode"])
        self.model.seen_classes = list(state["seen_classes"])
        self.model.old_classes = list(state["old_classes"])
        self.model.new_classes = list(state["new_classes"])
        self.model.phase_old_digest = state["phase_old_digest"]
        self.model.base_handoff = copy.deepcopy(state["base_handoff"])
        self.model.backbone.freeze_all()
        self.model.eval()

    def _incremental_admission(
        self,
        *,
        phase: int,
        crossfit: Mapping[str, Any],
        current_validation: Mapping[str, Any],
        candidate_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        transform = candidate_report["transform"]
        transport = candidate_report["transport"]
        maximum_transform_rmse = max(
            float(transform["spectral"]["selected_normalized_rmse"]),
            float(transform["spatial"]["selected_normalized_rmse"]),
        )
        checks = {
            "crossfit_accuracy": float(crossfit["accuracy"])
            >= self._inc_float("incremental_min_crossfit_accuracy", default=0.0),
            "crossfit_minimum_class_accuracy": float(
                crossfit["minimum_per_class_accuracy"]
            )
            >= self._inc_float(
                "incremental_min_crossfit_min_class_accuracy", default=0.0
            ),
            "crossfit_classification_violation": float(
                crossfit["classification_violation_rate"]
            )
            <= self._inc_float(
                "incremental_max_crossfit_classification_violation", default=1.0
            ),
            "current_validation_accuracy": float(current_validation["accuracy"])
            >= self._inc_float("incremental_min_current_val_accuracy", default=0.0),
            "current_validation_minimum_class_accuracy": float(
                current_validation["minimum_per_class_accuracy"]
            )
            >= self._inc_float(
                "incremental_min_current_val_min_class_accuracy", default=0.0
            ),
            "new_to_old_invasion": float(
                current_validation["new_to_old_invasion_rate"]
            )
            <= self._inc_float("incremental_max_new_to_old_invasion", default=1.0),
            "transport_validation_error": maximum_transform_rmse
            <= self._inc_float(
                "incremental_max_transport_normalized_rmse", default=1.50
            ),
            "factor_transport_closure": float(
                transport["maximum_closure_error"]
            )
            <= self._inc_float(
                "incremental_max_transport_closure_error", default=1e-5
            ),
        }
        enforce = self._inc_bool("incremental_admission_enforce", default=False)
        structural = checks["factor_transport_closure"] and math.isfinite(
            maximum_transform_rmse
        )
        valid = structural and (all(checks.values()) if enforce else True)
        return {
            "phase": int(phase),
            "method": self.METHOD_NAME,
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
            "checks": checks,
            "enforced": enforce,
            "valid": bool(valid),
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "selection_uses_current_validation_only": True,
            "selection_uses_old_validation": False,
            "uses_old_real_training_samples": False,
            "uses_old_feature_replay": False,
            "uses_trainable_transport": False,
        }

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @staticmethod
    def _incremental_json_safe(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()
        if isinstance(value, Mapping):
            return {
                str(key): IncrementalPhaseTrainer._incremental_json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                IncrementalPhaseTrainer._incremental_json_safe(item)
                for item in value
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _save_incremental_reports(
        self,
        phase: int,
        history: Mapping[str, Any],
    ) -> Dict[str, str]:
        phase_dir = os.path.join(
            os.path.abspath(str(getattr(self, "save_dir", "./outputs"))),
            f"phase_{int(phase)}",
        )
        reports_dir = os.path.join(phase_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        history_path = os.path.join(reports_dir, "incremental_history.json")
        handoff_path = os.path.join(phase_dir, "incremental_handoff.pt")
        temporary = history_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(self._incremental_json_safe(history), stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, history_path)
        torch.save(
            self._incremental_json_safe(
                {
                    "phase": phase,
                    "status": history.get("status"),
                    "commit_report": history.get("commit_report"),
                    "admission": history.get("admission"),
                }
            ),
            handoff_path,
        )
        return {
            "history": history_path,
            "handoff": handoff_path,
        }

    # ------------------------------------------------------------------
    # Main phase
    # ------------------------------------------------------------------

    def train_incremental_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        phase = int(phase)
        epochs = int(epochs)
        if phase <= 0:
            raise ValueError("train_incremental_phase requires phase >= 1")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self._validate_incremental_configuration(epochs=epochs, lr=lr)

        old_classes, new_classes, seen_classes = self._resolve_phase_classes(phase)
        accepted_state = self._capture_accepted_state()
        self.dataset.start_phase(phase)

        controlled_plasticity = self._inc_bool(
            "incremental_controlled_plasticity", default=True
        )
        phase_context = self.model.begin_incremental_phase(
            phase=phase,
            old_class_ids=old_classes,
            new_class_ids=new_classes,
            controlled_plasticity=controlled_plasticity,
        )
        trainable_parameters = self.model.incremental_trainable_parameters()
        if not controlled_plasticity or not trainable_parameters:
            self._restore_accepted_state(accepted_state)
            raise RuntimeError(
                "this training entry requires controlled plasticity; run the "
                "frozen-backbone baseline through a separate zero-update evaluation"
            )

        learning_rate = self._inc_float("lr_inc", default=float(lr))
        optimizer = optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=self._inc_float(
                "incremental_weight_decay", default=1e-4
            ),
        )
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        if self._inc_bool("incremental_use_cosine_scheduler", default=True):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
                eta_min=learning_rate
                * self._inc_float("incremental_min_lr_ratio", default=0.10),
            )

        train_loader = self.dataset.get_phase_dataloader(
            phase,
            split="train",
            batch_size=int(batch_size),
            shuffle=False,
        )
        current_val_loader = self.dataset.get_phase_dataloader(
            phase,
            split="val",
            batch_size=int(batch_size),
            shuffle=False,
        )
        cumulative_val_loader = self.dataset.get_cumulative_dataloader(
            phase,
            split="val",
            batch_size=int(batch_size),
            shuffle=False,
        )
        train_data = self._collect_current_phase_tensors(
            train_loader,
            new_classes,
            context=f"phase_{phase}.current_train",
        )
        labels = train_data["labels"]
        assert torch.is_tensor(labels)
        folds = self._incremental_folds(labels, new_classes)
        fold_count = len(next(iter(folds.values())))

        history: Dict[str, Any] = {
            "phase": phase,
            "method": self.METHOD_NAME,
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
            "old_classes": list(old_classes),
            "new_classes": list(new_classes),
            "seen_classes": list(seen_classes),
            "controlled_plasticity": True,
            "trainable_parameter_names": self.model.incremental_trainable_parameter_names(),
            "uses_old_real_training_samples": False,
            "uses_old_feature_replay": False,
            "uses_trainable_transport": False,
            "epoch_metrics": [],
            "selection_protocol": (
                "current_validation_factor_geometry_then_q05_then_min_class"
            ),
        }

        best_backbone_state: Optional[Dict[str, Tensor]] = None
        best_score = (float("-inf"), float("-inf"), float("-inf"))
        best_epoch = -1
        best_candidate_report: Optional[Dict[str, Any]] = None
        best_current_validation: Optional[Dict[str, Any]] = None

        try:
            for epoch in range(epochs):
                step_metrics: List[Dict[str, Any]] = []
                steps = self._inc_int("incremental_steps_per_epoch", default=fold_count)
                for step in range(steps):
                    metric = self._training_step(
                        phase_context=phase_context,
                        train_data=train_data,
                        folds=folds,
                        fold_index=(epoch * steps + step) % fold_count,
                        optimizer=optimizer,
                        trainable_parameters=trainable_parameters,
                    )
                    step_metrics.append(metric)
                if scheduler is not None:
                    scheduler.step()

                transform, temporary_rows, _, candidate_report = self._fit_full_candidate(
                    phase_context, train_data, folds
                )
                del transform
                current_validation = self._evaluate_loader_with_rows(
                    current_val_loader,
                    rows=temporary_rows,
                    class_ids=seen_classes,
                    old_classes=old_classes,
                    new_classes=new_classes,
                    allowed_classes=new_classes,
                    context=f"phase_{phase}.epoch_{epoch + 1}.current_validation",
                )
                score = (
                    float(current_validation["accuracy"]),
                    float(current_validation["q05_gap"]),
                    float(current_validation["minimum_per_class_accuracy"]),
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_backbone_state = copy.deepcopy(
                        self.model.backbone.state_dict()
                    )
                    best_candidate_report = copy.deepcopy(candidate_report)
                    best_current_validation = copy.deepcopy(current_validation)

                average = {
                    key: sum(float(item[key]) for item in step_metrics) / len(step_metrics)
                    for key in step_metrics[0]
                }
                history["epoch_metrics"].append(
                    {
                        "epoch": epoch + 1,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        **average,
                        "current_validation": current_validation,
                    }
                )
                print(
                    f"[Incremental] phase={phase} epoch={epoch + 1}/{epochs} "
                    f"loss={average['total']:.4f} "
                    f"query={average['accuracy']:.2f}% "
                    f"val={current_validation['accuracy']:.2f}% "
                    f"valQ05={current_validation['q05_gap']:.3f} "
                    f"coordS={average['spectral_coordinate_rmse']:.4f} "
                    f"coordP={average['spatial_coordinate_rmse']:.4f}"
                )

            if best_backbone_state is None:
                raise RuntimeError("incremental training produced no selected checkpoint")
            self.model.backbone.load_state_dict(best_backbone_state, strict=True)
            history["best_epoch"] = best_epoch + 1
            history["best_current_validation"] = best_current_validation
            history["best_candidate_report"] = best_candidate_report

            crossfit = self._crossfit_certificate(
                phase_context,
                train_data,
                folds,
                seen_classes,
                old_classes,
                new_classes,
            )
            transform, temporary_rows, new_rows, candidate_report = self._fit_full_candidate(
                phase_context, train_data, folds
            )
            current_validation = self._evaluate_loader_with_rows(
                current_val_loader,
                rows=temporary_rows,
                class_ids=seen_classes,
                old_classes=old_classes,
                new_classes=new_classes,
                allowed_classes=new_classes,
                context=f"phase_{phase}.selected_current_validation",
            )
            cumulative_candidate = self._evaluate_loader_with_rows(
                cumulative_val_loader,
                rows=temporary_rows,
                class_ids=seen_classes,
                old_classes=old_classes,
                new_classes=new_classes,
                allowed_classes=seen_classes,
                context=f"phase_{phase}.candidate_cumulative_validation",
            )
            admission = self._incremental_admission(
                phase=phase,
                crossfit=crossfit,
                current_validation=current_validation,
                candidate_report=candidate_report,
            )
            history["crossfit_certificate"] = crossfit
            history["selected_current_validation"] = current_validation
            history["candidate_cumulative_validation"] = cumulative_candidate
            history["admission"] = admission

            if not admission["valid"]:
                self._restore_accepted_state(accepted_state)
                history.update(
                    {
                        "status": "REJECTED",
                        "committed": False,
                        "failed_checks": admission["failed_checks"],
                    }
                )
                history["artifact_paths"] = self._save_incremental_reports(
                    phase, history
                )
                return history

            commit_report = self.model.commit_incremental_phase(
                transform=transform,
                new_rows=new_rows,
                phase=phase,
            )
            full_index = torch.arange(labels.numel(), device=labels.device)
            final_train_output = self._current_output(
                train_data, full_index, require_grad=False
            )
            statistics = self.model.update_geometry_statistics(
                features=final_train_output["joint_feature"],
                labels=labels,
                class_ids=seen_classes,
            )
            geometry_admission = self.model.geometry_admission_report(
                seen_classes,
                maximum_reconstruction_error=self._inc_float(
                    "maximum_geometry_reconstruction_error", default=0.75
                ),
                minimum_effective_dimension=self._inc_float(
                    "minimum_geometry_effective_dimension", default=1.25
                ),
                require_statistics=True,
            )

            # Bank corruption and geometry-fit quality are different.
            # Structural failures always reject the phase.  Effective-dimension
            # and reconstruction thresholds are empirical quality gates and are
            # enforced only when incremental admission enforcement is enabled.
            structural_ok = bool(
                geometry_admission.get(
                    "structural_ok",
                    geometry_admission.get("ok", False),
                )
            )
            quality_ok = bool(
                geometry_admission.get(
                    "quality_ok",
                    geometry_admission.get("ok", False),
                )
            )
            quality_enforced = self._inc_bool(
                "incremental_geometry_quality_enforce",
                "incremental_admission_enforce",
                default=False,
            )
            geometry_admission["quality_enforced"] = bool(
                quality_enforced
            )
            geometry_admission["accepted_for_phase"] = bool(
                structural_ok and (quality_ok or not quality_enforced)
            )

            if not structural_ok:
                errors = geometry_admission.get(
                    "structural_errors",
                    geometry_admission.get("errors", []),
                )
                raise RuntimeError(
                    "post-commit GeometryBank structural validation failed: "
                    + "; ".join(str(value) for value in errors)
                )

            if quality_enforced and not quality_ok:
                errors = geometry_admission.get(
                    "quality_errors",
                    geometry_admission.get("errors", []),
                )
                raise RuntimeError(
                    "post-commit geometry quality admission failed: "
                    + "; ".join(str(value) for value in errors)
                )

            if not quality_ok:
                warning = (
                    "geometry quality warning retained because "
                    "incremental admission enforcement is disabled: "
                    + "; ".join(
                        str(value)
                        for value in geometry_admission.get(
                            "quality_errors",
                            geometry_admission.get("errors", []),
                        )
                    )
                )
                history.setdefault("warnings", []).append(warning)
                print(f"[Incremental warning] {warning}")

            committed_validation = self._evaluate_loader_with_rows(
                cumulative_val_loader,
                rows=None,
                class_ids=seen_classes,
                old_classes=old_classes,
                new_classes=new_classes,
                allowed_classes=seen_classes,
                context=f"phase_{phase}.committed_cumulative_validation",
            )
            if hasattr(self.dataset, "finalize_phase"):
                self.dataset.finalize_phase(phase)

            history.update(
                {
                    "status": "COMMITTED",
                    "committed": True,
                    "commit_report": commit_report,
                    "new_class_statistics": statistics,
                    "geometry_admission": geometry_admission,
                    "final_metrics": committed_validation,
                    "temporary_phase_context_deleted": True,
                }
            )
            hook = getattr(self, "pre_finalize_incremental_hook", None)
            if callable(hook):
                hook_result = hook(
                    phase=phase,
                    model=self.model,
                    dataset=self.dataset,
                    report=history,
                )
                if hook_result is not None:
                    history["phase_artifacts"] = hook_result
            history["artifact_paths"] = self._save_incremental_reports(
                phase, history
            )
            save_checkpoint = getattr(self, "save_checkpoint", None)
            if callable(save_checkpoint):
                history["checkpoint_path"] = save_checkpoint(phase, history)
            return history
        except Exception:
            self._restore_accepted_state(accepted_state)
            raise








# from __future__ import annotations

# import json
# import math
# import os
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from losses.loss import (
#     candidate_descriptor_trust_region_loss,
#     joint_energy_boundary_certificate_loss,
#     phase_consistent_conditional_joint_consolidation_loss,
#     two_sided_old_new_invasion_loss,
# )


# class IncrementalPhaseTrainer:
#     """Strict descriptor-only incremental admission for PC-STGB.

#     The encoder, projection, final base response prior, classifier, and every
#     committed class row remain immutable.  Current samples estimate provisional
#     schema-v5 rows.  Only bounded occupancy and conditional-tangent distribution
#     corrections are optimized.  Occupancy bases, tangent bases, and the
#     occupancy--tangent coupling are never optimized: they remain empirical
#     statistics of the current class support.

#     The optimization uses only the deployed conditional joint energy

#         E_c(z,g) = E_c^occ(z) + beta_T E_c^tan(g | z,c)

#     on three sources: current real tuples, coupled old replay, and held-out
#     current cross-fit queries.  Admission is atomic and digest protected.
#     """

#     METHOD_NAME = "PC-STGB"
#     BANK_SCHEMA_VERSION = 5
#     JOINT_FACTORIZATION = "p(z|c)prod_k p(g_k|z,c)"
#     STACK_BUILD_ID = "PC-STGB-INCREMENTAL-CONDITIONAL-ADMISSION-V5"

#     # ------------------------------------------------------------------
#     # Configuration
#     # ------------------------------------------------------------------
#     def _inc_value(self, name: str, default: Any = None) -> Any:
#         local = getattr(self, name, None)
#         if local is not None:
#             return local
#         args = getattr(self, "args", None)
#         if args is not None and hasattr(args, name):
#             value = getattr(args, name)
#             if value is not None:
#                 return value
#         if default is not None:
#             return default
#         raise RuntimeError(f"Missing incremental option {name!r}")

#     def _inc_float(self, name: str, default: float) -> float:
#         value = float(self._inc_value(name, default))
#         if not math.isfinite(value):
#             raise RuntimeError(f"{name} must be finite")
#         return value

#     def _inc_int(self, name: str, default: int) -> int:
#         value = self._inc_value(name, default)
#         if isinstance(value, bool):
#             raise RuntimeError(f"{name} must be an integer, not bool")
#         integer = int(value)
#         if float(value) != float(integer):
#             raise RuntimeError(f"{name} must be an integer")
#         return integer

#     def _inc_bool(self, name: str, default: bool) -> bool:
#         value = self._inc_value(name, default)
#         if isinstance(value, bool):
#             return value
#         if isinstance(value, int) and value in {0, 1}:
#             return bool(value)
#         if isinstance(value, str):
#             token = value.strip().lower()
#             if token in {"1", "true", "yes", "y", "on"}:
#                 return True
#             if token in {"0", "false", "no", "n", "off", "none", "null", ""}:
#                 return False
#         raise RuntimeError(f"{name} must be an explicit boolean")

#     def _validate_incremental_configuration(self, epochs: int) -> None:
#         positive = {
#             "descriptor_lr": self._inc_float("descriptor_lr", 1e-3),
#             "incremental_temperature": self._inc_float(
#                 "incremental_temperature", 0.20
#             ),
#             "incremental_certificate_temperature": self._inc_float(
#                 "incremental_certificate_temperature", 0.20
#             ),
#             "incremental_invasion_temperature": self._inc_float(
#                 "incremental_invasion_temperature", 0.20
#             ),
#             "incremental_grad_clip": self._inc_float(
#                 "incremental_grad_clip", 5.0
#             ),
#             "incremental_max_mean_shift": self._inc_float(
#                 "incremental_max_mean_shift", 0.25
#             ),
#             "incremental_max_log_eigval_shift": self._inc_float(
#                 "incremental_max_log_eigval_shift", 0.35
#             ),
#             "incremental_max_log_residual_shift": self._inc_float(
#                 "incremental_max_log_residual_shift", 0.35
#             ),
#             "incremental_max_response_mean_shift": self._inc_float(
#                 "incremental_max_response_mean_shift", 0.25
#             ),
#             "incremental_max_response_log_eigval_shift": self._inc_float(
#                 "incremental_max_response_log_eigval_shift", 0.35
#             ),
#             "incremental_max_response_log_residual_shift": self._inc_float(
#                 "incremental_max_response_log_residual_shift", 0.35
#             ),
#         }
#         failures = [name for name, value in positive.items() if value <= 0.0]
#         if failures:
#             raise RuntimeError(
#                 f"Incremental positive options are invalid: {failures}"
#             )

#         nonnegative_names = (
#             "incremental_margin",
#             "incremental_certificate_margin",
#             "incremental_certificate_kappa",
#             "incremental_invasion_margin",
#             "incremental_new_weight",
#             "incremental_old_replay_weight",
#             "incremental_crossfit_weight",
#             "incremental_certificate_weight",
#             "incremental_invasion_weight",
#             "incremental_trust_weight",
#         )
#         for name in nonnegative_names:
#             defaults = {
#                 "incremental_margin": 0.30,
#                 "incremental_certificate_margin": 0.0,
#                 "incremental_certificate_kappa": 1.0,
#                 "incremental_invasion_margin": 0.30,
#                 "incremental_new_weight": 1.0,
#                 "incremental_old_replay_weight": 1.0,
#                 "incremental_crossfit_weight": 1.0,
#                 "incremental_certificate_weight": 0.10,
#                 "incremental_invasion_weight": 1.0,
#                 "incremental_trust_weight": 0.05,
#             }
#             if self._inc_float(name, defaults[name]) < 0.0:
#                 raise RuntimeError(f"{name} must be non-negative")

#         if self._inc_int("incremental_crossfit_folds", 3) < 2:
#             raise RuntimeError("incremental_crossfit_folds must be at least two")
#         if epochs > 0 and self._inc_int("incremental_steps_per_epoch", 10) <= 0:
#             raise RuntimeError("incremental_steps_per_epoch must be positive")
#         if self._inc_int("incremental_old_replay_samples_per_class", 32) <= 0:
#             raise RuntimeError(
#                 "incremental_old_replay_samples_per_class must be positive"
#             )

#         forbidden = (
#             "use_geometry_transport",
#             "use_sglat_transport",
#             "allow_old_model_transport",
#             "use_energy_calibrator",
#             "use_adaptive_boundary",
#             "use_incremental_adapter",
#             "use_geometry_gated_adapter",
#             "allow_incremental_projection_training",
#             "use_raw_spectral_gaussian",
#             "use_independent_response_replay",
#         )
#         active = [name for name in forbidden if self._inc_bool(name, False)]
#         if active:
#             raise RuntimeError(
#                 "Retired incremental mechanisms are enabled: " + ", ".join(active)
#             )

#     # ------------------------------------------------------------------
#     # Frozen current-phase tuple collection
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _collect_incremental_payload(
#         self,
#         loader: Any,
#         allowed_classes: Sequence[int],
#         *,
#         context: str,
#     ) -> Dict[str, torch.Tensor]:
#         allowed = set(int(value) for value in allowed_classes)
#         feature_parts: List[torch.Tensor] = []
#         response_parts: List[torch.Tensor] = []
#         label_parts: List[torch.Tensor] = []
#         previous = bool(self.model.training)
#         self.model.eval()
#         try:
#             for batch in loader:
#                 payload = self._extract_model_geometry_tuple(
#                     batch,
#                     require_grad=False,
#                     require_response_views=True,
#                     context=context,
#                 )
#                 labels = payload["labels"].long().flatten()
#                 observed = set(
#                     int(value)
#                     for value in torch.unique(labels).detach().cpu().tolist()
#                 )
#                 leaked = sorted(observed - allowed)
#                 if leaked:
#                     raise RuntimeError(
#                         f"{context}: loader exposed forbidden classes {leaked}"
#                     )
#                 feature_parts.append(payload["features"].detach())
#                 response_parts.append(payload["spectral_responses"].detach())
#                 label_parts.append(labels.detach())
#         finally:
#             self.model.train(previous)

#         if not feature_parts:
#             raise RuntimeError(f"{context}: loader is empty")
#         result = {
#             "features": torch.cat(feature_parts, dim=0),
#             "spectral_responses": torch.cat(response_parts, dim=0),
#             "responses": torch.cat(response_parts, dim=0),
#             "labels": torch.cat(label_parts, dim=0),
#             "joint_factorization": self.JOINT_FACTORIZATION,
#         }
#         observed = set(
#             int(value)
#             for value in torch.unique(result["labels"]).detach().cpu().tolist()
#         )
#         if observed != allowed:
#             raise RuntimeError(
#                 f"{context}: observed classes {sorted(observed)} do not match "
#                 f"expected classes {sorted(allowed)}"
#             )
#         return result

#     def _enter_incremental_mode(
#         self,
#         phase: int,
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         seen_classes: Sequence[int],
#     ) -> None:
#         self.model.set_incremental_mode(
#             phase=int(phase),
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=len(old_classes),
#         )
#         self.assert_clean_incremental_contract(
#             phase,
#             old_classes,
#             new_classes,
#             seen_classes,
#             context=f"phase_{phase}.incremental_contract",
#         )
#         self.model.assert_frozen_modules()
#         for parameter in self.model.parameters():
#             if parameter.requires_grad or parameter.grad is not None:
#                 raise RuntimeError(
#                     "Model parameters must be frozen and gradient-free in "
#                     "incremental phases"
#                 )

#     # ------------------------------------------------------------------
#     # Candidate construction and bounded descriptor corrections
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _build_incremental_candidate_rows(
#         self,
#         payload: Mapping[str, torch.Tensor],
#         new_classes: Sequence[int],
#     ) -> Dict[int, Dict[str, torch.Tensor]]:
#         rows = self.model.build_candidate_geometry_rows(
#             new_classes,
#             payload["features"],
#             payload["labels"],
#             spectral_responses=payload["spectral_responses"],
#         )
#         self._assert_incremental_candidate_rows(rows, new_classes)
#         return rows

#     @staticmethod
#     def _parameter_mapping(
#         container: nn.ParameterDict,
#         class_ids: Sequence[int],
#     ) -> Dict[int, torch.Tensor]:
#         return {int(class_id): container[str(int(class_id))] for class_id in class_ids}

#     def _make_incremental_descriptor_parameters(
#         self,
#         initial_rows: Mapping[int, Mapping[str, Any]],
#         new_classes: Sequence[int],
#     ) -> Dict[str, nn.ParameterDict]:
#         groups = {
#             "mean": nn.ParameterDict(),
#             "log_eig": nn.ParameterDict(),
#             "log_res": nn.ParameterDict(),
#             "response_mean": nn.ParameterDict(),
#             "response_log_eig": nn.ParameterDict(),
#             "response_log_res": nn.ParameterDict(),
#         }
#         for class_id in new_classes:
#             row = initial_rows[int(class_id)]
#             key = str(int(class_id))
#             groups["mean"][key] = nn.Parameter(torch.zeros_like(row["mean"]))
#             groups["log_eig"][key] = nn.Parameter(
#                 torch.zeros_like(row["eigvals"])
#             )
#             groups["log_res"][key] = nn.Parameter(
#                 torch.zeros_like(torch.as_tensor(row["res_var"]).reshape(()))
#             )
#             groups["response_mean"][key] = nn.Parameter(
#                 torch.zeros_like(row["response_means"])
#             )
#             groups["response_log_eig"][key] = nn.Parameter(
#                 torch.zeros_like(row["response_eigvals"])
#             )
#             groups["response_log_res"][key] = nn.Parameter(
#                 torch.zeros_like(row["response_res_vars"])
#             )
#         return groups

#     def _refined_incremental_rows(
#         self,
#         initial_rows: Mapping[int, Mapping[str, Any]],
#         parameters: Mapping[str, nn.ParameterDict],
#         new_classes: Sequence[int],
#     ) -> Dict[int, Dict[str, Any]]:
#         bank = self.model.geometry_bank
#         rows = bank.refine_candidate_joint_rows(
#             initial_rows,
#             mean_deltas=self._parameter_mapping(parameters["mean"], new_classes),
#             log_eigval_deltas=self._parameter_mapping(
#                 parameters["log_eig"], new_classes
#             ),
#             log_residual_deltas=self._parameter_mapping(
#                 parameters["log_res"], new_classes
#             ),
#             response_mean_deltas=self._parameter_mapping(
#                 parameters["response_mean"], new_classes
#             ),
#             response_log_eigval_deltas=self._parameter_mapping(
#                 parameters["response_log_eig"], new_classes
#             ),
#             response_log_residual_deltas=self._parameter_mapping(
#                 parameters["response_log_res"], new_classes
#             ),
#             maximum_mean_shift=self._inc_float(
#                 "incremental_max_mean_shift", 0.25
#             ),
#             maximum_log_eigval_shift=self._inc_float(
#                 "incremental_max_log_eigval_shift", 0.35
#             ),
#             maximum_log_residual_shift=self._inc_float(
#                 "incremental_max_log_residual_shift", 0.35
#             ),
#             maximum_response_mean_shift=self._inc_float(
#                 "incremental_max_response_mean_shift", 0.25
#             ),
#             maximum_response_log_eigval_shift=self._inc_float(
#                 "incremental_max_response_log_eigval_shift", 0.35
#             ),
#             maximum_response_log_residual_shift=self._inc_float(
#                 "incremental_max_response_log_residual_shift", 0.35
#             ),
#         )
#         self._assert_incremental_candidate_rows(rows, new_classes)
#         return rows

#     @torch.no_grad()
#     def _project_incremental_parameters(
#         self,
#         parameters: Mapping[str, nn.ParameterDict],
#         new_classes: Sequence[int],
#     ) -> None:
#         limits = {
#             "mean": self._inc_float("incremental_max_mean_shift", 0.25),
#             "log_eig": self._inc_float(
#                 "incremental_max_log_eigval_shift", 0.35
#             ),
#             "log_res": self._inc_float(
#                 "incremental_max_log_residual_shift", 0.35
#             ),
#             "response_mean": self._inc_float(
#                 "incremental_max_response_mean_shift", 0.25
#             ),
#             "response_log_eig": self._inc_float(
#                 "incremental_max_response_log_eigval_shift", 0.35
#             ),
#             "response_log_res": self._inc_float(
#                 "incremental_max_response_log_residual_shift", 0.35
#             ),
#         }
#         for class_id in new_classes:
#             key = str(int(class_id))
#             for group in ("mean", "response_mean"):
#                 tensor = parameters[group][key]
#                 norm = tensor.norm()
#                 limit = limits[group]
#                 if float(norm.item()) > limit:
#                     tensor.mul_(limit / norm.clamp_min(torch.finfo(tensor.dtype).eps))
#             parameters["log_eig"][key].clamp_(
#                 -limits["log_eig"], limits["log_eig"]
#             )
#             parameters["log_res"][key].clamp_(
#                 -limits["log_res"], limits["log_res"]
#             )
#             parameters["response_log_eig"][key].clamp_(
#                 -limits["response_log_eig"], limits["response_log_eig"]
#             )
#             parameters["response_log_res"][key].clamp_(
#                 -limits["response_log_res"], limits["response_log_res"]
#             )

#     def _assert_incremental_candidate_rows(
#         self,
#         rows: Mapping[int, Mapping[str, Any]],
#         new_classes: Sequence[int],
#     ) -> None:
#         expected = set(int(value) for value in new_classes)
#         if set(int(value) for value in rows) != expected:
#             raise RuntimeError("Candidate row IDs do not match new classes")
#         required = (
#             "mean",
#             "basis",
#             "eigvals",
#             "res_var",
#             "active_rank",
#             "sample_count",
#             "response_basis",
#             "response_means",
#             "response_eigvals",
#             "response_res_vars",
#             "response_active_rank",
#             "response_stats_ready",
#             "response_coupling",
#             "response_coupling_reliability",
#             "response_coupling_explained_variance",
#             "response_coupling_ready",
#         )
#         for class_id in new_classes:
#             row = rows[int(class_id)]
#             missing = [name for name in required if row.get(name) is None]
#             if missing:
#                 raise RuntimeError(
#                     f"Candidate class {class_id} lacks fields {missing}"
#                 )
#             if not bool(torch.as_tensor(row["response_stats_ready"]).item()):
#                 raise RuntimeError(
#                     f"Candidate class {class_id} lacks tangent residual geometry"
#                 )
#             if not bool(torch.as_tensor(row["response_coupling_ready"]).item()):
#                 raise RuntimeError(
#                     f"Candidate class {class_id} lacks occupancy--tangent coupling"
#                 )
#             for name in required:
#                 value = row.get(name)
#                 if torch.is_tensor(value) and value.dtype != torch.bool:
#                     if not torch.isfinite(value.float()).all():
#                         raise RuntimeError(
#                             f"Candidate class {class_id} has invalid {name}"
#                         )

#     # ------------------------------------------------------------------
#     # Cross-fitted contexts
#     # ------------------------------------------------------------------
#     def _incremental_folds(
#         self,
#         labels: torch.Tensor,
#         new_classes: Sequence[int],
#     ) -> Dict[int, List[torch.Tensor]]:
#         fold_count = self._inc_int("incremental_crossfit_folds", 3)
#         seed = self._inc_int("seed", 0)
#         folds: Dict[int, List[torch.Tensor]] = {}
#         for class_id in new_classes:
#             index = torch.nonzero(
#                 labels.eq(int(class_id)), as_tuple=False
#             ).flatten()
#             if index.numel() < 2 * fold_count:
#                 raise RuntimeError(
#                     f"Class {class_id} has {index.numel()} samples; "
#                     f"{fold_count}-fold cross-fitting needs at least "
#                     f"{2 * fold_count}"
#                 )
#             generator = torch.Generator().manual_seed(
#                 seed + 104729 * int(class_id)
#             )
#             order = torch.randperm(index.numel(), generator=generator).to(
#                 index.device
#             )
#             split = list(torch.tensor_split(index.index_select(0, order), fold_count))
#             if any(part.numel() == 0 for part in split):
#                 raise RuntimeError(f"Class {class_id} produced an empty fold")
#             folds[int(class_id)] = split
#         return folds

#     @torch.no_grad()
#     def _build_incremental_crossfit_contexts(
#         self,
#         payload: Mapping[str, torch.Tensor],
#         new_classes: Sequence[int],
#     ) -> List[Dict[str, Any]]:
#         folds = self._incremental_folds(payload["labels"], new_classes)
#         fold_count = self._inc_int("incremental_crossfit_folds", 3)
#         contexts: List[Dict[str, Any]] = []
#         for fold_index in range(fold_count):
#             support_parts: List[torch.Tensor] = []
#             query_parts: List[torch.Tensor] = []
#             for class_id in new_classes:
#                 parts = folds[int(class_id)]
#                 query_parts.append(parts[fold_index])
#                 support_parts.append(
#                     torch.cat(
#                         [part for index, part in enumerate(parts) if index != fold_index]
#                     )
#                 )
#             support_index = torch.cat(support_parts)
#             query_index = torch.cat(query_parts)
#             support = {
#                 "features": payload["features"].index_select(0, support_index),
#                 "spectral_responses": payload["spectral_responses"].index_select(
#                     0, support_index
#                 ),
#                 "labels": payload["labels"].index_select(0, support_index),
#             }
#             contexts.append(
#                 {
#                     "initial_rows": self._build_incremental_candidate_rows(
#                         support, new_classes
#                     ),
#                     "query_features": payload["features"].index_select(
#                         0, query_index
#                     ),
#                     "query_responses": payload["spectral_responses"].index_select(
#                         0, query_index
#                     ),
#                     "query_labels": payload["labels"].index_select(0, query_index),
#                 }
#             )
#         return contexts

#     # ------------------------------------------------------------------
#     # Exact conditional joint objective
#     # ------------------------------------------------------------------
#     def _score_rows(
#         self,
#         features: torch.Tensor,
#         responses: torch.Tensor,
#         rows: Mapping[int, Mapping[str, Any]],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#     ) -> Dict[str, Any]:
#         scored = self.score_candidate_rows(
#             features,
#             responses,
#             old_class_ids=old_classes,
#             candidate_rows=rows,
#             candidate_class_ids=new_classes,
#             return_parts=True,
#         )
#         if scored.get("joint_factorization") != self.JOINT_FACTORIZATION:
#             raise RuntimeError("Candidate scoring returned the wrong factorization")
#         if not bool(scored.get("uses_coupling_inference_score", False)):
#             raise RuntimeError("Candidate scoring did not use occupancy--tangent coupling")
#         if bool(scored.get("uses_independent_response_factorization", True)):
#             raise RuntimeError("Candidate scoring used an independent response factorization")
#         return scored

#     def _trust_loss(
#         self,
#         parameters: Mapping[str, nn.ParameterDict],
#         new_classes: Sequence[int],
#     ) -> Dict[str, torch.Tensor]:
#         return candidate_descriptor_trust_region_loss(
#             mean_deltas=self._parameter_mapping(parameters["mean"], new_classes),
#             log_eigval_deltas=self._parameter_mapping(
#                 parameters["log_eig"], new_classes
#             ),
#             log_residual_deltas=self._parameter_mapping(
#                 parameters["log_res"], new_classes
#             ),
#             response_mean_deltas=self._parameter_mapping(
#                 parameters["response_mean"], new_classes
#             ),
#             response_log_eigval_deltas=self._parameter_mapping(
#                 parameters["response_log_eig"], new_classes
#             ),
#             response_log_residual_deltas=self._parameter_mapping(
#                 parameters["response_log_res"], new_classes
#             ),
#             mean_scales=self._inc_float("incremental_max_mean_shift", 0.25),
#             log_eigval_scales=self._inc_float(
#                 "incremental_max_log_eigval_shift", 0.35
#             ),
#             log_residual_scales=self._inc_float(
#                 "incremental_max_log_residual_shift", 0.35
#             ),
#             response_mean_scales=self._inc_float(
#                 "incremental_max_response_mean_shift", 0.25
#             ),
#             response_log_eigval_scales=self._inc_float(
#                 "incremental_max_response_log_eigval_shift", 0.35
#             ),
#             response_log_residual_scales=self._inc_float(
#                 "incremental_max_response_log_residual_shift", 0.35
#             ),
#             return_parts=True,
#         )

#     def _incremental_objective(
#         self,
#         payload: Mapping[str, torch.Tensor],
#         old_replay: Mapping[str, torch.Tensor],
#         initial_rows: Mapping[int, Mapping[str, Any]],
#         parameters: Mapping[str, nn.ParameterDict],
#         crossfit_contexts: Sequence[Mapping[str, Any]],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         seen_classes: Sequence[int],
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[int, Dict[str, Any]]]:
#         rows = self._refined_incremental_rows(
#             initial_rows, parameters, new_classes
#         )
#         new_scored = self._score_rows(
#             payload["features"],
#             payload["spectral_responses"],
#             rows,
#             old_classes,
#             new_classes,
#         )
#         old_scored = self._score_rows(
#             old_replay["features"],
#             old_replay["spectral_responses"],
#             rows,
#             old_classes,
#             new_classes,
#         )
#         new_targets = self.global_to_seen_local(
#             payload["labels"], seen_classes, context="incremental.new_targets"
#         )
#         old_targets = self.global_to_seen_local(
#             old_replay["global_labels"],
#             seen_classes,
#             context="incremental.old_replay_targets",
#         )
#         margin = self._inc_float("incremental_margin", 0.30)
#         temperature = self._inc_float("incremental_temperature", 0.20)
#         new_pc = phase_consistent_conditional_joint_consolidation_loss(
#             new_scored["joint_energy"],
#             new_targets,
#             margin=margin,
#             temperature=temperature,
#             class_balanced=True,
#             joint_factorization=self.JOINT_FACTORIZATION,
#             return_parts=True,
#         )
#         old_pc = phase_consistent_conditional_joint_consolidation_loss(
#             old_scored["joint_energy"],
#             old_targets,
#             margin=margin,
#             temperature=temperature,
#             class_balanced=True,
#             joint_factorization=self.JOINT_FACTORIZATION,
#             return_parts=True,
#         )

#         crossfit_terms: List[torch.Tensor] = []
#         crossfit_violations: List[torch.Tensor] = []
#         for context in crossfit_contexts:
#             fold_rows = self._refined_incremental_rows(
#                 context["initial_rows"], parameters, new_classes
#             )
#             fold_scored = self._score_rows(
#                 context["query_features"],
#                 context["query_responses"],
#                 fold_rows,
#                 old_classes,
#                 new_classes,
#             )
#             fold_targets = self.global_to_seen_local(
#                 context["query_labels"],
#                 seen_classes,
#                 context="incremental.crossfit_targets",
#             )
#             fold_loss = phase_consistent_conditional_joint_consolidation_loss(
#                 fold_scored["joint_energy"],
#                 fold_targets,
#                 margin=margin,
#                 temperature=temperature,
#                 class_balanced=True,
#                 joint_factorization=self.JOINT_FACTORIZATION,
#                 return_parts=True,
#             )
#             crossfit_terms.append(fold_loss["total"])
#             crossfit_violations.append(
#                 fold_loss["classification_violation_rate"]
#             )
#         crossfit_total = torch.stack(crossfit_terms).mean()

#         combined_energy = torch.cat(
#             [old_scored["joint_energy"], new_scored["joint_energy"]], dim=0
#         )
#         combined_targets = torch.cat([old_targets, new_targets], dim=0)
#         certificate = joint_energy_boundary_certificate_loss(
#             combined_energy,
#             combined_targets,
#             margin=self._inc_float("incremental_certificate_margin", 0.0),
#             temperature=self._inc_float(
#                 "incremental_certificate_temperature", 0.20
#             ),
#             confidence_multiplier=self._inc_float(
#                 "incremental_certificate_kappa", 1.0
#             ),
#             minimum_samples_per_source=2,
#             joint_factorization=self.JOINT_FACTORIZATION,
#             return_parts=True,
#         )

#         old_mask = torch.tensor(
#             [class_id in set(old_classes) for class_id in seen_classes],
#             device=self.device,
#             dtype=torch.bool,
#         )
#         new_mask = ~old_mask
#         invasion = two_sided_old_new_invasion_loss(
#             old_scored["joint_energy"],
#             old_targets,
#             new_scored["joint_energy"],
#             new_targets,
#             old_class_mask=old_mask,
#             new_class_mask=new_mask,
#             margin=self._inc_float("incremental_invasion_margin", 0.30),
#             temperature=self._inc_float(
#                 "incremental_invasion_temperature", 0.20
#             ),
#             class_balanced=True,
#             joint_factorization=self.JOINT_FACTORIZATION,
#             return_parts=True,
#         )
#         trust = self._trust_loss(parameters, new_classes)

#         total = (
#             self._inc_float("incremental_new_weight", 1.0) * new_pc["total"]
#             + self._inc_float("incremental_old_replay_weight", 1.0)
#             * old_pc["total"]
#             + self._inc_float("incremental_crossfit_weight", 1.0)
#             * crossfit_total
#             + self._inc_float("incremental_certificate_weight", 0.10)
#             * certificate["total"]
#             + self._inc_float("incremental_invasion_weight", 1.0)
#             * invasion["total"]
#             + self._inc_float("incremental_trust_weight", 0.05)
#             * trust["total"]
#         )
#         if total.dim() != 0 or not torch.isfinite(total):
#             raise RuntimeError("Incremental PC-STGB objective is not finite")
#         parts = {
#             "total": total,
#             "new_joint": new_pc["total"],
#             "old_replay_joint": old_pc["total"],
#             "crossfit_joint": crossfit_total,
#             "boundary_certificate": certificate["total"],
#             "two_sided_invasion": invasion["total"],
#             "trust": trust["total"],
#             "new_classification_violation": new_pc[
#                 "classification_violation_rate"
#             ],
#             "old_replay_classification_violation": old_pc[
#                 "classification_violation_rate"
#             ],
#             "crossfit_classification_violation": torch.stack(
#                 crossfit_violations
#             ).mean(),
#             "certificate_violation_rate": certificate[
#                 "certificate_violation_rate"
#             ],
#             "minimum_certificate_lcb": certificate["minimum_lcb"],
#             "old_to_new_invasion_rate": invasion["old_to_new_invasion_rate"],
#             "new_to_old_invasion_rate": invasion["new_to_old_invasion_rate"],
#             "old_to_new_q05_gap": invasion["old_to_new_q05_gap"],
#             "new_to_old_q05_gap": invasion["new_to_old_q05_gap"],
#             "trust_maximum_class_penalty": trust[
#                 "maximum_class_penalty"
#             ],
#         }
#         return total, parts, rows

#     # ------------------------------------------------------------------
#     # Evaluation and statistics
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _energy_metrics(
#         energy: torch.Tensor,
#         targets: torch.Tensor,
#     ) -> Dict[str, Any]:
#         true = energy.gather(1, targets[:, None]).squeeze(1)
#         rivals = energy.clone()
#         rivals.scatter_(1, targets[:, None], float("inf"))
#         gap = rivals.min(dim=1).values - true
#         prediction = energy.argmin(dim=1)
#         return {
#             "accuracy": 100.0
#             * float(prediction.eq(targets).float().mean().item()),
#             "classification_violation_rate": float(
#                 gap.le(0.0).float().mean().item()
#             ),
#             "mean_gap": float(gap.mean().item()),
#             "q05_gap": float(torch.quantile(gap, 0.05).item()),
#             "minimum_gap": float(gap.min().item()),
#             "gap": gap,
#             "prediction": prediction,
#             "true_energy": true,
#         }

#     @torch.no_grad()
#     def _evaluate_incremental_candidate_loader(
#         self,
#         loader: Any,
#         rows: Mapping[int, Mapping[str, Any]],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         seen_classes: Sequence[int],
#         *,
#         allowed_classes: Sequence[int],
#         context: str,
#     ) -> Dict[str, Any]:
#         allowed = set(int(value) for value in allowed_classes)
#         energies: List[torch.Tensor] = []
#         labels: List[torch.Tensor] = []
#         previous = bool(self.model.training)
#         self.model.eval()
#         try:
#             for batch in loader:
#                 payload = self._extract_model_geometry_tuple(
#                     batch,
#                     require_grad=False,
#                     require_response_views=True,
#                     context=context,
#                 )
#                 observed = set(
#                     int(value)
#                     for value in torch.unique(payload["labels"]).detach().cpu().tolist()
#                 )
#                 leaked = sorted(observed - allowed)
#                 if leaked:
#                     raise RuntimeError(
#                         f"{context}: loader exposed forbidden classes {leaked}"
#                     )
#                 scored = self._score_rows(
#                     payload["features"],
#                     payload["spectral_responses"],
#                     rows,
#                     old_classes,
#                     new_classes,
#                 )
#                 energies.append(scored["joint_energy"].detach())
#                 labels.append(payload["labels"].detach())
#         finally:
#             self.model.train(previous)
#         if not energies:
#             raise RuntimeError(f"{context}: loader is empty")
#         energy = torch.cat(energies, dim=0)
#         labels_global = torch.cat(labels, dim=0)
#         targets = self.global_to_seen_local(
#             labels_global, seen_classes, context=f"{context}.targets"
#         )
#         metrics = self._energy_metrics(energy, targets)
#         prediction_global = self.seen_local_to_global(
#             metrics["prediction"], seen_classes, context=f"{context}.predictions"
#         )
#         per_class: Dict[int, float] = {}
#         for class_id in allowed_classes:
#             mask = labels_global.eq(int(class_id))
#             if bool(mask.any().item()):
#                 per_class[int(class_id)] = 100.0 * float(
#                     prediction_global[mask].eq(labels_global[mask]).float().mean().item()
#                 )
#         old_mask = torch.zeros_like(labels_global, dtype=torch.bool)
#         new_mask = torch.zeros_like(labels_global, dtype=torch.bool)
#         for class_id in old_classes:
#             old_mask |= labels_global.eq(int(class_id))
#         for class_id in new_classes:
#             new_mask |= labels_global.eq(int(class_id))

#         def accuracy(mask: torch.Tensor) -> float:
#             if not bool(mask.any().item()):
#                 return 0.0
#             return 100.0 * float(
#                 prediction_global[mask].eq(labels_global[mask]).float().mean().item()
#             )

#         old_accuracy = accuracy(old_mask)
#         new_accuracy = accuracy(new_mask)
#         harmonic = (
#             0.0
#             if old_accuracy + new_accuracy <= 0.0
#             else 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)
#         )
#         old_set = set(int(value) for value in old_classes)
#         new_set = set(int(value) for value in new_classes)
#         old_to_new = (
#             sum(
#                 int(int(value) in new_set)
#                 for value in prediction_global[old_mask].detach().cpu().tolist()
#             )
#             / max(int(old_mask.sum().item()), 1)
#         )
#         new_to_old = (
#             sum(
#                 int(int(value) in old_set)
#                 for value in prediction_global[new_mask].detach().cpu().tolist()
#             )
#             / max(int(new_mask.sum().item()), 1)
#         )
#         return {
#             "accuracy": metrics["accuracy"],
#             "old_accuracy": old_accuracy,
#             "new_accuracy": new_accuracy,
#             "old_new_harmonic_mean": harmonic,
#             "minimum_per_class_accuracy": min(per_class.values()) if per_class else 0.0,
#             "per_class_accuracy": per_class,
#             "classification_violation_rate": metrics[
#                 "classification_violation_rate"
#             ],
#             "mean_gap": metrics["mean_gap"],
#             "q05_gap": metrics["q05_gap"],
#             "minimum_gap": metrics["minimum_gap"],
#             "old_to_new_invasion_rate": float(old_to_new),
#             "new_to_old_invasion_rate": float(new_to_old),
#             "sample_count": int(labels_global.numel()),
#         }

#     @torch.no_grad()
#     def _incremental_crossfit_certificate(
#         self,
#         contexts: Sequence[Mapping[str, Any]],
#         parameters: Mapping[str, nn.ParameterDict],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         seen_classes: Sequence[int],
#     ) -> Dict[str, Any]:
#         energy_parts: List[torch.Tensor] = []
#         target_parts: List[torch.Tensor] = []
#         global_label_parts: List[torch.Tensor] = []
#         for context in contexts:
#             rows = self._refined_incremental_rows(
#                 context["initial_rows"], parameters, new_classes
#             )
#             scored = self._score_rows(
#                 context["query_features"],
#                 context["query_responses"],
#                 rows,
#                 old_classes,
#                 new_classes,
#             )
#             targets = self.global_to_seen_local(
#                 context["query_labels"],
#                 seen_classes,
#                 context="incremental.crossfit_certificate",
#             )
#             energy_parts.append(scored["joint_energy"].detach())
#             target_parts.append(targets.detach())
#             global_label_parts.append(context["query_labels"].detach())
#         energy = torch.cat(energy_parts, dim=0)
#         targets = torch.cat(target_parts, dim=0)
#         global_labels = torch.cat(global_label_parts, dim=0)
#         metrics = self._energy_metrics(energy, targets)
#         prediction_global = self.seen_local_to_global(
#             metrics["prediction"], seen_classes
#         )
#         per_class = {}
#         for class_id in new_classes:
#             mask = global_labels.eq(int(class_id))
#             per_class[int(class_id)] = 100.0 * float(
#                 prediction_global[mask].eq(global_labels[mask]).float().mean().item()
#             )
#         minimum_class = min(
#             new_classes, key=lambda class_id: (per_class[int(class_id)], int(class_id))
#         )
#         certificate = joint_energy_boundary_certificate_loss(
#             energy,
#             targets,
#             margin=self._inc_float("incremental_certificate_margin", 0.0),
#             temperature=self._inc_float(
#                 "incremental_certificate_temperature", 0.20
#             ),
#             confidence_multiplier=self._inc_float(
#                 "incremental_certificate_kappa", 1.0
#             ),
#             minimum_samples_per_source=2,
#             joint_factorization=self.JOINT_FACTORIZATION,
#             return_parts=True,
#         )
#         return {
#             "protocol": "stratified_new_class_crossfit_conditional_joint_energy",
#             "folds": len(contexts),
#             "accuracy": metrics["accuracy"],
#             "minimum_per_class_accuracy": per_class[int(minimum_class)],
#             "minimum_accuracy_class_id": int(minimum_class),
#             "per_class_accuracy": per_class,
#             "classification_violation_rate": metrics[
#                 "classification_violation_rate"
#             ],
#             "mean_gap": metrics["mean_gap"],
#             "q05_gap": metrics["q05_gap"],
#             "minimum_gap": metrics["minimum_gap"],
#             "certificate_violation_rate": float(
#                 certificate["certificate_violation_rate"].item()
#             ),
#             "minimum_certificate_lcb": float(
#                 certificate["minimum_lcb"].item()
#             ),
#             "query_count": int(targets.numel()),
#         }

#     @torch.no_grad()
#     def _attach_incremental_statistics(
#         self,
#         rows: Mapping[int, Mapping[str, Any]],
#         payload: Mapping[str, torch.Tensor],
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         seen_classes: Sequence[int],
#     ) -> Dict[int, Dict[str, Any]]:
#         scored = self._score_rows(
#             payload["features"],
#             payload["spectral_responses"],
#             rows,
#             old_classes,
#             new_classes,
#         )
#         energy = scored["joint_energy"]
#         targets = self.global_to_seen_local(
#             payload["labels"], seen_classes, context="incremental.statistics"
#         )
#         output: Dict[int, Dict[str, Any]] = {}
#         energy_levels = torch.tensor(
#             [0.50, 0.75, 0.90, 0.95], device=energy.device, dtype=energy.dtype
#         )
#         margin_levels = torch.tensor(
#             [0.05, 0.10], device=energy.device, dtype=energy.dtype
#         )
#         position = {int(class_id): index for index, class_id in enumerate(seen_classes)}
#         for class_id in new_classes:
#             mask = payload["labels"].eq(int(class_id))
#             local = position[int(class_id)]
#             class_energy = energy[mask]
#             true = class_energy[:, local]
#             rivals = class_energy.clone()
#             rivals[:, local] = float("inf")
#             margin = rivals.min(dim=1).values - true
#             row = dict(rows[int(class_id)])
#             row["energy_quantiles"] = torch.quantile(true.detach(), energy_levels)
#             row["margin_quantiles"] = torch.quantile(margin.detach(), margin_levels)
#             output[int(class_id)] = row
#         return output

#     # ------------------------------------------------------------------
#     # Admission and artifacts
#     # ------------------------------------------------------------------
#     def _incremental_admission_certificate(
#         self,
#         phase: int,
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         crossfit: Mapping[str, Any],
#         current_validation: Mapping[str, Any],
#         cumulative_validation: Mapping[str, Any],
#         old_integrity: Mapping[str, Any],
#     ) -> Dict[str, Any]:
#         checks = {
#             "crossfit_accuracy": float(crossfit["accuracy"])
#             >= self._inc_float("incremental_min_crossfit_accuracy", 0.0),
#             "crossfit_minimum_per_class_accuracy": float(
#                 crossfit["minimum_per_class_accuracy"]
#             )
#             >= self._inc_float(
#                 "incremental_min_crossfit_min_class_accuracy", 0.0
#             ),
#             "crossfit_classification_violation": float(
#                 crossfit["classification_violation_rate"]
#             )
#             <= self._inc_float(
#                 "incremental_max_crossfit_classification_violation", 1.0
#             ),
#             "crossfit_certificate_violation": float(
#                 crossfit["certificate_violation_rate"]
#             )
#             <= self._inc_float(
#                 "incremental_max_crossfit_certificate_violation", 1.0
#             ),
#             "cumulative_harmonic_mean": float(
#                 cumulative_validation["old_new_harmonic_mean"]
#             )
#             >= self._inc_float("incremental_min_old_new_harmonic_mean", 0.0),
#             "old_to_new_invasion": float(
#                 cumulative_validation["old_to_new_invasion_rate"]
#             )
#             <= self._inc_float("incremental_max_old_to_new_invasion", 1.0),
#             "new_to_old_invasion": float(
#                 cumulative_validation["new_to_old_invasion_rate"]
#             )
#             <= self._inc_float("incremental_max_new_to_old_invasion", 1.0),
#         }
#         enforce = self._inc_bool("incremental_admission_enforce", False)
#         valid = bool(all(checks.values())) if enforce else True
#         return {
#             "phase": int(phase),
#             "method": self.METHOD_NAME,
#             "schema_version": self.BANK_SCHEMA_VERSION,
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "old_classes": list(old_classes),
#             "new_classes": list(new_classes),
#             "checks": checks,
#             "enforced": enforce,
#             "valid": valid,
#             "crossfit": dict(crossfit),
#             "current_validation": dict(current_validation),
#             "cumulative_validation": dict(cumulative_validation),
#             "old_integrity": {
#                 "geometry_bank_contract_digest": old_integrity.get(
#                     "geometry_bank_contract_digest"
#                 ),
#                 "classifier_bound_contract_digest": old_integrity.get(
#                     "classifier_bound_contract_digest"
#                 ),
#                 "model_contract_digest": old_integrity.get("model_contract_digest"),
#             },
#             "selection_protocol": (
#                 "fixed_steps_final_state_no_validation_selected_checkpoint"
#             ),
#             "uses_raw_old_examples": False,
#             "uses_stored_old_features": False,
#             "uses_coupled_geometry_replay": True,
#             "refines_bases": False,
#             "refines_coupling": False,
#         }

#     @staticmethod
#     def _incremental_json_safe(value: Any) -> Any:
#         if torch.is_tensor(value):
#             tensor = value.detach().cpu()
#             return tensor.item() if tensor.numel() == 1 else tensor.tolist()
#         if isinstance(value, Mapping):
#             return {
#                 str(key): IncrementalPhaseTrainer._incremental_json_safe(item)
#                 for key, item in value.items()
#             }
#         if isinstance(value, (list, tuple)):
#             return [
#                 IncrementalPhaseTrainer._incremental_json_safe(item)
#                 for item in value
#             ]
#         if isinstance(value, (str, int, float, bool)) or value is None:
#             return value
#         return str(value)

#     def _save_incremental_reports(
#         self,
#         phase: int,
#         history: Mapping[str, Any],
#         certificate: Mapping[str, Any],
#         commit_report: Mapping[str, Any],
#     ) -> Dict[str, str]:
#         phase_dir = os.path.join(
#             os.path.abspath(str(self._inc_value("save_dir"))), f"phase_{int(phase)}"
#         )
#         reports_dir = os.path.join(phase_dir, "reports")
#         os.makedirs(reports_dir, exist_ok=True)
#         paths = {
#             "history": os.path.join(reports_dir, "incremental_history.json"),
#             "certificate": os.path.join(
#                 reports_dir, "incremental_admission_certificate.json"
#             ),
#             "handoff": os.path.join(phase_dir, "incremental_handoff.pt"),
#         }
#         with open(paths["history"], "w", encoding="utf-8") as stream:
#             json.dump(self._incremental_json_safe(history), stream, indent=2)
#         with open(paths["certificate"], "w", encoding="utf-8") as stream:
#             json.dump(self._incremental_json_safe(certificate), stream, indent=2)
#         torch.save(
#             self._incremental_json_safe(
#                 {
#                     "phase": phase,
#                     "certificate": certificate,
#                     "commit_report": commit_report,
#                 }
#             ),
#             paths["handoff"],
#         )
#         return paths

#     # ------------------------------------------------------------------
#     # Main phase
#     # ------------------------------------------------------------------
#     def train_incremental_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int = 64,
#         lr: float = 0.0,
#     ) -> Dict[str, Any]:
#         del lr
#         phase = int(phase)
#         epochs = int(epochs)
#         if phase <= 0:
#             raise ValueError("train_incremental_phase requires phase >= 1")
#         if epochs < 0:
#             raise ValueError("incremental epochs must be non-negative")
#         if int(batch_size) <= 0:
#             raise ValueError("batch_size must be positive")
#         self._validate_incremental_configuration(epochs)

#         old_classes, new_classes, seen_classes = self.resolve_phase_classes(phase)
#         self.dataset.start_phase(phase)
#         self._enter_incremental_mode(
#             phase, old_classes, new_classes, seen_classes
#         )
#         self.assert_bank_has_only_allowed_valid_rows(None, old_classes)
#         valid = self.model.geometry_bank.get_valid_mask()
#         occupied = [
#             class_id
#             for class_id in new_classes
#             if class_id < valid.numel() and bool(valid[class_id].item())
#         ]
#         if occupied:
#             raise RuntimeError(
#                 f"Incremental phase refuses to overwrite rows {occupied}"
#             )

#         old_integrity = self._old_bank_integrity_snapshot(old_classes)
#         contract_digest = self.model.geometry_bank.contract_digest()

#         train_loader = self.dataset.get_phase_dataloader(
#             phase, split="train", batch_size=int(batch_size), shuffle=False
#         )
#         current_val_loader = self.dataset.get_phase_dataloader(
#             phase, split="val", batch_size=int(batch_size), shuffle=False
#         )
#         cumulative_val_loader = self.dataset.get_cumulative_dataloader(
#             phase, split="val", batch_size=int(batch_size), shuffle=False
#         )
#         payload = self._collect_incremental_payload(
#             train_loader,
#             new_classes,
#             context=f"phase_{phase}.new_train",
#         )
#         initial_rows = self._build_incremental_candidate_rows(
#             payload, new_classes
#         )
#         contexts = self._build_incremental_crossfit_contexts(
#             payload, new_classes
#         )
#         generator = torch.Generator(device=self.device.type).manual_seed(
#             self._inc_int("seed", 0) + 65537 * phase
#         )
#         old_replay = self.sample_coupled_geometry_replay(
#             old_classes,
#             samples_per_class=self._inc_int(
#                 "incremental_old_replay_samples_per_class", 32
#             ),
#             seen_classes=seen_classes,
#             reliability_gated=self._inc_bool(
#                 "incremental_replay_reliability_gated", True
#             ),
#             generator=generator,
#         )
#         if old_replay.get("replay_factorization") != "z_then_g_given_z_c":
#             raise RuntimeError("Old replay is not occupancy-conditioned tangent replay")

#         parameters = self._make_incremental_descriptor_parameters(
#             initial_rows, new_classes
#         )
#         trainable = [
#             parameter
#             for group in parameters.values()
#             for parameter in group.parameters()
#         ]
#         optimizer = optim.Adam(
#             trainable,
#             lr=self._inc_float("descriptor_lr", 1e-3),
#             weight_decay=0.0,
#         )

#         history: Dict[str, Any] = {
#             "phase": phase,
#             "method": self.METHOD_NAME,
#             "schema_version": self.BANK_SCHEMA_VERSION,
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "old_classes": list(old_classes),
#             "new_classes": list(new_classes),
#             "seen_classes": list(seen_classes),
#             "selection_protocol": (
#                 "fixed_steps_final_state_no_validation_selected_checkpoint"
#             ),
#             "refines_bases": False,
#             "refines_coupling": False,
#             "loss": [],
#             "new_joint": [],
#             "old_replay_joint": [],
#             "crossfit_joint": [],
#             "boundary_certificate": [],
#             "two_sided_invasion": [],
#             "trust": [],
#             "new_classification_violation": [],
#             "old_replay_classification_violation": [],
#             "crossfit_classification_violation": [],
#             "certificate_violation_rate": [],
#             "minimum_certificate_lcb": [],
#             "old_to_new_invasion_rate": [],
#             "new_to_old_invasion_rate": [],
#         }

#         epoch_count = max(epochs, 1)
#         final_rows: Dict[int, Dict[str, Any]] = dict(initial_rows)
#         final_parts: Dict[str, torch.Tensor] = {}
#         for epoch in range(epoch_count):
#             if epochs > 0:
#                 for _ in range(self._inc_int("incremental_steps_per_epoch", 10)):
#                     optimizer.zero_grad(set_to_none=True)
#                     total, _, _ = self._incremental_objective(
#                         payload,
#                         old_replay,
#                         initial_rows,
#                         parameters,
#                         contexts,
#                         old_classes,
#                         new_classes,
#                         seen_classes,
#                     )
#                     total.backward()
#                     gradient_norm = torch.nn.utils.clip_grad_norm_(
#                         trainable,
#                         self._inc_float("incremental_grad_clip", 5.0),
#                     )
#                     if not torch.isfinite(torch.as_tensor(gradient_norm)):
#                         raise RuntimeError("Incremental descriptor gradient is invalid")
#                     optimizer.step()
#                     self._project_incremental_parameters(parameters, new_classes)

#             _, final_parts, final_rows = self._incremental_objective(
#                 payload,
#                 old_replay,
#                 initial_rows,
#                 parameters,
#                 contexts,
#                 old_classes,
#                 new_classes,
#                 seen_classes,
#             )
#             for key in (
#                 "total",
#                 "new_joint",
#                 "old_replay_joint",
#                 "crossfit_joint",
#                 "boundary_certificate",
#                 "two_sided_invasion",
#                 "trust",
#                 "new_classification_violation",
#                 "old_replay_classification_violation",
#                 "crossfit_classification_violation",
#                 "certificate_violation_rate",
#                 "minimum_certificate_lcb",
#                 "old_to_new_invasion_rate",
#                 "new_to_old_invasion_rate",
#             ):
#                 history_key = "loss" if key == "total" else key
#                 history[history_key].append(
#                     float(final_parts[key].detach().cpu().item())
#                 )
#             print(
#                 f"[PC-STGB Incremental] phase={phase} "
#                 f"epoch={epoch + 1}/{epoch_count} "
#                 f"loss={history['loss'][-1]:.4f} "
#                 f"newViol={history['new_classification_violation'][-1]:.3f} "
#                 f"oldInv={history['old_to_new_invasion_rate'][-1]:.3f} "
#                 f"newInv={history['new_to_old_invasion_rate'][-1]:.3f} "
#                 f"certViol={history['certificate_violation_rate'][-1]:.3f}"
#             )

#         self._assert_old_bank_integrity(
#             old_classes,
#             old_integrity,
#             context=f"phase_{phase}.post_optimization",
#         )
#         if self.model.geometry_bank.contract_digest() != contract_digest:
#             raise RuntimeError("Phase-invariant bank contract changed during optimization")

#         crossfit = self._incremental_crossfit_certificate(
#             contexts,
#             parameters,
#             old_classes,
#             new_classes,
#             seen_classes,
#         )
#         current_validation = self._evaluate_incremental_candidate_loader(
#             current_val_loader,
#             final_rows,
#             old_classes,
#             new_classes,
#             seen_classes,
#             allowed_classes=new_classes,
#             context=f"phase_{phase}.current_validation",
#         )
#         cumulative_validation = self._evaluate_incremental_candidate_loader(
#             cumulative_val_loader,
#             final_rows,
#             old_classes,
#             new_classes,
#             seen_classes,
#             allowed_classes=seen_classes,
#             context=f"phase_{phase}.cumulative_validation",
#         )
#         certificate = self._incremental_admission_certificate(
#             phase,
#             old_classes,
#             new_classes,
#             crossfit,
#             current_validation,
#             cumulative_validation,
#             old_integrity,
#         )
#         history["crossfit_certificate"] = crossfit
#         history["current_validation"] = current_validation
#         history["cumulative_candidate_validation"] = cumulative_validation
#         history["admission_certificate"] = certificate

#         if not bool(certificate["valid"]):
#             failed = [name for name, passed in certificate["checks"].items() if not passed]
#             history.update(
#                 {
#                     "status": "REJECTED",
#                     "committed": False,
#                     "phase_completed": False,
#                     "protocol_stop": True,
#                     "failed_checks": failed,
#                 }
#             )
#             commit_report = {
#                 "committed": False,
#                 "reason": "admission_checks_failed",
#                 "failed_checks": failed,
#             }
#             history["artifact_paths"] = self._save_incremental_reports(
#                 phase, history, certificate, commit_report
#             )
#             return history

#         final_rows = self._attach_incremental_statistics(
#             final_rows,
#             payload,
#             old_classes,
#             new_classes,
#             seen_classes,
#         )
#         commit_report = self.commit_new_class_rows_only(
#             new_classes,
#             final_rows,
#             phase_created=phase,
#             context=f"phase_{phase}.pc_stgb_commit",
#         )
#         self._assert_old_bank_integrity(
#             old_classes,
#             old_integrity,
#             context=f"phase_{phase}.post_commit",
#         )
#         if self.model.geometry_bank.contract_digest() != contract_digest:
#             raise RuntimeError("Phase-invariant bank contract changed during commit")

#         phase_certificate = self.geometry_phase_certificate(
#             phase,
#             seen_classes,
#             require_statistics=True,
#             require_response=True,
#             require_frozen=True,
#         )
#         if not bool(phase_certificate.get("ok", False)):
#             raise RuntimeError(
#                 f"Committed phase geometry is invalid: {phase_certificate.get('errors', [])}"
#             )
#         committed_metrics = self._evaluate_committed_incremental_loader(
#             cumulative_val_loader,
#             old_classes,
#             new_classes,
#             seen_classes,
#             context=f"phase_{phase}.committed_validation",
#         )
#         if hasattr(self.dataset, "finalize_phase"):
#             self.dataset.finalize_phase(phase)

#         history.update(
#             {
#                 "status": "COMMITTED",
#                 "committed": True,
#                 "phase_completed": True,
#                 "protocol_stop": False,
#                 "failed_checks": [],
#                 "commit_report": commit_report,
#                 "phase_geometry_certificate": phase_certificate,
#                 "final_metrics": committed_metrics,
#             }
#         )
#         history["artifact_paths"] = self._save_incremental_reports(
#             phase, history, certificate, commit_report
#         )
#         save_checkpoint = getattr(self, "save_checkpoint", None)
#         if callable(save_checkpoint):
#             history["checkpoint_path"] = save_checkpoint(phase, history)
#         return history

#     @torch.no_grad()
#     def _evaluate_committed_incremental_loader(
#         self,
#         loader: Any,
#         old_classes: Sequence[int],
#         new_classes: Sequence[int],
#         seen_classes: Sequence[int],
#         *,
#         context: str,
#     ) -> Dict[str, Any]:
#         energies: List[torch.Tensor] = []
#         labels: List[torch.Tensor] = []
#         previous = bool(self.model.training)
#         self.model.eval()
#         try:
#             for batch in loader:
#                 payload = self._extract_model_geometry_tuple(
#                     batch,
#                     require_grad=False,
#                     require_response_views=True,
#                     context=context,
#                 )
#                 scored = self.model.compute_logits_from_features(
#                     payload["features"],
#                     spectral_responses=payload["spectral_responses"],
#                     seen_classes=seen_classes,
#                     mode="pc_stgb",
#                     return_energy=True,
#                     return_parts=True,
#                 )
#                 energies.append(scored["joint_energy"].detach())
#                 labels.append(payload["labels"].detach())
#         finally:
#             self.model.train(previous)
#         energy = torch.cat(energies, dim=0)
#         labels_global = torch.cat(labels, dim=0)
#         targets = self.global_to_seen_local(
#             labels_global, seen_classes, context=f"{context}.targets"
#         )
#         metrics = self._energy_metrics(energy, targets)
#         prediction_global = self.seen_local_to_global(
#             metrics["prediction"], seen_classes
#         )
#         old_mask = torch.zeros_like(labels_global, dtype=torch.bool)
#         new_mask = torch.zeros_like(labels_global, dtype=torch.bool)
#         for class_id in old_classes:
#             old_mask |= labels_global.eq(int(class_id))
#         for class_id in new_classes:
#             new_mask |= labels_global.eq(int(class_id))

#         def accuracy(mask: torch.Tensor) -> float:
#             return 100.0 * float(
#                 prediction_global[mask].eq(labels_global[mask]).float().mean().item()
#             )

#         old_accuracy = accuracy(old_mask)
#         new_accuracy = accuracy(new_mask)
#         harmonic = (
#             0.0
#             if old_accuracy + new_accuracy <= 0.0
#             else 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)
#         )
#         old_set = set(old_classes)
#         new_set = set(new_classes)
#         old_to_new = sum(
#             int(int(value) in new_set)
#             for value in prediction_global[old_mask].detach().cpu().tolist()
#         ) / max(int(old_mask.sum().item()), 1)
#         new_to_old = sum(
#             int(int(value) in old_set)
#             for value in prediction_global[new_mask].detach().cpu().tolist()
#         ) / max(int(new_mask.sum().item()), 1)
#         per_class = {}
#         for class_id in seen_classes:
#             mask = labels_global.eq(int(class_id))
#             per_class[int(class_id)] = 100.0 * float(
#                 prediction_global[mask].eq(labels_global[mask]).float().mean().item()
#             )
#         return {
#             "accuracy": metrics["accuracy"],
#             "old_accuracy": old_accuracy,
#             "new_accuracy": new_accuracy,
#             "old_new_harmonic_mean": harmonic,
#             "minimum_per_class_accuracy": min(per_class.values()),
#             "per_class_accuracy": per_class,
#             "classification_violation_rate": metrics[
#                 "classification_violation_rate"
#             ],
#             "q05_gap": metrics["q05_gap"],
#             "old_to_new_invasion_rate": float(old_to_new),
#             "new_to_old_invasion_rate": float(new_to_old),
#         }
