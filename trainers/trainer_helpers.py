from __future__ import annotations

"""Shared runtime utilities for one-space HSI pairwise decision geometry.

The helper owns protocol/runtime validation and evaluation only.  It does not
implement continual-learning losses.  Evaluation reports the deployed equal-rule
classifier together with geometry diagnostics that match the current architecture:
class-cell coverage, rival invasion, no-cell rate, and true pairwise violations.
"""

from contextlib import contextmanager
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from models.geometry_bank import BoundaryCandidate, BoundaryGeometryBank

Tensor = torch.Tensor


class TrainerHelper:
    model: Any
    dataset: Any
    args: Any
    device: torch.device

    def cfg(self, name: str, default: Any) -> Any:
        return getattr(self.args, name, default)

    @staticmethod
    def _exact_nonnegative_int(value: object, *, name: str) -> int:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(f"{name} must be an integer")
            value = value.item()
        if isinstance(value, bool):
            raise RuntimeError(f"{name} must be an integer")
        if isinstance(value, Integral):
            result = int(value)
        elif isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                raise RuntimeError(f"{name} must be an integer")
            result = int(number)
        else:
            raise RuntimeError(f"{name} must be an integer")
        if result < 0:
            raise RuntimeError(f"{name} must be non-negative")
        return result

    @classmethod
    def _validated_class_ids(cls, values: Iterable[int], *, name: str) -> list[int]:
        ids = [cls._exact_nonnegative_int(value, name=name) for value in values]
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"{name} must contain unique non-negative IDs")
        return ids

    @staticmethod
    def _canonical_pair_list(
        class_ids: Sequence[int],
    ) -> list[tuple[int, int]]:
        ids = [int(value) for value in class_ids]
        return [
            (ids[left], ids[right])
            for left in range(len(ids))
            for right in range(left + 1, len(ids))
        ]

    def assert_dataset_contract(self) -> Dict[str, Any]:
        phase_map = getattr(self.dataset, "phase_to_classes", None)
        if not isinstance(phase_map, Mapping) or 0 not in phase_map:
            raise RuntimeError("dataset.phase_to_classes must define phase 0")

        schedule: Dict[int, list[int]] = {}
        used: set[int] = set()
        for raw_phase, raw_classes in phase_map.items():
            phase = self._exact_nonnegative_int(raw_phase, name="phase_id")
            ids = self._validated_class_ids(
                raw_classes, name=f"phase_{phase}_classes"
            )
            duplicate = used.intersection(ids)
            if duplicate:
                raise RuntimeError(
                    f"classes occur in multiple phases: {sorted(duplicate)}"
                )
            schedule[phase] = ids
            used.update(ids)

        if sorted(schedule) != list(range(len(schedule))):
            raise RuntimeError("phase IDs must be contiguous from zero")

        required = (
            "start_phase",
            "finalize_phase",
            "get_phase_dataloader",
            "get_cumulative_dataloader",
            "get_old_classes",
            "get_new_classes",
            "get_seen_classes",
            "assert_exemplar_free_contract",
        )
        missing = [
            name for name in required
            if not callable(getattr(self.dataset, name, None))
        ]
        if missing:
            raise RuntimeError(f"dataset lacks required APIs: {missing}")
        if self.dataset.assert_exemplar_free_contract() is not True:
            raise RuntimeError("dataset reports an exemplar-free protocol violation")

        base = self._validated_class_ids(
            self.dataset.get_new_classes(0), name="base_classes"
        )
        if base != schedule[0]:
            raise RuntimeError("phase map and get_new_classes(0) disagree")
        if len(base) < 2:
            raise RuntimeError(
                "pairwise decision geometry requires at least two base classes"
            )

        processed = getattr(self.dataset, "processed_cube", None)
        ordered = getattr(self.dataset, "ordered_spectral_cube", None)
        backbone = getattr(self.model, "backbone", None)
        if processed is None or ordered is None or backbone is None:
            raise RuntimeError("dataset/model lacks HSI cube or backbone state")
        if getattr(processed, "ndim", None) != 3 or getattr(ordered, "ndim", None) != 3:
            raise RuntimeError("HSI cubes must be rank three")
        if int(processed.shape[2]) != int(backbone.patch_bands):
            raise RuntimeError(
                "processed cube channels disagree with backbone.patch_bands"
            )
        if int(ordered.shape[2]) != int(backbone.spectral_bands):
            raise RuntimeError(
                "ordered cube bands disagree with backbone.spectral_bands"
            )
        if int(getattr(self.dataset, "patch_size", 0)) != int(backbone.patch_size):
            raise RuntimeError("dataset/backbone patch sizes disagree")

        return {
            "schedule": dict(sorted(schedule.items())),
            "base_classes": list(base),
            "protocol_report": True,
        }

    def assert_model_contract(self) -> Dict[str, Any]:
        required = (
            "encode",
            "forward",
            "initialize_candidate",
            "classify_coordinates",
            "commit_candidate",
            "pair_values",
            "class_boundary_response",
            "true_pair_margins",
            "validate_geometry",
            "validate_model_state",
        )
        missing = [
            name for name in required
            if not callable(getattr(self.model, name, None))
        ]
        if missing:
            raise RuntimeError(f"model lacks required APIs: {missing}")

        backbone = getattr(self.model, "backbone", None)
        bank = getattr(self.model, "geometry_bank", None)
        classifier = getattr(self.model, "classifier", None)
        if backbone is None or bank is None or classifier is None:
            raise RuntimeError("model must own backbone, geometry_bank, classifier")
        if not isinstance(bank, BoundaryGeometryBank):
            raise RuntimeError("model geometry_bank must be BoundaryGeometryBank")

        for name in (
            "patch_bands",
            "spectral_bands",
            "patch_size",
            "representation_dim",
        ):
            if int(getattr(backbone, name, 0)) <= 0:
                raise RuntimeError(f"backbone.{name} must be positive")

        representation_dim = int(backbone.representation_dim)
        if hasattr(backbone, "coordinate_projection"):
            raise RuntimeError(
                "one-space backbone must not expose coordinate_projection"
            )
        context_refiner = getattr(backbone, "context_refiner", None)
        if context_refiner is None:
            raise RuntimeError("backbone must expose context_refiner")
        if int(getattr(context_refiner, "spectral_dim", 0)) != representation_dim:
            raise RuntimeError(
                "context correction must use the canonical representation"
            )
        if int(bank.representation_dim) != representation_dim:
            raise RuntimeError("bank/backbone representation dimensions disagree")
        if any(True for _ in classifier.parameters()):
            raise RuntimeError("GeometryClassifier must remain parameter-free")
        if torch.device(self.model.device) != self.device or bank.device != self.device:
            raise RuntimeError("model/geometry are not on trainer device")
        if getattr(self.model, "dtype", bank.dtype) != bank.dtype:
            raise RuntimeError("model/geometry floating dtypes disagree")

        self.model.validate_model_state()
        return {
            "architecture": "one_space_hsi_pairwise_decision_geometry",
            "patch_bands": int(backbone.patch_bands),
            "spectral_bands": int(backbone.spectral_bands),
            "patch_size": int(backbone.patch_size),
            "representation_dim": representation_dim,
            "context_input_channels": int(context_refiner.input_channels),
            "context_spectral_dim": int(context_refiner.spectral_dim),
            "classifier_parameter_count": 0,
            "geometry_interface": (
                "pair_values + class_boundary_response + true_pair_margins"
            ),
            "spectral_normalization_fitted": bool(
                self.model.spectral_normalization_fitted
            ),
        }

    def unpack_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("HSI batches must be mappings")
        required = {"image", "raw_center_spectrum", "label"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"batch missing {sorted(missing)}")

        model_dtype = getattr(self.model.geometry_bank, "dtype", torch.float32)
        patch = torch.as_tensor(
            batch["image"], device=self.device, dtype=model_dtype
        )
        spectrum = torch.as_tensor(
            batch["raw_center_spectrum"],
            device=self.device,
            dtype=model_dtype,
        )
        labels = torch.as_tensor(batch["label"], device=self.device).flatten()

        backbone = self.model.backbone
        expected_patch = (
            patch.size(0) if patch.ndim else 0,
            int(backbone.patch_bands),
            int(backbone.patch_size),
            int(backbone.patch_size),
        )
        expected_spectrum = (
            patch.size(0) if patch.ndim else 0,
            int(backbone.spectral_bands),
        )
        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape) != expected_patch
        ):
            raise RuntimeError("processed HSI patch shape is invalid")
        if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
            raise RuntimeError("raw center spectrum shape is invalid")
        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(spectrum).all()
        ):
            raise RuntimeError("HSI batch contains NaN/Inf")
        if (
            labels.numel() != patch.size(0)
            or labels.dtype == torch.bool
            or labels.is_complex()
        ):
            raise RuntimeError("labels are invalid or batch-misaligned")
        if torch.is_floating_point(labels):
            if not bool(torch.isfinite(labels).all()) or not bool(
                labels.eq(labels.round()).all()
            ):
                raise RuntimeError("labels must be finite integers")
        labels = labels.to(dtype=torch.long)
        if bool((labels < 0).any()):
            raise RuntimeError("labels must be non-negative")
        return patch, spectrum, labels

    @torch.no_grad()
    def collect_labels(self, loader: Any) -> Tensor:
        labels: list[Tensor] = []
        for batch in loader:
            _, _, values = self.unpack_batch(batch)
            labels.append(values.detach().cpu())
        if not labels:
            raise RuntimeError("cannot collect labels from an empty loader")
        return torch.cat(labels, dim=0)

    @torch.no_grad()
    def collect_encoded(self, loader: Any) -> Dict[str, Tensor]:
        coordinates: list[Tensor] = []
        labels: list[Tensor] = []
        states = {
            module: bool(module.training)
            for module in self.model.modules()
        }
        try:
            self.model.eval()
            for batch in loader:
                patch, spectrum, values = self.unpack_batch(batch)
                output = self.model.encode(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                coordinates.append(output.coordinates.detach().cpu())
                labels.append(values.detach().cpu())
        finally:
            for module, state in states.items():
                module.training = state

        if not coordinates:
            raise RuntimeError("cannot encode an empty loader")
        z = torch.cat(coordinates, dim=0)
        y = torch.cat(labels, dim=0)
        if z.ndim != 2 or z.size(0) != y.numel():
            raise RuntimeError("collected representation is row-misaligned")
        if z.size(1) != int(self.model.representation_dim):
            raise RuntimeError("collected representation dimension is invalid")
        if not bool(torch.isfinite(z).all()):
            raise RuntimeError("collected representation contains NaN/Inf")
        return {"coordinates": z, "labels": y}

    @contextmanager
    def _temporary_eval_state(
        self,
        candidate: Optional[BoundaryCandidate],
    ):
        model_states = {
            module: bool(module.training)
            for module in self.model.modules()
        }
        candidate_state = None if candidate is None else bool(candidate.training)
        try:
            self.model.eval()
            if candidate is not None:
                candidate.eval()
            yield
        finally:
            for module, state in model_states.items():
                module.training = state
            if candidate is not None and candidate_state is not None:
                candidate.training = candidate_state

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: Any,
        *,
        class_ids: Sequence[int],
        target_class_ids: Optional[Sequence[int]] = None,
        candidate: Optional[BoundaryCandidate] = None,
        geometry_bank: Optional[BoundaryGeometryBank] = None,
    ) -> Dict[str, Any]:
        """Evaluate the deployed rule and decision-geometry diagnostics.

        ``target_class_ids`` controls which labels are expected/reported, while
        every sample is classified against the complete ``class_ids`` set.
        Pair-violation statistics use the true class's oriented incident
        boundaries and therefore directly measure decision-relevant overlap.
        """
        requested = self._validated_class_ids(
            class_ids, name="evaluation_class_ids"
        )
        targets_requested = (
            requested
            if target_class_ids is None
            else self._validated_class_ids(
                target_class_ids, name="target_class_ids"
            )
        )
        if not set(targets_requested).issubset(requested):
            raise ValueError(
                "target_class_ids must be a subset of evaluation class_ids"
            )
        if candidate is not None and geometry_bank is not None:
            raise ValueError("candidate and geometry_bank are mutually exclusive")

        active_bank = self.model.geometry_bank if geometry_bank is None else geometry_bank
        active_candidate = candidate if geometry_bank is None else None
        active_bank.validate_bank_state()
        if active_bank.device != self.model.geometry_bank.device:
            raise ValueError("evaluation geometry bank is on the wrong device")
        if active_bank.dtype != self.model.geometry_bank.dtype:
            raise ValueError("evaluation geometry bank uses the wrong dtype")
        if int(active_bank.representation_dim) != int(self.model.representation_dim):
            raise ValueError(
                "evaluation geometry has the wrong representation dimension"
            )

        total = 0
        correct = 0
        ce_sum = 0.0
        true_energy_sum = 0.0
        true_cell_violation_sum = 0.0
        rival_energy_sum = 0.0
        margin_sum = 0.0
        true_inside_count = 0
        rival_inside_count = 0
        no_cell_count = 0
        pair_violation_count = 0
        pair_relation_count = 0
        min_true_pair_margin_sum = 0.0
        has_rivals = len(requested) > 1

        diagnostic_pairs = self._canonical_pair_list(targets_requested)
        pair_accumulators: Dict[tuple[int, int], Dict[str, Any]] = {
            pair: {
                "left_count": 0,
                "right_count": 0,
                "left_violation_count": 0,
                "right_violation_count": 0,
                "left_margin_sum": 0.0,
                "right_margin_sum": 0.0,
            }
            for pair in diagnostic_pairs
        }

        class_total = {class_id: 0 for class_id in targets_requested}
        class_correct = {class_id: 0 for class_id in targets_requested}
        class_ce = {class_id: 0.0 for class_id in targets_requested}
        class_cell_violation = {class_id: 0.0 for class_id in targets_requested}
        class_inside = {class_id: 0 for class_id in targets_requested}
        class_rival_inside = {class_id: 0 for class_id in targets_requested}
        class_no_cell = {class_id: 0 for class_id in targets_requested}
        class_true_energy = {class_id: 0.0 for class_id in targets_requested}
        class_rival_energy = {class_id: 0.0 for class_id in targets_requested}
        class_margin = {class_id: 0.0 for class_id in targets_requested}
        class_pair_violation = {class_id: 0 for class_id in targets_requested}
        class_pair_relations = {class_id: 0 for class_id in targets_requested}
        class_min_pair_margin = {class_id: 0.0 for class_id in targets_requested}

        with self._temporary_eval_state(candidate):
            for batch in loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                observed = set(
                    int(value)
                    for value in labels.unique().detach().cpu().tolist()
                )
                outside = sorted(observed - set(targets_requested))
                if outside:
                    raise RuntimeError(
                        "evaluation loader contains labels outside target classes: "
                        f"{outside}"
                    )

                representation = self.model.encode(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                if geometry_bank is None:
                    output = self.model.classify_coordinates(
                        representation.coordinates,
                        class_ids=requested,
                        candidate=candidate,
                    )
                else:
                    output = self.model.classifier(
                        representation.coordinates,
                        geometry_bank=active_bank,
                        class_ids=requested,
                        candidate=None,
                    )

                actual_ids = [
                    int(value)
                    for value in output.class_ids.detach().cpu().tolist()
                ]
                if actual_ids != requested:
                    raise RuntimeError(
                        "classifier columns do not match requested classes"
                    )

                targets = self.model.classifier.targets_local(
                    labels, output.class_ids
                )
                rows = torch.arange(
                    labels.numel(), device=output.energy.device
                )
                true_energy = output.energy[rows, targets]
                inside = true_energy <= 0
                no_cell = output.energy.amin(dim=1) > 0
                per_ce = F.cross_entropy(
                    output.logits, targets, reduction="none"
                )
                per_cell_violation = F.relu(true_energy)

                ce_sum += float(per_ce.sum().item())
                true_cell_violation_sum += float(per_cell_violation.sum().item())
                true_energy_sum += float(true_energy.sum().item())
                true_inside_count += int(inside.sum().item())
                no_cell_count += int(no_cell.sum().item())

                pair_margins = None
                per_sample_min_pair_margin = None
                if has_rivals:
                    response = active_bank.class_boundary_response(
                        representation.coordinates,
                        labels,
                        class_ids=requested,
                        candidate=active_candidate,
                    )
                    pair_margins = response.margins
                    if pair_margins.shape != (
                        labels.numel(), len(requested) - 1
                    ):
                        raise RuntimeError(
                            "evaluation pair-response shape is invalid"
                        )
                    violations = pair_margins < 0
                    pair_violation_count += int(violations.sum().item())
                    pair_relation_count += int(violations.numel())
                    per_sample_min_pair_margin = pair_margins.amin(dim=1)
                    min_true_pair_margin_sum += float(
                        per_sample_min_pair_margin.sum().item()
                    )

                rival_energy = None
                rival_inside = None
                margin = None
                if has_rivals:
                    mask = F.one_hot(
                        targets, num_classes=len(requested)
                    ).to(torch.bool)
                    rival_energy = output.energy.masked_fill(
                        mask, torch.inf
                    ).amin(dim=1)
                    rival_inside = rival_energy < 0
                    if bool((inside & rival_inside).any()):
                        raise RuntimeError(
                            "pairwise geometry invariant violated: a sample "
                            "lies in two strict class interiors"
                        )
                    margin = rival_energy - true_energy
                    rival_inside_count += int(rival_inside.sum().item())
                    rival_energy_sum += float(rival_energy.sum().item())
                    margin_sum += float(margin.sum().item())

                if diagnostic_pairs:
                    pair_values = active_bank.pair_values(
                        representation.coordinates,
                        pair_ids=diagnostic_pairs,
                        candidate=active_candidate,
                    )
                    expected_pair_ids = torch.tensor(
                        diagnostic_pairs,
                        device=pair_values.pair_ids.device,
                        dtype=torch.long,
                    )
                    if pair_values.values.shape != (
                        labels.numel(), len(diagnostic_pairs)
                    ):
                        raise RuntimeError(
                            "pairwise diagnostic values have invalid shape"
                        )
                    if not bool(torch.equal(pair_values.pair_ids, expected_pair_ids)):
                        raise RuntimeError(
                            "pairwise diagnostics returned unexpected pair order"
                        )

                    for pair_index, pair in enumerate(diagnostic_pairs):
                        left_id, right_id = pair
                        h = pair_values.values[:, pair_index]
                        stats = pair_accumulators[pair]
                        left_mask = labels.eq(left_id)
                        right_mask = labels.eq(right_id)
                        left_count = int(left_mask.sum().item())
                        if left_count:
                            left_margin = h[left_mask]
                            stats["left_count"] += left_count
                            stats["left_violation_count"] += int(
                                left_margin.lt(0).sum().item()
                            )
                            stats["left_margin_sum"] += float(
                                left_margin.sum().item()
                            )
                        right_count = int(right_mask.sum().item())
                        if right_count:
                            right_margin = -h[right_mask]
                            stats["right_count"] += right_count
                            stats["right_violation_count"] += int(
                                right_margin.lt(0).sum().item()
                            )
                            stats["right_margin_sum"] += float(
                                right_margin.sum().item()
                            )

                prediction = output.prediction
                count = int(labels.numel())
                total += count
                correct += int(prediction.eq(labels).sum().item())

                for class_id in targets_requested:
                    class_mask = labels.eq(class_id)
                    count_class = int(class_mask.sum().item())
                    if count_class == 0:
                        continue
                    class_total[class_id] += count_class
                    class_correct[class_id] += int(
                        prediction[class_mask]
                        .eq(labels[class_mask])
                        .sum()
                        .item()
                    )
                    class_ce[class_id] += float(
                        per_ce[class_mask].sum().item()
                    )
                    class_cell_violation[class_id] += float(
                        per_cell_violation[class_mask].sum().item()
                    )
                    class_inside[class_id] += int(
                        inside[class_mask].sum().item()
                    )
                    class_no_cell[class_id] += int(
                        no_cell[class_mask].sum().item()
                    )
                    class_true_energy[class_id] += float(
                        true_energy[class_mask].sum().item()
                    )

                    if has_rivals:
                        assert (
                            rival_inside is not None
                            and rival_energy is not None
                            and margin is not None
                            and pair_margins is not None
                            and per_sample_min_pair_margin is not None
                        )
                        class_rival_inside[class_id] += int(
                            rival_inside[class_mask].sum().item()
                        )
                        class_rival_energy[class_id] += float(
                            rival_energy[class_mask].sum().item()
                        )
                        class_margin[class_id] += float(
                            margin[class_mask].sum().item()
                        )
                        class_pair_violation[class_id] += int(
                            pair_margins[class_mask].lt(0).sum().item()
                        )
                        class_pair_relations[class_id] += int(
                            pair_margins[class_mask].numel()
                        )
                        class_min_pair_margin[class_id] += float(
                            per_sample_min_pair_margin[class_mask].sum().item()
                        )

        if total == 0:
            raise RuntimeError("evaluation loader is empty")
        missing = [
            class_id
            for class_id in targets_requested
            if class_total[class_id] == 0
        ]
        if missing:
            raise RuntimeError(
                f"evaluation split is missing target classes: {missing}"
            )

        per_acc = {
            class_id: class_correct[class_id] / class_total[class_id]
            for class_id in targets_requested
        }
        per_ce_mean = {
            class_id: class_ce[class_id] / class_total[class_id]
            for class_id in targets_requested
        }
        per_cell_violation_mean = {
            class_id: class_cell_violation[class_id] / class_total[class_id]
            for class_id in targets_requested
        }
        per_cov = {
            class_id: class_inside[class_id] / class_total[class_id]
            for class_id in targets_requested
        }
        per_inv = {
            class_id: (
                class_rival_inside[class_id] / class_total[class_id]
                if has_rivals else 0.0
            )
            for class_id in targets_requested
        }
        per_no_cell = {
            class_id: class_no_cell[class_id] / class_total[class_id]
            for class_id in targets_requested
        }
        per_true = {
            class_id: class_true_energy[class_id] / class_total[class_id]
            for class_id in targets_requested
        }
        per_rival = {
            class_id: (
                class_rival_energy[class_id] / class_total[class_id]
                if has_rivals else None
            )
            for class_id in targets_requested
        }
        per_margin = {
            class_id: (
                class_margin[class_id] / class_total[class_id]
                if has_rivals else None
            )
            for class_id in targets_requested
        }
        per_pair_violation = {
            class_id: (
                class_pair_violation[class_id] / class_pair_relations[class_id]
                if has_rivals and class_pair_relations[class_id] > 0 else 0.0
            )
            for class_id in targets_requested
        }
        per_min_pair_margin = {
            class_id: (
                class_min_pair_margin[class_id] / class_total[class_id]
                if has_rivals else None
            )
            for class_id in targets_requested
        }

        def macro(values: Mapping[int, float]) -> float:
            return (
                sum(float(values[class_id]) for class_id in targets_requested)
                / len(targets_requested)
            )

        macro_rival = (
            macro({
                class_id: float(per_rival[class_id])
                for class_id in targets_requested
            })
            if has_rivals else None
        )
        macro_margin = (
            macro({
                class_id: float(per_margin[class_id])
                for class_id in targets_requested
            })
            if has_rivals else None
        )
        macro_min_pair_margin = (
            macro({
                class_id: float(per_min_pair_margin[class_id])
                for class_id in targets_requested
            })
            if has_rivals else None
        )

        pairwise_boundary_metrics: Dict[str, Any] = {}
        for left_id, right_id in diagnostic_pairs:
            stats = pair_accumulators[(left_id, right_id)]
            left_count = int(stats["left_count"])
            right_count = int(stats["right_count"])
            if left_count <= 0 or right_count <= 0:
                raise RuntimeError(
                    "pairwise diagnostics require evidence from both classes "
                    f"for pair ({left_id},{right_id})"
                )
            left_mean = float(stats["left_margin_sum"]) / left_count
            right_mean = float(stats["right_margin_sum"]) / right_count
            left_violation = int(stats["left_violation_count"]) / left_count
            right_violation = int(stats["right_violation_count"]) / right_count
            combined_violation = (
                int(stats["left_violation_count"])
                + int(stats["right_violation_count"])
            ) / (left_count + right_count)
            pairwise_boundary_metrics[f"{left_id}-{right_id}"] = {
                "left_class_id": left_id,
                "right_class_id": right_id,
                "left_count": left_count,
                "right_count": right_count,
                "left_violation_rate": left_violation,
                "right_violation_rate": right_violation,
                "combined_violation_rate": combined_violation,
                "left_mean_oriented_margin": left_mean,
                "right_mean_oriented_margin": right_mean,
                "mean_distribution_order_gap": left_mean + right_mean,
                "minimum_side_mean_margin": min(left_mean, right_mean),
            }

        if pairwise_boundary_metrics:
            worst_pair = max(
                pairwise_boundary_metrics,
                key=lambda key: pairwise_boundary_metrics[key][
                    "combined_violation_rate"
                ],
            )
            weakest_pair = min(
                pairwise_boundary_metrics,
                key=lambda key: pairwise_boundary_metrics[key][
                    "minimum_side_mean_margin"
                ],
            )
            pairwise_boundary_summary: Dict[str, Any] = {
                "pair_count": len(pairwise_boundary_metrics),
                "worst_violation_pair": worst_pair,
                "worst_combined_violation_rate": float(
                    pairwise_boundary_metrics[worst_pair][
                        "combined_violation_rate"
                    ]
                ),
                "weakest_mean_margin_pair": weakest_pair,
                "weakest_minimum_side_mean_margin": float(
                    pairwise_boundary_metrics[weakest_pair][
                        "minimum_side_mean_margin"
                    ]
                ),
            }
        else:
            pairwise_boundary_summary = {
                "pair_count": 0,
                "worst_violation_pair": None,
                "worst_combined_violation_rate": 0.0,
                "weakest_mean_margin_pair": None,
                "weakest_minimum_side_mean_margin": None,
            }

        return {
            "energy_convention": {
                "class_cell": (
                    "E_c(z) <= 0 iff all pairwise boundaries support class c"
                ),
                "class_score": (
                    "E_c(z) = -minimum oriented pairwise signed distance"
                ),
                "decision": "argmin_c E_c(z)",
                "decision_margin": "nearest_rival_energy - true_energy",
                "pair_violation": "s_yj(z) < 0",
                "no_cell": "min_c E_c(z) > 0",
            },
            "classification": ce_sum / total,
            "macro_classification": macro(per_ce_mean),
            # This remains a diagnostic alias only; it is no longer a training loss.
            "true_cell_violation": true_cell_violation_sum / total,
            "macro_true_cell_violation": macro(per_cell_violation_mean),
            "per_class_true_cell_violation": per_cell_violation_mean,
            "cell_fit": true_cell_violation_sum / total,
            "macro_cell_fit": macro(per_cell_violation_mean),
            "per_class_cell_fit": per_cell_violation_mean,
            "overall_accuracy": correct / total,
            "accuracy": correct / total,
            "balanced_accuracy": macro(per_acc),
            "minimum_class_accuracy": min(per_acc.values()),
            "evaluated_class_ids": list(requested),
            "target_class_ids": list(targets_requested),
            "class_counts": {
                class_id: class_total[class_id]
                for class_id in targets_requested
            },
            "per_class_accuracy": per_acc,
            "true_cell_coverage": true_inside_count / total,
            "macro_true_cell_coverage": macro(per_cov),
            "per_class_true_cell_coverage": per_cov,
            "rival_cell_invasion_rate": (
                rival_inside_count / total if has_rivals else 0.0
            ),
            "macro_rival_cell_invasion_rate": macro(per_inv),
            "per_class_rival_cell_invasion_rate": per_inv,
            "no_cell_rate": no_cell_count / total,
            "macro_no_cell_rate": macro(per_no_cell),
            "per_class_no_cell_rate": per_no_cell,
            "true_pair_violation_rate": (
                pair_violation_count / pair_relation_count
                if pair_relation_count else 0.0
            ),
            "macro_true_pair_violation_rate": macro(per_pair_violation),
            "per_class_true_pair_violation_rate": per_pair_violation,
            "pairwise_boundary_metrics": pairwise_boundary_metrics,
            "pairwise_boundary_summary": pairwise_boundary_summary,
            "mean_minimum_true_pair_margin": (
                min_true_pair_margin_sum / total if has_rivals else None
            ),
            "macro_mean_minimum_true_pair_margin": macro_min_pair_margin,
            "per_class_mean_minimum_true_pair_margin": per_min_pair_margin,
            "mean_true_energy": true_energy_sum / total,
            "macro_mean_true_energy": macro(per_true),
            "per_class_mean_true_energy": per_true,
            "mean_nearest_rival_energy": (
                rival_energy_sum / total if has_rivals else None
            ),
            "macro_mean_nearest_rival_energy": macro_rival,
            "per_class_mean_nearest_rival_energy": per_rival,
            "mean_decision_margin": (
                margin_sum / total if has_rivals else None
            ),
            "macro_mean_decision_margin": macro_margin,
            "per_class_mean_decision_margin": per_margin,
            "strict_cell_conflict_rate": 0.0,
        }

    @staticmethod
    def summarize_class_group(
        metrics: Mapping[str, Any],
        class_ids: Sequence[int],
    ) -> Dict[str, Any]:
        ids = [int(value) for value in class_ids]
        if not ids:
            raise ValueError("class group cannot be empty")

        def mapping(name: str) -> Mapping[Any, Any]:
            value = metrics.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"metrics lacks {name}")
            return value

        def value(row: Mapping[Any, Any], class_id: int) -> float:
            raw = row[class_id] if class_id in row else row.get(str(class_id))
            if raw is None:
                raise ValueError(f"metrics lacks class {class_id}")
            return float(raw)

        per_acc = mapping("per_class_accuracy")
        per_cov = mapping("per_class_true_cell_coverage")
        per_inv = mapping("per_class_rival_cell_invasion_rate")
        per_no_cell = mapping("per_class_no_cell_rate")
        per_pair = mapping("per_class_true_pair_violation_rate")
        per_cell = mapping("per_class_true_cell_violation")
        per_min_pair_margin = mapping(
            "per_class_mean_minimum_true_pair_margin"
        )
        per_margin = mapping("per_class_mean_decision_margin")

        return {
            "class_ids": ids,
            "balanced_accuracy": sum(value(per_acc, class_id) for class_id in ids) / len(ids),
            "minimum_class_accuracy": min(value(per_acc, class_id) for class_id in ids),
            "macro_true_cell_coverage": sum(value(per_cov, class_id) for class_id in ids) / len(ids),
            "macro_rival_cell_invasion_rate": sum(value(per_inv, class_id) for class_id in ids) / len(ids),
            "macro_no_cell_rate": sum(value(per_no_cell, class_id) for class_id in ids) / len(ids),
            "macro_true_pair_violation_rate": sum(value(per_pair, class_id) for class_id in ids) / len(ids),
            "macro_true_cell_violation": sum(value(per_cell, class_id) for class_id in ids) / len(ids),
            "macro_mean_minimum_true_pair_margin": sum(
                value(per_min_pair_margin, class_id) for class_id in ids
            ) / len(ids),
            # Compatibility alias for older reporting code only.
            "macro_cell_fit": sum(value(per_cell, class_id) for class_id in ids) / len(ids),
            "macro_mean_decision_margin": sum(value(per_margin, class_id) for class_id in ids) / len(ids),
        }

    def geometry_state_summary(self) -> Dict[str, Any]:
        bank = self.model.geometry_bank
        valid = bool(bank.validate_bank_state())
        class_ids = [
            int(value) for value in bank.class_ids.detach().cpu().tolist()
        ]
        norms = (
            torch.linalg.vector_norm(bank.normals, dim=1)
            if bank.pair_count
            else torch.empty(0)
        )
        expected_pair_count = len(class_ids) * (len(class_ids) - 1) // 2
        actual_pair_set = {
            tuple(map(int, row))
            for row in bank.pair_ids.detach().cpu().tolist()
        }
        expected_pair_set = set(self._canonical_pair_list(class_ids))
        complete_pair_geometry = (
            int(bank.pair_count) == expected_pair_count
            and actual_pair_set == expected_pair_set
        )
        if class_ids and not complete_pair_geometry:
            raise RuntimeError(
                "committed geometry is incomplete for its committed class set"
            )

        return {
            "structurally_valid": valid,
            "class_ids": class_ids,
            "class_count": len(bank),
            "pair_count": int(bank.pair_count),
            "expected_pair_count": int(expected_pair_count),
            "complete_pair_geometry": bool(complete_pair_geometry),
            "representation_dim": int(bank.representation_dim),
            "normal_norm_minimum": (
                None if norms.numel() == 0 else float(norms.amin().item())
            ),
            "normal_norm_maximum": (
                None if norms.numel() == 0 else float(norms.amax().item())
            ),
            "offset_minimum": (
                None if bank.offsets.numel() == 0
                else float(bank.offsets.amin().item())
            ),
            "offset_maximum": (
                None if bank.offsets.numel() == 0
                else float(bank.offsets.amax().item())
            ),
        }

    @staticmethod
    def json_safe(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
        except ImportError:
            pass
        if isinstance(value, Mapping):
            return {
                str(key): TrainerHelper.json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [TrainerHelper.json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def save_json(self, path: str, value: Mapping[str, Any]) -> str:
        destination = os.path.abspath(path)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        temporary = destination + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(
                    self.json_safe(value),
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


__all__ = ["TrainerHelper"]
















# from __future__ import annotations

# """Shared runtime utilities for one-space HSI pairwise decision geometry.

# The helper owns protocol/runtime validation and evaluation only.  It does not
# implement continual-learning losses.  Evaluation reports the deployed equal-rule
# classifier together with geometry diagnostics that match the current architecture:
# class-cell coverage, rival invasion, no-cell rate, and true pairwise violations.
# """

# from contextlib import contextmanager
# import json
# import math
# import os
# from numbers import Integral, Real
# from pathlib import Path
# from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import BoundaryCandidate, BoundaryGeometryBank

# Tensor = torch.Tensor


# class TrainerHelper:
#     model: Any
#     dataset: Any
#     args: Any
#     device: torch.device

#     def cfg(self, name: str, default: Any) -> Any:
#         return getattr(self.args, name, default)

#     @staticmethod
#     def _exact_nonnegative_int(value: object, *, name: str) -> int:
#         if torch.is_tensor(value):
#             if value.numel() != 1:
#                 raise RuntimeError(f"{name} must be an integer")
#             value = value.item()
#         if isinstance(value, bool):
#             raise RuntimeError(f"{name} must be an integer")
#         if isinstance(value, Integral):
#             result = int(value)
#         elif isinstance(value, Real):
#             number = float(value)
#             if not math.isfinite(number) or not number.is_integer():
#                 raise RuntimeError(f"{name} must be an integer")
#             result = int(number)
#         else:
#             raise RuntimeError(f"{name} must be an integer")
#         if result < 0:
#             raise RuntimeError(f"{name} must be non-negative")
#         return result

#     @classmethod
#     def _validated_class_ids(cls, values: Iterable[int], *, name: str) -> list[int]:
#         ids = [cls._exact_nonnegative_int(value, name=name) for value in values]
#         if not ids or len(ids) != len(set(ids)):
#             raise RuntimeError(f"{name} must contain unique non-negative IDs")
#         return ids

#     def assert_dataset_contract(self) -> Dict[str, Any]:
#         phase_map = getattr(self.dataset, "phase_to_classes", None)
#         if not isinstance(phase_map, Mapping) or 0 not in phase_map:
#             raise RuntimeError("dataset.phase_to_classes must define phase 0")

#         schedule: Dict[int, list[int]] = {}
#         used: set[int] = set()
#         for raw_phase, raw_classes in phase_map.items():
#             phase = self._exact_nonnegative_int(raw_phase, name="phase_id")
#             ids = self._validated_class_ids(
#                 raw_classes, name=f"phase_{phase}_classes"
#             )
#             duplicate = used.intersection(ids)
#             if duplicate:
#                 raise RuntimeError(
#                     f"classes occur in multiple phases: {sorted(duplicate)}"
#                 )
#             schedule[phase] = ids
#             used.update(ids)

#         if sorted(schedule) != list(range(len(schedule))):
#             raise RuntimeError("phase IDs must be contiguous from zero")

#         required = (
#             "start_phase",
#             "finalize_phase",
#             "get_phase_dataloader",
#             "get_cumulative_dataloader",
#             "get_old_classes",
#             "get_new_classes",
#             "get_seen_classes",
#             "assert_exemplar_free_contract",
#         )
#         missing = [
#             name for name in required
#             if not callable(getattr(self.dataset, name, None))
#         ]
#         if missing:
#             raise RuntimeError(f"dataset lacks required APIs: {missing}")
#         if self.dataset.assert_exemplar_free_contract() is not True:
#             raise RuntimeError("dataset reports an exemplar-free protocol violation")

#         base = self._validated_class_ids(
#             self.dataset.get_new_classes(0), name="base_classes"
#         )
#         if base != schedule[0]:
#             raise RuntimeError("phase map and get_new_classes(0) disagree")

#         processed = getattr(self.dataset, "processed_cube", None)
#         ordered = getattr(self.dataset, "ordered_spectral_cube", None)
#         backbone = getattr(self.model, "backbone", None)
#         if processed is None or ordered is None or backbone is None:
#             raise RuntimeError("dataset/model lacks HSI cube or backbone state")
#         if getattr(processed, "ndim", None) != 3 or getattr(ordered, "ndim", None) != 3:
#             raise RuntimeError("HSI cubes must be rank three")
#         if int(processed.shape[2]) != int(backbone.patch_bands):
#             raise RuntimeError(
#                 "processed cube channels disagree with backbone.patch_bands"
#             )
#         if int(ordered.shape[2]) != int(backbone.spectral_bands):
#             raise RuntimeError(
#                 "ordered cube bands disagree with backbone.spectral_bands"
#             )
#         if int(getattr(self.dataset, "patch_size", 0)) != int(backbone.patch_size):
#             raise RuntimeError("dataset/backbone patch sizes disagree")

#         return {
#             "schedule": dict(sorted(schedule.items())),
#             "base_classes": list(base),
#             "protocol_report": True,
#         }

#     def assert_model_contract(self) -> Dict[str, Any]:
#         required = (
#             "encode",
#             "forward",
#             "initialize_candidate",
#             "classify_coordinates",
#             "commit_candidate",
#             "pair_values",
#             "class_boundary_response",
#             "true_pair_margins",
#             "validate_geometry",
#             "validate_model_state",
#         )
#         missing = [
#             name for name in required
#             if not callable(getattr(self.model, name, None))
#         ]
#         if missing:
#             raise RuntimeError(f"model lacks required APIs: {missing}")

#         backbone = getattr(self.model, "backbone", None)
#         bank = getattr(self.model, "geometry_bank", None)
#         classifier = getattr(self.model, "classifier", None)
#         if backbone is None or bank is None or classifier is None:
#             raise RuntimeError("model must own backbone, geometry_bank, classifier")
#         if not isinstance(bank, BoundaryGeometryBank):
#             raise RuntimeError("model geometry_bank must be BoundaryGeometryBank")

#         for name in (
#             "patch_bands",
#             "spectral_bands",
#             "patch_size",
#             "representation_dim",
#         ):
#             if int(getattr(backbone, name, 0)) <= 0:
#                 raise RuntimeError(f"backbone.{name} must be positive")

#         representation_dim = int(backbone.representation_dim)
#         if hasattr(backbone, "coordinate_projection"):
#             raise RuntimeError(
#                 "one-space backbone must not expose coordinate_projection"
#             )
#         context_refiner = getattr(backbone, "context_refiner", None)
#         if context_refiner is None:
#             raise RuntimeError("backbone must expose context_refiner")
#         if int(getattr(context_refiner, "spectral_dim", 0)) != representation_dim:
#             raise RuntimeError(
#                 "context correction must use the canonical representation"
#             )
#         if int(bank.representation_dim) != representation_dim:
#             raise RuntimeError("bank/backbone representation dimensions disagree")
#         if any(True for _ in classifier.parameters()):
#             raise RuntimeError("GeometryClassifier must remain parameter-free")
#         if torch.device(self.model.device) != self.device or bank.device != self.device:
#             raise RuntimeError("model/geometry are not on trainer device")
#         if getattr(self.model, "dtype", bank.dtype) != bank.dtype:
#             raise RuntimeError("model/geometry floating dtypes disagree")

#         self.model.validate_model_state()
#         return {
#             "architecture": "one_space_hsi_pairwise_decision_geometry",
#             "patch_bands": int(backbone.patch_bands),
#             "spectral_bands": int(backbone.spectral_bands),
#             "patch_size": int(backbone.patch_size),
#             "representation_dim": representation_dim,
#             "context_input_channels": int(context_refiner.input_channels),
#             "context_spectral_dim": int(context_refiner.spectral_dim),
#             "classifier_parameter_count": 0,
#             "geometry_interface": (
#                 "pair_values + class_boundary_response + true_pair_margins"
#             ),
#             "spectral_normalization_fitted": bool(
#                 self.model.spectral_normalization_fitted
#             ),
#         }

#     def unpack_batch(
#         self,
#         batch: Mapping[str, Any],
#     ) -> tuple[Tensor, Tensor, Tensor]:
#         if not isinstance(batch, Mapping):
#             raise TypeError("HSI batches must be mappings")
#         required = {"image", "raw_center_spectrum", "label"}
#         missing = required - set(batch)
#         if missing:
#             raise KeyError(f"batch missing {sorted(missing)}")

#         model_dtype = getattr(self.model.geometry_bank, "dtype", torch.float32)
#         patch = torch.as_tensor(
#             batch["image"], device=self.device, dtype=model_dtype
#         )
#         spectrum = torch.as_tensor(
#             batch["raw_center_spectrum"],
#             device=self.device,
#             dtype=model_dtype,
#         )
#         labels = torch.as_tensor(batch["label"], device=self.device).flatten()

#         backbone = self.model.backbone
#         expected_patch = (
#             patch.size(0) if patch.ndim else 0,
#             int(backbone.patch_bands),
#             int(backbone.patch_size),
#             int(backbone.patch_size),
#         )
#         expected_spectrum = (
#             patch.size(0) if patch.ndim else 0,
#             int(backbone.spectral_bands),
#         )
#         if (
#             patch.ndim != 4
#             or patch.size(0) == 0
#             or tuple(patch.shape) != expected_patch
#         ):
#             raise RuntimeError("processed HSI patch shape is invalid")
#         if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
#             raise RuntimeError("raw center spectrum shape is invalid")
#         if not bool(torch.isfinite(patch).all()) or not bool(
#             torch.isfinite(spectrum).all()
#         ):
#             raise RuntimeError("HSI batch contains NaN/Inf")
#         if (
#             labels.numel() != patch.size(0)
#             or labels.dtype == torch.bool
#             or labels.is_complex()
#         ):
#             raise RuntimeError("labels are invalid or batch-misaligned")
#         if torch.is_floating_point(labels):
#             if not bool(torch.isfinite(labels).all()) or not bool(
#                 labels.eq(labels.round()).all()
#             ):
#                 raise RuntimeError("labels must be finite integers")
#         labels = labels.to(dtype=torch.long)
#         if bool((labels < 0).any()):
#             raise RuntimeError("labels must be non-negative")
#         return patch, spectrum, labels

#     @torch.no_grad()
#     def collect_labels(self, loader: Any) -> Tensor:
#         labels: list[Tensor] = []
#         for batch in loader:
#             _, _, values = self.unpack_batch(batch)
#             labels.append(values.detach().cpu())
#         if not labels:
#             raise RuntimeError("cannot collect labels from an empty loader")
#         return torch.cat(labels, dim=0)

#     @torch.no_grad()
#     def collect_encoded(self, loader: Any) -> Dict[str, Tensor]:
#         coordinates: list[Tensor] = []
#         labels: list[Tensor] = []
#         states = {
#             module: bool(module.training)
#             for module in self.model.modules()
#         }
#         try:
#             self.model.eval()
#             for batch in loader:
#                 patch, spectrum, values = self.unpack_batch(batch)
#                 output = self.model.encode(
#                     patch,
#                     center_spectrum=spectrum,
#                     return_aux=False,
#                 )
#                 coordinates.append(output.coordinates.detach().cpu())
#                 labels.append(values.detach().cpu())
#         finally:
#             for module, state in states.items():
#                 module.training = state

#         if not coordinates:
#             raise RuntimeError("cannot encode an empty loader")
#         z = torch.cat(coordinates, dim=0)
#         y = torch.cat(labels, dim=0)
#         if z.ndim != 2 or z.size(0) != y.numel():
#             raise RuntimeError("collected representation is row-misaligned")
#         if z.size(1) != int(self.model.representation_dim):
#             raise RuntimeError("collected representation dimension is invalid")
#         if not bool(torch.isfinite(z).all()):
#             raise RuntimeError("collected representation contains NaN/Inf")
#         return {"coordinates": z, "labels": y}

#     @contextmanager
#     def _temporary_eval_state(
#         self,
#         candidate: Optional[BoundaryCandidate],
#     ):
#         model_states = {
#             module: bool(module.training)
#             for module in self.model.modules()
#         }
#         candidate_state = None if candidate is None else bool(candidate.training)
#         try:
#             self.model.eval()
#             if candidate is not None:
#                 candidate.eval()
#             yield
#         finally:
#             for module, state in model_states.items():
#                 module.training = state
#             if candidate is not None and candidate_state is not None:
#                 candidate.training = candidate_state

#     @torch.no_grad()
#     def evaluate_loader(
#         self,
#         loader: Any,
#         *,
#         class_ids: Sequence[int],
#         target_class_ids: Optional[Sequence[int]] = None,
#         candidate: Optional[BoundaryCandidate] = None,
#         geometry_bank: Optional[BoundaryGeometryBank] = None,
#     ) -> Dict[str, Any]:
#         """Evaluate the deployed rule and decision-geometry diagnostics.

#         ``target_class_ids`` controls which labels are expected/reported, while
#         every sample is classified against the complete ``class_ids`` set.
#         Pair-violation statistics use the true class's oriented incident
#         boundaries and therefore directly measure decision-relevant overlap.
#         """
#         requested = self._validated_class_ids(
#             class_ids, name="evaluation_class_ids"
#         )
#         targets_requested = (
#             requested
#             if target_class_ids is None
#             else self._validated_class_ids(
#                 target_class_ids, name="target_class_ids"
#             )
#         )
#         if not set(targets_requested).issubset(requested):
#             raise ValueError(
#                 "target_class_ids must be a subset of evaluation class_ids"
#             )
#         if candidate is not None and geometry_bank is not None:
#             raise ValueError("candidate and geometry_bank are mutually exclusive")

#         active_bank = self.model.geometry_bank if geometry_bank is None else geometry_bank
#         active_candidate = candidate if geometry_bank is None else None
#         active_bank.validate_bank_state()
#         if active_bank.device != self.model.geometry_bank.device:
#             raise ValueError("evaluation geometry bank is on the wrong device")
#         if active_bank.dtype != self.model.geometry_bank.dtype:
#             raise ValueError("evaluation geometry bank uses the wrong dtype")
#         if int(active_bank.representation_dim) != int(self.model.representation_dim):
#             raise ValueError(
#                 "evaluation geometry has the wrong representation dimension"
#             )

#         total = 0
#         correct = 0
#         ce_sum = 0.0
#         true_energy_sum = 0.0
#         true_cell_violation_sum = 0.0
#         rival_energy_sum = 0.0
#         margin_sum = 0.0
#         true_inside_count = 0
#         rival_inside_count = 0
#         no_cell_count = 0
#         pair_violation_count = 0
#         pair_relation_count = 0
#         min_true_pair_margin_sum = 0.0
#         has_rivals = len(requested) > 1

#         class_total = {class_id: 0 for class_id in targets_requested}
#         class_correct = {class_id: 0 for class_id in targets_requested}
#         class_ce = {class_id: 0.0 for class_id in targets_requested}
#         class_cell_violation = {class_id: 0.0 for class_id in targets_requested}
#         class_inside = {class_id: 0 for class_id in targets_requested}
#         class_rival_inside = {class_id: 0 for class_id in targets_requested}
#         class_no_cell = {class_id: 0 for class_id in targets_requested}
#         class_true_energy = {class_id: 0.0 for class_id in targets_requested}
#         class_rival_energy = {class_id: 0.0 for class_id in targets_requested}
#         class_margin = {class_id: 0.0 for class_id in targets_requested}
#         class_pair_violation = {class_id: 0 for class_id in targets_requested}
#         class_pair_relations = {class_id: 0 for class_id in targets_requested}
#         class_min_pair_margin = {class_id: 0.0 for class_id in targets_requested}

#         with self._temporary_eval_state(candidate):
#             for batch in loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 observed = set(
#                     int(value)
#                     for value in labels.unique().detach().cpu().tolist()
#                 )
#                 outside = sorted(observed - set(targets_requested))
#                 if outside:
#                     raise RuntimeError(
#                         "evaluation loader contains labels outside target classes: "
#                         f"{outside}"
#                     )

#                 representation = self.model.encode(
#                     patch,
#                     center_spectrum=spectrum,
#                     return_aux=False,
#                 )
#                 if geometry_bank is None:
#                     output = self.model.classify_coordinates(
#                         representation.coordinates,
#                         class_ids=requested,
#                         candidate=candidate,
#                     )
#                 else:
#                     output = self.model.classifier(
#                         representation.coordinates,
#                         geometry_bank=active_bank,
#                         class_ids=requested,
#                         candidate=None,
#                     )

#                 actual_ids = [
#                     int(value)
#                     for value in output.class_ids.detach().cpu().tolist()
#                 ]
#                 if actual_ids != requested:
#                     raise RuntimeError(
#                         "classifier columns do not match requested classes"
#                     )

#                 targets = self.model.classifier.targets_local(
#                     labels, output.class_ids
#                 )
#                 rows = torch.arange(
#                     labels.numel(), device=output.energy.device
#                 )
#                 true_energy = output.energy[rows, targets]
#                 inside = true_energy <= 0
#                 no_cell = output.energy.amin(dim=1) > 0
#                 per_ce = F.cross_entropy(
#                     output.logits, targets, reduction="none"
#                 )
#                 per_cell_violation = F.relu(true_energy)

#                 ce_sum += float(per_ce.sum().item())
#                 true_cell_violation_sum += float(per_cell_violation.sum().item())
#                 true_energy_sum += float(true_energy.sum().item())
#                 true_inside_count += int(inside.sum().item())
#                 no_cell_count += int(no_cell.sum().item())

#                 pair_margins = None
#                 per_sample_min_pair_margin = None
#                 if has_rivals:
#                     response = active_bank.class_boundary_response(
#                         representation.coordinates,
#                         labels,
#                         class_ids=requested,
#                         candidate=active_candidate,
#                     )
#                     pair_margins = response.margins
#                     if pair_margins.shape != (
#                         labels.numel(), len(requested) - 1
#                     ):
#                         raise RuntimeError(
#                             "evaluation pair-response shape is invalid"
#                         )
#                     violations = pair_margins < 0
#                     pair_violation_count += int(violations.sum().item())
#                     pair_relation_count += int(violations.numel())
#                     per_sample_min_pair_margin = pair_margins.amin(dim=1)
#                     min_true_pair_margin_sum += float(
#                         per_sample_min_pair_margin.sum().item()
#                     )

#                 rival_energy = None
#                 rival_inside = None
#                 margin = None
#                 if has_rivals:
#                     mask = F.one_hot(
#                         targets, num_classes=len(requested)
#                     ).to(torch.bool)
#                     rival_energy = output.energy.masked_fill(
#                         mask, torch.inf
#                     ).amin(dim=1)
#                     rival_inside = rival_energy < 0
#                     if bool((inside & rival_inside).any()):
#                         raise RuntimeError(
#                             "pairwise geometry invariant violated: a sample "
#                             "lies in two strict class interiors"
#                         )
#                     margin = rival_energy - true_energy
#                     rival_inside_count += int(rival_inside.sum().item())
#                     rival_energy_sum += float(rival_energy.sum().item())
#                     margin_sum += float(margin.sum().item())

#                 prediction = output.prediction
#                 count = int(labels.numel())
#                 total += count
#                 correct += int(prediction.eq(labels).sum().item())

#                 for class_id in targets_requested:
#                     class_mask = labels.eq(class_id)
#                     count_class = int(class_mask.sum().item())
#                     if count_class == 0:
#                         continue
#                     class_total[class_id] += count_class
#                     class_correct[class_id] += int(
#                         prediction[class_mask]
#                         .eq(labels[class_mask])
#                         .sum()
#                         .item()
#                     )
#                     class_ce[class_id] += float(
#                         per_ce[class_mask].sum().item()
#                     )
#                     class_cell_violation[class_id] += float(
#                         per_cell_violation[class_mask].sum().item()
#                     )
#                     class_inside[class_id] += int(
#                         inside[class_mask].sum().item()
#                     )
#                     class_no_cell[class_id] += int(
#                         no_cell[class_mask].sum().item()
#                     )
#                     class_true_energy[class_id] += float(
#                         true_energy[class_mask].sum().item()
#                     )

#                     if has_rivals:
#                         assert (
#                             rival_inside is not None
#                             and rival_energy is not None
#                             and margin is not None
#                             and pair_margins is not None
#                             and per_sample_min_pair_margin is not None
#                         )
#                         class_rival_inside[class_id] += int(
#                             rival_inside[class_mask].sum().item()
#                         )
#                         class_rival_energy[class_id] += float(
#                             rival_energy[class_mask].sum().item()
#                         )
#                         class_margin[class_id] += float(
#                             margin[class_mask].sum().item()
#                         )
#                         class_pair_violation[class_id] += int(
#                             pair_margins[class_mask].lt(0).sum().item()
#                         )
#                         class_pair_relations[class_id] += int(
#                             pair_margins[class_mask].numel()
#                         )
#                         class_min_pair_margin[class_id] += float(
#                             per_sample_min_pair_margin[class_mask].sum().item()
#                         )

#         if total == 0:
#             raise RuntimeError("evaluation loader is empty")
#         missing = [
#             class_id
#             for class_id in targets_requested
#             if class_total[class_id] == 0
#         ]
#         if missing:
#             raise RuntimeError(
#                 f"evaluation split is missing target classes: {missing}"
#             )

#         per_acc = {
#             class_id: class_correct[class_id] / class_total[class_id]
#             for class_id in targets_requested
#         }
#         per_ce_mean = {
#             class_id: class_ce[class_id] / class_total[class_id]
#             for class_id in targets_requested
#         }
#         per_cell_violation_mean = {
#             class_id: class_cell_violation[class_id] / class_total[class_id]
#             for class_id in targets_requested
#         }
#         per_cov = {
#             class_id: class_inside[class_id] / class_total[class_id]
#             for class_id in targets_requested
#         }
#         per_inv = {
#             class_id: (
#                 class_rival_inside[class_id] / class_total[class_id]
#                 if has_rivals else 0.0
#             )
#             for class_id in targets_requested
#         }
#         per_no_cell = {
#             class_id: class_no_cell[class_id] / class_total[class_id]
#             for class_id in targets_requested
#         }
#         per_true = {
#             class_id: class_true_energy[class_id] / class_total[class_id]
#             for class_id in targets_requested
#         }
#         per_rival = {
#             class_id: (
#                 class_rival_energy[class_id] / class_total[class_id]
#                 if has_rivals else None
#             )
#             for class_id in targets_requested
#         }
#         per_margin = {
#             class_id: (
#                 class_margin[class_id] / class_total[class_id]
#                 if has_rivals else None
#             )
#             for class_id in targets_requested
#         }
#         per_pair_violation = {
#             class_id: (
#                 class_pair_violation[class_id] / class_pair_relations[class_id]
#                 if has_rivals and class_pair_relations[class_id] > 0 else 0.0
#             )
#             for class_id in targets_requested
#         }
#         per_min_pair_margin = {
#             class_id: (
#                 class_min_pair_margin[class_id] / class_total[class_id]
#                 if has_rivals else None
#             )
#             for class_id in targets_requested
#         }

#         def macro(values: Mapping[int, float]) -> float:
#             return (
#                 sum(float(values[class_id]) for class_id in targets_requested)
#                 / len(targets_requested)
#             )

#         macro_rival = (
#             macro({
#                 class_id: float(per_rival[class_id])
#                 for class_id in targets_requested
#             })
#             if has_rivals else None
#         )
#         macro_margin = (
#             macro({
#                 class_id: float(per_margin[class_id])
#                 for class_id in targets_requested
#             })
#             if has_rivals else None
#         )
#         macro_min_pair_margin = (
#             macro({
#                 class_id: float(per_min_pair_margin[class_id])
#                 for class_id in targets_requested
#             })
#             if has_rivals else None
#         )

#         return {
#             "energy_convention": {
#                 "class_cell": (
#                     "E_c(z) <= 0 iff all pairwise boundaries support class c"
#                 ),
#                 "class_score": (
#                     "E_c(z) = -minimum oriented pairwise signed distance"
#                 ),
#                 "decision": "argmin_c E_c(z)",
#                 "decision_margin": "nearest_rival_energy - true_energy",
#                 "pair_violation": "s_yj(z) < 0",
#                 "no_cell": "min_c E_c(z) > 0",
#             },
#             "classification": ce_sum / total,
#             "macro_classification": macro(per_ce_mean),
#             # This remains a diagnostic alias only; it is no longer a training loss.
#             "true_cell_violation": true_cell_violation_sum / total,
#             "macro_true_cell_violation": macro(per_cell_violation_mean),
#             "per_class_true_cell_violation": per_cell_violation_mean,
#             "cell_fit": true_cell_violation_sum / total,
#             "macro_cell_fit": macro(per_cell_violation_mean),
#             "per_class_cell_fit": per_cell_violation_mean,
#             "overall_accuracy": correct / total,
#             "accuracy": correct / total,
#             "balanced_accuracy": macro(per_acc),
#             "minimum_class_accuracy": min(per_acc.values()),
#             "evaluated_class_ids": list(requested),
#             "target_class_ids": list(targets_requested),
#             "class_counts": {
#                 class_id: class_total[class_id]
#                 for class_id in targets_requested
#             },
#             "per_class_accuracy": per_acc,
#             "true_cell_coverage": true_inside_count / total,
#             "macro_true_cell_coverage": macro(per_cov),
#             "per_class_true_cell_coverage": per_cov,
#             "rival_cell_invasion_rate": (
#                 rival_inside_count / total if has_rivals else 0.0
#             ),
#             "macro_rival_cell_invasion_rate": macro(per_inv),
#             "per_class_rival_cell_invasion_rate": per_inv,
#             "no_cell_rate": no_cell_count / total,
#             "macro_no_cell_rate": macro(per_no_cell),
#             "per_class_no_cell_rate": per_no_cell,
#             "true_pair_violation_rate": (
#                 pair_violation_count / pair_relation_count
#                 if pair_relation_count else 0.0
#             ),
#             "macro_true_pair_violation_rate": macro(per_pair_violation),
#             "per_class_true_pair_violation_rate": per_pair_violation,
#             "mean_minimum_true_pair_margin": (
#                 min_true_pair_margin_sum / total if has_rivals else None
#             ),
#             "macro_mean_minimum_true_pair_margin": macro_min_pair_margin,
#             "per_class_mean_minimum_true_pair_margin": per_min_pair_margin,
#             "mean_true_energy": true_energy_sum / total,
#             "macro_mean_true_energy": macro(per_true),
#             "per_class_mean_true_energy": per_true,
#             "mean_nearest_rival_energy": (
#                 rival_energy_sum / total if has_rivals else None
#             ),
#             "macro_mean_nearest_rival_energy": macro_rival,
#             "per_class_mean_nearest_rival_energy": per_rival,
#             "mean_decision_margin": (
#                 margin_sum / total if has_rivals else None
#             ),
#             "macro_mean_decision_margin": macro_margin,
#             "per_class_mean_decision_margin": per_margin,
#             "strict_cell_conflict_rate": 0.0,
#         }

#     @staticmethod
#     def summarize_class_group(
#         metrics: Mapping[str, Any],
#         class_ids: Sequence[int],
#     ) -> Dict[str, Any]:
#         ids = [int(value) for value in class_ids]
#         if not ids:
#             raise ValueError("class group cannot be empty")

#         def mapping(name: str) -> Mapping[Any, Any]:
#             value = metrics.get(name)
#             if not isinstance(value, Mapping):
#                 raise ValueError(f"metrics lacks {name}")
#             return value

#         def value(row: Mapping[Any, Any], class_id: int) -> float:
#             raw = row[class_id] if class_id in row else row.get(str(class_id))
#             if raw is None:
#                 raise ValueError(f"metrics lacks class {class_id}")
#             return float(raw)

#         per_acc = mapping("per_class_accuracy")
#         per_cov = mapping("per_class_true_cell_coverage")
#         per_inv = mapping("per_class_rival_cell_invasion_rate")
#         per_no_cell = mapping("per_class_no_cell_rate")
#         per_pair = mapping("per_class_true_pair_violation_rate")
#         per_cell = mapping("per_class_true_cell_violation")
#         per_margin = mapping("per_class_mean_decision_margin")

#         return {
#             "class_ids": ids,
#             "balanced_accuracy": sum(value(per_acc, class_id) for class_id in ids) / len(ids),
#             "minimum_class_accuracy": min(value(per_acc, class_id) for class_id in ids),
#             "macro_true_cell_coverage": sum(value(per_cov, class_id) for class_id in ids) / len(ids),
#             "macro_rival_cell_invasion_rate": sum(value(per_inv, class_id) for class_id in ids) / len(ids),
#             "macro_no_cell_rate": sum(value(per_no_cell, class_id) for class_id in ids) / len(ids),
#             "macro_true_pair_violation_rate": sum(value(per_pair, class_id) for class_id in ids) / len(ids),
#             "macro_true_cell_violation": sum(value(per_cell, class_id) for class_id in ids) / len(ids),
#             # Compatibility alias for older reporting code only.
#             "macro_cell_fit": sum(value(per_cell, class_id) for class_id in ids) / len(ids),
#             "macro_mean_decision_margin": sum(value(per_margin, class_id) for class_id in ids) / len(ids),
#         }

#     def geometry_state_summary(self) -> Dict[str, Any]:
#         bank = self.model.geometry_bank
#         valid = bool(bank.validate_bank_state())
#         class_ids = [
#             int(value) for value in bank.class_ids.detach().cpu().tolist()
#         ]
#         norms = (
#             torch.linalg.vector_norm(bank.normals, dim=1)
#             if bank.pair_count
#             else torch.empty(0)
#         )
#         return {
#             "structurally_valid": valid,
#             "class_ids": class_ids,
#             "class_count": len(bank),
#             "pair_count": int(bank.pair_count),
#             "representation_dim": int(bank.representation_dim),
#             "normal_norm_minimum": (
#                 None if norms.numel() == 0 else float(norms.amin().item())
#             ),
#             "normal_norm_maximum": (
#                 None if norms.numel() == 0 else float(norms.amax().item())
#             ),
#             "offset_minimum": (
#                 None if bank.offsets.numel() == 0
#                 else float(bank.offsets.amin().item())
#             ),
#             "offset_maximum": (
#                 None if bank.offsets.numel() == 0
#                 else float(bank.offsets.amax().item())
#             ),
#         }

#     @staticmethod
#     def json_safe(value: Any) -> Any:
#         if torch.is_tensor(value):
#             tensor = value.detach().cpu()
#             return tensor.item() if tensor.numel() == 1 else tensor.tolist()
#         try:
#             import numpy as np
#             if isinstance(value, np.ndarray):
#                 return value.tolist()
#             if isinstance(value, (np.integer, np.floating, np.bool_)):
#                 return value.item()
#         except ImportError:
#             pass
#         if isinstance(value, Mapping):
#             return {
#                 str(key): TrainerHelper.json_safe(item)
#                 for key, item in value.items()
#             }
#         if isinstance(value, (list, tuple, set)):
#             return [TrainerHelper.json_safe(item) for item in value]
#         if isinstance(value, Path):
#             return str(value)
#         if isinstance(value, float) and not math.isfinite(value):
#             return str(value)
#         if isinstance(value, (str, int, float, bool)) or value is None:
#             return value
#         return str(value)

#     def save_json(self, path: str, value: Mapping[str, Any]) -> str:
#         destination = os.path.abspath(path)
#         os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#         temporary = destination + ".tmp"
#         try:
#             with open(temporary, "w", encoding="utf-8") as stream:
#                 json.dump(
#                     self.json_safe(value),
#                     stream,
#                     indent=2,
#                     sort_keys=True,
#                 )
#                 stream.flush()
#                 os.fsync(stream.fileno())
#             os.replace(temporary, destination)
#         except Exception:
#             if os.path.exists(temporary):
#                 os.remove(temporary)
#             raise
#         return destination


# __all__ = ["TrainerHelper"]
