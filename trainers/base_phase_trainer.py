from __future__ import annotations

"""Base-phase training for one-space HSI pairwise decision geometry.

Phase 0 jointly learns:

1. the spectral-primary HSI representation z;
2. one shared affine boundary for every base-class pair.

The base objective is

    L_base = lambda_cls * L_CE + lambda_sep * L_sep,

where ``L_sep`` is the pair-balanced distribution-separation objective defined
in ``losses.loss``.  The previous decision-cell fit ``relu(E_y)`` is not used.

The candidate contains every base-base pair, so base separation is applied to
the complete trainable base geometry whenever both sides of a pair occur in a
minibatch.  Validation remains diagnostic only; the final configured epoch is
committed exactly, preserving the fixed-schedule protocol.
"""

import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Sequence

import torch
from tqdm import tqdm

from losses.loss import geometry_training_objective
from models.geometry_bank import BoundaryCandidate

Tensor = torch.Tensor


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _as_int(value: object, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be an integer")
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    raise ValueError(f"{name} must be an integer")


class BasePhaseTrainer:
    @staticmethod
    def _base_class_ids(values: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(_as_int(v, "base_class_id") for v in values)
        if not ids or len(ids) != len(set(ids)) or any(v < 0 for v in ids):
            raise ValueError(
                "base class IDs must be unique non-negative integers"
            )
        if len(ids) < 2:
            raise ValueError(
                "pairwise decision geometry requires at least two base classes"
            )
        return ids

    @staticmethod
    def _all_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
        return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)

    @staticmethod
    def _active_candidate_pairs(
        labels: Tensor,
        candidate: BoundaryCandidate,
    ) -> set[tuple[int, int]]:
        """Candidate pairs for which both class sides occur in this batch."""
        present = set(
            int(v)
            for v in torch.as_tensor(labels)
            .detach()
            .flatten()
            .cpu()
            .unique()
            .tolist()
        )
        return {
            (int(row[0]), int(row[1]))
            for row in candidate.pair_ids.detach().cpu().tolist()
            if int(row[0]) in present and int(row[1]) in present
        }

    @torch.no_grad()
    def _ensure_spectral_normalization(self, loader: Any) -> None:
        if bool(self.model.spectral_normalization_fitted):
            return

        spectra: list[Tensor] = []
        for batch in loader:
            _, center_spectrum, _ = self.unpack_batch(batch)
            spectra.append(center_spectrum.detach())

        if not spectra:
            raise RuntimeError("spectral-normalization loader is empty")

        self.model.fit_spectral_normalization(
            torch.cat(spectra, dim=0),
            overwrite=False,
        )
        if not bool(self.model.spectral_normalization_fitted):
            raise RuntimeError("spectral normalization was not fitted")

    @torch.no_grad()
    def _collect_coordinates(self, loader: Any) -> Dict[str, Tensor]:
        if not bool(self.model.spectral_normalization_fitted):
            raise RuntimeError(
                "fit spectral normalization before coordinate collection"
            )

        states = {
            module: bool(module.training)
            for module in self.model.modules()
        }
        try:
            self.model.eval()
            coordinates: list[Tensor] = []
            labels: list[Tensor] = []

            for batch in loader:
                patch, spectrum, target = self.unpack_batch(batch)
                output = self.model.encode(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                coordinates.append(output.coordinates.detach())
                labels.append(target.detach())
        finally:
            for module, state in states.items():
                module.training = state

        if not labels:
            raise RuntimeError("coordinate collection loader is empty")

        z = torch.cat(coordinates, dim=0)
        y = torch.cat(labels, dim=0).flatten()

        if (
            z.ndim != 2
            or z.size(0) != y.numel()
            or z.size(1) != self.model.representation_dim
        ):
            raise RuntimeError("collected representation is invalid")
        if (
            z.device != self.model.geometry_bank.device
            or z.dtype != self.model.geometry_bank.dtype
        ):
            raise RuntimeError(
                "collected representation and geometry disagree in device/dtype"
            )
        if not bool(torch.isfinite(z).all()):
            raise RuntimeError("coordinate collection produced NaN/Inf")

        return {"coordinates": z, "labels": y}

    def _class_uniform_risk_weights(
        self,
        labels: Tensor,
        class_ids: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        """Weights only the global CE term toward uniform class risk."""
        counts = torch.stack(
            [torch.as_tensor(labels).eq(class_id).sum() for class_id in class_ids]
        ).to(device=self.device, dtype=torch.float32)

        if bool((counts <= 0).any()):
            raise RuntimeError("every base class must occur in training")

        total = counts.sum()
        weights = total / (len(class_ids) * counts)

        # For empirical class-uniform CE, mean_i w[y_i] must equal one.
        empirical = (counts * weights).sum() / total
        if not bool(
            torch.allclose(
                empirical,
                torch.ones_like(empirical),
                rtol=1e-6,
                atol=1e-7,
            )
        ):
            raise RuntimeError(
                "class-uniform risk weights are normalized incorrectly"
            )
        return weights, counts

    @torch.no_grad()
    def _evaluate_training_objective(
        self,
        loader: Any,
        *,
        class_ids: tuple[int, ...],
        candidate: BoundaryCandidate,
        class_risk_weights: Tensor,
        classification_weight: float,
        separation_weight: float,
    ) -> Dict[str, Any]:
        """Evaluate CE + separation on the complete split representation.

        Encoding is still performed in evaluation minibatches, but the collected
        coordinates are evaluated as one logical decision batch.  This matters
        because pairwise distribution separation is a class-pair objective:
        deterministic evaluation minibatches may contain only subsets of
        classes even though every class is present in the split.  Pair coverage
        reported here therefore reflects the complete split, not incidental
        minibatch co-occurrence.
        """
        collected = self._collect_coordinates(loader)
        coordinates = collected["coordinates"]
        labels = collected["labels"]

        output = self.model.classify_coordinates(
            coordinates,
            class_ids=class_ids,
            candidate=candidate,
        )
        objective = geometry_training_objective(
            output=output,
            coordinates=coordinates,
            labels_global=labels,
            geometry_bank=self.model.geometry_bank,
            candidate=candidate,
            class_risk_weights=class_risk_weights,
            classification_weight=classification_weight,
            separation_weight=separation_weight,
        )

        active_pairs = self._active_candidate_pairs(labels, candidate)
        if objective.active_pair_count != len(active_pairs):
            raise RuntimeError(
                "full-split objective active-pair count disagrees with labels"
            )

        candidate_pair_count = int(candidate.pair_ids.size(0))
        pair_coverage = (
            len(active_pairs) / candidate_pair_count
            if candidate_pair_count
            else 1.0
        )

        total = float(objective.total.item())
        if not math.isfinite(total):
            raise RuntimeError("evaluated geometry objective is not finite")

        return {
            "total": total,
            "classification": float(objective.classification.item()),
            "separation": float(objective.separation.item()),
            "accuracy": float(objective.accuracy.item()),
            "active_pair_incidence_count": int(objective.active_pair_count),
            "mean_active_pairs_per_batch": float(objective.active_pair_count),
            "covered_pair_count": int(len(active_pairs)),
            "candidate_pair_count": candidate_pair_count,
            "pair_coverage": float(pair_coverage),
        }

    def train_base_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        if int(phase) != 0:
            raise ValueError("BasePhaseTrainer handles phase 0 only")
        if int(epochs) <= 0 or int(batch_size) <= 0:
            raise ValueError("epochs and batch_size must be positive")

        learning_rate = float(lr)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("lr must be finite and positive")

        self.model.validate_model_state()
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError(
                "base training requires an empty committed geometry bank"
            )

        self.dataset.start_phase(0)
        class_ids = self._base_class_ids(
            self.dataset.get_new_classes(0)
        )

        eval_batch = int(self.cfg("eval_batch_size", 256))
        if eval_batch <= 0:
            raise ValueError("eval_batch_size must be positive")

        init_loader = self.dataset.get_phase_dataloader(
            0,
            split="train",
            batch_size=eval_batch,
            shuffle=False,
        )
        train_loader = self.dataset.get_phase_dataloader(
            0,
            split="train",
            batch_size=int(batch_size),
            shuffle=True,
        )
        train_eval_loader = self.dataset.get_phase_dataloader(
            0,
            split="train",
            batch_size=eval_batch,
            shuffle=False,
        )
        val_loader = self.dataset.get_phase_dataloader(
            0,
            split="val",
            batch_size=eval_batch,
            shuffle=False,
        )

        # Spectral normalization is fitted once from real base TRAIN center
        # spectra before any geometry is initialized.
        self._ensure_spectral_normalization(init_loader)

        # Candidate initialization uses real base TRAIN coordinates only.
        initial = self._collect_coordinates(init_loader)
        candidate = self.model.initialize_candidate(
            initial["coordinates"],
            initial["labels"],
            class_ids,
        )
        if not isinstance(candidate, BoundaryCandidate):
            raise RuntimeError(
                "model returned the wrong geometry candidate type"
            )
        if (
            candidate.new_class_ids != class_ids
            or candidate.validate_state() is not True
        ):
            raise RuntimeError("base boundary candidate is invalid")

        expected_pair_count = len(class_ids) * (len(class_ids) - 1) // 2
        if int(candidate.pair_ids.size(0)) != expected_pair_count:
            raise RuntimeError(
                "base candidate does not contain every base-base class pair"
            )

        # These weights affect only global CE. Pairwise separation is already
        # balanced over both sides of each class pair.
        class_weights, class_counts = self._class_uniform_risk_weights(
            initial["labels"],
            class_ids,
        )

        backbone_parameters = [
            parameter
            for parameter in self.model.backbone.parameters()
            if parameter.requires_grad
        ]
        geometry_parameters = [
            parameter
            for parameter in candidate.parameters()
            if parameter.requires_grad
        ]
        if not backbone_parameters or not geometry_parameters:
            raise RuntimeError(
                "backbone and boundary candidate must both be trainable"
            )
        if {id(p) for p in backbone_parameters}.intersection(
            id(p) for p in geometry_parameters
        ):
            raise RuntimeError(
                "backbone and geometry optimizer groups overlap"
            )

        weight_decay = _finite_nonnegative(
            "weight_decay",
            self.cfg("weight_decay", 1e-4),
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
                {
                    "params": geometry_parameters,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        )

        objective_weights = {
            "classification_weight": _finite_nonnegative(
                "base_classification_weight",
                self.cfg("base_classification_weight", 1.0),
            ),
            "separation_weight": _finite_nonnegative(
                "base_separation_weight",
                self.cfg("base_separation_weight", 1.0),
            ),
        }
        if not any(weight > 0.0 for weight in objective_weights.values()):
            raise ValueError(
                "at least one base objective weight must be positive"
            )

        gradient_clip = _finite_nonnegative(
            "gradient_clip",
            self.cfg("gradient_clip", 0.0),
        )
        optimized_parameters = (
            backbone_parameters + geometry_parameters
        )

        history: list[Dict[str, Any]] = []

        progress = tqdm(
            range(int(epochs)),
            desc="Base pairwise distribution geometry",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            self.model.train()
            candidate.train()

            classification_sum = 0.0
            accuracy_sum = 0.0
            sample_count = 0

            separation_sum = 0.0
            active_pair_incidence_count = 0
            active_pair_count_sum = 0
            batch_count = 0
            active_pair_union: set[tuple[int, int]] = set()

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)

                optimizer.zero_grad(set_to_none=True)

                result = self.model(
                    patch,
                    center_spectrum=spectrum,
                    class_ids=class_ids,
                    candidate=candidate,
                    return_aux=False,
                )

                objective = geometry_training_objective(
                    output=result.classification,
                    coordinates=result.representation.coordinates,
                    labels_global=labels,
                    geometry_bank=self.model.geometry_bank,
                    candidate=candidate,
                    class_risk_weights=class_weights,
                    **objective_weights,
                )

                batch_pairs = self._active_candidate_pairs(
                    labels,
                    candidate,
                )
                if objective.active_pair_count != len(batch_pairs):
                    raise RuntimeError(
                        "loss active-pair count disagrees with candidate/batch labels"
                    )

                objective.total.backward()

                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in optimized_parameters
                ):
                    raise RuntimeError(
                        "base optimization produced NaN/Inf gradients"
                    )

                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        optimized_parameters,
                        gradient_clip,
                        error_if_nonfinite=True,
                    )

                optimizer.step()

                if not self._all_finite(optimized_parameters):
                    raise RuntimeError(
                        "base optimization produced NaN/Inf parameters"
                    )
                candidate.validate_state()

                count = int(labels.numel())
                if count <= 0:
                    raise RuntimeError(
                        "base training produced an empty batch"
                    )

                classification_sum += (
                    float(objective.classification.detach().item())
                    * count
                )
                accuracy_sum += (
                    float(objective.accuracy.detach().item())
                    * count
                )
                sample_count += count

                if objective.active_pair_count:
                    separation_sum += (
                        float(objective.separation.detach().item())
                        * objective.active_pair_count
                    )
                    active_pair_incidence_count += (
                        objective.active_pair_count
                    )

                active_pair_count_sum += objective.active_pair_count
                active_pair_union.update(batch_pairs)
                batch_count += 1

            if sample_count == 0 or batch_count == 0:
                raise RuntimeError("base training loader is empty")

            train_classification = classification_sum / sample_count
            train_separation = (
                separation_sum / active_pair_incidence_count
                if active_pair_incidence_count
                else 0.0
            )
            train_total = (
                objective_weights["classification_weight"]
                * train_classification
                + objective_weights["separation_weight"]
                * train_separation
            )
            train_accuracy = accuracy_sum / sample_count
            train_pair_coverage = (
                len(active_pair_union) / expected_pair_count
                if expected_pair_count
                else 1.0
            )

            if (
                objective_weights["separation_weight"] > 0.0
                and active_pair_incidence_count == 0
            ):
                raise RuntimeError(
                    "base separation is enabled but no class pair co-occurred "
                    "in any training minibatch"
                )

            # Validation remains diagnostic only and does not select epochs.
            # Existing geometry metrics such as cell coverage remain useful
            # diagnostics, but they are not part of the new training objective.
            validation = self.evaluate_loader(
                val_loader,
                class_ids=class_ids,
                candidate=candidate,
            )

            record = {
                "epoch": epoch + 1,
                "total": float(train_total),
                "classification": float(train_classification),
                "separation": float(train_separation),
                "accuracy": float(train_accuracy),
                "active_pair_incidence_count": int(
                    active_pair_incidence_count
                ),
                "mean_active_pairs_per_batch": float(
                    active_pair_count_sum / batch_count
                ),
                "covered_pair_count": int(len(active_pair_union)),
                "candidate_pair_count": int(expected_pair_count),
                "pair_coverage": float(train_pair_coverage),
                "validation": validation,
            }
            history.append(record)

            tqdm.write(
                f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
                f"loss={record['total']:.4f} | "
                f"CE={record['classification']:.4f} | "
                f"Sep={record['separation']:.4f} | "
                f"Pairs={record['covered_pair_count']}/"
                f"{record['candidate_pair_count']} | "
                f"train_acc={100.0 * record['accuracy']:.2f}% | "
                f"val_BA={100.0 * validation['balanced_accuracy']:.2f}% | "
                f"MinClass={100.0 * validation['minimum_class_accuracy']:.2f}% | "
                f"CellCov={100.0 * validation['true_cell_coverage']:.2f}% | "
                f"RivalInv={100.0 * validation['rival_cell_invasion_rate']:.2f}% | "
                f"Margin={validation['mean_decision_margin']:.4f}",
                file=sys.stdout,
            )
            progress.set_postfix(
                loss=f"{record['total']:.4f}",
                sep=f"{record['separation']:.4f}",
                pairs=f"{record['covered_pair_count']}/{expected_pair_count}",
            )

        progress.close()

        if len(history) != int(epochs):
            raise RuntimeError(
                "base training did not complete the configured epoch schedule"
            )

        # Compute the exact new objective diagnostics once on deterministic
        # train/validation loaders before commit. The committed candidate is the
        # exact same learned geometry, so these values describe the final state.
        candidate.validate_state()

        final_train_objective = self._evaluate_training_objective(
            train_eval_loader,
            class_ids=class_ids,
            candidate=candidate,
            class_risk_weights=class_weights,
            **objective_weights,
        )
        final_validation_objective = self._evaluate_training_objective(
            val_loader,
            class_ids=class_ids,
            candidate=candidate,
            class_risk_weights=class_weights,
            **objective_weights,
        )

        # Fixed-schedule protocol: persist exactly the final configured epoch.
        self.model.commit_candidate(candidate)
        self.model.eval()
        self.model.validate_model_state()

        geometry_train = self.evaluate_loader(
            train_eval_loader,
            class_ids=class_ids,
            candidate=None,
        )
        geometry_validation = self.evaluate_loader(
            val_loader,
            class_ids=class_ids,
            candidate=None,
        )
        geometry_train["training_objective"] = final_train_objective
        geometry_validation["training_objective"] = (
            final_validation_objective
        )

        class_weight_report = {
            int(class_id): float(class_weights[index].item())
            for index, class_id in enumerate(class_ids)
        }

        return {
            "phase": 0,
            "class_ids": list(class_ids),
            "history": history,
            "final_epoch": int(epochs),
            "final_epoch_report": history[-1],
            "geometry_train": geometry_train,
            "geometry_validation": geometry_validation,
            "geometry_summary": {
                "representation_dim": int(self.model.representation_dim),
                "class_count": len(self.model.geometry_bank),
                "pair_count": int(self.model.geometry_bank.pair_count),
                "objective": (
                    "class-uniform global classification + pair-balanced "
                    "distribution separation"
                ),
                "separation_geometry": (
                    "class distributions projected through their shared "
                    "learned pairwise decision direction"
                ),
                "separation_pair_set": "all base-base pairs",
                "persistent_geometry": (
                    "shared learned pairwise affine boundaries"
                ),
                "classifier": (
                    "parameter-free equal-rule energy over all base classes"
                ),
                "prototype_use": "none",
                "explicit_margin": "none",
                "training_schedule": (
                    "fixed epochs; persistent state is the final epoch"
                ),
                "validation_role": (
                    "diagnostic only; never used for epoch selection"
                ),
                "base_train_class_counts": {
                    int(class_id): int(class_counts[index].item())
                    for index, class_id in enumerate(class_ids)
                },
                "class_risk_weights": class_weight_report,
                "final_training_epoch_pair_coverage": float(
                    history[-1]["pair_coverage"]
                ),
                "full_train_objective_pair_coverage": float(
                    final_train_objective["pair_coverage"]
                ),
                "full_validation_objective_pair_coverage": float(
                    final_validation_objective["pair_coverage"]
                ),
            },
            "geometry_committed": True,
        }


__all__ = ["BasePhaseTrainer"]









# from __future__ import annotations

# """Base-phase training for pairwise decision geometry.

# Phase 0 learns the HSI backbone and the exact pairwise boundaries that will be
# persisted.  There is no measured-box finalization and no train/deployment
# geometry mismatch.
# """

# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Sequence

# import torch
# from tqdm import tqdm

# from losses.loss import geometry_training_objective
# from models.geometry_bank import BoundaryCandidate

# Tensor = torch.Tensor


# def _finite_nonnegative(name: str, value: float) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0.0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# def _as_int(value: object, name: str) -> int:
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer")
#         value = value.item()
#     if isinstance(value, bool):
#         raise ValueError(f"{name} must be an integer")
#     if isinstance(value, Integral):
#         return int(value)
#     if isinstance(value, Real):
#         number = float(value)
#         if math.isfinite(number) and number.is_integer():
#             return int(number)
#     raise ValueError(f"{name} must be an integer")


# class BasePhaseTrainer:
#     @staticmethod
#     def _base_class_ids(values: Sequence[int]) -> tuple[int, ...]:
#         ids = tuple(_as_int(v, "base_class_id") for v in values)
#         if not ids or len(ids) != len(set(ids)) or any(v < 0 for v in ids):
#             raise ValueError("base class IDs must be unique non-negative integers")
#         if len(ids) < 2:
#             raise ValueError("pairwise decision geometry requires at least two base classes")
#         return ids

#     @staticmethod
#     def _all_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
#         return all(bool(torch.isfinite(p).all()) for p in parameters)

#     @torch.no_grad()
#     def _ensure_spectral_normalization(self, loader: Any) -> None:
#         if bool(self.model.spectral_normalization_fitted):
#             return
#         spectra: list[Tensor] = []
#         for batch in loader:
#             _, center_spectrum, _ = self.unpack_batch(batch)
#             spectra.append(center_spectrum.detach())
#         if not spectra:
#             raise RuntimeError("spectral-normalization loader is empty")
#         self.model.fit_spectral_normalization(torch.cat(spectra, dim=0), overwrite=False)
#         if not bool(self.model.spectral_normalization_fitted):
#             raise RuntimeError("spectral normalization was not fitted")

#     @torch.no_grad()
#     def _collect_coordinates(self, loader: Any) -> Dict[str, Tensor]:
#         if not bool(self.model.spectral_normalization_fitted):
#             raise RuntimeError("fit spectral normalization before coordinate collection")
#         states = {m: bool(m.training) for m in self.model.modules()}
#         try:
#             self.model.eval()
#             coordinates: list[Tensor] = []
#             labels: list[Tensor] = []
#             for batch in loader:
#                 patch, spectrum, target = self.unpack_batch(batch)
#                 output = self.model.encode(patch, center_spectrum=spectrum)
#                 coordinates.append(output.coordinates.detach())
#                 labels.append(target.detach())
#         finally:
#             for module, state in states.items():
#                 module.training = state
#         if not labels:
#             raise RuntimeError("coordinate collection loader is empty")
#         z = torch.cat(coordinates, dim=0)
#         y = torch.cat(labels, dim=0).flatten()
#         if z.ndim != 2 or z.size(0) != y.numel() or z.size(1) != self.model.representation_dim:
#             raise RuntimeError("collected representation is invalid")
#         if z.device != self.model.geometry_bank.device or z.dtype != self.model.geometry_bank.dtype:
#             raise RuntimeError("collected representation and geometry disagree in device/dtype")
#         if not bool(torch.isfinite(z).all()):
#             raise RuntimeError("coordinate collection produced NaN/Inf")
#         return {"coordinates": z, "labels": y}

#     def _class_uniform_risk_weights(
#         self,
#         labels: Tensor,
#         class_ids: tuple[int, ...],
#     ) -> tuple[Tensor, Tensor]:
#         counts = torch.stack([labels.eq(c).sum() for c in class_ids]).to(
#             device=self.device, dtype=torch.float32
#         )
#         if bool((counts <= 0).any()):
#             raise RuntimeError("every base class must occur in training")
#         total = counts.sum()
#         weights = total / (len(class_ids) * counts)
#         empirical = (counts * weights).sum() / total
#         if not bool(torch.allclose(empirical, torch.ones_like(empirical), rtol=1e-6, atol=1e-7)):
#             raise RuntimeError("class-uniform risk weights are normalized incorrectly")
#         return weights, counts

#     def train_base_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int = 64,
#         lr: float = 1e-4,
#     ) -> Dict[str, Any]:
#         if int(phase) != 0:
#             raise ValueError("BasePhaseTrainer handles phase 0 only")
#         if int(epochs) <= 0 or int(batch_size) <= 0:
#             raise ValueError("epochs and batch_size must be positive")
#         learning_rate = float(lr)
#         if not math.isfinite(learning_rate) or learning_rate <= 0.0:
#             raise ValueError("lr must be finite and positive")

#         self.model.validate_model_state()
#         if len(self.model.geometry_bank) != 0:
#             raise RuntimeError("base training requires an empty geometry bank")

#         self.dataset.start_phase(0)
#         class_ids = self._base_class_ids(self.dataset.get_new_classes(0))
#         eval_batch = int(self.cfg("eval_batch_size", 256))
#         if eval_batch <= 0:
#             raise ValueError("eval_batch_size must be positive")

#         init_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=eval_batch, shuffle=False
#         )
#         train_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=int(batch_size), shuffle=True
#         )
#         train_eval_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=eval_batch, shuffle=False
#         )
#         val_loader = self.dataset.get_phase_dataloader(
#             0, split="val", batch_size=eval_batch, shuffle=False
#         )

#         self._ensure_spectral_normalization(init_loader)
#         initial = self._collect_coordinates(init_loader)
#         candidate = self.model.initialize_candidate(
#             initial["coordinates"], initial["labels"], class_ids
#         )
#         if not isinstance(candidate, BoundaryCandidate):
#             raise RuntimeError("model returned the wrong geometry candidate type")
#         if candidate.new_class_ids != class_ids or candidate.validate_state() is not True:
#             raise RuntimeError("base boundary candidate is invalid")

#         class_weights, class_counts = self._class_uniform_risk_weights(
#             initial["labels"], class_ids
#         )
#         backbone_parameters = [p for p in self.model.backbone.parameters() if p.requires_grad]
#         geometry_parameters = [p for p in candidate.parameters() if p.requires_grad]
#         if not backbone_parameters or not geometry_parameters:
#             raise RuntimeError("backbone and boundary candidate must both be trainable")
#         if {id(p) for p in backbone_parameters}.intersection(id(p) for p in geometry_parameters):
#             raise RuntimeError("backbone and geometry optimizer groups overlap")

#         weight_decay = _finite_nonnegative("weight_decay", self.cfg("weight_decay", 1e-4))
#         optimizer = torch.optim.AdamW(
#             [
#                 {"params": backbone_parameters, "lr": learning_rate, "weight_decay": weight_decay},
#                 {"params": geometry_parameters, "lr": learning_rate, "weight_decay": 0.0},
#             ]
#         )
#         objective_weights = {
#             "classification_weight": _finite_nonnegative(
#                 "base_classification_weight", self.cfg("base_classification_weight", 1.0)
#             ),
#             "fit_weight": _finite_nonnegative(
#                 "base_fit_weight", self.cfg("base_fit_weight", 1.0)
#             ),
#         }
#         if not any(v > 0.0 for v in objective_weights.values()):
#             raise ValueError("at least one base objective weight must be positive")
#         gradient_clip = _finite_nonnegative(
#             "gradient_clip", self.cfg("gradient_clip", 0.0)
#         )
#         optimized_parameters = backbone_parameters + geometry_parameters

#         history: list[Dict[str, Any]] = []

#         progress = tqdm(
#             range(int(epochs)),
#             desc="Base pairwise decision geometry",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             self.model.train()
#             candidate.train()
#             sums = {"total": 0.0, "classification": 0.0, "fit": 0.0, "accuracy": 0.0}
#             sample_count = 0

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 optimizer.zero_grad(set_to_none=True)
#                 result = self.model(
#                     patch,
#                     center_spectrum=spectrum,
#                     class_ids=class_ids,
#                     candidate=candidate,
#                 )
#                 objective = geometry_training_objective(
#                     output=result.classification,
#                     labels_global=labels,
#                     geometry_bank=self.model.geometry_bank,
#                     candidate=candidate,
#                     class_risk_weights=class_weights,
#                     **objective_weights,
#                 )
#                 objective.total.backward()
#                 if any(
#                     p.grad is not None and not bool(torch.isfinite(p.grad).all())
#                     for p in optimized_parameters
#                 ):
#                     raise RuntimeError("base optimization produced NaN/Inf gradients")
#                 if gradient_clip > 0.0:
#                     torch.nn.utils.clip_grad_norm_(
#                         optimized_parameters, gradient_clip, error_if_nonfinite=True
#                     )
#                 optimizer.step()
#                 if not self._all_finite(optimized_parameters):
#                     raise RuntimeError("base optimization produced NaN/Inf parameters")
#                 candidate.validate_state()

#                 count = int(labels.numel())
#                 sample_count += count
#                 for name in ("total", "classification", "fit"):
#                     sums[name] += float(getattr(objective, name).detach().item()) * count
#                 sums["accuracy"] += float(objective.accuracy.detach().item()) * count

#             if sample_count == 0:
#                 raise RuntimeError("base training loader is empty")

#             validation = self.evaluate_loader(
#                 val_loader,
#                 class_ids=class_ids,
#                 candidate=candidate,
#             )
#             validation_objective = (
#                 objective_weights["classification_weight"] * float(validation["macro_classification"])
#                 + objective_weights["fit_weight"] * float(validation["macro_cell_fit"])
#             )
#             if not math.isfinite(validation_objective):
#                 raise RuntimeError("validation geometry objective is not finite")
#             validation["geometry_objective"] = validation_objective

#             record = {
#                 "epoch": epoch + 1,
#                 "total": sums["total"] / sample_count,
#                 "classification": sums["classification"] / sample_count,
#                 "fit": sums["fit"] / sample_count,
#                 "accuracy": sums["accuracy"] / sample_count,
#                 "validation": validation,
#             }
#             history.append(record)

#             tqdm.write(
#                 f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
#                 f"loss={record['total']:.4f} | CE={record['classification']:.4f} | "
#                 f"Fit={record['fit']:.4f} | train_acc={100.0 * record['accuracy']:.2f}% | "
#                 f"val_GObj={validation_objective:.4f} | "
#                 f"val_BA={100.0 * validation['balanced_accuracy']:.2f}% | "
#                 f"MinClass={100.0 * validation['minimum_class_accuracy']:.2f}% | "
#                 f"CellCov={100.0 * validation['true_cell_coverage']:.2f}% | "
#                 f"RivalInv={100.0 * validation['rival_cell_invasion_rate']:.2f}% | "
#                 f"Margin={validation['mean_decision_margin']:.4f}",
#                 file=sys.stdout,
#             )
#             progress.set_postfix(
#                 loss=f"{record['total']:.4f}",
#                 val_GObj=f"{validation_objective:.4f}",
#             )
#         progress.close()

#         if len(history) != int(epochs):
#             raise RuntimeError(
#                 "base training did not complete the configured epoch schedule"
#             )

#         # Fixed-schedule protocol: persist the state produced by the final
#         # configured epoch. Validation metrics are diagnostic only.
#         candidate.validate_state()
#         self.model.commit_candidate(candidate)
#         self.model.eval()
#         self.model.validate_model_state()

#         geometry_train = self.evaluate_loader(
#             train_eval_loader,
#             class_ids=class_ids,
#             candidate=None,
#         )
#         geometry_validation = self.evaluate_loader(
#             val_loader,
#             class_ids=class_ids,
#             candidate=None,
#         )
#         for metrics in (geometry_train, geometry_validation):
#             metrics["geometry_objective"] = (
#                 objective_weights["classification_weight"] * float(metrics["macro_classification"])
#                 + objective_weights["fit_weight"] * float(metrics["macro_cell_fit"])
#             )

#         class_weight_report = {
#             int(class_id): float(class_weights[index].item())
#             for index, class_id in enumerate(class_ids)
#         }
#         return {
#             "phase": 0,
#             "class_ids": list(class_ids),
#             "history": history,
#             "final_epoch": int(epochs),
#             "final_epoch_report": history[-1],
#             "geometry_train": geometry_train,
#             "geometry_validation": geometry_validation,
#             "geometry_summary": {
#                 "representation_dim": int(self.model.representation_dim),
#                 "class_count": len(self.model.geometry_bank),
#                 "pair_count": int(self.model.geometry_bank.pair_count),
#                 "objective": "class-uniform classification + decision-cell fit",
#                 "persistent_geometry": "shared learned pairwise affine boundaries",
#                 "strict_interior_overlap": "impossible by construction",
#                 "training_schedule": "fixed epochs; persistent state is the final epoch",
#                 "validation_role": "diagnostic only; never used for epoch selection",
#                 "base_train_class_counts": {
#                     int(class_id): int(class_counts[index].item())
#                     for index, class_id in enumerate(class_ids)
#                 },
#                 "class_risk_weights": class_weight_report,
#             },
#             "geometry_committed": True,
#         }


# __all__ = ["BasePhaseTrainer"]


