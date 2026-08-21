from __future__ import annotations

"""Incremental training for HSI pairwise decision geometry.

Phase t > 0 uses two semantically different data sources:

1. REAL current-phase HSI, which supplies new-class evidence;
2. TEMPORARY boundary-selected historical HSI variation, reconstructed directly
   from the persistent spectral-variation bank before the backbone is updated.

The phase-start procedure is:

    direct old spectral supports
        -> encode once with F_{t-1}
        -> initialize old-new/new-new candidate boundaries with real new HSI
        -> select the old support most vulnerable to every old-new boundary
        -> cache each selected old sample's class-incident old boundary response

Incremental optimization keeps two aligned streams. The classification stream
contains the current real-new minibatch plus the complete small selected-replay
set. The separation stream adds only the REAL current-phase classes missing from
that shuffled minibatch, one cyclic support row per missing class. This makes
old-new and new-new pair coverage deterministic without changing CE exposure.
The objective is

    L_inc = lambda_cls * L_CE
          + lambda_sep * L_sep
          + lambda_pres * L_pres,

where ``L_sep`` acts only on candidate old-new/new-new pairs and ``L_pres``
preserves committed old-old class-incident decision coordinates.  Old-old
boundary parameters remain fixed.

The implementation is class-count agnostic. For O historical classes and N
current classes it requires exactly

    old-old committed pairs = O(O-1)/2
    candidate pairs         = O*N + N(N-1)/2
    committed after phase   = (O+N)(O+N-1)/2

and each historical response target has width O-1.
"""

import math
import sys
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from losses.loss import (
    geometry_training_objective,
    historical_response_preservation_objective,
)
from models.geometry_bank import BoundaryCandidate
from models.spectral_replay import (
    SpectralReplayDataset,
    SpectralReplayGenerator,
    SpectralReplaySelection,
)

Tensor = torch.Tensor


def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _canonical_pair_set(class_ids: Sequence[int]) -> set[tuple[int, int]]:
    ids = [int(value) for value in class_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("class IDs must be unique")
    return {
        (min(left, right), max(left, right))
        for index, left in enumerate(ids)
        for right in ids[index + 1 :]
    }


def _expected_candidate_pair_set(
    old_ids: Sequence[int],
    new_ids: Sequence[int],
) -> set[tuple[int, int]]:
    old = [int(value) for value in old_ids]
    new = [int(value) for value in new_ids]
    if set(old).intersection(new):
        raise ValueError("old and new class IDs overlap")
    return {
        (min(old_id, new_id), max(old_id, new_id))
        for old_id in old
        for new_id in new
    }.union(_canonical_pair_set(new))


class IncrementalPhaseTrainer:
    """Train one incremental phase without retaining old real HSI."""

    def _phase_classes(
        self,
        phase: int,
    ) -> tuple[list[int], list[int], list[int]]:
        old_ids = [int(value) for value in self.dataset.get_old_classes(int(phase))]
        new_ids = [int(value) for value in self.dataset.get_new_classes(int(phase))]
        seen_ids = [int(value) for value in self.dataset.get_seen_classes(int(phase))]

        if not old_ids:
            raise RuntimeError("incremental phase has no old classes")
        if not new_ids:
            raise RuntimeError("incremental phase has no new classes")
        if len(old_ids) != len(set(old_ids)) or len(new_ids) != len(set(new_ids)):
            raise RuntimeError("phase class IDs must be unique")
        if set(old_ids).intersection(new_ids):
            raise RuntimeError("old and new class IDs overlap")
        if seen_ids != old_ids + new_ids:
            raise RuntimeError(
                "seen classes must equal historical classes followed by current classes"
            )

        committed = [int(value) for value in self.model.committed_class_ids]
        if committed != old_ids:
            raise RuntimeError(
                f"committed geometry {committed} does not match phase-{phase} "
                f"historical classes {old_ids}"
            )

        expected_old_pair_set = _canonical_pair_set(old_ids)
        committed_pair_set = {
            tuple(map(int, row))
            for row in self.model.geometry_bank.pair_ids.detach().cpu().tolist()
        }
        if committed_pair_set != expected_old_pair_set:
            raise RuntimeError(
                "committed historical geometry is not the complete old-old "
                "pair set for the current phase"
            )
        if int(self.model.geometry_bank.pair_count) != len(expected_old_pair_set):
            raise RuntimeError(
                "committed historical geometry pair count is inconsistent"
            )

        variation_bank = getattr(
            self,
            "spectral_variation_bank",
            getattr(self, "spectral_replay_bank", None),
        )
        if variation_bank is None:
            raise RuntimeError("trainer lacks spectral variation bank")
        replay_ids = [int(value) for value in variation_bank.class_ids.tolist()]
        if replay_ids != old_ids:
            raise RuntimeError(
                f"spectral variation state {replay_ids} does not match historical "
                f"classes {old_ids}"
            )
        variation_bank.validate_state()
        return old_ids, new_ids, seen_ids

    @staticmethod
    def _class_uniform_risk_weights_from_counts(
        counts_by_class: Mapping[int, int],
        class_ids: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        ids = [int(value) for value in class_ids]
        counts_list: list[int] = []
        for class_id in ids:
            value = int(counts_by_class.get(class_id, 0))
            if value <= 0:
                raise RuntimeError(
                    f"incremental decision stream lacks class {class_id}"
                )
            counts_list.append(value)

        counts = torch.tensor(counts_list, device=device, dtype=dtype)
        total = counts.sum()
        weights = total / (len(ids) * counts)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise RuntimeError("class-uniform incremental CE weights are invalid")
        empirical = (counts * weights).sum() / total
        if not bool(torch.allclose(empirical, torch.ones_like(empirical), rtol=1e-6, atol=1e-7)):
            raise RuntimeError("incremental CE weights are normalized incorrectly")
        return weights, counts.to(dtype=torch.long)

    @staticmethod
    def _active_candidate_pairs(
        labels: Tensor,
        candidate: BoundaryCandidate,
    ) -> set[tuple[int, int]]:
        present = set(
            int(value)
            for value in torch.as_tensor(labels)
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

    @staticmethod
    def _current_class_positions(
        *,
        dataset: Any,
        labels: Tensor,
        class_ids: Sequence[int],
    ) -> Dict[int, list[int]]:
        """Map every current class to positions in the real TRAIN dataset.

        ``labels`` comes from the same non-shuffled loader as ``dataset`` so
        positions are exact dataset positions, not global HSI sample IDs.
        """
        y = torch.as_tensor(labels).detach().cpu().flatten().to(torch.long)
        if len(dataset) != int(y.numel()):
            raise RuntimeError(
                "current TRAIN dataset and collected label order are misaligned"
            )

        positions: Dict[int, list[int]] = {}
        for class_id in [int(value) for value in class_ids]:
            rows = torch.nonzero(y.eq(class_id), as_tuple=False).flatten().tolist()
            if not rows:
                raise RuntimeError(
                    f"current TRAIN dataset lacks class {class_id}"
                )
            positions[class_id] = [int(value) for value in rows]
        return positions

    @staticmethod
    def _materialize_missing_new_support(
        *,
        dataset: Any,
        missing_class_ids: Sequence[int],
        positions_by_class: Mapping[int, Sequence[int]],
        cursors_by_class: Dict[int, int],
    ) -> Dict[str, Any] | None:
        """Return one real current-phase sample for each missing new class.

        These rows are used ONLY by pairwise separation. They are not added to
        the CE stream, so natural current-phase classification exposure remains
        unchanged. A per-class cyclic cursor avoids repeatedly using one fixed
        anchor while introducing no persistent memory or arbitrary support size.
        """
        missing = [int(value) for value in missing_class_ids]
        if not missing:
            return None

        samples: list[Mapping[str, Any]] = []
        for class_id in missing:
            positions = [int(value) for value in positions_by_class[class_id]]
            cursor = int(cursors_by_class.get(class_id, 0))
            dataset_position = positions[cursor % len(positions)]
            cursors_by_class[class_id] = cursor + 1

            sample = dataset[dataset_position]
            if not isinstance(sample, Mapping):
                raise RuntimeError(
                    "current TRAIN dataset samples must be mappings"
                )
            sample_label = int(
                torch.as_tensor(sample["label"]).item()
            )
            if sample_label != class_id:
                raise RuntimeError(
                    "class-complete separation support selected the wrong class"
                )
            samples.append(sample)

        return dict(default_collate(samples))

    @staticmethod
    def _replay_loader(
        dataset: SpectralReplayDataset,
        *,
        batch_size: int,
    ) -> DataLoader:
        if len(dataset) <= 0:
            raise RuntimeError("selected replay dataset is empty")
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    @staticmethod
    def _materialize_replay_batch(
        dataset: SpectralReplayDataset,
    ) -> Dict[str, Any]:
        """Materialize the complete selected replay set as one small batch.

        The complete selected set is deliberately paired with every real-new
        minibatch.  This guarantees that every old class is available whenever a
        new class appears, so old-new distribution separation cannot disappear
        because of random ConcatDataset batching.
        """
        if len(dataset) <= 0:
            raise RuntimeError("cannot materialize empty replay dataset")
        samples = [dataset[index] for index in range(len(dataset))]
        batch = default_collate(samples)
        required = {
            "image",
            "raw_center_spectrum",
            "label",
            "old_boundary_response",
            "old_rival_class_ids",
        }
        missing = required - set(batch)
        if missing:
            raise RuntimeError(
                "selected replay is missing historical preservation metadata: "
                f"{sorted(missing)}"
            )
        return dict(batch)

    def _preservation_targets(
        self,
        replay_batch: Mapping[str, Any],
        *,
        expected_rows: int,
        expected_width: int,
    ) -> tuple[Tensor, Tensor]:
        if "old_boundary_response" not in replay_batch or "old_rival_class_ids" not in replay_batch:
            raise RuntimeError("replay batch lacks historical boundary-response targets")
        target = torch.as_tensor(
            replay_batch["old_boundary_response"],
            device=self.device,
            dtype=self.model.geometry_bank.dtype,
        )
        rivals = torch.as_tensor(
            replay_batch["old_rival_class_ids"],
            device=self.device,
            dtype=torch.long,
        )
        if (
            target.ndim != 2
            or target.size(0) != expected_rows
            or target.size(1) != int(expected_width)
        ):
            raise RuntimeError(
                "historical replay target shape is invalid: "
                f"expected [{expected_rows},{expected_width}], "
                f"found {list(target.shape)}"
            )
        if rivals.shape != target.shape:
            raise RuntimeError("historical replay rival IDs are misaligned")
        if not bool(torch.isfinite(target).all()):
            raise RuntimeError("historical replay targets contain NaN/Inf")
        return target, rivals

    @torch.no_grad()
    def _evaluate_preservation(
        self,
        selection: SpectralReplaySelection,
        *,
        old_ids: Sequence[int],
    ) -> Dict[str, float]:
        batch = self._materialize_replay_batch(selection.dataset)
        patch, spectrum, labels = self.unpack_batch(batch)
        target, rivals = self._preservation_targets(
            batch,
            expected_rows=int(labels.numel()),
            expected_width=len(old_ids) - 1,
        )

        states = {
            module: bool(module.training)
            for module in self.model.modules()
        }
        try:
            self.model.eval()
            representation = self.model.encode(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            )
            current = self.model.class_boundary_response(
                representation.coordinates,
                labels,
                class_ids=old_ids,
                candidate=None,
            )
            objective = historical_response_preservation_objective(
                current=current,
                target_margins=target,
                target_rival_class_ids=rivals,
                weight=1.0,
            )
        finally:
            for module, state in states.items():
                module.training = state

        return {
            "mean_absolute_drift": float(
                objective.mean_absolute_drift.item()
            ),
            "max_absolute_drift": float(
                objective.max_absolute_drift.item()
            ),
        }

    @staticmethod
    def _stability_plasticity_diagnostic(
        *,
        old_metrics: Mapping[str, Any],
        new_metrics: Mapping[str, Any],
        preservation: Mapping[str, float],
    ) -> Dict[str, float]:
        old_ba = float(old_metrics["balanced_accuracy"])
        new_ba = float(new_metrics["balanced_accuracy"])
        denominator = old_ba + new_ba
        harmonic = 0.0 if denominator == 0.0 else 2.0 * old_ba * new_ba / denominator
        values = {
            "old_replay_balanced_accuracy": old_ba,
            "new_validation_balanced_accuracy": new_ba,
            "harmonic_old_replay_new_validation": harmonic,
            "old_replay_pair_violation_rate": float(
                old_metrics["macro_true_pair_violation_rate"]
            ),
            "new_validation_pair_violation_rate": float(
                new_metrics["macro_true_pair_violation_rate"]
            ),
            "historical_response_mean_absolute_drift": float(
                preservation["mean_absolute_drift"]
            ),
            "historical_response_max_absolute_drift": float(
                preservation["max_absolute_drift"]
            ),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("incremental diagnostic contains NaN/Inf")
        return values

    def train_incremental_phase(
        self,
        *,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        phase = int(phase)
        epochs = int(epochs)
        batch_size = int(batch_size)
        learning_rate = _finite_positive("lr", lr)

        if phase <= 0:
            raise ValueError("incremental phase must be > 0")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model.validate_model_state()
        self.dataset.start_phase(phase)
        old_ids, new_ids, seen_ids = self._phase_classes(phase)

        eval_batch = int(self.cfg("eval_batch_size", 256))
        if eval_batch <= 0:
            raise ValueError("eval_batch_size must be positive")

        real_train_loader = self.dataset.get_phase_dataloader(
            phase,
            split="train",
            batch_size=batch_size,
            shuffle=True,
        )
        real_train_eval_loader = self.dataset.get_phase_dataloader(
            phase,
            split="train",
            batch_size=eval_batch,
            shuffle=False,
        )
        real_val_loader = self.dataset.get_phase_dataloader(
            phase,
            split="val",
            batch_size=eval_batch,
            shuffle=False,
        )

        current_labels = self.collect_labels(real_train_eval_loader)
        observed_new = sorted(
            int(value) for value in current_labels.unique().tolist()
        )
        if observed_new != sorted(new_ids):
            raise RuntimeError(
                f"phase-{phase} real TRAIN classes {observed_new} do not match "
                f"current classes {sorted(new_ids)}"
            )
        current_counts = {
            class_id: int(current_labels.eq(class_id).sum().item())
            for class_id in new_ids
        }
        if any(value <= 0 for value in current_counts.values()):
            raise RuntimeError("every current class must have real TRAIN evidence")

        current_train_dataset = real_train_eval_loader.dataset
        current_positions_by_class = self._current_class_positions(
            dataset=current_train_dataset,
            labels=current_labels,
            class_ids=new_ids,
        )

        # ------------------------------------------------------------------
        # 1. Direct historical HSI support construction at phase start.
        # ------------------------------------------------------------------
        variation_bank = getattr(
            self,
            "spectral_variation_bank",
            getattr(self, "spectral_replay_bank", None),
        )
        if variation_bank is None:
            raise RuntimeError("trainer lacks spectral variation bank")
        replay_generator = SpectralReplayGenerator(
            bank=variation_bank,
            preprocessor=self.replay_preprocessor,
        )
        tqdm.write(
            f"[Phase {phase}] direct spectral-variation replay | "
            f"old_classes={len(old_ids)} | optimization_steps=0",
            file=sys.stdout,
        )
        replay_result = replay_generator.generate(
            model=self.model,
            class_ids=old_ids,
            batch_size=eval_batch,
        )

        # ------------------------------------------------------------------
        # 2. Initialize only old-new and new-new candidate boundaries using
        #    phase-start old supports and REAL current-class coordinates.
        # ------------------------------------------------------------------
        new_init = self.collect_encoded(real_train_eval_loader)
        initialization_coordinates = torch.cat(
            [
                replay_result.support_coordinates.to(
                    device=self.device,
                    dtype=self.model.geometry_bank.dtype,
                ),
                new_init["coordinates"].to(
                    device=self.device,
                    dtype=self.model.geometry_bank.dtype,
                ),
            ],
            dim=0,
        )
        initialization_labels = torch.cat(
            [
                replay_result.initialization_dataset.labels.to(self.device),
                new_init["labels"].to(self.device),
            ],
            dim=0,
        )
        candidate = self.model.initialize_candidate(
            initialization_coordinates,
            initialization_labels,
            new_ids,
        )
        if not isinstance(candidate, BoundaryCandidate):
            raise RuntimeError("model returned the wrong incremental candidate type")
        if candidate.new_class_ids != tuple(new_ids):
            raise RuntimeError("incremental candidate class IDs are incorrect")
        candidate.validate_state()

        expected_candidate_pair_set = _expected_candidate_pair_set(
            old_ids,
            new_ids,
        )
        expected_candidate_pairs = len(expected_candidate_pair_set)
        candidate_pair_set = {
            tuple(map(int, row))
            for row in candidate.pair_ids.detach().cpu().tolist()
        }
        if candidate_pair_set != expected_candidate_pair_set:
            raise RuntimeError(
                "incremental candidate must contain exactly every old-new "
                "and new-new pair, with no old-old or duplicate relation"
            )
        if int(candidate.pair_ids.size(0)) != expected_candidate_pairs:
            raise RuntimeError("incremental candidate pair count is inconsistent")

        expected_seen_pair_count = len(seen_ids) * (len(seen_ids) - 1) // 2
        if (
            int(self.model.geometry_bank.pair_count)
            + expected_candidate_pairs
            != expected_seen_pair_count
        ):
            raise RuntimeError(
                "old committed geometry plus candidate cannot form a complete "
                "seen-class geometry after commit"
            )

        # ------------------------------------------------------------------
        # 3. Candidate geometry selects decision-critical historical supports,
        #    then committed old-old geometry defines their preservation targets.
        # ------------------------------------------------------------------
        selection = replay_generator.select_boundary_supports(
            model=self.model,
            replay=replay_result,
            candidate=candidate,
            old_class_ids=old_ids,
            new_class_ids=new_ids,
        )
        selection = replay_generator.attach_phase_start_boundary_response(
            model=self.model,
            selection=selection,
        )
        selected_old_classes = sorted(
            int(value)
            for value in selection.dataset.labels.unique().tolist()
        )
        if selected_old_classes != sorted(old_ids):
            raise RuntimeError(
                "boundary-selected replay must retain at least one support for every old class"
            )

        expected_old_new_keys = {
            f"{old_id}:{new_id}"
            for old_id in old_ids
            for new_id in new_ids
        }
        if set(selection.pair_to_pool_index) != expected_old_new_keys:
            raise RuntimeError(
                "boundary selection does not cover every old-new relation"
            )
        if set(selection.pair_to_selected_index) != expected_old_new_keys:
            raise RuntimeError(
                "selected replay index map does not cover every old-new relation"
            )
        response_width = int(
            selection.diagnostics.get("historical_response_width", -1)
        )
        if response_width != len(old_ids) - 1:
            raise RuntimeError(
                "historical boundary-response width must equal old_class_count - 1"
            )

        replay_eval_loader = self._replay_loader(
            selection.dataset,
            batch_size=eval_batch,
        )
        replay_start_geometry = self.evaluate_loader(
            replay_eval_loader,
            class_ids=old_ids,
            target_class_ids=old_ids,
            candidate=None,
        )
        replay_start_preservation = self._evaluate_preservation(
            selection,
            old_ids=old_ids,
        )

        # Materialize once; inputs/targets stay fixed while the backbone changes.
        replay_batch = self._materialize_replay_batch(selection.dataset)
        replay_patch, replay_spectrum, replay_labels = self.unpack_batch(replay_batch)
        preservation_target, preservation_rivals = self._preservation_targets(
            replay_batch,
            expected_rows=int(replay_labels.numel()),
            expected_width=len(old_ids) - 1,
        )

        # ------------------------------------------------------------------
        # 4. Class-uniform CE weights reflect the ACTUAL epoch exposure.  Every
        #    real new sample is seen once per epoch while the complete small
        #    selected replay set is paired with every real-new minibatch.
        # ------------------------------------------------------------------
        try:
            real_batch_count = len(real_train_loader)
        except TypeError as exc:
            raise RuntimeError(
                "incremental real TRAIN loader must expose a finite length"
            ) from exc
        if real_batch_count <= 0:
            raise RuntimeError("incremental real TRAIN loader is empty")

        replay_counts = {
            class_id: int(replay_labels.eq(class_id).sum().item())
            for class_id in old_ids
        }
        effective_counts: Dict[int, int] = {
            class_id: replay_counts[class_id] * real_batch_count
            for class_id in old_ids
        }
        effective_counts.update(current_counts)
        class_weights, effective_count_tensor = self._class_uniform_risk_weights_from_counts(
            effective_counts,
            seen_ids,
            device=self.device,
            dtype=self.model.geometry_bank.dtype,
        )

        backbone_parameters = [
            parameter
            for parameter in self.model.backbone.parameters()
            if parameter.requires_grad
        ]
        candidate_parameters = [
            parameter
            for parameter in candidate.parameters()
            if parameter.requires_grad
        ]
        if not backbone_parameters or not candidate_parameters:
            raise RuntimeError(
                "incremental backbone and candidate geometry must both be trainable"
            )
        if {id(parameter) for parameter in backbone_parameters}.intersection(
            id(parameter) for parameter in candidate_parameters
        ):
            raise RuntimeError("incremental optimizer parameter groups overlap")

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
                    "params": candidate_parameters,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        )

        classification_weight = _finite_nonnegative(
            "incremental_classification_weight",
            self.cfg(
                "incremental_classification_weight",
                self.cfg("base_classification_weight", 1.0),
            ),
        )
        separation_weight = _finite_nonnegative(
            "incremental_separation_weight",
            self.cfg(
                "incremental_separation_weight",
                self.cfg("base_separation_weight", 1.0),
            ),
        )
        preservation_weight = _finite_nonnegative(
            "preservation_weight",
            self.cfg("preservation_weight", 1.0),
        )
        if (
            classification_weight == 0.0
            and separation_weight == 0.0
            and preservation_weight == 0.0
        ):
            raise ValueError("at least one incremental objective weight must be positive")

        gradient_clip = _finite_nonnegative(
            "gradient_clip",
            self.cfg("gradient_clip", 0.0),
        )
        optimized_parameters = backbone_parameters + candidate_parameters

        history: list[Dict[str, Any]] = []
        progress = tqdm(
            range(epochs),
            desc=f"Incremental decision geometry phase {phase}",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        # Temporary phase-local cursors over REAL current-phase data. They
        # cycle through class samples whenever a shuffled minibatch is missing
        # that class, avoiding a single fixed support anchor. Nothing is
        # checkpointed or carried into the next phase.
        current_support_cursors = {
            class_id: 0 for class_id in new_ids
        }

        for epoch in progress:
            self.model.train()
            candidate.train()

            step_total_sum = 0.0
            step_classification_sum = 0.0
            step_preservation_sum = 0.0
            step_preservation_max_sum = 0.0
            step_count = 0

            separation_sum = 0.0
            active_pair_incidence_count = 0
            active_pair_count_sum = 0
            active_pair_union: set[tuple[int, int]] = set()

            decision_correct = 0
            decision_sample_count = 0
            separation_support_rows = 0
            separation_support_steps = 0

            for real_batch in real_train_loader:
                new_patch, new_spectrum, new_labels = self.unpack_batch(real_batch)

                optimizer.zero_grad(set_to_none=True)

                new_representation = self.model.encode(
                    new_patch,
                    center_spectrum=new_spectrum,
                    return_aux=False,
                )
                replay_representation = self.model.encode(
                    replay_patch,
                    center_spectrum=replay_spectrum,
                    return_aux=False,
                )

                # ------------------------------------------------------
                # Classification stream: unchanged natural exposure.
                # ------------------------------------------------------
                decision_coordinates = torch.cat(
                    [
                        new_representation.coordinates,
                        replay_representation.coordinates,
                    ],
                    dim=0,
                )
                decision_labels = torch.cat(
                    [new_labels, replay_labels],
                    dim=0,
                )

                # ------------------------------------------------------
                # Pair-separation stream: ensure every current class is
                # represented. Random shuffling does not guarantee that every
                # new-new pair co-occurs in a minibatch. Add exactly one REAL
                # current-phase sample for each currently missing new class.
                # These support rows affect separation only, never CE.
                # ------------------------------------------------------
                present_new = set(
                    int(value)
                    for value in new_labels.detach().cpu().unique().tolist()
                )
                missing_new = [
                    class_id
                    for class_id in new_ids
                    if class_id not in present_new
                ]

                support_batch = self._materialize_missing_new_support(
                    dataset=current_train_dataset,
                    missing_class_ids=missing_new,
                    positions_by_class=current_positions_by_class,
                    cursors_by_class=current_support_cursors,
                )
                if support_batch is None:
                    separation_coordinates = decision_coordinates
                    separation_labels = decision_labels
                else:
                    (
                        support_patch,
                        support_spectrum,
                        support_labels,
                    ) = self.unpack_batch(support_batch)
                    support_representation = self.model.encode(
                        support_patch,
                        center_spectrum=support_spectrum,
                        return_aux=False,
                    )
                    separation_coordinates = torch.cat(
                        [
                            new_representation.coordinates,
                            support_representation.coordinates,
                            replay_representation.coordinates,
                        ],
                        dim=0,
                    )
                    separation_labels = torch.cat(
                        [
                            new_labels,
                            support_labels,
                            replay_labels,
                        ],
                        dim=0,
                    )
                    separation_support_rows += int(support_labels.numel())
                    separation_support_steps += 1

                observed_separation_new = set(
                    int(value)
                    for value in separation_labels.detach().cpu().unique().tolist()
                    if int(value) in set(new_ids)
                )
                if observed_separation_new != set(new_ids):
                    raise RuntimeError(
                        "class-complete separation stream is missing a current class"
                    )

                classification = self.model.classify_coordinates(
                    decision_coordinates,
                    class_ids=seen_ids,
                    candidate=candidate,
                )
                decision_objective = geometry_training_objective(
                    output=classification,
                    coordinates=decision_coordinates,
                    labels_global=decision_labels,
                    geometry_bank=self.model.geometry_bank,
                    candidate=candidate,
                    class_risk_weights=class_weights,
                    classification_weight=classification_weight,
                    separation_weight=separation_weight,
                    separation_coordinates=separation_coordinates,
                    separation_labels_global=separation_labels,
                )

                current_response = self.model.class_boundary_response(
                    replay_representation.coordinates,
                    replay_labels,
                    class_ids=old_ids,
                    candidate=None,
                )
                preservation_objective = historical_response_preservation_objective(
                    current=current_response,
                    target_margins=preservation_target,
                    target_rival_class_ids=preservation_rivals,
                    weight=preservation_weight,
                )

                total_objective = (
                    decision_objective.total
                    + preservation_objective.total
                )
                if not bool(torch.isfinite(total_objective)):
                    raise RuntimeError("incremental objective is NaN/Inf")
                total_objective.backward()

                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in optimized_parameters
                ):
                    raise RuntimeError(
                        "incremental optimization produced NaN/Inf gradients"
                    )
                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        optimized_parameters,
                        gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()

                if any(
                    not bool(torch.isfinite(parameter).all())
                    for parameter in optimized_parameters
                ):
                    raise RuntimeError(
                        "incremental optimization produced NaN/Inf parameters"
                    )
                candidate.validate_state()

                batch_pairs = self._active_candidate_pairs(
                    separation_labels,
                    candidate,
                )
                if decision_objective.active_pair_count != len(batch_pairs):
                    raise RuntimeError(
                        "loss active-pair count disagrees with the separation stream"
                    )
                if (
                    separation_weight > 0.0
                    and batch_pairs != expected_candidate_pair_set
                ):
                    missing_pairs = sorted(
                        expected_candidate_pair_set - batch_pairs
                    )
                    raise RuntimeError(
                        "class-complete separation stream failed to expose "
                        f"candidate pairs: {missing_pairs}"
                    )

                step_total_sum += float(total_objective.detach().item())
                step_classification_sum += float(
                    decision_objective.classification.detach().item()
                )
                step_preservation_sum += float(
                    preservation_objective.mean_absolute_drift.detach().item()
                )
                step_preservation_max_sum += float(
                    preservation_objective.max_absolute_drift.detach().item()
                )
                step_count += 1

                if decision_objective.active_pair_count:
                    separation_sum += (
                        float(decision_objective.separation.detach().item())
                        * decision_objective.active_pair_count
                    )
                    active_pair_incidence_count += decision_objective.active_pair_count
                active_pair_count_sum += decision_objective.active_pair_count
                active_pair_union.update(batch_pairs)

                count = int(decision_labels.numel())
                decision_sample_count += count
                decision_correct += int(
                    classification.prediction.eq(decision_labels).sum().item()
                )

            if step_count == 0 or decision_sample_count == 0:
                raise RuntimeError("incremental optimization executed no training step")
            if separation_weight > 0.0 and active_pair_incidence_count == 0:
                raise RuntimeError(
                    "incremental separation is enabled but no candidate pair had both sides"
                )

            mean_separation = (
                separation_sum / active_pair_incidence_count
                if active_pair_incidence_count else 0.0
            )
            pair_coverage = (
                len(active_pair_union) / expected_candidate_pairs
                if expected_candidate_pairs else 1.0
            )
            if (
                separation_weight > 0.0
                and active_pair_union != expected_candidate_pair_set
            ):
                missing_pairs = sorted(
                    expected_candidate_pair_set - active_pair_union
                )
                raise RuntimeError(
                    "class-complete incremental separation failed its pair "
                    f"coverage invariant in epoch {epoch + 1}; "
                    f"missing={missing_pairs}"
                )

            new_validation = self.evaluate_loader(
                real_val_loader,
                class_ids=seen_ids,
                target_class_ids=new_ids,
                candidate=candidate,
            )
            old_replay_validation = self.evaluate_loader(
                replay_eval_loader,
                class_ids=seen_ids,
                target_class_ids=old_ids,
                candidate=candidate,
            )
            preservation_diagnostic = self._evaluate_preservation(
                selection,
                old_ids=old_ids,
            )
            diagnostic = self._stability_plasticity_diagnostic(
                old_metrics=old_replay_validation,
                new_metrics=new_validation,
                preservation=preservation_diagnostic,
            )

            record = {
                "epoch": epoch + 1,
                "total": step_total_sum / step_count,
                "classification": step_classification_sum / step_count,
                "separation": float(mean_separation),
                "preservation_mean_absolute_drift": (
                    step_preservation_sum / step_count
                ),
                "preservation_mean_max_absolute_drift": (
                    step_preservation_max_sum / step_count
                ),
                "decision_accuracy": decision_correct / decision_sample_count,
                "active_pair_incidence_count": int(active_pair_incidence_count),
                "mean_active_pairs_per_step": float(
                    active_pair_count_sum / step_count
                ),
                "covered_candidate_pair_count": int(len(active_pair_union)),
                "candidate_pair_count": int(expected_candidate_pairs),
                "pair_coverage": float(pair_coverage),
                "separation_support_rows_added": int(
                    separation_support_rows
                ),
                "separation_support_step_count": int(
                    separation_support_steps
                ),
                "validation_diagnostic": diagnostic,
                "new_validation": new_validation,
                "old_replay_compatibility": old_replay_validation,
                "historical_response_preservation": preservation_diagnostic,
            }
            history.append(record)

            tqdm.write(
                f"[Phase {phase} | Epoch {epoch + 1:03d}/{epochs:03d}] "
                f"loss={record['total']:.4f} | "
                f"CE={record['classification']:.4f} | "
                f"Sep={record['separation']:.4f} | "
                f"Pres={record['preservation_mean_absolute_drift']:.4f} | "
                f"Pairs={record['covered_candidate_pair_count']}/"
                f"{record['candidate_pair_count']} | "
                f"decision_acc={100.0 * record['decision_accuracy']:.2f}% | "
                f"oldReplay_BA={100.0 * old_replay_validation['balanced_accuracy']:.2f}% | "
                f"newVal_BA={100.0 * new_validation['balanced_accuracy']:.2f}% | "
                f"HistDrift={preservation_diagnostic['mean_absolute_drift']:.4f}",
                file=sys.stdout,
            )
            progress.set_postfix(
                loss=f"{record['total']:.4f}",
                sep=f"{record['separation']:.4f}",
                drift=f"{preservation_diagnostic['mean_absolute_drift']:.4f}",
            )

        progress.close()
        if len(history) != epochs:
            raise RuntimeError(
                "incremental training did not complete the configured epoch schedule"
            )

        # Fixed-schedule protocol: keep the final configured state.  The
        # orchestration layer commits this exact candidate after it appends the
        # new classes' REAL spectral variation state.
        candidate.validate_state()
        self.model.eval()

        current_train_geometry = self.evaluate_loader(
            real_train_eval_loader,
            class_ids=seen_ids,
            target_class_ids=new_ids,
            candidate=candidate,
        )
        current_validation_geometry = self.evaluate_loader(
            real_val_loader,
            class_ids=seen_ids,
            target_class_ids=new_ids,
            candidate=candidate,
        )
        replay_seen_geometry = self.evaluate_loader(
            replay_eval_loader,
            class_ids=seen_ids,
            target_class_ids=old_ids,
            candidate=candidate,
        )
        replay_old_geometry = self.evaluate_loader(
            replay_eval_loader,
            class_ids=old_ids,
            target_class_ids=old_ids,
            candidate=None,
        )
        final_preservation = self._evaluate_preservation(
            selection,
            old_ids=old_ids,
        )

        effective_count_report = {
            int(class_id): int(effective_count_tensor[index].item())
            for index, class_id in enumerate(seen_ids)
        }
        class_weight_report = {
            int(class_id): float(class_weights[index].item())
            for index, class_id in enumerate(seen_ids)
        }
        replay_diagnostics = dict(replay_result.diagnostics)
        replay_diagnostics["boundary_selection"] = dict(selection.diagnostics)
        replay_diagnostics["phase_start_preservation"] = replay_start_preservation
        replay_diagnostics["final_preservation"] = final_preservation

        return {
            "phase": phase,
            "old_class_ids": old_ids,
            "new_class_ids": new_ids,
            "seen_class_ids": seen_ids,
            "history": history,
            "final_epoch": epochs,
            "final_epoch_report": history[-1],
            "candidate_geometry": candidate,
            "current_train_geometry": current_train_geometry,
            "current_validation_geometry": current_validation_geometry,
            "old_replay_seen_geometry": replay_seen_geometry,
            "old_replay_old_geometry": replay_old_geometry,
            "replay_start_geometry": replay_start_geometry,
            "historical_response_preservation": final_preservation,
            "replay_diagnostics": replay_diagnostics,
            "phase_summary": {
                "objective": (
                    "global all-seen CE + candidate pairwise distribution separation "
                    "+ committed-old historical boundary-response preservation"
                ),
                "persistent_old_geometry": "old-old pairwise boundaries fixed",
                "trainable_geometry": "old-new and new-new pairwise boundaries only",
                "replay_construction": (
                    "direct correlated ordered-spectral variation; no optimization"
                ),
                "replay_selection": (
                    "minimum old-side signed-distance support per old-new boundary"
                ),
                "selected_replay_support_count": len(selection.dataset),
                "real_new_train_counts": current_counts,
                "effective_ce_class_counts": effective_count_report,
                "class_risk_weights": class_weight_report,
                "old_class_count": len(old_ids),
                "new_class_count": len(new_ids),
                "seen_class_count": len(seen_ids),
                "committed_old_pair_count": int(self.model.geometry_bank.pair_count),
                "candidate_pair_count": int(expected_candidate_pairs),
                "expected_committed_pair_count_after_phase": int(
                    expected_seen_pair_count
                ),
                "old_new_relation_count": len(old_ids) * len(new_ids),
                "historical_response_width": len(old_ids) - 1,
                "decision_batch_policy": (
                    "CE uses each shuffled real-new minibatch plus the complete "
                    "boundary-selected old support set"
                ),
                "separation_batch_policy": (
                    "candidate separation uses the same decision batch plus "
                    "one real current-phase sample for each new class absent "
                    "from that minibatch; support rows affect separation only"
                ),
                "final_pair_coverage": float(history[-1]["pair_coverage"]),
                "preservation_target": (
                    "class-incident committed old boundary responses"
                ),
                "training_schedule": (
                    "fixed epochs; candidate from final epoch is returned for exact commit"
                ),
                "validation_role": (
                    "real-new validation, selected-replay compatibility, and historical "
                    "response drift are diagnostics only; they never select an epoch"
                ),
            },
        }


__all__ = ["IncrementalPhaseTrainer"]











# from __future__ import annotations

# """Incremental phase for the pairwise NECIL-HSI architecture.

# Phase t > 0 uses exactly two data sources:

#     1. REAL current-phase HSI;
#     2. TEMPORARY old spectral replay generated before the backbone is updated.

# Persistent old-old boundaries remain fixed.  A trainable BoundaryCandidate
# contains only the old-new and new-new boundaries introduced by the phase.

# The same objective used at base phase is used on the mixed seen-class data:

#     CE(-E, y) + ReLU(E_y)

# with class-uniform empirical risk.

# Therefore the incremental responsibilities are explicit:

#     replay old + fixed old-old geometry
#         -> representation-drift control / boundary preservation

#     replay old + real new
#         -> old-new boundary learning

#     real new + real new
#         -> new-new boundary learning

#     one pairwise energy for every seen class
#         -> equal-rule classification without a cumulative trainable head

# No feature transport, teacher/KD, current-class head, prototype memory,
# spectral-descriptor classifier, risk temperature, boundary margin, or
# coordinate-alignment loss is used.
# """

# import math
# import sys
# from typing import Any, Dict, Mapping, Sequence

# import torch
# from torch.utils.data import ConcatDataset, DataLoader, default_collate
# from tqdm import tqdm

# from losses.loss import geometry_training_objective
# from models.geometry_bank import BoundaryCandidate
# from models.spectral_replay import SpectralReplayGenerator

# Tensor = torch.Tensor


# def _finite_positive(name: str, value: float) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result <= 0.0:
#         raise ValueError(f"{name} must be finite and positive")
#     return result


# def _finite_nonnegative(name: str, value: float) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0.0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# class IncrementalPhaseTrainer:
#     """Train one incremental phase without retaining old real HSI."""

#     def _phase_classes(
#         self,
#         phase: int,
#     ) -> tuple[list[int], list[int], list[int]]:
#         old_ids = [int(v) for v in self.dataset.get_old_classes(int(phase))]
#         new_ids = [int(v) for v in self.dataset.get_new_classes(int(phase))]
#         seen_ids = [int(v) for v in self.dataset.get_seen_classes(int(phase))]

#         if not old_ids:
#             raise RuntimeError("incremental phase has no old classes")
#         if not new_ids:
#             raise RuntimeError("incremental phase has no new classes")
#         if len(old_ids) != len(set(old_ids)) or len(new_ids) != len(set(new_ids)):
#             raise RuntimeError("phase class IDs must be unique")
#         if set(old_ids).intersection(new_ids):
#             raise RuntimeError("old and new class IDs overlap")
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError(
#                 "seen classes must equal historical classes followed by current classes"
#             )
#         committed = [int(v) for v in self.model.committed_class_ids]
#         if committed != old_ids:
#             raise RuntimeError(
#                 f"committed geometry {committed} does not match phase-{phase} "
#                 f"historical classes {old_ids}"
#             )
#         replay_ids = [int(v) for v in self.spectral_replay_bank.class_ids.tolist()]
#         if replay_ids != old_ids:
#             raise RuntimeError(
#                 f"spectral replay state {replay_ids} does not match historical "
#                 f"classes {old_ids}"
#             )
#         return old_ids, new_ids, seen_ids

#     @staticmethod
#     def _class_uniform_risk_weights(
#         labels: Tensor,
#         class_ids: Sequence[int],
#         *,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> tuple[Tensor, Tensor]:
#         ids = [int(v) for v in class_ids]
#         y = torch.as_tensor(labels, device=device, dtype=torch.long).flatten()
#         counts = torch.stack([y.eq(class_id).sum() for class_id in ids]).to(dtype=dtype)
#         if bool((counts <= 0).any()):
#             missing = [ids[i] for i, count in enumerate(counts) if float(count.item()) <= 0.0]
#             raise RuntimeError(f"mixed incremental training lacks classes {missing}")
#         total = counts.sum()
#         weights = total / (len(ids) * counts)
#         if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
#             raise RuntimeError("class-uniform incremental risk weights are invalid")
#         return weights, counts.to(dtype=torch.long)

#     @staticmethod
#     def _collate_model_inputs(
#         samples: Sequence[Mapping[str, Any]],
#     ) -> Dict[str, Any]:
#         """Collate the shared model-input contract of real and replay HSI.

#         Real dataset samples may additionally contain ``coord``,
#         ``global_index`` or reporting metadata.  Synthetic replay has no real
#         spatial coordinate and must not fabricate one.  Incremental optimization
#         consumes exactly the same three fields as ``TrainerHelper.unpack_batch``:

#             image
#             raw_center_spectrum
#             label

#         Therefore mixed real/replay batching explicitly projects both dataset
#         item types onto this common architecture contract before collation.
#         """
#         if not samples:
#             raise RuntimeError("cannot collate an empty mixed HSI batch")

#         required = ("image", "raw_center_spectrum", "label")
#         projected = []
#         for sample_index, sample in enumerate(samples):
#             if not isinstance(sample, Mapping):
#                 raise TypeError(
#                     f"mixed HSI sample {sample_index} is not a mapping"
#                 )
#             missing = [key for key in required if key not in sample]
#             if missing:
#                 raise KeyError(
#                     f"mixed HSI sample {sample_index} lacks required "
#                     f"model fields {missing}"
#                 )
#             projected.append(
#                 {key: sample[key] for key in required}
#             )
#         return default_collate(projected)

#     def _make_mixed_loader(
#         self,
#         real_loader: Any,
#         replay_dataset: Any,
#         *,
#         batch_size: int,
#         phase: int,
#     ) -> DataLoader:
#         if not hasattr(real_loader, "dataset"):
#             raise RuntimeError("real phase loader does not expose its dataset")
#         combined = ConcatDataset([real_loader.dataset, replay_dataset])
#         generator = torch.Generator()
#         generator.manual_seed(int(self.cfg("seed", 0)) + 20011 * int(phase))
#         return DataLoader(
#             combined,
#             batch_size=int(batch_size),
#             shuffle=True,
#             num_workers=int(self.cfg("num_workers", 0)),
#             collate_fn=self._collate_model_inputs,
#             pin_memory=bool(getattr(real_loader, "pin_memory", False)),
#             drop_last=False,
#             generator=generator,
#         )

#     @staticmethod
#     def _combine_validation_diagnostics(
#         *,
#         old_metrics: Mapping[str, Any],
#         new_metrics: Mapping[str, Any],
#         old_class_count: int,
#         new_class_count: int,
#         classification_weight: float,
#         fit_weight: float,
#     ) -> Dict[str, float]:
#         old_count = int(old_class_count)
#         new_count = int(new_class_count)
#         if old_count <= 0 or new_count <= 0:
#             raise ValueError("selection groups must both contain classes")
#         total_count = old_count + new_count

#         def combine(key: str) -> float:
#             old_value = float(old_metrics[key])
#             new_value = float(new_metrics[key])
#             value = (
#                 old_count * old_value + new_count * new_value
#             ) / total_count
#             if not math.isfinite(value):
#                 raise RuntimeError(f"combined selection metric {key} is not finite")
#             return value

#         macro_classification = combine("macro_classification")
#         macro_cell_fit = combine("macro_cell_fit")
#         balanced_accuracy = combine("balanced_accuracy")
#         minimum_class_accuracy = min(
#             float(old_metrics["minimum_class_accuracy"]),
#             float(new_metrics["minimum_class_accuracy"]),
#         )
#         objective = (
#             float(classification_weight) * macro_classification
#             + float(fit_weight) * macro_cell_fit
#         )
#         if not math.isfinite(objective):
#             raise RuntimeError("incremental selection objective is not finite")

#         return {
#             "geometry_objective": objective,
#             "macro_classification": macro_classification,
#             "macro_cell_fit": macro_cell_fit,
#             "balanced_accuracy": balanced_accuracy,
#             "minimum_class_accuracy": minimum_class_accuracy,
#         }

#     def train_incremental_phase(
#         self,
#         *,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#     ) -> Dict[str, Any]:
#         phase = int(phase)
#         epochs = int(epochs)
#         batch_size = int(batch_size)
#         learning_rate = _finite_positive("lr", lr)

#         if phase <= 0:
#             raise ValueError("incremental phase must be > 0")
#         if epochs <= 0:
#             raise ValueError("epochs must be positive")
#         if batch_size <= 0:
#             raise ValueError("batch_size must be positive")

#         self.model.validate_model_state()
#         self.dataset.start_phase(phase)
#         old_ids, new_ids, seen_ids = self._phase_classes(phase)

#         eval_batch = int(self.cfg("eval_batch_size", 256))
#         if eval_batch <= 0:
#             raise ValueError("eval_batch_size must be positive")

#         real_train_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=True,
#         )
#         real_train_eval_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=eval_batch,
#             shuffle=False,
#         )
#         real_val_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="val",
#             batch_size=eval_batch,
#             shuffle=False,
#         )

#         # ------------------------------------------------------------------
#         # 1. Verify current real TRAIN data and derive replay amount.
#         # ------------------------------------------------------------------
#         current_labels = self.collect_labels(real_train_eval_loader)
#         observed_new = sorted(int(v) for v in current_labels.unique().tolist())
#         if observed_new != sorted(new_ids):
#             raise RuntimeError(
#                 f"phase-{phase} real TRAIN classes {observed_new} do not match "
#                 f"current classes {sorted(new_ids)}"
#             )
#         current_counts = {
#             class_id: int(current_labels.eq(class_id).sum().item())
#             for class_id in new_ids
#         }
#         if any(value <= 0 for value in current_counts.values()):
#             raise RuntimeError("every current class must have real TRAIN evidence")

#         # Replay is temporary synthesis evidence, not an exemplar buffer.
#         # Use one class-balanced synthesis budget at the scale of the existing
#         # training batch rather than duplicating the size of the current TRAIN
#         # set for every historical class.  Class-uniform empirical risk later
#         # gives every seen class equal optimization weight independent of sample
#         # count.
#         replay_per_class = max(
#             2,
#             int(math.ceil(batch_size / len(old_ids))),
#         )

#         # ------------------------------------------------------------------
#         # 2. Generate old spectral replay BEFORE changing the backbone.
#         # ------------------------------------------------------------------
#         replay_generator = SpectralReplayGenerator(
#             bank=self.spectral_replay_bank,
#             preprocessor=self.replay_preprocessor,
#         )
#         replay_steps = int(self.cfg("replay_steps", 100))
#         replay_total = replay_per_class * len(old_ids)
#         tqdm.write(
#             f"[Phase {phase}] spectral replay synthesis | "
#             f"old_classes={len(old_ids)} | "
#             f"samples/class={replay_per_class} | "
#             f"total={replay_total} | "
#             f"steps={replay_steps} | "
#             f"execution_batch={batch_size}"
#         )
#         replay_result = replay_generator.generate(
#             model=self.model,
#             class_ids=old_ids,
#             samples_per_class=replay_per_class,
#             steps=replay_steps,
#             lr=_finite_positive("replay_lr", self.cfg("replay_lr", 1e-2)),
#             seed=int(self.cfg("seed", 0)) + 30011 * phase,
#             optimization_batch_size=batch_size,
#         )
#         replay_loader = DataLoader(
#             replay_result.dataset,
#             batch_size=eval_batch,
#             shuffle=False,
#             num_workers=0,
#             drop_last=False,
#         )

#         # Replay is generated against the exact phase-start old geometry.
#         replay_start_geometry = self.evaluate_loader(
#             replay_loader,
#             class_ids=old_ids,
#             target_class_ids=old_ids,
#             candidate=None,
#         )

#         # ------------------------------------------------------------------
#         # 3. Initialize only old-new and new-new boundaries using evidence from
#         #    both sides: replay-old + real-new.
#         # ------------------------------------------------------------------
#         old_init = self.collect_encoded(replay_loader)
#         new_init = self.collect_encoded(real_train_eval_loader)
#         initialization_coordinates = torch.cat(
#             [
#                 old_init["coordinates"].to(self.device),
#                 new_init["coordinates"].to(self.device),
#             ],
#             dim=0,
#         )
#         initialization_labels = torch.cat(
#             [
#                 old_init["labels"].to(self.device),
#                 new_init["labels"].to(self.device),
#             ],
#             dim=0,
#         )
#         candidate = self.model.initialize_candidate(
#             initialization_coordinates,
#             initialization_labels,
#             new_ids,
#         )
#         if not isinstance(candidate, BoundaryCandidate):
#             raise RuntimeError("model returned the wrong incremental candidate type")
#         if candidate.new_class_ids != tuple(new_ids):
#             raise RuntimeError("incremental candidate class IDs are incorrect")
#         candidate.validate_state()

#         # ------------------------------------------------------------------
#         # 4. Mixed seen-class optimization with the SAME base objective.
#         # ------------------------------------------------------------------
#         mixed_loader = self._make_mixed_loader(
#             real_train_loader,
#             replay_result.dataset,
#             batch_size=batch_size,
#             phase=phase,
#         )
#         mixed_eval_loader = DataLoader(
#             mixed_loader.dataset,
#             batch_size=eval_batch,
#             shuffle=False,
#             num_workers=0,
#             collate_fn=self._collate_model_inputs,
#             drop_last=False,
#         )
#         mixed_labels = self.collect_labels(mixed_eval_loader)
#         class_weights, class_counts = self._class_uniform_risk_weights(
#             mixed_labels,
#             seen_ids,
#             device=self.device,
#             dtype=self.model.geometry_bank.dtype,
#         )

#         backbone_parameters = [
#             parameter
#             for parameter in self.model.backbone.parameters()
#             if parameter.requires_grad
#         ]
#         candidate_parameters = [
#             parameter
#             for parameter in candidate.parameters()
#             if parameter.requires_grad
#         ]
#         if not backbone_parameters or not candidate_parameters:
#             raise RuntimeError(
#                 "incremental backbone and new boundary candidate must both be trainable"
#             )
#         if {id(p) for p in backbone_parameters}.intersection(
#             id(p) for p in candidate_parameters
#         ):
#             raise RuntimeError("incremental optimizer parameter groups overlap")

#         weight_decay = _finite_nonnegative(
#             "weight_decay",
#             self.cfg("weight_decay", 1e-4),
#         )
#         optimizer = torch.optim.AdamW(
#             [
#                 {
#                     "params": backbone_parameters,
#                     "lr": learning_rate,
#                     "weight_decay": weight_decay,
#                 },
#                 {
#                     "params": candidate_parameters,
#                     "lr": learning_rate,
#                     "weight_decay": 0.0,
#                 },
#             ]
#         )

#         # Incremental learning uses exactly the same decision objective as base.
#         classification_weight = _finite_nonnegative(
#             "base_classification_weight",
#             self.cfg("base_classification_weight", 1.0),
#         )
#         fit_weight = _finite_nonnegative(
#             "base_fit_weight",
#             self.cfg("base_fit_weight", 1.0),
#         )
#         if classification_weight == 0.0 and fit_weight == 0.0:
#             raise ValueError("at least one geometry objective weight must be positive")

#         gradient_clip = _finite_nonnegative(
#             "gradient_clip",
#             self.cfg("gradient_clip", 0.0),
#         )
#         optimized_parameters = backbone_parameters + candidate_parameters

#         history: list[Dict[str, Any]] = []

#         progress = tqdm(
#             range(epochs),
#             desc=f"Incremental pairwise geometry phase {phase}",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             self.model.train()
#             candidate.train()
#             sums = {
#                 "total": 0.0,
#                 "classification": 0.0,
#                 "fit": 0.0,
#                 "accuracy": 0.0,
#             }
#             sample_count = 0

#             for batch in mixed_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 optimizer.zero_grad(set_to_none=True)
#                 output = self.model(
#                     patch,
#                     center_spectrum=spectrum,
#                     class_ids=seen_ids,
#                     candidate=candidate,
#                 )
#                 objective = geometry_training_objective(
#                     output=output.classification,
#                     labels_global=labels,
#                     geometry_bank=self.model.geometry_bank,
#                     candidate=candidate,
#                     class_risk_weights=class_weights,
#                     classification_weight=classification_weight,
#                     fit_weight=fit_weight,
#                 )
#                 objective.total.backward()

#                 if any(
#                     parameter.grad is not None
#                     and not bool(torch.isfinite(parameter.grad).all())
#                     for parameter in optimized_parameters
#                 ):
#                     raise RuntimeError(
#                         "incremental optimization produced NaN/Inf gradients"
#                     )
#                 if gradient_clip > 0.0:
#                     torch.nn.utils.clip_grad_norm_(
#                         optimized_parameters,
#                         gradient_clip,
#                         error_if_nonfinite=True,
#                     )
#                 optimizer.step()

#                 if any(
#                     not bool(torch.isfinite(parameter).all())
#                     for parameter in optimized_parameters
#                 ):
#                     raise RuntimeError(
#                         "incremental optimization produced NaN/Inf parameters"
#                     )
#                 candidate.validate_state()

#                 count = int(labels.numel())
#                 sample_count += count
#                 sums["total"] += float(objective.total.detach().item()) * count
#                 sums["classification"] += (
#                     float(objective.classification.detach().item()) * count
#                 )
#                 sums["fit"] += float(objective.fit.detach().item()) * count
#                 sums["accuracy"] += float(objective.accuracy.detach().item()) * count

#             if sample_count == 0:
#                 raise RuntimeError("mixed incremental training loader is empty")

#             # Real current validation measures plasticity/new->old interference.
#             new_validation = self.evaluate_loader(
#                 real_val_loader,
#                 class_ids=seen_ids,
#                 target_class_ids=new_ids,
#                 candidate=candidate,
#             )
#             # Replay-old compatibility measures stability against the historical
#             # decision state.  It is a constraint diagnostic, not real old validation.
#             old_replay_validation = self.evaluate_loader(
#                 replay_loader,
#                 class_ids=seen_ids,
#                 target_class_ids=old_ids,
#                 candidate=candidate,
#             )
#             diagnostic = self._combine_validation_diagnostics(
#                 old_metrics=old_replay_validation,
#                 new_metrics=new_validation,
#                 old_class_count=len(old_ids),
#                 new_class_count=len(new_ids),
#                 classification_weight=classification_weight,
#                 fit_weight=fit_weight,
#             )

#             record = {
#                 "epoch": epoch + 1,
#                 "total": sums["total"] / sample_count,
#                 "classification": sums["classification"] / sample_count,
#                 "fit": sums["fit"] / sample_count,
#                 "accuracy": sums["accuracy"] / sample_count,
#                 "validation_diagnostic": diagnostic,
#                 "new_validation": new_validation,
#                 "old_replay_compatibility": old_replay_validation,
#             }
#             history.append(record)

#             tqdm.write(
#                 f"[Phase {phase} | Epoch {epoch + 1:03d}/{epochs:03d}] "
#                 f"loss={record['total']:.4f} | "
#                 f"CE={record['classification']:.4f} | "
#                 f"Fit={record['fit']:.4f} | "
#                 f"train_acc={100.0 * record['accuracy']:.2f}% | "
#                 f"diag_GObj={diagnostic['geometry_objective']:.4f} | "
#                 f"oldReplay_BA={100.0 * old_replay_validation['balanced_accuracy']:.2f}% | "
#                 f"newVal_BA={100.0 * new_validation['balanced_accuracy']:.2f}% | "
#                 f"newVal_Inv={100.0 * new_validation['rival_cell_invasion_rate']:.2f}%",
#                 file=sys.stdout,
#             )
#             progress.set_postfix(
#                 loss=f"{record['total']:.4f}",
#                 diag_GObj=f"{diagnostic['geometry_objective']:.4f}",
#             )
#         progress.close()

#         if len(history) != epochs:
#             raise RuntimeError(
#                 "incremental training did not complete the configured epoch schedule"
#             )

#         # Fixed-schedule protocol: keep the final configured incremental state.
#         candidate.validate_state()
#         self.model.eval()

#         current_train_geometry = self.evaluate_loader(
#             real_train_eval_loader,
#             class_ids=seen_ids,
#             target_class_ids=new_ids,
#             candidate=candidate,
#         )
#         current_validation_geometry = self.evaluate_loader(
#             real_val_loader,
#             class_ids=seen_ids,
#             target_class_ids=new_ids,
#             candidate=candidate,
#         )
#         replay_seen_geometry = self.evaluate_loader(
#             replay_loader,
#             class_ids=seen_ids,
#             target_class_ids=old_ids,
#             candidate=candidate,
#         )
#         # Representation drift is measured against the untouched historical
#         # old-old geometry only.
#         replay_old_geometry = self.evaluate_loader(
#             replay_loader,
#             class_ids=old_ids,
#             target_class_ids=old_ids,
#             candidate=None,
#         )

#         class_count_report = {
#             int(class_id): int(class_counts[index].item())
#             for index, class_id in enumerate(seen_ids)
#         }
#         class_weight_report = {
#             int(class_id): float(class_weights[index].item())
#             for index, class_id in enumerate(seen_ids)
#         }

#         return {
#             "phase": phase,
#             "old_class_ids": old_ids,
#             "new_class_ids": new_ids,
#             "seen_class_ids": seen_ids,
#             "history": history,
#             "final_epoch": epochs,
#             "final_epoch_report": history[-1],
#             "candidate_geometry": candidate,
#             "current_train_geometry": current_train_geometry,
#             "current_validation_geometry": current_validation_geometry,
#             "old_replay_seen_geometry": replay_seen_geometry,
#             "old_replay_old_geometry": replay_old_geometry,
#             "replay_start_geometry": replay_start_geometry,
#             "replay_diagnostics": replay_result.diagnostics,
#             "phase_summary": {
#                 "objective": (
#                     "class-uniform classification + decision-cell fit on "
#                     "real-new plus replay-old HSI"
#                 ),
#                 "persistent_old_geometry": "old-old pairwise boundaries fixed",
#                 "trainable_geometry": "old-new and new-new pairwise boundaries only",
#                 "replay_samples_per_old_class": replay_per_class,
#                 "replay_total_samples": replay_per_class * len(old_ids),
#                 "replay_budget_policy": (
#                     "temporary class-balanced synthesis at training-batch scale"
#                 ),
#                 "replay_optimization_batch_size": batch_size,
#                 "real_new_train_counts": current_counts,
#                 "mixed_train_class_counts": class_count_report,
#                 "class_risk_weights": class_weight_report,
#                 "training_schedule": (
#                     "fixed epochs; persistent state is the final epoch"
#                 ),
#                 "validation_role": (
#                     "real-new validation and replay-old compatibility are "
#                     "diagnostics only; they never select an epoch"
#                 ),
#             },
#         }


# __all__ = ["IncrementalPhaseTrainer"]


