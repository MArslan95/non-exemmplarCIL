from __future__ import annotations

"""Phase orchestration for HSI pairwise decision geometry.

Persistent continual state consists of:

1. the model/backbone and committed pairwise decision geometry;
2. compact REAL-data-derived ordered spectral variation for every finalized class.

Replay samples themselves are never checkpointed.  They are reconstructed
transiently at the start of an incremental phase and selected by the candidate
old-new decision geometry.
"""

import hashlib
import os
from typing import Any, Dict, Mapping

import numpy as np
import torch

from data.hsi_dataloader_pytorch import LoadHSIPreprocessor
from models.spectral_replay import (
    FrozenHSIPreprocessor,
    SpectralVariationBank,
)
from trainers.base_phase_trainer import BasePhaseTrainer
from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
from trainers.trainer_helpers import TrainerHelper


_CHECKPOINT_VERSION = 2


class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
    """Own phase lifecycle, persistent geometry, and HSI variation state."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Any,
        args: Any,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.args = args

        configured_device = getattr(args, "device", None)
        requested = (
            str(configured_device)
            if configured_device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        try:
            self.device = torch.device(requested)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid training device {requested!r}") from exc

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {self.device} was requested but CUDA is unavailable"
                )
            index = 0 if self.device.index is None else self.device.index
            if not 0 <= index < torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device index {index} is unavailable; "
                    f"device_count={torch.cuda.device_count()}"
                )

        self.model.to(self.device)
        self.save_dir = os.path.abspath(
            str(getattr(args, "save_dir", "./outputs"))
        )
        os.makedirs(self.save_dir, exist_ok=True)

        dataset_contract = self.assert_dataset_contract()
        self.assert_model_contract()

        self.phase_schedule = {
            int(phase): [int(value) for value in classes]
            for phase, classes in dataset_contract["schedule"].items()
        }
        self.base_class_ids = [
            int(value) for value in dataset_contract["base_classes"]
        ]
        self.history: Dict[int, Dict[str, Any]] = {}

        # Phase-0 preprocessing is immutable across the continual sequence.
        preprocessor_path = os.path.join(
            self.save_dir, "phase_0_preprocessor.npz"
        )
        if not os.path.isfile(preprocessor_path):
            raise FileNotFoundError(
                "spectral variation replay requires the frozen phase-0 preprocessing "
                f"state: {preprocessor_path}"
            )
        preprocessing_state = LoadHSIPreprocessor(preprocessor_path)
        self.replay_preprocessor = FrozenHSIPreprocessor(
            preprocessing_state
        ).to(self.device)

        spectral_bands = int(self.model.backbone.spectral_bands)
        if int(self.replay_preprocessor.raw_bands) != spectral_bands:
            raise RuntimeError(
                "replay preprocessor and backbone ordered-band counts disagree"
            )
        if int(self.replay_preprocessor.processed_bands) != int(
            self.model.backbone.patch_bands
        ):
            raise RuntimeError(
                "replay preprocessor and backbone processed-band counts disagree"
            )

        self.spectral_variation_bank = SpectralVariationBank(spectral_bands)
        # Temporary compatibility alias for code outside this trainer.  Both
        # names reference the same object; new code should use variation_bank.
        self.spectral_replay_bank = self.spectral_variation_bank

    # ------------------------------------------------------------------
    # Dataset / checkpoint identity
    # ------------------------------------------------------------------

    @staticmethod
    def _array_fingerprint(value: Any) -> str:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    def _split_fingerprint(self) -> str:
        split_by_class = getattr(self.dataset, "_split_by_class", None)
        if not isinstance(split_by_class, Mapping):
            return "unavailable"

        digest = hashlib.sha256()
        for class_id in sorted(int(value) for value in split_by_class):
            row = split_by_class[class_id]
            if not isinstance(row, Mapping):
                raise RuntimeError("dataset class split must be a mapping")
            digest.update(np.asarray([class_id], dtype=np.int64).tobytes())
            for split in ("train", "val", "test"):
                if split not in row:
                    raise RuntimeError(
                        f"dataset class {class_id} lacks {split} split"
                    )
                values = np.ascontiguousarray(
                    np.asarray(row[split], dtype=np.int64).reshape(-1)
                )
                digest.update(split.encode("ascii"))
                digest.update(
                    np.asarray([values.size], dtype=np.int64).tobytes()
                )
                digest.update(memoryview(values).cast("B"))
        return digest.hexdigest()

    def _dataset_identity(self) -> Dict[str, Any]:
        processed_cube = getattr(self.dataset, "processed_cube", None)
        ordered_cube = getattr(self.dataset, "ordered_spectral_cube", None)
        labels = getattr(self.dataset, "labels", None)
        coords = getattr(self.dataset, "coords", None)
        if processed_cube is None or ordered_cube is None:
            raise RuntimeError(
                "dataset must expose processed and ordered spectral cubes"
            )
        if labels is None or coords is None:
            raise RuntimeError("dataset must expose labels and coordinates")

        return {
            "seed": int(
                getattr(
                    self.dataset,
                    "seed",
                    getattr(self.args, "seed", 0),
                )
            ),
            "phase_schedule": self.phase_schedule,
            "class_order_original_ids": [
                int(value)
                for value in getattr(
                    self.dataset,
                    "class_order_original_ids",
                    sorted({
                        class_id
                        for ids in self.phase_schedule.values()
                        for class_id in ids
                    }),
                )
            ],
            "split_strategy": str(
                getattr(
                    self.dataset,
                    "split_strategy",
                    getattr(self.args, "split_strategy", ""),
                )
            ),
            "spatial_partition_mode": str(
                getattr(self.dataset, "spatial_partition_mode", "")
            ),
            "context_policy": str(
                getattr(self.dataset, "context_policy", "")
            ),
            "patch_size": int(getattr(self.dataset, "patch_size")),
            "train_ratio": (
                None
                if getattr(self.args, "train_ratio", None) is None
                else float(getattr(self.args, "train_ratio"))
            ),
            "val_ratio": (
                None
                if getattr(self.args, "val_ratio", None) is None
                else float(getattr(self.args, "val_ratio"))
            ),
            "require_patch_disjoint": bool(
                getattr(
                    self.dataset,
                    "require_patch_disjoint",
                    getattr(self.args, "require_patch_disjoint", False),
                )
            ),
            "pca_components": int(
                getattr(
                    self.args,
                    "pca_components",
                    processed_cube.shape[2],
                )
            ),
            "pca_whiten": bool(getattr(self.args, "pca_whiten", False)),
            "processed_cube_shape": [
                int(value) for value in processed_cube.shape
            ],
            "ordered_spectral_cube_shape": [
                int(value) for value in ordered_cube.shape
            ],
            "processed_cube_sha256": self._array_fingerprint(processed_cube),
            "ordered_spectral_cube_sha256": self._array_fingerprint(ordered_cube),
            "labels_sha256": self._array_fingerprint(labels),
            "coordinates_sha256": self._array_fingerprint(coords),
            "split_membership_sha256": self._split_fingerprint(),
        }

    def _replay_preprocessor_state(self) -> Dict[str, Any]:
        preprocessor = self.replay_preprocessor
        return {
            "raw_bands": int(preprocessor.raw_bands),
            "fit_pixel_count": int(preprocessor.fit_pixel_count),
            "normalization_mean": (
                preprocessor.normalization_mean.detach().cpu().clone()
            ),
            "normalization_std": (
                preprocessor.normalization_std.detach().cpu().clone()
            ),
            "pca_components": (
                preprocessor.pca_components.detach().cpu().clone()
            ),
            "pca_mean": preprocessor.pca_mean.detach().cpu().clone(),
            "pca_variance": (
                preprocessor.pca_variance.detach().cpu().clone()
            ),
            "whiten": bool(preprocessor.whiten),
            "fit_scope": str(preprocessor.fit_scope),
        }

    @staticmethod
    def _same_preprocessor_state(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> bool:
        scalar_keys = (
            "raw_bands",
            "fit_pixel_count",
            "whiten",
            "fit_scope",
        )
        for key in scalar_keys:
            if left.get(key) != right.get(key):
                return False

        tensor_keys = (
            "normalization_mean",
            "normalization_std",
            "pca_components",
            "pca_mean",
            "pca_variance",
        )
        for key in tensor_keys:
            if key not in left or key not in right:
                return False
            a = torch.as_tensor(left[key]).cpu()
            b = torch.as_tensor(right[key]).cpu()
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if not torch.equal(a, b):
                return False
        return True

    # ------------------------------------------------------------------
    # Finalized-state contracts
    # ------------------------------------------------------------------

    def _finalized_phases(self) -> list[int]:
        raw = getattr(self.dataset, "finalized_phases", None)
        if raw is None:
            raise RuntimeError("dataset must expose finalized_phases")
        phases = [int(value) for value in raw]
        if (
            phases != sorted(set(phases))
            or any(value < 0 for value in phases)
        ):
            raise RuntimeError("dataset finalized_phases is invalid")
        return phases

    def _seen_classes(self, phase: int) -> list[int]:
        phase = int(phase)
        if phase not in self.phase_schedule:
            raise ValueError(f"unknown phase {phase}")
        result: list[int] = []
        for current in range(phase + 1):
            result.extend(self.phase_schedule[current])
        return result

    @staticmethod
    def _canonical_pair_set(
        class_ids: list[int],
    ) -> set[tuple[int, int]]:
        return {
            (min(left, right), max(left, right))
            for index, left in enumerate(class_ids)
            for right in class_ids[index + 1 :]
        }

    @staticmethod
    def _expected_candidate_pair_set(
        old_ids: list[int],
        new_ids: list[int],
    ) -> set[tuple[int, int]]:
        return {
            (min(old_id, new_id), max(old_id, new_id))
            for old_id in old_ids
            for new_id in new_ids
        }.union(Trainer._canonical_pair_set(new_ids))

    def _assert_spectral_normalization_ready(self) -> None:
        if not bool(
            getattr(self.model, "spectral_normalization_fitted", False)
        ):
            raise RuntimeError(
                "ordered-spectrum normalization is not fitted from base training data"
            )

    def _assert_finalized_state(self, phase: int) -> None:
        phase = int(phase)
        expected_phases = list(range(phase + 1))
        if self._finalized_phases() != expected_phases:
            raise RuntimeError(f"expected finalized phases {expected_phases}")
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError(
                "a finalized continual state cannot retain an active phase"
            )

        seen = self._seen_classes(phase)
        committed = [int(value) for value in self.model.committed_class_ids]
        if committed != seen:
            raise RuntimeError(
                "committed geometry does not match the finalized schedule: "
                f"bank={committed}, schedule={seen}"
            )

        expected_pair_set = self._canonical_pair_set(seen)
        actual_pair_set = {
            tuple(map(int, row))
            for row in self.model.geometry_bank.pair_ids.detach().cpu().tolist()
        }
        if actual_pair_set != expected_pair_set:
            raise RuntimeError(
                "finalized committed geometry is not the complete pair set "
                "for all seen classes"
            )
        if int(self.model.geometry_bank.pair_count) != len(expected_pair_set):
            raise RuntimeError(
                "finalized committed geometry pair count is inconsistent"
            )

        variation_ids = [
            int(value)
            for value in self.spectral_variation_bank.class_ids.tolist()
        ]
        if variation_ids != seen:
            raise RuntimeError(
                "spectral variation state does not match the finalized schedule: "
                f"variation={variation_ids}, schedule={seen}"
            )
        self.spectral_variation_bank.validate_state()
        self._assert_spectral_normalization_ready()
        self.model.validate_model_state()

    # ------------------------------------------------------------------
    # Base phase
    # ------------------------------------------------------------------

    def run_base_only(
        self,
        *,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        if self._finalized_phases():
            raise RuntimeError("base phase requires an unfinalized dataset")
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError("base phase requires an empty geometry bank")
        if len(self.spectral_variation_bank) != 0:
            raise RuntimeError(
                "base phase requires an empty spectral variation bank"
            )

        result = dict(
            self.train_base_phase(
                phase=0,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
            )
        )
        required = {
            "class_ids",
            "final_epoch",
            "final_epoch_report",
            "geometry_train",
            "geometry_validation",
            "geometry_committed",
        }
        missing = required - set(result)
        if missing:
            raise RuntimeError(
                "BasePhaseTrainer returned an incomplete result: "
                f"{sorted(missing)}"
            )

        class_ids = [int(value) for value in result["class_ids"]]
        if class_ids != self.base_class_ids:
            raise RuntimeError(
                "base trainer class IDs disagree with dataset schedule"
            )
        if result["geometry_committed"] is not True:
            raise RuntimeError(
                "BasePhaseTrainer did not commit the learned base boundaries"
            )
        if [int(value) for value in self.model.committed_class_ids] != class_ids:
            raise RuntimeError(
                "committed class IDs disagree with base trainer output"
            )

        evaluation_batch_size = int(self.cfg("eval_batch_size", 256))
        if evaluation_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        base_train_loader = self.dataset.get_phase_dataloader(
            0,
            split="train",
            batch_size=evaluation_batch_size,
            shuffle=False,
        )
        self.spectral_variation_bank.append_from_loader(
            base_train_loader,
            class_ids=class_ids,
        )
        if [
            int(value)
            for value in self.spectral_variation_bank.class_ids.tolist()
        ] != class_ids:
            raise RuntimeError("base spectral variation state is incomplete")

        self._assert_spectral_normalization_ready()
        self.model.validate_model_state()
        geometry_state = self.geometry_state_summary()

        self.dataset.finalize_phase(0)
        self._assert_finalized_state(0)

        test_loader = self.dataset.get_cumulative_dataloader(
            0,
            split="test",
            batch_size=evaluation_batch_size,
            shuffle=False,
        )
        geometry_test = self.evaluate_loader(
            test_loader,
            class_ids=class_ids,
            candidate=None,
        )

        persistent = {
            "phase": 0,
            "class_ids": class_ids,
            "final_epoch": int(result["final_epoch"]),
            "optimization_history": result.get("history", []),
            "final_epoch_report": result["final_epoch_report"],
            "geometry_train": result["geometry_train"],
            "geometry_validation": result["geometry_validation"],
            "geometry_test": geometry_test,
            "geometry_summary": result.get("geometry_summary", {}),
            "geometry_state": geometry_state,
            "spectral_variation_state": self.spectral_variation_bank.summary(),
            # Reporting compatibility only.
            "spectral_replay_state": self.spectral_variation_bank.summary(),
            "runtime_scope": "continual",
            "incremental_enabled": True,
        }
        self.history[0] = persistent

        phase_dir = os.path.join(self.save_dir, "phase_0")
        os.makedirs(phase_dir, exist_ok=True)
        checkpoint_path = self.save_checkpoint(
            os.path.join(phase_dir, "checkpoint.pth")
        )
        report_path = self.save_json(
            os.path.join(phase_dir, "base_geometry_report.json"),
            persistent,
        )

        result.update(
            {
                "geometry_test": geometry_test,
                "geometry_state": geometry_state,
                "spectral_variation_state": self.spectral_variation_bank.summary(),
                "spectral_replay_state": self.spectral_variation_bank.summary(),
                "checkpoint": checkpoint_path,
                "report": report_path,
                "phase_summary": persistent,
                "runtime_scope": "continual",
                "incremental_enabled": True,
            }
        )
        return result

    # ------------------------------------------------------------------
    # Incremental phase lifecycle
    # ------------------------------------------------------------------

    def run_incremental_phase(
        self,
        *,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        phase = int(phase)
        if phase <= 0 or phase not in self.phase_schedule:
            raise ValueError(
                "run_incremental_phase requires a valid phase > 0"
            )
        if self._finalized_phases() != list(range(phase)):
            raise RuntimeError(
                f"phase {phase} requires finalized prefix {list(range(phase))}"
            )
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")
        self._assert_finalized_state(phase - 1)

        # Reporting-only reference captured before any phase-t update. It is
        # never used for training or epoch selection and upgrades compatible
        # older checkpoints whose history lacks explicit per-pair diagnostics.
        evaluation_batch_size = int(self.cfg("eval_batch_size", 256))
        if evaluation_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        pre_phase_old_ids = self._seen_classes(phase - 1)
        pre_phase_test_loader = self.dataset.get_cumulative_dataloader(
            phase - 1,
            split="test",
            batch_size=evaluation_batch_size,
            shuffle=False,
        )
        pre_phase_test = self.evaluate_loader(
            pre_phase_test_loader,
            class_ids=pre_phase_old_ids,
            candidate=None,
        )

        result = dict(
            self.train_incremental_phase(
                phase=phase,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
            )
        )
        required = {
            "old_class_ids",
            "new_class_ids",
            "seen_class_ids",
            "candidate_geometry",
            "current_train_geometry",
            "current_validation_geometry",
            "old_replay_seen_geometry",
            "old_replay_old_geometry",
            "historical_response_preservation",
            "replay_diagnostics",
            "final_epoch",
            "final_epoch_report",
        }
        missing = required - set(result)
        if missing:
            raise RuntimeError(
                "IncrementalPhaseTrainer returned an incomplete result: "
                f"{sorted(missing)}"
            )

        old_ids = [int(value) for value in result["old_class_ids"]]
        new_ids = [int(value) for value in result["new_class_ids"]]
        seen_ids = [int(value) for value in result["seen_class_ids"]]
        if old_ids != self._seen_classes(phase - 1):
            raise RuntimeError(
                "incremental trainer historical classes disagree with finalized state"
            )
        if new_ids != self.phase_schedule[phase]:
            raise RuntimeError(
                "incremental trainer current classes disagree with schedule"
            )
        if seen_ids != old_ids + new_ids:
            raise RuntimeError(
                "incremental trainer seen-class order is invalid"
            )

        candidate = result["candidate_geometry"]
        if not hasattr(candidate, "new_class_ids"):
            raise RuntimeError("incremental trainer returned invalid candidate geometry")
        if candidate.new_class_ids != tuple(new_ids):
            raise RuntimeError("incremental candidate class IDs are invalid")
        candidate.validate_state()

        expected_candidate_pair_set = self._expected_candidate_pair_set(
            old_ids,
            new_ids,
        )
        candidate_pair_set = {
            tuple(map(int, row))
            for row in candidate.pair_ids.detach().cpu().tolist()
        }
        if candidate_pair_set != expected_candidate_pair_set:
            raise RuntimeError(
                "incremental candidate pair identities are incomplete or invalid"
            )

        expected_seen_pair_set = self._canonical_pair_set(seen_ids)
        old_pair_set = {
            tuple(map(int, row))
            for row in self.model.geometry_bank.pair_ids.detach().cpu().tolist()
        }
        if old_pair_set.union(candidate_pair_set) != expected_seen_pair_set:
            raise RuntimeError(
                "old committed plus candidate geometry cannot form the complete "
                "seen-class pair state"
            )

        spectral_loader = self.dataset.get_phase_dataloader(
            phase,
            split="train",
            batch_size=evaluation_batch_size,
            shuffle=False,
        )

        # Pre-validate new spectral variation without mutating persistent state.
        # This prevents a malformed real phase from being discovered only after
        # the geometry candidate has been committed.
        validation_bank = SpectralVariationBank(
            int(self.model.backbone.spectral_bands)
        )
        validation_bank.append_from_loader(
            spectral_loader,
            class_ids=new_ids,
        )
        validation_bank.validate_state()

        # Commit the exact final candidate, then append only REAL current-phase
        # spectral variation.  Pseudo replay never updates historical rows.
        self.model.commit_candidate(candidate)
        self.model.eval()
        self.model.validate_model_state()
        self.spectral_variation_bank.append_from_loader(
            spectral_loader,
            class_ids=new_ids,
        )

        if [int(value) for value in self.model.committed_class_ids] != seen_ids:
            raise RuntimeError(
                "geometry commit did not produce the complete seen-class state"
            )
        committed_pair_set = {
            tuple(map(int, row))
            for row in self.model.geometry_bank.pair_ids.detach().cpu().tolist()
        }
        if committed_pair_set != expected_seen_pair_set:
            raise RuntimeError(
                "geometry commit did not produce the complete seen-class pair set"
            )
        if int(self.model.geometry_bank.pair_count) != len(expected_seen_pair_set):
            raise RuntimeError(
                "geometry commit produced an inconsistent pair count"
            )
        if [
            int(value)
            for value in self.spectral_variation_bank.class_ids.tolist()
        ] != seen_ids:
            raise RuntimeError(
                "spectral variation state did not append exactly the new classes"
            )

        geometry_state = self.geometry_state_summary()
        self.dataset.finalize_phase(phase)
        self._assert_finalized_state(phase)

        test_loader = self.dataset.get_cumulative_dataloader(
            phase,
            split="test",
            batch_size=evaluation_batch_size,
            shuffle=False,
        )
        cumulative_test = self.evaluate_loader(
            test_loader,
            class_ids=seen_ids,
            candidate=None,
        )

        old_test = self.summarize_class_group(cumulative_test, old_ids)
        new_test = self.summarize_class_group(cumulative_test, new_ids)
        denominator = (
            float(old_test["balanced_accuracy"])
            + float(new_test["balanced_accuracy"])
        )
        harmonic = (
            0.0
            if denominator == 0.0
            else (
                2.0
                * float(old_test["balanced_accuracy"])
                * float(new_test["balanced_accuracy"])
                / denominator
            )
        )

        previous_test = pre_phase_test
        previous_old = self.summarize_class_group(previous_test, old_ids)
        previous_pair_metrics = previous_test.get(
            "pairwise_boundary_metrics", {}
        )
        current_pair_metrics = cumulative_test.get(
            "pairwise_boundary_metrics", {}
        )
        old_pairwise_delta: Dict[str, Dict[str, float]] = {}
        for left_id, right_id in sorted(self._canonical_pair_set(old_ids)):
            key = f"{left_id}-{right_id}"
            previous_pair = previous_pair_metrics.get(key)
            current_pair = current_pair_metrics.get(key)
            if not isinstance(previous_pair, Mapping) or not isinstance(
                current_pair, Mapping
            ):
                continue
            old_pairwise_delta[key] = {
                "combined_violation_rate_delta": (
                    float(current_pair["combined_violation_rate"])
                    - float(previous_pair["combined_violation_rate"])
                ),
                "minimum_side_mean_margin_delta": (
                    float(current_pair["minimum_side_mean_margin"])
                    - float(previous_pair["minimum_side_mean_margin"])
                ),
                "distribution_order_gap_delta": (
                    float(current_pair["mean_distribution_order_gap"])
                    - float(previous_pair["mean_distribution_order_gap"])
                ),
            }

        preservation = {
            "old_balanced_accuracy_delta": (
                float(old_test["balanced_accuracy"])
                - float(previous_old["balanced_accuracy"])
            ),
            "old_cell_coverage_delta": (
                float(old_test["macro_true_cell_coverage"])
                - float(previous_old["macro_true_cell_coverage"])
            ),
            "old_pair_violation_delta": (
                float(old_test["macro_true_pair_violation_rate"])
                - float(previous_old["macro_true_pair_violation_rate"])
            ),
            "old_no_cell_rate_delta": (
                float(old_test["macro_no_cell_rate"])
                - float(previous_old["macro_no_cell_rate"])
            ),
            "old_rival_invasion_delta": (
                float(old_test["macro_rival_cell_invasion_rate"])
                - float(previous_old["macro_rival_cell_invasion_rate"])
            ),
            "old_decision_margin_delta": (
                float(old_test["macro_mean_decision_margin"])
                - float(previous_old["macro_mean_decision_margin"])
            ),
            "selected_replay_historical_response_drift": dict(
                result["historical_response_preservation"]
            ),
            "old_pairwise_boundary_delta": old_pairwise_delta,
        }

        persistent = {
            "phase": phase,
            "old_class_ids": old_ids,
            "new_class_ids": new_ids,
            "seen_class_ids": seen_ids,
            "final_epoch": int(result["final_epoch"]),
            "optimization_history": result.get("history", []),
            "final_epoch_report": result["final_epoch_report"],
            "current_train_geometry": result["current_train_geometry"],
            "current_validation_geometry": result["current_validation_geometry"],
            "old_replay_seen_geometry": result["old_replay_seen_geometry"],
            "old_replay_old_geometry": result["old_replay_old_geometry"],
            "replay_start_geometry": result.get("replay_start_geometry"),
            "pre_phase_historical_test_reference": pre_phase_test,
            "historical_response_preservation": result[
                "historical_response_preservation"
            ],
            "replay_diagnostics": result["replay_diagnostics"],
            "geometry_test": cumulative_test,
            "old_test": old_test,
            "new_test": new_test,
            "harmonic_old_new_accuracy": harmonic,
            "boundary_preservation": preservation,
            "geometry_state": geometry_state,
            "spectral_variation_state": self.spectral_variation_bank.summary(),
            "spectral_replay_state": self.spectral_variation_bank.summary(),
            "phase_summary": result.get("phase_summary", {}),
            "pair_count_contract": {
                "old_committed_pairs": len(self._canonical_pair_set(old_ids)),
                "candidate_pairs": len(expected_candidate_pair_set),
                "seen_committed_pairs": len(expected_seen_pair_set),
            },
        }
        self.history[phase] = persistent

        phase_dir = os.path.join(self.save_dir, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)
        checkpoint_path = self.save_checkpoint(
            os.path.join(phase_dir, "checkpoint.pth")
        )
        report_path = self.save_json(
            os.path.join(phase_dir, "incremental_geometry_report.json"),
            persistent,
        )

        result.update(
            {
                "geometry_test": cumulative_test,
                "old_test": old_test,
                "new_test": new_test,
                "harmonic_old_new_accuracy": harmonic,
                "boundary_preservation": preservation,
                "geometry_state": geometry_state,
                "spectral_variation_state": self.spectral_variation_bank.summary(),
                "spectral_replay_state": self.spectral_variation_bank.summary(),
                "checkpoint": checkpoint_path,
                "report": report_path,
                "phase_summary": persistent,
            }
        )
        return result

    def train_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
        **_: Any,
    ) -> Dict[str, Any]:
        phase = int(phase)
        if phase == 0:
            return self.run_base_only(
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
            )
        return self.run_incremental_phase(
            phase=phase,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
        )

    def run_remaining_phases(self) -> Dict[int, Dict[str, Any]]:
        """Run every phase not yet finalized."""
        finalized = self._finalized_phases()
        if finalized and finalized != list(range(finalized[-1] + 1)):
            raise RuntimeError(
                "finalized phases must form a contiguous prefix"
            )

        results: Dict[int, Dict[str, Any]] = {}
        next_phase = len(finalized)
        if next_phase == 0:
            results[0] = self.run_base_only(
                epochs=int(self.cfg("epochs_base", 100)),
                batch_size=int(self.cfg("batch_size", 64)),
                lr=float(self.cfg("lr", 1e-4)),
            )
            next_phase = 1

        for phase in range(next_phase, len(self.phase_schedule)):
            results[phase] = self.run_incremental_phase(
                phase=phase,
                epochs=int(self.cfg("epochs_inc", 15)),
                batch_size=int(self.cfg("batch_size", 64)),
                lr=float(self.cfg("lr_inc", self.cfg("lr", 1e-4))),
            )
        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        finalized = self._finalized_phases()
        if not finalized:
            raise RuntimeError("checkpointing requires a finalized phase")
        phase = finalized[-1]
        self._assert_finalized_state(phase)
        expected_history = set(range(phase + 1))
        if set(self.history) != expected_history:
            raise RuntimeError(
                "history must contain every finalized phase before checkpointing"
            )

        destination = os.path.abspath(path)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        payload = {
            "contract_version": _CHECKPOINT_VERSION,
            "phase": int(phase),
            "model": self.model.state_dict(),
            "history": self.history,
            "base_class_ids": self.base_class_ids,
            "phase_schedule": self.phase_schedule,
            "model_contract": self.assert_model_contract(),
            "geometry_state": self.geometry_state_summary(),
            "spectral_variation_state": self.spectral_variation_bank.state_dict(),
            "replay_preprocessor_state": self._replay_preprocessor_state(),
            "dataset_identity": self._dataset_identity(),
            "dataset_state": {
                "finalized_phases": finalized,
                "current_phase": getattr(self.dataset, "current_phase", None),
            },
        }

        temporary = destination + ".tmp"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise
        return destination

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        source = os.path.abspath(path)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)

        try:
            payload = torch.load(
                source,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(source, map_location=self.device)

        if not isinstance(payload, Mapping):
            raise RuntimeError("checkpoint payload must be a mapping")
        if int(payload.get("contract_version", -1)) != _CHECKPOINT_VERSION:
            raise RuntimeError(
                "checkpoint uses an incompatible trainer contract; legacy "
                "mean/variance or cell-fit checkpoints are intentionally not loaded"
            )

        phase = int(payload.get("phase", -1))
        if phase < 0 or phase not in self.phase_schedule:
            raise RuntimeError("checkpoint phase is invalid")

        saved_ids = [
            int(value) for value in payload.get("base_class_ids", [])
        ]
        if saved_ids != self.base_class_ids:
            raise RuntimeError(
                "checkpoint base classes differ from current schedule"
            )

        saved_schedule_raw = payload.get("phase_schedule")
        if not isinstance(saved_schedule_raw, Mapping):
            raise RuntimeError("checkpoint phase schedule must be a mapping")
        saved_schedule = {
            int(current): [int(value) for value in classes]
            for current, classes in saved_schedule_raw.items()
        }
        if saved_schedule != self.phase_schedule:
            raise RuntimeError(
                "checkpoint phase schedule differs from current dataset"
            )

        saved_dataset_identity = payload.get("dataset_identity")
        if not isinstance(saved_dataset_identity, Mapping):
            raise RuntimeError("checkpoint lacks a valid dataset identity")
        if dict(saved_dataset_identity) != self._dataset_identity():
            raise RuntimeError(
                "checkpoint dataset protocol differs from current dataset"
            )

        saved_contract = payload.get("model_contract")
        if not isinstance(saved_contract, Mapping):
            raise RuntimeError("checkpoint lacks a valid model contract")
        current_contract = self.assert_model_contract()
        structural_fields = (
            "patch_bands",
            "spectral_bands",
            "patch_size",
            "representation_dim",
            "context_input_channels",
            "context_spectral_dim",
            "classifier_parameter_count",
        )
        for name in structural_fields:
            if int(saved_contract.get(name, -1)) != int(current_contract[name]):
                raise RuntimeError(
                    f"checkpoint model structure differs at {name}: "
                    f"saved={saved_contract.get(name)}, current={current_contract[name]}"
                )

        saved_preprocessor = payload.get("replay_preprocessor_state")
        if not isinstance(saved_preprocessor, Mapping):
            raise RuntimeError(
                "checkpoint lacks spectral replay preprocessing state"
            )
        if not self._same_preprocessor_state(
            saved_preprocessor,
            self._replay_preprocessor_state(),
        ):
            raise RuntimeError(
                "checkpoint replay preprocessing differs from current base preprocessing"
            )

        saved_history = payload.get("history")
        if not isinstance(saved_history, Mapping):
            raise RuntimeError("checkpoint history must be a mapping")
        restored_history = {
            int(current): dict(value)
            for current, value in saved_history.items()
        }
        expected_history = set(range(phase + 1))
        if set(restored_history) != expected_history:
            raise RuntimeError(
                "checkpoint history does not match its finalized phase"
            )

        dataset_state_raw = payload.get("dataset_state")
        if not isinstance(dataset_state_raw, Mapping):
            raise RuntimeError("checkpoint dataset_state must be a mapping")
        finalized = [
            int(value)
            for value in dataset_state_raw.get("finalized_phases", [])
        ]
        expected_finalized = list(range(phase + 1))
        if finalized != expected_finalized:
            raise RuntimeError(
                "checkpoint finalized phases are inconsistent"
            )
        if dataset_state_raw.get("current_phase") is not None:
            raise RuntimeError(
                "a finalized checkpoint cannot contain an active phase"
            )
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError(
                "cannot restore while the dataset has an active phase"
            )

        current_finalized = self._finalized_phases()
        if current_finalized and current_finalized != expected_finalized[:len(current_finalized)]:
            raise RuntimeError(
                "current dataset finalized state is incompatible with checkpoint"
            )

        model_state = payload.get("model")
        if not isinstance(model_state, Mapping):
            raise RuntimeError("checkpoint model state must be a mapping")
        variation_state = payload.get("spectral_variation_state")
        if not isinstance(variation_state, Mapping):
            raise RuntimeError("checkpoint lacks spectral variation state")

        self.model.load_state_dict(model_state, strict=True)
        self.model.to(self.device)
        self.model.validate_model_state()

        self.spectral_variation_bank.load_state_dict(variation_state)
        seen = self._seen_classes(phase)
        if [int(value) for value in self.model.committed_class_ids] != seen:
            raise RuntimeError(
                "loaded geometry does not match checkpoint phase schedule"
            )
        if [
            int(value)
            for value in self.spectral_variation_bank.class_ids.tolist()
        ] != seen:
            raise RuntimeError(
                "loaded spectral variation state does not match checkpoint phase schedule"
            )

        self.history = restored_history
        self.dataset.finalized_phases = list(finalized)
        self.dataset.current_phase = None

        self._assert_finalized_state(phase)
        self.model.eval()
        return dict(payload)


__all__ = ["Trainer"]



















# from __future__ import annotations

# """Phase orchestration for HSI pairwise decision geometry.

# Persistent continual state consists of:

# 1. the model/backbone and committed pairwise decision geometry;
# 2. compact REAL-data-derived ordered spectral variation for every finalized class.

# Replay samples themselves are never checkpointed.  They are reconstructed
# transiently at the start of an incremental phase and selected by the candidate
# old-new decision geometry.
# """

# import hashlib
# import os
# from typing import Any, Dict, Mapping

# import numpy as np
# import torch

# from data.hsi_dataloader_pytorch import LoadHSIPreprocessor
# from models.spectral_replay import (
#     FrozenHSIPreprocessor,
#     SpectralVariationBank,
# )
# from trainers.base_phase_trainer import BasePhaseTrainer
# from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
# from trainers.trainer_helpers import TrainerHelper


# _CHECKPOINT_VERSION = 2


# class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
#     """Own phase lifecycle, persistent geometry, and HSI variation state."""

#     def __init__(
#         self,
#         model: torch.nn.Module,
#         dataset: Any,
#         args: Any,
#     ) -> None:
#         self.model = model
#         self.dataset = dataset
#         self.args = args

#         configured_device = getattr(args, "device", None)
#         requested = (
#             str(configured_device)
#             if configured_device is not None
#             else ("cuda:0" if torch.cuda.is_available() else "cpu")
#         )
#         try:
#             self.device = torch.device(requested)
#         except (TypeError, RuntimeError) as exc:
#             raise ValueError(f"invalid training device {requested!r}") from exc

#         if self.device.type == "cuda":
#             if not torch.cuda.is_available():
#                 raise RuntimeError(
#                     f"CUDA device {self.device} was requested but CUDA is unavailable"
#                 )
#             index = 0 if self.device.index is None else self.device.index
#             if not 0 <= index < torch.cuda.device_count():
#                 raise RuntimeError(
#                     f"CUDA device index {index} is unavailable; "
#                     f"device_count={torch.cuda.device_count()}"
#                 )

#         self.model.to(self.device)
#         self.save_dir = os.path.abspath(
#             str(getattr(args, "save_dir", "./outputs"))
#         )
#         os.makedirs(self.save_dir, exist_ok=True)

#         dataset_contract = self.assert_dataset_contract()
#         self.assert_model_contract()

#         self.phase_schedule = {
#             int(phase): [int(value) for value in classes]
#             for phase, classes in dataset_contract["schedule"].items()
#         }
#         self.base_class_ids = [
#             int(value) for value in dataset_contract["base_classes"]
#         ]
#         self.history: Dict[int, Dict[str, Any]] = {}

#         # Phase-0 preprocessing is immutable across the continual sequence.
#         preprocessor_path = os.path.join(
#             self.save_dir, "phase_0_preprocessor.npz"
#         )
#         if not os.path.isfile(preprocessor_path):
#             raise FileNotFoundError(
#                 "spectral variation replay requires the frozen phase-0 preprocessing "
#                 f"state: {preprocessor_path}"
#             )
#         preprocessing_state = LoadHSIPreprocessor(preprocessor_path)
#         self.replay_preprocessor = FrozenHSIPreprocessor(
#             preprocessing_state
#         ).to(self.device)

#         spectral_bands = int(self.model.backbone.spectral_bands)
#         if int(self.replay_preprocessor.raw_bands) != spectral_bands:
#             raise RuntimeError(
#                 "replay preprocessor and backbone ordered-band counts disagree"
#             )
#         if int(self.replay_preprocessor.processed_bands) != int(
#             self.model.backbone.patch_bands
#         ):
#             raise RuntimeError(
#                 "replay preprocessor and backbone processed-band counts disagree"
#             )

#         self.spectral_variation_bank = SpectralVariationBank(spectral_bands)
#         # Temporary compatibility alias for code outside this trainer.  Both
#         # names reference the same object; new code should use variation_bank.
#         self.spectral_replay_bank = self.spectral_variation_bank

#     # ------------------------------------------------------------------
#     # Dataset / checkpoint identity
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _array_fingerprint(value: Any) -> str:
#         array = np.ascontiguousarray(np.asarray(value))
#         digest = hashlib.sha256()
#         digest.update(array.dtype.str.encode("ascii"))
#         digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
#         digest.update(memoryview(array).cast("B"))
#         return digest.hexdigest()

#     def _split_fingerprint(self) -> str:
#         split_by_class = getattr(self.dataset, "_split_by_class", None)
#         if not isinstance(split_by_class, Mapping):
#             return "unavailable"

#         digest = hashlib.sha256()
#         for class_id in sorted(int(value) for value in split_by_class):
#             row = split_by_class[class_id]
#             if not isinstance(row, Mapping):
#                 raise RuntimeError("dataset class split must be a mapping")
#             digest.update(np.asarray([class_id], dtype=np.int64).tobytes())
#             for split in ("train", "val", "test"):
#                 if split not in row:
#                     raise RuntimeError(
#                         f"dataset class {class_id} lacks {split} split"
#                     )
#                 values = np.ascontiguousarray(
#                     np.asarray(row[split], dtype=np.int64).reshape(-1)
#                 )
#                 digest.update(split.encode("ascii"))
#                 digest.update(
#                     np.asarray([values.size], dtype=np.int64).tobytes()
#                 )
#                 digest.update(memoryview(values).cast("B"))
#         return digest.hexdigest()

#     def _dataset_identity(self) -> Dict[str, Any]:
#         processed_cube = getattr(self.dataset, "processed_cube", None)
#         ordered_cube = getattr(self.dataset, "ordered_spectral_cube", None)
#         labels = getattr(self.dataset, "labels", None)
#         coords = getattr(self.dataset, "coords", None)
#         if processed_cube is None or ordered_cube is None:
#             raise RuntimeError(
#                 "dataset must expose processed and ordered spectral cubes"
#             )
#         if labels is None or coords is None:
#             raise RuntimeError("dataset must expose labels and coordinates")

#         return {
#             "seed": int(
#                 getattr(
#                     self.dataset,
#                     "seed",
#                     getattr(self.args, "seed", 0),
#                 )
#             ),
#             "phase_schedule": self.phase_schedule,
#             "class_order_original_ids": [
#                 int(value)
#                 for value in getattr(
#                     self.dataset,
#                     "class_order_original_ids",
#                     sorted({
#                         class_id
#                         for ids in self.phase_schedule.values()
#                         for class_id in ids
#                     }),
#                 )
#             ],
#             "split_strategy": str(
#                 getattr(
#                     self.dataset,
#                     "split_strategy",
#                     getattr(self.args, "split_strategy", ""),
#                 )
#             ),
#             "spatial_partition_mode": str(
#                 getattr(self.dataset, "spatial_partition_mode", "")
#             ),
#             "context_policy": str(
#                 getattr(self.dataset, "context_policy", "")
#             ),
#             "patch_size": int(getattr(self.dataset, "patch_size")),
#             "train_ratio": (
#                 None
#                 if getattr(self.args, "train_ratio", None) is None
#                 else float(getattr(self.args, "train_ratio"))
#             ),
#             "val_ratio": (
#                 None
#                 if getattr(self.args, "val_ratio", None) is None
#                 else float(getattr(self.args, "val_ratio"))
#             ),
#             "require_patch_disjoint": bool(
#                 getattr(
#                     self.dataset,
#                     "require_patch_disjoint",
#                     getattr(self.args, "require_patch_disjoint", False),
#                 )
#             ),
#             "pca_components": int(
#                 getattr(
#                     self.args,
#                     "pca_components",
#                     processed_cube.shape[2],
#                 )
#             ),
#             "pca_whiten": bool(getattr(self.args, "pca_whiten", False)),
#             "processed_cube_shape": [
#                 int(value) for value in processed_cube.shape
#             ],
#             "ordered_spectral_cube_shape": [
#                 int(value) for value in ordered_cube.shape
#             ],
#             "processed_cube_sha256": self._array_fingerprint(processed_cube),
#             "ordered_spectral_cube_sha256": self._array_fingerprint(ordered_cube),
#             "labels_sha256": self._array_fingerprint(labels),
#             "coordinates_sha256": self._array_fingerprint(coords),
#             "split_membership_sha256": self._split_fingerprint(),
#         }

#     def _replay_preprocessor_state(self) -> Dict[str, Any]:
#         preprocessor = self.replay_preprocessor
#         return {
#             "raw_bands": int(preprocessor.raw_bands),
#             "fit_pixel_count": int(preprocessor.fit_pixel_count),
#             "normalization_mean": (
#                 preprocessor.normalization_mean.detach().cpu().clone()
#             ),
#             "normalization_std": (
#                 preprocessor.normalization_std.detach().cpu().clone()
#             ),
#             "pca_components": (
#                 preprocessor.pca_components.detach().cpu().clone()
#             ),
#             "pca_mean": preprocessor.pca_mean.detach().cpu().clone(),
#             "pca_variance": (
#                 preprocessor.pca_variance.detach().cpu().clone()
#             ),
#             "whiten": bool(preprocessor.whiten),
#             "fit_scope": str(preprocessor.fit_scope),
#         }

#     @staticmethod
#     def _same_preprocessor_state(
#         left: Mapping[str, Any],
#         right: Mapping[str, Any],
#     ) -> bool:
#         scalar_keys = (
#             "raw_bands",
#             "fit_pixel_count",
#             "whiten",
#             "fit_scope",
#         )
#         for key in scalar_keys:
#             if left.get(key) != right.get(key):
#                 return False

#         tensor_keys = (
#             "normalization_mean",
#             "normalization_std",
#             "pca_components",
#             "pca_mean",
#             "pca_variance",
#         )
#         for key in tensor_keys:
#             if key not in left or key not in right:
#                 return False
#             a = torch.as_tensor(left[key]).cpu()
#             b = torch.as_tensor(right[key]).cpu()
#             if a.shape != b.shape or a.dtype != b.dtype:
#                 return False
#             if not torch.equal(a, b):
#                 return False
#         return True

#     # ------------------------------------------------------------------
#     # Finalized-state contracts
#     # ------------------------------------------------------------------

#     def _finalized_phases(self) -> list[int]:
#         raw = getattr(self.dataset, "finalized_phases", None)
#         if raw is None:
#             raise RuntimeError("dataset must expose finalized_phases")
#         phases = [int(value) for value in raw]
#         if (
#             phases != sorted(set(phases))
#             or any(value < 0 for value in phases)
#         ):
#             raise RuntimeError("dataset finalized_phases is invalid")
#         return phases

#     def _seen_classes(self, phase: int) -> list[int]:
#         phase = int(phase)
#         if phase not in self.phase_schedule:
#             raise ValueError(f"unknown phase {phase}")
#         result: list[int] = []
#         for current in range(phase + 1):
#             result.extend(self.phase_schedule[current])
#         return result

#     def _assert_spectral_normalization_ready(self) -> None:
#         if not bool(
#             getattr(self.model, "spectral_normalization_fitted", False)
#         ):
#             raise RuntimeError(
#                 "ordered-spectrum normalization is not fitted from base training data"
#             )

#     def _assert_finalized_state(self, phase: int) -> None:
#         phase = int(phase)
#         expected_phases = list(range(phase + 1))
#         if self._finalized_phases() != expected_phases:
#             raise RuntimeError(f"expected finalized phases {expected_phases}")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError(
#                 "a finalized continual state cannot retain an active phase"
#             )

#         seen = self._seen_classes(phase)
#         committed = [int(value) for value in self.model.committed_class_ids]
#         if committed != seen:
#             raise RuntimeError(
#                 "committed geometry does not match the finalized schedule: "
#                 f"bank={committed}, schedule={seen}"
#             )

#         variation_ids = [
#             int(value)
#             for value in self.spectral_variation_bank.class_ids.tolist()
#         ]
#         if variation_ids != seen:
#             raise RuntimeError(
#                 "spectral variation state does not match the finalized schedule: "
#                 f"variation={variation_ids}, schedule={seen}"
#             )
#         self.spectral_variation_bank.validate_state()
#         self._assert_spectral_normalization_ready()
#         self.model.validate_model_state()

#     # ------------------------------------------------------------------
#     # Base phase
#     # ------------------------------------------------------------------

#     def run_base_only(
#         self,
#         *,
#         epochs: int,
#         batch_size: int = 64,
#         lr: float = 1e-4,
#     ) -> Dict[str, Any]:
#         if self._finalized_phases():
#             raise RuntimeError("base phase requires an unfinalized dataset")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")
#         if len(self.model.geometry_bank) != 0:
#             raise RuntimeError("base phase requires an empty geometry bank")
#         if len(self.spectral_variation_bank) != 0:
#             raise RuntimeError(
#                 "base phase requires an empty spectral variation bank"
#             )

#         result = dict(
#             self.train_base_phase(
#                 phase=0,
#                 epochs=int(epochs),
#                 batch_size=int(batch_size),
#                 lr=float(lr),
#             )
#         )
#         required = {
#             "class_ids",
#             "final_epoch",
#             "final_epoch_report",
#             "geometry_train",
#             "geometry_validation",
#             "geometry_committed",
#         }
#         missing = required - set(result)
#         if missing:
#             raise RuntimeError(
#                 "BasePhaseTrainer returned an incomplete result: "
#                 f"{sorted(missing)}"
#             )

#         class_ids = [int(value) for value in result["class_ids"]]
#         if class_ids != self.base_class_ids:
#             raise RuntimeError(
#                 "base trainer class IDs disagree with dataset schedule"
#             )
#         if result["geometry_committed"] is not True:
#             raise RuntimeError(
#                 "BasePhaseTrainer did not commit the learned base boundaries"
#             )
#         if [int(value) for value in self.model.committed_class_ids] != class_ids:
#             raise RuntimeError(
#                 "committed class IDs disagree with base trainer output"
#             )

#         evaluation_batch_size = int(self.cfg("eval_batch_size", 256))
#         if evaluation_batch_size <= 0:
#             raise ValueError("eval_batch_size must be positive")
#         base_train_loader = self.dataset.get_phase_dataloader(
#             0,
#             split="train",
#             batch_size=evaluation_batch_size,
#             shuffle=False,
#         )
#         self.spectral_variation_bank.append_from_loader(
#             base_train_loader,
#             class_ids=class_ids,
#         )
#         if [
#             int(value)
#             for value in self.spectral_variation_bank.class_ids.tolist()
#         ] != class_ids:
#             raise RuntimeError("base spectral variation state is incomplete")

#         self._assert_spectral_normalization_ready()
#         self.model.validate_model_state()
#         geometry_state = self.geometry_state_summary()

#         self.dataset.finalize_phase(0)
#         self._assert_finalized_state(0)

#         test_loader = self.dataset.get_cumulative_dataloader(
#             0,
#             split="test",
#             batch_size=evaluation_batch_size,
#             shuffle=False,
#         )
#         geometry_test = self.evaluate_loader(
#             test_loader,
#             class_ids=class_ids,
#             candidate=None,
#         )

#         persistent = {
#             "phase": 0,
#             "class_ids": class_ids,
#             "final_epoch": int(result["final_epoch"]),
#             "optimization_history": result.get("history", []),
#             "final_epoch_report": result["final_epoch_report"],
#             "geometry_train": result["geometry_train"],
#             "geometry_validation": result["geometry_validation"],
#             "geometry_test": geometry_test,
#             "geometry_summary": result.get("geometry_summary", {}),
#             "geometry_state": geometry_state,
#             "spectral_variation_state": self.spectral_variation_bank.summary(),
#             # Reporting compatibility only.
#             "spectral_replay_state": self.spectral_variation_bank.summary(),
#             "runtime_scope": "continual",
#             "incremental_enabled": True,
#         }
#         self.history[0] = persistent

#         phase_dir = os.path.join(self.save_dir, "phase_0")
#         os.makedirs(phase_dir, exist_ok=True)
#         checkpoint_path = self.save_checkpoint(
#             os.path.join(phase_dir, "checkpoint.pth")
#         )
#         report_path = self.save_json(
#             os.path.join(phase_dir, "base_geometry_report.json"),
#             persistent,
#         )

#         result.update(
#             {
#                 "geometry_test": geometry_test,
#                 "geometry_state": geometry_state,
#                 "spectral_variation_state": self.spectral_variation_bank.summary(),
#                 "spectral_replay_state": self.spectral_variation_bank.summary(),
#                 "checkpoint": checkpoint_path,
#                 "report": report_path,
#                 "phase_summary": persistent,
#                 "runtime_scope": "continual",
#                 "incremental_enabled": True,
#             }
#         )
#         return result

#     # ------------------------------------------------------------------
#     # Incremental phase lifecycle
#     # ------------------------------------------------------------------

#     def run_incremental_phase(
#         self,
#         *,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#     ) -> Dict[str, Any]:
#         phase = int(phase)
#         if phase <= 0 or phase not in self.phase_schedule:
#             raise ValueError(
#                 "run_incremental_phase requires a valid phase > 0"
#             )
#         if self._finalized_phases() != list(range(phase)):
#             raise RuntimeError(
#                 f"phase {phase} requires finalized prefix {list(range(phase))}"
#             )
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")
#         self._assert_finalized_state(phase - 1)

#         result = dict(
#             self.train_incremental_phase(
#                 phase=phase,
#                 epochs=int(epochs),
#                 batch_size=int(batch_size),
#                 lr=float(lr),
#             )
#         )
#         required = {
#             "old_class_ids",
#             "new_class_ids",
#             "seen_class_ids",
#             "candidate_geometry",
#             "current_train_geometry",
#             "current_validation_geometry",
#             "old_replay_seen_geometry",
#             "old_replay_old_geometry",
#             "historical_response_preservation",
#             "replay_diagnostics",
#             "final_epoch",
#             "final_epoch_report",
#         }
#         missing = required - set(result)
#         if missing:
#             raise RuntimeError(
#                 "IncrementalPhaseTrainer returned an incomplete result: "
#                 f"{sorted(missing)}"
#             )

#         old_ids = [int(value) for value in result["old_class_ids"]]
#         new_ids = [int(value) for value in result["new_class_ids"]]
#         seen_ids = [int(value) for value in result["seen_class_ids"]]
#         if old_ids != self._seen_classes(phase - 1):
#             raise RuntimeError(
#                 "incremental trainer historical classes disagree with finalized state"
#             )
#         if new_ids != self.phase_schedule[phase]:
#             raise RuntimeError(
#                 "incremental trainer current classes disagree with schedule"
#             )
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError(
#                 "incremental trainer seen-class order is invalid"
#             )

#         candidate = result["candidate_geometry"]
#         if not hasattr(candidate, "new_class_ids"):
#             raise RuntimeError("incremental trainer returned invalid candidate geometry")
#         if candidate.new_class_ids != tuple(new_ids):
#             raise RuntimeError("incremental candidate class IDs are invalid")
#         candidate.validate_state()

#         evaluation_batch_size = int(self.cfg("eval_batch_size", 256))
#         if evaluation_batch_size <= 0:
#             raise ValueError("eval_batch_size must be positive")
#         spectral_loader = self.dataset.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=evaluation_batch_size,
#             shuffle=False,
#         )

#         # Pre-validate new spectral variation without mutating persistent state.
#         # This prevents a malformed real phase from being discovered only after
#         # the geometry candidate has been committed.
#         validation_bank = SpectralVariationBank(
#             int(self.model.backbone.spectral_bands)
#         )
#         validation_bank.append_from_loader(
#             spectral_loader,
#             class_ids=new_ids,
#         )
#         validation_bank.validate_state()

#         # Commit the exact final candidate, then append only REAL current-phase
#         # spectral variation.  Pseudo replay never updates historical rows.
#         self.model.commit_candidate(candidate)
#         self.model.eval()
#         self.model.validate_model_state()
#         self.spectral_variation_bank.append_from_loader(
#             spectral_loader,
#             class_ids=new_ids,
#         )

#         if [int(value) for value in self.model.committed_class_ids] != seen_ids:
#             raise RuntimeError(
#                 "geometry commit did not produce the complete seen-class state"
#             )
#         if [
#             int(value)
#             for value in self.spectral_variation_bank.class_ids.tolist()
#         ] != seen_ids:
#             raise RuntimeError(
#                 "spectral variation state did not append exactly the new classes"
#             )

#         geometry_state = self.geometry_state_summary()
#         self.dataset.finalize_phase(phase)
#         self._assert_finalized_state(phase)

#         test_loader = self.dataset.get_cumulative_dataloader(
#             phase,
#             split="test",
#             batch_size=evaluation_batch_size,
#             shuffle=False,
#         )
#         cumulative_test = self.evaluate_loader(
#             test_loader,
#             class_ids=seen_ids,
#             candidate=None,
#         )

#         old_test = self.summarize_class_group(cumulative_test, old_ids)
#         new_test = self.summarize_class_group(cumulative_test, new_ids)
#         denominator = (
#             float(old_test["balanced_accuracy"])
#             + float(new_test["balanced_accuracy"])
#         )
#         harmonic = (
#             0.0
#             if denominator == 0.0
#             else (
#                 2.0
#                 * float(old_test["balanced_accuracy"])
#                 * float(new_test["balanced_accuracy"])
#                 / denominator
#             )
#         )

#         previous_test = self.history[phase - 1]["geometry_test"]
#         previous_old = self.summarize_class_group(previous_test, old_ids)
#         preservation = {
#             "old_balanced_accuracy_delta": (
#                 float(old_test["balanced_accuracy"])
#                 - float(previous_old["balanced_accuracy"])
#             ),
#             "old_cell_coverage_delta": (
#                 float(old_test["macro_true_cell_coverage"])
#                 - float(previous_old["macro_true_cell_coverage"])
#             ),
#             "old_pair_violation_delta": (
#                 float(old_test["macro_true_pair_violation_rate"])
#                 - float(previous_old["macro_true_pair_violation_rate"])
#             ),
#             "old_no_cell_rate_delta": (
#                 float(old_test["macro_no_cell_rate"])
#                 - float(previous_old["macro_no_cell_rate"])
#             ),
#             "old_rival_invasion_delta": (
#                 float(old_test["macro_rival_cell_invasion_rate"])
#                 - float(previous_old["macro_rival_cell_invasion_rate"])
#             ),
#             "old_decision_margin_delta": (
#                 float(old_test["macro_mean_decision_margin"])
#                 - float(previous_old["macro_mean_decision_margin"])
#             ),
#             "selected_replay_historical_response_drift": dict(
#                 result["historical_response_preservation"]
#             ),
#         }

#         persistent = {
#             "phase": phase,
#             "old_class_ids": old_ids,
#             "new_class_ids": new_ids,
#             "seen_class_ids": seen_ids,
#             "final_epoch": int(result["final_epoch"]),
#             "optimization_history": result.get("history", []),
#             "final_epoch_report": result["final_epoch_report"],
#             "current_train_geometry": result["current_train_geometry"],
#             "current_validation_geometry": result["current_validation_geometry"],
#             "old_replay_seen_geometry": result["old_replay_seen_geometry"],
#             "old_replay_old_geometry": result["old_replay_old_geometry"],
#             "replay_start_geometry": result.get("replay_start_geometry"),
#             "historical_response_preservation": result[
#                 "historical_response_preservation"
#             ],
#             "replay_diagnostics": result["replay_diagnostics"],
#             "geometry_test": cumulative_test,
#             "old_test": old_test,
#             "new_test": new_test,
#             "harmonic_old_new_accuracy": harmonic,
#             "boundary_preservation": preservation,
#             "geometry_state": geometry_state,
#             "spectral_variation_state": self.spectral_variation_bank.summary(),
#             "spectral_replay_state": self.spectral_variation_bank.summary(),
#             "phase_summary": result.get("phase_summary", {}),
#         }
#         self.history[phase] = persistent

#         phase_dir = os.path.join(self.save_dir, f"phase_{phase}")
#         os.makedirs(phase_dir, exist_ok=True)
#         checkpoint_path = self.save_checkpoint(
#             os.path.join(phase_dir, "checkpoint.pth")
#         )
#         report_path = self.save_json(
#             os.path.join(phase_dir, "incremental_geometry_report.json"),
#             persistent,
#         )

#         result.update(
#             {
#                 "geometry_test": cumulative_test,
#                 "old_test": old_test,
#                 "new_test": new_test,
#                 "harmonic_old_new_accuracy": harmonic,
#                 "boundary_preservation": preservation,
#                 "geometry_state": geometry_state,
#                 "spectral_variation_state": self.spectral_variation_bank.summary(),
#                 "spectral_replay_state": self.spectral_variation_bank.summary(),
#                 "checkpoint": checkpoint_path,
#                 "report": report_path,
#                 "phase_summary": persistent,
#             }
#         )
#         return result

#     def train_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#         **_: Any,
#     ) -> Dict[str, Any]:
#         phase = int(phase)
#         if phase == 0:
#             return self.run_base_only(
#                 epochs=int(epochs),
#                 batch_size=int(batch_size),
#                 lr=float(lr),
#             )
#         return self.run_incremental_phase(
#             phase=phase,
#             epochs=int(epochs),
#             batch_size=int(batch_size),
#             lr=float(lr),
#         )

#     def run_remaining_phases(self) -> Dict[int, Dict[str, Any]]:
#         """Run every phase not yet finalized."""
#         finalized = self._finalized_phases()
#         if finalized and finalized != list(range(finalized[-1] + 1)):
#             raise RuntimeError(
#                 "finalized phases must form a contiguous prefix"
#             )

#         results: Dict[int, Dict[str, Any]] = {}
#         next_phase = len(finalized)
#         if next_phase == 0:
#             results[0] = self.run_base_only(
#                 epochs=int(self.cfg("epochs_base", 100)),
#                 batch_size=int(self.cfg("batch_size", 64)),
#                 lr=float(self.cfg("lr", 1e-4)),
#             )
#             next_phase = 1

#         for phase in range(next_phase, len(self.phase_schedule)):
#             results[phase] = self.run_incremental_phase(
#                 phase=phase,
#                 epochs=int(self.cfg("epochs_inc", 15)),
#                 batch_size=int(self.cfg("batch_size", 64)),
#                 lr=float(self.cfg("lr_inc", self.cfg("lr", 1e-4))),
#             )
#         return results

#     # ------------------------------------------------------------------
#     # Checkpointing
#     # ------------------------------------------------------------------

#     def save_checkpoint(self, path: str) -> str:
#         finalized = self._finalized_phases()
#         if not finalized:
#             raise RuntimeError("checkpointing requires a finalized phase")
#         phase = finalized[-1]
#         self._assert_finalized_state(phase)
#         expected_history = set(range(phase + 1))
#         if set(self.history) != expected_history:
#             raise RuntimeError(
#                 "history must contain every finalized phase before checkpointing"
#             )

#         destination = os.path.abspath(path)
#         os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#         payload = {
#             "contract_version": _CHECKPOINT_VERSION,
#             "phase": int(phase),
#             "model": self.model.state_dict(),
#             "history": self.history,
#             "base_class_ids": self.base_class_ids,
#             "phase_schedule": self.phase_schedule,
#             "model_contract": self.assert_model_contract(),
#             "geometry_state": self.geometry_state_summary(),
#             "spectral_variation_state": self.spectral_variation_bank.state_dict(),
#             "replay_preprocessor_state": self._replay_preprocessor_state(),
#             "dataset_identity": self._dataset_identity(),
#             "dataset_state": {
#                 "finalized_phases": finalized,
#                 "current_phase": getattr(self.dataset, "current_phase", None),
#             },
#         }

#         temporary = destination + ".tmp"
#         try:
#             torch.save(payload, temporary)
#             os.replace(temporary, destination)
#         except Exception:
#             if os.path.exists(temporary):
#                 os.remove(temporary)
#             raise
#         return destination

#     def load_checkpoint(self, path: str) -> Dict[str, Any]:
#         source = os.path.abspath(path)
#         if not os.path.isfile(source):
#             raise FileNotFoundError(source)

#         try:
#             payload = torch.load(
#                 source,
#                 map_location=self.device,
#                 weights_only=True,
#             )
#         except TypeError:
#             payload = torch.load(source, map_location=self.device)

#         if not isinstance(payload, Mapping):
#             raise RuntimeError("checkpoint payload must be a mapping")
#         if int(payload.get("contract_version", -1)) != _CHECKPOINT_VERSION:
#             raise RuntimeError(
#                 "checkpoint uses an incompatible trainer contract; legacy "
#                 "mean/variance or cell-fit checkpoints are intentionally not loaded"
#             )

#         phase = int(payload.get("phase", -1))
#         if phase < 0 or phase not in self.phase_schedule:
#             raise RuntimeError("checkpoint phase is invalid")

#         saved_ids = [
#             int(value) for value in payload.get("base_class_ids", [])
#         ]
#         if saved_ids != self.base_class_ids:
#             raise RuntimeError(
#                 "checkpoint base classes differ from current schedule"
#             )

#         saved_schedule_raw = payload.get("phase_schedule")
#         if not isinstance(saved_schedule_raw, Mapping):
#             raise RuntimeError("checkpoint phase schedule must be a mapping")
#         saved_schedule = {
#             int(current): [int(value) for value in classes]
#             for current, classes in saved_schedule_raw.items()
#         }
#         if saved_schedule != self.phase_schedule:
#             raise RuntimeError(
#                 "checkpoint phase schedule differs from current dataset"
#             )

#         saved_dataset_identity = payload.get("dataset_identity")
#         if not isinstance(saved_dataset_identity, Mapping):
#             raise RuntimeError("checkpoint lacks a valid dataset identity")
#         if dict(saved_dataset_identity) != self._dataset_identity():
#             raise RuntimeError(
#                 "checkpoint dataset protocol differs from current dataset"
#             )

#         saved_contract = payload.get("model_contract")
#         if not isinstance(saved_contract, Mapping):
#             raise RuntimeError("checkpoint lacks a valid model contract")
#         current_contract = self.assert_model_contract()
#         structural_fields = (
#             "patch_bands",
#             "spectral_bands",
#             "patch_size",
#             "representation_dim",
#             "context_input_channels",
#             "context_spectral_dim",
#             "classifier_parameter_count",
#         )
#         for name in structural_fields:
#             if int(saved_contract.get(name, -1)) != int(current_contract[name]):
#                 raise RuntimeError(
#                     f"checkpoint model structure differs at {name}: "
#                     f"saved={saved_contract.get(name)}, current={current_contract[name]}"
#                 )

#         saved_preprocessor = payload.get("replay_preprocessor_state")
#         if not isinstance(saved_preprocessor, Mapping):
#             raise RuntimeError(
#                 "checkpoint lacks spectral replay preprocessing state"
#             )
#         if not self._same_preprocessor_state(
#             saved_preprocessor,
#             self._replay_preprocessor_state(),
#         ):
#             raise RuntimeError(
#                 "checkpoint replay preprocessing differs from current base preprocessing"
#             )

#         saved_history = payload.get("history")
#         if not isinstance(saved_history, Mapping):
#             raise RuntimeError("checkpoint history must be a mapping")
#         restored_history = {
#             int(current): dict(value)
#             for current, value in saved_history.items()
#         }
#         expected_history = set(range(phase + 1))
#         if set(restored_history) != expected_history:
#             raise RuntimeError(
#                 "checkpoint history does not match its finalized phase"
#             )

#         dataset_state_raw = payload.get("dataset_state")
#         if not isinstance(dataset_state_raw, Mapping):
#             raise RuntimeError("checkpoint dataset_state must be a mapping")
#         finalized = [
#             int(value)
#             for value in dataset_state_raw.get("finalized_phases", [])
#         ]
#         expected_finalized = list(range(phase + 1))
#         if finalized != expected_finalized:
#             raise RuntimeError(
#                 "checkpoint finalized phases are inconsistent"
#             )
#         if dataset_state_raw.get("current_phase") is not None:
#             raise RuntimeError(
#                 "a finalized checkpoint cannot contain an active phase"
#             )
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError(
#                 "cannot restore while the dataset has an active phase"
#             )

#         current_finalized = self._finalized_phases()
#         if current_finalized and current_finalized != expected_finalized[:len(current_finalized)]:
#             raise RuntimeError(
#                 "current dataset finalized state is incompatible with checkpoint"
#             )

#         model_state = payload.get("model")
#         if not isinstance(model_state, Mapping):
#             raise RuntimeError("checkpoint model state must be a mapping")
#         variation_state = payload.get("spectral_variation_state")
#         if not isinstance(variation_state, Mapping):
#             raise RuntimeError("checkpoint lacks spectral variation state")

#         self.model.load_state_dict(model_state, strict=True)
#         self.model.to(self.device)
#         self.model.validate_model_state()

#         self.spectral_variation_bank.load_state_dict(variation_state)
#         seen = self._seen_classes(phase)
#         if [int(value) for value in self.model.committed_class_ids] != seen:
#             raise RuntimeError(
#                 "loaded geometry does not match checkpoint phase schedule"
#             )
#         if [
#             int(value)
#             for value in self.spectral_variation_bank.class_ids.tolist()
#         ] != seen:
#             raise RuntimeError(
#                 "loaded spectral variation state does not match checkpoint phase schedule"
#             )

#         self.history = restored_history
#         self.dataset.finalized_phases = list(finalized)
#         self.dataset.current_phase = None

#         self._assert_finalized_state(phase)
#         self.model.eval()
#         return dict(payload)


# __all__ = ["Trainer"]
