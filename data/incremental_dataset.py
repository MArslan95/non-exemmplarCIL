"""incremental HSI dataset and protocol utilities.

This module enforces paired central spectral interventions, base-train-only
preprocessing indices, strict phase-local train access, and physical deletion of
finalized train patches/response centers. Raw spectra and one-sided augmented
patches are deliberately absent from the model-facing dataset contract.
"""


from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
    """Return unique ints while preserving order.

    CIL class order is part of the protocol. Do not replace this with
    ``sorted(set(values))`` in trainer/evaluator code; that silently changes
    shuffled class-order experiments.
    """
    out: List[int] = []
    seen: Set[int] = set()
    for value in values:
        v = int(value)
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _as_1d_int_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr



_PROTOCOL_PLAN_VERSION = 2


def _labels_sha256(labels: np.ndarray) -> str:
    labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64).reshape(-1))
    return hashlib.sha256(labels.tobytes()).hexdigest()



def _validate_split_configuration(
    *,
    train_ratio: float,
    val_ratio: float,
    min_train_per_class: int,
) -> None:
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    min_train_per_class = int(min_train_per_class)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            "train_ratio + val_ratio must be < 1 so a test partition remains; "
            f"got {train_ratio + val_ratio:.6f}."
        )
    if min_train_per_class <= 0:
        raise ValueError(
            f"min_train_per_class must be positive, got {min_train_per_class}."
        )


def _split_one_class(
    class_indices: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    min_train_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one deterministic classwise split without sample leakage."""
    indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    rng = np.random.RandomState(int(seed))
    shuffled = rng.permutation(indices)
    n_samples = int(shuffled.size)

    if n_samples == 1:
        return shuffled.copy(), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    if n_samples == 2:
        return shuffled[:1].copy(), np.empty((0,), dtype=np.int64), shuffled[1:].copy()

    requested_train = max(
        int(min_train_per_class),
        int(round(n_samples * float(train_ratio))),
        1,
    )
    requested_val = (
        max(1, int(round(n_samples * float(val_ratio))))
        if float(val_ratio) > 0.0
        else 0
    )

    # A non-empty test set is mandatory. A validation sample is mandatory only
    # when val_ratio > 0. This clips an infeasible min_train_per_class request
    # rather than silently deleting validation/test samples.
    max_train = n_samples - requested_val - 1
    if max_train < 1 and requested_val > 0:
        requested_val = max(0, n_samples - 2)
        max_train = n_samples - requested_val - 1
    n_train = max(1, min(requested_train, max_train))

    remaining_after_train = n_samples - n_train
    n_val = min(requested_val, max(0, remaining_after_train - 1))
    n_test = n_samples - n_train - n_val
    if n_test < 1:
        raise RuntimeError(
            "Internal split error: test partition is empty for a class with "
            f"n={n_samples}, train={n_train}, val={n_val}."
        )

    return (
        shuffled[:n_train].copy(),
        shuffled[n_train : n_train + n_val].copy(),
        shuffled[n_train + n_val :].copy(),
    )


def _jsonable_protocol_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable_protocol_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_protocol_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_incremental_protocol_plan(
    labels: np.ndarray,
    *,
    base_classes: int,
    increment: int,
    train_ratio: float = 0.20,
    val_ratio: float = 0.10,
    min_train_per_class: int = 20,
    seed: int = 42,
    shuffle_order: bool = False,
    class_order: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the complete CIL class/split plan before PCA or patch extraction.

    ``labels`` must follow the same labeled-pixel order later used by
    ``ImageCubes``. The strict loader's ``ExtractLabeledPixelIndex`` provides
    exactly that ordering. The returned base-training indices can therefore be
    converted directly into a preprocessing fit mask.
    """
    labels = _as_1d_int_array(labels, name="labels")
    if int(labels.min()) < 0:
        raise ValueError(f"labels must be non-negative, got min={int(labels.min())}.")

    _validate_split_configuration(
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        min_train_per_class=int(min_train_per_class),
    )

    input_classes = sorted(int(value) for value in np.unique(labels).tolist())
    num_classes = len(input_classes)
    base_classes = int(base_classes)
    increment = int(increment)
    if not 2 <= base_classes <= num_classes:
        raise ValueError(
            f"base_classes must be in [2,{num_classes}], got {base_classes}."
        )
    if increment <= 0:
        raise ValueError(f"increment must be positive, got {increment}.")

    if class_order is not None:
        resolved_order = [int(value) for value in class_order]
        if len(resolved_order) != len(set(resolved_order)):
            raise ValueError(f"class_order contains duplicates: {resolved_order}.")
        if set(resolved_order) != set(input_classes):
            raise ValueError(
                "class_order must be a permutation of the labels present in the "
                f"dataset. order={resolved_order}, present={input_classes}."
            )
    elif bool(shuffle_order):
        rng = np.random.RandomState(int(seed))
        resolved_order = [int(value) for value in rng.permutation(input_classes).tolist()]
    else:
        resolved_order = list(input_classes)

    label_map = {int(global_id): int(seq_id) for seq_id, global_id in enumerate(resolved_order)}
    inv_label_map = {int(seq_id): int(global_id) for global_id, seq_id in label_map.items()}
    remapped_labels = np.fromiter(
        (label_map[int(value)] for value in labels.tolist()),
        dtype=np.int64,
        count=int(labels.size),
    )

    remaining = num_classes - base_classes
    num_phases = 1 + int(np.ceil(remaining / float(increment))) if remaining > 0 else 1
    phase_to_classes: Dict[int, List[int]] = {}
    class_to_phase: Dict[int, int] = {}
    for phase in range(num_phases):
        if phase == 0:
            phase_classes = list(range(base_classes))
        else:
            start = base_classes + (phase - 1) * increment
            stop = min(start + increment, num_classes)
            phase_classes = list(range(start, stop))
        if not phase_classes:
            raise RuntimeError(f"Protocol construction created an empty phase {phase}.")
        phase_to_classes[int(phase)] = phase_classes
        for class_id in phase_classes:
            class_to_phase[int(class_id)] = int(phase)

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    split_class_counts: Dict[int, Dict[str, int]] = {}
    for seq_id in range(num_classes):
        class_indices = np.flatnonzero(remapped_labels == seq_id).astype(np.int64)
        train_idx, val_idx, test_idx = _split_one_class(
            class_indices,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            min_train_per_class=int(min_train_per_class),
            seed=int(seed) + int(seq_id),
        )
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        test_parts.append(test_idx)
        split_class_counts[int(seq_id)] = {
            "all": int(class_indices.size),
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        }

    def concatenate(parts: Sequence[np.ndarray]) -> np.ndarray:
        nonempty = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts if len(part)]
        return (
            np.concatenate(nonempty).astype(np.int64, copy=False)
            if nonempty
            else np.empty((0,), dtype=np.int64)
        )

    train_indices = concatenate(train_parts)
    val_indices = concatenate(val_parts)
    test_indices = concatenate(test_parts)
    all_split = concatenate([train_indices, val_indices, test_indices])
    if all_split.size != labels.size or np.unique(all_split).size != labels.size:
        raise RuntimeError(
            "Protocol split does not cover every labeled sample exactly once: "
            f"covered={all_split.size}, unique={np.unique(all_split).size}, total={labels.size}."
        )

    base_class_ids = phase_to_classes[0]
    base_mask = np.isin(remapped_labels[train_indices], np.asarray(base_class_ids, dtype=np.int64))
    base_train_indices = train_indices[base_mask].astype(np.int64, copy=True)
    if base_train_indices.size == 0:
        raise RuntimeError("Base protocol contains no base-training samples.")
    bad_base = sorted(
        set(int(value) for value in remapped_labels[base_train_indices].tolist())
        - set(base_class_ids)
    )
    if bad_base:
        raise RuntimeError(f"Base preprocessing indices contain future classes: {bad_base}.")

    return {
        "version": int(_PROTOCOL_PLAN_VERSION),
        "num_samples": int(labels.size),
        "labels_sha256": _labels_sha256(labels),
        "input_classes": input_classes,
        "num_classes": int(num_classes),
        "class_order": resolved_order,
        "label_map": label_map,
        "inv_label_map": inv_label_map,
        "remapped_labels": remapped_labels,
        "base_classes": int(base_classes),
        "increment": int(increment),
        "num_phases": int(num_phases),
        "phase_to_classes": phase_to_classes,
        "class_to_phase": class_to_phase,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "base_train_indices": base_train_indices,
        "split_class_counts": split_class_counts,
        "split_policy": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "min_train_per_class": int(min_train_per_class),
            "seed": int(seed),
            "shuffle_order": bool(shuffle_order),
            "explicit_class_order": class_order is not None,
        },
    }


def _normalize_int_key_mapping(value: Mapping[Any, Any]) -> Dict[int, Any]:
    return {int(key): item for key, item in dict(value).items()}


def validate_incremental_protocol_plan(
    plan: Mapping[str, Any],
    labels: np.ndarray,
    *,
    expected_base_classes: Optional[int] = None,
    expected_increment: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize and validate a plan before it is attached to patch tensors."""
    if not isinstance(plan, Mapping):
        raise TypeError(f"protocol_plan must be a mapping, got {type(plan)}.")
    labels = _as_1d_int_array(labels, name="labels")
    normalized = dict(plan)

    # Compatibility for the one broken intermediate schema that emitted every
    # current field except ``version``.  Upgrade only that exact complete shape;
    # genuinely incomplete plans still fail below.
    if "version" not in normalized:
        normalized["version"] = 1

    required = {
        "version", "num_samples", "labels_sha256", "input_classes",
        "num_classes", "class_order", "label_map", "inv_label_map",
        "base_classes", "increment", "num_phases", "phase_to_classes",
        "class_to_phase", "train_indices", "val_indices", "test_indices",
        "base_train_indices", "split_policy",
    }
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"protocol_plan is missing required fields: {missing}.")

    plan_version = int(normalized["version"])
    if plan_version not in {1, _PROTOCOL_PLAN_VERSION}:
        raise ValueError(
            f"Unsupported protocol_plan version={plan_version}; "
            f"supported versions are 1 and {_PROTOCOL_PLAN_VERSION}."
        )
    normalized["version"] = plan_version

    if int(normalized["num_samples"]) != int(labels.size):
        raise ValueError(
            "protocol_plan sample count does not match labels: "
            f"plan={normalized['num_samples']}, labels={labels.size}."
        )
    if str(normalized["labels_sha256"]) != _labels_sha256(labels):
        raise ValueError(
            "protocol_plan labels fingerprint does not match the supplied label "
            "array. The plan and ImageCubes ordering are not the same."
        )

    normalized["input_classes"] = [int(value) for value in normalized["input_classes"]]
    normalized["class_order"] = [int(value) for value in normalized["class_order"]]
    normalized["label_map"] = {
        int(key): int(value) for key, value in _normalize_int_key_mapping(normalized["label_map"]).items()
    }
    normalized["inv_label_map"] = {
        int(key): int(value) for key, value in _normalize_int_key_mapping(normalized["inv_label_map"]).items()
    }
    normalized["phase_to_classes"] = {
        int(key): [int(value) for value in values]
        for key, values in _normalize_int_key_mapping(normalized["phase_to_classes"]).items()
    }
    normalized["class_to_phase"] = {
        int(key): int(value)
        for key, value in _normalize_int_key_mapping(normalized["class_to_phase"]).items()
    }
    for name in ("train_indices", "val_indices", "test_indices", "base_train_indices"):
        normalized[name] = np.asarray(normalized[name], dtype=np.int64).reshape(-1)

    base_classes = int(normalized["base_classes"])
    increment = int(normalized["increment"])
    if expected_base_classes is not None and base_classes != int(expected_base_classes):
        raise ValueError(
            f"protocol_plan base_classes={base_classes} does not match requested "
            f"base_classes={int(expected_base_classes)}."
        )
    if expected_increment is not None and increment != int(expected_increment):
        raise ValueError(
            f"protocol_plan increment={increment} does not match requested "
            f"increment={int(expected_increment)}."
        )

    num_classes = int(normalized["num_classes"])
    if set(normalized["class_order"]) != set(normalized["input_classes"]):
        raise ValueError("protocol_plan class_order is not a permutation of input_classes.")
    if len(normalized["class_order"]) != num_classes:
        raise ValueError("protocol_plan class_order length does not match num_classes.")

    recomputed = np.fromiter(
        (normalized["label_map"][int(value)] for value in labels.tolist()),
        dtype=np.int64,
        count=int(labels.size),
    )
    if "remapped_labels" in normalized:
        stored = np.asarray(normalized["remapped_labels"], dtype=np.int64).reshape(-1)
        if not np.array_equal(stored, recomputed):
            raise ValueError("protocol_plan remapped_labels do not match label_map.")
    normalized["remapped_labels"] = recomputed

    expected_phases = list(range(int(normalized["num_phases"])))
    if sorted(normalized["phase_to_classes"]) != expected_phases:
        raise ValueError(
            "protocol_plan phase keys must be contiguous: "
            f"got={sorted(normalized['phase_to_classes'])}, expected={expected_phases}."
        )
    phase_union = [
        class_id
        for phase in expected_phases
        for class_id in normalized["phase_to_classes"][phase]
    ]
    if phase_union != list(range(num_classes)):
        raise ValueError(
            "protocol_plan phases must cover sequential class IDs exactly once: "
            f"got={phase_union}."
        )

    all_split = np.concatenate(
        [normalized["train_indices"], normalized["val_indices"], normalized["test_indices"]]
    )
    if all_split.size != labels.size or np.unique(all_split).size != labels.size:
        raise ValueError("protocol_plan train/val/test indices do not partition the samples.")
    if all_split.size and (int(all_split.min()) < 0 or int(all_split.max()) >= labels.size):
        raise ValueError("protocol_plan contains indices outside the supplied label array.")

    train_set = set(int(value) for value in normalized["train_indices"].tolist())
    if not set(int(value) for value in normalized["base_train_indices"].tolist()).issubset(train_set):
        raise ValueError("base_train_indices are not a subset of train_indices.")
    base_ids = set(normalized["phase_to_classes"][0])
    bad = sorted(
        set(int(value) for value in recomputed[normalized["base_train_indices"]].tolist()) - base_ids
    )
    if bad:
        raise ValueError(f"base_train_indices contain future classes: {bad}.")

    return normalized



class HSIPatchDataset(Dataset):
    """Patch dataset that lazily constructs paired PC-SIRG intervention views.

    Persistent arrays contain the original processed patch and only the processed
    positive/negative center vectors. Full [K,C,H,W] views are materialized for
    one sample at a time, avoiding two duplicated HSI patch stores.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        *,
        coords: np.ndarray,
        sample_indices: np.ndarray,
        spectral_positive_centers: np.ndarray,
        spectral_negative_centers: np.ndarray,
        spectral_step_sizes: np.ndarray,
        intervention_definition_version: int,
    ) -> None:
        patches_np = np.ascontiguousarray(patches, dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        coords_np = np.asarray(coords, dtype=np.int64)
        sample_indices_np = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        positive_np = np.ascontiguousarray(spectral_positive_centers, dtype=np.float32)
        negative_np = np.ascontiguousarray(spectral_negative_centers, dtype=np.float32)
        steps_np = np.ascontiguousarray(spectral_step_sizes, dtype=np.float32)

        n = int(labels_np.size)
        if patches_np.ndim != 4:
            raise ValueError(f"patches must be [N,C,H,W], got {patches_np.shape}.")
        if patches_np.shape[0] != n:
            raise ValueError(f"patch/label length mismatch: {patches_np.shape[0]} vs {n}.")
        if coords_np.shape != (n, 2):
            raise ValueError(f"coords must be [N,2], got {coords_np.shape}.")
        if sample_indices_np.shape != (n,):
            raise ValueError(
                f"sample_indices must be [N], got {sample_indices_np.shape}."
            )
        expected_prefix = (n,)
        if positive_np.ndim != 3 or negative_np.shape != positive_np.shape:
            raise ValueError(
                "spectral centers must be paired arrays with identical [N,K,C] shape; "
                f"got {positive_np.shape} and {negative_np.shape}."
            )
        if positive_np.shape[0] != expected_prefix[0]:
            raise ValueError("spectral center sample count differs from patches.")
        if positive_np.shape[2] != patches_np.shape[1]:
            raise ValueError(
                "spectral center channel count must equal processed patch channels: "
                f"centers={positive_np.shape[2]}, patches={patches_np.shape[1]}."
            )
        k = int(positive_np.shape[1])
        if k <= 0:
            raise ValueError("At least one spectral intervention is required.")
        if steps_np.shape != (n, k):
            raise ValueError(
                f"spectral_step_sizes must be [N,K]={n,k}, got {steps_np.shape}."
            )
        for name, value in (
            ("patches", patches_np),
            ("spectral_positive_centers", positive_np),
            ("spectral_negative_centers", negative_np),
            ("spectral_step_sizes", steps_np),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf.")
        if np.any(np.abs(steps_np) <= 1e-12):
            raise ValueError("spectral_step_sizes must be non-zero.")

        self.patches = torch.from_numpy(patches_np).float()
        self.labels = torch.from_numpy(labels_np).long()
        self.coords = torch.from_numpy(coords_np).long()
        self.sample_indices = torch.from_numpy(sample_indices_np).long()
        self.spectral_positive_centers = torch.from_numpy(positive_np).float()
        self.spectral_negative_centers = torch.from_numpy(negative_np).float()
        self.spectral_step_sizes = torch.from_numpy(steps_np).float()
        self.intervention_definition_version = int(intervention_definition_version)
        self.num_interventions = k

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch = self.patches[int(idx)]
        if patch.ndim != 3:
            raise RuntimeError(f"Stored patch must be [C,H,W], got {tuple(patch.shape)}.")
        positive = patch.unsqueeze(0).repeat(self.num_interventions, 1, 1, 1)
        negative = patch.unsqueeze(0).repeat(self.num_interventions, 1, 1, 1)
        center_y = int(patch.shape[1] // 2)
        center_x = int(patch.shape[2] // 2)
        positive[:, :, center_y, center_x] = self.spectral_positive_centers[int(idx)]
        negative[:, :, center_y, center_x] = self.spectral_negative_centers[int(idx)]
        return {
            "image": patch,
            "label": self.labels[int(idx)],
            "coord": self.coords[int(idx)],
            "sample_index": self.sample_indices[int(idx)],
            "spectral_positive_patches": positive,
            "spectral_negative_patches": negative,
            "spectral_step_sizes": self.spectral_step_sizes[int(idx)],
            "intervention_definition_version": torch.tensor(
                self.intervention_definition_version, dtype=torch.long
            ),
        }



class ClassBalancedBatchSampler(Sampler[List[int]]):
    """Deterministic class-balanced sampler with unique samples per batch.

    Geometry estimation must never duplicate one observation inside the same
    batch. Duplicate rows artificially reduce covariance rank, inflate apparent
    support, and bias spectral-feature correlation. Samples may reappear in a
    later batch or epoch, but every batch is internally unique.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        classes_per_batch: Optional[int] = None,
        samples_per_class: Optional[int] = None,
        seed: int = 42,
        drop_last: bool = False,
        require_unique_within_batch: bool = True,
    ) -> None:
        labels_np = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if labels_np.size == 0:
            raise ValueError("ClassBalancedBatchSampler received zero labels.")

        self.labels_np = labels_np
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.require_unique_within_batch = bool(require_unique_within_batch)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")

        self.class_to_indices: Dict[int, np.ndarray] = {
            int(class_id): np.flatnonzero(labels_np == int(class_id)).astype(np.int64)
            for class_id in sorted(np.unique(labels_np).tolist())
        }
        self.class_to_indices = {
            class_id: indices
            for class_id, indices in self.class_to_indices.items()
            if indices.size > 0
        }
        self.classes = list(self.class_to_indices)
        if not self.classes:
            raise ValueError("ClassBalancedBatchSampler found no classes.")

        if classes_per_batch is None or int(classes_per_batch) <= 0:
            classes_per_batch = min(len(self.classes), self.batch_size)
        self.classes_per_batch = int(
            max(1, min(int(classes_per_batch), len(self.classes), self.batch_size))
        )

        if samples_per_class is None or int(samples_per_class) <= 0:
            samples_per_class = max(1, self.batch_size // self.classes_per_batch)
        self.samples_per_class = int(max(1, int(samples_per_class)))
        if self.classes_per_batch * self.samples_per_class > self.batch_size:
            raise ValueError(
                "classes_per_batch * samples_per_class exceeds batch_size: "
                f"{self.classes_per_batch} * {self.samples_per_class} > {self.batch_size}. "
                "Do not silently shrink geometry support."
            )

        if self.require_unique_within_batch:
            insufficient = {
                int(class_id): int(indices.size)
                for class_id, indices in self.class_to_indices.items()
                if int(indices.size) < self.samples_per_class
            }
            if insufficient:
                raise ValueError(
                    "Unique class-balanced batches are impossible because some classes "
                    f"have fewer than {self.samples_per_class} samples: {insufficient}. "
                    "Reduce geometry_samples_per_class or increase the class train split; "
                    "never duplicate samples to fake covariance support."
                )

        self.effective_batch_size = self.classes_per_batch * self.samples_per_class
        if self.effective_batch_size <= 0:
            raise RuntimeError("Resolved balanced batch size is zero.")

        coverage_batches = int(np.ceil(len(self.classes) / float(self.classes_per_batch)))
        sample_batches = (
            labels_np.size // self.effective_batch_size
            if self.drop_last
            else int(np.ceil(labels_np.size / float(self.effective_batch_size)))
        )
        self.num_batches = max(coverage_batches, sample_batches, 1)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + 1009 * self.epoch)
        class_queue = [int(v) for v in rng.permutation(self.classes).tolist()]
        class_cursor = 0

        pools: Dict[int, np.ndarray] = {
            class_id: rng.permutation(source)
            for class_id, source in self.class_to_indices.items()
        }
        pointers: Dict[int, int] = {class_id: 0 for class_id in self.classes}

        def next_classes() -> List[int]:
            nonlocal class_queue, class_cursor
            chosen: List[int] = []
            while len(chosen) < self.classes_per_batch:
                if class_cursor >= len(class_queue):
                    class_queue = [int(v) for v in rng.permutation(self.classes).tolist()]
                    class_cursor = 0
                class_id = int(class_queue[class_cursor])
                class_cursor += 1
                if class_id not in chosen:
                    chosen.append(class_id)
            return chosen

        def draw_unique(class_id: int, count: int) -> List[int]:
            source = self.class_to_indices[class_id]
            if self.require_unique_within_batch and int(source.size) < int(count):
                raise RuntimeError(
                    f"Class {class_id} has {source.size} samples but batch requests {count}."
                )
            selected: List[int] = []
            selected_set: Set[int] = set()
            while len(selected) < count:
                if pointers[class_id] >= len(pools[class_id]):
                    pools[class_id] = rng.permutation(source)
                    pointers[class_id] = 0
                candidate = int(pools[class_id][pointers[class_id]])
                pointers[class_id] += 1
                if self.require_unique_within_batch and candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
            return selected

        for _ in range(self.num_batches):
            chosen = next_classes()
            batch: List[int] = []
            for class_id in chosen:
                batch.extend(draw_unique(class_id, self.samples_per_class))
            if len(batch) != len(set(batch)):
                raise RuntimeError(
                    "Balanced sampler produced duplicate indices inside one batch."
                )
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.effective_batch_size:
                continue
            yield batch


class IncrementalHSIDataset:
    """Strict non-exemplar PC-SIRG dataset manager.

    The protocol is constructed before preprocessing. The manager stores no raw
    spectra and exposes no one-sided spectral augmentation. It receives paired
    processed center interventions produced in ordered wavelength space before
    the frozen normalization/PCA transform.
    """

    @staticmethod
    def build_protocol_plan(
        labels: np.ndarray,
        *,
        base_classes: int,
        increment: int,
        train_ratio: float = 0.20,
        val_ratio: float = 0.10,
        min_train_per_class: int = 20,
        seed: int = 42,
        shuffle_order: bool = False,
        class_order: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        return build_incremental_protocol_plan(
            labels,
            base_classes=int(base_classes),
            increment=int(increment),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            min_train_per_class=int(min_train_per_class),
            seed=int(seed),
            shuffle_order=bool(shuffle_order),
            class_order=class_order,
        )

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        gt_shape: Tuple[int, int],
        GT: np.ndarray,
        base_classes: int,
        increment: int,
        train_ratio: float = 0.20,
        val_ratio: float = 0.10,
        seed: int = 42,
        shuffle_order: bool = False,
        device: str = "cuda",
        min_train_per_class: int = 20,
        num_workers: int = 0,
        strict_non_exemplar: bool = True,
        target_names: Optional[List[str]] = None,
        label_policy: Optional[Dict[str, Any]] = None,
        class_balanced_train_batches: bool = True,
        strict_unique_balanced_batches: bool = True,
        geometry_batch_classes: int = 0,
        geometry_samples_per_class: int = 0,
        spectral_positive_centers: Optional[np.ndarray] = None,
        spectral_negative_centers: Optional[np.ndarray] = None,
        spectral_step_sizes: Optional[np.ndarray] = None,
        intervention_metadata: Optional[Mapping[str, Any]] = None,
        require_response_views: bool = True,
        purge_finalized_train_payload: bool = True,
        protocol_plan: Optional[Mapping[str, Any]] = None,
        class_order: Optional[Sequence[int]] = None,
    ) -> None:
        self.patches = np.ascontiguousarray(patches, dtype=np.float32).copy()
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.gt_shape = tuple(int(value) for value in gt_shape)
        self.GT = np.asarray(GT, dtype=np.int64)
        n = int(self.labels.size)

        if self.patches.ndim != 4:
            raise ValueError(f"patches must be [N,C,H,W], got {self.patches.shape}.")
        if self.patches.shape[0] != n:
            raise ValueError(f"patch/label mismatch: {self.patches.shape[0]} vs {n}.")
        if self.coords.shape != (n, 2):
            raise ValueError(f"coords must be [N,2], got {self.coords.shape}.")
        if n == 0 or int(self.labels.min()) < 0:
            raise ValueError("labels must be a non-empty non-negative array.")
        if len(self.gt_shape) != 2 or tuple(self.GT.shape[:2]) != self.gt_shape:
            raise ValueError(
                f"gt_shape={self.gt_shape} does not match GT={self.GT.shape[:2]}."
            )
        rows, cols = self.coords[:, 0], self.coords[:, 1]
        if (
            int(rows.min()) < 0
            or int(cols.min()) < 0
            or int(rows.max()) >= self.gt_shape[0]
            or int(cols.max()) >= self.gt_shape[1]
        ):
            raise ValueError("coords contain locations outside gt_shape.")
        if not np.isfinite(self.patches).all():
            raise ValueError("patches contain NaN/Inf.")

        self.base_classes = int(base_classes)
        self.increment = int(increment)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.device = str(device)
        self.min_train_per_class = int(min_train_per_class)
        self.num_workers = int(num_workers)
        self.strict_non_exemplar = bool(strict_non_exemplar)
        self.class_balanced_train_batches = bool(class_balanced_train_batches)
        self.strict_unique_balanced_batches = bool(strict_unique_balanced_batches)
        self.geometry_batch_classes = int(geometry_batch_classes)
        self.geometry_samples_per_class = int(geometry_samples_per_class)
        self.require_response_views = bool(require_response_views)
        # Strict mode must physically erase finalized training payloads. Access
        # control alone is not enough to demonstrate an exemplar-free state.
        self.purge_finalized_train_payload = bool(purge_finalized_train_payload)
        if self.strict_non_exemplar and not self.purge_finalized_train_payload:
            raise ValueError(
                "strict_non_exemplar=True requires purge_finalized_train_payload=True."
            )
        self._purged_train_indices: Set[int] = set()
        self._input_target_names = None if target_names is None else [str(v) for v in target_names]
        self.target_names: Optional[List[str]] = None
        self.pin_memory = self.device.startswith("cuda")

        if protocol_plan is None:
            plan = self.build_protocol_plan(
                self.labels,
                base_classes=self.base_classes,
                increment=self.increment,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                min_train_per_class=self.min_train_per_class,
                seed=self.seed,
                shuffle_order=bool(shuffle_order),
                class_order=class_order,
            )
        else:
            plan = validate_incremental_protocol_plan(
                protocol_plan,
                self.labels,
                expected_base_classes=self.base_classes,
                expected_increment=self.increment,
            )
        self._protocol_plan = validate_incremental_protocol_plan(
            plan,
            self.labels,
            expected_base_classes=self.base_classes,
            expected_increment=self.increment,
        )
        self.all_classes = list(self._protocol_plan["input_classes"])
        self.num_classes = int(self._protocol_plan["num_classes"])
        self.class_order = list(self._protocol_plan["class_order"])
        self.label_map = dict(self._protocol_plan["label_map"])
        self.inv_label_map = dict(self._protocol_plan["inv_label_map"])
        self.remapped_labels = np.asarray(
            self._protocol_plan["remapped_labels"], dtype=np.int64
        ).copy()
        self.num_phases = int(self._protocol_plan["num_phases"])
        self.phase_to_classes = {
            int(key): [int(value) for value in values]
            for key, values in self._protocol_plan["phase_to_classes"].items()
        }
        self.class_to_phase = {
            int(key): int(value)
            for key, value in self._protocol_plan["class_to_phase"].items()
        }
        self.train_indices = np.asarray(self._protocol_plan["train_indices"], dtype=np.int64).copy()
        self.val_indices = np.asarray(self._protocol_plan["val_indices"], dtype=np.int64).copy()
        self.test_indices = np.asarray(self._protocol_plan["test_indices"], dtype=np.int64).copy()
        self.base_train_indices = np.asarray(
            self._protocol_plan["base_train_indices"], dtype=np.int64
        ).copy()

        self.label_policy = self._normalize_label_policy(label_policy)
        self.has_background = bool(self.label_policy.get("has_background", True))
        self.background_label = self.label_policy.get(
            "background_label", 0 if self.has_background else None
        )
        self.raw_class_values = [
            int(value) for value in self.label_policy.get("raw_class_values", [])
        ]
        self.target_names_by_seq = self._build_target_names_by_seq(self._input_target_names)
        self.target_names = (
            None if self.target_names_by_seq is None else list(self.target_names_by_seq)
        )

        self.intervention_metadata = dict(intervention_metadata or {})
        self.intervention_definition_version = int(
            self.intervention_metadata.get(
                "definition_version", self.intervention_metadata.get("version", 1)
            )
        )
        self._spectral_positive_centers: Optional[np.ndarray] = None
        self._spectral_negative_centers: Optional[np.ndarray] = None
        self._spectral_step_sizes: Optional[np.ndarray] = None
        self.num_spectral_interventions = 0
        self._initialize_response_views(
            spectral_positive_centers,
            spectral_negative_centers,
            spectral_step_sizes,
        )

        self.current_phase = 0
        self.finalized_phases: Set[int] = set()
        self.finalized_classes: Set[int] = set()
        self._memory_build_active = False
        self._memory_build_classes: Set[int] = set()

        self._validate_class_zero_policy()
        self._validate_full_class_coverage()
        self._validate_protocol_integrity()
        self.assert_preprocess_fit_indices()
        self._print_stats()

    def _initialize_response_views(
        self,
        positive: Optional[np.ndarray],
        negative: Optional[np.ndarray],
        steps: Optional[np.ndarray],
    ) -> None:
        supplied = [positive is not None, negative is not None, steps is not None]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "PC-SIRG response input is incomplete. Provide positive centers, "
                "negative centers, and step sizes together."
            )
        if not all(supplied):
            if self.require_response_views:
                raise ValueError(
                    "PC-SIRG requires paired processed intervention centers and "
                    "central-difference step sizes."
                )
            return

        pos = np.ascontiguousarray(positive, dtype=np.float32)
        neg = np.ascontiguousarray(negative, dtype=np.float32)
        if pos.ndim != 3 or neg.shape != pos.shape:
            raise ValueError(
                "spectral_positive_centers and spectral_negative_centers must "
                f"share [N,K,C] shape; got {pos.shape} and {neg.shape}."
            )
        n, k, channels = pos.shape
        if n != len(self.labels):
            raise ValueError(f"response sample count={n} != labels={len(self.labels)}.")
        if channels != self.patches.shape[1]:
            raise ValueError(
                f"response channels={channels} != patch channels={self.patches.shape[1]}."
            )
        if k <= 0:
            raise ValueError("At least one intervention is required.")
        step_array = np.asarray(steps, dtype=np.float32)
        if step_array.ndim == 0:
            step_array = np.full((n, k), float(step_array), dtype=np.float32)
        elif step_array.ndim == 1 and step_array.size == k:
            step_array = np.broadcast_to(step_array.reshape(1, k), (n, k)).copy()
        elif step_array.ndim == 1 and step_array.size == n:
            step_array = np.broadcast_to(step_array.reshape(n, 1), (n, k)).copy()
        elif step_array.shape == (n, k):
            step_array = np.ascontiguousarray(step_array, dtype=np.float32)
        else:
            raise ValueError(
                f"step sizes must be scalar, [K], [N], or [N,K]; got {step_array.shape}."
            )
        for name, array in (("positive", pos), ("negative", neg), ("steps", step_array)):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} response data contain NaN/Inf.")
        if np.any(np.abs(step_array) <= 1e-12):
            raise ValueError("response step sizes must be non-zero.")

        required_flags = {
            "center_pixel_only": True,
            "applied_before_preprocessing_and_pca": True,
            "ordered_wavelength_required": True,
        }
        for key, expected in required_flags.items():
            if key in self.intervention_metadata and bool(self.intervention_metadata[key]) != expected:
                raise ValueError(
                    f"intervention_metadata[{key!r}] must be {expected} for PC-SIRG."
                )
        metadata_k = self.intervention_metadata.get("num_interventions")
        if metadata_k is not None and int(metadata_k) != k:
            raise ValueError(
                f"intervention metadata K={metadata_k} differs from array K={k}."
            )

        self._spectral_positive_centers = pos.copy()
        self._spectral_negative_centers = neg.copy()
        self._spectral_step_sizes = step_array.copy()
        self.num_spectral_interventions = int(k)
        self.intervention_metadata.update(
            {
                "definition_version": int(self.intervention_definition_version),
                "num_interventions": int(k),
                "center_pixel_only": True,
                "applied_before_preprocessing_and_pca": True,
                "ordered_wavelength_required": True,
                "persistent_form": "processed_positive_negative_center_vectors",
            }
        )

    # ------------------------------------------------------------------
    # Label/protocol validation
    # ------------------------------------------------------------------
    def _normalize_label_policy(self, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        policy = dict(value or {})
        policy["has_background"] = bool(policy.get("has_background", True))
        policy["background_label"] = (
            int(policy.get("background_label", 0))
            if policy["has_background"]
            else None
        )
        policy["raw_class_values"] = [
            int(item) for item in policy.get("raw_class_values", [])
        ]
        return policy

    def _validate_class_zero_policy(self) -> None:
        if 0 in set(int(x) for x in self.labels.tolist()) and 0 not in set(
            int(x) for x in self.remapped_labels.tolist()
        ):
            raise RuntimeError("Sequential foreground class 0 disappeared during remapping.")

    def _build_target_names_by_seq(
        self, target_names: Optional[List[str]]
    ) -> Optional[List[str]]:
        if target_names is None:
            return None
        result: List[str] = []
        for sequential_id in range(len(self.all_classes)):
            input_id = int(self.inv_label_map.get(sequential_id, sequential_id))
            result.append(
                str(target_names[input_id])
                if 0 <= input_id < len(target_names)
                else f"Class {input_id}"
            )
        return result

    def _validate_full_class_coverage(self) -> None:
        present = set(int(x) for x in np.unique(self.remapped_labels).tolist())
        expected = set(range(self.num_classes))
        if present != expected:
            raise RuntimeError(
                f"Remapped class space is broken: present={sorted(present)}, "
                f"expected={sorted(expected)}."
            )

    def _validate_protocol_integrity(self) -> None:
        phase_union = [
            int(class_id)
            for phase in range(self.num_phases)
            for class_id in self.phase_to_classes.get(phase, [])
        ]
        if phase_union != list(range(self.num_classes)):
            raise RuntimeError(
                f"Phase partition is invalid: {phase_union} != {list(range(self.num_classes))}."
            )
        all_split = np.concatenate(
            [self.train_indices, self.val_indices, self.test_indices]
        ).astype(np.int64)
        if all_split.size != len(self.labels) or np.unique(all_split).size != len(self.labels):
            raise RuntimeError("Train/validation/test indices do not partition samples exactly once.")

    def _validate_phase(self, phase: int) -> int:
        value = int(phase)
        if value < 0 or value >= self.num_phases:
            raise ValueError(f"Invalid phase {value}; expected 0..{self.num_phases - 1}.")
        return value

    # ------------------------------------------------------------------
    # Class/protocol helpers
    # ------------------------------------------------------------------
    def seq_to_global(self, seq_id: int) -> int:
        return int(self.inv_label_map[int(seq_id)])

    def global_to_seq(self, global_id: int) -> int:
        return int(self.label_map[int(global_id)])

    def get_phase_classes(self, phase: int) -> List[int]:
        return list(self.phase_to_classes[self._validate_phase(phase)])

    def get_classes_up_to_phase(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        return _ordered_unique_ints(
            class_id
            for phase_index in range(phase + 1)
            for class_id in self.phase_to_classes[phase_index]
        )

    def get_seen_classes(self, phase: int) -> List[int]:
        return self.get_classes_up_to_phase(phase)

    def get_old_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        return [] if phase == 0 else self.get_classes_up_to_phase(phase - 1)

    def get_new_classes(self, phase: int) -> List[int]:
        return self.get_phase_classes(phase)

    def get_future_classes(self, phase: int) -> List[int]:
        seen = set(self.get_seen_classes(phase))
        return [class_id for class_id in range(self.num_classes) if class_id not in seen]

    def classifier_num_classes_for_phase(self, phase: int) -> int:
        seen = self.get_seen_classes(phase)
        expected = list(range(max(seen) + 1))
        if seen != expected:
            raise RuntimeError(
                f"Seen sequential IDs must be contiguous: seen={seen}, expected={expected}."
            )
        return len(seen)

    def get_phase_class_splits(self, phase: int) -> Dict[str, List[int]]:
        return {
            "old": self.get_old_classes(phase),
            "new": self.get_new_classes(phase),
            "seen": self.get_seen_classes(phase),
            "future": self.get_future_classes(phase),
        }

    def assert_classifier_contract(
        self, phase: int, logits_or_width: Any, *, context: str = ""
    ) -> None:
        expected = self.classifier_num_classes_for_phase(phase)
        if torch.is_tensor(logits_or_width):
            if logits_or_width.ndim != 2:
                raise RuntimeError(f"{context}: logits must be [B,C].")
            observed = int(logits_or_width.shape[1])
        else:
            observed = int(logits_or_width)
        if observed != expected:
            raise RuntimeError(
                f"{context}: classifier width={observed}, expected={expected}."
            )

    def assert_labels_for_phase(
        self,
        labels: Any,
        phase: int,
        *,
        context: str = "",
        current_train_only: bool = False,
        cumulative: bool = True,
    ) -> None:
        y = _as_1d_int_array(
            labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels,
            name="labels",
        )
        allowed = (
            self.get_new_classes(phase)
            if current_train_only or not cumulative
            else self.get_seen_classes(phase)
        )
        bad = sorted(set(int(v) for v in y.tolist()) - set(allowed))
        if bad:
            raise RuntimeError(f"{context}: labels {bad} are outside allowed classes {allowed}.")

    def export_protocol_plan(self, *, jsonable: bool = True) -> Dict[str, Any]:
        plan = dict(self._protocol_plan)
        for key, value in (
            ("remapped_labels", self.remapped_labels),
            ("train_indices", self.train_indices),
            ("val_indices", self.val_indices),
            ("test_indices", self.test_indices),
            ("base_train_indices", self.base_train_indices),
        ):
            plan[key] = value.copy()
        return _jsonable_protocol_value(plan) if jsonable else plan

    def get_phase_split_indices(self, phase: int, split: str) -> np.ndarray:
        indices = self._get_split_indices(split)
        mask = np.isin(self.remapped_labels[indices], self.get_phase_classes(phase))
        return indices[mask].astype(np.int64, copy=True)

    def get_preprocess_fit_indices(self, phase: int = 0) -> np.ndarray:
        if self._validate_phase(phase) != 0:
            raise ValueError("Preprocessing is fitted once on phase-0 training samples.")
        self.assert_preprocess_fit_indices()
        return self.base_train_indices.copy()

    def get_preprocess_fit_coordinates(self) -> np.ndarray:
        return self.coords[self.get_preprocess_fit_indices()].copy()

    def assert_preprocess_fit_indices(self) -> None:
        indices = self.base_train_indices
        if indices.size == 0 or np.unique(indices).size != indices.size:
            raise RuntimeError("base_train_indices must be non-empty and unique.")
        if int(indices.min()) < 0 or int(indices.max()) >= len(self.labels):
            raise RuntimeError("base_train_indices contain out-of-range values.")
        if not set(indices.tolist()).issubset(set(self.train_indices.tolist())):
            raise RuntimeError("base_train_indices are not a subset of train_indices.")
        bad = sorted(
            set(self.remapped_labels[indices].tolist())
            - set(self.phase_to_classes[0])
        )
        if bad:
            raise RuntimeError(f"Preprocessing fit indices contain future classes: {bad}.")

    # ------------------------------------------------------------------
    # Response-view contract
    # ------------------------------------------------------------------
    def has_response_views(self) -> bool:
        return bool(
            self._spectral_positive_centers is not None
            and self._spectral_negative_centers is not None
            and self._spectral_step_sizes is not None
            and self.num_spectral_interventions > 0
        )

    def get_response_contract(self) -> Dict[str, Any]:
        return {
            "available": self.has_response_views(),
            "definition_version": int(self.intervention_definition_version),
            "num_interventions": int(self.num_spectral_interventions),
            "center_pixel_only": True,
            "created_before_preprocessing_and_pca": True,
            "batch_fields": [
                "spectral_positive_patches",
                "spectral_negative_patches",
                "spectral_step_sizes",
            ],
            "metadata": dict(self.intervention_metadata),
        }

    def get_response_centers(
        self, indices: Sequence[int], *, split: str = "train"
    ) -> Dict[str, np.ndarray]:
        if not self.has_response_views():
            raise RuntimeError("No PC-SIRG response views are available.")
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("indices is empty.")
        allowed = set(self._get_split_indices(split).tolist())
        outside = [int(v) for v in idx.tolist() if int(v) not in allowed]
        if outside:
            raise PermissionError(
                f"Requested response centers outside split={split!r}: {outside[:10]}."
            )
        if str(split).lower() in {"train", "all"} and self.strict_non_exemplar:
            for class_id in np.unique(self.remapped_labels[idx]).tolist():
                self._check_class_split_access(int(class_id), str(split).lower())
        assert self._spectral_positive_centers is not None
        assert self._spectral_negative_centers is not None
        assert self._spectral_step_sizes is not None
        return {
            "positive_centers": self._spectral_positive_centers[idx].copy(),
            "negative_centers": self._spectral_negative_centers[idx].copy(),
            "step_sizes": self._spectral_step_sizes[idx].copy(),
        }

    # ------------------------------------------------------------------
    # Strict phase transitions and purge
    # ------------------------------------------------------------------
    def export_runtime_state(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "protocol_labels_sha256": str(self._protocol_plan["labels_sha256"]),
            "num_classes": int(self.num_classes),
            "phase_to_classes": {
                int(key): [int(v) for v in values]
                for key, values in self.phase_to_classes.items()
            },
            "current_phase": int(self.current_phase),
            "finalized_phases": sorted(self.finalized_phases),
            "finalized_classes": sorted(self.finalized_classes),
            "purge_finalized_train_payload": bool(self.purge_finalized_train_payload),
            "intervention_definition_version": int(self.intervention_definition_version),
            "num_spectral_interventions": int(self.num_spectral_interventions),
        }

    def restore_runtime_state(
        self,
        state: Mapping[str, Any],
        *,
        purge_finalized_train_payload: bool = True,
    ) -> None:
        if not isinstance(state, Mapping) or int(state.get("version", -1)) not in {1, 2}:
            raise RuntimeError("Unsupported dataset runtime state.")
        if str(state.get("protocol_labels_sha256", "")) != str(
            self._protocol_plan["labels_sha256"]
        ):
            raise RuntimeError("Dataset runtime state belongs to another protocol.")
        if int(state.get("num_classes", -1)) != self.num_classes:
            raise RuntimeError("Dataset runtime-state class count mismatch.")
        if int(state.get("version", 1)) >= 2:
            if int(state.get("intervention_definition_version", -1)) != int(
                self.intervention_definition_version
            ):
                raise RuntimeError("Dataset intervention-definition version mismatch.")
            if int(state.get("num_spectral_interventions", -1)) != int(
                self.num_spectral_interventions
            ):
                raise RuntimeError("Dataset intervention-count mismatch.")
        finalized = sorted(set(int(v) for v in state.get("finalized_phases", [])))
        if finalized and finalized != list(range(max(finalized) + 1)):
            raise RuntimeError(f"Finalized phases must be contiguous, got {finalized}.")
        expected_classes = sorted(
            class_id
            for phase in finalized
            for class_id in self.phase_to_classes[phase]
        )
        saved_classes = sorted(set(int(v) for v in state.get("finalized_classes", [])))
        if saved_classes != expected_classes:
            raise RuntimeError("Finalized class state does not match finalized phases.")
        self.current_phase = self._validate_phase(int(state.get("current_phase", 0)))
        self.finalized_phases = set(finalized)
        self.finalized_classes = set(saved_classes)
        self._memory_build_active = False
        self._memory_build_classes.clear()
        if purge_finalized_train_payload:
            for phase in finalized:
                self._purge_phase_train_payload(phase)
        self.assert_non_exemplar()

    def assert_phase_transition_ready(self, next_phase: int) -> None:
        next_phase = self._validate_phase(next_phase)
        if sorted(self.finalized_phases) != list(range(next_phase)):
            raise RuntimeError(
                f"Cannot enter phase {next_phase}; finalized={sorted(self.finalized_phases)}."
            )
        if set(self.get_old_classes(next_phase)) != self.finalized_classes:
            raise RuntimeError("Finalized classes do not match the old-class protocol.")
        self.assert_non_exemplar()

    def start_phase(self, phase: int) -> None:
        phase = self._validate_phase(phase)
        if phase > 0:
            self.assert_phase_transition_ready(phase)
        if phase in self.finalized_phases:
            raise RuntimeError(f"Phase {phase} is already finalized.")
        if self._memory_build_active:
            raise RuntimeError("Cannot change phase during memory construction.")
        self.current_phase = phase

    def finalize_phase(self, phase: int) -> None:
        phase = self._validate_phase(phase)
        if phase != self.current_phase or phase in self.finalized_phases:
            raise RuntimeError(
                f"Cannot finalize phase={phase}; current={self.current_phase}, "
                f"finalized={sorted(self.finalized_phases)}."
            )
        if self._memory_build_active:
            raise RuntimeError("Cannot finalize during memory construction.")
        if sorted(self.finalized_phases) != list(range(phase)):
            raise RuntimeError("Previous phases have not all been finalized.")
        self.finalized_phases.add(phase)
        self.finalized_classes.update(self.phase_to_classes[phase])
        if self.purge_finalized_train_payload:
            self._purge_phase_train_payload(phase)

    def _purge_phase_train_payload(self, phase: int) -> None:
        indices = self.get_phase_split_indices(phase, "train")
        if indices.size == 0:
            return
        self.patches[indices] = 0.0
        if self._spectral_positive_centers is not None:
            self._spectral_positive_centers[indices] = 0.0
        if self._spectral_negative_centers is not None:
            self._spectral_negative_centers[indices] = 0.0
        self._purged_train_indices.update(int(v) for v in indices.tolist())

    def is_phase_finalized(self, phase: int) -> bool:
        return int(phase) in self.finalized_phases

    def is_class_finalized(self, class_id: int) -> bool:
        return int(class_id) in self.finalized_classes

    def get_accessible_train_classes(self) -> List[int]:
        return (
            list(range(self.num_classes))
            if not self.strict_non_exemplar
            else list(self.phase_to_classes[self.current_phase])
        )

    def _is_train_access_allowed(self, class_id: int) -> bool:
        class_id = int(class_id)
        if not self.strict_non_exemplar:
            return True
        if class_id in self.finalized_classes:
            return False
        if self._memory_build_active and class_id in self._memory_build_classes:
            return True
        return class_id in self.phase_to_classes[self.current_phase]

    def _check_class_split_access(self, class_id: int, split: str) -> None:
        split_name = str(split).lower()
        if split_name not in {"train", "all"}:
            return
        if split_name == "all" and self.strict_non_exemplar and int(class_id) in self.finalized_classes:
            raise PermissionError(
                f"split='all' would expose finalized class {class_id} training pixels. "
                "Use validation/test maps after phase finalization."
            )
        if split_name == "train" and not self._is_train_access_allowed(class_id):
            raise PermissionError(
                f"Strict non-exemplar violation: old TRAIN access denied for class {class_id}."
            )

    @contextmanager
    def memory_build_context(self, phase: int):
        phase = self._validate_phase(phase)
        if phase != self.current_phase or phase in self.finalized_phases:
            raise PermissionError("Memory can be constructed only for the active unfinalized phase.")
        if self._memory_build_active:
            raise RuntimeError("Nested memory_build_context is not allowed.")
        self._memory_build_active = True
        self._memory_build_classes = set(self.phase_to_classes[phase])
        try:
            yield
        finally:
            self._memory_build_active = False
            self._memory_build_classes.clear()

    def assert_non_exemplar(self) -> bool:
        if not self.strict_non_exemplar:
            raise RuntimeError("strict_non_exemplar is disabled.")
        for class_id in self.finalized_classes:
            if self._is_train_access_allowed(class_id):
                raise RuntimeError(f"Finalized class {class_id} still has train access.")
        if self._purged_train_indices:
            indices = np.asarray(sorted(self._purged_train_indices), dtype=np.int64)
            if np.any(self.patches[indices] != 0.0):
                raise RuntimeError("Purged old training patches remain in memory.")
            if self._spectral_positive_centers is not None and np.any(
                self._spectral_positive_centers[indices] != 0.0
            ):
                raise RuntimeError("Purged old positive response centers remain in memory.")
            if self._spectral_negative_centers is not None and np.any(
                self._spectral_negative_centers[indices] != 0.0
            ):
                raise RuntimeError("Purged old negative response centers remain in memory.")
        self.assert_preprocess_fit_indices()
        return True

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _get_split_indices(self, split: str) -> np.ndarray:
        token = str(split).lower()
        if token == "train":
            return self.train_indices
        if token == "val":
            return self.val_indices
        if token == "test":
            return self.test_indices
        if token in {"all", "full", "labeled", "all_labeled"}:
            return np.arange(len(self.remapped_labels), dtype=np.int64)
        raise ValueError(f"Unknown split {split!r}; use train, val, test, or all.")

    def _assert_indices_match_classes(
        self, indices: np.ndarray, class_ids: Sequence[int], *, context: str
    ) -> None:
        if indices.size == 0:
            return
        bad = sorted(
            set(int(v) for v in self.remapped_labels[indices].tolist())
            - set(int(v) for v in class_ids)
        )
        if bad:
            raise RuntimeError(f"{context}: indices include disallowed labels {bad}.")

    def _make_loader(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        *,
        balanced: bool = False,
        active_classes: Optional[List[int]] = None,
    ) -> DataLoader:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("Requested DataLoader has zero samples.")
        if not self.has_response_views():
            raise RuntimeError("PC-SIRG loader cannot run without paired response views.")
        assert self._spectral_positive_centers is not None
        assert self._spectral_negative_centers is not None
        assert self._spectral_step_sizes is not None
        dataset = HSIPatchDataset(
            self.patches[idx],
            self.remapped_labels[idx],
            coords=self.coords[idx],
            sample_indices=idx,
            spectral_positive_centers=self._spectral_positive_centers[idx],
            spectral_negative_centers=self._spectral_negative_centers[idx],
            spectral_step_sizes=self._spectral_step_sizes[idx],
            intervention_definition_version=self.intervention_definition_version,
        )
        common = {
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
        }
        if balanced and int(dataset.labels.unique().numel()) > 1:
            present = [int(v) for v in dataset.labels.unique().tolist()]
            classes_per_batch = self.geometry_batch_classes or (
                len(active_classes) if active_classes is not None else len(present)
            )
            classes_per_batch = max(2, min(int(classes_per_batch), len(present), int(batch_size)))
            samples_per_class = self.geometry_samples_per_class or max(
                2, int(batch_size) // classes_per_batch
            )
            sampler = ClassBalancedBatchSampler(
                labels=dataset.labels,
                batch_size=int(batch_size),
                classes_per_batch=classes_per_batch,
                samples_per_class=int(samples_per_class),
                seed=self.seed + self.current_phase * 1009,
                drop_last=False,
                require_unique_within_batch=self.strict_unique_balanced_batches,
            )
            return DataLoader(dataset, batch_sampler=sampler, **common)
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=False,
            **common,
        )

    def get_phase_dataloader(
        self,
        phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        phase = self._validate_phase(phase)
        split_name = str(split).lower()
        active = self.get_phase_classes(phase)
        if split_name == "train" and self.strict_non_exemplar:
            if phase != self.current_phase or phase in self.finalized_phases:
                raise PermissionError(
                    "Raw train access is restricted to the active unfinalized phase."
                )
        if split_name in {"all", "full", "labeled", "all_labeled"}:
            for class_id in active:
                self._check_class_split_access(class_id, "all")
        indices = self._get_split_indices(split_name)
        idx = indices[np.isin(self.remapped_labels[indices], active)]
        self._assert_indices_match_classes(
            idx, active, context=f"get_phase_dataloader({split_name}, phase={phase})"
        )
        balanced = (
            split_name == "train"
            and self.class_balanced_train_batches
            and bool(shuffle)
        )
        return self._make_loader(
            idx,
            batch_size,
            bool(shuffle) if split_name == "train" else False,
            balanced=balanced,
            active_classes=active,
        )

    def get_class_balanced_phase_loader(
        self,
        phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        previous = self.class_balanced_train_batches
        self.class_balanced_train_batches = True
        try:
            return self.get_phase_dataloader(phase, split, batch_size, shuffle)
        finally:
            self.class_balanced_train_batches = previous

    def get_cumulative_dataloader(
        self,
        up_to_phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
        allow_train_old: bool = False,
    ) -> DataLoader:
        phase = self._validate_phase(up_to_phase)
        split_name = str(split).lower()
        if split_name == "train" and self.strict_non_exemplar:
            if allow_train_old:
                raise PermissionError("Old raw training samples cannot be exposed.")
            if phase != self.current_phase or phase in self.finalized_phases:
                raise PermissionError(
                    "Train access is restricted to the active unfinalized phase."
                )
            active = self.get_new_classes(self.current_phase)
        else:
            active = self.get_seen_classes(phase)
        if split_name in {"all", "full", "labeled", "all_labeled"}:
            for class_id in active:
                self._check_class_split_access(class_id, "all")
        indices = self._get_split_indices(split_name)
        idx = indices[np.isin(self.remapped_labels[indices], active)]
        self._assert_indices_match_classes(
            idx, active, context=f"get_cumulative_dataloader({split_name}, phase={phase})"
        )
        balanced = (
            split_name == "train"
            and self.class_balanced_train_batches
            and bool(shuffle)
        )
        return self._make_loader(
            idx,
            batch_size,
            bool(shuffle) if split_name == "train" else False,
            balanced=balanced,
            active_classes=active,
        )

    def get_class_indices(self, class_id: int, split: str = "train") -> np.ndarray:
        class_id = int(class_id)
        self._check_class_split_access(class_id, split)
        indices = self._get_split_indices(split)
        return indices[self.remapped_labels[indices] == class_id].copy()

    def get_class_patches(self, class_id: int, split: str = "train") -> np.ndarray:
        indices = self.get_class_indices(class_id, split)
        if indices.size == 0:
            raise ValueError(f"No samples for class={class_id}, split={split!r}.")
        return self.patches[indices].copy()

    def get_class_counts(self) -> Dict[int, int]:
        return {
            class_id: int(np.sum(self.remapped_labels == class_id))
            for class_id in range(self.num_classes)
        }

    def get_split_class_counts(self, split: str = "train") -> Dict[int, int]:
        indices = self._get_split_indices(split)
        return {
            class_id: int(np.sum(self.remapped_labels[indices] == class_id))
            for class_id in range(self.num_classes)
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_label_policy_summary(self) -> Dict[str, Any]:
        return {
            "has_background": self.has_background,
            "background_label": self.background_label,
            "raw_class_values": list(self.raw_class_values),
            "class_order": list(self.class_order),
            "label_map": dict(self.label_map),
            "inv_label_map": dict(self.inv_label_map),
            "display_contract": "display 0 is background; semantic class c uses display c+1",
        }

    def protocol_state(self) -> Dict[str, Any]:
        return {
            "current_phase": int(self.current_phase),
            "strict_non_exemplar": bool(self.strict_non_exemplar),
            "finalized_phases": sorted(self.finalized_phases),
            "finalized_classes": sorted(self.finalized_classes),
            "accessible_train_classes": self.get_accessible_train_classes(),
            "phase_class_splits": self.get_phase_class_splits(self.current_phase),
            "classifier_width": self.classifier_num_classes_for_phase(self.current_phase),
            "response_contract": self.get_response_contract(),
            "purged_train_indices_count": len(self._purged_train_indices),
            "runtime_state": self.export_runtime_state(),
        }

    def _print_stats(self) -> None:
        print("[IncrementalHSIDataset] PC-SIRG protocol initialized")
        print(
            f"  classes={self.num_classes} | phases={self.num_phases} | "
            f"strict_non_exemplar={self.strict_non_exemplar}"
        )
        print(f"  class_order={self.class_order}")
        print(f"  phase_to_classes={self.phase_to_classes}")
        print(
            f"  split sizes: train={len(self.train_indices)} | "
            f"val={len(self.val_indices)} | test={len(self.test_indices)}"
        )
        print(
            f"  paired responses: available={self.has_response_views()} | "
            f"K={self.num_spectral_interventions} | "
            f"version={self.intervention_definition_version} | center_only=True"
        )
        print(
            "  persistent training payload after finalization: none; "
            "only aggregate GeometryBank rows may survive"
        )







# from __future__ import annotations

# from contextlib import contextmanager
# import hashlib
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset, Sampler


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     """Return unique ints while preserving order.

#     CIL class order is part of the protocol. Do not replace this with
#     ``sorted(set(values))`` in trainer/evaluator code; that silently changes
#     shuffled class-order experiments.
#     """
#     out: List[int] = []
#     seen: Set[int] = set()
#     for value in values:
#         v = int(value)
#         if v not in seen:
#             out.append(v)
#             seen.add(v)
#     return out


# def _as_1d_int_array(values: Any, *, name: str) -> np.ndarray:
#     arr = np.asarray(values, dtype=np.int64).reshape(-1)
#     if arr.size == 0:
#         raise ValueError(f"{name} is empty.")
#     return arr


# _PROTOCOL_PLAN_VERSION = 2


# def _labels_sha256(labels: np.ndarray) -> str:
#     labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64).reshape(-1))
#     return hashlib.sha256(labels.tobytes()).hexdigest()


# def _array_sha256(values: np.ndarray) -> str:
#     array = np.ascontiguousarray(values)
#     return hashlib.sha256(array.tobytes()).hexdigest()


# _PHYSICAL_SPECTRAL_SPACES = {
#     "raw_wavelength",
#     "sensor_corrected_wavelength",
# }
# _ALLOWED_SPECTRAL_SPACES = _PHYSICAL_SPECTRAL_SPACES | {
#     "base_train_standardized_wavelength",
#     "pca_components",
#     "unknown",
# }


# def _validate_split_configuration(
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     min_train_per_class: int,
# ) -> None:
#     train_ratio = float(train_ratio)
#     val_ratio = float(val_ratio)
#     min_train_per_class = int(min_train_per_class)
#     if not 0.0 < train_ratio < 1.0:
#         raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}.")
#     if not 0.0 <= val_ratio < 1.0:
#         raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}.")
#     if train_ratio + val_ratio >= 1.0:
#         raise ValueError(
#             "train_ratio + val_ratio must be < 1 so a test partition remains; "
#             f"got {train_ratio + val_ratio:.6f}."
#         )
#     if min_train_per_class <= 0:
#         raise ValueError(
#             f"min_train_per_class must be positive, got {min_train_per_class}."
#         )


# def _split_one_class(
#     class_indices: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     min_train_per_class: int,
#     seed: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Create one deterministic classwise split without sample leakage."""
#     indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
#     if indices.size == 0:
#         return (
#             np.empty((0,), dtype=np.int64),
#             np.empty((0,), dtype=np.int64),
#             np.empty((0,), dtype=np.int64),
#         )

#     rng = np.random.RandomState(int(seed))
#     shuffled = rng.permutation(indices)
#     n_samples = int(shuffled.size)

#     if n_samples == 1:
#         return shuffled.copy(), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
#     if n_samples == 2:
#         return shuffled[:1].copy(), np.empty((0,), dtype=np.int64), shuffled[1:].copy()

#     requested_train = max(
#         int(min_train_per_class),
#         int(round(n_samples * float(train_ratio))),
#         1,
#     )
#     requested_val = (
#         max(1, int(round(n_samples * float(val_ratio))))
#         if float(val_ratio) > 0.0
#         else 0
#     )

#     # A non-empty test set is mandatory. A validation sample is mandatory only
#     # when val_ratio > 0. This clips an infeasible min_train_per_class request
#     # rather than silently deleting validation/test samples.
#     max_train = n_samples - requested_val - 1
#     if max_train < 1 and requested_val > 0:
#         requested_val = max(0, n_samples - 2)
#         max_train = n_samples - requested_val - 1
#     n_train = max(1, min(requested_train, max_train))

#     remaining_after_train = n_samples - n_train
#     n_val = min(requested_val, max(0, remaining_after_train - 1))
#     n_test = n_samples - n_train - n_val
#     if n_test < 1:
#         raise RuntimeError(
#             "Internal split error: test partition is empty for a class with "
#             f"n={n_samples}, train={n_train}, val={n_val}."
#         )

#     return (
#         shuffled[:n_train].copy(),
#         shuffled[n_train : n_train + n_val].copy(),
#         shuffled[n_train + n_val :].copy(),
#     )


# def _jsonable_protocol_value(value: Any) -> Any:
#     if isinstance(value, np.ndarray):
#         return value.tolist()
#     if isinstance(value, (np.integer, np.floating, np.bool_)):
#         return value.item()
#     if isinstance(value, Mapping):
#         return {str(key): _jsonable_protocol_value(item) for key, item in value.items()}
#     if isinstance(value, (list, tuple)):
#         return [_jsonable_protocol_value(item) for item in value]
#     if isinstance(value, (str, int, float, bool)) or value is None:
#         return value
#     return str(value)


# def build_incremental_protocol_plan(
#     labels: np.ndarray,
#     *,
#     base_classes: int,
#     increment: int,
#     train_ratio: float = 0.20,
#     val_ratio: float = 0.10,
#     min_train_per_class: int = 20,
#     seed: int = 42,
#     shuffle_order: bool = False,
#     class_order: Optional[Sequence[int]] = None,
# ) -> Dict[str, Any]:
#     """Build the complete CIL class/split plan before PCA or patch extraction.

#     ``labels`` must follow the same labeled-pixel order later used by
#     ``ImageCubes``. The strict loader's ``ExtractLabeledPixelIndex`` provides
#     exactly that ordering. The returned base-training indices can therefore be
#     converted directly into a preprocessing fit mask.
#     """
#     labels = _as_1d_int_array(labels, name="labels")
#     if int(labels.min()) < 0:
#         raise ValueError(f"labels must be non-negative, got min={int(labels.min())}.")

#     _validate_split_configuration(
#         train_ratio=float(train_ratio),
#         val_ratio=float(val_ratio),
#         min_train_per_class=int(min_train_per_class),
#     )

#     input_classes = sorted(int(value) for value in np.unique(labels).tolist())
#     num_classes = len(input_classes)
#     base_classes = int(base_classes)
#     increment = int(increment)
#     if not 2 <= base_classes <= num_classes:
#         raise ValueError(
#             f"base_classes must be in [2,{num_classes}], got {base_classes}."
#         )
#     if increment <= 0:
#         raise ValueError(f"increment must be positive, got {increment}.")

#     if class_order is not None:
#         resolved_order = [int(value) for value in class_order]
#         if len(resolved_order) != len(set(resolved_order)):
#             raise ValueError(f"class_order contains duplicates: {resolved_order}.")
#         if set(resolved_order) != set(input_classes):
#             raise ValueError(
#                 "class_order must be a permutation of the labels present in the "
#                 f"dataset. order={resolved_order}, present={input_classes}."
#             )
#     elif bool(shuffle_order):
#         rng = np.random.RandomState(int(seed))
#         resolved_order = [int(value) for value in rng.permutation(input_classes).tolist()]
#     else:
#         resolved_order = list(input_classes)

#     label_map = {int(global_id): int(seq_id) for seq_id, global_id in enumerate(resolved_order)}
#     inv_label_map = {int(seq_id): int(global_id) for global_id, seq_id in label_map.items()}
#     remapped_labels = np.fromiter(
#         (label_map[int(value)] for value in labels.tolist()),
#         dtype=np.int64,
#         count=int(labels.size),
#     )

#     remaining = num_classes - base_classes
#     num_phases = 1 + int(np.ceil(remaining / float(increment))) if remaining > 0 else 1
#     phase_to_classes: Dict[int, List[int]] = {}
#     class_to_phase: Dict[int, int] = {}
#     for phase in range(num_phases):
#         if phase == 0:
#             phase_classes = list(range(base_classes))
#         else:
#             start = base_classes + (phase - 1) * increment
#             stop = min(start + increment, num_classes)
#             phase_classes = list(range(start, stop))
#         if not phase_classes:
#             raise RuntimeError(f"Protocol construction created an empty phase {phase}.")
#         phase_to_classes[int(phase)] = phase_classes
#         for class_id in phase_classes:
#             class_to_phase[int(class_id)] = int(phase)

#     train_parts: List[np.ndarray] = []
#     val_parts: List[np.ndarray] = []
#     test_parts: List[np.ndarray] = []
#     split_class_counts: Dict[int, Dict[str, int]] = {}
#     for seq_id in range(num_classes):
#         class_indices = np.flatnonzero(remapped_labels == seq_id).astype(np.int64)
#         train_idx, val_idx, test_idx = _split_one_class(
#             class_indices,
#             train_ratio=float(train_ratio),
#             val_ratio=float(val_ratio),
#             min_train_per_class=int(min_train_per_class),
#             seed=int(seed) + int(seq_id),
#         )
#         train_parts.append(train_idx)
#         val_parts.append(val_idx)
#         test_parts.append(test_idx)
#         split_class_counts[int(seq_id)] = {
#             "all": int(class_indices.size),
#             "train": int(train_idx.size),
#             "val": int(val_idx.size),
#             "test": int(test_idx.size),
#         }

#     def concatenate(parts: Sequence[np.ndarray]) -> np.ndarray:
#         nonempty = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts if len(part)]
#         return (
#             np.concatenate(nonempty).astype(np.int64, copy=False)
#             if nonempty
#             else np.empty((0,), dtype=np.int64)
#         )

#     train_indices = concatenate(train_parts)
#     val_indices = concatenate(val_parts)
#     test_indices = concatenate(test_parts)
#     all_split = concatenate([train_indices, val_indices, test_indices])
#     if all_split.size != labels.size or np.unique(all_split).size != labels.size:
#         raise RuntimeError(
#             "Protocol split does not cover every labeled sample exactly once: "
#             f"covered={all_split.size}, unique={np.unique(all_split).size}, total={labels.size}."
#         )

#     base_class_ids = phase_to_classes[0]
#     base_mask = np.isin(remapped_labels[train_indices], np.asarray(base_class_ids, dtype=np.int64))
#     base_train_indices = train_indices[base_mask].astype(np.int64, copy=True)
#     if base_train_indices.size == 0:
#         raise RuntimeError("Base protocol contains no base-training samples.")
#     bad_base = sorted(
#         set(int(value) for value in remapped_labels[base_train_indices].tolist())
#         - set(base_class_ids)
#     )
#     if bad_base:
#         raise RuntimeError(f"Base preprocessing indices contain future classes: {bad_base}.")

#     return {
#         "version": int(_PROTOCOL_PLAN_VERSION),
#         "num_samples": int(labels.size),
#         "labels_sha256": _labels_sha256(labels),
#         "input_classes": input_classes,
#         "num_classes": int(num_classes),
#         "class_order": resolved_order,
#         "label_map": label_map,
#         "inv_label_map": inv_label_map,
#         "remapped_labels": remapped_labels,
#         "base_classes": int(base_classes),
#         "increment": int(increment),
#         "num_phases": int(num_phases),
#         "phase_to_classes": phase_to_classes,
#         "class_to_phase": class_to_phase,
#         "train_indices": train_indices,
#         "val_indices": val_indices,
#         "test_indices": test_indices,
#         "base_train_indices": base_train_indices,
#         "split_class_counts": split_class_counts,
#         "split_policy": {
#             "train_ratio": float(train_ratio),
#             "val_ratio": float(val_ratio),
#             "min_train_per_class": int(min_train_per_class),
#             "seed": int(seed),
#             "shuffle_order": bool(shuffle_order),
#             "explicit_class_order": class_order is not None,
#         },
#     }


# def _normalize_int_key_mapping(value: Mapping[Any, Any]) -> Dict[int, Any]:
#     return {int(key): item for key, item in dict(value).items()}


# def validate_incremental_protocol_plan(
#     plan: Mapping[str, Any],
#     labels: np.ndarray,
#     *,
#     expected_base_classes: Optional[int] = None,
#     expected_increment: Optional[int] = None,
# ) -> Dict[str, Any]:
#     """Normalize and validate a plan before it is attached to patch tensors."""
#     if not isinstance(plan, Mapping):
#         raise TypeError(f"protocol_plan must be a mapping, got {type(plan)}.")
#     labels = _as_1d_int_array(labels, name="labels")
#     normalized = dict(plan)

#     required = {
#         "version", "num_samples", "labels_sha256", "input_classes",
#         "num_classes", "class_order", "label_map", "inv_label_map",
#         "base_classes", "increment", "num_phases", "phase_to_classes",
#         "class_to_phase", "train_indices", "val_indices", "test_indices",
#         "base_train_indices", "split_policy",
#     }
#     missing = sorted(required - set(normalized))
#     if missing:
#         raise ValueError(f"protocol_plan is missing required fields: {missing}.")

#     if int(normalized["num_samples"]) != int(labels.size):
#         raise ValueError(
#             "protocol_plan sample count does not match labels: "
#             f"plan={normalized['num_samples']}, labels={labels.size}."
#         )
#     if str(normalized["labels_sha256"]) != _labels_sha256(labels):
#         raise ValueError(
#             "protocol_plan labels fingerprint does not match the supplied label "
#             "array. The plan and ImageCubes ordering are not the same."
#         )

#     normalized["input_classes"] = [int(value) for value in normalized["input_classes"]]
#     normalized["class_order"] = [int(value) for value in normalized["class_order"]]
#     normalized["label_map"] = {
#         int(key): int(value) for key, value in _normalize_int_key_mapping(normalized["label_map"]).items()
#     }
#     normalized["inv_label_map"] = {
#         int(key): int(value) for key, value in _normalize_int_key_mapping(normalized["inv_label_map"]).items()
#     }
#     normalized["phase_to_classes"] = {
#         int(key): [int(value) for value in values]
#         for key, values in _normalize_int_key_mapping(normalized["phase_to_classes"]).items()
#     }
#     normalized["class_to_phase"] = {
#         int(key): int(value)
#         for key, value in _normalize_int_key_mapping(normalized["class_to_phase"]).items()
#     }
#     for name in ("train_indices", "val_indices", "test_indices", "base_train_indices"):
#         normalized[name] = np.asarray(normalized[name], dtype=np.int64).reshape(-1)

#     base_classes = int(normalized["base_classes"])
#     increment = int(normalized["increment"])
#     if expected_base_classes is not None and base_classes != int(expected_base_classes):
#         raise ValueError(
#             f"protocol_plan base_classes={base_classes} does not match requested "
#             f"base_classes={int(expected_base_classes)}."
#         )
#     if expected_increment is not None and increment != int(expected_increment):
#         raise ValueError(
#             f"protocol_plan increment={increment} does not match requested "
#             f"increment={int(expected_increment)}."
#         )

#     num_classes = int(normalized["num_classes"])
#     if set(normalized["class_order"]) != set(normalized["input_classes"]):
#         raise ValueError("protocol_plan class_order is not a permutation of input_classes.")
#     if len(normalized["class_order"]) != num_classes:
#         raise ValueError("protocol_plan class_order length does not match num_classes.")

#     recomputed = np.fromiter(
#         (normalized["label_map"][int(value)] for value in labels.tolist()),
#         dtype=np.int64,
#         count=int(labels.size),
#     )
#     if "remapped_labels" in normalized:
#         stored = np.asarray(normalized["remapped_labels"], dtype=np.int64).reshape(-1)
#         if not np.array_equal(stored, recomputed):
#             raise ValueError("protocol_plan remapped_labels do not match label_map.")
#     normalized["remapped_labels"] = recomputed

#     expected_phases = list(range(int(normalized["num_phases"])))
#     if sorted(normalized["phase_to_classes"]) != expected_phases:
#         raise ValueError(
#             "protocol_plan phase keys must be contiguous: "
#             f"got={sorted(normalized['phase_to_classes'])}, expected={expected_phases}."
#         )
#     phase_union = [
#         class_id
#         for phase in expected_phases
#         for class_id in normalized["phase_to_classes"][phase]
#     ]
#     if phase_union != list(range(num_classes)):
#         raise ValueError(
#             "protocol_plan phases must cover sequential class IDs exactly once: "
#             f"got={phase_union}."
#         )

#     all_split = np.concatenate(
#         [normalized["train_indices"], normalized["val_indices"], normalized["test_indices"]]
#     )
#     if all_split.size != labels.size or np.unique(all_split).size != labels.size:
#         raise ValueError("protocol_plan train/val/test indices do not partition the samples.")
#     if all_split.size and (int(all_split.min()) < 0 or int(all_split.max()) >= labels.size):
#         raise ValueError("protocol_plan contains indices outside the supplied label array.")

#     train_set = set(int(value) for value in normalized["train_indices"].tolist())
#     if not set(int(value) for value in normalized["base_train_indices"].tolist()).issubset(train_set):
#         raise ValueError("base_train_indices are not a subset of train_indices.")
#     base_ids = set(normalized["phase_to_classes"][0])
#     bad = sorted(
#         set(int(value) for value in recomputed[normalized["base_train_indices"]].tolist()) - base_ids
#     )
#     if bad:
#         raise ValueError(f"base_train_indices contain future classes: {bad}.")

#     return normalized


# class HSIPatchDataset(Dataset):
#     def __init__(
#         self,
#         patches: np.ndarray,
#         labels: np.ndarray,
#         coords: Optional[np.ndarray] = None,
#         return_metadata: bool = False,
#         center_spectra: Optional[np.ndarray] = None,
#         spectra_are_physical: bool = False,
#         return_input_center_when_no_spectra: bool = False,
#         spectral_augmented_patches: Optional[np.ndarray] = None,
#     ):
#         patches = np.ascontiguousarray(patches, dtype=np.float32)
#         labels = np.asarray(labels, dtype=np.int64).reshape(-1)

#         if len(patches) != len(labels):
#             raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

#         self.patches = torch.from_numpy(patches).float()
#         self.labels = torch.from_numpy(labels).long()

#         if spectral_augmented_patches is None:
#             self.spectral_augmented_patches = None
#         else:
#             augmented = np.ascontiguousarray(spectral_augmented_patches, dtype=np.float32)
#             if augmented.shape != patches.shape:
#                 raise ValueError(
#                     "spectral_augmented_patches must exactly match patches shape: "
#                     f"augmented={augmented.shape}, patches={patches.shape}"
#                 )
#             if not np.isfinite(augmented).all():
#                 raise ValueError("spectral_augmented_patches contain NaN or Inf values")
#             self.spectral_augmented_patches = torch.from_numpy(augmented).float()

#         self.return_metadata = bool(return_metadata)
#         self.spectra_are_physical = bool(spectra_are_physical)
#         self.return_input_center_when_no_spectra = bool(return_input_center_when_no_spectra)

#         if center_spectra is None:
#             self.center_spectra = None
#         else:
#             center_spectra = np.asarray(center_spectra, dtype=np.float32)
#             if len(center_spectra) != len(labels):
#                 raise ValueError(f"center_spectra/label length mismatch: {len(center_spectra)} vs {len(labels)}")
#             if center_spectra.ndim != 2:
#                 center_spectra = center_spectra.reshape(len(labels), -1)
#             self.center_spectra = torch.from_numpy(np.ascontiguousarray(center_spectra)).float()

#         if coords is None:
#             self.coords = torch.empty((len(labels), 0), dtype=torch.long)
#         else:
#             coords = np.asarray(coords, dtype=np.int64)
#             if len(coords) != len(labels):
#                 raise ValueError(f"coord/label length mismatch: {len(coords)} vs {len(labels)}")
#             if coords.ndim != 2 or coords.shape[1] != 2:
#                 raise ValueError(f"coords must be [N,2], got {coords.shape}")
#             self.coords = torch.from_numpy(coords).long()

#     def __len__(self) -> int:
#         return int(len(self.labels))

#     @staticmethod
#     def _center_spectrum(patch: torch.Tensor) -> torch.Tensor:
#         # Patches are normally [C,H,W]. Fallbacks keep compatibility with flat inputs.
#         if patch.dim() == 3:
#             return patch[:, patch.size(1) // 2, patch.size(2) // 2].float()
#         if patch.dim() == 2:
#             return patch.float().mean(dim=-1)
#         return patch.float().flatten()

#     def __getitem__(self, idx: int):
#         patch = self.patches[idx]
#         label = self.labels[idx]
#         if not self.return_metadata:
#             return patch, label
#         # Return physical wavelength-ordered center spectra only when the
#         # IncrementalHSIDataset explicitly provides them.  Do NOT silently fall
#         # back to PCA/reduced patch-center spectra here: several trainer paths
#         # interpret the third tuple field as raw spectral metadata.  Returning an
#         # empty tensor is safer; the model/helper can still derive a non-physical
#         # reduced center vector from the patch when needed.
#         if self.center_spectra is not None:
#             spectrum = self.center_spectra[idx]
#         elif self.return_input_center_when_no_spectra:
#             spectrum = self._center_spectrum(patch)
#         else:
#             spectrum = torch.empty((0,), dtype=patch.dtype)
#         coord = self.coords[idx] if self.coords.numel() > 0 else torch.empty((0,), dtype=torch.long)
#         if self.spectral_augmented_patches is not None:
#             spectral_view = self.spectral_augmented_patches[idx]
#             return patch, label, spectrum, coord, spectral_view
#         return patch, label, spectrum, coord


# class ClassBalancedBatchSampler(Sampler[List[int]]):
#     """Deterministic cyclic class-balanced sampler.

#     Every class receives regular batch exposure. Samples are cycled and
#     reshuffled only after a class pool is exhausted, so minority replacement is
#     explicit rather than hidden in unconstrained random sampling.
#     """

#     def __init__(
#         self,
#         labels: torch.Tensor,
#         batch_size: int,
#         classes_per_batch: Optional[int] = None,
#         samples_per_class: Optional[int] = None,
#         seed: int = 42,
#         drop_last: bool = False,
#     ) -> None:
#         labels_np = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
#         if labels_np.size == 0:
#             raise ValueError("ClassBalancedBatchSampler received zero labels.")

#         self.labels_np = labels_np
#         self.batch_size = int(batch_size)
#         self.seed = int(seed)
#         self.drop_last = bool(drop_last)
#         if self.batch_size <= 0:
#             raise ValueError(f"batch_size must be positive, got {self.batch_size}.")

#         self.class_to_indices: Dict[int, np.ndarray] = {
#             int(class_id): np.flatnonzero(labels_np == int(class_id)).astype(np.int64)
#             for class_id in sorted(np.unique(labels_np).tolist())
#         }
#         self.class_to_indices = {
#             class_id: indices
#             for class_id, indices in self.class_to_indices.items()
#             if indices.size > 0
#         }
#         self.classes = list(self.class_to_indices)
#         if not self.classes:
#             raise ValueError("ClassBalancedBatchSampler found no classes.")

#         if classes_per_batch is None or int(classes_per_batch) <= 0:
#             classes_per_batch = min(len(self.classes), self.batch_size)
#         self.classes_per_batch = int(
#             max(1, min(int(classes_per_batch), len(self.classes), self.batch_size))
#         )

#         if samples_per_class is None or int(samples_per_class) <= 0:
#             samples_per_class = max(1, self.batch_size // self.classes_per_batch)
#         self.samples_per_class = int(max(1, int(samples_per_class)))
#         if self.classes_per_batch * self.samples_per_class > self.batch_size:
#             self.samples_per_class = max(1, self.batch_size // self.classes_per_batch)

#         self.effective_batch_size = self.classes_per_batch * self.samples_per_class
#         if self.effective_batch_size <= 0:
#             raise RuntimeError("Resolved balanced batch size is zero.")

#         if self.drop_last:
#             self.num_batches = labels_np.size // self.effective_batch_size
#         else:
#             self.num_batches = int(np.ceil(labels_np.size / float(self.effective_batch_size)))
#         self.num_batches = max(1, int(self.num_batches))
#         self.epoch = 0

#     def set_epoch(self, epoch: int) -> None:
#         self.epoch = int(epoch)

#     def __len__(self) -> int:
#         return int(self.num_batches)

#     def __iter__(self):
#         rng = np.random.RandomState(self.seed + 1009 * self.epoch)
#         class_order = np.asarray(self.classes, dtype=np.int64)
#         rng.shuffle(class_order)
#         class_cursor = 0

#         pools: Dict[int, np.ndarray] = {}
#         pointers: Dict[int, int] = {}
#         for class_id, source in self.class_to_indices.items():
#             pools[class_id] = rng.permutation(source)
#             pointers[class_id] = 0

#         def draw(class_id: int, count: int) -> List[int]:
#             selected: List[int] = []
#             while len(selected) < count:
#                 pool = pools[class_id]
#                 pointer = pointers[class_id]
#                 available = int(pool.size - pointer)
#                 take = min(count - len(selected), available)
#                 if take > 0:
#                     selected.extend(int(value) for value in pool[pointer : pointer + take].tolist())
#                     pointers[class_id] = pointer + take
#                 if pointers[class_id] >= pool.size:
#                     pools[class_id] = rng.permutation(self.class_to_indices[class_id])
#                     pointers[class_id] = 0
#             return selected

#         for _ in range(self.num_batches):
#             if self.classes_per_batch == len(class_order):
#                 chosen = class_order.tolist()
#             else:
#                 if class_cursor + self.classes_per_batch > len(class_order):
#                     rng.shuffle(class_order)
#                     class_cursor = 0
#                 chosen = class_order[
#                     class_cursor : class_cursor + self.classes_per_batch
#                 ].tolist()
#                 class_cursor += self.classes_per_batch

#             batch: List[int] = []
#             for class_id in chosen:
#                 batch.extend(draw(int(class_id), self.samples_per_class))
#             rng.shuffle(batch)
#             if self.drop_last and len(batch) < self.effective_batch_size:
#                 continue
#             yield batch


# class IncrementalHSIDataset:
#     """
#     Strict non-exemplar incremental dataset manager.

#     The class/split protocol can be built before PCA through
#     ``build_protocol_plan`` and then injected through ``protocol_plan``. This
#     guarantees that normalization/PCA is fitted on exactly the phase-0 training
#     pixels and that the same indices are reused after patch extraction.

#     Label 0 is always a valid sequential foreground class here. Raw background
#     removal belongs to the HSI loader.
#     """

#     @staticmethod
#     def build_protocol_plan(
#         labels: np.ndarray,
#         *,
#         base_classes: int,
#         increment: int,
#         train_ratio: float = 0.20,
#         val_ratio: float = 0.10,
#         min_train_per_class: int = 20,
#         seed: int = 42,
#         shuffle_order: bool = False,
#         class_order: Optional[Sequence[int]] = None,
#     ) -> Dict[str, Any]:
#         return build_incremental_protocol_plan(
#             labels,
#             base_classes=int(base_classes),
#             increment=int(increment),
#             train_ratio=float(train_ratio),
#             val_ratio=float(val_ratio),
#             min_train_per_class=int(min_train_per_class),
#             seed=int(seed),
#             shuffle_order=bool(shuffle_order),
#             class_order=class_order,
#         )

#     def __init__(
#         self,
#         patches: np.ndarray,
#         labels: np.ndarray,
#         coords: np.ndarray,
#         gt_shape: Tuple[int, int],
#         GT: np.ndarray,
#         base_classes: int,
#         increment: int,
#         train_ratio: float = 0.2,
#         val_ratio: float = 0.1,
#         seed: int = 42,
#         shuffle_order: bool = False,
#         device: str = "cuda",
#         min_train_per_class: int = 20,
#         num_workers: int = 0,
#         strict_non_exemplar: bool = True,
#         target_names: Optional[List[str]] = None,
#         label_policy: Optional[Dict] = None,
#         class_balanced_train_batches: bool = True,
#         geometry_batch_classes: int = 0,
#         geometry_samples_per_class: int = 0,
#         return_metadata: bool = False,
#         raw_spectra: Optional[np.ndarray] = None,
#         center_spectra: Optional[np.ndarray] = None,
#         spectra_are_physical: bool = False,
#         spectral_augmented_patches: Optional[np.ndarray] = None,
#         spectral_view_patches: Optional[np.ndarray] = None,
#         spectral_metadata: Optional[Mapping[str, Any]] = None,
#         spectral_consistency_enabled: bool = False,
#         purge_finalized_train_payload: bool = False,
#         protocol_plan: Optional[Mapping[str, Any]] = None,
#         class_order: Optional[Sequence[int]] = None,
#     ):
#         self.patches = np.ascontiguousarray(patches, dtype=np.float32).copy()
#         self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
#         self.coords = np.asarray(coords, dtype=np.int64)
#         self.gt_shape = tuple(int(value) for value in gt_shape)
#         self.GT = np.asarray(GT, dtype=np.int64)

#         if len(self.patches) != len(self.labels):
#             raise ValueError(
#                 f"patch/label length mismatch: {len(self.patches)} vs {len(self.labels)}"
#             )
#         if len(self.coords) != len(self.labels):
#             raise ValueError(
#                 f"coord/label length mismatch: {len(self.coords)} vs {len(self.labels)}"
#             )
#         if self.coords.ndim != 2 or self.coords.shape[1] != 2:
#             raise ValueError(f"coords must be [N,2], got {self.coords.shape}.")
#         if self.labels.size == 0:
#             raise ValueError("Empty labels passed to IncrementalHSIDataset.")
#         if int(self.labels.min()) < 0:
#             raise ValueError(
#                 f"Negative labels passed to IncrementalHSIDataset: min={int(self.labels.min())}. "
#                 "The loader must remove background/ignore labels first."
#             )
#         if len(self.gt_shape) != 2 or tuple(self.GT.shape[:2]) != self.gt_shape:
#             raise ValueError(
#                 f"gt_shape={self.gt_shape} does not match GT spatial shape={self.GT.shape[:2]}."
#             )
#         if self.coords.size:
#             rows, columns = self.coords[:, 0], self.coords[:, 1]
#             if (
#                 int(rows.min()) < 0
#                 or int(columns.min()) < 0
#                 or int(rows.max()) >= self.gt_shape[0]
#                 or int(columns.max()) >= self.gt_shape[1]
#             ):
#                 raise ValueError("coords contain values outside gt_shape.")

#         if spectral_augmented_patches is not None and spectral_view_patches is not None:
#             first = np.asarray(spectral_augmented_patches)
#             second = np.asarray(spectral_view_patches)
#             if first.shape != second.shape or not np.array_equal(first, second):
#                 raise ValueError(
#                     "spectral_augmented_patches and spectral_view_patches were both "
#                     "provided but do not contain the same aligned view"
#                 )
#         augmented_source = (
#             spectral_augmented_patches
#             if spectral_augmented_patches is not None
#             else spectral_view_patches
#         )
#         self._spectral_augmented_patches: Optional[np.ndarray] = None
#         if augmented_source is not None:
#             augmented = np.ascontiguousarray(augmented_source, dtype=np.float32)
#             if augmented.shape != self.patches.shape:
#                 raise ValueError(
#                     "Processed spectral view must exactly match model patches: "
#                     f"view={augmented.shape}, patches={self.patches.shape}"
#                 )
#             if not np.isfinite(augmented).all():
#                 raise ValueError("Processed spectral view contains NaN or Inf values")
#             self._spectral_augmented_patches = augmented.copy()

#         self.base_classes = int(base_classes)
#         self.increment = int(increment)
#         self.train_ratio = float(train_ratio)
#         self.val_ratio = float(val_ratio)
#         self.seed = int(seed)
#         self.device = str(device)
#         self.min_train_per_class = int(min_train_per_class)
#         self.num_workers = int(num_workers)
#         self.strict_non_exemplar = bool(strict_non_exemplar)
#         self.class_balanced_train_batches = bool(class_balanced_train_batches)
#         self.geometry_batch_classes = int(geometry_batch_classes)
#         self.geometry_samples_per_class = int(geometry_samples_per_class)
#         self.return_metadata = bool(return_metadata)
#         self.spectral_consistency_enabled = bool(spectral_consistency_enabled)
#         self.spectral_augmentation_enabled = self._spectral_augmented_patches is not None
#         self.return_spectral_view = bool(
#             self.return_metadata and self.spectral_augmentation_enabled
#         )
#         self.purge_finalized_train_payload = bool(purge_finalized_train_payload)
#         self._purged_train_indices: Set[int] = set()
#         self.target_names = target_names
#         self.pin_memory = self.device.startswith("cuda")

#         if protocol_plan is None:
#             plan = self.build_protocol_plan(
#                 self.labels,
#                 base_classes=self.base_classes,
#                 increment=self.increment,
#                 train_ratio=self.train_ratio,
#                 val_ratio=self.val_ratio,
#                 min_train_per_class=self.min_train_per_class,
#                 seed=self.seed,
#                 shuffle_order=bool(shuffle_order),
#                 class_order=class_order,
#             )
#         else:
#             plan = validate_incremental_protocol_plan(
#                 protocol_plan,
#                 self.labels,
#                 expected_base_classes=self.base_classes,
#                 expected_increment=self.increment,
#             )
#             split_policy = dict(plan.get("split_policy", {}))
#             expected_policy = {
#                 "train_ratio": self.train_ratio,
#                 "val_ratio": self.val_ratio,
#                 "min_train_per_class": self.min_train_per_class,
#                 "seed": self.seed,
#             }
#             for name, expected in expected_policy.items():
#                 if name not in split_policy:
#                     continue
#                 actual = split_policy[name]
#                 if isinstance(expected, float):
#                     matches = abs(float(actual) - expected) <= 1e-12
#                 else:
#                     matches = int(actual) == int(expected)
#                 if not matches:
#                     raise ValueError(
#                         f"protocol_plan {name}={actual} does not match constructor {name}={expected}."
#                     )

#         self._protocol_plan = validate_incremental_protocol_plan(
#             plan,
#             self.labels,
#             expected_base_classes=self.base_classes,
#             expected_increment=self.increment,
#         )
#         self.all_classes = list(self._protocol_plan["input_classes"])
#         self.num_classes = int(self._protocol_plan["num_classes"])
#         self.class_order = list(self._protocol_plan["class_order"])
#         self.label_map = dict(self._protocol_plan["label_map"])
#         self.inv_label_map = dict(self._protocol_plan["inv_label_map"])
#         self.remapped_labels = np.asarray(
#             self._protocol_plan["remapped_labels"], dtype=np.int64
#         ).copy()
#         self.num_phases = int(self._protocol_plan["num_phases"])
#         self.phase_to_classes = {
#             int(key): list(values)
#             for key, values in self._protocol_plan["phase_to_classes"].items()
#         }
#         self.class_to_phase = dict(self._protocol_plan["class_to_phase"])
#         self.train_indices = np.asarray(
#             self._protocol_plan["train_indices"], dtype=np.int64
#         ).copy()
#         self.val_indices = np.asarray(
#             self._protocol_plan["val_indices"], dtype=np.int64
#         ).copy()
#         self.test_indices = np.asarray(
#             self._protocol_plan["test_indices"], dtype=np.int64
#         ).copy()
#         self.base_train_indices = np.asarray(
#             self._protocol_plan["base_train_indices"], dtype=np.int64
#         ).copy()

#         self.label_policy = self._normalize_label_policy(label_policy)
#         self.has_background = bool(self.label_policy.get("has_background", True))
#         self.background_label = self.label_policy.get(
#             "background_label", 0 if self.has_background else None
#         )
#         self.raw_class_values = [
#             int(value) for value in self.label_policy.get("raw_class_values", [])
#         ]
#         self.target_names_by_seq = self._build_target_names_by_seq(target_names)

#         raw_center = center_spectra if center_spectra is not None else raw_spectra
#         self._center_spectra: Optional[np.ndarray] = None
#         self.spectral_metadata: Dict[str, Any] = dict(spectral_metadata or {})
#         if raw_center is not None:
#             raw_center = np.asarray(raw_center, dtype=np.float32)
#             if raw_center.ndim != 2:
#                 raw_center = raw_center.reshape(len(self.labels), -1)
#             if len(raw_center) != len(self.labels):
#                 raise ValueError(
#                     "center_spectra/label length mismatch: "
#                     f"{len(raw_center)} vs {len(self.labels)}"
#                 )
#             if not np.isfinite(raw_center).all():
#                 raise ValueError("center_spectra contain NaN or Inf values.")
#             self._center_spectra = np.ascontiguousarray(raw_center, dtype=np.float32).copy()

#             space = str(
#                 self.spectral_metadata.get(
#                     "space",
#                     "raw_wavelength" if bool(spectra_are_physical) else "unknown",
#                 )
#             ).strip().lower()
#             if space not in _ALLOWED_SPECTRAL_SPACES:
#                 raise ValueError(
#                     f"Unknown spectral metadata space={space!r}; expected one of "
#                     f"{sorted(_ALLOWED_SPECTRAL_SPACES)}."
#                 )
#             physical = bool(self.spectral_metadata.get("physical", space in _PHYSICAL_SPECTRAL_SPACES))
#             if bool(spectra_are_physical) and not physical:
#                 raise ValueError(
#                     "spectra_are_physical=True conflicts with spectral_metadata physical=False."
#                 )
#             if physical and space not in _PHYSICAL_SPECTRAL_SPACES:
#                 raise ValueError(
#                     f"spectral space {space!r} cannot be marked physical."
#                 )

#             actual_spectra_hash = _array_sha256(self._center_spectra)
#             actual_coords_hash = _array_sha256(self.coords)
#             expected_spectra_hash = self.spectral_metadata.get("center_spectra_sha256")
#             expected_coords_hash = self.spectral_metadata.get("coords_sha256")
#             if expected_spectra_hash is not None and str(expected_spectra_hash) != actual_spectra_hash:
#                 raise ValueError("center_spectra fingerprint does not match spectral_metadata.")
#             if expected_coords_hash is not None and str(expected_coords_hash) != actual_coords_hash:
#                 raise ValueError("coordinate fingerprint does not match spectral_metadata.")

#             self.spectral_metadata.update({
#                 "version": int(self.spectral_metadata.get("version", 1)),
#                 "space": space,
#                 "physical": physical,
#                 "source": str(self.spectral_metadata.get("source", "center_pixel")),
#                 "sample_count": int(self._center_spectra.shape[0]),
#                 "band_count": int(self._center_spectra.shape[1]),
#                 "band_order_preserved": bool(
#                     self.spectral_metadata.get("band_order_preserved", space != "pca_components")
#                 ),
#                 "pca_applied": bool(self.spectral_metadata.get("pca_applied", space == "pca_components")),
#                 "center_spectra_sha256": actual_spectra_hash,
#                 "coords_sha256": actual_coords_hash,
#             })
#         else:
#             self.spectral_metadata = {
#                 "version": 1,
#                 "space": "unavailable",
#                 "physical": False,
#                 "source": "none",
#                 "sample_count": 0,
#                 "band_count": 0,
#                 "band_order_preserved": False,
#                 "pca_applied": False,
#                 "center_spectra_sha256": None,
#                 "coords_sha256": _array_sha256(self.coords),
#             }

#         self.spectra_are_physical = bool(
#             self._center_spectra is not None
#             and self.spectral_metadata.get("physical", False)
#         )
#         if self.spectral_consistency_enabled:
#             if self._center_spectra is None:
#                 raise ValueError(
#                     "spectral_consistency_enabled=True requires aligned center_spectra."
#                 )
#             if not self.spectra_are_physical:
#                 raise ValueError(
#                     "Spectral consistency requires raw_wavelength or "
#                     "sensor_corrected_wavelength center spectra."
#                 )
#             if not self.return_metadata:
#                 raise ValueError(
#                     "spectral_consistency_enabled=True requires return_metadata=True "
#                     "so each training batch contains aligned spectra."
#                 )
#             if self._spectral_augmented_patches is None:
#                 raise ValueError(
#                     "spectral_consistency_enabled=True requires aligned processed "
#                     "spectral_augmented_patches. Create the perturbation in raw "
#                     "wavelength space, then apply the same fitted preprocessor/PCA."
#                 )

#         self.current_phase = 0
#         self.finalized_phases: Set[int] = set()
#         self.finalized_classes: Set[int] = set()
#         self._memory_build_active = False
#         self._memory_build_classes: Set[int] = set()

#         self._validate_class_zero_policy()
#         self._validate_full_class_coverage()
#         self._validate_protocol_integrity()
#         self.assert_preprocess_fit_indices()
#         self._print_stats()

#     @property
#     def raw_spectra(self) -> Optional[np.ndarray]:
#         """Deprecated direct-access compatibility view.

#         Strict non-exemplar mode forbids bypassing phase-aware accessors. In
#         non-strict mode a copy is returned so external code cannot mutate the
#         dataset's spectral store in place.
#         """
#         if self.strict_non_exemplar:
#             raise AttributeError(
#                 "Direct raw_spectra access is disabled in strict non-exemplar mode. "
#                 "Use get_class_spectra(), a metadata DataLoader, or the phase-aware data helpers."
#             )
#         return None if self._center_spectra is None else self._center_spectra.copy()

#     def has_spectral_augmented_view(self) -> bool:
#         """Return whether an aligned processed physical spectral view is available."""
#         return bool(
#             self._spectral_augmented_patches is not None
#             and self._spectral_augmented_patches.shape == self.patches.shape
#         )

#     def get_spectral_augmented_view(
#         self,
#         indices: Sequence[int],
#         *,
#         split: str = "train",
#     ) -> np.ndarray:
#         """Return a copy of the aligned view under strict phase access rules.

#         This accessor exists for diagnostics. Normal training must consume the
#         view through the phase-aware DataLoader so labels, spectra, coordinates,
#         and both patch views remain aligned.
#         """
#         if self._spectral_augmented_patches is None:
#             raise RuntimeError("No spectral augmented view is available")
#         idx = np.asarray(indices, dtype=np.int64).reshape(-1)
#         if idx.size == 0:
#             return np.empty((0, *self.patches.shape[1:]), dtype=np.float32)
#         if int(idx.min()) < 0 or int(idx.max()) >= len(self.labels):
#             raise ValueError("spectral view indices are outside the dataset")
#         if str(split).lower() == "train" and self.strict_non_exemplar:
#             classes = np.unique(self.remapped_labels[idx]).tolist()
#             for class_id in classes:
#                 self._check_class_split_access(int(class_id), "train")
#         return self._spectral_augmented_patches[idx].copy()

#     # ============================================================
#     # Label policy helpers
#     # ============================================================
#     def _normalize_label_policy(self, label_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Normalize dataset label metadata.

#         The incremental manager receives sequential training labels. Raw GT
#         background handling must be decided in the loader/ImageCubes stage. This
#         policy is carried here only for validation, reporting, and visualization.

#         Rules:
#             - if has_background=True, raw background is display-only and should
#               not appear in self.labels;
#             - if has_background=False, raw class 0 is allowed to be a real class;
#             - model/trainer class ids remain sequential 0..K-1 either way.
#         """
#         policy = dict(label_policy or {})

#         if "has_background" not in policy:
#             # Conservative default: most classic HSI datasets use raw 0 as background.
#             # Datasets with raw class 0 real must pass has_background=False from the loader.
#             policy["has_background"] = True

#         policy["has_background"] = bool(policy["has_background"])

#         if policy["has_background"]:
#             try:
#                 policy["background_label"] = int(policy.get("background_label", 0))
#             except Exception:
#                 policy["background_label"] = 0
#         else:
#             policy["background_label"] = None

#         raw_values = policy.get("raw_class_values", None)
#         if raw_values is None:
#             raw_values = []
#         policy["raw_class_values"] = [int(v) for v in raw_values]

#         return policy

#     # ============================================================
#     # Validation
#     # ============================================================
#     def _validate_class_zero_policy(self) -> None:
#         """
#         Validate class-0/background semantics without deleting label 0.

#         Important:
#             self.labels are already training labels from ImageCubes. They are
#             not necessarily raw GT labels. Therefore this method never assumes
#             input label 0 is background. It only checks consistency between the
#             loader-provided label_policy and the sequential label space.
#         """
#         input_has_zero = 0 in set(int(x) for x in self.labels.tolist())
#         remap_has_zero = 0 in set(int(x) for x in self.remapped_labels.tolist())

#         if input_has_zero and not remap_has_zero:
#             raise RuntimeError(
#                 "Input label 0 existed but disappeared after incremental remapping. "
#                 "This is invalid for datasets where class 0 is real and also invalid "
#                 "for already-remapped foreground labels."
#             )

#         if not self.has_background:
#             # Raw class 0 may be a real class. This is valid. Do not warn just
#             # because sequential class 0 exists; it must exist in any sequential
#             # class space.
#             if self.raw_class_values and 0 not in self.raw_class_values:
#                 print(
#                     "[IncrementalHSIDataset:WARN] label_policy says has_background=False, "
#                     "but raw_class_values does not include raw class 0. This may still be "
#                     "valid, but check the loader metadata."
#                 )
#         else:
#             # Background must have been removed before this manager. If raw
#             # background label was 0, sequential class 0 is still allowed because
#             # it now means the first foreground class, not background.
#             if self.background_label is None:
#                 print(
#                     "[IncrementalHSIDataset:WARN] has_background=True but background_label is None. "
#                     "Visualization/reporting will treat display value 0 as background/unlabeled."
#                 )


#     def _build_target_names_by_seq(self, target_names: Optional[List[str]]) -> Optional[List[str]]:
#         if target_names is None:
#             return None

#         out: List[str] = []
#         for sid in range(len(self.all_classes)):
#             gid = self.inv_label_map.get(sid, sid)
#             if int(gid) < len(target_names):
#                 out.append(str(target_names[int(gid)]))
#             else:
#                 out.append(f"Class {gid}")
#         return out

#     def _validate_full_class_coverage(self) -> None:
#         present = set(int(x) for x in np.unique(self.remapped_labels).tolist())
#         expected = set(range(self.num_classes))
#         missing = sorted(expected - present)
#         extra = sorted(present - expected)

#         if missing or extra:
#             raise RuntimeError(
#                 f"Remapped label space is broken. Missing={missing}, extra={extra}, "
#                 f"num_classes={self.num_classes}, class_order={self.class_order}"
#             )

#     def _validate_protocol_integrity(self) -> None:
#         """Fail fast on class-order, phase, and split corruption."""
#         phase_union: List[int] = []
#         for p in range(self.num_phases):
#             cls = self.phase_to_classes.get(p, [])
#             if not cls:
#                 raise RuntimeError(f"Phase {p} has no classes.")
#             phase_union.extend(int(c) for c in cls)
#         if phase_union != list(range(self.num_classes)):
#             raise RuntimeError(
#                 f"Phase partition must cover contiguous sequential ids exactly once. "
#                 f"got={phase_union}, expected={list(range(self.num_classes))}"
#             )
#         for c in range(self.num_classes):
#             if c not in self.class_to_phase:
#                 raise RuntimeError(f"class_to_phase missing class {c}")
#         all_split = np.concatenate([self.train_indices, self.val_indices, self.test_indices]).astype(np.int64)
#         if all_split.size != len(np.unique(all_split)):
#             raise RuntimeError("Train/val/test splits overlap. This leaks samples across splits.")
#         valid = set(range(len(self.remapped_labels)))
#         bad = sorted(set(int(i) for i in all_split.tolist()).difference(valid))
#         if bad:
#             raise RuntimeError(f"Split indices outside dataset range: {bad[:10]}")
#         missing = sorted(valid.difference(set(int(i) for i in all_split.tolist())))
#         # A class with n=1 may be train-only, but every sample must still appear in exactly one split.
#         if missing:
#             raise RuntimeError(f"Some samples are absent from all splits: first_missing={missing[:10]}")

#     def _validate_phase(self, phase: int) -> int:
#         phase = int(phase)
#         if phase < 0 or phase >= self.num_phases:
#             raise ValueError(f"Invalid phase {phase}. Valid range: 0..{self.num_phases - 1}")
#         return phase

#     # ============================================================
#     # Basic mapping helpers
#     # ============================================================
#     def seq_to_global(self, seq_id: int) -> int:
#         return int(self.inv_label_map[int(seq_id)])

#     def global_to_seq(self, global_id: int) -> int:
#         return int(self.label_map[int(global_id)])

#     def get_phase_classes(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         return list(self.phase_to_classes[phase])

#     def get_classes_up_to_phase(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         classes: List[int] = []
#         for p in range(phase + 1):
#             classes.extend(self.phase_to_classes[p])
#         return _ordered_unique_ints(classes)

#     def get_seen_classes(self, phase: int) -> List[int]:
#         """Alias used by trainers/evaluators: cumulative sequential ids."""
#         return self.get_classes_up_to_phase(phase)

#     def get_old_classes(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         if phase == 0:
#             return []
#         return self.get_classes_up_to_phase(phase - 1)

#     def get_new_classes(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         return list(self.phase_to_classes[phase])

#     def get_future_classes(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         seen = set(self.get_seen_classes(phase))
#         return [int(c) for c in range(self.num_classes) if int(c) not in seen]

#     def classifier_num_classes_for_phase(self, phase: int) -> int:
#         """Required classifier width for global sequential CE/logits.

#         The project uses global sequential labels. Therefore phase ``p`` logits
#         must contain columns ``0..max(seen_classes)``. Since phases are allocated
#         in contiguous sequential ids, this equals ``len(seen_classes)``. Keeping
#         this method explicit catches accidental full-dataset classifier outputs
#         during early phases.
#         """
#         seen = self.get_seen_classes(phase)
#         if not seen:
#             raise RuntimeError(f"No seen classes for phase {phase}.")
#         expected = list(range(max(seen) + 1))
#         if seen != expected:
#             raise RuntimeError(
#                 f"Seen classes are not contiguous global ids: seen={seen}, expected={expected}. "
#                 "The trainer/classifier uses global sequential labels and requires contiguous phase ids."
#             )
#         return int(max(seen) + 1)

#     def get_phase_class_splits(self, phase: int) -> Dict[str, List[int]]:
#         phase = self._validate_phase(phase)
#         return {
#             "old": self.get_old_classes(phase),
#             "new": self.get_new_classes(phase),
#             "seen": self.get_seen_classes(phase),
#             "future": self.get_future_classes(phase),
#         }

#     def assert_classifier_contract(self, phase: int, logits_or_width: Any, *, context: str = "") -> None:
#         """Assert classifier output matches currently seen class space."""
#         phase = self._validate_phase(phase)
#         expected = self.classifier_num_classes_for_phase(phase)
#         if torch.is_tensor(logits_or_width):
#             if logits_or_width.dim() != 2:
#                 raise RuntimeError(f"{context}: logits must be [B,C], got {tuple(logits_or_width.shape)}")
#             got = int(logits_or_width.size(1))
#         else:
#             got = int(logits_or_width)
#         if got != expected:
#             raise RuntimeError(
#                 f"{context}: classifier width mismatch for phase {phase}: got {got}, expected {expected}. "
#                 f"Seen classes={self.get_seen_classes(phase)}. Do not expose future-class logits during this phase."
#             )

#     def assert_labels_for_phase(
#         self,
#         labels: Any,
#         phase: int,
#         *,
#         context: str = "",
#         current_train_only: bool = False,
#         cumulative: bool = True,
#     ) -> None:
#         """Validate label tensors before CE/loss/evaluation.

#         ``current_train_only=True`` is for raw train batches in strict NECIL.
#         ``cumulative=True`` is for validation/test over all seen classes.
#         """
#         phase = self._validate_phase(phase)
#         y = _as_1d_int_array(labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels, name="labels")
#         if y.min() < 0:
#             raise RuntimeError(f"{context}: negative labels entered phase {phase}: min={int(y.min())}")
#         allowed = self.get_new_classes(phase) if current_train_only else (self.get_seen_classes(phase) if cumulative else self.get_new_classes(phase))
#         allowed_set = set(int(c) for c in allowed)
#         bad = sorted(set(int(v) for v in y.tolist()).difference(allowed_set))
#         if bad:
#             raise RuntimeError(
#                 f"{context}: labels outside allowed phase classes. bad={bad}, allowed={allowed}, "
#                 f"phase={phase}, current_train_only={current_train_only}, cumulative={cumulative}"
#             )

#     def get_geometry_reservation_plan(
#         self,
#         phase: int = 0,
#         *,
#         feature_dim: Optional[int] = None,
#         rank: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         """Return schedule-only future capacity without exposing future data."""
#         phase = self._validate_phase(phase)
#         return {
#             "phase": int(phase),
#             "seen_classes": self.get_seen_classes(phase),
#             "future_class_slots": self.get_future_classes(phase),
#             "total_class_capacity": int(self.num_classes),
#             "seen_classifier_width": int(self.classifier_num_classes_for_phase(phase)),
#             "feature_dim": None if feature_dim is None else int(feature_dim),
#             "rank": None if rank is None else int(rank),
#             "allowed_reservation": [
#                 "empty_geometry_rows",
#                 "center_clearance_budget",
#                 "subspace_capacity_budget",
#                 "residual_variance_floor",
#             ],
#             "forbidden_reservation": [
#                 "future_raw_patches",
#                 "future_raw_spectra",
#                 "future_feature_vectors",
#                 "future_class_statistics",
#                 "future_labels_in_base_loss",
#             ],
#         }

#     def export_protocol_plan(self, *, jsonable: bool = True) -> Dict[str, Any]:
#         plan = dict(self._protocol_plan)
#         plan["remapped_labels"] = self.remapped_labels.copy()
#         plan["train_indices"] = self.train_indices.copy()
#         plan["val_indices"] = self.val_indices.copy()
#         plan["test_indices"] = self.test_indices.copy()
#         plan["base_train_indices"] = self.base_train_indices.copy()
#         return _jsonable_protocol_value(plan) if bool(jsonable) else plan

#     def get_phase_split_indices(self, phase: int, split: str) -> np.ndarray:
#         phase = self._validate_phase(phase)
#         split_indices = self._get_split_indices(split)
#         phase_classes = self.get_phase_classes(phase)
#         mask = np.isin(self.remapped_labels[split_indices], phase_classes)
#         return split_indices[mask].astype(np.int64, copy=True)

#     def get_preprocess_fit_indices(self, phase: int = 0) -> np.ndarray:
#         """Return the exact phase-0 train indices used to fit normalization/PCA."""
#         phase = self._validate_phase(phase)
#         if phase != 0:
#             raise ValueError(
#                 "The preprocessing transform is fitted once on phase-0 training "
#                 "samples and then frozen; phase must therefore be 0."
#             )
#         self.assert_preprocess_fit_indices()
#         return self.base_train_indices.copy()

#     def get_preprocess_fit_coordinates(self) -> np.ndarray:
#         return self.coords[self.get_preprocess_fit_indices()].copy()

#     def assert_preprocess_fit_indices(self) -> None:
#         base_ids = set(int(value) for value in self.phase_to_classes[0])
#         fit = np.asarray(self.base_train_indices, dtype=np.int64).reshape(-1)
#         if fit.size == 0:
#             raise RuntimeError("base_train_indices is empty.")
#         if np.unique(fit).size != fit.size:
#             raise RuntimeError("base_train_indices contains duplicates.")
#         if int(fit.min()) < 0 or int(fit.max()) >= len(self.remapped_labels):
#             raise RuntimeError("base_train_indices contains out-of-range values.")
#         train_set = set(int(value) for value in self.train_indices.tolist())
#         if not set(int(value) for value in fit.tolist()).issubset(train_set):
#             raise RuntimeError("base_train_indices is not a subset of train_indices.")
#         bad = sorted(
#             set(int(value) for value in self.remapped_labels[fit].tolist()) - base_ids
#         )
#         if bad:
#             raise RuntimeError(
#                 f"Preprocessing fit indices contain future classes: {bad}."
#             )

#     # ============================================================
#     # Protocol controls
#     # ============================================================
#     def start_phase(self, phase: int) -> None:
#         phase = self._validate_phase(phase)
#         if phase in self.finalized_phases:
#             raise RuntimeError(f"Phase {phase} is already finalized and cannot be restarted.")
#         missing_previous = [
#             previous for previous in range(phase) if previous not in self.finalized_phases
#         ]
#         if missing_previous:
#             raise RuntimeError(
#                 f"Cannot start phase {phase}; previous phases are not finalized: {missing_previous}."
#             )
#         if self._memory_build_active:
#             raise RuntimeError("Cannot change phase while memory_build_context is active.")
#         self.current_phase = int(phase)

#     def finalize_phase(self, phase: int) -> None:
#         phase = self._validate_phase(phase)
#         if phase != self.current_phase:
#             raise RuntimeError(
#                 f"Cannot finalize phase {phase}; current_phase={self.current_phase}."
#             )
#         if phase in self.finalized_phases:
#             raise RuntimeError(f"Phase {phase} is already finalized.")
#         if self._memory_build_active:
#             raise RuntimeError("Cannot finalize a phase while memory_build_context is active.")
#         missing_previous = [
#             previous for previous in range(phase) if previous not in self.finalized_phases
#         ]
#         if missing_previous:
#             raise RuntimeError(
#                 f"Cannot finalize phase {phase}; previous phases are not finalized: {missing_previous}."
#             )

#         self.finalized_phases.add(int(phase))
#         self.finalized_classes.update(int(value) for value in self.phase_to_classes[phase])
#         if self.purge_finalized_train_payload:
#             self._purge_phase_train_payload(phase)
#         self._memory_build_classes.clear()

#     def _purge_phase_train_payload(self, phase: int) -> None:
#         """Erase finalized TRAIN patches/spectra while retaining labels and coordinates.

#         Validation and test payloads are untouched. This optional defense makes
#         accidental old-training reuse impossible even if a caller bypasses the
#         public accessors. GeometryBank construction must finish before phase
#         finalization.
#         """
#         indices = self.get_phase_split_indices(int(phase), "train")
#         if indices.size == 0:
#             return
#         self.patches[indices] = 0.0
#         if self._center_spectra is not None:
#             self._center_spectra[indices] = 0.0
#         if self._spectral_augmented_patches is not None:
#             self._spectral_augmented_patches[indices] = 0.0
#         self._purged_train_indices.update(int(value) for value in indices.tolist())

#     def is_phase_finalized(self, phase: int) -> bool:
#         return int(phase) in self.finalized_phases

#     def is_class_finalized(self, cls: int) -> bool:
#         return int(cls) in self.finalized_classes

#     def get_accessible_train_classes(self) -> List[int]:
#         if not self.strict_non_exemplar:
#             return list(range(self.num_classes))
#         return list(self.phase_to_classes[self.current_phase])

#     def _is_train_access_allowed(self, cls: int) -> bool:
#         """Return whether raw TRAIN samples of ``cls`` may be read now.

#         This is the strict exemplar-free access contract. Once a phase is
#         finalized, its raw training samples are old exemplars and must never be
#         exposed again. The only exception is the short memory_build_context for
#         the current phase before finalization, where descriptors are being built
#         from current-class data.
#         """
#         cls = int(cls)
#         if cls < 0 or cls >= self.num_classes:
#             raise ValueError(f"class id {cls} outside sequential label space [0,{self.num_classes - 1}]")

#         if not self.strict_non_exemplar:
#             return True

#         if cls in self.finalized_classes:
#             return False

#         if self._memory_build_active and cls in self._memory_build_classes:
#             return True

#         if cls in self.phase_to_classes[self.current_phase]:
#             return True

#         return False

#     def _check_class_split_access(self, cls: int, split: str) -> None:
#         cls = int(cls)
#         split = str(split).lower()

#         if cls < 0 or cls >= self.num_classes:
#             raise ValueError(f"class id {cls} outside sequential label space [0,{self.num_classes - 1}]")

#         if split != "train":
#             return

#         if not self._is_train_access_allowed(cls):
#             phase_of_cls = self.class_to_phase.get(cls, None)
#             raise PermissionError(
#                 f"Strict non-exemplar protocol violation: raw TRAIN access denied for class {cls} "
#                 f"(phase={phase_of_cls}, current_phase={self.current_phase}, finalized={cls in self.finalized_classes}, "
#                 f"memory_build_active={self._memory_build_active}). Use GeometryBank descriptors / synthetic geometry replay, "
#                 f"not old raw patches or old raw spectra."
#             )

#     @contextmanager
#     def memory_build_context(self, phase: int):
#         phase = self._validate_phase(phase)
#         if phase != self.current_phase:
#             raise ValueError(
#                 f"memory_build_context phase={phase} must match current_phase={self.current_phase}"
#             )
#         if phase in self.finalized_phases:
#             raise PermissionError(
#                 f"Cannot rebuild raw-sample memory for finalized phase {phase}."
#             )
#         if self._memory_build_active:
#             raise RuntimeError("Nested memory_build_context calls are not allowed.")

#         allowed_classes = set(self.phase_to_classes[phase])

#         prev_active = self._memory_build_active
#         prev_classes = set(self._memory_build_classes)

#         self._memory_build_active = True
#         self._memory_build_classes = allowed_classes

#         try:
#             yield
#         finally:
#             self._memory_build_active = prev_active
#             self._memory_build_classes = prev_classes

#     # ============================================================
#     # Split plan access
#     # ============================================================
#     def _create_splits(self) -> None:
#         raise RuntimeError(
#             "Splits are immutable and must be created by build_protocol_plan() "
#             "before preprocessing. Rebuilding them after PCA would break the "
#             "base-train preprocessing contract."
#         )

#     # ============================================================
#     # Diagnostics
#     # ============================================================
#     def _print_stats(self) -> None:
#         print("[IncrementalHSIDataset] Initialized")
#         print(f"  Total classes: {self.num_classes} | Phases: {self.num_phases}")
#         print(f"  Input classes: {self.all_classes}")
#         print(f"  Class order: {self.class_order}")
#         print(f"  Strict non-exemplar: {self.strict_non_exemplar}")
#         print(
#             f"  Protocol plan: version={self._protocol_plan['version']} | "
#             f"fingerprint={str(self._protocol_plan['labels_sha256'])[:12]}... | "
#             f"base-PCA-fit samples={len(self.base_train_indices)}"
#         )
#         print(
#             f"  Spectral metadata: available={self.has_spectral_metadata()}, "
#             f"physical={self.has_physical_spectra()}, dim={self.get_spectra_dim()}"
#         )
#         print(
#             "  Processed spectral view: "
#             f"available={self.has_spectral_augmented_view()}, "
#             f"returned={self.return_spectral_view}"
#         )
#         print(
#             f"  Label policy: has_background={self.has_background}, "
#             f"background_label={self.background_label}, raw_class_values={self.raw_class_values}"
#         )
#         print(
#             "  Sequential class 0 present: "
#             f"{0 in set(int(x) for x in self.remapped_labels.tolist())} "
#             "(sequential class 0 is a real foreground training class, not background)"
#         )

#         print(f"  Split sizes: train={len(self.train_indices)} | val={len(self.val_indices)} | test={len(self.test_indices)}")

#         for p in range(self.num_phases):
#             global_ids = [self.seq_to_global(sid) for sid in self.phase_to_classes[p]]
#             names = []
#             if self.target_names_by_seq is not None:
#                 for sid in self.phase_to_classes[p]:
#                     if int(sid) < len(self.target_names_by_seq):
#                         names.append(self.target_names_by_seq[int(sid)])
#                     else:
#                         names.append(f"Class {self.seq_to_global(sid)}")
#             print(f"  Phase {p}: Sequential {self.phase_to_classes[p]} (Input labels {global_ids})")
#             if names:
#                 print(f"           Names: {names}")

#     def get_label_policy_summary(self) -> Dict[str, Any]:
#         return {
#             "has_background": bool(self.has_background),
#             "background_label": self.background_label,
#             "raw_class_values": list(self.raw_class_values),
#             "class_order": [int(c) for c in self.class_order],
#             "label_map": {int(k): int(v) for k, v in self.label_map.items()},
#             "inv_label_map": {int(k): int(v) for k, v in self.inv_label_map.items()},
#             "note": (
#                 "Training labels are sequential 0..K-1. Display value 0 in maps "
#                 "is reserved for background/unseen/suppressed; class c is displayed as c+1."
#             ),
#         }

#     def has_spectral_metadata(self) -> bool:
#         return self._center_spectra is not None and int(self._center_spectra.size) > 0

#     def has_physical_spectra(self) -> bool:
#         return bool(
#             self.has_spectral_metadata()
#             and self.spectral_metadata.get("physical", False)
#             and self.spectral_metadata.get("space") in _PHYSICAL_SPECTRAL_SPACES
#         )

#     def get_spectra_dim(self) -> int:
#         if self._center_spectra is None or self._center_spectra.ndim != 2:
#             return 0
#         return int(self._center_spectra.shape[1])

#     def get_spectral_metadata_summary(self) -> Dict[str, Any]:
#         summary = dict(self.spectral_metadata)
#         summary.update({
#             "available": bool(self.has_spectral_metadata()),
#             "physical": bool(self.has_physical_spectra()),
#             "dim": int(self.get_spectra_dim()),
#             "spectral_consistency_enabled": bool(self.spectral_consistency_enabled),
#             "purge_finalized_train_payload": bool(self.purge_finalized_train_payload),
#             "purged_train_samples": int(len(self._purged_train_indices)),
#             "note": (
#                 "Center spectra are aligned wavelength metadata for "
#                 "Spectral-Geometry Consistency and spectral-aware admission. "
#                 "PCA components must never be used as physical wavelength spectra."
#             ),
#         })
#         return summary

#     def _spectra_for_indices(
#         self,
#         idx: np.ndarray,
#         *,
#         require_physical: bool = True,
#     ) -> torch.Tensor:
#         if self._center_spectra is None:
#             raise AttributeError("No center-spectral metadata are available in this dataset.")
#         if bool(require_physical) and not self.has_physical_spectra():
#             raise RuntimeError(
#                 "Requested physical center spectra, but the metadata space is "
#                 f"{self.spectral_metadata.get('space')!r}. Use raw_wavelength or "
#                 "sensor_corrected_wavelength metadata for spectral consistency."
#             )
#         idx = np.asarray(idx, dtype=np.int64).reshape(-1)
#         if len(idx) == 0:
#             raise ValueError("No indices provided for spectral metadata lookup.")
#         return torch.from_numpy(
#             np.ascontiguousarray(self._center_spectra[idx])
#         ).float()

#     def get_class_spectra(
#         self,
#         cls: int,
#         split: str = "train",
#         require_physical: bool = True,
#     ) -> torch.Tensor:
#         """Return phase-authorized center spectra for one class and split."""
#         idx = self.get_class_indices(int(cls), split=split)
#         if len(idx) == 0:
#             raise ValueError(f"No spectra found for class {cls} in split '{split}'")
#         return self._spectra_for_indices(idx, require_physical=bool(require_physical))

#     def get_class_center_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
#         return self.get_class_spectra(cls, split=split, require_physical=True)

#     def get_class_raw_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
#         return self.get_class_spectra(cls, split=split, require_physical=True)

#     def get_class_spectral_summary(self, cls: int, split: str = "train") -> torch.Tensor:
#         return self.get_class_spectra(cls, split=split, require_physical=True)

#     def get_class_counts(self) -> Dict[int, int]:
#         return {
#             int(c): int((self.remapped_labels == int(c)).sum())
#             for c in range(self.num_classes)
#         }

#     def get_split_class_counts(self, split: str = "train") -> Dict[int, int]:
#         indices = self._get_split_indices(split)
#         return {
#             int(c): int((self.remapped_labels[indices] == int(c)).sum())
#             for c in range(self.num_classes)
#         }

#     def protocol_state(self) -> Dict[str, object]:
#         return {
#             "current_phase": int(self.current_phase),
#             "strict_non_exemplar": bool(self.strict_non_exemplar),
#             "finalized_phases": sorted(int(p) for p in self.finalized_phases),
#             "finalized_classes": sorted(int(c) for c in self.finalized_classes),
#             "accessible_train_classes": self.get_accessible_train_classes(),
#             "phase_class_splits": self.get_phase_class_splits(self.current_phase),
#             "classifier_width": self.classifier_num_classes_for_phase(self.current_phase),
#             "memory_build_active": bool(self._memory_build_active),
#             "memory_build_classes": sorted(int(c) for c in self._memory_build_classes),
#             "spectral_metadata": self.get_spectral_metadata_summary(),
#             "spectral_augmented_view_available": bool(
#                 self.has_spectral_augmented_view()
#             ),
#             "purged_train_indices_count": int(len(self._purged_train_indices)),
#             "protocol_plan_version": int(self._protocol_plan["version"]),
#             "protocol_labels_sha256": str(self._protocol_plan["labels_sha256"]),
#             "base_train_indices_count": int(len(self.base_train_indices)),
#         }

#     def assert_non_exemplar(self) -> bool:
#         if not self.strict_non_exemplar:
#             raise RuntimeError("strict_non_exemplar is disabled.")
#         if self._memory_build_active and (
#             self.finalized_classes & self._memory_build_classes
#         ):
#             raise RuntimeError(
#                 "memory_build_context exposes finalized old classes."
#             )
#         for class_id in self.finalized_classes:
#             if self._is_train_access_allowed(int(class_id)):
#                 raise RuntimeError(
#                     f"Finalized class {class_id} still has raw train access."
#                 )
#         self.assert_preprocess_fit_indices()
#         if self.purge_finalized_train_payload and self._purged_train_indices:
#             idx = np.asarray(sorted(self._purged_train_indices), dtype=np.int64)
#             if np.any(self.patches[idx] != 0.0):
#                 raise RuntimeError("Purged finalized training patches are still present.")
#             if self._center_spectra is not None and np.any(self._center_spectra[idx] != 0.0):
#                 raise RuntimeError("Purged finalized training spectra are still present.")
#             if (
#                 self._spectral_augmented_patches is not None
#                 and np.any(self._spectral_augmented_patches[idx] != 0.0)
#             ):
#                 raise RuntimeError(
#                     "Purged finalized spectral-view training patches are still present."
#                 )
#         return True

#     def assert_no_old_train_access(self, cls: int) -> None:
#         self._check_class_split_access(int(cls), "train")

#     def _assert_indices_match_classes(self, idx: np.ndarray, class_ids: Sequence[int], *, context: str) -> None:
#         idx = np.asarray(idx, dtype=np.int64).reshape(-1)
#         class_ids = _ordered_unique_ints(class_ids)
#         if idx.size == 0:
#             return
#         if idx.min() < 0 or idx.max() >= len(self.remapped_labels):
#             raise RuntimeError(f"{context}: indices outside dataset range.")
#         allowed = set(int(c) for c in class_ids)
#         labels = self.remapped_labels[idx]
#         bad = sorted(set(int(v) for v in labels.tolist()).difference(allowed))
#         if bad:
#             raise RuntimeError(f"{context}: indices contain labels outside active classes. bad={bad}, allowed={class_ids}")

#     def _make_loader(
#         self,
#         idx: np.ndarray,
#         batch_size: int,
#         shuffle: bool,
#         *,
#         balanced: bool = False,
#         active_classes: Optional[List[int]] = None,
#     ) -> DataLoader:
#         idx = np.asarray(idx, dtype=np.int64)
#         if idx.size == 0:
#             raise ValueError(
#                 "Requested DataLoader has zero samples. Check phase split, class order, "
#                 "or strict non-exemplar access policy."
#             )

#         dataset = HSIPatchDataset(
#             self.patches[idx],
#             self.remapped_labels[idx],
#             coords=self.coords[idx],
#             return_metadata=self.return_metadata,
#             center_spectra=(self._center_spectra[idx] if self.has_physical_spectra() else None),
#             spectra_are_physical=bool(self.has_physical_spectra()),
#             return_input_center_when_no_spectra=False,
#             spectral_augmented_patches=(
#                 self._spectral_augmented_patches[idx]
#                 if self.has_spectral_augmented_view()
#                 else None
#             ),
#         )

#         if balanced:
#             present = sorted(int(c) for c in dataset.labels.unique().tolist())
#             if len(present) <= 1:
#                 return DataLoader(
#                     dataset,
#                     batch_size=int(batch_size),
#                     shuffle=bool(shuffle),
#                     pin_memory=self.pin_memory,
#                     num_workers=self.num_workers,
#                     drop_last=False,
#                 )

#             classes_per_batch = int(self.geometry_batch_classes)
#             if classes_per_batch <= 0:
#                 classes_per_batch = len(active_classes) if active_classes is not None else len(present)
#             classes_per_batch = max(2, min(classes_per_batch, len(present), int(batch_size)))

#             samples_per_class = int(self.geometry_samples_per_class)
#             if samples_per_class <= 0:
#                 samples_per_class = max(2, int(batch_size) // classes_per_batch)

#             sampler = ClassBalancedBatchSampler(
#                 labels=dataset.labels,
#                 batch_size=int(batch_size),
#                 classes_per_batch=classes_per_batch,
#                 samples_per_class=samples_per_class,
#                 seed=self.seed + self.current_phase * 1009,
#                 drop_last=False,
#             )

#             return DataLoader(
#                 dataset,
#                 batch_sampler=sampler,
#                 pin_memory=self.pin_memory,
#                 num_workers=self.num_workers,
#             )

#         return DataLoader(
#             dataset,
#             batch_size=int(batch_size),
#             shuffle=bool(shuffle),
#             pin_memory=self.pin_memory,
#             num_workers=self.num_workers,
#             drop_last=False,
#         )

#     # ============================================================
#     # Split access helpers
#     # ============================================================
#     def _get_split_indices(self, split: str) -> np.ndarray:
#         split = split.lower()
#         if split == "train":
#             return self.train_indices
#         if split == "val":
#             return self.val_indices
#         if split == "test":
#             return self.test_indices
#         if split == "all":
#             return np.arange(len(self.remapped_labels), dtype=np.int64)
#         raise ValueError(f"Unknown split '{split}'. Use one of: train, val, test, all")

#     def get_class_indices(self, cls: int, split: str = "train") -> np.ndarray:
#         cls = int(cls)
#         split = split.lower()
#         self._check_class_split_access(cls, split)

#         indices = self._get_split_indices(split)
#         mask = self.remapped_labels[indices] == cls
#         return indices[mask]

#     def get_class_patches(self, cls: int, split: str = "train") -> np.ndarray:
#         idx = self.get_class_indices(cls, split=split)
#         if len(idx) == 0:
#             raise ValueError(f"No samples found for class {cls} in split '{split}'")
#         return self.patches[idx]

#     # ============================================================
#     # DataLoaders
#     # ============================================================
#     def get_class_balanced_phase_loader(
#         self,
#         phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#     ) -> DataLoader:
#         """Return a class-balanced phase loader for geometry construction/refinement.

#         This is a named wrapper so the trainer can request balanced current-phase
#         batches without relying on implicit flags.  It preserves strict
#         non-exemplar access: train split exposes only current-phase raw samples.
#         """
#         old_flag = self.class_balanced_train_batches
#         try:
#             self.class_balanced_train_batches = True
#             return self.get_phase_dataloader(
#                 phase=phase,
#                 split=split,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#             )
#         finally:
#             self.class_balanced_train_batches = old_flag

#     def get_phase_dataloader(
#         self,
#         phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#     ) -> DataLoader:
#         phase = self._validate_phase(phase)
#         split = split.lower()

#         active_classes = self.phase_to_classes[phase]

#         if split == "train":
#             if self.strict_non_exemplar and phase != self.current_phase:
#                 raise PermissionError(
#                     f"Raw TRAIN loader requested for phase {phase}, but current_phase is {self.current_phase}. "
#                     "Strict non-exemplar mode only allows current-phase raw train access."
#                 )

#         indices = self._get_split_indices(split)
#         mask = np.isin(self.remapped_labels[indices], active_classes)
#         idx = indices[mask]
#         self._assert_indices_match_classes(idx, active_classes, context=f"get_phase_dataloader({split}, phase={phase})")

#         use_balanced = (
#             split == "train"
#             and bool(self.class_balanced_train_batches)
#             and bool(shuffle)
#         )

#         return self._make_loader(
#             idx,
#             batch_size=batch_size,
#             shuffle=(shuffle if split == "train" else False),
#             balanced=use_balanced,
#             active_classes=active_classes,
#         )

#     def get_cumulative_dataloader(
#         self,
#         up_to_phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#         allow_train_old: bool = False,
#     ) -> DataLoader:
#         up_to_phase = self._validate_phase(up_to_phase)
#         split = split.lower()

#         if split == "train" and self.strict_non_exemplar:
#             if bool(allow_train_old):
#                 raise PermissionError(
#                     "allow_train_old=True would expose raw old training samples in strict non-exemplar mode. "
#                     "Use GeometryBank synthetic replay instead."
#                 )
#             if int(up_to_phase) != int(self.current_phase):
#                 raise PermissionError(
#                     f"Strict train loader requested up_to_phase={up_to_phase}, but "
#                     f"current_phase={self.current_phase}. Raw training access is "
#                     "restricted to the current phase."
#                 )
#             active_classes = list(self.phase_to_classes[self.current_phase])
#         else:
#             active_classes = self.get_seen_classes(up_to_phase)

#         indices = self._get_split_indices(split)
#         mask = np.isin(self.remapped_labels[indices], active_classes)
#         idx = indices[mask]

#         self._assert_indices_match_classes(idx, active_classes, context=f"get_cumulative_dataloader({split}, phase={up_to_phase})")

#         use_balanced = (
#             split == "train"
#             and bool(self.class_balanced_train_batches)
#             and bool(shuffle)
#         )

#         return self._make_loader(
#             idx,
#             batch_size=batch_size,
#             shuffle=(shuffle if split == "train" else False),
#             balanced=use_balanced,
#             active_classes=active_classes,
#         )

#     def _package_direct_data(
#         self,
#         idx: np.ndarray,
#         *,
#         include_spectra: bool,
#         require_physical: bool = True,
#     ):
#         base = (
#             self.patches[idx],
#             self.remapped_labels[idx],
#             self.coords[idx],
#         )
#         if not bool(include_spectra):
#             return base
#         spectra = self._spectra_for_indices(idx, require_physical=bool(require_physical))
#         return (*base, spectra.numpy())

#     def get_cumulative_test_data(
#         self,
#         phase: int,
#         *,
#         include_spectra: bool = False,
#         require_physical: bool = True,
#     ):
#         phase = self._validate_phase(phase)
#         active_classes = self.get_seen_classes(phase)
#         mask = np.isin(self.remapped_labels[self.test_indices], active_classes)
#         idx = self.test_indices[mask]
#         self._assert_indices_match_classes(
#             idx, active_classes, context=f"get_cumulative_test_data(phase={phase})"
#         )
#         return self._package_direct_data(
#             idx, include_spectra=include_spectra, require_physical=require_physical
#         )

#     def get_cumulative_val_data(
#         self,
#         phase: int,
#         *,
#         include_spectra: bool = False,
#         require_physical: bool = True,
#     ):
#         phase = self._validate_phase(phase)
#         active_classes = self.get_seen_classes(phase)
#         mask = np.isin(self.remapped_labels[self.val_indices], active_classes)
#         idx = self.val_indices[mask]
#         self._assert_indices_match_classes(
#             idx, active_classes, context=f"get_cumulative_val_data(phase={phase})"
#         )
#         return self._package_direct_data(
#             idx, include_spectra=include_spectra, require_physical=require_physical
#         )

#     def get_current_train_data(
#         self,
#         phase: Optional[int] = None,
#         *,
#         include_spectra: bool = False,
#         require_physical: bool = True,
#     ):
#         phase = self.current_phase if phase is None else self._validate_phase(phase)
#         if self.strict_non_exemplar and int(phase) != int(self.current_phase):
#             raise PermissionError(
#                 f"Raw train data requested for phase {phase}, but current_phase={self.current_phase}. "
#                 "Strict NECIL exposes only current-phase train samples."
#             )
#         active_classes = self.get_new_classes(phase)
#         mask = np.isin(self.remapped_labels[self.train_indices], active_classes)
#         idx = self.train_indices[mask]
#         self._assert_indices_match_classes(
#             idx, active_classes, context=f"get_current_train_data(phase={phase})"
#         )
#         return self._package_direct_data(
#             idx, include_spectra=include_spectra, require_physical=require_physical
#         )
