from __future__ import annotations

"""Leakage-auditable class-incremental HSI protocol and patch data manager.

The protocol follows the common NECIL evaluation setup: each phase trains on
the current classes, while evaluation uses one global label space over all
classes seen so far.  The spatial branch receives a processed patch, while the
spectral branch receives the raw, ordered, full-band centre spectrum.  No old
training samples are returned by a phase-training API.
"""

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_SPLITS = ("train", "val", "test")


def _ids(values: Iterable[int], *, name: str, allow_empty: bool = False) -> List[int]:
    result = [int(value) for value in values]
    if not result and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)) or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain unique non-negative IDs")
    return result


def _normalize_split_name(split: str) -> str:
    token = str(split).strip().lower()
    return {"training": "train", "validation": "val", "testing": "test"}.get(token, token)


def _seed_worker(_: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def _phase_schedule(class_count: int, base_classes: int, increment: int) -> Tuple[List[int], Dict[int, List[int]]]:
    total, base, step = int(class_count), int(base_classes), int(increment)
    if total <= 0 or base <= 0 or base > total:
        raise ValueError("class_count and base_classes are invalid")
    if base < total and step <= 0:
        raise ValueError("increment must be positive")
    sizes = [base]
    remaining = total - base
    while remaining:
        current = min(step, remaining)
        sizes.append(current)
        remaining -= current
    phases: Dict[int, List[int]] = {}
    start = 0
    for phase, size in enumerate(sizes):
        phases[phase] = list(range(start, start + size))
        start += size
    return sizes, phases


def _split_counts(
    total: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[int, int, int]:
    count = int(total)
    train_fraction, val_fraction = float(train_ratio), float(val_ratio)
    if count < 3:
        raise ValueError(
            "each class needs at least one train, validation, and test sample"
        )
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("base geometry requires train>0, val>0 and train+val<1")
    # Ratios define the protocol.  The only adjustment is the mathematically
    # necessary one-sample floor for non-empty train/validation/test splits.
    # No per-class sample quota is allowed to override the declared ratio.
    max_train = count - 2
    n_train = min(max(int(round(count * train_fraction)), 1), max_train)
    remaining = count - n_train
    n_val = min(max(int(round(count * val_fraction)), 1), remaining - 1)
    n_test = remaining - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise RuntimeError("requested split is infeasible for a class")
    return n_train, n_val, n_test


def _random_class_split(
    indices: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    n_train, n_val, _ = _split_counts(
        len(values), train_ratio=train_ratio, val_ratio=val_ratio
    )
    shuffled = np.random.RandomState(int(seed) % (2**32)).permutation(values)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {
        "train": train, "val": val, "test": test,
        "all": np.concatenate((train, val, test)),
    }


def _spatial_class_split(
    indices: np.ndarray,
    coords: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    block_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    xy = np.asarray(coords, dtype=np.int64)[values]
    size = int(block_size)
    if size <= 0:
        raise ValueError("spatial_block_size must be positive")
    block_ids = (xy[:, 0] // size).astype(np.int64) * (int(xy[:, 1].max() // size) + 1) + xy[:, 1] // size
    blocks = [values[block_ids == block] for block in np.unique(block_ids)]
    need_groups = 3
    if len(blocks) < need_groups:
        raise RuntimeError(
            "a class occupies too few spatial blocks for the requested split; "
            "reduce spatial_block_size or use random_pixel"
        )
    rng = np.random.RandomState(int(seed) % (2**32))
    rng.shuffle(blocks)
    target_train, target_val, _ = _split_counts(
        len(values), train_ratio=train_ratio, val_ratio=val_ratio
    )
    train_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    train_count = val_count = 0
    for block in blocks:
        remaining_blocks = len(blocks) - len(train_blocks) - len(val_blocks) - len(test_blocks)
        reserve = 2
        if train_count < target_train and remaining_blocks > reserve:
            train_blocks.append(block)
            train_count += len(block)
        elif val_count < target_val and remaining_blocks > 1:
            val_blocks.append(block)
            val_count += len(block)
        else:
            test_blocks.append(block)
    if not train_blocks or not val_blocks or not test_blocks:
        raise RuntimeError("spatial block allocation produced an empty split")
    train = np.concatenate(train_blocks)
    val = np.concatenate(val_blocks) if val_blocks else np.empty(0, dtype=np.int64)
    test = np.concatenate(test_blocks)
    return {"train": train, "val": val, "test": test, "all": np.concatenate((train, val, test))}


def _validate_split_by_class(
    split_by_class: Mapping[int, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    class_count: int,
) -> None:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if set(int(value) for value in split_by_class) != set(range(int(class_count))):
        raise ValueError("split_by_class does not cover every class")
    used: set[int] = set()
    for class_id in range(int(class_count)):
        row = split_by_class[class_id]
        arrays = {name: np.asarray(row[name], dtype=np.int64).reshape(-1) for name in (*_SPLITS, "all")}
        local_sets = [set(arrays[name].tolist()) for name in _SPLITS]
        if local_sets[0] & local_sets[1] or local_sets[0] & local_sets[2] or local_sets[1] & local_sets[2]:
            raise ValueError(f"class {class_id} train/val/test indices overlap")
        if set(arrays["all"].tolist()) != set.union(*local_sets):
            raise ValueError(f"class {class_id} all indices are inconsistent")
        if any(arrays[name].size == 0 for name in _SPLITS):
            raise ValueError(
                f"class {class_id} requires non-empty train, val, and test splits"
            )
        if any(array.size and not np.all(y[array] == class_id) for array in arrays.values()):
            raise ValueError(f"class {class_id} split contains another class")
        current = set(arrays["all"].tolist())
        if used & current:
            raise ValueError("a sample appears in more than one class split")
        used |= current


def _collect_indices(
    split_by_class: Mapping[int, Mapping[str, np.ndarray]],
    class_ids: Sequence[int],
    split: str,
) -> np.ndarray:
    token = _normalize_split_name(split)
    if token not in {*_SPLITS, "all"}:
        raise ValueError(f"unknown split {split!r}")
    ids = _ids(class_ids, name="class_ids", allow_empty=True)
    if not ids:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.asarray(split_by_class[class_id][token], dtype=np.int64) for class_id in ids])


def build_incremental_protocol(
    input_labels: np.ndarray,
    input_coords: np.ndarray,
    *,
    gt_shape: Sequence[int],
    class_count: int,
    base_classes: int,
    increment: int,
    train_ratio: float,
    val_ratio: float,
    seed: int = 0,
    shuffle_order: bool = False,
    class_order: Optional[Sequence[int]] = None,
    target_names: Optional[Sequence[str]] = None,
    split_strategy: str = "random_pixel",
    spatial_block_size: int = 33,
    predefined_splits: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
    **_: Any,
) -> Dict[str, Any]:
    original_labels = np.asarray(input_labels, dtype=np.int64).reshape(-1)
    coords = np.asarray(input_coords, dtype=np.int64)
    shape = tuple(int(value) for value in gt_shape)
    if len(shape) != 2 or coords.shape != (len(original_labels), 2):
        raise ValueError("labels, coordinates and gt_shape are inconsistent")
    original_ids = sorted(int(value) for value in np.unique(original_labels))
    if original_ids != list(range(int(class_count))):
        raise ValueError("input labels must be contiguous zero-based class IDs")

    if class_order is not None:
        order = _ids(class_order, name="class_order")
    elif shuffle_order:
        order = np.random.RandomState(int(seed) % (2**32)).permutation(original_ids).tolist()
    else:
        order = original_ids
    if set(order) != set(original_ids) or len(order) != len(original_ids):
        raise ValueError("class_order must be a complete class permutation")

    original_to_global = {original: global_id for global_id, original in enumerate(order)}
    global_to_original = {global_id: original for original, global_id in original_to_global.items()}
    global_labels = np.asarray([original_to_global[int(label)] for label in original_labels], dtype=np.int64)
    phase_sizes, phase_to_classes = _phase_schedule(class_count, base_classes, increment)

    strategy = str(split_strategy).strip().lower()
    if predefined_splits is not None:
        split_by_class = {
            int(class_id): {
                name: np.asarray(row[name], dtype=np.int64).reshape(-1)
                for name in _SPLITS
            }
            for class_id, row in predefined_splits.items()
        }
        for row in split_by_class.values():
            row["all"] = np.concatenate([row[name] for name in _SPLITS])
        strategy = "predefined"
    elif strategy == "random_pixel":
        split_by_class = {
            class_id: _random_class_split(
                np.flatnonzero(global_labels == class_id), train_ratio=train_ratio,
                val_ratio=val_ratio, seed=seed + 1009 * class_id,
            )
            for class_id in range(int(class_count))
        }
    elif strategy == "spatial_block":
        split_by_class = {
            class_id: _spatial_class_split(
                np.flatnonzero(global_labels == class_id), coords, train_ratio=train_ratio,
                val_ratio=val_ratio,
                block_size=spatial_block_size, seed=seed + 1009 * class_id,
            )
            for class_id in range(int(class_count))
        }
    else:
        raise ValueError("split_strategy must be random_pixel, spatial_block, or predefined")
    _validate_split_by_class(split_by_class, global_labels, class_count)

    names = [str(value) for value in (target_names or [])]
    if names and len(names) != int(class_count):
        raise ValueError("target_names must contain one name per class")
    global_names = [names[original] for original in order] if names else []
    phase_train = {phase: _collect_indices(split_by_class, classes, "train") for phase, classes in phase_to_classes.items()}
    phase_val = {phase: _collect_indices(split_by_class, classes, "val") for phase, classes in phase_to_classes.items()}
    phase_test = {phase: _collect_indices(split_by_class, classes, "test") for phase, classes in phase_to_classes.items()}
    cumulative_test = {
        phase: _collect_indices(split_by_class, list(range(sum(phase_sizes[:phase + 1]))), "test")
        for phase in phase_to_classes
    }
    return {
        "training_mode": "class_incremental",
        "seed": int(seed), "class_count": int(class_count), "base_classes": int(base_classes),
        "increment": int(increment), "phase_sizes": phase_sizes, "phase_to_classes": phase_to_classes,
        "class_order_original_ids": order, "original_to_global": original_to_global,
        "global_to_original": global_to_original, "original_labels": original_labels,
        "global_labels": global_labels, "global_target_names": global_names,
        "split_by_class": split_by_class, "phase_train_indices": phase_train,
        "phase_val_indices": phase_val, "phase_test_indices": phase_test,
        "cumulative_test_indices": cumulative_test, "base_train_indices": phase_train[0].copy(),
        "split_strategy": strategy,
        "spatial_partition_mode": (
            "classwise_block_assignment"
            if strategy == "spatial_block"
            else "pixel_indices"
        ),
        "single_head_global_labels": True,
    }


def build_base_protocol(*args: Any, class_count: int, base_classes: int, **kwargs: Any) -> Dict[str, Any]:
    return build_incremental_protocol(
        *args, class_count=class_count, base_classes=base_classes,
        increment=max(int(class_count) - int(base_classes), 1), **kwargs,
    )


class IncrementalHSIPatchDataset(Dataset):
    def __init__(self, manager: "IncrementalHSIDatasetManager", indices: Sequence[int]) -> None:
        self.manager = manager
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self.indices.size == 0:
            raise ValueError("dataset indices cannot be empty")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        sample_index = int(self.indices[int(item)])
        row, col = self.manager.coords[sample_index]
        patch = self.manager.extract_patch(int(row), int(col))
        raw_center_spectrum = self.manager.ordered_spectral_cube[int(row), int(col)]
        return {
            "image": torch.from_numpy(np.moveaxis(patch, -1, 0).copy()).float(),
            "raw_center_spectrum": torch.from_numpy(
                raw_center_spectrum.copy()
            ).float(),
            "label": torch.tensor(int(self.manager.labels[sample_index]), dtype=torch.long),
            "coord": torch.tensor((int(row), int(col)), dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }


class IncrementalHSIDatasetManager:
    """Select current-class training data and seen-class evaluation data."""

    def __init__(
        self,
        *,
        processed_cube: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        protocol: Mapping[str, Any],
        target_names: Sequence[str],
        gt_shape: Sequence[int],
        patch_size: int,
        ordered_spectral_cube: Optional[np.ndarray] = None,
        num_workers: int = 0,
        device: str = "cpu",
        seed: int = 0,
        require_patch_disjoint: bool = False,
        context_policy: str = "full_scene_reflect",
        **_: Any,
    ) -> None:
        self.processed_cube = np.ascontiguousarray(processed_cube, dtype=np.float32)
        self.gt_shape = tuple(int(value) for value in gt_shape)
        self.coords = np.asarray(coords, dtype=np.int64)
        supplied_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        original_labels = np.asarray(protocol["original_labels"], dtype=np.int64).reshape(-1)
        global_labels = np.asarray(protocol["global_labels"], dtype=np.int64).reshape(-1)
        if np.array_equal(supplied_labels, original_labels) or np.array_equal(supplied_labels, global_labels):
            self.labels = global_labels
        else:
            raise ValueError("labels do not match the protocol")
        if self.processed_cube.ndim != 3 or self.processed_cube.shape[:2] != self.gt_shape:
            raise ValueError("processed_cube and gt_shape are incompatible")
        if not np.isfinite(self.processed_cube).all():
            raise ValueError("processed_cube contains NaN/Inf")
        if self.coords.shape != (len(self.labels), 2):
            raise ValueError("coords must align with labels")
        if np.any(self.coords < 0) or np.any(
            self.coords >= np.asarray(self.gt_shape, dtype=np.int64)
        ):
            raise ValueError("coords contain positions outside gt_shape")
        if ordered_spectral_cube is None:
            raise ValueError(
                "ordered_spectral_cube is required for the full-band spectral branch"
            )
        ordered = np.asarray(ordered_spectral_cube, dtype=np.float32)
        if (
            ordered.ndim != 3
            or ordered.shape[:2] != self.gt_shape
            or ordered.shape[2] <= 0
        ):
            raise ValueError("ordered_spectral_cube has an incompatible shape")
        if not np.isfinite(ordered).all():
            raise ValueError("ordered_spectral_cube contains NaN/Inf")
        self.ordered_spectral_cube = np.ascontiguousarray(ordered)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be positive and odd")
        radius = self.patch_size // 2
        self.padded_cube = np.pad(
            self.processed_cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect"
        )
        self.phase_to_classes = {
            int(phase): _ids(classes, name=f"phase_to_classes[{phase}]")
            for phase, classes in protocol["phase_to_classes"].items()
        }
        self._split_by_class = {
            int(class_id): {name: np.asarray(values, dtype=np.int64) for name, values in row.items()}
            for class_id, row in protocol["split_by_class"].items()
        }
        _validate_split_by_class(self._split_by_class, self.labels, int(protocol["class_count"]))
        names = [str(value) for value in target_names]
        global_names = [str(value) for value in protocol.get("global_target_names", [])]
        class_count = int(protocol["class_count"])
        if global_names and len(global_names) != class_count:
            raise ValueError("protocol global_target_names has the wrong length")
        if not global_names and len(names) != class_count:
            raise ValueError("target_names must contain one name per class")
        self.target_names = global_names or [names[index] for index in protocol["class_order_original_ids"]]
        self.class_order_original_ids = [int(value) for value in protocol["class_order_original_ids"]]
        self.global_to_original = {int(k): int(v) for k, v in protocol["global_to_original"].items()}
        self.split_strategy = str(protocol["split_strategy"])
        self.spatial_partition_mode = str(
            protocol.get("spatial_partition_mode", "pixel_indices")
        )
        self.require_patch_disjoint = bool(require_patch_disjoint)
        self.context_policy = str(context_policy).strip()
        if self.context_policy != "full_scene_reflect":
            raise ValueError(
                "this manager implements context_policy='full_scene_reflect' only"
            )
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.pin_memory = str(device).startswith("cuda")
        self.current_phase: Optional[int] = None
        self.finalized_phases: List[int] = []

        if self.require_patch_disjoint:
            self._assert_patch_disjoint_splits()

    @property
    def nb_tasks(self) -> int:
        return len(self.phase_to_classes)

    @property
    def base_classes(self) -> List[int]:
        return list(self.phase_to_classes[0])

    def _validate_phase(self, phase: int) -> int:
        value = int(phase)
        if value not in self.phase_to_classes:
            raise ValueError(f"unknown phase {phase}")
        return value

    def start_phase(self, phase: int) -> None:
        value = self._validate_phase(phase)
        if self.current_phase is not None:
            raise RuntimeError(f"phase {self.current_phase} is already active")
        if value in self.finalized_phases:
            raise RuntimeError(f"phase {value} is already finalized")
        if self.finalized_phases != list(range(value)):
            raise RuntimeError("phases must start in order after prior finalization")
        self.current_phase = value

    def finalize_phase(self, phase: int) -> None:
        value = self._validate_phase(phase)
        if self.current_phase != value:
            raise RuntimeError(f"phase {value} is not the active phase")
        self.finalized_phases.append(value)
        self.current_phase = None

    def get_task_size(self, phase: int) -> int:
        return len(self.phase_to_classes[self._validate_phase(phase)])

    def get_new_classes(self, phase: int) -> List[int]:
        return list(self.phase_to_classes[self._validate_phase(phase)])

    def get_old_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        return [class_id for previous in range(phase) for class_id in self.phase_to_classes[previous]]

    def get_seen_classes(self, phase: int) -> List[int]:
        return self.get_old_classes(phase) + self.get_new_classes(phase)

    def extract_patch(self, row: int, col: int) -> np.ndarray:
        size = self.patch_size
        return self.padded_cube[int(row):int(row) + size, int(col):int(col) + size]

    def _assert_patch_disjoint_splits(self) -> None:
        """Verify that no patch windows overlap across train/val/test splits."""
        split_indices = {
            split: _collect_indices(
                self._split_by_class,
                list(range(len(self._split_by_class))),
                split,
            )
            for split in _SPLITS
        }
        for left_index, left_name in enumerate(_SPLITS):
            left = self.coords[split_indices[left_name]]
            for right_name in _SPLITS[left_index + 1 :]:
                right = self.coords[split_indices[right_name]]
                for start in range(0, len(left), 2048):
                    delta = np.abs(
                        left[start : start + 2048, None, :] - right[None, :, :]
                    )
                    if np.any(np.all(delta < self.patch_size, axis=2)):
                        raise RuntimeError(
                            f"{left_name}/{right_name} patches overlap; choose a "
                            "guarded predefined split or set require_patch_disjoint=False"
                        )

    def get_dataset(self, class_ids: Sequence[int], split: str) -> IncrementalHSIPatchDataset:
        return IncrementalHSIPatchDataset(self, _collect_indices(self._split_by_class, class_ids, split))

    def _loader(self, dataset: Dataset, *, batch_size: int, shuffle: bool, phase: int, split: str) -> DataLoader:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        generator = torch.Generator().manual_seed(self.seed + 1009 * int(phase) + {"train": 11, "val": 17, "test": 23, "all": 29}[split])
        return DataLoader(
            dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=False,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
            generator=generator,
        )

    def get_phase_dataloader(
        self, phase: int, split: str = "train", batch_size: int = 64, shuffle: bool = True,
    ) -> DataLoader:
        phase = self._validate_phase(phase)
        if self.current_phase != phase:
            raise RuntimeError("phase train/validation loaders require an active phase")
        token = _normalize_split_name(split)
        if token not in {"train", "val"}:
            raise ValueError("active phase loaders are restricted to train or val")
        dataset = self.get_dataset(self.get_new_classes(phase), token)
        return self._loader(dataset, batch_size=batch_size, shuffle=shuffle if token == "train" else False, phase=phase, split=token)

    def get_cumulative_dataloader(
        self, phase: int, split: str = "test", batch_size: int = 256, shuffle: bool = False,
    ) -> DataLoader:
        phase = self._validate_phase(phase)
        if phase not in self.finalized_phases:
            raise RuntimeError("cumulative evaluation requires a finalized phase")
        token = _normalize_split_name(split)
        if token not in {"val", "test"}:
            raise ValueError("cumulative loaders are evaluation-only")
        dataset = self.get_dataset(self.get_seen_classes(phase), token)
        return self._loader(dataset, batch_size=batch_size, shuffle=shuffle, phase=phase, split=token)

    def get_reporting_dataloader(
        self, phase: int, *, split: str = "test", batch_size: int = 256,
    ) -> DataLoader:
        phase = self._validate_phase(phase)
        if phase not in self.finalized_phases:
            raise RuntimeError("reporting requires a finalized phase")
        token = _normalize_split_name(split)
        if token not in {"test", "all"}:
            raise ValueError("reporting split must be test or all")
        dataset = self.get_dataset(self.get_seen_classes(phase), token)
        return self._loader(dataset, batch_size=batch_size, shuffle=False, phase=phase, split=token)

    def protocol_report(self) -> Dict[str, Any]:
        return {
            "tasks": self.nb_tasks,
            "phase_to_classes": {phase: list(classes) for phase, classes in self.phase_to_classes.items()},
            "class_order_original_ids": list(self.class_order_original_ids),
            "split_strategy": self.split_strategy,
            "spatial_partition_mode": self.spatial_partition_mode,
            "require_patch_disjoint": self.require_patch_disjoint,
            "context_policy": self.context_policy,
            "current_phase": self.current_phase,
            "finalized_phases": list(self.finalized_phases),
            "single_head_global_labels": True,
            "current_phase_training_classes_only": True,
            "cumulative_seen_class_evaluation": True,
            "stores_old_training_samples": False,
        }

    def assert_exemplar_free_contract(self) -> bool:
        if self.finalized_phases != list(range(len(self.finalized_phases))):
            raise RuntimeError("finalized phase state is not contiguous")
        if self.current_phase is not None and self.current_phase in self.finalized_phases:
            raise RuntimeError("an active phase cannot also be finalized")
        if any(
            token in self.__dict__
            for token in ("exemplars", "replay_buffer", "memory_samples")
        ):
            raise RuntimeError("dataset manager contains forbidden replay state")
        return True


__all__ = [
    "build_incremental_protocol", "build_base_protocol", "IncrementalHSIPatchDataset",
    "IncrementalHSIDatasetManager",
]












# from __future__ import annotations

# """class-incremental HSI protocol and patch data manager.

# The protocol follows the common NECIL evaluation setup: each phase trains on
# the current classes, while evaluation uses one global label space over all
# classes seen so far.  The spatial branch receives a processed patch, while the
# spectral branch receives the raw, ordered, full-band centre spectrum.  No old
# training samples are returned by a phase-training API.
# """

# import random
# from numbers import Integral, Real
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset


# _SPLITS = ("train", "val", "test")


# def _exact_integer(value: object, *, name: str) -> int:
#     """Convert an integer-like scalar without silently truncating it."""
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer scalar")
#         value = value.item()
#     if isinstance(value, (bool, np.bool_)):
#         raise ValueError(f"{name} must be an integer")
#     if isinstance(value, Integral):
#         return int(value)
#     if isinstance(value, Real):
#         number = float(value)
#         if np.isfinite(number) and number.is_integer():
#             return int(number)
#     raise ValueError(f"{name} must be an integer")


# def _integer_array(value: object, *, name: str) -> np.ndarray:
#     """Return int64 data without accepting fractional, Boolean, or NaN IDs."""
#     raw = np.asarray(value)
#     if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
#         raise ValueError(f"{name} must contain integer values")
#     if np.issubdtype(raw.dtype, np.floating):
#         if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
#             raise ValueError(f"{name} must contain finite integer values")
#     if raw.size and np.issubdtype(raw.dtype, np.integer):
#         lower, upper = np.iinfo(np.int64).min, np.iinfo(np.int64).max
#         if np.any(raw < lower) or np.any(raw > upper):
#             raise ValueError(f"{name} contains values outside int64 range")
#     try:
#         converted = raw.astype(np.int64, copy=False)
#     except (TypeError, ValueError, OverflowError) as exc:
#         raise ValueError(f"{name} must contain integer values") from exc
#     return converted


# def _ids(values: Iterable[int], *, name: str, allow_empty: bool = False) -> List[int]:
#     result = [
#         _exact_integer(value, name=f"{name} value")
#         for value in values
#     ]
#     if not result and not allow_empty:
#         raise ValueError(f"{name} cannot be empty")
#     if len(result) != len(set(result)) or any(value < 0 for value in result):
#         raise ValueError(f"{name} must contain unique non-negative IDs")
#     return result


# def _normalize_split_name(split: str) -> str:
#     token = str(split).strip().lower()
#     return {"training": "train", "validation": "val", "testing": "test"}.get(token, token)


# def _seed_worker(_: int) -> None:
#     seed = int(torch.initial_seed() % (2**32))
#     random.seed(seed)
#     np.random.seed(seed)


# def _phase_schedule(class_count: int, base_classes: int, increment: int) -> Tuple[List[int], Dict[int, List[int]]]:
#     total = _exact_integer(class_count, name="class_count")
#     base = _exact_integer(base_classes, name="base_classes")
#     step = _exact_integer(increment, name="increment")
#     if total <= 0 or base <= 0 or base > total:
#         raise ValueError("class_count and base_classes are invalid")
#     if base < total and step <= 0:
#         raise ValueError("increment must be positive")
#     sizes = [base]
#     remaining = total - base
#     while remaining:
#         current = min(step, remaining)
#         sizes.append(current)
#         remaining -= current
#     phases: Dict[int, List[int]] = {}
#     start = 0
#     for phase, size in enumerate(sizes):
#         phases[phase] = list(range(start, start + size))
#         start += size
#     return sizes, phases


# def _split_counts(
#     total: int,
#     *,
#     train_ratio: float,
#     val_ratio: float,
# ) -> Tuple[int, int, int]:
#     count = _exact_integer(total, name="class sample count")
#     train_fraction, val_fraction = float(train_ratio), float(val_ratio)
#     if count < 3:
#         raise ValueError(
#             "each class needs at least three samples for non-empty "
#             "train/validation/test splits"
#         )
#     if (
#         not np.isfinite(train_fraction)
#         or not np.isfinite(val_fraction)
#         or train_fraction <= 0
#         or val_fraction <= 0
#         or train_fraction + val_fraction >= 1
#     ):
#         raise ValueError("base geometry requires train>0, val>0 and train+val<1")

#     # Hamilton allocation gives the integer partition closest to the declared
#     # three-way proportions while preserving the exact class total. It adds no
#     # per-class floor and avoids Python's tie-to-even rounding behaviour.
#     proportions = np.asarray(
#         (train_fraction, val_fraction, 1.0 - train_fraction - val_fraction),
#         dtype=np.float64,
#     )
#     quotas = proportions * count
#     counts = np.floor(quotas).astype(np.int64)
#     remainder = count - int(counts.sum())
#     if remainder:
#         order = np.argsort(-(quotas - counts), kind="stable")
#         counts[order[:remainder]] += 1
#     n_train, n_val, n_test = (int(value) for value in counts)
#     if min(n_train, n_val, n_test) <= 0:
#         raise ValueError(
#             "declared train/validation ratios produce an empty split for a "
#             f"class with {count} samples; change the published split protocol "
#             "explicitly rather than applying a hidden per-class floor"
#         )
#     return n_train, n_val, n_test


# def _random_class_split(
#     indices: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     seed: int,
# ) -> Dict[str, np.ndarray]:
#     values = _integer_array(indices, name="class sample indices").reshape(-1)
#     n_train, n_val, _ = _split_counts(
#         len(values), train_ratio=train_ratio, val_ratio=val_ratio
#     )
#     shuffled = np.random.RandomState(int(seed) % (2**32)).permutation(values)
#     train = shuffled[:n_train]
#     val = shuffled[n_train:n_train + n_val]
#     test = shuffled[n_train + n_val:]
#     return {
#         "train": train, "val": val, "test": test,
#         "all": np.concatenate((train, val, test)),
#     }


# def _spatial_class_split(
#     indices: np.ndarray,
#     coords: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     block_size: int,
#     seed: int,
# ) -> Dict[str, np.ndarray]:
#     values = _integer_array(indices, name="class sample indices").reshape(-1)
#     xy = _integer_array(coords, name="sample coordinates")[values]
#     size = _exact_integer(block_size, name="spatial_block_size")
#     if size <= 0:
#         raise ValueError("spatial_block_size must be positive")
#     block_ids = (xy[:, 0] // size).astype(np.int64) * (int(xy[:, 1].max() // size) + 1) + xy[:, 1] // size
#     blocks = [values[block_ids == block] for block in np.unique(block_ids)]
#     need_groups = 3
#     if len(blocks) < need_groups:
#         raise RuntimeError(
#             "a class occupies too few spatial blocks for the requested split; "
#             "reduce spatial_block_size or use random_pixel"
#         )
#     rng = np.random.RandomState(int(seed) % (2**32))
#     rng.shuffle(blocks)
#     target_train, target_val, _ = _split_counts(
#         len(values), train_ratio=train_ratio, val_ratio=val_ratio
#     )
#     train_blocks: List[np.ndarray] = []
#     val_blocks: List[np.ndarray] = []
#     test_blocks: List[np.ndarray] = []
#     train_count = val_count = 0
#     for block in blocks:
#         remaining_blocks = len(blocks) - len(train_blocks) - len(val_blocks) - len(test_blocks)
#         reserve = 2
#         if train_count < target_train and remaining_blocks > reserve:
#             train_blocks.append(block)
#             train_count += len(block)
#         elif val_count < target_val and remaining_blocks > 1:
#             val_blocks.append(block)
#             val_count += len(block)
#         else:
#             test_blocks.append(block)
#     if not train_blocks or not val_blocks or not test_blocks:
#         raise RuntimeError("spatial block allocation produced an empty split")
#     train = np.concatenate(train_blocks)
#     val = np.concatenate(val_blocks) if val_blocks else np.empty(0, dtype=np.int64)
#     test = np.concatenate(test_blocks)
#     return {"train": train, "val": val, "test": test, "all": np.concatenate((train, val, test))}


# def _validate_split_by_class(
#     split_by_class: Mapping[int, Mapping[str, np.ndarray]],
#     labels: np.ndarray,
#     class_count: int,
# ) -> None:
#     y = _integer_array(labels, name="global labels").reshape(-1)
#     count = _exact_integer(class_count, name="class_count")
#     split_ids = {
#         _exact_integer(value, name="split class ID")
#         for value in split_by_class
#     }
#     if split_ids != set(range(count)):
#         raise ValueError("split_by_class does not cover every class")
#     used: set[int] = set()
#     for class_id in range(count):
#         row = split_by_class[class_id]
#         missing_names = set((*_SPLITS, "all")) - set(row)
#         if missing_names:
#             raise ValueError(
#                 f"class {class_id} split is missing {sorted(missing_names)}"
#             )
#         arrays = {
#             name: _integer_array(
#                 row[name],
#                 name=f"class {class_id} {name} indices",
#             ).reshape(-1)
#             for name in (*_SPLITS, "all")
#         }
#         for name, array in arrays.items():
#             if np.unique(array).size != array.size:
#                 raise ValueError(
#                     f"class {class_id} {name} split contains duplicate indices"
#                 )
#             if array.size and (array.min() < 0 or array.max() >= y.size):
#                 raise ValueError(
#                     f"class {class_id} {name} split contains invalid indices"
#                 )
#         local_sets = [set(arrays[name].tolist()) for name in _SPLITS]
#         if local_sets[0] & local_sets[1] or local_sets[0] & local_sets[2] or local_sets[1] & local_sets[2]:
#             raise ValueError(f"class {class_id} train/val/test indices overlap")
#         if set(arrays["all"].tolist()) != set.union(*local_sets):
#             raise ValueError(f"class {class_id} all indices are inconsistent")
#         if any(arrays[name].size == 0 for name in _SPLITS):
#             raise ValueError(
#                 f"class {class_id} requires non-empty train, val, and test splits"
#             )
#         if any(array.size and not np.all(y[array] == class_id) for array in arrays.values()):
#             raise ValueError(f"class {class_id} split contains another class")
#         current = set(arrays["all"].tolist())
#         expected = set(np.flatnonzero(y == class_id).tolist())
#         if current != expected:
#             raise ValueError(
#                 f"class {class_id} split does not account for every sample"
#             )
#         if used & current:
#             raise ValueError("a sample appears in more than one class split")
#         used |= current


# def _collect_indices(
#     split_by_class: Mapping[int, Mapping[str, np.ndarray]],
#     class_ids: Sequence[int],
#     split: str,
# ) -> np.ndarray:
#     token = _normalize_split_name(split)
#     if token not in {*_SPLITS, "all"}:
#         raise ValueError(f"unknown split {split!r}")
#     ids = _ids(class_ids, name="class_ids", allow_empty=True)
#     if not ids:
#         return np.empty(0, dtype=np.int64)
#     return np.concatenate(
#         [
#             _integer_array(
#                 split_by_class[class_id][token],
#                 name=f"class {class_id} {token} indices",
#             ).reshape(-1)
#             for class_id in ids
#         ]
#     )


# def build_incremental_protocol(
#     input_labels: np.ndarray,
#     input_coords: np.ndarray,
#     *,
#     gt_shape: Sequence[int],
#     class_count: int,
#     base_classes: int,
#     increment: int,
#     train_ratio: float,
#     val_ratio: float,
#     seed: int = 0,
#     shuffle_order: bool = False,
#     class_order: Optional[Sequence[int]] = None,
#     target_names: Optional[Sequence[str]] = None,
#     split_strategy: str = "random_pixel",
#     spatial_block_size: int = 33,
#     predefined_splits: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
# ) -> Dict[str, Any]:
#     original_labels = _integer_array(
#         input_labels,
#         name="input_labels",
#     ).reshape(-1)
#     coords = _integer_array(input_coords, name="input_coords")
#     shape = tuple(
#         _exact_integer(value, name="gt_shape value") for value in gt_shape
#     )
#     protocol_seed = _exact_integer(seed, name="seed")
#     if protocol_seed < 0:
#         raise ValueError("seed must be non-negative")
#     if len(shape) != 2 or coords.shape != (len(original_labels), 2):
#         raise ValueError("labels, coordinates and gt_shape are inconsistent")
#     if not original_labels.size or min(shape) <= 0:
#         raise ValueError("labels and gt_shape must be non-empty")
#     if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
#         raise ValueError("coordinates must identify unique labelled pixels")
#     if (
#         np.any(coords[:, 0] < 0)
#         or np.any(coords[:, 0] >= shape[0])
#         or np.any(coords[:, 1] < 0)
#         or np.any(coords[:, 1] >= shape[1])
#     ):
#         raise ValueError("coordinates lie outside gt_shape")
#     total_classes = _exact_integer(class_count, name="class_count")
#     original_ids = sorted(int(value) for value in np.unique(original_labels))
#     if original_ids != list(range(total_classes)):
#         raise ValueError("input labels must be contiguous zero-based class IDs")

#     if class_order is not None and bool(shuffle_order):
#         raise ValueError("choose either class_order or shuffle_order, not both")
#     if class_order is not None:
#         order = _ids(class_order, name="class_order")
#     elif shuffle_order:
#         order = np.random.RandomState(protocol_seed).permutation(original_ids).tolist()
#     else:
#         order = original_ids
#     if set(order) != set(original_ids) or len(order) != len(original_ids):
#         raise ValueError("class_order must be a complete class permutation")

#     original_to_global = {original: global_id for global_id, original in enumerate(order)}
#     global_to_original = {global_id: original for original, global_id in original_to_global.items()}
#     global_labels = np.asarray([original_to_global[int(label)] for label in original_labels], dtype=np.int64)
#     phase_sizes, phase_to_classes = _phase_schedule(class_count, base_classes, increment)

#     strategy = str(split_strategy).strip().lower()
#     if predefined_splits is not None:
#         predefined_original = {
#             _exact_integer(class_id, name="predefined class ID"): row
#             for class_id, row in predefined_splits.items()
#         }
#         if set(predefined_original) != set(original_ids):
#             raise ValueError(
#                 "predefined_splits must be keyed by every original dataset "
#                 "class ID exactly once"
#             )
#         split_by_class = {
#             original_to_global[original_id]: {
#                 name: _integer_array(
#                     row[name],
#                     name=f"predefined class {original_id} {name} indices",
#                 ).reshape(-1)
#                 for name in _SPLITS
#             }
#             for original_id, row in predefined_original.items()
#         }
#         for row in split_by_class.values():
#             row["all"] = np.concatenate([row[name] for name in _SPLITS])
#         strategy = "predefined"
#     elif strategy == "random_pixel":
#         split_by_class = {
#             class_id: _random_class_split(
#                 np.flatnonzero(global_labels == class_id), train_ratio=train_ratio,
#                 val_ratio=val_ratio,
#                 seed=protocol_seed + 1009 * global_to_original[class_id],
#             )
#             for class_id in range(total_classes)
#         }
#     elif strategy == "spatial_block":
#         split_by_class = {
#             class_id: _spatial_class_split(
#                 np.flatnonzero(global_labels == class_id), coords, train_ratio=train_ratio,
#                 val_ratio=val_ratio,
#                 block_size=spatial_block_size,
#                 seed=protocol_seed + 1009 * global_to_original[class_id],
#             )
#             for class_id in range(total_classes)
#         }
#     else:
#         raise ValueError("split_strategy must be random_pixel, spatial_block, or predefined")
#     _validate_split_by_class(split_by_class, global_labels, total_classes)

#     names = (
#         []
#         if target_names is None
#         else [str(value) for value in target_names]
#     )
#     if names and len(names) != total_classes:
#         raise ValueError("target_names must contain one name per class")
#     global_names = [names[original] for original in order] if names else []
#     phase_train = {phase: _collect_indices(split_by_class, classes, "train") for phase, classes in phase_to_classes.items()}
#     phase_val = {phase: _collect_indices(split_by_class, classes, "val") for phase, classes in phase_to_classes.items()}
#     phase_test = {phase: _collect_indices(split_by_class, classes, "test") for phase, classes in phase_to_classes.items()}
#     cumulative_test = {
#         phase: _collect_indices(split_by_class, list(range(sum(phase_sizes[:phase + 1]))), "test")
#         for phase in phase_to_classes
#     }
#     return {
#         "training_mode": "non_exemplar_class_incremental",
#         "seed": protocol_seed, "class_count": total_classes,
#         "base_classes": _exact_integer(base_classes, name="base_classes"),
#         "increment": _exact_integer(increment, name="increment"),
#         "phase_sizes": phase_sizes, "phase_to_classes": phase_to_classes,
#         "class_order_original_ids": order, "original_to_global": original_to_global,
#         "global_to_original": global_to_original, "original_labels": original_labels,
#         "global_labels": global_labels, "global_target_names": global_names,
#         "split_by_class": split_by_class, "phase_train_indices": phase_train,
#         "phase_val_indices": phase_val, "phase_test_indices": phase_test,
#         "cumulative_test_indices": cumulative_test, "base_train_indices": phase_train[0].copy(),
#         "split_strategy": strategy,
#         "spatial_partition_mode": (
#             "classwise_block_assignment"
#             if strategy == "spatial_block"
#             else "pixel_indices"
#         ),
#         "single_head_global_labels": True,
#         "current_phase_training_classes_only": True,
#         "cumulative_seen_class_evaluation": True,
#         "stores_old_training_samples": False,
#     }


# def build_base_protocol(*args: Any, class_count: int, base_classes: int, **kwargs: Any) -> Dict[str, Any]:
#     total = _exact_integer(class_count, name="class_count")
#     base = _exact_integer(base_classes, name="base_classes")
#     return build_incremental_protocol(
#         *args, class_count=total, base_classes=base,
#         increment=max(total - base, 1), **kwargs,
#     )


# class IncrementalHSIPatchDataset(Dataset):
#     def __init__(self, manager: "IncrementalHSIDatasetManager", indices: Sequence[int]) -> None:
#         self.manager = manager
#         self.indices = _integer_array(indices, name="dataset indices").reshape(-1)
#         if self.indices.size == 0:
#             raise ValueError("dataset indices cannot be empty")
#         if self.indices.min() < 0 or self.indices.max() >= len(manager.labels):
#             raise ValueError("dataset indices are outside the labelled samples")
#         if np.unique(self.indices).size != self.indices.size:
#             raise ValueError("dataset indices cannot contain duplicates")

#     def __len__(self) -> int:
#         return int(self.indices.size)

#     def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
#         sample_index = int(self.indices[int(item)])
#         row, col = self.manager.coords[sample_index]
#         patch = self.manager.extract_patch(int(row), int(col))
#         raw_center_spectrum = self.manager.ordered_spectral_cube[int(row), int(col)]
#         return {
#             "image": torch.from_numpy(np.moveaxis(patch, -1, 0).copy()).float(),
#             "raw_center_spectrum": torch.from_numpy(
#                 raw_center_spectrum.copy()
#             ).float(),
#             "label": torch.tensor(int(self.manager.labels[sample_index]), dtype=torch.long),
#             "coord": torch.tensor((int(row), int(col)), dtype=torch.long),
#             "sample_index": torch.tensor(sample_index, dtype=torch.long),
#         }


# class IncrementalHSIDatasetManager:
#     """Select current-class training data and seen-class evaluation data."""

#     def __init__(
#         self,
#         *,
#         processed_cube: np.ndarray,
#         labels: np.ndarray,
#         coords: np.ndarray,
#         protocol: Mapping[str, Any],
#         target_names: Sequence[str],
#         gt_shape: Sequence[int],
#         patch_size: int,
#         ordered_spectral_cube: Optional[np.ndarray] = None,
#         num_workers: int = 0,
#         device: str = "cpu",
#         seed: Optional[int] = None,
#         require_patch_disjoint: bool = False,
#         context_policy: str = "full_scene_reflect",
#     ) -> None:
#         self.processed_cube = np.ascontiguousarray(processed_cube, dtype=np.float32)
#         self.gt_shape = tuple(
#             _exact_integer(value, name="gt_shape value") for value in gt_shape
#         )
#         self.coords = _integer_array(coords, name="coords")
#         supplied_labels = _integer_array(labels, name="labels").reshape(-1)
#         original_labels = _integer_array(
#             protocol["original_labels"],
#             name="protocol original_labels",
#         ).reshape(-1)
#         global_labels = _integer_array(
#             protocol["global_labels"],
#             name="protocol global_labels",
#         ).reshape(-1)
#         if np.array_equal(supplied_labels, original_labels) or np.array_equal(supplied_labels, global_labels):
#             self.labels = global_labels
#         else:
#             raise ValueError("labels do not match the protocol")
#         if self.processed_cube.ndim != 3 or self.processed_cube.shape[:2] != self.gt_shape:
#             raise ValueError("processed_cube and gt_shape are incompatible")
#         if not np.isfinite(self.processed_cube).all():
#             raise ValueError("processed_cube contains NaN/Inf")
#         if self.coords.shape != (len(self.labels), 2):
#             raise ValueError("coords must align with labels")
#         if np.unique(self.coords, axis=0).shape[0] != self.coords.shape[0]:
#             raise ValueError("coords contain duplicate labelled-pixel locations")
#         if np.any(self.coords < 0) or np.any(
#             self.coords >= np.asarray(self.gt_shape, dtype=np.int64)
#         ):
#             raise ValueError("coords contain positions outside gt_shape")
#         if ordered_spectral_cube is None:
#             raise ValueError(
#                 "ordered_spectral_cube is required for the full-band spectral branch"
#             )
#         ordered = np.asarray(ordered_spectral_cube, dtype=np.float32)
#         if (
#             ordered.ndim != 3
#             or ordered.shape[:2] != self.gt_shape
#             or ordered.shape[2] <= 0
#         ):
#             raise ValueError("ordered_spectral_cube has an incompatible shape")
#         if not np.isfinite(ordered).all():
#             raise ValueError("ordered_spectral_cube contains NaN/Inf")
#         self.ordered_spectral_cube = np.ascontiguousarray(ordered)
#         self.patch_size = _exact_integer(patch_size, name="patch_size")
#         if self.patch_size <= 0 or self.patch_size % 2 == 0:
#             raise ValueError("patch_size must be positive and odd")
#         radius = self.patch_size // 2
#         self.padded_cube = np.pad(
#             self.processed_cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect"
#         )
#         self.phase_to_classes = {
#             _exact_integer(phase, name="phase ID"): _ids(
#                 classes,
#                 name=f"phase_to_classes[{phase}]",
#             )
#             for phase, classes in protocol["phase_to_classes"].items()
#         }
#         if sorted(self.phase_to_classes) != list(range(len(self.phase_to_classes))):
#             raise ValueError("phase IDs must be contiguous from zero")
#         flattened = [
#             class_id
#             for phase in range(len(self.phase_to_classes))
#             for class_id in self.phase_to_classes[phase]
#         ]
#         class_count = _exact_integer(
#             protocol["class_count"],
#             name="protocol class_count",
#         )
#         if flattened != list(range(class_count)):
#             raise ValueError(
#                 "phase schedule must cover global class IDs once in order"
#             )
#         self._split_by_class = {
#             _exact_integer(class_id, name="split class ID"): {
#                 name: _integer_array(
#                     values,
#                     name=f"class {class_id} {name} indices",
#                 )
#                 for name, values in row.items()
#             }
#             for class_id, row in protocol["split_by_class"].items()
#         }
#         _validate_split_by_class(self._split_by_class, self.labels, class_count)
#         names = [str(value) for value in target_names]
#         global_names = [str(value) for value in protocol.get("global_target_names", [])]
#         if global_names and len(global_names) != class_count:
#             raise ValueError("protocol global_target_names has the wrong length")
#         if not global_names and len(names) != class_count:
#             raise ValueError("target_names must contain one name per class")
#         self.target_names = global_names or [names[index] for index in protocol["class_order_original_ids"]]
#         self.class_order_original_ids = _ids(
#             protocol["class_order_original_ids"],
#             name="class_order_original_ids",
#         )
#         self.global_to_original = {
#             _exact_integer(key, name="global class ID"): _exact_integer(
#                 value,
#                 name="original class ID",
#             )
#             for key, value in protocol["global_to_original"].items()
#         }
#         self.split_strategy = str(protocol["split_strategy"])
#         self.spatial_partition_mode = str(
#             protocol.get("spatial_partition_mode", "pixel_indices")
#         )
#         self.require_patch_disjoint = bool(require_patch_disjoint)
#         self.context_policy = str(context_policy).strip()
#         if self.context_policy != "full_scene_reflect":
#             raise ValueError(
#                 "this manager implements context_policy='full_scene_reflect' only"
#             )
#         protocol_seed = _exact_integer(protocol["seed"], name="protocol seed")
#         self.seed = (
#             protocol_seed
#             if seed is None
#             else _exact_integer(seed, name="dataset-manager seed")
#         )
#         if self.seed < 0:
#             raise ValueError("seed must be non-negative")
#         if self.seed != protocol_seed:
#             raise ValueError("dataset-manager seed disagrees with protocol seed")
#         self.num_workers = _exact_integer(num_workers, name="num_workers")
#         if self.num_workers < 0:
#             raise ValueError("num_workers must be non-negative")
#         self.pin_memory = str(device).startswith("cuda")
#         self.current_phase: Optional[int] = None
#         self.finalized_phases: List[int] = []

#         if self.require_patch_disjoint:
#             self._assert_patch_disjoint_splits()

#     @property
#     def nb_tasks(self) -> int:
#         return len(self.phase_to_classes)

#     @property
#     def base_classes(self) -> List[int]:
#         return list(self.phase_to_classes[0])

#     def _validate_phase(self, phase: int) -> int:
#         value = _exact_integer(phase, name="phase")
#         if value not in self.phase_to_classes:
#             raise ValueError(f"unknown phase {phase}")
#         return value

#     def start_phase(self, phase: int) -> None:
#         value = self._validate_phase(phase)
#         if self.current_phase is not None:
#             raise RuntimeError(f"phase {self.current_phase} is already active")
#         if value in self.finalized_phases:
#             raise RuntimeError(f"phase {value} is already finalized")
#         if self.finalized_phases != list(range(value)):
#             raise RuntimeError("phases must start in order after prior finalization")
#         self.current_phase = value

#     def finalize_phase(self, phase: int) -> None:
#         value = self._validate_phase(phase)
#         if self.current_phase != value:
#             raise RuntimeError(f"phase {value} is not the active phase")
#         self.finalized_phases.append(value)
#         self.current_phase = None

#     def get_task_size(self, phase: int) -> int:
#         return len(self.phase_to_classes[self._validate_phase(phase)])

#     def get_new_classes(self, phase: int) -> List[int]:
#         return list(self.phase_to_classes[self._validate_phase(phase)])

#     def get_old_classes(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         return [class_id for previous in range(phase) for class_id in self.phase_to_classes[previous]]

#     def get_seen_classes(self, phase: int) -> List[int]:
#         return self.get_old_classes(phase) + self.get_new_classes(phase)

#     def extract_patch(self, row: int, col: int) -> np.ndarray:
#         row_id = _exact_integer(row, name="patch row")
#         col_id = _exact_integer(col, name="patch column")
#         if not (
#             0 <= row_id < self.gt_shape[0]
#             and 0 <= col_id < self.gt_shape[1]
#         ):
#             raise ValueError("patch center lies outside gt_shape")
#         size = self.patch_size
#         return self.padded_cube[
#             row_id : row_id + size,
#             col_id : col_id + size,
#         ]

#     def _assert_patch_disjoint_splits(self) -> None:
#         """Verify that no patch windows overlap across train/val/test splits."""
#         split_indices = {
#             split: _collect_indices(
#                 self._split_by_class,
#                 list(range(len(self._split_by_class))),
#                 split,
#             )
#             for split in _SPLITS
#         }
#         for left_index, left_name in enumerate(_SPLITS):
#             left = self.coords[split_indices[left_name]]
#             for right_name in _SPLITS[left_index + 1 :]:
#                 right = self.coords[split_indices[right_name]]
#                 for start in range(0, len(left), 2048):
#                     delta = np.abs(
#                         left[start : start + 2048, None, :] - right[None, :, :]
#                     )
#                     if np.any(np.all(delta < self.patch_size, axis=2)):
#                         raise RuntimeError(
#                             f"{left_name}/{right_name} patches overlap; choose a "
#                             "guarded predefined split or set require_patch_disjoint=False"
#                         )

#     def _get_dataset(self, class_ids: Sequence[int], split: str) -> IncrementalHSIPatchDataset:
#         return IncrementalHSIPatchDataset(self, _collect_indices(self._split_by_class, class_ids, split))

#     def _loader(self, dataset: Dataset, *, batch_size: int, shuffle: bool, phase: int, split: str) -> DataLoader:
#         size = _exact_integer(batch_size, name="batch_size")
#         if size <= 0:
#             raise ValueError("batch_size must be positive")
#         phase_id = _exact_integer(phase, name="phase")
#         generator = torch.Generator().manual_seed(
#             self.seed
#             + 1009 * phase_id
#             + {"train": 11, "val": 17, "test": 23, "all": 29}[split]
#         )
#         return DataLoader(
#             dataset, batch_size=size, shuffle=bool(shuffle), num_workers=self.num_workers,
#             pin_memory=self.pin_memory, drop_last=False,
#             persistent_workers=self.num_workers > 0,
#             worker_init_fn=_seed_worker if self.num_workers > 0 else None,
#             generator=generator,
#         )

#     def get_phase_dataloader(
#         self, phase: int, split: str = "train", batch_size: int = 64, shuffle: bool = True,
#     ) -> DataLoader:
#         phase = self._validate_phase(phase)
#         if self.current_phase != phase:
#             raise RuntimeError("phase train/validation loaders require an active phase")
#         token = _normalize_split_name(split)
#         if token not in {"train", "val"}:
#             raise ValueError("active phase loaders are restricted to train or val")
#         dataset = self._get_dataset(self.get_new_classes(phase), token)
#         return self._loader(dataset, batch_size=batch_size, shuffle=shuffle if token == "train" else False, phase=phase, split=token)

#     def get_cumulative_dataloader(
#         self, phase: int, split: str = "test", batch_size: int = 256, shuffle: bool = False,
#     ) -> DataLoader:
#         phase = self._validate_phase(phase)
#         if phase not in self.finalized_phases:
#             raise RuntimeError("cumulative evaluation requires a finalized phase")
#         token = _normalize_split_name(split)
#         if token not in {"val", "test"}:
#             raise ValueError("cumulative loaders are evaluation-only")
#         dataset = self._get_dataset(self.get_seen_classes(phase), token)
#         return self._loader(dataset, batch_size=batch_size, shuffle=shuffle, phase=phase, split=token)

#     def get_reporting_dataloader(
#         self, phase: int, *, split: str = "test", batch_size: int = 256,
#     ) -> DataLoader:
#         phase = self._validate_phase(phase)
#         if phase not in self.finalized_phases:
#             raise RuntimeError("reporting requires a finalized phase")
#         token = _normalize_split_name(split)
#         if token not in {"test", "all"}:
#             raise ValueError("reporting split must be test or all")
#         dataset = self._get_dataset(self.get_seen_classes(phase), token)
#         return self._loader(dataset, batch_size=batch_size, shuffle=False, phase=phase, split=token)

#     def protocol_report(self) -> Dict[str, Any]:
#         return {
#             "tasks": self.nb_tasks,
#             "phase_to_classes": {phase: list(classes) for phase, classes in self.phase_to_classes.items()},
#             "class_order_original_ids": list(self.class_order_original_ids),
#             "split_strategy": self.split_strategy,
#             "spatial_partition_mode": self.spatial_partition_mode,
#             "require_patch_disjoint": self.require_patch_disjoint,
#             "context_policy": self.context_policy,
#             "current_phase": self.current_phase,
#             "finalized_phases": list(self.finalized_phases),
#             "single_head_global_labels": True,
#             "current_phase_training_classes_only": True,
#             "cumulative_seen_class_evaluation": True,
#             "stores_old_training_samples": False,
#         }

#     def assert_exemplar_free_contract(self) -> bool:
#         if self.finalized_phases != list(range(len(self.finalized_phases))):
#             raise RuntimeError("finalized phase state is not contiguous")
#         if self.current_phase is not None and self.current_phase in self.finalized_phases:
#             raise RuntimeError("an active phase cannot also be finalized")
#         if any(
#             token in self.__dict__
#             for token in ("exemplars", "replay_buffer", "memory_samples")
#         ):
#             raise RuntimeError("dataset manager contains forbidden replay state")
#         report = self.protocol_report()
#         return bool(
#             report["single_head_global_labels"]
#             and report["current_phase_training_classes_only"]
#             and report["cumulative_seen_class_evaluation"]
#             and not report["stores_old_training_samples"]
#         )


# __all__ = [
#     "build_incremental_protocol", "build_base_protocol", "IncrementalHSIPatchDataset",
#     "IncrementalHSIDatasetManager",
# ]

