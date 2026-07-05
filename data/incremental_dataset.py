

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


class HSIPatchDataset(Dataset):
    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        coords: Optional[np.ndarray] = None,
        return_metadata: bool = False,
        center_spectra: Optional[np.ndarray] = None,
        spectra_are_physical: bool = False,
        return_input_center_when_no_spectra: bool = False,
    ):
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if len(patches) != len(labels):
            raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels).long()
        self.return_metadata = bool(return_metadata)
        self.spectra_are_physical = bool(spectra_are_physical)
        self.return_input_center_when_no_spectra = bool(return_input_center_when_no_spectra)

        if center_spectra is None:
            self.center_spectra = None
        else:
            center_spectra = np.asarray(center_spectra, dtype=np.float32)
            if len(center_spectra) != len(labels):
                raise ValueError(f"center_spectra/label length mismatch: {len(center_spectra)} vs {len(labels)}")
            if center_spectra.ndim != 2:
                center_spectra = center_spectra.reshape(len(labels), -1)
            self.center_spectra = torch.from_numpy(np.ascontiguousarray(center_spectra)).float()

        if coords is None:
            self.coords = torch.empty((len(labels), 0), dtype=torch.long)
        else:
            coords = np.asarray(coords, dtype=np.int64)
            if len(coords) != len(labels):
                raise ValueError(f"coord/label length mismatch: {len(coords)} vs {len(labels)}")
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"coords must be [N,2], got {coords.shape}")
            self.coords = torch.from_numpy(coords).long()

    def __len__(self) -> int:
        return int(len(self.labels))

    @staticmethod
    def _center_spectrum(patch: torch.Tensor) -> torch.Tensor:
        # Patches are normally [C,H,W]. Fallbacks keep compatibility with flat inputs.
        if patch.dim() == 3:
            return patch[:, patch.size(1) // 2, patch.size(2) // 2].float()
        if patch.dim() == 2:
            return patch.float().mean(dim=-1)
        return patch.float().flatten()

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        label = self.labels[idx]
        if not self.return_metadata:
            return patch, label
        # Return physical wavelength-ordered center spectra only when the
        # IncrementalHSIDataset explicitly provides them.  Do NOT silently fall
        # back to PCA/reduced patch-center spectra here: several trainer paths
        # interpret the third tuple field as raw spectral metadata.  Returning an
        # empty tensor is safer; the model/helper can still derive a non-physical
        # reduced center vector from the patch when needed.
        if self.center_spectra is not None:
            spectrum = self.center_spectra[idx]
        elif self.return_input_center_when_no_spectra:
            spectrum = self._center_spectrum(patch)
        else:
            spectrum = torch.empty((0,), dtype=patch.dtype)
        coord = self.coords[idx] if self.coords.numel() > 0 else torch.empty((0,), dtype=torch.long)
        return patch, label, spectrum, coord



class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Class-balanced batch sampler for geometry-native HSI base training.

    Local low-rank geometry is not compatible with random imbalanced batches:
    singleton rare classes produce degenerate covariance and rank-0 geometry.
    This sampler builds each train batch from multiple classes with multiple
    samples per class, sampling with replacement when a class is small.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        classes_per_batch: Optional[int] = None,
        samples_per_class: Optional[int] = None,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        labels_np = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if labels_np.size == 0:
            raise ValueError("ClassBalancedBatchSampler received zero labels.")

        self.labels_np = labels_np
        self.batch_size = int(max(1, batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.class_to_indices: Dict[int, np.ndarray] = {}
        for c in sorted(np.unique(labels_np).tolist()):
            idx = np.where(labels_np == int(c))[0].astype(np.int64)
            if idx.size > 0:
                self.class_to_indices[int(c)] = idx

        self.classes = sorted(self.class_to_indices.keys())
        if not self.classes:
            raise ValueError("ClassBalancedBatchSampler found no classes.")

        if classes_per_batch is None or int(classes_per_batch) <= 0:
            classes_per_batch = min(len(self.classes), self.batch_size)
        self.classes_per_batch = int(max(1, min(classes_per_batch, len(self.classes), self.batch_size)))

        if samples_per_class is None or int(samples_per_class) <= 0:
            samples_per_class = max(2, self.batch_size // self.classes_per_batch)
        self.samples_per_class = int(max(1, samples_per_class))

        self.effective_batch_size = self.classes_per_batch * self.samples_per_class
        if self.effective_batch_size > self.batch_size:
            self.samples_per_class = max(1, self.batch_size // self.classes_per_batch)
            self.effective_batch_size = self.classes_per_batch * self.samples_per_class

        self.num_batches = int(np.ceil(labels_np.size / float(max(self.effective_batch_size, 1))))
        if self.drop_last:
            self.num_batches = int(labels_np.size // float(max(self.effective_batch_size, 1)))
        self.num_batches = max(1, self.num_batches)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        classes = np.asarray(self.classes, dtype=np.int64)

        for _ in range(self.num_batches):
            replace_classes = len(classes) < self.classes_per_batch
            chosen_classes = rng.choice(
                classes,
                size=self.classes_per_batch,
                replace=replace_classes,
            )

            batch: List[int] = []
            for c in chosen_classes.tolist():
                pool = self.class_to_indices[int(c)]
                replace_samples = pool.size < self.samples_per_class
                sampled = rng.choice(
                    pool,
                    size=self.samples_per_class,
                    replace=replace_samples,
                )
                batch.extend(int(i) for i in sampled.tolist())

            rng.shuffle(batch)
            if not batch:
                continue
            if self.drop_last and len(batch) < self.effective_batch_size:
                continue
            yield batch


class IncrementalHSIDataset:
    """
    Strict non-exemplar incremental dataset manager.

    Important:
    The dataset manager never assumes label 0 is background. If label 0 is in
    labels, it is treated as a valid class.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        gt_shape: Tuple[int, int],
        GT: np.ndarray,
        base_classes: int,
        increment: int,
        train_ratio: float = 0.2,
        val_ratio: float = 0.1,
        seed: int = 42,
        shuffle_order: bool = False,
        device: str = "cuda",
        min_train_per_class: int = 20,
        num_workers: int = 0,
        strict_non_exemplar: bool = True,
        target_names: Optional[List[str]] = None,
        label_policy: Optional[Dict] = None,
        class_balanced_train_batches: bool = True,
        geometry_batch_classes: int = 0,
        geometry_samples_per_class: int = 0,
        return_metadata: bool = False,
        raw_spectra: Optional[np.ndarray] = None,
        center_spectra: Optional[np.ndarray] = None,
        spectra_are_physical: bool = False,
    ):
        # Raw arrays from ImageCubes.
        self.patches = np.asarray(patches, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.gt_shape = gt_shape
        self.GT = GT
        self.spectra_are_physical = bool(spectra_are_physical)
        # raw_spectra/center_spectra are per-sample physical center spectra aligned
        # with patches/labels. They are descriptors metadata, not old exemplars;
        # strict non-exemplar access rules below still prevent old raw train use.
        raw_center = center_spectra if center_spectra is not None else raw_spectra
        self.raw_spectra = None
        if raw_center is not None:
            raw_center = np.asarray(raw_center, dtype=np.float32)
            if raw_center.ndim != 2:
                raw_center = raw_center.reshape(len(self.labels), -1)
            if len(raw_center) != len(self.labels):
                raise ValueError(f"raw_spectra/label length mismatch: {len(raw_center)} vs {len(self.labels)}")
            self.raw_spectra = np.ascontiguousarray(raw_center, dtype=np.float32)
            if not np.isfinite(self.raw_spectra).all():
                self.raw_spectra = np.nan_to_num(self.raw_spectra, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            # Do NOT auto-mark metadata as physical.  In reduced-band PCA runs,
            # center_spectra may be PCA/component vectors, not wavelength-ordered
            # spectra.  The caller owns this flag.
            self.spectra_are_physical = bool(spectra_are_physical)

        if len(self.patches) != len(self.labels):
            raise ValueError(f"patch/label length mismatch: {len(self.patches)} vs {len(self.labels)}")
        if len(self.coords) != len(self.labels):
            raise ValueError(f"coord/label length mismatch: {len(self.coords)} vs {len(self.labels)}")
        if self.labels.size == 0:
            raise ValueError("Empty labels passed to IncrementalHSIDataset.")
        if self.labels.min() < 0:
            raise ValueError(
                f"Negative labels passed to IncrementalHSIDataset: min={self.labels.min()}. "
                f"Loader must remove background/ignore labels before incremental split."
            )

        # Settings.
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
        self.geometry_batch_classes = int(geometry_batch_classes)
        self.geometry_samples_per_class = int(geometry_samples_per_class)
        self.return_metadata = bool(return_metadata)
        self.target_names = target_names
        self.label_policy = self._normalize_label_policy(label_policy)
        self.has_background = bool(self.label_policy.get("has_background", True))
        self.background_label = self.label_policy.get("background_label", 0 if self.has_background else None)
        self.raw_class_values = [int(v) for v in self.label_policy.get("raw_class_values", [])]

        self.pin_memory = self.device.startswith("cuda")

        # Cache for semantic tokens / concept tokens.
        self._semantic_token_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        # Protocol state.
        self.current_phase: int = 0
        self.finalized_phases: Set[int] = set()
        self.finalized_classes: Set[int] = set()
        self._memory_build_active: bool = False
        self._memory_build_classes: Set[int] = set()

        # Class order and remapping.
        # labels may already be 0..K-1, but we still remap to sequential IDs
        # according to class_order. If label 0 exists, it is included.
        self.all_classes = sorted(int(x) for x in np.unique(self.labels).tolist())
        if self.all_classes[0] != 0:
            print(
                f"[IncrementalHSIDataset:WARN] smallest label is {self.all_classes[0]}, "
                f"not 0. The manager will remap to sequential IDs."
            )

        self.num_classes = len(self.all_classes)

        if self.base_classes <= 0 or self.base_classes > self.num_classes:
            raise ValueError(f"base_classes={self.base_classes} invalid for num_classes={self.num_classes}")
        if self.increment <= 0:
            raise ValueError(f"increment must be > 0, got {self.increment}")

        if shuffle_order:
            rng = np.random.RandomState(self.seed)
            self.class_order = rng.permutation(self.all_classes).tolist()
        else:
            self.class_order = list(self.all_classes)

        self.label_map = {global_id: seq_id for seq_id, global_id in enumerate(self.class_order)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.remapped_labels = np.array(
            [self.label_map[int(l)] for l in self.labels],
            dtype=np.int64,
        )

        # target_names provided by the loader are indexed by input label id
        # after ImageCubes mapping. Convert them once to sequential phase ids.
        self.target_names_by_seq = self._build_target_names_by_seq(target_names)

        self._validate_class_zero_policy()
        self._validate_full_class_coverage()

        # Phase partition in sequential label space.
        remaining_classes = self.num_classes - self.base_classes
        self.num_phases = 1 + int(np.ceil(remaining_classes / self.increment))

        self.phase_to_classes: Dict[int, List[int]] = {}
        self.class_to_phase: Dict[int, int] = {}

        for phase in range(self.num_phases):
            if phase == 0:
                cls_list = list(range(self.base_classes))
            else:
                start = self.base_classes + (phase - 1) * self.increment
                end = min(start + self.increment, self.num_classes)
                cls_list = list(range(start, end))

            self.phase_to_classes[phase] = cls_list
            for cls in cls_list:
                self.class_to_phase[int(cls)] = int(phase)

        self._create_splits()
        self._validate_protocol_integrity()
        self._print_stats()

    # ============================================================
    # Label policy helpers
    # ============================================================
    def _normalize_label_policy(self, label_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize dataset label metadata.

        The incremental manager receives sequential training labels. Raw GT
        background handling must be decided in the loader/ImageCubes stage. This
        policy is carried here only for validation, reporting, and visualization.

        Rules:
            - if has_background=True, raw background is display-only and should
              not appear in self.labels;
            - if has_background=False, raw class 0 is allowed to be a real class;
            - model/trainer class ids remain sequential 0..K-1 either way.
        """
        policy = dict(label_policy or {})

        if "has_background" not in policy:
            # Conservative default: most classic HSI datasets use raw 0 as background.
            # Datasets with raw class 0 real must pass has_background=False from the loader.
            policy["has_background"] = True

        policy["has_background"] = bool(policy["has_background"])

        if policy["has_background"]:
            try:
                policy["background_label"] = int(policy.get("background_label", 0))
            except Exception:
                policy["background_label"] = 0
        else:
            policy["background_label"] = None

        raw_values = policy.get("raw_class_values", None)
        if raw_values is None:
            raw_values = []
        policy["raw_class_values"] = [int(v) for v in raw_values]

        return policy

    # ============================================================
    # Validation
    # ============================================================
    def _validate_class_zero_policy(self) -> None:
        """
        Validate class-0/background semantics without deleting label 0.

        Important:
            self.labels are already training labels from ImageCubes. They are
            not necessarily raw GT labels. Therefore this method never assumes
            input label 0 is background. It only checks consistency between the
            loader-provided label_policy and the sequential label space.
        """
        input_has_zero = 0 in set(int(x) for x in self.labels.tolist())
        remap_has_zero = 0 in set(int(x) for x in self.remapped_labels.tolist())

        if input_has_zero and not remap_has_zero:
            raise RuntimeError(
                "Input label 0 existed but disappeared after incremental remapping. "
                "This is invalid for datasets where class 0 is real and also invalid "
                "for already-remapped foreground labels."
            )

        if not self.has_background:
            # Raw class 0 may be a real class. This is valid. Do not warn just
            # because sequential class 0 exists; it must exist in any sequential
            # class space.
            if self.raw_class_values and 0 not in self.raw_class_values:
                print(
                    "[IncrementalHSIDataset:WARN] label_policy says has_background=False, "
                    "but raw_class_values does not include raw class 0. This may still be "
                    "valid, but check the loader metadata."
                )
        else:
            # Background must have been removed before this manager. If raw
            # background label was 0, sequential class 0 is still allowed because
            # it now means the first foreground class, not background.
            if self.background_label is None:
                print(
                    "[IncrementalHSIDataset:WARN] has_background=True but background_label is None. "
                    "Visualization/reporting will treat display value 0 as background/unlabeled."
                )


    def _build_target_names_by_seq(self, target_names: Optional[List[str]]) -> Optional[List[str]]:
        if target_names is None:
            return None

        out: List[str] = []
        for sid in range(len(self.all_classes)):
            gid = self.inv_label_map.get(sid, sid)
            if int(gid) < len(target_names):
                out.append(str(target_names[int(gid)]))
            else:
                out.append(f"Class {gid}")
        return out

    def _validate_full_class_coverage(self) -> None:
        present = set(int(x) for x in np.unique(self.remapped_labels).tolist())
        expected = set(range(self.num_classes))
        missing = sorted(expected - present)
        extra = sorted(present - expected)

        if missing or extra:
            raise RuntimeError(
                f"Remapped label space is broken. Missing={missing}, extra={extra}, "
                f"num_classes={self.num_classes}, class_order={self.class_order}"
            )

    def _validate_protocol_integrity(self) -> None:
        """Fail fast on class-order, phase, and split corruption."""
        phase_union: List[int] = []
        for p in range(self.num_phases):
            cls = self.phase_to_classes.get(p, [])
            if not cls:
                raise RuntimeError(f"Phase {p} has no classes.")
            phase_union.extend(int(c) for c in cls)
        if phase_union != list(range(self.num_classes)):
            raise RuntimeError(
                f"Phase partition must cover contiguous sequential ids exactly once. "
                f"got={phase_union}, expected={list(range(self.num_classes))}"
            )
        for c in range(self.num_classes):
            if c not in self.class_to_phase:
                raise RuntimeError(f"class_to_phase missing class {c}")
        all_split = np.concatenate([self.train_indices, self.val_indices, self.test_indices]).astype(np.int64)
        if all_split.size != len(np.unique(all_split)):
            raise RuntimeError("Train/val/test splits overlap. This leaks samples across splits.")
        valid = set(range(len(self.remapped_labels)))
        bad = sorted(set(int(i) for i in all_split.tolist()).difference(valid))
        if bad:
            raise RuntimeError(f"Split indices outside dataset range: {bad[:10]}")
        missing = sorted(valid.difference(set(int(i) for i in all_split.tolist())))
        # A class with n=1 may be train-only, but every sample must still appear in exactly one split.
        if missing:
            raise RuntimeError(f"Some samples are absent from all splits: first_missing={missing[:10]}")

    def _validate_phase(self, phase: int) -> int:
        phase = int(phase)
        if phase < 0 or phase >= self.num_phases:
            raise ValueError(f"Invalid phase {phase}. Valid range: 0..{self.num_phases - 1}")
        return phase

    # ============================================================
    # Basic mapping helpers
    # ============================================================
    def seq_to_global(self, seq_id: int) -> int:
        return int(self.inv_label_map[int(seq_id)])

    def global_to_seq(self, global_id: int) -> int:
        return int(self.label_map[int(global_id)])

    def get_phase_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        return list(self.phase_to_classes[phase])

    def get_classes_up_to_phase(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        classes: List[int] = []
        for p in range(phase + 1):
            classes.extend(self.phase_to_classes[p])
        return _ordered_unique_ints(classes)

    def get_seen_classes(self, phase: int) -> List[int]:
        """Alias used by trainers/evaluators: cumulative sequential ids."""
        return self.get_classes_up_to_phase(phase)

    def get_old_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        if phase == 0:
            return []
        return self.get_classes_up_to_phase(phase - 1)

    def get_new_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        return list(self.phase_to_classes[phase])

    def get_future_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        seen = set(self.get_seen_classes(phase))
        return [int(c) for c in range(self.num_classes) if int(c) not in seen]

    def classifier_num_classes_for_phase(self, phase: int) -> int:
        """Required classifier width for global sequential CE/logits.

        The project uses global sequential labels. Therefore phase ``p`` logits
        must contain columns ``0..max(seen_classes)``. Since phases are allocated
        in contiguous sequential ids, this equals ``len(seen_classes)``. Keeping
        this method explicit catches accidental full-dataset classifier outputs
        during early phases.
        """
        seen = self.get_seen_classes(phase)
        if not seen:
            raise RuntimeError(f"No seen classes for phase {phase}.")
        expected = list(range(max(seen) + 1))
        if seen != expected:
            raise RuntimeError(
                f"Seen classes are not contiguous global ids: seen={seen}, expected={expected}. "
                "The trainer/classifier uses global sequential labels and requires contiguous phase ids."
            )
        return int(max(seen) + 1)

    def get_phase_class_splits(self, phase: int) -> Dict[str, List[int]]:
        phase = self._validate_phase(phase)
        return {
            "old": self.get_old_classes(phase),
            "new": self.get_new_classes(phase),
            "seen": self.get_seen_classes(phase),
            "future": self.get_future_classes(phase),
        }

    def assert_classifier_contract(self, phase: int, logits_or_width: Any, *, context: str = "") -> None:
        """Assert classifier output matches currently seen class space."""
        phase = self._validate_phase(phase)
        expected = self.classifier_num_classes_for_phase(phase)
        if torch.is_tensor(logits_or_width):
            if logits_or_width.dim() != 2:
                raise RuntimeError(f"{context}: logits must be [B,C], got {tuple(logits_or_width.shape)}")
            got = int(logits_or_width.size(1))
        else:
            got = int(logits_or_width)
        if got != expected:
            raise RuntimeError(
                f"{context}: classifier width mismatch for phase {phase}: got {got}, expected {expected}. "
                f"Seen classes={self.get_seen_classes(phase)}. Do not expose future-class logits during this phase."
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
        """Validate label tensors before CE/loss/evaluation.

        ``current_train_only=True`` is for raw train batches in strict NECIL.
        ``cumulative=True`` is for validation/test over all seen classes.
        """
        phase = self._validate_phase(phase)
        y = _as_1d_int_array(labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels, name="labels")
        if y.min() < 0:
            raise RuntimeError(f"{context}: negative labels entered phase {phase}: min={int(y.min())}")
        allowed = self.get_new_classes(phase) if current_train_only else (self.get_seen_classes(phase) if cumulative else self.get_new_classes(phase))
        allowed_set = set(int(c) for c in allowed)
        bad = sorted(set(int(v) for v in y.tolist()).difference(allowed_set))
        if bad:
            raise RuntimeError(
                f"{context}: labels outside allowed phase classes. bad={bad}, allowed={allowed}, "
                f"phase={phase}, current_train_only={current_train_only}, cumulative={cumulative}"
            )

    def get_geometry_reservation_plan(self, phase: int = 0, *, feature_dim: Optional[int] = None, rank: Optional[int] = None) -> Dict[str, Any]:
        """Return an abstract future-capacity plan without future data access.

        This is safe for base training because it uses only the public protocol
        schedule: number/order of classes and phase sizes. It does not expose
        future patches, labels-per-sample, spectra, features, or class statistics.
        """
        phase = self._validate_phase(phase)
        seen = self.get_seen_classes(phase)
        future = self.get_future_classes(phase)
        return {
            "phase": int(phase),
            "seen_classes": seen,
            "future_class_slots": future,
            "total_class_capacity": int(self.num_classes),
            "seen_classifier_width": int(self.classifier_num_classes_for_phase(phase)),
            "feature_dim": None if feature_dim is None else int(feature_dim),
            "rank": None if rank is None else int(rank),
            "allowed_reservation": [
                "empty_geometry_rows",
                "center_margin_budget",
                "subspace_orthogonality_budget",
                "residual_variance_floor",
                "transport_trust_region_budget",
            ],
            "forbidden_reservation": [
                "future_raw_patches",
                "future_raw_spectra",
                "future_feature_vectors",
                "future_class_statistics",
                "future_labels_in_base_loss",
            ],
        }

    # ============================================================
    # Protocol controls
    # ============================================================
    def start_phase(self, phase: int) -> None:
        phase = self._validate_phase(phase)
        self.current_phase = phase
        self._invalidate_train_caches_for_locked_classes()

    def finalize_phase(self, phase: int) -> None:
        phase = self._validate_phase(phase)

        self.finalized_phases.add(phase)
        self.finalized_classes.update(self.phase_to_classes[phase])
        self._memory_build_active = False
        self._memory_build_classes.clear()
        self._invalidate_train_caches_for_locked_classes()

    def is_phase_finalized(self, phase: int) -> bool:
        return int(phase) in self.finalized_phases

    def is_class_finalized(self, cls: int) -> bool:
        return int(cls) in self.finalized_classes

    def get_accessible_train_classes(self) -> List[int]:
        if not self.strict_non_exemplar:
            return list(range(self.num_classes))
        return list(self.phase_to_classes[self.current_phase])

    def _invalidate_train_caches_for_locked_classes(self) -> None:
        keys_to_delete = []
        accessible = set(self.get_accessible_train_classes())

        for key in list(self._semantic_token_cache.keys()):
            cls, descriptor = key
            if "_train" not in descriptor:
                continue
            if self.strict_non_exemplar and int(cls) not in accessible:
                keys_to_delete.append(key)

        for k in keys_to_delete:
            del self._semantic_token_cache[k]

    def _is_train_access_allowed(self, cls: int) -> bool:
        """Return whether raw TRAIN samples of ``cls`` may be read now.

        This is the strict exemplar-free access contract. Once a phase is
        finalized, its raw training samples are old exemplars and must never be
        exposed again. The only exception is the short memory_build_context for
        the current phase before finalization, where descriptors are being built
        from current-class data.
        """
        cls = int(cls)
        if cls < 0 or cls >= self.num_classes:
            raise ValueError(f"class id {cls} outside sequential label space [0,{self.num_classes - 1}]")

        if not self.strict_non_exemplar:
            return True

        if cls in self.finalized_classes:
            return False

        if self._memory_build_active and cls in self._memory_build_classes:
            return True

        if cls in self.phase_to_classes[self.current_phase]:
            return True

        return False

    def _check_class_split_access(self, cls: int, split: str) -> None:
        cls = int(cls)
        split = str(split).lower()

        if cls < 0 or cls >= self.num_classes:
            raise ValueError(f"class id {cls} outside sequential label space [0,{self.num_classes - 1}]")

        if split != "train":
            return

        if not self._is_train_access_allowed(cls):
            phase_of_cls = self.class_to_phase.get(cls, None)
            raise PermissionError(
                f"Strict non-exemplar protocol violation: raw TRAIN access denied for class {cls} "
                f"(phase={phase_of_cls}, current_phase={self.current_phase}, finalized={cls in self.finalized_classes}, "
                f"memory_build_active={self._memory_build_active}). Use GeometryBank descriptors / synthetic geometry replay, "
                f"not old raw patches or old raw spectra."
            )

    @contextmanager
    def memory_build_context(self, phase: int):
        phase = int(phase)
        if phase != self.current_phase:
            raise ValueError(
                f"memory_build_context phase={phase} must match current_phase={self.current_phase}"
            )

        allowed_classes = set(self.phase_to_classes[phase])

        prev_active = self._memory_build_active
        prev_classes = set(self._memory_build_classes)

        self._memory_build_active = True
        self._memory_build_classes = allowed_classes

        try:
            yield
        finally:
            self._memory_build_active = prev_active
            self._memory_build_classes = prev_classes

    # ============================================================
    # Split creation
    # ============================================================
    def _create_splits(self) -> None:
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []

        for seq_id in range(self.num_classes):
            class_indices = np.where(self.remapped_labels == seq_id)[0]
            n_samples = len(class_indices)

            if n_samples == 0:
                continue

            rng = np.random.RandomState(self.seed + seq_id)
            shuffled = rng.permutation(class_indices)

            if n_samples == 1:
                self.train_indices.extend(shuffled.tolist())
                continue

            if n_samples == 2:
                self.train_indices.append(int(shuffled[0]))
                self.test_indices.append(int(shuffled[1]))
                continue

            n_train = max(self.min_train_per_class, int(round(n_samples * self.train_ratio)))
            n_val = max(1, int(round(n_samples * self.val_ratio)))

            if n_train + n_val >= n_samples:
                overflow = (n_train + n_val) - (n_samples - 1)

                reduce_val = min(overflow, max(0, n_val - 1))
                n_val -= reduce_val
                overflow -= reduce_val

                if overflow > 0:
                    n_train = max(1, n_train - overflow)

            n_train = max(1, min(n_train, n_samples - 2))
            n_val = max(1, min(n_val, n_samples - n_train - 1))
            n_test = n_samples - n_train - n_val

            assert n_train >= 1
            assert n_val >= 1
            assert n_test >= 1
            assert n_train + n_val + n_test == n_samples

            self.train_indices.extend(shuffled[:n_train].tolist())
            self.val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
            self.test_indices.extend(shuffled[n_train + n_val:].tolist())

        self.train_indices = np.array(self.train_indices, dtype=np.int64)
        self.val_indices = np.array(self.val_indices, dtype=np.int64)
        self.test_indices = np.array(self.test_indices, dtype=np.int64)

    # ============================================================
    # Diagnostics
    # ============================================================
    def _print_stats(self) -> None:
        print("[IncrementalHSIDataset] Initialized")
        print(f"  Total classes: {self.num_classes} | Phases: {self.num_phases}")
        print(f"  Input classes: {self.all_classes}")
        print(f"  Class order: {self.class_order}")
        print(f"  Strict non-exemplar: {self.strict_non_exemplar}")
        print(
            f"  Spectral metadata: available={self.has_spectral_metadata()}, "
            f"physical={self.has_physical_spectra()}, dim={self.get_spectra_dim()}"
        )
        print(
            f"  Label policy: has_background={self.has_background}, "
            f"background_label={self.background_label}, raw_class_values={self.raw_class_values}"
        )
        print(
            "  Sequential class 0 present: "
            f"{0 in set(int(x) for x in self.remapped_labels.tolist())} "
            "(sequential class 0 is a real foreground training class, not background)"
        )

        print(f"  Split sizes: train={len(self.train_indices)} | val={len(self.val_indices)} | test={len(self.test_indices)}")

        for p in range(self.num_phases):
            global_ids = [self.seq_to_global(sid) for sid in self.phase_to_classes[p]]
            names = []
            if self.target_names_by_seq is not None:
                for sid in self.phase_to_classes[p]:
                    if int(sid) < len(self.target_names_by_seq):
                        names.append(self.target_names_by_seq[int(sid)])
                    else:
                        names.append(f"Class {self.seq_to_global(sid)}")
            print(f"  Phase {p}: Sequential {self.phase_to_classes[p]} (Input labels {global_ids})")
            if names:
                print(f"           Names: {names}")

    def get_label_policy_summary(self) -> Dict[str, Any]:
        return {
            "has_background": bool(self.has_background),
            "background_label": self.background_label,
            "raw_class_values": list(self.raw_class_values),
            "class_order": [int(c) for c in self.class_order],
            "label_map": {int(k): int(v) for k, v in self.label_map.items()},
            "inv_label_map": {int(k): int(v) for k, v in self.inv_label_map.items()},
            "note": (
                "Training labels are sequential 0..K-1. Display value 0 in maps "
                "is reserved for background/unseen/suppressed; class c is displayed as c+1."
            ),
        }

    def has_spectral_metadata(self) -> bool:
        return self.raw_spectra is not None and int(self.raw_spectra.size) > 0

    def has_physical_spectra(self) -> bool:
        return bool(self.has_spectral_metadata() and self.spectra_are_physical)

    def get_spectra_dim(self) -> int:
        if self.raw_spectra is None or self.raw_spectra.ndim != 2:
            return 0
        return int(self.raw_spectra.shape[1])

    def get_spectral_metadata_summary(self) -> Dict[str, Any]:
        return {
            "available": bool(self.has_spectral_metadata()),
            "physical": bool(self.has_physical_spectra()),
            "dim": int(self.get_spectra_dim()),
            "note": (
                "Physical spectra are wavelength-ordered raw center spectra for SCB-GR spectral-risk mining. "
                "If physical=False, the metadata must not be used for derivative spectral-shape scoring."
            ),
        }

    def _spectra_for_indices(self, idx: np.ndarray, *, require_physical: bool = True) -> torch.Tensor:
        if self.raw_spectra is None:
            raise AttributeError("No center-spectral metadata are available in this dataset.")
        if bool(require_physical) and not bool(self.spectra_are_physical):
            raise RuntimeError(
                "Requested physical center spectra, but IncrementalHSIDataset.spectra_are_physical=False. "
                "This usually means the metadata are PCA/reduced components. Do not use them for spectral derivatives."
            )
        if len(idx) == 0:
            raise ValueError("No indices provided for spectral metadata lookup.")
        return torch.from_numpy(np.ascontiguousarray(self.raw_spectra[idx])).float()

    def get_class_spectra(self, cls: int, split: str = "train", require_physical: bool = True) -> torch.Tensor:
        """Return per-sample center spectra for one class/split.

        By default this returns only physical wavelength-ordered spectra.  This
        is deliberate: trainer/helper code may use this method for SCB-GR
        spectral-shape descriptors, where PCA/reduced components would be wrong.

        The same strict non-exemplar access rule as get_class_patches applies:
        old train spectra are denied after their phase is finalized. They are
        not old exemplars; they are used only while building the current class
        geometry state.
        """
        idx = self.get_class_indices(int(cls), split=split)
        if len(idx) == 0:
            raise ValueError(f"No spectra found for class {cls} in split '{split}'")
        return self._spectra_for_indices(idx, require_physical=bool(require_physical))

    def get_class_center_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
        return self.get_class_spectra(cls, split=split, require_physical=True)

    def get_class_raw_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
        return self.get_class_spectra(cls, split=split, require_physical=True)

    def get_class_spectral_summary(self, cls: int, split: str = "train") -> torch.Tensor:
        # Keep this conservative. Reduced/non-physical spectra are already
        # available from the patch center through the model/helper fallback.
        return self.get_class_spectra(cls, split=split, require_physical=True)

    def get_class_counts(self) -> Dict[int, int]:
        return {
            int(c): int((self.remapped_labels == int(c)).sum())
            for c in range(self.num_classes)
        }

    def get_split_class_counts(self, split: str = "train") -> Dict[int, int]:
        indices = self._get_split_indices(split)
        return {
            int(c): int((self.remapped_labels[indices] == int(c)).sum())
            for c in range(self.num_classes)
        }

    def protocol_state(self) -> Dict[str, object]:
        return {
            "current_phase": int(self.current_phase),
            "strict_non_exemplar": bool(self.strict_non_exemplar),
            "finalized_phases": sorted(int(p) for p in self.finalized_phases),
            "finalized_classes": sorted(int(c) for c in self.finalized_classes),
            "accessible_train_classes": self.get_accessible_train_classes(),
            "phase_class_splits": self.get_phase_class_splits(self.current_phase),
            "classifier_width": self.classifier_num_classes_for_phase(self.current_phase),
            "memory_build_active": bool(self._memory_build_active),
            "memory_build_classes": sorted(int(c) for c in self._memory_build_classes),
            "spectral_metadata": self.get_spectral_metadata_summary(),
        }

    def assert_no_old_train_access(self, cls: int) -> None:
        self._check_class_split_access(int(cls), "train")

    def _assert_indices_match_classes(self, idx: np.ndarray, class_ids: Sequence[int], *, context: str) -> None:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        class_ids = _ordered_unique_ints(class_ids)
        if idx.size == 0:
            return
        if idx.min() < 0 or idx.max() >= len(self.remapped_labels):
            raise RuntimeError(f"{context}: indices outside dataset range.")
        allowed = set(int(c) for c in class_ids)
        labels = self.remapped_labels[idx]
        bad = sorted(set(int(v) for v in labels.tolist()).difference(allowed))
        if bad:
            raise RuntimeError(f"{context}: indices contain labels outside active classes. bad={bad}, allowed={class_ids}")

    def _make_loader(
        self,
        idx: np.ndarray,
        batch_size: int,
        shuffle: bool,
        *,
        balanced: bool = False,
        active_classes: Optional[List[int]] = None,
    ) -> DataLoader:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            raise ValueError(
                "Requested DataLoader has zero samples. Check phase split, class order, "
                "or strict non-exemplar access policy."
            )

        dataset = HSIPatchDataset(
            self.patches[idx],
            self.remapped_labels[idx],
            coords=self.coords[idx],
            return_metadata=self.return_metadata,
            center_spectra=(self.raw_spectra[idx] if self.has_physical_spectra() else None),
            spectra_are_physical=bool(self.has_physical_spectra()),
            return_input_center_when_no_spectra=False,
        )

        if balanced:
            present = sorted(int(c) for c in dataset.labels.unique().tolist())
            if len(present) <= 1:
                return DataLoader(
                    dataset,
                    batch_size=int(batch_size),
                    shuffle=bool(shuffle),
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    drop_last=False,
                )

            classes_per_batch = int(self.geometry_batch_classes)
            if classes_per_batch <= 0:
                classes_per_batch = len(active_classes) if active_classes is not None else len(present)
            classes_per_batch = max(2, min(classes_per_batch, len(present), int(batch_size)))

            samples_per_class = int(self.geometry_samples_per_class)
            if samples_per_class <= 0:
                samples_per_class = max(2, int(batch_size) // classes_per_batch)

            sampler = ClassBalancedBatchSampler(
                labels=dataset.labels,
                batch_size=int(batch_size),
                classes_per_batch=classes_per_batch,
                samples_per_class=samples_per_class,
                seed=self.seed + self.current_phase * 1009,
                drop_last=False,
            )

            return DataLoader(
                dataset,
                batch_sampler=sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )

        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=False,
        )

    # ============================================================
    # Split access helpers
    # ============================================================
    def _get_split_indices(self, split: str) -> np.ndarray:
        split = split.lower()
        if split == "train":
            return self.train_indices
        if split == "val":
            return self.val_indices
        if split == "test":
            return self.test_indices
        if split == "all":
            return np.arange(len(self.remapped_labels), dtype=np.int64)
        raise ValueError(f"Unknown split '{split}'. Use one of: train, val, test, all")

    def get_class_indices(self, cls: int, split: str = "train") -> np.ndarray:
        cls = int(cls)
        split = split.lower()
        self._check_class_split_access(cls, split)

        indices = self._get_split_indices(split)
        mask = self.remapped_labels[indices] == cls
        return indices[mask]

    def get_class_patches(self, cls: int, split: str = "train") -> np.ndarray:
        idx = self.get_class_indices(cls, split=split)
        if len(idx) == 0:
            raise ValueError(f"No samples found for class {cls} in split '{split}'")
        return self.patches[idx]

    # ============================================================
    # DataLoaders
    # ============================================================
    def get_class_balanced_phase_loader(
        self,
        phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        """Return a class-balanced phase loader for geometry construction/refinement.

        This is a named wrapper so the trainer can request balanced current-phase
        batches without relying on implicit flags.  It preserves strict
        non-exemplar access: train split exposes only current-phase raw samples.
        """
        old_flag = self.class_balanced_train_batches
        try:
            self.class_balanced_train_batches = True
            return self.get_phase_dataloader(
                phase=phase,
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        finally:
            self.class_balanced_train_batches = old_flag

    def get_phase_dataloader(
        self,
        phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        phase = self._validate_phase(phase)
        split = split.lower()

        active_classes = self.phase_to_classes[phase]

        if split == "train":
            if self.strict_non_exemplar and phase != self.current_phase:
                raise PermissionError(
                    f"Raw TRAIN loader requested for phase {phase}, but current_phase is {self.current_phase}. "
                    "Strict non-exemplar mode only allows current-phase raw train access."
                )

        indices = self._get_split_indices(split)
        mask = np.isin(self.remapped_labels[indices], active_classes)
        idx = indices[mask]
        self._assert_indices_match_classes(idx, active_classes, context=f"get_phase_dataloader({split}, phase={phase})")

        use_balanced = (
            split == "train"
            and bool(self.class_balanced_train_batches)
            and bool(shuffle)
        )

        return self._make_loader(
            idx,
            batch_size=batch_size,
            shuffle=(shuffle if split == "train" else False),
            balanced=use_balanced,
            active_classes=active_classes,
        )

    def get_cumulative_dataloader(
        self,
        up_to_phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
        allow_train_old: bool = False,
    ) -> DataLoader:
        up_to_phase = self._validate_phase(up_to_phase)
        split = split.lower()

        if split == "train" and self.strict_non_exemplar:
            if bool(allow_train_old):
                raise PermissionError(
                    "allow_train_old=True would expose raw old training samples in strict non-exemplar mode. "
                    "Use GeometryBank synthetic replay instead."
                )
            # Critical protocol rule: training loaders expose only current-phase raw samples.
            active_classes = list(self.phase_to_classes[self.current_phase])
        else:
            active_classes = self.get_seen_classes(up_to_phase)

        indices = self._get_split_indices(split)
        mask = np.isin(self.remapped_labels[indices], active_classes)
        idx = indices[mask]

        self._assert_indices_match_classes(idx, active_classes, context=f"get_cumulative_dataloader({split}, phase={up_to_phase})")

        use_balanced = (
            split == "train"
            and bool(self.class_balanced_train_batches)
            and bool(shuffle)
        )

        return self._make_loader(
            idx,
            batch_size=batch_size,
            shuffle=(shuffle if split == "train" else False),
            balanced=use_balanced,
            active_classes=active_classes,
        )

    def get_cumulative_test_data(self, phase: int):
        phase = self._validate_phase(phase)
        active_classes = self.get_seen_classes(phase)
        mask = np.isin(self.remapped_labels[self.test_indices], active_classes)
        idx = self.test_indices[mask]
        self._assert_indices_match_classes(idx, active_classes, context=f"get_cumulative_test_data(phase={phase})")
        return self.patches[idx], self.remapped_labels[idx], self.coords[idx]

    def get_cumulative_val_data(self, phase: int):
        phase = self._validate_phase(phase)
        active_classes = self.get_seen_classes(phase)
        mask = np.isin(self.remapped_labels[self.val_indices], active_classes)
        idx = self.val_indices[mask]
        self._assert_indices_match_classes(idx, active_classes, context=f"get_cumulative_val_data(phase={phase})")
        return self.patches[idx], self.remapped_labels[idx], self.coords[idx]

    def get_current_train_data(self, phase: Optional[int] = None):
        phase = self.current_phase if phase is None else self._validate_phase(phase)
        if self.strict_non_exemplar and int(phase) != int(self.current_phase):
            raise PermissionError(
                f"Raw train data requested for phase {phase}, but current_phase={self.current_phase}. "
                "Strict NECIL exposes only current-phase train samples."
            )
        active_classes = self.get_new_classes(phase)
        mask = np.isin(self.remapped_labels[self.train_indices], active_classes)
        idx = self.train_indices[mask]
        self._assert_indices_match_classes(idx, active_classes, context=f"get_current_train_data(phase={phase})")
        return self.patches[idx], self.remapped_labels[idx], self.coords[idx]

    # ============================================================
    # K-means concept extraction
    # ============================================================
    def _kmeans_numpy(
        self,
        x: np.ndarray,
        k: int,
        seed: int = 42,
        max_iters: int = 30,
    ) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got shape={x.shape}")

        n, d = x.shape
        if n == 0:
            raise ValueError("Empty input to k-means")

        k = max(1, min(int(k), n))
        rng = np.random.RandomState(seed)

        centers = np.empty((k, d), dtype=np.float32)
        first = rng.randint(0, n)
        centers[0] = x[first]
        dist2 = ((x - centers[0]) ** 2).sum(axis=1)

        for i in range(1, k):
            probs = dist2 / max(dist2.sum(), 1e-12)
            idx = rng.choice(n, p=probs)
            centers[i] = x[idx]
            dist2 = np.minimum(dist2, ((x - centers[i]) ** 2).sum(axis=1))

        for _ in range(max_iters):
            assign = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            new_centers = centers.copy()
            for i in range(k):
                mask = assign == i
                if mask.any():
                    new_centers[i] = x[mask].mean(axis=0)

            if np.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        return centers.astype(np.float32)

    # ============================================================
    # Semantic concept / token access
    # ============================================================
    def get_class_concept_tokens(
        self,
        cls: int,
        split: str = "train",
        num_concepts: int = 4,
        use_cache: bool = True,
    ) -> torch.Tensor:
        cls = int(cls)
        split = split.lower()
        num_concepts = int(max(1, num_concepts))

        self._check_class_split_access(cls, split)

        cache_key = (cls, f"concept_{split}_{num_concepts}")
        if use_cache and cache_key in self._semantic_token_cache:
            return self._semantic_token_cache[cache_key].clone()

        class_indices = self.get_class_indices(cls, split=split)
        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {cls} in split '{split}'")

        class_patches = self.patches[class_indices]
        per_sample_summaries = class_patches.mean(axis=(2, 3)).astype(np.float32)

        if per_sample_summaries.shape[0] <= num_concepts:
            concepts = per_sample_summaries
        else:
            concepts = self._kmeans_numpy(
                per_sample_summaries,
                k=num_concepts,
                seed=self.seed + cls,
            )

        token = torch.from_numpy(concepts).float()

        if use_cache:
            self._semantic_token_cache[cache_key] = token.clone()

        return token

    def get_class_semantic_token(
        self,
        cls: int,
        split: str = "train",
        use_cache: bool = True,
    ) -> torch.Tensor:
        cls = int(cls)
        split = split.lower()
        self._check_class_split_access(cls, split)

        cache_key = (cls, f"coarse_{split}")
        if use_cache and cache_key in self._semantic_token_cache:
            return self._semantic_token_cache[cache_key].clone()

        concepts = self.get_class_concept_tokens(
            cls=cls,
            split=split,
            num_concepts=4,
            use_cache=use_cache,
        )
        token = concepts.mean(dim=0, keepdim=True)

        if use_cache:
            self._semantic_token_cache[cache_key] = token.clone()

        return token

    def clear_semantic_token_cache(self) -> None:
        self._semantic_token_cache.clear()








# from __future__ import annotations

# from contextlib import contextmanager
# from typing import Dict, List, Optional, Set, Tuple, Any

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset, Sampler


# class HSIPatchDataset(Dataset):
#     def __init__(
#         self,
#         patches: np.ndarray,
#         labels: np.ndarray,
#         coords: Optional[np.ndarray] = None,
#         return_metadata: bool = False,
#         center_spectra: Optional[np.ndarray] = None,
#     ):
#         patches = np.ascontiguousarray(patches, dtype=np.float32)
#         labels = np.asarray(labels, dtype=np.int64).reshape(-1)

#         if len(patches) != len(labels):
#             raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

#         self.patches = torch.from_numpy(patches).float()
#         self.labels = torch.from_numpy(labels).long()
#         self.return_metadata = bool(return_metadata)

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
#         # Prefer physical wavelength-ordered raw center spectra when provided by
#         # ImageCubes.  Fallback to the model-input center spectrum, which may be
#         # PCA and must be marked non-physical by the trainer/model.
#         spectrum = self.center_spectra[idx] if self.center_spectra is not None else self._center_spectrum(patch)
#         coord = self.coords[idx] if self.coords.numel() > 0 else torch.empty((0,), dtype=torch.long)
#         return patch, label, spectrum, coord



# class ClassBalancedBatchSampler(Sampler[List[int]]):
#     """
#     Class-balanced batch sampler for geometry-native HSI base training.

#     Local low-rank geometry is not compatible with random imbalanced batches:
#     singleton rare classes produce degenerate covariance and rank-0 geometry.
#     This sampler builds each train batch from multiple classes with multiple
#     samples per class, sampling with replacement when a class is small.
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
#         self.batch_size = int(max(1, batch_size))
#         self.seed = int(seed)
#         self.drop_last = bool(drop_last)

#         self.class_to_indices: Dict[int, np.ndarray] = {}
#         for c in sorted(np.unique(labels_np).tolist()):
#             idx = np.where(labels_np == int(c))[0].astype(np.int64)
#             if idx.size > 0:
#                 self.class_to_indices[int(c)] = idx

#         self.classes = sorted(self.class_to_indices.keys())
#         if not self.classes:
#             raise ValueError("ClassBalancedBatchSampler found no classes.")

#         if classes_per_batch is None or int(classes_per_batch) <= 0:
#             classes_per_batch = min(len(self.classes), self.batch_size)
#         self.classes_per_batch = int(max(1, min(classes_per_batch, len(self.classes), self.batch_size)))

#         if samples_per_class is None or int(samples_per_class) <= 0:
#             samples_per_class = max(2, self.batch_size // self.classes_per_batch)
#         self.samples_per_class = int(max(1, samples_per_class))

#         self.effective_batch_size = self.classes_per_batch * self.samples_per_class
#         if self.effective_batch_size > self.batch_size:
#             self.samples_per_class = max(1, self.batch_size // self.classes_per_batch)
#             self.effective_batch_size = self.classes_per_batch * self.samples_per_class

#         self.num_batches = int(np.ceil(labels_np.size / float(max(self.effective_batch_size, 1))))
#         if self.drop_last:
#             self.num_batches = int(labels_np.size // float(max(self.effective_batch_size, 1)))
#         self.num_batches = max(1, self.num_batches)
#         self.epoch = 0

#     def set_epoch(self, epoch: int) -> None:
#         self.epoch = int(epoch)

#     def __len__(self) -> int:
#         return self.num_batches

#     def __iter__(self):
#         rng = np.random.RandomState(self.seed + self.epoch)
#         classes = np.asarray(self.classes, dtype=np.int64)

#         for _ in range(self.num_batches):
#             replace_classes = len(classes) < self.classes_per_batch
#             chosen_classes = rng.choice(
#                 classes,
#                 size=self.classes_per_batch,
#                 replace=replace_classes,
#             )

#             batch: List[int] = []
#             for c in chosen_classes.tolist():
#                 pool = self.class_to_indices[int(c)]
#                 replace_samples = pool.size < self.samples_per_class
#                 sampled = rng.choice(
#                     pool,
#                     size=self.samples_per_class,
#                     replace=replace_samples,
#                 )
#                 batch.extend(int(i) for i in sampled.tolist())

#             rng.shuffle(batch)
#             if not batch:
#                 continue
#             if self.drop_last and len(batch) < self.effective_batch_size:
#                 continue
#             yield batch


# class IncrementalHSIDataset:
#     """
#     Strict non-exemplar incremental dataset manager.

#     Important:
#     The dataset manager never assumes label 0 is background. If label 0 is in
#     labels, it is treated as a valid class.
#     """

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
#     ):
#         # Raw arrays from ImageCubes.
#         self.patches = np.asarray(patches, dtype=np.float32)
#         self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
#         self.coords = np.asarray(coords, dtype=np.int64)
#         self.gt_shape = gt_shape
#         self.GT = GT
#         self.spectra_are_physical = bool(spectra_are_physical)
#         # raw_spectra/center_spectra are per-sample physical center spectra aligned
#         # with patches/labels. They are descriptors metadata, not old exemplars;
#         # strict non-exemplar access rules below still prevent old raw train use.
#         raw_center = center_spectra if center_spectra is not None else raw_spectra
#         self.raw_spectra = None
#         if raw_center is not None:
#             raw_center = np.asarray(raw_center, dtype=np.float32)
#             if raw_center.ndim != 2:
#                 raw_center = raw_center.reshape(len(self.labels), -1)
#             if len(raw_center) != len(self.labels):
#                 raise ValueError(f"raw_spectra/label length mismatch: {len(raw_center)} vs {len(self.labels)}")
#             self.raw_spectra = np.ascontiguousarray(raw_center, dtype=np.float32)
#             # Do NOT auto-mark metadata as physical.  In reduced-band PCA runs,
#             # center_spectra may be PCA/component vectors, not wavelength-ordered
#             # spectra.  The caller owns this flag.
#             self.spectra_are_physical = bool(spectra_are_physical)

#         if len(self.patches) != len(self.labels):
#             raise ValueError(f"patch/label length mismatch: {len(self.patches)} vs {len(self.labels)}")
#         if len(self.coords) != len(self.labels):
#             raise ValueError(f"coord/label length mismatch: {len(self.coords)} vs {len(self.labels)}")
#         if self.labels.size == 0:
#             raise ValueError("Empty labels passed to IncrementalHSIDataset.")
#         if self.labels.min() < 0:
#             raise ValueError(
#                 f"Negative labels passed to IncrementalHSIDataset: min={self.labels.min()}. "
#                 f"Loader must remove background/ignore labels before incremental split."
#             )

#         # Settings.
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
#         self.target_names = target_names
#         self.label_policy = self._normalize_label_policy(label_policy)
#         self.has_background = bool(self.label_policy.get("has_background", True))
#         self.background_label = self.label_policy.get("background_label", 0 if self.has_background else None)
#         self.raw_class_values = [int(v) for v in self.label_policy.get("raw_class_values", [])]

#         self.pin_memory = self.device.startswith("cuda")

#         # Cache for semantic tokens / concept tokens.
#         self._semantic_token_cache: Dict[Tuple[int, str], torch.Tensor] = {}

#         # Protocol state.
#         self.current_phase: int = 0
#         self.finalized_phases: Set[int] = set()
#         self.finalized_classes: Set[int] = set()
#         self._memory_build_active: bool = False
#         self._memory_build_classes: Set[int] = set()

#         # Class order and remapping.
#         # labels may already be 0..K-1, but we still remap to sequential IDs
#         # according to class_order. If label 0 exists, it is included.
#         self.all_classes = sorted(int(x) for x in np.unique(self.labels).tolist())
#         if self.all_classes[0] != 0:
#             print(
#                 f"[IncrementalHSIDataset:WARN] smallest label is {self.all_classes[0]}, "
#                 f"not 0. The manager will remap to sequential IDs."
#             )

#         self.num_classes = len(self.all_classes)

#         if self.base_classes <= 0 or self.base_classes > self.num_classes:
#             raise ValueError(f"base_classes={self.base_classes} invalid for num_classes={self.num_classes}")
#         if self.increment <= 0:
#             raise ValueError(f"increment must be > 0, got {self.increment}")

#         if shuffle_order:
#             rng = np.random.RandomState(self.seed)
#             self.class_order = rng.permutation(self.all_classes).tolist()
#         else:
#             self.class_order = list(self.all_classes)

#         self.label_map = {global_id: seq_id for seq_id, global_id in enumerate(self.class_order)}
#         self.inv_label_map = {v: k for k, v in self.label_map.items()}

#         self.remapped_labels = np.array(
#             [self.label_map[int(l)] for l in self.labels],
#             dtype=np.int64,
#         )

#         # target_names provided by the loader are indexed by input label id
#         # after ImageCubes mapping. Convert them once to sequential phase ids.
#         self.target_names_by_seq = self._build_target_names_by_seq(target_names)

#         self._validate_class_zero_policy()
#         self._validate_full_class_coverage()

#         # Phase partition in sequential label space.
#         remaining_classes = self.num_classes - self.base_classes
#         self.num_phases = 1 + int(np.ceil(remaining_classes / self.increment))

#         self.phase_to_classes: Dict[int, List[int]] = {}
#         self.class_to_phase: Dict[int, int] = {}

#         for phase in range(self.num_phases):
#             if phase == 0:
#                 cls_list = list(range(self.base_classes))
#             else:
#                 start = self.base_classes + (phase - 1) * self.increment
#                 end = min(start + self.increment, self.num_classes)
#                 cls_list = list(range(start, end))

#             self.phase_to_classes[phase] = cls_list
#             for cls in cls_list:
#                 self.class_to_phase[int(cls)] = int(phase)

#         self._create_splits()
#         self._print_stats()

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
#         return classes

#     # ============================================================
#     # Protocol controls
#     # ============================================================
#     def start_phase(self, phase: int) -> None:
#         phase = self._validate_phase(phase)
#         self.current_phase = phase
#         self._invalidate_train_caches_for_locked_classes()

#     def finalize_phase(self, phase: int) -> None:
#         phase = self._validate_phase(phase)

#         self.finalized_phases.add(phase)
#         self.finalized_classes.update(self.phase_to_classes[phase])
#         self._memory_build_active = False
#         self._memory_build_classes.clear()
#         self._invalidate_train_caches_for_locked_classes()

#     def is_phase_finalized(self, phase: int) -> bool:
#         return int(phase) in self.finalized_phases

#     def is_class_finalized(self, cls: int) -> bool:
#         return int(cls) in self.finalized_classes

#     def get_accessible_train_classes(self) -> List[int]:
#         if not self.strict_non_exemplar:
#             return list(range(self.num_classes))
#         return list(self.phase_to_classes[self.current_phase])

#     def _invalidate_train_caches_for_locked_classes(self) -> None:
#         keys_to_delete = []
#         accessible = set(self.get_accessible_train_classes())

#         for key in list(self._semantic_token_cache.keys()):
#             cls, descriptor = key
#             if "_train" not in descriptor:
#                 continue
#             if self.strict_non_exemplar and int(cls) not in accessible:
#                 keys_to_delete.append(key)

#         for k in keys_to_delete:
#             del self._semantic_token_cache[k]

#     def _is_train_access_allowed(self, cls: int) -> bool:
#         cls = int(cls)

#         if not self.strict_non_exemplar:
#             return True

#         if cls in self.phase_to_classes[self.current_phase]:
#             return True

#         if self._memory_build_active and cls in self._memory_build_classes:
#             return True

#         return False

#     def _check_class_split_access(self, cls: int, split: str) -> None:
#         cls = int(cls)
#         split = str(split).lower()

#         if split != "train":
#             return

#         if not self._is_train_access_allowed(cls):
#             phase_of_cls = self.class_to_phase.get(cls, None)
#             raise PermissionError(
#                 f"Strict non-exemplar protocol violation: raw TRAIN access denied for class {cls} "
#                 f"(phase={phase_of_cls}, current_phase={self.current_phase}, finalized={cls in self.finalized_classes}). "
#                 f"Use stored geometry memory instead of old raw training patches."
#             )

#     @contextmanager
#     def memory_build_context(self, phase: int):
#         phase = int(phase)
#         if phase != self.current_phase:
#             raise ValueError(
#                 f"memory_build_context phase={phase} must match current_phase={self.current_phase}"
#             )

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
#     # Split creation
#     # ============================================================
#     def _create_splits(self) -> None:
#         self.train_indices: List[int] = []
#         self.val_indices: List[int] = []
#         self.test_indices: List[int] = []

#         for seq_id in range(self.num_classes):
#             class_indices = np.where(self.remapped_labels == seq_id)[0]
#             n_samples = len(class_indices)

#             if n_samples == 0:
#                 continue

#             rng = np.random.RandomState(self.seed + seq_id)
#             shuffled = rng.permutation(class_indices)

#             if n_samples == 1:
#                 self.train_indices.extend(shuffled.tolist())
#                 continue

#             if n_samples == 2:
#                 self.train_indices.append(int(shuffled[0]))
#                 self.test_indices.append(int(shuffled[1]))
#                 continue

#             n_train = max(self.min_train_per_class, int(round(n_samples * self.train_ratio)))
#             n_val = max(1, int(round(n_samples * self.val_ratio)))

#             if n_train + n_val >= n_samples:
#                 overflow = (n_train + n_val) - (n_samples - 1)

#                 reduce_val = min(overflow, max(0, n_val - 1))
#                 n_val -= reduce_val
#                 overflow -= reduce_val

#                 if overflow > 0:
#                     n_train = max(1, n_train - overflow)

#             n_train = max(1, min(n_train, n_samples - 2))
#             n_val = max(1, min(n_val, n_samples - n_train - 1))
#             n_test = n_samples - n_train - n_val

#             assert n_train >= 1
#             assert n_val >= 1
#             assert n_test >= 1
#             assert n_train + n_val + n_test == n_samples

#             self.train_indices.extend(shuffled[:n_train].tolist())
#             self.val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
#             self.test_indices.extend(shuffled[n_train + n_val:].tolist())

#         self.train_indices = np.array(self.train_indices, dtype=np.int64)
#         self.val_indices = np.array(self.val_indices, dtype=np.int64)
#         self.test_indices = np.array(self.test_indices, dtype=np.int64)

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
#             f"  Label policy: has_background={self.has_background}, "
#             f"background_label={self.background_label}, raw_class_values={self.raw_class_values}"
#         )
#         print(
#             "  Sequential class 0 present: "
#             f"{0 in self.all_classes} "
#             "(this is a real training class id, not automatically background)"
#         )

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

#     def get_class_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
#         """Return per-sample physical center spectra for one class/split.

#         This follows the same strict non-exemplar access rule as get_class_patches:
#         old train spectra are denied after their phase is finalized. They are not
#         stored as memory; they are used only while building the current class row.
#         """
#         if self.raw_spectra is None:
#             raise AttributeError("No raw/physical center spectra are available in this dataset.")
#         idx = self.get_class_indices(int(cls), split=split)
#         if len(idx) == 0:
#             raise ValueError(f"No spectra found for class {cls} in split '{split}'")
#         return torch.from_numpy(np.ascontiguousarray(self.raw_spectra[idx])).float()

#     def get_class_center_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
#         return self.get_class_spectra(cls, split=split)

#     def get_class_raw_spectra(self, cls: int, split: str = "train") -> torch.Tensor:
#         return self.get_class_spectra(cls, split=split)

#     def get_class_spectral_summary(self, cls: int, split: str = "train") -> torch.Tensor:
#         return self.get_class_spectra(cls, split=split)

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
#             "memory_build_active": bool(self._memory_build_active),
#             "memory_build_classes": sorted(int(c) for c in self._memory_build_classes),
#         }

#     def assert_no_old_train_access(self, cls: int) -> None:
#         self._check_class_split_access(int(cls), "train")

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
#             center_spectra=(self.raw_spectra[idx] if self.raw_spectra is not None else None),
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

#         if split == "train" and self.strict_non_exemplar and not allow_train_old:
#             # Critical protocol rule: training loaders never expose old raw samples.
#             active_classes = list(self.phase_to_classes[self.current_phase])
#         else:
#             active_classes: List[int] = []
#             for p in range(up_to_phase + 1):
#                 active_classes.extend(self.phase_to_classes[p])

#         indices = self._get_split_indices(split)
#         mask = np.isin(self.remapped_labels[indices], active_classes)
#         idx = indices[mask]

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

#     def get_cumulative_test_data(self, phase: int):
#         phase = self._validate_phase(phase)
#         active_classes = self.get_classes_up_to_phase(phase)
#         mask = np.isin(self.remapped_labels[self.test_indices], active_classes)
#         idx = self.test_indices[mask]
#         return self.patches[idx], self.remapped_labels[idx], self.coords[idx]

#     # ============================================================
#     # K-means concept extraction
#     # ============================================================
#     def _kmeans_numpy(
#         self,
#         x: np.ndarray,
#         k: int,
#         seed: int = 42,
#         max_iters: int = 30,
#     ) -> np.ndarray:
#         if x.ndim != 2:
#             raise ValueError(f"Expected x to be 2D, got shape={x.shape}")

#         n, d = x.shape
#         if n == 0:
#             raise ValueError("Empty input to k-means")

#         k = max(1, min(int(k), n))
#         rng = np.random.RandomState(seed)

#         centers = np.empty((k, d), dtype=np.float32)
#         first = rng.randint(0, n)
#         centers[0] = x[first]
#         dist2 = ((x - centers[0]) ** 2).sum(axis=1)

#         for i in range(1, k):
#             probs = dist2 / max(dist2.sum(), 1e-12)
#             idx = rng.choice(n, p=probs)
#             centers[i] = x[idx]
#             dist2 = np.minimum(dist2, ((x - centers[i]) ** 2).sum(axis=1))

#         for _ in range(max_iters):
#             assign = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
#             new_centers = centers.copy()
#             for i in range(k):
#                 mask = assign == i
#                 if mask.any():
#                     new_centers[i] = x[mask].mean(axis=0)

#             if np.allclose(new_centers, centers, atol=1e-5):
#                 centers = new_centers
#                 break
#             centers = new_centers

#         return centers.astype(np.float32)

#     # ============================================================
#     # Semantic concept / token access
#     # ============================================================
#     def get_class_concept_tokens(
#         self,
#         cls: int,
#         split: str = "train",
#         num_concepts: int = 4,
#         use_cache: bool = True,
#     ) -> torch.Tensor:
#         cls = int(cls)
#         split = split.lower()
#         num_concepts = int(max(1, num_concepts))

#         self._check_class_split_access(cls, split)

#         cache_key = (cls, f"concept_{split}_{num_concepts}")
#         if use_cache and cache_key in self._semantic_token_cache:
#             return self._semantic_token_cache[cache_key].clone()

#         class_indices = self.get_class_indices(cls, split=split)
#         if len(class_indices) == 0:
#             raise ValueError(f"No samples found for class {cls} in split '{split}'")

#         class_patches = self.patches[class_indices]
#         per_sample_summaries = class_patches.mean(axis=(2, 3)).astype(np.float32)

#         if per_sample_summaries.shape[0] <= num_concepts:
#             concepts = per_sample_summaries
#         else:
#             concepts = self._kmeans_numpy(
#                 per_sample_summaries,
#                 k=num_concepts,
#                 seed=self.seed + cls,
#             )

#         token = torch.from_numpy(concepts).float()

#         if use_cache:
#             self._semantic_token_cache[cache_key] = token.clone()

#         return token

#     def get_class_semantic_token(
#         self,
#         cls: int,
#         split: str = "train",
#         use_cache: bool = True,
#     ) -> torch.Tensor:
#         cls = int(cls)
#         split = split.lower()
#         self._check_class_split_access(cls, split)

#         cache_key = (cls, f"coarse_{split}")
#         if use_cache and cache_key in self._semantic_token_cache:
#             return self._semantic_token_cache[cache_key].clone()

#         concepts = self.get_class_concept_tokens(
#             cls=cls,
#             split=split,
#             num_concepts=4,
#             use_cache=use_cache,
#         )
#         token = concepts.mean(dim=0, keepdim=True)

#         if use_cache:
#             self._semantic_token_cache[cache_key] = token.clone()

#         return token

#     def clear_semantic_token_cache(self) -> None:
#         self._semantic_token_cache.clear()
