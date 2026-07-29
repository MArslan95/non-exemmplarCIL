from __future__ import annotations

"""Base-phase training for transport-verified HSI factor geometry.

This trainer is aligned with the revised architecture:

    raw ordered center spectrum -> spectral branch -> z_s
    center-relative reduced patch -> spatial branch -> z_p
    z = [z_s ; z_p]

Persistent class memory after base handoff:

    p(z | c) = N(mu_c, L_c L_c^T + Psi_c)

with

    Psi_c = diag(psi_c,s I_Ds, psi_c,p I_Dp).

The ordered-spectrum relation p(h | c) is stored only for pair-risk margins.
It is never a second classifier.

Base protocol
-------------
1. Train the backbone and a temporary linear head with class-balanced CE.
2. At the first geometry epoch, fit and freeze aggregate global priors from the
   current full base-training representation.
3. Fit one frozen pair-risk temperature contract from provisional full-training
   rows, without committing any class row.
4. Shape the representation with bidirectional support/query factor-energy
   episodes. Support rows are detached; gradients flow through query features.
5. Select checkpoints using current validation factor geometry, not the
   temporary linear head.
6. Restore the best representation, run out-of-fold geometry evaluation, fit
   final rows once, and atomically commit the base GeometryBank.
7. Delete the temporary linear head and freeze the accepted base backbone.

No old samples, old features, teacher, spectral-conditioned classifier,
trainable transport, feature replay, or persistent class row is used during
base optimization.
"""

import copy
import json
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

try:
    from losses.loss import (
        base_ce_warmup_objective,
        base_training_objective,
        pair_risk_confusion_statistics,
        pairwise_directional_invasion_matrix,
        risk_guided_all_rival_geometry_loss,
    )
except ImportError:  # Standalone validation against generated files.
    from factor_geometry_loss_v2 import (
        base_ce_warmup_objective,
        base_training_objective,
        pair_risk_confusion_statistics,
        pairwise_directional_invasion_matrix,
        risk_guided_all_rival_geometry_loss,
    )


_MISSING = object()
_CLASSIFICATION_FACTORIZATION = "p(z|c)"
_SPECTRAL_RELATION = "p(h|c)"


# =============================================================================
# Shared helpers
# =============================================================================


def _unique_ids(
    values: Iterable[int],
    *,
    name: str = "class_ids",
    allow_empty: bool = False,
) -> List[int]:
    output: List[int] = []
    observed: set[int] = set()
    for value in values:
        class_id = int(value)
        if class_id < 0:
            raise ValueError(f"{name} contains negative class ID {class_id}")
        if class_id in observed:
            raise ValueError(f"{name} contains duplicate class ID {class_id}")
        observed.add(class_id)
        output.append(class_id)
    if not output and not allow_empty:
        raise ValueError(f"{name} is empty")
    return output


def _item_label(item: Any) -> int:
    if isinstance(item, Mapping):
        label = item.get(
            "label",
            item.get("labels", item.get("target")),
        )
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        label = item[1]
    else:
        raise RuntimeError(
            "balanced episodic sampling requires a map-style sample "
            "containing one label"
        )

    tensor = torch.as_tensor(label).long().reshape(-1)
    if tensor.numel() != 1:
        raise RuntimeError(
            "each dataset item must contain exactly one label"
        )
    return int(tensor.item())


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


# =============================================================================
# Balanced episodic sampler
# =============================================================================


class GeometryBalancedBatchSampler(Sampler[List[int]]):
    """M-way/K-shot base sampler with balanced class-pair exposure.

    The sampler is used only while training phase 0. It does not retain any
    sample after the base phase and is not exemplar memory.
    """

    def __init__(
        self,
        dataset: Any,
        class_ids: Sequence[int],
        *,
        classes_per_batch: int,
        samples_per_class: int,
        num_batches: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if (
            not hasattr(dataset, "__len__")
            or not hasattr(dataset, "__getitem__")
        ):
            raise TypeError(
                "GeometryBalancedBatchSampler requires a map-style dataset"
            )

        self.dataset = dataset
        self.class_ids = _unique_ids(class_ids)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0

        if not 2 <= self.classes_per_batch <= len(self.class_ids):
            raise ValueError(
                "classes_per_batch must lie in [2, number of base classes]"
            )
        if (
            self.samples_per_class < 6
            or self.samples_per_class % 2 != 0
        ):
            raise ValueError(
                "samples_per_class must be even and at least six"
            )

        self.indices_by_class: Dict[int, List[int]] = {
            class_id: []
            for class_id in self.class_ids
        }
        allowed = set(self.class_ids)
        for index in range(len(dataset)):
            class_id = _item_label(dataset[index])
            if class_id in allowed:
                self.indices_by_class[class_id].append(index)

        missing = [
            class_id
            for class_id, indices
            in self.indices_by_class.items()
            if not indices
        ]
        if missing:
            raise RuntimeError(
                f"base split has no samples for classes {missing}"
            )

        insufficient = {
            class_id: len(indices)
            for class_id, indices
            in self.indices_by_class.items()
            if len(indices) < self.samples_per_class
        }
        if insufficient:
            raise RuntimeError(
                "the base split cannot provide unique within-episode samples; "
                f"reduce geometry_samples_per_class: {insufficient}"
            )

        episode_size = (
            self.classes_per_batch
            * self.samples_per_class
        )
        sample_total = sum(
            len(values)
            for values in self.indices_by_class.values()
        )
        self.num_batches = int(
            num_batches
            if num_batches is not None
            else max(
                1,
                math.ceil(
                    sample_total / float(episode_size)
                ),
            )
        )

        self.last_pair_counts: Dict[
            Tuple[int, int],
            int,
        ] = {}
        self.last_class_counts: Dict[int, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def _subset_score(
        subset: Sequence[int],
        pair_counts: Mapping[
            Tuple[int, int],
            int,
        ],
        class_counts: Mapping[int, int],
    ) -> Tuple[int, int, int, Tuple[int, ...]]:
        pairs = [
            tuple(sorted((
                int(subset[left]),
                int(subset[right]),
            )))
            for left in range(len(subset))
            for right in range(left + 1, len(subset))
        ]
        pair_values = [
            pair_counts.get(pair, 0)
            for pair in pairs
        ]
        class_values = [
            class_counts.get(int(class_id), 0)
            for class_id in subset
        ]
        return (
            sum(pair_values),
            max(pair_values, default=0),
            sum(class_values),
            tuple(sorted(int(v) for v in subset)),
        )

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator().manual_seed(
            self.seed + 1009 * self.epoch
        )
        pair_counts: Dict[
            Tuple[int, int],
            int,
        ] = {}
        class_counts = {
            class_id: 0
            for class_id in self.class_ids
        }

        for batch_index in range(self.num_batches):
            if self.classes_per_batch == len(self.class_ids):
                chosen = list(self.class_ids)
            else:
                candidates: List[List[int]] = []
                offset = (
                    batch_index
                    * self.classes_per_batch
                ) % len(self.class_ids)
                candidates.append(
                    [
                        self.class_ids[
                            (offset + position)
                            % len(self.class_ids)
                        ]
                        for position
                        in range(self.classes_per_batch)
                    ]
                )
                for _ in range(64):
                    order = torch.randperm(
                        len(self.class_ids),
                        generator=generator,
                    )
                    candidates.append(
                        [
                            self.class_ids[int(index)]
                            for index in order[
                                : self.classes_per_batch
                            ]
                        ]
                    )
                chosen = min(
                    candidates,
                    key=lambda subset: self._subset_score(
                        subset,
                        pair_counts,
                        class_counts,
                    ),
                )

            for class_id in chosen:
                class_counts[class_id] += 1
            for left in range(len(chosen)):
                for right in range(left + 1, len(chosen)):
                    pair = tuple(sorted((
                        chosen[left],
                        chosen[right],
                    )))
                    pair_counts[pair] = (
                        pair_counts.get(pair, 0) + 1
                    )

            episode: List[int] = []
            for class_id in chosen:
                source = self.indices_by_class[class_id]
                order = torch.randperm(
                    len(source),
                    generator=generator,
                )[: self.samples_per_class]
                selected = [
                    source[int(index)]
                    for index in order
                ]
                if len(selected) != len(set(selected)):
                    raise RuntimeError(
                        f"duplicate sample inside class {class_id}"
                    )
                episode.extend(selected)

            if len(episode) != len(set(episode)):
                raise RuntimeError(
                    "duplicate dataset index inside one episode"
                )
            shuffle = torch.randperm(
                len(episode),
                generator=generator,
            ).tolist()
            yield [
                episode[index]
                for index in shuffle
            ]

        self.last_pair_counts = pair_counts
        self.last_class_counts = class_counts


# =============================================================================
# Base trainer
# =============================================================================


class BasePhaseTrainer:
    """Phase-0 trainer for factor GeometryBank preparation.

    Expected host attributes
    ------------------------
    model:
        Revised ``NECILModel``.
    dataset:
        Incremental dataset wrapper exposing phase dataloaders.
    device:
        Optional device; defaults to ``model.device``.
    args:
        Optional configuration namespace.
    _unpack_hsi_batch(batch):
        Existing trainer helper returning at least processed patch and labels.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _cfg_any(
        self,
        names: Sequence[str],
        *,
        default: Any = _MISSING,
    ) -> Any:
        for name in names:
            local = getattr(self, name, _MISSING)
            if local is not _MISSING and local is not None:
                return local
            args = getattr(self, "args", None)
            if args is not None and hasattr(args, name):
                value = getattr(args, name)
                if value is not None:
                    return value
        if default is _MISSING:
            raise RuntimeError(
                "missing required option; expected one of "
                f"{list(names)}"
            )
        return default

    def _cfg_float(
        self,
        *names: str,
        default: Any = _MISSING,
    ) -> float:
        return _finite_float(
            self._cfg_any(names, default=default),
            name=names[0],
        )

    def _cfg_int(
        self,
        *names: str,
        default: Any = _MISSING,
    ) -> int:
        value = self._cfg_any(
            names,
            default=default,
        )
        if (
            isinstance(value, bool)
            or float(value) != float(int(value))
        ):
            raise RuntimeError(
                f"{names[0]} must be an integer"
            )
        return int(value)

    @staticmethod
    def _parse_bool(
        value: Any,
        name: str,
    ) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }:
                return True
            if token in {
                "0",
                "false",
                "no",
                "n",
                "off",
            }:
                return False
        raise RuntimeError(
            f"{name} must be an explicit boolean"
        )

    def _cfg_bool(
        self,
        *names: str,
        default: Any = _MISSING,
    ) -> bool:
        return self._parse_bool(
            self._cfg_any(
                names,
                default=default,
            ),
            names[0],
        )

    @property
    def _trainer_device(self) -> torch.device:
        return torch.device(
            getattr(
                self,
                "device",
                getattr(self.model, "device", "cpu"),
            )
        )

    def _warmup_epochs(self) -> int:
        value = self._cfg_int(
            "base_ce_warmup_epochs",
            "base_geometry_warmup_epochs",
            default=10,
        )
        if value < 0:
            raise RuntimeError(
                "base_ce_warmup_epochs must be non-negative"
            )
        return value

    def _target_geometry_weight(self) -> float:
        value = self._cfg_float(
            "base_geometry_weight",
            default=1.0,
        )
        if value < 0.0:
            raise RuntimeError(
                "base_geometry_weight must be non-negative"
            )
        return value

    def _geometry_weight_for_epoch(
        self,
        epoch: int,
    ) -> float:
        target = self._target_geometry_weight()
        warmup = self._warmup_epochs()
        ramp = self._cfg_int(
            "base_geometry_ramp_epochs",
            default=10,
        )
        if ramp < 0:
            raise RuntimeError(
                "base_geometry_ramp_epochs must be non-negative"
            )
        if epoch < warmup or target == 0.0:
            return 0.0
        if ramp == 0:
            return target
        progress = min(
            1.0,
            float(epoch - warmup + 1)
            / float(ramp),
        )
        return target * progress

    def _base_margin(self) -> float:
        value = self._cfg_float(
            "base_geometry_margin",
            default=0.50,
        )
        if value < 0.0:
            raise RuntimeError(
                "base_geometry_margin must be non-negative"
            )
        return value

    def _risk_strength(self) -> float:
        value = self._cfg_float(
            "base_risk_strength",
            "risk_strength",
            default=0.50,
        )
        if value < 0.0:
            raise RuntimeError(
                "base_risk_strength must be non-negative"
            )
        return value

    def _geometry_temperature(self) -> float:
        value = self._cfg_float(
            "base_geometry_temperature",
            default=0.50,
        )
        if value <= 0.0:
            raise RuntimeError(
                "base_geometry_temperature must be positive"
            )
        return value

    def _maximum_margin(self) -> Optional[float]:
        value = self._cfg_any(
            ("base_maximum_pair_margin",),
            default=None,
        )
        if value is None:
            return None
        result = _finite_float(
            value,
            name="base_maximum_pair_margin",
        )
        if result <= 0.0:
            raise RuntimeError(
                "base_maximum_pair_margin must be positive"
            )
        return result

    def _validate_base_configuration(
        self,
        phase_class_ids: Sequence[int],
    ) -> None:
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError(
                "BasePhaseTrainer has no model"
            )
        if not callable(
            getattr(
                model,
                "assert_architecture_contract",
                None,
            )
        ):
            raise RuntimeError(
                "model must expose "
                "assert_architecture_contract()"
            )
        model.assert_architecture_contract()

        summary = model.architecture_summary()
        if summary.get(
            "classification_factorization"
        ) != _CLASSIFICATION_FACTORIZATION:
            raise RuntimeError(
                "base trainer requires factor geometry p(z|c)"
            )
        if summary.get(
            "spectral_relation_factorization"
        ) != _SPECTRAL_RELATION:
            raise RuntimeError(
                "base trainer requires separate pair-risk p(h|c)"
            )
        if summary.get(
            "joint_feature"
        ) != "direct_[z_s;z_p]":
            raise RuntimeError(
                "joint feature must be exact [z_s;z_p]"
            )
        if bool(
            summary.get(
                "trainable_transport_network",
                True,
            )
        ):
            raise RuntimeError(
                "base architecture contains a trainable transport"
            )
        if bool(
            summary.get(
                "uses_geometry_replay_for_training",
                True,
            )
        ):
            raise RuntimeError(
                "geometry replay must not be a training mechanism"
            )

        if (
            model.feature_dim
            != model.spectral_dim + model.spatial_dim
        ):
            raise RuntimeError(
                "model geometry dimensions are inconsistent"
            )
        if (
            model.geometry_bank.feature_dim
            != model.feature_dim
        ):
            raise RuntimeError(
                "GeometryBank feature dimension mismatch"
            )
        if bool(
            model.geometry_bank.valid_mask().any().item()
        ):
            raise RuntimeError(
                "base optimization must start with no committed rows"
            )
        if bool(
            model.classifier.require_bound_contract
        ):
            raise RuntimeError(
                "classifier must remain unbound before base handoff"
            )
        if any(
            parameter.requires_grad
            for parameter
            in model.geometry_bank.parameters()
        ):
            raise RuntimeError(
                "GeometryBank must contain buffers only"
            )
        if any(
            parameter.requires_grad
            for parameter
            in model.classifier.parameters()
        ):
            raise RuntimeError(
                "classifier must be parameter-free"
            )

        ids = _unique_ids(
            phase_class_ids,
            name="phase_class_ids",
        )
        if len(ids) < 2:
            raise RuntimeError(
                "base phase requires at least two classes"
            )

        self._base_margin()
        self._risk_strength()
        self._geometry_temperature()
        self._maximum_margin()

        smoothing = self._cfg_float(
            "label_smoothing",
            default=0.0,
        )
        if not 0.0 <= smoothing < 1.0:
            raise RuntimeError(
                "label_smoothing must lie in [0,1)"
            )
        if (
            self._cfg_float(
                "grad_clip_base",
                default=5.0,
            )
            <= 0.0
        ):
            raise RuntimeError(
                "grad_clip_base must be positive"
            )
        if (
            self._cfg_float(
                "weight_decay",
                default=1e-4,
            )
            < 0.0
        ):
            raise RuntimeError(
                "weight_decay must be non-negative"
            )
        if not self._cfg_bool(
            "base_class_balance",
            default=True,
        ):
            raise RuntimeError(
                "base_class_balance must be true"
            )
        if not self._cfg_bool(
            "base_use_geometry_balanced_sampler",
            default=True,
        ):
            raise RuntimeError(
                "base geometry requires balanced episodes"
            )

        samples_per_class = self._cfg_int(
            "geometry_samples_per_class",
            default=8,
        )
        if (
            samples_per_class < 6
            or samples_per_class % 2 != 0
        ):
            raise RuntimeError(
                "geometry_samples_per_class must be even "
                "and at least six"
            )

    # ------------------------------------------------------------------
    # Loader and batch contract
    # ------------------------------------------------------------------

    def _build_geometry_balanced_loader(
        self,
        source_loader: DataLoader,
        phase_class_ids: Sequence[int],
    ) -> DataLoader:
        class_count = len(phase_class_ids)
        configured_m = self._cfg_int(
            "geometry_batch_classes",
            default=0,
        )
        classes_per_batch = (
            class_count
            if configured_m == 0
            else configured_m
        )
        samples_per_class = self._cfg_int(
            "geometry_samples_per_class",
            default=8,
        )
        sampler = GeometryBalancedBatchSampler(
            source_loader.dataset,
            phase_class_ids,
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class,
            num_batches=(
                None
                if self._cfg_int(
                    "base_steps_per_epoch",
                    default=0,
                )
                <= 0
                else self._cfg_int(
                    "base_steps_per_epoch",
                    default=0,
                )
            ),
            seed=self._cfg_int(
                "seed",
                default=0,
            ),
        )
        self._episode_classes = classes_per_batch
        self._episode_samples_per_class = (
            samples_per_class
        )

        kwargs: Dict[str, Any] = {
            "dataset": source_loader.dataset,
            "batch_sampler": sampler,
            "num_workers": int(
                getattr(
                    source_loader,
                    "num_workers",
                    0,
                )
            ),
            "collate_fn": getattr(
                source_loader,
                "collate_fn",
                None,
            ),
            "pin_memory": bool(
                getattr(
                    source_loader,
                    "pin_memory",
                    False,
                )
            ),
            "worker_init_fn": getattr(
                source_loader,
                "worker_init_fn",
                None,
            ),
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = bool(
                getattr(
                    source_loader,
                    "persistent_workers",
                    False,
                )
            )
            prefetch = getattr(
                source_loader,
                "prefetch_factor",
                None,
            )
            if prefetch is not None:
                kwargs["prefetch_factor"] = prefetch

        loader = DataLoader(**kwargs)
        print(
            "[Base episodes] "
            f"classes={class_count} | "
            f"M={classes_per_batch} | "
            f"K={samples_per_class} | "
            f"batch={classes_per_batch * samples_per_class} | "
            f"steps={len(sampler)}"
        )
        return loader

    def _assert_geometry_batch(
        self,
        labels_global: Tensor,
    ) -> None:
        unique, counts = torch.unique(
            labels_global,
            sorted=True,
            return_counts=True,
        )
        expected_m = int(
            getattr(
                self,
                "_episode_classes",
                unique.numel(),
            )
        )
        expected_k = int(
            getattr(
                self,
                "_episode_samples_per_class",
                counts.min().item(),
            )
        )
        if int(unique.numel()) != expected_m:
            raise RuntimeError(
                f"episode has {unique.numel()} classes; "
                f"expected {expected_m}"
            )
        if not bool(
            counts.eq(expected_k).all().item()
        ):
            raise RuntimeError(
                f"episode counts={counts.tolist()}, "
                f"expected K={expected_k}"
            )

    @staticmethod
    def _mapping_sources(
        batch: Any,
    ) -> List[Mapping[str, Any]]:
        if not isinstance(batch, Mapping):
            return []
        sources: List[Mapping[str, Any]] = [batch]
        for key in (
            "hsi",
            "inputs",
            "spectral",
            "metadata",
            "raw",
        ):
            nested = batch.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)
        return sources

    def _find_raw_inputs(
        self,
        batch: Any,
        processed_patch: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        raw_patch_names = (
            "raw_spectral_patch",
            "raw_spectral_patches",
            "raw_patch",
            "raw_patches",
            "physical_spectral_patch",
            "physical_patch",
            "original_spectral_patch",
            "original_patch",
            "hsi_raw_patch",
        )
        raw_center_names = (
            "raw_center_spectrum",
            "raw_center_spectra",
            "center_spectrum_raw",
            "center_spectra_raw",
        )

        raw_patch: Optional[Tensor] = None
        raw_center: Optional[Tensor] = None
        for source in self._mapping_sources(batch):
            for name in raw_patch_names:
                value = source.get(name)
                if torch.is_tensor(value):
                    raw_patch = value
                    break
            for name in raw_center_names:
                value = source.get(name)
                if torch.is_tensor(value):
                    raw_center = value
                    break

        expected_patch = (
            processed_patch.size(0),
            int(self.model.backbone.raw_num_bands),
            processed_patch.size(2),
            processed_patch.size(3),
        )
        expected_center = (
            processed_patch.size(0),
            int(self.model.backbone.raw_num_bands),
        )

        if isinstance(batch, (tuple, list)):
            for value in batch[2:]:
                if not torch.is_tensor(value):
                    continue
                if (
                    raw_patch is None
                    and tuple(value.shape)
                    == expected_patch
                ):
                    raw_patch = value
                elif (
                    raw_center is None
                    and tuple(value.shape)
                    == expected_center
                ):
                    raw_center = value

        if raw_patch is not None and raw_center is not None:
            # Keep the full patch when both are present; the backbone extracts
            # the center deterministically.
            raw_center = None

        if (
            raw_patch is None
            and raw_center is None
            and tuple(processed_patch.shape)
            == expected_patch
        ):
            raw_patch = processed_patch

        if raw_patch is None and raw_center is None:
            raise RuntimeError(
                "the revised backbone requires either an aligned raw physical "
                "spectral patch or a raw ordered center spectrum"
            )

        if raw_patch is not None:
            if tuple(raw_patch.shape) != expected_patch:
                raise RuntimeError(
                    "raw spectral patch shape mismatch: "
                    f"expected {expected_patch}, "
                    f"got {tuple(raw_patch.shape)}"
                )
            if not torch.isfinite(raw_patch).all():
                raise RuntimeError(
                    "raw spectral patch contains NaN/Inf"
                )

        if raw_center is not None:
            if tuple(raw_center.shape) != expected_center:
                raise RuntimeError(
                    "raw center spectrum shape mismatch: "
                    f"expected {expected_center}, "
                    f"got {tuple(raw_center.shape)}"
                )
            if not torch.isfinite(raw_center).all():
                raise RuntimeError(
                    "raw center spectrum contains NaN/Inf"
                )

        return raw_patch, raw_center

    def _extract_batch_payload(
        self,
        batch: Any,
        *,
        require_grad: bool,
    ) -> Dict[str, Tensor]:
        unpack = getattr(
            self,
            "_unpack_hsi_batch",
            None,
        )
        if not callable(unpack):
            raise RuntimeError(
                "trainer must expose _unpack_hsi_batch(batch)"
            )

        unpacked = unpack(batch)
        if not isinstance(
            unpacked,
            (tuple, list),
        ) or len(unpacked) < 2:
            raise RuntimeError(
                "_unpack_hsi_batch must return at least "
                "(processed_patch, labels)"
            )
        processed_patch = torch.as_tensor(
            unpacked[0],
        ).float()
        labels = torch.as_tensor(
            unpacked[1],
        ).long().flatten()

        raw_patch, raw_center = self._find_raw_inputs(
            batch,
            processed_patch,
        )

        processed_device = processed_patch.to(
            self._trainer_device,
            non_blocking=True,
        )
        labels_device = labels.to(
            self._trainer_device,
            non_blocking=True,
        )
        raw_patch_device = (
            None
            if raw_patch is None
            else raw_patch.float().to(
                self._trainer_device,
                non_blocking=True,
            )
        )
        raw_center_device = (
            None
            if raw_center is None
            else raw_center.float().to(
                self._trainer_device,
                non_blocking=True,
            )
        )

        context = (
            nullcontext()
            if require_grad
            else torch.inference_mode()
        )
        with context:
            output = self.model.forward_features(
                processed_device,
                raw_spectral_patch=
                    raw_patch_device,
                raw_center_spectrum=
                    raw_center_device,
                deterministic=not require_grad,
            )

        joint = output["joint_feature"]
        spectral = output["spectral_feature"]
        spatial = output["spatial_feature"]
        raw_spectrum = output[
            "raw_center_spectrum"
        ]
        batch_size = labels_device.numel()

        if joint.shape != (
            batch_size,
            self.model.feature_dim,
        ):
            raise RuntimeError(
                "joint feature payload is misaligned"
            )
        if spectral.shape != (
            batch_size,
            self.model.spectral_dim,
        ):
            raise RuntimeError(
                "spectral feature payload is misaligned"
            )
        if spatial.shape != (
            batch_size,
            self.model.spatial_dim,
        ):
            raise RuntimeError(
                "spatial feature payload is misaligned"
            )
        if raw_spectrum.shape != (
            batch_size,
            self.model.backbone.raw_num_bands,
        ):
            raise RuntimeError(
                "raw center spectrum payload is misaligned"
            )
        if not torch.equal(
            joint,
            torch.cat(
                [spectral, spatial],
                dim=1,
            ),
        ):
            raise RuntimeError(
                "joint feature is not exact [z_s;z_p]"
            )

        return {
            "patches": processed_device,
            "labels": labels_device,
            "spectral_feature": spectral,
            "spatial_feature": spatial,
            "joint_feature": joint,
            "features": joint,
            "raw_center_spectrum":
                raw_spectrum,
        }

    @torch.no_grad()
    def _collect_payload(
        self,
        loader: DataLoader,
    ) -> Dict[str, Tensor]:
        outputs: Dict[str, List[Tensor]] = {
            "spectral_feature": [],
            "spatial_feature": [],
            "joint_feature": [],
            "raw_center_spectrum": [],
            "labels": [],
        }

        was_training = self.model.training
        self.model.eval()
        memory_context = (
            self.dataset.memory_build_context(0)
            if hasattr(
                self.dataset,
                "memory_build_context",
            )
            else nullcontext()
        )
        try:
            with memory_context:
                for batch in loader:
                    payload = self._extract_batch_payload(
                        batch,
                        require_grad=False,
                    )
                    for name in outputs:
                        outputs[name].append(
                            payload[name].detach()
                        )
        finally:
            self.model.train(was_training)

        if not outputs["labels"]:
            raise RuntimeError(
                "cannot collect payload from an empty loader"
            )
        result = {
            name: torch.cat(values, dim=0)
            for name, values in outputs.items()
        }
        result["features"] = result["joint_feature"]
        count = result["labels"].numel()
        if any(
            value.size(0) != count
            for name, value in result.items()
            if name not in {"labels"}
        ):
            raise RuntimeError(
                "collected payload is misaligned"
            )
        return result

    # ------------------------------------------------------------------
    # Base trainability
    # ------------------------------------------------------------------

    def _make_base_head(
        self,
        phase_class_ids: Sequence[int],
    ) -> nn.Linear:
        head = self.model.ensure_base_ce_head(
            phase_class_ids
        )
        if not isinstance(head, nn.Linear):
            raise RuntimeError(
                "ensure_base_ce_head() must return nn.Linear"
            )
        if head.in_features != self.model.feature_dim:
            raise RuntimeError(
                "base CE head input width is not model.feature_dim"
            )
        return head

    def _set_base_trainability(
        self,
        base_head: nn.Module,
    ) -> None:
        self.model.set_base_mode()
        for parameter in base_head.parameters():
            parameter.requires_grad_(True)

        forbidden = [
            name
            for name, parameter
            in self.model.named_parameters()
            if parameter.requires_grad
            and not (
                name.startswith("backbone.")
                or name.startswith("base_ce_head.")
            )
        ]
        if forbidden:
            raise RuntimeError(
                "unexpected trainable base parameters: "
                f"{forbidden[:20]}"
            )

    def _base_trainable_parameters(
        self,
    ) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        observed: set[int] = set()
        for parameter in self.model.parameters():
            if (
                parameter.requires_grad
                and id(parameter) not in observed
            ):
                observed.add(id(parameter))
                parameters.append(parameter)
        if not parameters:
            raise RuntimeError(
                "base phase has no trainable parameters"
            )
        return parameters

    # ------------------------------------------------------------------
    # Cross-fitting
    # ------------------------------------------------------------------

    def _crossfit_indices(
        self,
        labels_global: Tensor,
        class_order_global: Sequence[int],
        *,
        step: int,
        epoch: int,
    ) -> Tuple[Tensor, Tensor]:
        support: List[Tensor] = []
        query: List[Tensor] = []
        minimum = self._cfg_int(
            "base_min_samples_per_class_in_batch",
            default=6,
        )
        for class_id in class_order_global:
            indices = torch.nonzero(
                labels_global.eq(int(class_id)),
                as_tuple=False,
            ).flatten()
            if indices.numel() < minimum:
                raise RuntimeError(
                    f"class {class_id} has "
                    f"{indices.numel()} samples; "
                    f"expected at least {minimum}"
                )

            generator = torch.Generator().manual_seed(
                self._cfg_int(
                    "seed",
                    default=0,
                )
                + 130363 * int(epoch)
                + 8191 * int(step)
                + 257 * int(class_id)
            )
            order = torch.randperm(
                indices.numel(),
                generator=generator,
            ).to(indices.device)
            shuffled = indices.index_select(
                0,
                order,
            )
            split = shuffled.numel() // 2
            first = shuffled[:split]
            second = shuffled[split:]
            if first.numel() < 3 or second.numel() < 3:
                raise RuntimeError(
                    f"class {class_id} support/query folds "
                    f"require at least three samples"
                )
            support.append(first)
            query.append(second)

        return (
            torch.cat(support),
            torch.cat(query),
        )

    @torch.no_grad()
    def _temporary_rows_from_support(
        self,
        payload: Mapping[str, Tensor],
        support_index: Tensor,
        class_order_global: Sequence[int],
    ) -> Dict[int, Dict[str, Tensor]]:
        rows = self.model.build_geometry_rows(
            class_ids=class_order_global,
            features=payload["joint_feature"].index_select(
                0,
                support_index,
            ).detach(),
            raw_center_spectra=payload[
                "raw_center_spectrum"
            ].index_select(
                0,
                support_index,
            ).detach(),
            labels=payload["labels"].index_select(
                0,
                support_index,
            ).detach(),
            sample_weights=None,
        )
        return {
            int(class_id): {
                key: (
                    value.detach()
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in row.items()
            }
            for class_id, row in rows.items()
        }

    def _one_way_crossfit_episode(
        self,
        *,
        payload: Mapping[str, Tensor],
        support_index: Tensor,
        query_index: Tensor,
        class_order_global: Sequence[int],
        geometry_weight: float,
    ) -> Dict[str, Tensor]:
        rows = self._temporary_rows_from_support(
            payload,
            support_index,
            class_order_global,
        )
        pair_risk = (
            self.model.geometry_bank
            .pair_risk_matrix(
                class_order_global,
                rows=rows,
            )
        )

        query_features = payload[
            "joint_feature"
        ].index_select(0, query_index)
        query_labels = payload["labels"].index_select(
            0,
            query_index,
        )
        scored = self.model.compute_logits_from_features(
            query_features,
            class_ids=class_order_global,
            temporary_rows=rows,
            targets=query_labels,
            targets_are_global=True,
            return_parts=True,
        )
        base_logits = self.model.base_ce_logits(
            query_features
        )

        objective = base_training_objective(
            base_logits,
            query_labels,
            class_order_global,
            query_energy=scored["energy"],
            query_targets_global=query_labels,
            pair_risk=pair_risk,
            geometry_weight=geometry_weight,
            base_margin=self._base_margin(),
            risk_strength=self._risk_strength(),
            geometry_temperature=
                self._geometry_temperature(),
            maximum_margin=self._maximum_margin(),
            label_smoothing=self._cfg_float(
                "label_smoothing",
                default=0.0,
            ),
            class_balanced=True,
            return_parts=True,
        )
        assert isinstance(objective, Mapping)

        geometry = (
            risk_guided_all_rival_geometry_loss(
                scored["energy"],
                scored["class_ids"],
                query_labels,
                pair_risk=pair_risk,
                base_margin=self._base_margin(),
                risk_strength=self._risk_strength(),
                temperature=
                    self._geometry_temperature(),
                maximum_margin=
                    self._maximum_margin(),
                class_balanced=True,
                return_parts=True,
            )
        )
        assert isinstance(geometry, Mapping)

        return {
            "total": objective["total"],
            "ce": objective["ce"],
            "geometry": objective["geometry"],
            "head_accuracy":
                objective["ce_accuracy"],
            "geometry_accuracy":
                geometry["accuracy"],
            "mean_gap":
                geometry["mean_gap"],
            "q05_gap":
                geometry["q05_gap"],
            "minimum_gap":
                geometry["minimum_gap"],
            "classification_violation_rate":
                geometry[
                    "classification_violation_rate"
                ],
            "margin_violation_rate":
                geometry["margin_violation_rate"],
            "mean_nearest_pair_risk":
                geometry[
                    "nearest_pair_risk"
                ].mean().detach(),
            "mean_required_margin":
                geometry[
                    "nearest_required_margin"
                ].mean().detach(),
        }

    @staticmethod
    def _average_episode_results(
        first: Mapping[str, Tensor],
        second: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        if set(first) != set(second):
            raise RuntimeError(
                "bidirectional episode metrics differ"
            )
        return {
            key: 0.5 * (
                first[key] + second[key]
            )
            for key in first
        }

    def _cross_fitted_episode(
        self,
        payload: Mapping[str, Tensor],
        *,
        step: int,
        epoch: int,
        geometry_weight: float,
    ) -> Dict[str, Tensor]:
        class_order_global = [
            int(value)
            for value in torch.unique(
                payload["labels"],
                sorted=True,
            ).detach().cpu().tolist()
        ]
        support, query = self._crossfit_indices(
            payload["labels"],
            class_order_global,
            step=step,
            epoch=epoch,
        )
        first = self._one_way_crossfit_episode(
            payload=payload,
            support_index=support,
            query_index=query,
            class_order_global=
                class_order_global,
            geometry_weight=geometry_weight,
        )
        second = self._one_way_crossfit_episode(
            payload=payload,
            support_index=query,
            query_index=support,
            class_order_global=
                class_order_global,
            geometry_weight=geometry_weight,
        )
        return self._average_episode_results(
            first,
            second,
        )

    def _ce_only_episode(
        self,
        payload: Mapping[str, Tensor],
        phase_class_ids: Sequence[int],
    ) -> Dict[str, Tensor]:
        logits = self.model.base_ce_logits(
            payload["joint_feature"]
        )
        ce = base_ce_warmup_objective(
            logits,
            payload["labels"],
            phase_class_ids,
            label_smoothing=self._cfg_float(
                "label_smoothing",
                default=0.0,
            ),
            class_balanced=True,
            return_parts=True,
        )
        assert isinstance(ce, Mapping)
        zero = ce["total"] * 0.0
        return {
            "total": ce["total"],
            "ce": ce["total"],
            "geometry": zero,
            "head_accuracy": ce["accuracy"],
            "geometry_accuracy": zero.detach(),
            "mean_gap": zero.detach(),
            "q05_gap": zero.detach(),
            "minimum_gap": zero.detach(),
            "classification_violation_rate":
                zero.detach(),
            "margin_violation_rate":
                zero.detach(),
            "mean_nearest_pair_risk":
                zero.detach(),
            "mean_required_margin":
                zero.detach(),
        }

    # ------------------------------------------------------------------
    # Priors and pair-risk contract
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _initialize_geometry_contract(
        self,
        train_payload: Mapping[str, Tensor],
        class_ids: Sequence[int],
    ) -> Dict[str, Any]:
        bank = self.model.geometry_bank
        if bool(bank.global_priors_ready.item()):
            bank.assert_global_priors_ready()
            if not bool(
                bank.global_priors_frozen.item()
            ):
                raise RuntimeError(
                    "existing global priors are not frozen"
                )
            priors = {
                "reused": True,
                "sample_count": int(
                    train_payload["labels"].numel()
                ),
            }
        else:
            priors = self.model.fit_global_priors(
                train_payload["joint_feature"],
                train_payload[
                    "raw_center_spectrum"
                ],
                freeze=True,
                overwrite=False,
            )

        full_index = torch.arange(
            train_payload["labels"].numel(),
            device=train_payload["labels"].device,
        )
        rows = self._temporary_rows_from_support(
            train_payload,
            full_index,
            class_ids,
        )

        if bool(
            bank.overlap_temperatures_ready.item()
        ):
            if not bool(
                bank.overlap_temperatures_frozen.item()
            ):
                raise RuntimeError(
                    "existing overlap temperatures are not frozen"
                )
            temperatures = {
                "reused": True,
                "spectral_temperature": float(
                    bank.spectral_overlap_temperature.item()
                ),
                "geometry_temperature": float(
                    bank.geometry_overlap_temperature.item()
                ),
            }
        else:
            # The revised GeometryBank accepts detached provisional rows here.
            # No persistent class row is committed.
            temperatures = (
                bank.fit_overlap_temperatures(
                    class_ids,
                    rows=rows,
                    freeze=True,
                    overwrite=False,
                )
            )

        if bool(bank.valid_mask().any().item()):
            raise RuntimeError(
                "global geometry contract initialization "
                "must not commit class rows"
            )
        return {
            "global_priors": dict(priors),
            "overlap_temperatures":
                dict(temperatures),
            "provisional_rows_committed": False,
        }

    # ------------------------------------------------------------------
    # Epoch training
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        trainable_parameters: Sequence[nn.Parameter],
        phase_class_ids: Sequence[int],
        *,
        epoch: int,
        geometry_weight: float,
    ) -> Dict[str, float]:
        if bool(
            self.model.geometry_bank
            .valid_mask().any().item()
        ):
            raise RuntimeError(
                "persistent class rows must remain empty "
                "during base optimization"
            )

        self.model.train(True)
        if self.model.base_ce_head is None:
            raise RuntimeError(
                "base CE head is absent"
            )
        self.model.base_ce_head.train(True)

        sampler = getattr(
            loader,
            "batch_sampler",
            None,
        )
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        metric_names = (
            "total",
            "ce",
            "geometry",
            "head_accuracy",
            "geometry_accuracy",
            "mean_gap",
            "q05_gap",
            "minimum_gap",
            "classification_violation_rate",
            "margin_violation_rate",
            "mean_nearest_pair_risk",
            "mean_required_margin",
        )
        sums = {
            name: 0.0
            for name in metric_names
        }
        gradient_norm_sum = 0.0
        steps = 0

        for step, batch in enumerate(loader):
            payload = self._extract_batch_payload(
                batch,
                require_grad=True,
            )
            self._assert_geometry_batch(
                payload["labels"]
            )
            optimizer.zero_grad(
                set_to_none=True
            )

            result = (
                self._ce_only_episode(
                    payload,
                    phase_class_ids,
                )
                if geometry_weight <= 0.0
                else self._cross_fitted_episode(
                    payload,
                    step=step,
                    epoch=epoch,
                    geometry_weight=geometry_weight,
                )
            )
            loss = result["total"]
            if (
                loss.dim() != 0
                or not torch.isfinite(loss)
            ):
                raise RuntimeError(
                    "base loss is not a finite scalar"
                )
            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    list(trainable_parameters),
                    self._cfg_float(
                        "grad_clip_base",
                        default=5.0,
                    ),
                )
            )
            if not torch.isfinite(
                torch.as_tensor(gradient_norm)
            ):
                raise RuntimeError(
                    "base gradient norm is NaN/Inf"
                )
            optimizer.step()

            for name in metric_names:
                sums[name] += float(
                    result[name]
                    .detach()
                    .float()
                    .mean()
                    .cpu()
                    .item()
                )
            gradient_norm_sum += float(
                torch.as_tensor(
                    gradient_norm
                ).detach().cpu().item()
            )
            steps += 1

        if steps == 0:
            raise RuntimeError(
                "base training loader is empty"
            )
        result = {
            name: value / steps
            for name, value in sums.items()
        }
        result["gradient_norm"] = (
            gradient_norm_sum / steps
        )
        return result

    # ------------------------------------------------------------------
    # Geometry scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _full_temporary_rows(
        self,
        train_payload: Mapping[str, Tensor],
        class_ids: Sequence[int],
    ) -> Dict[int, Dict[str, Tensor]]:
        index = torch.arange(
            train_payload["labels"].numel(),
            device=train_payload["labels"].device,
        )
        return self._temporary_rows_from_support(
            train_payload,
            index,
            class_ids,
        )

    @torch.no_grad()
    def _score_with_rows(
        self,
        payload: Mapping[str, Tensor],
        class_ids: Sequence[int],
        rows: Mapping[int, Mapping[str, Any]],
        *,
        mode: str = "factor_geometry",
    ) -> Dict[str, Any]:
        scored = self.model.compute_logits_from_features(
            payload["joint_feature"],
            class_ids=class_ids,
            temporary_rows=rows,
            targets=payload["labels"],
            targets_are_global=True,
            mode=mode,
            return_parts=True,
            return_diagnostics=True,
        )
        diagnostics = dict(
            scored["diagnostics"]
        )

        pair_risk = (
            self.model.geometry_bank
            .pair_risk_matrix(
                class_ids,
                rows=rows,
            )
        )
        geometry = (
            risk_guided_all_rival_geometry_loss(
                scored["energy"],
                scored["class_ids"],
                payload["labels"],
                pair_risk=pair_risk,
                base_margin=self._base_margin(),
                risk_strength=self._risk_strength(),
                temperature=
                    self._geometry_temperature(),
                maximum_margin=
                    self._maximum_margin(),
                class_balanced=True,
                return_parts=True,
            )
        )
        assert isinstance(geometry, Mapping)

        class_tensor = scored[
            "class_ids"
        ]
        predicted_global = (
            class_tensor.index_select(
                0,
                scored["energy"].argmin(dim=1),
            )
        )
        per_class_accuracy: Dict[int, float] = {}
        per_class_correct: Dict[int, int] = {}
        per_class_total: Dict[int, int] = {}
        for class_id in class_ids:
            selected = payload[
                "labels"
            ].eq(int(class_id))
            total = int(
                selected.sum().item()
            )
            correct = int(
                predicted_global[selected]
                .eq(payload["labels"][selected])
                .sum()
                .item()
            )
            per_class_total[int(class_id)] = total
            per_class_correct[int(class_id)] = correct
            per_class_accuracy[int(class_id)] = (
                100.0 * correct / max(total, 1)
            )

        minimum_class = min(
            class_ids,
            key=lambda class_id: (
                per_class_accuracy[int(class_id)],
                int(class_id),
            ),
        )

        invasion = (
            pairwise_directional_invasion_matrix(
                scored["energy"],
                scored["class_ids"],
                payload["labels"],
            )
        )
        risk_confusion = (
            pair_risk_confusion_statistics(
                pair_risk,
                invasion["invasion_matrix"],
            )
        )

        return {
            "mode": mode,
            "accuracy": 100.0 * float(
                predicted_global
                .eq(payload["labels"])
                .float()
                .mean()
                .item()
            ),
            "mean_gap": float(
                geometry["mean_gap"].item()
            ),
            "q05_gap": float(
                geometry["q05_gap"].item()
            ),
            "minimum_gap": float(
                geometry["minimum_gap"].item()
            ),
            "classification_violation_rate":
                float(
                    geometry[
                        "classification_violation_rate"
                    ].item()
                ),
            "margin_violation_rate":
                float(
                    geometry[
                        "margin_violation_rate"
                    ].item()
                ),
            "minimum_per_class_accuracy":
                float(
                    per_class_accuracy[
                        int(minimum_class)
                    ]
                ),
            "minimum_accuracy_class_id":
                int(minimum_class),
            "per_class_accuracy":
                per_class_accuracy,
            "per_class_correct":
                per_class_correct,
            "per_class_total":
                per_class_total,
            "maximum_directional_invasion":
                float(
                    invasion[
                        "maximum_directional_invasion"
                    ].item()
                ),
            "mean_directional_invasion":
                float(
                    invasion[
                        "mean_directional_invasion"
                    ].item()
                ),
            "pearson_risk_invasion":
                float(
                    risk_confusion[
                        "pearson_risk_invasion"
                    ].item()
                ),
            "diagnostics": diagnostics,
            "energy":
                scored["energy"].detach(),
            "class_ids":
                scored["class_ids"].detach(),
            "targets":
                payload["labels"].detach(),
            "pair_risk":
                pair_risk.detach(),
            "invasion_matrix":
                invasion[
                    "invasion_matrix"
                ].detach(),
            "gap_q05_matrix":
                invasion[
                    "gap_q05_matrix"
                ].detach(),
        }

    @torch.no_grad()
    def _score_committed_payload(
        self,
        payload: Mapping[str, Tensor],
        class_ids: Sequence[int],
        *,
        mode: str = "factor_geometry",
    ) -> Dict[str, Any]:
        rows = {
            int(class_id):
                self.model.geometry_bank
                .get_class_row(
                    int(class_id),
                    clone=False,
                )
            for class_id in class_ids
        }
        return self._score_with_rows(
            payload,
            class_ids,
            rows,
            mode=mode,
        )

    @torch.no_grad()
    def _evaluate_out_of_fold(
        self,
        payload: Mapping[str, Tensor],
        class_ids: Sequence[int],
    ) -> Dict[str, Any]:
        ids = _unique_ids(class_ids)
        labels = payload["labels"]
        configured_folds = self._cfg_int(
            "base_oof_folds",
            default=3,
        )
        if configured_folds < 2:
            raise RuntimeError(
                "base_oof_folds must be at least two"
            )

        folds_by_class: Dict[
            int,
            List[Tensor],
        ] = {}
        for class_id in ids:
            indices = torch.nonzero(
                labels.eq(class_id),
                as_tuple=False,
            ).flatten()
            if indices.numel() < 6:
                raise RuntimeError(
                    f"class {class_id} has "
                    f"{indices.numel()} samples; "
                    "OOF requires at least six"
                )
            fold_count = min(
                configured_folds,
                max(
                    2,
                    int(indices.numel() // 3),
                ),
            )
            generator = torch.Generator().manual_seed(
                self._cfg_int(
                    "seed",
                    default=0,
                )
                + 9176 * class_id
            )
            order = torch.randperm(
                indices.numel(),
                generator=generator,
            ).to(indices.device)
            folds = list(
                torch.tensor_split(
                    indices.index_select(0, order),
                    fold_count,
                )
            )
            if any(
                fold.numel() == 0
                for fold in folds
            ):
                raise RuntimeError(
                    f"class {class_id} cannot populate OOF folds"
                )
            folds_by_class[class_id] = folds

        fold_count = min(
            len(folds)
            for folds in folds_by_class.values()
        )
        modes = (
            "factor_geometry",
            "quadratic_only",
            "diagonal_geometry",
            "prototype",
        )
        energy_by_mode: Dict[
            str,
            List[Tensor],
        ] = {
            mode: []
            for mode in modes
        }
        target_parts: List[Tensor] = []
        pair_risk_parts: List[Tensor] = []

        for fold_index in range(fold_count):
            support_parts: List[Tensor] = []
            query_parts: List[Tensor] = []
            for class_id in ids:
                folds = folds_by_class[class_id]
                query_parts.append(
                    folds[fold_index]
                )
                support_parts.append(
                    torch.cat(
                        [
                            fold
                            for index, fold
                            in enumerate(folds)
                            if index != fold_index
                        ]
                    )
                )

            support_index = torch.cat(
                support_parts
            )
            query_index = torch.cat(
                query_parts
            )
            rows = self._temporary_rows_from_support(
                payload,
                support_index,
                ids,
            )
            query_features = payload[
                "joint_feature"
            ].index_select(0, query_index)
            query_labels = labels.index_select(
                0,
                query_index,
            )
            pair_risk = (
                self.model.geometry_bank
                .pair_risk_matrix(
                    ids,
                    rows=rows,
                )
            )
            pair_risk_parts.append(
                pair_risk
            )
            for mode in modes:
                scored = (
                    self.model
                    .compute_logits_from_features(
                        query_features,
                        class_ids=ids,
                        temporary_rows=rows,
                        mode=mode,
                        return_parts=True,
                    )
                )
                energy_by_mode[mode].append(
                    scored["energy"].detach()
                )
            target_parts.append(
                query_labels
            )

        targets = torch.cat(
            target_parts,
            dim=0,
        )
        if targets.numel() != labels.numel():
            raise RuntimeError(
                "OOF did not evaluate every base sample exactly once"
            )

        class_tensor = torch.tensor(
            ids,
            device=targets.device,
            dtype=torch.long,
        )
        results: Dict[str, Any] = {
            "protocol":
                "stratified_out_of_fold_factor_geometry",
            "folds": fold_count,
            "classification_factorization":
                _CLASSIFICATION_FACTORIZATION,
        }
        for mode in modes:
            energy = torch.cat(
                energy_by_mode[mode],
                dim=0,
            )
            predicted_global = (
                class_tensor.index_select(
                    0,
                    energy.argmin(dim=1),
                )
            )
            results[mode] = {
                "accuracy": 100.0 * float(
                    predicted_global
                    .eq(targets)
                    .float()
                    .mean()
                    .item()
                )
            }
            if mode == "factor_geometry":
                mean_pair_risk = torch.stack(
                    pair_risk_parts,
                ).mean(dim=0)
                geometry = (
                    risk_guided_all_rival_geometry_loss(
                        energy,
                        class_tensor,
                        targets,
                        pair_risk=
                            mean_pair_risk,
                        base_margin=
                            self._base_margin(),
                        risk_strength=
                            self._risk_strength(),
                        temperature=
                            self._geometry_temperature(),
                        maximum_margin=
                            self._maximum_margin(),
                        class_balanced=True,
                        return_parts=True,
                    )
                )
                assert isinstance(
                    geometry,
                    Mapping,
                )
                invasion = (
                    pairwise_directional_invasion_matrix(
                        energy,
                        class_tensor,
                        targets,
                    )
                )
                risk_stats = (
                    pair_risk_confusion_statistics(
                        mean_pair_risk,
                        invasion[
                            "invasion_matrix"
                        ],
                    )
                )

                per_class: Dict[int, float] = {}
                for class_id in ids:
                    selected = targets.eq(class_id)
                    per_class[class_id] = (
                        100.0
                        * float(
                            predicted_global[selected]
                            .eq(targets[selected])
                            .float()
                            .mean()
                            .item()
                        )
                    )
                results[mode].update(
                    {
                        "mean_gap": float(
                            geometry[
                                "mean_gap"
                            ].item()
                        ),
                        "q05_gap": float(
                            geometry[
                                "q05_gap"
                            ].item()
                        ),
                        "minimum_gap": float(
                            geometry[
                                "minimum_gap"
                            ].item()
                        ),
                        "classification_violation_rate":
                            float(
                                geometry[
                                    "classification_violation_rate"
                                ].item()
                            ),
                        "margin_violation_rate":
                            float(
                                geometry[
                                    "margin_violation_rate"
                                ].item()
                            ),
                        "minimum_per_class_accuracy":
                            min(per_class.values()),
                        "per_class_accuracy":
                            per_class,
                        "maximum_directional_invasion":
                            float(
                                invasion[
                                    "maximum_directional_invasion"
                                ].item()
                            ),
                        "mean_directional_invasion":
                            float(
                                invasion[
                                    "mean_directional_invasion"
                                ].item()
                            ),
                        "pearson_risk_invasion":
                            float(
                                risk_stats[
                                    "pearson_risk_invasion"
                                ].item()
                            ),
                        "pair_risk":
                            mean_pair_risk.detach(),
                    }
                )

        results["factor_minus_prototype"] = (
            results["factor_geometry"][
                "accuracy"
            ]
            - results["prototype"][
                "accuracy"
            ]
        )
        results["factor_minus_diagonal"] = (
            results["factor_geometry"][
                "accuracy"
            ]
            - results["diagonal_geometry"][
                "accuracy"
            ]
        )
        return results

    @torch.no_grad()
    def _evaluate_sampling_diagnostic(
        self,
        class_ids: Sequence[int],
    ) -> Dict[str, Any]:
        samples = (
            self.model.geometry_bank
            .sample_geometry(
                class_ids,
                self._cfg_int(
                    "base_diagnostic_samples_per_class",
                    default=32,
                ),
            )
        )
        scored = self.model.compute_logits_from_features(
            samples["features"],
            class_ids=class_ids,
            targets=samples["labels"],
            targets_are_global=True,
            return_parts=True,
            return_diagnostics=True,
        )
        return {
            "accuracy": 100.0
            * float(
                scored["diagnostics"][
                    "accuracy"
                ]
            ),
            "mean_margin": float(
                scored["diagnostics"][
                    "mean_margin"
                ]
            ),
            "violation_rate": float(
                scored["diagnostics"][
                    "violation_rate"
                ]
            ),
            "training_use": False,
            "persistent_samples": False,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _jsonify(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            return (
                tensor.item()
                if tensor.numel() == 1
                else tensor.tolist()
            )
        if isinstance(value, Mapping):
            return {
                str(key):
                    BasePhaseTrainer._jsonify(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                BasePhaseTrainer._jsonify(item)
                for item in value
            ]
        if isinstance(value, set):
            return [
                BasePhaseTrainer._jsonify(item)
                for item in sorted(
                    value,
                    key=str,
                )
            ]
        if (
            isinstance(
                value,
                (str, int, float, bool),
            )
            or value is None
        ):
            return value
        return str(value)

    def _output_dir(self) -> str:
        path = os.path.abspath(
            str(
                self._cfg_any(
                    ("save_dir",),
                    default="./outputs",
                )
            )
        )
        os.makedirs(
            path,
            exist_ok=True,
        )
        return path

    def _class_name(
        self,
        class_id: int,
    ) -> str:
        class_id = int(class_id)
        for owner in (
            getattr(self, "dataset", None),
            getattr(self, "args", None),
        ):
            if owner is None:
                continue
            for attribute in (
                "target_names",
                "class_names",
            ):
                source = getattr(
                    owner,
                    attribute,
                    None,
                )
                value = None
                if isinstance(source, Mapping):
                    value = source.get(
                        class_id,
                        source.get(str(class_id)),
                    )
                elif (
                    isinstance(source, Sequence)
                    and not isinstance(
                        source,
                        (str, bytes, bytearray),
                    )
                    and 0 <= class_id < len(source)
                ):
                    value = source[class_id]
                if (
                    value is not None
                    and str(value).strip()
                ):
                    return str(value).strip()
        return f"Class-{class_id}"

    def _build_final_report(
        self,
        class_ids: Sequence[int],
        *,
        train_metrics: Mapping[str, Any],
        validation_metrics: Mapping[str, Any],
        oof_metrics: Mapping[str, Any],
        sampling_metrics: Mapping[str, Any],
        model_handoff: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ids = _unique_ids(class_ids)
        admission = (
            self.model.geometry_admission_report(
                ids,
                maximum_reconstruction_error=
                    self._cfg_float(
                        "maximum_geometry_reconstruction_error",
                        default=0.75,
                    ),
                minimum_effective_dimension=
                    self._cfg_float(
                        "minimum_geometry_effective_dimension",
                        default=1.25,
                    ),
                require_statistics=True,
            )
        )

        factor_oof = oof_metrics[
            "factor_geometry"
        ]
        gates = {
            "structural_geometry_valid":
                bool(admission.get("ok", False)),
            "factor_beats_prototype":
                float(
                    oof_metrics[
                        "factor_minus_prototype"
                    ]
                ) >= self._cfg_float(
                    "base_min_factor_gain_over_prototype",
                    default=0.0,
                ),
            "factor_beats_diagonal":
                float(
                    oof_metrics[
                        "factor_minus_diagonal"
                    ]
                ) >= self._cfg_float(
                    "base_min_factor_gain_over_diagonal",
                    default=0.0,
                ),
            "minimum_class_accuracy":
                float(
                    factor_oof[
                        "minimum_per_class_accuracy"
                    ]
                ) >= self._cfg_float(
                    "base_admission_min_class_accuracy",
                    default=0.0,
                ),
            "classification_violation":
                float(
                    factor_oof[
                        "classification_violation_rate"
                    ]
                ) <= self._cfg_float(
                    "base_admission_max_classification_violation",
                    default=1.0,
                ),
        }
        enforced = self._cfg_bool(
            "base_admission_enforce",
            default=False,
        )
        failures = [
            name
            for name, passed in gates.items()
            if not bool(passed)
        ]

        bank = self.model.geometry_bank
        return {
            "phase": 0,
            "method":
                "transport-verified spectral-spatial factor geometry",
            "classification_factorization":
                _CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization":
                _SPECTRAL_RELATION,
            "class_ids": ids,
            "class_names": {
                class_id:
                    self._class_name(class_id)
                for class_id in ids
            },
            "feature_contract": {
                "token_dim": int(
                    self.model.backbone.token_dim
                ),
                "spectral_dim": int(
                    self.model.spectral_dim
                ),
                "spatial_dim": int(
                    self.model.spatial_dim
                ),
                "feature_dim": int(
                    self.model.feature_dim
                ),
                "joint_feature":
                    "exact_[z_s;z_p]",
                "final_feature_normalization":
                    False,
            },
            "training_contract": {
                "temporary_linear_head": True,
                "bidirectional_crossfit": True,
                "query_rows_detached": True,
                "query_loss_weighting": False,
                "persistent_rows_during_epochs":
                    False,
                "trainable_transport": False,
                "geometry_replay_training":
                    False,
            },
            "global_priors": {
                "ready": bool(
                    bank.global_priors_ready.item()
                ),
                "frozen": bool(
                    bank.global_priors_frozen.item()
                ),
                "spectral_residual_prior":
                    float(
                        bank.global_residual_prior_spectral.item()
                    ),
                "spatial_residual_prior":
                    float(
                        bank.global_residual_prior_spatial.item()
                    ),
            },
            "pair_risk_contract": {
                "temperatures_ready": bool(
                    bank.overlap_temperatures_ready.item()
                ),
                "temperatures_frozen": bool(
                    bank.overlap_temperatures_frozen.item()
                ),
                "spectral_temperature":
                    float(
                        bank.spectral_overlap_temperature.item()
                    ),
                "geometry_temperature":
                    float(
                        bank.geometry_overlap_temperature.item()
                    ),
                "used_for_inference": False,
                "used_for_training_margin": True,
            },
            "train": dict(train_metrics),
            "validation": dict(
                validation_metrics
            ),
            "out_of_fold": dict(
                oof_metrics
            ),
            "sampling_diagnostic": dict(
                sampling_metrics
            ),
            "geometry_admission":
                admission,
            "admission_enforced": enforced,
            "admission_gates": gates,
            "admission_passed":
                not failures,
            "admission_failures":
                failures,
            "memory_cost":
                bank.memory_cost_summary(),
            "model_handoff":
                dict(model_handoff),
            "best_epoch":
                history.get("best_epoch"),
        }

    def finalize_base_handoff(
        self,
        class_ids: Sequence[int],
        report: Mapping[str, Any],
        history: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ids = _unique_ids(class_ids)
        handoff = {
            "phase": 0,
            "method":
                "transport-verified spectral-spatial factor geometry",
            "classification_factorization":
                _CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization":
                _SPECTRAL_RELATION,
            "base_classes": ids,
            "token_dim": int(
                self.model.backbone.token_dim
            ),
            "spectral_dim": int(
                self.model.spectral_dim
            ),
            "spatial_dim": int(
                self.model.spatial_dim
            ),
            "feature_dim": int(
                self.model.feature_dim
            ),
            "geometry_bank_ready":
                bool(
                    self.model.geometry_bank
                    .valid_mask().any().item()
                ),
            "classifier_bound":
                bool(
                    self.model.classifier
                    .require_bound_contract
                ),
            "bank_contract_digest":
                self.model.classifier
                .bank_contract_digest(
                    self.model.geometry_bank
                ),
            "rows_digest":
                self.model.geometry_bank
                .rows_digest(ids),
            "admission_passed":
                bool(
                    report.get(
                        "admission_passed",
                        False,
                    )
                ),
            "feature_space_policy":
                "frozen_until_incremental_phase_policy_is_selected",
            "geometry_report":
                dict(report),
        }

        output = self._output_dir()
        paths = {
            "handoff_json_path":
                os.path.join(
                    output,
                    "phase_0_base_handoff.json",
                ),
            "handoff_pt_path":
                os.path.join(
                    output,
                    "phase_0_base_handoff.pt",
                ),
            "report_json_path":
                os.path.join(
                    output,
                    "phase_0_base_geometry_report.json",
                ),
            "history_json_path":
                os.path.join(
                    output,
                    "phase_0_base_history.json",
                ),
        }
        with open(
            paths["handoff_json_path"],
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                self._jsonify(handoff),
                stream,
                indent=2,
            )
        torch.save(
            self._jsonify(handoff),
            paths["handoff_pt_path"],
        )
        with open(
            paths["report_json_path"],
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                self._jsonify(report),
                stream,
                indent=2,
            )
        with open(
            paths["history_json_path"],
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                self._jsonify(history),
                stream,
                indent=2,
            )

        handoff.update(paths)
        self.model.base_handoff = handoff
        return handoff

    # ------------------------------------------------------------------
    # Main phase entry
    # ------------------------------------------------------------------

    def train_base_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        phase = int(phase)
        epochs = int(epochs)
        if phase != 0:
            raise ValueError(
                "train_base_phase() is phase-0 only"
            )
        if epochs <= 0:
            raise RuntimeError(
                "epochs must be positive"
            )
        if float(lr) <= 0.0:
            raise RuntimeError(
                "lr must be positive"
            )

        self.dataset.start_phase(phase)
        phase_class_ids = _unique_ids(
            self.dataset.phase_to_classes[phase],
            name="phase_class_ids",
        )
        self._validate_base_configuration(
            phase_class_ids
        )

        print(
            "==== Base Phase | "
            "Spectral-Spatial Factor Geometry ===="
        )
        self.model.set_base_mode()
        base_head = self._make_base_head(
            phase_class_ids
        )
        self._set_base_trainability(
            base_head
        )
        if bool(
            self.model.geometry_bank
            .valid_mask().any().item()
        ):
            raise RuntimeError(
                "base phase must begin with zero committed rows"
            )

        raw_train_loader = (
            self.dataset
            .get_phase_dataloader(
                phase,
                split="train",
                batch_size=int(batch_size),
                shuffle=False,
            )
        )
        train_loader = (
            self._build_geometry_balanced_loader(
                raw_train_loader,
                phase_class_ids,
            )
        )
        train_eval_loader = (
            self.dataset
            .get_phase_dataloader(
                phase,
                split="train",
                batch_size=int(batch_size),
                shuffle=False,
            )
        )
        val_loader = (
            self.dataset
            .get_cumulative_dataloader(
                phase,
                split="val",
                batch_size=int(batch_size),
                shuffle=False,
            )
        )

        trainable = (
            self._base_trainable_parameters()
        )
        optimizer = optim.AdamW(
            trainable,
            lr=float(lr),
            weight_decay=self._cfg_float(
                "weight_decay",
                default=1e-4,
            ),
        )
        scheduler: Optional[
            optim.lr_scheduler.LRScheduler
        ] = None
        if self._cfg_bool(
            "base_use_cosine_scheduler",
            default=True,
        ):
            scheduler = (
                optim.lr_scheduler
                .CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs),
                    eta_min=float(lr)
                    * self._cfg_float(
                        "base_min_lr_ratio",
                        default=0.05,
                    ),
                )
            )

        history: Dict[str, Any] = {
            "architecture":
                self.model.architecture_summary(),
            "train": [],
            "validation": [],
            "persistent_rows_during_training":
                [],
            "geometry_contract": None,
            "best_epoch": None,
            "best_validation_accuracy": None,
            "best_validation_q05_gap": None,
        }

        geometry_contract_ready = False
        best_state: Optional[
            Dict[str, Any]
        ] = None
        best_epoch = -1
        best_accuracy = float("-inf")
        best_q05 = float("-inf")
        best_min_class = float("-inf")
        stale_evaluations = 0
        patience = self._cfg_int(
            "base_early_stop_patience",
            default=0,
        )
        eval_every = max(
            1,
            self._cfg_int(
                "base_eval_every",
                default=1,
            ),
        )

        print(
            "[Base contract] CE warm-up -> frozen aggregate priors -> "
            "frozen pair-risk temperatures -> bidirectional support/query "
            "factor geometry -> atomic final commit."
        )
        print(
            "[Base memory] No persistent class row is committed during "
            "optimization."
        )

        for epoch in range(epochs):
            geometry_weight = (
                self._geometry_weight_for_epoch(
                    epoch
                )
            )

            if (
                geometry_weight > 0.0
                and not geometry_contract_ready
            ):
                contract_payload = (
                    self._collect_payload(
                        train_eval_loader
                    )
                )
                history["geometry_contract"] = (
                    self._initialize_geometry_contract(
                        contract_payload,
                        phase_class_ids,
                    )
                )
                geometry_contract_ready = True
                print(
                    "[Base geometry contract] "
                    "global priors and pair-risk temperatures "
                    "fitted and frozen; committed rows=0"
                )

            train_stats = self._train_epoch(
                train_loader,
                optimizer,
                trainable,
                phase_class_ids,
                epoch=epoch,
                geometry_weight=
                    geometry_weight,
            )
            if scheduler is not None:
                scheduler.step()

            validation_metrics: Dict[
                str,
                Any,
            ] = {}
            should_evaluate = (
                (epoch + 1) % eval_every == 0
                or epoch == epochs - 1
            )
            if (
                should_evaluate
                and geometry_contract_ready
            ):
                train_payload = (
                    self._collect_payload(
                        train_eval_loader
                    )
                )
                val_payload = (
                    self._collect_payload(
                        val_loader
                    )
                )
                rows = self._full_temporary_rows(
                    train_payload,
                    phase_class_ids,
                )
                validation_metrics = (
                    self._score_with_rows(
                        val_payload,
                        phase_class_ids,
                        rows,
                    )
                )
                train_geometry = (
                    self._score_with_rows(
                        train_payload,
                        phase_class_ids,
                        rows,
                    )
                )
                validation_metrics[
                    "train_geometry_accuracy"
                ] = train_geometry["accuracy"]

                accuracy = float(
                    validation_metrics[
                        "accuracy"
                    ]
                )
                q05 = float(
                    validation_metrics[
                        "q05_gap"
                    ]
                )
                min_class = float(
                    validation_metrics[
                        "minimum_per_class_accuracy"
                    ]
                )
                improved = (
                    accuracy > best_accuracy + 1e-12
                    or (
                        abs(
                            accuracy - best_accuracy
                        ) <= 1e-12
                        and q05 > best_q05 + 1e-12
                    )
                    or (
                        abs(
                            accuracy - best_accuracy
                        ) <= 1e-12
                        and abs(
                            q05 - best_q05
                        ) <= 1e-12
                        and min_class > best_min_class
                    )
                )
                if improved:
                    best_accuracy = accuracy
                    best_q05 = q05
                    best_min_class = min_class
                    best_epoch = epoch
                    best_state = copy.deepcopy(
                        self.model.state_dict()
                    )
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1

            history["train"].append(
                {
                    "epoch": epoch + 1,
                    "geometry_weight":
                        geometry_weight,
                    "learning_rate":
                        float(
                            optimizer.param_groups[0][
                                "lr"
                            ]
                        ),
                    **train_stats,
                }
            )
            history["validation"].append(
                {
                    "epoch": epoch + 1,
                    **validation_metrics,
                }
            )
            committed_count = int(
                self.model.geometry_bank
                .valid_mask().sum().item()
            )
            history[
                "persistent_rows_during_training"
            ].append(committed_count)
            if committed_count != 0:
                raise RuntimeError(
                    "persistent class rows appeared during base optimization"
                )

            message = (
                f"[Base] Epoch {epoch + 1:03d}/{epochs} | "
                f"Loss={train_stats['total']:.4f} | "
                f"CE={train_stats['ce']:.4f} | "
                f"Geom={train_stats['geometry']:.4f} | "
                f"GeomW={geometry_weight:.3f} | "
                f"Head={100.0 * train_stats['head_accuracy']:.2f}%"
            )
            if geometry_weight > 0.0:
                message += (
                    f" | QueryGeom="
                    f"{100.0 * train_stats['geometry_accuracy']:.2f}%"
                    f" | GapQ05={train_stats['q05_gap']:.3f}"
                )
            if validation_metrics:
                message += (
                    f" | Val={validation_metrics['accuracy']:.2f}%"
                    f" | ValQ05={validation_metrics['q05_gap']:.3f}"
                    f" | ValMin="
                    f"{validation_metrics['minimum_per_class_accuracy']:.2f}%"
                )
            message += (
                f" | Rows={committed_count}"
            )
            print(message)

            if (
                patience > 0
                and geometry_contract_ready
                and stale_evaluations >= patience
            ):
                print(
                    "[Base early stop] no validation geometry improvement "
                    f"for {stale_evaluations} evaluations"
                )
                break

        # A CE-only experiment still needs a valid geometry contract before
        # final handoff.
        if not geometry_contract_ready:
            contract_payload = (
                self._collect_payload(
                    train_eval_loader
                )
            )
            history["geometry_contract"] = (
                self._initialize_geometry_contract(
                    contract_payload,
                    phase_class_ids,
                )
            )
            geometry_contract_ready = True
            rows = self._full_temporary_rows(
                contract_payload,
                phase_class_ids,
            )
            val_payload = self._collect_payload(
                val_loader
            )
            validation_metrics = (
                self._score_with_rows(
                    val_payload,
                    phase_class_ids,
                    rows,
                )
            )
            best_accuracy = float(
                validation_metrics["accuracy"]
            )
            best_q05 = float(
                validation_metrics["q05_gap"]
            )
            best_min_class = float(
                validation_metrics[
                    "minimum_per_class_accuracy"
                ]
            )
            best_epoch = epochs - 1
            best_state = copy.deepcopy(
                self.model.state_dict()
            )

        if best_state is None:
            raise RuntimeError(
                "base training produced no geometry-selected checkpoint"
            )
        if bool(
            self.model.geometry_bank
            .valid_mask().any().item()
        ):
            raise RuntimeError(
                "persistent rows appeared before final handoff"
            )

        self.model.load_state_dict(
            best_state,
            strict=True,
        )
        self.model.set_base_mode()
        history["best_epoch"] = best_epoch + 1
        history[
            "best_validation_accuracy"
        ] = best_accuracy
        history[
            "best_validation_q05_gap"
        ] = best_q05

        final_train_payload = (
            self._collect_payload(
                train_eval_loader
            )
        )
        final_oof = (
            self._evaluate_out_of_fold(
                final_train_payload,
                phase_class_ids,
            )
        )

        model_handoff = (
            self.model.finalize_base_geometry(
                base_class_ids=
                    phase_class_ids,
                features=final_train_payload[
                    "joint_feature"
                ],
                raw_center_spectra=
                    final_train_payload[
                        "raw_center_spectrum"
                    ],
                labels=final_train_payload[
                    "labels"
                ],
                sample_weights=None,
                update_statistics=True,
                # Temperatures were fitted before geometry shaping and
                # restored with the selected checkpoint.
                fit_overlap_temperatures=False,
                drop_ce_head=True,
            )
        )

        final_val_payload = (
            self._collect_payload(
                val_loader
            )
        )
        final_train_metrics = (
            self._score_committed_payload(
                final_train_payload,
                phase_class_ids,
            )
        )
        final_val_metrics = (
            self._score_committed_payload(
                final_val_payload,
                phase_class_ids,
            )
        )
        sampling_metrics = (
            self._evaluate_sampling_diagnostic(
                phase_class_ids
            )
        )

        final_report = (
            self._build_final_report(
                phase_class_ids,
                train_metrics=
                    final_train_metrics,
                validation_metrics=
                    final_val_metrics,
                oof_metrics=final_oof,
                sampling_metrics=
                    sampling_metrics,
                model_handoff=
                    model_handoff,
                history=history,
            )
        )

        if not bool(
            final_report[
                "geometry_admission"
            ].get("ok", False)
        ):
            raise RuntimeError(
                "final base GeometryBank is structurally invalid: "
                + "; ".join(
                    final_report[
                        "geometry_admission"
                    ].get("errors", [])
                )
            )
        if (
            self._cfg_bool(
                "base_admission_enforce",
                default=False,
            )
            and not bool(
                final_report[
                    "admission_passed"
                ]
            )
        ):
            raise RuntimeError(
                "final base admission failed: "
                f"{final_report['admission_failures']}"
            )

        history[
            "final_train_metrics"
        ] = final_train_metrics
        history[
            "final_validation_metrics"
        ] = final_val_metrics
        history[
            "final_oof_metrics"
        ] = final_oof
        history[
            "sampling_diagnostic"
        ] = sampling_metrics
        history[
            "base_geometry_report"
        ] = final_report

        pre_finalize_hook = getattr(
            self,
            "pre_finalize_base_hook",
            None,
        )
        if callable(pre_finalize_hook):
            hook_result = pre_finalize_hook(
                phase=0,
                model=self.model,
                dataset=self.dataset,
                report=final_report,
                history=history,
            )
            if hook_result is not None:
                history[
                    "pre_finalize_artifacts"
                ] = hook_result

        if hasattr(
            self.dataset,
            "finalize_phase",
        ):
            self.dataset.finalize_phase(phase)

        handoff = self.finalize_base_handoff(
            phase_class_ids,
            final_report,
            history,
        )
        history["base_handoff"] = handoff

        if hasattr(self, "save_checkpoint"):
            self.save_checkpoint(
                phase,
                history,
            )

        factor_oof = final_oof[
            "factor_geometry"
        ]
        print(
            "[Base handoff] "
            f"BestEpoch={best_epoch + 1} | "
            f"Val={final_val_metrics['accuracy']:.2f}% | "
            f"OOF={factor_oof['accuracy']:.2f}% | "
            f"Prototype={final_oof['prototype']['accuracy']:.2f}% | "
            f"Diagonal={final_oof['diagonal_geometry']['accuracy']:.2f}% | "
            f"MinClass={factor_oof['minimum_per_class_accuracy']:.2f}% | "
            f"Q05Gap={factor_oof['q05_gap']:.3f} | "
            f"Rows={int(self.model.geometry_bank.valid_mask().sum().item())} | "
            f"Bound={self.model.classifier.require_bound_contract}"
        )
        return history
