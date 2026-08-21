from __future__ import annotations

"""Parameter-free equal-rule classifier over pairwise decision geometry.

The classifier contains no class-specific trainable parameters.  Historical and
new classes are scored by the same geometry rule

    E_c(z) = - min_{j != c} s_cj(z),

and prediction is

    argmin_c E_c(z).

All class-pair mathematics remains owned by ``BoundaryGeometryBank``.  This
module only converts geometry energy to logits and a global class prediction.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from models.geometry_bank import BoundaryCandidate, BoundaryGeometryBank

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassifierOutput:
    """Equal-rule classifier output in ``class_ids`` column order."""

    logits: Tensor
    energy: Tensor
    class_ids: Tensor
    prediction: Tensor

    @property
    def inside_cell(self) -> Tensor:
        """Whether each sample lies inside each requested class decision cell."""
        return self.energy <= 0


class GeometryClassifier(nn.Module):
    """Parameter-free classifier using the same signed energy for every class."""

    @staticmethod
    def targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
        """Map global class IDs to classifier-column indices."""
        labels = torch.as_tensor(labels_global)
        ids = torch.as_tensor(class_ids)

        if labels.device != ids.device:
            raise ValueError("labels_global and class_ids must share a device")

        labels = labels.flatten()
        if labels.numel() == 0:
            raise ValueError("labels_global cannot be empty")
        if labels.dtype == torch.bool or labels.is_complex():
            raise ValueError("labels_global must contain integer class IDs")
        if torch.is_floating_point(labels):
            if not bool(torch.isfinite(labels).all()) or not bool(
                labels.eq(labels.round()).all()
            ):
                raise ValueError(
                    "labels_global must contain finite integer class IDs"
                )
        labels = labels.to(dtype=torch.long)

        if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("class_ids must be non-empty rank-one int64")
        if ids.unique().numel() != ids.numel() or bool((ids < 0).any()):
            raise ValueError("class_ids must be unique non-negative IDs")

        matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
        counts = matches.sum(dim=1)
        if not bool(counts.eq(1).all()):
            missing = (
                labels[counts.eq(0)]
                .unique()
                .detach()
                .cpu()
                .tolist()
            )
            raise ValueError(
                f"labels are outside classifier classes: {missing}"
            )
        return matches.to(torch.long).argmax(dim=1)

    def forward(
        self,
        coordinates: Tensor,
        *,
        geometry_bank: BoundaryGeometryBank,
        class_ids: Sequence[int],
        candidate: BoundaryCandidate | None = None,
    ) -> ClassifierOutput:
        if not isinstance(geometry_bank, BoundaryGeometryBank):
            raise TypeError("geometry_bank must be BoundaryGeometryBank")
        geometry_bank.validate_bank_state()
        if candidate is not None:
            if not isinstance(candidate, BoundaryCandidate):
                raise TypeError("candidate must be BoundaryCandidate or None")
            candidate.validate_state()

        score = geometry_bank.score(
            coordinates,
            class_ids=class_ids,
            candidate=candidate,
        )
        energy = score.energy
        ids = score.class_ids

        if energy.ndim != 2 or energy.size(0) == 0 or energy.size(1) == 0:
            raise RuntimeError("geometry energy must be non-empty [N,C]")
        if ids.shape != (energy.size(1),) or ids.dtype != torch.long:
            raise RuntimeError(
                "geometry class_ids are misaligned with energy columns"
            )
        if not bool(torch.isfinite(energy).all()):
            raise RuntimeError("geometry energy contains NaN/Inf")

        # This is intentionally exact.  No temperature, class-specific scale,
        # calibration vector, or task-dependent bias is introduced.
        logits = -energy
        prediction = ids.index_select(0, energy.argmin(dim=1))

        return ClassifierOutput(
            logits=logits,
            energy=energy,
            class_ids=ids,
            prediction=prediction,
        )


__all__ = ["ClassifierOutput", "GeometryClassifier"]
