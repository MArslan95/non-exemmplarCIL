from __future__ import annotations

"""Thin integration for one-space HSI features and pairwise decision geometry.

``NECILModel`` deliberately contains no continual-learning strategy.  It owns
only the three architectural components used in every phase:

    HSI backbone -> pairwise decision geometry -> equal-rule classifier.

The geometry bank is the persistent historical decision reference.  This model
therefore exposes thin, validated wrappers for the geometry operations required
by classification, pair-distribution separation, replay selection, and
historical boundary-response preservation.  Loss weighting, replay scheduling,
and phase optimization remain trainer responsibilities.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
from models.classifier import ClassifierOutput, GeometryClassifier
from models.geometry_bank import (
    BoundaryCandidate,
    BoundaryGeometryBank,
    ClassBoundaryResponse,
    PairwiseGeometryValues,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class NECILOutput:
    representation: BackboneOutput
    classification: ClassifierOutput


class NECILModel(nn.Module):
    """HSI backbone + persistent pairwise geometry + equal-rule classifier."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.backbone = HSIMambaFeatureExtractor(args)

        if not hasattr(self.backbone, "representation_dim"):
            raise RuntimeError("backbone must expose representation_dim")
        if hasattr(self.backbone, "coordinate_projection"):
            raise RuntimeError(
                "one-space backbone must not use a coordinate projection"
            )

        reference = next(self.backbone.parameters(), None)
        if reference is None:
            raise RuntimeError("backbone must contain trainable parameters")
        dtype = (
            reference.dtype
            if reference.dtype in (torch.float32, torch.float64)
            else torch.float32
        )

        self.geometry_bank = BoundaryGeometryBank(
            representation_dim=int(self.backbone.representation_dim),
            device=reference.device,
            dtype=dtype,
        )
        self.classifier = GeometryClassifier()
        self.validate_model_state()

    @property
    def device(self) -> torch.device:
        parameter = next(self.backbone.parameters(), None)
        if parameter is None:
            raise RuntimeError("backbone has no parameters")
        return parameter.device

    @property
    def dtype(self) -> torch.dtype:
        parameter = next(self.backbone.parameters(), None)
        if parameter is None:
            raise RuntimeError("backbone has no parameters")
        if parameter.dtype not in (torch.float32, torch.float64):
            raise RuntimeError(
                "backbone parameters must use float32 or float64"
            )
        return parameter.dtype

    @property
    def representation_dim(self) -> int:
        return int(self.backbone.representation_dim)

    @property
    def committed_class_ids(self) -> tuple[int, ...]:
        return tuple(
            int(v)
            for v in self.geometry_bank.class_ids.detach().cpu().tolist()
        )

    @property
    def spectral_normalization_fitted(self) -> bool:
        return bool(self.backbone.spectral_normalization_fitted)

    def _validate_component_contract(self) -> None:
        if self.geometry_bank.device != self.device:
            raise RuntimeError(
                "backbone and geometry bank are on different devices"
            )
        if self.geometry_bank.dtype != self.dtype:
            raise RuntimeError(
                "backbone and geometry bank use different floating dtypes"
            )
        if self.geometry_bank.representation_dim != self.representation_dim:
            raise RuntimeError(
                "backbone and geometry representation dimensions disagree"
            )

    @torch.no_grad()
    def fit_spectral_normalization(
        self,
        base_center_spectra: Tensor,
        *,
        overwrite: bool = False,
    ) -> None:
        """Fit the ordered-spectrum normalization exactly once before geometry."""
        if len(self.geometry_bank) != 0:
            raise RuntimeError(
                "spectral normalization cannot change after geometry commit"
            )
        self.backbone.fit_spectral_normalization(
            base_center_spectra,
            overwrite=overwrite,
        )
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "spectral normalization did not enter fitted state"
            )

    def encode(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        return_aux: bool = False,
    ) -> BackboneOutput:
        """Encode HSI input into the single canonical representation space."""
        self._validate_component_contract()
        output = self.backbone(
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )
        z = output.coordinates

        if (
            z.ndim != 2
            or z.size(0) == 0
            or z.size(1) != self.representation_dim
        ):
            raise RuntimeError(
                "backbone coordinates violate [N,representation_dim]"
            )
        if z.device != self.geometry_bank.device:
            raise RuntimeError(
                "backbone coordinates and geometry must share a device"
            )
        if z.dtype != self.geometry_bank.dtype:
            raise RuntimeError(
                "backbone coordinates and geometry must share a dtype"
            )
        if not bool(torch.isfinite(z).all()):
            raise RuntimeError("backbone coordinates contain NaN/Inf")
        return output

    def initialize_candidate(
        self,
        coordinates: Tensor,
        labels: Tensor,
        class_ids: Sequence[int],
    ) -> BoundaryCandidate:
        """Initialize exactly the new-new and old-new boundaries of a phase."""
        self._validate_component_contract()
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "fit spectral normalization before geometry initialization"
            )
        return self.geometry_bank.initialize_candidate(
            coordinates,
            labels,
            new_class_ids=class_ids,
        )

    @torch.no_grad()
    def commit_candidate(self, candidate: BoundaryCandidate) -> None:
        """Persist the exact learned current-phase pairwise boundaries."""
        self._validate_component_contract()
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "cannot commit geometry before spectral normalization"
            )
        self.geometry_bank.commit_candidate(candidate)
        self.validate_model_state()

    # ------------------------------------------------------------------
    # Geometry interface
    # ------------------------------------------------------------------
    def pair_values(
        self,
        coordinates: Tensor,
        *,
        pair_ids: Sequence[Sequence[int]] | Tensor,
        candidate: Optional[BoundaryCandidate] = None,
    ) -> PairwiseGeometryValues:
        """Evaluate explicit h_ab(z) values through the bank-owned geometry API."""
        self._validate_component_contract()
        return self.geometry_bank.pair_values(
            coordinates,
            pair_ids=pair_ids,
            candidate=candidate,
        )

    def class_boundary_response(
        self,
        coordinates: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> ClassBoundaryResponse:
        """Return class-incident oriented decision coordinates r_c(z)."""
        self._validate_component_contract()
        return self.geometry_bank.class_boundary_response(
            coordinates,
            labels,
            class_ids=class_ids,
            candidate=candidate,
        )

    def true_pair_margins(
        self,
        coordinates: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> Tensor:
        """Return all oriented class-vs-rival margins for each labeled sample."""
        self._validate_component_contract()
        return self.geometry_bank.true_pair_margins(
            coordinates,
            labels,
            class_ids=class_ids,
            candidate=candidate,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify_coordinates(
        self,
        coordinates: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> ClassifierOutput:
        self._validate_component_contract()
        return self.classifier(
            coordinates,
            geometry_bank=self.geometry_bank,
            class_ids=class_ids,
            candidate=candidate,
        )

    def forward(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
        return_aux: bool = False,
    ) -> NECILOutput:
        representation = self.encode(
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )
        classification = self.classify_coordinates(
            representation.coordinates,
            class_ids=class_ids,
            candidate=candidate,
        )
        return NECILOutput(
            representation=representation,
            classification=classification,
        )

    def validate_geometry(self) -> bool:
        return bool(self.geometry_bank.validate_bank_state())

    def validate_model_state(self) -> bool:
        if hasattr(self.backbone, "coordinate_projection"):
            raise RuntimeError(
                "one-space backbone must not contain coordinate_projection"
            )
        self._validate_component_contract()

        # The classifier must remain a pure decision rule.  Any learned
        # class-specific parameter here would reintroduce an expanding
        # classifier head and undermine the equal-rule bias-control design.
        if any(True for _ in self.classifier.parameters()):
            raise RuntimeError(
                "GeometryClassifier must remain parameter-free"
            )

        if len(self.geometry_bank) and not self.spectral_normalization_fitted:
            raise RuntimeError(
                "committed geometry requires fitted spectral normalization"
            )

        self.validate_geometry()
        return True


__all__ = ["NECILModel", "NECILOutput"]

















# from __future__ import annotations

# """Thin integration for one-space HSI features and pairwise decision geometry."""

# from dataclasses import dataclass
# from typing import Any, Optional, Sequence

# import torch
# import torch.nn as nn

# from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
# from models.classifier import ClassifierOutput, GeometryClassifier
# from models.geometry_bank import BoundaryCandidate, BoundaryGeometryBank

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class NECILOutput:
#     representation: BackboneOutput
#     classification: ClassifierOutput


# class NECILModel(nn.Module):
#     """HSI backbone + persistent pairwise boundary geometry + equal-rule classifier."""

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.backbone = HSIMambaFeatureExtractor(args)
#         if not hasattr(self.backbone, "representation_dim"):
#             raise RuntimeError("backbone must expose representation_dim")
#         if hasattr(self.backbone, "coordinate_projection"):
#             raise RuntimeError("one-space backbone must not use a coordinate projection")

#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone must contain trainable parameters")
#         dtype = reference.dtype if reference.dtype in (torch.float32, torch.float64) else torch.float32

#         self.geometry_bank = BoundaryGeometryBank(
#             representation_dim=int(self.backbone.representation_dim),
#             device=reference.device,
#             dtype=dtype,
#         )
#         self.classifier = GeometryClassifier()
#         self.validate_model_state()

#     @property
#     def device(self) -> torch.device:
#         parameter = next(self.backbone.parameters(), None)
#         if parameter is None:
#             raise RuntimeError("backbone has no parameters")
#         return parameter.device

#     @property
#     def representation_dim(self) -> int:
#         return int(self.backbone.representation_dim)

#     @property
#     def committed_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(v) for v in self.geometry_bank.class_ids.detach().cpu().tolist())

#     @property
#     def spectral_normalization_fitted(self) -> bool:
#         return bool(self.backbone.spectral_normalization_fitted)

#     def _validate_component_devices(self) -> None:
#         if self.geometry_bank.device != self.device:
#             raise RuntimeError("backbone and geometry bank are on different devices")

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if len(self.geometry_bank) != 0:
#             raise RuntimeError("spectral normalization cannot change after geometry commit")
#         self.backbone.fit_spectral_normalization(base_center_spectra, overwrite=overwrite)
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization did not enter fitted state")

#     def encode(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         self._validate_component_devices()
#         output = self.backbone(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         z = output.coordinates
#         if z.ndim != 2 or z.size(0) == 0 or z.size(1) != self.representation_dim:
#             raise RuntimeError("backbone coordinates violate [N,representation_dim]")
#         if z.device != self.geometry_bank.device or z.dtype != self.geometry_bank.dtype:
#             raise RuntimeError("backbone coordinates and geometry must share device/dtype")
#         if not bool(torch.isfinite(z).all()):
#             raise RuntimeError("backbone coordinates contain NaN/Inf")
#         return output

#     def initialize_candidate(
#         self,
#         coordinates: Tensor,
#         labels: Tensor,
#         class_ids: Sequence[int],
#     ) -> BoundaryCandidate:
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("fit spectral normalization before geometry initialization")
#         return self.geometry_bank.initialize_candidate(
#             coordinates,
#             labels,
#             new_class_ids=class_ids,
#         )

#     @torch.no_grad()
#     def commit_candidate(self, candidate: BoundaryCandidate) -> None:
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("cannot commit geometry before spectral normalization")
#         self.geometry_bank.commit_candidate(candidate)
#         self.validate_model_state()

#     def classify_coordinates(
#         self,
#         coordinates: Tensor,
#         *,
#         class_ids: Sequence[int],
#         candidate: Optional[BoundaryCandidate] = None,
#     ) -> ClassifierOutput:
#         self._validate_component_devices()
#         return self.classifier(
#             coordinates,
#             geometry_bank=self.geometry_bank,
#             class_ids=class_ids,
#             candidate=candidate,
#         )

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         class_ids: Sequence[int],
#         candidate: Optional[BoundaryCandidate] = None,
#         return_aux: bool = False,
#     ) -> NECILOutput:
#         representation = self.encode(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         classification = self.classify_coordinates(
#             representation.coordinates,
#             class_ids=class_ids,
#             candidate=candidate,
#         )
#         return NECILOutput(representation=representation, classification=classification)

#     def validate_geometry(self) -> bool:
#         return bool(self.geometry_bank.validate_bank_state())

#     def validate_model_state(self) -> bool:
#         if hasattr(self.backbone, "coordinate_projection"):
#             raise RuntimeError("one-space backbone must not contain coordinate_projection")
#         if int(self.geometry_bank.representation_dim) != self.representation_dim:
#             raise RuntimeError("backbone and geometry representation dimensions disagree")
#         self._validate_component_devices()
#         if any(True for _ in self.classifier.parameters()):
#             raise RuntimeError("GeometryClassifier must remain parameter-free")
#         if len(self.geometry_bank) and not self.spectral_normalization_fitted:
#             raise RuntimeError("committed geometry requires fitted spectral normalization")
#         self.validate_geometry()
#         return True


# __all__ = ["NECILModel", "NECILOutput"]
