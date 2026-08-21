from __future__ import annotations

"""Direct spectral-variation replay for pairwise NECIL-HSI.

This module replaces iterative model-inversion replay with a compact,
data-derived representation of ordered hyperspectral variation.

Persistent historical state for class c is measured once from REAL TRAIN
center spectra when the class is first learned:

    N_c                      number of real spectra
    a_c in R^B               ordered spectral anchor
    V_c in R^(R_c x B)       orthonormal correlated variation directions
    l_c, u_c in R^R_c        observed lower/upper coefficients per direction
    s_c in R^R_c             RMS coefficient magnitude for diagnostics

The anchor is not a classifier prototype.  Classification remains exclusively
in the persistent pairwise BoundaryGeometryBank.  The spectral state exists
only to reconstruct temporary historical HSI evidence.

For each stored mode v_ck, direct replay constructs

    a_c + l_ck v_ck,
    a_c + u_ck v_ck,

using coefficient extrema measured from real spectra.  No Gaussian sampling,
random feature sampling, gradient-based input optimization, replay learning
rate, or replay optimization schedule is used.

Incremental replay is intentionally two-stage:

1. build a direct historical support pool from the stored spectral variation;
2. after old-new candidate boundaries are initialized, select for each old-new
   pair the old support with the smallest old-side signed boundary distance.

Thus replay is selected by the same pairwise decision geometry used by the
classifier rather than generated independently of the incoming classes.

The current processed replay patch remains center-consistent and spatially
neutral.  That is a deliberate temporary limitation: this file improves the
ordered spectral replay mechanism only.  Compact center-relative context
variation should be added separately rather than fabricated here.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.geometry_bank import BoundaryCandidate

Tensor = torch.Tensor


_STATE_VERSION = 3


def _class_ids(values: Sequence[int], *, name: str) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain integer class IDs")
        if isinstance(value, Integral):
            item = int(value)
        elif isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError(f"{name} must contain integer class IDs")
            item = int(number)
        else:
            raise ValueError(f"{name} must contain integer class IDs")
        if item < 0:
            raise ValueError(f"{name} must contain non-negative class IDs")
        result.append(item)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique non-negative class IDs")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        result = int(number)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _pair(a: int, b: int) -> tuple[int, int]:
    left, right = int(a), int(b)
    if left == right:
        raise ValueError("pair requires two different classes")
    return (left, right) if left < right else (right, left)


class SpectralVariationBank:
    """Persistent non-exemplar ordered spectral variation state.

    Each class is represented by an affine spectral variation family measured
    from its real TRAIN center spectra:

        s = anchor_c + sum_k alpha_k v_ck.

    ``lower_coefficients`` and ``upper_coefficients`` store the empirical range
    of each coefficient alpha_k observed in real data.  They are not probability
    parameters and no distributional assumption is made.

    Storage uses fixed [C,B,B] / [C,B] tensors for a simple checkpoint contract.
    ``mode_counts[c]`` states how many rows are valid for class c.  Every
    numerically non-zero centered SVD direction is retained.  Numerical rank is
    determined by ``torch.linalg.matrix_rank`` using machine-precision-scaled
    tolerance; this is a numerical validity rule, not a research threshold or
    tunable replay hyperparameter.
    """

    def __init__(self, spectral_bands: int) -> None:
        bands = _positive_int(spectral_bands, name="spectral_bands")
        self.spectral_bands = bands
        self.class_ids = torch.empty(0, dtype=torch.long)
        self.counts = torch.empty(0, dtype=torch.long)
        self.mode_counts = torch.empty(0, dtype=torch.long)
        self.anchors = torch.empty((0, bands), dtype=torch.float32)
        self.bases = torch.empty((0, bands, bands), dtype=torch.float32)
        self.lower_coefficients = torch.empty((0, bands), dtype=torch.float32)
        self.upper_coefficients = torch.empty((0, bands), dtype=torch.float32)
        self.mode_scales = torch.empty((0, bands), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.class_ids.numel())

    def validate_state(self) -> bool:
        rows = len(self)
        b = self.spectral_bands
        if self.class_ids.shape != (rows,) or self.class_ids.dtype != torch.long:
            raise RuntimeError("spectral variation class_ids are invalid")
        if self.counts.shape != (rows,) or self.counts.dtype != torch.long:
            raise RuntimeError("spectral variation counts are invalid")
        if self.mode_counts.shape != (rows,) or self.mode_counts.dtype != torch.long:
            raise RuntimeError("spectral variation mode_counts are invalid")
        if self.anchors.shape != (rows, b):
            raise RuntimeError("spectral variation anchors have invalid shape")
        if self.bases.shape != (rows, b, b):
            raise RuntimeError("spectral variation bases have invalid shape")
        for name, tensor in (
            ("lower_coefficients", self.lower_coefficients),
            ("upper_coefficients", self.upper_coefficients),
            ("mode_scales", self.mode_scales),
        ):
            if tensor.shape != (rows, b):
                raise RuntimeError(f"spectral variation {name} has invalid shape")
        tensors = (
            self.anchors,
            self.bases,
            self.lower_coefficients,
            self.upper_coefficients,
            self.mode_scales,
        )
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise RuntimeError("spectral variation floating state must use float32")
        if rows:
            ids = [int(v) for v in self.class_ids.tolist()]
            if len(ids) != len(set(ids)) or any(v < 0 for v in ids):
                raise RuntimeError("spectral variation class IDs are invalid")
            if bool((self.counts <= 0).any()):
                raise RuntimeError("spectral variation counts must be positive")
            if bool((self.mode_counts < 0).any()) or bool((self.mode_counts > b).any()):
                raise RuntimeError("spectral variation mode counts are invalid")
            if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
                raise RuntimeError("spectral variation state contains NaN/Inf")
            if bool((self.mode_scales < 0).any()):
                raise RuntimeError("spectral variation mode scales must be non-negative")

            for row in range(rows):
                count = int(self.mode_counts[row].item())
                if count == 0:
                    continue
                basis = self.bases[row, :count]
                gram = basis @ basis.T
                identity = torch.eye(count, dtype=gram.dtype)
                if not bool(torch.allclose(gram, identity, rtol=1e-4, atol=1e-5)):
                    raise RuntimeError(
                        f"spectral variation basis for class {int(self.class_ids[row])} "
                        "is not orthonormal"
                    )
                lower = self.lower_coefficients[row, :count]
                upper = self.upper_coefficients[row, :count]
                if bool((lower > upper).any()):
                    raise RuntimeError("spectral variation coefficient ranges are inverted")
                if bool((lower > 0).any()) or bool((upper < 0).any()):
                    raise RuntimeError(
                        "centered spectral coefficient ranges must contain zero"
                    )
        return True

    @staticmethod
    def _measure_class(class_spectra: Tensor, bands: int) -> Dict[str, Tensor | int]:
        x = torch.as_tensor(class_spectra, dtype=torch.float32, device="cpu")
        if x.ndim != 2 or x.size(0) == 0 or x.size(1) != bands:
            raise ValueError(f"class spectra must be non-empty [N,{bands}]")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("class spectra contain NaN/Inf")

        count = int(x.size(0))
        anchor = x.mean(dim=0)
        centered = x - anchor[None, :]
        mode_count = (
            int(torch.linalg.matrix_rank(centered).item())
            if count > 1
            else 0
        )
        mode_count = min(mode_count, bands)

        basis_padded = torch.zeros((bands, bands), dtype=torch.float32)
        lower_padded = torch.zeros(bands, dtype=torch.float32)
        upper_padded = torch.zeros(bands, dtype=torch.float32)
        scale_padded = torch.zeros(bands, dtype=torch.float32)

        if mode_count:
            # Complete centered SVD up to the mathematically available N-1
            # directions. No explained-variance threshold is introduced.
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[:mode_count].to(torch.float32)
            coefficients = centered @ basis.T
            # Centered coefficients have zero empirical mean, so zero belongs
            # to every coefficient range mathematically.  Enforce that exact
            # invariant against floating-point SVD roundoff without introducing
            # a research tolerance or widening any range beyond the anchor.
            zero = torch.zeros((), dtype=coefficients.dtype)
            lower = torch.minimum(coefficients.amin(dim=0), zero)
            upper = torch.maximum(coefficients.amax(dim=0), zero)
            scale = torch.sqrt(coefficients.square().mean(dim=0))

            basis_padded[:mode_count] = basis
            lower_padded[:mode_count] = lower
            upper_padded[:mode_count] = upper
            scale_padded[:mode_count] = scale

            if singular.numel() < mode_count:
                raise RuntimeError("SVD returned fewer modes than requested")

        return {
            "count": count,
            "mode_count": mode_count,
            "anchor": anchor.to(torch.float32),
            "basis": basis_padded,
            "lower": lower_padded,
            "upper": upper_padded,
            "scale": scale_padded,
        }

    def append_real_spectra(
        self,
        spectra: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int],
    ) -> None:
        """Measure and append new-class variation from REAL TRAIN spectra only."""
        self.validate_state()
        new_ids = _class_ids(class_ids, name="class_ids")
        existing = set(int(v) for v in self.class_ids.tolist())
        overlap = sorted(existing.intersection(new_ids))
        if overlap:
            raise ValueError(f"spectral variation already exists for classes {overlap}")

        x = torch.as_tensor(spectra, dtype=torch.float32, device="cpu")
        y = torch.as_tensor(labels, device="cpu").flatten()
        if x.ndim != 2 or x.size(0) == 0 or x.size(1) != self.spectral_bands:
            raise ValueError(f"spectra must be non-empty [N,{self.spectral_bands}]")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("spectra contain NaN/Inf")
        if y.numel() != x.size(0) or y.dtype == torch.bool or y.is_complex():
            raise ValueError("labels are invalid or row-misaligned")
        if torch.is_floating_point(y):
            if not bool(torch.isfinite(y).all()) or not bool(y.eq(y.round()).all()):
                raise ValueError("labels must contain finite integer IDs")
        y = y.to(torch.long)

        observed = sorted(int(v) for v in y.unique().tolist())
        if observed != sorted(new_ids):
            raise ValueError(
                f"real spectra contain classes {observed}; expected {sorted(new_ids)}"
            )

        measured = [
            self._measure_class(x[y.eq(class_id)], self.spectral_bands)
            for class_id in new_ids
        ]

        self.class_ids = torch.cat(
            (self.class_ids, torch.tensor(new_ids, dtype=torch.long)), dim=0
        )
        self.counts = torch.cat(
            (
                self.counts,
                torch.tensor([int(row["count"]) for row in measured], dtype=torch.long),
            ),
            dim=0,
        )
        self.mode_counts = torch.cat(
            (
                self.mode_counts,
                torch.tensor(
                    [int(row["mode_count"]) for row in measured], dtype=torch.long
                ),
            ),
            dim=0,
        )
        self.anchors = torch.cat(
            (self.anchors, torch.stack([row["anchor"] for row in measured])), dim=0
        )
        self.bases = torch.cat(
            (self.bases, torch.stack([row["basis"] for row in measured])), dim=0
        )
        self.lower_coefficients = torch.cat(
            (
                self.lower_coefficients,
                torch.stack([row["lower"] for row in measured]),
            ),
            dim=0,
        )
        self.upper_coefficients = torch.cat(
            (
                self.upper_coefficients,
                torch.stack([row["upper"] for row in measured]),
            ),
            dim=0,
        )
        self.mode_scales = torch.cat(
            (self.mode_scales, torch.stack([row["scale"] for row in measured])),
            dim=0,
        )
        self.validate_state()

    def append_from_loader(self, loader: Any, *, class_ids: Sequence[int]) -> None:
        """Measure new rows from a REAL current-phase TRAIN loader."""
        spectra: list[Tensor] = []
        labels: list[Tensor] = []
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("spectral variation fitting expects mapping batches")
            required = {"raw_center_spectrum", "label"}
            missing = required - set(batch)
            if missing:
                raise KeyError(f"spectral variation batch lacks {sorted(missing)}")
            spectra.append(torch.as_tensor(batch["raw_center_spectrum"]).detach().cpu())
            labels.append(torch.as_tensor(batch["label"]).detach().cpu().flatten())
        if not spectra:
            raise RuntimeError("cannot fit spectral variation from an empty loader")
        self.append_real_spectra(
            torch.cat(spectra, dim=0),
            torch.cat(labels, dim=0),
            class_ids=class_ids,
        )

    def rows(
        self,
        class_ids: Sequence[int],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Tensor]:
        self.validate_state()
        requested = _class_ids(class_ids, name="class_ids")
        lookup = {int(v): i for i, v in enumerate(self.class_ids.tolist())}
        missing = [class_id for class_id in requested if class_id not in lookup]
        if missing:
            raise ValueError(f"spectral variation is missing classes {missing}")
        index = torch.tensor([lookup[c] for c in requested], dtype=torch.long)
        dev = torch.device(device)
        return {
            "counts": self.counts.index_select(0, index).to(device=dev),
            "mode_counts": self.mode_counts.index_select(0, index).to(device=dev),
            "anchors": self.anchors.index_select(0, index).to(device=dev, dtype=dtype),
            "bases": self.bases.index_select(0, index).to(device=dev, dtype=dtype),
            "lower_coefficients": self.lower_coefficients.index_select(0, index).to(
                device=dev, dtype=dtype
            ),
            "upper_coefficients": self.upper_coefficients.index_select(0, index).to(
                device=dev, dtype=dtype
            ),
            "mode_scales": self.mode_scales.index_select(0, index).to(
                device=dev, dtype=dtype
            ),
        }

    def state_dict(self) -> Dict[str, Any]:
        self.validate_state()
        return {
            "contract_version": _STATE_VERSION,
            "state_type": "ordered_spectral_variation",
            "spectral_bands": int(self.spectral_bands),
            "class_ids": self.class_ids.clone(),
            "counts": self.counts.clone(),
            "mode_counts": self.mode_counts.clone(),
            "anchors": self.anchors.clone(),
            "bases": self.bases.clone(),
            "lower_coefficients": self.lower_coefficients.clone(),
            "upper_coefficients": self.upper_coefficients.clone(),
            "mode_scales": self.mode_scales.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("spectral variation state must be a mapping")
        if int(state.get("contract_version", -1)) != _STATE_VERSION:
            raise RuntimeError(
                "spectral variation checkpoint uses an incompatible contract; "
                "legacy mean/variance replay is intentionally not converted"
            )
        if state.get("state_type") != "ordered_spectral_variation":
            raise RuntimeError("checkpoint does not contain ordered spectral variation")
        if int(state.get("spectral_bands", -1)) != self.spectral_bands:
            raise RuntimeError("spectral variation state uses a different band count")
        required = {
            "class_ids",
            "counts",
            "mode_counts",
            "anchors",
            "bases",
            "lower_coefficients",
            "upper_coefficients",
            "mode_scales",
        }
        missing = required - set(state)
        if missing:
            raise RuntimeError(f"spectral variation state lacks {sorted(missing)}")

        self.class_ids = torch.as_tensor(state["class_ids"], dtype=torch.long, device="cpu").clone()
        self.counts = torch.as_tensor(state["counts"], dtype=torch.long, device="cpu").clone()
        self.mode_counts = torch.as_tensor(
            state["mode_counts"], dtype=torch.long, device="cpu"
        ).clone()
        self.anchors = torch.as_tensor(
            state["anchors"], dtype=torch.float32, device="cpu"
        ).clone()
        self.bases = torch.as_tensor(
            state["bases"], dtype=torch.float32, device="cpu"
        ).clone()
        self.lower_coefficients = torch.as_tensor(
            state["lower_coefficients"], dtype=torch.float32, device="cpu"
        ).clone()
        self.upper_coefficients = torch.as_tensor(
            state["upper_coefficients"], dtype=torch.float32, device="cpu"
        ).clone()
        self.mode_scales = torch.as_tensor(
            state["mode_scales"], dtype=torch.float32, device="cpu"
        ).clone()
        self.validate_state()

    def summary(self) -> Dict[str, Any]:
        self.validate_state()
        return {
            "contract_version": _STATE_VERSION,
            "class_ids": [int(v) for v in self.class_ids.tolist()],
            "class_count": len(self),
            "spectral_bands": int(self.spectral_bands),
            "real_train_counts": {
                int(class_id): int(count)
                for class_id, count in zip(self.class_ids.tolist(), self.counts.tolist())
            },
            "mode_counts": {
                int(class_id): int(count)
                for class_id, count in zip(
                    self.class_ids.tolist(), self.mode_counts.tolist()
                )
            },
            "role": (
                "ordered spectral anchor + complete centered correlated variation; "
                "temporary replay construction only; never classification"
            ),
            "distribution_assumption": "none",
        }


class FrozenHSIPreprocessor(nn.Module):
    """Differentiable copy of the CURRENT frozen base HSI preprocessor.

    This class intentionally accepts exactly one preprocessing contract: the
    flat state returned by the current ``FitHSIPreprocessorFromProtocol`` and
    ``LoadHSIPreprocessor`` implementations:

        raw_bands
        fit_pixel_count
        normalization_mean
        normalization_std
        pca_components
        pca_mean
        pca_variance
        whiten
        fit_scope

    The replay transform is exactly the same as ``ApplyHSIPreprocessor``:

        normalized = (raw - normalization_mean) / normalization_std

        if PCA is active:
            processed = (normalized - pca_mean) @ pca_components.T

            if whiten:
                processed /= sqrt(pca_variance)

    Ordered center spectra themselves remain in the original sensor-band order.
    """

    def __init__(self, state: Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(state, Mapping):
            raise TypeError("preprocessing state must be a mapping")

        required = {
            "raw_bands",
            "fit_pixel_count",
            "normalization_mean",
            "normalization_std",
            "pca_components",
            "pca_mean",
            "pca_variance",
            "whiten",
            "fit_scope",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(
                "preprocessing state does not match the current HSI loader "
                f"contract; missing {sorted(missing)}"
            )

        fit_scope = str(state["fit_scope"]).strip().lower()
        if fit_scope != "base_train":
            raise ValueError(
                "spectral replay requires preprocessing fitted from base_train"
            )

        bands = int(state["raw_bands"])
        fit_pixel_count = int(state["fit_pixel_count"])
        if bands <= 0:
            raise ValueError("raw_bands must be positive")
        if fit_pixel_count <= 0:
            raise ValueError("fit_pixel_count must be positive")

        mean = torch.as_tensor(
            state["normalization_mean"],
            dtype=torch.float32,
        ).flatten()
        std = torch.as_tensor(
            state["normalization_std"],
            dtype=torch.float32,
        ).flatten()
        components = torch.as_tensor(
            state["pca_components"],
            dtype=torch.float32,
        )
        pca_mean = torch.as_tensor(
            state["pca_mean"],
            dtype=torch.float32,
        ).flatten()
        variance = torch.as_tensor(
            state["pca_variance"],
            dtype=torch.float32,
        ).flatten()
        whiten = bool(state["whiten"])

        if mean.shape != (bands,) or std.shape != (bands,):
            raise ValueError(
                "normalization state has incompatible dimensions"
            )
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(std).all()
        ):
            raise ValueError(
                "normalization state contains NaN/Inf"
            )
        if bool((std <= 0).any()):
            raise ValueError(
                "normalization_std must be positive"
            )

        if components.ndim != 2 or components.size(1) != bands:
            raise ValueError(
                "pca_components must have shape [K,raw_bands]"
            )

        pca_active = bool(components.numel())
        if pca_active:
            if pca_mean.shape != (bands,):
                raise ValueError(
                    "pca_mean has incompatible dimensions"
                )
            if variance.shape != (components.size(0),):
                raise ValueError(
                    "pca_variance has incompatible dimensions"
                )
            if not bool(torch.isfinite(components).all()):
                raise ValueError(
                    "pca_components contains NaN/Inf"
                )
            if not bool(torch.isfinite(pca_mean).all()):
                raise ValueError(
                    "pca_mean contains NaN/Inf"
                )
            if not bool(torch.isfinite(variance).all()):
                raise ValueError(
                    "pca_variance contains NaN/Inf"
                )
            if whiten and bool((variance <= 0).any()):
                raise ValueError(
                    "PCA whitening requires positive variance"
                )
        else:
            if pca_mean.numel() != 0 or variance.numel() != 0:
                raise ValueError(
                    "inactive PCA must have empty pca_mean and pca_variance"
                )
            if whiten:
                raise ValueError(
                    "whiten=True requires active PCA"
                )

        self.raw_bands = bands
        self.fit_pixel_count = fit_pixel_count
        self.pca_active = pca_active
        self.whiten = whiten
        self.fit_scope = fit_scope

        self.register_buffer(
            "normalization_mean",
            mean,
        )
        self.register_buffer(
            "normalization_std",
            std,
        )
        self.register_buffer(
            "pca_components",
            components,
        )
        self.register_buffer(
            "pca_mean",
            pca_mean,
        )
        self.register_buffer(
            "pca_variance",
            variance,
        )

    @property
    def processed_bands(self) -> int:
        return (
            int(self.pca_components.size(0))
            if self.pca_active
            else int(self.raw_bands)
        )

    def transform_spectra(
        self,
        spectra: Tensor,
    ) -> Tensor:
        """Apply the exact current HSI preprocessing equation to spectra."""
        x = torch.as_tensor(spectra)
        if (
            x.ndim != 2
            or x.size(0) == 0
            or x.size(1) != self.raw_bands
        ):
            raise ValueError(
                f"spectra must be non-empty [N,{self.raw_bands}]"
            )
        if not torch.is_floating_point(x) or x.is_complex():
            raise TypeError(
                "spectra must be real floating point"
            )
        if x.device != self.normalization_mean.device:
            raise ValueError(
                "spectra and frozen preprocessor must share a device"
            )
        if not bool(torch.isfinite(x).all()):
            raise ValueError(
                "spectra contain NaN/Inf"
            )

        normalized = (
            x
            - self.normalization_mean.to(dtype=x.dtype)
        ) / self.normalization_std.to(dtype=x.dtype)

        if not self.pca_active:
            if not bool(torch.isfinite(normalized).all()):
                raise RuntimeError(
                    "frozen HSI normalization produced NaN/Inf"
                )
            return normalized

        transformed = (
            normalized
            - self.pca_mean.to(dtype=x.dtype)
        ) @ self.pca_components.to(dtype=x.dtype).T

        if self.whiten:
            transformed = transformed / torch.sqrt(
                self.pca_variance.to(dtype=x.dtype)
            )

        if not bool(torch.isfinite(transformed).all()):
            raise RuntimeError(
                "frozen HSI preprocessing produced NaN/Inf"
            )
        return transformed

    def neutral_patch(
        self,
        spectra: Tensor,
        *,
        patch_size: int,
    ) -> Tensor:
        """Derive neutral replay context from the same ordered spectrum.

        Spectral replay reconstructs old spectral evidence.  It does not store or
        invent an old spatial exemplar.  The processed version of the synthetic
        center spectrum is therefore repeated spatially.  For the current
        center-relative context branch this gives neutral old context while the
        raw spectral branch and processed context remain exactly consistent.
        """
        size = int(patch_size)
        if size <= 0 or size % 2 == 0:
            raise ValueError(
                "patch_size must be positive and odd"
            )
        processed = self.transform_spectra(spectra)
        return (
            processed[:, :, None, None]
            .expand(-1, -1, size, size)
            .contiguous()
        )




class SpectralReplayDataset(Dataset):
    """Temporary replay data matching the HSI model-input contract.

    ``replay_mode_index``/``replay_mode_side`` identify how a support was
    reconstructed from historical spectral variation.

    After boundary-based selection, ``old_boundary_response`` and
    ``old_rival_class_ids`` may be attached.  They contain the phase-start
    *class-incident* old decision coordinates for each replay row:

        r_c(z) = [s_cj(z)]_{j != c}.

    These targets are required by the incremental preservation objective and
    must not be discarded by the replay training stream.  They intentionally do
    not store values for old-old boundaries unrelated to the replay class.
    """

    def __init__(
        self,
        processed_spectra: Tensor,
        spectra: Tensor,
        labels: Tensor,
        *,
        patch_size: int,
        mode_indices: Optional[Tensor] = None,
        mode_sides: Optional[Tensor] = None,
        old_boundary_response: Optional[Tensor] = None,
        old_rival_class_ids: Optional[Tensor] = None,
    ) -> None:
        processed = torch.as_tensor(
            processed_spectra, dtype=torch.float32, device="cpu"
        ).contiguous()
        raw = torch.as_tensor(
            spectra, dtype=torch.float32, device="cpu"
        ).contiguous()
        y = torch.as_tensor(
            labels, dtype=torch.long, device="cpu"
        ).flatten().contiguous()

        size = _positive_int(patch_size, name="patch_size")
        if size % 2 == 0:
            raise ValueError("patch_size must be odd")
        if processed.ndim != 2 or processed.size(0) == 0:
            raise ValueError("processed_spectra must be non-empty [N,K]")
        if raw.ndim != 2 or raw.size(0) != processed.size(0):
            raise ValueError("raw_center_spectrum is batch-misaligned")
        if y.shape != (processed.size(0),):
            raise ValueError("labels are batch-misaligned")
        if not bool(torch.isfinite(processed).all()) or not bool(
            torch.isfinite(raw).all()
        ):
            raise ValueError("replay tensors contain NaN/Inf")

        n = int(y.numel())
        if mode_indices is None:
            modes = torch.full((n,), -1, dtype=torch.long)
        else:
            modes = torch.as_tensor(
                mode_indices, dtype=torch.long, device="cpu"
            ).flatten()
        if mode_sides is None:
            sides = torch.zeros(n, dtype=torch.int8)
        else:
            sides = torch.as_tensor(
                mode_sides, dtype=torch.int8, device="cpu"
            ).flatten()
        if modes.shape != (n,) or sides.shape != (n,):
            raise ValueError("replay support metadata is batch-misaligned")
        if bool((modes < -1).any()):
            raise ValueError("mode_indices must be -1 for anchor or non-negative")
        allowed_sides = torch.tensor([-1, 0, 1], dtype=torch.long)
        if not bool(torch.isin(sides.to(torch.long), allowed_sides).all()):
            raise ValueError("mode_sides must contain only -1, 0, or +1")
        if bool(((modes == -1) != (sides == 0)).any()):
            raise ValueError("anchor metadata requires mode_index=-1 and side=0")

        historical = None
        rivals = None
        if old_boundary_response is not None or old_rival_class_ids is not None:
            if old_boundary_response is None or old_rival_class_ids is None:
                raise ValueError(
                    "old_boundary_response and old_rival_class_ids must be supplied together"
                )
            historical = torch.as_tensor(
                old_boundary_response, dtype=torch.float32, device="cpu"
            ).contiguous()
            rivals = torch.as_tensor(
                old_rival_class_ids, dtype=torch.long, device="cpu"
            ).contiguous()
            if historical.ndim != 2 or historical.size(0) != n:
                raise ValueError("old_boundary_response must have shape [N,C_old-1]")
            if rivals.shape != historical.shape:
                raise ValueError(
                    "old_rival_class_ids must align with old_boundary_response"
                )
            if not bool(torch.isfinite(historical).all()):
                raise ValueError("old_boundary_response contains NaN/Inf")
            if bool((rivals < 0).any()):
                raise ValueError("old_rival_class_ids must be non-negative")
            if historical.size(1) > 0:
                for row in range(n):
                    row_rivals = rivals[row]
                    if row_rivals.unique().numel() != row_rivals.numel():
                        raise ValueError("old rival IDs must be unique within each row")
                    if bool(row_rivals.eq(y[row]).any()):
                        raise ValueError("a historical rival cannot equal the replay label")

        self.processed_spectra = processed
        self.raw_center_spectrum = raw
        self.labels = y
        self.mode_indices = modes.contiguous()
        self.mode_sides = sides.contiguous()
        self.patch_size = size
        self.old_boundary_response = historical
        self.old_rival_class_ids = rivals

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        processed = self.processed_spectra[index]
        patch = processed[:, None, None].expand(
            processed.numel(), self.patch_size, self.patch_size
        ).contiguous()
        item: Dict[str, Tensor] = {
            "image": patch,
            "raw_center_spectrum": self.raw_center_spectrum[index],
            "label": self.labels[index],
            "replay_mode_index": self.mode_indices[index],
            "replay_mode_side": self.mode_sides[index],
        }
        if self.old_boundary_response is not None:
            item["old_boundary_response"] = self.old_boundary_response[index]
            item["old_rival_class_ids"] = self.old_rival_class_ids[index]
        return item

    def subset(self, indices: Sequence[int] | Tensor) -> "SpectralReplayDataset":
        index = torch.as_tensor(
            indices, dtype=torch.long, device="cpu"
        ).flatten()
        if index.numel() == 0:
            raise ValueError("replay subset cannot be empty")
        if bool((index < 0).any()) or bool((index >= len(self)).any()):
            raise ValueError("replay subset indices are out of range")
        if index.unique().numel() != index.numel():
            raise ValueError("replay subset indices must be unique")
        return SpectralReplayDataset(
            self.processed_spectra.index_select(0, index),
            self.raw_center_spectrum.index_select(0, index),
            self.labels.index_select(0, index),
            patch_size=self.patch_size,
            mode_indices=self.mode_indices.index_select(0, index),
            mode_sides=self.mode_sides.index_select(0, index),
            old_boundary_response=(
                None
                if self.old_boundary_response is None
                else self.old_boundary_response.index_select(0, index)
            ),
            old_rival_class_ids=(
                None
                if self.old_rival_class_ids is None
                else self.old_rival_class_ids.index_select(0, index)
            ),
        )

    def with_historical_response(
        self,
        response: Tensor,
        rival_class_ids: Tensor,
    ) -> "SpectralReplayDataset":
        return SpectralReplayDataset(
            self.processed_spectra,
            self.raw_center_spectrum,
            self.labels,
            patch_size=self.patch_size,
            mode_indices=self.mode_indices,
            mode_sides=self.mode_sides,
            old_boundary_response=response,
            old_rival_class_ids=rival_class_ids,
        )


@dataclass(frozen=True)
class SpectralReplayResult:
    """Direct replay construction before current-phase boundary selection."""

    initialization_dataset: SpectralReplayDataset
    support_pool: SpectralReplayDataset
    support_coordinates: Tensor
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SpectralReplaySelection:
    """Boundary-selected replay used by incremental optimization."""

    dataset: SpectralReplayDataset
    selected_pool_indices: Tensor
    selected_coordinates: Tensor
    pair_to_pool_index: Dict[str, int]
    pair_to_selected_index: Dict[str, int]
    diagnostics: Dict[str, Any]


class SpectralReplayGenerator:
    """Construct direct spectral supports and select decision-critical replay."""

    def __init__(
        self,
        *,
        bank: SpectralVariationBank,
        preprocessor: FrozenHSIPreprocessor,
    ) -> None:
        if not isinstance(bank, SpectralVariationBank):
            raise TypeError("bank must be SpectralVariationBank")
        if not isinstance(preprocessor, FrozenHSIPreprocessor):
            raise TypeError("preprocessor must be FrozenHSIPreprocessor")
        if bank.spectral_bands != preprocessor.raw_bands:
            raise ValueError("spectral bank and preprocessor band counts disagree")
        self.bank = bank
        self.preprocessor = preprocessor

    @staticmethod
    def _model_contract(
        model: nn.Module,
        bank: SpectralVariationBank,
    ) -> tuple[int, int, int]:
        if (
            not hasattr(model, "committed_class_ids")
            or not hasattr(model, "backbone")
            or not hasattr(model, "geometry_bank")
        ):
            raise TypeError(
                "model must expose committed_class_ids, backbone, and geometry_bank"
            )
        patch_size = int(getattr(model.backbone, "patch_size", 0))
        patch_bands = int(getattr(model.backbone, "patch_bands", 0))
        spectral_bands = int(getattr(model.backbone, "spectral_bands", 0))
        if (
            patch_size <= 0
            or patch_size % 2 == 0
            or patch_bands <= 0
            or spectral_bands <= 0
        ):
            raise RuntimeError("model backbone exposes an invalid HSI input contract")
        if spectral_bands != bank.spectral_bands:
            raise RuntimeError(
                "model and spectral variation bank disagree on ordered bands"
            )
        return patch_size, patch_bands, spectral_bands

    def _dataset_from_raw(
        self,
        spectra: Tensor,
        labels: Tensor,
        *,
        patch_size: int,
        mode_indices: Tensor,
        mode_sides: Tensor,
    ) -> SpectralReplayDataset:
        device = self.preprocessor.normalization_mean.device
        raw = torch.as_tensor(spectra, dtype=torch.float32, device=device)
        processed = self.preprocessor.transform_spectra(raw)
        return SpectralReplayDataset(
            processed.detach().cpu(),
            raw.detach().cpu(),
            torch.as_tensor(labels, dtype=torch.long, device="cpu"),
            patch_size=patch_size,
            mode_indices=mode_indices,
            mode_sides=mode_sides,
        )

    @staticmethod
    @torch.no_grad()
    def _encode_dataset(
        model: nn.Module,
        dataset: SpectralReplayDataset,
        *,
        batch_size: int,
    ) -> Tensor:
        size = _positive_int(batch_size, name="batch_size")
        device = torch.device(model.device)
        states = {module: bool(module.training) for module in model.modules()}
        coordinates: list[Tensor] = []
        try:
            model.eval()
            for start in range(0, len(dataset), size):
                stop = min(start + size, len(dataset))
                processed = dataset.processed_spectra[start:stop].to(device=device)
                raw = dataset.raw_center_spectrum[start:stop].to(device=device)
                patch = processed[:, :, None, None].expand(
                    -1, -1, dataset.patch_size, dataset.patch_size
                ).contiguous()
                output = model.encode(
                    patch,
                    center_spectrum=raw,
                    return_aux=False,
                )
                coordinates.append(output.coordinates.detach().cpu())
        finally:
            for module, state in states.items():
                module.training = state
        if not coordinates:
            raise RuntimeError("cannot encode an empty replay dataset")
        result = torch.cat(coordinates, dim=0)
        if result.size(0) != len(dataset) or not bool(torch.isfinite(result).all()):
            raise RuntimeError("replay support encoding is invalid")
        return result

    @torch.no_grad()
    def generate(
        self,
        *,
        model: nn.Module,
        class_ids: Sequence[int],
        batch_size: int,
    ) -> SpectralReplayResult:
        """Construct anchors and empirical mode-extreme historical supports.

        This contains no optimizer, random synthesis, or backward pass.  The
        support pool is encoded exactly once with the phase-start model and the
        resulting coordinates are reused for candidate initialization and later
        boundary-conditioned selection.
        """
        old_ids = _class_ids(class_ids, name="class_ids")
        committed = [int(v) for v in model.committed_class_ids]
        if committed != old_ids:
            raise RuntimeError(
                "direct replay must use complete committed old classes; "
                f"committed={committed}, requested={old_ids}"
            )
        patch_size, patch_bands, _ = self._model_contract(model, self.bank)
        if patch_bands != self.preprocessor.processed_bands:
            raise RuntimeError(
                "model and replay preprocessor disagree on processed bands"
            )

        model_device = torch.device(model.device)
        self.preprocessor.to(model_device)
        rows = self.bank.rows(
            old_ids,
            device=model_device,
            dtype=torch.float32,
        )

        pool_spectra: list[Tensor] = []
        pool_labels: list[int] = []
        pool_modes: list[int] = []
        pool_sides: list[int] = []
        supports_per_class: Dict[int, int] = {}

        for row, class_id in enumerate(old_ids):
            anchor = rows["anchors"][row]
            basis = rows["bases"][row]
            lower = rows["lower_coefficients"][row]
            upper = rows["upper_coefficients"][row]
            mode_count = int(rows["mode_counts"][row].item())

            pool_spectra.append(anchor)
            pool_labels.append(class_id)
            pool_modes.append(-1)
            pool_sides.append(0)
            count = 1

            for mode in range(mode_count):
                direction = basis[mode]
                low_support = anchor + lower[mode] * direction
                high_support = anchor + upper[mode] * direction

                # A zero-width empirical coefficient produces the anchor exactly
                # and contributes no distinct replay evidence.
                if not torch.equal(low_support, anchor):
                    pool_spectra.append(low_support)
                    pool_labels.append(class_id)
                    pool_modes.append(mode)
                    pool_sides.append(-1)
                    count += 1
                if not torch.equal(high_support, anchor) and not torch.equal(
                    high_support, low_support
                ):
                    pool_spectra.append(high_support)
                    pool_labels.append(class_id)
                    pool_modes.append(mode)
                    pool_sides.append(1)
                    count += 1

            supports_per_class[class_id] = count

        if not pool_spectra:
            raise RuntimeError("direct replay produced no historical support")

        pool = self._dataset_from_raw(
            torch.stack(pool_spectra),
            torch.tensor(pool_labels, dtype=torch.long),
            patch_size=patch_size,
            mode_indices=torch.tensor(pool_modes, dtype=torch.long),
            mode_sides=torch.tensor(pool_sides, dtype=torch.int8),
        )
        support_coordinates = self._encode_dataset(
            model,
            pool,
            batch_size=batch_size,
        )

        diagnostics = {
            "class_ids": old_ids,
            "construction": "direct empirical spectral-variation supports",
            "optimization_steps": 0,
            "backward_passes": 0,
            "phase_start_forward_support_count": len(pool),
            "initialization_support_count": len(pool),
            "support_pool_count": len(pool),
            "supports_per_class": supports_per_class,
            "persistent_state_used": (
                "ordered spectral anchors + centered SVD variation bases + "
                "empirical coefficient ranges"
            ),
            "distribution_assumption": "none",
            "context_policy": (
                "neutral processed context derived from each reconstructed ordered spectrum"
            ),
        }
        return SpectralReplayResult(
            initialization_dataset=pool,
            support_pool=pool,
            support_coordinates=support_coordinates,
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def select_boundary_supports(
        self,
        *,
        model: nn.Module,
        replay: SpectralReplayResult,
        candidate: BoundaryCandidate,
        old_class_ids: Sequence[int],
        new_class_ids: Sequence[int],
    ) -> SpectralReplaySelection:
        """Select one most vulnerable old support for every old-new boundary.

        Selection uses ``BoundaryGeometryBank.pair_values`` rather than reading
        candidate normals/offsets directly.  This guarantees replay uses exactly
        the same normalized affine geometry as training and classification.
        """
        old_ids = _class_ids(old_class_ids, name="old_class_ids")
        new_ids = _class_ids(new_class_ids, name="new_class_ids")
        if set(old_ids).intersection(new_ids):
            raise ValueError("old and new class IDs overlap")
        if [int(v) for v in model.committed_class_ids] != old_ids:
            raise RuntimeError(
                "model committed classes do not match requested old classes"
            )
        if candidate.new_class_ids != tuple(new_ids):
            raise RuntimeError(
                "candidate new classes do not match requested new classes"
            )

        bank = getattr(model, "geometry_bank", None)
        if bank is None:
            raise TypeError("model must expose geometry_bank")
        bank.validate_bank_state()

        pool = replay.support_pool
        coordinates_cpu = torch.as_tensor(
            replay.support_coordinates
        ).detach().cpu()
        if coordinates_cpu.ndim != 2 or coordinates_cpu.size(0) != len(pool):
            raise RuntimeError("replay support coordinates are invalid")
        coordinates = coordinates_cpu.to(device=bank.device, dtype=bank.dtype)
        labels = pool.labels.to(device=bank.device)

        old_new_pairs = [
            _pair(old_id, new_id)
            for old_id in old_ids
            for new_id in new_ids
        ]
        pair_tensor = torch.tensor(
            old_new_pairs,
            device=bank.device,
            dtype=torch.long,
        )
        pair_geometry = bank.pair_values(
            coordinates,
            pair_ids=pair_tensor,
            candidate=candidate,
        )

        selected_pool_rows: list[int] = []
        pair_to_pool_index: Dict[str, int] = {}
        pair_signed_distance: Dict[str, float] = {}

        for column, pair in enumerate(old_new_pairs):
            left, right = pair
            old_id = left if left in old_ids else right
            new_id = right if old_id == left else left
            class_pool = torch.nonzero(
                labels.eq(old_id), as_tuple=False
            ).flatten()
            if class_pool.numel() == 0:
                raise RuntimeError(f"support pool lacks old class {old_id}")

            h = pair_geometry.values.index_select(0, class_pool)[:, column]
            signed_old = h if old_id == left else -h
            local = int(signed_old.argmin().item())
            pool_index = int(class_pool[local].item())
            key = f"{old_id}:{new_id}"
            pair_to_pool_index[key] = pool_index
            pair_signed_distance[key] = float(signed_old[local].item())
            selected_pool_rows.append(pool_index)

        # A single historical support may be the most vulnerable point for more
        # than one incoming class.  Keep one physical row while retaining both
        # pair-to-pool and pair-to-selected mappings.
        unique_selected = list(dict.fromkeys(selected_pool_rows))
        if not unique_selected:
            raise RuntimeError("boundary-conditioned replay selected no support")
        selection_index = torch.tensor(unique_selected, dtype=torch.long)
        pool_to_selected = {
            pool_index: selected_index
            for selected_index, pool_index in enumerate(unique_selected)
        }
        pair_to_selected_index = {
            key: pool_to_selected[pool_index]
            for key, pool_index in pair_to_pool_index.items()
        }

        dataset = pool.subset(selection_index)
        selected_coordinates = coordinates_cpu.index_select(0, selection_index)
        per_old = {
            old_id: int(dataset.labels.eq(old_id).sum().item())
            for old_id in old_ids
        }
        diagnostics = {
            "selection_rule": (
                "one minimum old-side signed-distance support per old-new pair; "
                "duplicate physical supports deduplicated"
            ),
            "old_class_ids": old_ids,
            "new_class_ids": new_ids,
            "old_new_pair_count": len(old_ids) * len(new_ids),
            "selected_unique_support_count": len(dataset),
            "selected_supports_per_old_class": per_old,
            "pair_signed_distance": pair_signed_distance,
            "pair_to_pool_index": pair_to_pool_index,
            "pair_to_selected_index": pair_to_selected_index,
        }
        return SpectralReplaySelection(
            dataset=dataset,
            selected_pool_indices=selection_index,
            selected_coordinates=selected_coordinates,
            pair_to_pool_index=pair_to_pool_index,
            pair_to_selected_index=pair_to_selected_index,
            diagnostics=diagnostics,
        )

    @staticmethod
    @torch.no_grad()
    def attach_phase_start_boundary_response(
        *,
        model: nn.Module,
        selection: SpectralReplaySelection,
    ) -> SpectralReplaySelection:
        """Attach class-incident historical decision coordinates.

        For selected replay item i of old class c, cache only

            r_c(z_i) = [s_cj(z_i)]_{j in C_old, j != c}.

        This is the exact target used by incremental historical-response
        preservation.  Unrelated old-old boundaries are intentionally excluded.
        """
        bank = getattr(model, "geometry_bank", None)
        if bank is None:
            raise TypeError("model must expose geometry_bank")
        bank.validate_bank_state()
        old_ids = [int(v) for v in model.committed_class_ids]
        if len(old_ids) < 2:
            raise RuntimeError(
                "historical boundary response requires at least two committed classes"
            )

        dataset = selection.dataset
        coordinates_cpu = torch.as_tensor(
            selection.selected_coordinates
        ).detach().cpu()
        if coordinates_cpu.ndim != 2 or coordinates_cpu.size(0) != len(dataset):
            raise RuntimeError("selected replay coordinates are invalid")

        coordinates = coordinates_cpu.to(device=bank.device, dtype=bank.dtype)
        labels = dataset.labels.to(device=bank.device)
        response = bank.class_boundary_response(
            coordinates,
            labels,
            class_ids=old_ids,
            candidate=None,
        )
        signed_dataset = dataset.with_historical_response(
            response.margins.detach().cpu(),
            response.rival_class_ids.detach().cpu(),
        )

        diagnostics = dict(selection.diagnostics)
        diagnostics.update(
            {
                "phase_start_historical_target": (
                    "class-incident committed old boundary responses"
                ),
                "historical_response_width": int(response.margins.size(1)),
            }
        )
        return SpectralReplaySelection(
            dataset=signed_dataset,
            selected_pool_indices=selection.selected_pool_indices,
            selected_coordinates=coordinates_cpu,
            pair_to_pool_index=dict(selection.pair_to_pool_index),
            pair_to_selected_index=dict(selection.pair_to_selected_index),
            diagnostics=diagnostics,
        )


# Temporary compatibility name so checkpoint/trainer migration can be staged.
# New code should import SpectralVariationBank explicitly.
SpectralConstraintBank = SpectralVariationBank


__all__ = [
    "FrozenHSIPreprocessor",
    "SpectralConstraintBank",
    "SpectralVariationBank",
    "SpectralReplayDataset",
    "SpectralReplayGenerator",
    "SpectralReplayResult",
    "SpectralReplaySelection",
]
