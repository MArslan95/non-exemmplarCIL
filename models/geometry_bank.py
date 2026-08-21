from __future__ import annotations

"""Pairwise decision geometry for one-space NECIL-HSI.

The HSI backbone produces one canonical feature z in R^D.  For every unordered
pair of seen classes {a,b}, the geometry stores one shared affine boundary

    h_ab(z) = n_ab^T z + q_ab,    a < b.

The positive side belongs to class a and the negative side belongs to class b.
For class c and rival j, define the oriented signed distance

    s_cj(z) =  h_cj(z)   if c < j,
             -h_jc(z)   if j < c.

The class decision cell is the intersection of all its pairwise half-spaces,

    C_c = {z : s_cj(z) >= 0 for every j != c}.

A single class energy is used everywhere,

    E_c(z) = - min_{j != c} s_cj(z).

Therefore E_c(z) <= 0 means z is inside class c's decision cell and prediction
uses argmin_c E_c(z).

Unlike an axis-aligned box, the geometry stores explicit discriminative
boundaries.  For any two classes a and b, their strict interiors cannot overlap
by construction because the same shared boundary is used with opposite signs.

The bank stores no exemplars, prototypes, Gaussian statistics, covariance,
replay data, alignment model, or phase schedule.  Its role is nevertheless the
same persistent role that a prototype bank plays in many exemplar-free CIL
systems: it is the historical decision reference reused for classification,
feature-space separation, replay selection, and continual preservation.  The
difference is the retained object.  Instead of one class centroid, this bank
retains the discriminative relation between every class pair.

A BoundaryCandidate contains only the new pairwise boundaries introduced in the
current phase.  Those exact boundaries are the geometry committed at phase
finalization; there is no train-time/deployment geometry mismatch.
"""

from dataclasses import dataclass
import math
from itertools import combinations
from numbers import Integral, Real
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _as_int(value: object, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
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


def _class_ids(values: Iterable[int], *, name: str = "class_ids") -> tuple[int, ...]:
    ids = tuple(_as_int(value, name) for value in values)
    if not ids:
        raise ValueError(f"{name} cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} must be unique")
    if any(class_id < 0 for class_id in ids):
        raise ValueError(f"{name} must be non-negative")
    return ids


def _pair(a: int, b: int) -> tuple[int, int]:
    left = _as_int(a, "class_id")
    right = _as_int(b, "class_id")
    if left == right:
        raise ValueError("a pair must contain two different classes")
    return (left, right) if left < right else (right, left)


def _required_pairs(class_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ids = _class_ids(class_ids)
    return tuple(_pair(a, b) for a, b in combinations(ids, 2))


@dataclass(frozen=True)
class GeometryScore:
    """Per-class signed decision-cell energies in ``class_ids`` column order."""

    energy: Tensor
    class_ids: Tensor

    @property
    def inside(self) -> Tensor:
        return self.energy <= 0


@dataclass(frozen=True)
class PairwiseGeometryValues:
    """Signed affine values for explicit unordered class pairs.

    ``values[:, p]`` is h_ab(z) for ``pair_ids[p] == [a,b]`` with ``a < b``.
    Positive values therefore favor the left class and negative values favor
    the right class.  Normals are unit length and offsets use the same signed
    distance scale.

    This is the common geometry representation used by separation, replay
    selection, diagnostics, and any other operation that needs explicit
    class-pair decision information.
    """

    values: Tensor
    pair_ids: Tensor
    normals: Tensor
    offsets: Tensor


@dataclass(frozen=True)
class ClassBoundaryResponse:
    """Class-oriented historical decision coordinates.

    For row i with label c, ``margins[i]`` contains s_cj(z_i) against every
    requested rival j != c.  ``rival_class_ids[i]`` identifies the corresponding
    rival columns.  Positive margins favor the sample's own class.

    This response is the prototype-equivalent historical reference of the
    geometry architecture: it can be cached at phase start and preserved while
    the backbone evolves, without forcing the complete feature vector to remain
    fixed.
    """

    margins: Tensor
    rival_class_ids: Tensor
    labels: Tensor
    class_ids: Tensor


class BoundaryCandidate(nn.Module):
    """Trainable current-phase pairwise boundaries.

    ``new_class_ids`` are the classes introduced in the current phase.
    ``pair_ids`` contains every new-new and old-new pair required to extend the
    committed bank to all classes visible in the phase.
    """

    def __init__(
        self,
        *,
        new_class_ids: Sequence[int],
        pair_ids: Tensor,
        normals: Tensor,
        offsets: Tensor,
    ) -> None:
        super().__init__()
        new_ids = _class_ids(new_class_ids, name="new_class_ids")
        pairs = torch.as_tensor(pair_ids)
        normal = torch.as_tensor(normals)
        offset = torch.as_tensor(offsets)

        if pairs.ndim != 2 or pairs.size(1) != 2 or pairs.dtype != torch.long:
            raise ValueError("pair_ids must be int64 [P,2]")
        if normal.ndim != 2 or normal.size(0) != pairs.size(0) or normal.size(1) <= 0:
            raise ValueError("normals must be [P,D]")
        if offset.shape != (pairs.size(0),):
            raise ValueError("offsets must be [P]")
        if normal.device != offset.device or normal.device != pairs.device:
            raise ValueError("candidate tensors must share a device")
        if normal.dtype != offset.dtype or normal.dtype not in (torch.float32, torch.float64):
            raise ValueError("candidate geometry must use float32 or float64")
        if not bool(torch.isfinite(normal).all()) or not bool(torch.isfinite(offset).all()):
            raise ValueError("candidate geometry contains NaN/Inf")
        if pairs.numel() and bool((pairs[:, 0] >= pairs[:, 1]).any()):
            raise ValueError("each pair_id row must satisfy left < right")

        pair_list = [tuple(map(int, row)) for row in pairs.detach().cpu().tolist()]
        if len(pair_list) != len(set(pair_list)):
            raise ValueError("candidate pair_ids must be unique")
        if pairs.size(0) and bool((torch.linalg.vector_norm(normal, dim=1) <= 0).any()):
            raise ValueError("candidate boundary normals must be non-zero")

        self.register_buffer(
            "new_class_id_tensor",
            torch.tensor(new_ids, device=normal.device, dtype=torch.long),
        )
        self.register_buffer("pair_ids", pairs.detach().clone())
        self.raw_normals = nn.Parameter(normal.detach().clone())
        self.offsets = nn.Parameter(offset.detach().clone())

    @property
    def new_class_ids(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.new_class_id_tensor.detach().cpu().tolist())

    @property
    def representation_dim(self) -> int:
        return int(self.raw_normals.size(1))

    @property
    def normals(self) -> Tensor:
        norms = torch.linalg.vector_norm(self.raw_normals, dim=1, keepdim=True)
        if self.raw_normals.size(0) and bool((norms <= 0).any()):
            raise RuntimeError("candidate contains a zero boundary normal")
        if not bool(torch.isfinite(norms).all()):
            raise RuntimeError("candidate normal norm is NaN/Inf")
        return self.raw_normals / norms

    def validate_state(self) -> bool:
        _class_ids(self.new_class_ids, name="new_class_ids")
        if self.pair_ids.ndim != 2 or self.pair_ids.size(1) != 2:
            raise RuntimeError("candidate pair_ids must be [P,2]")
        if self.pair_ids.dtype != torch.long:
            raise RuntimeError("candidate pair_ids must be int64")
        if self.raw_normals.ndim != 2 or self.raw_normals.size(0) != self.pair_ids.size(0):
            raise RuntimeError("candidate normals are misaligned with pair_ids")
        if self.offsets.shape != (self.pair_ids.size(0),):
            raise RuntimeError("candidate offsets are misaligned with pair_ids")
        if not bool(torch.isfinite(self.raw_normals).all()) or not bool(
            torch.isfinite(self.offsets).all()
        ):
            raise RuntimeError("candidate geometry contains NaN/Inf")
        if self.pair_ids.size(0) and bool((self.pair_ids[:, 0] >= self.pair_ids[:, 1]).any()):
            raise RuntimeError("candidate pair rows must satisfy left < right")
        rows = [tuple(map(int, row)) for row in self.pair_ids.detach().cpu().tolist()]
        if len(rows) != len(set(rows)):
            raise RuntimeError("candidate pair_ids are duplicated")
        _ = self.normals
        return True


class BoundaryGeometryBank(nn.Module):
    """Persistent pairwise decision geometry for all committed classes."""

    def __init__(
        self,
        representation_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.representation_dim = _as_int(representation_dim, "representation_dim")
        if self.representation_dim <= 0:
            raise ValueError("representation_dim must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("geometry dtype must be float32 or float64")
        dev = torch.device(device)
        self.register_buffer("class_ids", torch.empty(0, device=dev, dtype=torch.long))
        self.register_buffer("pair_ids", torch.empty((0, 2), device=dev, dtype=torch.long))
        self.register_buffer(
            "normals",
            torch.empty((0, self.representation_dim), device=dev, dtype=dtype),
        )
        self.register_buffer("offsets", torch.empty(0, device=dev, dtype=dtype))

    @property
    def device(self) -> torch.device:
        return self.normals.device

    @property
    def dtype(self) -> torch.dtype:
        return self.normals.dtype

    def __len__(self) -> int:
        return int(self.class_ids.numel())

    @property
    def pair_count(self) -> int:
        return int(self.pair_ids.size(0))

    def _coordinates(self, value: Tensor, *, name: str = "coordinates") -> Tensor:
        z = torch.as_tensor(value)
        if z.ndim != 2 or z.size(0) == 0 or z.size(1) != self.representation_dim:
            raise ValueError(
                f"{name} must be non-empty [N,{self.representation_dim}]"
            )
        if z.device != self.device:
            raise ValueError(f"{name} must be on geometry device {self.device}")
        if z.dtype != self.dtype:
            raise ValueError(f"{name} must use geometry dtype {self.dtype}")
        if not bool(torch.isfinite(z).all()):
            raise ValueError(f"{name} contains NaN/Inf")
        return z

    def _labels(self, value: Tensor, *, rows: int) -> Tensor:
        y = torch.as_tensor(value)
        if y.device != self.device:
            raise ValueError("labels must share geometry device")
        y = y.flatten()
        if y.numel() != rows or y.dtype == torch.bool or y.is_complex():
            raise ValueError("labels are invalid or row-misaligned")
        if torch.is_floating_point(y):
            if not bool(torch.isfinite(y).all()) or not bool(y.eq(y.round()).all()):
                raise ValueError("labels must contain finite integer class IDs")
        y = y.to(dtype=torch.long)
        if bool((y < 0).any()):
            raise ValueError("labels must be non-negative")
        return y

    @staticmethod
    def _separator_initialization(
        left_coordinates: Tensor,
        right_coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Deterministically initialize one separating plane from phase data.

        The class means are used only to initialize the trainable boundary; they
        are neither stored nor used by the classifier.  If the means coincide,
        a non-zero observed cross-class difference is used.  If the two classes
        are feature-identical, a discriminative boundary cannot be initialized
        and the caller fails loudly rather than inventing a direction.
        """
        if left_coordinates.ndim != 2 or right_coordinates.ndim != 2:
            raise ValueError("pair coordinates must be rank two")
        if left_coordinates.size(0) == 0 or right_coordinates.size(0) == 0:
            raise ValueError("both classes must have at least one coordinate")
        if left_coordinates.size(1) != right_coordinates.size(1):
            raise ValueError("pair coordinate dimensions disagree")

        left_mean = left_coordinates.mean(dim=0)
        right_mean = right_coordinates.mean(dim=0)
        delta = left_mean - right_mean
        delta_norm = torch.linalg.vector_norm(delta)

        if float(delta_norm.item()) == 0.0:
            replacement: Optional[Tensor] = None
            for row in left_coordinates:
                differences = row.unsqueeze(0) - right_coordinates
                norms = torch.linalg.vector_norm(differences, dim=1)
                index = int(norms.argmax().item())
                if float(norms[index].item()) > 0.0:
                    replacement = differences[index]
                    break
            if replacement is None:
                raise RuntimeError(
                    "two classes are feature-identical; a pairwise boundary cannot "
                    "be initialized"
                )
            delta = replacement
            delta_norm = torch.linalg.vector_norm(delta)

        normal = delta / delta_norm
        midpoint = 0.5 * (left_mean + right_mean)
        offset = -torch.dot(normal, midpoint)

        if not bool(torch.isfinite(normal).all()) or not bool(torch.isfinite(offset)):
            raise RuntimeError("boundary initialization produced NaN/Inf")
        return normal, offset

    def initialize_candidate(
        self,
        coordinates: Tensor,
        labels: Tensor,
        new_class_ids: Sequence[int],
    ) -> BoundaryCandidate:
        """Create the exact set of pairwise boundaries introduced by a phase.

        The provided coordinates/labels must contain every class participating
        in a new pair.  At base phase this is the real base training split.  At
        incremental phases old-new initialization therefore requires old-class
        replay evidence; the geometry bank never fabricates missing old data.
        """
        z = self._coordinates(coordinates)
        y = self._labels(labels, rows=z.size(0))
        new_ids = _class_ids(new_class_ids, name="new_class_ids")
        committed = tuple(int(v) for v in self.class_ids.detach().cpu().tolist())
        overlap = sorted(set(committed).intersection(new_ids))
        if overlap:
            raise ValueError(f"new classes are already committed: {overlap}")

        visible = committed + new_ids
        if len(visible) != len(set(visible)):
            raise RuntimeError("visible class IDs are not unique")
        candidate_pairs = [
            pair for pair in _required_pairs(visible)
            if pair[0] in new_ids or pair[1] in new_ids
        ]

        present = set(int(v) for v in y.unique().detach().cpu().tolist())
        required_classes = set(v for pair in candidate_pairs for v in pair)
        missing = sorted(required_classes - present)
        if missing:
            raise RuntimeError(
                "candidate initialization lacks feature evidence for classes "
                f"{missing}"
            )

        normals: list[Tensor] = []
        offsets: list[Tensor] = []
        for left, right in candidate_pairs:
            normal, offset = self._separator_initialization(
                z[y.eq(left)],
                z[y.eq(right)],
            )
            normals.append(normal)
            offsets.append(offset)

        if candidate_pairs:
            pair_tensor = torch.tensor(candidate_pairs, device=self.device, dtype=torch.long)
            normal_tensor = torch.stack(normals, dim=0)
            offset_tensor = torch.stack(offsets, dim=0)
        else:
            pair_tensor = torch.empty((0, 2), device=self.device, dtype=torch.long)
            normal_tensor = torch.empty(
                (0, self.representation_dim), device=self.device, dtype=self.dtype
            )
            offset_tensor = torch.empty(0, device=self.device, dtype=self.dtype)

        candidate = BoundaryCandidate(
            new_class_ids=new_ids,
            pair_ids=pair_tensor,
            normals=normal_tensor,
            offsets=offset_tensor,
        )
        self._validate_candidate(candidate)
        return candidate

    def _validate_candidate(self, candidate: BoundaryCandidate) -> None:
        if not isinstance(candidate, BoundaryCandidate):
            raise TypeError("candidate must be BoundaryCandidate")
        candidate.validate_state()
        if candidate.representation_dim != self.representation_dim:
            raise ValueError("candidate representation dimension is incompatible")
        if candidate.raw_normals.device != self.device or candidate.raw_normals.dtype != self.dtype:
            raise ValueError("candidate geometry device/dtype is incompatible")

        committed = tuple(int(v) for v in self.class_ids.detach().cpu().tolist())
        new_ids = candidate.new_class_ids
        if set(committed).intersection(new_ids):
            raise ValueError("candidate contains already committed classes")
        visible = committed + new_ids
        expected = {
            pair for pair in _required_pairs(visible)
            if pair[0] in new_ids or pair[1] in new_ids
        }
        actual = {
            tuple(map(int, row)) for row in candidate.pair_ids.detach().cpu().tolist()
        }
        if actual != expected:
            raise ValueError(
                "candidate pair set does not exactly match the new phase boundary set"
            )

        committed_pairs = {
            tuple(map(int, row)) for row in self.pair_ids.detach().cpu().tolist()
        }
        duplicate = committed_pairs.intersection(actual)
        if duplicate:
            raise ValueError(f"candidate duplicates committed boundaries: {sorted(duplicate)}")

    def commit_candidate(self, candidate: BoundaryCandidate) -> None:
        """Persist the exact learned current-phase boundaries."""
        self.validate_bank_state()
        self._validate_candidate(candidate)
        with torch.no_grad():
            new_ids = candidate.new_class_id_tensor.detach().clone()
            new_pairs = candidate.pair_ids.detach().clone()
            new_normals = candidate.normals.detach().clone()
            new_offsets = candidate.offsets.detach().clone()

            self.class_ids = torch.cat([self.class_ids, new_ids], dim=0)
            self.pair_ids = torch.cat([self.pair_ids, new_pairs], dim=0)
            self.normals = torch.cat([self.normals, new_normals], dim=0)
            self.offsets = torch.cat([self.offsets, new_offsets], dim=0)
        self.validate_bank_state()

    def _merged_state(
        self,
        candidate: Optional[BoundaryCandidate],
    ) -> tuple[tuple[int, ...], Tensor, Tensor, Tensor]:
        committed = tuple(int(v) for v in self.class_ids.detach().cpu().tolist())
        if candidate is None:
            return committed, self.pair_ids, self.normals, self.offsets

        self._validate_candidate(candidate)
        visible = committed + candidate.new_class_ids
        pairs = torch.cat([self.pair_ids, candidate.pair_ids], dim=0)
        normals = torch.cat([self.normals, candidate.normals], dim=0)
        offsets = torch.cat([self.offsets, candidate.offsets], dim=0)
        return visible, pairs, normals, offsets

    @staticmethod
    def _canonical_pair_sequence(
        pair_ids: Sequence[Sequence[int]] | Tensor,
        *,
        name: str = "pair_ids",
    ) -> tuple[tuple[int, int], ...]:
        value = torch.as_tensor(pair_ids, device="cpu")
        if value.ndim != 2 or value.size(1) != 2:
            raise ValueError(f"{name} must have shape [P,2]")
        if value.size(0) == 0:
            raise ValueError(f"{name} cannot be empty")
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"{name} must contain integer class IDs")
        if torch.is_floating_point(value):
            if not bool(torch.isfinite(value).all()) or not bool(
                value.eq(value.round()).all()
            ):
                raise ValueError(f"{name} must contain finite integer class IDs")
        rows = tuple(
            _pair(int(row[0].item()), int(row[1].item()))
            for row in value
        )
        if len(rows) != len(set(rows)):
            raise ValueError(f"{name} must not contain duplicate pairs")
        return rows

    def pair_values(
        self,
        coordinates: Tensor,
        *,
        pair_ids: Sequence[Sequence[int]] | Tensor,
        candidate: Optional[BoundaryCandidate] = None,
    ) -> PairwiseGeometryValues:
        """Evaluate explicit pairwise decision coordinates.

        This is the primary low-level geometry API.  The bank owns the affine
        boundary mathematics; losses and replay code consume these values rather
        than reimplementing normal/offset orientation themselves.

        The returned value for pair (a,b), a<b, is

            h_ab(z) = n_ab^T z + q_ab,

        on a true signed-distance scale.  Candidate boundaries may be included
        during training; committed-only calls pass ``candidate=None``.
        """
        z = self._coordinates(coordinates)
        requested_pairs = self._canonical_pair_sequence(pair_ids)
        visible, all_pairs, all_normals, all_offsets = self._merged_state(candidate)

        required_classes = {class_id for pair in requested_pairs for class_id in pair}
        missing_classes = sorted(required_classes - set(visible))
        if missing_classes:
            raise ValueError(
                f"requested pair classes are not represented: {missing_classes}"
            )

        pair_rows = {
            tuple(map(int, row)): index
            for index, row in enumerate(all_pairs.detach().cpu().tolist())
        }
        missing_pairs = [pair for pair in requested_pairs if pair not in pair_rows]
        if missing_pairs:
            raise RuntimeError(
                "geometry does not contain requested pairwise boundaries: "
                f"{missing_pairs}"
            )

        indices = torch.tensor(
            [pair_rows[pair] for pair in requested_pairs],
            device=self.device,
            dtype=torch.long,
        )
        normals = all_normals.index_select(0, indices)
        offsets = all_offsets.index_select(0, indices)
        norms = torch.linalg.vector_norm(normals, dim=1, keepdim=True)
        if bool((norms <= 0).any()) or not bool(torch.isfinite(norms).all()):
            raise RuntimeError("geometry contains an invalid boundary normal")

        # Normalize the complete affine function.  Candidate normals are already
        # unit length, but scaling both n and q keeps signed-distance semantics
        # correct for any valid loaded state as well.
        unit_normals = normals / norms
        unit_offsets = offsets / norms.squeeze(1)
        values = z @ unit_normals.transpose(0, 1) + unit_offsets

        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("pairwise geometry evaluation produced NaN/Inf")

        return PairwiseGeometryValues(
            values=values,
            pair_ids=torch.tensor(
                requested_pairs,
                device=self.device,
                dtype=torch.long,
            ),
            normals=unit_normals,
            offsets=unit_offsets,
        )

    def class_boundary_response(
        self,
        coordinates: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> ClassBoundaryResponse:
        """Return the incident decision coordinates of each labeled sample.

        For a sample of class c this returns exactly the C-1 oriented margins
        s_cj(z), j != c, in the requested class set.  These coordinates are the
        appropriate historical reference for continual preservation: they retain
        how a class relates to its rivals without storing an exemplar prototype
        or constraining representation directions that do not affect old
        pairwise decisions.
        """
        z = self._coordinates(coordinates)
        y = self._labels(labels, rows=z.size(0))
        requested = _class_ids(class_ids, name="class_ids")
        if len(requested) < 2:
            raise ValueError("class_boundary_response requires at least two classes")

        requested_set = set(requested)
        observed = set(int(v) for v in y.unique().detach().cpu().tolist())
        outside = sorted(observed - requested_set)
        if outside:
            raise ValueError(f"labels contain classes outside class_ids: {outside}")

        pairs = _required_pairs(requested)
        pair_geometry = self.pair_values(
            z,
            pair_ids=pairs,
            candidate=candidate,
        )
        pair_rows = {
            pair: index for index, pair in enumerate(pairs)
        }

        margins = torch.empty(
            (z.size(0), len(requested) - 1),
            device=self.device,
            dtype=self.dtype,
        )
        rivals = torch.empty(
            (z.size(0), len(requested) - 1),
            device=self.device,
            dtype=torch.long,
        )

        for class_id in requested:
            mask = y.eq(class_id)
            if not bool(mask.any()):
                continue
            rival_ids = tuple(rival for rival in requested if rival != class_id)
            columns = torch.tensor(
                [pair_rows[_pair(class_id, rival)] for rival in rival_ids],
                device=self.device,
                dtype=torch.long,
            )
            block = pair_geometry.values.index_select(1, columns)[mask]
            signs = torch.tensor(
                [1.0 if class_id < rival else -1.0 for rival in rival_ids],
                device=self.device,
                dtype=self.dtype,
            )
            margins[mask] = block * signs.unsqueeze(0)
            rivals[mask] = torch.tensor(
                rival_ids,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(mask.sum().item()), -1)

        if not bool(torch.isfinite(margins).all()):
            raise RuntimeError("class boundary response produced NaN/Inf")
        return ClassBoundaryResponse(
            margins=margins,
            rival_class_ids=rivals,
            labels=y,
            class_ids=torch.tensor(
                requested,
                device=self.device,
                dtype=torch.long,
            ),
        )

    def true_pair_margins(
        self,
        coordinates: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> Tensor:
        """Convenience view used by pairwise separation/training objectives."""
        return self.class_boundary_response(
            coordinates,
            labels,
            class_ids=class_ids,
            candidate=candidate,
        ).margins

    def score(
        self,
        coordinates: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> GeometryScore:
        z = self._coordinates(coordinates)
        requested = _class_ids(class_ids, name="class_ids")
        visible, _, _, _ = self._merged_state(candidate)

        if not set(requested).issubset(set(visible)):
            missing = sorted(set(requested) - set(visible))
            raise ValueError(f"requested classes are not represented: {missing}")

        class_id_tensor = torch.tensor(requested, device=self.device, dtype=torch.long)
        if len(requested) == 1:
            return GeometryScore(
                energy=torch.zeros((z.size(0), 1), device=self.device, dtype=self.dtype),
                class_ids=class_id_tensor,
            )

        required = _required_pairs(requested)
        pair_geometry = self.pair_values(
            z,
            pair_ids=required,
            candidate=candidate,
        )
        required_index = {
            pair: index for index, pair in enumerate(required)
        }

        energies: list[Tensor] = []
        for class_id in requested:
            oriented: list[Tensor] = []
            for rival in requested:
                if rival == class_id:
                    continue
                pair = _pair(class_id, rival)
                distance = pair_geometry.values[:, required_index[pair]]
                oriented.append(distance if class_id < rival else -distance)
            min_margin = torch.stack(oriented, dim=1).amin(dim=1)
            energies.append(-min_margin)

        energy = torch.stack(energies, dim=1)
        if not bool(torch.isfinite(energy).all()):
            raise RuntimeError("geometry scoring produced NaN/Inf")
        return GeometryScore(energy=energy, class_ids=class_id_tensor)

    def predict(
        self,
        coordinates: Tensor,
        *,
        class_ids: Sequence[int],
        candidate: Optional[BoundaryCandidate] = None,
    ) -> Tensor:
        score = self.score(coordinates, class_ids=class_ids, candidate=candidate)
        return score.class_ids.index_select(0, score.energy.argmin(dim=1))

    def closest_boundary_point(
        self,
        coordinates: Tensor,
        *,
        class_id: int,
    ) -> Dict[str, Tensor]:
        """Project each query onto the active boundary of one committed class.

        This is reporting/replay support only.  It does not update geometry.
        """
        z = self._coordinates(coordinates)
        requested = _as_int(class_id, "class_id")
        committed = tuple(int(v) for v in self.class_ids.detach().cpu().tolist())
        if requested not in committed:
            raise ValueError(f"class {requested} is not committed")
        if len(committed) < 2:
            raise RuntimeError("a class needs at least one rival boundary")

        pair_rows = {
            tuple(map(int, row)): index
            for index, row in enumerate(self.pair_ids.detach().cpu().tolist())
        }
        margins: list[Tensor] = []
        oriented_normals: list[Tensor] = []
        rivals: list[int] = []
        for rival in committed:
            if rival == requested:
                continue
            pair = _pair(requested, rival)
            index = pair_rows[pair]
            normal = self.normals[index]
            norm = torch.linalg.vector_norm(normal)
            if float(norm.item()) <= 0.0:
                raise RuntimeError("geometry contains a zero boundary normal")
            normal = normal / norm
            distance = z @ normal + self.offsets[index]
            if requested < rival:
                margins.append(distance)
                oriented_normals.append(normal)
            else:
                margins.append(-distance)
                oriented_normals.append(-normal)
            rivals.append(rival)

        margin_matrix = torch.stack(margins, dim=1)
        active = margin_matrix.argmin(dim=1)
        normal_matrix = torch.stack(oriented_normals, dim=0)
        active_normals = normal_matrix.index_select(0, active)
        active_margin = margin_matrix.gather(1, active.unsqueeze(1)).squeeze(1)
        boundary = z - active_margin.unsqueeze(1) * active_normals
        rival_tensor = torch.tensor(rivals, device=self.device, dtype=torch.long)
        return {
            "boundary_point": boundary,
            "rival_class_id": rival_tensor.index_select(0, active),
            "signed_margin": active_margin,
        }

    def get_pair_geometry(
        self,
        left_class_id: int,
        right_class_id: int,
        *,
        candidate: Optional[BoundaryCandidate] = None,
    ) -> Dict[str, Tensor]:
        """Return one committed or current-candidate pairwise boundary."""
        pair = _pair(left_class_id, right_class_id)
        visible, pair_ids, normals, offsets = self._merged_state(candidate)
        if pair[0] not in visible or pair[1] not in visible:
            raise RuntimeError(f"pair {pair} contains a class outside visible geometry")

        rows = {
            tuple(map(int, row)): index
            for index, row in enumerate(pair_ids.detach().cpu().tolist())
        }
        if pair not in rows:
            raise RuntimeError(f"pair {pair} is not represented")
        index = rows[pair]
        normal = normals[index]
        norm = torch.linalg.vector_norm(normal)
        if float(norm.item()) <= 0.0 or not bool(torch.isfinite(norm)):
            raise RuntimeError("stored pair normal is invalid")
        return {
            "pair_ids": pair_ids[index].detach().clone(),
            "normal": (normal / norm).detach().clone(),
            "offset": (offsets[index] / norm).detach().clone(),
        }

    def validate_bank_state(self) -> bool:
        if self.class_ids.ndim != 1 or self.class_ids.dtype != torch.long:
            raise RuntimeError("class_ids must be rank-one int64")
        if self.class_ids.numel() != self.class_ids.unique().numel():
            raise RuntimeError("committed class_ids must be unique")
        if bool((self.class_ids < 0).any()):
            raise RuntimeError("committed class_ids must be non-negative")
        if self.pair_ids.ndim != 2 or self.pair_ids.shape[1:] != (2,):
            raise RuntimeError("pair_ids must be [P,2]")
        if self.pair_ids.dtype != torch.long:
            raise RuntimeError("pair_ids must be int64")
        if self.normals.shape != (self.pair_ids.size(0), self.representation_dim):
            raise RuntimeError("stored normals have invalid shape")
        if self.offsets.shape != (self.pair_ids.size(0),):
            raise RuntimeError("stored offsets have invalid shape")
        if not bool(torch.isfinite(self.normals).all()) or not bool(torch.isfinite(self.offsets).all()):
            raise RuntimeError("stored geometry contains NaN/Inf")
        if self.pair_ids.size(0) and bool((self.pair_ids[:, 0] >= self.pair_ids[:, 1]).any()):
            raise RuntimeError("stored pair rows must satisfy left < right")
        rows = [tuple(map(int, row)) for row in self.pair_ids.detach().cpu().tolist()]
        if len(rows) != len(set(rows)):
            raise RuntimeError("stored pair_ids are duplicated")
        if self.normals.size(0) and bool((torch.linalg.vector_norm(self.normals, dim=1) <= 0).any()):
            raise RuntimeError("stored boundary normals must be non-zero")

        committed = tuple(int(v) for v in self.class_ids.detach().cpu().tolist())
        expected = set(_required_pairs(committed)) if len(committed) >= 2 else set()
        if set(rows) != expected:
            raise RuntimeError("stored pairwise geometry is incomplete or inconsistent")
        return True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Class and pair counts grow over phases; resize buffers to checkpoint
        # shapes before delegating to nn.Module's strict loader.
        for name in ("class_ids", "pair_ids", "normals", "offsets"):
            saved = state_dict.get(prefix + name)
            if torch.is_tensor(saved):
                current = getattr(self, name)
                if saved.dtype != current.dtype:
                    error_msgs.append(f"{prefix}{name} has incompatible dtype")
                    return
                if name == "normals":
                    if saved.ndim != 2 or saved.size(1) != self.representation_dim:
                        error_msgs.append(f"{prefix}{name} has incompatible shape")
                        return
                elif name == "pair_ids":
                    if saved.ndim != 2 or saved.size(1) != 2:
                        error_msgs.append(f"{prefix}{name} has incompatible shape")
                        return
                elif saved.ndim != 1:
                    error_msgs.append(f"{prefix}{name} has incompatible shape")
                    return
                setattr(self, name, torch.empty_like(saved, device=current.device))

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        try:
            self.validate_bank_state()
        except (RuntimeError, ValueError) as exc:
            error_msgs.append(f"{prefix}invalid boundary geometry state: {exc}")


__all__ = [
    "BoundaryCandidate",
    "BoundaryGeometryBank",
    "ClassBoundaryResponse",
    "GeometryScore",
    "PairwiseGeometryValues",
]





