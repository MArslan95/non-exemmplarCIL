
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-12
_RAW_MEMORY_NAMES = {
    "raw_samples", "raw_patches", "old_samples", "old_patches", "stored_samples",
    "stored_patches", "feature_memory", "old_features", "stored_features",
    "exemplars", "exemplar_memory", "memory_features", "memory_patches",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", "none", "null", ""}:
            return False
    return bool(value)


def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for v in values:
        c = int(v)
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def orthonormalize_columns(basis: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal-column version of [D, R]."""
    if not torch.is_tensor(basis):
        raise TypeError("basis must be a torch.Tensor")
    if basis.dim() != 2:
        raise ValueError(f"basis must be [D,R], got {tuple(basis.shape)}")
    if basis.numel() == 0 or basis.size(1) == 0:
        return basis
    basis = torch.nan_to_num(basis.float(), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        q, _ = torch.linalg.qr(basis, mode="reduced")
    except RuntimeError:
        u, _, _ = torch.linalg.svd(basis, full_matrices=False)
        q = u
    return q[:, : basis.size(1)]


def complete_orthonormal_basis(active_basis: torch.Tensor, rank: int) -> torch.Tensor:
    """Complete [D, q] active basis to [D, rank] without using data exemplars."""
    if not torch.is_tensor(active_basis):
        raise TypeError("active_basis must be a torch.Tensor")
    if active_basis.dim() != 2:
        raise ValueError(f"active_basis must be [D,q], got {tuple(active_basis.shape)}")

    d = int(active_basis.size(0))
    rank = int(max(0, min(int(rank), d)))
    device, dtype = active_basis.device, active_basis.dtype
    if rank == 0:
        return torch.empty((d, 0), device=device, dtype=dtype)

    cols: List[torch.Tensor] = []
    if active_basis.numel() > 0 and active_basis.size(1) > 0:
        q = orthonormalize_columns(active_basis[:, :rank].to(dtype=torch.float32)).to(device=device, dtype=dtype)
        for j in range(q.size(1)):
            if q[:, j].norm() > 1e-7:
                cols.append(q[:, j])

    eye = torch.eye(d, device=device, dtype=dtype)
    for j in range(d):
        v = eye[:, j].clone()
        for u in cols:
            v = v - torch.dot(v, u) * u
        n = v.norm()
        if n > 1e-7:
            cols.append(v / n)
        if len(cols) >= rank:
            break

    if len(cols) < rank:
        raise RuntimeError(f"Could not complete orthonormal basis to rank={rank} in dim={d}.")
    return torch.stack(cols[:rank], dim=1)


class GeometryBank(nn.Module):
    """
    Exemplar-free HSI class-geometry memory.

    Stored memory is limited to compact class descriptors:
      - class mean mu_c
      - low-rank basis U_c
      - eigenvalues lambda_c
      - residual variance sigma_c^2
      - optional spectral prototype / band importance
      - sample count, reliability, phase_created, frozen status

    The bank never stores raw HSI patches, raw old samples, or old feature batches.
    Feature tensors may be passed temporarily to `extract_geometry` or
    `add_or_update_class_geometry(features=...)`, but are reduced immediately into
    descriptors and discarded.
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        device: Union[str, torch.device] = "cpu",
        variance_floor: float = 1e-4,
        variance_shrinkage: float = 0.10,
        max_variance_ratio: float = 50.0,
        min_reliability: float = 0.05,
        reliability_sample_alpha: float = 20.0,
        rank_energy_threshold: float = 0.95,
        rank_eigen_ratio_threshold: float = 1e-3,
        min_active_rank: int = 1,
        small_class_rank_threshold_1: int = 30,
        small_class_rank_threshold_2: int = 80,
        small_class_rank_threshold_3: int = 150,
        small_class_rank_cap_1: int = 1,
        small_class_rank_cap_2: int = 3,
        small_class_rank_cap_3: int = 4,
        small_class_extra_shrinkage: float = 0.35,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        self.rank = int(max(0, min(int(rank), self.d_model)))
        self.variance_floor = float(max(float(variance_floor), 1e-12))
        self.variance_shrinkage = float(max(0.0, min(float(variance_shrinkage), 1.0)))
        self.max_variance_ratio = float(max(float(max_variance_ratio), 1.0))
        self.min_reliability = float(max(0.0, min(float(min_reliability), 1.0)))
        self.reliability_sample_alpha = float(max(float(reliability_sample_alpha), 1.0))
        self.rank_energy_threshold = float(max(0.50, min(float(rank_energy_threshold), 0.999)))
        self.rank_eigen_ratio_threshold = float(max(float(rank_eigen_ratio_threshold), 0.0))
        self.min_active_rank = int(max(0, min(int(min_active_rank), self.rank)))

        t1 = int(max(2, small_class_rank_threshold_1))
        t2 = int(max(t1 + 1, small_class_rank_threshold_2))
        t3 = int(max(t2 + 1, small_class_rank_threshold_3))
        self.small_class_rank_thresholds = (t1, t2, t3)
        self.small_class_rank_caps = (
            int(max(1, min(int(small_class_rank_cap_1), self.rank))) if self.rank > 0 else 0,
            int(max(1, min(int(small_class_rank_cap_2), self.rank))) if self.rank > 0 else 0,
            int(max(1, min(int(small_class_rank_cap_3), self.rank))) if self.rank > 0 else 0,
        )
        self.small_class_extra_shrinkage = float(max(0.0, min(float(small_class_extra_shrinkage), 0.85)))

        dev = torch.device(device)
        self.register_buffer("means", torch.empty((0, self.d_model), device=dev))
        self.register_buffer("bases", torch.empty((0, self.d_model, self.rank), device=dev))
        self.register_buffer("eigvals", torch.empty((0, self.rank), device=dev))
        self.register_buffer("res_vars", torch.empty((0,), device=dev))
        self.register_buffer("active_ranks", torch.empty((0,), dtype=torch.long, device=dev))
        self.register_buffer("sample_counts", torch.empty((0,), device=dev))
        self.register_buffer("reliability", torch.empty((0,), device=dev))
        self.register_buffer("feature_reliability", torch.empty((0,), device=dev))
        self.register_buffer("band_importances", torch.empty((0, 0), device=dev))
        self.register_buffer("band_reliability", torch.empty((0,), device=dev))
        self.register_buffer("spectral_prototypes", torch.empty((0, 0), device=dev))
        self.register_buffer("spectral_reliability", torch.empty((0,), device=dev))
        self.register_buffer("phase_created", torch.empty((0,), dtype=torch.long, device=dev))
        self.register_buffer("frozen_class_mask", torch.empty((0,), dtype=torch.bool, device=dev))
        self.register_buffer("_band_dim", torch.tensor(0, dtype=torch.long, device=dev))

        # Compatibility empty tensors expected by older classifier/trainer code.
        self.register_buffer("spectral_curve_means", torch.empty((0, 0), device=dev))
        self.register_buffer("spectral_curve_vars", torch.empty((0, 0), device=dev))
        self.register_buffer("spectral_curve_d1", torch.empty((0, 0), device=dev))
        self.register_buffer("spectral_curve_d2", torch.empty((0, 0), device=dev))
        self.register_buffer("spectral_shape_reliability", torch.empty((0,), device=dev))

    # ------------------------------------------------------------------
    # basic properties / validation
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.means.device

    def __len__(self) -> int:
        return int(self.means.size(0))

    @property
    def feature_dim(self) -> int:
        """Compatibility alias used by trainer/classifier code.

        The GeometryBank feature dimension is the canonical projected z-space
        dimension.  Keep this as an alias to d_model so no module silently
        creates a different feature-space contract.
        """
        return int(self.d_model)

    def _dtype(self) -> torch.dtype:
        return self.means.dtype if self.means.numel() > 0 else torch.float32

    @property
    def resvars(self) -> torch.Tensor:
        return self.res_vars

    @resvars.setter
    def resvars(self, value: torch.Tensor) -> None:
        self.res_vars = value

    @property
    def spectral_protos(self) -> torch.Tensor:
        return self.spectral_prototypes

    @spectral_protos.setter
    def spectral_protos(self, value: torch.Tensor) -> None:
        self.spectral_prototypes = value

    def _assert_no_raw_memory_attrs(self) -> None:
        bad = [name for name in self.__dict__.keys() if name.lower() in _RAW_MEMORY_NAMES]
        # Also check registered buffers/parameters by name.
        bad.extend([name for name in self._buffers.keys() if name.lower() in _RAW_MEMORY_NAMES])
        bad.extend([name for name in self._parameters.keys() if name.lower() in _RAW_MEMORY_NAMES])
        if bad:
            raise RuntimeError(
                "GeometryBank contains forbidden exemplar-like memory fields: "
                f"{sorted(set(bad))}. Store only compact class statistics."
            )

    def _valid_class_id(self, class_id: int, *, existing: bool = True) -> int:
        c = int(class_id)
        if c < 0:
            raise IndexError(f"class_id must be non-negative, got {c}")
        if existing and c >= len(self):
            raise IndexError(f"class_id={c} out of range for bank size {len(self)}")
        return c

    def _sample_count_rank_cap(self, n: int) -> int:
        n = int(max(0, n))
        if self.rank <= 0 or n <= 1:
            return 0
        t1, t2, t3 = self.small_class_rank_thresholds
        c1, c2, c3 = self.small_class_rank_caps
        if n < t1:
            cap = c1
        elif n < t2:
            cap = c2
        elif n < t3:
            cap = c3
        else:
            cap = self.rank
        return int(max(0, min(self.rank, n - 1, cap)))

    def _adaptive_shrinkage(self, n: int) -> float:
        n = int(max(1, n))
        extra = self.small_class_extra_shrinkage * min(1.0, self.reliability_sample_alpha / float(n))
        return float(max(0.0, min(self.variance_shrinkage + extra, 0.90)))

    def _ensure_band_dim(self, band_dim: int, dtype: Optional[torch.dtype] = None) -> None:
        band_dim = int(max(0, band_dim))
        if band_dim <= 0:
            return
        dtype = dtype or self._dtype()
        cur = int(self._band_dim.item())
        if cur > 0 and cur != band_dim:
            raise ValueError(f"band/spectral dimension mismatch: existing={cur}, requested={band_dim}")
        if cur == band_dim:
            return
        C = len(self)
        self._band_dim = torch.tensor(band_dim, dtype=torch.long, device=self.device)
        self.band_importances = torch.full((C, band_dim), 1.0 / float(band_dim), device=self.device, dtype=dtype)
        self.band_reliability = torch.full((C,), self.min_reliability, device=self.device, dtype=dtype)
        self.spectral_prototypes = torch.zeros((C, band_dim), device=self.device, dtype=dtype)
        self.spectral_reliability = torch.full((C,), self.min_reliability, device=self.device, dtype=dtype)
        self.spectral_curve_means = torch.zeros((C, band_dim), device=self.device, dtype=dtype)
        self.spectral_curve_vars = torch.full((C, band_dim), self.variance_floor, device=self.device, dtype=dtype)
        self.spectral_curve_d1 = torch.zeros((C, max(band_dim - 1, 0)), device=self.device, dtype=dtype)
        self.spectral_curve_d2 = torch.zeros((C, max(band_dim - 2, 0)), device=self.device, dtype=dtype)
        self.spectral_shape_reliability = torch.full((C,), self.min_reliability, device=self.device, dtype=dtype)

    def _prepare_mean(self, mean: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(mean, device=self.device, dtype=self._dtype()).flatten()
        if t.numel() != self.d_model:
            raise ValueError(f"mean must have shape [{self.d_model}], got {tuple(t.shape)}")
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_basis(self, basis: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(basis, device=self.device, dtype=self._dtype())
        if t.dim() != 2:
            raise ValueError(f"basis must be [D,R], got {tuple(t.shape)}")
        if t.size(0) == self.rank and t.size(1) == self.d_model:
            t = t.t()
        if t.size(0) != self.d_model:
            raise ValueError(f"basis first dimension must be {self.d_model}, got {t.size(0)}")
        if t.size(1) > self.rank:
            t = t[:, : self.rank]
        if t.size(1) < self.rank:
            # Complete only from nonzero columns.
            norms = t.norm(dim=0) if t.numel() > 0 else torch.empty((0,), device=self.device)
            active = t[:, norms > 1e-8] if norms.numel() and bool((norms > 1e-8).any().item()) else torch.empty((self.d_model, 0), device=self.device, dtype=self._dtype())
            return complete_orthonormal_basis(active, self.rank)
        return complete_orthonormal_basis(t, self.rank)

    def _prepare_eigvals(self, eigvals: torch.Tensor, fallback: Union[float, torch.Tensor]) -> torch.Tensor:
        fb = torch.as_tensor(fallback, device=self.device, dtype=self._dtype()).reshape(()).clamp_min(self.variance_floor)
        t = torch.as_tensor(eigvals, device=self.device, dtype=self._dtype()).flatten()
        if t.numel() > self.rank:
            t = t[: self.rank]
        elif t.numel() < self.rank:
            t = torch.cat([t, torch.full((self.rank - t.numel(),), float(fb.item()), device=self.device, dtype=self._dtype())])
        t = torch.nan_to_num(t, nan=float(fb.item()), posinf=float(fb.item()), neginf=float(fb.item()))
        t = t.clamp_min(self.variance_floor)
        # Active dimensions are expected sorted; full vector placeholders may include residual values.
        return t

    def _prepare_band_vector(self, value: Optional[torch.Tensor], band_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self._dtype()
        if band_dim <= 0:
            return torch.empty((0,), device=self.device, dtype=dtype), torch.tensor(self.min_reliability, device=self.device, dtype=dtype)
        if value is None or torch.as_tensor(value).numel() == 0:
            b = torch.full((band_dim,), 1.0 / float(band_dim), device=self.device, dtype=dtype)
        else:
            raw = torch.as_tensor(value, device=self.device, dtype=dtype).flatten()
            if raw.numel() != band_dim:
                raise ValueError(f"band/spectral vector must have {band_dim} values, got {raw.numel()}")
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            if bool((raw < 0).any().item()):
                b = torch.softmax(raw, dim=0)
            else:
                b = raw.clamp_min(0.0)
                b = b / b.sum().clamp_min(1e-8) if b.sum() > 1e-8 else torch.full((band_dim,), 1.0 / float(band_dim), device=self.device, dtype=dtype)
        b = b.clamp_min(0.0)
        b = b / b.sum().clamp_min(1e-8)

        entropy = -(b.clamp_min(1e-12) * b.clamp_min(1e-12).log()).sum()
        max_entropy = torch.log(torch.tensor(float(max(band_dim, 2)), device=self.device, dtype=dtype)).clamp_min(1e-12)
        rel = (1.0 - entropy / max_entropy).clamp(self.min_reliability, 1.0)
        return b.detach(), rel.detach()

    def _spectral_shape_from_proto(self, proto: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype = self._dtype()
        band_dim = int(proto.numel())
        if band_dim <= 0:
            z = torch.empty((0,), device=self.device, dtype=dtype)
            return {"mean": z, "var": z, "d1": z, "d2": z, "reliability": torch.tensor(self.min_reliability, device=self.device, dtype=dtype)}
        p = torch.nan_to_num(proto.to(device=self.device, dtype=dtype).flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        d1 = p[1:] - p[:-1] if band_dim >= 2 else torch.empty((0,), device=self.device, dtype=dtype)
        d2 = d1[1:] - d1[:-1] if band_dim >= 3 else torch.empty((0,), device=self.device, dtype=dtype)
        e_curve = p.pow(2).mean().sqrt()
        e_der = d1.pow(2).mean().sqrt() if d1.numel() else torch.tensor(0.0, device=self.device, dtype=dtype)
        rel = (e_der / (e_curve + e_der + 1e-8)).clamp(self.min_reliability, 1.0)
        return {
            "mean": p.detach(),
            "var": torch.full((band_dim,), self.variance_floor, device=self.device, dtype=dtype),
            "d1": d1.detach(),
            "d2": d2.detach(),
            "reliability": rel.detach(),
        }

    @torch.no_grad()
    def assert_bank_valid(self, seen_classes: Optional[Iterable[int]] = None, *, strict: bool = True) -> Dict[str, Any]:
        errors: List[str] = []
        self._assert_no_raw_memory_attrs()
        C = len(self)
        band_dim = int(self._band_dim.item())

        expected = {
            "means": (C, self.d_model),
            "bases": (C, self.d_model, self.rank),
            "eigvals": (C, self.rank),
            "res_vars": (C,),
            "active_ranks": (C,),
            "sample_counts": (C,),
            "reliability": (C,),
            "feature_reliability": (C,),
            "band_importances": (C, band_dim),
            "band_reliability": (C,),
            "spectral_prototypes": (C, band_dim),
            "spectral_reliability": (C,),
            "phase_created": (C,),
            "frozen_class_mask": (C,),
        }
        for name, shape in expected.items():
            value = getattr(self, name, None)
            if not torch.is_tensor(value):
                errors.append(f"{name} is not a tensor")
            elif tuple(value.shape) != tuple(shape):
                errors.append(f"{name} shape mismatch: got {tuple(value.shape)}, expected {shape}")

        finite_names = [
            "means", "bases", "eigvals", "res_vars", "sample_counts",
            "reliability", "feature_reliability", "band_importances",
            "band_reliability", "spectral_prototypes", "spectral_reliability",
        ]
        for name in finite_names:
            value = getattr(self, name, None)
            if torch.is_tensor(value) and value.numel() > 0 and not torch.isfinite(value).all():
                errors.append(f"{name} contains NaN/Inf")

        if self.eigvals.numel() > 0 and bool((self.eigvals < self.variance_floor).any().item()):
            errors.append("eigvals contain values below variance_floor")
        if self.res_vars.numel() > 0 and bool((self.res_vars < self.variance_floor).any().item()):
            errors.append("res_vars contain values below variance_floor")
        if self.sample_counts.numel() > 0 and bool((self.sample_counts < 0).any().item()):
            errors.append("sample_counts contain negative values")

        if C > 0 and self.bases.numel() > 0 and self.rank > 0:
            eye = torch.eye(self.rank, device=self.device, dtype=self._dtype())
            gram = torch.bmm(self.bases.transpose(1, 2), self.bases)
            valid_rows = self.sample_counts > 0 if self.sample_counts.numel() == C else torch.zeros((C,), device=self.device, dtype=torch.bool)
            if bool(valid_rows.any().item()):
                max_ortho_err = (gram[valid_rows] - eye).abs().max()
                if float(max_ortho_err.detach().cpu().item()) > 1e-3:
                    errors.append(f"basis columns are not orthonormal; max_err={float(max_ortho_err):.4e}")

        for c in range(C):
            n = int(float(self.sample_counts[c].detach().cpu().item())) if self.sample_counts.numel() > c else 0
            r = int(self.active_ranks[c].detach().cpu().item()) if self.active_ranks.numel() > c else 0
            cap = self._sample_count_rank_cap(n)
            if n <= 0 and r != 0:
                errors.append(f"class {c}: active_rank must be 0 when sample_count=0, got {r}")
            if n > 0 and not (0 <= r <= cap):
                errors.append(f"class {c}: active_rank={r} exceeds cap={cap} for n={n}")
            if r > 1:
                e = self.eigvals[c, :r]
                if bool((e[:-1] + 1e-8 < e[1:]).any().item()):
                    errors.append(f"class {c}: active eigvals must be sorted descending")

        if band_dim > 0 and C > 0:
            valid_rows = self.sample_counts > 0
            row_sum = self.band_importances.sum(dim=1)
            bad_band = valid_rows & ((row_sum - 1.0).abs() > 1e-3)
            if bool(bad_band.any().item()):
                errors.append(f"band_importances rows must sum to 1 for valid classes; bad={torch.nonzero(bad_band).flatten().tolist()[:10]}")

        if seen_classes is not None:
            ids = _ordered_unique_ints(seen_classes)
            valid = self.get_valid_mask()
            missing = [c for c in ids if c < 0 or c >= C or valid.numel() <= c or not bool(valid[c].item())]
            if missing:
                errors.append(f"GeometryBank missing valid rows for seen classes: {missing}")

        result = {"ok": len(errors) == 0, "num_rows": C, "band_dim": band_dim, "errors": errors}
        if strict and errors:
            raise RuntimeError("GeometryBank validity check failed: " + "; ".join(errors))
        return result

    validate_consistency = assert_bank_valid

    # ------------------------------------------------------------------
    # geometry extraction / row writes
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_low_rank_geometry(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        if data is None or not torch.is_tensor(data):
            raise TypeError("features/data must be a tensor")
        if data.dim() != 2 or data.size(1) != self.d_model:
            raise ValueError(f"features must be [N,{self.d_model}], got {tuple(data.shape)}")
        data = torch.nan_to_num(data.to(device=self.device, dtype=self._dtype()), nan=0.0, posinf=0.0, neginf=0.0)
        n, d = int(data.size(0)), int(data.size(1))
        if n <= 0:
            raise ValueError("cannot extract geometry from zero samples")

        mean = data.mean(dim=0)
        centered = data - mean.view(1, -1)
        total_var = centered.pow(2).sum(dim=1).mean().clamp_min(self.variance_floor)
        avg_var = (total_var / float(max(d, 1))).clamp_min(self.variance_floor)
        rank_cap = self._sample_count_rank_cap(n)

        if n <= 1 or rank_cap <= 0 or self.rank <= 0:
            active_rank = 0
            active_basis = torch.empty((d, 0), device=self.device, dtype=self._dtype())
            active_eigvals = torch.empty((0,), device=self.device, dtype=self._dtype())
            residual_total_var = total_var
            res_var = avg_var
        else:
            try:
                _, s, vh = torch.linalg.svd(centered, full_matrices=False)
                raw_eig = (s.pow(2) / float(max(n - 1, 1))).clamp_min(0.0)
                raw_basis = vh.t().contiguous()
            except RuntimeError:
                cov = centered.t().matmul(centered) / float(max(n - 1, 1))
                evals, evecs = torch.linalg.eigh(cov)
                order = torch.argsort(evals, descending=True)
                raw_eig = evals.index_select(0, order).clamp_min(0.0)
                raw_basis = evecs.index_select(1, order).contiguous()

            max_possible = min(self.rank, rank_cap, int(raw_eig.numel()), d)
            if max_possible <= 0 or float(raw_eig[:max_possible].sum().detach().cpu().item()) <= 0:
                active_rank = 0
                active_basis = torch.empty((d, 0), device=self.device, dtype=self._dtype())
                active_eigvals = torch.empty((0,), device=self.device, dtype=self._dtype())
                residual_total_var = total_var
                res_var = avg_var
            else:
                vals = raw_eig[:max_possible]
                cumulative = torch.cumsum(vals, dim=0) / vals.sum().clamp_min(_EPS)
                hit = (cumulative >= self.rank_energy_threshold).nonzero(as_tuple=False)
                energy_rank = int(hit[0].item()) + 1 if hit.numel() else max_possible
                strength_rank = int((vals / vals[0].clamp_min(_EPS) >= self.rank_eigen_ratio_threshold).sum().item())
                min_rank = min(self.min_active_rank, max_possible)
                active_rank = int(max(min_rank, min(energy_rank, strength_rank, max_possible)))
                active_basis = orthonormalize_columns(raw_basis[:, :active_rank]).to(device=self.device, dtype=self._dtype())
                active_eigvals = vals[:active_rank].to(device=self.device, dtype=self._dtype())

                shrink = self._adaptive_shrinkage(n)
                active_eigvals = (1.0 - shrink) * active_eigvals + shrink * avg_var
                active_eigvals = active_eigvals.clamp(
                    min=self.variance_floor,
                    max=float((avg_var * self.max_variance_ratio).detach().cpu().item()),
                )
                residual = centered - centered.matmul(active_basis).matmul(active_basis.t())
                residual_total_var = residual.pow(2).sum(dim=1).mean().clamp_min(self.variance_floor)
                res_var = (residual_total_var / float(max(d - active_rank, 1))).clamp_min(self.variance_floor)
                res_var = ((1.0 - 0.5 * shrink) * res_var + (0.5 * shrink) * avg_var).clamp_min(self.variance_floor)

        basis = complete_orthonormal_basis(active_basis, self.rank)
        eigvals = torch.full((self.rank,), float(res_var.detach().cpu().item()), device=self.device, dtype=self._dtype())
        if active_rank > 0:
            # Active eigvals already descending from SVD/eigh order.
            eigvals[:active_rank] = active_eigvals[:active_rank].clamp_min(self.variance_floor)
        sample_rel = torch.tensor(float(n) / float(n + self.reliability_sample_alpha), device=self.device, dtype=self._dtype())
        compact_rel = (1.0 - residual_total_var / total_var.clamp_min(self.variance_floor)).clamp(self.min_reliability, 1.0)
        rank_rel = torch.tensor(
            self.min_reliability if rank_cap <= 0 else max(self.min_reliability, float(active_rank) / float(max(rank_cap, 1))),
            device=self.device,
            dtype=self._dtype(),
        )
        feature_rel = (0.45 * sample_rel + 0.30 * compact_rel + 0.25 * rank_rel).clamp(self.min_reliability, 1.0)

        return {
            "mean": mean.detach(),
            "basis": basis.detach(),
            "eigvals": eigvals.detach(),
            "res_var": res_var.reshape(()).detach(),
            "active_rank": torch.tensor(active_rank, device=self.device, dtype=torch.long),
            "sample_count": torch.tensor(float(n), device=self.device, dtype=self._dtype()),
            "feature_reliability": feature_rel.detach(),
            "reliability": feature_rel.detach(),
        }

    @torch.no_grad()
    def extract_geometry(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        spectral_summary: Optional[torch.Tensor] = None,
        band_weights: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: bool = True,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Reduce temporary feature batches into class descriptors."""
        if features is None or labels is None:
            return {}
        features = torch.as_tensor(features, device=self.device, dtype=self._dtype())
        labels = torch.as_tensor(labels, device=self.device, dtype=torch.long).flatten()
        if features.dim() != 2 or features.size(1) != self.d_model:
            raise ValueError(f"features must be [N,{self.d_model}], got {tuple(features.shape)}")
        if labels.numel() != features.size(0):
            raise ValueError(f"labels/features mismatch: {labels.numel()} vs {features.size(0)}")
        if labels.numel() == 0:
            return {}
        if int(labels.min().item()) < 0:
            raise ValueError(f"negative class labels are forbidden in GeometryBank: {torch.unique(labels).tolist()}")

        spec = None
        spec_physical = bool(spectral_summary_is_physical)
        if spectral_summary is not None and torch.as_tensor(spectral_summary).numel() > 0:
            spec = torch.as_tensor(spectral_summary, device=self.device, dtype=self._dtype())
            if spec.dim() != 2 or spec.size(0) != features.size(0):
                raise ValueError(f"spectral_summary must be [N,S] aligned with features, got {tuple(spec.shape)}")
            self._ensure_band_dim(int(spec.size(1)), self._dtype())

        bands = None
        if band_weights is not None and torch.as_tensor(band_weights).numel() > 0:
            bands = torch.as_tensor(band_weights, device=self.device, dtype=self._dtype())
            if bands.dim() != 2 or bands.size(0) != features.size(0):
                raise ValueError(f"band_weights must be [N,S] aligned with features, got {tuple(bands.shape)}")
            self._ensure_band_dim(int(bands.size(1)), self._dtype())

        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for cls_t in torch.unique(labels, sorted=True):
            cls = int(cls_t.item())
            mask = labels == cls_t
            row = self._extract_low_rank_geometry(features[mask])
            band_vec = None
            spectral_proto = None
            spectral_rel = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype())

            if spec is not None:
                spectral_proto = spec[mask].mean(dim=0).detach()
                if spec_physical:
                    spectral_rel = torch.tensor(float(mask.sum().item()) / float(mask.sum().item() + self.reliability_sample_alpha), device=self.device, dtype=self._dtype()).clamp(self.min_reliability, 1.0)
                else:
                    spectral_rel = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype())
                band_vec = spectral_proto.abs()
            if bands is not None:
                b = bands[mask].mean(dim=0).detach()
                band_vec = b if band_vec is None else 0.5 * band_vec + 0.5 * b

            if band_vec is not None:
                b, br = self._prepare_band_vector(band_vec, int(self._band_dim.item()))
                row["band_importance"] = b
                row["band_reliability"] = br
            if spectral_proto is not None:
                row["spectral_prototype"] = spectral_proto
                row["spectral_reliability"] = spectral_rel
            out[cls] = row
        return out

    @torch.no_grad()
    def ensure_class_count(self, count: int, spectral_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
        count = int(max(0, count))
        dtype = dtype or self._dtype()
        if spectral_dim > 0:
            self._ensure_band_dim(int(spectral_dim), dtype)
        while len(self) < count:
            self._append_empty_row(dtype=dtype)

    ensure_num_classes = ensure_class_count

    @torch.no_grad()
    def _append_empty_row(self, dtype: Optional[torch.dtype] = None) -> None:
        dtype = dtype or self._dtype()
        band_dim = int(self._band_dim.item())
        c = len(self)
        mean = torch.zeros((1, self.d_model), device=self.device, dtype=dtype)
        basis = complete_orthonormal_basis(torch.empty((self.d_model, 0), device=self.device, dtype=dtype), self.rank).unsqueeze(0)
        eig = torch.full((1, self.rank), self.variance_floor, device=self.device, dtype=dtype)
        rv = torch.full((1,), self.variance_floor, device=self.device, dtype=dtype)
        zero_l = torch.zeros((1,), device=self.device, dtype=torch.long)
        rel = torch.full((1,), self.min_reliability, device=self.device, dtype=dtype)
        count0 = torch.zeros((1,), device=self.device, dtype=dtype)

        self.means = torch.cat([self.means, mean], dim=0)
        self.bases = torch.cat([self.bases, basis], dim=0)
        self.eigvals = torch.cat([self.eigvals, eig], dim=0)
        self.res_vars = torch.cat([self.res_vars, rv], dim=0)
        self.active_ranks = torch.cat([self.active_ranks, zero_l], dim=0)
        self.sample_counts = torch.cat([self.sample_counts, count0], dim=0)
        self.reliability = torch.cat([self.reliability, rel], dim=0)
        self.feature_reliability = torch.cat([self.feature_reliability, rel.clone()], dim=0)
        self.band_reliability = torch.cat([self.band_reliability, rel.clone()], dim=0)
        self.spectral_reliability = torch.cat([self.spectral_reliability, rel.clone()], dim=0)
        self.phase_created = torch.cat([self.phase_created, torch.full((1,), -1, device=self.device, dtype=torch.long)], dim=0)
        self.frozen_class_mask = torch.cat([self.frozen_class_mask, torch.zeros((1,), device=self.device, dtype=torch.bool)], dim=0)

        if band_dim > 0:
            self.band_importances = torch.cat([self.band_importances, torch.full((1, band_dim), 1.0 / float(band_dim), device=self.device, dtype=dtype)], dim=0)
            self.spectral_prototypes = torch.cat([self.spectral_prototypes, torch.zeros((1, band_dim), device=self.device, dtype=dtype)], dim=0)
            self.spectral_curve_means = torch.cat([self.spectral_curve_means, torch.zeros((1, band_dim), device=self.device, dtype=dtype)], dim=0)
            self.spectral_curve_vars = torch.cat([self.spectral_curve_vars, torch.full((1, band_dim), self.variance_floor, device=self.device, dtype=dtype)], dim=0)
            self.spectral_curve_d1 = torch.cat([self.spectral_curve_d1, torch.zeros((1, max(band_dim - 1, 0)), device=self.device, dtype=dtype)], dim=0)
            self.spectral_curve_d2 = torch.cat([self.spectral_curve_d2, torch.zeros((1, max(band_dim - 2, 0)), device=self.device, dtype=dtype)], dim=0)
            self.spectral_shape_reliability = torch.cat([self.spectral_shape_reliability, rel.clone()], dim=0)
        else:
            self.band_importances = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
            self.spectral_prototypes = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
            self.spectral_curve_means = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
            self.spectral_curve_vars = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
            self.spectral_curve_d1 = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
            self.spectral_curve_d2 = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
            self.spectral_shape_reliability = torch.cat([self.spectral_shape_reliability, rel.clone()], dim=0)

    def _assert_update_allowed(self, class_id: int, allow_frozen_update: bool = False) -> None:
        c = self._valid_class_id(class_id, existing=False)
        if c < len(self) and self.frozen_class_mask.numel() == len(self) and bool(self.frozen_class_mask[c].item()):
            if not bool(allow_frozen_update):
                raise RuntimeError(
                    f"Refusing to overwrite frozen GeometryBank row {c}. "
                    "Old-class descriptors are immutable in the clean NECIL path."
                )

    @torch.no_grad()
    def add_or_update_class_geometry(
        self,
        class_id: int,
        *,
        features: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        basis: Optional[torch.Tensor] = None,
        eigvals: Optional[torch.Tensor] = None,
        res_var: Optional[torch.Tensor] = None,
        residual_variance: Optional[torch.Tensor] = None,
        spectral_prototype: Optional[torch.Tensor] = None,
        band_importance: Optional[torch.Tensor] = None,
        sample_count: Optional[Union[int, float, torch.Tensor]] = None,
        active_rank: Optional[Union[int, torch.Tensor]] = None,
        reliability: Optional[Union[float, torch.Tensor]] = None,
        feature_reliability: Optional[Union[float, torch.Tensor]] = None,
        band_reliability: Optional[Union[float, torch.Tensor]] = None,
        spectral_reliability: Optional[Union[float, torch.Tensor]] = None,
        phase_created: int = -1,
        freeze: bool = False,
        allow_frozen_update: bool = False,
    ) -> None:
        """Add or update exactly one global class row.

        Use features only for temporary descriptor extraction. The features are
        not stored after this method returns.
        """
        c = self._valid_class_id(class_id, existing=False)
        self._assert_update_allowed(c, allow_frozen_update=allow_frozen_update)

        if features is not None:
            geom = self._extract_low_rank_geometry(features)
            mean = geom["mean"]
            basis = geom["basis"]
            eigvals = geom["eigvals"]
            res_var = geom["res_var"]
            sample_count = geom["sample_count"] if sample_count is None else sample_count
            active_rank = geom["active_rank"] if active_rank is None else active_rank
            feature_reliability = geom["feature_reliability"] if feature_reliability is None else feature_reliability
            reliability = geom["reliability"] if reliability is None else reliability

        rv = res_var if res_var is not None else residual_variance
        if mean is None or basis is None or eigvals is None or rv is None:
            raise ValueError("mean, basis, eigvals, and res_var are required unless features are provided")

        # spectral/band capacity first
        band_dim = 0
        if spectral_prototype is not None and torch.as_tensor(spectral_prototype).numel() > 0:
            band_dim = int(torch.as_tensor(spectral_prototype).numel())
        if band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
            band_dim = max(band_dim, int(torch.as_tensor(band_importance).numel()))
        if band_dim > 0:
            self._ensure_band_dim(band_dim, self._dtype())

        self.ensure_class_count(c + 1, spectral_dim=band_dim, dtype=self._dtype())

        mean_t = self._prepare_mean(mean)
        basis_t = self._prepare_basis(basis)
        rv_t = torch.as_tensor(rv, device=self.device, dtype=self._dtype()).reshape(()).clamp_min(self.variance_floor)
        eig_t = self._prepare_eigvals(eigvals, rv_t)

        count_t = torch.tensor(0.0, device=self.device, dtype=self._dtype()) if sample_count is None else torch.as_tensor(sample_count, device=self.device, dtype=self._dtype()).reshape(()).clamp_min(0.0)
        n_i = int(float(count_t.detach().cpu().item()))
        cap = self._sample_count_rank_cap(n_i)
        if active_rank is None:
            ar_t = torch.tensor(cap, device=self.device, dtype=torch.long)
        else:
            ar_t = torch.as_tensor(active_rank, device=self.device, dtype=torch.long).reshape(()).clamp(0, cap)
        if n_i <= 0:
            ar_t = torch.tensor(0, device=self.device, dtype=torch.long)

        feat_rel_t = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype()) if feature_reliability is None else torch.as_tensor(feature_reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)
        rel_t = feat_rel_t if reliability is None else torch.as_tensor(reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)

        band_dim_now = int(self._band_dim.item())
        band_t, band_rel_t = self._prepare_band_vector(band_importance, band_dim_now)
        if band_reliability is not None:
            band_rel_t = torch.as_tensor(band_reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)

        spec_t = torch.empty((0,), device=self.device, dtype=self._dtype())
        spec_rel_t = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype())
        if band_dim_now > 0:
            if spectral_prototype is None or torch.as_tensor(spectral_prototype).numel() == 0:
                spec_t = torch.zeros((band_dim_now,), device=self.device, dtype=self._dtype())
            else:
                spec_t = torch.as_tensor(spectral_prototype, device=self.device, dtype=self._dtype()).flatten()
                if spec_t.numel() != band_dim_now:
                    raise ValueError(f"spectral_prototype must have {band_dim_now} values, got {spec_t.numel()}")
                spec_t = torch.nan_to_num(spec_t, nan=0.0, posinf=0.0, neginf=0.0)
            if spectral_reliability is not None:
                spec_rel_t = torch.as_tensor(spectral_reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)
            else:
                spec_rel_t = torch.tensor(self.min_reliability if spectral_prototype is None else max(self.min_reliability, float(rel_t.item())), device=self.device, dtype=self._dtype())

        self.means[c] = mean_t
        self.bases[c] = basis_t
        self.eigvals[c] = eig_t
        self.res_vars[c] = rv_t
        self.active_ranks[c] = ar_t
        self.sample_counts[c] = count_t
        self.feature_reliability[c] = feat_rel_t
        self.reliability[c] = rel_t
        self.band_reliability[c] = band_rel_t
        self.spectral_reliability[c] = spec_rel_t
        self.phase_created[c] = int(phase_created)
        if band_dim_now > 0:
            self.band_importances[c] = band_t
            self.spectral_prototypes[c] = spec_t
            shape = self._spectral_shape_from_proto(spec_t)
            self.spectral_curve_means[c] = shape["mean"]
            self.spectral_curve_vars[c] = shape["var"]
            if self.spectral_curve_d1.numel() > 0:
                self.spectral_curve_d1[c] = shape["d1"]
            if self.spectral_curve_d2.numel() > 0:
                self.spectral_curve_d2[c] = shape["d2"]
            self.spectral_shape_reliability[c] = shape["reliability"]

        if bool(freeze):
            self.frozen_class_mask[c] = True

        self.assert_bank_valid(strict=True)

    # Compatibility wrappers.
    @torch.no_grad()
    def add_class(self, mean, basis, eigvals, res_var, **kwargs: Any) -> None:
        self.add_or_update_class_geometry(len(self), mean=mean, basis=basis, eigvals=eigvals, res_var=res_var, **kwargs)

    @torch.no_grad()
    def update_class(self, cls_id, mean, basis, eigvals, res_var, allow_frozen_update: bool = False, **kwargs: Any) -> None:
        self.add_or_update_class_geometry(int(cls_id), mean=mean, basis=basis, eigvals=eigvals, res_var=res_var, allow_frozen_update=allow_frozen_update, **kwargs)

    @torch.no_grad()
    def update_class_geometry(self, class_id, mean, basis, eigvals, resvar=None, res_var=None, allow_frozen_update: bool = False, **kwargs: Any) -> None:
        rv = res_var if res_var is not None else resvar
        self.add_or_update_class_geometry(int(class_id), mean=mean, basis=basis, eigvals=eigvals, res_var=rv, allow_frozen_update=allow_frozen_update, **kwargs)

    # ------------------------------------------------------------------
    # freeze / access / bank views
    # ------------------------------------------------------------------
    @torch.no_grad()
    def freeze_classes(self, class_ids: Iterable[int]) -> None:
        self.ensure_class_count(max([int(c) for c in class_ids], default=-1) + 1)
        for c in _ordered_unique_ints(class_ids):
            self._valid_class_id(c, existing=True)
            self.frozen_class_mask[c] = True

    @torch.no_grad()
    def freeze_classes_up_to(self, count: int) -> None:
        count = int(max(0, count))
        self.ensure_class_count(count)
        self.frozen_class_mask[:count] = True

    @torch.no_grad()
    def unfreeze_all_classes(self) -> None:
        self.frozen_class_mask = torch.zeros((len(self),), device=self.device, dtype=torch.bool)

    def get_valid_mask(self) -> torch.Tensor:
        C = len(self)
        if C == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)
        finite = (
            torch.isfinite(self.means).all(dim=1)
            & torch.isfinite(self.bases).flatten(1).all(dim=1)
            & torch.isfinite(self.eigvals).all(dim=1)
            & torch.isfinite(self.res_vars)
            & torch.isfinite(self.sample_counts)
            & torch.isfinite(self.reliability)
        )
        rank_ok = torch.zeros((C,), device=self.device, dtype=torch.bool)
        for c in range(C):
            n = int(float(self.sample_counts[c].detach().cpu().item()))
            r = int(self.active_ranks[c].detach().cpu().item())
            rank_ok[c] = n > 0 and 0 <= r <= self._sample_count_rank_cap(n)
        return finite & rank_ok & (self.sample_counts > 0)

    def get_variances(self) -> torch.Tensor:
        if len(self) == 0:
            return torch.empty((0, self.rank + 1), device=self.device, dtype=self._dtype())
        return torch.cat([self.eigvals, self.res_vars.view(-1, 1)], dim=1)

    def get_bank(self) -> Dict[str, torch.Tensor]:
        valid = self.get_valid_mask()
        return {
            "means": self.means,
            "bases": self.bases,
            "raw_bases": self.bases,
            "eigvals": self.eigvals,
            "res_vars": self.res_vars,
            "resvars": self.res_vars,
            "variances": self.get_variances(),
            "active_ranks": self.active_ranks,
            "sample_counts": self.sample_counts,
            "reliability": self.reliability,
            "feature_reliability": self.feature_reliability,
            "band_importances": self.band_importances,
            "band_importance": self.band_importances,
            "band_reliability": self.band_reliability,
            "spectral_prototypes": self.spectral_prototypes,
            "spectral_protos": self.spectral_prototypes,
            "spectral_means": self.spectral_prototypes,
            "spectral_reliability": self.spectral_reliability,
            "phase_created": self.phase_created,
            "frozen_class_mask": self.frozen_class_mask,
            "valid_mask": valid,
            "spectral_dim": self._band_dim.clone(),
            "spectral_curve_means": self.spectral_curve_means,
            "spectral_curve_vars": self.spectral_curve_vars,
            "spectral_curve_d1": self.spectral_curve_d1,
            "spectral_curve_d2": self.spectral_curve_d2,
            "spectral_shape_reliability": self.spectral_shape_reliability,
        }

    get_subspace_bank = get_bank

    def get_seen_class_bank(self, seen_classes: Iterable[int]) -> Dict[str, torch.Tensor]:
        ids = _ordered_unique_ints(seen_classes)
        self.assert_bank_valid(ids, strict=True)
        idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        bank = self.get_bank()
        out: Dict[str, torch.Tensor] = {"class_ids": idx}
        for key, value in bank.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == len(self):
                out[key] = value.index_select(0, idx)
            elif torch.is_tensor(value):
                out[key] = value
        return out

    def get_class_geometry(self, class_id: int) -> Dict[str, torch.Tensor]:
        c = self._valid_class_id(class_id, existing=True)
        if self.sample_counts.numel() <= c or float(self.sample_counts[c].item()) <= 0:
            raise RuntimeError(f"class {c} has no valid geometry row")
        return {
            "class_id": torch.tensor(c, device=self.device, dtype=torch.long),
            "mean": self.means[c].detach().clone(),
            "basis": self.bases[c].detach().clone(),
            "eigvals": self.eigvals[c].detach().clone(),
            "res_var": self.res_vars[c].detach().clone(),
            "active_rank": self.active_ranks[c].detach().clone(),
            "sample_count": self.sample_counts[c].detach().clone(),
            "reliability": self.reliability[c].detach().clone(),
            "band_importance": self.band_importances[c].detach().clone() if self.band_importances.numel() else torch.empty((0,), device=self.device),
            "spectral_prototype": self.spectral_prototypes[c].detach().clone() if self.spectral_prototypes.numel() else torch.empty((0,), device=self.device),
            "phase_created": self.phase_created[c].detach().clone(),
            "frozen": self.frozen_class_mask[c].detach().clone(),
        }


    # ------------------------------------------------------------------
    # phase-0 handoff / row immutability helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def snapshot_rows(self, class_ids: Iterable[int]) -> Dict[str, torch.Tensor]:
        """Snapshot selected rows for immutability checks.

        This preserves incremental components: it does not alter replay,
        candidate insertion, diagnostics, or transport-ablation methods.  It only
        gives trainers a way to prove that frozen old/base rows were not changed.
        """
        ids = _ordered_unique_ints(class_ids)
        if not ids:
            return {
                "class_ids": torch.empty((0,), device=self.device, dtype=torch.long),
                "means": torch.empty((0, self.d_model), device=self.device, dtype=self._dtype()),
                "bases": torch.empty((0, self.d_model, self.rank), device=self.device, dtype=self._dtype()),
                "eigvals": torch.empty((0, self.rank), device=self.device, dtype=self._dtype()),
                "res_vars": torch.empty((0,), device=self.device, dtype=self._dtype()),
                "active_ranks": torch.empty((0,), device=self.device, dtype=torch.long),
                "sample_counts": torch.empty((0,), device=self.device, dtype=self._dtype()),
                "reliability": torch.empty((0,), device=self.device, dtype=self._dtype()),
                "frozen_class_mask": torch.empty((0,), device=self.device, dtype=torch.bool),
            }
        for c in ids:
            self._valid_class_id(c, existing=True)
        idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        snap: Dict[str, torch.Tensor] = {
            "class_ids": idx.detach().clone(),
            "means": self.means.index_select(0, idx).detach().clone(),
            "bases": self.bases.index_select(0, idx).detach().clone(),
            "eigvals": self.eigvals.index_select(0, idx).detach().clone(),
            "res_vars": self.res_vars.index_select(0, idx).detach().clone(),
            "active_ranks": self.active_ranks.index_select(0, idx).detach().clone(),
            "sample_counts": self.sample_counts.index_select(0, idx).detach().clone(),
            "reliability": self.reliability.index_select(0, idx).detach().clone(),
            "feature_reliability": self.feature_reliability.index_select(0, idx).detach().clone(),
            "phase_created": self.phase_created.index_select(0, idx).detach().clone(),
            "frozen_class_mask": self.frozen_class_mask.index_select(0, idx).detach().clone(),
        }
        if self.band_importances.numel() > 0 and self.band_importances.size(0) == len(self):
            snap["band_importances"] = self.band_importances.index_select(0, idx).detach().clone()
        if self.spectral_prototypes.numel() > 0 and self.spectral_prototypes.size(0) == len(self):
            snap["spectral_prototypes"] = self.spectral_prototypes.index_select(0, idx).detach().clone()
        return snap

    @torch.no_grad()
    def assert_rows_unchanged(
        self,
        snapshot: Dict[str, torch.Tensor],
        class_ids: Optional[Iterable[int]] = None,
        context: str = "GeometryBank",
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        check_frozen_mask: bool = True,
    ) -> None:
        """Assert selected GeometryBank rows are bitwise/numerically unchanged.

        Use this after base handoff and later during incremental training.  It is
        intentionally strict for old/base rows, but it does not forbid new-row
        insertion in later phases.
        """
        if not isinstance(snapshot, dict) or "class_ids" not in snapshot:
            raise RuntimeError(f"{context}: invalid row snapshot.")
        snap_ids = [int(c) for c in torch.as_tensor(snapshot["class_ids"]).detach().cpu().view(-1).tolist()]
        ids = snap_ids if class_ids is None else _ordered_unique_ints(class_ids)
        if ids != snap_ids:
            missing = [c for c in ids if c not in snap_ids]
            if missing:
                raise RuntimeError(f"{context}: snapshot does not contain requested rows {missing}.")
        idx_map = {c: i for i, c in enumerate(snap_ids)}
        bank_idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        snap_idx = torch.as_tensor([idx_map[c] for c in ids], device=self.device, dtype=torch.long)

        checks = (
            "means", "bases", "eigvals", "res_vars", "active_ranks",
            "sample_counts", "reliability", "feature_reliability", "phase_created",
        )
        failures: List[str] = []
        for key in checks:
            if key not in snapshot or not hasattr(self, key):
                continue
            current = getattr(self, key).index_select(0, bank_idx)
            old = torch.as_tensor(snapshot[key], device=self.device, dtype=current.dtype).index_select(0, snap_idx)
            if current.dtype.is_floating_point:
                ok = torch.allclose(current, old, atol=float(atol), rtol=float(rtol))
                if not bool(ok):
                    diff = float((current - old).abs().max().detach().cpu().item())
                    failures.append(f"{key} changed max_abs={diff:.4e}")
            else:
                if not bool(torch.equal(current, old)):
                    failures.append(f"{key} changed")

        if check_frozen_mask and "frozen_class_mask" in snapshot:
            current_frozen = self.frozen_class_mask.index_select(0, bank_idx)
            old_frozen = torch.as_tensor(snapshot["frozen_class_mask"], device=self.device, dtype=torch.bool).index_select(0, snap_idx)
            if not bool(torch.equal(current_frozen, old_frozen)):
                failures.append("frozen_class_mask changed")

        if failures:
            raise RuntimeError(f"{context}: frozen GeometryBank rows changed: " + "; ".join(failures))

    @torch.no_grad()
    def assert_phase0_base_handoff_ready(
        self,
        base_class_ids: Iterable[int],
        *,
        freeze: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Validate the phase-0 GeometryBank contract without deleting incremental APIs.

        Contract:
          1. every base class has a valid compact geometry row;
          2. no future/non-base row is valid after the base phase;
          3. base rows are frozen when requested;
          4. the bank remains strict non-exemplar memory.
        """
        ids = _ordered_unique_ints(base_class_ids)
        if not ids:
            raise RuntimeError("Phase-0 handoff requires at least one base class id.")
        self.assert_bank_valid(seen_classes=ids, strict=True)
        valid = self.get_valid_mask()
        max_base = max(ids)
        future_valid: List[int] = []
        for c in range(len(self)):
            if c not in set(ids) and valid.numel() > c and bool(valid[c].item()):
                # Base phase must not accidentally build future rows.  This is
                # independent of incremental support; later phases may validly
                # create rows after phase 0.
                if c > max_base or c not in ids:
                    future_valid.append(int(c))
        errors: List[str] = []
        if future_valid:
            errors.append(f"future/non-base valid rows after phase 0: {future_valid}")
        if bool(freeze):
            self.freeze_classes(ids)
        if self.frozen_class_mask.numel() >= max(ids) + 1:
            not_frozen = [c for c in ids if not bool(self.frozen_class_mask[c].item())]
            if not_frozen:
                errors.append(f"base rows not frozen: {not_frozen}")
        diag = self.compute_geometry_diagnostics(seen_classes=ids)
        result: Dict[str, Any] = {
            "ok": len(errors) == 0,
            "base_class_ids": ids,
            "num_base_classes": len(ids),
            "num_valid_rows": int(valid.sum().item()) if valid.numel() else 0,
            "future_valid_rows": future_valid,
            "errors": errors,
            "diagnostics": diag,
        }
        if strict and errors:
            raise RuntimeError("Phase-0 GeometryBank handoff failed: " + "; ".join(errors))
        return result

    def sample_geometry_replay(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Compatibility wrapper for incremental trainers.

        Keep incremental replay API intact while using the canonical bank sampler.
        """
        return self.sample_replay(*args, **kwargs)

    # ------------------------------------------------------------------
    # replay sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_replay(
        self,
        class_ids: Iterable[int],
        samples_per_class: Union[int, Mapping[int, int]] = 16,
        *,
        seen_classes: Optional[Iterable[int]] = None,
        label_to_local: Optional[Mapping[int, int]] = None,
        parallel_scale: float = 1.0,
        residual_scale: float = 0.25,
        reliability_gated: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample synthetic old features from stored low-rank Gaussian descriptors.

        Returns:
            features: [M, D]
            global_labels: [M] original sequential global class ids
            local_labels: [M] labels mapped to current seen-class classifier columns
        """
        ids = _ordered_unique_ints(class_ids)
        if label_to_local is None:
            if seen_classes is None:
                label_to_local = {int(c): int(c) for c in range(len(self))}
            else:
                label_to_local = {int(c): i for i, c in enumerate(_ordered_unique_ints(seen_classes))}
        else:
            label_to_local = {int(k): int(v) for k, v in dict(label_to_local).items()}

        valid = self.get_valid_mask()
        feats: List[torch.Tensor] = []
        labs_g: List[torch.Tensor] = []
        labs_l: List[torch.Tensor] = []
        for c in ids:
            self._valid_class_id(c, existing=True)
            if valid.numel() <= c or not bool(valid[c].item()):
                raise RuntimeError(f"Cannot sample replay: class {c} has no valid GeometryBank row")
            if c not in label_to_local:
                raise RuntimeError(f"Cannot sample replay: class {c} missing from local label mapping")
            if isinstance(samples_per_class, Mapping):
                n = int(max(0, samples_per_class.get(c, 0)))
            else:
                n = int(max(0, samples_per_class))
            if n <= 0:
                continue

            r = int(self.active_ranks[c].detach().cpu().item())
            eps = torch.zeros((n, self.d_model), device=self.device, dtype=self._dtype())
            gate = torch.tensor(1.0, device=self.device, dtype=self._dtype())
            if bool(reliability_gated):
                rho = self.reliability[c].clamp(self.min_reliability, 1.0)
                gate = 0.20 + 0.80 * rho

            if r > 0:
                z = torch.randn((n, r), device=self.device, dtype=self._dtype(), generator=generator)
                eig = self.eigvals[c, :r].clamp_min(self.variance_floor)
                eig = gate * eig + (1.0 - gate) * torch.tensor(self.variance_floor, device=self.device, dtype=self._dtype())
                eps = eps + z.mul(eig.sqrt().view(1, -1) * float(parallel_scale)).matmul(self.bases[c, :, :r].t())
            res = self.res_vars[c].clamp_min(self.variance_floor)
            res = gate * res + (1.0 - gate) * torch.tensor(self.variance_floor, device=self.device, dtype=self._dtype())
            eps = eps + torch.randn((n, self.d_model), device=self.device, dtype=self._dtype(), generator=generator) * res.sqrt() * float(residual_scale)

            x = self.means[c].view(1, -1) + eps
            feats.append(x)
            labs_g.append(torch.full((n,), c, device=self.device, dtype=torch.long))
            labs_l.append(torch.full((n,), int(label_to_local[c]), device=self.device, dtype=torch.long))

        if not feats:
            return {
                "features": torch.empty((0, self.d_model), device=self.device, dtype=self._dtype()),
                "global_labels": torch.empty((0,), device=self.device, dtype=torch.long),
                "local_labels": torch.empty((0,), device=self.device, dtype=torch.long),
            }
        features = torch.cat(feats, dim=0)
        global_labels = torch.cat(labs_g, dim=0)
        local_labels = torch.cat(labs_l, dim=0)
        if features.dim() != 2 or features.size(1) != self.d_model:
            raise RuntimeError(f"sampled features have wrong shape {tuple(features.shape)}")
        if global_labels.numel() != features.size(0) or local_labels.numel() != features.size(0):
            raise RuntimeError("sampled replay labels/features length mismatch")
        return {"features": features.detach(), "global_labels": global_labels.detach(), "local_labels": local_labels.detach()}

    @torch.no_grad()
    def sample_synthetic_features(
        self,
        class_ids: Optional[Iterable[int]] = None,
        samples_per_class: int = 16,
        parallel_scale: float = 1.0,
        residual_scale: float = 0.25,
        class_sample_counts: Optional[Union[Dict[int, int], torch.Tensor]] = None,
        reliability_gated: bool = True,
        **_: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = self.get_valid_mask()
        if class_ids is None:
            ids = [c for c in range(len(self)) if valid.numel() > c and bool(valid[c].item())]
        else:
            ids = _ordered_unique_ints(class_ids)

        counts: Union[int, Mapping[int, int]]
        if class_sample_counts is None:
            counts = int(samples_per_class)
        elif isinstance(class_sample_counts, dict):
            counts = {int(k): int(v) for k, v in class_sample_counts.items()}
        else:
            t = torch.as_tensor(class_sample_counts).flatten()
            counts = {int(c): int(float(t[int(c)].item())) if int(c) < t.numel() else 0 for c in ids}

        out = self.sample_replay(
            ids,
            samples_per_class=counts,
            seen_classes=list(range(len(self))),
            parallel_scale=parallel_scale,
            residual_scale=residual_scale,
            reliability_gated=reliability_gated,
        )
        return out["features"], out["global_labels"]

    # ------------------------------------------------------------------
    # diagnostics / overlap
    # ------------------------------------------------------------------
    @torch.no_grad()
    def pairwise_center_distance(self) -> torch.Tensor:
        C = len(self)
        if C == 0:
            return torch.empty((0, 0), device=self.device, dtype=self._dtype())
        dist = torch.cdist(self.means, self.means, p=2)
        valid = self.get_valid_mask()
        if valid.numel() == C:
            dist[~valid, :] = float("inf")
            dist[:, ~valid] = float("inf")
        dist[torch.eye(C, device=self.device, dtype=torch.bool)] = float("inf")
        return dist

    @torch.no_grad()
    def pairwise_subspace_overlap(self) -> torch.Tensor:
        C = len(self)
        out = torch.zeros((C, C), device=self.device, dtype=self._dtype())
        valid = self.get_valid_mask()
        for i in range(C):
            if valid.numel() == C and not bool(valid[i].item()):
                continue
            ri = int(self.active_ranks[i].item()) if self.active_ranks.numel() > i else 0
            if ri <= 0:
                continue
            Ui = self.bases[i, :, :ri]
            for j in range(C):
                if i == j or (valid.numel() == C and not bool(valid[j].item())):
                    continue
                rj = int(self.active_ranks[j].item()) if self.active_ranks.numel() > j else 0
                if rj <= 0:
                    continue
                Uj = self.bases[j, :, :rj]
                out[i, j] = (Ui.t().matmul(Uj).pow(2).sum() / float(max(min(ri, rj), 1))).clamp(0.0, 1.0)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def pairwise_band_similarity(self) -> torch.Tensor:
        C = len(self)
        out = torch.zeros((C, C), device=self.device, dtype=self._dtype())
        if self.band_importances.numel() == 0 or self.band_importances.size(1) == 0:
            return out
        b = self.band_importances.clamp_min(0.0)
        b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
        out = b.matmul(b.t()).clamp(0.0, 1.0)
        valid = self.get_valid_mask()
        if valid.numel() == C:
            out[~valid, :] = 0.0
            out[:, ~valid] = 0.0
        out[torch.eye(C, device=self.device, dtype=torch.bool)] = 0.0
        return out

    @torch.no_grad()
    def pairwise_spectral_similarity(self) -> torch.Tensor:
        C = len(self)
        out = torch.zeros((C, C), device=self.device, dtype=self._dtype())
        if self.spectral_prototypes.numel() == 0 or self.spectral_prototypes.size(1) == 0:
            return out
        s = self.spectral_prototypes
        s = s - s.mean(dim=1, keepdim=True)
        s = s / s.norm(dim=1, keepdim=True).clamp_min(1e-8)
        out = s.matmul(s.t()).clamp(0.0, 1.0)
        valid = self.get_valid_mask()
        if valid.numel() == C:
            out[~valid, :] = 0.0
            out[:, ~valid] = 0.0
        out[torch.eye(C, device=self.device, dtype=torch.bool)] = 0.0
        return out

    pairwise_spectral_shape_similarity = pairwise_spectral_similarity

    @torch.no_grad()
    def geometry_conflict_matrix(
        self,
        center_margin: float = 1.0,
        subspace_weight: float = 1.0,
        band_weight: float = 0.25,
        spectral_shape_weight: float = 0.25,
        reliability_weighted: bool = True,
        **_: Any,
    ) -> torch.Tensor:
        C = len(self)
        if C == 0:
            return torch.empty((0, 0), device=self.device, dtype=self._dtype())
        center = self.pairwise_center_distance()
        center_risk = torch.relu(float(center_margin) - center) / max(float(center_margin), 1e-8)
        risk = torch.nan_to_num(center_risk, nan=0.0, posinf=0.0, neginf=0.0)
        risk = risk + float(subspace_weight) * self.pairwise_subspace_overlap()
        risk = risk + float(band_weight) * self.pairwise_band_similarity()
        risk = risk + float(spectral_shape_weight) * self.pairwise_spectral_similarity()
        if reliability_weighted and self.reliability.numel() == C:
            rel = self.reliability.clamp(self.min_reliability, 1.0)
            uncertainty = 2.0 - rel.view(-1, 1) - rel.view(1, -1)
            risk = risk * (1.0 + 0.5 * uncertainty.clamp(0.0, 2.0))
        valid = self.get_valid_mask()
        if valid.numel() == C:
            risk[~valid, :] = 0.0
            risk[:, ~valid] = 0.0
        risk[torch.eye(C, device=self.device, dtype=torch.bool)] = 0.0
        return torch.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def old_new_subspace_overlap_report(
        self,
        old_class_count: int,
        new_class_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        old_count = int(max(0, min(int(old_class_count), len(self))))
        if old_count <= 0 or old_count >= len(self):
            return {"max_overlap": 0.0, "mean_overlap": 0.0, "pair": None, "old_class_id": None, "new_class_id": None}
        valid = self.get_valid_mask()
        old_ids = [c for c in range(old_count) if valid.numel() > c and bool(valid[c].item())]
        if new_class_ids is None:
            new_ids = [c for c in range(old_count, len(self)) if valid.numel() > c and bool(valid[c].item())]
        else:
            new_ids = [int(c) for c in new_class_ids if 0 <= int(c) < len(self) and valid.numel() > int(c) and bool(valid[int(c)].item())]
        if not old_ids or not new_ids:
            return {"max_overlap": 0.0, "mean_overlap": 0.0, "pair": None, "old_class_id": None, "new_class_id": None}
        sub = self.pairwise_subspace_overlap()
        vals = sub[torch.as_tensor(old_ids, device=self.device)][:, torch.as_tensor(new_ids, device=self.device)]
        flat = int(vals.argmax().item())
        i = flat // int(vals.size(1))
        j = flat % int(vals.size(1))
        return {
            "max_overlap": float(vals.max().item()),
            "mean_overlap": float(vals.mean().item()),
            "pair": (int(old_ids[i]), int(new_ids[j])),
            "old_class_id": int(old_ids[i]),
            "new_class_id": int(new_ids[j]),
        }

    @torch.no_grad()
    def compute_geometry_diagnostics(
        self,
        seen_classes: Optional[Iterable[int]] = None,
        old_class_ids: Optional[Iterable[int]] = None,
        new_class_ids: Optional[Iterable[int]] = None,
        reference_snapshot: Optional[Dict[str, torch.Tensor]] = None,
        center_margin: float = 1.0,
    ) -> Dict[str, Any]:
        if seen_classes is None:
            seen = [c for c in range(len(self)) if bool(self.get_valid_mask()[c].item())] if len(self) else []
        else:
            seen = _ordered_unique_ints(seen_classes)
        self.assert_bank_valid(seen, strict=True) if seen else self.assert_bank_valid(strict=True)

        valid = self.get_valid_mask()
        ids = [c for c in seen if 0 <= c < len(self) and valid.numel() > c and bool(valid[c].item())]
        center_dist = self.pairwise_center_distance()
        sub_overlap = self.pairwise_subspace_overlap()
        conflict = self.geometry_conflict_matrix(center_margin=center_margin)

        finite_center = center_dist[torch.isfinite(center_dist)]
        diag: Dict[str, Any] = {
            "num_rows": int(len(self)),
            "num_valid_rows": int(valid.sum().item()) if valid.numel() else 0,
            "seen_classes": ids,
            "center_distance_min": float(finite_center.min().item()) if finite_center.numel() else 0.0,
            "center_distance_mean": float(finite_center.mean().item()) if finite_center.numel() else 0.0,
            "subspace_overlap_max": float(sub_overlap.max().item()) if sub_overlap.numel() else 0.0,
            "subspace_overlap_mean": float(sub_overlap[sub_overlap > 0].mean().item()) if bool((sub_overlap > 0).any().item()) else 0.0,
            "residual_variance_min": float(self.res_vars[valid].min().item()) if bool(valid.any().item()) else 0.0,
            "residual_variance_mean": float(self.res_vars[valid].mean().item()) if bool(valid.any().item()) else 0.0,
            "residual_variance_max": float(self.res_vars[valid].max().item()) if bool(valid.any().item()) else 0.0,
            "geometry_conflict_max": float(conflict.max().item()) if conflict.numel() else 0.0,
            "geometry_conflict_mean": float(conflict[conflict > 0].mean().item()) if bool((conflict > 0).any().item()) else 0.0,
            "completeness_ok": True,
            "missing_seen_classes": [],
        }

        # Reserve/certificate-friendly metrics used by the base trainer.
        if bool(valid.any().item()):
            valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
            diag["mean_reliability"] = float(self.reliability.index_select(0, valid_ids).mean().item())
            diag["min_reliability"] = float(self.reliability.index_select(0, valid_ids).min().item())
            diag["mean_active_rank"] = float(self.active_ranks.index_select(0, valid_ids).float().mean().item())
            diag["feature_rank_usage"] = float(self.active_ranks.index_select(0, valid_ids).float().mean().item() / max(float(self.rank), 1.0))
        else:
            diag["mean_reliability"] = 0.0
            diag["min_reliability"] = 0.0
            diag["mean_active_rank"] = 0.0
            diag["feature_rank_usage"] = 0.0

        band_sim = self.pairwise_band_similarity()
        spec_sim = self.pairwise_spectral_similarity()
        diag["mean_band_similarity"] = float(band_sim[band_sim > 0].mean().item()) if bool((band_sim > 0).any().item()) else 0.0
        diag["band_similarity_max"] = float(band_sim.max().item()) if band_sim.numel() else 0.0
        diag["mean_spectral_similarity"] = float(spec_sim[spec_sim > 0].mean().item()) if bool((spec_sim > 0).any().item()) else 0.0
        diag["spectral_similarity_max"] = float(spec_sim.max().item()) if spec_sim.numel() else 0.0
        # Higher is better: compact reserve proxy using low overlap/conflict and
        # moderate rank usage. This is diagnostic only; training loss controls the geometry.
        diag["geometry_reserve_score"] = float(max(0.0, 1.0 - diag.get("subspace_overlap_max", 0.0) - 0.25 * diag.get("geometry_conflict_max", 0.0)))

        if seen:
            missing = [c for c in seen if c < 0 or c >= len(self) or valid.numel() <= c or not bool(valid[c].item())]
            diag["missing_seen_classes"] = missing
            diag["completeness_ok"] = len(missing) == 0

        if old_class_ids is not None and new_class_ids is not None:
            old_ids = [c for c in _ordered_unique_ints(old_class_ids) if c in ids]
            new_ids = [c for c in _ordered_unique_ints(new_class_ids) if c in ids]
            if old_ids and new_ids:
                old_t = torch.as_tensor(old_ids, device=self.device, dtype=torch.long)
                new_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
                ov = sub_overlap.index_select(0, old_t).index_select(1, new_t)
                rk = conflict.index_select(0, old_t).index_select(1, new_t)
                diag["old_new_overlap_max"] = float(ov.max().item()) if ov.numel() else 0.0
                diag["old_new_overlap_mean"] = float(ov.mean().item()) if ov.numel() else 0.0
                diag["old_new_conflict_max"] = float(rk.max().item()) if rk.numel() else 0.0
                diag["old_new_conflict_mean"] = float(rk.mean().item()) if rk.numel() else 0.0
            else:
                diag["old_new_overlap_max"] = 0.0
                diag["old_new_overlap_mean"] = 0.0
                diag["old_new_conflict_max"] = 0.0
                diag["old_new_conflict_mean"] = 0.0

        if reference_snapshot is not None:
            drift = self.compare_snapshot(reference_snapshot, class_ids=seen)
            diag.update({f"drift_{k}": v for k, v in drift.items()})

        return diag

    geometry_diagnostics = compute_geometry_diagnostics

    @torch.no_grad()
    def top_geometry_conflicts(self, k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        risk = self.geometry_conflict_matrix(**kwargs)
        C = int(risk.size(0))
        if C <= 1:
            return []
        mask = torch.triu(torch.ones_like(risk, dtype=torch.bool), diagonal=1)
        vals = risk[mask]
        if vals.numel() == 0:
            return []
        pairs = mask.nonzero(as_tuple=False)
        top_vals, top_idx = torch.topk(vals, k=min(int(k), vals.numel()))
        center = self.pairwise_center_distance()
        sub = self.pairwise_subspace_overlap()
        out: List[Dict[str, Any]] = []
        for score, pos in zip(top_vals.detach().cpu(), top_idx.detach().cpu()):
            i, j = pairs[int(pos.item())].tolist()
            out.append({
                "class_i": int(i),
                "class_j": int(j),
                "conflict": float(score.item()),
                "center_distance": float(center[i, j].item()) if torch.isfinite(center[i, j]) else float("inf"),
                "subspace_overlap": float(sub[i, j].item()),
            })
        return out

    # ------------------------------------------------------------------
    # snapshots / drift diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def export_snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "means": self.means.detach().clone(),
            "bases": self.bases.detach().clone(),
            "eigvals": self.eigvals.detach().clone(),
            "res_vars": self.res_vars.detach().clone(),
            "active_ranks": self.active_ranks.detach().clone(),
            "sample_counts": self.sample_counts.detach().clone(),
            "reliability": self.reliability.detach().clone(),
            "feature_reliability": self.feature_reliability.detach().clone(),
            "band_importances": self.band_importances.detach().clone(),
            "band_reliability": self.band_reliability.detach().clone(),
            "spectral_prototypes": self.spectral_prototypes.detach().clone(),
            "spectral_reliability": self.spectral_reliability.detach().clone(),
            "phase_created": self.phase_created.detach().clone(),
            "frozen_class_mask": self.frozen_class_mask.detach().clone(),
            "band_dim": torch.tensor(int(self._band_dim.item()), device=self.device, dtype=torch.long),
            "feature_dim": torch.tensor(int(self.d_model), device=self.device, dtype=torch.long),
            "rank": torch.tensor(int(self.rank), device=self.device, dtype=torch.long),
            "variances": self.get_variances().detach().clone(),
            "spectral_curve_means": self.spectral_curve_means.detach().clone(),
            "spectral_curve_vars": self.spectral_curve_vars.detach().clone(),
            "spectral_curve_d1": self.spectral_curve_d1.detach().clone(),
            "spectral_curve_d2": self.spectral_curve_d2.detach().clone(),
            "spectral_shape_reliability": self.spectral_shape_reliability.detach().clone(),
        }

    @torch.no_grad()
    def load_snapshot(self, snapshot: Dict[str, torch.Tensor], strict: bool = True) -> None:
        if not snapshot:
            if strict:
                raise ValueError("empty GeometryBank snapshot")
            return
        required = ("means", "bases", "eigvals", "res_vars")
        missing = [k for k in required if k not in snapshot]
        if missing:
            if strict:
                raise ValueError(f"snapshot missing keys: {missing}")
            return

        dtype = self._dtype()
        means = torch.as_tensor(snapshot["means"], device=self.device, dtype=dtype)
        bases = torch.as_tensor(snapshot["bases"], device=self.device, dtype=dtype)
        eigvals = torch.as_tensor(snapshot["eigvals"], device=self.device, dtype=dtype)
        res_vars = torch.as_tensor(snapshot["res_vars"], device=self.device, dtype=dtype).flatten()
        if means.dim() != 2 or means.size(1) != self.d_model:
            raise ValueError(f"snapshot means must be [C,{self.d_model}], got {tuple(means.shape)}")
        C = int(means.size(0))
        band_dim = int(torch.as_tensor(snapshot.get("band_dim", 0)).item()) if "band_dim" in snapshot else 0
        bands = snapshot.get("band_importances", None)
        if bands is not None and torch.as_tensor(bands).numel() > 0:
            band_dim = int(torch.as_tensor(bands).shape[1])
        self.reset_storage(band_dim=band_dim, dtype=dtype)
        self.ensure_class_count(C, spectral_dim=band_dim, dtype=dtype)

        self.means.copy_(means)
        self.bases.copy_(torch.stack([self._prepare_basis(bases[c]) for c in range(C)], dim=0))
        self.eigvals.copy_(torch.stack([self._prepare_eigvals(eigvals[c], res_vars[c]) for c in range(C)], dim=0))
        self.res_vars.copy_(res_vars.clamp_min(self.variance_floor))
        for key in ("active_ranks", "sample_counts", "reliability", "feature_reliability", "band_reliability", "spectral_reliability", "phase_created", "frozen_class_mask"):
            if key in snapshot and torch.as_tensor(snapshot[key]).numel() == getattr(self, key).numel():
                getattr(self, key).copy_(torch.as_tensor(snapshot[key], device=self.device, dtype=getattr(self, key).dtype).reshape_as(getattr(self, key)))
        if band_dim > 0:
            if "band_importances" in snapshot and torch.as_tensor(snapshot["band_importances"]).shape == self.band_importances.shape:
                self.band_importances.copy_(torch.as_tensor(snapshot["band_importances"], device=self.device, dtype=dtype))
            if "spectral_prototypes" in snapshot and torch.as_tensor(snapshot["spectral_prototypes"]).shape == self.spectral_prototypes.shape:
                self.spectral_prototypes.copy_(torch.as_tensor(snapshot["spectral_prototypes"], device=self.device, dtype=dtype))
            # Restore spectral-shape descriptors when present; otherwise rebuild
            # them from spectral prototypes so diagnostics remain available.
            restored_curves = False
            for key in ("spectral_curve_means", "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2", "spectral_shape_reliability"):
                if key in snapshot and hasattr(self, key) and torch.as_tensor(snapshot[key]).shape == getattr(self, key).shape:
                    getattr(self, key).copy_(torch.as_tensor(snapshot[key], device=self.device, dtype=getattr(self, key).dtype))
                    restored_curves = True
            if not restored_curves and self.spectral_prototypes.numel() > 0:
                for c in range(C):
                    if self.sample_counts.numel() > c and float(self.sample_counts[c].item()) > 0:
                        shape = self._spectral_shape_from_proto(self.spectral_prototypes[c])
                        self.spectral_curve_means[c] = shape["mean"]
                        self.spectral_curve_vars[c] = shape["var"]
                        if self.spectral_curve_d1.numel() > 0:
                            self.spectral_curve_d1[c] = shape["d1"]
                        if self.spectral_curve_d2.numel() > 0:
                            self.spectral_curve_d2[c] = shape["d2"]
                        self.spectral_shape_reliability[c] = shape["reliability"]
        self.assert_bank_valid(strict=True)

    @torch.no_grad()
    def reset_storage(self, band_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
        dtype = dtype or self._dtype()
        dev = self.device
        band_dim = int(max(0, band_dim))
        self.means = torch.empty((0, self.d_model), device=dev, dtype=dtype)
        self.bases = torch.empty((0, self.d_model, self.rank), device=dev, dtype=dtype)
        self.eigvals = torch.empty((0, self.rank), device=dev, dtype=dtype)
        self.res_vars = torch.empty((0,), device=dev, dtype=dtype)
        self.active_ranks = torch.empty((0,), dtype=torch.long, device=dev)
        self.sample_counts = torch.empty((0,), device=dev, dtype=dtype)
        self.reliability = torch.empty((0,), device=dev, dtype=dtype)
        self.feature_reliability = torch.empty((0,), device=dev, dtype=dtype)
        self.band_reliability = torch.empty((0,), device=dev, dtype=dtype)
        self.spectral_reliability = torch.empty((0,), device=dev, dtype=dtype)
        self.phase_created = torch.empty((0,), dtype=torch.long, device=dev)
        self.frozen_class_mask = torch.empty((0,), dtype=torch.bool, device=dev)
        self._band_dim = torch.tensor(band_dim, dtype=torch.long, device=dev)
        self.band_importances = torch.empty((0, band_dim), device=dev, dtype=dtype)
        self.spectral_prototypes = torch.empty((0, band_dim), device=dev, dtype=dtype)
        self.spectral_curve_means = torch.empty((0, band_dim), device=dev, dtype=dtype)
        self.spectral_curve_vars = torch.empty((0, band_dim), device=dev, dtype=dtype)
        self.spectral_curve_d1 = torch.empty((0, max(band_dim - 1, 0)), device=dev, dtype=dtype)
        self.spectral_curve_d2 = torch.empty((0, max(band_dim - 2, 0)), device=dev, dtype=dtype)
        self.spectral_shape_reliability = torch.empty((0,), device=dev, dtype=dtype)

    @torch.no_grad()
    def compare_snapshot(self, snapshot: Dict[str, torch.Tensor], class_ids: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        if not snapshot or "means" not in snapshot or "bases" not in snapshot:
            return {"center_drift_mean": 0.0, "center_drift_max": 0.0, "basis_drift_mean": 0.0, "basis_drift_max": 0.0}
        C0 = int(torch.as_tensor(snapshot["means"]).size(0))
        C = min(C0, len(self))
        ids = list(range(C)) if class_ids is None else [int(c) for c in class_ids if 0 <= int(c) < C]
        if not ids:
            return {"center_drift_mean": 0.0, "center_drift_max": 0.0, "basis_drift_mean": 0.0, "basis_drift_max": 0.0}
        idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        old_means = torch.as_tensor(snapshot["means"], device=self.device, dtype=self._dtype()).index_select(0, idx)
        new_means = self.means.index_select(0, idx)
        center = (new_means - old_means).norm(dim=1)

        old_bases = torch.as_tensor(snapshot["bases"], device=self.device, dtype=self._dtype()).index_select(0, idx)
        new_bases = self.bases.index_select(0, idx)
        active_old = torch.as_tensor(snapshot.get("active_ranks", self.active_ranks[:C]), device=self.device, dtype=torch.long).flatten()
        basis_drifts = []
        for local_i, c in enumerate(ids):
            r = int(min(self.active_ranks[c].item(), active_old[c].item(), self.rank))
            if r <= 0:
                basis_drifts.append(torch.tensor(0.0, device=self.device, dtype=self._dtype()))
            else:
                ov = old_bases[local_i, :, :r].t().matmul(new_bases[local_i, :, :r]).pow(2).sum() / float(max(r, 1))
                basis_drifts.append((1.0 - ov.clamp(0.0, 1.0)).detach())
        basis = torch.stack(basis_drifts) if basis_drifts else torch.zeros((1,), device=self.device, dtype=self._dtype())

        eig_drift = torch.zeros((1,), device=self.device, dtype=self._dtype())
        if "eigvals" in snapshot:
            old_e = torch.as_tensor(snapshot["eigvals"], device=self.device, dtype=self._dtype()).index_select(0, idx)
            new_e = self.eigvals.index_select(0, idx)
            eig_drift = (new_e - old_e).abs().mean(dim=1)
        rv_drift = torch.zeros((1,), device=self.device, dtype=self._dtype())
        if "res_vars" in snapshot:
            old_rv = torch.as_tensor(snapshot["res_vars"], device=self.device, dtype=self._dtype()).flatten().index_select(0, idx)
            new_rv = self.res_vars.index_select(0, idx)
            rv_drift = (new_rv - old_rv).abs()

        return {
            "center_drift_mean": float(center.mean().item()),
            "center_drift_max": float(center.max().item()),
            "basis_drift_mean": float(basis.mean().item()),
            "basis_drift_max": float(basis.max().item()),
            "eigval_drift_mean": float(eig_drift.mean().item()),
            "eigval_drift_max": float(eig_drift.max().item()),
            "resvar_drift_mean": float(rv_drift.mean().item()),
            "resvar_drift_max": float(rv_drift.max().item()),
        }

    # ------------------------------------------------------------------
    # candidate descriptor insertion and clean descriptor transport
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_candidate_geometry_rows(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        spectral_summary: Optional[torch.Tensor] = None,
        band_weights: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: bool = True,
        class_ids: Optional[Iterable[int]] = None,
        **_: Any,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        rows = self.extract_geometry(features, labels, spectral_summary=spectral_summary, band_weights=band_weights, spectral_summary_is_physical=spectral_summary_is_physical)
        if class_ids is None:
            return rows
        allowed = set(_ordered_unique_ints(class_ids))
        return {int(c): g for c, g in rows.items() if int(c) in allowed}

    @torch.no_grad()
    def commit_candidate_geometry_rows(
        self,
        candidate_rows: Dict[int, Dict[str, torch.Tensor]],
        *,
        allow_frozen_update: bool = False,
        phase_created: int = -1,
        freeze: bool = False,
        context: str = "candidate_commit",
    ) -> Dict[str, Any]:
        committed: List[int] = []
        for c in sorted(int(k) for k in candidate_rows.keys()):
            g = candidate_rows[c]
            self.add_or_update_class_geometry(
                c,
                mean=g["mean"],
                basis=g["basis"],
                eigvals=g["eigvals"],
                res_var=g["res_var"],
                spectral_prototype=g.get("spectral_prototype"),
                band_importance=g.get("band_importance"),
                sample_count=g.get("sample_count"),
                active_rank=g.get("active_rank"),
                reliability=g.get("reliability"),
                feature_reliability=g.get("feature_reliability"),
                band_reliability=g.get("band_reliability"),
                spectral_reliability=g.get("spectral_reliability"),
                phase_created=phase_created,
                freeze=freeze,
                allow_frozen_update=allow_frozen_update,
            )
            committed.append(c)
        return {"active": len(committed), "committed_class_ids": committed, "context": str(context)}

    @torch.no_grad()
    def correct_new_descriptors_against_old(
        self,
        old_class_count: int,
        new_class_ids: Iterable[int],
        *,
        overlap_threshold: float = 0.60,
        mean_push: float = 0.10,
        basis_projection_strength: float = 0.35,
        variance_shrink: float = 0.10,
        topk_old: int = 3,
        **_: Any,
    ) -> Dict[str, Any]:
        """Descriptor-level transport: move only new descriptors away from old geometry.

        This is the meaningful transport operation for the clean NECIL method.
        Old rows are never changed.
        """
        old_count = int(max(0, min(int(old_class_count), len(self))))
        new_ids = [int(c) for c in _ordered_unique_ints(new_class_ids) if 0 <= int(c) < len(self)]
        if old_count <= 0 or not new_ids:
            return {"active": 0, "corrected_class_ids": [], "max_overlap_before": 0.0, "max_overlap_after": 0.0}

        valid = self.get_valid_mask()
        old_ids = [c for c in range(old_count) if valid.numel() > c and bool(valid[c].item())]
        new_ids = [c for c in new_ids if valid.numel() > c and bool(valid[c].item())]
        if not old_ids or not new_ids:
            return {"active": 0, "corrected_class_ids": [], "max_overlap_before": 0.0, "max_overlap_after": 0.0}

        before = self.pairwise_subspace_overlap()
        old_t = torch.as_tensor(old_ids, device=self.device, dtype=torch.long)
        new_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
        ov_before = before.index_select(0, old_t).index_select(1, new_t)
        corrected: List[int] = []

        for cls in new_ids:
            self._assert_update_allowed(cls, allow_frozen_update=False)
            scores = before.index_select(0, old_t)[:, cls]
            if scores.numel() == 0 or float(scores.max().item()) <= float(overlap_threshold):
                continue
            k = min(int(max(1, topk_old)), int(scores.numel()))
            _, pos = torch.topk(scores, k=k, largest=True)
            U = self.bases[cls].detach().clone()
            mu = self.means[cls].detach().clone()
            eig = self.eigvals[cls].detach().clone()
            rv = self.res_vars[cls].detach().clone()
            projector = torch.zeros((self.d_model, self.d_model), device=self.device, dtype=self._dtype())
            push = torch.zeros((self.d_model,), device=self.device, dtype=self._dtype())

            for p in pos.tolist():
                old_cls = old_ids[int(p)]
                r = int(self.active_ranks[old_cls].item())
                if r > 0:
                    Uo = self.bases[old_cls, :, :r]
                    projector = projector + Uo.matmul(Uo.t()) / float(k)
                direction = mu - self.means[old_cls]
                direction = direction / direction.norm().clamp_min(1e-8)
                push = push + direction / float(k)

            U_new = complete_orthonormal_basis(U - float(basis_projection_strength) * projector.matmul(U), self.rank)
            mu_new = mu + float(mean_push) * push / push.norm().clamp_min(1e-8)
            eig_new = (eig * (1.0 - float(variance_shrink))).clamp_min(self.variance_floor)
            rv_new = (rv * (1.0 - 0.5 * float(variance_shrink))).clamp_min(self.variance_floor)

            self.add_or_update_class_geometry(
                cls,
                mean=mu_new,
                basis=U_new,
                eigvals=eig_new,
                res_var=rv_new,
                spectral_prototype=self.spectral_prototypes[cls] if self.spectral_prototypes.numel() else None,
                band_importance=self.band_importances[cls] if self.band_importances.numel() else None,
                sample_count=self.sample_counts[cls],
                active_rank=self.active_ranks[cls],
                reliability=self.reliability[cls],
                feature_reliability=self.feature_reliability[cls],
                phase_created=int(self.phase_created[cls].item()),
                allow_frozen_update=False,
            )
            corrected.append(cls)

        after = self.pairwise_subspace_overlap()
        ov_after = after.index_select(0, old_t).index_select(1, new_t)
        return {
            "active": len(corrected),
            "corrected_class_ids": corrected,
            "max_overlap_before": float(ov_before.max().item()) if ov_before.numel() else 0.0,
            "max_overlap_after": float(ov_after.max().item()) if ov_after.numel() else 0.0,
        }

    def transport_frozen_geometry(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError(
            "transport_frozen_geometry is disabled in the clean NECIL-HSI bank. "
            "Do not move old frozen rows. Use correct_new_descriptors_against_old() "
            "to transport only new descriptors away from old geometry."
        )

    validated_insert_with_transport = build_candidate_geometry_rows

    # ------------------------------------------------------------------
    # risk-weighted replay compatibility
    # ------------------------------------------------------------------
    @torch.no_grad()
    def old_replay_risk_weights(
        self,
        old_class_count: int,
        new_class_ids: Optional[Iterable[int]] = None,
        min_weight: float = 0.25,
        max_weight: float = 3.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        old_count = int(max(0, min(int(old_class_count), len(self))))
        if old_count <= 0:
            return torch.empty((0,), device=self.device, dtype=self._dtype())
        weights = torch.ones((old_count,), device=self.device, dtype=self._dtype())
        if len(self) <= old_count:
            return weights
        if new_class_ids is None:
            new_ids = [c for c in range(old_count, len(self)) if bool(self.get_valid_mask()[c].item())]
        else:
            new_ids = [int(c) for c in new_class_ids if 0 <= int(c) < len(self)]
        if not new_ids:
            return weights
        risk = self.geometry_conflict_matrix(**kwargs)
        vals = risk[:old_count].index_select(1, torch.as_tensor(new_ids, device=self.device, dtype=torch.long)).max(dim=1).values
        if float(vals.max().item()) <= 0:
            return weights
        weights = vals / vals.mean().clamp_min(1e-8)
        return weights.clamp(float(min_weight), float(max_weight)).detach()

    @torch.no_grad()
    def old_replay_sample_counts(
        self,
        old_class_count: int,
        new_class_ids: Optional[Iterable[int]] = None,
        base_samples_per_class: int = 16,
        min_samples_per_class: int = 4,
        max_multiplier: float = 3.0,
        **kwargs: Any,
    ) -> Dict[int, int]:
        old_count = int(max(0, min(int(old_class_count), len(self))))
        if old_count <= 0:
            return {}
        weights = self.old_replay_risk_weights(old_count, new_class_ids, max_weight=max_multiplier, **kwargs)
        valid = self.get_valid_mask()
        out: Dict[int, int] = {}
        for c in range(old_count):
            if valid.numel() <= c or not bool(valid[c].item()):
                out[c] = 0
            else:
                n = int(round(int(base_samples_per_class) * float(weights[c].item())))
                out[c] = max(int(min_samples_per_class), min(n, int(round(base_samples_per_class * max_multiplier))))
        return out

    @torch.no_grad()
    def sample_risk_weighted_old_features(
        self,
        old_class_count: int,
        new_class_ids: Optional[Iterable[int]] = None,
        base_samples_per_class: int = 16,
        min_samples_per_class: int = 4,
        max_multiplier: float = 3.0,
        parallel_scale: float = 1.0,
        residual_scale: float = 0.25,
        reliability_gated: bool = True,
        **risk_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        counts = self.old_replay_sample_counts(
            old_class_count,
            new_class_ids,
            base_samples_per_class=base_samples_per_class,
            min_samples_per_class=min_samples_per_class,
            max_multiplier=max_multiplier,
            **risk_kwargs,
        )
        x, y = self.sample_synthetic_features(
            class_ids=range(int(old_class_count)),
            class_sample_counts=counts,
            parallel_scale=parallel_scale,
            residual_scale=residual_scale,
            reliability_gated=reliability_gated,
        )
        return x, y, counts

    @torch.no_grad()
    def memory_cost_summary(self, bytes_per_float: int = 4) -> Dict[str, Any]:
        tensors = {
            "means": self.means,
            "bases": self.bases,
            "eigvals": self.eigvals,
            "res_vars": self.res_vars,
            "sample_counts": self.sample_counts,
            "reliability": self.reliability,
            "band_importances": self.band_importances,
            "spectral_prototypes": self.spectral_prototypes,
        }
        elems = {k: int(v.numel()) for k, v in tensors.items() if torch.is_tensor(v)}
        total = int(sum(elems.values()))
        return {
            "num_rows": int(len(self)),
            "num_valid_rows": int(self.get_valid_mask().sum().item()) if len(self) else 0,
            "feature_dim": int(self.d_model),
            "rank": int(self.rank),
            "band_dim": int(self._band_dim.item()),
            "actual_float_elements": total,
            "actual_fp32_kb": float(total * int(bytes_per_float) / 1024.0),
            "component_float_elements": elems,
            "stores_raw_samples": False,
        }

    @torch.no_grad()
    def geometry_health_summary(self, class_names: Optional[Sequence[str]] = None, topk_bands: int = 5) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        valid = self.get_valid_mask()
        for c in range(len(self)):
            rows.append({
                "class_id": int(c),
                "class_name": str(class_names[c]) if class_names is not None and c < len(class_names) else None,
                "valid": bool(valid[c].item()) if valid.numel() > c else False,
                "sample_count": float(self.sample_counts[c].item()) if self.sample_counts.numel() > c else 0.0,
                "active_rank": int(self.active_ranks[c].item()) if self.active_ranks.numel() > c else 0,
                "res_var": float(self.res_vars[c].item()) if self.res_vars.numel() > c else 0.0,
                "reliability": float(self.reliability[c].item()) if self.reliability.numel() > c else 0.0,
                "phase_created": int(self.phase_created[c].item()) if self.phase_created.numel() > c else -1,
                "frozen": bool(self.frozen_class_mask[c].item()) if self.frozen_class_mask.numel() > c else False,
            })
        return {
            "num_rows": int(len(self)),
            "num_valid_rows": int(valid.sum().item()) if valid.numel() else 0,
            "class_geometry": rows,
            "global_geometry": self.compute_geometry_diagnostics(),
            "memory_cost": self.memory_cost_summary(),
        }















# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# _EPS = 1e-12
# _RAW_MEMORY_NAMES = {
#     "raw_samples", "raw_patches", "old_samples", "old_patches", "stored_samples",
#     "stored_patches", "feature_memory", "old_features", "stored_features",
#     "exemplars", "exemplar_memory", "memory_features", "memory_patches",
# }


# def _as_bool(value: Any, default: bool = False) -> bool:
#     if value is None:
#         return bool(default)
#     if isinstance(value, bool):
#         return value
#     if isinstance(value, (int, float)):
#         return bool(value)
#     if isinstance(value, str):
#         v = value.strip().lower()
#         if v in {"1", "true", "yes", "y", "on"}:
#             return True
#         if v in {"0", "false", "no", "n", "off", "none", "null", ""}:
#             return False
#     return bool(value)


# def _ordered_unique_ints(values: Iterable[int]) -> List[int]:
#     out: List[int] = []
#     seen = set()
#     for v in values:
#         c = int(v)
#         if c not in seen:
#             out.append(c)
#             seen.add(c)
#     return out


# def orthonormalize_columns(basis: torch.Tensor) -> torch.Tensor:
#     """Return an orthonormal-column version of [D, R]."""
#     if not torch.is_tensor(basis):
#         raise TypeError("basis must be a torch.Tensor")
#     if basis.dim() != 2:
#         raise ValueError(f"basis must be [D,R], got {tuple(basis.shape)}")
#     if basis.numel() == 0 or basis.size(1) == 0:
#         return basis
#     basis = torch.nan_to_num(basis.float(), nan=0.0, posinf=0.0, neginf=0.0)
#     try:
#         q, _ = torch.linalg.qr(basis, mode="reduced")
#     except RuntimeError:
#         u, _, _ = torch.linalg.svd(basis, full_matrices=False)
#         q = u
#     return q[:, : basis.size(1)]


# def complete_orthonormal_basis(active_basis: torch.Tensor, rank: int) -> torch.Tensor:
#     """Complete [D, q] active basis to [D, rank] without using data exemplars."""
#     if not torch.is_tensor(active_basis):
#         raise TypeError("active_basis must be a torch.Tensor")
#     if active_basis.dim() != 2:
#         raise ValueError(f"active_basis must be [D,q], got {tuple(active_basis.shape)}")

#     d = int(active_basis.size(0))
#     rank = int(max(0, min(int(rank), d)))
#     device, dtype = active_basis.device, active_basis.dtype
#     if rank == 0:
#         return torch.empty((d, 0), device=device, dtype=dtype)

#     cols: List[torch.Tensor] = []
#     if active_basis.numel() > 0 and active_basis.size(1) > 0:
#         q = orthonormalize_columns(active_basis[:, :rank].to(dtype=torch.float32)).to(device=device, dtype=dtype)
#         for j in range(q.size(1)):
#             if q[:, j].norm() > 1e-7:
#                 cols.append(q[:, j])

#     eye = torch.eye(d, device=device, dtype=dtype)
#     for j in range(d):
#         v = eye[:, j].clone()
#         for u in cols:
#             v = v - torch.dot(v, u) * u
#         n = v.norm()
#         if n > 1e-7:
#             cols.append(v / n)
#         if len(cols) >= rank:
#             break

#     if len(cols) < rank:
#         raise RuntimeError(f"Could not complete orthonormal basis to rank={rank} in dim={d}.")
#     return torch.stack(cols[:rank], dim=1)


# class GeometryBank(nn.Module):
#     """
#     Exemplar-free HSI class-geometry memory.

#     Stored memory is limited to compact class descriptors:
#       - class mean mu_c
#       - low-rank basis U_c
#       - eigenvalues lambda_c
#       - residual variance sigma_c^2
#       - optional spectral prototype / band importance
#       - sample count, reliability, phase_created, frozen status

#     The bank never stores raw HSI patches, raw old samples, or old feature batches.
#     Feature tensors may be passed temporarily to `extract_geometry` or
#     `add_or_update_class_geometry(features=...)`, but are reduced immediately into
#     descriptors and discarded.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         rank: int,
#         device: Union[str, torch.device] = "cpu",
#         variance_floor: float = 1e-4,
#         variance_shrinkage: float = 0.10,
#         max_variance_ratio: float = 50.0,
#         min_reliability: float = 0.05,
#         reliability_sample_alpha: float = 20.0,
#         rank_energy_threshold: float = 0.95,
#         rank_eigen_ratio_threshold: float = 1e-3,
#         min_active_rank: int = 1,
#         small_class_rank_threshold_1: int = 30,
#         small_class_rank_threshold_2: int = 80,
#         small_class_rank_threshold_3: int = 150,
#         small_class_rank_cap_1: int = 1,
#         small_class_rank_cap_2: int = 3,
#         small_class_rank_cap_3: int = 4,
#         small_class_extra_shrinkage: float = 0.35,
#         **_: Any,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive")
#         self.rank = int(max(0, min(int(rank), self.d_model)))
#         self.variance_floor = float(max(float(variance_floor), 1e-12))
#         self.variance_shrinkage = float(max(0.0, min(float(variance_shrinkage), 1.0)))
#         self.max_variance_ratio = float(max(float(max_variance_ratio), 1.0))
#         self.min_reliability = float(max(0.0, min(float(min_reliability), 1.0)))
#         self.reliability_sample_alpha = float(max(float(reliability_sample_alpha), 1.0))
#         self.rank_energy_threshold = float(max(0.50, min(float(rank_energy_threshold), 0.999)))
#         self.rank_eigen_ratio_threshold = float(max(float(rank_eigen_ratio_threshold), 0.0))
#         self.min_active_rank = int(max(0, min(int(min_active_rank), self.rank)))

#         t1 = int(max(2, small_class_rank_threshold_1))
#         t2 = int(max(t1 + 1, small_class_rank_threshold_2))
#         t3 = int(max(t2 + 1, small_class_rank_threshold_3))
#         self.small_class_rank_thresholds = (t1, t2, t3)
#         self.small_class_rank_caps = (
#             int(max(1, min(int(small_class_rank_cap_1), self.rank))) if self.rank > 0 else 0,
#             int(max(1, min(int(small_class_rank_cap_2), self.rank))) if self.rank > 0 else 0,
#             int(max(1, min(int(small_class_rank_cap_3), self.rank))) if self.rank > 0 else 0,
#         )
#         self.small_class_extra_shrinkage = float(max(0.0, min(float(small_class_extra_shrinkage), 0.85)))

#         dev = torch.device(device)
#         self.register_buffer("means", torch.empty((0, self.d_model), device=dev))
#         self.register_buffer("bases", torch.empty((0, self.d_model, self.rank), device=dev))
#         self.register_buffer("eigvals", torch.empty((0, self.rank), device=dev))
#         self.register_buffer("res_vars", torch.empty((0,), device=dev))
#         self.register_buffer("active_ranks", torch.empty((0,), dtype=torch.long, device=dev))
#         self.register_buffer("sample_counts", torch.empty((0,), device=dev))
#         self.register_buffer("reliability", torch.empty((0,), device=dev))
#         self.register_buffer("feature_reliability", torch.empty((0,), device=dev))
#         self.register_buffer("band_importances", torch.empty((0, 0), device=dev))
#         self.register_buffer("band_reliability", torch.empty((0,), device=dev))
#         self.register_buffer("spectral_prototypes", torch.empty((0, 0), device=dev))
#         self.register_buffer("spectral_reliability", torch.empty((0,), device=dev))
#         self.register_buffer("phase_created", torch.empty((0,), dtype=torch.long, device=dev))
#         self.register_buffer("frozen_class_mask", torch.empty((0,), dtype=torch.bool, device=dev))
#         self.register_buffer("_band_dim", torch.tensor(0, dtype=torch.long, device=dev))

#         # Compatibility empty tensors expected by older classifier/trainer code.
#         self.register_buffer("spectral_curve_means", torch.empty((0, 0), device=dev))
#         self.register_buffer("spectral_curve_vars", torch.empty((0, 0), device=dev))
#         self.register_buffer("spectral_curve_d1", torch.empty((0, 0), device=dev))
#         self.register_buffer("spectral_curve_d2", torch.empty((0, 0), device=dev))
#         self.register_buffer("spectral_shape_reliability", torch.empty((0,), device=dev))

#     # ------------------------------------------------------------------
#     # basic properties / validation
#     # ------------------------------------------------------------------
#     @property
#     def device(self) -> torch.device:
#         return self.means.device

#     def __len__(self) -> int:
#         return int(self.means.size(0))

#     def _dtype(self) -> torch.dtype:
#         return self.means.dtype if self.means.numel() > 0 else torch.float32

#     @property
#     def resvars(self) -> torch.Tensor:
#         return self.res_vars

#     @resvars.setter
#     def resvars(self, value: torch.Tensor) -> None:
#         self.res_vars = value

#     @property
#     def spectral_protos(self) -> torch.Tensor:
#         return self.spectral_prototypes

#     @spectral_protos.setter
#     def spectral_protos(self, value: torch.Tensor) -> None:
#         self.spectral_prototypes = value

#     def _assert_no_raw_memory_attrs(self) -> None:
#         bad = [name for name in self.__dict__.keys() if name.lower() in _RAW_MEMORY_NAMES]
#         # Also check registered buffers/parameters by name.
#         bad.extend([name for name in self._buffers.keys() if name.lower() in _RAW_MEMORY_NAMES])
#         bad.extend([name for name in self._parameters.keys() if name.lower() in _RAW_MEMORY_NAMES])
#         if bad:
#             raise RuntimeError(
#                 "GeometryBank contains forbidden exemplar-like memory fields: "
#                 f"{sorted(set(bad))}. Store only compact class statistics."
#             )

#     def _valid_class_id(self, class_id: int, *, existing: bool = True) -> int:
#         c = int(class_id)
#         if c < 0:
#             raise IndexError(f"class_id must be non-negative, got {c}")
#         if existing and c >= len(self):
#             raise IndexError(f"class_id={c} out of range for bank size {len(self)}")
#         return c

#     def _sample_count_rank_cap(self, n: int) -> int:
#         n = int(max(0, n))
#         if self.rank <= 0 or n <= 1:
#             return 0
#         t1, t2, t3 = self.small_class_rank_thresholds
#         c1, c2, c3 = self.small_class_rank_caps
#         if n < t1:
#             cap = c1
#         elif n < t2:
#             cap = c2
#         elif n < t3:
#             cap = c3
#         else:
#             cap = self.rank
#         return int(max(0, min(self.rank, n - 1, cap)))

#     def _adaptive_shrinkage(self, n: int) -> float:
#         n = int(max(1, n))
#         extra = self.small_class_extra_shrinkage * min(1.0, self.reliability_sample_alpha / float(n))
#         return float(max(0.0, min(self.variance_shrinkage + extra, 0.90)))

#     def _ensure_band_dim(self, band_dim: int, dtype: Optional[torch.dtype] = None) -> None:
#         band_dim = int(max(0, band_dim))
#         if band_dim <= 0:
#             return
#         dtype = dtype or self._dtype()
#         cur = int(self._band_dim.item())
#         if cur > 0 and cur != band_dim:
#             raise ValueError(f"band/spectral dimension mismatch: existing={cur}, requested={band_dim}")
#         if cur == band_dim:
#             return
#         C = len(self)
#         self._band_dim = torch.tensor(band_dim, dtype=torch.long, device=self.device)
#         self.band_importances = torch.full((C, band_dim), 1.0 / float(band_dim), device=self.device, dtype=dtype)
#         self.band_reliability = torch.full((C,), self.min_reliability, device=self.device, dtype=dtype)
#         self.spectral_prototypes = torch.zeros((C, band_dim), device=self.device, dtype=dtype)
#         self.spectral_reliability = torch.full((C,), self.min_reliability, device=self.device, dtype=dtype)
#         self.spectral_curve_means = torch.zeros((C, band_dim), device=self.device, dtype=dtype)
#         self.spectral_curve_vars = torch.full((C, band_dim), self.variance_floor, device=self.device, dtype=dtype)
#         self.spectral_curve_d1 = torch.zeros((C, max(band_dim - 1, 0)), device=self.device, dtype=dtype)
#         self.spectral_curve_d2 = torch.zeros((C, max(band_dim - 2, 0)), device=self.device, dtype=dtype)
#         self.spectral_shape_reliability = torch.full((C,), self.min_reliability, device=self.device, dtype=dtype)

#     def _prepare_mean(self, mean: torch.Tensor) -> torch.Tensor:
#         t = torch.as_tensor(mean, device=self.device, dtype=self._dtype()).flatten()
#         if t.numel() != self.d_model:
#             raise ValueError(f"mean must have shape [{self.d_model}], got {tuple(t.shape)}")
#         return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

#     def _prepare_basis(self, basis: torch.Tensor) -> torch.Tensor:
#         t = torch.as_tensor(basis, device=self.device, dtype=self._dtype())
#         if t.dim() != 2:
#             raise ValueError(f"basis must be [D,R], got {tuple(t.shape)}")
#         if t.size(0) == self.rank and t.size(1) == self.d_model:
#             t = t.t()
#         if t.size(0) != self.d_model:
#             raise ValueError(f"basis first dimension must be {self.d_model}, got {t.size(0)}")
#         if t.size(1) > self.rank:
#             t = t[:, : self.rank]
#         if t.size(1) < self.rank:
#             # Complete only from nonzero columns.
#             norms = t.norm(dim=0) if t.numel() > 0 else torch.empty((0,), device=self.device)
#             active = t[:, norms > 1e-8] if norms.numel() and bool((norms > 1e-8).any().item()) else torch.empty((self.d_model, 0), device=self.device, dtype=self._dtype())
#             return complete_orthonormal_basis(active, self.rank)
#         return complete_orthonormal_basis(t, self.rank)

#     def _prepare_eigvals(self, eigvals: torch.Tensor, fallback: Union[float, torch.Tensor]) -> torch.Tensor:
#         fb = torch.as_tensor(fallback, device=self.device, dtype=self._dtype()).reshape(()).clamp_min(self.variance_floor)
#         t = torch.as_tensor(eigvals, device=self.device, dtype=self._dtype()).flatten()
#         if t.numel() > self.rank:
#             t = t[: self.rank]
#         elif t.numel() < self.rank:
#             t = torch.cat([t, torch.full((self.rank - t.numel(),), float(fb.item()), device=self.device, dtype=self._dtype())])
#         t = torch.nan_to_num(t, nan=float(fb.item()), posinf=float(fb.item()), neginf=float(fb.item()))
#         t = t.clamp_min(self.variance_floor)
#         # Active dimensions are expected sorted; full vector placeholders may include residual values.
#         return t

#     def _prepare_band_vector(self, value: Optional[torch.Tensor], band_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         dtype = self._dtype()
#         if band_dim <= 0:
#             return torch.empty((0,), device=self.device, dtype=dtype), torch.tensor(self.min_reliability, device=self.device, dtype=dtype)
#         if value is None or torch.as_tensor(value).numel() == 0:
#             b = torch.full((band_dim,), 1.0 / float(band_dim), device=self.device, dtype=dtype)
#         else:
#             raw = torch.as_tensor(value, device=self.device, dtype=dtype).flatten()
#             if raw.numel() != band_dim:
#                 raise ValueError(f"band/spectral vector must have {band_dim} values, got {raw.numel()}")
#             raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
#             if bool((raw < 0).any().item()):
#                 b = torch.softmax(raw, dim=0)
#             else:
#                 b = raw.clamp_min(0.0)
#                 b = b / b.sum().clamp_min(1e-8) if b.sum() > 1e-8 else torch.full((band_dim,), 1.0 / float(band_dim), device=self.device, dtype=dtype)
#         b = b.clamp_min(0.0)
#         b = b / b.sum().clamp_min(1e-8)

#         entropy = -(b.clamp_min(1e-12) * b.clamp_min(1e-12).log()).sum()
#         max_entropy = torch.log(torch.tensor(float(max(band_dim, 2)), device=self.device, dtype=dtype)).clamp_min(1e-12)
#         rel = (1.0 - entropy / max_entropy).clamp(self.min_reliability, 1.0)
#         return b.detach(), rel.detach()

#     def _spectral_shape_from_proto(self, proto: torch.Tensor) -> Dict[str, torch.Tensor]:
#         dtype = self._dtype()
#         band_dim = int(proto.numel())
#         if band_dim <= 0:
#             z = torch.empty((0,), device=self.device, dtype=dtype)
#             return {"mean": z, "var": z, "d1": z, "d2": z, "reliability": torch.tensor(self.min_reliability, device=self.device, dtype=dtype)}
#         p = torch.nan_to_num(proto.to(device=self.device, dtype=dtype).flatten(), nan=0.0, posinf=0.0, neginf=0.0)
#         d1 = p[1:] - p[:-1] if band_dim >= 2 else torch.empty((0,), device=self.device, dtype=dtype)
#         d2 = d1[1:] - d1[:-1] if band_dim >= 3 else torch.empty((0,), device=self.device, dtype=dtype)
#         e_curve = p.pow(2).mean().sqrt()
#         e_der = d1.pow(2).mean().sqrt() if d1.numel() else torch.tensor(0.0, device=self.device, dtype=dtype)
#         rel = (e_der / (e_curve + e_der + 1e-8)).clamp(self.min_reliability, 1.0)
#         return {
#             "mean": p.detach(),
#             "var": torch.full((band_dim,), self.variance_floor, device=self.device, dtype=dtype),
#             "d1": d1.detach(),
#             "d2": d2.detach(),
#             "reliability": rel.detach(),
#         }

#     @torch.no_grad()
#     def assert_bank_valid(self, seen_classes: Optional[Iterable[int]] = None, *, strict: bool = True) -> Dict[str, Any]:
#         errors: List[str] = []
#         self._assert_no_raw_memory_attrs()
#         C = len(self)
#         band_dim = int(self._band_dim.item())

#         expected = {
#             "means": (C, self.d_model),
#             "bases": (C, self.d_model, self.rank),
#             "eigvals": (C, self.rank),
#             "res_vars": (C,),
#             "active_ranks": (C,),
#             "sample_counts": (C,),
#             "reliability": (C,),
#             "feature_reliability": (C,),
#             "band_importances": (C, band_dim),
#             "band_reliability": (C,),
#             "spectral_prototypes": (C, band_dim),
#             "spectral_reliability": (C,),
#             "phase_created": (C,),
#             "frozen_class_mask": (C,),
#         }
#         for name, shape in expected.items():
#             value = getattr(self, name, None)
#             if not torch.is_tensor(value):
#                 errors.append(f"{name} is not a tensor")
#             elif tuple(value.shape) != tuple(shape):
#                 errors.append(f"{name} shape mismatch: got {tuple(value.shape)}, expected {shape}")

#         finite_names = [
#             "means", "bases", "eigvals", "res_vars", "sample_counts",
#             "reliability", "feature_reliability", "band_importances",
#             "band_reliability", "spectral_prototypes", "spectral_reliability",
#         ]
#         for name in finite_names:
#             value = getattr(self, name, None)
#             if torch.is_tensor(value) and value.numel() > 0 and not torch.isfinite(value).all():
#                 errors.append(f"{name} contains NaN/Inf")

#         if self.eigvals.numel() > 0 and bool((self.eigvals < self.variance_floor).any().item()):
#             errors.append("eigvals contain values below variance_floor")
#         if self.res_vars.numel() > 0 and bool((self.res_vars < self.variance_floor).any().item()):
#             errors.append("res_vars contain values below variance_floor")
#         if self.sample_counts.numel() > 0 and bool((self.sample_counts < 0).any().item()):
#             errors.append("sample_counts contain negative values")

#         if C > 0 and self.bases.numel() > 0 and self.rank > 0:
#             eye = torch.eye(self.rank, device=self.device, dtype=self._dtype())
#             gram = torch.bmm(self.bases.transpose(1, 2), self.bases)
#             valid_rows = self.sample_counts > 0 if self.sample_counts.numel() == C else torch.zeros((C,), device=self.device, dtype=torch.bool)
#             if bool(valid_rows.any().item()):
#                 max_ortho_err = (gram[valid_rows] - eye).abs().max()
#                 if float(max_ortho_err.detach().cpu().item()) > 1e-3:
#                     errors.append(f"basis columns are not orthonormal; max_err={float(max_ortho_err):.4e}")

#         for c in range(C):
#             n = int(float(self.sample_counts[c].detach().cpu().item())) if self.sample_counts.numel() > c else 0
#             r = int(self.active_ranks[c].detach().cpu().item()) if self.active_ranks.numel() > c else 0
#             cap = self._sample_count_rank_cap(n)
#             if n <= 0 and r != 0:
#                 errors.append(f"class {c}: active_rank must be 0 when sample_count=0, got {r}")
#             if n > 0 and not (0 <= r <= cap):
#                 errors.append(f"class {c}: active_rank={r} exceeds cap={cap} for n={n}")
#             if r > 1:
#                 e = self.eigvals[c, :r]
#                 if bool((e[:-1] + 1e-8 < e[1:]).any().item()):
#                     errors.append(f"class {c}: active eigvals must be sorted descending")

#         if band_dim > 0 and C > 0:
#             valid_rows = self.sample_counts > 0
#             row_sum = self.band_importances.sum(dim=1)
#             bad_band = valid_rows & ((row_sum - 1.0).abs() > 1e-3)
#             if bool(bad_band.any().item()):
#                 errors.append(f"band_importances rows must sum to 1 for valid classes; bad={torch.nonzero(bad_band).flatten().tolist()[:10]}")

#         if seen_classes is not None:
#             ids = _ordered_unique_ints(seen_classes)
#             valid = self.get_valid_mask()
#             missing = [c for c in ids if c < 0 or c >= C or valid.numel() <= c or not bool(valid[c].item())]
#             if missing:
#                 errors.append(f"GeometryBank missing valid rows for seen classes: {missing}")

#         result = {"ok": len(errors) == 0, "num_rows": C, "band_dim": band_dim, "errors": errors}
#         if strict and errors:
#             raise RuntimeError("GeometryBank validity check failed: " + "; ".join(errors))
#         return result

#     validate_consistency = assert_bank_valid

#     # ------------------------------------------------------------------
#     # geometry extraction / row writes
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _extract_low_rank_geometry(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
#         if data is None or not torch.is_tensor(data):
#             raise TypeError("features/data must be a tensor")
#         if data.dim() != 2 or data.size(1) != self.d_model:
#             raise ValueError(f"features must be [N,{self.d_model}], got {tuple(data.shape)}")
#         data = torch.nan_to_num(data.to(device=self.device, dtype=self._dtype()), nan=0.0, posinf=0.0, neginf=0.0)
#         n, d = int(data.size(0)), int(data.size(1))
#         if n <= 0:
#             raise ValueError("cannot extract geometry from zero samples")

#         mean = data.mean(dim=0)
#         centered = data - mean.view(1, -1)
#         total_var = centered.pow(2).sum(dim=1).mean().clamp_min(self.variance_floor)
#         avg_var = (total_var / float(max(d, 1))).clamp_min(self.variance_floor)
#         rank_cap = self._sample_count_rank_cap(n)

#         if n <= 1 or rank_cap <= 0 or self.rank <= 0:
#             active_rank = 0
#             active_basis = torch.empty((d, 0), device=self.device, dtype=self._dtype())
#             active_eigvals = torch.empty((0,), device=self.device, dtype=self._dtype())
#             residual_total_var = total_var
#             res_var = avg_var
#         else:
#             try:
#                 _, s, vh = torch.linalg.svd(centered, full_matrices=False)
#                 raw_eig = (s.pow(2) / float(max(n - 1, 1))).clamp_min(0.0)
#                 raw_basis = vh.t().contiguous()
#             except RuntimeError:
#                 cov = centered.t().matmul(centered) / float(max(n - 1, 1))
#                 evals, evecs = torch.linalg.eigh(cov)
#                 order = torch.argsort(evals, descending=True)
#                 raw_eig = evals.index_select(0, order).clamp_min(0.0)
#                 raw_basis = evecs.index_select(1, order).contiguous()

#             max_possible = min(self.rank, rank_cap, int(raw_eig.numel()), d)
#             if max_possible <= 0 or float(raw_eig[:max_possible].sum().detach().cpu().item()) <= 0:
#                 active_rank = 0
#                 active_basis = torch.empty((d, 0), device=self.device, dtype=self._dtype())
#                 active_eigvals = torch.empty((0,), device=self.device, dtype=self._dtype())
#                 residual_total_var = total_var
#                 res_var = avg_var
#             else:
#                 vals = raw_eig[:max_possible]
#                 cumulative = torch.cumsum(vals, dim=0) / vals.sum().clamp_min(_EPS)
#                 hit = (cumulative >= self.rank_energy_threshold).nonzero(as_tuple=False)
#                 energy_rank = int(hit[0].item()) + 1 if hit.numel() else max_possible
#                 strength_rank = int((vals / vals[0].clamp_min(_EPS) >= self.rank_eigen_ratio_threshold).sum().item())
#                 min_rank = min(self.min_active_rank, max_possible)
#                 active_rank = int(max(min_rank, min(energy_rank, strength_rank, max_possible)))
#                 active_basis = orthonormalize_columns(raw_basis[:, :active_rank]).to(device=self.device, dtype=self._dtype())
#                 active_eigvals = vals[:active_rank].to(device=self.device, dtype=self._dtype())

#                 shrink = self._adaptive_shrinkage(n)
#                 active_eigvals = (1.0 - shrink) * active_eigvals + shrink * avg_var
#                 active_eigvals = active_eigvals.clamp(
#                     min=self.variance_floor,
#                     max=float((avg_var * self.max_variance_ratio).detach().cpu().item()),
#                 )
#                 residual = centered - centered.matmul(active_basis).matmul(active_basis.t())
#                 residual_total_var = residual.pow(2).sum(dim=1).mean().clamp_min(self.variance_floor)
#                 res_var = (residual_total_var / float(max(d - active_rank, 1))).clamp_min(self.variance_floor)
#                 res_var = ((1.0 - 0.5 * shrink) * res_var + (0.5 * shrink) * avg_var).clamp_min(self.variance_floor)

#         basis = complete_orthonormal_basis(active_basis, self.rank)
#         eigvals = torch.full((self.rank,), float(res_var.detach().cpu().item()), device=self.device, dtype=self._dtype())
#         if active_rank > 0:
#             # Active eigvals already descending from SVD/eigh order.
#             eigvals[:active_rank] = active_eigvals[:active_rank].clamp_min(self.variance_floor)
#         sample_rel = torch.tensor(float(n) / float(n + self.reliability_sample_alpha), device=self.device, dtype=self._dtype())
#         compact_rel = (1.0 - residual_total_var / total_var.clamp_min(self.variance_floor)).clamp(self.min_reliability, 1.0)
#         rank_rel = torch.tensor(
#             self.min_reliability if rank_cap <= 0 else max(self.min_reliability, float(active_rank) / float(max(rank_cap, 1))),
#             device=self.device,
#             dtype=self._dtype(),
#         )
#         feature_rel = (0.45 * sample_rel + 0.30 * compact_rel + 0.25 * rank_rel).clamp(self.min_reliability, 1.0)

#         return {
#             "mean": mean.detach(),
#             "basis": basis.detach(),
#             "eigvals": eigvals.detach(),
#             "res_var": res_var.reshape(()).detach(),
#             "active_rank": torch.tensor(active_rank, device=self.device, dtype=torch.long),
#             "sample_count": torch.tensor(float(n), device=self.device, dtype=self._dtype()),
#             "feature_reliability": feature_rel.detach(),
#             "reliability": feature_rel.detach(),
#         }

#     @torch.no_grad()
#     def extract_geometry(
#         self,
#         features: torch.Tensor,
#         labels: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = True,
#     ) -> Dict[int, Dict[str, torch.Tensor]]:
#         """Reduce temporary feature batches into class descriptors."""
#         if features is None or labels is None:
#             return {}
#         features = torch.as_tensor(features, device=self.device, dtype=self._dtype())
#         labels = torch.as_tensor(labels, device=self.device, dtype=torch.long).flatten()
#         if features.dim() != 2 or features.size(1) != self.d_model:
#             raise ValueError(f"features must be [N,{self.d_model}], got {tuple(features.shape)}")
#         if labels.numel() != features.size(0):
#             raise ValueError(f"labels/features mismatch: {labels.numel()} vs {features.size(0)}")
#         if labels.numel() == 0:
#             return {}
#         if int(labels.min().item()) < 0:
#             raise ValueError(f"negative class labels are forbidden in GeometryBank: {torch.unique(labels).tolist()}")

#         spec = None
#         spec_physical = bool(spectral_summary_is_physical)
#         if spectral_summary is not None and torch.as_tensor(spectral_summary).numel() > 0:
#             spec = torch.as_tensor(spectral_summary, device=self.device, dtype=self._dtype())
#             if spec.dim() != 2 or spec.size(0) != features.size(0):
#                 raise ValueError(f"spectral_summary must be [N,S] aligned with features, got {tuple(spec.shape)}")
#             self._ensure_band_dim(int(spec.size(1)), self._dtype())

#         bands = None
#         if band_weights is not None and torch.as_tensor(band_weights).numel() > 0:
#             bands = torch.as_tensor(band_weights, device=self.device, dtype=self._dtype())
#             if bands.dim() != 2 or bands.size(0) != features.size(0):
#                 raise ValueError(f"band_weights must be [N,S] aligned with features, got {tuple(bands.shape)}")
#             self._ensure_band_dim(int(bands.size(1)), self._dtype())

#         out: Dict[int, Dict[str, torch.Tensor]] = {}
#         for cls_t in torch.unique(labels, sorted=True):
#             cls = int(cls_t.item())
#             mask = labels == cls_t
#             row = self._extract_low_rank_geometry(features[mask])
#             band_vec = None
#             spectral_proto = None
#             spectral_rel = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype())

#             if spec is not None:
#                 spectral_proto = spec[mask].mean(dim=0).detach()
#                 if spec_physical:
#                     spectral_rel = torch.tensor(float(mask.sum().item()) / float(mask.sum().item() + self.reliability_sample_alpha), device=self.device, dtype=self._dtype()).clamp(self.min_reliability, 1.0)
#                 else:
#                     spectral_rel = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype())
#                 band_vec = spectral_proto.abs()
#             if bands is not None:
#                 b = bands[mask].mean(dim=0).detach()
#                 band_vec = b if band_vec is None else 0.5 * band_vec + 0.5 * b

#             if band_vec is not None:
#                 b, br = self._prepare_band_vector(band_vec, int(self._band_dim.item()))
#                 row["band_importance"] = b
#                 row["band_reliability"] = br
#             if spectral_proto is not None:
#                 row["spectral_prototype"] = spectral_proto
#                 row["spectral_reliability"] = spectral_rel
#             out[cls] = row
#         return out

#     @torch.no_grad()
#     def ensure_class_count(self, count: int, spectral_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
#         count = int(max(0, count))
#         dtype = dtype or self._dtype()
#         if spectral_dim > 0:
#             self._ensure_band_dim(int(spectral_dim), dtype)
#         while len(self) < count:
#             self._append_empty_row(dtype=dtype)

#     ensure_num_classes = ensure_class_count

#     @torch.no_grad()
#     def _append_empty_row(self, dtype: Optional[torch.dtype] = None) -> None:
#         dtype = dtype or self._dtype()
#         band_dim = int(self._band_dim.item())
#         c = len(self)
#         mean = torch.zeros((1, self.d_model), device=self.device, dtype=dtype)
#         basis = complete_orthonormal_basis(torch.empty((self.d_model, 0), device=self.device, dtype=dtype), self.rank).unsqueeze(0)
#         eig = torch.full((1, self.rank), self.variance_floor, device=self.device, dtype=dtype)
#         rv = torch.full((1,), self.variance_floor, device=self.device, dtype=dtype)
#         zero_l = torch.zeros((1,), device=self.device, dtype=torch.long)
#         rel = torch.full((1,), self.min_reliability, device=self.device, dtype=dtype)
#         count0 = torch.zeros((1,), device=self.device, dtype=dtype)

#         self.means = torch.cat([self.means, mean], dim=0)
#         self.bases = torch.cat([self.bases, basis], dim=0)
#         self.eigvals = torch.cat([self.eigvals, eig], dim=0)
#         self.res_vars = torch.cat([self.res_vars, rv], dim=0)
#         self.active_ranks = torch.cat([self.active_ranks, zero_l], dim=0)
#         self.sample_counts = torch.cat([self.sample_counts, count0], dim=0)
#         self.reliability = torch.cat([self.reliability, rel], dim=0)
#         self.feature_reliability = torch.cat([self.feature_reliability, rel.clone()], dim=0)
#         self.band_reliability = torch.cat([self.band_reliability, rel.clone()], dim=0)
#         self.spectral_reliability = torch.cat([self.spectral_reliability, rel.clone()], dim=0)
#         self.phase_created = torch.cat([self.phase_created, torch.full((1,), -1, device=self.device, dtype=torch.long)], dim=0)
#         self.frozen_class_mask = torch.cat([self.frozen_class_mask, torch.zeros((1,), device=self.device, dtype=torch.bool)], dim=0)

#         if band_dim > 0:
#             self.band_importances = torch.cat([self.band_importances, torch.full((1, band_dim), 1.0 / float(band_dim), device=self.device, dtype=dtype)], dim=0)
#             self.spectral_prototypes = torch.cat([self.spectral_prototypes, torch.zeros((1, band_dim), device=self.device, dtype=dtype)], dim=0)
#             self.spectral_curve_means = torch.cat([self.spectral_curve_means, torch.zeros((1, band_dim), device=self.device, dtype=dtype)], dim=0)
#             self.spectral_curve_vars = torch.cat([self.spectral_curve_vars, torch.full((1, band_dim), self.variance_floor, device=self.device, dtype=dtype)], dim=0)
#             self.spectral_curve_d1 = torch.cat([self.spectral_curve_d1, torch.zeros((1, max(band_dim - 1, 0)), device=self.device, dtype=dtype)], dim=0)
#             self.spectral_curve_d2 = torch.cat([self.spectral_curve_d2, torch.zeros((1, max(band_dim - 2, 0)), device=self.device, dtype=dtype)], dim=0)
#             self.spectral_shape_reliability = torch.cat([self.spectral_shape_reliability, rel.clone()], dim=0)
#         else:
#             self.band_importances = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
#             self.spectral_prototypes = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
#             self.spectral_curve_means = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
#             self.spectral_curve_vars = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
#             self.spectral_curve_d1 = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
#             self.spectral_curve_d2 = torch.empty((c + 1, 0), device=self.device, dtype=dtype)
#             self.spectral_shape_reliability = torch.cat([self.spectral_shape_reliability, rel.clone()], dim=0)

#     def _assert_update_allowed(self, class_id: int, allow_frozen_update: bool = False) -> None:
#         c = self._valid_class_id(class_id, existing=False)
#         if c < len(self) and self.frozen_class_mask.numel() == len(self) and bool(self.frozen_class_mask[c].item()):
#             if not bool(allow_frozen_update):
#                 raise RuntimeError(
#                     f"Refusing to overwrite frozen GeometryBank row {c}. "
#                     "Old-class descriptors are immutable in the clean NECIL path."
#                 )

#     @torch.no_grad()
#     def add_or_update_class_geometry(
#         self,
#         class_id: int,
#         *,
#         features: Optional[torch.Tensor] = None,
#         mean: Optional[torch.Tensor] = None,
#         basis: Optional[torch.Tensor] = None,
#         eigvals: Optional[torch.Tensor] = None,
#         res_var: Optional[torch.Tensor] = None,
#         residual_variance: Optional[torch.Tensor] = None,
#         spectral_prototype: Optional[torch.Tensor] = None,
#         band_importance: Optional[torch.Tensor] = None,
#         sample_count: Optional[Union[int, float, torch.Tensor]] = None,
#         active_rank: Optional[Union[int, torch.Tensor]] = None,
#         reliability: Optional[Union[float, torch.Tensor]] = None,
#         feature_reliability: Optional[Union[float, torch.Tensor]] = None,
#         band_reliability: Optional[Union[float, torch.Tensor]] = None,
#         spectral_reliability: Optional[Union[float, torch.Tensor]] = None,
#         phase_created: int = -1,
#         freeze: bool = False,
#         allow_frozen_update: bool = False,
#     ) -> None:
#         """Add or update exactly one global class row.

#         Use features only for temporary descriptor extraction. The features are
#         not stored after this method returns.
#         """
#         c = self._valid_class_id(class_id, existing=False)
#         self._assert_update_allowed(c, allow_frozen_update=allow_frozen_update)

#         if features is not None:
#             geom = self._extract_low_rank_geometry(features)
#             mean = geom["mean"]
#             basis = geom["basis"]
#             eigvals = geom["eigvals"]
#             res_var = geom["res_var"]
#             sample_count = geom["sample_count"] if sample_count is None else sample_count
#             active_rank = geom["active_rank"] if active_rank is None else active_rank
#             feature_reliability = geom["feature_reliability"] if feature_reliability is None else feature_reliability
#             reliability = geom["reliability"] if reliability is None else reliability

#         rv = res_var if res_var is not None else residual_variance
#         if mean is None or basis is None or eigvals is None or rv is None:
#             raise ValueError("mean, basis, eigvals, and res_var are required unless features are provided")

#         # spectral/band capacity first
#         band_dim = 0
#         if spectral_prototype is not None and torch.as_tensor(spectral_prototype).numel() > 0:
#             band_dim = int(torch.as_tensor(spectral_prototype).numel())
#         if band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
#             band_dim = max(band_dim, int(torch.as_tensor(band_importance).numel()))
#         if band_dim > 0:
#             self._ensure_band_dim(band_dim, self._dtype())

#         self.ensure_class_count(c + 1, spectral_dim=band_dim, dtype=self._dtype())

#         mean_t = self._prepare_mean(mean)
#         basis_t = self._prepare_basis(basis)
#         rv_t = torch.as_tensor(rv, device=self.device, dtype=self._dtype()).reshape(()).clamp_min(self.variance_floor)
#         eig_t = self._prepare_eigvals(eigvals, rv_t)

#         count_t = torch.tensor(0.0, device=self.device, dtype=self._dtype()) if sample_count is None else torch.as_tensor(sample_count, device=self.device, dtype=self._dtype()).reshape(()).clamp_min(0.0)
#         n_i = int(float(count_t.detach().cpu().item()))
#         cap = self._sample_count_rank_cap(n_i)
#         if active_rank is None:
#             ar_t = torch.tensor(cap, device=self.device, dtype=torch.long)
#         else:
#             ar_t = torch.as_tensor(active_rank, device=self.device, dtype=torch.long).reshape(()).clamp(0, cap)
#         if n_i <= 0:
#             ar_t = torch.tensor(0, device=self.device, dtype=torch.long)

#         feat_rel_t = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype()) if feature_reliability is None else torch.as_tensor(feature_reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)
#         rel_t = feat_rel_t if reliability is None else torch.as_tensor(reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)

#         band_dim_now = int(self._band_dim.item())
#         band_t, band_rel_t = self._prepare_band_vector(band_importance, band_dim_now)
#         if band_reliability is not None:
#             band_rel_t = torch.as_tensor(band_reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)

#         spec_t = torch.empty((0,), device=self.device, dtype=self._dtype())
#         spec_rel_t = torch.tensor(self.min_reliability, device=self.device, dtype=self._dtype())
#         if band_dim_now > 0:
#             if spectral_prototype is None or torch.as_tensor(spectral_prototype).numel() == 0:
#                 spec_t = torch.zeros((band_dim_now,), device=self.device, dtype=self._dtype())
#             else:
#                 spec_t = torch.as_tensor(spectral_prototype, device=self.device, dtype=self._dtype()).flatten()
#                 if spec_t.numel() != band_dim_now:
#                     raise ValueError(f"spectral_prototype must have {band_dim_now} values, got {spec_t.numel()}")
#                 spec_t = torch.nan_to_num(spec_t, nan=0.0, posinf=0.0, neginf=0.0)
#             if spectral_reliability is not None:
#                 spec_rel_t = torch.as_tensor(spectral_reliability, device=self.device, dtype=self._dtype()).reshape(()).clamp(self.min_reliability, 1.0)
#             else:
#                 spec_rel_t = torch.tensor(self.min_reliability if spectral_prototype is None else max(self.min_reliability, float(rel_t.item())), device=self.device, dtype=self._dtype())

#         self.means[c] = mean_t
#         self.bases[c] = basis_t
#         self.eigvals[c] = eig_t
#         self.res_vars[c] = rv_t
#         self.active_ranks[c] = ar_t
#         self.sample_counts[c] = count_t
#         self.feature_reliability[c] = feat_rel_t
#         self.reliability[c] = rel_t
#         self.band_reliability[c] = band_rel_t
#         self.spectral_reliability[c] = spec_rel_t
#         self.phase_created[c] = int(phase_created)
#         if band_dim_now > 0:
#             self.band_importances[c] = band_t
#             self.spectral_prototypes[c] = spec_t
#             shape = self._spectral_shape_from_proto(spec_t)
#             self.spectral_curve_means[c] = shape["mean"]
#             self.spectral_curve_vars[c] = shape["var"]
#             if self.spectral_curve_d1.numel() > 0:
#                 self.spectral_curve_d1[c] = shape["d1"]
#             if self.spectral_curve_d2.numel() > 0:
#                 self.spectral_curve_d2[c] = shape["d2"]
#             self.spectral_shape_reliability[c] = shape["reliability"]

#         if bool(freeze):
#             self.frozen_class_mask[c] = True

#         self.assert_bank_valid(strict=True)

#     # Compatibility wrappers.
#     @torch.no_grad()
#     def add_class(self, mean, basis, eigvals, res_var, **kwargs: Any) -> None:
#         self.add_or_update_class_geometry(len(self), mean=mean, basis=basis, eigvals=eigvals, res_var=res_var, **kwargs)

#     @torch.no_grad()
#     def update_class(self, cls_id, mean, basis, eigvals, res_var, allow_frozen_update: bool = False, **kwargs: Any) -> None:
#         self.add_or_update_class_geometry(int(cls_id), mean=mean, basis=basis, eigvals=eigvals, res_var=res_var, allow_frozen_update=allow_frozen_update, **kwargs)

#     @torch.no_grad()
#     def update_class_geometry(self, class_id, mean, basis, eigvals, resvar=None, res_var=None, allow_frozen_update: bool = False, **kwargs: Any) -> None:
#         rv = res_var if res_var is not None else resvar
#         self.add_or_update_class_geometry(int(class_id), mean=mean, basis=basis, eigvals=eigvals, res_var=rv, allow_frozen_update=allow_frozen_update, **kwargs)

#     # ------------------------------------------------------------------
#     # freeze / access / bank views
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def freeze_classes(self, class_ids: Iterable[int]) -> None:
#         self.ensure_class_count(max([int(c) for c in class_ids], default=-1) + 1)
#         for c in _ordered_unique_ints(class_ids):
#             self._valid_class_id(c, existing=True)
#             self.frozen_class_mask[c] = True

#     @torch.no_grad()
#     def freeze_classes_up_to(self, count: int) -> None:
#         count = int(max(0, count))
#         self.ensure_class_count(count)
#         self.frozen_class_mask[:count] = True

#     @torch.no_grad()
#     def unfreeze_all_classes(self) -> None:
#         self.frozen_class_mask = torch.zeros((len(self),), device=self.device, dtype=torch.bool)

#     def get_valid_mask(self) -> torch.Tensor:
#         C = len(self)
#         if C == 0:
#             return torch.empty((0,), device=self.device, dtype=torch.bool)
#         finite = (
#             torch.isfinite(self.means).all(dim=1)
#             & torch.isfinite(self.bases).flatten(1).all(dim=1)
#             & torch.isfinite(self.eigvals).all(dim=1)
#             & torch.isfinite(self.res_vars)
#             & torch.isfinite(self.sample_counts)
#             & torch.isfinite(self.reliability)
#         )
#         rank_ok = torch.zeros((C,), device=self.device, dtype=torch.bool)
#         for c in range(C):
#             n = int(float(self.sample_counts[c].detach().cpu().item()))
#             r = int(self.active_ranks[c].detach().cpu().item())
#             rank_ok[c] = n > 0 and 0 <= r <= self._sample_count_rank_cap(n)
#         return finite & rank_ok & (self.sample_counts > 0)

#     def get_variances(self) -> torch.Tensor:
#         if len(self) == 0:
#             return torch.empty((0, self.rank + 1), device=self.device, dtype=self._dtype())
#         return torch.cat([self.eigvals, self.res_vars.view(-1, 1)], dim=1)

#     def get_bank(self) -> Dict[str, torch.Tensor]:
#         valid = self.get_valid_mask()
#         return {
#             "means": self.means,
#             "bases": self.bases,
#             "raw_bases": self.bases,
#             "eigvals": self.eigvals,
#             "res_vars": self.res_vars,
#             "resvars": self.res_vars,
#             "variances": self.get_variances(),
#             "active_ranks": self.active_ranks,
#             "sample_counts": self.sample_counts,
#             "reliability": self.reliability,
#             "feature_reliability": self.feature_reliability,
#             "band_importances": self.band_importances,
#             "band_importance": self.band_importances,
#             "band_reliability": self.band_reliability,
#             "spectral_prototypes": self.spectral_prototypes,
#             "spectral_protos": self.spectral_prototypes,
#             "spectral_means": self.spectral_prototypes,
#             "spectral_reliability": self.spectral_reliability,
#             "phase_created": self.phase_created,
#             "frozen_class_mask": self.frozen_class_mask,
#             "valid_mask": valid,
#             "spectral_dim": self._band_dim.clone(),
#             "spectral_curve_means": self.spectral_curve_means,
#             "spectral_curve_vars": self.spectral_curve_vars,
#             "spectral_curve_d1": self.spectral_curve_d1,
#             "spectral_curve_d2": self.spectral_curve_d2,
#             "spectral_shape_reliability": self.spectral_shape_reliability,
#         }

#     get_subspace_bank = get_bank

#     def get_seen_class_bank(self, seen_classes: Iterable[int]) -> Dict[str, torch.Tensor]:
#         ids = _ordered_unique_ints(seen_classes)
#         self.assert_bank_valid(ids, strict=True)
#         idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
#         bank = self.get_bank()
#         out: Dict[str, torch.Tensor] = {"class_ids": idx}
#         for key, value in bank.items():
#             if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == len(self):
#                 out[key] = value.index_select(0, idx)
#             elif torch.is_tensor(value):
#                 out[key] = value
#         return out

#     def get_class_geometry(self, class_id: int) -> Dict[str, torch.Tensor]:
#         c = self._valid_class_id(class_id, existing=True)
#         if self.sample_counts.numel() <= c or float(self.sample_counts[c].item()) <= 0:
#             raise RuntimeError(f"class {c} has no valid geometry row")
#         return {
#             "class_id": torch.tensor(c, device=self.device, dtype=torch.long),
#             "mean": self.means[c].detach().clone(),
#             "basis": self.bases[c].detach().clone(),
#             "eigvals": self.eigvals[c].detach().clone(),
#             "res_var": self.res_vars[c].detach().clone(),
#             "active_rank": self.active_ranks[c].detach().clone(),
#             "sample_count": self.sample_counts[c].detach().clone(),
#             "reliability": self.reliability[c].detach().clone(),
#             "band_importance": self.band_importances[c].detach().clone() if self.band_importances.numel() else torch.empty((0,), device=self.device),
#             "spectral_prototype": self.spectral_prototypes[c].detach().clone() if self.spectral_prototypes.numel() else torch.empty((0,), device=self.device),
#             "phase_created": self.phase_created[c].detach().clone(),
#             "frozen": self.frozen_class_mask[c].detach().clone(),
#         }

#     # ------------------------------------------------------------------
#     # replay sampling
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def sample_replay(
#         self,
#         class_ids: Iterable[int],
#         samples_per_class: Union[int, Mapping[int, int]] = 16,
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         label_to_local: Optional[Mapping[int, int]] = None,
#         parallel_scale: float = 1.0,
#         residual_scale: float = 0.25,
#         reliability_gated: bool = True,
#         generator: Optional[torch.Generator] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Sample synthetic old features from stored low-rank Gaussian descriptors.

#         Returns:
#             features: [M, D]
#             global_labels: [M] original sequential global class ids
#             local_labels: [M] labels mapped to current seen-class classifier columns
#         """
#         ids = _ordered_unique_ints(class_ids)
#         if label_to_local is None:
#             if seen_classes is None:
#                 label_to_local = {int(c): int(c) for c in range(len(self))}
#             else:
#                 label_to_local = {int(c): i for i, c in enumerate(_ordered_unique_ints(seen_classes))}
#         else:
#             label_to_local = {int(k): int(v) for k, v in dict(label_to_local).items()}

#         valid = self.get_valid_mask()
#         feats: List[torch.Tensor] = []
#         labs_g: List[torch.Tensor] = []
#         labs_l: List[torch.Tensor] = []
#         for c in ids:
#             self._valid_class_id(c, existing=True)
#             if valid.numel() <= c or not bool(valid[c].item()):
#                 raise RuntimeError(f"Cannot sample replay: class {c} has no valid GeometryBank row")
#             if c not in label_to_local:
#                 raise RuntimeError(f"Cannot sample replay: class {c} missing from local label mapping")
#             if isinstance(samples_per_class, Mapping):
#                 n = int(max(0, samples_per_class.get(c, 0)))
#             else:
#                 n = int(max(0, samples_per_class))
#             if n <= 0:
#                 continue

#             r = int(self.active_ranks[c].detach().cpu().item())
#             eps = torch.zeros((n, self.d_model), device=self.device, dtype=self._dtype())
#             gate = torch.tensor(1.0, device=self.device, dtype=self._dtype())
#             if bool(reliability_gated):
#                 rho = self.reliability[c].clamp(self.min_reliability, 1.0)
#                 gate = 0.20 + 0.80 * rho

#             if r > 0:
#                 z = torch.randn((n, r), device=self.device, dtype=self._dtype(), generator=generator)
#                 eig = self.eigvals[c, :r].clamp_min(self.variance_floor)
#                 eig = gate * eig + (1.0 - gate) * torch.tensor(self.variance_floor, device=self.device, dtype=self._dtype())
#                 eps = eps + z.mul(eig.sqrt().view(1, -1) * float(parallel_scale)).matmul(self.bases[c, :, :r].t())
#             res = self.res_vars[c].clamp_min(self.variance_floor)
#             res = gate * res + (1.0 - gate) * torch.tensor(self.variance_floor, device=self.device, dtype=self._dtype())
#             eps = eps + torch.randn((n, self.d_model), device=self.device, dtype=self._dtype(), generator=generator) * res.sqrt() * float(residual_scale)

#             x = self.means[c].view(1, -1) + eps
#             feats.append(x)
#             labs_g.append(torch.full((n,), c, device=self.device, dtype=torch.long))
#             labs_l.append(torch.full((n,), int(label_to_local[c]), device=self.device, dtype=torch.long))

#         if not feats:
#             return {
#                 "features": torch.empty((0, self.d_model), device=self.device, dtype=self._dtype()),
#                 "global_labels": torch.empty((0,), device=self.device, dtype=torch.long),
#                 "local_labels": torch.empty((0,), device=self.device, dtype=torch.long),
#             }
#         features = torch.cat(feats, dim=0)
#         global_labels = torch.cat(labs_g, dim=0)
#         local_labels = torch.cat(labs_l, dim=0)
#         if features.dim() != 2 or features.size(1) != self.d_model:
#             raise RuntimeError(f"sampled features have wrong shape {tuple(features.shape)}")
#         if global_labels.numel() != features.size(0) or local_labels.numel() != features.size(0):
#             raise RuntimeError("sampled replay labels/features length mismatch")
#         return {"features": features.detach(), "global_labels": global_labels.detach(), "local_labels": local_labels.detach()}

#     @torch.no_grad()
#     def sample_synthetic_features(
#         self,
#         class_ids: Optional[Iterable[int]] = None,
#         samples_per_class: int = 16,
#         parallel_scale: float = 1.0,
#         residual_scale: float = 0.25,
#         class_sample_counts: Optional[Union[Dict[int, int], torch.Tensor]] = None,
#         reliability_gated: bool = True,
#         **_: Any,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         valid = self.get_valid_mask()
#         if class_ids is None:
#             ids = [c for c in range(len(self)) if valid.numel() > c and bool(valid[c].item())]
#         else:
#             ids = _ordered_unique_ints(class_ids)

#         counts: Union[int, Mapping[int, int]]
#         if class_sample_counts is None:
#             counts = int(samples_per_class)
#         elif isinstance(class_sample_counts, dict):
#             counts = {int(k): int(v) for k, v in class_sample_counts.items()}
#         else:
#             t = torch.as_tensor(class_sample_counts).flatten()
#             counts = {int(c): int(float(t[int(c)].item())) if int(c) < t.numel() else 0 for c in ids}

#         out = self.sample_replay(
#             ids,
#             samples_per_class=counts,
#             seen_classes=list(range(len(self))),
#             parallel_scale=parallel_scale,
#             residual_scale=residual_scale,
#             reliability_gated=reliability_gated,
#         )
#         return out["features"], out["global_labels"]

#     # ------------------------------------------------------------------
#     # diagnostics / overlap
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def pairwise_center_distance(self) -> torch.Tensor:
#         C = len(self)
#         if C == 0:
#             return torch.empty((0, 0), device=self.device, dtype=self._dtype())
#         dist = torch.cdist(self.means, self.means, p=2)
#         valid = self.get_valid_mask()
#         if valid.numel() == C:
#             dist[~valid, :] = float("inf")
#             dist[:, ~valid] = float("inf")
#         dist[torch.eye(C, device=self.device, dtype=torch.bool)] = float("inf")
#         return dist

#     @torch.no_grad()
#     def pairwise_subspace_overlap(self) -> torch.Tensor:
#         C = len(self)
#         out = torch.zeros((C, C), device=self.device, dtype=self._dtype())
#         valid = self.get_valid_mask()
#         for i in range(C):
#             if valid.numel() == C and not bool(valid[i].item()):
#                 continue
#             ri = int(self.active_ranks[i].item()) if self.active_ranks.numel() > i else 0
#             if ri <= 0:
#                 continue
#             Ui = self.bases[i, :, :ri]
#             for j in range(C):
#                 if i == j or (valid.numel() == C and not bool(valid[j].item())):
#                     continue
#                 rj = int(self.active_ranks[j].item()) if self.active_ranks.numel() > j else 0
#                 if rj <= 0:
#                     continue
#                 Uj = self.bases[j, :, :rj]
#                 out[i, j] = (Ui.t().matmul(Uj).pow(2).sum() / float(max(min(ri, rj), 1))).clamp(0.0, 1.0)
#         return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

#     @torch.no_grad()
#     def pairwise_band_similarity(self) -> torch.Tensor:
#         C = len(self)
#         out = torch.zeros((C, C), device=self.device, dtype=self._dtype())
#         if self.band_importances.numel() == 0 or self.band_importances.size(1) == 0:
#             return out
#         b = self.band_importances.clamp_min(0.0)
#         b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
#         out = b.matmul(b.t()).clamp(0.0, 1.0)
#         valid = self.get_valid_mask()
#         if valid.numel() == C:
#             out[~valid, :] = 0.0
#             out[:, ~valid] = 0.0
#         out[torch.eye(C, device=self.device, dtype=torch.bool)] = 0.0
#         return out

#     @torch.no_grad()
#     def pairwise_spectral_similarity(self) -> torch.Tensor:
#         C = len(self)
#         out = torch.zeros((C, C), device=self.device, dtype=self._dtype())
#         if self.spectral_prototypes.numel() == 0 or self.spectral_prototypes.size(1) == 0:
#             return out
#         s = self.spectral_prototypes
#         s = s - s.mean(dim=1, keepdim=True)
#         s = s / s.norm(dim=1, keepdim=True).clamp_min(1e-8)
#         out = s.matmul(s.t()).clamp(0.0, 1.0)
#         valid = self.get_valid_mask()
#         if valid.numel() == C:
#             out[~valid, :] = 0.0
#             out[:, ~valid] = 0.0
#         out[torch.eye(C, device=self.device, dtype=torch.bool)] = 0.0
#         return out

#     pairwise_spectral_shape_similarity = pairwise_spectral_similarity

#     @torch.no_grad()
#     def geometry_conflict_matrix(
#         self,
#         center_margin: float = 1.0,
#         subspace_weight: float = 1.0,
#         band_weight: float = 0.25,
#         spectral_shape_weight: float = 0.25,
#         reliability_weighted: bool = True,
#         **_: Any,
#     ) -> torch.Tensor:
#         C = len(self)
#         if C == 0:
#             return torch.empty((0, 0), device=self.device, dtype=self._dtype())
#         center = self.pairwise_center_distance()
#         center_risk = torch.relu(float(center_margin) - center) / max(float(center_margin), 1e-8)
#         risk = torch.nan_to_num(center_risk, nan=0.0, posinf=0.0, neginf=0.0)
#         risk = risk + float(subspace_weight) * self.pairwise_subspace_overlap()
#         risk = risk + float(band_weight) * self.pairwise_band_similarity()
#         risk = risk + float(spectral_shape_weight) * self.pairwise_spectral_similarity()
#         if reliability_weighted and self.reliability.numel() == C:
#             rel = self.reliability.clamp(self.min_reliability, 1.0)
#             uncertainty = 2.0 - rel.view(-1, 1) - rel.view(1, -1)
#             risk = risk * (1.0 + 0.5 * uncertainty.clamp(0.0, 2.0))
#         valid = self.get_valid_mask()
#         if valid.numel() == C:
#             risk[~valid, :] = 0.0
#             risk[:, ~valid] = 0.0
#         risk[torch.eye(C, device=self.device, dtype=torch.bool)] = 0.0
#         return torch.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)

#     @torch.no_grad()
#     def old_new_subspace_overlap_report(
#         self,
#         old_class_count: int,
#         new_class_ids: Optional[Iterable[int]] = None,
#     ) -> Dict[str, Any]:
#         old_count = int(max(0, min(int(old_class_count), len(self))))
#         if old_count <= 0 or old_count >= len(self):
#             return {"max_overlap": 0.0, "mean_overlap": 0.0, "pair": None, "old_class_id": None, "new_class_id": None}
#         valid = self.get_valid_mask()
#         old_ids = [c for c in range(old_count) if valid.numel() > c and bool(valid[c].item())]
#         if new_class_ids is None:
#             new_ids = [c for c in range(old_count, len(self)) if valid.numel() > c and bool(valid[c].item())]
#         else:
#             new_ids = [int(c) for c in new_class_ids if 0 <= int(c) < len(self) and valid.numel() > int(c) and bool(valid[int(c)].item())]
#         if not old_ids or not new_ids:
#             return {"max_overlap": 0.0, "mean_overlap": 0.0, "pair": None, "old_class_id": None, "new_class_id": None}
#         sub = self.pairwise_subspace_overlap()
#         vals = sub[torch.as_tensor(old_ids, device=self.device)][:, torch.as_tensor(new_ids, device=self.device)]
#         flat = int(vals.argmax().item())
#         i = flat // int(vals.size(1))
#         j = flat % int(vals.size(1))
#         return {
#             "max_overlap": float(vals.max().item()),
#             "mean_overlap": float(vals.mean().item()),
#             "pair": (int(old_ids[i]), int(new_ids[j])),
#             "old_class_id": int(old_ids[i]),
#             "new_class_id": int(new_ids[j]),
#         }

#     @torch.no_grad()
#     def compute_geometry_diagnostics(
#         self,
#         seen_classes: Optional[Iterable[int]] = None,
#         old_class_ids: Optional[Iterable[int]] = None,
#         new_class_ids: Optional[Iterable[int]] = None,
#         reference_snapshot: Optional[Dict[str, torch.Tensor]] = None,
#         center_margin: float = 1.0,
#     ) -> Dict[str, Any]:
#         if seen_classes is None:
#             seen = [c for c in range(len(self)) if bool(self.get_valid_mask()[c].item())] if len(self) else []
#         else:
#             seen = _ordered_unique_ints(seen_classes)
#         self.assert_bank_valid(seen, strict=True) if seen else self.assert_bank_valid(strict=True)

#         valid = self.get_valid_mask()
#         ids = [c for c in seen if 0 <= c < len(self) and valid.numel() > c and bool(valid[c].item())]
#         center_dist = self.pairwise_center_distance()
#         sub_overlap = self.pairwise_subspace_overlap()
#         conflict = self.geometry_conflict_matrix(center_margin=center_margin)

#         finite_center = center_dist[torch.isfinite(center_dist)]
#         diag: Dict[str, Any] = {
#             "num_rows": int(len(self)),
#             "num_valid_rows": int(valid.sum().item()) if valid.numel() else 0,
#             "seen_classes": ids,
#             "center_distance_min": float(finite_center.min().item()) if finite_center.numel() else 0.0,
#             "center_distance_mean": float(finite_center.mean().item()) if finite_center.numel() else 0.0,
#             "subspace_overlap_max": float(sub_overlap.max().item()) if sub_overlap.numel() else 0.0,
#             "subspace_overlap_mean": float(sub_overlap[sub_overlap > 0].mean().item()) if bool((sub_overlap > 0).any().item()) else 0.0,
#             "residual_variance_min": float(self.res_vars[valid].min().item()) if bool(valid.any().item()) else 0.0,
#             "residual_variance_mean": float(self.res_vars[valid].mean().item()) if bool(valid.any().item()) else 0.0,
#             "residual_variance_max": float(self.res_vars[valid].max().item()) if bool(valid.any().item()) else 0.0,
#             "geometry_conflict_max": float(conflict.max().item()) if conflict.numel() else 0.0,
#             "geometry_conflict_mean": float(conflict[conflict > 0].mean().item()) if bool((conflict > 0).any().item()) else 0.0,
#             "completeness_ok": True,
#             "missing_seen_classes": [],
#         }

#         if seen:
#             missing = [c for c in seen if c < 0 or c >= len(self) or valid.numel() <= c or not bool(valid[c].item())]
#             diag["missing_seen_classes"] = missing
#             diag["completeness_ok"] = len(missing) == 0

#         if old_class_ids is not None and new_class_ids is not None:
#             old_ids = [c for c in _ordered_unique_ints(old_class_ids) if c in ids]
#             new_ids = [c for c in _ordered_unique_ints(new_class_ids) if c in ids]
#             if old_ids and new_ids:
#                 old_t = torch.as_tensor(old_ids, device=self.device, dtype=torch.long)
#                 new_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
#                 ov = sub_overlap.index_select(0, old_t).index_select(1, new_t)
#                 rk = conflict.index_select(0, old_t).index_select(1, new_t)
#                 diag["old_new_overlap_max"] = float(ov.max().item()) if ov.numel() else 0.0
#                 diag["old_new_overlap_mean"] = float(ov.mean().item()) if ov.numel() else 0.0
#                 diag["old_new_conflict_max"] = float(rk.max().item()) if rk.numel() else 0.0
#                 diag["old_new_conflict_mean"] = float(rk.mean().item()) if rk.numel() else 0.0
#             else:
#                 diag["old_new_overlap_max"] = 0.0
#                 diag["old_new_overlap_mean"] = 0.0
#                 diag["old_new_conflict_max"] = 0.0
#                 diag["old_new_conflict_mean"] = 0.0

#         if reference_snapshot is not None:
#             drift = self.compare_snapshot(reference_snapshot, class_ids=seen)
#             diag.update({f"drift_{k}": v for k, v in drift.items()})

#         return diag

#     geometry_diagnostics = compute_geometry_diagnostics

#     @torch.no_grad()
#     def top_geometry_conflicts(self, k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
#         risk = self.geometry_conflict_matrix(**kwargs)
#         C = int(risk.size(0))
#         if C <= 1:
#             return []
#         mask = torch.triu(torch.ones_like(risk, dtype=torch.bool), diagonal=1)
#         vals = risk[mask]
#         if vals.numel() == 0:
#             return []
#         pairs = mask.nonzero(as_tuple=False)
#         top_vals, top_idx = torch.topk(vals, k=min(int(k), vals.numel()))
#         center = self.pairwise_center_distance()
#         sub = self.pairwise_subspace_overlap()
#         out: List[Dict[str, Any]] = []
#         for score, pos in zip(top_vals.detach().cpu(), top_idx.detach().cpu()):
#             i, j = pairs[int(pos.item())].tolist()
#             out.append({
#                 "class_i": int(i),
#                 "class_j": int(j),
#                 "conflict": float(score.item()),
#                 "center_distance": float(center[i, j].item()) if torch.isfinite(center[i, j]) else float("inf"),
#                 "subspace_overlap": float(sub[i, j].item()),
#             })
#         return out

#     # ------------------------------------------------------------------
#     # snapshots / drift diagnostics
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def export_snapshot(self) -> Dict[str, torch.Tensor]:
#         return {
#             "means": self.means.detach().clone(),
#             "bases": self.bases.detach().clone(),
#             "eigvals": self.eigvals.detach().clone(),
#             "res_vars": self.res_vars.detach().clone(),
#             "active_ranks": self.active_ranks.detach().clone(),
#             "sample_counts": self.sample_counts.detach().clone(),
#             "reliability": self.reliability.detach().clone(),
#             "feature_reliability": self.feature_reliability.detach().clone(),
#             "band_importances": self.band_importances.detach().clone(),
#             "band_reliability": self.band_reliability.detach().clone(),
#             "spectral_prototypes": self.spectral_prototypes.detach().clone(),
#             "spectral_reliability": self.spectral_reliability.detach().clone(),
#             "phase_created": self.phase_created.detach().clone(),
#             "frozen_class_mask": self.frozen_class_mask.detach().clone(),
#             "band_dim": torch.tensor(int(self._band_dim.item()), device=self.device, dtype=torch.long),
#         }

#     @torch.no_grad()
#     def load_snapshot(self, snapshot: Dict[str, torch.Tensor], strict: bool = True) -> None:
#         if not snapshot:
#             if strict:
#                 raise ValueError("empty GeometryBank snapshot")
#             return
#         required = ("means", "bases", "eigvals", "res_vars")
#         missing = [k for k in required if k not in snapshot]
#         if missing:
#             if strict:
#                 raise ValueError(f"snapshot missing keys: {missing}")
#             return

#         dtype = self._dtype()
#         means = torch.as_tensor(snapshot["means"], device=self.device, dtype=dtype)
#         bases = torch.as_tensor(snapshot["bases"], device=self.device, dtype=dtype)
#         eigvals = torch.as_tensor(snapshot["eigvals"], device=self.device, dtype=dtype)
#         res_vars = torch.as_tensor(snapshot["res_vars"], device=self.device, dtype=dtype).flatten()
#         if means.dim() != 2 or means.size(1) != self.d_model:
#             raise ValueError(f"snapshot means must be [C,{self.d_model}], got {tuple(means.shape)}")
#         C = int(means.size(0))
#         band_dim = int(torch.as_tensor(snapshot.get("band_dim", 0)).item()) if "band_dim" in snapshot else 0
#         bands = snapshot.get("band_importances", None)
#         if bands is not None and torch.as_tensor(bands).numel() > 0:
#             band_dim = int(torch.as_tensor(bands).shape[1])
#         self.reset_storage(band_dim=band_dim, dtype=dtype)
#         self.ensure_class_count(C, spectral_dim=band_dim, dtype=dtype)

#         self.means.copy_(means)
#         self.bases.copy_(torch.stack([self._prepare_basis(bases[c]) for c in range(C)], dim=0))
#         self.eigvals.copy_(torch.stack([self._prepare_eigvals(eigvals[c], res_vars[c]) for c in range(C)], dim=0))
#         self.res_vars.copy_(res_vars.clamp_min(self.variance_floor))
#         for key in ("active_ranks", "sample_counts", "reliability", "feature_reliability", "band_reliability", "spectral_reliability", "phase_created", "frozen_class_mask"):
#             if key in snapshot and torch.as_tensor(snapshot[key]).numel() == getattr(self, key).numel():
#                 getattr(self, key).copy_(torch.as_tensor(snapshot[key], device=self.device, dtype=getattr(self, key).dtype).reshape_as(getattr(self, key)))
#         if band_dim > 0:
#             if "band_importances" in snapshot and torch.as_tensor(snapshot["band_importances"]).shape == self.band_importances.shape:
#                 self.band_importances.copy_(torch.as_tensor(snapshot["band_importances"], device=self.device, dtype=dtype))
#             if "spectral_prototypes" in snapshot and torch.as_tensor(snapshot["spectral_prototypes"]).shape == self.spectral_prototypes.shape:
#                 self.spectral_prototypes.copy_(torch.as_tensor(snapshot["spectral_prototypes"], device=self.device, dtype=dtype))
#         self.assert_bank_valid(strict=True)

#     @torch.no_grad()
#     def reset_storage(self, band_dim: int = 0, dtype: Optional[torch.dtype] = None) -> None:
#         dtype = dtype or self._dtype()
#         dev = self.device
#         band_dim = int(max(0, band_dim))
#         self.means = torch.empty((0, self.d_model), device=dev, dtype=dtype)
#         self.bases = torch.empty((0, self.d_model, self.rank), device=dev, dtype=dtype)
#         self.eigvals = torch.empty((0, self.rank), device=dev, dtype=dtype)
#         self.res_vars = torch.empty((0,), device=dev, dtype=dtype)
#         self.active_ranks = torch.empty((0,), dtype=torch.long, device=dev)
#         self.sample_counts = torch.empty((0,), device=dev, dtype=dtype)
#         self.reliability = torch.empty((0,), device=dev, dtype=dtype)
#         self.feature_reliability = torch.empty((0,), device=dev, dtype=dtype)
#         self.band_reliability = torch.empty((0,), device=dev, dtype=dtype)
#         self.spectral_reliability = torch.empty((0,), device=dev, dtype=dtype)
#         self.phase_created = torch.empty((0,), dtype=torch.long, device=dev)
#         self.frozen_class_mask = torch.empty((0,), dtype=torch.bool, device=dev)
#         self._band_dim = torch.tensor(band_dim, dtype=torch.long, device=dev)
#         self.band_importances = torch.empty((0, band_dim), device=dev, dtype=dtype)
#         self.spectral_prototypes = torch.empty((0, band_dim), device=dev, dtype=dtype)
#         self.spectral_curve_means = torch.empty((0, band_dim), device=dev, dtype=dtype)
#         self.spectral_curve_vars = torch.empty((0, band_dim), device=dev, dtype=dtype)
#         self.spectral_curve_d1 = torch.empty((0, max(band_dim - 1, 0)), device=dev, dtype=dtype)
#         self.spectral_curve_d2 = torch.empty((0, max(band_dim - 2, 0)), device=dev, dtype=dtype)
#         self.spectral_shape_reliability = torch.empty((0,), device=dev, dtype=dtype)

#     @torch.no_grad()
#     def compare_snapshot(self, snapshot: Dict[str, torch.Tensor], class_ids: Optional[Iterable[int]] = None) -> Dict[str, Any]:
#         if not snapshot or "means" not in snapshot or "bases" not in snapshot:
#             return {"center_drift_mean": 0.0, "center_drift_max": 0.0, "basis_drift_mean": 0.0, "basis_drift_max": 0.0}
#         C0 = int(torch.as_tensor(snapshot["means"]).size(0))
#         C = min(C0, len(self))
#         ids = list(range(C)) if class_ids is None else [int(c) for c in class_ids if 0 <= int(c) < C]
#         if not ids:
#             return {"center_drift_mean": 0.0, "center_drift_max": 0.0, "basis_drift_mean": 0.0, "basis_drift_max": 0.0}
#         idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
#         old_means = torch.as_tensor(snapshot["means"], device=self.device, dtype=self._dtype()).index_select(0, idx)
#         new_means = self.means.index_select(0, idx)
#         center = (new_means - old_means).norm(dim=1)

#         old_bases = torch.as_tensor(snapshot["bases"], device=self.device, dtype=self._dtype()).index_select(0, idx)
#         new_bases = self.bases.index_select(0, idx)
#         active_old = torch.as_tensor(snapshot.get("active_ranks", self.active_ranks[:C]), device=self.device, dtype=torch.long).flatten()
#         basis_drifts = []
#         for local_i, c in enumerate(ids):
#             r = int(min(self.active_ranks[c].item(), active_old[c].item(), self.rank))
#             if r <= 0:
#                 basis_drifts.append(torch.tensor(0.0, device=self.device, dtype=self._dtype()))
#             else:
#                 ov = old_bases[local_i, :, :r].t().matmul(new_bases[local_i, :, :r]).pow(2).sum() / float(max(r, 1))
#                 basis_drifts.append((1.0 - ov.clamp(0.0, 1.0)).detach())
#         basis = torch.stack(basis_drifts) if basis_drifts else torch.zeros((1,), device=self.device, dtype=self._dtype())

#         eig_drift = torch.zeros((1,), device=self.device, dtype=self._dtype())
#         if "eigvals" in snapshot:
#             old_e = torch.as_tensor(snapshot["eigvals"], device=self.device, dtype=self._dtype()).index_select(0, idx)
#             new_e = self.eigvals.index_select(0, idx)
#             eig_drift = (new_e - old_e).abs().mean(dim=1)
#         rv_drift = torch.zeros((1,), device=self.device, dtype=self._dtype())
#         if "res_vars" in snapshot:
#             old_rv = torch.as_tensor(snapshot["res_vars"], device=self.device, dtype=self._dtype()).flatten().index_select(0, idx)
#             new_rv = self.res_vars.index_select(0, idx)
#             rv_drift = (new_rv - old_rv).abs()

#         return {
#             "center_drift_mean": float(center.mean().item()),
#             "center_drift_max": float(center.max().item()),
#             "basis_drift_mean": float(basis.mean().item()),
#             "basis_drift_max": float(basis.max().item()),
#             "eigval_drift_mean": float(eig_drift.mean().item()),
#             "eigval_drift_max": float(eig_drift.max().item()),
#             "resvar_drift_mean": float(rv_drift.mean().item()),
#             "resvar_drift_max": float(rv_drift.max().item()),
#         }

#     # ------------------------------------------------------------------
#     # candidate descriptor insertion and clean descriptor transport
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def build_candidate_geometry_rows(
#         self,
#         features: torch.Tensor,
#         labels: torch.Tensor,
#         spectral_summary: Optional[torch.Tensor] = None,
#         band_weights: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = True,
#         class_ids: Optional[Iterable[int]] = None,
#         **_: Any,
#     ) -> Dict[int, Dict[str, torch.Tensor]]:
#         rows = self.extract_geometry(features, labels, spectral_summary=spectral_summary, band_weights=band_weights, spectral_summary_is_physical=spectral_summary_is_physical)
#         if class_ids is None:
#             return rows
#         allowed = set(_ordered_unique_ints(class_ids))
#         return {int(c): g for c, g in rows.items() if int(c) in allowed}

#     @torch.no_grad()
#     def commit_candidate_geometry_rows(
#         self,
#         candidate_rows: Dict[int, Dict[str, torch.Tensor]],
#         *,
#         allow_frozen_update: bool = False,
#         phase_created: int = -1,
#         freeze: bool = False,
#         context: str = "candidate_commit",
#     ) -> Dict[str, Any]:
#         committed: List[int] = []
#         for c in sorted(int(k) for k in candidate_rows.keys()):
#             g = candidate_rows[c]
#             self.add_or_update_class_geometry(
#                 c,
#                 mean=g["mean"],
#                 basis=g["basis"],
#                 eigvals=g["eigvals"],
#                 res_var=g["res_var"],
#                 spectral_prototype=g.get("spectral_prototype"),
#                 band_importance=g.get("band_importance"),
#                 sample_count=g.get("sample_count"),
#                 active_rank=g.get("active_rank"),
#                 reliability=g.get("reliability"),
#                 feature_reliability=g.get("feature_reliability"),
#                 band_reliability=g.get("band_reliability"),
#                 spectral_reliability=g.get("spectral_reliability"),
#                 phase_created=phase_created,
#                 freeze=freeze,
#                 allow_frozen_update=allow_frozen_update,
#             )
#             committed.append(c)
#         return {"active": len(committed), "committed_class_ids": committed, "context": str(context)}

#     @torch.no_grad()
#     def correct_new_descriptors_against_old(
#         self,
#         old_class_count: int,
#         new_class_ids: Iterable[int],
#         *,
#         overlap_threshold: float = 0.60,
#         mean_push: float = 0.10,
#         basis_projection_strength: float = 0.35,
#         variance_shrink: float = 0.10,
#         topk_old: int = 3,
#         **_: Any,
#     ) -> Dict[str, Any]:
#         """Descriptor-level transport: move only new descriptors away from old geometry.

#         This is the meaningful transport operation for the clean NECIL method.
#         Old rows are never changed.
#         """
#         old_count = int(max(0, min(int(old_class_count), len(self))))
#         new_ids = [int(c) for c in _ordered_unique_ints(new_class_ids) if 0 <= int(c) < len(self)]
#         if old_count <= 0 or not new_ids:
#             return {"active": 0, "corrected_class_ids": [], "max_overlap_before": 0.0, "max_overlap_after": 0.0}

#         valid = self.get_valid_mask()
#         old_ids = [c for c in range(old_count) if valid.numel() > c and bool(valid[c].item())]
#         new_ids = [c for c in new_ids if valid.numel() > c and bool(valid[c].item())]
#         if not old_ids or not new_ids:
#             return {"active": 0, "corrected_class_ids": [], "max_overlap_before": 0.0, "max_overlap_after": 0.0}

#         before = self.pairwise_subspace_overlap()
#         old_t = torch.as_tensor(old_ids, device=self.device, dtype=torch.long)
#         new_t = torch.as_tensor(new_ids, device=self.device, dtype=torch.long)
#         ov_before = before.index_select(0, old_t).index_select(1, new_t)
#         corrected: List[int] = []

#         for cls in new_ids:
#             self._assert_update_allowed(cls, allow_frozen_update=False)
#             scores = before.index_select(0, old_t)[:, cls]
#             if scores.numel() == 0 or float(scores.max().item()) <= float(overlap_threshold):
#                 continue
#             k = min(int(max(1, topk_old)), int(scores.numel()))
#             _, pos = torch.topk(scores, k=k, largest=True)
#             U = self.bases[cls].detach().clone()
#             mu = self.means[cls].detach().clone()
#             eig = self.eigvals[cls].detach().clone()
#             rv = self.res_vars[cls].detach().clone()
#             projector = torch.zeros((self.d_model, self.d_model), device=self.device, dtype=self._dtype())
#             push = torch.zeros((self.d_model,), device=self.device, dtype=self._dtype())

#             for p in pos.tolist():
#                 old_cls = old_ids[int(p)]
#                 r = int(self.active_ranks[old_cls].item())
#                 if r > 0:
#                     Uo = self.bases[old_cls, :, :r]
#                     projector = projector + Uo.matmul(Uo.t()) / float(k)
#                 direction = mu - self.means[old_cls]
#                 direction = direction / direction.norm().clamp_min(1e-8)
#                 push = push + direction / float(k)

#             U_new = complete_orthonormal_basis(U - float(basis_projection_strength) * projector.matmul(U), self.rank)
#             mu_new = mu + float(mean_push) * push / push.norm().clamp_min(1e-8)
#             eig_new = (eig * (1.0 - float(variance_shrink))).clamp_min(self.variance_floor)
#             rv_new = (rv * (1.0 - 0.5 * float(variance_shrink))).clamp_min(self.variance_floor)

#             self.add_or_update_class_geometry(
#                 cls,
#                 mean=mu_new,
#                 basis=U_new,
#                 eigvals=eig_new,
#                 res_var=rv_new,
#                 spectral_prototype=self.spectral_prototypes[cls] if self.spectral_prototypes.numel() else None,
#                 band_importance=self.band_importances[cls] if self.band_importances.numel() else None,
#                 sample_count=self.sample_counts[cls],
#                 active_rank=self.active_ranks[cls],
#                 reliability=self.reliability[cls],
#                 feature_reliability=self.feature_reliability[cls],
#                 phase_created=int(self.phase_created[cls].item()),
#                 allow_frozen_update=False,
#             )
#             corrected.append(cls)

#         after = self.pairwise_subspace_overlap()
#         ov_after = after.index_select(0, old_t).index_select(1, new_t)
#         return {
#             "active": len(corrected),
#             "corrected_class_ids": corrected,
#             "max_overlap_before": float(ov_before.max().item()) if ov_before.numel() else 0.0,
#             "max_overlap_after": float(ov_after.max().item()) if ov_after.numel() else 0.0,
#         }

#     def transport_frozen_geometry(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
#         raise RuntimeError(
#             "transport_frozen_geometry is disabled in the clean NECIL-HSI bank. "
#             "Do not move old frozen rows. Use correct_new_descriptors_against_old() "
#             "to transport only new descriptors away from old geometry."
#         )

#     validated_insert_with_transport = build_candidate_geometry_rows

#     # ------------------------------------------------------------------
#     # risk-weighted replay compatibility
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def old_replay_risk_weights(
#         self,
#         old_class_count: int,
#         new_class_ids: Optional[Iterable[int]] = None,
#         min_weight: float = 0.25,
#         max_weight: float = 3.0,
#         **kwargs: Any,
#     ) -> torch.Tensor:
#         old_count = int(max(0, min(int(old_class_count), len(self))))
#         if old_count <= 0:
#             return torch.empty((0,), device=self.device, dtype=self._dtype())
#         weights = torch.ones((old_count,), device=self.device, dtype=self._dtype())
#         if len(self) <= old_count:
#             return weights
#         if new_class_ids is None:
#             new_ids = [c for c in range(old_count, len(self)) if bool(self.get_valid_mask()[c].item())]
#         else:
#             new_ids = [int(c) for c in new_class_ids if 0 <= int(c) < len(self)]
#         if not new_ids:
#             return weights
#         risk = self.geometry_conflict_matrix(**kwargs)
#         vals = risk[:old_count].index_select(1, torch.as_tensor(new_ids, device=self.device, dtype=torch.long)).max(dim=1).values
#         if float(vals.max().item()) <= 0:
#             return weights
#         weights = vals / vals.mean().clamp_min(1e-8)
#         return weights.clamp(float(min_weight), float(max_weight)).detach()

#     @torch.no_grad()
#     def old_replay_sample_counts(
#         self,
#         old_class_count: int,
#         new_class_ids: Optional[Iterable[int]] = None,
#         base_samples_per_class: int = 16,
#         min_samples_per_class: int = 4,
#         max_multiplier: float = 3.0,
#         **kwargs: Any,
#     ) -> Dict[int, int]:
#         old_count = int(max(0, min(int(old_class_count), len(self))))
#         if old_count <= 0:
#             return {}
#         weights = self.old_replay_risk_weights(old_count, new_class_ids, max_weight=max_multiplier, **kwargs)
#         valid = self.get_valid_mask()
#         out: Dict[int, int] = {}
#         for c in range(old_count):
#             if valid.numel() <= c or not bool(valid[c].item()):
#                 out[c] = 0
#             else:
#                 n = int(round(int(base_samples_per_class) * float(weights[c].item())))
#                 out[c] = max(int(min_samples_per_class), min(n, int(round(base_samples_per_class * max_multiplier))))
#         return out

#     @torch.no_grad()
#     def sample_risk_weighted_old_features(
#         self,
#         old_class_count: int,
#         new_class_ids: Optional[Iterable[int]] = None,
#         base_samples_per_class: int = 16,
#         min_samples_per_class: int = 4,
#         max_multiplier: float = 3.0,
#         parallel_scale: float = 1.0,
#         residual_scale: float = 0.25,
#         reliability_gated: bool = True,
#         **risk_kwargs: Any,
#     ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
#         counts = self.old_replay_sample_counts(
#             old_class_count,
#             new_class_ids,
#             base_samples_per_class=base_samples_per_class,
#             min_samples_per_class=min_samples_per_class,
#             max_multiplier=max_multiplier,
#             **risk_kwargs,
#         )
#         x, y = self.sample_synthetic_features(
#             class_ids=range(int(old_class_count)),
#             class_sample_counts=counts,
#             parallel_scale=parallel_scale,
#             residual_scale=residual_scale,
#             reliability_gated=reliability_gated,
#         )
#         return x, y, counts

#     @torch.no_grad()
#     def memory_cost_summary(self, bytes_per_float: int = 4) -> Dict[str, Any]:
#         tensors = {
#             "means": self.means,
#             "bases": self.bases,
#             "eigvals": self.eigvals,
#             "res_vars": self.res_vars,
#             "sample_counts": self.sample_counts,
#             "reliability": self.reliability,
#             "band_importances": self.band_importances,
#             "spectral_prototypes": self.spectral_prototypes,
#         }
#         elems = {k: int(v.numel()) for k, v in tensors.items() if torch.is_tensor(v)}
#         total = int(sum(elems.values()))
#         return {
#             "num_rows": int(len(self)),
#             "num_valid_rows": int(self.get_valid_mask().sum().item()) if len(self) else 0,
#             "feature_dim": int(self.d_model),
#             "rank": int(self.rank),
#             "band_dim": int(self._band_dim.item()),
#             "actual_float_elements": total,
#             "actual_fp32_kb": float(total * int(bytes_per_float) / 1024.0),
#             "component_float_elements": elems,
#             "stores_raw_samples": False,
#         }

#     @torch.no_grad()
#     def geometry_health_summary(self, class_names: Optional[Sequence[str]] = None, topk_bands: int = 5) -> Dict[str, Any]:
#         rows: List[Dict[str, Any]] = []
#         valid = self.get_valid_mask()
#         for c in range(len(self)):
#             rows.append({
#                 "class_id": int(c),
#                 "class_name": str(class_names[c]) if class_names is not None and c < len(class_names) else None,
#                 "valid": bool(valid[c].item()) if valid.numel() > c else False,
#                 "sample_count": float(self.sample_counts[c].item()) if self.sample_counts.numel() > c else 0.0,
#                 "active_rank": int(self.active_ranks[c].item()) if self.active_ranks.numel() > c else 0,
#                 "res_var": float(self.res_vars[c].item()) if self.res_vars.numel() > c else 0.0,
#                 "reliability": float(self.reliability[c].item()) if self.reliability.numel() > c else 0.0,
#                 "phase_created": int(self.phase_created[c].item()) if self.phase_created.numel() > c else -1,
#                 "frozen": bool(self.frozen_class_mask[c].item()) if self.frozen_class_mask.numel() > c else False,
#             })
#         return {
#             "num_rows": int(len(self)),
#             "num_valid_rows": int(valid.sum().item()) if valid.numel() else 0,
#             "class_geometry": rows,
#             "global_geometry": self.compute_geometry_diagnostics(),
#             "memory_cost": self.memory_cost_summary(),
#         }
