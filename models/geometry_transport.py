from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


_EPS = 1e-12


@dataclass(frozen=True)
class TransportWeights:
    """Weights for spectral-guided low-rank safe insertion."""

    center: float = 0.20
    mahalanobis: float = 0.35
    subspace: float = 0.25
    spectral: float = 0.10
    band: float = 0.05
    volume: float = 0.05

    def normalized(self) -> "TransportWeights":
        vals = [
            max(float(self.center), 0.0),
            max(float(self.mahalanobis), 0.0),
            max(float(self.subspace), 0.0),
            max(float(self.spectral), 0.0),
            max(float(self.band), 0.0),
            max(float(self.volume), 0.0),
        ]
        s = sum(vals)
        if s <= 0.0:
            return TransportWeights()
        return TransportWeights(*(v / s for v in vals))


# -----------------------------------------------------------------------------
# Validation utilities
# -----------------------------------------------------------------------------


def _finite_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.numel() == 0:
        raise ValueError(f"{name} is empty")
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().detach().cpu().item())
        raise ValueError(f"{name} contains {bad} NaN/Inf values")
    return x


def _as_1d(x: torch.Tensor, name: str, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    x = _finite_tensor(x, name)
    if x.dim() == 0:
        x = x.view(1)
    if x.dim() != 1:
        raise ValueError(f"{name} must be [C], got {tuple(x.shape)}")
    if device is not None or dtype is not None:
        x = x.to(device=device if device is not None else x.device, dtype=dtype if dtype is not None else x.dtype)
    return x


def _as_2d(x: torch.Tensor, name: str, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    x = _finite_tensor(x, name)
    if x.dim() != 2:
        raise ValueError(f"{name} must be [C,D], got {tuple(x.shape)}")
    if device is not None or dtype is not None:
        x = x.to(device=device if device is not None else x.device, dtype=dtype if dtype is not None else x.dtype)
    return x


def _as_3d(x: torch.Tensor, name: str, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    x = _finite_tensor(x, name)
    if x.dim() != 3:
        raise ValueError(f"{name} must be [C,D,R], got {tuple(x.shape)}")
    if device is not None or dtype is not None:
        x = x.to(device=device if device is not None else x.device, dtype=dtype if dtype is not None else x.dtype)
    return x


def _prepare_active_ranks(active_ranks: Optional[torch.Tensor], C: int, R: int, device: torch.device) -> torch.Tensor:
    if active_ranks is None or not torch.is_tensor(active_ranks) or active_ranks.numel() != C:
        return torch.full((C,), int(R), device=device, dtype=torch.long)
    return active_ranks.to(device=device).long().flatten().clamp(min=0, max=R)


@torch.no_grad()
def orthonormalize_basis(basis: torch.Tensor, active_rank: Optional[int] = None) -> torch.Tensor:
    """Return a basis with orthonormal active columns and zero inactive columns."""
    if not torch.is_tensor(basis) or basis.dim() != 2:
        raise ValueError(f"basis must be [D,R], got {None if basis is None else tuple(basis.shape)}")
    U = torch.nan_to_num(basis.float(), nan=0.0, posinf=0.0, neginf=0.0)
    D, R = int(U.size(0)), int(U.size(1))
    r = R if active_rank is None else int(max(0, min(int(active_rank), R, D)))
    out = torch.zeros_like(U)
    if r <= 0:
        return out
    X = U[:, :r]
    try:
        q, _ = torch.linalg.qr(X, mode="reduced")
        out[:, :r] = q[:, :r]
    except RuntimeError:
        try:
            q, _, _ = torch.linalg.svd(X, full_matrices=False)
            out[:, :r] = q[:, :r]
        except RuntimeError:
            # Deterministic fallback: first r coordinate axes.
            out[:r, :r] = torch.eye(r, device=U.device, dtype=U.dtype)
    return out


@torch.no_grad()
def assert_descriptor_bank_valid(
    means: torch.Tensor,
    bases: torch.Tensor,
    eigvals: torch.Tensor,
    res_vars: torch.Tensor,
    *,
    active_ranks: Optional[torch.Tensor] = None,
    class_ids: Optional[Iterable[int]] = None,
    name: str = "geometry",
    atol: float = 5e-3,
) -> None:
    """Validate compact low-rank GeometryBank-style descriptors."""
    means = _as_2d(means, f"{name}.means")
    bases = _as_3d(bases, f"{name}.bases", device=means.device, dtype=means.dtype)
    eigvals = _as_2d(eigvals, f"{name}.eigvals", device=means.device, dtype=means.dtype)
    res_vars = _as_1d(res_vars, f"{name}.res_vars", device=means.device, dtype=means.dtype)
    C, D = int(means.size(0)), int(means.size(1))
    if bases.size(0) != C or bases.size(1) != D:
        raise ValueError(f"{name}: bases shape {tuple(bases.shape)} incompatible with means {tuple(means.shape)}")
    if eigvals.size(0) != C or eigvals.size(1) != bases.size(2):
        raise ValueError(f"{name}: eigvals shape {tuple(eigvals.shape)} incompatible with bases {tuple(bases.shape)}")
    if res_vars.numel() != C:
        raise ValueError(f"{name}: res_vars must have C={C} entries, got {res_vars.numel()}")
    if (eigvals < -float(atol)).any():
        raise ValueError(f"{name}: eigenvalues must be non-negative")
    if (res_vars < -float(atol)).any():
        raise ValueError(f"{name}: residual variances must be non-negative")
    if class_ids is not None:
        ids = [int(c) for c in class_ids]
        if len(ids) != C:
            raise ValueError(f"{name}: class_ids length {len(ids)} does not match C={C}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: class_ids contain duplicates: {ids}")
        if any(c < 0 for c in ids):
            raise ValueError(f"{name}: class_ids must be non-negative global ids")
    ar = _prepare_active_ranks(active_ranks, C, int(bases.size(2)), means.device)
    for c in range(C):
        r = int(ar[c].detach().cpu().item())
        if r <= 0:
            continue
        U = bases[c, :, :r]
        gram = U.t().matmul(U)
        err = (gram - torch.eye(r, device=gram.device, dtype=gram.dtype)).abs().max()
        if float(err.detach().cpu().item()) > float(atol):
            raise ValueError(f"{name}: basis row {c} is not orthonormal over active rank {r}; max_err={float(err):.6f}")
        if not torch.all(eigvals[c, :r][:-1] >= eigvals[c, :r][1:] - float(atol)):
            raise ValueError(f"{name}: eigenvalues for row {c} must be sorted descending over active rank")


# -----------------------------------------------------------------------------
# Low-rank geometry operations
# -----------------------------------------------------------------------------


@torch.no_grad()
def low_rank_mahalanobis_energy(
    points: torch.Tensor,
    means: torch.Tensor,
    bases: torch.Tensor,
    eigvals: torch.Tensor,
    res_vars: torch.Tensor,
    *,
    active_ranks: Optional[torch.Tensor] = None,
    variance_floor: float = 1e-4,
    normalize_by_dim: bool = True,
) -> torch.Tensor:
    """Energy of points under low-rank residual Gaussian descriptors.

    Output is [N,C]. Lower means the point lies deeper in a class ellipsoid.
    """
    points = _as_2d(points, "points")
    means = _as_2d(means, "means", device=points.device, dtype=points.dtype)
    bases = _as_3d(bases, "bases", device=points.device, dtype=points.dtype)
    eigvals = _as_2d(eigvals, "eigvals", device=points.device, dtype=points.dtype).clamp_min(float(variance_floor))
    res_vars = _as_1d(res_vars, "res_vars", device=points.device, dtype=points.dtype).clamp_min(float(variance_floor))
    C, D, R = int(bases.size(0)), int(bases.size(1)), int(bases.size(2))
    if points.size(1) != D or means.shape != (C, D) or eigvals.shape != (C, R):
        raise ValueError("low_rank_mahalanobis_energy: incompatible descriptor shapes")
    ar = _prepare_active_ranks(active_ranks, C, R, points.device)
    mask = (torch.arange(R, device=points.device).view(1, R) < ar.view(C, 1)).to(points.dtype)

    delta = points.unsqueeze(1) - means.unsqueeze(0)                  # [N,C,D]
    coeff = torch.einsum("ncd,cdr->ncr", delta, bases) * mask.unsqueeze(0)
    recon = torch.einsum("ncr,cdr->ncd", coeff, bases)
    residual = delta - recon
    parallel = ((coeff.pow(2) / eigvals.unsqueeze(0)) * mask.unsqueeze(0)).sum(dim=-1)
    orthogonal = residual.pow(2).sum(dim=-1) / res_vars.view(1, C)
    energy = parallel + orthogonal
    if bool(normalize_by_dim):
        energy = energy / max(D, 1)
    return torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=0.0)


@torch.no_grad()
def subspace_overlap_matrix(
    bases_a: torch.Tensor,
    bases_b: torch.Tensor,
    *,
    active_ranks_a: Optional[torch.Tensor] = None,
    active_ranks_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise active-rank subspace overlap in [0,1]."""
    A = _as_3d(bases_a, "bases_a")
    B = _as_3d(bases_b, "bases_b", device=A.device, dtype=A.dtype)
    if A.size(1) != B.size(1):
        raise ValueError(f"basis feature dimension mismatch: {A.size(1)} vs {B.size(1)}")
    Ca, _, Ra = A.shape
    Cb, _, Rb = B.shape
    ara = _prepare_active_ranks(active_ranks_a, int(Ca), int(Ra), A.device)
    arb = _prepare_active_ranks(active_ranks_b, int(Cb), int(Rb), A.device)
    out = torch.zeros((Ca, Cb), device=A.device, dtype=A.dtype)
    for i in range(Ca):
        ri = int(ara[i].detach().cpu().item())
        if ri <= 0:
            continue
        Ui = A[i, :, :ri]
        for j in range(Cb):
            rj = int(arb[j].detach().cpu().item())
            if rj <= 0:
                continue
            Uj = B[j, :, :rj]
            out[i, j] = Ui.t().matmul(Uj).pow(2).sum() / float(max(min(ri, rj), 1))
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


@torch.no_grad()
def low_rank_log_volume(
    eigvals: torch.Tensor,
    res_vars: torch.Tensor,
    *,
    active_ranks: Optional[torch.Tensor] = None,
    feature_dim: Optional[int] = None,
    variance_floor: float = 1e-4,
    normalize_by_dim: bool = True,
) -> torch.Tensor:
    eigvals = _as_2d(eigvals, "eigvals")
    res_vars = _as_1d(res_vars, "res_vars", device=eigvals.device, dtype=eigvals.dtype)
    C, R = int(eigvals.size(0)), int(eigvals.size(1))
    D = int(feature_dim or R)
    ar = _prepare_active_ranks(active_ranks, C, R, eigvals.device).clamp(max=min(D, R))
    mask = (torch.arange(R, device=eigvals.device).view(1, R) < ar.view(C, 1)).to(eigvals.dtype)
    log_eig = eigvals.clamp_min(float(variance_floor)).log()
    log_res = res_vars.clamp_min(float(variance_floor)).log()
    volume = (log_eig * mask).sum(dim=1) + (D - ar.to(eigvals.dtype)) * log_res
    if bool(normalize_by_dim):
        volume = volume / max(D, 1)
    return torch.nan_to_num(volume, nan=0.0, posinf=1e6, neginf=-1e6)


@torch.no_grad()
def positive_cosine_similarity(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Positive cosine similarity for spectral prototypes or band signatures."""
    if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b):
        return None
    if a.numel() == 0 or b.numel() == 0:
        return None
    A = _as_2d(a, "a")
    B = _as_2d(b, "b", device=A.device, dtype=A.dtype)
    if A.size(1) != B.size(1):
        return None
    A = F.normalize(torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-8)
    B = F.normalize(torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-8)
    return A.matmul(B.t()).clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Transport-risk construction
# -----------------------------------------------------------------------------


@torch.no_grad()
def compute_safe_insertion_cost(
    *,
    new_means: torch.Tensor,
    new_bases: torch.Tensor,
    new_eigvals: torch.Tensor,
    new_res_vars: torch.Tensor,
    old_means: torch.Tensor,
    old_bases: torch.Tensor,
    old_eigvals: torch.Tensor,
    old_res_vars: torch.Tensor,
    new_active_ranks: Optional[torch.Tensor] = None,
    old_active_ranks: Optional[torch.Tensor] = None,
    new_spectral: Optional[torch.Tensor] = None,
    old_spectral: Optional[torch.Tensor] = None,
    new_band: Optional[torch.Tensor] = None,
    old_band: Optional[torch.Tensor] = None,
    weights: TransportWeights = TransportWeights(),
    center_margin: float = 1.0,
    old_energy_margin: float = 1.0,
    subspace_overlap_target: float = 0.35,
    max_new_log_volume_over_old: float = 0.25,
    variance_floor: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Compute old/new risk matrix and a row-stochastic safe-insertion plan.

    Risk is high when a new descriptor lies too close to or inside frozen old
    geometry, has an aligned tangent subspace, has a confusing spectral/band
    signature, or owns an overly broad covariance volume.
    """
    new_means = _as_2d(new_means, "new_means")
    old_means = _as_2d(old_means, "old_means", device=new_means.device, dtype=new_means.dtype)
    new_bases = _as_3d(new_bases, "new_bases", device=new_means.device, dtype=new_means.dtype)
    old_bases = _as_3d(old_bases, "old_bases", device=new_means.device, dtype=new_means.dtype)
    new_eigvals = _as_2d(new_eigvals, "new_eigvals", device=new_means.device, dtype=new_means.dtype)
    old_eigvals = _as_2d(old_eigvals, "old_eigvals", device=new_means.device, dtype=new_means.dtype)
    new_res_vars = _as_1d(new_res_vars, "new_res_vars", device=new_means.device, dtype=new_means.dtype)
    old_res_vars = _as_1d(old_res_vars, "old_res_vars", device=new_means.device, dtype=new_means.dtype)
    assert_descriptor_bank_valid(new_means, new_bases, new_eigvals, new_res_vars, active_ranks=new_active_ranks, name="new")
    assert_descriptor_bank_valid(old_means, old_bases, old_eigvals, old_res_vars, active_ranks=old_active_ranks, name="old")

    N, D = int(new_means.size(0)), int(new_means.size(1))
    O = int(old_means.size(0))
    w = weights.normalized()

    dist = torch.cdist(new_means, old_means, p=2) / max(D ** 0.5, 1.0)
    center_component = F.relu(float(center_margin) - dist) / max(float(center_margin), _EPS)

    old_energy = low_rank_mahalanobis_energy(
        new_means,
        old_means,
        old_bases,
        old_eigvals,
        old_res_vars,
        active_ranks=old_active_ranks,
        variance_floor=float(variance_floor),
        normalize_by_dim=True,
    )
    mahal_component = F.relu(float(old_energy_margin) - old_energy) / max(float(old_energy_margin), _EPS)

    subspace = subspace_overlap_matrix(
        new_bases,
        old_bases,
        active_ranks_a=new_active_ranks,
        active_ranks_b=old_active_ranks,
    )
    subspace_component = F.relu(subspace - float(subspace_overlap_target)) / max(1.0 - float(subspace_overlap_target), _EPS)

    geom_gate = (center_component + mahal_component + subspace_component).clamp(0.0, 3.0) / 3.0
    spec_sim = positive_cosine_similarity(new_spectral, old_spectral)
    if spec_sim is None:
        spectral_component = torch.zeros((N, O), device=new_means.device, dtype=new_means.dtype)
    else:
        spectral_component = spec_sim.to(device=new_means.device, dtype=new_means.dtype) * geom_gate.detach()
    band_sim = positive_cosine_similarity(new_band, old_band)
    if band_sim is None:
        band_component = torch.zeros((N, O), device=new_means.device, dtype=new_means.dtype)
    else:
        band_component = band_sim.to(device=new_means.device, dtype=new_means.dtype) * geom_gate.detach()

    new_vol = low_rank_log_volume(
        new_eigvals,
        new_res_vars,
        active_ranks=new_active_ranks,
        feature_dim=D,
        variance_floor=float(variance_floor),
    )
    old_vol = low_rank_log_volume(
        old_eigvals,
        old_res_vars,
        active_ranks=old_active_ranks,
        feature_dim=D,
        variance_floor=float(variance_floor),
    )
    volume_component = F.relu(new_vol.view(N, 1) - old_vol.view(1, O) - float(max_new_log_volume_over_old))

    risk = (
        float(w.center) * center_component
        + float(w.mahalanobis) * mahal_component
        + float(w.subspace) * subspace_component
        + float(w.spectral) * spectral_component
        + float(w.band) * band_component
        + float(w.volume) * volume_component
    )
    risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
    row_sum = risk.sum(dim=1, keepdim=True)
    plan = torch.where(row_sum > 0.0, risk / row_sum.clamp_min(_EPS), torch.zeros_like(risk))

    return {
        "risk": risk,
        "transport_plan": plan,
        "center_component": center_component,
        "mahalanobis_component": mahal_component,
        "subspace_component": subspace_component,
        "spectral_component": spectral_component,
        "band_component": band_component,
        "volume_component": volume_component,
        "old_energy_of_new_means": old_energy,
        "center_distance": dist,
        "subspace_overlap": subspace,
        "new_log_volume": new_vol,
        "old_log_volume": old_vol,
    }


# -----------------------------------------------------------------------------
# Safe descriptor insertion
# -----------------------------------------------------------------------------


@torch.no_grad()
def _deterministic_escape_direction(
    new_mu: torch.Tensor,
    weighted_old_mu: torch.Tensor,
    weighted_old_basis: Optional[torch.Tensor],
    row_index: int,
) -> torch.Tensor:
    direction = new_mu - weighted_old_mu
    if direction.norm() > 1e-8:
        return F.normalize(direction, dim=0, eps=1e-8)
    D = int(new_mu.numel())
    eye = torch.eye(D, device=new_mu.device, dtype=new_mu.dtype)
    v = eye[int(row_index) % D]
    if weighted_old_basis is not None and weighted_old_basis.numel() > 0:
        P = weighted_old_basis.matmul(weighted_old_basis.t())
        v = v - P.matmul(v)
    if v.norm() <= 1e-8:
        v = eye[(int(row_index) + 1) % D]
    return F.normalize(v, dim=0, eps=1e-8)


@torch.no_grad()
def _project_basis_away_from_old(
    U_new: torch.Tensor,
    old_bases: torch.Tensor,
    plan_row: torch.Tensor,
    *,
    active_rank_new: int,
    active_ranks_old: torch.Tensor,
    basis_projection_strength: float,
) -> torch.Tensor:
    D, R = int(U_new.size(0)), int(U_new.size(1))
    r_new = int(max(0, min(int(active_rank_new), R, D)))
    if r_new <= 0:
        return torch.zeros_like(U_new)
    strength = float(max(0.0, min(float(basis_projection_strength), 1.0)))
    if strength <= 0.0 or float(plan_row.sum().detach().cpu().item()) <= 0.0:
        return orthonormalize_basis(U_new, r_new)
    P = torch.zeros((D, D), device=U_new.device, dtype=U_new.dtype)
    for o in range(int(old_bases.size(0))):
        w = float(plan_row[o].detach().cpu().item())
        if w <= 0.0:
            continue
        ro = int(active_ranks_old[o].detach().cpu().item())
        if ro <= 0:
            continue
        Uo = old_bases[o, :, :ro]
        P = P + w * Uo.matmul(Uo.t())
    risk_strength = float(plan_row.sum().clamp(0.0, 1.0).detach().cpu().item())
    candidate = U_new.clone()
    candidate[:, :r_new] = U_new[:, :r_new] - strength * risk_strength * P.matmul(U_new[:, :r_new])
    corrected = orthonormalize_basis(candidate, r_new)
    # Preserve identity by blending basis columns, then reorthonormalizing.
    beta = min(strength * risk_strength, 0.85)
    blended = (1.0 - beta) * U_new + beta * corrected
    return orthonormalize_basis(blended, r_new)


@torch.no_grad()
def safe_insert_new_geometry(
    *,
    new_class_ids: Iterable[int],
    old_class_ids: Iterable[int],
    new_means: torch.Tensor,
    new_bases: torch.Tensor,
    new_eigvals: torch.Tensor,
    new_res_vars: torch.Tensor,
    old_means: torch.Tensor,
    old_bases: torch.Tensor,
    old_eigvals: torch.Tensor,
    old_res_vars: torch.Tensor,
    new_active_ranks: Optional[torch.Tensor] = None,
    old_active_ranks: Optional[torch.Tensor] = None,
    new_spectral: Optional[torch.Tensor] = None,
    old_spectral: Optional[torch.Tensor] = None,
    new_band: Optional[torch.Tensor] = None,
    old_band: Optional[torch.Tensor] = None,
    weights: TransportWeights = TransportWeights(),
    center_margin: float = 1.0,
    old_energy_margin: float = 1.0,
    subspace_overlap_target: float = 0.35,
    max_new_log_volume_over_old: float = 0.25,
    mean_push: float = 0.35,
    max_mean_shift: float = 1.25,
    basis_projection_strength: float = 0.55,
    variance_shrink: float = 0.20,
    residual_shrink: float = 0.15,
    variance_floor: float = 1e-4,
) -> Dict[str, torch.Tensor | Dict[str, float]]:
    """Insert new descriptors into frozen old geometry without moving old rows.

    The returned descriptors are new-class descriptors only. Old descriptors are
    used only as frozen constraints and are never modified.
    """
    new_ids = [int(c) for c in new_class_ids]
    old_ids = [int(c) for c in old_class_ids]
    if not new_ids:
        raise ValueError("new_class_ids must not be empty")
    if len(set(new_ids)) != len(new_ids) or len(set(old_ids)) != len(old_ids):
        raise ValueError("class ids must be unique global ids")
    if any(c < 0 for c in new_ids + old_ids):
        raise ValueError("class ids must be non-negative global ids")
    if set(new_ids).intersection(old_ids):
        raise ValueError("new_class_ids and old_class_ids must be disjoint")

    new_means = _as_2d(new_means, "new_means").detach().clone()
    old_means_in = _as_2d(old_means, "old_means", device=new_means.device, dtype=new_means.dtype)
    old_means_snapshot = old_means_in.detach().clone()
    old_bases_snapshot = _as_3d(old_bases, "old_bases", device=new_means.device, dtype=new_means.dtype).detach().clone()
    old_eig_snapshot = _as_2d(old_eigvals, "old_eigvals", device=new_means.device, dtype=new_means.dtype).detach().clone()
    old_res_snapshot = _as_1d(old_res_vars, "old_res_vars", device=new_means.device, dtype=new_means.dtype).detach().clone()

    new_bases = _as_3d(new_bases, "new_bases", device=new_means.device, dtype=new_means.dtype).detach().clone()
    new_eigvals = _as_2d(new_eigvals, "new_eigvals", device=new_means.device, dtype=new_means.dtype).detach().clone().clamp_min(float(variance_floor))
    new_res_vars = _as_1d(new_res_vars, "new_res_vars", device=new_means.device, dtype=new_means.dtype).detach().clone().clamp_min(float(variance_floor))

    N, D, R = int(new_bases.size(0)), int(new_bases.size(1)), int(new_bases.size(2))
    O = int(old_bases_snapshot.size(0))
    if len(new_ids) != N or len(old_ids) != O:
        raise ValueError(f"class id lengths must match descriptor rows: new {len(new_ids)} vs {N}, old {len(old_ids)} vs {O}")

    old_ar = _prepare_active_ranks(old_active_ranks, O, int(old_bases_snapshot.size(2)), new_means.device)
    new_ar = _prepare_active_ranks(new_active_ranks, N, R, new_means.device)
    for n in range(N):
        new_bases[n] = orthonormalize_basis(new_bases[n], int(new_ar[n].item()))
    for o in range(O):
        old_bases_snapshot[o] = orthonormalize_basis(old_bases_snapshot[o], int(old_ar[o].item()))

    assert_descriptor_bank_valid(new_means, new_bases, new_eigvals, new_res_vars, active_ranks=new_ar, class_ids=new_ids, name="new_before")
    assert_descriptor_bank_valid(old_means_snapshot, old_bases_snapshot, old_eig_snapshot, old_res_snapshot, active_ranks=old_ar, class_ids=old_ids, name="old_frozen")

    cost = compute_safe_insertion_cost(
        new_means=new_means,
        new_bases=new_bases,
        new_eigvals=new_eigvals,
        new_res_vars=new_res_vars,
        old_means=old_means_snapshot,
        old_bases=old_bases_snapshot,
        old_eigvals=old_eig_snapshot,
        old_res_vars=old_res_snapshot,
        new_active_ranks=new_ar,
        old_active_ranks=old_ar,
        new_spectral=new_spectral,
        old_spectral=old_spectral,
        new_band=new_band,
        old_band=old_band,
        weights=weights,
        center_margin=float(center_margin),
        old_energy_margin=float(old_energy_margin),
        subspace_overlap_target=float(subspace_overlap_target),
        max_new_log_volume_over_old=float(max_new_log_volume_over_old),
        variance_floor=float(variance_floor),
    )
    plan = cost["transport_plan"]
    risk = cost["risk"]

    before_old_energy = cost["old_energy_of_new_means"].clone()
    before_subspace = cost["subspace_overlap"].clone()
    before_volume = cost["new_log_volume"].clone()

    out_means = new_means.clone()
    out_bases = new_bases.clone()
    out_eigvals = new_eigvals.clone()
    out_res_vars = new_res_vars.clone()

    old_volume = cost["old_log_volume"]
    old_volume_cap = old_volume.max().detach() + float(max_new_log_volume_over_old)

    for n in range(N):
        p = plan[n]
        risk_strength = float(risk[n].sum().clamp(0.0, 1.0).detach().cpu().item())
        if risk_strength <= 0.0 or float(p.sum().detach().cpu().item()) <= 0.0:
            continue
        weighted_old_mu = p.matmul(old_means_snapshot)
        # Use a small weighted basis for deterministic fallback direction only.
        worst_o = int(risk[n].argmax().detach().cpu().item())
        ro = int(old_ar[worst_o].detach().cpu().item())
        worst_basis = old_bases_snapshot[worst_o, :, :ro] if ro > 0 else None
        direction = _deterministic_escape_direction(new_means[n], weighted_old_mu, worst_basis, n)
        min_old_energy = float(before_old_energy[n].min().detach().cpu().item())
        energy_deficit = max(0.0, float(old_energy_margin) - min_old_energy) / max(float(old_energy_margin), _EPS)
        push = float(mean_push) * max(risk_strength, energy_deficit)
        push = min(push, float(max_mean_shift))
        shifted = new_means[n] + push * direction
        delta = shifted - new_means[n]
        if delta.norm() > float(max_mean_shift):
            delta = delta * (float(max_mean_shift) / delta.norm().clamp_min(1e-8))
        out_means[n] = new_means[n] + delta

        out_bases[n] = _project_basis_away_from_old(
            new_bases[n],
            old_bases_snapshot,
            p,
            active_rank_new=int(new_ar[n].item()),
            active_ranks_old=old_ar,
            basis_projection_strength=float(basis_projection_strength),
        )

        shrink = 1.0 - float(variance_shrink) * risk_strength
        res_shrink = 1.0 - float(residual_shrink) * risk_strength
        shrink = max(shrink, 0.25)
        res_shrink = max(res_shrink, 0.25)
        out_eigvals[n] = (out_eigvals[n] * shrink).clamp_min(float(variance_floor))
        out_res_vars[n] = (out_res_vars[n] * res_shrink).clamp_min(float(variance_floor))

        cur_vol = low_rank_log_volume(
            out_eigvals[n : n + 1],
            out_res_vars[n : n + 1],
            active_ranks=new_ar[n : n + 1],
            feature_dim=D,
            variance_floor=float(variance_floor),
        )[0]
        if float(cur_vol.detach().cpu().item()) > float(old_volume_cap.detach().cpu().item()):
            excess = (cur_vol - old_volume_cap).clamp_min(0.0)
            # Multiplying all variances by exp(-excess) reduces normalized logdet.
            factor = torch.exp(-excess).clamp(0.10, 1.0)
            out_eigvals[n] = (out_eigvals[n] * factor).clamp_min(float(variance_floor))
            out_res_vars[n] = (out_res_vars[n] * factor).clamp_min(float(variance_floor))

    # Sort eigenvalues descending and keep basis columns aligned only approximately.
    # Because this module only shrinks eigenvalues elementwise, sorting normally preserves order.
    out_eigvals = torch.sort(out_eigvals.clamp_min(float(variance_floor)), dim=1, descending=True).values
    out_res_vars = out_res_vars.clamp_min(float(variance_floor))
    for n in range(N):
        out_bases[n] = orthonormalize_basis(out_bases[n], int(new_ar[n].item()))

    assert_descriptor_bank_valid(out_means, out_bases, out_eigvals, out_res_vars, active_ranks=new_ar, class_ids=new_ids, name="new_after")
    # Old rows must be bitwise unchanged inside this function.
    if not torch.equal(old_means_snapshot, old_means_in.detach().clone()):
        raise RuntimeError("old_means changed during safe insertion")

    after_cost = compute_safe_insertion_cost(
        new_means=out_means,
        new_bases=out_bases,
        new_eigvals=out_eigvals,
        new_res_vars=out_res_vars,
        old_means=old_means_snapshot,
        old_bases=old_bases_snapshot,
        old_eigvals=old_eig_snapshot,
        old_res_vars=old_res_snapshot,
        new_active_ranks=new_ar,
        old_active_ranks=old_ar,
        new_spectral=new_spectral,
        old_spectral=old_spectral,
        new_band=new_band,
        old_band=old_band,
        weights=weights,
        center_margin=float(center_margin),
        old_energy_margin=float(old_energy_margin),
        subspace_overlap_target=float(subspace_overlap_target),
        max_new_log_volume_over_old=float(max_new_log_volume_over_old),
        variance_floor=float(variance_floor),
    )

    shift = (out_means - new_means).norm(dim=1)
    after_old_energy = after_cost["old_energy_of_new_means"]
    after_subspace = after_cost["subspace_overlap"]
    after_volume = after_cost["new_log_volume"]
    diagnostics = {
        "max_risk_before": float(risk.max().detach().cpu().item()) if risk.numel() else 0.0,
        "mean_risk_before": float(risk.mean().detach().cpu().item()) if risk.numel() else 0.0,
        "max_risk_after": float(after_cost["risk"].max().detach().cpu().item()) if risk.numel() else 0.0,
        "mean_risk_after": float(after_cost["risk"].mean().detach().cpu().item()) if risk.numel() else 0.0,
        "min_old_energy_before": float(before_old_energy.min().detach().cpu().item()),
        "min_old_energy_after": float(after_old_energy.min().detach().cpu().item()),
        "max_subspace_overlap_before": float(before_subspace.max().detach().cpu().item()),
        "max_subspace_overlap_after": float(after_subspace.max().detach().cpu().item()),
        "mean_center_shift": float(shift.mean().detach().cpu().item()),
        "max_center_shift": float(shift.max().detach().cpu().item()),
        "mean_log_volume_before": float(before_volume.mean().detach().cpu().item()),
        "mean_log_volume_after": float(after_volume.mean().detach().cpu().item()),
        "safe": float(
            (float(after_cost["risk"].max().detach().cpu().item()) <= float(risk.max().detach().cpu().item()) + 1e-6)
            and (float(after_old_energy.min().detach().cpu().item()) >= float(before_old_energy.min().detach().cpu().item()) - 1e-6)
        ),
    }

    return {
        "class_ids": torch.tensor(new_ids, device=new_means.device, dtype=torch.long),
        "means": out_means,
        "bases": out_bases,
        "eigvals": out_eigvals,
        "res_vars": out_res_vars,
        "active_ranks": new_ar,
        "transport_plan": plan,
        "risk_before": risk,
        "risk_after": after_cost["risk"],
        "cost_before": cost,
        "cost_after": after_cost,
        "diagnostics": diagnostics,
    }


# -----------------------------------------------------------------------------
# Diagnostics / reports
# -----------------------------------------------------------------------------


@torch.no_grad()
def safe_insertion_diagnostics(
    *,
    before_means: torch.Tensor,
    before_bases: torch.Tensor,
    before_eigvals: torch.Tensor,
    before_res_vars: torch.Tensor,
    after_means: torch.Tensor,
    after_bases: torch.Tensor,
    after_eigvals: torch.Tensor,
    after_res_vars: torch.Tensor,
    old_means: torch.Tensor,
    old_bases: torch.Tensor,
    old_eigvals: torch.Tensor,
    old_res_vars: torch.Tensor,
    before_active_ranks: Optional[torch.Tensor] = None,
    after_active_ranks: Optional[torch.Tensor] = None,
    old_active_ranks: Optional[torch.Tensor] = None,
    variance_floor: float = 1e-4,
) -> Dict[str, float]:
    before_energy = low_rank_mahalanobis_energy(
        before_means, old_means, old_bases, old_eigvals, old_res_vars,
        active_ranks=old_active_ranks, variance_floor=variance_floor,
    )
    after_energy = low_rank_mahalanobis_energy(
        after_means, old_means, old_bases, old_eigvals, old_res_vars,
        active_ranks=old_active_ranks, variance_floor=variance_floor,
    )
    before_overlap = subspace_overlap_matrix(before_bases, old_bases, active_ranks_a=before_active_ranks, active_ranks_b=old_active_ranks)
    after_overlap = subspace_overlap_matrix(after_bases, old_bases, active_ranks_a=after_active_ranks, active_ranks_b=old_active_ranks)
    shift = (after_means - before_means.to(after_means.device, after_means.dtype)).norm(dim=1)
    return {
        "min_old_energy_before": float(before_energy.min().detach().cpu().item()),
        "min_old_energy_after": float(after_energy.min().detach().cpu().item()),
        "mean_old_energy_before": float(before_energy.mean().detach().cpu().item()),
        "mean_old_energy_after": float(after_energy.mean().detach().cpu().item()),
        "max_subspace_overlap_before": float(before_overlap.max().detach().cpu().item()) if before_overlap.numel() else 0.0,
        "max_subspace_overlap_after": float(after_overlap.max().detach().cpu().item()) if after_overlap.numel() else 0.0,
        "mean_center_shift": float(shift.mean().detach().cpu().item()),
        "max_center_shift": float(shift.max().detach().cpu().item()),
    }


# -----------------------------------------------------------------------------
# Legacy feature-drift transport: disabled in the clean method
# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_ridge_transport(*args, **kwargs):
    raise RuntimeError(
        "estimate_ridge_transport is disabled in the clean Spectral-Guided Low-Rank Geometry Preservation path. "
        "It estimates old-model to new-model feature drift and encourages moving old geometry. "
        "Use safe_insert_new_geometry(...) to move only new descriptors relative to frozen old geometry."
    )


@torch.no_grad()
def estimate_gls_transport(*args, **kwargs):
    raise RuntimeError(
        "estimate_gls_transport is disabled in the clean Spectral-Guided Low-Rank Geometry Preservation path. "
        "Use safe_insert_new_geometry(...) instead."
    )


@torch.no_grad()
def project_transport_to_low_rank_residual(*args, **kwargs):
    raise RuntimeError(
        "Affine residual transport is not part of the clean new-descriptor insertion module. "
        "Use safe_insert_new_geometry(...), which preserves old rows and modifies only new descriptors."
    )


@torch.no_grad()
def transport_diagnostics(*args, **kwargs) -> Dict[str, float]:
    raise RuntimeError(
        "transport_diagnostics for old-model feature drift is disabled. Use safe_insertion_diagnostics(...)."
    )


__all__ = [
    "TransportWeights",
    "orthonormalize_basis",
    "assert_descriptor_bank_valid",
    "low_rank_mahalanobis_energy",
    "subspace_overlap_matrix",
    "low_rank_log_volume",
    "positive_cosine_similarity",
    "compute_safe_insertion_cost",
    "safe_insert_new_geometry",
    "safe_insertion_diagnostics",
    # legacy names intentionally fail fast
    "estimate_ridge_transport",
    "estimate_gls_transport",
    "project_transport_to_low_rank_residual",
    "transport_diagnostics",
]















# from __future__ import annotations

# from typing import Dict, Tuple

# import torch


# _EPS = 1e-12


# def _finite_2d(x: torch.Tensor, name: str) -> torch.Tensor:
#     if not torch.is_tensor(x):
#         raise TypeError(f"{name} must be a torch.Tensor.")
#     if x.dim() != 2:
#         raise ValueError(f"{name} must be [N,D], got {tuple(x.shape)}")
#     if x.numel() == 0:
#         raise ValueError(f"{name} is empty.")
#     return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


# def _stable_solve(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
#     """Solve lhs @ X = rhs with Cholesky/solve/pinv fallback."""
#     lhs = torch.nan_to_num(lhs.float(), nan=0.0, posinf=0.0, neginf=0.0)
#     rhs = torch.nan_to_num(rhs.float(), nan=0.0, posinf=0.0, neginf=0.0)
#     try:
#         return torch.linalg.solve(lhs, rhs)
#     except RuntimeError:
#         try:
#             L = torch.linalg.cholesky(lhs)
#             return torch.cholesky_solve(rhs, L)
#         except RuntimeError:
#             return torch.linalg.pinv(lhs).matmul(rhs)


# def _clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
#     v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
#     max_norm = float(max(float(max_norm), 1e-8))
#     n = v.norm()
#     if torch.isfinite(n) and float(n.detach().cpu().item()) > max_norm:
#         v = v * (max_norm / n.clamp_min(1e-8))
#     return v


# @torch.no_grad()
# def project_transport_to_low_rank_residual(
#     A: torch.Tensor,
#     *,
#     low_rank: int = 4,
#     max_delta_fro: float = 1.5,
# ) -> torch.Tensor:
#     """Return ``I + Delta`` with low-rank, norm-clipped residual drift.

#     This is not just a safety clamp.  It enforces the transport model used by the
#     HSI architecture: old descriptors may evolve only through a compact residual
#     subspace, matching the low-rank GeometryBank memory.
#     """
#     if not torch.is_tensor(A):
#         raise TypeError("A must be a torch.Tensor.")
#     if A.dim() != 2 or A.size(0) != A.size(1):
#         raise ValueError(f"A must be square [D,D], got {tuple(A.shape)}")
#     A = torch.nan_to_num(A.float(), nan=0.0, posinf=0.0, neginf=0.0)
#     d = int(A.size(0))
#     eye = torch.eye(d, device=A.device, dtype=A.dtype)
#     delta = A - eye

#     r = int(max(0, min(int(low_rank), d)))
#     if r <= 0:
#         delta = torch.zeros_like(delta)
#     else:
#         try:
#             U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
#             delta = (U[:, :r] * S[:r].view(1, -1)).matmul(Vh[:r])
#         except RuntimeError:
#             delta = torch.zeros_like(delta)

#     max_fro = float(max(float(max_delta_fro), 1e-8))
#     fro = delta.norm()
#     if torch.isfinite(fro) and float(fro.detach().cpu().item()) > max_fro:
#         delta = delta * (max_fro / fro.clamp_min(1e-8))
#     return torch.nan_to_num(eye + delta, nan=0.0, posinf=0.0, neginf=0.0)


# def _adaptive_ridge_value(x: torch.Tensor, ridge: float) -> float:
#     """Scale ridge by feature covariance trace so it is stable across datasets."""
#     ridge = float(max(float(ridge), 0.0))
#     if x.numel() == 0:
#         return max(ridge, 1e-6)
#     denom = float(max(int(x.size(0)) - 1, 1))
#     # Mean feature variance per dimension.  This avoids the same numeric ridge
#     # being too weak for SA/PU and too strong after normalized IP features.
#     scale = float((x.pow(2).sum() / denom / max(int(x.size(1)), 1)).detach().cpu().item())
#     return max(ridge, ridge * max(scale, 1.0), 1e-6)


# @torch.no_grad()
# def estimate_ridge_transport(
#     z_old: torch.Tensor,
#     z_new: torch.Tensor,
#     *,
#     ridge: float = 1e-3,
#     identity_blend: float = 0.75,
#     center: bool = True,
#     low_rank: int = 4,
#     max_delta_fro: float = 1.5,
#     max_b_norm: float = 0.75,
#     residual_target: bool = True,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Estimate HSI low-rank residual transport ``T(z)=z@(I+Delta)+b``.

#     The important change from ordinary affine transport is that we regress the
#     **drift** ``z_new - z_old`` and not the entire new feature.  This prevents the
#     estimator from relearning the identity map from few current-phase samples and
#     makes the learned component exactly the phase drift that old descriptors need.
#     """
#     z_old = _finite_2d(z_old, "z_old")
#     z_new = _finite_2d(z_new, "z_new")
#     if z_old.shape != z_new.shape:
#         raise ValueError(f"z_old/z_new shape mismatch: {tuple(z_old.shape)} vs {tuple(z_new.shape)}")

#     device, dtype = z_old.device, z_old.dtype
#     n, d = int(z_old.size(0)), int(z_old.size(1))
#     eye = torch.eye(d, device=device, dtype=dtype)

#     mu_old = z_old.mean(dim=0)
#     mu_new = z_new.mean(dim=0)
#     x = z_old - mu_old if center else z_old

#     if residual_target:
#         target = (z_new - z_old)
#         y = target - target.mean(dim=0) if center else target
#         denom = float(max(n - 1, 1))
#         cov = x.t().matmul(x) / denom
#         cross = x.t().matmul(y) / denom
#         ridge_eff = _adaptive_ridge_value(x, ridge)
#         delta = _stable_solve(cov + ridge_eff * eye, cross)
#         blend = float(max(0.0, min(float(identity_blend), 1.0)))
#         delta = (1.0 - blend) * delta
#         A = eye + delta
#     else:
#         y = z_new - mu_new if center else z_new
#         denom = float(max(n - 1, 1))
#         cov = x.t().matmul(x) / denom
#         cross = x.t().matmul(y) / denom
#         ridge_eff = _adaptive_ridge_value(x, ridge)
#         A_raw = _stable_solve(cov + ridge_eff * eye, cross)
#         blend = float(max(0.0, min(float(identity_blend), 1.0)))
#         A = blend * eye + (1.0 - blend) * A_raw

#     A = project_transport_to_low_rank_residual(A, low_rank=int(low_rank), max_delta_fro=float(max_delta_fro))
#     b = mu_new - mu_old.matmul(A)
#     b = _clip_vector_norm(b, max_norm=float(max_b_norm))
#     return A.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


# @torch.no_grad()
# def estimate_gls_transport(
#     z_old: torch.Tensor,
#     z_new: torch.Tensor,
#     *,
#     ridge: float = 1e-3,
#     identity_blend: float = 0.75,
#     target_cov: str = "diag",
#     low_rank: int = 4,
#     max_delta_fro: float = 1.5,
#     max_b_norm: float = 0.75,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Mahalanobis-weighted residual transport.

#     Use this as an ablation after ridge transport works.  It whitens the drift
#     target by the current covariance, then still returns low-rank ``I+Delta``.
#     """
#     z_old = _finite_2d(z_old, "z_old")
#     z_new = _finite_2d(z_new, "z_new")
#     if z_old.shape != z_new.shape:
#         raise ValueError(f"z_old/z_new shape mismatch: {tuple(z_old.shape)} vs {tuple(z_new.shape)}")

#     n, d = int(z_old.size(0)), int(z_old.size(1))
#     device, dtype = z_old.device, z_old.dtype
#     eye = torch.eye(d, device=device, dtype=dtype)
#     mu_old = z_old.mean(dim=0)
#     mu_new = z_new.mean(dim=0)
#     x = z_old - mu_old
#     drift = z_new - z_old
#     y = drift - drift.mean(dim=0)
#     denom = float(max(n - 1, 1))
#     ridge_eff = _adaptive_ridge_value(x, ridge)

#     cov_old = x.t().matmul(x) / denom + ridge_eff * eye
#     cov_y = y.t().matmul(y) / denom
#     if str(target_cov).lower().strip() == "diag":
#         cov_y = torch.diag(torch.diag(cov_y))
#     cov_y = cov_y + ridge_eff * eye

#     try:
#         evals, evecs = torch.linalg.eigh(cov_y)
#         evals = evals.clamp_min(1e-6)
#         y_inv_sqrt = evecs.matmul(torch.diag(evals.rsqrt())).matmul(evecs.t())
#         y_sqrt = evecs.matmul(torch.diag(evals.sqrt())).matmul(evecs.t())
#         cross = x.t().matmul(y.matmul(y_inv_sqrt)) / denom
#         delta_w = _stable_solve(cov_old, cross)
#         delta = delta_w.matmul(y_sqrt)
#     except RuntimeError:
#         cross = x.t().matmul(y) / denom
#         delta = _stable_solve(cov_old, cross)

#     blend = float(max(0.0, min(float(identity_blend), 1.0)))
#     A = eye + (1.0 - blend) * delta
#     A = project_transport_to_low_rank_residual(A, low_rank=int(low_rank), max_delta_fro=float(max_delta_fro))
#     b = mu_new - mu_old.matmul(A)
#     b = _clip_vector_norm(b, max_norm=float(max_b_norm))
#     return A.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


# @torch.no_grad()
# def transport_diagnostics(z_old: torch.Tensor, z_new: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
#     """Return diagnostics for transport admission and ablation tables."""
#     z_old = _finite_2d(z_old, "z_old")
#     z_new = _finite_2d(z_new, "z_new")
#     if z_old.shape != z_new.shape:
#         raise ValueError(f"z_old/z_new shape mismatch: {tuple(z_old.shape)} vs {tuple(z_new.shape)}")
#     A = torch.nan_to_num(A.to(z_old.device, z_old.dtype), nan=0.0, posinf=0.0, neginf=0.0)
#     b = torch.nan_to_num(b.to(z_old.device, z_old.dtype).flatten(), nan=0.0, posinf=0.0, neginf=0.0)
#     d = int(z_old.size(1))
#     if A.shape != (d, d):
#         raise ValueError(f"A shape {tuple(A.shape)} incompatible with z dim {d}")
#     if b.numel() != d:
#         raise ValueError(f"b length {b.numel()} incompatible with z dim {d}")

#     eye = torch.eye(d, device=z_old.device, dtype=z_old.dtype)
#     pred_before = z_old
#     pred_after = z_old.matmul(A) + b
#     rmse_before = (pred_before - z_new).pow(2).mean().sqrt()
#     rmse_after = (pred_after - z_new).pow(2).mean().sqrt()
#     delta = A - eye
#     try:
#         s = torch.linalg.svdvals(delta)
#         delta_rank_1e3 = int((s > 1e-3).sum().detach().cpu().item())
#     except RuntimeError:
#         delta_rank_1e3 = -1
#     drift = z_new - z_old
#     drift_norm = drift.pow(2).mean().sqrt()
#     moved = pred_after - z_old
#     move_norm = moved.pow(2).mean().sqrt()
#     return {
#         "rmse_before": float(rmse_before.detach().cpu().item()),
#         "rmse_after": float(rmse_after.detach().cpu().item()),
#         "rmse_gain": float((rmse_before - rmse_after).detach().cpu().item()),
#         "rmse_ratio": float((rmse_after / rmse_before.clamp_min(_EPS)).detach().cpu().item()),
#         "A_minus_I_fro": float(delta.norm().detach().cpu().item()),
#         "A_minus_I_maxabs": float(delta.abs().max().detach().cpu().item()),
#         "b_norm": float(b.norm().detach().cpu().item()),
#         "drift_rmse": float(drift_norm.detach().cpu().item()),
#         "transport_move_rmse": float(move_norm.detach().cpu().item()),
#         "delta_rank_1e3": float(delta_rank_1e3),
#         "pairs": float(z_old.size(0)),
#         "dim": float(d),
#     }
