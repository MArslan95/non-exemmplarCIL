from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
Row = Dict[str, Tensor]
_EPS = 1e-12
_FORBIDDEN_MEMORY_NAMES = {
    "raw_samples",
    "raw_patches",
    "raw_spectra",
    "old_samples",
    "old_patches",
    "stored_samples",
    "stored_patches",
    "feature_memory",
    "old_features",
    "stored_features",
    "token_memory",
    "feature_queue",
    "exemplars",
    "exemplar_indices",
    "replay_samples",
    "teacher_logits",
    "teacher_features",
    "generated_images",
    "task_classifiers",
}


@dataclass(frozen=True)
class EnergyOutput:
    """Energy matrix with an explicit class-column contract."""

    energy: Tensor          # [N, C]
    class_ids: Tensor       # [C]
    quadratic: Tensor       # [N, C]
    volume: Tensor          # [N, C]


@dataclass(frozen=True)
class BranchSimilarityTransform:
    """Branchwise similarity transform in PyTorch row-vector convention.

    For row features, the transform is
        z_new = z_old @ A.T + b
    where A = blockdiag(a_s R_s, a_p R_p).
    """

    spectral_rotation: Tensor   # [Ds, Ds], orthogonal
    spatial_rotation: Tensor    # [Dp, Dp], orthogonal
    spectral_scale: float
    spatial_scale: float
    spectral_bias: Tensor       # [Ds]
    spatial_bias: Tensor        # [Dp]
    spectral_level: int = 2
    spatial_level: int = 2


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _unique_ids(values: Iterable[int]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        class_id = int(value)
        if class_id < 0:
            raise ValueError(f"class IDs must be non-negative, got {class_id}")
        if class_id not in seen:
            seen.add(class_id)
            ids.append(class_id)
    return ids


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _effective_sample_count(weights: Tensor) -> Tensor:
    total = weights.sum().clamp_min(_EPS)
    return total.square() / weights.square().sum().clamp_min(_EPS)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.sum().clamp_min(_EPS)
    return (weights[:, None] * values).sum(dim=0) / denominator


def _weighted_covariance(values: Tensor, weights: Tensor, mean: Tensor) -> Tensor:
    centered = values - mean
    total = weights.sum().clamp_min(_EPS)
    denominator = (total - weights.square().sum() / total).clamp_min(1.0)
    weighted = centered * weights.sqrt().unsqueeze(1)
    return _symmetrize(weighted.transpose(0, 1) @ weighted / denominator)


def _safe_cholesky(matrix: Tensor, *, jitter: float = 1e-7, attempts: int = 5) -> Tensor:
    matrix = _symmetrize(matrix)
    eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
    current = float(max(jitter, 0.0))
    for _ in range(max(int(attempts), 1)):
        chol, info = torch.linalg.cholesky_ex(matrix + current * eye)
        if bool((info == 0).all()):
            return chol
        current = 10.0 * max(current, 1e-12)
    raise RuntimeError("Cholesky factorization failed after jitter escalation")


def _weighted_quantile(values: Tensor, quantile: float) -> Tensor:
    # Used only for robust scale diagnostics. Current implementation is unweighted
    # because robust weights are already incorporated in the iterative centers.
    q = float(min(max(quantile, 0.0), 1.0))
    return torch.quantile(values, q)


def _rank_cap(effective_count: float, maximum_rank: int, dimension: int) -> int:
    # N-2 is deliberate: one degree for centering and one safety degree.
    return int(
        max(
            0,
            min(
                int(maximum_rank),
                int(dimension),
                max(int(math.floor(effective_count)) - 2, 0),
            ),
        )
    )


def _branch_noise_initialization(
    covariance: Tensor,
    branch_slice: slice,
    prior: Tensor,
    *,
    variance_floor: float,
    shrinkage: float,
) -> Tensor:
    block = _symmetrize(covariance[branch_slice, branch_slice])
    eigvals = torch.linalg.eigvalsh(block).clamp_min(0.0)
    positive = eigvals[eigvals > float(variance_floor)]
    if positive.numel() >= 2:
        lower = positive[: max(positive.numel() // 2, 1)]
        raw = lower.median()
    else:
        raw = block.diagonal().mean().clamp_min(variance_floor)
    return ((1.0 - shrinkage) * raw + shrinkage * prior).clamp_min(
        variance_floor
    )


def _make_psi_vector(
    spectral_variance: Tensor,
    spatial_variance: Tensor,
    spectral_dim: int,
    spatial_dim: int,
) -> Tensor:
    return torch.cat(
        [
            spectral_variance.reshape(1).expand(spectral_dim),
            spatial_variance.reshape(1).expand(spatial_dim),
        ],
        dim=0,
    )


def covariance_from_row(
    row: Mapping[str, Tensor],
    *,
    spectral_dim: int,
    spatial_dim: int,
) -> Tensor:
    loading = row["loading"]
    active_rank = int(torch.as_tensor(row["active_rank"]).item())
    active_loading = loading[:, :active_rank]
    psi = _make_psi_vector(
        torch.as_tensor(row["residual_var_spectral"], device=loading.device, dtype=loading.dtype),
        torch.as_tensor(row["residual_var_spatial"], device=loading.device, dtype=loading.dtype),
        spectral_dim,
        spatial_dim,
    )
    return _symmetrize(active_loading @ active_loading.transpose(0, 1) + torch.diag(psi))


# -----------------------------------------------------------------------------
# Main bank
# -----------------------------------------------------------------------------


class HSIFactorGeometryBank(nn.Module):
    """Strict exemplar-free HSI class memory.

    Classification memory:
        p(z | c) = N(mu_c, L_c L_c^T + Psi_c)

    where z = [z_s ; z_p] and
        Psi_c = diag(psi_c,s I_Ds, psi_c,p I_Dp).

    Raw ordered spectra are summarized separately as diagonal spectral-shape
    statistics. Those statistics NEVER enter the class energy directly; they are
    used only to compute class-pair boundary risk.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        spectral_dim: int = 32,
        spatial_dim: int = 32,
        maximum_rank: int = 4,
        *,
        raw_spectral_dim: int,
        spectral_resample_length: int = 32,
        spectral_derivative_weight: float = 0.5,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        variance_floor_absolute: float = 1e-4,
        variance_floor_relative: float = 1e-3,
        minimum_variance_shrinkage: float = 0.10,
        extra_rare_class_shrinkage: float = 0.35,
        shrinkage_sample_strength: float = 20.0,
        factor_fit_iterations: int = 3,
        whitened_eigen_excess_threshold: float = 0.50,
        rank_energy_threshold: float = 0.95,
        rank_eigen_ratio_threshold: float = 1e-3,
        volume_weight: float = 0.50,
        robust_iterations: int = 3,
        robust_huber_delta: float = 2.5,
        spatial_purity_power: float = 1.0,
        reliability_sample_strength: float = 20.0,
        invalid_energy: float = 1e6,
    ) -> None:
        super().__init__()
        self.spectral_dim = int(spectral_dim)
        self.spatial_dim = int(spatial_dim)
        self.feature_dim = self.spectral_dim + self.spatial_dim
        self.maximum_rank = int(maximum_rank)
        self.raw_spectral_dim = int(raw_spectral_dim)
        self.spectral_resample_length = int(spectral_resample_length)
        self.spectral_shape_dim = 2 * self.spectral_resample_length - 1

        if self.spectral_dim <= 0 or self.spatial_dim <= 0:
            raise ValueError("spectral_dim and spatial_dim must be positive")
        if not 0 <= self.maximum_rank <= self.feature_dim:
            raise ValueError("maximum_rank must be in [0, feature_dim]")
        if self.raw_spectral_dim <= 1:
            raise ValueError("raw_spectral_dim must be greater than one")
        if self.spectral_resample_length <= 2:
            raise ValueError("spectral_resample_length must be greater than two")

        self.spectral_derivative_weight = float(spectral_derivative_weight)
        self.variance_floor_absolute = float(max(variance_floor_absolute, 1e-12))
        self.variance_floor_relative = float(max(variance_floor_relative, 0.0))
        self.minimum_variance_shrinkage = float(
            min(max(minimum_variance_shrinkage, 0.0), 0.95)
        )
        self.extra_rare_class_shrinkage = float(
            min(max(extra_rare_class_shrinkage, 0.0), 0.95)
        )
        self.shrinkage_sample_strength = float(max(shrinkage_sample_strength, 0.0))
        self.factor_fit_iterations = int(max(factor_fit_iterations, 1))
        self.whitened_eigen_excess_threshold = float(
            max(whitened_eigen_excess_threshold, 0.0)
        )
        self.rank_energy_threshold = float(
            min(max(rank_energy_threshold, 0.50), 0.9999)
        )
        self.rank_eigen_ratio_threshold = float(
            min(max(rank_eigen_ratio_threshold, 0.0), 1.0)
        )
        self.volume_weight = float(max(volume_weight, 0.0))
        self.robust_iterations = int(max(robust_iterations, 1))
        self.robust_huber_delta = float(max(robust_huber_delta, 1e-3))
        self.spatial_purity_power = float(max(spatial_purity_power, 0.0))
        self.reliability_sample_strength = float(max(reliability_sample_strength, 1e-6))
        self.invalid_energy = float(max(invalid_energy, 1.0))

        dev = torch.device(device)

        # Base-fitted global priors. They are aggregate statistics, not samples.
        self.register_buffer(
            "global_residual_prior_spectral",
            torch.tensor(1.0, device=dev, dtype=dtype),
        )
        self.register_buffer(
            "global_residual_prior_spatial",
            torch.tensor(1.0, device=dev, dtype=dtype),
        )
        self.register_buffer(
            "global_spectral_shape_variance",
            torch.ones(self.spectral_shape_dim, device=dev, dtype=dtype),
        )
        self.register_buffer(
            "global_priors_ready",
            torch.tensor(False, device=dev, dtype=torch.bool),
        )
        self.register_buffer(
            "global_priors_frozen",
            torch.tensor(False, device=dev, dtype=torch.bool),
        )

        # Base-fitted pair-distance temperatures.
        self.register_buffer(
            "spectral_overlap_temperature",
            torch.tensor(1.0, device=dev, dtype=dtype),
        )
        self.register_buffer(
            "geometry_overlap_temperature",
            torch.tensor(1.0, device=dev, dtype=dtype),
        )
        self.register_buffer(
            "overlap_temperatures_ready",
            torch.tensor(False, device=dev, dtype=torch.bool),
        )
        self.register_buffer(
            "overlap_temperatures_frozen",
            torch.tensor(False, device=dev, dtype=torch.bool),
        )

        self._register_row_buffers(dev, dtype)

    # ------------------------------------------------------------------
    # Buffer schema
    # ------------------------------------------------------------------

    def _register_row_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        c, d, r, h = 0, self.feature_dim, self.maximum_rank, self.spectral_shape_dim
        self.register_buffer("means", torch.empty((c, d), device=device, dtype=dtype))
        self.register_buffer("loadings", torch.empty((c, d, r), device=device, dtype=dtype))
        self.register_buffer("active_ranks", torch.empty((c,), device=device, dtype=torch.long))
        self.register_buffer(
            "residual_vars_spectral", torch.empty((c,), device=device, dtype=dtype)
        )
        self.register_buffer(
            "residual_vars_spatial", torch.empty((c,), device=device, dtype=dtype)
        )

        self.register_buffer(
            "spectral_shape_means", torch.empty((c, h), device=device, dtype=dtype)
        )
        self.register_buffer(
            "spectral_shape_vars", torch.empty((c, h), device=device, dtype=dtype)
        )
        self.register_buffer(
            "spectral_shape_reliability", torch.empty((c,), device=device, dtype=dtype)
        )

        self.register_buffer("sample_counts", torch.empty((c,), device=device, dtype=dtype))
        self.register_buffer(
            "effective_sample_counts", torch.empty((c,), device=device, dtype=dtype)
        )
        self.register_buffer("geometry_reliability", torch.empty((c,), device=device, dtype=dtype))
        self.register_buffer(
            "reconstruction_errors", torch.empty((c,), device=device, dtype=dtype)
        )
        self.register_buffer("outlier_rates", torch.empty((c,), device=device, dtype=dtype))

        self.register_buffer("energy_quantiles", torch.empty((c, 3), device=device, dtype=dtype))
        self.register_buffer("margin_quantiles", torch.empty((c, 3), device=device, dtype=dtype))
        self.register_buffer("statistics_ready", torch.empty((c,), device=device, dtype=torch.bool))
        self.register_buffer("phase_created", torch.empty((c,), device=device, dtype=torch.long))
        self.register_buffer("phase_updated", torch.empty((c,), device=device, dtype=torch.long))
        self.register_buffer("row_valid", torch.empty((c,), device=device, dtype=torch.bool))

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    def __len__(self) -> int:
        return int(self.row_valid.numel())

    # ------------------------------------------------------------------
    # Canonicalization and spectral shape
    # ------------------------------------------------------------------

    def _canonicalize_features(self, features: Tensor) -> Tensor:
        value = torch.as_tensor(features, device=self.device, dtype=self.dtype)
        if value.dim() != 2 or value.size(1) != self.feature_dim:
            raise ValueError(
                f"features must be [N,{self.feature_dim}], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError("features contain NaN/Inf")
        return value

    def _canonicalize_raw_spectra(self, raw_spectra: Tensor) -> Tensor:
        value = torch.as_tensor(raw_spectra, device=self.device, dtype=self.dtype)
        if value.dim() != 2 or value.size(1) != self.raw_spectral_dim:
            raise ValueError(
                f"raw_spectra must be [N,{self.raw_spectral_dim}], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError("raw_spectra contain NaN/Inf")
        return value

    def spectral_shape_descriptor(self, raw_spectra: Tensor) -> Tensor:
        """SNV-normalized ordered shape plus first differences.

        This descriptor is a relational prior only. It is not used in the
        geometry energy or final prediction rule.
        """
        spectra = self._canonicalize_raw_spectra(raw_spectra)
        resampled = F.interpolate(
            spectra.unsqueeze(1),
            size=self.spectral_resample_length,
            mode="linear",
            align_corners=True,
        ).squeeze(1)
        mean = resampled.mean(dim=1, keepdim=True)
        std = resampled.std(dim=1, keepdim=True, unbiased=False).clamp_min(
            math.sqrt(self.variance_floor_absolute)
        )
        normalized = (resampled - mean) / std
        derivative = normalized[:, 1:] - normalized[:, :-1]
        descriptor = torch.cat(
            [normalized, self.spectral_derivative_weight * derivative], dim=1
        )
        if descriptor.size(1) != self.spectral_shape_dim:
            raise RuntimeError("spectral descriptor dimension contract failed")
        return descriptor

    # ------------------------------------------------------------------
    # Global priors
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_global_priors(
        self,
        features: Tensor,
        raw_spectra: Tensor,
        *,
        freeze: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, float]:
        x = self._canonicalize_features(features)
        raw = self._canonicalize_raw_spectra(raw_spectra)
        if x.size(0) != raw.size(0):
            raise ValueError("features and raw_spectra are not aligned")
        if x.size(0) < 3:
            raise ValueError("at least three base-training samples are required")
        if bool(self.global_priors_ready.item()) and not overwrite:
            raise RuntimeError("global priors already exist")
        if bool(self.global_priors_frozen.item()) and overwrite:
            raise RuntimeError("frozen global priors cannot be overwritten")
        if bool(self.row_valid.any()):
            raise RuntimeError("fit global priors before committing class rows")

        spectral = x[:, : self.spectral_dim]
        spatial = x[:, self.spectral_dim :]
        spectral_var = spectral.var(dim=0, unbiased=True).mean().clamp_min(
            self.variance_floor_absolute
        )
        spatial_var = spatial.var(dim=0, unbiased=True).mean().clamp_min(
            self.variance_floor_absolute
        )
        shape = self.spectral_shape_descriptor(raw)
        shape_var = shape.var(dim=0, unbiased=True).clamp_min(
            self.variance_floor_absolute
        )

        self.global_residual_prior_spectral.copy_(spectral_var)
        self.global_residual_prior_spatial.copy_(spatial_var)
        self.global_spectral_shape_variance.copy_(shape_var)
        self.global_priors_ready.fill_(True)
        self.global_priors_frozen.fill_(bool(freeze))
        return {
            "sample_count": float(x.size(0)),
            "spectral_prior": float(spectral_var.item()),
            "spatial_prior": float(spatial_var.item()),
            "shape_variance_mean": float(shape_var.mean().item()),
            "frozen": float(bool(freeze)),
        }

    def assert_global_priors_ready(self) -> None:
        if not bool(self.global_priors_ready.item()):
            raise RuntimeError(
                "global priors are absent; fit them once from base-training data"
            )
        values = (
            self.global_residual_prior_spectral,
            self.global_residual_prior_spatial,
            self.global_spectral_shape_variance,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError("global priors contain NaN/Inf")
        if float(self.global_residual_prior_spectral.item()) <= 0.0:
            raise RuntimeError("spectral residual prior must be positive")
        if float(self.global_residual_prior_spatial.item()) <= 0.0:
            raise RuntimeError("spatial residual prior must be positive")
        if bool((self.global_spectral_shape_variance <= 0).any()):
            raise RuntimeError("spectral shape variance prior must be positive")

    # ------------------------------------------------------------------
    # Robust weighting and factor fitting
    # ------------------------------------------------------------------

    def _external_weights(
        self, sample_weights: Optional[Tensor], sample_count: int
    ) -> Tensor:
        if sample_weights is None:
            weights = torch.ones(sample_count, device=self.device, dtype=self.dtype)
        else:
            weights = torch.as_tensor(
                sample_weights, device=self.device, dtype=self.dtype
            ).flatten()
            if weights.numel() != sample_count:
                raise ValueError("sample_weights must contain one value per sample")
            if not torch.isfinite(weights).all():
                raise RuntimeError("sample_weights contain NaN/Inf")
            weights = weights.clamp(0.0, 1.0)
            if self.spatial_purity_power != 1.0:
                weights = weights.pow(self.spatial_purity_power)
        if float(weights.sum().item()) <= _EPS:
            raise ValueError("sample_weights sum to zero")
        return weights / weights.mean().clamp_min(_EPS)

    def _robust_weights(
        self,
        features: Tensor,
        descriptors: Tensor,
        sample_weights: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        base = self._external_weights(sample_weights, features.size(0))
        weights = base.clone()
        gate = torch.ones_like(weights)

        for _ in range(self.robust_iterations):
            f_center = _weighted_mean(features, weights)
            h_center = _weighted_mean(descriptors, weights)
            f_dist = (features - f_center).norm(dim=1)
            h_dist = (descriptors - h_center).norm(dim=1)
            f_scale = _weighted_quantile(f_dist[f_dist > 0], 0.5) if bool((f_dist > 0).any()) else features.new_tensor(1.0)
            h_scale = _weighted_quantile(h_dist[h_dist > 0], 0.5) if bool((h_dist > 0).any()) else descriptors.new_tensor(1.0)
            f_scale = f_scale.clamp_min(math.sqrt(self.variance_floor_absolute))
            h_scale = h_scale.clamp_min(math.sqrt(self.variance_floor_absolute))
            joint = torch.sqrt((f_dist / f_scale).square() + (h_dist / h_scale).square())
            gate = torch.clamp(
                joint.new_tensor(self.robust_huber_delta) / joint.clamp_min(_EPS),
                max=1.0,
            )
            weights = base * gate
            weights = weights / weights.mean().clamp_min(_EPS)

        return weights, _effective_sample_count(weights), gate.lt(0.999).float().mean()

    def _variance_floor(self, covariance: Tensor) -> float:
        mean_variance = float(covariance.diagonal().mean().clamp_min(0.0).item())
        return max(
            self.variance_floor_absolute,
            self.variance_floor_relative * mean_variance,
        )

    def _shrinkage_strength(self, effective_count: float) -> float:
        rare = (
            self.shrinkage_sample_strength
            / (effective_count + self.shrinkage_sample_strength)
            if self.shrinkage_sample_strength > 0.0
            else 0.0
        )
        return min(
            self.minimum_variance_shrinkage
            + self.extra_rare_class_shrinkage * rare,
            0.95,
        )

    def _select_rank(self, whitened_eigenvalues: Tensor, rank_cap: int) -> int:
        if rank_cap <= 0 or whitened_eigenvalues.numel() == 0:
            return 0
        values = whitened_eigenvalues[:rank_cap]
        excess = (values - 1.0).clamp_min(0.0)
        if float(excess.sum().item()) <= _EPS:
            return 0

        signal = values > (1.0 + self.whitened_eigen_excess_threshold)
        if float(excess[0].item()) > _EPS:
            ratio = excess / excess[0].clamp_min(_EPS)
            signal = signal & ratio.ge(self.rank_eigen_ratio_threshold)
        signal_count = int(signal.sum().item())
        if signal_count <= 0:
            return 0

        cumulative = torch.cumsum(excess, dim=0) / excess.sum().clamp_min(_EPS)
        hit = torch.nonzero(cumulative >= self.rank_energy_threshold, as_tuple=False)
        energy_rank = int(hit[0].item()) + 1 if hit.numel() else rank_cap
        return int(min(rank_cap, signal_count, energy_rank))

    def _fit_factor_geometry(self, values: Tensor, weights: Tensor) -> Dict[str, Tensor]:
        self.assert_global_priors_ready()
        mean = _weighted_mean(values, weights)
        covariance = _weighted_covariance(values, weights, mean)
        effective = float(_effective_sample_count(weights).item())
        floor = self._variance_floor(covariance)
        shrinkage = self._shrinkage_strength(effective)

        s_slice = slice(0, self.spectral_dim)
        p_slice = slice(self.spectral_dim, self.feature_dim)
        psi_s = _branch_noise_initialization(
            covariance,
            s_slice,
            self.global_residual_prior_spectral,
            variance_floor=floor,
            shrinkage=shrinkage,
        )
        psi_p = _branch_noise_initialization(
            covariance,
            p_slice,
            self.global_residual_prior_spatial,
            variance_floor=floor,
            shrinkage=shrinkage,
        )

        loading = values.new_zeros((self.feature_dim, self.maximum_rank))
        active_rank = 0
        rank_cap = _rank_cap(effective, self.maximum_rank, self.feature_dim)

        for _ in range(self.factor_fit_iterations):
            previous_s, previous_p = psi_s.clone(), psi_p.clone()
            psi = _make_psi_vector(
                psi_s, psi_p, self.spectral_dim, self.spatial_dim
            )
            inv_sqrt = psi.rsqrt()
            whitened = _symmetrize(
                inv_sqrt[:, None] * covariance * inv_sqrt[None, :]
            )
            eigvals, eigvecs = torch.linalg.eigh(whitened)
            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals.index_select(0, order)
            eigvecs = eigvecs.index_select(1, order)
            active_rank = self._select_rank(eigvals, rank_cap)

            loading.zero_()
            if active_rank > 0:
                excess = (eigvals[:active_rank] - 1.0).clamp_min(0.0).sqrt()
                loading[:, :active_rank] = (
                    psi.sqrt().unsqueeze(1)
                    * eigvecs[:, :active_rank]
                    * excess.unsqueeze(0)
                )

            active = loading[:, :active_rank]
            residual = _symmetrize(covariance - active @ active.transpose(0, 1))
            raw_s = residual[s_slice, s_slice].diagonal().mean().clamp_min(floor)
            raw_p = residual[p_slice, p_slice].diagonal().mean().clamp_min(floor)
            psi_s = (
                (1.0 - shrinkage) * raw_s
                + shrinkage * self.global_residual_prior_spectral
            ).clamp_min(floor)
            psi_p = (
                (1.0 - shrinkage) * raw_p
                + shrinkage * self.global_residual_prior_spatial
            ).clamp_min(floor)

            delta_s = (psi_s - previous_s).abs() / previous_s.clamp_min(floor)
            delta_p = (psi_p - previous_p).abs() / previous_p.clamp_min(floor)
            if float(torch.maximum(delta_s, delta_p).item()) < 1e-3:
                break

        if active_rank < self.maximum_rank:
            loading[:, active_rank:] = 0.0
        psi = _make_psi_vector(psi_s, psi_p, self.spectral_dim, self.spatial_dim)
        reconstructed = _symmetrize(
            loading[:, :active_rank] @ loading[:, :active_rank].transpose(0, 1)
            + torch.diag(psi)
        )
        reconstruction_error = (
            (reconstructed - covariance).norm() / covariance.norm().clamp_min(_EPS)
        )
        sample_reliability = values.new_tensor(
            effective / (effective + self.reliability_sample_strength)
        )
        geometry_reliability = (
            sample_reliability * (1.0 - reconstruction_error.clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)

        return {
            "mean": mean,
            "loading": loading,
            "active_rank": torch.tensor(active_rank, device=self.device, dtype=torch.long),
            "residual_var_spectral": psi_s,
            "residual_var_spatial": psi_p,
            "effective_sample_count": values.new_tensor(effective),
            "geometry_reliability": geometry_reliability,
            "reconstruction_error": reconstruction_error,
        }

    def _fit_spectral_shape(self, descriptors: Tensor, weights: Tensor) -> Dict[str, Tensor]:
        self.assert_global_priors_ready()
        mean = _weighted_mean(descriptors, weights)
        centered = descriptors - mean
        total = weights.sum().clamp_min(_EPS)
        denominator = (total - weights.square().sum() / total).clamp_min(1.0)
        variance = (weights[:, None] * centered.square()).sum(dim=0) / denominator
        effective = float(_effective_sample_count(weights).item())
        shrinkage = self._shrinkage_strength(effective)
        floor = max(
            self.variance_floor_absolute,
            self.variance_floor_relative * float(variance.mean().item()),
        )
        variance = (
            (1.0 - shrinkage) * variance
            + shrinkage * self.global_spectral_shape_variance
        ).clamp_min(floor)
        reliability = descriptors.new_tensor(
            effective / (effective + self.reliability_sample_strength)
        ).clamp(0.0, 1.0)
        return {
            "spectral_shape_mean": mean,
            "spectral_shape_var": variance,
            "spectral_shape_reliability": reliability,
        }

    @torch.no_grad()
    def extract_rows(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        raw_spectra: Tensor,
        sample_weights: Optional[Tensor] = None,
        class_ids: Optional[Iterable[int]] = None,
    ) -> Dict[int, Row]:
        x = self._canonicalize_features(features)
        raw = self._canonicalize_raw_spectra(raw_spectra)
        y = torch.as_tensor(labels, device=self.device, dtype=torch.long).flatten()
        if x.size(0) != raw.size(0) or y.numel() != x.size(0):
            raise ValueError("features, raw_spectra, and labels are not aligned")
        if bool((y < 0).any()):
            raise ValueError("negative class labels are forbidden")
        descriptors = self.spectral_shape_descriptor(raw)

        weights = None
        if sample_weights is not None:
            weights = torch.as_tensor(
                sample_weights, device=self.device, dtype=self.dtype
            ).flatten()
            if weights.numel() != x.size(0):
                raise ValueError("sample_weights length mismatch")

        allowed = None if class_ids is None else set(_unique_ids(class_ids))
        rows: Dict[int, Row] = {}
        for class_tensor in torch.unique(y, sorted=True):
            class_id = int(class_tensor.item())
            if allowed is not None and class_id not in allowed:
                continue
            mask = y.eq(class_tensor)
            class_x = x[mask]
            class_h = descriptors[mask]
            class_w = None if weights is None else weights[mask]
            robust_weights, effective, outlier_rate = self._robust_weights(
                class_x, class_h, class_w
            )
            geometry = self._fit_factor_geometry(class_x, robust_weights)
            shape = self._fit_spectral_shape(class_h, robust_weights)
            rows[class_id] = {
                "mean": geometry["mean"].detach(),
                "loading": geometry["loading"].detach(),
                "active_rank": geometry["active_rank"].detach(),
                "residual_var_spectral": geometry["residual_var_spectral"].detach(),
                "residual_var_spatial": geometry["residual_var_spatial"].detach(),
                "spectral_shape_mean": shape["spectral_shape_mean"].detach(),
                "spectral_shape_var": shape["spectral_shape_var"].detach(),
                "spectral_shape_reliability": shape[
                    "spectral_shape_reliability"
                ].detach(),
                "sample_count": class_x.new_tensor(float(class_x.size(0))),
                "effective_sample_count": effective.detach(),
                "geometry_reliability": geometry["geometry_reliability"].detach(),
                "reconstruction_error": geometry["reconstruction_error"].detach(),
                "outlier_rate": outlier_rate.detach(),
            }

        if allowed is not None and set(rows) != allowed:
            missing = sorted(allowed - set(rows))
            raise RuntimeError(f"no samples were provided for classes {missing}")
        return rows

    extract_geometry = extract_rows

    # ------------------------------------------------------------------
    # Row validation and storage
    # ------------------------------------------------------------------

    def _append_empty_row(self) -> None:
        def append(name: str, value: Tensor) -> None:
            setattr(self, name, torch.cat([getattr(self, name), value], dim=0))

        append("means", torch.zeros((1, self.feature_dim), device=self.device, dtype=self.dtype))
        append(
            "loadings",
            torch.zeros(
                (1, self.feature_dim, self.maximum_rank),
                device=self.device,
                dtype=self.dtype,
            ),
        )
        append("active_ranks", torch.zeros((1,), device=self.device, dtype=torch.long))
        append(
            "residual_vars_spectral",
            torch.full(
                (1,), self.variance_floor_absolute, device=self.device, dtype=self.dtype
            ),
        )
        append(
            "residual_vars_spatial",
            torch.full(
                (1,), self.variance_floor_absolute, device=self.device, dtype=self.dtype
            ),
        )
        append(
            "spectral_shape_means",
            torch.zeros((1, self.spectral_shape_dim), device=self.device, dtype=self.dtype),
        )
        append(
            "spectral_shape_vars",
            torch.full(
                (1, self.spectral_shape_dim),
                self.variance_floor_absolute,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        append(
            "spectral_shape_reliability",
            torch.zeros((1,), device=self.device, dtype=self.dtype),
        )
        for name in (
            "sample_counts",
            "effective_sample_counts",
            "geometry_reliability",
            "reconstruction_errors",
            "outlier_rates",
        ):
            append(name, torch.zeros((1,), device=self.device, dtype=self.dtype))
        append("energy_quantiles", torch.zeros((1, 3), device=self.device, dtype=self.dtype))
        append("margin_quantiles", torch.zeros((1, 3), device=self.device, dtype=self.dtype))
        append("statistics_ready", torch.zeros((1,), device=self.device, dtype=torch.bool))
        append("phase_created", torch.full((1,), -1, device=self.device, dtype=torch.long))
        append("phase_updated", torch.full((1,), -1, device=self.device, dtype=torch.long))
        append("row_valid", torch.zeros((1,), device=self.device, dtype=torch.bool))

    def ensure_class_count(self, count: int) -> None:
        while len(self) < int(count):
            self._append_empty_row()

    def _normalize_row(self, row: Mapping[str, Any]) -> Row:
        required = (
            "mean",
            "loading",
            "active_rank",
            "residual_var_spectral",
            "residual_var_spatial",
            "spectral_shape_mean",
            "spectral_shape_var",
            "spectral_shape_reliability",
            "sample_count",
            "effective_sample_count",
        )
        missing = [name for name in required if row.get(name) is None]
        if missing:
            raise RuntimeError(f"row is missing required fields {missing}")

        def value(name: str) -> Tensor:
            return torch.as_tensor(row[name], device=self.device, dtype=self.dtype)

        mean = value("mean").flatten()
        loading = value("loading")
        active_rank = int(torch.as_tensor(row["active_rank"]).item())
        psi_s = value("residual_var_spectral").reshape(())
        psi_p = value("residual_var_spatial").reshape(())
        shape_mean = value("spectral_shape_mean").flatten()
        shape_var = value("spectral_shape_var").flatten()
        shape_reliability = value("spectral_shape_reliability").reshape(())
        sample_count = float(torch.as_tensor(row["sample_count"]).item())
        effective_count = float(torch.as_tensor(row["effective_sample_count"]).item())

        expected = {
            "mean": (mean, (self.feature_dim,)),
            "loading": (loading, (self.feature_dim, self.maximum_rank)),
            "spectral_shape_mean": (shape_mean, (self.spectral_shape_dim,)),
            "spectral_shape_var": (shape_var, (self.spectral_shape_dim,)),
        }
        for name, (tensor, shape) in expected.items():
            if tuple(tensor.shape) != shape:
                raise RuntimeError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if not 0 <= active_rank <= self.maximum_rank:
            raise RuntimeError("invalid active_rank")
        if sample_count <= 0.0 or effective_count <= 0.0:
            raise RuntimeError("sample counts must be positive")
        if effective_count > sample_count + 1e-3:
            raise RuntimeError("effective_sample_count cannot exceed sample_count")

        loading = loading.clone()
        if active_rank < self.maximum_rank:
            loading[:, active_rank:] = 0.0
        psi_s = psi_s.clamp_min(self.variance_floor_absolute)
        psi_p = psi_p.clamp_min(self.variance_floor_absolute)
        shape_var = shape_var.clamp_min(self.variance_floor_absolute)

        tensors = (mean, loading, psi_s, psi_p, shape_mean, shape_var, shape_reliability)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise RuntimeError("row contains NaN/Inf")

        def scalar(name: str, default: float) -> Tensor:
            return torch.as_tensor(
                row.get(name, default), device=self.device, dtype=self.dtype
            ).reshape(())

        return {
            "mean": mean,
            "loading": loading,
            "active_rank": torch.tensor(active_rank, device=self.device, dtype=torch.long),
            "residual_var_spectral": psi_s,
            "residual_var_spatial": psi_p,
            "spectral_shape_mean": shape_mean,
            "spectral_shape_var": shape_var,
            "spectral_shape_reliability": shape_reliability.clamp(0.0, 1.0),
            "sample_count": torch.tensor(sample_count, device=self.device, dtype=self.dtype),
            "effective_sample_count": torch.tensor(
                effective_count, device=self.device, dtype=self.dtype
            ),
            "geometry_reliability": scalar("geometry_reliability", 0.0).clamp(0.0, 1.0),
            "reconstruction_error": scalar("reconstruction_error", 1.0).clamp_min(0.0),
            "outlier_rate": scalar("outlier_rate", 0.0).clamp(0.0, 1.0),
        }

    def _write_row(
        self,
        class_id: int,
        row: Mapping[str, Any],
        *,
        phase_created: int,
        phase_updated: int,
    ) -> None:
        normalized = self._normalize_row(row)
        self.ensure_class_count(class_id + 1)
        assignments = {
            "means": "mean",
            "loadings": "loading",
            "active_ranks": "active_rank",
            "residual_vars_spectral": "residual_var_spectral",
            "residual_vars_spatial": "residual_var_spatial",
            "spectral_shape_means": "spectral_shape_mean",
            "spectral_shape_vars": "spectral_shape_var",
            "spectral_shape_reliability": "spectral_shape_reliability",
            "sample_counts": "sample_count",
            "effective_sample_counts": "effective_sample_count",
            "geometry_reliability": "geometry_reliability",
            "reconstruction_errors": "reconstruction_error",
            "outlier_rates": "outlier_rate",
        }
        for destination, source in assignments.items():
            getattr(self, destination)[class_id] = normalized[source]
        self.energy_quantiles[class_id].zero_()
        self.margin_quantiles[class_id].zero_()
        self.statistics_ready[class_id] = False
        self.phase_created[class_id] = int(phase_created)
        self.phase_updated[class_id] = int(phase_updated)
        self.row_valid[class_id] = True

    def valid_mask(self) -> Tensor:
        if len(self) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)
        if self.maximum_rank == 0:
            inactive_zero = torch.ones(len(self), device=self.device, dtype=torch.bool)
        else:
            inactive_mask = (
                torch.arange(self.maximum_rank, device=self.device)
                .view(1, 1, -1)
                .ge(self.active_ranks.view(-1, 1, 1))
            )
            inactive_zero = torch.where(
                inactive_mask,
                self.loadings.abs(),
                torch.zeros_like(self.loadings),
            ).amax(dim=(1, 2)).le(1e-7)
        finite = (
            torch.isfinite(self.means).all(dim=1)
            & torch.isfinite(self.loadings).flatten(1).all(dim=1)
            & torch.isfinite(self.residual_vars_spectral)
            & torch.isfinite(self.residual_vars_spatial)
            & torch.isfinite(self.spectral_shape_means).all(dim=1)
            & torch.isfinite(self.spectral_shape_vars).all(dim=1)
            & torch.isfinite(self.sample_counts)
            & torch.isfinite(self.effective_sample_counts)
        )
        floors = (
            self.residual_vars_spectral.ge(self.variance_floor_absolute)
            & self.residual_vars_spatial.ge(self.variance_floor_absolute)
            & self.spectral_shape_vars.ge(self.variance_floor_absolute).all(dim=1)
        )
        ranks = self.active_ranks.ge(0) & self.active_ranks.le(self.maximum_rank)
        counts = (
            self.sample_counts.gt(0)
            & self.effective_sample_counts.gt(0)
            & self.effective_sample_counts.le(self.sample_counts + 1e-3)
        )
        return self.row_valid & finite & floors & ranks & counts & inactive_zero

    def assert_valid(
        self,
        class_ids: Optional[Iterable[int]] = None,
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        names = set(self.__dict__) | set(self._buffers) | set(self._parameters)
        forbidden = sorted(name for name in names if name.lower() in _FORBIDDEN_MEMORY_NAMES)
        errors: list[str] = []
        if forbidden:
            errors.append(f"forbidden exemplar fields: {forbidden}")
        ids = list(range(len(self))) if class_ids is None else _unique_ids(class_ids)
        valid = self.valid_mask()
        for class_id in ids:
            if class_id >= len(self) or not bool(valid[class_id].item()):
                errors.append(f"class {class_id}: invalid or absent row")
                continue
            row = self.get_class_row(class_id, clone=False)
            covariance = covariance_from_row(
                row, spectral_dim=self.spectral_dim, spatial_dim=self.spatial_dim
            )
            try:
                _safe_cholesky(covariance)
            except RuntimeError:
                errors.append(f"class {class_id}: covariance is not positive definite")

        report = {
            "ok": not errors,
            "class_ids": ids,
            "valid_rows": int(valid.sum().item()),
            "errors": errors,
        }
        if strict and errors:
            raise RuntimeError("HSIFactorGeometryBank invalid: " + "; ".join(errors))
        return report

    def _row_from_buffers(self, class_id: int) -> Row:
        if class_id >= len(self) or not bool(self.valid_mask()[class_id].item()):
            raise RuntimeError(f"class {class_id} has no valid row")
        return {
            "mean": self.means[class_id],
            "loading": self.loadings[class_id],
            "active_rank": self.active_ranks[class_id],
            "residual_var_spectral": self.residual_vars_spectral[class_id],
            "residual_var_spatial": self.residual_vars_spatial[class_id],
            "spectral_shape_mean": self.spectral_shape_means[class_id],
            "spectral_shape_var": self.spectral_shape_vars[class_id],
            "spectral_shape_reliability": self.spectral_shape_reliability[class_id],
            "sample_count": self.sample_counts[class_id],
            "effective_sample_count": self.effective_sample_counts[class_id],
            "geometry_reliability": self.geometry_reliability[class_id],
            "reconstruction_error": self.reconstruction_errors[class_id],
            "outlier_rate": self.outlier_rates[class_id],
        }

    def get_class_row(self, class_id: int, *, clone: bool = True) -> Row:
        row = self._row_from_buffers(int(class_id))
        if clone:
            return {name: value.detach().clone() for name, value in row.items()}
        return row

    def get_bank(
        self, class_ids: Optional[Iterable[int]] = None
    ) -> Dict[str, Tensor]:
        ids = (
            torch.nonzero(self.valid_mask(), as_tuple=False).flatten().tolist()
            if class_ids is None
            else _unique_ids(class_ids)
        )
        self.assert_valid(ids, strict=True)
        index = torch.tensor(ids, device=self.device, dtype=torch.long)
        return {
            "class_ids": index,
            "means": self.means.index_select(0, index),
            "loadings": self.loadings.index_select(0, index),
            "active_ranks": self.active_ranks.index_select(0, index),
            "residual_vars_spectral": self.residual_vars_spectral.index_select(0, index),
            "residual_vars_spatial": self.residual_vars_spatial.index_select(0, index),
            "spectral_shape_means": self.spectral_shape_means.index_select(0, index),
            "spectral_shape_vars": self.spectral_shape_vars.index_select(0, index),
            "spectral_shape_reliability": self.spectral_shape_reliability.index_select(0, index),
            "sample_counts": self.sample_counts.index_select(0, index),
            "effective_sample_counts": self.effective_sample_counts.index_select(0, index),
            "geometry_reliability": self.geometry_reliability.index_select(0, index),
            "reconstruction_errors": self.reconstruction_errors.index_select(0, index),
            "outlier_rates": self.outlier_rates.index_select(0, index),
            "energy_quantiles": self.energy_quantiles.index_select(0, index),
            "margin_quantiles": self.margin_quantiles.index_select(0, index),
            "statistics_ready": self.statistics_ready.index_select(0, index),
            "phase_created": self.phase_created.index_select(0, index),
            "phase_updated": self.phase_updated.index_select(0, index),
        }

    def _stack_rows(
        self,
        class_ids: Sequence[int],
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        ids = _unique_ids(class_ids)
        if not ids:
            raise ValueError("class_ids cannot be empty")
        normalized: list[Row] = []
        for class_id in ids:
            if rows is not None and class_id in rows:
                normalized.append(self._normalize_row(rows[class_id]))
            else:
                normalized.append(self.get_class_row(class_id, clone=False))
        fields = (
            "mean",
            "loading",
            "active_rank",
            "residual_var_spectral",
            "residual_var_spatial",
            "spectral_shape_mean",
            "spectral_shape_var",
            "spectral_shape_reliability",
            "geometry_reliability",
        )
        output = {
            field: torch.stack([row[field] for row in normalized]) for field in fields
        }
        output["class_ids"] = torch.tensor(ids, device=self.device, dtype=torch.long)
        return output

    # ------------------------------------------------------------------
    # Exact factor energy
    # ------------------------------------------------------------------

    def energy_matrix(
        self,
        features: Tensor,
        class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> EnergyOutput:
        x = self._canonicalize_features(features)
        stacked = self._stack_rows(class_ids, rows)
        means = stacked["mean"]
        loadings = stacked["loading"]
        active_ranks = stacked["active_rank"].long()
        psi_s = stacked["residual_var_spectral"].clamp_min(
            self.variance_floor_absolute
        )
        psi_p = stacked["residual_var_spatial"].clamp_min(
            self.variance_floor_absolute
        )
        class_count = means.size(0)
        maximum_rank = loadings.size(2)

        rank_index = torch.arange(maximum_rank, device=self.device).view(1, 1, -1)
        active_mask = rank_index < active_ranks.view(class_count, 1, 1)
        active_loadings = loadings * active_mask.to(self.dtype)

        inv_psi = torch.cat(
            [
                psi_s[:, None].expand(-1, self.spectral_dim),
                psi_p[:, None].expand(-1, self.spatial_dim),
            ],
            dim=1,
        ).reciprocal()
        delta = x[:, None, :] - means[None, :, :]
        base = (delta.square() * inv_psi[None, :, :]).sum(dim=2)

        weighted_loading = active_loadings * inv_psi[:, :, None]
        if maximum_rank == 0:
            quadratic = base
            logdet_m = torch.zeros(class_count, device=self.device, dtype=self.dtype)
        else:
            gram = torch.einsum("cdr,cdq->crq", active_loadings, weighted_loading)
            identity = torch.eye(maximum_rank, device=self.device, dtype=self.dtype)
            m = gram + identity.unsqueeze(0)
            chol = _safe_cholesky(m)

            q = torch.einsum("ncd,cdr->ncr", delta * inv_psi[None, :, :], active_loadings)
            solution = torch.cholesky_solve(q.unsqueeze(-1), chol.unsqueeze(0)).squeeze(-1)
            correction = (q * solution).sum(dim=2)
            quadratic = (base - correction).clamp_min(0.0)
            logdet_m = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)
        logdet = (
            self.spectral_dim * psi_s.log()
            + self.spatial_dim * psi_p.log()
            + logdet_m
        )
        volume = logdet.view(1, class_count).expand(x.size(0), class_count)
        energy = (quadratic + self.volume_weight * volume) / float(self.feature_dim)
        return EnergyOutput(
            energy=energy,
            class_ids=stacked["class_ids"],
            quadratic=quadratic / float(self.feature_dim),
            volume=volume / float(self.feature_dim),
        )

    joint_energy_matrix = energy_matrix

    def predict(
        self,
        features: Tensor,
        class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Tensor:
        output = self.energy_matrix(features, class_ids, rows=rows)
        return output.class_ids.index_select(0, output.energy.argmin(dim=1))

    # ------------------------------------------------------------------
    # Pair distances and risk
    # ------------------------------------------------------------------

    def _factor_logdet_and_solve(self, row: Mapping[str, Tensor], vector: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self._normalize_row(row)
        loading = normalized["loading"][:, : int(normalized["active_rank"].item())]
        psi = _make_psi_vector(
            normalized["residual_var_spectral"],
            normalized["residual_var_spatial"],
            self.spectral_dim,
            self.spatial_dim,
        )
        inv = psi.reciprocal()
        if loading.size(1) == 0:
            quadratic = (vector.square() * inv).sum()
            logdet = psi.log().sum()
            return quadratic, logdet
        m = torch.eye(loading.size(1), device=self.device, dtype=self.dtype) + (
            loading.transpose(0, 1) @ (inv[:, None] * loading)
        )
        chol = _safe_cholesky(m)
        q = loading.transpose(0, 1) @ (inv * vector)
        correction = q @ torch.cholesky_solve(q[:, None], chol).squeeze(1)
        quadratic = (vector.square() * inv).sum() - correction
        logdet = psi.log().sum() + 2.0 * torch.log(torch.diagonal(chol)).sum()
        return quadratic, logdet

    def spectral_shape_distance_matrix(
        self,
        class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Tensor:
        stacked = self._stack_rows(class_ids, rows)
        means = stacked["spectral_shape_mean"]
        variances = stacked["spectral_shape_var"].clamp_min(
            self.variance_floor_absolute
        )
        diff = means[:, None, :] - means[None, :, :]
        avg = 0.5 * (variances[:, None, :] + variances[None, :, :])
        first = 0.125 * (diff.square() / avg).sum(dim=2)
        second = 0.5 * torch.log(
            avg / torch.sqrt(variances[:, None, :] * variances[None, :, :])
        ).sum(dim=2)
        distance = (first + second) / float(self.spectral_shape_dim)
        distance.fill_diagonal_(0.0)
        return distance

    def factor_bhattacharyya_distance_matrix(
        self,
        class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Tensor:
        ids = _unique_ids(class_ids)
        normalized_rows = [
            self._normalize_row(rows[class_id])
            if rows is not None and class_id in rows
            else self.get_class_row(class_id, clone=False)
            for class_id in ids
        ]
        count = len(ids)
        distance = torch.zeros((count, count), device=self.device, dtype=self.dtype)

        # Precompute individual log determinants.
        logdets: list[Tensor] = []
        for row in normalized_rows:
            zero = torch.zeros(self.feature_dim, device=self.device, dtype=self.dtype)
            _, logdet = self._factor_logdet_and_solve(row, zero)
            logdets.append(logdet)

        for i in range(count):
            for j in range(i + 1, count):
                row_i, row_j = normalized_rows[i], normalized_rows[j]
                rank_i = int(row_i["active_rank"].item())
                rank_j = int(row_j["active_rank"].item())
                average_loading = torch.cat(
                    [
                        row_i["loading"][:, :rank_i] / math.sqrt(2.0),
                        row_j["loading"][:, :rank_j] / math.sqrt(2.0),
                    ],
                    dim=1,
                )
                average_row: Row = {
                    "mean": 0.5 * (row_i["mean"] + row_j["mean"]),
                    "loading": torch.zeros(
                        (self.feature_dim, self.maximum_rank * 2),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "active_rank": torch.tensor(
                        rank_i + rank_j, device=self.device, dtype=torch.long
                    ),
                    "residual_var_spectral": 0.5
                    * (
                        row_i["residual_var_spectral"]
                        + row_j["residual_var_spectral"]
                    ),
                    "residual_var_spatial": 0.5
                    * (
                        row_i["residual_var_spatial"]
                        + row_j["residual_var_spatial"]
                    ),
                }
                # Local helper because average rank can be 2Rmax.
                psi = _make_psi_vector(
                    average_row["residual_var_spectral"],
                    average_row["residual_var_spatial"],
                    self.spectral_dim,
                    self.spatial_dim,
                )
                inv = psi.reciprocal()
                mean_delta = row_i["mean"] - row_j["mean"]
                if average_loading.size(1) == 0:
                    quadratic = (mean_delta.square() * inv).sum()
                    logdet_avg = psi.log().sum()
                else:
                    m = torch.eye(
                        average_loading.size(1), device=self.device, dtype=self.dtype
                    ) + average_loading.transpose(0, 1) @ (
                        inv[:, None] * average_loading
                    )
                    chol = _safe_cholesky(m)
                    q = average_loading.transpose(0, 1) @ (inv * mean_delta)
                    correction = q @ torch.cholesky_solve(q[:, None], chol).squeeze(1)
                    quadratic = (mean_delta.square() * inv).sum() - correction
                    logdet_avg = psi.log().sum() + 2.0 * torch.log(
                        torch.diagonal(chol)
                    ).sum()
                value = 0.125 * quadratic + 0.5 * (
                    logdet_avg - 0.5 * (logdets[i] + logdets[j])
                )
                value = value.clamp_min(0.0) / float(self.feature_dim)
                distance[i, j] = value
                distance[j, i] = value
        return distance

    @torch.no_grad()
    def fit_overlap_temperatures(
        self,
        base_class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
        freeze: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, float]:
        """Fit global pair-risk scales from committed or provisional rows.

        Passing ``rows`` is the base-training path: detached cross-fitted or
        full-training provisional rows define the two aggregate temperatures
        before any persistent class row is committed. No sample-level state is
        retained.
        """
        if bool(self.overlap_temperatures_ready.item()) and not overwrite:
            raise RuntimeError("overlap temperatures already exist")
        if bool(self.overlap_temperatures_frozen.item()) and overwrite:
            raise RuntimeError("frozen overlap temperatures cannot be overwritten")
        ids = _unique_ids(base_class_ids)
        if len(ids) < 2:
            raise ValueError("at least two base classes are required")
        if rows is not None and set(int(key) for key in rows) != set(ids):
            raise RuntimeError(
                "provisional rows must exactly cover base_class_ids"
            )
        shape = self.spectral_shape_distance_matrix(ids, rows=rows)
        geometry = self.factor_bhattacharyya_distance_matrix(ids, rows=rows)
        upper = torch.triu_indices(len(ids), len(ids), offset=1, device=self.device)
        shape_values = shape[upper[0], upper[1]]
        geometry_values = geometry[upper[0], upper[1]]
        shape_nonzero = shape_values[shape_values > _EPS]
        geometry_nonzero = geometry_values[geometry_values > _EPS]
        tau_h = (
            shape_nonzero.median()
            if shape_nonzero.numel()
            else shape_values.new_tensor(1.0)
        ).clamp_min(1e-6)
        tau_z = (
            geometry_nonzero.median()
            if geometry_nonzero.numel()
            else geometry_values.new_tensor(1.0)
        ).clamp_min(1e-6)
        self.spectral_overlap_temperature.copy_(tau_h)
        self.geometry_overlap_temperature.copy_(tau_z)
        self.overlap_temperatures_ready.fill_(True)
        self.overlap_temperatures_frozen.fill_(bool(freeze))
        return {
            "spectral_temperature": float(tau_h.item()),
            "geometry_temperature": float(tau_z.item()),
            "frozen": float(bool(freeze)),
            "source": (
                "provisional_rows"
                if rows is not None
                else "committed_rows"
            ),
        }

    def pair_risk_matrix(
        self,
        class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Tensor:
        if not bool(self.overlap_temperatures_ready.item()):
            raise RuntimeError("overlap temperatures have not been fitted")
        stacked = self._stack_rows(class_ids, rows)
        d_h = self.spectral_shape_distance_matrix(class_ids, rows=rows)
        d_z = self.factor_bhattacharyya_distance_matrix(class_ids, rows=rows)
        q_h = stacked["spectral_shape_reliability"].clamp(0.0, 1.0)
        q_z = stacked["geometry_reliability"].clamp(0.0, 1.0)
        reliability = torch.sqrt(
            q_h[:, None] * q_h[None, :] * q_z[:, None] * q_z[None, :]
        )
        risk = torch.exp(
            -0.5
            * (
                d_h / self.spectral_overlap_temperature.clamp_min(1e-6)
                + d_z / self.geometry_overlap_temperature.clamp_min(1e-6)
            )
        ) * reliability
        risk.fill_diagonal_(0.0)
        return risk.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Sampling diagnostics only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_geometry(
        self,
        class_ids: Sequence[int],
        samples_per_class: int,
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Tensor]:
        ids = _unique_ids(class_ids)
        count = int(samples_per_class)
        if count <= 0:
            raise ValueError("samples_per_class must be positive")
        features: list[Tensor] = []
        labels: list[Tensor] = []
        for class_id in ids:
            row = (
                self._normalize_row(rows[class_id])
                if rows is not None and class_id in rows
                else self.get_class_row(class_id, clone=False)
            )
            rank = int(row["active_rank"].item())
            factors = torch.randn(
                (count, rank),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            ) if rank > 0 else torch.empty((count, 0), device=self.device, dtype=self.dtype)
            residual = torch.randn(
                (count, self.feature_dim),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            residual[:, : self.spectral_dim] *= row["residual_var_spectral"].sqrt()
            residual[:, self.spectral_dim :] *= row["residual_var_spatial"].sqrt()
            sample = row["mean"].view(1, -1) + residual
            if rank > 0:
                sample = sample + factors @ row["loading"][:, :rank].transpose(0, 1)
            features.append(sample)
            labels.append(
                torch.full((count,), class_id, device=self.device, dtype=torch.long)
            )
        return {"features": torch.cat(features, dim=0), "labels": torch.cat(labels, dim=0)}

    # Compatibility alias. These samples are diagnostics, not a training contract.
    sample_replay = sample_geometry

    # ------------------------------------------------------------------
    # Exact branchwise transport
    # ------------------------------------------------------------------

    def _validate_transform(
        self, transform: BranchSimilarityTransform
    ) -> BranchSimilarityTransform:
        rs = torch.as_tensor(
            transform.spectral_rotation, device=self.device, dtype=self.dtype
        )
        rp = torch.as_tensor(
            transform.spatial_rotation, device=self.device, dtype=self.dtype
        )
        bs = torch.as_tensor(
            transform.spectral_bias, device=self.device, dtype=self.dtype
        ).flatten()
        bp = torch.as_tensor(
            transform.spatial_bias, device=self.device, dtype=self.dtype
        ).flatten()
        if rs.shape != (self.spectral_dim, self.spectral_dim):
            raise ValueError("spectral_rotation has the wrong shape")
        if rp.shape != (self.spatial_dim, self.spatial_dim):
            raise ValueError("spatial_rotation has the wrong shape")
        if bs.shape != (self.spectral_dim,) or bp.shape != (self.spatial_dim,):
            raise ValueError("branch bias has the wrong shape")
        if transform.spectral_scale <= 0.0 or transform.spatial_scale <= 0.0:
            raise ValueError("branch scales must be positive")
        for name, rotation in (("spectral", rs), ("spatial", rp)):
            identity = torch.eye(rotation.size(0), device=self.device, dtype=self.dtype)
            error = (rotation.transpose(0, 1) @ rotation - identity).norm()
            if float(error.item()) > 1e-3:
                raise RuntimeError(f"{name} rotation is not orthogonal: error={error.item():.6f}")
        if not all(torch.isfinite(tensor).all() for tensor in (rs, rp, bs, bp)):
            raise RuntimeError("transform contains NaN/Inf")
        if transform.spectral_level not in (0, 1, 2) or transform.spatial_level not in (0, 1, 2):
            raise ValueError("transform levels must be 0, 1, or 2")
        return BranchSimilarityTransform(
            spectral_rotation=rs,
            spatial_rotation=rp,
            spectral_scale=float(transform.spectral_scale),
            spatial_scale=float(transform.spatial_scale),
            spectral_bias=bs,
            spatial_bias=bp,
            spectral_level=int(transform.spectral_level),
            spatial_level=int(transform.spatial_level),
        )

    def transform_matrix_and_bias(
        self, transform: BranchSimilarityTransform
    ) -> tuple[Tensor, Tensor]:
        transform = self._validate_transform(transform)
        a = torch.zeros(
            (self.feature_dim, self.feature_dim),
            device=self.device,
            dtype=self.dtype,
        )
        a[: self.spectral_dim, : self.spectral_dim] = (
            transform.spectral_scale * transform.spectral_rotation
        )
        a[self.spectral_dim :, self.spectral_dim :] = (
            transform.spatial_scale * transform.spatial_rotation
        )
        b = torch.cat([transform.spectral_bias, transform.spatial_bias], dim=0)
        return a, b

    def apply_transform(
        self, features: Tensor, transform: BranchSimilarityTransform
    ) -> Tensor:
        x = self._canonicalize_features(features)
        a, b = self.transform_matrix_and_bias(transform)
        return x @ a.transpose(0, 1) + b.view(1, -1)

    apply_transport = apply_transform

    @torch.no_grad()
    def build_transported_rows(
        self,
        class_ids: Sequence[int],
        *,
        transform: BranchSimilarityTransform,
    ) -> tuple[Dict[int, Row], Dict[str, Any]]:
        ids = _unique_ids(class_ids)
        self.assert_valid(ids, strict=True)
        transform = self._validate_transform(transform)
        a, b = self.transform_matrix_and_bias(transform)
        transported: Dict[int, Row] = {}
        closure_errors: list[float] = []
        for class_id in ids:
            row = self.get_class_row(class_id, clone=False)
            new_row = {
                "mean": (a @ row["mean"] + b).detach(),
                "loading": (a @ row["loading"]).detach(),
                "active_rank": row["active_rank"].detach().clone(),
                "residual_var_spectral": (
                    transform.spectral_scale ** 2 * row["residual_var_spectral"]
                ).detach(),
                "residual_var_spatial": (
                    transform.spatial_scale ** 2 * row["residual_var_spatial"]
                ).detach(),
                "spectral_shape_mean": row["spectral_shape_mean"].detach().clone(),
                "spectral_shape_var": row["spectral_shape_var"].detach().clone(),
                "spectral_shape_reliability": row[
                    "spectral_shape_reliability"
                ].detach().clone(),
                "sample_count": row["sample_count"].detach().clone(),
                "effective_sample_count": row[
                    "effective_sample_count"
                ].detach().clone(),
                "geometry_reliability": row["geometry_reliability"].detach().clone(),
                "reconstruction_error": row["reconstruction_error"].detach().clone(),
                "outlier_rate": row["outlier_rate"].detach().clone(),
            }
            old_cov = covariance_from_row(
                row, spectral_dim=self.spectral_dim, spatial_dim=self.spatial_dim
            )
            expected = _symmetrize(a @ old_cov @ a.transpose(0, 1))
            actual = covariance_from_row(
                new_row, spectral_dim=self.spectral_dim, spatial_dim=self.spatial_dim
            )
            error = (expected - actual).norm() / expected.norm().clamp_min(_EPS)
            closure_errors.append(float(error.item()))
            transported[class_id] = new_row
        return transported, {
            "class_ids": ids,
            "spectral_level": transform.spectral_level,
            "spatial_level": transform.spatial_level,
            "spectral_scale": transform.spectral_scale,
            "spatial_scale": transform.spatial_scale,
            "mean_closure_error": float(sum(closure_errors) / max(len(closure_errors), 1)),
            "maximum_closure_error": float(max(closure_errors) if closure_errors else 0.0),
            "exact_factor_transport": True,
        }

    # ------------------------------------------------------------------
    # Statistics, diagnostics, and commits
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_statistics(
        self,
        features: Tensor,
        labels: Tensor,
        class_ids: Sequence[int],
        *,
        rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Dict[str, float]:
        ids = _unique_ids(class_ids)
        x = self._canonicalize_features(features)
        y = torch.as_tensor(labels, device=self.device, dtype=torch.long).flatten()
        if y.numel() != x.size(0):
            raise ValueError("features and labels are not aligned")
        mapping = {class_id: index for index, class_id in enumerate(ids)}
        if any(int(label) not in mapping for label in y.tolist()):
            raise ValueError("labels contain classes outside class_ids")
        targets = torch.tensor(
            [mapping[int(label)] for label in y.tolist()],
            device=self.device,
            dtype=torch.long,
        )
        output = self.energy_matrix(x, ids, rows=rows)
        true_energy = output.energy.gather(1, targets[:, None]).squeeze(1)
        rivals = output.energy.clone()
        rivals.scatter_(1, targets[:, None], float("inf"))
        margin = rivals.min(dim=1).values - true_energy
        prediction = output.energy.argmin(dim=1)

        for local_index, class_id in enumerate(ids):
            if rows is not None and class_id in rows:
                continue
            mask = targets.eq(local_index)
            if not bool(mask.any()):
                continue
            self.energy_quantiles[class_id] = torch.quantile(
                true_energy[mask],
                torch.tensor([0.50, 0.95, 0.99], device=self.device, dtype=self.dtype),
            )
            self.margin_quantiles[class_id] = torch.quantile(
                margin[mask],
                torch.tensor([0.01, 0.05, 0.50], device=self.device, dtype=self.dtype),
            )
            self.statistics_ready[class_id] = True
        return {
            "accuracy": float(prediction.eq(targets).float().mean().item()),
            "margin_mean": float(margin.mean().item()),
            "margin_q05": float(torch.quantile(margin, 0.05).item()),
            "violation_rate": float(margin.lt(0).float().mean().item()),
        }

    def effective_dimension(
        self, class_ids: Optional[Sequence[int]] = None
    ) -> Tensor:
        ids = (
            torch.nonzero(self.valid_mask(), as_tuple=False).flatten().tolist()
            if class_ids is None
            else _unique_ids(class_ids)
        )
        values: list[Tensor] = []
        for class_id in ids:
            covariance = covariance_from_row(
                self.get_class_row(class_id, clone=False),
                spectral_dim=self.spectral_dim,
                spatial_dim=self.spatial_dim,
            )
            trace = covariance.trace()
            square_trace = (covariance @ covariance).trace()
            values.append(trace.square() / square_trace.clamp_min(_EPS))
        return torch.stack(values) if values else torch.empty((0,), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def admission_report(
        self,
        class_ids: Sequence[int],
        *,
        maximum_reconstruction_error: float = 0.75,
        minimum_effective_dimension: float = 1.25,
        require_statistics: bool = True,
    ) -> Dict[str, Any]:
        """Audit structural validity separately from geometry-fit quality.

        ``structural_ok`` covers conditions required for safe inference and
        checkpointing: valid finite rows, positive-definite covariances, and
        (optionally) available energy/margin statistics.

        ``quality_ok`` covers empirical adequacy diagnostics: effective
        covariance dimension and factor reconstruction error.  A quality
        failure is not a corrupt bank; trainers may enforce it as an
        experiment admission gate, but smoke runs may retain it as a warning.

        ``ok`` is retained for backward compatibility and remains the strict
        conjunction of structural and quality checks.
        """
        ids = _unique_ids(class_ids)
        if not ids:
            raise ValueError("class_ids cannot be empty")
        if not math.isfinite(float(maximum_reconstruction_error)):
            raise ValueError(
                "maximum_reconstruction_error must be finite"
            )
        if float(maximum_reconstruction_error) < 0.0:
            raise ValueError(
                "maximum_reconstruction_error must be non-negative"
            )
        if not math.isfinite(float(minimum_effective_dimension)):
            raise ValueError(
                "minimum_effective_dimension must be finite"
            )
        if float(minimum_effective_dimension) <= 0.0:
            raise ValueError(
                "minimum_effective_dimension must be positive"
            )

        validity = self.assert_valid(ids, strict=False)
        structural_errors = list(validity["errors"])
        quality_errors: list[str] = []

        index = torch.tensor(
            ids,
            device=self.device,
            dtype=torch.long,
        )
        dimensions = self.effective_dimension(ids)
        reconstruction = self.reconstruction_errors.index_select(0, index)
        ready = self.statistics_ready.index_select(0, index)

        if dimensions.numel() != len(ids):
            structural_errors.append(
                "effective-dimension vector does not match class IDs"
            )
        if reconstruction.numel() != len(ids):
            structural_errors.append(
                "reconstruction-error vector does not match class IDs"
            )
        if not torch.isfinite(dimensions).all():
            structural_errors.append(
                "effective feature dimensions contain NaN/Inf"
            )
        if not torch.isfinite(reconstruction).all():
            structural_errors.append(
                "factor reconstruction errors contain NaN/Inf"
            )

        per_class: Dict[int, Dict[str, Any]] = {}
        low_dimension_classes: list[int] = []
        high_reconstruction_classes: list[int] = []
        for local_index, class_id in enumerate(ids):
            dimension = (
                float(dimensions[local_index].item())
                if local_index < dimensions.numel()
                else float("nan")
            )
            reconstruction_error = (
                float(reconstruction[local_index].item())
                if local_index < reconstruction.numel()
                else float("nan")
            )
            statistics_available = (
                bool(ready[local_index].item())
                if local_index < ready.numel()
                else False
            )
            dimension_ok = (
                math.isfinite(dimension)
                and dimension >= float(minimum_effective_dimension)
            )
            reconstruction_ok = (
                math.isfinite(reconstruction_error)
                and reconstruction_error
                <= float(maximum_reconstruction_error)
            )
            if not dimension_ok:
                low_dimension_classes.append(int(class_id))
            if not reconstruction_ok:
                high_reconstruction_classes.append(int(class_id))
            per_class[int(class_id)] = {
                "effective_dimension": dimension,
                "minimum_effective_dimension": float(
                    minimum_effective_dimension
                ),
                "effective_dimension_ok": bool(dimension_ok),
                "reconstruction_error": reconstruction_error,
                "maximum_reconstruction_error": float(
                    maximum_reconstruction_error
                ),
                "reconstruction_error_ok": bool(reconstruction_ok),
                "statistics_ready": statistics_available,
            }

        if low_dimension_classes:
            quality_errors.append(
                "effective feature dimension is below threshold for "
                f"classes {low_dimension_classes}"
            )
        if high_reconstruction_classes:
            quality_errors.append(
                "factor reconstruction error exceeds threshold for "
                f"classes {high_reconstruction_classes}"
            )

        missing_statistics = [
            int(class_id)
            for local_index, class_id in enumerate(ids)
            if local_index >= ready.numel()
            or not bool(ready[local_index].item())
        ]
        if require_statistics and missing_statistics:
            structural_errors.append(
                "energy/margin statistics are absent for "
                f"{missing_statistics}"
            )

        statistics_complete = not missing_statistics
        minimum_dimension = (
            float(dimensions.min().item())
            if dimensions.numel() and torch.isfinite(dimensions).all()
            else float("nan")
        )
        maximum_error = (
            float(reconstruction.max().item())
            if reconstruction.numel()
            and torch.isfinite(reconstruction).all()
            else float("nan")
        )
        minimum_q05 = (
            float(
                self.margin_quantiles.index_select(0, index)[:, 1]
                .min()
                .item()
            )
            if statistics_complete
            else float("nan")
        )

        structural_ok = not structural_errors
        quality_ok = not quality_errors
        return {
            "ok": bool(structural_ok and quality_ok),
            "structural_ok": bool(structural_ok),
            "quality_ok": bool(quality_ok),
            "errors": [*structural_errors, *quality_errors],
            "structural_errors": structural_errors,
            "quality_errors": quality_errors,
            "warnings": list(quality_errors),
            "class_ids": ids,
            "per_class": per_class,
            "low_effective_dimension_classes": low_dimension_classes,
            "high_reconstruction_error_classes": (
                high_reconstruction_classes
            ),
            "minimum_effective_dimension": minimum_dimension,
            "maximum_reconstruction_error": maximum_error,
            "minimum_margin_q05": minimum_q05,
            "statistics_complete": bool(statistics_complete),
            "require_statistics": bool(require_statistics),
            "quality_thresholds": {
                "maximum_reconstruction_error": float(
                    maximum_reconstruction_error
                ),
                "minimum_effective_dimension": float(
                    minimum_effective_dimension
                ),
            },
        }

    def _atomic_write(
        self,
        rows: Mapping[int, Mapping[str, Any]],
        *,
        phase: int,
        replace_existing: bool,
    ) -> None:
        snapshot = self.export_snapshot()
        try:
            for class_id, row in rows.items():
                class_id = int(class_id)
                occupied = class_id < len(self) and bool(self.row_valid[class_id].item())
                if occupied and not replace_existing:
                    raise RuntimeError(f"refusing to overwrite occupied row {class_id}")
                if not occupied and replace_existing:
                    raise RuntimeError(f"cannot transport absent row {class_id}")
                created = int(self.phase_created[class_id].item()) if occupied else int(phase)
                self._write_row(
                    class_id,
                    row,
                    phase_created=created,
                    phase_updated=int(phase),
                )
            valid_ids = torch.nonzero(self.row_valid, as_tuple=False).flatten().tolist()
            self.assert_valid(valid_ids, strict=True)
        except Exception:
            self.load_snapshot(snapshot, strict=True)
            raise

    @torch.no_grad()
    def commit_base_rows(
        self,
        rows: Mapping[int, Mapping[str, Any]],
        *,
        base_class_ids: Sequence[int],
        phase: int = 0,
    ) -> Dict[str, Any]:
        ids = _unique_ids(base_class_ids)
        if set(rows) != set(ids):
            raise RuntimeError("base row IDs do not match base_class_ids")
        if bool(self.row_valid.any()):
            raise RuntimeError("base rows can only be committed to an empty bank")
        self._atomic_write(rows, phase=phase, replace_existing=False)
        return {
            "class_ids": ids,
            "phase": int(phase),
            "digest": self.rows_digest(ids),
            "atomic": True,
        }

    @torch.no_grad()
    def commit_incremental_phase(
        self,
        *,
        transported_old_rows: Mapping[int, Mapping[str, Any]],
        new_rows: Mapping[int, Mapping[str, Any]],
        old_class_ids: Sequence[int],
        new_class_ids: Sequence[int],
        phase: int,
        expected_old_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        old_ids = _unique_ids(old_class_ids)
        new_ids = _unique_ids(new_class_ids)
        if set(old_ids) & set(new_ids):
            raise RuntimeError("old/new class IDs overlap")
        if set(transported_old_rows) != set(old_ids):
            raise RuntimeError("transported old row IDs mismatch")
        if set(new_rows) != set(new_ids):
            raise RuntimeError("new row IDs mismatch")
        self.assert_valid(old_ids, strict=True)
        digest_before = self.rows_digest(old_ids)
        if expected_old_digest is not None and digest_before != str(expected_old_digest):
            raise RuntimeError("old-row digest mismatch before phase commit")

        snapshot = self.export_snapshot()
        try:
            self._atomic_write(transported_old_rows, phase=phase, replace_existing=True)
            self._atomic_write(new_rows, phase=phase, replace_existing=False)
            self.assert_valid([*old_ids, *new_ids], strict=True)
        except Exception:
            self.load_snapshot(snapshot, strict=True)
            raise
        return {
            "old_class_ids": old_ids,
            "new_class_ids": new_ids,
            "phase": int(phase),
            "old_digest_before": digest_before,
            "bank_digest_after": self.rows_digest([*old_ids, *new_ids]),
            "atomic": True,
        }

    # ------------------------------------------------------------------
    # Snapshots, digest, and memory audit
    # ------------------------------------------------------------------

    @staticmethod
    def _digest_tensor(hasher: "hashlib._Hash", tensor: Tensor) -> None:
        value = tensor.detach().to("cpu").contiguous()
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(str(tuple(value.shape)).encode("utf-8"))
        hasher.update(value.numpy().tobytes())

    def rows_digest(self, class_ids: Sequence[int]) -> str:
        ids = _unique_ids(class_ids)
        self.assert_valid(ids, strict=True)
        hasher = hashlib.sha256()
        hasher.update(str(self.SCHEMA_VERSION).encode("utf-8"))
        for class_id in ids:
            hasher.update(str(class_id).encode("utf-8"))
            row = self.get_class_row(class_id, clone=False)
            for name in sorted(row):
                hasher.update(name.encode("utf-8"))
                self._digest_tensor(hasher, row[name])
        return hasher.hexdigest()

    def export_snapshot(self) -> Dict[str, Tensor]:
        return {
            name: value.detach().clone()
            for name, value in self._buffers.items()
            if value is not None
        }

    @torch.no_grad()
    def load_snapshot(
        self,
        snapshot: Mapping[str, Tensor],
        *,
        strict: bool = True,
    ) -> None:
        current = set(self._buffers)
        supplied = set(snapshot)
        if strict and current != supplied:
            missing = sorted(current - supplied)
            extra = sorted(supplied - current)
            raise RuntimeError(f"snapshot mismatch: missing={missing}, extra={extra}")
        for name in current & supplied:
            target = self._buffers[name]
            value = torch.as_tensor(snapshot[name], device=self.device)
            if target is not None:
                value = value.to(dtype=target.dtype)
            setattr(self, name, value.detach().clone())

    @torch.no_grad()
    def reset_rows(self) -> None:
        device, dtype = self.device, self.dtype
        row_names = [
            "means",
            "loadings",
            "active_ranks",
            "residual_vars_spectral",
            "residual_vars_spatial",
            "spectral_shape_means",
            "spectral_shape_vars",
            "spectral_shape_reliability",
            "sample_counts",
            "effective_sample_counts",
            "geometry_reliability",
            "reconstruction_errors",
            "outlier_rates",
            "energy_quantiles",
            "margin_quantiles",
            "statistics_ready",
            "phase_created",
            "phase_updated",
            "row_valid",
        ]
        for name in row_names:
            del self._buffers[name]
        self._register_row_buffers(device, dtype)

    def memory_cost_summary(self) -> Dict[str, Any]:
        persistent = sum(
            value.numel() * value.element_size()
            for value in self._buffers.values()
            if value is not None
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "classes": int(self.valid_mask().sum().item()),
            "bytes": int(persistent),
            "megabytes": float(persistent / (1024.0 ** 2)),
            "stores_sample_level_memory": False,
            "classification_factorization": "p(z|c)",
            "spectral_relation_factorization": "p(h|c)",
            "feature_dim": self.feature_dim,
            "spectral_dim": self.spectral_dim,
            "spatial_dim": self.spatial_dim,
            "maximum_rank": self.maximum_rank,
        }


GeometryBank = HSIFactorGeometryBank
JointGeometryBank = HSIFactorGeometryBank
TransportClosedGeometryBank = HSIFactorGeometryBank










# from __future__ import annotations

# from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
# import hashlib
# import math

# import torch
# import torch.nn as nn


# Tensor = torch.Tensor
# Row = Dict[str, Tensor]
# _EPS = 1e-12
# _FORBIDDEN_MEMORY_NAMES = {
#     "raw_samples",
#     "raw_patches",
#     "old_samples",
#     "old_patches",
#     "stored_samples",
#     "stored_patches",
#     "feature_memory",
#     "old_features",
#     "stored_features",
#     "exemplars",
#     "exemplar_memory",
#     "memory_features",
#     "memory_patches",
# }


# def _unique_ids(values: Iterable[int]) -> list[int]:
#     ids: list[int] = []
#     seen: set[int] = set()
#     for value in values:
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError(f"class IDs must be non-negative, got {class_id}")
#         if class_id not in seen:
#             seen.add(class_id)
#             ids.append(class_id)
#     return ids


# def _symmetrize(matrix: Tensor) -> Tensor:
#     return 0.5 * (matrix + matrix.transpose(-1, -2))


# def _orthonormalize(basis: Tensor) -> Tensor:
#     if basis.dim() != 2:
#         raise ValueError("basis must be [D,R]")
#     if basis.size(1) == 0:
#         return basis
#     work = torch.nan_to_num(basis.float(), nan=0.0, posinf=0.0, neginf=0.0)
#     try:
#         q, _ = torch.linalg.qr(work, mode="reduced")
#     except RuntimeError:
#         q, _, _ = torch.linalg.svd(work, full_matrices=False)
#     return q[:, : basis.size(1)].to(dtype=basis.dtype)


# def _effective_sample_count(weights: Tensor) -> Tensor:
#     total = weights.sum().clamp_min(_EPS)
#     return total.square() / weights.square().sum().clamp_min(_EPS)


# def _low_rank_covariance(
#     basis: Tensor,
#     eigvals: Tensor,
#     residual_variance: Tensor,
#     active_rank: int,
#     dimension: int,
# ) -> Tensor:
#     residual = residual_variance.reshape(()).clamp_min(_EPS)
#     covariance = torch.eye(
#         dimension, device=basis.device, dtype=basis.dtype
#     ) * residual
#     if active_rank > 0:
#         active_basis = basis[:, :active_rank]
#         active_values = eigvals[:active_rank]
#         increments = (active_values - residual).clamp_min(0.0)
#         covariance = covariance + (
#             active_basis * increments.unsqueeze(0)
#         ) @ active_basis.transpose(0, 1)
#     return _symmetrize(covariance)


# def _factorize_covariance(
#     covariance: Tensor,
#     *,
#     maximum_rank: int,
#     variance_floor: float,
#     retained_energy: float,
#     noise_ratio: float,
#     minimum_rank: int = 0,
# ) -> Dict[str, Tensor]:
#     if covariance.dim() != 2 or covariance.size(0) != covariance.size(1):
#         raise ValueError("covariance must be square")
#     dimension = covariance.size(0)
#     covariance = _symmetrize(
#         torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
#     )
#     floor = float(max(variance_floor, 1e-12))
#     ridge = torch.eye(
#         dimension, device=covariance.device, dtype=covariance.dtype
#     ) * floor
#     eigenvalues, eigenvectors = torch.linalg.eigh(covariance + ridge)
#     order = torch.argsort(eigenvalues, descending=True)
#     eigenvalues = eigenvalues.index_select(0, order).clamp_min(floor)
#     eigenvectors = eigenvectors.index_select(1, order)

#     rank_cap = int(max(0, min(maximum_rank, dimension)))
#     if rank_cap == 0:
#         residual = eigenvalues.mean().clamp_min(floor)
#         return {
#             "basis": covariance.new_zeros((dimension, 0)),
#             "eigvals": covariance.new_zeros((0,)),
#             "res_var": residual,
#             "active_rank": torch.tensor(
#                 0, device=covariance.device, dtype=torch.long
#             ),
#             "approximation_error": covariance.new_tensor(0.0),
#         }

#     tail = eigenvalues[rank_cap:]
#     residual = (
#         tail.mean()
#         if tail.numel() > 0
#         else eigenvalues[-1]
#     ).clamp_min(floor)
#     candidate = eigenvalues[:rank_cap]
#     excess = (candidate - residual).clamp_min(0.0)
#     total_excess = excess.sum()

#     if float(total_excess.item()) <= _EPS:
#         active_rank = 0
#     else:
#         cumulative = torch.cumsum(excess, dim=0) / total_excess
#         hit = torch.nonzero(cumulative >= retained_energy, as_tuple=False)
#         energy_rank = int(hit[0].item()) + 1 if hit.numel() else rank_cap
#         signal_rank = int(
#             candidate.ge(residual * float(max(noise_ratio, 1.0))).sum().item()
#         )
#         active_rank = min(rank_cap, energy_rank, signal_rank)
#         active_rank = max(
#             active_rank,
#             min(int(max(minimum_rank, 0)), rank_cap),
#         )

#     basis = covariance.new_zeros((dimension, rank_cap))
#     eigvals = covariance.new_full((rank_cap,), float(residual.item()))
#     if active_rank > 0:
#         active_basis = _orthonormalize(eigenvectors[:, :active_rank])
#         basis[:, :active_rank] = active_basis
#         eigvals[:active_rank] = candidate[:active_rank]

#     reconstructed = _low_rank_covariance(
#         basis, eigvals, residual, active_rank, dimension
#     )
#     denominator = covariance.norm().clamp_min(_EPS)
#     approximation_error = (reconstructed - covariance).norm() / denominator
#     return {
#         "basis": basis,
#         "eigvals": eigvals,
#         "res_var": residual,
#         "active_rank": torch.tensor(
#             active_rank, device=covariance.device, dtype=torch.long
#         ),
#         "approximation_error": approximation_error,
#     }


# def _low_rank_energy_from_delta(
#     delta: Tensor,
#     *,
#     bases: Tensor,
#     eigvals: Tensor,
#     residual_variances: Tensor,
#     active_ranks: Tensor,
#     variance_floor: float,
#     logdet_weight: float,
#     normalize_by_dimension: bool,
#     valid_mask: Optional[Tensor] = None,
#     invalid_energy: float = 1e6,
# ) -> Dict[str, Tensor]:
#     if delta.dim() != 3:
#         raise ValueError("delta must be [N,C,D]")
#     sample_count, class_count, dimension = delta.shape
#     if bases.dim() != 3 or bases.shape[:2] != (class_count, dimension):
#         raise ValueError("bases must be [C,D,R]")
#     maximum_rank = bases.size(2)
#     if eigvals.shape != (class_count, maximum_rank):
#         raise ValueError("eigvals must be [C,R]")
#     if residual_variances.numel() != class_count:
#         raise ValueError("residual_variances must contain C values")
#     if active_ranks.numel() != class_count:
#         raise ValueError("active_ranks must contain C values")

#     device, dtype = delta.device, delta.dtype
#     bases = bases.to(device=device, dtype=dtype)
#     eigvals = eigvals.to(device=device, dtype=dtype)
#     residual_variances = residual_variances.to(
#         device=device, dtype=dtype
#     ).flatten()
#     active_ranks = active_ranks.to(device=device, dtype=torch.long).flatten()
#     if bool((active_ranks < 0).any()) or bool(
#         (active_ranks > maximum_rank).any()
#     ):
#         raise ValueError("active_ranks are outside [0,R]")

#     floor = float(max(variance_floor, 1e-12))
#     eigvals = eigvals.clamp_min(floor)
#     residual_variances = residual_variances.clamp_min(floor)

#     rank_index = torch.arange(
#         maximum_rank, device=device
#     ).view(1, 1, maximum_rank)
#     active = rank_index < active_ranks.view(1, class_count, 1)
#     active_float = active.to(dtype=dtype)

#     coordinates = torch.einsum("ncd,cdr->ncr", delta, bases)
#     active_coordinates = coordinates * active_float
#     parallel = (
#         active_coordinates.square() / eigvals.unsqueeze(0)
#     ).sum(dim=2)
#     reconstruction = torch.einsum(
#         "ncr,cdr->ncd", active_coordinates, bases
#     )
#     orthogonal = delta - reconstruction
#     residual = (
#         orthogonal.square().sum(dim=2)
#         / residual_variances.view(1, class_count)
#     )
#     quadratic = parallel + residual

#     logdet = (
#         active_float * eigvals.unsqueeze(0).log()
#     ).sum(dim=2).squeeze(0)
#     logdet = logdet + (
#         dimension - active_ranks
#     ).to(dtype) * residual_variances.log()
#     volume = logdet.view(1, class_count).expand(sample_count, class_count)
#     energy = quadratic + float(logdet_weight) * volume

#     if normalize_by_dimension:
#         scale = float(max(dimension, 1))
#         parallel = parallel / scale
#         residual = residual / scale
#         quadratic = quadratic / scale
#         volume = volume / scale
#         energy = energy / scale

#     if valid_mask is not None:
#         valid_mask = valid_mask.to(device=device, dtype=torch.bool).flatten()
#         if valid_mask.numel() != class_count:
#             raise ValueError("valid_mask must contain C values")
#         invalid = torch.full_like(energy, float(invalid_energy))
#         energy = torch.where(valid_mask.view(1, -1), energy, invalid)
#         quadratic = torch.where(valid_mask.view(1, -1), quadratic, invalid)
#         parallel = torch.where(
#             valid_mask.view(1, -1), parallel, torch.zeros_like(parallel)
#         )
#         residual = torch.where(
#             valid_mask.view(1, -1), residual, torch.zeros_like(residual)
#         )
#         volume = torch.where(
#             valid_mask.view(1, -1), volume, torch.zeros_like(volume)
#         )

#     return {
#         "energy": energy,
#         "quadratic": quadratic,
#         "parallel": parallel,
#         "residual": residual,
#         "volume": volume,
#     }


# class SpectralConditionedGeometryBank(nn.Module):
#     """Aggregate memory for p(s|c) p(z|s,c).

#     ``s`` is a fixed low-dimensional anchor obtained from ordered physical-band
#     spectra. ``z`` is the current spectral-spatial feature. The bank stores only
#     aggregate class rows and supports analytical affine push-forward when the
#     feature coordinate changes.
#     """

#     SCHEMA_VERSION = 1

#     def __init__(
#         self,
#         d_model: int,
#         rank: int,
#         *,
#         raw_spectral_dim: int,
#         spectral_anchor_dim: int = 12,
#         spectral_rank: int = 4,
#         device: Union[str, torch.device] = "cpu",
#         dtype: torch.dtype = torch.float32,
#         feature_variance_floor: float = 1e-4,
#         spectral_variance_floor: float = 1e-4,
#         feature_shrinkage: float = 0.10,
#         spectral_shrinkage: float = 0.20,
#         shrinkage_tau: float = 20.0,
#         retained_energy: float = 0.95,
#         noise_ratio: float = 1.50,
#         coupling_ridge: float = 1e-2,
#         coupling_shrinkage_tau: float = 20.0,
#         robust_iterations: int = 3,
#         robust_huber_delta: float = 2.5,
#         spatial_purity_power: float = 1.0,
#         spectral_weight: float = 1.0,
#         feature_weight: float = 1.0,
#         logdet_weight: float = 1.0,
#         boundary_oversample_factor: int = 8,
#         maximum_transport_condition: float = 2.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.rank = int(rank)
#         self.raw_spectral_dim = int(raw_spectral_dim)
#         self.spectral_anchor_dim = int(spectral_anchor_dim)
#         self.spectral_rank = int(spectral_rank)

#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive")
#         if not 0 <= self.rank <= self.d_model:
#             raise ValueError("rank must be in [0,d_model]")
#         if self.raw_spectral_dim <= 0:
#             raise ValueError("raw_spectral_dim must be positive")
#         if not 1 <= self.spectral_anchor_dim <= self.raw_spectral_dim:
#             raise ValueError(
#                 "spectral_anchor_dim must be in [1,raw_spectral_dim]"
#             )
#         if not 0 <= self.spectral_rank <= self.spectral_anchor_dim:
#             raise ValueError(
#                 "spectral_rank must be in [0,spectral_anchor_dim]"
#             )

#         self.feature_variance_floor = float(
#             max(feature_variance_floor, 1e-12)
#         )
#         self.spectral_variance_floor = float(
#             max(spectral_variance_floor, 1e-12)
#         )
#         self.feature_shrinkage = float(
#             min(max(feature_shrinkage, 0.0), 0.95)
#         )
#         self.spectral_shrinkage = float(
#             min(max(spectral_shrinkage, 0.0), 0.95)
#         )
#         self.shrinkage_tau = float(max(shrinkage_tau, 0.0))
#         self.retained_energy = float(
#             min(max(retained_energy, 0.50), 0.999)
#         )
#         self.noise_ratio = float(max(noise_ratio, 1.0))
#         self.coupling_ridge = float(max(coupling_ridge, 1e-8))
#         self.coupling_shrinkage_tau = float(
#             max(coupling_shrinkage_tau, 0.0)
#         )
#         self.robust_iterations = int(max(1, robust_iterations))
#         self.robust_huber_delta = float(max(robust_huber_delta, 1e-3))
#         self.spatial_purity_power = float(max(spatial_purity_power, 0.0))
#         self.spectral_weight = float(max(spectral_weight, 0.0))
#         self.feature_weight = float(max(feature_weight, 0.0))
#         if self.spectral_weight == 0.0 and self.feature_weight == 0.0:
#             raise ValueError("at least one energy weight must be positive")
#         self.logdet_weight = float(max(logdet_weight, 0.0))
#         self.boundary_oversample_factor = int(
#             max(2, boundary_oversample_factor)
#         )
#         self.maximum_transport_condition = float(
#             max(maximum_transport_condition, 1.0)
#         )

#         dev = torch.device(device)

#         self.register_buffer(
#             "anchor_band_mean",
#             torch.zeros(self.raw_spectral_dim, device=dev, dtype=dtype),
#         )
#         self.register_buffer(
#             "anchor_band_std",
#             torch.ones(self.raw_spectral_dim, device=dev, dtype=dtype),
#         )
#         self.register_buffer(
#             "anchor_basis",
#             torch.zeros(
#                 self.raw_spectral_dim,
#                 self.spectral_anchor_dim,
#                 device=dev,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "anchor_scales",
#             torch.ones(self.spectral_anchor_dim, device=dev, dtype=dtype),
#         )
#         self.register_buffer(
#             "anchor_active_dim",
#             torch.tensor(0, device=dev, dtype=torch.long),
#         )
#         self.register_buffer(
#             "anchor_sample_count",
#             torch.tensor(0, device=dev, dtype=torch.long),
#         )
#         self.register_buffer(
#             "anchor_ready",
#             torch.tensor(False, device=dev, dtype=torch.bool),
#         )
#         self.register_buffer(
#             "anchor_frozen",
#             torch.tensor(False, device=dev, dtype=torch.bool),
#         )

#         self._register_row_buffers(dev, dtype)

#     def _register_row_buffers(
#         self, device: torch.device, dtype: torch.dtype
#     ) -> None:
#         c, s, d = 0, self.spectral_anchor_dim, self.d_model
#         rs, rz = self.spectral_rank, self.rank

#         self.register_buffer(
#             "spectral_means", torch.empty((c, s), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "spectral_bases",
#             torch.empty((c, s, rs), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "spectral_eigvals",
#             torch.empty((c, rs), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "spectral_res_vars", torch.empty((c,), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "spectral_active_ranks",
#             torch.empty((c,), device=device, dtype=torch.long),
#         )

#         self.register_buffer(
#             "feature_means", torch.empty((c, d), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "spectral_couplings",
#             torch.empty((c, d, s), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "feature_bases",
#             torch.empty((c, d, rz), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "feature_eigvals",
#             torch.empty((c, rz), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "feature_res_vars", torch.empty((c,), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "feature_active_ranks",
#             torch.empty((c,), device=device, dtype=torch.long),
#         )

#         self.register_buffer(
#             "sample_counts", torch.empty((c,), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "effective_sample_counts",
#             torch.empty((c,), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "reliability", torch.empty((c,), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "coupling_explained_variance",
#             torch.empty((c,), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "outlier_rates", torch.empty((c,), device=device, dtype=dtype)
#         )
#         self.register_buffer(
#             "energy_quantiles",
#             torch.empty((c, 3), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "margin_quantiles",
#             torch.empty((c, 3), device=device, dtype=dtype),
#         )
#         self.register_buffer(
#             "statistics_ready",
#             torch.empty((c,), device=device, dtype=torch.bool),
#         )
#         self.register_buffer(
#             "phase_created",
#             torch.empty((c,), device=device, dtype=torch.long),
#         )
#         self.register_buffer(
#             "phase_updated",
#             torch.empty((c,), device=device, dtype=torch.long),
#         )
#         self.register_buffer(
#             "row_valid", torch.empty((c,), device=device, dtype=torch.bool)
#         )

#     @property
#     def device(self) -> torch.device:
#         return self.feature_means.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.feature_means.dtype

#     def __len__(self) -> int:
#         return int(self.row_valid.numel())

#     def _canonicalize_features(self, features: Tensor) -> Tensor:
#         value = torch.as_tensor(
#             features, device=self.device, dtype=self.dtype
#         )
#         if value.dim() != 2 or value.size(1) != self.d_model:
#             raise ValueError(
#                 f"features must be [N,{self.d_model}], got {tuple(value.shape)}"
#             )
#         if not torch.isfinite(value).all():
#             raise RuntimeError("features contain NaN/Inf")
#         return value

#     def _canonicalize_raw_spectra(self, raw_spectra: Tensor) -> Tensor:
#         value = torch.as_tensor(
#             raw_spectra, device=self.device, dtype=self.dtype
#         )
#         if value.dim() != 2 or value.size(1) != self.raw_spectral_dim:
#             raise ValueError(
#                 "raw_spectra must be "
#                 f"[N,{self.raw_spectral_dim}], got {tuple(value.shape)}"
#             )
#         if not torch.isfinite(value).all():
#             raise RuntimeError("raw_spectra contain NaN/Inf")
#         return value

#     def _canonicalize_anchors(self, anchors: Tensor) -> Tensor:
#         value = torch.as_tensor(
#             anchors, device=self.device, dtype=self.dtype
#         )
#         if value.dim() != 2 or value.size(1) != self.spectral_anchor_dim:
#             raise ValueError(
#                 "spectral_anchors must be "
#                 f"[N,{self.spectral_anchor_dim}], got {tuple(value.shape)}"
#             )
#         if not torch.isfinite(value).all():
#             raise RuntimeError("spectral_anchors contain NaN/Inf")
#         return value

#     @torch.no_grad()
#     def fit_spectral_anchor(
#         self,
#         raw_spectra: Tensor,
#         *,
#         freeze: bool = True,
#         overwrite: bool = False,
#     ) -> Dict[str, Any]:
#         spectra = self._canonicalize_raw_spectra(raw_spectra)
#         if spectra.size(0) < 3:
#             raise ValueError("at least three spectra are required")
#         if bool(self.anchor_ready.item()) and not overwrite:
#             raise RuntimeError("spectral anchor already exists")
#         if bool(self.anchor_frozen.item()) and overwrite:
#             raise RuntimeError("frozen spectral anchor cannot be overwritten")
#         if bool(self.row_valid.any()):
#             raise RuntimeError(
#                 "fit the spectral anchor before committing class rows"
#             )

#         band_mean = spectra.mean(dim=0)
#         band_std = spectra.std(dim=0, unbiased=False).clamp_min(
#             math.sqrt(self.spectral_variance_floor)
#         )
#         standardized = (spectra - band_mean) / band_std
#         standardized = standardized - standardized.mean(dim=0, keepdim=True)

#         _, singular_values, vh = torch.linalg.svd(
#             standardized, full_matrices=False
#         )
#         active_dim = min(
#             self.spectral_anchor_dim,
#             max(int(spectra.size(0)) - 1, 1),
#             int(vh.size(0)),
#         )
#         basis = torch.zeros_like(self.anchor_basis)
#         scales = torch.ones_like(self.anchor_scales)
#         basis[:, :active_dim] = _orthonormalize(
#             vh[:active_dim].transpose(0, 1)
#         )
#         eigenvalues = (
#             singular_values[:active_dim].square()
#             / float(max(int(spectra.size(0)) - 1, 1))
#         ).clamp_min(self.spectral_variance_floor)
#         scales[:active_dim] = eigenvalues.sqrt()

#         self.anchor_band_mean.copy_(band_mean)
#         self.anchor_band_std.copy_(band_std)
#         self.anchor_basis.copy_(basis)
#         self.anchor_scales.copy_(scales)
#         self.anchor_active_dim.fill_(active_dim)
#         self.anchor_sample_count.fill_(int(spectra.size(0)))
#         self.anchor_ready.fill_(True)
#         self.anchor_frozen.fill_(bool(freeze))

#         return {
#             "sample_count": int(spectra.size(0)),
#             "raw_spectral_dim": self.raw_spectral_dim,
#             "anchor_dim": self.spectral_anchor_dim,
#             "active_anchor_dim": active_dim,
#             "frozen": bool(freeze),
#         }

#     def assert_anchor_ready(self) -> None:
#         if not bool(self.anchor_ready.item()):
#             raise RuntimeError(
#                 "spectral anchor is absent; fit it once from base raw spectra"
#             )
#         if int(self.anchor_active_dim.item()) <= 0:
#             raise RuntimeError("spectral anchor has no active dimensions")
#         if not torch.isfinite(self.anchor_basis).all():
#             raise RuntimeError("spectral anchor basis contains NaN/Inf")
#         if bool((self.anchor_scales <= 0).any()):
#             raise RuntimeError("spectral anchor scales must be positive")

#     def encode_spectra(self, raw_spectra: Tensor) -> Tensor:
#         self.assert_anchor_ready()
#         spectra = self._canonicalize_raw_spectra(raw_spectra)
#         standardized = (
#             spectra - self.anchor_band_mean.view(1, -1)
#         ) / self.anchor_band_std.view(1, -1)
#         anchors = (
#             standardized @ self.anchor_basis
#         ) / self.anchor_scales.view(1, -1)
#         active_dim = int(self.anchor_active_dim.item())
#         if active_dim < self.spectral_anchor_dim:
#             anchors = anchors.clone()
#             anchors[:, active_dim:] = 0.0
#         return anchors

#     def _resolve_anchors(
#         self,
#         *,
#         raw_spectra: Optional[Tensor],
#         spectral_anchors: Optional[Tensor],
#     ) -> Tensor:
#         if (raw_spectra is None) == (spectral_anchors is None):
#             raise ValueError(
#                 "pass exactly one of raw_spectra or spectral_anchors"
#             )
#         if raw_spectra is not None:
#             return self.encode_spectra(raw_spectra)
#         return self._canonicalize_anchors(spectral_anchors)

#     def _external_weights(
#         self,
#         sample_weights: Optional[Tensor],
#         sample_count: int,
#     ) -> Tensor:
#         if sample_weights is None:
#             weights = torch.ones(
#                 sample_count, device=self.device, dtype=self.dtype
#             )
#         else:
#             weights = torch.as_tensor(
#                 sample_weights, device=self.device, dtype=self.dtype
#             ).flatten()
#             if weights.numel() != sample_count:
#                 raise ValueError(
#                     "sample_weights must contain one value per sample"
#                 )
#             if not torch.isfinite(weights).all():
#                 raise RuntimeError("sample_weights contain NaN/Inf")
#             weights = weights.clamp(0.0, 1.0)
#             if self.spatial_purity_power != 1.0:
#                 weights = weights.pow(self.spatial_purity_power)
#         if float(weights.sum().item()) <= _EPS:
#             raise ValueError("sample_weights sum to zero")
#         return weights / weights.mean().clamp_min(_EPS)

#     def _joint_robust_weights(
#         self,
#         features: Tensor,
#         anchors: Tensor,
#         sample_weights: Optional[Tensor],
#     ) -> Tuple[Tensor, Tensor, Tensor]:
#         count = features.size(0)
#         base = self._external_weights(sample_weights, count)
#         weights = base.clone()
#         outlier_gate = torch.ones_like(weights)

#         for _ in range(self.robust_iterations):
#             denominator = weights.sum().clamp_min(_EPS)
#             feature_center = (
#                 weights[:, None] * features
#             ).sum(dim=0) / denominator
#             spectral_center = (
#                 weights[:, None] * anchors
#             ).sum(dim=0) / denominator

#             feature_distance = (features - feature_center).norm(dim=1)
#             spectral_distance = (anchors - spectral_center).norm(dim=1)
#             feature_positive = feature_distance[feature_distance > 0]
#             spectral_positive = spectral_distance[spectral_distance > 0]
#             feature_scale = (
#                 feature_positive.median()
#                 if feature_positive.numel()
#                 else features.new_tensor(1.0)
#             ).clamp_min(math.sqrt(self.feature_variance_floor))
#             spectral_scale = (
#                 spectral_positive.median()
#                 if spectral_positive.numel()
#                 else anchors.new_tensor(1.0)
#             ).clamp_min(math.sqrt(self.spectral_variance_floor))

#             joint_distance = torch.sqrt(
#                 (feature_distance / feature_scale).square()
#                 + (spectral_distance / spectral_scale).square()
#             )
#             outlier_gate = torch.clamp(
#                 joint_distance.new_tensor(self.robust_huber_delta)
#                 / joint_distance.clamp_min(_EPS),
#                 max=1.0,
#             )
#             weights = base * outlier_gate
#             weights = weights / weights.mean().clamp_min(_EPS)

#         effective = _effective_sample_count(weights)
#         outlier_rate = outlier_gate.lt(0.999).float().mean()
#         return weights, effective, outlier_rate

#     def _rank_cap(
#         self, effective_count: float, maximum_rank: int, dimension: int
#     ) -> int:
#         return int(
#             max(
#                 0,
#                 min(
#                     maximum_rank,
#                     dimension,
#                     max(int(math.floor(effective_count)) - 1, 0),
#                 ),
#             )
#         )

#     def _fit_low_rank(
#         self,
#         values: Tensor,
#         weights: Tensor,
#         *,
#         maximum_rank: int,
#         variance_floor: float,
#         base_shrinkage: float,
#     ) -> Dict[str, Tensor]:
#         count, dimension = values.shape
#         if count <= 0:
#             raise ValueError("cannot fit geometry from zero samples")
#         total_weight = weights.sum().clamp_min(_EPS)
#         mean = (weights[:, None] * values).sum(dim=0) / total_weight
#         centered = values - mean
#         effective = _effective_sample_count(weights)
#         denominator = (
#             total_weight
#             - weights.square().sum() / total_weight
#         ).clamp_min(1.0)
#         weighted = centered * weights.sqrt().unsqueeze(1)
#         covariance = _symmetrize(weighted.transpose(0, 1) @ weighted / denominator)

#         rank_cap = self._rank_cap(
#             float(effective.item()), maximum_rank, dimension
#         )
#         factor = _factorize_covariance(
#             covariance,
#             maximum_rank=rank_cap,
#             variance_floor=variance_floor,
#             retained_energy=self.retained_energy,
#             noise_ratio=self.noise_ratio,
#         )

#         average_variance = (
#             covariance.diagonal().mean().clamp_min(variance_floor)
#         )
#         count_shrinkage = (
#             self.shrinkage_tau
#             / (float(effective.item()) + self.shrinkage_tau)
#             if self.shrinkage_tau > 0.0
#             else 0.0
#         )
#         shrinkage = min(base_shrinkage + count_shrinkage, 0.95)
#         residual = (
#             (1.0 - shrinkage) * factor["res_var"]
#             + shrinkage * average_variance
#         ).clamp_min(variance_floor)
#         eigvals = factor["eigvals"].clone()
#         active_rank = int(factor["active_rank"].item())
#         if active_rank > 0:
#             eigvals[:active_rank] = (
#                 (1.0 - shrinkage) * eigvals[:active_rank]
#                 + shrinkage * average_variance
#             ).clamp_min(variance_floor)
#         if active_rank < eigvals.numel():
#             eigvals[active_rank:] = residual

#         compact_basis = factor["basis"]
#         compact_eigvals = eigvals
#         basis = values.new_zeros((dimension, maximum_rank))
#         full_eigvals = values.new_full(
#             (maximum_rank,), float(residual.item())
#         )
#         if compact_basis.size(1) > 0:
#             basis[:, : compact_basis.size(1)] = compact_basis
#             full_eigvals[: compact_eigvals.numel()] = compact_eigvals

#         covariance_shrunk = _low_rank_covariance(
#             basis,
#             full_eigvals,
#             residual,
#             active_rank,
#             dimension,
#         )
#         explained = (
#             1.0
#             - (covariance_shrunk - covariance).norm()
#             / covariance.norm().clamp_min(_EPS)
#         ).clamp(0.0, 1.0)
#         return {
#             "mean": mean,
#             "basis": basis,
#             "eigvals": full_eigvals,
#             "res_var": residual,
#             "active_rank": factor["active_rank"],
#             "effective_count": effective,
#             "captured_energy": explained,
#         }

#     def _fit_conditional_feature(
#         self,
#         features: Tensor,
#         anchors: Tensor,
#         weights: Tensor,
#         spectral_mean: Tensor,
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#         delta = anchors - spectral_mean.view(1, -1)
#         design = torch.cat(
#             [torch.ones_like(delta[:, :1]), delta], dim=1
#         )
#         normalized = weights / weights.sum().clamp_min(_EPS)
#         gram = design.transpose(0, 1) @ (
#             normalized[:, None] * design
#         )
#         penalty = torch.eye(
#             design.size(1), device=self.device, dtype=self.dtype
#         ) * self.coupling_ridge
#         penalty[0, 0] = 0.0
#         cross = design.transpose(0, 1) @ (
#             normalized[:, None] * features
#         )
#         try:
#             coefficients = torch.linalg.solve(gram + penalty, cross)
#         except RuntimeError:
#             coefficients = torch.linalg.pinv(gram + penalty) @ cross

#         intercept = coefficients[0]
#         coupling = coefficients[1:].transpose(0, 1).contiguous()
#         effective = float(_effective_sample_count(weights).item())
#         shrinkage = (
#             effective / (effective + self.coupling_shrinkage_tau)
#             if self.coupling_shrinkage_tau > 0.0
#             else 1.0
#         )
#         coupling = coupling * shrinkage
#         predicted = intercept.view(1, -1) + delta @ coupling.transpose(0, 1)
#         residual = features - predicted

#         feature_centered = features - (
#             normalized[:, None] * features
#         ).sum(dim=0, keepdim=True)
#         total = (
#             normalized[:, None] * feature_centered.square()
#         ).sum()
#         unexplained = (
#             normalized[:, None] * residual.square()
#         ).sum()
#         explained = (
#             1.0 - unexplained / total.clamp_min(self.feature_variance_floor)
#         ).clamp(0.0, 1.0)
#         return intercept, coupling, residual, explained

#     @torch.no_grad()
#     def extract_rows(
#         self,
#         features: Tensor,
#         labels: Tensor,
#         *,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         sample_weights: Optional[Tensor] = None,
#         class_ids: Optional[Iterable[int]] = None,
#     ) -> Dict[int, Row]:
#         x = self._canonicalize_features(features)
#         s = self._resolve_anchors(
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#         )
#         y = torch.as_tensor(
#             labels, device=self.device, dtype=torch.long
#         ).flatten()
#         if x.size(0) != s.size(0) or y.numel() != x.size(0):
#             raise ValueError("features, spectra, and labels are not aligned")
#         if bool((y < 0).any()):
#             raise ValueError("negative class labels are forbidden")
#         weights = None
#         if sample_weights is not None:
#             weights = torch.as_tensor(
#                 sample_weights, device=self.device, dtype=self.dtype
#             ).flatten()
#             if weights.numel() != x.size(0):
#                 raise ValueError("sample_weights length mismatch")

#         allowed = None if class_ids is None else set(_unique_ids(class_ids))
#         rows: Dict[int, Row] = {}
#         for class_tensor in torch.unique(y, sorted=True):
#             class_id = int(class_tensor.item())
#             if allowed is not None and class_id not in allowed:
#                 continue
#             mask = y.eq(class_tensor)
#             class_x = x[mask]
#             class_s = s[mask]
#             class_w = None if weights is None else weights[mask]
#             robust_weights, effective_count, outlier_rate = (
#                 self._joint_robust_weights(class_x, class_s, class_w)
#             )

#             spectral = self._fit_low_rank(
#                 class_s,
#                 robust_weights,
#                 maximum_rank=self.spectral_rank,
#                 variance_floor=self.spectral_variance_floor,
#                 base_shrinkage=self.spectral_shrinkage,
#             )
#             intercept, coupling, feature_residual, coupling_r2 = (
#                 self._fit_conditional_feature(
#                     class_x,
#                     class_s,
#                     robust_weights,
#                     spectral["mean"],
#                 )
#             )
#             feature = self._fit_low_rank(
#                 feature_residual,
#                 robust_weights,
#                 maximum_rank=self.rank,
#                 variance_floor=self.feature_variance_floor,
#                 base_shrinkage=self.feature_shrinkage,
#             )
#             feature_mean = intercept + feature["mean"]
#             sample_reliability = class_x.new_tensor(
#                 float(effective_count.item())
#                 / (
#                     float(effective_count.item())
#                     + max(self.shrinkage_tau, 1.0)
#                 )
#             )
#             reliability = (
#                 0.30 * sample_reliability
#                 + 0.25 * spectral["captured_energy"]
#                 + 0.25 * feature["captured_energy"]
#                 + 0.20 * coupling_r2
#             ).clamp(0.0, 1.0)

#             rows[class_id] = {
#                 "spectral_mean": spectral["mean"].detach(),
#                 "spectral_basis": spectral["basis"].detach(),
#                 "spectral_eigvals": spectral["eigvals"].detach(),
#                 "spectral_res_var": spectral["res_var"].detach(),
#                 "spectral_active_rank": spectral[
#                     "active_rank"
#                 ].detach(),
#                 "feature_mean": feature_mean.detach(),
#                 "spectral_coupling": coupling.detach(),
#                 "feature_basis": feature["basis"].detach(),
#                 "feature_eigvals": feature["eigvals"].detach(),
#                 "feature_res_var": feature["res_var"].detach(),
#                 "feature_active_rank": feature[
#                     "active_rank"
#                 ].detach(),
#                 "sample_count": class_x.new_tensor(float(class_x.size(0))),
#                 "effective_sample_count": effective_count.detach(),
#                 "reliability": reliability.detach(),
#                 "coupling_explained_variance": coupling_r2.detach(),
#                 "outlier_rate": outlier_rate.detach(),
#             }

#         if allowed is not None and set(rows) != allowed:
#             missing = sorted(allowed - set(rows))
#             raise RuntimeError(f"no samples were provided for classes {missing}")
#         return rows

#     extract_geometry = extract_rows

#     def _append_empty_row(self) -> None:
#         def append(name: str, value: Tensor) -> None:
#             setattr(
#                 self,
#                 name,
#                 torch.cat([getattr(self, name), value], dim=0),
#             )

#         append(
#             "spectral_means",
#             torch.zeros(
#                 (1, self.spectral_anchor_dim),
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "spectral_bases",
#             torch.zeros(
#                 (1, self.spectral_anchor_dim, self.spectral_rank),
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "spectral_eigvals",
#             torch.full(
#                 (1, self.spectral_rank),
#                 self.spectral_variance_floor,
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "spectral_res_vars",
#             torch.full(
#                 (1,),
#                 self.spectral_variance_floor,
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "spectral_active_ranks",
#             torch.zeros((1,), device=self.device, dtype=torch.long),
#         )
#         append(
#             "feature_means",
#             torch.zeros(
#                 (1, self.d_model), device=self.device, dtype=self.dtype
#             ),
#         )
#         append(
#             "spectral_couplings",
#             torch.zeros(
#                 (1, self.d_model, self.spectral_anchor_dim),
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "feature_bases",
#             torch.zeros(
#                 (1, self.d_model, self.rank),
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "feature_eigvals",
#             torch.full(
#                 (1, self.rank),
#                 self.feature_variance_floor,
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "feature_res_vars",
#             torch.full(
#                 (1,),
#                 self.feature_variance_floor,
#                 device=self.device,
#                 dtype=self.dtype,
#             ),
#         )
#         append(
#             "feature_active_ranks",
#             torch.zeros((1,), device=self.device, dtype=torch.long),
#         )
#         for name in (
#             "sample_counts",
#             "effective_sample_counts",
#             "reliability",
#             "coupling_explained_variance",
#             "outlier_rates",
#         ):
#             append(
#                 name,
#                 torch.zeros((1,), device=self.device, dtype=self.dtype),
#             )
#         append(
#             "energy_quantiles",
#             torch.zeros((1, 3), device=self.device, dtype=self.dtype),
#         )
#         append(
#             "margin_quantiles",
#             torch.zeros((1, 3), device=self.device, dtype=self.dtype),
#         )
#         append(
#             "statistics_ready",
#             torch.zeros((1,), device=self.device, dtype=torch.bool),
#         )
#         append(
#             "phase_created",
#             torch.full((1,), -1, device=self.device, dtype=torch.long),
#         )
#         append(
#             "phase_updated",
#             torch.full((1,), -1, device=self.device, dtype=torch.long),
#         )
#         append(
#             "row_valid",
#             torch.zeros((1,), device=self.device, dtype=torch.bool),
#         )

#     def ensure_class_count(self, count: int) -> None:
#         while len(self) < int(count):
#             self._append_empty_row()

#     def _normalize_row(self, row: Mapping[str, Any]) -> Row:
#         required = (
#             "spectral_mean",
#             "spectral_basis",
#             "spectral_eigvals",
#             "spectral_res_var",
#             "spectral_active_rank",
#             "feature_mean",
#             "spectral_coupling",
#             "feature_basis",
#             "feature_eigvals",
#             "feature_res_var",
#             "feature_active_rank",
#             "sample_count",
#             "effective_sample_count",
#         )
#         missing = [name for name in required if row.get(name) is None]
#         if missing:
#             raise RuntimeError(f"row is missing required fields {missing}")

#         def tensor(name: str) -> Tensor:
#             return torch.as_tensor(
#                 row[name], device=self.device, dtype=self.dtype
#             )

#         spectral_mean = tensor("spectral_mean").flatten()
#         feature_mean = tensor("feature_mean").flatten()
#         spectral_basis = tensor("spectral_basis")
#         feature_basis = tensor("feature_basis")
#         spectral_eigvals = tensor("spectral_eigvals").flatten()
#         feature_eigvals = tensor("feature_eigvals").flatten()
#         coupling = tensor("spectral_coupling")
#         spectral_res = tensor("spectral_res_var").reshape(())
#         feature_res = tensor("feature_res_var").reshape(())
#         spectral_active = int(
#             torch.as_tensor(row["spectral_active_rank"]).item()
#         )
#         feature_active = int(
#             torch.as_tensor(row["feature_active_rank"]).item()
#         )
#         sample_count = float(torch.as_tensor(row["sample_count"]).item())
#         effective_count = float(
#             torch.as_tensor(row["effective_sample_count"]).item()
#         )

#         expected_shapes = {
#             "spectral_mean": (
#                 spectral_mean,
#                 (self.spectral_anchor_dim,),
#             ),
#             "feature_mean": (feature_mean, (self.d_model,)),
#             "spectral_basis": (
#                 spectral_basis,
#                 (self.spectral_anchor_dim, self.spectral_rank),
#             ),
#             "feature_basis": (
#                 feature_basis,
#                 (self.d_model, self.rank),
#             ),
#             "spectral_eigvals": (
#                 spectral_eigvals,
#                 (self.spectral_rank,),
#             ),
#             "feature_eigvals": (
#                 feature_eigvals,
#                 (self.rank,),
#             ),
#             "spectral_coupling": (
#                 coupling,
#                 (self.d_model, self.spectral_anchor_dim),
#             ),
#         }
#         for name, (value, shape) in expected_shapes.items():
#             if tuple(value.shape) != shape:
#                 raise RuntimeError(
#                     f"{name} shape {tuple(value.shape)} != {shape}"
#                 )

#         if not 0 <= spectral_active <= self.spectral_rank:
#             raise RuntimeError("invalid spectral_active_rank")
#         if not 0 <= feature_active <= self.rank:
#             raise RuntimeError("invalid feature_active_rank")
#         if sample_count <= 0.0 or effective_count <= 0.0:
#             raise RuntimeError("sample counts must be positive")
#         if effective_count > sample_count + 1e-3:
#             raise RuntimeError(
#                 "effective_sample_count cannot exceed sample_count"
#             )

#         if spectral_active > 0:
#             active = _orthonormalize(
#                 spectral_basis[:, :spectral_active]
#             )
#             spectral_basis = spectral_basis.clone()
#             spectral_basis[:, :spectral_active] = active
#         if spectral_active < self.spectral_rank:
#             spectral_basis = spectral_basis.clone()
#             spectral_basis[:, spectral_active:] = 0.0

#         if feature_active > 0:
#             active = _orthonormalize(
#                 feature_basis[:, :feature_active]
#             )
#             feature_basis = feature_basis.clone()
#             feature_basis[:, :feature_active] = active
#         if feature_active < self.rank:
#             feature_basis = feature_basis.clone()
#             feature_basis[:, feature_active:] = 0.0

#         spectral_res = spectral_res.clamp_min(
#             self.spectral_variance_floor
#         )
#         feature_res = feature_res.clamp_min(
#             self.feature_variance_floor
#         )
#         spectral_eigvals = spectral_eigvals.clamp_min(
#             self.spectral_variance_floor
#         )
#         feature_eigvals = feature_eigvals.clamp_min(
#             self.feature_variance_floor
#         )
#         if spectral_active < self.spectral_rank:
#             spectral_eigvals = spectral_eigvals.clone()
#             spectral_eigvals[spectral_active:] = spectral_res
#         if feature_active < self.rank:
#             feature_eigvals = feature_eigvals.clone()
#             feature_eigvals[feature_active:] = feature_res

#         finite = (
#             spectral_mean,
#             feature_mean,
#             spectral_basis,
#             feature_basis,
#             spectral_eigvals,
#             feature_eigvals,
#             coupling,
#             spectral_res,
#             feature_res,
#         )
#         if not all(torch.isfinite(value).all() for value in finite):
#             raise RuntimeError("row contains NaN/Inf")

#         def scalar(name: str, default: float) -> Tensor:
#             return torch.as_tensor(
#                 row.get(name, default),
#                 device=self.device,
#                 dtype=self.dtype,
#             ).reshape(())

#         return {
#             "spectral_mean": spectral_mean,
#             "spectral_basis": spectral_basis,
#             "spectral_eigvals": spectral_eigvals,
#             "spectral_res_var": spectral_res,
#             "spectral_active_rank": torch.tensor(
#                 spectral_active, device=self.device, dtype=torch.long
#             ),
#             "feature_mean": feature_mean,
#             "spectral_coupling": coupling,
#             "feature_basis": feature_basis,
#             "feature_eigvals": feature_eigvals,
#             "feature_res_var": feature_res,
#             "feature_active_rank": torch.tensor(
#                 feature_active, device=self.device, dtype=torch.long
#             ),
#             "sample_count": torch.tensor(
#                 sample_count, device=self.device, dtype=self.dtype
#             ),
#             "effective_sample_count": torch.tensor(
#                 effective_count, device=self.device, dtype=self.dtype
#             ),
#             "reliability": scalar("reliability", 0.0).clamp(0.0, 1.0),
#             "coupling_explained_variance": scalar(
#                 "coupling_explained_variance", 0.0
#             ).clamp(0.0, 1.0),
#             "outlier_rate": scalar(
#                 "outlier_rate", 0.0
#             ).clamp(0.0, 1.0),
#         }

#     def _write_row(
#         self,
#         class_id: int,
#         row: Mapping[str, Any],
#         *,
#         phase_created: int,
#         phase_updated: int,
#     ) -> None:
#         normalized = self._normalize_row(row)
#         self.ensure_class_count(class_id + 1)

#         assignments = {
#             "spectral_means": "spectral_mean",
#             "spectral_bases": "spectral_basis",
#             "spectral_eigvals": "spectral_eigvals",
#             "spectral_res_vars": "spectral_res_var",
#             "spectral_active_ranks": "spectral_active_rank",
#             "feature_means": "feature_mean",
#             "spectral_couplings": "spectral_coupling",
#             "feature_bases": "feature_basis",
#             "feature_eigvals": "feature_eigvals",
#             "feature_res_vars": "feature_res_var",
#             "feature_active_ranks": "feature_active_rank",
#             "sample_counts": "sample_count",
#             "effective_sample_counts": "effective_sample_count",
#             "reliability": "reliability",
#             "coupling_explained_variance":
#                 "coupling_explained_variance",
#             "outlier_rates": "outlier_rate",
#         }
#         for destination, source in assignments.items():
#             getattr(self, destination)[class_id] = normalized[source]
#         self.energy_quantiles[class_id].zero_()
#         self.margin_quantiles[class_id].zero_()
#         self.statistics_ready[class_id] = False
#         self.phase_created[class_id] = int(phase_created)
#         self.phase_updated[class_id] = int(phase_updated)
#         self.row_valid[class_id] = True

#     def _atomic_write(
#         self,
#         rows: Mapping[int, Mapping[str, Any]],
#         *,
#         phase: int,
#         replace_existing: bool,
#     ) -> None:
#         snapshot = self.export_snapshot()
#         try:
#             for class_id, row in rows.items():
#                 class_id = int(class_id)
#                 occupied = (
#                     class_id < len(self)
#                     and bool(self.row_valid[class_id].item())
#                 )
#                 if occupied and not replace_existing:
#                     raise RuntimeError(
#                         f"refusing to overwrite occupied row {class_id}"
#                     )
#                 if not occupied and replace_existing:
#                     raise RuntimeError(
#                         f"cannot transport absent row {class_id}"
#                     )
#                 created = (
#                     int(self.phase_created[class_id].item())
#                     if occupied
#                     else int(phase)
#                 )
#                 self._write_row(
#                     class_id,
#                     row,
#                     phase_created=created,
#                     phase_updated=int(phase),
#                 )
#             valid_ids = torch.nonzero(
#                 self.row_valid, as_tuple=False
#             ).flatten().tolist()
#             self.assert_valid(valid_ids, strict=True)
#         except Exception:
#             self.load_snapshot(snapshot, strict=True)
#             raise

#     @torch.no_grad()
#     def commit_base_rows(
#         self,
#         rows: Mapping[int, Mapping[str, Any]],
#         *,
#         base_class_ids: Sequence[int],
#         phase: int = 0,
#     ) -> Dict[str, Any]:
#         ids = _unique_ids(base_class_ids)
#         if set(rows) != set(ids):
#             raise RuntimeError("base row IDs do not match base_class_ids")
#         if bool(self.row_valid.any()):
#             raise RuntimeError("base rows can only be committed to an empty bank")
#         self._atomic_write(rows, phase=phase, replace_existing=False)
#         return {
#             "class_ids": ids,
#             "phase": int(phase),
#             "digest": self.rows_digest(ids),
#             "atomic": True,
#         }

#     @torch.no_grad()
#     def commit_incremental_phase(
#         self,
#         *,
#         transported_old_rows: Mapping[int, Mapping[str, Any]],
#         new_rows: Mapping[int, Mapping[str, Any]],
#         old_class_ids: Sequence[int],
#         new_class_ids: Sequence[int],
#         phase: int,
#         expected_old_digest: Optional[str] = None,
#     ) -> Dict[str, Any]:
#         old_ids = _unique_ids(old_class_ids)
#         new_ids = _unique_ids(new_class_ids)
#         if set(old_ids) & set(new_ids):
#             raise RuntimeError("old/new class IDs overlap")
#         if set(transported_old_rows) != set(old_ids):
#             raise RuntimeError("transported old row IDs mismatch")
#         if set(new_rows) != set(new_ids):
#             raise RuntimeError("new row IDs mismatch")
#         self.assert_valid(old_ids, strict=True)
#         digest_before = self.rows_digest(old_ids)
#         if expected_old_digest is not None and (
#             digest_before != str(expected_old_digest)
#         ):
#             raise RuntimeError("old-row digest mismatch before phase commit")

#         snapshot = self.export_snapshot()
#         try:
#             self._atomic_write(
#                 transported_old_rows,
#                 phase=phase,
#                 replace_existing=True,
#             )
#             self._atomic_write(
#                 new_rows,
#                 phase=phase,
#                 replace_existing=False,
#             )
#             self.assert_valid([*old_ids, *new_ids], strict=True)
#         except Exception:
#             self.load_snapshot(snapshot, strict=True)
#             raise
#         return {
#             "old_class_ids": old_ids,
#             "new_class_ids": new_ids,
#             "phase": int(phase),
#             "old_digest_before": digest_before,
#             "bank_digest_after": self.rows_digest([*old_ids, *new_ids]),
#             "atomic": True,
#         }

#     def valid_mask(self) -> Tensor:
#         if len(self) == 0:
#             return torch.empty(
#                 (0,), device=self.device, dtype=torch.bool
#             )
#         finite = (
#             torch.isfinite(self.spectral_means).all(dim=1)
#             & torch.isfinite(self.spectral_bases).flatten(1).all(dim=1)
#             & torch.isfinite(self.spectral_eigvals).all(dim=1)
#             & torch.isfinite(self.spectral_res_vars)
#             & torch.isfinite(self.feature_means).all(dim=1)
#             & torch.isfinite(self.spectral_couplings).flatten(1).all(dim=1)
#             & torch.isfinite(self.feature_bases).flatten(1).all(dim=1)
#             & torch.isfinite(self.feature_eigvals).all(dim=1)
#             & torch.isfinite(self.feature_res_vars)
#             & torch.isfinite(self.sample_counts)
#             & torch.isfinite(self.effective_sample_counts)
#         )
#         floors = (
#             self.spectral_eigvals.ge(self.spectral_variance_floor).all(dim=1)
#             & self.spectral_res_vars.ge(self.spectral_variance_floor)
#             & self.feature_eigvals.ge(self.feature_variance_floor).all(dim=1)
#             & self.feature_res_vars.ge(self.feature_variance_floor)
#         )
#         ranks = (
#             self.spectral_active_ranks.ge(0)
#             & self.spectral_active_ranks.le(self.spectral_rank)
#             & self.feature_active_ranks.ge(0)
#             & self.feature_active_ranks.le(self.rank)
#         )
#         counts = (
#             self.sample_counts.gt(0)
#             & self.effective_sample_counts.gt(0)
#             & self.effective_sample_counts.le(self.sample_counts + 1e-3)
#         )
#         return self.row_valid & finite & floors & ranks & counts

#     def assert_valid(
#         self,
#         class_ids: Optional[Iterable[int]] = None,
#         *,
#         strict: bool = True,
#     ) -> Dict[str, Any]:
#         names = set(self.__dict__) | set(self._buffers) | set(self._parameters)
#         forbidden = sorted(
#             name for name in names
#             if name.lower() in _FORBIDDEN_MEMORY_NAMES
#         )
#         errors: list[str] = []
#         if forbidden:
#             errors.append(f"forbidden exemplar fields: {forbidden}")

#         ids = (
#             list(range(len(self)))
#             if class_ids is None
#             else _unique_ids(class_ids)
#         )
#         valid = self.valid_mask()
#         for class_id in ids:
#             if class_id >= len(self) or not bool(valid[class_id].item()):
#                 errors.append(f"class {class_id}: invalid or absent row")
#                 continue
#             spectral_rank = int(self.spectral_active_ranks[class_id].item())
#             feature_rank = int(self.feature_active_ranks[class_id].item())
#             if spectral_rank > 0:
#                 basis = self.spectral_bases[
#                     class_id, :, :spectral_rank
#                 ]
#                 gram = basis.transpose(0, 1) @ basis
#                 identity = torch.eye(
#                     spectral_rank, device=self.device, dtype=self.dtype
#                 )
#                 if not torch.allclose(
#                     gram, identity, atol=1e-3, rtol=0.0
#                 ):
#                     errors.append(
#                         f"class {class_id}: spectral basis is not orthonormal"
#                     )
#             if feature_rank > 0:
#                 basis = self.feature_bases[
#                     class_id, :, :feature_rank
#                 ]
#                 gram = basis.transpose(0, 1) @ basis
#                 identity = torch.eye(
#                     feature_rank, device=self.device, dtype=self.dtype
#                 )
#                 if not torch.allclose(
#                     gram, identity, atol=1e-3, rtol=0.0
#                 ):
#                     errors.append(
#                         f"class {class_id}: feature basis is not orthonormal"
#                     )

#         report = {
#             "ok": not errors,
#             "class_ids": ids,
#             "valid_rows": int(valid.sum().item()),
#             "errors": errors,
#         }
#         if strict and errors:
#             raise RuntimeError(
#                 "SpectralConditionedGeometryBank invalid: "
#                 + "; ".join(errors)
#             )
#         return report

#     def _row_from_buffers(self, class_id: int) -> Row:
#         if class_id >= len(self) or not bool(self.valid_mask()[class_id]):
#             raise RuntimeError(f"class {class_id} has no valid row")
#         return {
#             "spectral_mean": self.spectral_means[class_id],
#             "spectral_basis": self.spectral_bases[class_id],
#             "spectral_eigvals": self.spectral_eigvals[class_id],
#             "spectral_res_var": self.spectral_res_vars[class_id],
#             "spectral_active_rank": self.spectral_active_ranks[class_id],
#             "feature_mean": self.feature_means[class_id],
#             "spectral_coupling": self.spectral_couplings[class_id],
#             "feature_basis": self.feature_bases[class_id],
#             "feature_eigvals": self.feature_eigvals[class_id],
#             "feature_res_var": self.feature_res_vars[class_id],
#             "feature_active_rank": self.feature_active_ranks[class_id],
#             "sample_count": self.sample_counts[class_id],
#             "effective_sample_count": self.effective_sample_counts[class_id],
#             "reliability": self.reliability[class_id],
#             "coupling_explained_variance":
#                 self.coupling_explained_variance[class_id],
#             "outlier_rate": self.outlier_rates[class_id],
#         }

#     def get_class_row(self, class_id: int, *, clone: bool = True) -> Row:
#         row = self._row_from_buffers(int(class_id))
#         if clone:
#             return {
#                 name: value.detach().clone()
#                 for name, value in row.items()
#             }
#         return row

#     def get_bank(self, class_ids: Optional[Iterable[int]] = None) -> Dict[str, Tensor]:
#         ids = (
#             torch.nonzero(
#                 self.valid_mask(), as_tuple=False
#             ).flatten().tolist()
#             if class_ids is None
#             else _unique_ids(class_ids)
#         )
#         self.assert_valid(ids, strict=True)
#         index = torch.tensor(ids, device=self.device, dtype=torch.long)
#         return {
#             "class_ids": index,
#             "spectral_means": self.spectral_means.index_select(0, index),
#             "spectral_bases": self.spectral_bases.index_select(0, index),
#             "spectral_eigvals": self.spectral_eigvals.index_select(0, index),
#             "spectral_res_vars": self.spectral_res_vars.index_select(0, index),
#             "spectral_active_ranks":
#                 self.spectral_active_ranks.index_select(0, index),
#             "feature_means": self.feature_means.index_select(0, index),
#             "spectral_couplings":
#                 self.spectral_couplings.index_select(0, index),
#             "feature_bases": self.feature_bases.index_select(0, index),
#             "feature_eigvals": self.feature_eigvals.index_select(0, index),
#             "feature_res_vars": self.feature_res_vars.index_select(0, index),
#             "feature_active_ranks":
#                 self.feature_active_ranks.index_select(0, index),
#             "sample_counts": self.sample_counts.index_select(0, index),
#             "effective_sample_counts":
#                 self.effective_sample_counts.index_select(0, index),
#             "reliability": self.reliability.index_select(0, index),
#             "coupling_explained_variance":
#                 self.coupling_explained_variance.index_select(0, index),
#             "outlier_rates": self.outlier_rates.index_select(0, index),
#             "energy_quantiles": self.energy_quantiles.index_select(0, index),
#             "margin_quantiles": self.margin_quantiles.index_select(0, index),
#             "statistics_ready": self.statistics_ready.index_select(0, index),
#             "phase_created": self.phase_created.index_select(0, index),
#             "phase_updated": self.phase_updated.index_select(0, index),
#         }

#     def _stack_rows(
#         self,
#         class_ids: Sequence[int],
#         rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#     ) -> Dict[str, Tensor]:
#         ids = _unique_ids(class_ids)
#         if not ids:
#             raise ValueError("class_ids cannot be empty")
#         normalized_rows: list[Row] = []
#         for class_id in ids:
#             if rows is not None and class_id in rows:
#                 normalized_rows.append(self._normalize_row(rows[class_id]))
#             else:
#                 normalized_rows.append(
#                     self.get_class_row(class_id, clone=False)
#                 )

#         fields = (
#             "spectral_mean",
#             "spectral_basis",
#             "spectral_eigvals",
#             "spectral_res_var",
#             "spectral_active_rank",
#             "feature_mean",
#             "spectral_coupling",
#             "feature_basis",
#             "feature_eigvals",
#             "feature_res_var",
#             "feature_active_rank",
#         )
#         stacked = {
#             field: torch.stack([row[field] for row in normalized_rows])
#             for field in fields
#         }
#         stacked["class_ids"] = torch.tensor(
#             ids, device=self.device, dtype=torch.long
#         )
#         stacked["valid_mask"] = torch.ones(
#             len(ids), device=self.device, dtype=torch.bool
#         )
#         return stacked

#     def joint_energy_matrix(
#         self,
#         features: Tensor,
#         class_ids: Sequence[int],
#         *,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         return_parts: bool = False,
#     ) -> Union[Tensor, Dict[str, Tensor]]:
#         x = self._canonicalize_features(features)
#         s = self._resolve_anchors(
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#         )
#         if x.size(0) != s.size(0):
#             raise ValueError("features and spectra are not aligned")
#         components = self._stack_rows(class_ids, rows)

#         spectral_delta = (
#             s[:, None, :] - components["spectral_mean"][None, :, :]
#         )
#         spectral_parts = _low_rank_energy_from_delta(
#             spectral_delta,
#             bases=components["spectral_basis"],
#             eigvals=components["spectral_eigvals"],
#             residual_variances=components["spectral_res_var"],
#             active_ranks=components["spectral_active_rank"],
#             variance_floor=self.spectral_variance_floor,
#             logdet_weight=self.logdet_weight,
#             normalize_by_dimension=True,
#             valid_mask=components["valid_mask"],
#         )

#         conditional_shift = torch.einsum(
#             "ncs,cds->ncd",
#             spectral_delta,
#             components["spectral_coupling"],
#         )
#         conditional_mean = (
#             components["feature_mean"][None, :, :]
#             + conditional_shift
#         )
#         feature_delta = x[:, None, :] - conditional_mean
#         feature_parts = _low_rank_energy_from_delta(
#             feature_delta,
#             bases=components["feature_basis"],
#             eigvals=components["feature_eigvals"],
#             residual_variances=components["feature_res_var"],
#             active_ranks=components["feature_active_rank"],
#             variance_floor=self.feature_variance_floor,
#             logdet_weight=self.logdet_weight,
#             normalize_by_dimension=True,
#             valid_mask=components["valid_mask"],
#         )

#         joint = (
#             self.spectral_weight * spectral_parts["energy"]
#             + self.feature_weight * feature_parts["energy"]
#         )
#         if not return_parts:
#             return joint
#         return {
#             "energy": joint,
#             "spectral_energy": spectral_parts["energy"],
#             "feature_energy": feature_parts["energy"],
#             "spectral_quadratic": spectral_parts["quadratic"],
#             "feature_quadratic": feature_parts["quadratic"],
#             "spectral_volume": spectral_parts["volume"],
#             "feature_volume": feature_parts["volume"],
#             "conditional_feature_mean": conditional_mean,
#         }

#     def predict(
#         self,
#         features: Tensor,
#         class_ids: Sequence[int],
#         *,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#     ) -> Tensor:
#         energy = self.joint_energy_matrix(
#             features,
#             class_ids,
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#             rows=rows,
#         )
#         ids = torch.tensor(
#             _unique_ids(class_ids), device=self.device, dtype=torch.long
#         )
#         return ids.index_select(0, energy.argmin(dim=1))

#     def _sample_low_rank(
#         self,
#         mean: Tensor,
#         basis: Tensor,
#         eigvals: Tensor,
#         residual_variance: Tensor,
#         active_rank: int,
#         count: int,
#         generator: Optional[torch.Generator],
#     ) -> Tensor:
#         dimension = mean.numel()
#         result = mean.view(1, -1).expand(count, -1).clone()
#         if active_rank > 0:
#             coefficients = torch.randn(
#                 (count, active_rank),
#                 device=self.device,
#                 dtype=self.dtype,
#                 generator=generator,
#             )
#             coefficients = coefficients * eigvals[
#                 :active_rank
#             ].sqrt().view(1, -1)
#             active_basis = basis[:, :active_rank]
#             result = result + coefficients @ active_basis.transpose(0, 1)
#         residual_noise = torch.randn(
#             (count, dimension),
#             device=self.device,
#             dtype=self.dtype,
#             generator=generator,
#         )
#         if active_rank > 0:
#             active_basis = basis[:, :active_rank]
#             residual_noise = residual_noise - (
#                 residual_noise @ active_basis
#             ) @ active_basis.transpose(0, 1)
#         result = result + residual_variance.sqrt() * residual_noise
#         return result

#     @torch.no_grad()
#     def sample_replay(
#         self,
#         class_ids: Sequence[int],
#         samples_per_class: int,
#         *,
#         rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         generator: Optional[torch.Generator] = None,
#     ) -> Dict[str, Tensor]:
#         ids = _unique_ids(class_ids)
#         count = int(samples_per_class)
#         if count <= 0:
#             raise ValueError("samples_per_class must be positive")

#         features: list[Tensor] = []
#         anchors: list[Tensor] = []
#         labels: list[Tensor] = []
#         for class_id in ids:
#             row = (
#                 self._normalize_row(rows[class_id])
#                 if rows is not None and class_id in rows
#                 else self.get_class_row(class_id, clone=False)
#             )
#             spectral_rank = int(row["spectral_active_rank"].item())
#             feature_rank = int(row["feature_active_rank"].item())
#             sampled_s = self._sample_low_rank(
#                 row["spectral_mean"],
#                 row["spectral_basis"],
#                 row["spectral_eigvals"],
#                 row["spectral_res_var"],
#                 spectral_rank,
#                 count,
#                 generator,
#             )
#             feature_residual = self._sample_low_rank(
#                 torch.zeros(
#                     self.d_model, device=self.device, dtype=self.dtype
#                 ),
#                 row["feature_basis"],
#                 row["feature_eigvals"],
#                 row["feature_res_var"],
#                 feature_rank,
#                 count,
#                 generator,
#             )
#             conditional_mean = (
#                 row["feature_mean"].view(1, -1)
#                 + (
#                     sampled_s - row["spectral_mean"].view(1, -1)
#                 ) @ row["spectral_coupling"].transpose(0, 1)
#             )
#             features.append(conditional_mean + feature_residual)
#             anchors.append(sampled_s)
#             labels.append(
#                 torch.full(
#                     (count,),
#                     class_id,
#                     device=self.device,
#                     dtype=torch.long,
#                 )
#             )

#         return {
#             "features": torch.cat(features, dim=0),
#             "spectral_anchors": torch.cat(anchors, dim=0),
#             "labels": torch.cat(labels, dim=0),
#         }

#     def _validate_transport(
#         self,
#         matrix: Tensor,
#         spectral_shift: Tensor,
#         bias: Tensor,
#     ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
#         matrix = torch.as_tensor(
#             matrix, device=self.device, dtype=self.dtype
#         )
#         spectral_shift = torch.as_tensor(
#             spectral_shift, device=self.device, dtype=self.dtype
#         )
#         bias = torch.as_tensor(
#             bias, device=self.device, dtype=self.dtype
#         ).flatten()
#         if matrix.shape != (self.d_model, self.d_model):
#             raise ValueError("transport matrix must be [D,D]")
#         if spectral_shift.shape != (
#             self.d_model, self.spectral_anchor_dim
#         ):
#             raise ValueError("spectral_shift must be [D,S]")
#         if bias.shape != (self.d_model,):
#             raise ValueError("transport bias must be [D]")
#         if not all(
#             torch.isfinite(value).all()
#             for value in (matrix, spectral_shift, bias)
#         ):
#             raise RuntimeError("transport contains NaN/Inf")

#         singular = torch.linalg.svdvals(matrix)
#         minimum = float(singular.min().item())
#         maximum = float(singular.max().item())
#         condition = maximum / max(minimum, _EPS)
#         if minimum <= _EPS:
#             raise RuntimeError("transport matrix is singular")
#         if condition > self.maximum_transport_condition:
#             raise RuntimeError(
#                 "transport condition number "
#                 f"{condition:.4f} exceeds "
#                 f"{self.maximum_transport_condition:.4f}"
#             )
#         return matrix, spectral_shift, bias, {
#             "minimum_singular_value": minimum,
#             "maximum_singular_value": maximum,
#             "condition_number": condition,
#         }

#     def apply_transport(
#         self,
#         features: Tensor,
#         spectral_anchors: Tensor,
#         *,
#         matrix: Tensor,
#         spectral_shift: Tensor,
#         bias: Tensor,
#     ) -> Tensor:
#         x = self._canonicalize_features(features)
#         s = self._canonicalize_anchors(spectral_anchors)
#         if x.size(0) != s.size(0):
#             raise ValueError("features and anchors are not aligned")
#         matrix, spectral_shift, bias, _ = self._validate_transport(
#             matrix, spectral_shift, bias
#         )
#         return (
#             x @ matrix.transpose(0, 1)
#             + s @ spectral_shift.transpose(0, 1)
#             + bias.view(1, -1)
#         )

#     @torch.no_grad()
#     def build_transported_rows(
#         self,
#         class_ids: Sequence[int],
#         *,
#         matrix: Tensor,
#         spectral_shift: Tensor,
#         bias: Tensor,
#     ) -> Tuple[Dict[int, Row], Dict[str, Any]]:
#         ids = _unique_ids(class_ids)
#         self.assert_valid(ids, strict=True)
#         matrix, spectral_shift, bias, transport_report = (
#             self._validate_transport(matrix, spectral_shift, bias)
#         )

#         transported: Dict[int, Row] = {}
#         errors: list[float] = []
#         for class_id in ids:
#             row = self.get_class_row(class_id, clone=False)
#             spectral_mean = row["spectral_mean"]
#             feature_mean = (
#                 matrix @ row["feature_mean"]
#                 + spectral_shift @ spectral_mean
#                 + bias
#             )
#             coupling = (
#                 matrix @ row["spectral_coupling"]
#                 + spectral_shift
#             )
#             old_covariance = _low_rank_covariance(
#                 row["feature_basis"],
#                 row["feature_eigvals"],
#                 row["feature_res_var"],
#                 int(row["feature_active_rank"].item()),
#                 self.d_model,
#             )
#             new_covariance = _symmetrize(
#                 matrix @ old_covariance @ matrix.transpose(0, 1)
#             )
#             old_rank = int(row["feature_active_rank"].item())
#             identity = torch.eye(
#                 self.d_model, device=self.device, dtype=self.dtype
#             )
#             orthogonal = torch.allclose(
#                 matrix.transpose(0, 1) @ matrix,
#                 identity,
#                 atol=1e-5,
#                 rtol=1e-5,
#             )
#             if orthogonal:
#                 transformed_basis = torch.zeros_like(row["feature_basis"])
#                 if old_rank > 0:
#                     transformed_basis[:, :old_rank] = _orthonormalize(
#                         matrix @ row["feature_basis"][:, :old_rank]
#                     )
#                 factor = {
#                     "basis": transformed_basis,
#                     "eigvals": row["feature_eigvals"].detach().clone(),
#                     "res_var": row["feature_res_var"].detach().clone(),
#                     "active_rank": row[
#                         "feature_active_rank"
#                     ].detach().clone(),
#                     "approximation_error": row[
#                         "feature_res_var"
#                     ].new_tensor(0.0),
#                 }
#             else:
#                 factor = _factorize_covariance(
#                     new_covariance,
#                     maximum_rank=self.rank,
#                     variance_floor=self.feature_variance_floor,
#                     retained_energy=self.retained_energy,
#                     noise_ratio=self.noise_ratio,
#                     minimum_rank=old_rank,
#                 )
#             errors.append(float(factor["approximation_error"].item()))
#             transported[class_id] = {
#                 "spectral_mean": row["spectral_mean"].detach().clone(),
#                 "spectral_basis": row["spectral_basis"].detach().clone(),
#                 "spectral_eigvals": row[
#                     "spectral_eigvals"
#                 ].detach().clone(),
#                 "spectral_res_var": row[
#                     "spectral_res_var"
#                 ].detach().clone(),
#                 "spectral_active_rank": row[
#                     "spectral_active_rank"
#                 ].detach().clone(),
#                 "feature_mean": feature_mean.detach(),
#                 "spectral_coupling": coupling.detach(),
#                 "feature_basis": factor["basis"].detach(),
#                 "feature_eigvals": factor["eigvals"].detach(),
#                 "feature_res_var": factor["res_var"].detach(),
#                 "feature_active_rank": factor[
#                     "active_rank"
#                 ].detach(),
#                 "sample_count": row["sample_count"].detach().clone(),
#                 "effective_sample_count": row[
#                     "effective_sample_count"
#                 ].detach().clone(),
#                 "reliability": row["reliability"].detach().clone(),
#                 "coupling_explained_variance": row[
#                     "coupling_explained_variance"
#                 ].detach().clone(),
#                 "outlier_rate": row["outlier_rate"].detach().clone(),
#             }

#         transport_report.update(
#             {
#                 "class_ids": ids,
#                 "mean_covariance_approximation_error":
#                     float(sum(errors) / max(len(errors), 1)),
#                 "maximum_covariance_approximation_error":
#                     float(max(errors) if errors else 0.0),
#             }
#         )
#         return transported, transport_report

#     @torch.no_grad()
#     def sample_boundary_replay(
#         self,
#         source_class_ids: Sequence[int],
#         scoring_class_ids: Sequence[int],
#         samples_per_class: int,
#         *,
#         source_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         scoring_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         oversample_factor: Optional[int] = None,
#         typical_quantile: float = 0.95,
#         generator: Optional[torch.Generator] = None,
#     ) -> Dict[str, Tensor]:
#         source_ids = _unique_ids(source_class_ids)
#         scoring_ids = _unique_ids(scoring_class_ids)
#         if not set(source_ids).issubset(scoring_ids):
#             raise ValueError(
#                 "source classes must be included in scoring classes"
#             )
#         factor = (
#             self.boundary_oversample_factor
#             if oversample_factor is None
#             else int(max(2, oversample_factor))
#         )
#         count = int(samples_per_class)
#         pool = self.sample_replay(
#             source_ids,
#             count * factor,
#             rows=source_rows,
#             generator=generator,
#         )
#         energy = self.joint_energy_matrix(
#             pool["features"],
#             scoring_ids,
#             spectral_anchors=pool["spectral_anchors"],
#             rows=scoring_rows,
#         )
#         scoring_tensor = torch.tensor(
#             scoring_ids, device=self.device, dtype=torch.long
#         )
#         scoring_position = {
#             class_id: index for index, class_id in enumerate(scoring_ids)
#         }

#         selected_indices: list[Tensor] = []
#         selected_rivals: list[Tensor] = []
#         selected_margins: list[Tensor] = []
#         selected_typicality: list[Tensor] = []

#         for class_id in source_ids:
#             pool_indices = torch.nonzero(
#                 pool["labels"].eq(class_id), as_tuple=False
#             ).flatten()
#             class_energy = energy.index_select(0, pool_indices)
#             true_position = scoring_position[class_id]
#             true_energy = class_energy[:, true_position]
#             rival_energy = class_energy.clone()
#             rival_energy[:, true_position] = float("inf")
#             nearest, rival_position = rival_energy.min(dim=1)
#             margin = nearest - true_energy

#             cutoff = torch.quantile(
#                 true_energy, float(min(max(typical_quantile, 0.5), 0.999))
#             )
#             typical = true_energy.le(cutoff)
#             positive_boundary = typical & margin.ge(0.0)
#             candidate_order = torch.argsort(margin)
#             preferred = candidate_order[
#                 positive_boundary.index_select(0, candidate_order)
#             ]
#             fallback_mask = typical & ~positive_boundary
#             fallback = candidate_order[
#                 fallback_mask.index_select(0, candidate_order)
#             ]
#             combined = torch.cat([preferred, fallback], dim=0)
#             chosen_local = combined[: min(count, combined.numel())]
#             if chosen_local.numel() < count:
#                 raise RuntimeError(
#                     f"insufficient typical replay for class {class_id}"
#                 )
#             chosen_global = pool_indices.index_select(0, chosen_local)
#             selected_indices.append(chosen_global)
#             selected_rivals.append(
#                 scoring_tensor.index_select(
#                     0, rival_position.index_select(0, chosen_local)
#                 )
#             )
#             selected_margins.append(
#                 margin.index_select(0, chosen_local)
#             )
#             selected_typicality.append(
#                 true_energy.index_select(0, chosen_local)
#             )

#         chosen = torch.cat(selected_indices, dim=0)
#         return {
#             "features": pool["features"].index_select(0, chosen),
#             "spectral_anchors": pool[
#                 "spectral_anchors"
#             ].index_select(0, chosen),
#             "labels": pool["labels"].index_select(0, chosen),
#             "rival_labels": torch.cat(selected_rivals, dim=0),
#             "margins": torch.cat(selected_margins, dim=0),
#             "true_energy": torch.cat(selected_typicality, dim=0),
#         }

#     @torch.no_grad()
#     def update_statistics(
#         self,
#         features: Tensor,
#         labels: Tensor,
#         class_ids: Sequence[int],
#         *,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#     ) -> Dict[str, float]:
#         ids = _unique_ids(class_ids)
#         x = self._canonicalize_features(features)
#         s = self._resolve_anchors(
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#         )
#         y = torch.as_tensor(
#             labels, device=self.device, dtype=torch.long
#         ).flatten()
#         if y.numel() != x.size(0) or s.size(0) != x.size(0):
#             raise ValueError("features, spectra, and labels are not aligned")
#         mapping = {class_id: index for index, class_id in enumerate(ids)}
#         if any(int(value) not in mapping for value in y.tolist()):
#             raise ValueError("labels contain classes outside class_ids")
#         targets = torch.tensor(
#             [mapping[int(value)] for value in y.tolist()],
#             device=self.device,
#             dtype=torch.long,
#         )
#         energy = self.joint_energy_matrix(
#             x,
#             ids,
#             spectral_anchors=s,
#             rows=rows,
#         )
#         true_energy = energy.gather(1, targets[:, None]).squeeze(1)
#         rivals = energy.clone()
#         rivals.scatter_(1, targets[:, None], float("inf"))
#         margin = rivals.min(dim=1).values - true_energy
#         prediction = energy.argmin(dim=1)

#         for local_id, class_id in enumerate(ids):
#             if rows is not None and class_id in rows:
#                 continue
#             mask = targets.eq(local_id)
#             if not bool(mask.any()):
#                 continue
#             self.energy_quantiles[class_id] = torch.quantile(
#                 true_energy[mask],
#                 torch.tensor(
#                     [0.50, 0.95, 0.99],
#                     device=self.device,
#                     dtype=self.dtype,
#                 ),
#             )
#             self.margin_quantiles[class_id] = torch.quantile(
#                 margin[mask],
#                 torch.tensor(
#                     [0.01, 0.05, 0.50],
#                     device=self.device,
#                     dtype=self.dtype,
#                 ),
#             )
#             self.statistics_ready[class_id] = True

#         return {
#             "accuracy": float(
#                 prediction.eq(targets).float().mean().item()
#             ),
#             "margin_mean": float(margin.mean().item()),
#             "margin_q05": float(torch.quantile(margin, 0.05).item()),
#             "violation_rate": float(margin.lt(0).float().mean().item()),
#         }

#     def effective_dimension(
#         self, class_ids: Optional[Sequence[int]] = None
#     ) -> Tensor:
#         ids = (
#             torch.nonzero(
#                 self.valid_mask(), as_tuple=False
#             ).flatten().tolist()
#             if class_ids is None
#             else _unique_ids(class_ids)
#         )
#         values: list[Tensor] = []
#         for class_id in ids:
#             row = self.get_class_row(class_id, clone=False)
#             rank = int(row["feature_active_rank"].item())
#             trace = (
#                 row["feature_eigvals"][:rank].sum()
#                 + (self.d_model - rank) * row["feature_res_var"]
#             )
#             square_trace = (
#                 row["feature_eigvals"][:rank].square().sum()
#                 + (self.d_model - rank)
#                 * row["feature_res_var"].square()
#             )
#             values.append(
#                 trace.square() / square_trace.clamp_min(_EPS)
#             )
#         return (
#             torch.stack(values)
#             if values
#             else torch.empty(
#                 (0,), device=self.device, dtype=self.dtype
#             )
#         )

#     @torch.no_grad()
#     def admission_report(
#         self,
#         class_ids: Sequence[int],
#         *,
#         minimum_effective_dimension: float = 1.25,
#         require_statistics: bool = True,
#     ) -> Dict[str, Any]:
#         ids = _unique_ids(class_ids)
#         validity = self.assert_valid(ids, strict=False)
#         errors = list(validity["errors"])
#         dimensions = self.effective_dimension(ids)
#         minimum_dimension = (
#             float(dimensions.min().item()) if dimensions.numel() else 0.0
#         )
#         if minimum_dimension < float(minimum_effective_dimension):
#             errors.append("effective feature dimension is below threshold")
#         if require_statistics:
#             missing = [
#                 class_id for class_id in ids
#                 if class_id >= len(self)
#                 or not bool(self.statistics_ready[class_id].item())
#             ]
#             if missing:
#                 errors.append(
#                     f"energy/margin statistics are absent for {missing}"
#                 )
#         minimum_q05 = (
#             float(
#                 self.margin_quantiles[
#                     torch.tensor(ids, device=self.device), 1
#                 ].min().item()
#             )
#             if ids and all(
#                 bool(self.statistics_ready[class_id].item())
#                 for class_id in ids
#             )
#             else float("nan")
#         )
#         return {
#             "ok": not errors,
#             "errors": errors,
#             "class_ids": ids,
#             "minimum_effective_dimension": minimum_dimension,
#             "minimum_margin_q05": minimum_q05,
#         }

#     @staticmethod
#     def _digest_tensor(hasher: "hashlib._Hash", tensor: Tensor) -> None:
#         value = tensor.detach().to("cpu").contiguous()
#         hasher.update(str(value.dtype).encode("utf-8"))
#         hasher.update(str(tuple(value.shape)).encode("utf-8"))
#         hasher.update(value.numpy().tobytes())

#     def rows_digest(self, class_ids: Sequence[int]) -> str:
#         ids = _unique_ids(class_ids)
#         self.assert_valid(ids, strict=True)
#         hasher = hashlib.sha256()
#         hasher.update(str(self.SCHEMA_VERSION).encode("utf-8"))
#         for class_id in ids:
#             hasher.update(str(class_id).encode("utf-8"))
#             row = self.get_class_row(class_id, clone=False)
#             for name in sorted(row):
#                 hasher.update(name.encode("utf-8"))
#                 self._digest_tensor(hasher, row[name])
#         return hasher.hexdigest()

#     def export_snapshot(self) -> Dict[str, Tensor]:
#         return {
#             name: value.detach().clone()
#             for name, value in self._buffers.items()
#             if value is not None
#         }

#     @torch.no_grad()
#     def load_snapshot(
#         self,
#         snapshot: Mapping[str, Tensor],
#         *,
#         strict: bool = True,
#     ) -> None:
#         current = set(self._buffers)
#         supplied = set(snapshot)
#         if strict and current != supplied:
#             missing = sorted(current - supplied)
#             extra = sorted(supplied - current)
#             raise RuntimeError(
#                 f"snapshot mismatch: missing={missing}, extra={extra}"
#             )
#         for name in current & supplied:
#             value = torch.as_tensor(snapshot[name], device=self.device)
#             target = self._buffers[name]
#             if target is not None:
#                 value = value.to(dtype=target.dtype)
#             setattr(self, name, value.detach().clone())

#     @torch.no_grad()
#     def reset_rows(self) -> None:
#         anchor_state = {
#             name: getattr(self, name).detach().clone()
#             for name in (
#                 "anchor_band_mean",
#                 "anchor_band_std",
#                 "anchor_basis",
#                 "anchor_scales",
#                 "anchor_active_dim",
#                 "anchor_sample_count",
#                 "anchor_ready",
#                 "anchor_frozen",
#             )
#         }
#         device, dtype = self.device, self.dtype
#         row_names = [
#             name for name in self._buffers
#             if not name.startswith("anchor_")
#         ]
#         for name in row_names:
#             del self._buffers[name]
#         self._register_row_buffers(device, dtype)
#         for name, value in anchor_state.items():
#             getattr(self, name).copy_(value)

#     def memory_cost_summary(self) -> Dict[str, Any]:
#         persistent = sum(
#             value.numel() * value.element_size()
#             for value in self._buffers.values()
#             if value is not None
#         )
#         return {
#             "schema_version": self.SCHEMA_VERSION,
#             "classes": int(self.valid_mask().sum().item()),
#             "bytes": int(persistent),
#             "megabytes": float(persistent / (1024.0 ** 2)),
#             "stores_sample_level_memory": False,
#             "factorization": "p(s|c)p(z|s,c)",
#             "spectral_anchor_dim": self.spectral_anchor_dim,
#             "feature_dim": self.d_model,
#         }


# GeometryBank = SpectralConditionedGeometryBank
# JointGeometryBank = SpectralConditionedGeometryBank
 