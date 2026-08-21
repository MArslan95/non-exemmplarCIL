from __future__ import annotations

"""Spectral-primary Mamba backbone for single-space HSI geometry learning.

The backbone has three responsibilities:

1. learn ordered spectral evidence from the normalized full-band center spectrum;
2. extract center-relative neighborhood context from the HSI patch;
3. use that context only as a residual refinement of the spectral feature and
   expose that final HSI representation directly as the geometry coordinates.

The center spectrum is the semantic anchor because the patch label belongs to
the center pixel. Spatial information does not create a second class feature
space. It models how the neighborhood differs from the center and learns a
residual correction in the same feature space.

The backbone contains no class state, replay, prototype, classifier, geometry
boundary, transport, or incremental-phase update logic. The output coordinate
tensor is the canonical representation consumed directly by the support bank.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _positive_int(value: object, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a positive integer")
        value = value.item()
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


def _spectral_positions(
    bands: int,
    coordinates: Optional[Sequence[float]],
) -> Tensor:
    """Return normalized band positions without fabricating wavelengths."""
    if coordinates is None:
        values = torch.arange(bands, dtype=torch.float32)
    else:
        values = torch.as_tensor(coordinates, dtype=torch.float32).flatten()
        if values.numel() != bands:
            raise ValueError(
                f"spectral coordinates have {values.numel()} values; expected {bands}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("spectral coordinates contain NaN or Inf")
        if bool((values[1:] <= values[:-1]).any()):
            raise ValueError("spectral coordinates must be strictly increasing")
    if bands == 1:
        return torch.zeros(1, dtype=torch.float32)
    span = values[-1] - values[0]
    if not bool(torch.isfinite(span)) or not bool(span > 0):
        raise ValueError("spectral coordinates must have a finite positive span")
    return 2.0 * (values - values[0]) / span - 1.0


def _spatial_positions(patch_size: int) -> Tensor:
    axis = torch.linspace(-1.0, 1.0, patch_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def _serpentine_orders(patch_size: int) -> Tensor:
    """Horizontal/vertical forward/reverse locality-preserving scan orders."""
    grid = torch.arange(patch_size * patch_size).reshape(patch_size, patch_size)
    horizontal = torch.cat(
        [grid[row] if row % 2 == 0 else grid[row].flip(0) for row in range(patch_size)]
    )
    vertical = torch.cat(
        [
            grid[:, col] if col % 2 == 0 else grid[:, col].flip(0)
            for col in range(patch_size)
        ]
    )
    return torch.stack(
        (horizontal, horizontal.flip(0), vertical, vertical.flip(0)),
        dim=0,
    )


class FrozenBandStandardizer(nn.Module):
    """Persistent phase-0 per-band input normalization.

    This stores only aggregate preprocessing statistics. It is not a class
    representation, descriptor, replay buffer, or incremental-learning state.
    """

    def __init__(self, bands: int) -> None:
        super().__init__()
        self.bands = _positive_int(bands, "spectral_bands")
        self.register_buffer("mean", torch.zeros(self.bands, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(self.bands, dtype=torch.float32))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def set_statistics(
        self,
        mean: Tensor | Sequence[float],
        scale: Tensor | Sequence[float],
        *,
        overwrite: bool = False,
    ) -> None:
        if bool(self.fitted.item()) and not overwrite:
            raise RuntimeError("spectral normalization is already fitted")
        mean_tensor = torch.as_tensor(
            mean,
            device=self.mean.device,
            dtype=torch.float32,
        ).flatten()
        scale_tensor = torch.as_tensor(
            scale,
            device=self.scale.device,
            dtype=torch.float32,
        ).flatten()
        if mean_tensor.numel() != self.bands or scale_tensor.numel() != self.bands:
            raise ValueError(
                f"spectral normalization statistics must have {self.bands} values"
            )
        if not bool(torch.isfinite(mean_tensor).all()) or not bool(
            torch.isfinite(scale_tensor).all()
        ):
            raise ValueError("spectral normalization statistics contain NaN or Inf")
        if bool((scale_tensor <= 0).any()):
            raise ValueError("spectral normalization scales must be positive")
        self.mean.copy_(mean_tensor)
        self.scale.copy_(scale_tensor)
        self.fitted.fill_(True)

    @torch.no_grad()
    def fit(self, spectra: Tensor, *, overwrite: bool = False) -> None:
        values = torch.as_tensor(spectra)
        if values.ndim != 2 or values.size(0) == 0 or values.size(1) != self.bands:
            raise ValueError(f"base spectra must be non-empty [N,{self.bands}]")
        if not torch.is_floating_point(values) or not bool(torch.isfinite(values).all()):
            raise ValueError("base spectra must contain finite floating-point values")
        values = values.to(dtype=torch.float32)
        mean = values.mean(dim=0)
        standard_deviation = values.std(dim=0, unbiased=False)
        # A constant band carries no standardized variation. Unit scale keeps
        # its centered value at zero without inventing variance.
        scale = torch.where(
            standard_deviation > 0,
            standard_deviation,
            torch.ones_like(standard_deviation),
        )
        self.set_statistics(mean, scale, overwrite=overwrite)

    def forward(self, spectrum: Tensor) -> Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError(
                "spectral normalization is not fitted; load the phase-0 "
                "preprocessing mean/std or fit it on phase-0 training spectra"
            )
        compute_dtype = (
            torch.float32
            if spectrum.dtype in (torch.float16, torch.bfloat16)
            else spectrum.dtype
        )
        mean = self.mean.to(device=spectrum.device, dtype=compute_dtype)
        scale = self.scale.to(device=spectrum.device, dtype=compute_dtype)
        normalized = (spectrum.to(dtype=compute_dtype) - mean) / scale
        normalized = normalized.to(dtype=spectrum.dtype)
        if not bool(torch.isfinite(normalized).all()):
            raise RuntimeError("spectral normalization produced NaN or Inf")
        return normalized


class SelectiveStateSpace(nn.Module):
    """Self-contained Mamba-style selective state-space mixer."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = _positive_int(d_model, "d_model")
        self.d_state = _positive_int(d_state, "d_state")
        self.d_conv = _positive_int(d_conv, "d_conv")
        self.expand = _positive_int(expand, "expand")
        self.d_inner = self.d_model * self.expand
        self.dt_rank = max(1, math.ceil(self.d_model / 16))
        if not 0.0 < dt_min < dt_max:
            raise ValueError("require 0 < dt_min < dt_max")

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        state_index = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(state_index.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        scale = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -scale, scale)
        initial_dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(dt_min), math.log(dt_max))
        )
        inverse_softplus = initial_dt + torch.log(-torch.expm1(-initial_dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def _scan(self, x: Tensor, delta: Tensor, b: Tensor, c: Tensor) -> Tensor:
        batch, length, _ = x.shape
        scan_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        state = torch.zeros(
            batch,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=scan_dtype,
        )
        a = -self.A_log.to(dtype=scan_dtype).exp()
        x_float = x.to(dtype=scan_dtype)
        delta_float = delta.to(dtype=scan_dtype)
        b_float = b.to(dtype=scan_dtype)
        c_float = c.to(dtype=scan_dtype)
        d = self.D.to(dtype=scan_dtype)
        outputs = []
        for index in range(length):
            step = delta_float[:, index]
            transition = torch.exp(step.unsqueeze(-1) * a.unsqueeze(0))
            input_state = (
                step.unsqueeze(-1)
                * b_float[:, index].unsqueeze(1)
                * x_float[:, index].unsqueeze(-1)
            )
            state = transition * state + input_state
            output = (state * c_float[:, index].unsqueeze(1)).sum(-1)
            output = output + d * x_float[:, index]
            outputs.append(output)
        return torch.stack(outputs, dim=1).to(dtype=x.dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        if (
            tokens.ndim != 3
            or tokens.size(0) == 0
            or tokens.size(1) == 0
            or tokens.size(-1) != self.d_model
        ):
            raise ValueError(f"tokens must be [N,L,{self.d_model}]")
        if not torch.is_floating_point(tokens) or tokens.is_complex():
            raise TypeError("tokens must be a real floating-point tensor")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("tokens contain NaN or Inf")
        length = tokens.size(1)
        x, gate = self.in_proj(tokens).chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)
        projected = self.x_proj(x)
        dt_low, b, c = torch.split(
            projected,
            (self.dt_rank, self.d_state, self.d_state),
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_low))
        mixed = self._scan(x, delta, b, c) * F.silu(gate)
        output = self.out_proj(mixed)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("selective state-space mixer produced NaN or Inf")
        return output


class MambaBlock(nn.Module):
    """Standard pre-normalized residual selective-SSM block."""

    def __init__(
        self,
        width: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mixer = SelectiveStateSpace(
            width,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.dropout(self.mixer(self.norm(tokens)))


class TokenPool(nn.Module):
    """Learned pooling used to summarize ordered spectral tokens."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.size(0) == 0 or tokens.size(1) == 0:
            raise ValueError("tokens must be non-empty [N,L,D]")
        weights = F.softmax(self.score(self.norm(tokens)).squeeze(-1), dim=1)
        return (tokens * weights.unsqueeze(-1)).sum(1), weights


class OrderedSpectralEncoder(nn.Module):
    """Shared-weight bidirectional Mamba over normalized physical bands."""

    def __init__(
        self,
        bands: int,
        token_dim: int,
        feature_dim: int,
        *,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        spectral_positions: Optional[Sequence[float]],
    ) -> None:
        super().__init__()
        self.bands = _positive_int(bands, "spectral_bands")
        self.token_dim = _positive_int(token_dim, "d_model")
        self.feature_dim = _positive_int(feature_dim, "spectral_feature_dim")
        positions = _spectral_positions(self.bands, spectral_positions)
        self.register_buffer("band_positions", positions)
        self.normalizer = FrozenBandStandardizer(self.bands)

        # Amplitude plus band position is tokenization, not a separate
        # handcrafted descriptor. Band relationships are learned by the SSM.
        self.embedding = nn.Linear(2, self.token_dim)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    self.token_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(_positive_int(layers, "num_spectral_layers"))
            ]
        )
        self.pool = TokenPool(self.token_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.feature_dim, bias=False),
        )

    def _encode(self, tokens: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, spectrum: Tensor) -> Dict[str, Tensor]:
        if (
            spectrum.ndim != 2
            or spectrum.size(0) == 0
            or spectrum.size(1) != self.bands
        ):
            raise ValueError(f"center_spectrum must be [N,{self.bands}]")
        if not torch.is_floating_point(spectrum) or spectrum.is_complex():
            raise TypeError("center_spectrum must be a real floating-point tensor")
        if not bool(torch.isfinite(spectrum).all()):
            raise ValueError("center_spectrum contains NaN or Inf")
        spectrum = self.normalizer(spectrum)
        position = self.band_positions.to(spectrum).expand(spectrum.size(0), -1)
        tokens = self.embedding(torch.stack((spectrum, position), dim=-1))

        paired = torch.cat((tokens, tokens.flip(1)), dim=0)
        encoded = self._encode(paired)
        forward, reverse = encoded.chunk(2, dim=0)
        tokens = 0.5 * (forward + reverse.flip(1))

        pooled, weights = self.pool(tokens)
        return {
            "feature": self.output(pooled),
            "tokens": tokens,
            "band_weights": weights,
        }



class SpectralConditionedContextPool(nn.Module):
    """Pool center-relative context tokens with the spectral anchor as query."""

    def __init__(self, token_dim: int, spectral_dim: int) -> None:
        super().__init__()
        self.token_dim = _positive_int(token_dim, "context_token_dim")
        self.spectral_dim = _positive_int(spectral_dim, "spectral_feature_dim")
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.query = nn.Linear(self.spectral_dim, self.token_dim, bias=False)
        self.key = nn.Linear(self.token_dim, self.token_dim, bias=False)

    def forward(
        self,
        tokens: Tensor,
        spectral_feature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            tokens.ndim != 3
            or tokens.size(0) == 0
            or tokens.size(1) == 0
            or tokens.size(2) != self.token_dim
        ):
            raise ValueError(
                f"context tokens must be non-empty [N,L,{self.token_dim}]"
            )
        if spectral_feature.shape != (tokens.size(0), self.spectral_dim):
            raise ValueError(
                f"spectral_feature must be [N,{self.spectral_dim}]"
            )
        if tokens.device != spectral_feature.device:
            raise ValueError(
                "context tokens and spectral_feature must share a device"
            )
        if tokens.dtype != spectral_feature.dtype:
            raise ValueError(
                "context tokens and spectral_feature must share a dtype"
            )
        if not bool(torch.isfinite(tokens).all()) or not bool(
            torch.isfinite(spectral_feature).all()
        ):
            raise ValueError("context pooling inputs contain NaN or Inf")

        query = self.query(spectral_feature).unsqueeze(1)
        keys = self.key(self.token_norm(tokens))
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.token_dim)
        weights = F.softmax(scores, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class CenterRelativeContextRefiner(nn.Module):
    """Refine the spectral anchor using only center-relative patch context.

    The patch branch is deliberately not an independent class representation.
    It receives the PCA/patch-space difference between each pixel and the
    center pixel, plus spatial position, and returns a correction with the same
    dimensionality as the spectral feature.

    The correction projection is zero-initialized. Therefore the initial model
    is exactly the spectral-only model, and neighborhood context is learned
    only through useful gradients from the single downstream representation.
    """

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        token_dim: int,
        spectral_dim: int,
        *,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.input_channels = _positive_int(input_channels, "num_bands")
        self.patch_size = _positive_int(patch_size, "patch_size")
        self.token_dim = _positive_int(token_dim, "d_model")
        self.spectral_dim = _positive_int(
            spectral_dim, "spectral_feature_dim"
        )
        self.token_count = self.patch_size**2
        groups = _positive_int(norm_groups, "stem_norm_groups")

        self.register_buffer(
            "spatial_positions",
            _spatial_positions(self.patch_size),
        )
        orders = _serpentine_orders(self.patch_size)
        self.register_buffer("scan_orders", orders, persistent=False)
        self.register_buffer(
            "inverse_scan_orders",
            orders.argsort(dim=1),
            persistent=False,
        )

        hidden = max(self.token_dim // 2, 16)
        if groups > hidden or hidden % groups != 0:
            raise ValueError(
                "stem_norm_groups must divide the context stem width "
                f"({hidden})"
            )

        # Context contains only relative HSI patch values and spatial position.
        # Absolute center-class semantics are supplied by spectral_feature.
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels + 2,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, self.token_dim, kernel_size=1, bias=False),
        )
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    self.token_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(_positive_int(layers, "num_spatial_layers"))
            ]
        )
        self.pool = SpectralConditionedContextPool(
            self.token_dim,
            self.spectral_dim,
        )
        self.correction_norm = nn.LayerNorm(self.token_dim)
        self.correction_proj = nn.Linear(
            self.token_dim,
            self.spectral_dim,
            bias=False,
        )
        nn.init.zeros_(self.correction_proj.weight)

    def _encode(self, tokens: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(
        self,
        patch: Tensor,
        spectral_feature: Tensor,
    ) -> Dict[str, Tensor]:
        expected = (
            self.input_channels,
            self.patch_size,
            self.patch_size,
        )
        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape[1:]) != expected
        ):
            raise ValueError(
                "patch must be "
                f"[N,{expected[0]},{expected[1]},{expected[2]}]"
            )
        if spectral_feature.shape != (
            patch.size(0),
            self.spectral_dim,
        ):
            raise ValueError(
                f"spectral_feature must be [N,{self.spectral_dim}]"
            )
        if not torch.is_floating_point(patch) or patch.is_complex():
            raise TypeError("patch must be a real floating-point tensor")
        if not torch.is_floating_point(
            spectral_feature
        ) or spectral_feature.is_complex():
            raise TypeError(
                "spectral_feature must be a real floating-point tensor"
            )
        if patch.device != spectral_feature.device:
            raise ValueError(
                "patch and spectral_feature must share a device"
            )
        if patch.dtype != spectral_feature.dtype:
            raise ValueError(
                "patch and spectral_feature must share a dtype"
            )
        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(spectral_feature).all()
        ):
            raise ValueError("context-refiner inputs contain NaN or Inf")

        center_index = self.patch_size // 2
        center = patch[:, :, center_index, center_index]
        relative = patch - center[:, :, None, None]
        positions = self.spatial_positions.to(patch).expand(
            patch.size(0),
            -1,
            -1,
            -1,
        )

        tokens = self.stem(torch.cat((relative, positions), dim=1))
        tokens = tokens.flatten(2).transpose(1, 2)

        batch = tokens.size(0)
        scanned = torch.cat(
            [
                tokens.index_select(1, order)
                for order in self.scan_orders
            ],
            dim=0,
        )
        encoded = self._encode(scanned).reshape(
            4,
            batch,
            self.token_count,
            self.token_dim,
        )
        restored = torch.stack(
            [
                encoded[index].index_select(
                    1,
                    self.inverse_scan_orders[index],
                )
                for index in range(4)
            ],
            dim=0,
        )
        tokens = restored.mean(dim=0)

        pooled, weights = self.pool(tokens, spectral_feature)
        correction = self.correction_proj(
            self.correction_norm(pooled)
        )
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError(
                "context refiner produced NaN or Inf correction"
            )
        return {
            "correction": correction,
            "tokens": tokens,
            "context_weights": weights,
            "relative_patch": relative,
        }


@dataclass(frozen=True)
class BackboneOutput:
    """Canonical one-space HSI representation consumed by the support bank.

    ``coordinates`` is the only deployed representation. ``feature`` remains a
    read-only compatibility alias so older trainer/evaluation code can migrate
    without maintaining a second tensor or a second semantic space.
    """

    coordinates: Tensor
    spectral_feature: Tensor
    context_correction: Tensor
    spectral_tokens: Optional[Tensor] = None
    context_tokens: Optional[Tensor] = None
    band_weights: Optional[Tensor] = None
    context_weights: Optional[Tensor] = None
    relative_patch: Optional[Tensor] = None

    @property
    def feature(self) -> Tensor:
        """Backward-compatible alias for the canonical geometry coordinates."""
        return self.coordinates


class HSIMambaFeatureExtractor(nn.Module):
    """Spectral-primary HSI backbone producing one canonical geometry space."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.patch_bands = _positive_int(
            args.num_bands,
            "num_bands",
        )
        self.spectral_bands = _positive_int(
            getattr(
                args,
                "spectral_bands",
                getattr(args, "raw_num_bands", args.num_bands),
            ),
            "spectral_bands",
        )
        self.patch_size = _positive_int(
            args.patch_size,
            "patch_size",
        )
        if self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")

        token_dim = _positive_int(args.d_model, "d_model")

        # One-space contract: the final HSI representation is the geometry
        # representation. ``representation_dim`` is the canonical option; the
        # ``spectral_feature_dim`` fallback keeps current experiment configs
        # usable during migration. A separate coordinate bottleneck is not used.
        representation_dim = _positive_int(
            getattr(
                args,
                "representation_dim",
                getattr(args, "spectral_feature_dim", 48),
            ),
            "representation_dim",
        )

        d_state = _positive_int(
            getattr(args, "d_state", 16),
            "d_state",
        )
        d_conv = _positive_int(
            getattr(args, "d_conv", 4),
            "d_conv",
        )
        expand = _positive_int(
            getattr(args, "expand", 2),
            "expand",
        )
        dropout = float(
            getattr(args, "mamba_dropout", 0.0)
        )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "mamba_dropout must lie in [0,1)"
            )

        self.representation_dim = representation_dim
        # Compatibility aliases. Both names intentionally refer to the same
        # deployed one-space dimensionality.
        self.feature_dim = self.representation_dim
        self.coordinate_dim = self.representation_dim

        spectral_positions = getattr(
            args,
            "band_wavelengths",
            None,
        )
        if spectral_positions is None:
            spectral_positions = getattr(
                args,
                "band_positions",
                None,
            )

        self.spectral_encoder = OrderedSpectralEncoder(
            self.spectral_bands,
            token_dim,
            self.feature_dim,
            layers=getattr(args, "num_spectral_layers", 2),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            spectral_positions=spectral_positions,
        )

        normalization_mean = getattr(
            args,
            "spectral_normalization_mean",
            None,
        )
        normalization_scale = getattr(
            args,
            "spectral_normalization_std",
            None,
        )
        if (normalization_mean is None) != (
            normalization_scale is None
        ):
            raise ValueError(
                "spectral_normalization_mean and "
                "spectral_normalization_std must be supplied together"
            )
        if normalization_mean is not None:
            self.spectral_encoder.normalizer.set_statistics(
                normalization_mean,
                normalization_scale,
            )

        self.context_refiner = CenterRelativeContextRefiner(
            self.patch_bands,
            self.patch_size,
            token_dim,
            self.feature_dim,
            layers=getattr(args, "num_spatial_layers", 2),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_groups=getattr(args, "stem_norm_groups", 1),
        )

    @property
    def spectral_normalization_fitted(self) -> bool:
        """Whether phase-0 ordered-spectrum statistics are available."""
        return bool(
            self.spectral_encoder.normalizer.fitted.item()
        )

    @torch.no_grad()
    def set_spectral_normalization(
        self,
        mean: Tensor | Sequence[float],
        scale: Tensor | Sequence[float],
        *,
        overwrite: bool = False,
    ) -> None:
        """Load phase-0 preprocessing statistics for the ordered spectrum."""
        self.spectral_encoder.normalizer.set_statistics(
            mean,
            scale,
            overwrite=overwrite,
        )

    @torch.no_grad()
    def fit_spectral_normalization(
        self,
        base_center_spectra: Tensor,
        *,
        overwrite: bool = False,
    ) -> None:
        """Fit once from phase-0 training center spectra when absent."""
        self.spectral_encoder.normalizer.fit(
            base_center_spectra,
            overwrite=overwrite,
        )

    def forward(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        return_aux: bool = False,
    ) -> BackboneOutput:
        if not torch.is_tensor(patch) or not torch.is_floating_point(
            patch
        ):
            raise TypeError(
                "patch must be a floating-point tensor"
            )
        if not torch.is_tensor(
            center_spectrum
        ) or not torch.is_floating_point(center_spectrum):
            raise TypeError(
                "center_spectrum must be a floating-point tensor"
            )
        if patch.is_complex() or center_spectrum.is_complex():
            raise TypeError(
                "backbone inputs must be real floating-point tensors"
            )

        expected_patch = (
            self.patch_bands,
            self.patch_size,
            self.patch_size,
        )
        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape[1:]) != expected_patch
        ):
            raise ValueError(
                "patch must be "
                f"[N,{self.patch_bands},"
                f"{self.patch_size},{self.patch_size}]"
            )
        if center_spectrum.shape != (
            patch.size(0),
            self.spectral_bands,
        ):
            raise ValueError(
                "center_spectrum must be "
                f"[N,{self.spectral_bands}]"
            )
        if patch.device != center_spectrum.device:
            raise ValueError(
                "patch and center_spectrum must share a device"
            )
        if patch.dtype != center_spectrum.dtype:
            raise ValueError(
                "patch and center_spectrum must share a dtype"
            )
        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(center_spectrum).all()
        ):
            raise ValueError(
                "backbone inputs contain NaN or Inf"
            )

        spectral = self.spectral_encoder(center_spectrum)
        context = self.context_refiner( patch, spectral["feature"], )
        coordinates = spectral["feature"] + context["correction"]
        if coordinates.shape != (
            patch.size(0),
            self.representation_dim,
        ):
            raise RuntimeError(
                "coordinates have shape "
                f"{tuple(coordinates.shape)}, expected "
                f"({patch.size(0)},{self.representation_dim})"
            )
        if not bool(torch.isfinite(coordinates).all()):
            raise RuntimeError(
                "backbone produced NaN or Inf values"
            )

        return BackboneOutput(
            coordinates=coordinates,
            spectral_feature=spectral["feature"],
            context_correction=context["correction"],
            spectral_tokens=(   spectral["tokens"] if return_aux else None ),
            context_tokens=(     context["tokens"] if return_aux else None ),
            band_weights=(     spectral["band_weights"] if return_aux else None ),
            context_weights=(  context["context_weights"]  if return_aux else None  ),
            relative_patch=(
                context["relative_patch"]
                if return_aux
                else None
            ),
        )


__all__ = [
    "BackboneOutput",
    "HSIMambaFeatureExtractor",
]














# from __future__ import annotations

# """Spectral-primary Mamba backbone for single-space HSI geometry learning.

# The backbone has three responsibilities:

# 1. learn ordered spectral evidence from the normalized full-band center spectrum;
# 2. extract center-relative neighborhood context from the HSI patch;
# 3. use that context only as a residual refinement of the spectral feature and
#    produce one class-independent coordinate vector for the geometry bank.

# The center spectrum is the semantic anchor because the patch label belongs to
# the center pixel. Spatial information does not create a second class feature
# space. It models how the neighborhood differs from the center and learns a
# residual correction in the same feature space.

# The backbone contains no class state, replay, prototype, classifier, geometry
# boundary, or incremental-phase update logic.
# """

# from dataclasses import dataclass
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Optional, Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# Tensor = torch.Tensor


# def _positive_int(value: object, name: str) -> int:
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be a positive integer")
#         value = value.item()
#     if isinstance(value, bool):
#         raise ValueError(f"{name} must be a positive integer")
#     if isinstance(value, Integral):
#         result = int(value)
#     elif isinstance(value, Real):
#         number = float(value)
#         if not math.isfinite(number) or not number.is_integer():
#             raise ValueError(f"{name} must be a positive integer")
#         result = int(number)
#     else:
#         raise ValueError(f"{name} must be a positive integer")
#     if result <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return result


# def _spectral_positions(
#     bands: int,
#     coordinates: Optional[Sequence[float]],
# ) -> Tensor:
#     """Return normalized band positions without fabricating wavelengths."""
#     if coordinates is None:
#         values = torch.arange(bands, dtype=torch.float32)
#     else:
#         values = torch.as_tensor(coordinates, dtype=torch.float32).flatten()
#         if values.numel() != bands:
#             raise ValueError(
#                 f"spectral coordinates have {values.numel()} values; expected {bands}"
#             )
#         if not bool(torch.isfinite(values).all()):
#             raise ValueError("spectral coordinates contain NaN or Inf")
#         if bool((values[1:] <= values[:-1]).any()):
#             raise ValueError("spectral coordinates must be strictly increasing")
#     if bands == 1:
#         return torch.zeros(1, dtype=torch.float32)
#     span = values[-1] - values[0]
#     if not bool(torch.isfinite(span)) or not bool(span > 0):
#         raise ValueError("spectral coordinates must have a finite positive span")
#     return 2.0 * (values - values[0]) / span - 1.0


# def _spatial_positions(patch_size: int) -> Tensor:
#     axis = torch.linspace(-1.0, 1.0, patch_size)
#     yy, xx = torch.meshgrid(axis, axis, indexing="ij")
#     return torch.stack((xx, yy), dim=0).unsqueeze(0)


# def _serpentine_orders(patch_size: int) -> Tensor:
#     """Horizontal/vertical forward/reverse locality-preserving scan orders."""
#     grid = torch.arange(patch_size * patch_size).reshape(patch_size, patch_size)
#     horizontal = torch.cat(
#         [grid[row] if row % 2 == 0 else grid[row].flip(0) for row in range(patch_size)]
#     )
#     vertical = torch.cat(
#         [
#             grid[:, col] if col % 2 == 0 else grid[:, col].flip(0)
#             for col in range(patch_size)
#         ]
#     )
#     return torch.stack(
#         (horizontal, horizontal.flip(0), vertical, vertical.flip(0)),
#         dim=0,
#     )


# class FrozenBandStandardizer(nn.Module):
#     """Persistent phase-0 per-band input normalization.

#     This stores only aggregate preprocessing statistics. It is not a class
#     representation, descriptor, replay buffer, or incremental-learning state.
#     """

#     def __init__(self, bands: int) -> None:
#         super().__init__()
#         self.bands = _positive_int(bands, "spectral_bands")
#         self.register_buffer("mean", torch.zeros(self.bands, dtype=torch.float32))
#         self.register_buffer("scale", torch.ones(self.bands, dtype=torch.float32))
#         self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

#     @torch.no_grad()
#     def set_statistics(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("spectral normalization is already fitted")
#         mean_tensor = torch.as_tensor(
#             mean,
#             device=self.mean.device,
#             dtype=torch.float32,
#         ).flatten()
#         scale_tensor = torch.as_tensor(
#             scale,
#             device=self.scale.device,
#             dtype=torch.float32,
#         ).flatten()
#         if mean_tensor.numel() != self.bands or scale_tensor.numel() != self.bands:
#             raise ValueError(
#                 f"spectral normalization statistics must have {self.bands} values"
#             )
#         if not bool(torch.isfinite(mean_tensor).all()) or not bool(
#             torch.isfinite(scale_tensor).all()
#         ):
#             raise ValueError("spectral normalization statistics contain NaN or Inf")
#         if bool((scale_tensor <= 0).any()):
#             raise ValueError("spectral normalization scales must be positive")
#         self.mean.copy_(mean_tensor)
#         self.scale.copy_(scale_tensor)
#         self.fitted.fill_(True)

#     @torch.no_grad()
#     def fit(self, spectra: Tensor, *, overwrite: bool = False) -> None:
#         values = torch.as_tensor(spectra)
#         if values.ndim != 2 or values.size(0) == 0 or values.size(1) != self.bands:
#             raise ValueError(f"base spectra must be non-empty [N,{self.bands}]")
#         if not torch.is_floating_point(values) or not bool(torch.isfinite(values).all()):
#             raise ValueError("base spectra must contain finite floating-point values")
#         values = values.to(dtype=torch.float32)
#         mean = values.mean(dim=0)
#         standard_deviation = values.std(dim=0, unbiased=False)
#         # A constant band carries no standardized variation. Unit scale keeps
#         # its centered value at zero without inventing variance.
#         scale = torch.where(
#             standard_deviation > 0,
#             standard_deviation,
#             torch.ones_like(standard_deviation),
#         )
#         self.set_statistics(mean, scale, overwrite=overwrite)

#     def forward(self, spectrum: Tensor) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError(
#                 "spectral normalization is not fitted; load the phase-0 "
#                 "preprocessing mean/std or fit it on phase-0 training spectra"
#             )
#         compute_dtype = (
#             torch.float32
#             if spectrum.dtype in (torch.float16, torch.bfloat16)
#             else spectrum.dtype
#         )
#         mean = self.mean.to(device=spectrum.device, dtype=compute_dtype)
#         scale = self.scale.to(device=spectrum.device, dtype=compute_dtype)
#         normalized = (spectrum.to(dtype=compute_dtype) - mean) / scale
#         normalized = normalized.to(dtype=spectrum.dtype)
#         if not bool(torch.isfinite(normalized).all()):
#             raise RuntimeError("spectral normalization produced NaN or Inf")
#         return normalized


# class SelectiveStateSpace(nn.Module):
#     """Self-contained Mamba-style selective state-space mixer."""

#     def __init__(
#         self,
#         d_model: int,
#         *,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dt_min: float = 1e-3,
#         dt_max: float = 1e-1,
#     ) -> None:
#         super().__init__()
#         self.d_model = _positive_int(d_model, "d_model")
#         self.d_state = _positive_int(d_state, "d_state")
#         self.d_conv = _positive_int(d_conv, "d_conv")
#         self.expand = _positive_int(expand, "expand")
#         self.d_inner = self.d_model * self.expand
#         self.dt_rank = max(1, math.ceil(self.d_model / 16))
#         if not 0.0 < dt_min < dt_max:
#             raise ValueError("require 0 < dt_min < dt_max")

#         self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
#         self.conv1d = nn.Conv1d(
#             self.d_inner,
#             self.d_inner,
#             kernel_size=self.d_conv,
#             groups=self.d_inner,
#             padding=self.d_conv - 1,
#         )
#         self.x_proj = nn.Linear(
#             self.d_inner,
#             self.dt_rank + 2 * self.d_state,
#             bias=False,
#         )
#         self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

#         state_index = torch.arange(1, self.d_state + 1, dtype=torch.float32)
#         self.A_log = nn.Parameter(state_index.log().repeat(self.d_inner, 1))
#         self.D = nn.Parameter(torch.ones(self.d_inner))
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

#         scale = self.dt_rank ** -0.5
#         nn.init.uniform_(self.dt_proj.weight, -scale, scale)
#         initial_dt = torch.exp(
#             torch.empty(self.d_inner).uniform_(math.log(dt_min), math.log(dt_max))
#         )
#         inverse_softplus = initial_dt + torch.log(-torch.expm1(-initial_dt))
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inverse_softplus)

#     def _scan(self, x: Tensor, delta: Tensor, b: Tensor, c: Tensor) -> Tensor:
#         batch, length, _ = x.shape
#         scan_dtype = (
#             torch.float32
#             if x.dtype in (torch.float16, torch.bfloat16)
#             else x.dtype
#         )
#         state = torch.zeros(
#             batch,
#             self.d_inner,
#             self.d_state,
#             device=x.device,
#             dtype=scan_dtype,
#         )
#         a = -self.A_log.to(dtype=scan_dtype).exp()
#         x_float = x.to(dtype=scan_dtype)
#         delta_float = delta.to(dtype=scan_dtype)
#         b_float = b.to(dtype=scan_dtype)
#         c_float = c.to(dtype=scan_dtype)
#         d = self.D.to(dtype=scan_dtype)
#         outputs = []
#         for index in range(length):
#             step = delta_float[:, index]
#             transition = torch.exp(step.unsqueeze(-1) * a.unsqueeze(0))
#             input_state = (
#                 step.unsqueeze(-1)
#                 * b_float[:, index].unsqueeze(1)
#                 * x_float[:, index].unsqueeze(-1)
#             )
#             state = transition * state + input_state
#             output = (state * c_float[:, index].unsqueeze(1)).sum(-1)
#             output = output + d * x_float[:, index]
#             outputs.append(output)
#         return torch.stack(outputs, dim=1).to(dtype=x.dtype)

#     def forward(self, tokens: Tensor) -> Tensor:
#         if (
#             tokens.ndim != 3
#             or tokens.size(0) == 0
#             or tokens.size(1) == 0
#             or tokens.size(-1) != self.d_model
#         ):
#             raise ValueError(f"tokens must be [N,L,{self.d_model}]")
#         if not torch.is_floating_point(tokens) or tokens.is_complex():
#             raise TypeError("tokens must be a real floating-point tensor")
#         if not bool(torch.isfinite(tokens).all()):
#             raise ValueError("tokens contain NaN or Inf")
#         length = tokens.size(1)
#         x, gate = self.in_proj(tokens).chunk(2, dim=-1)
#         x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
#         x = F.silu(x)
#         projected = self.x_proj(x)
#         dt_low, b, c = torch.split(
#             projected,
#             (self.dt_rank, self.d_state, self.d_state),
#             dim=-1,
#         )
#         delta = F.softplus(self.dt_proj(dt_low))
#         mixed = self._scan(x, delta, b, c) * F.silu(gate)
#         output = self.out_proj(mixed)
#         if not bool(torch.isfinite(output).all()):
#             raise RuntimeError("selective state-space mixer produced NaN or Inf")
#         return output


# class MambaBlock(nn.Module):
#     """Standard pre-normalized residual selective-SSM block."""

#     def __init__(
#         self,
#         width: int,
#         *,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dropout: float,
#     ) -> None:
#         super().__init__()
#         self.norm = nn.LayerNorm(width)
#         self.mixer = SelectiveStateSpace(
#             width,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, tokens: Tensor) -> Tensor:
#         return tokens + self.dropout(self.mixer(self.norm(tokens)))


# class TokenPool(nn.Module):
#     """Learned pooling used to summarize ordered spectral tokens."""

#     def __init__(self, width: int) -> None:
#         super().__init__()
#         self.norm = nn.LayerNorm(width)
#         self.score = nn.Linear(width, 1, bias=False)

#     def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
#         if tokens.ndim != 3 or tokens.size(0) == 0 or tokens.size(1) == 0:
#             raise ValueError("tokens must be non-empty [N,L,D]")
#         weights = F.softmax(self.score(self.norm(tokens)).squeeze(-1), dim=1)
#         return (tokens * weights.unsqueeze(-1)).sum(1), weights


# class OrderedSpectralEncoder(nn.Module):
#     """Shared-weight bidirectional Mamba over normalized physical bands."""

#     def __init__(
#         self,
#         bands: int,
#         token_dim: int,
#         feature_dim: int,
#         *,
#         layers: int,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dropout: float,
#         spectral_positions: Optional[Sequence[float]],
#     ) -> None:
#         super().__init__()
#         self.bands = _positive_int(bands, "spectral_bands")
#         self.token_dim = _positive_int(token_dim, "d_model")
#         self.feature_dim = _positive_int(feature_dim, "spectral_feature_dim")
#         positions = _spectral_positions(self.bands, spectral_positions)
#         self.register_buffer("band_positions", positions)
#         self.normalizer = FrozenBandStandardizer(self.bands)

#         # Amplitude plus band position is tokenization, not a separate
#         # handcrafted descriptor. Band relationships are learned by the SSM.
#         self.embedding = nn.Linear(2, self.token_dim)
#         self.blocks = nn.ModuleList(
#             [
#                 MambaBlock(
#                     self.token_dim,
#                     d_state=d_state,
#                     d_conv=d_conv,
#                     expand=expand,
#                     dropout=dropout,
#                 )
#                 for _ in range(_positive_int(layers, "num_spectral_layers"))
#             ]
#         )
#         self.pool = TokenPool(self.token_dim)
#         self.output = nn.Sequential(
#             nn.LayerNorm(self.token_dim),
#             nn.Linear(self.token_dim, self.feature_dim, bias=False),
#         )

#     def _encode(self, tokens: Tensor) -> Tensor:
#         for block in self.blocks:
#             tokens = block(tokens)
#         return tokens

#     def forward(self, spectrum: Tensor) -> Dict[str, Tensor]:
#         if (
#             spectrum.ndim != 2
#             or spectrum.size(0) == 0
#             or spectrum.size(1) != self.bands
#         ):
#             raise ValueError(f"center_spectrum must be [N,{self.bands}]")
#         if not torch.is_floating_point(spectrum) or spectrum.is_complex():
#             raise TypeError("center_spectrum must be a real floating-point tensor")
#         if not bool(torch.isfinite(spectrum).all()):
#             raise ValueError("center_spectrum contains NaN or Inf")
#         spectrum = self.normalizer(spectrum)
#         position = self.band_positions.to(spectrum).expand(spectrum.size(0), -1)
#         tokens = self.embedding(torch.stack((spectrum, position), dim=-1))

#         paired = torch.cat((tokens, tokens.flip(1)), dim=0)
#         encoded = self._encode(paired)
#         forward, reverse = encoded.chunk(2, dim=0)
#         tokens = 0.5 * (forward + reverse.flip(1))

#         pooled, weights = self.pool(tokens)
#         return {
#             "feature": self.output(pooled),
#             "tokens": tokens,
#             "band_weights": weights,
#         }



# class SpectralConditionedContextPool(nn.Module):
#     """Pool center-relative context tokens with the spectral anchor as query."""

#     def __init__(self, token_dim: int, spectral_dim: int) -> None:
#         super().__init__()
#         self.token_dim = _positive_int(token_dim, "context_token_dim")
#         self.spectral_dim = _positive_int(spectral_dim, "spectral_feature_dim")
#         self.token_norm = nn.LayerNorm(self.token_dim)
#         self.query = nn.Linear(self.spectral_dim, self.token_dim, bias=False)
#         self.key = nn.Linear(self.token_dim, self.token_dim, bias=False)

#     def forward(
#         self,
#         tokens: Tensor,
#         spectral_feature: Tensor,
#     ) -> tuple[Tensor, Tensor]:
#         if (
#             tokens.ndim != 3
#             or tokens.size(0) == 0
#             or tokens.size(1) == 0
#             or tokens.size(2) != self.token_dim
#         ):
#             raise ValueError(
#                 f"context tokens must be non-empty [N,L,{self.token_dim}]"
#             )
#         if spectral_feature.shape != (tokens.size(0), self.spectral_dim):
#             raise ValueError(
#                 f"spectral_feature must be [N,{self.spectral_dim}]"
#             )
#         if tokens.device != spectral_feature.device:
#             raise ValueError(
#                 "context tokens and spectral_feature must share a device"
#             )
#         if tokens.dtype != spectral_feature.dtype:
#             raise ValueError(
#                 "context tokens and spectral_feature must share a dtype"
#             )
#         if not bool(torch.isfinite(tokens).all()) or not bool(
#             torch.isfinite(spectral_feature).all()
#         ):
#             raise ValueError("context pooling inputs contain NaN or Inf")

#         query = self.query(spectral_feature).unsqueeze(1)
#         keys = self.key(self.token_norm(tokens))
#         scores = (query * keys).sum(dim=-1) / math.sqrt(self.token_dim)
#         weights = F.softmax(scores, dim=1)
#         pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
#         return pooled, weights


# class CenterRelativeContextRefiner(nn.Module):
#     """Refine the spectral anchor using only center-relative patch context.

#     The patch branch is deliberately not an independent class representation.
#     It receives the PCA/patch-space difference between each pixel and the
#     center pixel, plus spatial position, and returns a correction with the same
#     dimensionality as the spectral feature.

#     The correction projection is zero-initialized. Therefore the initial model
#     is exactly the spectral-only model, and neighborhood context is learned
#     only through useful gradients from the single downstream representation.
#     """

#     def __init__(
#         self,
#         input_channels: int,
#         patch_size: int,
#         token_dim: int,
#         spectral_dim: int,
#         *,
#         layers: int,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dropout: float,
#         norm_groups: int,
#     ) -> None:
#         super().__init__()
#         self.input_channels = _positive_int(input_channels, "num_bands")
#         self.patch_size = _positive_int(patch_size, "patch_size")
#         self.token_dim = _positive_int(token_dim, "d_model")
#         self.spectral_dim = _positive_int(
#             spectral_dim, "spectral_feature_dim"
#         )
#         self.token_count = self.patch_size**2
#         groups = _positive_int(norm_groups, "stem_norm_groups")

#         self.register_buffer(
#             "spatial_positions",
#             _spatial_positions(self.patch_size),
#         )
#         orders = _serpentine_orders(self.patch_size)
#         self.register_buffer("scan_orders", orders, persistent=False)
#         self.register_buffer(
#             "inverse_scan_orders",
#             orders.argsort(dim=1),
#             persistent=False,
#         )

#         hidden = max(self.token_dim // 2, 16)
#         if groups > hidden or hidden % groups != 0:
#             raise ValueError(
#                 "stem_norm_groups must divide the context stem width "
#                 f"({hidden})"
#             )

#         # Context contains only relative HSI patch values and spatial position.
#         # Absolute center-class semantics are supplied by spectral_feature.
#         self.stem = nn.Sequential(
#             nn.Conv2d(
#                 self.input_channels + 2,
#                 hidden,
#                 kernel_size=1,
#                 bias=False,
#             ),
#             nn.GroupNorm(groups, hidden),
#             nn.GELU(),
#             nn.Conv2d(
#                 hidden,
#                 hidden,
#                 kernel_size=3,
#                 padding=1,
#                 groups=hidden,
#                 bias=False,
#             ),
#             nn.GELU(),
#             nn.Conv2d(hidden, self.token_dim, kernel_size=1, bias=False),
#         )
#         self.blocks = nn.ModuleList(
#             [
#                 MambaBlock(
#                     self.token_dim,
#                     d_state=d_state,
#                     d_conv=d_conv,
#                     expand=expand,
#                     dropout=dropout,
#                 )
#                 for _ in range(_positive_int(layers, "num_spatial_layers"))
#             ]
#         )
#         self.pool = SpectralConditionedContextPool(
#             self.token_dim,
#             self.spectral_dim,
#         )
#         self.correction_norm = nn.LayerNorm(self.token_dim)
#         self.correction_proj = nn.Linear(
#             self.token_dim,
#             self.spectral_dim,
#             bias=False,
#         )
#         nn.init.zeros_(self.correction_proj.weight)

#     def _encode(self, tokens: Tensor) -> Tensor:
#         for block in self.blocks:
#             tokens = block(tokens)
#         return tokens

#     def forward(
#         self,
#         patch: Tensor,
#         spectral_feature: Tensor,
#     ) -> Dict[str, Tensor]:
#         expected = (
#             self.input_channels,
#             self.patch_size,
#             self.patch_size,
#         )
#         if (
#             patch.ndim != 4
#             or patch.size(0) == 0
#             or tuple(patch.shape[1:]) != expected
#         ):
#             raise ValueError(
#                 "patch must be "
#                 f"[N,{expected[0]},{expected[1]},{expected[2]}]"
#             )
#         if spectral_feature.shape != (
#             patch.size(0),
#             self.spectral_dim,
#         ):
#             raise ValueError(
#                 f"spectral_feature must be [N,{self.spectral_dim}]"
#             )
#         if not torch.is_floating_point(patch) or patch.is_complex():
#             raise TypeError("patch must be a real floating-point tensor")
#         if not torch.is_floating_point(
#             spectral_feature
#         ) or spectral_feature.is_complex():
#             raise TypeError(
#                 "spectral_feature must be a real floating-point tensor"
#             )
#         if patch.device != spectral_feature.device:
#             raise ValueError(
#                 "patch and spectral_feature must share a device"
#             )
#         if patch.dtype != spectral_feature.dtype:
#             raise ValueError(
#                 "patch and spectral_feature must share a dtype"
#             )
#         if not bool(torch.isfinite(patch).all()) or not bool(
#             torch.isfinite(spectral_feature).all()
#         ):
#             raise ValueError("context-refiner inputs contain NaN or Inf")

#         center_index = self.patch_size // 2
#         center = patch[:, :, center_index, center_index]
#         relative = patch - center[:, :, None, None]
#         positions = self.spatial_positions.to(patch).expand(
#             patch.size(0),
#             -1,
#             -1,
#             -1,
#         )

#         tokens = self.stem(torch.cat((relative, positions), dim=1))
#         tokens = tokens.flatten(2).transpose(1, 2)

#         batch = tokens.size(0)
#         scanned = torch.cat(
#             [
#                 tokens.index_select(1, order)
#                 for order in self.scan_orders
#             ],
#             dim=0,
#         )
#         encoded = self._encode(scanned).reshape(
#             4,
#             batch,
#             self.token_count,
#             self.token_dim,
#         )
#         restored = torch.stack(
#             [
#                 encoded[index].index_select(
#                     1,
#                     self.inverse_scan_orders[index],
#                 )
#                 for index in range(4)
#             ],
#             dim=0,
#         )
#         tokens = restored.mean(dim=0)

#         pooled, weights = self.pool(tokens, spectral_feature)
#         correction = self.correction_proj(
#             self.correction_norm(pooled)
#         )
#         if not bool(torch.isfinite(correction).all()):
#             raise RuntimeError(
#                 "context refiner produced NaN or Inf correction"
#             )
#         return {
#             "correction": correction,
#             "tokens": tokens,
#             "context_weights": weights,
#             "relative_patch": relative,
#         }


# class SharedDirectionProjection(nn.Module):
#     """Class-independent projection from one HSI feature to geometry coordinates."""

#     def __init__(self, input_dim: int, coordinate_dim: int) -> None:
#         super().__init__()
#         self.input_dim = _positive_int(input_dim, "projection_input_dim")
#         self.coordinate_dim = _positive_int(
#             coordinate_dim,
#             "coordinate_dim",
#         )
#         self.weight = nn.Parameter(
#             torch.empty(self.coordinate_dim, self.input_dim)
#         )
#         nn.init.orthogonal_(self.weight)

#     def forward(self, feature: Tensor) -> Tensor:
#         if (
#             feature.ndim != 2
#             or feature.size(0) == 0
#             or feature.size(1) != self.input_dim
#         ):
#             raise ValueError(
#                 f"feature must be [N,{self.input_dim}]"
#             )
#         if not torch.is_floating_point(feature) or feature.is_complex():
#             raise TypeError(
#                 "feature must be a real floating-point tensor"
#             )
#         if not bool(torch.isfinite(feature).all()):
#             raise ValueError(
#                 "feature must contain finite floating-point values"
#             )

#         norms = self.weight.norm(dim=1)
#         if not bool(torch.isfinite(norms).all()) or bool(
#             (norms == 0).any()
#         ):
#             raise RuntimeError(
#                 "coordinate projection contains an invalid direction"
#             )
#         directions = F.normalize(self.weight, dim=1)
#         coordinates = F.linear(feature, directions)
#         if not bool(torch.isfinite(coordinates).all()):
#             raise RuntimeError(
#                 "coordinate projection produced NaN or Inf"
#             )
#         return coordinates


# @dataclass(frozen=True)
# class BackboneOutput:
#     """Single-space representation consumed by the class geometry bank."""

#     feature: Tensor
#     coordinates: Tensor
#     spectral_feature: Tensor
#     context_correction: Tensor
#     spectral_tokens: Optional[Tensor] = None
#     context_tokens: Optional[Tensor] = None
#     band_weights: Optional[Tensor] = None
#     context_weights: Optional[Tensor] = None
#     relative_patch: Optional[Tensor] = None


# class HSIMambaFeatureExtractor(nn.Module):
#     """Spectral-primary HSI backbone producing one geometry coordinate vector."""

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.patch_bands = _positive_int(
#             args.num_bands,
#             "num_bands",
#         )
#         self.spectral_bands = _positive_int(
#             getattr(
#                 args,
#                 "spectral_bands",
#                 getattr(args, "raw_num_bands", args.num_bands),
#             ),
#             "spectral_bands",
#         )
#         self.patch_size = _positive_int(
#             args.patch_size,
#             "patch_size",
#         )
#         if self.patch_size % 2 == 0:
#             raise ValueError("patch_size must be odd")

#         token_dim = _positive_int(args.d_model, "d_model")
#         feature_dim = _positive_int(
#             getattr(args, "spectral_feature_dim", 48),
#             "spectral_feature_dim",
#         )
#         # `coordinate_dim` is the new single-space option. The fallback keeps
#         # existing experiment configs usable while joint_coordinate_dim becomes
#         # irrelevant to this backbone.
#         coordinate_dim = _positive_int(
#             getattr(
#                 args,
#                 "coordinate_dim",
#                 getattr(
#                     args,
#                     "spectral_coordinate_dim",
#                     feature_dim,
#                 ),
#             ),
#             "coordinate_dim",
#         )

#         d_state = _positive_int(
#             getattr(args, "d_state", 16),
#             "d_state",
#         )
#         d_conv = _positive_int(
#             getattr(args, "d_conv", 4),
#             "d_conv",
#         )
#         expand = _positive_int(
#             getattr(args, "expand", 2),
#             "expand",
#         )
#         dropout = float(
#             getattr(args, "mamba_dropout", 0.0)
#         )
#         if not 0.0 <= dropout < 1.0:
#             raise ValueError(
#                 "mamba_dropout must lie in [0,1)"
#             )

#         self.feature_dim = feature_dim
#         self.coordinate_dim = coordinate_dim

#         spectral_positions = getattr(
#             args,
#             "band_wavelengths",
#             None,
#         )
#         if spectral_positions is None:
#             spectral_positions = getattr(
#                 args,
#                 "band_positions",
#                 None,
#             )

#         self.spectral_encoder = OrderedSpectralEncoder(
#             self.spectral_bands,
#             token_dim,
#             self.feature_dim,
#             layers=getattr(args, "num_spectral_layers", 2),
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             dropout=dropout,
#             spectral_positions=spectral_positions,
#         )

#         normalization_mean = getattr(
#             args,
#             "spectral_normalization_mean",
#             None,
#         )
#         normalization_scale = getattr(
#             args,
#             "spectral_normalization_std",
#             None,
#         )
#         if (normalization_mean is None) != (
#             normalization_scale is None
#         ):
#             raise ValueError(
#                 "spectral_normalization_mean and "
#                 "spectral_normalization_std must be supplied together"
#             )
#         if normalization_mean is not None:
#             self.spectral_encoder.normalizer.set_statistics(
#                 normalization_mean,
#                 normalization_scale,
#             )

#         self.context_refiner = CenterRelativeContextRefiner(
#             self.patch_bands,
#             self.patch_size,
#             token_dim,
#             self.feature_dim,
#             layers=getattr(args, "num_spatial_layers", 2),
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             dropout=dropout,
#             norm_groups=getattr(args, "stem_norm_groups", 1),
#         )
#         self.coordinate_projection = SharedDirectionProjection(
#             self.feature_dim,
#             self.coordinate_dim,
#         )

#     @property
#     def spectral_normalization_fitted(self) -> bool:
#         """Whether phase-0 ordered-spectrum statistics are available."""
#         return bool(
#             self.spectral_encoder.normalizer.fitted.item()
#         )

#     @torch.no_grad()
#     def set_spectral_normalization(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         """Load phase-0 preprocessing statistics for the ordered spectrum."""
#         self.spectral_encoder.normalizer.set_statistics(
#             mean,
#             scale,
#             overwrite=overwrite,
#         )

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         """Fit once from phase-0 training center spectra when absent."""
#         self.spectral_encoder.normalizer.fit(
#             base_center_spectra,
#             overwrite=overwrite,
#         )

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         if not torch.is_tensor(patch) or not torch.is_floating_point(
#             patch
#         ):
#             raise TypeError(
#                 "patch must be a floating-point tensor"
#             )
#         if not torch.is_tensor(
#             center_spectrum
#         ) or not torch.is_floating_point(center_spectrum):
#             raise TypeError(
#                 "center_spectrum must be a floating-point tensor"
#             )
#         if patch.is_complex() or center_spectrum.is_complex():
#             raise TypeError(
#                 "backbone inputs must be real floating-point tensors"
#             )

#         expected_patch = (
#             self.patch_bands,
#             self.patch_size,
#             self.patch_size,
#         )
#         if (
#             patch.ndim != 4
#             or patch.size(0) == 0
#             or tuple(patch.shape[1:]) != expected_patch
#         ):
#             raise ValueError(
#                 "patch must be "
#                 f"[N,{self.patch_bands},"
#                 f"{self.patch_size},{self.patch_size}]"
#             )
#         if center_spectrum.shape != (
#             patch.size(0),
#             self.spectral_bands,
#         ):
#             raise ValueError(
#                 "center_spectrum must be "
#                 f"[N,{self.spectral_bands}]"
#             )
#         if patch.device != center_spectrum.device:
#             raise ValueError(
#                 "patch and center_spectrum must share a device"
#             )
#         if patch.dtype != center_spectrum.dtype:
#             raise ValueError(
#                 "patch and center_spectrum must share a dtype"
#             )
#         if not bool(torch.isfinite(patch).all()) or not bool(
#             torch.isfinite(center_spectrum).all()
#         ):
#             raise ValueError(
#                 "backbone inputs contain NaN or Inf"
#             )

#         spectral = self.spectral_encoder(center_spectrum)
#         context = self.context_refiner(
#             patch,
#             spectral["feature"],
#         )

#         # One semantic feature space: context only changes the spectral anchor
#         # through a residual correction of identical dimensionality.
#         feature = spectral["feature"] + context["correction"]
#         coordinates = self.coordinate_projection(feature)

#         if feature.shape != (
#             patch.size(0),
#             self.feature_dim,
#         ):
#             raise RuntimeError(
#                 "feature has shape "
#                 f"{tuple(feature.shape)}, expected "
#                 f"({patch.size(0)},{self.feature_dim})"
#             )
#         if coordinates.shape != (
#             patch.size(0),
#             self.coordinate_dim,
#         ):
#             raise RuntimeError(
#                 "coordinates have shape "
#                 f"{tuple(coordinates.shape)}, expected "
#                 f"({patch.size(0)},{self.coordinate_dim})"
#             )
#         if not bool(torch.isfinite(feature).all()) or not bool(
#             torch.isfinite(coordinates).all()
#         ):
#             raise RuntimeError(
#                 "backbone produced NaN or Inf values"
#             )

#         return BackboneOutput(
#             feature=feature,
#             coordinates=coordinates,
#             spectral_feature=spectral["feature"],
#             context_correction=context["correction"],
#             spectral_tokens=(
#                 spectral["tokens"] if return_aux else None
#             ),
#             context_tokens=(
#                 context["tokens"] if return_aux else None
#             ),
#             band_weights=(
#                 spectral["band_weights"] if return_aux else None
#             ),
#             context_weights=(
#                 context["context_weights"]
#                 if return_aux
#                 else None
#             ),
#             relative_patch=(
#                 context["relative_patch"]
#                 if return_aux
#                 else None
#             ),
#         )


# __all__ = [
#     "BackboneOutput",
#     "HSIMambaFeatureExtractor",
# ]
