from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


# =============================================================================
# Shared utilities
# =============================================================================


def _valid_group_count(channels: int, requested: int) -> int:
    channels = int(channels)
    requested = max(1, min(int(requested), channels))
    for groups in range(requested, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


@contextmanager
def _temporary_eval_mode(module: nn.Module) -> Iterator[None]:
    """Temporarily disable dropout/state updates while preserving gradients."""
    states = {child: child.training for child in module.modules()}
    try:
        module.eval()
        yield
    finally:
        for child, training in states.items():
            child.training = training


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    """Return [1,L,D] deterministic ordered-axis positional encoding."""
    if length <= 0 or width <= 0:
        raise ValueError("length and width must be positive")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    half = max((width + 1) // 2, 1)
    frequency = torch.exp(
        torch.arange(half, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = position * frequency.unsqueeze(0)
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles[:, : encoding[:, 0::2].shape[1]])
    if width > 1:
        encoding[:, 1::2] = torch.cos(
            angles[:, : encoding[:, 1::2].shape[1]]
        )
    return encoding.unsqueeze(0)


class RMSNorm(nn.Module):
    """RMS normalization without batch-dependent running statistics."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.width = int(width)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.width))

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.width:
            raise RuntimeError(
                f"RMSNorm expected width {self.width}, got {x.size(-1)}"
            )
        rms = x.float().square().mean(dim=-1, keepdim=True)
        rms = rms.add(self.eps).sqrt().to(dtype=x.dtype)
        return (x / rms) * self.weight


# =============================================================================
# Stable sequence blocks
# =============================================================================


class ChannelwiseS4DKernel(nn.Module):
    """Stable channel-specific diagonal state-space convolution.

    This is an S4D-style diagonal kernel, not a Mamba selective scan.
    """

    def __init__(
        self,
        width: int,
        state_dim: int = 16,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        kernel_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.state_dim = int(state_dim)
        self.kernel_dropout = float(kernel_dropout)
        if self.width <= 0 or self.state_dim <= 0:
            raise ValueError("width and state_dim must be positive")
        if not 0.0 < float(dt_min) < float(dt_max):
            raise ValueError("require 0 < dt_min < dt_max")
        if not 0.0 <= self.kernel_dropout < 1.0:
            raise ValueError("kernel_dropout must lie in [0,1)")

        base = torch.arange(1, self.state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(base.log().repeat(self.width, 1))
        self.B = nn.Parameter(torch.randn(self.width, self.state_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(self.width, self.state_dim) * 0.02)
        self.log_dt = nn.Parameter(
            torch.empty(self.width).uniform_(
                math.log(float(dt_min)), math.log(float(dt_max))
            )
        )
        self.D = nn.Parameter(torch.ones(self.width))

    def _kernel(
        self,
        length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if int(length) <= 0:
            raise ValueError("sequence length must be positive")

        A = -torch.exp(self.A_log.float()).to(device=device)
        dt = torch.exp(self.log_dt.float()).clamp(1e-4, 1.0).to(device=device)
        exponent = (dt[:, None] * A).clamp(min=-60.0, max=-1e-6)
        discrete_A = torch.exp(exponent)
        discrete_B = (
            torch.expm1(exponent)
            / A
            * self.B.float().to(device=device)
        )
        powers = torch.arange(length, device=device, dtype=torch.float32)
        transition = torch.pow(
            discrete_A.unsqueeze(-1), powers.view(1, 1, length)
        )
        kernel = torch.einsum(
            "dn,dn,dnl->dl",
            self.C.float().to(device=device),
            discrete_B,
            transition,
        )
        kernel = torch.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training and self.kernel_dropout > 0.0:
            kernel = F.dropout(
                kernel,
                p=self.kernel_dropout,
                training=True,
            )
        return kernel.to(dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(-1) != self.width:
            raise ValueError(
                f"expected [B,L,{self.width}], got {tuple(x.shape)}"
            )
        length = x.size(1)
        kernel = self._kernel(
            length,
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(1)
        sequence = x.transpose(1, 2)
        filtered = F.conv1d(
            F.pad(sequence, (length - 1, 0)),
            kernel,
            groups=self.width,
        )
        filtered = filtered[..., :length].transpose(1, 2)
        skip = self.D.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        return filtered + skip * x


class StableSSMBlock(nn.Module):
    """Pre-normalized gated diagonal-SSM residual block."""

    def __init__(
        self,
        width: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layer",
        residual_scale_init: float = 0.7,
        kernel_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.inner_width = self.width * int(expand)
        self.conv_kernel = int(conv_kernel)
        if self.width <= 0 or self.inner_width <= 0 or self.conv_kernel <= 0:
            raise ValueError("invalid SSM block dimensions")

        self.norm = (
            RMSNorm(self.width)
            if str(norm_type).strip().lower() == "rms"
            else nn.LayerNorm(self.width)
        )
        self.in_proj = nn.Linear(self.width, 2 * self.inner_width, bias=False)
        self.local_conv = nn.Conv1d(
            self.inner_width,
            self.inner_width,
            kernel_size=self.conv_kernel,
            groups=self.inner_width,
            bias=True,
        )
        self.ssm = ChannelwiseS4DKernel(
            self.inner_width,
            state_dim=state_dim,
            kernel_dropout=kernel_dropout,
        )
        self.out_proj = nn.Linear(self.inner_width, self.width, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.residual_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(residual_scale_init))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(-1) != self.width:
            raise ValueError(
                f"expected [B,L,{self.width}], got {tuple(x.shape)}"
            )
        residual = x
        projected, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        local = self.local_conv(
            F.pad(
                projected.transpose(1, 2),
                (self.conv_kernel - 1, 0),
            )
        ).transpose(1, 2)
        state = self.ssm(F.silu(local))
        update = self.out_proj(state * F.silu(gate))
        scale = torch.sigmoid(self.residual_logit).to(dtype=update.dtype)
        return residual + scale * self.dropout(update)


# Compatibility aliases. Do not describe these as selective Mamba blocks.
MambaBlock = StableSSMBlock
S4DKernel = ChannelwiseS4DKernel


# =============================================================================
# Scale-preserving branch projection
# =============================================================================


class ScalePreservingProjectionHead(nn.Module):
    """Residual projection without final L2 or LayerNorm.

    The output remains Euclidean and keeps radial information required by the
    factor Gaussian. The head is one of the controlled-plasticity modules.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_multiplier: int = 2,
        dropout: float = 0.0,
        residual_scale_init: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        hidden = max(self.output_dim * int(hidden_multiplier), self.output_dim)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("projection dimensions must be positive")

        self.norm = nn.LayerNorm(self.input_dim)
        self.skip = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.delta = nn.Sequential(
            nn.Linear(self.input_dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.output_dim, bias=False),
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.residual_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(residual_scale_init))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"expected [B,{self.input_dim}], got {tuple(x.shape)}"
            )
        normalized = self.norm(x)
        scale = torch.sigmoid(self.residual_logit).to(dtype=x.dtype)
        output = self.skip(x) + scale * self.delta(normalized) + self.bias
        if not torch.isfinite(output).all():
            raise RuntimeError("projection head produced NaN/Inf")
        return output


# =============================================================================
# Raw ordered-spectrum encoder
# =============================================================================


class RawOrderedSpectralEncoder(nn.Module):
    """Encode the raw ordered center spectrum into z_s.

    The input is the physical center-pixel spectrum, not PCA components from
    the spatial patch. Band order is represented by deterministic positional
    encoding, and a first-difference channel exposes local spectral shape while
    retaining the original amplitude channel.
    """

    def __init__(
        self,
        raw_bands: int,
        token_dim: int,
        output_dim: int,
        *,
        layers: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layer",
        residual_scale_init: float = 0.7,
        kernel_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.raw_bands = int(raw_bands)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        if self.raw_bands <= 1:
            raise ValueError("raw_bands must be greater than one")

        self.token_embed = nn.Linear(2, self.token_dim, bias=True)
        self.register_buffer(
            "ordered_position",
            _sinusoidal_positions(self.raw_bands, self.token_dim),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            [
                StableSSMBlock(
                    self.token_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand=expand,
                    dropout=dropout,
                    norm_type=norm_type,
                    residual_scale_init=residual_scale_init,
                    kernel_dropout=kernel_dropout,
                )
                for _ in range(int(layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        attention_hidden = max(self.token_dim // 2, 16)
        self.band_importance = nn.Sequential(
            nn.Linear(self.token_dim, attention_hidden),
            nn.GELU(),
            nn.Linear(attention_hidden, 1),
        )
        self.projection = ScalePreservingProjectionHead(
            self.token_dim,
            self.output_dim,
            dropout=projection_dropout,
        )

    @staticmethod
    def _band_descriptors(center_spectrum: Tensor) -> Tensor:
        difference = torch.zeros_like(center_spectrum)
        difference[:, 1:] = center_spectrum[:, 1:] - center_spectrum[:, :-1]
        return torch.stack([center_spectrum, difference], dim=-1)

    def _encode_bidirectional(self, tokens: Tensor) -> Tensor:
        paired = torch.cat([tokens, torch.flip(tokens, dims=[1])], dim=0)
        for layer in self.layers:
            paired = layer(paired)
        forward, reverse = paired.chunk(2, dim=0)
        reverse = torch.flip(reverse, dims=[1])
        return 0.5 * (forward + reverse)

    def forward(self, center_spectrum: Tensor) -> Dict[str, Tensor]:
        if center_spectrum.dim() != 2 or center_spectrum.size(1) != self.raw_bands:
            raise ValueError(
                f"expected center spectrum [B,{self.raw_bands}], "
                f"got {tuple(center_spectrum.shape)}"
            )
        if not torch.isfinite(center_spectrum).all():
            raise RuntimeError("center spectrum contains NaN/Inf")

        values = center_spectrum.float()
        descriptors = self._band_descriptors(values)
        tokens_initial = self.token_embed(descriptors)
        tokens_initial = tokens_initial + self.ordered_position.to(
            device=tokens_initial.device,
            dtype=tokens_initial.dtype,
        )
        mixed = self._encode_bidirectional(tokens_initial)
        normalized = self.output_norm(mixed)
        logits = self.band_importance(
            F.normalize(normalized, dim=-1, eps=1e-6)
        ).squeeze(-1)
        weights = F.softmax(logits, dim=1)
        pooled = torch.sum(mixed * weights.unsqueeze(-1), dim=1)
        feature = self.projection(pooled)
        return {
            "feature": feature,
            "tokens": normalized,
            "band_weights": weights,
            "band_descriptors": descriptors,
            "pooled": pooled,
        }


# =============================================================================
# Center-relative spatial encoder
# =============================================================================


class CenterRelativeSpatialEncoder(nn.Module):
    """Encode a base-reduced HSI patch after subtracting its center spectrum.

    The input channel axis may contain PCA components; no convolution is ever
    performed across component index. Convolution is only spatial (1x1 mixing
    plus depthwise 3x3 context), and the absolute center spectrum is removed
    before any learned spatial operation.
    """

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        token_dim: int,
        output_dim: int,
        *,
        layers: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_groups: int = 8,
        norm_type: str = "layer",
        residual_scale_init: float = 0.7,
        kernel_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.patch_size = int(patch_size)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.token_count = self.patch_size * self.patch_size
        if self.input_channels <= 0 or self.patch_size <= 0:
            raise ValueError("invalid spatial encoder dimensions")

        hidden = max(self.token_dim // 2, 32)
        groups = _valid_group_count(hidden, norm_groups)
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden, kernel_size=1, bias=False),
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
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.token_dim, kernel_size=1, bias=False),
        )

        self.row_embed = nn.Parameter(
            torch.randn(1, self.patch_size, self.token_dim) * 0.02
        )
        self.col_embed = nn.Parameter(
            torch.randn(1, self.patch_size, self.token_dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                StableSSMBlock(
                    self.token_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    expand=expand,
                    dropout=dropout,
                    norm_type=norm_type,
                    residual_scale_init=residual_scale_init,
                    kernel_dropout=kernel_dropout,
                )
                for _ in range(int(layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        attention_hidden = max(self.token_dim // 2, 16)
        self.spatial_importance = nn.Sequential(
            nn.Linear(self.token_dim, attention_hidden),
            nn.GELU(),
            nn.Linear(attention_hidden, 1),
        )
        self.direction_logits = nn.Parameter(torch.zeros(4))
        self.projection = ScalePreservingProjectionHead(
            self.token_dim,
            self.output_dim,
            dropout=projection_dropout,
        )

        grid = torch.arange(self.token_count, dtype=torch.long).reshape(
            self.patch_size, self.patch_size
        )
        orders = torch.stack(
            [
                grid.flatten(),
                torch.flip(grid.flatten(), dims=[0]),
                grid.transpose(0, 1).flatten(),
                torch.flip(grid.transpose(0, 1).flatten(), dims=[0]),
            ],
            dim=0,
        )
        self.register_buffer("scan_orders", orders, persistent=False)
        self.register_buffer(
            "inverse_scan_orders",
            torch.argsort(orders, dim=1),
            persistent=False,
        )

    def _position_encoding(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        position = self.row_embed[:, :, None, :] + self.col_embed[:, None, :, :]
        return position.reshape(1, self.token_count, self.token_dim).to(
            device=device, dtype=dtype
        )

    def _encode(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def center_relative_patch(self, patch: Tensor) -> Tensor:
        if patch.dim() != 4 or patch.size(1) != self.input_channels:
            raise ValueError(
                f"expected patch [B,{self.input_channels},H,W], "
                f"got {tuple(patch.shape)}"
            )
        if patch.size(2) != self.patch_size or patch.size(3) != self.patch_size:
            raise ValueError(
                f"expected {self.patch_size}x{self.patch_size} patch, "
                f"got {patch.size(2)}x{patch.size(3)}"
            )
        center = patch[:, :, self.patch_size // 2, self.patch_size // 2]
        return patch - center[:, :, None, None]

    def forward(self, patch: Tensor) -> Dict[str, Tensor]:
        if not torch.isfinite(patch).all():
            raise RuntimeError("spatial patch contains NaN/Inf")
        relative = self.center_relative_patch(patch.float())
        feature_map = self.stem(relative)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = tokens + self._position_encoding(
            dtype=tokens.dtype,
            device=tokens.device,
        )

        batch = tokens.size(0)
        ordered = torch.cat(
            [tokens.index_select(1, order) for order in self.scan_orders],
            dim=0,
        )
        encoded = self._encode(ordered)
        encoded = encoded.reshape(
            4,
            batch,
            self.token_count,
            self.token_dim,
        )
        restored = torch.stack(
            [
                encoded[index].index_select(
                    1, self.inverse_scan_orders[index]
                )
                for index in range(4)
            ],
            dim=0,
        )
        direction_weights = F.softmax(self.direction_logits, dim=0).to(
            dtype=restored.dtype
        )
        mixed = torch.sum(
            restored * direction_weights.view(4, 1, 1, 1), dim=0
        )
        normalized = self.output_norm(mixed)
        logits = self.spatial_importance(
            F.normalize(normalized, dim=-1, eps=1e-6)
        ).squeeze(-1)
        weights = F.softmax(logits, dim=1)
        pooled = torch.sum(mixed * weights.unsqueeze(-1), dim=1)
        feature = self.projection(pooled)

        relative_rms = relative.float().square().mean(
            dim=(1, 2, 3)
        ).sqrt().to(dtype=feature.dtype)
        return {
            "feature": feature,
            "tokens": normalized,
            "spatial_weights": weights,
            "direction_weights": direction_weights,
            "pooled": pooled,
            "center_relative_rms": relative_rms,
            "feature_map": feature_map,
        }


# =============================================================================
# Final backbone
# =============================================================================


@dataclass(frozen=True)
class BackboneDimensions:
    spectral_dim: int
    spatial_dim: int

    @property
    def joint_dim(self) -> int:
        return int(self.spectral_dim + self.spatial_dim)


class SSMBackbone(nn.Module):
    """Dual-branch HSI encoder for transport-closed factor geometry.

    Deployed coordinate
    -------------------
        z_s = E_s(raw center spectrum)       in R^Ds
        z_p = E_p(reduced patch - center)    in R^Dp
        z   = [z_s ; z_p]                    in R^(Ds+Dp)

    There is deliberately no learned fusion after concatenation. This explicit
    partition is required by:

        Psi_c = diag(psi_c,s I_Ds, psi_c,p I_Dp)

    and by branchwise transport:

        A_t = blockdiag(a_s R_s, a_p R_p).

    The spatial input must already be reduced using a base-training-only band
    reducer. The backbone cannot verify how that reducer was fitted; the data
    pipeline must enforce the leakage-safe protocol.
    """

    CONTRACT_VERSION = 2

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.model_input_bands = int(args.num_bands)
        self.raw_num_bands = int(
            getattr(args, "raw_num_bands", self.model_input_bands)
        )
        self.patch_size = int(args.patch_size)
        self.token_dim = int(args.d_model)
        self.dimensions = BackboneDimensions(
            spectral_dim=int(getattr(args, "spectral_feature_dim", 32)),
            spatial_dim=int(getattr(args, "spatial_feature_dim", 32)),
        )
        self.spectral_feature_dim = self.dimensions.spectral_dim
        self.spatial_feature_dim = self.dimensions.spatial_dim
        self.feature_dim = self.dimensions.joint_dim
        # Compatibility with code that previously used backbone.d_model as the
        # final feature width. New code should use feature_dim.
        self.d_model = self.feature_dim

        if self.model_input_bands <= 0 or self.raw_num_bands <= 1:
            raise ValueError("invalid input band dimensions")
        if self.patch_size <= 0 or self.token_dim <= 0:
            raise ValueError("invalid backbone dimensions")

        layer_count_s = int(getattr(args, "num_spectral_layers", 2))
        layer_count_p = int(getattr(args, "num_spatial_layers", getattr(args, "num_layers", 2)))
        state_dim = int(getattr(args, "d_state", 16))
        conv_kernel = int(getattr(args, "d_conv", 4))
        expand = int(getattr(args, "expand", 2))
        dropout = float(getattr(args, "dropout", 0.1))
        norm_type = str(getattr(args, "backbone_norm", "layer"))
        residual_scale = float(getattr(args, "ssm_residual_scale_init", 0.7))
        kernel_dropout = float(getattr(args, "ssm_kernel_dropout", 0.0))
        projection_dropout = float(
            getattr(args, "projection_dropout", 0.0)
        )

        self.spectral_encoder = RawOrderedSpectralEncoder(
            raw_bands=self.raw_num_bands,
            token_dim=self.token_dim,
            output_dim=self.spectral_feature_dim,
            layers=layer_count_s,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand=expand,
            dropout=dropout,
            norm_type=norm_type,
            residual_scale_init=residual_scale,
            kernel_dropout=kernel_dropout,
            projection_dropout=projection_dropout,
        )
        self.spatial_encoder = CenterRelativeSpatialEncoder(
            input_channels=self.model_input_bands,
            patch_size=self.patch_size,
            token_dim=self.token_dim,
            output_dim=self.spatial_feature_dim,
            layers=layer_count_p,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand=expand,
            dropout=dropout,
            norm_groups=int(getattr(args, "stem_norm_groups", 8)),
            norm_type=norm_type,
            residual_scale_init=residual_scale,
            kernel_dropout=kernel_dropout,
            projection_dropout=projection_dropout,
        )

        self._fully_frozen = False
        self._incremental_policy = "base"
        self._partial_training_modules: List[nn.Module] = []

    # ------------------------------------------------------------------
    # Input and output contracts
    # ------------------------------------------------------------------

    def _validate_model_patch(self, patch: Tensor) -> None:
        if not torch.is_tensor(patch):
            raise TypeError("patch must be a tensor")
        expected = (
            patch.size(0),
            self.model_input_bands,
            self.patch_size,
            self.patch_size,
        ) if patch.dim() == 4 else None
        if patch.dim() != 4:
            raise ValueError(
                f"expected model patch [B,C,H,W], got {tuple(patch.shape)}"
            )
        if tuple(patch.shape) != expected:
            raise ValueError(
                f"expected [B,{self.model_input_bands},{self.patch_size},"
                f"{self.patch_size}], got {tuple(patch.shape)}"
            )
        if not torch.isfinite(patch).all():
            raise RuntimeError("model patch contains NaN/Inf")

    def _resolve_raw_center_spectrum(
        self,
        model_patch: Tensor,
        *,
        raw_spectral_patch: Optional[Tensor],
        raw_center_spectrum: Optional[Tensor],
    ) -> Tensor:
        if raw_spectral_patch is not None and raw_center_spectrum is not None:
            raise ValueError(
                "pass at most one of raw_spectral_patch and raw_center_spectrum"
            )
        if raw_center_spectrum is not None:
            value = raw_center_spectrum
            if value.dim() != 2 or value.size(1) != self.raw_num_bands:
                raise ValueError(
                    f"raw_center_spectrum must be [B,{self.raw_num_bands}], "
                    f"got {tuple(value.shape)}"
                )
            if value.size(0) != model_patch.size(0):
                raise ValueError("raw center spectrum batch mismatch")
            if not torch.isfinite(value).all():
                raise RuntimeError("raw center spectrum contains NaN/Inf")
            return value

        if raw_spectral_patch is not None:
            expected = (
                model_patch.size(0),
                self.raw_num_bands,
                self.patch_size,
                self.patch_size,
            )
            if tuple(raw_spectral_patch.shape) != expected:
                raise ValueError(
                    f"raw_spectral_patch must be {expected}, got "
                    f"{tuple(raw_spectral_patch.shape)}"
                )
            if not torch.isfinite(raw_spectral_patch).all():
                raise RuntimeError("raw_spectral_patch contains NaN/Inf")
            return raw_spectral_patch[
                :,
                :,
                self.patch_size // 2,
                self.patch_size // 2,
            ]

        if self.raw_num_bands == self.model_input_bands:
            return model_patch[
                :,
                :,
                self.patch_size // 2,
                self.patch_size // 2,
            ]
        raise RuntimeError(
            "raw ordered center spectrum is required because the spatial "
            "model input has been spectrally reduced"
        )

    @staticmethod
    @torch.no_grad()
    def branch_statistics(feature: Tensor, prefix: str) -> Dict[str, Tensor]:
        if feature.dim() != 2:
            raise ValueError("branch feature must be [B,D]")
        norms = feature.norm(dim=1)
        return {
            f"{prefix}_norm_mean": norms.mean().detach(),
            f"{prefix}_norm_std": norms.std(unbiased=False).detach(),
            f"{prefix}_norm_cv": (
                norms.std(unbiased=False) / norms.mean().clamp_min(1e-8)
            ).detach(),
            f"{prefix}_dimension_variance_mean": feature.var(
                dim=0, unbiased=False
            ).mean().detach(),
        }

    def backbone_contract(self) -> Dict[str, Any]:
        return {
            "version": self.CONTRACT_VERSION,
            "model_patch_shape": (
                f"[B,{self.model_input_bands},{self.patch_size},{self.patch_size}]"
            ),
            "raw_center_shape": f"[B,{self.raw_num_bands}]",
            "spectral_coordinate": "raw_ordered_center_spectrum",
            "spatial_coordinate": "base_reduced_center_relative_patch",
            "spectral_feature_dim": self.spectral_feature_dim,
            "spatial_feature_dim": self.spatial_feature_dim,
            "joint_feature_dim": self.feature_dim,
            "joint_rule": "direct_concatenation_[z_s;z_p]",
            "learned_post_concatenation_fusion": False,
            "final_l2_normalization": False,
            "final_feature_normalization": False,
            "spatial_absolute_center_removed": True,
            "spectral_band_order_preserved": True,
            "classification_factorization": "p(z|c)",
            "compatible_residual_covariance": (
                "diag(psi_s I_Ds, psi_p I_Dp)"
            ),
            "compatible_transport": (
                "blockdiag(a_s R_s, a_p R_p)"
            ),
            "incremental_policy": self._incremental_policy,
            "uses_task_specific_adapter": False,
            "stores_exemplars": False,
            "spatial_reducer_requirement": (
                "fitted_on_base_training_pixels_only_and_frozen"
            ),
            "ssm_family": "channelwise_diagonal_s4d",
            "is_mamba_selective_scan": False,
        }

    # ------------------------------------------------------------------
    # Parameter and mode policy
    # ------------------------------------------------------------------

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(bool(trainable))
            parameter.grad = None

    def train(self, mode: bool = True) -> "SSMBackbone":
        requested = bool(mode)
        if self._fully_frozen or self._incremental_policy == "frozen":
            super().train(False)
            return self
        if self._incremental_policy == "controlled":
            super().train(False)
            for module in self._partial_training_modules:
                module.train(requested)
            return self
        super().train(requested)
        return self

    def enable_base_training(self) -> None:
        self._fully_frozen = False
        self._incremental_policy = "base"
        self._partial_training_modules.clear()
        for parameter in self.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        super().train(True)

    def freeze_for_incremental(self) -> None:
        """Frozen-backbone ablation and safest first incremental baseline."""
        self._fully_frozen = False
        self._incremental_policy = "frozen"
        self._partial_training_modules.clear()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        super().train(False)

    def enable_incremental_plasticity(
        self,
        *,
        train_last_spectral_block: bool = True,
        train_last_spatial_block: bool = True,
        train_projection_heads: bool = True,
        train_late_norms: bool = True,
    ) -> List[str]:
        """Enable only the controlled late modules used by transport fitting.

        The previous accepted encoder must be retained externally as a frozen
        phase observer. The trainer must apply coordinate transport loss and
        evidence-gated row transport when this mode is active.
        """
        self._fully_frozen = False
        self._incremental_policy = "controlled"
        self._partial_training_modules.clear()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

        modules: List[nn.Module] = []
        if train_last_spectral_block and len(self.spectral_encoder.layers):
            modules.append(self.spectral_encoder.layers[-1])
        if train_last_spatial_block and len(self.spatial_encoder.layers):
            modules.append(self.spatial_encoder.layers[-1])
        if train_projection_heads:
            modules.extend(
                [
                    self.spectral_encoder.projection,
                    self.spatial_encoder.projection,
                ]
            )
        if train_late_norms:
            modules.extend(
                [
                    self.spectral_encoder.output_norm,
                    self.spatial_encoder.output_norm,
                ]
            )

        # Preserve order while removing duplicate module identities.
        unique_modules: List[nn.Module] = []
        seen: set[int] = set()
        for module in modules:
            if id(module) not in seen:
                seen.add(id(module))
                unique_modules.append(module)
        self._partial_training_modules = unique_modules
        for module in self._partial_training_modules:
            self._set_module_trainable(module, True)
        self.train(True)
        return self.incremental_trainable_parameter_names()

    def freeze_all(self) -> None:
        self._fully_frozen = True
        self._incremental_policy = "frozen_all"
        self._partial_training_modules.clear()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        super().train(False)

    def incremental_trainable_parameters(self) -> List[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def incremental_trainable_parameter_names(self) -> List[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def assert_incremental_policy(self) -> Dict[str, Any]:
        trainable = self.incremental_trainable_parameter_names()
        if self._incremental_policy == "frozen" and trainable:
            raise RuntimeError(
                f"frozen incremental policy has trainable parameters: {trainable[:8]}"
            )
        if self._incremental_policy == "controlled" and not trainable:
            raise RuntimeError("controlled incremental policy has no trainable parameters")

        allowed_prefixes = (
            "spectral_encoder.layers.",
            "spatial_encoder.layers.",
            "spectral_encoder.projection.",
            "spatial_encoder.projection.",
            "spectral_encoder.output_norm.",
            "spatial_encoder.output_norm.",
        )
        if self._incremental_policy == "controlled":
            illegal = [
                name for name in trainable
                if not name.startswith(allowed_prefixes)
            ]
            if illegal:
                raise RuntimeError(
                    f"controlled policy exposed illegal parameters: {illegal[:8]}"
                )
        active_modules = [
            name
            for name, module in self.named_modules()
            if name and module.training
        ]
        return {
            "policy": self._incremental_policy,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "trainable_parameter_names": trainable,
            "active_training_modules": active_modules,
        }

    @torch.no_grad()
    def make_phase_observer(self) -> "SSMBackbone":
        """Deep-copy the accepted phase encoder for current-sample pairing."""
        observer = copy.deepcopy(self)
        observer.freeze_all()
        return observer

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        model_patch: Tensor,
        *,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        self._validate_model_patch(model_patch)
        center = self._resolve_raw_center_spectrum(
            model_patch,
            raw_spectral_patch=raw_spectral_patch,
            raw_center_spectrum=raw_center_spectrum,
        )
        spectral = self.spectral_encoder(center)
        spatial = self.spatial_encoder(model_patch)

        z_s = spectral["feature"]
        z_p = spatial["feature"]
        if z_s.shape != (model_patch.size(0), self.spectral_feature_dim):
            raise RuntimeError("spectral feature shape contract failed")
        if z_p.shape != (model_patch.size(0), self.spatial_feature_dim):
            raise RuntimeError("spatial feature shape contract failed")

        joint = torch.cat([z_s, z_p], dim=1)
        if joint.shape != (model_patch.size(0), self.feature_dim):
            raise RuntimeError("joint feature shape contract failed")
        if not torch.equal(joint[:, : self.spectral_feature_dim], z_s):
            raise RuntimeError("joint spectral slice contract failed")
        if not torch.equal(joint[:, self.spectral_feature_dim :], z_p):
            raise RuntimeError("joint spatial slice contract failed")
        if not torch.isfinite(joint).all():
            raise RuntimeError("backbone produced NaN/Inf joint features")

        statistics = {
            **self.branch_statistics(z_s, "spectral_feature"),
            **self.branch_statistics(z_p, "spatial_feature"),
            **self.branch_statistics(joint, "joint_feature"),
        }
        return {
            "spectral_feature": z_s,
            "spectral_features": z_s,
            "spatial_feature": z_p,
            "spatial_features": z_p,
            "joint_feature": joint,
            "joint_features": joint,
            "geometry_features": joint,
            "features": joint,
            "raw_center_spectrum": center,
            "raw_center_spectra": center,
            "spectral_tokens": spectral["tokens"],
            "spatial_tokens": spatial["tokens"],
            "band_weights": spectral["band_weights"],
            "spatial_weights": spatial["spatial_weights"],
            "spatial_direction_weights": spatial["direction_weights"],
            "center_relative_rms": spatial["center_relative_rms"],
            "feature_stats": statistics,
            "backbone_contract": self.backbone_contract(),
        }

    def forward(
        self,
        model_patch: Tensor,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        return self._forward_impl(
            model_patch,
            raw_spectral_patch=raw_spectral_patch,
            raw_center_spectrum=raw_center_spectrum,
        )

    def forward_geometry(
        self,
        model_patch: Tensor,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Deterministic geometry forward while preserving gradients."""
        with _temporary_eval_mode(self):
            return self._forward_impl(
                model_patch,
                raw_spectral_patch=raw_spectral_patch,
                raw_center_spectrum=raw_center_spectrum,
            )

    def encode_bridge(
        self,
        model_patch: Tensor,
        *,
        raw_spectral_patch: Optional[Tensor] = None,
        raw_center_spectrum: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        if deterministic:
            return self.forward_geometry(
                model_patch,
                raw_spectral_patch=raw_spectral_patch,
                raw_center_spectrum=raw_center_spectrum,
            )
        return self.forward(
            model_patch,
            raw_spectral_patch=raw_spectral_patch,
            raw_center_spectrum=raw_center_spectrum,
        )

    forward_bridge = forward_geometry


# Explicit descriptive alias.
HSIFactorBackbone = SSMBackbone


















# from __future__ import annotations

# import math
# from contextlib import contextmanager
# from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # =============================================================================
# # Shared utilities
# # =============================================================================


# def _valid_group_count(channels: int, requested: int) -> int:
#     """Return the largest requested-or-smaller divisor for GroupNorm."""
#     channels = int(channels)
#     requested = max(1, min(int(requested), channels))
#     for groups in range(requested, 0, -1):
#         if channels % groups == 0:
#             return groups
#     return 1


# def _inverse_sigmoid(value: float) -> float:
#     value = min(max(float(value), 1e-4), 1.0 - 1e-4)
#     return math.log(value / (1.0 - value))


# @contextmanager
# def _temporary_eval_mode(module: nn.Module) -> Iterator[None]:
#     """Disable stochastic/state-updating behavior without disabling gradients."""
#     states = {child: child.training for child in module.modules()}
#     try:
#         module.eval()
#         yield
#     finally:
#         # Direct assignment avoids recursively changing children multiple times.
#         for child, training in states.items():
#             child.training = training


# class RMSNorm(nn.Module):
#     """RMS normalization without batch-dependent state."""

#     def __init__(self, d_model: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.eps = float(eps)
#         self.weight = nn.Parameter(torch.ones(int(d_model)))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.size(-1) != self.weight.numel():
#             raise RuntimeError(
#                 f"RMSNorm expected last dimension {self.weight.numel()}, "
#                 f"got {x.size(-1)}"
#             )
#         rms = x.float().square().mean(dim=-1, keepdim=True)
#         rms = rms.add(self.eps).sqrt().to(dtype=x.dtype)
#         return (x / rms) * self.weight


# # =============================================================================
# # Stable diagonal state-space blocks
# # =============================================================================


# class ChannelwiseS4DKernel(nn.Module):
#     """Stable channel-specific diagonal state-space convolution.

#     This is an S4D-style diagonal state-space kernel. It is not a Mamba
#     selective scan: its state parameters do not depend on the current input.
#     The explicit name prevents the paper from claiming a mechanism that the
#     implementation does not contain.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         d_state: int = 16,
#         dt_min: float = 1e-3,
#         dt_max: float = 1e-1,
#         kernel_dropout: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.d_state = int(d_state)
#         self.kernel_dropout = float(kernel_dropout)

#         if self.d_model <= 0 or self.d_state <= 0:
#             raise ValueError("d_model and d_state must be positive")
#         if not 0.0 < float(dt_min) < float(dt_max):
#             raise ValueError("Require 0 < dt_min < dt_max")
#         if not 0.0 <= self.kernel_dropout < 1.0:
#             raise ValueError("kernel_dropout must lie in [0,1)")

#         # Stable continuous poles A=-exp(A_log).
#         base = torch.arange(1, self.d_state + 1, dtype=torch.float32)
#         self.A_log = nn.Parameter(base.log().repeat(self.d_model, 1))
#         self.B = nn.Parameter(
#             torch.randn(self.d_model, self.d_state) * 0.02
#         )
#         self.C = nn.Parameter(
#             torch.randn(self.d_model, self.d_state) * 0.02
#         )

#         low = math.log(float(dt_min))
#         high = math.log(float(dt_max))
#         self.log_dt = nn.Parameter(
#             torch.empty(self.d_model).uniform_(low, high)
#         )
#         self.D = nn.Parameter(torch.ones(self.d_model))

#     def _kernel(
#         self,
#         length: int,
#         dtype: torch.dtype,
#         device: torch.device,
#     ) -> torch.Tensor:
#         length = int(length)
#         if length <= 0:
#             raise ValueError(f"length must be positive, got {length}")

#         # Kernel construction stays in float32 for stable exponentiation.
#         A = -torch.exp(self.A_log.float()).to(device=device)
#         dt = torch.exp(self.log_dt.float()).clamp(1e-4, 1.0)
#         dt = dt.to(device=device)
#         exponent = (dt[:, None] * A).clamp(min=-60.0, max=-1e-6)

#         discrete_A = torch.exp(exponent)
#         # Zero-order-hold discretization:
#         # B_bar = A^{-1}(exp(dt*A)-I)B.
#         discrete_B = (
#             torch.expm1(exponent)
#             / A
#             * self.B.float().to(device=device)
#         )

#         powers = torch.arange(
#             length, device=device, dtype=torch.float32
#         )
#         transition_powers = torch.pow(
#             discrete_A.unsqueeze(-1),
#             powers.view(1, 1, length),
#         )
#         kernel = torch.einsum(
#             "dn,dn,dnl->dl",
#             self.C.float().to(device=device),
#             discrete_B,
#             transition_powers,
#         )
#         kernel = torch.nan_to_num(
#             kernel, nan=0.0, posinf=0.0, neginf=0.0
#         )
#         if self.training and self.kernel_dropout > 0.0:
#             kernel = F.dropout(
#                 kernel,
#                 p=self.kernel_dropout,
#                 training=True,
#             )
#         return kernel.to(dtype=dtype)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 3:
#             raise ValueError(
#                 f"Expected x [B,L,D], got {tuple(x.shape)}"
#             )
#         _, length, width = x.shape
#         if width != self.d_model:
#             raise ValueError(
#                 f"Expected hidden width {self.d_model}, got {width}"
#             )

#         kernel = self._kernel(
#             length, x.dtype, x.device
#         ).unsqueeze(1)
#         sequence = x.transpose(1, 2)
#         filtered = F.conv1d(
#             F.pad(sequence, (length - 1, 0)),
#             kernel,
#             groups=self.d_model,
#         )
#         filtered = filtered[..., :length].transpose(1, 2)
#         skip = self.D.to(
#             device=x.device, dtype=x.dtype
#         ).view(1, 1, -1)
#         return filtered + skip * x


# class StableSSMBlock(nn.Module):
#     """Pre-normalized gated diagonal-SSM block.

#     The residual update is bounded by a learned sigmoid scalar. This keeps the
#     feature field stable enough for low-rank class geometry while still
#     permitting representation learning during the base phase.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         d_state: int = 16,
#         d_conv: int = 4,
#         expand: int = 2,
#         dropout: float = 0.1,
#         norm_type: str = "layer",
#         residual_scale_init: float = 0.70,
#         kernel_dropout: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.d_inner = int(d_model) * int(expand)
#         self.d_conv = int(d_conv)
#         if self.d_model <= 0 or self.d_inner <= 0 or self.d_conv <= 0:
#             raise ValueError("Invalid SSM block dimensions")

#         if str(norm_type).strip().lower() == "rms":
#             self.norm = RMSNorm(self.d_model)
#         else:
#             self.norm = nn.LayerNorm(self.d_model)

#         self.in_proj = nn.Linear(
#             self.d_model, 2 * self.d_inner, bias=False
#         )
#         self.local_conv = nn.Conv1d(
#             self.d_inner,
#             self.d_inner,
#             kernel_size=self.d_conv,
#             padding=0,
#             groups=self.d_inner,
#             bias=True,
#         )
#         self.ssm = ChannelwiseS4DKernel(
#             self.d_inner,
#             int(d_state),
#             kernel_dropout=kernel_dropout,
#         )
#         self.out_proj = nn.Linear(
#             self.d_inner, self.d_model, bias=False
#         )
#         self.dropout = nn.Dropout(float(dropout))
#         self.residual_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(float(residual_scale_init))
#             )
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 3 or x.size(-1) != self.d_model:
#             raise ValueError(
#                 f"Expected x [B,L,{self.d_model}], "
#                 f"got {tuple(x.shape)}"
#             )

#         residual = x
#         projected, gate = self.in_proj(
#             self.norm(x)
#         ).chunk(2, dim=-1)

#         local = self.local_conv(
#             F.pad(
#                 projected.transpose(1, 2),
#                 (self.d_conv - 1, 0),
#             )
#         ).transpose(1, 2)

#         state = self.ssm(F.silu(local))
#         update = self.out_proj(
#             state * F.silu(gate)
#         )
#         update = self.dropout(update)
#         scale = torch.sigmoid(
#             self.residual_logit
#         ).to(dtype=update.dtype)
#         return residual + scale * update


# # Compatibility aliases for old imports only.
# # Do not describe this block as Mamba in the paper.
# MambaBlock = StableSSMBlock
# S4DKernel = ChannelwiseS4DKernel


# # =============================================================================
# # PCA-component encoder without false local adjacency
# # =============================================================================


# class ComponentSetBlock(nn.Module):
#     """Global component-context block without neighbouring-component kernels.

#     PCA components have stable identities after the base-fitted PCA transform,
#     but adjacent component indices are not adjacent wavelengths. This block
#     permits global component interaction through shared summary statistics
#     instead of causal/local convolution over component order. Learned component
#     identities retain the fixed PCA basis semantics, so the block is not claimed
#     to be permutation invariant.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         expand: int = 2,
#         dropout: float = 0.1,
#         residual_scale_init: float = 0.50,
#         norm_type: str = "layer",
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         hidden = self.d_model * int(expand)
#         if hidden <= 0:
#             raise ValueError("ComponentSetBlock hidden width must be positive")

#         if str(norm_type).strip().lower() == "rms":
#             self.norm = RMSNorm(self.d_model)
#         else:
#             self.norm = nn.LayerNorm(self.d_model)

#         self.context_proj = nn.Sequential(
#             nn.Linear(2 * self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.gate = nn.Sequential(
#             nn.Linear(2 * self.d_model, self.d_model),
#             nn.Sigmoid(),
#         )
#         self.token_mlp = nn.Sequential(
#             nn.Linear(self.d_model, hidden),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(hidden, self.d_model),
#         )
#         self.dropout = nn.Dropout(float(dropout))
#         self.residual_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(float(residual_scale_init))
#             )
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 3 or x.size(-1) != self.d_model:
#             raise ValueError(
#                 f"Expected component tokens [B,C,{self.d_model}], "
#                 f"got {tuple(x.shape)}"
#             )

#         normalized = self.norm(x)
#         global_mean = normalized.mean(dim=1)
#         global_rms = normalized.float().square().mean(
#             dim=1
#         ).add(1e-6).sqrt().to(dtype=x.dtype)
#         context = self.context_proj(
#             torch.cat([global_mean, global_rms], dim=-1)
#         ).unsqueeze(1)
#         context = context.expand(-1, x.size(1), -1)

#         gate = self.gate(
#             torch.cat([normalized, context], dim=-1)
#         )
#         update = self.token_mlp(
#             normalized + context
#         ) * gate

#         scale = torch.sigmoid(
#             self.residual_logit
#         ).to(dtype=update.dtype)
#         return x + scale * self.dropout(update)


# # =============================================================================
# # Spectral-spatial stem
# # =============================================================================


# class SpectralSpatialStem(nn.Module):
#     """HSI stem with an explicit amplitude-preserving response path.

#     The normalized convolutional path learns robust spatial context. A parallel
#     unnormalized 1x1 projection preserves first-order sensitivity to center
#     spectral interventions, which GroupNorm alone can partially suppress.

#     Component tokens use center-aware descriptors. No convolution is performed
#     over the PCA-component index.
#     """

#     descriptor_dim = 7

#     def __init__(
#         self,
#         in_bands: int,
#         d_model: int,
#         dropout: float = 0.1,
#         norm_groups: int = 8,
#         amplitude_skip_scale_init: float = 0.15,
#     ) -> None:
#         super().__init__()
#         self.in_bands = int(in_bands)
#         self.d_model = int(d_model)
#         if self.in_bands <= 0 or self.d_model <= 0:
#             raise ValueError("in_bands and d_model must be positive")

#         hidden = max(self.d_model // 2, 32)
#         groups = _valid_group_count(
#             hidden, int(norm_groups)
#         )

#         self.context_stem = nn.Sequential(
#             nn.Conv2d(
#                 self.in_bands,
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
#             nn.GroupNorm(groups, hidden),
#             nn.GELU(),
#             nn.Conv2d(
#                 hidden,
#                 self.d_model,
#                 kernel_size=1,
#                 bias=False,
#             ),
#         )

#         # Direct linear path from processed spectral vector to spatial feature
#         # map. It preserves amplitude and directional response information.
#         self.amplitude_projection = nn.Conv2d(
#             self.in_bands,
#             self.d_model,
#             kernel_size=1,
#             bias=False,
#         )
#         self.amplitude_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(
#                     float(amplitude_skip_scale_init)
#                 )
#             )
#         )

#         component_hidden = max(
#             self.d_model // 2, 32
#         )
#         self.component_embed = nn.Sequential(
#             nn.Linear(
#                 self.descriptor_dim,
#                 component_hidden,
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 component_hidden,
#                 self.d_model,
#             ),
#         )
#         self.spatial_dropout = nn.Dropout(
#             float(dropout)
#         )
#         self.component_dropout = nn.Dropout(
#             float(dropout)
#         )

#     @staticmethod
#     def _component_descriptors(
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         _, _, height, width = x.shape
#         center_h, center_w = height // 2, width // 2

#         center = x[:, :, center_h, center_w]
#         global_mean = x.mean(dim=(-1, -2))
#         global_std = x.float().std(
#             dim=(-1, -2), unbiased=False
#         ).to(dtype=x.dtype)

#         h0 = max(center_h - 1, 0)
#         h1 = min(center_h + 2, height)
#         w0 = max(center_w - 1, 0)
#         w1 = min(center_w + 2, width)
#         local_mean = x[:, :, h0:h1, w0:w1].mean(
#             dim=(-1, -2)
#         )

#         center_global_contrast = center - global_mean
#         local_global_contrast = local_mean - global_mean

#         horizontal = (
#             x[:, :, :, 1:] - x[:, :, :, :-1]
#             if width > 1
#             else torch.zeros_like(x[:, :, :, :1])
#         )
#         vertical = (
#             x[:, :, 1:, :] - x[:, :, :-1, :]
#             if height > 1
#             else torch.zeros_like(x[:, :, :1, :])
#         )
#         gradient = 0.5 * (
#             horizontal.abs().mean(dim=(-1, -2))
#             + vertical.abs().mean(dim=(-1, -2))
#         )

#         return torch.stack(
#             [
#                 center,
#                 local_mean,
#                 global_mean,
#                 global_std,
#                 center_global_contrast,
#                 local_global_contrast,
#                 gradient,
#             ],
#             dim=-1,
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         if x.dim() != 4:
#             raise ValueError(
#                 f"Expected x [B,C,H,W], got {tuple(x.shape)}"
#             )
#         if x.size(1) != self.in_bands:
#             raise ValueError(
#                 f"Expected {self.in_bands} input channels, "
#                 f"got {x.size(1)}"
#             )
#         if not torch.isfinite(x).all():
#             raise RuntimeError(
#                 "Backbone input contains NaN/Inf"
#             )

#         x32 = x.float()
#         context_map = self.context_stem(x32)
#         amplitude_map = self.amplitude_projection(
#             x32
#         )
#         amplitude_scale = torch.sigmoid(
#             self.amplitude_logit
#         ).to(dtype=context_map.dtype)
#         spatial_map = (
#             context_map
#             + amplitude_scale * amplitude_map
#         )

#         spatial_tokens = spatial_map.flatten(
#             2
#         ).transpose(1, 2)
#         descriptors = self._component_descriptors(
#             x32
#         )
#         component_tokens = self.component_embed(
#             descriptors
#         )

#         dropped_component_tokens = self.component_dropout(component_tokens)
#         dropped_spatial_tokens = self.spatial_dropout(spatial_tokens)
#         return {
#             "band_tokens_init": dropped_component_tokens,
#             "component_tokens_init": dropped_component_tokens,
#             "spatial_tokens_init": dropped_spatial_tokens,
#             "component_descriptors": descriptors,
#             "spatial_feature_map": spatial_map,
#             "context_feature_map": context_map,
#             "amplitude_feature_map": amplitude_map,
#             "amplitude_skip_scale":
#                 amplitude_scale.detach(),
#         }


# # =============================================================================
# # Component/physical spectral encoder
# # =============================================================================


# class SpectralSSMEncoder(nn.Module):
#     """Physical-band SSM or PCA-component set encoder.

#     - Physical ordered bands: a bidirectional diagonal SSM is valid.
#     - PCA-reduced components: global set-context blocks are used, because local
#       convolution over neighbouring principal-component indices has no
#       wavelength interpretation.

#     A deterministic observed-patch descriptor is produced by the final backbone.
#     No finite-difference spectral intervention is used by the deployed method.
#     """

#     PHYSICAL_MODE = "physical_wavelength"
#     COMPONENT_MODE = "reduced_component"

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(args.d_model)
#         self.num_bands = int(args.num_bands)

#         explicit_mode = getattr(
#             args, "model_input_axis_mode", None
#         )
#         explicit_flag = getattr(
#             args,
#             "model_input_spectral_axis_is_physical",
#             None,
#         )

#         if explicit_mode is not None:
#             token = str(explicit_mode).strip().lower()
#             if token in {
#                 "physical",
#                 "physical_wavelength",
#                 "raw_band",
#                 "raw_bands",
#             }:
#                 self.axis_mode = self.PHYSICAL_MODE
#             elif token in {
#                 "component",
#                 "components",
#                 "pca",
#                 "reduced_component",
#             }:
#                 self.axis_mode = self.COMPONENT_MODE
#             else:
#                 raise ValueError(
#                     "model_input_axis_mode must be physical_wavelength "
#                     "or reduced_component"
#                 )
#         elif explicit_flag is not None:
#             self.axis_mode = (
#                 self.PHYSICAL_MODE
#                 if bool(explicit_flag)
#                 else self.COMPONENT_MODE
#             )
#         else:
#             raw_bands = int(
#                 getattr(
#                     args,
#                     "raw_num_bands",
#                     self.num_bands,
#                 )
#             )
#             pca_components = int(
#                 getattr(
#                     args,
#                     "pca_components",
#                     self.num_bands,
#                 )
#             )
#             self.axis_mode = (
#                 self.COMPONENT_MODE
#                 if pca_components < raw_bands
#                 else self.PHYSICAL_MODE
#             )

#         self.input_axis_is_physical = (
#             self.axis_mode == self.PHYSICAL_MODE
#         )
#         self.component_identity = nn.Parameter(
#             torch.randn(
#                 1,
#                 self.num_bands,
#                 self.d_model,
#             )
#             * 0.02
#         )

#         norm_type = str(
#             getattr(args, "backbone_norm", "layer")
#         ).lower()
#         residual_scale_init = float(
#             getattr(
#                 args,
#                 "ssm_residual_scale_init",
#                 0.70,
#             )
#         )
#         component_residual_init = float(
#             getattr(
#                 args,
#                 "component_residual_scale_init",
#                 0.50,
#             )
#         )
#         layer_count = int(
#             getattr(args, "num_spectral_layers", 2)
#         )

#         if self.input_axis_is_physical:
#             self.physical_layers = nn.ModuleList(
#                 [
#                     StableSSMBlock(
#                         self.d_model,
#                         int(args.d_state),
#                         d_conv=int(
#                             getattr(args, "d_conv", 4)
#                         ),
#                         expand=int(
#                             getattr(args, "expand", 2)
#                         ),
#                         dropout=float(args.dropout),
#                         norm_type=norm_type,
#                         residual_scale_init=
#                             residual_scale_init,
#                         kernel_dropout=float(
#                             getattr(
#                                 args,
#                                 "ssm_kernel_dropout",
#                                 0.0,
#                             )
#                         ),
#                     )
#                     for _ in range(layer_count)
#                 ]
#             )
#             self.component_layers = nn.ModuleList()
#         else:
#             self.physical_layers = nn.ModuleList()
#             self.component_layers = nn.ModuleList(
#                 [
#                     ComponentSetBlock(
#                         self.d_model,
#                         expand=int(
#                             getattr(args, "expand", 2)
#                         ),
#                         dropout=float(args.dropout),
#                         residual_scale_init=
#                             component_residual_init,
#                         norm_type=norm_type,
#                     )
#                     for _ in range(layer_count)
#                 ]
#             )

#         hidden = max(self.d_model // 2, 32)
#         self.channel_importance = nn.Sequential(
#             nn.Linear(
#                 self.d_model, hidden
#             ),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#     @property
#     def layers(self) -> nn.ModuleList:
#         """Compatibility view for old code that inspects `.layers`."""
#         return (
#             self.physical_layers
#             if self.input_axis_is_physical
#             else self.component_layers
#         )

#     def _encode_physical(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         paired = torch.cat(
#             [x, torch.flip(x, dims=[1])],
#             dim=0,
#         )
#         for layer in self.physical_layers:
#             paired = layer(paired)
#         forward, reverse = paired.chunk(2, dim=0)
#         reverse = torch.flip(reverse, dims=[1])
#         return 0.5 * (forward + reverse)

#     def _encode_components(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         for layer in self.component_layers:
#             x = layer(x)
#         return x

#     def forward(
#         self,
#         band_tokens_init: torch.Tensor,
#     ) -> Tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         if band_tokens_init.dim() != 3:
#             raise ValueError(
#                 "Expected channel/component tokens [B,C,D], "
#                 f"got {tuple(band_tokens_init.shape)}"
#             )
#         _, channels, width = band_tokens_init.shape
#         if (
#             channels != self.num_bands
#             or width != self.d_model
#         ):
#             raise ValueError(
#                 f"Expected [B,{self.num_bands},{self.d_model}], "
#                 f"got {tuple(band_tokens_init.shape)}"
#             )

#         x = (
#             band_tokens_init
#             + self.component_identity
#         )
#         mixed = (
#             self._encode_physical(x)
#             if self.input_axis_is_physical
#             else self._encode_components(x)
#         )
#         tokens = self.norm(mixed)

#         importance_input = F.normalize(
#             tokens, dim=-1, eps=1e-6
#         )
#         logits = self.channel_importance(
#             importance_input
#         ).squeeze(-1)
#         weights = F.softmax(logits, dim=-1)

#         # Pool the unnormalized residual stream so radial information remains
#         # available to the canonical low-rank Gaussian geometry.
#         features = torch.sum(
#             mixed * weights.unsqueeze(-1),
#             dim=1,
#         )
#         return features, weights, tokens


# # =============================================================================
# # Spatial encoder
# # =============================================================================


# class SpatialSSMEncoder(nn.Module):
#     """Four-direction raster/column diagonal-SSM encoder.

#     All directions share SSM parameters. A learned global softmax combines
#     directions and is initialized uniformly, retaining symmetry at startup.
#     """

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()
#         self.patch_size = int(args.patch_size)
#         self.d_model = int(args.d_model)
#         self.num_patches = (
#             self.patch_size * self.patch_size
#         )
#         if self.patch_size <= 0:
#             raise ValueError("patch_size must be positive")

#         self.row_embed = nn.Parameter(
#             torch.randn(
#                 1,
#                 self.patch_size,
#                 self.d_model,
#             )
#             * 0.02
#         )
#         self.col_embed = nn.Parameter(
#             torch.randn(
#                 1,
#                 self.patch_size,
#                 self.d_model,
#             )
#             * 0.02
#         )

#         norm_type = str(
#             getattr(args, "backbone_norm", "layer")
#         ).lower()
#         residual_scale_init = float(
#             getattr(
#                 args,
#                 "ssm_residual_scale_init",
#                 0.70,
#             )
#         )
#         self.layers = nn.ModuleList(
#             [
#                 StableSSMBlock(
#                     self.d_model,
#                     int(args.d_state),
#                     d_conv=int(
#                         getattr(args, "d_conv", 4)
#                     ),
#                     expand=int(
#                         getattr(args, "expand", 2)
#                     ),
#                     dropout=float(args.dropout),
#                     norm_type=norm_type,
#                     residual_scale_init=
#                         residual_scale_init,
#                     kernel_dropout=float(
#                         getattr(
#                             args,
#                             "ssm_kernel_dropout",
#                             0.0,
#                         )
#                     ),
#                 )
#                 for _ in range(
#                     int(args.num_layers)
#                 )
#             ]
#         )
#         self.norm = nn.LayerNorm(
#             self.d_model
#         )

#         hidden = max(
#             self.d_model // 2, 32
#         )
#         self.spatial_importance = nn.Sequential(
#             nn.Linear(
#                 self.d_model, hidden
#             ),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#         )

#         grid = torch.arange(
#             self.num_patches,
#             dtype=torch.long,
#         ).reshape(
#             self.patch_size,
#             self.patch_size,
#         )
#         orders = torch.stack(
#             [
#                 grid.flatten(),
#                 torch.flip(
#                     grid.flatten(), dims=[0]
#                 ),
#                 grid.transpose(
#                     0, 1
#                 ).flatten(),
#                 torch.flip(
#                     grid.transpose(
#                         0, 1
#                     ).flatten(),
#                     dims=[0],
#                 ),
#             ],
#             dim=0,
#         )
#         inverse = torch.argsort(
#             orders, dim=1
#         )
#         self.register_buffer(
#             "scan_orders",
#             orders,
#             persistent=False,
#         )
#         self.register_buffer(
#             "inverse_scan_orders",
#             inverse,
#             persistent=False,
#         )
#         self.direction_logits = nn.Parameter(
#             torch.zeros(orders.size(0))
#         )

#     def _encode(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def _position_encoding(
#         self,
#         dtype: torch.dtype,
#     ) -> torch.Tensor:
#         position = (
#             self.row_embed[:, :, None, :]
#             + self.col_embed[:, None, :, :]
#         )
#         return position.reshape(
#             1,
#             self.num_patches,
#             self.d_model,
#         ).to(dtype=dtype)

#     def forward(
#         self,
#         spatial_tokens_init: torch.Tensor,
#     ) -> Tuple[
#         torch.Tensor,
#         Dict[str, torch.Tensor],
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         if spatial_tokens_init.dim() != 3:
#             raise ValueError(
#                 "Expected spatial tokens [B,HW,D], "
#                 f"got {tuple(spatial_tokens_init.shape)}"
#             )
#         _, count, width = (
#             spatial_tokens_init.shape
#         )
#         if (
#             count != self.num_patches
#             or width != self.d_model
#         ):
#             raise ValueError(
#                 f"Expected [B,{self.num_patches},{self.d_model}], "
#                 f"got {tuple(spatial_tokens_init.shape)}"
#             )

#         x = (
#             spatial_tokens_init
#             + self._position_encoding(
#                 spatial_tokens_init.dtype
#             )
#         )
#         batch = x.size(0)
#         ordered = torch.cat(
#             [
#                 x.index_select(1, order)
#                 for order in self.scan_orders
#             ],
#             dim=0,
#         )
#         encoded = self._encode(ordered)
#         encoded_by_direction = encoded.reshape(
#             self.scan_orders.size(0),
#             batch,
#             self.num_patches,
#             self.d_model,
#         )
#         restored = torch.stack(
#             [
#                 encoded_by_direction[
#                     direction
#                 ].index_select(
#                     1,
#                     self.inverse_scan_orders[
#                         direction
#                     ],
#                 )
#                 for direction in range(
#                     self.scan_orders.size(0)
#                 )
#             ],
#             dim=0,
#         )
#         direction_weights = F.softmax(
#             self.direction_logits,
#             dim=0,
#         ).to(dtype=restored.dtype)
#         mixed = torch.sum(
#             restored
#             * direction_weights.view(
#                 -1, 1, 1, 1
#             ),
#             dim=0,
#         )
#         tokens = self.norm(mixed)

#         importance_input = F.normalize(
#             tokens, dim=-1, eps=1e-6
#         )
#         logits = self.spatial_importance(
#             importance_input
#         ).squeeze(-1)
#         weights = F.softmax(logits, dim=-1)
#         features = torch.sum(
#             mixed * weights.unsqueeze(-1),
#             dim=1,
#         )

#         patterns = {
#             "direction_weights":
#                 direction_weights.detach(),
#         }
#         return (
#             features,
#             patterns,
#             tokens,
#             weights,
#         )


# # =============================================================================
# # Scale-preserving fusion
# # =============================================================================


# class TokenFusion(nn.Module):
#     """Cross-conditioned fusion without final feature normalization."""

#     def __init__(
#         self,
#         d_model: int,
#         dropout: float = 0.1,
#         residual_scale: float = 0.30,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.spec_norm = nn.LayerNorm(
#             self.d_model
#         )
#         self.spat_norm = nn.LayerNorm(
#             self.d_model
#         )
#         self.spec_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )
#         self.spat_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )

#         self.cross_gate = nn.Sequential(
#             nn.Linear(
#                 4 * self.d_model,
#                 self.d_model,
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 self.d_model,
#                 self.d_model,
#             ),
#             nn.Sigmoid(),
#         )
#         self.feature_mlp = nn.Sequential(
#             nn.Linear(
#                 2 * self.d_model,
#                 self.d_model,
#             ),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(
#                 self.d_model,
#                 self.d_model,
#             ),
#         )
#         self.token_norm = nn.LayerNorm(
#             self.d_model
#         )
#         self.dropout = nn.Dropout(
#             float(dropout)
#         )
#         self.fusion_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(
#                     float(residual_scale)
#                 )
#             )
#         )

#     def forward(
#         self,
#         spectral_tokens: torch.Tensor,
#         spatial_tokens: torch.Tensor,
#         spectral_features: torch.Tensor | None = None,
#         spatial_features: torch.Tensor | None = None,
#     ) -> Tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         if (
#             spectral_tokens.dim() != 3
#             or spatial_tokens.dim() != 3
#         ):
#             raise ValueError(
#                 "Fusion expects two token tensors"
#             )
#         if (
#             spectral_tokens.size(0)
#             != spatial_tokens.size(0)
#         ):
#             raise ValueError(
#                 "Fusion branch batch sizes differ"
#             )
#         if (
#             spectral_tokens.size(-1)
#             != self.d_model
#             or spatial_tokens.size(-1)
#             != self.d_model
#         ):
#             raise ValueError(
#                 "Fusion branch widths differ"
#             )

#         spectral = (
#             spectral_tokens.mean(dim=1)
#             if spectral_features is None
#             else spectral_features
#         )
#         spatial = (
#             spatial_tokens.mean(dim=1)
#             if spatial_features is None
#             else spatial_features
#         )

#         spectral_view = self.spec_proj(
#             self.spec_norm(spectral)
#         )
#         spatial_view = self.spat_proj(
#             self.spat_norm(spatial)
#         )
#         gate_input = torch.cat(
#             [
#                 spectral_view,
#                 spatial_view,
#                 torch.abs(
#                     spectral_view
#                     - spatial_view
#                 ),
#                 spectral_view * spatial_view,
#             ],
#             dim=-1,
#         )
#         gate = self.cross_gate(gate_input)

#         # Preserve raw branch amplitudes in the base stream.
#         base = (
#             gate * spectral
#             + (1.0 - gate) * spatial
#         )
#         delta = self.feature_mlp(
#             torch.cat(
#                 [spectral_view, spatial_view],
#                 dim=-1,
#             )
#         )
#         scale = torch.sigmoid(
#             self.fusion_logit
#         ).to(dtype=delta.dtype)
#         fused_feature = (
#             base
#             + scale * self.dropout(delta)
#         )

#         # Token normalization is diagnostic/auxiliary only. The feature used by
#         # geometry is intentionally not normalized.
#         fused_tokens = self.token_norm(
#             torch.cat(
#                 [spectral_tokens, spatial_tokens],
#                 dim=1,
#             )
#         )
#         return (
#             fused_feature,
#             fused_tokens,
#             gate,
#         )



# # =============================================================================
# # Fixed HSI spectral descriptor
# # =============================================================================


# class FixedHSISpectralDescriptor(nn.Module):
#     """Deterministic spectral object used by the conditional GeometryBank.

#     The descriptor is computed directly from the observed preprocessed HSI
#     patch and contains no learned parameters. For both physical bands and PCA
#     components it contains the center vector and center-to-local contrast. A
#     first spectral difference is added only when the channel axis represents
#     ordered physical wavelengths.

#     This distinction prevents the model from imposing false wavelength
#     locality on adjacent PCA components.
#     """

#     VERSION = 1

#     def __init__(
#         self,
#         channels: int,
#         *,
#         axis_mode: str,
#         local_radius: int = 1,
#         purity_floor: float = 0.05,
#         purity_temperature: float = 1.0,
#     ) -> None:
#         super().__init__()
#         self.channels = int(channels)
#         self.axis_mode = str(axis_mode)
#         self.local_radius = int(max(0, local_radius))
#         self.purity_floor = float(min(max(purity_floor, 0.0), 1.0))
#         self.purity_temperature = float(max(purity_temperature, 1e-6))
#         self.input_axis_is_physical = (
#             self.axis_mode == SpectralSSMEncoder.PHYSICAL_MODE
#         )
#         if self.channels <= 0:
#             raise ValueError("channels must be positive")

#         base_width = 2 * self.channels
#         derivative_width = self.channels - 1 if self.input_axis_is_physical else 0
#         self.output_dim = int(base_width + derivative_width)

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         if x.dim() != 4 or x.size(1) != self.channels:
#             raise ValueError(
#                 f"descriptor expects [B,{self.channels},H,W], got {tuple(x.shape)}"
#             )
#         if not torch.isfinite(x).all():
#             raise RuntimeError("descriptor input contains NaN/Inf")

#         values = x.float()
#         _, _, height, width = values.shape
#         center_h, center_w = height // 2, width // 2
#         center = values[:, :, center_h, center_w]

#         radius = self.local_radius
#         h0 = max(center_h - radius, 0)
#         h1 = min(center_h + radius + 1, height)
#         w0 = max(center_w - radius, 0)
#         w1 = min(center_w + radius + 1, width)
#         local_mean = values[:, :, h0:h1, w0:w1].mean(dim=(-1, -2))
#         global_std = values.std(dim=(-1, -2), unbiased=False).clamp_min(1e-6)
#         center_local = center - local_mean

#         blocks = [center, center_local]
#         if self.input_axis_is_physical:
#             spectral_difference = center[:, 1:] - center[:, :-1]
#             blocks.append(spectral_difference)
#         descriptor = torch.cat(blocks, dim=1)
#         if descriptor.size(1) != self.output_dim:
#             raise RuntimeError("spectral descriptor width changed unexpectedly")

#         normalized_disagreement = center_local.abs() / global_std
#         mixedness = normalized_disagreement.mean(dim=1)
#         purity = torch.exp(-mixedness / self.purity_temperature).clamp(
#             self.purity_floor, 1.0
#         )
#         return {
#             "spectral_descriptors": descriptor,
#             "spatial_purity": purity,
#             "center_spectrum": center,
#             "local_mean_spectrum": local_mean,
#             "center_local_contrast": center_local,
#         }

#     def contract(self) -> Dict[str, Any]:
#         return {
#             "version": self.VERSION,
#             "axis_mode": self.axis_mode,
#             "output_dim": self.output_dim,
#             "blocks": (
#                 ["center", "center_local_contrast", "first_spectral_difference"]
#                 if self.input_axis_is_physical
#                 else ["center_component", "center_local_component_contrast"]
#             ),
#             "learned": False,
#             "phase_invariant": True,
#             "uses_false_pca_adjacency": False,
#         }


# # =============================================================================
# # Final stable-coordinate backbone
# # =============================================================================


# class SSMBackbone(nn.Module):
#     """HSI bridge encoder aligned with immutable low-rank GeometryBank rows.

#     The deployed geometry feature space is fixed after phase 0. This is a
#     necessary contract while old bank rows remain immutable. Controlled feature
#     plasticity must not be added here unless the implementation also transports
#     or rebuilds every old aggregate row in the new coordinate system.

#     The backbone additionally emits a deterministic direct spectral descriptor
#     and a patch-purity score used only for robust GeometryBank row estimation.
#     """

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.d_model = int(args.d_model)
#         self.num_bands = int(args.num_bands)
#         self.patch_size = int(args.patch_size)
#         self._incremental_frozen = False
#         self._fully_frozen = False
#         self._partial_training_modules: List[nn.Module] = []

#         self.stem = SpectralSpatialStem(
#             in_bands=self.num_bands,
#             d_model=self.d_model,
#             dropout=float(args.dropout),
#             norm_groups=int(getattr(args, "stem_norm_groups", 8)),
#             amplitude_skip_scale_init=float(
#                 getattr(args, "amplitude_skip_scale_init", 0.15)
#             ),
#         )
#         self.spectral_encoder = SpectralSSMEncoder(args)
#         self.spatial_encoder = SpatialSSMEncoder(args)
#         self.token_fusion = TokenFusion(
#             self.d_model,
#             dropout=float(args.dropout),
#             residual_scale=float(getattr(args, "fusion_residual_scale", 0.30)),
#         )
#         self.output_dropout = nn.Dropout(
#             float(getattr(args, "backbone_output_dropout", 0.0))
#         )
#         self.raw_num_bands = int(getattr(args, "raw_num_bands", self.num_bands))
#         configured_source = getattr(args, "spectral_descriptor_source", None)
#         if configured_source is None:
#             configured_source = (
#                 "raw_spectrum"
#                 if self.raw_num_bands != self.num_bands
#                 else "model_input"
#             )
#         self.spectral_descriptor_source = str(configured_source).strip().lower()
#         if self.spectral_descriptor_source not in {"raw_spectrum", "model_input"}:
#             raise ValueError(
#                 "spectral_descriptor_source must be raw_spectrum or model_input"
#             )
#         if self.spectral_descriptor_source == "raw_spectrum":
#             descriptor_channels = self.raw_num_bands
#             descriptor_axis_mode = SpectralSSMEncoder.PHYSICAL_MODE
#         else:
#             descriptor_channels = self.num_bands
#             descriptor_axis_mode = self.spectral_encoder.axis_mode

#         self.spectral_descriptor = FixedHSISpectralDescriptor(
#             descriptor_channels,
#             axis_mode=descriptor_axis_mode,
#             local_radius=int(getattr(args, "descriptor_local_radius", 1)),
#             purity_floor=float(getattr(args, "spatial_purity_floor", 0.05)),
#             purity_temperature=float(
#                 getattr(args, "spatial_purity_temperature", 1.0)
#             ),
#         )
#         self.spectral_descriptor_dim = self.spectral_descriptor.output_dim
#         self.spectral_descriptor_version = self.spectral_descriptor.VERSION

#     # ------------------------------------------------------------------
#     # Parameter and mode policy
#     # ------------------------------------------------------------------
#     def train(self, mode: bool = True) -> "SSMBackbone":
#         requested = bool(mode)
#         if self._fully_frozen or self._incremental_frozen:
#             super().train(False)
#             return self
#         if self._partial_training_modules:
#             super().train(False)
#             for module in self._partial_training_modules:
#                 module.train(requested)
#             return self
#         super().train(requested)
#         return self

#     @staticmethod
#     def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
#         for parameter in module.parameters():
#             parameter.requires_grad_(bool(trainable))
#             parameter.grad = None

#     def enable_base_training(self) -> None:
#         self._incremental_frozen = False
#         self._fully_frozen = False
#         self._partial_training_modules.clear()
#         for parameter in self.parameters():
#             parameter.requires_grad_(True)
#             parameter.grad = None
#         self.train(True)

#     def freeze_for_incremental(self) -> None:
#         """Freeze the complete deployed feature map after base handoff."""
#         self._incremental_frozen = True
#         self._fully_frozen = False
#         self._partial_training_modules.clear()
#         for parameter in self.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None
#         super().train(False)

#     def enable_incremental_plasticity(self) -> None:
#         raise RuntimeError(
#             "Incremental backbone plasticity is incompatible with immutable "
#             "GeometryBank rows. Add an explicit aggregate-row transport/rebuild "
#             "mechanism before enabling this path."
#         )

#     def freeze_all(self) -> None:
#         self._incremental_frozen = False
#         self._fully_frozen = True
#         self._partial_training_modules.clear()
#         for parameter in self.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None
#         super().train(False)

#     def incremental_trainable_parameters(self) -> List[nn.Parameter]:
#         if self._incremental_frozen:
#             return []
#         return [p for p in self.parameters() if p.requires_grad]

#     def assert_frozen_for_incremental(self) -> None:
#         if not self._incremental_frozen:
#             raise RuntimeError("incremental freeze flag is not active")
#         trainable = [name for name, p in self.named_parameters() if p.requires_grad]
#         gradients = [name for name, p in self.named_parameters() if p.grad is not None]
#         active_modules = [
#             module.__class__.__name__ for module in self.modules() if module.training
#         ]
#         violations: List[str] = []
#         if trainable:
#             violations.append(f"trainable_parameters={trainable[:8]}")
#         if gradients:
#             violations.append(f"retained_gradients={gradients[:8]}")
#         if active_modules:
#             violations.append(f"training_modules={active_modules[:8]}")
#         if violations:
#             raise RuntimeError(
#                 "invalid frozen incremental backbone: " + "; ".join(violations)
#             )

#     def get_last_blocks(self) -> List[nn.Module]:
#         blocks: List[nn.Module] = []
#         if len(self.spectral_encoder.layers):
#             blocks.append(self.spectral_encoder.layers[-1])
#         if len(self.spatial_encoder.layers):
#             blocks.append(self.spatial_encoder.layers[-1])
#         return blocks

#     def unfreeze_last_blocks(self, *, allow_ablation: bool = False) -> None:
#         if not allow_ablation:
#             raise RuntimeError("last-block unfreezing is an explicit ablation only")
#         self.freeze_all()
#         self._fully_frozen = False
#         self._partial_training_modules = self.get_last_blocks()
#         for block in self._partial_training_modules:
#             self._set_module_trainable(block, True)
#         self.train(True)

#     def unfreeze_token_fusion(self, *, allow_ablation: bool = False) -> None:
#         if not allow_ablation:
#             raise RuntimeError("fusion unfreezing is an explicit ablation only")
#         self.freeze_all()
#         self._fully_frozen = False
#         self._partial_training_modules = [self.token_fusion]
#         self._set_module_trainable(self.token_fusion, True)
#         self.train(True)

#     # ------------------------------------------------------------------
#     # Contracts and statistics
#     # ------------------------------------------------------------------
#     def backbone_contract(self) -> Dict[str, Any]:
#         return {
#             "input_shape": "[B,C,H,W]",
#             "input_channel_role": self.spectral_encoder.axis_mode,
#             "component_encoder": (
#                 "global_component_context"
#                 if not self.spectral_encoder.input_axis_is_physical
#                 else "bidirectional_diagonal_ssm"
#             ),
#             "false_pca_wavelength_locality": False,
#             "ssm_family": "channelwise_diagonal_s4d",
#             "is_mamba_selective_scan": False,
#             "spatial_scan_directions": 4,
#             "batch_dependent_normalization": False,
#             "amplitude_preserving_stem_path": True,
#             "geometry_feature_space": "stable_unnormalized_euclidean_z",
#             "feature_space_immutable_after_base": self._incremental_frozen,
#             "strict_incremental_backbone_frozen": self._incremental_frozen,
#             "spectral_descriptor_source": self.spectral_descriptor_source,
#             "spectral_descriptor": self.spectral_descriptor.contract(),
#             "uses_spectral_interventions": False,
#             "uses_task_specific_adapters": False,
#             "stores_exemplars": False,
#         }

#     @staticmethod
#     @torch.no_grad()
#     def feature_statistics(features: torch.Tensor) -> Dict[str, torch.Tensor]:
#         if features.dim() != 2:
#             raise ValueError("features must be [B,D]")
#         norms = features.norm(dim=-1)
#         return {
#             "feature_norm_mean": norms.mean().detach(),
#             "feature_norm_std": norms.std(unbiased=False).detach(),
#             "feature_norm_cv": (
#                 norms.std(unbiased=False) / norms.mean().clamp_min(1e-8)
#             ).detach(),
#             "feature_abs_mean": features.abs().mean().detach(),
#             "feature_dimension_variance_mean": features.var(
#                 dim=0, unbiased=False
#             ).mean().detach(),
#         }

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------
#     def _validate_patch_input(self, x: torch.Tensor) -> None:
#         if not torch.is_tensor(x):
#             raise TypeError("x must be a tensor")
#         if x.dim() != 4:
#             raise ValueError(f"expected x [B,C,H,W], got {tuple(x.shape)}")
#         _, channels, height, width = x.shape
#         if channels != self.num_bands:
#             raise ValueError(f"expected {self.num_bands} channels, got {channels}")
#         if height != self.patch_size or width != self.patch_size:
#             raise ValueError(
#                 f"expected {self.patch_size}x{self.patch_size}, got {height}x{width}"
#             )
#         if not torch.isfinite(x).all():
#             raise RuntimeError("backbone input contains NaN/Inf")

#     def _resolve_descriptor_input(
#         self,
#         x: torch.Tensor,
#         raw_spectral_patch: torch.Tensor | None,
#     ) -> torch.Tensor:
#         if self.spectral_descriptor_source == "model_input":
#             return x
#         if raw_spectral_patch is None:
#             raise RuntimeError(
#                 "raw_spectral_patch is required because the main HSI descriptor "
#                 "is computed in the ordered physical-band space"
#             )
#         if raw_spectral_patch.dim() != 4:
#             raise ValueError("raw_spectral_patch must be [B,B_raw,H,W]")
#         expected = (
#             x.size(0),
#             self.raw_num_bands,
#             self.patch_size,
#             self.patch_size,
#         )
#         if tuple(raw_spectral_patch.shape) != expected:
#             raise ValueError(
#                 f"raw_spectral_patch must be {expected}, got "
#                 f"{tuple(raw_spectral_patch.shape)}"
#             )
#         if not torch.isfinite(raw_spectral_patch).all():
#             raise RuntimeError("raw_spectral_patch contains NaN/Inf")
#         return raw_spectral_patch

#     def _forward_impl(
#         self,
#         x: torch.Tensor,
#         raw_spectral_patch: torch.Tensor | None = None,
#     ) -> Dict[str, Any]:
#         self._validate_patch_input(x)
#         descriptor_input = self._resolve_descriptor_input(x, raw_spectral_patch)
#         direct = self.spectral_descriptor(descriptor_input)
#         stem = self.stem(x)
#         component_features, component_weights, component_tokens = (
#             self.spectral_encoder(stem["component_tokens_init"])
#         )
#         (
#             spatial_features,
#             spatial_patterns,
#             spatial_tokens,
#             spatial_weights,
#         ) = self.spatial_encoder(stem["spatial_tokens_init"])
#         features, fused_tokens, fusion_gate = self.token_fusion(
#             component_tokens,
#             spatial_tokens,
#             spectral_features=component_features,
#             spatial_features=spatial_features,
#         )
#         features = self.output_dropout(features)
#         if not torch.isfinite(features).all():
#             raise RuntimeError("backbone produced NaN/Inf features")

#         return {
#             "features": features,
#             "geometry_features": features,
#             "bridge_features": features,
#             "spectral_descriptors": direct["spectral_descriptors"],
#             "spatial_purity": direct["spatial_purity"],
#             "center_spectrum": direct["center_spectrum"],
#             "local_mean_spectrum": direct["local_mean_spectrum"],
#             "center_local_contrast": direct["center_local_contrast"],
#             "spectral_features": component_features,
#             "component_features": component_features,
#             "spatial_features": spatial_features,
#             "band_weights": component_weights,
#             "component_weights": component_weights,
#             "spatial_weights": spatial_weights,
#             "fusion_gate": fusion_gate,
#             "spatial_patterns": spatial_patterns,
#             "spectral_tokens": component_tokens,
#             "component_tokens": component_tokens,
#             "spatial_tokens": spatial_tokens,
#             "fused_tokens": fused_tokens,
#             "component_descriptors": stem["component_descriptors"],
#             "amplitude_skip_scale": stem["amplitude_skip_scale"],
#             "feature_stats": self.feature_statistics(features),
#             "backbone_contract": self.backbone_contract(),
#         }

#     def forward(
#         self,
#         x: torch.Tensor,
#         raw_spectral_patch: torch.Tensor | None = None,
#     ) -> Dict[str, Any]:
#         return self._forward_impl(x, raw_spectral_patch)

#     def forward_geometry(
#         self,
#         x: torch.Tensor,
#         raw_spectral_patch: torch.Tensor | None = None,
#     ) -> Dict[str, Any]:
#         """Deterministic geometry/descriptor forward with gradients preserved."""
#         with _temporary_eval_mode(self):
#             return self._forward_impl(x, raw_spectral_patch)

#     def encode_bridge(
#         self,
#         x: torch.Tensor,
#         *,
#         raw_spectral_patch: torch.Tensor | None = None,
#         deterministic: bool = False,
#     ) -> Dict[str, Any]:
#         if deterministic:
#             return self.forward_geometry(x, raw_spectral_patch)
#         return self.forward(x, raw_spectral_patch)

#     forward_bridge = forward_geometry















# from __future__ import annotations

# import math
# from contextlib import contextmanager
# from typing import Any, Dict, Iterator, List, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # =============================================================================
# # Shared utilities
# # =============================================================================


# def _valid_group_count(channels: int, requested: int) -> int:
#     """Return the largest requested-or-smaller divisor for GroupNorm."""
#     channels = int(channels)
#     requested = max(1, min(int(requested), channels))
#     for groups in range(requested, 0, -1):
#         if channels % groups == 0:
#             return groups
#     return 1


# def _inverse_sigmoid(value: float) -> float:
#     value = min(max(float(value), 1e-4), 1.0 - 1e-4)
#     return math.log(value / (1.0 - value))


# @contextmanager
# def _temporary_eval_mode(module: nn.Module) -> Iterator[None]:
#     """Disable stochastic/state-updating behavior without disabling gradients."""
#     states = {child: child.training for child in module.modules()}
#     try:
#         module.eval()
#         yield
#     finally:
#         # Direct assignment avoids recursively changing children multiple times.
#         for child, training in states.items():
#             child.training = training


# class RMSNorm(nn.Module):
#     """RMS normalization without batch-dependent state."""

#     def __init__(self, d_model: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.eps = float(eps)
#         self.weight = nn.Parameter(torch.ones(int(d_model)))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.size(-1) != self.weight.numel():
#             raise RuntimeError(
#                 f"RMSNorm expected last dimension {self.weight.numel()}, "
#                 f"got {x.size(-1)}"
#             )
#         rms = x.float().square().mean(dim=-1, keepdim=True)
#         rms = rms.add(self.eps).sqrt().to(dtype=x.dtype)
#         return (x / rms) * self.weight


# # =============================================================================
# # Stable diagonal state-space blocks
# # =============================================================================


# class ChannelwiseS4DKernel(nn.Module):
#     """Stable channel-specific diagonal state-space convolution.

#     This is an S4D-style diagonal state-space kernel. It is not a Mamba
#     selective scan: its state parameters do not depend on the current input.
#     The explicit name prevents the paper from claiming a mechanism that the
#     implementation does not contain.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         d_state: int = 16,
#         dt_min: float = 1e-3,
#         dt_max: float = 1e-1,
#         kernel_dropout: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.d_state = int(d_state)
#         self.kernel_dropout = float(kernel_dropout)

#         if self.d_model <= 0 or self.d_state <= 0:
#             raise ValueError("d_model and d_state must be positive")
#         if not 0.0 < float(dt_min) < float(dt_max):
#             raise ValueError("Require 0 < dt_min < dt_max")
#         if not 0.0 <= self.kernel_dropout < 1.0:
#             raise ValueError("kernel_dropout must lie in [0,1)")

#         # Stable continuous poles A=-exp(A_log).
#         base = torch.arange(1, self.d_state + 1, dtype=torch.float32)
#         self.A_log = nn.Parameter(base.log().repeat(self.d_model, 1))
#         self.B = nn.Parameter(
#             torch.randn(self.d_model, self.d_state) * 0.02
#         )
#         self.C = nn.Parameter(
#             torch.randn(self.d_model, self.d_state) * 0.02
#         )

#         low = math.log(float(dt_min))
#         high = math.log(float(dt_max))
#         self.log_dt = nn.Parameter(
#             torch.empty(self.d_model).uniform_(low, high)
#         )
#         self.D = nn.Parameter(torch.ones(self.d_model))

#     def _kernel(
#         self,
#         length: int,
#         dtype: torch.dtype,
#         device: torch.device,
#     ) -> torch.Tensor:
#         length = int(length)
#         if length <= 0:
#             raise ValueError(f"length must be positive, got {length}")

#         # Kernel construction stays in float32 for stable exponentiation.
#         A = -torch.exp(self.A_log.float()).to(device=device)
#         dt = torch.exp(self.log_dt.float()).clamp(1e-4, 1.0)
#         dt = dt.to(device=device)
#         exponent = (dt[:, None] * A).clamp(min=-60.0, max=-1e-6)

#         discrete_A = torch.exp(exponent)
#         # Zero-order-hold discretization:
#         # B_bar = A^{-1}(exp(dt*A)-I)B.
#         discrete_B = (
#             torch.expm1(exponent)
#             / A
#             * self.B.float().to(device=device)
#         )

#         powers = torch.arange(
#             length, device=device, dtype=torch.float32
#         )
#         transition_powers = torch.pow(
#             discrete_A.unsqueeze(-1),
#             powers.view(1, 1, length),
#         )
#         kernel = torch.einsum(
#             "dn,dn,dnl->dl",
#             self.C.float().to(device=device),
#             discrete_B,
#             transition_powers,
#         )
#         kernel = torch.nan_to_num(
#             kernel, nan=0.0, posinf=0.0, neginf=0.0
#         )
#         if self.training and self.kernel_dropout > 0.0:
#             kernel = F.dropout(
#                 kernel,
#                 p=self.kernel_dropout,
#                 training=True,
#             )
#         return kernel.to(dtype=dtype)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 3:
#             raise ValueError(
#                 f"Expected x [B,L,D], got {tuple(x.shape)}"
#             )
#         _, length, width = x.shape
#         if width != self.d_model:
#             raise ValueError(
#                 f"Expected hidden width {self.d_model}, got {width}"
#             )

#         kernel = self._kernel(
#             length, x.dtype, x.device
#         ).unsqueeze(1)
#         sequence = x.transpose(1, 2)
#         filtered = F.conv1d(
#             F.pad(sequence, (length - 1, 0)),
#             kernel,
#             groups=self.d_model,
#         )
#         filtered = filtered[..., :length].transpose(1, 2)
#         skip = self.D.to(
#             device=x.device, dtype=x.dtype
#         ).view(1, 1, -1)
#         return filtered + skip * x


# class StableSSMBlock(nn.Module):
#     """Pre-normalized gated diagonal-SSM block.

#     The residual update is bounded by a learned sigmoid scalar. This keeps the
#     feature field stable enough for low-rank class geometry while still
#     permitting representation learning during the base phase.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         d_state: int = 16,
#         d_conv: int = 4,
#         expand: int = 2,
#         dropout: float = 0.1,
#         norm_type: str = "layer",
#         residual_scale_init: float = 0.70,
#         kernel_dropout: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.d_inner = int(d_model) * int(expand)
#         self.d_conv = int(d_conv)
#         if self.d_model <= 0 or self.d_inner <= 0 or self.d_conv <= 0:
#             raise ValueError("Invalid SSM block dimensions")

#         if str(norm_type).strip().lower() == "rms":
#             self.norm = RMSNorm(self.d_model)
#         else:
#             self.norm = nn.LayerNorm(self.d_model)

#         self.in_proj = nn.Linear(
#             self.d_model, 2 * self.d_inner, bias=False
#         )
#         self.local_conv = nn.Conv1d(
#             self.d_inner,
#             self.d_inner,
#             kernel_size=self.d_conv,
#             padding=0,
#             groups=self.d_inner,
#             bias=True,
#         )
#         self.ssm = ChannelwiseS4DKernel(
#             self.d_inner,
#             int(d_state),
#             kernel_dropout=kernel_dropout,
#         )
#         self.out_proj = nn.Linear(
#             self.d_inner, self.d_model, bias=False
#         )
#         self.dropout = nn.Dropout(float(dropout))
#         self.residual_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(float(residual_scale_init))
#             )
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 3 or x.size(-1) != self.d_model:
#             raise ValueError(
#                 f"Expected x [B,L,{self.d_model}], "
#                 f"got {tuple(x.shape)}"
#             )

#         residual = x
#         projected, gate = self.in_proj(
#             self.norm(x)
#         ).chunk(2, dim=-1)

#         local = self.local_conv(
#             F.pad(
#                 projected.transpose(1, 2),
#                 (self.d_conv - 1, 0),
#             )
#         ).transpose(1, 2)

#         state = self.ssm(F.silu(local))
#         update = self.out_proj(
#             state * F.silu(gate)
#         )
#         update = self.dropout(update)
#         scale = torch.sigmoid(
#             self.residual_logit
#         ).to(dtype=update.dtype)
#         return residual + scale * update


# # Compatibility aliases for old imports only.
# # Do not describe this block as Mamba in the paper.
# MambaBlock = StableSSMBlock
# S4DKernel = ChannelwiseS4DKernel


# # =============================================================================
# # PCA-component encoder without false local adjacency
# # =============================================================================


# class ComponentSetBlock(nn.Module):
#     """Global component-context block without neighbouring-component kernels.

#     PCA components have stable identities after the base-fitted PCA transform,
#     but adjacent component indices are not adjacent wavelengths. This block
#     permits global component interaction through permutation-equivariant
#     summary statistics instead of causal/local convolution over component
#     order.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         expand: int = 2,
#         dropout: float = 0.1,
#         residual_scale_init: float = 0.50,
#         norm_type: str = "layer",
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         hidden = self.d_model * int(expand)
#         if hidden <= 0:
#             raise ValueError("ComponentSetBlock hidden width must be positive")

#         if str(norm_type).strip().lower() == "rms":
#             self.norm = RMSNorm(self.d_model)
#         else:
#             self.norm = nn.LayerNorm(self.d_model)

#         self.context_proj = nn.Sequential(
#             nn.Linear(2 * self.d_model, self.d_model),
#             nn.GELU(),
#             nn.Linear(self.d_model, self.d_model),
#         )
#         self.gate = nn.Sequential(
#             nn.Linear(2 * self.d_model, self.d_model),
#             nn.Sigmoid(),
#         )
#         self.token_mlp = nn.Sequential(
#             nn.Linear(self.d_model, hidden),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(hidden, self.d_model),
#         )
#         self.dropout = nn.Dropout(float(dropout))
#         self.residual_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(float(residual_scale_init))
#             )
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 3 or x.size(-1) != self.d_model:
#             raise ValueError(
#                 f"Expected component tokens [B,C,{self.d_model}], "
#                 f"got {tuple(x.shape)}"
#             )

#         normalized = self.norm(x)
#         global_mean = normalized.mean(dim=1)
#         global_rms = normalized.float().square().mean(
#             dim=1
#         ).add(1e-6).sqrt().to(dtype=x.dtype)
#         context = self.context_proj(
#             torch.cat([global_mean, global_rms], dim=-1)
#         ).unsqueeze(1)
#         context = context.expand(-1, x.size(1), -1)

#         gate = self.gate(
#             torch.cat([normalized, context], dim=-1)
#         )
#         update = self.token_mlp(
#             normalized + context
#         ) * gate

#         scale = torch.sigmoid(
#             self.residual_logit
#         ).to(dtype=update.dtype)
#         return x + scale * self.dropout(update)


# # =============================================================================
# # Spectral-spatial stem
# # =============================================================================


# class SpectralSpatialStem(nn.Module):
#     """HSI stem with an explicit amplitude-preserving response path.

#     The normalized convolutional path learns robust spatial context. A parallel
#     unnormalized 1x1 projection preserves first-order sensitivity to center
#     spectral interventions, which GroupNorm alone can partially suppress.

#     Component tokens use center-aware descriptors. No convolution is performed
#     over the PCA-component index.
#     """

#     descriptor_dim = 7

#     def __init__(
#         self,
#         in_bands: int,
#         d_model: int,
#         dropout: float = 0.1,
#         norm_groups: int = 8,
#         amplitude_skip_scale_init: float = 0.15,
#     ) -> None:
#         super().__init__()
#         self.in_bands = int(in_bands)
#         self.d_model = int(d_model)
#         if self.in_bands <= 0 or self.d_model <= 0:
#             raise ValueError("in_bands and d_model must be positive")

#         hidden = max(self.d_model // 2, 32)
#         groups = _valid_group_count(
#             hidden, int(norm_groups)
#         )

#         self.context_stem = nn.Sequential(
#             nn.Conv2d(
#                 self.in_bands,
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
#             nn.GroupNorm(groups, hidden),
#             nn.GELU(),
#             nn.Conv2d(
#                 hidden,
#                 self.d_model,
#                 kernel_size=1,
#                 bias=False,
#             ),
#         )

#         # Direct linear path from processed spectral vector to spatial feature
#         # map. It preserves amplitude and directional response information.
#         self.amplitude_projection = nn.Conv2d(
#             self.in_bands,
#             self.d_model,
#             kernel_size=1,
#             bias=False,
#         )
#         self.amplitude_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(
#                     float(amplitude_skip_scale_init)
#                 )
#             )
#         )

#         component_hidden = max(
#             self.d_model // 2, 32
#         )
#         self.component_embed = nn.Sequential(
#             nn.Linear(
#                 self.descriptor_dim,
#                 component_hidden,
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 component_hidden,
#                 self.d_model,
#             ),
#         )
#         self.spatial_dropout = nn.Dropout(
#             float(dropout)
#         )
#         self.component_dropout = nn.Dropout(
#             float(dropout)
#         )

#     @staticmethod
#     def _component_descriptors(
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         _, _, height, width = x.shape
#         center_h, center_w = height // 2, width // 2

#         center = x[:, :, center_h, center_w]
#         global_mean = x.mean(dim=(-1, -2))
#         global_std = x.float().std(
#             dim=(-1, -2), unbiased=False
#         ).to(dtype=x.dtype)

#         h0 = max(center_h - 1, 0)
#         h1 = min(center_h + 2, height)
#         w0 = max(center_w - 1, 0)
#         w1 = min(center_w + 2, width)
#         local_mean = x[:, :, h0:h1, w0:w1].mean(
#             dim=(-1, -2)
#         )

#         center_global_contrast = center - global_mean
#         local_global_contrast = local_mean - global_mean

#         horizontal = (
#             x[:, :, :, 1:] - x[:, :, :, :-1]
#             if width > 1
#             else torch.zeros_like(x[:, :, :, :1])
#         )
#         vertical = (
#             x[:, :, 1:, :] - x[:, :, :-1, :]
#             if height > 1
#             else torch.zeros_like(x[:, :, :1, :])
#         )
#         gradient = 0.5 * (
#             horizontal.abs().mean(dim=(-1, -2))
#             + vertical.abs().mean(dim=(-1, -2))
#         )

#         return torch.stack(
#             [
#                 center,
#                 local_mean,
#                 global_mean,
#                 global_std,
#                 center_global_contrast,
#                 local_global_contrast,
#                 gradient,
#             ],
#             dim=-1,
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         if x.dim() != 4:
#             raise ValueError(
#                 f"Expected x [B,C,H,W], got {tuple(x.shape)}"
#             )
#         if x.size(1) != self.in_bands:
#             raise ValueError(
#                 f"Expected {self.in_bands} input channels, "
#                 f"got {x.size(1)}"
#             )
#         if not torch.isfinite(x).all():
#             raise RuntimeError(
#                 "Backbone input contains NaN/Inf"
#             )

#         x32 = x.float()
#         context_map = self.context_stem(x32)
#         amplitude_map = self.amplitude_projection(
#             x32
#         )
#         amplitude_scale = torch.sigmoid(
#             self.amplitude_logit
#         ).to(dtype=context_map.dtype)
#         spatial_map = (
#             context_map
#             + amplitude_scale * amplitude_map
#         )

#         spatial_tokens = spatial_map.flatten(
#             2
#         ).transpose(1, 2)
#         descriptors = self._component_descriptors(
#             x32
#         )
#         component_tokens = self.component_embed(
#             descriptors
#         )

#         dropped_component_tokens = self.component_dropout(component_tokens)
#         dropped_spatial_tokens = self.spatial_dropout(spatial_tokens)
#         return {
#             "band_tokens_init": dropped_component_tokens,
#             "component_tokens_init": dropped_component_tokens,
#             "spatial_tokens_init": dropped_spatial_tokens,
#             "component_descriptors": descriptors,
#             "spatial_feature_map": spatial_map,
#             "context_feature_map": context_map,
#             "amplitude_feature_map": amplitude_map,
#             "amplitude_skip_scale":
#                 amplitude_scale.detach(),
#         }


# # =============================================================================
# # Component/physical spectral encoder
# # =============================================================================


# class SpectralSSMEncoder(nn.Module):
#     """Physical-band SSM or PCA-component set encoder.

#     - Physical ordered bands: a bidirectional diagonal SSM is valid.
#     - PCA-reduced components: global set-context blocks are used, because local
#       convolution over neighbouring principal-component indices has no
#       wavelength interpretation.

#     The physical spectral intervention response used by PC-SIRG is computed
#     outside this module by applying +/- raw-spectrum interventions, the frozen
#     preprocessing transform, and repeated deterministic canonical forwards.
#     """

#     PHYSICAL_MODE = "physical_wavelength"
#     COMPONENT_MODE = "reduced_component"

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(args.d_model)
#         self.num_bands = int(args.num_bands)

#         explicit_mode = getattr(
#             args, "model_input_axis_mode", None
#         )
#         explicit_flag = getattr(
#             args,
#             "model_input_spectral_axis_is_physical",
#             None,
#         )

#         if explicit_mode is not None:
#             token = str(explicit_mode).strip().lower()
#             if token in {
#                 "physical",
#                 "physical_wavelength",
#                 "raw_band",
#                 "raw_bands",
#             }:
#                 self.axis_mode = self.PHYSICAL_MODE
#             elif token in {
#                 "component",
#                 "components",
#                 "pca",
#                 "reduced_component",
#             }:
#                 self.axis_mode = self.COMPONENT_MODE
#             else:
#                 raise ValueError(
#                     "model_input_axis_mode must be physical_wavelength "
#                     "or reduced_component"
#                 )
#         elif explicit_flag is not None:
#             self.axis_mode = (
#                 self.PHYSICAL_MODE
#                 if bool(explicit_flag)
#                 else self.COMPONENT_MODE
#             )
#         else:
#             raw_bands = int(
#                 getattr(
#                     args,
#                     "raw_num_bands",
#                     self.num_bands,
#                 )
#             )
#             pca_components = int(
#                 getattr(
#                     args,
#                     "pca_components",
#                     self.num_bands,
#                 )
#             )
#             self.axis_mode = (
#                 self.COMPONENT_MODE
#                 if pca_components < raw_bands
#                 else self.PHYSICAL_MODE
#             )

#         self.input_axis_is_physical = (
#             self.axis_mode == self.PHYSICAL_MODE
#         )
#         self.component_identity = nn.Parameter(
#             torch.randn(
#                 1,
#                 self.num_bands,
#                 self.d_model,
#             )
#             * 0.02
#         )

#         norm_type = str(
#             getattr(args, "backbone_norm", "layer")
#         ).lower()
#         residual_scale_init = float(
#             getattr(
#                 args,
#                 "ssm_residual_scale_init",
#                 0.70,
#             )
#         )
#         component_residual_init = float(
#             getattr(
#                 args,
#                 "component_residual_scale_init",
#                 0.50,
#             )
#         )
#         layer_count = int(
#             getattr(args, "num_spectral_layers", 2)
#         )

#         if self.input_axis_is_physical:
#             self.physical_layers = nn.ModuleList(
#                 [
#                     StableSSMBlock(
#                         self.d_model,
#                         int(args.d_state),
#                         d_conv=int(
#                             getattr(args, "d_conv", 4)
#                         ),
#                         expand=int(
#                             getattr(args, "expand", 2)
#                         ),
#                         dropout=float(args.dropout),
#                         norm_type=norm_type,
#                         residual_scale_init=
#                             residual_scale_init,
#                         kernel_dropout=float(
#                             getattr(
#                                 args,
#                                 "ssm_kernel_dropout",
#                                 0.0,
#                             )
#                         ),
#                     )
#                     for _ in range(layer_count)
#                 ]
#             )
#             self.component_layers = nn.ModuleList()
#         else:
#             self.physical_layers = nn.ModuleList()
#             self.component_layers = nn.ModuleList(
#                 [
#                     ComponentSetBlock(
#                         self.d_model,
#                         expand=int(
#                             getattr(args, "expand", 2)
#                         ),
#                         dropout=float(args.dropout),
#                         residual_scale_init=
#                             component_residual_init,
#                         norm_type=norm_type,
#                     )
#                     for _ in range(layer_count)
#                 ]
#             )

#         hidden = max(self.d_model // 2, 32)
#         self.channel_importance = nn.Sequential(
#             nn.Linear(
#                 self.d_model, hidden
#             ),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#         )
#         self.norm = nn.LayerNorm(self.d_model)

#     @property
#     def layers(self) -> nn.ModuleList:
#         """Compatibility view for old code that inspects `.layers`."""
#         return (
#             self.physical_layers
#             if self.input_axis_is_physical
#             else self.component_layers
#         )

#     def _encode_physical(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         paired = torch.cat(
#             [x, torch.flip(x, dims=[1])],
#             dim=0,
#         )
#         for layer in self.physical_layers:
#             paired = layer(paired)
#         forward, reverse = paired.chunk(2, dim=0)
#         reverse = torch.flip(reverse, dims=[1])
#         return 0.5 * (forward + reverse)

#     def _encode_components(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         for layer in self.component_layers:
#             x = layer(x)
#         return x

#     def forward(
#         self,
#         band_tokens_init: torch.Tensor,
#     ) -> Tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         if band_tokens_init.dim() != 3:
#             raise ValueError(
#                 "Expected channel/component tokens [B,C,D], "
#                 f"got {tuple(band_tokens_init.shape)}"
#             )
#         _, channels, width = band_tokens_init.shape
#         if (
#             channels != self.num_bands
#             or width != self.d_model
#         ):
#             raise ValueError(
#                 f"Expected [B,{self.num_bands},{self.d_model}], "
#                 f"got {tuple(band_tokens_init.shape)}"
#             )

#         x = (
#             band_tokens_init
#             + self.component_identity
#         )
#         mixed = (
#             self._encode_physical(x)
#             if self.input_axis_is_physical
#             else self._encode_components(x)
#         )
#         tokens = self.norm(mixed)

#         importance_input = F.normalize(
#             tokens, dim=-1, eps=1e-6
#         )
#         logits = self.channel_importance(
#             importance_input
#         ).squeeze(-1)
#         weights = F.softmax(logits, dim=-1)

#         # Pool the unnormalized residual stream so radial information remains
#         # available to the canonical low-rank Gaussian geometry.
#         features = torch.sum(
#             mixed * weights.unsqueeze(-1),
#             dim=1,
#         )
#         return features, weights, tokens


# # =============================================================================
# # Spatial encoder
# # =============================================================================


# class SpatialSSMEncoder(nn.Module):
#     """Four-direction raster/column diagonal-SSM encoder.

#     All directions share SSM parameters. A learned global softmax combines
#     directions and is initialized uniformly, retaining symmetry at startup.
#     """

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()
#         self.patch_size = int(args.patch_size)
#         self.d_model = int(args.d_model)
#         self.num_patches = (
#             self.patch_size * self.patch_size
#         )
#         if self.patch_size <= 0:
#             raise ValueError("patch_size must be positive")

#         self.row_embed = nn.Parameter(
#             torch.randn(
#                 1,
#                 self.patch_size,
#                 self.d_model,
#             )
#             * 0.02
#         )
#         self.col_embed = nn.Parameter(
#             torch.randn(
#                 1,
#                 self.patch_size,
#                 self.d_model,
#             )
#             * 0.02
#         )

#         norm_type = str(
#             getattr(args, "backbone_norm", "layer")
#         ).lower()
#         residual_scale_init = float(
#             getattr(
#                 args,
#                 "ssm_residual_scale_init",
#                 0.70,
#             )
#         )
#         self.layers = nn.ModuleList(
#             [
#                 StableSSMBlock(
#                     self.d_model,
#                     int(args.d_state),
#                     d_conv=int(
#                         getattr(args, "d_conv", 4)
#                     ),
#                     expand=int(
#                         getattr(args, "expand", 2)
#                     ),
#                     dropout=float(args.dropout),
#                     norm_type=norm_type,
#                     residual_scale_init=
#                         residual_scale_init,
#                     kernel_dropout=float(
#                         getattr(
#                             args,
#                             "ssm_kernel_dropout",
#                             0.0,
#                         )
#                     ),
#                 )
#                 for _ in range(
#                     int(args.num_layers)
#                 )
#             ]
#         )
#         self.norm = nn.LayerNorm(
#             self.d_model
#         )

#         hidden = max(
#             self.d_model // 2, 32
#         )
#         self.spatial_importance = nn.Sequential(
#             nn.Linear(
#                 self.d_model, hidden
#             ),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#         )

#         grid = torch.arange(
#             self.num_patches,
#             dtype=torch.long,
#         ).reshape(
#             self.patch_size,
#             self.patch_size,
#         )
#         orders = torch.stack(
#             [
#                 grid.flatten(),
#                 torch.flip(
#                     grid.flatten(), dims=[0]
#                 ),
#                 grid.transpose(
#                     0, 1
#                 ).flatten(),
#                 torch.flip(
#                     grid.transpose(
#                         0, 1
#                     ).flatten(),
#                     dims=[0],
#                 ),
#             ],
#             dim=0,
#         )
#         inverse = torch.argsort(
#             orders, dim=1
#         )
#         self.register_buffer(
#             "scan_orders",
#             orders,
#             persistent=False,
#         )
#         self.register_buffer(
#             "inverse_scan_orders",
#             inverse,
#             persistent=False,
#         )
#         self.direction_logits = nn.Parameter(
#             torch.zeros(orders.size(0))
#         )

#     def _encode(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def _position_encoding(
#         self,
#         dtype: torch.dtype,
#     ) -> torch.Tensor:
#         position = (
#             self.row_embed[:, :, None, :]
#             + self.col_embed[:, None, :, :]
#         )
#         return position.reshape(
#             1,
#             self.num_patches,
#             self.d_model,
#         ).to(dtype=dtype)

#     def forward(
#         self,
#         spatial_tokens_init: torch.Tensor,
#     ) -> Tuple[
#         torch.Tensor,
#         Dict[str, torch.Tensor],
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         if spatial_tokens_init.dim() != 3:
#             raise ValueError(
#                 "Expected spatial tokens [B,HW,D], "
#                 f"got {tuple(spatial_tokens_init.shape)}"
#             )
#         _, count, width = (
#             spatial_tokens_init.shape
#         )
#         if (
#             count != self.num_patches
#             or width != self.d_model
#         ):
#             raise ValueError(
#                 f"Expected [B,{self.num_patches},{self.d_model}], "
#                 f"got {tuple(spatial_tokens_init.shape)}"
#             )

#         x = (
#             spatial_tokens_init
#             + self._position_encoding(
#                 spatial_tokens_init.dtype
#             )
#         )
#         batch = x.size(0)
#         ordered = torch.cat(
#             [
#                 x.index_select(1, order)
#                 for order in self.scan_orders
#             ],
#             dim=0,
#         )
#         encoded = self._encode(ordered)
#         encoded_by_direction = encoded.reshape(
#             self.scan_orders.size(0),
#             batch,
#             self.num_patches,
#             self.d_model,
#         )
#         restored = torch.stack(
#             [
#                 encoded_by_direction[
#                     direction
#                 ].index_select(
#                     1,
#                     self.inverse_scan_orders[
#                         direction
#                     ],
#                 )
#                 for direction in range(
#                     self.scan_orders.size(0)
#                 )
#             ],
#             dim=0,
#         )
#         direction_weights = F.softmax(
#             self.direction_logits,
#             dim=0,
#         ).to(dtype=restored.dtype)
#         mixed = torch.sum(
#             restored
#             * direction_weights.view(
#                 -1, 1, 1, 1
#             ),
#             dim=0,
#         )
#         tokens = self.norm(mixed)

#         importance_input = F.normalize(
#             tokens, dim=-1, eps=1e-6
#         )
#         logits = self.spatial_importance(
#             importance_input
#         ).squeeze(-1)
#         weights = F.softmax(logits, dim=-1)
#         features = torch.sum(
#             mixed * weights.unsqueeze(-1),
#             dim=1,
#         )

#         patterns = {
#             "direction_weights":
#                 direction_weights.detach(),
#         }
#         return (
#             features,
#             patterns,
#             tokens,
#             weights,
#         )


# # =============================================================================
# # Scale-preserving fusion
# # =============================================================================


# class TokenFusion(nn.Module):
#     """Cross-conditioned fusion without final feature normalization."""

#     def __init__(
#         self,
#         d_model: int,
#         dropout: float = 0.1,
#         residual_scale: float = 0.30,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         self.spec_norm = nn.LayerNorm(
#             self.d_model
#         )
#         self.spat_norm = nn.LayerNorm(
#             self.d_model
#         )
#         self.spec_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )
#         self.spat_proj = nn.Linear(
#             self.d_model,
#             self.d_model,
#             bias=False,
#         )

#         self.cross_gate = nn.Sequential(
#             nn.Linear(
#                 4 * self.d_model,
#                 self.d_model,
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 self.d_model,
#                 self.d_model,
#             ),
#             nn.Sigmoid(),
#         )
#         self.feature_mlp = nn.Sequential(
#             nn.Linear(
#                 2 * self.d_model,
#                 self.d_model,
#             ),
#             nn.GELU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(
#                 self.d_model,
#                 self.d_model,
#             ),
#         )
#         self.token_norm = nn.LayerNorm(
#             self.d_model
#         )
#         self.dropout = nn.Dropout(
#             float(dropout)
#         )
#         self.fusion_logit = nn.Parameter(
#             torch.tensor(
#                 _inverse_sigmoid(
#                     float(residual_scale)
#                 )
#             )
#         )

#     def forward(
#         self,
#         spectral_tokens: torch.Tensor,
#         spatial_tokens: torch.Tensor,
#         spectral_features: torch.Tensor | None = None,
#         spatial_features: torch.Tensor | None = None,
#     ) -> Tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         if (
#             spectral_tokens.dim() != 3
#             or spatial_tokens.dim() != 3
#         ):
#             raise ValueError(
#                 "Fusion expects two token tensors"
#             )
#         if (
#             spectral_tokens.size(0)
#             != spatial_tokens.size(0)
#         ):
#             raise ValueError(
#                 "Fusion branch batch sizes differ"
#             )
#         if (
#             spectral_tokens.size(-1)
#             != self.d_model
#             or spatial_tokens.size(-1)
#             != self.d_model
#         ):
#             raise ValueError(
#                 "Fusion branch widths differ"
#             )

#         spectral = (
#             spectral_tokens.mean(dim=1)
#             if spectral_features is None
#             else spectral_features
#         )
#         spatial = (
#             spatial_tokens.mean(dim=1)
#             if spatial_features is None
#             else spatial_features
#         )

#         spectral_view = self.spec_proj(
#             self.spec_norm(spectral)
#         )
#         spatial_view = self.spat_proj(
#             self.spat_norm(spatial)
#         )
#         gate_input = torch.cat(
#             [
#                 spectral_view,
#                 spatial_view,
#                 torch.abs(
#                     spectral_view
#                     - spatial_view
#                 ),
#                 spectral_view * spatial_view,
#             ],
#             dim=-1,
#         )
#         gate = self.cross_gate(gate_input)

#         # Preserve raw branch amplitudes in the base stream.
#         base = (
#             gate * spectral
#             + (1.0 - gate) * spatial
#         )
#         delta = self.feature_mlp(
#             torch.cat(
#                 [spectral_view, spatial_view],
#                 dim=-1,
#             )
#         )
#         scale = torch.sigmoid(
#             self.fusion_logit
#         ).to(dtype=delta.dtype)
#         fused_feature = (
#             base
#             + scale * self.dropout(delta)
#         )

#         # Token normalization is diagnostic/auxiliary only. The feature used by
#         # geometry is intentionally not normalized.
#         fused_tokens = self.token_norm(
#             torch.cat(
#                 [spectral_tokens, spatial_tokens],
#                 dim=1,
#             )
#         )
#         return (
#             fused_feature,
#             fused_tokens,
#             gate,
#         )


# # =============================================================================
# # Final backbone
# # =============================================================================


# class SSMBackbone(nn.Module):
#     """Geometry-preserving backbone for PC-SIRG NECIL-HSI.

#     The backbone produces one unnormalized Euclidean feature vector. PC-SIRG
#     spectral responses are not stored or predicted by a separate head. They are
#     obtained by deterministic repeated forwards of +/- raw-spectrum
#     interventions after the same frozen preprocessing transform.

#     Main-method incremental contract:
#       - preprocessing frozen;
#       - backbone frozen and evaluation-only;
#       - projection frozen;
#       - old GeometryBank rows immutable;
#       - only provisional current PC-SIRG rows are optimized.
#     """

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(args.d_model)
#         self.num_bands = int(args.num_bands)
#         self.patch_size = int(args.patch_size)
#         self._incremental_frozen = False

#         self.stem = SpectralSpatialStem(
#             in_bands=self.num_bands,
#             d_model=self.d_model,
#             dropout=float(args.dropout),
#             norm_groups=int(
#                 getattr(
#                     args,
#                     "stem_norm_groups",
#                     8,
#                 )
#             ),
#             amplitude_skip_scale_init=float(
#                 getattr(
#                     args,
#                     "amplitude_skip_scale_init",
#                     0.15,
#                 )
#             ),
#         )
#         self.spectral_encoder = (
#             SpectralSSMEncoder(args)
#         )
#         self.spatial_encoder = (
#             SpatialSSMEncoder(args)
#         )
#         self.token_fusion = TokenFusion(
#             self.d_model,
#             dropout=float(args.dropout),
#             residual_scale=float(
#                 getattr(
#                     args,
#                     "fusion_residual_scale",
#                     0.30,
#                 )
#             ),
#         )
#         self.output_dropout = nn.Dropout(
#             float(
#                 getattr(
#                     args,
#                     "backbone_output_dropout",
#                     0.0,
#                 )
#             )
#         )

#     def train(
#         self,
#         mode: bool = True,
#     ) -> "SSMBackbone":
#         # A surrounding model.train() call must not reactivate stochastic
#         # backbone behavior after the strict incremental freeze.
#         effective_mode = (
#             False
#             if self._incremental_frozen
#             else bool(mode)
#         )
#         super().train(effective_mode)
#         return self

#     def backbone_contract(
#         self,
#     ) -> Dict[str, Any]:
#         return {
#             "input_shape": "[B,C,H,W]",
#             "input_channel_role":
#                 self.spectral_encoder.axis_mode,
#             "component_encoder": (
#                 "global_set_context"
#                 if not self.spectral_encoder.input_axis_is_physical
#                 else "bidirectional_diagonal_ssm"
#             ),
#             "false_pca_wavelength_locality": False,
#             "ssm_family":
#                 "channelwise_diagonal_s4d",
#             "is_mamba_selective_scan": False,
#             "spatial_scan_directions": 4,
#             "physical_channel_scan_directions": (
#                 2
#                 if self.spectral_encoder.input_axis_is_physical
#                 else 0
#             ),
#             "final_feature_normalization": False,
#             "output_feature_space":
#                 "unnormalized_euclidean_h",
#             "amplitude_preserving_stem_path": True,
#             "batch_dependent_normalization": False,
#             "pc_sirg_response_extraction": (
#                 "external_symmetric_raw_spectral_interventions_"
#                 "through_same_frozen_canonical_forward"
#             ),
#             "raw_spectral_gaussian_sidecar": False,
#             "latent_spectral_feature_coupling": False,
#             "strict_incremental_backbone_frozen":
#                 self._incremental_frozen,
#         }

#     def get_last_blocks(
#         self,
#     ) -> List[nn.Module]:
#         blocks: List[nn.Module] = []
#         if len(self.spectral_encoder.layers):
#             blocks.append(
#                 self.spectral_encoder.layers[-1]
#             )
#         if len(self.spatial_encoder.layers):
#             blocks.append(
#                 self.spatial_encoder.layers[-1]
#             )
#         return blocks

#     def enable_base_training(
#         self,
#     ) -> None:
#         self._incremental_frozen = False
#         for parameter in self.parameters():
#             parameter.requires_grad_(True)
#             parameter.grad = None
#         self.train(True)

#     def freeze_for_incremental(
#         self,
#     ) -> None:
#         self._incremental_frozen = True
#         for parameter in self.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None
#         super().train(False)

#     # Compatibility name. It now performs the full strict freeze.
#     freeze_all = freeze_for_incremental

#     def assert_frozen_for_incremental(
#         self,
#     ) -> None:
#         if not self._incremental_frozen:
#             raise RuntimeError(
#                 "Backbone incremental freeze flag is not active"
#             )
#         active = [
#             name
#             for name, parameter in self.named_parameters()
#             if parameter.requires_grad
#         ]
#         gradients = [
#             name
#             for name, parameter in self.named_parameters()
#             if parameter.grad is not None
#         ]
#         training_modules = [
#             module.__class__.__name__
#             for module in self.modules()
#             if module.training
#         ]
#         if active:
#             raise RuntimeError(
#                 f"Frozen backbone has trainable parameters: {active}"
#             )
#         if gradients:
#             raise RuntimeError(
#                 f"Frozen backbone retains gradients: {gradients}"
#             )
#         if training_modules:
#             raise RuntimeError(
#                 "Frozen backbone has modules in training mode: "
#                 f"{training_modules[:8]}"
#             )

#     def unfreeze_last_blocks(
#         self,
#         *,
#         allow_ablation: bool = False,
#     ) -> None:
#         if not allow_ablation:
#             raise RuntimeError(
#                 "Partial backbone unfreezing is forbidden in the strict "
#                 "PC-SIRG main method. Pass allow_ablation=True only for an "
#                 "explicit non-strict ablation."
#             )
#         self._incremental_frozen = False
#         for block in self.get_last_blocks():
#             for parameter in block.parameters():
#                 parameter.requires_grad_(True)
#         self.train(True)

#     def unfreeze_token_fusion(
#         self,
#         *,
#         allow_ablation: bool = False,
#     ) -> None:
#         if not allow_ablation:
#             raise RuntimeError(
#                 "Fusion unfreezing is forbidden in the strict PC-SIRG main "
#                 "method. Pass allow_ablation=True only for an ablation."
#             )
#         self._incremental_frozen = False
#         for parameter in self.token_fusion.parameters():
#             parameter.requires_grad_(True)
#         self.train(True)

#     @staticmethod
#     @torch.no_grad()
#     def feature_statistics(
#         features: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         norms = features.norm(dim=-1)
#         return {
#             "feature_norm_mean":
#                 norms.mean().detach(),
#             "feature_norm_std":
#                 norms.std(
#                     unbiased=False
#                 ).detach(),
#             "feature_norm_cv": (
#                 norms.std(unbiased=False)
#                 / norms.mean().clamp_min(1e-8)
#             ).detach(),
#             "feature_abs_mean":
#                 features.abs().mean().detach(),
#             "feature_dimension_variance_mean":
#                 features.var(
#                     dim=0,
#                     unbiased=False,
#                 ).mean().detach(),
#         }

#     def _forward_impl(
#         self,
#         x: torch.Tensor,
#     ) -> Dict[str, Any]:
#         if x.dim() != 4:
#             raise ValueError(
#                 f"Expected x [B,C,H,W], got {tuple(x.shape)}"
#             )
#         _, channels, height, width = x.shape
#         if channels != self.num_bands:
#             raise ValueError(
#                 f"Expected {self.num_bands} channels, "
#                 f"got {channels}"
#             )
#         if (
#             height != self.patch_size
#             or width != self.patch_size
#         ):
#             raise ValueError(
#                 f"Expected spatial size "
#                 f"{self.patch_size}x{self.patch_size}, "
#                 f"got {height}x{width}"
#             )
#         if not torch.isfinite(x).all():
#             raise RuntimeError(
#                 "Backbone input contains NaN/Inf"
#             )

#         stem = self.stem(x)
#         component_features, component_weights, component_tokens = (
#             self.spectral_encoder(
#                 stem["component_tokens_init"]
#             )
#         )
#         (
#             spatial_features,
#             spatial_patterns,
#             spatial_tokens,
#             spatial_weights,
#         ) = self.spatial_encoder(
#             stem["spatial_tokens_init"]
#         )
#         (
#             features,
#             fused_tokens,
#             fusion_gate,
#         ) = self.token_fusion(
#             component_tokens,
#             spatial_tokens,
#             spectral_features=component_features,
#             spatial_features=spatial_features,
#         )
#         features = self.output_dropout(features)
#         if not torch.isfinite(features).all():
#             raise RuntimeError(
#                 "Backbone produced NaN/Inf features"
#             )

#         statistics = self.feature_statistics(
#             features
#         )
#         contract = self.backbone_contract()
#         return {
#             "features": features,
#             "geometry_features": features,
#             # Compatibility aliases.
#             "spectral_features": component_features,
#             "component_features": component_features,
#             "spatial_features": spatial_features,
#             "band_weights": component_weights,
#             "component_weights": component_weights,
#             "spatial_weights": spatial_weights,
#             "fusion_gate": fusion_gate,
#             "spatial_patterns": spatial_patterns,
#             "spectral_tokens": component_tokens,
#             "component_tokens": component_tokens,
#             "spatial_tokens": spatial_tokens,
#             "fused_tokens": fused_tokens,
#             "component_descriptors":
#                 stem["component_descriptors"],
#             "amplitude_skip_scale":
#                 stem["amplitude_skip_scale"],
#             "feature_stats": statistics,
#             "backbone_contract": contract,
#         }

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> Dict[str, Any]:
#         return self._forward_impl(x)

#     def forward_geometry(
#         self,
#         x: torch.Tensor,
#     ) -> Dict[str, Any]:
#         """Deterministic geometry forward with gradients still enabled.

#         Use this for:
#           - base support-row construction;
#           - base query joint-energy training;
#           - +/- intervention response extraction;
#           - frozen incremental feature extraction.

#         During frozen incremental evaluation callers should additionally use
#         ``torch.inference_mode()``.
#         """
#         with _temporary_eval_mode(self):
#             return self._forward_impl(x)
