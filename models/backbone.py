import math
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMSNorm keeps feature scale more stable than BatchNorm for small-batch HSI CIL.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class S4DKernel(nn.Module):
    """
    Lightweight diagonal state-space kernel.

    This is not a full S4/Mamba implementation. It is a compact stable sequence
    mixer suitable for small HSI datasets where over-parameterized sequence models
    can overfit quickly.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        kernel_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.kernel_dropout = float(kernel_dropout)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.register_buffer("A", -A)

        self.B = nn.Parameter(torch.randn(self.d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(self.d_state) * 0.02)

        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.D = nn.Parameter(torch.zeros(1))

    def _compute_kernel(self, L: int, dA: torch.Tensor, dB: torch.Tensor) -> torch.Tensor:
        powers = torch.arange(L, device=dA.device, dtype=dA.dtype)
        dA_pows = dA.unsqueeze(-1) ** powers.unsqueeze(0)
        kernel = torch.einsum("n,n,nl->l", self.C, dB, dA_pows)
        kernel = torch.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training and self.kernel_dropout > 0.0:
            kernel = F.dropout(kernel, p=self.kernel_dropout, training=True)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x to be (B,L,D), got {tuple(x.shape)}")

        _, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected feature dim {self.d_model}, got {D}")

        dt = torch.exp(self.log_dt).clamp(min=1e-4, max=1.0)
        dA = torch.exp(torch.clamp(dt * self.A, -10.0, 10.0))
        dB = (dA - 1.0) / self.A.clamp_max(-1e-4) * self.B

        K = self._compute_kernel(L, dA, dB)
        K = K.unsqueeze(0).unsqueeze(0).expand(D, 1, -1)

        x_conv = x.transpose(1, 2)  # [B,D,L]
        y = F.conv1d(F.pad(x_conv, (L - 1, 0)), K, groups=D)
        y = y[:, :, :L].transpose(1, 2)
        return y + self.D * x


class MambaBlock(nn.Module):
    """
    Lightweight gated SSM block.

    Uses pre-norm residual structure and an optional residual scale. The residual
    scale is useful for incremental phases because it reduces abrupt feature drift.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "layer",
        residual_scale_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_inner = int(d_model * expand)

        if str(norm_type).lower() == "rms":
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=int(d_conv),
            padding=int(d_conv) - 1,
            groups=self.d_inner,
        )
        self.ssm = S4DKernel(self.d_inner, int(d_state))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x to be (B,L,D), got {tuple(x.shape)}")

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        xc = self.conv1d(x_proj.transpose(1, 2))
        x_conv = xc[:, :, -x_proj.size(1):].transpose(1, 2)

        x_ssm = self.ssm(x_conv)
        y = x_ssm * torch.sigmoid(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        scale = torch.clamp(self.residual_scale, 0.0, 1.5)
        return residual + scale * y


class SpectralSpatialStem(nn.Module):
    """
    Shared local HSI stem before branch separation.

    Uses GroupNorm instead of BatchNorm3d. BatchNorm is fragile in HSI CIL because
    batch composition changes per phase and batch sizes are usually small.
    """

    def __init__(self, in_bands: int, d_model: int, dropout: float = 0.1, norm_groups: int = 8):
        super().__init__()
        del in_bands
        hidden = max(d_model // 2, 16)
        groups = max(1, min(int(norm_groups), hidden))

        self.stem3d = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )

        self.band_proj = nn.Linear(hidden, d_model)
        self.spatial_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected x to be (B,C,H,W), got {tuple(x.shape)}")

        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x3 = x.unsqueeze(1)       # [B,1,C,H,W]
        feat3 = self.stem3d(x3)   # [B,Hid,C,H,W]

        band_feat = feat3.mean(dim=(-1, -2)).transpose(1, 2)       # [B,C,Hid]
        band_tokens_init = self.dropout(self.band_proj(band_feat)) # [B,C,D]

        spatial_feat = feat3.mean(dim=2).flatten(2).transpose(1, 2)      # [B,HW,Hid]
        spatial_tokens_init = self.dropout(self.spatial_proj(spatial_feat)) # [B,HW,D]

        return {
            "band_tokens_init": band_tokens_init,
            "spatial_tokens_init": spatial_tokens_init,
        }


class SpectralSSMEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = int(args.d_model)
        self.num_bands = int(args.num_bands)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_bands, self.d_model) * 0.02)

        norm_type = str(getattr(args, "backbone_norm", "layer")).lower()
        residual_scale_init = float(getattr(args, "ssm_residual_scale_init", 1.0))

        self.layers = nn.ModuleList([
            MambaBlock(
                self.d_model,
                int(args.d_state),
                d_conv=getattr(args, "d_conv", 4),
                expand=getattr(args, "expand", 2),
                dropout=float(args.dropout),
                norm_type=norm_type,
                residual_scale_init=residual_scale_init,
            )
            for _ in range(int(args.num_spectral_layers))
        ])

        self.band_importance = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2 if self.d_model >= 32 else self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model // 2 if self.d_model >= 32 else self.d_model, 1),
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, band_tokens_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if band_tokens_init.dim() != 3:
            raise ValueError(f"Expected band_tokens_init to be (B,C,D), got {tuple(band_tokens_init.shape)}")

        _, C, _ = band_tokens_init.shape
        if C > self.pos_embed.size(1):
            raise ValueError(f"Input has {C} bands but positional table supports {self.pos_embed.size(1)}.")

        x_spectral = band_tokens_init + self.pos_embed[:, :C, :]

        for layer in self.layers:
            x_spectral = layer(x_spectral)

        spectral_tokens = self.norm(x_spectral)

        token_norm = F.normalize(spectral_tokens, dim=-1, eps=1e-6)
        band_logits = self.band_importance(token_norm).squeeze(-1)
        band_weights = F.softmax(band_logits, dim=-1)

        spectral_features = (spectral_tokens * band_weights.unsqueeze(-1)).sum(dim=1)

        return spectral_features, band_weights, spectral_tokens


class SpatialSSMEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_size = int(args.patch_size)
        self.d_model = int(args.d_model)
        self.num_patches = self.patch_size * self.patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)

        norm_type = str(getattr(args, "backbone_norm", "layer")).lower()
        residual_scale_init = float(getattr(args, "ssm_residual_scale_init", 1.0))

        self.layers = nn.ModuleList([
            MambaBlock(
                self.d_model,
                int(args.d_state),
                d_conv=getattr(args, "d_conv", 4),
                expand=getattr(args, "expand", 2),
                dropout=float(args.dropout),
                norm_type=norm_type,
                residual_scale_init=residual_scale_init,
            )
            for _ in range(int(args.num_layers))
        ])

        self.norm = nn.LayerNorm(self.d_model)

        self.spatial_importance = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2 if self.d_model >= 32 else self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model // 2 if self.d_model >= 32 else self.d_model, 1),
        )

        self.pattern_heads = nn.ModuleDict({
            "texture": nn.Linear(self.d_model, 64),
            "edge": nn.Linear(self.d_model, 64),
            "structure": nn.Linear(self.d_model, 64),
        })

    def forward(self, spatial_tokens_init: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if spatial_tokens_init.dim() != 3:
            raise ValueError(f"Expected spatial_tokens_init to be (B,HW,D), got {tuple(spatial_tokens_init.shape)}")

        _, N, _ = spatial_tokens_init.shape
        if N != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} spatial tokens, got {N}")

        x_seq = spatial_tokens_init + self.pos_embed[:, :N, :]

        for layer in self.layers:
            x_seq = layer(x_seq)

        spatial_tokens = self.norm(x_seq)

        token_norm = F.normalize(spatial_tokens, dim=-1, eps=1e-6)
        spatial_logits = self.spatial_importance(token_norm).squeeze(-1)
        spatial_weights = F.softmax(spatial_logits, dim=-1)

        spatial_features = (spatial_tokens * spatial_weights.unsqueeze(-1)).sum(dim=1)

        patterns = {k: head(spatial_features) for k, head in self.pattern_heads.items()}
        return spatial_features, patterns, spatial_tokens, spatial_weights


class TokenFusion(nn.Module):
    """
    Cross-context spectral-spatial token fusion.

    Important:
    - token-level normalization is allowed inside attention/gating;
    - output feature is not L2-normalized;
    - the fused feature preserves Euclidean scale for the GeometryBank.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, residual_scale: float = 0.5):
        super().__init__()
        self.spec_proj = nn.Linear(d_model, d_model)
        self.spat_proj = nn.Linear(d_model, d_model)

        self.cross_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, d_model),
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        spectral_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        spectral_features: torch.Tensor = None,
        spatial_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if spectral_tokens.dim() != 3 or spatial_tokens.dim() != 3:
            raise ValueError(
                f"Expected spectral/spatial tokens as [B,N,D], got {tuple(spectral_tokens.shape)} and {tuple(spatial_tokens.shape)}"
            )

        spec_ctx = spectral_tokens.mean(dim=1, keepdim=True)
        spat_ctx = spatial_tokens.mean(dim=1, keepdim=True)

        spec_view = self.spec_proj(spectral_tokens)
        spat_view = self.spat_proj(spatial_tokens)

        fused_spec = spec_view + spat_ctx
        fused_spat = spat_view + spec_ctx

        spec_mean = fused_spec.mean(dim=1)
        spat_mean = fused_spat.mean(dim=1)

        if spectral_features is not None:
            spec_mean = 0.5 * (spec_mean + spectral_features)
        if spatial_features is not None:
            spat_mean = 0.5 * (spat_mean + spatial_features)

        gate_in = torch.cat(
            [spec_mean, spat_mean, spec_ctx.squeeze(1), spat_ctx.squeeze(1)],
            dim=-1,
        )
        g = self.cross_gate(gate_in)

        fused_in = torch.cat([g * spec_mean, (1.0 - g) * spat_mean], dim=-1)
        fused_delta = self.feature_mlp(fused_in)

        # Stable residual fusion. Avoid letting the MLP rewrite the geometry stream.
        base = 0.5 * (spec_mean + spat_mean)
        fused_feature = base + self.residual_scale * fused_delta
        fused_feature = self.out_norm(self.dropout(fused_feature))

        fused_tokens = torch.cat([fused_spec, fused_spat], dim=1)
        fused_tokens = self.token_norm(fused_tokens)

        return fused_feature, fused_tokens, g


class SSMBackbone(nn.Module):
    """
    SSM-based HSI backbone with spectral-spatial token outputs.

    Geometry-aware usage:
        The output feature is not L2-normalized. GeometryBank/classifier need
        Euclidean scale, radial information, anisotropic variance, and residual
        variance. Normalized tokens are used only for attention/affinity.
    """

    def __init__(self, args):
        super().__init__()
        self.d_model = int(args.d_model)
        self.num_bands = int(args.num_bands)
        self.patch_size = int(args.patch_size)

        self.stem = SpectralSpatialStem(
            in_bands=self.num_bands,
            d_model=self.d_model,
            dropout=float(args.dropout),
            norm_groups=int(getattr(args, "stem_norm_groups", 8)),
        )
        self.spectral_encoder = SpectralSSMEncoder(args)
        self.spatial_encoder = SpatialSSMEncoder(args)
        self.token_fusion = TokenFusion(
            self.d_model,
            dropout=float(args.dropout),
            residual_scale=float(getattr(args, "fusion_residual_scale", 0.5)),
        )

        self.feature_scale = nn.Parameter(torch.tensor(1.0))
        self.output_dropout = nn.Dropout(float(getattr(args, "backbone_output_dropout", 0.0)))

    def get_last_blocks(self) -> List[nn.Module]:
        blocks: List[nn.Module] = []
        if len(self.spectral_encoder.layers) > 0:
            blocks.append(self.spectral_encoder.layers[-1])
        if len(self.spatial_encoder.layers) > 0:
            blocks.append(self.spatial_encoder.layers[-1])
        return blocks

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self):
        for block in self.get_last_blocks():
            for p in block.parameters():
                p.requires_grad = True

    def unfreeze_token_fusion(self):
        for p in self.token_fusion.parameters():
            p.requires_grad = True

    def feature_statistics(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "feature_norm_mean": features.norm(dim=-1).mean(),
            "feature_norm_std": features.norm(dim=-1).std(unbiased=False),
            "feature_abs_mean": features.abs().mean(),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected x to be (B,C,H,W), got {tuple(x.shape)}")

        _, C, H, W = x.shape
        if C != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, got {C}")
        if H != self.patch_size or W != self.patch_size:
            raise ValueError(f"Input size ({H},{W}) must match patch_size {self.patch_size}")

        stem_out = self.stem(x)

        spectral_features, band_weights, spectral_tokens = self.spectral_encoder(
            stem_out["band_tokens_init"]
        )
        spatial_features, spatial_patterns, spatial_tokens, spatial_weights = self.spatial_encoder(
            stem_out["spatial_tokens_init"]
        )

        features, fused_tokens, fusion_gate = self.token_fusion(
            spectral_tokens,
            spatial_tokens,
            spectral_features=spectral_features,
            spatial_features=spatial_features,
        )

        scale = torch.clamp(self.feature_scale, 0.25, 4.0)
        features = self.output_dropout(features * scale)

        stats = self.feature_statistics(features)

        return {
            "features": features,
            "spectral_features": spectral_features,
            "spatial_features": spatial_features,
            "band_weights": band_weights,
            "spatial_weights": spatial_weights,
            "fusion_gate": fusion_gate,
            "spatial_patterns": spatial_patterns,
            "spectral_tokens": spectral_tokens,
            "spatial_tokens": spatial_tokens,
            "fused_tokens": fused_tokens,
            "feature_stats": stats,
        }
