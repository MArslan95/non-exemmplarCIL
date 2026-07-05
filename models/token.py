import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class HSISemanticConceptEncoder(nn.Module):
    """
    Geometry-safe semantic refinement and token-level spectral-spatial manifold module
    for non-exemplar HSI class-incremental learning.

    Core principle
    --------------
    The feature stream used by the geometry bank must preserve Euclidean scale.
    Therefore:
        - semantic matching may use normalized views
        - token affinities may use normalized token space
        - pooled output features are never L2-normalized by this module
        - semantic refinement is residual and conservative

    Token manifold preservation
    ---------------------------
    Builds differentiable token relation graphs:
        - spectral_affinity: band-token relation graph
        - spatial_affinity: spatial-token relation graph
        - cross_affinity: spatial-to-spectral relation graph
        - fused_affinity: joint spectral-spatial relation graph

    Improvements over the basic version
    -----------------------------------
    1) Optional top-k sparsification of token affinities.
    2) Optional symmetric self-affinity normalization.
    3) Reliability-weighted token relation loss.
    4) Safer semantic residual with bounded alpha.
    5) Optional detachment of relation targets.
    6) Shape-safe target matching for class-level and sample-level targets.
    """

    def __init__(
        self,
        feature_dim: int,
        concept_dim: int,
        dropout: float = 0.1,
        token_temperature: float = 0.07,
        affinity_eps: float = 1e-6,
        base_alpha: float = 0.10,
        inc_alpha: float = 0.02,
        max_alpha: float = 0.15,
        topk_ratio: float = 1.0,
        symmetric_affinity: bool = True,
        detach_relation_targets: bool = True,
    ):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.concept_dim = int(concept_dim)
        self.token_temperature = float(token_temperature)
        self.affinity_eps = float(affinity_eps)
        self.base_alpha = float(base_alpha)
        self.inc_alpha = float(inc_alpha)
        self.max_alpha = float(max_alpha)
        self.topk_ratio = float(topk_ratio)
        self.symmetric_affinity = bool(symmetric_affinity)
        self.detach_relation_targets = bool(detach_relation_targets)

        if self.feature_dim != self.concept_dim:
            self.input_adapter = nn.Linear(self.feature_dim, self.concept_dim)
            self.output_adapter = nn.Linear(self.concept_dim, self.feature_dim)
        else:
            self.input_adapter = nn.Identity()
            self.output_adapter = nn.Identity()

        self.semantic_refine = nn.Sequential(
            nn.Linear(self.concept_dim, self.concept_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.concept_dim, self.concept_dim),
        )

        self.concept_gate = nn.Sequential(
            nn.Linear(self.concept_dim * 2, self.concept_dim),
            nn.GELU(),
            nn.Linear(self.concept_dim, self.concept_dim),
            nn.Sigmoid(),
        )

        # Learnable residual strength, bounded in forward.
        self.base_alpha_scale = nn.Parameter(torch.tensor(1.0))
        self.inc_alpha_scale = nn.Parameter(torch.tensor(1.0))

    # =========================================================
    # Basic helpers
    # =========================================================
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-6)

    def _safe_temperature(self) -> float:
        return max(float(self.token_temperature), float(self.affinity_eps))

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=ref.device, dtype=ref.dtype)

    def _bounded_alpha(self, phase: int, disable_semantic_in_incremental: bool = False) -> torch.Tensor:
        if int(phase) > 0 and disable_semantic_in_incremental:
            return torch.tensor(0.0, device=self.base_alpha_scale.device, dtype=self.base_alpha_scale.dtype)

        if int(phase) == 0:
            raw = self.base_alpha * F.softplus(self.base_alpha_scale)
        else:
            raw = self.inc_alpha * F.softplus(self.inc_alpha_scale)

        return torch.clamp(raw, min=0.0, max=self.max_alpha)

    # =========================================================
    # Concept attention / feature refinement
    # =========================================================
    def _attend(self, features_unit: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        """
        features_unit: [B, D]
        concepts:      [C, K, D]
        returns:       [B, D]
        """
        if features_unit.dim() != 2:
            raise ValueError(f"features_unit must be [B,D], got {tuple(features_unit.shape)}")
        if concepts.dim() != 3:
            raise ValueError(f"concepts must be [C,K,D], got {tuple(concepts.shape)}")

        B, D = features_unit.shape
        C, K, D2 = concepts.shape
        if D != D2:
            raise ValueError(f"Feature/concept dim mismatch: {D} vs {D2}")

        concepts_flat = concepts.reshape(C * K, D)
        concepts_unit = self._normalize(concepts_flat)

        sim = torch.matmul(features_unit, concepts_unit.t())
        attn = F.softmax(sim / self._safe_temperature(), dim=-1)
        token = torch.matmul(attn, concepts_unit)
        return self._normalize(token)

    def refine_features(
        self,
        features: torch.Tensor,
        concept_bank: Optional[torch.Tensor] = None,
        phase: int = 0,
        disable: bool = False,
        disable_semantic_in_incremental: bool = True,
    ) -> torch.Tensor:
        """
        Conservative geometry-safe semantic correction.

        The returned feature remains in the original Euclidean feature space.
        This function never L2-normalizes the final output.
        """
        if features.dim() != 2:
            raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")

        base = features
        if disable or concept_bank is None or concept_bank.numel() == 0:
            return base

        alpha = self._bounded_alpha(
            phase=phase,
            disable_semantic_in_incremental=disable_semantic_in_incremental,
        ).to(device=base.device, dtype=base.dtype)

        if float(alpha.detach().item()) <= 0.0:
            return base

        base_concept = self.input_adapter(base)
        base_unit = self._normalize(base_concept)

        concept_bank = concept_bank.to(device=base.device, dtype=base.dtype)
        if concept_bank.size(-1) != self.concept_dim:
            raise ValueError(
                f"concept_bank last dim must equal concept_dim={self.concept_dim}, "
                f"got {concept_bank.size(-1)}"
            )

        semantic = self._attend(base_unit, concept_bank)
        semantic = self._normalize(self.semantic_refine(semantic))

        gate_in = torch.cat([base_unit, semantic], dim=-1)
        gate = self.concept_gate(gate_in)
        semantic_unit = self._normalize(gate * semantic + (1.0 - gate) * base_unit)

        delta_unit = semantic_unit - base_unit
        delta_norm = delta_unit.norm(dim=-1, keepdim=True)
        delta_unit = delta_unit / delta_norm.clamp_min(1e-6)

        delta = self.output_adapter(delta_unit)

        # Preserve radial scale from the original geometry feature.
        scale = base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out = base + alpha * scale * delta
        return out

    # =========================================================
    # Token affinity builders
    # =========================================================
    def _apply_topk(self, sim: torch.Tensor, topk_ratio: Optional[float] = None) -> torch.Tensor:
        ratio = self.topk_ratio if topk_ratio is None else float(topk_ratio)
        if ratio >= 1.0:
            return sim
        if ratio <= 0.0:
            return sim

        n = sim.size(-1)
        k = max(1, int(round(float(n) * ratio)))
        if k >= n:
            return sim

        vals, idx = torch.topk(sim, k=k, dim=-1)
        masked = torch.full_like(sim, fill_value=-1e4)
        masked.scatter_(-1, idx, vals)
        return masked

    def _row_normalize(self, mat: torch.Tensor) -> torch.Tensor:
        return mat / mat.sum(dim=-1, keepdim=True).clamp_min(self.affinity_eps)

    def _token_affinity(
        self,
        tokens: torch.Tensor,
        *,
        topk_ratio: Optional[float] = None,
        symmetric: Optional[bool] = None,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")

        symmetric = self.symmetric_affinity if symmetric is None else bool(symmetric)

        t = self._normalize(tokens)
        sim = torch.matmul(t, t.transpose(1, 2)) / self._safe_temperature()
        sim = self._apply_topk(sim, topk_ratio=topk_ratio)
        aff = F.softmax(sim, dim=-1)

        if symmetric:
            aff = 0.5 * (aff + aff.transpose(1, 2))
            aff = self._row_normalize(aff)

        return aff

    def _cross_affinity(
        self,
        spatial_tokens: torch.Tensor,
        spectral_tokens: torch.Tensor,
        *,
        topk_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        if spatial_tokens.dim() != 3 or spectral_tokens.dim() != 3:
            raise ValueError(
                f"Expected spatial_tokens and spectral_tokens as 3D tensors, "
                f"got {tuple(spatial_tokens.shape)} and {tuple(spectral_tokens.shape)}"
            )

        ts = self._normalize(spatial_tokens)
        tb = self._normalize(spectral_tokens)
        sim = torch.matmul(ts, tb.transpose(1, 2)) / self._safe_temperature()
        sim = self._apply_topk(sim, topk_ratio=topk_ratio)
        return F.softmax(sim, dim=-1)

    def build_token_relations(
        self,
        spectral_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        *,
        topk_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if spectral_tokens is None or spatial_tokens is None:
            raise ValueError("spectral_tokens and spatial_tokens are required to build token relations")

        spectral_affinity = self._token_affinity(spectral_tokens, topk_ratio=topk_ratio)
        spatial_affinity = self._token_affinity(spatial_tokens, topk_ratio=topk_ratio)
        cross_affinity = self._cross_affinity(
            spatial_tokens=spatial_tokens,
            spectral_tokens=spectral_tokens,
            topk_ratio=topk_ratio,
        )

        fused_tokens = torch.cat([spectral_tokens, spatial_tokens], dim=1)
        fused_affinity = self._token_affinity(fused_tokens, topk_ratio=topk_ratio)

        return {
            "spectral_affinity": spectral_affinity,
            "spatial_affinity": spatial_affinity,
            "cross_affinity": cross_affinity,
            "fused_affinity": fused_affinity,
        }

    @torch.no_grad()
    def summarize_class_token_relations(
        self,
        spectral_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rel = self.build_token_relations(spectral_tokens, spatial_tokens)
        return {k: v.mean(dim=0).detach() for k, v in rel.items()}

    # =========================================================
    # Token manifold preservation loss
    # =========================================================
    def _match_target_shape(self, current: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Supports:
            current [B,N,N], target [N,N]
            current [N,N],   target [N,N]
            current [B,N,M], target [N,M]
            current [N,M],   target [N,M]
        """
        if self.detach_relation_targets:
            target = target.detach()

        if current.dim() == 3 and target.dim() == 2:
            target = target.unsqueeze(0).expand(current.size(0), -1, -1)
        elif current.dim() == 2 and target.dim() == 3:
            target = target.mean(dim=0)

        if current.shape != target.shape:
            raise ValueError(
                f"Token relation target shape mismatch: current={tuple(current.shape)}, "
                f"target={tuple(target.shape)}"
            )
        return current, target

    def _relation_mse(
        self,
        current: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        reliability_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current is None or target is None:
            if current is not None:
                return self._zero(current)
            if target is not None:
                return torch.tensor(0.0, device=target.device, dtype=target.dtype)
            return torch.tensor(0.0)

        target = target.to(device=current.device, dtype=current.dtype)
        current, target = self._match_target_shape(current, target)

        err = (current - target).pow(2)
        if err.dim() >= 2:
            per_item = err.flatten(start_dim=1).mean(dim=1) if err.dim() == 3 else err.mean().view(1)
        else:
            per_item = err

        if reliability_weight is not None:
            w = reliability_weight.to(device=current.device, dtype=current.dtype).flatten()
            if per_item.numel() == 1 and w.numel() != 1:
                return per_item.mean() * w.mean().clamp(0.0, 1.0)
            if w.numel() == per_item.numel():
                return (per_item * w.clamp(0.0, 1.0)).sum() / w.clamp(0.0, 1.0).sum().clamp_min(1e-6)
            return per_item.mean() * w.mean().clamp(0.0, 1.0)

        return per_item.mean()

    def token_relation_loss(
        self,
        spectral_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        relation_targets: Dict[str, torch.Tensor],
        spectral_weight: float = 1.0,
        spatial_weight: float = 1.0,
        cross_weight: float = 1.0,
        fused_weight: float = 0.0,
        reliability_weight: Optional[torch.Tensor] = None,
        topk_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        current = self.build_token_relations(
            spectral_tokens=spectral_tokens,
            spatial_tokens=spatial_tokens,
            topk_ratio=topk_ratio,
        )
        current_mean = {k: v.mean(dim=0) for k, v in current.items()}

        device = spectral_tokens.device
        dtype = spectral_tokens.dtype
        zero = torch.tensor(0.0, device=device, dtype=dtype)

        if relation_targets is None or len(relation_targets) == 0:
            return {"total": zero, "spectral": zero, "spatial": zero, "cross": zero, "fused": zero}

        spec_loss = self._relation_mse(
            current_mean.get("spectral_affinity"),
            relation_targets.get("spectral_affinity"),
            reliability_weight=reliability_weight,
        ) if "spectral_affinity" in relation_targets else zero

        spat_loss = self._relation_mse(
            current_mean.get("spatial_affinity"),
            relation_targets.get("spatial_affinity"),
            reliability_weight=reliability_weight,
        ) if "spatial_affinity" in relation_targets else zero

        cross_loss = self._relation_mse(
            current_mean.get("cross_affinity"),
            relation_targets.get("cross_affinity"),
            reliability_weight=reliability_weight,
        ) if "cross_affinity" in relation_targets else zero

        fused_loss = zero
        if fused_weight > 0.0 and "fused_affinity" in relation_targets:
            fused_loss = self._relation_mse(
                current_mean.get("fused_affinity"),
                relation_targets.get("fused_affinity"),
                reliability_weight=reliability_weight,
            )

        total = (
            float(spectral_weight) * spec_loss
            + float(spatial_weight) * spat_loss
            + float(cross_weight) * cross_loss
            + float(fused_weight) * fused_loss
        )

        return {
            "total": total,
            "spectral": spec_loss,
            "spatial": spat_loss,
            "cross": cross_loss,
            "fused": fused_loss,
        }

    # =========================================================
    # Forward
    # =========================================================
    def forward(
        self,
        features: torch.Tensor,
        concept_bank: Optional[torch.Tensor] = None,
        phase: int = 0,
        disable: bool = False,
        spectral_tokens: Optional[torch.Tensor] = None,
        spatial_tokens: Optional[torch.Tensor] = None,
        return_token_relations: bool = False,
        disable_semantic_in_incremental: bool = True,
        topk_ratio: Optional[float] = None,
    ):
        refined = self.refine_features(
            features=features,
            concept_bank=concept_bank,
            phase=phase,
            disable=disable,
            disable_semantic_in_incremental=disable_semantic_in_incremental,
        )

        if not return_token_relations:
            return refined

        token_relations = None
        if spectral_tokens is not None and spatial_tokens is not None:
            token_relations = self.build_token_relations(
                spectral_tokens=spectral_tokens,
                spatial_tokens=spatial_tokens,
                topk_ratio=topk_ratio,
            )

        return {
            "features": refined,
            "token_relations": token_relations,
        }


