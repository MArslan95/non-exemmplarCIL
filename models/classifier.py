from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassifierOutput:
    """Explicit classifier output with a fixed class-column contract."""

    logits: Tensor          # [N, C]
    energy: Tensor          # [N, C], selected scoring mode
    class_ids: Tensor       # [C], global class IDs in exact column order
    factor_energy: Tensor   # [N, C], deployed factor-Gaussian energy
    quadratic: Tensor       # [N, C], factor Mahalanobis term / D
    volume: Tensor          # [N, C], log-volume term / D


def _unique_ids(
    values: Iterable[int],
    *,
    name: str,
    allow_empty: bool = False,
) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        class_id = int(value)
        if class_id < 0:
            raise ValueError(f"{name} contains negative class ID {class_id}")
        if class_id in seen:
            raise ValueError(f"{name} contains duplicate class ID {class_id}")
        seen.add(class_id)
        ids.append(class_id)
    if not ids and not allow_empty:
        raise ValueError(f"{name} is empty")
    return ids


def _validate_features(features: Tensor, feature_dim: int) -> Tensor:
    if not torch.is_tensor(features):
        raise TypeError("features must be a torch.Tensor")
    if features.dim() != 2 or features.size(1) != int(feature_dim):
        raise ValueError(
            f"features must be [N,{int(feature_dim)}], "
            f"got {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("features contain NaN/Inf")
    return features


class FactorGeometryEnergyClassifier(nn.Module):
    """Parameter-free equal-prior classifier for the HSI factor GeometryBank.

    Deployed class model
    --------------------
        p(z | c) = N(mu_c, L_c L_c^T + Psi_c)

    where
        z = [z_s ; z_p]
        Psi_c = diag(psi_c,s I_Ds, psi_c,p I_Dp).

    The GeometryBank owns all class statistics and all factor-energy
    calculations. This classifier owns only:

    1. static GeometryBank contract validation;
    2. explicit class-column ordering;
    3. conversion from energy to logits;
    4. global/local label mapping;
    5. classification and old/new invasion diagnostics.

    Raw spectral-shape statistics p(h | c) never enter inference here. They
    belong to the pair-risk module used by the training loss.

    No trainable classifier weights, class-specific biases, task heads,
    reliability penalties, or old/new calibration terms are used.
    """

    FACTOR_GEOMETRY = "factor_geometry"
    QUADRATIC_ONLY_ABLATION = "quadratic_only"
    DIAGONAL_GEOMETRY_ABLATION = "diagonal_geometry"
    PROTOTYPE_ABLATION = "prototype"

    SUPPORTED_MODES = (
        FACTOR_GEOMETRY,
        QUADRATIC_ONLY_ABLATION,
        DIAGONAL_GEOMETRY_ABLATION,
        PROTOTYPE_ABLATION,
    )

    def __init__(
        self,
        *,
        feature_dim: int,
        temperature: float = 1.0,
        expected_spectral_dim: Optional[int] = None,
        expected_spatial_dim: Optional[int] = None,
        expected_bank_schema_version: Optional[int] = 2,
        require_bound_contract: bool = False,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        self.expected_spectral_dim = (
            None if expected_spectral_dim is None
            else int(expected_spectral_dim)
        )
        self.expected_spatial_dim = (
            None if expected_spatial_dim is None
            else int(expected_spatial_dim)
        )
        if (
            self.expected_spectral_dim is not None
            and self.expected_spectral_dim <= 0
        ):
            raise ValueError("expected_spectral_dim must be positive")
        if (
            self.expected_spatial_dim is not None
            and self.expected_spatial_dim <= 0
        ):
            raise ValueError("expected_spatial_dim must be positive")
        if (
            self.expected_spectral_dim is not None
            and self.expected_spatial_dim is not None
            and self.expected_spectral_dim + self.expected_spatial_dim
            != self.feature_dim
        ):
            raise ValueError(
                "expected spectral/spatial dimensions do not sum to feature_dim"
            )

        self.expected_bank_schema_version = (
            None if expected_bank_schema_version is None
            else int(expected_bank_schema_version)
        )

        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.register_buffer(
            "_temperature",
            torch.tensor(temperature, dtype=torch.float32),
        )

        self.require_bound_contract = bool(require_bound_contract)
        self._bound_bank_contract_digest: Optional[str] = None
        self._last_class_ids: list[int] = []

    # ------------------------------------------------------------------
    # Global temperature
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return float(self._temperature.item())

    @torch.no_grad()
    def set_temperature(
        self,
        value: float,
        *,
        allow_after_binding: bool = False,
    ) -> None:
        """Set one global temperature, normally before base handoff."""
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if (
            self._bound_bank_contract_digest is not None
            and not allow_after_binding
        ):
            raise RuntimeError(
                "temperature is frozen after binding the base bank contract"
            )
        self._temperature.fill_(value)

    # ------------------------------------------------------------------
    # Mode and label contracts
    # ------------------------------------------------------------------

    @classmethod
    def normalize_mode(cls, mode: str) -> str:
        token = str(mode or cls.FACTOR_GEOMETRY).strip().lower()
        token = token.replace("-", "_").replace(" ", "_")
        aliases = {
            "factor": cls.FACTOR_GEOMETRY,
            "factor_geometry": cls.FACTOR_GEOMETRY,
            "geometry": cls.FACTOR_GEOMETRY,
            "geometry_only": cls.FACTOR_GEOMETRY,
            "factor_gaussian": cls.FACTOR_GEOMETRY,
            "quadratic": cls.QUADRATIC_ONLY_ABLATION,
            "quadratic_only": cls.QUADRATIC_ONLY_ABLATION,
            "no_volume": cls.QUADRATIC_ONLY_ABLATION,
            "diagonal": cls.DIAGONAL_GEOMETRY_ABLATION,
            "diagonal_geometry": cls.DIAGONAL_GEOMETRY_ABLATION,
            "prototype": cls.PROTOTYPE_ABLATION,
            "nearest_mean": cls.PROTOTYPE_ABLATION,
        }
        if token not in aliases:
            raise ValueError(
                f"unsupported classifier mode {mode!r}; "
                f"supported={cls.SUPPORTED_MODES}"
            )
        return aliases[token]

    @staticmethod
    def global_to_local_labels(
        labels_global: Tensor,
        class_ids: Union[Sequence[int], Tensor],
    ) -> Tensor:
        """Map arbitrary global IDs to exact energy-column indices."""
        if not torch.is_tensor(labels_global):
            raise TypeError("labels_global must be a tensor")

        labels = labels_global.long().flatten()
        classes = torch.as_tensor(
            class_ids,
            device=labels.device,
            dtype=torch.long,
        ).flatten()
        if classes.numel() == 0:
            raise ValueError("class_ids is empty")
        if classes.unique().numel() != classes.numel():
            raise ValueError("class_ids contains duplicates")

        matches = labels[:, None].eq(classes[None, :])
        match_count = matches.sum(dim=1)
        if bool(match_count.ne(1).any()):
            missing = labels[match_count.eq(0)].detach().cpu().unique().tolist()
            raise RuntimeError(
                f"labels contain classes outside classifier class_ids: {missing}"
            )
        return matches.to(torch.long).argmax(dim=1)

    @staticmethod
    def local_to_global_labels(
        labels_local: Tensor,
        class_ids: Union[Sequence[int], Tensor],
    ) -> Tensor:
        local = labels_local.long().flatten()
        classes = torch.as_tensor(
            class_ids,
            device=local.device,
            dtype=torch.long,
        ).flatten()
        if classes.numel() == 0:
            raise ValueError("class_ids is empty")
        if local.numel() and (
            int(local.min().item()) < 0
            or int(local.max().item()) >= classes.numel()
        ):
            raise RuntimeError("labels_local is outside class-column range")
        return classes.index_select(0, local)

    @staticmethod
    def cross_entropy_from_global_targets(
        logits: Tensor,
        targets_global: Tensor,
        class_ids: Union[Sequence[int], Tensor],
        *,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be [N,C]")
        local = FactorGeometryEnergyClassifier.global_to_local_labels(
            targets_global,
            class_ids,
        )
        if local.numel() != logits.size(0):
            raise ValueError("target/logit batch mismatch")
        return F.cross_entropy(
            logits,
            local,
            label_smoothing=float(label_smoothing),
        )

    # ------------------------------------------------------------------
    # Static bank contract
    # ------------------------------------------------------------------

    @staticmethod
    def _require_bank_api(geometry_bank: Any) -> None:
        if geometry_bank is None:
            raise ValueError("geometry_bank is required")
        required_methods = (
            "energy_matrix",
            "get_bank",
            "get_class_row",
            "valid_mask",
            "assert_valid",
        )
        missing = [
            name for name in required_methods
            if not callable(getattr(geometry_bank, name, None))
        ]
        if missing:
            raise TypeError(
                "geometry_bank does not implement the factor-geometry API; "
                f"missing methods {missing}"
            )

    @staticmethod
    def _bank_static_contract_state(geometry_bank: Any) -> Dict[str, Any]:
        return {
            "bank_class": type(geometry_bank).__name__,
            "schema_version": int(
                getattr(geometry_bank, "SCHEMA_VERSION", -1)
            ),
            "classification_factorization": "p(z|c)",
            "spectral_dim": int(
                getattr(geometry_bank, "spectral_dim", -1)
            ),
            "spatial_dim": int(
                getattr(geometry_bank, "spatial_dim", -1)
            ),
            "feature_dim": int(
                getattr(geometry_bank, "feature_dim", -1)
            ),
            "maximum_rank": int(
                getattr(geometry_bank, "maximum_rank", -1)
            ),
            "volume_weight": float(
                getattr(geometry_bank, "volume_weight", float("nan"))
            ),
            "variance_floor_absolute": float(
                getattr(
                    geometry_bank,
                    "variance_floor_absolute",
                    float("nan"),
                )
            ),
            "spectral_shape_dim": int(
                getattr(geometry_bank, "spectral_shape_dim", -1)
            ),
            "uses_trainable_classifier_weights": False,
            "uses_class_specific_bias": False,
        }

    @classmethod
    def bank_contract_digest(cls, geometry_bank: Any) -> str:
        cls._require_bank_api(geometry_bank)
        state = cls._bank_static_contract_state(geometry_bank)
        return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()

    def _validate_static_bank_contract(
        self,
        geometry_bank: Any,
        *,
        feature_device: torch.device,
        enforce_binding: bool = True,
    ) -> Dict[str, Any]:
        self._require_bank_api(geometry_bank)
        state = self._bank_static_contract_state(geometry_bank)

        if (
            self.expected_bank_schema_version is not None
            and state["schema_version"]
            != self.expected_bank_schema_version
        ):
            raise RuntimeError(
                "classifier and GeometryBank schema versions differ: "
                f"{self.expected_bank_schema_version} vs "
                f"{state['schema_version']}"
            )
        if state["feature_dim"] != self.feature_dim:
            raise RuntimeError(
                "classifier and GeometryBank feature dimensions differ: "
                f"{self.feature_dim} vs {state['feature_dim']}"
            )
        if (
            self.expected_spectral_dim is not None
            and state["spectral_dim"] != self.expected_spectral_dim
        ):
            raise RuntimeError(
                "classifier and GeometryBank spectral dimensions differ"
            )
        if (
            self.expected_spatial_dim is not None
            and state["spatial_dim"] != self.expected_spatial_dim
        ):
            raise RuntimeError(
                "classifier and GeometryBank spatial dimensions differ"
            )

        bank_device = torch.device(getattr(geometry_bank, "device"))
        if bank_device != torch.device(feature_device):
            raise RuntimeError(
                f"features are on {feature_device}, GeometryBank is on "
                f"{bank_device}"
            )

        digest = self.bank_contract_digest(geometry_bank)
        if (
            enforce_binding
            and self.require_bound_contract
            and self._bound_bank_contract_digest is None
        ):
            raise RuntimeError(
                "classifier requires a bound GeometryBank contract"
            )
        if (
            self._bound_bank_contract_digest is not None
            and digest != self._bound_bank_contract_digest
        ):
            raise RuntimeError(
                "GeometryBank static energy contract changed after binding"
            )
        return {**state, "contract_digest": digest}

    @torch.no_grad()
    def bind_geometry_bank_contract(
        self,
        geometry_bank: Any,
        *,
        require_committed_rows: bool = True,
        enforce_after_binding: bool = True,
        overwrite: bool = False,
    ) -> str:
        """Bind to static geometry/energy settings, never to mutable rows."""
        self._require_bank_api(geometry_bank)
        if require_committed_rows and not bool(
            geometry_bank.valid_mask().any().item()
        ):
            raise RuntimeError(
                "cannot bind before at least one valid class row is committed"
            )
        self._validate_static_bank_contract(
            geometry_bank,
            feature_device=torch.device(getattr(geometry_bank, "device")),
            enforce_binding=False,
        )
        digest = self.bank_contract_digest(geometry_bank)
        if (
            self._bound_bank_contract_digest is not None
            and self._bound_bank_contract_digest != digest
            and not overwrite
        ):
            raise RuntimeError(
                "classifier is already bound to another bank contract"
            )
        self._bound_bank_contract_digest = digest
        if enforce_after_binding:
            self.require_bound_contract = True
        return digest

    @property
    def bound_bank_contract_digest(self) -> Optional[str]:
        return self._bound_bank_contract_digest

    def classifier_contract(self) -> Dict[str, Any]:
        return {
            "classifier": type(self).__name__,
            "classification_factorization": "p(z|c)",
            "spectral_relation_usage": "pair-risk margins only",
            "feature_dim": self.feature_dim,
            "temperature": self.temperature,
            "logit_rule": "logits=-energy/global_temperature",
            "deployed_mode": self.FACTOR_GEOMETRY,
            "supports_temporary_rows": True,
            "uses_trainable_classifier_weights": False,
            "uses_class_specific_bias": False,
            "uses_task_specific_head": False,
            "uses_reliability_logit_penalty": False,
            "uses_raw_spectra_at_inference": False,
            "bound_bank_contract_digest":
                self._bound_bank_contract_digest,
        }

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "bound_bank_contract_digest":
                self._bound_bank_contract_digest,
            "require_bound_contract": self.require_bound_contract,
            "classifier_contract": self.classifier_contract(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            self._bound_bank_contract_digest = None
            return
        stored = state.get("classifier_contract")
        if isinstance(stored, Mapping):
            immutable = (
                "classification_factorization",
                "feature_dim",
                "uses_trainable_classifier_weights",
                "uses_class_specific_bias",
                "uses_raw_spectra_at_inference",
            )
            current = self.classifier_contract()
            mismatches = [
                f"{name}: checkpoint={stored.get(name)!r}, "
                f"current={current.get(name)!r}"
                for name in immutable
                if name in stored and stored.get(name) != current.get(name)
            ]
            if mismatches:
                raise RuntimeError(
                    "classifier contract mismatch: " + "; ".join(mismatches)
                )
        digest = state.get("bound_bank_contract_digest")
        self._bound_bank_contract_digest = (
            None if digest is None else str(digest)
        )
        self.require_bound_contract = bool(
            state.get("require_bound_contract", self.require_bound_contract)
        )

    # ------------------------------------------------------------------
    # Class-row validation and scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_seen_classes(
        geometry_bank: Any,
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> list[int]:
        valid = geometry_bank.valid_mask().detach().cpu().bool()
        committed = torch.nonzero(
            valid,
            as_tuple=False,
        ).flatten().tolist()
        temporary = (
            []
            if temporary_rows is None
            else [int(class_id) for class_id in temporary_rows]
        )
        return sorted(set(committed) | set(temporary))

    def _validate_rows(
        self,
        geometry_bank: Any,
        *,
        class_ids: Sequence[int],
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]],
    ) -> None:
        temporary_ids = (
            set()
            if temporary_rows is None
            else {int(class_id) for class_id in temporary_rows}
        )
        class_set = set(class_ids)
        unknown = sorted(temporary_ids - class_set)
        if unknown:
            raise RuntimeError(
                f"temporary_rows contain classes outside class_ids: {unknown}"
            )

        committed_ids = [
            class_id for class_id in class_ids
            if class_id not in temporary_ids
        ]
        if committed_ids:
            report = geometry_bank.assert_valid(
                committed_ids,
                strict=False,
            )
            if not bool(report.get("ok", False)):
                raise RuntimeError(
                    "invalid committed GeometryBank rows: "
                    + "; ".join(report.get("errors", []))
                )

    @staticmethod
    def _temporary_or_committed_row(
        geometry_bank: Any,
        class_id: int,
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]],
    ) -> Mapping[str, Any]:
        if temporary_rows is not None and class_id in temporary_rows:
            return temporary_rows[class_id]
        return geometry_bank.get_class_row(class_id, clone=False)

    def _stack_ablation_rows(
        self,
        geometry_bank: Any,
        class_ids: Sequence[int],
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        means: list[Tensor] = []
        psi_s: list[Tensor] = []
        psi_p: list[Tensor] = []

        spectral_dim = int(getattr(geometry_bank, "spectral_dim"))
        spatial_dim = int(getattr(geometry_bank, "spatial_dim"))
        for class_id in class_ids:
            row = self._temporary_or_committed_row(
                geometry_bank,
                class_id,
                temporary_rows,
            )
            mean = torch.as_tensor(
                row["mean"],
                device=device,
                dtype=dtype,
            ).flatten()
            residual_s = torch.as_tensor(
                row["residual_var_spectral"],
                device=device,
                dtype=dtype,
            ).reshape(())
            residual_p = torch.as_tensor(
                row["residual_var_spatial"],
                device=device,
                dtype=dtype,
            ).reshape(())

            if mean.shape != (self.feature_dim,):
                raise RuntimeError(
                    f"class {class_id}: mean has shape {tuple(mean.shape)}, "
                    f"expected {(self.feature_dim,)}"
                )
            if not torch.isfinite(mean).all():
                raise RuntimeError(f"class {class_id}: mean contains NaN/Inf")
            if (
                not torch.isfinite(residual_s)
                or not torch.isfinite(residual_p)
                or float(residual_s.item()) <= 0.0
                or float(residual_p.item()) <= 0.0
            ):
                raise RuntimeError(
                    f"class {class_id}: invalid branch residual variance"
                )
            means.append(mean)
            psi_s.append(residual_s)
            psi_p.append(residual_p)

        return {
            "means": torch.stack(means),
            "psi_s": torch.stack(psi_s),
            "psi_p": torch.stack(psi_p),
            "spectral_dim": torch.tensor(
                spectral_dim,
                device=device,
                dtype=torch.long,
            ),
            "spatial_dim": torch.tensor(
                spatial_dim,
                device=device,
                dtype=torch.long,
            ),
        }

    def compute_energy(
        self,
        features: Tensor,
        *,
        class_ids: Sequence[int],
        geometry_bank: Any,
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
        mode: str = FACTOR_GEOMETRY,
    ) -> Tuple[ClassifierOutput, Dict[str, Any]]:
        features = _validate_features(features, self.feature_dim)
        ids = _unique_ids(class_ids, name="class_ids")
        selected_mode = self.normalize_mode(mode)

        contract = self._validate_static_bank_contract(
            geometry_bank,
            feature_device=features.device,
        )
        self._validate_rows(
            geometry_bank,
            class_ids=ids,
            temporary_rows=temporary_rows,
        )

        factor = geometry_bank.energy_matrix(
            features,
            ids,
            rows=temporary_rows,
        )
        expected_ids = torch.tensor(
            ids,
            device=features.device,
            dtype=torch.long,
        )
        if not torch.equal(factor.class_ids, expected_ids):
            raise RuntimeError(
                "GeometryBank returned a different class-column order"
            )
        expected_shape = (features.size(0), len(ids))
        for name, value in (
            ("factor energy", factor.energy),
            ("quadratic", factor.quadratic),
            ("volume", factor.volume),
        ):
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"{name} shape {tuple(value.shape)} != {expected_shape}"
                )
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{name} contains NaN/Inf")

        selected_energy = factor.energy
        ablation_parts: Dict[str, Tensor] = {}

        if selected_mode == self.QUADRATIC_ONLY_ABLATION:
            selected_energy = factor.quadratic

        elif selected_mode in (
            self.DIAGONAL_GEOMETRY_ABLATION,
            self.PROTOTYPE_ABLATION,
        ):
            rows = self._stack_ablation_rows(
                geometry_bank,
                ids,
                temporary_rows,
                device=features.device,
                dtype=features.dtype,
            )
            means = rows["means"]
            delta = features[:, None, :] - means[None, :, :]

            prototype_energy = (
                delta.square().sum(dim=2) / float(self.feature_dim)
            )
            ablation_parts["prototype_energy"] = prototype_energy

            if selected_mode == self.PROTOTYPE_ABLATION:
                selected_energy = prototype_energy
            else:
                spectral_dim = int(rows["spectral_dim"].item())
                spatial_dim = int(rows["spatial_dim"].item())
                psi_s = rows["psi_s"].clamp_min(
                    float(contract["variance_floor_absolute"])
                )
                psi_p = rows["psi_p"].clamp_min(
                    float(contract["variance_floor_absolute"])
                )
                quadratic_diag = (
                    delta[:, :, :spectral_dim].square().sum(dim=2)
                    / psi_s.view(1, -1)
                    + delta[:, :, spectral_dim:].square().sum(dim=2)
                    / psi_p.view(1, -1)
                )
                volume_diag = (
                    spectral_dim * psi_s.log()
                    + spatial_dim * psi_p.log()
                ).view(1, -1).expand_as(quadratic_diag)
                diagonal_energy = (
                    quadratic_diag
                    + float(contract["volume_weight"]) * volume_diag
                ) / float(self.feature_dim)
                ablation_parts["diagonal_energy"] = diagonal_energy
                selected_energy = diagonal_energy

        logits = -selected_energy / self._temperature.to(
            device=selected_energy.device,
            dtype=selected_energy.dtype,
        )
        if not torch.isfinite(logits).all():
            raise RuntimeError("geometry logits contain NaN/Inf")

        output = ClassifierOutput(
            logits=logits,
            energy=selected_energy,
            class_ids=factor.class_ids,
            factor_energy=factor.energy,
            quadratic=factor.quadratic,
            volume=factor.volume,
        )
        parts: Dict[str, Any] = {
            "mode": selected_mode,
            "contract_digest": contract["contract_digest"],
            "classification_factorization": "p(z|c)",
            "spectral_shape_used_for_inference": False,
            "uses_temporary_rows": temporary_rows is not None,
            "bank_mutated": False,
            **ablation_parts,
        }
        return output, parts

    def compute_logits(
        self,
        features: Tensor,
        *,
        class_ids: Sequence[int],
        geometry_bank: Any,
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
        mode: str = FACTOR_GEOMETRY,
        return_parts: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        output, parts = self.compute_energy(
            features,
            class_ids=class_ids,
            geometry_bank=geometry_bank,
            temporary_rows=temporary_rows,
            mode=mode,
        )
        self.assert_logits_valid(
            output.logits,
            class_ids=output.class_ids,
        )
        if not return_parts:
            return output.logits
        return {
            "logits": output.logits,
            "energy": output.energy,
            "class_ids": output.class_ids,
            "factor_energy": output.factor_energy,
            "quadratic": output.quadratic,
            "volume": output.volume,
            **parts,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def assert_logits_valid(
        logits: Tensor,
        *,
        class_ids: Union[Sequence[int], Tensor],
        targets_local: Optional[Tensor] = None,
    ) -> None:
        classes = torch.as_tensor(
            class_ids,
            device=logits.device,
            dtype=torch.long,
        ).flatten()
        if logits.dim() != 2:
            raise RuntimeError("logits must be [N,C]")
        if logits.size(1) != classes.numel():
            raise RuntimeError(
                "logit width does not match class_ids"
            )
        if classes.unique().numel() != classes.numel():
            raise RuntimeError("class_ids contains duplicates")
        if not torch.isfinite(logits).all():
            raise RuntimeError("logits contain NaN/Inf")
        if targets_local is not None:
            targets = targets_local.to(logits.device).long().flatten()
            if targets.numel() != logits.size(0):
                raise RuntimeError("target/logit batch mismatch")
            if targets.numel() and (
                int(targets.min().item()) < 0
                or int(targets.max().item()) >= logits.size(1)
            ):
                raise RuntimeError(
                    "targets_local must use class-column indices"
                )

    @staticmethod
    def energy_margin_statistics(
        energy: Tensor,
        targets_local: Tensor,
    ) -> Dict[str, Tensor]:
        if energy.dim() != 2:
            raise ValueError("energy must be [N,C]")
        labels = targets_local.to(energy.device).long().flatten()
        if labels.numel() != energy.size(0):
            raise ValueError("energy/target batch mismatch")
        if energy.size(1) < 2:
            raise ValueError(
                "margin statistics require at least two classes"
            )

        true_energy = energy.gather(1, labels[:, None]).squeeze(1)
        rival_energy = energy.clone()
        rival_energy.scatter_(1, labels[:, None], float("inf"))
        nearest_rival_energy, nearest_rival = rival_energy.min(dim=1)
        margin = nearest_rival_energy - true_energy
        return {
            "true_energy": true_energy,
            "nearest_rival_energy": nearest_rival_energy,
            "nearest_rival_local": nearest_rival,
            "margin": margin,
            "mean_margin": margin.mean(),
            "minimum_margin": margin.min(),
            "q01_margin": torch.quantile(margin, 0.01),
            "q05_margin": torch.quantile(margin, 0.05),
            "violation_rate": margin.lt(0.0).float().mean(),
            "accuracy": energy.argmin(dim=1).eq(labels).float().mean(),
        }

    @staticmethod
    def _phase_masks(
        class_ids: Tensor,
        *,
        old_classes: Optional[Iterable[int]],
        new_classes: Optional[Iterable[int]],
    ) -> Tuple[Tensor, Tensor]:
        ids = [int(value) for value in class_ids.detach().cpu().tolist()]
        old_set = (
            None if old_classes is None
            else {int(value) for value in old_classes}
        )
        new_set = (
            None if new_classes is None
            else {int(value) for value in new_classes}
        )

        if old_set is None and new_set is None:
            empty = torch.zeros(
                len(ids),
                device=class_ids.device,
                dtype=torch.bool,
            )
            return empty, empty.clone()
        if old_set is None:
            old_set = set(ids) - set(new_set or set())
        if new_set is None:
            new_set = set(ids) - set(old_set)

        assert old_set is not None and new_set is not None
        if old_set & new_set or old_set | new_set != set(ids):
            raise RuntimeError(
                "old_classes and new_classes must partition class_ids"
            )
        return (
            torch.tensor(
                [class_id in old_set for class_id in ids],
                device=class_ids.device,
                dtype=torch.bool,
            ),
            torch.tensor(
                [class_id in new_set for class_id in ids],
                device=class_ids.device,
                dtype=torch.bool,
            ),
        )

    @torch.no_grad()
    def classifier_diagnostics(
        self,
        *,
        output: ClassifierOutput,
        targets_local: Optional[Tensor] = None,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        predicted_local = output.energy.argmin(dim=1)
        predicted_global = self.local_to_global_labels(
            predicted_local,
            output.class_ids,
        )
        counts = torch.bincount(
            predicted_local,
            minlength=output.class_ids.numel(),
        ).detach().cpu()

        class_list = output.class_ids.detach().cpu().tolist()
        result: Dict[str, Any] = {
            "classification_factorization": "p(z|c)",
            "spectral_shape_used_for_inference": False,
            "prediction_distribution": {
                int(class_list[index]): int(counts[index].item())
                for index in range(len(class_list))
            },
            "predicted_global": predicted_global,
            "uses_trainable_classifier_weights": False,
            "uses_class_specific_bias": False,
        }
        if targets_local is None:
            return result

        targets = targets_local.to(output.energy.device).long().flatten()
        stats = self.energy_margin_statistics(output.energy, targets)
        predictions = output.energy.argmin(dim=1)
        correct = predictions.eq(targets)
        result.update(
            {
                "accuracy": float(correct.float().mean().item()),
                "mean_margin": float(stats["mean_margin"].item()),
                "minimum_margin": float(stats["minimum_margin"].item()),
                "q01_margin": float(stats["q01_margin"].item()),
                "q05_margin": float(stats["q05_margin"].item()),
                "violation_rate": float(
                    stats["violation_rate"].item()
                ),
                "mean_true_quadratic": float(
                    output.quadratic.gather(
                        1,
                        targets[:, None],
                    ).mean().item()
                ),
                "mean_true_volume": float(
                    output.volume.gather(
                        1,
                        targets[:, None],
                    ).mean().item()
                ),
            }
        )

        old_mask, new_mask = self._phase_masks(
            output.class_ids,
            old_classes=old_classes,
            new_classes=new_classes,
        )
        if bool(old_mask.any()) or bool(new_mask.any()):
            old_samples = old_mask.index_select(0, targets)
            new_samples = new_mask.index_select(0, targets)
            result["old_accuracy"] = (
                float(correct[old_samples].float().mean().item())
                if bool(old_samples.any())
                else float("nan")
            )
            result["new_accuracy"] = (
                float(correct[new_samples].float().mean().item())
                if bool(new_samples.any())
                else float("nan")
            )
            result["old_to_new_invasion"] = (
                float(
                    new_mask.index_select(
                        0,
                        predictions[old_samples],
                    ).float().mean().item()
                )
                if bool(old_samples.any()) and bool(new_mask.any())
                else 0.0
            )
            result["new_to_old_invasion"] = (
                float(
                    old_mask.index_select(
                        0,
                        predictions[new_samples],
                    ).float().mean().item()
                )
                if bool(new_samples.any()) and bool(old_mask.any())
                else 0.0
            )

            old_accuracy = result["old_accuracy"]
            new_accuracy = result["new_accuracy"]
            if (
                math.isfinite(old_accuracy)
                and math.isfinite(new_accuracy)
                and old_accuracy + new_accuracy > 0.0
            ):
                result["old_new_harmonic_mean"] = (
                    2.0 * old_accuracy * new_accuracy
                    / (old_accuracy + new_accuracy)
                )
            else:
                result["old_new_harmonic_mean"] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Forward interfaces
    # ------------------------------------------------------------------

    def forward_from_outputs(
        self,
        model_output: Mapping[str, Any],
        *,
        geometry_bank: Any,
        class_ids: Optional[Iterable[int]] = None,
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[Tensor, Dict[str, Any]]:
        feature = None
        for key in (
            "joint_feature",
            "joint_features",
            "geometry_features",
            "features",
        ):
            if key in model_output:
                feature = model_output[key]
                break
        if feature is None:
            raise KeyError(
                "model_output must contain one of: joint_feature, "
                "joint_features, geometry_features, features"
            )
        return self.forward(
            feature,
            geometry_bank=geometry_bank,
            class_ids=class_ids,
            temporary_rows=temporary_rows,
            **kwargs,
        )

    def forward(
        self,
        features: Tensor,
        *,
        geometry_bank: Any,
        class_ids: Optional[Iterable[int]] = None,
        temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
        mode: str = FACTOR_GEOMETRY,
        targets: Optional[Tensor] = None,
        targets_are_global: bool = False,
        old_classes: Optional[Iterable[int]] = None,
        new_classes: Optional[Iterable[int]] = None,
        return_energy: bool = False,
        return_parts: bool = False,
        return_diagnostics: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        ids = (
            self._infer_seen_classes(geometry_bank, temporary_rows)
            if class_ids is None
            else _unique_ids(class_ids, name="class_ids")
        )
        if not ids:
            raise RuntimeError("no class geometry rows are available")
        self._last_class_ids = ids

        output, parts = self.compute_energy(
            features,
            class_ids=ids,
            geometry_bank=geometry_bank,
            temporary_rows=temporary_rows,
            mode=mode,
        )

        local_targets: Optional[Tensor] = None
        if targets is not None:
            local_targets = (
                self.global_to_local_labels(
                    targets,
                    output.class_ids,
                )
                if targets_are_global
                else targets.to(output.logits.device).long().flatten()
            )
        self.assert_logits_valid(
            output.logits,
            class_ids=output.class_ids,
            targets_local=local_targets,
        )

        if not (
            return_energy
            or return_parts
            or return_diagnostics
        ):
            return output.logits

        result: Dict[str, Any] = {
            "logits": output.logits,
            "energy": output.energy,
            "class_ids": output.class_ids,
            "factor_energy": output.factor_energy,
            "quadratic": output.quadratic,
            "volume": output.volume,
            **parts,
        }
        if local_targets is not None:
            result["targets_local"] = local_targets

        if return_diagnostics:
            result["diagnostics"] = self.classifier_diagnostics(
                output=output,
                targets_local=local_targets,
                old_classes=old_classes,
                new_classes=new_classes,
            )
        return result


GeometryEnergyClassifier = FactorGeometryEnergyClassifier
TransportClosedGeometryClassifier = FactorGeometryEnergyClassifier














# from __future__ import annotations

# from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
# import hashlib

# import torch
# import torch.nn as nn


# Tensor = torch.Tensor


# def _unique_ids(
#     values: Iterable[int],
#     *,
#     name: str,
#     allow_empty: bool = False,
# ) -> list[int]:
#     ids: list[int] = []
#     seen: set[int] = set()
#     for value in values:
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError(f"{name} contains negative class ID {class_id}")
#         if class_id in seen:
#             raise ValueError(f"{name} contains duplicate class ID {class_id}")
#         seen.add(class_id)
#         ids.append(class_id)
#     if not ids and not allow_empty:
#         raise ValueError(f"{name} is empty")
#     return ids


# def _validate_features(features: Tensor, d_model: int) -> Tensor:
#     if not torch.is_tensor(features):
#         raise TypeError("features must be a tensor")
#     if features.dim() != 2 or features.size(1) != int(d_model):
#         raise ValueError(
#             f"features must be [N,{int(d_model)}], got {tuple(features.shape)}"
#         )
#     if not torch.isfinite(features).all():
#         raise RuntimeError("features contain NaN/Inf")
#     return features


# class SpectralConditionedEnergyClassifier(nn.Module):
#     """Parameter-free classifier for p(s|c) p(z|s,c).

#     The GeometryBank owns all class statistics and all energy calculations.
#     This class owns only:

#     1. immutable bank-contract validation;
#     2. explicit class-column ordering;
#     3. conversion from energy to logits;
#     4. label mapping and old/new boundary diagnostics.

#     No classifier weight, prototype weight, class-specific bias, task head,
#     calibration network, reliability penalty, or candidate-row correction is
#     learned here.
#     """

#     JOINT_MODE = "spectral_conditioned_joint"
#     CONDITIONAL_FEATURE_ABLATION = "conditional_feature_only"
#     SPECTRAL_ABLATION = "spectral_only"
#     SUPPORTED_MODES = (
#         JOINT_MODE,
#         CONDITIONAL_FEATURE_ABLATION,
#         SPECTRAL_ABLATION,
#     )

#     def __init__(
#         self,
#         *,
#         d_model: int,
#         temperature: float = 1.0,
#         expected_spectral_anchor_dim: Optional[int] = None,
#         expected_bank_schema_version: Optional[int] = 1,
#         require_bound_contract: bool = False,
#     ) -> None:
#         super().__init__()
#         self.d_model = int(d_model)
#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive")

#         temperature = float(temperature)
#         if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
#             raise ValueError("temperature must be finite and positive")
#         self.register_buffer(
#             "_temperature",
#             torch.tensor(temperature, dtype=torch.float32),
#         )

#         self.expected_spectral_anchor_dim = (
#             None
#             if expected_spectral_anchor_dim is None
#             else int(expected_spectral_anchor_dim)
#         )
#         if (
#             self.expected_spectral_anchor_dim is not None
#             and self.expected_spectral_anchor_dim <= 0
#         ):
#             raise ValueError("expected_spectral_anchor_dim must be positive")

#         self.expected_bank_schema_version = (
#             None
#             if expected_bank_schema_version is None
#             else int(expected_bank_schema_version)
#         )
#         self.require_bound_contract = bool(require_bound_contract)
#         self._bound_bank_contract_digest: Optional[str] = None
#         self._last_seen_classes: list[int] = []

#     @property
#     def temperature(self) -> float:
#         return float(self._temperature.item())

#     @torch.no_grad()
#     def set_temperature(
#         self,
#         value: float,
#         *,
#         allow_after_binding: bool = False,
#     ) -> None:
#         """Set one global temperature, normally before base handoff."""
#         value = float(value)
#         if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
#             raise ValueError("temperature must be finite and positive")
#         if (
#             self._bound_bank_contract_digest is not None
#             and not allow_after_binding
#         ):
#             raise RuntimeError(
#                 "temperature is frozen after binding the base bank contract"
#             )
#         self._temperature.fill_(value)

#     @staticmethod
#     def normalize_mode(mode: str) -> str:
#         token = str(
#             mode or SpectralConditionedEnergyClassifier.JOINT_MODE
#         ).strip().lower().replace("-", "_")
#         aliases = {
#             "joint": SpectralConditionedEnergyClassifier.JOINT_MODE,
#             "joint_energy": SpectralConditionedEnergyClassifier.JOINT_MODE,
#             "spectral_conditioned_joint":
#                 SpectralConditionedEnergyClassifier.JOINT_MODE,
#             "hsi_joint_geometry":
#                 SpectralConditionedEnergyClassifier.JOINT_MODE,
#             "conditional_feature_only":
#                 SpectralConditionedEnergyClassifier.CONDITIONAL_FEATURE_ABLATION,
#             "feature_only_ablation":
#                 SpectralConditionedEnergyClassifier.CONDITIONAL_FEATURE_ABLATION,
#             "spectral_only":
#                 SpectralConditionedEnergyClassifier.SPECTRAL_ABLATION,
#             "spectral_only_ablation":
#                 SpectralConditionedEnergyClassifier.SPECTRAL_ABLATION,
#         }
#         if token not in aliases:
#             raise ValueError(
#                 f"unsupported classifier mode {mode!r}; "
#                 f"supported={SpectralConditionedEnergyClassifier.SUPPORTED_MODES}"
#             )
#         return aliases[token]

#     @staticmethod
#     def global_to_local_labels(
#         labels: Tensor,
#         seen_classes: Sequence[int],
#     ) -> Tensor:
#         if not torch.is_tensor(labels):
#             raise TypeError("labels must be a tensor")
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         flat = labels.long().flatten()
#         local = torch.full_like(flat, -1)
#         for column, class_id in enumerate(seen):
#             local[flat.eq(class_id)] = column
#         if bool(local.lt(0).any()):
#             missing = sorted(
#                 set(int(value) for value in flat[local.lt(0)].cpu().tolist())
#             )
#             raise RuntimeError(
#                 f"labels contain classes outside seen_classes: {missing}"
#             )
#         return local.to(labels.device)

#     @staticmethod
#     def local_to_global_labels(
#         labels_local: Tensor,
#         seen_classes: Sequence[int],
#     ) -> Tensor:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         local = labels_local.long().flatten()
#         if local.numel() and (
#             int(local.min().item()) < 0
#             or int(local.max().item()) >= len(seen)
#         ):
#             raise RuntimeError("local labels are outside seen-class range")
#         mapping = torch.tensor(seen, device=local.device, dtype=torch.long)
#         return mapping.index_select(0, local)

#     def _static_contract_state(self) -> Dict[str, Any]:
#         return {
#             "classifier": type(self).__name__,
#             "factorization": "p(s|c)p(z|s,c)",
#             "d_model": self.d_model,
#             "temperature": self.temperature,
#             "expected_spectral_anchor_dim":
#                 self.expected_spectral_anchor_dim,
#             "expected_bank_schema_version":
#                 self.expected_bank_schema_version,
#             "uses_trainable_classifier_weights": False,
#             "uses_class_specific_bias": False,
#             "uses_task_specific_head": False,
#         }

#     def classifier_contract_digest(self) -> str:
#         payload = (
#             self._static_contract_state(),
#             self._bound_bank_contract_digest,
#         )
#         return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

#     def get_extra_state(self) -> Dict[str, Any]:
#         return {
#             "bound_bank_contract_digest":
#                 self._bound_bank_contract_digest,
#             "require_bound_contract": self.require_bound_contract,
#             "static_contract": self._static_contract_state(),
#         }

#     def set_extra_state(self, state: Any) -> None:
#         if not isinstance(state, Mapping):
#             self._bound_bank_contract_digest = None
#             return
#         stored = state.get("static_contract")
#         if isinstance(stored, Mapping):
#             current = self._static_contract_state()
#             mismatches: list[str] = []
#             for name, expected in stored.items():
#                 if name not in current:
#                     continue
#                 actual = current[name]
#                 if isinstance(expected, float) or isinstance(actual, float):
#                     equal = (
#                         expected is actual
#                         if expected is None or actual is None
#                         else abs(float(expected) - float(actual)) <= 1e-12
#                     )
#                 else:
#                     equal = expected == actual
#                 if not equal:
#                     mismatches.append(
#                         f"{name}: checkpoint={expected!r}, current={actual!r}"
#                     )
#             if mismatches:
#                 raise RuntimeError(
#                     "classifier contract mismatch: " + "; ".join(mismatches)
#                 )

#         digest = state.get("bound_bank_contract_digest")
#         self._bound_bank_contract_digest = (
#             None if digest is None else str(digest)
#         )
#         self.require_bound_contract = bool(
#             state.get(
#                 "require_bound_contract",
#                 self.require_bound_contract,
#             )
#         )

#     @property
#     def bound_bank_contract_digest(self) -> Optional[str]:
#         return self._bound_bank_contract_digest

#     def _require_bank_api(self, geometry_bank: Any) -> None:
#         if geometry_bank is None:
#             raise ValueError("geometry_bank is required")
#         required_methods = (
#             "joint_energy_matrix",
#             "get_bank",
#             "valid_mask",
#             "assert_valid",
#             "contract_digest",
#         )
#         missing = [
#             name
#             for name in required_methods
#             if not callable(getattr(geometry_bank, name, None))
#         ]
#         if missing:
#             raise TypeError(
#                 "geometry_bank does not implement the "
#                 "spectral-conditioned geometry API; "
#                 f"missing methods {missing}"
#             )

#     def bind_geometry_bank_contract(
#         self,
#         geometry_bank: Any,
#         *,
#         require_frozen_anchor: bool = True,
#         require_committed_rows: bool = True,
#         enforce_after_binding: bool = True,
#         overwrite: bool = False,
#     ) -> str:
#         """Bind to the immutable anchor/energy contract, not to class rows."""
#         self._require_bank_api(geometry_bank)
#         if not bool(getattr(geometry_bank, "anchor_ready").item()):
#             raise RuntimeError("cannot bind an uninitialized spectral anchor")
#         if require_frozen_anchor and not bool(
#             getattr(geometry_bank, "anchor_frozen").item()
#         ):
#             raise RuntimeError("cannot bind an unfrozen spectral anchor")

#         if require_committed_rows and not bool(
#             geometry_bank.valid_mask().any().item()
#         ):
#             raise RuntimeError(
#                 "cannot bind before at least one valid class row is committed"
#             )
#         self._validate_static_bank_contract(
#             geometry_bank,
#             device=getattr(geometry_bank, "device"),
#             enforce_binding=False,
#         )
#         digest = str(geometry_bank.contract_digest()).strip().lower()
#         if len(digest) != 64 or any(
#             char not in "0123456789abcdef" for char in digest
#         ):
#             raise RuntimeError("GeometryBank contract digest is invalid")
#         if (
#             self._bound_bank_contract_digest is not None
#             and self._bound_bank_contract_digest != digest
#             and not overwrite
#         ):
#             raise RuntimeError(
#                 "classifier is already bound to another bank contract"
#             )
#         self._bound_bank_contract_digest = digest
#         if enforce_after_binding:
#             self.require_bound_contract = True
#         return digest

#     def classifier_contract(self) -> Dict[str, Any]:
#         return {
#             **self._static_contract_state(),
#             "logit_rule": "logits=-energy/temperature",
#             "energy_owner": "SpectralConditionedGeometryBank",
#             "supports_temporary_rows": True,
#             "supports_feature_ablation": True,
#             "supports_spectral_ablation": True,
#             "uses_teacher": False,
#             "uses_prototypes_as_classifier_weights": False,
#             "uses_reliability_logit_penalty": False,
#             "bound_bank_contract_digest":
#                 self._bound_bank_contract_digest,
#         }

#     def _validate_static_bank_contract(
#         self,
#         geometry_bank: Any,
#         *,
#         device: torch.device,
#         enforce_binding: bool = True,
#     ) -> Dict[str, Any]:
#         self._require_bank_api(geometry_bank)
#         schema_version = int(
#             getattr(geometry_bank, "SCHEMA_VERSION", -1)
#         )
#         if (
#             self.expected_bank_schema_version is not None
#             and schema_version != self.expected_bank_schema_version
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank schema versions differ: "
#                 f"{self.expected_bank_schema_version} vs {schema_version}"
#             )
#         if int(getattr(geometry_bank, "d_model", -1)) != self.d_model:
#             raise RuntimeError(
#                 "classifier and GeometryBank feature dimensions differ"
#             )
#         if torch.device(getattr(geometry_bank, "device")) != torch.device(
#             device
#         ):
#             raise RuntimeError(
#                 "features and GeometryBank must share one device"
#             )

#         anchor_dim = int(
#             getattr(geometry_bank, "spectral_anchor_dim", -1)
#         )
#         if (
#             self.expected_spectral_anchor_dim is not None
#             and anchor_dim != self.expected_spectral_anchor_dim
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank spectral-anchor dimensions "
#                 f"differ: {self.expected_spectral_anchor_dim} vs {anchor_dim}"
#             )
#         if not bool(getattr(geometry_bank, "anchor_ready").item()):
#             raise RuntimeError("GeometryBank spectral anchor is absent")

#         digest = str(geometry_bank.contract_digest()).strip().lower()
#         if len(digest) != 64:
#             raise RuntimeError("GeometryBank contract digest is invalid")
#         if (
#             enforce_binding
#             and self.require_bound_contract
#             and self._bound_bank_contract_digest is None
#         ):
#             raise RuntimeError(
#                 "classifier requires a bound GeometryBank contract"
#             )
#         if (
#             self._bound_bank_contract_digest is not None
#             and digest != self._bound_bank_contract_digest
#         ):
#             raise RuntimeError(
#                 "GeometryBank anchor/energy contract changed after binding"
#             )
#         if (
#             self._bound_bank_contract_digest is not None
#             and not bool(getattr(geometry_bank, "anchor_frozen").item())
#         ):
#             raise RuntimeError(
#                 "a bound classifier requires a frozen spectral anchor"
#             )

#         return {
#             "schema_version": schema_version,
#             "spectral_anchor_dim": anchor_dim,
#             "contract_digest": digest,
#             "spectral_weight": float(
#                 getattr(geometry_bank, "spectral_weight")
#             ),
#             "feature_weight": float(
#                 getattr(geometry_bank, "feature_weight")
#             ),
#         }

#     def _validate_bank_rows(
#         self,
#         geometry_bank: Any,
#         *,
#         seen_classes: Sequence[int],
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]],
#         device: torch.device,
#     ) -> Dict[str, Any]:
#         contract = self._validate_static_bank_contract(
#             geometry_bank,
#             device=device,
#         )
#         seen = list(seen_classes)
#         temporary_ids = (
#             set()
#             if temporary_rows is None
#             else set(int(class_id) for class_id in temporary_rows)
#         )
#         unknown = sorted(temporary_ids - set(seen))
#         if unknown:
#             raise RuntimeError(
#                 f"temporary_rows contain classes outside seen_classes: {unknown}"
#             )
#         committed_ids = [
#             class_id for class_id in seen if class_id not in temporary_ids
#         ]
#         if committed_ids:
#             report = geometry_bank.assert_valid(
#                 committed_ids,
#                 strict=False,
#             )
#             if not bool(report.get("ok", False)):
#                 raise RuntimeError(
#                     "invalid committed GeometryBank rows: "
#                     + "; ".join(report.get("errors", []))
#                 )
#         return contract

#     @staticmethod
#     def _infer_seen(
#         geometry_bank: Any,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#     ) -> list[int]:
#         valid = geometry_bank.valid_mask()
#         committed = torch.nonzero(
#             valid.detach().cpu().bool(),
#             as_tuple=False,
#         ).flatten().tolist()
#         temporary = (
#             []
#             if temporary_rows is None
#             else [int(class_id) for class_id in temporary_rows]
#         )
#         return sorted(set(committed) | set(temporary))

#     def compute_energy(
#         self,
#         features: Tensor,
#         *,
#         seen_classes: Sequence[int],
#         geometry_bank: Any,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         mode: str = JOINT_MODE,
#         return_parts: bool = True,
#     ) -> Tuple[Tensor, Dict[str, Any]]:
#         features = _validate_features(features, self.d_model)
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         selected_mode = self.normalize_mode(mode)
#         contract = self._validate_bank_rows(
#             geometry_bank,
#             seen_classes=seen,
#             temporary_rows=temporary_rows,
#             device=features.device,
#         )

#         result = geometry_bank.joint_energy_matrix(
#             features,
#             seen,
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#             rows=temporary_rows,
#             return_parts=True,
#         )
#         required = {
#             "energy",
#             "spectral_energy",
#             "feature_energy",
#             "conditional_feature_mean",
#         }
#         missing = required.difference(result)
#         if missing:
#             raise RuntimeError(
#                 f"GeometryBank joint energy is missing {sorted(missing)}"
#             )

#         weighted_spectral = (
#             float(contract["spectral_weight"]) * result["spectral_energy"]
#         )
#         weighted_feature = (
#             float(contract["feature_weight"]) * result["feature_energy"]
#         )
#         reconstructed_joint = weighted_spectral + weighted_feature
#         if not torch.allclose(
#             result["energy"],
#             reconstructed_joint,
#             atol=1e-5,
#             rtol=1e-5,
#         ):
#             raise RuntimeError(
#                 "GeometryBank joint energy violates its declared weights"
#             )

#         if selected_mode == self.JOINT_MODE:
#             energy = result["energy"]
#         elif selected_mode == self.CONDITIONAL_FEATURE_ABLATION:
#             energy = weighted_feature
#         else:
#             energy = weighted_spectral

#         expected_shape = (features.size(0), len(seen))
#         if tuple(energy.shape) != expected_shape:
#             raise RuntimeError(
#                 f"energy shape {tuple(energy.shape)} != {expected_shape}"
#             )
#         if not torch.isfinite(energy).all():
#             raise RuntimeError("geometry energy contains NaN/Inf")

#         parts: Dict[str, Any] = {
#             **result,
#             "weighted_spectral_energy": weighted_spectral,
#             "weighted_feature_energy": weighted_feature,
#             "selected_energy": energy,
#             "mode": selected_mode,
#             "global_class_ids": torch.tensor(
#                 seen, device=features.device, dtype=torch.long
#             ),
#             "joint_factorization": "p(s|c)p(z|s,c)",
#             "contract_digest": contract["contract_digest"],
#             "uses_temporary_rows": temporary_rows is not None,
#             "bank_mutated": False,
#         }
#         return energy, parts if return_parts else {}

#     def _energy_to_logits(self, energy: Tensor) -> Tensor:
#         logits = -energy / self._temperature.to(
#             device=energy.device,
#             dtype=energy.dtype,
#         )
#         if not torch.isfinite(logits).all():
#             raise RuntimeError("geometry logits contain NaN/Inf")
#         return logits

#     def compute_logits(
#         self,
#         features: Tensor,
#         *,
#         seen_classes: Sequence[int],
#         geometry_bank: Any,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         mode: str = JOINT_MODE,
#         return_parts: bool = False,
#     ) -> Union[Tensor, Dict[str, Any]]:
#         energy, parts = self.compute_energy(
#             features,
#             seen_classes=seen_classes,
#             geometry_bank=geometry_bank,
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#             temporary_rows=temporary_rows,
#             mode=mode,
#             return_parts=True,
#         )
#         logits = self._energy_to_logits(energy)
#         self.assert_logits_valid(
#             logits,
#             seen_classes=seen_classes,
#         )
#         if not return_parts:
#             return logits
#         return {
#             "logits": logits,
#             **parts,
#         }

#     def forward_from_outputs(
#         self,
#         model_output: Mapping[str, Any],
#         *,
#         geometry_bank: Any,
#         seen_classes: Optional[Iterable[int]] = None,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         mode: str = JOINT_MODE,
#         **kwargs: Any,
#     ) -> Union[Tensor, Dict[str, Any]]:
#         if "geometry_features" not in model_output:
#             raise KeyError("model_output lacks geometry_features")
#         raw_spectra = model_output.get("raw_center_spectra")
#         spectral_anchors = model_output.get("spectral_anchors")
#         if (raw_spectra is None) == (spectral_anchors is None):
#             raise KeyError(
#                 "model_output must contain exactly one of "
#                 "raw_center_spectra or spectral_anchors"
#             )
#         return self.forward(
#             model_output["geometry_features"],
#             seen_classes=seen_classes,
#             geometry_bank=geometry_bank,
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#             temporary_rows=temporary_rows,
#             mode=mode,
#             **kwargs,
#         )

#     @staticmethod
#     def assert_logits_valid(
#         logits: Tensor,
#         *,
#         seen_classes: Sequence[int],
#         targets_local: Optional[Tensor] = None,
#     ) -> None:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         if not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError("logits must be [N,C]")
#         if logits.size(1) != len(seen):
#             raise RuntimeError(
#                 "logit width does not match seen_classes"
#             )
#         if not torch.isfinite(logits).all():
#             raise RuntimeError("logits contain NaN/Inf")
#         if targets_local is not None:
#             targets = targets_local.to(
#                 logits.device
#             ).long().flatten()
#             if targets.numel() != logits.size(0):
#                 raise RuntimeError("target/logit batch mismatch")
#             if targets.numel() and (
#                 int(targets.min().item()) < 0
#                 or int(targets.max().item()) >= logits.size(1)
#             ):
#                 raise RuntimeError(
#                     "targets_local must use seen-local indices"
#                 )

#     @staticmethod
#     def energy_margin_statistics(
#         energy: Tensor,
#         targets_local: Tensor,
#     ) -> Dict[str, Tensor]:
#         if energy.dim() != 2:
#             raise ValueError("energy must be [N,C]")
#         labels = targets_local.to(
#             energy.device
#         ).long().flatten()
#         if labels.numel() != energy.size(0):
#             raise ValueError("energy/target batch mismatch")
#         true_energy = energy.gather(
#             1, labels[:, None]
#         ).squeeze(1)
#         rival_energy = energy.clone()
#         rival_energy.scatter_(
#             1, labels[:, None], float("inf")
#         )
#         nearest_rival_energy, nearest_rival = rival_energy.min(dim=1)
#         margin = nearest_rival_energy - true_energy
#         return {
#             "true_energy": true_energy,
#             "nearest_rival_energy": nearest_rival_energy,
#             "nearest_rival_local": nearest_rival,
#             "margin": margin,
#             "mean_margin": margin.mean(),
#             "minimum_margin": margin.min(),
#             "q01_margin": torch.quantile(margin, 0.01),
#             "q05_margin": torch.quantile(margin, 0.05),
#             "violation_rate": margin.lt(0.0).float().mean(),
#             "accuracy": energy.argmin(dim=1).eq(labels).float().mean(),
#         }

#     @staticmethod
#     def _phase_masks(
#         seen_classes: Sequence[int],
#         *,
#         old_classes: Optional[Iterable[int]],
#         new_classes: Optional[Iterable[int]],
#         device: torch.device,
#     ) -> Tuple[Tensor, Tensor]:
#         seen = list(seen_classes)
#         old_set = set(int(value) for value in (old_classes or []))
#         new_set = set(int(value) for value in (new_classes or []))
#         if old_classes is None and new_classes is None:
#             return (
#                 torch.zeros(len(seen), device=device, dtype=torch.bool),
#                 torch.zeros(len(seen), device=device, dtype=torch.bool),
#             )
#         if old_classes is None:
#             old_set = set(seen) - new_set
#         if new_classes is None:
#             new_set = set(seen) - old_set
#         if old_set & new_set or old_set | new_set != set(seen):
#             raise RuntimeError(
#                 "old/new classes must partition seen_classes"
#             )
#         return (
#             torch.tensor(
#                 [class_id in old_set for class_id in seen],
#                 device=device,
#                 dtype=torch.bool,
#             ),
#             torch.tensor(
#                 [class_id in new_set for class_id in seen],
#                 device=device,
#                 dtype=torch.bool,
#             ),
#         )

#     @torch.no_grad()
#     def classifier_diagnostics(
#         self,
#         *,
#         logits: Tensor,
#         joint_energy: Tensor,
#         spectral_energy: Tensor,
#         feature_energy: Tensor,
#         seen_classes: Sequence[int],
#         targets_local: Optional[Tensor] = None,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#     ) -> Dict[str, Any]:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         self.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets_local=targets_local,
#         )
#         predicted_local = logits.argmax(dim=1)
#         predicted_global = self.local_to_global_labels(
#             predicted_local,
#             seen,
#         )
#         counts = torch.bincount(
#             predicted_local,
#             minlength=len(seen),
#         ).cpu()
#         output: Dict[str, Any] = {
#             "joint_factorization": "p(s|c)p(z|s,c)",
#             "prediction_distribution": {
#                 seen[index]: int(counts[index].item())
#                 for index in range(len(seen))
#             },
#             "predicted_global": predicted_global,
#             "uses_trainable_classifier_weights": False,
#             "uses_class_specific_bias": False,
#         }
#         if targets_local is None:
#             return output

#         targets = targets_local.to(
#             logits.device
#         ).long().flatten()
#         joint_stats = self.energy_margin_statistics(
#             joint_energy,
#             targets,
#         )
#         feature_prediction = feature_energy.argmin(dim=1)
#         spectral_prediction = spectral_energy.argmin(dim=1)
#         joint_prediction = joint_energy.argmin(dim=1)
#         feature_correct = feature_prediction.eq(targets)
#         spectral_correct = spectral_prediction.eq(targets)
#         joint_correct = joint_prediction.eq(targets)

#         output.update(
#             {
#                 "accuracy": float(joint_correct.float().mean().item()),
#                 "conditional_feature_accuracy": float(
#                     feature_correct.float().mean().item()
#                 ),
#                 "spectral_only_accuracy": float(
#                     spectral_correct.float().mean().item()
#                 ),
#                 "spectral_help_rate": float(
#                     ((~feature_correct) & joint_correct)
#                     .float()
#                     .mean()
#                     .item()
#                 ),
#                 "spectral_harm_rate": float(
#                     (feature_correct & (~joint_correct))
#                     .float()
#                     .mean()
#                     .item()
#                 ),
#                 "joint_feature_disagreement_rate": float(
#                     joint_prediction.ne(feature_prediction)
#                     .float()
#                     .mean()
#                     .item()
#                 ),
#                 "joint_mean_margin": float(
#                     joint_stats["mean_margin"].item()
#                 ),
#                 "joint_minimum_margin": float(
#                     joint_stats["minimum_margin"].item()
#                 ),
#                 "joint_q01_margin": float(
#                     joint_stats["q01_margin"].item()
#                 ),
#                 "joint_q05_margin": float(
#                     joint_stats["q05_margin"].item()
#                 ),
#                 "joint_violation_rate": float(
#                     joint_stats["violation_rate"].item()
#                 ),
#             }
#         )

#         nearest_rival = joint_stats["nearest_rival_local"]
#         true_spectral = spectral_energy.gather(
#             1, targets[:, None]
#         ).squeeze(1)
#         rival_spectral = spectral_energy.gather(
#             1, nearest_rival[:, None]
#         ).squeeze(1)
#         spectral_gap_on_joint_rival = rival_spectral - true_spectral
#         output.update(
#             {
#                 "spectral_gap_on_joint_rival_mean": float(
#                     spectral_gap_on_joint_rival.mean().item()
#                 ),
#                 "spectral_gap_on_joint_rival_q05": float(
#                     torch.quantile(
#                         spectral_gap_on_joint_rival, 0.05
#                     ).item()
#                 ),
#                 "spectral_support_rate": float(
#                     spectral_gap_on_joint_rival.gt(0.0)
#                     .float()
#                     .mean()
#                     .item()
#                 ),
#             }
#         )

#         old_mask, new_mask = self._phase_masks(
#             seen,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             device=logits.device,
#         )
#         if bool(old_mask.any()) or bool(new_mask.any()):
#             old_samples = old_mask.index_select(0, targets)
#             new_samples = new_mask.index_select(0, targets)
#             output["old_accuracy"] = (
#                 float(joint_correct[old_samples].float().mean().item())
#                 if bool(old_samples.any())
#                 else 0.0
#             )
#             output["new_accuracy"] = (
#                 float(joint_correct[new_samples].float().mean().item())
#                 if bool(new_samples.any())
#                 else 0.0
#             )
#             output["old_to_new_invasion"] = (
#                 float(
#                     new_mask.index_select(
#                         0, joint_prediction[old_samples]
#                     )
#                     .float()
#                     .mean()
#                     .item()
#                 )
#                 if bool(old_samples.any()) and bool(new_mask.any())
#                 else 0.0
#             )
#             output["new_to_old_invasion"] = (
#                 float(
#                     old_mask.index_select(
#                         0, joint_prediction[new_samples]
#                     )
#                     .float()
#                     .mean()
#                     .item()
#                 )
#                 if bool(new_samples.any()) and bool(old_mask.any())
#                 else 0.0
#             )
#         return output

#     def forward(
#         self,
#         features: Tensor,
#         *,
#         geometry_bank: Any,
#         seen_classes: Optional[Iterable[int]] = None,
#         raw_spectra: Optional[Tensor] = None,
#         spectral_anchors: Optional[Tensor] = None,
#         temporary_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         mode: str = JOINT_MODE,
#         targets: Optional[Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#     ) -> Union[Tensor, Dict[str, Any]]:
#         selected_mode = self.normalize_mode(mode)
#         seen = (
#             self._infer_seen(
#                 geometry_bank,
#                 temporary_rows,
#             )
#             if seen_classes is None
#             else _unique_ids(
#                 seen_classes,
#                 name="seen_classes",
#             )
#         )
#         if not seen:
#             raise RuntimeError("no seen classes are available")
#         self._last_seen_classes = seen

#         scored = self.compute_logits(
#             features,
#             seen_classes=seen,
#             geometry_bank=geometry_bank,
#             raw_spectra=raw_spectra,
#             spectral_anchors=spectral_anchors,
#             temporary_rows=temporary_rows,
#             mode=selected_mode,
#             return_parts=True,
#         )
#         assert isinstance(scored, dict)
#         logits = scored["logits"]
#         selected_energy = scored["selected_energy"]
#         joint_energy = scored["energy"]

#         local_targets: Optional[Tensor] = None
#         if targets is not None:
#             local_targets = (
#                 self.global_to_local_labels(
#                     targets,
#                     seen,
#                 )
#                 if targets_are_global
#                 else targets.to(features.device).long().flatten()
#             )
#         self.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets_local=local_targets,
#         )

#         if not (
#             return_energy
#             or return_parts
#             or return_diagnostics
#         ):
#             return logits

#         output: Dict[str, Any] = {
#             "logits": logits,
#             "energy": selected_energy,
#             "selected_energy": selected_energy,
#             "joint_energy": joint_energy,
#             "spectral_energy": scored["spectral_energy"],
#             "feature_energy": scored["feature_energy"],
#             "weighted_spectral_energy":
#                 scored["weighted_spectral_energy"],
#             "weighted_feature_energy":
#                 scored["weighted_feature_energy"],
#             "seen_classes": torch.tensor(
#                 seen,
#                 device=features.device,
#                 dtype=torch.long,
#             ),
#             "mode": selected_mode,
#             "joint_factorization": "p(s|c)p(z|s,c)",
#         }
#         if return_parts:
#             output.update(scored)
#             output["energy"] = selected_energy
#             output["selected_energy"] = selected_energy
#             output["joint_energy"] = joint_energy
#         if return_diagnostics:
#             diagnostics = self.classifier_diagnostics(
#                 logits=self._energy_to_logits(joint_energy),
#                 joint_energy=joint_energy,
#                 spectral_energy=scored["weighted_spectral_energy"],
#                 feature_energy=scored["weighted_feature_energy"],
#                 seen_classes=seen,
#                 targets_local=local_targets,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#             )
#             diagnostics["selected_mode"] = selected_mode
#             if local_targets is not None:
#                 diagnostics["selected_mode_accuracy"] = float(
#                     logits.argmax(dim=1)
#                     .eq(local_targets)
#                     .float()
#                     .mean()
#                     .item()
#                 )
#             output["diagnostics"] = diagnostics
#         return output


# GeometryEnergyClassifier = SpectralConditionedEnergyClassifier










# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# import hashlib

# import torch
# import torch.nn as nn


# _INVALID_LOGIT = -1e9


# def _unique_ids(
#     values: Iterable[int],
#     *,
#     name: str,
#     allow_empty: bool = False,
# ) -> List[int]:
#     ids: List[int] = []
#     observed = set()
#     for value in values:
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError(f"{name} contains negative class ID {class_id}")
#         if class_id in observed:
#             raise ValueError(f"{name} contains duplicate class ID {class_id}")
#         observed.add(class_id)
#         ids.append(class_id)
#     if not ids and not allow_empty:
#         raise ValueError(f"{name} is empty")
#     return ids


# def _validate_features(features: torch.Tensor, d_model: int) -> torch.Tensor:
#     if not torch.is_tensor(features):
#         raise TypeError("features must be a tensor")
#     if features.dim() != 2 or features.size(1) != int(d_model):
#         raise ValueError(
#             f"features must be [B,{int(d_model)}], got {tuple(features.shape)}"
#         )
#     if not torch.isfinite(features).all():
#         raise RuntimeError("features contain NaN or Inf")
#     return features


# def _validate_responses(
#     responses: torch.Tensor,
#     *,
#     batch_size: int,
#     num_interventions: int,
#     d_model: int,
# ) -> torch.Tensor:
#     if not torch.is_tensor(responses):
#         raise TypeError("spectral_responses must be a tensor")
#     expected = (int(batch_size), int(num_interventions), int(d_model))
#     if tuple(responses.shape) != expected:
#         raise ValueError(
#             f"spectral_responses must be {expected}, got {tuple(responses.shape)}"
#         )
#     if not torch.isfinite(responses).all():
#         raise RuntimeError("spectral_responses contain NaN or Inf")
#     return responses


# class GeometryEnergyClassifier(nn.Module):
#     """Parameter-free PC-STGB conditional joint-energy classifier.

#     The persistent class parameters live exclusively in GeometryBank. The
#     classifier applies one deployed rule:

#         E_c(x) = E_c^occ(z) + beta_T E_c^tan(g_1,...,g_K | z)
#         logits_c = -tau E_c(x)

#     ``z`` is the canonical feature and ``g_k`` is the central finite-difference
#     feature response to deterministic ordered-band spectral intervention ``k``.

#     Responsibilities
#     ----------------
#     GeometryBank:
#         row validation, row normalization, occupancy energy, response energy,
#         candidate row stacking, atomic memory.
#     Classifier:
#         input contract, class-column ordering, logits, label conversion,
#         phase diagnostics.

#     The tangent term is conditioned on occupancy coordinates through the
#     GeometryBank coupling row. No raw-spectrum classifier, calibration layer,
#     reliability logit penalty, or silent feature-only fallback is used.
#     """

#     METHOD_MODE = "pc_stgb"
#     LEGACY_METHOD_MODE = "pc_sirg"
#     FEATURE_ONLY_ABLATION_MODE = "feature_only_ablation"

#     def __init__(
#         self,
#         initial_classes: int = 0,
#         d_model: int = 128,
#         logit_scale: float = 8.0,
#         variance_floor: float = 1e-4,
#         response_variance_floor: float = 1e-4,
#         response_weight: Optional[float] = None,
#         energy_logdet_weight: float = 1.0,
#         response_logdet_weight: float = 1.0,
#         expected_num_interventions: Optional[int] = 2,
#         expected_intervention_definition_version: Optional[int] = 1,
#         expected_bank_schema_version: Optional[int] = 5,
#         expected_contract_digest: Optional[str] = None,
#         require_bound_contract: bool = False,
#         normalize_energy_by_dim: bool = True,
#         energy_normalize_by_dim: Optional[bool] = None,
#         logit_clip: float = 0.0,
#         invalid_logit: float = _INVALID_LOGIT,
#         invalid_class_energy: float = 1e6,
#     ) -> None:
#         super().__init__()
#         self.num_classes = int(max(0, initial_classes))
#         self.d_model = int(d_model)
#         self.logit_scale = float(logit_scale)
#         self.variance_floor = float(max(variance_floor, 1e-12))
#         self.response_variance_floor = float(
#             max(response_variance_floor, 1e-12)
#         )
#         self.response_weight = (
#             None if response_weight is None else float(response_weight)
#         )
#         self.energy_logdet_weight = float(energy_logdet_weight)
#         self.response_logdet_weight = float(response_logdet_weight)
#         self.expected_num_interventions = (
#             None
#             if expected_num_interventions is None
#             else int(expected_num_interventions)
#         )
#         self.expected_intervention_definition_version = (
#             None
#             if expected_intervention_definition_version is None
#             else int(expected_intervention_definition_version)
#         )
#         self.expected_bank_schema_version = (
#             None
#             if expected_bank_schema_version is None
#             else int(expected_bank_schema_version)
#         )
#         self.require_bound_contract = bool(require_bound_contract)
#         self._bound_contract_digest: Optional[str] = None
#         if expected_contract_digest is not None:
#             token = str(expected_contract_digest).strip().lower()
#             if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
#                 raise ValueError("expected_contract_digest must be a SHA-256 hex string")
#             self._bound_contract_digest = token
#         self.invalid_logit = float(invalid_logit)
#         self.invalid_class_energy = float(max(invalid_class_energy, 1.0))

#         if self.d_model <= 0:
#             raise ValueError("d_model must be positive")
#         if self.logit_scale <= 0.0:
#             raise ValueError("logit_scale must be positive")
#         if self.energy_logdet_weight <= 0.0:
#             raise ValueError("energy_logdet_weight must be positive")
#         if self.response_logdet_weight <= 0.0:
#             raise ValueError("response_logdet_weight must be positive")
#         if self.response_weight is not None and self.response_weight < 0.0:
#             raise ValueError("response_weight must be non-negative")
#         if (
#             self.expected_num_interventions is not None
#             and self.expected_num_interventions <= 0
#         ):
#             raise ValueError("expected_num_interventions must be positive")
#         if (
#             self.expected_intervention_definition_version is not None
#             and self.expected_intervention_definition_version <= 0
#         ):
#             raise ValueError(
#                 "expected_intervention_definition_version must be positive"
#             )
#         if (
#             self.expected_bank_schema_version is not None
#             and self.expected_bank_schema_version <= 0
#         ):
#             raise ValueError("expected_bank_schema_version must be positive")

#         if energy_normalize_by_dim is not None:
#             normalize_energy_by_dim = bool(energy_normalize_by_dim)
#         self.normalize_energy_by_dim = bool(normalize_energy_by_dim)
#         self.energy_normalize_by_dim = self.normalize_energy_by_dim
#         if not self.normalize_energy_by_dim:
#             raise ValueError(
#                 "occupancy and response energies must be dimension-normalized"
#             )
#         if abs(float(logit_clip)) > 1e-12:
#             raise ValueError(
#                 "logit clipping is forbidden because it changes energy ordering"
#             )

#         # Compatibility attributes used by the active model contract.
#         self.logdet_energy_weight = self.energy_logdet_weight
#         self.use_logdet_energy = True
#         self._last_seen_classes: List[int] = list(range(self.num_classes))
#         self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

#     # ------------------------------------------------------------------
#     # Contract and compatibility
#     # ------------------------------------------------------------------
#     @staticmethod
#     def normalize_mode(mode: str) -> str:
#         token = str(mode or "pc_stgb").strip().lower().replace("-", "_")
#         if token in {
#             "pc_stgb",
#             "pc_sirg",
#             "joint_geometry",
#             "joint_energy",
#             "conditional_joint_geometry",
#             "spectral_tangent_geometry",
#             "spectral_response_geometry",
#             "geometry",
#             "geometry_only",
#         }:
#             # Legacy PC-STGB names resolve to the conditional PC-STGB rule.
#             return GeometryEnergyClassifier.METHOD_MODE
#         if token in {
#             "feature_only_ablation",
#             "occupancy_only_ablation",
#             "low_rank_geometry_ablation",
#         }:
#             return GeometryEnergyClassifier.FEATURE_ONLY_ABLATION_MODE
#         raise ValueError(
#             f"unsupported classifier mode {mode!r}; "
#             "use pc_stgb or feature_only_ablation"
#         )

#     def expand(self, num_new_classes: int, phase: int = 0) -> None:
#         del phase
#         self.num_classes += int(max(0, num_new_classes))

#     def expand_to_seen_classes(self, seen_classes: Iterable[int]) -> None:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         self._last_seen_classes = seen
#         self.num_classes = len(seen)

#     def freeze_all_adaptation(self) -> None:
#         return

#     @property
#     def bound_contract_digest(self) -> Optional[str]:
#         return self._bound_contract_digest

#     def bind_geometry_bank_contract(
#         self,
#         geometry_bank: Any,
#         *,
#         require_frozen_prior: bool = True,
#         overwrite: bool = False,
#     ) -> str:
#         """Bind inference to one frozen phase-invariant bank contract.

#         Call this once at the final base handoff. Incremental scoring then fails
#         immediately if the response prior or intervention contract changes.
#         """
#         if geometry_bank is None or not callable(
#             getattr(geometry_bank, "contract_digest", None)
#         ):
#             raise TypeError("geometry_bank must expose contract_digest()")
#         bank = geometry_bank.get_bank()
#         ready = bank.get("response_prior_ready")
#         frozen = bank.get("response_prior_frozen")
#         if not torch.is_tensor(ready) or not bool(ready.item()):
#             raise RuntimeError("cannot bind a bank with an uninitialized response prior")
#         if require_frozen_prior and (
#             not torch.is_tensor(frozen) or not bool(frozen.item())
#         ):
#             raise RuntimeError("cannot bind a bank whose response prior is not frozen")
#         digest = str(geometry_bank.contract_digest()).strip().lower()
#         if len(digest) != 64:
#             raise RuntimeError("GeometryBank contract_digest() is not SHA-256")
#         if (
#             self._bound_contract_digest is not None
#             and self._bound_contract_digest != digest
#             and not overwrite
#         ):
#             raise RuntimeError("classifier is already bound to a different bank contract")
#         self._bound_contract_digest = digest
#         return digest

#     def clear_bound_contract(self) -> None:
#         if self.require_bound_contract:
#             raise RuntimeError(
#                 "cannot clear the bank contract while require_bound_contract=True"
#             )
#         self._bound_contract_digest = None

#     def _static_contract_state(self) -> Dict[str, Any]:
#         return {
#             "d_model": self.d_model,
#             "logit_scale": self.logit_scale,
#             "variance_floor": self.variance_floor,
#             "response_variance_floor": self.response_variance_floor,
#             "response_weight": self.response_weight,
#             "energy_logdet_weight": self.energy_logdet_weight,
#             "response_logdet_weight": self.response_logdet_weight,
#             "expected_num_interventions": self.expected_num_interventions,
#             "expected_intervention_definition_version": (
#                 self.expected_intervention_definition_version
#             ),
#             "expected_bank_schema_version": self.expected_bank_schema_version,
#             "normalize_energy_by_dim": self.normalize_energy_by_dim,
#         }

#     def classifier_contract_digest(self) -> str:
#         payload = (
#             self._static_contract_state(),
#             self._bound_contract_digest,
#             self.METHOD_MODE,
#         )
#         return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

#     def get_extra_state(self) -> Dict[str, Any]:
#         return {
#             "bound_contract_digest": self._bound_contract_digest,
#             "classifier_contract_digest": self.classifier_contract_digest(),
#             "require_bound_contract": self.require_bound_contract,
#             "static_contract": self._static_contract_state(),
#         }

#     def set_extra_state(self, state: Any) -> None:
#         if not isinstance(state, Mapping):
#             self._bound_contract_digest = None
#             return
#         stored_contract = state.get("static_contract")
#         if isinstance(stored_contract, Mapping):
#             current = self._static_contract_state()
#             mismatches: List[str] = []
#             for name, expected in stored_contract.items():
#                 if name not in current:
#                     continue
#                 actual = current[name]
#                 if isinstance(expected, float) or isinstance(actual, float):
#                     if expected is None or actual is None:
#                         equal = expected is actual
#                     else:
#                         equal = abs(float(expected) - float(actual)) <= 1e-12
#                 else:
#                     equal = expected == actual
#                 if not equal:
#                     mismatches.append(f"{name}: checkpoint={expected!r}, current={actual!r}")
#             if mismatches:
#                 raise RuntimeError(
#                     "classifier static contract mismatch during checkpoint load: "
#                     + "; ".join(mismatches)
#                 )
#         digest = state.get("bound_contract_digest")
#         self._bound_contract_digest = None if digest is None else str(digest)
#         self.require_bound_contract = bool(
#             state.get("require_bound_contract", self.require_bound_contract)
#         )

#     def classifier_contract(self) -> Dict[str, Any]:
#         return {
#             "mode": self.METHOD_MODE,
#             "legacy_mode_alias": self.LEGACY_METHOD_MODE,
#             "feature_space": "canonical_euclidean_z",
#             "spectral_object": "central_finite_difference_feature_tangent",
#             "joint_factorization": "p(z|c)prod_k p(g_k|z,c)",
#             "energy": (
#                 "occupancy_energy+response_weight*"
#                 "conditional_tangent_energy"
#             ),
#             "logit_rule": "logits=-logit_scale*joint_energy",
#             "logit_scale": float(self.logit_scale),
#             "variance_floor": float(self.variance_floor),
#             "response_variance_floor": float(
#                 self.response_variance_floor
#             ),
#             "configured_response_weight": self.response_weight,
#             "energy_logdet_weight": float(self.energy_logdet_weight),
#             "response_logdet_weight": float(
#                 self.response_logdet_weight
#             ),
#             "expected_num_interventions":
#                 self.expected_num_interventions,
#             "expected_intervention_definition_version":
#                 self.expected_intervention_definition_version,
#             "expected_bank_schema_version": self.expected_bank_schema_version,
#             "bound_contract_digest": self._bound_contract_digest,
#             "require_bound_contract": self.require_bound_contract,
#             "uses_spectral_inference_score": True,
#             "uses_spectral_response_inference_score": True,
#             "uses_conditional_tangent_inference_score": True,
#             "uses_raw_spectral_gaussian": False,
#             "uses_coupling_inference_score": True,
#             "uses_independent_response_factorization": False,
#             "uses_reliability_logit_penalty": False,
#             "uses_calibration": False,
#             "candidate_scoring_mutates_bank": False,
#         }

#     # ------------------------------------------------------------------
#     # Labels
#     # ------------------------------------------------------------------
#     @staticmethod
#     def global_to_local_labels(
#         labels: torch.Tensor,
#         seen_classes: Sequence[int],
#     ) -> torch.Tensor:
#         if not torch.is_tensor(labels):
#             raise TypeError("labels must be a tensor")
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         flat = labels.long().flatten()
#         local = torch.full_like(flat, -1)
#         for column, class_id in enumerate(seen):
#             local[flat == class_id] = column
#         if bool((local < 0).any().item()):
#             missing = sorted(
#                 set(
#                     int(value)
#                     for value in flat[local < 0].detach().cpu().tolist()
#                 )
#             )
#             raise RuntimeError(
#                 f"labels contain classes outside seen_classes: {missing}"
#             )
#         return local.to(labels.device)

#     @staticmethod
#     def local_to_global_labels(
#         labels_local: torch.Tensor,
#         seen_classes: Sequence[int],
#     ) -> torch.Tensor:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         local = labels_local.long().flatten()
#         if local.numel() and (
#             int(local.min().item()) < 0
#             or int(local.max().item()) >= len(seen)
#         ):
#             raise RuntimeError(
#                 "local labels are outside the seen-class column range"
#             )
#         mapping = torch.tensor(
#             seen, device=local.device, dtype=torch.long
#         )
#         return mapping.index_select(0, local)

#     # ------------------------------------------------------------------
#     # GeometryBank validation
#     # ------------------------------------------------------------------
#     def _require_pc_stgb_bank(
#         self,
#         geometry_bank: Any,
#         *,
#         class_ids: Sequence[int],
#         device: torch.device,
#         allow_empty_classes: bool = False,
#     ) -> Dict[str, Any]:
#         if geometry_bank is None:
#             raise ValueError("geometry_bank is required")
#         required_methods = (
#             "get_bank",
#             "get_valid_mask",
#             "assert_bank_valid",
#             "geometry_energy_matrix",
#             "conditional_response_energy_matrix",
#             "joint_energy_matrix",
#             "candidate_joint_energy_matrix",
#             "contract_digest",
#         )
#         missing = [
#             name
#             for name in required_methods
#             if not callable(getattr(geometry_bank, name, None))
#         ]
#         if missing:
#             raise TypeError(
#                 "geometry_bank is not the conditional PC-STGB bank; "
#                 f"missing methods {missing}"
#             )

#         schema_version = int(getattr(geometry_bank, "SCHEMA_VERSION", -1))
#         if (
#             self.expected_bank_schema_version is not None
#             and schema_version != self.expected_bank_schema_version
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank schema versions differ: "
#                 f"{self.expected_bank_schema_version} vs {schema_version}"
#             )

#         bank_dim = int(getattr(geometry_bank, "d_model", -1))
#         if bank_dim != self.d_model:
#             raise RuntimeError(
#                 f"GeometryBank d_model={bank_dim} "
#                 f"!= classifier d_model={self.d_model}"
#             )
#         bank_device = torch.device(getattr(geometry_bank, "device"))
#         if bank_device != device:
#             raise RuntimeError(
#                 "features and GeometryBank must share one device"
#             )

#         bank_floor = float(getattr(geometry_bank, "variance_floor"))
#         response_floor = float(
#             getattr(geometry_bank, "response_variance_floor")
#         )
#         bank_logdet = float(
#             getattr(geometry_bank, "energy_logdet_weight")
#         )
#         bank_weight = float(getattr(geometry_bank, "response_weight"))
#         intervention_count = int(
#             getattr(geometry_bank, "num_interventions")
#         )
#         intervention_version = int(
#             getattr(geometry_bank, "intervention_definition_version")
#         )

#         if abs(bank_floor - self.variance_floor) > max(
#             1e-12, 1e-6 * bank_floor
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank occupancy variance floors differ"
#             )
#         if abs(response_floor - self.response_variance_floor) > max(
#             1e-12, 1e-6 * response_floor
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank response variance floors differ"
#             )
#         if abs(bank_logdet - self.energy_logdet_weight) > 1e-12:
#             raise RuntimeError(
#                 "classifier and GeometryBank occupancy log-volume weights differ"
#             )
#         if (
#             self.response_weight is not None
#             and abs(bank_weight - self.response_weight) > 1e-12
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank response weights differ: "
#                 f"{self.response_weight} vs {bank_weight}"
#             )
#         if (
#             self.expected_num_interventions is not None
#             and intervention_count != self.expected_num_interventions
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank intervention counts differ: "
#                 f"{self.expected_num_interventions} vs {intervention_count}"
#             )
#         if (
#             self.expected_intervention_definition_version is not None
#             and intervention_version
#             != self.expected_intervention_definition_version
#         ):
#             raise RuntimeError(
#                 "classifier and GeometryBank intervention definition "
#                 f"versions differ: "
#                 f"{self.expected_intervention_definition_version} "
#                 f"vs {intervention_version}"
#             )

#         bank_state = geometry_bank.get_bank()
#         prior_ready = bank_state.get("response_prior_ready")
#         prior_frozen = bank_state.get("response_prior_frozen")
#         coupling_ready = bank_state.get("response_coupling_ready")
#         if not torch.is_tensor(prior_ready) or not bool(prior_ready.item()):
#             raise RuntimeError("GeometryBank response prior is not initialized")
#         if not torch.is_tensor(prior_frozen):
#             raise RuntimeError("GeometryBank does not expose response_prior_frozen")
#         if not torch.is_tensor(coupling_ready):
#             raise RuntimeError("GeometryBank does not expose response_coupling_ready")

#         contract_digest = str(geometry_bank.contract_digest()).strip().lower()
#         if len(contract_digest) != 64:
#             raise RuntimeError("GeometryBank contract digest is invalid")
#         if self.require_bound_contract and self._bound_contract_digest is None:
#             raise RuntimeError(
#                 "classifier requires a bound GeometryBank contract; call "
#                 "bind_geometry_bank_contract() at final base handoff"
#             )
#         if (
#             self._bound_contract_digest is not None
#             and contract_digest != self._bound_contract_digest
#         ):
#             raise RuntimeError(
#                 "GeometryBank phase-invariant contract changed after binding"
#             )
#         if self._bound_contract_digest is not None and not bool(
#             prior_frozen.item()
#         ):
#             raise RuntimeError(
#                 "a bound PC-STGB classifier requires a frozen response prior"
#             )

#         ids = list(class_ids)
#         if ids or not allow_empty_classes:
#             report = geometry_bank.assert_bank_valid(ids, strict=False)
#             if not bool(report.get("ok", False)):
#                 raise RuntimeError(
#                     "invalid PC-STGB GeometryBank rows: "
#                     + "; ".join(report.get("errors", []))
#                 )
#             index = torch.tensor(ids, device=coupling_ready.device, dtype=torch.long)
#             if ids and not bool(coupling_ready.index_select(0, index).all().item()):
#                 raise RuntimeError(
#                     "seen classes contain rows without occupancy-tangent coupling"
#                 )

#         return {
#             "schema_version": schema_version,
#             "response_weight": bank_weight,
#             "num_interventions": intervention_count,
#             "intervention_definition_version": intervention_version,
#             "contract_digest": contract_digest,
#             "response_prior_frozen": bool(prior_frozen.item()),
#         }

#     # Legacy private name retained for active-model compatibility.
#     _require_pc_sirg_bank = _require_pc_stgb_bank

#     @staticmethod
#     def _infer_seen(geometry_bank: Any) -> List[int]:
#         bank = geometry_bank.get_bank()
#         valid = bank.get("valid_mask")
#         response_ready = bank.get("response_stats_ready")
#         coupling_ready = bank.get("response_coupling_ready")
#         if not torch.is_tensor(valid):
#             raise RuntimeError(
#                 "PC-STGB GeometryBank does not expose valid_mask"
#             )
#         mask = valid.detach().cpu().bool().flatten()
#         for name, value in (
#             ("response_stats_ready", response_ready),
#             ("response_coupling_ready", coupling_ready),
#         ):
#             if not torch.is_tensor(value):
#                 raise RuntimeError(
#                     f"PC-STGB GeometryBank does not expose {name}"
#                 )
#             candidate = value.detach().cpu().bool().flatten()
#             if candidate.shape != mask.shape:
#                 raise RuntimeError(f"{name} shape does not match valid_mask")
#             mask &= candidate
#         ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
#         return _unique_ids(ids, name="inferred_seen_classes")

#     # ------------------------------------------------------------------
#     # Scoring
#     # ------------------------------------------------------------------
#     def compute_joint_energy(
#         self,
#         features: torch.Tensor,
#         spectral_responses: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#         return_parts: bool = True,
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         features = _validate_features(features, self.d_model)
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         contract = self._require_pc_sirg_bank(
#             geometry_bank,
#             class_ids=seen,
#             device=features.device,
#         )
#         responses = _validate_responses(
#             spectral_responses,
#             batch_size=features.size(0),
#             num_interventions=int(contract["num_interventions"]),
#             d_model=self.d_model,
#         )
#         if responses.device != features.device:
#             raise RuntimeError(
#                 "features and spectral_responses must share one device"
#             )
#         responses = responses.to(dtype=features.dtype)

#         result = geometry_bank.joint_energy_matrix(
#             features,
#             responses,
#             seen,
#             response_weight=float(contract["response_weight"]),
#             normalize_by_dim=True,
#             feature_logdet_weight=self.energy_logdet_weight,
#             response_logdet_weight=self.response_logdet_weight,
#             return_parts=True,
#         )
#         required = {
#             "energy",
#             "occupancy_energy",
#             "response_energy",
#             "conditional_response_energy",
#             "occupancy",
#             "response",
#             "joint_factorization",
#         }
#         missing = required.difference(result)
#         if missing:
#             raise RuntimeError(
#                 "PC-STGB joint energy is missing fields: "
#                 f"{sorted(missing)}"
#             )

#         if result["joint_factorization"] != "p(z|c)prod_k p(g_k|z,c)":
#             raise RuntimeError(
#                 "GeometryBank returned the wrong joint factorization"
#             )
#         energy = result["energy"]
#         if energy.shape != (features.size(0), len(seen)):
#             raise RuntimeError(
#                 "PC-STGB joint energy has an invalid shape"
#             )
#         if not torch.isfinite(energy).all():
#             raise RuntimeError(
#                 "PC-STGB joint energy contains NaN or Inf"
#             )

#         bank_state = geometry_bank.get_bank()
#         index = torch.tensor(seen, device=features.device, dtype=torch.long)
#         coupling_reliability = bank_state[
#             "response_coupling_reliability"
#         ].index_select(0, index)
#         coupling_explained_variance = bank_state[
#             "response_coupling_explained_variance"
#         ].index_select(0, index)
#         parts: Dict[str, Any] = {
#             "energy": energy,
#             "joint_energy": energy,
#             "occupancy_energy": result["occupancy_energy"],
#             "response_energy": result["conditional_response_energy"],
#             "conditional_tangent_energy": result[
#                 "conditional_response_energy"
#             ],
#             "weighted_response_energy":
#                 float(contract["response_weight"])
#                 * result["conditional_response_energy"],
#             "weighted_conditional_tangent_energy":
#                 float(contract["response_weight"])
#                 * result["conditional_response_energy"],
#             "occupancy_parallel": result["occupancy"]["parallel"],
#             "occupancy_residual": result["occupancy"]["residual"],
#             "occupancy_quadratic": result["occupancy"]["quadratic"],
#             "occupancy_volume": result["occupancy"]["volume"],
#             "response_parallel": result["response"]["parallel"],
#             "response_residual": result["response"]["residual"],
#             "response_quadratic": result["response"]["quadratic"],
#             "response_volume": result["response"]["volume"],
#             "conditional_tangent_parallel": result["response"]["parallel"],
#             "conditional_tangent_residual": result["response"]["residual"],
#             "conditional_tangent_quadratic": result["response"]["quadratic"],
#             "conditional_tangent_volume": result["response"]["volume"],
#             "response_weight": features.new_tensor(
#                 float(contract["response_weight"])
#             ),
#             "num_interventions": torch.tensor(
#                 int(contract["num_interventions"]),
#                 device=features.device,
#                 dtype=torch.long,
#             ),
#             "intervention_definition_version": torch.tensor(
#                 int(contract["intervention_definition_version"]),
#                 device=features.device,
#                 dtype=torch.long,
#             ),
#             "bank_schema_version": torch.tensor(
#                 int(contract["schema_version"]),
#                 device=features.device,
#                 dtype=torch.long,
#             ),
#             "global_class_ids": index,
#             "coupling_reliability": coupling_reliability,
#             "coupling_explained_variance": coupling_explained_variance,
#             "joint_factorization": result["joint_factorization"],
#             "contract_digest": contract["contract_digest"],
#             "uses_coupling_inference_score": True,
#         }
#         return energy, parts if return_parts else {}

#     def compute_occupancy_energy_ablation(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#         return_parts: bool = True,
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """Explicit occupancy-only ablation; never an automatic fallback."""
#         features = _validate_features(features, self.d_model)
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         self._require_pc_sirg_bank(
#             geometry_bank,
#             class_ids=seen,
#             device=features.device,
#         )
#         result = geometry_bank.geometry_energy_matrix(
#             features,
#             seen,
#             normalize_by_dim=True,
#             logdet_weight=self.energy_logdet_weight,
#             invalid_class_energy=self.invalid_class_energy,
#             return_parts=True,
#         )
#         energy = result["energy"]
#         if not torch.isfinite(energy).all():
#             raise RuntimeError(
#                 "occupancy ablation energy contains NaN or Inf"
#             )
#         return energy, result if return_parts else {}

#     def _energy_to_logits(self, energy: torch.Tensor) -> torch.Tensor:
#         logits = -self.logit_scale * energy
#         if not torch.isfinite(logits).all():
#             raise RuntimeError("PC-STGB logits contain NaN or Inf")
#         return logits

#     def compute_joint_logits(
#         self,
#         features: torch.Tensor,
#         spectral_responses: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         geometry_bank: Any,
#         return_parts: bool = False,
#     ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         energy, parts = self.compute_joint_energy(
#             features,
#             spectral_responses,
#             seen_classes=seen,
#             geometry_bank=geometry_bank,
#             return_parts=True,
#         )
#         logits = self._energy_to_logits(energy)
#         self.assert_logits_valid(logits, seen_classes=seen)
#         if not return_parts:
#             return logits
#         return {"logits": logits, "raw_energy": energy, **parts}

#     def compute_candidate_joint_logits(
#         self,
#         features: torch.Tensor,
#         spectral_responses: torch.Tensor,
#         *,
#         geometry_bank: Any,
#         old_class_ids: Iterable[int],
#         candidate_rows: Mapping[int, Mapping[str, Any]],
#         candidate_class_ids: Optional[Iterable[int]] = None,
#         return_parts: bool = False,
#     ) -> Union[torch.Tensor, Dict[str, Any]]:
#         features = _validate_features(features, self.d_model)
#         old_ids = _unique_ids(
#             old_class_ids,
#             name="old_class_ids",
#             allow_empty=True,
#         )
#         candidate_ids = (
#             _unique_ids(
#                 candidate_class_ids,
#                 name="candidate_class_ids",
#             )
#             if candidate_class_ids is not None
#             else _unique_ids(
#                 (int(key) for key in candidate_rows),
#                 name="candidate_class_ids",
#             )
#         )
#         if set(old_ids) & set(candidate_ids):
#             raise RuntimeError(
#                 "old and candidate class sets overlap"
#             )
#         if set(int(key) for key in candidate_rows) != set(candidate_ids):
#             raise RuntimeError(
#                 "candidate_rows do not match candidate_class_ids"
#             )

#         contract = self._require_pc_sirg_bank(
#             geometry_bank,
#             class_ids=old_ids,
#             device=features.device,
#             allow_empty_classes=True,
#         )
#         responses = _validate_responses(
#             spectral_responses,
#             batch_size=features.size(0),
#             num_interventions=int(contract["num_interventions"]),
#             d_model=self.d_model,
#         )
#         if responses.device != features.device:
#             raise RuntimeError(
#                 "features and spectral_responses must share one device"
#             )
#         responses = responses.to(dtype=features.dtype)
#         seen = [*old_ids, *candidate_ids]

#         result = geometry_bank.candidate_joint_energy_matrix(
#             features,
#             responses,
#             seen,
#             candidate_rows,
#             response_weight=float(contract["response_weight"]),
#             normalize_by_dim=True,
#             feature_logdet_weight=self.energy_logdet_weight,
#             response_logdet_weight=self.response_logdet_weight,
#             return_parts=True,
#         )
#         required = {
#             "energy",
#             "occupancy_energy",
#             "response_energy",
#             "conditional_response_energy",
#             "occupancy",
#             "response",
#             "joint_factorization",
#         }
#         missing = required.difference(result)
#         if missing:
#             raise RuntimeError(
#                 "candidate PC-STGB energy is missing fields: "
#                 f"{sorted(missing)}"
#             )
#         if result["joint_factorization"] != "p(z|c)prod_k p(g_k|z,c)":
#             raise RuntimeError(
#                 "candidate GeometryBank returned the wrong joint factorization"
#             )
#         energy = result["energy"]
#         if energy.shape != (features.size(0), len(seen)):
#             raise RuntimeError(
#                 "candidate PC-STGB energy has an invalid shape"
#             )
#         if not torch.isfinite(energy).all():
#             raise RuntimeError(
#                 "candidate PC-STGB energy contains NaN or Inf"
#             )

#         logits = self._energy_to_logits(energy)
#         self.assert_logits_valid(logits, seen_classes=seen)
#         if not return_parts:
#             return logits
#         bank_state = geometry_bank.get_bank()
#         old_index = torch.tensor(old_ids, device=features.device, dtype=torch.long)
#         old_coupling_reliability = (
#             bank_state["response_coupling_reliability"].index_select(0, old_index)
#             if old_ids
#             else features.new_empty((0,))
#         )
#         return {
#             "logits": logits,
#             "energy": energy,
#             "raw_energy": energy,
#             "joint_energy": energy,
#             "occupancy_energy": result["occupancy_energy"],
#             "response_energy": result["conditional_response_energy"],
#             "conditional_tangent_energy": result[
#                 "conditional_response_energy"
#             ],
#             "weighted_response_energy":
#                 float(contract["response_weight"])
#                 * result["conditional_response_energy"],
#             "weighted_conditional_tangent_energy":
#                 float(contract["response_weight"])
#                 * result["conditional_response_energy"],
#             "occupancy": result["occupancy"],
#             "response": result["response"],
#             "response_weight": features.new_tensor(
#                 float(contract["response_weight"])
#             ),
#             "seen_classes": torch.tensor(
#                 seen, device=features.device, dtype=torch.long
#             ),
#             "old_class_ids": old_index,
#             "candidate_class_ids": torch.tensor(
#                 candidate_ids, device=features.device, dtype=torch.long
#             ),
#             "old_coupling_reliability": old_coupling_reliability,
#             "joint_factorization": result["joint_factorization"],
#             "contract_digest": contract["contract_digest"],
#             "bank_mutated": False,
#             "uses_spectral_inference_score": True,
#             "uses_coupling_inference_score": True,
#             "uses_independent_response_factorization": False,
#         }


#     # Existing trainer names are retained, but now require spectral responses.
#     compute_geometry_energy = compute_joint_energy
#     compute_geometry_logits = compute_joint_logits
#     compute_candidate_geometry_logits = compute_candidate_joint_logits

#     # ------------------------------------------------------------------
#     # Validation and diagnostics
#     # ------------------------------------------------------------------
#     def assert_logits_valid(
#         self,
#         logits: torch.Tensor,
#         *,
#         seen_classes: Sequence[int],
#         targets: Optional[torch.Tensor] = None,
#         **_: Any,
#     ) -> None:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         if not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError("logits must be [B,S]")
#         if logits.size(1) != len(seen):
#             raise RuntimeError(
#                 "logit width does not match seen_classes"
#             )
#         if not torch.isfinite(logits).all():
#             raise RuntimeError("PC-STGB logits contain NaN or Inf")
#         if targets is not None:
#             targets = targets.to(logits.device).long().flatten()
#             if targets.numel() != logits.size(0):
#                 raise RuntimeError("target/logit batch mismatch")
#             if targets.numel() and (
#                 int(targets.min().item()) < 0
#                 or int(targets.max().item()) >= logits.size(1)
#             ):
#                 raise RuntimeError("targets must be seen-local")

#     @torch.no_grad()
#     def energy_margin_statistics(
#         self,
#         energy: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         if energy.numel() == 0:
#             zero = self._zero * 0.0
#             return {
#                 "mean_margin": zero,
#                 "minimum_margin": zero,
#                 "q05_margin": zero,
#                 "violation_rate": zero,
#                 "accuracy": zero,
#             }
#         labels = labels.to(energy.device).long().flatten()
#         if energy.dim() != 2 or labels.numel() != energy.size(0):
#             raise ValueError("energy/labels have incompatible shapes")
#         true = energy.gather(1, labels[:, None]).squeeze(1)
#         rivals = energy.clone()
#         rivals.scatter_(1, labels[:, None], float("inf"))
#         margin = rivals.min(dim=1).values - true
#         return {
#             "mean_margin": margin.mean(),
#             "minimum_margin": margin.min(),
#             "q05_margin": torch.quantile(margin, 0.05),
#             "violation_rate": (margin <= 0.0).float().mean(),
#             "accuracy": energy.argmin(dim=1).eq(labels).float().mean(),
#         }

#     @staticmethod
#     def _phase_masks(
#         seen_classes: Sequence[int],
#         *,
#         old_classes: Optional[Iterable[int]],
#         new_classes: Optional[Iterable[int]],
#         old_class_count: Optional[int],
#         device: torch.device,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         seen = list(seen_classes)
#         if old_classes is None and new_classes is None:
#             count = min(max(int(old_class_count or 0), 0), len(seen))
#             old_set = set(seen[:count])
#             new_set = set(seen[count:])
#         else:
#             old_set = set(int(value) for value in (old_classes or []))
#             new_set = set(int(value) for value in (new_classes or []))
#             if old_classes is None:
#                 old_set = set(seen) - new_set
#             if new_classes is None:
#                 new_set = set(seen) - old_set
#         if old_set & new_set or old_set | new_set != set(seen):
#             raise RuntimeError(
#                 "old/new classes must form a disjoint partition "
#                 "of seen_classes"
#             )
#         old_mask = torch.tensor(
#             [class_id in old_set for class_id in seen],
#             device=device,
#             dtype=torch.bool,
#         )
#         new_mask = torch.tensor(
#             [class_id in new_set for class_id in seen],
#             device=device,
#             dtype=torch.bool,
#         )
#         return old_mask, new_mask

#     @torch.no_grad()
#     def classifier_diagnostics(
#         self,
#         logits: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         targets_local: Optional[torch.Tensor] = None,
#         joint_energy: Optional[torch.Tensor] = None,
#         occupancy_energy: Optional[torch.Tensor] = None,
#         response_energy: Optional[torch.Tensor] = None,
#         coupling_reliability: Optional[torch.Tensor] = None,
#         coupling_explained_variance: Optional[torch.Tensor] = None,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         seen = _unique_ids(seen_classes, name="seen_classes")
#         self.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets=targets_local,
#         )
#         predicted_local = logits.argmax(dim=1)
#         predicted_global = self.local_to_global_labels(
#             predicted_local, seen
#         )
#         counts = torch.bincount(
#             predicted_local, minlength=len(seen)
#         ).detach().cpu()
#         output: Dict[str, Any] = {
#             "seen_classes": seen,
#             "output_dim": len(seen),
#             "prediction_distribution": {
#                 seen[index]: int(counts[index].item())
#                 for index in range(len(seen))
#             },
#             "prediction_global":
#                 predicted_global.detach().cpu().tolist(),
#             "uses_calibration": False,
#             "uses_spectral_inference_score": True,
#             "uses_spectral_response_inference_score": True,
#             "uses_raw_spectral_gaussian": False,
#             "uses_coupling_inference_score": True,
#             "uses_independent_response_factorization": False,
#             "joint_factorization": "p(z|c)prod_k p(g_k|z,c)",
#             "uses_reliability_logit_penalty": False,
#             "uses_log_volume": True,
#         }
#         if coupling_reliability is not None and coupling_reliability.numel():
#             output["coupling_reliability_mean"] = float(
#                 coupling_reliability.float().mean().item()
#             )
#             output["coupling_reliability_min"] = float(
#                 coupling_reliability.float().min().item()
#             )
#         if (
#             coupling_explained_variance is not None
#             and coupling_explained_variance.numel()
#         ):
#             output["coupling_explained_variance_mean"] = float(
#                 coupling_explained_variance.float().mean().item()
#             )
#             output["coupling_explained_variance_min"] = float(
#                 coupling_explained_variance.float().min().item()
#             )
#         if targets_local is None:
#             return output

#         labels = targets_local.to(logits.device).long().flatten()
#         correct = predicted_local.eq(labels)
#         output["accuracy"] = float(correct.float().mean().item())

#         old_mask, new_mask = self._phase_masks(
#             seen,
#             old_classes=old_classes,
#             new_classes=new_classes,
#             old_class_count=old_class_count,
#             device=logits.device,
#         )
#         old_samples = old_mask.index_select(0, labels)
#         new_samples = new_mask.index_select(0, labels)
#         output["old_accuracy"] = (
#             float(correct[old_samples].float().mean().item())
#             if bool(old_samples.any()) else 0.0
#         )
#         output["new_accuracy"] = (
#             float(correct[new_samples].float().mean().item())
#             if bool(new_samples.any()) else 0.0
#         )

#         output["old_to_new_invasion"] = (
#             float(
#                 new_mask.index_select(
#                     0, predicted_local[old_samples]
#                 ).float().mean().item()
#             )
#             if bool(old_samples.any()) and bool(new_mask.any())
#             else 0.0
#         )
#         output["new_to_old_invasion"] = (
#             float(
#                 old_mask.index_select(
#                     0, predicted_local[new_samples]
#                 ).float().mean().item()
#             )
#             if bool(new_samples.any()) and bool(old_mask.any())
#             else 0.0
#         )

#         if occupancy_energy is not None:
#             occupancy_predicted = occupancy_energy.argmin(dim=1)
#             occupancy_correct = occupancy_predicted.eq(labels)
#             output["occupancy_only_accuracy"] = float(
#                 occupancy_correct.float().mean().item()
#             )
#             help_rate = float(
#                 ((~occupancy_correct) & correct).float().mean().item()
#             )
#             harm_rate = float(
#                 (occupancy_correct & (~correct)).float().mean().item()
#             )
#             output["conditional_tangent_help_rate"] = help_rate
#             output["conditional_tangent_harm_rate"] = harm_rate
#             # Compatibility aliases. These refer to p(g|z,c), not p(g|c).
#             output["response_help_rate"] = help_rate
#             output["response_harm_rate"] = harm_rate
#             stats = self.energy_margin_statistics(
#                 occupancy_energy, labels
#             )
#             output.update({
#                 f"occupancy_{key}": float(value.item())
#                 for key, value in stats.items()
#             })

#         if response_energy is not None:
#             conditional_accuracy = float(
#                 response_energy.argmin(dim=1).eq(labels).float().mean().item()
#             )
#             output["conditional_tangent_accuracy"] = conditional_accuracy
#             # Not a response-only classifier: this energy is conditioned on z.
#             output["response_only_accuracy"] = conditional_accuracy
#             stats = self.energy_margin_statistics(response_energy, labels)
#             output.update({
#                 f"conditional_tangent_{key}": float(value.item())
#                 for key, value in stats.items()
#             })

#         if joint_energy is not None:
#             stats = self.energy_margin_statistics(joint_energy, labels)
#             output.update({
#                 f"joint_{key}": float(value.item())
#                 for key, value in stats.items()
#             })
#         return output

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------
#     def forward(
#         self,
#         features: torch.Tensor,
#         seen_classes: Optional[Iterable[int]] = None,
#         geometry_bank: Any = None,
#         *,
#         spectral_responses: Optional[torch.Tensor] = None,
#         responses: Optional[torch.Tensor] = None,
#         bank: Any = None,
#         mode: str = "pc_stgb",
#         targets: Optional[torch.Tensor] = None,
#         targets_are_global: bool = False,
#         old_classes: Optional[Iterable[int]] = None,
#         new_classes: Optional[Iterable[int]] = None,
#         old_class_count: Optional[int] = None,
#         return_energy: bool = False,
#         return_parts: bool = False,
#         return_diagnostics: bool = False,
#         **_: Any,
#     ) -> Union[torch.Tensor, Dict[str, Any]]:
#         supplied_bank = (
#             geometry_bank if geometry_bank is not None else bank
#         )
#         if supplied_bank is None:
#             raise ValueError("forward requires geometry_bank")
#         mode = self.normalize_mode(mode)
#         seen = (
#             self._infer_seen(supplied_bank)
#             if seen_classes is None
#             else _unique_ids(seen_classes, name="seen_classes")
#         )
#         self.expand_to_seen_classes(seen)

#         selected_responses = (
#             spectral_responses
#             if spectral_responses is not None
#             else responses
#         )
#         if mode == self.METHOD_MODE:
#             if selected_responses is None:
#                 raise RuntimeError(
#                     "PC-STGB inference requires spectral_responses [B,K,D]; "
#                     "feature-only fallback is forbidden"
#                 )
#             scored = self.compute_joint_logits(
#                 features,
#                 selected_responses,
#                 seen_classes=seen,
#                 geometry_bank=supplied_bank,
#                 return_parts=True,
#             )
#             assert isinstance(scored, dict)
#             logits = scored["logits"]
#             joint_energy = scored["joint_energy"]
#             occupancy_energy = scored["occupancy_energy"]
#             response_energy = scored["response_energy"]
#             coupling_reliability = scored.get("coupling_reliability")
#             coupling_explained_variance = scored.get(
#                 "coupling_explained_variance"
#             )
#         else:
#             occupancy_energy, occupancy_parts = (
#                 self.compute_occupancy_energy_ablation(
#                     features,
#                     seen_classes=seen,
#                     geometry_bank=supplied_bank,
#                     return_parts=True,
#                 )
#             )
#             logits = self._energy_to_logits(occupancy_energy)
#             joint_energy = occupancy_energy
#             response_energy = None
#             coupling_reliability = None
#             coupling_explained_variance = None
#             scored = {
#                 "logits": logits,
#                 "energy": occupancy_energy,
#                 "raw_energy": occupancy_energy,
#                 "joint_energy": occupancy_energy,
#                 "occupancy_energy": occupancy_energy,
#                 "occupancy": occupancy_parts,
#                 "ablation": True,
#             }

#         local_targets = None
#         if targets is not None:
#             local_targets = (
#                 self.global_to_local_labels(targets, seen)
#                 if targets_are_global
#                 else targets.to(features.device).long().flatten()
#             )
#         self.assert_logits_valid(
#             logits,
#             seen_classes=seen,
#             targets=local_targets,
#         )

#         if not (return_energy or return_parts or return_diagnostics):
#             return logits

#         output: Dict[str, Any] = {
#             "logits": logits,
#             "energy": joint_energy,
#             "joint_energy": joint_energy,
#             "occupancy_energy": occupancy_energy,
#             "response_energy": response_energy,
#             "seen_classes": torch.tensor(
#                 seen, device=features.device, dtype=torch.long
#             ),
#             "mode": mode,
#         }
#         if return_parts:
#             output.update(scored)
#         if return_diagnostics:
#             output["diagnostics"] = self.classifier_diagnostics(
#                 logits,
#                 seen_classes=seen,
#                 targets_local=local_targets,
#                 joint_energy=joint_energy,
#                 occupancy_energy=occupancy_energy,
#                 response_energy=response_energy,
#                 coupling_reliability=coupling_reliability,
#                 coupling_explained_variance=coupling_explained_variance,
#                 old_classes=old_classes,
#                 new_classes=new_classes,
#                 old_class_count=old_class_count,
#             )
#         return output
