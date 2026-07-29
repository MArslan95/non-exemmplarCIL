from __future__ import annotations

"""Shared trainer contracts for transport-verified HSI factor geometry.

This module contains no optimization logic. It validates the integrated model,
phase schedule, label-column mapping, finalized aggregate memory, and report
serialization for the architecture:

    classification: p(z | c)
    spectral relation: p(h | c) for pair-risk margins only
    joint coordinate: z = [z_s ; z_p]

No spectral anchor, conditional-feature classifier, trainable transport,
feature replay, teacher, or sample-level memory is accepted by these helpers.
"""

import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import torch
import torch.nn.functional as F


class TrainerHelper:
    CLASSIFICATION_FACTORIZATION = "p(z|c)"
    SPECTRAL_RELATION_FACTORIZATION = "p(h|c)"

    # ------------------------------------------------------------------
    # Batch and label utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_hsi_batch(batch: Any) -> Tuple[Any, Any, Any, Any]:
        """Return reduced/model patch, global label, metadata, coordinates.

        Raw physical-band patches or raw center spectra remain in the original
        mapping/tuple. ``BasePhaseTrainer._find_raw_inputs`` resolves them so a
        reduced/PCA patch is never mistaken for an ordered physical spectrum.
        """
        if isinstance(batch, Mapping):
            sources: List[Mapping[str, Any]] = [batch]
            for key in ("hsi", "inputs", "data", "spectral", "metadata"):
                nested = batch.get(key)
                if isinstance(nested, Mapping):
                    sources.append(nested)

            patch = None
            label = None
            for source in sources:
                for name in (
                    "patch",
                    "patches",
                    "image",
                    "images",
                    "model_patch",
                    "reduced_patch",
                    "x",
                ):
                    value = source.get(name)
                    if torch.is_tensor(value):
                        patch = value
                        break
                if patch is not None:
                    break

            for source in sources:
                for name in ("label", "labels", "target", "targets", "y"):
                    value = source.get(name)
                    if torch.is_tensor(value):
                        label = value
                        break
                if label is not None:
                    break

            if patch is None or label is None:
                raise RuntimeError(
                    "batch mapping must contain a reduced/model patch and labels"
                )
            metadata = batch.get("metadata")
            coordinates = batch.get(
                "coord", batch.get("coords", batch.get("coordinate"))
            )
            return patch, label, metadata, coordinates

        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise RuntimeError("batch tuple/list must contain patch and label")
            # Additional tuple fields remain available to the caller through the
            # original batch object. The third item may be a raw spectral patch.
            return (
                batch[0],
                batch[1],
                batch[2] if len(batch) > 2 else None,
                batch[3] if len(batch) > 3 else None,
            )

        raise RuntimeError(f"unsupported batch type: {type(batch)!r}")

    @staticmethod
    def _class_ids(
        values: Iterable[int],
        *,
        name: str = "class_ids",
        allow_empty: bool = False,
    ) -> List[int]:
        output: List[int] = []
        observed: set[int] = set()
        for value in values:
            class_id = int(value)
            if class_id < 0:
                raise RuntimeError(f"{name} contains negative class ID {class_id}")
            if class_id in observed:
                raise RuntimeError(f"{name} contains duplicate class ID {class_id}")
            observed.add(class_id)
            output.append(class_id)
        if not output and not allow_empty:
            raise RuntimeError(f"{name} is empty")
        return output

    @staticmethod
    def global_to_local_labels(
        labels_global: torch.Tensor,
        class_order: Sequence[int],
        *,
        context: str = "global_to_local_labels",
    ) -> torch.Tensor:
        if not torch.is_tensor(labels_global):
            raise TypeError(f"{context}: labels must be a tensor")
        classes = TrainerHelper._class_ids(
            class_order, name=f"{context}.class_order"
        )
        labels = labels_global.long().flatten()
        class_tensor = torch.tensor(
            classes, device=labels.device, dtype=torch.long
        )
        matches = labels[:, None].eq(class_tensor[None, :])
        counts = matches.sum(dim=1)
        if bool(counts.ne(1).any()):
            bad = labels[counts.ne(1)].detach().cpu().unique().tolist()
            raise RuntimeError(f"{context}: labels outside class order: {bad}")
        return matches.to(torch.long).argmax(dim=1)

    @staticmethod
    def local_to_global_labels(
        labels_local: torch.Tensor,
        class_order: Sequence[int],
        *,
        context: str = "local_to_global_labels",
    ) -> torch.Tensor:
        classes = TrainerHelper._class_ids(
            class_order, name=f"{context}.class_order"
        )
        labels = labels_local.long().flatten()
        if labels.numel() and (
            int(labels.min().item()) < 0
            or int(labels.max().item()) >= len(classes)
        ):
            raise RuntimeError(f"{context}: local labels are out of range")
        mapping = torch.tensor(classes, device=labels.device, dtype=torch.long)
        return mapping.index_select(0, labels)

    @staticmethod
    def class_balanced_cross_entropy_local(
        logits: torch.Tensor,
        targets_local: torch.Tensor,
        *,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Small compatibility utility for already-local targets.

        Official base training should use ``base_ce_warmup_objective`` from the
        revised loss file, which requires explicit global class IDs.
        """
        if not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError("logits must be [N,C]")
        if logits.size(0) == 0 or not torch.isfinite(logits).all():
            raise RuntimeError("logits are empty or contain NaN/Inf")
        targets = targets_local.to(logits.device, torch.long).flatten()
        if targets.numel() != logits.size(0):
            raise RuntimeError("targets/logits are misaligned")
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= logits.size(1):
            raise RuntimeError("targets are outside the logit range")
        smoothing = float(label_smoothing)
        if not 0.0 <= smoothing < 1.0:
            raise RuntimeError("label_smoothing must lie in [0,1)")
        per_sample = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=smoothing,
        )
        terms = [
            per_sample[targets.eq(class_id)].mean()
            for class_id in torch.unique(targets, sorted=True)
        ]
        return torch.stack(terms).mean()

    # ------------------------------------------------------------------
    # Architecture and dataset contracts
    # ------------------------------------------------------------------

    def assert_architecture_contract(self) -> Dict[str, Any]:
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError("trainer has no model")
        model.assert_architecture_contract()
        summary = dict(model.architecture_summary())

        expected = {
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
            "joint_feature": "direct_[z_s;z_p]",
            "trainable_transport_network": False,
            "old_rows_evolve": True,
            "uses_geometry_replay_for_training": False,
            "stores_exemplars": False,
            "stores_old_features": False,
            "stores_old_spectra": False,
            "uses_knowledge_distillation": False,
            "uses_task_adapters": False,
        }
        failures = {
            key: (summary.get(key), expected_value)
            for key, expected_value in expected.items()
            if summary.get(key) != expected_value
        }
        if failures:
            raise RuntimeError(f"architecture contract mismatch: {failures}")

        bank = model.geometry_bank
        classifier = model.classifier
        backbone = model.backbone
        if int(model.feature_dim) != int(model.spectral_dim + model.spatial_dim):
            raise RuntimeError("model spectral/spatial dimensions are inconsistent")
        if int(bank.feature_dim) != int(model.feature_dim):
            raise RuntimeError("model and GeometryBank feature dimensions differ")
        if int(bank.spectral_dim) != int(model.spectral_dim):
            raise RuntimeError("model and GeometryBank spectral dimensions differ")
        if int(bank.spatial_dim) != int(model.spatial_dim):
            raise RuntimeError("model and GeometryBank spatial dimensions differ")
        if int(bank.raw_spectral_dim) != int(backbone.raw_num_bands):
            raise RuntimeError("bank and backbone raw spectral dimensions differ")
        if any(parameter.requires_grad for parameter in bank.parameters()):
            raise RuntimeError("GeometryBank must contain aggregate buffers only")
        if any(parameter.requires_grad for parameter in classifier.parameters()):
            raise RuntimeError("geometry classifier must be parameter-free")

        classifier_contract = dict(classifier.classifier_contract())
        required_classifier = {
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_usage": "pair-risk margins only",
            "uses_trainable_classifier_weights": False,
            "uses_class_specific_bias": False,
            "uses_task_specific_head": False,
            "uses_reliability_logit_penalty": False,
            "uses_raw_spectra_at_inference": False,
        }
        classifier_failures = {
            key: (classifier_contract.get(key), expected_value)
            for key, expected_value in required_classifier.items()
            if classifier_contract.get(key) != expected_value
        }
        if classifier_failures:
            raise RuntimeError(
                f"classifier contract mismatch: {classifier_failures}"
            )

        backbone_contract = dict(backbone.backbone_contract())
        required_backbone = {
            "joint_rule": "direct_concatenation_[z_s;z_p]",
            "learned_post_concatenation_fusion": False,
            "final_feature_normalization": False,
            "spatial_absolute_center_removed": True,
            "spectral_band_order_preserved": True,
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "compatible_transport": "blockdiag(a_s R_s, a_p R_p)",
            "uses_task_specific_adapter": False,
            "stores_exemplars": False,
        }
        backbone_failures = {
            key: (backbone_contract.get(key), expected_value)
            for key, expected_value in required_backbone.items()
            if backbone_contract.get(key) != expected_value
        }
        if backbone_failures:
            raise RuntimeError(f"backbone contract mismatch: {backbone_failures}")
        return summary

    def assert_base_architecture_contract(self) -> Dict[str, Any]:
        return self.assert_architecture_contract()

    def assert_dataset_contract(self) -> Dict[str, Any]:
        dataset = getattr(self, "dataset", None)
        if dataset is None:
            raise RuntimeError("trainer has no dataset")
        phase_map = getattr(dataset, "phase_to_classes", None)
        if phase_map is None:
            raise RuntimeError("dataset must expose phase_to_classes")

        if isinstance(phase_map, Mapping):
            phase_keys = sorted(int(key) for key in phase_map)
            if phase_keys != list(range(len(phase_keys))):
                raise RuntimeError(
                    "phase_to_classes keys must be contiguous starting at zero"
                )
            schedule = {
                phase: self._class_ids(
                    phase_map[phase], name=f"phase_{phase}_classes"
                )
                for phase in phase_keys
            }
        elif isinstance(phase_map, Sequence):
            schedule = {
                phase: self._class_ids(
                    values, name=f"phase_{phase}_classes"
                )
                for phase, values in enumerate(phase_map)
            }
        else:
            raise RuntimeError("phase_to_classes must be a mapping or sequence")

        if 0 not in schedule or len(schedule[0]) < 2:
            raise RuntimeError("base phase requires at least two classes")
        observed: set[int] = set()
        overlaps: Dict[int, List[int]] = {}
        for phase, class_ids in schedule.items():
            duplicate = sorted(observed & set(class_ids))
            if duplicate:
                overlaps[phase] = duplicate
            observed.update(class_ids)
        if overlaps:
            raise RuntimeError(
                f"classes are repeated across incremental phases: {overlaps}"
            )

        required = (
            "start_phase",
            "get_phase_dataloader",
            "get_cumulative_dataloader",
        )
        missing = [
            name for name in required if not callable(getattr(dataset, name, None))
        ]
        if missing:
            raise RuntimeError(f"dataset lacks required methods: {missing}")
        return {
            "schedule": schedule,
            "base_classes": list(schedule[0]),
            "number_of_phases": len(schedule),
            "all_classes": sorted(observed),
        }

    def assert_base_dataset_contract(self) -> List[int]:
        return list(self.assert_dataset_contract()["base_classes"])

    # ------------------------------------------------------------------
    # Final geometry and memory contracts
    # ------------------------------------------------------------------

    def _arg_float(self, name: str, default: float) -> float:
        value = float(getattr(self.args, name, default))
        if not math.isfinite(value):
            raise RuntimeError(f"{name} must be finite")
        return value

    def base_geometry_certificate(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        """Summarize the factor-geometry base gates used for checkpointing."""
        if not isinstance(report, Mapping):
            raise TypeError("base geometry report must be a mapping")
        admission = report.get("geometry_admission")
        oof = report.get("out_of_fold")
        if not isinstance(admission, Mapping):
            raise RuntimeError("base report lacks geometry_admission")
        if not isinstance(oof, Mapping):
            raise RuntimeError("base report lacks out_of_fold metrics")
        factor = oof.get("factor_geometry")
        prototype = oof.get("prototype")
        diagonal = oof.get("diagonal_geometry")
        if not all(isinstance(value, Mapping) for value in (factor, prototype, diagonal)):
            raise RuntimeError("OOF factor/prototype/diagonal metrics are incomplete")

        checks = {
            "structural_geometry_valid": bool(admission.get("ok", False)),
            "factor_gain_over_prototype": float(
                oof.get(
                    "factor_minus_prototype",
                    float(factor["accuracy"]) - float(prototype["accuracy"]),
                )
            )
            >= self._arg_float("base_min_factor_gain_over_prototype", 0.0),
            "factor_gain_over_diagonal": float(
                oof.get(
                    "factor_minus_diagonal",
                    float(factor["accuracy"]) - float(diagonal["accuracy"]),
                )
            )
            >= self._arg_float("base_min_factor_gain_over_diagonal", 0.0),
            "minimum_class_accuracy": float(
                factor.get("minimum_per_class_accuracy", 0.0)
            )
            >= self._arg_float("base_admission_min_class_accuracy", 0.0),
            "classification_violation": float(
                factor.get("classification_violation_rate", 1.0)
            )
            <= self._arg_float(
                "base_admission_max_classification_violation", 1.0
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        return {
            "passed": not failures,
            "failed_checks": failures,
            "checks": checks,
            "protocol": "out_of_fold_factor_vs_diagonal_vs_prototype",
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
        }

    def assert_final_base_memory(self, base_classes: Iterable[int]) -> Dict[str, Any]:
        ids = self._class_ids(base_classes, name="base_classes")
        model = self.model
        bank = model.geometry_bank
        classifier = model.classifier

        report = bank.assert_valid(ids, strict=True)
        if not bool(report.get("ok", False)):
            raise RuntimeError(
                "final base GeometryBank is invalid: "
                + "; ".join(report.get("errors", []))
            )

        valid_rows = torch.nonzero(
            bank.valid_mask(), as_tuple=False
        ).flatten().detach().cpu().tolist()
        if set(valid_rows) != set(ids) or len(valid_rows) != len(ids):
            raise RuntimeError(
                f"valid GeometryBank rows {valid_rows} do not match {ids}"
            )
        bank.assert_global_priors_ready()
        if not bool(bank.global_priors_frozen.item()):
            raise RuntimeError("base global geometry priors are not frozen")
        if not bool(bank.overlap_temperatures_ready.item()):
            raise RuntimeError("pair-risk overlap temperatures are absent")
        if not bool(bank.overlap_temperatures_frozen.item()):
            raise RuntimeError("pair-risk overlap temperatures are not frozen")

        bank_contract_digest = classifier.bank_contract_digest(bank)
        if classifier.bound_bank_contract_digest != bank_contract_digest:
            raise RuntimeError("classifier is not bound to the static bank contract")
        if not bool(classifier.require_bound_contract):
            raise RuntimeError("classifier contract enforcement is disabled")
        if getattr(model, "base_ce_head", None) is not None:
            raise RuntimeError("final base checkpoint still contains the CE head")
        if set(model.infer_seen_classes()) != set(ids):
            raise RuntimeError("model seen classes do not match committed base rows")
        if set(getattr(model, "old_classes", [])) != set(ids):
            raise RuntimeError("model old classes do not match base rows")
        if getattr(model, "new_classes", []):
            raise RuntimeError("final base model unexpectedly contains new classes")
        if int(getattr(model, "current_phase", -1)) != 0:
            raise RuntimeError("final base model has an invalid phase index")
        if getattr(model, "phase_mode", None) != "evaluation":
            raise RuntimeError("final base model must be in evaluation mode")
        if any(parameter.requires_grad for parameter in model.backbone.parameters()):
            raise RuntimeError("accepted base backbone is not frozen")

        memory = model.memory_snapshot()
        if bool(memory.get("stores_sample_level_memory", True)):
            raise RuntimeError("memory audit reports sample-level memory")
        if bool(memory.get("stores_phase_observer", True)):
            raise RuntimeError("base checkpoint contains a phase observer")

        return {
            "ok": True,
            "base_classes": ids,
            "valid_rows": valid_rows,
            "bank_schema_version": int(bank.SCHEMA_VERSION),
            "bank_contract_digest": bank_contract_digest,
            "rows_digest": bank.rows_digest(ids),
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
            "global_priors_frozen": True,
            "overlap_temperatures_frozen": True,
            "future_encoder_policy": "frozen_baseline_or_controlled_plasticity",
            "stores_exemplars": False,
            "stores_old_features": False,
            "stores_old_spectra": False,
            "stores_phase_observer": False,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()
        if isinstance(value, Mapping):
            return {
                str(key): TrainerHelper._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [TrainerHelper._json_safe(item) for item in value]
        if isinstance(value, set):
            return [
                TrainerHelper._json_safe(item)
                for item in sorted(value, key=str)
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def save_json_diagnostics(self, path: str, data: Mapping[str, Any]) -> None:
        path = os.path.abspath(str(path))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(self._json_safe(dict(data)), stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise







# from __future__ import annotations

# import json
# import math
# import os
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# import torch
# import torch.nn.functional as F


# class TrainerHelper:
#     """Shared cross-phase invariants for strict PC-STGB NECIL-HSI.

#     The helper owns protocol enforcement only.  It does not estimate geometry,
#     define the learning objective, refine descriptors, or decide whether a
#     candidate phase should be committed.

#     Persistent memory contract
#     --------------------------
#     Every valid class row contains one connected aggregate model:

#     * low-rank occupancy geometry ``p(z|c)`` in the unnormalised canonical
#       Euclidean feature space;
#     * an occupancy-to-tangent coupling that predicts the expected physical
#       finite-difference response from the query occupancy coordinates;
#     * low-rank residual tangent geometry ``p(g_k|z,c)``;
#     * aggregate support, reliability, phase, energy, and margin statistics.

#     The deployed score in every phase is

#         E_c = E_c^occupancy + beta_T * E_c^tangent(g|z,c).

#     The helper never estimates geometry or defines losses.  It enforces that
#     base, incremental, replay, candidate, checkpoint, and diagnostic paths use
#     the same schema-v5 conditional joint factorisation and frozen base prior.
#     No raw old patches, old spectra, individual old features/responses,
#     teacher, adapter, transport, calibrator, or independent response side-bank
#     is permitted.
#     """

#     METHOD_NAME = "PC-STGB"
#     BANK_SCHEMA_VERSION = 5
#     JOINT_FACTORIZATION = "p(z|c)prod_k p(g_k|z,c)"

#     FEATURE_ROW_FIELDS: Tuple[str, ...] = (
#         "means",
#         "bases",
#         "eigvals",
#         "res_vars",
#         "active_ranks",
#         "sample_counts",
#         "reliability",
#         "captured_energy",
#         "noise_floors",
#     )
#     RESPONSE_ROW_FIELDS: Tuple[str, ...] = (
#         "response_bases",
#         "response_means",
#         "response_eigvals",
#         "response_res_vars",
#         "response_active_ranks",
#         "response_sample_counts",
#         "response_reliability",
#         "response_captured_energy",
#         "response_stats_ready",
#     )
#     COUPLING_ROW_FIELDS: Tuple[str, ...] = (
#         "response_couplings",
#         "response_coupling_reliability",
#         "response_coupling_explained_variance",
#         "response_coupling_ready",
#     )
#     GLOBAL_CONTRACT_FIELDS: Tuple[str, ...] = (
#         "response_prior_mean",
#         "response_prior_variance",
#         "response_prior_sample_count",
#         "response_prior_ready",
#         "response_prior_frozen",
#     )
#     STATISTIC_ROW_FIELDS: Tuple[str, ...] = (
#         "energy_quantiles",
#         "margin_quantiles",
#         "energy_stats_ready",
#         "margin_stats_ready",
#         "phase_created",
#         "frozen_class_mask",
#     )
#     PERSISTENT_ROW_FIELDS: Tuple[str, ...] = (
#         *FEATURE_ROW_FIELDS,
#         *RESPONSE_ROW_FIELDS,
#         *COUPLING_ROW_FIELDS,
#         *STATISTIC_ROW_FIELDS,
#     )

#     # ------------------------------------------------------------------
#     # Generic utilities
#     # ------------------------------------------------------------------
#     def _zero(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if torch.is_tensor(ref):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     @staticmethod
#     def _as_class_list(
#         class_ids: Iterable[int],
#         *,
#         name: str = "class_ids",
#         allow_empty: bool = False,
#     ) -> List[int]:
#         result: List[int] = []
#         observed = set()
#         for value in class_ids:
#             class_id = int(value)
#             if class_id < 0:
#                 raise RuntimeError(f"{name} contains negative class ID {class_id}")
#             if class_id in observed:
#                 raise RuntimeError(f"{name} contains duplicate class ID {class_id}")
#             observed.add(class_id)
#             result.append(class_id)
#         if not result and not allow_empty:
#             raise RuntimeError(f"{name} is empty")
#         return result

#     def _cfg_bool(self, name: str, default: bool = False) -> bool:
#         args = getattr(self, "args", None)
#         local = getattr(self, name, None)
#         value = local if local is not None else getattr(args, name, default) if args is not None else default
#         if value is None:
#             value = default
#         if isinstance(value, bool):
#             return value
#         if isinstance(value, int) and value in {0, 1}:
#             return bool(value)
#         if isinstance(value, str):
#             token = value.strip().lower()
#             if token in {"1", "true", "yes", "y", "on"}:
#                 return True
#             if token in {"0", "false", "no", "n", "off", "none", "null", ""}:
#                 return False
#         raise RuntimeError(f"{name} must be an explicit boolean, got {value!r}")

#     def _cfg_float(self, name: str, default: float) -> float:
#         args = getattr(self, "args", None)
#         local = getattr(self, name, None)
#         value = local if local is not None else getattr(args, name, default) if args is not None else default
#         value = default if value is None else value
#         result = float(value)
#         if not math.isfinite(result):
#             raise RuntimeError(f"{name} must be finite, got {value!r}")
#         return result

#     def _cfg_int(self, name: str, default: int) -> int:
#         args = getattr(self, "args", None)
#         local = getattr(self, name, None)
#         value = local if local is not None else getattr(args, name, default) if args is not None else default
#         value = default if value is None else value
#         if isinstance(value, bool):
#             raise RuntimeError(f"{name} must be an integer, not bool")
#         result = int(value)
#         if float(value) != float(result):
#             raise RuntimeError(f"{name} must be an integer, got {value!r}")
#         return result

#     @staticmethod
#     def _first_tensor(mapping: Mapping[str, Any], names: Sequence[str]) -> Optional[torch.Tensor]:
#         for name in names:
#             value = mapping.get(name)
#             if torch.is_tensor(value):
#                 return value
#         return None

#     @staticmethod
#     def _unpack_hsi_batch(batch: Any) -> Tuple[Any, Any, Any, Any]:
#         """Return input patch, global label, optional metadata, and coordinates.

#         The third item is retained as a compatibility metadata slot.  PC-STGB
#         does not consume a raw centre spectrum in the model or classifier.
#         Spectral evidence is supplied only through paired processed +/- views.
#         """
#         if isinstance(batch, Mapping):
#             x = batch.get("image", batch.get("patch", batch.get("patches")))
#             y = batch.get("label", batch.get("labels", batch.get("target")))
#             metadata = batch.get(
#                 "physical_center_spectrum",
#                 batch.get("center_spectrum", batch.get("spectrum", batch.get("spectra"))),
#             )
#             coords = batch.get("coord", batch.get("coords", batch.get("coordinate")))
#             if x is None or y is None:
#                 raise RuntimeError(
#                     "Batch mapping must contain image/patch/patches and label/labels/target"
#                 )
#             return x, y, metadata, coords
#         if isinstance(batch, (tuple, list)):
#             if len(batch) < 2:
#                 raise RuntimeError("Batch tuple/list must contain at least input and label")
#             return (
#                 batch[0],
#                 batch[1],
#                 batch[2] if len(batch) > 2 else None,
#                 batch[3] if len(batch) > 3 else None,
#             )
#         raise RuntimeError(f"Unsupported batch type: {type(batch)}")

#     def _stable_ce(
#         self,
#         logits: torch.Tensor,
#         labels_local: torch.Tensor,
#         *,
#         class_balanced: bool = False,
#     ) -> torch.Tensor:
#         if not torch.is_tensor(logits) or logits.numel() == 0:
#             return self._zero(logits if torch.is_tensor(logits) else None)
#         if logits.dim() != 2 or not torch.isfinite(logits).all():
#             raise RuntimeError("CE logits must be a finite [B,C] tensor")
#         labels = labels_local.to(device=logits.device, dtype=torch.long).flatten()
#         if labels.numel() != logits.size(0):
#             raise RuntimeError("CE labels/logits batch mismatch")
#         self.assert_valid_ce_targets(labels, logits.size(1), "stable_ce")
#         if abs(self._cfg_float("ce_logit_clip", 0.0)) > 1e-12:
#             raise RuntimeError("ce_logit_clip must be zero; clipping changes score geometry")
#         smoothing = self._cfg_float("label_smoothing", 0.0)
#         if not 0.0 <= smoothing < 1.0:
#             raise RuntimeError("label_smoothing must lie in [0,1)")
#         per_sample = F.cross_entropy(
#             logits,
#             labels,
#             label_smoothing=smoothing,
#             reduction="none",
#         )
#         if not class_balanced:
#             return per_sample.mean()
#         terms = [
#             per_sample[labels.eq(class_id)].mean()
#             for class_id in torch.unique(labels, sorted=True)
#         ]
#         return torch.stack(terms).mean() if terms else logits.sum() * 0.0

#     # ------------------------------------------------------------------
#     # Global <-> seen-local labels
#     # ------------------------------------------------------------------
#     def _classes_tensor(
#         self,
#         class_ids: Iterable[int],
#         *,
#         device: Optional[torch.device] = None,
#         name: str = "class_ids",
#     ) -> torch.Tensor:
#         ids = self._as_class_list(class_ids, name=name)
#         return torch.tensor(
#             ids,
#             device=self.device if device is None else device,
#             dtype=torch.long,
#         )

#     def assert_global_labels_in_set(
#         self,
#         labels_global: torch.Tensor,
#         allowed_classes: Iterable[int],
#         context: str,
#     ) -> None:
#         if not torch.is_tensor(labels_global):
#             raise RuntimeError(f"{context}: labels must be a tensor")
#         labels = labels_global.long().flatten()
#         if labels.numel() == 0:
#             raise RuntimeError(f"{context}: labels are empty")
#         allowed = self._classes_tensor(
#             allowed_classes,
#             device=labels.device,
#             name=f"{context}.allowed_classes",
#         )
#         valid = labels[:, None].eq(allowed[None, :]).any(dim=1)
#         if not bool(valid.all().item()):
#             bad = torch.unique(labels[~valid]).detach().cpu().tolist()
#             raise RuntimeError(
#                 f"{context}: labels outside allowed classes; bad={bad}, "
#                 f"allowed={allowed.detach().cpu().tolist()}"
#             )

#     @staticmethod
#     def assert_valid_ce_targets(
#         labels_local: torch.Tensor,
#         num_classes: int,
#         context: str,
#     ) -> None:
#         if not torch.is_tensor(labels_local):
#             raise RuntimeError(f"{context}: targets must be a tensor")
#         labels = labels_local.long().flatten()
#         if labels.numel() == 0:
#             raise RuntimeError(f"{context}: targets are empty")
#         if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(num_classes):
#             raise RuntimeError(
#                 f"{context}: targets [{int(labels.min())},{int(labels.max())}] "
#                 f"outside [0,{int(num_classes)-1}]"
#             )

#     def global_to_seen_local(
#         self,
#         labels_global: torch.Tensor,
#         seen_classes: Iterable[int],
#         *,
#         context: str = "global_to_seen_local",
#     ) -> torch.Tensor:
#         labels = labels_global.long().flatten()
#         seen = self._classes_tensor(
#             seen_classes,
#             device=labels.device,
#             name=f"{context}.seen_classes",
#         )
#         matches = labels[:, None].eq(seen[None, :])
#         valid = matches.any(dim=1)
#         if not bool(valid.all().item()):
#             bad = torch.unique(labels[~valid]).detach().cpu().tolist()
#             raise RuntimeError(f"{context}: labels outside seen classes: {bad}")
#         output = matches.long().argmax(dim=1)
#         self.assert_valid_ce_targets(output, int(seen.numel()), context)
#         return output

#     def seen_local_to_global(
#         self,
#         predictions_local: torch.Tensor,
#         seen_classes: Iterable[int],
#         *,
#         context: str = "seen_local_to_global",
#     ) -> torch.Tensor:
#         predictions = predictions_local.long().flatten()
#         seen = self._classes_tensor(
#             seen_classes,
#             device=predictions.device,
#             name=f"{context}.seen_classes",
#         )
#         if predictions.numel() == 0:
#             return predictions
#         self.assert_valid_ce_targets(predictions, int(seen.numel()), context)
#         return seen.index_select(0, predictions)

#     def global_to_phase_local(
#         self,
#         labels_global: torch.Tensor,
#         phase_classes: Iterable[int],
#         *,
#         context: str = "global_to_phase_local",
#     ) -> torch.Tensor:
#         return self.global_to_seen_local(labels_global, phase_classes, context=context)

#     @staticmethod
#     def assert_seen_logits(
#         logits: torch.Tensor,
#         seen_classes: Iterable[int],
#         context: str,
#     ) -> None:
#         if not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError(f"{context}: logits must be [B,S]")
#         seen = [int(value) for value in seen_classes]
#         if logits.size(1) != len(seen):
#             raise RuntimeError(
#                 f"{context}: logits width={logits.size(1)} != seen width={len(seen)}"
#             )
#         if not torch.isfinite(logits).all():
#             raise RuntimeError(f"{context}: logits contain NaN/Inf")

#     def cross_entropy_for_seen_logits(
#         self,
#         logits: torch.Tensor,
#         labels_global: torch.Tensor,
#         seen_classes: Iterable[int],
#         *,
#         context: str = "seen_ce",
#         class_balanced: bool = True,
#     ) -> torch.Tensor:
#         self.assert_seen_logits(logits, seen_classes, context)
#         labels_local = self.global_to_seen_local(
#             labels_global.to(logits.device), seen_classes, context=context
#         )
#         return self._stable_ce(logits, labels_local, class_balanced=class_balanced)

#     # ------------------------------------------------------------------
#     # Canonical feature and response extraction
#     # ------------------------------------------------------------------
#     @staticmethod
#     def assert_feature_tensor(
#         features: torch.Tensor,
#         *,
#         expected_dim: Optional[int] = None,
#         context: str = "features",
#     ) -> None:
#         if not torch.is_tensor(features) or features.dim() != 2:
#             raise RuntimeError(f"{context}: features must be [B,D]")
#         if expected_dim is not None and features.size(1) != int(expected_dim):
#             raise RuntimeError(
#                 f"{context}: feature dimension={features.size(1)} != {int(expected_dim)}"
#             )
#         if not torch.isfinite(features).all():
#             raise RuntimeError(f"{context}: features contain NaN/Inf")

#     @staticmethod
#     def assert_response_tensor(
#         responses: torch.Tensor,
#         *,
#         batch_size: Optional[int] = None,
#         num_interventions: Optional[int] = None,
#         feature_dim: Optional[int] = None,
#         context: str = "spectral_responses",
#     ) -> None:
#         if not torch.is_tensor(responses) or responses.dim() != 3:
#             raise RuntimeError(f"{context}: responses must be [B,K,D]")
#         if batch_size is not None and responses.size(0) != int(batch_size):
#             raise RuntimeError(f"{context}: batch size mismatch")
#         if num_interventions is not None and responses.size(1) != int(num_interventions):
#             raise RuntimeError(
#                 f"{context}: intervention count={responses.size(1)} != {int(num_interventions)}"
#             )
#         if feature_dim is not None and responses.size(2) != int(feature_dim):
#             raise RuntimeError(
#                 f"{context}: feature dimension={responses.size(2)} != {int(feature_dim)}"
#             )
#         if not torch.isfinite(responses).all():
#             raise RuntimeError(f"{context}: responses contain NaN/Inf")

#     def _response_views_from_batch(
#         self,
#         batch: Any,
#         x: torch.Tensor,
#         *,
#         required: bool = True,
#         context: str = "response_views",
#     ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#         positive: Optional[torch.Tensor] = None
#         negative: Optional[torch.Tensor] = None
#         steps: Optional[torch.Tensor] = None
#         version: Optional[Any] = None

#         if isinstance(batch, Mapping):
#             nested = batch.get("pc_stgb", batch.get("pc_sirg"))
#             sources = [batch, nested if isinstance(nested, Mapping) else {}]
#             for source in sources:
#                 if positive is None:
#                     positive = self._first_tensor(
#                         source,
#                         (
#                             "spectral_positive_patches",
#                             "positive_spectral_patches",
#                             "spectral_pos_patches",
#                             "pc_stgb_positive_patches",
#                             "pc_sirg_positive_patches",
#                             "positive_intervention_patches",
#                         ),
#                     )
#                 if negative is None:
#                     negative = self._first_tensor(
#                         source,
#                         (
#                             "spectral_negative_patches",
#                             "negative_spectral_patches",
#                             "spectral_neg_patches",
#                             "pc_stgb_negative_patches",
#                             "pc_sirg_negative_patches",
#                             "negative_intervention_patches",
#                         ),
#                     )
#                 if steps is None:
#                     steps = self._first_tensor(
#                         source,
#                         (
#                             "spectral_step_sizes",
#                             "spectral_response_steps",
#                             "intervention_step_sizes",
#                             "pc_stgb_step_sizes",
#                             "pc_sirg_step_sizes",
#                         ),
#                     )
#                 if version is None:
#                     version = source.get(
#                         "intervention_definition_version",
#                         source.get(
#                             "pc_stgb_intervention_version",
#                             source.get("pc_sirg_intervention_version"),
#                         ),
#                     )

#             legacy_view = self._first_tensor(
#                 batch,
#                 (
#                     "spectral_augmented_image",
#                     "spectral_augmented_patch",
#                     "spectral_view",
#                     "augmented_patch",
#                     "image_aug",
#                 ),
#             )
#             if legacy_view is not None and (
#                 positive is None or negative is None or steps is None
#             ):
#                 raise RuntimeError(
#                     f"{context}: one-sided spectral augmentation is retired; "
#                     "provide paired +/- views and finite central-difference step sizes"
#                 )
#         elif isinstance(batch, (tuple, list)) and len(batch) >= 7:
#             positive = batch[4] if torch.is_tensor(batch[4]) else None
#             negative = batch[5] if torch.is_tensor(batch[5]) else None
#             steps = batch[6] if torch.is_tensor(batch[6]) else None
#             version = batch[7] if len(batch) > 7 else None

#         if positive is None or negative is None or steps is None:
#             if required:
#                 raise RuntimeError(
#                     f"{context}: PC-STGB requires positive patches, negative patches, "
#                     "and central-difference step sizes"
#                 )
#             return None

#         positive = positive.to(device=x.device, dtype=x.dtype, non_blocking=True)
#         negative = negative.to(device=x.device, dtype=x.dtype, non_blocking=True)
#         steps = steps.to(device=x.device, dtype=x.dtype, non_blocking=True)

#         model_k = int(getattr(self.model, "num_interventions", 0))
#         if model_k <= 0:
#             raise RuntimeError(f"{context}: model has invalid intervention count")
#         if positive.dim() == 4 and model_k == 1:
#             positive = positive.unsqueeze(1)
#             negative = negative.unsqueeze(1)
#         if positive.dim() != 5 or negative.shape != positive.shape:
#             raise RuntimeError(f"{context}: paired views must have identical [B,K,C,H,W] shape")
#         if positive.size(0) != x.size(0) or tuple(positive.shape[2:]) != tuple(x.shape[1:]):
#             raise RuntimeError(f"{context}: paired views are not aligned with the original patch batch")
#         if positive.size(1) != model_k:
#             raise RuntimeError(
#                 f"{context}: intervention count={positive.size(1)} != model K={model_k}"
#             )
#         if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
#             raise RuntimeError(f"{context}: paired views contain NaN/Inf")

#         batch_size = x.size(0)
#         if steps.dim() == 0:
#             pass
#         elif steps.dim() == 1:
#             if steps.numel() == model_k:
#                 steps = steps.unsqueeze(0).expand(batch_size, -1)
#             elif steps.numel() != batch_size:
#                 raise RuntimeError(f"{context}: 1-D steps must contain B or K values")
#         elif steps.dim() == 2:
#             if steps.shape != (batch_size, model_k):
#                 raise RuntimeError(f"{context}: 2-D steps must be [B,K]")
#         else:
#             raise RuntimeError(f"{context}: steps must be scalar, [B], [K], or [B,K]")
#         if not torch.isfinite(steps).all() or bool(steps.abs().le(1e-12).any().item()):
#             raise RuntimeError(f"{context}: step sizes must be finite and non-zero")

#         expected_version = int(getattr(self.model, "intervention_definition_version", 0))
#         if version is not None:
#             observed_version = int(torch.as_tensor(version).reshape(-1)[0].item())
#             if observed_version != expected_version:
#                 raise RuntimeError(
#                     f"{context}: intervention version={observed_version} != model version={expected_version}"
#                 )
#         return positive, negative, steps

#     def _extract_model_geometry_features(
#         self,
#         x: torch.Tensor,
#         *,
#         deterministic: bool = True,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#         require_physical_spectra: bool = False,
#     ) -> Dict[str, Any]:
#         """Extract canonical ``z`` through the only supported feature path.

#         Legacy raw spectral-summary arguments are accepted solely to produce a
#         precise migration error.  Raw spectra must never enter the canonical
#         feature map in PC-STGB.
#         """
#         if spectral_summary is not None or spectral_summary_is_physical or require_physical_spectra:
#             raise RuntimeError(
#                 "Raw spectral summaries are retired. Use paired processed +/- views "
#                 "and compute central responses in canonical z-space."
#             )
#         if not torch.is_tensor(x) or x.dim() < 2:
#             raise RuntimeError("model input must be a tensor with a batch dimension")
#         method = getattr(self.model, "extract_canonical_geometry_features", None)
#         if not callable(method):
#             raise AttributeError("Model must expose extract_canonical_geometry_features()")
#         output = method(x, deterministic=bool(deterministic), return_dict=True)
#         if torch.is_tensor(output):
#             output = {"features": output}
#         if not isinstance(output, Mapping):
#             raise RuntimeError("Canonical feature extractor must return a tensor or mapping")
#         result = dict(output)
#         feature = next(
#             (
#                 result[key]
#                 for key in (
#                     "canonical_projected_features",
#                     "canonical_features",
#                     "geometry_features",
#                     "projected_features",
#                     "features",
#                 )
#                 if torch.is_tensor(result.get(key))
#             ),
#             None,
#         )
#         if feature is None:
#             raise RuntimeError("Canonical feature extractor returned no feature tensor")
#         expected_dim = int(
#             getattr(
#                 getattr(self.model, "geometry_bank", None),
#                 "feature_dim",
#                 getattr(self.model, "d_model", feature.size(1)),
#             )
#         )
#         self.assert_feature_tensor(
#             feature,
#             expected_dim=expected_dim,
#             context="canonical geometry features",
#         )
#         result.update(
#             {
#                 "features": feature,
#                 "projected_features": feature,
#                 "geometry_features": feature,
#                 "canonical_features": feature,
#                 "canonical_projected_features": feature,
#                 "geometry_feature_space": "unnormalized_euclidean_residual_projected_z",
#                 "classifier_feature_space": "unnormalized_euclidean_residual_projected_z",
#             }
#         )
#         return result

#     def _extract_model_geometry_tuple(
#         self,
#         batch: Any,
#         *,
#         require_grad: bool,
#         require_response_views: bool = True,
#         context: str = "geometry_tuple",
#     ) -> Dict[str, torch.Tensor]:
#         """Extract one aligned occupancy--tangent tuple through the model.

#         The original patch and all +/- intervention views are evaluated under
#         one deterministic module state by ``extract_joint_geometry_tuple``.
#         Separate feature and response forwards are forbidden in the main path.
#         """
#         x, labels, _, _ = self._unpack_hsi_batch(batch)
#         x = x.float().to(self.device, non_blocking=True)
#         labels = labels.long().to(self.device, non_blocking=True).flatten()
#         views = self._response_views_from_batch(
#             batch,
#             x,
#             required=require_response_views,
#             context=f"{context}.response_views",
#         )
#         manager = torch.enable_grad() if require_grad else torch.inference_mode()
#         with manager:
#             if views is None:
#                 feature_output = self._extract_model_geometry_features(
#                     x, deterministic=True
#                 )
#                 features = feature_output["features"]
#                 responses = features.new_empty(
#                     (features.size(0), 0, features.size(1))
#                 )
#                 positive_features = features.new_empty(
#                     (features.size(0), 0, features.size(1))
#                 )
#                 negative_features = positive_features.clone()
#                 steps = features.new_empty((0,))
#                 joint_factorization = None
#             else:
#                 positive, negative, steps = views
#                 method = getattr(
#                     self.model, "extract_joint_geometry_tuple", None
#                 )
#                 if not callable(method):
#                     raise AttributeError(
#                         "PC-STGB model must expose extract_joint_geometry_tuple()"
#                     )
#                 output = method(
#                     x,
#                     positive,
#                     negative,
#                     step_sizes=steps,
#                     deterministic=True,
#                     return_view_features=True,
#                 )
#                 if not isinstance(output, Mapping):
#                     raise RuntimeError(
#                         "extract_joint_geometry_tuple() must return a mapping"
#                     )
#                 features = output.get("features")
#                 responses = output.get(
#                     "spectral_responses", output.get("responses")
#                 )
#                 positive_features = output.get("positive_features")
#                 negative_features = output.get("negative_features")
#                 joint_factorization = output.get("joint_factorization")
#                 if joint_factorization != self.JOINT_FACTORIZATION:
#                     raise RuntimeError(
#                         f"{context}: model returned joint factorisation "
#                         f"{joint_factorization!r}, expected "
#                         f"{self.JOINT_FACTORIZATION!r}"
#                     )
#                 if not torch.is_tensor(positive_features) or not torch.is_tensor(
#                     negative_features
#                 ):
#                     raise RuntimeError(
#                         f"{context}: aligned view features were not returned"
#                     )
#                 self.assert_response_tensor(
#                     responses,
#                     batch_size=x.size(0),
#                     num_interventions=int(self.model.num_interventions),
#                     feature_dim=int(self.model.d_model),
#                     context=f"{context}.spectral_responses",
#                 )

#         self.assert_feature_tensor(
#             features,
#             expected_dim=int(self.model.d_model),
#             context=f"{context}.features",
#         )
#         if features.size(0) != labels.numel():
#             raise RuntimeError(f"{context}: feature/label batch mismatch")
#         if responses.device != features.device:
#             raise RuntimeError(
#                 f"{context}: features and tangents must share one device"
#             )
#         return {
#             "patches": x,
#             "labels": labels,
#             "features": features,
#             "canonical_features": features,
#             "spectral_responses": responses,
#             "responses": responses,
#             "positive_features": positive_features,
#             "negative_features": negative_features,
#             "step_sizes": steps,
#             "joint_factorization": joint_factorization,
#         }


#     # ------------------------------------------------------------------
#     # GeometryBank access and validation
#     # ------------------------------------------------------------------
#     def _geometry_bank_object(self) -> Any:
#         bank = getattr(self.model, "geometry_bank", None)
#         if bank is None:
#             raise RuntimeError("Model has no GeometryBank")
#         if int(getattr(bank, "SCHEMA_VERSION", -1)) != self.BANK_SCHEMA_VERSION:
#             raise RuntimeError(
#                 "TrainerHelper requires the schema-v5 conditional PC-STGB bank"
#             )
#         return bank

#     def _bank_contract_digest(self) -> str:
#         bank = self._geometry_bank_object()
#         method = getattr(bank, "contract_digest", None)
#         if not callable(method):
#             raise RuntimeError("GeometryBank must expose contract_digest()")
#         digest = str(method()).strip().lower()
#         if len(digest) != 64:
#             raise RuntimeError("GeometryBank contract digest is invalid")
#         return digest

#     def _classifier_bound_digest(self) -> Optional[str]:
#         classifier = getattr(self.model, "classifier", None)
#         if classifier is None:
#             raise RuntimeError("Model has no geometry classifier")
#         value = getattr(classifier, "bound_contract_digest", None)
#         if callable(value):
#             value = value()
#         return None if value is None else str(value).strip().lower()

#     def _model_contract_digest(self) -> Optional[str]:
#         method = getattr(self.model, "model_contract_digest", None)
#         return str(method()) if callable(method) else None

#     def _canonicalize_bank(
#         self,
#         bank: Mapping[str, Any],
#         *,
#         require_response_schema: bool = True,
#         require_coupling_schema: bool = True,
#         require_prior_schema: bool = True,
#     ) -> Dict[str, Any]:
#         if not isinstance(bank, Mapping):
#             raise TypeError("GeometryBank state must be a mapping")
#         output = dict(bank)
#         required_rows = list(self.FEATURE_ROW_FIELDS)
#         if require_response_schema:
#             required_rows.extend(self.RESPONSE_ROW_FIELDS)
#         if require_coupling_schema:
#             required_rows.extend(self.COUPLING_ROW_FIELDS)
#         missing = [
#             name for name in required_rows
#             if not torch.is_tensor(output.get(name))
#         ]
#         if missing:
#             raise RuntimeError(
#                 f"GeometryBank mapping is missing schema-v5 row tensors: {missing}"
#             )
#         row_count = int(output["means"].size(0))
#         for name in required_rows:
#             tensor = output[name]
#             if tensor.dim() == 0 or tensor.size(0) != row_count:
#                 raise RuntimeError(
#                     f"GeometryBank {name} has shape {tuple(tensor.shape)}; "
#                     f"expected {row_count} rows"
#                 )
#             if (
#                 tensor.dtype != torch.bool
#                 and tensor.numel()
#                 and not torch.isfinite(tensor.float()).all()
#             ):
#                 raise RuntimeError(f"GeometryBank {name} contains NaN/Inf")

#         if require_prior_schema:
#             missing_prior = [
#                 name for name in self.GLOBAL_CONTRACT_FIELDS
#                 if not torch.is_tensor(output.get(name))
#             ]
#             if missing_prior:
#                 raise RuntimeError(
#                     "GeometryBank mapping lacks the frozen response-prior "
#                     f"contract: {missing_prior}"
#                 )
#             for name in self.GLOBAL_CONTRACT_FIELDS:
#                 tensor = output[name]
#                 if (
#                     tensor.dtype != torch.bool
#                     and tensor.numel()
#                     and not torch.isfinite(tensor.float()).all()
#                 ):
#                     raise RuntimeError(f"GeometryBank {name} contains NaN/Inf")

#         if not torch.is_tensor(output.get("variances")):
#             output["variances"] = torch.cat(
#                 [output["eigvals"], output["res_vars"].reshape(-1, 1)],
#                 dim=1,
#             )
#         if not torch.is_tensor(output.get("valid_mask")):
#             counts = output["sample_counts"].flatten()
#             mask = torch.isfinite(counts) & counts.gt(0)
#             if require_response_schema:
#                 mask &= output["response_stats_ready"].bool().flatten()
#             if require_coupling_schema:
#                 mask &= output["response_coupling_ready"].bool().flatten()
#             output["valid_mask"] = mask

#         valid = output["valid_mask"].bool().flatten()
#         if valid.numel() != row_count:
#             raise RuntimeError("GeometryBank valid_mask row count mismatch")
#         if require_response_schema:
#             bad = torch.nonzero(
#                 valid & ~output["response_stats_ready"].bool().flatten(),
#                 as_tuple=False,
#             ).flatten().tolist()
#             if bad:
#                 raise RuntimeError(
#                     f"Valid rows lack tangent residual geometry: {bad}"
#                 )
#         if require_coupling_schema:
#             bad = torch.nonzero(
#                 valid & ~output["response_coupling_ready"].bool().flatten(),
#                 as_tuple=False,
#             ).flatten().tolist()
#             if bad:
#                 raise RuntimeError(
#                     f"Valid rows lack occupancy--tangent coupling: {bad}"
#                 )
#         return output

#     def _safe_get_subspace_bank(self, require_ready: bool = True) -> Dict[str, Any]:
#         """Compatibility name for retrieving the live schema-v5 PC-STGB bank."""
#         geometry_bank = self._geometry_bank_object()
#         get_bank = getattr(geometry_bank, "get_bank", None)
#         if not callable(get_bank):
#             raise AttributeError("GeometryBank must expose get_bank()")
#         bank = self._canonicalize_bank(get_bank())
#         if require_ready:
#             report = geometry_bank.assert_bank_valid(strict=False)
#             if not bool(report.get("ok", False)):
#                 raise RuntimeError(
#                     "Invalid PC-STGB bank: "
#                     + "; ".join(report.get("errors", []))
#                 )
#             valid = geometry_bank.get_valid_mask().detach().clone()
#             bank["valid_mask"] = valid
#             if not bool(valid.any().item()):
#                 raise RuntimeError("GeometryBank has no valid PC-STGB rows")
#             geometry_bank.assert_response_prior_ready()
#         return bank

#     def assert_bank_ready_for_seen_classes(
#         self,
#         bank: Optional[Mapping[str, Any]],
#         seen_classes: Iterable[int],
#         *,
#         require_statistics: bool = False,
#         require_spectral: bool = True,
#         require_response: Optional[bool] = None,
#         require_joint_state: bool = True,
#         require_any_joint_ready: bool = False,
#         require_frozen: bool = False,
#         require_frozen_prior: bool = True,
#         require_bound_contract: bool = False,
#     ) -> None:
#         """Validate complete conditional joint rows for every seen class."""
#         ids = self._as_class_list(seen_classes, name="seen_classes")
#         response_required = bool(
#             require_spectral if require_response is None else require_response
#         )
#         coupling_required = bool(require_joint_state or require_any_joint_ready)
#         geometry_bank = self._geometry_bank_object()
#         report = geometry_bank.assert_bank_valid(ids, strict=False)
#         if not bool(report.get("ok", False)):
#             raise RuntimeError(
#                 "GeometryBank rows are invalid: "
#                 + "; ".join(report.get("errors", []))
#             )
#         valid = geometry_bank.get_valid_mask()
#         missing = [
#             class_id for class_id in ids
#             if class_id >= valid.numel() or not bool(valid[class_id])
#         ]
#         if missing:
#             raise RuntimeError(
#                 f"GeometryBank rows are missing or invalid: {missing}"
#             )

#         geometry_bank.assert_response_prior_ready()
#         if require_frozen_prior and not bool(
#             geometry_bank.response_prior_frozen.item()
#         ):
#             raise RuntimeError("PC-STGB response prior is not frozen")

#         if require_statistics:
#             incomplete: Dict[str, List[int]] = {}
#             for name in ("energy_stats_ready", "margin_stats_ready"):
#                 state = getattr(geometry_bank, name, None)
#                 if not torch.is_tensor(state):
#                     incomplete[name] = list(ids)
#                     continue
#                 bad = [class_id for class_id in ids if not bool(state[class_id])]
#                 if bad:
#                     incomplete[name] = bad
#             if incomplete:
#                 raise RuntimeError(
#                     f"GeometryBank statistics are incomplete: {incomplete}"
#                 )

#         if response_required:
#             state = getattr(geometry_bank, "response_stats_ready", None)
#             if not torch.is_tensor(state):
#                 raise RuntimeError("GeometryBank lacks response_stats_ready")
#             bad = [class_id for class_id in ids if not bool(state[class_id])]
#             if bad:
#                 raise RuntimeError(
#                     f"Conditional tangent residual geometry is missing: {bad}"
#                 )

#         coupling = getattr(geometry_bank, "response_coupling_ready", None)
#         if coupling_required:
#             if not torch.is_tensor(coupling):
#                 raise RuntimeError("GeometryBank lacks response_coupling_ready")
#             ready = [bool(coupling[class_id]) for class_id in ids]
#             if require_any_joint_ready:
#                 if not any(ready):
#                     raise RuntimeError(
#                         "No occupancy--tangent coupled class row is ready"
#                     )
#             elif not all(ready):
#                 bad = [
#                     class_id for class_id, flag in zip(ids, ready) if not flag
#                 ]
#                 raise RuntimeError(
#                     f"Occupancy--tangent coupling is missing: {bad}"
#                 )

#         if require_frozen:
#             unfrozen = [
#                 class_id for class_id in ids
#                 if not bool(geometry_bank.frozen_class_mask[class_id])
#             ]
#             if unfrozen:
#                 raise RuntimeError(
#                     f"GeometryBank rows are not frozen: {unfrozen}"
#                 )

#         contract_digest = self._bank_contract_digest()
#         bound_digest = self._classifier_bound_digest()
#         classifier = self.model.classifier
#         if require_bound_contract or bool(
#             getattr(classifier, "require_bound_contract", False)
#         ):
#             if bound_digest is None:
#                 raise RuntimeError(
#                     "Classifier is not bound to the final GeometryBank contract"
#                 )
#             if bound_digest != contract_digest:
#                 raise RuntimeError(
#                     "Classifier and GeometryBank contract digests differ"
#                 )

#         if bank is not None:
#             canonical = self._canonicalize_bank(bank)
#             if canonical["means"].size(0) != len(geometry_bank):
#                 raise RuntimeError(
#                     "Supplied GeometryBank mapping is stale or has wrong row count"
#                 )
#             live_valid = geometry_bank.get_valid_mask().detach().cpu()
#             supplied_valid = canonical["valid_mask"].detach().cpu()
#             if not torch.equal(live_valid, supplied_valid):
#                 raise RuntimeError(
#                     "Supplied GeometryBank mapping does not match live valid rows"
#                 )

#     def assert_bank_has_only_allowed_valid_rows(
#         self,
#         bank: Optional[Mapping[str, Any]],
#         allowed_classes: Iterable[int],
#     ) -> None:
#         allowed = set(
#             self._as_class_list(allowed_classes, name="allowed_classes")
#         )
#         geometry_bank = self._geometry_bank_object()
#         valid_rows = torch.nonzero(
#             geometry_bank.get_valid_mask(), as_tuple=False
#         ).flatten().detach().cpu().tolist()
#         leaked = [
#             int(class_id) for class_id in valid_rows
#             if int(class_id) not in allowed
#         ]
#         if leaked:
#             raise RuntimeError(
#                 "GeometryBank contains valid rows outside the allowed set: "
#                 f"{leaked}"
#             )
#         if bank is not None:
#             canonical = self._canonicalize_bank(bank)
#             mapping_rows = torch.nonzero(
#                 canonical["valid_mask"].detach().cpu().bool().flatten(),
#                 as_tuple=False,
#             ).flatten().tolist()
#             if mapping_rows != valid_rows:
#                 raise RuntimeError(
#                     "Supplied GeometryBank mapping does not match live valid rows"
#                 )

#     def compute_bank_validity_diagnostics(
#         self,
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#     ) -> Dict[str, Any]:
#         geometry_bank = self._geometry_bank_object()
#         ids = (
#             self._as_class_list(seen_classes, name="seen_classes")
#             if seen_classes is not None
#             else torch.nonzero(
#                 geometry_bank.get_valid_mask(), as_tuple=False
#             ).flatten().tolist()
#         )
#         report = geometry_bank.assert_bank_valid(
#             ids if ids else None, strict=False
#         )
#         diagnostics = (
#             geometry_bank.compute_geometry_diagnostics(ids)
#             if ids and callable(
#                 getattr(geometry_bank, "compute_geometry_diagnostics", None)
#             )
#             else {}
#         )

#         def state(name: str) -> Dict[int, bool]:
#             tensor = getattr(geometry_bank, name, None)
#             if not torch.is_tensor(tensor):
#                 return {int(class_id): False for class_id in ids}
#             return {
#                 int(class_id): bool(tensor[class_id]) for class_id in ids
#             }

#         return {
#             "valid": bool(report.get("ok", False)),
#             "errors": list(report.get("errors", [])),
#             "method": self.METHOD_NAME,
#             "schema_version": int(
#                 getattr(geometry_bank, "SCHEMA_VERSION", -1)
#             ),
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "class_ids": ids,
#             "valid_rows": torch.nonzero(
#                 geometry_bank.get_valid_mask(), as_tuple=False
#             ).flatten().tolist(),
#             "energy_stats_ready": state("energy_stats_ready"),
#             "margin_stats_ready": state("margin_stats_ready"),
#             "response_stats_ready": state("response_stats_ready"),
#             "response_coupling_ready": state("response_coupling_ready"),
#             "frozen": state("frozen_class_mask"),
#             "response_prior_ready": bool(
#                 geometry_bank.response_prior_ready.item()
#             ),
#             "response_prior_frozen": bool(
#                 geometry_bank.response_prior_frozen.item()
#             ),
#             "geometry_bank_contract_digest": self._bank_contract_digest(),
#             "classifier_bound_contract_digest": self._classifier_bound_digest(),
#             "model_contract_digest": self._model_contract_digest(),
#             "geometry_diagnostics": diagnostics,
#         }

#     # ------------------------------------------------------------------
#     # Exact old-row and phase-contract immutability
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def snapshot_bank_rows(
#         self,
#         bank: Optional[Mapping[str, Any]],
#         class_ids: Iterable[int],
#     ) -> Dict[str, Any]:
#         del bank
#         ids = self._as_class_list(
#             class_ids, name="snapshot_class_ids", allow_empty=True
#         )
#         rows = (
#             self._geometry_bank_object().snapshot_rows(ids)
#             if ids
#             else {
#                 "class_ids": torch.empty(
#                     0, device=self.device, dtype=torch.long
#                 )
#             }
#         )
#         return {
#             "rows": rows,
#             "class_ids": list(ids),
#             "geometry_bank_contract_digest": self._bank_contract_digest(),
#             "classifier_bound_contract_digest": self._classifier_bound_digest(),
#             "model_contract_digest": self._model_contract_digest(),
#         }

#     @torch.no_grad()
#     def assert_bank_rows_unchanged(
#         self,
#         before: Mapping[str, Any],
#         after: Optional[Mapping[str, Any]],
#         class_ids: Iterable[int],
#         context: str,
#         *,
#         atol: float = 0.0,
#         check_frozen_mask: bool = True,
#     ) -> None:
#         del after, check_frozen_mask
#         ids = self._as_class_list(
#             class_ids, name=f"{context}.class_ids", allow_empty=True
#         )
#         if abs(float(atol)) > 0.0:
#             raise RuntimeError(
#                 f"{context}: old-row immutability requires exact equality"
#             )
#         if ids:
#             row_snapshot = before.get("rows", before)
#             method = getattr(
#                 self._geometry_bank_object(), "assert_rows_identical", None
#             )
#             if not callable(method):
#                 raise RuntimeError(
#                     "GeometryBank must expose assert_rows_identical()"
#                 )
#             method(row_snapshot, ids, context=context)

#         expected_bank = before.get("geometry_bank_contract_digest")
#         if expected_bank is not None and self._bank_contract_digest() != str(
#             expected_bank
#         ):
#             raise RuntimeError(
#                 f"{context}: phase-invariant GeometryBank contract changed"
#             )
#         expected_bound = before.get("classifier_bound_contract_digest")
#         if expected_bound != self._classifier_bound_digest():
#             raise RuntimeError(
#                 f"{context}: classifier bank binding changed"
#             )

#     def _old_bank_integrity_snapshot(
#         self,
#         old_class_ids: Iterable[int],
#     ) -> Dict[str, Any]:
#         ids = self._as_class_list(
#             old_class_ids, name="old_class_ids", allow_empty=True
#         )
#         return self.snapshot_bank_rows(None, ids)

#     def _assert_old_bank_integrity(
#         self,
#         old_class_ids: Iterable[int],
#         snapshot: Mapping[str, Any],
#         *,
#         context: str,
#         atol: float = 0.0,
#     ) -> None:
#         ids = self._as_class_list(
#             old_class_ids, name="old_class_ids", allow_empty=True
#         )
#         self.assert_bank_rows_unchanged(
#             snapshot, None, ids, context, atol=atol
#         )

#     # ------------------------------------------------------------------
#     # Shared conditional joint-energy and coupled-replay wrappers
#     # ------------------------------------------------------------------
#     def _joint_geometry_energy_from_bank(
#         self,
#         features: torch.Tensor,
#         spectral_responses: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         bank: Optional[Any] = None,
#         return_parts: bool = False,
#     ) -> Union[torch.Tensor, Dict[str, Any]]:
#         ids = self._as_class_list(seen_classes, name="seen_classes")
#         live_bank = self._geometry_bank_object()
#         source = live_bank if bank is None else bank
#         if not callable(getattr(source, "joint_energy_matrix", None)):
#             raise RuntimeError(
#                 "Main PC-STGB scoring requires a live GeometryBank object; "
#                 "a detached mapping cannot enforce the frozen contract"
#             )
#         expected_dim = int(getattr(live_bank, "feature_dim"))
#         self.assert_feature_tensor(
#             features,
#             expected_dim=expected_dim,
#             context="joint energy features",
#         )
#         self.assert_response_tensor(
#             spectral_responses,
#             batch_size=features.size(0),
#             num_interventions=int(self.model.num_interventions),
#             feature_dim=expected_dim,
#             context="joint energy tangents",
#         )
#         if features.device != spectral_responses.device:
#             raise RuntimeError(
#                 "Joint-energy features and tangents must share one device"
#             )
#         method = getattr(self.model, "compute_logits_from_features", None)
#         if not callable(method):
#             raise RuntimeError(
#                 "Model must expose compute_logits_from_features()"
#             )
#         result = method(
#             features,
#             spectral_responses=spectral_responses,
#             seen_classes=ids,
#             geometry_bank=source,
#             mode="pc_stgb",
#             return_energy=True,
#             return_parts=True,
#         )
#         if not isinstance(result, Mapping):
#             raise RuntimeError("Joint scoring must return a mapping")
#         output = dict(result)
#         energy = output.get("joint_energy", output.get("energy"))
#         if not torch.is_tensor(energy) or energy.shape != (
#             features.size(0), len(ids)
#         ):
#             raise RuntimeError(
#                 "Joint scoring returned an invalid energy matrix"
#             )
#         if not torch.isfinite(energy).all():
#             raise RuntimeError("Joint energy contains NaN/Inf")
#         if output.get("joint_factorization") != self.JOINT_FACTORIZATION:
#             raise RuntimeError(
#                 "Scoring did not use the conditional PC-STGB factorisation"
#             )
#         if output.get("uses_coupling_inference_score") is False:
#             raise RuntimeError(
#                 "Scoring explicitly reports that coupling was not used"
#             )
#         output.setdefault("joint_energy", energy)
#         output.setdefault("energy", energy)
#         output.setdefault("conditional_tangent_energy", output.get("response_energy"))
#         output["joint_factorization"] = self.JOINT_FACTORIZATION
#         return output if return_parts else energy

#     def _geometry_energy_from_bank(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         spectral_responses: Optional[torch.Tensor] = None,
#         bank: Optional[Any] = None,
#         return_parts: bool = False,
#         feature_only_ablation: bool = False,
#     ) -> Union[torch.Tensor, Dict[str, Any]]:
#         """Compatibility wrapper with an explicit occupancy-only ablation."""
#         if spectral_responses is None:
#             if not feature_only_ablation:
#                 raise RuntimeError(
#                     "Main PC-STGB scoring requires spectral_responses [B,K,D]. "
#                     "Set feature_only_ablation=True only for a declared ablation."
#                 )
#             ids = self._as_class_list(seen_classes, name="seen_classes")
#             geometry_bank = self._geometry_bank_object() if bank is None else bank
#             method = getattr(geometry_bank, "geometry_energy_matrix", None)
#             if not callable(method):
#                 raise RuntimeError(
#                     "Occupancy-only ablation requires "
#                     "GeometryBank.geometry_energy_matrix()"
#                 )
#             return method(
#                 features,
#                 ids,
#                 normalize_by_dim=True,
#                 return_parts=return_parts,
#             )
#         return self._joint_geometry_energy_from_bank(
#             features,
#             spectral_responses,
#             seen_classes=seen_classes,
#             bank=bank,
#             return_parts=return_parts,
#         )

#     def _dual_geometry_energy_matrix(
#         self,
#         features: torch.Tensor,
#         bank: Optional[Any],
#         *,
#         spectral_responses: Optional[torch.Tensor] = None,
#         seen_classes: Optional[Iterable[int]] = None,
#         return_parts: bool = False,
#         **_: Any,
#     ) -> Union[torch.Tensor, Dict[str, Any]]:
#         if spectral_responses is None:
#             raise RuntimeError(
#                 "_dual_geometry_energy_matrix requires conditional tangents"
#             )
#         ids = (
#             self._as_class_list(seen_classes, name="seen_classes")
#             if seen_classes is not None
#             else torch.nonzero(
#                 self._geometry_bank_object().get_valid_mask(),
#                 as_tuple=False,
#             ).flatten().tolist()
#         )
#         return self._joint_geometry_energy_from_bank(
#             features,
#             spectral_responses,
#             seen_classes=ids,
#             bank=bank,
#             return_parts=return_parts,
#         )

#     def score_candidate_rows(
#         self,
#         features: torch.Tensor,
#         spectral_responses: torch.Tensor,
#         *,
#         old_class_ids: Iterable[int],
#         candidate_rows: Mapping[int, Mapping[str, Any]],
#         candidate_class_ids: Optional[Iterable[int]] = None,
#         return_parts: bool = True,
#     ) -> Dict[str, Any]:
#         """Differentiably score candidates and prove the bank was not mutated."""
#         old_ids = self._as_class_list(
#             old_class_ids, name="old_class_ids", allow_empty=True
#         )
#         candidate_ids = (
#             [int(class_id) for class_id in candidate_rows]
#             if candidate_class_ids is None
#             else self._as_class_list(
#                 candidate_class_ids, name="candidate_class_ids"
#             )
#         )
#         if set(candidate_ids) != set(int(key) for key in candidate_rows):
#             raise RuntimeError(
#                 "candidate_rows do not match candidate_class_ids"
#             )
#         required_candidate = (
#             "response_stats_ready",
#             "response_coupling",
#             "response_coupling_reliability",
#             "response_coupling_explained_variance",
#             "response_coupling_ready",
#         )
#         for class_id, row in candidate_rows.items():
#             missing = [name for name in required_candidate if row.get(name) is None]
#             if missing:
#                 raise RuntimeError(
#                     f"candidate class {class_id} lacks conditional fields: {missing}"
#                 )
#             if not bool(torch.as_tensor(row["response_stats_ready"]).item()):
#                 raise RuntimeError(
#                     f"candidate class {class_id} lacks tangent residual geometry"
#                 )
#             if not bool(torch.as_tensor(row["response_coupling_ready"]).item()):
#                 raise RuntimeError(
#                     f"candidate class {class_id} lacks occupancy--tangent coupling"
#                 )

#         before = self._old_bank_integrity_snapshot(old_ids)
#         method = getattr(self.model, "score_candidate_geometry_rows", None)
#         if not callable(method):
#             raise RuntimeError(
#                 "Model must expose score_candidate_geometry_rows()"
#             )
#         result = method(
#             features,
#             spectral_responses,
#             old_class_ids=old_ids,
#             candidate_rows=candidate_rows,
#             candidate_class_ids=candidate_ids,
#             return_parts=return_parts,
#         )
#         if not isinstance(result, Mapping):
#             raise RuntimeError("Candidate scoring must return a mapping")
#         output = dict(result)
#         if output.get("joint_factorization") != self.JOINT_FACTORIZATION:
#             raise RuntimeError(
#                 "Candidate scoring used the wrong joint factorisation"
#             )
#         if output.get("uses_coupling_inference_score") is not True:
#             raise RuntimeError(
#                 "Candidate scoring did not certify coupling-aware inference"
#             )
#         if output.get("uses_independent_response_factorization") is not False:
#             raise RuntimeError(
#                 "Candidate scoring did not reject the independence model"
#             )
#         if output.get("bank_mutated") is not False:
#             raise RuntimeError("Candidate scoring reports bank mutation")
#         self._assert_old_bank_integrity(
#             old_ids,
#             before,
#             context="candidate scoring immutability",
#             atol=0.0,
#         )
#         return output

#     @torch.no_grad()
#     def sample_coupled_geometry_replay(
#         self,
#         class_ids: Iterable[int],
#         samples_per_class: Union[int, Mapping[int, int]] = 16,
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         **kwargs: Any,
#     ) -> Dict[str, torch.Tensor]:
#         """Sample occupancy first and tangent conditionally from the same row."""
#         ids = self._as_class_list(class_ids, name="replay_class_ids")
#         self.assert_bank_ready_for_seen_classes(
#             None,
#             ids,
#             require_statistics=False,
#             require_response=True,
#             require_joint_state=True,
#             require_frozen=True,
#             require_frozen_prior=True,
#             require_bound_contract=True,
#         )
#         method = getattr(self.model, "sample_geometry_replay", None)
#         if not callable(method):
#             raise RuntimeError("Model must expose sample_geometry_replay()")
#         replay = method(
#             ids,
#             samples_per_class=samples_per_class,
#             seen_classes=(
#                 ids
#                 if seen_classes is None
#                 else self._as_class_list(
#                     seen_classes, name="replay_seen_classes"
#                 )
#             ),
#             **kwargs,
#         )
#         if not isinstance(replay, Mapping):
#             raise RuntimeError("Coupled replay must return a mapping")
#         output = dict(replay)
#         features = output.get("features")
#         responses = output.get(
#             "spectral_responses", output.get("responses")
#         )
#         labels = output.get("global_labels")
#         self.assert_feature_tensor(
#             features,
#             expected_dim=int(self.model.d_model),
#             context="coupled replay features",
#         )
#         self.assert_response_tensor(
#             responses,
#             batch_size=features.size(0),
#             num_interventions=int(self.model.num_interventions),
#             feature_dim=int(self.model.d_model),
#             context="coupled replay tangents",
#         )
#         if not torch.is_tensor(labels) or labels.numel() != features.size(0):
#             raise RuntimeError("Coupled replay labels are missing or misaligned")
#         self.assert_global_labels_in_set(
#             labels, ids, "coupled replay labels"
#         )
#         output["spectral_responses"] = responses
#         output["responses"] = responses
#         output["joint_factorization"] = self.JOINT_FACTORIZATION
#         output["replay_factorization"] = "z_then_g_given_z_c"
#         return output

#     # Compatibility name used by some incremental trainers.
#     sample_geometry_replay = sample_coupled_geometry_replay

#     # ------------------------------------------------------------------
#     # Phase class order and architecture contract
#     # ------------------------------------------------------------------
#     def resolve_phase_classes(self, phase: int) -> Tuple[List[int], List[int], List[int]]:
#         phase = int(phase)
#         if phase <= 0:
#             raise ValueError("Incremental phase must be greater than zero")
#         phase_map = getattr(self.dataset, "phase_to_classes", None)
#         if phase_map is None:
#             raise RuntimeError("dataset.phase_to_classes is required")

#         def at(index: int) -> List[int]:
#             try:
#                 values = phase_map[index]
#             except (KeyError, IndexError, TypeError) as error:
#                 raise RuntimeError(f"dataset.phase_to_classes has no phase {index}") from error
#             return self._as_class_list(values, name=f"phase_{index}_classes")

#         new_classes = at(phase)
#         if callable(getattr(self.dataset, "get_classes_up_to_phase", None)):
#             old_classes = self._as_class_list(
#                 self.dataset.get_classes_up_to_phase(phase - 1),
#                 name="old_classes",
#             )
#         else:
#             flattened: List[int] = []
#             for previous in range(phase):
#                 flattened.extend(at(previous))
#             old_classes = self._as_class_list(flattened, name="old_classes")
#         overlap = sorted(set(old_classes).intersection(new_classes))
#         if overlap:
#             raise RuntimeError(f"phase {phase} old/new class overlap: {overlap}")
#         return old_classes, new_classes, [*old_classes, *new_classes]

#     def _seen_class_ids_before_phase(self, phase: int) -> List[int]:
#         values: List[int] = []
#         for index in range(max(int(phase), 0)):
#             values.extend(int(value) for value in self.dataset.phase_to_classes[index])
#         return self._as_class_list(values, name="seen_before_phase", allow_empty=True)

#     def _seen_class_ids_through_phase(self, phase: int) -> List[int]:
#         values: List[int] = []
#         for index in range(max(int(phase), 0) + 1):
#             values.extend(int(value) for value in self.dataset.phase_to_classes[index])
#         return self._as_class_list(values, name="seen_through_phase")

#     def assert_clean_incremental_contract(
#         self,
#         phase: int,
#         old_classes: Iterable[int],
#         new_classes: Iterable[int],
#         seen_classes: Iterable[int],
#         *,
#         context: str = "incremental_contract",
#     ) -> None:
#         phase = int(phase)
#         old_ids = self._as_class_list(
#             old_classes, name=f"{context}.old_classes"
#         )
#         new_ids = self._as_class_list(
#             new_classes, name=f"{context}.new_classes"
#         )
#         seen_ids = self._as_class_list(
#             seen_classes, name=f"{context}.seen_classes"
#         )
#         if phase <= 0:
#             raise RuntimeError(f"{context}: phase must be > 0")
#         if set(old_ids) & set(new_ids):
#             raise RuntimeError(f"{context}: old/new classes overlap")
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError(
#                 f"{context}: seen_classes must preserve exact old+new order"
#             )

#         configured_mode = str(
#             getattr(
#                 getattr(self, "args", None),
#                 "incremental_update_mode",
#                 "pc_stgb_row_replay",
#             )
#         ).strip().lower().replace("-", "_")
#         if configured_mode == "pc_sirg_row_replay":
#             configured_mode = "pc_stgb_row_replay"
#         model_mode = str(
#             getattr(self.model, "incremental_update_mode", configured_mode)
#         ).strip().lower().replace("-", "_")
#         if model_mode == "pc_sirg_row_replay":
#             model_mode = "pc_stgb_row_replay"
#         if (
#             configured_mode != "pc_stgb_row_replay"
#             or model_mode != "pc_stgb_row_replay"
#         ):
#             raise RuntimeError(
#                 f"{context}: args/model must use pc_stgb_row_replay; "
#                 f"args={configured_mode!r}, model={model_mode!r}"
#             )

#         forbidden = (
#             "use_geometry_transport",
#             "use_sglat_transport",
#             "allow_old_model_transport",
#             "use_energy_calibrator",
#             "use_adaptive_boundary",
#             "use_incremental_adapter",
#             "use_geometry_gated_adapter",
#             "allow_incremental_projection_training",
#             "normalize_geometry_features",
#             "use_raw_spectral_gaussian",
#             # This legacy switch denotes the retired raw-spectrum coupling,
#             # not the mandatory occupancy--tangent coupling stored by PC-STGB.
#             "use_spectral_feature_coupling",
#         )
#         active = [name for name in forbidden if self._cfg_bool(name, False)]
#         if active:
#             raise RuntimeError(
#                 f"{context}: forbidden architecture switches are active: {active}"
#             )

#         method = getattr(self.model, "feature_contract", None)
#         if not callable(method):
#             raise RuntimeError(
#                 f"{context}: model must expose feature_contract()"
#             )
#         contract = method()
#         expected = {
#             "method": "PC-STGB",
#             "geometry_bank_schema_version": self.BANK_SCHEMA_VERSION,
#             "geometry_feature_space":
#                 "unnormalized_euclidean_residual_projected_z",
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "spectral_object":
#                 "central_finite_difference_canonical_feature_tangent",
#             "spectral_role":
#                 "conditional_training_coupled_replay_and_inference",
#             "inference_geometry":
#                 "occupancy_conditioned_spectral_tangent_geometry",
#             "raw_spectral_gaussian": False,
#             "occupancy_tangent_coupling": True,
#             "independent_response_factorization": False,
#             "old_rows_immutable": True,
#             "feature_normalization": False,
#             "response_prior_ready": True,
#             "response_prior_frozen": True,
#         }
#         failures = [
#             f"{key}={contract.get(key)!r}, expected {value!r}"
#             for key, value in expected.items()
#             if contract.get(key) != value
#         ]
#         if failures:
#             raise RuntimeError(
#                 f"{context}: PC-STGB contract mismatch: "
#                 + "; ".join(failures)
#             )
#         if int(contract.get("num_interventions", -1)) != int(
#             self.model.num_interventions
#         ):
#             raise RuntimeError(f"{context}: intervention count mismatch")
#         if int(contract.get("intervention_definition_version", -1)) != int(
#             self.model.intervention_definition_version
#         ):
#             raise RuntimeError(
#                 f"{context}: intervention definition version mismatch"
#             )

#         live_valid = torch.nonzero(
#             self._geometry_bank_object().get_valid_mask(), as_tuple=False
#         ).flatten().detach().cpu().tolist()
#         if set(live_valid) != set(old_ids):
#             raise RuntimeError(
#                 f"{context}: committed rows {live_valid} must contain exactly "
#                 f"the old classes {old_ids} before candidate admission"
#             )
#         self.assert_bank_ready_for_seen_classes(
#             self._safe_get_subspace_bank(require_ready=True),
#             old_ids,
#             require_statistics=True,
#             require_response=True,
#             require_joint_state=True,
#             require_frozen=True,
#             require_frozen_prior=True,
#             require_bound_contract=True,
#         )
#         self.assert_bank_has_only_allowed_valid_rows(None, old_ids)


#     # ------------------------------------------------------------------
#     # Atomic current-phase conditional-row commit
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def commit_new_class_rows_only(
#         self,
#         class_ids: Iterable[int],
#         candidate_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         *,
#         features: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         spectral_responses: Optional[torch.Tensor] = None,
#         phase_created: Optional[int] = None,
#         freeze: bool = True,
#         context: str = "commit_new_class_rows_only",
#         **legacy: Any,
#     ) -> Dict[str, Any]:
#         ids = self._as_class_list(
#             class_ids, name=f"{context}.class_ids", allow_empty=True
#         )
#         if not ids:
#             return {
#                 "active": 0,
#                 "committed_class_ids": [],
#                 "joint_factorization": self.JOINT_FACTORIZATION,
#             }
#         phase = int(getattr(self.model, "current_phase", 0))
#         if phase <= 0:
#             raise RuntimeError(
#                 f"{context}: new-row commit is incremental-phase only"
#             )
#         old_ids = self._seen_class_ids_before_phase(phase)
#         allowed_new = self._as_class_list(
#             self.dataset.phase_to_classes[phase],
#             name=f"{context}.allowed_new",
#         )
#         if set(ids) & set(old_ids):
#             raise RuntimeError(f"{context}: refusing to overwrite old rows")
#         invalid = [class_id for class_id in ids if class_id not in allowed_new]
#         if invalid:
#             raise RuntimeError(
#                 f"{context}: rows are not current-phase classes: {invalid}"
#             )
#         if legacy.get("spectral_summary") is not None or legacy.get(
#             "spectral_summary_is_physical"
#         ):
#             raise RuntimeError(
#                 f"{context}: raw spectral summaries are retired"
#             )
#         if not bool(freeze):
#             raise RuntimeError(
#                 f"{context}: incremental rows must freeze at atomic commitment"
#             )

#         self.assert_clean_incremental_contract(
#             phase,
#             old_ids,
#             allowed_new,
#             [*old_ids, *allowed_new],
#             context=f"{context}.precommit",
#         )
#         if candidate_rows is None:
#             if features is None or labels is None or spectral_responses is None:
#                 raise RuntimeError(
#                     f"{context}: pass complete candidate_rows, or "
#                     "features + labels + spectral_responses"
#                 )
#             build = getattr(
#                 self.model, "build_candidate_geometry_rows", None
#             )
#             if not callable(build):
#                 raise RuntimeError(
#                     "Model must expose build_candidate_geometry_rows()"
#                 )
#             candidate_rows = build(
#                 ids,
#                 features,
#                 labels,
#                 spectral_responses=spectral_responses,
#             )
#         if not isinstance(candidate_rows, Mapping):
#             raise TypeError(f"{context}: candidate_rows must be a mapping")
#         if set(int(key) for key in candidate_rows) != set(ids):
#             raise RuntimeError(
#                 f"{context}: candidate row IDs do not match current-phase IDs"
#             )
#         for class_id, row in candidate_rows.items():
#             if not bool(torch.as_tensor(row.get("response_stats_ready", False)).item()):
#                 raise RuntimeError(
#                     f"{context}: class {class_id} lacks tangent residual geometry"
#                 )
#             if not bool(torch.as_tensor(row.get("response_coupling_ready", False)).item()):
#                 raise RuntimeError(
#                     f"{context}: class {class_id} lacks occupancy--tangent coupling"
#                 )

#         old_snapshot = self._old_bank_integrity_snapshot(old_ids)
#         geometry_bank = self._geometry_bank_object()
#         old_digest = geometry_bank.rows_digest(old_ids)
#         contract_digest = self._bank_contract_digest()
#         if self._classifier_bound_digest() != contract_digest:
#             raise RuntimeError(
#                 f"{context}: classifier is not bound to the live bank contract"
#             )
#         commit = getattr(
#             self.model, "commit_candidate_geometry_rows", None
#         )
#         if not callable(commit):
#             raise RuntimeError(
#                 "Model must expose commit_candidate_geometry_rows()"
#             )
#         result = commit(
#             candidate_rows,
#             old_class_ids=old_ids,
#             expected_class_ids=ids,
#             phase_created=(
#                 phase if phase_created is None else int(phase_created)
#             ),
#             freeze=True,
#             expected_old_digest=old_digest,
#             expected_contract_digest=contract_digest,
#             context=context,
#         )
#         self._assert_old_bank_integrity(
#             old_ids,
#             old_snapshot,
#             context=f"{context}: old conditional-row immutability",
#             atol=0.0,
#         )
#         if self._bank_contract_digest() != contract_digest:
#             raise RuntimeError(
#                 f"{context}: phase-invariant bank contract changed"
#             )
#         if self._classifier_bound_digest() != contract_digest:
#             raise RuntimeError(
#                 f"{context}: classifier binding changed during commit"
#             )
#         self.assert_bank_ready_for_seen_classes(
#             None,
#             ids,
#             require_statistics=False,
#             require_response=True,
#             require_joint_state=True,
#             require_frozen=True,
#             require_frozen_prior=True,
#             require_bound_contract=True,
#         )
#         output = dict(result)
#         output.update(
#             {
#                 "committed_class_ids": ids,
#                 "old_class_ids": old_ids,
#                 "joint_factorization": self.JOINT_FACTORIZATION,
#                 "geometry_bank_contract_digest": contract_digest,
#                 "classifier_bound_contract_digest":
#                     self._classifier_bound_digest(),
#                 "model_contract_digest": self._model_contract_digest(),
#             }
#         )
#         return output

#     def _commit_refined_feature_rows(self, *args: Any, **kwargs: Any) -> None:
#         del args, kwargs
#         raise RuntimeError(
#             "Feature-only row commit is retired. Rebuild the complete "
#             "occupancy-conditioned tangent row and call "
#             "commit_new_class_rows_only()."
#         )

#     # ------------------------------------------------------------------
#     # Phase certificate and JSON persistence
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def geometry_phase_certificate(
#         self,
#         phase: int,
#         class_ids: Iterable[int],
#         *,
#         require_statistics: bool = True,
#         require_spectral: bool = True,
#         require_response: Optional[bool] = None,
#         require_any_joint_ready: bool = False,
#         require_frozen: bool = False,
#     ) -> Dict[str, Any]:
#         ids = self._as_class_list(
#             class_ids, name="certificate_class_ids"
#         )
#         response_required = bool(
#             require_spectral if require_response is None else require_response
#         )
#         geometry_bank = self._geometry_bank_object()
#         state = geometry_bank.phase_geometry_state_report(
#             ids,
#             freeze=False,
#             require_statistics=require_statistics,
#             require_response=response_required,
#             context=f"phase_{int(phase)}_pc_stgb_certificate",
#         )
#         errors = list(state.get("errors", []))
#         index = torch.tensor(
#             ids, device=geometry_bank.device, dtype=torch.long
#         )
#         response_ready = geometry_bank.response_stats_ready.index_select(
#             0, index
#         )
#         coupling_ready = geometry_bank.response_coupling_ready.index_select(
#             0, index
#         )
#         if response_required and not bool(response_ready.all().item()):
#             bad = [
#                 class_id for class_id, flag in zip(
#                     ids, response_ready.detach().cpu().tolist()
#                 ) if not flag
#             ]
#             errors.append(f"missing tangent residual rows: {bad}")
#         if require_any_joint_ready:
#             if not bool(coupling_ready.any().item()):
#                 errors.append("no occupancy--tangent coupled row is active")
#         elif not bool(coupling_ready.all().item()):
#             bad = [
#                 class_id for class_id, flag in zip(
#                     ids, coupling_ready.detach().cpu().tolist()
#                 ) if not flag
#             ]
#             errors.append(f"missing occupancy--tangent coupling rows: {bad}")
#         if not bool(geometry_bank.response_prior_ready.item()):
#             errors.append("response prior is absent")
#         if not bool(geometry_bank.response_prior_frozen.item()):
#             errors.append("response prior is not frozen")
#         if require_frozen:
#             unfrozen = [
#                 class_id for class_id in ids
#                 if not bool(geometry_bank.frozen_class_mask[class_id])
#             ]
#             if unfrozen:
#                 errors.append(f"unfrozen classes: {unfrozen}")

#         bank_digest = self._bank_contract_digest()
#         bound_digest = self._classifier_bound_digest()
#         if bound_digest != bank_digest:
#             errors.append(
#                 "classifier is not bound to the current GeometryBank contract"
#             )
#         diagnostics = (
#             geometry_bank.compute_geometry_diagnostics(ids)
#             if callable(
#                 getattr(geometry_bank, "compute_geometry_diagnostics", None)
#             )
#             else {}
#         )
#         state.update(
#             {
#                 "phase": int(phase),
#                 "method": self.METHOD_NAME,
#                 "schema_version": self.BANK_SCHEMA_VERSION,
#                 "joint_factorization": self.JOINT_FACTORIZATION,
#                 "errors": errors,
#                 "ok": not errors and bool(state.get("ok", True)),
#                 "feature_logdet_weight": float(
#                     geometry_bank.energy_logdet_weight
#                 ),
#                 "response_logdet_weight": float(
#                     getattr(
#                         self.model.classifier,
#                         "response_logdet_weight",
#                         1.0,
#                     )
#                 ),
#                 "response_weight": float(geometry_bank.response_weight),
#                 "normalize_by_dimension": bool(
#                     getattr(
#                         self.model.classifier,
#                         "normalize_energy_by_dim",
#                         True,
#                     )
#                 ),
#                 "inference_uses_conditional_tangent_score": True,
#                 "uses_occupancy_tangent_coupling": True,
#                 "uses_independent_response_factorization": False,
#                 "raw_spectral_gaussian": False,
#                 "old_rows_immutable": True,
#                 "response_ready_rate": float(
#                     response_ready.float().mean().item()
#                 ),
#                 "response_ready_count": int(response_ready.sum().item()),
#                 "coupling_ready_rate": float(
#                     coupling_ready.float().mean().item()
#                 ),
#                 "coupling_ready_count": int(coupling_ready.sum().item()),
#                 "response_prior_ready": bool(
#                     geometry_bank.response_prior_ready.item()
#                 ),
#                 "response_prior_frozen": bool(
#                     geometry_bank.response_prior_frozen.item()
#                 ),
#                 "geometry_bank_contract_digest": bank_digest,
#                 "classifier_bound_contract_digest": bound_digest,
#                 "classifier_contract_digest":
#                     self.model.classifier.classifier_contract_digest(),
#                 "model_contract_digest": self._model_contract_digest(),
#                 "geometry_diagnostics": diagnostics,
#             }
#         )
#         return state


#     @staticmethod
#     def _json_safe(value: Any) -> Any:
#         if torch.is_tensor(value):
#             tensor = value.detach().cpu()
#             return tensor.item() if tensor.numel() == 1 else tensor.tolist()
#         if isinstance(value, Mapping):
#             return {str(key): TrainerHelper._json_safe(item) for key, item in value.items()}
#         if isinstance(value, (tuple, list)):
#             return [TrainerHelper._json_safe(item) for item in value]
#         if isinstance(value, set):
#             return [TrainerHelper._json_safe(item) for item in sorted(value, key=str)]
#         if isinstance(value, (str, int, float, bool)) or value is None:
#             return value
#         return str(value)

#     def save_json_diagnostics(self, path: str, data: Mapping[str, Any]) -> None:
#         directory = os.path.dirname(path)
#         if directory:
#             os.makedirs(directory, exist_ok=True)
#         temporary = f"{path}.tmp"
#         with open(temporary, "w", encoding="utf-8") as stream:
#             json.dump(self._json_safe(dict(data)), stream, indent=2)
#             stream.flush()
#             os.fsync(stream.fileno())
#         os.replace(temporary, path)









# from __future__ import annotations
# import json
# import os
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# import torch
# import torch.nn.functional as F

# from losses.loss import geometry_energy_matrix


# class TrainerHelper:
#     """Shared invariants for strict joint-geometry NECIL-HSI.

#         This helper owns protocol enforcement, not model novelty. It guarantees:
#           * global dataset labels and explicit seen-local classifier columns;
#           * one unnormalized canonical Euclidean feature space;
#           * one feature-only low-rank Gaussian inference energy;
#           * complete low-rank physical spectral geometry in every persistent row;
#           * an explicit joint-ready or feature-only-fallback coupling state;
#           * bitwise immutability of all old feature, spectral, and coupling fields;
#           * atomic insertion of complete current-phase joint rows.

#         It never stores samples, invents sample counts, performs geometry transport,
#         calibrates logits, or commits feature-only rows. Physical spectra are side
#         metadata for geometry estimation and diagnostics; they never alter logits.
#         """

#     FEATURE_ROW_FIELDS = (
#         "means", "bases", "eigvals", "res_vars", "active_ranks",
#         "sample_counts", "reliability", "feature_reliability",
#         "captured_energy", "condition_reliability", "noise_floors",
#     )
#     SPECTRAL_ROW_FIELDS = (
#         "spectral_prototypes", "spectral_diag_vars", "spectral_bases",
#         "spectral_eigvals", "spectral_res_vars", "spectral_active_ranks",
#         "spectral_reliability", "spectral_stats_ready",
#     )
#     COUPLING_ROW_FIELDS = (
#         "coupling_left", "coupling_right", "coupling_corrs",
#         "coupling_active_ranks", "coupling_reliability",
#         "coupling_stability", "coupling_stats_ready",
#         "coupling_fallback_mask",
#     )
#     STATISTIC_ROW_FIELDS = (
#         "energy_quantiles", "tail_energy_quantiles", "margin_quantiles",
#         "energy_stats_ready", "tail_stats_ready", "margin_stats_ready",
#         "bootstrap_center_uncertainty", "bootstrap_center_uncertainty_ratio",
#         "bootstrap_subspace_instability", "bootstrap_rank_stability",
#         "bootstrap_stats_ready", "phase_created", "frozen_class_mask",
#     )

#     # ------------------------------------------------------------------
#     # Generic utilities
#     # ------------------------------------------------------------------
#     def _zero(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if torch.is_tensor(ref):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     @staticmethod
#     def _as_class_list(class_ids: Iterable[int]) -> List[int]:
#         result: List[int] = []
#         seen = set()
#         for value in class_ids:
#             class_id = int(value)
#             if class_id < 0:
#                 raise RuntimeError(f"class IDs must be non-negative, got {class_id}")
#             if class_id not in seen:
#                 result.append(class_id)
#                 seen.add(class_id)
#         return result

#     def _cfg_bool(self, name: str, default: bool = False) -> bool:
#         args = getattr(self, "args", None)
#         value = getattr(self, name, getattr(args, name, default) if args is not None else default)
#         if value is None:
#             return bool(default)
#         if isinstance(value, bool):
#             return value
#         if isinstance(value, int) and value in {0, 1}:
#             return bool(value)
#         if isinstance(value, str):
#             token = value.strip().lower()
#             if token in {"1", "true", "yes", "y", "on"}:
#                 return True
#             if token in {"0", "false", "no", "n", "off", "none", "null", ""}:
#                 return False
#         raise RuntimeError(f"{name} must be an explicit boolean, got {value!r}")

#     def _cfg_float(self, name: str, default: float) -> float:
#         args = getattr(self, "args", None)
#         value = getattr(self, name, getattr(args, name, default) if args is not None else default)
#         value = default if value is None else value
#         result = float(value)
#         if not torch.isfinite(torch.tensor(result)):
#             raise RuntimeError(f"{name} must be finite, got {value!r}")
#         return result

#     def _cfg_int(self, name: str, default: int) -> int:
#         args = getattr(self, "args", None)
#         value = getattr(self, name, getattr(args, name, default) if args is not None else default)
#         value = default if value is None else value
#         if isinstance(value, bool):
#             raise RuntimeError(f"{name} must be an integer, not bool")
#         result = int(value)
#         if float(value) != float(result):
#             raise RuntimeError(f"{name} must be an integer, got {value!r}")
#         return result

#     @staticmethod
#     def _unpack_hsi_batch(batch: Any) -> Tuple[Any, Any, Any, Any]:
#         """Return patches, global labels, raw center spectra, and coordinates.

#         The spectral tensor must be the wavelength-ordered physical center-pixel
#         spectrum produced by the dataset. PCA channels or patch means are not a
#         valid replacement.
#         """
#         if isinstance(batch, Mapping):
#             x = batch.get("image", batch.get("patch", batch.get("patches")))
#             y = batch.get("label", batch.get("labels", batch.get("target")))
#             spectra = batch.get(
#                 "physical_center_spectrum",
#                 batch.get(
#                     "center_spectrum",
#                     batch.get("spectrum", batch.get("spectra")),
#                 ),
#             )
#             coords = batch.get("coord", batch.get("coords", batch.get("coordinate")))
#             if x is None or y is None:
#                 raise RuntimeError(
#                     "Batch mapping must contain image/patch/patches and label/labels/target"
#                 )
#             return x, y, spectra, coords
#         if isinstance(batch, (tuple, list)):
#             if len(batch) < 2:
#                 raise RuntimeError("Batch tuple/list must contain at least input and label")
#             return (
#                 batch[0],
#                 batch[1],
#                 batch[2] if len(batch) > 2 else None,
#                 batch[3] if len(batch) > 3 else None,
#             )
#         raise RuntimeError(f"Unsupported batch type: {type(batch)}")

#     def _stable_ce(
#         self,
#         logits: torch.Tensor,
#         labels_local: torch.Tensor,
#         *,
#         class_balanced: bool = False,
#     ) -> torch.Tensor:
#         if not torch.is_tensor(logits) or logits.numel() == 0:
#             return self._zero(logits if torch.is_tensor(logits) else None)
#         if logits.dim() != 2 or not torch.isfinite(logits).all():
#             raise RuntimeError("CE logits must be a finite [B,C] tensor")
#         labels = labels_local.to(device=logits.device, dtype=torch.long).flatten()
#         if labels.numel() != logits.size(0):
#             raise RuntimeError("CE labels/logits batch mismatch")
#         self.assert_valid_ce_targets(labels, logits.size(1), "stable_ce")
#         if abs(self._cfg_float("ce_logit_clip", 0.0)) > 1e-12:
#             raise RuntimeError(
#                 "ce_logit_clip must be zero; clipping changes the geometry-logit field"
#             )
#         smoothing = self._cfg_float("label_smoothing", 0.0)
#         if not 0.0 <= smoothing < 1.0:
#             raise RuntimeError("label_smoothing must lie in [0,1)")
#         per_sample = F.cross_entropy(
#             logits,
#             labels,
#             label_smoothing=smoothing,
#             reduction="none",
#         )
#         if not class_balanced:
#             return per_sample.mean()
#         terms = [
#             per_sample[labels == cls].mean()
#             for cls in torch.unique(labels, sorted=True)
#         ]
#         return torch.stack(terms).mean() if terms else logits.sum() * 0.0

#     # ------------------------------------------------------------------
#     # Strict global <-> seen-local label contract
#     # ------------------------------------------------------------------
#     def _classes_tensor(
#         self, class_ids: Iterable[int], *, device: Optional[torch.device] = None
#     ) -> torch.Tensor:
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             raise RuntimeError("class_ids must be non-empty")
#         return torch.tensor(
#             ids,
#             device=self.device if device is None else device,
#             dtype=torch.long,
#         )

#     def assert_global_labels_in_set(
#         self,
#         labels_global: torch.Tensor,
#         allowed_classes: Iterable[int],
#         context: str,
#     ) -> None:
#         if not torch.is_tensor(labels_global):
#             raise RuntimeError(f"{context}: labels must be a tensor")
#         labels = labels_global.long().flatten()
#         if labels.numel() == 0:
#             raise RuntimeError(f"{context}: labels are empty")
#         allowed = self._classes_tensor(allowed_classes, device=labels.device)
#         valid = torch.zeros_like(labels, dtype=torch.bool)
#         for class_id in allowed:
#             valid |= labels.eq(class_id)
#         if not bool(valid.all()):
#             bad = torch.unique(labels[~valid]).detach().cpu().tolist()
#             raise RuntimeError(
#                 f"{context}: labels outside allowed classes; bad={bad}, "
#                 f"allowed={allowed.detach().cpu().tolist()}"
#             )

#     @staticmethod
#     def assert_valid_ce_targets(
#         labels_local: torch.Tensor, num_classes: int, context: str
#     ) -> None:
#         if not torch.is_tensor(labels_local):
#             raise RuntimeError(f"{context}: targets must be a tensor")
#         labels = labels_local.long().flatten()
#         if labels.numel() == 0:
#             raise RuntimeError(f"{context}: targets are empty")
#         if int(labels.min()) < 0 or int(labels.max()) >= int(num_classes):
#             raise RuntimeError(
#                 f"{context}: targets [{int(labels.min())},{int(labels.max())}] "
#                 f"outside [0,{int(num_classes)-1}]"
#             )

#     def global_to_seen_local(
#         self,
#         labels_global: torch.Tensor,
#         seen_classes: Iterable[int],
#         *,
#         context: str = "global_to_seen_local",
#     ) -> torch.Tensor:
#         labels = labels_global.long().flatten()
#         seen = self._as_class_list(seen_classes)
#         self.assert_global_labels_in_set(labels, seen, context)
#         output = torch.full_like(labels, -1)
#         for local_id, global_id in enumerate(seen):
#             output[labels == global_id] = local_id
#         self.assert_valid_ce_targets(output, len(seen), context)
#         return output

#     def seen_local_to_global(
#         self,
#         predictions_local: torch.Tensor,
#         seen_classes: Iterable[int],
#         *,
#         context: str = "seen_local_to_global",
#     ) -> torch.Tensor:
#         predictions = predictions_local.long().flatten()
#         seen = self._classes_tensor(seen_classes, device=predictions.device)
#         if predictions.numel() == 0:
#             return predictions
#         self.assert_valid_ce_targets(predictions, int(seen.numel()), context)
#         return seen.index_select(0, predictions)

#     def global_to_phase_local(
#         self,
#         labels_global: torch.Tensor,
#         phase_classes: Iterable[int],
#         *,
#         context: str = "global_to_phase_local",
#     ) -> torch.Tensor:
#         return self.global_to_seen_local(labels_global, phase_classes, context=context)

#     @staticmethod
#     def assert_seen_logits(
#         logits: torch.Tensor,
#         seen_classes: Iterable[int],
#         context: str,
#     ) -> None:
#         if not torch.is_tensor(logits) or logits.dim() != 2:
#             raise RuntimeError(f"{context}: logits must be [B,S]")
#         seen = [int(v) for v in seen_classes]
#         if logits.size(1) != len(seen):
#             raise RuntimeError(
#                 f"{context}: logits width={logits.size(1)} != seen width={len(seen)}"
#             )
#         if not torch.isfinite(logits).all():
#             raise RuntimeError(f"{context}: logits contain NaN/Inf")

#     def cross_entropy_for_seen_logits(
#         self,
#         logits: torch.Tensor,
#         labels_global: torch.Tensor,
#         seen_classes: Iterable[int],
#         *,
#         context: str = "seen_ce",
#         class_balanced: bool = True,
#     ) -> torch.Tensor:
#         self.assert_seen_logits(logits, seen_classes, context)
#         labels_local = self.global_to_seen_local(
#             labels_global.to(logits.device), seen_classes, context=context
#         )
#         return self._stable_ce(logits, labels_local, class_balanced=class_balanced)

#     # ------------------------------------------------------------------
#     # Canonical feature extraction
#     # ------------------------------------------------------------------
#     @staticmethod
#     def assert_feature_tensor(
#         features: torch.Tensor,
#         *,
#         expected_dim: Optional[int] = None,
#         context: str = "features",
#     ) -> None:
#         if not torch.is_tensor(features) or features.dim() != 2:
#             raise RuntimeError(f"{context}: features must be [B,D]")
#         if expected_dim is not None and features.size(1) != int(expected_dim):
#             raise RuntimeError(
#                 f"{context}: feature dimension={features.size(1)} != {int(expected_dim)}"
#             )
#         if not torch.isfinite(features).all():
#             raise RuntimeError(f"{context}: features contain NaN/Inf")

#     def _validate_hsi_spectral_axis_contract(
#         self,
#         spectral_dim: int,
#         *,
#         context: str,
#     ) -> Dict[str, Any]:
#         """Validate optional wavelength-axis and bad-band metadata.

#         The numerical spectrum contract is mandatory. Exact wavelengths may be
#         unavailable for some public HSI loaders, but when an axis or bad-band
#         mask is supplied it must align exactly with the physical spectrum.
#         """
#         dataset = getattr(self, "dataset", None)
#         args = getattr(self, "args", None)

#         def first_attr(names: Sequence[str]) -> Any:
#             for owner in (dataset, args):
#                 if owner is None:
#                     continue
#                 for name in names:
#                     value = getattr(owner, name, None)
#                     if value is not None:
#                         return value
#             return None

#         axis_value = first_attr(("wavelengths", "spectral_axis", "band_centers"))
#         mask_value = first_attr(("bad_band_mask", "valid_band_mask"))
#         report: Dict[str, Any] = {
#             "spectral_dim": int(spectral_dim),
#             "wavelength_axis_available": axis_value is not None,
#             "bad_band_mask_available": mask_value is not None,
#         }
#         if axis_value is not None:
#             axis = torch.as_tensor(axis_value, dtype=torch.float64).flatten()
#             if axis.numel() != int(spectral_dim) or not torch.isfinite(axis).all():
#                 raise RuntimeError(
#                     f"{context}: wavelength axis must contain {spectral_dim} finite values"
#                 )
#             delta = axis[1:] - axis[:-1]
#             if delta.numel() and not bool(((delta > 0).all() or (delta < 0).all()).item()):
#                 raise RuntimeError(f"{context}: wavelength axis must be strictly monotonic")
#             report["axis_direction"] = "ascending" if delta.numel() == 0 or bool((delta > 0).all()) else "descending"
#         if mask_value is not None:
#             mask = torch.as_tensor(mask_value).bool().flatten()
#             if mask.numel() != int(spectral_dim):
#                 raise RuntimeError(
#                     f"{context}: bad/valid-band mask must contain {spectral_dim} values"
#                 )
#             report["active_band_count"] = int(mask.sum().item())
#         return report

#     def _validate_physical_spectral_summary(
#         self,
#         spectra: Optional[torch.Tensor],
#         *,
#         batch_size: int,
#         device: torch.device,
#         dtype: torch.dtype,
#         required: bool,
#         context: str,
#     ) -> Optional[torch.Tensor]:
#         if spectra is None or not torch.is_tensor(spectra) or spectra.numel() == 0:
#             if required:
#                 raise RuntimeError(
#                     f"{context}: wavelength-ordered physical center spectra are required"
#                 )
#             return None
#         summary = spectra.to(device=device, dtype=dtype, non_blocking=True)
#         if summary.dim() != 2 or summary.size(0) != int(batch_size) or summary.size(1) <= 1:
#             raise RuntimeError(
#                 f"{context}: physical spectra must be [B,S] with B={batch_size}; "
#                 f"got {tuple(summary.shape)}"
#             )
#         if not torch.isfinite(summary).all():
#             raise RuntimeError(f"{context}: physical spectra contain NaN/Inf")
#         if not self._cfg_bool("external_spectra_are_physical", required):
#             raise RuntimeError(
#                 f"{context}: external_spectra_are_physical must be true; PCA channels "
#                 "cannot be used as physical spectral geometry"
#             )
#         bank = getattr(self.model, "geometry_bank", None)
#         spectral_dim = int(getattr(bank, "_spectral_dim", torch.tensor(0)).item()) if bank is not None else 0
#         if spectral_dim > 0 and summary.size(1) != spectral_dim:
#             raise RuntimeError(
#                 f"{context}: spectral dimension {summary.size(1)} != bank dimension {spectral_dim}"
#             )
#         self._validate_hsi_spectral_axis_contract(summary.size(1), context=context)
#         return summary

#     def _extract_model_geometry_features(
#         self,
#         x: torch.Tensor,
#         *,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#         require_physical_spectra: bool = False,
#     ) -> Dict[str, Any]:
#         if not torch.is_tensor(x) or x.dim() < 2:
#             raise RuntimeError("model input must be a tensor with a batch dimension")
#         summary = self._validate_physical_spectral_summary(
#             spectral_summary,
#             batch_size=x.size(0),
#             device=x.device,
#             dtype=x.dtype,
#             required=bool(require_physical_spectra),
#             context="geometry feature extraction",
#         )
#         if summary is not None and not bool(spectral_summary_is_physical):
#             raise RuntimeError(
#                 "A supplied spectral summary must be explicitly marked physical"
#             )
#         method = getattr(self.model, "extract_canonical_projected_features", None)
#         if not callable(method):
#             method = getattr(self.model, "extract_projected_features", None)
#         if not callable(method):
#             raise AttributeError(
#                 "Model must expose extract_canonical_projected_features() or "
#                 "extract_projected_features()"
#             )
#         output = method(
#             x,
#             spectral_summary=summary,
#             spectral_summary_is_physical=bool(summary is not None),
#         )
#         if torch.is_tensor(output):
#             output = {"features": output}
#         if not isinstance(output, Mapping):
#             raise RuntimeError("Canonical feature extractor must return a tensor or mapping")
#         result = dict(output)
#         feature = next(
#             (
#                 result[key]
#                 for key in (
#                     "canonical_projected_features", "canonical_features",
#                     "geometry_features", "projected_features", "features",
#                 )
#                 if torch.is_tensor(result.get(key))
#             ),
#             None,
#         )
#         if feature is None:
#             raise RuntimeError("Canonical feature extractor returned no feature tensor")
#         expected_dim = int(
#             getattr(
#                 getattr(self.model, "geometry_bank", None),
#                 "feature_dim",
#                 getattr(self.model, "d_model", feature.size(1)),
#             )
#         )
#         self.assert_feature_tensor(
#             feature,
#             expected_dim=expected_dim,
#             context="canonical geometry features",
#         )
#         returned_physical = result.get("spectral_summary_is_physical", summary is not None)
#         if torch.is_tensor(returned_physical):
#             returned_physical = bool(returned_physical.item())
#         if summary is not None and not bool(returned_physical):
#             raise RuntimeError("Model rejected the physical spectral metadata contract")
#         result.update(
#             {
#                 "features": feature,
#                 "projected_features": feature,
#                 "geometry_features": feature,
#                 "canonical_features": feature,
#                 "canonical_projected_features": feature,
#                 "spectral_summary": summary if summary is not None else feature.new_empty((feature.size(0), 0)),
#                 "spectral_summary_is_physical": bool(summary is not None),
#                 "geometry_feature_space": "canonical_euclidean_z",
#                 "classifier_feature_space": "canonical_euclidean_z",
#             }
#         )
#         return result

#     # ------------------------------------------------------------------
#     # Canonical GeometryBank access and validation
#     # ------------------------------------------------------------------
#     def _geometry_bank_object(self) -> Any:
#         geometry_bank = getattr(self.model, "geometry_bank", None)
#         if geometry_bank is None:
#             raise RuntimeError("Model has no GeometryBank")
#         return geometry_bank

#     def _canonicalize_bank(
#         self,
#         bank: Mapping[str, Any],
#         *,
#         require_joint_schema: bool = True,
#     ) -> Dict[str, Any]:
#         """Validate and expose the complete persistent joint-row schema."""
#         output = dict(bank)
#         required = list(self.FEATURE_ROW_FIELDS)
#         if require_joint_schema:
#             required.extend(self.SPECTRAL_ROW_FIELDS)
#             required.extend(self.COUPLING_ROW_FIELDS)
#         missing = [key for key in required if not torch.is_tensor(output.get(key))]
#         if missing:
#             raise RuntimeError(f"GeometryBank mapping is missing tensors: {missing}")
#         row_count = int(output["means"].size(0))
#         for key in required:
#             tensor = output[key]
#             if tensor.dim() == 0 or tensor.size(0) != row_count:
#                 raise RuntimeError(
#                     f"GeometryBank {key} has leading size {tuple(tensor.shape)}, expected {row_count} rows"
#                 )
#         if not torch.is_tensor(output.get("variances")):
#             output["variances"] = torch.cat(
#                 [output["eigvals"], output["res_vars"].reshape(-1, 1)], dim=1
#             )
#         if not torch.is_tensor(output.get("valid_mask")):
#             counts = output["sample_counts"].flatten()
#             output["valid_mask"] = torch.isfinite(counts) & counts.gt(0)
#         if require_joint_schema and row_count:
#             ready = output["coupling_stats_ready"].bool().flatten()
#             fallback = output["coupling_fallback_mask"].bool().flatten()
#             if not bool((ready ^ fallback).all().item()):
#                 raise RuntimeError(
#                     "Every persistent row must be exactly joint-ready or feature-only fallback"
#                 )
#         return output

#     def _safe_get_subspace_bank(self, require_ready: bool = True) -> Dict[str, Any]:
#         geometry_bank = self._geometry_bank_object()
#         get_bank = getattr(geometry_bank, "get_bank", None)
#         if not callable(get_bank):
#             raise AttributeError("GeometryBank must expose get_bank()")
#         bank = self._canonicalize_bank(get_bank(), require_joint_schema=True)
#         if require_ready:
#             geometry_bank.assert_bank_valid(strict=True)
#             valid = geometry_bank.get_valid_mask().detach().clone()
#             bank["valid_mask"] = valid
#             if not bool(valid.any()):
#                 raise RuntimeError("GeometryBank has no valid rows")
#         return bank

#     def assert_bank_ready_for_seen_classes(
#         self,
#         bank: Optional[Mapping[str, Any]],
#         seen_classes: Iterable[int],
#         *,
#         require_statistics: bool = False,
#         require_spectral: bool = True,
#         require_joint_state: bool = True,
#         require_any_joint_ready: bool = False,
#         require_frozen: bool = False,
#     ) -> None:
#         ids = self._as_class_list(seen_classes)
#         if not ids:
#             raise RuntimeError("seen_classes is empty")
#         geometry_bank = self._geometry_bank_object()
#         geometry_bank.assert_bank_valid(ids, strict=True)
#         valid = geometry_bank.get_valid_mask()
#         missing = [cid for cid in ids if cid >= valid.numel() or not bool(valid[cid])]
#         if missing:
#             raise RuntimeError(f"GeometryBank rows are missing or invalid: {missing}")
#         if require_statistics:
#             missing_by_name = {
#                 name: [cid for cid in ids if not bool(getattr(geometry_bank, name)[cid])]
#                 for name in ("energy_stats_ready", "tail_stats_ready", "margin_stats_ready")
#             }
#             incomplete = {name: values for name, values in missing_by_name.items() if values}
#             if incomplete:
#                 raise RuntimeError(f"GeometryBank statistics are incomplete: {incomplete}")
#         if require_spectral:
#             missing_spectral = [cid for cid in ids if not bool(geometry_bank.spectral_stats_ready[cid])]
#             if missing_spectral:
#                 raise RuntimeError(
#                     f"Physical low-rank spectral geometry is missing: {missing_spectral}"
#                 )
#         if require_joint_state:
#             ready = geometry_bank.coupling_stats_ready[ids].bool()
#             fallback = geometry_bank.coupling_fallback_mask[ids].bool()
#             bad = [ids[i] for i in range(len(ids)) if not bool((ready ^ fallback)[i])]
#             if bad:
#                 raise RuntimeError(f"Joint coupling state is incomplete or contradictory: {bad}")
#             if require_any_joint_ready and not bool(ready.any().item()):
#                 raise RuntimeError(
#                     "No class has reliable spectral-feature coupling; the joint method is inactive"
#                 )
#         if require_frozen:
#             unfrozen = [cid for cid in ids if not bool(geometry_bank.frozen_class_mask[cid])]
#             if unfrozen:
#                 raise RuntimeError(f"GeometryBank rows are not frozen: {unfrozen}")
#         if bank is not None:
#             canonical = self._canonicalize_bank(bank, require_joint_schema=True)
#             if canonical["means"].size(0) != len(geometry_bank):
#                 raise RuntimeError("Supplied GeometryBank mapping is stale or has wrong row count")

#     def assert_bank_has_only_allowed_valid_rows(
#         self,
#         bank: Optional[Mapping[str, Any]],
#         allowed_classes: Iterable[int],
#     ) -> None:
#         allowed = set(self._as_class_list(allowed_classes))
#         geometry_bank = self._geometry_bank_object()
#         valid_rows = torch.nonzero(
#             geometry_bank.get_valid_mask(), as_tuple=False
#         ).flatten().detach().cpu().tolist()
#         leaked = [int(cid) for cid in valid_rows if int(cid) not in allowed]
#         if leaked:
#             raise RuntimeError(
#                 f"GeometryBank contains valid rows outside the allowed set: {leaked}"
#             )
#         if bank is not None:
#             canonical = self._canonicalize_bank(bank, require_joint_schema=True)
#             mapping_rows = torch.nonzero(
#                 canonical["valid_mask"].detach().cpu().bool().flatten(),
#                 as_tuple=False,
#             ).flatten().tolist()
#             if mapping_rows != valid_rows:
#                 raise RuntimeError("Supplied GeometryBank mapping does not match live valid rows")

#     def compute_bank_validity_diagnostics(
#         self,
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#     ) -> Dict[str, Any]:
#         geometry_bank = self._geometry_bank_object()
#         report = geometry_bank.assert_bank_valid(
#             self._as_class_list(seen_classes) if seen_classes is not None else None,
#             strict=False,
#         )
#         valid = geometry_bank.get_valid_mask()
#         ids = (
#             self._as_class_list(seen_classes)
#             if seen_classes is not None
#             else torch.nonzero(valid, as_tuple=False).flatten().tolist()
#         )
#         joint = (
#             geometry_bank.joint_spectral_feature_report(ids)
#             if ids and callable(getattr(geometry_bank, "joint_spectral_feature_report", None))
#             else {"joint_ready_count": 0, "fallback_count": 0, "joint_ready_rate": 0.0, "per_class": []}
#         )
#         def state(name: str) -> Dict[int, bool]:
#             tensor = getattr(geometry_bank, name)
#             return {int(cid): bool(tensor[cid]) for cid in ids}
#         return {
#             "valid": not report.get("errors"),
#             "errors": list(report.get("errors", [])),
#             "class_ids": ids,
#             "valid_rows": torch.nonzero(valid, as_tuple=False).flatten().tolist(),
#             "energy_stats_ready": state("energy_stats_ready"),
#             "tail_stats_ready": state("tail_stats_ready"),
#             "margin_stats_ready": state("margin_stats_ready"),
#             "spectral_stats_ready": state("spectral_stats_ready"),
#             "coupling_stats_ready": state("coupling_stats_ready"),
#             "coupling_fallback": state("coupling_fallback_mask"),
#             "frozen": state("frozen_class_mask"),
#             "joint_spectral_feature": joint,
#         }

#     # ------------------------------------------------------------------
#     # Old-row immutability
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def snapshot_bank_rows(
#         self,
#         bank: Optional[Mapping[str, Any]],
#         class_ids: Iterable[int],
#     ) -> Dict[str, torch.Tensor]:
#         del bank
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             return {"class_ids": torch.empty(0, device=self.device, dtype=torch.long)}
#         return self._geometry_bank_object().snapshot_rows(ids)

#     @torch.no_grad()
#     def assert_bank_rows_unchanged(
#         self,
#         before: Mapping[str, torch.Tensor],
#         after: Optional[Mapping[str, Any]],
#         class_ids: Iterable[int],
#         context: str,
#         *,
#         atol: float = 0.0,
#         check_frozen_mask: bool = True,
#     ) -> None:
#         del after
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             return
#         if abs(float(atol)) > 0.0:
#             raise RuntimeError(
#                 f"{context}: old-row immutability requires exact equality; atol must be 0"
#             )
#         geometry_bank = self._geometry_bank_object()
#         exact = getattr(geometry_bank, "assert_rows_identical", None)
#         if callable(exact):
#             exact(before, ids, context=context)
#             return
#         geometry_bank.assert_rows_unchanged(
#             before,
#             ids,
#             context=context,
#             atol=0.0,
#             rtol=0.0,
#             check_frozen_mask=bool(check_frozen_mask),
#         )

#     def _old_bank_integrity_snapshot(
#         self, old_class_ids: Iterable[int]
#     ) -> Dict[str, torch.Tensor]:
#         ids = self._as_class_list(old_class_ids)
#         return self.snapshot_bank_rows(None, ids) if ids else {}

#     def _assert_old_bank_integrity(
#         self,
#         old_class_ids: Iterable[int],
#         snapshot: Mapping[str, torch.Tensor],
#         *,
#         context: str,
#         atol: float = 0.0,
#     ) -> None:
#         ids = self._as_class_list(old_class_ids)
#         if ids:
#             self.assert_bank_rows_unchanged(
#                 snapshot,
#                 None,
#                 ids,
#                 context,
#                 atol=atol,
#                 check_frozen_mask=True,
#             )

#     # ------------------------------------------------------------------
#     # Exact shared energy wrappers
#     # ------------------------------------------------------------------
#     def _geometry_energy_from_bank(
#         self,
#         features: torch.Tensor,
#         *,
#         seen_classes: Iterable[int],
#         bank: Optional[Union[Mapping[str, Any], Any]] = None,
#         return_parts: bool = False,
#     ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#         ids = self._as_class_list(seen_classes)
#         self.assert_feature_tensor(features, context="geometry energy features")
#         live_bank = self._geometry_bank_object()
#         bank_source = live_bank if bank is None else bank
#         logdet = float(getattr(live_bank, "energy_logdet_weight", 1.0))
#         if logdet <= 0.0:
#             raise RuntimeError("Full low-rank Gaussian likelihood requires positive logdet weight")
#         classifier = getattr(self.model, "classifier", None)
#         if classifier is not None:
#             if not bool(getattr(classifier, "normalize_energy_by_dim", False)):
#                 raise RuntimeError("Classifier energy must be dimension normalized")
#             if abs(float(getattr(classifier, "energy_logdet_weight", logdet)) - logdet) > 1e-12:
#                 raise RuntimeError("Classifier and GeometryBank logdet weights disagree")
#         return geometry_energy_matrix(
#             features=features,
#             bank=bank_source,
#             seen_classes=ids,
#             normalize_by_dim=True,
#             logdet_weight=logdet,
#             invalid_class_energy=float(
#                 getattr(classifier, "invalid_class_energy", 1e6)
#             ),
#             return_parts=return_parts,
#         )

#     def _dual_geometry_energy_matrix(
#         self,
#         features: torch.Tensor,
#         bank: Optional[Union[Mapping[str, Any], Any]],
#         *,
#         seen_classes: Optional[Iterable[int]] = None,
#         return_parts: bool = False,
#         **_: Any,
#     ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#         ids = (
#             self._as_class_list(seen_classes)
#             if seen_classes is not None
#             else torch.nonzero(
#                 self._geometry_bank_object().get_valid_mask(), as_tuple=False
#             ).flatten().tolist()
#         )
#         return self._geometry_energy_from_bank(
#             features,
#             seen_classes=ids,
#             bank=bank,
#             return_parts=return_parts,
#         )

#     def _geometry_energy_matrix(
#         self,
#         features: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         active_ranks: torch.Tensor,
#         sample_counts: Optional[torch.Tensor] = None,
#         return_parts: bool = False,
#         **_: Any,
#     ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#         if not torch.is_tensor(sample_counts):
#             raise RuntimeError(
#                 "sample_counts is required; statistical support must never be fabricated"
#             )
#         if variances.dim() != 2 or variances.size(1) != bases.size(2) + 1:
#             raise RuntimeError("variances must be [C,R+1]")
#         if sample_counts.numel() != means.size(0):
#             raise RuntimeError("sample_counts must contain one value per row")
#         temporary_bank: Dict[str, Any] = {
#             "means": means,
#             "bases": bases,
#             "eigvals": variances[:, :-1],
#             "res_vars": variances[:, -1],
#             "active_ranks": active_ranks,
#             "sample_counts": sample_counts,
#             "valid_mask": torch.ones(means.size(0), device=means.device, dtype=torch.bool),
#             "variance_floor": float(
#                 getattr(self._geometry_bank_object(), "variance_floor", 1e-4)
#             ),
#             "energy_logdet_weight": float(
#                 getattr(self._geometry_bank_object(), "energy_logdet_weight", 1.0)
#             ),
#         }
#         return geometry_energy_matrix(
#             features=features,
#             bank=temporary_bank,
#             normalize_by_dim=True,
#             return_parts=return_parts,
#         )

#     # ------------------------------------------------------------------
#     # Phase class order and incremental contract
#     # ------------------------------------------------------------------
#     def resolve_phase_classes(
#         self, phase: int
#     ) -> Tuple[List[int], List[int], List[int]]:
#         """Resolve old, current, and seen class IDs without assuming contiguous IDs.

#         ``phase_to_classes`` may be either a sequence indexed by phase or a
#         mapping keyed by phase. Class order is preserved because it defines the
#         seen-local energy-column order.
#         """
#         phase = int(phase)
#         if phase <= 0:
#             raise ValueError("Incremental phase must be greater than zero")
#         phase_map = getattr(self.dataset, "phase_to_classes", None)
#         if phase_map is None:
#             raise RuntimeError("dataset.phase_to_classes is required")

#         def phase_classes(index: int) -> List[int]:
#             try:
#                 values = phase_map[index]
#             except (KeyError, IndexError, TypeError) as exc:
#                 raise RuntimeError(
#                     f"dataset.phase_to_classes has no phase {index}"
#                 ) from exc
#             classes = self._as_class_list(values)
#             if not classes:
#                 raise RuntimeError(f"dataset phase {index} has no classes")
#             return classes

#         new_classes = phase_classes(phase)
#         if callable(getattr(self.dataset, "get_classes_up_to_phase", None)):
#             old_classes = self._as_class_list(
#                 self.dataset.get_classes_up_to_phase(phase - 1)
#             )
#         else:
#             old_classes = self._as_class_list(
#                 class_id
#                 for previous_phase in range(phase)
#                 for class_id in phase_classes(previous_phase)
#             )
#         if not old_classes:
#             raise RuntimeError(
#                 f"phase {phase} has no old classes; phase 0 was not finalized"
#             )
#         overlap = sorted(set(old_classes).intersection(new_classes))
#         if overlap:
#             raise RuntimeError(
#                 f"phase {phase} old/new class overlap: {overlap}"
#             )
#         return old_classes, new_classes, [*old_classes, *new_classes]

#     def _seen_class_ids_before_phase(self, phase: int) -> List[int]:
#         ids: List[int] = []
#         for index in range(max(int(phase), 0)):
#             ids.extend(int(v) for v in self.dataset.phase_to_classes[index])
#         return self._as_class_list(ids)

#     def _seen_class_ids_through_phase(self, phase: int) -> List[int]:
#         ids: List[int] = []
#         for index in range(max(int(phase), 0) + 1):
#             ids.extend(int(v) for v in self.dataset.phase_to_classes[index])
#         return self._as_class_list(ids)

#     def assert_clean_incremental_contract(
#         self,
#         phase: int,
#         old_classes: Iterable[int],
#         new_classes: Iterable[int],
#         seen_classes: Iterable[int],
#         *,
#         context: str = "incremental_contract",
#     ) -> None:
#         phase = int(phase)
#         old_ids = self._as_class_list(old_classes)
#         new_ids = self._as_class_list(new_classes)
#         seen_ids = self._as_class_list(seen_classes)
#         if phase <= 0:
#             raise RuntimeError(f"{context}: phase must be > 0")
#         if not old_ids or not new_ids:
#             raise RuntimeError(f"{context}: old and new class lists must be non-empty")
#         if set(old_ids) & set(new_ids):
#             raise RuntimeError(f"{context}: old/new class lists overlap")
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError(
#                 f"{context}: seen_classes must preserve old+new order; "
#                 f"old={old_ids}, new={new_ids}, seen={seen_ids}"
#             )
#         configured_mode = str(
#             getattr(getattr(self, "args", None), "incremental_update_mode", "joint_geometry_admission")
#         ).strip().lower().replace("-", "_")
#         model_mode = str(
#             getattr(self.model, "incremental_update_mode", configured_mode)
#         ).strip().lower().replace("-", "_")
#         if configured_mode != "joint_geometry_admission" or model_mode != "joint_geometry_admission":
#             raise RuntimeError(
#                 f"{context}: args/model must both use joint_geometry_admission; "
#                 f"args={configured_mode!r}, model={model_mode!r}"
#             )
#         forbidden = (
#             "use_geometry_transport", "use_sglat_transport",
#             "allow_old_model_transport", "use_energy_calibrator",
#             "use_adaptive_boundary", "use_incremental_adapter",
#             "use_geometry_gated_adapter", "allow_incremental_projection_training",
#             "normalize_geometry_features",
#         )
#         active = [name for name in forbidden if self._cfg_bool(name, False)]
#         if active:
#             raise RuntimeError(f"{context}: forbidden architecture switches are active: {active}")
#         contract_method = getattr(self.model, "feature_contract", None)
#         if not callable(contract_method):
#             raise RuntimeError(f"{context}: model must expose feature_contract()")
#         contract = contract_method()
#         if contract.get("geometry_feature_space") != "euclidean_residual_projected_z":
#             raise RuntimeError(f"{context}: canonical Euclidean z-space contract is broken")
#         if contract.get("spectral_geometry") != "physical_raw_center_spectrum_low_rank_gaussian":
#             raise RuntimeError(f"{context}: low-rank physical spectral geometry is required")
#         if contract.get("joint_geometry") != "whitened_latent_spectral_feature_correlation":
#             raise RuntimeError(f"{context}: joint latent coupling contract is required")
#         if contract.get("inference_geometry") != "feature_low_rank_gaussian_only":
#             raise RuntimeError(f"{context}: inference must remain feature-only")
#         geometry_bank = self._geometry_bank_object()
#         classifier = getattr(self.model, "classifier", None)
#         bank_logdet = float(getattr(geometry_bank, "energy_logdet_weight", 1.0))
#         classifier_logdet = float(getattr(classifier, "energy_logdet_weight", bank_logdet))
#         if bank_logdet <= 0.0 or abs(bank_logdet - classifier_logdet) > 1e-12:
#             raise RuntimeError(f"{context}: classifier/bank likelihood contract mismatch")
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         self.assert_bank_ready_for_seen_classes(
#             bank,
#             old_ids,
#             require_statistics=True,
#             require_spectral=True,
#             require_joint_state=True,
#             require_any_joint_ready=bool(getattr(self.model, "require_joint_coupling_for_base", True)),
#             require_frozen=True,
#         )
#         self.assert_bank_has_only_allowed_valid_rows(bank, old_ids)

#     # ------------------------------------------------------------------
#     # Safe current-phase row commit
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def commit_new_class_rows_only(
#         self,
#         class_ids: Iterable[int],
#         candidate_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
#         *,
#         features: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         spectral_summary: Optional[torch.Tensor] = None,
#         spectral_summary_is_physical: bool = False,
#         phase_created: Optional[int] = None,
#         freeze: bool = False,
#         context: str = "commit_new_class_rows_only",
#     ) -> Dict[str, Any]:
#         """Atomically commit complete joint rows for current-phase classes only.

#         Feature-only tensor blocks are deliberately unsupported. If descriptors
#         were refined, the caller must rebuild/re-estimate spectral geometry and
#         coupling, then pass complete candidate rows.
#         """
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             return {"active": 0, "committed_class_ids": []}
#         phase = int(getattr(self.model, "current_phase", 0))
#         if phase <= 0:
#             raise RuntimeError(f"{context}: new-row commit is incremental-phase only")
#         old_ids = self._seen_class_ids_before_phase(phase)
#         allowed_new = self._as_class_list(self.dataset.phase_to_classes[phase])
#         if set(ids) & set(old_ids):
#             raise RuntimeError(f"{context}: refusing to overwrite old rows")
#         invalid = [cid for cid in ids if cid not in allowed_new]
#         if invalid:
#             raise RuntimeError(f"{context}: rows are not current-phase classes: {invalid}")
#         del features, labels, spectral_summary, spectral_summary_is_physical
#         if candidate_rows is None:
#             raise RuntimeError(
#                 f"{context}: pass fully certified candidate_rows with physical spectral, "
#                 "coupling, energy, tail-energy, and margin statistics. Raw features "
#                 "cannot be committed directly because admission must remain external "
#                 "to the live bank until certification is complete."
#             )
#         if not isinstance(candidate_rows, Mapping):
#             raise TypeError(
#                 f"{context}: feature-only descriptor blocks are retired; pass complete joint candidate rows"
#             )
#         if set(int(k) for k in candidate_rows) != set(ids):
#             raise RuntimeError(f"{context}: candidate row IDs do not match current-phase IDs")
#         if not bool(freeze):
#             raise RuntimeError(
#                 f"{context}: incremental rows must be frozen at atomic admission"
#             )
#         geometry_bank = self._geometry_bank_object()
#         old_snapshot = self._old_bank_integrity_snapshot(old_ids)
#         old_digest = geometry_bank.rows_digest(old_ids)
#         commit = getattr(geometry_bank, "commit_incremental_geometry_rows", None)
#         if not callable(commit):
#             raise RuntimeError(
#                 "GeometryBank must expose commit_incremental_geometry_rows()"
#             )
#         result = commit(
#             candidate_rows,
#             old_class_ids=old_ids,
#             phase_created=phase if phase_created is None else int(phase_created),
#             expected_old_digest=old_digest,
#             require_spectral=True,
#             require_statistics=True,
#             context=context,
#         )
#         self._assert_old_bank_integrity(
#             old_ids,
#             old_snapshot,
#             context=f"{context}: old joint-row immutability",
#             atol=0.0,
#         )
#         self.assert_bank_ready_for_seen_classes(
#             None,
#             ids,
#             require_statistics=True,
#             require_spectral=True,
#             require_joint_state=True,
#             require_frozen=True,
#         )
#         return dict(result)

#     def _commit_refined_feature_rows(self, *args: Any, **kwargs: Any) -> None:
#         raise RuntimeError(
#             "Feature-only row commit is retired. Re-estimate the complete joint row "
#             "after refinement and call commit_new_class_rows_only(candidate_rows=...)."
#         )

#     # ------------------------------------------------------------------
#     # Minimal phase certificate and persistence
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def geometry_phase_certificate(
#         self,
#         phase: int,
#         class_ids: Iterable[int],
#         *,
#         require_statistics: bool = True,
#         require_spectral: bool = True,
#         require_any_joint_ready: bool = False,
#         require_frozen: bool = False,
#     ) -> Dict[str, Any]:
#         ids = self._as_class_list(class_ids)
#         geometry_bank = self._geometry_bank_object()
#         state = geometry_bank.phase_geometry_state_report(
#             ids,
#             freeze=False,
#             require_statistics=require_statistics,
#             require_uncertainty=False,
#             require_spectral=require_spectral,
#             context=f"phase_{int(phase)}_certificate",
#         )
#         errors = list(state.get("errors", []))
#         if require_any_joint_ready and not bool(geometry_bank.coupling_stats_ready[ids].any().item()):
#             errors.append("no reliable spectral-feature coupling is active")
#         unfrozen = [cid for cid in ids if not bool(geometry_bank.frozen_class_mask[cid])]
#         if require_frozen and unfrozen:
#             errors.append(f"unfrozen classes: {unfrozen}")
#         joint = geometry_bank.joint_spectral_feature_report(ids)
#         state.update(
#             {
#                 "phase": int(phase),
#                 "errors": errors,
#                 "ok": not errors and bool(state.get("ok", False)),
#                 "energy_logdet_weight": float(geometry_bank.energy_logdet_weight),
#                 "normalize_by_dimension": bool(
#                     getattr(getattr(self.model, "classifier", None), "normalize_energy_by_dim", True)
#                 ),
#                 "inference_uses_spectral_score": False,
#                 "inference_uses_coupling_score": False,
#                 "old_rows_immutable": True,
#                 "joint_spectral_feature": joint,
#                 "joint_ready_rate": float(joint["joint_ready_rate"]),
#                 "joint_ready_count": int(joint["joint_ready_count"]),
#                 "joint_fallback_count": int(joint["fallback_count"]),
#             }
#         )
#         return state

#     @staticmethod
#     def _json_safe(value: Any) -> Any:
#         if torch.is_tensor(value):
#             tensor = value.detach().cpu()
#             return tensor.item() if tensor.numel() == 1 else tensor.tolist()
#         if isinstance(value, Mapping):
#             return {str(k): TrainerHelper._json_safe(v) for k, v in value.items()}
#         if isinstance(value, (tuple, list)):
#             return [TrainerHelper._json_safe(v) for v in value]
#         if isinstance(value, (str, int, float, bool)) or value is None:
#             return value
#         return str(value)

#     def save_json_diagnostics(self, path: str, data: Mapping[str, Any]) -> None:
#         directory = os.path.dirname(path)
#         if directory:
#             os.makedirs(directory, exist_ok=True)
#         with open(path, "w", encoding="utf-8") as stream:
#             json.dump(self._json_safe(dict(data)), stream, indent=2)


