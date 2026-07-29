from __future__ import annotations

"""Top-level orchestration for transport-verified factor-geometry NECIL.

Phase 0 is executed by ``BasePhaseTrainer``.  Phases t>0 are executed by
``IncrementalPhaseTrainer``.  Accepted checkpoints contain only the deployed
backbone, aggregate GeometryBank, parameter-free classifier state, phase
metadata, and diagnostics; temporary phase observers are never persisted in
the model checkpoint.
"""

import copy
import math
import os
from typing import Any, Dict, List, Mapping, Optional
import torch
from trainers.base_phase_trainer import BasePhaseTrainer
from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
from trainers.trainer_helpers import TrainerHelper


class Trainer(IncrementalPhaseTrainer, BasePhaseTrainer, TrainerHelper):
    CHECKPOINT_FORMAT_VERSION = 4

    def __init__(self, model: torch.nn.Module, dataset: Any, args: Any) -> None:
        if model is None:
            raise TypeError("Trainer requires a model")
        if dataset is None:
            raise TypeError("Trainer requires a dataset")
        if args is None:
            raise TypeError("Trainer requires explicit configuration")

        self.args = args
        self.device = self._resolve_device(getattr(args, "device", "cpu"))
        self.save_dir = os.path.abspath(
            str(getattr(args, "save_dir", "./outputs"))
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.debug = self._parse_bool(
            getattr(args, "debug_verbose", False), "debug_verbose"
        )
        self._last_base_geometry_report: Optional[Dict[str, Any]] = None
        self._last_phase_history: Optional[Dict[str, Any]] = None

        self.assert_architecture_contract()
        dataset_contract = self.assert_dataset_contract()
        self.phase_schedule = dict(dataset_contract["schedule"])
        self.base_classes = list(dataset_contract["base_classes"])
        self._prepare_initial_state()

    # ------------------------------------------------------------------
    # Device and accepted phase state
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y", "on"}:
                return True
            if token in {"0", "false", "no", "n", "off"}:
                return False
        raise RuntimeError(f"{name} must be an explicit boolean")

    @staticmethod
    def _resolve_device(value: Any) -> torch.device:
        token = str(value).strip().lower()
        if token == "gpu":
            token = "cuda"
        requested = torch.device(token)
        if requested.type == "cpu":
            return requested
        if requested.type != "cuda":
            raise RuntimeError(f"unsupported device type {requested.type!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = (
            torch.cuda.current_device()
            if requested.index is None
            else requested.index
        )
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError(f"invalid CUDA device index {index}")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    def _cumulative_classes(self, phase: int) -> List[int]:
        if phase not in self.phase_schedule:
            raise RuntimeError(f"phase {phase} is outside the schedule")
        output: List[int] = []
        for index in range(phase + 1):
            output.extend(int(value) for value in self.phase_schedule[index])
        return output

    def _phase_for_valid_rows(self, valid_rows: List[int]) -> Optional[int]:
        valid_set = set(int(value) for value in valid_rows)
        for phase in sorted(self.phase_schedule):
            if valid_set == set(self._cumulative_classes(phase)):
                return int(phase)
        return None

    def _prepare_initial_state(self) -> None:
        bank = self.model.geometry_bank
        valid = self.model.infer_seen_classes()
        if not valid:
            partial_contract = any(
                (
                    bool(bank.global_priors_ready.item()),
                    bool(bank.overlap_temperatures_ready.item()),
                    bool(self.model.classifier.require_bound_contract),
                    len(bank) > 0,
                )
            )
            if partial_contract:
                raise RuntimeError(
                    "the model contains a partial geometry state without valid "
                    "committed rows; start fresh or load an accepted checkpoint"
                )
            self.model.set_base_mode()
            return

        phase = self._phase_for_valid_rows(valid)
        if phase is None:
            raise RuntimeError(
                f"existing rows {valid} do not match any cumulative phase schedule"
            )
        self._set_accepted_phase_state(phase)
        self._assert_final_phase_memory(phase)

    def _set_accepted_phase_state(self, phase: int) -> None:
        seen = self._cumulative_classes(phase)
        self.model.current_phase = int(phase)
        self.model.phase_mode = "evaluation"
        self.model.seen_classes = list(seen)
        self.model.old_classes = list(seen)
        self.model.new_classes = []
        self.model.phase_old_digest = None
        self.model.backbone.freeze_all()
        self.model.classifier.require_bound_contract = True
        self.model.eval()

    def _accepted_phase(self) -> int:
        valid = self.model.infer_seen_classes()
        if not valid:
            return -1
        phase = self._phase_for_valid_rows(valid)
        if phase is None:
            raise RuntimeError("committed rows do not match the phase schedule")
        return phase

    def _assert_final_phase_memory(self, phase: int) -> Dict[str, Any]:
        seen = self._cumulative_classes(phase)
        bank = self.model.geometry_bank
        validity = bank.assert_valid(seen, strict=False)
        if not validity.get("ok", False):
            raise RuntimeError(
                "accepted GeometryBank is invalid: "
                + "; ".join(validity.get("errors", []))
            )
        actual = self.model.infer_seen_classes()
        if set(actual) != set(seen):
            raise RuntimeError(
                f"accepted rows {actual} do not match expected phase-{phase} rows {seen}"
            )
        bank.assert_global_priors_ready()
        if not bool(bank.global_priors_frozen.item()):
            raise RuntimeError("global priors are not frozen")
        if not bool(bank.overlap_temperatures_ready.item()):
            raise RuntimeError("pair-risk temperatures are absent")
        if not bool(bank.overlap_temperatures_frozen.item()):
            raise RuntimeError("pair-risk temperatures are not frozen")
        contract = self.model.classifier.bank_contract_digest(bank)
        if self.model.classifier.bound_bank_contract_digest != contract:
            raise RuntimeError("classifier is not bound to the static bank contract")
        if not self.model.classifier.require_bound_contract:
            raise RuntimeError("classifier contract enforcement is disabled")
        if self.model.phase_mode != "evaluation":
            raise RuntimeError("accepted model must be in evaluation mode")
        if self.model.new_classes:
            raise RuntimeError("accepted model retains active new classes")
        if any(parameter.requires_grad for parameter in self.model.backbone.parameters()):
            raise RuntimeError("accepted backbone must be frozen")
        return {
            "ok": True,
            "phase": int(phase),
            "seen_classes": list(seen),
            "valid_rows": actual,
            "rows_digest": bank.rows_digest(seen),
            "bank_contract_digest": contract,
            "classification_factorization": "p(z|c)",
            "spectral_relation_factorization": "p(h|c)",
            "stores_sample_level_memory": False,
        }

    # ------------------------------------------------------------------
    # Phase routing
    # ------------------------------------------------------------------

    def train_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        phase = int(phase)
        if phase not in self.phase_schedule:
            raise ValueError(
                f"phase {phase} is outside dataset schedule {sorted(self.phase_schedule)}"
            )
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(float(lr)) or float(lr) <= 0.0:
            raise ValueError("learning rate must be finite and positive")

        self.assert_architecture_contract()
        self.assert_dataset_contract()
        accepted = self._accepted_phase()
        if phase == 0:
            if accepted >= 0:
                raise RuntimeError("phase 0 is already finalized")
            self.model.set_base_mode()
            self._print_execution_summary(phase=0)
            return self.train_base_phase(
                phase=0,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
            )

        if accepted != phase - 1:
            raise RuntimeError(
                f"phase {phase} requires accepted phase {phase - 1}; "
                f"current accepted phase is {accepted}"
            )
        self._print_execution_summary(phase=phase)
        return self.train_incremental_phase(
            phase=phase,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
        )

    def _print_execution_summary(self, *, phase: int) -> None:
        bank = self.model.geometry_bank
        parameters = sum(int(p.numel()) for p in self.model.parameters())
        print(
            "[Trainer] "
            f"phase={phase} | device={self.device} | parameters={parameters:,} | "
            f"token_dim={self.model.backbone.token_dim} | "
            f"spectral_dim={self.model.spectral_dim} | "
            f"spatial_dim={self.model.spatial_dim} | "
            f"feature_dim={self.model.feature_dim} | "
            f"maximum_rank={bank.maximum_rank} | raw_bands={bank.raw_spectral_dim}"
        )
        if phase == 0:
            print(
                "[Base objective] CE warm-up + bidirectional cross-fitted "
                "risk-guided factor-energy shaping."
            )
        else:
            print(
                "[Incremental objective] current-query factor-energy separation + "
                "coordinate consistency under analytical branch transport."
            )
            print(
                "[Incremental memory] exact old-row pushforward + new aggregate "
                "rows; no old samples, feature replay, teacher, or trainable transport."
            )

    # ------------------------------------------------------------------
    # Checkpoint contract
    # ------------------------------------------------------------------

    @staticmethod
    def _clone(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {str(key): Trainer._clone(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(Trainer._clone(item) for item in value)
        if isinstance(value, list):
            return [Trainer._clone(item) for item in value]
        return copy.deepcopy(value)

    def _runtime_contract(self) -> Dict[str, Any]:
        model = self.model
        bank = model.geometry_bank
        backbone = model.backbone
        return {
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
            "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
            "architecture_version": int(model.ARCHITECTURE_VERSION),
            "backbone_contract_version": int(backbone.CONTRACT_VERSION),
            "bank_schema_version": int(bank.SCHEMA_VERSION),
            "model_input_bands": int(backbone.model_input_bands),
            "raw_spectral_dim": int(bank.raw_spectral_dim),
            "patch_size": int(backbone.patch_size),
            "token_dim": int(backbone.token_dim),
            "spectral_dim": int(model.spectral_dim),
            "spatial_dim": int(model.spatial_dim),
            "feature_dim": int(model.feature_dim),
            "maximum_rank": int(bank.maximum_rank),
            "spectral_shape_dim": int(bank.spectral_shape_dim),
            "spectral_resample_length": int(bank.spectral_resample_length),
            "volume_weight": float(bank.volume_weight),
            "variance_floor_absolute": float(bank.variance_floor_absolute),
            "variance_floor_relative": float(bank.variance_floor_relative),
            "classifier_temperature": float(model.classifier.temperature),
            "joint_feature": "direct_[z_s;z_p]",
            "trainable_transport_network": False,
            "geometry_replay_training": False,
            "stores_exemplars": False,
            "stores_old_features": False,
            "stores_old_spectra": False,
            "uses_knowledge_distillation": False,
        }

    def _assert_runtime_contract(self, runtime: Mapping[str, Any]) -> None:
        current = self._runtime_contract()
        errors = []
        for key, current_value in current.items():
            saved = runtime.get(key)
            if isinstance(current_value, float):
                try:
                    equal = math.isclose(
                        float(saved), current_value, rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = saved == current_value
            if not equal:
                errors.append(
                    f"{key}: checkpoint={saved!r}, current={current_value!r}"
                )
        if errors:
            raise RuntimeError("checkpoint runtime contract mismatch: " + "; ".join(errors))

    def _compact_memory_audit(self) -> Dict[str, Any]:
        snapshot = dict(self.model.memory_snapshot())
        snapshot.pop("bank", None)
        snapshot["memory_cost"] = self.model.geometry_bank.memory_cost_summary()
        return snapshot

    def _preload_geometry_bank_buffers(
        self,
        model_state: Mapping[str, Any],
    ) -> None:
        bank = self.model.geometry_bank
        snapshot: Dict[str, torch.Tensor] = {}
        missing = []
        for name in bank._buffers:
            key = f"geometry_bank.{name}"
            value = model_state.get(key)
            if not torch.is_tensor(value):
                missing.append(key)
            else:
                snapshot[name] = value
        if missing:
            raise RuntimeError(
                "checkpoint GeometryBank buffers are incomplete: "
                f"{missing[:12]}"
            )
        bank.load_snapshot(snapshot, strict=True)

    def save_checkpoint(
        self,
        phase: int,
        history: Mapping[str, Any],
        evaluator_metrics: Optional[Mapping[str, Any]] = None,
    ) -> str:
        phase = int(phase)
        if phase not in self.phase_schedule:
            raise RuntimeError(f"cannot save unknown phase {phase}")
        if not isinstance(history, Mapping):
            raise TypeError("history must be a mapping")
        if self._accepted_phase() != phase:
            raise RuntimeError("only the currently accepted phase can be checkpointed")
        memory_report = self._assert_final_phase_memory(phase)

        if phase == 0:
            final_report = history.get("base_geometry_report")
            if not isinstance(final_report, Mapping):
                raise RuntimeError("history.base_geometry_report is missing")
            certificate = self.base_geometry_certificate(final_report)
            if not certificate["checks"]["structural_geometry_valid"]:
                raise RuntimeError("refusing to save structurally invalid base geometry")
        else:
            admission = history.get("admission")
            if not isinstance(admission, Mapping) or not bool(admission.get("valid", False)):
                raise RuntimeError("incremental checkpoint lacks a valid admission record")
            certificate = dict(admission)

        phase_dir = os.path.join(self.save_dir, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)
        path = os.path.join(phase_dir, "checkpoint.pth")
        temporary = path + ".tmp"
        seen = self._cumulative_classes(phase)
        checkpoint: Dict[str, Any] = {
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "checkpoint_kind": "accepted_necil_phase",
            "phase": phase,
            "runtime_contract": self._runtime_contract(),
            "model_state_dict": {
                key: self._clone(value)
                for key, value in self.model.state_dict().items()
            },
            "memory_audit": self._clone(self._compact_memory_audit()),
            "base_classes": list(self.base_classes),
            "seen_classes": list(seen),
            "memory_report": self._clone(memory_report),
            "geometry_certificate": self._clone(certificate),
            "history": self._clone(dict(history)),
            "args": dict(vars(self.args)) if hasattr(self.args, "__dict__") else {},
            "torch_rng_state": torch.get_rng_state().cpu(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = [
                state.cpu() for state in torch.cuda.get_rng_state_all()
            ]
        if evaluator_metrics is not None:
            checkpoint["evaluator_metrics"] = self._clone(dict(evaluator_metrics))

        try:
            with open(temporary, "wb") as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise
        print(f"[Saved accepted phase checkpoint] {path}")
        return path

    def load_checkpoint(
        self,
        path: str,
        *,
        strict: bool = True,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        path = os.path.abspath(str(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("checkpoint payload must be a mapping")
        phase = int(checkpoint.get("phase", -1))
        if phase not in self.phase_schedule:
            raise RuntimeError(f"checkpoint phase {phase} is outside the dataset schedule")
        if int(checkpoint.get("format_version", -1)) != self.CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError("unsupported checkpoint format")
        runtime = checkpoint.get("runtime_contract")
        if not isinstance(runtime, Mapping):
            raise RuntimeError("checkpoint runtime contract is missing")
        if strict:
            self._assert_runtime_contract(runtime)

        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise RuntimeError("checkpoint model_state_dict is missing")
        self._preload_geometry_bank_buffers(model_state)
        self.model.load_state_dict(model_state, strict=strict)

        expected_seen = self._cumulative_classes(phase)
        saved_seen = [int(value) for value in checkpoint.get("seen_classes", [])]
        if strict and saved_seen != expected_seen:
            raise RuntimeError(
                f"checkpoint seen classes {saved_seen} do not match schedule {expected_seen}"
            )
        self._set_accepted_phase_state(phase)
        memory_report = self._assert_final_phase_memory(phase)

        audit = checkpoint.get("memory_audit")
        if isinstance(audit, Mapping):
            saved_rows_digest = audit.get("rows_digest")
            current_rows_digest = self.model.geometry_bank.rows_digest(expected_seen)
            if strict and saved_rows_digest != current_rows_digest:
                raise RuntimeError("checkpoint row digest failed after loading")
            saved_contract = audit.get("bank_contract_digest")
            current_contract = self.model.classifier.bank_contract_digest(
                self.model.geometry_bank
            )
            if strict and saved_contract != current_contract:
                raise RuntimeError("checkpoint static bank contract failed after loading")

        history = checkpoint.get("history")
        if isinstance(history, Mapping):
            self._last_phase_history = copy.deepcopy(dict(history))
            if phase == 0:
                final_report = history.get("base_geometry_report")
                if isinstance(final_report, Mapping):
                    self._last_base_geometry_report = dict(final_report)
                handoff = history.get("base_handoff")
                if isinstance(handoff, Mapping):
                    self.model.base_handoff = copy.deepcopy(dict(handoff))

        if restore_rng:
            cpu_rng = checkpoint.get("torch_rng_state")
            if torch.is_tensor(cpu_rng):
                torch.set_rng_state(cpu_rng.cpu())
            cuda_rng = checkpoint.get("cuda_rng_state_all")
            if torch.cuda.is_available() and isinstance(cuda_rng, (list, tuple)):
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
        checkpoint = dict(checkpoint)
        checkpoint["loaded_memory_report"] = memory_report
        return checkpoint












# from __future__ import annotations

# """Top-level trainer and checkpoint orchestration for factor-geometry NECIL.

# The class executes phase 0 through ``BasePhaseTrainer``. It is intentionally
# future-ready: a later incremental mixin may provide ``train_incremental_phase``
# without changing the accepted checkpoint format or architecture contract.
# """

# import copy
# import math
# import os
# from typing import Any, Dict, Mapping, Optional

# import torch

# from trainers.base_phase_trainer import BasePhaseTrainer
# from trainers.trainer_helpers import TrainerHelper



# class Trainer(BasePhaseTrainer, TrainerHelper):
#     CHECKPOINT_FORMAT_VERSION = 3

#     def __init__(self, model: torch.nn.Module, dataset: Any, args: Any) -> None:
#         if model is None:
#             raise TypeError("Trainer requires a model")
#         if dataset is None:
#             raise TypeError("Trainer requires a dataset")
#         if args is None:
#             raise TypeError("Trainer requires explicit configuration")

#         self.args = args
#         self.device = self._resolve_device(getattr(args, "device", "cpu"))
#         self.save_dir = os.path.abspath(
#             str(getattr(args, "save_dir", "./outputs"))
#         )
#         os.makedirs(self.save_dir, exist_ok=True)

#         self.model = model.to(self.device)
#         self.dataset = dataset
#         self.debug = self._parse_bool(
#             getattr(args, "debug_verbose", False), "debug_verbose"
#         )
#         self._last_base_geometry_report: Optional[Dict[str, Any]] = None
#         self._last_base_stats: Dict[str, float] = {}

#         self.assert_architecture_contract()
#         dataset_contract = self.assert_dataset_contract()
#         self.phase_schedule = dict(dataset_contract["schedule"])
#         self.base_classes = list(dataset_contract["base_classes"])
#         self._prepare_initial_state()

#     # ------------------------------------------------------------------
#     # Device and accepted state
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _parse_bool(value: Any, name: str) -> bool:
#         if isinstance(value, bool):
#             return value
#         if isinstance(value, int) and value in (0, 1):
#             return bool(value)
#         if isinstance(value, str):
#             token = value.strip().lower()
#             if token in {"1", "true", "yes", "y", "on"}:
#                 return True
#             if token in {"0", "false", "no", "n", "off"}:
#                 return False
#         raise RuntimeError(f"{name} must be an explicit boolean")

#     @staticmethod
#     def _resolve_device(value: Any) -> torch.device:
#         token = str(value).strip().lower()
#         if token == "gpu":
#             token = "cuda"
#         requested = torch.device(token)
#         if requested.type == "cpu":
#             return requested
#         if requested.type != "cuda":
#             raise RuntimeError(f"unsupported device type {requested.type!r}")
#         if not torch.cuda.is_available():
#             raise RuntimeError("CUDA was requested but is unavailable")
#         index = (
#             torch.cuda.current_device()
#             if requested.index is None
#             else requested.index
#         )
#         if not 0 <= index < torch.cuda.device_count():
#             raise RuntimeError(f"invalid CUDA device index {index}")
#         torch.cuda.set_device(index)
#         return torch.device(f"cuda:{index}")

#     def _prepare_initial_state(self) -> None:
#         bank = self.model.geometry_bank
#         valid = self.model.infer_seen_classes()
#         if not valid:
#             partial_contract = any(
#                 (
#                     bool(bank.global_priors_ready.item()),
#                     bool(bank.overlap_temperatures_ready.item()),
#                     bool(self.model.classifier.require_bound_contract),
#                     len(bank) > 0,
#                 )
#             )
#             if partial_contract:
#                 raise RuntimeError(
#                     "the model contains a partial base geometry state without "
#                     "valid committed rows; start fresh or load an accepted checkpoint"
#                 )
#             self.model.set_base_mode()
#             return

#         if set(valid) != set(self.base_classes):
#             raise RuntimeError(
#                 f"existing rows {valid} do not match base classes {self.base_classes}"
#             )
#         self._set_final_base_state()
#         self.assert_final_base_memory(self.base_classes)

#     def _set_final_base_state(self) -> None:
#         ids = list(self.base_classes)
#         self.model.current_phase = 0
#         self.model.phase_mode = "evaluation"
#         self.model.seen_classes = list(ids)
#         self.model.old_classes = list(ids)
#         self.model.new_classes = []
#         self.model.phase_old_digest = None
#         self.model.backbone.freeze_all()
#         self.model.classifier.require_bound_contract = True
#         self.model.eval()

#     # ------------------------------------------------------------------
#     # Phase routing
#     # ------------------------------------------------------------------

#     def train_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#     ) -> Dict[str, Any]:
#         phase = int(phase)
#         if phase not in self.phase_schedule:
#             raise ValueError(
#                 f"phase {phase} is outside dataset schedule {sorted(self.phase_schedule)}"
#             )
#         if int(epochs) <= 0:
#             raise ValueError("epochs must be positive")
#         if int(batch_size) <= 0:
#             raise ValueError("batch_size must be positive")
#         if not math.isfinite(float(lr)) or float(lr) <= 0.0:
#             raise ValueError("learning rate must be finite and positive")

#         self.assert_architecture_contract()
#         self.assert_dataset_contract()

#         if phase == 0:
#             if bool(self.model.geometry_bank.valid_mask().any().item()):
#                 raise RuntimeError("phase 0 is already finalized")
#             self.model.set_base_mode()
#             self._print_execution_summary(phase=0)
#             return self.train_base_phase(
#                 phase=0,
#                 epochs=int(epochs),
#                 batch_size=int(batch_size),
#                 lr=float(lr),
#             )

#         incremental = getattr(self, "train_incremental_phase", None)
#         if not callable(incremental):
#             raise RuntimeError(
#                 "incremental phase routing is ready, but no coherent "
#                 "IncrementalPhaseTrainer is installed. Do not fall back to "
#                 "the retired trainable-transport/replay trainer."
#             )
#         self._print_execution_summary(phase=phase)
#         return incremental(
#             phase=phase,
#             epochs=int(epochs),
#             batch_size=int(batch_size),
#             lr=float(lr),
#         )

#     def _print_execution_summary(self, *, phase: int) -> None:
#         bank = self.model.geometry_bank
#         parameters = sum(
#             int(parameter.numel()) for parameter in self.model.parameters()
#         )
#         print(
#             "[Trainer] "
#             f"phase={phase} | device={self.device} | parameters={parameters:,} | "
#             f"token_dim={self.model.backbone.token_dim} | "
#             f"spectral_dim={self.model.spectral_dim} | "
#             f"spatial_dim={self.model.spatial_dim} | "
#             f"feature_dim={self.model.feature_dim} | "
#             f"maximum_rank={bank.maximum_rank} | "
#             f"raw_bands={bank.raw_spectral_dim}"
#         )
#         if phase == 0:
#             print(
#                 "[Base objective] class-balanced CE warm-up + bidirectional "
#                 "cross-fitted risk-guided factor-energy shaping."
#             )
#             print(
#                 "[Base memory] zero persistent class rows during optimization; "
#                 "one atomic p(z|c) handoff."
#             )
#         else:
#             print(
#                 "[Incremental contract] temporary phase-start observer + "
#                 "analytical branch transport + real-current-query geometry; "
#                 "no feature replay or trainable transport."
#             )

#     # ------------------------------------------------------------------
#     # Checkpoint contract
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _clone(value: Any) -> Any:
#         if torch.is_tensor(value):
#             return value.detach().cpu().clone()
#         if isinstance(value, Mapping):
#             return {str(key): Trainer._clone(item) for key, item in value.items()}
#         if isinstance(value, tuple):
#             return tuple(Trainer._clone(item) for item in value)
#         if isinstance(value, list):
#             return [Trainer._clone(item) for item in value]
#         return copy.deepcopy(value)

#     def _runtime_contract(self) -> Dict[str, Any]:
#         model = self.model
#         bank = model.geometry_bank
#         backbone = model.backbone
#         return {
#             "format_version": self.CHECKPOINT_FORMAT_VERSION,
#             "classification_factorization": self.CLASSIFICATION_FACTORIZATION,
#             "spectral_relation_factorization": self.SPECTRAL_RELATION_FACTORIZATION,
#             "architecture_version": int(model.ARCHITECTURE_VERSION),
#             "backbone_contract_version": int(backbone.CONTRACT_VERSION),
#             "bank_schema_version": int(bank.SCHEMA_VERSION),
#             "model_input_bands": int(backbone.model_input_bands),
#             "raw_spectral_dim": int(bank.raw_spectral_dim),
#             "patch_size": int(backbone.patch_size),
#             "token_dim": int(backbone.token_dim),
#             "spectral_dim": int(model.spectral_dim),
#             "spatial_dim": int(model.spatial_dim),
#             "feature_dim": int(model.feature_dim),
#             "maximum_rank": int(bank.maximum_rank),
#             "spectral_shape_dim": int(bank.spectral_shape_dim),
#             "spectral_resample_length": int(bank.spectral_resample_length),
#             "volume_weight": float(bank.volume_weight),
#             "variance_floor_absolute": float(bank.variance_floor_absolute),
#             "variance_floor_relative": float(bank.variance_floor_relative),
#             "classifier_temperature": float(model.classifier.temperature),
#             "joint_feature": "direct_[z_s;z_p]",
#             "trainable_transport_network": False,
#             "geometry_replay_training": False,
#             "stores_exemplars": False,
#             "stores_old_features": False,
#             "stores_old_spectra": False,
#             "uses_knowledge_distillation": False,
#             "future_encoder_policy": "frozen_baseline_or_controlled_plasticity",
#         }

#     def _assert_runtime_contract(self, runtime: Mapping[str, Any]) -> None:
#         current = self._runtime_contract()
#         errors = []
#         for key, current_value in current.items():
#             saved = runtime.get(key)
#             if isinstance(current_value, float):
#                 try:
#                     equal = math.isclose(
#                         float(saved), current_value, rel_tol=0.0, abs_tol=1e-12
#                     )
#                 except (TypeError, ValueError):
#                     equal = False
#             else:
#                 equal = saved == current_value
#             if not equal:
#                 errors.append(
#                     f"{key}: checkpoint={saved!r}, current={current_value!r}"
#                 )
#         if errors:
#             raise RuntimeError("checkpoint runtime contract mismatch: " + "; ".join(errors))

#     def _compact_memory_audit(self) -> Dict[str, Any]:
#         snapshot = dict(self.model.memory_snapshot())
#         snapshot.pop("bank", None)  # model_state_dict already contains the bank.
#         snapshot["memory_cost"] = self.model.geometry_bank.memory_cost_summary()
#         return snapshot

#     def _preload_geometry_bank_buffers(
#         self,
#         model_state: Mapping[str, Any],
#     ) -> None:
#         """Resize dynamic row buffers before strict ``load_state_dict``.

#         Sparse global class IDs make row buffers phase-dependent. A fresh bank
#         has zero rows, so normal PyTorch copying would report shape mismatches.
#         ``load_snapshot`` replaces all buffers first; strict model loading then
#         validates the complete state normally.
#         """
#         bank = self.model.geometry_bank
#         snapshot: Dict[str, torch.Tensor] = {}
#         missing = []
#         for name in bank._buffers:
#             key = f"geometry_bank.{name}"
#             value = model_state.get(key)
#             if not torch.is_tensor(value):
#                 missing.append(key)
#             else:
#                 snapshot[name] = value
#         if missing:
#             raise RuntimeError(
#                 "checkpoint GeometryBank buffers are incomplete: "
#                 f"{missing[:12]}"
#             )
#         bank.load_snapshot(snapshot, strict=True)

#     def save_checkpoint(
#         self,
#         phase: int,
#         history: Mapping[str, Any],
#         evaluator_metrics: Optional[Mapping[str, Any]] = None,
#     ) -> str:
#         if int(phase) != 0:
#             raise RuntimeError(
#                 "this checkpoint writer currently accepts only a finalized base phase"
#             )
#         if not isinstance(history, Mapping):
#             raise TypeError("history must be a mapping")

#         memory_report = self.assert_final_base_memory(self.base_classes)
#         final_report = history.get("base_geometry_report")
#         if not isinstance(final_report, Mapping):
#             raise RuntimeError("history.base_geometry_report is missing")
#         certificate = self.base_geometry_certificate(final_report)
#         if not certificate["checks"]["structural_geometry_valid"]:
#             raise RuntimeError("refusing to save structurally invalid base geometry")
#         enforce = self._parse_bool(
#             getattr(self.args, "base_admission_enforce", False),
#             "base_admission_enforce",
#         )
#         if enforce and not bool(certificate["passed"]):
#             raise RuntimeError(
#                 "refusing to save a base checkpoint that failed enforced gates: "
#                 f"{certificate['failed_checks']}"
#             )

#         phase_dir = os.path.join(self.save_dir, "phase_0")
#         os.makedirs(phase_dir, exist_ok=True)
#         path = os.path.join(phase_dir, "checkpoint.pth")
#         temporary = path + ".tmp"
#         checkpoint: Dict[str, Any] = {
#             "format_version": self.CHECKPOINT_FORMAT_VERSION,
#             "checkpoint_kind": "accepted_base_phase",
#             "phase": 0,
#             "runtime_contract": self._runtime_contract(),
#             "model_state_dict": {
#                 key: self._clone(value)
#                 for key, value in self.model.state_dict().items()
#             },
#             "memory_audit": self._clone(self._compact_memory_audit()),
#             "base_classes": list(self.base_classes),
#             "memory_report": self._clone(memory_report),
#             "geometry_certificate": self._clone(certificate),
#             "history": self._clone(dict(history)),
#             "args": dict(vars(self.args)) if hasattr(self.args, "__dict__") else {},
#             "torch_rng_state": torch.get_rng_state().cpu(),
#         }
#         if torch.cuda.is_available():
#             checkpoint["cuda_rng_state_all"] = [
#                 state.cpu() for state in torch.cuda.get_rng_state_all()
#             ]
#         if evaluator_metrics is not None:
#             checkpoint["evaluator_metrics"] = self._clone(dict(evaluator_metrics))

#         try:
#             with open(temporary, "wb") as stream:
#                 torch.save(checkpoint, stream)
#                 stream.flush()
#                 os.fsync(stream.fileno())
#             os.replace(temporary, path)
#         except Exception:
#             if os.path.exists(temporary):
#                 os.remove(temporary)
#             raise
#         print(f"[Saved base checkpoint] {path}")
#         return path

#     def load_checkpoint(
#         self,
#         path: str,
#         *,
#         strict: bool = True,
#         restore_rng: bool = True,
#     ) -> Dict[str, Any]:
#         path = os.path.abspath(str(path))
#         if not os.path.isfile(path):
#             raise FileNotFoundError(path)
#         try:
#             checkpoint = torch.load(
#                 path, map_location=self.device, weights_only=False
#             )
#         except TypeError:
#             checkpoint = torch.load(path, map_location=self.device)
#         if not isinstance(checkpoint, Mapping):
#             raise RuntimeError("checkpoint payload must be a mapping")
#         if int(checkpoint.get("phase", -1)) != 0:
#             raise RuntimeError("this loader expects an accepted base checkpoint")
#         if int(checkpoint.get("format_version", -1)) != self.CHECKPOINT_FORMAT_VERSION:
#             raise RuntimeError("unsupported checkpoint format")
#         runtime = checkpoint.get("runtime_contract")
#         if not isinstance(runtime, Mapping):
#             raise RuntimeError("checkpoint runtime contract is missing")
#         if strict:
#             self._assert_runtime_contract(runtime)

#         model_state = checkpoint.get("model_state_dict")
#         if not isinstance(model_state, Mapping):
#             raise RuntimeError("checkpoint model_state_dict is missing")
#         self._preload_geometry_bank_buffers(model_state)
#         self.model.load_state_dict(model_state, strict=strict)

#         saved_classes = [int(value) for value in checkpoint.get("base_classes", [])]
#         if strict and saved_classes != list(self.base_classes):
#             raise RuntimeError(
#                 f"checkpoint classes {saved_classes} do not match dataset classes "
#                 f"{self.base_classes}"
#             )
#         self._set_final_base_state()
#         memory_report = self.assert_final_base_memory(self.base_classes)

#         audit = checkpoint.get("memory_audit")
#         if isinstance(audit, Mapping):
#             saved_rows_digest = audit.get("rows_digest")
#             current_rows_digest = self.model.geometry_bank.rows_digest(
#                 self.base_classes
#             )
#             if strict and saved_rows_digest != current_rows_digest:
#                 raise RuntimeError("checkpoint row digest failed after loading")
#             saved_contract = audit.get("bank_contract_digest")
#             current_contract = self.model.classifier.bank_contract_digest(
#                 self.model.geometry_bank
#             )
#             if strict and saved_contract != current_contract:
#                 raise RuntimeError("checkpoint static bank contract failed after loading")

#         history = checkpoint.get("history")
#         if isinstance(history, Mapping):
#             final_report = history.get("base_geometry_report")
#             if isinstance(final_report, Mapping):
#                 self._last_base_geometry_report = dict(final_report)
#             handoff = history.get("base_handoff")
#             if isinstance(handoff, Mapping):
#                 self.model.base_handoff = copy.deepcopy(dict(handoff))
#         self._last_base_stats = {
#             "valid_rows": float(len(memory_report["valid_rows"])),
#         }

#         if restore_rng:
#             cpu_rng = checkpoint.get("torch_rng_state")
#             if torch.is_tensor(cpu_rng):
#                 torch.set_rng_state(cpu_rng.cpu())
#             cuda_rng = checkpoint.get("cuda_rng_state_all")
#             if torch.cuda.is_available() and isinstance(cuda_rng, (list, tuple)):
#                 torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
#         return dict(checkpoint)






# from __future__ import annotations

# import hashlib
# import json
# import math
# import os
# import tempfile
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# import torch

# from trainers.base_phase_trainer import BasePhaseTrainer
# from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
# from trainers.trainer_helpers import TrainerHelper


# _MISSING = object()


# class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
#     """Strict PC-STGB phase orchestrator and checkpoint authority.

#     The class routes phase 0 to the validated base trainer and later phases to
#     the schema-v5 conditional incremental trainer.  It is the checkpoint and
#     runtime-contract authority for the complete strict non-exemplar run.

#     The trainer owns orchestration and persistence.  Geometry estimation remains
#     in :class:`GeometryBank`, scoring remains in
#     :class:`GeometryEnergyClassifier`, and phase-specific objectives remain in
#     the base/incremental trainer classes.
#     """

#     CHECKPOINT_FORMAT_VERSION = 2
#     ARCHITECTURE_ID = "pc_stgb_conditional_joint_necil_v5"
#     METHOD_NAME = "PC-STGB"
#     BANK_SCHEMA_VERSION = 5
#     JOINT_FACTORIZATION = "p(z|c)prod_k p(g_k|z,c)"

#     def __init__(self, model: torch.nn.Module, dataset: Any, args: Any) -> None:
#         if args is None:
#             raise TypeError("Trainer requires an explicit configuration object")
#         if model is None:
#             raise TypeError("Trainer requires a model")
#         if dataset is None:
#             raise TypeError("Trainer requires a dataset")

#         self.args = args
#         self.device = self._resolve_device(self._require_arg("device"))
#         self.save_dir = os.path.abspath(str(self._require_arg("save_dir")))
#         self.debug = self._parse_bool(
#             self._optional_arg("debug_verbose", False), "debug_verbose"
#         )
#         self.model = model.to(self.device)
#         self.dataset = dataset
#         os.makedirs(self.save_dir, exist_ok=True)

#         self._last_base_geometry_report: Optional[Dict[str, Any]] = None
#         self._last_base_geometry_certificate: Optional[Dict[str, Any]] = None
#         self._last_base_stats: Dict[str, float] = {}
#         self._base_ce_head: Optional[torch.nn.Module] = None

#         self.assert_architecture_contract()
#         self._assert_dataset_response_contract()
#         self._assert_base_stack_contract()
#         self._prepare_initial_model_state()

#     # ------------------------------------------------------------------
#     # Shared configuration dispatch
#     # ------------------------------------------------------------------
#     # BasePhaseTrainer accepts one or more option aliases and a keyword
#     # default. TrainerHelper historically passes the default positionally.
#     # These wrappers support both forms without MRO-dependent behavior.
#     @staticmethod
#     def _split_cfg_call(
#         names: Sequence[Any], default: Any
#     ) -> tuple[tuple[str, ...], Any]:
#         values = list(names)
#         if values and not isinstance(values[-1], str):
#             if default is not _MISSING:
#                 raise RuntimeError("configuration default was supplied twice")
#             default = values.pop()
#         if not values or not all(isinstance(name, str) for name in values):
#             raise RuntimeError("configuration option names must be strings")
#         return tuple(values), default

#     def _cfg_bool(self, *names: Any, default: Any = _MISSING) -> bool:
#         resolved_names, resolved_default = self._split_cfg_call(names, default)
#         value = BasePhaseTrainer._cfg_any(
#             self, resolved_names, default=resolved_default
#         )
#         return self._parse_bool(value, resolved_names[0])

#     def _cfg_float(self, *names: Any, default: Any = _MISSING) -> float:
#         resolved_names, resolved_default = self._split_cfg_call(names, default)
#         value = BasePhaseTrainer._cfg_any(
#             self, resolved_names, default=resolved_default
#         )
#         result = float(value)
#         if not math.isfinite(result):
#             raise RuntimeError(f"{resolved_names[0]} must be finite")
#         return result

#     def _cfg_int(self, *names: Any, default: Any = _MISSING) -> int:
#         resolved_names, resolved_default = self._split_cfg_call(names, default)
#         value = BasePhaseTrainer._cfg_any(
#             self, resolved_names, default=resolved_default
#         )
#         if isinstance(value, bool):
#             raise RuntimeError(
#                 f"{resolved_names[0]} must be an integer, not bool"
#             )
#         result = int(value)
#         if float(value) != float(result):
#             raise RuntimeError(f"{resolved_names[0]} must be an integer")
#         return result

#     # ------------------------------------------------------------------
#     # Configuration and device
#     # ------------------------------------------------------------------
#     def _require_arg(self, name: str) -> Any:
#         if not hasattr(self.args, name):
#             raise RuntimeError(f"Missing required trainer option {name!r}")
#         value = getattr(self.args, name)
#         if value is None:
#             raise RuntimeError(f"Trainer option {name!r} cannot be None")
#         return value

#     def _optional_arg(self, name: str, default: Any) -> Any:
#         value = getattr(self.args, name, _MISSING)
#         return default if value is _MISSING or value is None else value

#     @staticmethod
#     def _resolve_device(value: Any) -> torch.device:
#         token = str(value).strip().lower()
#         if token == "gpu":
#             token = "cuda"
#         try:
#             requested = torch.device(token)
#         except (TypeError, RuntimeError, ValueError) as exc:
#             raise RuntimeError(
#                 f"Invalid device {value!r}; use cpu, cuda, or cuda:<index>"
#             ) from exc
#         if requested.type == "cpu":
#             return torch.device("cpu")
#         if requested.type != "cuda":
#             raise RuntimeError(f"Unsupported device type {requested.type!r}")
#         if not torch.cuda.is_available():
#             raise RuntimeError(
#                 f"device={value!r} requests CUDA, but CUDA is unavailable"
#             )
#         count = int(torch.cuda.device_count())
#         index = (
#             int(torch.cuda.current_device())
#             if requested.index is None
#             else int(requested.index)
#         )
#         if index < 0 or index >= count:
#             raise RuntimeError(
#                 f"Requested cuda:{index}, but only {count} device(s) are visible"
#             )
#         torch.cuda.set_device(index)
#         return torch.device(f"cuda:{index}")

#     @staticmethod
#     def _parse_bool(value: Any, name: str) -> bool:
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

#     @staticmethod
#     def _require_object_attr(owner: Any, name: str, owner_name: str) -> Any:
#         if owner is None or not hasattr(owner, name):
#             raise RuntimeError(f"{owner_name} must expose {name!r}")
#         value = getattr(owner, name)
#         if value is None:
#             raise RuntimeError(f"{owner_name}.{name} cannot be None")
#         return value

#     def _optional_bool(self, name: str, default: bool) -> bool:
#         return self._parse_bool(self._optional_arg(name, default), name)

#     def _optional_float(self, name: str, default: float) -> float:
#         value = float(self._optional_arg(name, default))
#         if not math.isfinite(value):
#             raise RuntimeError(f"{name} must be finite, got {value!r}")
#         return value

#     @staticmethod
#     def _normalized_mode(value: Any) -> str:
#         token = str(value or "pc_stgb_row_replay").strip().lower().replace("-", "_")
#         if token in {"pc_sirg", "pc_sirg_row_replay", "joint_geometry"}:
#             return "pc_stgb_row_replay"
#         return token

#     def _response_views_from_batch(
#         self,
#         batch: Any,
#         x: torch.Tensor,
#         *,
#         fine: bool = False,
#         required: bool = True,
#         context: str = "response_views",
#     ):
#         """Unified base/helper intervention-view parser.

#         Base training requests optional fine h/2 views, while shared helper
#         extraction supplies a diagnostic context.  One method supports both
#         contracts and validates the intervention-definition version.
#         """
#         result = BasePhaseTrainer._response_views_from_batch(
#             self, batch, x, fine=bool(fine), required=bool(required)
#         )
#         if result is None or fine or not isinstance(batch, Mapping):
#             return result
#         version = None
#         for source in self._view_sources(batch):
#             if version is None:
#                 version = source.get(
#                     "intervention_definition_version",
#                     source.get(
#                         "pc_stgb_intervention_version",
#                         source.get("pc_sirg_intervention_version"),
#                     ),
#                 )
#         if version is not None:
#             observed = int(torch.as_tensor(version).reshape(-1)[0].item())
#             expected = int(self.model.intervention_definition_version)
#             if observed != expected:
#                 raise RuntimeError(
#                     f"{context}: intervention version={observed} "
#                     f"!= model version={expected}"
#                 )
#         return result

#     # ------------------------------------------------------------------
#     # Architecture contract
#     # ------------------------------------------------------------------
#     def assert_architecture_contract(self) -> None:
#         checker = getattr(self.model, "assert_architecture_contract", None)
#         if not callable(checker):
#             raise RuntimeError("Model must expose assert_architecture_contract()")
#         checker()

#         contract_method = getattr(self.model, "feature_contract", None)
#         if not callable(contract_method):
#             raise RuntimeError("Model must expose feature_contract()")
#         contract = contract_method()
#         if not isinstance(contract, Mapping):
#             raise RuntimeError("model.feature_contract() must return a mapping")

#         expected = {
#             "method": self.METHOD_NAME,
#             "geometry_bank_schema_version": self.BANK_SCHEMA_VERSION,
#             "geometry_feature_space":
#                 "unnormalized_euclidean_residual_projected_z",
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "classifier_contract":
#                 "dimension_normalized_conditional_joint_energy",
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
#         }
#         mismatched = {
#             key: (contract.get(key), target)
#             for key, target in expected.items()
#             if contract.get(key) != target
#         }
#         if mismatched:
#             raise RuntimeError(f"PC-STGB feature contract mismatch: {mismatched}")

#         configured_mode = self._normalized_mode(
#             self._optional_arg("incremental_update_mode", "pc_stgb_row_replay")
#         )
#         model_mode = self._normalized_mode(
#             contract.get(
#                 "incremental_update_mode",
#                 getattr(self.model, "incremental_update_mode", ""),
#             )
#         )
#         if configured_mode != "pc_stgb_row_replay" or model_mode != configured_mode:
#             raise RuntimeError(
#                 "args and model must use incremental_update_mode="
#                 f"'pc_stgb_row_replay'; args={configured_mode!r}, "
#                 f"model={model_mode!r}"
#             )

#         bank = self._require_object_attr(self.model, "geometry_bank", "model")
#         classifier = self._require_object_attr(self.model, "classifier", "model")
#         if int(getattr(bank, "SCHEMA_VERSION", -1)) != self.BANK_SCHEMA_VERSION:
#             raise RuntimeError("Trainer requires the schema-v5 PC-STGB GeometryBank")

#         num_interventions = int(contract.get("num_interventions", 0))
#         response_rank = int(contract.get("response_rank", 0))
#         if num_interventions <= 0 or response_rank <= 0:
#             raise RuntimeError(
#                 "PC-STGB requires positive intervention count and tangent rank"
#             )
#         if int(bank.num_interventions) != num_interventions:
#             raise RuntimeError("Model and GeometryBank intervention counts differ")
#         if int(bank.response_rank) != response_rank:
#             raise RuntimeError("Model and GeometryBank tangent ranks differ")
#         if int(
#             self._optional_arg("num_spectral_interventions", num_interventions)
#         ) != num_interventions:
#             raise RuntimeError("CLI and model intervention counts differ")

#         bank_weight = float(bank.response_weight)
#         classifier_weight = float(classifier.response_weight)
#         contract_weight = float(contract.get("response_weight", bank_weight))
#         if max(
#             abs(bank_weight - classifier_weight),
#             abs(bank_weight - contract_weight),
#         ) > 1e-12:
#             raise RuntimeError(
#                 "Model, GeometryBank, and classifier tangent weights differ"
#             )
#         if bank_weight <= 0.0:
#             raise RuntimeError("spectral_response_weight must be positive")
#         if not bool(classifier.normalize_energy_by_dim):
#             raise RuntimeError("Classifier must use dimension-normalized energy")
#         if float(bank.energy_logdet_weight) <= 0.0:
#             raise RuntimeError("Occupancy energy requires a positive log-volume weight")
#         if float(classifier.response_logdet_weight) <= 0.0:
#             raise RuntimeError("Tangent energy requires a positive log-volume weight")

#         classifier_contract = classifier.classifier_contract()
#         classifier_expected = {
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "uses_coupling_inference_score": True,
#             "uses_independent_response_factorization": False,
#             "uses_raw_spectral_gaussian": False,
#             "uses_calibration": False,
#         }
#         classifier_failures = {
#             key: (classifier_contract.get(key), value)
#             for key, value in classifier_expected.items()
#             if classifier_contract.get(key) != value
#         }
#         if classifier_failures:
#             raise RuntimeError(
#                 "PC-STGB classifier contract mismatch: "
#                 f"{classifier_failures}"
#             )

#         if self._optional_bool("normalize_geometry_features", False):
#             raise RuntimeError("PC-STGB forbids normalized geometry features")

#         forbidden_flags = (
#             "use_incremental_adapter",
#             "use_geometry_gated_adapter",
#             "use_geometry_transport",
#             "use_sglat_transport",
#             "allow_old_model_transport",
#             "use_energy_calibrator",
#             "use_geometry_calibrator",
#             "use_adaptive_boundary",
#             "allow_incremental_projection_training",
#             "use_raw_spectral_gaussian",
#             "use_spectral_feature_coupling",
#             "use_independent_geometry_replay",
#             "use_independent_response_replay",
#         )
#         active_flags = [
#             name
#             for name in forbidden_flags
#             if hasattr(self.args, name)
#             and getattr(self.args, name) is not None
#             and self._parse_bool(getattr(self.args, name), name)
#         ]
#         if active_flags:
#             raise RuntimeError(
#                 f"Retired or contradictory architecture branches are enabled: "
#                 f"{active_flags}"
#             )

#         retired_weights = (
#             "base_gics_weight",
#             "pgr_weight",
#             "pgr_geometry_margin_weight",
#             "pgr_clearance_weight",
#             "pgr_condition_weight",
#             "pgr_spectral_risk_weight",
#             "base_admission_weight",
#             "base_spectral_fisher_weight",
#             "base_joint_coupling_weight",
#         )
#         active_weights = [
#             name
#             for name in retired_weights
#             if hasattr(self.args, name)
#             and getattr(self.args, name) is not None
#             and abs(float(getattr(self.args, name))) > 1e-12
#         ]
#         if active_weights:
#             raise RuntimeError(
#                 "Retired loss weights must be removed or set to zero: "
#                 f"{active_weights}"
#             )

#     def _assert_dataset_response_contract(self) -> None:
#         if not self._optional_bool("base_require_response_views", True):
#             raise RuntimeError("base_require_response_views must be true")

#         capability_names = (
#             "has_spectral_response_views",
#             "return_spectral_response_views",
#             "return_spectral_intervention_views",
#             "spectral_interventions_enabled",
#         )
#         observed = False
#         for name in capability_names:
#             if not hasattr(self.dataset, name):
#                 continue
#             observed = True
#             value = getattr(self.dataset, name)
#             enabled = bool(value() if callable(value) else value)
#             if not enabled:
#                 raise RuntimeError(
#                     f"Dataset {name} is false, but paired +/- tangent views "
#                     "are required"
#                 )
#         if not observed and self.debug:
#             print(
#                 "[Trainer] No static tangent-view capability flag was found; "
#                 "the first batch will be validated strictly."
#             )

#     def _assert_base_stack_contract(self) -> None:
#         required_model = (
#             "feature_contract",
#             "assert_architecture_contract",
#             "extract_joint_geometry_tuple",
#             "compute_logits_from_features",
#             "ensure_base_ce_head",
#             "drop_base_ce_head",
#             "build_candidate_geometry_rows",
#             "score_candidate_geometry_rows",
#             "finalize_base_geometry",
#             "sample_geometry_replay",
#             "geometry_diagnostics",
#             "export_memory_snapshot",
#             "load_memory_snapshot",
#             "model_contract_digest",
#             "set_base_mode",
#             "set_incremental_mode",
#             "commit_candidate_geometry_rows",
#             "assert_frozen_modules",
#             "set_phase",
#         )
#         missing_model = [
#             name
#             for name in required_model
#             if not callable(getattr(self.model, name, None))
#         ]
#         if missing_model:
#             raise RuntimeError(
#                 f"PC-STGB model contract is incomplete: {missing_model}"
#             )

#         bank = self._require_object_attr(self.model, "geometry_bank", "model")
#         required_bank = (
#             "get_bank",
#             "get_valid_mask",
#             "assert_bank_valid",
#             "assert_phase0_base_handoff_ready",
#             "assert_response_prior_ready",
#             "contract_digest",
#             "snapshot_rows",
#             "assert_rows_identical",
#             "rows_digest",
#             "export_snapshot",
#             "load_snapshot",
#             "phase_geometry_state_report",
#             "compute_geometry_diagnostics",
#             "refine_candidate_joint_rows",
#             "candidate_joint_energy_matrix",
#             "commit_incremental_geometry_rows",
#         )
#         missing_bank = [
#             name for name in required_bank if not callable(getattr(bank, name, None))
#         ]
#         if missing_bank:
#             raise RuntimeError(
#                 f"PC-STGB GeometryBank contract is incomplete: {missing_bank}"
#             )

#         required_state = (
#             "means",
#             "bases",
#             "eigvals",
#             "res_vars",
#             "active_ranks",
#             "sample_counts",
#             "response_bases",
#             "response_means",
#             "response_eigvals",
#             "response_res_vars",
#             "response_active_ranks",
#             "response_sample_counts",
#             "response_reliability",
#             "response_stats_ready",
#             "response_couplings",
#             "response_coupling_reliability",
#             "response_coupling_explained_variance",
#             "response_coupling_ready",
#             "response_prior_mean",
#             "response_prior_variance",
#             "response_prior_sample_count",
#             "response_prior_ready",
#             "response_prior_frozen",
#             "energy_stats_ready",
#             "margin_stats_ready",
#             "phase_created",
#             "frozen_class_mask",
#         )
#         missing_state = [
#             name
#             for name in required_state
#             if not torch.is_tensor(getattr(bank, name, None))
#         ]
#         if missing_state:
#             raise RuntimeError(
#                 "PC-STGB GeometryBank persistent state is incomplete: "
#                 f"{missing_state}"
#             )

#         classifier = self.model.classifier
#         required_classifier = (
#             "classifier_contract",
#             "classifier_contract_digest",
#             "bind_geometry_bank_contract",
#             "compute_joint_logits",
#             "compute_candidate_joint_logits",
#             "expand_to_seen_classes",
#         )
#         missing_classifier = [
#             name
#             for name in required_classifier
#             if not callable(getattr(classifier, name, None))
#         ]
#         if missing_classifier:
#             raise RuntimeError(
#                 f"PC-STGB classifier contract is incomplete: {missing_classifier}"
#             )

#     # ------------------------------------------------------------------
#     # Phase routing
#     # ------------------------------------------------------------------
#     def _prepare_initial_model_state(self) -> None:
#         bank = self.model.geometry_bank
#         if len(bank) == 0:
#             self.model.set_phase(0)
#             self.model.set_base_mode(train_backbone=True, train_projection=True)
#             self.model.current_phase = 0
#             self.model.old_class_count = 0
#             self.model.old_classes = []
#             self.model.new_classes = []
#             return

#         valid = torch.nonzero(
#             bank.get_valid_mask(), as_tuple=False
#         ).flatten().detach().cpu().tolist()
#         if not valid:
#             raise RuntimeError(
#                 "Model contains allocated GeometryBank rows without valid class "
#                 "memory. Start phase 0 from an empty bank."
#             )
#         self._set_post_base_state(valid)

#     def _set_post_base_state(self, seen_classes: Iterable[int]) -> None:
#         seen = self._as_class_list(seen_classes, name="post_base_seen_classes")
#         self.model.current_phase = 0
#         self.model.seen_classes = list(seen)
#         self.model.old_classes = list(seen)
#         self.model.new_classes = []
#         self.model.old_class_count = len(seen)
#         self.model.current_num_classes = len(seen)
#         self.model.base_mode_active = False
#         self.model.incremental_mode_active = False
#         self.model.classifier.expand_to_seen_classes(seen)

#         modules = (
#             getattr(self.model, "backbone", None),
#             getattr(self.model, "projection", None),
#             getattr(self.model, "norm", None),
#             getattr(self.model, "classifier", None),
#             getattr(self.model, "base_ce_head", None),
#         )
#         for module in modules:
#             if module is None:
#                 continue
#             module.eval()
#             for parameter in module.parameters():
#                 parameter.requires_grad_(False)
#                 parameter.grad = None
#         residual = getattr(self.model, "projection_residual_logit", None)
#         if isinstance(residual, torch.nn.Parameter):
#             residual.requires_grad_(False)
#             residual.grad = None
#         self.model.eval()

#     def _set_post_phase_state(
#         self, seen_classes: Iterable[int], phase: int
#     ) -> None:
#         seen = self._as_class_list(
#             seen_classes, name="post_phase_seen_classes"
#         )
#         phase = int(phase)
#         self._set_post_base_state(seen)
#         self.model.current_phase = phase
#         self.model.seen_classes = list(seen)
#         self.model.old_classes = list(seen)
#         self.model.new_classes = []
#         self.model.old_class_count = len(seen)
#         self.model.current_num_classes = len(seen)
#         self.model.base_mode_active = False
#         self.model.incremental_mode_active = False

#     def _seen_classes_for_phase(self, phase: int) -> List[int]:
#         phase = int(phase)
#         phase_map = self._require_object_attr(
#             self.dataset, "phase_to_classes", "dataset"
#         )
#         if phase < 0:
#             raise RuntimeError("phase must be non-negative")
#         values: List[int] = []
#         for index in range(phase + 1):
#             try:
#                 current = phase_map[index]
#             except (KeyError, IndexError, TypeError) as exc:
#                 raise RuntimeError(
#                     f"dataset.phase_to_classes has no phase {index}"
#                 ) from exc
#             values.extend(int(value) for value in current)
#         return self._as_class_list(values, name=f"seen_through_phase_{phase}")

#     def _print_execution_summary(self) -> None:
#         bank = self.model.geometry_bank
#         total_parameters = sum(
#             int(parameter.numel()) for parameter in self.model.parameters()
#         )
#         print(
#             f"[Trainer] method={self.METHOD_NAME} | scope=base+incremental | "
#             f"parameters={total_parameters:,} | device={self.device} | "
#             f"occupancy_rank={int(bank.rank)} | "
#             f"tangent_rank={int(bank.response_rank)} | "
#             f"interventions={int(bank.num_interventions)} | "
#             f"tangent_weight={float(bank.response_weight):.4f}"
#         )
#         print(
#             "[Base Objective] temporary CE + held-out conditional joint-energy "
#             "consolidation + real-sample lower-boundary certification."
#         )
#         print(
#             "[Base Memory] zero persistent rows during optimization; one final "
#             "frozen-prior atomic reconstruction and classifier binding."
#         )

#     def train_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#     ) -> Dict[str, Any]:
#         phase = int(phase)
#         epochs = int(epochs)
#         batch_size = int(batch_size)
#         lr = float(lr)
#         if phase < 0:
#             raise ValueError("phase must be non-negative")
#         if batch_size <= 0:
#             raise ValueError("batch_size must be positive")
#         if phase == 0:
#             if epochs <= 0:
#                 raise ValueError("base epochs must be positive")
#             if not math.isfinite(lr) or lr <= 0.0:
#                 raise ValueError("base learning rate must be finite and positive")
#         else:
#             if epochs < 0:
#                 raise ValueError("incremental epochs must be non-negative")
#             if not math.isfinite(lr):
#                 raise ValueError("incremental learning-rate argument must be finite")

#         self.assert_architecture_contract()
#         self._assert_dataset_response_contract()
#         self._assert_base_stack_contract()

#         if phase == 0:
#             if len(self.model.geometry_bank) != 0:
#                 raise RuntimeError(
#                     "Cannot retrain phase 0 over an existing persistent bank. "
#                     "Create a fresh model or explicitly reset the complete run."
#                 )
#             self.model.set_phase(0)
#             self.model.set_base_mode(train_backbone=True, train_projection=True)
#             self._print_execution_summary()
#             return self.train_base_phase(
#                 phase=0,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 lr=lr,
#             )

#         return self.train_incremental_phase(
#             phase=phase,
#             epochs=epochs,
#             batch_size=batch_size,
#             lr=lr,
#         )

#     # ------------------------------------------------------------------
#     # Checkpoint contracts and persistence
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _clone(value: Any) -> Any:
#         if torch.is_tensor(value):
#             return value.detach().cpu().clone()
#         if isinstance(value, Mapping):
#             return {str(key): Trainer._clone(item) for key, item in value.items()}
#         if isinstance(value, tuple):
#             return tuple(Trainer._clone(item) for item in value)
#         if isinstance(value, list):
#             return [Trainer._clone(item) for item in value]
#         return value

#     @staticmethod
#     def _digest_update(digest: "hashlib._Hash", value: Any) -> None:
#         if torch.is_tensor(value):
#             tensor = value.detach().cpu().contiguous()
#             digest.update(b"tensor")
#             digest.update(str(tensor.dtype).encode("utf-8"))
#             digest.update(str(tuple(tensor.shape)).encode("utf-8"))
#             if tensor.numel():
#                 digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
#             return
#         if isinstance(value, Mapping):
#             digest.update(b"mapping")
#             for key in sorted(value, key=lambda item: str(item)):
#                 digest.update(str(key).encode("utf-8"))
#                 Trainer._digest_update(digest, value[key])
#             return
#         if isinstance(value, (tuple, list)):
#             digest.update(b"sequence")
#             for item in value:
#                 Trainer._digest_update(digest, item)
#             return
#         digest.update(repr(value).encode("utf-8"))

#     @classmethod
#     def _content_sha256(cls, value: Any) -> str:
#         digest = hashlib.sha256()
#         cls._digest_update(digest, value)
#         return digest.hexdigest()

#     @staticmethod
#     def _file_sha256(path: str) -> str:
#         digest = hashlib.sha256()
#         with open(path, "rb") as stream:
#             for chunk in iter(lambda: stream.read(1024 * 1024), b""):
#                 digest.update(chunk)
#         return digest.hexdigest()

#     def _runtime_contract(self, *, phase: int = 0) -> Dict[str, Any]:
#         phase = int(phase)
#         feature_contract = dict(self.model.feature_contract())
#         bank = self.model.geometry_bank
#         classifier = self.model.classifier
#         bank_digest = (
#             bank.contract_digest()
#             if bool(bank.response_prior_ready.item())
#             else None
#         )
#         return {
#             "format_version": self.CHECKPOINT_FORMAT_VERSION,
#             "architecture_id": self.ARCHITECTURE_ID,
#             "method": self.METHOD_NAME,
#             "schema_version": self.BANK_SCHEMA_VERSION,
#             "joint_factorization": self.JOINT_FACTORIZATION,
#             "phase": phase,
#             "base_only_checkpoint": phase == 0,
#             "feature_contract": feature_contract,
#             "feature_dim": int(bank.feature_dim),
#             "occupancy_rank": int(bank.rank),
#             "tangent_rank": int(bank.response_rank),
#             "num_interventions": int(bank.num_interventions),
#             "intervention_definition_version": int(
#                 bank.intervention_definition_version
#             ),
#             "tangent_weight": float(bank.response_weight),
#             "occupancy_logdet_weight": float(bank.energy_logdet_weight),
#             "tangent_logdet_weight": float(classifier.response_logdet_weight),
#             "normalize_energy_by_dim": bool(classifier.normalize_energy_by_dim),
#             "response_prior_ready": bool(bank.response_prior_ready.item()),
#             "response_prior_frozen": bool(bank.response_prior_frozen.item()),
#             "geometry_bank_contract_digest": bank_digest,
#             "classifier_bound_contract_digest": classifier.bound_contract_digest,
#             "classifier_contract_digest": classifier.classifier_contract_digest(),
#             "model_contract_digest": self.model.model_contract_digest(),
#             "base_objective":
#                 "temporary_ce_plus_conditional_joint_margin_plus_real_boundary_certificate",
#             "classifier": "conditional_joint_occupancy_tangent_energy",
#             "persistent_memory":
#                 "frozen_aggregate_occupancy_coupling_and_tangent_residual_rows",
#             "coupled_replay": True,
#             "inference_uses_conditional_tangent": True,
#             "uses_independent_response_factorization": False,
#             "uses_latent_boundary_witnesses": False,
#             "raw_spectral_gaussian": False,
#             "old_rows_immutable": True,
#             "raw_old_exemplars": False,
#             "stored_old_features": False,
#             "stored_old_responses": False,
#         }

#     def _assert_loaded_runtime_contract(
#         self, runtime: Mapping[str, Any], *, strict: bool
#     ) -> None:
#         if not strict:
#             return
#         errors: List[str] = []
#         current = self._runtime_contract(phase=int(runtime.get("phase", 0)))
#         exact = (
#             "architecture_id",
#             "method",
#             "schema_version",
#             "joint_factorization",
#             "feature_dim",
#             "occupancy_rank",
#             "tangent_rank",
#             "num_interventions",
#             "intervention_definition_version",
#             "normalize_energy_by_dim",
#             "classifier",
#             "persistent_memory",
#             "coupled_replay",
#             "inference_uses_conditional_tangent",
#             "uses_independent_response_factorization",
#             "uses_latent_boundary_witnesses",
#             "raw_spectral_gaussian",
#             "old_rows_immutable",
#         )
#         for key in exact:
#             if runtime.get(key) != current.get(key):
#                 errors.append(
#                     f"{key}={runtime.get(key)!r} != {current.get(key)!r}"
#                 )
#         for key in (
#             "tangent_weight",
#             "occupancy_logdet_weight",
#             "tangent_logdet_weight",
#         ):
#             saved = float(runtime.get(key, float("nan")))
#             if not math.isfinite(saved) or not math.isclose(
#                 saved, float(current[key]), rel_tol=0.0, abs_tol=1e-12
#             ):
#                 errors.append(f"{key}={saved!r} != {current[key]!r}")

#         saved_contract = runtime.get("feature_contract")
#         if not isinstance(saved_contract, Mapping):
#             errors.append("feature_contract is missing")
#         else:
#             current_contract = self.model.feature_contract()
#             for key in (
#                 "contract_version",
#                 "method",
#                 "d_model",
#                 "subspace_rank",
#                 "response_rank",
#                 "num_interventions",
#                 "response_weight",
#                 "intervention_definition_version",
#                 "geometry_bank_schema_version",
#                 "geometry_feature_space",
#                 "joint_factorization",
#                 "classifier_contract",
#                 "incremental_update_mode",
#                 "spectral_object",
#                 "spectral_role",
#                 "inference_geometry",
#                 "raw_spectral_gaussian",
#                 "occupancy_tangent_coupling",
#                 "independent_response_factorization",
#             ):
#                 if saved_contract.get(key) != current_contract.get(key):
#                     errors.append(
#                         f"feature_contract[{key}]={saved_contract.get(key)!r} "
#                         f"!= {current_contract.get(key)!r}"
#                     )
#         if errors:
#             raise RuntimeError(
#                 "PC-STGB checkpoint runtime contract mismatch: "
#                 + "; ".join(errors)
#             )

#     def _validate_final_memory_contract(
#         self,
#         seen_classes: Sequence[int],
#         *,
#         require_statistics: bool = True,
#     ) -> None:
#         self.assert_bank_ready_for_seen_classes(
#             None,
#             seen_classes,
#             require_statistics=require_statistics,
#             require_response=True,
#             require_joint_state=True,
#             require_frozen=True,
#             require_frozen_prior=True,
#             require_bound_contract=True,
#         )
#         self.assert_bank_has_only_allowed_valid_rows(None, seen_classes)
#         bank_digest = self.model.geometry_bank.contract_digest()
#         if self.model.classifier.bound_contract_digest != bank_digest:
#             raise RuntimeError(
#                 "Classifier is not bound to the current GeometryBank contract"
#             )
#         if not bool(self.model.classifier.require_bound_contract):
#             raise RuntimeError("Classifier contract enforcement is disabled")

#     def save_checkpoint(
#         self,
#         phase: int,
#         history: Mapping[str, Any],
#         evaluator_metrics: Optional[Mapping[str, Any]] = None,
#     ) -> str:
#         phase = int(phase)
#         if phase < 0:
#             raise RuntimeError("checkpoint phase must be non-negative")
#         if not isinstance(history, Mapping):
#             raise TypeError("history must be a mapping")
#         seen_classes = self._seen_classes_for_phase(phase)
#         self._validate_final_memory_contract(seen_classes)

#         base_report = history.get("base_geometry_report")
#         if not isinstance(base_report, Mapping):
#             base_report = getattr(self.model, "base_geometry_report", None)
#         base_handoff = history.get("base_handoff")
#         if not isinstance(base_handoff, Mapping):
#             base_handoff = getattr(self.model, "base_handoff", None)
#         if not isinstance(base_report, Mapping):
#             raise RuntimeError("Cannot save without the originating base_geometry_report")
#         if not isinstance(base_handoff, Mapping):
#             raise RuntimeError("Cannot save without the originating base_handoff")
#         if not bool(base_report.get("structural_valid", False)):
#             raise RuntimeError("Cannot save a run built from an invalid base bank")

#         phase_report = self.geometry_phase_certificate(
#             phase,
#             seen_classes,
#             require_statistics=True,
#             require_response=True,
#             require_frozen=True,
#         )
#         if phase_report.get("errors") or not bool(phase_report.get("ok", False)):
#             raise RuntimeError(
#                 f"Cannot save invalid phase-{phase} geometry: "
#                 f"{phase_report.get('errors', [])}"
#             )

#         phase_dir = os.path.join(self.save_dir, f"phase_{phase}")
#         os.makedirs(phase_dir, exist_ok=True)
#         path = os.path.join(phase_dir, "checkpoint.pth")
#         manifest_path = path + ".sha256"

#         memory_snapshot = self._clone(self.model.export_memory_snapshot())
#         memory_digest = self._content_sha256(memory_snapshot)
#         model_state = {
#             key: value.detach().cpu().clone()
#             if torch.is_tensor(value)
#             else self._clone(value)
#             for key, value in self.model.state_dict().items()
#         }

#         dataset_state = None
#         dataset_export = getattr(self.dataset, "export_runtime_state", None)
#         if callable(dataset_export):
#             dataset_state = self._clone(dataset_export())
#         elif self._optional_bool("checkpoint_requires_dataset_state", self._optional_bool("base_checkpoint_requires_dataset_state", False)):
#             raise RuntimeError(
#                 "checkpoint_requires_dataset_state=true, but the dataset "
#                 "does not expose export_runtime_state()"
#             )

#         runtime_contract = self._runtime_contract(phase=phase)
#         checkpoint: Dict[str, Any] = {
#             "format_version": self.CHECKPOINT_FORMAT_VERSION,
#             "phase": phase,
#             "runtime_contract": runtime_contract,
#             "model_state_dict": model_state,
#             "memory_snapshot": memory_snapshot,
#             "memory_snapshot_sha256": memory_digest,
#             "geometry_bank_contract_digest":
#                 self.model.geometry_bank.contract_digest(),
#             "classifier_bound_contract_digest":
#                 self.model.classifier.bound_contract_digest,
#             "classifier_contract_digest":
#                 self.model.classifier.classifier_contract_digest(),
#             "model_contract_digest": self.model.model_contract_digest(),
#             "dataset_runtime_state": dataset_state,
#             "seen_classes": list(seen_classes),
#             "history": self._clone(dict(history)),
#             "base_geometry_report": self._clone(dict(base_report)),
#             "base_handoff": self._clone(dict(base_handoff)),
#             "phase_geometry_report": self._clone(dict(phase_report)),
#             "incremental_admission_certificate": self._clone(
#                 history.get("admission_certificate")
#             ) if isinstance(history.get("admission_certificate"), Mapping) else None,
#             "incremental_commit_report": self._clone(
#                 history.get("commit_report")
#             ) if isinstance(history.get("commit_report"), Mapping) else None,
#             "args": dict(vars(self.args)) if hasattr(self.args, "__dict__") else {},
#             "torch_rng_state": torch.get_rng_state().cpu(),
#         }
#         if torch.cuda.is_available():
#             checkpoint["cuda_rng_state_all"] = [
#                 state.cpu() for state in torch.cuda.get_rng_state_all()
#             ]
#         if evaluator_metrics is not None:
#             checkpoint["evaluator_metrics"] = self._clone(
#                 dict(evaluator_metrics)
#             )

#         descriptor, temporary = tempfile.mkstemp(
#             prefix=".checkpoint_", suffix=".pth.tmp", dir=phase_dir
#         )
#         os.close(descriptor)
#         manifest_tmp = manifest_path + ".tmp"
#         try:
#             with open(temporary, "wb") as stream:
#                 torch.save(checkpoint, stream)
#                 stream.flush()
#                 os.fsync(stream.fileno())
#             file_digest = self._file_sha256(temporary)
#             with open(manifest_tmp, "w", encoding="utf-8") as stream:
#                 json.dump(
#                     {
#                         "sha256": file_digest,
#                         "memory_snapshot_sha256": memory_digest,
#                         "format_version": self.CHECKPOINT_FORMAT_VERSION,
#                         "architecture_id": self.ARCHITECTURE_ID,
#                         "method": self.METHOD_NAME,
#                         "schema_version": self.BANK_SCHEMA_VERSION,
#                         "joint_factorization": self.JOINT_FACTORIZATION,
#                         "phase": phase,
#                         "seen_classes": list(seen_classes),
#                         "geometry_bank_contract_digest":
#                             checkpoint["geometry_bank_contract_digest"],
#                         "classifier_bound_contract_digest":
#                             checkpoint["classifier_bound_contract_digest"],
#                         "model_contract_digest":
#                             checkpoint["model_contract_digest"],
#                     },
#                     stream,
#                     indent=2,
#                 )
#                 stream.flush()
#                 os.fsync(stream.fileno())
#             os.replace(temporary, path)
#             os.replace(manifest_tmp, manifest_path)
#         except Exception:
#             for candidate in (temporary, manifest_tmp):
#                 try:
#                     if os.path.exists(candidate):
#                         os.remove(candidate)
#                 except OSError:
#                     pass
#             raise
#         print(f"[Saved PC-STGB Checkpoint] {path}")
#         return path

#     def load_checkpoint(
#         self,
#         path: str,
#         *,
#         strict: bool = True,
#         verify_checksum: bool = True,
#         restore_rng: bool = True,
#     ) -> Dict[str, Any]:
#         path = os.path.abspath(str(path))
#         if not os.path.isfile(path):
#             raise FileNotFoundError(path)

#         manifest: Dict[str, Any] = {}
#         if verify_checksum:
#             manifest_path = path + ".sha256"
#             if not os.path.isfile(manifest_path):
#                 raise RuntimeError(f"Checkpoint manifest is missing: {manifest_path}")
#             with open(manifest_path, "r", encoding="utf-8") as stream:
#                 manifest = json.load(stream)
#             actual = self._file_sha256(path)
#             expected = str(manifest.get("sha256", ""))
#             if actual != expected:
#                 raise RuntimeError(
#                     "Checkpoint checksum mismatch: "
#                     f"expected={expected}, actual={actual}"
#                 )
#             if strict:
#                 manifest_expected = {
#                     "architecture_id": self.ARCHITECTURE_ID,
#                     "method": self.METHOD_NAME,
#                     "schema_version": self.BANK_SCHEMA_VERSION,
#                     "joint_factorization": self.JOINT_FACTORIZATION,
#                 }
#                 failures = {
#                     key: (manifest.get(key), value)
#                     for key, value in manifest_expected.items()
#                     if manifest.get(key) != value
#                 }
#                 if failures:
#                     raise RuntimeError(
#                         f"Checkpoint manifest contract mismatch: {failures}"
#                     )

#         try:
#             checkpoint = torch.load(
#                 path, map_location=self.device, weights_only=False
#             )
#         except TypeError:
#             checkpoint = torch.load(path, map_location=self.device)
#         if not isinstance(checkpoint, Mapping):
#             raise RuntimeError("Checkpoint payload must be a mapping")
#         phase = int(checkpoint.get("phase", -1))
#         if phase < 0:
#             raise RuntimeError("Checkpoint phase must be non-negative")
#         if int(checkpoint.get("format_version", -1)) != self.CHECKPOINT_FORMAT_VERSION:
#             raise RuntimeError(
#                 "Unsupported checkpoint format version: "
#                 f"{checkpoint.get('format_version')!r}"
#             )
#         runtime = checkpoint.get("runtime_contract")
#         if not isinstance(runtime, Mapping):
#             raise RuntimeError("Checkpoint runtime contract is missing")
#         self._assert_loaded_runtime_contract(runtime, strict=strict)

#         model_state = checkpoint.get("model_state_dict")
#         memory = checkpoint.get("memory_snapshot")
#         if not isinstance(model_state, Mapping):
#             raise RuntimeError("Checkpoint is missing model_state_dict")
#         if not isinstance(memory, Mapping):
#             raise RuntimeError("Checkpoint is missing memory_snapshot")
#         calculated_memory_digest = self._content_sha256(memory)
#         saved_memory_digest = str(checkpoint.get("memory_snapshot_sha256", ""))
#         if strict and calculated_memory_digest != saved_memory_digest:
#             raise RuntimeError("Checkpoint memory snapshot digest mismatch")
#         if verify_checksum and manifest.get("memory_snapshot_sha256") not in (
#             None,
#             calculated_memory_digest,
#         ):
#             raise RuntimeError("Manifest and checkpoint memory digests disagree")

#         self.model.load_state_dict(model_state, strict=strict)
#         self.model.load_memory_snapshot(dict(memory), strict=strict)

#         dataset_state = checkpoint.get("dataset_runtime_state")
#         if isinstance(dataset_state, Mapping):
#             restore = getattr(self.dataset, "restore_runtime_state", None)
#             if callable(restore):
#                 restore(dataset_state, purge_finalized_train_payload=False)
#             elif strict and self._optional_bool(
#                 "checkpoint_requires_dataset_state",
#                 self._optional_bool("base_checkpoint_requires_dataset_state", False),
#             ):
#                 raise RuntimeError("Dataset cannot restore saved runtime state")

#         seen_classes = [int(value) for value in checkpoint.get("seen_classes", [])]
#         expected_seen = self._seen_classes_for_phase(phase)
#         if strict and seen_classes != expected_seen:
#             raise RuntimeError(
#                 f"Checkpoint classes={seen_classes} != dataset classes={expected_seen}"
#             )
#         seen_classes = seen_classes or expected_seen
#         self._set_post_phase_state(seen_classes, phase)
#         self._validate_final_memory_contract(seen_classes)

#         actual_bank_digest = self.model.geometry_bank.contract_digest()
#         actual_bound_digest = self.model.classifier.bound_contract_digest
#         actual_classifier_digest = self.model.classifier.classifier_contract_digest()
#         actual_model_digest = self.model.model_contract_digest()
#         expected_digests = {
#             "geometry_bank_contract_digest": actual_bank_digest,
#             "classifier_bound_contract_digest": actual_bound_digest,
#             "classifier_contract_digest": actual_classifier_digest,
#             "model_contract_digest": actual_model_digest,
#         }
#         mismatched = {
#             key: (checkpoint.get(key), value)
#             for key, value in expected_digests.items()
#             if strict and checkpoint.get(key) != value
#         }
#         if mismatched:
#             raise RuntimeError(
#                 f"Checkpoint contract digest mismatch after restore: {mismatched}"
#             )
#         if verify_checksum:
#             for key in (
#                 "geometry_bank_contract_digest",
#                 "classifier_bound_contract_digest",
#                 "model_contract_digest",
#             ):
#                 saved = manifest.get(key)
#                 if saved is not None and saved != expected_digests[key]:
#                     raise RuntimeError(
#                         f"Manifest {key} disagrees with restored model"
#                     )

#         base_report = checkpoint.get("base_geometry_report")
#         base_handoff = checkpoint.get("base_handoff")
#         self.model.base_geometry_report = base_report
#         self.model.base_handoff = base_handoff
#         self._last_base_geometry_report = (
#             dict(base_report) if isinstance(base_report, Mapping) else None
#         )
#         self._last_base_geometry_certificate = self._last_base_geometry_report

#         if restore_rng:
#             cpu_rng = checkpoint.get("torch_rng_state")
#             if torch.is_tensor(cpu_rng):
#                 torch.set_rng_state(cpu_rng.cpu())
#             cuda_rng = checkpoint.get("cuda_rng_state_all")
#             if torch.cuda.is_available() and isinstance(cuda_rng, (list, tuple)):
#                 torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
#         return dict(checkpoint)
