
from __future__ import annotations

from contextlib import nullcontext
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from losses.loss import geometry_energy_matrix
except Exception:  # pragma: no cover - compile/runtime compatibility for renamed packages
    geometry_energy_matrix = None


class TrainerHelper:
    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------
    def _zero(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        if torch.is_tensor(ref):
            return ref.sum() * 0.0
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def _as_class_list(self, class_ids: Iterable[int]) -> List[int]:
        return [int(c) for c in class_ids]

    def _detach_clone(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().clone()

    def _json_safe(self, obj):
        if torch.is_tensor(obj):
            obj = obj.detach().cpu()
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        try:
            import numpy as _np
            if isinstance(obj, (_np.integer, _np.floating)):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _class_name(self, cls: int) -> str:
        cls = int(cls)
        names = getattr(self.dataset, "target_names", None)
        if names is not None:
            try:
                return str(names[cls])
            except Exception:
                pass
        names = getattr(self.args, "target_names", None)
        if names is not None:
            try:
                return str(names[cls])
            except Exception:
                pass
        return f"Class-{cls}"

    def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
            return self._zero(logits)
        labels = labels.long().view(-1).to(logits.device)
        if labels.numel() == 0:
            return self._zero(logits)
        if labels.numel() != logits.size(0):
            raise RuntimeError(f"CE batch mismatch: logits={logits.size(0)}, labels={labels.numel()}")
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= logits.size(1):
            raise RuntimeError(
                f"CE label range [{int(labels.min())},{int(labels.max())}] incompatible with logits width={logits.size(1)}"
            )
        clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
        smoothing = float(getattr(self, "label_smoothing", getattr(self.args, "label_smoothing", 0.0)))
        return F.cross_entropy(logits.clamp(-clip, clip), labels, label_smoothing=smoothing)

    def _unpack_hsi_batch(self, batch):
        """
        Accept both legacy (patches, labels) batches and metadata batches
        (patches, labels, center_spectrum, coord). Metadata spectra are part
        of the PG-RGA data contract: reduced/PCA patches go to the backbone,
        raw physical center spectra go to GeometryBank spectral-shape scoring.
        """
        if isinstance(batch, dict):
            x = batch.get("image", batch.get("patch", batch.get("patches", None)))
            y = batch.get("label", batch.get("labels", None))
            spectra = batch.get("spectrum", batch.get("spectra", None))
            coords = batch.get("coord", batch.get("coords", None))
            if x is None or y is None:
                raise RuntimeError("Batch dict must contain image/patches and label/labels.")
            return x, y, spectra, coords
        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise RuntimeError(f"Batch tuple/list must have at least 2 fields, got {len(batch)}")
            x, y = batch[0], batch[1]
            spectra = batch[2] if len(batch) >= 3 else None
            coords = batch[3] if len(batch) >= 4 else None
            return x, y, spectra, coords
        raise RuntimeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _center_spectrum_from_tensor(x: torch.Tensor) -> torch.Tensor:
        """Return a center-pixel spectral vector from an HSI tensor.

        For center-pixel HSI classification, the label belongs to the center
        pixel, not the whole patch. Patch-mean spectra can mix neighboring
        classes and poison SRGP spectral-shape descriptors.
        """
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a tensor, got {type(x)}")
        if x.dim() == 4:          # [B, S, H, W]
            return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
        if x.dim() == 3:          # [B, S, L] or equivalent spectral sequence
            return x[:, :, x.size(-1) // 2]
        if x.dim() == 2:          # [B, S]
            return x
        return x.flatten(1)

    def _normalize_spectral_metadata_tensor(
        self,
        spectra: torch.Tensor,
        *,
        ref_x: Optional[torch.Tensor] = None,
        expected_n: Optional[int] = None,
    ) -> torch.Tensor:
        """Normalize metadata spectra to [B,S] without flattening patches.

        This is a critical SRGP safety helper. If raw spectra arrive as
        [B,S,H,W], the label belongs to the center pixel, so we take only the
        center spectrum. Flattening the whole patch would create [B,S*H*W] and
        poison spectral-shape descriptors.
        """
        if not torch.is_tensor(spectra):
            spectra = torch.as_tensor(spectra)
        if ref_x is not None and torch.is_tensor(ref_x):
            s = spectra.to(device=ref_x.device, dtype=ref_x.dtype, non_blocking=True)
            n = int(ref_x.size(0))
        else:
            s = spectra.float()
            n = int(expected_n or (s.size(0) if s.dim() > 0 else 1))

        if s.numel() == 0:
            return s.reshape(n, 0)
        if s.dim() == 4:          # [B,S,H,W]
            s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
        elif s.dim() == 3:        # [B,S,L] or [B,H,W]-like metadata
            if s.size(0) != n:
                s = s.reshape(n, -1)
            elif s.size(1) > 0 and s.size(2) > 1:
                s = s[:, :, s.size(-1) // 2]
            else:
                s = s.reshape(n, -1)
        elif s.dim() == 1:
            if n > 1 and s.numel() % n == 0:
                s = s.reshape(n, -1)
            else:
                s = s.reshape(1, -1)
        elif s.dim() != 2:
            s = s.reshape(n, -1)
        if s.size(0) != n:
            raise RuntimeError(f"spectral metadata batch mismatch: spectra={tuple(s.shape)}, expected_n={n}")
        return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    def _dataset_spectra_are_physical(self) -> Optional[bool]:
        """Return dataset-declared physical-spectra status when available."""
        for attr in ("spectra_are_physical", "raw_spectra_are_physical", "center_spectra_are_physical"):
            if hasattr(self.dataset, attr):
                value = getattr(self.dataset, attr)
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
        fn = getattr(self.dataset, "has_physical_spectra", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return None
        return None

    def _cfg_bool(self, name: str, default: bool = False) -> bool:
        value = getattr(self, name, getattr(self.args, name, default))
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

    def _spectral_summary_is_physical_default(self, spectral_dim: int = 0, *, source: str = "input") -> bool:
        """Decide whether spectral_summary has physical wavelength order.

        Physical spectral-shape losses are allowed only for raw wavelength-ordered
        spectra.  PCA/reduced summaries are always non-physical, even if they
        arrive through batch metadata.
        """
        pca_components = int(getattr(self.args, "pca_components", 0) or 0)
        uses_pca = self._cfg_bool("use_pca", pca_components > 0)
        dim = int(spectral_dim or 0)
        if uses_pca and pca_components > 0 and dim > 0 and dim <= pca_components:
            return False

        explicit = getattr(self, "spectral_summary_is_physical", getattr(self.args, "spectral_summary_is_physical", None))
        if explicit is not None:
            value = self._cfg_bool("spectral_summary_is_physical", bool(explicit))
            if value and uses_pca and pca_components > 0 and dim <= pca_components:
                return False
            return bool(value)

        if source in {"batch_metadata", "dataset_raw", "external"}:
            ds_flag = self._dataset_spectra_are_physical()
            if ds_flag is not None:
                return bool(ds_flag) and not (uses_pca and pca_components > 0 and dim <= pca_components)
            return self._cfg_bool("raw_spectral_summary_is_physical", True) and not (uses_pca and pca_components > 0 and dim <= pca_components)

        if uses_pca:
            return False
        return self._cfg_bool("input_spectral_summary_is_physical", False)

    def _get_class_external_spectra_with_flag(
        self,
        cls: int,
        split: str,
        expected_n: int,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Try to obtain raw/physical center spectra from the dataset.

        Prefer dataset methods that support ``require_physical=True``. If a
        dataset returns PCA/reduced metadata, the tensor may still be returned,
        but the physical flag is False so SRGP derivative scoring stays off.
        """
        method_names = (
            "get_class_spectra",
            "get_class_spectrum",
            "get_class_center_spectra",
            "get_class_raw_spectra",
            "get_class_spectral_summary",
            "get_class_center_spectrum",
        )
        dataset_physical = self._dataset_spectra_are_physical()
        for name in method_names:
            fn = getattr(self.dataset, name, None)
            if not callable(fn):
                continue
            call_attempts = (
                dict(split=split, require_physical=True),
                dict(split=split),
                {},
            )
            for kwargs in call_attempts:
                try:
                    val = fn(int(cls), **kwargs)
                except TypeError:
                    continue
                except Exception:
                    continue
                if val is None:
                    continue
                try:
                    t = val.detach().cpu().float() if torch.is_tensor(val) else torch.as_tensor(val).float()
                    t = self._normalize_spectral_metadata_tensor(t, expected_n=int(expected_n)).cpu().float()
                except Exception:
                    continue
                if t.numel() == 0 or t.dim() != 2 or t.size(0) != int(expected_n):
                    continue
                physical = bool(dataset_physical) if dataset_physical is not None else ("raw" in name or "physical" in name or "center_spectra" in name)
                # Safety: metadata with the same width as PCA input is reduced, not raw.
                pca_components = int(getattr(self.args, "pca_components", 0) or 0)
                uses_pca = self._cfg_bool("use_pca", pca_components > 0)
                if uses_pca and pca_components > 0 and int(t.size(1)) <= pca_components:
                    physical = False
                return t, bool(physical)
        return None, False

    def _get_class_external_spectra(self, cls: int, split: str, expected_n: int) -> Optional[torch.Tensor]:
        spectra, _ = self._get_class_external_spectra_with_flag(cls, split, expected_n)
        return spectra

    def _resolve_batch_spectral_summary(
        self,
        x: torch.Tensor,
        *,
        spectra: Optional[torch.Tensor] = None,
        model_out: Optional[Dict[str, torch.Tensor]] = None,
        source: str = "input",
        spectral_summary_is_physical: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Return spectral summary and physical-band flag for SRGP.

        Priority:
            1) explicit spectra from dataset/loader metadata, center-pixel only;
            2) model_out['spectral_summary'];
            3) center spectrum from input tensor.
        """
        if spectra is not None and torch.is_tensor(spectra) and spectra.numel() > 0:
            ss = self._normalize_spectral_metadata_tensor(spectra, ref_x=x)
            physical = bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else self._spectral_summary_is_physical_default(int(ss.size(1)), source=source)
            return ss, physical

        if isinstance(model_out, dict):
            ss = model_out.get("spectral_summary", None)
            if torch.is_tensor(ss) and ss.numel() > 0:
                ss = self._normalize_spectral_metadata_tensor(ss, ref_x=x)
                if ss.size(0) == x.size(0):
                    flag = model_out.get("spectral_summary_is_physical", None)
                    if torch.is_tensor(flag) and flag.numel() == 1:
                        return ss, bool(flag.detach().cpu().item())
                    if isinstance(flag, bool):
                        return ss, flag
                    physical = bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else self._spectral_summary_is_physical_default(int(ss.size(1)), source="model")
                    return ss, physical

        ss = self._center_spectrum_from_tensor(x).to(device=x.device, dtype=x.dtype)
        if ss.dim() != 2:
            ss = ss.flatten(1)
        physical = bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else self._spectral_summary_is_physical_default(int(ss.size(1)), source="input")
        return torch.nan_to_num(ss, nan=0.0, posinf=0.0, neginf=0.0), physical

    def compute_spectral_metadata_diagnostics(
        self,
        spectral_summary: Optional[torch.Tensor],
        *,
        source: str,
        physical: bool,
    ) -> Dict[str, object]:
        dim = int(spectral_summary.size(1)) if torch.is_tensor(spectral_summary) and spectral_summary.dim() == 2 else 0
        pca_components = int(getattr(self.args, "pca_components", 0) or 0)
        reduced = bool(pca_components > 0 and dim <= pca_components)
        return {
            "source": str(source),
            "spectral_dim": dim,
            "physical": bool(physical and not reduced),
            "pca_components": pca_components,
            "reduced_or_pca": reduced,
            "spectral_shape_active": bool(physical and not reduced),
        }


    # ------------------------------------------------------------------
    # Clean GeometryBank validation/access
    # ------------------------------------------------------------------
    def _canonicalize_bank(self, bank: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create aliases required by the SRGP feature+spectral geometry path."""
        if "res_vars" not in bank and "resvars" in bank:
            bank["res_vars"] = bank["resvars"]
        if "resvars" not in bank and "res_vars" in bank:
            bank["resvars"] = bank["res_vars"]

        if "variances" not in bank and "eigvals" in bank and "res_vars" in bank:
            eig = bank["eigvals"]
            res = bank["res_vars"]
            if torch.is_tensor(eig) and torch.is_tensor(res) and eig.numel() > 0 and res.numel() > 0:
                bank["variances"] = torch.cat([eig, res.unsqueeze(-1)], dim=-1)

        if "eigvals" not in bank and "variances" in bank and torch.is_tensor(bank["variances"]):
            bank["eigvals"] = bank["variances"][:, :-1]
        if "res_vars" not in bank and "variances" in bank and torch.is_tensor(bank["variances"]):
            bank["res_vars"] = bank["variances"][:, -1]
            bank["resvars"] = bank["res_vars"]

        if "band_importances" not in bank and "band_importance" in bank:
            bank["band_importances"] = bank["band_importance"]
        if "band_importance" not in bank and "band_importances" in bank:
            bank["band_importance"] = bank["band_importances"]

        # SRGP spectral-shape aliases.  The fixed GeometryBank uses plural names;
        # older intermediate files may use singular names.  Keep both stable.
        alias_pairs = (
            ("spectral_curve_means", "spectral_curve_mean"),
            ("spectral_curve_vars", "spectral_curve_var"),
            ("spectral_curve_d1", "spectral_d1"),
            ("spectral_curve_d2", "spectral_d2"),
        )
        for primary, alias in alias_pairs:
            if primary not in bank and alias in bank:
                bank[primary] = bank[alias]
            if alias not in bank and primary in bank:
                bank[alias] = bank[primary]
        if "spectral_shape_reliability" not in bank and "spectral_reliability" in bank:
            bank["spectral_shape_reliability"] = bank["spectral_reliability"]
        if "spectral_reliability" not in bank and "spectral_shape_reliability" in bank:
            bank["spectral_reliability"] = bank["spectral_shape_reliability"]
        return bank

    def _valid_mask_from_bank(self, bank: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Strict valid-row mask for the cleaned GeometryBank.

        Do not fabricate reliability/sample-count fallbacks here.  Capacity rows,
        NaN rows, rows with invalid active rank, or invalid band signatures must
        not be scoreable by the classifier/losses.
        """
        bank = self._canonicalize_bank(bank)
        means = bank.get("means", None)
        bases = bank.get("bases", None)
        variances = bank.get("variances", None)
        counts = bank.get("sample_counts", None)
        active = bank.get("active_ranks", None)
        reliability = bank.get("reliability", None)
        if not torch.is_tensor(means) or means.dim() != 2:
            raise RuntimeError("GeometryBank must expose means [C,D].")
        C = int(means.size(0))
        if not torch.is_tensor(bases) or bases.dim() != 3 or bases.size(0) != C:
            raise RuntimeError("GeometryBank must expose bases [C,D,R].")
        if not torch.is_tensor(variances) or variances.dim() != 2 or variances.size(0) != C:
            raise RuntimeError("GeometryBank must expose variances [C,R+1].")
        if not torch.is_tensor(counts) or counts.numel() != C:
            raise RuntimeError("GeometryBank must expose real sample_counts [C]. Capacity rows are not valid memory.")
        if not torch.is_tensor(active) or active.numel() != C:
            raise RuntimeError("GeometryBank must expose active_ranks [C].")
        if not torch.is_tensor(reliability) or reliability.numel() != C:
            raise RuntimeError("GeometryBank must expose reliability [C].")

        device = means.device
        counts = counts.to(device=device).flatten()
        active = active.to(device=device).long().flatten()
        reliability = reliability.to(device=device).flatten()
        finite = (
            torch.isfinite(means).all(dim=1)
            & torch.isfinite(bases).flatten(1).all(dim=1)
            & torch.isfinite(variances).all(dim=1)
            & torch.isfinite(counts)
            & torch.isfinite(reliability)
        )
        R = int(bases.size(2))
        active_ok = (active >= 0) & (active <= R)
        sample_cap_ok = active <= torch.clamp(counts.long() - 1, min=0, max=R)
        valid = (counts > 0) & finite & active_ok & sample_cap_ok

        bands = bank.get("band_importances", bank.get("band_importance", None))
        if torch.is_tensor(bands) and bands.numel() > 0:
            if bands.dim() != 2 or bands.size(0) != C:
                raise RuntimeError("GeometryBank band_importances must be [C,S] when present.")
            b = bands.to(device=device)
            bfinite = torch.isfinite(b).all(dim=1)
            bsum = b.clamp_min(0.0).sum(dim=1)
            valid = valid & bfinite & (bsum > 1e-8)

        # SRGP spectral-shape rows are not mandatory for synthetic replay, but
        # when present they must be finite for a row to participate in SRGP
        # spectral conflict/diagnostic scoring.
        for skey in ("spectral_curve_means", "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2"):
            sv = bank.get(skey, None)
            if torch.is_tensor(sv) and sv.numel() > 0:
                if sv.dim() != 2 or sv.size(0) != C:
                    raise RuntimeError(f"GeometryBank {skey} must be [C,S*] when present, got {tuple(sv.shape)}")
                valid = valid & torch.isfinite(sv.to(device=device)).all(dim=1)
        srel = bank.get("spectral_shape_reliability", None)
        if torch.is_tensor(srel) and srel.numel() > 0:
            if srel.numel() != C:
                raise RuntimeError(f"spectral_shape_reliability must have C={C} entries, got {srel.numel()}")
            valid = valid & torch.isfinite(srel.to(device=device).flatten())
        return valid

    def _bank_valid_mask(self, bank: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._valid_mask_from_bank(bank)

    def _safe_get_subspace_bank(self, require_ready: bool = True) -> Dict[str, torch.Tensor]:
        if not hasattr(self.model, "get_subspace_bank"):
            raise AttributeError("Model must expose get_subspace_bank().")
        bank = self.model.get_subspace_bank()
        if not isinstance(bank, dict):
            raise TypeError(f"get_subspace_bank() must return dict, got {type(bank)}")
        bank = self._canonicalize_bank(bank)
        if not require_ready:
            return bank

        required = ("means", "bases", "variances", "eigvals", "res_vars", "sample_counts", "active_ranks", "reliability")
        missing = [k for k in required if k not in bank or not torch.is_tensor(bank[k]) or bank[k].numel() == 0]
        if missing:
            raise RuntimeError(f"GeometryBank missing required non-empty keys: {missing}")
        means, bases, variances = bank["means"], bank["bases"], bank["variances"]
        if means.dim() != 2:
            raise RuntimeError(f"bank['means'] must be [C,D], got {tuple(means.shape)}")
        if bases.dim() != 3:
            raise RuntimeError(f"bank['bases'] must be [C,D,R], got {tuple(bases.shape)}")
        if variances.dim() != 2:
            raise RuntimeError(f"bank['variances'] must be [C,R+1], got {tuple(variances.shape)}")
        C, D = means.shape
        R = int(bases.size(2))
        if bases.size(0) != C or bases.size(1) != D or variances.size(0) != C or variances.size(1) != R + 1:
            raise RuntimeError(
                f"GeometryBank shape mismatch: means={tuple(means.shape)}, bases={tuple(bases.shape)}, variances={tuple(variances.shape)}"
            )
        for key in ("means", "bases", "variances", "eigvals", "res_vars", "sample_counts", "active_ranks", "reliability"):
            if not torch.isfinite(bank[key].float()).all():
                raise RuntimeError(f"GeometryBank contains NaN/Inf in {key}.")
        if bool((variances < 0).any().item()):
            bad = float(variances.min().detach().cpu().item())
            raise RuntimeError(f"GeometryBank variances must be non-negative, min={bad:.3e}.")

        device = means.device
        bank["sample_counts"] = bank["sample_counts"].to(device=device).flatten()
        bank["active_ranks"] = bank["active_ranks"].to(device=device).long().flatten()
        bank["reliability"] = bank["reliability"].to(device=device, dtype=means.dtype).flatten()
        if bank["sample_counts"].numel() != C or bank["active_ranks"].numel() != C or bank["reliability"].numel() != C:
            raise RuntimeError("GeometryBank sample_counts/active_ranks/reliability must each have C entries.")
        active = bank["active_ranks"]
        counts = bank["sample_counts"]
        if bool(((active < 0) | (active > R)).any().item()):
            raise RuntimeError("GeometryBank active_ranks outside [0,R].")
        if bool((active > torch.clamp(counts.long() - 1, min=0, max=R)).any().item()):
            bad = ((active > torch.clamp(counts.long() - 1, min=0, max=R))).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise RuntimeError(f"GeometryBank active_rank exceeds sample_count-1 for rows: {bad[:20]}")

        # Orthonormality for valid rows only. Capacity rows are allowed to be zero.
        valid_pre = counts > 0
        if bool(valid_pre.any().item()):
            eye = torch.eye(R, device=bases.device, dtype=bases.dtype)
            bad_rows = []
            for c in valid_pre.nonzero(as_tuple=False).flatten().tolist():
                r = int(active[c].detach().item())
                if r <= 0:
                    continue
                gram = bases[c, :, :r].transpose(0, 1).matmul(bases[c, :, :r])
                if not torch.allclose(gram, eye[:r, :r], atol=float(getattr(self.args, "bank_basis_ortho_atol", 2e-3)), rtol=0.0):
                    bad_rows.append(int(c))
            if bad_rows:
                raise RuntimeError(f"GeometryBank basis columns are not orthonormal for valid rows: {bad_rows[:20]}")

        bank["valid_mask"] = self._valid_mask_from_bank(bank).to(device=device)
        if not bool(bank["valid_mask"].any().item()):
            raise RuntimeError("GeometryBank has no valid built rows; all sample_counts are zero or invalid.")
        return bank

    def _validate_bank_has_classes(self, bank: Dict[str, torch.Tensor], class_ids: Iterable[int]) -> None:
        self.assert_bank_ready_for_seen_classes(bank, class_ids)

    def assert_bank_ready_for_seen_classes(self, bank: Dict[str, torch.Tensor], seen_classes: Iterable[int]) -> None:
        ids = self._as_class_list(seen_classes)
        if not ids:
            raise RuntimeError("seen_classes is empty; cannot validate GeometryBank.")
        bank = self._canonicalize_bank(bank)
        valid = self._valid_mask_from_bank(bank).to(bank["means"].device)
        C = int(bank["means"].size(0))
        missing = []
        for c in ids:
            if c < 0 or c >= C or c >= valid.numel() or not bool(valid[c].detach().item()):
                missing.append(int(c))
        if missing:
            raise RuntimeError(f"GeometryBank rows missing/invalid for seen classes: {missing}")

    def assert_bank_has_only_allowed_valid_rows(self, bank: Dict[str, torch.Tensor], allowed_classes: Iterable[int]) -> None:
        bank = self._canonicalize_bank(bank)
        valid = self._valid_mask_from_bank(bank)
        allowed = set(self._as_class_list(allowed_classes))
        valid_rows = [int(i) for i in valid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()]
        future = [c for c in valid_rows if c not in allowed]
        if future:
            raise RuntimeError(f"GeometryBank has valid rows outside allowed classes: {future}. This is future-class leakage.")

    @torch.no_grad()
    def snapshot_bank_rows(self, bank: Dict[str, torch.Tensor], class_ids: Iterable[int]) -> Dict[str, torch.Tensor]:
        ids = self._as_class_list(class_ids)
        if not ids:
            return {}
        bank = self._canonicalize_bank(bank)
        max_id = max(ids)
        snap: Dict[str, torch.Tensor] = {}
        for key in (
            "means", "bases", "variances", "eigvals", "res_vars", "active_ranks", "reliability",
            "feature_reliability", "sample_counts", "band_importances", "band_reliability",
            "spectral_curve_means", "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2",
            "spectral_shape_reliability",
        ):
            value = bank.get(key, None)
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) > max_id:
                snap[key] = value[ids].detach().clone()
        return snap

    @torch.no_grad()
    def assert_bank_rows_unchanged(
        self,
        before: Dict[str, torch.Tensor],
        after: Dict[str, torch.Tensor],
        class_ids: Iterable[int],
        context: str,
        *,
        atol: float = 1e-6,
    ) -> None:
        ids = self._as_class_list(class_ids)
        if not ids or not before:
            return
        after = self._canonicalize_bank(after)
        bad: List[str] = []
        max_id = max(ids)
        for key, old_v in before.items():
            cur = after.get(key, None)
            if not torch.is_tensor(cur) or cur.dim() == 0 or cur.size(0) <= max_id:
                bad.append(f"{key}:missing")
                continue
            cur_v = cur[ids].detach().to(device=old_v.device, dtype=old_v.dtype)
            if cur_v.shape != old_v.shape or not torch.allclose(cur_v, old_v, atol=float(atol), rtol=0.0):
                diff = float((cur_v - old_v).abs().max().item()) if cur_v.shape == old_v.shape else float("inf")
                bad.append(f"{key}:maxdiff={diff:.3e}")
        if bad:
            raise RuntimeError(f"GeometryBank rows changed during {context}. Mutated tensors: {bad[:20]}")

    def compute_bank_validity_diagnostics(
        self,
        bank: Optional[Dict[str, torch.Tensor]] = None,
        *,
        seen_classes: Optional[Iterable[int]] = None,
        allowed_classes: Optional[Iterable[int]] = None,
    ) -> Dict[str, object]:
        diag: Dict[str, object] = {"valid": False, "error": None}
        try:
            bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
            valid = self._valid_mask_from_bank(bank)
            valid_rows = [int(i) for i in valid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()]
            diag.update({
                "num_rows": int(bank["means"].size(0)),
                "feature_dim": int(bank["means"].size(1)),
                "rank": int(bank["bases"].size(2)),
                "valid_rows": valid_rows,
                "sample_counts": self._json_safe(bank["sample_counts"]),
                "active_ranks": self._json_safe(bank["active_ranks"]),
                "reliability": self._json_safe(bank["reliability"]),
            })
            if seen_classes is not None:
                seen = self._as_class_list(seen_classes)
                missing = [c for c in seen if c not in valid_rows]
                diag["seen_classes"] = seen
                diag["missing_seen_rows"] = missing
            if allowed_classes is not None:
                allowed = set(self._as_class_list(allowed_classes))
                future = [c for c in valid_rows if c not in allowed]
                diag["future_valid_rows"] = future
            diag["valid"] = not diag.get("missing_seen_rows") and not diag.get("future_valid_rows")
        except Exception as exc:
            diag["error"] = str(exc)
        return diag

    def _class_memory_is_valid(self, cls: int) -> bool:
        try:
            bank = self._safe_get_subspace_bank(require_ready=False)
            bank = self._canonicalize_bank(bank)
        except Exception:
            return False
        cls = int(cls)
        means, bases, vars_, counts = bank.get("means"), bank.get("bases"), bank.get("variances"), bank.get("sample_counts")
        if not (torch.is_tensor(means) and torch.is_tensor(bases) and torch.is_tensor(vars_) and torch.is_tensor(counts)):
            return False
        if cls < 0 or cls >= means.size(0) or cls >= bases.size(0) or cls >= vars_.size(0) or cls >= counts.numel():
            return False
        try:
            valid = self._valid_mask_from_bank(bank)
            return bool(valid.numel() > cls and valid[cls].detach().item())
        except Exception:
            return bool(float(counts[cls].detach().item()) > 0.0 and torch.isfinite(means[cls]).all() and torch.isfinite(bases[cls]).all() and torch.isfinite(vars_[cls]).all())

    # ------------------------------------------------------------------
    # Label mapping / logit-convention helpers
    # ------------------------------------------------------------------
    def _classes_tensor(self, class_ids: Iterable[int], *, device=None) -> torch.Tensor:
        ids = [int(c) for c in class_ids]
        if not ids:
            raise RuntimeError("class_ids must be non-empty.")
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"class_ids contains duplicates: {ids}")
        if min(ids) < 0:
            raise RuntimeError(f"class_ids must be non-negative global IDs, got {ids}")
        return torch.as_tensor(ids, device=device if device is not None else self.device, dtype=torch.long)

    def assert_global_labels_in_set(self, labels_global: torch.Tensor, allowed_classes: Iterable[int], context: str) -> None:
        if labels_global is None or not torch.is_tensor(labels_global):
            raise RuntimeError(f"{context}: labels_global must be a tensor.")
        y = labels_global.to(device=self.device).long().view(-1)
        if y.numel() == 0:
            raise RuntimeError(f"{context}: empty label tensor.")
        if int(y.min().detach().item()) < 0:
            raise RuntimeError(f"{context}: negative/background label found: min={int(y.min().detach().item())}")
        allowed = self._classes_tensor(allowed_classes, device=y.device)
        if hasattr(torch, "isin"):
            ok = torch.isin(y, allowed)
        else:
            ok = torch.zeros_like(y, dtype=torch.bool)
            for c in allowed:
                ok |= y == int(c)
        if not bool(ok.all().item()):
            bad = torch.unique(y[~ok]).detach().cpu().tolist()
            raise RuntimeError(f"{context}: labels outside allowed global classes. bad={bad}, allowed={allowed.detach().cpu().tolist()}")

    def assert_valid_ce_targets(self, labels_local: torch.Tensor, num_classes: int, context: str) -> None:
        if labels_local is None or not torch.is_tensor(labels_local):
            raise RuntimeError(f"{context}: CE targets must be a tensor.")
        y = labels_local.long().view(-1)
        if y.numel() == 0:
            raise RuntimeError(f"{context}: empty CE target tensor.")
        n = int(num_classes)
        if n <= 0:
            raise RuntimeError(f"{context}: num_classes must be positive, got {n}.")
        lo = int(y.min().detach().item())
        hi = int(y.max().detach().item())
        if lo < 0 or hi >= n:
            raise RuntimeError(f"{context}: CE target range [{lo},{hi}] incompatible with num_classes={n}.")

    def global_to_seen_local(self, labels_global: torch.Tensor, seen_classes: Iterable[int], *, context: str = "global_to_seen_local") -> torch.Tensor:
        """Map global dataset labels to compact seen-class CE columns.

        Dataset labels and GeometryBank rows are global IDs.  Classifier logits in
        the clean path are [B, len(seen_classes)], so CE targets must be local
        indices in exactly that order.  This function never guesses local labels.
        """
        y = labels_global.to(device=self.device).long().view(-1)
        self.assert_global_labels_in_set(y, seen_classes, context)
        seen = [int(c) for c in seen_classes]
        lut = {c: i for i, c in enumerate(seen)}
        local = torch.full_like(y, -1)
        for c, i in lut.items():
            local[y == int(c)] = int(i)
        self.assert_valid_ce_targets(local, len(seen), context)
        return local

    def seen_local_to_global(self, preds_local: torch.Tensor, seen_classes: Iterable[int], *, context: str = "seen_local_to_global") -> torch.Tensor:
        p = preds_local.to(device=self.device).long().view(-1)
        seen = self._classes_tensor(seen_classes, device=p.device)
        self.assert_valid_ce_targets(p, int(seen.numel()), context)
        return seen.index_select(0, p)

    def global_to_phase_local(self, labels_global: torch.Tensor, phase_classes: Iterable[int], *, context: str = "global_to_phase_local") -> torch.Tensor:
        return self.global_to_seen_local(labels_global, phase_classes, context=context)

    def assert_seen_logits(self, logits: torch.Tensor, seen_classes: Iterable[int], context: str) -> None:
        if logits is None or not torch.is_tensor(logits):
            raise RuntimeError(f"{context}: logits must be a tensor.")
        if logits.dim() != 2:
            raise RuntimeError(f"{context}: logits must be [B,C], got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{context}: logits contain NaN/Inf.")
        n_seen = len([int(c) for c in seen_classes])
        if logits.size(1) != n_seen:
            raise RuntimeError(f"{context}: seen-local logits width {logits.size(1)} != len(seen_classes)={n_seen}.")

    def slice_or_validate_logits_for_seen(
        self,
        logits: torch.Tensor,
        seen_classes: Iterable[int],
        *,
        logit_convention: Literal["seen_local", "global_full"],
        context: str,
    ) -> torch.Tensor:
        """Return [B, len(seen_classes)] logits with explicit convention only."""
        if logits is None or not torch.is_tensor(logits) or logits.dim() != 2:
            raise RuntimeError(f"{context}: logits must be [B,C], got {None if logits is None else tuple(logits.shape)}")
        seen = self._classes_tensor(seen_classes, device=logits.device)
        convention = str(logit_convention).lower().strip()
        if convention == "seen_local":
            self.assert_seen_logits(logits, seen.tolist(), context)
            return logits
        if convention == "global_full":
            if int(seen.max().detach().item()) >= logits.size(1):
                raise RuntimeError(
                    f"{context}: global-full logits width={logits.size(1)} cannot score max seen class {int(seen.max().detach().item())}."
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{context}: logits contain NaN/Inf.")
            return logits.index_select(1, seen)
        raise RuntimeError(f"{context}: unsupported logit_convention={logit_convention!r}; use 'seen_local' or 'global_full'.")

    def cross_entropy_for_seen_logits(
        self,
        logits: torch.Tensor,
        labels_global: torch.Tensor,
        seen_classes: Iterable[int],
        *,
        logit_convention: Literal["seen_local", "global_full"] = "seen_local",
        context: str = "seen_ce",
        class_weighting: bool = False,
    ) -> torch.Tensor:
        logits_seen = self.slice_or_validate_logits_for_seen(logits, seen_classes, logit_convention=logit_convention, context=context)
        labels_local = self.global_to_seen_local(labels_global.to(logits_seen.device), seen_classes, context=context)
        if labels_local.numel() != logits_seen.size(0):
            raise RuntimeError(f"{context}: labels/logits batch mismatch: {labels_local.numel()} vs {logits_seen.size(0)}")
        if not class_weighting:
            return self._stable_ce(logits_seen, labels_local)
        counts = torch.bincount(labels_local, minlength=logits_seen.size(1)).float().to(logits_seen.device)
        weights = torch.zeros_like(counts)
        valid = counts > 0
        weights[valid] = 1.0 / counts[valid].sqrt().clamp_min(1.0)
        weights = weights / weights[valid].mean().clamp_min(1e-8)
        clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
        smoothing = float(getattr(self, "label_smoothing", getattr(self.args, "label_smoothing", 0.0)))
        return F.cross_entropy(logits_seen.clamp(-clip, clip), labels_local, weight=weights.to(logits_seen.dtype), label_smoothing=smoothing)

    def masked_weighted_ce_new(
        self,
        logits: torch.Tensor,
        labels_global: torch.Tensor,
        new_classes: Iterable[int],
        seen_classes: Iterable[int],
        *,
        logit_convention: Literal["seen_local", "global_full"],
        context: str = "masked_weighted_ce_new",
    ) -> torch.Tensor:
        """CE over current new columns with explicit logit convention.

        Prefer cross_entropy_for_seen_logits() for the clean incremental main
        path.  This helper is kept only for ablations that intentionally mask old
        classes during a new-class auxiliary loss.
        """
        new_ids = [int(c) for c in new_classes]
        seen_ids = [int(c) for c in seen_classes]
        self.assert_global_labels_in_set(labels_global, new_ids, context)
        logits_seen = self.slice_or_validate_logits_for_seen(logits, seen_ids, logit_convention=logit_convention, context=context)
        seen_pos = {c: i for i, c in enumerate(seen_ids)}
        missing = [c for c in new_ids if c not in seen_pos]
        if missing:
            raise RuntimeError(f"{context}: new classes not present in seen_classes: {missing}")
        new_cols = torch.as_tensor([seen_pos[c] for c in new_ids], device=logits_seen.device, dtype=torch.long)
        logits_new = logits_seen.index_select(1, new_cols)
        labels_new_local = self.global_to_phase_local(labels_global.to(logits_seen.device), new_ids, context=context)
        counts = torch.bincount(labels_new_local, minlength=len(new_ids)).float().to(logits_seen.device)
        weights = torch.zeros_like(counts)
        valid = counts > 0
        weights[valid] = counts.sum() / counts[valid].clamp_min(1.0)
        weights = weights / weights[valid].mean().clamp_min(1e-8)
        return F.cross_entropy(logits_new, labels_new_local, weight=weights.to(logits_new.dtype))

    def accuracy_for_global_classes(
        self,
        logits: torch.Tensor,
        labels_global: torch.Tensor,
        target_global_classes: Iterable[int],
        seen_classes: Iterable[int],
        *,
        logit_convention: Literal["seen_local", "global_full"],
        context: str = "accuracy_for_global_classes",
    ) -> Tuple[int, int]:
        target_ids = [int(c) for c in target_global_classes]
        if not target_ids:
            return 0, 0
        logits_seen = self.slice_or_validate_logits_for_seen(logits, seen_classes, logit_convention=logit_convention, context=context)
        labels_global = labels_global.to(device=logits_seen.device).long().view(-1)
        if labels_global.numel() != logits_seen.size(0):
            raise RuntimeError(f"{context}: labels/logits batch mismatch: {labels_global.numel()} vs {logits_seen.size(0)}")
        target_t = self._classes_tensor(target_ids, device=logits_seen.device)
        if hasattr(torch, "isin"):
            valid = torch.isin(labels_global, target_t)
        else:
            valid = torch.zeros_like(labels_global, dtype=torch.bool)
            for c in target_t:
                valid |= labels_global == int(c)
        if not bool(valid.any().item()):
            return 0, 0
        pred_local = logits_seen[valid].argmax(dim=1)
        pred_global = self.seen_local_to_global(pred_local, seen_classes, context=context)
        return int((pred_global == labels_global[valid]).sum().item()), int(valid.sum().item())

    def compute_label_mapping_diagnostics(self, labels_global: torch.Tensor, seen_classes: Iterable[int], *, context: str = "label_diag") -> Dict[str, object]:
        y = labels_global.detach().long().view(-1).cpu() if torch.is_tensor(labels_global) else torch.as_tensor(labels_global).long().view(-1)
        seen = [int(c) for c in seen_classes]
        bad = sorted(set(int(v) for v in y.unique().tolist()) - set(seen)) if y.numel() else []
        return {
            "context": context,
            "label_min": int(y.min().item()) if y.numel() else None,
            "label_max": int(y.max().item()) if y.numel() else None,
            "unique_labels": [int(v) for v in y.unique().tolist()] if y.numel() else [],
            "seen_classes": seen,
            "bad_labels": bad,
            "valid": len(bad) == 0 and y.numel() > 0,
        }

    def compute_logit_convention_diagnostics(
        self,
        logits: torch.Tensor,
        seen_classes: Iterable[int],
        *,
        logit_convention: Literal["seen_local", "global_full"],
        context: str = "logit_diag",
    ) -> Dict[str, object]:
        seen = [int(c) for c in seen_classes]
        diag = {
            "context": context,
            "logit_shape": list(logits.shape) if torch.is_tensor(logits) else None,
            "seen_classes": seen,
            "logit_convention": str(logit_convention),
            "valid": False,
            "error": None,
        }
        try:
            _ = self.slice_or_validate_logits_for_seen(logits, seen, logit_convention=logit_convention, context=context)
            diag["valid"] = True
        except Exception as exc:
            diag["error"] = str(exc)
        return diag

    # Deprecated wrappers: kept to make accidental old calls fail loudly.
    def _labels_are_local(self, y: torch.Tensor, class_ids: Iterable[int]) -> bool:
        raise RuntimeError(
            "Ambiguous label auto-detection is disabled. Dataset labels must be global IDs; "
            "use global_to_seen_local() or global_to_phase_local() explicitly."
        )

    def _global_to_local_labels(self, y: torch.Tensor, class_ids: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        local = self.global_to_phase_local(y, class_ids, context="_global_to_local_labels")
        valid = torch.ones_like(local, dtype=torch.bool)
        return local, valid

    def _masked_weighted_ce_new(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        new_class_ids,
        seen_classes: Optional[Iterable[int]] = None,
        *,
        logit_convention: Optional[Literal["seen_local", "global_full"]] = None,
    ) -> torch.Tensor:
        if seen_classes is None or logit_convention is None:
            raise RuntimeError(
                "_masked_weighted_ce_new now requires seen_classes and explicit logit_convention. "
                "Do not guess whether logits are seen-local or global-full."
            )
        return self.masked_weighted_ce_new(logits, y, new_class_ids, seen_classes, logit_convention=logit_convention)

    def _incremental_accuracy_with_count(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        new_class_ids,
        seen_classes: Optional[Iterable[int]] = None,
        *,
        logit_convention: Optional[Literal["seen_local", "global_full"]] = None,
    ) -> Tuple[int, int]:
        if seen_classes is None or logit_convention is None:
            raise RuntimeError(
                "_incremental_accuracy_with_count now requires seen_classes and explicit logit_convention. "
                "Use accuracy_for_global_classes()."
            )
        return self.accuracy_for_global_classes(
            logits, y, new_class_ids, seen_classes, logit_convention=logit_convention, context="_incremental_accuracy_with_count"
        )

    # ------------------------------------------------------------------
    # Geometry memory extraction/update
    # ------------------------------------------------------------------
    def _bank_feature_space(self) -> str:
        """Feature space used to build GeometryBank rows.

        PG-RGA contract:
            * phase 0/base always builds canonical pre-adapter geometry;
            * frozen old rows are never rebuilt in incremental phases;
            * current new rows may be built in scoring/adapted space only when
              the model is explicitly in geometry-gated-adapter incremental mode.

        This prevents a common bug: setting
        ``--incremental_update_mode geometry_gated_adapter`` in the command
        should not make the base GeometryBank use adapted/scoring features.
        """
        phase = int(getattr(getattr(self, "model", None), "current_phase", 0))
        base_active = bool(getattr(getattr(self, "model", None), "base_mode_active", phase == 0))
        inc_active = bool(getattr(getattr(self, "model", None), "incremental_mode_active", False))
        if phase <= 0 or base_active or not inc_active:
            return "canonical"

        explicit = str(getattr(self.args, "geometry_bank_feature_space", "")).lower().strip()
        if explicit in {"canonical", "pre_adapter", "base"}:
            return "canonical"
        if explicit in {"scoring", "adapted", "post_adapter"}:
            return "scoring"

        update_mode = str(getattr(self.args, "incremental_update_mode", getattr(self.model, "incremental_update_mode", "descriptor_only"))).lower().strip()
        adapter_enabled = update_mode in {"geometry_gated_adapter", "g2rpa", "g2-rpa", "adapter", "gated_adapter"}
        adapter_enabled = adapter_enabled or bool(getattr(self.model, "use_geometry_gated_adapter", False))
        return "scoring" if adapter_enabled else "canonical"

    def assert_feature_tensor(
        self,
        features: torch.Tensor,
        *,
        expected_dim: Optional[int] = None,
        context: str = "features",
    ) -> None:
        if features is None or not torch.is_tensor(features):
            raise RuntimeError(f"{context}: features must be a tensor.")
        if features.dim() != 2:
            raise RuntimeError(f"{context}: features must be [B,D], got {tuple(features.shape)}")
        if expected_dim is not None and int(expected_dim) > 0 and int(features.size(1)) != int(expected_dim):
            raise RuntimeError(f"{context}: feature dim {int(features.size(1))} != expected_dim {int(expected_dim)}")
        if not torch.isfinite(features).all():
            raise RuntimeError(f"{context}: features contain NaN/Inf.")

    def compute_feature_space_diagnostics(
        self,
        features: torch.Tensor,
        *,
        expected_dim: Optional[int] = None,
        geometry_feature_space: str = "unknown",
        classifier_feature_space: Optional[str] = None,
        context: str = "feature_space",
    ) -> Dict[str, object]:
        diag = {
            "context": context,
            "shape": list(features.shape) if torch.is_tensor(features) else None,
            "expected_dim": expected_dim,
            "geometry_feature_space": geometry_feature_space,
            "classifier_feature_space": classifier_feature_space or geometry_feature_space,
            "finite": bool(torch.is_tensor(features) and torch.isfinite(features).all().item()) if torch.is_tensor(features) and features.numel() else False,
            "valid": False,
            "error": None,
        }
        try:
            self.assert_feature_tensor(features, expected_dim=expected_dim, context=context)
            if geometry_feature_space not in {"canonical", "scoring"}:
                raise RuntimeError(f"unsupported geometry_feature_space={geometry_feature_space!r}")
            if classifier_feature_space is not None and classifier_feature_space != geometry_feature_space:
                raise RuntimeError(f"feature-space mismatch: bank={geometry_feature_space}, classifier={classifier_feature_space}")
            diag["valid"] = True
        except Exception as exc:
            diag["error"] = str(exc)
        return diag

    def _extract_model_geometry_features(
        self,
        x: torch.Tensor,
        *,
        spectral_summary: Optional[torch.Tensor] = None,
        spectral_summary_is_physical: bool = False,
        space: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract canonical or scoring geometry features for bank construction.

        Rules:
            * canonical space is the only legal base-phase bank space;
            * scoring/adapted space is legal only when the updated NECILModel
              explicitly exposes adapted extraction;
            * never silently substitute canonical features when scoring/adapted
              features were requested.
        """
        space = str(space or self._bank_feature_space()).lower().strip()
        if space in {"pre_adapter", "base"}:
            space = "canonical"
        if space in {"post_adapter", "adapted"}:
            space = "scoring"
        if space not in {"canonical", "scoring"}:
            raise RuntimeError(f"Unsupported geometry feature space {space!r}; use 'canonical' or 'scoring'.")

        phase = int(getattr(getattr(self, "model", None), "current_phase", 0))
        if phase <= 0 and space != "canonical":
            raise RuntimeError("Base phase must build GeometryBank rows from canonical pre-adapter features.")

        def _normalize_feature_output(val: Any, *, source: str) -> Dict[str, torch.Tensor]:
            if isinstance(val, dict):
                od = dict(val)
            elif torch.is_tensor(val):
                od = {"features": val}
            else:
                raise RuntimeError(f"{source} returned unsupported type {type(val)}")

            if space == "canonical":
                key_order = (
                    "canonical_features",
                    "canonical_projected_features",
                    "pre_adapter_features",
                    "base_features",
                    "projected_features",
                    "features",
                    "z",
                )
            else:
                key_order = (
                    "scoring_features",
                    "adapted_features",
                    "adapted_projected_features",
                    "geometry_features",
                    "features",
                    "z",
                )

            feat = None
            for key in key_order:
                if torch.is_tensor(od.get(key, None)):
                    feat = od[key]
                    break
            if not torch.is_tensor(feat):
                raise RuntimeError(f"{source} returned no usable feature tensor for space={space!r}.")

            od["features"] = feat
            od["projected_features"] = feat
            if space == "canonical":
                od["canonical_features"] = feat
                od["canonical_projected_features"] = feat
                od.setdefault("pre_adapter_features", feat)
            else:
                od["scoring_features"] = feat
                od["adapted_features"] = feat
                od["adapted_projected_features"] = feat
            return od

        out: Optional[Dict[str, torch.Tensor]] = None
        errors: List[str] = []

        # Prefer the explicit API added to NECILModel for the PG-RGA contract.
        if space == "canonical":
            fn = getattr(self.model, "extract_canonical_projected_features", None)
            if callable(fn):
                for kwargs in (
                    dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical)),
                    {},
                ):
                    try:
                        out = _normalize_feature_output(fn(x, **kwargs), source="extract_canonical_projected_features")
                        break
                    except TypeError as exc:
                        errors.append(str(exc))
                        continue
        else:
            fn = getattr(self.model, "extract_adapted_projected_features", None)
            if callable(fn):
                for kwargs in (
                    dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical)),
                    {},
                ):
                    try:
                        out = _normalize_feature_output(fn(x, **kwargs), source="extract_adapted_projected_features")
                        break
                    except TypeError as exc:
                        errors.append(str(exc))
                        continue

        # Backward-compatible geometry extractor, but only if it truly supports
        # the requested space. For scoring/adapted space, failure remains fatal.
        if out is None:
            geom_fn = getattr(self.model, "extract_geometry_features", None)
            if callable(geom_fn):
                attempts = (
                    dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical), space=space, return_dict=True),
                    dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical), space=space),
                    dict(space=space, return_dict=True),
                    dict(space=space),
                )
                if space == "canonical":
                    attempts = attempts + (
                        dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical), return_dict=True),
                        dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical)),
                        dict(return_dict=True),
                        {},
                    )
                for kwargs in attempts:
                    try:
                        out = _normalize_feature_output(geom_fn(x, **kwargs), source="extract_geometry_features")
                        break
                    except TypeError as exc:
                        errors.append(str(exc))
                        continue

        if out is None and space == "scoring":
            details = errors[-1] if errors else "no adapted extractor found"
            raise RuntimeError(
                "Requested scoring/adapted GeometryBank space, but the model does not expose "
                "extract_adapted_projected_features() or a compatible extract_geometry_features(..., space='scoring'). "
                f"Last error: {details}"
            )

        # Canonical fallback for base and non-adapter incremental settings.
        if out is None:
            proj_fn = getattr(self.model, "extract_projected_features", None)
            if not callable(proj_fn):
                details = errors[-1] if errors else "no compatible extractor"
                raise AttributeError(
                    "Model must expose extract_canonical_projected_features() or extract_projected_features() "
                    f"for canonical GeometryBank rows. Last error: {details}"
                )
            for kwargs in (
                dict(spectral_summary=spectral_summary, spectral_summary_is_physical=bool(spectral_summary_is_physical)),
                {},
            ):
                try:
                    out = _normalize_feature_output(proj_fn(x, **kwargs), source="extract_projected_features")
                    break
                except TypeError as exc:
                    errors.append(str(exc))
                    continue
            if out is None:
                raise RuntimeError(f"extract_projected_features exists but did not accept compatible arguments: {errors[-1] if errors else 'unknown'}")

        feat = out["features"]
        expected_dim = int(
            getattr(
                getattr(self.model, "geometry_bank", None),
                "feature_dim",
                getattr(self.model, "d_model", getattr(self.args, "d_model", 0)),
            ) or 0
        )
        self.assert_feature_tensor(
            feat,
            expected_dim=expected_dim if expected_dim > 0 else None,
            context=f"geometry_features[{space}]",
        )
        out["geometry_feature_space"] = space
        out["classifier_feature_space"] = space
        return out

    @torch.no_grad()
    def _extract_backbone_outputs_for_class(self, cls: int, split: str = "train") -> Dict[str, torch.Tensor]:
        cls = int(cls)
        patches = self.dataset.get_class_patches(cls, split=split)
        x_cpu = patches.detach().cpu().float() if torch.is_tensor(patches) else torch.from_numpy(patches).float()
        if x_cpu.numel() == 0 or x_cpu.size(0) == 0:
            raise RuntimeError(f"No patches available for class {cls} split='{split}'.")
        if not (hasattr(self.model, "extract_geometry_features") or hasattr(self.model, "extract_projected_features")):
            raise AttributeError("SRGP GeometryBank construction requires model.extract_geometry_features() or extract_projected_features().")

        external_spectra, external_is_physical = self._get_class_external_spectra_with_flag(
            cls, split=split, expected_n=int(x_cpu.size(0))
        )
        if external_spectra is not None:
            input_channels = int(x_cpu.size(1)) if x_cpu.dim() >= 2 else 0
            spectra_dim = int(external_spectra.size(1)) if external_spectra.dim() == 2 else 0
            pca_components = int(getattr(self.args, "pca_components", 0) or 0)
            uses_pca = self._cfg_bool("use_pca", pca_components > 0)
            if uses_pca and input_channels > 0 and spectra_dim == input_channels:
                external_is_physical = False
        expected_dim = int(getattr(self.model, "d_model", getattr(self.args, "d_model", 0)))
        bs = int(max(1, getattr(self.args, "subspace_extract_batch_size", 256)))
        was_training = bool(self.model.training)
        self.model.eval()
        feats: List[torch.Tensor] = []
        base_feats: List[torch.Tensor] = []
        adapter_gates: List[torch.Tensor] = []
        spectral: List[torch.Tensor] = []
        bands: List[torch.Tensor] = []
        physical_flags: List[bool] = []
        have_band = True
        try:
            for start in range(0, x_cpu.size(0), bs):
                xb = x_cpu[start:start + bs].to(self.device, non_blocking=True)
                sb = None
                if external_spectra is not None:
                    sb = external_spectra[start:start + bs].to(self.device, non_blocking=True)
                out = self._extract_model_geometry_features(
                    xb,
                    spectral_summary=sb,
                    spectral_summary_is_physical=bool(external_is_physical),
                    space=self._bank_feature_space(),
                )
                if not isinstance(out, dict) or "features" not in out:
                    raise RuntimeError("geometry feature extraction must return dict with key 'features'.")

                feat = out["features"]
                if feat.dim() != 2 or (expected_dim > 0 and feat.size(1) != expected_dim):
                    raise RuntimeError(
                        f"GeometryBank features must be [B,{expected_dim}] in {out.get('geometry_feature_space', 'unknown')} space, "
                        f"got {tuple(feat.shape)}"
                    )
                if not torch.isfinite(feat).all():
                    raise RuntimeError(f"Non-finite GeometryBank features for class {cls}.")
                feats.append(feat.detach().cpu())
                # G²RPA contract: new-class GeometryBank rows are built from the
                # final scoring z-space; keep canonical/pre-adapter features only
                # for diagnostics and adapter-effect reports.
                bf = out.get("canonical_features", out.get("base_features", out.get("pre_adapter_features", None)))
                if torch.is_tensor(bf) and bf.shape == feat.shape:
                    base_feats.append(bf.detach().cpu())
                gate = out.get("adapter_gate", None)
                if torch.is_tensor(gate) and gate.size(0) == feat.size(0):
                    adapter_gates.append(gate.detach().cpu())

                ss, is_phys = self._resolve_batch_spectral_summary(
                    xb,
                    spectra=sb,
                    model_out=out,
                    source="dataset_raw" if sb is not None else "input",
                    spectral_summary_is_physical=bool(external_is_physical) if sb is not None else None,
                )
                if ss.dim() != 2 or ss.size(0) != xb.size(0):
                    raise RuntimeError(f"SRGP spectral_summary must be [B,S], got {tuple(ss.shape)}")
                spectral.append(ss.detach().cpu())
                physical_flags.append(bool(is_phys))

                # Prefer the model-computed band_summary because NECILModel
                # builds it from the same spectral_summary that is passed into
                # this call. Backbone/raw band_weights may still live in the
                # reduced PCA input space (e.g. 30), while raw spectral metadata
                # lives in physical wavelength space (e.g. 200). GeometryBank
                # currently has one spectral/band descriptor width, so never
                # pass a reduced-band vector together with raw spectral curves.
                bw = out.get("band_summary", out.get("band_importance", None))
                if not (torch.is_tensor(bw) and bw.dim() == 2 and bw.size(0) == xb.size(0) and bw.numel() > 0):
                    bw = out.get("band_weights", None)

                if torch.is_tensor(bw) and bw.dim() == 2 and bw.size(0) == xb.size(0) and bw.numel() > 0:
                    if torch.is_tensor(ss) and ss.dim() == 2 and ss.size(0) == xb.size(0) and ss.size(1) > 0 and bw.size(1) != ss.size(1):
                        # Raw spectral descriptor and reduced PCA-band descriptor
                        # cannot share the same GeometryBank descriptor matrix.
                        # Keep the raw spectral descriptor; GeometryBank will
                        # derive a stable band_importance from it.
                        have_band = False
                    else:
                        bands.append(bw.detach().cpu())
                else:
                    have_band = False
        finally:
            self.model.train(was_training)

        features = torch.cat(feats, dim=0).to(self.device)
        spectral_summary = torch.cat(spectral, dim=0).to(self.device)
        spectral_is_physical = bool(physical_flags) and all(physical_flags)
        out = {
            "features": features,
            "spectral_summary": spectral_summary,
            "spectral_summary_is_physical": torch.tensor(float(spectral_is_physical), device=self.device),
        }
        if len(base_feats) == len(feats):
            out["base_features"] = torch.cat(base_feats, dim=0).to(self.device)
        if len(adapter_gates) == len(feats):
            out["adapter_gate"] = torch.cat(adapter_gates, dim=0).to(self.device)
        if have_band and len(bands) == len(feats):
            band_weights = torch.cat(bands, dim=0).to(self.device)
            # Band weights live in model-input band space (often PCA=30), while
            # spectral_summary may be raw physical space (e.g., 200 bands). Do
            # not drop valid band weights just because those dimensions differ.
            if band_weights.size(0) == features.size(0) and band_weights.dim() == 2 and band_weights.size(1) > 0 and torch.isfinite(band_weights).all():
                out["band_weights"] = band_weights
        return out

    @torch.no_grad()
    def _extract_class_geometry_dict(self, cls: int, split: str = "train") -> Dict[str, torch.Tensor]:
        cls = int(cls)
        outs = self._extract_backbone_outputs_for_class(cls, split=split)
        features = outs["features"]
        labels = torch.full((features.size(0),), cls, device=features.device, dtype=torch.long)
        bank = getattr(self.model, "geometry_bank", None)
        if bank is None or not hasattr(bank, "extract_geometry"):
            raise AttributeError("model.geometry_bank.extract_geometry() is required.")
        spectral_for_bank = outs.get("spectral_summary", None)
        band_for_bank = outs.get("band_weights", None)
        if (
            torch.is_tensor(spectral_for_bank)
            and torch.is_tensor(band_for_bank)
            and spectral_for_bank.dim() == 2
            and band_for_bank.dim() == 2
            and spectral_for_bank.size(0) == band_for_bank.size(0)
            and spectral_for_bank.size(1) > 0
            and band_for_bank.size(1) > 0
            and spectral_for_bank.size(1) != band_for_bank.size(1)
        ):
            # This is the exact PCA/raw-spectra failure mode: spectral_summary is
            # raw physical spectra (IP: 200), while band_weights are PCA/reduced
            # channels (IP: 30). GeometryBank has one descriptor width, so the
            # correct PG-RGA behavior is to keep the physical spectral descriptor
            # and let GeometryBank derive band_importance from it, not to mix
            # 200-D and 30-D descriptors in the same storage.
            band_for_bank = None

        try:
            geom = bank.extract_geometry(
                features=features,
                labels=labels,
                spectral_summary=spectral_for_bank,
                band_weights=band_for_bank,
                spectral_summary_is_physical=bool(torch.as_tensor(outs.get("spectral_summary_is_physical", 0.0)).detach().cpu().item()),
            )
        except TypeError:
            geom = bank.extract_geometry(
                features=features,
                labels=labels,
                spectral_summary=spectral_for_bank,
                band_weights=band_for_bank,
            )
        if cls not in geom:
            raise RuntimeError(f"Geometry extraction failed for class {cls}.")
        g = geom[cls]
        required = ("mean", "basis", "eigvals", "res_var", "active_rank", "reliability", "sample_count")
        missing = [k for k in required if k not in g]
        if missing:
            raise RuntimeError(f"Geometry extraction for class {cls} missing keys: {missing}")
        return g

    @torch.no_grad()
    def _extract_class_geometry(self, cls: int, split: str = "train", rank=None):
        del rank
        g = self._extract_class_geometry_dict(cls, split=split)
        return (
            g["mean"], g["basis"], g["eigvals"], g["res_var"],
            None, g.get("band_importance", None), g["active_rank"], g["reliability"], g["sample_count"],
        )

    # ------------------------------------------------------------------
    # Descriptor-refinement support for clean incremental phase
    # ------------------------------------------------------------------
    def _make_refined_bank_view(
        self,
        class_ids: Iterable[int],
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        *,
        active_ranks: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
        band_importances: Optional[torch.Tensor] = None,
        spectral_curve_means: Optional[torch.Tensor] = None,
        spectral_curve_vars: Optional[torch.Tensor] = None,
        spectral_curve_d1: Optional[torch.Tensor] = None,
        spectral_curve_d2: Optional[torch.Tensor] = None,
        spectral_shape_reliability: Optional[torch.Tensor] = None,
        base_bank: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a temporary scoring bank with only selected rows replaced.

        This is used by descriptor-only refinement. It does not write to
        GeometryBank. Old rows are cloned from the frozen bank and remain
        unchanged.
        """
        ids = self._as_class_list(class_ids)
        if not ids:
            raise RuntimeError("Cannot create refined bank view with empty class_ids.")
        bank = self._canonicalize_bank(base_bank if base_bank is not None else self._safe_get_subspace_bank(require_ready=True))
        out: Dict[str, torch.Tensor] = {}
        for key, value in bank.items():
            if torch.is_tensor(value):
                out[key] = value.detach().clone()
            else:
                out[key] = value

        means = means.to(device=out["means"].device, dtype=out["means"].dtype)
        bases = bases.to(device=out["bases"].device, dtype=out["bases"].dtype)
        variances = variances.to(device=out["variances"].device, dtype=out["variances"].dtype)
        if means.dim() != 2 or means.size(0) != len(ids):
            raise RuntimeError(f"refined means must be [K,D], got {tuple(means.shape)} for K={len(ids)}")
        if bases.dim() != 3 or bases.size(0) != len(ids):
            raise RuntimeError(f"refined bases must be [K,D,R], got {tuple(bases.shape)} for K={len(ids)}")
        if variances.dim() != 2 or variances.size(0) != len(ids):
            raise RuntimeError(f"refined variances must be [K,R+1], got {tuple(variances.shape)} for K={len(ids)}")

        max_id = max(ids)
        if max_id >= out["means"].size(0):
            raise RuntimeError(f"Refined class id {max_id} exceeds GeometryBank rows {out['means'].size(0)}")
        out["means"][ids] = means
        out["bases"][ids] = bases
        out["variances"][ids] = variances
        out["eigvals"] = out["variances"][:, :-1]
        out["res_vars"] = out["variances"][:, -1]
        out["resvars"] = out["res_vars"]

        if active_ranks is not None and torch.is_tensor(active_ranks) and "active_ranks" in out:
            out["active_ranks"][ids] = active_ranks.to(device=out["active_ranks"].device).long().flatten()
        if reliability is not None and torch.is_tensor(reliability) and "reliability" in out:
            out["reliability"][ids] = reliability.to(device=out["reliability"].device, dtype=out["reliability"].dtype).flatten()
        if sample_counts is not None and torch.is_tensor(sample_counts) and "sample_counts" in out:
            out["sample_counts"][ids] = sample_counts.to(device=out["sample_counts"].device, dtype=out["sample_counts"].dtype).flatten()
        if band_importances is not None and torch.is_tensor(band_importances) and "band_importances" in out:
            if out["band_importances"].dim() == 2 and out["band_importances"].size(1) == band_importances.size(1):
                out["band_importances"][ids] = band_importances.to(device=out["band_importances"].device, dtype=out["band_importances"].dtype)
                out["band_importance"] = out["band_importances"]

        spectral_rows = {
            "spectral_curve_means": spectral_curve_means,
            "spectral_curve_vars": spectral_curve_vars,
            "spectral_curve_d1": spectral_curve_d1,
            "spectral_curve_d2": spectral_curve_d2,
        }
        for key, val in spectral_rows.items():
            if val is not None and torch.is_tensor(val) and key in out and torch.is_tensor(out[key]) and out[key].numel() > 0:
                if out[key].dim() == 2 and out[key].size(1) == val.size(1):
                    out[key][ids] = val.to(device=out[key].device, dtype=out[key].dtype)
        if spectral_shape_reliability is not None and torch.is_tensor(spectral_shape_reliability) and "spectral_shape_reliability" in out:
            out["spectral_shape_reliability"][ids] = spectral_shape_reliability.to(
                device=out["spectral_shape_reliability"].device,
                dtype=out["spectral_shape_reliability"].dtype,
            ).flatten()

        out = self._canonicalize_bank(out)
        out["valid_mask"] = self._valid_mask_from_bank(out).to(out["means"].device)
        return out

    @torch.no_grad()
    def commit_new_class_rows_only(
        self,
        class_ids: Iterable[int],
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        *,
        active_ranks: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
        feature_reliability: Optional[torch.Tensor] = None,
        band_importances: Optional[torch.Tensor] = None,
        band_reliability: Optional[torch.Tensor] = None,
        spectral_curve_means: Optional[torch.Tensor] = None,
        spectral_curve_vars: Optional[torch.Tensor] = None,
        spectral_curve_d1: Optional[torch.Tensor] = None,
        spectral_curve_d2: Optional[torch.Tensor] = None,
        spectral_shape_reliability: Optional[torch.Tensor] = None,
        context: str = "commit_new_class_rows_only",
    ) -> None:
        ids = self._as_class_list(class_ids)
        if not ids:
            return
        phase = int(getattr(self.model, "current_phase", 0))
        old_class_count = int(getattr(self.model, "old_class_count", 0))
        if phase > 0:
            old_bad = [c for c in ids if c < old_class_count]
            if old_bad:
                raise RuntimeError(f"{context}: attempted to commit into frozen old rows: {old_bad}")
            if hasattr(self.dataset, "phase_to_classes") and phase in self.dataset.phase_to_classes:
                phase_ids = set(int(c) for c in self.dataset.phase_to_classes[phase])
                future_bad = [c for c in ids if c not in phase_ids]
                if future_bad:
                    raise RuntimeError(f"{context}: attempted to commit rows not in current phase {phase}: {future_bad}")

        bank_before = self._safe_get_subspace_bank(require_ready=True)
        old_ids = list(range(max(old_class_count, 0))) if phase > 0 else []
        old_snapshot = self.snapshot_bank_rows(bank_before, old_ids)

        means = means.detach().to(self.device)
        bases = bases.detach().to(self.device)
        variances = variances.detach().to(self.device)
        K = len(ids)
        if means.dim() != 2 or means.size(0) != K:
            raise RuntimeError(f"{context}: means must be [K,D], got {tuple(means.shape)} for K={K}")
        if bases.dim() != 3 or bases.size(0) != K or bases.size(1) != means.size(1):
            raise RuntimeError(f"{context}: bases must be [K,D,R], got {tuple(bases.shape)}")
        if variances.dim() != 2 or variances.size(0) != K or variances.size(1) != bases.size(2) + 1:
            raise RuntimeError(f"{context}: variances must be [K,R+1], got {tuple(variances.shape)}")
        if not torch.isfinite(means).all() or not torch.isfinite(bases).all() or not torch.isfinite(variances).all():
            raise RuntimeError(f"{context}: descriptors contain NaN/Inf.")
        if bool((variances < 0).any().item()):
            raise RuntimeError(f"{context}: variances must be non-negative.")
        R = int(bases.size(2))
        eye = torch.eye(R, device=bases.device, dtype=bases.dtype)
        for i in range(K):
            gram = bases[i].transpose(0, 1).matmul(bases[i])
            if not torch.allclose(gram, eye, atol=float(getattr(self.args, "bank_basis_ortho_atol", 2e-3)), rtol=0.0):
                raise RuntimeError(f"{context}: new basis row for class {ids[i]} is not orthonormal.")

        def rows_for_ids(t):
            if t is None or not torch.is_tensor(t):
                return None
            tt = t.detach().to(self.device)
            if tt.dim() == 0:
                return tt
            if tt.size(0) == K:
                return tt
            if tt.size(0) > max(ids):
                return tt[ids]
            raise RuntimeError(f"{context}: tensor first dim {tt.size(0)} cannot provide rows for class ids {ids}")

        gb = getattr(self.model, "geometry_bank", None)
        var_floor = float(getattr(self.args, "geom_var_floor", 1e-4))
        if gb is not None and hasattr(gb, "apply_refined_feature_rows"):
            gb.apply_refined_feature_rows(
                ids,
                means=means,
                bases=bases,
                eigvals=variances[:, :-1].clamp_min(var_floor),
                res_vars=variances[:, -1].clamp_min(var_floor),
                reliability=rows_for_ids(reliability),
                feature_reliability=rows_for_ids(feature_reliability),
                active_ranks=rows_for_ids(active_ranks),
                sample_counts=rows_for_ids(sample_counts),
                band_importances=rows_for_ids(band_importances),
                band_reliability=rows_for_ids(band_reliability),
                spectral_curve_means=rows_for_ids(spectral_curve_means),
                spectral_curve_vars=rows_for_ids(spectral_curve_vars),
                spectral_curve_d1=rows_for_ids(spectral_curve_d1),
                spectral_curve_d2=rows_for_ids(spectral_curve_d2),
                spectral_shape_reliability=rows_for_ids(spectral_shape_reliability),
                allow_frozen_update=False,
            )
        else:
            if not hasattr(self.model, "refresh_class_subspace"):
                raise AttributeError("Model must expose geometry_bank.apply_refined_feature_rows() or refresh_class_subspace().")
            def row_or_none(t, idx):
                if t is None or not torch.is_tensor(t):
                    return None
                tt = t.detach().to(self.device)
                if tt.dim() == 0:
                    return tt
                if tt.size(0) == K:
                    return tt[idx]
                if tt.size(0) > max(ids):
                    return tt[ids[idx]]
                return None
            for local_idx, cls in enumerate(ids):
                self.model.refresh_class_subspace(
                    cls=int(cls),
                    mean=means[local_idx],
                    basis=bases[local_idx],
                    eigvals=variances[local_idx, :-1].clamp_min(var_floor),
                    res_var=variances[local_idx, -1].clamp_min(var_floor),
                    active_rank=row_or_none(active_ranks, local_idx) if row_or_none(active_ranks, local_idx) is not None else torch.tensor(R, device=self.device, dtype=torch.long),
                    reliability=row_or_none(reliability, local_idx) if row_or_none(reliability, local_idx) is not None else torch.tensor(1.0, device=self.device, dtype=means.dtype),
                    sample_count=row_or_none(sample_counts, local_idx) if row_or_none(sample_counts, local_idx) is not None else torch.tensor(1.0, device=self.device, dtype=means.dtype),
                    feature_reliability=row_or_none(feature_reliability, local_idx),
                    band_importance=row_or_none(band_importances, local_idx),
                    band_reliability=row_or_none(band_reliability, local_idx),
                    spectral_curve_mean=row_or_none(spectral_curve_means, local_idx),
                    spectral_curve_var=row_or_none(spectral_curve_vars, local_idx),
                    spectral_curve_d1=row_or_none(spectral_curve_d1, local_idx),
                    spectral_curve_d2=row_or_none(spectral_curve_d2, local_idx),
                    spectral_shape_reliability=row_or_none(spectral_shape_reliability, local_idx),
                )

        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None and hasattr(gb, "validate_consistency"):
            gb.validate_consistency(strict=True)
        bank_after = self._safe_get_subspace_bank(require_ready=True)
        if old_ids:
            self.assert_bank_rows_unchanged(old_snapshot, bank_after, old_ids, context=context)
        self.assert_bank_ready_for_seen_classes(bank_after, ids)
        counts = bank_after["sample_counts"].to(self.device)
        bad_counts = [c for c in ids if float(counts[c].detach().item()) <= 0.0]
        if bad_counts:
            raise RuntimeError(f"{context}: committed new rows have non-positive sample counts: {bad_counts}")

    @torch.no_grad()
    def _commit_refined_feature_rows(
        self,
        class_ids: Iterable[int],
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        *,
        active_ranks: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
        feature_reliability: Optional[torch.Tensor] = None,
        band_importances: Optional[torch.Tensor] = None,
        band_reliability: Optional[torch.Tensor] = None,
        spectral_curve_means: Optional[torch.Tensor] = None,
        spectral_curve_vars: Optional[torch.Tensor] = None,
        spectral_curve_d1: Optional[torch.Tensor] = None,
        spectral_curve_d2: Optional[torch.Tensor] = None,
        spectral_shape_reliability: Optional[torch.Tensor] = None,
        context: str = "descriptor_refinement",
    ) -> None:
        return self.commit_new_class_rows_only(
            class_ids,
            means,
            bases,
            variances,
            active_ranks=active_ranks,
            reliability=reliability,
            sample_counts=sample_counts,
            feature_reliability=feature_reliability,
            band_importances=band_importances,
            band_reliability=band_reliability,
            spectral_curve_means=spectral_curve_means,
            spectral_curve_vars=spectral_curve_vars,
            spectral_curve_d1=spectral_curve_d1,
            spectral_curve_d2=spectral_curve_d2,
            spectral_shape_reliability=spectral_shape_reliability,
            context=context,
        )

    # ------------------------------------------------------------------
    # Overlap-aware admission for incremental new rows
    # ------------------------------------------------------------------
    def _overlap_cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self, name, getattr(self.args, name, default)))

    def _overlap_cfg_int(self, name: str, default: int) -> int:
        return int(getattr(self, name, getattr(self.args, name, default)))

    def _overlap_cfg_bool(self, name: str, default: bool) -> bool:
        value = getattr(self, name, getattr(self.args, name, default))
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _safe_active_rank_scalar(self, active_rank, fallback: int) -> int:
        try:
            ar = int(torch.as_tensor(active_rank).detach().cpu().item())
        except Exception:
            ar = int(fallback)
        return max(0, min(ar, int(fallback)))

    @torch.no_grad()
    def _candidate_old_overlap_risk(
        self,
        *,
        mean: torch.Tensor,
        basis: torch.Tensor,
        active_rank,
        band_importance: Optional[torch.Tensor] = None,
        spectral_shape: Optional[Dict[str, torch.Tensor]] = None,
        spectral_curve_d1: Optional[torch.Tensor] = None,
        old_class_count: Optional[int] = None,
    ) -> Dict[str, object]:
        """Measure how much a candidate new row overlaps frozen old rows.

        SRGP risk is descriptor-only and HSI-aware:
            center proximity + feature-subspace overlap + band similarity
            + spectral-shape similarity.
        """
        old_class_count = int(getattr(self.model, "old_class_count", 0) if old_class_count is None else old_class_count)
        if old_class_count <= 0:
            return {"max_risk": 0.0, "top": None, "pairs": []}
        try:
            bank = self._safe_get_subspace_bank(require_ready=True)
        except Exception:
            return {"max_risk": 0.0, "top": None, "pairs": []}

        bank = self._canonicalize_bank(bank)
        means = bank.get("means", None)
        bases = bank.get("bases", None)
        ranks = bank.get("active_ranks", None)
        counts = bank.get("sample_counts", None)
        rel = bank.get("reliability", None)
        bands = bank.get("band_importances", bank.get("band_importance", None))
        old_d1 = bank.get("spectral_curve_d1", None)
        if not (torch.is_tensor(means) and torch.is_tensor(bases)):
            return {"max_risk": 0.0, "top": None, "pairs": []}

        old_count = min(old_class_count, int(means.size(0)))
        valid_old = torch.ones(old_count, device=means.device, dtype=torch.bool)
        if torch.is_tensor(counts) and counts.numel() >= old_count:
            valid_old = counts[:old_count].to(means.device).flatten() > 0
        if not bool(valid_old.any().item()):
            return {"max_risk": 0.0, "top": None, "pairs": []}

        dtype = means.dtype
        device = means.device
        mu_new = torch.as_tensor(mean, device=device, dtype=dtype).flatten()
        U_new_full = torch.as_tensor(basis, device=device, dtype=dtype)
        if U_new_full.dim() != 2:
            return {"max_risk": 0.0, "top": None, "pairs": []}
        R = int(U_new_full.size(1))
        rn = self._safe_active_rank_scalar(active_rank, R)
        U_new = U_new_full[:, :rn] if rn > 0 else torch.empty((U_new_full.size(0), 0), device=device, dtype=dtype)

        bw_new = None
        if band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
            bw_new = torch.as_tensor(band_importance, device=device, dtype=dtype).flatten().clamp_min(0.0)
            if bw_new.sum() > 1e-8:
                bw_new = bw_new / bw_new.norm().clamp_min(1e-8)
            else:
                bw_new = None

        d1_new = None
        if spectral_curve_d1 is not None and torch.as_tensor(spectral_curve_d1).numel() > 0:
            d1_new = torch.as_tensor(spectral_curve_d1, device=device, dtype=dtype).flatten()
        elif isinstance(spectral_shape, dict):
            for key in ("spectral_curve_d1", "spectral_d1"):
                val = spectral_shape.get(key, None)
                if torch.is_tensor(val) and val.numel() > 0:
                    d1_new = val.to(device=device, dtype=dtype).flatten()
                    break
        if d1_new is not None:
            d1_new = F.normalize(torch.nan_to_num(d1_new, nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()

        center_w = self._overlap_cfg_float("overlap_admission_center_weight", 0.40)
        subspace_w = self._overlap_cfg_float("overlap_admission_subspace_weight", 0.35)
        band_w = self._overlap_cfg_float("overlap_admission_band_weight", 0.10)
        spec_w = self._overlap_cfg_float("overlap_admission_spectral_shape_weight", 0.15)
        dscale = max(float(mu_new.numel()) ** 0.5, 1.0)

        pairs: List[Dict[str, object]] = []
        for old_cls in range(old_count):
            if not bool(valid_old[old_cls].item()):
                continue
            dist = torch.norm(means[old_cls].to(device=device, dtype=dtype) - mu_new, p=2) / dscale
            center_risk = torch.exp(-dist).clamp(0.0, 1.0)

            ro = int(ranks[old_cls].detach().item()) if torch.is_tensor(ranks) and ranks.numel() > old_cls else int(bases.size(2))
            ro = max(0, min(ro, int(bases.size(2))))
            subspace_overlap = torch.tensor(0.0, device=device, dtype=dtype)
            if ro > 0 and rn > 0:
                U_old = bases[old_cls, :, :ro].to(device=device, dtype=dtype)
                denom = float(max(min(ro, rn), 1))
                subspace_overlap = (U_old.t() @ U_new).pow(2).sum() / denom
                subspace_overlap = subspace_overlap.clamp(0.0, 1.0)

            band_sim = torch.tensor(0.0, device=device, dtype=dtype)
            if bw_new is not None and torch.is_tensor(bands) and bands.dim() == 2 and bands.size(0) > old_cls and bands.size(1) == bw_new.numel():
                bo = bands[old_cls].to(device=device, dtype=dtype).flatten().clamp_min(0.0)
                if bo.sum() > 1e-8:
                    bo = bo / bo.norm().clamp_min(1e-8)
                    band_sim = torch.dot(bo, bw_new).clamp(0.0, 1.0)

            spec_sim = torch.tensor(0.0, device=device, dtype=dtype)
            if d1_new is not None and torch.is_tensor(old_d1) and old_d1.dim() == 2 and old_d1.size(0) > old_cls and old_d1.size(1) == d1_new.numel():
                od = F.normalize(torch.nan_to_num(old_d1[old_cls].to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()
                spec_sim = torch.dot(od, d1_new).clamp(0.0, 1.0)

            risk = center_w * center_risk + subspace_w * subspace_overlap + band_w * band_sim + spec_w * spec_sim
            if torch.is_tensor(rel) and rel.numel() > old_cls:
                uncertainty = (1.0 - rel[old_cls].to(device=device, dtype=dtype).clamp(0.05, 1.0)).clamp(0.0, 1.0)
                risk = risk * (1.0 + 0.25 * uncertainty)
            risk = risk.clamp_min(0.0)

            pairs.append(
                {
                    "old_class": int(old_cls),
                    "old_name": self._class_name(old_cls),
                    "feature_center_distance": float((dist * dscale).detach().cpu().item()),
                    "scaled_center_distance": float(dist.detach().cpu().item()),
                    "center_risk": float(center_risk.detach().cpu().item()),
                    "feature_overlap": float(subspace_overlap.detach().cpu().item()),
                    "band_similarity": float(band_sim.detach().cpu().item()),
                    "spectral_shape_similarity": float(spec_sim.detach().cpu().item()),
                    "risk_score": float(risk.detach().cpu().item()),
                }
            )

        pairs.sort(key=lambda r: float(r["risk_score"]), reverse=True)
        top = pairs[0] if pairs else None
        return {"max_risk": float(top["risk_score"]) if top else 0.0, "top": top, "pairs": pairs}

    @torch.no_grad()
    def _apply_overlap_aware_admission(self, cls: int, g: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Diagnostic-only overlap check for a candidate new row.

        The previous helper actively shrank new eigenspectra and capped rank when
        a candidate overlapped old rows. That is an ad-hoc admission heuristic,
        not the clean NECIL-HSI method. The clean path now leaves the statistical
        descriptor untouched at insertion time; overlap is handled by
        descriptor-only refinement in ``incremental_phase_trainer.py`` and by
        diagnostics reported after the phase.

        This method only records/prints risk information if requested. It never
        mutates ``mean``, ``basis``, ``eigvals``, ``res_var``, or ``active_rank``.
        """
        phase = int(getattr(self.model, "current_phase", 0))
        old_class_count = int(getattr(self.model, "old_class_count", 0))
        cls = int(cls)
        if phase <= 0 or old_class_count <= 0 or cls < old_class_count:
            return g

        risk_info = self._candidate_old_overlap_risk(
            mean=g["mean"],
            basis=g["basis"],
            active_rank=g.get("active_rank", None),
            band_importance=g.get("band_importance", None),
            spectral_shape=g.get("spectral_shape", None),
            spectral_curve_d1=g.get("spectral_curve_d1", None),
            old_class_count=old_class_count,
        )
        max_risk = float(risk_info.get("max_risk", 0.0))
        threshold = self._overlap_cfg_float("overlap_admission_risk_threshold", 0.80)
        event = {
            "phase": int(phase),
            "new_class": int(cls),
            "new_name": self._class_name(cls),
            "max_old_overlap_risk": float(max_risk),
            "admission_gate": 1.0,
            "descriptor_mutated": False,
            "threshold": float(threshold),
            "top_old": risk_info.get("top"),
        }
        events = getattr(self, "_last_overlap_admission_events", [])
        events.append(event)
        self._last_overlap_admission_events = events[-50:]
        if (max_risk >= threshold and self._overlap_cfg_bool("print_overlap_admission", True)) or bool(getattr(self, "debug", False)):
            top = event.get("top_old") or {}
            print(
                f"[OverlapDiagnostic] phase={phase} new={cls}({self._class_name(cls)}) "
                f"risk={max_risk:.4f} threshold={threshold:.3f} "
                f"top_old={top.get('old_class', 'NA')}({top.get('old_name', 'NA')}) | "
                "descriptor_mutated=False"
            )
        return g

    @torch.no_grad()
    @torch.no_grad()
    def _build_class_memory_from_current_phase(self, cls: int, split: str = "train") -> None:
        cls = int(cls)
        phase = int(getattr(self.model, "current_phase", 0))
        old_class_count = int(getattr(self.model, "old_class_count", 0))
        if phase > 0 and cls < old_class_count:
            raise RuntimeError(f"Attempted to rebuild frozen old class {cls} during incremental phase {phase}.")
        if hasattr(self.dataset, "phase_to_classes") and phase in self.dataset.phase_to_classes:
            phase_ids = set(int(c) for c in self.dataset.phase_to_classes[phase])
            if cls not in phase_ids:
                raise RuntimeError(f"Attempted to build class {cls} outside current phase {phase} classes {sorted(phase_ids)}.")

        old_snapshot = None
        old_ids = list(range(max(old_class_count, 0))) if phase > 0 else []
        if old_ids:
            old_snapshot = self.snapshot_bank_rows(self._safe_get_subspace_bank(require_ready=True), old_ids)

        g = self._extract_class_geometry_dict(cls, split=split)
        g = self._apply_overlap_aware_admission(cls, g)
        if not hasattr(self.model, "refresh_class_subspace"):
            raise AttributeError("Model must expose refresh_class_subspace().")

        # PG-RGA bank write contract:
        #   base phase -> canonical geometry rows;
        #   incremental phase -> current new rows only;
        #   old rows must remain immutable;
        #   spectral_prototype/band_importance are stored as descriptors only,
        #   not as classifier branches.
        self.model.refresh_class_subspace(
            cls=cls,
            mean=g["mean"],
            basis=g["basis"],
            eigvals=g["eigvals"],
            res_var=g["res_var"],
            active_rank=g["active_rank"],
            reliability=g["reliability"],
            sample_count=g["sample_count"],
            feature_reliability=g.get("feature_reliability", g.get("reliability", None)),
            band_importance=g.get("band_importance", None),
            band_reliability=g.get("band_reliability", None),
            spectral_prototype=g.get("spectral_prototype", g.get("spectral_proto", None)),
            spectral_reliability=g.get("spectral_reliability", g.get("spectral_shape_reliability", None)),
            phase_created=phase,
            allow_frozen_update=False,
        )

        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None and hasattr(gb, "validate_consistency"):
            gb.validate_consistency(strict=True)
        if old_ids and old_snapshot is not None:
            self.assert_bank_rows_unchanged(
                old_snapshot,
                self._safe_get_subspace_bank(require_ready=True),
                old_ids,
                context=f"build_class_memory_phase_{phase}_cls_{cls}",
            )

    @torch.no_grad()
    def _infer_spectral_dim_from_dataset(self, class_ids: List[int], split: str = "train") -> int:
        """Infer raw spectral capacity when available, otherwise input-band dim."""
        for cls in class_ids:
            try:
                patches = self.dataset.get_class_patches(int(cls), split=split)
                n = int(patches.shape[0] if hasattr(patches, "shape") else patches.size(0))
                spectra, physical = self._get_class_external_spectra_with_flag(int(cls), split=split, expected_n=n)
                if torch.is_tensor(spectra) and spectra.dim() == 2 and spectra.size(1) > 0 and bool(physical):
                    return int(spectra.size(1))
            except Exception:
                pass
        for cls in class_ids:
            try:
                patches = self.dataset.get_class_patches(int(cls), split=split)
                shape = patches.shape if hasattr(patches, "shape") else tuple(patches.size())
                if len(shape) >= 2:
                    return int(shape[1])
            except Exception:
                continue
        return 0

    @torch.no_grad()
    def _bootstrap_phase_classes(self, phase: int, split: str = "train", force_rebuild: bool = False) -> None:
        phase = int(phase)
        class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
        if not class_ids:
            return
        spectral_dim = self._infer_spectral_dim_from_dataset(class_ids, split=split)
        if hasattr(self.model, "ensure_class_capacity"):
            self.model.ensure_class_capacity(max(class_ids) + 1, spectral_dim=spectral_dim)
        ctx = self.dataset.memory_build_context(phase) if hasattr(self.dataset, "memory_build_context") else nullcontext()
        with ctx:
            for cls in class_ids:
                if force_rebuild or not self._class_memory_is_valid(cls):
                    self._build_class_memory_from_current_phase(cls, split=split)

    @torch.no_grad()
    def _finalize_phase_memory(self, phase: int, split: str = "train") -> None:
        phase = int(phase)
        phase_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
        ctx = self.dataset.memory_build_context(phase) if hasattr(self.dataset, "memory_build_context") else nullcontext()
        with ctx:
            for cls in phase_ids:
                self._build_class_memory_from_current_phase(cls, split=split)
        if hasattr(self.dataset, "finalize_phase"):
            self.dataset.finalize_phase(phase)

        # Freeze exactly the classes that are now seen.  Do not rely on
        # freeze_classes_up_to(max+1) unless the bank has no explicit class-id
        # freezer, because non-contiguous schedules would freeze unused capacity.
        seen = self._seen_class_ids_through_phase(phase)
        if hasattr(self.model, "freeze_classes"):
            self.model.freeze_classes(seen)
        elif hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes"):
            self.model.geometry_bank.freeze_classes(seen)
        elif hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes_up_to") and seen:
            self.model.geometry_bank.freeze_classes_up_to(max(seen) + 1)

    @torch.no_grad()
    def _refresh_classes_for_validation(self, phase: int, class_ids: Iterable[int], split: str = "train", force_rebuild: bool = True) -> None:
        if not bool(getattr(self, "refresh_before_validation", getattr(self.args, "refresh_before_validation", True))):
            return
        phase = int(phase)
        class_ids = self._as_class_list(class_ids)
        old_training_state = bool(self.model.training)
        old_class_count = int(getattr(self.model, "old_class_count", 0))
        ctx = self.dataset.memory_build_context(phase) if hasattr(self.dataset, "memory_build_context") else nullcontext()
        with ctx:
            for cls in class_ids:
                if phase > 0 and int(cls) < old_class_count:
                    raise RuntimeError(f"Attempted to refresh old class {cls} during incremental phase {phase}.")
                if force_rebuild or not self._class_memory_is_valid(cls):
                    self._build_class_memory_from_current_phase(cls, split=split)
        self.model.train(old_training_state)

    def _should_refresh_for_validation(self, epoch: int) -> bool:
        if not bool(getattr(self, "refresh_before_validation", getattr(self.args, "refresh_before_validation", True))):
            return False
        every = int(getattr(self, "validation_refresh_every", getattr(self.args, "validation_refresh_every", 1)))
        return every > 0 and ((int(epoch) + 1) % every == 0)

    def _seen_class_ids_before_phase(self, phase: int) -> List[int]:
        ids: List[int] = []
        for p in range(max(int(phase), 0)):
            ids.extend(int(c) for c in self.dataset.phase_to_classes[p])
        return sorted(set(ids))

    def _seen_class_ids_through_phase(self, phase: int) -> List[int]:
        ids: List[int] = []
        for p in range(max(int(phase), 0) + 1):
            ids.extend(int(c) for c in self.dataset.phase_to_classes[p])
        return sorted(set(ids))

    # ------------------------------------------------------------------
    # Old-bank snapshot/integrity
    # ------------------------------------------------------------------
    def _snapshot_old_bank(self, old_class_count: int) -> Dict[str, torch.Tensor]:
        old_class_count = int(old_class_count)
        bank = self._safe_get_subspace_bank(require_ready=True)
        keep = (
            "means", "bases", "eigvals", "res_vars", "variances", "active_ranks",
            "reliability", "feature_reliability", "sample_counts", "band_importances",
            "band_reliability", "spectral_curve_means", "spectral_curve_vars",
            "spectral_curve_d1", "spectral_curve_d2", "spectral_shape_reliability",
        )
        snap: Dict[str, torch.Tensor] = {}
        for key in keep:
            v = bank.get(key, None)
            if torch.is_tensor(v):
                snap[key] = v[:old_class_count].detach().clone() if v.dim() > 0 else v.detach().clone()
        snap = self._canonicalize_bank(snap)
        missing = [k for k in ("means", "bases", "variances", "active_ranks", "reliability", "sample_counts") if k not in snap]
        if missing:
            raise RuntimeError(f"Old-bank snapshot missing keys: {missing}")
        return snap

    @torch.no_grad()
    def _old_bank_integrity_snapshot(self, old_class_ids: Iterable[int]) -> Dict[str, torch.Tensor]:
        ids = self._as_class_list(old_class_ids)
        if not ids:
            return {}
        bank = self._safe_get_subspace_bank(require_ready=True)
        out: Dict[str, torch.Tensor] = {}
        for key in (
            "means", "bases", "variances", "active_ranks", "reliability",
            "sample_counts", "band_importances", "spectral_curve_means",
            "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2",
            "spectral_shape_reliability",
        ):
            v = bank.get(key, None)
            if torch.is_tensor(v) and v.dim() > 0 and v.size(0) > max(ids):
                out[key] = v[ids].detach().clone()
        return out

    @torch.no_grad()
    def _assert_old_bank_integrity(self, old_class_ids: Iterable[int], snapshot: Dict[str, torch.Tensor], *, context: str, atol: float = 1e-6) -> None:
        ids = self._as_class_list(old_class_ids)
        if not ids or not snapshot:
            return
        bank = self._safe_get_subspace_bank(require_ready=True)
        bad: List[str] = []
        for key, old_v in snapshot.items():
            cur = bank.get(key, None)
            if not torch.is_tensor(cur) or cur.dim() == 0 or cur.size(0) <= max(ids):
                bad.append(f"{key}:missing")
                continue
            cur_v = cur[ids].detach().to(device=old_v.device, dtype=old_v.dtype)
            if cur_v.shape != old_v.shape or not torch.allclose(cur_v, old_v, atol=float(atol), rtol=0.0):
                diff = float((cur_v - old_v).abs().max().item()) if cur_v.shape == old_v.shape else float("inf")
                bad.append(f"{key}:maxdiff={diff:.3e}")
        if bad:
            raise RuntimeError(f"Old GeometryBank rows changed during {context}. Mutated tensors: {bad[:12]}")

    def _make_loss_scoring_bank(self, raw_bank: Optional[Dict[str, torch.Tensor]] = None, old_class_count: Optional[int] = None, classifier_mode: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Return an explicit scoring-bank view for losses/diagnostics.

        This function intentionally does not call model.compute_energy_from_features;
        trainer-side candidate banks and refined banks must be
        scored against the tensors passed here, not against the model's internal
        GeometryBank by accident.
        """
        del old_class_count, classifier_mode
        bank = self._canonicalize_bank(raw_bank if raw_bank is not None else self._safe_get_subspace_bank(require_ready=True))
        # Clone tensors so temporary scoring views cannot mutate the live bank.
        out: Dict[str, torch.Tensor] = {}
        for key, value in bank.items():
            out[key] = value.detach().clone() if torch.is_tensor(value) else value
        out = self._canonicalize_bank(out)
        out["valid_mask"] = self._valid_mask_from_bank(out).to(out["means"].device)
        if not bool(out["valid_mask"].any().item()):
            raise RuntimeError("Loss scoring bank has no valid rows.")
        return out

    # ------------------------------------------------------------------
    # Energy wrappers and replay diagnostics
    # ------------------------------------------------------------------
    def _geometry_energy_matrix(
        self,
        features: torch.Tensor,
        means: torch.Tensor,
        bases: torch.Tensor,
        variances: torch.Tensor,
        active_ranks: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        sample_counts: Optional[torch.Tensor] = None,
        return_parts: bool = False,
    ) -> torch.Tensor:
        classifier = getattr(self.model, "classifier", None)
        if classifier is not None and hasattr(classifier, "geometry_energy"):
            return classifier.geometry_energy(
                features=features,
                means=means,
                bases=bases,
                variances=variances,
                active_ranks=active_ranks,
                reliability=reliability,
                sample_counts=sample_counts,
                return_parts=return_parts,
            )
        if geometry_energy_matrix is None:
            raise RuntimeError("No geometry energy implementation available.")
        return geometry_energy_matrix(
            features=features,
            means=means,
            bases=bases,
            variances=variances,
            active_ranks=active_ranks,
            reliability=reliability,
            sample_counts=sample_counts,
            variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
            reliability_energy_weight=float(getattr(self.args, "reliability_energy_weight", 0.05)),
            residual_variance_scale=float(getattr(self.args, "residual_variance_scale", 1.0)),
            normalize_by_dim=bool(getattr(self.args, "energy_normalize_by_dim", True)),
            invalid_class_energy=float(getattr(self.args, "invalid_class_energy", 1e6)),
            use_logdet_energy=bool(getattr(self.args, "use_logdet_energy", True)),
            logdet_energy_weight=float(getattr(self.args, "logdet_energy_weight", getattr(self.args, "geometry_logdet_weight", 0.05))),
            logdet_normalize_by_dim=bool(getattr(self.args, "logdet_normalize_by_dim", True)),
            center_logdet_energy=bool(getattr(self.args, "center_logdet_energy", True)),
            return_parts=return_parts,
        )

    def _dual_geometry_energy_matrix(
        self,
        features: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        spectral_summary: Optional[torch.Tensor] = None,
        *,
        spectral_summary_is_physical: Optional[bool] = None,
        classifier_mode: Optional[str] = None,
        return_parts: bool = False,
    ) -> torch.Tensor:
        """Score features against the explicit bank passed by the caller.

        This is the critical helper fix.  The previous implementation tried to
        call model.compute_energy_from_features(), which silently used the live
        internal GeometryBank and ignored temporary/refined/temporary bank
        views.  That made admission and replay-health diagnostics
        report numbers for the wrong geometry.
        """
        bank = self._make_loss_scoring_bank(bank)
        if features is None or not torch.is_tensor(features) or features.numel() == 0:
            z = self._zero(features)
            return {"energy": z.view(0, 0)} if return_parts else z.view(0, 0)
        if features.dim() != 2:
            raise RuntimeError(f"features must be [B,D], got {tuple(features.shape)}")
        if int(features.size(1)) != int(bank["means"].size(1)):
            raise RuntimeError(
                f"feature/bank dimension mismatch: features D={int(features.size(1))}, "
                f"bank D={int(bank['means'].size(1))}"
            )

        mode = str(classifier_mode or getattr(self.args, "eval_classifier_mode", "geometry_only")).lower().strip()
        spectral_modes = {"srgp", "srgp_real", "real_spectral", "spectral_geometry"}
        spectral_active = (
            mode in spectral_modes
            and spectral_summary is not None
            and torch.is_tensor(spectral_summary)
            and spectral_summary.numel() > 0
            and bool(spectral_summary_is_physical)
        )

        if geometry_energy_matrix is not None:
            return geometry_energy_matrix(
                features=features,
                means=bank["means"],
                bases=bank["bases"],
                variances=bank["variances"],
                active_ranks=bank.get("active_ranks", None),
                reliability=bank.get("reliability", None),
                sample_counts=bank.get("sample_counts", None),
                variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
                reliability_energy_weight=float(getattr(self.args, "reliability_energy_weight", 0.05)),
                residual_variance_scale=float(getattr(self.args, "residual_variance_scale", 0.75)),
                normalize_by_dim=bool(getattr(self.args, "energy_normalize_by_dim", True)),
                invalid_class_energy=float(getattr(self.args, "invalid_class_energy", 1e6)),
                use_logdet_energy=bool(getattr(self.args, "use_logdet_energy", True)),
                logdet_energy_weight=float(getattr(self.args, "logdet_energy_weight", getattr(self.args, "geometry_logdet_weight", 0.05))),
                logdet_normalize_by_dim=bool(getattr(self.args, "logdet_normalize_by_dim", True)),
                center_logdet_energy=bool(getattr(self.args, "center_logdet_energy", True)),
                spectral_summary=spectral_summary if spectral_active else None,
                spectral_curve_means=bank.get("spectral_curve_means", None),
                spectral_curve_vars=bank.get("spectral_curve_vars", None),
                spectral_curve_d1=bank.get("spectral_curve_d1", None),
                spectral_curve_d2=bank.get("spectral_curve_d2", None),
                spectral_shape_reliability=bank.get("spectral_shape_reliability", None),
                use_spectral_residual_energy=bool(spectral_active),
                spectral_energy_weight=float(getattr(self.args, "spectral_energy_weight", 0.05)),
                spectral_summary_is_physical=bool(spectral_active),
                spectral_require_physical_summary=True,
                return_parts=return_parts,
            )

        return self._geometry_energy_matrix(
            features=features,
            means=bank["means"],
            bases=bank["bases"],
            variances=bank["variances"],
            active_ranks=bank.get("active_ranks", None),
            reliability=bank.get("reliability", None),
            sample_counts=bank.get("sample_counts", None),
            return_parts=return_parts,
        )

    def _active_basis(self, bases: torch.Tensor, active_ranks: Optional[torch.Tensor], cls: int) -> torch.Tensor:
        R = int(bases.size(2))
        if torch.is_tensor(active_ranks) and active_ranks.numel() > cls:
            r = max(0, min(int(active_ranks[cls].detach().item()), R))
        else:
            r = R
        return bases[int(cls), :, :r]

    @torch.no_grad()
    def _pgr_bank_reserve_metrics(self, bank: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        try:
            bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
        except Exception:
            return {}
        out: Dict[str, float] = {}
        try:
            out["pgr_feature_subspace_overlap"] = float(self._global_subspace_overlap_loss(bank).detach().cpu().item())
            out["pgr_feature_residual_var"] = float(self._bank_residual_variance_loss(bank).detach().cpu().item())
            out["pgr_feature_rank_usage"] = float(self._bank_active_rank_loss(bank).detach().cpu().item())
            out["pgr_reserve_score"] = float(1.0 / (1.0 + out["pgr_feature_subspace_overlap"] + 0.25 * out["pgr_feature_residual_var"] + 0.25 * out["pgr_feature_rank_usage"]))
        except Exception:
            pass
        return out

    def _global_subspace_overlap_loss(self, bank: Optional[Dict[str, torch.Tensor]] = None, *, basis_key: str = "bases", rank_key: str = "active_ranks") -> torch.Tensor:
        bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
        bases = bank.get(basis_key, None)
        if not torch.is_tensor(bases) or bases.numel() == 0:
            return self._zero()
        ranks = bank.get(rank_key, None)
        counts = bank.get("sample_counts", None)
        vals = []
        for i in range(bases.size(0)):
            if torch.is_tensor(counts) and counts.numel() > i and float(counts[i].item()) <= 0:
                continue
            Ui = self._active_basis(bases, ranks, i)
            if Ui.numel() == 0:
                continue
            for j in range(i + 1, bases.size(0)):
                if torch.is_tensor(counts) and counts.numel() > j and float(counts[j].item()) <= 0:
                    continue
                Uj = self._active_basis(bases, ranks, j)
                if Uj.numel() == 0:
                    continue
                vals.append((Ui.t() @ Uj).pow(2).mean())
        return torch.stack(vals).mean() if vals else bases.sum() * 0.0

    def _bank_residual_variance_loss(self, bank: Optional[Dict[str, torch.Tensor]] = None, *, variance_key: str = "variances") -> torch.Tensor:
        bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
        v = bank.get(variance_key, None)
        if not torch.is_tensor(v) or v.numel() == 0:
            return self._zero()
        res = v[:, -1]
        counts = bank.get("sample_counts", None)
        if torch.is_tensor(counts) and counts.numel() == res.numel():
            valid = counts.to(res.device) > 0
            if bool(valid.any().item()):
                res = res[valid]
        return torch.log1p(res.clamp_min(0.0)).mean()

    def _bank_active_rank_loss(self, bank: Optional[Dict[str, torch.Tensor]] = None, *, basis_key: str = "bases", rank_key: str = "active_ranks") -> torch.Tensor:
        bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
        ranks = bank.get(rank_key, None)
        bases = bank.get(basis_key, None)
        if not torch.is_tensor(ranks) or not torch.is_tensor(bases) or ranks.numel() == 0:
            return self._zero()
        values = ranks.float()
        counts = bank.get("sample_counts", None)
        if torch.is_tensor(counts) and counts.numel() == values.numel():
            valid = counts.to(values.device) > 0
            if bool(valid.any().item()):
                values = values[valid]
        return (values / float(max(bases.size(2), 1))).mean()

    # ------------------------------------------------------------------
    # Base/incremental geometry certificate helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _geometry_certificate_from_bank(
        self,
        *,
        phase: Optional[int] = None,
        class_ids: Optional[Iterable[int]] = None,
        val_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        """Compact certificate consumed by the base/incremental handoff.

        This does not change training.  It records whether the current bank is
        safe enough to serve as the frozen old geometry for future phases.
        """
        bank = self._safe_get_subspace_bank(require_ready=True)
        bank = self._canonicalize_bank(bank)
        valid = self._valid_mask_from_bank(bank)
        ids = self._as_class_list(class_ids) if class_ids is not None else [i for i in range(bank["means"].size(0)) if bool(valid[i].item())]
        ids = [int(c) for c in ids if 0 <= int(c) < bank["means"].size(0)]
        if not ids:
            return {"ok": False, "reason": "no valid class ids", "phase": int(getattr(self.model, "current_phase", 0) if phase is None else phase)}
        idx = torch.as_tensor(ids, device=bank["means"].device, dtype=torch.long)
        valid_sel = valid[idx]
        rel = bank["reliability"].detach().to(bank["means"].device)[idx]
        ranks = bank["active_ranks"].detach().to(bank["means"].device)[idx].float()
        counts = bank["sample_counts"].detach().to(bank["means"].device)[idx]
        resvars = bank["variances"].detach().to(bank["means"].device)[idx, -1]

        sub_max = sub_mean = band_max = band_mean = conflict_max = conflict_mean = 0.0
        gb = getattr(self.model, "geometry_bank", None)
        try:
            if gb is not None and hasattr(gb, "pairwise_subspace_overlap"):
                sub = gb.pairwise_subspace_overlap().detach()
                ss = sub.index_select(0, idx).index_select(1, idx)
                mask = ~torch.eye(ss.size(0), device=ss.device, dtype=torch.bool)
                vals = ss[mask] if ss.numel() > 1 else torch.empty(0, device=ss.device)
                if vals.numel() > 0:
                    sub_max = float(vals.max().cpu().item())
                    sub_mean = float(vals.mean().cpu().item())
            if gb is not None and hasattr(gb, "pairwise_band_similarity"):
                bs = gb.pairwise_band_similarity().detach().index_select(0, idx).index_select(1, idx)
                mask = ~torch.eye(bs.size(0), device=bs.device, dtype=torch.bool)
                vals = bs[mask] if bs.numel() > 1 else torch.empty(0, device=bs.device)
                if vals.numel() > 0:
                    band_max = float(vals.max().cpu().item())
                    band_mean = float(vals.mean().cpu().item())
            if gb is not None and hasattr(gb, "geometry_conflict_matrix"):
                cm = gb.geometry_conflict_matrix().detach().index_select(0, idx).index_select(1, idx)
                mask = ~torch.eye(cm.size(0), device=cm.device, dtype=torch.bool)
                vals = cm[mask] if cm.numel() > 1 else torch.empty(0, device=cm.device)
                vals = vals[vals > 0]
                if vals.numel() > 0:
                    conflict_max = float(vals.max().cpu().item())
                    conflict_mean = float(vals.mean().cpu().item())
        except Exception:
            pass

        geom_acc = float((val_stats or {}).get("acc", 0.0))
        min_rel = float(rel[valid_sel].min().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0
        mean_rel = float(rel[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0
        cert = {
            "phase": int(getattr(self.model, "current_phase", 0) if phase is None else phase),
            "class_ids": ids,
            "geom_acc": geom_acc,
            "valid_rows": int(valid_sel.sum().detach().cpu().item()),
            "expected_rows": int(len(ids)),
            "min_reliability": min_rel,
            "mean_reliability": mean_rel,
            "mean_active_rank": float(ranks[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0,
            "mean_sample_count": float(counts[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0,
            "mean_residual_var": float(resvars[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0,
            "max_subspace_overlap": sub_max,
            "mean_subspace_overlap": sub_mean,
            "max_band_similarity": band_max,
            "mean_band_similarity": band_mean,
            "max_geometry_conflict": conflict_max,
            "mean_geometry_conflict": conflict_mean,
        }
        cert["ok"] = bool(
            cert["valid_rows"] == cert["expected_rows"]
            and cert["min_reliability"] >= float(getattr(self.args, "base_cert_min_reliability", 0.15))
            and cert["mean_reliability"] >= float(getattr(self.args, "base_cert_min_mean_reliability", 0.35))
            and cert["max_subspace_overlap"] <= float(getattr(self.args, "base_cert_max_subspace_overlap", 0.65))
            and cert["max_geometry_conflict"] <= float(getattr(self.args, "base_cert_max_geometry_conflict", 2.0))
        )
        return cert

    # ------------------------------------------------------------------
    # Geometry diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _geometry_bank_diagnostics(self, class_ids: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, float]]:
        bank = self._safe_get_subspace_bank(require_ready=True)
        ids = self._as_class_list(class_ids) if class_ids is not None else list(range(bank["means"].size(0)))
        rows: Dict[int, Dict[str, float]] = {}
        for c in ids:
            if c < 0 or c >= bank["means"].size(0):
                continue
            rows[int(c)] = {
                "count": float(bank["sample_counts"][c].detach().item()),
                "active_rank": float(bank["active_ranks"][c].detach().item()),
                "reliability": float(bank["reliability"][c].detach().item()),
                "residual_var": float(bank["variances"][c, -1].detach().item()),
                "mean_norm": float(bank["means"][c].detach().norm().item()),
            }
        return rows

    @torch.no_grad()
    def _print_base_geometry_diagnostics(self, phase_class_ids: Iterable[int]) -> None:
        if not bool(getattr(self.args, "print_base_geometry_diagnostics", True)) and not bool(getattr(self, "debug", False)):
            return
        try:
            diag = self._geometry_bank_diagnostics(phase_class_ids)
        except Exception as exc:
            print(f"[Base Geometry Diagnostics] unavailable: {exc}")
            return
        print("[Base Geometry Diagnostics]")
        print("  cls | count | rank | rel   | resvar    | mean_norm")
        for cls in sorted(diag.keys()):
            d = diag[cls]
            print(f"  {cls:3d} | {d['count']:5.0f} | {d['active_rank']:4.0f} | {d['reliability']:5.3f} | {d['residual_var']:9.5f} | {d['mean_norm']:9.4f}")
        if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "geometry_diagnostics"):
            try:
                gd = self.model.geometry_bank.geometry_diagnostics()
                keys = ["feature_subspace_overlap", "feature_rank_usage", "band_overlap", "geometry_conflict_mean", "geometry_conflict_max", "geometry_reserve_score"]
                msg = []
                for k in keys:
                    v = gd.get(k, None)
                    if torch.is_tensor(v) and v.numel() == 1:
                        msg.append(f"{k}={float(v.item()):.4f}")
                if msg:
                    print("  " + " | ".join(msg))
            except Exception:
                pass

    @torch.no_grad()
    def _collect_bank_class_stats(self, phase_class_ids: Iterable[int], topk_bands: int = 5) -> List[Dict[str, object]]:
        bank = self._safe_get_subspace_bank(require_ready=True)
        ids = self._as_class_list(phase_class_ids)
        rows: List[Dict[str, object]] = []
        bands = bank.get("band_importances", None)
        for c in ids:
            if c < 0 or c >= bank["means"].size(0):
                continue
            eig = bank["variances"][c, :-1].detach().float().cpu()
            r = int(bank["active_ranks"][c].detach().item())
            eig_active = eig[:max(0, min(r, eig.numel()))]
            row: Dict[str, object] = {
                "class_id": int(c),
                "class_name": self._class_name(c),
                "sample_count": float(bank["sample_counts"][c].detach().item()),
                "valid_memory": bool(float(bank["sample_counts"][c].detach().item()) > 0.0),
                "feature_active_rank": int(r),
                "feature_rank_fraction": float(r / max(bank["bases"].size(2), 1)),
                "final_reliability": float(bank["reliability"][c].detach().item()),
                "feature_residual_var": float(bank["variances"][c, -1].detach().item()),
                "feature_mean_norm": float(bank["means"][c].detach().norm().item()),
                "spectral_shape_reliability": float(bank.get("spectral_shape_reliability", torch.zeros_like(bank["reliability"]))[c].detach().item()) if torch.is_tensor(bank.get("spectral_shape_reliability", None)) and bank.get("spectral_shape_reliability").numel() > c else 0.0,
                "feature_eig_min": float(eig_active.min().item()) if eig_active.numel() else 0.0,
                "feature_eig_max": float(eig_active.max().item()) if eig_active.numel() else 0.0,
                "feature_eig_ratio": float(eig_active.max().item() / max(eig_active.min().item(), 1e-12)) if eig_active.numel() else 0.0,
            }
            if torch.is_tensor(bands) and bands.numel() > 0 and c < bands.size(0) and bands.size(1) > 0:
                b = bands[c].detach().float().cpu().clamp_min(0.0)
                b = b / b.sum().clamp_min(1e-12)
                k = min(int(topk_bands), int(b.numel()))
                vals, idx = torch.topk(b, k=k)
                row.update({
                    "band_entropy": float((-(b * b.clamp_min(1e-12).log()).sum()).item()),
                    "band_max_weight": float(b.max().item()),
                    "band_top_indices": [int(i.item()) for i in idx],
                    "band_top_values": [float(v.item()) for v in vals],
                })
            else:
                row.update({"band_entropy": -1.0, "band_max_weight": -1.0, "band_top_indices": [], "band_top_values": []})
            rows.append(row)
        return rows

    @torch.no_grad()
    def _subspace_pair_risks(self, phase_class_ids: Iterable[int], top_k: int = 20) -> List[Dict[str, object]]:
        bank = self._safe_get_subspace_bank(require_ready=True)
        ids = self._as_class_list(phase_class_ids)
        means, bases, ranks = bank["means"], bank["bases"], bank.get("active_ranks", None)
        bands = bank.get("band_importances", None)
        rows: List[Dict[str, object]] = []
        for ii, ci in enumerate(ids):
            if ci >= means.size(0):
                continue
            Ui = self._active_basis(bases, ranks, ci)
            for cj in ids[ii + 1:]:
                if cj >= means.size(0):
                    continue
                Uj = self._active_basis(bases, ranks, cj)
                if Ui.numel() > 0 and Uj.numel() > 0:
                    f_ov = float((Ui.t() @ Uj).pow(2).sum().div(max(min(Ui.size(1), Uj.size(1)), 1)).detach().cpu().item())
                else:
                    f_ov = 0.0
                f_dist = float(torch.dist(means[ci], means[cj], p=2).detach().cpu().item())
                b_sim = 0.0
                if torch.is_tensor(bands) and bands.numel() > 0 and ci < bands.size(0) and cj < bands.size(0):
                    bi = F.normalize(bands[ci].clamp_min(0.0), dim=0)
                    bj = F.normalize(bands[cj].clamp_min(0.0), dim=0)
                    b_sim = float(torch.dot(bi, bj).clamp(0, 1).detach().cpu().item())
                risk = float(f_ov + 0.25 * b_sim + 1.0 / (1.0 + f_dist))
                rows.append({
                    "class_i": int(ci), "class_j": int(cj),
                    "name_i": self._class_name(ci), "name_j": self._class_name(cj),
                    "feature_overlap": f_ov, "raw_feature_overlap": f_ov,
                    "band_similarity": b_sim,
                    "feature_center_distance": f_dist,
                    "risk_score": risk,
                })
        rows.sort(key=lambda r: float(r["risk_score"]), reverse=True)
        return rows[:int(top_k)]

    @torch.no_grad()
    def _old_new_pair_risks(
        self,
        old_class_ids: Iterable[int],
        new_class_ids: Iterable[int],
        top_k: int = 20,
    ) -> List[Dict[str, object]]:
        """Rank old/new SRGP descriptor conflicts for incremental diagnostics."""
        bank = self._safe_get_subspace_bank(require_ready=True)
        bank = self._canonicalize_bank(bank)
        old_ids = self._as_class_list(old_class_ids)
        new_ids = self._as_class_list(new_class_ids)
        if not old_ids or not new_ids:
            return []

        means, bases, ranks = bank["means"], bank["bases"], bank.get("active_ranks", None)
        counts = bank.get("sample_counts", None)
        rel = bank.get("reliability", None)
        bands = bank.get("band_importances", bank.get("band_importance", None))
        d1 = bank.get("spectral_curve_d1", None)
        rows: List[Dict[str, object]] = []
        dscale = max(float(means.size(1)) ** 0.5, 1.0)
        center_w = self._overlap_cfg_float("old_new_risk_center_weight", 0.40)
        subspace_w = self._overlap_cfg_float("old_new_risk_subspace_weight", 0.35)
        band_w = self._overlap_cfg_float("old_new_risk_band_weight", 0.10)
        spec_w = self._overlap_cfg_float("old_new_risk_spectral_shape_weight", 0.15)

        for oi in old_ids:
            if oi < 0 or oi >= means.size(0):
                continue
            if torch.is_tensor(counts) and counts.numel() > oi and float(counts[oi].detach().item()) <= 0.0:
                continue
            Ui = self._active_basis(bases, ranks, oi)
            for nj in new_ids:
                if nj < 0 or nj >= means.size(0):
                    continue
                if torch.is_tensor(counts) and counts.numel() > nj and float(counts[nj].detach().item()) <= 0.0:
                    continue
                Uj = self._active_basis(bases, ranks, nj)
                dist = torch.norm(means[oi] - means[nj], p=2)
                scaled = dist / dscale
                center_risk = torch.exp(-scaled).clamp(0.0, 1.0)
                if Ui.numel() > 0 and Uj.numel() > 0:
                    denom = float(max(min(Ui.size(1), Uj.size(1)), 1))
                    f_ov_t = (Ui.t() @ Uj).pow(2).sum() / denom
                    f_ov_t = f_ov_t.clamp(0.0, 1.0)
                else:
                    f_ov_t = torch.tensor(0.0, device=means.device, dtype=means.dtype)

                b_sim_t = torch.tensor(0.0, device=means.device, dtype=means.dtype)
                if torch.is_tensor(bands) and bands.dim() == 2 and bands.numel() > 0 and oi < bands.size(0) and nj < bands.size(0):
                    bi = bands[oi].clamp_min(0.0)
                    bj = bands[nj].clamp_min(0.0)
                    if bi.numel() == bj.numel() and bi.sum() > 1e-8 and bj.sum() > 1e-8:
                        bi = bi / bi.norm().clamp_min(1e-8)
                        bj = bj / bj.norm().clamp_min(1e-8)
                        b_sim_t = torch.dot(bi, bj).clamp(0.0, 1.0)

                spec_sim_t = torch.tensor(0.0, device=means.device, dtype=means.dtype)
                if torch.is_tensor(d1) and d1.dim() == 2 and d1.size(0) > max(oi, nj) and d1.size(1) > 0:
                    oi_d = F.normalize(torch.nan_to_num(d1[oi], nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()
                    nj_d = F.normalize(torch.nan_to_num(d1[nj], nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()
                    spec_sim_t = torch.dot(oi_d, nj_d).clamp(0.0, 1.0)

                risk_t = center_w * center_risk + subspace_w * f_ov_t + band_w * b_sim_t + spec_w * spec_sim_t
                if torch.is_tensor(rel) and rel.numel() > oi:
                    uncertainty = (1.0 - rel[oi].clamp(0.05, 1.0)).clamp(0.0, 1.0)
                    risk_t = risk_t * (1.0 + 0.25 * uncertainty)
                risk = float(risk_t.detach().cpu().item())
                rows.append(
                    {
                        "old_class": int(oi),
                        "new_class": int(nj),
                        "old_name": self._class_name(oi),
                        "new_name": self._class_name(nj),
                        "feature_overlap": float(f_ov_t.detach().cpu().item()),
                        "band_similarity": float(b_sim_t.detach().cpu().item()),
                        "spectral_shape_similarity": float(spec_sim_t.detach().cpu().item()),
                        "feature_center_distance": float(dist.detach().cpu().item()),
                        "scaled_center_distance": float(scaled.detach().cpu().item()),
                        "center_risk": float(center_risk.detach().cpu().item()),
                        "risk_score": risk,
                    }
                )
        rows.sort(key=lambda r: float(r["risk_score"]), reverse=True)
        return rows[: int(top_k)]

    @torch.no_grad()
    def _phase_old_new_pair_risks(self, phase: Optional[int] = None, top_k: int = 20) -> List[Dict[str, object]]:
        phase = int(getattr(self.model, "current_phase", 0) if phase is None else phase)
        if phase <= 0 or not hasattr(self.dataset, "phase_to_classes"):
            return []
        old_ids = self._seen_class_ids_before_phase(phase)
        new_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
        return self._old_new_pair_risks(old_ids, new_ids, top_k=top_k)

    @torch.no_grad()
    def _phase_overlap_summary(self, phase: Optional[int] = None, top_k: int = 20) -> Dict[str, object]:
        pairs = self._phase_old_new_pair_risks(phase=phase, top_k=top_k)
        if not pairs:
            return {"num_pairs": 0, "max_risk": 0.0, "mean_risk": 0.0, "top_pair": None}
        risks = [float(p["risk_score"]) for p in pairs]
        return {
            "num_pairs": int(len(pairs)),
            "max_risk": float(max(risks)),
            "mean_risk": float(sum(risks) / max(len(risks), 1)),
            "top_pair": pairs[0],
        }

    @torch.no_grad()
    def _energy_margin_health(self, loader, phase_class_ids: Iterable[int]) -> Dict[str, object]:
        bank = self._safe_get_subspace_bank(require_ready=True)
        ids = self._as_class_list(phase_class_ids)
        id_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
        stats = {c: {"n": 0, "correct": 0, "viol": 0, "margin_sum": 0.0, "margin_min": float("inf")} for c in ids}
        was_training = bool(self.model.training)
        self.model.eval()
        for batch in loader:
            x, y, spectra, _ = self._unpack_hsi_batch(batch)
            x = x.to(self.device, non_blocking=True).float()
            y = y.to(self.device, non_blocking=True).long().view(-1)
            spectral_summary, spec_is_physical = self._resolve_batch_spectral_summary(
                x, spectra=spectra, source="batch_metadata" if torch.is_tensor(spectra) and spectra.numel() > 0 else "input"
            )
            try:
                out = self.model.extract_projected_features(
                    x,
                    spectral_summary=spectral_summary,
                    spectral_summary_is_physical=bool(spec_is_physical),
                )
            except TypeError:
                out = self.model.extract_projected_features(x)
            features = out["features"]
            ss, ss_phys = self._resolve_batch_spectral_summary(
                x, spectra=spectra, model_out=out, source="batch_metadata" if torch.is_tensor(spectra) and spectra.numel() > 0 else "input", spectral_summary_is_physical=spec_is_physical
            )
            energy = self._dual_geometry_energy_matrix(
                features=features,
                bank=bank,
                spectral_summary=ss,
                spectral_summary_is_physical=bool(ss_phys),
                return_parts=False,
            )
            e_sel = energy.index_select(1, id_tensor)
            pred = id_tensor[e_sel.argmin(dim=1)]
            y_local = torch.full_like(y, -1)
            for li, c in enumerate(ids):
                y_local[y == int(c)] = int(li)
            valid = y_local >= 0
            if not bool(valid.any().item()):
                continue
            e_valid, yv, ylv, predv = e_sel[valid], y[valid], y_local[valid], pred[valid]
            true_e = e_valid.gather(1, ylv.view(-1, 1)).squeeze(1)
            mask = torch.zeros_like(e_valid, dtype=torch.bool).scatter(1, ylv.view(-1, 1), True)
            nearest_wrong = e_valid.masked_fill(mask, float("inf")).min(dim=1).values
            margin = nearest_wrong - true_e
            for c in ids:
                m = yv == int(c)
                if not bool(m.any().item()):
                    continue
                s = stats[int(c)]
                mg = margin[m]
                s["n"] += int(m.sum().item())
                s["correct"] += int((predv[m] == yv[m]).sum().item())
                s["viol"] += int((mg <= 0).sum().item())
                s["margin_sum"] += float(mg.sum().detach().cpu().item())
                s["margin_min"] = min(s["margin_min"], float(mg.min().detach().cpu().item()))
        self.model.train(was_training)
        rows = []
        total_n = total_correct = total_viol = 0
        total_margin = 0.0
        min_margin = float("inf")
        for c in ids:
            s = stats[int(c)]
            n = max(int(s["n"]), 1)
            rows.append({
                "class_id": int(c), "class_name": self._class_name(c), "n": int(s["n"]),
                "accuracy": 100.0 * float(s["correct"]) / n,
                "mean_margin": float(s["margin_sum"]) / n,
                "min_margin": 0.0 if s["margin_min"] == float("inf") else float(s["margin_min"]),
                "violation_rate": 100.0 * float(s["viol"]) / n,
            })
            total_n += int(s["n"]); total_correct += int(s["correct"]); total_viol += int(s["viol"]); total_margin += float(s["margin_sum"])
            if s["margin_min"] != float("inf"):
                min_margin = min(min_margin, float(s["margin_min"]))
        denom = max(total_n, 1)
        return {"overall": {"n": total_n, "accuracy": 100.0 * total_correct / denom, "mean_margin": total_margin / denom, "min_margin": 0.0 if min_margin == float("inf") else min_margin, "violation_rate": 100.0 * total_viol / denom}, "per_class": rows}

    @torch.no_grad()
    def _sample_bank_anchors_for_diagnostics(self, class_ids: Iterable[int], samples_per_class: int = 64, parallel_scale: float = 1.0, residual_scale: float = 0.30) -> Tuple[torch.Tensor, torch.Tensor]:
        bank = self._safe_get_subspace_bank(require_ready=True)
        ids = self._as_class_list(class_ids)
        gb = getattr(self.model, "geometry_bank", None)
        if gb is not None and hasattr(gb, "sample_synthetic_features"):
            try:
                return gb.sample_synthetic_features(
                    ids,
                    samples_per_class=int(samples_per_class),
                    parallel_scale=float(parallel_scale),
                    residual_scale=float(residual_scale),
                    reliability_gated=bool(getattr(self.args, "diagnostic_reliability_gated_anchors", True)),
                )
            except TypeError:
                return gb.sample_synthetic_features(ids, samples_per_class=int(samples_per_class), parallel_scale=float(parallel_scale), residual_scale=float(residual_scale))
        means, bases, variances = bank["means"], bank["bases"], bank["variances"]
        xs, ys = [], []
        for c in ids:
            if c < 0 or c >= means.size(0) or float(bank["sample_counts"][c].item()) <= 0:
                continue
            n = int(max(samples_per_class, 1)); r = int(bank["active_ranks"][c].item())
            eps = torch.zeros((n, means.size(1)), device=self.device, dtype=means.dtype)
            if r > 0:
                z = torch.randn((n, r), device=self.device, dtype=means.dtype)
                eps += (z * variances[c, :r].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4))).sqrt()) @ bases[c, :, :r].t()
            eps += torch.randn_like(eps) * variances[c, -1].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4))).sqrt() * float(residual_scale)
            xs.append(means[c].unsqueeze(0) + eps)
            ys.append(torch.full((n,), int(c), device=self.device, dtype=torch.long))
        if not xs:
            return torch.empty((0, int(getattr(self.args, "d_model", 0))), device=self.device), torch.empty((0,), device=self.device, dtype=torch.long)
        return torch.cat(xs, 0), torch.cat(ys, 0)

    @torch.no_grad()
    def _anchor_replay_health(self, phase_class_ids: Iterable[int], samples_per_class: int = 64) -> Dict[str, object]:
        ids = self._as_class_list(phase_class_ids)
        x, y = self._sample_bank_anchors_for_diagnostics(ids, samples_per_class=samples_per_class, parallel_scale=float(getattr(self.args, "gfa_parallel_scale", 1.0)), residual_scale=float(getattr(self.args, "gfa_residual_scale", 0.25)))
        if x.numel() == 0:
            return {"overall": {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "min_margin": 0.0, "violation_rate": 100.0}, "per_class": []}
        bank = self._safe_get_subspace_bank(require_ready=True)
        id_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
        energy = self._dual_geometry_energy_matrix(x, bank, return_parts=False)
        e_sel = energy.index_select(1, id_tensor)
        y_local = torch.full_like(y, -1)
        for li, c in enumerate(ids):
            y_local[y == int(c)] = int(li)
        pred = id_tensor[e_sel.argmin(dim=1)]
        true_e = e_sel.gather(1, y_local.view(-1, 1)).squeeze(1)
        label_mask = torch.zeros_like(e_sel, dtype=torch.bool).scatter(1, y_local.view(-1, 1), True)
        nearest_wrong = e_sel.masked_fill(label_mask, float("inf")).min(dim=1).values
        margin = nearest_wrong - true_e
        rows = []
        for c in ids:
            m = y == int(c)
            if not bool(m.any().item()):
                continue
            n = int(m.sum().item())
            rows.append({
                "class_id": int(c), "class_name": self._class_name(c), "n": n,
                "anchor_accuracy": 100.0 * float((pred[m] == y[m]).sum().item()) / max(n, 1),
                "anchor_mean_margin": float(margin[m].mean().item()),
                "anchor_min_margin": float(margin[m].min().item()),
                "anchor_violation_rate": 100.0 * float((margin[m] <= 0).sum().item()) / max(n, 1),
            })
        total_n = int(y.numel())
        return {"overall": {"n": total_n, "accuracy": 100.0 * int((pred == y).sum().item()) / max(total_n, 1), "mean_margin": float(margin.mean().item()), "min_margin": float(margin.min().item()), "violation_rate": 100.0 * int((margin <= 0).sum().item()) / max(total_n, 1)}, "per_class": rows}

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()
    def diagnose_full_base_geometry(self, loader, phase_class_ids: Iterable[int], anchors_per_class: int = 64, topk_pairs: int = 20, topk_bands: int = 5) -> Dict[str, object]:
        """Geometry health report for the PG-RGA path.

        Kept diagnostics:
            - class GeometryBank rows;
            - geometry-energy margin health;
            - subspace/center/band old-new risk;
            - synthetic GeometryBank replay health.

        Removed diagnostics:
            - SCB-GR boundary anchors;
            - SGLAT/transport diagnostics;
            - MSSL/manifold losses.
        """
        ids = self._as_class_list(phase_class_ids)
        bank_internal = None
        if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "geometry_health_summary"):
            try:
                names = [self._class_name(i) for i in range(max(ids) + 1)] if ids else []
                bank_internal = self.model.geometry_bank.geometry_health_summary(class_names=names, topk_bands=topk_bands)
            except Exception as exc:
                bank_internal = {"error": str(exc)}
        phase = int(getattr(self.model, "current_phase", 0))
        old_new_pairs = self._phase_old_new_pair_risks(phase=phase, top_k=topk_pairs)
        report = {
            "class_ids": ids,
            "phase": phase,
            "bank_internal": bank_internal,
            "class_geometry": self._collect_bank_class_stats(ids, topk_bands=topk_bands),
            "energy_margin": self._energy_margin_health(loader, ids),
            "subspace_risk_pairs": self._subspace_pair_risks(ids, top_k=topk_pairs),
            "old_new_risk_pairs": old_new_pairs,
            "old_new_overlap_summary": self._phase_overlap_summary(phase=phase, top_k=topk_pairs),
            "overlap_admission_events": list(getattr(self, "_last_overlap_admission_events", [])),
            "anchor_replay": self._anchor_replay_health(ids, samples_per_class=anchors_per_class),
        }
        alerts = []
        em = report["energy_margin"]["overall"]
        ar = report["anchor_replay"]["overall"]
        if float(em["violation_rate"]) > 5.0:
            alerts.append(f"High validation energy violation rate: {em['violation_rate']:.2f}%")
        if float(ar["accuracy"]) < 95.0:
            alerts.append(f"Low GeometryBank replay self-accuracy: {ar['accuracy']:.2f}%")
        pairs = report["subspace_risk_pairs"]
        if pairs and float(pairs[0]["feature_overlap"]) > 0.70:
            alerts.append(f"High feature subspace overlap: {pairs[0]['name_i']} vs {pairs[0]['name_j']} = {pairs[0]['feature_overlap']:.4f}")
        old_new_risk_alert = float(getattr(self.args, "old_new_overlap_risk_alert", 0.85))
        old_new_subspace_alert = float(getattr(self.args, "old_new_subspace_overlap_alert", 0.60))
        if old_new_pairs:
            top = old_new_pairs[0]
            if float(top.get("risk_score", 0.0)) >= old_new_risk_alert:
                alerts.append(
                    f"High old/new geometry conflict: old {top['old_name']} vs new {top['new_name']} "
                    f"risk={float(top['risk_score']):.4f}"
                )
            if float(top.get("feature_overlap", 0.0)) >= old_new_subspace_alert:
                alerts.append(
                    f"High old/new subspace overlap: old {top['old_name']} vs new {top['new_name']} "
                    f"overlap={float(top.get('feature_overlap', 0.0)):.4f}"
                )
        report["alerts"] = alerts
        return self._json_safe(report)

    def _write_csv_rows(self, path: str, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(row.get(k, "")) if isinstance(row.get(k, ""), (list, dict)) else row.get(k, "") for k in keys})

    def _save_geometry_diagnostics_to_files(self, report: Dict[str, object], phase: int = 0, output_dir: Optional[str] = None) -> Dict[str, str]:
        phase = int(phase)
        if output_dir is None:
            root = getattr(self.args, "run_dir", getattr(self, "save_dir", "."))
            output_dir = os.path.join(root, f"phase_{phase}")
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            "json": os.path.join(output_dir, f"phase_{phase}_geometry_diagnostics.json"),
            "class_csv": os.path.join(output_dir, f"phase_{phase}_geometry_class_stats.csv"),
            "energy_csv": os.path.join(output_dir, f"phase_{phase}_geometry_energy_margins.csv"),
            "subspace_csv": os.path.join(output_dir, f"phase_{phase}_geometry_subspace_pairs.csv"),
            "old_new_csv": os.path.join(output_dir, f"phase_{phase}_geometry_old_new_pairs.csv"),
            "anchor_csv": os.path.join(output_dir, f"phase_{phase}_geometry_anchor_stats.csv"),
            "txt": os.path.join(output_dir, f"phase_{phase}_geometry_diagnostics.txt"),
        }
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(self._json_safe(report), f, indent=2)
        self._write_csv_rows(paths["class_csv"], report.get("class_geometry", []))
        self._write_csv_rows(paths["energy_csv"], report.get("energy_margin", {}).get("per_class", []))
        self._write_csv_rows(paths["subspace_csv"], report.get("subspace_risk_pairs", []))
        self._write_csv_rows(paths["old_new_csv"], report.get("old_new_risk_pairs", []))
        self._write_csv_rows(paths["anchor_csv"], report.get("anchor_replay", {}).get("per_class", []))
        with open(paths["txt"], "w", encoding="utf-8") as f:
            f.write(self._format_geometry_diagnostics_text(report))
        return paths

    def _format_geometry_diagnostics_text(self, report: Dict[str, object]) -> str:
        lines = ["Geometry Diagnostics", "=" * 90, "", "Alerts", "-" * 90]
        alerts = report.get("alerts", [])
        if alerts:
            lines.extend(f"[WARN] {a}" for a in alerts)
        else:
            lines.append("No major geometry alarms triggered by current thresholds.")

        lines += ["", "Class Geometry", "-" * 90, "cls name                 n     rank rel    resvar    band-H  band-max"]
        for r in report.get("class_geometry", []):
            lines.append(
                f"{int(r['class_id']):3d} {str(r['class_name'])[:20]:20s} "
                f"{float(r['sample_count']):5.0f} {int(r['feature_active_rank']):5d} "
                f"{float(r['final_reliability']):5.3f} {float(r['feature_residual_var']):9.5f} "
                f"{float(r['band_entropy']):7.3f} {float(r['band_max_weight']):8.4f}"
            )

        ov = report.get("energy_margin", {}).get("overall", {})
        lines += ["", "Energy Margin Health", "-" * 90]
        lines.append(
            f"Overall: acc={float(ov.get('accuracy', 0.0)):.2f}% | "
            f"mean_margin={float(ov.get('mean_margin', 0.0)):.6f} | "
            f"min_margin={float(ov.get('min_margin', 0.0)):.6f} | "
            f"viol={float(ov.get('violation_rate', 0.0)):.2f}%"
        )
        lines.append("cls name                 n     acc     mean_margin   min_margin    viol")
        for r in report.get("energy_margin", {}).get("per_class", []):
            lines.append(
                f"{int(r['class_id']):3d} {str(r['class_name'])[:20]:20s} "
                f"{int(r['n']):5d} {float(r['accuracy']):7.2f} "
                f"{float(r['mean_margin']):13.6f} {float(r['min_margin']):12.6f} "
                f"{float(r['violation_rate']):7.2f}%"
            )

        lines += ["", "Top Geometry Risk Pairs", "-" * 90, "pair                                      z-overlap band-sim z-dist    risk"]
        for r in report.get("subspace_risk_pairs", [])[:15]:
            pair = f"{r['name_i']} / {r['name_j']}"
            lines.append(
                f"{pair[:40]:40s} {float(r['feature_overlap']):9.4f} "
                f"{float(r.get('band_similarity', 0.0)):8.4f} "
                f"{float(r['feature_center_distance']):8.4f} {float(r['risk_score']):8.4f}"
            )

        old_new = report.get("old_new_risk_pairs", [])
        if old_new:
            lines += ["", "Top Old/New Geometry Conflicts", "-" * 90, "old -> new                                z-overlap band-sim spec-sim z-dist    risk"]
            for r in old_new[:15]:
                pair = f"{r['old_name']} -> {r['new_name']}"
                lines.append(
                    f"{pair[:40]:40s} {float(r.get('feature_overlap', 0.0)):9.4f} "
                    f"{float(r.get('band_similarity', 0.0)):8.4f} "
                    f"{float(r.get('spectral_shape_similarity', 0.0)):8.4f} "
                    f"{float(r.get('feature_center_distance', 0.0)):8.4f} "
                    f"{float(r.get('risk_score', 0.0)):8.4f}"
                )

        events = report.get("overlap_admission_events", [])
        if events:
            lines += ["", "Old/New Overlap Diagnostic Events", "-" * 90, "new class                                risk     gate    top old"]
            for e in events[-15:]:
                top = e.get("top_old", {}) if isinstance(e.get("top_old", {}), dict) else {}
                lines.append(
                    f"{str(e.get('new_name', e.get('new_class', 'NA')))[:40]:40s} "
                    f"{float(e.get('max_old_overlap_risk', 0.0)):8.4f} "
                    f"{float(e.get('admission_gate', 1.0)):7.3f} "
                    f"{str(top.get('old_name', top.get('old_class', 'NA')))[:20]:20s}"
                )

        av = report.get("anchor_replay", {}).get("overall", {})
        lines += ["", "GeometryBank Replay Health", "-" * 90]
        lines.append(
            f"Overall: acc={float(av.get('accuracy', 0.0)):.2f}% | "
            f"mean_margin={float(av.get('mean_margin', 0.0)):.6f} | "
            f"min_margin={float(av.get('min_margin', 0.0)):.6f} | "
            f"viol={float(av.get('violation_rate', 0.0)):.2f}%"
        )
        return "\n".join(lines) + "\n"

    def _print_geometry_diagnostics_summary(self, report: Dict[str, object]) -> None:
        em = report.get("energy_margin", {}).get("overall", {})
        ar = report.get("anchor_replay", {}).get("overall", {})
        old_new_summary = report.get("old_new_overlap_summary", {})
        print(
            "[Geometry Health] "
            f"energy_acc={float(em.get('accuracy', 0.0)):.2f}% | "
            f"energy_viol={float(em.get('violation_rate', 0.0)):.2f}% | "
            f"replay_acc={float(ar.get('accuracy', 0.0)):.2f}% | "
            f"replay_viol={float(ar.get('violation_rate', 0.0)):.2f}% | "
            f"old_new_max_risk={float(old_new_summary.get('max_risk', 0.0)):.4f}"
        )
        top_pair = old_new_summary.get("top_pair", None)
        if isinstance(top_pair, dict):
            print(
                "[Geometry Health] Top old/new conflict: "
                f"old {top_pair.get('old_name', top_pair.get('old_class', 'NA'))} -> "
                f"new {top_pair.get('new_name', top_pair.get('new_class', 'NA'))} | "
                f"risk={float(top_pair.get('risk_score', 0.0)):.4f} | "
                f"overlap={float(top_pair.get('feature_overlap', 0.0)):.4f} | "
                f"spec_sim={float(top_pair.get('spectral_shape_similarity', 0.0)):.4f}"
            )
        for alert in report.get("alerts", [])[:10]:
            print(f"[Geometry Health WARN] {alert}")


    # ------------------------------------------------------------------
    # Structured diagnostics / JSON saving
    # ------------------------------------------------------------------
    def save_json_diagnostics(self, path: str, data: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(data), f, indent=2)

    def compute_old_new_overlap_diagnostics(
        self,
        old_classes: Iterable[int],
        new_classes: Iterable[int],
        *,
        top_k: int = 20,
    ) -> Dict[str, object]:
        pairs = self._old_new_pair_risks(old_classes, new_classes, top_k=top_k)
        risks = [float(p.get("risk_score", 0.0)) for p in pairs]
        return {
            "old_classes": self._as_class_list(old_classes),
            "new_classes": self._as_class_list(new_classes),
            "num_pairs": len(pairs),
            "max_risk": max(risks) if risks else 0.0,
            "mean_risk": sum(risks) / max(len(risks), 1) if risks else 0.0,
            "top_pair": pairs[0] if pairs else None,
            "unsafe_pairs": [p for p in pairs if float(p.get("risk_score", 0.0)) >= float(getattr(self.args, "old_new_overlap_risk_alert", 0.85))],
            "pairs": pairs,
        }

    # ------------------------------------------------------------------
    # SGLAT shared diagnostics
    # ------------------------------------------------------------------












# from __future__ import annotations

# from contextlib import nullcontext
# import csv
# import json
# import os
# from typing import Dict, Iterable, List, Optional, Tuple

# import torch
# import torch.nn.functional as F

# try:
#     from losses.loss import geometry_energy_matrix
# except Exception:  # pragma: no cover - compile/runtime compatibility for renamed packages
#     geometry_energy_matrix = None

# try:
#     from losses.loss import sample_boundary_geometry_features
# except Exception:  # pragma: no cover - optional SCB-GR diagnostic helper
#     sample_boundary_geometry_features = None


# class TrainerHelper:
#     # ------------------------------------------------------------------
#     # Generic utilities
#     # ------------------------------------------------------------------
#     def _zero(self, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
#         if torch.is_tensor(ref):
#             return ref.sum() * 0.0
#         return torch.tensor(0.0, device=self.device, dtype=torch.float32)

#     def _as_class_list(self, class_ids: Iterable[int]) -> List[int]:
#         return [int(c) for c in class_ids]

#     def _detach_clone(self, x: torch.Tensor) -> torch.Tensor:
#         return x.detach().clone()

#     def _json_safe(self, obj):
#         if torch.is_tensor(obj):
#             obj = obj.detach().cpu()
#             if obj.numel() == 1:
#                 return obj.item()
#             return obj.tolist()
#         if isinstance(obj, dict):
#             return {str(k): self._json_safe(v) for k, v in obj.items()}
#         if isinstance(obj, (list, tuple)):
#             return [self._json_safe(v) for v in obj]
#         try:
#             import numpy as _np
#             if isinstance(obj, (_np.integer, _np.floating)):
#                 return obj.item()
#             if isinstance(obj, _np.ndarray):
#                 return obj.tolist()
#         except Exception:
#             pass
#         if isinstance(obj, (str, int, float, bool)) or obj is None:
#             return obj
#         return str(obj)

#     def _class_name(self, cls: int) -> str:
#         cls = int(cls)
#         names = getattr(self.dataset, "target_names", None)
#         if names is not None:
#             try:
#                 return str(names[cls])
#             except Exception:
#                 pass
#         names = getattr(self.args, "target_names", None)
#         if names is not None:
#             try:
#                 return str(names[cls])
#             except Exception:
#                 pass
#         return f"Class-{cls}"

#     def _stable_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         if logits is None or not torch.is_tensor(logits) or logits.numel() == 0:
#             return self._zero(logits)
#         labels = labels.long().view(-1).to(logits.device)
#         if labels.numel() == 0:
#             return self._zero(logits)
#         if labels.numel() != logits.size(0):
#             raise RuntimeError(f"CE batch mismatch: logits={logits.size(0)}, labels={labels.numel()}")
#         if int(labels.min().item()) < 0 or int(labels.max().item()) >= logits.size(1):
#             raise RuntimeError(
#                 f"CE label range [{int(labels.min())},{int(labels.max())}] incompatible with logits width={logits.size(1)}"
#             )
#         clip = float(getattr(self, "ce_logit_clip", getattr(self.args, "ce_logit_clip", 50.0)))
#         smoothing = float(getattr(self, "label_smoothing", getattr(self.args, "label_smoothing", 0.0)))
#         return F.cross_entropy(logits.clamp(-clip, clip), labels, label_smoothing=smoothing)

#     def _unpack_hsi_batch(self, batch):
#         """
#         Accept both legacy (patches, labels) batches and metadata batches
#         (patches, labels, center_spectrum, coord). Metadata spectra are part
#         of the SRGP/SCB-GR contract: reduced/PCA patches go to the backbone,
#         raw physical center spectra go to GeometryBank spectral-shape scoring.
#         """
#         if isinstance(batch, dict):
#             x = batch.get("image", batch.get("patch", batch.get("patches", None)))
#             y = batch.get("label", batch.get("labels", None))
#             spectra = batch.get("spectrum", batch.get("spectra", None))
#             coords = batch.get("coord", batch.get("coords", None))
#             if x is None or y is None:
#                 raise RuntimeError("Batch dict must contain image/patches and label/labels.")
#             return x, y, spectra, coords
#         if isinstance(batch, (tuple, list)):
#             if len(batch) < 2:
#                 raise RuntimeError(f"Batch tuple/list must have at least 2 fields, got {len(batch)}")
#             x, y = batch[0], batch[1]
#             spectra = batch[2] if len(batch) >= 3 else None
#             coords = batch[3] if len(batch) >= 4 else None
#             return x, y, spectra, coords
#         raise RuntimeError(f"Unsupported batch type: {type(batch)}")

#     @staticmethod
#     def _center_spectrum_from_tensor(x: torch.Tensor) -> torch.Tensor:
#         """Return a center-pixel spectral vector from an HSI tensor.

#         For center-pixel HSI classification, the label belongs to the center
#         pixel, not the whole patch. Patch-mean spectra can mix neighboring
#         classes and poison SRGP spectral-shape descriptors.
#         """
#         if not torch.is_tensor(x):
#             raise TypeError(f"x must be a tensor, got {type(x)}")
#         if x.dim() == 4:          # [B, S, H, W]
#             return x[:, :, x.size(-2) // 2, x.size(-1) // 2]
#         if x.dim() == 3:          # [B, S, L] or equivalent spectral sequence
#             return x[:, :, x.size(-1) // 2]
#         if x.dim() == 2:          # [B, S]
#             return x
#         return x.flatten(1)

#     def _normalize_spectral_metadata_tensor(
#         self,
#         spectra: torch.Tensor,
#         *,
#         ref_x: Optional[torch.Tensor] = None,
#         expected_n: Optional[int] = None,
#     ) -> torch.Tensor:
#         """Normalize metadata spectra to [B,S] without flattening patches.

#         This is a critical SRGP safety helper. If raw spectra arrive as
#         [B,S,H,W], the label belongs to the center pixel, so we take only the
#         center spectrum. Flattening the whole patch would create [B,S*H*W] and
#         poison spectral-shape descriptors.
#         """
#         if not torch.is_tensor(spectra):
#             spectra = torch.as_tensor(spectra)
#         if ref_x is not None and torch.is_tensor(ref_x):
#             s = spectra.to(device=ref_x.device, dtype=ref_x.dtype, non_blocking=True)
#             n = int(ref_x.size(0))
#         else:
#             s = spectra.float()
#             n = int(expected_n or (s.size(0) if s.dim() > 0 else 1))

#         if s.numel() == 0:
#             return s.reshape(n, 0)
#         if s.dim() == 4:          # [B,S,H,W]
#             s = s[:, :, s.size(-2) // 2, s.size(-1) // 2]
#         elif s.dim() == 3:        # [B,S,L] or [B,H,W]-like metadata
#             if s.size(0) != n:
#                 s = s.reshape(n, -1)
#             elif s.size(1) > 0 and s.size(2) > 1:
#                 s = s[:, :, s.size(-1) // 2]
#             else:
#                 s = s.reshape(n, -1)
#         elif s.dim() == 1:
#             if n > 1 and s.numel() % n == 0:
#                 s = s.reshape(n, -1)
#             else:
#                 s = s.reshape(1, -1)
#         elif s.dim() != 2:
#             s = s.reshape(n, -1)
#         if s.size(0) != n:
#             raise RuntimeError(f"spectral metadata batch mismatch: spectra={tuple(s.shape)}, expected_n={n}")
#         return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

#     def _dataset_spectra_are_physical(self) -> Optional[bool]:
#         """Return dataset-declared physical-spectra status when available."""
#         for attr in ("spectra_are_physical", "raw_spectra_are_physical", "center_spectra_are_physical"):
#             if hasattr(self.dataset, attr):
#                 value = getattr(self.dataset, attr)
#                 if isinstance(value, bool):
#                     return value
#                 if isinstance(value, (int, float)):
#                     return bool(value)
#         fn = getattr(self.dataset, "has_physical_spectra", None)
#         if callable(fn):
#             try:
#                 return bool(fn())
#             except Exception:
#                 return None
#         return None

#     def _cfg_bool(self, name: str, default: bool = False) -> bool:
#         value = getattr(self, name, getattr(self.args, name, default))
#         if value is None:
#             return bool(default)
#         if isinstance(value, bool):
#             return value
#         if isinstance(value, (int, float)):
#             return bool(value)
#         if isinstance(value, str):
#             v = value.strip().lower()
#             if v in {"1", "true", "yes", "y", "on"}:
#                 return True
#             if v in {"0", "false", "no", "n", "off", "none", "null", ""}:
#                 return False
#         return bool(value)

#     def _spectral_summary_is_physical_default(self, spectral_dim: int = 0, *, source: str = "input") -> bool:
#         """Decide whether spectral_summary has physical wavelength order.

#         Spectral derivatives are valid only over raw wavelength-ordered HSI
#         bands. PCA/input summaries are allowed for band diagnostics but must not
#         activate physical spectral-shape energy.
#         """
#         explicit = getattr(self, "spectral_summary_is_physical", getattr(self.args, "spectral_summary_is_physical", None))
#         if explicit is not None:
#             return self._cfg_bool("spectral_summary_is_physical", bool(explicit))

#         if source in {"batch_metadata", "dataset_raw", "external"}:
#             ds_flag = self._dataset_spectra_are_physical()
#             if ds_flag is not None:
#                 return bool(ds_flag)
#             return self._cfg_bool("raw_spectral_summary_is_physical", True)

#         pca_components = int(getattr(self.args, "pca_components", 0) or 0)
#         uses_pca = self._cfg_bool("use_pca", pca_components > 0)
#         if uses_pca:
#             if pca_components <= 0 or int(spectral_dim) <= pca_components:
#                 return False
#             if int(spectral_dim) == pca_components:
#                 return False
#         return self._cfg_bool("input_spectral_summary_is_physical", not uses_pca)

#     def _get_class_external_spectra_with_flag(
#         self,
#         cls: int,
#         split: str,
#         expected_n: int,
#     ) -> Tuple[Optional[torch.Tensor], bool]:
#         """Try to obtain raw/physical center spectra from the dataset.

#         Prefer dataset methods that support ``require_physical=True``. If a
#         dataset returns PCA/reduced metadata, the tensor may still be returned,
#         but the physical flag is False so SRGP derivative scoring stays off.
#         """
#         method_names = (
#             "get_class_spectra",
#             "get_class_spectrum",
#             "get_class_center_spectra",
#             "get_class_raw_spectra",
#             "get_class_spectral_summary",
#             "get_class_center_spectrum",
#         )
#         dataset_physical = self._dataset_spectra_are_physical()
#         for name in method_names:
#             fn = getattr(self.dataset, name, None)
#             if not callable(fn):
#                 continue
#             call_attempts = (
#                 dict(split=split, require_physical=True),
#                 dict(split=split),
#                 {},
#             )
#             for kwargs in call_attempts:
#                 try:
#                     val = fn(int(cls), **kwargs)
#                 except TypeError:
#                     continue
#                 except Exception:
#                     continue
#                 if val is None:
#                     continue
#                 try:
#                     t = val.detach().cpu().float() if torch.is_tensor(val) else torch.as_tensor(val).float()
#                     t = self._normalize_spectral_metadata_tensor(t, expected_n=int(expected_n)).cpu().float()
#                 except Exception:
#                     continue
#                 if t.numel() == 0 or t.dim() != 2 or t.size(0) != int(expected_n):
#                     continue
#                 physical = bool(dataset_physical) if dataset_physical is not None else ("raw" in name or "physical" in name or "center_spectra" in name)
#                 # Safety: metadata with the same width as PCA input is reduced, not raw.
#                 pca_components = int(getattr(self.args, "pca_components", 0) or 0)
#                 uses_pca = self._cfg_bool("use_pca", pca_components > 0)
#                 if uses_pca and pca_components > 0 and int(t.size(1)) <= pca_components:
#                     physical = False
#                 return t, bool(physical)
#         return None, False

#     def _get_class_external_spectra(self, cls: int, split: str, expected_n: int) -> Optional[torch.Tensor]:
#         spectra, _ = self._get_class_external_spectra_with_flag(cls, split, expected_n)
#         return spectra

#     def _resolve_batch_spectral_summary(
#         self,
#         x: torch.Tensor,
#         *,
#         spectra: Optional[torch.Tensor] = None,
#         model_out: Optional[Dict[str, torch.Tensor]] = None,
#         source: str = "input",
#         spectral_summary_is_physical: Optional[bool] = None,
#     ) -> Tuple[torch.Tensor, bool]:
#         """Return spectral summary and physical-band flag for SRGP.

#         Priority:
#             1) explicit spectra from dataset/loader metadata, center-pixel only;
#             2) model_out['spectral_summary'];
#             3) center spectrum from input tensor.
#         """
#         if spectra is not None and torch.is_tensor(spectra) and spectra.numel() > 0:
#             ss = self._normalize_spectral_metadata_tensor(spectra, ref_x=x)
#             physical = bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else self._spectral_summary_is_physical_default(int(ss.size(1)), source=source)
#             return ss, physical

#         if isinstance(model_out, dict):
#             ss = model_out.get("spectral_summary", None)
#             if torch.is_tensor(ss) and ss.numel() > 0:
#                 ss = self._normalize_spectral_metadata_tensor(ss, ref_x=x)
#                 if ss.size(0) == x.size(0):
#                     flag = model_out.get("spectral_summary_is_physical", None)
#                     if torch.is_tensor(flag) and flag.numel() == 1:
#                         return ss, bool(flag.detach().cpu().item())
#                     if isinstance(flag, bool):
#                         return ss, flag
#                     physical = bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else self._spectral_summary_is_physical_default(int(ss.size(1)), source="model")
#                     return ss, physical

#         ss = self._center_spectrum_from_tensor(x).to(device=x.device, dtype=x.dtype)
#         if ss.dim() != 2:
#             ss = ss.flatten(1)
#         physical = bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else self._spectral_summary_is_physical_default(int(ss.size(1)), source="input")
#         return torch.nan_to_num(ss, nan=0.0, posinf=0.0, neginf=0.0), physical

#     def _spatial_spectral_manifold_loss(
#         self,
#         features: torch.Tensor,
#         spectra: Optional[torch.Tensor],
#         labels: torch.Tensor,
#         coords: Optional[torch.Tensor] = None,
#         *,
#         weight: float = 0.0,
#         margin: float = 1.0,
#         temperature: float = 0.20,
#         neg_k: int = 4,
#         spatial_radius: float = 2.0,
#         require_same_label_positive: bool = True,
#         return_parts: bool = False,
#     ):
#         """
#         MSSL-inspired HSI manifold regularizer on the canonical projected z.

#         It is deliberately a regularizer, not a replacement for the GeometryBank.
#         Positives are selected from same-label local/spectral neighbors when
#         possible; negatives are different-label hard neighbors. The negative
#         term uses a margin instead of the paper's raw subtractive objective to
#         avoid unbounded mini-batch optimization.
#         """
#         if weight <= 0.0 or features is None or not torch.is_tensor(features) or features.numel() == 0:
#             z = self._zero(features)
#             out = {"total": z, "loss": z, "pos": z, "neg": z, "mean_pos_weight": z, "mean_neg_weight": z}
#             return out if return_parts else z
#         if features.dim() != 2:
#             raise RuntimeError(f"MSSL features must be [B,D], got {tuple(features.shape)}")
#         B = int(features.size(0))
#         if B <= 2:
#             z0 = features.sum() * 0.0
#             out = {"total": z0, "loss": z0, "pos": z0, "neg": z0, "mean_pos_weight": z0, "mean_neg_weight": z0}
#             return out if return_parts else z0

#         labels = labels.to(device=features.device).long().view(-1)
#         if labels.numel() != B:
#             raise RuntimeError(f"MSSL labels/features mismatch: {labels.numel()} vs {B}")

#         if spectra is None or not torch.is_tensor(spectra) or spectra.numel() == 0:
#             # Last-resort fallback. Prefer dataloader center spectra or model spectral_summary.
#             spectra = features.detach()
#         spectra = spectra.to(device=features.device, dtype=features.dtype)
#         if spectra.dim() != 2:
#             spectra = spectra.flatten(1)
#         if spectra.size(0) != B:
#             raise RuntimeError(f"MSSL spectra/features mismatch: {spectra.size(0)} vs {B}")

#         z_feat = F.normalize(features, dim=1)
#         z_spec = F.normalize(torch.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
#         feat_dist = torch.cdist(z_feat, z_feat, p=2)
#         spec_dist = torch.cdist(z_spec, z_spec, p=2)
#         eye = torch.eye(B, device=features.device, dtype=torch.bool)

#         same = labels[:, None].eq(labels[None, :])
#         diff = ~same

#         if coords is not None and torch.is_tensor(coords) and coords.numel() > 0:
#             coords_t = coords.to(device=features.device, dtype=features.dtype)
#             if coords_t.dim() == 2 and coords_t.size(0) == B and coords_t.size(1) >= 2:
#                 coord_dist = torch.cdist(coords_t[:, :2], coords_t[:, :2], p=2)
#                 spatial_pos = coord_dist <= float(spatial_radius)
#                 spatial_neg = coord_dist > float(spatial_radius)
#             else:
#                 spatial_pos = ~eye
#                 spatial_neg = ~eye
#         else:
#             spatial_pos = ~eye
#             spatial_neg = ~eye

#         # Positive: prefer same-label, spatially local, spectrally nearest.
#         pos_mask = spatial_pos & (~eye)
#         if bool(require_same_label_positive):
#             pos_mask = pos_mask & same
#         spec_for_pos = spec_dist.masked_fill(~pos_mask, float("inf"))
#         has_pos = torch.isfinite(spec_for_pos).any(dim=1)

#         # Fallback 1: any same-label non-self sample. Fallback 2: nearest spectral non-self.
#         same_nonself = same & (~eye)
#         spec_same = spec_dist.masked_fill(~same_nonself, float("inf"))
#         has_same = torch.isfinite(spec_same).any(dim=1)
#         spec_any = spec_dist.masked_fill(eye, float("inf"))
#         pos_idx_primary = spec_for_pos.argmin(dim=1)
#         pos_idx_same = spec_same.argmin(dim=1)
#         pos_idx_any = spec_any.argmin(dim=1)
#         pos_idx = torch.where(has_pos, pos_idx_primary, torch.where(has_same, pos_idx_same, pos_idx_any))

#         row = torch.arange(B, device=features.device)
#         pos_fdist = feat_dist[row, pos_idx]
#         pos_sdist = spec_dist[row, pos_idx]
#         temp2 = 2.0 * float(temperature) * float(temperature) + 1e-12
#         pos_w = torch.exp(-(pos_sdist.detach().pow(2)) / temp2).clamp(0.05, 1.0)
#         pos_loss = (pos_w * pos_fdist).mean()

#         # Negative: prefer different-label hard neighbors. If a batch has one class only,
#         # fallback to non-local samples, then any non-self sample.
#         neg_mask = diff & (~eye)
#         has_neg = neg_mask.any(dim=1)
#         nonlocal_mask = spatial_neg & (~eye)
#         any_mask = ~eye
#         neg_mask = torch.where(has_neg[:, None], neg_mask, torch.where(nonlocal_mask.any(dim=1)[:, None], nonlocal_mask, any_mask))
#         neg_d = feat_dist.masked_fill(~neg_mask, float("inf"))
#         k = max(1, min(int(neg_k), B - 1))
#         neg_idx = neg_d.topk(k=k, largest=False, dim=1).indices
#         anchor = row.view(B, 1).expand_as(neg_idx)
#         neg_fdist = feat_dist[anchor, neg_idx]
#         neg_sdist = spec_dist[anchor, neg_idx]
#         # Close spectral negatives are dangerous for HSI confusion; penalize them more.
#         neg_w = torch.exp(-(neg_sdist.detach().pow(2)) / temp2).clamp(0.05, 1.0)
#         neg_loss = (neg_w * torch.relu(float(margin) - neg_fdist)).mean()

#         raw = pos_loss + neg_loss
#         total = float(weight) * raw
#         out = {
#             "total": total,
#             "loss": raw.detach(),
#             "pos": pos_loss.detach(),
#             "neg": neg_loss.detach(),
#             "mean_pos_weight": pos_w.mean().detach(),
#             "mean_neg_weight": neg_w.mean().detach(),
#         }
#         return out if return_parts else total

#     # ------------------------------------------------------------------
#     # Clean GeometryBank validation/access
#     # ------------------------------------------------------------------
#     def _canonicalize_bank(self, bank: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Create aliases required by the SRGP feature+spectral geometry path."""
#         if "res_vars" not in bank and "resvars" in bank:
#             bank["res_vars"] = bank["resvars"]
#         if "resvars" not in bank and "res_vars" in bank:
#             bank["resvars"] = bank["res_vars"]

#         if "variances" not in bank and "eigvals" in bank and "res_vars" in bank:
#             eig = bank["eigvals"]
#             res = bank["res_vars"]
#             if torch.is_tensor(eig) and torch.is_tensor(res) and eig.numel() > 0 and res.numel() > 0:
#                 bank["variances"] = torch.cat([eig, res.unsqueeze(-1)], dim=-1)

#         if "eigvals" not in bank and "variances" in bank and torch.is_tensor(bank["variances"]):
#             bank["eigvals"] = bank["variances"][:, :-1]
#         if "res_vars" not in bank and "variances" in bank and torch.is_tensor(bank["variances"]):
#             bank["res_vars"] = bank["variances"][:, -1]
#             bank["resvars"] = bank["res_vars"]

#         if "band_importances" not in bank and "band_importance" in bank:
#             bank["band_importances"] = bank["band_importance"]
#         if "band_importance" not in bank and "band_importances" in bank:
#             bank["band_importance"] = bank["band_importances"]

#         # SRGP spectral-shape aliases.  The fixed GeometryBank uses plural names;
#         # older intermediate files may use singular names.  Keep both stable.
#         alias_pairs = (
#             ("spectral_curve_means", "spectral_curve_mean"),
#             ("spectral_curve_vars", "spectral_curve_var"),
#             ("spectral_curve_d1", "spectral_d1"),
#             ("spectral_curve_d2", "spectral_d2"),
#         )
#         for primary, alias in alias_pairs:
#             if primary not in bank and alias in bank:
#                 bank[primary] = bank[alias]
#             if alias not in bank and primary in bank:
#                 bank[alias] = bank[primary]
#         if "spectral_shape_reliability" not in bank and "spectral_reliability" in bank:
#             bank["spectral_shape_reliability"] = bank["spectral_reliability"]
#         if "spectral_reliability" not in bank and "spectral_shape_reliability" in bank:
#             bank["spectral_reliability"] = bank["spectral_shape_reliability"]
#         return bank

#     def _valid_mask_from_bank(self, bank: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """Strict valid-row mask for the cleaned GeometryBank.

#         Do not fabricate reliability/sample-count fallbacks here.  Capacity rows,
#         NaN rows, rows with invalid active rank, or invalid band signatures must
#         not be scoreable by the classifier/losses.
#         """
#         bank = self._canonicalize_bank(bank)
#         means = bank.get("means", None)
#         bases = bank.get("bases", None)
#         variances = bank.get("variances", None)
#         counts = bank.get("sample_counts", None)
#         active = bank.get("active_ranks", None)
#         reliability = bank.get("reliability", None)
#         if not torch.is_tensor(means) or means.dim() != 2:
#             raise RuntimeError("GeometryBank must expose means [C,D].")
#         C = int(means.size(0))
#         if not torch.is_tensor(bases) or bases.dim() != 3 or bases.size(0) != C:
#             raise RuntimeError("GeometryBank must expose bases [C,D,R].")
#         if not torch.is_tensor(variances) or variances.dim() != 2 or variances.size(0) != C:
#             raise RuntimeError("GeometryBank must expose variances [C,R+1].")
#         if not torch.is_tensor(counts) or counts.numel() != C:
#             raise RuntimeError("GeometryBank must expose real sample_counts [C]. Capacity rows are not valid memory.")
#         if not torch.is_tensor(active) or active.numel() != C:
#             raise RuntimeError("GeometryBank must expose active_ranks [C].")
#         if not torch.is_tensor(reliability) or reliability.numel() != C:
#             raise RuntimeError("GeometryBank must expose reliability [C].")

#         device = means.device
#         counts = counts.to(device=device).flatten()
#         active = active.to(device=device).long().flatten()
#         reliability = reliability.to(device=device).flatten()
#         finite = (
#             torch.isfinite(means).all(dim=1)
#             & torch.isfinite(bases).flatten(1).all(dim=1)
#             & torch.isfinite(variances).all(dim=1)
#             & torch.isfinite(counts)
#             & torch.isfinite(reliability)
#         )
#         R = int(bases.size(2))
#         active_ok = (active >= 0) & (active <= R)
#         sample_cap_ok = active <= torch.clamp(counts.long() - 1, min=0, max=R)
#         valid = (counts > 0) & finite & active_ok & sample_cap_ok

#         bands = bank.get("band_importances", bank.get("band_importance", None))
#         if torch.is_tensor(bands) and bands.numel() > 0:
#             if bands.dim() != 2 or bands.size(0) != C:
#                 raise RuntimeError("GeometryBank band_importances must be [C,S] when present.")
#             b = bands.to(device=device)
#             bfinite = torch.isfinite(b).all(dim=1)
#             bsum = b.clamp_min(0.0).sum(dim=1)
#             valid = valid & bfinite & (bsum > 1e-8)

#         # SRGP spectral-shape rows are not mandatory for synthetic replay, but
#         # when present they must be finite for a row to participate in SRGP
#         # spectral conflict/diagnostic scoring.
#         for skey in ("spectral_curve_means", "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2"):
#             sv = bank.get(skey, None)
#             if torch.is_tensor(sv) and sv.numel() > 0:
#                 if sv.dim() != 2 or sv.size(0) != C:
#                     raise RuntimeError(f"GeometryBank {skey} must be [C,S*] when present, got {tuple(sv.shape)}")
#                 valid = valid & torch.isfinite(sv.to(device=device)).all(dim=1)
#         srel = bank.get("spectral_shape_reliability", None)
#         if torch.is_tensor(srel) and srel.numel() > 0:
#             if srel.numel() != C:
#                 raise RuntimeError(f"spectral_shape_reliability must have C={C} entries, got {srel.numel()}")
#             valid = valid & torch.isfinite(srel.to(device=device).flatten())
#         return valid

#     def _bank_valid_mask(self, bank: Dict[str, torch.Tensor]) -> torch.Tensor:
#         return self._valid_mask_from_bank(bank)

#     def _safe_get_subspace_bank(self, require_ready: bool = True) -> Dict[str, torch.Tensor]:
#         if not hasattr(self.model, "get_subspace_bank"):
#             raise AttributeError("Model must expose get_subspace_bank().")
#         bank = self.model.get_subspace_bank()
#         if not isinstance(bank, dict):
#             raise TypeError(f"get_subspace_bank() must return dict, got {type(bank)}")
#         bank = self._canonicalize_bank(bank)
#         if not require_ready:
#             return bank

#         required = ("means", "bases", "variances", "sample_counts")
#         for key in required:
#             if key not in bank or not torch.is_tensor(bank[key]) or bank[key].numel() == 0:
#                 raise RuntimeError(f"GeometryBank missing required key '{key}'.")
#         means, bases, variances = bank["means"], bank["bases"], bank["variances"]
#         if means.dim() != 2:
#             raise RuntimeError(f"bank['means'] must be [C,D], got {tuple(means.shape)}")
#         if bases.dim() != 3:
#             raise RuntimeError(f"bank['bases'] must be [C,D,R], got {tuple(bases.shape)}")
#         if variances.dim() != 2:
#             raise RuntimeError(f"bank['variances'] must be [C,R+1], got {tuple(variances.shape)}")
#         C, D = means.shape
#         if bases.size(0) != C or bases.size(1) != D or variances.size(0) != C or variances.size(1) != bases.size(2) + 1:
#             raise RuntimeError(
#                 f"GeometryBank shape mismatch: means={tuple(means.shape)}, bases={tuple(bases.shape)}, variances={tuple(variances.shape)}"
#             )
#         if not torch.isfinite(means).all() or not torch.isfinite(bases).all() or not torch.isfinite(variances).all():
#             raise RuntimeError("GeometryBank contains NaN/Inf in feature geometry.")

#         device = means.device
#         for key in ("active_ranks", "reliability"):
#             if key not in bank or not torch.is_tensor(bank[key]) or bank[key].numel() != C:
#                 raise RuntimeError(
#                     f"GeometryBank missing required key '{key}' with width C={C}. "
#                     "Do not fabricate active ranks or reliability for scoring."
#                 )
#         bank["sample_counts"] = bank["sample_counts"].to(device=device)
#         bank["active_ranks"] = bank["active_ranks"].to(device=device).long().flatten()
#         bank["reliability"] = bank["reliability"].to(device=device, dtype=means.dtype).flatten()
#         bank["valid_mask"] = self._valid_mask_from_bank(bank).to(device=device)
#         if not bool(bank["valid_mask"].any().item()):
#             raise RuntimeError("GeometryBank has no valid built rows; all sample_counts are zero or invalid.")
#         return bank

#     def _validate_bank_has_classes(self, bank: Dict[str, torch.Tensor], class_ids: Iterable[int]) -> None:
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             return
#         C = int(bank["means"].size(0))
#         max_id = max(ids)
#         if max_id >= C:
#             raise RuntimeError(f"GeometryBank has C={C} rows but requested class id {max_id}.")
#         counts = bank.get("sample_counts", None)
#         if torch.is_tensor(counts):
#             missing = [int(c) for c in ids if c < counts.numel() and float(counts[c].detach().item()) <= 0.0]
#             if missing:
#                 raise RuntimeError(f"GeometryBank rows exist but are not built for classes: {missing}")

#     def _class_memory_is_valid(self, cls: int) -> bool:
#         try:
#             bank = self._safe_get_subspace_bank(require_ready=False)
#             bank = self._canonicalize_bank(bank)
#         except Exception:
#             return False
#         cls = int(cls)
#         means, bases, vars_, counts = bank.get("means"), bank.get("bases"), bank.get("variances"), bank.get("sample_counts")
#         if not (torch.is_tensor(means) and torch.is_tensor(bases) and torch.is_tensor(vars_) and torch.is_tensor(counts)):
#             return False
#         if cls < 0 or cls >= means.size(0) or cls >= bases.size(0) or cls >= vars_.size(0) or cls >= counts.numel():
#             return False
#         try:
#             valid = self._valid_mask_from_bank(bank)
#             return bool(valid.numel() > cls and valid[cls].detach().item())
#         except Exception:
#             return bool(float(counts[cls].detach().item()) > 0.0 and torch.isfinite(means[cls]).all() and torch.isfinite(bases[cls]).all() and torch.isfinite(vars_[cls]).all())

#     # ------------------------------------------------------------------
#     # Label helpers
#     # ------------------------------------------------------------------
#     def _labels_are_local(self, y: torch.Tensor, class_ids: Iterable[int]) -> bool:
#         class_ids = self._as_class_list(class_ids)
#         if y is None or y.numel() == 0:
#             return False
#         unique = set(int(v) for v in y.detach().cpu().unique().tolist())
#         return unique.issubset(set(range(len(class_ids))))

#     def _global_to_local_labels(self, y: torch.Tensor, class_ids: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor]:
#         class_ids = self._as_class_list(class_ids)
#         y = y.long().view(-1)
#         if self._labels_are_local(y, class_ids):
#             valid = (y >= 0) & (y < len(class_ids))
#             return y, valid
#         y_local = torch.full_like(y, -1)
#         for local_idx, global_cls in enumerate(class_ids):
#             y_local[y == int(global_cls)] = int(local_idx)
#         valid = y_local >= 0
#         return y_local, valid

#     def _masked_weighted_ce_new(self, logits: torch.Tensor, y: torch.Tensor, new_class_ids) -> torch.Tensor:
#         ids = self._as_class_list(new_class_ids)
#         if not ids:
#             return self._zero(logits)
#         class_ids = torch.tensor(ids, device=logits.device, dtype=torch.long)
#         if int(class_ids.max().item()) >= logits.size(1):
#             raise RuntimeError(f"max new class id {int(class_ids.max())} exceeds logits width {logits.size(1)}")
#         logits_new = logits.index_select(1, class_ids)
#         y_local, valid = self._global_to_local_labels(y.to(logits.device), ids)
#         if not bool(valid.any().item()):
#             return self._zero(logits)
#         logits_new = logits_new[valid]
#         y_local = y_local[valid]
#         counts = torch.bincount(y_local, minlength=len(ids)).float().to(logits.device)
#         weights = counts.sum() / counts.clamp_min(1.0)
#         weights = weights / weights.mean().clamp_min(1e-8)
#         return F.cross_entropy(logits_new, y_local, weight=weights.to(logits.dtype))

#     def _incremental_accuracy_with_count(self, logits: torch.Tensor, y: torch.Tensor, new_class_ids) -> Tuple[int, int]:
#         ids = self._as_class_list(new_class_ids)
#         if not ids:
#             return 0, 0
#         class_ids = torch.tensor(ids, device=logits.device, dtype=torch.long)
#         if int(class_ids.max().item()) >= logits.size(1):
#             raise RuntimeError(f"max new class id {int(class_ids.max())} exceeds logits width {logits.size(1)}")
#         pred_local = logits.index_select(1, class_ids).argmax(dim=1)
#         y_local, valid = self._global_to_local_labels(y.to(logits.device), ids)
#         if not bool(valid.any().item()):
#             return 0, 0
#         return int((pred_local[valid] == y_local[valid]).sum().item()), int(valid.sum().item())

#     # ------------------------------------------------------------------
#     # Geometry memory extraction/update
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _extract_backbone_outputs_for_class(self, cls: int, split: str = "train") -> Dict[str, torch.Tensor]:
#         cls = int(cls)
#         patches = self.dataset.get_class_patches(cls, split=split)
#         x_cpu = patches.detach().cpu().float() if torch.is_tensor(patches) else torch.from_numpy(patches).float()
#         if x_cpu.numel() == 0 or x_cpu.size(0) == 0:
#             raise RuntimeError(f"No patches available for class {cls} split='{split}'.")
#         if not hasattr(self.model, "extract_projected_features"):
#             raise AttributeError("SRGP GeometryBank construction requires model.extract_projected_features(x).")

#         external_spectra, external_is_physical = self._get_class_external_spectra_with_flag(
#             cls, split=split, expected_n=int(x_cpu.size(0))
#         )
#         if external_spectra is not None:
#             input_channels = int(x_cpu.size(1)) if x_cpu.dim() >= 2 else 0
#             spectra_dim = int(external_spectra.size(1)) if external_spectra.dim() == 2 else 0
#             pca_components = int(getattr(self.args, "pca_components", 0) or 0)
#             uses_pca = self._cfg_bool("use_pca", pca_components > 0)
#             if uses_pca and input_channels > 0 and spectra_dim == input_channels:
#                 external_is_physical = False
#         expected_dim = int(getattr(self.model, "d_model", getattr(self.args, "d_model", 0)))
#         bs = int(max(1, getattr(self.args, "subspace_extract_batch_size", 256)))
#         was_training = bool(self.model.training)
#         self.model.eval()
#         feats: List[torch.Tensor] = []
#         base_feats: List[torch.Tensor] = []
#         adapter_gates: List[torch.Tensor] = []
#         spectral: List[torch.Tensor] = []
#         bands: List[torch.Tensor] = []
#         physical_flags: List[bool] = []
#         have_band = True
#         try:
#             for start in range(0, x_cpu.size(0), bs):
#                 xb = x_cpu[start:start + bs].to(self.device, non_blocking=True)
#                 sb = None
#                 if external_spectra is not None:
#                     sb = external_spectra[start:start + bs].to(self.device, non_blocking=True)
#                 try:
#                     out = self.model.extract_projected_features(
#                         xb,
#                         spectral_summary=sb,
#                         spectral_summary_is_physical=bool(external_is_physical),
#                     )
#                 except TypeError:
#                     # Backward compatibility with older model files.  We still
#                     # recover spectra locally and pass them to the bank.
#                     out = self.model.extract_projected_features(xb)
#                 if not isinstance(out, dict) or "features" not in out:
#                     raise RuntimeError("extract_projected_features(x) must return dict with key 'features'.")

#                 feat = out["features"]
#                 if "projected_features" in out and torch.is_tensor(out["projected_features"]):
#                     projected = out["projected_features"]
#                     if projected.shape != feat.shape or not torch.allclose(feat, projected, atol=1e-5, rtol=1e-4):
#                         raise RuntimeError(
#                             "Canonical z-space mismatch: out['features'] differs from "
#                             "out['projected_features']. GeometryBank must be built from one final z-space only."
#                         )
#                 if feat.dim() != 2 or (expected_dim > 0 and feat.size(1) != expected_dim):
#                     raise RuntimeError(f"Canonical projected features must be [B,{expected_dim}], got {tuple(feat.shape)}")
#                 if not torch.isfinite(feat).all():
#                     raise RuntimeError(f"Non-finite canonical projected features for class {cls}.")
#                 feats.append(feat.detach().cpu())
#                 # G²RPA contract: GeometryBank rows for new classes are built
#                 # from final adapted z. Keep pre-adapter/gate only for diagnostics.
#                 bf = out.get("base_features", out.get("pre_adapter_features", None))
#                 if torch.is_tensor(bf) and bf.shape == feat.shape:
#                     base_feats.append(bf.detach().cpu())
#                 gate = out.get("adapter_gate", None)
#                 if torch.is_tensor(gate) and gate.size(0) == feat.size(0):
#                     adapter_gates.append(gate.detach().cpu())

#                 ss, is_phys = self._resolve_batch_spectral_summary(
#                     xb,
#                     spectra=sb,
#                     model_out=out,
#                     source="dataset_raw" if sb is not None else "input",
#                     spectral_summary_is_physical=bool(external_is_physical) if sb is not None else None,
#                 )
#                 if ss.dim() != 2 or ss.size(0) != xb.size(0):
#                     raise RuntimeError(f"SRGP spectral_summary must be [B,S], got {tuple(ss.shape)}")
#                 spectral.append(ss.detach().cpu())
#                 physical_flags.append(bool(is_phys))

#                 bw = out.get("band_weights", None)
#                 if not (torch.is_tensor(bw) and bw.dim() == 2 and bw.size(0) == xb.size(0) and bw.numel() > 0):
#                     bw = out.get("band_summary", out.get("band_importance", None))
#                 if torch.is_tensor(bw) and bw.dim() == 2 and bw.size(0) == xb.size(0) and bw.numel() > 0:
#                     bands.append(bw.detach().cpu())
#                 else:
#                     have_band = False
#         finally:
#             self.model.train(was_training)

#         features = torch.cat(feats, dim=0).to(self.device)
#         spectral_summary = torch.cat(spectral, dim=0).to(self.device)
#         spectral_is_physical = bool(physical_flags) and all(physical_flags)
#         out = {
#             "features": features,
#             "spectral_summary": spectral_summary,
#             "spectral_summary_is_physical": torch.tensor(float(spectral_is_physical), device=self.device),
#         }
#         if len(base_feats) == len(feats):
#             out["base_features"] = torch.cat(base_feats, dim=0).to(self.device)
#         if len(adapter_gates) == len(feats):
#             out["adapter_gate"] = torch.cat(adapter_gates, dim=0).to(self.device)
#         if have_band and len(bands) == len(feats):
#             band_weights = torch.cat(bands, dim=0).to(self.device)
#             # Band weights live in model-input band space (often PCA=30), while
#             # spectral_summary may be raw physical space (e.g., 200 bands). Do
#             # not drop valid band weights just because those dimensions differ.
#             if band_weights.size(0) == features.size(0) and band_weights.dim() == 2 and band_weights.size(1) > 0 and torch.isfinite(band_weights).all():
#                 out["band_weights"] = band_weights
#         return out

#     @torch.no_grad()
#     def _extract_class_geometry_dict(self, cls: int, split: str = "train") -> Dict[str, torch.Tensor]:
#         cls = int(cls)
#         outs = self._extract_backbone_outputs_for_class(cls, split=split)
#         features = outs["features"]
#         labels = torch.full((features.size(0),), cls, device=features.device, dtype=torch.long)
#         bank = getattr(self.model, "geometry_bank", None)
#         if bank is None or not hasattr(bank, "extract_geometry"):
#             raise AttributeError("model.geometry_bank.extract_geometry() is required.")
#         try:
#             geom = bank.extract_geometry(
#                 features=features,
#                 labels=labels,
#                 spectral_summary=outs.get("spectral_summary", None),
#                 band_weights=outs.get("band_weights", None),
#                 spectral_summary_is_physical=bool(torch.as_tensor(outs.get("spectral_summary_is_physical", 0.0)).detach().cpu().item()),
#             )
#         except TypeError:
#             geom = bank.extract_geometry(
#                 features=features,
#                 labels=labels,
#                 spectral_summary=outs.get("spectral_summary", None),
#                 band_weights=outs.get("band_weights", None),
#             )
#         if cls not in geom:
#             raise RuntimeError(f"Geometry extraction failed for class {cls}.")
#         g = geom[cls]
#         required = ("mean", "basis", "eigvals", "res_var", "active_rank", "reliability", "sample_count")
#         missing = [k for k in required if k not in g]
#         if missing:
#             raise RuntimeError(f"Geometry extraction for class {cls} missing keys: {missing}")
#         return g

#     @torch.no_grad()
#     def _extract_class_geometry(self, cls: int, split: str = "train", rank=None):
#         del rank
#         g = self._extract_class_geometry_dict(cls, split=split)
#         return (
#             g["mean"], g["basis"], g["eigvals"], g["res_var"],
#             None, g.get("band_importance", None), g["active_rank"], g["reliability"], g["sample_count"],
#         )

#     # ------------------------------------------------------------------
#     # Descriptor-refinement support for clean incremental phase
#     # ------------------------------------------------------------------
#     def _make_refined_bank_view(
#         self,
#         class_ids: Iterable[int],
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         *,
#         active_ranks: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         band_importances: Optional[torch.Tensor] = None,
#         spectral_curve_means: Optional[torch.Tensor] = None,
#         spectral_curve_vars: Optional[torch.Tensor] = None,
#         spectral_curve_d1: Optional[torch.Tensor] = None,
#         spectral_curve_d2: Optional[torch.Tensor] = None,
#         spectral_shape_reliability: Optional[torch.Tensor] = None,
#         base_bank: Optional[Dict[str, torch.Tensor]] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Return a temporary scoring bank with only selected rows replaced.

#         This is used by descriptor-only refinement. It does not write to
#         GeometryBank. Old rows are cloned from the frozen bank and remain
#         unchanged.
#         """
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             raise RuntimeError("Cannot create refined bank view with empty class_ids.")
#         bank = self._canonicalize_bank(base_bank if base_bank is not None else self._safe_get_subspace_bank(require_ready=True))
#         out: Dict[str, torch.Tensor] = {}
#         for key, value in bank.items():
#             if torch.is_tensor(value):
#                 out[key] = value.detach().clone()
#             else:
#                 out[key] = value

#         means = means.to(device=out["means"].device, dtype=out["means"].dtype)
#         bases = bases.to(device=out["bases"].device, dtype=out["bases"].dtype)
#         variances = variances.to(device=out["variances"].device, dtype=out["variances"].dtype)
#         if means.dim() != 2 or means.size(0) != len(ids):
#             raise RuntimeError(f"refined means must be [K,D], got {tuple(means.shape)} for K={len(ids)}")
#         if bases.dim() != 3 or bases.size(0) != len(ids):
#             raise RuntimeError(f"refined bases must be [K,D,R], got {tuple(bases.shape)} for K={len(ids)}")
#         if variances.dim() != 2 or variances.size(0) != len(ids):
#             raise RuntimeError(f"refined variances must be [K,R+1], got {tuple(variances.shape)} for K={len(ids)}")

#         max_id = max(ids)
#         if max_id >= out["means"].size(0):
#             raise RuntimeError(f"Refined class id {max_id} exceeds GeometryBank rows {out['means'].size(0)}")
#         out["means"][ids] = means
#         out["bases"][ids] = bases
#         out["variances"][ids] = variances
#         out["eigvals"] = out["variances"][:, :-1]
#         out["res_vars"] = out["variances"][:, -1]
#         out["resvars"] = out["res_vars"]

#         if active_ranks is not None and torch.is_tensor(active_ranks) and "active_ranks" in out:
#             out["active_ranks"][ids] = active_ranks.to(device=out["active_ranks"].device).long().flatten()
#         if reliability is not None and torch.is_tensor(reliability) and "reliability" in out:
#             out["reliability"][ids] = reliability.to(device=out["reliability"].device, dtype=out["reliability"].dtype).flatten()
#         if sample_counts is not None and torch.is_tensor(sample_counts) and "sample_counts" in out:
#             out["sample_counts"][ids] = sample_counts.to(device=out["sample_counts"].device, dtype=out["sample_counts"].dtype).flatten()
#         if band_importances is not None and torch.is_tensor(band_importances) and "band_importances" in out:
#             if out["band_importances"].dim() == 2 and out["band_importances"].size(1) == band_importances.size(1):
#                 out["band_importances"][ids] = band_importances.to(device=out["band_importances"].device, dtype=out["band_importances"].dtype)
#                 out["band_importance"] = out["band_importances"]

#         spectral_rows = {
#             "spectral_curve_means": spectral_curve_means,
#             "spectral_curve_vars": spectral_curve_vars,
#             "spectral_curve_d1": spectral_curve_d1,
#             "spectral_curve_d2": spectral_curve_d2,
#         }
#         for key, val in spectral_rows.items():
#             if val is not None and torch.is_tensor(val) and key in out and torch.is_tensor(out[key]) and out[key].numel() > 0:
#                 if out[key].dim() == 2 and out[key].size(1) == val.size(1):
#                     out[key][ids] = val.to(device=out[key].device, dtype=out[key].dtype)
#         if spectral_shape_reliability is not None and torch.is_tensor(spectral_shape_reliability) and "spectral_shape_reliability" in out:
#             out["spectral_shape_reliability"][ids] = spectral_shape_reliability.to(
#                 device=out["spectral_shape_reliability"].device,
#                 dtype=out["spectral_shape_reliability"].dtype,
#             ).flatten()

#         out = self._canonicalize_bank(out)
#         out["valid_mask"] = self._valid_mask_from_bank(out).to(out["means"].device)
#         return out

#     @torch.no_grad()
#     def _commit_refined_feature_rows(
#         self,
#         class_ids: Iterable[int],
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         *,
#         active_ranks: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         feature_reliability: Optional[torch.Tensor] = None,
#         band_importances: Optional[torch.Tensor] = None,
#         band_reliability: Optional[torch.Tensor] = None,
#         spectral_curve_means: Optional[torch.Tensor] = None,
#         spectral_curve_vars: Optional[torch.Tensor] = None,
#         spectral_curve_d1: Optional[torch.Tensor] = None,
#         spectral_curve_d2: Optional[torch.Tensor] = None,
#         spectral_shape_reliability: Optional[torch.Tensor] = None,
#         context: str = "descriptor_refinement",
#     ) -> None:
#         """Commit refined descriptors for current new classes only.

#         This helper is intentionally strict: during phase > 0 it refuses to write
#         any row below ``model.old_class_count``. It routes through
#         ``model.refresh_class_subspace`` so GeometryBank shape policy remains in
#         one place.
#         """
#         ids = self._as_class_list(class_ids)
#         if not ids:
#             return
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         phase = int(getattr(self.model, "current_phase", 0))
#         forbidden = [c for c in ids if phase > 0 and c < old_class_count]
#         if forbidden:
#             raise RuntimeError(f"{context}: attempted to commit refined descriptors into frozen old rows: {forbidden}")
#         if not hasattr(self.model, "refresh_class_subspace"):
#             raise AttributeError("Model must expose refresh_class_subspace() to commit refined descriptor rows.")

#         means = means.detach().to(self.device)
#         bases = bases.detach().to(self.device)
#         variances = variances.detach().to(self.device)
#         if variances.size(1) != bases.size(2) + 1:
#             raise RuntimeError(f"variances must be [K,R+1], got {tuple(variances.shape)} with bases {tuple(bases.shape)}")

#         def rows_for_ids(t):
#             if t is None or not torch.is_tensor(t):
#                 return None
#             tt = t.detach().to(self.device)
#             if tt.dim() == 0:
#                 return tt
#             if tt.size(0) == len(ids):
#                 return tt
#             if tt.size(0) > max(ids):
#                 return tt[ids]
#             raise RuntimeError(f"{context}: tensor with first dim {tt.size(0)} cannot provide rows for class ids {ids}")

#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "apply_refined_feature_rows"):
#             gb.apply_refined_feature_rows(
#                 ids,
#                 means=means,
#                 bases=bases,
#                 eigvals=variances[:, :-1].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4))),
#                 res_vars=variances[:, -1].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4))),
#                 reliability=rows_for_ids(reliability),
#                 feature_reliability=rows_for_ids(feature_reliability),
#                 active_ranks=rows_for_ids(active_ranks),
#                 sample_counts=rows_for_ids(sample_counts),
#                 band_importances=rows_for_ids(band_importances),
#                 band_reliability=rows_for_ids(band_reliability),
#                 spectral_curve_means=rows_for_ids(spectral_curve_means),
#                 spectral_curve_vars=rows_for_ids(spectral_curve_vars),
#                 spectral_curve_d1=rows_for_ids(spectral_curve_d1),
#                 spectral_curve_d2=rows_for_ids(spectral_curve_d2),
#                 spectral_shape_reliability=rows_for_ids(spectral_shape_reliability),
#                 allow_frozen_update=False,
#             )
#             if hasattr(gb, "validate_consistency"):
#                 gb.validate_consistency(strict=True)
#             return

#         def row_or_none(t, idx):
#             if t is None or not torch.is_tensor(t):
#                 return None
#             return t.detach().to(self.device)[idx]

#         for local_idx, cls in enumerate(ids):
#             eigvals = variances[local_idx, :-1].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4)))
#             res_var = variances[local_idx, -1].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4)))
#             ar = row_or_none(active_ranks, local_idx)
#             rel = row_or_none(reliability, local_idx)
#             cnt = row_or_none(sample_counts, local_idx)
#             frel = row_or_none(feature_reliability, local_idx)
#             bimp = row_or_none(band_importances, local_idx)
#             brel = row_or_none(band_reliability, local_idx)
#             sc_mean = row_or_none(spectral_curve_means, local_idx)
#             sc_var = row_or_none(spectral_curve_vars, local_idx)
#             sc_d1 = row_or_none(spectral_curve_d1, local_idx)
#             sc_d2 = row_or_none(spectral_curve_d2, local_idx)
#             sc_rel = row_or_none(spectral_shape_reliability, local_idx)
#             spectral_shape = None
#             if sc_mean is not None or sc_var is not None or sc_d1 is not None or sc_d2 is not None or sc_rel is not None:
#                 spectral_shape = {
#                     "spectral_curve_mean": sc_mean,
#                     "spectral_curve_var": sc_var,
#                     "spectral_curve_d1": sc_d1,
#                     "spectral_curve_d2": sc_d2,
#                     "spectral_shape_reliability": sc_rel,
#                 }
#             self.model.refresh_class_subspace(
#                 cls=int(cls),
#                 mean=means[local_idx],
#                 basis=bases[local_idx],
#                 eigvals=eigvals,
#                 res_var=res_var,
#                 active_rank=ar if ar is not None else torch.tensor(bases.size(2), device=self.device, dtype=torch.long),
#                 reliability=rel if rel is not None else torch.tensor(1.0, device=self.device, dtype=means.dtype),
#                 sample_count=cnt if cnt is not None else torch.tensor(1.0, device=self.device, dtype=means.dtype),
#                 feature_reliability=frel,
#                 band_importance=bimp,
#                 band_reliability=brel,
#                 spectral_shape=spectral_shape,
#                 spectral_curve_mean=sc_mean,
#                 spectral_curve_var=sc_var,
#                 spectral_curve_d1=sc_d1,
#                 spectral_curve_d2=sc_d2,
#                 spectral_shape_reliability=sc_rel,
#             )

#     # ------------------------------------------------------------------
#     # Overlap-aware admission for incremental new rows
#     # ------------------------------------------------------------------
#     def _overlap_cfg_float(self, name: str, default: float) -> float:
#         return float(getattr(self, name, getattr(self.args, name, default)))

#     def _overlap_cfg_int(self, name: str, default: int) -> int:
#         return int(getattr(self, name, getattr(self.args, name, default)))

#     def _overlap_cfg_bool(self, name: str, default: bool) -> bool:
#         value = getattr(self, name, getattr(self.args, name, default))
#         if isinstance(value, str):
#             return value.strip().lower() in {"1", "true", "yes", "y", "on"}
#         return bool(value)

#     def _safe_active_rank_scalar(self, active_rank, fallback: int) -> int:
#         try:
#             ar = int(torch.as_tensor(active_rank).detach().cpu().item())
#         except Exception:
#             ar = int(fallback)
#         return max(0, min(ar, int(fallback)))

#     @torch.no_grad()
#     def _candidate_old_overlap_risk(
#         self,
#         *,
#         mean: torch.Tensor,
#         basis: torch.Tensor,
#         active_rank,
#         band_importance: Optional[torch.Tensor] = None,
#         spectral_shape: Optional[Dict[str, torch.Tensor]] = None,
#         spectral_curve_d1: Optional[torch.Tensor] = None,
#         old_class_count: Optional[int] = None,
#     ) -> Dict[str, object]:
#         """Measure how much a candidate new row overlaps frozen old rows.

#         SRGP risk is descriptor-only and HSI-aware:
#             center proximity + feature-subspace overlap + band similarity
#             + spectral-shape similarity.
#         """
#         old_class_count = int(getattr(self.model, "old_class_count", 0) if old_class_count is None else old_class_count)
#         if old_class_count <= 0:
#             return {"max_risk": 0.0, "top": None, "pairs": []}
#         try:
#             bank = self._safe_get_subspace_bank(require_ready=True)
#         except Exception:
#             return {"max_risk": 0.0, "top": None, "pairs": []}

#         bank = self._canonicalize_bank(bank)
#         means = bank.get("means", None)
#         bases = bank.get("bases", None)
#         ranks = bank.get("active_ranks", None)
#         counts = bank.get("sample_counts", None)
#         rel = bank.get("reliability", None)
#         bands = bank.get("band_importances", bank.get("band_importance", None))
#         old_d1 = bank.get("spectral_curve_d1", None)
#         if not (torch.is_tensor(means) and torch.is_tensor(bases)):
#             return {"max_risk": 0.0, "top": None, "pairs": []}

#         old_count = min(old_class_count, int(means.size(0)))
#         valid_old = torch.ones(old_count, device=means.device, dtype=torch.bool)
#         if torch.is_tensor(counts) and counts.numel() >= old_count:
#             valid_old = counts[:old_count].to(means.device).flatten() > 0
#         if not bool(valid_old.any().item()):
#             return {"max_risk": 0.0, "top": None, "pairs": []}

#         dtype = means.dtype
#         device = means.device
#         mu_new = torch.as_tensor(mean, device=device, dtype=dtype).flatten()
#         U_new_full = torch.as_tensor(basis, device=device, dtype=dtype)
#         if U_new_full.dim() != 2:
#             return {"max_risk": 0.0, "top": None, "pairs": []}
#         R = int(U_new_full.size(1))
#         rn = self._safe_active_rank_scalar(active_rank, R)
#         U_new = U_new_full[:, :rn] if rn > 0 else torch.empty((U_new_full.size(0), 0), device=device, dtype=dtype)

#         bw_new = None
#         if band_importance is not None and torch.as_tensor(band_importance).numel() > 0:
#             bw_new = torch.as_tensor(band_importance, device=device, dtype=dtype).flatten().clamp_min(0.0)
#             if bw_new.sum() > 1e-8:
#                 bw_new = bw_new / bw_new.norm().clamp_min(1e-8)
#             else:
#                 bw_new = None

#         d1_new = None
#         if spectral_curve_d1 is not None and torch.as_tensor(spectral_curve_d1).numel() > 0:
#             d1_new = torch.as_tensor(spectral_curve_d1, device=device, dtype=dtype).flatten()
#         elif isinstance(spectral_shape, dict):
#             for key in ("spectral_curve_d1", "spectral_d1"):
#                 val = spectral_shape.get(key, None)
#                 if torch.is_tensor(val) and val.numel() > 0:
#                     d1_new = val.to(device=device, dtype=dtype).flatten()
#                     break
#         if d1_new is not None:
#             d1_new = F.normalize(torch.nan_to_num(d1_new, nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()

#         center_w = self._overlap_cfg_float("overlap_admission_center_weight", 0.40)
#         subspace_w = self._overlap_cfg_float("overlap_admission_subspace_weight", 0.35)
#         band_w = self._overlap_cfg_float("overlap_admission_band_weight", 0.10)
#         spec_w = self._overlap_cfg_float("overlap_admission_spectral_shape_weight", 0.15)
#         dscale = max(float(mu_new.numel()) ** 0.5, 1.0)

#         pairs: List[Dict[str, object]] = []
#         for old_cls in range(old_count):
#             if not bool(valid_old[old_cls].item()):
#                 continue
#             dist = torch.norm(means[old_cls].to(device=device, dtype=dtype) - mu_new, p=2) / dscale
#             center_risk = torch.exp(-dist).clamp(0.0, 1.0)

#             ro = int(ranks[old_cls].detach().item()) if torch.is_tensor(ranks) and ranks.numel() > old_cls else int(bases.size(2))
#             ro = max(0, min(ro, int(bases.size(2))))
#             subspace_overlap = torch.tensor(0.0, device=device, dtype=dtype)
#             if ro > 0 and rn > 0:
#                 U_old = bases[old_cls, :, :ro].to(device=device, dtype=dtype)
#                 denom = float(max(min(ro, rn), 1))
#                 subspace_overlap = (U_old.t() @ U_new).pow(2).sum() / denom
#                 subspace_overlap = subspace_overlap.clamp(0.0, 1.0)

#             band_sim = torch.tensor(0.0, device=device, dtype=dtype)
#             if bw_new is not None and torch.is_tensor(bands) and bands.dim() == 2 and bands.size(0) > old_cls and bands.size(1) == bw_new.numel():
#                 bo = bands[old_cls].to(device=device, dtype=dtype).flatten().clamp_min(0.0)
#                 if bo.sum() > 1e-8:
#                     bo = bo / bo.norm().clamp_min(1e-8)
#                     band_sim = torch.dot(bo, bw_new).clamp(0.0, 1.0)

#             spec_sim = torch.tensor(0.0, device=device, dtype=dtype)
#             if d1_new is not None and torch.is_tensor(old_d1) and old_d1.dim() == 2 and old_d1.size(0) > old_cls and old_d1.size(1) == d1_new.numel():
#                 od = F.normalize(torch.nan_to_num(old_d1[old_cls].to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()
#                 spec_sim = torch.dot(od, d1_new).clamp(0.0, 1.0)

#             risk = center_w * center_risk + subspace_w * subspace_overlap + band_w * band_sim + spec_w * spec_sim
#             if torch.is_tensor(rel) and rel.numel() > old_cls:
#                 uncertainty = (1.0 - rel[old_cls].to(device=device, dtype=dtype).clamp(0.05, 1.0)).clamp(0.0, 1.0)
#                 risk = risk * (1.0 + 0.25 * uncertainty)
#             risk = risk.clamp_min(0.0)

#             pairs.append(
#                 {
#                     "old_class": int(old_cls),
#                     "old_name": self._class_name(old_cls),
#                     "feature_center_distance": float((dist * dscale).detach().cpu().item()),
#                     "scaled_center_distance": float(dist.detach().cpu().item()),
#                     "center_risk": float(center_risk.detach().cpu().item()),
#                     "feature_overlap": float(subspace_overlap.detach().cpu().item()),
#                     "band_similarity": float(band_sim.detach().cpu().item()),
#                     "spectral_shape_similarity": float(spec_sim.detach().cpu().item()),
#                     "risk_score": float(risk.detach().cpu().item()),
#                 }
#             )

#         pairs.sort(key=lambda r: float(r["risk_score"]), reverse=True)
#         top = pairs[0] if pairs else None
#         return {"max_risk": float(top["risk_score"]) if top else 0.0, "top": top, "pairs": pairs}

#     @torch.no_grad()
#     def _apply_overlap_aware_admission(self, cls: int, g: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Diagnostic-only overlap check for a candidate new row.

#         The previous helper actively shrank new eigenspectra and capped rank when
#         a candidate overlapped old rows. That is an ad-hoc admission heuristic,
#         not the clean NECIL-HSI method. The clean path now leaves the statistical
#         descriptor untouched at insertion time; overlap is handled by
#         descriptor-only refinement in ``incremental_phase_trainer.py`` and by
#         diagnostics reported after the phase.

#         This method only records/prints risk information if requested. It never
#         mutates ``mean``, ``basis``, ``eigvals``, ``res_var``, or ``active_rank``.
#         """
#         phase = int(getattr(self.model, "current_phase", 0))
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         cls = int(cls)
#         if phase <= 0 or old_class_count <= 0 or cls < old_class_count:
#             return g

#         risk_info = self._candidate_old_overlap_risk(
#             mean=g["mean"],
#             basis=g["basis"],
#             active_rank=g.get("active_rank", None),
#             band_importance=g.get("band_importance", None),
#             spectral_shape=g.get("spectral_shape", None),
#             spectral_curve_d1=g.get("spectral_curve_d1", None),
#             old_class_count=old_class_count,
#         )
#         max_risk = float(risk_info.get("max_risk", 0.0))
#         threshold = self._overlap_cfg_float("overlap_admission_risk_threshold", 0.80)
#         event = {
#             "phase": int(phase),
#             "new_class": int(cls),
#             "new_name": self._class_name(cls),
#             "max_old_overlap_risk": float(max_risk),
#             "admission_gate": 1.0,
#             "descriptor_mutated": False,
#             "threshold": float(threshold),
#             "top_old": risk_info.get("top"),
#         }
#         events = getattr(self, "_last_overlap_admission_events", [])
#         events.append(event)
#         self._last_overlap_admission_events = events[-50:]
#         if (max_risk >= threshold and self._overlap_cfg_bool("print_overlap_admission", True)) or bool(getattr(self, "debug", False)):
#             top = event.get("top_old") or {}
#             print(
#                 f"[OverlapDiagnostic] phase={phase} new={cls}({self._class_name(cls)}) "
#                 f"risk={max_risk:.4f} threshold={threshold:.3f} "
#                 f"top_old={top.get('old_class', 'NA')}({top.get('old_name', 'NA')}) | "
#                 "descriptor_mutated=False"
#             )
#         return g

#     @torch.no_grad()
#     def _build_class_memory_from_current_phase(self, cls: int, split: str = "train") -> None:
#         cls = int(cls)
#         phase = int(getattr(self.model, "current_phase", 0))
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         if phase > 0 and cls < old_class_count:
#             raise RuntimeError(f"Attempted to rebuild frozen old class {cls} during incremental phase {phase}.")

#         g = self._extract_class_geometry_dict(cls, split=split)
#         g = self._apply_overlap_aware_admission(cls, g)
#         if not hasattr(self.model, "refresh_class_subspace"):
#             raise AttributeError("Model must expose refresh_class_subspace().")
#         self.model.refresh_class_subspace(
#             cls=cls,
#             mean=g["mean"],
#             basis=g["basis"],
#             eigvals=g["eigvals"],
#             res_var=g["res_var"],
#             active_rank=g["active_rank"],
#             reliability=g["reliability"],
#             sample_count=g["sample_count"],
#             feature_reliability=g.get("feature_reliability", g.get("reliability", None)),
#             band_importance=g.get("band_importance", None),
#             band_reliability=g.get("band_reliability", None),
#             spectral_shape=g.get("spectral_shape", None),
#             spectral_curve_mean=g.get("spectral_curve_mean", g.get("spectral_curve_means", None)),
#             spectral_curve_var=g.get("spectral_curve_var", g.get("spectral_curve_vars", None)),
#             spectral_curve_d1=g.get("spectral_curve_d1", None),
#             spectral_curve_d2=g.get("spectral_curve_d2", None),
#             spectral_shape_reliability=g.get("spectral_shape_reliability", None),
#         )
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "validate_consistency"):
#             gb.validate_consistency(strict=True)

#     @torch.no_grad()
#     def _infer_spectral_dim_from_dataset(self, class_ids: List[int], split: str = "train") -> int:
#         """Infer raw spectral capacity when available, otherwise input-band dim."""
#         for cls in class_ids:
#             try:
#                 patches = self.dataset.get_class_patches(int(cls), split=split)
#                 n = int(patches.shape[0] if hasattr(patches, "shape") else patches.size(0))
#                 spectra, physical = self._get_class_external_spectra_with_flag(int(cls), split=split, expected_n=n)
#                 if torch.is_tensor(spectra) and spectra.dim() == 2 and spectra.size(1) > 0 and bool(physical):
#                     return int(spectra.size(1))
#             except Exception:
#                 pass
#         for cls in class_ids:
#             try:
#                 patches = self.dataset.get_class_patches(int(cls), split=split)
#                 shape = patches.shape if hasattr(patches, "shape") else tuple(patches.size())
#                 if len(shape) >= 2:
#                     return int(shape[1])
#             except Exception:
#                 continue
#         return 0

#     @torch.no_grad()
#     def _bootstrap_phase_classes(self, phase: int, split: str = "train", force_rebuild: bool = False) -> None:
#         phase = int(phase)
#         class_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         if not class_ids:
#             return
#         spectral_dim = self._infer_spectral_dim_from_dataset(class_ids, split=split)
#         if hasattr(self.model, "ensure_class_capacity"):
#             self.model.ensure_class_capacity(max(class_ids) + 1, spectral_dim=spectral_dim)
#         ctx = self.dataset.memory_build_context(phase) if hasattr(self.dataset, "memory_build_context") else nullcontext()
#         with ctx:
#             for cls in class_ids:
#                 if force_rebuild or not self._class_memory_is_valid(cls):
#                     self._build_class_memory_from_current_phase(cls, split=split)

#     @torch.no_grad()
#     def _finalize_phase_memory(self, phase: int, split: str = "train") -> None:
#         phase = int(phase)
#         phase_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         ctx = self.dataset.memory_build_context(phase) if hasattr(self.dataset, "memory_build_context") else nullcontext()
#         with ctx:
#             for cls in phase_ids:
#                 self._build_class_memory_from_current_phase(cls, split=split)
#         if hasattr(self.dataset, "finalize_phase"):
#             self.dataset.finalize_phase(phase)
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "freeze_classes_up_to"):
#             seen = self._seen_class_ids_through_phase(phase)
#             if seen:
#                 self.model.geometry_bank.freeze_classes_up_to(max(seen) + 1)

#     @torch.no_grad()
#     def _refresh_classes_for_validation(self, phase: int, class_ids: Iterable[int], split: str = "train", force_rebuild: bool = True) -> None:
#         if not bool(getattr(self, "refresh_before_validation", getattr(self.args, "refresh_before_validation", True))):
#             return
#         phase = int(phase)
#         class_ids = self._as_class_list(class_ids)
#         old_training_state = bool(self.model.training)
#         old_class_count = int(getattr(self.model, "old_class_count", 0))
#         ctx = self.dataset.memory_build_context(phase) if hasattr(self.dataset, "memory_build_context") else nullcontext()
#         with ctx:
#             for cls in class_ids:
#                 if phase > 0 and int(cls) < old_class_count:
#                     raise RuntimeError(f"Attempted to refresh old class {cls} during incremental phase {phase}.")
#                 if force_rebuild or not self._class_memory_is_valid(cls):
#                     self._build_class_memory_from_current_phase(cls, split=split)
#         self.model.train(old_training_state)

#     def _should_refresh_for_validation(self, epoch: int) -> bool:
#         if not bool(getattr(self, "refresh_before_validation", getattr(self.args, "refresh_before_validation", True))):
#             return False
#         every = int(getattr(self, "validation_refresh_every", getattr(self.args, "validation_refresh_every", 1)))
#         return every > 0 and ((int(epoch) + 1) % every == 0)

#     def _seen_class_ids_before_phase(self, phase: int) -> List[int]:
#         ids: List[int] = []
#         for p in range(max(int(phase), 0)):
#             ids.extend(int(c) for c in self.dataset.phase_to_classes[p])
#         return sorted(set(ids))

#     def _seen_class_ids_through_phase(self, phase: int) -> List[int]:
#         ids: List[int] = []
#         for p in range(max(int(phase), 0) + 1):
#             ids.extend(int(c) for c in self.dataset.phase_to_classes[p])
#         return sorted(set(ids))

#     # ------------------------------------------------------------------
#     # Old-bank snapshot/integrity
#     # ------------------------------------------------------------------
#     def _snapshot_old_bank(self, old_class_count: int) -> Dict[str, torch.Tensor]:
#         old_class_count = int(old_class_count)
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         keep = (
#             "means", "bases", "eigvals", "res_vars", "variances", "active_ranks",
#             "reliability", "feature_reliability", "sample_counts", "band_importances",
#             "band_reliability", "spectral_curve_means", "spectral_curve_vars",
#             "spectral_curve_d1", "spectral_curve_d2", "spectral_shape_reliability",
#         )
#         snap: Dict[str, torch.Tensor] = {}
#         for key in keep:
#             v = bank.get(key, None)
#             if torch.is_tensor(v):
#                 snap[key] = v[:old_class_count].detach().clone() if v.dim() > 0 else v.detach().clone()
#         snap = self._canonicalize_bank(snap)
#         missing = [k for k in ("means", "bases", "variances", "active_ranks", "reliability", "sample_counts") if k not in snap]
#         if missing:
#             raise RuntimeError(f"Old-bank snapshot missing keys: {missing}")
#         return snap

#     @torch.no_grad()
#     def _old_bank_integrity_snapshot(self, old_class_ids: Iterable[int]) -> Dict[str, torch.Tensor]:
#         ids = self._as_class_list(old_class_ids)
#         if not ids:
#             return {}
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         out: Dict[str, torch.Tensor] = {}
#         for key in (
#             "means", "bases", "variances", "active_ranks", "reliability",
#             "sample_counts", "band_importances", "spectral_curve_means",
#             "spectral_curve_vars", "spectral_curve_d1", "spectral_curve_d2",
#             "spectral_shape_reliability",
#         ):
#             v = bank.get(key, None)
#             if torch.is_tensor(v) and v.dim() > 0 and v.size(0) > max(ids):
#                 out[key] = v[ids].detach().clone()
#         return out

#     @torch.no_grad()
#     def _assert_old_bank_integrity(self, old_class_ids: Iterable[int], snapshot: Dict[str, torch.Tensor], *, context: str, atol: float = 1e-6) -> None:
#         ids = self._as_class_list(old_class_ids)
#         if not ids or not snapshot:
#             return
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         bad: List[str] = []
#         for key, old_v in snapshot.items():
#             cur = bank.get(key, None)
#             if not torch.is_tensor(cur) or cur.dim() == 0 or cur.size(0) <= max(ids):
#                 bad.append(f"{key}:missing")
#                 continue
#             cur_v = cur[ids].detach().to(device=old_v.device, dtype=old_v.dtype)
#             if cur_v.shape != old_v.shape or not torch.allclose(cur_v, old_v, atol=float(atol), rtol=0.0):
#                 diff = float((cur_v - old_v).abs().max().item()) if cur_v.shape == old_v.shape else float("inf")
#                 bad.append(f"{key}:maxdiff={diff:.3e}")
#         if bad:
#             raise RuntimeError(f"Old GeometryBank rows changed during {context}. Mutated tensors: {bad[:12]}")

#     def _make_loss_scoring_bank(self, raw_bank: Optional[Dict[str, torch.Tensor]] = None, old_class_count: Optional[int] = None, classifier_mode: Optional[str] = None) -> Dict[str, torch.Tensor]:
#         del old_class_count, classifier_mode
#         return self._canonicalize_bank(raw_bank if raw_bank is not None else self._safe_get_subspace_bank(require_ready=True))

#     # ------------------------------------------------------------------
#     # Energy wrappers and replay diagnostics
#     # ------------------------------------------------------------------
#     def _geometry_energy_matrix(
#         self,
#         features: torch.Tensor,
#         means: torch.Tensor,
#         bases: torch.Tensor,
#         variances: torch.Tensor,
#         active_ranks: Optional[torch.Tensor] = None,
#         reliability: Optional[torch.Tensor] = None,
#         sample_counts: Optional[torch.Tensor] = None,
#         return_parts: bool = False,
#     ) -> torch.Tensor:
#         classifier = getattr(self.model, "classifier", None)
#         if classifier is not None and hasattr(classifier, "geometry_energy"):
#             return classifier.geometry_energy(
#                 features=features,
#                 means=means,
#                 bases=bases,
#                 variances=variances,
#                 active_ranks=active_ranks,
#                 reliability=reliability,
#                 sample_counts=sample_counts,
#                 return_parts=return_parts,
#             )
#         if geometry_energy_matrix is None:
#             raise RuntimeError("No geometry energy implementation available.")
#         return geometry_energy_matrix(
#             features=features,
#             means=means,
#             bases=bases,
#             variances=variances,
#             active_ranks=active_ranks,
#             reliability=reliability,
#             sample_counts=sample_counts,
#             variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
#             reliability_energy_weight=float(getattr(self.args, "reliability_energy_weight", 0.05)),
#             residual_variance_scale=float(getattr(self.args, "residual_variance_scale", 1.0)),
#             normalize_by_dim=bool(getattr(self.args, "energy_normalize_by_dim", True)),
#             invalid_class_energy=float(getattr(self.args, "invalid_class_energy", 1e6)),
#             use_logdet_energy=bool(getattr(self.args, "use_logdet_energy", True)),
#             logdet_energy_weight=float(getattr(self.args, "logdet_energy_weight", getattr(self.args, "geometry_logdet_weight", 0.05))),
#             logdet_normalize_by_dim=bool(getattr(self.args, "logdet_normalize_by_dim", True)),
#             center_logdet_energy=bool(getattr(self.args, "center_logdet_energy", True)),
#             return_parts=return_parts,
#         )

#     def _dual_geometry_energy_matrix(
#         self,
#         features: torch.Tensor,
#         bank: Dict[str, torch.Tensor],
#         spectral_summary: Optional[torch.Tensor] = None,
#         *,
#         spectral_summary_is_physical: Optional[bool] = None,
#         return_parts: bool = False,
#     ) -> torch.Tensor:
#         bank = self._canonicalize_bank(bank)
#         mode = str(getattr(self.args, "eval_classifier_mode", "srgp")).lower().strip()
#         if hasattr(self.model, "compute_energy_from_features"):
#             try:
#                 out = self.model.compute_energy_from_features(
#                     features,
#                     classifier_mode=mode,
#                     spectral_summary=spectral_summary,
#                     spectral_summary_is_physical=bool(spectral_summary_is_physical) if spectral_summary_is_physical is not None else False,
#                     return_parts=return_parts,
#                 )
#                 return out
#             except Exception:
#                 pass
#         if geometry_energy_matrix is None:
#             return self._geometry_energy_matrix(
#                 features=features,
#                 means=bank["means"],
#                 bases=bank["bases"],
#                 variances=bank["variances"],
#                 active_ranks=bank.get("active_ranks", None),
#                 reliability=bank.get("reliability", None),
#                 sample_counts=bank.get("sample_counts", None),
#                 return_parts=return_parts,
#             )
#         return geometry_energy_matrix(
#             features=features,
#             means=bank["means"],
#             bases=bank["bases"],
#             variances=bank["variances"],
#             active_ranks=bank.get("active_ranks", None),
#             reliability=bank.get("reliability", None),
#             sample_counts=bank.get("sample_counts", None),
#             variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
#             reliability_energy_weight=float(getattr(self.args, "reliability_energy_weight", 0.05)),
#             residual_variance_scale=float(getattr(self.args, "residual_variance_scale", 0.75)),
#             normalize_by_dim=bool(getattr(self.args, "energy_normalize_by_dim", True)),
#             invalid_class_energy=float(getattr(self.args, "invalid_class_energy", 1e6)),
#             use_logdet_energy=bool(getattr(self.args, "use_logdet_energy", True)),
#             logdet_energy_weight=float(getattr(self.args, "logdet_energy_weight", getattr(self.args, "geometry_logdet_weight", 0.05))),
#             logdet_normalize_by_dim=bool(getattr(self.args, "logdet_normalize_by_dim", True)),
#             center_logdet_energy=bool(getattr(self.args, "center_logdet_energy", True)),
#             spectral_summary=spectral_summary,
#             spectral_curve_means=bank.get("spectral_curve_means", None),
#             spectral_curve_vars=bank.get("spectral_curve_vars", None),
#             spectral_curve_d1=bank.get("spectral_curve_d1", None),
#             spectral_curve_d2=bank.get("spectral_curve_d2", None),
#             spectral_shape_reliability=bank.get("spectral_shape_reliability", None),
#             use_spectral_residual_energy=bool(getattr(self.args, "use_spectral_geometry", True)) and spectral_summary is not None,
#             spectral_energy_weight=float(getattr(self.args, "spectral_energy_weight", 0.05)),
#             spectral_summary_is_physical=(
#                 bool(spectral_summary_is_physical)
#                 if spectral_summary_is_physical is not None
#                 else self._spectral_summary_is_physical_default(
#                     int(spectral_summary.size(1)) if torch.is_tensor(spectral_summary) and spectral_summary.dim() == 2 else 0,
#                     source="external" if torch.is_tensor(spectral_summary) else "input",
#                 )
#             ),
#             return_parts=return_parts,
#         )

#     def _active_basis(self, bases: torch.Tensor, active_ranks: Optional[torch.Tensor], cls: int) -> torch.Tensor:
#         R = int(bases.size(2))
#         if torch.is_tensor(active_ranks) and active_ranks.numel() > cls:
#             r = max(0, min(int(active_ranks[cls].detach().item()), R))
#         else:
#             r = R
#         return bases[int(cls), :, :r]

#     @torch.no_grad()
#     def _pgr_bank_reserve_metrics(self, bank: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
#         try:
#             bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
#         except Exception:
#             return {}
#         out: Dict[str, float] = {}
#         try:
#             out["pgr_feature_subspace_overlap"] = float(self._global_subspace_overlap_loss(bank).detach().cpu().item())
#             out["pgr_feature_residual_var"] = float(self._bank_residual_variance_loss(bank).detach().cpu().item())
#             out["pgr_feature_rank_usage"] = float(self._bank_active_rank_loss(bank).detach().cpu().item())
#             out["pgr_reserve_score"] = float(1.0 / (1.0 + out["pgr_feature_subspace_overlap"] + 0.25 * out["pgr_feature_residual_var"] + 0.25 * out["pgr_feature_rank_usage"]))
#         except Exception:
#             pass
#         return out

#     def _global_subspace_overlap_loss(self, bank: Optional[Dict[str, torch.Tensor]] = None, *, basis_key: str = "bases", rank_key: str = "active_ranks") -> torch.Tensor:
#         bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
#         bases = bank.get(basis_key, None)
#         if not torch.is_tensor(bases) or bases.numel() == 0:
#             return self._zero()
#         ranks = bank.get(rank_key, None)
#         counts = bank.get("sample_counts", None)
#         vals = []
#         for i in range(bases.size(0)):
#             if torch.is_tensor(counts) and counts.numel() > i and float(counts[i].item()) <= 0:
#                 continue
#             Ui = self._active_basis(bases, ranks, i)
#             if Ui.numel() == 0:
#                 continue
#             for j in range(i + 1, bases.size(0)):
#                 if torch.is_tensor(counts) and counts.numel() > j and float(counts[j].item()) <= 0:
#                     continue
#                 Uj = self._active_basis(bases, ranks, j)
#                 if Uj.numel() == 0:
#                     continue
#                 vals.append((Ui.t() @ Uj).pow(2).mean())
#         return torch.stack(vals).mean() if vals else bases.sum() * 0.0

#     def _bank_residual_variance_loss(self, bank: Optional[Dict[str, torch.Tensor]] = None, *, variance_key: str = "variances") -> torch.Tensor:
#         bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
#         v = bank.get(variance_key, None)
#         if not torch.is_tensor(v) or v.numel() == 0:
#             return self._zero()
#         res = v[:, -1]
#         counts = bank.get("sample_counts", None)
#         if torch.is_tensor(counts) and counts.numel() == res.numel():
#             valid = counts.to(res.device) > 0
#             if bool(valid.any().item()):
#                 res = res[valid]
#         return torch.log1p(res.clamp_min(0.0)).mean()

#     def _bank_active_rank_loss(self, bank: Optional[Dict[str, torch.Tensor]] = None, *, basis_key: str = "bases", rank_key: str = "active_ranks") -> torch.Tensor:
#         bank = self._canonicalize_bank(bank if bank is not None else self._safe_get_subspace_bank(require_ready=True))
#         ranks = bank.get(rank_key, None)
#         bases = bank.get(basis_key, None)
#         if not torch.is_tensor(ranks) or not torch.is_tensor(bases) or ranks.numel() == 0:
#             return self._zero()
#         values = ranks.float()
#         counts = bank.get("sample_counts", None)
#         if torch.is_tensor(counts) and counts.numel() == values.numel():
#             valid = counts.to(values.device) > 0
#             if bool(valid.any().item()):
#                 values = values[valid]
#         return (values / float(max(bases.size(2), 1))).mean()

#     # ------------------------------------------------------------------
#     # Base/incremental geometry certificate helpers
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _geometry_certificate_from_bank(
#         self,
#         *,
#         phase: Optional[int] = None,
#         class_ids: Optional[Iterable[int]] = None,
#         val_stats: Optional[Dict[str, float]] = None,
#     ) -> Dict[str, object]:
#         """Compact certificate consumed by the base/incremental handoff.

#         This does not change training.  It records whether the current bank is
#         safe enough to serve as the frozen old geometry for future phases.
#         """
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         bank = self._canonicalize_bank(bank)
#         valid = self._valid_mask_from_bank(bank)
#         ids = self._as_class_list(class_ids) if class_ids is not None else [i for i in range(bank["means"].size(0)) if bool(valid[i].item())]
#         ids = [int(c) for c in ids if 0 <= int(c) < bank["means"].size(0)]
#         if not ids:
#             return {"ok": False, "reason": "no valid class ids", "phase": int(getattr(self.model, "current_phase", 0) if phase is None else phase)}
#         idx = torch.as_tensor(ids, device=bank["means"].device, dtype=torch.long)
#         valid_sel = valid[idx]
#         rel = bank["reliability"].detach().to(bank["means"].device)[idx]
#         ranks = bank["active_ranks"].detach().to(bank["means"].device)[idx].float()
#         counts = bank["sample_counts"].detach().to(bank["means"].device)[idx]
#         resvars = bank["variances"].detach().to(bank["means"].device)[idx, -1]

#         sub_max = sub_mean = band_max = band_mean = conflict_max = conflict_mean = 0.0
#         gb = getattr(self.model, "geometry_bank", None)
#         try:
#             if gb is not None and hasattr(gb, "pairwise_subspace_overlap"):
#                 sub = gb.pairwise_subspace_overlap().detach()
#                 ss = sub.index_select(0, idx).index_select(1, idx)
#                 mask = ~torch.eye(ss.size(0), device=ss.device, dtype=torch.bool)
#                 vals = ss[mask] if ss.numel() > 1 else torch.empty(0, device=ss.device)
#                 if vals.numel() > 0:
#                     sub_max = float(vals.max().cpu().item())
#                     sub_mean = float(vals.mean().cpu().item())
#             if gb is not None and hasattr(gb, "pairwise_band_similarity"):
#                 bs = gb.pairwise_band_similarity().detach().index_select(0, idx).index_select(1, idx)
#                 mask = ~torch.eye(bs.size(0), device=bs.device, dtype=torch.bool)
#                 vals = bs[mask] if bs.numel() > 1 else torch.empty(0, device=bs.device)
#                 if vals.numel() > 0:
#                     band_max = float(vals.max().cpu().item())
#                     band_mean = float(vals.mean().cpu().item())
#             if gb is not None and hasattr(gb, "geometry_conflict_matrix"):
#                 cm = gb.geometry_conflict_matrix().detach().index_select(0, idx).index_select(1, idx)
#                 mask = ~torch.eye(cm.size(0), device=cm.device, dtype=torch.bool)
#                 vals = cm[mask] if cm.numel() > 1 else torch.empty(0, device=cm.device)
#                 vals = vals[vals > 0]
#                 if vals.numel() > 0:
#                     conflict_max = float(vals.max().cpu().item())
#                     conflict_mean = float(vals.mean().cpu().item())
#         except Exception:
#             pass

#         geom_acc = float((val_stats or {}).get("acc", 0.0))
#         min_rel = float(rel[valid_sel].min().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0
#         mean_rel = float(rel[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0
#         cert = {
#             "phase": int(getattr(self.model, "current_phase", 0) if phase is None else phase),
#             "class_ids": ids,
#             "geom_acc": geom_acc,
#             "valid_rows": int(valid_sel.sum().detach().cpu().item()),
#             "expected_rows": int(len(ids)),
#             "min_reliability": min_rel,
#             "mean_reliability": mean_rel,
#             "mean_active_rank": float(ranks[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0,
#             "mean_sample_count": float(counts[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0,
#             "mean_residual_var": float(resvars[valid_sel].mean().detach().cpu().item()) if bool(valid_sel.any().item()) else 0.0,
#             "max_subspace_overlap": sub_max,
#             "mean_subspace_overlap": sub_mean,
#             "max_band_similarity": band_max,
#             "mean_band_similarity": band_mean,
#             "max_geometry_conflict": conflict_max,
#             "mean_geometry_conflict": conflict_mean,
#         }
#         cert["ok"] = bool(
#             cert["valid_rows"] == cert["expected_rows"]
#             and cert["min_reliability"] >= float(getattr(self.args, "base_cert_min_reliability", 0.15))
#             and cert["mean_reliability"] >= float(getattr(self.args, "base_cert_min_mean_reliability", 0.35))
#             and cert["max_subspace_overlap"] <= float(getattr(self.args, "base_cert_max_subspace_overlap", 0.65))
#             and cert["max_geometry_conflict"] <= float(getattr(self.args, "base_cert_max_geometry_conflict", 2.0))
#         )
#         return cert

#     # ------------------------------------------------------------------
#     # Geometry diagnostics
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _geometry_bank_diagnostics(self, class_ids: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, float]]:
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         ids = self._as_class_list(class_ids) if class_ids is not None else list(range(bank["means"].size(0)))
#         rows: Dict[int, Dict[str, float]] = {}
#         for c in ids:
#             if c < 0 or c >= bank["means"].size(0):
#                 continue
#             rows[int(c)] = {
#                 "count": float(bank["sample_counts"][c].detach().item()),
#                 "active_rank": float(bank["active_ranks"][c].detach().item()),
#                 "reliability": float(bank["reliability"][c].detach().item()),
#                 "residual_var": float(bank["variances"][c, -1].detach().item()),
#                 "mean_norm": float(bank["means"][c].detach().norm().item()),
#             }
#         return rows

#     @torch.no_grad()
#     def _print_base_geometry_diagnostics(self, phase_class_ids: Iterable[int]) -> None:
#         if not bool(getattr(self.args, "print_base_geometry_diagnostics", True)) and not bool(getattr(self, "debug", False)):
#             return
#         try:
#             diag = self._geometry_bank_diagnostics(phase_class_ids)
#         except Exception as exc:
#             print(f"[Base Geometry Diagnostics] unavailable: {exc}")
#             return
#         print("[Base Geometry Diagnostics]")
#         print("  cls | count | rank | rel   | resvar    | mean_norm")
#         for cls in sorted(diag.keys()):
#             d = diag[cls]
#             print(f"  {cls:3d} | {d['count']:5.0f} | {d['active_rank']:4.0f} | {d['reliability']:5.3f} | {d['residual_var']:9.5f} | {d['mean_norm']:9.4f}")
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "geometry_diagnostics"):
#             try:
#                 gd = self.model.geometry_bank.geometry_diagnostics()
#                 keys = ["feature_subspace_overlap", "feature_rank_usage", "band_overlap", "geometry_conflict_mean", "geometry_conflict_max", "geometry_reserve_score"]
#                 msg = []
#                 for k in keys:
#                     v = gd.get(k, None)
#                     if torch.is_tensor(v) and v.numel() == 1:
#                         msg.append(f"{k}={float(v.item()):.4f}")
#                 if msg:
#                     print("  " + " | ".join(msg))
#             except Exception:
#                 pass

#     @torch.no_grad()
#     def _collect_bank_class_stats(self, phase_class_ids: Iterable[int], topk_bands: int = 5) -> List[Dict[str, object]]:
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         ids = self._as_class_list(phase_class_ids)
#         rows: List[Dict[str, object]] = []
#         bands = bank.get("band_importances", None)
#         for c in ids:
#             if c < 0 or c >= bank["means"].size(0):
#                 continue
#             eig = bank["variances"][c, :-1].detach().float().cpu()
#             r = int(bank["active_ranks"][c].detach().item())
#             eig_active = eig[:max(0, min(r, eig.numel()))]
#             row: Dict[str, object] = {
#                 "class_id": int(c),
#                 "class_name": self._class_name(c),
#                 "sample_count": float(bank["sample_counts"][c].detach().item()),
#                 "valid_memory": bool(float(bank["sample_counts"][c].detach().item()) > 0.0),
#                 "feature_active_rank": int(r),
#                 "feature_rank_fraction": float(r / max(bank["bases"].size(2), 1)),
#                 "final_reliability": float(bank["reliability"][c].detach().item()),
#                 "feature_residual_var": float(bank["variances"][c, -1].detach().item()),
#                 "feature_mean_norm": float(bank["means"][c].detach().norm().item()),
#                 "spectral_shape_reliability": float(bank.get("spectral_shape_reliability", torch.zeros_like(bank["reliability"]))[c].detach().item()) if torch.is_tensor(bank.get("spectral_shape_reliability", None)) and bank.get("spectral_shape_reliability").numel() > c else 0.0,
#                 "feature_eig_min": float(eig_active.min().item()) if eig_active.numel() else 0.0,
#                 "feature_eig_max": float(eig_active.max().item()) if eig_active.numel() else 0.0,
#                 "feature_eig_ratio": float(eig_active.max().item() / max(eig_active.min().item(), 1e-12)) if eig_active.numel() else 0.0,
#             }
#             if torch.is_tensor(bands) and bands.numel() > 0 and c < bands.size(0) and bands.size(1) > 0:
#                 b = bands[c].detach().float().cpu().clamp_min(0.0)
#                 b = b / b.sum().clamp_min(1e-12)
#                 k = min(int(topk_bands), int(b.numel()))
#                 vals, idx = torch.topk(b, k=k)
#                 row.update({
#                     "band_entropy": float((-(b * b.clamp_min(1e-12).log()).sum()).item()),
#                     "band_max_weight": float(b.max().item()),
#                     "band_top_indices": [int(i.item()) for i in idx],
#                     "band_top_values": [float(v.item()) for v in vals],
#                 })
#             else:
#                 row.update({"band_entropy": -1.0, "band_max_weight": -1.0, "band_top_indices": [], "band_top_values": []})
#             rows.append(row)
#         return rows

#     @torch.no_grad()
#     def _subspace_pair_risks(self, phase_class_ids: Iterable[int], top_k: int = 20) -> List[Dict[str, object]]:
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         ids = self._as_class_list(phase_class_ids)
#         means, bases, ranks = bank["means"], bank["bases"], bank.get("active_ranks", None)
#         bands = bank.get("band_importances", None)
#         rows: List[Dict[str, object]] = []
#         for ii, ci in enumerate(ids):
#             if ci >= means.size(0):
#                 continue
#             Ui = self._active_basis(bases, ranks, ci)
#             for cj in ids[ii + 1:]:
#                 if cj >= means.size(0):
#                     continue
#                 Uj = self._active_basis(bases, ranks, cj)
#                 if Ui.numel() > 0 and Uj.numel() > 0:
#                     f_ov = float((Ui.t() @ Uj).pow(2).sum().div(max(min(Ui.size(1), Uj.size(1)), 1)).detach().cpu().item())
#                 else:
#                     f_ov = 0.0
#                 f_dist = float(torch.dist(means[ci], means[cj], p=2).detach().cpu().item())
#                 b_sim = 0.0
#                 if torch.is_tensor(bands) and bands.numel() > 0 and ci < bands.size(0) and cj < bands.size(0):
#                     bi = F.normalize(bands[ci].clamp_min(0.0), dim=0)
#                     bj = F.normalize(bands[cj].clamp_min(0.0), dim=0)
#                     b_sim = float(torch.dot(bi, bj).clamp(0, 1).detach().cpu().item())
#                 risk = float(f_ov + 0.25 * b_sim + 1.0 / (1.0 + f_dist))
#                 rows.append({
#                     "class_i": int(ci), "class_j": int(cj),
#                     "name_i": self._class_name(ci), "name_j": self._class_name(cj),
#                     "feature_overlap": f_ov, "raw_feature_overlap": f_ov,
#                     "band_similarity": b_sim,
#                     "feature_center_distance": f_dist,
#                     "risk_score": risk,
#                 })
#         rows.sort(key=lambda r: float(r["risk_score"]), reverse=True)
#         return rows[:int(top_k)]

#     @torch.no_grad()
#     def _old_new_pair_risks(
#         self,
#         old_class_ids: Iterable[int],
#         new_class_ids: Iterable[int],
#         top_k: int = 20,
#     ) -> List[Dict[str, object]]:
#         """Rank old/new SRGP descriptor conflicts for incremental diagnostics."""
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         bank = self._canonicalize_bank(bank)
#         old_ids = self._as_class_list(old_class_ids)
#         new_ids = self._as_class_list(new_class_ids)
#         if not old_ids or not new_ids:
#             return []

#         means, bases, ranks = bank["means"], bank["bases"], bank.get("active_ranks", None)
#         counts = bank.get("sample_counts", None)
#         rel = bank.get("reliability", None)
#         bands = bank.get("band_importances", bank.get("band_importance", None))
#         d1 = bank.get("spectral_curve_d1", None)
#         rows: List[Dict[str, object]] = []
#         dscale = max(float(means.size(1)) ** 0.5, 1.0)
#         center_w = self._overlap_cfg_float("old_new_risk_center_weight", 0.40)
#         subspace_w = self._overlap_cfg_float("old_new_risk_subspace_weight", 0.35)
#         band_w = self._overlap_cfg_float("old_new_risk_band_weight", 0.10)
#         spec_w = self._overlap_cfg_float("old_new_risk_spectral_shape_weight", 0.15)

#         for oi in old_ids:
#             if oi < 0 or oi >= means.size(0):
#                 continue
#             if torch.is_tensor(counts) and counts.numel() > oi and float(counts[oi].detach().item()) <= 0.0:
#                 continue
#             Ui = self._active_basis(bases, ranks, oi)
#             for nj in new_ids:
#                 if nj < 0 or nj >= means.size(0):
#                     continue
#                 if torch.is_tensor(counts) and counts.numel() > nj and float(counts[nj].detach().item()) <= 0.0:
#                     continue
#                 Uj = self._active_basis(bases, ranks, nj)
#                 dist = torch.norm(means[oi] - means[nj], p=2)
#                 scaled = dist / dscale
#                 center_risk = torch.exp(-scaled).clamp(0.0, 1.0)
#                 if Ui.numel() > 0 and Uj.numel() > 0:
#                     denom = float(max(min(Ui.size(1), Uj.size(1)), 1))
#                     f_ov_t = (Ui.t() @ Uj).pow(2).sum() / denom
#                     f_ov_t = f_ov_t.clamp(0.0, 1.0)
#                 else:
#                     f_ov_t = torch.tensor(0.0, device=means.device, dtype=means.dtype)

#                 b_sim_t = torch.tensor(0.0, device=means.device, dtype=means.dtype)
#                 if torch.is_tensor(bands) and bands.dim() == 2 and bands.numel() > 0 and oi < bands.size(0) and nj < bands.size(0):
#                     bi = bands[oi].clamp_min(0.0)
#                     bj = bands[nj].clamp_min(0.0)
#                     if bi.numel() == bj.numel() and bi.sum() > 1e-8 and bj.sum() > 1e-8:
#                         bi = bi / bi.norm().clamp_min(1e-8)
#                         bj = bj / bj.norm().clamp_min(1e-8)
#                         b_sim_t = torch.dot(bi, bj).clamp(0.0, 1.0)

#                 spec_sim_t = torch.tensor(0.0, device=means.device, dtype=means.dtype)
#                 if torch.is_tensor(d1) and d1.dim() == 2 and d1.size(0) > max(oi, nj) and d1.size(1) > 0:
#                     oi_d = F.normalize(torch.nan_to_num(d1[oi], nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()
#                     nj_d = F.normalize(torch.nan_to_num(d1[nj], nan=0.0, posinf=0.0, neginf=0.0).view(1, -1), dim=1, eps=1e-8).flatten()
#                     spec_sim_t = torch.dot(oi_d, nj_d).clamp(0.0, 1.0)

#                 risk_t = center_w * center_risk + subspace_w * f_ov_t + band_w * b_sim_t + spec_w * spec_sim_t
#                 if torch.is_tensor(rel) and rel.numel() > oi:
#                     uncertainty = (1.0 - rel[oi].clamp(0.05, 1.0)).clamp(0.0, 1.0)
#                     risk_t = risk_t * (1.0 + 0.25 * uncertainty)
#                 risk = float(risk_t.detach().cpu().item())
#                 rows.append(
#                     {
#                         "old_class": int(oi),
#                         "new_class": int(nj),
#                         "old_name": self._class_name(oi),
#                         "new_name": self._class_name(nj),
#                         "feature_overlap": float(f_ov_t.detach().cpu().item()),
#                         "band_similarity": float(b_sim_t.detach().cpu().item()),
#                         "spectral_shape_similarity": float(spec_sim_t.detach().cpu().item()),
#                         "feature_center_distance": float(dist.detach().cpu().item()),
#                         "scaled_center_distance": float(scaled.detach().cpu().item()),
#                         "center_risk": float(center_risk.detach().cpu().item()),
#                         "risk_score": risk,
#                     }
#                 )
#         rows.sort(key=lambda r: float(r["risk_score"]), reverse=True)
#         return rows[: int(top_k)]

#     @torch.no_grad()
#     def _phase_old_new_pair_risks(self, phase: Optional[int] = None, top_k: int = 20) -> List[Dict[str, object]]:
#         phase = int(getattr(self.model, "current_phase", 0) if phase is None else phase)
#         if phase <= 0 or not hasattr(self.dataset, "phase_to_classes"):
#             return []
#         old_ids = self._seen_class_ids_before_phase(phase)
#         new_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         return self._old_new_pair_risks(old_ids, new_ids, top_k=top_k)

#     @torch.no_grad()
#     def _phase_overlap_summary(self, phase: Optional[int] = None, top_k: int = 20) -> Dict[str, object]:
#         pairs = self._phase_old_new_pair_risks(phase=phase, top_k=top_k)
#         if not pairs:
#             return {"num_pairs": 0, "max_risk": 0.0, "mean_risk": 0.0, "top_pair": None}
#         risks = [float(p["risk_score"]) for p in pairs]
#         return {
#             "num_pairs": int(len(pairs)),
#             "max_risk": float(max(risks)),
#             "mean_risk": float(sum(risks) / max(len(risks), 1)),
#             "top_pair": pairs[0],
#         }

#     @torch.no_grad()
#     def _energy_margin_health(self, loader, phase_class_ids: Iterable[int]) -> Dict[str, object]:
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         ids = self._as_class_list(phase_class_ids)
#         id_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
#         stats = {c: {"n": 0, "correct": 0, "viol": 0, "margin_sum": 0.0, "margin_min": float("inf")} for c in ids}
#         was_training = bool(self.model.training)
#         self.model.eval()
#         for batch in loader:
#             x, y, spectra, _ = self._unpack_hsi_batch(batch)
#             x = x.to(self.device, non_blocking=True).float()
#             y = y.to(self.device, non_blocking=True).long().view(-1)
#             spectral_summary, spec_is_physical = self._resolve_batch_spectral_summary(
#                 x, spectra=spectra, source="batch_metadata" if torch.is_tensor(spectra) and spectra.numel() > 0 else "input"
#             )
#             try:
#                 out = self.model.extract_projected_features(
#                     x,
#                     spectral_summary=spectral_summary,
#                     spectral_summary_is_physical=bool(spec_is_physical),
#                 )
#             except TypeError:
#                 out = self.model.extract_projected_features(x)
#             features = out["features"]
#             ss, ss_phys = self._resolve_batch_spectral_summary(
#                 x, spectra=spectra, model_out=out, source="batch_metadata" if torch.is_tensor(spectra) and spectra.numel() > 0 else "input", spectral_summary_is_physical=spec_is_physical
#             )
#             energy = self._dual_geometry_energy_matrix(
#                 features=features,
#                 bank=bank,
#                 spectral_summary=ss,
#                 spectral_summary_is_physical=bool(ss_phys),
#                 return_parts=False,
#             )
#             e_sel = energy.index_select(1, id_tensor)
#             pred = id_tensor[e_sel.argmin(dim=1)]
#             y_local = torch.full_like(y, -1)
#             for li, c in enumerate(ids):
#                 y_local[y == int(c)] = int(li)
#             valid = y_local >= 0
#             if not bool(valid.any().item()):
#                 continue
#             e_valid, yv, ylv, predv = e_sel[valid], y[valid], y_local[valid], pred[valid]
#             true_e = e_valid.gather(1, ylv.view(-1, 1)).squeeze(1)
#             mask = torch.zeros_like(e_valid, dtype=torch.bool).scatter(1, ylv.view(-1, 1), True)
#             nearest_wrong = e_valid.masked_fill(mask, float("inf")).min(dim=1).values
#             margin = nearest_wrong - true_e
#             for c in ids:
#                 m = yv == int(c)
#                 if not bool(m.any().item()):
#                     continue
#                 s = stats[int(c)]
#                 mg = margin[m]
#                 s["n"] += int(m.sum().item())
#                 s["correct"] += int((predv[m] == yv[m]).sum().item())
#                 s["viol"] += int((mg <= 0).sum().item())
#                 s["margin_sum"] += float(mg.sum().detach().cpu().item())
#                 s["margin_min"] = min(s["margin_min"], float(mg.min().detach().cpu().item()))
#         self.model.train(was_training)
#         rows = []
#         total_n = total_correct = total_viol = 0
#         total_margin = 0.0
#         min_margin = float("inf")
#         for c in ids:
#             s = stats[int(c)]
#             n = max(int(s["n"]), 1)
#             rows.append({
#                 "class_id": int(c), "class_name": self._class_name(c), "n": int(s["n"]),
#                 "accuracy": 100.0 * float(s["correct"]) / n,
#                 "mean_margin": float(s["margin_sum"]) / n,
#                 "min_margin": 0.0 if s["margin_min"] == float("inf") else float(s["margin_min"]),
#                 "violation_rate": 100.0 * float(s["viol"]) / n,
#             })
#             total_n += int(s["n"]); total_correct += int(s["correct"]); total_viol += int(s["viol"]); total_margin += float(s["margin_sum"])
#             if s["margin_min"] != float("inf"):
#                 min_margin = min(min_margin, float(s["margin_min"]))
#         denom = max(total_n, 1)
#         return {"overall": {"n": total_n, "accuracy": 100.0 * total_correct / denom, "mean_margin": total_margin / denom, "min_margin": 0.0 if min_margin == float("inf") else min_margin, "violation_rate": 100.0 * total_viol / denom}, "per_class": rows}

#     @torch.no_grad()
#     def _sample_bank_anchors_for_diagnostics(self, class_ids: Iterable[int], samples_per_class: int = 64, parallel_scale: float = 1.0, residual_scale: float = 0.30) -> Tuple[torch.Tensor, torch.Tensor]:
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         ids = self._as_class_list(class_ids)
#         gb = getattr(self.model, "geometry_bank", None)
#         if gb is not None and hasattr(gb, "sample_synthetic_features"):
#             try:
#                 return gb.sample_synthetic_features(
#                     ids,
#                     samples_per_class=int(samples_per_class),
#                     parallel_scale=float(parallel_scale),
#                     residual_scale=float(residual_scale),
#                     reliability_gated=bool(getattr(self.args, "diagnostic_reliability_gated_anchors", True)),
#                 )
#             except TypeError:
#                 return gb.sample_synthetic_features(ids, samples_per_class=int(samples_per_class), parallel_scale=float(parallel_scale), residual_scale=float(residual_scale))
#         means, bases, variances = bank["means"], bank["bases"], bank["variances"]
#         xs, ys = [], []
#         for c in ids:
#             if c < 0 or c >= means.size(0) or float(bank["sample_counts"][c].item()) <= 0:
#                 continue
#             n = int(max(samples_per_class, 1)); r = int(bank["active_ranks"][c].item())
#             eps = torch.zeros((n, means.size(1)), device=self.device, dtype=means.dtype)
#             if r > 0:
#                 z = torch.randn((n, r), device=self.device, dtype=means.dtype)
#                 eps += (z * variances[c, :r].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4))).sqrt()) @ bases[c, :, :r].t()
#             eps += torch.randn_like(eps) * variances[c, -1].clamp_min(float(getattr(self.args, "geom_var_floor", 1e-4))).sqrt() * float(residual_scale)
#             xs.append(means[c].unsqueeze(0) + eps)
#             ys.append(torch.full((n,), int(c), device=self.device, dtype=torch.long))
#         if not xs:
#             return torch.empty((0, int(getattr(self.args, "d_model", 0))), device=self.device), torch.empty((0,), device=self.device, dtype=torch.long)
#         return torch.cat(xs, 0), torch.cat(ys, 0)

#     @torch.no_grad()
#     def _anchor_replay_health(self, phase_class_ids: Iterable[int], samples_per_class: int = 64) -> Dict[str, object]:
#         ids = self._as_class_list(phase_class_ids)
#         x, y = self._sample_bank_anchors_for_diagnostics(ids, samples_per_class=samples_per_class, parallel_scale=float(getattr(self.args, "gfa_parallel_scale", 1.0)), residual_scale=float(getattr(self.args, "gfa_residual_scale", 0.25)))
#         if x.numel() == 0:
#             return {"overall": {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "min_margin": 0.0, "violation_rate": 100.0}, "per_class": []}
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         id_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
#         energy = self._dual_geometry_energy_matrix(x, bank, return_parts=False)
#         e_sel = energy.index_select(1, id_tensor)
#         y_local = torch.full_like(y, -1)
#         for li, c in enumerate(ids):
#             y_local[y == int(c)] = int(li)
#         pred = id_tensor[e_sel.argmin(dim=1)]
#         true_e = e_sel.gather(1, y_local.view(-1, 1)).squeeze(1)
#         label_mask = torch.zeros_like(e_sel, dtype=torch.bool).scatter(1, y_local.view(-1, 1), True)
#         nearest_wrong = e_sel.masked_fill(label_mask, float("inf")).min(dim=1).values
#         margin = nearest_wrong - true_e
#         rows = []
#         for c in ids:
#             m = y == int(c)
#             if not bool(m.any().item()):
#                 continue
#             n = int(m.sum().item())
#             rows.append({
#                 "class_id": int(c), "class_name": self._class_name(c), "n": n,
#                 "anchor_accuracy": 100.0 * float((pred[m] == y[m]).sum().item()) / max(n, 1),
#                 "anchor_mean_margin": float(margin[m].mean().item()),
#                 "anchor_min_margin": float(margin[m].min().item()),
#                 "anchor_violation_rate": 100.0 * float((margin[m] <= 0).sum().item()) / max(n, 1),
#             })
#         total_n = int(y.numel())
#         return {"overall": {"n": total_n, "accuracy": 100.0 * int((pred == y).sum().item()) / max(total_n, 1), "mean_margin": float(margin.mean().item()), "min_margin": float(margin.min().item()), "violation_rate": 100.0 * int((margin <= 0).sum().item()) / max(total_n, 1)}, "per_class": rows}

#     @torch.no_grad()
#     def _sample_boundary_anchors_for_diagnostics(
#         self,
#         old_class_ids: Iterable[int],
#         new_class_ids: Iterable[int],
#         *,
#         samples_per_pair: int = 12,
#         top_k: int = 20,
#     ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
#         """Sample SCB-GR old-boundary anchors for diagnostics only."""
#         old_ids = self._as_class_list(old_class_ids)
#         new_ids = self._as_class_list(new_class_ids)
#         if not old_ids or not new_ids or sample_boundary_geometry_features is None:
#             return torch.empty(0, 0, device=self.device), torch.empty(0, dtype=torch.long, device=self.device), {"pair_count": 0, "boundary_anchor_count": 0}
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         bank = self._canonicalize_bank(bank)
#         max_id = max(old_ids + new_ids)
#         if max_id >= bank["means"].size(0):
#             return torch.empty(0, bank["means"].size(1), device=self.device), torch.empty(0, dtype=torch.long, device=self.device), {"pair_count": 0, "boundary_anchor_count": 0}

#         old_t = torch.as_tensor(old_ids, device=bank["means"].device, dtype=torch.long)
#         new_t = torch.as_tensor(new_ids, device=bank["means"].device, dtype=torch.long)
#         pairs_global = self._old_new_pair_risks(old_ids, new_ids, top_k=top_k)
#         old_pos = {c: i for i, c in enumerate(old_ids)}
#         new_pos = {c: i for i, c in enumerate(new_ids)}
#         pairs_local = []
#         for p in pairs_global:
#             o = int(p.get("old_class", -1)); n = int(p.get("new_class", -1))
#             if o in old_pos and n in new_pos:
#                 pairs_local.append((old_pos[o], new_pos[n]))
#         if not pairs_local:
#             pairs_local = [(i, j) for i in range(len(old_ids)) for j in range(len(new_ids))]

#         x, y, meta = sample_boundary_geometry_features(
#             bank["means"].index_select(0, old_t),
#             bank["bases"].index_select(0, old_t),
#             bank["variances"].index_select(0, old_t),
#             new_means=bank["means"].index_select(0, new_t),
#             new_bases=bank["bases"].index_select(0, new_t),
#             risk_pairs=pairs_local,
#             old_active_ranks=bank.get("active_ranks", None).index_select(0, old_t) if torch.is_tensor(bank.get("active_ranks", None)) else None,
#             old_reliability=bank.get("reliability", None).index_select(0, old_t) if torch.is_tensor(bank.get("reliability", None)) else None,
#             old_sample_counts=bank.get("sample_counts", None).index_select(0, old_t) if torch.is_tensor(bank.get("sample_counts", None)) else None,
#             old_class_ids=old_ids,
#             samples_per_pair=int(samples_per_pair),
#             variance_floor=float(getattr(self.args, "geom_var_floor", 1e-4)),
#             parallel_scale=float(getattr(self.args, "boundary_replay_parallel_scale", 0.15)),
#             residual_scale=float(getattr(self.args, "boundary_replay_residual_scale", 0.05)),
#             return_metadata=True,
#         )
#         return x.to(self.device), y.to(self.device).long(), self._json_safe(meta)

#     @torch.no_grad()
#     def _boundary_anchor_replay_health(self, phase: Optional[int] = None, samples_per_pair: int = 12, top_k: int = 20) -> Dict[str, object]:
#         phase = int(getattr(self.model, "current_phase", 0) if phase is None else phase)
#         if phase <= 0:
#             return {"overall": {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "min_margin": 0.0, "violation_rate": 0.0}, "per_class": [], "meta": {"pair_count": 0, "boundary_anchor_count": 0}}
#         old_ids = self._seen_class_ids_before_phase(phase)
#         new_ids = [int(c) for c in self.dataset.phase_to_classes[phase]]
#         x, y, meta = self._sample_boundary_anchors_for_diagnostics(old_ids, new_ids, samples_per_pair=samples_per_pair, top_k=top_k)
#         if x.numel() == 0:
#             return {"overall": {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "min_margin": 0.0, "violation_rate": 100.0}, "per_class": [], "meta": meta}
#         bank = self._safe_get_subspace_bank(require_ready=True)
#         seen_ids = sorted(set(old_ids + new_ids))
#         id_tensor = torch.tensor(seen_ids, device=self.device, dtype=torch.long)
#         energy = self._dual_geometry_energy_matrix(x, bank, return_parts=False)
#         e_sel = energy.index_select(1, id_tensor)
#         y_local = torch.full_like(y, -1)
#         for li, c in enumerate(seen_ids):
#             y_local[y == int(c)] = int(li)
#         valid = y_local >= 0
#         if not bool(valid.any().item()):
#             return {"overall": {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "min_margin": 0.0, "violation_rate": 100.0}, "per_class": [], "meta": meta}
#         e_sel = e_sel[valid]; yv = y[valid]; ylv = y_local[valid]
#         pred = id_tensor[e_sel.argmin(dim=1)]
#         true_e = e_sel.gather(1, ylv.view(-1, 1)).squeeze(1)
#         label_mask = torch.zeros_like(e_sel, dtype=torch.bool).scatter(1, ylv.view(-1, 1), True)
#         nearest_wrong = e_sel.masked_fill(label_mask, float("inf")).min(dim=1).values
#         margin = nearest_wrong - true_e
#         rows = []
#         for c in old_ids:
#             m = yv == int(c)
#             if not bool(m.any().item()):
#                 continue
#             n = int(m.sum().item())
#             rows.append({
#                 "class_id": int(c), "class_name": self._class_name(c), "n": n,
#                 "boundary_anchor_accuracy": 100.0 * float((pred[m] == yv[m]).sum().item()) / max(n, 1),
#                 "boundary_anchor_mean_margin": float(margin[m].mean().item()),
#                 "boundary_anchor_min_margin": float(margin[m].min().item()),
#                 "boundary_anchor_violation_rate": 100.0 * float((margin[m] <= 0).sum().item()) / max(n, 1),
#             })
#         total_n = int(yv.numel())
#         return {
#             "overall": {
#                 "n": total_n,
#                 "accuracy": 100.0 * int((pred == yv).sum().item()) / max(total_n, 1),
#                 "mean_margin": float(margin.mean().item()),
#                 "min_margin": float(margin.min().item()),
#                 "violation_rate": 100.0 * int((margin <= 0).sum().item()) / max(total_n, 1),
#             },
#             "per_class": rows,
#             "meta": meta,
#         }

#     @torch.no_grad()
#     def diagnose_full_base_geometry(self, loader, phase_class_ids: Iterable[int], anchors_per_class: int = 64, topk_pairs: int = 20, topk_bands: int = 5) -> Dict[str, object]:
#         ids = self._as_class_list(phase_class_ids)
#         bank_internal = None
#         if hasattr(self.model, "geometry_bank") and hasattr(self.model.geometry_bank, "geometry_health_summary"):
#             try:
#                 names = [self._class_name(i) for i in range(max(ids) + 1)] if ids else []
#                 bank_internal = self.model.geometry_bank.geometry_health_summary(class_names=names, topk_bands=topk_bands)
#             except Exception as exc:
#                 bank_internal = {"error": str(exc)}
#         phase = int(getattr(self.model, "current_phase", 0))
#         old_new_pairs = self._phase_old_new_pair_risks(phase=phase, top_k=topk_pairs)
#         report = {
#             "class_ids": ids,
#             "phase": phase,
#             "bank_internal": bank_internal,
#             "class_geometry": self._collect_bank_class_stats(ids, topk_bands=topk_bands),
#             "energy_margin": self._energy_margin_health(loader, ids),
#             "subspace_risk_pairs": self._subspace_pair_risks(ids, top_k=topk_pairs),
#             "old_new_risk_pairs": old_new_pairs,
#             "old_new_overlap_summary": self._phase_overlap_summary(phase=phase, top_k=topk_pairs),
#             "overlap_admission_events": list(getattr(self, "_last_overlap_admission_events", [])),
#             "anchor_replay": self._anchor_replay_health(ids, samples_per_class=anchors_per_class),
#             "boundary_anchor_replay": self._boundary_anchor_replay_health(phase=phase, samples_per_pair=int(getattr(self.args, "boundary_replay_samples_per_pair", 12)), top_k=topk_pairs),
#         }
#         alerts = []
#         em = report["energy_margin"]["overall"]
#         ar = report["anchor_replay"]["overall"]
#         br = report.get("boundary_anchor_replay", {}).get("overall", {})
#         if float(em["violation_rate"]) > 5.0:
#             alerts.append(f"High validation energy violation rate: {em['violation_rate']:.2f}%")
#         if float(ar["accuracy"]) < 95.0:
#             alerts.append(f"Low anchor self-accuracy: {ar['accuracy']:.2f}%")
#         if phase > 0 and float(br.get("n", 0)) > 0 and float(br.get("violation_rate", 0.0)) > 10.0:
#             alerts.append(f"High SCB-GR boundary-anchor violation rate: {br['violation_rate']:.2f}%")
#         pairs = report["subspace_risk_pairs"]
#         if pairs and float(pairs[0]["feature_overlap"]) > 0.70:
#             alerts.append(f"High feature subspace overlap: {pairs[0]['name_i']} vs {pairs[0]['name_j']} = {pairs[0]['feature_overlap']:.4f}")
#         old_new_pairs = report.get("old_new_risk_pairs", [])
#         old_new_risk_alert = float(getattr(self.args, "old_new_overlap_risk_alert", 0.85))
#         old_new_subspace_alert = float(getattr(self.args, "old_new_subspace_overlap_alert", 0.60))
#         if old_new_pairs:
#             top = old_new_pairs[0]
#             if float(top.get("risk_score", 0.0)) >= old_new_risk_alert:
#                 alerts.append(
#                     f"High old/new geometry conflict: old {top['old_name']} vs new {top['new_name']} "
#                     f"risk={float(top['risk_score']):.4f}"
#                 )
#             if float(top.get("feature_overlap", 0.0)) >= old_new_subspace_alert:
#                 alerts.append(
#                     f"High old/new subspace overlap: old {top['old_name']} vs new {top['new_name']} "
#                     f"overlap={float(top['feature_overlap']):.4f}"
#                 )
#         report["alerts"] = alerts
#         return self._json_safe(report)

#     def _write_csv_rows(self, path: str, rows: List[Dict[str, object]]) -> None:
#         if not rows:
#             return
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         keys: List[str] = []
#         for row in rows:
#             for k in row.keys():
#                 if k not in keys:
#                     keys.append(k)
#         with open(path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=keys)
#             writer.writeheader()
#             for row in rows:
#                 writer.writerow({k: json.dumps(row.get(k, "")) if isinstance(row.get(k, ""), (list, dict)) else row.get(k, "") for k in keys})

#     def _save_geometry_diagnostics_to_files(self, report: Dict[str, object], phase: int = 0, output_dir: Optional[str] = None) -> Dict[str, str]:
#         phase = int(phase)
#         if output_dir is None:
#             root = getattr(self.args, "run_dir", getattr(self, "save_dir", "."))
#             output_dir = os.path.join(root, f"phase_{phase}")
#         os.makedirs(output_dir, exist_ok=True)
#         paths = {
#             "json": os.path.join(output_dir, f"phase_{phase}_geometry_diagnostics.json"),
#             "class_csv": os.path.join(output_dir, f"phase_{phase}_geometry_class_stats.csv"),
#             "energy_csv": os.path.join(output_dir, f"phase_{phase}_geometry_energy_margins.csv"),
#             "subspace_csv": os.path.join(output_dir, f"phase_{phase}_geometry_subspace_pairs.csv"),
#             "old_new_csv": os.path.join(output_dir, f"phase_{phase}_geometry_old_new_pairs.csv"),
#             "anchor_csv": os.path.join(output_dir, f"phase_{phase}_geometry_anchor_stats.csv"),
#             "boundary_anchor_csv": os.path.join(output_dir, f"phase_{phase}_geometry_boundary_anchor_stats.csv"),
#             "txt": os.path.join(output_dir, f"phase_{phase}_geometry_diagnostics.txt"),
#         }
#         with open(paths["json"], "w", encoding="utf-8") as f:
#             json.dump(self._json_safe(report), f, indent=2)
#         self._write_csv_rows(paths["class_csv"], report.get("class_geometry", []))
#         self._write_csv_rows(paths["energy_csv"], report.get("energy_margin", {}).get("per_class", []))
#         self._write_csv_rows(paths["subspace_csv"], report.get("subspace_risk_pairs", []))
#         self._write_csv_rows(paths["old_new_csv"], report.get("old_new_risk_pairs", []))
#         self._write_csv_rows(paths["anchor_csv"], report.get("anchor_replay", {}).get("per_class", []))
#         self._write_csv_rows(paths["boundary_anchor_csv"], report.get("boundary_anchor_replay", {}).get("per_class", []))
#         with open(paths["txt"], "w", encoding="utf-8") as f:
#             f.write(self._format_geometry_diagnostics_text(report))
#         return paths

#     def _format_geometry_diagnostics_text(self, report: Dict[str, object]) -> str:
#         lines = ["Geometry Diagnostics", "=" * 90, "", "Alerts", "-" * 90]
#         alerts = report.get("alerts", [])
#         if alerts:
#             lines.extend(f"[WARN] {a}" for a in alerts)
#         else:
#             lines.append("No major geometry alarms triggered by current thresholds.")
#         lines += ["", "Class Geometry", "-" * 90, "cls name                 n     rank rel    resvar    band-H  band-max"]
#         for r in report.get("class_geometry", []):
#             lines.append(
#                 f"{int(r['class_id']):3d} {str(r['class_name'])[:20]:20s} "
#                 f"{float(r['sample_count']):5.0f} {int(r['feature_active_rank']):5d} "
#                 f"{float(r['final_reliability']):5.3f} {float(r['feature_residual_var']):9.5f} "
#                 f"{float(r['band_entropy']):7.3f} {float(r['band_max_weight']):8.4f}"
#             )
#         ov = report.get("energy_margin", {}).get("overall", {})
#         lines += ["", "Energy Margin Health", "-" * 90]
#         lines.append(f"Overall: acc={float(ov.get('accuracy', 0.0)):.2f}% | mean_margin={float(ov.get('mean_margin', 0.0)):.6f} | min_margin={float(ov.get('min_margin', 0.0)):.6f} | viol={float(ov.get('violation_rate', 0.0)):.2f}%")
#         lines.append("cls name                 n     acc     mean_margin   min_margin    viol")
#         for r in report.get("energy_margin", {}).get("per_class", []):
#             lines.append(f"{int(r['class_id']):3d} {str(r['class_name'])[:20]:20s} {int(r['n']):5d} {float(r['accuracy']):7.2f} {float(r['mean_margin']):13.6f} {float(r['min_margin']):12.6f} {float(r['violation_rate']):7.2f}%")
#         lines += ["", "Top Geometry Risk Pairs", "-" * 90, "pair                                      z-overlap band-sim z-dist    risk"]
#         for r in report.get("subspace_risk_pairs", [])[:15]:
#             pair = f"{r['name_i']} / {r['name_j']}"
#             lines.append(f"{pair[:40]:40s} {float(r['feature_overlap']):9.4f} {float(r.get('band_similarity', 0.0)):8.4f} {float(r['feature_center_distance']):8.4f} {float(r['risk_score']):8.4f}")
#         old_new = report.get("old_new_risk_pairs", [])
#         if old_new:
#             lines += ["", "Top Old/New SRGP Conflicts", "-" * 90, "old -> new                                z-overlap band-sim spec-sim z-dist    risk"]
#             for r in old_new[:15]:
#                 pair = f"{r['old_name']} -> {r['new_name']}"
#                 lines.append(
#                     f"{pair[:40]:40s} {float(r.get('feature_overlap', 0.0)):9.4f} "
#                     f"{float(r.get('band_similarity', 0.0)):8.4f} "
#                     f"{float(r.get('spectral_shape_similarity', 0.0)):8.4f} "
#                     f"{float(r.get('feature_center_distance', 0.0)):8.4f} {float(r.get('risk_score', 0.0)):8.4f}"
#                 )
#         events = report.get("overlap_admission_events", [])
#         if events:
#             lines += ["", "Overlap-Aware Admission Events", "-" * 90, "new class                                risk     gate    top old"]
#             for e in events[-15:]:
#                 top = e.get("top_old", {}) if isinstance(e.get("top_old", {}), dict) else {}
#                 lines.append(
#                     f"{str(e.get('new_name', e.get('new_class', 'NA')))[:40]:40s} "
#                     f"{float(e.get('max_old_overlap_risk', 0.0)):8.4f} "
#                     f"{float(e.get('admission_gate', 1.0)):7.3f} "
#                     f"{str(top.get('old_name', top.get('old_class', 'NA')))[:20]:20s}"
#                 )
#         av = report.get("anchor_replay", {}).get("overall", {})
#         lines += ["", "Anchor Replay Health", "-" * 90]
#         lines.append(f"Overall: acc={float(av.get('accuracy', 0.0)):.2f}% | mean_margin={float(av.get('mean_margin', 0.0)):.6f} | min_margin={float(av.get('min_margin', 0.0)):.6f} | viol={float(av.get('violation_rate', 0.0)):.2f}%")
#         bv = report.get("boundary_anchor_replay", {}).get("overall", {})
#         bm = report.get("boundary_anchor_replay", {}).get("meta", {})
#         lines += ["", "SCB-GR Boundary Anchor Health", "-" * 90]
#         lines.append(f"Overall: n={int(bv.get('n', 0))} | pairs={int(float(bm.get('pair_count', 0)))} | acc={float(bv.get('accuracy', 0.0)):.2f}% | mean_margin={float(bv.get('mean_margin', 0.0)):.6f} | min_margin={float(bv.get('min_margin', 0.0)):.6f} | viol={float(bv.get('violation_rate', 0.0)):.2f}%")
#         return "\n".join(lines) + "\n"

#     def _print_geometry_diagnostics_summary(self, report: Dict[str, object]) -> None:
#         em = report.get("energy_margin", {}).get("overall", {})
#         ar = report.get("anchor_replay", {}).get("overall", {})
#         old_new_summary = report.get("old_new_overlap_summary", {})
#         br = report.get("boundary_anchor_replay", {}).get("overall", {})
#         print(
#             "[Geometry Health] "
#             f"energy_acc={float(em.get('accuracy', 0.0)):.2f}% | "
#             f"energy_viol={float(em.get('violation_rate', 0.0)):.2f}% | "
#             f"anchor_acc={float(ar.get('accuracy', 0.0)):.2f}% | "
#             f"anchor_viol={float(ar.get('violation_rate', 0.0)):.2f}% | "
#             f"boundary_viol={float(br.get('violation_rate', 0.0)):.2f}% | "
#             f"old_new_max_risk={float(old_new_summary.get('max_risk', 0.0)):.4f}"
#         )
#         top_pair = old_new_summary.get("top_pair", None)
#         if isinstance(top_pair, dict):
#             print(
#                 "[Geometry Health] Top old/new conflict: "
#                 f"old {top_pair.get('old_name', top_pair.get('old_class', 'NA'))} -> "
#                 f"new {top_pair.get('new_name', top_pair.get('new_class', 'NA'))} | "
#                 f"risk={float(top_pair.get('risk_score', 0.0)):.4f} | "
#                 f"overlap={float(top_pair.get('feature_overlap', 0.0)):.4f} | "
#                 f"spec_sim={float(top_pair.get('spectral_shape_similarity', 0.0)):.4f}"
#             )
#         for alert in report.get("alerts", [])[:10]:
#             print(f"[Geometry Health WARN] {alert}")

