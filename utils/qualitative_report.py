from __future__ import annotations

"""Post-training qualitative maps for the finalized NECIL-HSI base phase.

This module is reporting-only. It reads all labeled pixels of the finalized
base classes after training/model selection has finished, obtains predictions
from the already-finalized model, and renders full-scene ground-truth and
prediction maps. It never changes parameters, normalization, or class geometry.
"""

import os
from typing import Any, Dict, Mapping

import numpy as np
import torch

from utils.visualize import save_full_phase_qualitative_maps


@torch.no_grad()
def generate_phase_qualitative_maps(
    *,
    model: Any,
    dataset: Any,
    phase: int,
    output_dir: str,
    device: str | torch.device,
    batch_size: int = 256,
    cmap_name: str = "nipy_spectral",
    dpi: int = 300,
) -> Dict[str, str]:
    """Generate full labeled-scene GT/prediction maps for any finalized phase.

    This is reporting-only. The ``split="all"`` reporting loader is accessed
    only after the requested phase has been finalized, and the resulting rows
    never enter optimization, model selection, replay construction, or geometry
    updates.
    """
    phase = int(phase)
    if phase < 0:
        raise ValueError("phase must be non-negative")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")

    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError("model and reporting device disagree")
    model.validate_model_state()

    finalized = [
        int(value)
        for value in getattr(dataset, "finalized_phases", [])
    ]
    if phase not in finalized:
        raise RuntimeError(
            f"qualitative reporting requires finalized phase {phase}"
        )
    if getattr(dataset, "current_phase", None) is not None:
        raise RuntimeError(
            "qualitative reporting cannot run during an active phase"
        )

    get_seen = getattr(dataset, "get_seen_classes", None)
    reporting_loader = getattr(dataset, "get_reporting_dataloader", None)
    if not callable(get_seen) or not callable(reporting_loader):
        raise RuntimeError("dataset lacks finalized reporting APIs")

    class_ids = [int(value) for value in get_seen(phase)]
    if (
        not class_ids
        or len(class_ids) != len(set(class_ids))
        or any(value < 0 for value in class_ids)
    ):
        raise RuntimeError("dataset returned invalid seen class IDs")

    committed = [int(value) for value in model.committed_class_ids]
    if committed != class_ids:
        raise RuntimeError(
            f"committed geometry {committed} does not match "
            f"phase-{phase} seen classes {class_ids}"
        )

    target_names = list(getattr(dataset, "target_names", []))
    if not target_names:
        raise RuntimeError("dataset must expose target_names")
    total_classes = len(target_names)
    if max(class_ids) >= total_classes:
        raise RuntimeError("seen class IDs exceed dataset class count")

    loader = reporting_loader(
        phase,
        split="all",
        batch_size=int(batch_size),
    )

    coords_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []

    model.eval()
    for batch in loader:
        if not isinstance(batch, Mapping):
            raise TypeError("reporting batches must be mappings")
        required = {"image", "raw_center_spectrum", "label", "coord"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"reporting batch missing {sorted(missing)}")

        patch = torch.as_tensor(
            batch["image"],
            device=evaluation_device,
            dtype=torch.float32,
        )
        spectrum = torch.as_tensor(
            batch["raw_center_spectrum"],
            device=evaluation_device,
            dtype=torch.float32,
        )
        labels = torch.as_tensor(
            batch["label"],
            device=evaluation_device,
        ).flatten()
        coords = torch.as_tensor(batch["coord"]).detach().cpu()

        backbone = model.backbone
        expected_patch = (
            patch.size(0) if patch.ndim > 0 else 0,
            int(backbone.patch_bands),
            int(backbone.patch_size),
            int(backbone.patch_size),
        )
        expected_spectrum = (
            patch.size(0) if patch.ndim > 0 else 0,
            int(backbone.spectral_bands),
        )
        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape) != expected_patch
        ):
            raise RuntimeError(
                "reporting patch shape disagrees with the backbone"
            )
        if (
            spectrum.ndim != 2
            or tuple(spectrum.shape) != expected_spectrum
        ):
            raise RuntimeError(
                "reporting spectrum shape disagrees with the backbone"
            )
        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(spectrum).all()
        ):
            raise RuntimeError("reporting input contains NaN/Inf")

        if (
            labels.numel() != patch.size(0)
            or labels.dtype == torch.bool
            or labels.is_complex()
        ):
            raise RuntimeError(
                "reporting labels are invalid or batch-misaligned"
            )
        if torch.is_floating_point(labels):
            if not bool(torch.isfinite(labels).all()) or not bool(
                labels.eq(labels.round()).all()
            ):
                raise RuntimeError(
                    "reporting labels must be finite integers"
                )
        labels = labels.to(dtype=torch.long)
        if bool((labels < 0).any()):
            raise RuntimeError("reporting labels must be non-negative")

        allowed = torch.tensor(
            class_ids,
            device=labels.device,
            dtype=torch.long,
        )
        if not bool(torch.isin(labels, allowed).all()):
            raise RuntimeError(
                "reporting loader returned a phase-invisible class"
            )

        if coords.dtype == torch.bool or coords.is_complex():
            raise RuntimeError(
                "reporting coordinates must contain integers"
            )
        if torch.is_floating_point(coords):
            if not bool(torch.isfinite(coords).all()) or not bool(
                coords.eq(coords.round()).all()
            ):
                raise RuntimeError(
                    "reporting coordinates must contain finite integers"
                )
        coords = coords.long()
        if tuple(coords.shape) != (labels.numel(), 2):
            raise RuntimeError(
                "reporting coordinates must be [B,2]"
            )

        output = model(
            patch,
            center_spectrum=spectrum,
            class_ids=class_ids,
            candidate=None,
            return_aux=False,
        ).classification

        returned_ids = [
            int(value)
            for value in output.class_ids.detach().cpu().tolist()
        ]
        if returned_ids != class_ids:
            raise RuntimeError(
                "classifier column order changed during reporting"
            )
        expected_prediction = output.class_ids.index_select(
            0,
            output.energy.argmin(dim=1),
        )
        if not torch.equal(
            output.prediction,
            expected_prediction,
        ):
            raise RuntimeError(
                "reporting prediction is not the minimum-energy class"
            )

        coords_parts.append(coords)
        target_parts.append(labels.detach().cpu())
        prediction_parts.append(
            output.prediction.detach().cpu()
        )

    if not target_parts:
        raise RuntimeError("reporting loader is empty")

    coords = torch.cat(coords_parts).numpy()
    targets = torch.cat(target_parts).numpy()
    predictions = torch.cat(prediction_parts).numpy()

    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise RuntimeError(
            "reporting coordinates contain duplicates"
        )

    return save_full_phase_qualitative_maps(
        output_dir=os.path.abspath(output_dir),
        phase=phase,
        gt_shape=dataset.gt_shape,
        coords=coords,
        targets=targets,
        predictions=predictions,
        class_ids=class_ids,
        total_classes=total_classes,
        target_names=target_names,
        cmap_name=str(cmap_name),
        dpi=int(dpi),
    )


@torch.no_grad()
def generate_base_qualitative_maps(
    *,
    model: Any,
    dataset: Any,
    output_dir: str,
    device: str | torch.device,
    batch_size: int = 256,
    cmap_name: str = "nipy_spectral",
    dpi: int = 300,
) -> Dict[str, str]:
    """Backward-compatible phase-0 wrapper."""
    return generate_phase_qualitative_maps(
        model=model,
        dataset=dataset,
        phase=0,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cmap_name=cmap_name,
        dpi=dpi,
    )


__all__ = ["generate_phase_qualitative_maps", "generate_base_qualitative_maps"]











# from __future__ import annotations

# """Post-training qualitative maps for the finalized NECIL-HSI base phase.

# This module is reporting-only. It reads all labeled pixels of the finalized
# base classes after training/model selection has finished, obtains predictions
# from the already-finalized model, and renders full-scene ground-truth and
# prediction maps. It never changes parameters, normalization, or class geometry.
# """

# import os
# from typing import Any, Dict, Mapping

# import numpy as np
# import torch

# from utils.visualize import save_full_phase_qualitative_maps


# @torch.no_grad()
# def generate_base_qualitative_maps(
#     *,
#     model: Any,
#     dataset: Any,
#     output_dir: str,
#     device: str | torch.device,
#     batch_size: int = 256,
#     cmap_name: str = "nipy_spectral",
#     dpi: int = 300,
# ) -> Dict[str, str]:
#     """Generate full labeled-scene maps after base phase 0 is finalized."""
#     if int(batch_size) <= 0:
#         raise ValueError("batch_size must be positive")
#     if int(dpi) <= 0:
#         raise ValueError("dpi must be positive")

#     evaluation_device = torch.device(device)
#     if torch.device(model.device) != evaluation_device:
#         raise RuntimeError("model and reporting device disagree")
#     model.validate_model_state()

#     finalized = [int(value) for value in getattr(dataset, "finalized_phases", [])]
#     if finalized != [0]:
#         raise RuntimeError("base qualitative reporting requires finalized phase 0 only")
#     if getattr(dataset, "current_phase", None) is not None:
#         raise RuntimeError("qualitative reporting cannot run during an active phase")

#     get_seen = getattr(dataset, "get_seen_classes", None)
#     reporting_loader = getattr(dataset, "get_reporting_dataloader", None)
#     if not callable(get_seen) or not callable(reporting_loader):
#         raise RuntimeError("dataset lacks finalized reporting APIs")

#     class_ids = [int(value) for value in get_seen(0)]
#     if not class_ids or len(class_ids) != len(set(class_ids)) or any(value < 0 for value in class_ids):
#         raise RuntimeError("dataset returned invalid base class IDs")

#     committed = [int(value) for value in model.committed_class_ids]
#     if committed != class_ids:
#         raise RuntimeError(
#             f"committed geometry {committed} does not match finalized base classes {class_ids}"
#         )

#     target_names = list(getattr(dataset, "target_names", []))
#     if not target_names:
#         raise RuntimeError("dataset must expose target_names")
#     total_classes = len(target_names)
#     if max(class_ids) >= total_classes:
#         raise RuntimeError("base class IDs exceed dataset class count")

#     loader = reporting_loader(
#         0,
#         split="all",
#         batch_size=int(batch_size),
#     )

#     coords_parts: list[torch.Tensor] = []
#     target_parts: list[torch.Tensor] = []
#     prediction_parts: list[torch.Tensor] = []

#     model.eval()
#     for batch in loader:
#         if not isinstance(batch, Mapping):
#             raise TypeError("reporting batches must be mappings")
#         required = {"image", "raw_center_spectrum", "label", "coord"}
#         missing = required - set(batch)
#         if missing:
#             raise KeyError(f"reporting batch missing {sorted(missing)}")

#         patch = torch.as_tensor(
#             batch["image"],
#             device=evaluation_device,
#             dtype=torch.float32,
#         )
#         spectrum = torch.as_tensor(
#             batch["raw_center_spectrum"],
#             device=evaluation_device,
#             dtype=torch.float32,
#         )
#         labels = torch.as_tensor(
#             batch["label"],
#             device=evaluation_device,
#         ).flatten()
#         coords = torch.as_tensor(batch["coord"]).detach().cpu()

#         backbone = model.backbone
#         expected_patch = (
#             patch.size(0) if patch.ndim > 0 else 0,
#             int(backbone.patch_bands),
#             int(backbone.patch_size),
#             int(backbone.patch_size),
#         )
#         expected_spectrum = (
#             patch.size(0) if patch.ndim > 0 else 0,
#             int(backbone.spectral_bands),
#         )
#         if patch.ndim != 4 or patch.size(0) == 0 or tuple(patch.shape) != expected_patch:
#             raise RuntimeError("reporting patch shape disagrees with the backbone")
#         if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
#             raise RuntimeError("reporting spectrum shape disagrees with the backbone")
#         if not bool(torch.isfinite(patch).all()) or not bool(torch.isfinite(spectrum).all()):
#             raise RuntimeError("reporting input contains NaN/Inf")

#         if labels.numel() != patch.size(0) or labels.dtype == torch.bool or labels.is_complex():
#             raise RuntimeError("reporting labels are invalid or batch-misaligned")
#         if torch.is_floating_point(labels):
#             if not bool(torch.isfinite(labels).all()) or not bool(labels.eq(labels.round()).all()):
#                 raise RuntimeError("reporting labels must be finite integers")
#         labels = labels.to(dtype=torch.long)
#         if bool((labels < 0).any()):
#             raise RuntimeError("reporting labels must be non-negative")
#         if not bool(torch.isin(labels, torch.tensor(class_ids, device=labels.device)).all()):
#             raise RuntimeError("reporting loader returned a class outside the finalized base set")

#         if coords.dtype == torch.bool or coords.is_complex():
#             raise RuntimeError("reporting coordinates must contain integers")
#         if torch.is_floating_point(coords):
#             if not bool(torch.isfinite(coords).all()) or not bool(coords.eq(coords.round()).all()):
#                 raise RuntimeError("reporting coordinates must contain finite integers")
#         coords = coords.long()
#         if tuple(coords.shape) != (labels.numel(), 2):
#             raise RuntimeError("reporting coordinates must be [B,2]")

#         output = model(
#             patch,
#             center_spectrum=spectrum,
#             class_ids=class_ids,
#             candidate=None,
#             return_aux=False,
#         ).classification
#         returned_ids = [int(value) for value in output.class_ids.detach().cpu().tolist()]
#         if returned_ids != class_ids:
#             raise RuntimeError("classifier column order changed during reporting")
#         expected_prediction = output.class_ids.index_select(
#             0,
#             output.energy.argmin(dim=1),
#         )
#         if not torch.equal(output.prediction, expected_prediction):
#             raise RuntimeError("reporting prediction is not the minimum-energy class")

#         coords_parts.append(coords)
#         target_parts.append(labels.detach().cpu())
#         prediction_parts.append(output.prediction.detach().cpu())

#     if not target_parts:
#         raise RuntimeError("reporting loader is empty")

#     coords = torch.cat(coords_parts).numpy()
#     targets = torch.cat(target_parts).numpy()
#     predictions = torch.cat(prediction_parts).numpy()

#     if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
#         raise RuntimeError("reporting coordinates contain duplicates")

#     return save_full_phase_qualitative_maps(
#         output_dir=os.path.abspath(output_dir),
#         phase=0,
#         gt_shape=dataset.gt_shape,
#         coords=coords,
#         targets=targets,
#         predictions=predictions,
#         class_ids=class_ids,
#         total_classes=total_classes,
#         target_names=target_names,
#         cmap_name=str(cmap_name),
#         dpi=int(dpi),
#     )


# __all__ = ["generate_base_qualitative_maps"]


