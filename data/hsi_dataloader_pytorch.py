"""HSI loading For NECIL.

Responsibilities
----------------
- Resolve raw HSI label/background policy.
- Build the class-incremental protocol before fitting preprocessing.
- Fit normalization/PCA only on phase-0 training pixels.
- Apply physical +/- gain and wavelength-tilt interventions before the frozen
  preprocessing transform.
- Return processed intervention center vectors for lazy center-only patch
  construction by ``IncrementalHSIDataset``.

This module never sends raw spectra to the model and never creates a one-sided
``spectral_augmented_patch``.
"""

from __future__ import annotations

import os
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# DATASET CONFIGURATIONS
# ============================================================
DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "LK": {
        "data_file": "WHU_Hi_LongKou.mat",
        "data_key": "WHU_Hi_LongKou",
        "gt_file": "WHU_Hi_LongKou_gt.mat",
        "gt_key": "WHU_Hi_LongKou_gt",
        "num_classes": 9,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
            "Narrow-leaf soybean", "Rice", "Water",
            "Roads and houses", "Mixed weed"
        ],
    },
    "HH": {
        "data_file": "WHU_Hi_HongHu.mat",
        "data_key": "WHU_Hi_HongHu",
        "gt_file": "WHU_Hi_HongHu_gt.mat",
        "gt_key": "WHU_Hi_HongHu_gt",
        "num_classes": 22,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Red roof", "Road", "Bare soil", "Cotton",
            "Cotton firewood", "Rape", "Chinese cabbage",
            "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis",
            "Brassica chinensis", "Small Brassica chinensis", "Lactuca sativa",
            "Celtuce", "Film covered lettuce", "Romaine lettuce",
            "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree"
        ],
    },
    "HC": {
        "data_file": "WHU_Hi_HanChuan.mat",
        "data_key": "WHU_Hi_HanChuan",
        "gt_file": "WHU_Hi_HanChuan_gt.mat",
        "gt_key": "WHU_Hi_HanChuan_gt",
        "num_classes": 16,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Strawberry", "Cowpea", "Soybean", "Sorghum",
            "Water spinach", "Watermelon", "Greens", "Trees", "Grass",
            "Red roof", "Gray roof", "Plastic", "Bare soil", "Road",
            "Bright object", "Water"
        ],
    },
    "BS": {
        "data_file": "Botswana.mat",
        "data_key": "Botswana",
        "gt_file": "Botswana_gt.mat",
        "gt_key": "Botswana_gt",
        "num_classes": 14,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Water", "Hippo Grass", "Floodplain Grasses 1", "Floodplain Grasses 2",
            "Reeds 1", "Riparian", "Firescar 2", "Island Interior", "Woodlands",
            "Acacia Shrublands", "Acacia Grasslands", "Short Mopane",
            "Mixed Mopane", "Exposed Soils"
        ],
    },
    "PU": {
        "data_file": "PaviaU.mat",
        "data_key": "paviaU",
        "gt_file": "PaviaU_gt.mat",
        "gt_key": "paviaU_gt",
        "num_classes": 9,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Asphalt", "Meadows", "Gravel", "Trees", "Painted", "Soil",
            "Bitumen", "Bricks", "Shadows"
        ],
    },
    "PC": {
        "data_file": "Pavia.mat",
        "data_key": "pavia",
        "gt_file": "Pavia_gt.mat",
        "gt_key": "pavia_gt",
        "num_classes": 9,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Water", "Trees", "Asphalt", "Bricks", "Bitumen",
            "Tiles", "Shadows", "Meadows", "Soil"
        ],
    },
    "SA": {
        "data_file": "Salinas_corrected.mat",
        "data_key": "salinas_corrected",
        "gt_file": "Salinas_gt.mat",
        "gt_key": "salinas_gt",
        "num_classes": 16,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Weeds_1", "Weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth",
            "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
            "Corn_Weeds", "Lettuce_4wk", "Lettuce_5wk", "Lettuce_6wk",
            "Lettuce_7wk", "Vinyard_untrained", "Vinyard_trellis"
        ],
    },
    "UH13": {
        "data_file": "HU13.mat",
        "data_key": "HSI",
        "gt_file": "HU13_gt.mat",
        "gt_key": "gt",
        "num_classes": 15,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Healthy grass", "Stressed grass", "Synthetic grass", "Trees",
            "Soil", "Water", "Residential", "Commercial", "Road",
            "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
            "Tennis Court", "Running Track"
        ],
    },
    "QUH": {
        "data_file": "QUH-Qingyun.mat",
        "data_key": "Chengqu",
        "gt_file": "QUH-Qingyun_GT.mat",
        "gt_key": "ChengquGT",
        "num_classes": 6,
        "has_background": False,
        "background_label": None,
        "target_names": [
            "Trees", "Concrete building", "Car", "Ironhide building",
            "Plastic playground", "Asphalt road"
        ],
    },
    "PI": {
        "data_file": "QUH-Pingan.mat",
        "data_key": "Haigang",
        "gt_file": "QUH-Pingan_GT.mat",
        "gt_key": "HaigangGT",
        "num_classes": 10,
        "has_background": False,
        "background_label": None,
        "target_names": [
            "Ship", "Seawater", "Trees", "Concrete structure building",
            "Floating pier", "Brick houses", "Steel houses",
            "Wharf construction land", "Car", "Road"
        ],
    },
    "TH": {
        "data_file": "QUH-Tangdaowan.mat",
        "data_key": "Tangdaowan",
        "gt_file": "QUH-Tangdaowan_GT.mat",
        "gt_key": "TangdaowanGT",
        "num_classes": 18,
        "has_background": False,
        "background_label": None,
        "target_names": [
            "Rubber track", "Flagging", "Sandy", "Asphalt", "Boardwalk", "Rocky shallows",
            "Grassland", "Bulrush", "Gravel road", "Ligustrum vicaryi", "Coniferous pine",
            "Spiraea", "Bare soil", "Buxus sinica", "Photinia serrulata", "Populus",
            "Ulmus pumila L", "Seawater"
        ],
    },
    "IP": {
        "data_file": "Indian_pines_corrected.mat",
        "data_key": "indian_pines_corrected",
        "gt_file": "Indian_pines_gt.mat",
        "gt_key": "indian_pines_gt",
        "num_classes": 16,
        "has_background": True,
        "background_label": 0,
        "target_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
            "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
        ],
    },
}


# ============================================================
# VALIDATION / LABEL POLICY
# ============================================================
IGNORE_LABEL_VALUES = (255,)


def _valid_gt_values(GT: np.ndarray, ignore_values=IGNORE_LABEL_VALUES) -> np.ndarray:
    vals = np.unique(np.asarray(GT).astype(np.int64))
    mask = vals >= 0
    for ig in ignore_values:
        mask &= vals != int(ig)
    return vals[mask]


def _load_mat_key(path: str, key: str) -> np.ndarray:
    obj = sio.loadmat(path)
    if key not in obj:
        available = [k for k in obj.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {available}")
    return obj[key]


def _validate_hsi_gt_shapes(HSI: np.ndarray, GT: np.ndarray, method: str) -> None:
    if HSI.ndim != 3:
        raise ValueError(f"{method}: HSI must be [H,W,B], got shape={HSI.shape}")
    if GT.ndim != 2:
        raise ValueError(f"{method}: GT must be [H,W], got shape={GT.shape}")
    if HSI.shape[:2] != GT.shape:
        raise ValueError(f"{method}: HSI spatial shape {HSI.shape[:2]} does not match GT {GT.shape}")


def resolve_label_policy(
    method: str,
    GT: np.ndarray,
    *,
    trust_config: bool = True,
) -> Dict[str, Any]:
    """
    Resolve raw-label -> sequential-training-label mapping.

    This function never treats raw GT 0 as background unless DATASET_INFO says
    has_background=True. That is non-negotiable; otherwise class-0-real datasets
    get corrupted before NECIL training starts.
    """
    del trust_config

    if method not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {method}. Available: {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[method]
    gt_vals = _valid_gt_values(GT)
    if gt_vals.size == 0:
        raise ValueError("GT contains no valid labels.")

    has_background = bool(info.get("has_background", True))
    background_label = info.get("background_label", 0 if has_background else None)
    num_classes_cfg = int(info["num_classes"])
    target_names = list(info.get("target_names", []))

    if "class_values" in info and info["class_values"] is not None:
        class_values = sorted(int(v) for v in info["class_values"])
    elif has_background:
        class_values = sorted(int(v) for v in gt_vals.tolist() if int(v) != int(background_label))
    else:
        class_values = sorted(int(v) for v in gt_vals.tolist())

        maybe_zero_background = (
            0 in gt_vals.tolist()
            and len(gt_vals) == num_classes_cfg + 1
            and int(gt_vals.min()) == 0
            and int(gt_vals.max()) == num_classes_cfg
            and len(target_names) == num_classes_cfg
        )
        if maybe_zero_background:
            raise RuntimeError(
                f"[LabelPolicy:ERROR] {method}: config says has_background=False, "
                f"but GT has values 0..{num_classes_cfg} and exactly {num_classes_cfg} "
                "target names. This would create a fake extra class. Set "
                "has_background=True and background_label=0."
            )

    if len(class_values) == 0:
        raise RuntimeError(
            f"No foreground classes resolved for {method}. "
            f"has_background={has_background}, background_label={background_label}, gt_vals={gt_vals.tolist()}"
        )

    present = set(int(v) for v in gt_vals.tolist())
    missing = [v for v in class_values if v not in present]
    if missing:
        print(f"[LabelPolicy:WARN] {method}: configured class values missing from GT: {missing}")

    if len(class_values) != num_classes_cfg:
        print(
            f"[LabelPolicy:WARN] {method}: config num_classes={num_classes_cfg}, "
            f"but resolved present class values={len(class_values)} -> {class_values}. "
            "Using resolved class values."
        )

    label_to_train = {raw: i for i, raw in enumerate(class_values)}
    train_to_label = {i: raw for raw, i in label_to_train.items()}

    if len(target_names) > len(class_values):
        target_names = target_names[:len(class_values)]
    elif len(target_names) < len(class_values):
        target_names = target_names + [f"Class {v}" for v in class_values[len(target_names):]]

    return {
        "method": method,
        "has_background": has_background,
        "background_label": background_label,
        "raw_class_values": class_values,
        "label_to_train": label_to_train,
        "train_to_label": train_to_label,
        "num_classes": len(class_values),
        "target_names": target_names,
        "gt_unique_values": gt_vals.tolist(),
    }


def _foreground_mask_from_policy(GT: np.ndarray, policy: Dict[str, Any]) -> np.ndarray:
    raw_class_values = np.asarray(policy["raw_class_values"], dtype=np.int64)
    if raw_class_values.size == 0:
        raise ValueError("label_policy.raw_class_values is empty.")
    return np.isin(np.asarray(GT, dtype=np.int64), raw_class_values)


def _fit_pixels_from_mask(HSI: np.ndarray, fit_mask: Optional[np.ndarray]) -> np.ndarray:
    flat = np.asarray(HSI, dtype=np.float32).reshape(-1, HSI.shape[-1])
    if fit_mask is None:
        return flat

    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != HSI.shape[:2]:
        raise ValueError(f"fit_mask shape {mask.shape} must match HSI spatial shape {HSI.shape[:2]}")

    fit = flat[mask.reshape(-1)]
    if fit.shape[0] == 0:
        raise ValueError("No pixels available for normalization/PCA fitting after applying fit_mask.")
    return fit



def ExtractLabeledPixelIndex(
    GT: np.ndarray,
    *,
    label_policy: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mapped foreground labels and coordinates without extracting patches.

    This lightweight index is deliberately available before normalization/PCA.
    The experiment controller can therefore create the class-incremental protocol
    and its train/validation/test split before fitting any spectral preprocessing.
    """
    gt = np.asarray(GT, dtype=np.int64)
    if gt.ndim != 2:
        raise ValueError(f"GT must be [H,W], got {gt.shape}")

    raw_class_values = [int(value) for value in label_policy["raw_class_values"]]
    label_to_train = {
        int(key): int(value)
        for key, value in dict(label_policy["label_to_train"]).items()
    }
    mask = np.isin(gt, np.asarray(raw_class_values, dtype=np.int64))
    rows, cols = np.where(mask)
    coords = np.stack([rows, cols], axis=1).astype(np.int64, copy=False)

    raw_labels = gt[rows, cols]
    mapped = np.empty(raw_labels.shape[0], dtype=np.int64)
    for index, raw_label in enumerate(raw_labels.tolist()):
        if int(raw_label) not in label_to_train:
            raise RuntimeError(
                f"Raw label {raw_label} is absent from label_to_train."
            )
        mapped[index] = label_to_train[int(raw_label)]

    expected = list(range(int(label_policy["num_classes"])))
    present = sorted(int(value) for value in np.unique(mapped).tolist())
    if present != expected:
        raise RuntimeError(
            "Mapped labeled-pixel index must cover contiguous semantic IDs "
            f"{expected}; found {present}."
        )
    return mapped, coords


def BuildPreprocessFitMask(
    gt_shape: Sequence[int],
    coords: np.ndarray,
    sample_indices: Sequence[int],
    *,
    labels: Optional[np.ndarray] = None,
    allowed_classes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Build a spatial fit mask from exact sample indices.

    ``sample_indices`` must index the same labeled-pixel order returned by
    :func:`ExtractLabeledPixelIndex` and later by :func:`ImageCubes`.
    When ``allowed_classes`` is supplied, every selected sample is verified to
    belong to that class set. This catches accidental future-class leakage.
    """
    shape = tuple(int(value) for value in gt_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"gt_shape must contain positive H,W values, got {shape}")

    coords_array = np.asarray(coords, dtype=np.int64)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(f"coords must be [N,2], got {coords_array.shape}")

    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("sample_indices is empty; cannot fit preprocessing.")
    if int(indices.min()) < 0 or int(indices.max()) >= len(coords_array):
        raise IndexError(
            f"sample_indices outside [0,{len(coords_array) - 1}]: "
            f"min={int(indices.min())}, max={int(indices.max())}"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("sample_indices contains duplicates.")

    if labels is not None:
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels_array) != len(coords_array):
            raise ValueError(
                f"labels/coords length mismatch: {len(labels_array)} vs "
                f"{len(coords_array)}"
            )
        if allowed_classes is not None:
            allowed = set(int(value) for value in allowed_classes)
            selected_classes = set(
                int(value) for value in labels_array[indices].tolist()
            )
            invalid = sorted(selected_classes.difference(allowed))
            if invalid:
                raise RuntimeError(
                    "Preprocessing fit indices contain classes outside the "
                    f"allowed base-training set: invalid={invalid}, "
                    f"allowed={sorted(allowed)}"
                )

    selected_coords = coords_array[indices]
    rows = selected_coords[:, 0]
    cols = selected_coords[:, 1]
    if (
        int(rows.min()) < 0
        or int(cols.min()) < 0
        or int(rows.max()) >= shape[0]
        or int(cols.max()) >= shape[1]
    ):
        raise RuntimeError(
            f"Selected coordinates fall outside GT shape {shape}."
        )

    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    if int(mask.sum()) != len(indices):
        raise RuntimeError(
            "Fit-mask cardinality differs from selected sample count. "
            "Coordinates may contain duplicates."
        )
    return mask


def _validate_fit_mask(
    HSI: np.ndarray,
    fit_mask: np.ndarray,
    *,
    name: str = "fit_mask",
) -> np.ndarray:
    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape != HSI.shape[:2]:
        raise ValueError(
            f"{name} shape {mask.shape} must match HSI spatial shape "
            f"{HSI.shape[:2]}"
        )
    if not bool(mask.any()):
        raise ValueError(f"{name} selects zero pixels.")
    return mask


def _normalization_state(
    HSI: np.ndarray,
    fit_mask: np.ndarray,
    *,
    eps: float,
) -> Dict[str, Any]:
    fit_pixels = _fit_pixels_from_mask(HSI, fit_mask)
    mean = fit_pixels.mean(axis=0).astype(np.float32)
    std = fit_pixels.std(axis=0).astype(np.float32)
    safe_std = np.maximum(std, float(eps)).astype(np.float32)
    return {
        "mean": mean,
        "std": safe_std,
        "eps": float(eps),
        "raw_bands": int(HSI.shape[-1]),
    }


def _apply_normalization_state(
    HSI: np.ndarray,
    state: Mapping[str, Any],
) -> np.ndarray:
    cube = np.asarray(HSI, dtype=np.float32)
    mean = np.asarray(state["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(state["std"], dtype=np.float32).reshape(1, 1, -1)
    if mean.shape[-1] != cube.shape[-1] or std.shape[-1] != cube.shape[-1]:
        raise ValueError(
            "Normalization state band count does not match input cube: "
            f"state={mean.shape[-1]}, cube={cube.shape[-1]}"
        )
    normalized = (cube - mean) / std
    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _fit_reduction_state(
    normalized_hsi: np.ndarray,
    *,
    fit_mask: np.ndarray,
    method: str,
    n_components: int,
    whiten: bool,
) -> Dict[str, Any]:
    from sklearn.decomposition import IncrementalPCA, PCA

    H, W, B = normalized_hsi.shape
    del H, W
    fit_pixels = _fit_pixels_from_mask(normalized_hsi, fit_mask)
    maximum = min(int(B), int(fit_pixels.shape[0]))
    components = int(n_components)
    if components <= 0:
        raise ValueError(f"n_components must be positive, got {components}")
    if components > maximum:
        print(
            f"[PCA:WARN] requested {components} components but maximum fit rank "
            f"is {maximum}; clipping."
        )
        components = maximum

    method_normalized = str(method).strip().lower()
    if method_normalized == "pca":
        estimator = PCA(
            n_components=components,
            whiten=bool(whiten),
            svd_solver="auto",
        )
        estimator.fit(fit_pixels)
    elif method_normalized == "ipca":
        if bool(whiten):
            raise ValueError(
                "Whitened IncrementalPCA is not supported by this strict loader."
            )
        estimator = IncrementalPCA(n_components=components)
        minimum_batch = max(components, components * 20)
        batch_count = min(
            256,
            max(1, int(np.ceil(fit_pixels.shape[0] / float(minimum_batch)))),
        )
        for batch in np.array_split(fit_pixels, batch_count):
            if batch.shape[0] >= components:
                estimator.partial_fit(batch)
    else:
        raise ValueError(
            f"Unknown reduction method {method!r}; use 'PCA' or 'iPCA'."
        )

    explained_ratio = np.asarray(
        getattr(estimator, "explained_variance_ratio_", np.zeros(components)),
        dtype=np.float32,
    )
    explained_variance = np.asarray(
        getattr(estimator, "explained_variance_", np.ones(components)),
        dtype=np.float32,
    )
    singular_values = np.asarray(
        getattr(estimator, "singular_values_", np.zeros(components)),
        dtype=np.float32,
    )

    return {
        "active": True,
        "method": method_normalized,
        "input_bands": int(B),
        "n_components": int(components),
        "whiten": bool(whiten),
        "mean": np.asarray(estimator.mean_, dtype=np.float32),
        "components": np.asarray(estimator.components_, dtype=np.float32),
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
        "singular_values": singular_values,
        "fit_samples": int(fit_pixels.shape[0]),
    }


def _apply_reduction_state(
    normalized_hsi: np.ndarray,
    state: Mapping[str, Any],
) -> np.ndarray:
    if not bool(state.get("active", False)):
        return np.asarray(normalized_hsi, dtype=np.float32)

    cube = np.asarray(normalized_hsi, dtype=np.float32)
    H, W, B = cube.shape
    components = np.asarray(state["components"], dtype=np.float32)
    mean = np.asarray(state["mean"], dtype=np.float32).reshape(1, -1)
    if components.ndim != 2 or components.shape[1] != B or mean.shape[1] != B:
        raise ValueError(
            "Reduction state does not match normalized cube dimensions: "
            f"components={components.shape}, mean={mean.shape}, bands={B}"
        )

    flat = cube.reshape(-1, B)
    reduced = (flat - mean) @ components.T
    if bool(state.get("whiten", False)):
        explained = np.asarray(
            state["explained_variance"],
            dtype=np.float32,
        ).reshape(1, -1)
        reduced = reduced / np.sqrt(np.maximum(explained, 1e-12))

    reduced = np.nan_to_num(
        reduced,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return reduced.reshape(H, W, components.shape[0]).astype(np.float32)


def FitHSIPreprocessor(
    HSI_raw: np.ndarray,
    *,
    fit_mask: np.ndarray,
    apply_reduction: bool,
    reduction_method: str = "PCA",
    n_components: int = 30,
    whiten: bool = False,
    normalization_eps: float = 1e-8,
    fit_scope: str = "base_train",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit normalization and optional PCA on an explicit pixel subset.

    Returns:
        model_input_hsi:
            normalized cube, optionally PCA-reduced;
        normalized_physical_hsi:
            normalized wavelength-ordered cube for optional metadata only;
        state:
            complete immutable preprocessing state needed for exact reuse.
    """
    raw = np.asarray(HSI_raw, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"HSI_raw must be [H,W,B], got {raw.shape}")

    mask = _validate_fit_mask(raw, fit_mask)
    normalization = _normalization_state(
        raw,
        mask,
        eps=float(normalization_eps),
    )
    normalized = _apply_normalization_state(raw, normalization)

    if bool(apply_reduction):
        reduction = _fit_reduction_state(
            normalized,
            fit_mask=mask,
            method=reduction_method,
            n_components=int(n_components),
            whiten=bool(whiten),
        )
        model_input = _apply_reduction_state(normalized, reduction)
    else:
        reduction = {
            "active": False,
            "method": "identity",
            "input_bands": int(raw.shape[-1]),
            "n_components": int(raw.shape[-1]),
            "whiten": False,
            "fit_samples": int(mask.sum()),
        }
        model_input = normalized

    state: Dict[str, Any] = {
        "version": 1,
        "fit_scope": str(fit_scope),
        "fit_pixel_count": int(mask.sum()),
        "spatial_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "raw_bands": int(raw.shape[-1]),
        "normalization": normalization,
        "reduction": reduction,
    }
    return model_input, normalized, state


def ApplyHSIPreprocessor(
    HSI_raw: np.ndarray,
    state: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a previously fitted immutable normalization/PCA state."""
    if int(state.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported preprocessing state version: {state.get('version')}"
        )
    raw = np.asarray(HSI_raw, dtype=np.float32)
    expected_bands = int(state["raw_bands"])
    if raw.ndim != 3 or raw.shape[-1] != expected_bands:
        raise ValueError(
            f"Raw cube must be [H,W,{expected_bands}], got {raw.shape}"
        )

    normalized = _apply_normalization_state(
        raw,
        state["normalization"],
    )
    model_input = _apply_reduction_state(
        normalized,
        state["reduction"],
    )
    return model_input, normalized



def ApplyHSIPreprocessorToVectors(
    spectra: np.ndarray,
    state: Mapping[str, Any],
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Apply the frozen HSI preprocessor to aligned spectral vectors.

    ``ApplyHSIPreprocessor`` accepts an image cube. A temporary [N,1,B] cube is
    used so the exact same normalization and PCA equations are applied without
    reimplementing either transform.
    """
    values = np.asarray(spectra, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] <= 1:
        raise ValueError(f"spectra must be [N,B] with B>1, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("spectra contain NaN/Inf.")
    expected_bands = int(state.get("raw_bands", -1))
    if values.shape[1] != expected_bands:
        raise ValueError(
            f"spectral vectors have {values.shape[1]} bands; preprocessing expects "
            f"{expected_bands}."
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    outputs: List[np.ndarray] = []
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        processed, _ = ApplyHSIPreprocessor(values[start:stop, None, :], state)
        processed = np.asarray(processed, dtype=np.float32)
        if processed.ndim != 3 or processed.shape[:2] != (stop - start, 1):
            raise RuntimeError(
                "ApplyHSIPreprocessor returned an invalid vector-cube shape: "
                f"{processed.shape}."
            )
        outputs.append(processed[:, 0, :])
    result = np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Processed spectral vectors contain NaN/Inf.")
    return result


def BuildPCSIRGInterventionCenters(
    raw_center_spectra: np.ndarray,
    preprocessing_state: Mapping[str, Any],
    *,
    gain_step: float = 0.03,
    tilt_step: float = 0.03,
    chunk_size: int = 4096,
    definition_version: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build paired center-only PC-SIRG interventions before preprocessing.

    Two fixed physical directions are used:

    1. relative multiplicative gain, ``s +/- a_g s``;
    2. smooth zero-mean wavelength tilt,
       ``s +/- a_t (t_lambda * s)``.

    The returned arrays are processed center vectors [N,2,C_model]. They are not
    raw spectra and are safe to inject lazily into model-input patches. The raw
    center spectra should be discarded after this function returns.
    """
    spectra = np.asarray(raw_center_spectra, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[1] <= 1:
        raise ValueError(
            f"raw_center_spectra must be [N,B] with B>1, got {spectra.shape}."
        )
    if not np.isfinite(spectra).all():
        raise ValueError("raw_center_spectra contain NaN/Inf.")
    steps = np.asarray([float(gain_step), float(tilt_step)], dtype=np.float32)
    if not np.isfinite(steps).all() or np.any(np.abs(steps) <= 1e-12):
        raise ValueError("gain_step and tilt_step must be finite and non-zero.")
    if np.any(np.abs(steps) >= 0.5):
        raise ValueError(
            "Intervention steps must remain small (<0.5) for a local central response."
        )

    n, bands = spectra.shape
    wavelength_axis = np.linspace(-1.0, 1.0, bands, dtype=np.float32)
    wavelength_axis -= float(wavelength_axis.mean())
    directions = np.stack(
        [spectra, spectra * wavelength_axis.reshape(1, -1)], axis=1
    )
    positive_raw = spectra[:, None, :] + steps[None, :, None] * directions
    negative_raw = spectra[:, None, :] - steps[None, :, None] * directions
    if not np.isfinite(positive_raw).all() or not np.isfinite(negative_raw).all():
        raise RuntimeError("Physical spectral interventions produced NaN/Inf.")

    positive = ApplyHSIPreprocessorToVectors(
        positive_raw.reshape(n * 2, bands),
        preprocessing_state,
        chunk_size=int(chunk_size),
    ).reshape(n, 2, -1)
    negative = ApplyHSIPreprocessorToVectors(
        negative_raw.reshape(n * 2, bands),
        preprocessing_state,
        chunk_size=int(chunk_size),
    ).reshape(n, 2, -1)

    metadata: Dict[str, Any] = {
        "definition_version": int(definition_version),
        "num_interventions": 2,
        "interventions": [
            {
                "index": 0,
                "name": "multiplicative_gain",
                "raw_rule": "s_plus_minus = s +/- a_g*s",
                "step": float(gain_step),
            },
            {
                "index": 1,
                "name": "smooth_wavelength_tilt",
                "raw_rule": "s_plus_minus = s +/- a_t*(zero_mean_wavelength_axis*s)",
                "step": float(tilt_step),
            },
        ],
        "center_pixel_only": True,
        "applied_before_preprocessing_and_pca": True,
        "ordered_wavelength_required": True,
        "persistent_form": "processed_positive_negative_center_vectors",
        "raw_spectra_are_model_inputs": False,
        "processed_center_shape": [int(v) for v in positive.shape],
        "raw_band_count": int(bands),
        "model_channel_count": int(positive.shape[-1]),
        "source_center_sha256": _array_sha256(spectra),
    }
    return positive, negative, steps, metadata


def BuildPCSIRGInterventionCentersFromCube(
    raw_hsi: np.ndarray,
    coords: np.ndarray,
    preprocessing_state: Mapping[str, Any],
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract ordered raw center spectra and build PC-SIRG interventions."""
    cube = np.asarray(raw_hsi, dtype=np.float32)
    coordinates = np.asarray(coords, dtype=np.int64)
    if cube.ndim != 3:
        raise ValueError(f"raw_hsi must be [H,W,B], got {cube.shape}.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coords must be [N,2], got {coordinates.shape}.")
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    if coordinates.size and (
        int(rows.min()) < 0
        or int(cols.min()) < 0
        or int(rows.max()) >= cube.shape[0]
        or int(cols.max()) >= cube.shape[1]
    ):
        raise ValueError("coords contain values outside raw_hsi.")
    centers = np.ascontiguousarray(cube[rows, cols, :], dtype=np.float32)
    return BuildPCSIRGInterventionCenters(
        centers, preprocessing_state, **kwargs
    )


def ValidatePCSIRGCenterAlignment(
    patches: np.ndarray,
    raw_center_spectra: np.ndarray,
    preprocessing_state: Mapping[str, Any],
    *,
    tolerance: float = 5e-4,
    chunk_size: int = 4096,
) -> float:
    """Verify that direct vector preprocessing matches extracted patch centers."""
    patch_array = np.asarray(patches, dtype=np.float32)
    if patch_array.ndim != 4:
        raise ValueError(f"patches must be [N,C,H,W], got {patch_array.shape}.")
    direct = ApplyHSIPreprocessorToVectors(
        raw_center_spectra, preprocessing_state, chunk_size=int(chunk_size)
    )
    center = patch_array[:, :, patch_array.shape[2] // 2, patch_array.shape[3] // 2]
    if center.shape != direct.shape:
        raise RuntimeError(
            f"Patch/direct center shapes differ: {center.shape} vs {direct.shape}."
        )
    error = float(np.max(np.abs(center - direct))) if center.size else 0.0
    if error > float(tolerance):
        raise RuntimeError(
            "Patch-center and direct spectral preprocessing disagree: "
            f"max_error={error:.6g}, tolerance={float(tolerance):.6g}."
        )
    return error

def SaveHSIPreprocessor(path: str, state: Mapping[str, Any]) -> str:
    """Persist preprocessing state as one portable NPZ file."""
    reduction = dict(state["reduction"])
    normalization = dict(state["normalization"])
    arrays = {
        "normalization_mean": np.asarray(
            normalization["mean"],
            dtype=np.float32,
        ),
        "normalization_std": np.asarray(
            normalization["std"],
            dtype=np.float32,
        ),
        "reduction_mean": np.asarray(
            reduction.get("mean", np.empty((0,), dtype=np.float32)),
            dtype=np.float32,
        ),
        "reduction_components": np.asarray(
            reduction.get(
                "components",
                np.empty((0, int(state["raw_bands"])), dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        "reduction_explained_variance": np.asarray(
            reduction.get(
                "explained_variance",
                np.empty((0,), dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        "reduction_explained_variance_ratio": np.asarray(
            reduction.get(
                "explained_variance_ratio",
                np.empty((0,), dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        "reduction_singular_values": np.asarray(
            reduction.get(
                "singular_values",
                np.empty((0,), dtype=np.float32),
            ),
            dtype=np.float32,
        ),
    }
    metadata = {
        "version": int(state["version"]),
        "fit_scope": str(state["fit_scope"]),
        "fit_pixel_count": int(state["fit_pixel_count"]),
        "spatial_shape": [int(v) for v in state["spatial_shape"]],
        "raw_bands": int(state["raw_bands"]),
        "normalization_eps": float(normalization["eps"]),
        "reduction_active": bool(reduction.get("active", False)),
        "reduction_method": str(reduction.get("method", "identity")),
        "reduction_input_bands": int(
            reduction.get("input_bands", state["raw_bands"])
        ),
        "reduction_n_components": int(
            reduction.get("n_components", state["raw_bands"])
        ),
        "reduction_whiten": bool(reduction.get("whiten", False)),
        "reduction_fit_samples": int(
            reduction.get("fit_samples", state["fit_pixel_count"])
        ),
    }

    import json

    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".tmp.npz"
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata)),
        **arrays,
    )
    os.replace(temporary, destination)
    return destination


def LoadHSIPreprocessor(path: str) -> Dict[str, Any]:
    """Load a state saved by :func:`SaveHSIPreprocessor`."""
    import json

    source = os.path.abspath(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        reduction_active = bool(metadata["reduction_active"])
        reduction: Dict[str, Any] = {
            "active": reduction_active,
            "method": str(metadata["reduction_method"]),
            "input_bands": int(metadata["reduction_input_bands"]),
            "n_components": int(metadata["reduction_n_components"]),
            "whiten": bool(metadata["reduction_whiten"]),
            "fit_samples": int(metadata["reduction_fit_samples"]),
        }
        if reduction_active:
            reduction.update(
                {
                    "mean": archive["reduction_mean"].astype(np.float32),
                    "components": archive[
                        "reduction_components"
                    ].astype(np.float32),
                    "explained_variance": archive[
                        "reduction_explained_variance"
                    ].astype(np.float32),
                    "explained_variance_ratio": archive[
                        "reduction_explained_variance_ratio"
                    ].astype(np.float32),
                    "singular_values": archive[
                        "reduction_singular_values"
                    ].astype(np.float32),
                }
            )

        return {
            "version": int(metadata["version"]),
            "fit_scope": str(metadata["fit_scope"]),
            "fit_pixel_count": int(metadata["fit_pixel_count"]),
            "spatial_shape": [int(v) for v in metadata["spatial_shape"]],
            "raw_bands": int(metadata["raw_bands"]),
            "normalization": {
                "mean": archive["normalization_mean"].astype(np.float32),
                "std": archive["normalization_std"].astype(np.float32),
                "eps": float(metadata["normalization_eps"]),
                "raw_bands": int(metadata["raw_bands"]),
            },
            "reduction": reduction,
        }


def _normalize_hsi(
    HSI: np.ndarray,
    *,
    fit_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Backward-compatible normalization helper.

    Strict CIL callers should pass an explicit base-training mask through
    :func:`FitHSIPreprocessor`; this helper remains for standalone utilities.
    """
    cube = np.asarray(HSI, dtype=np.float32)
    mask = (
        np.ones(cube.shape[:2], dtype=bool)
        if fit_mask is None
        else _validate_fit_mask(cube, fit_mask)
    )
    state = _normalization_state(cube, mask, eps=float(eps))
    return _apply_normalization_state(cube, state)


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================
def DimensionalityReduction(
    HSI: np.ndarray,
    method: str = "PCA",
    n_components: int = 30,
    *,
    whiten: bool = False,
    fit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Backward-compatible dimensionality-reduction helper.

    The input is assumed already normalized. Strict CIL callers should use
    :func:`FitHSIPreprocessor` with the explicit phase-0 training mask.
    """
    cube = np.asarray(HSI, dtype=np.float32)
    mask = (
        np.ones(cube.shape[:2], dtype=bool)
        if fit_mask is None
        else _validate_fit_mask(cube, fit_mask)
    )
    state = _fit_reduction_state(
        cube,
        fit_mask=mask,
        method=method,
        n_components=int(n_components),
        whiten=bool(whiten),
    )
    explained = float(
        np.asarray(
            state.get("explained_variance_ratio", []),
            dtype=np.float32,
        ).sum()
    )
    print(
        f"Applying {method}: {cube.shape[-1]} bands -> "
        f"{state['n_components']} components | "
        f"fit_pixels={int(mask.sum())}"
    )
    print(
        f"PCA explained variance on fit set: {explained:.2%} | "
        f"whiten={bool(whiten)}"
    )
    return _apply_reduction_state(cube, state)


# ============================================================
# DATA LOADING
# ============================================================
def LoadRawHSIData(
    method: str,
    base_dir: str = "./datasets",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    List[str],
    bool,
    Dict[str, Any],
]:
    """Load an unnormalized physical cube and resolved label policy."""
    if method not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset: {method}. Available: {list(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[method]
    data_path = os.path.join(base_dir, info["data_file"])
    gt_path = os.path.join(base_dir, info["gt_file"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    raw = _load_mat_key(data_path, info["data_key"]).astype(np.float32)
    gt = _load_mat_key(gt_path, info["gt_key"]).astype(np.int64)
    _validate_hsi_gt_shapes(raw, gt, method)
    policy = resolve_label_policy(method, gt)
    return (
        raw,
        gt,
        int(policy["num_classes"]),
        list(policy["target_names"]),
        bool(policy["has_background"]),
        policy,
    )


def LoadHSIData(
    method: str,
    base_dir: str = "./datasets",
    apply_reduction: bool = False,
    reduction_method: str = "PCA",
    n_components: int = 15,
    return_label_policy: bool = False,
    fit_preprocess_on_labeled: bool = True,
    return_raw_hsi: bool = False,
    *,
    preprocess_fit_mask: Optional[np.ndarray] = None,
    preprocess_fit_scope: str = "foreground",
    preprocess_state: Optional[Mapping[str, Any]] = None,
    return_preprocess_state: bool = False,
) -> Tuple[Any, ...]:
    """Load HSI data with an explicit, reusable preprocessing contract.

    Strict class-incremental use:
        1. call :func:`LoadRawHSIData`;
        2. call :func:`ExtractLabeledPixelIndex`;
        3. construct the protocol/splits;
        4. call :func:`BuildPreprocessFitMask` using phase-0 train indices;
        5. call this function with ``preprocess_fit_scope='base_train'`` and the
           explicit mask, or call :func:`FitHSIPreprocessor` directly.

    ``preprocess_fit_scope='base_train'`` without an explicit mask is rejected.
    This prevents silent fallback to all labeled/future-class pixels.
    """
    (
        raw,
        gt,
        num_classes,
        target_names,
        has_background,
        policy,
    ) = LoadRawHSIData(method, base_dir)

    scope = str(preprocess_fit_scope).strip().lower().replace("-", "_")
    if preprocess_state is not None:
        model_input, normalized_physical = ApplyHSIPreprocessor(
            raw,
            preprocess_state,
        )
        state = dict(preprocess_state)
        scope = str(state.get("fit_scope", "precomputed"))
    else:
        if preprocess_fit_mask is not None:
            fit_mask = _validate_fit_mask(raw, preprocess_fit_mask)
            if scope in {"", "foreground", "labeled", "all_labeled"}:
                scope = "explicit_mask"
        elif scope in {"base_train", "phase0_train", "phase_0_train"}:
            raise ValueError(
                "preprocess_fit_scope='base_train' requires an explicit "
                "preprocess_fit_mask built from phase-0 training indices."
            )
        elif scope in {"foreground", "labeled", "all_labeled"}:
            if not bool(fit_preprocess_on_labeled):
                raise ValueError(
                    "preprocess_fit_scope requests labeled pixels but "
                    "fit_preprocess_on_labeled is false."
                )
            fit_mask = _foreground_mask_from_policy(gt, policy)
            scope = "all_labeled"
            print(
                "[Preprocess:WARN] normalization/PCA are fitted on all labeled "
                "classes. This is backward-compatible but not strict CIL. "
                "Use preprocess_fit_scope='base_train' with an explicit mask."
            )
        elif scope in {"full", "full_scene", "scene"}:
            fit_mask = np.ones(raw.shape[:2], dtype=bool)
            scope = "full_scene"
        else:
            raise ValueError(
                "preprocess_fit_scope must be one of: base_train, foreground, "
                f"full_scene; got {preprocess_fit_scope!r}."
            )

        model_input, normalized_physical, state = FitHSIPreprocessor(
            raw,
            fit_mask=fit_mask,
            apply_reduction=bool(apply_reduction),
            reduction_method=reduction_method,
            n_components=int(n_components),
            whiten=False,
            fit_scope=scope,
        )

    if bool(apply_reduction) != bool(state["reduction"]["active"]):
        raise RuntimeError(
            "Requested apply_reduction does not match supplied preprocessing "
            "state."
        )

    gt_values = _valid_gt_values(gt)
    gt_min = int(gt_values.min()) if gt_values.size else -1
    gt_max = int(gt_values.max()) if gt_values.size else -1
    print(
        f"Loaded {method}: HSI={model_input.shape}, GT={gt.shape}, "
        f"GT(min,max)=({gt_min},{gt_max}), unique_valid={len(gt_values)}, "
        f"has_background={has_background}, "
        f"background_label={policy['background_label']}, "
        f"num_classes(train)={num_classes}, "
        f"raw_class_values={policy['raw_class_values']}, "
        f"preprocess_fit={scope}, "
        f"fit_pixels={int(state['fit_pixel_count'])}"
    )

    output: List[Any] = [
        model_input,
        gt,
        num_classes,
        target_names,
        has_background,
    ]
    if return_label_policy:
        output.append(policy)
    if return_raw_hsi:
        output.append(normalized_physical.astype(np.float32, copy=True))
    if return_preprocess_state:
        output.append(state)
    return tuple(output)


# ============================================================
# CENTER-SPECTRUM METADATA
# ============================================================
SPECTRAL_SPACES = {
    "raw_wavelength",
    "sensor_corrected_wavelength",
    "base_train_standardized_wavelength",
    "pca_components",
}
PHYSICAL_SPECTRAL_SPACES = {
    "raw_wavelength",
    "sensor_corrected_wavelength",
}


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def BuildCenterSpectralMetadata(
    center_spectra: np.ndarray,
    coords: np.ndarray,
    *,
    spectral_space: str,
    source: str = "center_pixel",
    preprocessing: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe and fingerprint per-sample center-spectrum metadata.

    Spectral-angle and wavelength-shape losses are permitted only for
    ``raw_wavelength`` and ``sensor_corrected_wavelength`` spaces. PCA vectors
    and standardized wavelength vectors remain useful metadata, but they must
    never be presented as physical spectra.
    """
    spectra = np.asarray(center_spectra, dtype=np.float32)
    coordinates = np.asarray(coords, dtype=np.int64)
    if spectra.ndim != 2:
        raise ValueError(f"center_spectra must be [N,B], got {spectra.shape}")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coords must be [N,2], got {coordinates.shape}")
    if len(spectra) != len(coordinates):
        raise ValueError(
            f"center_spectra/coords length mismatch: {len(spectra)} vs {len(coordinates)}"
        )
    if not np.isfinite(spectra).all():
        raise ValueError("center_spectra contain NaN or Inf values.")

    space = str(spectral_space).strip().lower()
    if space not in SPECTRAL_SPACES:
        raise ValueError(
            f"Unknown spectral_space={spectral_space!r}. Expected one of {sorted(SPECTRAL_SPACES)}."
        )
    physical = space in PHYSICAL_SPECTRAL_SPACES
    return {
        "version": 1,
        "space": space,
        "physical": bool(physical),
        "source": str(source),
        "sample_count": int(spectra.shape[0]),
        "band_count": int(spectra.shape[1]),
        "band_order_preserved": bool(space != "pca_components"),
        "pca_applied": bool(space == "pca_components"),
        "center_spectra_sha256": _array_sha256(spectra),
        "coords_sha256": _array_sha256(coordinates),
        "preprocessing": dict(preprocessing or {}),
    }


# ============================================================
# PATCH EXTRACTION
# ============================================================
def ImageCubes(
    HSI: np.ndarray,
    GT: np.ndarray,
    WS: int = 11,
    removeZeroLabels: bool = True,
    has_background: bool = True,
    num_classes: Optional[int] = None,
    pytorch_format: bool = True,
    label_policy: Optional[Dict[str, Any]] = None,
    return_center_spectra: bool = False,
    physical_hsi_for_center_spectra: Optional[np.ndarray] = None,
    *,
    raw_hsi_for_spectra: Optional[np.ndarray] = None,
    spectral_space: str = "raw_wavelength",
    return_spectral_metadata: bool = False,
) -> Tuple[Any, ...]:
    """Extract spectral-spatial patches and aligned center spectra.

    ``HSI`` is the model-input cube, normally the frozen PCA representation.
    ``physical_hsi_for_center_spectra`` is a separate wavelength-ordered cube
    used only for HSI spectral metadata. The deprecated ``raw_hsi_for_spectra``
    name remains as a compatibility alias.

    Return shapes:
        - default: ``patches, labels, coords``;
        - with center spectra: ``patches, labels, coords, center_spectra``;
        - with metadata: ``patches, labels, coords, center_spectra, metadata``.
    """
    if num_classes is None and label_policy is None:
        raise ValueError("ImageCubes requires num_classes or label_policy.")
    if HSI.ndim != 3 or GT.ndim != 2:
        raise ValueError(f"Expected HSI [H,W,B] and GT [H,W], got {HSI.shape}, {GT.shape}")
    if HSI.shape[:2] != GT.shape:
        raise ValueError(f"HSI/GT spatial mismatch: {HSI.shape[:2]} vs {GT.shape}")
    if int(WS) <= 0 or int(WS) % 2 == 0:
        raise ValueError(f"WS must be a positive odd integer, got {WS}")
    if bool(return_spectral_metadata) and not bool(return_center_spectra):
        raise ValueError("return_spectral_metadata=True requires return_center_spectra=True.")

    if physical_hsi_for_center_spectra is not None and raw_hsi_for_spectra is not None:
        raise ValueError(
            "Provide only physical_hsi_for_center_spectra; raw_hsi_for_spectra is a deprecated alias."
        )
    spectra_source = (
        physical_hsi_for_center_spectra
        if physical_hsi_for_center_spectra is not None
        else raw_hsi_for_spectra
    )
    if bool(return_center_spectra) and spectra_source is None:
        raise ValueError(
            "return_center_spectra=True requires a separate wavelength-ordered "
            "physical_hsi_for_center_spectra cube. Do not silently use PCA patch centers."
        )

    HSI = np.ascontiguousarray(HSI, dtype=np.float32)
    GT = np.asarray(GT, dtype=np.int64)
    center_source: Optional[np.ndarray] = None
    if spectra_source is not None:
        center_source = np.ascontiguousarray(spectra_source, dtype=np.float32)
        if center_source.ndim != 3:
            raise ValueError(
                f"physical_hsi_for_center_spectra must be [H,W,B], got {center_source.shape}"
            )
        if center_source.shape[:2] != GT.shape:
            raise ValueError(
                "physical_hsi_for_center_spectra spatial shape "
                f"{center_source.shape[:2]} must match GT {GT.shape}"
            )
        if not np.isfinite(center_source).all():
            raise ValueError("physical_hsi_for_center_spectra contains NaN or Inf values.")
        space = str(spectral_space).strip().lower()
        if space not in SPECTRAL_SPACES:
            raise ValueError(
                f"Unknown spectral_space={spectral_space!r}. Expected one of {sorted(SPECTRAL_SPACES)}."
            )

    num_rows, num_cols, num_bands = HSI.shape
    margin = int(WS) // 2
    padded_data = np.pad(
        HSI,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="reflect",
    )

    if label_policy is not None:
        raw_class_values = set(int(v) for v in label_policy["raw_class_values"])
        label_to_train = {
            int(k): int(v)
            for k, v in dict(label_policy["label_to_train"]).items()
        }
        num_classes = int(label_policy["num_classes"])
        has_background = bool(label_policy["has_background"])
        mask = (
            np.isin(GT, sorted(raw_class_values))
            if removeZeroLabels
            else (GT >= 0) & (GT != 255)
        )
    else:
        num_classes = int(num_classes)
        if removeZeroLabels:
            if has_background:
                mask = (GT > 0) & (GT <= num_classes)
                label_to_train = {raw: raw - 1 for raw in range(1, num_classes + 1)}
            else:
                mask = (GT >= 0) & (GT < num_classes)
                label_to_train = {raw: raw for raw in range(num_classes)}
        else:
            mask = (GT >= 0) & (GT != 255)
            label_to_train = {
                int(raw): int(raw)
                for raw in np.unique(GT[mask]).astype(int).tolist()
            }

    labeled_rows, labeled_cols = np.where(mask)
    num_labeled = int(len(labeled_rows))
    print(
        f"  Labeled pixels: {num_labeled} / {num_rows * num_cols} "
        f"({100 * num_labeled / max(num_rows * num_cols, 1):.1f}%)"
    )
    if num_labeled == 0:
        raise RuntimeError("No labeled pixels after applying label policy.")

    estimated_gb = num_labeled * int(WS) * int(WS) * num_bands * 4 / (1024 ** 3)
    print(f"  Allocating {estimated_gb:.2f} GiB for {num_labeled} patches")

    image_cubes = np.empty(
        (num_labeled, int(WS), int(WS), num_bands), dtype=np.float32
    )
    patch_labels = np.empty(num_labeled, dtype=np.int64)
    coords = np.empty((num_labeled, 2), dtype=np.int64)
    center_spectra: Optional[np.ndarray] = None
    if bool(return_center_spectra):
        assert center_source is not None
        center_spectra = np.empty(
            (num_labeled, int(center_source.shape[-1])), dtype=np.float32
        )

    for i, (r, c) in enumerate(zip(labeled_rows, labeled_cols)):
        pr, pc = int(r) + margin, int(c) + margin
        image_cubes[i] = padded_data[
            pr - margin : pr + margin + 1,
            pc - margin : pc + margin + 1,
            :,
        ]
        raw_label = int(GT[r, c])
        if raw_label not in label_to_train:
            raise RuntimeError(
                f"Raw label {raw_label} at {(int(r), int(c))} is not in label_to_train. "
                f"Policy: {label_to_train}"
            )
        patch_labels[i] = int(label_to_train[raw_label])
        coords[i] = [int(r), int(c)]
        if center_spectra is not None:
            center_spectra[i] = center_source[int(r), int(c), :]

    expected_min, expected_max = 0, int(num_classes) - 1
    if patch_labels.min() < expected_min or patch_labels.max() > expected_max:
        raise RuntimeError(
            f"Label range broken after mapping: [{patch_labels.min()}..{patch_labels.max()}], "
            f"expected [{expected_min}..{expected_max}] "
            f"(has_background={has_background}, removeZeroLabels={removeZeroLabels})"
        )

    present = sorted(int(c) for c in np.unique(patch_labels).tolist())
    missing = sorted(set(range(int(num_classes))) - set(present))
    if missing:
        print(f"[ImageCubes:WARN] Missing train-label classes after extraction: {missing}")
    counts = {int(c): int((patch_labels == c).sum()) for c in present}
    print(
        f"  ImageCubes: {len(patch_labels)} patches, "
        f"labels [{patch_labels.min()}-{patch_labels.max()}], "
        f"classes={present}, has_background={has_background}, num_classes={num_classes}"
    )
    print(f"  Class counts: {counts}")

    if pytorch_format:
        image_cubes = image_cubes.transpose(0, 3, 1, 2).copy()

    if center_spectra is None:
        return image_cubes, patch_labels, coords

    metadata = BuildCenterSpectralMetadata(
        center_spectra,
        coords,
        spectral_space=str(spectral_space),
        source="center_pixel",
    )
    if bool(return_spectral_metadata):
        return image_cubes, patch_labels, coords, center_spectra, metadata
    return image_cubes, patch_labels, coords, center_spectra


# ============================================================
# SIMPLE PYTORCH DATASET / DATALOADER FACTORY
# ============================================================
class HSICubeDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(patches) != len(labels):
            raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch, label


def _stratified_indices(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot split empty labels.")
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if not (0.0 <= float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if float(train_ratio) + float(val_ratio) >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, got {float(train_ratio) + float(val_ratio):.3f}"
        )

    train_idx, val_idx, test_idx = [], [], []

    for cls in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cls)[0]
        rng = np.random.RandomState(int(seed) + int(cls))
        idx = rng.permutation(idx)

        n = len(idx)
        if n == 1:
            train_idx.extend(idx.tolist())
            continue
        if n == 2:
            train_idx.append(int(idx[0]))
            test_idx.append(int(idx[1]))
            continue

        n_train = max(1, int(round(n * float(train_ratio))))
        n_val = max(1, int(round(n * float(val_ratio)))) if float(val_ratio) > 0.0 else 0

        if n_train + n_val >= n:
            overflow = n_train + n_val - (n - 1)
            reduce_val = min(overflow, max(0, n_val - 1))
            n_val -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                n_train = max(1, n_train - overflow)

        if n_val == 0 and n - n_train >= 2 and float(val_ratio) > 0.0:
            n_val = 1

        n_train = max(1, min(n_train, n - n_val - 1))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def create_dataloader(
    method: str,
    base_dir: str = "./datasets",
    WS: int = 11,
    batch_size: int = 64,
    train_ratio: float = 0.2,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    apply_reduction: bool = False,
    reduction_method: str = "PCA",
    n_components: int = 30,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    strict_cil: bool = False,
):
    if bool(strict_cil):
        raise RuntimeError(
            "create_dataloader() is a legacy standalone helper and is invalid for strict CIL. "
            "Use LoadRawHSIData -> protocol plan -> base-train fit mask -> "
            "FitHSIPreprocessor -> ImageCubes instead."
        )
    print(
        "[create_dataloader:WARN] using legacy non-incremental preprocessing. "
        "Do not use this helper for NECIL experiments."
    )
    HSI, GT, num_classes, target_names, has_bg, policy = LoadHSIData(
        method,
        base_dir,
        apply_reduction=apply_reduction,
        reduction_method=reduction_method,
        n_components=n_components,
        return_label_policy=True,
    )
    patches, labels, coords = ImageCubes(
        HSI,
        GT,
        WS=WS,
        removeZeroLabels=True,
        has_background=has_bg,
        num_classes=num_classes,
        pytorch_format=True,
        label_policy=policy,
    )

    print(f"Extracted {len(labels)} patches of shape {patches.shape}")

    train_idx, val_idx, test_idx = _stratified_indices(labels, train_ratio, val_ratio, seed)

    train_dataset = HSICubeDataset(patches[train_idx], labels[train_idx])
    val_dataset = HSICubeDataset(patches[val_idx], labels[val_idx])
    test_dataset = HSICubeDataset(patches[test_idx], labels[test_idx])

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    train_loader = DataLoader(train_dataset, shuffle=bool(shuffle), **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    info = {
        "num_classes": int(num_classes),
        "target_names": target_names,
        "num_bands": int(patches.shape[1]),
        "patch_size": int(WS),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "coords": coords,
        "label_policy": policy,
        "class_counts": {int(c): int((labels == c).sum()) for c in np.unique(labels).tolist()},
        "train_class_counts": {int(c): int((labels[train_idx] == c).sum()) for c in np.unique(labels).tolist()},
        "val_class_counts": {int(c): int((labels[val_idx] == c).sum()) for c in np.unique(labels).tolist()},
        "test_class_counts": {int(c): int((labels[test_idx] == c).sum()) for c in np.unique(labels).tolist()},
    }

    print(f"Split: Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']}")
    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch HSI DataLoader")
    print("=" * 60)

    dataset = "IP"
    base_dir = "./datasets"
    WS = 11

    try:
        HSI, GT, num_classes, target_names, has_bg, policy = LoadHSIData(
            dataset,
            base_dir,
            apply_reduction=True,
            reduction_method="PCA",
            n_components=30,
            return_label_policy=True,
        )
        print("\n1. Raw Data:")
        print(f"   HSI shape: {HSI.shape} (H, W, Bands)")
        print(f"   GT shape: {GT.shape}")
        print(f"   num_classes: {num_classes}")
        print(f"   has_background: {has_bg}")
        print(f"   raw_class_values: {policy['raw_class_values']}")

        patches, labels, coords = ImageCubes(
            HSI,
            GT,
            WS=WS,
            removeZeroLabels=True,
            has_background=has_bg,
            num_classes=num_classes,
            pytorch_format=True,
            label_policy=policy,
        )
        print("\n2. 3D Patches:")
        print(f"   Patches shape: {patches.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Coords shape: {coords.shape}")
        print(f"   Label range: [{labels.min()}..{labels.max()}]")
        print(f"   Classes: {np.unique(labels).tolist()}")

        assert labels.min() == 0, "Mapped train labels must start at 0."
        assert labels.max() == num_classes - 1, "Mapped train labels must end at num_classes-1."

        if has_bg:
            assert policy["background_label"] == 0, "Background datasets should explicitly ignore raw GT label 0."
            assert 0 not in policy["raw_class_values"], "Raw background label 0 leaked into foreground classes."

        print("\n[OK] All tests passed.")

    except FileNotFoundError as e:
        print(f"\n[File Error] {e}")
    except Exception as e:
        print(f"\n[Unexpected Error] {e}")
        raise






# """
# HSI Data Loader - PyTorch Version
# =================================

# Strict label-policy loader for NECIL-HSI.

# Contract with IncrementalHSIDataset:
# 1. Raw GT background/ignore labels are removed here.
# 2. Labels returned by ImageCubes are always sequential 0..K-1.
# 3. Raw class 0 is preserved only for datasets configured with has_background=False.
# 4. WHU-Hi HC/HH use raw 0 as background/ignore, not as a semantic class.

# This file intentionally does not implement incremental logic. It only loads HSI
# arrays, resolves label policy, performs optional spectral reduction, extracts
# patches, and returns sequential labels for the incremental dataset manager.
# """

# from __future__ import annotations

# import os
# from typing import Any, Dict, List, Optional, Tuple, Union

# import numpy as np
# import scipy.io as sio
# import torch
# from torch.utils.data import DataLoader, Dataset


# # ============================================================
# # DATASET CONFIGURATIONS
# # ============================================================
# DATASET_INFO: Dict[str, Dict[str, Any]] = {
#     "LK": {
#         "data_file": "WHU_Hi_LongKou.mat",
#         "data_key": "WHU_Hi_LongKou",
#         "gt_file": "WHU_Hi_LongKou_gt.mat",
#         "gt_key": "WHU_Hi_LongKou_gt",
#         "num_classes": 9,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
#             "Narrow-leaf soybean", "Rice", "Water",
#             "Roads and houses", "Mixed weed"
#         ],
#     },
#     "HH": {
#         "data_file": "WHU_Hi_HongHu.mat",
#         "data_key": "WHU_Hi_HongHu",
#         "gt_file": "WHU_Hi_HongHu_gt.mat",
#         "gt_key": "WHU_Hi_HongHu_gt",
#         "num_classes": 22,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Red roof", "Road", "Bare soil", "Cotton",
#             "Cotton firewood", "Rape", "Chinese cabbage",
#             "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis",
#             "Brassica chinensis", "Small Brassica chinensis", "Lactuca sativa",
#             "Celtuce", "Film covered lettuce", "Romaine lettuce",
#             "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree"
#         ],
#     },
#     "HC": {
#         "data_file": "WHU_Hi_HanChuan.mat",
#         "data_key": "WHU_Hi_HanChuan",
#         "gt_file": "WHU_Hi_HanChuan_gt.mat",
#         "gt_key": "WHU_Hi_HanChuan_gt",
#         "num_classes": 16,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Strawberry", "Cowpea", "Soybean", "Sorghum",
#             "Water spinach", "Watermelon", "Greens", "Trees", "Grass",
#             "Red roof", "Gray roof", "Plastic", "Bare soil", "Road",
#             "Bright object", "Water"
#         ],
#     },
#     "BS": {
#         "data_file": "Botswana.mat",
#         "data_key": "Botswana",
#         "gt_file": "Botswana_gt.mat",
#         "gt_key": "Botswana_gt",
#         "num_classes": 14,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Water", "Hippo Grass", "Floodplain Grasses 1", "Floodplain Grasses 2",
#             "Reeds 1", "Riparian", "Firescar 2", "Island Interior", "Woodlands",
#             "Acacia Shrublands", "Acacia Grasslands", "Short Mopane",
#             "Mixed Mopane", "Exposed Soils"
#         ],
#     },
#     "PU": {
#         "data_file": "PaviaU.mat",
#         "data_key": "paviaU",
#         "gt_file": "PaviaU_gt.mat",
#         "gt_key": "paviaU_gt",
#         "num_classes": 9,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Asphalt", "Meadows", "Gravel", "Trees", "Painted", "Soil",
#             "Bitumen", "Bricks", "Shadows"
#         ],
#     },
#     "PC": {
#         "data_file": "Pavia.mat",
#         "data_key": "pavia",
#         "gt_file": "Pavia_gt.mat",
#         "gt_key": "pavia_gt",
#         "num_classes": 9,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Water", "Trees", "Asphalt", "Bricks", "Bitumen",
#             "Tiles", "Shadows", "Meadows", "Soil"
#         ],
#     },
#     "SA": {
#         "data_file": "Salinas_corrected.mat",
#         "data_key": "salinas_corrected",
#         "gt_file": "Salinas_gt.mat",
#         "gt_key": "salinas_gt",
#         "num_classes": 16,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Weeds_1", "Weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth",
#             "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
#             "Corn_Weeds", "Lettuce_4wk", "Lettuce_5wk", "Lettuce_6wk",
#             "Lettuce_7wk", "Vinyard_untrained", "Vinyard_trellis"
#         ],
#     },
#     "UH13": {
#         "data_file": "HU13.mat",
#         "data_key": "HSI",
#         "gt_file": "HU13_gt.mat",
#         "gt_key": "gt",
#         "num_classes": 15,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Healthy grass", "Stressed grass", "Synthetic grass", "Trees",
#             "Soil", "Water", "Residential", "Commercial", "Road",
#             "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
#             "Tennis Court", "Running Track"
#         ],
#     },
#     "QUH": {
#         "data_file": "QUH-Qingyun.mat",
#         "data_key": "Chengqu",
#         "gt_file": "QUH-Qingyun_GT.mat",
#         "gt_key": "ChengquGT",
#         "num_classes": 6,
#         "has_background": False,
#         "background_label": None,
#         "target_names": [
#             "Trees", "Concrete building", "Car", "Ironhide building",
#             "Plastic playground", "Asphalt road"
#         ],
#     },
#     "PI": {
#         "data_file": "QUH-Pingan.mat",
#         "data_key": "Haigang",
#         "gt_file": "QUH-Pingan_GT.mat",
#         "gt_key": "HaigangGT",
#         "num_classes": 10,
#         "has_background": False,
#         "background_label": None,
#         "target_names": [
#             "Ship", "Seawater", "Trees", "Concrete structure building",
#             "Floating pier", "Brick houses", "Steel houses",
#             "Wharf construction land", "Car", "Road"
#         ],
#     },
#     "TH": {
#         "data_file": "QUH-Tangdaowan.mat",
#         "data_key": "Tangdaowan",
#         "gt_file": "QUH-Tangdaowan_GT.mat",
#         "gt_key": "TangdaowanGT",
#         "num_classes": 18,
#         "has_background": False,
#         "background_label": None,
#         "target_names": [
#             "Rubber track", "Flagging", "Sandy", "Asphalt", "Boardwalk", "Rocky shallows",
#             "Grassland", "Bulrush", "Gravel road", "Ligustrum vicaryi", "Coniferous pine",
#             "Spiraea", "Bare soil", "Buxus sinica", "Photinia serrulata", "Populus",
#             "Ulmus pumila L", "Seawater"
#         ],
#     },
#     "IP": {
#         "data_file": "Indian_pines_corrected.mat",
#         "data_key": "indian_pines_corrected",
#         "gt_file": "Indian_pines_gt.mat",
#         "gt_key": "indian_pines_gt",
#         "num_classes": 16,
#         "has_background": True,
#         "background_label": 0,
#         "target_names": [
#             "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
#             "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
#             "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
#             "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
#         ],
#     },
# }


# # ============================================================
# # VALIDATION / LABEL POLICY
# # ============================================================
# IGNORE_LABEL_VALUES = (255,)


# def _valid_gt_values(GT: np.ndarray, ignore_values=IGNORE_LABEL_VALUES) -> np.ndarray:
#     vals = np.unique(np.asarray(GT).astype(np.int64))
#     mask = vals >= 0
#     for ig in ignore_values:
#         mask &= vals != int(ig)
#     return vals[mask]


# def _load_mat_key(path: str, key: str) -> np.ndarray:
#     obj = sio.loadmat(path)
#     if key not in obj:
#         available = [k for k in obj.keys() if not k.startswith("__")]
#         raise KeyError(f"Key '{key}' not found in {path}. Available keys: {available}")
#     return obj[key]


# def _validate_hsi_gt_shapes(HSI: np.ndarray, GT: np.ndarray, method: str) -> None:
#     if HSI.ndim != 3:
#         raise ValueError(f"{method}: HSI must be [H,W,B], got shape={HSI.shape}")
#     if GT.ndim != 2:
#         raise ValueError(f"{method}: GT must be [H,W], got shape={GT.shape}")
#     if HSI.shape[:2] != GT.shape:
#         raise ValueError(f"{method}: HSI spatial shape {HSI.shape[:2]} does not match GT {GT.shape}")


# def resolve_label_policy(
#     method: str,
#     GT: np.ndarray,
#     *,
#     trust_config: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Resolve raw-label -> sequential-training-label mapping.

#     This function never treats raw GT 0 as background unless DATASET_INFO says
#     has_background=True. That is non-negotiable; otherwise class-0-real datasets
#     get corrupted before NECIL training starts.
#     """
#     del trust_config

#     if method not in DATASET_INFO:
#         raise ValueError(f"Unknown dataset: {method}. Available: {list(DATASET_INFO.keys())}")

#     info = DATASET_INFO[method]
#     gt_vals = _valid_gt_values(GT)
#     if gt_vals.size == 0:
#         raise ValueError("GT contains no valid labels.")

#     has_background = bool(info.get("has_background", True))
#     background_label = info.get("background_label", 0 if has_background else None)
#     num_classes_cfg = int(info["num_classes"])
#     target_names = list(info.get("target_names", []))

#     if "class_values" in info and info["class_values"] is not None:
#         class_values = sorted(int(v) for v in info["class_values"])
#     elif has_background:
#         class_values = sorted(int(v) for v in gt_vals.tolist() if int(v) != int(background_label))
#     else:
#         class_values = sorted(int(v) for v in gt_vals.tolist())

#         maybe_zero_background = (
#             0 in gt_vals.tolist()
#             and len(gt_vals) == num_classes_cfg + 1
#             and int(gt_vals.min()) == 0
#             and int(gt_vals.max()) == num_classes_cfg
#             and len(target_names) == num_classes_cfg
#         )
#         if maybe_zero_background:
#             raise RuntimeError(
#                 f"[LabelPolicy:ERROR] {method}: config says has_background=False, "
#                 f"but GT has values 0..{num_classes_cfg} and exactly {num_classes_cfg} "
#                 "target names. This would create a fake extra class. Set "
#                 "has_background=True and background_label=0."
#             )

#     if len(class_values) == 0:
#         raise RuntimeError(
#             f"No foreground classes resolved for {method}. "
#             f"has_background={has_background}, background_label={background_label}, gt_vals={gt_vals.tolist()}"
#         )

#     present = set(int(v) for v in gt_vals.tolist())
#     missing = [v for v in class_values if v not in present]
#     if missing:
#         print(f"[LabelPolicy:WARN] {method}: configured class values missing from GT: {missing}")

#     if len(class_values) != num_classes_cfg:
#         print(
#             f"[LabelPolicy:WARN] {method}: config num_classes={num_classes_cfg}, "
#             f"but resolved present class values={len(class_values)} -> {class_values}. "
#             "Using resolved class values."
#         )

#     label_to_train = {raw: i for i, raw in enumerate(class_values)}
#     train_to_label = {i: raw for raw, i in label_to_train.items()}

#     if len(target_names) > len(class_values):
#         target_names = target_names[:len(class_values)]
#     elif len(target_names) < len(class_values):
#         target_names = target_names + [f"Class {v}" for v in class_values[len(target_names):]]

#     return {
#         "method": method,
#         "has_background": has_background,
#         "background_label": background_label,
#         "raw_class_values": class_values,
#         "label_to_train": label_to_train,
#         "train_to_label": train_to_label,
#         "num_classes": len(class_values),
#         "target_names": target_names,
#         "gt_unique_values": gt_vals.tolist(),
#     }


# def _foreground_mask_from_policy(GT: np.ndarray, policy: Dict[str, Any]) -> np.ndarray:
#     raw_class_values = np.asarray(policy["raw_class_values"], dtype=np.int64)
#     if raw_class_values.size == 0:
#         raise ValueError("label_policy.raw_class_values is empty.")
#     return np.isin(np.asarray(GT, dtype=np.int64), raw_class_values)


# def _fit_pixels_from_mask(HSI: np.ndarray, fit_mask: Optional[np.ndarray]) -> np.ndarray:
#     flat = np.asarray(HSI, dtype=np.float32).reshape(-1, HSI.shape[-1])
#     if fit_mask is None:
#         return flat

#     mask = np.asarray(fit_mask, dtype=bool)
#     if mask.shape != HSI.shape[:2]:
#         raise ValueError(f"fit_mask shape {mask.shape} must match HSI spatial shape {HSI.shape[:2]}")

#     fit = flat[mask.reshape(-1)]
#     if fit.shape[0] == 0:
#         raise ValueError("No pixels available for normalization/PCA fitting after applying fit_mask.")
#     return fit


# def _normalize_hsi(
#     HSI: np.ndarray,
#     *,
#     fit_mask: Optional[np.ndarray] = None,
#     eps: float = 1e-8,
# ) -> np.ndarray:
#     """
#     Per-band z-score.

#     The mean/std are fitted on foreground/labeled pixels when fit_mask is given.
#     That is intentional for geometry NECIL: background-dominated statistics
#     flatten rare class spectral structure before the GeometryBank can model it.
#     """
#     HSI = np.asarray(HSI, dtype=np.float32)
#     fit_pixels = _fit_pixels_from_mask(HSI, fit_mask)
#     mean = fit_pixels.mean(axis=0, keepdims=True).reshape(1, 1, -1)
#     std = fit_pixels.std(axis=0, keepdims=True).reshape(1, 1, -1)
#     HSI = (HSI - mean) / (std + float(eps))
#     return np.nan_to_num(HSI, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# # ============================================================
# # DIMENSIONALITY REDUCTION
# # ============================================================
# def DimensionalityReduction(
#     HSI: np.ndarray,
#     method: str = "PCA",
#     n_components: int = 30,
#     *,
#     whiten: bool = False,
#     fit_mask: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#     """
#     Reduce spectral dimension.

#     PCA/iPCA is fitted on foreground/labeled pixels when fit_mask is supplied,
#     then applied to the full scene. Whitening stays disabled by default because
#     the GeometryBank explicitly models variance structure.
#     """
#     from sklearn.decomposition import IncrementalPCA, PCA

#     HSI = np.asarray(HSI, dtype=np.float32)
#     H, W, B = HSI.shape
#     n_components = int(n_components)
#     fit_pixels = _fit_pixels_from_mask(HSI, fit_mask)
#     max_components = min(B, fit_pixels.shape[0])

#     if n_components <= 0:
#         raise ValueError(f"n_components must be positive, got {n_components}")
#     if n_components > max_components:
#         print(f"[PCA:WARN] requested {n_components} components but max fit rank is {max_components}; clipping.")
#         n_components = max_components
#     if n_components == B:
#         print(f"[PCA:INFO] requested components equal input bands ({B}); returning normalized HSI without reduction.")
#         return HSI.astype(np.float32, copy=False)

#     method_norm = str(method).lower()
#     full_pixels = HSI.reshape(-1, B)

#     fit_scope = "foreground/labeled pixels" if fit_mask is not None else "full scene"
#     print(f"Applying {method}: {B} bands -> {n_components} components | fit={fit_scope}")

#     if method_norm == "pca":
#         pca = PCA(n_components=n_components, whiten=bool(whiten), svd_solver="auto")
#         pca.fit(fit_pixels)
#         reduced = pca.transform(full_pixels)
#         print(f"PCA explained variance on fit set: {pca.explained_variance_ratio_.sum():.2%} | whiten={bool(whiten)}")

#     elif method_norm == "ipca":
#         n_batches = min(256, max(1, fit_pixels.shape[0] // max(n_components * 20, 1)))
#         inc_pca = IncrementalPCA(n_components=n_components)
#         for batch in np.array_split(fit_pixels, n_batches):
#             if batch.shape[0] > 0:
#                 inc_pca.partial_fit(batch)
#         reduced = inc_pca.transform(full_pixels)
#         print(f"iPCA completed with {n_batches} fit batches | fit={fit_scope}")

#     else:
#         raise ValueError(f"Unknown reduction method: {method}. Use 'PCA' or 'iPCA'.")

#     reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
#     return reduced.reshape(H, W, n_components).astype(np.float32)


# # ============================================================
# # DATA LOADING
# # ============================================================
# def LoadHSIData(
#     method: str,
#     base_dir: str = "./datasets",
#     apply_reduction: bool = False,
#     reduction_method: str = "PCA",
#     n_components: int = 15,
#     return_label_policy: bool = False,
#     fit_preprocess_on_labeled: bool = True,
#     return_raw_hsi: bool = False,
# ) -> Union[
#     Tuple[np.ndarray, np.ndarray, int, List[str], bool],
#     Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any]],
#     Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any], np.ndarray],
# ]:
#     if method not in DATASET_INFO:
#         raise ValueError(f"Unknown dataset: {method}. Available: {list(DATASET_INFO.keys())}")

#     info = DATASET_INFO[method]
#     data_path = os.path.join(base_dir, info["data_file"])
#     gt_path = os.path.join(base_dir, info["gt_file"])

#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Data file not found: {data_path}")
#     if not os.path.exists(gt_path):
#         raise FileNotFoundError(f"GT file not found: {gt_path}")

#     HSI_raw_loaded = _load_mat_key(data_path, info["data_key"]).astype(np.float32)
#     GT = _load_mat_key(gt_path, info["gt_key"]).astype(np.int64)
#     _validate_hsi_gt_shapes(HSI_raw_loaded, GT, method)

#     policy = resolve_label_policy(method, GT)
#     fit_mask = _foreground_mask_from_policy(GT, policy) if bool(fit_preprocess_on_labeled) else None

#     # Normalize the physical wavelength-ordered cube first.
#     #
#     # IMPORTANT FOR REDUCED-BAND RUNS:
#     # When PCA/iPCA reduction is active, the model input has S=n_components
#     # channels.  In that regime we must NOT also return the original B-band
#     # physical cube as per-sample metadata, because the backbone band weights
#     # have dimension S while raw spectra have dimension B.  Passing both causes
#     # the exact failure:
#     #     band_weights spectral dim mismatch: S vs B
#     #
#     # Therefore return_raw_hsi only returns the physical cube when no spectral
#     # reduction is applied.  With reduction enabled, downstream datasets fall
#     # back to the reduced center vector from the patch itself.  That vector is
#     # useful as reduced-band/component metadata, but it is not wavelength-
#     # ordered physical spectra.
#     HSI_physical = _normalize_hsi(HSI_raw_loaded, fit_mask=fit_mask)
#     HSI = HSI_physical

#     if apply_reduction:
#         HSI = DimensionalityReduction(
#             HSI,
#             method=reduction_method,
#             n_components=n_components,
#             whiten=False,
#             fit_mask=fit_mask,
#         )
#         raw_hsi_physical = None
#         if bool(return_raw_hsi):
#             print(
#                 "[SRGP:INFO] Reduced-band mode active: not returning original raw spectra "
#                 "as metadata. Loader/dataset will use reduced center vectors instead."
#             )
#     else:
#         raw_hsi_physical = HSI_physical.astype(np.float32, copy=True) if bool(return_raw_hsi) else None

#     num_classes = int(policy["num_classes"])
#     target_names = list(policy["target_names"])
#     has_background = bool(policy["has_background"])

#     gt_vals_valid = _valid_gt_values(GT)
#     gt_min = int(gt_vals_valid.min()) if gt_vals_valid.size else -1
#     gt_max = int(gt_vals_valid.max()) if gt_vals_valid.size else -1
#     fit_scope = "foreground/labeled" if fit_mask is not None else "full-scene"

#     print(
#         f"Loaded {method}: HSI={HSI.shape}, GT={GT.shape}, "
#         f"GT(min,max)=({gt_min},{gt_max}), unique_valid={len(gt_vals_valid)}, "
#         f"has_background={has_background}, background_label={policy['background_label']}, "
#         f"num_classes(train)={num_classes}, raw_class_values={policy['raw_class_values']}, "
#         f"preprocess_fit={fit_scope}"
#     )

#     if return_label_policy and bool(return_raw_hsi):
#         return HSI, GT, num_classes, target_names, has_background, policy, raw_hsi_physical
#     if return_label_policy:
#         return HSI, GT, num_classes, target_names, has_background, policy
#     if bool(return_raw_hsi):
#         return HSI, GT, num_classes, target_names, has_background, raw_hsi_physical
#     return HSI, GT, num_classes, target_names, has_background


# # ============================================================
# # PATCH EXTRACTION
# # ============================================================
# def ImageCubes(
#     HSI: np.ndarray,
#     GT: np.ndarray,
#     WS: int = 11,
#     removeZeroLabels: bool = True,
#     has_background: bool = True,
#     num_classes: Optional[int] = None,
#     pytorch_format: bool = True,
#     label_policy: Optional[Dict[str, Any]] = None,
#     return_center_spectra: bool = False,
#     raw_hsi_for_spectra: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     if num_classes is None and label_policy is None:
#         raise ValueError("ImageCubes requires num_classes or label_policy.")
#     if HSI.ndim != 3 or GT.ndim != 2:
#         raise ValueError(f"Expected HSI [H,W,B] and GT [H,W], got {HSI.shape}, {GT.shape}")
#     if HSI.shape[:2] != GT.shape:
#         raise ValueError(f"HSI/GT spatial mismatch: {HSI.shape[:2]} vs {GT.shape}")
#     if int(WS) <= 0 or int(WS) % 2 == 0:
#         raise ValueError(f"WS must be a positive odd integer, got {WS}")

#     HSI = np.ascontiguousarray(HSI, dtype=np.float32)
#     GT = np.asarray(GT, dtype=np.int64)
#     raw_spectra_source = None
#     if raw_hsi_for_spectra is not None:
#         raw_spectra_source = np.ascontiguousarray(raw_hsi_for_spectra, dtype=np.float32)
#         if raw_spectra_source.shape[:2] != GT.shape:
#             raise ValueError(
#                 f"raw_hsi_for_spectra spatial shape {raw_spectra_source.shape[:2]} must match GT {GT.shape}"
#             )

#     num_rows, num_cols, num_bands = HSI.shape
#     margin = int(WS) // 2

#     padded_data = np.pad(
#         HSI,
#         ((margin, margin), (margin, margin), (0, 0)),
#         mode="reflect",
#     )

#     if label_policy is not None:
#         raw_class_values = set(int(v) for v in label_policy["raw_class_values"])
#         label_to_train = {int(k): int(v) for k, v in dict(label_policy["label_to_train"]).items()}
#         num_classes = int(label_policy["num_classes"])
#         has_background = bool(label_policy["has_background"])

#         if removeZeroLabels:
#             mask = np.isin(GT, sorted(raw_class_values))
#         else:
#             mask = (GT >= 0) & (GT != 255)
#     else:
#         num_classes = int(num_classes)
#         if removeZeroLabels:
#             if has_background:
#                 mask = (GT > 0) & (GT <= num_classes)
#                 label_to_train = {raw: raw - 1 for raw in range(1, num_classes + 1)}
#             else:
#                 mask = (GT >= 0) & (GT < num_classes)
#                 label_to_train = {raw: raw for raw in range(0, num_classes)}
#         else:
#             mask = (GT >= 0) & (GT != 255)
#             label_to_train = {int(raw): int(raw) for raw in np.unique(GT[mask]).astype(int).tolist()}

#     labeled_rows, labeled_cols = np.where(mask)
#     num_labeled = int(len(labeled_rows))

#     print(
#         f"  Labeled pixels: {num_labeled} / {num_rows * num_cols} "
#         f"({100 * num_labeled / max(num_rows * num_cols, 1):.1f}%)"
#     )

#     if num_labeled == 0:
#         raise RuntimeError("No labeled pixels after applying label policy.")

#     estimated_gb = num_labeled * int(WS) * int(WS) * num_bands * 4 / (1024 ** 3)
#     print(f"  Allocating {estimated_gb:.2f} GiB for {num_labeled} patches")

#     image_cubes = np.empty((num_labeled, int(WS), int(WS), num_bands), dtype=np.float32)
#     patch_labels = np.empty(num_labeled, dtype=np.int64)
#     coords = np.empty((num_labeled, 2), dtype=np.int64)
#     center_spectra = None
#     if bool(return_center_spectra) and raw_spectra_source is not None:
#         center_spectra = np.empty((num_labeled, int(raw_spectra_source.shape[-1])), dtype=np.float32)

#     for i, (r, c) in enumerate(zip(labeled_rows, labeled_cols)):
#         pr, pc = int(r) + margin, int(c) + margin
#         image_cubes[i] = padded_data[
#             pr - margin:pr + margin + 1,
#             pc - margin:pc + margin + 1,
#             :
#         ]

#         raw_label = int(GT[r, c])
#         if raw_label not in label_to_train:
#             raise RuntimeError(
#                 f"Raw label {raw_label} at {(int(r), int(c))} is not in label_to_train. "
#                 f"Policy: {label_to_train}"
#             )
#         patch_labels[i] = int(label_to_train[raw_label])
#         coords[i] = [int(r), int(c)]
#         if center_spectra is not None:
#             center_spectra[i] = raw_spectra_source[int(r), int(c), :]

#     expected_min, expected_max = 0, int(num_classes) - 1
#     if patch_labels.min() < expected_min or patch_labels.max() > expected_max:
#         raise RuntimeError(
#             f"Label range broken after mapping: [{patch_labels.min()}..{patch_labels.max()}], "
#             f"expected [{expected_min}..{expected_max}] "
#             f"(has_background={has_background}, removeZeroLabels={removeZeroLabels})"
#         )

#     present = sorted(int(c) for c in np.unique(patch_labels).tolist())
#     missing = sorted(set(range(int(num_classes))) - set(present))
#     if missing:
#         print(f"[ImageCubes:WARN] Missing train-label classes after extraction: {missing}")

#     counts = {int(c): int((patch_labels == c).sum()) for c in present}
#     print(
#         f"  ImageCubes: {len(patch_labels)} patches, "
#         f"labels [{patch_labels.min()}-{patch_labels.max()}], "
#         f"classes={present}, has_background={has_background}, num_classes={num_classes}"
#     )
#     print(f"  Class counts: {counts}")

#     if pytorch_format:
#         image_cubes = image_cubes.transpose(0, 3, 1, 2).copy()

#     if center_spectra is not None:
#         return image_cubes, patch_labels, coords, center_spectra
#     return image_cubes, patch_labels, coords


# # ============================================================
# # SIMPLE PYTORCH DATASET / DATALOADER FACTORY
# # ============================================================
# class HSICubeDataset(Dataset):
#     def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
#         patches = np.ascontiguousarray(patches, dtype=np.float32)
#         labels = np.asarray(labels, dtype=np.int64).reshape(-1)
#         if len(patches) != len(labels):
#             raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

#         self.patches = torch.from_numpy(patches).float()
#         self.labels = torch.from_numpy(labels).long()
#         self.transform = transform

#     def __len__(self):
#         return int(len(self.labels))

#     def __getitem__(self, idx):
#         patch = self.patches[idx]
#         label = self.labels[idx]
#         if self.transform:
#             patch = self.transform(patch)
#         return patch, label


# def _stratified_indices(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
#     labels = np.asarray(labels, dtype=np.int64).reshape(-1)
#     if labels.size == 0:
#         raise ValueError("Cannot split empty labels.")
#     if not (0.0 < float(train_ratio) < 1.0):
#         raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
#     if not (0.0 <= float(val_ratio) < 1.0):
#         raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
#     if float(train_ratio) + float(val_ratio) >= 1.0:
#         raise ValueError(
#             f"train_ratio + val_ratio must be < 1.0, got {float(train_ratio) + float(val_ratio):.3f}"
#         )

#     train_idx, val_idx, test_idx = [], [], []

#     for cls in sorted(np.unique(labels).tolist()):
#         idx = np.where(labels == cls)[0]
#         rng = np.random.RandomState(int(seed) + int(cls))
#         idx = rng.permutation(idx)

#         n = len(idx)
#         if n == 1:
#             train_idx.extend(idx.tolist())
#             continue
#         if n == 2:
#             train_idx.append(int(idx[0]))
#             test_idx.append(int(idx[1]))
#             continue

#         n_train = max(1, int(round(n * float(train_ratio))))
#         n_val = max(1, int(round(n * float(val_ratio)))) if float(val_ratio) > 0.0 else 0

#         if n_train + n_val >= n:
#             overflow = n_train + n_val - (n - 1)
#             reduce_val = min(overflow, max(0, n_val - 1))
#             n_val -= reduce_val
#             overflow -= reduce_val
#             if overflow > 0:
#                 n_train = max(1, n_train - overflow)

#         if n_val == 0 and n - n_train >= 2 and float(val_ratio) > 0.0:
#             n_val = 1

#         n_train = max(1, min(n_train, n - n_val - 1))
#         n_test = n - n_train - n_val
#         if n_test <= 0:
#             n_test = 1
#             if n_val > 0:
#                 n_val -= 1
#             else:
#                 n_train -= 1

#         train_idx.extend(idx[:n_train].tolist())
#         val_idx.extend(idx[n_train:n_train + n_val].tolist())
#         test_idx.extend(idx[n_train + n_val:].tolist())

#     return (
#         np.asarray(train_idx, dtype=np.int64),
#         np.asarray(val_idx, dtype=np.int64),
#         np.asarray(test_idx, dtype=np.int64),
#     )


# def create_dataloader(
#     method: str,
#     base_dir: str = "./datasets",
#     WS: int = 11,
#     batch_size: int = 64,
#     train_ratio: float = 0.2,
#     val_ratio: float = 0.1,
#     shuffle: bool = True,
#     seed: int = 42,
#     apply_reduction: bool = False,
#     reduction_method: str = "PCA",
#     n_components: int = 30,
#     num_workers: int = 0,
#     pin_memory: Optional[bool] = None,
# ):
#     HSI, GT, num_classes, target_names, has_bg, policy = LoadHSIData(
#         method,
#         base_dir,
#         apply_reduction=apply_reduction,
#         reduction_method=reduction_method,
#         n_components=n_components,
#         return_label_policy=True,
#     )
#     patches, labels, coords = ImageCubes(
#         HSI,
#         GT,
#         WS=WS,
#         removeZeroLabels=True,
#         has_background=has_bg,
#         num_classes=num_classes,
#         pytorch_format=True,
#         label_policy=policy,
#     )

#     print(f"Extracted {len(labels)} patches of shape {patches.shape}")

#     train_idx, val_idx, test_idx = _stratified_indices(labels, train_ratio, val_ratio, seed)

#     train_dataset = HSICubeDataset(patches[train_idx], labels[train_idx])
#     val_dataset = HSICubeDataset(patches[val_idx], labels[val_idx])
#     test_dataset = HSICubeDataset(patches[test_idx], labels[test_idx])

#     if pin_memory is None:
#         pin_memory = torch.cuda.is_available()

#     loader_kwargs = dict(
#         batch_size=int(batch_size),
#         num_workers=int(num_workers),
#         pin_memory=bool(pin_memory),
#     )
#     train_loader = DataLoader(train_dataset, shuffle=bool(shuffle), **loader_kwargs)
#     val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
#     test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

#     info = {
#         "num_classes": int(num_classes),
#         "target_names": target_names,
#         "num_bands": int(patches.shape[1]),
#         "patch_size": int(WS),
#         "n_train": int(len(train_idx)),
#         "n_val": int(len(val_idx)),
#         "n_test": int(len(test_idx)),
#         "coords": coords,
#         "label_policy": policy,
#         "class_counts": {int(c): int((labels == c).sum()) for c in np.unique(labels).tolist()},
#         "train_class_counts": {int(c): int((labels[train_idx] == c).sum()) for c in np.unique(labels).tolist()},
#         "val_class_counts": {int(c): int((labels[val_idx] == c).sum()) for c in np.unique(labels).tolist()},
#         "test_class_counts": {int(c): int((labels[test_idx] == c).sum()) for c in np.unique(labels).tolist()},
#     }

#     print(f"Split: Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']}")
#     return train_loader, val_loader, test_loader, info


# if __name__ == "__main__":
#     print("=" * 60)
#     print("Testing PyTorch HSI DataLoader")
#     print("=" * 60)

#     dataset = "IP"
#     base_dir = "./datasets"
#     WS = 11

#     try:
#         HSI, GT, num_classes, target_names, has_bg, policy = LoadHSIData(
#             dataset,
#             base_dir,
#             apply_reduction=True,
#             reduction_method="PCA",
#             n_components=30,
#             return_label_policy=True,
#         )
#         print("\n1. Raw Data:")
#         print(f"   HSI shape: {HSI.shape} (H, W, Bands)")
#         print(f"   GT shape: {GT.shape}")
#         print(f"   num_classes: {num_classes}")
#         print(f"   has_background: {has_bg}")
#         print(f"   raw_class_values: {policy['raw_class_values']}")

#         patches, labels, coords = ImageCubes(
#             HSI,
#             GT,
#             WS=WS,
#             removeZeroLabels=True,
#             has_background=has_bg,
#             num_classes=num_classes,
#             pytorch_format=True,
#             label_policy=policy,
#         )
#         print("\n2. 3D Patches:")
#         print(f"   Patches shape: {patches.shape}")
#         print(f"   Labels shape: {labels.shape}")
#         print(f"   Coords shape: {coords.shape}")
#         print(f"   Label range: [{labels.min()}..{labels.max()}]")
#         print(f"   Classes: {np.unique(labels).tolist()}")

#         assert labels.min() == 0, "Mapped train labels must start at 0."
#         assert labels.max() == num_classes - 1, "Mapped train labels must end at num_classes-1."

#         if has_bg:
#             assert policy["background_label"] == 0, "Background datasets should explicitly ignore raw GT label 0."
#             assert 0 not in policy["raw_class_values"], "Raw background label 0 leaked into foreground classes."

#         print("\n[OK] All tests passed.")

#     except FileNotFoundError as e:
#         print(f"\n[File Error] {e}")
#     except Exception as e:
#         print(f"\n[Unexpected Error] {e}")
#         raise
