"""
HSI Data Loader - PyTorch Version
=================================

Strict label-policy loader for NECIL-HSI.

Contract with IncrementalHSIDataset:
1. Raw GT background/ignore labels are removed here.
2. Labels returned by ImageCubes are always sequential 0..K-1.
3. Raw class 0 is preserved only for datasets configured with has_background=False.
4. WHU-Hi HC/HH use raw 0 as background/ignore, not as a semantic class.

This file intentionally does not implement incremental logic. It only loads HSI
arrays, resolves label policy, performs optional spectral reduction, extracts
patches, and returns sequential labels for the incremental dataset manager.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _normalize_hsi(
    HSI: np.ndarray,
    *,
    fit_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Per-band z-score.

    The mean/std are fitted on foreground/labeled pixels when fit_mask is given.
    That is intentional for geometry NECIL: background-dominated statistics
    flatten rare class spectral structure before the GeometryBank can model it.
    """
    HSI = np.asarray(HSI, dtype=np.float32)
    fit_pixels = _fit_pixels_from_mask(HSI, fit_mask)
    mean = fit_pixels.mean(axis=0, keepdims=True).reshape(1, 1, -1)
    std = fit_pixels.std(axis=0, keepdims=True).reshape(1, 1, -1)
    HSI = (HSI - mean) / (std + float(eps))
    return np.nan_to_num(HSI, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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
    """
    Reduce spectral dimension.

    PCA/iPCA is fitted on foreground/labeled pixels when fit_mask is supplied,
    then applied to the full scene. Whitening stays disabled by default because
    the GeometryBank explicitly models variance structure.
    """
    from sklearn.decomposition import IncrementalPCA, PCA

    HSI = np.asarray(HSI, dtype=np.float32)
    H, W, B = HSI.shape
    n_components = int(n_components)
    fit_pixels = _fit_pixels_from_mask(HSI, fit_mask)
    max_components = min(B, fit_pixels.shape[0])

    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if n_components > max_components:
        print(f"[PCA:WARN] requested {n_components} components but max fit rank is {max_components}; clipping.")
        n_components = max_components
    if n_components == B:
        print(f"[PCA:INFO] requested components equal input bands ({B}); returning normalized HSI without reduction.")
        return HSI.astype(np.float32, copy=False)

    method_norm = str(method).lower()
    full_pixels = HSI.reshape(-1, B)

    fit_scope = "foreground/labeled pixels" if fit_mask is not None else "full scene"
    print(f"Applying {method}: {B} bands -> {n_components} components | fit={fit_scope}")

    if method_norm == "pca":
        pca = PCA(n_components=n_components, whiten=bool(whiten), svd_solver="auto")
        pca.fit(fit_pixels)
        reduced = pca.transform(full_pixels)
        print(f"PCA explained variance on fit set: {pca.explained_variance_ratio_.sum():.2%} | whiten={bool(whiten)}")

    elif method_norm == "ipca":
        n_batches = min(256, max(1, fit_pixels.shape[0] // max(n_components * 20, 1)))
        inc_pca = IncrementalPCA(n_components=n_components)
        for batch in np.array_split(fit_pixels, n_batches):
            if batch.shape[0] > 0:
                inc_pca.partial_fit(batch)
        reduced = inc_pca.transform(full_pixels)
        print(f"iPCA completed with {n_batches} fit batches | fit={fit_scope}")

    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'PCA' or 'iPCA'.")

    reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
    return reduced.reshape(H, W, n_components).astype(np.float32)


# ============================================================
# DATA LOADING
# ============================================================
def LoadHSIData(
    method: str,
    base_dir: str = "./datasets",
    apply_reduction: bool = False,
    reduction_method: str = "PCA",
    n_components: int = 15,
    return_label_policy: bool = False,
    fit_preprocess_on_labeled: bool = True,
    return_raw_hsi: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, int, List[str], bool],
    Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any]],
    Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any], np.ndarray],
]:
    if method not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {method}. Available: {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[method]
    data_path = os.path.join(base_dir, info["data_file"])
    gt_path = os.path.join(base_dir, info["gt_file"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    HSI_raw_loaded = _load_mat_key(data_path, info["data_key"]).astype(np.float32)
    GT = _load_mat_key(gt_path, info["gt_key"]).astype(np.int64)
    _validate_hsi_gt_shapes(HSI_raw_loaded, GT, method)

    policy = resolve_label_policy(method, GT)
    fit_mask = _foreground_mask_from_policy(GT, policy) if bool(fit_preprocess_on_labeled) else None

    # Normalize the physical wavelength-ordered cube first.
    #
    # SCB-GR / SRGP contract:
    #   1) HSI is the model input cube.  If PCA/iPCA is active, this becomes the
    #      reduced/component cube used by the backbone.
    #   2) raw_hsi_physical is optional metadata only.  It keeps the normalized
    #      original wavelength-ordered cube so ImageCubes can return raw center
    #      spectra for GeometryBank spectral-shape/risk descriptors.
    #
    # Do NOT feed raw_hsi_physical into the backbone band-weight branch.  Its
    # spectral dimension can differ from the PCA input dimension by design.
    HSI_physical = _normalize_hsi(HSI_raw_loaded, fit_mask=fit_mask)
    HSI = HSI_physical
    raw_hsi_physical = HSI_physical.astype(np.float32, copy=True) if bool(return_raw_hsi) else None

    if apply_reduction:
        HSI = DimensionalityReduction(
            HSI,
            method=reduction_method,
            n_components=n_components,
            whiten=False,
            fit_mask=fit_mask,
        )
        if bool(return_raw_hsi):
            print(
                "[SCB-GR:INFO] Reduced-band mode active: model input uses reduced "
                "components, while normalized raw physical spectra are returned as "
                "metadata for GeometryBank spectral-shape/risk only."
            )

    num_classes = int(policy["num_classes"])
    target_names = list(policy["target_names"])
    has_background = bool(policy["has_background"])

    gt_vals_valid = _valid_gt_values(GT)
    gt_min = int(gt_vals_valid.min()) if gt_vals_valid.size else -1
    gt_max = int(gt_vals_valid.max()) if gt_vals_valid.size else -1
    fit_scope = "foreground/labeled" if fit_mask is not None else "full-scene"

    print(
        f"Loaded {method}: HSI={HSI.shape}, GT={GT.shape}, "
        f"GT(min,max)=({gt_min},{gt_max}), unique_valid={len(gt_vals_valid)}, "
        f"has_background={has_background}, background_label={policy['background_label']}, "
        f"num_classes(train)={num_classes}, raw_class_values={policy['raw_class_values']}, "
        f"preprocess_fit={fit_scope}"
    )

    if return_label_policy and bool(return_raw_hsi):
        return HSI, GT, num_classes, target_names, has_background, policy, raw_hsi_physical
    if return_label_policy:
        return HSI, GT, num_classes, target_names, has_background, policy
    if bool(return_raw_hsi):
        return HSI, GT, num_classes, target_names, has_background, raw_hsi_physical
    return HSI, GT, num_classes, target_names, has_background


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
    raw_hsi_for_spectra: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_classes is None and label_policy is None:
        raise ValueError("ImageCubes requires num_classes or label_policy.")
    if HSI.ndim != 3 or GT.ndim != 2:
        raise ValueError(f"Expected HSI [H,W,B] and GT [H,W], got {HSI.shape}, {GT.shape}")
    if HSI.shape[:2] != GT.shape:
        raise ValueError(f"HSI/GT spatial mismatch: {HSI.shape[:2]} vs {GT.shape}")
    if int(WS) <= 0 or int(WS) % 2 == 0:
        raise ValueError(f"WS must be a positive odd integer, got {WS}")

    HSI = np.ascontiguousarray(HSI, dtype=np.float32)
    GT = np.asarray(GT, dtype=np.int64)
    raw_spectra_source = None
    if raw_hsi_for_spectra is not None:
        raw_spectra_source = np.ascontiguousarray(raw_hsi_for_spectra, dtype=np.float32)
        if raw_spectra_source.shape[:2] != GT.shape:
            raise ValueError(
                f"raw_hsi_for_spectra spatial shape {raw_spectra_source.shape[:2]} must match GT {GT.shape}"
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
        label_to_train = {int(k): int(v) for k, v in dict(label_policy["label_to_train"]).items()}
        num_classes = int(label_policy["num_classes"])
        has_background = bool(label_policy["has_background"])

        if removeZeroLabels:
            mask = np.isin(GT, sorted(raw_class_values))
        else:
            mask = (GT >= 0) & (GT != 255)
    else:
        num_classes = int(num_classes)
        if removeZeroLabels:
            if has_background:
                mask = (GT > 0) & (GT <= num_classes)
                label_to_train = {raw: raw - 1 for raw in range(1, num_classes + 1)}
            else:
                mask = (GT >= 0) & (GT < num_classes)
                label_to_train = {raw: raw for raw in range(0, num_classes)}
        else:
            mask = (GT >= 0) & (GT != 255)
            label_to_train = {int(raw): int(raw) for raw in np.unique(GT[mask]).astype(int).tolist()}

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

    image_cubes = np.empty((num_labeled, int(WS), int(WS), num_bands), dtype=np.float32)
    patch_labels = np.empty(num_labeled, dtype=np.int64)
    coords = np.empty((num_labeled, 2), dtype=np.int64)
    center_spectra = None
    if bool(return_center_spectra) and raw_spectra_source is not None:
        center_spectra = np.empty((num_labeled, int(raw_spectra_source.shape[-1])), dtype=np.float32)

    for i, (r, c) in enumerate(zip(labeled_rows, labeled_cols)):
        pr, pc = int(r) + margin, int(c) + margin
        image_cubes[i] = padded_data[
            pr - margin:pr + margin + 1,
            pc - margin:pc + margin + 1,
            :
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
            center_spectra[i] = raw_spectra_source[int(r), int(c), :]

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

    if center_spectra is not None:
        return image_cubes, patch_labels, coords, center_spectra
    return image_cubes, patch_labels, coords


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
):
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
