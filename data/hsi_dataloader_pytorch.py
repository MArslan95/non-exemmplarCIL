from __future__ import annotations

"""HSI loading and preprocessing for class-incremental experiments.

The loader maps foreground labels once, keeps the processed spectral-spatial
cube consistent across phases, and exposes ordinary PyTorch dataset helpers.
The incremental protocol decides which samples and classes are available.
"""

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "IP": {
        "data_file": "Indian_pines_corrected.mat", "data_key": "indian_pines_corrected",
        "gt_file": "Indian_pines_gt.mat", "gt_key": "indian_pines_gt", "num_classes": 16,
        "target_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
            "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
        ],
    },
    "SA": {
        "data_file": "Salinas_corrected.mat", "data_key": "salinas_corrected",
        "gt_file": "Salinas_gt.mat", "gt_key": "salinas_gt", "num_classes": 16,
        "target_names": [
            "Weeds_1", "Weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth",
            "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
            "Corn_Weeds", "Lettuce_4wk", "Lettuce_5wk", "Lettuce_6wk",
            "Lettuce_7wk", "Vinyard_untrained", "Vinyard_trellis",
        ],
    },
    "PU": {
        "data_file": "PaviaU.mat", "data_key": "paviaU", "gt_file": "PaviaU_gt.mat",
        "gt_key": "paviaU_gt", "num_classes": 9,
        "target_names": ["Asphalt", "Meadows", "Gravel", "Trees", "Painted", "Soil", "Bitumen", "Bricks", "Shadows"],
    },
    "PC": {
        "data_file": "Pavia.mat", "data_key": "pavia", "gt_file": "Pavia_gt.mat",
        "gt_key": "pavia_gt", "num_classes": 9,
        "target_names": ["Water", "Trees", "Asphalt", "Bricks", "Bitumen", "Tiles", "Shadows", "Meadows", "Soil"],
    },
    "UH13": {
        "data_file": "HU13.mat", "data_key": "HSI", "gt_file": "HU13_gt.mat",
        "gt_key": "gt", "num_classes": 15,
        "target_names": [
            "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water",
            "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1",
            "Parking Lot 2", "Tennis Court", "Running Track",
        ],
    },
    "BS": {
        "data_file": "Botswana.mat", "data_key": "Botswana", "gt_file": "Botswana_gt.mat",
        "gt_key": "Botswana_gt", "num_classes": 14,
        "target_names": [
            "Water", "Hippo Grass", "Floodplain Grasses 1", "Floodplain Grasses 2", "Reeds 1",
            "Riparian", "Firescar 2", "Island Interior", "Woodlands", "Acacia Shrublands",
            "Acacia Grasslands", "Short Mopane", "Mixed Mopane", "Exposed Soils",
        ],
    },
    "LK": {
        "data_file": "WHU_Hi_LongKou.mat", "data_key": "WHU_Hi_LongKou",
        "gt_file": "WHU_Hi_LongKou_gt.mat", "gt_key": "WHU_Hi_LongKou_gt", "num_classes": 9,
        "target_names": ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean", "Rice", "Water", "Roads and houses", "Mixed weed"],
    },
    "HH": {
        "data_file": "WHU_Hi_HongHu.mat", "data_key": "WHU_Hi_HongHu",
        "gt_file": "WHU_Hi_HongHu_gt.mat", "gt_key": "WHU_Hi_HongHu_gt", "num_classes": 22,
        "target_names": [
            "Red roof", "Road", "Bare soil", "Cotton", "Cotton firewood", "Rape", "Chinese cabbage",
            "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis", "Brassica chinensis",
            "Small Brassica chinensis", "Lactuca sativa", "Celtuce", "Film covered lettuce",
            "Romaine lettuce", "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree",
        ],
    },
    "HC": {
        "data_file": "WHU_Hi_HanChuan.mat", "data_key": "WHU_Hi_HanChuan",
        "gt_file": "WHU_Hi_HanChuan_gt.mat", "gt_key": "WHU_Hi_HanChuan_gt", "num_classes": 16,
        "target_names": ["Strawberry", "Cowpea", "Soybean", "Sorghum", "Water spinach", "Watermelon", "Greens", "Trees", "Grass", "Red roof", "Gray roof", "Plastic", "Bare soil", "Road", "Bright object", "Water"],
    },
    "QUH": {
        "data_file": "QUH-Qingyun.mat", "data_key": "Chengqu", "gt_file": "QUH-Qingyun_GT.mat",
        "gt_key": "ChengquGT", "num_classes": 6, "background_label": None,
        "target_names": ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"],
    },
    "PI": {
        "data_file": "QUH-Pingan.mat", "data_key": "Haigang", "gt_file": "QUH-Pingan_GT.mat",
        "gt_key": "HaigangGT", "num_classes": 10, "background_label": None,
        "target_names": ["Ship", "Seawater", "Trees", "Concrete structure building", "Floating pier", "Brick houses", "Steel houses", "Wharf construction land", "Car", "Road"],
    },
    "TH": {
        "data_file": "QUH-Tangdaowan.mat", "data_key": "Tangdaowan", "gt_file": "QUH-Tangdaowan_GT.mat",
        "gt_key": "TangdaowanGT", "num_classes": 18, "background_label": None,
        "target_names": ["Rubber track", "Flagging", "Sandy", "Asphalt", "Boardwalk", "Rocky shallows", "Grassland", "Bulrush", "Gravel road", "Ligustrum vicaryi", "Coniferous pine", "Spiraea", "Bare soil", "Buxus sinica", "Photinia serrulata", "Populus", "Ulmus pumila L", "Seawater"],
    },
}

IGNORE_LABEL_VALUES = (255,)


def _load_mat_key(path: str, key: str) -> np.ndarray:
    payload = sio.loadmat(path)
    if key not in payload:
        available = sorted(name for name in payload if not name.startswith("__"))
        raise KeyError(f"{key!r} not found in {path}; available keys={available}")
    return payload[key]


def resolve_label_policy(method: str, gt: np.ndarray) -> Dict[str, Any]:
    if method not in DATASET_INFO:
        raise ValueError(f"unknown dataset {method!r}; available={sorted(DATASET_INFO)}")
    info = DATASET_INFO[method]
    values = [int(v) for v in np.unique(np.asarray(gt, dtype=np.int64)) if int(v) >= 0 and int(v) not in IGNORE_LABEL_VALUES]
    background = info.get("background_label", 0)
    classes = values if background is None else [value for value in values if value != background]
    classes = sorted(classes)
    expected = int(info["num_classes"])
    if len(classes) != expected:
        raise RuntimeError(f"{method}: expected {expected} foreground classes, found {classes}")
    names = [str(value) for value in info["target_names"]]
    if len(names) != expected:
        raise RuntimeError(f"{method}: target_names does not match num_classes")
    raw_to_class = {raw_label: class_id for class_id, raw_label in enumerate(classes)}
    return {
        "method": method, "background_label": background, "raw_class_values": classes,
        "label_to_train": raw_to_class,
        "train_to_label": {value: key for key, value in raw_to_class.items()},
        "num_classes": expected, "target_names": names,
    }


def ExtractLabeledPixelIndex(gt: np.ndarray, *, label_policy: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    labels_map = np.asarray(gt, dtype=np.int64)
    if labels_map.ndim != 2:
        raise ValueError("GT must have shape [H,W]")
    raw_classes = np.asarray(label_policy["raw_class_values"], dtype=np.int64)
    rows, cols = np.where(np.isin(labels_map, raw_classes))
    coords = np.stack((rows, cols), axis=1).astype(np.int64, copy=False)
    mapping = {int(key): int(value) for key, value in label_policy["label_to_train"].items()}
    labels = np.asarray([mapping[int(value)] for value in labels_map[rows, cols]], dtype=np.int64)
    return labels, coords


def BuildPreprocessFitMask(gt_shape: Sequence[int], coords: np.ndarray, sample_indices: Sequence[int], **_: Any) -> np.ndarray:
    shape = tuple(int(value) for value in gt_shape)
    coordinates = np.asarray(coords, dtype=np.int64)
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if len(shape) != 2 or min(shape) <= 0 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("gt_shape and coords are invalid")
    if indices.size == 0 or indices.min() < 0 or indices.max() >= len(coordinates):
        raise ValueError("sample_indices must select valid labelled pixels")
    if np.unique(indices).size != indices.size:
        raise ValueError("sample_indices cannot contain duplicates")
    if (
        np.any(coordinates[:, 0] < 0)
        or np.any(coordinates[:, 0] >= shape[0])
        or np.any(coordinates[:, 1] < 0)
        or np.any(coordinates[:, 1] >= shape[1])
    ):
        raise ValueError("coords lie outside gt_shape")
    mask = np.zeros(shape, dtype=bool)
    selected = coordinates[indices]
    mask[selected[:, 0], selected[:, 1]] = True
    return mask


def _fit_preprocessor(cube: np.ndarray, fit_mask: np.ndarray, *, n_components: Optional[int], whiten: bool) -> Dict[str, Any]:
    raw = np.asarray(cube, dtype=np.float32)
    mask = np.asarray(fit_mask, dtype=bool)
    if (
        raw.ndim != 3
        or raw.shape[2] == 0
        or not np.isfinite(raw).all()
        or mask.shape != raw.shape[:2]
        or not mask.any()
    ):
        raise ValueError("cube must be [H,W,B] and fit_mask must select pixels")
    pixels = raw[mask]
    mean = pixels.mean(axis=0).astype(np.float32)
    observed_std = pixels.std(axis=0)
    std = np.where(observed_std > 0.0, observed_std, 1.0).astype(np.float32)
    state: Dict[str, Any] = {
        "raw_bands": int(raw.shape[-1]), "normalization_mean": mean,
        "normalization_std": std, "fit_pixel_count": int(pixels.shape[0]),
        "pca_components": np.empty((0, raw.shape[-1]), dtype=np.float32),
        "pca_mean": np.empty(0, dtype=np.float32), "pca_variance": np.empty(0, dtype=np.float32),
        "whiten": bool(whiten), "fit_scope": "provided_mask",
    }
    if n_components is not None and int(n_components) <= 0:
        raise ValueError("n_components must be positive")
    if n_components is not None and int(n_components) < raw.shape[-1]:
        from sklearn.decomposition import PCA

        count = min(int(n_components), raw.shape[-1], pixels.shape[0])
        normalized_pixels = (pixels - mean) / std
        pca = PCA(n_components=count, whiten=False, svd_solver="auto").fit(normalized_pixels)
        state["pca_components"] = pca.components_.astype(np.float32)
        state["pca_mean"] = pca.mean_.astype(np.float32)
        state["pca_variance"] = pca.explained_variance_.astype(np.float32)
    return state


def ValidateHSIPreprocessorState(state: Mapping[str, Any]) -> bool:
    bands = int(state["raw_bands"])
    fit_pixel_count = int(state["fit_pixel_count"])
    mean = np.asarray(state["normalization_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(state["normalization_std"], dtype=np.float32).reshape(-1)
    components = np.asarray(state["pca_components"], dtype=np.float32)
    if bands <= 0 or fit_pixel_count <= 0:
        raise ValueError("preprocessor dimensions/count must be positive")
    if mean.shape != (bands,) or std.shape != (bands,):
        raise ValueError("normalization state has incompatible dimensions")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization state contains invalid values")
    if not str(state.get("fit_scope", "")).strip():
        raise ValueError("preprocessor state must record its fit_scope")
    if components.ndim != 2 or components.shape[1] != bands:
        raise ValueError("PCA components have an incompatible shape")
    if components.size:
        pca_mean = np.asarray(state["pca_mean"], dtype=np.float32).reshape(-1)
        variance = np.asarray(state["pca_variance"], dtype=np.float32).reshape(-1)
        if pca_mean.shape != (bands,) or variance.shape != (components.shape[0],):
            raise ValueError("PCA state has incompatible dimensions")
        if not np.isfinite(components).all() or not np.isfinite(variance).all():
            raise ValueError("PCA state contains NaN/Inf")
        if bool(state["whiten"]) and np.any(variance <= 0):
            raise ValueError("whitening requires positive PCA variance")
    return True


def ApplyHSIPreprocessor(HSI_raw: np.ndarray, state: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ValidateHSIPreprocessorState(state)
    raw = np.asarray(HSI_raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != int(state["raw_bands"]):
        raise ValueError("raw cube does not match preprocessing state")
    if not np.isfinite(raw).all():
        raise ValueError("raw cube contains NaN or Inf")
    mean = np.asarray(state["normalization_mean"], dtype=np.float32)
    std = np.asarray(state["normalization_std"], dtype=np.float32)
    processed = (raw - mean) / std
    components = np.asarray(state["pca_components"], dtype=np.float32)
    if components.size:
        flat = processed.reshape(-1, processed.shape[-1])
        transformed = (flat - np.asarray(state["pca_mean"], dtype=np.float32)) @ components.T
        if bool(state["whiten"]):
            transformed /= np.sqrt(np.asarray(state["pca_variance"], dtype=np.float32))
        processed = transformed.reshape(raw.shape[0], raw.shape[1], -1)
    processed = np.asarray(processed, dtype=np.float32)
    if not np.isfinite(processed).all():
        raise RuntimeError("preprocessing produced NaN/Inf")
    return np.ascontiguousarray(processed), np.ascontiguousarray(raw)


def FitHSIPreprocessor(
    HSI_raw: np.ndarray, *, fit_mask: Optional[np.ndarray] = None,
    apply_reduction: bool = True, reduction_method: str = "PCA",
    n_components: int = 30, whiten: bool = False, **_: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    use_reduction = bool(apply_reduction)
    if use_reduction and str(reduction_method).strip().lower() != "pca":
        raise ValueError("only PCA reduction is supported")
    if bool(whiten) and not use_reduction:
        raise ValueError("whiten=True requires PCA reduction")
    raw = np.asarray(HSI_raw, dtype=np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError("HSI_raw must be finite [H,W,B]")
    if fit_mask is None:
        raise ValueError(
            "fit_mask is required; preprocessing statistics must be fitted on "
            "the phase-0 training split"
        )
    mask = np.asarray(fit_mask, dtype=bool)
    state = _fit_preprocessor(
        raw,
        mask,
        n_components=int(n_components) if use_reduction else None,
        whiten=bool(whiten),
    )
    processed, ordered = ApplyHSIPreprocessor(raw, state)
    return processed, ordered, state


def FitHSIPreprocessorFromProtocol(
    HSI_raw: np.ndarray, *, coords: np.ndarray, protocol: Mapping[str, Any],
    fit_scope: str = "base_train", **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    scope = str(fit_scope).strip().lower()
    if scope != "base_train":
        raise ValueError(
            "fit_scope must be 'base_train'; fitting on all pixels leaks "
            "validation/test information"
        )
    fit_mask = BuildPreprocessFitMask(
        HSI_raw.shape[:2],
        coords,
        protocol["base_train_indices"],
    )
    processed, ordered, state = FitHSIPreprocessor(
        HSI_raw,
        fit_mask=fit_mask,
        **kwargs,
    )
    state["fit_scope"] = "base_train"
    return processed, ordered, state


def SaveHSIPreprocessor(path: str, state: Mapping[str, Any]) -> str:
    ValidateHSIPreprocessorState(state)
    destination = os.path.abspath(path)
    if os.path.dirname(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as stream:
        np.savez_compressed(
            stream, raw_bands=np.asarray(int(state["raw_bands"])),
            fit_pixel_count=np.asarray(int(state["fit_pixel_count"])),
            normalization_mean=np.asarray(state["normalization_mean"], dtype=np.float32),
            normalization_std=np.asarray(state["normalization_std"], dtype=np.float32),
            pca_components=np.asarray(state["pca_components"], dtype=np.float32),
            pca_mean=np.asarray(state["pca_mean"], dtype=np.float32),
            pca_variance=np.asarray(state["pca_variance"], dtype=np.float32),
            whiten=np.asarray(bool(state["whiten"])),
            fit_scope=np.asarray(str(state["fit_scope"])),
        )
    return destination


def LoadHSIPreprocessor(path: str) -> Dict[str, Any]:
    with np.load(os.path.abspath(path), allow_pickle=False) as archive:
        state = {
            "raw_bands": int(archive["raw_bands"].item()),
            "fit_pixel_count": int(archive["fit_pixel_count"].item()),
            "normalization_mean": archive["normalization_mean"].astype(np.float32),
            "normalization_std": archive["normalization_std"].astype(np.float32),
            "pca_components": archive["pca_components"].astype(np.float32),
            "pca_mean": archive["pca_mean"].astype(np.float32),
            "pca_variance": archive["pca_variance"].astype(np.float32),
            "whiten": bool(archive["whiten"].item()),
            "fit_scope": str(archive["fit_scope"].item()),
        }
    ValidateHSIPreprocessorState(state)
    return state


def LoadRawHSIData(method: str, base_dir: str = "./datasets") -> Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any]]:
    if method not in DATASET_INFO:
        raise ValueError(f"unknown dataset {method!r}; available={sorted(DATASET_INFO)}")
    info = DATASET_INFO[method]
    data_path, gt_path = os.path.join(base_dir, info["data_file"]), os.path.join(base_dir, info["gt_file"])
    if not os.path.isfile(data_path):
        raise FileNotFoundError(data_path)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    raw = np.asarray(_load_mat_key(data_path, info["data_key"]), dtype=np.float32)
    gt = np.asarray(_load_mat_key(gt_path, info["gt_key"]), dtype=np.int64)
    if raw.ndim != 3 or gt.ndim != 2 or raw.shape[:2] != gt.shape:
        raise ValueError(f"incompatible HSI/GT shapes: {raw.shape}, {gt.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError("HSI contains NaN/Inf")
    policy = resolve_label_policy(method, gt)
    return raw, gt, policy["num_classes"], policy["target_names"], policy["background_label"] is not None, policy


class HSICubeDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray) -> None:
        x, y = np.ascontiguousarray(patches, dtype=np.float32), np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 4 or len(x) == 0:
            raise ValueError("patches must be non-empty rank-four data")
        if len(x) != len(y):
            raise ValueError("patch and label counts differ")
        if not np.isfinite(x).all() or np.any(y < 0):
            raise ValueError("patches/labels contain invalid values")
        self.patches, self.labels = torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.patches[index], self.labels[index]


def ImageCubes(
    HSI: np.ndarray, GT: np.ndarray, WS: int = 11, *,
    label_policy: Mapping[str, Any], pytorch_format: bool = True, **_: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube = np.asarray(HSI, dtype=np.float32)
    ground_truth = np.asarray(GT)
    if (
        cube.ndim != 3
        or cube.shape[2] == 0
        or ground_truth.ndim != 2
        or cube.shape[:2] != ground_truth.shape
    ):
        raise ValueError("HSI must be [H,W,B] and align with rank-two GT")
    if not np.isfinite(cube).all():
        raise ValueError("HSI contains NaN or Inf")
    labels, coords = ExtractLabeledPixelIndex(GT, label_policy=label_policy)
    size = int(WS)
    if size <= 0 or size % 2 == 0:
        raise ValueError("WS must be positive and odd")
    radius = size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    patches = np.empty((len(labels), size, size, cube.shape[-1]), dtype=np.float32)
    for index, (row, col) in enumerate(coords):
        patches[index] = padded[row:row + size, col:col + size]
    if pytorch_format:
        patches = np.moveaxis(patches, -1, 1).copy()
    return patches, labels, coords


def create_dataloader(
    patches: np.ndarray, labels: np.ndarray, *, batch_size: int = 64,
    shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False, **_: Any,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(num_workers) < 0:
        raise ValueError("num_workers cannot be negative")
    return DataLoader(
        HSICubeDataset(patches, labels), batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
    )


__all__ = [
    "DATASET_INFO", "LoadRawHSIData", "resolve_label_policy", "ExtractLabeledPixelIndex",
    "BuildPreprocessFitMask", "FitHSIPreprocessor", "FitHSIPreprocessorFromProtocol",
    "ApplyHSIPreprocessor", "ValidateHSIPreprocessorState", "SaveHSIPreprocessor",
    "LoadHSIPreprocessor", "ImageCubes", "HSICubeDataset", "create_dataloader",
]
