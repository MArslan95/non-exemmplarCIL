"""
Configuration for NECIL-HSI Project
====================================
Contains dataset configurations and hyperparameters.
"""
"""
Configuration for NECIL-HSI Project
====================================
Stabilized for WHU-Hi HanChuan (HC) and physical grounding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# from dataclasses import dataclass, field
# from typing import Dict, List, Optional
# from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for a single HSI dataset."""
    name: str
    file_name: str
    mat_name: str
    gt_file_name: str
    gt_mat_name: str
    image_width: int
    image_height: int
    bands_num: int
    num_classes: int
    has_background: bool = True


# ============================================
# Dataset Configurations
# ============================================
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "Indian_pines": DatasetConfig(
        name="Indian_pines",
        file_name="Indian_pines_corrected.mat",
        mat_name="indian_pines_corrected",
        gt_file_name="Indian_pines_gt.mat",
        gt_mat_name="indian_pines_gt",
        image_width=145,
        image_height=145,
        bands_num=200,
        num_classes=16,
        has_background=True
    ),
    "PaviaU": DatasetConfig(
        name="PaviaU",
        file_name="PaviaU.mat",
        mat_name="paviaU",
        gt_file_name="PaviaU_gt.mat",
        gt_mat_name="paviaU_gt",
        image_width=610,
        image_height=340,
        bands_num=103,
        num_classes=9,
        has_background=True
    ),
    "PaviaC": DatasetConfig(
        name="PaviaC",
        file_name="Pavia.mat",
        mat_name="pavia",
        gt_file_name="Pavia_gt.mat",
        gt_mat_name="pavia_gt",
        image_width=1096,
        image_height=715,
        bands_num=102,
        num_classes=9,
        has_background=True
    ),
    "Salinas": DatasetConfig(
        name="Salinas",
        file_name="Salinas_corrected.mat",
        mat_name="salinas_corrected",
        gt_file_name="Salinas_gt.mat",
        gt_mat_name="salinas_gt",
        image_width=217,
        image_height=512,
        bands_num=204,
        num_classes=16,
        has_background=True
    ),
    "Houston13": DatasetConfig(
        name="Houston13",
        file_name="HU13.mat",
        mat_name="ori_data",  # Common key name, will auto-detect
        gt_file_name="HU13_gt.mat",
        gt_mat_name="map",  # Common key name, will auto-detect
        image_width=349,
        image_height=1905,
        bands_num=144,
        num_classes=15,
        has_background=True
    ),
    "KSC": DatasetConfig(
        name="KSC",
        file_name="KSC.mat",
        mat_name="KSC",
        gt_file_name="KSC_gt.mat",
        gt_mat_name="KSC_gt",
        image_width=512,
        image_height=614,
        bands_num=176,
        num_classes=13,
        has_background=True
    ),
    "Botswana": DatasetConfig(
        name="Botswana",
        file_name="Botswana.mat",
        mat_name="Botswana",
        gt_file_name="Botswana_gt.mat",
        gt_mat_name="Botswana_gt",
        image_width=1476,
        image_height=256,
        bands_num=145,
        num_classes=14,
        has_background=True
    ),
    "LK": DatasetConfig(
        name="LongKou",
        file_name="WHU_Hi_LongKou.mat",
        mat_name="WHU_Hi_LongKou",
        gt_file_name="WHU_Hi_LongKou_gt.mat",
        gt_mat_name="WHU_Hi_LongKou_gt",
        image_width=550,
        image_height=400,
        bands_num=270,
        num_classes=9,
        has_background=True  # Label 0 is Background (15k pixels) -> DELETE
    ),

    "HC": DatasetConfig(
        name="HanChuan",
        file_name="WHU_Hi_HanChuan.mat",
        mat_name="WHU_Hi_HanChuan",
        gt_file_name="WHU_Hi_HanChuan_gt.mat",
        gt_mat_name="WHU_Hi_HanChuan_gt",
        image_width=1217,
        image_height=303,
        bands_num=274,
        num_classes=16,
        has_background=True  # Label 0 is Background (111k pixels) -> DELETE
    ),

    "HH": DatasetConfig(
        name="HongHu",
        file_name="WHU_Hi_HongHu.mat",
        mat_name="WHU_Hi_HongHu",
        gt_file_name="WHU_Hi_HongHu_gt.mat",
        gt_mat_name="WHU_Hi_HongHu_gt",
        image_width=940,
        image_height=475,
        bands_num=270,
        num_classes=22,
        has_background=True  # Label 0 is Background (59k pixels) -> DELETE
    ),
}


@dataclass
class Config:
    """Main configuration for NECIL-HSI training."""
    
    # Dataset
    dataset_name: str = "Indian_pines"
    data_dir: str = "./datasets"
    # Patch extraction (Changed to 11 for better spatial context)
    patch_size: int = 11  
    
    # Model architecture
    d_model: int = 128  
    num_heads: int = 4  
    dropout: float = 0.1
    max_classes: int = 100
    
    # Incremental Stability
    memory_update_rate: float = 0.005 # Slow concept mixing
    old_grad_scale: float = 0.01      # Protect old concepts
    logit_scale_old: float = 1.1      # Boost old class priority
    
    # Training
    batch_size: int = 64
    epochs_base: int = 100  
    epochs_inc: int = 50  
    lr: float = 1e-4 # Lowered for stability
    
    # ----------------------------------------
    # Loss Weights (The Orchestrator)
    # ----------------------------------------
    lambda_ce: float = 1.0
    lambda_supcon: float = 0.5       
    
    lambda_cma: float = 0.3          
    lambda_stability: float = 1.0    
    lambda_cc: float = 0.1           
    lambda_distill: float = 1.0      
    
    # Physical Grounding (STRENGTHENED)
    lambda_spectral: float = 0.5     # Increased to lock DNA
    lambda_spatial: float = 0.2      # Increased for texture consistency
    lambda_contrastive: float = 0.1  
    lambda_separation: float = 0.2   

    device: str = "cuda"
    seed: int = 42
    # Logging
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    
    def get_dataset_config(self) -> DatasetConfig:
        """Get configuration for the selected dataset."""
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Available: {list(DATASET_CONFIGS.keys())}")
        return DATASET_CONFIGS[self.dataset_name]


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


# Incremental learning setup per dataset
INCREMENTAL_SETUP = {
    "Indian_pines": {"base_classes": 8, "increment": 2},  # 8 base + 4 phases of 2
    "PaviaU": {"base_classes": 5, "increment": 2},  # 5 base + 2 phases of 2
    "PaviaC": {"base_classes": 5, "increment": 2},  # 5 base + 2 phases of 2
    "Salinas": {"base_classes": 8, "increment": 2},  # 8 base + 4 phases of 2
    "Houston13": {"base_classes": 8, "increment": 2},  # 8 base + 3-4 phases
    "KSC": {"base_classes": 7, "increment": 2},  # 7 base + 3 phases of 2
    "Botswana": {"base_classes": 7, "increment": 2},  # 7 base + 3-4 phases
}
