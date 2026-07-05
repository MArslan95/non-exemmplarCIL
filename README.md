# Geometry-Native NECIL-HSI: Non-Exemplar Class-Incremental Learning for Hyperspectral Image Classification

A PyTorch implementation of a **Geometry-Native Non-Exemplar Class-Incremental Learning framework for Hyperspectral Image Classification**.

This project does **not** store old exemplars. Instead, each class is represented through a reliability-aware geometric memory consisting of class means, low-rank subspaces, variance estimates, spectral prototypes, band-importance statistics, and concept anchors. Incremental learning is performed using geometry replay, geometry alignment, old/new separation, and calibrated geometry-based classification.

---

## Core Idea

Most class-incremental HSI methods either store old samples or rely on simple prototype memory. This project follows a stricter non-exemplar setting:

```text
No old raw pixels.
No old image patches.
No exemplar buffer.

Class c memory:
  ├── mean vector
  ├── low-rank basis
  ├── eigenvalue / variance profile
  ├── residual variance
  ├── active rank
  ├── reliability score
  ├── spectral prototype
  ├── band importance
  └── concept anchors



  HSI Cube
  ↓
PCA / Spectral Reduction
  ↓
Patch Extraction
  ↓
Spectral-Spatial Backbone
  ↓
Projection Head
  ↓
Geometry Feature Space
  ↓
Reliability-Aware Geometry Bank
  ├── Class mean
  ├── Low-rank basis
  ├── Variance / residual variance
  ├── Active rank
  ├── Reliability
  ├── Spectral prototype
  └── Band importance
  ↓
Geometry Transport Calibration
  ↓
Geometry-Native Classifier
  ↓
Predictions


GNECILHSI/
├── data/
│   ├── hsi_dataloader_pytorch.py     # HSI loading, PCA, patch extraction
│   └── incremental_dataset.py        # Class-incremental phase splits
│
├── models/
│   ├── backbone.py                   # Spectral-spatial feature extractor
│   ├── token.py                      # Optional semantic/token refinement
│   ├── geometry_bank.py              # Reliability-aware low-rank geometry memory
│   ├── classifier.py                 # Geometry-native classifier
│   └── necil_model.py                # Full NECIL-HSI model
│
├── losses/
│   └── necil_losses.py               # Geometry, spectral, token, replay losses
│
├── trainers/
│   ├── trainer.py                    # Base and incremental training loop
│   └── trainer_helpers.py            # Geometry extraction and memory refresh
│
├── utils/
│   ├── eval.py                       # CIL metrics and classification reports
│   └── visualize.py                  # Phase maps, confidence maps, training curves
│
├── datasets/                         # HSI datasets
├── checkpoints/                      # Saved models, reports, maps
├── main.py                           # Entry point
└── requirements.txt
