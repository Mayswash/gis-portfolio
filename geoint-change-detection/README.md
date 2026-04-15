# Urban Change Detection from Satellite Imagery
### Bay Area, California — 2018 to 2024

A complete machine learning pipeline for detecting urban land change using Sentinel-2 satellite imagery. Built from scratch — from raw satellite data through preprocessing, dataset construction, model training, and full-scene inference — using Python, PyTorch, and Google Earth Engine.

---

## Results

| Model | IoU | F1 Score | Seg Accuracy | Cls Accuracy |
|-------|-----|----------|--------------|--------------|
| CNN Baseline | 0.682 | 0.809 | 0.977 | 0.999 |
| **U-Net (final)** | **0.900** | **0.947** | **0.993** | **1.000** |

The U-Net model achieved **0.900 IoU and 0.947 F1** on held-out test patches — a strong result for real-world satellite change detection.

---

## What This Project Does

The pipeline detects areas where land cover changed between 2018 and 2024 across the San Francisco Bay Area. It flags vegetation loss as a proxy for urban expansion, development, or disturbance events such as wildfire.

Given two satellite images of the same location taken years apart, the model outputs a per-pixel binary mask indicating where significant change occurred.

---

## Pipeline Overview

```
Google Earth Engine          Local Preprocessing           Model Training
─────────────────────        ──────────────────────        ──────────────────────
Export Sentinel-2    →       Cloud masking              →  CNN baseline
imagery (2018/2024)          NDVI computation              U-Net upgrade
26 bands, 10m res            Adaptive thresholding         Dual-head output:
Bay Area scene               Patch extraction               - per-pixel mask
                             Spatial train/val/test split   - patch-level label
                             9-channel input tensors
```

---

## Key Technical Details

**Satellite Data**
- Source: Sentinel-2 L2A via Google Earth Engine
- Resolution: 10 meters per pixel
- Dates: May–September 2018 and May–September 2024 (same season, both years)
- Coverage: San Francisco Bay Area (~55 km × 55 km)

**Preprocessing**
- Cloud and shadow masking using the Scene Classification Layer (SCL)
- NDVI-based change label generation with adaptive Otsu thresholding
- Morphological cleanup to remove speckle noise from labels
- Spatially-aware train/val/test splits (512×512 pixel blocks) to prevent data leakage from overlapping patches

**Input Tensor — 9 channels per patch**
```
[R₂₀₁₈  G₂₀₁₈  B₂₀₁₈  R₂₀₂₄  G₂₀₂₄  B₂₀₂₄  NDVI₂₀₁₈  NDVI₂₀₂₄  NDVI_delta]
```

**Dataset**
- Patch size: 64×64 pixels (640m × 640m on the ground)
- Total patches: 11,414 (balanced 1:1 positive/negative)
- Train / Val / Test: 7,461 / 2,669 / 1,284

**Model Architecture — U-Net**
- Encoder: 4 ConvBlocks with MaxPool downsampling (9 → 32 → 64 → 128 → 256 channels)
- Bottleneck: 512 channels
- Decoder: 4 transposed convolution blocks with skip connections
- Segmentation head: 1×1 conv → sigmoid → per-pixel change probability
- Classification head: global average pool → FC → sigmoid → patch-level label
- Parameters: ~7.8M trainable

**Training**
- Loss: 70% (BCE + Dice) segmentation + 30% BCE classification
- Optimizer: AdamW with cosine annealing LR schedule
- Augmentation: random horizontal and vertical flips
- Early stopping: patience of 7 epochs
- Hardware: NVIDIA T4 GPU (Google Colab)

---

## Inference

The inference script runs the trained U-Net across a full 5,570 × 5,567 pixel scene using a sliding window with 50% overlap. Overlapping predictions are averaged to reduce edge artifacts. The output is a georeferenced GeoTIFF (EPSG:4326) that can be loaded directly into QGIS or any GIS tool.

**Full scene result:** 516,619 pixels flagged as changed (1.67% of scene area)

---

## Repository Structure

```
├── change_detection.py       # Preprocessing + patch extraction pipeline
├── dataset/
│   ├── combined/             # 9-channel input patches (.npy, float16)
│   ├── img2018/              # 2018 RGB patches
│   ├── img2024/              # 2024 RGB patches
│   ├── mask/                 # Binary change masks
│   ├── metadata.json         # Per-patch stats and bounding boxes
│   ├── splits.json           # Spatial train/val/test split
│   └── norm_stats.json       # Per-channel mean and std for normalization
├── best_model_unet.pth       # Trained U-Net weights
├── best_model.pth            # CNN baseline weights
└── inference_change_mask.tif # Full-scene georeferenced output
```

---

## How to Run

**1. Preprocessing (local)**
```bash
pip install rasterio numpy matplotlib scipy
python change_detection.py
```
Requires `sentinel_2018.tif` and `sentinel_2024.tif` in the working directory.

**2. Training (Google Colab with GPU)**

Open the training notebook in Colab, mount Google Drive, and run all cells. The dataset is copied to local Colab disk before training for faster I/O.

**3. Inference**

Run the inference script in Colab with the trained model and input GeoTIFFs. Outputs `inference_change_mask.tif` to Google Drive.

---

## Tools and Libraries

| Tool | Purpose |
|------|---------|
| Google Earth Engine | Satellite data export |
| rasterio | GeoTIFF reading and writing |
| NumPy / SciPy | Preprocessing and morphological operations |
| PyTorch | Model definition and training |
| Matplotlib | Visualization |
| QGIS | Georeferenced output inspection |

---

## About

Built as part of a GEOINT (Geospatial Intelligence) portfolio project. The goal was to build a production-quality change detection pipeline end-to-end — not just train a model, but handle the full stack from satellite data acquisition through inference and georeferenced output.

The Bay Area was chosen as the study area given the high rate of urban development and land cover change documented between 2018 and 2024, including wildfire impacts from the 2020 LNU Lightning Complex fires visible in the northern portion of the scene.
