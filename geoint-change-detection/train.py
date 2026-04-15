# =========================================================
# train.py
# U-Net Change Detection — Training + Inference
# Run in Google Colab with GPU enabled
#
# Step 1: Mount Google Drive
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Step 2: Run this script
# =========================================================

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from scipy.ndimage import binary_opening

# =========================================================
# PATHS — update if your Drive folder name differs
# =========================================================
DRIVE_DIR   = "/content/drive/MyDrive/GEOINT Projects"
DATASET_DIR = "/content/dataset"
NORM_STATS  = "/content/dataset/norm_stats.json"
SAVE_PATH   = f"{DRIVE_DIR}/best_model_unet.pth"
IMG_2018    = f"{DRIVE_DIR}/sentinel_2018.tif"
IMG_2024    = f"{DRIVE_DIR}/sentinel_2024.tif"
OUTPUT_MASK = f"{DRIVE_DIR}/inference_change_mask.tif"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# =========================================================
# COPY DATASET TO LOCAL DISK
# Reading thousands of .npy files directly from Drive is
# too slow — copy once to Colab local SSD first
# =========================================================
if not os.path.exists(DATASET_DIR):
    print("Copying dataset from Drive to local disk...")
    shutil.copytree(f"{DRIVE_DIR}/dataset", DATASET_DIR)
    print("✓ Copy complete")
else:
    print("✓ Dataset already on local disk")

print(os.listdir(DATASET_DIR))

# =========================================================
# DATASET
# =========================================================
class ChangeDataset(Dataset):
    """
    Loads paired (combined, mask, label) tensors from the dataset folder.
    combined : (9, 64, 64) float32 — 9-channel input tensor
    mask     : (1, 64, 64) float32 — per-pixel change label
    label    : scalar float          — patch-level change label (0 or 1)
    """
    def __init__(self, dataset_dir, split, norm_stats_path, augment=False):
        self.dataset_dir = dataset_dir
        self.augment     = augment

        with open(os.path.join(dataset_dir, "splits.json")) as f:
            splits = json.load(f)
        self.ids = splits[split]

        with open(os.path.join(dataset_dir, "metadata.json")) as f:
            meta_list = json.load(f)
        self.meta = {m["id"]: m for m in meta_list}

        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std  = np.array(stats["std"],  dtype=np.float32)
        self.std  = np.where(self.std < 1e-6, 1.0, self.std)

        print(f"  {split}: {len(self.ids)} patches")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        combined = np.load(
            os.path.join(self.dataset_dir, "combined", f"{pid}.npy")
        ).astype(np.float32)

        mask = np.load(
            os.path.join(self.dataset_dir, "mask", f"{pid}.npy")
        ).astype(np.float32)

        combined = (combined - self.mean) / self.std
        combined = np.clip(combined, -10.0, 10.0)
        combined = np.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0)

        if self.augment and random.random() > 0.5:
            combined = np.flip(combined, axis=0).copy()
            mask     = np.flip(mask,     axis=0).copy()
        if self.augment and random.random() > 0.5:
            combined = np.flip(combined, axis=1).copy()
            mask     = np.flip(mask,     axis=1).copy()

        combined = torch.from_numpy(combined.transpose(2, 0, 1))  # (9,64,64)
        mask     = torch.from_numpy(mask).unsqueeze(0)            # (1,64,64)
        mask     = torch.clamp(mask, 0.0, 1.0)

        label = torch.tensor(
            float(self.meta[pid]["change_ratio"] >= 0.02),
            dtype=torch.float32
        )
        label = torch.clamp(label, 0.0, 1.0)

        return combined, mask, label

# =========================================================
# DATALOADERS
# =========================================================
print("\nLoading dataset...")
train_ds = ChangeDataset(DATASET_DIR, "train", NORM_STATS, augment=True)
val_ds   = ChangeDataset(DATASET_DIR, "val",   NORM_STATS, augment=False)
test_ds  = ChangeDataset(DATASET_DIR, "test",  NORM_STATS, augment=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)

# =========================================================
# SANITY CHECK
# =========================================================
print("\nRunning sanity check...")
combined_sample, mask_sample, label_sample = next(iter(train_loader))
print(f"  Combined — min: {combined_sample.min():.3f} max: {combined_sample.max():.3f} "
      f"NaNs: {torch.isnan(combined_sample).sum().item()}")
print(f"  Mask     — min: {mask_sample.min():.3f} max: {mask_sample.max():.3f} "
      f"NaNs: {torch.isnan(mask_sample).sum().item()}")
print(f"  Label    — unique: {label_sample.unique().tolist()}")
print("  ✓ Sanity check passed" if torch.isnan(combined_sample).sum() == 0
      else "  ✗ NaNs detected — check preprocessing")

# =========================================================
# U-NET MODEL
# =========================================================
class ConvBlock(nn.Module):
    """Conv → BN → ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class ChangeUNet(nn.Module):
    """
    U-Net with dual output heads:
      - Segmentation head: per-pixel change probability (B, 1, 64, 64)
      - Classification head: patch-level change probability (B,)
    Input: (B, 9, 64, 64)
    """
    def __init__(self, in_channels=9):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.seg_head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        cls = self.cls_head(b).squeeze(1)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg = self.seg_head(d1)
        return seg, cls

model = ChangeUNet(in_channels=9).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nU-Net ready — {total_params:,} trainable parameters")

# =========================================================
# LOSS + METRICS
# =========================================================
def dice_loss(pred, target, eps=1e-6):
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

def combined_loss(seg_pred, seg_target, cls_pred, cls_target,
                  seg_weight=0.7, cls_weight=0.3):
    seg_pred   = torch.clamp(seg_pred,   1e-6, 1 - 1e-6)
    cls_pred   = torch.clamp(cls_pred,   1e-6, 1 - 1e-6)
    seg_target = torch.clamp(seg_target, 0.0,  1.0)
    cls_target = torch.clamp(cls_target, 0.0,  1.0)
    bce_seg    = F.binary_cross_entropy(seg_pred, seg_target)
    dice       = dice_loss(seg_pred, seg_target)
    seg_loss   = 0.5 * bce_seg + 0.5 * dice
    cls_loss   = F.binary_cross_entropy(cls_pred, cls_target)
    return seg_weight * seg_loss + cls_weight * cls_loss

def compute_metrics(seg_pred, seg_target, cls_pred, cls_target, thresh=0.5):
    seg_bin = (seg_pred > thresh).float()
    tp = (seg_bin * seg_target).sum()
    fp = (seg_bin * (1 - seg_target)).sum()
    fn = ((1 - seg_bin) * seg_target).sum()
    tn = ((1 - seg_bin) * (1 - seg_target)).sum()
    iou     = (tp / (tp + fp + fn + 1e-6)).item()
    f1      = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
    seg_acc = ((tp + tn) / (tp + tn + fp + fn + 1e-6)).item()
    cls_bin = (cls_pred > thresh).float()
    cls_acc = (cls_bin == cls_target).float().mean().item()
    return {"iou": iou, "f1": f1, "seg_acc": seg_acc, "cls_acc": cls_acc}

# =========================================================
# TRAINING LOOP
# =========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss  = 0
    all_metrics = {"iou": 0, "f1": 0, "seg_acc": 0, "cls_acc": 0}
    skipped     = 0

    for combined, mask, label in loader:
        combined = combined.to(device)
        mask     = mask.to(device)
        label    = label.to(device)

        optimizer.zero_grad()
        seg_pred, cls_pred = model(combined)

        if torch.isnan(seg_pred).any() or torch.isnan(cls_pred).any():
            skipped += 1
            optimizer.zero_grad()
            continue

        loss = combined_loss(seg_pred, mask, cls_pred, label)

        if torch.isnan(loss):
            skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        m = compute_metrics(seg_pred.detach(), mask, cls_pred.detach(), label)
        for k in all_metrics:
            all_metrics[k] += m[k]

    n = max(len(loader) - skipped, 1)
    if skipped > 0:
        print(f"    ⚠ Skipped {skipped} batches with NaN outputs")
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss  = 0
    all_metrics = {"iou": 0, "f1": 0, "seg_acc": 0, "cls_acc": 0}
    skipped     = 0

    for combined, mask, label in loader:
        combined = combined.to(device)
        mask     = mask.to(device)
        label    = label.to(device)

        seg_pred, cls_pred = model(combined)

        if torch.isnan(seg_pred).any() or torch.isnan(cls_pred).any():
            skipped += 1
            continue

        loss = combined_loss(seg_pred, mask, cls_pred, label)

        if torch.isnan(loss):
            skipped += 1
            continue

        total_loss += loss.item()
        m = compute_metrics(seg_pred, mask, cls_pred, label)
        for k in all_metrics:
            all_metrics[k] += m[k]

    n = max(len(loader) - skipped, 1)
    if skipped > 0:
        print(f"    ⚠ Skipped {skipped} batches with NaN outputs")
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}

# =========================================================
# TRAIN
# =========================================================
EPOCHS    = 30
LR        = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

history = {"train_loss": [], "val_loss": [],
           "train_iou":  [], "val_iou":  [],
           "train_f1":   [], "val_f1":   []}

best_val_loss = float("inf")
patience      = 7
no_improve    = 0

print(f"\nTraining on {DEVICE} for up to {EPOCHS} epochs\n")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_m = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss,   val_m   = evaluate(model, val_loader, DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_iou"].append(train_m["iou"])
    history["val_iou"].append(val_m["iou"])
    history["train_f1"].append(train_m["f1"])
    history["val_f1"].append(val_m["f1"])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), SAVE_PATH)
        saved_marker  = " ← saved"
    else:
        no_improve   += 1
        saved_marker  = ""

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss {train_loss:.4f}/{val_loss:.4f} | "
        f"IoU {train_m['iou']:.3f}/{val_m['iou']:.3f} | "
        f"F1 {train_m['f1']:.3f}/{val_m['f1']:.3f} | "
        f"ClsAcc {train_m['cls_acc']:.3f}/{val_m['cls_acc']:.3f}"
        f"{saved_marker}"
    )

    if no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\n✓ Best model saved to {SAVE_PATH}")

# =========================================================
# TRAINING CURVES
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"],   label="val")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history["train_iou"], label="train")
axes[1].plot(history["val_iou"],   label="val")
axes[1].set_title("IoU"); axes[1].legend()
axes[2].plot(history["train_f1"], label="train")
axes[2].plot(history["val_f1"],   label="val")
axes[2].set_title("F1"); axes[2].legend()
plt.tight_layout()
plt.show()

# =========================================================
# TEST SET EVALUATION
# =========================================================
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
test_loss, test_m = evaluate(model, test_loader, DEVICE)

print("\n📊 Test Set Results:")
print(f"  Loss    : {test_loss:.4f}")
print(f"  IoU     : {test_m['iou']:.4f}")
print(f"  F1      : {test_m['f1']:.4f}")
print(f"  Seg Acc : {test_m['seg_acc']:.4f}")
print(f"  Cls Acc : {test_m['cls_acc']:.4f}")

# =========================================================
# VISUAL PREDICTIONS
# =========================================================
model.eval()
combined_batch, mask_batch, label_batch = next(iter(test_loader))
combined_batch = combined_batch.to(DEVICE)

with torch.no_grad():
    seg_preds, cls_preds = model(combined_batch)

seg_preds = seg_preds.cpu().numpy()
cls_preds = cls_preds.cpu().numpy()
masks     = mask_batch.numpy()
inputs    = combined_batch.cpu().numpy()

with open(NORM_STATS) as f:
    stats = json.load(f)
mean = np.array(stats["mean"], dtype=np.float32)
std  = np.array(stats["std"],  dtype=np.float32)

def denormalize_rgb(chw_tensor, mean, std):
    rgb = chw_tensor[:3].transpose(1, 2, 0)
    rgb = rgb * std[:3] + mean[:3]
    return np.clip(rgb, 0, 1)

n_show = 6
fig, axes = plt.subplots(n_show, 4, figsize=(12, n_show * 3))

for i in range(n_show):
    rgb = denormalize_rgb(inputs[i], mean, std)
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title("2018 RGB input", fontsize=8)
    axes[i, 1].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title("Ground truth mask", fontsize=8)
    axes[i, 2].imshow(seg_preds[i, 0], cmap="hot", vmin=0, vmax=1)
    axes[i, 2].set_title("Predicted mask", fontsize=8)
    axes[i, 3].imshow((seg_preds[i, 0] > 0.5).astype(np.uint8),
                       cmap="gray", vmin=0, vmax=1)
    axes[i, 3].set_title(f"Binary pred\ncls={cls_preds[i]:.2f}", fontsize=8)
    for ax in axes[i]:
        ax.axis("off")

plt.suptitle("U-Net Test Predictions", fontsize=12)
plt.tight_layout()
plt.show()

# =========================================================
# FULL SCENE INFERENCE
# =========================================================
PATCH_SIZE = 64
STRIDE     = 32

def get_band_index(src, name):
    for i, desc in enumerate(src.descriptions, 1):
        if desc and desc.upper() == name.upper():
            return i
    raise ValueError(f"Band '{name}' not found.")

def load_scene(path):
    with rasterio.open(path) as src:
        try:
            r = src.read(get_band_index(src, "TCI_R")).astype(np.float32)
            g = src.read(get_band_index(src, "TCI_G")).astype(np.float32)
            b = src.read(get_band_index(src, "TCI_B")).astype(np.float32)
            rgb = np.stack([r, g, b], axis=-1) / 255.0
        except ValueError:
            r = src.read(get_band_index(src, "B4")).astype(np.float32)
            g = src.read(get_band_index(src, "B3")).astype(np.float32)
            b = src.read(get_band_index(src, "B2")).astype(np.float32)
            rgb = np.stack([r, g, b], axis=-1) / 10000.0
        red  = src.read(get_band_index(src, "B4")).astype(np.float32)
        nir  = src.read(get_band_index(src, "B8")).astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-6)
        transform = src.transform
        crs       = src.crs
    return rgb, ndvi, transform, crs

print("\nRunning full scene inference...")
print("Loading scenes...")
rgb2018, ndvi2018, transform, crs = load_scene(IMG_2018)
rgb2024, ndvi2024, _,         _   = load_scene(IMG_2024)
print(f"  Scene shape: {rgb2018.shape}")

combined_flat = np.concatenate([rgb2018.flatten(), rgb2024.flatten()])
p98 = np.percentile(combined_flat[combined_flat > 0], 98)
rgb2018 = np.clip(rgb2018 / p98, 0, 1)
rgb2024 = np.clip(rgb2024 / p98, 0, 1)

ndvi2018  = np.nan_to_num((ndvi2018 + 1) / 2, nan=0.5)
ndvi2024  = np.nan_to_num((ndvi2024 + 1) / 2, nan=0.5)
ndvi_delta = ndvi2024 - ndvi2018

H, W      = rgb2018.shape[:2]
prob_map  = np.zeros((H, W), dtype=np.float32)
count_map = np.zeros((H, W), dtype=np.float32)
total_patches = 0

model.eval()
with torch.no_grad():
    for row in range(0, H - PATCH_SIZE + 1, STRIDE):
        for col in range(0, W - PATCH_SIZE + 1, STRIDE):
            r0, r1 = row, row + PATCH_SIZE
            c0, c1 = col, col + PATCH_SIZE

            p2018  = rgb2018[r0:r1, c0:c1]
            p2024  = rgb2024[r0:r1, c0:c1]
            pn18   = ndvi2018[r0:r1, c0:c1][..., np.newaxis]
            pn24   = ndvi2024[r0:r1, c0:c1][..., np.newaxis]
            pdelta = ndvi_delta[r0:r1, c0:c1][..., np.newaxis]

            patch = np.concatenate([p2018, p2024, pn18, pn24, pdelta], axis=-1)
            patch = (patch - mean) / std
            patch = np.clip(patch, -10.0, 10.0)
            patch = np.nan_to_num(patch, nan=0.0)

            tensor = torch.from_numpy(
                patch.transpose(2, 0, 1)
            ).unsqueeze(0).to(DEVICE)

            seg_pred, _ = model(tensor)
            prob = seg_pred.squeeze().cpu().numpy()

            prob_map[r0:r1, c0:c1]  += prob
            count_map[r0:r1, c0:c1] += 1
            total_patches += 1

count_map = np.where(count_map == 0, 1, count_map)
prob_map  = prob_map / count_map
print(f"  ✓ Processed {total_patches} patches")

THRESHOLD   = 0.5
change_mask = (prob_map > THRESHOLD).astype(np.uint8)
change_mask = binary_opening(change_mask, iterations=1).astype(np.uint8)

changed = change_mask.sum()
total   = change_mask.size
print(f"\nChanged area: {changed} pixels ({100*changed/total:.2f}%)")

with rasterio.open(IMG_2018) as src:
    profile = src.profile
    profile.update(count=1, dtype="uint8")

with rasterio.open(OUTPUT_MASK, "w", **profile) as dst:
    dst.write(change_mask, 1)
print(f"✓ Exported → {OUTPUT_MASK}")

# =========================================================
# INFERENCE VISUALIZATION
# =========================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(np.clip(rgb2018, 0, 1))
axes[0].set_title("2018 RGB"); axes[0].axis("off")

axes[1].imshow(np.clip(rgb2024, 0, 1))
axes[1].set_title("2024 RGB"); axes[1].axis("off")

axes[2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
axes[2].set_title("Change Probability Map"); axes[2].axis("off")

axes[3].imshow(change_mask, cmap="gray", vmin=0, vmax=1)
axes[3].set_title(f"Change Mask (t={THRESHOLD})"); axes[3].axis("off")

plt.suptitle("U-Net Inference — Full Scene", fontsize=13)
plt.tight_layout()
plt.show()