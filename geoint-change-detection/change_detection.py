import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter, binary_opening
from rasterio.warp import reproject, Resampling
import os
import json
import random

# =========================================================
# 1. BAND LOOKUP
# =========================================================
def get_band_index(src, name):
    for i, desc in enumerate(src.descriptions, 1):
        if desc and desc.upper() == name.upper():
            return i
    raise ValueError(f"Band '{name}' not found. Available: {src.descriptions}")

# =========================================================
# 2. ALIGNMENT
# =========================================================
def check_and_align(path_src, path_ref, output_path="aligned.tif"):
    with rasterio.open(path_ref) as ref, rasterio.open(path_src) as src:
        src_shape = (src.height, src.width)
        ref_shape = (ref.height, ref.width)
        if (src.crs == ref.crs and
            src_shape == ref_shape and
            np.allclose(src.transform[:6], ref.transform[:6])):
            print("✓ Alignment OK")
            return path_src
        print("✗ Reprojecting to match reference grid...")
        profile = src.profile.copy()
        profile.update({
            "crs":       ref.crs,
            "transform": ref.transform,
            "width":     ref.width,
            "height":    ref.height,
        })
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                desc = (src.descriptions[i-1] or "").upper()
                resample_method = (Resampling.nearest
                                   if desc in ["SCL", "QA60"]
                                   else Resampling.bilinear)
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=resample_method
                )
        print(f"✓ Reprojection complete → {output_path}")
        return output_path

# =========================================================
# 3. CLOUD MASK
# =========================================================
def compute_cloud_mask(path):
    with rasterio.open(path) as src:
        descriptions = [d.upper() if d else "" for d in src.descriptions]
        if "SCL" in descriptions:
            scl = src.read(descriptions.index("SCL") + 1)
            bad = np.isin(scl, [3, 8, 9, 10, 11])
            print(f"  Cloud mask: SCL — {bad.sum()} bad pixels ({100*bad.mean():.1f}%)")
        elif "QA60" in descriptions:
            qa = src.read(descriptions.index("QA60") + 1).astype(np.int32)
            bad = ((qa & (1 << 10)) != 0) | ((qa & (1 << 11)) != 0)
            print(f"  Cloud mask: QA60 — {bad.sum()} bad pixels ({100*bad.mean():.1f}%)")
        else:
            print("  ⚠ No cloud mask found — all pixels treated as clean")
            return np.ones((src.height, src.width), dtype=bool)
        bad = binary_dilation(bad, iterations=2)
        return ~bad

# =========================================================
# 4. RGB LOADING
# =========================================================
def load_rgb(path):
    with rasterio.open(path) as src:
        try:
            r = src.read(get_band_index(src, "TCI_R"))
            g = src.read(get_band_index(src, "TCI_G"))
            b = src.read(get_band_index(src, "TCI_B"))
            scale = 255.0
            print(f"  RGB: TCI bands in {path}")
        except ValueError:
            r = src.read(get_band_index(src, "B4"))
            g = src.read(get_band_index(src, "B3"))
            b = src.read(get_band_index(src, "B2"))
            scale = 10000.0
            print(f"  RGB: B4/B3/B2 in {path}")
        return np.stack([r, g, b], axis=-1).astype(np.float32), scale

def normalize_pair(img1, img2):
    combined = np.concatenate([img1.flatten(), img2.flatten()])
    p98 = np.percentile(combined[combined > 0], 98)
    return np.clip(img1 / p98, 0, 1), np.clip(img2 / p98, 0, 1)

# =========================================================
# 5. NDVI
# =========================================================
def compute_ndvi(path, valid_mask=None):
    with rasterio.open(path) as src:
        red  = src.read(get_band_index(src, "B4")).astype(np.float32)
        nir  = src.read(get_band_index(src, "B8")).astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-6)
    if valid_mask is not None:
        ndvi[~valid_mask] = np.nan
    return ndvi

# =========================================================
# 6. ADAPTIVE THRESHOLD
# =========================================================
def adaptive_threshold(ndvi_change):
    valid = ndvi_change[np.isfinite(ndvi_change)]
    loss  = valid[valid < 0]
    if len(loss) < 100:
        print("  Adaptive threshold: not enough data, using fallback -0.1")
        return -0.1
    percentile_thresh = np.percentile(loss, 10)
    threshold = min(percentile_thresh, -0.08)
    print(f"  Adaptive threshold: {threshold:.4f} "
          f"(percentile={percentile_thresh:.4f}, floor=-0.08)")
    return threshold

# =========================================================
# 7. SPATIAL SPLIT
# =========================================================
def assign_split(row, col, block_size=512):
    block_row = row // block_size
    block_col = col // block_size
    key = (block_row + block_col) % 10
    if key < 7:
        return "train"
    elif key < 9:
        return "val"
    else:
        return "test"

# =========================================================
# 8. CHANNEL-WISE NORMALIZATION STATS
# =========================================================
def compute_norm_stats(metadata, output_dir="dataset", sample_size=200):
    sample    = metadata[:sample_size]
    all_means = []
    all_stds  = []

    for m in sample:
        arr = np.load(
            os.path.join(output_dir, "combined", f"{m['id']}.npy")
        ).astype(np.float32)
        all_means.append(arr.mean(axis=(0, 1)))
        all_stds.append(arr.std(axis=(0, 1)))

    mean  = np.mean(all_means, axis=0)
    std   = np.mean(all_stds,  axis=0)
    stats = {"mean": mean.tolist(), "std": std.tolist()}

    out_path = os.path.join(output_dir, "norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Normalization stats saved → {out_path}")
    print(f"  Channels : [R18 G18 B18 R24 G24 B24 NDVI18 NDVI24 NDVIdelta]")
    print(f"  Mean     : {np.round(mean, 4).tolist()}")
    print(f"  Std      : {np.round(std,  4).tolist()}")
    return stats

# =========================================================
# 9. DATASET INTEGRITY CHECK
# =========================================================
def validate_dataset(output_dir="dataset", n=20, patch_size=64, n_channels=9):
    with open(os.path.join(output_dir, "metadata.json")) as f:
        metadata = json.load(f)

    sample = random.sample(metadata, min(n, len(metadata)))
    errors = []

    for m in sample:
        pid      = m["id"]
        combined = np.load(
            os.path.join(output_dir, "combined", f"{pid}.npy")
        ).astype(np.float32)
        mask = np.load(os.path.join(output_dir, "mask", f"{pid}.npy"))

        if combined.shape != (patch_size, patch_size, n_channels):
            errors.append(f"{pid}: wrong combined shape {combined.shape}")
        if mask.shape != (patch_size, patch_size):
            errors.append(f"{pid}: wrong mask shape {mask.shape}")
        if not np.isfinite(combined).all():
            errors.append(f"{pid}: contains NaN or Inf")
        if mask.max() > 1 or mask.min() < 0:
            errors.append(f"{pid}: mask values out of [0,1]")

    if errors:
        print(f"\n✗ Integrity check failed ({len(errors)} errors):")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"✓ Dataset integrity check passed ({len(sample)} patches validated)")

# =========================================================
# 10. PATCH EXTRACTION
# =========================================================
def extract_patches(img2018, img2024, ndvi2018_full, ndvi2024_full,
                    mask, valid, patch_size=64, stride=32,
                    min_valid=0.9, min_change_ratio=0.02, max_neg_ratio=1.0,
                    output_dir="dataset"):

    rng = np.random.default_rng(42)

    dirs = {
        "img2018":  os.path.join(output_dir, "img2018"),
        "img2024":  os.path.join(output_dir, "img2024"),
        "mask":     os.path.join(output_dir, "mask"),
        "combined": os.path.join(output_dir, "combined"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    H, W           = mask.shape
    total          = 0
    skipped_cloud  = 0
    skipped_change = 0
    pos_coords     = []
    neg_coords     = []

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            total += 1
            r0, r1 = row, row + patch_size
            c0, c1 = col, col + patch_size

            if valid[r0:r1, c0:c1].mean() < min_valid:
                skipped_cloud += 1
                continue

            change_ratio = float(mask[r0:r1, c0:c1].mean())

            if 0 < change_ratio < min_change_ratio:
                skipped_change += 1
                continue

            if change_ratio >= min_change_ratio:
                pos_coords.append((row, col))
            else:
                neg_coords.append((row, col))

    n_pos = len(pos_coords)
    if n_pos == 0:
        raise ValueError(
            "No positive patches found. "
            "Try lowering min_change_ratio or adjusting the NDVI threshold."
        )

    n_neg_before = len(neg_coords)
    if max_neg_ratio is not None and n_neg_before > int(n_pos * max_neg_ratio):
        max_neg    = int(n_pos * max_neg_ratio)
        keep_idx   = rng.choice(n_neg_before, max_neg, replace=False)
        neg_coords = [neg_coords[i] for i in keep_idx]
        print(f"  Class balancing: kept {max_neg} / {n_neg_before} negatives "
              f"to match {n_pos} positives (ratio {max_neg_ratio}:1)")

    all_coords = pos_coords + neg_coords
    all_coords = [all_coords[i] for i in rng.permutation(len(all_coords))]

    metadata = []
    for row, col in all_coords:
        r0, r1 = row, row + patch_size
        c0, c1 = col, col + patch_size

        p2018 = img2018[r0:r1, c0:c1]
        p2024 = img2024[r0:r1, c0:c1]
        pmask = mask[r0:r1, c0:c1]

        ndvi18_patch = (np.nan_to_num(
            ndvi2018_full[r0:r1, c0:c1], nan=0.0) + 1) / 2
        ndvi24_patch = (np.nan_to_num(
            ndvi2024_full[r0:r1, c0:c1], nan=0.0) + 1) / 2

        ndvi18_patch = ndvi18_patch[..., np.newaxis]
        ndvi24_patch = ndvi24_patch[..., np.newaxis]
        ndvi_delta   = ndvi24_patch - ndvi18_patch

        combined = np.concatenate(
            [p2018, p2024, ndvi18_patch, ndvi24_patch, ndvi_delta], axis=-1
        )

        ys, xs = np.where(pmask == 1)
        bbox   = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                  if len(xs) > 0 else None)

        change_ratio = float(pmask.mean())
        valid_ratio  = float(valid[r0:r1, c0:c1].mean())
        patch_id     = f"patch_{row:05d}_{col:05d}"

        np.save(os.path.join(dirs["img2018"],  f"{patch_id}.npy"),
                p2018.astype(np.float16))
        np.save(os.path.join(dirs["img2024"],  f"{patch_id}.npy"),
                p2024.astype(np.float16))
        np.save(os.path.join(dirs["mask"],     f"{patch_id}.npy"), pmask)
        np.save(os.path.join(dirs["combined"], f"{patch_id}.npy"),
                combined.astype(np.float16))

        metadata.append({
            "id":           patch_id,
            "row":          row,
            "col":          col,
            "change_ratio": round(change_ratio, 4),
            "valid_ratio":  round(valid_ratio, 4),
            "label":        "positive" if change_ratio >= min_change_ratio else "negative",
            "bbox":         bbox,
        })

    splits = {"train": [], "val": [], "test": []}
    for m in metadata:
        splits[assign_split(m["row"], m["col"])].append(m["id"])

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(output_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    saved     = len(metadata)
    final_pos = sum(1 for m in metadata if m["label"] == "positive")
    final_neg = sum(1 for m in metadata if m["label"] == "negative")

    print(f"\n✓ Patch extraction complete")
    print(f"  Total windows inspected      : {total}")
    print(f"  Skipped (clouds)             : {skipped_cloud}")
    print(f"  Skipped (insufficient change): {skipped_change}")
    print(f"  Saved                        : {saved}")
    print(f"  Positive patches             : {final_pos} ({100*final_pos/max(saved,1):.1f}%)")
    print(f"  Negative patches             : {final_neg} ({100*final_neg/max(saved,1):.1f}%)")
    print(f"  Train / Val / Test           : "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"  Output folder                : {output_dir}/")
    print(f"  Combined tensor shape        : (64, 64, 9)")

    return metadata, splits

# =========================================================
# 11. DATASET STATS
# =========================================================
def print_dataset_stats(metadata, output_dir="dataset"):
    ratios = [m["change_ratio"] for m in metadata]
    pos    = [r for r in ratios if r > 0]

    with open(os.path.join(output_dir, "splits.json")) as f:
        splits = json.load(f)

    print("\n📊 Dataset Stats:")
    print(f"  Total patches     : {len(ratios)}")
    print(f"  Positive patches  : {len(pos)}")
    print(f"  Negative patches  : {len(ratios) - len(pos)}")
    print(f"  Mean change ratio : {np.mean(ratios):.4f}")
    print(f"  Median            : {np.median(ratios):.4f}")
    print(f"  Max               : {np.max(ratios):.4f}")
    print(f"  Train / Val / Test: "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")

    plt.figure(figsize=(8, 3))
    plt.hist(ratios, bins=50, color="steelblue", edgecolor="white")
    plt.title("Patch Change Ratio Distribution")
    plt.xlabel("Change ratio (fraction of pixels changed)")
    plt.ylabel("Patch count")
    plt.tight_layout()
    plt.show()

# =========================================================
# 12. DATASET PREVIEW
# =========================================================
def preview_patches(output_dir="dataset", n=6):
    with open(os.path.join(output_dir, "metadata.json")) as f:
        metadata = json.load(f)

    pos    = [m for m in metadata if m["label"] == "positive"]
    sample = pos[:n] if len(pos) >= n else metadata[:n]

    fig, axes = plt.subplots(n, 6, figsize=(18, n * 3))

    for i, meta in enumerate(sample):
        pid      = meta["id"]
        p2018    = np.load(os.path.join(output_dir, "img2018",  f"{pid}.npy"))
        p2024    = np.load(os.path.join(output_dir, "img2024",  f"{pid}.npy"))
        pmask    = np.load(os.path.join(output_dir, "mask",     f"{pid}.npy"))
        combined = np.load(
            os.path.join(output_dir, "combined", f"{pid}.npy")
        ).astype(np.float32)

        axes[i, 0].imshow(p2018.astype(np.float32))
        axes[i, 0].set_title("2018 RGB", fontsize=7)

        axes[i, 1].imshow(p2024.astype(np.float32))
        axes[i, 1].set_title("2024 RGB", fontsize=7)

        axes[i, 2].imshow(combined[:, :, :3])
        axes[i, 2].set_title("combined[:3]\n(= 2018)", fontsize=7)

        axes[i, 3].imshow(combined[:, :, 3:6])
        axes[i, 3].set_title("combined[3:6]\n(= 2024)", fontsize=7)

        axes[i, 4].imshow(combined[:, :, 8].astype(np.float32),
                          cmap="bwr", vmin=-0.5, vmax=0.5)
        axes[i, 4].set_title("combined[8]\n(NDVI delta)", fontsize=7)

        axes[i, 5].imshow(pmask, cmap="gray", vmin=0, vmax=1)
        axes[i, 5].set_title(f"mask\nchange={meta['change_ratio']:.2f}",
                              fontsize=7)

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle("Dataset Preview — 9-Channel Tensor + Mask", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()

# =========================================================
# 13. MAIN PIPELINE
# =========================================================
if __name__ == "__main__":
    REF = "sentinel_2018.tif"
    SRC = check_and_align("sentinel_2024.tif", REF)

    print("\nComputing cloud masks...")
    valid2018 = compute_cloud_mask(REF)
    valid2024 = compute_cloud_mask(SRC)
    combined_valid = valid2018 & valid2024
    print(f"  Clean pixels: {combined_valid.sum()} / {combined_valid.size} "
          f"({100*combined_valid.mean():.1f}%)")

    print("\nLoading RGB...")
    img2018_raw, _ = load_rgb(REF)
    img2024_raw, _ = load_rgb(SRC)
    img2018, img2024 = normalize_pair(img2018_raw, img2024_raw)

    print("\nComputing NDVI...")
    ndvi2018 = compute_ndvi(REF, valid_mask=valid2018)
    ndvi2024 = compute_ndvi(SRC, valid_mask=valid2024)

    ndvi2018 = np.where(combined_valid, ndvi2018, np.nan)
    ndvi2024 = np.where(combined_valid, ndvi2024, np.nan)

    ndvi_diff            = ndvi2024 - ndvi2018
    ndvi_change          = ndvi_diff.copy()
    ndvi_change[~np.isfinite(ndvi_change)] = 0
    ndvi_change          = gaussian_filter(ndvi_change, sigma=1)
    ndvi_change[~combined_valid] = np.nan

    print("\nComputing adaptive threshold...")
    threshold = adaptive_threshold(ndvi_change)

    change_mask = (
        (ndvi2018 > 0.2) &
        (ndvi_change < threshold) &
        combined_valid &
        np.isfinite(ndvi_change)
    ).astype(np.uint8)

    change_mask = binary_opening(change_mask, iterations=1).astype(np.uint8)

    changed_pixels = change_mask.sum()
    total_valid    = combined_valid.sum()
    print(f"\nChanged area: {changed_pixels} pixels "
          f"({100 * changed_pixels / total_valid:.2f}% of valid area)")

    with rasterio.open(REF) as src:
        profile = src.profile
        profile.update(count=1, dtype=change_mask.dtype)
    with rasterio.open("urban_change_mask.tif", "w", **profile) as dst:
        dst.write(change_mask, 1)
    print("✓ Exported urban_change_mask.tif")

    print("\nExtracting patches...")
    metadata, splits = extract_patches(
        img2018, img2024, ndvi2018, ndvi2024,
        change_mask, combined_valid,
        patch_size=64,
        stride=32,
        min_valid=0.9,
        min_change_ratio=0.02,
        max_neg_ratio=1.0,
        output_dir="dataset"
    )

    validate_dataset("dataset", n=20, patch_size=64, n_channels=9)
    print_dataset_stats(metadata, output_dir="dataset")
    compute_norm_stats(metadata, output_dir="dataset", sample_size=200)
    preview_patches("dataset", n=6)