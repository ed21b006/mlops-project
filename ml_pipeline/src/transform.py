# DVC Stage 2: Data Transformation

import json
import logging
import os
import shutil
import torch
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(BASE_DIR, "data", "ingested")
OUT_DIR = os.path.join(BASE_DIR, "data", "transformed")
BASELINE_SHARED_PATH = os.environ.get("BASELINE_SHARED_PATH", "/app/data/baseline_stats.json")

def compute_baseline(images_tensor):
    # Compute per-pixel mean and variance for drift detection
    num_samples = images_tensor.size(0)
    num_pixels = images_tensor[0].numel()

    sum_pixels = torch.zeros(num_pixels, dtype=torch.float64)
    sumsq_pixels = torch.zeros(num_pixels, dtype=torch.float64)
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    chunk_size = 2048
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = images_tensor[start:end].to(torch.float32)
        if chunk.max().item() > 1.0:
            chunk = chunk / 255.0

        flat = chunk.reshape(chunk.size(0), -1)
        sum_pixels += flat.sum(dim=0, dtype=torch.float64)
        sumsq_pixels += (flat * flat).sum(dim=0, dtype=torch.float64)

        total_sum += float(flat.sum().item())
        total_sumsq += float((flat * flat).sum().item())
        total_count += flat.numel()

        del chunk, flat

    mean_pixels = sum_pixels / num_samples
    var_pixels = (sumsq_pixels / num_samples) - (mean_pixels * mean_pixels)
    var_pixels = torch.clamp(var_pixels, min=0.0)

    g_mean = total_sum / total_count
    g_var = max((total_sumsq / total_count) - (g_mean * g_mean), 0.0)
    g_std = g_var ** 0.5
    
    return {
        "pixel_means": mean_pixels.tolist(),
        "pixel_variances": var_pixels.tolist(),
        "global_mean": float(g_mean),
        "global_std": float(g_std),
        "num_samples": num_samples,
    }
# Assisted by AI to optimize memory usage
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    logger.info("Loading train data...")
    train_data = torch.load(os.path.join(IN_DIR, "train.pt"), map_location="cpu", weights_only=True)
    train_images = train_data["images"]
    train_labels = train_data["labels"]
    
    logger.info("Loaded train data")
    
    baseline = compute_baseline(train_images)
    
    train_size = int(0.8 * len(train_images))
    val_size = len(train_images) - train_size
    
    indices = torch.randperm(len(train_images), device="cpu")
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Save validation split first
    torch.save({
        "images": train_images[val_indices].clone(),
        "labels": train_labels[val_indices].clone()
    }, os.path.join(OUT_DIR, "val.pt"))
    
    # Build and save train split
    train_images = train_images[train_indices].clone()
    train_labels = train_labels[train_indices].clone()
    torch.save({
        "images": train_images,
        "labels": train_labels
    }, os.path.join(OUT_DIR, "train.pt"))
    
    del train_data
    del train_images, train_labels, indices, train_indices, val_indices
    gc.collect()
    
    logger.info("Loading test data...")
    test_data = torch.load(os.path.join(IN_DIR, "test.pt"), map_location="cpu", weights_only=True)
    torch.save(test_data, os.path.join(OUT_DIR, "test.pt"))
    del test_data
    gc.collect()
    
    baseline_path = os.path.join(OUT_DIR, "baseline_stats.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    try:
        shared_dir = os.path.dirname(BASELINE_SHARED_PATH)
        os.makedirs(shared_dir, exist_ok=True)
        shutil.copy2(baseline_path, BASELINE_SHARED_PATH)
        logger.info(f"Baseline stats copied to shared volume: {BASELINE_SHARED_PATH}")
    except Exception as e:
        logger.warning(f"Could not copy baseline to shared volume: {e}")

    logger.info("Transformation complete.")

if __name__ == "__main__":
    main()
