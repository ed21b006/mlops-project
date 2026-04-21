# DVC Stage 1: Data Ingestion
import gc
import gzip
import json
import logging
import os
import struct
import shutil
import numpy as np
import torch
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "ingested")

# Assisted by AI
def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic number {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, 1, rows, cols)

# Assisted by AI
def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic number {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def _get_s3_client():
    """Create a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _get_bucket():
    return os.environ.get("MINIO_BUCKET", "feedback-images")


def load_new_feedback():
    # Load only unused feedback from MinIO S3 + PostgreSQL
    db_uri = os.environ.get("DB_URI", "postgresql://postgres:postgres@postgres:5432/mlops")

    try:
        import psycopg2
        conn = psycopg2.connect(db_uri)

        # Count unused feedback with image data
        count_cur = conn.cursor()
        count_cur.execute("SELECT COUNT(*) FROM feedback WHERE used_for_training = FALSE AND image_key IS NOT NULL")
        total_rows = count_cur.fetchone()[0]
        count_cur.close()

        if total_rows == 0:
            conn.close()
            logger.info("No new feedback to ingest.")
            return None, None, []

        # Pre-allocate arrays
        images = np.empty((total_rows, 1, 28, 28), dtype=np.uint8)
        labels = np.empty(total_rows, dtype=np.int64)
        row_ids = []

        # Fetch metadata
        cur = conn.cursor(name="feedback_meta_stream")
        cur.itersize = 5000
        cur.execute(
            "SELECT id, correct_label, image_key FROM feedback "
            "WHERE used_for_training = FALSE AND image_key IS NOT NULL"
        )

        s3 = _get_s3_client()
        bucket = _get_bucket()
        valid_idx = 0

        for row_id, label, image_key in cur:
            try:
                # Stream image from MinIO one at a time, for memory efficiency
                resp = s3.get_object(Bucket=bucket, Key=image_key)
                pixel_bytes = resp["Body"].read()
                pixels = np.frombuffer(pixel_bytes, dtype=np.float32)

                if pixels.size == 784:
                    if pixels.max(initial=0.0) <= 1.0:
                        pixels = pixels * 255.0
                    pixels = np.clip(pixels, 0.0, 255.0).astype(np.uint8, copy=False)
                    images[valid_idx, 0, :, :] = pixels.reshape(28, 28)
                    labels[valid_idx] = label
                    row_ids.append(row_id)
                    valid_idx += 1
            except Exception as e:
                logger.warning(f"Failed to load feedback {row_id} from MinIO: {e}")

        cur.close()
        conn.close()

        if valid_idx == 0:
            return None, None, []

        images = images[:valid_idx]
        labels = labels[:valid_idx]

        logger.info(f"Loaded {valid_idx} new feedback samples from MinIO")
        return images, labels, row_ids

    except Exception as e:
        logger.warning(f"Could not load feedback data: {e}")
        return None, None, []

# Assisted by AI
def mark_feedback_as_used(row_ids):
    # Mark feedback rows as consumed by training
    if not row_ids:
        return

    db_uri = os.environ.get("DB_URI", "postgresql://postgres:postgres@postgres:5432/mlops")
    try:
        import psycopg2
        conn = psycopg2.connect(db_uri)
        cur = conn.cursor()
        # Batch update in chunks to avoid large SQL statements
        chunk_size = 500
        for i in range(0, len(row_ids), chunk_size):
            chunk = row_ids[i:i + chunk_size]
            cur.execute(
                "UPDATE feedback SET used_for_training = TRUE WHERE id = ANY(%s)",
                (chunk,),
            )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Marked {len(row_ids)} feedback rows as used_for_training")
    except Exception as e:
        logger.warning(f"Failed to mark feedback as used: {e}")


def compute_label_distribution(labels):
    counts = np.bincount(labels, minlength=10)
    return [{"digit": int(i), "count": int(c)} for i, c in enumerate(counts)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    needed = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for f in needed:
        if not os.path.exists(os.path.join(RAW_DIR, f)):
            raise FileNotFoundError(f"Missing {f} in {RAW_DIR}")

    joined_path = os.path.join(OUT_DIR, "joined_train.pt")
    train_out_path = os.path.join(OUT_DIR, "train.pt")
    test_out_path = os.path.join(OUT_DIR, "test.pt")
    label_dist_path = os.path.join(OUT_DIR, "label_distribution.json")

    # Load only unused feedback first
    fb_imgs, fb_lbls, fb_ids = load_new_feedback()

    if os.path.exists(joined_path) and fb_imgs is None:
        logger.info("No new feedback and joined dataset already exists. Reusing cached training data.")

        if not os.path.exists(train_out_path):
            shutil.copy2(joined_path, train_out_path)
            logger.info(f"Copied cached joined data to {train_out_path}")

        if not os.path.exists(test_out_path):
            logger.info("Test split missing, rebuilding from raw MNIST test data...")
            test_imgs = read_idx_images(os.path.join(RAW_DIR, "t10k-images-idx3-ubyte.gz"))
            test_lbls = read_idx_labels(os.path.join(RAW_DIR, "t10k-labels-idx1-ubyte.gz"))
            torch.save({"images": torch.from_numpy(test_imgs), "labels": torch.from_numpy(test_lbls)}, test_out_path)
            del test_imgs, test_lbls
            gc.collect()

        if not os.path.exists(label_dist_path):
            logger.info("Label distribution missing, recomputing from cached joined data...")
            cached = torch.load(joined_path, weights_only=True)
            dist = compute_label_distribution(cached["labels"].numpy())
            del cached
            gc.collect()
            with open(label_dist_path, "w") as f:
                json.dump(dist, f, indent=2)

        logger.info("Ingestion complete.")
        return

    # Check if we have a previously joined dataset to build on
    if os.path.exists(joined_path):
        logger.info("Loading previously joined training data...")
        prev = torch.load(joined_path, weights_only=True)
        train_imgs = prev["images"].numpy()
        train_lbls = prev["labels"].numpy()
        logger.info(f"Loaded {len(train_lbls)} samples from previous joined data")
    else:
        logger.info("Loading raw MNIST train data (first run)...")
        train_imgs = read_idx_images(os.path.join(RAW_DIR, "train-images-idx3-ubyte.gz"))
        train_lbls = read_idx_labels(os.path.join(RAW_DIR, "train-labels-idx1-ubyte.gz"))
        logger.info(f"Loaded {len(train_lbls)} raw training samples")

    if fb_imgs is not None:
        train_imgs = np.concatenate([train_imgs, fb_imgs], axis=0)
        train_lbls = np.concatenate([train_lbls, fb_lbls], axis=0)
        del fb_imgs, fb_lbls
        logger.info(f"After merging new feedback: {len(train_lbls)} total train samples")

    # Save the joined dataset for this and future retrain cycles
    torch.save(
        {"images": torch.from_numpy(train_imgs), "labels": torch.from_numpy(train_lbls)},
        joined_path,
    )
    logger.info(f"Saved joined training data to {joined_path}")

    # Also save as train.pt for downstream DVC stages
    torch.save(
        {"images": torch.from_numpy(train_imgs), "labels": torch.from_numpy(train_lbls)},
        train_out_path,
    )
    dist = compute_label_distribution(train_lbls)

    del train_imgs, train_lbls
    gc.collect()

    # Mark feedback as consumed after successful save
    if fb_ids:
        mark_feedback_as_used(fb_ids)

    logger.info("Loading raw MNIST test data...")
    test_imgs = read_idx_images(os.path.join(RAW_DIR, "t10k-images-idx3-ubyte.gz"))
    test_lbls = read_idx_labels(os.path.join(RAW_DIR, "t10k-labels-idx1-ubyte.gz"))
    logger.info(f"Loaded {len(test_lbls)} test samples")

    torch.save({"images": torch.from_numpy(test_imgs), "labels": torch.from_numpy(test_lbls)}, test_out_path)

    del test_imgs, test_lbls
    gc.collect()

    with open(label_dist_path, "w") as f:
        json.dump(dist, f, indent=2)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()