import logging
import os
import numpy as np
import psycopg2
import boto3
from botocore.client import Config
from datetime import datetime

logger = logging.getLogger(__name__)

# Assisted by AI
def _get_s3_client():
    """Create a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "admin"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def get_bucket():
    return os.environ.get("MINIO_BUCKET", "feedback-images")


def init_db(db_uri):
    # Create feedback table if doesn't exist (image_key references MinIO object)
    conn = psycopg2.connect(db_uri)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            predicted_label INTEGER NOT NULL,
            correct_label INTEGER NOT NULL,
            is_correct BOOLEAN NOT NULL,
            image_key TEXT,
            used_for_training BOOLEAN NOT NULL DEFAULT FALSE
        )
    """)
    # Assisted by AI to improve ingestion speed of DVC DAG in mlpipeline
    # Partial index for fast lookup of unconsumed feedback
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_unused
        ON feedback (used_for_training) WHERE NOT used_for_training
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Feedback DB initialized at {db_uri}")


# Insert feedback into DB and upload image to MinIO
def add_feedback(db_uri, predicted_label, correct_label, pixel_array=None):
    conn = psycopg2.connect(db_uri)
    cur = conn.cursor()
    is_correct = predicted_label == correct_label

    # Assisted by AI
    # Insert row first to get the ID for the S3 key
    cur.execute(
        "INSERT INTO feedback (timestamp, predicted_label, correct_label, is_correct) VALUES (%s, %s, %s, %s) RETURNING id",
        (datetime.utcnow(), predicted_label, correct_label, is_correct),
    )
    row_id = cur.fetchone()[0]

    # Upload pixel data to MinIO if provided
    image_key = None
    if pixel_array is not None:
        try:
            pixel_bytes = np.array(pixel_array, dtype=np.float32).tobytes()
            image_key = f"feedback/{row_id}.npy"
            s3 = _get_s3_client()
            s3.put_object(
                Bucket=get_bucket(),
                Key=image_key,
                Body=pixel_bytes,
                ContentType="application/octet-stream",
            )
            cur.execute("UPDATE feedback SET image_key = %s WHERE id = %s", (image_key, row_id))
        except Exception as e:
            logger.error(f"Failed to upload image to MinIO: {e}")

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM feedback")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = true")
    correct_count = cur.fetchone()[0]

    cur.close()
    conn.close()

    accuracy = correct_count / total if total > 0 else None
    return total, accuracy


def get_stats(db_uri):
    conn = psycopg2.connect(db_uri)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM feedback")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = true")
    correct = cur.fetchone()[0]

    cur.close()
    conn.close()

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(correct / total, 4) if total > 0 else None,
    }


def get_all_feedback_with_pixels(db_uri):
    # Get all feedback entries and images from MinIO
    try:
        conn = psycopg2.connect(db_uri)
        cur = conn.cursor()
        cur.execute("SELECT correct_label, image_key FROM feedback WHERE image_key IS NOT NULL")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}")
        return []

    s3 = _get_s3_client()
    bucket = get_bucket()
    entries = []
    for label, image_key in rows:
        try:
            resp = s3.get_object(Bucket=bucket, Key=image_key)
            pixel_bytes = resp["Body"].read()
            pixels = np.frombuffer(pixel_bytes, dtype=np.float32)
            if pixels.size == 784:
                entries.append({"correct_label": label, "pixel_array": pixels.tolist()})
        except Exception:
            continue
    return entries
