# Airflow DAG for feedback data pipeline + model retraining
# DAG: ingest feedback -> validate schema -> compute drift baselines -> trigger_retrain


import os
import json
import logging
import psycopg2
from datetime import datetime, timedelta
import numpy as np
import requests
import boto3
from botocore.client import Config
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import Variable
from smtp.email_utils import send_failure_email, send_success_summary_email


BASELINE_DIR = os.environ.get("BASELINE_OUTPUT_DIR", "/opt/airflow/baseline_data")
MLPIPELINE_URL = os.environ.get("MLPIPELINE_URL", "http://ml_pipeline:8001")
DB_URI = os.environ.get("DB_URI", "postgresql://postgres:postgres@postgres:5432/mlops")

# MinIO config
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "feedback-images")

# For the airflow DAG
default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def get_s3_client():
    # Create S3 client for MinIO
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ingest_feedback(**kwargs):
    # Read feedback counts from DB, and finding used vs unused
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM feedback")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM feedback WHERE used_for_training = FALSE AND image_key IS NOT NULL")
        unseen = cur.fetchone()[0]

        cur.close()
        conn.close()
    except Exception as e:
        logging.info(f"Failed to connect to feedback DB or execute query: {e}")
        # Global vars used by airflow
        kwargs["ti"].xcom_push(key="count", value=0)
        kwargs["ti"].xcom_push(key="unseen_count", value=0)
        return

    logging.info(f"Feedback: {total} total, {unseen} unseen (not yet used for training)")
    kwargs["ti"].xcom_push(key="count", value=total)
    kwargs["ti"].xcom_push(key="unseen_count", value=unseen)

# Assisted by AI
def validate_schema(**kwargs):
    # Validate feedback entries by downloading images from MinIO
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, correct_label, image_key FROM feedback "
            "WHERE image_key IS NOT NULL AND used_for_training = FALSE"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        logging.info(f"Failed to get data from feedback DB: {e}")
        kwargs["ti"].xcom_push(key="valid_count", value=0)
        return

    s3 = get_s3_client()
    valid_count = 0
    invalid_count = 0
    for row_id, label, image_key in rows:
        try:
            resp = s3.get_object(Bucket=MINIO_BUCKET, Key=image_key)
            pixel_bytes = resp["Body"].read()
            pixels = np.frombuffer(pixel_bytes, dtype=np.float32)
            if pixels.size == 784 and isinstance(label, int) and 0 <= label <= 9:
                valid_count += 1
            else:
                invalid_count += 1
        except Exception:
            invalid_count += 1

    logging.info(f"Schema validation: {valid_count} valid, {invalid_count} invalid")
    kwargs["ti"].xcom_push(key="valid_count", value=valid_count)


def calculate_baselines(**kwargs):
    # Compute mean/variance of pixels from all feedback data (reads from MinIO)
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        cur.execute("SELECT image_key FROM feedback WHERE image_key IS NOT NULL")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        logging.info(f"Failed to fetch feedback keys: {e}")
        return

    s3 = get_s3_client()
    all_pixels = []
    for (image_key,) in rows:
        try:
            resp = s3.get_object(Bucket=MINIO_BUCKET, Key=image_key)
            pixel_bytes = resp["Body"].read()
            pixels = np.frombuffer(pixel_bytes, dtype=np.float32)
            if pixels.size == 784:
                all_pixels.append(pixels)
        except Exception:
            continue

    if not all_pixels:
        logging.info("No valid pixel data for baseline computation")
        return

    pixels_np = np.array(all_pixels)
    baseline = {
        "source": "feedback",
        "num_samples": len(all_pixels),
        "pixel_means": pixels_np.mean(axis=0).tolist(),
        "pixel_variances": pixels_np.var(axis=0).tolist(),
        "global_mean": float(pixels_np.mean()),
        "global_std": float(pixels_np.std()),
        "computed_at": datetime.utcnow().isoformat(),
    }

    os.makedirs(BASELINE_DIR, exist_ok=True)
    with open(os.path.join(BASELINE_DIR, "feedback_baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    logging.info(f"Baseline computed from {len(all_pixels)} samples, mean={baseline['global_mean']:.4f}")


def trigger_retrain(**kwargs):
    # Trigger retraining via the mlpipeline API
    logging.info("Triggering model retraining...")

    try:
        retrain_resp = requests.post(f"{MLPIPELINE_URL}/retrain", timeout=300)
        retrain_resp.raise_for_status()
        result = retrain_resp.json()
        logging.info(f"Retraining result: {result}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400 and "Need at least 10" in e.response.text:
            logging.info("Not enough new feedback data to retrain today.")
        else:
            logging.error(f"Retraining HTTP error: {e.response.text}")
            raise
    except Exception as e:
        logging.error(f"Retraining request failed: {e}")
        raise

# For extracting global xcom values of the dag
def get_xcom_value(task_instance, task_id, key, default=0):
    if not task_instance:
        return default
    value = task_instance.xcom_pull(task_ids=task_id, key=key)
    return value if value is not None else default

# Callback to send email on success dag
def dag_success_callback(context):
    task_instance = context.get("ti")
    stats = {
        "feedback_total": get_xcom_value(task_instance, "ingest_feedback", "count", 0),
        "feedback_unseen": get_xcom_value(task_instance, "ingest_feedback", "unseen_count", 0),
        "feedback_valid": get_xcom_value(task_instance, "validate_schema", "valid_count", 0),
    }
    send_success_summary_email(context, stats)


with DAG(
    "mnist_data_pipeline",
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    description="Process feedback, validate, compute baselines, trigger retrain",
    schedule="@daily",
    catchup=False,
    on_failure_callback=send_failure_email,
    on_success_callback=dag_success_callback,
) as dag:

    t1 = PythonOperator(task_id="ingest_feedback", python_callable=ingest_feedback)
    t2 = PythonOperator(task_id="validate_schema", python_callable=validate_schema)
    t3 = PythonOperator(task_id="calculate_baselines", python_callable=calculate_baselines)
    t4 = PythonOperator(task_id="trigger_retrain", python_callable=trigger_retrain)

    t1 >> t2 >> t3 >> t4
