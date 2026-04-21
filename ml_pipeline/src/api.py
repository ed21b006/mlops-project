"""
Endpoints:
  POST /retrain  — Runs `dvc repro --force`
  GET  /health   — Health check
  GET  /dag      — Returns DVC DAG visualization (raw text)
  GET  /metrics  — Returns current DVC metrics
  GET  /status   — Returns DVC pipeline status
"""

import json
import logging
import os
import subprocess
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Pipeline Manager")

WORKDIR = os.environ.get("DVC_WORKDIR", "/app")


def run_dvc_command(args, timeout=600):
    # Run a DVC command
    cmd = ["dvc"] + args
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=timeout)
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.info(result.stderr)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


# Run the DVC pipeline on startup if no production model exists
def run_initial_pipeline():
    # Lazy import to avoid holding mlflow in memory permanently
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
    except Exception:
        mlflow.set_tracking_uri("file:///tmp/mlruns")

    client = MlflowClient()
    needs_training = False
    try:
        # Check if a production model exists
        client.get_model_version_by_alias(name="mnist_model", alias="production")
        logger.info("Found existing 'mnist_model@production'. Skipping initial pipeline.")
    except Exception:
        logger.info("No production model found. Running initial DVC pipeline...")
        needs_training = True

    # Release mlflow objects before running the pipeline (child process has its own)
    del client
    import gc
    gc.collect()

    if needs_training:
        result = run_dvc_command(["repro"], timeout=900)
        if result["returncode"] == 0:
            logger.info("Initial pipeline completed successfully.")
        else:
            logger.error(f"Initial pipeline failed: {result['stderr']}")


@app.on_event("startup")
def startup_event():
    logger.info("Starting ML Pipeline Manager...")
    run_initial_pipeline()

# For backend to start only after mlflow is ready
@app.get("/health")
def health():
    return {"status": "ok"}

# Retrigger the entire DVC pipeline
@app.post("/retrain")
def retrain():
    # --force to ensure all stages run even if up-to-date (since we have SQL DB)
    logger.info("Retrain triggered — running dvc repro --force...") 
    result = run_dvc_command(["repro", "--force"], timeout=900)

    if result["returncode"] == 0:
        # Read deployment result if available
        deploy_path = os.path.join(WORKDIR, "metrics", "deploy_result.json")
        deploy_info = {}
        if os.path.exists(deploy_path):
            with open(deploy_path) as f:
                deploy_info = json.load(f)

        return {
            "status": "success",
            "message": "DVC pipeline completed",
            "deployed": deploy_info.get("deployed", False),
            "details": deploy_info,
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {result['stderr'][-500:]}",
        )


@app.get("/dag")
def dag():
    # Return the DVC DAG visualization
    result = run_dvc_command(["dag"])
    return {"dag": result["stdout"]}


@app.get("/metrics")
def metrics():
    # Return current DVC pipeline metrics
    metrics_dir = os.path.join(WORKDIR, "metrics")
    all_metrics = {}
    for fname in ["eval_metrics.json", "train_results.json", "register_result.json", "deploy_result.json"]:
        fpath = os.path.join(metrics_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                all_metrics[fname.replace(".json", "")] = json.load(f)
    return all_metrics


@app.get("/status")
def status():
    # Return DVC pipeline status
    result = run_dvc_command(["status"])
    return {"status": result["stdout"], "returncode": result["returncode"]}

@app.get("/baseline_stats")
def baseline_stats():
    # Return baseline stats for drift detection by backend
    baseline_path = os.path.join(WORKDIR, "data", "transformed", "baseline_stats.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Baseline stats not found")
