# DVC Stage 5: Model Registration

import json
import logging
import os
import sys
import mlflow
import mlflow.pytorch
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MNISTNet
from mlflow_utils import setup_mlflow, register_best_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    with open(os.path.join(BASE_DIR, "params.yaml")) as f:
        params = yaml.safe_load(f)
    min_acc = params["evaluate"]["min_accuracy"]
    min_f1 = params["evaluate"]["min_f1_score"]

    with open(os.path.join(BASE_DIR, "metrics", "eval_metrics.json")) as f:
        metrics = json.load(f)

    acc = metrics["accuracy"]
    f1 = metrics["f1_score_macro"]

    result = {
        "registered": False,
        "accuracy": acc,
        "f1_score": f1,
        "min_accuracy": min_acc,
        "min_f1_score": min_f1,
    }

    # Register model only if performance thresholds are met
    if acc >= min_acc and f1 >= min_f1:
        logger.info(f"Thresholds met (acc={acc:.4f}>={min_acc}, f1={f1:.4f}>={min_f1}). Registering...")

        checkpoint = torch.load(os.path.join(BASE_DIR, "models", "mnist_cnn.pth"), weights_only=True)
        model = MNISTNet(**checkpoint["arch_params"])
        model.load_state_dict(checkpoint["model_state_dict"])

        setup_mlflow()
        with mlflow.start_run(run_name="model_registration") as run:
            mlflow.log_metrics({"registered_accuracy": acc, "registered_f1": f1})
            mlflow.pytorch.log_model(
                model, "model",
                code_paths=[os.path.join(os.path.dirname(__file__), "model.py")],
            )
            try:
                register_best_model(run.info.run_id)
                result["registered"] = True
                result["run_id"] = run.info.run_id
                result["reason"] = "Thresholds met"
                logger.info("Model registered with 'production' alias.")
            except Exception as e:
                result["reason"] = f"Registration failed: {e}"
                logger.error(f"Registration failed: {e}")
    else:
        result["reason"] = f"Thresholds not met (acc={acc:.4f}<{min_acc} or f1={f1:.4f}<{min_f1})"
        logger.warning(result["reason"])

    os.makedirs(os.path.join(BASE_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "metrics", "register_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Registration stage complete.")


if __name__ == "__main__":
    main()
