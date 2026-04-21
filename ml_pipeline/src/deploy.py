# DVC Stage 6: Deployment Verification

import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlflow_utils import setup_mlflow, MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Read registration result
    with open(os.path.join(BASE_DIR, "metrics", "register_result.json")) as f:
        reg_result = json.load(f)

    result = {
        "deployed": False,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if not reg_result.get("registered"):
        result["reason"] = "Model was not registered — skipping deployment"
        logger.warning(result["reason"])
    else:
        # Verify model exists in MLflow registry with production alias
        try:
            setup_mlflow()
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            mv = client.get_model_version_by_alias(MODEL_NAME, "production")
            result["deployed"] = True
            result["model_version"] = mv.version
            result["model_name"] = MODEL_NAME
            result["reason"] = "Model verified in MLflow registry with production alias"
            logger.info(f"Deployment verified: {MODEL_NAME} v{mv.version} @ production")
        except Exception as e:
            result["reason"] = f"Deployment verification failed: {e}"
            logger.error(result["reason"])

    os.makedirs(os.path.join(BASE_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "metrics", "deploy_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Deployment stage complete.")


if __name__ == "__main__":
    main()
