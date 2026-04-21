import os
import time
import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "mnist_model")

# MLFlow may not be available immediately when the pipeline starts so need to retry
def setup_mlflow(max_retries=30, retry_delay=5):
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
    except Exception:
        mlflow.set_tracking_uri("file:///tmp/mlruns")

    for attempt in range(1, max_retries + 1):
        try:
            mlflow.set_experiment("mnist-digit-recognition")
            return
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to connect to MLflow after {max_retries} attempts: {e}")
                raise
            logger.warning(f"MLflow not ready (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
            time.sleep(retry_delay)

def register_best_model(run_id: str):
    # Registers the model from a given run and sets it as production
    client = MlflowClient()
    uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from run {run_id} as {MODEL_NAME}")
    
    try:
        model_version = mlflow.register_model(uri, MODEL_NAME)
        # Set alias to production
        client.set_registered_model_alias(MODEL_NAME, "production", model_version.version)
        logger.info(f"Successfully registered {MODEL_NAME} version {model_version.version} as production")
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise e
