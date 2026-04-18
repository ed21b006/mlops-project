import os

# all config from env that we saved in docker-compose file
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/mnist_cnn.pth")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
DB_URI = os.environ.get("DB_URI", "postgresql://postgres:postgres@postgres:5432/mlops")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", 3.0))
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "http://localhost,http://127.0.0.1").split(",")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://inference_server:5001")
MLPIPELINE_URL = os.environ.get("MLPIPELINE_URL", "http://ml_pipeline:8001")

# MinIO config for feedback images
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "feedback-images")