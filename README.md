# MNIST Digit Recognition — MLOps Project

Handwritten digit recognition using a CNN on MNIST with full MLOps tooling.

## Quick Start

```bash
# Start all services (training happens automatically!)
docker compose up --build -d
```

The system will automatically:
1. Initialize MinIO buckets and Postgres schemas.
2. Train the initial CNN model (via the `ml_pipeline` service) and register it to MLflow.
3. Automatically load the model into the `inference_server` once it's registered.

*Note: Initial startup may take a few minutes as the model trains for the first time.*

Open **http://localhost** to use the application.

## Services

| Service | URL | Login / Keys |
|---------|-----|-------|
| Frontend (Prod) | http://localhost | - |
| Frontend (Dev) | http://localhost:8082 | - |
| Backend API | http://localhost:8000/docs | - |
| Inference Server | http://localhost:5001 | - |
| ML Pipeline API | http://localhost:8001/docs | - |
| MLflow | http://localhost:5000 | - |
| Airflow | http://localhost:8080 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |
| Grafana | http://localhost:3000 | admin / admin |
| cAdvisor | http://localhost:8081 | - |

## Project Structure
```
├── frontend/        # HTML/JS canvas UI (Nginx) for prod
├── dev_frontend/    # HTML/JS canvas UI (Nginx) for dev
├── backend/         # FastAPI API (drift detection, data validation)
├── inference/       # FastAPI proxy + MLflow model serve, auto-reloading
├── ml_pipeline/     # CNN training (DVC + MLflow), exposed via FastAPI
├── airflow/         # Data engineering DAGs (retraining triggers)
├── monitoring/      # Prometheus, Alertmanager, Grafana config
├── postgres/        # Init scripts for Postgres DB
├── docs/            # Architecture, HLD, LLD, test plan, user manual
└── docker-compose.yml
```

## Running with MLflow

The `ml_pipeline` directory is structured as an [MLflow Project](https://mlflow.org/docs/latest/projects.html). If you want to train and experiment locally without Docker, you can run its entry points directly from your terminal using the `MLproject` file:

```bash
# Run the default complete pipeline (which maps to `dvc repro`)
mlflow run ml_pipeline

# Or run a specific stage (e.g., training) with custom parameters
mlflow run ml_pipeline -e train -P epochs=5 -P learning_rate=0.005 -P batch_size=32
```

## Docs
- [Architecture](docs/architecture.md)
- [High Level Design](docs/high_level_design.md)
- [Low Level Design](docs/low_level_design.md)
- [Test Plan](docs/test_plan.md)
- [User Manual](docs/user_manual.md)
- [Developer Guide](docs/developer_guide.md)
