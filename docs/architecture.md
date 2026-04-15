# Architecture

## Overview
Handwritten digit recognition system using CNN on MNIST. Fully local, no cloud.

## Components

```
User → Frontend (Nginx, port 80) → Backend (FastAPI, port 8000) → Inference Server (FastAPI wrapping MLflow, port 5001) → CNN Model
                                                                                         ↓
                                                                             Prometheus ← metrics endpoint (port 9090)
                                                                                  ↓
                                                                             Alertmanager (port 9093)
                                                                                  ↓
                                                                             Grafana (dashboards, port 3000)

ML Pipeline (FastAPI on port 8001) ← orchestration, provides baseline stats
MLflow (port 5000) ← experiment tracking and model registry
Airflow (port 8080) ← data engineering DAGs
PostgreSQL (port 5432) ← metadata database for feedback, Airflow, MLflow
MinIO (port 9000) ← S3-compatible object storage for feedback images
Node Exporter (port 9100) ← host hardware metrics
cAdvisor (port 8081) ← container metrics
```

## How data flows
1. User draws a digit on the canvas (frontend)
2. Frontend sends pixel array to backend via POST `/predict`
3. Backend fetches baseline stats from ML Pipeline, checks for drift, and proxies inference payload to Inference container, returning prediction + confidence.
4. User submits feedback via POST `/submit_feedback`. Backend saves metadata to Postgres and feedback images to MinIO.
5. Airflow DAG runs periodically to process unseen feedback from MinIO/Postgres, validates schemas, computes baselines, and triggers a `/retrain` webhook logic on ML Pipeline.
6. ML Pipeline pulls new training data, executes the DVC pipeline, trains the CNN, and registers the output weights to MLflow Model Registry's "production" alias.
7. Inference Server automatically polls MLflow Registry, detects the "production" alias update, handles traffic, rate limits requests, and hot-reloads the CNN Model into memory.
8. Prometheus continuously scrapes `/metrics` across containers. Grafana visualizes everything and Alertmanager fires notifications if necessary.

## Services (docker-compose)
| Service | Port | Purpose |
|---------|------|---------|
| frontend | 80 | Nginx serving static prod HTML/JS |
| dev_frontend | 8082 | Nginx serving static dev HTML/JS |
| backend | 8000 | FastAPI proxy validating shapes + drift detection |
| inference_server | 5001 | FastAPI + MLflow model endpoint; auto-reloading |
| ml_pipeline | 8001 | FastAPI managing DVC pipelines via webhook |
| mlflow_server | 5000 | Experiment tracking + model registry |
| airflow-apiserver | 8080 | DAG management UI |
| airflow-scheduler | - | Runs scheduled DAGs |
| postgres | 5432 | Primary Relational DB |
| minio / minio-init | 9000/9001 | S3-compatible Blob Storage |
| prometheus | 9090 | Metrics collection |
| alertmanager | 9093 | Alerts routing |
| grafana | 3000 | Dashboards + alerts rendering |
| cadvisor | 8081 | Container resource metrics |

Frontend and backend are loosely coupled — they only communicate via REST API through nginx proxy.
