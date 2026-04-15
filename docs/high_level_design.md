# High-Level Design

## Problem
Classify handwritten digits (0-9) from user-drawn images using a CNN model, with full MLOps tooling.

## Design Choices

**CNN over Random Forest**: CNNs are better at learning spatial patterns in images. Our model is 2 conv layers + 2 FC layers, under 1M parameters so it runs fast on CPU.

**FastAPI**: Built-in Swagger docs, async support, Pydantic validation. Better than Flask for API-heavy workloads.

**Static frontend (HTML/JS)**: No build step needed. Canvas API handles drawing. Keeps frontend and backend truly independent — connected only via REST calls.

**MLflow**: Tracks experiments (params, metrics, artifacts). Model Registry for versioning. Used directly for model serving via `mlflow models serve` inside a load-balanced proxy wrapper container.

**DVC + Airflow**: DVC handles ML pipeline reproducibility (each run tied to git commit, now orchestrated as a running web service that accepts API webhooks). Airflow handles async data engineering (querying relational metadata loops, fetching blob images out of S3 arrays, calculating statistics, and firing training HTTP pings).

**Postgres + MinIO**: Postgres functions as our strict relational datastore for application statistics, label counts, and Airflow configurations. MinIO stands as our fast, self-hosted, S3-compatible cloud object storage to warehouse heavy raw float-byte arrays for inference traces avoiding monolithic log bottlenecks.

**Prometheus + Alertmanager + Grafana**: Standard monitoring stack. Prometheus scrapes metrics, Grafana visualizes. Alert rules configure `Alertmanager` to fire off notifications conditionally (like high error rates >5% and data drift detections).

**Docker Compose**: Extensively complex compose map unifying 15 different components isolated into microservices across virtual bridge subnets while resolving internal container health checks dynamically.

## Design Paradigm
- ML pipeline: functional orchestrator (containerizes stateless scripts acting on REST endpoints for ingest/train/evaluate)
- Backend API: REST Gateway Proxy (DriftDetector logic handles payloads and farms heavy compute loads instantly out to dedicated MLflow model instances)
- Feedback loops: S3 object stores paired to Postgres guarantees high-volume robust ingestions while guaranteeing Airflow DAG safety.
