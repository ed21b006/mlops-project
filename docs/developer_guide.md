# Developer Guide: MNIST MLOps Platform

This document provides a highly detailed, comprehensive guide to the handwritten digit recognition MLOps project. It is intended for developers, data scientists, and DevOps practitioners who need to understand, maintain, or extend the system.

## 1. System Architecture Overview

The platform uses a microservices architecture orchestrated entirely via Docker Compose. It enforces strict separation of concerns, ensuring the Machine Learning lifecycle (training, tracking) is decoupled from the Application lifecycle (API, frontend), while being unified by a comprehensive monitoring stack.

### Containers

The `docker-compose.yml` file defines 15 distinct containers networked together under the `mlops` bridge network:

1.  **`frontend` & `dev_frontend` (Nginx):** 
    *   **Role**: Serves the static HTML/CSS/JS files to the user.
    *   **Reason**: Nginx is highly performant for static file serving. Proxying API requests (`/predict`, `/api/`) through Nginx avoids Cross-Origin Resource Sharing (CORS) complexities in the browser.
2.  **`backend` (FastAPI):** 
    *   **Role**: Prepares payload, fetches baseline expectations from `ml_pipeline`, tests drift, proxying `POST /invocations` to the inference logic. Exposes REST endpoints for prediction and feedback data collection. Saves data into Postgres.
    *   **Reason**: FastAPI is fast, natively asynchronous, and provides automatic OpenAPI (Swagger) documentation via Pydantic schemas.
3.  **`inference_server`:** 
    *   **Role**: Serves the PyTorch model dynamically.
    *   **Reason**: Wraps `mlflow models serve` inside a load-balancer logic to apply request limits/size limits. Checks MLflow model registry every 30 seconds for new models under the alias "production", applying seamless auto-reload updates.
4.  **`mlflow_server` (MLflow):** 
    *   **Role**: Experiment tracking and Model Registry. Runs on port 5000.
    *   **Reason**: Centralizes logging of hyperparameters, metrics, and model artifacts. It bridges the gap between training scripts and deployment.
5.  **`ml_pipeline`:** 
    *   **Role**: Automates ML lifecycle execution via HTTP endpoints.
    *   **Reason**: Hosts DVC pipelines. If the backend/Airflow detects enough drift, it triggers `/retrain` POST requests.
6.  **`airflow-*` (Init, APIServer, Scheduler, DagProcessor):** 
    *   **Role**: Data engineering orchestration. Initializes DB on Postgres, serves the UI (port 8080), triggers background DAGs to aggregate MinIO bucket contents.
    *   **Reason**: Airflow provides robust scheduling and execution of data transformations (e.g., aggregating user feedback to detect drift) and model update email workflows.
7.  **`postgres` & `minio`:** 
    *   **Role**: Central data backends.
    *   **Reason**: Postgres stores strict schema-backed feedback counters, metrics, and state maps for MLflow/Airflow. MinIO handles blob binary matrices (arrays of uncompressed float image pixels).
8.  **`prometheus`, `alertmanager` & `grafana`:** 
    *   **Role**: Time-series database for metrics scraping over time. Alerts trigger when threshold breaches.
9.  **`node-exporter` & `cadvisor`:** 
    *   **Role**: Exposes host-level and container-level resource utilization metrics respectively.

---

## 2. Module Deep Dive & Code Explanation

### 2.1 The ML Pipeline (`ml_pipeline/`)

The machine learning pipeline is defined using **DVC (Data Version Control)** and **MLflow**.

*   **`dvc.yaml`**: The blueprint of the pipeline. It defines a Directed Acyclic Graph (DAG) consisting of three stages: `ingest` -> `train` -> `evaluate`. For each stage, DVC tracks `deps` (dependencies, like scripts or data) and `outs` (outputs, like tensors or models). If a dependency hasn't changed, DVC skips that step on subsequent runs to save time.
*   **`params.yaml`**: Centralized configuration for hyperparameters (e.g., learning rate, epochs). DVC monitors this file; tweaking a param invalidates the pipeline cache, triggering a retraining.

#### Scripts:
1.  **`src/ingest.py`**:
    *   *Purpose*: Reads raw, binary MNIST IDX files (offline no-cloud mode), converts them into standardized normalized PyTorch tensors, and computes baseline drift statistics.
    *   *Mechanism*: Uses Python's `struct` and `gzip` modules to decode byte arrays. Calculates per-pixel variance and mean for the training set, saving it as `baseline_stats.json`. This baseline is critical for the backend to detect future data drift.
2.  **`src/train.py`**:
    *   *Purpose*: Initiates an MLflow run, constructs a lightweight Convolutional Neural Network (CNN), trains it, logs metrics, and saves the `.pth` artifact.
    *   *Mechanism*: Uses an `Adam` optimizer and Negative Log Likelihood (`NLLLoss`) via `F.log_softmax`. The nested `with mlflow.start_run():` logs all args from `params.yaml`.
3.  **`src/evaluate.py`**:
    *   *Purpose*: Tests the saved model against the hold-out dataset.
    *   *Mechanism*: Computes Accuracy, F1-macro, and a confusion matrix. Crucially, if metrics pass thresholds defined in params (e.g., `min_accuracy: 0.95`), it promotes the artifact directly into the MLflow Model Registry via `mlflow.register_model`.

### 2.2 Backend Application (`backend/`)

*   **`app.py`**: The FastAPI server.
    *   *Lifespan*: Injects `DriftDetector` and Postgres connections on startup.
    *   *`/predict` Endpoint*: Accepts either base64 images or flattened pixel arrays. It checks for drift via `/baseline_stats` logic, acts as a load-balancing API Gateway caching layer by forwarding inputs to the inner `inference_server`, and then logs metrics to Prometheus.
    *   *`/submit_feedback` Endpoint*: Converts true labels and inputs to save directly into a structured Postgres database (for metadata) and ships binary file images to `MinIO` (for blob storage). This avoids race conditions tied to previous text-file shared volume workflows.
    *   *Prometheus Instrumentation*: The `metrics_middleware` intercepts every HTTP request, measuring latency (`REQUEST_LATENCY` histogram) and count (`REQUEST_COUNT` counter). Custom gauges track accuracy and drift state.
*   **`drift_detector.py`**: 
    *   Dynamically downloads `baseline_stats` via the ML Pipeline's API. Calculates Z-scores. If an incoming image's pixels deviate by more than 3 standard deviations from the training pixel distribution, drift is flagged. This handles issues like users drawing with varying line thickness distributions.

### 2.2.1 Inference Server (`inference/`)

A new, decoupled component functioning specifically to serve active PyTorch weights securely to incoming backend traffic.
*   **`serve.py`**: Wrapper for `mlflow models serve`. It manages rate-limiting, enforces strict body byte sizes, prevents internal traffic bottlenecks, and runs a background thread polling MLflow. When the ML Pipeline tags a newly trained model with the "production" alias, the inference server safely drops its inner subprocess and spins up the new model instance automatically for 0-downtime updates.

### 2.3 Frontend Application (`frontend/` & `dev_frontend/`)

*   **`app.js`**: Standard vanilla JavaScript. Uses the HTML5 Canvas API to track mouse/touch coordinates `(moveTo, lineTo)`.
*   *Preprocessing*: Creates an off-screen, hidden 28x28 canvas to downscale the user's high-res drawing via `drawImage()`, strips RGBA to grayscale, and normalizes it to [0,1] floats before making the `fetch()` call to the backend. We host separate production and dev environments using parallel Nginx servers on ports 80 and 8082, maximizing containerization isolations.

### 2.4 Data Engineering (`airflow/`)

*   **`dags/mnist_pipeline_dag.py`**: Orchestrates feedback processing continuously on a scheduled cron basis.
    1.  *Ingest Data*: Reads unprocessed unseen feedback data queries out of `postgres`.
    2.  *Validate Schemas*: Connects to the AWS S3 compatible `MinIO` client via `boto3`. Tests input matrices bounds against strict `(784,)` Float lists and strict `0-9` label distributions.
    3.  *Assess Basics*: Calculates cumulative mathematical distribution aggregates off new data clusters to construct shifting baselines and evaluates edge-cases distributions.
    4.  *Trigger Webhooks*: Executes async REST calls back to `POST ML_PIPELINE/retrain`.
    5.  *Email Alerting*: Automates outbound emails leveraging SMTP plugins notifying engineering teams with aggregated payload statistics mapping.
## 3. Conducting Experiments

This platform is specifically designed to make ML experimentation safe, rigorously tracked, and entirely reproducible.

### Step-by-Step Experiment Execution

1.  **Modify Hyperparameters**: Open `ml_pipeline/params.yaml` and adjust values.
    ```yaml
    train:
      epochs: 10
      learning_rate: 0.005
    model:
      dropout: 0.3
    ```
2.  **Run the DVC Pipeline**: Executing `dvc repro` in the `ml_pipeline` directory triggers DVC to compute MD5 hashes of dependencies. Because `params.yaml` changed, DVC knows the `train` and `evaluate` stages are invalidated and re-runs them autonomously.
    ```bash
    cd ml_pipeline
    dvc repro
    ```
3.  **Track via MLflow**: Once execution finishes, navigate to `http://localhost:5000`. You will see a new run entry. MLflow will show you:
    *   The parameters used.
    *   The `train_loss` curve per epoch.
    *   The test accuracy.
    *   The raw `.pth` PyTorch weights artifact attached to the run.
4.  **Version Control**: Commit the changes to Git. Because DVC tracks metadata (via the `dvc.lock` file that gets generated), committing `dvc.lock` and `params.yaml` pairs that exact ML experiment to a Git Commit Hash.
    ```bash
    git add dvc.lock params.yaml
    git commit -m "Experiment: Increase dropout and lr for robust generalization"
    ```

### Promoting an Experiment to Production
If your new experiment yields high accuracy, `evaluate.py` registers it as version 'N' in the MLflow Model Registry and tags it with the alias "production".

To deploy this model:
There are **zero manual steps** required. The newly architected `inference_server` automatically polls the MLflow Model Registry every 30 seconds for the "production" alias. Once the alias pointer moves, the proxy wrapper will hot-reload its subprocess with the updated PyTorch weights securely without dropping traffic.

---

## 4. Reproducing a Historical Experiment

A core tenet of MLOps is reproducing old models exactly.

1.  Find the Git commit hash of the historical experiment using `git log`.
2.  Checkout that commit:
    ```bash
    git checkout <commit-hash>
    ```
3.  The repository's `params.yaml` and code scripts have now reverted to their historical state. The `dvc.lock` file specifies the exact hashes of the data and outputs required.
4.  Run the pipeline:
    ```bash
    dvc repro
    ```
5.  DVC will re-run the pipeline ensuring the newly generated `mnist_cnn.pth` perfectly matches the historical execution.

---

## 5. Monitoring & Alerting Workflows

The system autonomously evaluates its health using Prometheus rules.

1.  **How metrics flow**: The user submits an unusual image. FastAPI flags `data_drift=1`. The Prometheus container hits `http://backend:8000/metrics` every 15 seconds, dragging the `data_drift_flag` metric into its time-series database.
2.  **Alert Rules**: Prometheus checks `monitoring/prometheus/alert_rules.yml`. It has a rule: `expr: data_drift_flag == 1 for 5m`. If the drift flag stays at 1 for 5 uninterrupted minutes, it fires an Alert.
3.  **Grafana Sync**: Grafana constantly queries Prometheus via PromQL (Prometheus Query Language). The Data Drift panel evaluates `data_drift_flag`. If metric=1, the dashboard dynamically colors the panel red ("DRIFT DETECTED").

By continuously monitoring the `prediction_accuracy` (fueled by the Frontend Feedback mechanism), developers know exactly when model decay has reached a critical failure point requiring a fresh execution of the ML Pipeline described in Section 3.
