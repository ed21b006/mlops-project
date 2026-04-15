# Low-Level Design — API Endpoints

Base URL: `http://localhost:8000`

## POST /predict
Predict digit from image.

**Request:**
```json
{"pixel_array": [0.0, 0.0, ..., 0.98, ...]}  // 784 floats in [0,1]
```
or
```json
{"image_base64": "data:image/png;base64,..."}
```

**Response (200):**
```json
{
  "predicted_digit": 7,
  "confidence": 0.984,
  "probabilities": [0.001, 0.002, ...],
  "inference_time_ms": 12.4
}
```

**Errors:** 400 (no input), 422 (bad input), 503 (model not loaded)

## POST /submit_feedback
Submit correct label. Saves to Postgres and MinIO.

**Request:**
```json
{"correct_label": 5, "predicted_label": 3, "pixel_array": [...]}
```

**Response (200):**
```json
{"status": "success", "message": "Recorded"}
```

## GET /health
**Response:** `{"status": "healthy"}`

## GET /ready
**Response:** `{"ready": true}`

## GET /metrics
Returns Prometheus text format with:
- `http_requests_total` (counter by endpoint/method/status)
- `request_latency_seconds` (histogram)
- `model_confidence_score` (histogram)
- `data_drift_flag` (gauge, 0 or 1)
- `prediction_accuracy` (gauge from feedback)
- `feedback_total` (counter)
- `retrain_total` (counter, status of webhook triggers)

## GET /feedback_stats
**Response:** `{"total": 42, "correct": 39, "incorrect": 3, "accuracy": 0.928}`

---

# Low-Level Design — Internal Service APIs

## Inference Server (`http://localhost:5001`)
- **ANY `/invocations`**: Proxies deep directly into `mlflow models serve` subprocess enforcing body sizes and rate limits `(429 / 413 HTTP codes)`.
- **GET `/health`**: Returns proxy and subprocess version statuses `{"status": "ok", "model_version": "1"}`.

## ML Pipeline (`http://localhost:8001`)
- **POST `/retrain`**: Forces a full `dvc repro --force` rebuild asynchronously pushing artifacts into MLflow. Returns deployments summaries.
- **GET `/baseline_stats`**: Retrieves latest baseline distributions `{"source": "feedback", "pixel_means": [...], ...}`
- **GET `/dag`**: Returns ASCII DVC network map representations.
- **GET `/metrics` / `/status`**: Checks internal build pipelines statuses and evaluation configurations.
