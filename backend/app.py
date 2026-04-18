import base64
import io
import logging
import time
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

import config
from drift_detector import DriftDetector
from feedback_utils import init_db, add_feedback, get_stats
from schemas import PredictionRequest, PredictionResponse, FeedbackRequest, FeedbackResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = FastAPI(title="MNIST API")

# To allow frontend send data to backend (since both are on different ports)
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_HOSTS, allow_methods=["*"], allow_headers=["*"])

# Prometheus Metrics
REQ_COUNT = Counter("http_requests_total", "Requests", ["endpoint", "method", "status"])
REQ_LATENCY = Histogram("request_latency_seconds", "Latency", ["endpoint", "method"])
CONF_HIST = Histogram("model_confidence_score", "Confidence")
DRIFT_FLAG = Gauge("data_drift_flag", "Drift status")
ACC_GAUGE = Gauge("prediction_accuracy", "Accuracy")
FB_COUNT = Counter("feedback_total", "Feedback count", ["correct"])
RETRAIN_COUNT = Counter("retrain_total", "Retrains", ["status"])

# Globals (to not reload these on every request)
drift_detector = None
db_uri = None


@app.on_event("startup")
def startup():
    global drift_detector, db_uri
    logging.info("Starting app...")
    # baseline stats is exposed by mlpipeline container
    drift_detector = DriftDetector(f"{config.MLPIPELINE_URL}/baseline_stats", config.DRIFT_THRESHOLD)
    db_uri = config.DB_URI
    init_db(db_uri)

# Middleware function to track common metrics like count & latency
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        # Prometheus gonna make requests to /metrics every 15 secs so we dont want to track it
        return await call_next(request)
    # For other endpoints
    start = time.time()
    resp = await call_next(request)
    REQ_COUNT.labels(request.url.path, request.method, str(resp.status_code)).inc()
    REQ_LATENCY.labels(request.url.path, request.method).observe(time.time() - start)
    return resp

# convert incoming image to pixel_array
def parse_image(req: PredictionRequest):
    if req.image_base64:
        img_data = base64.b64decode(req.image_base64.split(",")[-1] if "," in req.image_base64 else req.image_base64)
        img = Image.open(io.BytesIO(img_data)).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32) / 255.0
        return pixels.flatten()
    elif req.pixel_array:
        return np.array(req.pixel_array, dtype=np.float32)
    raise HTTPException(400, "Missing image")

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.time()
    pixels = parse_image(req)
    
    if drift_detector:
        DRIFT_FLAG.set(1 if drift_detector.check(pixels) else 0)

    mlflow_payload = {"inputs": [pixels.reshape(1, 28, 28).tolist()]}

    try:
        # The MLFlow model server endpoint
        resp = requests.post(f"{config.INFERENCE_URL}/invocations", json=mlflow_payload, timeout=5)
        resp.raise_for_status()
        out = resp.json()['predictions'][0]
        probs = np.exp(out)
    except Exception as e:
        logging.error(f"Inference server error: {e}")
        raise HTTPException(503, "Inference server unavailable")

    digit = int(np.argmax(probs))
    conf = float(probs[digit])
    CONF_HIST.observe(conf)
    
    return PredictionResponse(
        predicted_digit=digit,
        confidence=round(conf, 2),
        probabilities=[round(float(p), 2) for p in probs],
        inference_time_ms=round((time.time() - start) * 1000, 2),
    )

# Endpoint to receive feedback from frontend and store in DB
@app.post("/submit_feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    total, accuracy = add_feedback(db_uri, req.predicted_label, req.correct_label, req.pixel_array)
    FB_COUNT.labels(str(req.predicted_label == req.correct_label).lower()).inc()
    if accuracy is not None:
        ACC_GAUGE.set(accuracy)

    return FeedbackResponse(
        status="success", 
        message="Recorded",
    )


@app.get("/health")
def health(): 
    return {"status": "healthy"}

@app.get("/ready")
def ready(): 
    return {"ready": True}

# can't use start_http_server(8000) because fastapi blocks it on 8000 port
@app.get("/metrics")
def metrics(): 
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/feedback_stats")
def feedback_stats(): 
    return get_stats(db_uri)

