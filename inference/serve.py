import asyncio
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict, deque
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from mlflow.client import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Model name is fixed accross the project for simple tracking and deployement
MLFLOW_PORT = int(os.getenv("MLFLOW_PORT", "5002")) # port where MLflow serve will run internally
PROXY_PORT = int(os.getenv("PROXY_PORT", "5001")) # port to receive requests
MODEL_NAME = os.getenv("MODEL_NAME", "mnist_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production") # Tag of the model
MODEL_POLL_SECONDS = int(os.getenv("MODEL_POLL_SECONDS", "30")) # Interval to check for new versions

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")) # Time window for api rate limiting
RATE_LIMIT_PER_WINDOW = int(os.getenv("RATE_LIMIT_PER_WINDOW", "60")) # Maximum requests per time window
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "1048576")) # Payload limit of 1MB

# Assisted by AI - Security configs
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
PROXY_TIMEOUT_SECONDS = float(os.getenv("PROXY_TIMEOUT_SECONDS", "10"))
ENABLE_HSTS = os.getenv("ENABLE_HSTS", "false").lower() == "true"

app = FastAPI(title="Inference Server")

rate_lock = asyncio.Lock()
requests_by_ip = defaultdict(deque)

process_lock = threading.Lock()
process = None
current_version = None

# To fetch latest production model
def get_latest_version():
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return mv.version
    except Exception as e:
        logger.debug(f"Could not fetch model version: {e}")
        return None

# Model serving
def start_mlflow_process():
    global process
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    process = subprocess.Popen(
        [
            "mlflow",
            "models",
            "serve",
            "-m",
            f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
            "--port",
            str(MLFLOW_PORT),
            "--host",
            "0.0.0.0",
            "--env-manager",
            "local",
        ],
        env=env,
    )

# To stop the MLflow process when a new model has to be served
def stop_mlflow_process():
    global process
    if process:
        process.terminate()
        process.wait()
        process = None

# Deploy new model
def refresh_model_if_needed():
    global current_version
    latest = get_latest_version()
    if latest is None:
        return
    if current_version != latest or process is None:
        logger.info(f"Starting MLflow serve for version: {latest}")
        stop_mlflow_process()
        start_mlflow_process()
        current_version = latest

# Background thread to monitor for new model versions
def model_monitor_loop():
    while True:
        time.sleep(MODEL_POLL_SECONDS)
        with process_lock:
            refresh_model_if_needed()

# Assisted by AI
def extract_client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"

# Assisted by AI
def apply_security_headers(response: Response) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    if ENABLE_HSTS:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

# Assisted by AI
async def enforce_rate_limit(request: Request):
    if RATE_LIMIT_PER_WINDOW <= 0:
        return None
    client_ip = extract_client_ip(request)
    now = time.time()
    async with rate_lock:
        window = requests_by_ip[client_ip]
        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_WINDOW:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - window[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)
    return None

# Assisted by AI
async def enforce_body_size(request: Request):
    if MAX_BODY_BYTES <= 0:
        return None
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    request.state.body = body
    return None

# Assisted by AI
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    rate_resp = await enforce_rate_limit(request)
    if rate_resp:
        apply_security_headers(rate_resp)
        return rate_resp
    size_resp = await enforce_body_size(request)
    if size_resp:
        apply_security_headers(size_resp)
        return size_resp
    resp = await call_next(request)
    apply_security_headers(resp)
    return resp


@app.on_event("startup")
def startup():
    global current_version
    with process_lock:
        current_version = get_latest_version()
        if current_version:
            logger.info(f"Starting MLflow serve for version: {current_version}")
            start_mlflow_process()
        else:
            logger.info("No production model found yet. Waiting for initial training to finish...")
    thread = threading.Thread(target=model_monitor_loop, daemon=True)
    thread.start()


@app.on_event("shutdown")
def shutdown():
    with process_lock:
        stop_mlflow_process()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": current_version}

# Assisted by AI
# Catch-all route to proxy requests to MLflow serve, with added security and rate limiting
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    with process_lock:
        if process is None:
            return JSONResponse(status_code=503, content={"detail": "Model not ready"})

    target_url = f"http://127.0.0.1:{MLFLOW_PORT}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    body = getattr(request.state, "body", None)
    if body is None:
        body = await request.body()

    async with httpx.AsyncClient(timeout=PROXY_TIMEOUT_SECONDS) as client:
        resp = await client.request(
            request.method,
            target_url,
            headers=headers,
            content=body,
        )

    excluded = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    return Response(content=resp.content, status_code=resp.status_code, headers=response_headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
