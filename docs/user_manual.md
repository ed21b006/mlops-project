# User Manual

## Getting Started
1. Open your browser and go to **http://localhost**
2. Check the top-right corner — it should say "Online" (green dot)
3. If it says "Offline", wait 30 seconds and refresh

## How to Use

### Drawing a digit
- Click and drag on the black canvas to draw
- Draw a digit (0-9) in the center
- Use the brush size slider to adjust thickness

### Getting a prediction
- Click **Predict** (or press Enter)
- The predicted digit and confidence will appear on the right
- A bar chart shows probabilities for each digit

### Clearing
- Click **Clear** (or press Escape) to start over

### Giving feedback
- After a prediction, click the correct digit button (0-9) under "Submit Correct Digit"
- This helps track the model's real-world accuracy

## Tips
- Draw large, centered digits for best results
- Use a thick brush
- The model expects white on black (like MNIST)

## Other Tools
- **Pipeline page**: Click "Pipeline" in the nav bar to see pipeline stages and tool links
- **MLflow**: http://localhost:5000 — experiment tracking
- **Airflow**: http://localhost:8080 — data pipeline (login: admin/admin)
- **Grafana**: http://localhost:3000 — monitoring dashboards (login: admin/admin)
- **Prometheus**: http://localhost:9090 — metrics
- **MinIO**: http://localhost:9001 — S3 Blob Console (login: minioadmin/minioadmin)
- **Alertmanager**: http://localhost:9093 — Check and configure alerting rules
- **API docs**: http://localhost:8000/docs — Swagger UI for FastAPI backend
- **Pipeline Manager docs**: http://localhost:8001/docs — API for triggered retraining pipelines
- **Inference Server docs**: http://localhost:5001/docs — Under-the-hood proxy configurations

## Starting/Stopping
```bash
docker compose up -d      # start everything
docker compose down       # stop
docker compose logs -f    # view logs
```
