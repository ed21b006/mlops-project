import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ready():
    r = client.get("/ready")
    assert r.status_code in [200, 503]


def test_predict_with_pixels():
    pixels = np.random.rand(784).tolist()
    r = client.post("/predict", json={"pixel_array": pixels})
    if r.status_code == 200:
        data = r.json()
        assert 0 <= data["predicted_digit"] <= 9
        assert len(data["probabilities"]) == 10
    else:
        assert r.status_code == 503


def test_predict_no_input():
    r = client.post("/predict", json={})
    assert r.status_code in [400, 422]


def test_predict_wrong_length():
    r = client.post("/predict", json={"pixel_array": [0.5] * 100})
    assert r.status_code == 422


def test_feedback():
    r = client.post("/submit_feedback", json={"correct_label": 5, "predicted_label": 5})
    assert r.status_code == 200
    assert r.json()["status"] == "success"


def test_feedback_invalid_label():
    r = client.post("/submit_feedback", json={"correct_label": 15, "predicted_label": 5})
    assert r.status_code == 422


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text # One metrics assures that prometheus is working


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
