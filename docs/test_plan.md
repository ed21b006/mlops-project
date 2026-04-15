# Test Plan

## Acceptance Criteria
- Model accuracy >= 95% on MNIST test set
- Inference latency < 200ms
- All API endpoints respond correctly
- Full user flow works: draw → predict → feedback
- Prometheus targets all UP
- Grafana dashboard shows live data

## Test Cases

### ML Pipeline
| ID | Test | Expected |
|----|------|----------|
| ML-1 | Model output shape | (batch, 10) |
| ML-2 | Output is log-probs | exp sum ≈ 1.0 |
| ML-3 | Params < 1M | Under 1M |
| ML-4 | Save/load roundtrip | Same output |

### Backend API
| ID | Test | Expected |
|----|------|----------|
| API-1 | GET /health | 200, status=healthy |
| API-2 | GET /ready | 200 or 503 |
| API-3 | POST /predict (valid) | 200, digit 0-9 |
| API-4 | POST /predict (no input) | 400/422 |
| API-5 | POST /predict (wrong length) | 422 |
| API-6 | POST /submit_feedback (valid) | 200, success |
| API-7 | POST /submit_feedback (bad label) | 422 |
| API-8 | GET /metrics | Prometheus format |

### System
| ID | Test | Expected |
|----|------|----------|
| SYS-1 | docker compose build | All images built |
| SYS-2 | docker compose up | All services running |
| SYS-3 | Frontend loads | Page at localhost:80 (prod) and 8082 (dev) |
| SYS-4 | Full flow | Draw → predict → feedback works |

## Running Tests
```bash
# pipeline tests
cd ml_pipeline && pytest tests/ -v

# backend tests
cd backend && pytest tests/ -v

# system check
docker compose up -d
curl http://localhost:8000/health
```
