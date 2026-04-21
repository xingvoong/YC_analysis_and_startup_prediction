# Phase 2 — ML Serving Infrastructure

Production serving layer for the YC Startup Validator.

---

## Architecture

```
  Client
    │
    ▼
┌─────────────────┐     ┌──────────────────┐
│  FastAPI :8000  │────▶│  Model Artifacts │
│  (Component 1)  │     │  (.pkl)          │
└────────┬────────┘     └──────────────────┘
         │
         │ log prediction
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Prediction DB  │     │  MLflow :5000    │
│  (SQLite)       │     │  (Component 2)   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       │ register model
┌─────────────────┐     ┌────────┴─────────┐
│  Drift Monitor  │     │  Retraining      │
│  :8501          │     │  Pipeline        │
│  (Component 4)  │     │  (Component 5)   │
└─────────────────┘     └──────────────────┘
```

---

## Components

| # | What | Status |
|---|------|--------|
| 1 | FastAPI model server | ✅ |
| 2 | MLflow experiment tracking | ✅ |
| 3 | Docker + Compose | ✅ |
| 4 | Prediction logging + drift dashboard | ✅ |
| 5 | Automated retraining pipeline | ✅ |

---

## Layout

```
phase2/
├── serving/
│   ├── api.py              # FastAPI app
│   ├── predictor.py        # Inference logic
│   ├── model_loader.py     # Singleton model loader
│   ├── schemas.py          # Pydantic I/O contracts
│   ├── Dockerfile
│   └── requirements.txt
│
├── lifecycle/
│   ├── train_tracked.py    # Train + log run to MLflow
│   ├── retrain.py          # Fetch → train → compare → promote
│   └── requirements.txt
│
├── observability/
│   ├── logger.py           # Writes each prediction to SQLite
│   ├── dashboard.py        # Streamlit drift monitoring UI
│   └── requirements.txt
│
├── docker-compose.yml
└── README.md
```

---

## Running

### Local (no Docker)

```bash
# Component 1 — API
cd phase2/serving
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Component 2 — MLflow UI
mlflow server --host 0.0.0.0 --port 5000

# Component 2 — Train with tracking (run from project root)
python3.11 phase2/lifecycle/train_tracked.py

# Component 4 — Drift dashboard
streamlit run phase2/observability/dashboard.py

# Component 5 — Retrain pipeline (run from project root)
python3.11 phase2/lifecycle/retrain.py
```

### Docker

```bash
cd phase2
docker-compose up
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Drift dashboard | http://localhost:8501 |

---

## API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/model/info` | Model metadata |
| `POST` | `/predict` | Score a startup description |

### Request

```json
POST /predict
{
  "description": "AI-powered DevOps platform for automated infrastructure management",
  "top_k": 8
}
```

### Response

```json
{
  "score": 72,
  "probability": 0.7231,
  "label": "Promising, but highly competitive",
  "level": "promising",
  "positive_signals": [
    { "word": "infrastructure", "contribution": 0.2341 }
  ],
  "negative_signals": [
    { "word": "automated", "contribution": -0.1423 }
  ],
  "model_version": "1.0.0"
}
```

### Score levels

| Range | Level |
|-------|-------|
| 75–100 | `strong` |
| 55–74 | `promising` |
| 35–54 | `weak` |
| 0–34 | `low` |
