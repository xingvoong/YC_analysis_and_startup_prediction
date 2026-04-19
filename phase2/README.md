# Phase 2 — Production ML Infrastructure

Phase 1 built an ML model and a Streamlit demo. Phase 2 turns that into a production system.

The goal: demonstrate the full lifecycle of an ML model in a real environment — versioned, served, monitored, containerized, and self-healing through automated retraining.

---

## Why Phase 2 Exists

A model in a notebook is not a product. A Streamlit app is a demo. Production means:

- The model is served over a typed API, not a UI
- Every experiment is tracked and reproducible
- Predictions are logged so you can catch drift before it becomes a problem
- The whole thing runs in a container — same on your laptop as in the cloud
- When new data arrives, the pipeline retrains, compares, and promotes automatically

That's what this phase builds.

---

## Project Overview

```
YC Analysis & Startup Prediction
│
├── Phase 1 (existing)
│   ├── Data pipeline        → raw → cleaned CSV
│   ├── EDA notebooks        → trend.ipynb, AI_era.ipynb
│   ├── Model training       → scripts/train_model.py
│   └── Streamlit app        → yc_startup_validator_app/
│
└── Phase 2 (this folder)
    ├── Component 1 ✅  FastAPI model server
    ├── Component 2 🔜  MLflow experiment tracking
    ├── Component 3 🔜  Docker + docker-compose
    ├── Component 4 🔜  Prediction logging + drift monitoring
    └── Component 5 🔜  Automated retraining pipeline
```

---

## Roadmap

### Component 1 — FastAPI Model Server ✅
**What:** REST API that wraps the trained model. Typed contracts, structured logging, request timing, and interactive Swagger docs.

**Why it matters:** This is how models ship. An API is versionable, testable, and callable from any client — mobile, web, another service. A Streamlit app is none of those things.

### Component 2 — MLflow Experiment Tracking 🔜
**What:** Wire MLflow into the training pipeline. Every run logs params, metrics, and artifacts. The best model gets registered in the model registry.

**Why it matters:** Without tracking, you can't answer "which model is in prod and why did we pick it." MLflow makes every experiment reproducible and comparable.

### Component 3 — Docker + docker-compose 🔜
**What:** Containerize the FastAPI server. Add a docker-compose file to run the API and MLflow UI together.

**Why it matters:** "Works on my machine" is not a deployment strategy. Docker makes the environment reproducible anywhere — laptop, CI, cloud.

### Component 4 — Prediction Logging + Drift Monitoring 🔜
**What:** Every call to `/predict` gets logged to a store. A monitoring dashboard shows score distributions over time and flags when they shift.

**Why it matters:** Models degrade silently. Drift monitoring is how you catch it before users do. This is the piece most junior engineers skip and most senior engineers wish someone had built.

### Component 5 — Automated Retraining Pipeline 🔜
**What:** A script that pulls fresh YC data, retrains the model, compares it against the current production model in MLflow, and promotes if better.

**Why it matters:** ML models are not static artifacts. The world changes, new batches ship, the distribution shifts. A retraining pipeline is the difference between a model and a system.

---

## Architecture

### Full Phase 2 System (target state)

```
                        ┌─────────────────────────────────────────┐
                        │              Phase 2 System              │
                        │                                         │
   Client               │   ┌──────────────┐                     │
 (browser / curl) ──────┼──▶│  FastAPI     │──┐                  │
                        │   │  :8000       │  │ predict()         │
                        │   └──────────────┘  │                  │
                        │          │           ▼                  │
                        │          │    ┌──────────────┐          │
                        │          │    │  Model       │          │
                        │          │    │  Artifacts   │          │
                        │          │    │  (.pkl)      │          │
                        │          │    └──────────────┘          │
                        │          │                              │
                        │          │ log prediction               │
                        │          ▼                              │
                        │   ┌──────────────┐                     │
                        │   │  Prediction  │                     │
                        │   │  Log Store   │                     │
                        │   └──────┬───────┘                     │
                        │          │                              │
                        │          ▼                              │
                        │   ┌──────────────┐   ┌──────────────┐  │
                        │   │  Drift       │   │  MLflow UI   │  │
                        │   │  Monitor     │   │  :5000       │  │
                        │   └──────────────┘   └──────────────┘  │
                        │                             ▲           │
                        │   ┌──────────────┐          │           │
                        │   │  Retraining  │──register model──────┘
                        │   │  Pipeline    │                     │
                        │   └──────────────┘                     │
                        └─────────────────────────────────────────┘
```

### Component 1 — Request Flow (current state)

```
  POST /predict
       │
       ▼
  ┌────────────────────────────────────────────┐
  │  FastAPI (api.py)                          │
  │                                            │
  │  1. Validate request (Pydantic schema)     │
  │  2. Log request metadata                   │
  │     ↓                                      │
  │  ┌─────────────────────────────────────┐   │
  │  │  predictor.py                       │   │
  │  │  - vectorizer.transform(description)│   │
  │  │  - model.predict_proba(X)           │   │
  │  │  - interpret_score(score)           │   │
  │  │  - explain_prediction(X, top_k)     │   │
  │  └─────────────────────────────────────┘   │
  │     ↓                                      │
  │  3. Serialize response (Pydantic schema)   │
  │  4. Add X-Process-Time-Ms header           │
  └────────────────────────────────────────────┘
       │
       ▼
  PredictResponse {
    score: int,
    probability: float,
    label: str,
    level: str,
    positive_signals: [...],
    negative_signals: [...],
    model_version: str
  }
```

### File Structure

```
phase2/
└── serving/
    ├── api.py            # FastAPI app — routes, middleware, lifespan
    ├── schemas.py        # Pydantic request/response models
    ├── predictor.py      # Prediction logic — decoupled from API layer
    ├── model_loader.py   # Singleton model loader with startup warm-up
    └── requirements.txt  # Pinned dependencies
```

---

## Component 1 — FastAPI Server

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe. Returns model load status. Use for container health checks. |
| `GET` | `/model/info` | Model metadata — type, feature count, version, description. |
| `POST` | `/predict` | Score a startup description. Returns probability, label, and signal words. |
| `GET` | `/docs` | Swagger UI — interactive, try requests from the browser. |
| `GET` | `/redoc` | ReDoc — clean, readable API documentation. |

### Request

```json
POST /predict
{
  "description": "We build developer tools for AI agents to manage memory across tasks",
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
    { "word": "developer", "contribution": 0.1192 },
    { "word": "tools",     "contribution": 0.1692 }
  ],
  "negative_signals": [
    { "word": "agents",  "contribution": -0.3235 },
    { "word": "context", "contribution": -0.1795 }
  ],
  "model_version": "1.0.0"
}
```

### Score Guide

| Score | Level | Meaning |
|-------|-------|---------|
| 75+ | `strong` | Very strong YC-style signal |
| 55–74 | `promising` | Promising, but highly competitive |
| 35–54 | `weak` | Weak signal — niche or execution-heavy |
| <35 | `low` | Low historical success similarity |

### Design Decisions

**Singleton model loader** — models load once at startup via `@lru_cache`. No re-loading per request. Cold start happens once; every subsequent request is fast.

**Predictor decoupled from API** — `predictor.py` has zero FastAPI imports. You can unit test prediction logic without spinning up the server.

**Typed contracts** — Pydantic schemas enforce input validation and output shape. Bad requests fail fast with a 422 before touching the model.

**Structured logging** — every request logs method, path, and duration in milliseconds. Ready to pipe into any log aggregator (Datadog, CloudWatch, Loki).

**`X-Process-Time-Ms` header** — every response carries its own latency. Useful for debugging and performance tracking without a separate APM tool.

---

## Running Component 1

```bash
# From project root
cd phase2/serving

# Install dependencies
pip install -r requirements.txt

# Start the server (with hot reload for development)
python3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- **http://localhost:8000/docs** — Swagger UI, try every endpoint interactively
- **http://localhost:8000/redoc** — clean documentation view
- **http://localhost:8000/health** — quick liveness check

### Quick curl test

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "AI-powered DevOps platform for automated infrastructure management"}'
```

---

## What's Next

Component 2 wires MLflow into the training pipeline. Every run becomes a tracked experiment. The best model goes into a registry. From that point on, the API loads from the registry — not raw `.pkl` files.

That's the difference between "I trained a model" and "I manage a model."
