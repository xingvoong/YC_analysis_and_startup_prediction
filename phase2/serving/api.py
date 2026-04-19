"""
YC Startup Validator — Production FastAPI Server
Phase 2: Model Serving Layer

Endpoints:
  GET  /health         — liveness + model status
  GET  /model/info     — model metadata
  POST /predict        — score a startup description
  GET  /docs           — Swagger UI (interactive)
  GET  /redoc          — ReDoc UI (clean read)
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_loader import get_artifacts
from predictor import run_prediction
from schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("yc_validator.api")


# ---------------------------------------------------------------------------
# Lifespan — warm up models before first request
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up model artifacts...")
    artifacts = get_artifacts()
    if not artifacts.loaded:
        logger.warning("Models failed to load on startup — /predict will return 503")
    else:
        logger.info("Models ready (version %s)", artifacts.version)
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YC Startup Validator API",
    description=(
        "Production ML API that scores a startup description against patterns "
        "learned from 18,995 historical YC companies.\n\n"
        "**Phase 2** of the YC Analysis project — demonstrates production-grade "
        "model serving with versioning, structured logging, and typed contracts."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    logger.info("%s %s — %dms", request.method, request.url.path, duration_ms)
    return response


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
def health():
    """
    Liveness probe. Returns model load status.
    Use this for container health checks and uptime monitoring.
    """
    artifacts = get_artifacts()
    return HealthResponse(
        status="ok" if artifacts.loaded else "degraded",
        model_loaded=artifacts.model is not None,
        vectorizer_loaded=artifacts.vectorizer is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Infrastructure"])
def model_info():
    """
    Returns metadata about the loaded model — type, feature count, version.
    Useful for debugging and auditing deployed artifacts.
    """
    artifacts = get_artifacts()
    if not artifacts.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_type=type(artifacts.model).__name__,
        vectorizer_type=type(artifacts.vectorizer).__name__,
        n_features=len(artifacts.vectorizer.get_feature_names_out()),
        classes=list(artifacts.model.classes_.astype(str)),
        model_version=artifacts.version,
        description=(
            "Logistic regression trained on TF-IDF vectors of 18,995 YC startup descriptions. "
            "Predicts probability that a startup matches patterns of companies that scaled."
        ),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Score a startup description against historical YC success patterns.

    Returns a 0–100 score, signal interpretation, and the top words
    driving the prediction in both directions.

    **Score guide:**
    - 75+ : Very strong YC-style signal
    - 55–74 : Promising, but highly competitive
    - 35–54 : Weak signal — niche or execution-heavy
    - <35 : Low historical success similarity
    """
    artifacts = get_artifacts()
    if not artifacts.loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    logger.info("Predicting — description length: %d chars", len(request.description))

    result = run_prediction(
        vectorizer=artifacts.vectorizer,
        model=artifacts.model,
        description=request.description,
        top_k=request.top_k,
        version=artifacts.version,
    )

    logger.info("Score: %d | Level: %s", result["score"], result["level"])
    return PredictResponse(**result)
