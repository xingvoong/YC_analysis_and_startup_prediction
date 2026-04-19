"""
Model loader with singleton pattern and lazy loading.
Models are loaded once at startup and reused across requests.
"""

import pickle
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

# Resolve model path relative to the original app's model directory
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
MODEL_DIR = BASE_DIR / "yc_startup_validator_app" / "model"


class ModelArtifacts:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.loaded = False
        self._load()

    def _load(self):
        vectorizer_path = MODEL_DIR / "vectorizer.pkl"
        model_path = MODEL_DIR / "success_model.pkl"

        if not vectorizer_path.exists():
            logger.error("vectorizer.pkl not found at %s", vectorizer_path)
            return
        if not model_path.exists():
            logger.error("success_model.pkl not found at %s", model_path)
            return

        try:
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.loaded = True
            logger.info("Models loaded successfully from %s", MODEL_DIR)
        except Exception as e:
            logger.exception("Failed to load models: %s", e)

    @property
    def version(self) -> str:
        return MODEL_VERSION


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    """Return cached model artifacts. Loads once, reuses forever."""
    return ModelArtifacts()
