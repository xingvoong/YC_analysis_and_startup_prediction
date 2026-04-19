from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Startup idea description",
        example="We build an AI agent that automates customer support for healthcare clinics.",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of top positive/negative signal words to return",
    )


class SignalWord(BaseModel):
    word: str
    contribution: float


class PredictResponse(BaseModel):
    score: int = Field(..., description="Predicted YC-style success probability (0–100)")
    probability: float = Field(..., description="Raw model probability (0.0–1.0)")
    label: str = Field(..., description="Human-readable interpretation of the score")
    level: str = Field(..., description="Signal level: strong / promising / weak / low")
    positive_signals: List[SignalWord] = Field(..., description="Words that increase the score")
    negative_signals: List[SignalWord] = Field(..., description="Words that decrease the score")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    vectorizer_type: str
    n_features: int
    classes: List[str]
    model_version: str
    description: str
