"""API request and response schemas for SupportPilot AI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single customer message prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Customer support message.",
        examples=[
            "Where is my order?"
        ],
    )


class PredictResponse(BaseModel):
    """Single intent prediction response."""

    text: str

    predicted_id: int
    predicted_intent: str

    confidence: float
    confidence_percent: float

    second_best_id: int
    second_best_intent: str

    second_best_confidence: float
    second_best_confidence_percent: float

    confidence_margin: float
    confidence_margin_percent: float

    final_intent: str

    min_confidence: float
    min_margin: float

    accepted: bool
    status: str
    reason: str


class TopKRequest(BaseModel):
    """Top-K intent prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Customer support message.",
        examples=[
            "Where is my package?"
        ],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of intent candidates "
            "to return."
        ),
    )


class TopKPrediction(BaseModel):
    """One ranked Top-K intent prediction."""

    rank: int
    predicted_id: int
    predicted_intent: str
    confidence: float
    confidence_percent: float


class TopKResponse(BaseModel):
    """Top-K intent prediction response."""

    text: str
    top_k: int

    predictions: list[
        TopKPrediction
    ]


class BatchPredictRequest(BaseModel):
    """Batch intent prediction request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Customer support messages "
            "to classify."
        ),
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Inference batch size.",
    )


class BatchPrediction(BaseModel):
    """Prediction result for one batch item."""

    text: str

    predicted_id: int
    predicted_intent: str

    final_intent: str

    confidence: float
    confidence_percent: float

    second_best_id: int
    second_best_intent: str
    second_best_confidence: float

    confidence_margin: float
    confidence_margin_percent: float

    accepted: bool
    status: str
    reason: str


class BatchPredictResponse(BaseModel):
    """Batch intent prediction response."""

    total: int
    accepted: int
    fallback: int

    predictions: list[
        BatchPrediction
    ]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
    device: str
    num_labels: int


class ModelInfoResponse(BaseModel):
    """Public production model metadata."""

    model_name: str
    device: str
    num_labels: int
    max_length: int
    min_confidence: float
    min_margin: float