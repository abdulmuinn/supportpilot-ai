"""HTTP routes for the SupportPilot AI API."""

from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
)

from supportpilot.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TopKRequest,
    TopKResponse,
)
from supportpilot.config import settings
from supportpilot.inference.model_loader import (
    load_model_bundle,
)
from supportpilot.inference.predictor import (
    predict_batch,
    predict_top_k,
    predict_with_fallback,
)


router = APIRouter()


def _clean_text(
    text: str,
) -> str:
    """Strip and validate one customer message."""

    clean_text = text.strip()

    if not clean_text:
        raise HTTPException(
            status_code=422,
            detail="Text must not be empty.",
        )

    return clean_text


@router.get("/")
def root() -> dict[str, str]:
    """Return basic API status."""

    return {
        "application": "SupportPilot AI",
        "status": "running",
        "message": (
            "SupportPilot AI API is running."
        ),
    }


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """Verify that the inference model can be loaded."""

    try:
        bundle = load_model_bundle()

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        ) from error

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=str(
            bundle.device
        ),
        num_labels=bundle.num_labels,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_information() -> ModelInfoResponse:
    """Return public model metadata."""

    try:
        bundle = load_model_bundle()

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        ) from error

    return ModelInfoResponse(
        model_name=(
            bundle.model
            .__class__
            .__name__
        ),
        device=str(
            bundle.device
        ),
        num_labels=bundle.num_labels,
        max_length=settings.max_length,
        min_confidence=(
            settings.min_confidence
        ),
        min_margin=settings.min_margin,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
)
def predict_intent(
    request: PredictRequest,
) -> dict:
    """Predict one customer support intent."""

    text = _clean_text(
        request.text
    )

    try:
        return predict_with_fallback(
            text
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Inference failed.",
        ) from error


@router.post(
    "/predict/top-k",
    response_model=TopKResponse,
)
def top_k_prediction(
    request: TopKRequest,
) -> dict:
    """Return ranked intent candidates."""

    text = _clean_text(
        request.text
    )

    try:
        return predict_top_k(
            text=text,
            top_k=request.top_k,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Top-K inference failed.",
        ) from error


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
)
def batch_prediction(
    request: BatchPredictRequest,
) -> dict:
    """Classify multiple customer messages."""

    clean_texts = [
        _clean_text(text)
        for text in request.texts
    ]

    try:
        predictions = predict_batch(
            texts=clean_texts,
            batch_size=request.batch_size,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail="Model is not available.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Batch inference failed.",
        ) from error

    accepted_count = sum(
        item["accepted"]
        for item in predictions
    )

    fallback_count = (
        len(predictions)
        - accepted_count
    )

    return {
        "total": len(predictions),
        "accepted": accepted_count,
        "fallback": fallback_count,
        "predictions": predictions,
    }