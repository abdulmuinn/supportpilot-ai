"""Batch inference helpers for the SupportPilot dashboard."""

from __future__ import annotations

from typing import Any, Sequence

from supportpilot.ui.api_client import (
    SupportPilotAPIClient,
)


MAX_API_REQUEST_SIZE = 100


def run_batch_inference(
    client: SupportPilotAPIClient,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    request_size: int = MAX_API_REQUEST_SIZE,
) -> dict[str, Any]:
    """Run large dashboard batches across multiple API requests."""

    if not texts:
        raise ValueError(
            "texts must contain at least one message."
        )

    if not 1 <= request_size <= MAX_API_REQUEST_SIZE:
        raise ValueError(
            "request_size must be between 1 and 100."
        )

    if not 1 <= batch_size <= 128:
        raise ValueError(
            "batch_size must be between 1 and 128."
        )

    combined_predictions: list[
        dict[str, Any]
    ] = []

    total = 0
    accepted = 0
    fallback = 0

    for start in range(
        0,
        len(texts),
        request_size,
    ):
        chunk = texts[
            start:start + request_size
        ]

        result = client.predict_batch(
            chunk,
            batch_size=batch_size,
        )

        total += result["total"]
        accepted += result["accepted"]
        fallback += result["fallback"]

        combined_predictions.extend(
            result["predictions"]
        )

    return {
        "total": total,
        "accepted": accepted,
        "fallback": fallback,
        "predictions": combined_predictions,
    }