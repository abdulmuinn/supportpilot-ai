"""Unit tests for the dashboard API client."""

import httpx
import pytest

from supportpilot.ui.api_client import (
    SupportPilotAPIClient,
    SupportPilotAPIError,
)


def make_client(
    handler,
) -> SupportPilotAPIClient:
    """Create an API client backed by a mock transport."""

    transport = httpx.MockTransport(
        handler
    )

    http_client = httpx.Client(
        transport=transport
    )

    return SupportPilotAPIClient(
        base_url="http://testserver",
        client=http_client,
    )


def test_health_request():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        assert request.method == "GET"
        assert request.url.path == "/health"

        return httpx.Response(
            200,
            json={
                "status": "healthy",
                "model_loaded": True,
                "device": "cpu",
                "num_labels": 46,
            },
        )

    client = make_client(
        handler
    )

    result = client.health()

    assert result["status"] == "healthy"
    assert result["num_labels"] == 46


def test_predict_request():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        assert request.method == "POST"
        assert request.url.path == "/predict"

        return httpx.Response(
            200,
            json={
                "predicted_intent":
                    "track_order",
                "final_intent":
                    "track_order",
                "accepted": True,
            },
        )

    client = make_client(
        handler
    )

    result = client.predict(
        "Where is my order?"
    )

    assert (
        result["final_intent"]
        == "track_order"
    )

    assert result["accepted"] is True


def test_top_k_request():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            json={
                "text": "Where is my package?",
                "top_k": 3,
                "predictions": [],
            },
        )

    client = make_client(
        handler
    )

    result = client.predict_top_k(
        "Where is my package?",
        top_k=3,
    )

    assert result["top_k"] == 3


def test_batch_request():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            json={
                "total": 2,
                "accepted": 2,
                "fallback": 0,
                "predictions": [],
            },
        )

    client = make_client(
        handler
    )

    result = client.predict_batch(
        [
            "Where is my order?",
            "Cancel my order.",
        ],
        batch_size=2,
    )

    assert result["total"] == 2


def test_api_error_response():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            503,
            json={
                "detail":
                    "Model is not available."
            },
        )

    client = make_client(
        handler
    )

    with pytest.raises(
        SupportPilotAPIError,
        match="Model is not available",
    ) as error:
        client.health()

    assert (
        error.value.status_code
        == 503
    )


def test_invalid_json_response():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            content=b"not-json",
        )

    client = make_client(
        handler
    )

    with pytest.raises(
        SupportPilotAPIError,
        match="invalid JSON",
    ):
        client.health()


def test_reject_empty_base_url():
    with pytest.raises(
        ValueError,
        match="base URL",
    ):
        SupportPilotAPIClient(
            base_url="   "
        )