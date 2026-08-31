"""API tests for SupportPilot AI."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from supportpilot.api.main import create_app


app = create_app()
client = TestClient(app)


class FakeModel:
    """Lightweight fake model used for API metadata tests."""


def fake_bundle():
    """Return lightweight fake model metadata."""

    return SimpleNamespace(
        model=FakeModel(),
        device="cpu",
        num_labels=46,
    )


def fake_prediction(
    text: str,
) -> dict:
    """Return a deterministic accepted prediction."""

    return {
        "text": text,
        "predicted_id": 43,
        "predicted_intent": "track_order",
        "confidence": 0.98,
        "confidence_percent": 98.0,
        "second_best_id": 17,
        "second_best_intent": "order_history",
        "second_best_confidence": 0.01,
        "second_best_confidence_percent": 1.0,
        "confidence_margin": 0.97,
        "confidence_margin_percent": 97.0,
        "final_intent": "track_order",
        "min_confidence": 0.70,
        "min_margin": 0.10,
        "accepted": True,
        "status": "accepted",
        "reason": (
            "Prediction satisfies "
            "the confidence policy."
        ),
    }


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == "SupportPilot AI"
    assert data["status"] == "running"


def test_health(
    monkeypatch,
):
    monkeypatch.setattr(
        "supportpilot.api.routes.load_model_bundle",
        fake_bundle,
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert data["num_labels"] == 46


def test_model_info(
    monkeypatch,
):
    monkeypatch.setattr(
        "supportpilot.api.routes.load_model_bundle",
        fake_bundle,
    )

    response = client.get(
        "/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "FakeModel"
    assert data["num_labels"] == 46

    assert "source" not in data
    assert "model_path" not in data


def test_predict(
    monkeypatch,
):
    monkeypatch.setattr(
        "supportpilot.api.routes.predict_with_fallback",
        lambda text: fake_prediction(text),
    )

    response = client.post(
        "/predict",
        json={
            "text": "Where is my order?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["predicted_intent"]
        == "track_order"
    )

    assert (
        data["final_intent"]
        == "track_order"
    )

    assert data["accepted"] is True


def test_predict_whitespace_rejected():
    response = client.post(
        "/predict",
        json={
            "text": "   "
        },
    )

    assert response.status_code == 422


def test_predict_missing_text_rejected():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_top_k(
    monkeypatch,
):
    monkeypatch.setattr(
        "supportpilot.api.routes.predict_top_k",
        lambda text, top_k: {
            "text": text,
            "top_k": top_k,
            "predictions": [
                {
                    "rank": 1,
                    "predicted_id": 42,
                    "predicted_intent":
                        "track_delivery",
                    "confidence": 0.95,
                    "confidence_percent": 95.0,
                },
                {
                    "rank": 2,
                    "predicted_id": 43,
                    "predicted_intent":
                        "track_order",
                    "confidence": 0.04,
                    "confidence_percent": 4.0,
                },
            ],
        },
    )

    response = client.post(
        "/predict/top-k",
        json={
            "text": "Where is my package?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["top_k"] == 2

    assert (
        data["predictions"][0]
        ["predicted_intent"]
        == "track_delivery"
    )


def test_invalid_top_k_rejected():
    response = client.post(
        "/predict/top-k",
        json={
            "text": "Where is my order?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_batch_prediction(
    monkeypatch,
):
    def fake_batch(
        texts,
        batch_size,
    ):
        return [
            {
                "text": text,
                "predicted_id": 43,
                "predicted_intent":
                    "track_order",
                "final_intent":
                    "track_order",
                "confidence": 0.98,
                "confidence_percent": 98.0,
                "second_best_id": 17,
                "second_best_intent":
                    "order_history",
                "second_best_confidence":
                    0.01,
                "confidence_margin":
                    0.97,
                "confidence_margin_percent":
                    97.0,
                "accepted": True,
                "status": "accepted",
                "reason": (
                    "Prediction satisfies "
                    "the confidence policy."
                ),
            }
            for text in texts
        ]

    monkeypatch.setattr(
        "supportpilot.api.routes.predict_batch",
        fake_batch,
    )

    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "Where is my order?",
                "Track my order.",
            ],
            "batch_size": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["accepted"] == 2
    assert data["fallback"] == 0


def test_blank_batch_item_rejected():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "Where is my order?",
                "   ",
            ],
            "batch_size": 2,
        },
    )

    assert response.status_code == 422


def test_health_returns_503_when_model_unavailable(
    monkeypatch,
):
    def fail_loading():
        raise RuntimeError(
            "Model unavailable"
        )

    monkeypatch.setattr(
        "supportpilot.api.routes.load_model_bundle",
        fail_loading,
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 503

    assert (
        response.json()["detail"]
        == "Model is not available."
    )