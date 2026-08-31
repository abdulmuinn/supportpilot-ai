"""Unit tests for SupportPilot prediction services."""

from types import SimpleNamespace

import pytest
import torch

from supportpilot.inference.model_loader import (
    ModelBundle,
)
from supportpilot.inference.predictor import (
    predict_batch,
    predict_top_k,
    predict_with_fallback,
)


class FakeTokenizer:
    """Minimal tokenizer used for predictor unit tests."""

    def __call__(
        self,
        texts,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        token_ids = []

        for text in texts:
            normalized = text.lower()

            if "cancel" in normalized:
                token_id = 1

            elif "weather" in normalized:
                token_id = 2

            else:
                token_id = 0

            token_ids.append(
                [token_id]
            )

        return {
            "input_ids": torch.tensor(
                token_ids,
                dtype=torch.long,
            )
        }


class FakeModel:
    """Minimal classification model with deterministic logits."""

    def __call__(
        self,
        input_ids,
        **kwargs,
    ):
        logits = []

        for token_id in input_ids[:, 0].tolist():

            if token_id == 0:
                row = [
                    5.0,
                    1.0,
                    0.0,
                ]

            elif token_id == 1:
                row = [
                    0.0,
                    5.0,
                    0.0,
                ]

            else:
                row = [
                    0.4,
                    0.3,
                    0.2,
                ]

            logits.append(
                row
            )

        return SimpleNamespace(
            logits=torch.tensor(
                logits,
                dtype=torch.float32,
            )
        )


@pytest.fixture
def fake_bundle():
    """Create a lightweight model bundle for unit tests."""

    return ModelBundle(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device=torch.device("cpu"),
        id2label={
            0: "track_order",
            1: "cancel_order",
            2: "sales_period",
        },
        label2id={
            "track_order": 0,
            "cancel_order": 1,
            "sales_period": 2,
        },
        num_labels=3,
        source="fake-model",
    )


def test_predict_top_k_returns_ranked_intents(
    fake_bundle,
):
    result = predict_top_k(
        "Where is my order?",
        top_k=2,
        bundle=fake_bundle,
    )

    assert result["top_k"] == 2
    assert len(
        result["predictions"]
    ) == 2

    assert (
        result["predictions"][0]
        ["predicted_intent"]
        == "track_order"
    )

    assert (
        result["predictions"][0]["rank"]
        == 1
    )


def test_accept_confident_prediction(
    fake_bundle,
):
    result = predict_with_fallback(
        "Where is my order?",
        bundle=fake_bundle,
    )

    assert (
        result["predicted_intent"]
        == "track_order"
    )

    assert (
        result["final_intent"]
        == "track_order"
    )

    assert result["accepted"] is True
    assert result["status"] == "accepted"


def test_fallback_low_confidence_prediction(
    fake_bundle,
):
    result = predict_with_fallback(
        "What is the weather today?",
        bundle=fake_bundle,
    )

    assert result["accepted"] is False

    assert (
        result["final_intent"]
        == "fallback"
    )

    assert result["status"] == "fallback"


def test_batch_prediction(
    fake_bundle,
):
    result = predict_batch(
        [
            "Where is my order?",
            "I want to cancel my order.",
            "What is the weather today?",
        ],
        bundle=fake_bundle,
        batch_size=3,
    )

    assert len(result) == 3

    assert (
        result[0]["final_intent"]
        == "track_order"
    )

    assert (
        result[1]["final_intent"]
        == "cancel_order"
    )

    assert (
        result[2]["final_intent"]
        == "fallback"
    )


def test_reject_empty_text(
    fake_bundle,
):
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        predict_with_fallback(
            "   ",
            bundle=fake_bundle,
        )


def test_reject_invalid_batch_size(
    fake_bundle,
):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        predict_batch(
            ["Where is my order?"],
            bundle=fake_bundle,
            batch_size=0,
        )