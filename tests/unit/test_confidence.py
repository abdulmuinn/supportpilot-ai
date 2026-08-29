"""Unit tests for the SupportPilot confidence policy."""

from supportpilot.inference.confidence import (
    evaluate_confidence,
)


def test_accept_high_confidence_prediction():
    decision = evaluate_confidence(
        predicted_intent="track_order",
        confidence=0.95,
        margin=0.80,
    )

    assert decision.accepted is True
    assert decision.final_intent == "track_order"
    assert decision.status == "accepted"


def test_fallback_when_confidence_is_low():
    decision = evaluate_confidence(
        predicted_intent="track_order",
        confidence=0.50,
        margin=0.40,
    )

    assert decision.accepted is False
    assert decision.final_intent == "fallback"
    assert decision.status == "fallback"


def test_fallback_when_margin_is_low():
    decision = evaluate_confidence(
        predicted_intent="track_order",
        confidence=0.90,
        margin=0.05,
    )

    assert decision.accepted is False
    assert decision.final_intent == "fallback"


def test_custom_thresholds():
    decision = evaluate_confidence(
        predicted_intent="track_order",
        confidence=0.85,
        margin=0.20,
        min_confidence=0.80,
        min_margin=0.15,
    )

    assert decision.accepted is True