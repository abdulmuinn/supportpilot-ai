"""Confidence policy for SupportPilot AI predictions."""

from __future__ import annotations

from dataclasses import dataclass

from supportpilot.config import settings


@dataclass(frozen=True)
class ConfidenceDecision:
    """Result of the confidence policy evaluation."""

    accepted: bool
    final_intent: str
    status: str
    reason: str


def evaluate_confidence(
    predicted_intent: str,
    confidence: float,
    margin: float,
    min_confidence: float | None = None,
    min_margin: float | None = None,
) -> ConfidenceDecision:
    """Decide whether a model prediction should be accepted.

    A prediction is accepted only when both the confidence
    and the confidence margin satisfy their configured thresholds.
    """

    confidence_threshold = (
        settings.min_confidence
        if min_confidence is None
        else min_confidence
    )

    margin_threshold = (
        settings.min_margin
        if min_margin is None
        else min_margin
    )

    confidence_passed = (
        confidence >= confidence_threshold
    )

    margin_passed = (
        margin >= margin_threshold
    )

    accepted = (
        confidence_passed
        and margin_passed
    )

    if accepted:
        return ConfidenceDecision(
            accepted=True,
            final_intent=predicted_intent,
            status="accepted",
            reason="Prediction satisfies the confidence policy.",
        )

    reasons: list[str] = []

    if not confidence_passed:
        reasons.append(
            "confidence below threshold"
        )

    if not margin_passed:
        reasons.append(
            "confidence margin below threshold"
        )

    return ConfidenceDecision(
        accepted=False,
        final_intent=settings.fallback_intent,
        status="fallback",
        reason=", ".join(reasons),
    )