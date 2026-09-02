"""Formatting helpers for the SupportPilot dashboard."""


def format_intent(intent: str | None) -> str:
    """Convert a machine-readable intent into a UI label."""

    if not intent:
        return "-"

    if intent == "fallback":
        return "Fallback / Human Review"

    return intent.replace("_", " ").title()


def format_percent(
    value: float | int | None,
    *,
    decimals: int = 2,
) -> str:
    """Format a percentage value for display."""

    if value is None:
        return "-"

    return f"{float(value):.{decimals}f}%"


def format_threshold(
    value: float | int | None,
    *,
    decimals: int = 0,
) -> str:
    """Format a 0-1 threshold as percentage."""

    if value is None:
        return "-"

    percentage = float(value) * 100

    return f"{percentage:.{decimals}f}%"