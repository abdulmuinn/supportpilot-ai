"""Unit tests for dashboard formatting helpers."""

from supportpilot.ui.formatting import (
    format_intent,
    format_percent,
    format_threshold,
)


def test_format_intent():
    assert (
        format_intent("track_order")
        == "Track Order"
    )


def test_format_fallback_intent():
    assert (
        format_intent("fallback")
        == "Fallback / Human Review"
    )


def test_format_empty_intent():
    assert format_intent("") == "-"
    assert format_intent(None) == "-"


def test_format_percent():
    assert (
        format_percent(99.8182)
        == "99.82%"
    )


def test_format_percent_custom_decimals():
    assert (
        format_percent(
            99.8182,
            decimals=4,
        )
        == "99.8182%"
    )


def test_format_threshold():
    assert (
        format_threshold(0.70)
        == "70%"
    )


def test_format_none_values():
    assert format_percent(None) == "-"
    assert format_threshold(None) == "-"