"""Unit tests for model loading configuration."""

import pytest

from supportpilot.inference.model_loader import (
    resolve_model_source,
)


def test_resolve_explicit_model_source():
    source = resolve_model_source(
        "organization/model-name"
    )

    assert source == "organization/model-name"


def test_strip_model_source_whitespace():
    source = resolve_model_source(
        "  organization/model-name  "
    )

    assert source == "organization/model-name"


def test_reject_empty_model_source():
    with pytest.raises(
        RuntimeError,
        match="No model source configured",
    ):
        resolve_model_source(
            ""
        )