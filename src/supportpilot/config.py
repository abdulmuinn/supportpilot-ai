"""Application configuration for SupportPilot AI."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    api_host: str
    api_port: int

    api_url: str

    model_id: str | None
    model_revision: str | None

    max_length: int

    min_confidence: float
    min_margin: float

    fallback_intent: str

    log_level: str


def _optional_env(name: str) -> str | None:
    """Return a stripped environment variable or None."""

    value = os.getenv(name)

    if value is None:
        return None

    value = value.strip()

    return value or None


def get_settings() -> Settings:
    """Build application settings from environment variables."""

    return Settings(
        api_host=os.getenv(
            "SUPPORTPILOT_API_HOST",
            "0.0.0.0",
        ),
        api_port=int(
            os.getenv(
                "SUPPORTPILOT_API_PORT",
                "8000",
            )
        ),
        api_url=os.getenv(
            "SUPPORTPILOT_API_URL",
            "http://127.0.0.1:8000",
        ),
        model_id=_optional_env(
            "SUPPORTPILOT_MODEL_ID"
        ),
        model_revision=_optional_env(
            "SUPPORTPILOT_MODEL_REVISION"
        ),
        max_length=int(
            os.getenv(
                "SUPPORTPILOT_MAX_LENGTH",
                "64",
            )
        ),
        min_confidence=float(
            os.getenv(
                "SUPPORTPILOT_MIN_CONFIDENCE",
                "0.70",
            )
        ),
        min_margin=float(
            os.getenv(
                "SUPPORTPILOT_MIN_MARGIN",
                "0.10",
            )
        ),
        fallback_intent=os.getenv(
            "SUPPORTPILOT_FALLBACK_INTENT",
            "fallback",
        ),
        log_level=os.getenv(
            "SUPPORTPILOT_LOG_LEVEL",
            "INFO",
        ).upper(),
    )


settings = get_settings()