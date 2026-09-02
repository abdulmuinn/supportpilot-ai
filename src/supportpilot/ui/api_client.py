"""HTTP client for the SupportPilot AI API."""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from supportpilot.config import settings


class SupportPilotAPIError(RuntimeError):
    """Raised when the SupportPilot API request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)

        self.status_code = status_code


class SupportPilotAPIClient:
    """Client used by the dashboard to call FastAPI."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_base_url = (
            settings.api_url
            if base_url is None
            else base_url
        )

        self.base_url = (
            resolved_base_url
            .strip()
            .rstrip("/")
        )

        if not self.base_url:
            raise ValueError(
                "API base URL must not be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        self.timeout = timeout

        self._client = client

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one API request and return parsed JSON."""

        request_timeout = (
            self.timeout
            if timeout is None
            else timeout
        )

        url = (
            f"{self.base_url}"
            f"/{endpoint.lstrip('/')}"
        )

        try:
            if self._client is None:
                response = httpx.request(
                    method=method,
                    url=url,
                    json=json,
                    timeout=request_timeout,
                )

            else:
                response = self._client.request(
                    method=method,
                    url=url,
                    json=json,
                    timeout=request_timeout,
                )

        except httpx.TimeoutException as error:
            raise SupportPilotAPIError(
                "SupportPilot API request timed out."
            ) from error

        except httpx.RequestError as error:
            raise SupportPilotAPIError(
                "Unable to connect to SupportPilot API."
            ) from error

        if response.is_error:
            detail = self._extract_error_detail(
                response
            )

            raise SupportPilotAPIError(
                detail,
                status_code=response.status_code,
            )

        try:
            data = response.json()

        except ValueError as error:
            raise SupportPilotAPIError(
                "SupportPilot API returned invalid JSON.",
                status_code=response.status_code,
            ) from error

        if not isinstance(data, dict):
            raise SupportPilotAPIError(
                "SupportPilot API returned an unexpected response."
            )

        return data

    @staticmethod
    def _extract_error_detail(
        response: httpx.Response,
    ) -> str:
        """Extract a readable API error message."""

        try:
            payload = response.json()

        except ValueError:
            return (
                f"SupportPilot API returned "
                f"HTTP {response.status_code}."
            )

        if isinstance(payload, dict):
            detail = payload.get(
                "detail"
            )

            if isinstance(detail, str):
                return detail

        return (
            f"SupportPilot API returned "
            f"HTTP {response.status_code}."
        )

    def health(
        self,
    ) -> dict[str, Any]:
        """Return API health information."""

        return self._request(
            "GET",
            "/health",
        )

    def model_info(
        self,
    ) -> dict[str, Any]:
        """Return public model metadata."""

        return self._request(
            "GET",
            "/model-info",
        )

    def predict(
        self,
        text: str,
    ) -> dict[str, Any]:
        """Predict one customer support message."""

        return self._request(
            "POST",
            "/predict",
            json={
                "text": text,
            },
        )

    def predict_top_k(
        self,
        text: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Return ranked intent candidates."""

        return self._request(
            "POST",
            "/predict/top-k",
            json={
                "text": text,
                "top_k": top_k,
            },
        )

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Predict multiple customer support messages."""

        return self._request(
            "POST",
            "/predict/batch",
            json={
                "texts": list(texts),
                "batch_size": batch_size,
            },
            timeout=60.0,
        )