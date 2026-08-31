"""FastAPI application factory for SupportPilot AI."""

from fastapi import FastAPI

from supportpilot.api.routes import (
    router,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title="SupportPilot AI API",
        description=(
            "REST API for e-commerce customer "
            "support intent classification "
            "powered by DistilBERT."
        ),
        version="0.1.0",
    )

    application.include_router(
        router
    )

    return application


app = create_app()