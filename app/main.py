"""FastAPI entry point for DocuVerse."""

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.routes import ask, health, router
from app.api.schemas import AskResponse, HealthResponse
from app.core.config import get_api_settings
from src.logging_config import configure_logging

configure_logging()
settings = get_api_settings()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.title,
    version=settings.version,
    description="HTTP API for grounded search across DocuVerse documents.",
)
app.include_router(router, prefix=settings.api_prefix)


@app.middleware("http")
async def log_request_lifecycle(request: Request, call_next):
    """Log safe request metadata without bodies, queries, or credentials."""

    request_id = uuid4().hex
    started = perf_counter()
    request_fields = {
        "operation": "application",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
    }
    logger.info(
        "API request received",
        extra={**request_fields, "event": "api_request_received"},
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(
            "API request failed",
            extra={
                **request_fields,
                "event": "api_request_exception",
                "error_type": type(exc).__name__,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        raise
    logger.info(
        "API request completed",
        extra={
            **request_fields,
            "event": "api_request_completed",
            "status_code": response.status_code,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        },
    )
    return response


app.get("/health", response_model=HealthResponse, tags=["system"])(health)
app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    tags=["rag"],
)(ask)
