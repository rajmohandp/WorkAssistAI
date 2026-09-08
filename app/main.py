"""FastAPI entry point for DocuVerse."""

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth_routes import router as auth_router
from app.api.routes import ask, health, readiness, router
from app.api.schemas import AskResponse, HealthResponse, ReadinessResponse
from app.core.config import get_api_settings
from app.core.database import DatabaseConnectionError
from app.repositories.pto_repository import PTORepositoryError
from src.config import get_environment_settings
from src.embeddings import EmbeddingError
from src.logging_config import configure_logging
from src.rag_chain import RAGError
from src.retriever import RetrievalError
from src.s3_loader import S3RepositoryError
from src.vector_store import VectorStoreError

configure_logging()
settings = get_api_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate deployment configuration before accepting traffic."""

    get_environment_settings().validate_backend_startup()
    yield

app = FastAPI(
    title=settings.title,
    version=settings.version,
    description="HTTP API for agentic employee support, PTO, and grounded search.",
    lifespan=lifespan,
)
environment = get_environment_settings()
environment.validate_cors_configuration()
if environment.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=environment.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(router, prefix=settings.api_prefix)
app.include_router(auth_router, prefix=settings.api_prefix)

_PROVIDER_ERRORS = (
    DatabaseConnectionError,
    PTORepositoryError,
    EmbeddingError,
    RAGError,
    RetrievalError,
    S3RepositoryError,
    VectorStoreError,
)


async def provider_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return a stable response when an external dependency is unavailable."""

    logger.error(
        "External provider request failed",
        extra={
            "operation": "application",
            "event": "provider_request_failed",
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "A required service is currently unavailable."},
    )


for provider_error in _PROVIDER_ERRORS:
    app.add_exception_handler(provider_error, provider_exception_handler)


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Prevent implementation details and tracebacks from reaching clients."""

    logger.error(
        "Unhandled API request failure",
        extra={
            "operation": "application",
            "event": "unhandled_api_failure",
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred."},
    )


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
app.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"description": "Database unavailable"}},
    tags=["system"],
)(readiness)
app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    tags=["rag"],
)(ask)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
