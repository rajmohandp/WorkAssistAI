"""Provider-neutral embedding model boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import EmbeddingSettings, get_embedding_settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Base error for embedding configuration or generation failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when the embedding provider is not configured correctly."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when an embedding provider cannot generate a vector."""


def _create_openai_embeddings(settings: EmbeddingSettings) -> Embeddings:
    return OpenAIEmbeddings(
        api_key=settings.api_key,
        model=settings.model,
        dimensions=settings.dimensions,
    )


_PROVIDER_FACTORIES: dict[str, Callable[[EmbeddingSettings], Embeddings]] = {
    "openai": _create_openai_embeddings,
}


def get_embedding_model(
    settings: EmbeddingSettings | None = None,
) -> Embeddings:
    """Return a LangChain embedding implementation for the selected provider."""

    try:
        resolved_settings = settings or get_embedding_settings()
        factory = _PROVIDER_FACTORIES.get(resolved_settings.provider)
        if factory is None:
            raise EmbeddingConfigurationError(
                f"Unsupported embedding provider: {resolved_settings.provider}"
            )
        model = factory(resolved_settings)
        logger.info(
            "Embedding model initialized",
            extra={
                "operation": "embedding",
                "event": "model_initialized",
                "provider": resolved_settings.provider,
                "model": resolved_settings.model,
                "dimensions": resolved_settings.dimensions,
            },
        )
        return model
    except EmbeddingConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise EmbeddingConfigurationError(
            "The embedding provider configuration is invalid."
        ) from exc


def generate_sample_embedding(
    text: str, *, model: Embeddings | None = None
) -> list[float]:
    """Embed one non-empty sample without logging or displaying its vector."""

    if not text.strip():
        raise EmbeddingGenerationError("Cannot embed an empty sample chunk.")

    try:
        logger.debug(
            "Embedding generation started",
            extra={"operation": "embedding", "event": "generation_started"},
        )
        embedding = (model or get_embedding_model()).embed_query(text)
        logger.info(
            "Embedding generation completed",
            extra={
                "operation": "embedding",
                "event": "generation_completed",
                "dimensions": len(embedding),
            },
        )
        return embedding
    except EmbeddingError:
        raise
    except Exception as exc:
        logger.error(
            "Embedding generation failed",
            extra={
                "operation": "embedding",
                "event": "generation_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise EmbeddingGenerationError(
            "The embedding provider could not generate an embedding."
        ) from exc
