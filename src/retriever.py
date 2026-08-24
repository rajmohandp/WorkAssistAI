"""Semantic similarity retrieval from the existing Pinecone index."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.config import get_retrieval_settings
from src.vector_store import connect_vector_store

logger = logging.getLogger(__name__)
ALLOWED_FILTER_FIELDS = frozenset({"filename", "file_type"})
MetadataFilter = dict[str, str]


class RetrievalError(RuntimeError):
    """Raised when semantic retrieval cannot be completed safely."""


@dataclass(frozen=True)
class SearchResult:
    """One ranked semantic-search result with its Pinecone score."""

    document: Document
    score: float | None
    rank: int


def _normalize_numeric_metadata(document: Document) -> Document:
    """Restore integer page/chunk fields serialized as floats by Pinecone."""

    metadata = dict(document.metadata)
    for key in ("page", "page_number", "chunk_index"):
        value = metadata.get(key)
        if isinstance(value, float) and value.is_integer():
            metadata[key] = int(value)
    return Document(page_content=document.page_content, metadata=metadata)


def _resolve_top_k(top_k: int | None) -> int:
    resolved = top_k if top_k is not None else get_retrieval_settings().top_k
    if resolved <= 0:
        raise ValueError("top_k must be greater than zero.")
    return resolved


def build_pinecone_filter(
    metadata_filter: MetadataFilter | None,
) -> dict[str, dict[str, str]] | None:
    """Validate application filters and create a Pinecone metadata expression."""

    if not metadata_filter:
        return None
    unsupported = set(metadata_filter) - ALLOWED_FILTER_FIELDS
    if unsupported:
        raise ValueError("Unsupported document metadata filter.")
    normalized = {
        key: value.strip()
        for key, value in metadata_filter.items()
        if isinstance(value, str) and value.strip()
    }
    if len(normalized) != len(metadata_filter):
        raise ValueError("Metadata filter values cannot be empty.")
    return {key: {"$eq": value} for key, value in normalized.items()}


def create_retriever(
    embedding_model: Embeddings | None = None,
    *,
    top_k: int | None = None,
    vector_store: Any | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> BaseRetriever:
    """Create a LangChain retriever backed by the configured Pinecone index."""

    resolved_top_k = _resolve_top_k(top_k)
    store = vector_store or connect_vector_store(embedding_model).vector_store
    search_kwargs: dict[str, Any] = {"k": resolved_top_k}
    if pinecone_filter := build_pinecone_filter(metadata_filter):
        search_kwargs["filter"] = pinecone_filter
    return store.as_retriever(search_kwargs=search_kwargs)


def search_documents(
    question: str,
    embedding_model: Embeddings | None = None,
    *,
    top_k: int | None = None,
    vector_store: Any | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[SearchResult]:
    """Embed a question and return ranked Pinecone chunks with scores."""

    if not question.strip():
        raise ValueError("Enter a question before searching.")

    resolved_top_k = _resolve_top_k(top_k)
    try:
        logger.debug(
            "Semantic retrieval started",
            extra={
                "operation": "retrieval",
                "event": "search_started",
                "top_k": resolved_top_k,
                "filtered": bool(metadata_filter),
            },
        )
        store = vector_store or connect_vector_store(embedding_model).vector_store
        search_kwargs: dict[str, Any] = {"k": resolved_top_k}
        if pinecone_filter := build_pinecone_filter(metadata_filter):
            search_kwargs["filter"] = pinecone_filter
        matches = store.similarity_search_with_score(question.strip(), **search_kwargs)
        results = [
            SearchResult(
                document=_normalize_numeric_metadata(document),
                score=float(score),
                rank=rank,
            )
            for rank, (document, score) in enumerate(matches, start=1)
        ]
        logger.info(
            "Pinecone retrieval completed",
            extra={
                "operation": "retrieval",
                "event": "pinecone_retrieval_completed",
                "result_count": len(results),
                "chunks_retrieved": len(results),
            },
        )
        return results
    except RetrievalError:
        raise
    except Exception as exc:
        logger.error(
            "Semantic retrieval failed",
            extra={
                "operation": "retrieval",
                "event": "search_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise RetrievalError("Document search could not be completed.") from exc
