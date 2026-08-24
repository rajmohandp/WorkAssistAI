"""Reusable application dependencies for the DocuVerse RAG service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.embeddings import get_embedding_model
from src.rag_chain import create_chat_model
from src.vector_store import connect_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGDependencies:
    """Long-lived provider clients used by question answering."""

    embedding_model: Embeddings
    vector_store: Any
    chat_model: BaseChatModel


@lru_cache(maxsize=1)
def get_rag_dependencies() -> RAGDependencies:
    """Initialize the RAG provider clients once per application process."""

    logger.info(
        "Initializing reusable RAG dependencies",
        extra={"operation": "application", "event": "rag_dependencies_started"},
    )
    embedding_model = get_embedding_model()
    connection = connect_vector_store(embedding_model)
    chat_model = create_chat_model()
    dependencies = RAGDependencies(
        embedding_model=embedding_model,
        vector_store=connection.vector_store,
        chat_model=chat_model,
    )
    logger.info(
        "Reusable RAG dependencies initialized",
        extra={"operation": "application", "event": "rag_dependencies_ready"},
    )
    return dependencies


def clear_rag_dependencies() -> None:
    """Clear cached dependencies for tests or deliberate configuration reloads."""

    get_rag_dependencies.cache_clear()
