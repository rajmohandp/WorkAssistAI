"""Split loaded documents into metadata-rich retrieval chunks."""

from __future__ import annotations

import logging
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document], *, settings: Settings | None = None
) -> list[Document]:
    """Recursively split documents while preserving their original metadata."""

    resolved_settings = settings or get_settings()
    logger.debug(
        "Document chunking started",
        extra={
            "operation": "chunking",
            "event": "chunking_started",
            "document_count": len(documents),
            "chunk_size": resolved_settings.chunk_size,
            "chunk_overlap": resolved_settings.chunk_overlap,
        },
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_settings.chunk_size,
        chunk_overlap=resolved_settings.chunk_overlap,
    )
    chunks: list[Document] = []

    for document in documents:
        document_chunks = splitter.split_documents([document])
        non_empty_chunks = [
            chunk for chunk in document_chunks if chunk.page_content.strip()
        ]

        for chunk_index, chunk in enumerate(non_empty_chunks):
            identity = "|".join(
                (
                    str(
                        chunk.metadata.get("document_id")
                        or chunk.metadata.get("source", "")
                    ),
                    str(chunk.metadata.get("document_version", "")),
                    str(chunk.metadata.get("page_number", "")),
                    str(chunk_index),
                    sha256(chunk.page_content.encode("utf-8")).hexdigest(),
                )
            )
            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": str(uuid5(NAMESPACE_URL, identity)),
                        "chunk_index": chunk_index,
                    },
                )
            )

    logger.info(
        "Document chunking completed",
        extra={
            "operation": "chunking",
            "event": "chunking_completed",
            "document_count": len(documents),
            "chunk_count": len(chunks),
        },
    )
    return chunks
