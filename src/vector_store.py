"""Pinecone vector-store connection and batched document ingestion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import PineconeSettings, get_pinecone_settings
from src.document_processor import chunk_documents
from src.embeddings import get_embedding_model
from src.s3_loader import (
    FailedDocument,
    S3DocumentMetadata,
    get_bucket_name,
    list_documents,
    load_documents,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int], None]


class VectorStoreError(RuntimeError):
    """Base error for Pinecone connection or ingestion failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when Pinecone configuration is invalid."""


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when the index and embedding dimensions differ."""


class BatchFailure(TypedDict):
    """Safe details for one failed ingestion batch."""

    batch: int
    chunks: int
    error: str


@dataclass(frozen=True)
class IngestionStatistics:
    """Summary of one Pinecone ingestion run."""

    chunks_processed: int
    vectors_inserted: int
    vectors_skipped: int
    failures: list[BatchFailure]


class DocumentSyncState(StrEnum):
    """Version comparison result for one S3 object."""

    NEW = "NEW"
    UPDATED = "UPDATED"
    UNCHANGED = "UNCHANGED"


@dataclass(frozen=True)
class IndexedDocument:
    """Version and vector IDs currently stored for one S3 key."""

    document_version: str
    vector_ids: tuple[str, ...]


@dataclass(frozen=True)
class SyncStatistics:
    """Document-level summary for one incremental S3 synchronization."""

    documents_scanned: int
    new_documents: int
    updated_documents: int
    unchanged_documents: int
    failed_documents: int
    chunks_added: int
    chunks_removed: int
    documents_removed_from_s3: int
    vectors_removed_from_pinecone: int
    documents_added: int
    documents_updated: int
    documents_removed: int
    failures: list[FailedDocument]


@dataclass(frozen=True)
class PineconeConnection:
    """Validated handles for the SDK and LangChain vector store."""

    index: Any
    vector_store: PineconeVectorStore
    settings: PineconeSettings
    dimension: int


@dataclass(frozen=True)
class VectorStoreStatus:
    """Read-only Pinecone repository status for application monitoring."""

    index_name: str
    vector_count: int
    indexed_documents: int | None


def connect_vector_store(
    embedding_model: Embeddings | None = None,
    *,
    settings: PineconeSettings | None = None,
    pinecone_client: Any | None = None,
) -> PineconeConnection:
    """Connect to an existing index and validate its embedding dimension."""

    try:
        resolved_settings = settings or get_pinecone_settings()
        logger.debug(
            "Pinecone connection validation started",
            extra={"operation": "pinecone", "event": "connection_started"},
        )
        resolved_embeddings = embedding_model or get_embedding_model()
        client = pinecone_client or Pinecone(api_key=resolved_settings.api_key)
        description = client.describe_index(resolved_settings.index_name)
        index_dimension = int(description.dimension)
        embedding_dimension = len(
            resolved_embeddings.embed_query("DocuVerse dimension validation")
        )

        if index_dimension != embedding_dimension:
            raise VectorDimensionMismatchError(
                "Pinecone index dimension does not match the embedding model "
                f"({index_dimension} != {embedding_dimension})."
            )

        index = client.Index(host=description.host)
        vector_store = PineconeVectorStore(
            index=index,
            embedding=resolved_embeddings,
            namespace=resolved_settings.namespace or None,
        )
        connection = PineconeConnection(
            index=index,
            vector_store=vector_store,
            settings=resolved_settings,
            dimension=index_dimension,
        )
        logger.info(
            "Pinecone connection validated",
            extra={
                "operation": "pinecone",
                "event": "connection_completed",
                "index": resolved_settings.index_name,
                "dimension": index_dimension,
            },
        )
        return connection
    except VectorStoreError:
        raise
    except (TypeError, ValueError) as exc:
        raise VectorStoreConfigurationError(
            "Pinecone configuration is invalid."
        ) from exc
    except Exception as exc:
        logger.error(
            "Pinecone connection failed",
            extra={
                "operation": "pinecone",
                "event": "connection_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise VectorStoreError("Could not connect to the Pinecone index.") from exc


def get_vector_store_status(
    *,
    settings: PineconeSettings | None = None,
    pinecone_client: Any | None = None,
) -> VectorStoreStatus:
    """Return index counts without creating embeddings or modifying vectors."""

    try:
        resolved_settings = settings or get_pinecone_settings()
        client = pinecone_client or Pinecone(api_key=resolved_settings.api_key)
        description = client.describe_index(resolved_settings.index_name)
        index = client.Index(host=description.host)
        stats = index.describe_index_stats()
        vector_count = int(stats.total_vector_count)

        return VectorStoreStatus(
            index_name=resolved_settings.index_name,
            vector_count=vector_count,
            indexed_documents=None,
        )
    except Exception as exc:
        logger.error(
            "Pinecone status check failed",
            extra={
                "operation": "pinecone",
                "event": "status_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise VectorStoreError("Could not connect to the Pinecone index.") from exc


def _existing_ids(index: Any, ids: list[str], namespace: str) -> set[str]:
    kwargs = {"namespace": namespace} if namespace else {}
    response = index.fetch(ids=ids, **kwargs)
    vectors = response.vectors if hasattr(response, "vectors") else response["vectors"]
    return set(vectors)


def _namespace_kwargs(namespace: str) -> dict[str, str]:
    return {"namespace": namespace} if namespace else {}


def _indexed_documents(index: Any, namespace: str) -> dict[str, IndexedDocument]:
    """Build the current S3-version manifest from Pinecone vector metadata."""

    kwargs = _namespace_kwargs(namespace)
    versions: dict[str, str] = {}
    vector_ids: dict[str, list[str]] = {}
    for id_batch in index.list(**kwargs):
        ids = list(id_batch)
        if not ids:
            continue
        response = index.fetch(ids=ids, **kwargs)
        vectors = response.vectors if hasattr(response, "vectors") else response["vectors"]
        for vector_id, vector in vectors.items():
            metadata = (
                vector.metadata
                if hasattr(vector, "metadata")
                else vector.get("metadata", {})
            )
            s3_key = metadata.get("s3_key")
            if not s3_key:
                continue
            key = str(s3_key)
            vector_ids.setdefault(key, []).append(str(vector_id))
            version = str(metadata.get("document_version", ""))
            if key not in versions:
                versions[key] = version
            elif versions[key] != version:
                versions[key] = ""

    return {
        key: IndexedDocument(
            document_version=versions.get(key, ""),
            vector_ids=tuple(ids),
        )
        for key, ids in vector_ids.items()
    }


def _metadata_version(item: S3DocumentMetadata) -> str:
    """Return the version hash produced by the S3 loader metadata boundary."""

    from datetime import datetime
    from hashlib import sha256

    last_modified = item.get("last_modified")
    normalized_modified = (
        last_modified.isoformat()
        if isinstance(last_modified, datetime)
        else str(last_modified or "")
    )
    identity = "|".join(
        (
            item.get("etag", ""),
            str(item.get("file_size", 0)),
            normalized_modified,
        )
    )
    return sha256(identity.encode("utf-8")).hexdigest()


def classify_document(
    item: S3DocumentMetadata,
    indexed: dict[str, IndexedDocument],
) -> DocumentSyncState:
    """Classify an S3 object against the indexed document manifest."""

    current = indexed.get(item["s3_key"])
    if current is None:
        return DocumentSyncState.NEW
    if current.document_version == _metadata_version(item):
        return DocumentSyncState.UNCHANGED
    return DocumentSyncState.UPDATED


def _delete_vectors(index: Any, ids: tuple[str, ...], namespace: str) -> int:
    if not ids:
        return 0
    index.delete(ids=list(ids), **_namespace_kwargs(namespace))
    return len(ids)


def find_deleted_documents(
    current_s3_keys: set[str],
    indexed: dict[str, IndexedDocument],
) -> dict[str, IndexedDocument]:
    """Return only indexed S3 keys that are absent from the current bucket."""

    return {
        s3_key: document
        for s3_key, document in indexed.items()
        if s3_key not in current_s3_keys
    }


def delete_removed_documents(
    index: Any,
    removed: dict[str, IndexedDocument],
    namespace: str,
) -> int:
    """Delete exact vector IDs for S3 documents confirmed as removed."""

    removed_vectors = 0
    for s3_key, document in removed.items():
        removed_vectors += _delete_vectors(index, document.vector_ids, namespace)
        logger.info(
            "Deleted document vectors removed from Pinecone",
            extra={
                "operation": "pinecone",
                "event": "deleted_document_removed",
                "document": s3_key,
                "vector_count": len(document.vector_ids),
            },
        )
    return removed_vectors


def synchronize_s3_documents(
    embedding_model: Embeddings | None = None,
    *,
    bucket_name: str | None = None,
    settings: PineconeSettings | None = None,
    pinecone_client: Any | None = None,
    s3_client: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SyncStatistics:
    """Synchronize only new or changed S3 documents into Pinecone."""

    resolved_bucket = bucket_name or get_bucket_name()
    logger.info(
        "Incremental document synchronization started",
        extra={"operation": "pinecone", "event": "sync_started"},
    )
    resolved_embeddings = embedding_model or get_embedding_model()
    connection = connect_vector_store(
        resolved_embeddings,
        settings=settings,
        pinecone_client=pinecone_client,
    )
    items = list_documents(resolved_bucket, s3_client=s3_client)
    indexed = _indexed_documents(connection.index, connection.settings.namespace)
    removed_documents = find_deleted_documents(
        {item["s3_key"] for item in items},
        indexed,
    )
    vectors_removed_from_pinecone = delete_removed_documents(
        connection.index,
        removed_documents,
        connection.settings.namespace,
    )
    states = {item["s3_key"]: classify_document(item, indexed) for item in items}
    failures: list[FailedDocument] = []
    chunks_added = 0
    chunks_removed = 0
    documents_added = 0
    documents_updated = 0
    processable = [
        item for item in items if states[item["s3_key"]] != DocumentSyncState.UNCHANGED
    ]

    for number, item in enumerate(processable, start=1):
        loaded = load_documents(
            resolved_bucket,
            document_metadata=[item],
            s3_client=s3_client,
        )
        if loaded.failed_documents:
            failures.extend(loaded.failed_documents)
        else:
            chunks = chunk_documents(loaded.documents)
            if not chunks:
                failures.append(
                    {
                        "filename": item["file_name"],
                        "s3_key": item["s3_key"],
                        "error": "The document did not contain indexable text.",
                    }
                )
            else:
                state = states[item["s3_key"]]
                if state == DocumentSyncState.UPDATED:
                    chunks_removed += _delete_vectors(
                        connection.index,
                        indexed[item["s3_key"]].vector_ids,
                        connection.settings.namespace,
                    )
                result = index_documents(
                    chunks,
                    resolved_embeddings,
                    settings=connection.settings,
                    pinecone_client=pinecone_client,
                    force_reindex=True,
                )
                chunks_added += result.vectors_inserted
                if result.failures:
                    failures.append(
                        {
                            "filename": item["file_name"],
                            "s3_key": item["s3_key"],
                            "error": "One or more vector batches could not be indexed.",
                        }
                    )
                elif state == DocumentSyncState.NEW:
                    documents_added += 1
                else:
                    documents_updated += 1
        if progress_callback:
            progress_callback(number, len(processable))

    counts = {state: list(states.values()).count(state) for state in DocumentSyncState}
    statistics = SyncStatistics(
        documents_scanned=len(items),
        new_documents=counts[DocumentSyncState.NEW],
        updated_documents=counts[DocumentSyncState.UPDATED],
        unchanged_documents=counts[DocumentSyncState.UNCHANGED],
        failed_documents=len(failures),
        chunks_added=chunks_added,
        chunks_removed=chunks_removed,
        documents_removed_from_s3=len(removed_documents),
        vectors_removed_from_pinecone=vectors_removed_from_pinecone,
        documents_added=documents_added,
        documents_updated=documents_updated,
        documents_removed=len(removed_documents),
        failures=failures,
    )
    logger.info(
        "Incremental document synchronization completed",
        extra={
            "operation": "pinecone",
            "event": "sync_completed",
            "documents_scanned": statistics.documents_scanned,
            "documents_added": statistics.documents_added,
            "documents_updated": statistics.documents_updated,
            "documents_removed": statistics.documents_removed,
            "failure_count": statistics.failed_documents,
            "vectors_added": statistics.chunks_added,
            "vectors_removed": (
                statistics.chunks_removed
                + statistics.vectors_removed_from_pinecone
            ),
        },
    )
    return statistics


def index_documents(
    chunks: list[Document],
    embedding_model: Embeddings | None = None,
    *,
    settings: PineconeSettings | None = None,
    pinecone_client: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    force_reindex: bool = False,
) -> IngestionStatistics:
    """Insert chunks in batches, optionally overwriting existing vector IDs."""

    connection = connect_vector_store(
        embedding_model,
        settings=settings,
        pinecone_client=pinecone_client,
    )
    batch_size = connection.settings.batch_size
    inserted = 0
    skipped = 0
    failures: list[BatchFailure] = []
    completed_ids: set[str] = set()
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        ids = [str(chunk.metadata.get("chunk_id", "")) for chunk in batch]
        invalid_count = sum(not chunk_id for chunk_id in ids)
        valid_pairs: list[tuple[Document, str]] = []
        batch_ids: set[str] = set()
        for chunk, chunk_id in zip(batch, ids, strict=True):
            if not chunk_id:
                continue
            if chunk_id in completed_ids or chunk_id in batch_ids:
                skipped += 1
                continue
            batch_ids.add(chunk_id)
            valid_pairs.append((chunk, chunk_id))

        try:
            if not valid_pairs:
                if invalid_count:
                    failures.append(
                        {
                            "batch": batch_number,
                            "chunks": invalid_count,
                            "error": "Chunks were missing chunk_id metadata.",
                        }
                    )
                if progress_callback:
                    progress_callback(batch_number, total_batches)
                continue

            existing = (
                set()
                if force_reindex
                else _existing_ids(
                    connection.index,
                    [chunk_id for _, chunk_id in valid_pairs],
                    connection.settings.namespace,
                )
            )
            new_pairs = [pair for pair in valid_pairs if pair[1] not in existing]
            skipped += len(existing)

            if new_pairs:
                new_documents, new_ids = zip(*new_pairs, strict=True)
                connection.vector_store.add_documents(
                    documents=list(new_documents),
                    ids=list(new_ids),
                )
                inserted += len(new_pairs)
            completed_ids.update(chunk_id for _, chunk_id in valid_pairs)

            if invalid_count:
                failures.append(
                    {
                        "batch": batch_number,
                        "chunks": invalid_count,
                        "error": "Chunks were missing chunk_id metadata.",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - isolate each provider batch.
            logger.warning(
                "Pinecone ingestion batch failed; continuing remaining batches",
                extra={
                    "operation": "pinecone",
                    "event": "batch_failed",
                    "batch": batch_number,
                    "chunk_count": len(valid_pairs),
                    "error_type": type(exc).__name__,
                },
            )
            failures.append(
                {
                    "batch": batch_number,
                    "chunks": len(valid_pairs),
                    "error": "The batch could not be indexed.",
                }
            )

        if progress_callback:
            progress_callback(batch_number, total_batches)

    return IngestionStatistics(
        chunks_processed=len(chunks),
        vectors_inserted=inserted,
        vectors_skipped=skipped,
        failures=failures,
    )
