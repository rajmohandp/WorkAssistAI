"""Tests for Pinecone connection validation and batched ingestion."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import PineconeSettings
from src.s3_loader import DocumentLoadResult
from src.vector_store import (
    DocumentSyncState,
    IndexedDocument,
    VectorDimensionMismatchError,
    _metadata_version,
    classify_document,
    delete_removed_documents,
    find_deleted_documents,
    get_vector_store_status,
    index_documents,
    synchronize_s3_documents,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 2.0, 3.0]


class Description:
    dimension = 3
    host = "index.example.invalid"


class FakeIndex:
    def __init__(self, existing=None, vector_metadata=None):
        self.config = type("Config", (), {"host": "index.example.invalid"})()
        self.existing = set(existing or [])
        self.vector_metadata = vector_metadata or {}
        self.fetch_batches = []
        self.deleted = []

    def fetch(self, ids, **kwargs):
        self.fetch_batches.append(ids)
        return {
            "vectors": {
                item: {"metadata": self.vector_metadata.get(item, {})}
                for item in ids
                if item in self.existing
            }
        }

    def list(self, **kwargs):
        return iter([list(self.existing)])

    def delete(self, ids, **kwargs):
        self.deleted.extend(ids)
        self.existing.difference_update(ids)


class FakePinecone:
    def __init__(self, index=None, dimension=3):
        self.index = index or FakeIndex()
        self.description = Description()
        self.description.dimension = dimension

    def describe_index(self, name):
        assert name == "docuverse"
        return self.description

    def Index(self, host):
        assert host == self.description.host
        return self.index


def settings(batch_size=2):
    return PineconeSettings(
        api_key="test-key",
        index_name="docuverse",
        batch_size=batch_size,
    )


def chunks():
    return [
        Document(page_content=f"content {index}", metadata={"chunk_id": f"id-{index}"})
        for index in range(5)
    ]


def test_rejects_index_dimension_mismatch():
    with pytest.raises(VectorDimensionMismatchError):
        index_documents(
            chunks(),
            FakeEmbeddings(),
            settings=settings(),
            pinecone_client=FakePinecone(dimension=4),
        )


def test_batches_inserts_and_skips_existing_vectors(monkeypatch):
    added_batches = []

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, documents, ids):
            added_batches.append(ids)
            return ids

    monkeypatch.setattr("src.vector_store.PineconeVectorStore", FakeVectorStore)
    index = FakeIndex(existing={"id-1", "id-4"})
    progress = []

    result = index_documents(
        chunks(),
        FakeEmbeddings(),
        settings=settings(batch_size=2),
        pinecone_client=FakePinecone(index=index),
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert result.chunks_processed == 5
    assert result.vectors_inserted == 3
    assert result.vectors_skipped == 2
    assert result.failures == []
    assert added_batches == [["id-0"], ["id-2", "id-3"]]
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_skips_duplicate_ids_in_input(monkeypatch):
    added_ids = []

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, documents, ids):
            added_ids.extend(ids)

    monkeypatch.setattr("src.vector_store.PineconeVectorStore", FakeVectorStore)
    duplicate_chunks = chunks()[:1] * 2

    result = index_documents(
        duplicate_chunks,
        FakeEmbeddings(),
        settings=settings(),
        pinecone_client=FakePinecone(),
    )

    assert added_ids == ["id-0"]
    assert result.vectors_inserted == 1
    assert result.vectors_skipped == 1


def test_force_reindex_overwrites_existing_ids(monkeypatch):
    added_ids = []

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, documents, ids):
            added_ids.extend(ids)

    monkeypatch.setattr("src.vector_store.PineconeVectorStore", FakeVectorStore)
    index = FakeIndex(existing={"id-0", "id-1"})

    result = index_documents(
        chunks()[:2],
        FakeEmbeddings(),
        settings=settings(),
        pinecone_client=FakePinecone(index=index),
        force_reindex=True,
    )

    assert added_ids == ["id-0", "id-1"]
    assert index.fetch_batches == []
    assert result.vectors_inserted == 2
    assert result.vectors_skipped == 0


def test_reports_vector_and_unique_document_counts():
    class StatusStats:
        total_vector_count = 3

    class StatusIndex:
        def describe_index_stats(self):
            return StatusStats()

        def list(self, **kwargs):
            return iter([["id-1", "id-2"], ["id-3"]])

        def fetch(self, ids, **kwargs):
            sources = {
                "id-1": "first.pdf",
                "id-2": "first.pdf",
                "id-3": "second.txt",
            }
            return {
                "vectors": {
                    vector_id: {"metadata": {"s3_key": sources[vector_id]}}
                    for vector_id in ids
                }
            }

    status = get_vector_store_status(
        settings=settings(),
        pinecone_client=FakePinecone(index=StatusIndex()),
    )

    assert status.index_name == "docuverse"
    assert status.vector_count == 3
    assert status.indexed_documents == 2


def s3_item(key, etag):
    return {
        "file_name": key,
        "s3_key": key,
        "file_type": "txt",
        "file_size": 10,
        "last_modified": "2026-08-20T12:00:00+00:00",
        "etag": etag,
    }


def test_classifies_new_updated_and_unchanged_documents():
    item = s3_item("policy.txt", "etag-1")
    current_version = _metadata_version(item)

    assert classify_document(item, {}) == DocumentSyncState.NEW
    assert classify_document(
        item,
        {
            "policy.txt": IndexedDocument(current_version, ("chunk-1",))
        },
    ) == DocumentSyncState.UNCHANGED
    changed = s3_item("policy.txt", "etag-2")
    assert classify_document(
        changed,
        {
            "policy.txt": IndexedDocument(current_version, ("chunk-1",))
        },
    ) == DocumentSyncState.UPDATED


def test_incremental_sync_skips_unchanged_and_replaces_updated(monkeypatch):
    unchanged = s3_item("unchanged.txt", "same")
    updated = s3_item("updated.txt", "new")
    new = s3_item("new.txt", "first")
    index = FakeIndex(
        existing={"unchanged-chunk", "old-1", "old-2"},
        vector_metadata={
            "unchanged-chunk": {
                "s3_key": "unchanged.txt",
                "document_version": _metadata_version(unchanged),
            },
            "old-1": {"s3_key": "updated.txt", "document_version": "old"},
            "old-2": {"s3_key": "updated.txt", "document_version": "old"},
        },
    )
    loaded_keys = []
    added_ids = []

    monkeypatch.setattr(
        "src.vector_store.list_documents",
        lambda bucket, s3_client: [unchanged, updated, new],
    )

    def fake_load(bucket, document_metadata, s3_client):
        item = document_metadata[0]
        loaded_keys.append(item["s3_key"])
        return DocumentLoadResult(
            documents=[
                Document(
                    page_content=f"Content for {item['s3_key']}",
                    metadata={"s3_key": item["s3_key"]},
                )
            ],
            loaded_files=1,
            failed_documents=[],
        )

    monkeypatch.setattr("src.vector_store.load_documents", fake_load)
    monkeypatch.setattr(
        "src.vector_store.chunk_documents",
        lambda documents: [
            Document(
                page_content=documents[0].page_content,
                metadata={
                    **documents[0].metadata,
                    "chunk_id": f"new-{documents[0].metadata['s3_key']}",
                },
            )
        ],
    )

    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

        def add_documents(self, documents, ids):
            added_ids.extend(ids)

    monkeypatch.setattr("src.vector_store.PineconeVectorStore", FakeVectorStore)

    result = synchronize_s3_documents(
        FakeEmbeddings(),
        bucket_name="documents",
        settings=settings(),
        pinecone_client=FakePinecone(index=index),
    )

    assert loaded_keys == ["updated.txt", "new.txt"]
    assert set(index.deleted) == {"old-1", "old-2"}
    assert set(added_ids) == {"new-updated.txt", "new-new.txt"}
    assert result.documents_scanned == 3
    assert result.new_documents == 1
    assert result.updated_documents == 1
    assert result.unchanged_documents == 1
    assert result.failed_documents == 0
    assert result.chunks_added == 2
    assert result.chunks_removed == 2
    assert result.documents_removed_from_s3 == 0
    assert result.vectors_removed_from_pinecone == 0
    assert result.documents_added == 1
    assert result.documents_updated == 1
    assert result.documents_removed == 0


def test_detects_and_deletes_only_documents_removed_from_s3(caplog):
    caplog.set_level("INFO")
    indexed = {
        "active.txt": IndexedDocument("current", ("active-1", "active-2")),
        "deleted.pdf": IndexedDocument("old", ("deleted-1", "deleted-2")),
    }
    index = FakeIndex(existing={"active-1", "active-2", "deleted-1", "deleted-2"})

    removed = find_deleted_documents({"active.txt"}, indexed)
    removed_count = delete_removed_documents(index, removed, "")

    assert set(removed) == {"deleted.pdf"}
    assert set(index.deleted) == {"deleted-1", "deleted-2"}
    assert {"active-1", "active-2"}.isdisjoint(index.deleted)
    assert removed_count == 2
    assert any(
        getattr(record, "document", None) == "deleted.pdf"
        for record in caplog.records
    )


def test_sync_removes_documents_missing_from_s3(monkeypatch):
    class FakeVectorStore:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("src.vector_store.PineconeVectorStore", FakeVectorStore)
    active = s3_item("active.txt", "same")
    index = FakeIndex(
        existing={"active-1", "deleted-1", "deleted-2"},
        vector_metadata={
            "active-1": {
                "s3_key": "active.txt",
                "document_version": _metadata_version(active),
            },
            "deleted-1": {"s3_key": "deleted.txt", "document_version": "old"},
            "deleted-2": {"s3_key": "deleted.txt", "document_version": "old"},
        },
    )
    monkeypatch.setattr(
        "src.vector_store.list_documents", lambda bucket, s3_client: [active]
    )

    result = synchronize_s3_documents(
        FakeEmbeddings(),
        bucket_name="documents",
        settings=settings(),
        pinecone_client=FakePinecone(index=index),
    )

    assert set(index.deleted) == {"deleted-1", "deleted-2"}
    assert result.documents_removed_from_s3 == 1
    assert result.vectors_removed_from_pinecone == 2
    assert result.documents_added == 0
    assert result.documents_updated == 0
    assert result.documents_removed == 1
    assert result.unchanged_documents == 1
