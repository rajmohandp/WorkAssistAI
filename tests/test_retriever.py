"""Tests for semantic retrieval without an LLM."""

import pytest
from langchain_core.documents import Document

from src.config import RetrievalSettings
from src.retriever import (
    RetrievalError,
    build_pinecone_filter,
    create_retriever,
    search_documents,
)


class FakeRetriever:
    pass


class FakeVectorStore:
    def __init__(self, matches=None, error=None):
        self.matches = matches or []
        self.error = error
        self.search_kwargs = None
        self.query = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever()

    def similarity_search_with_score(self, query, **search_kwargs):
        self.query = query
        self.search_kwargs = search_kwargs
        if self.error:
            raise self.error
        return self.matches[: search_kwargs["k"]]


def test_creates_langchain_retriever_with_configured_top_k():
    store = FakeVectorStore()

    retriever = create_retriever(top_k=7, vector_store=store)

    assert isinstance(retriever, FakeRetriever)
    assert store.search_kwargs == {"k": 7}


def test_returns_ranked_documents_text_metadata_and_scores():
    matches = [
        (
            Document(
                page_content="Relevant content",
                metadata={
                    "filename": "guide.pdf",
                    "page_number": 3.0,
                    "chunk_index": 1.0,
                },
            ),
            0.91,
        )
    ]
    store = FakeVectorStore(matches)

    results = search_documents("  What is relevant?  ", top_k=5, vector_store=store)

    assert store.query == "What is relevant?"
    assert store.search_kwargs == {"k": 5}
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(0.91)
    assert results[0].document.page_content == "Relevant content"
    assert results[0].document.metadata["page_number"] == 3
    assert results[0].document.metadata["chunk_index"] == 1
    assert isinstance(results[0].document.metadata["page_number"], int)


def test_passes_document_filter_directly_to_pinecone_search():
    store = FakeVectorStore()

    search_documents(
        "policy",
        vector_store=store,
        metadata_filter={"filename": "Employee Handbook.pdf"},
    )

    assert store.search_kwargs == {
        "k": 5,
        "filter": {"filename": {"$eq": "Employee Handbook.pdf"}},
    }


def test_passes_file_type_filter_to_langchain_retriever():
    store = FakeVectorStore()

    create_retriever(
        vector_store=store,
        metadata_filter={"file_type": "pdf"},
    )

    assert store.search_kwargs == {
        "k": 5,
        "filter": {"file_type": {"$eq": "pdf"}},
    }


def test_rejects_unsupported_or_empty_metadata_filters():
    with pytest.raises(ValueError):
        build_pinecone_filter({"s3_key": "secret/path"})
    with pytest.raises(ValueError):
        build_pinecone_filter({"filename": " "})


def test_rejects_empty_question_and_invalid_top_k():
    with pytest.raises(ValueError, match="question"):
        search_documents(" ", vector_store=FakeVectorStore())
    with pytest.raises(ValueError, match="top_k"):
        create_retriever(top_k=0, vector_store=FakeVectorStore())
    with pytest.raises(ValueError):
        RetrievalSettings(top_k=0)


def test_wraps_provider_errors_without_exposing_details():
    store = FakeVectorStore(error=RuntimeError("provider secret details"))

    with pytest.raises(RetrievalError) as caught:
        search_documents("question", vector_store=store)

    assert "provider secret details" not in str(caught.value)
