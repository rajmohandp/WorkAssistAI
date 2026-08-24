"""Tests for recursive document chunking."""

import pytest
from langchain_core.documents import Document

from src.config import Settings
from src.document_processor import chunk_documents


def test_chunks_documents_and_preserves_metadata():
    document = Document(
        page_content="A" * 80,
        metadata={"source": "s3://documents/sample.txt", "page_number": 2},
    )

    chunks = chunk_documents(
        [document], settings=Settings(chunk_size=30, chunk_overlap=5)
    )

    assert len(chunks) == 3
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["source"] == document.metadata["source"] for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 2 for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    repeated_chunks = chunk_documents(
        [document], settings=Settings(chunk_size=30, chunk_overlap=5)
    )
    assert [chunk.metadata["chunk_id"] for chunk in repeated_chunks] == [
        chunk.metadata["chunk_id"] for chunk in chunks
    ]


def test_removes_empty_documents_and_chunks():
    documents = [
        Document(page_content="   \n\t", metadata={"source": "empty"}),
        Document(page_content="Useful content", metadata={"source": "useful"}),
    ]

    chunks = chunk_documents(
        documents, settings=Settings(chunk_size=100, chunk_overlap=10)
    )

    assert len(chunks) == 1
    assert chunks[0].page_content == "Useful content"


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_rejects_invalid_chunk_configuration(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        Settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
