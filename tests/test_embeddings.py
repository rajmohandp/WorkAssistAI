"""Tests for the provider-neutral embedding boundary."""

import pytest
from langchain_core.embeddings import Embeddings

from src.config import EmbeddingSettings
from src.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingGenerationError,
    generate_sample_embedding,
    get_embedding_model,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text))] * 4 for text in texts]

    def embed_query(self, text):
        return [float(len(text))] * 4


class BrokenEmbeddings(FakeEmbeddings):
    def embed_query(self, text):
        raise RuntimeError("provider response with internal details")


def test_creates_openai_embedding_model():
    settings = EmbeddingSettings(
        provider="openai",
        model="text-embedding-3-small",
        api_key="test-key",
    )

    model = get_embedding_model(settings)

    assert model.model == "text-embedding-3-small"
    assert model.dimensions == 1024
    assert "test-key" not in repr(settings)


def test_generates_one_sample_embedding_through_langchain_interface():
    vector = generate_sample_embedding("Sample DocuVerse chunk", model=FakeEmbeddings())

    assert len(vector) == 4


def test_rejects_empty_sample():
    with pytest.raises(EmbeddingGenerationError, match="empty"):
        generate_sample_embedding("   ", model=FakeEmbeddings())


def test_wraps_provider_errors_without_exposing_details():
    with pytest.raises(EmbeddingGenerationError) as caught:
        generate_sample_embedding("sample", model=BrokenEmbeddings())

    assert "internal details" not in str(caught.value)


def test_rejects_unsupported_provider():
    settings = EmbeddingSettings(
        provider="local",
        model="future-model",
        api_key="not-used",
    )

    with pytest.raises(EmbeddingConfigurationError, match="Unsupported"):
        get_embedding_model(settings)


def test_rejects_invalid_embedding_dimensions():
    with pytest.raises(ValueError, match="DIMENSIONS"):
        EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=0,
        )
