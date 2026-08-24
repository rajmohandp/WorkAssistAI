"""Tests for process-wide reusable RAG dependencies."""

from app.core.dependencies import clear_rag_dependencies, get_rag_dependencies


def test_rag_dependencies_are_initialized_once_and_reused(monkeypatch):
    clear_rag_dependencies()
    calls = {"embeddings": 0, "pinecone": 0, "chat": 0}
    embedding_model = object()
    vector_store = object()
    chat_model = object()

    class Connection:
        pass

    connection = Connection()
    connection.vector_store = vector_store

    def create_embeddings():
        calls["embeddings"] += 1
        return embedding_model

    def connect(embeddings):
        assert embeddings is embedding_model
        calls["pinecone"] += 1
        return connection

    def create_chat():
        calls["chat"] += 1
        return chat_model

    monkeypatch.setattr(
        "app.core.dependencies.get_embedding_model",
        create_embeddings,
    )
    monkeypatch.setattr("app.core.dependencies.connect_vector_store", connect)
    monkeypatch.setattr("app.core.dependencies.create_chat_model", create_chat)

    first = get_rag_dependencies()
    second = get_rag_dependencies()

    assert first is second
    assert first.embedding_model is embedding_model
    assert first.vector_store is vector_store
    assert first.chat_model is chat_model
    assert calls == {"embeddings": 1, "pinecone": 1, "chat": 1}
    clear_rag_dependencies()
