"""Tests for the modular DocuVerse FastAPI backend."""

import os

os.environ.setdefault("AUTH_TOKEN_SECRET", "test-only-authentication-secret-value")

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.schemas import (
    AskRequest,
    AskResponse,
    PineconeStatusResponse,
    QueryResponse,
    S3StatusResponse,
    SourceResponse,
    SyncResponse,
)
from app.core.dependencies import RAGDependencies
from app.main import app
from app.services.rag_service import ask_question
from src.rag_chain import INSUFFICIENT_CONTEXT_MESSAGE, RAGResult

client = TestClient(app)
login_response = client.post(
    "/api/v1/auth/login",
    json={"username": "admin", "password": "welcome123"},
)
client.headers["Authorization"] = (
    f"Bearer {login_response.json()['access_token']}"
)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "workassist-api",
    }


def test_health_endpoint_does_not_call_external_services(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The health endpoint called an external service")

    monkeypatch.setattr("app.api.routes.s3_service.status", fail_if_called)
    monkeypatch.setattr("app.api.routes.pinecone_service.status", fail_if_called)
    monkeypatch.setattr("app.api.routes.rag_service.answer", fail_if_called)
    monkeypatch.setattr(
        "app.core.database.check_database_health",
        fail_if_called,
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "workassist-api",
    }


def test_versioned_health_endpoint_remains_available():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_endpoint_checks_database(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.api.routes.check_database_health", lambda: calls.append("checked") or True
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "workassist-api",
        "database": "reachable",
    }
    assert calls == ["checked"]


def test_readiness_endpoint_returns_safe_503(monkeypatch):
    from app.core.database import DatabaseConnectionError

    def unavailable():
        raise DatabaseConnectionError("sensitive database provider detail")

    monkeypatch.setattr("app.api.routes.check_database_health", unavailable)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "The service is not ready."}
    assert "sensitive" not in response.text


def test_repository_status_includes_available_document_names(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.s3_service.status",
        lambda: S3StatusResponse(
            connected=True,
            bucket="documents-bucket",
            supported_documents=3,
            documents=["employee_handbook.pdf", "policy.docx", "readme.txt"],
        ),
    )
    monkeypatch.setattr(
        "app.api.routes.pinecone_service.status",
        lambda: PineconeStatusResponse(
            connected=True,
            index_name="docuverse",
            indexed_documents=3,
            total_vectors=1016,
        ),
    )

    response = client.get("/api/v1/repository/status")

    assert response.status_code == 200
    assert response.json()["s3"]["documents"] == [
        "employee_handbook.pdf",
        "policy.docx",
        "readme.txt",
    ]


def test_repository_sync_endpoint_uses_backend_service(monkeypatch):
    expected = SyncResponse(
        documents_scanned=3,
        new_documents=1,
        updated_documents=1,
        unchanged_documents=1,
        failed_documents=0,
        chunks_added=20,
        chunks_removed=5,
        documents_removed_from_s3=0,
        vectors_removed_from_pinecone=0,
        documents_added=1,
        documents_updated=1,
        documents_removed=0,
    )
    monkeypatch.setattr(
        "app.api.routes.sync_service.synchronize",
        lambda: expected,
    )

    response = client.post("/api/v1/repository/sync")

    assert response.status_code == 200
    assert response.json() == expected.model_dump()


def test_repository_sync_sanitizes_unexpected_provider_failure(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.sync_service.synchronize",
        lambda: (_ for _ in ()).throw(PermissionError("socket access denied")),
    )

    response = client.post("/api/v1/repository/sync")

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Document synchronization could not connect to a required "
            "repository provider. Check the API logs and network access."
        )
    }
    assert "socket access denied" not in response.text


def test_query_endpoint_uses_rag_service(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.rag_service.answer",
        lambda request: QueryResponse(
            question=request.question,
            answer="Grounded answer [policy.pdf, Page 2]",
            retrieval_query="standalone policy question",
            sources=[
                SourceResponse(
                    document_name="policy.pdf",
                    page_number=2,
                    citation="[policy.pdf, Page 2]",
                    excerpt="Relevant policy excerpt.",
                )
            ],
        ),
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "Who is eligible for it?",
            "history": [
                {"role": "user", "content": "What is the policy?"},
                {"role": "assistant", "content": "It covers training."},
            ],
            "filters": {"file_type": "pdf"},
        },
    )

    assert response.status_code == 200
    assert response.json()["question"] == "Who is eligible for it?"
    assert response.json()["retrieval_query"] == "standalone policy question"
    assert response.json()["sources"][0]["document_name"] == "policy.pdf"


def test_rejects_invalid_query_payload():
    response = client.post("/api/v1/query", json={"question": ""})

    assert response.status_code == 422


def test_ask_endpoint_calls_reusable_rag_service(monkeypatch):
    result = RAGResult(
        answer="Employees receive paid vacation [employee_handbook.pdf, Page 12]",
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[
            SourceResponse(
                document_name="employee_handbook.pdf",
                page_number=12,
                citation="[employee_handbook.pdf, Page 12]",
                excerpt="Vacation policy excerpt.",
            )
        ],
        retrieval_query="employee vacation policies",
    )
    captured = {}

    def fake_ask_question(question):
        captured["question"] = question
        return result

    monkeypatch.setattr(
        "app.services.rag_service.ask_question",
        fake_ask_question,
    )

    response = client.post(
        "/ask",
        json={"question": "  What are the employee vacation policies?  "},
    )

    assert response.status_code == 200
    assert captured["question"] == "What are the employee vacation policies?"
    assert response.json() == {
        "question": "What are the employee vacation policies?",
        "answer": (
            "Employees receive paid vacation "
            "[employee_handbook.pdf, Page 12]"
        ),
        "sources": [{"document": "employee_handbook.pdf", "page": 12}],
        "resolution": "answered",
    }


def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_endpoint_handles_greeting_without_rag(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("RAG must not run for a greeting")

    monkeypatch.setattr(
        "app.services.rag_service.ask_question",
        fail_if_called,
    )

    response = client.post("/ask", json={"question": "  Hello, WorkAssist AI!  "})

    assert response.status_code == 200
    assert response.json()["answer"].startswith("Hello! 👋")
    assert response.json()["sources"] == []


def test_ask_endpoint_sends_greeting_plus_question_to_rag(monkeypatch):
    result = RAGResult(
        answer=INSUFFICIENT_CONTEXT_MESSAGE,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="vacation policy",
    )
    captured = {}

    def fake_ask_question(question, **_kwargs):
        captured["question"] = question
        return result

    monkeypatch.setattr("app.services.rag_service.ask_question", fake_ask_question)

    response = client.post(
        "/ask",
        json={"question": "Hi, what is our vacation policy?"},
    )

    assert response.status_code == 200
    assert captured["question"] == "Hi, what is our vacation policy?"


def test_ask_endpoint_passes_selected_document_filter(monkeypatch):
    captured = {}
    result = RAGResult(
        answer=INSUFFICIENT_CONTEXT_MESSAGE,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="vacation policy",
    )

    def fake_ask_question(question, **kwargs):
        captured.update(question=question, **kwargs)
        return result

    monkeypatch.setattr("app.services.rag_service.ask_question", fake_ask_question)

    response = client.post(
        "/ask",
        json={
            "question": "What is the vacation policy?",
            "document": "employee_handbook.pdf",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "question": "What is the vacation policy?",
        "metadata_filter": {"filename": "employee_handbook.pdf"},
    }


def test_ask_endpoint_returns_sanitized_service_error(monkeypatch):
    def fail(_question):
        raise ValueError("sensitive technical detail")

    monkeypatch.setattr("app.services.rag_service.ask_question", fail)

    response = client.post("/ask", json={"question": "What is the policy?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "WorkAssist AI could not complete the question."
    }
    assert "sensitive technical detail" not in response.text


def test_ask_request_trims_question_and_rejects_whitespace():
    request = AskRequest(question="  What is the retention policy?  ")

    assert request.question == "What is the retention policy?"
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_ask_response_defaults_to_empty_sources():
    response = AskResponse(question="Question", answer="Answer")

    assert response.model_dump() == {
        "question": "Question",
        "answer": "Answer",
        "sources": [],
        "evaluation": None,
        "resolution": "answered",
        "handoff": None,
    }


def test_reusable_rag_service_runs_framework_neutral_pipeline(monkeypatch):
    expected = RAGResult(
        answer=INSUFFICIENT_CONTEXT_MESSAGE,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="retention policy",
    )
    captured = {}
    dependencies = RAGDependencies(
        embedding_model=object(),
        vector_store=object(),
        chat_model=object(),
    )

    def fake_run(
        question,
        top_k,
        vector_store,
        chat_model,
        metadata_filter,
        conversation_history,
    ):
        captured.update(
            question=question,
            top_k=top_k,
            vector_store=vector_store,
            chat_model=chat_model,
            metadata_filter=metadata_filter,
            conversation_history=conversation_history,
        )
        return expected

    monkeypatch.setattr("app.services.rag_service.run_rag", fake_run)

    result = ask_question(
        "What is the retention policy?",
        top_k=3,
        metadata_filter={"file_type": "pdf"},
        conversation_history=[{"role": "user", "content": "Earlier question"}],
        dependencies=dependencies,
    )

    assert result.answer == expected.answer
    assert result.retrieval_query == expected.retrieval_query
    assert result.retrieved_documents == expected.retrieved_documents
    assert captured == {
        "question": "What is the retention policy?",
        "top_k": 3,
        "vector_store": dependencies.vector_store,
        "chat_model": dependencies.chat_model,
        "metadata_filter": {"file_type": "pdf"},
        "conversation_history": [
            {"role": "user", "content": "Earlier question"}
        ],
    }


def test_rag_service_attaches_evaluation_when_enabled(monkeypatch):
    expected = RAGResult(
        answer="The policy allows ten days.",
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="vacation policy",
    )
    dependencies = RAGDependencies(
        embedding_model=object(),
        vector_store=object(),
        chat_model=object(),
    )

    class Settings:
        rag_evaluation_enabled = True

    class Evaluation:
        @staticmethod
        def to_dict():
            return {
                "faithfulness_percentage": 90.0,
                "answer_relevance_percentage": 85.0,
                "retrieval_confidence_percentage": 80.0,
                "overall_confidence_percentage": 87.5,
                "status": "evaluated",
            }

    monkeypatch.setattr(
        "app.services.rag_service.run_rag",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_environment_settings",
        lambda: Settings(),
    )
    monkeypatch.setattr(
        "app.services.rag_service.evaluate_rag_result",
        lambda *_args, **_kwargs: Evaluation(),
    )

    result = ask_question("What is the policy?", dependencies=dependencies)

    assert result.answer == expected.answer
    assert result.evaluation == Evaluation.to_dict()


def test_ask_endpoint_returns_optional_evaluation(monkeypatch):
    evaluation = {
        "faithfulness_percentage": 90.0,
        "answer_relevance_percentage": 85.0,
        "retrieval_confidence_percentage": 80.0,
        "overall_confidence_percentage": 87.5,
        "status": "evaluated",
    }
    result = RAGResult(
        answer="The policy allows ten days.",
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="vacation policy",
        evaluation=evaluation,
    )
    monkeypatch.setattr(
        "app.services.rag_service.ask_question",
        lambda *_args, **_kwargs: result,
    )

    response = client.post("/ask", json={"question": "What is the policy?"})

    assert response.status_code == 200
    assert response.json()["evaluation"] == evaluation


def test_rag_service_logs_lifecycle_counts_without_question(monkeypatch):
    expected = RAGResult(
        answer=INSUFFICIENT_CONTEXT_MESSAGE,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="safe query",
    )
    dependencies = RAGDependencies(
        embedding_model=object(),
        vector_store=object(),
        chat_model=object(),
    )
    events = []

    monkeypatch.setattr(
        "app.services.rag_service.run_rag",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        "app.services.rag_service.logger.info",
        lambda message, *, extra: events.append((message, extra)),
    )

    secret_question = "What is the policy? OPENAI_API_KEY=do-not-log-this"
    ask_question(secret_question, dependencies=dependencies)

    assert [extra["event"] for _, extra in events] == [
        "rag_request_started",
        "rag_request_completed",
    ]
    assert events[1][1]["chunks_retrieved"] == 0
    assert "do-not-log-this" not in repr(events)
