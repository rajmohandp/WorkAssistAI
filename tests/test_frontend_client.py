"""Tests for the Streamlit-to-FastAPI HTTP boundary."""

import pytest
import requests

from app.frontend_client import (
    DocuVerseAPIError,
    ask_question_http,
    get_repository_status,
    synchronize_repository,
)


class FakeResponse:
    def __init__(self, payload, *, ok=True):
        self.payload = payload
        self.ok = ok

    def json(self):
        return self.payload


def test_ask_question_http_posts_question_and_returns_sources(monkeypatch):
    captured = {}
    payload = {
        "question": "What is the policy?",
        "answer": "The policy is documented.",
        "sources": [{"document": "policy.pdf", "page": 4}],
    }

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse(payload)

    monkeypatch.setattr("app.frontend_client.requests.post", fake_post)

    assert ask_question_http("What is the policy?", timeout=12) == payload
    assert captured == {
        "url": "http://localhost:8000/ask",
        "json": {"question": "What is the policy?"},
        "timeout": 12,
    }


def test_ask_question_http_handles_connection_error(monkeypatch):
    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("app.frontend_client.requests.post", fail)

    with pytest.raises(DocuVerseAPIError, match="Confirm FastAPI is running"):
        ask_question_http("What is the policy?")


def test_ask_question_http_sends_selected_document(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse({"answer": "Answer", "sources": []})

    monkeypatch.setattr("app.frontend_client.requests.post", fake_post)

    ask_question_http(
        "What is the policy?",
        document="employee_handbook.pdf",
    )

    assert captured["json"] == {
        "question": "What is the policy?",
        "document": "employee_handbook.pdf",
    }


def test_ask_question_http_uses_safe_api_error(monkeypatch):
    monkeypatch.setattr(
        "app.frontend_client.requests.post",
        lambda *_args, **_kwargs: FakeResponse(
            {"detail": "DocuVerse could not complete the question."},
            ok=False,
        ),
    )

    with pytest.raises(DocuVerseAPIError, match="could not complete"):
        ask_question_http("What is the policy?")


def test_ask_question_http_rejects_invalid_payload(monkeypatch):
    monkeypatch.setattr(
        "app.frontend_client.requests.post",
        lambda *_args, **_kwargs: FakeResponse({"sources": []}),
    )

    with pytest.raises(DocuVerseAPIError, match="did not contain an answer"):
        ask_question_http("What is the policy?")


def test_repository_status_returns_document_names(monkeypatch):
    payload = {
        "s3": {
            "connected": True,
            "bucket": "documents-bucket",
            "supported_documents": 2,
            "documents": ["handbook.pdf", "policy.docx"],
        },
        "pinecone": {
            "connected": True,
            "index_name": "docuverse",
            "indexed_documents": 2,
            "total_vectors": 20,
        },
    }
    monkeypatch.setattr(
        "app.frontend_client.requests.get",
        lambda *_args, **_kwargs: FakeResponse(payload),
    )

    assert get_repository_status() == payload


def test_synchronize_repository_calls_fastapi(monkeypatch):
    captured = {}
    payload = {
        "documents_scanned": 3,
        "documents_added": 1,
        "documents_updated": 0,
        "documents_removed": 0,
        "unchanged_documents": 2,
        "chunks_added": 10,
        "chunks_removed": 0,
        "vectors_removed_from_pinecone": 0,
        "failed_documents": 0,
    }

    def fake_post(url, timeout):
        captured.update(url=url, timeout=timeout)
        return FakeResponse(payload)

    monkeypatch.setattr("app.frontend_client.requests.post", fake_post)

    assert synchronize_repository(timeout=120) == payload
    assert captured == {
        "url": "http://localhost:8000/api/v1/repository/sync",
        "timeout": 120,
    }
