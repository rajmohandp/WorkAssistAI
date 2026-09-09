"""Combined requests preserve arithmetic, policy evidence, and partial results."""

import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import docuverse_graph
from app.agent.state import create_agent_state
from app.core.security import AuthenticatedUser, get_current_user
from app.main import app
from src.rag_chain import RAGResult, SourceCitation

QUESTION = (
    "Do I have enough vacation for 24 hours next week, and how much notice must I give?"
)


@pytest.fixture
def providers(monkeypatch):
    balance = {
        "found": True,
        "error": None,
        "balance_year": 2026,
        "last_updated": "2026-08-29T21:34:02",
        "balances": [
            {
                "pto_type": "VACATION",
                "available_hours": "56.00",
                "accrued_hours": "48.00",
                "used_hours": "24.00",
                "pending_hours": "8.00",
            }
        ],
    }
    # Synthetic policy evidence, not a statement of the organization's policy.
    policy = RAGResult(
        answer="Give 10 business days' notice [handbook.pdf, Page 4].",
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[
            SourceCitation(
                document_name="handbook.pdf",
                page_number=4,
                citation="[handbook.pdf, Page 4]",
                excerpt="Vacation requests require 10 business days' notice.",
            )
        ],
        retrieval_query="vacation notice",
    )
    values = {"balance": balance, "policy": policy, "pto_calls": [], "rag_calls": []}

    class Tool:
        async def ainvoke(self, arguments):
            values["pto_calls"].append(arguments)
            if isinstance(values["balance"], Exception):
                raise values["balance"]
            return values["balance"]

    def retrieve(question, **kwargs):
        values["rag_calls"].append((question, kwargs))
        if isinstance(values["policy"], Exception):
            raise values["policy"]
        return values["policy"]

    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", Tool())
    monkeypatch.setattr("app.services.rag_service.ask_question", retrieve)
    return values


def invoke(question=QUESTION, **kwargs):
    return asyncio.run(
        docuverse_graph.ainvoke(
            create_agent_state(
                question,
                user_id="EMP001",
                username="user01",
                user_role="user",
                **kwargs,
            )
        )
    )


@pytest.mark.parametrize(
    "question",
    [
        QUESTION,
        "How much notice must I give, and do I have enough vacation for 24 hours next week?",
        "Do I need 48 hours of notice, and can I take 24 hours of vacation next week?",
        "Can I take 24 hours of vacation next week? What does the notice policy say?",
    ],
)
def test_both_clauses_and_correct_hours(providers, question):
    result = invoke(question, metadata_filter={"filename": "handbook.pdf"})
    assert result["intent"] == "pto_and_policy"
    assert result["data_source"] == "aws_mysql_and_pinecone"
    assert str(result["requested_hours"]) == "24"
    assert "32.00 hours" in result["final_answer"]
    assert "10 business days" in result["final_answer"]
    assert "[handbook.pdf, Page 4]" in result["final_answer"]
    assert "does not approve the dates" in result["final_answer"]
    assert result["partial_answer"] is False
    assert len(providers["pto_calls"]) == len(providers["rag_calls"]) == 1
    query, kwargs = providers["rag_calls"][0]
    assert "vacation" in query
    assert "24 hours" not in query
    assert "EMP001" not in query
    assert kwargs == {"metadata_filter": {"filename": "handbook.pdf"}}


@pytest.mark.parametrize(
    "question",
    [
        "How much notice must I give for vacation?",
        "Can I take vacation and what notice is required?",
        "What does the handbook say about my PTO balance?",
        "Can I use sick leave for vacation, and what notice is required?",
    ],
)
def test_policy_only_avoids_database(providers, question):
    # A generic 'can I take' clause without a quantity is policy ambiguity.
    result = invoke(question)
    assert result["intent"] == "document_question"
    assert providers["pto_calls"] == []


@pytest.mark.parametrize("failure", ["exception", "missing", "refusal", "uncited"])
def test_policy_failure_preserves_balance(providers, failure):
    policy = providers["policy"]
    providers["policy"] = {
        "exception": TimeoutError("private provider diagnostic"),
        "missing": replace(policy, resolution_status="insufficient_context"),
        "refusal": replace(
            policy, resolution_status="security_refusal", answer="Refused."
        ),
        "uncited": replace(policy, sources_used=[]),
    }[failure]
    result = invoke()
    assert "32.00 hours" in result["final_answer"]
    assert "10 business days" not in result["final_answer"]
    assert "private provider" not in result["final_answer"]
    assert result["partial_answer"] is True


@pytest.mark.parametrize("failure", ["exception", "tool_error", "not_found"])
def test_database_failure_preserves_policy(providers, failure):
    if failure == "exception":
        providers["balance"] = ConnectionError("private database diagnostic")
    else:
        providers["balance"].update(
            found=False,
            balances=[],
            error="unavailable" if failure == "tool_error" else None,
        )
    result = invoke()
    assert "10 business days" in result["final_answer"]
    assert "32.00" not in result["final_answer"]
    assert "private database" not in result["final_answer"]
    assert result["partial_answer"] is True


def test_both_fail_safely(providers):
    providers["balance"] = ConnectionError("secret")
    providers["policy"] = TimeoutError("secret")
    result = invoke()
    assert "couldn't retrieve PTO" in result["final_answer"]
    assert "couldn't verify" in result["final_answer"]
    assert "secret" not in result["final_answer"]
    assert result["partial_answer"] is True


def test_unauthorized_identity_never_calls_providers(providers):
    result = invoke(QUESTION + " For EMP002.")
    assert "not authorized" in result["final_answer"]
    assert not providers["pto_calls"]
    assert not providers["rag_calls"]


def test_category_clarification_preserves_combined_request(providers):
    balance = providers["balance"]
    balance["balances"].append({**balance["balances"][0], "pto_type": "SICK"})
    question = QUESTION.replace("vacation", "PTO")
    first = invoke(question)
    assert "Please specify which category" in first["final_answer"]
    assert first["partial_answer"] is True
    balance["balances"].pop()
    second = invoke(
        "Vacation",
        conversation_history=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": first["final_answer"]},
        ],
    )
    assert second["intent"] == "pto_and_policy"
    assert "32.00 hours" in second["final_answer"]
    assert "10 business days" in second["final_answer"]
    assert "vacation" in providers["rag_calls"][-1][0]


def test_balance_listing_with_policy(providers):
    result = invoke("What is my vacation balance, and how much notice must I give?")
    assert "56.00 hours available" in result["final_answer"]
    assert "10 business days" in result["final_answer"]
    assert result["partial_answer"] is False


@pytest.mark.parametrize("policy_available", [True, False])
def test_api_preserves_sources_and_partial_status(providers, policy_available):
    if not policy_available:
        providers["policy"] = TimeoutError("private diagnostic")
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        "user01", "EMP001", "user"
    )
    try:
        response = TestClient(app).post("/api/v1/ask", json={"question": QUESTION})
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 200
    body = response.json()
    assert "32.00 hours" in body["answer"]
    assert body["resolution"] == ("answered" if policy_available else "partial")
    assert body["sources"] == (
        [{"document": "handbook.pdf", "page": 4}] if policy_available else []
    )
