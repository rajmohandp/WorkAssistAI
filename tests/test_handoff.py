"""Policy routing and authenticated human-handoff regression tests."""

import asyncio
import os

os.environ.setdefault("AUTH_TOKEN_SECRET", "test-only-authentication-secret-value")

from fastapi.testclient import TestClient

from app.agent.graph import docuverse_graph
from app.agent.state import create_agent_state
from app.main import app
from app.services.handoff_service import handoff_service
from src.guardrails import SECURITY_REFUSAL_MESSAGE
from src.rag_chain import INSUFFICIENT_CONTEXT_MESSAGE, RAGResult


def _rag_result(status: str) -> RAGResult:
    answer = (
        INSUFFICIENT_CONTEXT_MESSAGE
        if status == "insufficient_context"
        else "Grounded handbook answer [handbook.pdf, Page 4]"
    )
    return RAGResult(
        answer=answer,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="sick leave planned vacation policy",
        resolution_status=status,
    )


def _invoke(monkeypatch, result: RAGResult):
    monkeypatch.setattr(
        "app.services.rag_service.ask_question", lambda *_a, **_k: result
    )
    state = create_agent_state(
        "Can I use sick leave for a planned vacation?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )
    return asyncio.run(docuverse_graph.ainvoke(state))


def test_policy_use_question_routes_to_handbook_without_escalation(monkeypatch):
    handoff_service.clear()

    result = _invoke(monkeypatch, _rag_result("grounded"))

    assert result["intent"] == "document_question"
    assert result["data_source"] == "pinecone"
    assert result["escalation_required"] is False
    assert handoff_service.list_handoffs() == []


def test_insufficient_policy_answer_creates_one_idempotent_handoff(monkeypatch):
    handoff_service.clear()
    rag_result = _rag_result("insufficient_context")

    first = _invoke(monkeypatch, rag_result)
    second = _invoke(monkeypatch, rag_result)

    assert first["escalation_required"] is True
    assert first["handoff_status"] == "queued"
    assert first["handoff_id"] == second["handoff_id"]
    assert first["handoff_id"] in first["final_answer"]
    assert len(handoff_service.list_handoffs()) == 1


def test_incorrect_pto_balance_request_escalates_without_data_lookup(monkeypatch):
    class ForbiddenPTOTool:
        async def ainvoke(self, _arguments):
            raise AssertionError("PTO database must not run for a disputed balance")

    handoff_service.clear()
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", ForbiddenPTOTool())
    monkeypatch.setattr(
        "app.services.rag_service.ask_question",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("Document retrieval must not run for direct escalation")
        ),
    )
    state = create_agent_state(
        "My PTO balance is incorrect. Can someone review it?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )

    result = asyncio.run(docuverse_graph.ainvoke(state))

    assert result["intent"] == "human_escalation"
    assert result["data_source"] is None
    assert result["tool_name"] == "human_handoff"
    assert result["escalation_reason"] == "user_requested_human"
    assert result["escalation_required"] is True
    assert "PTO balance concern" in result["final_answer"]
    assert len(handoff_service.list_handoffs()) == 1


def test_pto_deduction_discrepancy_escalates_without_balance_calculation(monkeypatch):
    class ForbiddenPTOTool:
        async def ainvoke(self, _arguments):
            raise AssertionError("PTO database must not calculate a disputed deduction")

    handoff_service.clear()
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", ForbiddenPTOTool())
    monkeypatch.setattr(
        "app.services.rag_service.ask_question",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("Document retrieval must not run for a disputed deduction")
        ),
    )
    state = create_agent_state(
        "I used only 8 hours of Vacation, but 16 hours were deducted. "
        "Can you correct it?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )

    result = asyncio.run(docuverse_graph.ainvoke(state))

    assert result["intent"] == "human_escalation"
    assert result["escalation_reason"] == "user_requested_human"
    assert result["escalation_required"] is True
    assert result["data_source"] is None
    assert "PTO balance concern" in result["final_answer"]


def test_security_refusal_does_not_escalate(monkeypatch):
    handoff_service.clear()
    refusal = RAGResult(
        answer=SECURITY_REFUSAL_MESSAGE,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="",
        resolution_status="security_refusal",
    )

    result = _invoke(monkeypatch, refusal)

    assert result["escalation_required"] is False
    assert handoff_service.list_handoffs() == []


def test_handoff_queue_is_admin_only(monkeypatch):
    handoff_service.clear()
    _invoke(monkeypatch, _rag_result("insufficient_context"))
    client = TestClient(app)

    user_login = client.post(
        "/api/v1/auth/login",
        json={"username": "user01", "password": "welcome123"},
    ).json()
    denied = client.get(
        "/api/v1/handoffs",
        headers={"Authorization": f"Bearer {user_login['access_token']}"},
    )

    admin_login = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "welcome123"},
    ).json()
    allowed = client.get(
        "/api/v1/handoffs",
        headers={"Authorization": f"Bearer {admin_login['access_token']}"},
    )

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()[0]["employee_id"] == "EMP001"
