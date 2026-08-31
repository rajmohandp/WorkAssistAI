"""Routing and authorization tests for the DocuVerse LangGraph."""

from __future__ import annotations

import asyncio

from app.agent.graph import build_graph, docuverse_graph
from app.agent.state import create_agent_state
from src.rag_chain import RAGResult


def pto_result(pto_type: str = "VACATION") -> dict[str, object]:
    return {
        "employee_id": "EMP001",
        "employee_name": "Example Employee",
        "balance_year": 2026,
        "balances": [
            {
                "pto_type": pto_type,
                "opening_balance": "40.00",
                "accrued_hours": "48.00",
                "used_hours": "24.00",
                "pending_hours": "8.00",
                "adjusted_hours": "0.00",
                "available_hours": "56.00",
            }
        ],
        "last_updated": "2026-02-03T04:05:06Z",
        "source": "aws_mysql",
        "found": True,
        "error": None,
    }


def rag_result() -> RAGResult:
    return RAGResult(
        answer="Grounded document answer [handbook.pdf, Page 2]",
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="document question",
    )


class FakePTOTool:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append(arguments)
        return self.result


def invoke(question: str, **identity: object) -> dict[str, object]:
    state = create_agent_state(question, **identity)
    return asyncio.run(docuverse_graph.ainvoke(state))


def fail_rag(*_args, **_kwargs):
    raise AssertionError("Pinecone RAG must not run for this request")


def test_current_pto_calls_database_and_not_pinecone(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "What is my current PTO balance?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )

    assert len(tool.calls) == 1
    assert tool.calls[0]["employee_id"] == "EMP001"
    assert tool.calls[0]["user_name"] is None
    assert result["data_source"] == "aws_mysql"
    assert "balance year 2026" in result["final_answer"]
    assert "Last updated: 2026-02-03T04:05:06+00:00" in result["final_answer"]


def test_sick_hours_calls_database(monkeypatch):
    tool = FakePTOTool(pto_result("SICK"))
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "How many sick hours do I have?",
        user_id="EMP001",
        user_role="user",
    )

    assert tool.calls[0]["pto_type"] == "sick"
    assert result["intent"] == "pto_balance"


def test_requested_pto_hours_routes_to_database_and_checks_sufficiency(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "Can I take 24 hours of PTO next week?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )

    assert len(tool.calls) == 1
    assert result["intent"] == "pto_request_feasibility"
    assert result["data_source"] == "aws_mysql"
    assert str(result["requested_hours"]) == "24"
    assert result["requested_period"] == "next week"
    assert "enough to cover 24 hours next week" in result["final_answer"]
    assert "Balance year: 2026" in result["final_answer"]
    assert "Last updated: 2026-02-03T04:05:06+00:00" in result["final_answer"]
    assert "does not approve the dates" in result["final_answer"]


def test_requested_vacation_hours_include_projected_remaining_balance(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "How much Vacation will remain if I take 16 hours next week?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )

    assert result["intent"] == "pto_request_feasibility"
    assert "56.00 hours available" in result["final_answer"]
    assert "projected remaining balance" in result["final_answer"]
    assert "40.00 hours" in result["final_answer"]


def test_requested_pto_hours_requires_category_when_multiple_exist(monkeypatch):
    result_with_categories = pto_result()
    second_balance = dict(result_with_categories["balances"][0])
    second_balance["pto_type"] = "SICK"
    result_with_categories["balances"] = [
        *result_with_categories["balances"],
        second_balance,
    ]
    tool = FakePTOTool(result_with_categories)
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "Can I take 24 hours of PTO next week?",
        user_id="EMP001",
        user_role="user",
    )

    assert "multiple PTO categories" in result["final_answer"]
    assert "Please specify which category" in result["final_answer"]


def test_standalone_category_continues_pending_pto_request(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)
    history = [
        {"role": "user", "content": "Can I take 24 hours of PTO next week?"},
        {
            "role": "assistant",
            "content": (
                "You have multiple PTO categories for balance year 2026: "
                "Vacation, Sick. Please specify which category should cover "
                "the 24 hours."
            ),
        },
    ]
    state = create_agent_state(
        "Vacation",
        username="user01",
        user_role="user",
        conversation_history=history,
    )

    result = asyncio.run(docuverse_graph.ainvoke(state))

    assert result["intent"] == "pto_request_feasibility"
    assert result["data_source"] == "aws_mysql"
    assert result["requested_pto_type"] == "vacation"
    assert str(result["requested_hours"]) == "24"
    assert result["requested_period"] == "next week"
    assert tool.calls[0]["pto_type"] == "vacation"
    assert "enough to cover 24 hours next week" in result["final_answer"]


def test_standalone_vacation_without_clarification_uses_documents(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr(
        "app.services.rag_service.ask_question",
        lambda *_args, **_kwargs: rag_result(),
    )

    result = invoke("Vacation", username="user01", user_role="user")

    assert result["intent"] == "document_question"
    assert result["data_source"] == "pinecone"
    assert tool.calls == []


def test_natural_pto_question_uses_authenticated_username(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "How much PTO do I have now?",
        username="user01",
        user_role="user",
    )

    assert result["intent"] == "pto_balance"
    assert result["data_source"] == "aws_mysql"
    assert tool.calls[0]["user_name"] == "user01"
    assert tool.calls[0]["employee_id"] is None


def test_admin_can_query_explicit_employee(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "What is EMP001's vacation balance?",
        user_id="ADMIN001",
        username="admin",
        user_role="admin",
    )

    assert tool.calls[0]["employee_id"] == "EMP001"
    assert result["error"] is None


def test_user_cannot_query_another_employee(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "What is EMP001's vacation balance?",
        user_id="EMP002",
        username="user02",
        user_role="user",
    )

    assert tool.calls == []
    assert "not authorized" in result["final_answer"]


def test_handbook_pto_policy_uses_pinecone(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", lambda *_a, **_k: rag_result())

    result = invoke("What does the employee handbook say about PTO?")

    assert tool.calls == []
    assert result["data_source"] == "pinecone"
    assert result["final_answer"].startswith("Grounded document answer")


def test_full_time_pto_allowance_uses_pinecone(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr(
        "app.services.rag_service.ask_question", lambda *_a, **_k: rag_result()
    )

    result = invoke("How much PTO is allowed for full-time employees?")

    assert tool.calls == []
    assert result["intent"] == "document_question"
    assert result["data_source"] == "pinecone"
    assert result["final_answer"].startswith("Grounded document answer")


def test_tenure_based_annual_pto_accrual_uses_pinecone(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr(
        "app.services.rag_service.ask_question", lambda *_a, **_k: rag_result()
    )

    result = invoke(
        "I have completed seven years of service. How much PTO do I accrue annually?"
    )

    assert tool.calls == []
    assert result["intent"] == "document_question"
    assert result["data_source"] == "pinecone"
    assert result["final_answer"].startswith("Grounded document answer")


def test_uploaded_pdf_summary_uses_pinecone(monkeypatch):
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", lambda *_a, **_k: rag_result())

    result = invoke("Summarize the uploaded benefits PDF")

    assert tool.calls == []
    assert result["intent"] == "document_question"
    assert result["data_source"] == "pinecone"


def test_database_failure_returns_safe_error(monkeypatch):
    failed = pto_result()
    failed.update(found=False, balances=[], error="PTO balance data is unavailable.")
    tool = FakePTOTool(failed)
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "What is my current PTO balance?",
        user_id="EMP001",
        user_role="user",
    )

    assert result["final_answer"] == (
        "I couldn't retrieve PTO balance data right now. Please try again."
    )
    assert "database" not in result["final_answer"].casefold()


def test_database_exception_preserves_unavailable_error(monkeypatch):
    class FailingPTOTool:
        async def ainvoke(self, _arguments):
            raise ConnectionError("database connection failed")

    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", FailingPTOTool())
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)

    result = invoke(
        "How much PTO do I have?",
        user_id="EMP001",
        username="user01",
        user_role="user",
    )

    assert result["error"] == "pto_database_unavailable"
    assert result["final_answer"] == (
        "I couldn't retrieve PTO balance data right now. Please try again."
    )


def test_unknown_intent_executes_neither_data_source(monkeypatch):
    async def classify_unknown(_state):
        return {"intent": "unknown", "error": "classification_failed"}

    monkeypatch.setattr("app.agent.graph.classify_request", classify_unknown)
    monkeypatch.setattr("app.services.rag_service.ask_question", fail_rag)
    tool = FakePTOTool(pto_result())
    monkeypatch.setattr("app.agent.nodes.get_current_pto_balance", tool)
    graph = build_graph()

    result = asyncio.run(
        graph.ainvoke(create_agent_state("An intentionally unknown request"))
    )

    assert tool.calls == []
    assert result["data_source"] is None
    assert result["final_answer"] == (
        "I couldn't complete that request right now. Please try again."
    )
