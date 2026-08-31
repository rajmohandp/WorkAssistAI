"""Focused tests for the typed LangGraph state."""

import pytest

from app.agent.state import create_agent_state


def test_create_agent_state_initializes_request_fields():
    state = create_agent_state(
        "  What is the vacation policy?  ",
        user_id="employee-1",
        username="user01",
        user_role="user",
        metadata_filter={"filename": "handbook.pdf"},
    )

    assert state["question"] == "What is the vacation policy?"
    assert state["user_id"] == "employee-1"
    assert state["username"] == "user01"
    assert state["user_role"] == "user"
    assert state["metadata_filter"] == {"filename": "handbook.pdf"}
    assert state["messages"] == []
    assert state["retrieved_documents"] == []
    assert state["final_answer"] is None
    assert state["error"] is None
    assert state["resolved_employee_id"] is None
    assert state["requested_pto_type"] is None
    assert state["requested_balance_year"] is None
    assert state["requested_hours"] is None
    assert state["requested_period"] is None


def test_create_agent_state_rejects_empty_question():
    with pytest.raises(ValueError, match="Question must not be empty"):
        create_agent_state("   ")
