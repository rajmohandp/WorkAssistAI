"""Typed state shared by the DocuVerse request graph."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.rag_chain import ConversationHistory, RAGResult
from src.retriever import MetadataFilter

UserRole = Literal["admin", "user"]
Intent = Literal[
    "pto_balance",
    "pto_request_feasibility",
    "pto_and_policy",
    "document_question",
    "human_escalation",
    "unknown",
]
DataSource = Literal["aws_mysql", "pinecone", "aws_mysql_and_pinecone"]
EscalationReason = Literal[
    "insufficient_document_context",
    "ambiguous_policy",
    "user_requested_human",
]


class AgentState(TypedDict):
    """Complete state carried through one graph execution."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str | None
    username: str | None
    user_role: UserRole | None
    question: str
    intent: Intent | None
    data_source: DataSource | None
    tool_name: str | None
    tool_result: Any | None
    balance_question: str | None
    policy_question: str | None
    pto_result: dict[str, Any] | None
    policy_result: RAGResult | None
    partial_answer: bool
    retrieved_documents: list[Document]
    final_answer: str | None
    error: str | None
    resolved_employee_id: str | None
    resolved_user_name: str | None
    requested_pto_type: str | None
    requested_balance_year: int | None
    requested_hours: Decimal | None
    requested_period: str | None
    metadata_filter: MetadataFilter | None
    conversation_history: ConversationHistory | None
    escalation_required: bool
    escalation_reason: EscalationReason | None
    handoff_id: str | None
    handoff_status: str | None


def create_agent_state(
    question: str,
    *,
    user_id: str | None = None,
    username: str | None = None,
    user_role: UserRole | None = None,
    messages: list[AnyMessage] | None = None,
    metadata_filter: MetadataFilter | None = None,
    conversation_history: ConversationHistory | None = None,
) -> AgentState:
    """Create an initialized state without provider or framework dependencies."""

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty.")
    return AgentState(
        messages=list(messages or []),
        user_id=user_id,
        username=username,
        user_role=user_role,
        question=normalized_question,
        intent=None,
        data_source=None,
        tool_name=None,
        tool_result=None,
        balance_question=None,
        policy_question=None,
        pto_result=None,
        policy_result=None,
        partial_answer=False,
        retrieved_documents=[],
        final_answer=None,
        error=None,
        resolved_employee_id=None,
        resolved_user_name=None,
        requested_pto_type=None,
        requested_balance_year=None,
        requested_hours=None,
        requested_period=None,
        metadata_filter=metadata_filter,
        conversation_history=conversation_history,
        escalation_required=False,
        escalation_reason=None,
        handoff_id=None,
        handoff_status=None,
    )
