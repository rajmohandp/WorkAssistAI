"""Deterministic request classification for the DocuVerse graph."""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.agent.state import AgentState, Intent
from src.rag_chain import RAGResult

logger = logging.getLogger(__name__)

_DOCUMENT_CUES = re.compile(
    r"\b(handbook|policy|policies|pdf|document|uploaded|summari[sz]e|"
    r"what does .+ say)\b",
    re.IGNORECASE,
)
_PTO_POLICY_CUES = re.compile(
    r"\b(allow(?:ed|ance)?|eligib(?:le|ility)|entitl(?:ed|ement)|"
    r"full[ -]time|part[ -]time|accrual rate|accru(?:e|es|ing)|"
    r"annual(?:ly)?|per (?:calendar )?year|years? of service|tenure|"
    r"how (?:does|is) pto accrue)\b",
    re.IGNORECASE,
)
_PTO_POLICY_USE_CUES = re.compile(
    r"\b(?:can|may)\s+i\s+(?:use|take)\s+"
    r"(?:sick|vacation|personal|pto|paid time off)"
    r"(?:\s+leave)?\s+(?:for|when|if)\b",
    re.IGNORECASE,
)
_HUMAN_ESCALATION_CUES = re.compile(
    r"\b(?:"
    r"(?:pto|vacation|sick|leave)\s+balance\s+(?:is\s+)?(?:incorrect|wrong)|"
    r"dispute\s+(?:my\s+)?(?:pto|vacation|sick|leave)\s+balance|"
    r"(?:someone|human|hr|agent|representative)\s+(?:can\s+)?(?:review|help)|"
    r"(?:review|correct|investigate)\s+(?:my\s+)?(?:pto|vacation|sick|leave)\s+balance|"
    r"(?:speak|talk)\s+(?:to|with)\s+(?:a\s+)?(?:human|hr|agent|representative)"
    r")\b",
    re.IGNORECASE,
)
_PTO_DEDUCTION_CUES = re.compile(
    r"\b(?:deducted|charged|removed|subtracted)\b",
    re.IGNORECASE,
)
_PTO_DISPUTE_CUES = re.compile(
    r"\b(?:incorrect(?:ly)?|wrong(?:ly)?|too many|only|but|"
    r"correct(?:ed|ion)?|fix|dispute|should have)\b",
    re.IGNORECASE,
)
_PTO_BALANCE_CUES = re.compile(
    r"\b(pto balance|current pto|vacation balance|sick balance|sick hours|"
    r"personal leave|available leave|accrued pto|used pto|pending pto|"
    r"remaining time off|floating[ -]holiday|my pto|pto hours|pto remaining|"
    r"how (?:much|many) pto)\b",
    re.IGNORECASE,
)
_PTO_DOMAIN_CUES = re.compile(
    r"\b(pto|vacation|sick leave|personal leave|leave|time off|"
    r"floating[ -]holiday)\b",
    re.IGNORECASE,
)
_PTO_USAGE_CUES = re.compile(
    r"\b(can i|take|use|request|enough|afford|cover|hours?|days?|"
    r"next week|tomorrow)\b",
    re.IGNORECASE,
)
_PTO_CATEGORY_ALIASES = {
    "vacation": "vacation",
    "sick": "sick",
    "personal": "personal",
    "personal leave": "personal",
    "floating holiday": "floating holiday",
    "floating-holiday": "floating holiday",
    "other": "other",
}


def _message_fields(message: object) -> tuple[str | None, str | None]:
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        return (
            role if isinstance(role, str) else None,
            content if isinstance(content, str) else None,
        )
    return None, None


def standalone_pto_category(question: str) -> str | None:
    """Return a category only when the entire reply is a supported PTO label."""

    normalized = " ".join(question.strip().casefold().split())
    return _PTO_CATEGORY_ALIASES.get(normalized)


def get_pto_clarification_question(state: AgentState) -> str | None:
    """Recover the prior PTO request only after DocuVerse asked for a category."""

    if standalone_pto_category(state["question"]) is None:
        return None
    history = state["conversation_history"] or []
    if not history:
        return None
    last_role, last_content = _message_fields(history[-1])
    if (
        last_role != "assistant"
        or not last_content
        or "multiple pto categories" not in last_content.casefold()
        or "please specify which category" not in last_content.casefold()
    ):
        return None
    for message in reversed(history[:-1]):
        role, content = _message_fields(message)
        if (
            role == "user"
            and content
            and _PTO_DOMAIN_CUES.search(content)
            and _PTO_USAGE_CUES.search(content)
        ):
            return content
    return None


async def classify_request(state: AgentState) -> dict[str, object]:
    """Choose a typed route, giving policy/document language precedence."""

    try:
        question = state["question"]
        clarification_question = get_pto_clarification_question(state)
        if _HUMAN_ESCALATION_CUES.search(question) or (
            _PTO_DOMAIN_CUES.search(question)
            and _PTO_DEDUCTION_CUES.search(question)
            and _PTO_DISPUTE_CUES.search(question)
        ):
            intent: Intent = "human_escalation"
            data_source = None
            tool_name = "human_handoff"
        elif clarification_question:
            intent: Intent = "pto_request_feasibility"
            data_source = "aws_mysql"
            tool_name = "get_current_pto_balance"
        elif (
            _DOCUMENT_CUES.search(question)
            or _PTO_POLICY_CUES.search(question)
            or _PTO_POLICY_USE_CUES.search(question)
        ):
            intent: Intent = "document_question"
            data_source = "pinecone"
            tool_name = "existing_rag"
        elif _PTO_BALANCE_CUES.search(question):
            intent = "pto_balance"
            data_source = "aws_mysql"
            tool_name = "get_current_pto_balance"
        elif _PTO_DOMAIN_CUES.search(question) and _PTO_USAGE_CUES.search(question):
            intent = "pto_request_feasibility"
            data_source = "aws_mysql"
            tool_name = "get_current_pto_balance"
        else:
            intent = "document_question"
            data_source = "pinecone"
            tool_name = "existing_rag"
    except Exception:
        logger.exception(
            "Request classification failed",
            extra={
                "operation": "orchestration",
                "event": "request_classification_failed",
            },
        )
        return {"intent": "unknown", "error": "classification_failed"}

    logger.info(
        "Request route selected",
        extra={
            "operation": "orchestration",
            "event": "request_route_selected",
            "intent": intent,
            "data_source": data_source,
            "tool_name": tool_name,
        },
    )
    return {
        "intent": intent,
        "data_source": data_source,
        "tool_name": tool_name,
        "escalation_reason": (
            "user_requested_human" if intent == "human_escalation" else None
        ),
    }


def route_classified_request(
    state: AgentState,
) -> Literal[
    "resolve_employee", "retrieve_documents", "escalate_to_human", "handle_error"
]:
    if state["intent"] == "human_escalation":
        return "escalate_to_human"
    if state["intent"] in {"pto_balance", "pto_request_feasibility"}:
        return "resolve_employee"
    if state["intent"] == "document_question":
        return "retrieve_documents"
    return "handle_error"


def route_after_employee_resolution(
    state: AgentState,
) -> Literal["query_pto_database", "handle_error"]:
    return "handle_error" if state["error"] else "query_pto_database"


def route_after_pto_query(
    state: AgentState,
) -> Literal["generate_response", "handle_error"]:
    return "handle_error" if state["error"] else "generate_response"


def route_after_retrieval(
    state: AgentState,
) -> Literal["generate_response", "escalate_to_human", "handle_error"]:
    if state["error"]:
        return "handle_error"
    result = state["tool_result"]
    if (
        isinstance(result, RAGResult)
        and result.resolution_status == "insufficient_context"
    ):
        return "escalate_to_human"
    return "generate_response"
