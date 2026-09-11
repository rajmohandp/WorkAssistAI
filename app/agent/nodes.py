"""Nodes used by the DocuVerse request graph."""

from __future__ import annotations

import logging
import re
from asyncio import to_thread
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.agent.router import get_pto_clarification_question
from app.agent.state import AgentState
from app.repositories.pto_repository import get_employee_email
from app.services import rag_service as rag_service_module
from app.services.email_service import send_escalation_email
from app.services.handoff_service import handoff_service
from app.tools.pto_tool import get_current_pto_balance
from src.rag_chain import RAGResult

logger = logging.getLogger(__name__)

_EMPLOYEE_ID_PATTERN = re.compile(
    r"\bEMP[A-Z0-9_-]*\d[A-Z0-9_-]*\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_HOURS_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b",
    re.IGNORECASE,
)
_PERIOD_PATTERN = re.compile(
    r"\b(next week|this week|tomorrow|today)\b",
    re.IGNORECASE,
)
_SAFE_GENERIC_ERROR = "I couldn't complete that request right now. Please try again."


def _pto_type_from_question(question: str) -> str | None:
    normalized = question.casefold().replace("_", " ").replace("-", " ")
    if "floating holiday" in normalized:
        return "floating holiday"
    for pto_type in ("vacation", "sick", "personal", "other"):
        if pto_type in normalized:
            return pto_type
    return None


def _balance_year_from_question(question: str) -> int | None:
    match = _YEAR_PATTERN.search(question)
    return int(match.group(1)) if match else None


def _requested_hours_from_question(question: str) -> Decimal | None:
    match = _HOURS_PATTERN.search(question)
    if not match:
        return None
    try:
        requested = Decimal(match.group(1))
    except InvalidOperation:
        return None
    return requested if requested > 0 else None


def _requested_period_from_question(question: str) -> str | None:
    match = _PERIOD_PATTERN.search(question)
    return match.group(1).casefold() if match else None


async def resolve_employee(state: AgentState) -> dict[str, object]:
    """Resolve self-service identity and enforce explicit-ID authorization."""

    clarification_question = get_pto_clarification_question(state)
    effective_question = clarification_question or state["question"]
    balance_question = state["balance_question"] or effective_question
    explicit_match = _EMPLOYEE_ID_PATTERN.search(effective_question)
    explicit_employee_id = explicit_match.group(0).upper() if explicit_match else None
    authenticated_employee_id = (
        state["user_id"].strip().upper() if state["user_id"] else None
    )
    authenticated_username = (
        state["username"].strip().casefold() if state["username"] else None
    )

    if explicit_employee_id:
        if (
            state["user_role"] != "admin"
            and explicit_employee_id != authenticated_employee_id
        ):
            return {"error": "employee_access_denied"}
        resolved_employee_id = explicit_employee_id
    else:
        if not authenticated_username and not authenticated_employee_id:
            return {"error": "employee_identity_required"}
        resolved_employee_id = authenticated_employee_id

    return {
        "resolved_employee_id": resolved_employee_id,
        "resolved_user_name": (
            None if resolved_employee_id else authenticated_username
        ),
        "requested_pto_type": (
            _pto_type_from_question(state["question"])
            if clarification_question
            else _pto_type_from_question(balance_question)
        ),
        "requested_balance_year": _balance_year_from_question(balance_question),
        "requested_hours": _requested_hours_from_question(balance_question),
        "requested_period": _requested_period_from_question(balance_question),
        "error": None,
    }


async def query_pto_database(state: AgentState) -> dict[str, object]:
    """Invoke only the bounded PTO tool for an authorized employee."""

    employee_id = state["resolved_employee_id"]
    user_name = state["resolved_user_name"]
    if not employee_id and not user_name:
        return {"error": "employee_identity_required"}
    try:
        result = await get_current_pto_balance.ainvoke(
            {
                "employee_id": employee_id,
                "user_name": user_name,
                "pto_type": state["requested_pto_type"],
                "balance_year": state["requested_balance_year"],
            }
        )
    # Tool/provider implementations can raise third-party exception types. This
    # graph boundary intentionally contains and sanitizes all of them.
    except Exception:  # noqa: BLE001
        return {"error": "pto_database_unavailable"}
    return {
        "tool_result": result,
        "data_source": "aws_mysql",
        "tool_name": "get_current_pto_balance",
    }


async def retrieve_documents(state: AgentState) -> dict[str, object]:
    """Run the existing synchronous RAG service without blocking the event loop."""

    service_kwargs: dict[str, object] = {}
    if state["conversation_history"] is not None:
        service_kwargs["conversation_history"] = state["conversation_history"]
    if state["metadata_filter"] is not None:
        service_kwargs["metadata_filter"] = state["metadata_filter"]
    result = await to_thread(
        rag_service_module.ask_question,
        state["question"],
        **service_kwargs,
    )
    return {
        "tool_result": result,
        "retrieved_documents": result.retrieved_documents,
        "data_source": "pinecone",
        "tool_name": "existing_rag",
        "error": None,
    }


async def query_pto_and_policy(state: AgentState) -> dict[str, object]:
    """Keep independent outcomes; a provider failure must not erase useful data."""

    pto_update = await query_pto_database(state)
    pto_result = pto_update.get("tool_result")
    category = state["requested_pto_type"] or "PTO"
    policy_question = f"Regarding {category} leave: {state['policy_question']}"
    kwargs: dict[str, object] = {}
    if state["metadata_filter"] is not None:
        kwargs["metadata_filter"] = state["metadata_filter"]
    # The focused question is standalone. Do not send personal balances or the
    # mixed conversation history to the policy model.
    try:
        policy_result = await to_thread(
            rag_service_module.ask_question, policy_question, **kwargs
        )
    except Exception:  # noqa: BLE001
        policy_result = None
    return {
        "pto_result": pto_result,
        "policy_result": policy_result,
        "retrieved_documents": (
            policy_result.retrieved_documents if policy_result else []
        ),
        "error": None,
    }


def _generate_combined_response(state: AgentState) -> dict[str, object]:
    result = state["pto_result"]
    balance_complete = bool(result and result.get("found") and not result.get("error"))
    if not result:
        balance_answer = "I couldn't retrieve PTO balance data right now. Please try again."
    elif state["requested_hours"] is not None or re.search(
        r"\b(enough|afford|cover|take|use|request)\b",
        state["balance_question"] or "",
        re.IGNORECASE,
    ):
        balance_answer = _generate_pto_feasibility_response(
            result, state["requested_hours"], state["requested_period"]
        )
        balance_complete = (
            balance_complete
            and state["requested_hours"] is not None
            and len(result.get("balances", [])) == 1
        )
    else:
        balance_answer = _generate_pto_response(result)

    policy = state["policy_result"]
    policy_complete = bool(
        isinstance(policy, RAGResult)
        and policy.resolution_status == "grounded"
        and policy.sources_used
    )
    if policy_complete or (
        isinstance(policy, RAGResult) and policy.resolution_status == "security_refusal"
    ):
        policy_answer = policy.answer
    else:
        policy_answer = (
            "I couldn't verify the requested policy guidance from the available "
            "documents. Please confirm the requirements with HR before submitting "
            "your dates."
        )
    return {
        "final_answer": f"{balance_answer}\n\nPolicy guidance: {policy_answer}",
        "partial_answer": not (balance_complete and policy_complete),
        "error": None,
    }


_ESCALATION_REASON_LABELS = {
    "user_requested_human": "User-requested escalation",
    "insufficient_document_context": "Insufficient handbook coverage",
    "ambiguous_policy": "Ambiguous policy question",
}


def _escalation_reason_label(reason: str) -> str:
    return _ESCALATION_REASON_LABELS.get(reason, reason.replace("_", " ").capitalize())


async def _notify_admin_of_escalation(
    state: AgentState, reason: str, record: Any
) -> None:
    """Best-effort admin email notification; never blocks the escalation."""

    try:
        employee_email = await to_thread(get_employee_email, state["user_id"])
    except Exception:  # noqa: BLE001
        employee_email = None

    reason_label = _escalation_reason_label(reason)
    try:
        await to_thread(
            send_escalation_email,
            subject=f"[WorkAssist AI] Escalation {record.handoff_id} - {reason_label}",
            body=(
                "A WorkAssist AI conversation has been escalated to human "
                "support.\n\n"
                f"Employee:        {state['username']} ({state['user_id']})\n"
                f"Escalation type: {reason_label}\n"
                f"Submitted:       {record.created_at}\n"
                f'Question:        "{state["question"]}"\n'
                f"Reference ID:    {record.handoff_id}\n\n"
                "Please review the employee's records and respond directly "
                "— the employee is cc'd on this email."
            ),
            cc=employee_email,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Escalation email could not be sent",
            extra={
                "operation": "escalation",
                "event": "escalation_email_failed",
                "handoff_id": record.handoff_id,
            },
        )


async def escalate_to_human(state: AgentState) -> dict[str, object]:
    """Queue an authenticated unresolved policy question for human review."""

    if not state["username"] or not state["user_id"]:
        return {"error": "employee_identity_required"}
    history = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in (state["conversation_history"] or [])
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
    try:
        reason = state["escalation_reason"] or "insufficient_document_context"
        record = await handoff_service.create_handoff(
            username=state["username"],
            employee_id=state["user_id"],
            question=state["question"],
            conversation_history=history,
            reason=reason,
        )
    except Exception:  # noqa: BLE001
        return {
            "error": "human_handoff_unavailable",
            "final_answer": (
                "I couldn't find a reliable answer, and the human handoff could "
                "not be created. Please contact HR directly."
            ),
        }
    answer = (
        "I've escalated your PTO balance concern to a human support agent. "
        f"Reference: {record.handoff_id}."
        if reason == "user_requested_human"
        else (
            "I couldn't find a reliable answer in the Employee Handbook. "
            "I've escalated your question to a human support agent. "
            f"Reference: {record.handoff_id}."
        )
    )
    await _notify_admin_of_escalation(state, reason, record)
    return {
        "escalation_required": True,
        "escalation_reason": reason,
        "handoff_id": record.handoff_id,
        "handoff_status": record.status,
        "final_answer": answer,
        "error": None,
    }


def _format_last_updated(value: object) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except ValueError:
        return None


def _generate_pto_response(result: dict[str, Any]) -> str:
    year = result.get("balance_year")
    if result.get("error"):
        return "I couldn't retrieve PTO balance data right now. Please try again."
    if not result.get("found"):
        return f"No PTO balance records were found for balance year {year}."

    lines = [f"PTO balances for balance year {year}:"]
    for balance in result.get("balances", []):
        label = str(balance["pto_type"]).replace("_", " ").title()
        lines.append(
            f"- {label}: {balance['available_hours']} hours available "
            f"({balance['accrued_hours']} accrued, {balance['used_hours']} used, "
            f"{balance['pending_hours']} pending)."
        )
    if last_updated := _format_last_updated(result.get("last_updated")):
        lines.append(f"Last updated: {last_updated}.")
    return "\n".join(lines)


def _generate_pto_feasibility_response(
    result: dict[str, Any],
    requested_hours: Decimal | None,
    requested_period: str | None,
) -> str:
    year = result.get("balance_year")
    if result.get("error"):
        return "I couldn't retrieve PTO balance data right now. Please try again."
    if not result.get("found"):
        return f"No PTO balance records were found for balance year {year}."
    if requested_hours is None:
        return (
            f"I found your PTO balances for balance year {year}, but I couldn't "
            "determine how many hours you want to use."
        )

    balances = result.get("balances", [])
    if len(balances) != 1:
        categories = ", ".join(
            str(balance["pto_type"]).replace("_", " ").title()
            for balance in balances
        )
        return (
            f"You have multiple PTO categories for balance year {year}: {categories}. "
            f"Please specify which category should cover the {requested_hours} hours."
        )

    balance = balances[0]
    available = Decimal(str(balance["available_hours"]))
    category = str(balance["pto_type"]).replace("_", " ").title()
    period = f" {requested_period}" if requested_period else ""
    if available >= requested_hours:
        projected_remaining = available - requested_hours
        comparison = (
            f"Your {category} balance has {available} hours available, which is "
            f"enough to cover {requested_hours} hours{period}. Your projected "
            f"remaining balance after using those hours will be "
            f"{projected_remaining} hours."
        )
    else:
        shortfall = requested_hours - available
        comparison = (
            f"Your {category} balance has {available} hours available, which is "
            f"{shortfall} hours short of the requested {requested_hours} hours{period}."
        )

    details = [comparison, f"Balance year: {year}."]
    if last_updated := _format_last_updated(result.get("last_updated")):
        details.append(f"Last updated: {last_updated}.")
    details.append(
        "This balance check does not approve the dates; submit the request through "
        "your organization's normal approval process."
    )
    return " ".join(details)


async def generate_response(state: AgentState) -> dict[str, object]:
    """Generate a user answer from the selected source without route details."""

    result = state["tool_result"]
    if state["intent"] == "pto_and_policy":
        return _generate_combined_response(state)
    if state["intent"] == "pto_balance" and isinstance(result, dict):
        return {"final_answer": _generate_pto_response(result), "error": None}
    if state["intent"] == "pto_request_feasibility" and isinstance(result, dict):
        return {
            "final_answer": _generate_pto_feasibility_response(
                result,
                state["requested_hours"],
                state["requested_period"],
            ),
            "error": None,
        }
    if state["intent"] == "document_question" and isinstance(result, RAGResult):
        return {"final_answer": result.answer, "error": None}
    return {"final_answer": _SAFE_GENERIC_ERROR, "error": "invalid_tool_result"}


async def handle_error(state: AgentState) -> dict[str, str]:
    """Return a safe response for routing, authorization, or provider failures."""

    if state["error"] == "employee_access_denied":
        answer = "You are not authorized to view that employee's PTO balance."
    elif state["error"] == "employee_identity_required":
        answer = "Your employee identity could not be verified for this PTO request."
    elif state["error"] == "pto_database_unavailable":
        answer = "I couldn't retrieve PTO balance data right now. Please try again."
    elif state["error"] == "human_handoff_unavailable":
        answer = (
            "I couldn't find a reliable answer, and the human handoff could not "
            "be created. Please contact HR directly."
        )
    else:
        answer = _SAFE_GENERIC_ERROR
    return {"final_answer": answer}
