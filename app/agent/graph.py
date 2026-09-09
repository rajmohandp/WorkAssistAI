"""Compiled LangGraph and asynchronous execution boundary."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    escalate_to_human,
    generate_response,
    handle_error,
    query_pto_and_policy,
    query_pto_database,
    resolve_employee,
    retrieve_documents,
)
from app.agent.router import (
    classify_request,
    route_after_employee_resolution,
    route_after_pto_query,
    route_after_retrieval,
    route_classified_request,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> CompiledStateGraph:
    """Build and compile the initial DocuVerse orchestration graph."""

    builder = StateGraph(AgentState)
    builder.add_node("classify_request", classify_request)
    builder.add_node("resolve_employee", resolve_employee)
    builder.add_node("query_pto_database", query_pto_database)
    builder.add_node("query_pto_and_policy", query_pto_and_policy)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("generate_response", generate_response)
    builder.add_node("handle_error", handle_error)
    builder.add_node("escalate_to_human", escalate_to_human)
    builder.add_edge(START, "classify_request")
    builder.add_conditional_edges("classify_request", route_classified_request)
    builder.add_conditional_edges(
        "resolve_employee",
        route_after_employee_resolution,
    )
    builder.add_conditional_edges("query_pto_database", route_after_pto_query)
    builder.add_conditional_edges("retrieve_documents", route_after_retrieval)
    builder.add_edge("escalate_to_human", END)
    builder.add_edge("generate_response", END)
    builder.add_edge("query_pto_and_policy", "generate_response")
    builder.add_edge("handle_error", END)
    return builder.compile()


docuverse_graph = build_graph()


async def execute_graph(state: AgentState) -> AgentState:
    """Execute one request with safe, structured lifecycle logging."""

    started = perf_counter()
    log_fields: dict[str, Any] = {
        "operation": "orchestration",
        "event": "graph_execution_started",
        "authenticated": bool(state["user_id"]),
    }
    logger.info("LangGraph execution started", extra=log_fields)
    try:
        result = await docuverse_graph.ainvoke(state)
    except Exception as exc:
        logger.exception(
            "LangGraph execution failed",
            extra={
                **log_fields,
                "event": "graph_execution_failed",
                "error_type": type(exc).__name__,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        raise
    logger.info(
        "LangGraph execution completed",
        extra={
            **log_fields,
            "event": "graph_execution_completed",
            "intent": result.get("intent"),
            "data_source": result.get("data_source"),
            "tool_name": result.get("tool_name"),
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        },
    )
    return result
