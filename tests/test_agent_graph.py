"""Focused tests for LangGraph construction."""

from langgraph.graph.state import CompiledStateGraph

from app.agent.graph import build_graph


def test_graph_compiles_with_expected_flow():
    graph = build_graph()

    assert isinstance(graph, CompiledStateGraph)
    drawable = graph.get_graph()
    assert {
        "classify_request",
        "resolve_employee",
        "query_pto_database",
        "retrieve_documents",
        "generate_response",
        "handle_error",
    }.issubset(drawable.nodes)
    edges = {(edge.source, edge.target) for edge in drawable.edges}
    assert ("__start__", "classify_request") in edges
    assert ("classify_request", "resolve_employee") in edges
    assert ("classify_request", "retrieve_documents") in edges
    assert ("resolve_employee", "query_pto_database") in edges
    assert ("query_pto_database", "generate_response") in edges
    assert ("retrieve_documents", "generate_response") in edges
    assert ("generate_response", "__end__") in edges
    assert ("handle_error", "__end__") in edges
