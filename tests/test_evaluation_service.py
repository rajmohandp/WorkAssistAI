"""Tests for the isolated, fail-open RAG evaluation layer."""

from langchain_core.documents import Document

from app.services.evaluation_service import (
    EvaluationJudgment,
    evaluate_rag_result,
)
from src.rag_chain import INSUFFICIENT_CONTEXT_MESSAGE, RAGResult


def _grounded_result() -> RAGResult:
    document = Document(
        page_content="Employees receive ten vacation days.",
        metadata={"filename": "handbook.pdf", "page": 1},
    )
    return RAGResult(
        answer="Employees receive ten vacation days. [handbook.pdf, Page 2]",
        retrieved_documents=[document, document],
        source_metadata=[
            {"similarity_score": 0.86},
            {"similarity_score": 0.72},
        ],
        sources_used=[],
        retrieval_query="vacation days",
    )


def test_evaluation_calculates_percentages(monkeypatch):
    monkeypatch.setattr(
        "app.services.evaluation_service._judge_response",
        lambda *_args, **_kwargs: EvaluationJudgment(
            supported_claims=2,
            partially_supported_claims=1,
            unsupported_claims=1,
            answer_relevance=0.8,
        ),
    )

    evaluation = evaluate_rag_result(
        "How many vacation days are available?",
        _grounded_result(),
        evaluator_model=object(),
    )

    assert evaluation.status == "evaluated"
    assert evaluation.faithfulness_percentage == 62.5
    assert evaluation.answer_relevance_percentage == 80.0
    assert evaluation.retrieval_confidence_percentage == 70.0
    assert evaluation.overall_confidence_percentage == 68.5


def test_fallback_answer_is_not_evaluated():
    result = RAGResult(
        answer=INSUFFICIENT_CONTEXT_MESSAGE,
        retrieved_documents=[],
        source_metadata=[],
        sources_used=[],
        retrieval_query="unknown",
    )

    evaluation = evaluate_rag_result(
        "Unknown question",
        result,
        evaluator_model=object(),
    )

    assert evaluation.status == "not_evaluated"
    assert evaluation.overall_confidence_percentage is None


def test_evaluation_failure_does_not_raise(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr("app.services.evaluation_service._judge_response", fail)

    evaluation = evaluate_rag_result(
        "How many vacation days are available?",
        _grounded_result(),
        evaluator_model=object(),
    )

    assert evaluation.status == "unavailable"
    assert evaluation.faithfulness_percentage is None
