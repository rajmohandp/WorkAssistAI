"""Application service for grounded DocuVerse questions."""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter

from app.api.schemas import (
    EvaluationResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)
from app.core.dependencies import RAGDependencies, get_rag_dependencies
from app.services.conversation_service import get_conversational_response
from app.services.evaluation_service import evaluate_rag_result
from src.config import get_environment_settings
from src.rag_chain import ConversationHistory, RAGResult, run_rag
from src.retriever import MetadataFilter

logger = logging.getLogger(__name__)


def ask_question(
    question: str,
    *,
    conversation_history: ConversationHistory | None = None,
    metadata_filter: MetadataFilter | None = None,
    top_k: int = 5,
    dependencies: RAGDependencies | None = None,
) -> RAGResult:
    """Run the complete reusable RAG workflow for one user question.

    This boundary is independent of FastAPI and Streamlit. It embeds the query,
    searches Pinecone, applies retrieval guardrails, invokes the configured LLM,
    and returns the grounded answer with retrieved source metadata.
    """

    started = perf_counter()
    logger.info(
        "RAG request started",
        extra={
            "operation": "application",
            "event": "rag_request_started",
            "top_k": top_k,
            "filtered": bool(metadata_filter),
        },
    )
    try:
        resolved_dependencies = dependencies or get_rag_dependencies()
        result = run_rag(
            question,
            top_k=top_k,
            vector_store=resolved_dependencies.vector_store,
            chat_model=resolved_dependencies.chat_model,
            metadata_filter=metadata_filter,
            conversation_history=conversation_history,
        )
        if get_environment_settings().rag_evaluation_enabled:
            evaluation = evaluate_rag_result(
                question,
                result,
                evaluator_model=resolved_dependencies.chat_model,
            )
            result = replace(result, evaluation=evaluation.to_dict())
    except Exception as exc:
        logger.exception(
            "RAG request failed",
            extra={
                "operation": "application",
                "event": "rag_request_exception",
                "error_type": type(exc).__name__,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        raise
    logger.info(
        "RAG request completed",
        extra={
            "operation": "application",
            "event": "rag_request_completed",
            "chunks_retrieved": len(result.retrieved_documents),
            "sources_used": len(result.sources_used),
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        },
    )
    return result


class RagService:
    """Translate API requests to the existing reusable RAG pipeline."""

    def __init__(self, dependencies: RAGDependencies | None = None) -> None:
        self._dependencies = dependencies

    def answer(self, request: QueryRequest) -> QueryResponse:
        if conversational_answer := get_conversational_response(request.question):
            logger.info(
                "Conversational query handled without RAG",
                extra={
                    "operation": "application",
                    "event": "conversation_short_circuited",
                },
            )
            return QueryResponse(
                question=request.question,
                answer=conversational_answer,
                retrieval_query="",
                sources=[],
            )
        history = [message.model_dump() for message in request.history]
        metadata_filter = (
            request.filters.to_pinecone_metadata() if request.filters else None
        )
        result = ask_question(
            request.question,
            metadata_filter=metadata_filter,
            conversation_history=history,
            dependencies=self._dependencies,
        )
        return QueryResponse(
            question=request.question,
            answer=result.answer,
            retrieval_query=result.retrieval_query,
            sources=[
                SourceResponse(
                    document_name=source.document_name,
                    page_number=source.page_number,
                    citation=source.citation,
                    excerpt=source.excerpt,
                )
                for source in result.sources_used
            ],
            evaluation=(
                EvaluationResponse(**result.evaluation)
                if result.evaluation
                else None
            ),
        )
