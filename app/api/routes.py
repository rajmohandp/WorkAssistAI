"""HTTP routes for the DocuVerse backend."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.agent.graph import execute_graph
from app.agent.state import create_agent_state
from app.api.schemas import (
    AskRequest,
    AskResponse,
    AskSourceResponse,
    ErrorResponse,
    EvaluationResponse,
    HandoffRecordResponse,
    HandoffResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RepositoryStatusResponse,
    SyncResponse,
)
from app.core.security import AuthenticatedUser, get_current_user, require_admin
from app.services.conversation_service import get_conversational_response
from app.services.handoff_service import handoff_service
from app.services.pinecone_service import PineconeService
from app.services.rag_service import RagService
from app.services.s3_service import S3Service
from app.services.sync_service import SyncService
from src.rag_chain import RAGError, RAGResult
from src.retriever import RetrievalError
from src.s3_loader import S3RepositoryError
from src.vector_store import VectorStoreError

logger = logging.getLogger(__name__)
router = APIRouter()
rag_service = RagService()
s3_service = S3Service()
pinecone_service = PineconeService()
sync_service = SyncService()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["rag"],
)
async def ask(
    request: AskRequest,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> AskResponse:
    """Answer one question using the reusable, framework-neutral RAG service."""

    try:
        if conversational_answer := get_conversational_response(request.question):
            logger.info(
                "Conversational API request handled without RAG",
                extra={
                    "operation": "application",
                    "event": "conversation_short_circuited",
                },
            )
            return AskResponse(
                question=request.question,
                answer=conversational_answer,
                sources=[],
            )
        service_kwargs = (
            {"metadata_filter": {"filename": request.document}}
            if request.document
            else {}
        )
        graph_state = create_agent_state(
            request.question,
            user_id=current_user.employee_id,
            username=current_user.username,
            user_role=current_user.role,
            metadata_filter=service_kwargs.get("metadata_filter"),
            conversation_history=(
                [message.model_dump() for message in request.history]
                if request.history
                else None
            ),
        )
        graph_result = await execute_graph(graph_state)
        result = graph_result["tool_result"]
        sources = (
            [
                AskSourceResponse(
                    document=source.document_name,
                    page=source.page_number,
                )
                for source in result.sources_used
            ]
            if isinstance(result, RAGResult)
            else []
        )
        evaluation = (
            EvaluationResponse(**result.evaluation)
            if isinstance(result, RAGResult) and result.evaluation
            else None
        )
        return AskResponse(
            question=request.question,
            answer=graph_result["final_answer"] or "The request could not be completed.",
            sources=sources,
            evaluation=evaluation,
            resolution=(
                "escalated" if graph_result["escalation_required"] else "answered"
            ),
            handoff=(
                HandoffResponse(
                    handoff_id=graph_result["handoff_id"],
                    status="queued",
                    reason=(
                        graph_result["escalation_reason"]
                        or "insufficient_document_context"
                    ),
                )
                if graph_result["escalation_required"]
                and graph_result["handoff_id"]
                else None
            ),
        )
    except (RAGError, RetrievalError, VectorStoreError, ValueError) as exc:
        logger.error(
            "Ask API request failed",
            extra={
                "operation": "application",
                "event": "api_ask_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WorkAssist AI could not complete the question.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected Ask API failure",
            extra={
                "operation": "application",
                "event": "api_ask_unexpected_failure",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the question.",
        ) from exc


@router.get(
    "/repository/status",
    response_model=RepositoryStatusResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["repository"],
)
async def repository_status(
    _current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> RepositoryStatusResponse:
    try:
        s3_status, pinecone_status = await asyncio.gather(
            run_in_threadpool(s3_service.status),
            run_in_threadpool(pinecone_service.status),
        )
        return RepositoryStatusResponse(s3=s3_status, pinecone=pinecone_status)
    except (S3RepositoryError, VectorStoreError, ValueError) as exc:
        logger.error(
            "Repository status API request failed",
            extra={
                "operation": "application",
                "event": "api_repository_status_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The document repository is currently unavailable.",
        ) from exc


@router.post(
    "/repository/sync",
    response_model=SyncResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["repository"],
)
async def synchronize_repository(
    _current_admin: Annotated[AuthenticatedUser, Depends(require_admin)],
) -> SyncResponse:
    """Incrementally synchronize the configured S3 bucket with Pinecone."""

    try:
        return await run_in_threadpool(sync_service.synchronize)
    except (S3RepositoryError, VectorStoreError, ValueError) as exc:
        logger.error(
            "Repository synchronization API request failed",
            extra={
                "operation": "application",
                "event": "api_repository_sync_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document synchronization could not be completed.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected repository synchronization failure",
            extra={
                "operation": "application",
                "event": "api_repository_sync_unexpected_failure",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Document synchronization could not connect to a required "
                "repository provider. Check the API logs and network access."
            ),
        ) from exc


@router.get(
    "/handoffs",
    response_model=list[HandoffRecordResponse],
    tags=["human-support"],
)
async def list_handoffs(
    _current_admin: Annotated[AuthenticatedUser, Depends(require_admin)],
) -> list[HandoffRecordResponse]:
    """List queued human-support cases for authenticated administrators."""

    return [
        HandoffRecordResponse(**record)
        for record in handoff_service.list_handoffs()
    ]


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["rag"],
)
async def query(
    request: QueryRequest,
    _current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> QueryResponse:
    try:
        return await run_in_threadpool(rag_service.answer, request)
    except (RAGError, RetrievalError, VectorStoreError, ValueError) as exc:
        logger.error(
            "RAG API request failed",
            extra={
                "operation": "application",
                "event": "api_query_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WorkAssist AI could not complete the query.",
        ) from exc
