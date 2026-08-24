"""HTTP routes for the DocuVerse backend."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.api.schemas import (
    AskRequest,
    AskResponse,
    AskSourceResponse,
    ErrorResponse,
    EvaluationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RepositoryStatusResponse,
    SyncResponse,
)
from app.services import rag_service as rag_service_module
from app.services.conversation_service import get_conversational_response
from app.services.pinecone_service import PineconeService
from app.services.rag_service import RagService
from app.services.s3_service import S3Service
from app.services.sync_service import SyncService
from src.rag_chain import RAGError
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
async def ask(request: AskRequest) -> AskResponse:
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
        result = await run_in_threadpool(
            rag_service_module.ask_question,
            request.question,
            **service_kwargs,
        )
        return AskResponse(
            question=request.question,
            answer=result.answer,
            sources=[
                AskSourceResponse(
                    document=source.document_name,
                    page=source.page_number,
                )
                for source in result.sources_used
            ],
            evaluation=(
                EvaluationResponse(**result.evaluation)
                if result.evaluation
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
            detail="DocuVerse could not complete the question.",
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
async def repository_status() -> RepositoryStatusResponse:
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
async def synchronize_repository() -> SyncResponse:
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


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["rag"],
)
async def query(request: QueryRequest) -> QueryResponse:
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
            detail="DocuVerse could not complete the query.",
        ) from exc
