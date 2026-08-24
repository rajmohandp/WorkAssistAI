"""Validated request and response models for the DocuVerse API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AskRequest(BaseModel):
    """Minimal validated request for a DocuVerse question."""

    question: str = Field(min_length=1, max_length=2000)
    document: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Question must not be empty.")
        return normalized

    @field_validator("document")
    @classmethod
    def normalize_document(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Document must not be empty when provided.")
        return normalized


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QueryFilter(BaseModel):
    document_name: str | None = Field(default=None, min_length=1, max_length=512)
    file_type: Literal["pdf", "txt", "docx"] | None = None

    @model_validator(mode="after")
    def validate_single_filter(self) -> QueryFilter:
        if self.document_name and self.file_type:
            raise ValueError("Select either document_name or file_type, not both.")
        return self

    def to_pinecone_metadata(self) -> dict[str, str] | None:
        if self.document_name:
            return {"filename": self.document_name}
        if self.file_type:
            return {"file_type": self.file_type}
        return None


class QueryRequest(AskRequest):
    history: list[ConversationMessage] = Field(default_factory=list, max_length=6)
    filters: QueryFilter | None = None


class SourceResponse(BaseModel):
    document_name: str
    page_number: int | None = None
    citation: str
    excerpt: str


class AskSourceResponse(BaseModel):
    """Concise source reference returned by the public ask endpoint."""

    document: str
    page: int | None = None


class EvaluationResponse(BaseModel):
    faithfulness_percentage: float | None = None
    answer_relevance_percentage: float | None = None
    retrieval_confidence_percentage: float | None = None
    overall_confidence_percentage: float | None = None
    status: Literal["evaluated", "not_evaluated", "unavailable"]


class AskResponse(BaseModel):
    """Minimal grounded answer contract."""

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1)
    sources: list[AskSourceResponse] = Field(default_factory=list)
    evaluation: EvaluationResponse | None = None


class QueryResponse(AskResponse):
    sources: list[SourceResponse] = Field(default_factory=list)
    retrieval_query: str


class S3StatusResponse(BaseModel):
    connected: bool
    bucket: str
    supported_documents: int
    documents: list[str] = Field(default_factory=list)


class PineconeStatusResponse(BaseModel):
    connected: bool
    index_name: str
    indexed_documents: int | None
    total_vectors: int


class RepositoryStatusResponse(BaseModel):
    s3: S3StatusResponse
    pinecone: PineconeStatusResponse


class SyncResponse(BaseModel):
    documents_scanned: int
    new_documents: int
    updated_documents: int
    unchanged_documents: int
    failed_documents: int
    chunks_added: int
    chunks_removed: int
    documents_removed_from_s3: int
    vectors_removed_from_pinecone: int
    documents_added: int
    documents_updated: int
    documents_removed: int


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    application: str = "DocuVerse"


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


JsonMetadata = dict[str, Any]
