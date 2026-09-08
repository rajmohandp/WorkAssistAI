"""List supported documents stored in the configured Amazon S3 bucket."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, TypedDict
from uuid import NAMESPACE_URL, uuid4, uuid5

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    ReadTimeoutError,
)
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.config import get_environment_settings

BUCKET_ENV_VAR = "S3_BUCKET_NAME"
SUPPORTED_FILE_TYPES = frozenset({"pdf", "txt", "docx"})
logger = logging.getLogger(__name__)


class S3DocumentMetadata(TypedDict):
    """Metadata returned for one supported S3 object."""

    file_name: str
    s3_key: str
    file_type: str
    file_size: int
    last_modified: Any
    etag: str


class FailedDocument(TypedDict):
    """Safe failure details suitable for logs and the UI."""

    filename: str
    s3_key: str
    error: str


@dataclass(frozen=True)
class DocumentLoadResult:
    """Documents extracted during one ingestion attempt."""

    documents: list[Document]
    loaded_files: int
    failed_documents: list[FailedDocument]


class S3RepositoryError(RuntimeError):
    """Base error for failures while accessing the document repository."""


class S3ConfigurationError(S3RepositoryError):
    """Raised when required S3 configuration is missing."""


class S3CredentialsError(S3RepositoryError):
    """Raised when AWS credentials are absent, incomplete, or invalid."""


class S3BucketNotFoundError(S3RepositoryError):
    """Raised when the configured bucket does not exist."""


class S3ObjectNotFoundError(S3RepositoryError):
    """Raised when a listed S3 object no longer exists."""


class S3AccessDeniedError(S3RepositoryError):
    """Raised when AWS denies access to the configured bucket."""


class S3NetworkError(S3RepositoryError):
    """Raised when S3 cannot be reached."""


def get_bucket_name() -> str:
    """Return the S3 bucket name from the local environment."""

    bucket_name = get_environment_settings().s3_bucket_name.strip()
    if not bucket_name:
        raise S3ConfigurationError(
            f"Set the {BUCKET_ENV_VAR} environment variable to an S3 bucket name."
        )
    return bucket_name


def _create_s3_client() -> Any:
    """Create an S3 client using boto3's standard AWS credential chain."""

    settings = get_environment_settings()
    session_options: dict[str, str] = {}
    if region := settings.aws_region.strip():
        session_options["region_name"] = region
    if profile := settings.aws_profile.strip():
        session_options["profile_name"] = profile
    else:
        for name in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
            if value := getattr(settings, name).get_secret_value():
                session_options[name] = value
    try:
        return boto3.Session(**session_options).client("s3")
    except ProfileNotFound as exc:
        raise S3CredentialsError(
            "The configured local AWS profile could not be found."
        ) from exc


def _raise_s3_client_error(
    exc: ClientError, *, object_operation: bool = False
) -> None:
    """Translate an AWS response without exposing its potentially sensitive text."""

    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    logger.error(
        "S3 request failed",
        extra={
            "operation": "s3",
            "event": "request_failed",
            "error_code": code,
            "http_status": status,
        },
    )
    if code in {"InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken"}:
        raise S3CredentialsError("AWS credentials are invalid or expired.") from exc
    object_missing = code in {
        "NoSuchKey",
        "NoSuchObject",
        "NotFound",
        "404",
    } or status == 404
    if object_operation and object_missing:
        raise S3ObjectNotFoundError("The requested S3 object does not exist.") from exc
    if code in {"NoSuchBucket", "NotFound", "404"} or status == 404:
        raise S3BucketNotFoundError(
            "The configured S3 bucket does not exist."
        ) from exc
    if code in {"AccessDenied", "AllAccessDisabled", "403"} or status == 403:
        raise S3AccessDeniedError(
            "Access to the configured S3 bucket was denied."
        ) from exc
    raise S3RepositoryError("Amazon S3 returned an unexpected error.") from exc


def list_documents(
    bucket_name: str | None = None, *, s3_client: Any | None = None
) -> list[S3DocumentMetadata]:
    """List PDF, TXT, and DOCX objects and return their S3 metadata."""

    resolved_bucket = bucket_name or get_bucket_name()
    logger.debug(
        "Starting S3 document scan",
        extra={"operation": "s3", "event": "scan_started"},
    )

    try:
        client = s3_client or _create_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        documents: list[S3DocumentMetadata] = []

        for page in paginator.paginate(Bucket=resolved_bucket):
            for item in page.get("Contents", []):
                key = item["Key"]
                path = PurePosixPath(key)
                file_type = path.suffix.lower().lstrip(".")
                if not path.name or file_type not in SUPPORTED_FILE_TYPES:
                    continue

                documents.append(
                    {
                        "file_name": path.name,
                        "s3_key": key,
                        "file_type": file_type,
                        "file_size": int(item.get("Size", 0)),
                        "last_modified": item.get("LastModified"),
                        "etag": str(item.get("ETag", "")).strip('"'),
                    }
                )

        logger.info(
            "S3 document scan completed",
            extra={
                "operation": "s3",
                "event": "scan_completed",
                "document_count": len(documents),
            },
        )
        return documents
    except (NoCredentialsError, PartialCredentialsError) as exc:
        logger.error(
            "S3 credentials unavailable",
            extra={
                "operation": "s3",
                "event": "credentials_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise S3CredentialsError(
            "AWS credentials are missing or incomplete."
        ) from exc
    except ClientError as exc:
        _raise_s3_client_error(exc)
    except (
        EndpointConnectionError,
        ConnectionClosedError,
        ConnectTimeoutError,
        ReadTimeoutError,
    ) as exc:
        logger.error(
            "S3 network operation failed",
            extra={
                "operation": "s3",
                "event": "network_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise S3NetworkError("Could not connect to Amazon S3.") from exc
    except BotoCoreError as exc:
        logger.error(
            "AWS communication failed",
            extra={
                "operation": "s3",
                "event": "communication_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise S3NetworkError("AWS communication failed.") from exc


def _loader_for(file_type: str, local_path: Path) -> Any:
    """Create the appropriate LangChain loader for a downloaded file."""

    if file_type == "pdf":
        return PyPDFLoader(str(local_path))
    if file_type == "txt":
        return TextLoader(str(local_path), autodetect_encoding=True)
    if file_type == "docx":
        return Docx2txtLoader(str(local_path))
    raise ValueError(f"Unsupported file type: {file_type}")


def _apply_s3_metadata(
    document: Document, *, bucket_name: str, item: S3DocumentMetadata
) -> Document:
    """Replace temporary-path metadata with stable S3 metadata."""

    loader_page = document.metadata.get("page")
    last_modified = item.get("last_modified")
    last_modified_value = (
        last_modified.isoformat()
        if isinstance(last_modified, datetime)
        else str(last_modified or "")
    )
    document_id = str(
        uuid5(NAMESPACE_URL, f"s3://{bucket_name}/{item['s3_key']}")
    )
    version_identity = "|".join(
        (
            item.get("etag", ""),
            str(item.get("file_size", 0)),
            last_modified_value,
        )
    )
    metadata = {
        **document.metadata,
        "source": f"s3://{bucket_name}/{item['s3_key']}",
        "s3_key": item["s3_key"],
        "filename": item["file_name"],
        "file_type": item["file_type"],
        "file_size": item["file_size"],
        "etag": item.get("etag", ""),
        "last_modified": last_modified_value,
        "document_id": document_id,
        "document_version": sha256(version_identity.encode("utf-8")).hexdigest(),
    }
    metadata.pop("page", None)
    if isinstance(loader_page, int):
        metadata["page"] = loader_page
        metadata["page_number"] = loader_page + 1

    return Document(page_content=document.page_content, metadata=metadata)


def load_documents(
    bucket_name: str | None = None,
    *,
    document_metadata: list[S3DocumentMetadata] | None = None,
    s3_client: Any | None = None,
) -> DocumentLoadResult:
    """Download and extract all supported S3 documents.

    Each downloaded object lives only inside a temporary directory. A corrupt or
    unreadable object is recorded as a failure while the remaining objects
    continue processing.
    """

    resolved_bucket = bucket_name or get_bucket_name()
    client = s3_client or _create_s3_client()
    items = (
        document_metadata
        if document_metadata is not None
        else list_documents(resolved_bucket, s3_client=client)
    )
    extracted: list[Document] = []
    failed: list[FailedDocument] = []
    loaded_files = 0

    with TemporaryDirectory(prefix="docuverse-") as temp_directory:
        temp_path = Path(temp_directory)
        for item in items:
            logger.debug(
                "Loading S3 document",
                extra={
                    "operation": "document_loading",
                    "event": "load_started",
                    "document": item["s3_key"],
                },
            )
            local_path = temp_path / f"{uuid4().hex}.{item['file_type']}"
            try:
                client.download_file(resolved_bucket, item["s3_key"], str(local_path))
                loaded = _loader_for(item["file_type"], local_path).load()
                extracted.extend(
                    _apply_s3_metadata(
                        document,
                        bucket_name=resolved_bucket,
                        item=item,
                    )
                    for document in loaded
                )
                loaded_files += 1
                logger.info(
                    "S3 document loaded",
                    extra={
                        "operation": "document_loading",
                        "event": "load_completed",
                        "document": item["s3_key"],
                        "extracted_documents": len(loaded),
                    },
                )
            except (NoCredentialsError, PartialCredentialsError) as exc:
                raise S3CredentialsError(
                    "AWS credentials are missing or incomplete."
                ) from exc
            except ClientError as exc:
                try:
                    _raise_s3_client_error(exc, object_operation=True)
                except S3ObjectNotFoundError:
                    logger.warning(
                        "S3 object is missing; continuing ingestion",
                        extra={
                            "operation": "document_loading",
                            "event": "object_missing",
                        },
                    )
                    failed.append(
                        {
                            "filename": item["file_name"],
                            "s3_key": item["s3_key"],
                            "error": "The S3 object no longer exists.",
                        }
                    )
            except (
                EndpointConnectionError,
                ConnectionClosedError,
                ConnectTimeoutError,
                ReadTimeoutError,
            ) as exc:
                raise S3NetworkError("Could not connect to Amazon S3.") from exc
            except BotoCoreError as exc:
                raise S3NetworkError("AWS communication failed.") from exc
            # Third-party parsers can raise format-specific exception types. The
            # file boundary intentionally contains all of them so one corrupt
            # object cannot abort the remaining ingestion batch.
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "S3 document loading failed; continuing ingestion",
                    extra={
                        "operation": "document_loading",
                        "event": "load_failed",
                        "document": item["s3_key"],
                        "error_type": type(exc).__name__,
                    },
                )
                failed.append(
                    {
                        "filename": item["file_name"],
                        "s3_key": item["s3_key"],
                        "error": "The document could not be downloaded or read.",
                    }
                )

    return DocumentLoadResult(
        documents=extracted,
        loaded_files=loaded_files,
        failed_documents=failed,
    )
