"""Small HTTP client used by the Streamlit frontend."""

from __future__ import annotations

from typing import Any

import requests

from src.config import get_environment_settings

API_BASE_URL = get_environment_settings().fastapi_url.rstrip("/")


class DocuVerseAPIError(RuntimeError):
    """Raised when the DocuVerse backend cannot satisfy a frontend request."""


def _safe_error_detail(response: requests.Response) -> str:
    """Return a user-safe API error without exposing response internals."""

    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
    except (requests.exceptions.JSONDecodeError, ValueError):
        detail = None
    return detail if isinstance(detail, str) else "The request could not be completed."


def get_repository_status(*, timeout: float = 10) -> dict[str, Any]:
    """Read repository status from FastAPI."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/repository/status",
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DocuVerseAPIError("The DocuVerse API is unavailable.") from exc
    return _read_json_response(response)


def synchronize_repository(*, timeout: float = 600) -> dict[str, Any]:
    """Request incremental S3-to-Pinecone synchronization from FastAPI."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/repository/sync",
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise DocuVerseAPIError(
            "Document synchronization timed out. Check the API logs."
        ) from exc
    except requests.ConnectionError as exc:
        raise DocuVerseAPIError(
            "Cannot connect to the DocuVerse API. Confirm FastAPI is running."
        ) from exc
    except requests.RequestException as exc:
        raise DocuVerseAPIError("Document synchronization could not be started.") from exc
    return _read_json_response(response)


def ask_question_http(
    question: str,
    *,
    document: str | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    """Submit a question to FastAPI and return its validated JSON response."""

    request_payload = {"question": question}
    if document:
        request_payload["document"] = document
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=request_payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise DocuVerseAPIError(
            "DocuVerse took too long to respond. Please try again."
        ) from exc
    except requests.ConnectionError as exc:
        raise DocuVerseAPIError(
            "Cannot connect to the DocuVerse API. Confirm FastAPI is running."
        ) from exc
    except requests.RequestException as exc:
        raise DocuVerseAPIError("The question could not be sent.") from exc

    payload = _read_json_response(response)
    if not isinstance(payload.get("answer"), str):
        raise DocuVerseAPIError("The API response did not contain an answer.")
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise DocuVerseAPIError("The API response contained invalid sources.")
    return payload


def _read_json_response(response: requests.Response) -> dict[str, Any]:
    if not response.ok:
        raise DocuVerseAPIError(_safe_error_detail(response))
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise DocuVerseAPIError("The API returned an invalid response.") from exc
    if not isinstance(payload, dict):
        raise DocuVerseAPIError("The API returned an invalid response.")
    return payload
