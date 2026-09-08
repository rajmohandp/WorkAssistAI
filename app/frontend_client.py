"""Small HTTP client used by the Streamlit frontend."""

from __future__ import annotations

from typing import Any, TypeAlias

import requests

from src.config import get_environment_settings

API_BASE_URL = get_environment_settings().backend_url.rstrip("/")
HTTPTimeout: TypeAlias = float | tuple[float, float]
DEFAULT_HTTP_TIMEOUT: HTTPTimeout = (5.0, 15.0)
CHAT_HTTP_TIMEOUT: HTTPTimeout = (5.0, 90.0)
SYNC_HTTP_TIMEOUT: HTTPTimeout = (5.0, 600.0)
BACKEND_UNAVAILABLE_MESSAGE = (
    "WorkAssist AI is temporarily unavailable. Please try again shortly."
)


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


def _authorization_headers(access_token: str) -> dict[str, str]:
    if not access_token:
        raise DocuVerseAPIError("Authentication is required.")
    return {"Authorization": f"Bearer {access_token}"}


def login_http(
    username: str,
    password: str,
    *,
    timeout: HTTPTimeout = DEFAULT_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Authenticate with FastAPI and return a backend-issued access token."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DocuVerseAPIError(BACKEND_UNAVAILABLE_MESSAGE) from exc
    payload = _read_json_response(response)
    if not isinstance(payload.get("access_token"), str):
        raise DocuVerseAPIError("The API returned an invalid authentication response.")
    return payload


def get_current_user_http(
    access_token: str,
    *,
    timeout: HTTPTimeout = DEFAULT_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Return identity established by the backend token."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/auth/me",
            headers=_authorization_headers(access_token),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DocuVerseAPIError(BACKEND_UNAVAILABLE_MESSAGE) from exc
    return _read_json_response(response)


def get_repository_status(
    access_token: str,
    *,
    timeout: HTTPTimeout = DEFAULT_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Read repository status from FastAPI."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/repository/status",
            headers=_authorization_headers(access_token),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DocuVerseAPIError(BACKEND_UNAVAILABLE_MESSAGE) from exc
    return _read_json_response(response)


def get_handoffs_http(
    access_token: str,
    *,
    timeout: HTTPTimeout = DEFAULT_HTTP_TIMEOUT,
) -> list[dict[str, Any]]:
    """Return the authenticated administrator's human-support queue."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/handoffs",
            headers=_authorization_headers(access_token),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DocuVerseAPIError(BACKEND_UNAVAILABLE_MESSAGE) from exc
    if not response.ok:
        raise DocuVerseAPIError(_safe_error_detail(response))
    try:
        payload = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise DocuVerseAPIError("The API returned an invalid response.") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise DocuVerseAPIError("The API returned an invalid handoff queue.")
    return payload


def synchronize_repository(
    access_token: str,
    *,
    timeout: HTTPTimeout = SYNC_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Request incremental S3-to-Pinecone synchronization from FastAPI."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/repository/sync",
            headers=_authorization_headers(access_token),
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise DocuVerseAPIError(
            "Document synchronization timed out. Check the API logs."
        ) from exc
    except requests.ConnectionError as exc:
        raise DocuVerseAPIError(
            BACKEND_UNAVAILABLE_MESSAGE
        ) from exc
    except requests.RequestException as exc:
        raise DocuVerseAPIError("Document synchronization could not be started.") from exc
    if not response.ok:
        detail = _safe_error_detail(response)
        if detail == "The request could not be completed.":
            detail = (
                "Document synchronization failed. Check FastAPI provider and "
                "network logs."
            )
        raise DocuVerseAPIError(detail)
    return _read_json_response(response)


def ask_question_http(
    question: str,
    *,
    document: str | None = None,
    access_token: str,
    history: list[dict[str, str]] | None = None,
    timeout: HTTPTimeout = CHAT_HTTP_TIMEOUT,
) -> dict[str, Any]:
    """Submit a question to FastAPI and return its validated JSON response."""

    request_payload = {"question": question}
    if document:
        request_payload["document"] = document
    if history:
        request_payload["history"] = [
            {"role": message["role"], "content": message["content"]}
            for message in history[-6:]
        ]
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            headers=_authorization_headers(access_token),
            json=request_payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise DocuVerseAPIError(
            "WorkAssist AI took too long to respond. Please try again."
        ) from exc
    except requests.ConnectionError as exc:
        raise DocuVerseAPIError(
            BACKEND_UNAVAILABLE_MESSAGE
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
