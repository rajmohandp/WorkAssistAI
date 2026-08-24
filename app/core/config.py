"""Configuration for the DocuVerse HTTP API."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import get_environment_settings


@dataclass(frozen=True)
class ApiSettings:
    """Non-secret API settings."""

    title: str = "DocuVerse API"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"


def get_api_settings() -> ApiSettings:
    """Load API settings without exposing provider credentials."""

    prefix = get_environment_settings().docuverse_api_prefix.strip()
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    return ApiSettings(api_prefix=prefix.rstrip("/"))
