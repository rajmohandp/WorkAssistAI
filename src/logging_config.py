"""Secret-safe structured logging configuration for DocuVerse."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.config import get_environment_settings
from src.guardrails import redact_sensitive_text

_HANDLER_MARKER = "_docuverse_structured_handler"
_STANDARD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


class StructuredJsonFormatter(logging.Formatter):
    """Format application logs as one redacted JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_text(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_FIELDS or key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = (
                    redact_sensitive_text(value) if isinstance(value, str) else value
                )
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Configure DocuVerse application namespaces exactly once."""

    level_name = get_environment_settings().log_level.strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(StructuredJsonFormatter())

    for logger_name in ("src", "app", "docuverse.app"):
        configured_logger = logging.getLogger(logger_name)
        configured_logger.setLevel(level)
        configured_logger.propagate = False
        if not any(
            getattr(existing, _HANDLER_MARKER, False)
            for existing in configured_logger.handlers
        ):
            configured_logger.addHandler(handler)
