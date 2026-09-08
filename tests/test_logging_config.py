"""Tests for secret-safe structured application logging."""

import json
import logging

from fastapi.testclient import TestClient

from app.main import app
from src.logging_config import StructuredJsonFormatter, configure_logging


def test_structured_formatter_includes_timestamp_level_and_operation():
    record = logging.LogRecord(
        name="src.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Document loaded",
        args=(),
        exc_info=None,
    )
    record.operation = "document_loading"
    record.event = "document_loaded"

    payload = json.loads(StructuredJsonFormatter().format(record))

    assert payload["timestamp"]
    assert payload["level"] == "INFO"
    assert payload["operation"] == "document_loading"
    assert payload["event"] == "document_loaded"


def test_structured_formatter_redacts_secrets():
    record = logging.LogRecord(
        name="src.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=10,
        msg="OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        args=(),
        exc_info=None,
    )

    output = StructuredJsonFormatter().format(record)

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in output
    assert "[REDACTED]" in output


def test_logging_configuration_is_idempotent():
    configure_logging()
    configure_logging()

    logger = logging.getLogger("src")
    structured_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_docuverse_structured_handler", False)
    ]
    assert len(structured_handlers) == 1


def test_structured_formatter_redacts_employee_fields():
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Request handled",
        args=(),
        exc_info=None,
    )
    record.username = "private-user"
    record.employee_id = "EMP-SECRET"
    record.question = "private employee question"

    output = StructuredJsonFormatter().format(record)

    assert "private-user" not in output
    assert "EMP-SECRET" not in output
    assert "private employee question" not in output


def test_api_middleware_logs_received_and_completed_without_body(monkeypatch):
    events = []

    def capture(message, *, extra):
        events.append((message, extra))

    monkeypatch.setattr("app.main.logger.info", capture)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert [extra["event"] for _, extra in events] == [
        "api_request_received",
        "api_request_completed",
    ]
    assert events[0][1]["method"] == "GET"
    assert events[0][1]["path"] == "/health"
    assert events[1][1]["status_code"] == 200
    assert "body" not in events[0][1]
    assert "query" not in events[0][1]
