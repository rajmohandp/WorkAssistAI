"""Unit tests for best-effort SMTP escalation notifications."""

from unittest.mock import MagicMock

import pytest

from app.services.email_service import EmailNotificationError, send_escalation_email


def _set_smtp_env(monkeypatch, **overrides):
    values = {
        "ADMIN_EMAIL": "admin@example.com",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "",
        "SMTP_PASSWORD": "",
        "SMTP_FROM_ADDRESS": "no-reply@example.com",
        "SMTP_USE_TLS": "true",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_raises_when_not_configured(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAIL", "")
    monkeypatch.setenv("SMTP_HOST", "")

    with pytest.raises(EmailNotificationError):
        send_escalation_email(subject="s", body="b")


def test_sends_with_admin_to_and_employee_cc(monkeypatch):
    _set_smtp_env(monkeypatch)
    smtp_client = MagicMock()
    smtp_client.__enter__.return_value = smtp_client
    mock_smtp = MagicMock(return_value=smtp_client)
    monkeypatch.setattr("smtplib.SMTP", mock_smtp)

    send_escalation_email(
        subject="Escalation WA-123",
        body="Employee needs help",
        cc="employee@example.com",
    )

    mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=10)
    smtp_client.starttls.assert_called_once()
    smtp_client.login.assert_not_called()
    sent_message = smtp_client.send_message.call_args.args[0]
    assert sent_message["To"] == "admin@example.com"
    assert sent_message["Cc"] == "employee@example.com"
    assert sent_message["Subject"] == "Escalation WA-123"
    assert smtp_client.send_message.call_args.kwargs["to_addrs"] == [
        "admin@example.com",
        "employee@example.com",
    ]


def test_sends_without_cc_when_employee_email_unknown(monkeypatch):
    _set_smtp_env(monkeypatch)
    smtp_client = MagicMock()
    smtp_client.__enter__.return_value = smtp_client
    monkeypatch.setattr("smtplib.SMTP", MagicMock(return_value=smtp_client))

    send_escalation_email(subject="s", body="b", cc=None)

    sent_message = smtp_client.send_message.call_args.args[0]
    assert "Cc" not in sent_message
    assert smtp_client.send_message.call_args.kwargs["to_addrs"] == [
        "admin@example.com"
    ]


def test_logs_in_when_username_configured(monkeypatch):
    _set_smtp_env(monkeypatch, SMTP_USERNAME="smtp-user", SMTP_PASSWORD="secret")
    smtp_client = MagicMock()
    smtp_client.__enter__.return_value = smtp_client
    monkeypatch.setattr("smtplib.SMTP", MagicMock(return_value=smtp_client))

    send_escalation_email(subject="s", body="b")

    smtp_client.login.assert_called_once_with("smtp-user", "secret")


def test_wraps_smtp_failure(monkeypatch):
    _set_smtp_env(monkeypatch)
    monkeypatch.setattr(
        "smtplib.SMTP", MagicMock(side_effect=OSError("connection refused"))
    )

    with pytest.raises(EmailNotificationError):
        send_escalation_email(subject="s", body="b")
