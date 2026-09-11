"""Best-effort SMTP notifications for human escalation handoffs."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from src.config import get_environment_settings


class EmailNotificationError(RuntimeError):
    """Raised when an escalation email cannot be built or sent."""


def send_escalation_email(*, subject: str, body: str, cc: str | None = None) -> None:
    """Send a plain-text escalation notification to the configured admin.

    Raises `EmailNotificationError` if SMTP is not configured or the send
    fails; callers decide whether that should block the caller's own flow.
    """

    settings = get_environment_settings()
    admin_email = settings.admin_email.strip()
    smtp_host = settings.smtp_host.strip()
    if not admin_email or not smtp_host:
        raise EmailNotificationError("Email notifications are not configured.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from_address.strip() or admin_email
    message["To"] = admin_email
    if cc:
        message["Cc"] = cc
    message.set_content(body)

    recipients = [admin_email, *([cc] if cc else [])]

    try:
        with smtplib.SMTP(smtp_host, settings.smtp_port, timeout=10) as client:
            if settings.smtp_use_tls:
                client.starttls()
            username = settings.smtp_username.strip()
            if username:
                client.login(username, settings.smtp_password.get_secret_value())
            client.send_message(message, to_addrs=recipients)
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailNotificationError("Escalation email could not be sent.") from exc
