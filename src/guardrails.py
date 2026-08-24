"""Deterministic safety boundaries for DocuVerse questions and document text."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
MAX_QUESTION_LENGTH = 2000
SECURITY_REFUSAL_MESSAGE = (
    "I cannot provide credentials, environment variables, internal prompts, "
    "or other application secrets."
)

_UNSAFE_REQUEST_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\b(?:reveal|show|print|return|repeat)\b.{0,50}\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(
        r"\b(?:reveal|show|print|return|extract|list)\b.{0,60}"
        r"\b(?:aws\s+credentials?|pinecone\s+credentials?|api\s+keys?|"
        r"environment\s+variables?|secrets?)\b",
        re.IGNORECASE,
    ),
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        r"\b(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|PINECONE_API_KEY|"
        r"OPENAI_API_KEY|GROQ_API_KEY|LLM_API_KEY)\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
    re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*[^\s,;]+"),
)


class InputValidationError(ValueError):
    """Raised when a question violates a deterministic input boundary."""


def validate_question(question: str) -> str:
    """Validate question size and control characters without logging its content."""

    if not isinstance(question, str) or not question.strip():
        raise InputValidationError("Enter a question before searching.")
    normalized = question.strip()
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise InputValidationError(
            f"Questions must be {MAX_QUESTION_LENGTH} characters or fewer."
        )
    if any(ord(character) < 32 and character not in "\n\r\t" for character in normalized):
        raise InputValidationError("The question contains unsupported control characters.")
    return normalized


def is_sensitive_request(question: str) -> bool:
    """Identify direct prompt-injection and secret-exfiltration requests."""

    return any(pattern.search(question) for pattern in _UNSAFE_REQUEST_PATTERNS) or any(
        pattern.search(question) for pattern in _SECRET_PATTERNS
    )


def redact_sensitive_text(text: str) -> str:
    """Remove known credentials, environment values, and embedded instructions."""

    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: (
                f"{match.group(1)}=[REDACTED]"
                if match.lastindex
                else "[REDACTED]"
            ),
            redacted,
        )

    safe_lines = []
    for line in redacted.splitlines():
        if any(pattern.search(line) for pattern in _UNSAFE_REQUEST_PATTERNS):
            safe_lines.append("[UNTRUSTED INSTRUCTION REMOVED]")
        else:
            safe_lines.append(line)
    return "\n".join(safe_lines)
