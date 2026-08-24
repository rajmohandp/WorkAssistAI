"""Lightweight handling for messages that do not require document retrieval."""

from __future__ import annotations

import re

GREETING_RESPONSE = (
    "Hello! 👋 Welcome to DocuVerse. I can help you find information from your "
    "organization's documents. What would you like to know?"
)
THANKS_RESPONSE = (
    "You're welcome! Let me know if you'd like to search for anything else in "
    "your documents."
)
ACKNOWLEDGEMENT_RESPONSE = (
    "Great! Ask me another question whenever you'd like to search your documents."
)
CAPABILITIES_RESPONSE = (
    "I can search your organization's documents and answer questions based on "
    "the information available in them. Ask me a question about the documents "
    "to get started."
)
HELP_RESPONSE = (
    "Ask me a question about the information stored in your organization's "
    "documents. I'll search the relevant content and provide a grounded answer."
)
GOODBYE_RESPONSE = (
    "Goodbye! 👋 Come back anytime you need help searching your documents."
)

_NON_WORD_PATTERN = re.compile(r"[^\w']+", re.UNICODE)
_GREETING_PATTERN = re.compile(
    r"(?:hi|hello|hey|good morning|good afternoon|good evening)"
    r"(?: docuverse)?",
    re.IGNORECASE,
)


def _normalize_message(message: str) -> str:
    """Normalize case, surrounding punctuation, and repeated whitespace."""

    normalized = _NON_WORD_PATTERN.sub(" ", message.casefold()).strip()
    return " ".join(normalized.split())


def get_conversational_response(message: str) -> str | None:
    """Return a static response, or None when document RAG should handle it."""

    normalized = _normalize_message(message)
    if not normalized:
        return None
    if _GREETING_PATTERN.fullmatch(normalized):
        return GREETING_RESPONSE
    if normalized in {"thanks", "thank you"}:
        return THANKS_RESPONSE
    if normalized in {"okay", "ok", "got it"}:
        return ACKNOWLEDGEMENT_RESPONSE
    if normalized in {"bye", "goodbye"}:
        return GOODBYE_RESPONSE
    if normalized == "what can you do":
        return CAPABILITIES_RESPONSE
    if normalized == "help":
        return HELP_RESPONSE
    return None
