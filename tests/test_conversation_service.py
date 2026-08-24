"""Tests for lightweight non-RAG conversation handling."""

import pytest

from app.services.conversation_service import (
    ACKNOWLEDGEMENT_RESPONSE,
    CAPABILITIES_RESPONSE,
    GOODBYE_RESPONSE,
    GREETING_RESPONSE,
    HELP_RESPONSE,
    THANKS_RESPONSE,
    get_conversational_response,
)


@pytest.mark.parametrize(
    "message",
    [
        "Hi",
        " hello! ",
        "HEY!!!",
        "Good morning",
        "Good afternoon.",
        "Good evening",
        "Hi DocuVerse",
        "Hello, DocuVerse!",
    ],
)
def test_greetings_are_handled_without_rag(message):
    assert get_conversational_response(message) == GREETING_RESPONSE


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Thanks!", THANKS_RESPONSE),
        ("Thank you", THANKS_RESPONSE),
        ("Okay", ACKNOWLEDGEMENT_RESPONSE),
        ("Got it.", ACKNOWLEDGEMENT_RESPONSE),
        ("Bye", GOODBYE_RESPONSE),
        ("Goodbye!", GOODBYE_RESPONSE),
        ("What can you do?", CAPABILITIES_RESPONSE),
        ("Help", HELP_RESPONSE),
    ],
)
def test_basic_conversation_is_handled_without_rag(message, expected):
    assert get_conversational_response(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "Hi, what is our vacation policy?",
        "Hello, how many sick days are available?",
        "What does the employee handbook say about remote work?",
        "Explain the company's training reimbursement policy.",
    ],
)
def test_document_questions_are_not_misclassified(message):
    assert get_conversational_response(message) is None


def test_empty_message_is_not_classified_as_conversation():
    assert get_conversational_response("  ...  ") is None
