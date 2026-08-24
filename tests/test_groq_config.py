"""Tests for Groq provider configuration."""

import pytest

from src.config import DEFAULT_GROQ_MODEL, GroqSettings
from src.rag_chain import create_chat_model


def test_groq_settings_hide_api_key():
    settings = GroqSettings(api_key="secret-value")

    assert "secret-value" not in repr(settings)
    assert settings.model == DEFAULT_GROQ_MODEL


def test_groq_api_key_is_required():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqSettings(api_key="")


def test_creates_configured_groq_chat_model():
    model = create_chat_model(
        GroqSettings(api_key="test-key", model="openai/gpt-oss-120b")
    )

    assert model.model_name == "openai/gpt-oss-120b"
    assert model.temperature <= 1e-8
