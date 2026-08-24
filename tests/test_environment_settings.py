"""Tests for centralized environment-backed configuration."""

from src.config import EnvironmentSettings


def test_requested_environment_variable_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-secret")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "docuverse-index")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("S3_BUCKET_NAME", "documents-bucket")
    monkeypatch.setenv("FASTAPI_URL", "http://api.example:8000")

    settings = EnvironmentSettings()

    assert settings.openai_api_key.get_secret_value() == "openai-secret"
    assert settings.pinecone_api_key.get_secret_value() == "pinecone-secret"
    assert settings.pinecone_index_name == "docuverse-index"
    assert settings.aws_access_key_id.get_secret_value() == "aws-access"
    assert settings.aws_secret_access_key.get_secret_value() == "aws-secret"
    assert settings.aws_region == "us-west-2"
    assert settings.s3_bucket_name == "documents-bucket"
    assert settings.fastapi_url == "http://api.example:8000"


def test_legacy_environment_aliases_remain_supported(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("FASTAPI_URL", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.setenv("AWS_S3_BUCKET", "legacy-bucket")
    monkeypatch.setenv("DOCUVERSE_API_URL", "http://legacy-api:8000")

    settings = EnvironmentSettings(_env_file=None)

    assert settings.aws_region == "us-east-2"
    assert settings.s3_bucket_name == "legacy-bucket"
    assert settings.fastapi_url == "http://legacy-api:8000"


def test_secret_values_are_redacted_in_settings_representation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-appear")

    representation = repr(EnvironmentSettings(_env_file=None))

    assert "must-not-appear" not in representation
    assert "**********" in representation
