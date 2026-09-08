"""Tests for centralized environment-backed configuration."""

from src.config import EnvironmentSettings


def test_requested_environment_variable_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-secret")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "docuverse-index")
    monkeypatch.setenv("AWS_PROFILE", "workassist-local")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("S3_BUCKET_NAME", "documents-bucket")
    monkeypatch.setenv("BACKEND_URL", "http://api.example:8000")
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://user:secret@db/app")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://one.example,https://two.example")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PINECONE_CLOUD", "aws")
    monkeypatch.setenv("PINECONE_REGION", "us-west-2")

    settings = EnvironmentSettings()

    assert settings.openai_api_key.get_secret_value() == "openai-secret"
    assert settings.pinecone_api_key.get_secret_value() == "pinecone-secret"
    assert settings.pinecone_index_name == "docuverse-index"
    assert settings.aws_profile == "workassist-local"
    assert settings.aws_region == "us-west-2"
    assert settings.s3_bucket_name == "documents-bucket"
    assert settings.fastapi_url == "http://api.example:8000"
    assert settings.backend_url == "http://api.example:8000"
    assert settings.database_url.get_secret_value().startswith("mysql+pymysql://")
    assert settings.jwt_secret_key.get_secret_value() == "x" * 32
    assert settings.allowed_origins == ["https://one.example", "https://two.example"]
    assert settings.app_env == "production"
    assert settings.pinecone_cloud == "aws"
    assert settings.pinecone_region == "us-west-2"


def test_backend_validation_reports_all_missing_required_variables(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "S3_BUCKET_NAME",
        "AWS_S3_BUCKET",
        "DATABASE_URL",
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_SSL_CA_PATH",
        "JWT_SECRET_KEY",
        "AUTH_TOKEN_SECRET",
        "AUTH_USERS_JSON",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = EnvironmentSettings(_env_file=None)

    try:
        settings.validate_backend_startup()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing configuration to fail validation")
    assert "OPENAI_API_KEY" in message
    assert "DATABASE_URL" in message
    assert "JWT_SECRET_KEY" in message


def test_legacy_environment_aliases_remain_supported(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("BACKEND_URL", raising=False)
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
    monkeypatch.setenv("DB_PASSWORD", "database-secret")

    representation = repr(EnvironmentSettings(_env_file=None))

    assert "must-not-appear" not in representation
    assert "database-secret" not in representation
    assert "**********" in representation


def test_production_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")

    settings = EnvironmentSettings(_env_file=None)

    try:
        settings.validate_cors_configuration()
    except RuntimeError as exc:
        assert "ALLOWED_ORIGINS" in str(exc)
    else:
        raise AssertionError("Production wildcard CORS should be rejected")


def test_non_production_allows_wildcard_cors(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")

    EnvironmentSettings(_env_file=None).validate_cors_configuration()


def test_backend_url_defaults_for_local_development(monkeypatch):
    for name in ("BACKEND_URL", "FASTAPI_URL", "DOCUVERSE_API_URL"):
        monkeypatch.delenv(name, raising=False)

    settings = EnvironmentSettings(_env_file=None)

    assert settings.backend_url == "http://localhost:8000"


def test_backend_url_supports_docker_service_name(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://backend:8000")

    settings = EnvironmentSettings(_env_file=None)

    assert settings.backend_url == "http://backend:8000"
