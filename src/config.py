"""Central application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_CHAT_PROVIDER = "openai"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = 1024
DEFAULT_PINECONE_BATCH_SIZE = 100
DEFAULT_RETRIEVER_TOP_K = 5
DEFAULT_RETRIEVAL_MIN_SCORE = 0.3


class EnvironmentSettings(BaseSettings):
    """Central environment-backed configuration for DocuVerse."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr = SecretStr("")
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_name: str = ""
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")
    aws_region: str = Field(
        default="",
        validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION"),
    )
    s3_bucket_name: str = Field(
        default="",
        validation_alias=AliasChoices("S3_BUCKET_NAME", "AWS_S3_BUCKET"),
    )
    fastapi_url: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("FASTAPI_URL", "DOCUVERSE_API_URL"),
    )
    docuverse_api_prefix: str = "/api/v1"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = DEFAULT_GROQ_MODEL
    chat_provider: str = DEFAULT_CHAT_PROVIDER
    openai_chat_model: str = DEFAULT_OPENAI_CHAT_MODEL
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    openai_embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    openai_embedding_dimensions: int = DEFAULT_OPENAI_EMBEDDING_DIMENSIONS
    pinecone_namespace: str = ""
    pinecone_batch_size: int = DEFAULT_PINECONE_BATCH_SIZE
    retriever_top_k: int = DEFAULT_RETRIEVER_TOP_K
    retrieval_min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE
    log_level: str = "INFO"
    rag_evaluation_enabled: bool = False
    db_host: str = ""
    db_port: int = 0
    db_name: str = ""
    db_user: str = ""
    db_password: SecretStr = SecretStr("")
    auth_token_secret: SecretStr = SecretStr("")
    auth_token_ttl_seconds: int = 1800


def get_environment_settings() -> EnvironmentSettings:
    """Load a fresh validated view of `.env` and process environment values."""

    return EnvironmentSettings()


@dataclass(frozen=True)
class DatabaseSettings:
    """Validated, secret-safe AWS RDS MySQL configuration."""

    host: str
    port: int
    name: str
    user: str
    password: str

    def __post_init__(self) -> None:
        required_values = {
            "DB_HOST": self.host,
            "DB_NAME": self.name,
            "DB_USER": self.user,
            "DB_PASSWORD": self.password,
        }
        missing = [name for name, value in required_values.items() if not value]
        if missing:
            raise ValueError(f"Missing database configuration: {', '.join(missing)}")
        if not 1 <= self.port <= 65535:
            raise ValueError("DB_PORT must be between 1 and 65535.")

    def __repr__(self) -> str:
        return (
            f"DatabaseSettings(host={self.host!r}, port={self.port!r}, "
            f"name={self.name!r}, user={self.user!r}, password='********')"
        )


def get_database_settings() -> DatabaseSettings:
    """Load database configuration without exposing the password."""

    environment = get_environment_settings()
    return DatabaseSettings(
        host=environment.db_host.strip(),
        port=environment.db_port,
        name=environment.db_name.strip(),
        user=environment.db_user.strip(),
        password=environment.db_password.get_secret_value(),
    )


@dataclass(frozen=True)
class Settings:
    """Validated DocuVerse application settings."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


def get_settings() -> Settings:
    """Load and validate settings from the environment."""

    environment = get_environment_settings()
    return Settings(
        chunk_size=environment.chunk_size,
        chunk_overlap=environment.chunk_overlap,
    )


@dataclass(frozen=True)
class GroqSettings:
    """Validated Groq provider configuration."""

    api_key: str
    model: str = DEFAULT_GROQ_MODEL

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required to use the Groq model.")
        if not self.model:
            raise ValueError("GROQ_MODEL cannot be empty.")

    def __repr__(self) -> str:
        """Prevent the API key from appearing in diagnostics or logs."""

        return f"GroqSettings(api_key='********', model={self.model!r})"


def get_groq_settings() -> GroqSettings:
    """Load Groq configuration without logging or displaying the API key."""

    environment = get_environment_settings()
    return GroqSettings(
        api_key=environment.groq_api_key.get_secret_value().strip(),
        model=environment.groq_model.strip(),
    )


@dataclass(frozen=True)
class ChatSettings:
    """Provider-neutral chat model configuration."""

    provider: str
    model: str
    api_key: str

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "groq"}:
            raise ValueError(f"Unsupported chat provider: {self.provider}")
        if not self.model:
            raise ValueError("The configured chat model cannot be empty.")
        if not self.api_key:
            raise ValueError(
                f"{self.provider.upper()}_API_KEY is required for chat generation."
            )

    def __repr__(self) -> str:
        """Prevent the provider API key from appearing in diagnostics."""

        return (
            f"ChatSettings(provider={self.provider!r}, model={self.model!r}, "
            "api_key='********')"
        )


def get_chat_settings() -> ChatSettings:
    """Load the selected chat provider and model from the environment."""

    environment = get_environment_settings()
    provider = environment.chat_provider.strip().lower()
    if provider == "openai":
        return ChatSettings(
            provider=provider,
            model=environment.openai_chat_model.strip(),
            api_key=environment.openai_api_key.get_secret_value().strip(),
        )
    if provider == "groq":
        return ChatSettings(
            provider=provider,
            model=environment.groq_model.strip(),
            api_key=environment.groq_api_key.get_secret_value().strip(),
        )
    raise ValueError(f"Unsupported chat provider: {provider}")


@dataclass(frozen=True)
class EmbeddingSettings:
    """Provider-neutral embedding configuration."""

    provider: str
    model: str
    api_key: str
    dimensions: int = DEFAULT_OPENAI_EMBEDDING_DIMENSIONS

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("EMBEDDING_PROVIDER cannot be empty.")
        if not self.model:
            raise ValueError("OPENAI_EMBEDDING_MODEL cannot be empty.")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        if self.dimensions <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS must be greater than zero.")

    def __repr__(self) -> str:
        """Prevent the API key from appearing in diagnostics or logs."""

        return (
            f"EmbeddingSettings(provider={self.provider!r}, model={self.model!r}, "
            f"dimensions={self.dimensions!r}, api_key='********')"
        )


def get_embedding_settings() -> EmbeddingSettings:
    """Load the selected embedding provider configuration."""

    environment = get_environment_settings()
    provider = environment.embedding_provider.strip().lower()
    if provider != "openai":
        raise ValueError(f"Unsupported embedding provider: {provider}")

    return EmbeddingSettings(
        provider=provider,
        model=environment.openai_embedding_model.strip(),
        api_key=environment.openai_api_key.get_secret_value().strip(),
        dimensions=environment.openai_embedding_dimensions,
    )


@dataclass(frozen=True)
class PineconeSettings:
    """Validated Pinecone vector-store configuration."""

    api_key: str
    index_name: str
    namespace: str = ""
    batch_size: int = DEFAULT_PINECONE_BATCH_SIZE

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required.")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is required.")
        if self.batch_size <= 0:
            raise ValueError("PINECONE_BATCH_SIZE must be greater than zero.")

    def __repr__(self) -> str:
        """Prevent the API key from appearing in diagnostics or logs."""

        return (
            "PineconeSettings(api_key='********', "
            f"index_name={self.index_name!r}, namespace={self.namespace!r}, "
            f"batch_size={self.batch_size!r})"
        )


def get_pinecone_settings() -> PineconeSettings:
    """Load Pinecone configuration from environment variables."""

    environment = get_environment_settings()
    return PineconeSettings(
        api_key=environment.pinecone_api_key.get_secret_value().strip(),
        index_name=environment.pinecone_index_name.strip(),
        namespace=environment.pinecone_namespace.strip(),
        batch_size=environment.pinecone_batch_size,
    )


@dataclass(frozen=True)
class RetrievalSettings:
    """Semantic retrieval configuration."""

    top_k: int = DEFAULT_RETRIEVER_TOP_K
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("RETRIEVER_TOP_K must be greater than zero.")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("RETRIEVAL_MIN_SCORE must be between -1 and 1.")


def get_retrieval_settings() -> RetrievalSettings:
    """Load semantic retrieval settings from the environment."""

    environment = get_environment_settings()
    return RetrievalSettings(
        top_k=environment.retriever_top_k,
        min_score=environment.retrieval_min_score,
    )
