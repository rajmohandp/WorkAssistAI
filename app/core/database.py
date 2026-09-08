"""Secure SQLAlchemy connection management for AWS RDS MySQL."""

from __future__ import annotations

import logging
import ssl
from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import URL, Engine, create_engine, make_url, text
from sqlalchemy.exc import ArgumentError, DBAPIError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import (
    DatabaseSettings,
    get_environment_settings,
)

logger = logging.getLogger(__name__)
POOL_SIZE = 3
MAX_OVERFLOW = 2
POOL_RECYCLE_SECONDS = 900
POOL_TIMEOUT_SECONDS = 10
CONNECT_TIMEOUT_SECONDS = 10


class DatabaseConnectionError(RuntimeError):
    """Raised when the application cannot safely reach the database."""


def build_database_url(settings: DatabaseSettings) -> URL:
    """Build a credential-safe SQLAlchemy URL without manual interpolation."""

    return URL.create(
        drivername="mysql+pymysql",
        username=settings.user,
        password=settings.password,
        host=settings.host,
        port=settings.port,
        database=settings.name,
        query={"charset": "utf8mb4"},
    )


def create_database_engine() -> Engine:
    """Create a conservative pool from the configured SQLAlchemy URL."""

    environment = get_environment_settings()
    database_url = environment.database_url.get_secret_value().strip()
    if not database_url:
        raise DatabaseConnectionError("DATABASE_URL is required.")
    try:
        url = make_url(database_url)
    except ArgumentError as exc:
        raise DatabaseConnectionError("DATABASE_URL is invalid.") from exc
    if url.drivername != "mysql+pymysql":
        raise DatabaseConnectionError(
            "DATABASE_URL must use the mysql+pymysql driver."
        )
    connect_args = {"connect_timeout": CONNECT_TIMEOUT_SECONDS}
    if environment.db_ssl_ca_path.strip():
        connect_args["ssl"] = {
            "ca": environment.db_ssl_ca_path.strip(),
            "check_hostname": True,
            "verify_mode": ssl.CERT_REQUIRED,
        }
    return create_engine(
        url,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        pool_timeout=POOL_TIMEOUT_SECONDS,
        connect_args=connect_args,
        echo=False,
        hide_parameters=True,
    )


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine and connection pool."""

    logger.info(
        "Initializing database connection pool",
        extra={"operation": "database", "event": "database_pool_started"},
    )
    engine = create_database_engine()
    logger.info(
        "Database connection pool initialized",
        extra={"operation": "database", "event": "database_pool_ready"},
    )
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide factory for request-scoped sessions."""

    return sessionmaker(
        bind=get_database_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency providing one database session per request."""

    session = get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_health(session: Session | None = None) -> bool:
    """Execute ``SELECT 1`` and return whether MySQL responds correctly."""

    owns_session = session is None
    resolved_session = session or get_session_factory()()
    try:
        healthy = resolved_session.execute(text("SELECT 1")).scalar_one() == 1
    except SQLAlchemyError as exc:
        discard_failed_session(resolved_session, exc)
        logger.error(
            "Database health check failed",
            extra={
                "operation": "database",
                "event": "database_health_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise DatabaseConnectionError("The database is unavailable.") from exc
    finally:
        if owns_session:
            resolved_session.close()

    logger.info(
        "Database health check completed",
        extra={
            "operation": "database",
            "event": "database_health_completed",
            "healthy": healthy,
        },
    )
    return healthy


def discard_failed_session(session: Session, exc: SQLAlchemyError) -> None:
    """Reset a failed transaction and discard an invalid pooled connection."""

    try:
        session.rollback()
    except SQLAlchemyError:
        pass
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        try:
            session.invalidate()
        except SQLAlchemyError:
            pass


def clear_database_dependencies() -> None:
    """Dispose cached connections for tests or deliberate configuration reloads."""

    if get_database_engine.cache_info().currsize:
        get_database_engine().dispose()
    get_session_factory.cache_clear()
    get_database_engine.cache_clear()
