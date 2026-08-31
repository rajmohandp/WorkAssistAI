"""Secure SQLAlchemy connection management for AWS RDS MySQL."""

from __future__ import annotations

import logging
import ssl
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import URL, Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import DatabaseSettings, get_database_settings

logger = logging.getLogger(__name__)
POOL_SIZE = 5
MAX_OVERFLOW = 10
POOL_RECYCLE_SECONDS = 1800
CONNECT_TIMEOUT_SECONDS = 10
RDS_CA_BUNDLE_PATH = Path(r"C:\Users\rmohan\certs\global-bundle.pem")


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


def create_database_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Create a pooled, TLS-verified MySQL engine."""

    resolved_settings = settings or get_database_settings()
    return create_engine(
        build_database_url(resolved_settings),
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        connect_args={
            "connect_timeout": CONNECT_TIMEOUT_SECONDS,
            "ssl": {
                "ca": str(RDS_CA_BUNDLE_PATH),
                "check_hostname": True,
                "verify_mode": ssl.CERT_REQUIRED,
            },
        },
        echo=False,
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


def clear_database_dependencies() -> None:
    """Dispose cached connections for tests or deliberate configuration reloads."""

    if get_database_engine.cache_info().currsize:
        get_database_engine().dispose()
    get_session_factory.cache_clear()
    get_database_engine.cache_clear()
