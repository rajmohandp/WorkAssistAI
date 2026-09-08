"""Tests for secure, pooled database configuration."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from app.core.database import (
    DatabaseConnectionError,
    build_database_url,
    check_database_health,
    create_database_engine,
    get_db_session,
)
from src.config import DatabaseSettings, get_database_settings


def database_settings(password: str = "p@ss/word") -> DatabaseSettings:
    return DatabaseSettings(
        host="docuverse.cluster.us-west-2.rds.amazonaws.com",
        port=3306,
        name="docuverse",
        user="docuverse_readonly",
        password=password,
    )


def test_database_settings_load_requested_environment_variables(monkeypatch):
    monkeypatch.setenv("DB_HOST", "db.example")
    monkeypatch.setenv("DB_PORT", "3307")
    monkeypatch.setenv("DB_NAME", "docuverse")
    monkeypatch.setenv("DB_USER", "readonly")
    monkeypatch.setenv("DB_PASSWORD", "database-secret")

    settings = get_database_settings()

    assert settings.host == "db.example"
    assert settings.port == 3307
    assert settings.name == "docuverse"
    assert settings.user == "readonly"
    assert settings.password == "database-secret"
    assert "database-secret" not in repr(settings)


def test_database_url_escapes_and_hides_password():
    url = build_database_url(database_settings())

    assert url.drivername == "mysql+pymysql"
    assert url.password == "p@ss/word"
    assert "p@ss/word" not in str(url)
    assert "***" in str(url)


def test_engine_uses_pool_pre_ping_and_verified_tls(monkeypatch):
    create_engine = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("app.core.database.create_engine", create_engine)
    monkeypatch.setenv(
        "DATABASE_URL", "mysql+pymysql://readonly:secret@db.example/workassist"
    )
    monkeypatch.setenv("DB_SSL_CA_PATH", "/run/secrets/global-bundle.pem")

    create_database_engine()

    kwargs = create_engine.call_args.kwargs
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_size"] == 3
    assert kwargs["max_overflow"] == 2
    assert kwargs["pool_recycle"] == 900
    assert kwargs["pool_timeout"] == 10
    assert kwargs["echo"] is False
    assert kwargs["hide_parameters"] is True
    tls_options = kwargs["connect_args"]["ssl"]
    assert tls_options["ca"] == "/run/secrets/global-bundle.pem"
    assert tls_options["verify_mode"] is not False
    assert tls_options["check_hostname"] is True


def test_database_session_dependency_closes_session(monkeypatch):
    session = MagicMock()
    factory = MagicMock(return_value=session)
    monkeypatch.setattr("app.core.database.get_session_factory", lambda: factory)
    dependency = get_db_session()

    assert next(dependency) is session
    with pytest.raises(StopIteration):
        next(dependency)
    session.close.assert_called_once_with()


def test_database_health_executes_select_one():
    session = MagicMock()
    session.execute.return_value.scalar_one.return_value = 1

    assert check_database_health(session) is True
    statement = session.execute.call_args.args[0]
    assert str(statement) == "SELECT 1"


def test_database_health_wraps_provider_errors():
    session = MagicMock()
    session.execute.side_effect = OperationalError("SELECT 1", {}, Exception())

    with pytest.raises(DatabaseConnectionError, match="database is unavailable"):
        check_database_health(session)

    session.rollback.assert_called_once_with()


def test_database_health_invalidates_expired_connection():
    session = MagicMock()
    session.execute.side_effect = OperationalError(
        "SELECT 1",
        {},
        Exception(),
        connection_invalidated=True,
    )

    with pytest.raises(DatabaseConnectionError):
        check_database_health(session)

    session.rollback.assert_called_once_with()
    session.invalidate.assert_called_once_with()
