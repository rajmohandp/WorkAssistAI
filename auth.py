"""Prototype authentication and role-based authorization for DocuVerse."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

from src.config import get_environment_settings

Role = Literal["admin", "user"]
_PASSWORD_ITERATIONS = 310_000


@dataclass(frozen=True)
class UserRecord:
    """Centralized prototype user configuration with a hashed password."""

    username: str
    employee_id: str
    password_salt: str
    password_hash: str
    role: Role


def _configured_users() -> dict[str, UserRecord]:
    """Load password hashes and user identities from protected configuration."""

    raw_users = get_environment_settings().auth_users_json.get_secret_value().strip()
    if not raw_users:
        raise RuntimeError("Missing required environment variable: AUTH_USERS_JSON")
    try:
        records = json.loads(raw_users)
        users = {
            str(record["username"]).strip().casefold(): UserRecord(
                username=str(record["username"]).strip().casefold(),
                employee_id=str(record["employee_id"]).strip(),
                password_salt=str(record["password_salt"]),
                password_hash=str(record["password_hash"]),
                role=record["role"],
            )
            for record in records
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("AUTH_USERS_JSON is not valid user configuration.") from exc
    if not users or any(user.role not in {"admin", "user"} for user in users.values()):
        raise RuntimeError("AUTH_USERS_JSON must contain valid admin or user records.")
    return users


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PASSWORD_ITERATIONS,
    ).hex()


def authenticate_user(username: str, password: str) -> UserRecord | None:
    """Validate credentials and return the centrally configured user record."""

    normalized_username = username.strip().casefold()
    users = _configured_users()
    user = users.get(normalized_username)
    salt = user.password_salt if user else "invalid-user-salt"
    expected_hash = user.password_hash if user else "0" * 64
    supplied_hash = _hash_password(password, salt)
    if user and hmac.compare_digest(supplied_hash, expected_hash):
        return user
    return None


def get_user(username: str) -> UserRecord | None:
    """Return a centrally configured user by normalized username."""

    return _configured_users().get(username.strip().casefold())


def login(
    session: MutableMapping[str, Any],
    username: str,
    password: str,
) -> bool:
    """Authenticate and establish the three required session attributes."""

    user = authenticate_user(username, password)
    if user is None:
        logout(session)
        return False
    session["authenticated"] = True
    session["username"] = user.username
    session["role"] = user.role
    return True


def logout(session: MutableMapping[str, Any]) -> None:
    """Remove all authentication and authorization state."""

    for key in ("authenticated", "username", "role", "access_token"):
        session.pop(key, None)


def is_authenticated(session: MutableMapping[str, Any]) -> bool:
    """Return whether the session has a complete authenticated identity."""

    return bool(
        session.get("authenticated")
        and session.get("username")
        and session.get("role") in {"admin", "user"}
    )


def has_role(session: MutableMapping[str, Any], required_role: Role) -> bool:
    """Authorize a session by role rather than username."""

    return is_authenticated(session) and session.get("role") == required_role


def is_admin(session: MutableMapping[str, Any]) -> bool:
    """Return whether the authenticated session has the admin role."""

    return has_role(session, "admin")
