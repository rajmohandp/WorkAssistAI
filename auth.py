"""Prototype authentication and role-based authorization for DocuVerse."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["admin", "user"]
_PASSWORD_ITERATIONS = 310_000


@dataclass(frozen=True)
class UserRecord:
    """Centralized prototype user configuration with a hashed password."""

    username: str
    password_salt: str
    password_hash: str
    role: Role


_SHARED_PROTOTYPE_SALT = "docuverse-prototype-v1"
_SHARED_PROTOTYPE_PASSWORD_HASH = (
    "cf74315433aa7ca9fc5eea7f5a68db891e4ac7704b0c4ce8104f1e490d8aba48"
)
_USERS: dict[str, UserRecord] = {
    "admin": UserRecord(
        username="admin",
        password_salt=_SHARED_PROTOTYPE_SALT,
        password_hash=_SHARED_PROTOTYPE_PASSWORD_HASH,
        role="admin",
    ),
    "user01": UserRecord(
        username="user01",
        password_salt=_SHARED_PROTOTYPE_SALT,
        password_hash=_SHARED_PROTOTYPE_PASSWORD_HASH,
        role="user",
    ),
}


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
    user = _USERS.get(normalized_username)
    salt = user.password_salt if user else _SHARED_PROTOTYPE_SALT
    expected_hash = (
        user.password_hash if user else _SHARED_PROTOTYPE_PASSWORD_HASH
    )
    supplied_hash = _hash_password(password, salt)
    if user and hmac.compare_digest(supplied_hash, expected_hash):
        return user
    return None


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

    for key in ("authenticated", "username", "role"):
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
