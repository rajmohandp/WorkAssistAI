"""Signed bearer-token authentication enforced by the FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from auth import UserRecord, get_user
from src.config import get_environment_settings

TOKEN_SALT = "workassist-access-token-v1"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    employee_id: str
    role: Literal["admin", "user"]


def _token_configuration() -> tuple[URLSafeTimedSerializer, int]:
    settings = get_environment_settings()
    secret = settings.jwt_secret_key.get_secret_value()
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET_KEY must contain at least 32 characters.")
    if settings.auth_token_ttl_seconds <= 0:
        raise RuntimeError("AUTH_TOKEN_TTL_SECONDS must be greater than zero.")
    return URLSafeTimedSerializer(secret, salt=TOKEN_SALT), settings.auth_token_ttl_seconds


def create_access_token(user: UserRecord) -> tuple[str, int]:
    serializer, ttl = _token_configuration()
    token = serializer.dumps(
        {
            "sub": user.username,
            "employee_id": user.employee_id,
            "role": user.role,
        }
    )
    return token, ttl


def decode_access_token(token: str) -> AuthenticatedUser:
    serializer, ttl = _token_configuration()
    try:
        payload = serializer.loads(token, max_age=ttl)
        username = payload["sub"]
        employee_id = payload["employee_id"]
        role = payload["role"]
    except (BadSignature, SignatureExpired, KeyError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    user = get_user(username) if isinstance(username, str) else None
    if (
        user is None
        or employee_id != user.employee_id
        or role != user.role
        or role not in {"admin", "user"}
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthenticatedUser(user.username, user.employee_id, user.role)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> AuthenticatedUser:
    return decode_access_token(token)


async def require_admin(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> AuthenticatedUser:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    return user
