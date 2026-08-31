"""Authentication endpoints for signed backend-issued access tokens."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import CurrentUserResponse, LoginRequest, TokenResponse
from app.core.security import AuthenticatedUser, create_access_token, get_current_user
from auth import authenticate_user

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    user = authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, ttl = create_access_token(user)
    return TokenResponse(access_token=token, expires_in=ttl)


@router.get("/me", response_model=CurrentUserResponse)
async def me(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> CurrentUserResponse:
    return CurrentUserResponse(
        username=user.username,
        employee_id=user.employee_id,
        role=user.role,
    )
