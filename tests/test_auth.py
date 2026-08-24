"""Authentication and RBAC tests for the DocuVerse prototype users."""

from auth import (
    authenticate_user,
    has_role,
    is_admin,
    is_authenticated,
    login,
    logout,
)


def test_admin_login_assigns_admin_role():
    session = {}

    assert login(session, "admin", "welcome123") is True
    assert session == {
        "authenticated": True,
        "username": "admin",
        "role": "admin",
    }
    assert is_authenticated(session)
    assert is_admin(session)
    assert has_role(session, "admin")


def test_user_login_assigns_user_role_without_admin_access():
    session = {}

    assert login(session, "user01", "welcome123") is True
    assert session == {
        "authenticated": True,
        "username": "user01",
        "role": "user",
    }
    assert is_authenticated(session)
    assert has_role(session, "user")
    assert not is_admin(session)


def test_invalid_login_is_rejected_and_clears_previous_identity():
    session = {
        "authenticated": True,
        "username": "admin",
        "role": "admin",
    }

    assert login(session, "wrong username", "wrong password") is False
    assert not is_authenticated(session)
    assert "username" not in session
    assert "role" not in session


def test_logout_clears_authentication_state():
    session = {
        "authenticated": True,
        "username": "admin",
        "role": "admin",
        "messages": ["preserved by reusable auth function"],
    }

    logout(session)

    assert not is_authenticated(session)
    assert "username" not in session
    assert "role" not in session
    assert session["messages"] == ["preserved by reusable auth function"]


def test_authentication_is_case_insensitive_for_username():
    user = authenticate_user("  ADMIN  ", "welcome123")

    assert user is not None
    assert user.role == "admin"


def test_wrong_password_is_rejected():
    assert authenticate_user("admin", "incorrect") is None
