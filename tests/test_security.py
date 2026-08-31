"""Backend bearer-authentication and authorization regression tests."""

from fastapi.testclient import TestClient

from app.main import app

TEST_SECRET = "test-only-authentication-secret-value"


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AUTH_TOKEN_SECRET", TEST_SECRET)
    return TestClient(app)


def _login(client: TestClient, username: str = "user01") -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "welcome123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_login_and_me_return_backend_established_identity(monkeypatch):
    client = _client(monkeypatch)
    token = _login(client)

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "username": "user01",
        "employee_id": "EMP001",
        "role": "user",
    }


def test_invalid_credentials_are_rejected(monkeypatch):
    client = _client(monkeypatch)

    response = client.post(
        "/api/v1/auth/login",
        json={"username": "user01", "password": "wrong"},
    )

    assert response.status_code == 401


def test_missing_and_altered_tokens_are_rejected(monkeypatch):
    client = _client(monkeypatch)
    token = _login(client)

    missing = client.post("/ask", json={"question": "What is the policy?"})
    altered = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}altered"},
    )

    assert missing.status_code == 401
    assert altered.status_code == 401


def test_normal_user_cannot_synchronize_repository(monkeypatch):
    client = _client(monkeypatch)
    token = _login(client)

    response = client.post(
        "/api/v1/repository/sync",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403


def test_question_schema_has_no_caller_controlled_identity():
    from app.api.schemas import AskRequest

    assert "username" not in AskRequest.model_fields
    assert "employee_id" not in AskRequest.model_fields
    assert "role" not in AskRequest.model_fields
