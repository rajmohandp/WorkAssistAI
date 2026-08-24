"""Streamlit-level authentication and RBAC regression tests."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FakeResponse:
    ok = True

    def json(self):
        return {
            "s3": {
                "connected": True,
                "bucket": "documents-bucket",
                "supported_documents": 0,
                "documents": [],
            },
            "pinecone": {
                "connected": True,
                "index_name": "docuverse",
                "indexed_documents": 0,
                "total_vectors": 0,
            },
        }


def _login(monkeypatch, username, password):
    monkeypatch.setattr(
        "app.frontend_client.requests.get",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    app.text_input[0].input(username)
    app.text_input[1].input(password)
    app.button[0].click()
    return app.run(timeout=10)


def test_unauthenticated_session_only_shows_login():
    app = AppTest.from_file(APP_PATH).run(timeout=10)

    assert not app.exception
    assert [field.label for field in app.text_input] == ["Username", "Password"]
    assert [button.label for button in app.button] == ["Login"]
    assert not app.radio


def test_admin_enters_admin_mode_with_sync_access(monkeypatch):
    app = _login(monkeypatch, "admin", "welcome123")

    assert not app.exception
    assert app.session_state.authenticated is True
    assert app.session_state.username == "admin"
    assert app.session_state.role == "admin"
    assert app.radio[0].value == "Admin"
    assert "Sync Documents" in [button.label for button in app.button]


def test_user_enters_user_mode_without_admin_access(monkeypatch):
    app = _login(monkeypatch, "user01", "welcome123")

    assert not app.exception
    assert app.session_state.authenticated is True
    assert app.session_state.username == "user01"
    assert app.session_state.role == "user"
    assert not app.radio
    assert "Sync Documents" not in [button.label for button in app.button]
    assert "Clear Conversation" in [button.label for button in app.button]


def test_invalid_login_remains_on_login_page(monkeypatch):
    app = _login(monkeypatch, "wrong username", "wrong password")

    assert not app.exception
    assert "authenticated" not in app.session_state
    assert [button.label for button in app.button] == ["Login"]
    assert any("Invalid username or password" in error.value for error in app.error)
