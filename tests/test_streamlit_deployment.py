"""Deployment configuration checks for the Streamlit frontend."""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_server_configuration():
    with (PROJECT_ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["server"] == {
        "address": "0.0.0.0",
        "port": 8501,
        "headless": True,
    }
    assert config["browser"]["gatherUsageStats"] is False
    assert config["client"]["showErrorDetails"] == "none"


def test_frontend_does_not_render_internal_backend_url():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert "API_BASE_URL" not in source
    assert 'st.caption(f"API:' not in source
