"""Compatibility entry point for the existing DocuVerse Streamlit UI.

The mature UI remains in ``app.py`` while the new ``app`` package hosts the
FastAPI backend. Run this file to use the recommended frontend entry point.
"""

from pathlib import Path
from runpy import run_path

ui_path = Path(__file__).with_name("app.py")
run_path(str(ui_path), run_name="__main__")
