FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/workassist

WORKDIR /app

COPY frontend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-compile -r /tmp/requirements.txt \
    && python -m pip check \
    && rm /tmp/requirements.txt

RUN groupadd --gid 10001 workassist \
    && useradd --uid 10001 --gid workassist --create-home workassist

COPY --chown=workassist:workassist app.py streamlit_app.py auth.py ./
COPY --chown=workassist:workassist app/__init__.py app/frontend_client.py ./app/
COPY --chown=workassist:workassist src/__init__.py src/config.py src/guardrails.py src/logging_config.py ./src/
COPY --chown=workassist:workassist assets/ ./assets/
COPY --chown=workassist:workassist .streamlit/config.toml ./.streamlit/config.toml

USER workassist

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
