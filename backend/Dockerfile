FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-compile -r /tmp/requirements.txt \
    && python -m pip check \
    && rm /tmp/requirements.txt

RUN groupadd --gid 10001 workassist \
    && useradd --uid 10001 --gid workassist --create-home workassist

COPY --chown=workassist:workassist app/ ./app/
COPY --chown=workassist:workassist src/ ./src/
COPY --chown=workassist:workassist auth.py ./auth.py

USER workassist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
