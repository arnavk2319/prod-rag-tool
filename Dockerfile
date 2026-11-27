# syntax=docker/dockerfile:1
# Multi-stage: build the venv with uv, ship only the venv + source on a slim base, run as non-root.
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12.8-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12.8-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
# mlflow artifact root: exists and is owned by the app user so a named volume inherits it
RUN mkdir -p /mlflow/artifacts && chown -R app:app /mlflow
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=build --chown=app:app /app/src /app/src
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app alembic.ini ./
# config/ and prompts/ are bind-mounted read-only by compose (they may be empty in a fresh clone)
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "rag_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
