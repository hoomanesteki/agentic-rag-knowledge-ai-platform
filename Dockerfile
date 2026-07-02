# API container for Cloud Run (min-instances 0). Cloud Run injects $PORT; we bind it.
# Build:  docker build -t skein-api .
# Run:    docker run -p 8080:8080 --env-file .env skein-api
FROM python:3.12-slim

# uv pinned to the version that produced uv.lock, so the image resolves the same graph.
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install locked dependencies first (no project, no dev) so this layer caches across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the app itself.
COPY . .
RUN uv sync --frozen --no-dev

# A non-root user; Cloud Run runs containers unprivileged and this matches that.
RUN useradd --create-home --uid 10001 skein && chown -R skein:skein /app
USER skein

EXPOSE 8080
# One worker keeps memory low on a cold-starting min-instances-0 service. Scale out with instances,
# not workers. Shell form so ${PORT} expands (Cloud Run sets it; default 8080 for local runs).
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
