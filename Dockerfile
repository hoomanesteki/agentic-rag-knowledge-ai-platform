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

# Create the non-root user up front and build as that user, so the venv is owned correctly without
# a `chown -R` layer that would duplicate the whole environment and roughly double the image size.
RUN useradd --create-home --uid 10001 skein
WORKDIR /app
RUN chown skein:skein /app
USER skein

# Install locked dependencies first (no project, no dev) so this layer caches across code changes.
COPY --chown=skein:skein pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the app itself (.venv is dockerignored, so this does not clobber the one just built).
COPY --chown=skein:skein . .
RUN uv sync --frozen --no-dev

EXPOSE 8080
# One worker keeps memory low on a cold-starting min-instances-0 service. Scale out with instances,
# not workers. Shell form so ${PORT} expands (Cloud Run sets it; default 8080 for local runs).
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
