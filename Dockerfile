# API image: `docker compose up api` (see README). Serves the model exported to ./models.
#
# Behind a TLS-intercepting proxy, pass the proxy's CA bundle as a BuildKit secret (see
# README): it is mounted only while dependencies download and never stored in the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first so code changes don't invalidate the (large) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=ca_bundle \
    if [ -s /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN --mount=type=secret,id=ca_bundle \
    if [ -s /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked --no-dev

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
# Migrate, then serve. MODEL_URI points at the mounted export (txcat export-model).
CMD ["sh", "-c", "alembic upgrade head && uvicorn txcat.api.app:app --host 0.0.0.0 --port 8000"]
