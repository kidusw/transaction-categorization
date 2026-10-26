# One Dockerfile, three images (see docker-compose.yml):
#   api        (default)                 core dependencies only: FastAPI + scikit-learn + skops
#   dashboard  --only-group dashboard    Streamlit + httpx only, talks to the API over HTTP
#   bootstrap  --group train             + data build, training, MLflow; runs once and exits
#
# Behind a TLS-intercepting proxy, pass the proxy's CA bundle as a BuildKit secret (see
# README): it is mounted only while dependencies download and never stored in the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

WORKDIR /app
# UV_NO_CACHE keeps uv's download cache (~330 MB) out of the image layers.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Which dependencies this image gets (uv sync flags).
ARG SYNC_ARGS="--no-default-groups"

# Dependencies first so code changes don't invalidate the (large) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=ca_bundle \
    if [ -s /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked $SYNC_ARGS --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN --mount=type=secret,id=ca_bundle \
    if [ -s /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked $SYNC_ARGS

ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8000
# Default: the API. MODEL_URI points at the exported model (txcat export-model / bootstrap).
CMD ["uvicorn", "txcat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
