# Production Dockerfile for Indian Budget RAG API
#
# python:3.13-slim to match pyproject.toml's requires-python (>=3.13) -- a
# 3.11 base previously here would force uv to fetch its own 3.13 at build
# time or fail outright.
FROM python:3.13-slim

# Install system dependencies (root)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# uv on a system path so it's usable after the USER switch below.
RUN pip install --no-cache-dir uv

# Hugging Face Spaces (and most managed platforms) run the container as
# UID 1000 regardless of this Dockerfile's USER -- so root-owned files and
# caches created before this point would be invisible to the process at
# runtime. Create a matching user and give it the working directory before
# any COPY or model download.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app
RUN chown user:user /app
USER user

# Copy dependency manifests first for better layer caching
COPY --chown=user pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Bake the embedding model into the image. Without this, every cold start on
# a free-tier Space (which sleeps and loses ephemeral disk) re-downloads the
# ~440MB model from the HF Hub inside the first user's request, since
# get_embedder() in api/dependencies.py is a lazy singleton. Baking makes
# that a local cache load instead of a network fetch. Only BGE: the
# cross-encoder reranker is disabled by default (ENABLE_RERANKING=false) and
# loads lazily only if that's turned on.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Copy application code (retrieval goes through Qdrant, not local files —
# data/ is ingestion-time only and isn't needed at runtime)
COPY --chown=user src/ ./src/

# Directories written at runtime (OTel file span export, evaluation reports)
RUN mkdir -p evaluation/reports evaluation/experiments traces

# Expose port
EXPOSE 8000

# Health check (stdlib-only: urlopen raises on connection failure or a
# non-2xx status, which is enough to mark the container unhealthy;
# 'requests' is not a project dependency and would always ImportError)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/live', timeout=5)"

# Run the application
CMD ["uv", "run", "python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
