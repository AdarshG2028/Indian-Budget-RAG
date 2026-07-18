# Production Dockerfile for Indian Budget RAG API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy application code (retrieval goes through Qdrant, not local files —
# data/ is ingestion-time only and isn't needed at runtime)
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p evaluation/reports evaluation/experiments

# Expose port
EXPOSE 8000

# Health check (stdlib-only: urlopen raises on connection failure or a
# non-2xx status, which is enough to mark the container unhealthy;
# 'requests' is not a project dependency and would always ImportError)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/live', timeout=5)"

# Run the application
CMD ["uv", "run", "python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
