"""
Configuration for the Embedding Pipeline.
Provides central configuration without hardcoding variables inside business logic.
"""
import os
from pathlib import Path

# Load .env from the project root (two levels up from this file: src/embeddings/config.py)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)  # override=False: real env vars win
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

try:
    import torch
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
except ImportError:
    _default_device = "cpu"

# Embedding Model settings
MODEL_NAME = "BAAI/bge-base-en-v1.5"
NORMALIZE_EMBEDDINGS = True

# Batching and processing
BATCH_SIZE = 32

# Allow environment variable override for DEVICE
DEVICE = os.getenv("EMBEDDING_DEVICE", _default_device)

# ---------------------------------------------------------------------------
# Qdrant Vector Database settings
# ---------------------------------------------------------------------------
# Override via environment variables for cloud / production deployments.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)   # None = local, no auth
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "indian_budget_2026")
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")  # Cosine | Dot | Euclid
