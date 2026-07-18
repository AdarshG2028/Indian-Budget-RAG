"""
Shared pytest configuration.

src/rag/pipeline.py and its dependencies (retrieval, llm, reranking) import
each other with bare module names (e.g. ``from retrieval import ...``)
rather than ``src.retrieval``. That convention matches how src/api/main.py
adds ``src/`` to sys.path at runtime. Mirror it here so tests can import
those modules the same way the app does.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
