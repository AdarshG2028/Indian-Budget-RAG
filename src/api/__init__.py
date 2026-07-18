"""
API module for Indian Budget RAG.
"""
from .config import settings

__all__ = ["app", "settings"]


def __getattr__(name: str):
    """Lazily load the FastAPI application when it is explicitly requested.

    Retrieval and telemetry modules import ``src.api`` as part of their module
    path.  Importing the application here caused those modules to initialise
    every router and model at import time, including during retrieval-only
    tests.
    """
    if name == "app":
        from .main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
