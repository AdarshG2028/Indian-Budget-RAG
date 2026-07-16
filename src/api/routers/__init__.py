"""
Routers module.
"""
from .health import router as health_router
from .rag import router as rag_router
from .retrieval import router as retrieval_router
from .evaluation import router as evaluation_router

__all__ = [
    "health_router",
    "rag_router",
    "retrieval_router",
    "evaluation_router"
]
