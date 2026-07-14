"""
Retrieval module for Indian Budget RAG.

This module provides a production-grade retrieval layer with:
- Dense vector retrieval
- Extensible architecture for hybrid search, reranking, etc.
- Strongly typed models
- Generic metadata filtering with operators
- Comprehensive logging and metrics
- Pipeline hooks for query expansion and reranking
"""

from .models import RetrievalResult, RetrievalConfig, RetrievalContext, RetrievalMetrics
from .retriever import BaseRetriever, DenseRetriever
from .vector_store import VectorStore, QdrantVectorStore
from .filters import FilterCondition, FilterOperator, FilterBuilder, FilterConverter

__all__ = [
    "RetrievalResult",
    "RetrievalConfig",
    "RetrievalContext",
    "RetrievalMetrics",
    "BaseRetriever",
    "DenseRetriever",
    "VectorStore",
    "QdrantVectorStore",
    "FilterCondition",
    "FilterOperator",
    "FilterBuilder",
    "FilterConverter",
]
