"""
Reranking module for Indian Budget RAG.

This module provides a production-grade reranking system with:
- Abstract BaseReranker interface for provider independence
- CrossEncoderReranker implementation with sentence-transformers
- Model management for efficient loading and caching
- Batch processing for efficiency
- Score normalization
- Robust error handling with fallback
- Detailed metrics and observability
- Extensible architecture for future rerankers (LLM, Hybrid, Ensemble)
"""

from .base import BaseReranker
from .models import RerankerConfig, RerankerOutput, RerankerResult, RerankerMetrics
from .cross_encoder import CrossEncoderReranker
from .model_manager import ModelManager, get_global_model_manager, set_global_model_manager
from .future import LLMReranker, HybridReranker, EnsembleReranker

__all__ = [
    "BaseReranker",
    "RerankerConfig",
    "RerankerOutput",
    "RerankerResult",
    "RerankerMetrics",
    "CrossEncoderReranker",
    "ModelManager",
    "get_global_model_manager",
    "set_global_model_manager",
    "LLMReranker",
    "HybridReranker",
    "EnsembleReranker",
]
