"""
RAG module for Indian Budget RAG.

This module provides the complete RAG pipeline orchestrating:
- Retrieval from vector database
- Context building from retrieved chunks
- LLM generation with prompt templates
- Source citations
- Comprehensive metrics and logging
"""

from .pipeline import RAGPipeline, RAGConfig, RAGResponse

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "RAGResponse",
]
