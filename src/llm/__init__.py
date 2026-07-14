"""
LLM module for Indian Budget RAG.

This module provides a production-grade LLM integration layer with:
- Abstract BaseLLM interface for provider independence
- Groq LLM implementation
- Context builder for formatting retrieved chunks
- Modular prompt templates (QA, summarization, comparison, analysis)
- Streaming response support
- Comprehensive metrics and logging
"""

from .base_llm import BaseLLM
from .groq_llm import GroqLLM
from .models import LLMConfig, LLMResponse, LLMStreamChunk, LLMMetrics
from .context_builder import ContextBuilder, ContextBuilderConfig
from .prompts import (
    PromptTemplate,
    QAPromptTemplate,
    SummarizationPromptTemplate,
    ComparisonPromptTemplate,
    AnalysisPromptTemplate,
    PromptTemplateRegistry
)

__all__ = [
    "BaseLLM",
    "GroqLLM",
    "LLMConfig",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMMetrics",
    "ContextBuilder",
    "ContextBuilderConfig",
    "PromptTemplate",
    "QAPromptTemplate",
    "SummarizationPromptTemplate",
    "ComparisonPromptTemplate",
    "AnalysisPromptTemplate",
    "PromptTemplateRegistry",
]
