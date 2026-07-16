"""
Evaluation framework for Indian Budget RAG.

This module provides a production-grade evaluation system with:
- Modular retrieval metrics (Recall@K, Precision@K, MRR, NDCG, Hit Rate, MAP)
- Experiment tracking and comparison
- Dataset loading (JSON/CSV)
- Comprehensive reporting (JSON/CSV/Markdown)
- Failure analysis
- Architecture for future generation evaluation
"""

from .models import (
    EvaluationQuery,
    GroundTruthChunk,
    RetrievedChunk,
    QueryResult,
    ExperimentMetadata,
    EvaluationReport,
    ExperimentComparison,
    MetricType,
    RelevanceLevel
)
from .data_loader import DataLoader, DatasetValidationError
from .runner import EvaluationRunner
from .experiment import ExperimentTracker
from .reporting import ReportGenerator

__all__ = [
    "EvaluationQuery",
    "GroundTruthChunk",
    "RetrievedChunk",
    "QueryResult",
    "ExperimentMetadata",
    "EvaluationReport",
    "ExperimentComparison",
    "MetricType",
    "RelevanceLevel",
    "DataLoader",
    "DatasetValidationError",
    "EvaluationRunner",
    "ExperimentTracker",
    "ReportGenerator",
]
