"""
Abstract interface for reranking components.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from .models import RerankerConfig, RerankerOutput, RerankerMetrics


class BaseReranker(ABC):
    """
    Abstract interface for reranking operations.
    
    This abstraction allows switching between different reranking strategies
    (Cross-Encoder, LLM-based, Hybrid, Ensemble) without changing the RAG pipeline.
    
    The pipeline should depend only on this interface, not on concrete implementations.
    """
    
    def __init__(self, config: RerankerConfig):
        """
        Initialize the reranker with configuration.
        
        Args:
            config: Reranker configuration
        """
        self.config = config
        self._metrics = RerankerMetrics(
            reranking_latency=0.0,
            avg_score_improvement=0.0,
            avg_rank_movement=0.0,
            largest_rank_movement=0,
            reordering_percentage=0.0,
            fallback_count=0,
            total_queries=0,
            model_name=config.model_name
        )
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> RerankerOutput:
        """
        Rerank a single query's retrieved chunks.
        
        Args:
            query: Original query text
            retrieved_chunks: List of retrieved chunks with metadata
            
        Returns:
            RerankerOutput with reranked results
        """
        pass
    
    @abstractmethod
    def rerank_batch(
        self,
        queries: List[str],
        retrieved_chunks_list: List[List[Dict[str, Any]]]
    ) -> List[RerankerOutput]:
        """
        Rerank multiple queries' retrieved chunks efficiently.
        
        Args:
            queries: List of query texts
            retrieved_chunks_list: List of retrieved chunks for each query
            
        Returns:
            List of RerankerOutput for each query
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the reranker configuration is valid and the model is accessible.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    def get_metrics(self) -> RerankerMetrics:
        """
        Get accumulated reranking metrics.
        
        Returns:
            RerankerMetrics object
        """
        return self._metrics
    
    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._metrics = RerankerMetrics(
            reranking_latency=0.0,
            avg_score_improvement=0.0,
            avg_rank_movement=0.0,
            largest_rank_movement=0,
            reordering_percentage=0.0,
            fallback_count=0,
            total_queries=0,
            model_name=self.config.model_name
        )
