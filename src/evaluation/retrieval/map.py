"""
MAP (Mean Average Precision) metric implementation.
"""
from typing import List, Optional

from .base import BaseMetric
from ..models import EvaluationQuery, RetrievedChunk


class MAP(BaseMetric):
    """
    Mean Average Precision: measures the mean of average precision across all queries.
    
    Average Precision = average of precision values at each rank where a relevant document is found.
    MAP = mean of Average Precision across all queries.
    
    This is a single-query implementation that returns Average Precision.
    The mean across queries should be computed by the evaluation runner.
    """
    
    def __init__(self, k: int = 10, min_relevance: int = 1):
        """
        Initialize MAP metric.
        
        Args:
            k: K value for AP@K
            min_relevance: Minimum relevance threshold
        """
        super().__init__(k=k, min_relevance=min_relevance)
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
    
    def calculate(
        self,
        query: EvaluationQuery,
        retrieved_chunks: List[RetrievedChunk]
    ) -> float:
        """
        Calculate Average Precision@K for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Average Precision@K value between 0 and 1
        """
        # Get top K retrieved chunks
        top_k_chunks = retrieved_chunks[:self.k]
        
        relevant_ids = query.get_relevant_chunk_ids(self.min_relevance)
        total_relevant = len(relevant_ids)
        
        if total_relevant == 0:
            return 0.0
        
        # Calculate precision at each rank
        precision_values = []
        relevant_count = 0
        
        for rank, chunk in enumerate(top_k_chunks, 1):
            if chunk.chunk_id in relevant_ids:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_values.append(precision_at_rank)
        
        # Average Precision is mean of precision values
        if not precision_values:
            return 0.0
        
        return sum(precision_values) / len(precision_values)
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"map@{self.k}"
