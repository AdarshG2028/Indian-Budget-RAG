"""
Precision@K metric implementation.
"""
from typing import List, Optional

from .base import BaseMetric
from ..models import EvaluationQuery, RetrievedChunk


class PrecisionAtK(BaseMetric):
    """
    Precision@K metric: measures the fraction of retrieved documents that are relevant in the top K results.
    
    Precision@K = (number of relevant documents in top K) / K
    """
    
    def __init__(self, k: int = 10, min_relevance: int = 1):
        """
        Initialize Precision@K metric.
        
        Args:
            k: K value for Precision@K
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
        Calculate Precision@K for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Precision@K value between 0 and 1
        """
        # Get top K retrieved chunks
        top_k_chunks = retrieved_chunks[:self.k]
        
        # If fewer than K chunks retrieved, use actual count
        actual_k = min(len(top_k_chunks), self.k)
        
        if actual_k == 0:
            return 0.0
        
        # Count relevant chunks in top K
        relevant_retrieved = self.get_relevant_retrieved_count(query, top_k_chunks)
        
        return relevant_retrieved / actual_k
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"precision@{self.k}"
