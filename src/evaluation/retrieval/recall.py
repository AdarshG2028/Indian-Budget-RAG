"""
Recall@K metric implementation.
"""
from typing import List, Optional

from .base import BaseMetric
from ..models import EvaluationQuery, RetrievedChunk


class RecallAtK(BaseMetric):
    """
    Recall@K metric: measures the fraction of relevant documents retrieved in the top K results.
    
    Recall@K = (number of relevant documents in top K) / (total number of relevant documents)
    """
    
    def __init__(self, k: int = 10, min_relevance: int = 1):
        """
        Initialize Recall@K metric.
        
        Args:
            k: K value for Recall@K
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
        Calculate Recall@K for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Recall@K value between 0 and 1
        """
        # Get top K retrieved chunks
        top_k_chunks = retrieved_chunks[:self.k]
        
        # Count relevant chunks in top K
        relevant_retrieved = self.get_relevant_retrieved_count(query, top_k_chunks)
        
        # Get total relevant chunks
        total_relevant = self.get_total_relevant_count(query)
        
        # Avoid division by zero
        if total_relevant == 0:
            return 0.0
        
        return relevant_retrieved / total_relevant
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"recall@{self.k}"
