"""
Hit Rate metric implementation.
"""
from typing import List, Optional

from .base import BaseMetric
from ..models import EvaluationQuery, RetrievedChunk


class HitRate(BaseMetric):
    """
    Hit Rate: measures whether at least one relevant document was retrieved in the top K results.
    
    Hit Rate@K = 1 if any relevant document in top K, else 0
    
    Also known as Success Rate@K.
    """
    
    def __init__(self, k: int = 10, min_relevance: int = 1):
        """
        Initialize Hit Rate metric.
        
        Args:
            k: K value for Hit Rate@K
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
        Calculate Hit Rate@K for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Hit Rate@K value (0 or 1)
        """
        # Get top K retrieved chunks
        top_k_chunks = retrieved_chunks[:self.k]
        
        # Check if any relevant chunk is in top K
        relevant_retrieved = self.get_relevant_retrieved_count(query, top_k_chunks)
        
        return 1.0 if relevant_retrieved > 0 else 0.0
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"hit_rate@{self.k}"
