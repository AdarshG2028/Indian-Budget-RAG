"""
MRR (Mean Reciprocal Rank) metric implementation.
"""
from typing import List, Optional

from .base import BaseMetric
from ..models import EvaluationQuery, RetrievedChunk


class MRR(BaseMetric):
    """
    Mean Reciprocal Rank: measures the reciprocal of the rank of the first relevant document.
    
    MRR = 1 / rank_of_first_relevant_document
    
    If no relevant document is retrieved, MRR = 0.
    """
    
    def __init__(self, min_relevance: int = 1):
        """
        Initialize MRR metric.
        
        Args:
            min_relevance: Minimum relevance threshold
        """
        super().__init__(min_relevance=min_relevance)
    
    def calculate(
        self,
        query: EvaluationQuery,
        retrieved_chunks: List[RetrievedChunk]
    ) -> float:
        """
        Calculate MRR for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            MRR value between 0 and 1
        """
        relevant_ids = query.get_relevant_chunk_ids(self.min_relevance)
        
        # Find rank of first relevant chunk
        for chunk in retrieved_chunks:
            if chunk.chunk_id in relevant_ids:
                return 1.0 / chunk.rank
        
        # No relevant chunk found
        return 0.0
    
    def get_name(self) -> str:
        """Get metric name."""
        return "mrr"
