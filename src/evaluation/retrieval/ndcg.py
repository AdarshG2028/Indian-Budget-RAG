"""
NDCG (Normalized Discounted Cumulative Gain) metric implementation.
"""
import math
from typing import List, Optional

from .base import BaseMetric
from ..models import EvaluationQuery, RetrievedChunk


class NDCG(BaseMetric):
    """
    Normalized Discounted Cumulative Gain: measures ranking quality with graded relevance.
    
    DCG = sum(relevance_i / log2(rank_i + 1)) for i in 1..K
    NDCG = DCG / IDCG (ideal DCG)
    
    Accounts for position bias and supports graded relevance.
    """
    
    def __init__(self, k: int = 10, min_relevance: int = 1):
        """
        Initialize NDCG metric.
        
        Args:
            k: K value for NDCG@K
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
        Calculate NDCG@K for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            NDCG@K value between 0 and 1
        """
        # Get top K retrieved chunks
        top_k_chunks = retrieved_chunks[:self.k]
        
        # Calculate DCG
        dcg = self._calculate_dcg(query, top_k_chunks)
        
        # Calculate ideal DCG (perfect ranking)
        idcg = self._calculate_idcg(query)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _calculate_dcg(
        self,
        query: EvaluationQuery,
        retrieved_chunks: List[RetrievedChunk]
    ) -> float:
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for chunk in retrieved_chunks:
            relevance = query.get_relevance_score(chunk.chunk_id)
            if relevance >= self.min_relevance:
                dcg += relevance / math.log2(chunk.rank + 1)
        return dcg
    
    def _calculate_idcg(self, query: EvaluationQuery) -> float:
        """Calculate Ideal DCG (perfect ranking)."""
        # Get all relevant chunks sorted by relevance (descending)
        relevant_chunks = [
            gt for gt in query.ground_truth 
            if gt.relevance >= self.min_relevance
        ]
        relevant_chunks.sort(key=lambda x: x.relevance, reverse=True)
        
        # Calculate DCG for perfect ranking
        idcg = 0.0
        for rank, gt in enumerate(relevant_chunks, 1):
            idcg += gt.relevance / math.log2(rank + 1)
        
        return idcg
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"ndcg@{self.k}"
