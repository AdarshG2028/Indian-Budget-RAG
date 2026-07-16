"""
Base interface for retrieval metrics.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models import EvaluationQuery, RetrievedChunk


class BaseMetric(ABC):
    """
    Abstract base class for retrieval metrics.
    
    All retrieval metrics should inherit from this class and implement
    the calculate method. This ensures consistent interface and allows
    easy addition of new metrics without changing the evaluation framework.
    """
    
    def __init__(self, k: Optional[int] = None, min_relevance: int = 1):
        """
        Initialize the metric.
        
        Args:
            k: Optional K value for @K metrics (e.g., Recall@5)
            min_relevance: Minimum relevance threshold for considering a chunk relevant
        """
        self.k = k
        self.min_relevance = min_relevance
    
    @abstractmethod
    def calculate(
        self,
        query: EvaluationQuery,
        retrieved_chunks: List[RetrievedChunk]
    ) -> float:
        """
        Calculate the metric for a single query.
        
        Args:
            query: The evaluation query with ground truth
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Metric value as float
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this metric.
        
        Returns:
            Metric name string
        """
        pass
    
    def get_relevant_retrieved_count(
        self,
        query: EvaluationQuery,
        retrieved_chunks: List[RetrievedChunk]
    ) -> int:
        """
        Count how many retrieved chunks are relevant.
        
        Args:
            query: The evaluation query
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Number of relevant chunks in retrieved results
        """
        relevant_ids = query.get_relevant_chunk_ids(self.min_relevance)
        retrieved_ids = {rc.chunk_id for rc in retrieved_chunks}
        return len(relevant_ids & retrieved_ids)
    
    def get_total_relevant_count(self, query: EvaluationQuery) -> int:
        """
        Get total number of relevant chunks for the query.
        
        Args:
            query: The evaluation query
            
        Returns:
            Total number of relevant chunks
        """
        return len(query.get_relevant_chunk_ids(self.min_relevance))
