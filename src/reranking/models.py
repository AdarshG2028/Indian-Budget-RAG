"""
Strongly typed models for reranking framework.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RerankerConfig:
    """
    Configuration for reranking operations.
    
    Args:
        model_name: Name of the reranker model
        device: Device for model inference (cpu/cuda)
        batch_size: Batch size for processing
        score_threshold: Minimum score threshold
        normalize_scores: Whether to normalize scores
        retrieve_top_k: Number of chunks to retrieve from dense search
        rerank_top_k: Number of chunks to rerank
        return_top_k: Number of chunks to return after reranking
        enable_fallback: Whether to fallback to original results on failure
        timeout: Timeout for reranking operations
    """
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    score_threshold: Optional[float] = None
    normalize_scores: bool = True
    retrieve_top_k: int = 20
    rerank_top_k: int = 20
    return_top_k: int = 5
    enable_fallback: bool = True
    timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.retrieve_top_k < self.rerank_top_k:
            raise ValueError(
                f"retrieve_top_k ({self.retrieve_top_k}) must be >= rerank_top_k ({self.rerank_top_k})"
            )
        if self.rerank_top_k < self.return_top_k:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) must be >= return_top_k ({self.return_top_k})"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


@dataclass
class RerankerResult:
    """
    Result of reranking a single chunk.
    
    Args:
        chunk_id: Unique identifier for the chunk
        original_rank: Rank from initial retrieval
        reranked_rank: Rank after reranking
        original_score: Score from initial retrieval
        reranked_score: Score after reranking
        score_delta: Change in score (reranked - original)
        text: Chunk text
        metadata: Additional chunk metadata
    """
    chunk_id: str
    original_rank: int
    reranked_rank: int
    original_score: float
    reranked_score: float
    score_delta: float
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RerankerOutput:
    """
    Complete output from reranking operation.
    
    Args:
        results: List of reranked results
        query: Original query
        reranking_latency: Time taken for reranking
        fallback_used: Whether fallback to original results was used
        model_name: Model used for reranking
    """
    results: List[RerankerResult]
    query: str
    reranking_latency: float
    fallback_used: bool
    model_name: str
    
    def get_top_k(self, k: int) -> List[RerankerResult]:
        """
        Get top K results by reranked rank.
        
        Args:
            k: Number of results to return
            
        Returns:
            List of top K reranked results
        """
        return [r for r in self.results if r.reranked_rank <= k]


@dataclass
class RerankerMetrics:
    """
    Detailed metrics for reranking operations.
    
    Args:
        reranking_latency: Time taken for reranking
        avg_score_improvement: Average improvement in scores
        avg_rank_movement: Average change in rank
        largest_rank_movement: Maximum change in rank
        reordering_percentage: Percentage of chunks that changed rank
        fallback_count: Number of times fallback was used
        total_queries: Total number of queries processed
        model_name: Model used for reranking
    """
    reranking_latency: float
    avg_score_improvement: float
    avg_rank_movement: float
    largest_rank_movement: int
    reordering_percentage: float
    fallback_count: int
    total_queries: int
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reranking_latency": self.reranking_latency,
            "avg_score_improvement": self.avg_score_improvement,
            "avg_rank_movement": self.avg_rank_movement,
            "largest_rank_movement": self.largest_rank_movement,
            "reordering_percentage": self.reordering_percentage,
            "fallback_count": self.fallback_count,
            "total_queries": self.total_queries,
            "model_name": self.model_name
        }
