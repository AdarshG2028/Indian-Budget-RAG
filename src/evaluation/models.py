"""
Strongly typed models for evaluation framework.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    RECALL_AT_K = "recall_at_k"
    PRECISION_AT_K = "precision_at_k"
    MRR = "mrr"
    NDCG = "ndcg"
    HIT_RATE = "hit_rate"
    MAP = "map"


class RelevanceLevel(int, Enum):
    """Graded relevance levels."""
    NOT_RELEVANT = 0
    SOMEWHAT_RELEVANT = 1
    RELEVANT = 2
    HIGHLY_RELEVANT = 3


@dataclass
class GroundTruthChunk:
    """
    Represents a ground truth chunk with graded relevance.
    
    Args:
        chunk_id: Unique identifier for the chunk
        relevance: Relevance score (0-3, higher is more relevant)
    """
    chunk_id: str
    relevance: int
    
    def __post_init__(self):
        """Validate relevance score."""
        if not 0 <= self.relevance <= 3:
            raise ValueError(f"Relevance must be between 0 and 3, got {self.relevance}")


@dataclass
class EvaluationQuery:
    """
    Represents a single evaluation query with ground truth.
    
    Args:
        query_id: Unique identifier for the query
        query_text: The actual query text
        ground_truth: List of relevant chunks with relevance scores
        metadata: Optional additional metadata
    """
    query_id: str
    query_text: str
    ground_truth: List[GroundTruthChunk]
    metadata: Optional[Dict[str, Any]] = None
    
    def get_relevant_chunk_ids(self, min_relevance: int = 1) -> set[str]:
        """
        Get set of chunk IDs with relevance >= min_relevance.
        
        Args:
            min_relevance: Minimum relevance threshold
            
        Returns:
            Set of relevant chunk IDs
        """
        return {
            gt.chunk_id 
            for gt in self.ground_truth 
            if gt.relevance >= min_relevance
        }
    
    def get_relevance_score(self, chunk_id: str) -> int:
        """
        Get relevance score for a specific chunk ID.
        
        Args:
            chunk_id: Chunk ID to look up
            
        Returns:
            Relevance score (0 if chunk not in ground truth)
        """
        for gt in self.ground_truth:
            if gt.chunk_id == chunk_id:
                return gt.relevance
        return 0


@dataclass
class RetrievedChunk:
    """
    Represents a retrieved chunk during evaluation.
    
    Args:
        chunk_id: Unique identifier for the chunk
        rank: Position in retrieved results (1-indexed)
        score: Similarity score from retrieval
        metadata: Optional additional metadata
    """
    chunk_id: str
    rank: int
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResult:
    """
    Results for a single query evaluation.
    
    Args:
        query: The evaluation query
        retrieved_chunks: List of retrieved chunks
        metrics: Dictionary of metric names to values
        latency: Retrieval latency in seconds
        embedding_latency: Embedding latency in seconds
        search_latency: Search latency in seconds
        chunks_returned: Number of chunks returned
        avg_similarity: Average similarity score
        highest_similarity: Highest similarity score
        lowest_similarity: Lowest similarity score
    """
    query: EvaluationQuery
    retrieved_chunks: List[RetrievedChunk]
    metrics: Dict[str, float]
    latency: float
    embedding_latency: float
    search_latency: float
    chunks_returned: int
    avg_similarity: float
    highest_similarity: float
    lowest_similarity: float
    
    def get_missing_chunks(self) -> List[str]:
        """
        Get ground truth chunks that were not retrieved.
        
        Returns:
            List of missing chunk IDs
        """
        retrieved_ids = {rc.chunk_id for rc in self.retrieved_chunks}
        relevant_ids = self.query.get_relevant_chunk_ids()
        return list(relevant_ids - retrieved_ids)
    
    def get_zero_recall(self) -> bool:
        """
        Check if query has zero recall (no relevant chunks retrieved).
        
        Returns:
            True if no relevant chunks were retrieved
        """
        relevant_ids = self.query.get_relevant_chunk_ids()
        retrieved_ids = {rc.chunk_id for rc in self.retrieved_chunks}
        return len(relevant_ids & retrieved_ids) == 0


@dataclass
class ExperimentMetadata:
    """
    Metadata for an evaluation experiment.
    
    Args:
        experiment_id: Unique identifier for the experiment
        timestamp: When the experiment was run
        embedding_model: Name of embedding model used
        retriever: Type of retriever used
        reranker: Type of reranker used (if any)
        vector_database: Vector database used
        chunk_size: Chunk size used
        chunk_overlap: Chunk overlap used
        collection: Collection name
        dataset: Dataset used for evaluation
        retrieval_config: Retrieval configuration
        additional_metadata: Optional additional metadata
    """
    experiment_id: str
    timestamp: datetime
    embedding_model: str
    retriever: str
    reranker: Optional[str]
    vector_database: str
    chunk_size: int
    chunk_overlap: int
    collection: str
    dataset: str
    retrieval_config: Dict[str, Any]
    additional_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "embedding_model": self.embedding_model,
            "retriever": self.retriever,
            "reranker": self.reranker,
            "vector_database": self.vector_database,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "collection": self.collection,
            "dataset": self.dataset,
            "retrieval_config": self.retrieval_config,
            "additional_metadata": self.additional_metadata or {}
        }


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for an experiment.
    
    Args:
        metadata: Experiment metadata
        query_results: Results for each query
        aggregate_metrics: Aggregate metrics across all queries
        failure_analysis: Failure analysis data
        latency_metrics: Latency metrics summary
    """
    metadata: ExperimentMetadata
    query_results: List[QueryResult]
    aggregate_metrics: Dict[str, float]
    failure_analysis: Dict[str, Any]
    latency_metrics: Dict[str, float]
    
    def get_zero_recall_queries(self) -> List[QueryResult]:
        """
        Get queries with zero recall.
        
        Returns:
            List of query results with zero recall
        """
        return [qr for qr in self.query_results if qr.get_zero_recall()]
    
    def get_most_frequently_missed_chunks(self, top_n: int = 10) -> List[tuple[str, int]]:
        """
        Get most frequently missed chunks across all queries.
        
        Args:
            top_n: Number of top chunks to return
            
        Returns:
            List of (chunk_id, miss_count) tuples
        """
        miss_counts: Dict[str, int] = {}
        
        for qr in self.query_results:
            for chunk_id in qr.get_missing_chunks():
                miss_counts[chunk_id] = miss_counts.get(chunk_id, 0) + 1
        
        return sorted(miss_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "aggregate_metrics": self.aggregate_metrics,
            "failure_analysis": self.failure_analysis,
            "latency_metrics": self.latency_metrics,
            "num_queries": len(self.query_results),
            "query_results": [
                {
                    "query_id": qr.query.query_id,
                    "query_text": qr.query.query_text,
                    "metrics": qr.metrics,
                    "latency": qr.latency,
                    "chunks_returned": qr.chunks_returned,
                    "avg_similarity": qr.avg_similarity,
                    "missing_chunks": qr.get_missing_chunks(),
                    "zero_recall": qr.get_zero_recall()
                }
                for qr in self.query_results
            ]
        }


@dataclass
class ExperimentComparison:
    """
    Comparison between two experiments.
    
    Args:
        experiment_a: First experiment report
        experiment_b: Second experiment report
        metric_differences: Dictionary of metric differences
        improvement_summary: Summary of improvements
    """
    experiment_a: EvaluationReport
    experiment_b: EvaluationReport
    metric_differences: Dict[str, Dict[str, float]]
    improvement_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_a_id": self.experiment_a.metadata.experiment_id,
            "experiment_b_id": self.experiment_b.metadata.experiment_id,
            "metric_differences": self.metric_differences,
            "improvement_summary": self.improvement_summary
        }
