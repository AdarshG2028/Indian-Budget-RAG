"""
Strongly typed models for retrieval results.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


@dataclass
class RetrievalResult:
    """
    Represents a single retrieved chunk with all relevant metadata.
    
    This model is independent of the vector database response format,
    making the retrieval layer portable across different backends.
    """
    chunk_id: str
    document: str
    year: int
    section: str
    subsection: str
    paragraph_start: int
    paragraph_end: int
    page_start: int
    page_end: int
    score: float
    rank: int
    text: str
    metadata: Dict[str, Any]
    
    @classmethod
    def from_qdrant_payload(cls, payload: Dict[str, Any], score: float, rank: int) -> "RetrievalResult":
        """
        Create a RetrievalResult from a Qdrant payload.
        
        Args:
            payload: Qdrant point payload
            score: Similarity score from vector search
            rank: Result rank position
            
        Returns:
            RetrievalResult instance
        """
        return cls(
            chunk_id=payload.get("chunk_id", ""),
            document=payload.get("document", ""),
            year=payload.get("year", 0),
            section=payload.get("section", ""),
            subsection=payload.get("subsection", ""),
            paragraph_start=payload.get("paragraph_start", 0),
            paragraph_end=payload.get("paragraph_end", 0),
            page_start=payload.get("page_start", 0),
            page_end=payload.get("page_end", 0),
            score=score,
            rank=rank,
            text=payload.get("text", ""),
            metadata={k: v for k, v in payload.items() if k not in [
                "chunk_id", "document", "year", "section", "subsection",
                "paragraph_start", "paragraph_end", "page_start", "page_end", "text"
            ]}
        )


@dataclass
class RetrievalMetrics:
    """
    Metrics for retrieval observability.
    """
    avg_similarity: float
    highest_similarity: float
    lowest_similarity: float
    chunks_returned: int
    retrieval_latency: float
    embedding_latency: float
    search_latency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/export."""
        return {
            "avg_similarity": self.avg_similarity,
            "highest_similarity": self.highest_similarity,
            "lowest_similarity": self.lowest_similarity,
            "chunks_returned": self.chunks_returned,
            "retrieval_latency": self.retrieval_latency,
            "embedding_latency": self.embedding_latency,
            "search_latency": self.search_latency,
        }


@dataclass
class RetrievalContext:
    """
    Complete context for a retrieval operation.
    
    Contains the query, configuration, timing information,
    metrics, and results for observability and debugging.
    """
    query: str
    top_k: int
    results: List[RetrievalResult]
    metrics: RetrievalMetrics
    filters: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/export."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "filters": self.filters,
            "score_threshold": self.score_threshold,
            "metrics": self.metrics.to_dict(),
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document": r.document,
                    "year": r.year,
                    "section": r.section,
                    "subsection": r.subsection,
                    "page_start": r.page_start,
                    "page_end": r.page_end,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in self.results
            ]
        }


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval operations.
    """
    top_k: int = 10
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
