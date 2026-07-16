"""
Retrieval service for business logic.
"""
import logging
from typing import Dict, Any, List

from retrieval import DenseRetriever, RetrievalConfig

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieval operations.
    
    Handles business logic for document retrieval.
    """
    
    def __init__(self, retriever: DenseRetriever):
        """
        Initialize retrieval service.
        
        Args:
            retriever: Dense retriever instance
        """
        self.retriever = retriever
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score
            config: Optional configuration override
            
        Returns:
            Retrieval results with metadata
        """
        retrieval_config = RetrievalConfig(
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        retrieval_context = self.retriever.retrieve(query, retrieval_config)
        
        # Convert results to API format
        results = [
            {
                "chunk_id": r.chunk_id,
                "document": r.document,
                "year": r.year,
                "section": r.section,
                "subsection": r.subsection,
                "text": r.text,
                "score": r.score,
                "rank": r.rank,
                "page_start": r.page_start,
                "page_end": r.page_end
            }
            for r in retrieval_context.results
        ]
        
        return {
            "results": results,
            "metrics": {
                "retrieval_latency": retrieval_context.metrics.retrieval_latency,
                "embedding_latency": retrieval_context.metrics.embedding_latency,
                "search_latency": retrieval_context.metrics.search_latency,
                "chunks_returned": len(results)
            }
        }
