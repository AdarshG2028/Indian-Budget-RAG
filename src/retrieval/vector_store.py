"""
Abstract vector store interface and Qdrant implementation.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import logging

from .models import RetrievalResult
from .filters import FilterBuilder, FilterConverter, FilterCondition

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    Abstract interface for vector store operations.
    
    This abstraction allows switching between different vector databases
    without changing the retrieval layer.
    """
    
    @abstractmethod
    def search_dense(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Union[Dict[str, Any], List[FilterCondition]]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform dense vector search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (dict or FilterCondition list)
            score_threshold: Minimum similarity score
            
        Returns:
            List of RetrievalResult objects
        """
        pass
    
    @abstractmethod
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        pass
    
    @abstractmethod
    def collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        pass


class QdrantVectorStore(VectorStore):
    """
    Qdrant implementation of the vector store interface.
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        distance: str = "Cosine"
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Run: uv add qdrant-client"
            )
        
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }
        self._distance = self._distance_map.get(distance, Distance.COSINE)
        
        logger.info(f"Connecting to Qdrant at {url} …")
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        logger.info("Connected to Qdrant.")
    
    def _normalize_filters(
        self, 
        filters: Optional[Union[Dict[str, Any], List[FilterCondition]]]
    ) -> List[FilterCondition]:
        """
        Normalize filters to FilterCondition list.
        
        Args:
            filters: Either a dict or list of FilterCondition objects
            
        Returns:
            List of FilterCondition objects
        """
        if filters is None:
            return []
        
        if isinstance(filters, list):
            return filters
        
        # Convert dict to FilterCondition list
        return FilterBuilder.from_dict(filters)
    
    def search_dense(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Union[Dict[str, Any], List[FilterCondition]]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform dense vector search in Qdrant.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (dict or FilterCondition list)
            score_threshold: Minimum similarity score
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            from qdrant_client.http import models as rest
        except ImportError:
            raise ImportError("qdrant-client is required")
        
        if not self.collection_exists():
            raise ValueError(f"Collection '{self.collection_name}' does not exist")
        
        # Normalize filters
        filter_conditions = self._normalize_filters(filters)
        
        # Convert to Qdrant filter
        qdrant_filter = FilterConverter.to_qdrant_filter(filter_conditions)
        
        # Perform search
        try:
            # Try newer API first
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold
            ).points
        except AttributeError:
            # Fall back to older API
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold
            )
        
        # Convert to RetrievalResult objects with ranks
        results = []
        for rank, result in enumerate(search_results, 1):
            results.append(
                RetrievalResult.from_qdrant_payload(result.payload, result.score, rank)
            )
        
        logger.info(f"Retrieved {len(results)} results from Qdrant")
        return results
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
    
    def collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
