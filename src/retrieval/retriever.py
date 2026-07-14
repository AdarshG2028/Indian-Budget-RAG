"""
Abstract retriever interface and dense retrieval implementation.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import time

from .models import RetrievalResult, RetrievalConfig, RetrievalContext, RetrievalMetrics
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract interface for retrieval operations.
    
    This abstraction allows different retrieval strategies (dense, hybrid, sparse)
    to be used interchangeably without changing application code.
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalContext:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query text
            config: Retrieval configuration
            
        Returns:
            RetrievalContext with results and metrics
        """
        pass
    
    @abstractmethod
    def retrieve_with_scores(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalContext:
        """
        Retrieve relevant chunks with similarity scores.
        
        Args:
            query: User query text
            config: Retrieval configuration
            
        Returns:
            RetrievalContext with results and metrics
        """
        pass


class DenseRetriever(BaseRetriever):
    """
    Dense vector retriever using semantic similarity.
    
    This retriever:
    - Embeds the query using the provided embedder
    - Performs dense vector search via the vector store
    - Applies metadata filtering
    - Returns typed retrieval results with context
    
    Pipeline hooks (for future extension):
    - query_preprocessor: Optional query preprocessing/expansion
    - reranker: Optional cross-encoder reranking
    """
    
    def __init__(
        self,
        embedder,
        vector_store: VectorStore,
        collection_name: str,
        query_preprocessor: Optional[callable] = None,
        reranker: Optional[callable] = None
    ):
        """
        Initialize the dense retriever.
        
        Args:
            embedder: Embedder instance with embed_query() method
            vector_store: Vector store instance
            collection_name: Name of the collection to search
            query_preprocessor: Optional query preprocessing function
            reranker: Optional reranking function
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.query_preprocessor = query_preprocessor
        self.reranker = reranker
        
        # Verify collection exists
        if not self.vector_store.collection_exists():
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                "Please run the embedding pipeline first."
            )
        
        logger.info(f"DenseRetriever initialized for collection '{self.collection_name}'")
    
    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalContext:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query text
            config: Retrieval configuration
            
        Returns:
            RetrievalContext with results and metrics
        """
        return self.retrieve_with_scores(query, config)
    
    def retrieve_with_scores(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalContext:
        """
        Retrieve relevant chunks with similarity scores.
        
        Args:
            query: User query text
            config: Retrieval configuration
            
        Returns:
            RetrievalContext with results and metrics
        """
        if config is None:
            config = RetrievalConfig()
        
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Pipeline hook: Query preprocessing
            if self.query_preprocessor:
                query = self.query_preprocessor(query)
                logger.debug("Query preprocessed")
            
            # Step 1: Embed the query
            embed_start = time.time()
            query_vector = self.embedder.embed_query(query)
            embed_time = time.time() - embed_start
            logger.debug(f"Query embedding took {embed_time:.3f}s")
            
            # Step 2: Perform dense search
            search_start = time.time()
            results = self.vector_store.search_dense(
                query_vector=query_vector,
                top_k=config.top_k,
                filters=config.filters,
                score_threshold=config.score_threshold
            )
            search_time = time.time() - search_start
            logger.debug(f"Vector search took {search_time:.3f}s")
            
            # Pipeline hook: Reranking
            if self.reranker:
                results = self.reranker(query, results)
                logger.debug("Results reranked")
            
            # Calculate metrics
            total_time = time.time() - start_time
            
            if results:
                scores = [r.score for r in results]
                metrics = RetrievalMetrics(
                    avg_similarity=sum(scores) / len(scores),
                    highest_similarity=max(scores),
                    lowest_similarity=min(scores),
                    chunks_returned=len(results),
                    retrieval_latency=total_time,
                    embedding_latency=embed_time,
                    search_latency=search_time
                )
            else:
                metrics = RetrievalMetrics(
                    avg_similarity=0.0,
                    highest_similarity=0.0,
                    lowest_similarity=0.0,
                    chunks_returned=0,
                    retrieval_latency=total_time,
                    embedding_latency=embed_time,
                    search_latency=search_time
                )
            
            # Create context
            context = RetrievalContext(
                query=query,
                top_k=config.top_k,
                results=results,
                metrics=metrics,
                filters=config.filters,
                score_threshold=config.score_threshold
            )
            
            logger.info(
                f"Retrieval completed in {total_time:.3f}s "
                f"(embed: {embed_time:.3f}s, search: {search_time:.3f}s), "
                f"retrieved {len(results)} chunks"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
