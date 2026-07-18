"""
Abstract retriever interface and dense retrieval implementation.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import time

from .models import RetrievalResult, RetrievalConfig, RetrievalContext, RetrievalMetrics
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


try:
    from src.observability import get_telemetry
except ImportError:
    from observability import get_telemetry
telemetry = get_telemetry()


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
        
        # Start Dense Retrieval span
        retrieval_span = None
        if telemetry:
            retrieval_span = telemetry.start_span(
                "retrieval.dense",
                attributes={
                    "retrieval.query_length": len(query),
                    "retrieval.top_k": config.top_k,
                    "retrieval.collection": self.collection_name
                }
            )
            telemetry.record_event("retrieval.started")
        
        try:
            # Pipeline hook: Query preprocessing
            if self.query_preprocessor:
                query = self.query_preprocessor(query)
                logger.debug("Query preprocessed")
            
            # Step 1: Embed the query
            embed_span = None
            if telemetry:
                embed_span = telemetry.start_span(
                    "retrieval.embed_query",
                    attributes={
                        "embedding.model": getattr(self.embedder, 'model_name', 'unknown'),
                        "embedding.query_length": len(query)
                    }
                )
                telemetry.record_event("retrieval.embed.started")
            
            embed_start = time.time()
            try:
                query_vector = self.embedder.embed_query(query)
                embed_time = time.time() - embed_start
                logger.debug(f"Query embedding took {embed_time:.3f}s")
                
                if telemetry:
                    telemetry.set_span_attribute(embed_span, "embedding.latency", embed_time)
                    telemetry.record_event("retrieval.embed.completed", {
                        "embedding.latency": embed_time
                    })
            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
                if telemetry:
                    telemetry.record_exception(e, embed_span)
                    telemetry.record_event("retrieval.embed.failed")
                raise
            finally:
                if telemetry and embed_span:
                    telemetry.end_span(embed_span)
            
            # Step 2: Perform dense search
            search_span = None
            if telemetry:
                search_span = telemetry.start_span(
                    "retrieval.vector_search",
                    attributes={
                        "search.collection": self.collection_name,
                        "search.top_k": config.top_k,
                        "search.score_threshold": config.score_threshold or 0.0
                    }
                )
                telemetry.record_event("retrieval.search.started")
            
            search_start = time.time()
            try:
                results = self.vector_store.search_dense(
                    query_vector=query_vector,
                    top_k=config.top_k,
                    filters=config.filters,
                    score_threshold=config.score_threshold
                )
                search_time = time.time() - search_start
                logger.debug(f"Vector search took {search_time:.3f}s")
                
                if telemetry:
                    telemetry.set_span_attribute(search_span, "search.latency", search_time)
                    telemetry.set_span_attribute(search_span, "search.results_count", len(results))
                    telemetry.record_event("retrieval.search.completed", {
                        "search.latency": search_time,
                        "search.results_count": len(results)
                    })
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                if telemetry:
                    telemetry.record_exception(e, search_span)
                    telemetry.record_event("retrieval.search.failed")
                raise
            finally:
                if telemetry and search_span:
                    telemetry.end_span(search_span)
            
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
            
            if telemetry:
                telemetry.set_span_attribute(retrieval_span, "retrieval.latency", total_time)
                telemetry.set_span_attribute(retrieval_span, "retrieval.chunks_returned", len(results))
                telemetry.set_span_attribute(retrieval_span, "retrieval.avg_similarity", metrics.avg_similarity)
                telemetry.record_event("retrieval.completed", {
                    "retrieval.latency": total_time,
                    "retrieval.chunks_returned": len(results),
                    "retrieval.avg_similarity": metrics.avg_similarity
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            if telemetry:
                telemetry.record_exception(e, retrieval_span)
                telemetry.record_event("retrieval.failed")
            raise
        finally:
            if telemetry and retrieval_span:
                telemetry.end_span(retrieval_span)
