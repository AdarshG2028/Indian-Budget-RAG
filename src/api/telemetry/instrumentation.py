"""
Business logic instrumentation for RAG operations.

Integrates TelemetryManager with RAG pipeline components for comprehensive observability.
"""
import logging
import time
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import contextmanager

from .manager import get_telemetry_manager
from .attributes import (
    SpanAttributes,
    get_retrieval_attributes,
    get_reranking_attributes,
    get_context_attributes,
    get_llm_attributes,
    get_streaming_attributes,
    get_error_attributes
)
from .events import (
    SpanEvents,
    get_retrieval_started_attributes,
    get_retrieval_completed_attributes,
    get_reranking_started_attributes,
    get_reranking_completed_attributes,
    get_generation_started_attributes,
    get_first_token_attributes,
    get_generation_completed_attributes
)
from .metrics import MetricsHelper

logger = logging.getLogger(__name__)


class RAGInstrumentation:
    """
    Instrumentation for RAG pipeline operations.
    
    Provides methods to instrument each component of the RAG pipeline
    with proper spans, events, and metrics.
    """
    
    def __init__(self):
        """Initialize RAG instrumentation."""
        self.tm = get_telemetry_manager()
        self.metrics = MetricsHelper(self.tm) if self.tm.is_initialized() else None
    
    def instrument_retrieval(
        self,
        query: str,
        embedding_model: str,
        top_k: int,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None
    ):
        """
        Context manager for instrumenting retrieval operations.
        
        Args:
            query: Search query
            embedding_model: Embedding model name
            top_k: Number of chunks to retrieve
            score_threshold: Optional score threshold
            collection_name: Optional collection name
            
        Yields:
            Instrumentation context for recording results
        """
        if not self.tm.is_initialized():
            yield None
            return
        
        with self.tm.start_span(
            "rag.retrieval",
            kind="internal"
        ) as span:
            if span:
                # Set initial attributes
                span.set_attribute(SpanAttributes.EMBEDDING_MODEL, embedding_model)
                span.set_attribute(SpanAttributes.TOP_K, top_k)
                if score_threshold:
                    span.set_attribute(SpanAttributes.SCORE_THRESHOLD, score_threshold)
                if collection_name:
                    span.set_attribute(SpanAttributes.COLLECTION_NAME, collection_name)
                
                # Record retrieval started event
                self.tm.record_event(
                    SpanEvents.RETRIEVAL_STARTED,
                    get_retrieval_started_attributes(query, top_k, embedding_model)
                )
            
            # Yield context for recording results
            class RetrievalContext:
                def __init__(self, instrumentation):
                    self.instrumentation = instrumentation
                
                def record_embedding(self, embedding_latency_ms: float):
                    """Record embedding generation."""
                    if self.instrumentation.tm.is_initialized():
                        self.instrumentation.tm.record_event(
                            SpanEvents.EMBEDDING_COMPLETED,
                            {"embedding_latency_ms": embedding_latency_ms}
                        )
                
                def record_search(self, search_latency_ms: float):
                    """Record vector search."""
                    if self.instrumentation.tm.is_initialized():
                        self.instrumentation.tm.record_event(
                            SpanEvents.SEARCH_COMPLETED,
                            {"search_latency_ms": search_latency_ms}
                        )
                
                def record_completion(
                    self,
                    retrieved_chunks: int,
                    average_similarity: float,
                    retrieval_latency_ms: float
                ):
                    """Record retrieval completion."""
                    if self.instrumentation.tm.is_initialized():
                        # Set completion attributes
                        attributes = get_retrieval_attributes(
                            embedding_model=embedding_model,
                            top_k=top_k,
                            retrieved_chunks=retrieved_chunks,
                            average_similarity=average_similarity,
                            retrieval_latency=retrieval_latency_ms,
                            score_threshold=score_threshold,
                            collection_name=collection_name
                        )
                        for key, value in attributes.items():
                            self.instrumentation.tm.set_span_attribute(key, value)
                        
                        # Record completion event
                        self.instrumentation.tm.record_event(
                            SpanEvents.RETRIEVAL_COMPLETED,
                            get_retrieval_completed_attributes(
                                retrieved_chunks, retrieval_latency_ms, average_similarity
                            )
                        )
                        
                        # Record metrics
                        if self.instrumentation.metrics:
                            self.instrumentation.metrics.record_retrieval(
                                embedding_model=embedding_model,
                                top_k=top_k,
                                retrieved_count=retrieved_chunks,
                                embedding_latency_ms=retrieval_latency_ms * 0.4,  # Approximate
                                retrieval_latency_ms=retrieval_latency_ms,
                                success=True
                            )
            
            yield RetrievalContext(self)
    
    def instrument_reranking(
        self,
        reranker_model: str,
        rerank_top_k: int
    ):
        """
        Context manager for instrumenting reranking operations.
        
        Args:
            reranker_model: Reranker model name
            rerank_top_k: Number of chunks to rerank
            
        Yields:
            Instrumentation context for recording results
        """
        if not self.tm.is_initialized():
            yield None
            return
        
        with self.tm.start_span(
            "rag.reranking",
            kind="internal"
        ) as span:
            if span:
                # Set initial attributes
                span.set_attribute(SpanAttributes.RERANKER_MODEL, reranker_model)
                span.set_attribute(SpanAttributes.RERANK_TOP_K, rerank_top_k)
                
                # Record reranking started event
                self.tm.record_event(
                    SpanEvents.RERANKING_STARTED,
                    get_reranking_started_attributes(rerank_top_k, reranker_model)
                )
            
            # Yield context for recording results
            class RerankingContext:
                def __init__(self, instrumentation):
                    self.instrumentation = instrumentation
                
                def record_completion(
                    self,
                    reordered_chunks: int,
                    reranking_latency_ms: float,
                    score_improvement: float,
                    fallback_used: bool = False,
                    fallback_count: int = 0
                ):
                    """Record reranking completion."""
                    if self.instrumentation.tm.is_initialized():
                        # Set completion attributes
                        attributes = get_reranking_attributes(
                            reranker_model=reranker_model,
                            rerank_top_k=rerank_top_k,
                            reordered_chunks=reordered_chunks,
                            reranking_latency=reranking_latency_ms,
                            fallback_used=fallback_used,
                            fallback_count=fallback_count
                        )
                        for key, value in attributes.items():
                            self.instrumentation.tm.set_span_attribute(key, value)
                        
                        # Record completion event
                        self.instrumentation.tm.record_event(
                            SpanEvents.RERANKING_COMPLETED,
                            get_reranking_completed_attributes(
                                reordered_chunks, reranking_latency_ms, score_improvement
                            )
                        )
                        
                        # Record metrics
                        if self.instrumentation.metrics:
                            self.instrumentation.metrics.record_reranking(
                                reranker_model=reranker_model,
                                chunks_count=rerank_top_k,
                                reranking_latency_ms=reranking_latency_ms,
                                success=not fallback_used
                            )
            
            yield RerankingContext(self)
    
    def instrument_context_building(
        self,
        max_tokens: int
    ):
        """
        Context manager for instrumenting context building operations.
        
        Args:
            max_tokens: Maximum token limit
            
        Yields:
            Instrumentation context for recording results
        """
        if not self.tm.is_initialized():
            yield None
            return
        
        with self.tm.start_span(
            "rag.context_building",
            kind="internal"
        ) as span:
            if span:
                span.set_attribute(SpanAttributes.MAX_TOKENS, max_tokens)
                self.tm.record_event(SpanEvents.CONTEXT_BUILDING_STARTED, {})
            
            class ContextBuildingContext:
                def __init__(self, instrumentation):
                    self.instrumentation = instrumentation
                
                def record_completion(
                    self,
                    context_tokens: int,
                    context_chunks: int,
                    context_building_latency_ms: float
                ):
                    """Record context building completion."""
                    if self.instrumentation.tm.is_initialized():
                        attributes = get_context_attributes(
                            context_tokens=context_tokens,
                            context_chunks=context_chunks,
                            max_tokens=max_tokens,
                            context_building_latency=context_building_latency_ms
                        )
                        for key, value in attributes.items():
                            self.instrumentation.tm.set_span_attribute(key, value)
                        
                        self.instrumentation.tm.record_event(
                            SpanEvents.CONTEXT_BUILDING_COMPLETED,
                            {"context_tokens": context_tokens, "context_chunks": context_chunks}
                        )
            
            yield ContextBuildingContext(self)
    
    def instrument_llm_generation(
        self,
        provider: str,
        model: str,
        streaming: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Context manager for instrumenting LLM generation operations.
        
        Args:
            provider: LLM provider
            model: Model name
            streaming: Whether streaming is enabled
            temperature: Optional temperature
            max_tokens: Optional max tokens
            
        Yields:
            Instrumentation context for recording results
        """
        if not self.tm.is_initialized():
            yield None
            return
        
        with self.tm.start_span(
            "rag.llm_generation",
            kind="client"
        ) as span:
            if span:
                # Set initial attributes
                span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)
                span.set_attribute(SpanAttributes.LLM_MODEL, model)
                span.set_attribute(SpanAttributes.STREAMING_ENABLED, streaming)
                if temperature:
                    span.set_attribute(SpanAttributes.TEMPERATURE, temperature)
                if max_tokens:
                    span.set_attribute(SpanAttributes.MAX_TOKENS, max_tokens)
            
            # Yield context for recording results
            class LLMGenerationContext:
                def __init__(self, instrumentation):
                    self.instrumentation = instrumentation
                    self.start_time = time.time()
                    self.first_token_time = None
                    self.token_count = 0
                
                def record_prompt(self, prompt_tokens: int):
                    """Record prompt construction."""
                    if self.instrumentation.tm.is_initialized():
                        self.instrumentation.tm.record_event(
                            SpanEvents.PROMPT_CONSTRUCTED,
                            {"prompt_tokens": prompt_tokens}
                        )
                        self.instrumentation.tm.set_span_attribute(
                            SpanAttributes.PROMPT_TOKENS, prompt_tokens
                        )
                
                def record_generation_started(self):
                    """Record generation started."""
                    if self.instrumentation.tm.is_initialized():
                        self.instrumentation.tm.record_event(
                            SpanEvents.GENERATION_STARTED,
                            get_generation_started_attributes(model, 0, streaming)
                        )
                
                def record_first_token(self):
                    """Record first token generated."""
                    if self.instrumentation.tm.is_initialized() and self.first_token_time is None:
                        self.first_token_time = time.time()
                        time_to_first_token = (self.first_token_time - self.start_time) * 1000
                        self.instrumentation.tm.record_event(
                            SpanEvents.FIRST_TOKEN_GENERATED,
                            get_first_token_attributes(time_to_first_token)
                        )
                        self.instrumentation.tm.set_span_attribute(
                            SpanAttributes.FIRST_TOKEN_LATENCY, time_to_first_token
                        )
                
                def record_token(self):
                    """Record a token generated."""
                    self.token_count += 1
                    if self.first_token_time is None:
                        self.record_first_token()
                
                def record_completion(
                    self,
                    completion_tokens: int,
                    total_tokens: int,
                    success: bool = True
                ):
                    """Record generation completion."""
                    if self.instrumentation.tm.is_initialized():
                        generation_latency = (time.time() - self.start_time) * 1000
                        
                        attributes = get_llm_attributes(
                            provider=provider,
                            model=model,
                            prompt_tokens=total_tokens - completion_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            generation_latency=generation_latency,
                            streaming_enabled=streaming,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        for key, value in attributes.items():
                            self.instrumentation.tm.set_span_attribute(key, value)
                        
                        self.instrumentation.tm.record_event(
                            SpanEvents.GENERATION_COMPLETED,
                            get_generation_completed_attributes(
                                completion_tokens, total_tokens, generation_latency
                            )
                        )
                        
                        # Record metrics
                        if self.instrumentation.metrics:
                            self.instrumentation.metrics.record_llm_generation(
                                provider=provider,
                                model=model,
                                prompt_tokens=total_tokens - completion_tokens,
                                completion_tokens=completion_tokens,
                                generation_latency_ms=generation_latency,
                                streaming=streaming,
                                success=success
                            )
                        
                        # Record streaming metrics if applicable
                        if streaming and self.first_token_time:
                            stream_duration = (time.time() - self.first_token_time) * 1000
                            self.instrumentation.metrics.record_streaming(
                                first_token_latency_ms=(self.first_token_time - self.start_time) * 1000,
                                token_count=self.token_count,
                                stream_duration_ms=stream_duration
                            )
            
            yield LLMGenerationContext(self)
    
    def instrument_streaming(self):
        """
        Context manager for instrumenting streaming operations.
        
        Yields:
            Instrumentation context for recording streaming events
        """
        if not self.tm.is_initialized():
            yield None
            return
        
        with self.tm.start_span(
            "rag.streaming",
            kind="internal"
        ) as span:
            if span:
                self.tm.record_event(SpanEvents.STREAM_OPENED, {})
            
            class StreamingContext:
                def __init__(self, instrumentation):
                    self.instrumentation = instrumentation
                    self.start_time = time.time()
                    self.first_token_time = None
                    self.token_count = 0
                
                def record_first_token(self):
                    """Record first token."""
                    if self.instrumentation.tm.is_initialized() and self.first_token_time is None:
                        self.first_token_time = time.time()
                        time_to_first_token = (self.first_token_time - self.start_time) * 1000
                        self.instrumentation.tm.record_event(
                            SpanEvents.FIRST_TOKEN_GENERATED,
                            get_first_token_attributes(time_to_first_token)
                        )
                
                def record_token(self):
                    """Record a token."""
                    self.token_count += 1
                    if self.first_token_time is None:
                        self.record_first_token()
                
                def record_completion(self, client_disconnected: bool = False):
                    """Record stream completion."""
                    if self.instrumentation.tm.is_initialized():
                        stream_duration = (time.time() - self.start_time) * 1000
                        first_token_latency = (
                            (self.first_token_time - self.start_time) * 1000
                            if self.first_token_time else None
                        )
                        
                        attributes = get_streaming_attributes(
                            first_token_latency=first_token_latency,
                            token_count=self.token_count,
                            stream_duration=stream_duration,
                            client_disconnected=client_disconnected
                        )
                        for key, value in attributes.items():
                            self.instrumentation.tm.set_span_attribute(key, value)
                        
                        self.instrumentation.tm.record_event(
                            SpanEvents.STREAM_CLOSED,
                            {"token_count": self.token_count, "duration_ms": stream_duration}
                        )
                
                def record_client_disconnected(self):
                    """Record client disconnection."""
                    if self.instrumentation.tm.is_initialized():
                        self.instrumentation.tm.record_event(
                            SpanEvents.CLIENT_DISCONNECTED, {}
                        )
                        self.instrumentation.tm.set_span_attribute(
                            SpanAttributes.CLIENT_DISCONNECTED, True
                        )
            
            yield StreamingContext(self)
    
    def record_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Record an exception with proper telemetry.
        
        Args:
            exception: Exception to record
            context: Additional context information
        """
        if self.tm.is_initialized():
            attributes = get_error_attributes(
                error_type=type(exception).__name__,
                error_message=str(exception),
                stack_trace=str(exception.__traceback__) if exception.__traceback__ else None
            )
            
            if context:
                attributes.update(context)
            
            self.tm.record_exception(exception, attributes)
    
    def record_evaluation(
        self,
        dataset: str,
        experiment_id: str,
        metrics: Dict[str, Any],
        success: bool = True
    ):
        """
        Record evaluation metrics.
        
        Args:
            dataset: Dataset name
            experiment_id: Experiment ID
            metrics: Evaluation metrics
            success: Whether evaluation was successful
        """
        if self.tm.is_initialized():
            from .attributes import get_evaluation_attributes
            attributes = get_evaluation_attributes(dataset, experiment_id, metrics, "success" if success else "failed")
            
            for key, value in attributes.items():
                self.tm.set_span_attribute(key, value)
            
            if self.metrics:
                self.metrics.record_evaluation(dataset, experiment_id, success)


# Global instrumentation instance
_rag_instrumentation = RAGInstrumentation()


def get_rag_instrumentation() -> RAGInstrumentation:
    """
    Get the global RAG instrumentation instance.
    
    Returns:
        RAGInstrumentation instance
    """
    return _rag_instrumentation
