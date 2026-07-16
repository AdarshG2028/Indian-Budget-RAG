"""
Streaming service for transport-agnostic event streaming.
"""
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..events import (
    EventType,
    RetrievalStartedEvent,
    RetrievalCompletedEvent,
    RerankingStartedEvent,
    RerankingCompletedEvent,
    ContextReadyEvent,
    GenerationStartedEvent,
    TokenEvent,
    CitationEvent,
    GenerationCompletedEvent,
    ErrorEvent
)

logger = logging.getLogger(__name__)


class StreamEventEmitter(ABC):
    """
    Abstract base class for stream event emission.
    
    This allows for different transport mechanisms (SSE, WebSocket)
    without changing the streaming logic.
    """
    
    @abstractmethod
    async def emit(self, event) -> None:
        """Emit a stream event."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the stream."""
        pass


class SSEEventEmitter(StreamEventEmitter):
    """
    Server-Sent Events (SSE) emitter.
    
    Emits events in SSE format for HTTP streaming.
    """
    
    def __init__(self):
        """Initialize SSE event emitter."""
        self._queue = []
    
    async def emit(self, event) -> None:
        """
        Emit an event in SSE format.
        
        Args:
            event: Stream event to emit
        """
        sse_data = event.to_sse()
        self._queue.append(sse_data)
    
    async def close(self) -> None:
        """Close the SSE stream."""
        pass
    
    def get_events(self) -> list[str]:
        """Get accumulated SSE events."""
        return self._queue


class StreamingService:
    """
    Service for managing streaming operations.
    
    Provides a unified interface for streaming RAG operations
    with structured events.
    """
    
    def __init__(self, request_id: str):
        """
        Initialize streaming service.
        
        Args:
            request_id: Unique request identifier
        """
        self.request_id = request_id
        self.emitter = SSEEventEmitter()
    
    async def emit_retrieval_started(self, query: str, top_k: int) -> None:
        """Emit retrieval started event."""
        event = RetrievalStartedEvent(
            request_id=self.request_id,
            data={"query": query, "top_k": top_k}
        )
        await self.emitter.emit(event)
    
    async def emit_retrieval_completed(
        self,
        chunks_retrieved: int,
        latency: float,
        top_chunks: list[Dict[str, Any]]
    ) -> None:
        """Emit retrieval completed event."""
        event = RetrievalCompletedEvent(
            request_id=self.request_id,
            data={
                "chunks_retrieved": chunks_retrieved,
                "latency": latency,
                "top_chunks": top_chunks[:3]  # Send top 3 for preview
            }
        )
        await self.emitter.emit(event)
    
    async def emit_reranking_started(self, model: str, chunks_to_rerank: int) -> None:
        """Emit reranking started event."""
        event = RerankingStartedEvent(
            request_id=self.request_id,
            data={"model": model, "chunks_to_rerank": chunks_to_rerank}
        )
        await self.emitter.emit(event)
    
    async def emit_reranking_completed(
        self,
        latency: float,
        chunks_reranked: int,
        fallback_used: bool
    ) -> None:
        """Emit reranking completed event."""
        event = RerankingCompletedEvent(
            request_id=self.request_id,
            data={
                "latency": latency,
                "chunks_reranked": chunks_reranked,
                "fallback_used": fallback_used
            }
        )
        await self.emitter.emit(event)
    
    async def emit_context_ready(self, context_tokens: int, citations_count: int) -> None:
        """Emit context ready event."""
        event = ContextReadyEvent(
            request_id=self.request_id,
            data={
                "context_tokens": context_tokens,
                "citations_count": citations_count
            }
        )
        await self.emitter.emit(event)
    
    async def emit_generation_started(self, model: str, prompt_tokens: int) -> None:
        """Emit generation started event."""
        event = GenerationStartedEvent(
            request_id=self.request_id,
            data={"model": model, "prompt_tokens": prompt_tokens}
        )
        await self.emitter.emit(event)
    
    async def emit_token(
        self,
        token: str,
        index: int,
        is_first: bool = False,
        is_last: bool = False
    ) -> None:
        """Emit token event."""
        event = TokenEvent(
            request_id=self.request_id,
            data={
                "token": token,
                "index": index,
                "is_first": is_first,
                "is_last": is_last
            }
        )
        await self.emitter.emit(event)
    
    async def emit_citation(
        self,
        chunk_id: str,
        document: str,
        year: int,
        section: str,
        page_start: int,
        page_end: int,
        similarity: float
    ) -> None:
        """Emit citation event."""
        event = CitationEvent(
            request_id=self.request_id,
            data={
                "chunk_id": chunk_id,
                "document": document,
                "year": year,
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
                "similarity": similarity
            }
        )
        await self.emitter.emit(event)
    
    async def emit_generation_completed(
        self,
        total_tokens: int,
        completion_tokens: int,
        latency: float
    ) -> None:
        """Emit generation completed event."""
        event = GenerationCompletedEvent(
            request_id=self.request_id,
            data={
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "latency": latency
            }
        )
        await self.emitter.emit(event)
    
    async def emit_error(self, error_code: str, message: str, details: Optional[Dict] = None) -> None:
        """Emit error event."""
        event = ErrorEvent(
            request_id=self.request_id,
            data={
                "error_code": error_code,
                "message": message,
                "details": details
            }
        )
        await self.emitter.emit(event)
    
    async def close(self) -> None:
        """Close the streaming service."""
        await self.emitter.close()
    
    def get_emitter(self) -> StreamEventEmitter:
        """Get the event emitter."""
        return self.emitter
