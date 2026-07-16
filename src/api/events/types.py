"""
Stream event types for structured SSE streaming.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Stream event types."""
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RERANKING_STARTED = "reranking_started"
    RERANKING_COMPLETED = "reranking_completed"
    CONTEXT_READY = "context_ready"
    GENERATION_STARTED = "generation_started"
    TOKEN = "token"
    CITATION = "citation"
    GENERATION_COMPLETED = "generation_completed"
    ERROR = "error"


class StreamEvent(BaseModel):
    """Base stream event model."""
    event: EventType
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[Dict[str, Any]] = None
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Event format."""
        event_str = f"event: {self.event.value}\n"
        event_str += f"data: {self.model_dump_json(exclude_none=True)}\n\n"
        return event_str


class RetrievalStartedEvent(StreamEvent):
    """Event emitted when retrieval starts."""
    event: Literal[EventType.RETRIEVAL_STARTED] = EventType.RETRIEVAL_STARTED


class RetrievalCompletedEvent(StreamEvent):
    """Event emitted when retrieval completes."""
    event: Literal[EventType.RETRIEVAL_COMPLETED] = EventType.RETRIEVAL_COMPLETED


class RerankingStartedEvent(StreamEvent):
    """Event emitted when reranking starts."""
    event: Literal[EventType.RERANKING_STARTED] = EventType.RERANKING_STARTED


class RerankingCompletedEvent(StreamEvent):
    """Event emitted when reranking completes."""
    event: Literal[EventType.RERANKING_COMPLETED] = EventType.RERANKING_COMPLETED


class ContextReadyEvent(StreamEvent):
    """Event emitted when context is ready."""
    event: Literal[EventType.CONTEXT_READY] = EventType.CONTEXT_READY


class GenerationStartedEvent(StreamEvent):
    """Event emitted when LLM generation starts."""
    event: Literal[EventType.GENERATION_STARTED] = EventType.GENERATION_STARTED


class TokenEvent(StreamEvent):
    """Event emitted for each generated token."""
    event: Literal[EventType.TOKEN] = EventType.TOKEN


class CitationEvent(StreamEvent):
    """Event emitted when citations are available."""
    event: Literal[EventType.CITATION] = EventType.CITATION


class GenerationCompletedEvent(StreamEvent):
    """Event emitted when generation completes."""
    event: Literal[EventType.GENERATION_COMPLETED] = EventType.GENERATION_COMPLETED


class ErrorEvent(StreamEvent):
    """Event emitted on error."""
    event: Literal[EventType.ERROR] = EventType.ERROR
