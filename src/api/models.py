"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    RERANKING_ERROR = "RERANKING_ERROR"
    LLM_ERROR = "LLM_ERROR"
    EVALUATION_ERROR = "EVALUATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(min_length=1, description="User question")
    stream: bool = Field(default=False, description="Enable streaming response")
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration override"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the budget allocation for agriculture?",
                "stream": False,
                "config": {
                    "top_k": 20,
                    "enable_rerank": True
                }
            }
        }
    )


class RetrievalRequest(BaseModel):
    """Request model for retrieval-only endpoint."""
    query: str = Field(min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "budget allocation agriculture",
                "top_k": 20,
                "score_threshold": 0.5
            }
        }
    )


class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint."""
    dataset: str = Field(description="Path to evaluation dataset")
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evaluation configuration"
    )
    save_experiment: bool = Field(default=True, description="Save experiment")
    output_formats: List[Literal["json", "csv", "markdown"]] = Field(
        default=["json", "markdown"],
        description="Output formats"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset": "data/evaluation/sample_queries.json",
                "config": {
                    "top_k": 10,
                    "enable_rerank": True
                },
                "save_experiment": True,
                "output_formats": ["json", "markdown"]
            }
        }
    )


class Citation(BaseModel):
    """Citation model."""
    chunk_id: str
    document: str
    year: int
    section: str
    page_start: int
    page_end: int
    similarity: float


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    request_id: str
    answer: str
    citations: Optional[List[Citation]] = None
    metrics: Dict[str, Any]
    reranker_used: bool
    reranker_metrics: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "answer": "The budget allocation for agriculture is...",
                "citations": [],
                "metrics": {
                    "retrieval_latency": 0.5,
                    "llm_latency": 2.3,
                    "total_latency": 3.0
                },
                "reranker_used": True,
                "reranker_metrics": {
                    "reranking_latency": 0.3,
                    "avg_score_improvement": 0.15
                }
            }
        }
    )


class RetrievalResult(BaseModel):
    """Single retrieval result."""
    chunk_id: str
    document: str
    year: int
    section: str
    subsection: Optional[str]
    text: str
    score: float
    rank: int
    page_start: int
    page_end: int


class RetrievalResponse(BaseModel):
    """Response model for retrieval endpoint."""
    request_id: str
    query: str
    results: List[RetrievalResult]
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response model."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "error_code": "RETRIEVAL_ERROR",
                "message": "Failed to retrieve documents",
                "details": {
                    "query": "budget allocation agriculture",
                    "error": "Connection timeout"
                }
            }
        }
    )
    
    request_id: str
    error_code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy", "degraded"]
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, Any]


class JobStatus(str, Enum):
    """Background job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationJobResponse(BaseModel):
    """Response model for evaluation job."""
    request_id: str
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
