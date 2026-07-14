"""
Strongly typed models for LLM operations.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterator


@dataclass
class LLMConfig:
    """
    Configuration for LLM operations.
    """
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("Top-p must be between 0 and 1")


@dataclass
class LLMResponse:
    """
    Response from LLM generation.
    """
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for logging/export."""
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency": self.latency,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata or {}
        }


@dataclass
class LLMStreamChunk:
    """
    Single chunk from streaming LLM response.
    """
    text: str
    delta: str
    is_complete: bool
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMMetrics:
    """
    Metrics for LLM operations observability.
    """
    prompt_construction_time: float
    llm_latency: float
    total_latency: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/export."""
        return {
            "prompt_construction_time": self.prompt_construction_time,
            "llm_latency": self.llm_latency,
            "total_latency": self.total_latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model
        }
