"""
Abstract interface for LLM operations.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .models import LLMConfig, LLMResponse, LLMStreamChunk


class BaseLLM(ABC):
    """
    Abstract interface for LLM operations.
    
    This abstraction allows switching between different LLM providers
    (Groq, OpenAI, Gemini, Ollama, vLLM) without changing the RAG pipeline.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM with configuration.
        
        Args:
            config: LLM configuration
        """
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate a complete response for the given prompt.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            LLMResponse with generated text and metadata
        """
        pass
    
    @abstractmethod
    def stream(self, prompt: str) -> Iterator[LLMStreamChunk]:
        """
        Stream response chunks for the given prompt.
        
        Args:
            prompt: Input prompt for the LLM
            
        Yields:
            LLMStreamChunk objects with incremental text
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the LLM configuration is valid and the service is accessible.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
