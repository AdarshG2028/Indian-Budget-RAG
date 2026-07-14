"""
Groq LLM implementation.
"""
import logging
import time
from typing import Iterator

from .base_llm import BaseLLM
from .models import LLMConfig, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    Groq API implementation of the LLM interface.
    
    Supports:
    - Multiple models (Llama 3, Mixtral, Gemma)
    - Streaming responses
    - Token counting
    - Error handling and retries
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Groq LLM client.
        
        Args:
            config: LLM configuration with Groq API key
        """
        super().__init__(config)
        
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package is required. Install it with: uv add groq"
            )
        
        if not config.api_key:
            raise ValueError("Groq API key is required in config")
        
        self.client = Groq(api_key=config.api_key)
        logger.info(f"Groq LLM initialized with model: {config.model_name}")
    
    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate a complete response for the given prompt.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout
            )
            
            latency = time.time() - start_time
            
            # Extract response data
            text = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason
            
            llm_response = LLMResponse(
                text=text,
                model=self.config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency=latency,
                finish_reason=finish_reason,
                metadata={"provider": "groq"}
            )
            
            logger.info(
                f"Groq generation completed in {latency:.3f}s, "
                f"{total_tokens} tokens ({prompt_tokens} prompt, {completion_tokens} completion)"
            )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def stream(self, prompt: str) -> Iterator[LLMStreamChunk]:
        """
        Stream response chunks for the given prompt.
        
        Args:
            prompt: Input prompt for the LLM
            
        Yields:
            LLMStreamChunk objects with incremental text
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
                timeout=self.config.timeout
            )
            
            full_text = ""
            
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
                
                yield LLMStreamChunk(
                    text=full_text,
                    delta=delta,
                    is_complete=chunk.choices[0].finish_reason is not None,
                    metadata={"provider": "groq"}
                )
                
                if chunk.choices[0].finish_reason:
                    break
                    
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate that the Groq configuration is valid and the service is accessible.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Test with a minimal request
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            logger.info("Groq configuration validated successfully")
            return True
        except Exception as e:
            logger.error(f"Groq configuration validation failed: {e}")
            return False
