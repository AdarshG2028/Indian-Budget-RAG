"""
Groq LLM implementation.
"""
import logging
import time
from typing import Iterator

from .base_llm import BaseLLM
from .models import LLMConfig, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)

try:
    from src.observability import get_telemetry
except ImportError:
    from observability import get_telemetry
telemetry = get_telemetry()


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
        
        # Start LLM API call span
        llm_api_span = None
        if telemetry:
            llm_api_span = telemetry.start_span(
                "llm.groq_api_call",
                attributes={
                    "llm.provider": "groq",
                    "llm.model": self.config.model_name,
                    "llm.temperature": self.config.temperature,
                    "llm.max_tokens": self.config.max_tokens,
                    "llm.prompt_length": len(prompt),
                    "llm.streaming": False
                }
            )
            telemetry.record_event("llm.api_call.started")
        
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
            
            if telemetry:
                telemetry.set_span_attribute(llm_api_span, "llm.latency", latency)
                telemetry.set_span_attribute(llm_api_span, "llm.prompt_tokens", prompt_tokens)
                telemetry.set_span_attribute(llm_api_span, "llm.completion_tokens", completion_tokens)
                telemetry.set_span_attribute(llm_api_span, "llm.total_tokens", total_tokens)
                telemetry.set_span_attribute(llm_api_span, "llm.finish_reason", finish_reason)
                telemetry.record_event("llm.api_call.completed", {
                    "llm.latency": latency,
                    "llm.prompt_tokens": prompt_tokens,
                    "llm.completion_tokens": completion_tokens,
                    "llm.total_tokens": total_tokens,
                    "llm.finish_reason": finish_reason
                })
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            if telemetry:
                telemetry.record_exception(e, llm_api_span)
                telemetry.record_event("llm.api_call.failed")
            raise
        finally:
            if telemetry and llm_api_span:
                telemetry.end_span(llm_api_span)
    
    def stream(self, prompt: str) -> Iterator[LLMStreamChunk]:
        """
        Stream response chunks for the given prompt.
        
        Args:
            prompt: Input prompt for the LLM
            
        Yields:
            LLMStreamChunk objects with incremental text
        """
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        # Start LLM streaming span
        llm_stream_span = None
        if telemetry:
            llm_stream_span = telemetry.start_span(
                "llm.groq_streaming",
                attributes={
                    "llm.provider": "groq",
                    "llm.model": self.config.model_name,
                    "llm.temperature": self.config.temperature,
                    "llm.max_tokens": self.config.max_tokens,
                    "llm.prompt_length": len(prompt),
                    "llm.streaming": True
                }
            )
            telemetry.record_event("llm.streaming.started")
        
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
                
                # Track first token latency
                if delta and first_token_time is None:
                    first_token_time = time.time() - start_time
                    if telemetry:
                        telemetry.record_event("llm.streaming.first_token", {
                            "llm.first_token_latency": first_token_time
                        })
                
                if delta:
                    token_count += 1
                    full_text += delta
                
                yield LLMStreamChunk(
                    text=full_text,
                    delta=delta,
                    is_complete=chunk.choices[0].finish_reason is not None,
                    metadata={"provider": "groq"}
                )
                
                if chunk.choices[0].finish_reason:
                    total_time = time.time() - start_time
                    if telemetry:
                        telemetry.record_event("llm.streaming.completed", {
                            "llm.total_latency": total_time,
                            "llm.first_token_latency": first_token_time or 0,
                            "llm.tokens_streamed": token_count,
                            "llm.tokens_per_second": token_count / total_time if total_time > 0 else 0
                        })
                    break
                    
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            if telemetry:
                telemetry.record_exception(e, llm_stream_span)
                telemetry.record_event("llm.streaming.failed")
            raise
        finally:
            if telemetry and llm_stream_span:
                total_time = time.time() - start_time
                telemetry.set_span_attribute(llm_stream_span, "llm.total_latency", total_time)
                telemetry.set_span_attribute(llm_stream_span, "llm.first_token_latency", first_token_time or 0)
                telemetry.set_span_attribute(llm_stream_span, "llm.tokens_streamed", token_count)
                telemetry.set_span_attribute(llm_stream_span, "llm.tokens_per_second", token_count / total_time if total_time > 0 else 0)
                telemetry.end_span(llm_stream_span)
    
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
