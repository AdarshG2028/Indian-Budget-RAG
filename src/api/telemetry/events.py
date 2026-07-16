"""
Event definitions for OpenTelemetry tracing.

Defines standard event names for RAG operations to make traces more understandable.
"""


class SpanEvents:
    """
    Standard event names for RAG operations.
    
    Events provide additional context within spans and help understand
    the flow of operations.
    """
    
    # Retrieval events
    RETRIEVAL_STARTED = "rag.retrieval.started"
    RETRIEVAL_COMPLETED = "rag.retrieval.completed"
    EMBEDDING_STARTED = "rag.retrieval.embedding.started"
    EMBEDDING_COMPLETED = "rag.retrieval.embedding.completed"
    SEARCH_STARTED = "rag.retrieval.search.started"
    SEARCH_COMPLETED = "rag.retrieval.search.completed"
    
    # Reranking events
    RERANKING_STARTED = "rag.reranking.started"
    RERANKING_COMPLETED = "rag.reranking.completed"
    RERANKING_FALLBACK = "rag.reranking.fallback"
    
    # Context building events
    CONTEXT_BUILDING_STARTED = "rag.context.building.started"
    CONTEXT_BUILDING_COMPLETED = "rag.context.building.completed"
    CONTEXT_TOKEN_LIMIT_REACHED = "rag.context.token_limit_reached"
    
    # LLM events
    GENERATION_STARTED = "rag.llm.generation.started"
    FIRST_TOKEN_GENERATED = "rag.llm.first_token_generated"
    GENERATION_COMPLETED = "rag.llm.generation.completed"
    PROMPT_CONSTRUCTED = "rag.llm.prompt_constructed"
    
    # Streaming events
    STREAM_OPENED = "rag.streaming.opened"
    STREAM_CLOSED = "rag.streaming.closed"
    CLIENT_DISCONNECTED = "rag.streaming.client_disconnected"
    
    # Evaluation events
    EVALUATION_STARTED = "rag.evaluation.started"
    EVALUATION_COMPLETED = "rag.evaluation.completed"
    EVALUATION_FAILED = "rag.evaluation.failed"
    
    # General events
    REQUEST_RECEIVED = "request.received"
    RESPONSE_SENT = "response.sent"
    ERROR_OCCURRED = "error.occurred"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"


def get_retrieval_started_attributes(
    query: str,
    top_k: int,
    embedding_model: str
) -> dict:
    """
    Get attributes for retrieval started event.
    
    Args:
        query: Search query
        top_k: Number of chunks to retrieve
        embedding_model: Embedding model name
        
    Returns:
        Event attributes
    """
    return {
        "query": query,
        "top_k": top_k,
        "embedding_model": embedding_model,
    }


def get_retrieval_completed_attributes(
    retrieved_count: int,
    retrieval_time: float,
    average_score: float
) -> dict:
    """
    Get attributes for retrieval completed event.
    
    Args:
        retrieved_count: Number of chunks retrieved
        retrieval_time: Retrieval time in milliseconds
        average_score: Average similarity score
        
    Returns:
        Event attributes
    """
    return {
        "retrieved_count": retrieved_count,
        "retrieval_time_ms": retrieval_time,
        "average_score": average_score,
    }


def get_reranking_started_attributes(
    chunks_count: int,
    reranker_model: str
) -> dict:
    """
    Get attributes for reranking started event.
    
    Args:
        chunks_count: Number of chunks to rerank
        reranker_model: Reranker model name
        
    Returns:
        Event attributes
    """
    return {
        "chunks_count": chunks_count,
        "reranker_model": reranker_model,
    }


def get_reranking_completed_attributes(
    reranked_count: int,
    reranking_time: float,
    score_improvement: float
) -> dict:
    """
    Get attributes for reranking completed event.
    
    Args:
        reranked_count: Number of chunks after reranking
        reranking_time: Reranking time in milliseconds
        score_improvement: Average score improvement
        
    Returns:
        Event attributes
    """
    return {
        "reranked_count": reranked_count,
        "reranking_time_ms": reranking_time,
        "score_improvement": score_improvement,
    }


def get_generation_started_attributes(
    model: str,
    prompt_tokens: int,
    streaming: bool
) -> dict:
    """
    Get attributes for generation started event.
    
    Args:
        model: LLM model name
        prompt_tokens: Number of prompt tokens
        streaming: Whether streaming is enabled
        
    Returns:
        Event attributes
    """
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "streaming": streaming,
    }


def get_first_token_attributes(
    time_to_first_token: float
) -> dict:
    """
    Get attributes for first token generated event.
    
    Args:
        time_to_first_token: Time to first token in milliseconds
        
    Returns:
        Event attributes
    """
    return {
        "time_to_first_token_ms": time_to_first_token,
    }


def get_generation_completed_attributes(
    completion_tokens: int,
    total_tokens: int,
    generation_time: float
) -> dict:
    """
    Get attributes for generation completed event.
    
    Args:
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens
        generation_time: Generation time in milliseconds
        
    Returns:
        Event attributes
    """
    return {
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "generation_time_ms": generation_time,
    }


def get_streaming_attributes(
    token_count: int,
    stream_duration: float
) -> dict:
    """
    Get attributes for streaming events.
    
    Args:
        token_count: Number of tokens streamed
        stream_duration: Stream duration in milliseconds
        
    Returns:
        Event attributes
    """
    return {
        "token_count": token_count,
        "stream_duration_ms": stream_duration,
        "tokens_per_second": (token_count / stream_duration) * 1000 if stream_duration > 0 else 0,
    }
