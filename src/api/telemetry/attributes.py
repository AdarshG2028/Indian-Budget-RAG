"""
Span attribute definitions using OpenTelemetry semantic conventions.

Provides standardized attribute names for all RAG operations.
"""
from typing import Dict, Any, Optional


class SpanAttributes:
    """
    Standardized span attributes using OpenTelemetry semantic conventions.
    
    Uses semantic conventions where possible and adds RAG-specific attributes.
    """
    
    # General attributes (using semantic conventions)
    REQUEST_ID = "request.id"
    QUERY_LENGTH = "query.length"
    USER_AGENT = "http.user_agent"
    ENDPOINT = "http.route"
    METHOD = "http.method"
    STATUS_CODE = "http.status_code"
    
    # RAG-specific attributes
    STREAM_ENABLED = "rag.stream.enabled"
    CONFIG_OVERRIDE = "rag.config.override"
    
    # Retriever attributes
    EMBEDDING_MODEL = "rag.retrieval.embedding_model"
    TOP_K = "rag.retrieval.top_k"
    RETRIEVED_CHUNKS = "rag.retrieval.retrieved_chunks"
    AVERAGE_SIMILARITY = "rag.retrieval.average_similarity"
    RETRIEVAL_LATENCY = "rag.retrieval.latency_ms"
    SCORE_THRESHOLD = "rag.retrieval.score_threshold"
    COLLECTION_NAME = "rag.retrieval.collection"
    
    # Reranker attributes
    RERANKER_MODEL = "rag.reranking.model"
    RERANK_TOP_K = "rag.reranking.rerank_top_k"
    REORDERED_CHUNKS = "rag.reranking.reordered_chunks"
    RERANKING_LATENCY = "rag.reranking.latency_ms"
    FALLBACK_USED = "rag.reranking.fallback_used"
    FALLBACK_COUNT = "rag.reranking.fallback_count"
    
    # Context builder attributes
    CONTEXT_TOKENS = "rag.context.tokens"
    CONTEXT_CHUNKS = "rag.context.chunks"
    MAX_TOKENS = "rag.context.max_tokens"
    CONTEXT_BUILDING_LATENCY = "rag.context.latency_ms"
    
    # LLM attributes
    LLM_PROVIDER = "rag.llm.provider"
    LLM_MODEL = "rag.llm.model"
    PROMPT_TOKENS = "rag.llm.prompt_tokens"
    COMPLETION_TOKENS = "rag.llm.completion_tokens"
    TOTAL_TOKENS = "rag.llm.total_tokens"
    STREAMING_ENABLED = "rag.llm.streaming_enabled"
    GENERATION_LATENCY = "rag.llm.generation_latency_ms"
    TEMPERATURE = "rag.llm.temperature"
    MAX_TOKENS = "rag.llm.max_tokens"
    
    # Evaluation attributes
    DATASET = "rag.evaluation.dataset"
    EXPERIMENT_ID = "rag.evaluation.experiment_id"
    METRICS = "rag.evaluation.metrics"
    EVALUATION_STATUS = "rag.evaluation.status"
    
    # Streaming attributes
    STREAM_OPENED = "rag.streaming.opened"
    FIRST_TOKEN_LATENCY = "rag.streaming.first_token_latency_ms"
    TOKEN_COUNT = "rag.streaming.token_count"
    STREAM_DURATION = "rag.streaming.duration_ms"
    TOKENS_PER_SECOND = "rag.streaming.tokens_per_second"
    CLIENT_DISCONNECTED = "rag.streaming.client_disconnected"
    
    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK_TRACE = "error.stack_trace"


def get_general_attributes(
    request_id: str,
    query: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get general span attributes.
    
    Args:
        request_id: Request identifier
        query: Query string
        endpoint: Endpoint path
        method: HTTP method
        user_agent: User agent string
        
    Returns:
        Dictionary of attributes
    """
    attributes = {
        SpanAttributes.REQUEST_ID: request_id,
    }
    
    if query:
        attributes[SpanAttributes.QUERY_LENGTH] = len(query)
    
    if endpoint:
        attributes[SpanAttributes.ENDPOINT] = endpoint
    
    if method:
        attributes[SpanAttributes.METHOD] = method
    
    if user_agent:
        attributes[SpanAttributes.USER_AGENT] = user_agent
    
    return attributes


def get_retrieval_attributes(
    embedding_model: str,
    top_k: int,
    retrieved_chunks: int,
    average_similarity: float,
    retrieval_latency: float,
    score_threshold: Optional[float] = None,
    collection_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get retrieval span attributes.
    
    Args:
        embedding_model: Embedding model name
        top_k: Number of chunks retrieved
        retrieved_chunks: Actual chunks retrieved
        average_similarity: Average similarity score
        retrieval_latency: Retrieval latency in milliseconds
        score_threshold: Optional score threshold
        collection_name: Optional collection name
        
    Returns:
        Dictionary of attributes
    """
    attributes = {
        SpanAttributes.EMBEDDING_MODEL: embedding_model,
        SpanAttributes.TOP_K: top_k,
        SpanAttributes.RETRIEVED_CHUNKS: retrieved_chunks,
        SpanAttributes.AVERAGE_SIMILARITY: average_similarity,
        SpanAttributes.RETRIEEVAL_LATENCY: retrieval_latency,
    }
    
    if score_threshold is not None:
        attributes[SpanAttributes.SCORE_THRESHOLD] = score_threshold
    
    if collection_name:
        attributes[SpanAttributes.COLLECTION_NAME] = collection_name
    
    return attributes


def get_reranking_attributes(
    reranker_model: str,
    rerank_top_k: int,
    reordered_chunks: int,
    reranking_latency: float,
    fallback_used: bool = False,
    fallback_count: int = 0
) -> Dict[str, Any]:
    """
    Get reranking span attributes.
    
    Args:
        reranker_model: Reranker model name
        rerank_top_k: Number of chunks to rerank
        reordered_chunks: Number of chunks after reranking
        reranking_latency: Reranking latency in milliseconds
        fallback_used: Whether fallback was used
        fallback_count: Number of fallbacks
        
    Returns:
        Dictionary of attributes
    """
    return {
        SpanAttributes.RERANKER_MODEL: reranker_model,
        SpanAttributes.RERANK_TOP_K: rerank_top_k,
        SpanAttributes.REORDERED_CHUNKS: reordered_chunks,
        SpanAttributes.RERANKING_LATENCY: reranking_latency,
        SpanAttributes.FALLBACK_USED: fallback_used,
        SpanAttributes.FALLBACK_COUNT: fallback_count,
    }


def get_context_attributes(
    context_tokens: int,
    context_chunks: int,
    max_tokens: int,
    context_building_latency: float
) -> Dict[str, Any]:
    """
    Get context building span attributes.
    
    Args:
        context_tokens: Number of tokens in context
        context_chunks: Number of chunks in context
        max_tokens: Maximum token limit
        context_building_latency: Context building latency in milliseconds
        
    Returns:
        Dictionary of attributes
    """
    return {
        SpanAttributes.CONTEXT_TOKENS: context_tokens,
        SpanAttributes.CONTEXT_CHUNKS: context_chunks,
        SpanAttributes.MAX_TOKENS: max_tokens,
        SpanAttributes.CONTEXT_BUILDING_LATENCY: context_building_latency,
    }


def get_llm_attributes(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    generation_latency: float,
    streaming_enabled: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get LLM span attributes.
    
    Args:
        provider: LLM provider (e.g., "groq", "openai")
        model: Model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens
        generation_latency: Generation latency in milliseconds
        streaming_enabled: Whether streaming was enabled
        temperature: Optional temperature
        max_tokens: Optional max tokens
        
    Returns:
        Dictionary of attributes
    """
    attributes = {
        SpanAttributes.LLM_PROVIDER: provider,
        SpanAttributes.LLM_MODEL: model,
        SpanAttributes.PROMPT_TOKENS: prompt_tokens,
        SpanAttributes.COMPLETION_TOKENS: completion_tokens,
        SpanAttributes.TOTAL_TOKENS: total_tokens,
        SpanAttributes.GENERATION_LATENCY: generation_latency,
        SpanAttributes.STREAMING_ENABLED: streaming_enabled,
    }
    
    if temperature is not None:
        attributes[SpanAttributes.TEMPERATURE] = temperature
    
    if max_tokens is not None:
        attributes[SpanAttributes.MAX_TOKENS] = max_tokens
    
    return attributes


def get_streaming_attributes(
    first_token_latency: Optional[float] = None,
    token_count: Optional[int] = None,
    stream_duration: Optional[float] = None,
    client_disconnected: bool = False
) -> Dict[str, Any]:
    """
    Get streaming span attributes.
    
    Args:
        first_token_latency: Time to first token in milliseconds
        token_count: Total token count
        stream_duration: Stream duration in milliseconds
        client_disconnected: Whether client disconnected
        
    Returns:
        Dictionary of attributes
    """
    attributes = {
        SpanAttributes.STREAM_OPENED: True,
        SpanAttributes.CLIENT_DISCONNECTED: client_disconnected,
    }
    
    if first_token_latency is not None:
        attributes[SpanAttributes.FIRST_TOKEN_LATENCY] = first_token_latency
    
    if token_count is not None:
        attributes[SpanAttributes.TOKEN_COUNT] = token_count
    
    if stream_duration is not None:
        attributes[SpanAttributes.STREAM_DURATION] = stream_duration
        if token_count and stream_duration > 0:
            attributes[SpanAttributes.TOKENS_PER_SECOND] = (token_count / stream_duration) * 1000
    
    return attributes


def get_evaluation_attributes(
    dataset: str,
    experiment_id: str,
    metrics: Dict[str, Any],
    status: str
) -> Dict[str, Any]:
    """
    Get evaluation span attributes.
    
    Args:
        dataset: Dataset path or name
        experiment_id: Experiment identifier
        metrics: Evaluation metrics
        status: Evaluation status
        
    Returns:
        Dictionary of attributes
    """
    return {
        SpanAttributes.DATASET: dataset,
        SpanAttributes.EXPERIMENT_ID: experiment_id,
        SpanAttributes.METRICS: str(metrics),
        SpanAttributes.EVALUATION_STATUS: status,
    }


def get_error_attributes(
    error_type: str,
    error_message: str,
    stack_trace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get error span attributes.
    
    Args:
        error_type: Type of error
        error_message: Error message
        stack_trace: Optional stack trace
        
    Returns:
        Dictionary of attributes
    """
    attributes = {
        SpanAttributes.ERROR_TYPE: error_type,
        SpanAttributes.ERROR_MESSAGE: error_message,
    }
    
    if stack_trace:
        attributes[SpanAttributes.ERROR_STACK_TRACE] = stack_trace
    
    return attributes
