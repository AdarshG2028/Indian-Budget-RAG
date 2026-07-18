"""
Metrics definitions for OpenTelemetry observability.

Defines standard metric names and types for RAG operations.
"""


class MetricNames:
    """
    Standard metric names for RAG operations.
    
    Follows OpenTelemetry semantic conventions where possible.
    """
    
    # Counter metrics
    HTTP_REQUESTS = "http.requests.total"
    HTTP_ERRORS = "http.requests.errors"
    RATE_LIMIT_ALLOWED = "http.rate_limit.allowed"
    RATE_LIMIT_BLOCKED = "http.rate_limit.blocked"
    STREAMED_RESPONSES = "rag.responses.streamed"
    RETRIEVAL_REQUESTS = "rag.retrieval.requests"
    RETRIEVAL_FAILURES = "rag.retrieval.failures"
    RERANKING_REQUESTS = "rag.reranking.requests"
    RERANKING_FAILURES = "rag.reranking.failures"
    LLM_REQUESTS = "rag.llm.requests"
    LLM_FAILURES = "rag.llm.failures"
    EVALUATION_RUNS = "rag.evaluation.runs"
    CACHE_HITS = "rag.cache.hits"
    CACHE_MISSES = "rag.cache.misses"
    
    # Histogram metrics
    HTTP_REQUEST_LATENCY = "http.request.duration"
    # Gauge-like: sampled per request (TelemetryManager's observable
    # gauge cannot be updated after creation, so a histogram is used;
    # read the latest bucket value or max in dashboards)
    RATE_LIMIT_TRACKED_CLIENTS = "http.rate_limit.tracked_clients"
    EMBEDDING_LATENCY = "rag.retrieval.embedding.duration"
    RETRIEVAL_LATENCY = "rag.retrieval.duration"
    RERANKING_LATENCY = "rag.reranking.duration"
    CONTEXT_BUILDING_LATENCY = "rag.context.duration"
    PROMPT_SIZE = "rag.llm.prompt.size"
    GENERATION_LATENCY = "rag.llm.generation.duration"
    END_TO_END_LATENCY = "rag.pipeline.duration"
    FIRST_TOKEN_LATENCY = "rag.streaming.first_token_latency"
    
    # Gauge metrics
    ACTIVE_REQUESTS = "http.requests.active"
    ACTIVE_STREAMS = "rag.streaming.active"
    LOADED_MODELS = "rag.models.loaded"
    EVALUATION_JOBS = "rag.evaluation.jobs.active"


class MetricAttributes:
    """
    Standard metric attributes using semantic conventions.
    """
    
    # HTTP attributes
    HTTP_METHOD = "http.method"
    HTTP_ROUTE = "http.route"
    HTTP_STATUS_CODE = "http.status_code"
    
    # RAG attributes
    OPERATION = "rag.operation"
    MODEL = "rag.model"
    PROVIDER = "rag.provider"
    ERROR_TYPE = "error.type"
    STATUS = "status"
    
    # Operation values
    OPERATION_RETRIEVAL = "retrieval"
    OPERATION_RERANKING = "reranking"
    OPERATION_GENERATION = "generation"
    OPERATION_EVALUATION = "evaluation"
    
    # Status values
    STATUS_SUCCESS = "success"
    STATUS_FAILURE = "failure"
    STATUS_ERROR = "error"


class MetricsHelper:
    """
    Helper class for recording metrics using TelemetryManager.
    """
    
    def __init__(self, telemetry_manager):
        """
        Initialize metrics helper.
        
        Args:
            telemetry_manager: TelemetryManager instance
        """
        self.tm = telemetry_manager
    
    def record_http_request(
        self,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float
    ):
        """
        Record HTTP request metrics.
        
        Args:
            method: HTTP method
            route: Route path
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
        """
        attributes = {
            MetricAttributes.HTTP_METHOD: method,
            MetricAttributes.HTTP_ROUTE: route,
            MetricAttributes.HTTP_STATUS_CODE: str(status_code),
        }
        
        self.tm.increment_counter(
            MetricNames.HTTP_REQUESTS,
            attributes=attributes
        )
        
        self.tm.record_histogram(
            MetricNames.HTTP_REQUEST_LATENCY,
            duration_ms,
            attributes=attributes
        )
        
        if status_code >= 400:
            self.tm.increment_counter(
                MetricNames.HTTP_ERRORS,
                attributes=attributes
            )
    
    def record_retrieval(
        self,
        embedding_model: str,
        top_k: int,
        retrieved_count: int,
        embedding_latency_ms: float,
        retrieval_latency_ms: float,
        success: bool = True
    ):
        """
        Record retrieval metrics.
        
        Args:
            embedding_model: Embedding model name
            top_k: Number of chunks requested
            retrieved_count: Number of chunks retrieved
            embedding_latency_ms: Embedding latency in milliseconds
            retrieval_latency_ms: Total retrieval latency in milliseconds
            success: Whether retrieval was successful
        """
        attributes = {
            MetricAttributes.OPERATION: MetricAttributes.OPERATION_RETRIEVAL,
            MetricAttributes.MODEL: embedding_model,
            MetricAttributes.STATUS: MetricAttributes.STATUS_SUCCESS if success else MetricAttributes.STATUS_FAILURE,
        }
        
        self.tm.increment_counter(
            MetricNames.RETRIEVAL_REQUESTS,
            attributes=attributes
        )
        
        self.tm.record_histogram(
            MetricNames.EMBEDDING_LATENCY,
            embedding_latency_ms,
            attributes=attributes
        )
        
        self.tm.record_histogram(
            MetricNames.RETRIEVAL_LATENCY,
            retrieval_latency_ms,
            attributes=attributes
        )
        
        if not success:
            self.tm.increment_counter(
                MetricNames.RETRIEVAL_FAILURES,
                attributes=attributes
            )
    
    def record_reranking(
        self,
        reranker_model: str,
        chunks_count: int,
        reranking_latency_ms: float,
        success: bool = True
    ):
        """
        Record reranking metrics.
        
        Args:
            reranker_model: Reranker model name
            chunks_count: Number of chunks reranked
            reranking_latency_ms: Reranking latency in milliseconds
            success: Whether reranking was successful
        """
        attributes = {
            MetricAttributes.OPERATION: MetricAttributes.OPERATION_RERANKING,
            MetricAttributes.MODEL: reranker_model,
            MetricAttributes.STATUS: MetricAttributes.STATUS_SUCCESS if success else MetricAttributes.STATUS_FAILURE,
        }
        
        self.tm.increment_counter(
            MetricNames.RERANKING_REQUESTS,
            attributes=attributes
        )
        
        self.tm.record_histogram(
            MetricNames.RERANKING_LATENCY,
            reranking_latency_ms,
            attributes=attributes
        )
        
        if not success:
            self.tm.increment_counter(
                MetricNames.RERANKING_FAILURES,
                attributes=attributes
            )
    
    def record_llm_generation(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        generation_latency_ms: float,
        streaming: bool = False,
        success: bool = True
    ):
        """
        Record LLM generation metrics.
        
        Args:
            provider: LLM provider
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            generation_latency_ms: Generation latency in milliseconds
            streaming: Whether streaming was enabled
            success: Whether generation was successful
        """
        attributes = {
            MetricAttributes.OPERATION: MetricAttributes.OPERATION_GENERATION,
            MetricAttributes.PROVIDER: provider,
            MetricAttributes.MODEL: model,
            MetricAttributes.STATUS: MetricAttributes.STATUS_SUCCESS if success else MetricAttributes.STATUS_FAILURE,
        }
        
        self.tm.increment_counter(
            MetricNames.LLM_REQUESTS,
            attributes=attributes
        )
        
        self.tm.record_histogram(
            MetricNames.PROMPT_SIZE,
            prompt_tokens,
            attributes=attributes
        )
        
        self.tm.record_histogram(
            MetricNames.GENERATION_LATENCY,
            generation_latency_ms,
            attributes=attributes
        )
        
        if streaming:
            self.tm.increment_counter(
                MetricNames.STREAMED_RESPONSES,
                attributes=attributes
            )
        
        if not success:
            self.tm.increment_counter(
                MetricNames.LLM_FAILURES,
                attributes=attributes
            )
    
    def record_streaming(
        self,
        first_token_latency_ms: float,
        token_count: int,
        stream_duration_ms: float
    ):
        """
        Record streaming metrics.
        
        Args:
            first_token_latency_ms: Time to first token in milliseconds
            token_count: Number of tokens streamed
            stream_duration_ms: Stream duration in milliseconds
        """
        self.tm.record_histogram(
            MetricNames.FIRST_TOKEN_LATENCY,
            first_token_latency_ms
        )
        
        tokens_per_second = (token_count / stream_duration_ms) * 1000 if stream_duration_ms > 0 else 0
        
        self.tm.record_histogram(
            "rag.streaming.tokens_per_second",
            tokens_per_second
        )
    
    def record_evaluation(
        self,
        dataset: str,
        experiment_id: str,
        success: bool = True
    ):
        """
        Record evaluation metrics.
        
        Args:
            dataset: Dataset name
            experiment_id: Experiment ID
            success: Whether evaluation was successful
        """
        attributes = {
            MetricAttributes.OPERATION: MetricAttributes.OPERATION_EVALUATION,
            MetricAttributes.STATUS: MetricAttributes.STATUS_SUCCESS if success else MetricAttributes.STATUS_FAILURE,
        }
        
        self.tm.increment_counter(
            MetricNames.EVALUATION_RUNS,
            attributes=attributes
        )
    
    def record_cache_hit(self, cache_key: str):
        """
        Record cache hit.
        
        Args:
            cache_key: Cache key
        """
        self.tm.increment_counter(
            MetricNames.CACHE_HITS,
            attributes={"cache_key": cache_key}
        )
    
    def record_cache_miss(self, cache_key: str):
        """
        Record cache miss.
        
        Args:
            cache_key: Cache key
        """
        self.tm.increment_counter(
            MetricNames.CACHE_MISSES,
            attributes={"cache_key": cache_key}
        )
    
    def record_end_to_end_latency(self, duration_ms: float, operation: str):
        """
        Record end-to-end pipeline latency.
        
        Args:
            duration_ms: Duration in milliseconds
            operation: Operation type
        """
        attributes = {
            MetricAttributes.OPERATION: operation,
        }
        
        self.tm.record_histogram(
            MetricNames.END_TO_END_LATENCY,
            duration_ms,
            attributes=attributes
        )
