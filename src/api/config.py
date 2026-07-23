"""
API configuration using Pydantic Settings for environment-based configuration.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from environment variables
    )
    
    # Application
    app_name: str = "Indian Budget RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = False
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # Authentication (optional)
    require_auth: bool = False
    api_key_header: str = "X-API-Key"
    api_keys: list[str] = []
    
    # Rate Limiting (protects Groq API quota/cost on /rag endpoints).
    # Paths are relative to api_prefix. rate_limit_rules maps a path
    # prefix to a per-endpoint override, e.g. from the environment:
    # RATE_LIMIT_RULES='{"/rag/query/stream": {"requests": 5, "window": 60}}'
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 10
    rate_limit_window_seconds: int = 60
    rate_limit_paths: list[str] = ["/rag/"]
    rate_limit_rules: dict[str, dict[str, int]] = {}
    # Enable ONLY behind a trusted reverse proxy that sets X-Forwarded-For
    rate_limit_trust_forwarded_for: bool = False

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "indian_budget_2026"
    qdrant_distance: str = "Cosine"
    
    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dim: int = 768
    
    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    groq_api_key: Optional[str] = None
    
    # Reranking
    enable_reranking: bool = False
    reranker_model: str = "ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    retrieve_top_k: int = 50
    rerank_top_k: int = 20
    return_top_k: int = 10
    
    # RAG Pipeline
    context_max_tokens: int = 4000
    include_citations: bool = True
    prompt_template: str = "qa"
    
    # Evaluation
    evaluation_output_dir: str = "evaluation/reports"
    evaluation_experiment_dir: str = "evaluation/experiments"
    
    # Background Tasks
    max_concurrent_evaluations: int = 3
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "indian-budget-rag"
    otel_environment: str = "development"
    otel_sampling_ratio: float = 1.0


# Global settings instance
settings = Settings()
