"""
FastAPI application for Indian Budget RAG API.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .middleware import (
    RequestIDMiddleware,
    LoggingMiddleware,
    ErrorHandlerMiddleware,
    RateLimitMiddleware
)
from .routers import health_router, rag_router, retrieval_router, evaluation_router
from .telemetry import configure_telemetry
from .telemetry.config import shutdown_telemetry

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"API prefix: {settings.api_prefix}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Configure OpenTelemetry
    environment = "development" if settings.debug else "production"
    configure_telemetry(app=app, environment=environment)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    shutdown_telemetry()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade RAG API for Indian Budget documents",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware (origins are config-driven; extend settings.cors_origins
# rather than widening to "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add custom middleware (added first = runs closest to the route, so
# rate-limited 429s still get request IDs and logging)
if settings.rate_limit_enabled:
    # Config paths are relative to api_prefix; compose full prefixes here
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
        limited_path_prefixes=tuple(
            f"{settings.api_prefix}{path}" for path in settings.rate_limit_paths
        ),
        rules={
            f"{settings.api_prefix}{path}": rule
            for path, rule in settings.rate_limit_rules.items()
        },
        trust_forwarded_for=settings.rate_limit_trust_forwarded_for,
        api_key_header=settings.api_key_header
    )
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Include routers
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(rag_router, prefix=settings.api_prefix)
app.include_router(retrieval_router, prefix=settings.api_prefix)
app.include_router(evaluation_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "api_prefix": settings.api_prefix,
        "docs": f"{settings.api_prefix}/docs",
        "health": f"{settings.api_prefix}/health/live"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
