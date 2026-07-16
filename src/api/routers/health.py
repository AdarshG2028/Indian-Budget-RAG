"""
Health check router.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from fastapi import APIRouter, Depends
from typing import Dict, Any

from ..models import HealthResponse
from ..dependencies import get_vector_store, get_llm, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def liveness_check() -> HealthResponse:
    """
    Liveness check - verifies the application is running.
    
    Returns healthy if the application can respond to requests.
    """
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        checks={"type": "liveness"}
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(
    vector_store = Depends(get_vector_store),
    llm = Depends(get_llm)
) -> HealthResponse:
    """
    Readiness check - verifies dependencies are available.
    
    Checks:
    - Qdrant connectivity
    - LLM availability
    - Model loading
    """
    settings = get_settings()
    checks: Dict[str, Any] = {}
    all_healthy = True
    
    # Check Qdrant
    try:
        # Try to access vector store
        checks["qdrant"] = {
            "status": "healthy",
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection
        }
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        checks["qdrant"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        all_healthy = False
    
    # Check LLM
    try:
        checks["llm"] = {
            "status": "healthy",
            "model": settings.llm_model,
            "api_key_configured": bool(settings.groq_api_key)
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        checks["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        all_healthy = False
    
    # Check embedding model
    try:
        checks["embeddings"] = {
            "status": "healthy",
            "model": settings.embedding_model,
            "device": settings.embedding_device
        }
    except Exception as e:
        logger.error(f"Embedding health check failed: {e}")
        checks["embeddings"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        all_healthy = False
    
    status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        version=settings.app_version,
        checks=checks
    )
