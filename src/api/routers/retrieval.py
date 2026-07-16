"""
Retrieval router for retrieval-only endpoints.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from fastapi import APIRouter, Depends, HTTPException

from ..models import RetrievalRequest, RetrievalResponse, RetrievalResult
from ..services import RetrievalService
from ..dependencies import get_retriever, get_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    retriever = Depends(get_retriever),
    request_id: str = Depends(get_request_id)
) -> RetrievalResponse:
    """
    Execute retrieval-only query.
    
    Args:
        request: Retrieval request
        retriever: Dense retriever
        request_id: Request identifier
        
    Returns:
        Retrieval response with results and metadata
    """
    try:
        logger.info(f"Processing retrieval query: {request_id}")
        
        # Create service instance
        service = RetrievalService(retriever)
        
        # Execute retrieval
        result = service.retrieve(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        # Convert results to API format
        results = [
            RetrievalResult(**r)
            for r in result["results"]
        ]
        
        return RetrievalResponse(
            request_id=request_id,
            query=request.query,
            results=results,
            metrics=result["metrics"]
        )
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
