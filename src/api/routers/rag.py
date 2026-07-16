"""
RAG router for query endpoints.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any

from ..models import QueryRequest, QueryResponse, Citation
from ..services import RAGService
from ..dependencies import get_rag_pipeline, get_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def query(
    query_request: QueryRequest,
    http_request: Request,
    pipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """
    Execute RAG query (non-streaming).
    
    Args:
        query_request: Query request
        http_request: HTTP request
        pipeline: RAG pipeline
        
    Returns:
        Query response with answer and metadata
    """
    try:
        request_id = getattr(http_request.state, "request_id", "unknown")
        logger.info(f"Processing RAG query: {request_id}")
        
        # Create service instance
        service = RAGService(pipeline)
        
        # Execute query (non-streaming)
        result = {}
        async for chunk in service.query(
            question=query_request.question,
            request_id=request_id,
            stream=False,
            config=query_request.config
        ):
            result = chunk
        
        # Build citations
        citations = None
        if result.get("citations"):
            citations = [
                Citation(
                    chunk_id=c.get("chunk_id", ""),
                    document=c.get("document", ""),
                    year=c.get("year", 0),
                    section=c.get("section", ""),
                    page_start=c.get("page_start", 0),
                    page_end=c.get("page_end", 0),
                    similarity=c.get("similarity", 0.0)
                )
                for c in result["citations"]
            ]
        
        return QueryResponse(
            request_id=request_id,
            answer=result["answer"],
            citations=citations,
            metrics=result["metrics"],
            reranker_used=result["reranker_used"],
            reranker_metrics=result["reranker_metrics"]
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(
    query_request: QueryRequest,
    http_request: Request,
    pipeline = Depends(get_rag_pipeline)
) -> StreamingResponse:
    """
    Execute RAG query with streaming.
    
    Uses Server-Sent Events (SSE) to stream structured events.
    
    Args:
        query_request: Query request
        http_request: HTTP request
        pipeline: RAG pipeline
        
    Returns:
        Streaming response with SSE events
    """
    try:
        request_id = getattr(http_request.state, "request_id", "unknown")
        logger.info(f"Processing RAG query (streaming): {request_id}")
        
        # Create service instance
        service = RAGService(pipeline)
        
        async def event_generator():
            """Generate SSE events."""
            try:
                async for event in service.query(
                    question=query_request.question,
                    request_id=request_id,
                    stream=True,
                    config=query_request.config
                ):
                    yield event
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_event = f'event: error\ndata: {{"error": "{str(e)}"}}\n\n'
                yield error_event
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"RAG stream setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
