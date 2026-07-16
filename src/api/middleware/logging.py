"""
Logging middleware for comprehensive request logging.
"""
import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request logging.
    
    Logs request ID, endpoint, latencies, and other metrics.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize logging middleware.
        
        Args:
            app: ASGI application
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Log request details and metrics.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response
        """
        start_time = time.time()
        
        # Get request ID from state (set by RequestIDMiddleware)
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request start
        logger.info(
            f"Request started - ID: {request_id}, "
            f"Method: {request.method}, "
            f"Path: {request.url.path}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request completion
        logger.info(
            f"Request completed - ID: {request_id}, "
            f"Method: {request.method}, "
            f"Path: {request.url.path}, "
            f"Status: {response.status_code}, "
            f"Duration: {duration:.3f}s"
        )
        
        # Add duration to response headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response
