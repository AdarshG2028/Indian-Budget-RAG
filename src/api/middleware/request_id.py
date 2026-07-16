"""
Request ID middleware for generating and injecting unique request IDs.
"""
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and inject unique request IDs.
    
    Adds a unique request ID to each request for tracing and logging.
    """
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        """
        Initialize request ID middleware.
        
        Args:
            app: ASGI application
            header_name: Header name for request ID
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        """
        Generate and inject request ID.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response with request ID header
        """
        # Check if request ID is already present in headers
        request_id = request.headers.get(self.header_name)
        
        # Generate new request ID if not present
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Inject request ID into request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response
