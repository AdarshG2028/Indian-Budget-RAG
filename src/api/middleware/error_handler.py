"""
Error handler middleware for standardized error responses.
"""
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

from ..models import ErrorResponse, ErrorCode

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for standardized error handling.
    
    Catches exceptions and returns standardized error responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Handle exceptions and return standardized error responses.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response or error response
        """
        try:
            return await call_next(request)
        except HTTPException as e:
            # Handle HTTP exceptions
            request_id = getattr(request.state, "request_id", "unknown")
            
            error_response = ErrorResponse(
                request_id=request_id,
                error_code=self._http_status_to_code(e.status_code),
                message=e.detail,
                details={"status_code": e.status_code}
            )
            
            logger.error(
                f"HTTP error - ID: {request_id}, "
                f"Status: {e.status_code}, "
                f"Message: {e.detail}"
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.model_dump(mode='json')
            )
        except Exception as e:
            # Handle unexpected exceptions
            request_id = getattr(request.state, "request_id", "unknown")
            
            error_response = ErrorResponse(
                request_id=request_id,
                error_code=ErrorCode.INTERNAL_ERROR,
                message="An internal error occurred",
                details={"exception": str(e)}
            )
            
            logger.error(
                f"Unexpected error - ID: {request_id}, "
                f"Exception: {type(e).__name__}, "
                f"Message: {str(e)}",
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(mode='json')
            )
    
    def _http_status_to_code(self, status_code: int) -> ErrorCode:
        """
        Convert HTTP status code to error code.
        
        Args:
            status_code: HTTP status code
            
        Returns:
            Error code
        """
        mapping = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.UNAUTHORIZED,
            404: ErrorCode.NOT_FOUND,
            429: ErrorCode.RATE_LIMITED,
        }
        return mapping.get(status_code, ErrorCode.INTERNAL_ERROR)
