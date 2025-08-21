"""
Ultra AI Project - API Module

This module provides the RESTful API and WebSocket interfaces for the Ultra AI Project,
including route definitions, middleware, authentication, and real-time communication.

Components:
- Routes: API endpoint definitions and handlers
- Middleware: Request/response processing and validation
- Authentication: API key and session-based authentication
- WebSocket: Real-time bidirectional communication
- Error handling and response formatting

Author: Ultra AI Team
Version: 1.0.0
"""

from .routes import app, router, get_system_status
from .middleware import (
    setup_middleware,
    RateLimitMiddleware,
    SecurityMiddleware,
    LoggingMiddleware,
    CORSMiddleware
)
from .auth import (
    authenticate_request,
    require_auth,
    get_current_user,
    APIKeyAuth,
    SessionAuth
)
from .websocket import (
    WebSocketManager,
    ConnectionManager,
    websocket_endpoint,
    broadcast_message
)

__all__ = [
    # Main app
    "app",
    "router", 
    "get_system_status",
    
    # Middleware
    "setup_middleware",
    "RateLimitMiddleware",
    "SecurityMiddleware", 
    "LoggingMiddleware",
    "CORSMiddleware",
    
    # Authentication
    "authenticate_request",
    "require_auth",
    "get_current_user",
    "APIKeyAuth",
    "SessionAuth",
    
    # WebSocket
    "WebSocketManager",
    "ConnectionManager",
    "websocket_endpoint",
    "broadcast_message",
]

# API module version
__version__ = "1.0.0"

# Module-level configuration
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# API Constants
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000
REQUEST_TIMEOUT = 300  # 5 minutes

# Response format constants
RESPONSE_SUCCESS = "success"
RESPONSE_ERROR = "error"
RESPONSE_WARNING = "warning"

# HTTP Status codes commonly used
HTTP_200_OK = 200
HTTP_201_CREATED = 201
HTTP_400_BAD_REQUEST = 400
HTTP_401_UNAUTHORIZED = 401
HTTP_403_FORBIDDEN = 403
HTTP_404_NOT_FOUND = 404
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_429_TOO_MANY_REQUESTS = 429
HTTP_500_INTERNAL_SERVER_ERROR = 500

# Export constants
__all__.extend([
    "API_VERSION",
    "API_PREFIX", 
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "REQUEST_TIMEOUT",
    "RESPONSE_SUCCESS",
    "RESPONSE_ERROR", 
    "RESPONSE_WARNING",
    "HTTP_200_OK",
    "HTTP_201_CREATED",
    "HTTP_400_BAD_REQUEST",
    "HTTP_401_UNAUTHORIZED",
    "HTTP_403_FORBIDDEN",
    "HTTP_404_NOT_FOUND",
    "HTTP_422_UNPROCESSABLE_ENTITY", 
    "HTTP_429_TOO_MANY_REQUESTS",
    "HTTP_500_INTERNAL_SERVER_ERROR",
])

# Standard API response models
from pydantic import BaseModel
from typing import Optional, List, Union
from datetime import datetime

class APIResponse(BaseModel):
    """Standard API response format."""
    status: str  # success, error, warning
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PaginatedResponse(BaseModel):
    """Paginated response format."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool

class ErrorDetail(BaseModel):
    """Error detail format."""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Export response models
__all__.extend([
    "APIResponse",
    "PaginatedResponse", 
    "ErrorDetail",
])

def create_success_response(
    message: str = "Success",
    data: Any = None,
    request_id: Optional[str] = None
) -> APIResponse:
    """Create a success response."""
    return APIResponse(
        status=RESPONSE_SUCCESS,
        message=message,
        data=data,
        timestamp=datetime.now(),
        request_id=request_id
    )

def create_error_response(
    message: str = "An error occurred",
    errors: Optional[List[str]] = None,
    request_id: Optional[str] = None
) -> APIResponse:
    """Create an error response."""
    return APIResponse(
        status=RESPONSE_ERROR,
        message=message,
        errors=errors,
        timestamp=datetime.now(),
        request_id=request_id
    )

def create_paginated_response(
    items: List[Any],
    total: int,
    page: int,
    size: int
) -> PaginatedResponse:
    """Create a paginated response."""
    pages = (total + size - 1) // size  # Ceiling division
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1
    )

# Export helper functions
__all__.extend([
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
])

# Module initialization
def initialize_api_module():
    """Initialize the API module."""
    logger.info("Initializing Ultra AI API module...")
    
    # Ensure required directories exist
    from pathlib import Path
    api_dirs = [
        "runtime/api",
        "runtime/api/logs",
        "runtime/api/sessions",
        "runtime/api/uploads"
    ]
    
    for dir_name in api_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    logger.info("API module initialized successfully")

# Initialize on import
initialize_api_module()

logger.info("Ultra AI API module loaded")
