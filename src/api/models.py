"""
Ultra AI Project - API Models

Standard response models and helper functions for the API module.

Author: Ultra AI Team
Version: 1.0.0
"""

from pydantic import BaseModel
from typing import Optional, List, Union, Any, Dict
from datetime import datetime

# Response format constants
RESPONSE_SUCCESS = "success"
RESPONSE_ERROR = "error"
RESPONSE_WARNING = "warning"

# API Constants
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000

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