"""
Ultra AI Project - Model Types

Common data structures and types for the models module.
This module breaks circular import dependencies by providing shared types.
"""

from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

# Model constants
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 60
MAX_CONTEXT_LENGTH = 128000

# Model provider constants
OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"
LOCAL_PROVIDER = "local"
HUGGINGFACE_PROVIDER = "huggingface"

# Model capability flags
CAPABILITY_TEXT_GENERATION = "text_generation"
CAPABILITY_CODE_GENERATION = "code_generation"
CAPABILITY_VISION = "vision"
CAPABILITY_AUDIO = "audio"
CAPABILITY_REASONING = "reasoning"
CAPABILITY_ANALYSIS = "analysis"
CAPABILITY_FUNCTION_CALLING = "function_calling"

# Model configuration defaults
DEFAULT_MODEL_CONFIG = {
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "timeout": DEFAULT_TIMEOUT,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "enable_streaming": False,
    "enable_caching": True,
    "cache_ttl": 3600,
}

class ModelStatus(Enum):
    """Model status enumeration."""
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"

class ModelInfo(BaseModel):
    """Model information structure."""
    name: str
    provider: str
    status: ModelStatus
    capabilities: List[str]
    context_window: int
    max_tokens: int
    cost_per_1k_tokens: Optional[Dict[str, float]] = None
    description: Optional[str] = None
    version: Optional[str] = None

class ModelUsage(BaseModel):
    """Model usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    processing_time: float = 0.0

class StandardResponse(BaseModel):
    """Standard model response format."""
    success: bool
    content: Optional[str] = None
    data: Optional[Any] = None
    usage: Optional[ModelUsage] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    timestamp: datetime
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RoutingStrategy(Enum):
    """Model routing strategy options."""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized" 
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"

def create_success_response(
    content: str,
    model: str,
    provider: str,
    usage: Optional[ModelUsage] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> StandardResponse:
    """Create a successful model response."""
    return StandardResponse(
        success=True,
        content=content,
        usage=usage,
        model=model,
        provider=provider,
        timestamp=datetime.now(),
        metadata=metadata
    )

def create_error_response(
    error: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> StandardResponse:
    """Create an error model response."""
    return StandardResponse(
        success=False,
        error=error,
        model=model,
        provider=provider,
        timestamp=datetime.now(),
        metadata=metadata
    )