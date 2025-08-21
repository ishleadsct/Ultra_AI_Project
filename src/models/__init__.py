"""
Ultra AI Project - Models Module

This module manages AI model interfaces, model routing, and interactions
with various AI services including OpenAI, Anthropic, local models, and custom implementations.

Components:
- ModelManager: Central model orchestration and lifecycle management
- LLMInterface: Unified interface for language model interactions
- VisionModels: Computer vision model implementations
- AudioModels: Audio processing and speech model implementations
- Model routing and load balancing
- Performance monitoring and cost tracking

Author: Ultra AI Team
Version: 1.0.0
"""

from .model_manager import ModelManager, ModelConfig, ModelMetrics
from .llm_interface import (
    LLMInterface, 
    LLMRequest, 
    LLMResponse, 
    MessageRole,
    CompletionRequest,
    ChatMessage,
    ModelProvider
)
from .vision_models import VisionModelManager, ImageAnalysisRequest, ImageAnalysisResponse
from .audio_models import AudioModelManager, AudioProcessingRequest, AudioProcessingResponse

__all__ = [
    # Model Manager
    "ModelManager",
    "ModelConfig", 
    "ModelMetrics",
    
    # LLM Interface
    "LLMInterface",
    "LLMRequest",
    "LLMResponse",
    "MessageRole",
    "CompletionRequest", 
    "ChatMessage",
    "ModelProvider",
    
    # Vision Models
    "VisionModelManager",
    "ImageAnalysisRequest",
    "ImageAnalysisResponse",
    
    # Audio Models
    "AudioModelManager", 
    "AudioProcessingRequest",
    "AudioProcessingResponse",
]

# Module version
__version__ = "1.0.0"

# Module-level configuration
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

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

# Export constants
__all__.extend([
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS", 
    "DEFAULT_TIMEOUT",
    "MAX_CONTEXT_LENGTH",
    "OPENAI_PROVIDER",
    "ANTHROPIC_PROVIDER",
    "LOCAL_PROVIDER", 
    "HUGGINGFACE_PROVIDER",
    "CAPABILITY_TEXT_GENERATION",
    "CAPABILITY_CODE_GENERATION",
    "CAPABILITY_VISION",
    "CAPABILITY_AUDIO",
    "CAPABILITY_REASONING",
    "CAPABILITY_ANALYSIS",
    "CAPABILITY_FUNCTION_CALLING",
])

# Common response formats
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

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
    capabilities: list[str]
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

# Export response models
__all__.extend([
    "ModelStatus",
    "ModelInfo",
    "ModelUsage", 
    "StandardResponse",
])

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

# Model routing strategies
class RoutingStrategy(Enum):
    """Model routing strategy options."""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized" 
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"

# Export routing
__all__.extend([
    "DEFAULT_MODEL_CONFIG",
    "RoutingStrategy",
])

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

# Export utility functions
__all__.extend([
    "create_success_response",
    "create_error_response",
])

# Module initialization
def initialize_models_module():
    """Initialize the models module."""
    logger.info("Initializing Ultra AI Models module...")
    
    # Create runtime directories
    runtime_dir = Path(__file__).parent.parent.parent / "runtime"
    model_dirs = [
        "models",
        "models/cache",
        "models/local",
        "models/embeddings",
        "models/logs"
    ]
    
    for dir_name in model_dirs:
        (runtime_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    logger.info("Models module initialized successfully")

# Initialize on import
initialize_models_module()

logger.info("Ultra AI Models module loaded")
