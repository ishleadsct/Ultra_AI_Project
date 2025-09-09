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

# Constants now imported from types module

# Common response formats - imported from types module to avoid circular imports
from .types import (
    ModelStatus, 
    ModelInfo, 
    ModelUsage, 
    StandardResponse, 
    RoutingStrategy,
    create_success_response,
    create_error_response,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    MAX_CONTEXT_LENGTH,
    OPENAI_PROVIDER,
    ANTHROPIC_PROVIDER,
    LOCAL_PROVIDER,
    HUGGINGFACE_PROVIDER,
    CAPABILITY_TEXT_GENERATION,
    CAPABILITY_CODE_GENERATION,
    CAPABILITY_VISION,
    CAPABILITY_AUDIO,
    CAPABILITY_REASONING,
    CAPABILITY_ANALYSIS,
    CAPABILITY_FUNCTION_CALLING,
    DEFAULT_MODEL_CONFIG
)

# Export response models
__all__.extend([
    "ModelStatus",
    "ModelInfo",
    "ModelUsage", 
    "StandardResponse",
])

# Configuration now imported from types module

# Export routing and configuration
__all__.extend([
    "DEFAULT_MODEL_CONFIG",  # Already imported from types
    "RoutingStrategy",       # Already imported from types
])

# Utility functions now imported from types module

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
