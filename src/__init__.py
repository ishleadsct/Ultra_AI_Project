"""
Ultra AI Project - Advanced AI system with multi-agent capabilities.

This package provides a comprehensive AI framework with:
- Multi-agent system architecture
- Advanced language model integration
- Vision and audio processing capabilities
- RESTful API and WebSocket support
- Comprehensive security and monitoring
- Extensible plugin system

Author: Ultra AI Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Ultra AI Team"
__email__ = "team@ultraai.dev"
__license__ = "MIT"
__description__ = "Advanced AI system with multi-agent capabilities and comprehensive tooling"

# Core imports
from .core.system_manager import SystemManager
from .core.task_coordinator import TaskCoordinator
from .core.memory_manager import MemoryManager

# API imports
from .api.routes import app as api_app

# Agent imports
from .agents.base_agent import BaseAgent
from .agents.code_agent import CodeAgent
from .agents.research_agent import ResearchAgent
from .agents.creative_agent import CreativeAgent
from .agents.analysis_agent import AnalysisAgent

# Model imports
from .models.model_manager import ModelManager
from .models.llm_interface import LLMInterface

# Utility imports
from .utils.logger import get_logger
from .utils.helpers import load_config, get_system_info

# Package metadata
__all__ = [
    # Core
    "SystemManager",
    "TaskCoordinator", 
    "MemoryManager",
    
    # API
    "api_app",
    
    # Agents
    "BaseAgent",
    "CodeAgent",
    "ResearchAgent", 
    "CreativeAgent",
    "AnalysisAgent",
    
    # Models
    "ModelManager",
    "LLMInterface",
    
    # Utils
    "get_logger",
    "load_config",
    "get_system_info",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
]

# Package configuration
import os
import logging
from pathlib import Path

# Set up package-level configuration
PACKAGE_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
RUNTIME_DIR = PACKAGE_ROOT / "runtime"

# Ensure runtime directories exist
for directory in ["logs", "storage", "temp", "cache", "models", "uploads"]:
    (RUNTIME_DIR / directory).mkdir(parents=True, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RUNTIME_DIR / "logs" / "ultra_ai.log")
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Ultra AI Project v{__version__} initialized")

# Environment validation
def validate_environment():
    """Validate the environment setup."""
    required_env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some features may not work properly")
    
    return len(missing_vars) == 0

# Validate environment on import
validate_environment()

# Package initialization complete
logger.info("Ultra AI Project package loaded successfully")
