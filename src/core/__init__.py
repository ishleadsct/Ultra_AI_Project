"""
Ultra AI Project - Core Module

This module contains the core functionality of the Ultra AI Project,
including system management, task coordination, memory management,
and security components.

Components:
- SystemManager: Central system orchestration and lifecycle management
- TaskCoordinator: Task scheduling, execution, and monitoring
- MemoryManager: Memory and context management across agents
- Security: Authentication, authorization, and security enforcement

Author: Ultra AI Team
Version: 1.0.0
"""

from .system_manager import SystemManager
from .task_coordinator import TaskCoordinator
from .memory_manager import MemoryManager
from .security import SecurityManager, authenticate_user, authorize_action

__all__ = [
    "SystemManager",
    "TaskCoordinator", 
    "MemoryManager",
    "SecurityManager",
    "authenticate_user",
    "authorize_action",
]

# Core module version
__version__ = "1.0.0"

# Module-level configuration
import logging

logger = logging.getLogger(__name__)
logger.info("Ultra AI Core module loaded")

# Core constants
CORE_MODULE_NAME = "ultra_ai.core"
DEFAULT_TASK_TIMEOUT = 300  # 5 minutes
DEFAULT_MEMORY_LIMIT = 1000  # Max items in memory
DEFAULT_SECURITY_LEVEL = "standard"

# Core module initialization
def initialize_core_module():
    """Initialize the core module with default settings."""
    logger.info("Initializing Ultra AI Core module...")
    
    # Create runtime directories if they don't exist
    from pathlib import Path
    runtime_dir = Path(__file__).parent.parent.parent / "runtime"
    
    core_dirs = [
        "core",
        "core/tasks", 
        "core/memory",
        "core/security",
        "core/locks"
    ]
    
    for dir_name in core_dirs:
        (runtime_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    logger.info("Core module initialized successfully")

# Initialize on import
initialize_core_module()
