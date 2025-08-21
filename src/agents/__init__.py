"""
Ultra AI Project - Agents Module

This module provides intelligent AI agents with specialized capabilities
for different tasks including code generation, research, creative writing, and analysis.

Components:
- BaseAgent: Core agent framework and interface
- CodeAgent: Programming and software development assistance
- ResearchAgent: Information gathering and knowledge synthesis
- CreativeAgent: Creative writing, storytelling, and content generation
- AnalysisAgent: Data analysis, reasoning, and problem solving
- Agent communication and coordination
- Task delegation and workflow management

Author: Ultra AI Team
Version: 1.0.0
"""

from .base_agent import BaseAgent, AgentConfig, AgentStatus, AgentCapability
from .code_agent import CodeAgent
from .research_agent import ResearchAgent
from .creative_agent import CreativeAgent
from .analysis_agent import AnalysisAgent

__all__ = [
    # Base Agent
    "BaseAgent",
    "AgentConfig",
    "AgentStatus", 
    "AgentCapability",
    
    # Specialized Agents
    "CodeAgent",
    "ResearchAgent",
    "CreativeAgent",
    "AnalysisAgent",
]

# Module version
__version__ = "1.0.0"

# Module-level configuration
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Agent type definitions
class AgentType(Enum):
    """Agent type enumeration."""
    CODE = "code"
    RESEARCH = "research"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    GENERAL = "general"

# Agent communication protocols
class MessageType(Enum):
    """Agent message types."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"

# Agent registry for dynamic discovery
AGENT_REGISTRY: Dict[str, type] = {
    "code": CodeAgent,
    "research": ResearchAgent,
    "creative": CreativeAgent,
    "analysis": AnalysisAgent,
}

# Default agent configurations
DEFAULT_AGENT_CONFIGS = {
    "code": {
        "max_concurrent_tasks": 3,
        "timeout": 600,  # 10 minutes
        "capabilities": ["code_generation", "code_review", "debugging", "testing"],
        "preferred_models": ["gpt-4", "claude-3-sonnet"],
        "memory_limit": 1000,
    },
    "research": {
        "max_concurrent_tasks": 5,
        "timeout": 1200,  # 20 minutes
        "capabilities": ["web_search", "document_analysis", "summarization"],
        "preferred_models": ["gpt-4", "claude-3-opus"],
        "memory_limit": 2000,
    },
    "creative": {
        "max_concurrent_tasks": 3,
        "timeout": 900,  # 15 minutes
        "capabilities": ["creative_writing", "storytelling", "content_generation"],
        "preferred_models": ["gpt-4", "claude-3-sonnet"],
        "memory_limit": 1500,
    },
    "analysis": {
        "max_concurrent_tasks": 4,
        "timeout": 800,  # 13 minutes
        "capabilities": ["data_analysis", "reasoning", "problem_solving"],
        "preferred_models": ["gpt-4", "claude-3-opus"],
        "memory_limit": 1200,
    }
}

# Export constants
__all__.extend([
    "AgentType",
    "MessageType", 
    "AGENT_REGISTRY",
    "DEFAULT_AGENT_CONFIGS",
])

# Agent factory functions
def create_agent(agent_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Optional[BaseAgent]:
    """Create an agent instance by type."""
    try:
        if agent_type not in AGENT_REGISTRY:
            logger.error(f"Unknown agent type: {agent_type}")
            return None
        
        agent_class = AGENT_REGISTRY[agent_type]
        
        # Merge default config with provided config
        default_config = DEFAULT_AGENT_CONFIGS.get(agent_type, {})
        if config:
            default_config.update(config)
        
        # Create agent instance
        agent = agent_class(config=default_config, **kwargs)
        
        logger.info(f"Created {agent_type} agent: {agent.agent_id}")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create {agent_type} agent: {e}")
        return None

async def get_agent_by_name(agent_name: str) -> Optional[BaseAgent]:
    """Get agent instance by name (placeholder for registry lookup)."""
    # This would typically look up agents in a running system registry
    # For now, return None to indicate agent not found
    logger.warning(f"Agent lookup not implemented: {agent_name}")
    return None

def get_available_agent_types() -> List[str]:
    """Get list of available agent types."""
    return list(AGENT_REGISTRY.keys())

def get_agent_capabilities(agent_type: str) -> List[str]:
    """Get capabilities for an agent type."""
    config = DEFAULT_AGENT_CONFIGS.get(agent_type, {})
    return config.get("capabilities", [])

# Export utility functions
__all__.extend([
    "create_agent",
    "get_agent_by_name",
    "get_available_agent_types",
    "get_agent_capabilities",
])

# Agent communication framework
from pydantic import BaseModel
from datetime import datetime

class AgentMessage(BaseModel):
    """Agent communication message."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=high

class AgentResponse(BaseModel):
    """Agent response structure."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    agent_id: str
    task_id: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = {}

# Export communication models
__all__.extend([
    "AgentMessage",
    "AgentResponse",
])

# Agent coordination utilities
class AgentCoordinator:
    """Utility class for agent coordination."""
    
    def __init__(self):
        self.active_agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent for coordination."""
        self.active_agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message between agents."""
        recipient = self.active_agents.get(message.recipient_id)
        if not recipient:
            logger.error(f"Recipient agent not found: {message.recipient_id}")
            return False
        
        try:
            # This would be implemented in the actual agent
            # await recipient.receive_message(message)
            logger.info(f"Message sent from {message.sender_id} to {message.recipient_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered agents."""
        return {
            agent_id: {
                "type": agent.agent_type,
                "status": agent.status.value,
                "active_tasks": len(getattr(agent, 'active_tasks', [])),
                "total_tasks": getattr(agent, 'total_tasks_processed', 0)
            }
            for agent_id, agent in self.active_agents.items()
        }

# Export coordination utilities
__all__.extend([
    "AgentCoordinator",
])

# Module initialization
def initialize_agents_module():
    """Initialize the agents module."""
    logger.info("Initializing Ultra AI Agents module...")
    
    # Create runtime directories
    from pathlib import Path
    runtime_dir = Path(__file__).parent.parent.parent / "runtime"
    agent_dirs = [
        "agents",
        "agents/logs",
        "agents/memory",
        "agents/tasks",
        "agents/outputs"
    ]
    
    for dir_name in agent_dirs:
        (runtime_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    logger.info("Agents module initialized successfully")

# Initialize on import
initialize_agents_module()

logger.info("Ultra AI Agents module loaded")
