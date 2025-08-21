from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid
import sys
from pathlib import Path

# Add src to path for absolute imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

class ToolCategory(Enum):
    CODE = "code"
    COMMUNICATION = "communication"

class ParameterType(Enum):
    STRING = "string"
    INTEGER = "integer"

@dataclass
class ToolParameter:
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None

@dataclass  
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None

class BaseTool(ABC):
    def __init__(self):
        self.tool_id = str(uuid.uuid4())
        logger.info(f"Initialized tool: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass
