"""
Ultra AI Project - Base Agent

Core agent framework providing the foundation for all specialized agents
with common functionality, lifecycle management, and communication protocols.
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..utils.logger import get_logger
from ..utils.helpers import generate_task_id, current_timestamp

logger = get_logger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentCapability(Enum):
    """Agent capability enumeration."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    TESTING = "testing"
    RESEARCH = "research"
    WEB_SEARCH = "web_search"
    DOCUMENT_ANALYSIS = "document_analysis"
    SUMMARIZATION = "summarization"
    CREATIVE_WRITING = "creative_writing"
    STORYTELLING = "storytelling"
    CONTENT_GENERATION = "content_generation"
    DATA_ANALYSIS = "data_analysis"
    REASONING = "reasoning"
    PROBLEM_SOLVING = "problem_solving"
    CONVERSATION = "conversation"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"

@dataclass
class AgentConfig:
    """Agent configuration settings."""
    name: str
    agent_type: str
    max_concurrent_tasks: int = 3
    timeout: float = 300.0  # 5 minutes
    memory_limit: int = 1000
    capabilities: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    enable_memory: bool = True
    enable_learning: bool = False
    enable_collaboration: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """Task structure for agent processing."""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMemory:
    """Agent memory item."""
    memory_id: str
    content: Any
    memory_type: str
    importance: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None

class BaseAgent(ABC):
    """Base class for all AI agents providing core functionality."""
    
    def __init__(self, config: Optional[AgentConfig] = None, 
                 model_manager=None, memory_manager=None, task_coordinator=None):
        # Agent identification
        self.agent_id = str(uuid.uuid4())
        self.config = config or AgentConfig(name="base_agent", agent_type="base")
        self.agent_type = self.config.agent_type
        self.name = self.config.name
        
        # Core components
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.task_coordinator = task_coordinator
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Task management
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: List[Task] = []
        self.total_tasks_processed = 0
        
        # Memory management
        self.agent_memory: Dict[str, AgentMemory] = {}
        self.conversation_contexts: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0,
            "uptime": 0.0,
            "error_rate": 0.0
        }
        
        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Capabilities
        self.capabilities = set(self.config.capabilities)
        
        logger.info(f"Created {self.agent_type} agent: {self.agent_id}")
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            logger.info(f"Initializing agent {self.agent_id}...")
            
            # Validate configuration
            if not await self._validate_config():
                return False
            
            # Initialize components
            await self._initialize_components()
            
            # Start worker tasks
            await self._start_workers()
            
            # Agent-specific initialization
            await self._agent_initialize()
            
            self.status = AgentStatus.IDLE
            self.running = True
            
            logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def _validate_config(self) -> bool:
        """Validate agent configuration."""
        try:
            # Check required fields
            if not self.config.name:
                logger.error("Agent name is required")
                return False
            
            if not self.config.agent_type:
                logger.error("Agent type is required")
                return False
            
            # Validate numeric values
            if self.config.max_concurrent_tasks <= 0:
                logger.error("max_concurrent_tasks must be positive")
                return False
            
            if self.config.timeout <= 0:
                logger.error("timeout must be positive")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialize agent components."""
        # Component initialization can be overridden by subclasses
        pass
    
    async def _start_workers(self):
        """Start worker tasks for processing."""
        # Start task processing workers
        for i in range(self.config.max_concurrent_tasks):
            worker = asyncio.create_task(self._task_worker(f"worker_{i}"))
            self.worker_tasks.append(worker)
        
        # Start maintenance worker
        maintenance_worker = asyncio.create_task(self._maintenance_worker())
        self.worker_tasks.append(maintenance_worker)
        
        logger.debug(f"Started {len(self.worker_tasks)} workers for agent {self.agent_id}")
    
    async def _task_worker(self, worker_id: str):
        """Worker task for processing agent tasks."""
        logger.debug(f"Task worker {worker_id} started for agent {self.agent_id}")
        
        while self.running:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=5.0)
                
                # Update agent status
                self.status = AgentStatus.BUSY
                self.last_activity = datetime.now()
                
                # Process the task
                await self._process_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
                # Update status
                if len(self.active_tasks) == 0:
                    self.status = AgentStatus.IDLE
                    
            except asyncio.TimeoutError:
                # No tasks available, continue
                if len(self.active_tasks) == 0:
                    self.status = AgentStatus.IDLE
                continue
            except asyncio.CancelledError:
                logger.debug(f"Task worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Task worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.debug(f"Task worker {worker_id} stopped")
    
    async def _maintenance_worker(self):
        """Maintenance worker for periodic tasks."""
        logger.debug(f"Maintenance worker started for agent {self.agent_id}")
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Update metrics
                await self._update_metrics()
                
                # Clean up old tasks
                await self._cleanup_completed_tasks()
                
                # Clean up memory
                if self.config.enable_memory:
                    await self._cleanup_memory()
                
                # Agent-specific maintenance
                await self._agent_maintenance()
                
            except asyncio.CancelledError:
                logger.debug("Maintenance worker cancelled")
                break
            except Exception as e:
                logger.error(f"Maintenance worker error: {e}")
        
        logger.debug("Maintenance worker stopped")
    
    async def _process_task(self, task: Task):
        """Process a single task."""
        start_time = time.time()
        task.started_at = datetime.now()
        task.status = TaskStatus.RUNNING
        self.active_tasks[task.task_id] = task
        
        try:
            logger.info(f"Processing task {task.task_id} (type: {task.task_type})")
            
            # Apply timeout if specified
            timeout = task.timeout or self.config.timeout
            
            # Execute the task with timeout
            result = await asyncio.wait_for(
                self._execute_task(task),
                timeout=timeout
            )
            
            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["tasks_completed"] += 1
            self.metrics["total_processing_time"] += processing_time
            self.metrics["avg_processing_time"] = (
                self.metrics["total_processing_time"] / 
                (self.metrics["tasks_completed"] + self.metrics["tasks_failed"])
            )
            
            logger.info(f"Task {task.task_id} completed in {processing_time:.2f}s")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {timeout}s"
            task.completed_at = datetime.now()
            
            self.metrics["tasks_failed"] += 1
            logger.warning(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            # Move to completed tasks
            self.active_tasks.pop(task.task_id, None)
            self.completed_tasks.append(task)
            self.total_tasks_processed += 1
            
            # Store in memory if enabled
            if self.config.enable_memory:
                await self._store_task_memory(task)
    
    @abstractmethod
    async def _execute_task(self, task: Task) -> Any:
        """Execute a specific task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _agent_initialize(self):
        """Agent-specific initialization. Can be overridden by subclasses."""
        pass
    
    async def _agent_maintenance(self):
        """Agent-specific maintenance. Can be overridden by subclasses."""
        pass
    
    async def submit_task(self, task_type: str, data: Dict[str, Any], 
                         priority: int = 1, timeout: Optional[float] = None) -> str:
        """Submit a task for processing."""
        task_id = generate_task_id()
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Add to queue
        await self.task_queue.put(task)
        
        logger.debug(f"Task {task_id} submitted to agent {self.agent_id}")
        return task_id
    
    async def process_task(self, task_data: Dict[str, Any]) -> Any:
        """Process a task directly (for external callers)."""
        task_id = await self.submit_task(
            task_type=task_data.get("type", "generic"),
            data=task_data.get("data", {}),
            priority=task_data.get("priority", 1),
            timeout=task_data.get("timeout")
        )
        
        # Wait for completion
        timeout = task_data.get("timeout", self.config.timeout)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if task is completed
            completed_task = None
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    completed_task = task
                    break
            
            if completed_task:
                if completed_task.status == TaskStatus.COMPLETED:
                    return completed_task.result
                else:
                    raise Exception(completed_task.error or "Task failed")
            
            await asyncio.sleep(0.1)
        
        # Task timed out
        raise TimeoutError(f"Task {task_id} timed out")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "result": task.result,
                    "error": task.error
                }
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Check if task is in active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Move to completed
            self.active_tasks.pop(task_id, None)
            self.completed_tasks.append(task)
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return list(self.capabilities)
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities
    
    async def store_memory(self, content: Any, memory_type: str = "general",
                          importance: float = 1.0, tags: Optional[List[str]] = None) -> str:
        """Store information in agent memory."""
        if not self.config.enable_memory:
            return ""
        
        memory_id = str(uuid.uuid4())
        memory = AgentMemory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or []
        )
        
        self.agent_memory[memory_id] = memory
        
        # Also store in global memory manager if available
        if self.memory_manager:
            await self.memory_manager.store_memory(
                memory_type=f"agent_{self.agent_id}_{memory_type}",
                content=content,
                metadata={"agent_id": self.agent_id, "memory_type": memory_type},
                importance=importance,
                tags=tags
            )
        
        return memory_id
    
    async def retrieve_memory(self, memory_type: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             limit: int = 10) -> List[AgentMemory]:
        """Retrieve memories from agent memory."""
        memories = list(self.agent_memory.values())
        
        # Filter by type
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Filter by tags
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        # Sort by importance and recency
        memories.sort(key=lambda x: (x.importance, x.accessed_at), reverse=True)
        
        # Update access information
        for memory in memories[:limit]:
            memory.accessed_at = datetime.now()
            memory.access_count += 1
        
        return memories[:limit]
    
    async def _store_task_memory(self, task: Task):
        """Store task result in memory."""
        if task.status == TaskStatus.COMPLETED:
            await self.store_memory(
                content={
                    "task_type": task.task_type,
                    "data": task.data,
                    "result": task.result,
                    "processing_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else 0
                },
                memory_type="task_result",
                importance=2.0,
                tags=["task", task.task_type]
            )
    
    async def _update_metrics(self):
        """Update agent metrics."""
        now = datetime.now()
        self.metrics["uptime"] = (now - self.created_at).total_seconds()
        
        total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        if total_tasks > 0:
            self.metrics["error_rate"] = self.metrics["tasks_failed"] / total_tasks
    
    async def _cleanup_completed_tasks(self):
        """Clean up old completed tasks."""
        # Keep only last 100 completed tasks
        if len(self.completed_tasks) > 100:
            self.completed_tasks = self.completed_tasks[-50:]
    
    async def _cleanup_memory(self):
        """Clean up old memory items."""
        if len(self.agent_memory) <= self.config.memory_limit:
            return
        
        # Sort by importance and access time
        memories = list(self.agent_memory.values())
        memories.sort(key=lambda x: (x.importance, x.accessed_at))
        
        # Remove least important/accessed memories
        to_remove = len(memories) - self.config.memory_limit
        for memory in memories[:to_remove]:
            del self.agent_memory[memory.memory_id]
        
        logger.debug(f"Cleaned up {to_remove} memory items for agent {self.agent_id}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "active_tasks": len(self.active_tasks),
            "total_tasks_processed": self.total_tasks_processed,
            "capabilities": list(self.capabilities),
            "metrics": self.metrics,
            "memory_items": len(self.agent_memory) if self.config.enable_memory else 0
        }
    
    async def health_check(self) -> bool:
        """Perform agent health check."""
        try:
            # Check if agent is running
            if not self.running:
                return False
            
            # Check worker tasks
            active_workers = sum(1 for worker in self.worker_tasks if not worker.done())
            if active_workers == 0:
                return False
            
            # Check if agent is responsive
            if self.status == AgentStatus.ERROR:
                return False
            
            # Check last activity (agent should be active within last 10 minutes)
            time_since_activity = (datetime.now() - self.last_activity).total_seconds()
            if time_since_activity > 600:  # 10 minutes
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for agent {self.agent_id}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the agent gracefully."""
        logger.info(f"Shutting down agent {self.agent_id}...")
        
        self.running = False
        self.status = AgentStatus.SHUTDOWN
        
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Cancel worker tasks
        for worker in self.worker_tasks:
            worker.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Agent-specific shutdown
        await self._agent_shutdown()
        
        logger.info(f"Agent {self.agent_id} shutdown complete")
    
    async def _agent_shutdown(self):
        """Agent-specific shutdown. Can be overridden by subclasses."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
