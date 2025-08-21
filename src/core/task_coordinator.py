"""
Ultra AI Project - Task Coordinator

Manages task scheduling, execution, monitoring, and coordination across agents.
Provides queuing, prioritization, retry logic, and distributed task execution.
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import generate_task_id

logger = get_logger(__name__)

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """Task definition and execution context."""
    id: str
    type: str
    agent_name: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary (excluding callback)."""
        data = asdict(self)
        data.pop('callback', None)  # Remove callback for serialization
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        return data

class TaskQueue:
    """Priority-based task queue."""
    
    def __init__(self):
        self.queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in TaskPriority
        }
        self.size = 0
    
    async def put(self, task: Task):
        """Add task to appropriate priority queue."""
        await self.queues[task.priority].put(task)
        self.size += 1
    
    async def get(self) -> Task:
        """Get highest priority task."""
        # Check queues in priority order (highest first)
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self.queues[priority]
            if not queue.empty():
                task = await queue.get()
                self.size -= 1
                return task
        
        # If no tasks available, wait for any task
        tasks = [queue.get() for queue in self.queues.values()]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Return the completed task
        task = done.pop().result()
        self.size -= 1
        return task
    
    def empty(self) -> bool:
        """Check if all queues are empty."""
        return self.size == 0
    
    def qsize(self) -> int:
        """Get total queue size."""
        return self.size

class TaskCoordinator:
    """Central task coordination and execution management."""
    
    def __init__(self, config: Dict[str, Any], memory_manager=None, security_manager=None):
        self.config = config
        self.memory_manager = memory_manager
        self.security_manager = security_manager
        
        # Task management
        self.task_queue = TaskQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_history: List[TaskResult] = []
        
        # Worker management
        self.max_workers = config.get("worker_concurrency", 4)
        self.workers: List[asyncio.Task] = []
        self.worker_stats: Dict[str, Dict] = {}
        
        # State management
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Metrics
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0.0,
            "queue_size": 0,
            "active_workers": 0
        }
        
        logger.info(f"TaskCoordinator initialized with {self.max_workers} workers")
    
    async def initialize(self):
        """Initialize the task coordinator."""
        try:
            logger.info("Initializing TaskCoordinator...")
            
            # Create worker tasks
            for i in range(self.max_workers):
                worker_id = f"worker_{i}"
                worker = asyncio.create_task(self._worker_loop(worker_id))
                self.workers.append(worker)
                self.worker_stats[worker_id] = {
                    "tasks_processed": 0,
                    "errors": 0,
                    "status": "idle",
                    "current_task": None
                }
            
            self.running = True
            logger.info(f"TaskCoordinator initialized with {len(self.workers)} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize TaskCoordinator: {e}")
            raise
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Update worker status
                self.worker_stats[worker_id]["status"] = "waiting"
                
                # Get next task (with timeout to allow periodic checks)
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Update worker status
                self.worker_stats[worker_id]["status"] = "processing"
                self.worker_stats[worker_id]["current_task"] = task.id
                
                # Execute task
                await self._execute_task(task, worker_id)
                
                # Update worker stats
                self.worker_stats[worker_id]["tasks_processed"] += 1
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.worker_stats[worker_id]["errors"] += 1
                await asyncio.sleep(1)  # Brief pause on error
        
        # Update final worker status
        self.worker_stats[worker_id]["status"] = "stopped"
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: Task, worker_id: str):
        """Execute a single task."""
        start_time = time.time()
        task.started_at = datetime.now()
        task.status = TaskStatus.RUNNING
        self.active_tasks[task.id] = task
        
        try:
            logger.info(f"Worker {worker_id} executing task {task.id} (type: {task.type})")
            
            # Check dependencies
            if not await self._check_dependencies(task):
                raise Exception("Task dependencies not satisfied")
            
            # Get agent for task execution
            agent = await self._get_agent_for_task(task)
            if not agent:
                raise Exception(f"No agent available for task type: {task.type}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.process_task(task),
                timeout=task.timeout
            )
            
            # Task completed successfully
            execution_time = time.time() - start_time
            task.completed_at = datetime.now()
            task.status = TaskStatus.COMPLETED
            
            task_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata=task.metadata
            )
            
            # Store result
            await self._store_task_result(task_result)
            
            # Execute callback if provided
            if task.callback:
                try:
                    await task.callback(task_result)
                except Exception as callback_error:
                    logger.error(f"Task callback error for {task.id}: {callback_error}")
            
            logger.info(f"Task {task.id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            await self._handle_task_timeout(task)
        except Exception as e:
            await self._handle_task_error(task, str(e))
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task.id, None)
            
            # Update metrics
            self.metrics["tasks_processed"] += 1
            self._update_metrics()
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                logger.warning(f"Task {task.id} dependency {dep_id} not completed")
                return False
            
            dep_result = self.completed_tasks[dep_id]
            if dep_result.status != TaskStatus.COMPLETED:
                logger.warning(f"Task {task.id} dependency {dep_id} failed")
                return False
        
        return True
    
    async def _get_agent_for_task(self, task: Task):
        """Get appropriate agent for task execution."""
        try:
            # This would be injected or resolved through a registry
            # For now, we'll assume the agent_name maps to available agents
            from ..agents.base_agent import get_agent_by_name
            return await get_agent_by_name(task.agent_name)
        except Exception as e:
            logger.error(f"Failed to get agent '{task.agent_name}': {e}")
            return None
    
    async def _handle_task_timeout(self, task: Task):
        """Handle task timeout."""
        logger.warning(f"Task {task.id} timed out after {task.timeout}s")
        
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        
        task_result = TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=f"Task timed out after {task.timeout}s",
            metadata=task.metadata
        )
        
        await self._store_task_result(task_result)
        self.metrics["tasks_failed"] += 1
    
    async def _handle_task_error(self, task: Task, error: str):
        """Handle task execution error."""
        logger.error(f"Task {task.id} failed: {error}")
        
        # Check if we should retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRY
            
            logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
            
            # Add back to queue with delay
            await asyncio.sleep(min(2 ** task.retry_count, 60))  # Exponential backoff
            await self.task_queue.put(task)
            return
        
        # Task failed permanently
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        
        task_result = TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=error,
            metadata=task.metadata
        )
        
        await self._store_task_result(task_result)
        self.metrics["tasks_failed"] += 1
    
    async def _store_task_result(self, result: TaskResult):
        """Store task result."""
        self.completed_tasks[result.task_id] = result
        self.task_history.append(result)
        
        # Keep history limited
        if len(self.task_history) > 10000:
            self.task_history = self.task_history[-5000:]  # Keep last 5000
        
        # Store in memory manager if available
        if self.memory_manager:
            await self.memory_manager.store_task_result(result)
    
    def _update_metrics(self):
        """Update performance metrics."""
        if self.task_history:
            # Calculate average execution time
            recent_tasks = self.task_history[-100:]  # Last 100 tasks
            total_time = sum(r.execution_time for r in recent_tasks if r.execution_time > 0)
            self.metrics["avg_execution_time"] = total_time / len(recent_tasks)
        
        self.metrics["queue_size"] = self.task_queue.qsize()
        self.metrics["active_workers"] = len([w for w in self.worker_stats.values() if w["status"] == "processing"])
    
    async def submit_task(self, 
                         task_type: str,
                         agent_name: str,
                         data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: float = 300.0,
                         dependencies: Optional[List[str]] = None,
                         callback: Optional[Callable] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new task for execution."""
        
        task_id = generate_task_id()
        
        task = Task(
            id=task_id,
            type=task_type,
            agent_name=agent_name,
            data=data,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies or [],
            callback=callback,
            metadata=metadata or {}
        )
        
        # Security check if security manager available
        if self.security_manager:
            if not await self.security_manager.authorize_task(task):
                raise PermissionError(f"Task {task_id} not authorized")
        
        await self.task_queue.put(task)
        logger.info(f"Task {task_id} submitted (type: {task_type}, agent: {agent_name})")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "progress": "running"
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "id": result.task_id,
                "status": result.status.value,
                "completed_at": result.timestamp.isoformat(),
                "execution_time": result.execution_time,
                "error": result.error
            }
        
        return None
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task."""
        return self.completed_tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Check if task is active
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            # Note: Actual cancellation of running task is complex
            # and would require cooperation from the agent
            logger.info(f"Task {task_id} marked for cancellation")
            return True
        
        return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "total_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "workers": {
                worker_id: stats for worker_id, stats in self.worker_stats.items()
            },
            "metrics": self.metrics
        }
    
    async def health_check(self) -> bool:
        """Perform health check."""
        try:
            # Check if workers are running
            active_workers = sum(1 for w in self.workers if not w.done())
            if active_workers == 0:
                return False
            
            # Check queue is responsive
            queue_size = self.task_queue.qsize()
            
            # Check for stuck tasks (running > 30 minutes)
            stuck_tasks = 0
            now = datetime.now()
            for task in self.active_tasks.values():
                if task.started_at and (now - task.started_at).total_seconds() > 1800:
                    stuck_tasks += 1
            
            if stuck_tasks > len(self.active_tasks) // 2:  # More than half stuck
                logger.warning(f"Many stuck tasks detected: {stuck_tasks}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"TaskCoordinator health check failed: {e}")
            return False
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task directly (for system manager interface)."""
        task_id = await self.submit_task(
            task_type=task_data.get("type", "generic"),
            agent_name=task_data.get("agent", "analysis_agent"),
            data=task_data.get("data", {}),
            priority=TaskPriority(task_data.get("priority", 2)),
            timeout=task_data.get("timeout", 300.0)
        )
        
        # Wait for completion with timeout
        timeout = task_data.get("timeout", 300.0)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_task_result(task_id)
            if result:
                return {
                    "task_id": task_id,
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "execution_time": result.execution_time
                }
            
            await asyncio.sleep(0.5)
        
        # Timeout
        await self.cancel_task(task_id)
        return {
            "task_id": task_id,
            "status": "timeout",
            "error": "Task execution timed out"
        }
    
    async def shutdown(self):
        """Shutdown the task coordinator."""
        logger.info("Shutting down TaskCoordinator...")
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Set shutdown event
        self.shutdown_event.set()
        
        logger.info("TaskCoordinator shutdown complete")
