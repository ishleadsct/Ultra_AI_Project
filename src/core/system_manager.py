"""
Ultra AI Project - System Manager

Central orchestration component that manages the entire Ultra AI system,
including initialization, configuration, agent lifecycle, and resource management.
"""

import asyncio
import os
import signal
import time
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from ..utils.logger import get_logger
from ..utils.helpers import load_config, get_system_info
from .task_coordinator import TaskCoordinator
from .memory_manager import MemoryManager
from .security import SecurityManager

logger = get_logger(__name__)

@dataclass
class SystemStatus:
    """System status information."""
    status: str
    uptime: float
    active_agents: int
    pending_tasks: int
    memory_usage: Dict[str, Any]
    last_health_check: datetime
    errors: List[str]

@dataclass
class SystemMetrics:
    """System performance metrics."""
    requests_processed: int
    avg_response_time: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    timestamp: datetime

class SystemManager:
    """Central system management and orchestration."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.status = "initializing"
        self.start_time = time.time()
        self.agents: Dict[str, Any] = {}
        self.active_tasks: Set[str] = set()
        self.health_checks: Dict[str, datetime] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.error_log: List[str] = []
        
        # Core components
        self.task_coordinator: Optional[TaskCoordinator] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.security_manager: Optional[SecurityManager] = None
        
        # Runtime state
        self.shutdown_event = asyncio.Event()
        self.health_check_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        logger.info("SystemManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the system and all components."""
        try:
            logger.info("Starting system initialization...")
            self.status = "initializing"
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Setup monitoring
            await self._setup_monitoring()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Validate system health
            if await self._validate_system_health():
                self.status = "running"
                logger.info("System initialization completed successfully")
                return True
            else:
                self.status = "error"
                logger.error("System validation failed")
                return False
                
        except Exception as e:
            self.status = "error"
            error_msg = f"System initialization failed: {e}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            return False
    
    async def _initialize_core_components(self):
        """Initialize core system components."""
        logger.info("Initializing core components...")
        
        # Initialize security manager
        self.security_manager = SecurityManager(self.config.get("security", {}))
        await self.security_manager.initialize()
        
        # Initialize memory manager
        memory_config = self.config.get("memory", {})
        self.memory_manager = MemoryManager(memory_config)
        await self.memory_manager.initialize()
        
        # Initialize task coordinator
        task_config = self.config.get("tasks", {})
        self.task_coordinator = TaskCoordinator(
            task_config, 
            self.memory_manager,
            self.security_manager
        )
        await self.task_coordinator.initialize()
        
        logger.info("Core components initialized")
    
    async def _initialize_agents(self):
        """Initialize and register agents."""
        logger.info("Initializing agents...")
        
        try:
            # Import agent classes
            from ..agents.code_agent import CodeAgent
            from ..agents.research_agent import ResearchAgent
            from ..agents.creative_agent import CreativeAgent
            from ..agents.analysis_agent import AnalysisAgent
            
            # Agent configurations
            agent_configs = self.config.get("agents", {})
            
            # Initialize agents
            agents_to_create = [
                ("code_agent", CodeAgent),
                ("research_agent", ResearchAgent),
                ("creative_agent", CreativeAgent),
                ("analysis_agent", AnalysisAgent),
            ]
            
            for agent_name, agent_class in agents_to_create:
                try:
                    agent_config = agent_configs.get(agent_name, {})
                    agent = agent_class(
                        agent_config,
                        self.memory_manager,
                        self.task_coordinator
                    )
                    
                    await agent.initialize()
                    self.agents[agent_name] = agent
                    logger.info(f"Agent '{agent_name}' initialized successfully")
                    
                except Exception as e:
                    error_msg = f"Failed to initialize agent '{agent_name}': {e}"
                    logger.error(error_msg)
                    self.error_log.append(error_msg)
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def _setup_monitoring(self):
        """Setup system monitoring tasks."""
        logger.info("Setting up system monitoring...")
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start metrics collection task
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("System monitoring setup complete")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _validate_system_health(self) -> bool:
        """Validate overall system health."""
        try:
            # Check core components
            components_healthy = all([
                self.security_manager and await self.security_manager.health_check(),
                self.memory_manager and await self.memory_manager.health_check(),
                self.task_coordinator and await self.task_coordinator.health_check(),
            ])
            
            if not components_healthy:
                logger.error("Core component health check failed")
                return False
            
            # Check agents
            agent_health = {}
            for name, agent in self.agents.items():
                try:
                    healthy = await agent.health_check()
                    agent_health[name] = healthy
                    if not healthy:
                        logger.warning(f"Agent '{name}' health check failed")
                except Exception as e:
                    logger.error(f"Health check failed for agent '{name}': {e}")
                    agent_health[name] = False
            
            # Log health status
            healthy_agents = sum(1 for healthy in agent_health.values() if healthy)
            logger.info(f"Agent health: {healthy_agents}/{len(self.agents)} healthy")
            
            return components_healthy and len(agent_health) > 0
            
        except Exception as e:
            logger.error(f"System health validation failed: {e}")
            return False
    
    async def _health_check_loop(self):
        """Continuous health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_check(self):
        """Perform a single health check cycle."""
        try:
            # Check system resources
            import psutil
            
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check component health
            component_status = {}
            for name, component in [
                ("security", self.security_manager),
                ("memory", self.memory_manager),
                ("tasks", self.task_coordinator),
            ]:
                if component:
                    component_status[name] = await component.health_check()
                else:
                    component_status[name] = False
            
            # Update health check timestamp
            self.health_checks["last_check"] = datetime.now()
            self.health_checks["memory_percent"] = memory_info.percent
            self.health_checks["cpu_percent"] = cpu_percent
            self.health_checks["components"] = component_status
            
            # Log warnings for resource usage
            if memory_info.percent > 80:
                logger.warning(f"High memory usage: {memory_info.percent}%")
            
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect and store system metrics."""
        while not self.shutdown_event.is_set():
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _collect_metrics(self):
        """Collect current system metrics."""
        try:
            import psutil
            
            # Get system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Get application metrics
            pending_tasks = len(self.active_tasks)
            active_agents = len([a for a in self.agents.values() if getattr(a, 'is_active', False)])
            
            # Calculate error rate (from last hour)
            recent_errors = len([
                error for error in self.error_log
                if datetime.now() - datetime.fromisoformat(error.split(" - ")[0]) < timedelta(hours=1)
            ]) if self.error_log else 0
            
            metrics = SystemMetrics(
                requests_processed=getattr(self, '_requests_processed', 0),
                avg_response_time=getattr(self, '_avg_response_time', 0.0),
                error_rate=recent_errors / max(getattr(self, '_requests_processed', 1), 1),
                memory_usage_mb=memory_info.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                active_connections=active_agents,
                timestamp=datetime.now()
            )
            
            # Store metrics (keep last 1440 = 24 hours of minute data)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1440:
                self.metrics_history.pop(0)
                
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def get_status(self) -> SystemStatus:
        """Get current system status."""
        uptime = time.time() - self.start_time
        
        memory_usage = {}
        if self.memory_manager:
            memory_usage = await self.memory_manager.get_usage_stats()
        
        return SystemStatus(
            status=self.status,
            uptime=uptime,
            active_agents=len(self.agents),
            pending_tasks=len(self.active_tasks),
            memory_usage=memory_usage,
            last_health_check=self.health_checks.get("last_check", datetime.now()),
            errors=self.error_log[-10:]  # Last 10 errors
        )
    
    async def get_metrics(self, hours: int = 1) -> List[SystemMetrics]:
        """Get system metrics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    async def get_agent(self, agent_name: str) -> Optional[Any]:
        """Get an agent by name."""
        return self.agents.get(agent_name)
    
    async def list_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.agents.keys())
    
    async def restart_agent(self, agent_name: str) -> bool:
        """Restart a specific agent."""
        try:
            if agent_name not in self.agents:
                logger.error(f"Agent '{agent_name}' not found")
                return False
            
            agent = self.agents[agent_name]
            
            # Stop the agent
            await agent.shutdown()
            
            # Reinitialize the agent
            await agent.initialize()
            
            logger.info(f"Agent '{agent_name}' restarted successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to restart agent '{agent_name}': {e}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            return False
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task through the task coordinator."""
        if not self.task_coordinator:
            raise RuntimeError("Task coordinator not initialized")
        
        task_id = task_data.get("id", f"task_{len(self.active_tasks)}")
        self.active_tasks.add(task_id)
        
        try:
            result = await self.task_coordinator.execute_task(task_data)
            return result
        finally:
            self.active_tasks.discard(task_id)
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        if self.status == "shutting_down":
            return
        
        logger.info("Initiating system shutdown...")
        self.status = "shutting_down"
        
        try:
            # Cancel monitoring tasks
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.metrics_task:
                self.metrics_task.cancel()
            
            # Shutdown agents
            for name, agent in self.agents.items():
                try:
                    await agent.shutdown()
                    logger.info(f"Agent '{name}' shutdown complete")
                except Exception as e:
                    logger.error(f"Error shutting down agent '{name}': {e}")
            
            # Shutdown core components
            if self.task_coordinator:
                await self.task_coordinator.shutdown()
            
            if self.memory_manager:
                await self.memory_manager.shutdown()
            
            if self.security_manager:
                await self.security_manager.shutdown()
            
            # Set shutdown event
            self.shutdown_event.set()
            
            self.status = "stopped"
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.status = "error"
