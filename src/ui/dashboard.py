"""
Ultra AI Project - Web Dashboard

Provides a comprehensive web-based dashboard for monitoring and managing
the Ultra AI system, including real-time metrics, task management, and system health.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiofiles

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from ..core.system_manager import SystemManager
from ..core.task_coordinator import TaskCoordinator
from ..utils.logger import Logger
from ..utils.helpers import format_duration, format_bytes


@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""
    timestamp: datetime
    system_health: bool
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    error_rate: float


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = Logger(__name__)
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected: {len(self.active_connections)} active connections")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected: {len(self.active_connections)} active connections")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.warning(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.warning(f"WebSocket broadcast failed: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected WebSockets."""
        message = json.dumps(data, default=str)
        await self.broadcast(message)


class Dashboard:
    """Web dashboard for Ultra AI system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dashboard."""
        self.config = config
        self.logger = Logger(__name__)
        
        # System components
        self.system_manager: Optional[SystemManager] = None
        self.task_coordinator: Optional[TaskCoordinator] = None
        
        # Dashboard state
        self.metrics_history: List[DashboardMetrics] = []
        self.max_metrics_history = 1000
        
        # WebSocket manager
        self.websocket_manager = WebSocketManager()
        
        # FastAPI app
        self.app = FastAPI(
            title="Ultra AI Dashboard",
            description="Web dashboard for Ultra AI system monitoring and management",
            version="1.0.0"
        )
        
        # Setup app
        self._setup_middleware()
        self._setup_templates()
        self._setup_routes()
        
        # Metrics collection task
        self.metrics_task: Optional[asyncio.Task] = None
        self.metrics_interval = 5  # seconds
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Session middleware
        secret_key = self.config.get('secret_key', 'ultra-ai-dashboard-secret')
        self.app.add_middleware(SessionMiddleware, secret_key=secret_key)
    
    def _setup_templates(self):
        """Setup Jinja2 templates."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        
        # Create directories if they don't exist
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)
        
        self.templates = Jinja2Templates(directory=template_dir)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "title": "Ultra AI Dashboard"}
            )
        
        @self.app.get("/api/health")
        async def api_health():
            """Health check endpoint."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            health = await self.system_manager.get_health_status()
            return JSONResponse(health)
        
        @self.app.get("/api/status")
        async def api_status():
            """System status endpoint."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            status = await self._get_system_status()
            return JSONResponse(status)
        
        @self.app.get("/api/metrics")
        async def api_metrics():
            """System metrics endpoint."""
            if not self.metrics_history:
                return JSONResponse({"metrics": []})
            
            # Return last 100 metrics
            recent_metrics = self.metrics_history[-100:]
            metrics_data = [asdict(metric) for metric in recent_metrics]
            
            return JSONResponse({"metrics": metrics_data})
        
        @self.app.get("/api/agents")
        async def api_agents():
            """Available agents endpoint."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            agents = await self.system_manager.get_agents()
            return JSONResponse({"agents": agents})
        
        @self.app.get("/api/tasks")
        async def api_tasks(
            status: Optional[str] = None,
            agent: Optional[str] = None,
            limit: int = 50,
            offset: int = 0
        ):
            """Tasks endpoint."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            tasks = await self.task_coordinator.list_tasks(
                status=status,
                agent_type=agent,
                limit=limit,
                offset=offset
            )
            
            return JSONResponse({
                "tasks": [task.to_dict() for task in tasks],
                "total": len(tasks),
                "limit": limit,
                "offset": offset
            })
        
        @self.app.post("/api/tasks")
        async def api_create_task(request: Request):
            """Create task endpoint."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            data = await request.json()
            
            try:
                task = await self.task_coordinator.create_task(
                    agent_type=data.get('agent_type'),
                    description=data.get('description'),
                    parameters=data.get('parameters', {})
                )
                
                # Broadcast task creation
                await self.websocket_manager.broadcast_json({
                    "type": "task_created",
                    "task": task.to_dict()
                })
                
                return JSONResponse({"task": task.to_dict()})
                
            except Exception as e:
                self.logger.error(f"Failed to create task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/tasks/{task_id}")
        async def api_get_task(task_id: str):
            """Get specific task endpoint."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            task = await self.task_coordinator.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return JSONResponse({"task": task.to_dict()})
        
        @self.app.delete("/api/tasks/{task_id}")
        async def api_cancel_task(task_id: str):
            """Cancel task endpoint."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            try:
                await self.task_coordinator.cancel_task(task_id)
                
                # Broadcast task cancellation
                await self.websocket_manager.broadcast_json({
                    "type": "task_cancelled",
                    "task_id": task_id
                })
                
                return JSONResponse({"message": "Task cancelled successfully"})
                
            except Exception as e:
                self.logger.error(f"Failed to cancel task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/logs")
        async def api_logs(
            level: str = "INFO",
            limit: int = 100,
            since: Optional[str] = None
        ):
            """System logs endpoint."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            logs = await self.system_manager.get_recent_logs(
                level=level,
                limit=limit,
                since=since
            )
            
            return JSONResponse({"logs": logs})
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Wait for client messages (ping/pong)
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)
    
    async def initialize(self, system_manager: SystemManager, task_coordinator: TaskCoordinator):
        """Initialize the dashboard with system components."""
        self.logger.info("Initializing dashboard...")
        
        self.system_manager = system_manager
        self.task_coordinator = task_coordinator
        
        # Start metrics collection
        self.metrics_task = asyncio.create_task(self._collect_metrics())
        
        # Create default templates if they don't exist
        await self._create_default_templates()
        
        self.logger.info("Dashboard initialized successfully")
    
    async def shutdown(self):
        """Shutdown the dashboard."""
        self.logger.info("Shutting down dashboard...")
        
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections
        for connection in self.websocket_manager.active_connections.copy():
            try:
                await connection.close()
            except Exception:
                pass
        
        self.logger.info("Dashboard shutdown complete")
    
    async def _collect_metrics(self):
        """Continuously collect system metrics."""
        self.logger.info("Starting metrics collection...")
        
        while True:
            try:
                metrics = await self._gather_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # Limit history size
                    if len(self.metrics_history) > self.max_metrics_history:
                        self.metrics_history = self.metrics_history[-self.max_metrics_history:]
                    
                    # Broadcast metrics update
                    await self.websocket_manager.broadcast_json({
                        "type": "metrics_update",
                        "metrics": asdict(metrics)
                    })
                
                await asyncio.sleep(self.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.metrics_interval)
    
    async def _gather_metrics(self) -> Optional[DashboardMetrics]:
        """Gather current system metrics."""
        if not self.system_manager or not self.task_coordinator:
            return None
        
        try:
            # Get system health
            health = await self.system_manager.get_health_status()
            
            # Get task statistics
            task_stats = await self.task_coordinator.get_task_statistics()
            
            # Get resource usage
            resources = health.get('resources', {})
            
            metrics = DashboardMetrics(
                timestamp=datetime.now(),
                system_health=health.get('healthy', False),
                active_tasks=task_stats.get('active', 0),
                completed_tasks=task_stats.get('completed', 0),
                failed_tasks=task_stats.get('failed', 0),
                cpu_usage=resources.get('cpu_percent', 0.0),
                memory_usage=resources.get('memory_percent', 0.0),
                disk_usage=resources.get('disk_percent', 0.0),
                response_time=health.get('response_time', 0.0),
                error_rate=health.get('error_rate', 0.0)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to gather metrics: {e}")
            return None
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.system_manager:
            return {"error": "System not initialized"}
        
        try:
            health = await self.system_manager.get_health_status()
            agents = await self.system_manager.get_agents()
            
            if self.task_coordinator:
                task_stats = await self.task_coordinator.get_task_statistics()
            else:
                task_stats = {}
            
            status = {
                "health": health,
                "agents": agents,
                "tasks": task_stats,
                "uptime": health.get('uptime', 0),
                "version": self.config.get('version', '1.0.0'),
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def _create_default_templates(self):
        """Create default HTML templates."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        
        # Main dashboard template
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
        }
        .status-healthy { color: #28a745; }
        .status-unhealthy { color: #dc3545; }
        .status-pending { color: #ffc107; }
        .log-entry {
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            margin-bottom: 0.25rem;
        }
        .websocket-status {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1050;
        }
    </style>
</head>
<body>
    <!-- WebSocket Status -->
    <div id="websocket-status" class="websocket-status">
        <span class="badge bg-secondary">
            <i class="fas fa-circle" id="ws-indicator"></i>
            <span id="ws-text">Connecting...</span>
        </span>
    </div>

    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-robot"></i>
                Ultra AI Dashboard
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text" id="current-time"></span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- System Status Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-heartbeat fa-2x mb-2"></i>
                        <h5 class="card-title">System Health</h5>
                        <h3 id="system-health">-</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-tasks fa-2x mb-2"></i>
                        <h5 class="card-title">Active Tasks</h5>
                        <h3 id="active-tasks">-</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-microchip fa-2x mb-2"></i>
                        <h5 class="card-title">CPU Usage</h5>
                        <h3 id="cpu-usage">-</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <i class="fas fa-memory fa-2x mb-2"></i>
                        <h5 class="card-title">Memory Usage</h5>
                        <h3 id="memory-usage">-</h3>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts and Tables -->
        <div class="row">
            <!-- Metrics Chart -->
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line"></i> System Metrics</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="metricsChart" height="100"></canvas>
                    </div>
                </div>
            </div>

            <!-- Agents Status -->
            <div class="col-lg-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-robot"></i> Agents</h5>
                    </div>
                    <div class="card-body">
                        <div id="agents-list">Loading...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tasks and Logs -->
        <div class="row mt-4">
            <!-- Recent Tasks -->
            <div class="col-lg-6">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5><i class="fas fa-list"></i> Recent Tasks</h5>
                        <button class="btn btn-sm btn-primary" data-bs-toggle="modal" data-bs-target="#createTaskModal">
                            <i class="fas fa-plus"></i> Create Task
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="tasks-list">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- System Logs -->
            <div class="col-lg-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-file-alt"></i> System Logs</h5>
                    </div>
                    <div class="card-body">
                        <div id="logs-container" style="height: 300px; overflow-y: auto;">
                            <div id="logs-list">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Create Task Modal -->
    <div class="modal fade" id="createTaskModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Create New Task</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="createTaskForm">
                        <div class="mb-3">
                            <label for="taskAgent" class="form-label">Agent Type</label>
                            <select class="form-select" id="taskAgent" required>
                                <option value="">Select an agent...</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label for="taskDescription" class="form-label">Description</label>
                            <textarea class="form-control" id="taskDescription" rows="3" required></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="createTask()">Create Task</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Dashboard JavaScript code
        let socket = null;
        let metricsChart = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeWebSocket();
            initializeChart();
            loadInitialData();
            updateCurrentTime();
            setInterval(updateCurrentTime, 1000);
        });
        
        function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws';
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function(event) {
                updateWebSocketStatus('connected');
                console.log('WebSocket connected');
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            socket.onclose = function(event) {
                updateWebSocketStatus('disconnected');
                console.log('WebSocket disconnected');
                // Reconnect after 5 seconds
                setTimeout(initializeWebSocket, 5000);
            };
            
            socket.onerror = function(error) {
                updateWebSocketStatus('error');
                console.error('WebSocket error:', error);
            };
        }
        
        function updateWebSocketStatus(status) {
            const indicator = document.getElementById('ws-indicator');
            const text = document.getElementById('ws-text');
            
            switch(status) {
                case 'connected':
                    indicator.style.color = '#28a745';
                    text.textContent = 'Connected';
                    break;
                case 'disconnected':
                    indicator.style.color = '#dc3545';
                    text.textContent = 'Disconnected';
                    break;
                case 'error':
                    indicator.style.color = '#ffc107';
                    text.textContent = 'Error';
                    break;
            }
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'metrics_update':
                    updateMetrics(data.metrics);
                    break;
                case 'task_created':
                case 'task_cancelled':
                    loadTasks();
                    break;
            }
        }
        
        function initializeChart() {
            const ctx = document.getElementById('metricsChart').getContext('2d');
            metricsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }, {
                        label: 'Memory Usage (%)',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
        
        function updateMetrics(metrics) {
            // Update metric cards
            document.getElementById('system-health').textContent = metrics.system_health ? 'Healthy' : 'Unhealthy';
            document.getElementById('active-tasks').textContent = metrics.active_tasks;
            document.getElementById('cpu-usage').textContent = metrics.cpu_usage.toFixed(1) + '%';
            document.getElementById('memory-usage').textContent = metrics.memory_usage.toFixed(1) + '%';
            
            // Update chart
            const time = new Date(metrics.timestamp).toLocaleTimeString();
            metricsChart.data.labels.push(time);
            metricsChart.data.datasets[0].data.push(metrics.cpu_usage);
            metricsChart.data.datasets[1].data.push(metrics.memory_usage);
            
            // Keep only last 20 data points
            if (metricsChart.data.labels.length > 20) {
                metricsChart.data.labels.shift();
                metricsChart.data.datasets[0].data.shift();
                metricsChart.data.datasets[1].data.shift();
            }
            
            metricsChart.update('none');
        }
        
        async function loadInitialData() {
            await Promise.all([
                loadAgents(),
                loadTasks(),
                loadLogs()
            ]);
        }
        
        async function loadAgents() {
            try {
                const response = await fetch('/api/agents');
                const data = await response.json();
                
                const agentsList = document.getElementById('agents-list');
                const taskAgentSelect = document.getElementById('taskAgent');
                
                let agentsHtml = '';
                taskAgentSelect.innerHTML = '<option value="">Select an agent...</option>';
                
                for (const [name, info] of Object.entries(data.agents)) {
                    const statusClass = info.status === 'active' ? 'status-healthy' : 'status-unhealthy';
                    const statusIcon = info.status === 'active' ? 'fa-check-circle' : 'fa-times-circle';
                    
                    agentsHtml += `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <div>
                                <strong>${name}</strong>
                                <br><small class="text-muted">${info.type}</small>
                            </div>
                            <i class="fas ${statusIcon} ${statusClass}"></i>
                        </div>
                    `;
                    
                    if (info.status === 'active') {
                        taskAgentSelect.innerHTML += `<option value="${name}">${name} (${info.type})</option>`;
                    }
                }
                
                agentsList.innerHTML = agentsHtml;
            } catch (error) {
                console.error('Failed to load agents:', error);
            }
        }
        
        async function loadTasks() {
            try {
                const response = await fetch('/api/tasks?limit=10');
                const data = await response.json();
                
                const tasksList = document.getElementById('tasks-list');
                let tasksHtml = '';
                
                for (const task of data.tasks) {
                    const statusClass = {
                        'completed': 'status-healthy',
                        'failed': 'status-unhealthy',
                        'running': 'status-pending',
                        'pending': 'status-pending'
                    }[task.status] || '';
                    
                    const statusIcon = {
                        'completed': 'fa-check-circle',
                        'failed': 'fa-times-circle',
                        'running': 'fa-spinner fa-spin',
                        'pending': 'fa-clock'
                    }[task.status] || 'fa-question-circle';
                    
                    tasksHtml += `
                        <div class="d-flex justify-content-between align-items-center mb-2 p-2 border rounded">
                            <div>
                                <strong>${task.agent_type}</strong>
                                <br><small>${task.description.substring(0, 50)}...</small>
                                <br><small class="text-muted">${new Date(task.created_at).toLocaleString()}</small>
                            </div>
                            <i class="fas ${statusIcon} ${statusClass}"></i>
</div>
                    `;
                }
                
                tasksList.innerHTML = tasksHtml || '<p class="text-muted">No tasks found</p>';
            } catch (error) {
                console.error('Failed to load tasks:', error);
                document.getElementById('tasks-list').innerHTML = '<p class="text-danger">Failed to load tasks</p>';
            }
        }
        
        async function loadLogs() {
            try {
                const response = await fetch('/api/logs?limit=20');
                const data = await response.json();
                
                const logsList = document.getElementById('logs-list');
                let logsHtml = '';
                
                for (const log of data.logs) {
                    const levelClass = {
                        'ERROR': 'text-danger',
                        'WARNING': 'text-warning',
                        'INFO': 'text-info',
                        'DEBUG': 'text-muted'
                    }[log.level] || '';
                    
                    const timestamp = new Date(log.timestamp).toLocaleTimeString();
                    
                    logsHtml += `
                        <div class="log-entry">
                            <span class="text-muted">${timestamp}</span>
                            <span class="badge bg-secondary">${log.level}</span>
                            <span class="${levelClass}">${log.message}</span>
                        </div>
                    `;
                }
                
                logsList.innerHTML = logsHtml || '<p class="text-muted">No logs found</p>';
                
                // Auto-scroll to bottom
                const logsContainer = document.getElementById('logs-container');
                logsContainer.scrollTop = logsContainer.scrollHeight;
            } catch (error) {
                console.error('Failed to load logs:', error);
                document.getElementById('logs-list').innerHTML = '<p class="text-danger">Failed to load logs</p>';
            }
        }
        
        async function createTask() {
            const agent = document.getElementById('taskAgent').value;
            const description = document.getElementById('taskDescription').value;
            
            if (!agent || !description) {
                alert('Please fill in all fields');
                return;
            }
            
            try {
                const response = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        agent_type: agent,
                        description: description,
                        parameters: {}
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // Close modal
                    const modal = bootstrap.Modal.getInstance(document.getElementById('createTaskModal'));
                    modal.hide();
                    
                    // Reset form
                    document.getElementById('createTaskForm').reset();
                    
                    // Refresh tasks list
                    loadTasks();
                    
                    // Show success message
                    showNotification('Task created successfully', 'success');
                } else {
                    const error = await response.json();
                    showNotification('Failed to create task: ' + error.detail, 'danger');
                }
            } catch (error) {
                console.error('Failed to create task:', error);
                showNotification('Failed to create task', 'danger');
            }
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
            notification.style.top = '70px';
            notification.style.right = '20px';
            notification.style.zIndex = '1050';
            notification.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 5000);
        }
        
        function updateCurrentTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString();
        }
        
        // Refresh data periodically
        setInterval(() => {
            loadTasks();
            loadLogs();
        }, 30000); // Every 30 seconds
    </script>
</body>
</html>
        """
        
        dashboard_file = os.path.join(template_dir, "dashboard.html")
        if not os.path.exists(dashboard_file):
            async with aiofiles.open(dashboard_file, 'w') as f:
                await f.write(dashboard_html)
            self.logger.info("Created default dashboard template")


# Utility functions for dashboard

def get_task_status_icon(status: str) -> str:
    """Get appropriate icon for task status."""
    icons = {
        'pending': '⏳',
        'running': '🏃',
        'completed': '✅',
        'failed': '❌',
        'cancelled': '🛑'
    }
    return icons.get(status, '❓')


def get_agent_status_icon(status: str) -> str:
    """Get appropriate icon for agent status."""
    icons = {
        'active': '🟢',
        'inactive': '🔴',
        'busy': '🟡',
        'error': '❌'
    }
    return icons.get(status, '❓')


def format_task_duration(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """Format task duration for display."""
    if not end_time:
        end_time = datetime.now()
    
    duration = end_time - start_time
    total_seconds = int(duration.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def calculate_success_rate(completed: int, failed: int) -> float:
    """Calculate task success rate."""
    total = completed + failed
    if total == 0:
        return 100.0
    return (completed / total) * 100.0


def get_system_health_color(health_status: bool) -> str:
    """Get color for system health status."""
    return "#28a745" if health_status else "#dc3545"


def get_cpu_usage_color(usage: float) -> str:
    """Get color based on CPU usage level."""
    if usage < 50:
        return "#28a745"  # Green
    elif usage < 80:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def get_memory_usage_color(usage: float) -> str:
    """Get color based on memory usage level."""
    if usage < 70:
        return "#28a745"  # Green
    elif usage < 90:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


class DashboardConfig:
    """Dashboard configuration settings."""
    
    def __init__(self):
        self.refresh_interval = 5  # seconds
        self.max_tasks_display = 20
        self.max_logs_display = 50
        self.max_metrics_history = 1000
        self.chart_data_points = 50
        
        # Theme settings
        self.primary_color = "#667eea"
        self.secondary_color = "#764ba2"
        self.success_color = "#28a745"
        self.danger_color = "#dc3545"
        self.warning_color = "#ffc107"
        self.info_color = "#17a2b8"
        
        # WebSocket settings
        self.websocket_ping_interval = 30  # seconds
        self.websocket_reconnect_delay = 5  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "refresh_interval": self.refresh_interval,
            "max_tasks_display": self.max_tasks_display,
            "max_logs_display": self.max_logs_display,
            "max_metrics_history": self.max_metrics_history,
            "chart_data_points": self.chart_data_points,
            "colors": {
                "primary": self.primary_color,
                "secondary": self.secondary_color,
                "success": self.success_color,
                "danger": self.danger_color,
                "warning": self.warning_color,
                "info": self.info_color
            },
            "websocket": {
                "ping_interval": self.websocket_ping_interval,
                "reconnect_delay": self.websocket_reconnect_delay
            }
        }


# Dashboard factory function
def create_dashboard(config: Dict[str, Any]) -> Dashboard:
    """Create and configure a dashboard instance."""
    dashboard = Dashboard(config)
    return dashboard


# Dashboard middleware for authentication (optional)
class DashboardAuthMiddleware:
    """Optional authentication middleware for dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = Logger(__name__)
        self.security = HTTPBearer(auto_error=False)
    
    async def authenticate(self, credentials: Optional[HTTPAuthorizationCredentials] = None):
        """Authenticate dashboard access."""
        if not self.config.get('enable_auth', False):
            return True
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate token (implement your authentication logic here)
        valid_tokens = self.config.get('valid_tokens', [])
        if credentials.credentials not in valid_tokens:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
        
        return True


# Export dashboard components
__all__ = [
    'Dashboard',
    'DashboardMetrics',
    'WebSocketManager',
    'DashboardConfig',
    'DashboardAuthMiddleware',
    'create_dashboard',
    'get_task_status_icon',
    'get_agent_status_icon',
    'format_task_duration',
    'calculate_success_rate'
]
