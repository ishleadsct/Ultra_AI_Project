"""
Ultra AI Project - Web Interface

Provides a comprehensive web interface for the Ultra AI system including
REST API endpoints, file upload/download, user management, and real-time features.
"""

import asyncio
import os
import json
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import aiofiles
import uuid

from fastapi import (
    FastAPI, Request, Response, Depends, HTTPException, 
    UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
)
from fastapi.responses import (
    JSONResponse, HTMLResponse, FileResponse, StreamingResponse, RedirectResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from ..core.system_manager import SystemManager
from ..core.task_coordinator import TaskCoordinator
from ..agents.base_agent import BaseAgent
from ..utils.logger import Logger
from ..utils.helpers import generate_id, format_bytes, validate_file_type
from ..utils.security import verify_token, hash_password, create_access_token


# Pydantic models for API requests/responses

class TaskCreateRequest(BaseModel):
    """Request model for creating a task."""
    agent_type: str = Field(..., description="Type of agent to use")
    description: str = Field(..., description="Task description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    priority: int = Field(default=0, description="Task priority (0-10)")
    timeout: Optional[int] = Field(default=None, description="Task timeout in seconds")


class TaskResponse(BaseModel):
    """Response model for task operations."""
    id: str
    agent_type: str
    description: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Any]
    error: Optional[str]
    progress: Optional[float]


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    username: str
    password: str


class UserCreateRequest(BaseModel):
    """Request model for creating a user."""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    roles: List[str] = Field(default_factory=list)


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""
    section: str
    key: str
    value: Any


class FileUploadResponse(BaseModel):
    """Response model for file uploads."""
    file_id: str
    filename: str
    size: int
    mime_type: str
    upload_time: datetime
    url: str


@dataclass
class WebSocketConnection:
    """WebSocket connection data."""
    websocket: WebSocket
    user_id: Optional[str]
    connection_time: datetime
    last_ping: datetime


class WebInterface:
    """Web interface for Ultra AI system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the web interface."""
        self.config = config
        self.logger = Logger(__name__)
        
        # System components
        self.system_manager: Optional[SystemManager] = None
        self.task_coordinator: Optional[TaskCoordinator] = None
        
        # Web interface state
        self.upload_dir = Path(config.get('upload_dir', './data/uploads'))
        self.max_file_size = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.allowed_file_types = config.get('allowed_file_types', [
            'text/plain', 'text/csv', 'application/json', 'application/pdf',
            'image/jpeg', 'image/png', 'image/gif', 'audio/wav', 'audio/mp3'
        ])
        
        # WebSocket connections
        self.websocket_connections: Dict[str, WebSocketConnection] = {}
        
        # Security
        self.security = HTTPBearer(auto_error=False)
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Ultra AI Web Interface",
            description="Comprehensive web interface for Ultra AI system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup application
        self._setup_middleware()
        self._setup_static_files()
        self._setup_routes()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
    
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
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Session middleware
        secret_key = self.config.get('secret_key', 'ultra-ai-web-secret')
        self.app.add_middleware(SessionMiddleware, secret_key=secret_key)
        
        # Trusted host middleware
        allowed_hosts = self.config.get('allowed_hosts', ["*"])
        if allowed_hosts != ["*"]:
            self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    
    def _setup_static_files(self):
        """Setup static file serving."""
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)
        
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Setup templates
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.templates = Jinja2Templates(directory=str(template_dir))
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        # Health and status endpoints
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            if not self.system_manager:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "message": "System not initialized"}
                )
            
            health = await self.system_manager.get_health_status()
            status_code = 200 if health.get('healthy', False) else 503
            
            return JSONResponse(status_code=status_code, content=health)
        
        @self.app.get("/api/v1/status")
        async def get_system_status(current_user=Depends(self.get_current_user)):
            """Get comprehensive system status."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            return await self._get_system_status()
        
        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(request: UserLoginRequest):
            """User login endpoint."""
            try:
                # Validate credentials (implement your authentication logic)
                if await self._validate_credentials(request.username, request.password):
                    token = create_access_token({"sub": request.username})
                    return {"access_token": token, "token_type": "bearer"}
                else:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                raise HTTPException(status_code=500, detail="Authentication error")
        
        @self.app.post("/auth/logout")
        async def logout(current_user=Depends(self.get_current_user)):
            """User logout endpoint."""
            # Implement logout logic (e.g., token blacklisting)
            return {"message": "Logged out successfully"}
        
        # Agent management endpoints
        @self.app.get("/api/v1/agents")
        async def list_agents(current_user=Depends(self.get_current_user)):
            """List available agents."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            agents = await self.system_manager.get_agents()
            return {"agents": agents}
        
        @self.app.get("/api/v1/agents/{agent_name}")
        async def get_agent_details(agent_name: str, current_user=Depends(self.get_current_user)):
            """Get detailed information about a specific agent."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            agent_info = await self.system_manager.get_agent_info(agent_name)
            if not agent_info:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            return {"agent": agent_info}
        
        # Task management endpoints
        @self.app.post("/api/v1/tasks", response_model=TaskResponse)
        async def create_task(
            request: TaskCreateRequest,
            background_tasks: BackgroundTasks,
            current_user=Depends(self.get_current_user)
        ):
            """Create a new task."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            try:
                task = await self.task_coordinator.create_task(
                    agent_type=request.agent_type,
                    description=request.description,
                    parameters=request.parameters,
                    priority=request.priority,
                    timeout=request.timeout,
                    user_id=current_user.get('sub')
                )
                
                # Notify via WebSocket
                background_tasks.add_task(
                    self._broadcast_websocket_message,
                    {"type": "task_created", "task": task.to_dict()}
                )
                
                return TaskResponse(**task.to_dict())
                
            except Exception as e:
                self.logger.error(f"Failed to create task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v1/tasks")
        async def list_tasks(
            status: Optional[str] = None,
            agent_type: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
            current_user=Depends(self.get_current_user)
        ):
            """List tasks with filtering options."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            tasks = await self.task_coordinator.list_tasks(
                status=status,
                agent_type=agent_type,
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            
            return {
                "tasks": [task.to_dict() for task in tasks],
                "total": len(tasks),
                "limit": limit,
                "offset": offset
            }
        
        @self.app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
        async def get_task(task_id: str, current_user=Depends(self.get_current_user)):
            """Get specific task details."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            task = await self.task_coordinator.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return TaskResponse(**task.to_dict())
        
        @self.app.delete("/api/v1/tasks/{task_id}")
        async def cancel_task(
            task_id: str,
            background_tasks: BackgroundTasks,
            current_user=Depends(self.get_current_user)
        ):
            """Cancel a task."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            try:
                await self.task_coordinator.cancel_task(task_id)
                
                # Notify via WebSocket
                background_tasks.add_task(
                    self._broadcast_websocket_message,
                    {"type": "task_cancelled", "task_id": task_id}
                )
                
                return {"message": "Task cancelled successfully"}
                
            except Exception as e:
                self.logger.error(f"Failed to cancel task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/v1/tasks/{task_id}/retry")
        async def retry_task(
            task_id: str,
            background_tasks: BackgroundTasks,
            current_user=Depends(self.get_current_user)
        ):
            """Retry a failed task."""
            if not self.task_coordinator:
                raise HTTPException(status_code=503, detail="Task coordinator not initialized")
            
            try:
                new_task = await self.task_coordinator.retry_task(task_id)
                
                # Notify via WebSocket
                background_tasks.add_task(
                    self._broadcast_websocket_message,
                    {"type": "task_retried", "original_task_id": task_id, "new_task": new_task.to_dict()}
                )
                
                return {"new_task": new_task.to_dict()}
                
            except Exception as e:
                self.logger.error(f"Failed to retry task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # File management endpoints
        @self.app.post("/api/v1/files", response_model=FileUploadResponse)
        async def upload_file(
            file: UploadFile = File(...),
            description: Optional[str] = Form(None),
            current_user=Depends(self.get_current_user)
        ):
            """Upload a file."""
            try:
                # Validate file
                if file.size > self.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {format_bytes(self.max_file_size)}"
                    )
                
                if file.content_type not in self.allowed_file_types:
                    raise HTTPException(
                        status_code=415,
                        detail=f"File type not allowed: {file.content_type}"
                    )
                
                # Generate unique filename
                file_id = generate_id()
                file_extension = Path(file.filename).suffix
                safe_filename = f"{file_id}{file_extension}"
                file_path = self.upload_dir / safe_filename
                
                # Create upload directory if it doesn't exist
                self.upload_dir.mkdir(parents=True, exist_ok=True)
                
                # Save file
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)
                
                # Store file metadata
                file_metadata = {
                    "id": file_id,
                    "original_filename": file.filename,
                    "safe_filename": safe_filename,
                    "size": len(content),
                    "mime_type": file.content_type,
                    "description": description,
                    "uploaded_by": current_user.get('sub'),
                    "upload_time": datetime.now(),
                    "path": str(file_path)
                }
                
                await self._store_file_metadata(file_metadata)
                
                return FileUploadResponse(
                    file_id=file_id,
                    filename=file.filename,
                    size=len(content),
                    mime_type=file.content_type,
                    upload_time=datetime.now(),
                    url=f"/api/v1/files/{file_id}"
                )
                
            except Exception as e:
                self.logger.error(f"File upload error: {e}")
                raise HTTPException(status_code=500, detail="File upload failed")
        
        @self.app.get("/api/v1/files/{file_id}")
        async def download_file(file_id: str, current_user=Depends(self.get_current_user)):
            """Download a file."""
            try:
                file_metadata = await self._get_file_metadata(file_id)
                if not file_metadata:
                    raise HTTPException(status_code=404, detail="File not found")
                
                file_path = Path(file_metadata['path'])
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found on disk")
                
                return FileResponse(
                    path=str(file_path),
                    filename=file_metadata['original_filename'],
                    media_type=file_metadata['mime_type']
                )
                
            except Exception as e:
                self.logger.error(f"File download error: {e}")
                raise HTTPException(status_code=500, detail="File download failed")
        
        @self.app.delete("/api/v1/files/{file_id}")
        async def delete_file(file_id: str, current_user=Depends(self.get_current_user)):
            """Delete a file."""
            try:
                file_metadata = await self._get_file_metadata(file_id)
                if not file_metadata:
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Check permissions (users can only delete their own files)
                if file_metadata['uploaded_by'] != current_user.get('sub'):
                    raise HTTPException(status_code=403, detail="Permission denied")
                
                # Delete file from disk
                file_path = Path(file_metadata['path'])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove metadata
                await self._delete_file_metadata(file_id)
                
                return {"message": "File deleted successfully"}
                
            except Exception as e:
                self.logger.error(f"File deletion error: {e}")
                raise HTTPException(status_code=500, detail="File deletion failed")
        
        @self.app.get("/api/v1/files")
        async def list_files(
            limit: int = 50,
            offset: int = 0,
            current_user=Depends(self.get_current_user)
        ):
            """List uploaded files."""
            try:
                files = await self._list_user_files(current_user.get('sub'), limit, offset)
                return {
                    "files": files,
                    "total": len(files),
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                self.logger.error(f"File listing error: {e}")
                raise HTTPException(status_code=500, detail="File listing failed")
        
        # Configuration endpoints
        @self.app.get("/api/v1/config")
        async def get_config(current_user=Depends(self.get_current_user)):
            """Get system configuration (sanitized)."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            config = await self.system_manager.get_sanitized_config()
            return {"config": config}
        
        @self.app.put("/api/v1/config")
        async def update_config(
            request: ConfigUpdateRequest,
            current_user=Depends(self.get_current_user)
        ):
            """Update system configuration."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            try:
                await self.system_manager.update_config(
                    request.section,
                    request.key,
                    request.value
                )
                
                return {"message": "Configuration updated successfully"}
                
            except Exception as e:
                self.logger.error(f"Config update error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # Logs endpoints
        @self.app.get("/api/v1/logs")
        async def get_logs(
            level: str = "INFO",
            limit: int = 100,
            since: Optional[str] = None,
            current_user=Depends(self.get_current_user)
        ):
            """Get system logs."""
            if not self.system_manager:
                raise HTTPException(status_code=503, detail="System not initialized")
            
            logs = await self.system_manager.get_recent_logs(
                level=level,
                limit=limit,
                since=since
            )
            
            return {"logs": logs}
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self._handle_websocket_connection(websocket)
        
        # Main web pages
        @self.app.get("/", response_class=HTMLResponse)
        async def web_home(request: Request):
            """Main web interface."""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "Ultra AI System"}
            )
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def web_dashboard(request: Request):
            """Dashboard web page."""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "title": "Ultra AI Dashboard"}
            )
        
        @self.app.get("/tasks", response_class=HTMLResponse)
        async def web_tasks(request: Request):
            """Tasks management web page."""
            return self.templates.TemplateResponse(
                "tasks.html",
                {"request": request, "title": "Task Management"}
            )
    
    async def initialize(self, system_manager: SystemManager, task_coordinator: TaskCoordinator):
        """Initialize the web interface."""
        self.logger.info("Initializing web interface...")
        
        self.system_manager = system_manager
        self.task_coordinator = task_coordinator
        
        # Create upload directory
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Start background tasks
        self.background_tasks.append(
            asyncio.create_task(self._websocket_ping_task())
        )
        
        self.logger.info("Web interface initialized successfully")
    
    async def shutdown(self):
        """Shutdown the web interface."""
        self.logger.info("Shutting down web interface...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connections
        for connection_id, connection in self.websocket_connections.items():
            try:
                await connection.websocket.close()
            except Exception:
                pass
        
        self.websocket_connections.clear()
        
        self.logger.info("Web interface shutdown complete")
    
    # Authentication and authorization methods
    
    async def get_current_user(self, token: Optional[str] = Depends(lambda: None)):
        """Get current authenticated user."""
        # If authentication is disabled, return a default user
        if not self.config.get('enable_auth', True):
            return {"sub": "anonymous", "roles": ["user"]}
        
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        try:
            payload = verify_token(token)
            return payload
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials."""
        # Implement your authentication logic here
        # This is a placeholder implementation
        return username == "admin" and password == "admin"
    
    # WebSocket management methods
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection."""
        connection_id = generate_id()
        
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                websocket=websocket,
                user_id=None,  # Will be set after authentication
                connection_time=datetime.now(),
                last_ping=datetime.now()
            )
            
            self.websocket_connections[connection_id] = connection
            self.logger.info(f"WebSocket connected: {connection_id}")
            
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_websocket_message(connection_id, message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    self.logger.error(f"WebSocket message error: {e}")
                    break
        
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
        
        finally:
            # Clean up connection
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
            self.logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def _handle_websocket_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = message.get('type')
        connection = self.websocket_connections.get(connection_id)
        
        if not connection:
            return
        
        if message_type == 'ping':
            connection.last_ping = datetime.now()
            await connection.websocket.send_text(json.dumps({"type": "pong"}))
        
        elif message_type == 'subscribe':
            # Handle subscription to specific events
            subscription = message.get('subscription')
            self.logger.info(f"WebSocket {connection_id} subscribed to {subscription}")
        
        elif message_type == 'authenticate':
            # Handle WebSocket authentication
            token = message.get('token')
            if token:
                try:
                    payload = verify_token(token)
                    connection.user_id = payload.get('sub')
                    await connection.websocket.send_text(json.dumps({
                        "type": "authenticated",
                        "user_id": connection.user_id
                    }))
                except Exception:
                    await connection.websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "message": "Invalid token"
                    }))
    
    async def _broadcast_websocket_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSockets."""
        if not self.websocket_connections:
            return
        
        message_text = json.dumps(message, default=str)
        disconnected = []
        
        for connection_id, connection in self.websocket_connections.items():
            try:
                await connection.websocket.send_text(message_text)
            except Exception as e:
                self.logger.warning(f"Failed to send WebSocket message to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Remove disconnected connections
        for connection_id in disconnected:
            del self.websocket_connections[connection_id]
    
    async def _websocket_ping_task(self):
        """Background task to ping WebSocket connections."""
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(minutes=5)
                
                disconnected = []
                
                for connection_id, connection in self.websocket_connections.items():
                    if connection.last_ping < timeout_threshold:
                        disconnected.append(connection_id)
                    else:
                        try:
                            await connection.websocket.send_text(json.dumps({"type": "ping"}))
                        except Exception:
                            disconnected.append(connection_id)
                
                # Remove timed out connections
                for connection_id in disconnected:
                    if connection_id in self.websocket_connections:
                        del self.websocket_connections[connection_id]
                        self.logger.info(f"WebSocket timeout: {connection_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"WebSocket ping task error: {e}")
    
    # File management methods
    
    async def _store_file_metadata(self, metadata: Dict[str, Any]):
        """Store file metadata."""
        # Implement file metadata storage (database, etc.)
        # This is a placeholder implementation
        metadata_file = self.upload_dir / f"{metadata['id']}.metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, default=str))
    
    async def _get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata."""
        metadata_file = self.upload_dir / f"{file_id}.metadata.json"
        if not metadata_file.exists():
            return None
        
        async with aiofiles.open(metadata_file, 'r') as f:
            content = await f.read()
            return json.loads(content)
    
    async def _delete_file_metadata(self, file_id: str):
        """Delete file metadata."""
        metadata_file = self.upload_dir / f"{file_id}.metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
    
    async def _list_user_files(self, user_id: str, limit: int, offset: int) -> List[Dict[str, Any]]:
        """List files uploaded by a specific user."""
        files = []
        metadata_files = list(self.upload_dir.glob("*.metadata.json"))
        
        for metadata_file in metadata_files[offset:offset + limit]:
            try:
                async with aiofiles.open(metadata_file, 'r') as f:
                    content = await f.read()
                    metadata = json.loads(content)
                    if metadata.get('uploaded_by') == user_id:
                        files.append(metadata)
            except Exception:
                continue
        
        return files
    
    # Utility methods
    
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
            
            return {
                "health": health,
                "agents": agents,
                "tasks": task_stats,
                "websocket_connections": len(self.websocket_connections),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
          self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}


# Utility functions for web interface

def create_web_interface(config: Dict[str, Any]) -> WebInterface:
    """Create and configure a web interface instance."""
    return WebInterface(config)


def get_mime_type(filename: str) -> str:
    """Get MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'


def is_safe_filename(filename: str) -> bool:
    """Check if filename is safe for storage."""
    # Remove path separators and other potentially dangerous characters
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_')
    return all(c in safe_chars for c in filename) and not filename.startswith('.')


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[^\w\s\-_\.]', '', filename)
    sanitized = re.sub(r'[-\s]+', '-', sanitized)
    return sanitized.strip('.-')


class FileValidator:
    """File validation utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_size = config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.allowed_types = set(config.get('allowed_file_types', []))
        self.blocked_extensions = set(config.get('blocked_extensions', [
            '.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.jar'
        ]))
    
    def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file."""
        errors = []
        
        # Check file size
        if file.size > self.max_size:
            errors.append(f"File too large. Maximum size: {format_bytes(self.max_size)}")
        
        # Check MIME type
        if self.allowed_types and file.content_type not in self.allowed_types:
            errors.append(f"File type not allowed: {file.content_type}")
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension in self.blocked_extensions:
            errors.append(f"File extension not allowed: {file_extension}")
        
        # Check filename safety
        if not is_safe_filename(file.filename):
            errors.append("Filename contains unsafe characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": []
        }


class WebInterfaceConfig:
    """Configuration for web interface."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.host = config_dict.get('host', '0.0.0.0')
        self.port = config_dict.get('port', 8000)
        self.workers = config_dict.get('workers', 1)
        self.reload = config_dict.get('reload', False)
        
        # Security settings
        self.enable_auth = config_dict.get('enable_auth', True)
        self.secret_key = config_dict.get('secret_key', 'ultra-ai-secret')
        self.cors_origins = config_dict.get('cors_origins', ["*"])
        self.allowed_hosts = config_dict.get('allowed_hosts', ["*"])
        
        # File upload settings
        self.upload_dir = config_dict.get('upload_dir', './data/uploads')
        self.max_file_size = config_dict.get('max_file_size', 100 * 1024 * 1024)
        self.allowed_file_types = config_dict.get('allowed_file_types', [])
        self.blocked_extensions = config_dict.get('blocked_extensions', [])
        
        # Rate limiting
        self.rate_limit_enabled = config_dict.get('rate_limit_enabled', True)
        self.rate_limit_requests = config_dict.get('rate_limit_requests', 100)
        self.rate_limit_window = config_dict.get('rate_limit_window', 3600)
        
        # WebSocket settings
        self.websocket_enabled = config_dict.get('websocket_enabled', True)
        self.websocket_ping_interval = config_dict.get('websocket_ping_interval', 30)
        self.websocket_timeout = config_dict.get('websocket_timeout', 300)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "server": {
                "host": self.host,
                "port": self.port,
                "workers": self.workers,
                "reload": self.reload
            },
            "security": {
                "enable_auth": self.enable_auth,
                "secret_key": self.secret_key,
                "cors_origins": self.cors_origins,
                "allowed_hosts": self.allowed_hosts
            },
            "files": {
                "upload_dir": self.upload_dir,
                "max_file_size": self.max_file_size,
                "allowed_file_types": self.allowed_file_types,
                "blocked_extensions": self.blocked_extensions
            },
            "rate_limiting": {
                "enabled": self.rate_limit_enabled,
                "requests": self.rate_limit_requests,
                "window": self.rate_limit_window
            },
            "websocket": {
                "enabled": self.websocket_enabled,
                "ping_interval": self.websocket_ping_interval,
                "timeout": self.websocket_timeout
            }
        }


class APIResponseFormatter:
    """Utility class for formatting API responses."""
    
    @staticmethod
    def success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """Format successful API response."""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if data is not None:
            response["data"] = data
        
        return response
    
    @staticmethod
    def error(message: str, code: str = "UNKNOWN_ERROR", details: Any = None) -> Dict[str, Any]:
        """Format error API response."""
        response = {
            "success": False,
            "error": {
                "message": message,
                "code": code,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if details is not None:
            response["error"]["details"] = details
        
        return response
    
    @staticmethod
    def paginated(
        items: List[Any],
        total: int,
        page: int,
        page_size: int,
        has_next: bool = None,
        has_prev: bool = None
    ) -> Dict[str, Any]:
        """Format paginated API response."""
        if has_next is None:
            has_next = (page * page_size) < total
        
        if has_prev is None:
            has_prev = page > 1
        
        return {
            "success": True,
            "data": {
                "items": items,
                "pagination": {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total + page_size - 1) // page_size,
                    "has_next": has_next,
                    "has_prev": has_prev
                }
            },
            "timestamp": datetime.now().isoformat()
        }


class WebInterfaceMiddleware:
    """Custom middleware for web interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = Logger(__name__)
        self.request_count = {}
        self.rate_limit_window = config.get('rate_limit_window', 3600)
        self.rate_limit_requests = config.get('rate_limit_requests', 100)
    
    async def rate_limit_middleware(self, request: Request, call_next):
        """Rate limiting middleware."""
        if not self.config.get('rate_limit_enabled', True):
            return await call_next(request)
        
        client_ip = request.client.host
        current_time = datetime.now()
        
        # Clean old entries
        cutoff_time = current_time - timedelta(seconds=self.rate_limit_window)
        self.request_count = {
            ip: timestamps for ip, timestamps in self.request_count.items()
            if any(ts > cutoff_time for ts in timestamps)
        }
        
        # Update current client's request count
        if client_ip not in self.request_count:
            self.request_count[client_ip] = []
        
        # Remove old timestamps for this client
        self.request_count[client_ip] = [
            ts for ts in self.request_count[client_ip] if ts > cutoff_time
        ]
        
        # Check rate limit
        if len(self.request_count[client_ip]) >= self.rate_limit_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": self.rate_limit_window
                }
            )
        
        # Add current request timestamp
        self.request_count[client_ip].append(current_time)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.rate_limit_requests - len(self.request_count[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int((current_time + timedelta(seconds=self.rate_limit_window)).timestamp()))
        
        return response
    
    async def logging_middleware(self, request: Request, call_next):
        """Request logging middleware."""
        start_time = datetime.now()
        
        # Log request
        self.logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
        
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log response
        self.logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} "
            f"({duration:.3f}s)"
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response


class StreamingResponseGenerator:
    """Generator for streaming responses."""
    
    def __init__(self, task_coordinator: TaskCoordinator):
        self.task_coordinator = task_coordinator
        self.logger = Logger(__name__)
    
    async def stream_task_results(self, task_id: str):
        """Stream task results as they become available."""
        try:
            while True:
                task = await self.task_coordinator.get_task(task_id)
                if not task:
                    yield json.dumps({"error": "Task not found"}) + "\n"
                    break
                
                # Yield current task status
                yield json.dumps({
                    "type": "task_update",
                    "task_id": task_id,
                    "status": task.status,
                    "progress": getattr(task, 'progress', None),
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
                
                # Check if task is complete
                if task.status in ['completed', 'failed', 'cancelled']:
                    yield json.dumps({
                        "type": "task_complete",
                        "task_id": task_id,
                        "result": task.result,
                        "error": task.error,
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"
                    break
                
                # Wait before next update
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error streaming task results: {e}")
            yield json.dumps({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }) + "\n"


# Template creation utilities

async def create_default_templates():
    """Create default HTML templates for web interface."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    # Basic index template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-robot"></i> Ultra AI System
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/dashboard">Dashboard</a>
                <a class="nav-link" href="/tasks">Tasks</a>
                <a class="nav-link" href="/docs">API Docs</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-8 mx-auto text-center">
                <h1 class="display-4">Welcome to Ultra AI</h1>
                <p class="lead">Advanced AI system for intelligent automation and assistance</p>
                <div class="mt-4">
                    <a href="/dashboard" class="btn btn-primary btn-lg me-3">
                        <i class="fas fa-tachometer-alt"></i> Dashboard
                    </a>
                    <a href="/docs" class="btn btn-outline-primary btn-lg">
                        <i class="fas fa-book"></i> API Documentation
                    </a>
                </div>
            </div>
        </div>
        
        <div class="row mt-5">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <i class="fas fa-robot fa-3x text-primary mb-3"></i>
                        <h5 class="card-title">AI Agents</h5>
                        <p class="card-text">Specialized agents for different tasks and domains</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <i class="fas fa-tasks fa-3x text-success mb-3"></i>
                        <h5 class="card-title">Task Management</h5>
                        <p class="card-text">Create, monitor, and manage AI tasks efficiently</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <i class="fas fa-chart-line fa-3x text-info mb-3"></i>
                        <h5 class="card-title">Analytics</h5>
                        <p class="card-text">Monitor system performance and task metrics</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    
    index_file = template_dir / "index.html"
    if not index_file.exists():
        async with aiofiles.open(index_file, 'w') as f:
            await f.write(index_html)


# Export main components
__all__ = [
    'WebInterface',
    'TaskCreateRequest',
    'TaskResponse',
    'UserLoginRequest',
    'UserCreateRequest',
    'ConfigUpdateRequest',
    'FileUploadResponse',
    'WebSocketConnection',
    'WebInterfaceConfig',
    'FileValidator',
    'APIResponseFormatter',
    'WebInterfaceMiddleware',
    'StreamingResponseGenerator',
    'create_web_interface',
    'create_default_templates'
]
