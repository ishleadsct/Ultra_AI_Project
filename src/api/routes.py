"""
Ultra AI Project - API Routes

FastAPI application with comprehensive routes for all system functionality,
including agent management, task execution, file handling, and system monitoring.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, Query, Path as PathParam
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from ..utils.logger import get_logger
from ..utils.helpers import load_config, generate_task_id, format_bytes
from .middleware import setup_middleware
from .auth import authenticate_request, get_current_user
from .models import (
    APIResponse, PaginatedResponse, ErrorDetail,
    create_success_response, create_error_response, create_paginated_response,
    API_PREFIX, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
)

logger = get_logger(__name__)

# Pydantic models for request/response
class TaskRequest(BaseModel):
    type: str = Field(..., description="Task type")
    agent: str = Field(..., description="Target agent name")
    data: Dict[str, Any] = Field(default_factory=dict, description="Task data")
    priority: int = Field(default=2, ge=1, le=5, description="Task priority (1-5)")
    timeout: float = Field(default=300.0, gt=0, description="Task timeout in seconds")
    callback_url: Optional[str] = Field(None, description="Callback URL for task completion")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[float] = None

class AgentInfo(BaseModel):
    name: str
    type: str
    status: str
    capabilities: List[str]
    current_tasks: int
    total_tasks_processed: int
    last_activity: Optional[datetime] = None

class SystemStatus(BaseModel):
    status: str
    uptime: float
    version: str
    active_agents: int
    pending_tasks: int
    total_requests: int
    memory_usage: Dict[str, Any]
    last_health_check: datetime

class ConversationRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    agent: Optional[str] = Field(default="analysis_agent")
    context: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    message_id: str
    response: str
    agent: str
    timestamp: datetime
    processing_time: float

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    mime_type: str
    checksum: str
    upload_time: datetime

class MemoryQuery(BaseModel):
    query: Optional[str] = None
    memory_type: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=50, le=1000)

# Global system manager reference (will be injected)
system_manager = None

# FastAPI app instance
app = FastAPI(
    title="Ultra AI API",
    description="Advanced AI system API with multi-agent capabilities",
    version="1.0.0",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json"
)

# Setup middleware
setup_middleware(app)

# Security scheme
security = HTTPBearer()

# Dependency to get current user
async def get_current_user_dep(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user."""
    return await get_current_user(credentials.credentials)

# Health and status endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get(f"{API_PREFIX}/status", response_model=SystemStatus, tags=["System"])
async def get_system_status():
    """Get comprehensive system status."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        status = await system_manager.get_status()
        
        return SystemStatus(
            status=status.status,
            uptime=status.uptime,
            version="1.0.0",
            active_agents=status.active_agents,
            pending_tasks=status.pending_tasks,
            total_requests=getattr(system_manager, '_total_requests', 0),
            memory_usage=status.memory_usage,
            last_health_check=status.last_health_check
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@app.get(f"{API_PREFIX}/metrics", tags=["System"])
async def get_system_metrics(hours: int = Query(default=1, ge=1, le=24)):
    """Get system metrics for specified time period."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        metrics = await system_manager.get_metrics(hours=hours)
        return create_success_response("Metrics retrieved successfully", data=metrics)
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Agent management endpoints
@app.get(f"{API_PREFIX}/agents", response_model=List[AgentInfo], tags=["Agents"])
async def list_agents(user=Depends(get_current_user_dep)):
    """List all available agents."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        agent_names = await system_manager.list_agents()
        agents = []
        
        for name in agent_names:
            agent = await system_manager.get_agent(name)
            if agent:
                agent_info = AgentInfo(
                    name=name,
                    type=getattr(agent, 'agent_type', 'unknown'),
                    status=getattr(agent, 'status', 'unknown'),
                    capabilities=getattr(agent, 'capabilities', []),
                    current_tasks=getattr(agent, 'current_tasks', 0),
                    total_tasks_processed=getattr(agent, 'total_tasks_processed', 0),
                    last_activity=getattr(agent, 'last_activity', None)
                )
                agents.append(agent_info)
        
        return agents
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

@app.get(f"{API_PREFIX}/agents/{{agent_name}}", response_model=AgentInfo, tags=["Agents"])
async def get_agent(agent_name: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Get specific agent information."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        agent = await system_manager.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        return AgentInfo(
            name=agent_name,
            type=getattr(agent, 'agent_type', 'unknown'),
            status=getattr(agent, 'status', 'unknown'),
            capabilities=getattr(agent, 'capabilities', []),
            current_tasks=getattr(agent, 'current_tasks', 0),
            total_tasks_processed=getattr(agent, 'total_tasks_processed', 0),
            last_activity=getattr(agent, 'last_activity', None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent information")

@app.post(f"{API_PREFIX}/agents/{{agent_name}}/restart", tags=["Agents"])
async def restart_agent(agent_name: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Restart a specific agent."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        success = await system_manager.restart_agent(agent_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        return create_success_response(f"Agent '{agent_name}' restarted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to restart agent")

# Task management endpoints
@app.post(f"{API_PREFIX}/tasks", response_model=TaskResponse, tags=["Tasks"])
async def create_task(task_request: TaskRequest, user=Depends(get_current_user_dep)):
    """Create and execute a new task."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        # Add user context to task data
        task_data = task_request.dict()
        task_data["user_id"] = user.id if user else None
        
        result = await system_manager.execute_task(task_data)
        
        return TaskResponse(
            task_id=result.get("task_id", "unknown"),
            status=result.get("status", "unknown"),
            created_at=datetime.now(),
            result=result.get("result"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(status_code=500, detail="Failed to create task")

@app.get(f"{API_PREFIX}/tasks/{{task_id}}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(task_id: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Get task status and result."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        task_coordinator = system_manager.task_coordinator
        if not task_coordinator:
            raise HTTPException(status_code=503, detail="Task coordinator not available")
        
        task_status = await task_coordinator.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        
        return TaskResponse(
            task_id=task_id,
            status=task_status.get("status", "unknown"),
            created_at=datetime.fromisoformat(task_status.get("created_at", datetime.now().isoformat())),
            started_at=datetime.fromisoformat(task_status["started_at"]) if task_status.get("started_at") else None,
            result=task_status.get("result"),
            error=task_status.get("error"),
            progress=task_status.get("progress")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task")

@app.delete(f"{API_PREFIX}/tasks/{{task_id}}", tags=["Tasks"])
async def cancel_task(task_id: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Cancel a pending or running task."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        task_coordinator = system_manager.task_coordinator
        if not task_coordinator:
            raise HTTPException(status_code=503, detail="Task coordinator not available")
        
        success = await task_coordinator.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found or cannot be cancelled")
        
        return create_success_response(f"Task '{task_id}' cancelled successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel task")

# Conversation endpoints
@app.post(f"{API_PREFIX}/chat", response_model=ConversationResponse, tags=["Conversation"])
async def chat(request: ConversationRequest, user=Depends(get_current_user_dep)):
    """Send a message and get AI response."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        # Create conversation task
        task_data = {
            "type": "chat",
            "agent": request.agent,
            "data": {
                "message": request.message,
                "conversation_id": request.conversation_id,
                "context": request.context,
                "user_id": user.id if user else None
            }
        }
        
        start_time = datetime.now()
        result = await system_manager.execute_task(task_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response_data = result.get("result", {})
        conversation_id = response_data.get("conversation_id", request.conversation_id or str(uuid.uuid4()))
        
        return ConversationResponse(
            conversation_id=conversation_id,
            message_id=str(uuid.uuid4()),
            response=response_data.get("response", "I'm sorry, I couldn't process your request."),
            agent=request.agent,
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

@app.get(f"{API_PREFIX}/conversations/{{conversation_id}}", tags=["Conversation"])
async def get_conversation(conversation_id: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Get conversation history."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        memory_manager = system_manager.memory_manager
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")
        
        conversation = await memory_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
        
        return create_success_response("Conversation retrieved successfully", data=conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation")

# File management endpoints
@app.post(f"{API_PREFIX}/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    user=Depends(get_current_user_dep)
):
    """Upload a file to the system."""
    try:
        # Read file content
        content = await file.read()
        
        # Parse tags
        file_tags = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Get file handler from system manager
        if not system_manager or not hasattr(system_manager, 'file_handler'):
            raise HTTPException(status_code=503, detail="File handler not available")
        
        file_handler = system_manager.file_handler
        file_id = await file_handler.upload_file(
            content,
            file.filename or "unknown",
            user_id=user.id if user else None,
            tags=file_tags
        )
        
        # Get file metadata
        metadata = await file_handler.get_file(file_id)
        
        return FileUploadResponse(
            file_id=file_id,
            filename=metadata.filename,
            size=metadata.file_size,
            mime_type=metadata.mime_type,
            checksum=metadata.checksum,
            upload_time=metadata.created_at
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.get(f"{API_PREFIX}/files/{{file_id}}", tags=["Files"])
async def get_file_info(file_id: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Get file metadata."""
    try:
        if not system_manager or not hasattr(system_manager, 'file_handler'):
            raise HTTPException(status_code=503, detail="File handler not available")
        
        file_handler = system_manager.file_handler
        metadata = await file_handler.get_file(file_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
        
        return create_success_response("File information retrieved", data=metadata.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file information")

@app.get(f"{API_PREFIX}/files/{{file_id}}/download", tags=["Files"])
async def download_file(file_id: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Download file content."""
    try:
        if not system_manager or not hasattr(system_manager, 'file_handler'):
            raise HTTPException(status_code=503, detail="File handler not available")
        
        file_handler = system_manager.file_handler
        metadata = await file_handler.get_file(file_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
        
        file_path = Path(metadata.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File content not found")
        
        return FileResponse(
            path=str(file_path),
            filename=metadata.original_filename,
            media_type=metadata.mime_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed {file_id}: {e}")
        raise HTTPException(status_code=500, detail="File download failed")

@app.delete(f"{API_PREFIX}/files/{{file_id}}", tags=["Files"])
async def delete_file(file_id: str = PathParam(...), user=Depends(get_current_user_dep)):
    """Delete a file."""
    try:
        if not system_manager or not hasattr(system_manager, 'file_handler'):
            raise HTTPException(status_code=503, detail="File handler not available")
        
        file_handler = system_manager.file_handler
        success = await file_handler.delete_file(file_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
        
        return create_success_response(f"File '{file_id}' deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File deletion failed {file_id}: {e}")
        raise HTTPException(status_code=500, detail="File deletion failed")

# Memory and search endpoints
@app.post(f"{API_PREFIX}/memory/search", tags=["Memory"])
async def search_memory(query: MemoryQuery, user=Depends(get_current_user_dep)):
    """Search system memory."""
    try:
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        memory_manager = system_manager.memory_manager
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")
        
        results = await memory_manager.search_memory(
            query=query.query,
            memory_type=query.memory_type,
            tags=query.tags,
            user_id=user.id if user else None,
            limit=query.limit
        )
        
        return create_success_response("Memory search completed", data=[item.to_dict() for item in results])
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail="Memory search failed")

# System management endpoints
@app.post(f"{API_PREFIX}/system/restart", tags=["System"])
async def restart_system(user=Depends(get_current_user_dep)):
    """Restart the entire system."""
    try:
        # Check if user has admin privileges
        if not user or getattr(user, 'role', None) not in ['admin', 'system']:
            raise HTTPException(status_code=403, detail="Insufficient privileges")
        
        if not system_manager:
            raise HTTPException(status_code=503, detail="System manager not available")
        
        # Schedule system restart
        async def restart_task():
            await asyncio.sleep(2)  # Give time for response
            await system_manager.shutdown()
            # System restart would be handled by process manager
        
        asyncio.create_task(restart_task())
        
        return create_success_response("System restart initiated")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System restart failed: {e}")
        raise HTTPException(status_code=500, detail="System restart failed")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            message=exc.detail,
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            message="Internal server error",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("Ultra AI API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Ultra AI API shutting down...")
    if system_manager:
        await system_manager.shutdown()

# Function to inject system manager
def set_system_manager(manager):
    """Set the system manager instance."""
    global system_manager
    system_manager = manager

# Function to start the server
def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    uvicorn.run(
        "src.api.routes:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

# Create router for external mounting
from fastapi import APIRouter
router = APIRouter(prefix=API_PREFIX)

# Add all routes to router for external use
for route in app.routes:
    if hasattr(route, 'path') and route.path.startswith(API_PREFIX):
        router.routes.append(route)
