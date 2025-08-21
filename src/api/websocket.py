"""
Ultra AI Project - WebSocket Manager

Real-time WebSocket communication system for live updates, streaming responses,
and bidirectional communication between clients and AI agents.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from pydantic import BaseModel, ValidationError
import jwt

from ..utils.logger import get_logger
from .auth import get_current_user, get_auth_manager

logger = get_logger(__name__)

class MessageType(Enum):
    """WebSocket message types."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    CHAT_MESSAGE = "chat_message"
    CHAT_RESPONSE = "chat_response"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    BROADCAST = "broadcast"

@dataclass
class WebSocketConnection:
    """WebSocket connection information."""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    username: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_authenticated: bool = False

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str
    data: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    target: Optional[str] = None  # For targeted messages

class WebSocketResponse(BaseModel):
    """WebSocket response format."""
    type: str
    data: Optional[Any] = None
    message_id: Optional[str] = None
    timestamp: datetime
    status: str = "success"
    error: Optional[str] = None

class ConnectionManager:
    """Manages WebSocket connections and message routing."""
    
    def __init__(self):
        # Active connections
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        
        # Subscriptions
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> connection_ids
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "authentication_attempts": 0,
            "authentication_successes": 0
        }
        
        self._register_default_handlers()
        logger.info("ConnectionManager initialized")
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.message_handlers.update({
            MessageType.PING.value: self._handle_ping,
            MessageType.AUTH.value: self._handle_auth,
            MessageType.CHAT_MESSAGE.value: self._handle_chat_message,
            MessageType.TASK_START.value: self._handle_task_start,
        })
    
    async def connect(self, websocket: WebSocket, connection_id: Optional[str] = None) -> str:
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        connection_id = connection_id or str(uuid.uuid4())
        
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket
        )
        
        self.connections[connection_id] = connection
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)
        
        # Send welcome message
        await self._send_to_connection(connection_id, WebSocketResponse(
            type=MessageType.CONNECT.value,
            data={"connection_id": connection_id},
            timestamp=datetime.now()
        ))
        
        logger.info(f"WebSocket connection established: {connection_id}")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Remove from user connections
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Remove from topic subscriptions
        for topic, subscribers in self.topic_subscribers.items():
            subscribers.discard(connection_id)
        
        # Remove connection
        del self.connections[connection_id]
        self.stats["active_connections"] = len(self.connections)
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def send_message(self, connection_id: str, message: WebSocketResponse) -> bool:
        """Send message to specific connection."""
        return await self._send_to_connection(connection_id, message)
    
    async def send_to_user(self, user_id: str, message: WebSocketResponse) -> int:
        """Send message to all connections of a user."""
        connection_ids = self.user_connections.get(user_id, set())
        sent_count = 0
        
        for connection_id in connection_ids.copy():  # Copy to avoid modification during iteration
            if await self._send_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast(self, message: WebSocketResponse, 
                       exclude_connections: Optional[Set[str]] = None) -> int:
        """Broadcast message to all connections."""
        exclude_connections = exclude_connections or set()
        sent_count = 0
        
        for connection_id in list(self.connections.keys()):
            if connection_id not in exclude_connections:
                if await self._send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_topic(self, topic: str, message: WebSocketResponse) -> int:
        """Broadcast message to all subscribers of a topic."""
        subscribers = self.topic_subscribers.get(topic, set())
        sent_count = 0
        
        for connection_id in subscribers.copy():
            if await self._send_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to a topic."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.add(topic)
        
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to topic: {topic}")
        return True
    
    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe connection from a topic."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.discard(topic)
        
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(connection_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        
        logger.debug(f"Connection {connection_id} unsubscribed from topic: {topic}")
        return True
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            try:
                message_data = json.loads(message)
                ws_message = WebSocketMessage(**message_data)
            except (json.JSONDecodeError, ValidationError) as e:
                await self._send_error(connection_id, "Invalid message format", str(e))
                return
            
            # Update connection activity
            connection = self.connections.get(connection_id)
            if connection:
                connection.last_activity = datetime.now()
            
            self.stats["messages_received"] += 1
            
            # Handle message based on type
            handler = self.message_handlers.get(ws_message.type)
            if handler:
                await handler(connection_id, ws_message)
            else:
                await self._send_error(connection_id, "Unknown message type", ws_message.type)
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(connection_id, "Message processing failed", str(e))
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketResponse) -> bool:
        """Send message to a specific connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        try:
            message_dict = message.dict()
            message_dict["timestamp"] = message.timestamp.isoformat()
            
            await connection.websocket.send_text(json.dumps(message_dict))
            self.stats["messages_sent"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            # Remove broken connection
            await self.disconnect(connection_id)
            return False
    
    async def _send_error(self, connection_id: str, error_message: str, details: str = ""):
        """Send error message to connection."""
        await self._send_to_connection(connection_id, WebSocketResponse(
            type=MessageType.ERROR.value,
            status="error",
            error=error_message,
            data={"details": details} if details else None,
            timestamp=datetime.now()
        ))
    
    async def _handle_ping(self, connection_id: str, message: WebSocketMessage):
        """Handle ping message."""
        await self._send_to_connection(connection_id, WebSocketResponse(
            type=MessageType.PONG.value,
            message_id=message.message_id,
            timestamp=datetime.now()
        ))
    
    async def _handle_auth(self, connection_id: str, message: WebSocketMessage):
        """Handle authentication message."""
        self.stats["authentication_attempts"] += 1
        
        try:
            data = message.data or {}
            token = data.get("token")
            
            if not token:
                await self._send_to_connection(connection_id, WebSocketResponse(
                    type=MessageType.AUTH_FAILED.value,
                    error="Token required",
                    timestamp=datetime.now()
                ))
                return
            
            # Validate token
            auth_manager = get_auth_manager()
            if not auth_manager or not auth_manager.jwt_auth:
                await self._send_to_connection(connection_id, WebSocketResponse(
                    type=MessageType.AUTH_FAILED.value,
                    error="Authentication not available",
                    timestamp=datetime.now()
                ))
                return
            
            token_data = await auth_manager.jwt_auth.validate_token(token)
            if not token_data:
                await self._send_to_connection(connection_id, WebSocketResponse(
                    type=MessageType.AUTH_FAILED.value,
                    error="Invalid token",
                    timestamp=datetime.now()
                ))
                return
            
            # Update connection with user info
            connection = self.connections[connection_id]
            connection.user_id = token_data.user_id
            connection.username = token_data.username
            connection.is_authenticated = True
            
            # Add to user connections
            if token_data.user_id not in self.user_connections:
                self.user_connections[token_data.user_id] = set()
            self.user_connections[token_data.user_id].add(connection_id)
            
            self.stats["authentication_successes"] += 1
            
            await self._send_to_connection(connection_id, WebSocketResponse(
                type=MessageType.AUTH_SUCCESS.value,
                data={
                    "user_id": token_data.user_id,
                    "username": token_data.username,
                    "role": token_data.role
                },
                timestamp=datetime.now()
            ))
            
            logger.info(f"WebSocket authentication successful: {connection_id} -> {token_data.username}")
            
        except Exception as e:
            logger.error(f"Authentication failed for {connection_id}: {e}")
            await self._send_to_connection(connection_id, WebSocketResponse(
                type=MessageType.AUTH_FAILED.value,
                error="Authentication failed",
                timestamp=datetime.now()
            ))
    
    async def _handle_chat_message(self, connection_id: str, message: WebSocketMessage):
        """Handle chat message."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_authenticated:
            await self._send_error(connection_id, "Authentication required")
            return
        
        try:
            data = message.data or {}
            chat_message = data.get("message")
            agent = data.get("agent", "analysis_agent")
            conversation_id = data.get("conversation_id")
            
            if not chat_message:
                await self._send_error(connection_id, "Message content required")
                return
            
            # Process chat message (integrate with system manager)
            # This would typically create a task and stream the response
            
            # For now, send acknowledgment
            await self._send_to_connection(connection_id, WebSocketResponse(
                type=MessageType.CHAT_RESPONSE.value,
                data={
                    "response": f"Received message: {chat_message}",
                    "agent": agent,
                    "conversation_id": conversation_id or str(uuid.uuid4())
                },
                message_id=message.message_id,
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            logger.error(f"Chat message handling failed: {e}")
            await self._send_error(connection_id, "Failed to process chat message")
    
    async def _handle_task_start(self, connection_id: str, message: WebSocketMessage):
        """Handle task start message."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_authenticated:
            await self._send_error(connection_id, "Authentication required")
            return
        
        try:
            data = message.data or {}
            task_type = data.get("type")
            agent = data.get("agent")
            task_data = data.get("data", {})
            
            if not task_type or not agent:
                await self._send_error(connection_id, "Task type and agent required")
                return
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Send task started confirmation
            await self._send_to_connection(connection_id, WebSocketResponse(
                type=MessageType.TASK_START.value,
                data={
                    "task_id": task_id,
                    "type": task_type,
                    "agent": agent,
                    "status": "started"
                },
                message_id=message.message_id,
                timestamp=datetime.now()
            ))
            
            # Subscribe to task updates
            await self.subscribe(connection_id, f"task_{task_id}")
            
        except Exception as e:
            logger.error(f"Task start handling failed: {e}")
            await self._send_error(connection_id, "Failed to start task")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register custom message handler."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information."""
        connection = self.connections.get(connection_id)
        if not connection:
            return None
        
        return {
            "connection_id": connection.connection_id,
            "user_id": connection.user_id,
            "username": connection.username,
            "connected_at": connection.connected_at.isoformat(),
            "last_activity": connection.last_activity.isoformat(),
            "is_authenticated": connection.is_authenticated,
            "subscriptions": list(connection.subscriptions),
            "metadata": connection.metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        authenticated_connections = sum(
            1 for conn in self.connections.values() 
            if conn.is_authenticated
        )
        
        return {
            **self.stats,
            "authenticated_connections": authenticated_connections,
            "topics": len(self.topic_subscribers),
            "unique_users": len(self.user_connections)
        }

class WebSocketManager:
    """Main WebSocket manager with system integration."""
    
    def __init__(self, system_manager=None):
        self.system_manager = system_manager
        self.connection_manager = ConnectionManager()
        
        # Task streaming
        self.active_streams: Dict[str, asyncio.Task] = {}
        
        # Register custom handlers
        self._register_system_handlers()
        
        logger.info("WebSocketManager initialized")
    
    def _register_system_handlers(self):
        """Register system-specific message handlers."""
        if self.system_manager:
            # Register handlers that interact with system manager
            pass
    
    async def handle_connection(self, websocket: WebSocket, 
                               connection_id: Optional[str] = None):
        """Handle WebSocket connection lifecycle."""
        connection_id = await self.connection_manager.connect(websocket, connection_id)
        
        try:
            while True:
                # Receive message
                message = await websocket.receive_text()
                await self.connection_manager.handle_message(connection_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self.connection_manager.disconnect(connection_id)
    
    async def stream_task_progress(self, task_id: str, progress_data: Dict[str, Any]):
        """Stream task progress to subscribers."""
        await self.connection_manager.broadcast_to_topic(
            f"task_{task_id}",
            WebSocketResponse(
                type=MessageType.TASK_PROGRESS.value,
                data={
                    "task_id": task_id,
                    "progress": progress_data
                },
                timestamp=datetime.now()
            )
        )
    
    async def notify_task_completion(self, task_id: str, result: Dict[str, Any]):
        """Notify task completion to subscribers."""
        await self.connection_manager.broadcast_to_topic(
            f"task_{task_id}",
            WebSocketResponse(
                type=MessageType.TASK_COMPLETE.value,
                data={
                    "task_id": task_id,
                    "result": result
                },
                timestamp=datetime.now()
            )
        )
    
    async def notify_system_status(self, status: Dict[str, Any]):
        """Broadcast system status to all connections."""
        await self.connection_manager.broadcast(
            WebSocketResponse(
                type=MessageType.SYSTEM_STATUS.value,
                data=status,
                timestamp=datetime.now()
            )
        )
    
    async def send_to_user(self, user_id: str, message_type: str, data: Any) -> int:
        """Send message to specific user."""
        return await self.connection_manager.send_to_user(
            user_id,
            WebSocketResponse(
                type=message_type,
                data=data,
                timestamp=datetime.now()
            )
        )

# Global WebSocket manager instance
_websocket_manager: Optional[WebSocketManager] = None

def initialize_websocket_manager(system_manager=None):
    """Initialize WebSocket manager."""
    global _websocket_manager
    _websocket_manager = WebSocketManager(system_manager)
    logger.info("WebSocket manager initialized")

def get_websocket_manager() -> Optional[WebSocketManager]:
    """Get WebSocket manager instance."""
    return _websocket_manager

# FastAPI WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket, 
                            connection_id: Optional[str] = Query(None)):
    """Main WebSocket endpoint for FastAPI."""
    if not _websocket_manager:
        await websocket.close(code=1003, reason="WebSocket manager not available")
        return
    
    await _websocket_manager.handle_connection(websocket, connection_id)

# Utility functions
async def broadcast_message(message_type: str, data: Any) -> int:
    """Broadcast message to all connected clients."""
    if not _websocket_manager:
        return 0
    
    return await _websocket_manager.connection_manager.broadcast(
        WebSocketResponse(
            type=message_type,
            data=data,
            timestamp=datetime.now()
        )
    )

async def send_to_topic(topic: str, message_type: str, data: Any) -> int:
    """Send message to topic subscribers."""
    if not _websocket_manager:
        return 0
    
    return await _websocket_manager.connection_manager.broadcast_to_topic(
        topic,
        WebSocketResponse(
            type=message_type,
            data=data,
            timestamp=datetime.now()
        )
    )

async def get_connection_stats() -> Dict[str, Any]:
    """Get WebSocket connection statistics."""
    if not _websocket_manager:
        return {}
    
    return _websocket_manager.connection_manager.get_stats()
