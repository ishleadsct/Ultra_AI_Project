"""
Ultra AI Project - API Authentication

Authentication and authorization system for API endpoints,
supporting multiple authentication methods and role-based access control.
"""

import jwt
import time
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel

from ..utils.logger import get_logger
from ..core.security import SecurityManager, User, UserRole

logger = get_logger(__name__)

# Authentication schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class AuthConfig:
    """Authentication configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.secret_key = config.get("secret_key", secrets.token_urlsafe(32))
        self.algorithm = config.get("algorithm", "HS256")
        self.access_token_expire_minutes = config.get("access_token_expire_minutes", 30)
        self.refresh_token_expire_days = config.get("refresh_token_expire_days", 7)
        self.api_key_expire_days = config.get("api_key_expire_days", 365)
        
        # Enable/disable authentication methods
        self.enable_jwt = config.get("enable_jwt", True)
        self.enable_api_key = config.get("enable_api_key", True)
        self.enable_session = config.get("enable_session", True)
        
        # Security settings
        self.require_https = config.get("require_https", False)
        self.allow_multiple_sessions = config.get("allow_multiple_sessions", True)

@dataclass
class TokenData:
    """Token payload data."""
    user_id: str
    username: str
    role: str
    session_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    permissions: Optional[List[str]] = None

class AuthRequest(BaseModel):
    """Authentication request model."""
    username: str
    password: str
    remember_me: bool = False

class AuthResponse(BaseModel):
    """Authentication response model."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str
    description: Optional[str] = None
    expires_in_days: Optional[int] = None
    permissions: Optional[List[str]] = None

class APIKeyResponse(BaseModel):
    """API key response model."""
    key_id: str
    api_key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    permissions: List[str]

class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(self, security_manager: SecurityManager, config: AuthConfig):
        self.security_manager = security_manager
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from storage."""
        # In a real implementation, this would load from database
        # For now, we'll use in-memory storage
        pass
    
    async def create_api_key(self, user: User, request: APIKeyRequest) -> APIKeyResponse:
        """Create a new API key for user."""
        try:
            # Generate API key
            key_id = f"ak_{secrets.token_urlsafe(8)}"
            api_key = f"ultra_ai_{secrets.token_urlsafe(32)}"
            
            # Set expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.now() + timedelta(days=request.expires_in_days)
            elif self.config.api_key_expire_days:
                expires_at = datetime.now() + timedelta(days=self.config.api_key_expire_days)
            
            # Store API key
            self.api_keys[api_key] = {
                "key_id": key_id,
                "user_id": user.id,
                "name": request.name,
                "description": request.description,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "permissions": request.permissions or [],
                "last_used": None,
                "is_active": True
            }
            
            logger.info(f"API key created for user {user.username}: {key_id}")
            
            return APIKeyResponse(
                key_id=key_id,
                api_key=api_key,
                name=request.name,
                created_at=datetime.now(),
                expires_at=expires_at,
                permissions=request.permissions or []
            )
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise HTTPException(status_code=500, detail="Failed to create API key")
    
    async def validate_api_key(self, api_key: str) -> Optional[User]:
        """Validate API key and return associated user."""
        try:
            if not api_key or api_key not in self.api_keys:
                return None
            
            key_data = self.api_keys[api_key]
            
            # Check if key is active
            if not key_data.get("is_active", True):
                return None
            
            # Check expiration
            expires_at = key_data.get("expires_at")
            if expires_at and datetime.now() > expires_at:
                logger.warning(f"Expired API key used: {key_data['key_id']}")
                return None
            
            # Get user
            user_id = key_data["user_id"]
            user = self.security_manager.users.get(user_id)
            
            if not user or not user.is_active:
                return None
            
            # Update last used timestamp
            key_data["last_used"] = datetime.now()
            
            logger.debug(f"API key validated for user: {user.username}")
            return user
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None
    
    async def revoke_api_key(self, user: User, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            # Find the API key
            for api_key, key_data in self.api_keys.items():
                if key_data["key_id"] == key_id and key_data["user_id"] == user.id:
                    key_data["is_active"] = False
                    key_data["revoked_at"] = datetime.now()
                    logger.info(f"API key revoked: {key_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False
    
    async def list_api_keys(self, user: User) -> List[Dict[str, Any]]:
        """List user's API keys."""
        try:
            keys = []
            for api_key, key_data in self.api_keys.items():
                if key_data["user_id"] == user.id:
                    keys.append({
                        "key_id": key_data["key_id"],
                        "name": key_data["name"],
                        "description": key_data["description"],
                        "created_at": key_data["created_at"],
                        "expires_at": key_data["expires_at"],
                        "last_used": key_data["last_used"],
                        "is_active": key_data["is_active"],
                        "permissions": key_data["permissions"]
                    })
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            return []

class JWTAuth:
    """JWT token authentication handler."""
    
    def __init__(self, security_manager: SecurityManager, config: AuthConfig):
        self.security_manager = security_manager
        self.config = config
        self.revoked_tokens: set = set()
    
    def create_access_token(self, user: User, session_id: Optional[str] = None) -> str:
        """Create JWT access token."""
        try:
            now = datetime.now()
            expires_at = now + timedelta(minutes=self.config.access_token_expire_minutes)
            
            payload = {
                "sub": user.id,
                "username": user.username,
                "role": user.role.value,
                "session_id": session_id,
                "iat": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
                "type": "access"
            }
            
            # Add permissions if available
            if user.permissions:
                payload["permissions"] = list(user.permissions)
            
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise HTTPException(status_code=500, detail="Failed to create token")
    
    def create_refresh_token(self, user: User, session_id: Optional[str] = None) -> str:
        """Create JWT refresh token."""
        try:
            now = datetime.now()
            expires_at = now + timedelta(days=self.config.refresh_token_expire_days)
            
            payload = {
                "sub": user.id,
                "username": user.username,
                "session_id": session_id,
                "iat": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
                "type": "refresh"
            }
            
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise HTTPException(status_code=500, detail="Failed to create refresh token")
    
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate JWT token and return token data."""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                return None
            
            # Decode token
            payload = jwt.decode(
                token, 
                self.config.secret_key, 
                algorithms=[self.config.algorithm]
            )
            
            # Extract token data
            user_id = payload.get("sub")
            username = payload.get("username")
            role = payload.get("role")
            session_id = payload.get("session_id")
            exp = payload.get("exp")
            permissions = payload.get("permissions", [])
            
            if not user_id or not username:
                return None
            
            # Check expiration
            if exp and datetime.now().timestamp() > exp:
                return None
            
            # Validate user exists and is active
            user = self.security_manager.users.get(user_id)
            if not user or not user.is_active:
                return None
            
            # Validate session if provided
            if session_id:
                session = self.security_manager.sessions.get(session_id)
                if not session or not session.is_active:
                    return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                role=role,
                session_id=session_id,
                expires_at=datetime.fromtimestamp(exp) if exp else None,
                permissions=permissions
            )
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token."""
        try:
            token_data = await self.validate_token(refresh_token)
            if not token_data:
                return None
            
            # Get user
            user = self.security_manager.users.get(token_data.user_id)
            if not user:
                return None
            
            # Create new access token
            return self.create_access_token(user, token_data.session_id)
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    def revoke_token(self, token: str):
        """Revoke a token."""
        self.revoked_tokens.add(token)
        
        # Cleanup old revoked tokens periodically
        if len(self.revoked_tokens) > 10000:
            self.revoked_tokens = set(list(self.revoked_tokens)[-5000:])

class SessionAuth:
    """Session-based authentication handler."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    async def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user."""
        try:
            return await self.security_manager.validate_session(session_id)
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None

class AuthManager:
    """Main authentication manager coordinating all auth methods."""
    
    def __init__(self, security_manager: SecurityManager, config: Optional[Dict[str, Any]] = None):
        self.security_manager = security_manager
        self.config = AuthConfig(config)
        
        # Initialize authentication handlers
        self.jwt_auth = JWTAuth(security_manager, self.config) if self.config.enable_jwt else None
        self.api_key_auth = APIKeyAuth(security_manager, self.config) if self.config.enable_api_key else None
        self.session_auth = SessionAuth(security_manager) if self.config.enable_session else None
        
        logger.info("AuthManager initialized")
    
    async def authenticate(self, username: str, password: str, 
                          ip_address: str = "", user_agent: str = "") -> AuthResponse:
        """Authenticate user with username/password."""
        try:
            # Authenticate with security manager
            session_id = await self.security_manager.authenticate_user(
                username, password, ip_address, user_agent
            )
            
            if not session_id:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Get user
            session = self.security_manager.sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=401, detail="Authentication failed")
            
            user = self.security_manager.users.get(session.user_id)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            # Create tokens
            access_token = ""
            refresh_token = None
            
            if self.jwt_auth:
                access_token = self.jwt_auth.create_access_token(user, session_id)
                refresh_token = self.jwt_auth.create_refresh_token(user, session_id)
            
            return AuthResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.config.access_token_expire_minutes * 60,
                user={
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "permissions": list(user.permissions)
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=500, detail="Authentication failed")
    
    async def validate_request(self, request: Request, 
                              token: Optional[str] = None,
                              api_key: Optional[str] = None) -> Optional[User]:
        """Validate request authentication."""
        try:
            # Try JWT token first
            if token and self.jwt_auth:
                token_data = await self.jwt_auth.validate_token(token)
                if token_data:
                    user = self.security_manager.users.get(token_data.user_id)
                    if user and user.is_active:
                        return user
            
            # Try API key
            if api_key and self.api_key_auth:
                user = await self.api_key_auth.validate_api_key(api_key)
                if user:
                    return user
            
            # Try session (from cookies)
            if self.session_auth:
                session_id = request.cookies.get("session_id")
                if session_id:
                    user = await self.session_auth.validate_session(session_id)
                    if user:
                        return user
            
            return None
            
        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            return None

# Global auth manager instance
_auth_manager: Optional[AuthManager] = None

def initialize_auth(security_manager: SecurityManager, config: Optional[Dict[str, Any]] = None):
    """Initialize authentication system."""
    global _auth_manager
    _auth_manager = AuthManager(security_manager, config)
    logger.info("Authentication system initialized")

async def authenticate_request(request: Request,
                              credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
                              api_key: Optional[str] = Depends(api_key_header)) -> Optional[User]:
    """Authenticate request using available methods."""
    if not _auth_manager:
        return None
    
    token = credentials.credentials if credentials else None
    return await _auth_manager.validate_request(request, token, api_key)

async def require_auth(user: Optional[User] = Depends(authenticate_request)) -> User:
    """Require authentication for endpoint."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user

async def require_role(required_role: UserRole):
    """Require specific role for endpoint."""
    def role_checker(user: User = Depends(require_auth)) -> User:
        if user.role.value < required_role.value:
            raise HTTPException(
                status_code=403,
                detail="Insufficient privileges"
            )
        return user
    return role_checker

async def require_permission(permission: str):
    """Require specific permission for endpoint."""
    def permission_checker(user: User = Depends(require_auth)) -> User:
        if not _auth_manager:
            raise HTTPException(status_code=500, detail="Auth manager not available")
        
        security_manager = _auth_manager.security_manager
        has_permission = asyncio.create_task(
            security_manager.check_permission(user, permission)
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail=f"Permission required: {permission}"
            )
        return user
    return permission_checker

async def get_current_user(token: str) -> Optional[User]:
    """Get current user from token."""
    if not _auth_manager or not _auth_manager.jwt_auth:
        return None
    
    token_data = await _auth_manager.jwt_auth.validate_token(token)
    if not token_data:
        return None
    
    return _auth_manager.security_manager.users.get(token_data.user_id)

def get_auth_manager() -> Optional[AuthManager]:
    """Get the global auth manager instance."""
    return _auth_manager
