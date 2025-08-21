"""
Ultra AI Project - Security Manager

Handles authentication, authorization, encryption, and security policies
for the Ultra AI system. Provides comprehensive security enforcement.
"""

import asyncio
import hashlib
import secrets
import jwt
import bcrypt
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from enum import Enum

from ..utils.logger import get_logger
from ..utils.helpers import generate_session_id

logger = get_logger(__name__)

class UserRole(Enum):
    """User role definitions."""
    GUEST = "guest"
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SYSTEM = "system"

class PermissionLevel(Enum):
    """Permission level definitions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class User:
    """User account information."""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    """Security event for auditing."""
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: str  # low, medium, high, critical

class SecurityManager:
    """Central security management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Security settings
        self.secret_key = config.get("secret_key", secrets.token_urlsafe(32))
        self.jwt_algorithm = config.get("algorithm", "HS256")
        self.session_timeout = config.get("session_timeout", 3600)  # 1 hour
        self.max_login_attempts = config.get("max_login_attempts", 5)
        self.lockout_duration = config.get("lockout_duration", 1800)  # 30 minutes
        
        # Storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.revoked_tokens: Set[str] = set()
        self.security_events: List[SecurityEvent] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max_requests = 100
        
        # Content filtering
        self.blocked_patterns: List[re.Pattern] = []
        self.sensitive_data_patterns: List[re.Pattern] = []
        
        # IP whitelist/blacklist
        self.ip_whitelist: Set[str] = set(config.get("ip_whitelist", []))
        self.ip_blacklist: Set[str] = set(config.get("ip_blacklist", []))
        
        self._initialize_security_patterns()
        logger.info("SecurityManager initialized")
    
    def _initialize_security_patterns(self):
        """Initialize security patterns for content filtering."""
        # Blocked content patterns
        blocked_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
        ]
        
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in blocked_patterns]
        
        # Sensitive data patterns
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
        ]
        
        self.sensitive_data_patterns = [re.compile(pattern) 
                                      for pattern in sensitive_patterns]
    
    async def initialize(self):
        """Initialize security manager."""
        try:
            logger.info("Initializing SecurityManager...")
            
            # Create default admin user if none exists
            if not self.users:
                await self._create_default_admin()
            
            # Load existing data
            await self._load_security_data()
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_loop())
            
            logger.info("SecurityManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SecurityManager: {e}")
            raise
    
    async def _create_default_admin(self):
        """Create default admin user."""
        admin_password = secrets.token_urlsafe(16)
        
        await self.create_user(
            username="admin",
            email="admin@ultraai.local",
            password=admin_password,
            role=UserRole.ADMIN
        )
        
        logger.warning(f"Created default admin user with password: {admin_password}")
        logger.warning("Please change the admin password immediately!")
    
    async def _load_security_data(self):
        """Load existing security data."""
        # This would typically load from a database
        # For now, we'll use file-based storage
        try:
            security_file = Path("runtime/security/security_data.json")
            if security_file.exists():
                with open(security_file, 'r') as f:
                    data = json.load(f)
                    
                # Load users (simplified for this example)
                for user_data in data.get("users", []):
                    user = User(
                        id=user_data["id"],
                        username=user_data["username"],
                        email=user_data["email"],
                        password_hash=user_data["password_hash"],
                        role=UserRole(user_data["role"]),
                        permissions=set(user_data.get("permissions", [])),
                        created_at=datetime.fromisoformat(user_data["created_at"]),
                        is_active=user_data.get("is_active", True),
                        is_verified=user_data.get("is_verified", False)
                    )
                    self.users[user.id] = user
                    
                logger.info(f"Loaded {len(self.users)} users")
                
        except Exception as e:
            logger.error(f"Failed to load security data: {e}")
    
    async def _save_security_data(self):
        """Save security data to storage."""
        try:
            security_dir = Path("runtime/security")
            security_dir.mkdir(parents=True, exist_ok=True)
            
            data = {
                "users": [
                    {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "password_hash": user.password_hash,
                        "role": user.role.value,
                        "permissions": list(user.permissions),
                        "created_at": user.created_at.isoformat(),
                        "is_active": user.is_active,
                        "is_verified": user.is_verified
                    }
                    for user in self.users.values()
                ]
            }
            
            with open(security_dir / "security_data.json", 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save security data: {e}")
    
    async def create_user(self, username: str, email: str, password: str, 
                         role: UserRole = UserRole.USER) -> str:
        """Create a new user account."""
        # Validate input
        if not self._validate_username(username):
            raise ValueError("Invalid username format")
        
        if not self._validate_email(email):
            raise ValueError("Invalid email format")
        
        if not self._validate_password(password):
            raise ValueError("Password does not meet requirements")
        
        # Check if user already exists
        for user in self.users.values():
            if user.username == username or user.email == email:
                raise ValueError("User already exists")
        
        # Create user
        user_id = f"user_{secrets.token_urlsafe(16)}"
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role
        )
        
        self.users[user_id] = user
        await self._save_security_data()
        
        # Log security event
        await self._log_security_event(
            event_type="user_created",
            user_id=user_id,
            details={"username": username, "email": email, "role": role.value},
            severity="medium"
        )
        
        logger.info(f"Created user: {username} ({user_id})")
        return user_id
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format."""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        
        pattern = re.compile(r'^[a-zA-Z0-9_.-]+$')
        return pattern.match(username) is not None
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return pattern.match(email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False
        
        # Check for at least one uppercase, lowercase, digit, and special char
        checks = [
            re.search(r'[A-Z]', password),
            re.search(r'[a-z]', password),
            re.search(r'\d', password),
            re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
        ]
        
        return sum(1 for check in checks if check) >= 3
    
    async def authenticate_user(self, username: str, password: str, 
                               ip_address: str = "", user_agent: str = "") -> Optional[str]:
        """Authenticate user and create session."""
        
        # Check rate limiting
        if not await self._check_rate_limit(ip_address):
            await self._log_security_event(
                event_type="rate_limit_exceeded",
                ip_address=ip_address,
                details={"username": username},
                severity="medium"
            )
            raise PermissionError("Rate limit exceeded")
        
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user:
            await self._log_security_event(
                event_type="login_failed",
                ip_address=ip_address,
                details={"username": username, "reason": "user_not_found"},
                severity="low"
            )
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            await self._log_security_event(
                event_type="login_failed",
                user_id=user.id,
                ip_address=ip_address,
                details={"username": username, "reason": "account_locked"},
                severity="medium"
            )
            raise PermissionError("Account is locked")
        
        # Check password
        if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(seconds=self.lockout_duration)
                logger.warning(f"Account locked for user {user.username}")
            
            await self._log_security_event(
                event_type="login_failed",
                user_id=user.id,
                ip_address=ip_address,
                details={"username": username, "reason": "invalid_password"},
                severity="medium"
            )
            return None
        
        # Check if user is active
        if not user.is_active:
            await self._log_security_event(
                event_type="login_failed",
                user_id=user.id,
                ip_address=ip_address,
                details={"username": username, "reason": "account_inactive"},
                severity="medium"
            )
            return None
        
        # Authentication successful
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Create session
        session_id = await self._create_session(user.id, ip_address, user_agent)
        
        await self._log_security_event(
            event_type="login_success",
            user_id=user.id,
            session_id=session_id,
            ip_address=ip_address,
            details={"username": username},
            severity="low"
        )
        
        logger.info(f"User authenticated: {username}")
        return session_id
    
    async def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new user session."""
        session_id = generate_session_id()
        expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if datetime.now() > session.expires_at:
            await self.logout_user(session_id)
            return None
        
        # Check if session is active
        if not session.is_active:
            return None

        # Update last activity
        session.last_activity = datetime.now()
        
        # Get user
        user = self.users.get(session.user_id)
        if not user or not user.is_active:
            await self.logout_user(session_id)
            return None
        
        return user
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.is_active = False
        
        await self._log_security_event(
            event_type="logout",
            user_id=session.user_id,
            session_id=session_id,
            ip_address=session.ip_address,
            details={},
            severity="low"
        )
        
        # Remove session after a delay (for auditing)
        asyncio.create_task(self._cleanup_session(session_id, delay=300))
        
        logger.info(f"User logged out: {session_id}")
        return True
    
    async def _cleanup_session(self, session_id: str, delay: int = 0):
        """Clean up session after delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        
        self.sessions.pop(session_id, None)
    
    async def generate_jwt_token(self, user_id: str, session_id: str) -> str:
        """Generate JWT token for API access."""
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "iat": time.time(),
            "exp": time.time() + self.session_timeout
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
        return token
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token."""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            
            # Validate session
            session_id = payload.get("session_id")
            user = await self.validate_session(session_id)
            
            if not user:
                return None
            
            return {
                "user_id": payload["user_id"],
                "session_id": session_id,
                "user": user
            }
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def revoke_token(self, token: str):
        """Revoke a JWT token."""
        self.revoked_tokens.add(token)
        
        # Clean up old revoked tokens periodically
        if len(self.revoked_tokens) > 10000:
            # Keep only recent tokens (this is simplified)
            self.revoked_tokens = set(list(self.revoked_tokens)[-5000:])
    
    async def check_permission(self, user: User, permission: str, resource: str = "") -> bool:
        """Check if user has specific permission."""
        # System users have all permissions
        if user.role == UserRole.SYSTEM:
            return True
        
        # Admin users have all permissions except system-level
        if user.role == UserRole.ADMIN and not permission.startswith("system."):
            return True
        
        # Check specific permissions
        full_permission = f"{permission}.{resource}" if resource else permission
        
        if full_permission in user.permissions:
            return True
        
        # Check wildcard permissions
        wildcard_permission = f"{permission}.*"
        if wildcard_permission in user.permissions:
            return True
        
        # Check role-based permissions
        role_permissions = self._get_role_permissions(user.role)
        return permission in role_permissions
    
    def _get_role_permissions(self, role: UserRole) -> Set[str]:
        """Get default permissions for role."""
        permissions_map = {
            UserRole.GUEST: {"read"},
            UserRole.USER: {"read", "write"},
            UserRole.MODERATOR: {"read", "write", "moderate"},
            UserRole.ADMIN: {"read", "write", "moderate", "admin"},
            UserRole.SYSTEM: {"read", "write", "moderate", "admin", "system"}
        }
        
        return permissions_map.get(role, set())
    
    async def authorize_task(self, task) -> bool:
        """Authorize task execution."""
        # This is a simplified authorization check
        # In a real system, this would be more sophisticated
        
        user_id = task.metadata.get("user_id")
        if not user_id:
            return True  # Allow system tasks
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Check if user can execute tasks
        return await self.check_permission(user, "execute", "tasks")
    
    async def filter_content(self, content: str) -> Tuple[str, bool]:
        """Filter content for security and return (filtered_content, is_safe)."""
        original_content = content
        is_safe = True
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(content):
                content = pattern.sub("[BLOCKED]", content)
                is_safe = False
        
        # Mask sensitive data
        for pattern in self.sensitive_data_patterns:
            content = pattern.sub("[REDACTED]", content)
        
        if not is_safe:
            await self._log_security_event(
                event_type="content_filtered",
                details={"original_length": len(original_content), "filtered_length": len(content)},
                severity="medium"
            )
        
        return content, is_safe
    
    async def _check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for identifier (IP, user, etc.)."""
        now = time.time()
        
        # Clean old entries
        if identifier in self.rate_limits:
            self.rate_limits[identifier] = [
                timestamp for timestamp in self.rate_limits[identifier]
                if now - timestamp < self.rate_limit_window
            ]
        else:
            self.rate_limits[identifier] = []
        
        # Check limit
        if len(self.rate_limits[identifier]) >= self.rate_limit_max_requests:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True
    
    async def _log_security_event(self, event_type: str, user_id: Optional[str] = None,
                                 session_id: Optional[str] = None, ip_address: str = "",
                                 details: Optional[Dict[str, Any]] = None, severity: str = "low"):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            timestamp=datetime.now(),
            details=details or {},
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Keep limited history
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        # Log to file/database in production
        logger.info(f"Security event: {event_type} - {severity} - {details}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired sessions and data."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_sessions()
                await self._cleanup_old_events()
                
            except Exception as e:
                logger.error(f"Security cleanup error: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if now > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _cleanup_old_events(self):
        """Remove old security events."""
        cutoff = datetime.now() - timedelta(days=30)
        old_count = len(self.security_events)
        
        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff
        ]
        
        removed = old_count - len(self.security_events)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old security events")
    
    async def health_check(self) -> bool:
        """Perform security health check."""
        try:
            # Check critical components
            checks = [
                len(self.users) > 0,  # Has users
                len(self.secret_key) >= 32,  # Strong secret key
                self.max_login_attempts > 0,  # Rate limiting enabled
            ]
            
            return all(checks)
            
        except Exception as e:
            logger.error(f"Security health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown security manager."""
        logger.info("Shutting down SecurityManager...")
        
        # Save security data
        await self._save_security_data()
        
        logger.info("SecurityManager shutdown complete")

# Convenience functions for external use
async def authenticate_user(username: str, password: str, security_manager: SecurityManager,
                           ip_address: str = "", user_agent: str = "") -> Optional[str]:
    """Authenticate user using security manager."""
    return await security_manager.authenticate_user(username, password, ip_address, user_agent)

async def authorize_action(user: User, action: str, resource: str, 
                          security_manager: SecurityManager) -> bool:
    """Authorize user action using security manager."""
    return await security_manager.check_permission(user, action, resource)
