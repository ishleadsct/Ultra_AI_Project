"""
Ultra AI Project - API Middleware

Comprehensive middleware components for request/response processing,
security, rate limiting, logging, CORS, and error handling.
"""

import time
import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
import httpx

from ..utils.logger import get_logger
from . import create_error_response

logger = get_logger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses with timing."""
    
    def __init__(self, app: FastAPI, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.start_times: Dict[str, float] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = time.time()
        self.start_times[request_id] = start_time
        
        # Log request
        if self.log_requests:
            client_ip = self._get_client_ip(request)
            logger.info(
                "Request received",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": client_ip,
                    "user_agent": request.headers.get("user-agent", ""),
                    "content_length": request.headers.get("content-length", "0")
                }
            )
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response
            if self.log_requests:  # Use same flag for response logging
                logger.info(
                    "Request completed",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "processing_time": processing_time,
                        "response_size": response.headers.get("content-length", "unknown")
                    }
                )
            
            # Add timing header
            response.headers["X-Process-Time"] = f"{processing_time:.3f}"
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time": processing_time
                }
            )
            raise
        finally:
            # Cleanup
            self.start_times.pop(request_id, None)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with configurable limits per IP/user."""
    
    def __init__(self, 
                 app: FastAPI,
                 requests_per_minute: int = 60,
                 burst_size: int = 10,
                 whitelist: Optional[Set[str]] = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.whitelist = whitelist or set()
        
        # Rate limiting storage
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.burst_counts: Dict[str, int] = defaultdict(int)
        self.last_cleanup = time.time()
        
        # Cleanup interval (5 minutes)
        self.cleanup_interval = 300
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        
        # Skip rate limiting for whitelisted IPs
        if client_ip in self.whitelist:
            return await call_next(request)
        
        # Cleanup old entries periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        # Check rate limits
        if not await self._check_rate_limit(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content=create_error_response(
                    message="Rate limit exceeded. Please try again later.",
                    request_id=getattr(request.state, 'request_id', None)
                ).dict(),
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_ip, current_time)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    async def _check_rate_limit(self, identifier: str, current_time: float) -> bool:
        """Check if request is within rate limits."""
        request_times = self.request_counts[identifier]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Check minute-based limit
        if len(request_times) >= self.requests_per_minute:
            return False
        
        # Check burst limit (requests in last 10 seconds)
        burst_cutoff = current_time - 10
        recent_requests = sum(1 for t in request_times if t > burst_cutoff)
        if recent_requests >= self.burst_size:
            return False
        
        # Add current request
        request_times.append(current_time)
        return True
    
    async def _get_remaining_requests(self, identifier: str, current_time: float) -> int:
        """Get remaining requests for the current minute."""
        request_times = self.request_counts[identifier]
        cutoff_time = current_time - 60
        
        # Count requests in current minute
        current_minute_requests = sum(1 for t in request_times if t > cutoff_time)
        return max(0, self.requests_per_minute - current_minute_requests)
    
    async def _cleanup_old_entries(self):
        """Clean up old rate limiting entries."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        for identifier in list(self.request_counts.keys()):
            request_times = self.request_counts[identifier]
            
            # Remove old requests
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()
            
            # Remove empty entries
            if not request_times:
                del self.request_counts[identifier]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and content validation."""
    
    def __init__(self, 
                 app: FastAPI,
                 enable_csrf: bool = True,
                 enable_xss_protection: bool = True,
                 max_request_size: int = 100 * 1024 * 1024):  # 100MB
        super().__init__(app)
        self.enable_csrf = enable_csrf
        self.enable_xss_protection = enable_xss_protection
        self.max_request_size = max_request_size
        
        # Security patterns
        self.xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            re.compile(r'\b(union|select|insert|update|delete|drop|create|alter)\b', re.IGNORECASE),
            re.compile(r'[\'";]', re.IGNORECASE),
            re.compile(r'--', re.IGNORECASE),
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content=create_error_response("Request too large").dict()
            )
        
        # Validate request content
        if self.enable_xss_protection:
            await self._validate_request_content(request)
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        await self._add_security_headers(response)
        
        return response
    
    async def _validate_request_content(self, request: Request):
        """Validate request content for security threats."""
        # Check URL parameters
        for key, value in request.query_params.items():
            if self._contains_malicious_content(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid content in parameter: {key}"
                )
        
        # Check headers
        for name, value in request.headers.items():
            if self._contains_malicious_content(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid content in header: {name}"
                )
    
    def _contains_malicious_content(self, content: str) -> bool:
        """Check if content contains malicious patterns."""
        if not isinstance(content, str):
            return False
        
        # Check XSS patterns
        for pattern in self.xss_patterns:
            if pattern.search(content):
                return True
        
        # Check SQL injection patterns (basic check)
        for pattern in self.sql_patterns:
            if pattern.search(content):
                return True
        
        return False
    
    async def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https: wss:; "
            "frame-ancestors 'none';"
        )
        
        # XSS Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Content Type Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Frame Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # Strict Transport Security
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "magnetometer=(), gyroscope=(), payment=()"
        )

class CacheMiddleware(BaseHTTPMiddleware):
    """Caching middleware for GET requests."""
    
    def __init__(self, app: FastAPI, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Create cache key
        cache_key = self._create_cache_key(request)
        
        # Check cache
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            response = JSONResponse(content=cached_response["content"])
            response.headers.update(cached_response["headers"])
            response.headers["X-Cache"] = "HIT"
            return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            await self._cache_response(cache_key, response)
            response.headers["X-Cache"] = "MISS"
        
        # Cleanup old cache entries
        current_time = time.time()
        if current_time - self.last_cleanup > 300:  # 5 minutes
            await self._cleanup_cache()
            self.last_cleanup = current_time
        
        return response
    
    def _create_cache_key(self, request: Request) -> str:
        """Create cache key from request."""
        url_without_query = str(request.url).split('?')[0]
        query_params = sorted(request.query_params.items())
        query_string = "&".join(f"{k}={v}" for k, v in query_params)
        return f"{url_without_query}?{query_string}" if query_string else url_without_query
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        if time.time() > cached_item["expires_at"]:
            del self.cache[cache_key]
            return None
        
        return cached_item
    
    async def _cache_response(self, cache_key: str, response: Response):
        """Cache response."""
        try:
            # Only cache JSON responses
            if response.headers.get("content-type", "").startswith("application/json"):
                content = json.loads(response.body.decode()) if hasattr(response, 'body') else None
                
                if content:
                    self.cache[cache_key] = {
                        "content": content,
                        "headers": dict(response.headers),
                        "expires_at": time.time() + self.cache_ttl,
                        "cached_at": time.time()
                    }
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    async def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time > item["expires_at"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

class CompressionMiddleware(BaseHTTPMiddleware):
    """Enhanced compression middleware."""
    
    def __init__(self, app: FastAPI, minimum_size: int = 1024):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if compression is appropriate
        if self._should_compress(request, response):
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def _should_compress(self, request: Request, response: Response) -> bool:
        """Determine if response should be compressed."""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return False
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "application/javascript",
            "text/html",
            "text/css",
            "text/plain",
            "text/xml"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return False
        
        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False
        
        return True

def setup_middleware(app: FastAPI, config: Optional[Dict[str, Any]] = None):
    """Setup all middleware components."""
    config = config or {}
    
    # CORS Middleware
    cors_config = config.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("origins", ["*"]),
        allow_credentials=cors_config.get("credentials", True),
        allow_methods=cors_config.get("methods", ["*"]),
        allow_headers=cors_config.get("headers", ["*"]),
        expose_headers=["X-Request-ID", "X-Process-Time", "X-Cache"]
    )
    
    # Compression Middleware
    if config.get("enable_compression", True):
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted Host Middleware
    trusted_hosts = config.get("trusted_hosts", ["*"])
    if trusted_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    
    # Security Middleware
    security_config = config.get("security", {})
    app.add_middleware(
        SecurityMiddleware,
        enable_csrf=security_config.get("enable_csrf", True),
        enable_xss_protection=security_config.get("enable_xss_protection", True),
        max_request_size=security_config.get("max_request_size", 100 * 1024 * 1024)
    )
    
    # Rate Limiting Middleware
    rate_limit_config = config.get("rate_limit", {})
    if rate_limit_config.get("enabled", True):
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rate_limit_config.get("requests_per_minute", 60),
            burst_size=rate_limit_config.get("burst_size", 10),
            whitelist=set(rate_limit_config.get("whitelist", []))
        )
    
    # Caching Middleware
    cache_config = config.get("cache", {})
    if cache_config.get("enabled", False):
        app.add_middleware(
            CacheMiddleware,
            cache_ttl=cache_config.get("ttl", 300)
        )
    
    # Logging Middleware
    logging_config = config.get("logging", {})
    app.add_middleware(
        LoggingMiddleware,
        log_requests=logging_config.get("log_requests", True),
        log_responses=logging_config.get("log_responses", False)
    )
    
    # Request ID Middleware (should be last to ensure ID is available to all other middleware)
    app.add_middleware(RequestIDMiddleware)
    
    logger.info("API middleware setup complete")
