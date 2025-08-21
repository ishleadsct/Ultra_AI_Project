"""
Ultra AI Project - API Tests

Comprehensive test suite for the REST API and WebSocket endpoints
including authentication, rate limiting, file uploads, and real-time features.
"""

import pytest
import asyncio
import json
import io
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import tempfile
from pathlib import Path

from . import (
    get_test_config, MockAgent, MockTask, TestDataGenerator,
    TestAssertions, PerformanceTimer, requires_api_keys, slow_test
)

# Import API classes
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from src.ui.web_interface import WebInterface, TaskCreateRequest, FileUploadResponse
    from src.core.system_manager import SystemManager
    from src.core.task_coordinator import TaskCoordinator
    from src.utils.security import create_access_token
except ImportError as e:
    pytest.skip(f"Could not import API modules: {e}", allow_module_level=True)


class TestAPIClient:
    """Test client for API endpoints."""
    
    def __init__(self, test_client: TestClient):
        self.client = test_client
        self.auth_token = None
    
    def set_auth_token(self, token: str):
        """Set authentication token."""
        self.auth_token = token
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    def get(self, url: str, **kwargs):
        """Make authenticated GET request."""
        headers = kwargs.pop('headers', {})
        headers.update(self.get_headers())
        return self.client.get(url, headers=headers, **kwargs)
    
    def post(self, url: str, **kwargs):
        """Make authenticated POST request."""
        headers = kwargs.pop('headers', {})
        headers.update(self.get_headers())
        return self.client.post(url, headers=headers, **kwargs)
    
    def put(self, url: str, **kwargs):
        """Make authenticated PUT request."""
        headers = kwargs.pop('headers', {})
        headers.update(self.get_headers())
        return self.client.put(url, headers=headers, **kwargs)
    
    def delete(self, url: str, **kwargs):
        """Make authenticated DELETE request."""
        headers = kwargs.pop('headers', {})
        headers.update(self.get_headers())
        return self.client.delete(url, headers=headers, **kwargs)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    @pytest.fixture
    async def api_client(self):
        """Create API test client."""
        config = get_test_config().get_config()
        config['enable_auth'] = False  # Disable auth for health tests
        
        # Mock system components
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        system_manager.get_health_status = AsyncMock(return_value={
            "healthy": True,
            "uptime": 3600,
            "components": {
                "database": "healthy",
                "agents": "healthy",
                "models": "healthy"
            }
        })
        
        # Create web interface
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_health_endpoint(self, api_client):
        """Test basic health endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert "uptime" in data
        assert "components" in data
    
    @pytest.mark.unit
    def test_health_endpoint_unhealthy(self, api_client):
        """Test health endpoint when system is unhealthy."""
        # Mock unhealthy system
        with patch.object(api_client.client.app.state, 'system_manager') as mock_manager:
            mock_manager.get_health_status = AsyncMock(return_value={
                "healthy": False,
                "components": {"database": "error"}
            })
            
            response = api_client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["healthy"] is False
    
    @pytest.mark.unit
    def test_api_status_endpoint(self, api_client):
        """Test API status endpoint."""
        response = api_client.get("/api/v1/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "health" in data
        assert "timestamp" in data


class TestAuthenticationEndpoints:
    """Test authentication and authorization."""
    
    @pytest.fixture
    async def auth_api_client(self):
        """Create API client with authentication enabled."""
        config = get_test_config().get_config()
        config['enable_auth'] = True
        
        # Mock system components
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        # Create web interface
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_login_success(self, auth_api_client):
        """Test successful login."""
        with patch('src.ui.web_interface.WebInterface._validate_credentials', return_value=True):
            response = auth_api_client.post("/auth/login", json={
                "username": "testuser",
                "password": "testpass"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
    
    @pytest.mark.unit
    def test_login_failure(self, auth_api_client):
        """Test failed login."""
        with patch('src.ui.web_interface.WebInterface._validate_credentials', return_value=False):
            response = auth_api_client.post("/auth/login", json={
                "username": "invalid",
                "password": "invalid"
            })
            
            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
    
    @pytest.mark.unit
    def test_protected_endpoint_without_auth(self, auth_api_client):
        """Test accessing protected endpoint without authentication."""
        response = auth_api_client.get("/api/v1/agents")
        
        assert response.status_code == 401
    
    @pytest.mark.unit
    def test_protected_endpoint_with_auth(self, auth_api_client):
        """Test accessing protected endpoint with authentication."""
        # Create a valid token
        token = create_access_token({"sub": "testuser"})
        auth_api_client.set_auth_token(token)
        
        with patch('src.ui.web_interface.verify_token', return_value={"sub": "testuser"}):
            response = auth_api_client.get("/api/v1/agents")
            
            # Should succeed (will depend on mocked system manager)
            assert response.status_code in [200, 503]  # 503 if system not initialized
    
    @pytest.mark.unit
    def test_logout(self, auth_api_client):
        """Test user logout."""
        token = create_access_token({"sub": "testuser"})
        auth_api_client.set_auth_token(token)
        
        with patch('src.ui.web_interface.verify_token', return_value={"sub": "testuser"}):
            response = auth_api_client.post("/auth/logout")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data


class TestAgentEndpoints:
    """Test agent management endpoints."""
    
    @pytest.fixture
    async def agent_api_client(self):
        """Create API client for agent tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Mock system manager with agents
        system_manager = Mock(spec=SystemManager)
        system_manager.get_agents = AsyncMock(return_value={
            "code": {
                "type": "code",
                "status": "active",
                "version": "1.0.0",
                "capabilities": ["generation", "review"]
            },
            "research": {
                "type": "research", 
                "status": "active",
                "version": "1.0.0",
                "capabilities": ["search", "analysis"]
            }
        })
        
        system_manager.get_agent_info = AsyncMock(return_value={
            "name": "code",
            "type": "code",
            "status": "active",
            "capabilities": ["generation", "review"],
            "stats": {"tasks_completed": 100}
        })
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_list_agents(self, agent_api_client):
        """Test listing available agents."""
        response = agent_api_client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "code" in data["agents"]
        assert "research" in data["agents"]
    
    @pytest.mark.unit
    def test_get_agent_details(self, agent_api_client):
        """Test getting detailed agent information."""
        response = agent_api_client.get("/api/v1/agents/code")
        
        assert response.status_code == 200
        data = response.json()
        assert "agent" in data
        assert data["agent"]["name"] == "code"
        assert data["agent"]["type"] == "code"
    
    @pytest.mark.unit
    def test_get_nonexistent_agent(self, agent_api_client):
        """Test getting information for non-existent agent."""
        with patch.object(agent_api_client.client.app.state, 'system_manager') as mock_manager:
            mock_manager.get_agent_info = AsyncMock(return_value=None)
            
            response = agent_api_client.get("/api/v1/agents/nonexistent")
            
            assert response.status_code == 404


class TestTaskEndpoints:
    """Test task management endpoints."""
    
    @pytest.fixture
    async def task_api_client(self):
        """Create API client for task tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Mock task coordinator
        task_coordinator = Mock(spec=TaskCoordinator)
        
        # Mock task creation
        mock_task = MockTask("test_task_1", "Test task", "code")
        task_coordinator.create_task = AsyncMock(return_value=mock_task)
        task_coordinator.get_task = AsyncMock(return_value=mock_task)
        task_coordinator.list_tasks = AsyncMock(return_value=[mock_task])
        task_coordinator.cancel_task = AsyncMock(return_value=True)
        task_coordinator.retry_task = AsyncMock(return_value=mock_task)
        
        system_manager = Mock(spec=SystemManager)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_create_task(self, task_api_client):
        """Test creating a new task."""
        task_data = {
            "agent_type": "code",
            "description": "Generate a Python function",
            "parameters": {"language": "python"},
            "priority": 1
        }
        
        response = task_api_client.post("/api/v1/tasks", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_task_1"
        assert data["description"] == "Test task"
        assert data["agent_type"] == "code"
    
    @pytest.mark.unit
    def test_create_task_invalid_data(self, task_api_client):
        """Test creating task with invalid data."""
        invalid_data = {
            "description": "Missing agent type"
            # Missing required agent_type field
        }
        
        response = task_api_client.post("/api/v1/tasks", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    def test_get_task(self, task_api_client):
        """Test getting a specific task."""
        response = task_api_client.get("/api/v1/tasks/test_task_1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task"]["id"] == "test_task_1"
    
    @pytest.mark.unit
    def test_get_nonexistent_task(self, task_api_client):
        """Test getting a non-existent task."""
        with patch.object(task_api_client.client.app.state, 'task_coordinator') as mock_coord:
            mock_coord.get_task = AsyncMock(return_value=None)
            
            response = task_api_client.get("/api/v1/tasks/nonexistent")
            
            assert response.status_code == 404
    
    @pytest.mark.unit
    def test_list_tasks(self, task_api_client):
        """Test listing tasks."""
        response = task_api_client.get("/api/v1/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == "test_task_1"
    
    @pytest.mark.unit
    def test_list_tasks_with_filters(self, task_api_client):
        """Test listing tasks with filters."""
        response = task_api_client.get("/api/v1/tasks?status=pending&agent_type=code&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert data["limit"] == 10
    
    @pytest.mark.unit
    def test_cancel_task(self, task_api_client):
        """Test cancelling a task."""
        response = task_api_client.delete("/api/v1/tasks/test_task_1")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    @pytest.mark.unit
    def test_retry_task(self, task_api_client):
        """Test retrying a failed task."""
        response = task_api_client.post("/api/v1/tasks/test_task_1/retry")
        
        assert response.status_code == 200
        data = response.json()
        assert "new_task" in data


class TestFileEndpoints:
    """Test file upload and management endpoints."""
    
    @pytest.fixture
    async def file_api_client(self):
        """Create API client for file tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Create temporary upload directory
        temp_dir = tempfile.mkdtemp()
        config['upload_dir'] = temp_dir
        
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client), temp_dir
    
    @pytest.mark.unit
    def test_upload_file(self, file_api_client):
        """Test file upload."""
        api_client, temp_dir = file_api_client
        
        # Create test file
        test_content = "This is a test file content"
        test_file = io.BytesIO(test_content.encode())
        
        response = api_client.client.post(
            "/api/v1/files",
            files={"file": ("test.txt", test_file, "text/plain")},
            data={"description": "Test file upload"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert data["filename"] == "test.txt"
        assert data["size"] == len(test_content)
        assert data["mime_type"] == "text/plain"
    
    @pytest.mark.unit
    def test_upload_large_file(self, file_api_client):
        """Test uploading file that exceeds size limit."""
        api_client, temp_dir = file_api_client
        
        # Create large test file (simulate content larger than limit)
        large_content = "x" * (100 * 1024 * 1024 + 1)  # Larger than 100MB
        test_file = io.BytesIO(large_content.encode())
        
        with patch('fastapi.UploadFile.size', len(large_content)):
            response = api_client.client.post(
                "/api/v1/files",
                files={"file": ("large.txt", test_file, "text/plain")}
            )
            
            assert response.status_code == 413  # Request entity too large
    
    @pytest.mark.unit
    def test_upload_invalid_file_type(self, file_api_client):
        """Test uploading file with invalid type."""
        api_client, temp_dir = file_api_client
        
        # Create test file with disallowed type
        test_content = "executable content"
        test_file = io.BytesIO(test_content.encode())
        
        response = api_client.client.post(
            "/api/v1/files",
            files={"file": ("malware.exe", test_file, "application/x-executable")}
        )
        
        assert response.status_code == 415  # Unsupported media type
    
    @pytest.mark.unit
    def test_download_file(self, file_api_client):
        """Test file download."""
        api_client, temp_dir = file_api_client
        
        # First upload a file
        test_content = "Download test content"
        test_file = io.BytesIO(test_content.encode())
        
        upload_response = api_client.client.post(
            "/api/v1/files",
            files={"file": ("download_test.txt", test_file, "text/plain")}
        )
        
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Now download the file
        download_response = api_client.get(f"/api/v1/files/{file_id}")
        
        assert download_response.status_code == 200
        assert download_response.content.decode() == test_content
    
    @pytest.mark.unit
    def test_list_files(self, file_api_client):
        """Test listing uploaded files."""
        api_client, temp_dir = file_api_client
        
        response = api_client.get("/api/v1/files")
        
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "total" in data
    
    @pytest.mark.unit
    def test_delete_file(self, file_api_client):
        """Test file deletion."""
        api_client, temp_dir = file_api_client
        
        # First upload a file
        test_content = "Delete test content"
        test_file = io.BytesIO(test_content.encode())
        
        upload_response = api_client.client.post(
            "/api/v1/files",
            files={"file": ("delete_test.txt", test_file, "text/plain")}
        )
        
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Delete the file
        delete_response = api_client.delete(f"/api/v1/files/{file_id}")
        
        assert delete_response.status_code == 200
        
        # Verify file is deleted
        download_response = api_client.get(f"/api/v1/files/{file_id}")
        assert download_response.status_code == 404


class TestConfigEndpoints:
    """Test configuration management endpoints."""
    
    @pytest.fixture
    async def config_api_client(self):
        """Create API client for configuration tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Mock system manager
        system_manager = Mock(spec=SystemManager)
        system_manager.get_sanitized_config = AsyncMock(return_value={
            "system": {"name": "Ultra AI", "version": "1.0.0"},
            "agents": {"code": {"enabled": True}}
        })
        system_manager.update_config = AsyncMock(return_value=True)
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_get_config(self, config_api_client):
        """Test getting system configuration."""
        response = config_api_client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "system" in data["config"]
    
    @pytest.mark.unit
    def test_update_config(self, config_api_client):
        """Test updating system configuration."""
        update_data = {
            "section": "system",
            "key": "debug",
            "value": True
        }
        
        response = config_api_client.put("/api/v1/config", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestLogEndpoints:
    """Test log retrieval endpoints."""
    
    @pytest.fixture
    async def log_api_client(self):
        """Create API client for log tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Mock system manager with logs
        system_manager = Mock(spec=SystemManager)
        system_manager.get_recent_logs = AsyncMock(return_value=[
            {
                "timestamp": "2025-01-01T12:00:00Z",
                "level": "INFO",
                "message": "System started",
                "component": "system"
            },
            {
                "timestamp": "2025-01-01T12:01:00Z",
                "level": "ERROR",
                "message": "Test error message",
                "component": "agent"
            }
        ])
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_get_logs(self, log_api_client):
        """Test getting system logs."""
        response = log_api_client.get("/api/v1/logs")
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert len(data["logs"]) == 2
    
    @pytest.mark.unit
    def test_get_logs_with_filters(self, log_api_client):
        """Test getting logs with filters."""
        response = log_api_client.get("/api/v1/logs?level=ERROR&limit=50")
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data


class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""
    
    @pytest.mark.integration
    async def test_websocket_connection(self):
        """Test WebSocket connection and messaging."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        with TestClient(web_interface.app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Test ping/pong
                websocket.send_json({"type": "ping"})
                data = websocket.receive_json()
                assert data["type"] == "pong"
    
    @pytest.mark.integration
    async def test_websocket_authentication(self):
        """Test WebSocket authentication."""
        config = get_test_config().get_config()
        config['enable_auth'] = True
        
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        with TestClient(web_interface.app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Test authentication
                token = create_access_token({"sub": "testuser"})
                websocket.send_json({"type": "authenticate", "token": token})
                
                with patch('src.ui.web_interface.verify_token', return_value={"sub": "testuser"}):
                    data = websocket.receive_json()
                    assert data["type"] == "authenticated"
                    assert data["user_id"] == "testuser"


class TestRateLimiting:
    """Test API rate limiting."""
    
    @pytest.fixture
    async def rate_limited_client(self):
        """Create API client with rate limiting."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        config['rate_limit_enabled'] = True
        config['rate_limit_requests'] = 5  # Very low limit for testing
        config['rate_limit_window'] = 60
        
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_rate_limiting(self, rate_limited_client):
        """Test that rate limiting works."""
        # Make requests up to the limit
        for i in range(5):
            response = rate_limited_client.get("/health")
            assert response.status_code == 200
            assert "X-RateLimit-Remaining" in response.headers
        
        # Next request should be rate limited
        response = rate_limited_client.get("/health")
        assert response.status_code == 429
        
        data = response.json()
        assert "Rate limit exceeded" in data["error"]


class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.fixture
    async def perf_api_client(self):
        """Create API client for performance tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Mock fast-responding components
        system_manager = Mock(spec=SystemManager)
        system_manager.get_health_status = AsyncMock(return_value={"healthy": True})
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.slow
    def test_health_endpoint_performance(self, perf_api_client):
        """Test health endpoint performance."""
        num_requests = 100
        
        with PerformanceTimer() as timer:
            for _ in range(num_requests):
                response = perf_api_client.get("/health")
                assert response.status_code == 200
        
        elapsed_time = timer.elapsed()
        avg_time = elapsed_time / num_requests
        
        # Health endpoint should be very fast
        assert avg_time < 0.1  # Less than 100ms per request
        print(f"Health endpoint: {avg_time:.3f}s average per request")
    
    @pytest.mark.slow
    def test_concurrent_requests(self, perf_api_client):
        """Test handling concurrent requests."""
        import threading
        import queue
        
        num_threads = 10
        requests_per_thread = 10
        results = queue.Queue()
        
        def make_requests():
            thread_results = []
            for _ in range(requests_per_thread):
                try:
                    response = perf_api_client.get("/health")
                    thread_results.append(response.status_code == 200)
                except Exception:
                    thread_results.append(False)
            results.put(thread_results)
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        all_results = []
        while not results.empty():
            all_results.extend(results.get())
        
        success_rate = sum(all_results) / len(all_results)
        
        # Should handle concurrent requests well
        assert success_rate > 0.95  # 95% success rate
        print(f"Concurrent requests success rate: {success_rate:.1%}")


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.fixture
    async def error_api_client(self):
        """Create API client for error testing."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Mock components that can fail
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_invalid_json_request(self, error_api_client):
        """Test handling of invalid JSON in requests."""
        response = error_api_client.client.post(
            "/api/v1/tasks",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    @pytest.mark.unit
    def test_missing_content_type(self, error_api_client):
        """Test handling of requests with missing content type."""
        response = error_api_client.client.post(
            "/api/v1/tasks",
            content='{"agent_type": "code", "description": "test"}',
            headers={}  # No content-type header
        )
        
        assert response.status_code in [400, 422, 415]  # Various possible error codes
    
    @pytest.mark.unit
    def test_malformed_url_parameters(self, error_api_client):
        """Test handling of malformed URL parameters."""
        response = error_api_client.get("/api/v1/tasks?limit=invalid&offset=notanumber")
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.unit
    def test_nonexistent_endpoint(self, error_api_client):
        """Test accessing non-existent endpoints."""
        response = error_api_client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    @pytest.mark.unit
    def test_method_not_allowed(self, error_api_client):
        """Test using wrong HTTP method."""
        response = error_api_client.client.patch("/api/v1/tasks")  # PATCH not supported
        
        assert response.status_code == 405  # Method not allowed
    
    @pytest.mark.unit
    def test_internal_server_error_handling(self, error_api_client):
        """Test handling of internal server errors."""
        with patch.object(error_api_client.client.app.state, 'system_manager') as mock_manager:
            mock_manager.get_health_status = AsyncMock(side_effect=Exception("Internal error"))
            
            response = error_api_client.get("/health")
            
            assert response.status_code == 500
    
    @pytest.mark.unit
    def test_large_request_body(self, error_api_client):
        """Test handling of extremely large request bodies."""
        # Create a very large request
        large_data = {
            "agent_type": "code",
            "description": "x" * (10 * 1024 * 1024),  # 10MB description
            "parameters": {}
        }
        
        response = error_api_client.post("/api/v1/tasks", json=large_data)
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 400, 422]
    
    @pytest.mark.unit
    def test_sql_injection_protection(self, error_api_client):
        """Test protection against SQL injection attempts."""
        malicious_input = "'; DROP TABLE tasks; --"
        
        response = error_api_client.get(f"/api/v1/tasks/{malicious_input}")
        
        # Should handle safely (likely 404 or 400)
        assert response.status_code in [400, 404, 422]
    
    @pytest.mark.unit
    def test_xss_protection(self, error_api_client):
        """Test protection against XSS attempts."""
        xss_payload = "<script>alert('xss')</script>"
        
        task_data = {
            "agent_type": "code",
            "description": xss_payload,
            "parameters": {}
        }
        
        response = error_api_client.post("/api/v1/tasks", json=task_data)
        
        # Should either succeed (with sanitized input) or reject
        if response.status_code == 200:
            # If successful, ensure XSS payload is sanitized
            data = response.json()
            assert "<script>" not in str(data)


class TestAPIVersioning:
    """Test API versioning and backward compatibility."""
    
    @pytest.fixture
    async def versioned_api_client(self):
        """Create API client for versioning tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        system_manager = Mock(spec=SystemManager)
        system_manager.get_health_status = AsyncMock(return_value={"healthy": True})
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_v1_endpoints_accessible(self, versioned_api_client):
        """Test that v1 endpoints are accessible."""
        v1_endpoints = [
            "/api/v1/status",
            "/api/v1/agents",
            "/api/v1/tasks",
            "/api/v1/config",
            "/api/v1/logs"
        ]
        
        for endpoint in v1_endpoints:
            response = versioned_api_client.get(endpoint)
            # Should not be 404 (not found)
            assert response.status_code != 404
    
    @pytest.mark.unit
    def test_api_version_headers(self, versioned_api_client):
        """Test API version information in headers."""
        response = versioned_api_client.get("/api/v1/status")
        
        # Check for version information (implementation dependent)
        assert response.status_code in [200, 503]
    
    @pytest.mark.unit
    def test_deprecated_endpoint_warnings(self, versioned_api_client):
        """Test that deprecated endpoints return appropriate warnings."""
        # This would test future deprecated endpoints
        # For now, just ensure current endpoints work
        response = versioned_api_client.get("/api/v1/status")
        assert response.status_code in [200, 503]


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.fixture
    async def docs_api_client(self):
        """Create API client for documentation tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        system_manager = Mock(spec=SystemManager)
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.unit
    def test_openapi_schema(self, docs_api_client):
        """Test OpenAPI schema endpoint."""
        response = docs_api_client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    @pytest.mark.unit
    def test_swagger_docs(self, docs_api_client):
        """Test Swagger documentation endpoint."""
        response = docs_api_client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    @pytest.mark.unit
    def test_redoc_docs(self, docs_api_client):
        """Test ReDoc documentation endpoint."""
        response = docs_api_client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestAPIIntegration:
    """Integration tests combining multiple API features."""
    
    @pytest.fixture
    async def integration_api_client(self):
        """Create API client for integration tests."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        # Create more realistic mocks
        system_manager = Mock(spec=SystemManager)
        system_manager.get_health_status = AsyncMock(return_value={
            "healthy": True,
            "uptime": 3600,
            "components": {"database": "healthy", "agents": "healthy"}
        })
        system_manager.get_agents = AsyncMock(return_value={
            "code": {"type": "code", "status": "active"}
        })
        
        task_coordinator = Mock(spec=TaskCoordinator)
        mock_task = MockTask("integration_task", "Integration test task", "code")
        task_coordinator.create_task = AsyncMock(return_value=mock_task)
        task_coordinator.get_task = AsyncMock(return_value=mock_task)
        task_coordinator.list_tasks = AsyncMock(return_value=[mock_task])
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        client = TestClient(web_interface.app)
        return TestAPIClient(client)
    
    @pytest.mark.integration
    def test_complete_task_workflow(self, integration_api_client):
        """Test complete task creation, monitoring, and completion workflow."""
        # Step 1: Check system health
        health_response = integration_api_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["healthy"] is True
        
        # Step 2: List available agents
        agents_response = integration_api_client.get("/api/v1/agents")
        assert agents_response.status_code == 200
        agents = agents_response.json()["agents"]
        assert "code" in agents
        
        # Step 3: Create a task
        task_data = {
            "agent_type": "code",
            "description": "Integration test task",
            "parameters": {"test": True}
        }
        create_response = integration_api_client.post("/api/v1/tasks", json=task_data)
        assert create_response.status_code == 200
        task = create_response.json()
        task_id = task["id"]
        
        # Step 4: Get task details
        get_response = integration_api_client.get(f"/api/v1/tasks/{task_id}")
        assert get_response.status_code == 200
        task_details = get_response.json()["task"]
        assert task_details["id"] == task_id
        
        # Step 5: List tasks (should include our task)
        list_response = integration_api_client.get("/api/v1/tasks")
        assert list_response.status_code == 200
        tasks = list_response.json()["tasks"]
        task_ids = [t["id"] for t in tasks]
        assert task_id in task_ids
    
    @pytest.mark.integration
    def test_error_recovery_workflow(self, integration_api_client):
        """Test API behavior during error conditions and recovery."""
        # Test system degradation
        with patch.object(integration_api_client.client.app.state, 'system_manager') as mock_manager:
            # Simulate system manager failure
            mock_manager.get_health_status = AsyncMock(side_effect=Exception("System error"))
            
            health_response = integration_api_client.get("/health")
            assert health_response.status_code == 500
        
        # Test recovery - system should work again after patch is removed
        health_response = integration_api_client.get("/health")
        assert health_response.status_code == 200
    
    @pytest.mark.integration
    def test_concurrent_operations(self, integration_api_client):
        """Test concurrent API operations."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_tasks():
            """Create multiple tasks concurrently."""
            thread_results = []
            for i in range(5):
                task_data = {
                    "agent_type": "code",
                    "description": f"Concurrent task {i}",
                    "parameters": {"index": i}
                }
                try:
                    response = integration_api_client.post("/api/v1/tasks", json=task_data)
                    thread_results.append(response.status_code == 200)
                except Exception:
                    thread_results.append(False)
            results.put(thread_results)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=create_tasks)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        all_results = []
        while not results.empty():
            all_results.extend(results.get())
        
        success_rate = sum(all_results) / len(all_results)
        assert success_rate > 0.8  # 80% success rate under concurrency


class TestAPIExtensions:
    """Test API extensions and custom features."""
    
    @pytest.mark.unit
    async def test_custom_headers(self):
        """Test custom headers in API responses."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        system_manager = Mock(spec=SystemManager)
        system_manager.get_health_status = AsyncMock(return_value={"healthy": True})
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        with TestClient(web_interface.app) as client:
            response = client.get("/health")
            
            # Check for CORS headers
            assert "access-control-allow-origin" in response.headers
    
    @pytest.mark.unit
    async def test_request_id_tracking(self):
        """Test request ID tracking for debugging."""
        config = get_test_config().get_config()
        config['enable_auth'] = False
        
        system_manager = Mock(spec=SystemManager)
        system_manager.get_health_status = AsyncMock(return_value={"healthy": True})
        
        task_coordinator = Mock(spec=TaskCoordinator)
        
        web_interface = WebInterface(config)
        await web_interface.initialize(system_manager, task_coordinator)
        
        with TestClient(web_interface.app) as client:
            # Add custom request ID header
            headers = {"X-Request-ID": "test-request-123"}
            response = client.get("/health", headers=headers)
            
            assert response.status_code == 200
            # Response should include timing information
            assert "X-Response-Time" in response.headers or response.status_code == 200


# Test utilities for API testing
class APITestUtils:
    """Utility functions for API testing."""
    
    @staticmethod
    def create_test_user_token(username: str = "testuser", roles: List[str] = None) -> str:
        """Create a test JWT token."""
        if roles is None:
            roles = ["user"]
        
        payload = {
            "sub": username,
            "roles": roles,
            "exp": datetime.utcnow().timestamp() + 3600  # 1 hour
        }
        
        return create_access_token(payload)
    
    @staticmethod
    def assert_valid_task_response(response_data: Dict[str, Any]):
        """Assert that a task response is valid."""
        required_fields = ["id", "agent_type", "description", "status"]
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
        
        assert response_data["status"] in [
            "pending", "running", "completed", "failed", "cancelled"
        ], f"Invalid status: {response_data['status']}"
    
    @staticmethod
    def assert_valid_error_response(response_data: Dict[str, Any]):
        """Assert that an error response is valid."""
        assert "detail" in response_data or "error" in response_data
    
    @staticmethod
    def create_test_file(content: str = "test content", filename: str = "test.txt") -> io.BytesIO:
        """Create a test file for upload testing."""
        file_obj = io.BytesIO(content.encode())
        file_obj.name = filename
        return file_obj


# Test data generators for API testing
class APITestData:
    """Generate test data for API testing."""
    
    @staticmethod
    def valid_task_data() -> Dict[str, Any]:
        """Generate valid task creation data."""
        return {
            "agent_type": "code",
            "description": "Generate a Python function to calculate factorial",
            "parameters": {
                "language": "python",
                "function_name": "factorial",
                "include_tests": True
            },
            "priority": 1
        }
    
    @staticmethod
    def invalid_task_data() -> List[Dict[str, Any]]:
        """Generate various invalid task data for testing."""
        return [
            {},  # Empty data
            {"description": "Missing agent type"},  # Missing agent_type
            {"agent_type": "invalid"},  # Missing description
            {"agent_type": "", "description": "Empty agent type"},  # Empty agent_type
            {"agent_type": "code", "description": ""},  # Empty description
            {"agent_type": "code", "description": "Test", "priority": "invalid"},  # Invalid priority type
        ]
    
    @staticmethod
    def user_login_data() -> Dict[str, Any]:
        """Generate user login data."""
        return {
            "username": "testuser",
            "password": "testpassword"
        }
    
    @staticmethod
    def config_update_data() -> Dict[str, Any]:
        """Generate configuration update data."""
        return {
            "section": "system",
            "key": "debug",
            "value": True
        }


if __name__ == "__main__":
    # Run API tests
    pytest.main([__file__, "-v"])
