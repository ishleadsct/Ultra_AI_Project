"""
Ultra AI Project - Test Suite

This module contains comprehensive tests for the Ultra AI system including
unit tests, integration tests, and end-to-end testing scenarios.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Add src directory to Python path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_CONFIG = {
    "test_mode": True,
    "log_level": "DEBUG",
    "database": {
        "url": "sqlite:///:memory:",
        "echo": False
    },
    "system": {
        "timeout": 30,
        "max_workers": 2
    },
    "agents": {
        "enabled": ["code", "research", "analysis", "creative"]
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8001,
        "enable_auth": False
    }
}

# Test fixtures and utilities
class TestConfig:
    """Test configuration class."""
    
    def __init__(self):
        self.config = TEST_CONFIG.copy()
        self.temp_dir = None
        self.test_files = []
    
    def get_config(self):
        """Get test configuration."""
        return self.config
    
    def set_temp_dir(self, temp_dir: Path):
        """Set temporary directory for tests."""
        self.temp_dir = temp_dir
        self.config["temp_dir"] = str(temp_dir)
        self.config["data_dir"] = str(temp_dir / "data")
        self.config["log_dir"] = str(temp_dir / "logs")
    
    def add_test_file(self, file_path: str):
        """Add test file to cleanup list."""
        self.test_files.append(file_path)
    
    def cleanup(self):
        """Clean up test files."""
        for file_path in self.test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
        self.test_files.clear()


# Global test configuration instance
test_config = TestConfig()


def get_test_config():
    """Get the global test configuration."""
    return test_config


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# Test utilities
class AsyncTestClient:
    """Async test client for HTTP testing."""
    
    def __init__(self, app):
        self.app = app
    
    async def request(self, method: str, url: str, **kwargs):
        """Make async HTTP request."""
        # Implementation would depend on your HTTP client library
        pass


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name: str, agent_type: str = "test"):
        self.name = name
        self.agent_type = agent_type
        self.status = "active"
        self.capabilities = ["test"]
    
    async def initialize(self):
        """Initialize mock agent."""
        pass
    
    async def process_task(self, task):
        """Process a mock task."""
        return {
            "success": True,
            "result": f"Mock result for task: {task.description}",
            "metadata": {
                "agent": self.name,
                "type": self.agent_type
            }
        }
    
    async def get_capabilities(self):
        """Get agent capabilities."""
        return self.capabilities
    
    async def health_check(self):
        """Check agent health."""
        return {"status": "healthy", "agent": self.name}


class MockTask:
    """Mock task for testing."""
    
    def __init__(self, task_id: str, description: str, agent_type: str = "test"):
        self.id = task_id
        self.description = description
        self.agent_type = agent_type
        self.status = "pending"
        self.created_at = None
        self.updated_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.progress = 0.0
        self.parameters = {}
    
    def to_dict(self):
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "agent_type": self.agent_type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "parameters": self.parameters
        }


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_task_data(count: int = 1, agent_type: str = "test"):
        """Generate test task data."""
        tasks = []
        for i in range(count):
            task = MockTask(
                task_id=f"test_task_{i}",
                description=f"Test task {i}",
                agent_type=agent_type
            )
            tasks.append(task)
        return tasks
    
    @staticmethod
    def generate_user_data(count: int = 1):
        """Generate test user data."""
        users = []
        for i in range(count):
            user = {
                "id": f"user_{i}",
                "username": f"testuser{i}",
                "email": f"test{i}@example.com",
                "full_name": f"Test User {i}",
                "roles": ["user"],
                "created_at": "2025-01-01T00:00:00Z"
            }
            users.append(user)
        return users
    
    @staticmethod
    def generate_config_data():
        """Generate test configuration data."""
        return {
            "system": {
                "name": "Test Ultra AI",
                "version": "1.0.0-test",
                "debug": True
            },
            "agents": {
                "code": {"enabled": True, "model": "test-model"},
                "research": {"enabled": True, "model": "test-model"},
                "analysis": {"enabled": True, "model": "test-model"},
                "creative": {"enabled": True, "model": "test-model"}
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8001,
                "enable_auth": False
            }
        }


class TestAssertions:
    """Custom assertions for testing."""
    
    @staticmethod
    def assert_task_valid(task):
        """Assert that a task object is valid."""
        assert task is not None
        assert hasattr(task, 'id')
        assert hasattr(task, 'description')
        assert hasattr(task, 'agent_type')
        assert hasattr(task, 'status')
        assert task.status in ['pending', 'running', 'completed', 'failed', 'cancelled']
    
    @staticmethod
    def assert_response_valid(response, expected_status: int = 200):
        """Assert that an HTTP response is valid."""
        assert response is not None
        assert hasattr(response, 'status_code')
        assert response.status_code == expected_status
    
    @staticmethod
    def assert_agent_valid(agent):
        """Assert that an agent object is valid."""
        assert agent is not None
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'agent_type')
        assert hasattr(agent, 'status')
        assert agent.status in ['active', 'inactive', 'error']
    
    @staticmethod
    def assert_config_valid(config):
        """Assert that a configuration object is valid."""
        assert config is not None
        assert isinstance(config, dict)
        assert 'system' in config
        assert 'agents' in config


class TestFileManager:
    """Manage test files and cleanup."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.created_files = []
        self.created_dirs = []
    
    def create_test_file(self, filename: str, content: str = "test content"):
        """Create a test file."""
        file_path = self.temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.created_files.append(file_path)
        return file_path
    
    def create_test_directory(self, dirname: str):
        """Create a test directory."""
        dir_path = self.temp_dir / dirname
        dir_path.mkdir(parents=True, exist_ok=True)
        
        self.created_dirs.append(dir_path)
        return dir_path
    
    def cleanup(self):
        """Clean up created files and directories."""
        # Remove files
        for file_path in self.created_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
        
        # Remove directories
        for dir_path in reversed(self.created_dirs):
            try:
                if dir_path.exists():
                    dir_path.rmdir()
            except Exception:
                pass
        
        self.created_files.clear()
        self.created_dirs.clear()


# Test decorators
def requires_api_keys(*keys):
    """Decorator to skip tests that require API keys."""
    def decorator(func):
        missing_keys = [key for key in keys if not os.getenv(key)]
        if missing_keys:
            return pytest.mark.skip(
                reason=f"Missing required API keys: {', '.join(missing_keys)}"
            )(func)
        return func
    return decorator


def slow_test(func):
    """Decorator to mark slow tests."""
    return pytest.mark.slow(func)


def integration_test(func):
    """Decorator to mark integration tests."""
    return pytest.mark.integration(func)


def e2e_test(func):
    """Decorator to mark end-to-end tests."""
    return pytest.mark.e2e(func)


# Test runners
def run_unit_tests():
    """Run only unit tests."""
    pytest.main(["-m", "unit", "-v"])


def run_integration_tests():
    """Run only integration tests."""
    pytest.main(["-m", "integration", "-v"])


def run_e2e_tests():
    """Run only end-to-end tests."""
    pytest.main(["-m", "e2e", "-v"])


def run_all_tests():
    """Run all tests."""
    pytest.main(["-v"])


def run_fast_tests():
    """Run only fast tests (exclude slow tests)."""
    pytest.main(["-m", "not slow", "-v"])


# Performance testing utilities
class PerformanceTimer:
    """Timer for performance testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        import time
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer."""
        import time
        self.end_time = time.time()
    
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MemoryProfiler:
    """Memory profiler for testing."""
    
    def __init__(self):
        self.start_memory = None
        self.end_memory = None
    
    def start(self):
        """Start memory profiling."""
        import psutil
        process = psutil.Process()
        self.start_memory = process.memory_info().rss
    
    def stop(self):
        """Stop memory profiling."""
        import psutil
        process = psutil.Process()
        self.end_memory = process.memory_info().rss
    
    def memory_used(self):
        """Get memory used in bytes."""
        if self.start_memory is None or self.end_memory is None:
            return None
        return self.end_memory - self.start_memory
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Export test utilities
__all__ = [
    'TEST_CONFIG',
    'TestConfig',
    'get_test_config',
    'AsyncTestClient',
    'MockAgent',
    'MockTask',
    'TestDataGenerator',
    'TestAssertions',
    'TestFileManager',
    'PerformanceTimer',
    'MemoryProfiler',
    'requires_api_keys',
    'slow_test',
    'integration_test',
    'e2e_test',
    'run_unit_tests',
    'run_integration_tests',
    'run_e2e_tests',
    'run_all_tests',
    'run_fast_tests'
]

# Test suite version
__version__ = '1.0.0'
__author__ = 'Ultra AI Team'
__description__ = 'Comprehensive test suite for Ultra AI system'
