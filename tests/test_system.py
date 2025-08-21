"""
Ultra AI Project - System Tests

Comprehensive test suite for the core system components including
system manager, task coordinator, and end-to-end system integration tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from pathlib import Path

from . import (
    get_test_config, MockAgent, MockTask, TestDataGenerator,
    TestAssertions, PerformanceTimer, MemoryProfiler,
    requires_api_keys, slow_test, integration_test, e2e_test
)

# Import system classes
try:
    from src.core.system_manager import SystemManager
    from src.core.task_coordinator import TaskCoordinator, Task, TaskStatus
    from src.core.memory_manager import MemoryManager
    from src.core.security import SecurityManager
    from src.agents.base_agent import BaseAgent
    from src.utils.logger import Logger
    from src.utils.helpers import generate_id
except ImportError as e:
    pytest.skip(f"Could not import system modules: {e}", allow_module_level=True)


class TestSystemManager:
    """Test cases for SystemManager class."""
    
    @pytest.fixture
    async def system_manager(self):
        """Create a test system manager instance."""
        config = get_test_config().get_config()
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        config['data_dir'] = temp_dir
        config['log_dir'] = temp_dir
        
        system_manager = SystemManager(config)
        await system_manager.initialize()
        
        yield system_manager
        
        # Cleanup
        await system_manager.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.unit
    async def test_system_manager_initialization(self, system_manager):
        """Test system manager initialization."""
        assert system_manager.is_initialized
        assert system_manager.config is not None
        assert system_manager.logger is not None
    
    @pytest.mark.unit
    async def test_system_health_check(self, system_manager):
        """Test system health check functionality."""
        health = await system_manager.get_health_status()
        
        assert isinstance(health, dict)
        assert 'healthy' in health
        assert 'uptime' in health
        assert 'components' in health
        assert isinstance(health['components'], dict)
    
    @pytest.mark.unit
    async def test_agent_registration(self, system_manager):
        """Test agent registration and management."""
        # Create mock agent
        agent = MockAgent("test_agent", "test")
        await agent.initialize()
        
        # Register agent
        await system_manager.register_agent("test_agent", agent)
        
        # Verify registration
        agents = await system_manager.get_agents()
        assert "test_agent" in agents
        assert agents["test_agent"]["type"] == "test"
        assert agents["test_agent"]["status"] == "active"
    
    @pytest.mark.unit
    async def test_agent_unregistration(self, system_manager):
        """Test agent unregistration."""
        # Register and then unregister agent
        agent = MockAgent("temp_agent", "test")
        await agent.initialize()
        
        await system_manager.register_agent("temp_agent", agent)
        agents_before = await system_manager.get_agents()
        assert "temp_agent" in agents_before
        
        await system_manager.unregister_agent("temp_agent")
        agents_after = await system_manager.get_agents()
        assert "temp_agent" not in agents_after
    
    @pytest.mark.unit
    async def test_agent_health_monitoring(self, system_manager):
        """Test agent health monitoring."""
        # Register agent
        agent = MockAgent("health_test_agent", "test")
        await agent.initialize()
        await system_manager.register_agent("health_test_agent", agent)
        
        # Check agent health
        agent_health = await system_manager.check_agent_health("health_test_agent")
        assert agent_health["status"] == "healthy"
        assert agent_health["agent"] == "health_test_agent"
    
    @pytest.mark.unit
    async def test_configuration_management(self, system_manager):
        """Test configuration management."""
        # Get current config
        config = await system_manager.get_sanitized_config()
        assert isinstance(config, dict)
        
        # Update configuration
        await system_manager.update_config("system", "test_key", "test_value")
        
        # Verify update
        updated_config = await system_manager.get_sanitized_config()
        assert updated_config.get("system", {}).get("test_key") == "test_value"
    
    @pytest.mark.unit
    async def test_resource_monitoring(self, system_manager):
        """Test system resource monitoring."""
        resources = await system_manager.get_resource_usage()
        
        assert isinstance(resources, dict)
        expected_keys = ['cpu_percent', 'memory_percent', 'disk_percent']
        for key in expected_keys:
            assert key in resources
            assert isinstance(resources[key], (int, float))
            assert 0 <= resources[key] <= 100
    
    @pytest.mark.unit
    async def test_logging_system(self, system_manager):
        """Test logging system integration."""
        # Generate test log entries
        test_message = f"Test log message {generate_id()}"
        system_manager.logger.info(test_message)
        
        # Retrieve recent logs
        logs = await system_manager.get_recent_logs(limit=10)
        
        assert isinstance(logs, list)
        if logs:  # If logging is working
            assert any(test_message in log.get('message', '') for log in logs)
    
    @pytest.mark.unit
    async def test_component_lifecycle(self, system_manager):
        """Test component lifecycle management."""
        # Test component registration
        mock_component = Mock()
        mock_component.initialize = AsyncMock()
        mock_component.shutdown = AsyncMock()
        mock_component.get_status = AsyncMock(return_value="active")
        
        await system_manager.register_component("test_component", mock_component)
        
        # Verify component is registered
        components = await system_manager.get_components()
        assert "test_component" in components
        
        # Test component shutdown
        await system_manager.shutdown_component("test_component")
        mock_component.shutdown.assert_called_once()
    
    @pytest.mark.unit
    async def test_error_handling(self, system_manager):
        """Test system error handling."""
        # Test handling of invalid agent registration
        with pytest.raises(Exception):
            await system_manager.register_agent("invalid_agent", None)
        
        # Test handling of non-existent agent
        agent_info = await system_manager.get_agent_info("non_existent_agent")
        assert agent_info is None
    
    @pytest.mark.integration
    async def test_system_startup_shutdown_cycle(self):
        """Test complete system startup and shutdown cycle."""
        config = get_test_config().get_config()
        temp_dir = tempfile.mkdtemp()
        config['data_dir'] = temp_dir
        config['log_dir'] = temp_dir
        
        try:
            # Initialize system
            system_manager = SystemManager(config)
            await system_manager.initialize()
            
            # Verify system is running
            assert system_manager.is_initialized
            health = await system_manager.get_health_status()
            assert health['healthy']
            
            # Shutdown system
            await system_manager.shutdown()
            assert not system_manager.is_initialized
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestTaskCoordinator:
    """Test cases for TaskCoordinator class."""
    
    @pytest.fixture
    async def task_coordinator(self):
        """Create a test task coordinator instance."""
        config = get_test_config().get_config()
        
        # Create temporary directory for database
        temp_dir = tempfile.mkdtemp()
        config['database']['url'] = f"sqlite:///{temp_dir}/test.db"
        
        task_coordinator = TaskCoordinator(config)
        await task_coordinator.initialize()
        
        yield task_coordinator
        
        # Cleanup
        await task_coordinator.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def coordinator_with_agents(self, task_coordinator):
        """Create task coordinator with registered agents."""
        # Create and register mock agents
        agents = {
            "code": MockAgent("code_agent", "code"),
            "research": MockAgent("research_agent", "research"),
            "analysis": MockAgent("analysis_agent", "analysis")
        }
        
        for agent_name, agent in agents.items():
            await agent.initialize()
            await task_coordinator.register_agent(agent_name, agent)
        
        return task_coordinator, agents
    
    @pytest.mark.unit
    async def test_task_coordinator_initialization(self, task_coordinator):
        """Test task coordinator initialization."""
        assert task_coordinator.is_initialized
        assert task_coordinator.config is not None
        assert task_coordinator.task_queue is not None
    
    @pytest.mark.unit
    async def test_agent_registration_with_coordinator(self, task_coordinator):
        """Test agent registration with task coordinator."""
        agent = MockAgent("test_agent", "test")
        await agent.initialize()
        
        await task_coordinator.register_agent("test_agent", agent)
        
        agents = await task_coordinator.get_available_agents()
        assert "test_agent" in agents
    
    @pytest.mark.unit
    async def test_task_creation(self, coordinator_with_agents):
        """Test task creation functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        task = await task_coordinator.create_task(
            agent_type="code",
            description="Test task creation",
            parameters={"test": True},
            priority=1
        )
        
        TestAssertions.assert_task_valid(task)
        assert task.agent_type == "code"
        assert task.description == "Test task creation"
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.unit
    async def test_task_retrieval(self, coordinator_with_agents):
        """Test task retrieval functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create a task
        created_task = await task_coordinator.create_task(
            agent_type="code",
            description="Test task retrieval",
            parameters={}
        )
        
        # Retrieve the task
        retrieved_task = await task_coordinator.get_task(created_task.id)
        
        assert retrieved_task is not None
        assert retrieved_task.id == created_task.id
        assert retrieved_task.description == created_task.description
    
    @pytest.mark.unit
    async def test_task_listing(self, coordinator_with_agents):
        """Test task listing functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = await task_coordinator.create_task(
                agent_type="code",
                description=f"Test task {i}",
                parameters={"index": i}
            )
            tasks.append(task)
        
        # List tasks
        listed_tasks = await task_coordinator.list_tasks(limit=10)
        
        assert len(listed_tasks) >= 3
        task_ids = [task.id for task in listed_tasks]
        for created_task in tasks:
            assert created_task.id in task_ids
    
    @pytest.mark.unit
    async def test_task_filtering(self, coordinator_with_agents):
        """Test task filtering functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create tasks with different agent types
        code_task = await task_coordinator.create_task(
            agent_type="code",
            description="Code task",
            parameters={}
        )
        
        research_task = await task_coordinator.create_task(
            agent_type="research",
            description="Research task",
            parameters={}
        )
        
        # Filter by agent type
        code_tasks = await task_coordinator.list_tasks(
            agent_type="code",
            limit=10
        )
        
        assert len(code_tasks) >= 1
        assert all(task.agent_type == "code" for task in code_tasks)
    
    @pytest.mark.unit
    async def test_task_execution(self, coordinator_with_agents):
        """Test task execution workflow."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create a task
        task = await task_coordinator.create_task(
            agent_type="code",
            description="Test task execution",
            parameters={"test": True}
        )
        
        # Execute the task
        result = await task_coordinator.execute_task(task.id)
        
        assert result is not None
        assert result.get("success") is True
        
        # Verify task status updated
        updated_task = await task_coordinator.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETED
    
    @pytest.mark.unit
    async def test_task_cancellation(self, coordinator_with_agents):
        """Test task cancellation functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create a task
        task = await task_coordinator.create_task(
            agent_type="code",
            description="Test task cancellation",
            parameters={}
        )
        
        # Cancel the task
        await task_coordinator.cancel_task(task.id)
        
        # Verify task status
        cancelled_task = await task_coordinator.get_task(task.id)
        assert cancelled_task.status == TaskStatus.CANCELLED
    
    @pytest.mark.unit
    async def test_task_retry(self, coordinator_with_agents):
        """Test task retry functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create a task and mark it as failed
        original_task = await task_coordinator.create_task(
            agent_type="code",
            description="Test task retry",
            parameters={}
        )
        
        # Simulate task failure
        await task_coordinator.mark_task_failed(original_task.id, "Test failure")
        
        # Retry the task
        retry_task = await task_coordinator.retry_task(original_task.id)
        
        assert retry_task is not None
        assert retry_task.id != original_task.id
        assert retry_task.description == original_task.description
        assert retry_task.status == TaskStatus.PENDING
    
    @pytest.mark.unit
    async def test_task_priority_handling(self, coordinator_with_agents):
        """Test task priority handling."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create tasks with different priorities
        low_priority_task = await task_coordinator.create_task(
            agent_type="code",
            description="Low priority task",
            parameters={},
            priority=1
        )
        
        high_priority_task = await task_coordinator.create_task(
            agent_type="code",
            description="High priority task",
            parameters={},
            priority=10
        )
        
        # Verify tasks are created with correct priorities
        assert low_priority_task.priority == 1
        assert high_priority_task.priority == 10
    
    @pytest.mark.unit
    async def test_task_timeout_handling(self, coordinator_with_agents):
        """Test task timeout handling."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create a task with short timeout
        task = await task_coordinator.create_task(
            agent_type="code",
            description="Test timeout task",
            parameters={},
            timeout=1  # 1 second timeout
        )
        
        # Simulate long-running task by patching agent
        with patch.object(agents["code"], 'process_task', 
                         side_effect=lambda t: asyncio.sleep(2)):
            with pytest.raises(asyncio.TimeoutError):
                await task_coordinator.execute_task(task.id)
    
    @pytest.mark.unit
    async def test_task_statistics(self, coordinator_with_agents):
        """Test task statistics functionality."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create multiple tasks
        for i in range(3):
            await task_coordinator.create_task(
                agent_type="code",
                description=f"Stats test task {i}",
                parameters={}
            )
        
        # Get statistics
        stats = await task_coordinator.get_task_statistics()
        
        assert isinstance(stats, dict)
        assert 'total' in stats
        assert 'pending' in stats
        assert stats['total'] >= 3
    
    @pytest.mark.integration
    async def test_concurrent_task_processing(self, coordinator_with_agents):
        """Test concurrent task processing."""
        task_coordinator, agents = coordinator_with_agents
        
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = await task_coordinator.create_task(
                agent_type="code",
                description=f"Concurrent task {i}",
                parameters={"index": i}
            )
            tasks.append(task)
        
        # Process tasks concurrently
        results = await asyncio.gather(*[
            task_coordinator.execute_task(task.id) for task in tasks
        ], return_exceptions=True)
        
        # Verify results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful_results) >= 3  # At least 60% success rate


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create a test memory manager instance."""
        config = get_test_config().get_config()
        temp_dir = tempfile.mkdtemp()
        config['memory']['storage_path'] = temp_dir
        
        memory_manager = MemoryManager(config)
        await memory_manager.initialize()
        
        yield memory_manager
        
        # Cleanup
        await memory_manager.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.unit
    async def test_memory_storage(self, memory_manager):
        """Test memory storage functionality."""
        # Store a memory
        memory_id = await memory_manager.store_memory(
            "test_memory",
            {"content": "This is a test memory", "type": "conversation"}
        )
        
        assert memory_id is not None
        
        # Retrieve the memory
        retrieved_memory = await memory_manager.get_memory(memory_id)
        assert retrieved_memory is not None
        assert retrieved_memory["content"] == "This is a test memory"
    
    @pytest.mark.unit
    async def test_memory_search(self, memory_manager):
        """Test memory search functionality."""
        # Store multiple memories
        memory_ids = []
        for i in range(3):
            memory_id = await memory_manager.store_memory(
                f"search_test_{i}",
                {"content": f"Search test memory {i}", "category": "test"}
            )
            memory_ids.append(memory_id)
        
        # Search memories
        search_results = await memory_manager.search_memories(
            query="search test",
            limit=10
        )
        
        assert len(search_results) >= 3
    
    @pytest.mark.unit
    async def test_memory_update(self, memory_manager):
        """Test memory update functionality."""
        # Store a memory
        memory_id = await memory_manager.store_memory(
            "update_test",
            {"content": "Original content", "version": 1}
        )
        
        # Update the memory
        await memory_manager.update_memory(
            memory_id,
            {"content": "Updated content", "version": 2}
        )
        
        # Verify update
        updated_memory = await memory_manager.get_memory(memory_id)
        assert updated_memory["content"] == "Updated content"
        assert updated_memory["version"] == 2
    
    @pytest.mark.unit
    async def test_memory_deletion(self, memory_manager):
        """Test memory deletion functionality."""
        # Store a memory
        memory_id = await memory_manager.store_memory(
            "delete_test",
            {"content": "To be deleted"}
        )
        
        # Verify memory exists
        memory = await memory_manager.get_memory(memory_id)
        assert memory is not None
        
        # Delete the memory
        await memory_manager.delete_memory(memory_id)
        
        # Verify memory is deleted
        deleted_memory = await memory_manager.get_memory(memory_id)
        assert deleted_memory is None
    
    @pytest.mark.unit
    async def test_memory_categorization(self, memory_manager):
        """Test memory categorization functionality."""
        # Store memories in different categories
        categories = ["conversation", "task", "knowledge"]
        memory_ids = {}
        
        for category in categories:
            memory_id = await memory_manager.store_memory(
                f"{category}_test",
                {"content": f"Test {category} memory", "category": category}
            )
            memory_ids[category] = memory_id
        
        # Retrieve memories by category
        for category in categories:
            memories = await memory_manager.get_memories_by_category(category)
            assert len(memories) >= 1
            assert any(memory["category"] == category for memory in memories)


class TestSecurityManager:
    """Test cases for SecurityManager class."""
    
    @pytest.fixture
    async def security_manager(self):
        """Create a test security manager instance."""
        config = get_test_config().get_config()
        config['security'] = {
            'enable_auth': True,
            'secret_key': 'test-secret-key',
            'token_expire_hours': 24
        }
        
        security_manager = SecurityManager(config)
        await security_manager.initialize()
        
        yield security_manager
        
        # Cleanup
        await security_manager.shutdown()
    
    @pytest.mark.unit
    async def test_user_authentication(self, security_manager):
        """Test user authentication functionality."""
        # Create test user
        user_id = await security_manager.create_user(
            username="testuser",
            password="testpassword",
            email="test@example.com"
        )
        
        assert user_id is not None
        
        # Test authentication
        auth_result = await security_manager.authenticate_user(
            "testuser",
            "testpassword"
        )
        
        assert auth_result is not None
        assert auth_result["username"] == "testuser"
    
    @pytest.mark.unit
    async def test_token_generation(self, security_manager):
        """Test JWT token generation."""
        # Create test user
        await security_manager.create_user(
            username="tokenuser",
            password="password",
            email="token@example.com"
        )
        
        # Generate token
        token = await security_manager.create_access_token(
            username="tokenuser",
            additional_claims={"role": "user"}
        )
        
        assert token is not None
        assert isinstance(token, str)
        
        # Verify token
        claims = await security_manager.verify_token(token)
        assert claims["sub"] == "tokenuser"
        assert claims["role"] == "user"
    
    @pytest.mark.unit
    async def test_authorization(self, security_manager):
        """Test authorization functionality."""
        # Create user with specific permissions
        user_id = await security_manager.create_user(
            username="authuser",
            password="password",
            email="auth@example.com",
            permissions=["tasks:create", "tasks:read"]
        )
        
        # Test authorization
        has_permission = await security_manager.check_permission(
            user_id,
            "tasks:create"
        )
        assert has_permission is True
        
        no_permission = await security_manager.check_permission(
            user_id,
            "admin:delete"
        )
        assert no_permission is False
    
    @pytest.mark.unit
    async def test_password_hashing(self, security_manager):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Hash password
        hashed = await security_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 20  # Hashed password should be longer
        
        # Verify password
        is_valid = await security_manager.verify_password(password, hashed)
        assert is_valid is True
        
        # Test with wrong password
        is_invalid = await security_manager.verify_password("wrong_password", hashed)
        assert is_invalid is False


class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create a complete integrated system for testing."""
        config = get_test_config().get_config()
        temp_dir = tempfile.mkdtemp()
        config['data_dir'] = temp_dir
        config['log_dir'] = temp_dir
        config['database']['url'] = f"sqlite:///{temp_dir}/test.db"
        
        # Initialize system components
        system_manager = SystemManager(config)
        await system_manager.initialize()
        
        task_coordinator = TaskCoordinator(config)
        await task_coordinator.initialize()
        
        memory_manager = MemoryManager(config)
        await memory_manager.initialize()
        
        security_manager = SecurityManager(config)
        await security_manager.initialize()
        
        # Create and register mock agents
        agents = {}
        for agent_type in ["code", "research", "analysis", "creative"]:
            agent = MockAgent(f"{agent_type}_agent", agent_type)
            await agent.initialize()
            await system_manager.register_agent(f"{agent_type}_agent", agent)
            await task_coordinator.register_agent(agent_type, agent)
            agents[agent_type] = agent
        
        system = {
            'system_manager': system_manager,
            'task_coordinator': task_coordinator,
            'memory_manager': memory_manager,
            'security_manager': security_manager,
            'agents': agents,
            'config': config,
            'temp_dir': temp_dir
        }
        
        yield system
        
        # Cleanup
        await system_manager.shutdown()
        await task_coordinator.shutdown()
        await memory_manager.shutdown()
        await security_manager.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.integration
    async def test_complete_system_workflow(self, integrated_system):
        """Test complete system workflow from start to finish."""
        system = integrated_system
        task_coordinator = system['task_coordinator']
        system_manager = system['system_manager']
        
        # Step 1: Verify system health
        health = await system_manager.get_health_status()
        assert health['healthy'] is True
        
        # Step 2: Create a task
        task = await task_coordinator.create_task(
            agent_type="code",
            description="Integration test: Generate a Python function",
            parameters={"language": "python", "function": "hello_world"}
        )
        
        TestAssertions.assert_task_valid(task)
        
        # Step 3: Execute the task
        result = await task_coordinator.execute_task(task.id)
        assert result["success"] is True
        
        # Step 4: Verify task completion
        completed_task = await task_coordinator.get_task(task.id)
        assert completed_task.status == TaskStatus.COMPLETED
        
        # Step 5: Check system statistics
        stats = await task_coordinator.get_task_statistics()
        assert stats['completed'] >= 1
    
    @pytest.mark.integration
    async def test_multi_agent_collaboration(self, integrated_system):
        """Test collaboration between multiple agents."""
        system = integrated_system
        task_coordinator = system['task_coordinator']
        
        # Create tasks for different agents
        tasks = []
        agent_types = ["code", "research", "analysis", "creative"]
        
        for agent_type in agent_types:
            task = await task_coordinator.create_task(
                agent_type=agent_type,
                description=f"Collaboration test for {agent_type} agent",
                parameters={"collaboration": True}
            )
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*[
            task_coordinator.execute_task(task.id) for task in tasks
        ])
        
        # Verify all tasks completed successfully
        assert all(result["success"] for result in results)
        
        # Verify all agents participated
        for task in tasks:
            completed_task = await task_coordinator.get_task(task.id)
            assert completed_task.status == TaskStatus.COMPLETED
    
    @pytest.mark.integration
    async def test_system_error_recovery(self, integrated_system):
        """Test system error recovery capabilities."""
        system = integrated_system
        task_coordinator = system['task_coordinator']
        agents = system['agents']
        
        # Simulate agent failure
        failing_agent = agents['code']
        
        with patch.object(failing_agent, 'process_task', side_effect=Exception("Agent failure")):
            # Create task that will fail
            task = await task_coordinator.create_task(
                agent_type="code",
                description="This task will fail",
                parameters={}
            )
            
            # Execute task and expect failure
            with pytest.raises(Exception):
                await task_coordinator.execute_task(task.id)
            
            # Verify task is marked as failed
            failed_task = await task_coordinator.get_task(task.id)
            assert failed_task.status == TaskStatus.FAILED
        
        # Test recovery - agent should work again after patch is removed
        recovery_task = await task_coordinator.create_task(
            agent_type="code",
            description="Recovery test task",
            parameters={}
        )
        
        recovery_result = await task_coordinator.execute_task(recovery_task.id)
        assert recovery_result["success"] is True
    
    @pytest.mark.integration
    async def test_system_scalability(self, integrated_system):
        """Test system scalability with multiple concurrent operations."""
        system = integrated_system
        task_coordinator = system['task_coordinator']
        
        # Create many tasks concurrently
        num_tasks = 20
        task_creation_tasks = []
        
        for i in range(num_tasks):
            task_creation_tasks.append(
                task_coordinator.create_task(
                    agent_type="code",
                    description=f"Scalability test task {i}",
                    parameters={"index": i}
                )
            )
        
        # Create all tasks concurrently
        created_tasks = await asyncio.gather(*task_creation_tasks)
        assert len(created_tasks) == num_tasks
        
        # Execute tasks in batches to test coordination
        batch_size = 5
        for i in range(0, num_tasks, batch_size):
            batch = created_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                task_coordinator.execute_task(task.id) for task in batch
            ], return_exceptions=True)
            
            # At least 80% should succeed
            successful = sum(1 for r in batch_results 
                           if isinstance(r, dict) and r.get("success"))
            assert successful >= len(batch) * 0.8
    
    @pytest.mark.integration
    async def test_memory_integration(self, integrated_system):
        """Test memory integration across system components."""
        system = integrated_system
        task_coordinator = system['task_coordinator']
        memory_manager = system['memory_manager']
        
        # Create a task and store its context in memory
        task = await task_coordinator.create_task(
            agent_type="research",
            description="Research AI trends for memory test",
            parameters={"store_context": True}
        )
        
        # Store task context in memory
        memory_id = await memory_manager.store_memory(
            f"task_context_{task.id}",
            {
                "task_id": task.id,
                "description": task.description,
                "agent_type": task.agent_type,
                "context": "AI research task context"
            }
        )
        
        # Execute task
        result = await task_coordinator.execute_task(task.id)
        assert result["success"] is True
        
        # Store task result in memory
        await memory_manager.store_memory(
            f"task_result_{task.id}",
            {
                "task_id": task.id,
                "result": result,
                "completion_time": datetime.now().isoformat()
            }
        )
        
        # Retrieve and verify memory
        context_memory = await memory_manager.get_memory(memory_id)
        assert context_memory["task_id"] == task.id
        
        # Search for related memories
        related_memories = await memory_manager.search_memories(
            query=f"task_{task.id}",
            limit=10
        )
        assert len(related_memories) >= 2  # Context and result memories
    
    @pytest.mark.integration
    async def test_security_integration(self, integrated_system):
        """Test security integration across system components."""
        system = integrated_system
        security_manager = system['security_manager']
        task_coordinator = system['task_coordinator']
        
        # Create test user
        user_id = await security_manager.create_user(
            username="integration_user",
            password="test_password",
            email="integration@test.com",
            permissions=["tasks:create", "tasks:read", "tasks:execute"]
        )
        
        # Authenticate user
        auth_result = await security_manager.authenticate_user(
            "integration_user",
            "test_password"
        )
        assert auth_result is not None
        
        # Create task with user context
        task = await task_coordinator.create_task(
            agent_type="code",
            description="Security integration test task",
            parameters={"user_id": user_id},
            user_id=user_id
        )
        
        # Verify user can access their task
        has_access = await security_manager.check_resource_access(
            user_id,
            "task",
            task.id,
            "read"
        )
        assert has_access is True
        
        # Execute task with user context
        result = await task_coordinator.execute_task(task.id, user_id=user_id)
        assert result["success"] is True


class TestSystemPerformance:
    """Performance tests for the system."""
    
    @pytest.fixture
    async def performance_system(self):
        """Create a system optimized for performance testing."""
        config = get_test_config().get_config()
        temp_dir = tempfile.mkdtemp()
        config['data_dir'] = temp_dir
        config['log_dir'] = temp_dir
        config['database']['url'] = f"sqlite:///{temp_dir}/perf_test.db"
        
        # Optimize for performance
        config['system']['max_workers'] = 10
        config['database']['pool_size'] = 20
        
        system_manager = SystemManager(config)
        await system_manager.initialize()
        
        task_coordinator = TaskCoordinator(config)
        await task_coordinator.initialize()
        
        # Register fast mock agents
        for i in range(4):
            agent = MockAgent(f"perf_agent_{i}", "performance")
            await agent.initialize()
            await task_coordinator.register_agent(f"perf_agent_{i}", agent)
        
        system = {
            'system_manager': system_manager,
            'task_coordinator': task_coordinator,
            'temp_dir': temp_dir
        }
        
        yield system
        
        # Cleanup
        await system_manager.shutdown()
        await task_coordinator.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.slow
    async def test_task_creation_performance(self, performance_system):
        """Test task creation performance."""
        task_coordinator = performance_system['task_coordinator']
        
        num_tasks = 100
        
        with PerformanceTimer() as timer:
            tasks = []
            for i in range(num_tasks):
                task = await task_coordinator.create_task(
                    agent_type="perf_agent_0",
                    description=f"Performance test task {i}",
                    parameters={"index": i}
                )
                tasks.append(task)
        
        elapsed_time = timer.elapsed()
        avg_time_per_task = elapsed_time / num_tasks
        
        # Performance assertions
        assert elapsed_time < 10.0  # Should complete in under 10 seconds
        assert avg_time_per_task < 0.1  # Each task creation should take less than 100ms
        
        print(f"Created {num_tasks} tasks in {elapsed_time:.2f}s "
              f"(avg: {avg_time_per_task:.3f}s per task)")
    
    @pytest.mark.slow
    async def test_concurrent_execution_performance(self, performance_system):
        """Test concurrent task execution performance."""
        task_coordinator = performance_system['task_coordinator']
        
        # Create tasks
        num_tasks = 50
        tasks = []
        for i in range(num_tasks):
            task = await task_coordinator.create_task(
                agent_type=f"perf_agent_{i % 4}",  # Distribute across agents
                description=f"Concurrent performance test {i}",
                parameters={"index": i}
            )
            tasks.append(task)
        
        # Execute tasks concurrently
        with PerformanceTimer() as timer:
            results = await asyncio.gather(*[
                task_coordinator.execute_task(task.id) for task in tasks
            ], return_exceptions=True)
        
        elapsed_time = timer.elapsed()
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        success_rate = len(successful_results) / num_tasks
        
        # Performance assertions
        assert elapsed_time < 30.0  # Should complete in under 30 seconds
        assert success_rate > 0.9  # 90% success rate
        
        print(f"Executed {num_tasks} tasks concurrently in {elapsed_time:.2f}s "
              f"(success rate: {success_rate:.1%})")
    
    @pytest.mark.slow
    async def test_memory_usage(self, performance_system):
        """Test system memory usage during operations."""
        task_coordinator = performance_system['task_coordinator']
        
        with MemoryProfiler() as profiler:
            # Create and execute many tasks to test memory usage
            num_tasks = 100
            
            for i in range(num_tasks):
                task = await task_coordinator.create_task(
                    agent_type="perf_agent_0",
                    description=f"Memory test task {i}",
                    parameters={"data": "x" * 1000}  # Some data per task
                )
                
                await task_coordinator.execute_task(task.id)
                
                # Force garbage collection every 10 tasks
                if i % 10 == 0:
                    import gc
                    gc.collect()
        
        memory_used = profiler.memory_used()
        memory_mb = memory_used / 1024 / 1024
        
        # Memory usage should be reasonable
        assert memory_mb < 200  # Less than 200MB for this test
        
        print(f"Memory used: {memory_mb:.2f} MB for {num_tasks} tasks")
    
    @pytest.mark.slow
    async def test_database_performance(self, performance_system):
        """Test database performance under load."""
        task_coordinator = performance_system['task_coordinator']
        
        # Test database write performance
        num_operations = 1000
        
        with PerformanceTimer() as timer:
            # Create many tasks (database writes)
            for i in range(num_operations):
                await task_coordinator.create_task(
                    agent_type="perf_agent_0",
                    description=f"DB performance test {i}",
                    parameters={"index": i}
                )
        
        write_time = timer.elapsed()
        
        # Test database read performance
        with PerformanceTimer() as timer:
            # Read tasks back (database reads)
            for i in range(10):  # Sample reads
                tasks = await task_coordinator.list_tasks(
                    limit=100,
                    offset=i * 100
                )
        
        read_time = timer.elapsed()
        
        # Performance assertions
        write_ops_per_sec = num_operations / write_time
        assert write_ops_per_sec > 50  # At least 50 writes per second
        
        print(f"Database performance: {write_ops_per_sec:.1f} writes/sec, "
              f"read time: {read_time:.3f}s")


class TestSystemResilience:
    """Test system resilience and fault tolerance."""
    
    @pytest.fixture
    async def resilient_system(self):
        """Create a system for resilience testing."""
        config = get_test_config().get_config()
        temp_dir = tempfile.mkdtemp()
        config['data_dir'] = temp_dir
        config['log_dir'] = temp_dir
        config['database']['url'] = f"sqlite:///{temp_dir}/resilience_test.db"
        
        # Configure for resilience
        config['system']['retry_attempts'] = 3
        config['system']['retry_delay'] = 0.1
        config['system']['circuit_breaker'] = True
        
        system_manager = SystemManager(config)
        await system_manager.initialize()
        
        task_coordinator = TaskCoordinator(config)
        await task_coordinator.initialize()
        
        # Create both reliable and unreliable agents
        reliable_agent = MockAgent("reliable_agent", "reliable")
        await reliable_agent.initialize()
        await task_coordinator.register_agent("reliable", reliable_agent)
        
        unreliable_agent = MockAgent("unreliable_agent", "unreliable")
        await unreliable_agent.initialize()
        await task_coordinator.register_agent("unreliable", unreliable_agent)
        
        system = {
            'system_manager': system_manager,
            'task_coordinator': task_coordinator,
            'reliable_agent': reliable_agent,
            'unreliable_agent': unreliable_agent,
            'temp_dir': temp_dir
        }
        
        yield system
        
        # Cleanup
        await system_manager.shutdown()
        await task_coordinator.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.integration
    async def test_agent_failure_recovery(self, resilient_system):
        """Test recovery from agent failures."""
        task_coordinator = resilient_system['task_coordinator']
        unreliable_agent = resilient_system['unreliable_agent']
        
        # Make agent fail intermittently
        call_count = 0
        original_process_task = unreliable_agent.process_task
        
        async def failing_process_task(task):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise Exception("Simulated agent failure")
            return await original_process_task(task)
        
        with patch.object(unreliable_agent, 'process_task', side_effect=failing_process_task):
            # Create task that will initially fail
            task = await task_coordinator.create_task(
                agent_type="unreliable",
                description="Resilience test task",
                parameters={}
            )
            
            # Execute with retry logic
            result = await task_coordinator.execute_task_with_retry(task.id, max_retries=3)
            
            # Should eventually succeed
            assert result["success"] is True
            assert call_count == 3  # Failed twice, succeeded on third attempt
    
    @pytest.mark.integration
    async def test_database_connection_recovery(self, resilient_system):
        """Test recovery from database connection issues."""
        task_coordinator = resilient_system['task_coordinator']
        
        # Simulate database connection failure
        original_db_execute = task_coordinator.db_session.execute
        
        call_count = 0
        async def failing_db_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Database connection lost")
            return await original_db_execute(*args, **kwargs)
        
        with patch.object(task_coordinator.db_session, 'execute', side_effect=failing_db_execute):
            # This should trigger database reconnection
            tasks = await task_coordinator.list_tasks(limit=5)
            
            # Should work after reconnection
            assert isinstance(tasks, list)
    
    @pytest.mark.integration
    async def test_system_overload_handling(self, resilient_system):
        """Test system behavior under overload conditions."""
        task_coordinator = resilient_system['task_coordinator']
        
        # Create many tasks quickly to simulate overload
        num_tasks = 200
        
        # Limit agent processing to simulate bottleneck
        with patch.object(resilient_system['reliable_agent'], 'process_task',
                         side_effect=lambda t: asyncio.sleep(0.1)):
            
            # Create tasks rapidly
            tasks = []
            for i in range(num_tasks):
                task = await task_coordinator.create_task(
                    agent_type="reliable",
                    description=f"Overload test task {i}",
                    parameters={}
                )
                tasks.append(task)
            
            # System should handle queue gracefully
            queue_stats = await task_coordinator.get_queue_statistics()
            assert queue_stats['queued'] > 0
            assert queue_stats['queued'] <= num_tasks
    
    @pytest.mark.integration
    async def test_graceful_shutdown(self, resilient_system):
        """Test graceful system shutdown under load."""
        system_manager = resilient_system['system_manager']
        task_coordinator = resilient_system['task_coordinator']
        
        # Start some long-running tasks
        tasks = []
        for i in range(5):
            task = await task_coordinator.create_task(
                agent_type="reliable",
                description=f"Long running task {i}",
                parameters={}
            )
            tasks.append(task)
        
        # Start executing tasks
        execution_tasks = [
            asyncio.create_task(task_coordinator.execute_task(task.id))
            for task in tasks
        ]
        
        # Give tasks time to start
        await asyncio.sleep(0.1)
        
        # Initiate shutdown
        shutdown_task = asyncio.create_task(system_manager.shutdown())
        
        # Wait for shutdown to complete
        await shutdown_task
        
        # Cancel remaining execution tasks
        for exec_task in execution_tasks:
            exec_task.cancel()
        
        # System should be shut down cleanly
        assert not system_manager.is_initialized


class TestSystemConfiguration:
    """Test system configuration and customization."""
    
    @pytest.mark.unit
    async def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = get_test_config().get_config()
        system_manager = SystemManager(valid_config)
        
        # Should initialize without errors
        await system_manager.initialize()
        await system_manager.shutdown()
        
        # Test invalid configuration
        invalid_config = {"invalid": "config"}
        
        with pytest.raises(Exception):
            invalid_system = SystemManager(invalid_config)
            await invalid_system.initialize()
    
    @pytest.mark.unit
    async def test_dynamic_configuration_update(self):
        """Test dynamic configuration updates."""
        config = get_test_config().get_config()
        temp_dir = tempfile.mkdtemp()
        config['data_dir'] = temp_dir
        
        try:
            system_manager = SystemManager(config)
            await system_manager.initialize()
            
            # Update configuration
            await system_manager.update_config("system", "max_workers", 8)
            
            # Verify update
            updated_config = await system_manager.get_config()
            assert updated_config["system"]["max_workers"] == 8
            
            await system_manager.shutdown()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.unit
    async def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        base_config = get_test_config().get_config()
        
        # Test development environment
        dev_config = base_config.copy()
        dev_config['environment'] = 'development'
        dev_config['debug'] = True
        
        dev_system = SystemManager(dev_config)
        await dev_system.initialize()
        assert dev_system.config['debug'] is True
        await dev_system.shutdown()
        
        # Test production environment
        prod_config = base_config.copy()
        prod_config['environment'] = 'production'
        prod_config['debug'] = False
        
        prod_system = SystemManager(prod_config)
        await prod_system.initialize()
        assert prod_system.config['debug'] is False
        await prod_system.shutdown()


# Test data generators for system testing
class SystemTestData:
    """Generate test data for system testing."""
    
    @staticmethod
    def generate_test_tasks(count: int = 10, agent_types: List[str] = None) -> List[Dict[str, Any]]:
        """Generate test task data."""
        if agent_types is None:
            agent_types = ["code", "research", "analysis", "creative"]
        
        tasks = []
        for i in range(count):
            agent_type = agent_types[i % len(agent_types)]
            task_data = {
                "agent_type": agent_type,
                "description": f"Test task {i} for {agent_type} agent",
                "parameters": {
                    "index": i,
                    "test_data": f"data_{i}",
                    "complexity": "medium"
                },
                "priority": (i % 5) + 1
            }
            tasks.append(task_data)
        
        return tasks
    
    @staticmethod
    def generate_stress_test_config() -> Dict[str, Any]:
        """Generate configuration for stress testing."""
        config = get_test_config().get_config()
        config.update({
            'system': {
                'max_workers': 20,
                'queue_size': 1000,
                'timeout': 300
            },
            'database': {
                'pool_size': 50,
                'max_overflow': 100
            },
            'agents': {
                'batch_size': 10,
                'concurrent_limit': 50
            }
        })
        return config


if __name__ == "__main__":
    # Run system tests
    pytest.main([__file__, "-v"])
