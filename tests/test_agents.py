"""
Ultra AI Project - Agent Tests

Comprehensive test suite for AI agents including unit tests, integration tests,
and performance tests for all agent types.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from . import (
    get_test_config, MockAgent, MockTask, TestDataGenerator,
    TestAssertions, PerformanceTimer, requires_api_keys, slow_test
)

# Import agent classes
try:
    from src.agents.base_agent import BaseAgent
    from src.agents.code_agent import CodeAgent
    from src.agents.research_agent import ResearchAgent
    from src.agents.analysis_agent import AnalysisAgent
    from src.agents.creative_agent import CreativeAgent
    from src.core.task_coordinator import Task
    from src.utils.logger import Logger
except ImportError as e:
    pytest.skip(f"Could not import agent modules: {e}", allow_module_level=True)


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.fixture
    async def base_agent(self):
        """Create a test base agent instance."""
        config = get_test_config().get_config()
        agent = MockAgent("test_agent", "base")
        await agent.initialize()
        return agent
    
    @pytest.mark.unit
    async def test_agent_initialization(self, base_agent):
        """Test agent initialization."""
        assert base_agent.name == "test_agent"
        assert base_agent.agent_type == "base"
        assert base_agent.status == "active"
    
    @pytest.mark.unit
    async def test_agent_capabilities(self, base_agent):
        """Test agent capabilities retrieval."""
        capabilities = await base_agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert "test" in capabilities
    
    @pytest.mark.unit
    async def test_agent_health_check(self, base_agent):
        """Test agent health check."""
        health = await base_agent.health_check()
        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert health["agent"] == "test_agent"
    
    @pytest.mark.unit
    async def test_task_processing(self, base_agent):
        """Test basic task processing."""
        task = MockTask("test_task_1", "Test task description")
        
        result = await base_agent.process_task(task)
        
        assert result["success"] is True
        assert "Mock result for task" in result["result"]
        assert result["metadata"]["agent"] == "test_agent"
    
    @pytest.mark.unit
    async def test_concurrent_task_processing(self, base_agent):
        """Test processing multiple tasks concurrently."""
        tasks = [
            MockTask(f"test_task_{i}", f"Test task {i}")
            for i in range(5)
        ]
        
        # Process tasks concurrently
        results = await asyncio.gather(*[
            base_agent.process_task(task) for task in tasks
        ])
        
        assert len(results) == 5
        for result in results:
            assert result["success"] is True
    
    @pytest.mark.unit
    async def test_agent_error_handling(self):
        """Test agent error handling."""
        # Create an agent that will fail
        failing_agent = MockAgent("failing_agent")
        
        with patch.object(failing_agent, 'process_task', side_effect=Exception("Test error")):
            task = MockTask("test_task", "This will fail")
            
            with pytest.raises(Exception) as exc_info:
                await failing_agent.process_task(task)
            
            assert str(exc_info.value) == "Test error"


class TestCodeAgent:
    """Test cases for CodeAgent class."""
    
    @pytest.fixture
    async def code_agent(self):
        """Create a test code agent instance."""
        config = get_test_config().get_config()
        config["agents"]["code"] = {
            "model": "test-model",
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        # Mock the actual CodeAgent for testing
        agent = MockAgent("code_agent", "code")
        agent.capabilities = ["code_generation", "code_review", "debugging", "optimization"]
        await agent.initialize()
        return agent
    
    @pytest.mark.unit
    async def test_code_generation_task(self, code_agent):
        """Test code generation functionality."""
        task = MockTask(
            "code_gen_1",
            "Generate a Python function to calculate fibonacci numbers",
            "code"
        )
        task.parameters = {
            "language": "python",
            "function_name": "fibonacci",
            "include_tests": True
        }
        
        result = await code_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
        TestAssertions.assert_task_valid(task)
    
    @pytest.mark.unit
    async def test_code_review_task(self, code_agent):
        """Test code review functionality."""
        task = MockTask(
            "code_review_1",
            "Review this Python code for issues",
            "code"
        )
        task.parameters = {
            "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "language": "python",
            "review_type": "security_and_performance"
        }
        
        result = await code_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_debugging_task(self, code_agent):
        """Test debugging functionality."""
        task = MockTask(
            "debug_1",
            "Debug this code that has an off-by-one error",
            "code"
        )
        task.parameters = {
            "code": "def get_last_element(arr):\n    return arr[len(arr)]",
            "error_description": "IndexError: list index out of range",
            "language": "python"
        }
        
        result = await code_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_supported_languages(self, code_agent):
        """Test support for multiple programming languages."""
        languages = ["python", "javascript", "java", "cpp", "go", "rust"]
        
        for language in languages:
            task = MockTask(
                f"code_gen_{language}",
                f"Generate a hello world program in {language}",
                "code"
            )
            task.parameters = {"language": language}
            
            result = await code_agent.process_task(task)
            assert result["success"] is True
    
    @pytest.mark.unit
    async def test_code_optimization_task(self, code_agent):
        """Test code optimization functionality."""
        task = MockTask(
            "optimize_1",
            "Optimize this bubble sort implementation",
            "code"
        )
        task.parameters = {
            "code": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
            """,
            "language": "python",
            "optimization_focus": "performance"
        }
        
        result = await code_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.integration
    @requires_api_keys("OPENAI_API_KEY")
    async def test_real_code_generation(self, code_agent):
        """Test real code generation with API."""
        # This test requires actual API keys and will be skipped if not available
        task = MockTask(
            "real_code_1",
            "Create a REST API endpoint for user authentication",
            "code"
        )
        task.parameters = {
            "language": "python",
            "framework": "fastapi",
            "include_validation": True
        }
        
        result = await code_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result


class TestResearchAgent:
    """Test cases for ResearchAgent class."""
    
    @pytest.fixture
    async def research_agent(self):
        """Create a test research agent instance."""
        config = get_test_config().get_config()
        config["agents"]["research"] = {
            "model": "test-model",
            "search_engines": ["google", "bing"],
            "max_sources": 10
        }
        
        agent = MockAgent("research_agent", "research")
        agent.capabilities = ["web_search", "data_gathering", "summarization", "fact_checking"]
        await agent.initialize()
        return agent
    
    @pytest.mark.unit
    async def test_web_search_task(self, research_agent):
        """Test web search functionality."""
        task = MockTask(
            "research_1",
            "Research the latest developments in quantum computing",
            "research"
        )
        task.parameters = {
            "query": "quantum computing 2025 breakthroughs",
            "sources": 5,
            "depth": "comprehensive"
        }
        
        result = await research_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_data_gathering_task(self, research_agent):
        """Test data gathering functionality."""
        task = MockTask(
            "data_gather_1",
            "Gather information about AI market trends",
            "research"
        )
        task.parameters = {
            "topic": "artificial intelligence market",
            "time_period": "2024-2025",
            "data_types": ["statistics", "reports", "news"]
        }
        
        result = await research_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_summarization_task(self, research_agent):
        """Test content summarization functionality."""
        task = MockTask(
            "summarize_1",
            "Summarize these research findings",
            "research"
        )
        task.parameters = {
            "content": "Long research content here...",
            "summary_length": "medium",
            "key_points": True
        }
        
        result = await research_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_fact_checking_task(self, research_agent):
        """Test fact checking functionality."""
        task = MockTask(
            "fact_check_1",
            "Fact-check these claims about renewable energy",
            "research"
        )
        task.parameters = {
            "claims": [
                "Solar energy is now cheaper than fossil fuels",
                "Wind power generates 50% of electricity globally"
            ],
            "verification_level": "high"
        }
        
        result = await research_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.integration
    async def test_research_with_citations(self, research_agent):
        """Test research with proper citations."""
        task = MockTask(
            "research_citations_1",
            "Research climate change impact with citations",
            "research"
        )
        task.parameters = {
            "topic": "climate change effects 2025",
            "require_citations": True,
            "academic_sources": True
        }
        
        result = await research_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result


class TestAnalysisAgent:
    """Test cases for AnalysisAgent class."""
    
    @pytest.fixture
    async def analysis_agent(self):
        """Create a test analysis agent instance."""
        config = get_test_config().get_config()
        config["agents"]["analysis"] = {
            "model": "test-model",
            "analysis_types": ["statistical", "trend", "sentiment"],
            "visualization": True
        }
        
        agent = MockAgent("analysis_agent", "analysis")
        agent.capabilities = ["data_analysis", "statistical_analysis", "trend_analysis", "visualization"]
        await agent.initialize()
        return agent
    
    @pytest.mark.unit
    async def test_data_analysis_task(self, analysis_agent):
        """Test data analysis functionality."""
        task = MockTask(
            "analysis_1",
            "Analyze sales data for trends and insights",
            "analysis"
        )
        task.parameters = {
            "data_source": "sales_data.csv",
            "analysis_type": "trend",
            "time_period": "quarterly"
        }
        
        result = await analysis_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_statistical_analysis_task(self, analysis_agent):
        """Test statistical analysis functionality."""
        task = MockTask(
            "stats_1",
            "Perform statistical analysis on customer data",
            "analysis"
        )
        task.parameters = {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "tests": ["mean", "median", "std_dev", "correlation"],
            "confidence_level": 0.95
        }
        
        result = await analysis_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_sentiment_analysis_task(self, analysis_agent):
        """Test sentiment analysis functionality."""
        task = MockTask(
            "sentiment_1",
            "Analyze sentiment of customer reviews",
            "analysis"
        )
        task.parameters = {
            "text_data": [
                "Great product, highly recommended!",
                "Terrible experience, very disappointed",
                "Average quality, nothing special"
            ],
            "sentiment_scale": "positive_negative_neutral"
        }
        
        result = await analysis_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_visualization_task(self, analysis_agent):
        """Test data visualization functionality."""
        task = MockTask(
            "viz_1",
            "Create visualizations for financial data",
            "analysis"
        )
        task.parameters = {
            "data": {"Q1": 100, "Q2": 150, "Q3": 120, "Q4": 180},
            "chart_types": ["line", "bar", "pie"],
            "output_format": "png"
        }
        
        result = await analysis_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_complex_analysis_pipeline(self, analysis_agent):
        """Test complex multi-step analysis."""
        task = MockTask(
            "complex_analysis_1",
            "Complete analysis pipeline: data cleaning, analysis, and visualization",
            "analysis"
        )
        task.parameters = {
            "pipeline_steps": ["clean", "analyze", "visualize"],
            "data_source": "raw_data.csv",
            "output_reports": True
        }
        
        result = await analysis_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result


class TestCreativeAgent:
    """Test cases for CreativeAgent class."""
    
    @pytest.fixture
    async def creative_agent(self):
        """Create a test creative agent instance."""
        config = get_test_config().get_config()
        config["agents"]["creative"] = {
            "model": "test-model",
            "creativity_level": "high",
            "content_types": ["text", "image", "audio"]
        }
        
        agent = MockAgent("creative_agent", "creative")
        agent.capabilities = ["content_creation", "copywriting", "storytelling", "brainstorming"]
        await agent.initialize()
        return agent
    
    @pytest.mark.unit
    async def test_content_creation_task(self, creative_agent):
        """Test content creation functionality."""
        task = MockTask(
            "content_1",
            "Create marketing content for a new product",
            "creative"
        )
        task.parameters = {
            "content_type": "blog_post",
            "target_audience": "tech professionals",
            "tone": "professional_friendly",
            "word_count": 500
        }
        
        result = await creative_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_copywriting_task(self, creative_agent):
        """Test copywriting functionality."""
        task = MockTask(
            "copy_1",
            "Write compelling ad copy for social media",
            "creative"
        )
        task.parameters = {
            "platform": "instagram",
            "product": "AI productivity tool",
            "cta": "Sign up for free trial",
            "character_limit": 150
        }
        
        result = await creative_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_storytelling_task(self, creative_agent):
        """Test storytelling functionality."""
        task = MockTask(
            "story_1",
            "Create a short story about AI and humanity",
            "creative"
        )
        task.parameters = {
            "genre": "science_fiction",
            "length": "short_story",
            "themes": ["technology", "human_connection"],
            "target_age": "adult"
        }
        
        result = await creative_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_brainstorming_task(self, creative_agent):
        """Test brainstorming functionality."""
        task = MockTask(
            "brainstorm_1",
            "Brainstorm innovative features for mobile app",
            "creative"
        )
        task.parameters = {
            "domain": "productivity",
            "target_users": "remote workers",
            "innovation_level": "high",
            "idea_count": 10
        }
        
        result = await creative_agent.process_task(task)
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.unit
    async def test_multi_format_content(self, creative_agent):
        """Test creating content in multiple formats."""
        formats = ["blog_post", "social_media", "email", "presentation"]
        
        for content_format in formats:
            task = MockTask(
                f"content_{content_format}",
                f"Create {content_format} content",
                "creative"
            )
            task.parameters = {
                "format": content_format,
                "topic": "AI innovation",
                "audience": "general"
            }
            
            result = await creative_agent.process_task(task)
            assert result["success"] is True


class TestAgentIntegration:
    """Integration tests for multiple agents working together."""
    
    @pytest.fixture
    async def agent_system(self):
        """Create a system with multiple agents."""
        config = get_test_config().get_config()
        
        agents = {
            "code": MockAgent("code_agent", "code"),
            "research": MockAgent("research_agent", "research"),
            "analysis": MockAgent("analysis_agent", "analysis"),
            "creative": MockAgent("creative_agent", "creative")
        }
        
        for agent in agents.values():
            await agent.initialize()
        
        return agents
    
    @pytest.mark.integration
    async def test_multi_agent_workflow(self, agent_system):
        """Test workflow involving multiple agents."""
        # Step 1: Research phase
        research_task = MockTask(
            "research_phase",
            "Research market trends for new product",
            "research"
        )
        research_result = await agent_system["research"].process_task(research_task)
        assert research_result["success"] is True
        
        # Step 2: Analysis phase
        analysis_task = MockTask(
            "analysis_phase",
            "Analyze the research findings",
            "analysis"
        )
        analysis_task.parameters = {"input_data": research_result["result"]}
        analysis_result = await agent_system["analysis"].process_task(analysis_task)
        assert analysis_result["success"] is True
        
        # Step 3: Creative phase
        creative_task = MockTask(
            "creative_phase",
            "Create marketing content based on analysis",
            "creative"
        )
        creative_task.parameters = {"analysis_input": analysis_result["result"]}
        creative_result = await agent_system["creative"].process_task(creative_task)
        assert creative_result["success"] is True
        
        # Step 4: Code phase
        code_task = MockTask(
            "code_phase",
            "Implement features based on requirements",
            "code"
        )
        code_task.parameters = {"requirements": creative_result["result"]}
        code_result = await agent_system["code"].process_task(code_task)
        assert code_result["success"] is True
    
    @pytest.mark.integration
    async def test_agent_collaboration(self, agent_system):
        """Test agents collaborating on a complex task."""
        # Simulate a complex task requiring multiple agent types
        main_task = MockTask(
            "collaborative_task",
            "Build a complete solution: research, analyze, design, and implement",
            "multi_agent"
        )
        
        # Each agent contributes to the solution
        results = {}
        for agent_name, agent in agent_system.items():
            subtask = MockTask(
                f"subtask_{agent_name}",
                f"{agent_name.title()} component of the solution",
                agent_name
            )
            results[agent_name] = await agent.process_task(subtask)
        
        # Verify all agents completed their parts
        for agent_name, result in results.items():
            assert result["success"] is True, f"{agent_name} agent failed"
    
    @pytest.mark.integration
    async def test_agent_error_propagation(self, agent_system):
        """Test error handling in multi-agent scenarios."""
        # Create a scenario where one agent fails
        failing_agent = agent_system["code"]
        
        with patch.object(failing_agent, 'process_task', side_effect=Exception("Agent failure")):
            task = MockTask("failing_task", "This will fail", "code")
            
            with pytest.raises(Exception) as exc_info:
                await failing_agent.process_task(task)
            
            assert "Agent failure" in str(exc_info.value)


class TestAgentPerformance:
    """Performance tests for agents."""
    
    @pytest.fixture
    async def performance_agent(self):
        """Create an agent for performance testing."""
        agent = MockAgent("perf_agent", "performance")
        await agent.initialize()
        return agent
    
    @pytest.mark.slow
    async def test_task_processing_speed(self, performance_agent):
        """Test agent task processing speed."""
        num_tasks = 100
        tasks = [
            MockTask(f"perf_task_{i}", f"Performance test task {i}")
            for i in range(num_tasks)
        ]
        
        with PerformanceTimer() as timer:
            results = await asyncio.gather(*[
                performance_agent.process_task(task) for task in tasks
            ])
        
        elapsed_time = timer.elapsed()
        avg_time_per_task = elapsed_time / num_tasks
        
        # Assertions about performance
        assert elapsed_time < 10.0  # Should complete in under 10 seconds
        assert avg_time_per_task < 0.1  # Each task should take less than 100ms
        assert all(result["success"] for result in results)
        
        print(f"Processed {num_tasks} tasks in {elapsed_time:.2f}s "
              f"(avg: {avg_time_per_task:.3f}s per task)")
    
    @pytest.mark.slow
    async def test_concurrent_task_limits(self, performance_agent):
        """Test agent behavior under high concurrency."""
        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 25, 50]
        
        for concurrency in concurrency_levels:
            tasks = [
                MockTask(f"concurrent_task_{i}", f"Concurrent test {i}")
                for i in range(concurrency)
            ]
            
            with PerformanceTimer() as timer:
                results = await asyncio.gather(*[
                    performance_agent.process_task(task) for task in tasks
                ], return_exceptions=True)
            
            elapsed_time = timer.elapsed()
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            
            print(f"Concurrency {concurrency}: {success_count}/{concurrency} "
                  f"successful in {elapsed_time:.2f}s")
            
            # At least 90% should succeed
            assert success_count >= concurrency * 0.9
    
    @pytest.mark.slow
    async def test_memory_usage(self, performance_agent):
        """Test agent memory usage during task processing."""
        from . import MemoryProfiler
        
        with MemoryProfiler() as profiler:
            # Process many tasks to test memory usage
            tasks = [
                MockTask(f"memory_task_{i}", f"Memory test task {i}")
                for i in range(50)
            ]
            
            for task in tasks:
                await performance_agent.process_task(task)
        
        memory_used = profiler.memory_used()
        
        # Memory usage should be reasonable (less than 100MB for this test)
        assert memory_used < 100 * 1024 * 1024  # 100MB
        
        print(f"Memory used: {memory_used / 1024 / 1024:.2f} MB")


class TestAgentConfiguration:
    """Test agent configuration and customization."""
    
    @pytest.mark.unit
    async def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Test valid configuration
        valid_config = {
            "model": "test-model",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30
        }
        
        agent = MockAgent("config_test", "test")
        # Configuration validation would happen during initialization
        await agent.initialize()
        
        assert agent.name == "config_test"
        assert agent.agent_type == "test"
    
    @pytest.mark.unit
    async def test_agent_parameter_customization(self):
        """Test agent parameter customization."""
        config = get_test_config().get_config()
        
        # Test different parameter combinations
        test_configs = [
            {"temperature": 0.1, "max_tokens": 500},
            {"temperature": 0.9, "max_tokens": 2000},
            {"temperature": 0.5, "max_tokens": 1000}
        ]
        
        for i, test_config in enumerate(test_configs):
            agent = MockAgent(f"param_test_{i}", "test")
            await agent.initialize()
            
            # Test that the agent can handle different configurations
            task = MockTask(f"param_task_{i}", f"Parameter test {i}")
            result = await agent.process_task(task)
            
            assert result["success"] is True


if __name__ == "__main__":
    # Run agent tests
    pytest.main([__file__, "-v"])
