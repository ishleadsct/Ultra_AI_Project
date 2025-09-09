"""
Ultra AI Project - Model Manager

Central management system for AI models, including routing, load balancing,
performance monitoring, cost tracking, and health management across providers.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import load_config, retry_with_backoff
from .types import (
    ModelInfo, ModelUsage, ModelStatus, StandardResponse, RoutingStrategy,
    create_success_response, create_error_response,
    DEFAULT_MODEL_CONFIG, OPENAI_PROVIDER, ANTHROPIC_PROVIDER
)

logger = get_logger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_streaming: bool = False
    enable_caching: bool = True
    cache_ttl: int = 3600
    cost_per_1k_tokens: Optional[Dict[str, float]] = None
    capabilities: List[str] = field(default_factory=list)
    context_window: int = 4096
    priority: int = 1  # Higher priority = preferred
    max_requests_per_minute: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ModelMetrics:
    """Model performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    requests_per_minute: float = 0.0
    uptime_percentage: float = 100.0
    
    def update(self, success: bool, tokens: int, cost: float, response_time: float):
        """Update metrics with new request data."""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.last_request_time = datetime.now()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update averages
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time) / self.total_requests
        )

class CircuitBreaker:
    """Circuit breaker for model failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_request(self) -> bool:
        """Check if requests are allowed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)):
                self.state = "half-open"
                return True
            return False
        
        # half-open state
        return True

class ModelRouter:
    """Intelligent model routing and load balancing."""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.BALANCED):
        self.strategy = strategy
        self.model_weights: Dict[str, float] = {}
        self.last_used_index = 0
    
    def select_model(self, available_models: List[ModelConfig], 
                    metrics: Dict[str, ModelMetrics],
                    request_context: Optional[Dict[str, Any]] = None) -> Optional[ModelConfig]:
        """Select the best model based on routing strategy."""
        if not available_models:
            return None
        
        if len(available_models) == 1:
            return available_models[0]
        
        if self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._select_by_cost(available_models)
        elif self.strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_by_performance(available_models, metrics)
        elif self.strategy == RoutingStrategy.BALANCED:
            return self._select_balanced(available_models, metrics)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_models)
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available_models, metrics)
        else:
            return available_models[0]
    
    def _select_by_cost(self, models: List[ModelConfig]) -> ModelConfig:
        """Select model with lowest cost."""
        def get_cost(model: ModelConfig) -> float:
            if not model.cost_per_1k_tokens:
                return float('inf')
            return model.cost_per_1k_tokens.get('input', 0) + model.cost_per_1k_tokens.get('output', 0)
        
        return min(models, key=get_cost)
    
    def _select_by_performance(self, models: List[ModelConfig], 
                              metrics: Dict[str, ModelMetrics]) -> ModelConfig:
        """Select model with best performance."""
        def get_performance_score(model: ModelConfig) -> float:
            model_metrics = metrics.get(model.name)
            if not model_metrics:
                return 0
            
            # Combine response time and error rate
            response_time_score = 1 / (model_metrics.avg_response_time + 0.1)
            error_rate_score = 1 - model_metrics.error_rate
            return response_time_score * error_rate_score
        
        return max(models, key=get_performance_score)
    
    def _select_balanced(self, models: List[ModelConfig], 
                        metrics: Dict[str, ModelMetrics]) -> ModelConfig:
        """Select model with balanced cost/performance."""
        def get_balanced_score(model: ModelConfig) -> float:
            # Cost component
            cost = float('inf')
            if model.cost_per_1k_tokens:
                cost = model.cost_per_1k_tokens.get('input', 0) + model.cost_per_1k_tokens.get('output', 0)
            cost_score = 1 / (cost + 0.001) if cost != float('inf') else 0
            
            # Performance component
            model_metrics = metrics.get(model.name)
            if model_metrics:
                perf_score = (1 - model_metrics.error_rate) / (model_metrics.avg_response_time + 0.1)
            else:
                perf_score = 0.5
            
            # Priority component
            priority_score = model.priority / 10.0
            
            return cost_score * 0.3 + perf_score * 0.5 + priority_score * 0.2
        
        return max(models, key=get_balanced_score)
    
    def _select_round_robin(self, models: List[ModelConfig]) -> ModelConfig:
        """Select model using round-robin."""
        self.last_used_index = (self.last_used_index + 1) % len(models)
        return models[self.last_used_index]
    
    def _select_least_loaded(self, models: List[ModelConfig], 
                           metrics: Dict[str, ModelMetrics]) -> ModelConfig:
        """Select model with least load."""
        def get_load(model: ModelConfig) -> float:
            model_metrics = metrics.get(model.name)
            if not model_metrics:
                return 0
            return model_metrics.requests_per_minute
        
        return min(models, key=get_load)

class ModelManager:
    """Central AI model management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Model configurations and instances
        self.models: Dict[str, ModelConfig] = {}
        self.model_instances: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Routing and load balancing
        default_strategy = self.config.get("routing_strategy", "balanced")
        self.router = ModelRouter(RoutingStrategy(default_strategy))
        
        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_enabled = self.config.get("enable_caching", True)
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("ModelManager initialized")
    
    async def initialize(self):
        """Initialize the model manager."""
        try:
            logger.info("Initializing ModelManager...")
            
            # Load model configurations
            await self._load_model_configs()
            
            # Initialize model instances
            await self._initialize_models()
            
            # Start background tasks
            self.running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            logger.info(f"ModelManager initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {e}")
            raise
    
    async def _load_model_configs(self):
        """Load model configurations from config files."""
        try:
            # Load from main config
            ai_models_config = load_config("config/ai_models.yaml")
            
            # Process each provider
            for provider_name, provider_config in ai_models_config.items():
                if provider_name in ["openai", "anthropic", "local", "huggingface"]:
                    await self._load_provider_models(provider_name, provider_config)
            
        except Exception as e:
            logger.error(f"Failed to load model configs: {e}")
    
    async def _load_provider_models(self, provider: str, config: Dict[str, Any]):
        """Load models for a specific provider."""
        try:
            models_config = config.get("models", {})
            
            for model_name, model_config in models_config.items():
                # Create model configuration
                model_cfg = ModelConfig(
                    name=f"{provider}_{model_name}",
                    provider=provider,
                    api_key=config.get("api_key"),
                    base_url=config.get("base_url"),
                    model_id=model_config.get("name"),
                    temperature=model_config.get("temperature", DEFAULT_MODEL_CONFIG["temperature"]),
                    max_tokens=model_config.get("max_tokens", DEFAULT_MODEL_CONFIG["max_tokens"]),
                    timeout=model_config.get("timeout", DEFAULT_MODEL_CONFIG["timeout"]),
                    cost_per_1k_tokens=model_config.get("cost_per_1k_tokens"),
                    capabilities=model_config.get("capabilities", []),
                    context_window=model_config.get("context_window", 4096),
                    priority=model_config.get("priority", 1)
                )
                
                self.models[model_cfg.name] = model_cfg
                self.model_metrics[model_cfg.name] = ModelMetrics()
                self.circuit_breakers[model_cfg.name] = CircuitBreaker()
                
                logger.debug(f"Loaded model config: {model_cfg.name}")
                
        except Exception as e:
            logger.error(f"Failed to load {provider} models: {e}")
    
    async def _initialize_models(self):
        """Initialize model instances."""
        try:
            for model_name, model_config in self.models.items():
                try:
                    instance = await self._create_model_instance(model_config)
                    if instance:
                        self.model_instances[model_name] = instance
                        logger.debug(f"Initialized model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize model {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
    
    async def _create_model_instance(self, config: ModelConfig) -> Optional[Any]:
        """Create model instance based on provider."""
        try:
            if config.provider == OPENAI_PROVIDER:
                return await self._create_openai_instance(config)
            elif config.provider == ANTHROPIC_PROVIDER:
                return await self._create_anthropic_instance(config)
            elif config.provider == "local":
                return await self._create_local_instance(config)
            elif config.provider == "huggingface":
                return await self._create_huggingface_instance(config)
            else:
                logger.warning(f"Unknown provider: {config.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create instance for {config.name}: {e}")
            return None
    
    async def _create_openai_instance(self, config: ModelConfig) -> Optional[Any]:
        """Create OpenAI client instance."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
            
            return client
            
        except ImportError:
            logger.error("OpenAI library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to create OpenAI instance: {e}")
            return None
    
    async def _create_anthropic_instance(self, config: ModelConfig) -> Optional[Any]:
        """Create Anthropic client instance."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
            
            return client
            
        except ImportError:
            logger.error("Anthropic library not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to create Anthropic instance: {e}")
            return None
    
    async def _create_local_instance(self, config: ModelConfig) -> Optional[Any]:
        """Create local model instance."""
        # This would integrate with local model frameworks like llama.cpp, transformers, etc.
        logger.info(f"Local model not yet implemented: {config.name}")
        return None
    
    async def _create_huggingface_instance(self, config: ModelConfig) -> Optional[Any]:
        """Create Hugging Face model instance."""
        # This would integrate with Hugging Face transformers
        logger.info(f"Hugging Face model not yet implemented: {config.name}")
        return None
    
    def get_available_models(self, capabilities: Optional[List[str]] = None) -> List[ModelConfig]:
        """Get list of available models, optionally filtered by capabilities."""
        available_models = []
        
        for model_name, model_config in self.models.items():
            # Check if model instance exists and circuit breaker allows requests
            if (model_name in self.model_instances and 
                self.circuit_breakers[model_name].can_request()):
                
                # Filter by capabilities if specified
                if capabilities:
                    if not any(cap in model_config.capabilities for cap in capabilities):
                        continue
                
                available_models.append(model_config)
        
        return available_models
    
    def select_model(self, capabilities: Optional[List[str]] = None,
                    request_context: Optional[Dict[str, Any]] = None) -> Optional[ModelConfig]:
        """Select the best model for a request."""
        available_models = self.get_available_models(capabilities)
        
        if not available_models:
            logger.warning(f"No available models found for capabilities: {capabilities}")
            return None
        
        return self.router.select_model(available_models, self.model_metrics, request_context)
    
    async def generate_completion(self, 
                                prompt: str,
                                model_name: Optional[str] = None,
                                temperature: Optional[float] = None,
                                max_tokens: Optional[int] = None,
                                **kwargs) -> StandardResponse:
        """Generate text completion using selected model."""
        start_time = time.time()
        model_config = None
        
        try:
            # Select model if not specified
            if model_name:
                model_config = self.models.get(model_name)
                if not model_config:
                    return create_error_response(f"Model not found: {model_name}")
            else:
                model_config = self.select_model(["text_generation"])
                if not model_config:
                    return create_error_response("No available models for text generation")
            
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers[model_config.name]
            if not circuit_breaker.can_request():
                return create_error_response(f"Model {model_config.name} is circuit-broken")
            
            # Check cache
            cache_key = self._generate_cache_key(prompt, model_config, temperature, max_tokens)
            if self.cache_enabled:
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    return cached_response
            
            # Get model instance
            model_instance = self.model_instances.get(model_config.name)
            if not model_instance:
                return create_error_response(f"Model instance not available: {model_config.name}")
            
            # Generate completion based on provider
            response = await self._generate_with_provider(
                model_instance, model_config, prompt, temperature, max_tokens, **kwargs
            )
            
            # Calculate metrics
            processing_time = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = self._calculate_cost(model_config, response.usage) if response.usage else 0.0
            
            # Update metrics and circuit breaker
            self.model_metrics[model_config.name].update(
                response.success, tokens_used, cost, processing_time
            )
            
            if response.success:
                circuit_breaker.record_success()
                
                # Cache successful response
                if self.cache_enabled and response.content:
                    self._cache_response(cache_key, response)
            else:
                circuit_breaker.record_failure()
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Completion generation failed: {e}")
            
            # Update metrics for failure
            if model_config:
                self.model_metrics[model_config.name].update(False, 0, 0.0, processing_time)
                self.circuit_breakers[model_config.name].record_failure()
            
            return create_error_response(f"Generation failed: {str(e)}")
    
    async def _generate_with_provider(self, 
                                    instance: Any,
                                    config: ModelConfig,
                                    prompt: str,
                                    temperature: Optional[float],
                                    max_tokens: Optional[int],
                                    **kwargs) -> StandardResponse:
        """Generate completion with specific provider."""
        try:
            # Use provided parameters or config defaults
            temp = temperature if temperature is not None else config.temperature
            max_tok = max_tokens if max_tokens is not None else config.max_tokens
            
            if config.provider == OPENAI_PROVIDER:
                return await self._openai_completion(instance, config, prompt, temp, max_tok, **kwargs)
            elif config.provider == ANTHROPIC_PROVIDER:
                return await self._anthropic_completion(instance, config, prompt, temp, max_tok, **kwargs)
            else:
                return create_error_response(f"Provider not implemented: {config.provider}")
                
        except Exception as e:
            return create_error_response(f"Provider generation failed: {str(e)}")
    
    async def _openai_completion(self, client, config: ModelConfig, prompt: str,
                               temperature: float, max_tokens: int, **kwargs) -> StandardResponse:
        """Generate completion using OpenAI."""
        try:
            response = await client.completions.create(
                model=config.model_id or "gpt-3.5-turbo-instruct",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response data
            content = response.choices[0].text if response.choices else ""
            
            # Create usage info
            usage = ModelUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            )
            
            usage.estimated_cost = self._calculate_cost(config, usage)
            
            return create_success_response(
                content=content,
                model=config.name,
                provider=config.provider,
                usage=usage
            )
            
        except Exception as e:
            return create_error_response(f"OpenAI completion failed: {str(e)}")
    
    async def _anthropic_completion(self, client, config: ModelConfig, prompt: str,
                                  temperature: float, max_tokens: int, **kwargs) -> StandardResponse:
        """Generate completion using Anthropic."""
        try:
            response = await client.messages.create(
                model=config.model_id or "claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response data
            content = response.content[0].text if response.content else ""
            
            # Create usage info
            usage = ModelUsage(
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            )
            
            usage.estimated_cost = self._calculate_cost(config, usage)
            
            return create_success_response(
                content=content,
                model=config.name,
                provider=config.provider,
                usage=usage
            )
            
        except Exception as e:
            return create_error_response(f"Anthropic completion failed: {str(e)}")
    
    def _calculate_cost(self, config: ModelConfig, usage: ModelUsage) -> float:
        """Calculate estimated cost for the request."""
        if not config.cost_per_1k_tokens or not usage:
            return 0.0
        
        input_cost = (usage.prompt_tokens / 1000) * config.cost_per_1k_tokens.get("input", 0)
        output_cost = (usage.completion_tokens / 1000) * config.cost_per_1k_tokens.get("output", 0)
        
        return input_cost + output_cost
    
    def _generate_cache_key(self, prompt: str, config: ModelConfig, 
                          temperature: Optional[float], max_tokens: Optional[int]) -> str:
        """Generate cache key for request."""
        import hashlib
        
        cache_data = {
            "prompt": prompt,
            "model": config.name,
            "temperature": temperature or config.temperature,
            "max_tokens": max_tokens or config.max_tokens
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[StandardResponse]:
        """Get cached response if valid."""
        cached_item = self.cache.get(cache_key)
        if not cached_item:
            return None
        
        # Check expiration
        if datetime.now() > cached_item["expires_at"]:
            del self.cache[cache_key]
            return None
        
        return StandardResponse(**cached_item["response"])
    
    def _cache_response(self, cache_key: str, response: StandardResponse):
        """Cache successful response."""
        if not response.success or not response.content:
            return
        
        expires_at = datetime.now() + timedelta(seconds=3600)  # 1 hour default
        
        self.cache[cache_key] = {
            "response": response.dict(),
            "expires_at": expires_at,
            "cached_at": datetime.now()
        }
    
    async def _cleanup_loop(self):
        """Periodic cleanup of cache and metrics."""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._cleanup_cache()
                await self._cleanup_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        now = datetime.now()
        expired_keys = []
        
        for key, item in self.cache.items():
            if now > item["expires_at"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _cleanup_metrics(self):
        """Reset old metrics data."""
        # Reset hourly metrics that are too old
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for model_name, metrics in self.model_metrics.items():
            if (metrics.last_request_time and 
                metrics.last_request_time < cutoff_time):
                # Reset old metrics but keep cumulative data
                metrics.avg_response_time = 0.0
                metrics.requests_per_minute = 0.0
    
    async def _metrics_loop(self):
        """Periodic metrics calculation."""
        while self.running:
            try:
                await asyncio.sleep(60)  # 1 minute
                await self._calculate_real_time_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
    
    async def _calculate_real_time_metrics(self):
        """Calculate real-time metrics like requests per minute."""
        for model_name, metrics in self.model_metrics.items():
            # Calculate requests per minute (simplified)
            if metrics.last_request_time:
                time_diff = (datetime.now() - metrics.last_request_time).total_seconds()
                if time_diff < 3600:  # Within last hour
                    metrics.requests_per_minute = metrics.total_requests / max(time_diff / 60, 1)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed model information."""
        config = self.models.get(model_name)
        if not config:
            return None
        
        metrics = self.model_metrics.get(model_name, ModelMetrics())
        circuit_breaker = self.circuit_breakers.get(model_name)
        
        # Determine status
        status = ModelStatus.AVAILABLE
        if model_name not in self.model_instances:
            status = ModelStatus.ERROR
        elif circuit_breaker and not circuit_breaker.can_request():
            status = ModelStatus.RATE_LIMITED
        
        return ModelInfo(
            name=config.name,
            provider=config.provider,
            status=status,
            capabilities=config.capabilities,
            context_window=config.context_window,
            max_tokens=config.max_tokens,
            cost_per_1k_tokens=config.cost_per_1k_tokens,
            description=config.metadata.get("description")
        )
    
    def get_all_models_info(self) -> List[ModelInfo]:
        """Get information for all models."""
        return [
            self.get_model_info(name) 
            for name in self.models.keys()
            if self.get_model_info(name)
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary."""
        total_requests = sum(m.total_requests for m in self.model_metrics.values())
        total_cost = sum(m.total_cost for m in self.model_metrics.values())
        total_tokens = sum(m.total_tokens for m in self.model_metrics.values())
        
        avg_error_rate = sum(m.error_rate for m in self.model_metrics.values()) / len(self.model_metrics) if self.model_metrics else 0
        
        return {
            "total_models": len(self.models),
            "available_models": len([m for m in self.models.keys() if m in self.model_instances]),
            "total_requests": total_requests,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "average_error_rate": avg_error_rate,
            "cache_size": len(self.cache),
            "routing_strategy": self.router.strategy.value
        }
    
    async def health_check(self) -> bool:
        """Perform health check on model manager."""
        try:
            # Check if we have available models
            available_models = self.get_available_models()
            if not available_models:
                return False
            
            # Check if background tasks are running
            if not self.running or not self.cleanup_task or self.cleanup_task.done():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"ModelManager health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown model manager."""
        logger.info("Shutting down ModelManager...")
        self.running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()

# Close model instances
        for instance in self.model_instances.values():
            if hasattr(instance, 'close'):
                try:
                    await instance.close()
                except Exception as e:
                    logger.error(f"Error closing model instance: {e}")
        
        logger.info("ModelManager shutdown complete")
