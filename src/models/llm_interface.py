"""
Ultra AI Project - LLM Interface

Unified interface for interacting with Large Language Models across different providers,
providing consistent API, message formatting, and response handling.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator

from ..utils.logger import get_logger
from .types import ModelUsage, StandardResponse, create_success_response, create_error_response

logger = get_logger(__name__)

class MessageRole(Enum):
    """Message roles for chat conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"

@dataclass
class ChatMessage:
    """Chat message structure."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMRequest(BaseModel):
    """LLM request structure."""
    messages: List[Dict[str, Any]] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")
    functions: Optional[List[Dict[str, Any]]] = Field(None, description="Available functions")
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Function call configuration")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice configuration")
    user: Optional[str] = Field(None, description="User identifier")
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate message format."""
        if not v:
            raise ValueError("Messages cannot be empty")
        
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            if msg['role'] not in [role.value for role in MessageRole]:
                raise ValueError(f"Invalid role: {msg['role']}")
        
        return v

class CompletionRequest(BaseModel):
    """Text completion request structure."""
    prompt: str = Field(..., description="Input prompt")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")
    echo: bool = Field(False, description="Echo the prompt in response")
    user: Optional[str] = Field(None, description="User identifier")

class LLMResponse(BaseModel):
    """LLM response structure."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    usage: Optional[ModelUsage] = Field(None, description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call result")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls")
    created: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    response_time: float = Field(..., description="Response time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class StreamingResponse:
    """Streaming response handler."""
    
    def __init__(self, response_iterator: AsyncGenerator, model: str, provider: str):
        self.response_iterator = response_iterator
        self.model = model
        self.provider = provider
        self.content_buffer = ""
        self.usage: Optional[ModelUsage] = None
        self.finish_reason: Optional[str] = None
        self.created = datetime.now()
        self.start_time = time.time()
    
    async def __aiter__(self):
        """Async iterator for streaming response."""
        async for chunk in self.response_iterator:
            yield chunk
    
    async def collect(self) -> LLMResponse:
        """Collect all chunks into a complete response."""
        async for chunk in self.response_iterator:
            if hasattr(chunk, 'content') and chunk.content:
                self.content_buffer += chunk.content
            
            # Update metadata from final chunk
            if hasattr(chunk, 'usage') and chunk.usage:
                self.usage = chunk.usage
            if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
                self.finish_reason = chunk.finish_reason
        
        response_time = time.time() - self.start_time
        
        return LLMResponse(
            content=self.content_buffer,
            model=self.model,
            provider=self.provider,
            usage=self.usage,
            finish_reason=self.finish_reason,
            created=self.created,
            response_time=response_time
        )

class LLMInterface:
    """Unified interface for Large Language Model interactions."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.conversation_contexts: Dict[str, List[ChatMessage]] = {}
        self.function_registry: Dict[str, callable] = {}
        
        logger.info("LLMInterface initialized")
    
    async def chat_completion(self, request: LLMRequest) -> Union[LLMResponse, StreamingResponse]:
        """Generate chat completion response."""
        try:
            start_time = time.time()
            
            # Select model if not specified
            if request.model:
                model_config = self.model_manager.models.get(request.model) if self.model_manager else None
                if not model_config:
                    raise ValueError(f"Model not found: {request.model}")
            else:
                if not self.model_manager:
                    raise ValueError("Model manager not available")
                
                model_config = self.model_manager.select_model(["text_generation", "chat"])
                if not model_config:
                    raise ValueError("No available models for chat completion")
            
            # Get model instance
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            model_instance = self.model_manager.model_instances.get(model_config.name)
            if not model_instance:
                raise ValueError(f"Model instance not available: {model_config.name}")
            
            # Generate response based on provider
            if model_config.provider == "openai":
                response = await self._openai_chat_completion(model_instance, model_config, request)
            elif model_config.provider == "anthropic":
                response = await self._anthropic_chat_completion(model_instance, model_config, request)
            else:
                raise ValueError(f"Provider not supported for chat: {model_config.provider}")
            
            # Handle streaming vs non-streaming
            if request.stream:
                return StreamingResponse(response, model_config.name, model_config.provider)
            else:
                # For non-streaming, collect the response
                if hasattr(response, '__aiter__'):
                    # It's a streaming response, collect it
                    streaming_resp = StreamingResponse(response, model_config.name, model_config.provider)
                    return await streaming_resp.collect()
                else:
                    # It's already a complete response
                    return response
                    
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def text_completion(self, request: CompletionRequest) -> Union[LLMResponse, StreamingResponse]:
        """Generate text completion response."""
        try:
            # Convert to chat format for unified handling
            chat_request = LLMRequest(
                messages=[{"role": "user", "content": request.prompt}],
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                stream=request.stream,
                user=request.user
            )
            
            return await self.chat_completion(chat_request)
            
        except Exception as e:
            logger.error(f"Text completion failed: {e}")
            raise
    
    async def _openai_chat_completion(self, client, config, request: LLMRequest):
        """Handle OpenAI chat completion."""
        try:
            # Prepare parameters
            params = {
                "model": config.model_id or "gpt-3.5-turbo",
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream
            }
            
            # Add optional parameters
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                params["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                params["presence_penalty"] = request.presence_penalty
            if request.stop is not None:
                params["stop"] = request.stop
            if request.functions is not None:
                params["functions"] = request.functions
            if request.function_call is not None:
                params["function_call"] = request.function_call
            if request.tools is not None:
                params["tools"] = request.tools
            if request.tool_choice is not None:
                params["tool_choice"] = request.tool_choice
            if request.user is not None:
                params["user"] = request.user
            
            # Make API call
            response = await client.chat.completions.create(**params)
            
            if request.stream:
                return self._process_openai_stream(response, config)
            else:
                return self._process_openai_response(response, config)
                
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise
    
    async def _anthropic_chat_completion(self, client, config, request: LLMRequest):
        """Handle Anthropic chat completion."""
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Prepare parameters
            params = {
                "model": config.model_id or "claude-3-sonnet-20240229",
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            if system_message:
                params["system"] = system_message
            
            if request.stop:
                params["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
            
            # Make API call
            response = await client.messages.create(**params)
            
            if request.stream:
                return self._process_anthropic_stream(response, config)
            else:
                return self._process_anthropic_response(response, config)
                
        except Exception as e:
            logger.error(f"Anthropic chat completion failed: {e}")
            raise
    
    def _process_openai_response(self, response, config) -> LLMResponse:
        """Process OpenAI non-streaming response."""
        try:
            choice = response.choices[0] if response.choices else None
            if not choice:
                raise ValueError("No response choices returned")
            
            content = choice.message.content or ""
            finish_reason = choice.finish_reason
            
            # Extract function/tool calls
            function_call = None
            tool_calls = None
            
            if hasattr(choice.message, 'function_call') and choice.message.function_call:
                function_call = {
                    "name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments
                }
            
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in choice.message.tool_calls
                ]
            
            # Create usage info
            usage = None
            if response.usage:
                usage = ModelUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
                
                if config.cost_per_1k_tokens:
                    input_cost = (usage.prompt_tokens / 1000) * config.cost_per_1k_tokens.get("input", 0)
                    output_cost = (usage.completion_tokens / 1000) * config.cost_per_1k_tokens.get("output", 0)
                    usage.estimated_cost = input_cost + output_cost
            
            return LLMResponse(
                content=content,
                model=config.name,
                provider=config.provider,
                usage=usage,
                finish_reason=finish_reason,
                function_call=function_call,
                tool_calls=tool_calls,
                response_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Failed to process OpenAI response: {e}")
            raise
    
    def _process_anthropic_response(self, response, config) -> LLMResponse:
        """Process Anthropic non-streaming response."""
        try:
            content = ""
            if response.content:
                content = response.content[0].text if response.content else ""
            
            finish_reason = response.stop_reason if hasattr(response, 'stop_reason') else None
            
            # Create usage info
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = ModelUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens
                )
                
                if config.cost_per_1k_tokens:
                    input_cost = (usage.prompt_tokens / 1000) * config.cost_per_1k_tokens.get("input", 0)
                    output_cost = (usage.completion_tokens / 1000) * config.cost_per_1k_tokens.get("output", 0)
                    usage.estimated_cost = input_cost + output_cost
            
            return LLMResponse(
                content=content,
                model=config.name,
                provider=config.provider,
                usage=usage,
                finish_reason=finish_reason,
                response_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Failed to process Anthropic response: {e}")
            raise
    
    async def _process_openai_stream(self, response, config):
        """Process OpenAI streaming response."""
        try:
            async for chunk in response:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Create chunk response
                    chunk_data = {
                        "content": delta.content if hasattr(delta, 'content') and delta.content else "",
                        "finish_reason": choice.finish_reason,
                        "usage": None
                    }
                    
                    # Add usage info for final chunk
                    if hasattr(chunk, 'usage') and chunk.usage:
                        chunk_data["usage"] = ModelUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        )
                    
                    yield type('StreamChunk', (), chunk_data)()
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def _process_anthropic_stream(self, response, config):
        """Process Anthropic streaming response."""
        try:
            async for chunk in response:
                chunk_data = {
                    "content": "",
                    "finish_reason": None,
                    "usage": None
                }
                
                if hasattr(chunk, 'delta') and chunk.delta:
                    if hasattr(chunk.delta, 'text'):
                        chunk_data["content"] = chunk.delta.text
                
                if hasattr(chunk, 'type'):
                    if chunk.type == "message_stop":
                        chunk_data["finish_reason"] = "stop"
                    elif chunk.type == "content_block_stop":
                        chunk_data["finish_reason"] = "stop"
                
                # Add usage info if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    chunk_data["usage"] = ModelUsage(
                        prompt_tokens=chunk.usage.input_tokens if hasattr(chunk.usage, 'input_tokens') else 0,
                        completion_tokens=chunk.usage.output_tokens if hasattr(chunk.usage, 'output_tokens') else 0,
                        total_tokens=(chunk.usage.input_tokens + chunk.usage.output_tokens) if hasattr(chunk.usage, 'input_tokens') and hasattr(chunk.usage, 'output_tokens') else 0
                    )
                
                yield type('StreamChunk', (), chunk_data)()
                
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    def add_conversation_context(self, conversation_id: str, message: ChatMessage):
        """Add message to conversation context."""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = []
        
        self.conversation_contexts[conversation_id].append(message)
        
        # Keep conversation context limited
        max_context = 50
        if len(self.conversation_contexts[conversation_id]) > max_context:
            self.conversation_contexts[conversation_id] = self.conversation_contexts[conversation_id][-max_context:]
    
    def get_conversation_context(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation context."""
        return self.conversation_contexts.get(conversation_id, [])
    
    def clear_conversation_context(self, conversation_id: str):
        """Clear conversation context."""
        if conversation_id in self.conversation_contexts:
            del self.conversation_contexts[conversation_id]
    
    def register_function(self, name: str, function: callable, description: str = ""):
        """Register a function for function calling."""
        self.function_registry[name] = {
            "function": function,
            "description": description
        }
        
        logger.debug(f"Registered function: {name}")
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered function."""
        if function_name not in self.function_registry:
            raise ValueError(f"Function not found: {function_name}")
        
        try:
            function = self.function_registry[function_name]["function"]
            
            # Execute function (handle both sync and async)
            if asyncio.iscoroutinefunction(function):
                result = await function(**arguments)
            else:
                result = function(**arguments)
            
            return result
            
        except Exception as e:
            logger.error(f"Function execution failed: {function_name} - {e}")
            raise
    
    def create_function_schema(self, function_name: str, parameters_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create function schema for API calls."""
        if function_name not in self.function_registry:
            raise ValueError(f"Function not found: {function_name}")
        
        return {
            "name": function_name,
            "description": self.function_registry[function_name]["description"],
            "parameters": parameters_schema
        }
    
    async def generate_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self.model_manager:
            raise ValueError("Model manager not available")
        
        # Select embedding model
        if model:
            model_config = self.model_manager.models.get(model)
            if not model_config:
                raise ValueError(f"Model not found: {model}")
        else:
            model_config = self.model_manager.select_model(["embeddings"])
            if not model_config:
                raise ValueError("No available embedding models")
        
        # Get model instance
        model_instance = self.model_manager.model_instances.get(model_config.name)
        if not model_instance:
            raise ValueError(f"Model instance not available: {model_config.name}")
        
        try:
            if model_config.provider == "openai":
                response = await model_instance.embeddings.create(
                    model=model_config.model_id or "text-embedding-ada-002",
                    input=texts
                )
                return [data.embedding for data in response.data]
            else:
                raise ValueError(f"Embeddings not supported for provider: {model_config.provider}")
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        if not self.model_manager:
            return {}
        
        return {
            "available_models": self.model_manager.get_all_models_info(),
            "metrics": self.model_manager.get_metrics_summary()
        }
