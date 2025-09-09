#!/usr/bin/env python3
"""
Ultra AI GGUF Model Integration
Real AI using GGUF models with llama-cpp-python
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import threading
import queue

# Import llama-cpp-python for real model loading
try:
    from llama_cpp import Llama
    llama_cpp_available = True
    logging.info("✓ llama-cpp-python available for real model loading")
except ImportError:
    llama_cpp_available = False
    logging.warning("⚠ llama-cpp-python not available, using simulated responses")

# Import context provider for enhanced AI responses
try:
    from .context_provider import context_provider
    context_available = True
    logging.info("✓ Context provider available (time, location, internet search)")
except ImportError:
    context_available = False
    logging.warning("⚠ Context provider not available")

# Import memory system for personal information storage
try:
    from .memory_system import ultra_ai_memory
    memory_available = True
    logging.info("✓ Memory system available (personal info, conversation history)")
except ImportError:
    memory_available = False
    logging.warning("⚠ Memory system not available")

# Import fast storage AI for memory operations
try:
    from .storage_ai import storage_ai
    storage_ai_available = True
    logging.info("✓ Fast Storage AI available (always-active memory management)")
except ImportError:
    storage_ai_available = False
    logging.warning("⚠ Storage AI not available")

# Import multi-layer dynamic memory system
try:
    from .dynamic_memory_layers import ultra_memory_manager
    dynamic_memory_available = True
    logging.info("✓ Dynamic Memory Layers available (multi-layer context injection)")
    # Start background maintenance
    ultra_memory_manager.start_background_maintenance()
except ImportError:
    dynamic_memory_available = False
    logging.warning("⚠ Dynamic Memory Layers not available")

class GGUFModelManager:
    """Manage GGUF models for Ultra AI."""
    
    def __init__(self):
        self.model_path = "/data/data/com.termux/files/home/Ultra_AI_Project/models/gguf"
        self.current_model = None
        self.loaded_model = None  # Actual llama-cpp model instance
        self.model_process = None
        self.response_queue = queue.Queue()
        self.is_loaded = False
        
        # Available models - Optimized configuration with full context utilization
        self.available_models = {
            "qwen2": {
                "file": "Qwen2-1.5B-Instruct.Q4_K_M.gguf",
                "size": "1.5B",
                "context_size": 2048,  # User-requested: Optimized for performance with 2048 max
                "description": "Ultra AI - General Intelligence Core",
                "specialization": "general_assistant",
                "personality": "I'm Ultra AI's General Intelligence Core, specializing in fast responses and general assistance."
            },
            "phi3": {
                "file": "Phi-3-mini-4k-instruct-q4.gguf", 
                "size": "3.8B",
                "context_size": 2048,  # User-requested: Optimized for performance with 2048 max
                "description": "Ultra AI - Reasoning & Analysis",
                "specialization": "reasoning_analysis", 
                "personality": "I'm Ultra AI's Reasoning & Analysis Core, specializing in logical thinking and problem-solving."
            },
            "codellama": {
                "file": "CodeLlama-7B-Instruct.Q4_K_M.gguf",
                "size": "7B", 
                "context_size": 2048,  # User-requested: Optimized for performance with 2048 max
                "description": "Ultra AI - Programming Expert",
                "specialization": "programming_coding",
                "personality": "I'm Ultra AI's Programming Expert Core, specializing in code generation, debugging, and software development."
            },
            "llama31": {
                "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "size": "8B",
                "context_size": 2048,  # User-requested: Optimized for performance with 2048 max
                "description": "Ultra AI - Advanced Intelligence",
                "specialization": "advanced_intelligence",
                "personality": "I'm Ultra AI's Advanced Intelligence Core, specializing in complex reasoning and comprehensive assistance."
            },
            "deepseek": {
                "file": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
                "size": "7B",
                "context_size": 2048,  # User-requested: Optimized for performance with 2048 max
                "description": "Ultra AI - Deep Code Analysis",
                "specialization": "deep_coding_analysis",
                "personality": "I'm Ultra AI's Deep Code Analysis Core, specializing in advanced programming, system architecture, and code optimization."
            }
        }
        
        logging.info("🤖 GGUF Model Manager initialized")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models."""
        models = {}
        for name, info in self.available_models.items():
            model_file = Path(self.model_path) / info["file"]
            models[name] = {
                **info,
                "available": model_file.exists(),
                "path": str(model_file)
            }
        return models
    
    async def load_model(self, model_name: str = "qwen2") -> Dict[str, Any]:
        """Load a GGUF model using llama-cpp-python."""
        if model_name not in self.available_models:
            return {"success": False, "error": f"Model {model_name} not found"}
        
        # If same model is already loaded, skip loading
        if self.current_model == model_name and self.is_loaded and self.loaded_model:
            return {
                "success": True,
                "model": model_name,
                "description": self.available_models[model_name]["description"],
                "message": f"Model {model_name} already loaded and ready"
            }
        
        model_info = self.available_models[model_name]
        model_file = Path(self.model_path) / model_info["file"]
        
        if not model_file.exists():
            return {"success": False, "error": f"Model file not found: {model_file}"}
        
        try:
            logging.info(f"🔄 Loading GGUF model: {model_name}...")
            
            # Unload previous model if loaded
            if self.loaded_model:
                del self.loaded_model
                self.loaded_model = None
                self.is_loaded = False
                logging.info("🗑️ Previous model unloaded")
            
            if llama_cpp_available:
                # Get optimal context size for this model
                context_size = model_info.get("context_size", 512)
                
                # Load the actual GGUF model with SmolChat-style lightweight config
                self.loaded_model = Llama(
                    model_path=str(model_file),
                    n_ctx=context_size,  # Model-specific minimal context for speed
                    n_threads=2,  # Fewer threads for lighter load
                    n_gpu_layers=0,  # CPU only for stability and compatibility
                    verbose=False,
                    n_batch=128,  # Smaller batch size
                    use_mlock=False,  # Don't lock memory
                    use_mmap=True,  # Use memory mapping for efficiency
                )
                
                self.current_model = model_name
                self.is_loaded = True
                
                logging.info(f"✅ Real GGUF model {model_name} loaded successfully!")
                return {
                    "success": True,
                    "model": model_name,
                    "description": model_info["description"],
                    "size": model_info["size"],
                    "message": f"Real GGUF model {model_name} loaded and ready!",
                    "real_model": True
                }
            else:
                # Fallback to simulation
                self.current_model = model_name
                self.is_loaded = True
                
                logging.info(f"⚠️ Simulated model {model_name} loaded (llama-cpp not available)")
                return {
                    "success": True,
                    "model": model_name,
                    "description": model_info["description"],
                    "size": model_info["size"],
                    "message": f"Model {model_name} loaded in simulation mode",
                    "real_model": False
                }
            
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_response(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """Generate AI response using loaded model with enhanced context, memory, and conversation continuity."""
        if not self.is_loaded:
            # Auto-load default model
            load_result = await self.load_model("qwen2")
            if not load_result["success"]:
                return load_result
        
        try:
            # Get comprehensive context from all memory layers
            user_info = {}
            memory_context = {}
            memory_used = False
            
            # Use dynamic memory layers for advanced context injection
            if dynamic_memory_available:
                try:
                    # Get user info from persistent memory
                    if memory_available:
                        memory_context = ultra_ai_memory.get_context_for_ai()
                        user_info = {
                            "name": memory_context.get("user_name"),
                            "location": memory_context.get("location"),
                            "preferences": memory_context.get("preferences", {})
                        }
                        # Auto-store new information
                        auto_stored = ultra_ai_memory.auto_store_from_message(prompt)
                    
                    memory_used = True
                    logging.info("🗄️ Using dynamic multi-layer memory system for context")
                except Exception as e:
                    logging.warning(f"Dynamic memory enhancement failed: {e}")
            
            # Enhance prompt with real-time context if available
            enhanced_prompt = prompt
            context_used = False
            
            if context_available:
                try:
                    enhanced_prompt = await context_provider.enhance_prompt_with_context(prompt)
                    context_used = True
                    logging.info("✨ Enhanced prompt with real-time context and search results")
                except Exception as e:
                    logging.warning(f"Context enhancement failed: {e}")
                    enhanced_prompt = prompt
            
            # Get current model's personality and specialization
            model_info = self.available_models.get(self.current_model, {})
            personality = model_info.get("personality", "I'm Ultra AI.")
            specialization = model_info.get("specialization", "general_assistant")
            
            if llama_cpp_available and self.loaded_model:
                # Use real GGUF model inference with dynamic context injection
                start_time = time.time()
                logging.info(f"🤖 Generating real AI response with {self.current_model} and dynamic context...")
                
                # Use dynamic memory system for advanced context injection
                dynamic_context = {}
                if dynamic_memory_available:
                    try:
                        # Get enhanced prompt with dynamic context from all memory layers
                        formatted_prompt, dynamic_context = await ultra_memory_manager.enhance_model_prompt(
                            enhanced_prompt if context_used else prompt,
                            personality,
                            user_info
                        )
                        logging.info(f"🎯 Dynamic context injection: {dynamic_context['metadata']['total_memories']} memories, {dynamic_context['metadata']['context_tokens']} tokens")
                    except Exception as e:
                        logging.warning(f"Dynamic context injection failed: {e}")
                        # Fallback to traditional prompt
                        formatted_prompt = f"{personality}\n\nUser: {enhanced_prompt}\n\nUltra AI:"
                else:
                    # Traditional system prompt for fallback
                    system_parts = [
                        f"You are {personality}",
                        "You have access to real-time information and personal memory about the user.",
                        "Always respond as Ultra AI and mention your specific role/specialization when relevant.",
                        "Provide complete, thoughtful responses. Do not cut off your sentences or thoughts."
                    ]
                    
                    # Add basic memory context if available
                    if memory_used and memory_context:
                        if memory_context.get("user_name"):
                            system_parts.append(f"The user's name is {memory_context['user_name']}.")
                        if memory_context.get("location"):
                            system_parts.append(f"The user is located in {memory_context['location']}.")
                    
                    system_prompt = " ".join(system_parts)
                    formatted_prompt = f"{system_prompt}\n\nUser: {enhanced_prompt}\n\nUltra AI:"
                
                response = self.loaded_model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["User:", "Human:", "\n\n\n"],  # Removed aggressive stop tokens to prevent cutoff
                    echo=False
                )
                
                end_time = time.time()
                response_text = response["choices"][0]["text"].strip()
                
                logging.info(f"✅ Real AI response generated in {(end_time - start_time):.2f}s")
                
                # Store the conversation in dynamic memory system
                if dynamic_memory_available:
                    await ultra_memory_manager.store_conversation(
                        prompt, response_text, self.current_model, dynamic_context
                    )
                elif memory_available:
                    ultra_ai_memory.add_conversation(prompt, response_text, self.current_model)
                
                return {
                    "success": True,
                    "response": response_text,
                    "model": self.current_model,
                    "specialization": specialization,
                    "personality": personality,
                    "tokens_used": response["usage"]["total_tokens"],
                    "real_model": True,
                    "inference_time": end_time - start_time,
                    "context_enhanced": context_used,
                    "memory_enhanced": memory_used,
                    "dynamic_context": dynamic_context.get("metadata", {}) if dynamic_context else {}
                }
            else:
                # Fallback to simulation with context and memory
                response = await self._simulate_model_response(enhanced_prompt, memory_context)
                
                # Store the conversation in memory
                if memory_available:
                    ultra_ai_memory.add_conversation(prompt, response, self.current_model)
                
                return {
                    "success": True,
                    "response": response,
                    "model": self.current_model,
                    "specialization": specialization,
                    "personality": personality,
                    "tokens_used": len(response.split()),
                    "real_model": False,
                    "context_enhanced": context_used,
                    "memory_enhanced": memory_used
                }
            
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_model_response(self, prompt: str, memory_context: Dict[str, Any] = None) -> str:
        """Simulate model response based on current model capabilities."""
        
        # Add small delay to simulate processing
        await asyncio.sleep(0.1)
        
        model = self.current_model or "qwen2"
        model_info = self.available_models.get(model, {})
        personality = model_info.get("personality", "I'm Ultra AI.")
        
        # Add personal context if available
        greeting_part = ""
        if memory_context and memory_context.get("user_name"):
            greeting_part = f"Hello {memory_context['user_name']}! "
        
        # Different response patterns based on model specialization
        if "code" in prompt.lower() or "python" in prompt.lower():
            if model == "codellama" or model == "deepseek":
                responses = [
                    f"{greeting_part}{personality} I'll help you with that code! Here's a solution:\n\n```python\n# Ultra AI Code Solution\nprint('Hello from Ultra AI Programming Expert!')\ndef ultra_solution():\n    return 'Code generated by Ultra AI'\n```\n\nThis solution demonstrates my programming expertise. Need any modifications?",
                    f"{greeting_part}As Ultra AI's Programming Expert, I can provide a Python implementation:\n\n```python\ndef solve_problem():\n    # Efficient solution by Ultra AI\n    result = 'Ultra AI Code Solution'\n    return result\n\nprint(solve_problem())\n```\n\nI specialize in code generation, debugging, and optimization.",
                    f"{greeting_part}Here's the code you need from Ultra AI:\n\n```python\n# Optimized implementation by Ultra AI\nimport os\nimport sys\n\ndef ultra_ai_solution():\n    return 'Advanced code by Ultra AI Programming Expert'\n\nresult = ultra_ai_solution()\nprint(result)\n```"
                ]
            else:
                responses = [
                    f"{greeting_part}{personality} I can help with coding! Here's a Python solution:\n\n```python\nprint('Hello from Ultra AI!')\n# General coding assistance\ndef ultra_helper():\n    return 'Ultra AI is here to help'\n```",
                    f"{greeting_part}Sure! Here's some code from Ultra AI:\n\n```python\n# Simple solution by Ultra AI\ndef example():\n    return 'Ultra AI response'\n\nprint(example())\n```"
                ]
        
        elif "explain" in prompt.lower() or "what" in prompt.lower():
            responses = [
                f"{greeting_part}{personality} I can explain concepts, solve problems, and provide detailed analysis. My specialization allows me to break down complex topics clearly. What would you like me to explain?",
                f"{greeting_part}As Ultra AI, I have access to extensive knowledge and real-time information. I can help with explanations, analysis, creative writing, and technical questions. How can I assist you?",
                f"{greeting_part}{personality} I can provide detailed explanations and help with complex topics. What specific subject interests you?"
            ]
        
        elif "create" in prompt.lower() or "write" in prompt.lower():
            responses = [
                f"{greeting_part}{personality} I'll create that for you! I can generate creative content, technical documentation, structured text, and more. What type of content do you need?",
                f"{greeting_part}Certainly! Here's what I've created as Ultra AI:\n\n**Ultra AI Creative Output**\nI specialize in generating high-quality content tailored to your specific requirements. What shall I create for you?",
                f"{greeting_part}{personality} I've generated content using my advanced capabilities. I can create various types of text, from technical documentation to creative writing."
            ]
        
        else:
            responses = [
                f"{greeting_part}{personality} I'm running locally on your device with real intelligence and memory capabilities. I can remember our conversations and assist you with various tasks. How can I help you today?",
                f"{greeting_part}Greetings! {personality} I'm your local Ultra AI assistant with access to extensive knowledge, real-time information, and memory of our interactions.",
                f"{greeting_part}{personality} I can help with analysis, creative tasks, coding, explanations, device control, and much more. I remember our previous conversations too. What do you need assistance with?",
                f"{greeting_part}Hi there! {personality} I'm running entirely on your device with real AI capabilities, memory, and access to current information. I'm ready to help!",
                f"{greeting_part}Welcome! {personality} I can assist with complex reasoning, creative projects, technical questions, and I'll remember everything we discuss for better continuity."
            ]
        
        import random
        return random.choice(responses)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "loaded": self.is_loaded,
            "current_model": self.current_model,
            "available_models": list(self.available_models.keys()),
            "model_info": self.available_models.get(self.current_model, {}) if self.current_model else None
        }

# Global GGUF manager instance
gguf_manager = GGUFModelManager()

# Alias for backward compatibility
GGUFAIManager = GGUFModelManager

async def get_gguf_ai_response(message: str, model: str = "qwen2") -> Dict[str, Any]:
    """Get AI response using GGUF models."""
    try:
        # Ensure model is loaded
        if gguf_manager.current_model != model:
            load_result = await gguf_manager.load_model(model)
            if not load_result["success"]:
                return load_result
        
        # Generate response
        result = await gguf_manager.generate_response(message)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test GGUF AI system
    async def test_gguf_ai():
        print("🤖 Ultra AI GGUF Integration Test")
        print("=" * 50)
        
        # Show available models
        models = gguf_manager.get_available_models()
        print("\n📚 Available Models:")
        for name, info in models.items():
            status = "✓" if info["available"] else "✗"
            print(f"  {status} {name}: {info['description']} ({info['size']})")
        
        # Test model loading and responses
        print("\n🧪 Testing Model Responses:")
        
        test_prompts = [
            "Hello, who are you?",
            "Write a Python function to calculate factorial",
            "Explain quantum computing",
            "Create a haiku about AI"
        ]
        
        for model in ["qwen2", "phi3", "codellama"]:
            if models.get(model, {}).get("available"):
                print(f"\n🔄 Testing {model.upper()}:")
                for prompt in test_prompts[:2]:  # Test 2 prompts per model
                    response = await get_gguf_ai_response(prompt, model)
                    if response["success"]:
                        print(f"  Q: {prompt}")
                        print(f"  A: {response['response'][:100]}...")
                        print()
                    else:
                        print(f"  Error: {response['error']}")
                break  # Test only first available model for demo
        
        # Show final status
        status = gguf_manager.get_model_status()
        print(f"\n📊 Final Status: {status}")
    
    asyncio.run(test_gguf_ai())