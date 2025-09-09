#!/usr/bin/env python3
"""
Ultra AI Storage Model
Always-active small model under 1GB for memory management and information storage/retrieval
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import queue

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import small model for fast storage operations
try:
    from llama_cpp import Llama
    llama_cpp_available = True
    logging.info("✓ llama-cpp-python available for storage model")
except ImportError:
    llama_cpp_available = False
    logging.warning("⚠ llama-cpp-python not available, using memory-only storage")

# Import memory system
try:
    from .memory_system import ultra_ai_memory
    memory_available = True
except ImportError:
    memory_available = False

class StorageAI:
    """Always-active small AI model for memory storage and retrieval operations."""
    
    def __init__(self):
        self.model_path = "/storage/emulated/0/AI_Models/.ultra_ai/models"
        
        # Use the fastest and smallest available models prioritized for speed
        self.storage_model_options = [
            "TinyLlama-1.1B-Chat-v0.3.Q4_0.gguf",    # Fastest quantization
            "TinyLlama-1.1B-Chat-v0.3.Q4_K_M.gguf",  # Fast, small
            "Qwen2-0.5B-Instruct.Q4_0.gguf",         # Smallest if available
            "Qwen2-1.5B-Instruct.Q4_0.gguf",         # Fastest quantization of existing
            "Qwen2-1.5B-Instruct.Q4_K_M.gguf",       # Fallback existing
        ]
        
        self.storage_model = None
        self.is_loaded = False
        self.is_running = False
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker_thread = None
        
        logging.info("🗄️ Storage AI initialized - Always-active memory manager")
    
    async def initialize_storage_model(self) -> Dict[str, Any]:
        """Initialize the small storage model."""
        if not llama_cpp_available:
            logging.info("📦 Running in memory-only mode (no GGUF model)")
            self.is_loaded = True
            return {"success": True, "mode": "memory_only", "message": "Storage AI running in memory-only mode"}
        
        # Try to find and load the smallest available model
        for model_file in self.storage_model_options:
            model_file_path = Path(self.model_path) / model_file
            if model_file_path.exists():
                try:
                    logging.info(f"🔄 Loading storage model: {model_file}...")
                    
                    self.storage_model = Llama(
                        model_path=str(model_file_path),
                        n_ctx=256,    # Very small context for maximum speed
                        n_threads=1,  # Single thread for minimal resource usage
                        n_batch=8,    # Small batch size for speed
                        n_gpu_layers=-1 if "CUDA" in str(model_file_path).upper() else 0,  # GPU if explicitly needed
                        use_mlock=True,   # Keep in RAM for speed
                        use_mmap=True,    # Memory mapping for efficiency
                        low_vram=False,   # Prioritize speed over VRAM usage
                        verbose=False,
                        seed=-1          # Random seed for variety
                    )
                    
                    self.is_loaded = True
                    logging.info(f"✅ Storage AI model {model_file} loaded successfully!")
                    
                    return {
                        "success": True,
                        "model": model_file,
                        "size": self._get_model_size(model_file_path),
                        "message": f"Storage AI loaded with {model_file}",
                        "mode": "gguf_model"
                    }
                    
                except Exception as e:
                    logging.warning(f"Failed to load {model_file}: {e}")
                    continue
        
        # Fallback to memory-only mode
        logging.info("📦 No suitable GGUF model found, running in memory-only mode")
        self.is_loaded = True
        return {"success": True, "mode": "memory_only", "message": "Storage AI running in memory-only mode"}
    
    def _get_model_size(self, model_path: Path) -> str:
        """Get approximate model size."""
        try:
            size_bytes = model_path.stat().st_size
            if size_bytes < 1024**3:  # Less than 1GB
                return f"{size_bytes / (1024**2):.1f}MB"
            else:
                return f"{size_bytes / (1024**3):.1f}GB"
        except:
            return "Unknown"
    
    async def start_background_service(self) -> Dict[str, Any]:
        """Start the always-active background storage service."""
        if not self.is_loaded:
            init_result = await self.initialize_storage_model()
            if not init_result["success"]:
                return init_result
        
        if self.is_running:
            return {"success": True, "message": "Storage AI already running"}
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._storage_worker, daemon=True)
        self.worker_thread.start()
        
        logging.info("🚀 Storage AI background service started")
        return {
            "success": True,
            "message": "Storage AI background service started",
            "mode": "gguf_model" if self.storage_model else "memory_only"
        }
    
    def _storage_worker(self):
        """Background worker thread for storage operations."""
        logging.info("🔄 Storage AI worker thread started")
        
        while self.is_running:
            try:
                # Process storage requests from the queue
                try:
                    request = self.request_queue.get(timeout=1.0)
                    response = self._process_storage_request(request)
                    self.response_queue.put(response)
                except queue.Empty:
                    continue  # No requests, continue listening
                    
            except Exception as e:
                logging.error(f"Storage worker error: {e}")
                time.sleep(0.5)
        
        logging.info("🛑 Storage AI worker thread stopped")
    
    def _process_storage_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a storage request using the storage model or memory operations."""
        try:
            operation = request.get("operation")
            data = request.get("data", {})
            
            if operation == "store_info":
                return self._handle_store_info(data)
            elif operation == "retrieve_info":
                return self._handle_retrieve_info(data)
            elif operation == "analyze_conversation":
                return self._handle_analyze_conversation(data)
            elif operation == "generate_summary":
                return self._handle_generate_summary(data)
            elif operation == "smart_search":
                return self._handle_smart_search(data)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logging.error(f"Storage request processing error: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_store_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle information storage request."""
        if not memory_available:
            return {"success": False, "error": "Memory system not available"}
        
        try:
            info_type = data.get("type", "general")
            content = data.get("content", "")
            context = data.get("context", "")
            importance = data.get("importance", 5)
            
            if info_type == "personal":
                key = data.get("key")
                value = data.get("value")
                if key and value:
                    success = ultra_ai_memory.store_personal_info(key, value)
                    return {"success": success, "operation": "store_personal", "key": key}
            
            elif info_type == "fact":
                success = ultra_ai_memory.remember_fact(content, context, importance)
                return {"success": success, "operation": "store_fact", "content": content[:50] + "..."}
            
            elif info_type == "auto_extract":
                results = ultra_ai_memory.auto_store_from_message(content)
                return {"success": True, "operation": "auto_extract", "results": results}
            
            return {"success": False, "error": "Invalid info type"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_retrieve_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle information retrieval request."""
        if not memory_available:
            return {"success": False, "error": "Memory system not available"}
        
        try:
            query_type = data.get("type", "general")
            query = data.get("query", "")
            
            if query_type == "personal":
                key = data.get("key")
                value = ultra_ai_memory.get_personal_info(key)
                return {"success": True, "operation": "retrieve_personal", "key": key, "value": value}
            
            elif query_type == "facts":
                limit = data.get("limit", 5)
                facts = ultra_ai_memory.recall_facts(query, limit)
                return {"success": True, "operation": "retrieve_facts", "facts": facts}
            
            elif query_type == "context":
                context = ultra_ai_memory.get_context_for_ai()
                return {"success": True, "operation": "retrieve_context", "context": context}
            
            elif query_type == "conversations":
                limit = data.get("limit", 5)
                conversations = ultra_ai_memory.get_recent_conversations(limit)
                return {"success": True, "operation": "retrieve_conversations", "conversations": conversations}
            
            return {"success": False, "error": "Invalid query type"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_analyze_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation for insights using storage AI model."""
        try:
            conversation = data.get("conversation", "")
            
            if self.storage_model and llama_cpp_available:
                # Use small model to analyze conversation
                prompt = f"""Analyze this conversation and extract:
1. Key topics discussed
2. User preferences mentioned
3. Important facts to remember
4. Emotional tone

Conversation: {conversation[:500]}

Analysis:"""
                
                response = self.storage_model(
                    prompt,
                    max_tokens=50,     # Very short responses for speed
                    temperature=0.1,   # Low temperature for fast, consistent responses
                    top_p=0.9,
                    top_k=20,         # Limit choices for speed
                    repeat_penalty=1.1,
                    stop=["\n", "Analysis:", "Topics:"],  # Quick stops
                    echo=False
                )
                
                analysis = response["choices"][0]["text"].strip()
                return {"success": True, "analysis": analysis, "method": "ai_model"}
            
            else:
                # Simple keyword-based analysis
                topics = []
                preferences = []
                
                # Extract topics and preferences using keywords
                words = conversation.lower().split()
                
                topic_keywords = ["about", "discuss", "topic", "subject", "regarding"]
                preference_keywords = ["like", "love", "prefer", "enjoy", "hate", "dislike"]
                
                for i, word in enumerate(words):
                    if word in topic_keywords and i + 1 < len(words):
                        topics.append(words[i + 1])
                    elif word in preference_keywords and i + 1 < len(words):
                        preferences.append(f"{word} {words[i + 1]}")
                
                analysis = f"Topics: {', '.join(topics[:3])}. Preferences: {', '.join(preferences[:3])}"
                return {"success": True, "analysis": analysis, "method": "keyword_extraction"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_generate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of stored information."""
        try:
            if not memory_available:
                return {"success": False, "error": "Memory system not available"}
            
            stats = ultra_ai_memory.get_memory_stats()
            recent_facts = ultra_ai_memory.recall_facts("", 3)
            
            if self.storage_model and llama_cpp_available:
                # Generate AI summary
                facts_text = ". ".join([fact["fact"] for fact in recent_facts])
                
                prompt = f"""Create a brief summary of this user's profile:
- Name: {stats.get('user_name', 'Not specified')}
- Location: {stats.get('user_location', 'Not specified')}
- Total conversations: {stats.get('total_conversations', 0)}
- Recent facts: {facts_text}

Summary:"""
                
                response = self.storage_model(
                    prompt,
                    max_tokens=40,     # Very brief summaries for speed
                    temperature=0.2,   # Low temperature for consistency
                    top_p=0.8,
                    top_k=15,         # Limit choices for speed
                    stop=["\n", "Summary:", "Profile:"],
                    echo=False
                )
                
                summary = response["choices"][0]["text"].strip()
                return {"success": True, "summary": summary, "stats": stats, "method": "ai_model"}
            
            else:
                # Generate simple summary
                summary = f"User profile: {stats.get('total_conversations', 0)} conversations recorded."
                if stats.get('user_name'):
                    summary += f" Name: {stats['user_name']}."
                if stats.get('user_location'):
                    summary += f" Location: {stats['user_location']}."
                
                return {"success": True, "summary": summary, "stats": stats, "method": "template"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_smart_search(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Smart search through stored information."""
        try:
            query = data.get("query", "").lower()
            search_type = data.get("search_type", "all")
            
            if not memory_available:
                return {"success": False, "error": "Memory system not available"}
            
            results = {
                "personal_info": [],
                "facts": [],
                "conversations": [],
                "total_matches": 0
            }
            
            # Search personal information
            if search_type in ["all", "personal"]:
                context = ultra_ai_memory.get_context_for_ai()
                for key, value in context.items():
                    if value and query in str(value).lower():
                        results["personal_info"].append({"key": key, "value": value})
                        results["total_matches"] += 1
            
            # Search facts
            if search_type in ["all", "facts"]:
                facts = ultra_ai_memory.recall_facts(query, 10)
                results["facts"] = facts
                results["total_matches"] += len(facts)
            
            # Search conversations
            if search_type in ["all", "conversations"]:
                conversations = ultra_ai_memory.get_recent_conversations(20)
                matching_convs = []
                for conv in conversations:
                    if query in conv.get("user_message", "").lower() or query in conv.get("ai_response", "").lower():
                        matching_convs.append(conv)
                        results["total_matches"] += 1
                
                results["conversations"] = matching_convs[:5]  # Limit to 5 matches
            
            return {"success": True, "results": results, "query": query}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def storage_request(self, operation: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a storage request and get response."""
        if not self.is_running:
            return {"success": False, "error": "Storage AI not running"}
        
        if data is None:
            data = {}
        
        request = {
            "operation": operation,
            "data": data,
            "timestamp": time.time()
        }
        
        # Add request to queue
        self.request_queue.put(request)
        
        # Wait for response (with timeout)
        try:
            response = self.response_queue.get(timeout=5.0)
            return response
        except queue.Empty:
            return {"success": False, "error": "Storage request timeout"}
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get storage AI status."""
        return {
            "loaded": self.is_loaded,
            "running": self.is_running,
            "has_model": self.storage_model is not None,
            "queue_size": self.request_queue.qsize(),
            "memory_available": memory_available,
            "llama_cpp_available": llama_cpp_available
        }
    
    async def stop_service(self):
        """Stop the storage service."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        
        if self.storage_model:
            del self.storage_model
            self.storage_model = None
        
        logging.info("🛑 Storage AI service stopped")

# Global storage AI instance
storage_ai = StorageAI()

# Convenience functions for easy access
async def store_information(info_type: str, **kwargs) -> Dict[str, Any]:
    """Store information using Storage AI."""
    return await storage_ai.storage_request("store_info", {"type": info_type, **kwargs})

async def retrieve_information(query_type: str, **kwargs) -> Dict[str, Any]:
    """Retrieve information using Storage AI."""
    return await storage_ai.storage_request("retrieve_info", {"type": query_type, **kwargs})

async def analyze_conversation(conversation: str) -> Dict[str, Any]:
    """Analyze conversation using Storage AI."""
    return await storage_ai.storage_request("analyze_conversation", {"conversation": conversation})

async def generate_memory_summary() -> Dict[str, Any]:
    """Generate memory summary using Storage AI."""
    return await storage_ai.storage_request("generate_summary", {})

async def smart_search(query: str, search_type: str = "all") -> Dict[str, Any]:
    """Smart search using Storage AI."""
    return await storage_ai.storage_request("smart_search", {"query": query, "search_type": search_type})

if __name__ == "__main__":
    # Test Storage AI system
    async def test_storage_ai():
        print("🗄️ Ultra AI Storage Model Test")
        print("=" * 50)
        
        # Initialize
        print("\n🚀 Initializing Storage AI...")
        init_result = await storage_ai.initialize_storage_model()
        print(f"Initialization: {init_result}")
        
        # Start service
        print("\n⚡ Starting background service...")
        start_result = await storage_ai.start_background_service()
        print(f"Service start: {start_result}")
        
        # Wait a moment for service to be ready
        await asyncio.sleep(1)
        
        # Test storage operations
        print("\n📝 Testing storage operations...")
        
        # Store personal info
        store_result = await store_information("personal", key="test_name", value="Alex")
        print(f"Store personal info: {store_result}")
        
        # Store a fact
        fact_result = await store_information("fact", content="User loves Python programming", context="test", importance=7)
        print(f"Store fact: {fact_result}")
        
        # Retrieve personal info
        retrieve_result = await retrieve_information("personal", key="test_name")
        print(f"Retrieve personal info: {retrieve_result}")
        
        # Generate summary
        summary_result = await generate_memory_summary()
        print(f"Memory summary: {summary_result}")
        
        # Test smart search
        search_result = await smart_search("python")
        print(f"Smart search: {search_result}")
        
        # Show status
        status = storage_ai.get_storage_status()
        print(f"\n📊 Storage AI Status: {status}")
        
        # Stop service
        print("\n🛑 Stopping service...")
        await storage_ai.stop_service()
        
    # Run test
    asyncio.run(test_storage_ai())