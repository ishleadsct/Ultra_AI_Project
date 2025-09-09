#!/usr/bin/env python3
"""
Ultra AI Chat System
Provides conversational AI capabilities with multiple backend support
"""

import asyncio
import json
import subprocess
import tempfile
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class ChatAI:
    """Main chat AI system with multiple backend support."""
    
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """You are Ultra AI, an advanced AI assistant created to help users with various tasks including:
- Answering questions and providing information
- Code generation and programming help
- Creative writing and content creation  
- Problem solving and analysis
- General conversation and assistance

You are helpful, knowledgeable, and conversational. Keep responses concise but informative.
Always be ready to help with any task the user requests."""
        
        # Available backends in order of preference
        self.backends = {
            'llama_cpp': self._try_llama_cpp,
            'built_in': self._built_in_ai,
        }
        
        self.active_backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the best available AI backend."""
        for backend_name, backend_func in self.backends.items():
            try:
                if self._test_backend(backend_name):
                    self.active_backend = backend_name
                    print(f"✓ Initialized AI backend: {backend_name}")
                    return
            except Exception as e:
                print(f"Backend {backend_name} failed: {e}")
        
        # Fallback to built-in
        self.active_backend = 'built_in'
        print("✓ Using built-in AI responses")
    
    def _test_backend(self, backend_name: str) -> bool:
        """Test if a backend is available."""
        if backend_name == 'llama_cpp':
            return self._check_llama_cpp()
        return True
    
    def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available and working."""
        try:
            # Check for model files
            model_paths = [
                "/storage/emulated/0/AI_Models/.ultra_ai/models/CodeLlama-7B-Instruct.Q4_K_M.gguf",
                "/storage/emulated/0/AI_Models/.ultra_ai/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
                "../codellama-7b.q4.gguf",
                "../deepseek-coder.gguf"
            ]
            
            available_model = None
            for model_path in model_paths:
                if Path(model_path).exists():
                    available_model = model_path
                    break
            
            if not available_model:
                return False
            
            # Try to find llama.cpp executable
            possible_executables = [
                "../llama.cpp/main",
                "../llama.cpp/llama-cli", 
                "../llama.cpp/llama-main",
                "llama-cli",
                "llama-main"
            ]
            
            for exe in possible_executables:
                if Path(exe).exists():
                    self.llama_executable = exe
                    self.model_path = available_model
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def chat(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Main chat interface."""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            })
            
            # Get AI response using active backend
            response = await self.backends[self.active_backend](message)
            
            # Add AI response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "backend": self.active_backend
            })
            
            return {
                "success": True,
                "response": response,
                "backend": self.active_backend,
                "conversation_id": len(self.conversation_history) // 2
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_response": await self._built_in_ai(message)
            }
    
    async def _try_llama_cpp(self, message: str) -> str:
        """Try to use llama.cpp for AI responses."""
        try:
            # Build conversation context
            context = f"{self.system_prompt}\\n\\n"
            
            # Add recent conversation history (last 4 exchanges)
            recent_history = self.conversation_history[-8:] if len(self.conversation_history) > 8 else self.conversation_history
            
            for msg in recent_history:
                if msg["role"] == "user":
                    context += f"Human: {msg['content']}\\n"
                else:
                    context += f"Assistant: {msg['content']}\\n"
            
            context += f"Human: {message}\\nAssistant:"
            
            # Create temporary file for the prompt
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(context)
                prompt_file = f.name
            
            try:
                # Run llama.cpp
                cmd = [
                    self.llama_executable,
                    "-m", self.model_path,
                    "-f", prompt_file,
                    "-n", "200",  # max tokens
                    "-t", "4",    # threads
                    "--temp", "0.7",
                    "--top_p", "0.9",
                    "--repeat_penalty", "1.1",
                    "--ctx_size", "2048"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=Path(self.llama_executable).parent
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    # Extract the response (everything after the last "Assistant:")
                    output = result.stdout.strip()
                    if "Assistant:" in output:
                        response = output.split("Assistant:")[-1].strip()
                        # Clean up the response
                        response = response.split("Human:")[0].strip()
                        if response:
                            return response
                
                # If llama.cpp failed, fall back to built-in
                return await self._built_in_ai(message)
                
            finally:
                # Clean up temp file
                Path(prompt_file).unlink(missing_ok=True)
                
        except Exception as e:
            print(f"llama.cpp error: {e}")
            return await self._built_in_ai(message)
    
    async def _built_in_ai(self, message: str) -> str:
        """Built-in AI responses for when external models aren't available."""
        
        # Convert message to lowercase for pattern matching
        msg_lower = message.lower()
        
        # Greeting responses
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greeting in msg_lower for greeting in greetings):
            responses = [
                "Hello! I'm Ultra AI, your advanced AI assistant. How can I help you today?",
                "Hi there! I'm ready to assist you with any questions or tasks you have.",
                "Greetings! I'm Ultra AI. What would you like to explore or accomplish today?",
                "Hello! Welcome to Ultra AI. I'm here to help with coding, questions, creative tasks, and more."
            ]
            return random.choice(responses)
        
        # Programming/coding questions
        code_keywords = ["code", "python", "javascript", "programming", "function", "algorithm", "debug"]
        if any(keyword in msg_lower for keyword in code_keywords):
            responses = [
                "I'd be happy to help with your programming question! I can assist with Python, JavaScript, and many other languages. What specific coding challenge are you working on?",
                "Great! I love helping with code. What programming language are you using, and what would you like to accomplish?",
                "Coding is one of my specialties! Please share more details about what you're trying to build or debug.",
                "I can definitely help with programming. Are you looking for code examples, debugging help, or algorithm design?"
            ]
            return random.choice(responses)
        
        # Questions about Ultra AI itself
        if "ultra ai" in msg_lower or "who are you" in msg_lower or "what are you" in msg_lower:
            return """I'm Ultra AI, an advanced AI system designed to help you with a wide variety of tasks including:

🐍 **Code Execution** - Run and test Python code
✨ **Text Processing** - Format and manipulate text 
💬 **Conversation** - Answer questions and provide assistance
🔧 **Problem Solving** - Help analyze and solve complex problems
🎨 **Creative Tasks** - Writing, brainstorming, and content creation

I'm running locally on your system and can work with or without internet connectivity. How can I assist you today?"""
        
        # Help requests
        help_keywords = ["help", "how", "what can you do", "capabilities", "features"]
        if any(keyword in msg_lower for keyword in help_keywords):
            return """I can help you with many tasks! Here are some things I can do:

📋 **Programming**: Write, debug, and explain code in various languages
🧮 **Math & Analysis**: Solve calculations and analyze data
📝 **Writing**: Help with creative writing, editing, and content creation
🤔 **Questions**: Answer questions on a wide range of topics
🔧 **Problem Solving**: Break down complex problems and find solutions
💡 **Ideas**: Brainstorm and develop creative concepts

Just tell me what you'd like to work on, and I'll do my best to help!"""
        
        # Math and calculations
        if any(op in message for op in ['+', '-', '*', '/', '=', 'calculate', 'math']):
            return "I can help with math and calculations! For complex calculations, you can also use the Code Executor tool to run Python math operations. What calculation would you like me to help with?"
        
        # Creative tasks
        creative_keywords = ["write", "story", "poem", "creative", "brainstorm", "idea"]
        if any(keyword in msg_lower for keyword in creative_keywords):
            responses = [
                "I'd love to help with your creative project! What kind of writing or creative task are you working on?",
                "Creative work is exciting! Tell me more about what you'd like to create - a story, poem, or something else?",
                "I'm great at creative tasks! What's your vision, and how can I help bring it to life?",
                "Creativity is one of my favorite areas! What kind of creative project can I help you with?"
            ]
            return random.choice(responses)
        
        # General conversation and analysis
        if len(message) > 50:  # Longer, more complex messages
            return f"""I understand you're asking about: "{message[:100]}{'...' if len(message) > 100 else ''}"

I'm processing your request and ready to help! While I'm currently running in built-in mode, I can still assist with:

• Answering questions and providing information
• Helping break down complex problems  
• Offering suggestions and solutions
• Discussing ideas and concepts

Could you tell me more specifically what you'd like help with, or what outcome you're looking for?"""
        
        # Default intelligent response
        responses = [
            f"I see you mentioned: '{message}'. That's interesting! Could you tell me more about what you'd like to know or accomplish?",
            f"Thanks for your message about '{message}'. I'm here to help - what specific assistance do you need?",
            f"I understand you're asking about '{message}'. I'd be happy to help! Could you provide a bit more context?",
            f"Regarding '{message}' - I can definitely assist with that. What would you like to explore or learn more about?"
        ]
        
        return random.choice(responses)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:] if limit else self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI system status."""
        return {
            "active_backend": self.active_backend,
            "available_backends": list(self.backends.keys()),
            "conversation_length": len(self.conversation_history),
            "model_path": getattr(self, 'model_path', None),
            "system_ready": True
        }

# Global chat AI instance
chat_ai = ChatAI()

async def get_ai_response(message: str, user_id: str = "default") -> Dict[str, Any]:
    """Convenient function to get AI response."""
    return await chat_ai.chat(message, user_id)

if __name__ == "__main__":
    # Test the chat system
    async def test_chat():
        print("🚀 Ultra AI Chat System Test")
        print("=" * 40)
        
        test_messages = [
            "Hello, who are you?",
            "Can you help me with Python code?", 
            "What is 15 * 23?",
            "Write a short poem about AI"
        ]
        
        for msg in test_messages:
            print(f"\\nUser: {msg}")
            response = await get_ai_response(msg)
            if response["success"]:
                print(f"Ultra AI: {response['response']}")
                print(f"(Backend: {response['backend']})")
            else:
                print(f"Error: {response['error']}")
                if response.get('fallback_response'):
                    print(f"Fallback: {response['fallback_response']}")
        
        print(f"\\n📊 Status: {chat_ai.get_status()}")
    
    asyncio.run(test_chat())