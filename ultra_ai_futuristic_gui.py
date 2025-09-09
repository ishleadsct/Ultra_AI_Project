#!/usr/bin/env python3
"""
Ultra AI Futuristic 3D Web Interface
Complete GUI with model switching, chat, voice controls, and futuristic design
"""

import sys
import json
import asyncio
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import threading
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ai.gguf_ai import gguf_manager, get_gguf_ai_response
    from integrations.termux_integration import termux_integration
    from voice.voice_activation import voice_activation
    from ai.dynamic_memory_layers import ultra_memory_manager
    from tools.simple_tools import SimpleCodeExecutor, SimpleMessageFormatter
    
    all_systems_available = True
    print("✅ All Ultra AI systems loaded successfully!")
    
except ImportError as e:
    print(f"⚠️ Some systems not available: {e}")
    all_systems_available = False

class UltraAIFuturisticHandler(BaseHTTPRequestHandler):
    """Futuristic 3D Ultra AI web interface handler."""
    
    def __init__(self, *args, **kwargs):
        self.voice_active = False
        self.current_model = "qwen2"
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to reduce log spam."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.serve_futuristic_interface()
        elif self.path == '/api/models':
            self.serve_models_api()
        elif self.path == '/api/status':
            self.serve_status_api()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/chat':
            self.handle_chat_request()
        elif self.path == '/api/switch-model':
            self.handle_model_switch()
        elif self.path == '/api/toggle-voice':
            self.handle_voice_toggle()
        elif self.path == '/api/termux':
            self.handle_termux_api()
        else:
            self.send_response(404)
            self.end_headers()
    
    def serve_futuristic_interface(self):
        """Serve the complete futuristic Ultra AI interface."""
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra AI - Futuristic Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            color: #00ffff;
            overflow-x: hidden;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            position: relative;
        }
        
        /* Futuristic Header */
        .header {
            text-align: center;
            margin-bottom: 30px;
            position: relative;
            z-index: 10;
        }
        
        .ultra-ai-title {
            font-size: 3.5em;
            font-weight: bold;
            background: linear-gradient(45deg, #00ffff, #0080ff, #8000ff);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: titlePulse 3s ease-in-out infinite;
            text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        }
        
        @keyframes titlePulse {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        /* AI Orb - 3D Pulsing Center */
        .ai-orb-container {
            display: flex;
            justify-content: center;
            margin: 30px 0;
            position: relative;
        }
        
        .ai-orb {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #00ffff, #0080ff, #000033);
            box-shadow: 
                0 0 30px rgba(0, 255, 255, 0.8),
                inset 0 0 30px rgba(0, 128, 255, 0.3);
            animation: orbPulse 2s ease-in-out infinite;
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .ai-orb:hover {
            transform: scale(1.1);
            box-shadow: 
                0 0 50px rgba(0, 255, 255, 1),
                inset 0 0 30px rgba(0, 128, 255, 0.5);
        }
        
        .ai-orb.voice-active {
            animation: voiceActive 1s ease-in-out infinite;
            box-shadow: 
                0 0 60px rgba(255, 0, 255, 1),
                inset 0 0 30px rgba(255, 0, 128, 0.5);
        }
        
        @keyframes orbPulse {
            0%, 100% { 
                transform: scale(1);
                box-shadow: 0 0 30px rgba(0, 255, 255, 0.8), inset 0 0 30px rgba(0, 128, 255, 0.3);
            }
            50% { 
                transform: scale(1.05);
                box-shadow: 0 0 40px rgba(0, 255, 255, 1), inset 0 0 40px rgba(0, 128, 255, 0.5);
            }
        }
        
        @keyframes voiceActive {
            0%, 100% { 
                transform: scale(1);
                box-shadow: 0 0 60px rgba(255, 0, 255, 1), inset 0 0 30px rgba(255, 0, 128, 0.5);
            }
            50% { 
                transform: scale(1.1);
                box-shadow: 0 0 80px rgba(255, 0, 255, 1.2), inset 0 0 40px rgba(255, 0, 128, 0.7);
            }
        }
        
        /* Model Selector */
        .model-selector {
            text-align: center;
            margin: 20px 0;
        }
        
        .model-select {
            background: rgba(0, 255, 255, 0.1);
            border: 2px solid #00ffff;
            color: #00ffff;
            padding: 12px 20px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .model-select:hover {
            background: rgba(0, 255, 255, 0.2);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        }
        
        /* Chat Interface */
        .chat-container {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 15px;
            backdrop-filter: blur(15px);
            margin: 20px 0;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 255, 255, 0.2);
        }
        
        .chat-header {
            background: linear-gradient(90deg, rgba(0, 255, 255, 0.2), rgba(0, 128, 255, 0.2));
            padding: 15px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .current-model {
            font-weight: bold;
            color: #00ffff;
        }
        
        .voice-toggle {
            background: rgba(255, 0, 255, 0.2);
            border: 1px solid #ff00ff;
            color: #ff00ff;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .voice-toggle:hover {
            background: rgba(255, 0, 255, 0.4);
            box-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
        }
        
        .voice-toggle.active {
            background: rgba(255, 0, 255, 0.6);
            box-shadow: 0 0 20px rgba(255, 0, 255, 0.8);
        }
        
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
        }
        
        .message {
            margin: 15px 0;
            padding: 12px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .message.user {
            background: rgba(0, 255, 255, 0.1);
            border-left: 4px solid #00ffff;
            margin-left: auto;
        }
        
        .message.ai {
            background: rgba(0, 128, 255, 0.1);
            border-left: 4px solid #0080ff;
        }
        
        .message.system {
            background: rgba(255, 255, 0, 0.1);
            border-left: 4px solid #ffff00;
            text-align: center;
            max-width: 100%;
        }
        
        .chat-input-container {
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(0, 255, 255, 0.5);
            color: #00ffff;
            padding: 15px;
            border-radius: 25px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .chat-input:focus {
            outline: none;
            border-color: #00ffff;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        }
        
        .send-btn {
            background: linear-gradient(45deg, #00ffff, #0080ff);
            border: none;
            color: white;
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4);
        }
        
        /* Controls Panel */
        .controls-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .control-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .control-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 255, 255, 0.2);
        }
        
        .control-btn {
            background: rgba(0, 255, 255, 0.2);
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        .control-btn:hover {
            background: rgba(0, 255, 255, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        }
        
        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            color: #00ffff;
            font-style: italic;
        }
        
        .loading.active {
            display: block;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .ultra-ai-title {
                font-size: 2.5em;
            }
            
            .ai-orb {
                width: 80px;
                height: 80px;
            }
            
            .controls-panel {
                grid-template-columns: 1fr;
            }
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.3);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 255, 255, 0.5);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 255, 255, 0.8);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="ultra-ai-title">ULTRA AI</h1>
            <p>Futuristic AI Interface with Advanced Capabilities</p>
        </div>
        
        <div class="ai-orb-container">
            <div class="ai-orb" id="aiOrb" title="Click to activate voice mode"></div>
        </div>
        
        <div class="model-selector">
            <select class="model-select" id="modelSelect">
                <option value="qwen2">🤖 qwen2 - General Intelligence Core (512 context)</option>
                <option value="phi3">🧠 phi3 - Reasoning & Analysis Core (1024 context)</option>
                <option value="codellama">💻 codellama - Programming Expert Core (1024 context)</option>
                <option value="llama31">🎯 llama31 - Advanced Intelligence Core (2048 context)</option>
                <option value="deepseek">🔬 deepseek - Deep Code Analysis Core (1024 context)</option>
            </select>
        </div>
        
        <div class="chat-container">
            <div class="chat-header">
                <span class="current-model">Current Model: <span id="currentModel">qwen2</span></span>
                <button class="voice-toggle" id="voiceToggle">🎤 Voice: OFF</button>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message system">
                    Welcome to Ultra AI! I'm your futuristic AI assistant with multiple specialized cores.
                    Select a model above and start chatting!
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="chatInput" placeholder="Ask me anything..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">SEND</button>
            </div>
        </div>
        
        <div class="loading" id="loading">🤖 AI is thinking...</div>
        
        <div class="controls-panel">
            <div class="control-card">
                <h3>🔋 Device Status</h3>
                <button class="control-btn" onclick="checkBattery()">Check Battery</button>
                <button class="control-btn" onclick="getLocation()">Get Location</button>
            </div>
            
            <div class="control-card">
                <h3>📱 Notifications</h3>
                <button class="control-btn" onclick="showNotification()">Test Notification</button>
                <button class="control-btn" onclick="vibrate()">Vibrate Device</button>
            </div>
            
            <div class="control-card">
                <h3>💾 Memory</h3>
                <button class="control-btn" onclick="getMemoryStats()">Memory Stats</button>
                <button class="control-btn" onclick="clearChat()">Clear Chat</button>
            </div>
        </div>
    </div>

    <script>
        let currentModel = 'qwen2';
        let voiceActive = false;
        let chatHistory = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            addSystemMessage('Ultra AI Futuristic Interface loaded successfully! 🚀');
            updateModelInfo();
        });
        
        // Model Selection
        document.getElementById('modelSelect').addEventListener('change', function() {
            const newModel = this.value;
            switchModel(newModel);
        });
        
        // Voice Toggle
        document.getElementById('voiceToggle').addEventListener('click', toggleVoice);
        document.getElementById('aiOrb').addEventListener('click', toggleVoice);
        
        function switchModel(modelName) {
            if (modelName === currentModel) return;
            
            showLoading(true);
            
            fetch('/api/switch-model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: modelName})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentModel = modelName;
                    document.getElementById('currentModel').textContent = modelName;
                    addSystemMessage(`Switched to ${modelName} - ${data.description || 'AI Core'}`);
                } else {
                    addSystemMessage(`Failed to switch to ${modelName}: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Model switch error:', error);
                addSystemMessage(`Error switching models: ${error.message}`);
            })
            .finally(() => {
                showLoading(false);
            });
        }
        
        function toggleVoice() {
            voiceActive = !voiceActive;
            
            const voiceBtn = document.getElementById('voiceToggle');
            const aiOrb = document.getElementById('aiOrb');
            
            if (voiceActive) {
                voiceBtn.textContent = '🎤 Voice: ON';
                voiceBtn.classList.add('active');
                aiOrb.classList.add('voice-active');
                addSystemMessage('🎤 Voice mode activated! Speak to interact.');
            } else {
                voiceBtn.textContent = '🎤 Voice: OFF';
                voiceBtn.classList.remove('active');
                aiOrb.classList.remove('voice-active');
                addSystemMessage('🔇 Voice mode deactivated.');
            }
            
            // Send voice toggle to backend
            fetch('/api/toggle-voice', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({active: voiceActive})
            });
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            showLoading(true);
            
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message: message,
                    model: currentModel,
                    history: chatHistory.slice(-5) // Send last 5 messages for context
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addMessage('ai', data.response);
                    if (data.model_info) {
                        // Optional: Show which model responded
                    }
                } else {
                    addMessage('system', `Error: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Chat error:', error);
                addMessage('system', `Connection error: ${error.message}`);
            })
            .finally(() => {
                showLoading(false);
            });
        }
        
        function addMessage(type, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.textContent = content;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Store in chat history
            chatHistory.push({type, content});
            if (chatHistory.length > 20) {
                chatHistory = chatHistory.slice(-20); // Keep last 20 messages
            }
        }
        
        function addSystemMessage(content) {
            addMessage('system', content);
        }
        
        function showLoading(show) {
            const loading = document.getElementById('loading');
            loading.classList.toggle('active', show);
        }
        
        // Device Controls
        function checkBattery() {
            callTermuxAPI('battery');
        }
        
        function getLocation() {
            callTermuxAPI('location');
        }
        
        function showNotification() {
            callTermuxAPI('notification', {message: 'Ultra AI Test Notification'});
        }
        
        function vibrate() {
            callTermuxAPI('vibrate');
        }
        
        function callTermuxAPI(api, params = {}) {
            fetch('/api/termux', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api, params})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addSystemMessage(`${api}: ${data.message || 'Success'}`);
                } else {
                    addSystemMessage(`${api} failed: ${data.error}`);
                }
            })
            .catch(error => {
                addSystemMessage(`${api} error: ${error.message}`);
            });
        }
        
        function getMemoryStats() {
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.memory_stats) {
                    const stats = data.memory_stats;
                    addSystemMessage(`Memory: ${stats.sessions || 0} sessions, ${stats.total_entries || 0} total entries`);
                } else {
                    addSystemMessage('Memory stats unavailable');
                }
            });
        }
        
        function clearChat() {
            document.getElementById('chatMessages').innerHTML = '';
            chatHistory = [];
            addSystemMessage('Chat cleared! Ready for new conversation.');
        }
        
        function updateModelInfo() {
            fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                if (data.models) {
                    addSystemMessage(`${Object.keys(data.models).length} AI models available and ready!`);
                }
            });
        }
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_models_api(self):
        """Serve available models API."""
        if all_systems_available:
            models = gguf_manager.get_available_models()
            response = {"models": models}
        else:
            response = {"models": {}}
        
        self.send_json_response(response)
    
    def serve_status_api(self):
        """Serve system status API."""
        status = {
            "systems_available": all_systems_available,
            "current_model": getattr(gguf_manager, 'current_model', None),
            "voice_active": self.voice_active,
            "termux_apis": len(termux_integration.available_apis) if all_systems_available else 0
        }
        
        if all_systems_available:
            try:
                stats = ultra_memory_manager.get_comprehensive_stats()
                status["memory_stats"] = {
                    "sessions": stats.get("SESSION", {}).get("sessions_available", 0),
                    "total_entries": sum(layer.get("total_entries", 0) for layer in stats.values() if isinstance(layer, dict))
                }
            except:
                status["memory_stats"] = {}
        
        self.send_json_response(status)
    
    def handle_chat_request(self):
        """Handle chat message requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # Debug log the raw data
            print(f"Raw chat data: {repr(post_data)}")
            
            data = json.loads(post_data)
            
            message = data.get('message', '')
            model = data.get('model', 'qwen2')
            
            if not message:
                self.send_json_response({"success": False, "error": "No message provided"})
                return
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            self.send_json_response({"success": False, "error": f"Invalid JSON data: {str(e)}"})
            return
        except Exception as e:
            print(f"Chat request error: {e}")
            self.send_json_response({"success": False, "error": f"Request processing error: {str(e)}"})
            return
        
        if all_systems_available:
            try:
                # Use asyncio to get AI response
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(get_gguf_ai_response(message, model))
                
                if response.get('success'):
                    self.send_json_response({
                        "success": True,
                        "response": response['response'],
                        "model": model,
                        "model_info": response.get('model_info', {})
                    })
                else:
                    self.send_json_response({
                        "success": False,
                        "error": response.get('error', 'AI response failed')
                    })
                
                loop.close()
                
            except Exception as e:
                self.send_json_response({
                    "success": False,
                    "error": f"Chat processing error: {str(e)}"
                })
        else:
            # Fallback response when systems not available
            self.send_json_response({
                "success": True,
                "response": f"Ultra AI {model} (Simulation Mode): I received your message '{message}'. All core systems are initializing. Please try again in a moment for full AI capabilities.",
                "model": model
            })
    
    def handle_model_switch(self):
        """Handle model switching requests."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)
        
        model = data.get('model', 'qwen2')
        
        if all_systems_available:
            try:
                # Actually switch the model in GGUF manager
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(gguf_manager.load_model(model))
                loop.close()
                
                if result.get('success'):
                    self.current_model = model
                    models = gguf_manager.get_available_models()
                    model_info = models.get(model, {})
                    self.send_json_response({
                        "success": True,
                        "model": model,
                        "description": model_info.get('description', f'{model} AI Core'),
                        "message": f"Switched to {model} successfully"
                    })
                else:
                    self.send_json_response({
                        "success": False,
                        "error": result.get('error', f'Failed to load {model}')
                    })
                    
            except Exception as e:
                self.send_json_response({
                    "success": False,
                    "error": f"Model switch error: {str(e)}"
                })
        else:
            self.send_json_response({
                "success": True,
                "model": model,
                "description": f"{model} AI Core (Simulation Mode)"
            })
    
    def handle_voice_toggle(self):
        """Handle voice activation toggle."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)
        
        self.voice_active = data.get('active', False)
        
        self.send_json_response({
            "success": True,
            "voice_active": self.voice_active
        })
    
    def handle_termux_api(self):
        """Handle Termux API calls."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)
        
        api = data.get('api', '')
        params = data.get('params', {})
        
        if not all_systems_available:
            self.send_json_response({
                "success": False,
                "error": "Termux integration not available"
            })
            return
        
        try:
            if api == 'battery':
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(termux_integration.get_battery_status())
                loop.close()
                
                if result.get('success'):
                    data = result['data']
                    message = f"Battery: {data.get('percentage', '?')}% - {data.get('status', 'Unknown')}"
                    self.send_json_response({"success": True, "message": message})
                else:
                    self.send_json_response({"success": False, "error": result.get('error', 'Battery check failed')})
            
            elif api == 'location':
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(termux_integration.get_location())
                loop.close()
                
                if result.get('success'):
                    data = result['data']
                    message = f"Location: {data.get('latitude', '?')}, {data.get('longitude', '?')}"
                    self.send_json_response({"success": True, "message": message})
                else:
                    self.send_json_response({"success": False, "error": result.get('error', 'Location failed')})
            
            elif api == 'notification':
                message = params.get('message', 'Ultra AI Notification')
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(termux_integration.show_notification(message))
                loop.close()
                
                if result.get('success'):
                    self.send_json_response({"success": True, "message": "Notification sent"})
                else:
                    self.send_json_response({"success": False, "error": result.get('error', 'Notification failed')})
            
            elif api == 'vibrate':
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(termux_integration.vibrate())
                loop.close()
                
                if result.get('success'):
                    self.send_json_response({"success": True, "message": "Device vibrated"})
                else:
                    self.send_json_response({"success": False, "error": result.get('error', 'Vibrate failed')})
            
            else:
                self.send_json_response({"success": False, "error": f"Unknown API: {api}"})
                
        except Exception as e:
            self.send_json_response({
                "success": False,
                "error": f"Termux API error: {str(e)}"
            })
    
    def send_json_response(self, data):
        """Send JSON response."""
        json_data = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data.encode())

def start_futuristic_gui(host='127.0.0.1', port=8888):
    """Start the futuristic Ultra AI GUI server."""
    import socket
    
    # Check if port is available, try alternatives if needed
    original_port = port
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            server_address = (host, port)
            httpd = HTTPServer(server_address, UltraAIFuturisticHandler)
            
            if port != original_port:
                print(f"⚠️  Port {original_port} was in use, using port {port} instead")
            
            print(f"🚀 Ultra AI Futuristic GUI starting on http://{host}:{port}")
            print("🎨 Features: 3D Interface, Model Switching, Voice Controls, Termux APIs")
            print("⏹️  Press Ctrl+C to stop")
            
            httpd.serve_forever()
            break
            
        except OSError as e:
            if "Address already in use" in str(e):
                port += 1
                if attempt < max_attempts - 1:
                    print(f"⚠️  Port {port-1} is in use, trying port {port}...")
                    continue
                else:
                    print(f"❌ Unable to find available port after {max_attempts} attempts")
                    return
            else:
                raise
        except KeyboardInterrupt:
            print("\n🛑 Ultra AI Futuristic GUI stopped")
            httpd.shutdown()
            break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra AI Futuristic GUI')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888, help='Port to bind to')
    
    args = parser.parse_args()
    
    start_futuristic_gui(args.host, args.port)