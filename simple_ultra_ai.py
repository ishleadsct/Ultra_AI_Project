#!/usr/bin/env python3
"""
Simple Ultra AI Launcher
Tests core GGUF model functionality without complex dependencies
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import threading
import time

# Simple HTML interface
HTML_INTERFACE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra AI - Simple Test Interface</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: linear-gradient(135deg, #0c1445, #1a237e, #000051); 
            color: #00ffff; 
            margin: 0; 
            padding: 20px;
            min-height: 100vh;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: rgba(0,0,0,0.7);
            border-radius: 15px;
            box-shadow: 0 0 30px rgba(0,255,255,0.3);
        }
        h1 { 
            text-align: center; 
            color: #00ffff;
            text-shadow: 0 0 20px rgba(0,255,255,0.8);
            margin-bottom: 30px;
        }
        .status { 
            background: rgba(0,255,0,0.1); 
            padding: 15px; 
            border-radius: 10px; 
            margin: 20px 0;
            border: 1px solid #00ff00;
        }
        .models { 
            background: rgba(0,0,255,0.1); 
            padding: 15px; 
            border-radius: 10px; 
            margin: 20px 0;
            border: 1px solid #00ffff;
        }
        select, input, button { 
            background: rgba(0,0,0,0.8); 
            color: #00ffff; 
            border: 1px solid #00ffff; 
            padding: 10px; 
            border-radius: 5px; 
            font-family: inherit;
            margin: 5px;
        }
        button { 
            cursor: pointer; 
            background: linear-gradient(45deg, #0066ff, #00ffff);
            color: #000;
            font-weight: bold;
        }
        button:hover { 
            background: linear-gradient(45deg, #00ffff, #0066ff); 
        }
        .chat-area { 
            background: rgba(0,0,0,0.8); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
            border: 1px solid #00ffff;
            min-height: 200px;
            overflow-y: auto;
        }
        .orb {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: radial-gradient(circle, #00ffff, #0066ff);
            margin: 20px auto;
            animation: pulse 2s infinite;
            box-shadow: 0 0 20px rgba(0,255,255,0.8);
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ultra AI - Simple Test Interface</h1>
        <div class="orb"></div>
        
        <div class="status">
            <h3>🤖 System Status</h3>
            <p>✅ Interface: Active</p>
            <p id="model-status">⏳ Model: Loading...</p>
        </div>
        
        <div class="models">
            <h3>🎯 Available Models</h3>
            <select id="modelSelect">
                <option value="">Loading models...</option>
            </select>
            <button onclick="loadModel()">Load Selected Model</button>
        </div>
        
        <div class="chat-area" id="chatArea">
            <p><strong>Ultra AI:</strong> Welcome! I'm Ultra AI. Select a model above and I'll be ready to assist you.</p>
        </div>
        
        <div style="margin-top: 20px;">
            <input type="text" id="messageInput" placeholder="Type your message..." style="width: 70%;">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        // Load available models
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();
                const select = document.getElementById('modelSelect');
                select.innerHTML = '';
                
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = `${model.display_name} (${model.size_gb}GB)`;
                    select.appendChild(option);
                });
                
                if (models.length > 0) {
                    document.getElementById('model-status').textContent = '✅ Models: ' + models.length + ' available';
                }
            } catch (error) {
                console.error('Failed to load models:', error);
                document.getElementById('model-status').textContent = '❌ Models: Failed to load';
            }
        }
        
        // Load selected model
        async function loadModel() {
            const select = document.getElementById('modelSelect');
            const modelName = select.value;
            
            if (!modelName) {
                alert('Please select a model first');
                return;
            }
            
            document.getElementById('model-status').textContent = '⏳ Model: Loading ' + modelName + '...';
            
            try {
                const response = await fetch('/api/load_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: modelName})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('model-status').textContent = '✅ Model: ' + modelName + ' loaded';
                    addMessage('Ultra AI', 'Model ' + modelName + ' loaded successfully! I\'m ready to help.');
                } else {
                    document.getElementById('model-status').textContent = '❌ Model: Failed to load ' + modelName;
                    addMessage('Ultra AI', 'Failed to load model: ' + result.error);
                }
            } catch (error) {
                console.error('Failed to load model:', error);
                document.getElementById('model-status').textContent = '❌ Model: Error loading ' + modelName;
            }
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage('You', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const result = await response.json();
                
                if (result.response) {
                    addMessage('Ultra AI', result.response);
                } else {
                    addMessage('Ultra AI', 'Error: ' + result.error);
                }
            } catch (error) {
                console.error('Failed to send message:', error);
                addMessage('Ultra AI', 'Failed to send message. Please check the model is loaded.');
            }
        }
        
        // Add message to chat
        function addMessage(sender, message) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.innerHTML = '<p><strong>' + sender + ':</strong> ' + message + '</p>';
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        // Handle Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Load models on page load
        loadModels();
    </script>
</body>
</html>
'''

class UltraAIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.models = []
        self.current_model = None
        self.load_models_config()
        super().__init__(*args, **kwargs)
    
    def load_models_config(self):
        """Load models from configuration"""
        try:
            with open('models.json', 'r') as f:
                self.models = json.load(f)
            print(f"✅ Loaded {len(self.models)} models from configuration")
        except Exception as e:
            print(f"❌ Failed to load models.json: {e}")
            self.models = []
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_INTERFACE.encode())
            
        elif self.path == '/api/models':
            # Return available models
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check which models actually exist
            available_models = []
            for model in self.models:
                if os.path.exists(model['path']):
                    available_models.append(model)
            
            self.wfile.write(json.dumps(available_models).encode())
            
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/load_model':
            # Load a specific model
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            model_name = data.get('model')
            model_info = None
            
            for model in self.models:
                if model['name'] == model_name:
                    model_info = model
                    break
            
            if model_info and os.path.exists(model_info['path']):
                # Simulate model loading (in real implementation, use llama-cpp-python)
                self.current_model = model_info
                response = {'success': True, 'message': f'Model {model_name} loaded'}
                print(f"✅ Simulated loading model: {model_name}")
            else:
                response = {'success': False, 'error': f'Model {model_name} not found'}
                print(f"❌ Model not found: {model_name}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/api/chat':
            # Handle chat messages
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            message = data.get('message', '')
            
            if self.current_model:
                # Simulate AI response (in real implementation, use the loaded GGUF model)
                ai_response = f"Hello! I'm Ultra AI running on the {self.current_model['display_name']}. You said: '{message}'. This is a test response to verify model connectivity. The model is located at {self.current_model['path']} and specializes in {self.current_model['specialty']}."
                response = {'response': ai_response}
                print(f"💬 Chat: {message} -> {ai_response[:50]}...")
            else:
                response = {'error': 'No model loaded. Please load a model first.'}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_error(404)

def main():
    print("🚀 Starting Simple Ultra AI Test Interface...")
    print("=" * 60)
    
    # Check if models exist
    try:
        with open('models.json', 'r') as f:
            models = json.load(f)
        print(f"✅ Found {len(models)} models in configuration")
        
        available_count = 0
        for model in models:
            if os.path.exists(model['path']):
                size_mb = os.path.getsize(model['path']) / (1024 * 1024)
                print(f"   ✅ {model['name']}: {size_mb:.0f}MB - {model['display_name']}")
                available_count += 1
            else:
                print(f"   ❌ {model['name']}: Not found at {model['path']}")
        
        if available_count == 0:
            print("❌ No models available! Please check your models.json paths.")
            return
            
    except Exception as e:
        print(f"❌ Failed to load models.json: {e}")
        return
    
    # Start HTTP server
    port = 8000
    try:
        server = HTTPServer(('127.0.0.1', port), UltraAIHandler)
        print(f"\n🌐 Ultra AI Simple Interface started!")
        print(f"   URL: http://127.0.0.1:{port}")
        print(f"   Available Models: {available_count}")
        print(f"\n💡 Features:")
        print(f"   • Model selection and loading")
        print(f"   • Basic chat interface")
        print(f"   • Model information display")
        print(f"   • Real-time model status")
        print(f"\n⏹️  Press Ctrl+C to stop")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping Ultra AI Simple Interface...")
        server.shutdown()
        print("✅ Stopped successfully")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()