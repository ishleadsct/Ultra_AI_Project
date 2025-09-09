#!/bin/bash

# Ultra AI Complete Launch Script
# Launches the full enhanced Ultra AI system with all features

clear
echo "🚀" "=" * 70 "🚀"
echo "          ULTRA AI - COMPLETE ENHANCED SYSTEM LAUNCH"
echo "🚀" "=" * 70 "🚀"
echo
echo "🎯 ENHANCED FEATURES ACTIVE:"
echo "   ✅ Voice Activation (Wake Words: Ultra AI, Ultra, Hey Ultra, OK Ultra)"
echo "   ✅ Real GGUF Model Loading & Switching (Qwen2, Phi-3, CodeLlama, etc)"
echo "   ✅ Multi-Layer Memory System (Volatile, Session, Persistent)"
echo "   ✅ Dynamic Context Injection (LocalRecall-inspired)"
echo "   ✅ Personal Information Storage & Recall"
echo "   ✅ Conversation Continuity & History"
echo "   ✅ Real-Time Context (GPS, Time, Date)"
echo "   ✅ Internet Search (Wikipedia & Reddit)"
echo "   ✅ Ultra AI Identity & Model Specializations"
echo "   ✅ Fast Storage AI (940MB model for memory operations)"
echo "   ✅ REST API Knowledge Base"
echo "   ✅ Enhanced Response Generation (No Cutoffs)"
echo "   ✅ Futuristic 3D Web Interface"
echo
echo "🤖 MODEL SPECIALIZATIONS:"
echo "   • Qwen2: Ultra AI - General Intelligence Core"
echo "   • Phi-3: Ultra AI - Reasoning & Analysis Core"
echo "   • CodeLlama: Ultra AI - Programming Expert Core"
echo "   • Llama-3.1: Ultra AI - Advanced Intelligence Core"
echo "   • DeepSeek: Ultra AI - Deep Code Analysis Core"
echo

# Set working directory
cd "$(dirname "$0")"

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python."
    exit 1
fi

# Find available port
echo "🔍 Finding available ports..."
MAIN_PORT=8888
RECALL_PORT=5555
STORAGE_PORT=7777

if netstat -an 2>/dev/null | grep -q ":$MAIN_PORT "; then
    MAIN_PORT=8889
    echo "   ⚠️  Port 8888 busy, using $MAIN_PORT"
fi

if netstat -an 2>/dev/null | grep -q ":$RECALL_PORT "; then
    RECALL_PORT=5556
    echo "   ⚠️  Port 5555 busy, using $RECALL_PORT"
fi

echo "   🌐 Main Interface: http://127.0.0.1:$MAIN_PORT"
echo "   📚 Knowledge API: http://127.0.0.1:$RECALL_PORT"

# Start Ultra Recall Knowledge Base in background
echo
echo "🗄️ Starting Ultra Recall Knowledge Base..."
python3 -c "
import sys
sys.path.append('src')
from src.ai.ultra_recall import ultra_recall_server
ultra_recall_server.port = $RECALL_PORT
ultra_recall_server.start_server()
import time
time.sleep(2)
print('   ✅ Knowledge Base API running on port $RECALL_PORT')
" &

RECALL_PID=$!

# Start Fast Storage AI in background
echo "⚡ Starting Fast Storage AI (940MB model)..."
python3 -c "
import sys, asyncio
sys.path.append('src')
from src.ai.storage_ai import storage_ai

async def start_storage():
    await storage_ai.initialize_storage_model()
    await storage_ai.start_background_service()
    print('   ✅ Fast Storage AI running (memory operations)')
    
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(start_storage())
import time
while True:
    time.sleep(1)
" &

STORAGE_PID=$!

# Wait for services to initialize
echo "⏳ Initializing all systems..."
sleep 3

# Start main Ultra AI interface
echo
echo "🎨 Starting Ultra AI Futuristic 3D Interface..."
echo "   - Blue Pulsing AI Orb Representation"
echo "   - Real GGUF Model Integration" 
echo "   - Voice Activation Controls"
echo "   - Dynamic Memory Context"
echo "   - All Termux API Integration"
echo

# Launch main interface
python3 ultra_ai_futuristic.py --port $MAIN_PORT &
MAIN_PID=$!

# Wait for main interface to start
sleep 2

# Show final status
echo
echo "🎉" "=" * 70 "🎉"
echo "              ULTRA AI COMPLETE SYSTEM READY!"
echo "🎉" "=" * 70 "🎉"
echo
echo "🌐 ACCESS POINTS:"
echo "   Main Interface:    http://127.0.0.1:$MAIN_PORT"
echo "   Knowledge API:     http://127.0.0.1:$RECALL_PORT/api/health"
echo
echo "📱 MOBILE ACCESS:"
echo "   Replace 127.0.0.1 with your device's IP for network access"
echo
echo "🎮 FEATURES TO TEST:"
echo "   1. Model Switching - Try different Ultra AI cores"
echo "   2. Voice Activation - Press voice button, say 'Ultra AI help me'"
echo "   3. Memory System - Tell Ultra AI your name and preferences"
echo "   4. Context Continuity - Ask follow-up questions"
echo "   5. Real-Time Search - Ask about current events"
echo "   6. Device Control - Try battery, location, WiFi commands"
echo
echo "🔧 API ENDPOINTS:"
echo "   Knowledge Search: GET /api/knowledge/search?q=your_query"
echo "   Store Memory: POST /api/knowledge/store"
echo "   Get Stats: GET /api/stats"
echo
echo "💡 EXAMPLE VOICE COMMANDS:"
echo "   'Ultra AI, what's my name?'"
echo "   'Hey Ultra, remember I prefer dark mode'"
echo "   'OK Ultra, help me with Python programming'"
echo "   'Ultra, what's the latest news about AI?'"
echo
echo "⏹️  Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo
    echo "🛑 Stopping Ultra AI Complete System..."
    
    if [ ! -z "$MAIN_PID" ]; then
        kill $MAIN_PID 2>/dev/null
        echo "   ✅ Main Interface stopped"
    fi
    
    if [ ! -z "$RECALL_PID" ]; then
        kill $RECALL_PID 2>/dev/null
        echo "   ✅ Knowledge Base stopped"
    fi
    
    if [ ! -z "$STORAGE_PID" ]; then
        kill $STORAGE_PID 2>/dev/null
        echo "   ✅ Storage AI stopped"
    fi
    
    # Kill any remaining Ultra AI processes
    pkill -f "ultra_ai" 2>/dev/null
    pkill -f "ultra_recall" 2>/dev/null
    pkill -f "storage_ai" 2>/dev/null
    
    echo "✅ Ultra AI Complete System stopped successfully"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Wait for user interrupt
while true; do
    sleep 1
    # Check if main process is still running
    if ! kill -0 $MAIN_PID 2>/dev/null; then
        echo "⚠️  Main process stopped unexpectedly"
        cleanup
    fi
done