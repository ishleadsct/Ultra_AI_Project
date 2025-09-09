#!/bin/bash

# Ultra AI - LocalRecall-inspired Knowledge Base Startup Script
# Starts Ultra Recall REST API server for persistent memory infrastructure

echo "🚀 Starting Ultra AI Knowledge Infrastructure..."
echo "=================================================="
echo "📚 LocalRecall-inspired REST API & Knowledge Base"
echo "🗄️ Persistent Memory for AI Agents on Android/Termux"
echo

# Set working directory
cd "$(dirname "$0")"

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python."
    exit 1
fi

# Start Ultra Recall API server
echo "🔄 Starting Ultra Recall API Server..."
echo "   - SQLite knowledge base with semantic search"
echo "   - REST API for memory management"
echo "   - Conversation history and context"
echo "   - Memory links and relationship mapping"
echo

# Check if port is available
PORT=5555
if netstat -an 2>/dev/null | grep -q ":$PORT "; then
    echo "⚠️ Port $PORT is already in use. Trying port 5556..."
    PORT=5556
fi

# Start the server
echo "🌐 Starting server on port $PORT..."
python3 -c "
import sys
sys.path.append('src')
from src.ai.ultra_recall import ultra_recall_server

print('🗄️ Ultra Recall - Knowledge Infrastructure')
print('   Based on LocalRecall open-source project')
print('   Optimized for Android/Termux environments')
print()

# Start the server
ultra_recall_server.port = $PORT
ultra_recall_server.start_server()

print()
print('📡 API Documentation:')
print('   Health Check: GET /api/health')
print('   Search: GET /api/knowledge/search?q=your_query')
print('   Store Knowledge: POST /api/knowledge/store')
print('   Store Conversation: POST /api/conversation/store')
print('   Get Context: GET /api/conversation/context?session_id=xxx')
print('   Statistics: GET /api/stats')
print()
print('🔧 Integration with Ultra AI models enabled')
print('   - Automatic conversation storage')
print('   - Semantic knowledge retrieval')
print('   - Context-aware responses')
print()
print('⏹️ Press Ctrl+C to stop the server')

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print()
    print('🛑 Stopping Ultra Recall server...')
    ultra_recall_server.stop_server()
    print('✅ Ultra Recall stopped successfully')
"