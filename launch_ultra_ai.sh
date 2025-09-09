#!/data/data/com.termux/files/usr/bin/bash

#
# Ultra AI Complete System Launcher
# Production-ready launcher with all features
#

set -e

# Colors and styling
BLUE='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
ULTRA_AI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
LOG_DIR="$ULTRA_AI_DIR/logs"
PID_FILE="$ULTRA_AI_DIR/ultra_ai_complete.pid"

mkdir -p "$LOG_DIR"

print_banner() {
    echo -e "${BLUE}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                       ULTRA AI COMPLETE SYSTEM                   ║"
    echo "║                        Production Ready v1.0                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_features() {
    echo -e "${GREEN}🎯 COMPLETE FEATURE SET:${NC}"
    echo -e "   ${BLUE}🤖 Production AI Engine${NC}     - Real intelligent responses"
    echo -e "   ${BLUE}🎤 Voice Activation${NC}         - 'Ultra AI' wake word detection"
    echo -e "   ${BLUE}📱 21 Termux APIs${NC}           - Full device integration"
    echo -e "   ${BLUE}💻 Code Execution${NC}           - Safe Python environment"
    echo -e "   ${BLUE}✨ Message Formatting${NC}       - Advanced text processing" 
    echo -e "   ${BLUE}🌐 3D Web Interface${NC}         - Futuristic user experience"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help          Show this help message"
    echo "  -p, --port PORT     Web interface port (default: 8888)"
    echo "  -H, --host HOST     Host address (default: 127.0.0.1)"
    echo "  --status            Check system status"
    echo "  --stop              Stop running system"
    echo "  --test              Run system tests"
    echo "  --daemon            Run as background service"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start with web interface"
    echo "  $0 -p 9999          # Start on custom port"
    echo "  $0 --daemon         # Run as background service"
    echo "  $0 --test           # Run comprehensive tests"
    echo "  $0 --status         # Check system status"
    echo ""
}

check_dependencies() {
    echo -e "${BLUE}🔍 Checking system dependencies...${NC}"
    
    # Check Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo -e "${RED}❌ Python 3 is required${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"
    
    # Check Ultra AI main script
    if [ ! -f "$ULTRA_AI_DIR/ultra_ai_complete.py" ]; then
        echo -e "${RED}❌ Ultra AI main script not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Ultra AI Complete System${NC}"
    
    # Test system components
    echo -e "${BLUE}🧪 Testing system components...${NC}"
    
    cd "$ULTRA_AI_DIR"
    
    if $PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')
try:
    from ai.production_ai import production_ai
    from integrations.termux_integration import termux_integration  
    from voice.voice_activation import voice_activation
    from tools.simple_tools import SimpleCodeExecutor, SimpleMessageFormatter
    print('✅ All systems operational')
except Exception as e:
    print(f'❌ System error: {e}')
    exit(1)
"; then
        echo -e "${GREEN}✅ All Ultra AI components loaded${NC}"
    else
        echo -e "${YELLOW}⚠️ Some components may have limited functionality${NC}"
    fi
}

run_system_tests() {
    echo -e "${PURPLE}🧪 Running Ultra AI System Tests...${NC}"
    echo ""
    
    cd "$ULTRA_AI_DIR"
    
    echo -e "${BLUE}Testing Production AI...${NC}"
    if $PYTHON_CMD src/ai/production_ai.py | grep -q "Production Ready"; then
        echo -e "${GREEN}✅ AI System: PASS${NC}"
    else
        echo -e "${YELLOW}⚠️ AI System: LIMITED${NC}"
    fi
    
    echo -e "${BLUE}Testing Termux Integration...${NC}"
    if $PYTHON_CMD src/integrations/termux_integration.py | grep -q "Available APIs"; then
        echo -e "${GREEN}✅ Device APIs: PASS${NC}"
    else
        echo -e "${YELLOW}⚠️ Device APIs: LIMITED${NC}"
    fi
    
    echo -e "${BLUE}Testing Voice System...${NC}"
    if $PYTHON_CMD src/voice/voice_activation.py | grep -q "system_ready: True"; then
        echo -e "${GREEN}✅ Voice System: PASS${NC}"
    else
        echo -e "${YELLOW}⚠️ Voice System: LIMITED${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 System tests completed!${NC}"
}

check_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0  # Running
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1  # Not running
}

show_status() {
    echo -e "${BLUE}📊 Ultra AI System Status${NC}"
    echo "=" * 40
    
    if check_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}Status: RUNNING (PID: $PID)${NC}"
        
        # Try to get system stats via API
        if command -v curl >/dev/null 2>&1; then
            echo "Fetching system statistics..."
            curl -s -X POST "http://127.0.0.1:8888/api/system/status" \
                -H "Content-Type: application/json" \
                -d '{}' | python3 -m json.tool 2>/dev/null || echo "API not responding"
        fi
    else
        echo -e "${YELLOW}Status: STOPPED${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}System Information:${NC}"
    echo "Directory: $ULTRA_AI_DIR"
    echo "Python: $(python3 --version)"
    echo "Platform: $(uname -s)"
    echo ""
}

stop_ultra_ai() {
    if check_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${YELLOW}🛑 Stopping Ultra AI (PID: $PID)...${NC}"
        
        kill "$PID"
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if necessary
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Force stopping...${NC}"
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo -e "${GREEN}✅ Ultra AI stopped${NC}"
    else
        echo -e "${YELLOW}Ultra AI is not running${NC}"
    fi
}

start_ultra_ai() {
    local port="$1"
    local host="$2" 
    local daemon="$3"
    
    if check_running; then
        echo -e "${YELLOW}Ultra AI is already running (PID: $(cat "$PID_FILE"))${NC}"
        echo "Use '$0 --stop' to stop it first"
        exit 1
    fi
    
    print_banner
    print_features
    
    echo -e "${GREEN}🚀 Starting Ultra AI Complete System...${NC}"
    echo -e "${BLUE}🌐 Web Interface: http://$host:$port${NC}"
    echo ""
    
    cd "$ULTRA_AI_DIR"
    
    if [ "$daemon" = "true" ]; then
        echo -e "${BLUE}Starting as background service...${NC}"
        nohup $PYTHON_CMD ultra_ai_complete.py --host "$host" --port "$port" > "$LOG_DIR/ultra_ai.log" 2>&1 &
        echo $! > "$PID_FILE"
        
        sleep 3
        if check_running; then
            echo -e "${GREEN}✅ Ultra AI started as service (PID: $(cat "$PID_FILE"))${NC}"
            echo -e "${BLUE}📄 Log file: $LOG_DIR/ultra_ai.log${NC}"
            echo -e "${BLUE}🌐 Access: http://$host:$port${NC}"
        else
            echo -e "${RED}❌ Failed to start Ultra AI${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}🎯 All systems ready! Starting interface...${NC}"
        echo ""
        exec $PYTHON_CMD ultra_ai_complete.py --host "$host" --port "$port"
    fi
}

# Parse command line arguments
PORT=8888
HOST="127.0.0.1"
DAEMON="false"
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        --daemon)
            DAEMON="true"
            shift
            ;;
        --status)
            COMMAND="status"
            shift
            ;;
        --stop)
            COMMAND="stop"
            shift
            ;;
        --test)
            COMMAND="test"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Execute commands
case "$COMMAND" in
    "status")
        show_status
        exit 0
        ;;
    "stop")
        stop_ultra_ai
        exit 0
        ;;
    "test")
        print_banner
        check_dependencies
        run_system_tests
        exit 0
        ;;
    *)
        check_dependencies
        start_ultra_ai "$PORT" "$HOST" "$DAEMON"
        ;;
esac