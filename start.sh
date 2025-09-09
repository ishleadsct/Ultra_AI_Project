#!/data/data/com.termux/files/usr/bin/bash

#
# Ultra AI Production Startup Script
# Provides multiple ways to start Ultra AI with proper error handling
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ULTRA_AI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
LOG_DIR="$ULTRA_AI_DIR/logs"
PID_FILE="$ULTRA_AI_DIR/ultra_ai.pid"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                    Ultra AI Project                      ║"
    echo "║                 Production Startup Script                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS] [MODE]"
    echo ""
    echo "MODES:"
    echo "  cli      Start interactive CLI interface (default)"
    echo "  api      Start API server"
    echo "  web      Start web interface"
    echo "  all      Start all interfaces"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --daemon        Run in daemon mode"
    echo "  -p, --port PORT     API server port (default: 8000)"
    echo "  -H, --host HOST     API server host (default: 127.0.0.1)"
    echo "  --debug             Enable debug mode"
    echo "  --log-level LEVEL   Set log level (DEBUG,INFO,WARNING,ERROR)"
    echo "  --config FILE       Use custom configuration file"
    echo "  --system-info       Show system information"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start CLI interface"
    echo "  $0 api              # Start API server"
    echo "  $0 api -p 9000      # Start API server on port 9000"
    echo "  $0 -d api           # Start API server as daemon"
    echo "  $0 --system-info    # Show system information"
}

check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo -e "${RED}Error: Python 3 is required but not found${NC}"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    
    # Check if Ultra AI main script exists
    if [ ! -f "$ULTRA_AI_DIR/ultra_ai.py" ]; then
        echo -e "${RED}Error: ultra_ai.py not found in $ULTRA_AI_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Ultra AI main script found${NC}"
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

stop_ultra_ai() {
    if check_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${YELLOW}Stopping Ultra AI (PID: $PID)...${NC}"
        kill "$PID"
        
        # Wait for process to stop
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Force killing Ultra AI...${NC}"
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo -e "${GREEN}✓ Ultra AI stopped${NC}"
    else
        echo -e "${YELLOW}Ultra AI is not running${NC}"
    fi
}

start_ultra_ai() {
    local mode="$1"
    local daemon="$2"
    local extra_args=("${@:3}")
    
    if check_running; then
        echo -e "${YELLOW}Ultra AI is already running (PID: $(cat "$PID_FILE"))${NC}"
        echo "Use './start.sh stop' to stop it first"
        exit 1
    fi
    
    echo -e "${BLUE}Starting Ultra AI in $mode mode...${NC}"
    
    cd "$ULTRA_AI_DIR"
    
    if [ "$daemon" = "true" ]; then
        # Start as daemon
        echo -e "${BLUE}Starting as daemon...${NC}"
        nohup $PYTHON_CMD ultra_ai.py --mode "$mode" "${extra_args[@]}" > "$LOG_DIR/ultra_ai.log" 2>&1 &
        echo $! > "$PID_FILE"
        
        # Wait a moment and check if process started successfully
        sleep 2
        if check_running; then
            echo -e "${GREEN}✓ Ultra AI started as daemon (PID: $(cat "$PID_FILE"))${NC}"
            echo -e "${BLUE}Log file: $LOG_DIR/ultra_ai.log${NC}"
        else
            echo -e "${RED}Error: Failed to start Ultra AI as daemon${NC}"
            exit 1
        fi
    else
        # Start in foreground
        exec $PYTHON_CMD ultra_ai.py --mode "$mode" "${extra_args[@]}"
    fi
}

# Parse command line arguments
MODE="cli"
DAEMON="false"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -d|--daemon)
            DAEMON="true"
            shift
            ;;
        -p|--port)
            EXTRA_ARGS+=("--port" "$2")
            shift 2
            ;;
        -H|--host)
            EXTRA_ARGS+=("--host" "$2")
            shift 2
            ;;
        --debug)
            EXTRA_ARGS+=("--debug")
            shift
            ;;
        --log-level)
            EXTRA_ARGS+=("--log-level" "$2")
            shift 2
            ;;
        --config)
            EXTRA_ARGS+=("--config" "$2")
            shift 2
            ;;
        --system-info)
            EXTRA_ARGS+=("--system-info")
            shift
            ;;
        stop)
            stop_ultra_ai
            exit 0
            ;;
        status)
            if check_running; then
                echo -e "${GREEN}Ultra AI is running (PID: $(cat "$PID_FILE"))${NC}"
            else
                echo -e "${YELLOW}Ultra AI is not running${NC}"
            fi
            exit 0
            ;;
        restart)
            stop_ultra_ai
            sleep 2
            start_ultra_ai "$MODE" "$DAEMON" "${EXTRA_ARGS[@]}"
            exit 0
            ;;
        cli|api|web|all)
            MODE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
print_banner

case "${MODE:-cli}" in
    cli|api|web|all)
        check_dependencies
        start_ultra_ai "$MODE" "$DAEMON" "${EXTRA_ARGS[@]}"
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        print_usage
        exit 1
        ;;
esac