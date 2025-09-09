#!/data/data/com.termux/files/usr/bin/bash

#
# Ultra AI Web GUI Launcher
# Simple launcher for the web-based GUI interface
#

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default settings
DEFAULT_PORT=8888
DEFAULT_HOST="127.0.0.1"

print_banner() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║        Ultra AI Web GUI Launcher       ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --port PORT     Port to run on (default: 8888)"
    echo "  -H, --host HOST     Host to bind to (default: 127.0.0.1)"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start on http://127.0.0.1:8888"
    echo "  $0 -p 9999          # Start on port 9999"
    echo "  $0 -H 0.0.0.0       # Bind to all interfaces"
}

# Parse arguments
PORT=$DEFAULT_PORT
HOST=$DEFAULT_HOST

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

print_banner

echo -e "${GREEN}🚀 Starting Ultra AI Futuristic GUI...${NC}"
echo -e "${BLUE}📍 URL: http://$HOST:$PORT${NC}"
echo -e "${BLUE}🎨 Features: 3D Interface, Model Switching, Voice Controls${NC}"
echo -e "${YELLOW}⏹️  Press Ctrl+C to stop${NC}"
echo ""

# Start the futuristic web GUI with fixed functionality
python3 ultra_ai_futuristic_gui.py --host "$HOST" --port "$PORT"