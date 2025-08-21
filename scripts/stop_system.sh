#!/bin/bash

# Ultra AI Project - System Shutdown Script
# Gracefully stops all components of the Ultra AI system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PID_DIR="./temp/pids"
LOG_DIR="./logs"
SHUTDOWN_TIMEOUT=30
FORCE_KILL_TIMEOUT=10

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to check if process is running
is_process_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

# Function to stop process gracefully
stop_process() {
    local pid_file=$1
    local process_name=$2
    local timeout=${3:-$SHUTDOWN_TIMEOUT}
    
    if [ ! -f "$pid_file" ]; then
        print_warning "$process_name: PID file not found"
        return 0
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! ps -p "$pid" > /dev/null 2>&1; then
        print_warning "$process_name: Process not running"
        rm -f "$pid_file"
        return 0
    fi
    
    print_status "Stopping $process_name (PID: $pid)..."
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || {
        print_warning "$process_name: Could not send SIGTERM"
        rm -f "$pid_file"
        return 1
    }
    
    # Wait for graceful shutdown
    local count=0
    while [ $count -lt $timeout ] && ps -p "$pid" > /dev/null 2>&1; do
        sleep 1
        count=$((count + 1))
        if [ $((count % 5)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    # Check if process stopped
    if ps -p "$pid" > /dev/null 2>&1; then
        print_warning "$process_name: Graceful shutdown timeout, forcing termination..."
        
        # Send SIGKILL for force termination
        kill -KILL "$pid" 2>/dev/null || {
            print_error "$process_name: Could not force kill process"
            return 1
        }
        
        # Wait for force kill to take effect
        local kill_count=0
        while [ $kill_count -lt $FORCE_KILL_TIMEOUT ] && ps -p "$pid" > /dev/null 2>&1; do
            sleep 1
            kill_count=$((kill_count + 1))
        done
        
        if ps -p "$pid" > /dev/null 2>&1; then
            print_error "$process_name: Could not terminate process"
            return 1
        else
            print_success "$process_name: Force terminated"
        fi
    else
        print_success "$process_name: Stopped gracefully"
    fi
    
    # Remove PID file
    rm -f "$pid_file"
    return 0
}

# Function to stop all background workers
stop_workers() {
    print_status "Stopping background workers..."
    
    local workers_found=false
    
    # Stop numbered workers
    for pid_file in "$PID_DIR"/worker_*.pid; do
        if [ -f "$pid_file" ]; then
            workers_found=true
            local worker_name=$(basename "$pid_file" .pid)
            stop_process "$pid_file" "$worker_name" 15
        fi
    done
    
    # Stop any remaining worker processes by pattern
    if pgrep -f "src/worker.py" > /dev/null; then
        print_status "Stopping remaining worker processes..."
        pkill -TERM -f "src/worker.py" || true
        sleep 5
        
        # Force kill if still running
        if pgrep -f "src/worker.py" > /dev/null; then
            print_warning "Force killing remaining worker processes..."
            pkill -KILL -f "src/worker.py" || true
        fi
    fi
    
    if [ "$workers_found" = true ]; then
        print_success "Background workers stopped"
    else
        print_warning "No background workers found"
    fi
}

# Function to stop monitoring services
stop_monitoring() {
    print_status "Stopping monitoring services..."
    
    local monitoring_found=false
    
    # Stop Prometheus
    if [ -f "$PID_DIR/prometheus.pid" ]; then
        monitoring_found=true
        stop_process "$PID_DIR/prometheus.pid" "Prometheus" 10
    fi
    
    # Stop system monitor
    if [ -f "$PID_DIR/monitor.pid" ]; then
        monitoring_found=true
        stop_process "$PID_DIR/monitor.pid" "System Monitor" 10
    fi
    
    # Stop any remaining monitoring processes
    if pgrep -f "src/monitor.py" > /dev/null; then
        print_status "Stopping remaining monitor processes..."
        pkill -TERM -f "src/monitor.py" || true
        sleep 3
        pkill -KILL -f "src/monitor.py" 2>/dev/null || true
    fi
    
    if [ "$monitoring_found" = true ]; then
        print_success "Monitoring services stopped"
    else
        print_warning "No monitoring services found"
    fi
}

# Function to stop main application
stop_main_application() {
    print_status "Stopping main application..."
    
    local app_stopped=false
    
    # Stop via PID file
    if [ -f "$PID_DIR/ultra_ai.pid" ]; then
        stop_process "$PID_DIR/ultra_ai.pid" "Ultra AI Main Application" $SHUTDOWN_TIMEOUT
        app_stopped=true
    fi
    
    # Stop any remaining uvicorn/gunicorn processes
    local processes=("uvicorn" "gunicorn")
    for process in "${processes[@]}"; do
        if pgrep -f "$process.*src.main" > /dev/null; then
            print_status "Stopping remaining $process processes..."
            pkill -TERM -f "$process.*src.main" || true
            sleep 5
            
            # Force kill if still running
            if pgrep -f "$process.*src.main" > /dev/null; then
                print_warning "Force killing remaining $process processes..."
                pkill -KILL -f "$process.*src.main" || true
            fi
            app_stopped=true
        fi
    done
    
    if [ "$app_stopped" = true ]; then
        print_success "Main application stopped"
    else
        print_warning "Main application was not running"
    fi
}

# Function to stop Redis server
stop_redis() {
    print_status "Stopping Redis server..."
    
    if [ -f "$PID_DIR/redis.pid" ]; then
        stop_process "$PID_DIR/redis.pid" "Redis Server" 10
    else
        # Try to stop Redis using redis-cli if available
        if command -v redis-cli > /dev/null; then
            if redis-cli ping > /dev/null 2>&1; then
                print_status "Stopping Redis using redis-cli..."
                redis-cli shutdown || true
                sleep 2
                print_success "Redis server stopped"
            else
                print_warning "Redis was not running"
            fi
        else
            print_warning "Redis PID file not found and redis-cli not available"
        fi
    fi
}

# Function to stop database services
stop_database() {
    if [ "$STOP_DATABASE" = "true" ] && [ "$DATABASE_TYPE" = "postgresql" ]; then
        print_status "Stopping PostgreSQL database..."
        
        if command -v pg_ctl > /dev/null && [ -n "$PGDATA" ]; then
            if pg_ctl status -D "$PGDATA" > /dev/null 2>&1; then
                pg_ctl stop -D "$PGDATA" -m fast
                print_success "PostgreSQL stopped"
            else
                print_warning "PostgreSQL was not running"
            fi
        else
            print_warning "PostgreSQL not configured or pg_ctl not available"
        fi
    fi
}

# Function to cleanup temporary files
cleanup_temp_files() {
    print_status "Cleaning up temporary files..."
    
    # Remove stale PID files
    if [ -d "$PID_DIR" ]; then
        find "$PID_DIR" -name "*.pid" -type f -delete 2>/dev/null || true
    fi
    
    # Clean up socket files
    find ./temp -name "*.sock" -type f -delete 2>/dev/null || true
    
    # Clean up old log files if requested
    if [ "$CLEANUP_LOGS" = "true" ]; then
        find "$LOG_DIR" -name "*.log.*" -mtime +7 -delete 2>/dev/null || true
    fi
    
    print_success "Temporary files cleaned up"
}

# Function to show running processes
show_running_processes() {
    print_status "Checking for remaining Ultra AI processes..."
    
    local patterns=("src/main.py" "src/worker.py" "src/monitor.py" "uvicorn.*src.main" "gunicorn.*src.main")
    local found_processes=false
    
    for pattern in "${patterns[@]}"; do
        local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            found_processes=true
            print_warning "Found remaining processes matching '$pattern':"
            echo "$pids" | while read -r pid; do
                if [ -n "$pid" ]; then
                    ps -p "$pid" -o pid,ppid,cmd 2>/dev/null || true
                fi
            done
        fi
    done
    
    if [ "$found_processes" = false ]; then
        print_success "No remaining Ultra AI processes found"
    else
        print_warning "Some processes may still be running. Use --force to kill them."
    fi
}

# Function to force kill all related processes
force_kill_all() {
    print_warning "Force killing all Ultra AI related processes..."
    
    local patterns=("src/main.py" "src/worker.py" "src/monitor.py" "uvicorn.*src.main" "gunicorn.*src.main")
    
    for pattern in "${patterns[@]}"; do
        if pgrep -f "$pattern" > /dev/null; then
            print_status "Force killing processes matching '$pattern'..."
            pkill -KILL -f "$pattern" 2>/dev/null || true
        fi
    done
    
    # Clean up all PID files
    if [ -d "$PID_DIR" ]; then
        rm -f "$PID_DIR"/*.pid
    fi
    
    print_success "Force kill completed"
}

# Function to check system ports
check_ports() {
    if [ "$CHECK_PORTS" = "true" ]; then
        print_status "Checking system ports..."
        
        local ports=("8000" "6379" "5432")
        
        for port in "${ports[@]}"; do
            if command -v netstat > /dev/null; then
                if netstat -tuln | grep ":$port " > /dev/null; then
                    print_warning "Port $port is still in use"
                    netstat -tulpn | grep ":$port " || true
                fi
            elif command -v ss > /dev/null; then
                if ss -tuln | grep ":$port " > /dev/null; then
                    print_warning "Port $port is still in use"
                    ss -tulpn | grep ":$port " || true
                fi
            fi
        done
    fi
}

# Function to display shutdown summary
show_shutdown_summary() {
    print_banner "============================================="
    print_banner "      Ultra AI System Shutdown Complete"
    print_banner "============================================="
    echo
    print_success "All components have been stopped"
    echo
    print_status "📁 Log files preserved in: $LOG_DIR/"
    print_status "🔄 Restart system: ./scripts/start_system.sh"
    print_status "🧹 Clean logs: rm -f $LOG_DIR/*.log"
    echo
    
    if [ "$KEEP_DATA" = "false" ]; then
        print_warning "Note: Use --keep-data to preserve application data"
    fi
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --force           Force kill all processes immediately"
    echo "  --timeout SECS    Set shutdown timeout in seconds (default: $SHUTDOWN_TIMEOUT)"
    echo "  --no-redis        Don't stop Redis server"
    echo "  --no-database     Don't stop database services"
    echo "  --cleanup-logs    Remove old log files (>7 days)"
    echo "  --keep-data       Don't clean up temporary data"
    echo "  --check-ports     Check if ports are still in use after shutdown"
    echo "  --quiet           Suppress non-error output"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Graceful shutdown"
    echo "  $0 --force        # Force immediate shutdown"
    echo "  $0 --timeout 60   # Wait up to 60 seconds for graceful shutdown"
    echo "  $0 --cleanup-logs # Clean up old log files during shutdown"
    echo ""
}

# Main function
main() {
    # Default settings
    FORCE_SHUTDOWN="false"
    STOP_REDIS="true"
    STOP_DATABASE="false"
    CLEANUP_LOGS="false"
    KEEP_DATA="true"
    CHECK_PORTS="false"
    QUIET="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_SHUTDOWN="true"
                shift
                ;;
            --timeout)
                SHUTDOWN_TIMEOUT="$2"
                shift 2
                ;;
            --no-redis)
                STOP_REDIS="false"
                shift
                ;;
            --no-database)
                STOP_DATABASE="false"
                shift
                ;;
            --cleanup-logs)
                CLEANUP_LOGS="true"
                shift
                ;;
            --keep-data)
                KEEP_DATA="true"
                shift
                ;;
            --check-ports)
                CHECK_PORTS="true"
                shift
                ;;
            --quiet)
                QUIET="true"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Suppress output if quiet mode
    if [ "$QUIET" = "true" ]; then
        exec 1>/dev/null 2>&1
    fi
    
    print_banner "============================================="
    print_banner "      Stopping Ultra AI System..."
    print_banner "============================================="
    echo
    
    # Load environment if available
    if [ -f ".env" ]; then
        set -a
        source .env 2>/dev/null || true
        set +a
    fi
    
    # Create directories if they don't exist
    mkdir -p "$PID_DIR" "$LOG_DIR"
    
    if [ "$FORCE_SHUTDOWN" = "true" ]; then
        force_kill_all
    else
        # Graceful shutdown in reverse order
        stop_monitoring
        stop_workers
        stop_main_application
        
        if [ "$STOP_REDIS" = "true" ]; then
            stop_redis
        fi
        
        stop_database
    fi
    
    # Cleanup
    cleanup_temp_files
    
    # Post-shutdown checks
    if [ "$FORCE_SHUTDOWN" = "false" ]; then
        show_running_processes
    fi
    
    check_ports
    
    # Show summary
    if [ "$QUIET" = "false" ]; then
        show_shutdown_summary
    fi
}

# Handle script interruption
trap 'print_error "Shutdown interrupted"; exit 1' INT TERM

# Run main function with all arguments
main "$@"
