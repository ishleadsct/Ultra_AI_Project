#!/bin/bash

# Ultra AI Project - System Startup Script
# Starts all components of the Ultra AI system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_WORKERS="4"
DEFAULT_LOG_LEVEL="info"
VENV_PATH="venv"
PID_DIR="./temp/pids"
LOG_DIR="./logs"

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is available
is_port_available() {
    local port=$1
    if command_exists netstat; then
        ! netstat -tuln | grep ":$port " > /dev/null
    elif command_exists ss; then
        ! ss -tuln | grep ":$port " > /dev/null
    elif command_exists lsof; then
        ! lsof -i ":$port" > /dev/null 2>&1
    else
        # Fallback: try to bind to the port
        ! timeout 1 bash -c "</dev/tcp/localhost/$port" 2>/dev/null
    fi
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

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=0
    
    print_status "Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if command_exists curl; then
            if curl -s -f "http://$host:$port/health" > /dev/null 2>&1; then
                print_success "$service_name is ready!"
                return 0
            fi
        elif command_exists wget; then
            if wget -q --spider "http://$host:$port/health" 2>/dev/null; then
                print_success "$service_name is ready!"
                return 0
            fi
        else
            # Fallback: check if port is responding
            if timeout 1 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
                print_success "$service_name is ready!"
                return 0
            fi
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -lt $max_attempts ]; then
            echo -n "."
            sleep 2
        fi
    done
    
    echo
    print_warning "$service_name may not be fully ready, but continuing..."
    return 1
}

# Function to activate virtual environment
activate_venv() {
    if [ -d "$VENV_PATH" ]; then
        print_status "Activating virtual environment..."
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            source "$VENV_PATH/Scripts/activate"
        else
            source "$VENV_PATH/bin/activate"
        fi
        print_success "Virtual environment activated"
    else
        print_warning "Virtual environment not found at $VENV_PATH"
        print_warning "Please run ./scripts/setup.sh first"
    fi
}

# Function to load environment variables
load_environment() {
    if [ -f ".env" ]; then
        print_status "Loading environment variables..."
        set -a  # Automatically export all variables
        source .env
        set +a
        print_success "Environment variables loaded"
    else
        print_warning ".env file not found, using defaults"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p "$PID_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "./temp"
    mkdir -p "./data"
    
    print_success "Directories created"
}

# Function to check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command_exists python3 && ! command_exists python; then
        missing_deps+=("python3")
    fi
    
    # Check pip
    if ! command_exists pip3 && ! command_exists pip; then
        missing_deps+=("pip")
    fi
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found. Please run ./scripts/setup.sh first"
        exit 1
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install missing dependencies or run ./scripts/install_dependencies.sh"
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

# Function to start Redis (if available and configured)
start_redis() {
    if [ "$START_REDIS" = "true" ]; then
        print_status "Starting Redis server..."
        
        if command_exists redis-server; then
            if ! is_process_running "$PID_DIR/redis.pid"; then
                # Check if Redis is already running on system
                if is_port_available 6379; then
                    redis-server --daemonize yes --pidfile "$PID_DIR/redis.pid" --logfile "$LOG_DIR/redis.log"
                    sleep 2
                    if is_process_running "$PID_DIR/redis.pid"; then
                        print_success "Redis server started"
                    else
                        print_error "Failed to start Redis server"
                        return 1
                    fi
                else
                    print_warning "Redis already running on port 6379"
                fi
            else
                print_success "Redis server already running"
            fi
        else
            print_warning "Redis not installed, skipping Redis startup"
        fi
    fi
}

# Function to start database (if using PostgreSQL)
start_database() {
    if [ "$START_DATABASE" = "true" ] && [ "$DATABASE_TYPE" = "postgresql" ]; then
        print_status "Starting PostgreSQL database..."
        
        if command_exists pg_ctl; then
            if ! pg_ctl status -D "$PGDATA" > /dev/null 2>&1; then
                pg_ctl start -D "$PGDATA" -l "$LOG_DIR/postgresql.log"
                sleep 3
                print_success "PostgreSQL started"
            else
                print_success "PostgreSQL already running"
            fi
        else
            print_warning "PostgreSQL not configured, using SQLite"
        fi
    fi
}

# Function to run database migrations
run_migrations() {
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        print_status "Running database migrations..."
        
        if [ -f "src/main.py" ]; then
            python src/main.py --migrate 2>/dev/null || {
                print_warning "Migration script not ready or no migrations needed"
            }
            print_success "Database migrations completed"
        else
            print_warning "Main application not found, skipping migrations"
        fi
    fi
}

# Function to start the main application
start_main_application() {
    print_status "Starting Ultra AI main application..."
    
    # Set default values from environment or use defaults
    HOST=${HOST:-$DEFAULT_HOST}
    PORT=${PORT:-$DEFAULT_PORT}
    WORKERS=${WORKERS:-$DEFAULT_WORKERS}
    LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}
    
    # Check if port is available
    if ! is_port_available "$PORT"; then
        print_error "Port $PORT is already in use"
        print_error "Please stop the existing service or use a different port"
        exit 1
    fi
    
    # Check if already running
    if is_process_running "$PID_DIR/ultra_ai.pid"; then
        print_warning "Ultra AI application already running"
        return 0
    fi
    
    # Start application based on mode
    if [ "$MODE" = "development" ] || [ "$DEBUG" = "true" ]; then
        print_status "Starting in development mode..."
        
        # Development mode with auto-reload
        python src/main.py \
            --host "$HOST" \
            --port "$PORT" \
            --log-level "$LOG_LEVEL" \
            --reload \
            > "$LOG_DIR/ultra_ai.log" 2>&1 &
        
        echo $! > "$PID_DIR/ultra_ai.pid"
        
    elif [ "$MODE" = "production" ]; then
        print_status "Starting in production mode..."
        
        # Production mode with Gunicorn
        if command_exists gunicorn; then
            gunicorn src.main:app \
                --bind "$HOST:$PORT" \
                --workers "$WORKERS" \
                --worker-class uvicorn.workers.UvicornWorker \
                --log-level "$LOG_LEVEL" \
                --access-logfile "$LOG_DIR/access.log" \
                --error-logfile "$LOG_DIR/error.log" \
                --pid "$PID_DIR/ultra_ai.pid" \
                --daemon
        else
            print_warning "Gunicorn not installed, falling back to Uvicorn"
            uvicorn src.main:app \
                --host "$HOST" \
                --port "$PORT" \
                --workers "$WORKERS" \
                --log-level "$LOG_LEVEL" \
                > "$LOG_DIR/ultra_ai.log" 2>&1 &
            echo $! > "$PID_DIR/ultra_ai.pid"
        fi
    else
        # Default mode
        print_status "Starting with Uvicorn..."
        
        uvicorn src.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --log-level "$LOG_LEVEL" \
            > "$LOG_DIR/ultra_ai.log" 2>&1 &
        
        echo $! > "$PID_DIR/ultra_ai.pid"
    fi
    
    # Wait for application to start
    sleep 3
    
    if is_process_running "$PID_DIR/ultra_ai.pid"; then
        print_success "Ultra AI application started successfully"
        print_success "PID: $(cat $PID_DIR/ultra_ai.pid)"
        
        # Wait for service to be ready
        wait_for_service "$HOST" "$PORT" "Ultra AI API"
        
        return 0
    else
        print_error "Failed to start Ultra AI application"
        if [ -f "$LOG_DIR/ultra_ai.log" ]; then
            print_error "Check logs: tail -f $LOG_DIR/ultra_ai.log"
        fi
        return 1
    fi
}

# Function to start background workers
start_workers() {
    if [ "$START_WORKERS" = "true" ]; then
        print_status "Starting background workers..."
        
        for i in $(seq 1 "$WORKER_COUNT"); do
            if ! is_process_running "$PID_DIR/worker_$i.pid"; then
                python src/worker.py \
                    --worker-id "$i" \
                    > "$LOG_DIR/worker_$i.log" 2>&1 &
                
                echo $! > "$PID_DIR/worker_$i.pid"
                print_success "Worker $i started (PID: $!)"
            else
                print_success "Worker $i already running"
            fi
        done
    fi
}

# Function to start monitoring services
start_monitoring() {
    if [ "$START_MONITORING" = "true" ]; then
        print_status "Starting monitoring services..."
        
        # Start Prometheus (if configured)
        if [ "$PROMETHEUS_ENABLED" = "true" ] && command_exists prometheus; then
            if ! is_process_running "$PID_DIR/prometheus.pid"; then
                prometheus \
                    --config.file=./config/prometheus.yml \
                    --storage.tsdb.path=./data/prometheus \
                    --web.console.libraries=./console_libraries \
                    --web.console.templates=./consoles \
                    > "$LOG_DIR/prometheus.log" 2>&1 &
                
                echo $! > "$PID_DIR/prometheus.pid"
                print_success "Prometheus started"
            fi
        fi
        
        # Start system monitor
        if [ -f "src/monitor.py" ]; then
            if ! is_process_running "$PID_DIR/monitor.pid"; then
                python src/monitor.py \
                    > "$LOG_DIR/monitor.log" 2>&1 &
                
                echo $! > "$PID_DIR/monitor.pid"
                print_success "System monitor started"
            fi
        fi
    fi
}

# Function to display startup information
show_startup_info() {
    print_banner "============================================="
    print_banner "        Ultra AI System Started!"
    print_banner "============================================="
    echo
    print_success "🚀 Application URL: http://$HOST:$PORT"
    print_success "📚 API Documentation: http://$HOST:$PORT/docs"
    print_success "🔧 Admin Interface: http://$HOST:$PORT/admin"
    print_success "📊 Health Check: http://$HOST:$PORT/health"
    echo
    print_status "📁 Log files location: $LOG_DIR/"
    print_status "🆔 PID files location: $PID_DIR/"
    print_status "🛑 Stop system: ./scripts/stop_system.sh"
    echo
    
    # Show running processes
    print_status "Running processes:"
    if is_process_running "$PID_DIR/ultra_ai.pid"; then
        echo "  ✓ Main Application (PID: $(cat $PID_DIR/ultra_ai.pid))"
    fi
    
    if [ "$START_REDIS" = "true" ] && is_process_running "$PID_DIR/redis.pid"; then
        echo "  ✓ Redis Server (PID: $(cat $PID_DIR/redis.pid))"
    fi
    
    if [ "$START_WORKERS" = "true" ]; then
        for i in $(seq 1 "${WORKER_COUNT:-0}"); do
            if is_process_running "$PID_DIR/worker_$i.pid"; then
                echo "  ✓ Worker $i (PID: $(cat $PID_DIR/worker_$i.pid))"
            fi
        done
    fi
    
    echo
    print_success "System startup completed successfully!"
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dev                 Start in development mode with auto-reload"
    echo "  --prod                Start in production mode with Gunicorn"
    echo "  --host HOST           Bind to specific host (default: $DEFAULT_HOST)"
    echo "  --port PORT           Bind to specific port (default: $DEFAULT_PORT)"
    echo "  --workers NUM         Number of worker processes (default: $DEFAULT_WORKERS)"
    echo "  --log-level LEVEL     Set log level (default: $DEFAULT_LOG_LEVEL)"
    echo "  --no-redis            Don't start Redis server"
    echo "  --no-workers          Don't start background workers"
    echo "  --no-monitoring       Don't start monitoring services"
    echo "  --no-migrations       Don't run database migrations"
    echo "  --daemon              Start in daemon mode (background)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start with default settings"
    echo "  $0 --dev              # Start in development mode"
    echo "  $0 --prod --workers 8 # Start in production with 8 workers"
    echo "  $0 --port 8080        # Start on port 8080"
    echo ""
}

# Main function
main() {
    # Default settings
    MODE="default"
    START_REDIS="true"
    START_WORKERS="true"
    START_MONITORING="false"
    RUN_MIGRATIONS="true"
    DAEMON_MODE="false"
    WORKER_COUNT="2"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                MODE="development"
                DEBUG="true"
                shift
                ;;
            --prod)
                MODE="production"
                shift
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --no-redis)
                START_REDIS="false"
                shift
                ;;
            --no-workers)
                START_WORKERS="false"
                shift
                ;;
            --no-monitoring)
                START_MONITORING="false"
                shift
                ;;
            --no-migrations)
                RUN_MIGRATIONS="false"
                shift
                ;;
            --daemon)
                DAEMON_MODE="true"
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
    
    print_banner "============================================="
    print_banner "      Starting Ultra AI System..."
    print_banner "============================================="
    echo
    
    # Preliminary checks
    check_dependencies
    load_environment
    create_directories
    activate_venv
    
    # Start services in order
    start_redis
    start_database
    run_migrations
    start_main_application
    start_workers
    start_monitoring
    
    # Show startup information
    if [ "$DAEMON_MODE" = "false" ]; then
        show_startup_info
        
        # Keep script running in foreground mode
        print_status "Press Ctrl+C to stop the system"
        trap 'print_status "Stopping system..."; ./scripts/stop_system.sh; exit 0' INT TERM
        
        # Monitor main process
        while is_process_running "$PID_DIR/ultra_ai.pid"; do
            sleep 5
        done
        
        print_error "Main application stopped unexpectedly"
        exit 1
    else
        show_startup_info
        print_success "System started in daemon mode"
    fi
}

# Run main function with all arguments
main "$@"
