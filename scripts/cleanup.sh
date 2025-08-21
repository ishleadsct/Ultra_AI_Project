#!/bin/bash

# Ultra AI Project - System Cleanup Script
# Cleans up temporary files, logs, cache, and other system debris

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
LOG_DIR="./logs"
TEMP_DIR="./temp"
CACHE_DIR="./cache"
MODELS_DIR="./models"
DATA_DIR="./data"
BACKUP_DIR="./backups"
PID_DIR="./temp/pids"

# Default settings
DEEP_CLEAN="false"
KEEP_LOGS="false"
KEEP_MODELS="true"
KEEP_DATA="true"
CLEAN_PYTHON="false"
CLEAN_NODE="false"
CLEAN_DOCKER="false"
DRY_RUN="false"
VERBOSE="false"

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

print_verbose() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${CYAN}[VERBOSE]${NC} $1"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get directory size
get_dir_size() {
    local dir=$1
    if [ -d "$dir" ] && command_exists du; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

# Function to count files in directory
count_files() {
    local dir=$1
    if [ -d "$dir" ]; then
        find "$dir" -type f 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Function to safely remove files/directories
safe_remove() {
    local target=$1
    local description=$2
    
    if [ ! -e "$target" ]; then
        print_verbose "Skipping $description - not found: $target"
        return 0
    fi
    
    local size=$(get_dir_size "$target")
    local count=$(count_files "$target")
    
    if [ "$DRY_RUN" = "true" ]; then
        print_status "DRY RUN: Would remove $description ($size, $count files): $target"
        return 0
    fi
    
    print_status "Removing $description ($size, $count files): $target"
    
    if [ -d "$target" ]; then
        rm -rf "$target" 2>/dev/null || {
            print_warning "Could not remove directory: $target"
            return 1
        }
    else
        rm -f "$target" 2>/dev/null || {
            print_warning "Could not remove file: $target"
            return 1
        }
    fi
    
    print_success "Removed $description"
    return 0
}

# Function to clean temporary files
clean_temp_files() {
    print_banner "Cleaning Temporary Files..."
    
    local cleaned=0
    
    # Clean main temp directory
    if [ -d "$TEMP_DIR" ]; then
        for item in "$TEMP_DIR"/*; do
            if [ -e "$item" ] && [[ ! "$item" =~ pids$ ]]; then
                safe_remove "$item" "temporary files"
                cleaned=$((cleaned + 1))
            fi
        done
    fi
    
    # Clean system temp files
    local temp_patterns=(
        "*.tmp"
        "*.temp"
        "*~"
        ".#*"
        "#*#"
        "*.bak"
        "*.swp"
        "*.swo"
        ".DS_Store"
        "Thumbs.db"
        "*.pid"
    )
    
    for pattern in "${temp_patterns[@]}"; do
        while IFS= read -r -d '' file; do
            if [[ ! "$file" =~ /temp/pids/ ]]; then
                safe_remove "$file" "temporary file ($pattern)"
                cleaned=$((cleaned + 1))
            fi
        done < <(find . -name "$pattern" -type f -print0 2>/dev/null)
    done
    
    print_success "Cleaned $cleaned temporary file groups"
}

# Function to clean log files
clean_log_files() {
    if [ "$KEEP_LOGS" = "true" ]; then
        print_status "Skipping log cleanup (--keep-logs enabled)"
        return 0
    fi
    
    print_banner "Cleaning Log Files..."
    
    local cleaned=0
    
    if [ -d "$LOG_DIR" ]; then
        # Remove old log files (older than 7 days by default)
        local log_age=${LOG_RETENTION_DAYS:-7}
        
        print_status "Removing log files older than $log_age days..."
        
        find "$LOG_DIR" -name "*.log" -type f -mtime +$log_age -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "old log file"
            cleaned=$((cleaned + 1))
        done
        
        # Remove empty log files
        find "$LOG_DIR" -name "*.log" -type f -empty -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "empty log file"
            cleaned=$((cleaned + 1))
        done
        
        # Compress large log files if gzip is available
        if command_exists gzip && [ "$DEEP_CLEAN" = "true" ]; then
            find "$LOG_DIR" -name "*.log" -type f -size +10M -print0 2>/dev/null | while IFS= read -r -d '' file; do
                if [ "$DRY_RUN" = "false" ]; then
                    print_status "Compressing large log file: $file"
                    gzip "$file" 2>/dev/null || print_warning "Failed to compress: $file"
                else
                    print_status "DRY RUN: Would compress large log file: $file"
                fi
            done
        fi
        
        # Clean rotated logs
        find "$LOG_DIR" -name "*.log.*" -type f -mtime +$log_age -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "rotated log file"
            cleaned=$((cleaned + 1))
        done
    fi
    
    print_success "Cleaned $cleaned log files"
}

# Function to clean cache files
clean_cache_files() {
    print_banner "Cleaning Cache Files..."
    
    local cleaned=0
    
    # Clean application cache
    if [ -d "$CACHE_DIR" ]; then
        safe_remove "$CACHE_DIR/*" "application cache"
        cleaned=$((cleaned + 1))
    fi
    
    # Clean Python cache
    if [ "$CLEAN_PYTHON" = "true" ] || [ "$DEEP_CLEAN" = "true" ]; then
        print_status "Cleaning Python cache files..."
        
        # Remove __pycache__ directories
        find . -name "__pycache__" -type d -print0 2>/dev/null | while IFS= read -r -d '' dir; do
            safe_remove "$dir" "Python cache directory"
            cleaned=$((cleaned + 1))
        done
        
        # Remove .pyc files
        find . -name "*.pyc" -type f -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "Python compiled file"
            cleaned=$((cleaned + 1))
        done
        
        # Remove .pyo files
        find . -name "*.pyo" -type f -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "Python optimized file"
            cleaned=$((cleaned + 1))
        done
        
        # Clean pip cache
        if command_exists pip; then
            if [ "$DRY_RUN" = "false" ]; then
                print_status "Cleaning pip cache..."
                pip cache purge --quiet 2>/dev/null || print_warning "Could not clean pip cache"
            else
                print_status "DRY RUN: Would clean pip cache"
            fi
        fi
    fi
    
    # Clean Node.js cache
    if [ "$CLEAN_NODE" = "true" ] || [ "$DEEP_CLEAN" = "true" ]; then
        if command_exists npm; then
            if [ "$DRY_RUN" = "false" ]; then
                print_status "Cleaning npm cache..."
                npm cache clean --force --silent 2>/dev/null || print_warning "Could not clean npm cache"
            else
                print_status "DRY RUN: Would clean npm cache"
            fi
        fi
        
        # Remove node_modules in development
        if [ -d "node_modules" ] && [ "$DEEP_CLEAN" = "true" ]; then
            safe_remove "node_modules" "Node.js modules"
            cleaned=$((cleaned + 1))
        fi
    fi
    
    print_success "Cleaned $cleaned cache groups"
}

# Function to clean AI model cache
clean_model_cache() {
    if [ "$KEEP_MODELS" = "true" ]; then
        print_status "Skipping model cache cleanup (--keep-models enabled)"
        return 0
    fi
    
    print_banner "Cleaning AI Model Cache..."
    
    local cleaned=0
    
    # Clean Hugging Face cache
    if [ -d "$MODELS_DIR/huggingface" ]; then
        if command_exists python; then
            cat > "$TEMP_DIR/clean_hf_cache.py" << 'PYEOF'
try:
    from huggingface_hub import scan_cache_dir
    import sys
    
    try:
        cache_info = scan_cache_dir()
        print(f"Current cache size: {cache_info.size_on_disk_str}")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
            print("DRY RUN: Would clean Hugging Face cache")
        else:
            delete_strategy = cache_info.delete_revisions()
            print(f"Will free: {delete_strategy.expected_freed_size_str}")
            delete_strategy.execute()
            print("✓ Hugging Face cache cleaned")
            
    except Exception as e:
        print(f"Could not clean Hugging Face cache: {e}")
        
except ImportError:
    print("huggingface_hub not available for cache cleanup")
PYEOF
            
            if [ "$DRY_RUN" = "true" ]; then
                python "$TEMP_DIR/clean_hf_cache.py" --dry-run
            else
                python "$TEMP_DIR/clean_hf_cache.py"
            fi
            cleaned=$((cleaned + 1))
        fi
    fi
    
    # Clean model download cache
    if [ -d "$MODELS_DIR/cache" ]; then
        safe_remove "$MODELS_DIR/cache/*" "model download cache"
        cleaned=$((cleaned + 1))
    fi
    
    # Clean temporary model files
    find "$MODELS_DIR" -name "*.tmp" -o -name "*.partial" -type f -print0 2>/dev/null | while IFS= read -r -d '' file; do
        safe_remove "$file" "temporary model file"
        cleaned=$((cleaned + 1))
    done
    
    print_success "Cleaned $cleaned model cache groups"
}

# Function to clean database files
clean_database_files() {
    if [ "$KEEP_DATA" = "true" ]; then
        print_status "Skipping database cleanup (--keep-data enabled)"
        return 0
    fi
    
    print_banner "Cleaning Database Files..."
    
    local cleaned=0
    
    # Clean SQLite WAL and SHM files
    find "$DATA_DIR" -name "*.db-wal" -o -name "*.db-shm" -type f -print0 2>/dev/null | while IFS= read -r -d '' file; do
        safe_remove "$file" "SQLite temporary file"
        cleaned=$((cleaned + 1))
    done
    
    # Clean old backup files
    if [ -d "$BACKUP_DIR" ]; then
        local backup_age=${BACKUP_RETENTION_DAYS:-30}
        print_status "Removing backups older than $backup_age days..."
        
        find "$BACKUP_DIR" -name "*.backup" -o -name "*.sql.gz" -type f -mtime +$backup_age -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "old backup file"
            cleaned=$((cleaned + 1))
        done
    fi
    
    print_success "Cleaned $cleaned database files"
}

# Function to clean Docker resources
clean_docker_resources() {
    if [ "$CLEAN_DOCKER" = "false" ] && [ "$DEEP_CLEAN" = "false" ]; then
        return 0
    fi
    
    if ! command_exists docker; then
        print_verbose "Docker not available, skipping Docker cleanup"
        return 0
    fi
    
    print_banner "Cleaning Docker Resources..."
    
    if [ "$DRY_RUN" = "true" ]; then
        print_status "DRY RUN: Would clean Docker resources"
        docker system df 2>/dev/null || true
        return 0
    fi
    
    print_status "Cleaning Docker images, containers, and volumes..."
    
    # Remove stopped containers
    docker container prune -f 2>/dev/null || print_warning "Could not prune containers"
    
    # Remove unused images
    docker image prune -f 2>/dev/null || print_warning "Could not prune images"
    
    # Remove unused volumes
    docker volume prune -f 2>/dev/null || print_warning "Could not prune volumes"
    
    # Remove unused networks
    docker network prune -f 2>/dev/null || print_warning "Could not prune networks"
    
    if [ "$DEEP_CLEAN" = "true" ]; then
        print_status "Deep cleaning Docker system..."
        docker system prune -af 2>/dev/null || print_warning "Could not perform deep Docker cleanup"
    fi
    
    print_success "Docker cleanup completed"
}

# Function to clean version control files
clean_vcs_files() {
    if [ "$DEEP_CLEAN" = "false" ]; then
        return 0
    fi
    
    print_banner "Cleaning Version Control Files..."
    
    local cleaned=0
    
    # Clean Git files
    if [ -d ".git" ]; then
        if [ "$DRY_RUN" = "false" ]; then
            print_status "Cleaning Git cache..."
            git gc --quiet 2>/dev/null || print_warning "Could not clean Git cache"
            git prune --quiet 2>/dev/null || print_warning "Could not prune Git objects"
        else
            print_status "DRY RUN: Would clean Git cache and prune objects"
        fi
        cleaned=$((cleaned + 1))
    fi
    
    # Remove editor backup files
    local editor_patterns=(
        "*.orig"
        "*.rej"
        "*~"
        ".#*"
        "#*#"
    )
    
    for pattern in "${editor_patterns[@]}"; do
        find . -name "$pattern" -type f -print0 2>/dev/null | while IFS= read -r -d '' file; do
            safe_remove "$file" "editor backup file"
            cleaned=$((cleaned + 1))
        done
    done
    
    print_success "Cleaned $cleaned version control file groups"
}

# Function to clean system-wide resources
clean_system_resources() {
    if [ "$DEEP_CLEAN" = "false" ]; then
        return 0
    fi
    
    print_banner "Cleaning System Resources..."
    
    # Clean system package managers (if available)
    if command_exists apt; then
        if [ "$DRY_RUN" = "false" ]; then
            print_status "Cleaning APT cache..."
            sudo apt autoremove -y --quiet 2>/dev/null || print_warning "Could not autoremove packages"
            sudo apt autoclean --quiet 2>/dev/null || print_warning "Could not clean APT cache"
        else
            print_status "DRY RUN: Would clean APT cache and autoremove packages"
        fi
    fi
    
    if command_exists yum; then
        if [ "$DRY_RUN" = "false" ]; then
            print_status "Cleaning YUM cache..."
            sudo yum clean all --quiet 2>/dev/null || print_warning "Could not clean YUM cache"
        else
            print_status "DRY RUN: Would clean YUM cache"
        fi
    fi
    
    if command_exists brew; then
        if [ "$DRY_RUN" = "false" ]; then
            print_status "Cleaning Homebrew cache..."
            brew cleanup --quiet 2>/dev/null || print_warning "Could not clean Homebrew cache"
        else
            print_status "DRY RUN: Would clean Homebrew cache"
        fi
    fi
    
    print_success "System cleanup completed"
}

# Function to show cleanup summary
show_cleanup_summary() {
    print_banner "============================================="
    print_banner "      Cleanup Summary"
    print_banner "============================================="
    echo
    
    # Calculate space freed
    local space_info=""
    if command_exists du; then
        local current_size=$(du -sh . 2>/dev/null | cut -f1)
        space_info=" (Current size: $current_size)"
    fi
    
    print_success "Cleanup completed successfully$space_info"
    echo
    
    # Show remaining directory sizes
    print_status "Remaining directory sizes:"
    local dirs=("$LOG_DIR" "$TEMP_DIR" "$CACHE_DIR" "$MODELS_DIR" "$DATA_DIR")
    
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            local size=$(get_dir_size "$dir")
            local count=$(count_files "$dir")
            print_status "  $(basename "$dir"): $size ($count files)"
        fi
    done
    
    echo
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "This was a dry run. Use without --dry-run to actually perform cleanup."
    fi
    
    print_status "💡 Tips:"
    print_status "  - Run with --deep for more thorough cleanup"
    print_status "  - Use --keep-logs to preserve log files"
    print_status "  - Use --verbose for detailed output"
    print_status "  - Schedule regular cleanup with cron"
    echo
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --deep              Perform deep cleanup (includes system resources)"
    echo "  --keep-logs         Don't remove log files"
    echo "  --keep-models       Don't clean model cache"
    echo "  --keep-data         Don't clean database files"
    echo "  --clean-python      Clean Python cache files"
    echo "  --clean-node        Clean Node.js cache files"
    echo "  --clean-docker      Clean Docker resources"
    echo "  --dry-run           Show what would be cleaned without doing it"
    echo "  --verbose           Show detailed output"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LOG_RETENTION_DAYS     Days to keep log files (default: 7)"
    echo "  BACKUP_RETENTION_DAYS  Days to keep backup files (default: 30)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Basic cleanup"
    echo "  $0 --deep --verbose   # Deep cleanup with detailed output"
    echo "  $0 --dry-run          # See what would be cleaned"
    echo "  $0 --keep-logs        # Clean everything except logs"
    echo "  $0 --clean-docker     # Include Docker resource cleanup"
    echo ""
    echo "Cleanup categories:"
    echo "  - Temporary files (*.tmp, *.bak, etc.)"
    echo "  - Log files (older than retention period)"
    echo "  - Cache files (application, Python, Node.js)"
    echo "  - AI model cache (Hugging Face, etc.)"
    echo "  - Database temporary files"
    echo "  - Docker resources (with --clean-docker)"
    echo "  - System package caches (with --deep)"
    echo ""
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --deep)
                DEEP_CLEAN="true"
                CLEAN_PYTHON="true"
                CLEAN_NODE="true"
                shift
                ;;
            --keep-logs)
                KEEP_LOGS="true"
                shift
                ;;
            --keep-models)
                KEEP_MODELS="true"
                shift
                ;;
            --keep-data)
                KEEP_DATA="true"
                shift
                ;;
            --clean-python)
                CLEAN_PYTHON="true"
                shift
                ;;
            --clean-node)
                CLEAN_NODE="true"
                shift
                ;;
            --clean-docker)
                CLEAN_DOCKER="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --verbose)
                VERBOSE="true"
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
    print_banner "      Ultra AI Project - System Cleanup"
    print_banner "============================================="
    echo
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "DRY RUN MODE - No files will actually be removed"
        echo
    fi
    
    # Create temp directory for cleanup scripts
    mkdir -p "$TEMP_DIR"
    
    # Perform cleanup operations
    clean_temp_files
    clean_log_files
    clean_cache_files
    clean_model_cache
    clean_database_files
    clean_docker_resources
    clean_vcs_files
    clean_system_resources
    
    # Show summary
    show_cleanup_summary
}

# Handle script interruption
trap 'print_error "Cleanup interrupted"; exit 1' INT TERM

# Run main function with all arguments
main "$@"
