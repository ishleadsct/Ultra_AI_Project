#!/bin/bash

# Ultra AI Project - Dependency Installation Script
# Installs all required system and Python dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            print_status "Detected Debian/Ubuntu system"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
            print_status "Detected RedHat/CentOS/Fedora system"
        elif [ -f /etc/arch-release ]; then
            OS="arch"
            print_status "Detected Arch Linux system"
        else
            OS="linux"
            print_status "Detected generic Linux system"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS system"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
        print_status "Detected Windows system"
    else
        OS="unknown"
        print_warning "Unknown operating system: $OSTYPE"
    fi
}

# Function to install system dependencies on Debian/Ubuntu
install_debian_dependencies() {
    print_status "Installing system dependencies for Debian/Ubuntu..."
    
    # Update package list
    sudo apt-get update
    
    # Install basic build tools and libraries
    sudo apt-get install -y \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        curl \
        wget \
        git \
        unzip \
        zip \
        tree \
        htop \
        nano \
        vim
    
    # Install Python and related tools
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-setuptools \
        python3-wheel
    
    # Install development libraries
    sudo apt-get install -y \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        libpng-dev \
        libfreetype6-dev \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran
    
    # Install database dependencies
    sudo apt-get install -y \
        sqlite3 \
        libsqlite3-dev \
        postgresql-client \
        libpq-dev
    
    # Install Redis (optional but recommended)
    if ! command_exists redis-server; then
        sudo apt-get install -y redis-server
        sudo systemctl enable redis-server
        print_success "Redis installed and enabled"
    fi
    
    print_success "Debian/Ubuntu system dependencies installed"
}

# Function to install system dependencies on RedHat/CentOS/Fedora
install_redhat_dependencies() {
    print_status "Installing system dependencies for RedHat/CentOS/Fedora..."
    
    # Detect package manager
    if command_exists dnf; then
        PKG_MGR="dnf"
    elif command_exists yum; then
        PKG_MGR="yum"
    else
        print_error "No compatible package manager found"
        exit 1
    fi
    
    # Update packages
    sudo $PKG_MGR update -y
    
    # Install development tools
    sudo $PKG_MGR groupinstall -y "Development Tools"
    sudo $PKG_MGR install -y \
        curl \
        wget \
        git \
        unzip \
        zip \
        tree \
        htop \
        nano \
        vim
    
    # Install Python and related tools
    sudo $PKG_MGR install -y \
        python3 \
        python3-pip \
        python3-devel \
        python3-setuptools \
        python3-wheel
    
    # Install development libraries
    sudo $PKG_MGR install -y \
        openssl-devel \
        libffi-devel \
        libxml2-devel \
        libxslt-devel \
        libjpeg-devel \
        libpng-devel \
        freetype-devel \
        blas-devel \
        lapack-devel \
        atlas-devel
    
    # Install database dependencies
    sudo $PKG_MGR install -y \
        sqlite \
        sqlite-devel \
        postgresql \
        postgresql-devel
    
    # Install Redis
    if ! command_exists redis-server; then
        sudo $PKG_MGR install -y redis
        sudo systemctl enable redis
        print_success "Redis installed and enabled"
    fi
    
    print_success "RedHat/CentOS/Fedora system dependencies installed"
}

# Function to install system dependencies on Arch Linux
install_arch_dependencies() {
    print_status "Installing system dependencies for Arch Linux..."
    
    # Update package database
    sudo pacman -Sy
    
    # Install base development tools
    sudo pacman -S --needed --noconfirm \
        base-devel \
        curl \
        wget \
        git \
        unzip \
        zip \
        tree \
        htop \
        nano \
        vim
    
    # Install Python and related tools
    sudo pacman -S --needed --noconfirm \
        python \
        python-pip \
        python-virtualenv \
        python-setuptools \
        python-wheel
    
    # Install development libraries
    sudo pacman -S --needed --noconfirm \
        openssl \
        libffi \
        libxml2 \
        libxslt \
        libjpeg-turbo \
        libpng \
        freetype2 \
        blas \
        lapack
    
    # Install database dependencies
    sudo pacman -S --needed --noconfirm \
        sqlite \
        postgresql \
        postgresql-libs
    
    # Install Redis
    if ! command_exists redis-server; then
        sudo pacman -S --needed --noconfirm redis
        sudo systemctl enable redis
        print_success "Redis installed and enabled"
    fi
    
    print_success "Arch Linux system dependencies installed"
}

# Function to install system dependencies on macOS
install_macos_dependencies() {
    print_status "Installing system dependencies for macOS..."
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for current session
        if [[ -d "/opt/homebrew" ]]; then
            export PATH="/opt/homebrew/bin:$PATH"
        else
            export PATH="/usr/local/bin:$PATH"
        fi
    fi
    
    # Update Homebrew
    brew update
    
    # Install basic tools
    brew install \
        curl \
        wget \
        git \
        unzip \
        zip \
        tree \
        htop \
        nano \
        vim
    
    # Install Python
    brew install python@3.11
    
    # Install development libraries
    brew install \
        openssl \
        libffi \
        libxml2 \
        libxslt \
        jpeg \
        libpng \
        freetype \
        openblas \
        lapack
    
    # Install database tools
    brew install \
        sqlite \
        postgresql@14
    
    # Install Redis
    if ! command_exists redis-server; then
        brew install redis
        brew services start redis
        print_success "Redis installed and started"
    fi
    
    print_success "macOS system dependencies installed"
}

# Function to install system dependencies on Windows
install_windows_dependencies() {
    print_status "Installing system dependencies for Windows..."
    
    # Check if chocolatey is installed
    if ! command_exists choco; then
        print_warning "Chocolatey not found. Please install it manually from https://chocolatey.org/"
        print_warning "Or use Windows Subsystem for Linux (WSL) for better compatibility"
        return
    fi
    
    # Install basic tools
    choco install -y \
        git \
        python3 \
        sqlite \
        redis-64 \
        wget \
        curl \
        7zip
    
    print_success "Windows system dependencies installed"
}

# Function to install Node.js and development tools
install_nodejs() {
    if [ "$INSTALL_NODEJS" = true ]; then
        print_status "Installing Node.js and development tools..."
        
        case $OS in
            "debian")
                # Install Node.js via NodeSource repository
                curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
                sudo apt-get install -y nodejs
                ;;
            "redhat")
                # Install Node.js via NodeSource repository
                curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
                sudo $PKG_MGR install -y nodejs npm
                ;;
            "arch")
                sudo pacman -S --needed --noconfirm nodejs npm
                ;;
            "macos")
                brew install node
                ;;
            "windows")
                choco install -y nodejs
                ;;
        esac
        
        if command_exists npm; then
            print_success "Node.js and npm installed"
            
            # Install global packages for development
            npm install -g \
                yarn \
                typescript \
                @types/node \
                prettier \
                eslint
                
            print_success "Node.js development tools installed"
        fi
    fi
}

# Function to install Docker
install_docker() {
    if [ "$INSTALL_DOCKER" = true ]; then
        print_status "Installing Docker..."
        
        case $OS in
            "debian")
                # Install Docker on Debian/Ubuntu
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
                sudo apt-get update
                sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
                ;;
            "redhat")
                # Install Docker on RedHat/CentOS
                sudo $PKG_MGR install -y yum-utils
                sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
                sudo $PKG_MGR install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
                ;;
            "arch")
                sudo pacman -S --needed --noconfirm docker docker-compose
                ;;
            "macos")
                print_warning "Please install Docker Desktop for Mac from https://www.docker.com/products/docker-desktop"
                return
                ;;
            "windows")
                print_warning "Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop"
                return
                ;;
        esac
        
        # Start and enable Docker
        if [[ "$OS" != "macos" && "$OS" != "windows" ]]; then
            sudo systemctl start docker
            sudo systemctl enable docker
            
            # Add current user to docker group
            sudo usermod -aG docker $USER
            print_success "Docker installed. Please log out and log back in to use Docker without sudo"
        fi
    fi
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Ensure pip is up to date
    python3 -m pip install --upgrade pip
    
    # Install core Python packages
    python3 -m pip install --user \
        virtualenv \
        pipenv \
        wheel \
        setuptools \
        build \
        twine
    
    # Install additional development tools if requested
    if [ "$INSTALL_DEV" = true ]; then
        python3 -m pip install --user \
            black \
            flake8 \
            mypy \
            pytest \
            pytest-cov \
            pytest-asyncio \
            pre-commit \
            bandit \
            safety
    fi
    
    print_success "Python dependencies installed"
}

# Function to install GPU support (NVIDIA)
install_gpu_support() {
    if [ "$INSTALL_GPU" = true ]; then
        print_status "Installing GPU support..."
        
        case $OS in
            "debian")
                # Install NVIDIA drivers and CUDA
                if lspci | grep -i nvidia > /dev/null; then
                    print_status "NVIDIA GPU detected, installing drivers..."
                    sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit
                    
                    # Add CUDA to PATH
                    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
                    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
                fi
                ;;
            "redhat")
                if lspci | grep -i nvidia > /dev/null; then
                    print_status "NVIDIA GPU detected, installing drivers..."
                    sudo $PKG_MGR install -y nvidia-driver cuda-toolkit
                fi
                ;;
            "arch")
                if lspci | grep -i nvidia > /dev/null; then
                    print_status "NVIDIA GPU detected, installing drivers..."
                    sudo pacman -S --needed --noconfirm nvidia nvidia-utils cuda
                fi
                ;;
            "macos")
                print_warning "GPU support on macOS requires Metal Performance Shaders"
                ;;
        esac
        
        print_success "GPU support installation completed"
    fi
}

# Function to verify installations
verify_installation() {
    print_status "Verifying installations..."
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python: $PYTHON_VERSION"
    else
        print_error "Python 3 not found"
    fi
    
    # Check pip
    if command_exists pip3; then
        PIP_VERSION=$(pip3 --version)
        print_success "pip: $PIP_VERSION"
    else
        print_error "pip3 not found"
    fi
    
    # Check Git
    if command_exists git; then
        GIT_VERSION=$(git --version)
        print_success "Git: $GIT_VERSION"
    else
        print_error "Git not found"
    fi
    
    # Check optional components
    if [ "$INSTALL_DOCKER" = true ] && command_exists docker; then
        DOCKER_VERSION=$(docker --version)
        print_success "Docker: $DOCKER_VERSION"
    fi
    
    if [ "$INSTALL_NODEJS" = true ] && command_exists node; then
        NODE_VERSION=$(node --version)
        print_success "Node.js: $NODE_VERSION"
    fi
    
    if command_exists redis-server; then
        print_success "Redis: Available"
    fi
}

# Function to display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dev         Install development tools and dependencies"
    echo "  --docker      Install Docker and Docker Compose"
    echo "  --nodejs      Install Node.js and development tools"
    echo "  --gpu         Install GPU support (NVIDIA CUDA)"
    echo "  --minimal     Install only essential dependencies"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Install basic dependencies"
    echo "  $0 --dev --docker     # Install with development tools and Docker"
    echo "  $0 --minimal          # Minimal installation"
    echo ""
}

# Main function
main() {
    echo "============================================="
    echo "   Ultra AI Project - Dependency Installer"
    echo "============================================="
    echo
    
    # Default options
    INSTALL_DEV=false
    INSTALL_DOCKER=false
    INSTALL_NODEJS=false
    INSTALL_GPU=false
    MINIMAL_INSTALL=false
    
    # Parse command line arguments
    for arg in "$@"; do
        case $arg in
            --dev)
                INSTALL_DEV=true
                ;;
            --docker)
                INSTALL_DOCKER=true
                ;;
            --nodejs)
                INSTALL_NODEJS=true
                ;;
            --gpu)
                INSTALL_GPU=true
                ;;
            --minimal)
                MINIMAL_INSTALL=true
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Detect operating system
    detect_os
    
    # Check for sudo access
    if [[ "$OS" != "windows" && "$OS" != "macos" ]]; then
        if ! sudo -v; then
            print_error "This script requires sudo access for system package installation"
            exit 1
        fi
    fi
    
    # Install system dependencies based on OS
    if [ "$MINIMAL_INSTALL" = false ]; then
        case $OS in
            "debian")
                install_debian_dependencies
                ;;
            "redhat")
                install_redhat_dependencies
                ;;
            "arch")
                install_arch_dependencies
                ;;
            "macos")
                install_macos_dependencies
                ;;
            "windows")
                install_windows_dependencies
                ;;
            *)
                print_warning "Unsupported operating system, skipping system dependencies"
                ;;
        esac
    fi
    
    # Install Python dependencies
    install_python_dependencies
    
    # Install optional components
    install_nodejs
    install_docker
    install_gpu_support
    
    # Verify installation
    verify_installation
    
    print_success "Dependency installation completed!"
    echo
    echo "Next steps:"
    echo "1. Run the main setup script: ./scripts/setup.sh"
    echo "2. Create and activate virtual environment"
    echo "3. Install project-specific dependencies"
    echo
    
    if [ "$INSTALL_DOCKER" = true ]; then
        print_warning "Note: You may need to log out and log back in to use Docker without sudo"
    fi
}

# Run main function with all arguments
main "$@"
