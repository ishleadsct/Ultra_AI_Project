#!/bin/bash

# Ultra AI Project - AI Models Download Script
# Downloads and caches AI models for offline use

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
MODELS_DIR="./models"
CACHE_DIR="$MODELS_DIR/cache"
HUGGINGFACE_DIR="$MODELS_DIR/huggingface"
LOCAL_DIR="$MODELS_DIR/local"
TEMP_DIR="./temp/downloads"
VENV_PATH="venv"

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

print_progress() {
    echo -e "${CYAN}[PROGRESS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check available disk space
check_disk_space() {
    local required_space_gb=$1
    local target_dir=$2
    
    print_status "Checking disk space for $target_dir..."
    
    if command_exists df; then
        local available_kb=$(df "$target_dir" | tail -1 | awk '{print $4}')
        local available_gb=$((available_kb / 1024 / 1024))
        
        print_status "Available space: ${available_gb}GB, Required: ${required_space_gb}GB"
        
        if [ $available_gb -lt $required_space_gb ]; then
            print_error "Insufficient disk space. Need ${required_space_gb}GB, have ${available_gb}GB"
            return 1
        else
            print_success "Sufficient disk space available"
            return 0
        fi
    else
        print_warning "Cannot check disk space (df command not available)"
        return 0
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating model directories..."
    
    mkdir -p "$MODELS_DIR"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$HUGGINGFACE_DIR"
    mkdir -p "$LOCAL_DIR"
    mkdir -p "$TEMP_DIR"
    
    # Set appropriate permissions
    chmod 755 "$MODELS_DIR" "$CACHE_DIR" "$HUGGINGFACE_DIR" "$LOCAL_DIR"
    chmod 700 "$TEMP_DIR"
    
    print_success "Directories created successfully"
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
        print_warning "Virtual environment not found. Some downloads may require it."
    fi
}

# Function to install required Python packages
install_download_dependencies() {
    print_status "Installing download dependencies..."
    
    local packages=(
        "huggingface_hub"
        "transformers[torch]"
        "datasets"
        "accelerate"
        "safetensors"
        "requests"
        "tqdm"
    )
    
    for package in "${packages[@]}"; do
        local pkg_name=${package%%[*}
        if python -c "import ${pkg_name}" 2>/dev/null; then
            print_success "$package already installed"
        else
            print_status "Installing $package..."
            pip install "$package" --quiet || {
                print_warning "Failed to install $package, continuing..."
            }
        fi
    done
}

# Function to download Hugging Face models
download_huggingface_model() {
    local model_name=$1
    local model_type=${2:-"transformers"}
    local cache_subdir=${3:-""}
    
    print_progress "Downloading Hugging Face model: $model_name"
    
    local target_dir="$HUGGINGFACE_DIR/$cache_subdir"
    mkdir -p "$target_dir"
    
    cat > "$TEMP_DIR/download_model.py" << PYEOF
import os
import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
    
    model_name = "$model_name"
    cache_dir = "$target_dir"
    
    print(f"Downloading {model_name} to {cache_dir}")
    
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        try:
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            
            print(f"✓ Model {model_name} downloaded and verified successfully")
            print(f"  Location: {model_path}")
            print(f"  Model type: {config.model_type}")
            print(f"  Vocab size: {tokenizer.vocab_size}")
            
        except Exception as e:
            print(f"⚠ Model downloaded but verification failed: {e}")
            
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Missing required packages: {e}")
    sys.exit(1)
PYEOF
    
    python "$TEMP_DIR/download_model.py"
    
    if [ $? -eq 0 ]; then
        print_success "Downloaded: $model_name"
    else
        print_error "Failed to download: $model_name"
        return 1
    fi
}

# Function to download essential models
download_essential_models() {
    print_banner "Downloading Essential Models..."
    
    local essential_models=(
        "microsoft/DialoGPT-small"
        "sentence-transformers/all-MiniLM-L6-v2"
        "distilbert-base-uncased"
        "t5-small"
    )
    
    for model in "${essential_models[@]}"; do
        download_huggingface_model "$model" "transformers" "essential"
        sleep 2
    done
    
    print_success "Essential models download completed"
}

# Function to download optional models
download_optional_models() {
    if [ "$DOWNLOAD_OPTIONAL" = "true" ]; then
        print_banner "Downloading Optional Models..."
        
        local optional_models=(
            "microsoft/DialoGPT-medium"
            "facebook/bart-base"
            "google/flan-t5-base"
            "sentence-transformers/all-mpnet-base-v2"
            "microsoft/codebert-base"
        )
        
        for model in "${optional_models[@]}"; do
            download_huggingface_model "$model" "transformers" "optional"
            sleep 3
        done
        
        print_success "Optional models download completed"
    fi
}

# Function to download large models
download_large_models() {
    if [ "$DOWNLOAD_LARGE" = "true" ]; then
        print_banner "Downloading Large Models..."
        print_warning "Large models require significant disk space and time"
        
        if ! check_disk_space 20 "$MODELS_DIR"; then
            print_error "Skipping large models due to insufficient disk space"
            return 1
        fi
        
        local large_models=(
            "microsoft/DialoGPT-large"
            "facebook/bart-large"
            "google/flan-t5-large"
            "microsoft/codebert-base-mlm"
            "huggingface/CodeBERTa-small-v1"
        )
        
        for model in "${large_models[@]}"; do
            print_status "Downloading large model: $model (this may take a while...)"
            download_huggingface_model "$model" "transformers" "large"
            sleep 5
        done
        
        print_success "Large models download completed"
    fi
}

# Function to download GGUF models for local inference
download_gguf_models() {
    if [ "$DOWNLOAD_GGUF" = "true" ]; then
        print_banner "Downloading GGUF Models for Local Inference..."
        
        local gguf_models=(
            "TheBloke/Llama-2-7B-Chat-GGUF:llama-2-7b-chat.q4_0.gguf"
            "TheBloke/CodeLlama-7B-Instruct-GGUF:codellama-7b-instruct.q4_0.gguf"
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF:mistral-7b-instruct-v0.1.q4_0.gguf"
        )
        
        for model_info in "${gguf_models[@]}"; do
            local repo_id="${model_info%:*}"
            local filename="${model_info#*:}"
            
            print_progress "Downloading GGUF model: $repo_id/$filename"
            
            cat > "$TEMP_DIR/download_gguf.py" << PYEOF
import os
from pathlib import Path
try:
    from huggingface_hub import hf_hub_download
    
    repo_id = "$repo_id"
    filename = "$filename"
    local_dir = "$LOCAL_DIR/gguf"
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            resume_download=True
        )
        print(f"✓ Downloaded {filename} to {file_path}")
        
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        
except ImportError:
    print("✗ huggingface_hub not available for GGUF downloads")
PYEOF
            
            python "$TEMP_DIR/download_gguf.py"
        done
        
        print_success "GGUF models download completed"
    fi
}

# Function to download vision models
download_vision_models() {
    if [ "$DOWNLOAD_VISION" = "true" ]; then
        print_banner "Downloading Vision Models..."
        
        local vision_models=(
            "microsoft/resnet-50"
            "google/vit-base-patch16-224"
            "microsoft/beit-base-patch16-224"
            "facebook/detr-resnet-50"
        )
        
        for model in "${vision_models[@]}"; do
            download_huggingface_model "$model" "transformers" "vision"
            sleep 3
        done
        
        print_success "Vision models download completed"
    fi
}

# Function to download audio models
download_audio_models() {
    if [ "$DOWNLOAD_AUDIO" = "true" ]; then
        print_banner "Downloading Audio Models..."
        
        local audio_models=(
            "openai/whisper-tiny"
            "openai/whisper-base"
            "facebook/wav2vec2-base-960h"
            "microsoft/speecht5_tts"
        )
        
        for model in "${audio_models[@]}"; do
            download_huggingface_model "$model" "transformers" "audio"
            sleep 3
        done
        
        print_success "Audio models download completed"
    fi
}

# Function to test downloaded models
test_models() {
    if [ "$TEST_MODELS" = "true" ]; then
        print_banner "Testing Downloaded Models..."
        
        cat > "$TEMP_DIR/test_models.py" << PYEOF
import os
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    
    models_dir = Path("./models/huggingface")
    
    model_dirs = []
    for subdir in ["essential", "optional", "large", "vision", "audio"]:
        subdir_path = models_dir / subdir
        if subdir_path.exists():
            for item in subdir_path.iterdir():
                if item.is_dir() and "models--" in item.name:
                    model_dirs.append(item)
    
    if not model_dirs:
        print("No models found to test")
        sys.exit(0)
    
    print(f"Testing {len(model_dirs)} models...")
    
    successful_tests = 0
    failed_tests = 0
    
    for model_dir in model_dirs[:3]:
        try:
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            print(f"\nTesting model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModel.from_pretrained(model_dir)
            
            print(f"✓ {model_name}: Loaded successfully")
            print(f"  - Vocab size: {tokenizer.vocab_size}")
            print(f"  - Model parameters: {model.num_parameters():,}")
            
            successful_tests += 1
            
        except Exception as e:
            print(f"✗ {model_name}: Failed to load - {e}")
            failed_tests += 1
    
    print(f"\nTest Results:")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {failed_tests}")
    
    if successful_tests > 0:
        print("✓ Model testing completed successfully")
    else:
        print("✗ All model tests failed")
        
except ImportError as e:
    print(f"Cannot test models: {e}")
    print("Install transformers package to enable model testing")
PYEOF
        
        python "$TEMP_DIR/test_models.py"
        print_success "Model testing completed"
    fi
}

# Function to cleanup download cache
cleanup_cache() {
    if [ "$CLEANUP_CACHE" = "true" ]; then
        print_status "Cleaning up download cache..."
        
        if [ -d "$TEMP_DIR" ]; then
            rm -rf "$TEMP_DIR"/*
        fi
        
        pip cache purge --quiet 2>/dev/null || true
        
        cat > "$TEMP_DIR/cleanup_cache.py" << PYEOF
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import scan_cache_dir
    
    cache_info = scan_cache_dir()
    print(f"Hugging Face cache size: {cache_info.size_on_disk_str}")
    
    delete_strategy = cache_info.delete_revisions()
    print(f"Will free {delete_strategy.expected_freed_size_str}")
    delete_strategy.execute()
    
    print("✓ Cache cleanup completed")
    
except ImportError:
    print("huggingface_hub not available for cache cleanup")
except Exception as e:
    print(f"Cache cleanup failed: {e}")
PYEOF
        
        python "$TEMP_DIR/cleanup_cache.py"
        print_success "Cache cleanup completed"
    fi
}

# Function to generate model inventory
generate_inventory() {
    print_status "Generating model inventory..."
    
    cat > "$TEMP_DIR/generate_inventory.py" << PYEOF
import os
import json
from pathlib import Path
from datetime import datetime

try:
    from transformers import AutoConfig
    
    models_dir = Path("./models")
    inventory = {
        "generated_at": datetime.now().isoformat(),
        "models": {},
        "summary": {
            "total_models": 0,
            "categories": {}
        }
    }
    
    for category in ["essential", "optional", "large", "vision", "audio"]:
        category_path = models_dir / "huggingface" / category
        if not category_path.exists():
            continue
            
        inventory["models"][category] = []
        model_count = 0
        
        for model_dir in category_path.iterdir():
            if model_dir.is_dir() and "models--" in model_dir.name:
                try:
                    model_name = model_dir.name.replace("models--", "").replace("--", "/")
                    
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        config = AutoConfig.from_pretrained(model_dir)
                        model_info = {
                            "name": model_name,
                            "path": str(model_dir),
                            "model_type": getattr(config, 'model_type', 'unknown'),
                            "size": sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()),
                            "files": len(list(model_dir.rglob('*')))
                        }
                    else:
                        model_info = {
                            "name": model_name,
                            "path": str(model_dir),
                            "model_type": "unknown",
                            "size": sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()),
                            "files": len(list(model_dir.rglob('*')))
                        }
                    
                    inventory["models"][category].append(model_info)
                    model_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not process {model_name}: {e}")
        
        inventory["summary"]["categories"][category] = model_count
        inventory["summary"]["total_models"] += model_count
    
    inventory_file = models_dir / "inventory.json"
    with open(inventory_file, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"✓ Model inventory saved to {inventory_file}")
    print(f"  Total models: {inventory['summary']['total_models']}")
    for category, count in inventory["summary"]["categories"].items():
        if count > 0:
            print(f"  {category}: {count} models")
    
except Exception as e:
    print(f"Failed to generate inventory: {e}")
PYEOF
    
    python "$TEMP_DIR/generate_inventory.py"
    print_success "Model inventory generated"
}

# Function to show download summary
show_summary() {
    print_banner "============================================="
    print_banner "      Model Download Summary"
    print_banner "============================================="
    echo
    
    if [ -d "$MODELS_DIR" ]; then
        print_status "📁 Models directory: $MODELS_DIR"
        
        if command_exists du; then
            local total_size=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
            print_status "💾 Total size: $total_size"
        fi
        
        local model_count=$(find "$MODELS_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "*.gguf" | wc -l)
        print_status "🤖 Model files: $model_count"
        
        echo
        print_success "Models are ready for use!"
        echo
        print_status "Next steps:"
        print_status "1. Update config/ai_models.yaml with local model paths"
        print_status "2. Set TRANSFORMERS_CACHE environment variable to $HUGGINGFACE_DIR"
        print_status "3. Test models with: python -c \"from transformers import pipeline; print('Models working!')\""
    else
        print_error "Models directory not found"
    fi
    
    echo
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --essential-only    Download only essential models (default)"
    echo "  --with-optional     Include optional models"
    echo "  --with-large        Include large models (requires >20GB space)"
    echo "  --with-gguf         Include GGUF models for local inference"
    echo "  --with-vision       Include computer vision models"
    echo "  --with-audio        Include audio processing models"
    echo "  --all               Download all available models"
    echo "  --test              Test downloaded models after download"
    echo "  --cleanup           Clean up cache after download"
    echo "  --no-inventory      Skip generating model inventory"
    echo "  --force             Force re-download even if models exist"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Download essential models only"
    echo "  $0 --with-optional --test   # Download essential + optional, then test"
    echo "  $0 --all --cleanup          # Download everything, clean cache"
    echo "  $0 --essential-only --force # Force re-download essential models"
    echo ""
    echo "Disk space requirements:"
    echo "  Essential models: ~2GB"
    echo "  Optional models: +5GB"
    echo "  Large models: +20GB"
    echo "  GGUF models: +10GB"
    echo "  Vision models: +3GB"
    echo "  Audio models: +5GB"
    echo ""
}

# Main function
main() {
    DOWNLOAD_OPTIONAL="false"
    DOWNLOAD_LARGE="false"
    DOWNLOAD_GGUF="false"
    DOWNLOAD_VISION="false"
    DOWNLOAD_AUDIO="false"
    TEST_MODELS="false"
    CLEANUP_CACHE="false"
    GENERATE_INVENTORY="true"
    FORCE_DOWNLOAD="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --essential-only)
                shift
                ;;
            --with-optional)
                DOWNLOAD_OPTIONAL="true"
                shift
                ;;
            --with-large)
                DOWNLOAD_LARGE="true"
                shift
                ;;
            --with-gguf)
                DOWNLOAD_GGUF="true"
                shift
                ;;
            --with-vision)
                DOWNLOAD_VISION="true"
                shift
                ;;
            --with-audio)
                DOWNLOAD_AUDIO="true"
                shift
                ;;
            --all)
                DOWNLOAD_OPTIONAL="true"
                DOWNLOAD_LARGE="true"
                DOWNLOAD_GGUF="true"
                DOWNLOAD_VISION="true"
                DOWNLOAD_AUDIO="true"
                shift
                ;;
            --test)
                TEST_MODELS="true"
                shift
                ;;
            --cleanup)
                CLEANUP_CACHE="true"
                shift
                ;;
            --no-inventory)
                GENERATE_INVENTORY="false"
                shift
                ;;
            --force)
                FORCE_DOWNLOAD="true"
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
    print_banner "    Ultra AI Project - Model Downloader"
    print_banner "============================================="
    echo
    
    check_disk_space 5 "."
    create_directories
    activate_venv
    install_download_dependencies
    
    download_essential_models
    download_optional_models
    download_large_models
    download_gguf_models
    download_vision_models
    download_audio_models
    
    test_models
    
    if [ "$GENERATE_INVENTORY" = "true" ]; then
        generate_inventory
    fi
    
    cleanup_cache
    show_summary
}

main "$@"
