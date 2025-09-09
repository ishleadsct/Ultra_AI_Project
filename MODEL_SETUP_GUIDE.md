# Ultra AI Model Setup Guide

## 📥 Required AI Models Download

Ultra AI requires 5 specialized GGUF models (21.6GB total) to function. Follow these steps to download and install them:

### 🛠️ Quick Setup (Automated)

Run the automated download script:
```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

### 📋 Manual Setup

Create the models directory and download each model:

```bash
# Create models directory
mkdir -p models/gguf
cd models/gguf

# Download all 5 required models
wget https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf -O Qwen2-1.5B-Instruct.Q4_K_M.gguf

wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf -O CodeLlama-7B-Instruct.Q4_K_M.gguf

wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

wget https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
```

### 📊 Model Specifications

| Model | Size | Purpose | Specialization |
|-------|------|---------|----------------|
| **Qwen2-1.5B** | 940MB | General Intelligence Core | Fast responses, general assistance |
| **Phi-3** | 2.3GB | Reasoning & Analysis | Logical thinking, problem-solving |
| **CodeLlama-7B** | 3.9GB | Programming Expert | Code generation, debugging |
| **Llama-3.1-8B** | 4.6GB | Advanced Intelligence | Complex reasoning, comprehensive tasks |
| **DeepSeek-Coder** | 9.7GB | Deep Code Analysis | Advanced programming, system architecture |

### ✅ Verification

After downloading, verify your setup:

```bash
# Check all models are present
ls -lah models/gguf/

# Run model test
python3 quick_model_test.py
```

Expected output:
```
✅ Ultra AI Model Infrastructure: WORKING
✅ Found 5/5 required models
✅ All model files accessible
✅ Total size: ~21.6GB
```

### 🚨 Storage Requirements

- **Minimum**: 25GB free space
- **Recommended**: 30GB free space (including system overhead)
- **Platform**: Android/Termux, Linux, macOS, Windows

### 📱 Termux-Specific Notes

For Android/Termux users:
```bash
# Ensure sufficient storage
termux-setup-storage

# Install required dependencies
pkg install python wget curl

# Verify Python version
python3 --version  # Should be 3.8+
```

### 🔧 Troubleshooting

**Download Issues:**
- Ensure stable internet connection
- Check available storage space: `df -h`
- For slow downloads, try resuming: `wget -c [URL]`

**File Permission Issues:**
```bash
chmod +x start_gui.sh
chmod +x launch_ultra_ai.sh
```

**Python Dependencies:**
```bash
pip install -r requirements_minimal.txt
```

### 🌐 Alternative Download Sources

If Hugging Face is slow, models are also available from:
- **Ollama**: `ollama pull qwen2:1.5b` (requires conversion)
- **Direct mirrors**: See `external_tools/model_mirrors.txt`

### 🚀 Quick Start After Setup

Once models are downloaded:
```bash
# Launch GUI interface
./start_gui.sh

# Or launch CLI interface
./start.sh

# Or complete system
./launch_complete_ultra_ai.sh
```

Access the web interface at: `http://127.0.0.1:8889`

---

## 📞 Support

If you encounter issues:
1. Check `troubleshooting.md`
2. Run `./scripts/system_check.sh`
3. Open an issue on GitHub with error logs

**Total Download Time**: 30-60 minutes depending on internet speed
**Installation Time**: 5 minutes
**First Launch**: 2-3 minutes (model loading)