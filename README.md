# Ultra AI Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Termux Compatible](https://img.shields.io/badge/Termux-Compatible-green.svg)](https://termux.com/)

🤖 **A complete AI system with 5 specialized models, futuristic GUI, and 21+ device integrations.**

**Production-ready AI system featuring:**
- 🖥️ **Futuristic 3D Web Interface** 
- 🧠 **5 Specialized AI Models** (General, Reasoning, Programming, Advanced, Deep Analysis)
- 📱 **21 Device APIs** (GPS, sensors, notifications, etc.)
- 💾 **Dynamic Memory System** with session persistence
- 🎤 **Voice Activation** and control
- ⚡ **Optimized Performance** (2048 token context)

## 🚀 Features

### Core Capabilities
- **Multi-Agent Architecture**: Specialized agents for code, research, analysis, and creative tasks
- **Unified AI Interface**: Seamless integration with multiple LLM providers (OpenAI, Anthropic, Hugging Face)
- **Advanced Memory Management**: Persistent context and learning capabilities
- **Security-First Design**: Enterprise-grade security and access controls
- **Real-time Processing**: WebSocket-based real-time communication
- **Scalable Architecture**: Modular design for easy expansion and customization

### AI Agents
- **Code Agent**: Automated code generation, review, and optimization
- **Research Agent**: Intelligent web research and data gathering
- **Analysis Agent**: Data analysis and insight generation
- **Creative Agent**: Content creation and creative assistance

### Interfaces
- **Web Dashboard**: Modern, responsive web interface
- **CLI Interface**: Powerful command-line tools
- **REST API**: RESTful API for integration
- **WebSocket API**: Real-time bidirectional communication

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU support optional (for local AI models)
- Internet connection (for cloud AI services)

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ishleadsct/Ultra_AI_Project.git
cd Ultra_AI_Project

# Install dependencies
pip install -r requirements_minimal.txt

# Download AI models (21.6GB - see MODEL_SETUP_GUIDE.md)
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# Launch the system
./start_gui.sh    # Web interface at http://127.0.0.1:8889
# or
./start.sh        # CLI interface
```

### 📥 Required AI Models

**⚠️ Important:** This system requires 5 AI models (21.6GB total) to function.

See **[MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md)** for detailed download instructions.

**Quick model setup:**
```bash
# Create models directory
mkdir -p models/gguf
cd models/gguf

# Download all 5 required models (this will take 30-60 minutes)
wget https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf -O Qwen2-1.5B-Instruct.Q4_K_M.gguf
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf -O CodeLlama-7B-Instruct.Q4_K_M.gguf
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
wget https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
```

# Start the system
./scripts/start_system.shManual Installation# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Configure the system
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml with your API keys and preferences

# Initialize the database
python src/main.py --init-db

# Start the system
python src/main.py⚙️ ConfigurationBasic ConfigurationEdit config/settings.yaml:# API Keys
openai:
  api_key: "your-openai-api-key"
  
anthropic:
  api_key: "your-anthropic-api-key"

# System Settings
system:
  log_level: "INFO"
  max_workers: 4
  enable_gpu: false

# Security
security:
  enable_auth: true
  secret_key: "your-secret-key"Advanced ConfigurationSee docs/INSTALLATION.md for detailed configuration options.🚀 UsageWeb InterfaceStart the system: ./scripts/start_system.shOpen your browser to http://localhost:8000Log in with your credentialsStart interacting with AI agentsCLI Interface# Interactive mode
python src/main.py --cli

# Direct commands
python src/main.py --agent code --task "Create a Python function to sort a list"
python src/main.py --agent research --task "Research latest AI developments"API Usageimport requests

# Create a task
response = requests.post('http://localhost:8000/api/tasks', json={
    'agent': 'code',
    'task': 'Generate a REST API endpoint',
    'parameters': {'language': 'python', 'framework': 'flask'}
})

task_id = response.json()['task_id']

# Get results
results = requests.get(f'http://localhost:8000/api/tasks/{task_id}')📁 Project StructureUltra_AI_Project/
├── src/                    # Source code
│   ├── core/              # Core system components
│   ├── agents/            # AI agents
│   ├── api/               # API layer
│   ├── ui/                # User interfaces
│   ├── models/            # AI model integrations
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── scripts/               # Automation scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
└── data/                  # Data storage🧪 Testing# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_api.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html🔧 DevelopmentSetting up Development Environment# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
mypy src/Adding New AgentsCreate a new agent class in src/agents/Inherit from BaseAgentImplement required methodsRegister in src/agents/__init__.pyAdd tests in tests/test_agents.py📖 DocumentationInstallation GuideAPI DocumentationArchitecture OverviewContributing Guidelines🤝 ContributingWe welcome contributions! Please see CONTRIBUTING.md for guidelines.Quick Contribution StepsFork the repositoryCreate a feature branchMake your changesAdd testsSubmit a pull request📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.🆘 SupportDocumentation: Check our docs directoryIssues: GitHub IssuesDiscussions: GitHub Discussions🗓️ Roadmap[ ] Multi-modal AI support (vision, audio)[ ] Plugin architecture[ ] Cloud deployment templates[ ] Advanced workflow automation[ ] Integration with external tools[ ] Mobile application📊 PerformanceResponse Time: < 2s for most operationsThroughput: 100+ concurrent requestsMemory Usage: ~500MB base + model overheadScalability: Horizontal scaling supported🔒 SecurityEnd-to-end encryption for sensitive dataAPI key management and rotationRole-based access controlAudit loggingSecurity scanning in CI/CD⭐ AcknowledgmentsOpenAI for GPT modelsAnthropic for Claude modelsHugging Face for open-source modelsThe Python AI communityMade with ❤️ by the Ultra AI Team 
