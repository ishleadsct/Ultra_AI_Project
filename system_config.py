
"""

Ultra AI System Configuration v3.0

Centralized configuration management for all system components

"""



import os

from pathlib import Path



class SystemConfig:

    def __init__(self):

        # Base paths

        self.ROOT_DIR = Path(__file__).parent

        self.ULTRA_FOLDER = Path.home() / "Ultra_Folder"

        self.MODELS_BASE = Path("/storage/emulated/0/AI_Models/.ultra_ai")

        

        # Model configurations

        self.MODEL_CONFIGS = [

            {

                "name": "llama-3.2-1b-instruct-q4_k_m",

                "role": "gatekeeper",

                "fusion_role": "primary_reasoning",

                "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",

                "path": str(self.MODELS_BASE / "models" / "llama-3.2-1b-instruct-q4_k_m.gguf"),

                "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

            },

            {

                "name": "llama-3.2-3b-instruct-q4_k_m",

                "role": "librarian",

                "fusion_role": "knowledge_management", 

                "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",

                "path": str(self.MODELS_BASE / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf"),

                "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

            }

        ]

        

        # Server configuration

        self.API_HOST = "127.0.0.1"

        self.API_PORT = 8000

        self.DASHBOARD_HOST = "127.0.0.1"

        self.DASHBOARD_PORT = 8080

        

        # Resource limits

        self.MAX_MEMORY_GB = 8

        self.MAX_CPU_CORES = 4

        self.TEMPERATURE_LIMIT = 85

