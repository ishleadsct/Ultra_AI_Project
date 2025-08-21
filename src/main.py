#!/usr/bin/env python3
"""
Ultra AI Project - Main Entry Point

This module serves as the primary entry point for the Ultra AI Project,
providing command-line interface and system initialization.
"""

import os
import sys
import asyncio
import argparse
import signal
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.system_manager import SystemManager
from utils.logger import get_logger
from utils.helpers import load_config, get_system_info
from api.routes import app as api_app
from ui.cli_interface import CLIInterface
from ui.web_interface import WebInterface

logger = get_logger(__name__)

class UltraAI:
    """Main Ultra AI application class."""
    
    def __init__(self):
        self.system_manager: Optional[SystemManager] = None
        self.cli_interface: Optional[CLIInterface] = None
        self.web_interface: Optional[WebInterface] = None
        self.running = False
        
    async def initialize(self, config_path: Optional[str] = None):
        """Initialize the Ultra AI system."""
        try:
            logger.info("Initializing Ultra AI Project...")
            
            # Load configuration
            config = load_config(config_path)
            
            # Initialize system manager
            self.system_manager = SystemManager(config)
            await self.system_manager.initialize()
            
            # Initialize interfaces
            self.cli_interface = CLIInterface(self.system_manager)
            self.web_interface = WebInterface(self.system_manager)
            
            logger.info("Ultra AI Project initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ultra AI: {e}")
            return False
    
    async def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the API server."""
        try:
            import uvicorn
            
            logger.info(f"Starting API server on {host}:{port}")
            
            config = uvicorn.Config(
                app=api_app,
                host=host,
                port=port,
                reload=False,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise
    
    async def start_cli(self):
        """Start the CLI interface."""
        try:
            logger.info("Starting CLI interface...")
            if self.cli_interface:
                await self.cli_interface.run()
            else:
                logger.error("CLI interface not initialized")
                
        except Exception as e:
            logger.error(f"Failed to start CLI: {e}")
            raise
    
    async def start_web_interface(self):
        """Start the web interface."""
        try:
            logger.info("Starting web interface...")
            if self.web_interface:
                await self.web_interface.start()
            else:
                logger.error("Web interface not initialized")
                
        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")
            raise
    
    async def run(self, mode: str = "api", **kwargs):
        """Run the Ultra AI system in specified mode."""
        self.running = True
        
        try:
            if mode == "api":
                await self.start_api_server(
                    host=kwargs.get("host", "127.0.0.1"),
                    port=kwargs.get("port", 8000)
                )
            elif mode == "cli":
                await self.start_cli()
            elif mode == "web":
                await self.start_web_interface()
            elif mode == "all":
                # Start all interfaces concurrently
                await asyncio.gather(
                    self.start_api_server(
                        host=kwargs.get("host", "127.0.0.1"),
                        port=kwargs.get("port", 8000)
                    ),
                    self.start_web_interface(),
                    return_exceptions=True
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Error running Ultra AI: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the Ultra AI system."""
        if not self.running:
            return
            
        logger.info("Shutting down Ultra AI Project...")
        self.running = False
        
        try:
            if self.system_manager:
                await self.system_manager.shutdown()
                
            if self.web_interface:
                await self.web_interface.stop()
                
            logger.info("Ultra AI Project shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def setup_signal_handlers(ultra_ai: UltraAI):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        if ultra_ai.running:
            asyncio.create_task(ultra_ai.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ultra AI Project - Advanced AI system with multi-agent capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Ultra AI Project 1.0.0"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "cli", "web", "all"],
        default="api",
        help="Runtime mode (default: api)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for API server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--system-info",
        action="store_true",
        help="Display system information and exit"
    )
    
    return parser

async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Display system info and exit if requested
    if args.system_info:
        info = get_system_info()
        print("Ultra AI Project - System Information")
        print("=" * 40)
        for key, value in info.items():
            print(f"{key}: {value}")
        return
    
    # Create and initialize Ultra AI
    ultra_ai = UltraAI()
    
    # Setup signal handlers
    setup_signal_handlers(ultra_ai)
    
    # Initialize system
    success = await ultra_ai.initialize(args.config)
    if not success:
        logger.error("Failed to initialize Ultra AI system")
        sys.exit(1)
    
    # Run in specified mode
    try:
        await ultra_ai.run(
            mode=args.mode,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

def sync_main():
    """Synchronous wrapper for main."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_main()
