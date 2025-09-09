#!/usr/bin/env python3
"""
Ultra AI Project - Production Entry Point

Production-ready main entry point with proper import handling
and comprehensive error management.
"""

import os
import sys
import asyncio
import argparse
import signal
import logging
from pathlib import Path
from typing import Optional

# Add src to path for imports - this fixes the relative import issues
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import modules
try:
    from core.system_manager import SystemManager
    from utils.logger import get_logger
    from utils.helpers import load_config, get_system_info
    from api.routes import app as api_app
    from ui.cli_interface import CLIInterface
    from ui.web_interface import WebInterface
    from tools.simple_tools import SimpleCodeExecutor, SimpleMessageFormatter
except ImportError as e:
    # Fallback for basic functionality
    print(f"Warning: Some modules not available: {e}")
    print("Running in basic mode...")
    
    # Create minimal logger
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # Import basic tools
    from tools.simple_tools import SimpleCodeExecutor, SimpleMessageFormatter
    
    SystemManager = None
    api_app = None
    CLIInterface = None
    WebInterface = None

logger = get_logger(__name__)

class UltraAI:
    """Main Ultra AI application class - Production Ready."""
    
    def __init__(self):
        self.system_manager: Optional[SystemManager] = None
        self.cli_interface: Optional[CLIInterface] = None
        self.web_interface: Optional[WebInterface] = None
        self.running = False
        self.tools = {}
        
    async def initialize(self, config_path: Optional[str] = None):
        """Initialize the Ultra AI system."""
        try:
            logger.info("Initializing Ultra AI Project...")
            
            # Initialize basic tools first (always available)
            await self._initialize_tools()
            
            # Try to load full system if available
            if SystemManager:
                try:
                    # Load configuration
                    if hasattr(sys.modules.get('utils.helpers', None), 'load_config'):
                        from utils.helpers import load_config
                        config = load_config(config_path)
                    else:
                        config = {}
                    
                    # Initialize system manager
                    self.system_manager = SystemManager(config)
                    await self.system_manager.initialize()
                    
                    # Initialize interfaces
                    if CLIInterface:
                        self.cli_interface = CLIInterface(self.system_manager)
                    if WebInterface:
                        self.web_interface = WebInterface(self.system_manager)
                        
                    logger.info("Full Ultra AI system initialized successfully")
                except Exception as e:
                    logger.warning(f"Full system initialization failed, using basic mode: {e}")
                    
            else:
                logger.info("Running in basic tools mode")
            
            logger.info("Ultra AI Project initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ultra AI: {e}")
            return False
    
    async def _initialize_tools(self):
        """Initialize basic tools."""
        try:
            # Initialize code executor
            self.tools['code_executor'] = SimpleCodeExecutor()
            logger.info(f"Initialized tool: {self.tools['code_executor'].name}")
            
            # Initialize message formatter
            self.tools['message_formatter'] = SimpleMessageFormatter()
            logger.info(f"Initialized tool: {self.tools['message_formatter'].name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise
    
    async def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the API server."""
        if not api_app:
            logger.error("API server not available - missing dependencies")
            return
            
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
            
        except ImportError:
            logger.error("uvicorn not available - cannot start API server")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise
    
    async def start_cli(self):
        """Start the CLI interface."""
        if not self.cli_interface:
            # Start basic CLI
            await self._start_basic_cli()
            return
            
        try:
            logger.info("Starting advanced CLI interface...")
            await self.cli_interface.run()
                
        except Exception as e:
            logger.error(f"Failed to start CLI: {e}")
            # Fallback to basic CLI
            await self._start_basic_cli()
    
    async def _start_basic_cli(self):
        """Start basic CLI with available tools."""
        print("\\n=== Ultra AI Basic CLI ===")
        print("Available commands:")
        print("  execute <code> - Execute Python code")
        print("  format <message> - Format a message")
        print("  help - Show this help")
        print("  exit - Exit the CLI")
        
        while self.running:
            try:
                command = input("\\nultra_ai> ").strip()
                
                if command.lower() in ['exit', 'quit']:
                    break
                elif command.lower() == 'help':
                    print("Available commands:")
                    print("  execute <code> - Execute Python code")
                    print("  format <message> - Format a message")
                    print("  help - Show this help")
                    print("  exit - Exit the CLI")
                elif command.startswith('execute '):
                    code = command[8:]
                    result = await self.tools['code_executor'].execute(code=code)
                    if result.success:
                        print(f"Output: {result.data.get('output', '')}")
                    else:
                        print(f"Error: {result.error}")
                elif command.startswith('format '):
                    message = command[7:]
                    result = await self.tools['message_formatter'].execute(message=message)
                    if result.success:
                        print(f"Formatted: {result.data}")
                    else:
                        print(f"Error: {result.error}")
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def start_web_interface(self):
        """Start the web interface."""
        if not self.web_interface:
            logger.error("Web interface not available")
            return
            
        try:
            logger.info("Starting web interface...")
            await self.web_interface.start()
                
        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")
            raise
    
    async def run(self, mode: str = "cli", **kwargs):
        """Run the Ultra AI system in specified mode."""
        self.running = True
        
        try:
            logger.info(f"Starting Ultra AI in {mode} mode...")
            
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
                tasks = []
                if api_app:
                    tasks.append(self.start_api_server(
                        host=kwargs.get("host", "127.0.0.1"),
                        port=kwargs.get("port", 8000)
                    ))
                if self.web_interface:
                    tasks.append(self.start_web_interface())
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    logger.warning("No advanced interfaces available, starting CLI")
                    await self.start_cli()
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
        ultra_ai.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ultra AI Project - Advanced AI system with multi-agent capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultra_ai.py                    # Start in CLI mode
  python ultra_ai.py --mode api         # Start API server
  python ultra_ai.py --mode web         # Start web interface
  python ultra_ai.py --mode all         # Start all interfaces
  python ultra_ai.py --system-info      # Show system information
"""
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
        default="cli",
        help="Runtime mode (default: cli)"
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
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Display system info and exit if requested
    if args.system_info:
        try:
            from utils.helpers import get_system_info
            info = get_system_info()
        except ImportError:
            info = {
                "Python Version": sys.version,
                "Platform": sys.platform,
                "Working Directory": os.getcwd()
            }
        
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