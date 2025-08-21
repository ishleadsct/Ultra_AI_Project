"""
Ultra AI Project - Command Line Interface

Provides a comprehensive command-line interface for interacting with the Ultra AI system.
Supports interactive mode, direct commands, and batch operations.
"""

import asyncio
import argparse
import sys
import json
import os
import signal
from typing import Dict, List, Optional, Any
from datetime import datetime
import readline
import atexit

from ..core.system_manager import SystemManager
from ..core.task_coordinator import TaskCoordinator
from ..agents.base_agent import BaseAgent
from ..utils.logger import Logger
from ..utils.helpers import format_duration, format_bytes


class CLIInterface:
    """Command-line interface for Ultra AI system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CLI interface."""
        self.config = config
        self.logger = Logger(__name__)
        self.system_manager = None
        self.task_coordinator = None
        self.interactive_mode = False
        self.running = True
        
        # CLI history
        self.history_file = os.path.expanduser('~/.ultra_ai_history')
        self.setup_readline()
        
        # Command registry
        self.commands = {
            'help': self.cmd_help,
            'status': self.cmd_status,
            'agents': self.cmd_agents,
            'task': self.cmd_task,
            'tasks': self.cmd_tasks,
            'models': self.cmd_models,
            'config': self.cmd_config,
            'logs': self.cmd_logs,
            'health': self.cmd_health,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'clear': self.cmd_clear,
            'history': self.cmd_history,
            'version': self.cmd_version,
        }
        
        # Command aliases
        self.aliases = {
            '?': 'help',
            'ls': 'agents',
            'ps': 'tasks',
            'stat': 'status',
            'q': 'quit',
            'x': 'exit',
            'cls': 'clear',
            'ver': 'version',
        }
    
    def setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            readline.set_completer(self.complete_command)
            readline.parse_and_bind('tab: complete')
            readline.set_completer_delims(' \t\n')
            
            # Load history
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
            
            # Set history length
            readline.set_history_length(1000)
            
            # Save history on exit
            atexit.register(self.save_history)
            
        except ImportError:
            self.logger.warning("Readline not available, command completion disabled")
    
    def save_history(self):
        """Save command history to file."""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            self.logger.warning(f"Could not save command history: {e}")
    
    def complete_command(self, text: str, state: int) -> Optional[str]:
        """Auto-complete commands."""
        if state == 0:
            # First call, generate completions
            all_commands = list(self.commands.keys()) + list(self.aliases.keys())
            self.completions = [cmd for cmd in all_commands if cmd.startswith(text)]
        
        try:
            return self.completions[state]
        except IndexError:
            return None
    
    async def initialize(self):
        """Initialize the CLI interface."""
        try:
            self.logger.info("Initializing CLI interface...")
            
            # Initialize system components
            self.system_manager = SystemManager(self.config)
            await self.system_manager.initialize()
            
            self.task_coordinator = TaskCoordinator(self.config)
            await self.task_coordinator.initialize()
            
            self.logger.info("CLI interface initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CLI interface: {e}")
            raise
    
    async def run_interactive(self):
        """Run the CLI in interactive mode."""
        self.interactive_mode = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.print_banner()
        self.print_help()
        
        while self.running:
            try:
                # Get user input
                prompt = self.get_prompt()
                try:
                    command_line = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting...")
                    break
                
                if not command_line:
                    continue
                
                # Process command
                await self.process_command(command_line)
                
            except Exception as e:
                self.print_error(f"Error processing command: {e}")
                self.logger.exception("CLI command error")
        
        await self.shutdown()
    
    async def run_command(self, command_line: str):
        """Run a single command and exit."""
        try:
            await self.process_command(command_line)
        except Exception as e:
            self.print_error(f"Error executing command: {e}")
            sys.exit(1)
        finally:
            await self.shutdown()
    
    async def process_command(self, command_line: str):
        """Process a command line input."""
        parts = command_line.split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Check for aliases
        if command in self.aliases:
            command = self.aliases[command]
        
        # Execute command
        if command in self.commands:
            try:
                await self.commands[command](args)
            except Exception as e:
                self.print_error(f"Command '{command}' failed: {e}")
                self.logger.exception(f"Command execution error: {command}")
        else:
            self.print_error(f"Unknown command: {command}. Type 'help' for available commands.")
    
    def signal_handler(self, signum, frame):
        """Handle system signals."""
        if signum in (signal.SIGINT, signal.SIGTERM):
            print("\nReceived interrupt signal, shutting down...")
            self.running = False
    
    async def shutdown(self):
        """Shutdown the CLI interface."""
        try:
            self.logger.info("Shutting down CLI interface...")
            
            if self.task_coordinator:
                await self.task_coordinator.shutdown()
            
            if self.system_manager:
                await self.system_manager.shutdown()
            
            self.logger.info("CLI interface shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during CLI shutdown: {e}")
    
    def get_prompt(self) -> str:
        """Get the command prompt."""
        if self.system_manager and self.system_manager.is_healthy():
            status = "🟢"
        else:
            status = "🔴"
        
        return f"ultra-ai {status} > "
    
    def print_banner(self):
        """Print the CLI banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                     Ultra AI System CLI                     ║
║                  Command Line Interface                     ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_help(self):
        """Print basic help information."""
        print("Type 'help' for available commands, 'exit' to quit")
        print()
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        print(f"❌ {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        print(f"⚠️  {message}")
    
    def print_info(self, message: str):
        """Print info message."""
        print(f"ℹ️  {message}")
    
    # Command implementations
    
    async def cmd_help(self, args: List[str]):
        """Show help information."""
        if args:
            # Help for specific command
            command = args[0].lower()
            if command in self.aliases:
                command = self.aliases[command]
            
            help_text = self.get_command_help(command)
            if help_text:
                print(help_text)
            else:
                self.print_error(f"No help available for command: {command}")
        else:
            # General help
            print("Available commands:")
            print()
            
            categories = {
                "System": ["status", "health", "version", "config"],
                "Agents": ["agents", "models"],
                "Tasks": ["task", "tasks"],
                "Utilities": ["logs", "history", "clear"],
                "Navigation": ["help", "exit", "quit"]
            }
            
            for category, commands in categories.items():
                print(f"{category}:")
                for cmd in commands:
                    description = self.get_command_description(cmd)
                    print(f"  {cmd:<12} - {description}")
                print()
            
            print("Aliases:")
            for alias, command in self.aliases.items():
                print(f"  {alias:<12} - {command}")
            print()
            
            print("Use 'help <command>' for detailed information about a specific command.")
    
    async def cmd_status(self, args: List[str]):
        """Show system status."""
        if not self.system_manager:
            self.print_error("System manager not initialized")
            return
        
        print("Ultra AI System Status")
        print("=" * 50)
        
        # System health
        health = await self.system_manager.get_health_status()
        status_icon = "🟢" if health.get('healthy', False) else "🔴"
        print(f"System Health: {status_icon} {'Healthy' if health.get('healthy', False) else 'Unhealthy'}")
        print(f"Uptime: {format_duration(health.get('uptime', 0))}")
        print()
        
        # Component status
        components = health.get('components', {})
        print("Components:")
        for component, status in components.items():
            status_icon = "🟢" if status == 'healthy' else "🔴"
            print(f"  {component:<15}: {status_icon} {status}")
        print()
        
        # Resource usage
        resources = health.get('resources', {})
        if resources:
            print("Resource Usage:")
            print(f"  CPU: {resources.get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {format_bytes(resources.get('memory_used', 0))} / {format_bytes(resources.get('memory_total', 0))}")
            print(f"  Disk: {format_bytes(resources.get('disk_used', 0))} / {format_bytes(resources.get('disk_total', 0))}")
            print()
        
        # Active tasks
        if self.task_coordinator:
            active_tasks = await self.task_coordinator.get_active_task_count()
            print(f"Active Tasks: {active_tasks}")
    
    async def cmd_agents(self, args: List[str]):
        """Show available agents."""
        if not self.system_manager:
            self.print_error("System manager not initialized")
            return
        
        agents = await self.system_manager.get_agents()
        
        if not agents:
            self.print_warning("No agents available")
            return
        
        print("Available Agents")
        print("=" * 50)
        
        for agent_name, agent_info in agents.items():
            status_icon = "🟢" if agent_info.get('status') == 'active' else "🔴"
            print(f"{status_icon} {agent_name}")
            print(f"   Type: {agent_info.get('type', 'unknown')}")
            print(f"   Status: {agent_info.get('status', 'unknown')}")
            print(f"   Version: {agent_info.get('version', 'unknown')}")
            
            capabilities = agent_info.get('capabilities', [])
            if capabilities:
                print(f"   Capabilities: {', '.join(capabilities)}")
            print()
    
    async def cmd_task(self, args: List[str]):
        """Create or manage tasks."""
        if not self.task_coordinator:
            self.print_error("Task coordinator not initialized")
            return
        
        if not args:
            self.print_error("Usage: task <action> [options]")
            self.print_info("Actions: create, get, cancel, retry")
            return
        
        action = args[0].lower()
        
        if action == "create":
            await self._create_task(args[1:])
        elif action == "get":
            await self._get_task(args[1:])
        elif action == "cancel":
            await self._cancel_task(args[1:])
        elif action == "retry":
            await self._retry_task(args[1:])
        else:
            self.print_error(f"Unknown task action: {action}")
    
    async def _create_task(self, args: List[str]):
        """Create a new task."""
        if len(args) < 2:
            self.print_error("Usage: task create <agent> <description>")
            return
        
        agent_type = args[0]
        description = " ".join(args[1:])
        
        try:
            task = await self.task_coordinator.create_task(
                agent_type=agent_type,
                description=description,
                parameters={}
            )
            
            self.print_success(f"Task created: {task.id}")
            print(f"Agent: {task.agent_type}")
            print(f"Status: {task.status}")
            print(f"Description: {task.description}")
            
        except Exception as e:
            self.print_error(f"Failed to create task: {e}")
    
    async def _get_task(self, args: List[str]):
        """Get task information."""
        if not args:
            self.print_error("Usage: task get <task_id>")
            return
        
        task_id = args[0]
        
        try:
            task = await self.task_coordinator.get_task(task_id)
            
            if not task:
                self.print_error(f"Task not found: {task_id}")
                return
            
            print(f"Task: {task.id}")
            print(f"Agent: {task.agent_type}")
            print(f"Status: {task.status}")
            print(f"Description: {task.description}")
            print(f"Created: {task.created_at}")
            
            if task.completed_at:
                print(f"Completed: {task.completed_at}")
                duration = (task.completed_at - task.created_at).total_seconds()
                print(f"Duration: {format_duration(duration)}")
            
            if task.result:
                print(f"Result: {task.result}")
            
            if task.error:
                print(f"Error: {task.error}")
                
        except Exception as e:
            self.print_error(f"Failed to get task: {e}")
    
    async def _cancel_task(self, args: List[str]):
        """Cancel a task."""
        if not args:
            self.print_error("Usage: task cancel <task_id>")
            return
        
        task_id = args[0]
        
        try:
            await self.task_coordinator.cancel_task(task_id)
            self.print_success(f"Task cancelled: {task_id}")
            
        except Exception as e:
            self.print_error(f"Failed to cancel task: {e}")
    
    async def _retry_task(self, args: List[str]):
        """Retry a failed task."""
        if not args:
            self.print_error("Usage: task retry <task_id>")
            return
        
        task_id = args[0]
        
        try:
            new_task = await self.task_coordinator.retry_task(task_id)
            self.print_success(f"Task retried. New task ID: {new_task.id}")
            
        except Exception as e:
            self.print_error(f"Failed to retry task: {e}")
    
    async def cmd_tasks(self, args: List[str]):
        """List tasks."""
        if not self.task_coordinator:
            self.print_error("Task coordinator not initialized")
            return
        
        # Parse arguments
        status_filter = None
        agent_filter = None
        limit = 20
        
        i = 0
        while i < len(args):
            if args[i] == "--status" and i + 1 < len(args):
                status_filter = args[i + 1]
                i += 2
            elif args[i] == "--agent" and i + 1 < len(args):
                agent_filter = args[i + 1]
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            tasks = await self.task_coordinator.list_tasks(
                status=status_filter,
                agent_type=agent_filter,
                limit=limit
            )
            
            if not tasks:
                self.print_warning("No tasks found")
                return
            
            print(f"Tasks (showing {len(tasks)} of {limit} max)")
            print("=" * 80)
            
            for task in tasks:
                status_icon = {
                    'pending': '⏳',
                    'running': '🏃',
                    'completed': '✅',
                    'failed': '❌',
                    'cancelled': '🛑'
                }.get(task.status, '❓')
                
                duration = ""
                if task.completed_at:
                    dur = (task.completed_at - task.created_at).total_seconds()
                    duration = f" ({format_duration(dur)})"
                
                print(f"{status_icon} {task.id[:8]} | {task.agent_type:<12} | {task.status:<10} | {task.description[:40]}{duration}")
            
        except Exception as e:
            self.print_error(f"Failed to list tasks: {e}")
    
    async def cmd_models(self, args: List[str]):
        """Show available AI models."""
        if not self.system_manager:
            self.print_error("System manager not initialized")
            return
        
        try:
            models = await self.system_manager.get_available_models()
            
            print("Available AI Models")
            print("=" * 50)
            
            for provider, provider_models in models.items():
                print(f"\n{provider.upper()}:")
                for model in provider_models:
                    status_icon = "🟢" if model.get('available', True) else "🔴"
                    print(f"  {status_icon} {model['name']}")
                    if 'description' in model:
                        print(f"     {model['description']}")
            
        except Exception as e:
            self.print_error(f"Failed to get models: {e}")
    
    async def cmd_config(self, args: List[str]):
        """Show or modify configuration."""
        if not args:
            # Show current configuration
            print("Current Configuration")
            print("=" * 50)
            
            # Show safe configuration (no secrets)
            safe_config = self._sanitize_config(self.config)
            print(json.dumps(safe_config, indent=2))
        else:
            self.print_warning("Configuration modification not implemented in CLI")
    
    async def cmd_logs(self, args: List[str]):
        """Show system logs."""
        log_level = "INFO"
        lines = 50
        
        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == "--level" and i + 1 < len(args):
                log_level = args[i + 1].upper()
                i += 2
            elif args[i] == "--lines" and i + 1 < len(args):
                lines = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        try:
            logs = await self.system_manager.get_recent_logs(
                level=log_level,
                limit=lines
            )
            
            print(f"Recent Logs ({log_level} level, last {lines} entries)")
            print("=" * 80)
            
            for log_entry in logs:
                timestamp = log_entry.get('timestamp', '')
                level = log_entry.get('level', 'INFO')
                message = log_entry.get('message', '')
                
                level_icon = {
                    'ERROR': '❌',
                    'WARNING': '⚠️',
                    'INFO': 'ℹ️',
                    'DEBUG': '🐛'
                }.get(level, 'ℹ️')
                
                print(f"{level_icon} {timestamp} [{level}] {message}")
                
        except Exception as e:
            self.print_error(f"Failed to get logs: {e}")
    
    async def cmd_health(self, args: List[str]):
        """Show detailed health information."""
        if not self.system_manager:
            self.print_error("System manager not initialized")
            return
        
        try:
            health = await self.system_manager.get_detailed_health()
            
            print("System Health Report")
            print("=" * 50)
            
            # Overall status
            overall_status = "Healthy" if health.get('healthy', False) else "Unhealthy"
            status_icon = "🟢" if health.get('healthy', False) else "🔴"
            print(f"Overall Status: {status_icon} {overall_status}")
            print()
            
            # Component details
            components = health.get('components', {})
            print("Component Health:")
            for component, details in components.items():
                status = details.get('status', 'unknown')
                status_icon = "🟢" if status == 'healthy' else "🔴"
                print(f"  {component}: {status_icon} {status}")
                
                if 'last_check' in details:
                    print(f"    Last Check: {details['last_check']}")
                if 'response_time' in details:
                    print(f"    Response Time: {details['response_time']:.2f}ms")
            print()
            
            # Performance metrics
            metrics = health.get('metrics', {})
            if metrics:
                print("Performance Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")
            
        except Exception as e:
            self.print_error(f"Failed to get health information: {e}")
    
    async def cmd_clear(self, args: List[str]):
        """Clear the screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    async def cmd_history(self, args: List[str]):
        """Show command history."""
        try:
            history_length = readline.get_current_history_length()
            lines = min(20, history_length)  # Show last 20 commands
            
            if args and args[0].isdigit():
                lines = min(int(args[0]), history_length)
            
            print(f"Command History (last {lines} commands)")
            print("=" * 50)
            
            for i in range(max(1, history_length - lines + 1), history_length + 1):
                command = readline.get_history_item(i)
                if command:
                    print(f"{i:3d}: {command}")
                    
        except Exception as e:
            self.print_error(f"Could not retrieve command history: {e}")
    
    async def cmd_version(self, args: List[str]):
        """Show version information."""
        print("Ultra AI System")
        print("=" * 30)
        print(f"Version: {self.config.get('version', '1.0.0')}")
        print(f"Build: {self.config.get('build', 'development')}")
        print(f"Python: {sys.version}")
        print()
        
        # Component versions
        if self.system_manager:
            components = await self.system_manager.get_component_versions()
            print("Component Versions:")
            for component, version in components.items():
                print(f"  {component}: {version}")
    
    async def cmd_exit(self, args: List[str]):
        """Exit the CLI."""
        if self.interactive_mode:
            print("Goodbye!")
        self.running = False
    
    def get_command_description(self, command: str) -> str:
        """Get short description for a command."""
        descriptions = {
            'help': 'Show help information',
            'status': 'Show system status',
            'agents': 'List available agents',
            'task': 'Create or manage tasks',
            'tasks': 'List tasks',
            'models': 'Show available AI models',
            'config': 'Show configuration',
            'logs': 'Show system logs',
            'health': 'Show detailed health information',
            'clear': 'Clear the screen',
            'history': 'Show command history',
            'version': 'Show version information',
            'exit': 'Exit the CLI',
            'quit': 'Exit the CLI'
        }
        return descriptions.get(command, 'No description available')
    
    def get_command_help(self, command: str) -> Optional[str]:
        """Get detailed help for a command."""
        help_texts = {
            'task': """
Task Management Commands:

task create <agent> <description>  - Create a new task
task get <task_id>                 - Get task information
task cancel <task_id>              - Cancel a running task
task retry <task_id>               - Retry a failed task

Examples:
  task create code "Generate a Python function"
  task get abc123
  task cancel abc123
            """,
            'tasks': """
List Tasks:

tasks [options]

Options:
  --status <status>    Filter by status (pending, running, completed, failed, cancelled)
  --agent <agent>      Filter by agent type
  --limit <number>     Limit number of results (default: 20)

Examples:
  tasks --status completed
  tasks --agent code --limit 10
            """,
            'logs': """
Show System Logs:

logs [options]

Options:
  --level <level>     Filter by log level (ERROR, WARNING, INFO, DEBUG)
  --lines <number>    Number of lines to show (default: 50)

Examples:
  logs --level ERROR
  logs --lines 100
            """
        }
        return help_texts.get(command)
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from configuration."""
        sanitized = {}
        sensitive_keys = {'api_key', 'secret', 'password', 'token', 'key'}
        
        for key, value in config.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_config(value)
            elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***HIDDEN***"
            else:
                sanitized[key] = value
        
        return sanitized


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Ultra AI System Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--command', '-c',
        type=str,
        help='Execute a single command and exit'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    # Direct command support
    parser.add_argument(
        '--agent',
        type=str,
        help='Agent type for direct task creation'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        help='Task description for direct task creation'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--health',
        action='store_true',
        help='Show health information and exit'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information and exit'
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Load configuration
    try:
        if args.config and os.path.exists(args.config):
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Initialize CLI
    cli = CLIInterface(config)
    
    try:
        await cli.initialize()
        
        # Handle different execution modes
        if args.interactive or (not args.command and not args.agent and not args.status and not args.health and not args.version):
            # Interactive mode
            await cli.run_interactive()
        
        elif args.command:
            # Single command mode
            await cli.run_command(args.command)
        
        elif args.agent and args.task:
            # Direct task creation
            command = f"task create {args.agent} {args.task}"
            await cli.run_command(command)
        
        elif args.status:
            # Status check
            await cli.run_command("status")
        
        elif args.health:
            # Health check
            await cli.run_command("health")
        
        elif args.version:
            # Version info
            await cli.run_command("version")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
