#!/usr/bin/env python3
"""
Ultra AI Intelligent Command Processor
Automatically detects and executes API functions based on user voice/text commands
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from integrations.termux_integration import termux_integration
    from tools.simple_tools import SimpleCodeExecutor, SimpleMessageFormatter
    termux_available = True
except ImportError:
    termux_available = False
    logging.warning("Termux integration not available for command processor")

class IntelligentCommandProcessor:
    """Process user commands and automatically trigger appropriate API functions."""
    
    def __init__(self):
        self.command_patterns = self._initialize_command_patterns()
        self.code_executor = SimpleCodeExecutor() if termux_available else None
        self.message_formatter = SimpleMessageFormatter() if termux_available else None
        
        logging.info("🧠 Intelligent Command Processor initialized")
    
    def _initialize_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for detecting user intents and commands."""
        
        return {
            # Battery commands
            'battery': {
                'patterns': [
                    r'(?:check|get|show|what.*is|how.*is).*battery',
                    r'battery.*(?:status|level|percentage|charge|power)',
                    r'how much.*(?:battery|power|charge)',
                    r'(?:phone|device).*(?:battery|power)',
                    r'(?:power|energy).*(?:left|remaining)'
                ],
                'keywords': ['battery', 'power', 'charge', 'energy', 'percentage'],
                'function': 'get_battery_status',
                'description': 'Get device battery information'
            },
            
            # Location commands  
            'location': {
                'patterns': [
                    r'(?:where|what.*is).*(?:location|position|place)',
                    r'(?:get|find|show).*(?:location|position|coordinates|gps)',
                    r'(?:current|my).*location',
                    r'where.*(?:am i|are we)',
                    r'gps.*(?:coordinates|position|data)'
                ],
                'keywords': ['location', 'gps', 'coordinates', 'position', 'where'],
                'function': 'get_location',
                'description': 'Get current location coordinates'
            },
            
            # WiFi commands
            'wifi': {
                'patterns': [
                    r'(?:check|get|show).*wifi',
                    r'wifi.*(?:status|info|connection|signal)',
                    r'(?:network|internet).*(?:connection|status|info)',
                    r'connected.*(?:network|wifi)',
                    r'scan.*(?:wifi|networks)'
                ],
                'keywords': ['wifi', 'network', 'internet', 'connection', 'signal'],
                'function': 'get_wifi_info',
                'description': 'Get WiFi connection information'
            },
            
            # Clipboard commands
            'clipboard': {
                'patterns': [
                    r'(?:get|check|show|paste).*clipboard',
                    r'clipboard.*(?:content|data|text)',
                    r'what.*(?:copied|clipboard)',
                    r'paste.*(?:clipboard|copied)',
                    r'copy.*(?:to|from).*clipboard'
                ],
                'keywords': ['clipboard', 'copy', 'paste', 'copied'],
                'function': 'get_clipboard',
                'description': 'Get clipboard content'
            },
            
            # Notification commands
            'notification': {
                'patterns': [
                    r'(?:send|create|show).*notification',
                    r'notify.*(?:me|user)',
                    r'alert.*(?:me|user|system)',
                    r'remind.*(?:me|user)'
                ],
                'keywords': ['notify', 'notification', 'alert', 'remind', 'message'],
                'function': 'send_notification',
                'description': 'Send system notification'
            },
            
            # Vibration commands
            'vibrate': {
                'patterns': [
                    r'(?:vibrate|buzz).*(?:phone|device)',
                    r'make.*(?:phone|device).*(?:vibrate|buzz)',
                    r'haptic.*(?:feedback|vibration)',
                    r'(?:phone|device).*(?:vibration|buzz)'
                ],
                'keywords': ['vibrate', 'buzz', 'haptic', 'shake'],
                'function': 'vibrate_device',
                'description': 'Vibrate the device'
            },
            
            # Torch/Flashlight commands
            'torch': {
                'patterns': [
                    r'(?:turn on|enable|activate).*(?:torch|flashlight|light)',
                    r'(?:torch|flashlight).*(?:on|off)',
                    r'(?:toggle|flash).*(?:torch|flashlight|light)',
                    r'use.*(?:flashlight|torch)'
                ],
                'keywords': ['torch', 'flashlight', 'light', 'flash', 'lamp'],
                'function': 'toggle_torch',
                'description': 'Toggle device flashlight'
            },
            
            # Voice commands
            'voice': {
                'patterns': [
                    r'(?:start|enable|activate).*voice',
                    r'voice.*(?:activation|recognition|listening)',
                    r'(?:listen|hear).*(?:voice|commands)',
                    r'(?:start|begin).*listening',
                    r'wake.*word.*(?:detection|activation)'
                ],
                'keywords': ['voice', 'listen', 'speech', 'activation', 'wake'],
                'function': 'control_voice',
                'description': 'Control voice activation'
            },
            
            # Code execution commands
            'execute': {
                'patterns': [
                    r'(?:run|execute|eval).*(?:code|python|script)',
                    r'(?:python|code).*(?:execution|run|eval)',
                    r'(?:calculate|compute|eval).*(?:expression|formula)',
                    r'run.*this.*code',
                    r'execute.*(?:command|script|program)'
                ],
                'keywords': ['execute', 'run', 'code', 'python', 'eval', 'calculate'],
                'function': 'execute_code',
                'description': 'Execute Python code'
            },
            
            # Text formatting commands
            'format': {
                'patterns': [
                    r'(?:format|style).*text',
                    r'(?:uppercase|lowercase|title).*(?:text|message)',
                    r'(?:bold|italic|code).*(?:text|format)',
                    r'convert.*(?:text|string).*(?:format|style)',
                    r'text.*(?:formatting|styling|conversion)'
                ],
                'keywords': ['format', 'style', 'uppercase', 'lowercase', 'bold', 'italic'],
                'function': 'format_text',
                'description': 'Format text messages'
            },
            
            # Sensor commands
            'sensors': {
                'patterns': [
                    r'(?:get|read|check).*sensor',
                    r'sensor.*(?:data|readings|values)',
                    r'(?:accelerometer|gyroscope|light|proximity)',
                    r'(?:motion|movement|orientation).*sensor',
                    r'device.*sensors'
                ],
                'keywords': ['sensor', 'accelerometer', 'gyroscope', 'light', 'proximity'],
                'function': 'get_sensor_data',
                'description': 'Get sensor data'
            }
        }
    
    async def process_command(self, message: str, source: str = "text") -> Dict[str, Any]:
        """
        Process user message and execute appropriate API functions.
        
        Args:
            message: User's message/command
            source: Source of command ('text' or 'voice')
        
        Returns:
            Dict containing results of executed commands and AI response
        """
        
        message_lower = message.lower()
        executed_commands = []
        command_results = []
        
        # Detect and execute commands
        for command_type, config in self.command_patterns.items():
            if self._matches_command(message_lower, config):
                try:
                    result = await self._execute_command(command_type, message, config)
                    if result:
                        executed_commands.append(command_type)
                        command_results.append({
                            'command': command_type,
                            'description': config['description'],
                            'result': result
                        })
                        logging.info(f"🎯 Executed {command_type} command from {source}")
                except Exception as e:
                    logging.error(f"Command execution error ({command_type}): {e}")
                    command_results.append({
                        'command': command_type,
                        'description': config['description'],
                        'result': {'success': False, 'error': str(e)}
                    })
        
        # Generate intelligent response
        response_text = await self._generate_response(message, executed_commands, command_results, source)
        
        return {
            'message': message,
            'source': source,
            'executed_commands': executed_commands,
            'command_results': command_results,
            'response': response_text,
            'success': True
        }
    
    def _matches_command(self, message: str, config: Dict[str, Any]) -> bool:
        """Check if message matches command patterns."""
        
        # Check regex patterns
        for pattern in config['patterns']:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        
        # Check keywords (require at least 2 matches for better accuracy)
        keyword_matches = sum(1 for keyword in config['keywords'] if keyword in message)
        if keyword_matches >= 2:
            return True
        
        # Single strong keyword match
        strong_keywords = config['keywords'][:2]  # First 2 are usually strongest
        if any(keyword in message for keyword in strong_keywords):
            return True
        
        return False
    
    async def _execute_command(self, command_type: str, message: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the specified command."""
        
        if not termux_available:
            return {'success': False, 'error': 'System APIs not available'}
        
        try:
            if command_type == 'battery':
                return await termux_integration.get_battery_status()
            
            elif command_type == 'location':
                return await termux_integration.get_location()
            
            elif command_type == 'wifi':
                return await termux_integration.get_wifi_info()
            
            elif command_type == 'clipboard':
                return await termux_integration.get_clipboard()
            
            elif command_type == 'notification':
                title, content = self._extract_notification_content(message)
                return await termux_integration.send_notification(title, content)
            
            elif command_type == 'vibrate':
                duration = self._extract_vibration_duration(message)
                return await termux_integration.vibrate(duration)
            
            elif command_type == 'torch':
                return await termux_integration.toggle_torch()
            
            elif command_type == 'voice':
                # Lazy import to avoid circular dependency
                from voice.voice_activation import voice_activation
                if 'start' in message.lower() or 'enable' in message.lower():
                    return await voice_activation.start_listening()
                else:
                    return await voice_activation.stop_listening()
            
            elif command_type == 'execute':
                code = self._extract_code(message)
                if code and self.code_executor:
                    return self.code_executor.execute_code(code)
                return {'success': False, 'error': 'No valid code found'}
            
            elif command_type == 'format':
                text, format_type = self._extract_format_request(message)
                if text and format_type and self.message_formatter:
                    return self.message_formatter.format_message(text, format_type)
                return {'success': False, 'error': 'No valid format request found'}
            
            elif command_type == 'sensors':
                sensor_type = self._extract_sensor_type(message)
                return await termux_integration.get_sensor_data(sensor_type)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        return None
    
    def _extract_notification_content(self, message: str) -> tuple:
        """Extract notification title and content from message."""
        # Simple extraction - can be made more sophisticated
        if 'notify' in message.lower():
            parts = message.split('notify', 1)
            if len(parts) > 1:
                content = parts[1].strip(' me about that').strip()
                return "Ultra AI Notification", content
        
        return "Ultra AI", "Notification sent as requested"
    
    def _extract_vibration_duration(self, message: str) -> int:
        """Extract vibration duration from message."""
        # Look for numbers in the message
        import re
        numbers = re.findall(r'\d+', message)
        if numbers:
            duration = int(numbers[0])
            # Reasonable limits
            return max(100, min(5000, duration))
        return 500  # Default 500ms
    
    def _extract_code(self, message: str) -> Optional[str]:
        """Extract code from message."""
        # Look for code blocks or python-like statements
        if '```' in message:
            parts = message.split('```')
            if len(parts) >= 3:
                return parts[1].strip()
        
        # Look for simple expressions
        if any(op in message for op in ['+', '-', '*', '/', '=', 'print(']):
            # Extract likely code portions
            sentences = message.split('.')
            for sentence in sentences:
                if any(op in sentence for op in ['+', '-', '*', '/', 'print(', '=']):
                    return sentence.strip()
        
        return None
    
    def _extract_format_request(self, message: str) -> tuple:
        """Extract text and format type from message."""
        text_to_format = None
        format_type = 'plain'
        
        # Detect format type
        if 'uppercase' in message.lower():
            format_type = 'uppercase'
        elif 'lowercase' in message.lower():
            format_type = 'lowercase'
        elif 'title' in message.lower():
            format_type = 'title'
        elif 'bold' in message.lower():
            format_type = 'bold'
        
        # Extract text (simple approach)
        if '"' in message:
            parts = message.split('"')
            if len(parts) >= 3:
                text_to_format = parts[1]
        
        return text_to_format, format_type
    
    def _extract_sensor_type(self, message: str) -> str:
        """Extract sensor type from message."""
        message_lower = message.lower()
        
        if 'light' in message_lower:
            return 'light'
        elif 'accelerometer' in message_lower or 'motion' in message_lower:
            return 'accelerometer'
        elif 'gyroscope' in message_lower:
            return 'gyroscope'
        elif 'proximity' in message_lower:
            return 'proximity'
        
        return 'light'  # Default sensor
    
    async def _generate_response(self, message: str, executed_commands: List[str], 
                                 command_results: List[Dict], source: str) -> str:
        """Generate intelligent response based on executed commands."""
        
        if not executed_commands:
            # No commands executed, return normal AI response
            return f"I understand you said: '{message}'. How can I help you with Ultra AI's capabilities?"
        
        # Build response based on executed commands
        response_parts = []
        
        if source == "voice":
            response_parts.append("I heard your voice command and executed the following:")
        else:
            response_parts.append("I understood your request and executed these actions:")
        
        for i, result in enumerate(command_results):
            command = result['command']
            description = result['description']
            cmd_result = result['result']
            
            if cmd_result.get('success', False):
                if command == 'battery' and 'summary' in cmd_result:
                    response_parts.append(f"• {description}: {cmd_result['summary']}")
                elif command == 'location' and 'summary' in cmd_result:
                    response_parts.append(f"• {description}: {cmd_result['summary']}")
                elif command == 'wifi' and 'summary' in cmd_result:
                    response_parts.append(f"• {description}: {cmd_result['summary']}")
                elif command == 'clipboard' and 'content' in cmd_result:
                    content = cmd_result['content'][:100]  # Limit length
                    response_parts.append(f"• {description}: {content}...")
                elif command == 'notification':
                    response_parts.append(f"• {description}: Notification sent successfully")
                elif command == 'vibrate':
                    response_parts.append(f"• {description}: Device vibration activated")
                elif command == 'torch':
                    response_parts.append(f"• {description}: Flashlight toggled")
                elif command == 'voice':
                    response_parts.append(f"• {description}: Voice system updated")
                elif command == 'execute' and 'output' in cmd_result:
                    response_parts.append(f"• {description}: {cmd_result['output']}")
                elif command == 'format' and 'formatted_text' in cmd_result:
                    response_parts.append(f"• {description}: {cmd_result['formatted_text']}")
                elif command == 'sensors' and 'data' in cmd_result:
                    response_parts.append(f"• {description}: Sensor data retrieved")
                else:
                    response_parts.append(f"• {description}: Completed successfully")
            else:
                error_msg = cmd_result.get('error', 'Unknown error')
                response_parts.append(f"• {description}: Failed ({error_msg})")
        
        # Add helpful closing
        if len(executed_commands) == 1:
            response_parts.append("\nIs there anything else you'd like me to help you with?")
        else:
            response_parts.append(f"\nI executed {len(executed_commands)} commands for you. What would you like to do next?")
        
        return "\n".join(response_parts)
    
    def get_available_commands(self) -> Dict[str, str]:
        """Get list of available commands and their descriptions."""
        return {cmd: config['description'] for cmd, config in self.command_patterns.items()}

# Global command processor instance
command_processor = IntelligentCommandProcessor()

if __name__ == "__main__":
    # Test command processor
    async def test_command_processor():
        print("🧠 Ultra AI Intelligent Command Processor Test")
        print("=" * 60)
        
        test_commands = [
            "Check my battery level",
            "Where am I right now?",
            "Get WiFi status",
            "What's in my clipboard?",
            "Notify me about this meeting",
            "Make the phone vibrate",
            "Turn on the flashlight",
            "Start voice activation",
            "Execute this code: print('Hello AI!')",
            "Format this text to uppercase: hello world",
            "Get light sensor data"
        ]
        
        print("📋 Available Commands:")
        for cmd, desc in command_processor.get_available_commands().items():
            print(f"  • {cmd}: {desc}")
        
        print("\n🧪 Testing Command Recognition:")
        
        for command in test_commands:
            print(f"\n📝 Command: '{command}'")
            result = await command_processor.process_command(command)
            
            if result['executed_commands']:
                print(f"   ✓ Detected: {', '.join(result['executed_commands'])}")
                print(f"   🤖 Response: {result['response'][:100]}...")
            else:
                print(f"   ⚪ No specific commands detected")
    
    asyncio.run(test_command_processor())