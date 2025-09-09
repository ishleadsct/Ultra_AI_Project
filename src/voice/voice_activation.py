#!/usr/bin/env python3
"""
Ultra AI Voice Activation System
Real voice recognition and activation using Termux APIs
"""

import asyncio
import subprocess
import json
import threading
import time
import logging
from typing import Dict, Any, Callable, Optional
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from integrations.termux_integration import termux_integration
    from ai.production_ai import get_production_ai_response
    termux_available = True
    logging.info("✓ Voice system with intelligent command processing")
except ImportError:
    termux_available = False
    logging.warning("Termux integration not available")

class VoiceActivation:
    """Complete voice activation system with wake word detection."""
    
    def __init__(self):
        self.is_listening = False
        self.wake_words = ["ultra ai", "ultra", "hey ultra", "ok ultra"]
        self.listening_thread = None
        self.voice_callback = None
        self.activation_callback = None
        
        # Voice recognition settings
        self.recognition_timeout = 10  # seconds
        self.silence_timeout = 3      # seconds
        
        logging.info("🎤 VoiceActivation system initialized")
    
    def set_voice_callback(self, callback: Callable):
        """Set callback for voice commands."""
        self.voice_callback = callback
    
    def set_activation_callback(self, callback: Callable):
        """Set callback for activation events."""
        self.activation_callback = callback
    
    async def start_listening(self) -> Dict[str, Any]:
        """Start continuous voice listening for wake words."""
        if not termux_available:
            return {"success": False, "error": "Voice recognition not available"}
        
        if not termux_integration.is_api_available('termux-speech-to-text'):
            return {"success": False, "error": "Speech-to-text API not available"}
        
        if self.is_listening:
            return {"success": False, "error": "Already listening"}
        
        self.is_listening = True
        
        # Start listening in a separate thread
        self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listening_thread.start()
        
        if self.activation_callback:
            await self.activation_callback("started_listening")
        
        return {
            "success": True,
            "message": "Voice activation started",
            "wake_words": self.wake_words
        }
    
    async def stop_listening(self) -> Dict[str, Any]:
        """Stop voice listening."""
        if not self.is_listening:
            return {"success": False, "error": "Not currently listening"}
        
        self.is_listening = False
        
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2)
        
        if self.activation_callback:
            await self.activation_callback("stopped_listening")
        
        return {
            "success": True,
            "message": "Voice activation stopped"
        }
    
    def _listen_loop(self):
        """Main listening loop running in separate thread."""
        while self.is_listening:
            try:
                # Get speech input
                result = subprocess.run(
                    ['termux-speech-to-text'],
                    capture_output=True,
                    text=True,
                    timeout=self.recognition_timeout
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout.strip().lower()
                    logging.info(f"🎤 Heard: {text}")
                    
                    # Check for wake words
                    if self._contains_wake_word(text):
                        logging.info("🚀 Wake word detected!")
                        asyncio.create_task(self._handle_activation(text))
                    
                # Small delay to prevent excessive CPU usage
                time.sleep(0.5)
                
            except subprocess.TimeoutExpired:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                logging.error(f"Voice listening error: {e}")
                # Brief pause before retrying
                time.sleep(2)
    
    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains any wake words."""
        text = text.lower()
        return any(wake_word in text for wake_word in self.wake_words)
    
    async def _handle_activation(self, text: str):
        """Handle wake word activation."""
        try:
            # Send activation notification
            if termux_integration.is_api_available('termux-notification'):
                await termux_integration.send_notification(
                    "Ultra AI Activated",
                    "I'm listening for your command..."
                )
            
            # Vibrate to confirm activation
            if termux_integration.is_api_available('termux-vibrate'):
                await termux_integration.vibrate(200)
            
            # Get the actual command (text after wake word)
            command = self._extract_command(text)
            
            if command:
                # Process the voice command
                await self._process_voice_command(command)
            else:
                # Wait for follow-up command
                await self._get_follow_up_command()
            
        except Exception as e:
            logging.error(f"Activation handling error: {e}")
    
    def _extract_command(self, text: str) -> Optional[str]:
        """Extract command from text after wake word."""
        text = text.lower()
        
        for wake_word in self.wake_words:
            if wake_word in text:
                # Find the wake word and get text after it
                wake_word_index = text.find(wake_word)
                command_start = wake_word_index + len(wake_word)
                command = text[command_start:].strip()
                
                # Remove common filler words
                command = command.replace("please", "").replace("can you", "").strip()
                
                if len(command) > 3:  # Minimum command length
                    return command
        
        return None
    
    async def _get_follow_up_command(self):
        """Get follow-up command after activation."""
        try:
            # Show toast to indicate listening
            if termux_integration.is_api_available('termux-toast'):
                await termux_integration.show_toast("Ultra AI listening...")
            
            # Get speech input with shorter timeout
            result = subprocess.run(
                ['termux-speech-to-text'],
                capture_output=True,
                text=True,
                timeout=self.silence_timeout
            )
            
            if result.returncode == 0 and result.stdout.strip():
                command = result.stdout.strip()
                await self._process_voice_command(command)
            else:
                # No follow-up command received
                if termux_integration.is_api_available('termux-toast'):
                    await termux_integration.show_toast("Ultra AI: No command received")
            
        except subprocess.TimeoutExpired:
            # Timeout waiting for command
            if termux_integration.is_api_available('termux-toast'):
                await termux_integration.show_toast("Ultra AI: Listening timeout")
        except Exception as e:
            logging.error(f"Follow-up command error: {e}")
    
    async def _process_voice_command(self, command: str):
        """Process a voice command using Ultra AI with intelligent command processing."""
        try:
            logging.info(f"🎤 Processing voice command: {command}")
            
            # Show processing notification
            if termux_integration.is_api_available('termux-toast'):
                await termux_integration.show_toast(f"Ultra AI: {command[:25]}...")
            
            # Use intelligent command processor for voice commands
            if termux_available:
                # Lazy import to avoid circular dependency
                from ai.command_processor import command_processor
                # First try the command processor for API functions
                command_result = await command_processor.process_command(command, "voice")
                
                if command_result.get("executed_commands"):
                    # Commands were executed, use the command processor response
                    ai_response = command_result["response"]
                    executed_commands = command_result["executed_commands"]
                    
                    logging.info(f"🎯 Voice executed {len(executed_commands)} commands: {executed_commands}")
                    
                    # Provide audio feedback for voice commands
                    if termux_integration.is_api_available('termux-vibrate'):
                        await termux_integration.vibrate(300)  # Confirm command execution
                    
                    # Send detailed notification for voice commands
                    if termux_integration.is_api_available('termux-notification'):
                        title = f"Ultra AI: {len(executed_commands)} Action{'s' if len(executed_commands) > 1 else ''}"
                        content = f"✓ {', '.join(executed_commands)}\n{ai_response[:80]}..."
                        await termux_integration.send_notification(title, content)
                    
                    # Call voice callback with enhanced info
                    if self.voice_callback:
                        await self.voice_callback({
                            "command": command,
                            "response": ai_response,
                            "source": "voice",
                            "executed_commands": executed_commands,
                            "command_results": command_result["command_results"]
                        })
                    
                    logging.info(f"🤖 Voice AI Response with {len(executed_commands)} actions")
                    return
            
            # No specific commands detected, use regular AI response
            response = await get_production_ai_response(f"Voice: {command}")
            
            if response["success"]:
                ai_response = response["response"]
                
                # Standard voice response feedback
                if termux_integration.is_api_available('termux-notification'):
                    await termux_integration.send_notification(
                        "Ultra AI Voice Response",
                        ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
                    )
                
                # Call voice callback
                if self.voice_callback:
                    await self.voice_callback({
                        "command": command,
                        "response": ai_response,
                        "source": "voice",
                        "model": response.get("model", "unknown")
                    })
                
                logging.info(f"🤖 Voice AI Response: {ai_response[:50]}...")
                
            else:
                # Error handling
                error_msg = "Sorry, I couldn't process that voice command"
                if termux_integration.is_api_available('termux-notification'):
                    await termux_integration.send_notification("Ultra AI Voice Error", error_msg)
                
                logging.error(f"Voice AI response error: {response.get('error', 'Unknown')}")
        
        except Exception as e:
            logging.error(f"Voice command processing error: {e}")
            if termux_integration.is_api_available('termux-toast'):
                await termux_integration.show_toast("Ultra AI: Voice processing failed")
    
    async def test_voice_recognition(self) -> Dict[str, Any]:
        """Test voice recognition functionality."""
        if not termux_integration.is_api_available('termux-speech-to-text'):
            return {"success": False, "error": "Speech-to-text not available"}
        
        try:
            if termux_integration.is_api_available('termux-toast'):
                await termux_integration.show_toast("Say something to test voice recognition...")
            
            result = subprocess.run(
                ['termux-speech-to-text'],
                capture_output=True,
                text=True,
                timeout=self.recognition_timeout
            )
            
            if result.returncode == 0:
                text = result.stdout.strip()
                return {
                    "success": True,
                    "recognized_text": text,
                    "message": f"Recognized: {text}"
                }
            else:
                return {"success": False, "error": "No speech recognized"}
        
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Voice recognition timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice activation system status."""
        return {
            "listening": self.is_listening,
            "wake_words": self.wake_words,
            "speech_to_text_available": termux_integration.is_api_available('termux-speech-to-text') if termux_available else False,
            "notifications_available": termux_integration.is_api_available('termux-notification') if termux_available else False,
            "toast_available": termux_integration.is_api_available('termux-toast') if termux_available else False,
            "vibrate_available": termux_integration.is_api_available('termux-vibrate') if termux_available else False,
            "system_ready": termux_available
        }

# Global voice activation instance
voice_activation = VoiceActivation()

if __name__ == "__main__":
    # Test voice activation system
    async def test_voice_system():
        print("🎤 Ultra AI Voice Activation Test")
        print("=" * 50)
        
        # Show status
        status = voice_activation.get_status()
        print("📊 System Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\\n🧪 Testing Voice Recognition:")
        test_result = await voice_activation.test_voice_recognition()
        if test_result["success"]:
            print(f"  ✓ {test_result['message']}")
        else:
            print(f"  ✗ {test_result['error']}")
        
        # Test wake word detection
        print("\\n🚀 Testing Wake Word Detection:")
        test_phrases = [
            "ultra ai turn on the lights",
            "hey ultra what's the weather",
            "ok ultra tell me a joke",
            "this has no wake word"
        ]
        
        for phrase in test_phrases:
            has_wake = voice_activation._contains_wake_word(phrase)
            print(f"  '{phrase}' -> Wake word: {'✓' if has_wake else '✗'}")
        
        print(f"\\n🎯 Wake words: {voice_activation.wake_words}")
    
    asyncio.run(test_voice_system())