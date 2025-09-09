#!/usr/bin/env python3
"""
Ultra AI Termux Integration
Real integration with all Termux APIs for device functionality
"""

import subprocess
import json
import asyncio
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

class TermuxIntegration:
    """Complete Termux API integration for Ultra AI."""
    
    def __init__(self):
        self.available_apis = {}
        self._check_termux_apis()
        logging.info(f"✓ TermuxIntegration initialized with {len(self.available_apis)} APIs")
    
    def _check_termux_apis(self):
        """Check which Termux APIs are available on the system."""
        
        # Common Termux API commands
        api_commands = [
            'termux-battery-status',
            'termux-clipboard-get', 'termux-clipboard-set',
            'termux-contact-list',
            'termux-dialog',
            'termux-location', 
            'termux-notification',
            'termux-sensor',
            'termux-share',
            'termux-sms-list', 'termux-sms-send',
            'termux-speech-to-text',
            'termux-storage-get',
            'termux-telephony-call', 'termux-telephony-cellinfo',
            'termux-toast',
            'termux-torch',
            'termux-vibrate',
            'termux-volume',
            'termux-wifi-connectioninfo', 'termux-wifi-scaninfo'
        ]
        
        for cmd in api_commands:
            try:
                result = subprocess.run(['which', cmd], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    self.available_apis[cmd] = result.stdout.strip()
            except Exception:
                pass
    
    def is_api_available(self, api_name: str) -> bool:
        """Check if a specific Termux API is available."""
        return api_name in self.available_apis
    
    async def get_battery_status(self) -> Dict[str, Any]:
        """Get device battery information."""
        if not self.is_api_available('termux-battery-status'):
            return {"success": False, "error": "Battery API not available"}
        
        try:
            result = subprocess.run(
                ['termux-battery-status'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                battery_data = json.loads(result.stdout)
                return {
                    "success": True,
                    "data": battery_data,
                    "summary": f"Battery: {battery_data.get('percentage', 'Unknown')}% - {battery_data.get('status', 'Unknown')}"
                }
            else:
                return {"success": False, "error": "Failed to get battery status"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_location(self) -> Dict[str, Any]:
        """Get device location."""
        if not self.is_api_available('termux-location'):
            return {"success": False, "error": "Location API not available"}
        
        try:
            result = subprocess.run(
                ['termux-location', '-p', 'gps'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                location_data = json.loads(result.stdout)
                return {
                    "success": True,
                    "data": location_data,
                    "summary": f"Location: {location_data.get('latitude', 'Unknown')}, {location_data.get('longitude', 'Unknown')}"
                }
            else:
                return {"success": False, "error": "Failed to get location"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def send_notification(self, title: str, content: str) -> Dict[str, Any]:
        """Send a system notification."""
        if not self.is_api_available('termux-notification'):
            return {"success": False, "error": "Notification API not available"}
        
        try:
            result = subprocess.run(
                ['termux-notification', '--title', title, '--content', content],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return {
                "success": result.returncode == 0,
                "message": f"Notification sent: {title}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def show_toast(self, message: str) -> Dict[str, Any]:
        """Show a toast message."""
        if not self.is_api_available('termux-toast'):
            return {"success": False, "error": "Toast API not available"}
        
        try:
            result = subprocess.run(
                ['termux-toast', message],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return {
                "success": result.returncode == 0,
                "message": f"Toast shown: {message}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def vibrate(self, duration: int = 1000) -> Dict[str, Any]:
        """Vibrate the device."""
        if not self.is_api_available('termux-vibrate'):
            return {"success": False, "error": "Vibrate API not available"}
        
        try:
            result = subprocess.run(
                ['termux-vibrate', '-d', str(duration)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return {
                "success": result.returncode == 0,
                "message": f"Device vibrated for {duration}ms"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_clipboard(self) -> Dict[str, Any]:
        """Get clipboard content."""
        if not self.is_api_available('termux-clipboard-get'):
            return {"success": False, "error": "Clipboard API not available"}
        
        try:
            result = subprocess.run(
                ['termux-clipboard-get'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "content": result.stdout.strip(),
                    "message": "Clipboard content retrieved"
                }
            else:
                return {"success": False, "error": "Failed to get clipboard"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def set_clipboard(self, content: str) -> Dict[str, Any]:
        """Set clipboard content."""
        if not self.is_api_available('termux-clipboard-set'):
            return {"success": False, "error": "Clipboard API not available"}
        
        try:
            result = subprocess.run(
                ['termux-clipboard-set'],
                input=content,
                text=True,
                capture_output=True,
                timeout=5
            )
            
            return {
                "success": result.returncode == 0,
                "message": f"Clipboard set: {content[:50]}..."
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_wifi_info(self) -> Dict[str, Any]:
        """Get WiFi connection information."""
        if not self.is_api_available('termux-wifi-connectioninfo'):
            return {"success": False, "error": "WiFi API not available"}
        
        try:
            result = subprocess.run(
                ['termux-wifi-connectioninfo'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                wifi_data = json.loads(result.stdout)
                return {
                    "success": True,
                    "data": wifi_data,
                    "summary": f"WiFi: {wifi_data.get('ssid', 'Unknown')} - {wifi_data.get('rssi', 'Unknown')} dBm"
                }
            else:
                return {"success": False, "error": "Failed to get WiFi info"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def scan_wifi(self) -> Dict[str, Any]:
        """Scan for available WiFi networks."""
        if not self.is_api_available('termux-wifi-scaninfo'):
            return {"success": False, "error": "WiFi scan API not available"}
        
        try:
            result = subprocess.run(
                ['termux-wifi-scaninfo'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                scan_data = json.loads(result.stdout)
                return {
                    "success": True,
                    "data": scan_data,
                    "count": len(scan_data),
                    "message": f"Found {len(scan_data)} WiFi networks"
                }
            else:
                return {"success": False, "error": "Failed to scan WiFi"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_sensor_data(self, sensor: str = "light", delay: int = 1000) -> Dict[str, Any]:
        """Get sensor data from device."""
        if not self.is_api_available('termux-sensor'):
            return {"success": False, "error": "Sensor API not available"}
        
        try:
            result = subprocess.run(
                ['termux-sensor', '-s', sensor, '-d', str(delay), '-n', '1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                sensor_data = json.loads(result.stdout)
                return {
                    "success": True,
                    "data": sensor_data,
                    "sensor": sensor,
                    "message": f"{sensor} sensor data retrieved"
                }
            else:
                return {"success": False, "error": f"Failed to get {sensor} sensor data"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def toggle_torch(self) -> Dict[str, Any]:
        """Toggle device torch/flashlight."""
        if not self.is_api_available('termux-torch'):
            return {"success": False, "error": "Torch API not available"}
        
        try:
            result = subprocess.run(
                ['termux-torch', 'on'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0:
                # Turn it off after 2 seconds
                await asyncio.sleep(2)
                subprocess.run(['termux-torch', 'off'], timeout=3)
                return {
                    "success": True,
                    "message": "Torch toggled (on for 2 seconds)"
                }
            else:
                return {"success": False, "error": "Failed to toggle torch"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def speech_to_text(self) -> Dict[str, Any]:
        """Convert speech to text."""
        if not self.is_api_available('termux-speech-to-text'):
            return {"success": False, "error": "Speech-to-text API not available"}
        
        try:
            result = subprocess.run(
                ['termux-speech-to-text'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                text = result.stdout.strip()
                return {
                    "success": True,
                    "text": text,
                    "message": f"Speech recognized: {text}"
                }
            else:
                return {"success": False, "error": "Failed to recognize speech"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_available_apis(self) -> List[str]:
        """Get list of available Termux APIs."""
        return list(self.available_apis.keys())
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "available_apis": len(self.available_apis),
            "api_list": list(self.available_apis.keys()),
            "termux_version": self._get_termux_version(),
            "android_info": self._get_android_info()
        }
    
    def _get_termux_version(self) -> str:
        """Get Termux version."""
        try:
            result = subprocess.run(
                ['getprop', 'ro.build.version.release'],
                capture_output=True,
                text=True,
                timeout=3
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except:
            return "Unknown"
    
    def _get_android_info(self) -> Dict[str, str]:
        """Get Android system information."""
        try:
            info = {}
            
            # Get Android version
            result = subprocess.run(['getprop', 'ro.build.version.release'], capture_output=True, text=True, timeout=2)
            info['android_version'] = result.stdout.strip() if result.returncode == 0 else "Unknown"
            
            # Get device model
            result = subprocess.run(['getprop', 'ro.product.model'], capture_output=True, text=True, timeout=2)
            info['device_model'] = result.stdout.strip() if result.returncode == 0 else "Unknown"
            
            # Get manufacturer
            result = subprocess.run(['getprop', 'ro.product.manufacturer'], capture_output=True, text=True, timeout=2)
            info['manufacturer'] = result.stdout.strip() if result.returncode == 0 else "Unknown"
            
            return info
        except:
            return {"android_version": "Unknown", "device_model": "Unknown", "manufacturer": "Unknown"}

# Global Termux integration instance
termux_integration = TermuxIntegration()

if __name__ == "__main__":
    # Test Termux integration
    async def test_termux_integration():
        print("🔧 Ultra AI Termux Integration Test")
        print("=" * 50)
        
        print(f"Available APIs: {len(termux_integration.available_apis)}")
        for api in termux_integration.get_available_apis():
            print(f"  ✓ {api}")
        
        print("\\n📱 System Information:")
        info = termux_integration.get_system_info()
        for key, value in info.items():
            if key != 'api_list':
                print(f"  {key}: {value}")
        
        # Test basic functionality
        print("\\n🔋 Testing Battery Status:")
        battery = await termux_integration.get_battery_status()
        if battery["success"]:
            print(f"  ✓ {battery['summary']}")
        else:
            print(f"  ✗ {battery['error']}")
        
        print("\\n📋 Testing Clipboard:")
        clipboard = await termux_integration.get_clipboard()
        if clipboard["success"]:
            print(f"  ✓ Clipboard: {clipboard['content'][:50]}...")
        else:
            print(f"  ✗ {clipboard['error']}")
        
        print("\\n📡 Testing WiFi Info:")
        wifi = await termux_integration.get_wifi_info()
        if wifi["success"]:
            print(f"  ✓ {wifi['summary']}")
        else:
            print(f"  ✗ {wifi['error']}")
    
    asyncio.run(test_termux_integration())