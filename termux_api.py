    return FallbackConfig()class TermuxPermission(Enum): """Termux API permissions""" CAMERA = "android.permission.CAMERA" RECORD_AUDIO = "android.permission.RECORD_AUDIO" ACCESS_FINE_LOCATION = "android.permission.ACCESS_FINE_LOCATION" ACCESS_COARSE_LOCATION = "android.permission.ACCESS_COARSE_LOCATION" READ_EXTERNAL_STORAGE = "android.permission.READ_EXTERNAL_STORAGE" WRITE_EXTERNAL_STORAGE = "android.permission.WRITE_EXTERNAL_STORAGE" SEND_SMS = "android.permission.SEND_SMS" RECEIVE_SMS = "android.permission.RECEIVE_SMS" CALL_PHONE = "android.permission.CALL_PHONE" READ_PHONE_STATE = "android.permission.READ_PHONE_STATE" VIBRATE = "android.permission.VIBRATE" WAKE_LOCK = "android.permission.WAKE_LOCK"class NotificationPriority(Enum): """Notification priority levels""" MIN = "min" LOW = "low" DEFAULT = "default" HIGH = "high" MAX = "max"@dataclass class BatteryInfo: """Battery status information""" percentage: int health: str plugged: str status: str temperature: float voltage: int@dataclass class LocationInfo: """Location information""" latitude: float longitude: float altitude: Optional[float] accuracy: Optional[float] bearing: Optional[float] speed: Optional[float] timestamp: float@dataclass class ContactInfo: """Contact information""" name: str number: str display_name: Optional[str] = None@dataclass class SensorReading: """Sensor data reading""" sensor_type: str values: List[float] timestamp: float accuracy: Optional[int] = Noneclass TermuxAPIError(Exception): """Custom exception for Termux API errors""" passclass PermissionError(TermuxAPIError): """Permission denied error""" passclass TermuxAPI: """Main Termux API wrapper class"""def __init__(self, config=None):
    self.config = config or get_config()
    self.enabled = self.config.termux.enabled
    self.granted_permissions = set()
    self.callback_handlers = {}
    self.monitoring_threads = {}
    self.lock = threading.Lock()
    
    # Check if Termux API is available
    self._check_termux_availability()
    
    # Initialize permissions
    self._initialize_permissions()
    
    logger.info("Termux API wrapper initialized")

def _check_termux_availability(self):
    """Check if Termux API is available"""
    if not self.enabled:
        logger.info("Termux API disabled in configuration")
        return
    
    # Check if we're in Termux environment
    if not os.environ.get('PREFIX', '').find('termux') != -1:
        logger.warning("Not running in Termux environment")
        self.enabled = False
        return
    
    # Check if termux-api is installed
    try:
        result = subprocess.run(['which', 'termux-api-start'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("termux-api not found. Install with: pkg install termux-api")
            self.enabled = False
            return
    except Exception as e:
        logger.error(f"Error checking termux-api availability: {e}")
        self.enabled = False
        return
    
    logger.info("Termux API is available")

def _initialize_permissions(self):
    """Initialize and check permissions"""
    if not self.enabled:
        return
    
    # Auto-request permissions if enabled
    if self.config.termux.auto_permissions:
        for permission in self.config.termux.permissions:
            try:
                self.request_permission(TermuxPermission(permission))
            except Exception as e:
                logger.warning(f"Failed to auto-request permission {permission}: {e}")

def _execute_command(self, command: List[str], input_data: str = None, 
                    timeout: int = 30) -> Dict[str, Any]:
    """Execute a termux-api command safely"""
    if not self.enabled:
        raise TermuxAPIError("Termux API is not enabled or available")
    
    try:
        # Add termux-api prefix if not present
        if not command[0].startswith('termux-'):
            command[0] = f'termux-{command[0]}'
        
        # Execute command
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse output
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if "permission" in error_msg.lower():
                raise PermissionError(f"Permission denied: {error_msg}")
            else:
                raise TermuxAPIError(f"Command failed: {error_msg}")
        
        # Try to parse JSON output
        try:
            if result.stdout.strip():
                return json.loads(result.stdout)
            else:
                return {"success": True}
        except json.JSONDecodeError:
            return {"output": result.stdout.strip()}
            
    except subprocess.TimeoutExpired:
        raise TermuxAPIError(f"Command timed out after {timeout} seconds")
    except Exception as e:
        raise TermuxAPIError(f"Failed to execute command: {e}")

# Permission Management

def request_permission(self, permission: TermuxPermission) -> bool:
    """Request a specific permission"""
    try:
        # Note: Termux permissions are usually granted through the app settings
        # This is more of a check than an actual request
        result = self._execute_command(['permission-get', permission.value])
        
        if result.get('granted', False):
            self.granted_permissions.add(permission)
            logger.info(f"Permission granted: {permission.value}")
            return True
        else:
            logger.warning(f"Permission not granted: {permission.value}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to request permission {permission.value}: {e}")
        return False

def check_permission(self, permission: TermuxPermission) -> bool:
    """Check if a permission is granted"""
    try:
        # For some permissions, we can test by attempting to use them
        if permission == TermuxPermission.VIBRATE:
            self._execute_command(['vibrate', '1'])
            return True
        
        # For others, we assume they're available if no error occurs
        return permission in self.granted_permissions
        
    except PermissionError:
        return False
    except Exception:
        return True  # Assume available if we can't check

# Battery and Power Management

def get_battery_status(self) -> BatteryInfo:
    """Get current battery status"""
    try:
        result = self._execute_command(['battery-status'])
        
        return BatteryInfo(
            percentage=result.get('percentage', 0),
            health=result.get('health', 'unknown'),
            plugged=result.get('plugged', 'unknown'),
            status=result.get('status', 'unknown'),
            temperature=result.get('temperature', 0.0),
            voltage=result.get('voltage', 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get battery status: {e}")
        # Return default values
        return BatteryInfo(
            percentage=100, health='unknown', plugged='unknown',
            status='unknown', temperature=0.0, voltage=0
        )

def start_battery_monitoring(self, callback: Callable[[BatteryInfo], None], 
                            interval: int = 60):
    """Start monitoring battery status"""
    if 'battery' in self.monitoring_threads:
        self.stop_battery_monitoring()
    
    def monitor_battery():
        while 'battery' in self.monitoring_threads:
            try:
                battery_info = self.get_battery_status()
                callback(battery_info)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Battery monitoring error: {e}")
                time.sleep(interval)
    
    thread = threading.Thread(target=monitor_battery, daemon=True)
    self.monitoring_threads['battery'] = thread
    thread.start()
    logger.info("Battery monitoring started")

def stop_battery_monitoring(self):
    """Stop battery monitoring"""
    if 'battery' in self.monitoring_threads:
        del self.monitoring_threads['battery']
        logger.info("Battery monitoring stopped")

# Notifications

def send_notification(self, title: str, content: str = "", 
                     priority: NotificationPriority = NotificationPriority.DEFAULT,
                     action: str = None, button1: str = None, button2: str = None,
                     button3: str = None, id: str = None) -> bool:
    """Send a notification"""
    try:
        cmd = ['notification']
        
        # Add title and content
        cmd.extend(['--title', title])
        if content:
            cmd.extend(['--content', content])
        
        # Add priority
        cmd.extend(['--priority', priority.value])
        
        # Add action
        if action:
            cmd.extend(['--action', action])
        
        # Add buttons
        if button1:
            cmd.extend(['--button1', button1])
        if button2:
            cmd.extend(['--button2', button2])
        if button3:
            cmd.extend(['--button3', button3])
        
        # Add ID
        if id:
            cmd.extend(['--id', id])
        
        self._execute_command(cmd)
        logger.debug(f"Notification sent: {title}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False

def remove_notification(self, notification_id: str) -> bool:
    """Remove a notification"""
    try:
        self._execute_command(['notification-remove', notification_id])
        return True
    except Exception as e:
        logger.error(f"Failed to remove notification: {e}")
        return False

# Device Information

def get_device_info(self) -> Dict[str, Any]:
    """Get device information"""
    try:
        # Get various device information
        info = {}
        
        # Try to get basic device info
        try:
            result = self._execute_command(['telephony-deviceinfo'])
            info.update(result)
        except Exception:
            pass
        
        # Add battery info
        try:
            battery = self.get_battery_status()
            info['battery'] = asdict(battery)
        except Exception:
            pass
        
        # Add network info
        try:
            wifi = self._execute_command(['wifi-connectioninfo'])
            info['wifi'] = wifi
        except Exception:
            pass
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return {}

# Storage and File System

def setup_storage(self) -> bool:
    """Setup storage access"""
    try:
        self._execute_command(['storage-get'])
        return True
    except Exception as e:
        logger.error(f"Failed to setup storage: {e}")
        return False

def share_file(self, file_path: str, action: str = "send") -> bool:
    """Share a file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self._execute_command(['share', '--action', action, file_path])
        return True
        
    except Exception as e:
        logger.error(f"Failed to share file: {e}")
        return False

# Audio and Microphone

def play_sound(self, file_path: str) -> bool:
    """Play an audio file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        self._execute_command(['media-player', 'play', file_path])
        return True
        
    except Exception as e:
        logger.error(f"Failed to play sound: {e}")
        return False

def record_audio(self, output_file: str, duration: int = 10, 
                encoder: str = "aac") -> bool:
    """Record audio"""
    try:
        cmd = ['microphone-record', '-f', output_file, '-d', str(duration), '-e', encoder]
        self._execute_command(cmd)
        return True
        
    except Exception as e:
        logger.error(f"Failed to record audio: {e}")
        return False

def text_to_speech(self, text: str, language: str = "en", 
                  engine: str = None, pitch: float = 1.0, 
                  rate: float = 1.0) -> bool:
    """Convert text to speech"""
    try:
        cmd = ['tts-speak']
        cmd.extend(['-l', language])
        cmd.extend(['-p', str(pitch)])
        cmd.extend(['-r', str(rate)])
        
        if engine:
            cmd.extend(['-e', engine])
        
        cmd.append(text)
        
        self._execute_command(cmd)
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert text to speech: {e}")
        return False

# Vibration and Haptic Feedback

def vibrate(self, duration: int = 1000, force: bool = False) -> bool:
    """Vibrate the device"""
    try:
        cmd = ['vibrate']
        if force:
            cmd.append('-f')
        cmd.append(str(duration))
        
        self._execute_command(cmd)
        return True
        
    except Exception as e:
        logger.error(f"Failed to vibrate: {e}")
        return False

# Camera

def take_photo(self, output_file: str, camera: str = "0") -> bool:
    """Take a photo"""
    try:
        cmd = ['camera-photo', '-c', camera, output_file]
        self._execute_command(cmd)
        return True
        
    except Exception as e:
        logger.error(f"Failed to take photo: {e}")
        return False

def get_camera_info(self) -> Dict[str, Any]:
    """Get camera information"""
    try:
        return self._execute_command(['camera-info'])
    except Exception as e:
        logger.error(f"Failed to get camera info: {e}")
        return {}

# Location Services

def get_location(self, provider: str = "gps", timeout: int = 30) -> Optional[LocationInfo]:
    """Get current location"""
    try:
        cmd = ['location', '-p', provider, '-t', str(timeout)]
        result = self._execute_command(cmd)
        
        if 'latitude' in result and 'longitude' in result:
            return LocationInfo(
                latitude=result['latitude'],
                longitude=result['longitude'],
                altitude=result.get('altitude'),
                accuracy=result.get('accuracy'),
                bearing=result.get('bearing'),
                speed=result.get('speed'),
                timestamp=time.time()
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get location: {e}")
        return None

# Clipboard

def set_clipboard(self, text: str) -> bool:
    """Set clipboard content"""
    try:
        self._execute_command(['clipboard-set'], input_data=text)
        return True
    except Exception as e:
        logger.error(f"Failed to set clipboard: {e}")
        return False

def get_clipboard(self) -> str:
    """Get clipboard content"""
    try:
        result = self._execute_command(['clipboard-get'])
        return result.get('output', '')
    except Exception as e:
        logger.error(f"Failed to get clipboard: {e}")
        return ""

# Sensors

def get_sensor_list(self) -> List[str]:
    """Get list of available sensors"""
    try:
        result = self._execute_command(['sensor', '-l'])
        return result.get('sensors', [])
    except Exception as e:
        logger.error(f"Failed to get sensor list: {e}")
        return []

def read_sensor(self, sensor_type: str, delay: int = 1000, 
               limit: int = 1) -> List[SensorReading]:
    """Read sensor data"""
    try:
        cmd = ['sensor', '-s', sensor_type, '-d', str(delay), '-n', str(limit)]
        result = self._execute_command(cmd)
        
        readings = []
        for reading in result.get('readings', []):
            readings.append(SensorReading(
                sensor_type=sensor_type,
                values=reading.get('values', []),
                timestamp=reading.get('timestamp', time.time()),
                accuracy=reading.get('accuracy')
            ))
        
        return readings
        
    except Exception as e:
        logger.error(f"Failed to read sensor {sensor_type}: {e}")
        return []

# Network and Connectivity

def get_wifi_info(self) -> Dict[str, Any]:
    """Get WiFi connection information"""
    try:
        return self._execute_command(['wifi-connectioninfo'])
    except Exception as e:
        logger.error(f"Failed to get WiFi info: {e}")
        return {}

def scan_wifi(self) -> List[Dict[str, Any]]:
    """Scan for WiFi networks"""
    try:
        result = self._execute_command(['wifi-scaninfo'])
        return result.get('networks', [])
    except Exception as e:
        logger.error(f"Failed to scan WiFi: {e}")
        return []

# Telephony (if available)

def send_sms(self, phone_number: str, message: str) -> bool:
    """Send SMS message"""
    try:
        cmd = ['sms-send', '-n', phone_number, message]
        self._execute_command(cmd)
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        return False

def get_call_log(self, limit: int = 10) -> List[Dict[str, Any]]:
    """Get call log"""
    try:
        cmd = ['call-log', '-l', str(limit)]
        result = self._execute_command(cmd)
        return result.get('calls', [])
    except Exception as e:
        logger.error(f"Failed to get call log: {e}")
        return []

# Torch/Flashlight

def toggle_torch(self, enable: bool) -> bool:
    """Toggle device torch/flashlight"""
    try:
        state = "on" if enable else "off"
        self._execute_command(['torch', state])
        return True
    except Exception as e:
        logger.error(f"Failed to toggle torch: {e}")
        return False

# Utility Methods

def is_available(self) -> bool:
    """Check if Termux API is available"""
    return self.enabled

def get_capabilities(self) -> List[str]:
    """Get list of available API capabilities"""
    capabilities = []
    
    # Test various commands to see what's available
    test_commands = [
        'battery-status', 'notification', 'vibrate', 'clipboard-get',
        'camera-info', 'location', 'sensor', 'wifi-connectioninfo',
        'storage-get', 'microphone-record', 'tts-speak'
    ]
    
    for cmd in test_commands:
        try:
            # Try a safe version of the command
            if cmd == 'vibrate':
                self._execute_command([cmd, '1'])
            elif cmd == 'notification':
                continue  # Skip notification test
            else:
                self._execute_command([cmd])
            capabilities.append(cmd)
        except Exception:
            pass
    
    return capabilities

def cleanup(self):
    """Cleanup resources"""
    # Stop all monitoring threads
    for name in list(self.monitoring_threads.keys()):
        if name == 'battery':
            self.stop_battery_monitoring()
    
    logger.info("Termux API cleanup completed")Global instance_termux_api = Nonedef get_termux_api() -> TermuxAPI: """Get the global Termux API instance""" global _termux_api if _termux_api is None: _termux_api = TermuxAPI() return _termux_apiConvenience functionsdef send_notification(title: str, content: str = "", **kwargs) -> bool: """Send a notification""" return get_termux_api().send_notification(title, content, **kwargs)def get_battery_status() -> BatteryInfo: """Get battery status""" return get_termux_api().get_battery_status()def vibrate(duration: int = 1000) -> bool: """Vibrate the device""" return get_termux_api().vibrate(duration)def text_to_speech(text: str, **kwargs) -> bool: """Convert text to speech""" return get_termux_api().text_to_speech(text, **kwargs)def set_clipboard(text: str) -> bool: """Set clipboard content""" return get_termux_api().set_clipboard(text)def get_clipboard() -> str: """Get clipboard content""" return get_termux_api().get_clipboard()if name == "main": # Test Termux API functionality api = TermuxAPI()print("Testing Termux API")
print("=" * 40)

# Test availability
print(f"API Available: {api.is_available()}")

if api.is_available():
    # Test capabilities
    caps = api.get_capabilities()
    print(f"Capabilities: {caps}")
    
    # Test battery status
    try:
        battery = api.get_battery_status()
        print(f"Battery: {battery.percentage}%, {battery.status}")
    except Exception as e:
        print(f"Battery test failed: {e}")
    
    # Test device info
    try:
        info = api.get_device_info()
        print(f"Device info keys: {list(info.keys())}")
    except Exception as e:
        print(f"Device info test failed: {e}")
    
    # Test notification
    try:
        api.send_notification("Ultra AI Test", "Termux API is working!")
        print("Notification sent successfully")
    except Exception as e:
        print(f"Notification test failed: {e}")
    
    # Test clipboard
    try:
        test_text = "Ultra AI Termux API Test"
        api.set_clipboard(test_text)
        retrieved = api.get_clipboard()
        print(f"Clipboard test: {retrieved == test_text}")
    except Exception as e:
        print(f"Clipboard test failed: {e}")

print("Termux API test completed")
