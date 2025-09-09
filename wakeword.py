    return FallbackConfig()class WakeWordEngine(Enum): """Available wake word detection engines""" PORCUPINE = "porcupine" MFCC = "mfcc" DISABLED = "disabled"class DetectionMethod(Enum): """Detection method types""" TRUE_DETECTION = "true_detection" LISTEN_ONLY = "listen_only"  # Explicitly not true detection@dataclass class WakeWordConfig: """Wake word detection configuration""" enabled: bool = False engine: WakeWordEngine = WakeWordEngine.MFCC sensitivity: float = 0.5 wake_word: str = "ultra" sample_rate: int = 16000 frame_length: int = 512 detection_threshold: float = 0.7 listen_only_mode: bool = False  # Fallback mode, not true detection audio_device: Optional[str] = None@dataclass class DetectionResult: """Wake word detection result""" detected: bool confidence: float engine: WakeWordEngine method: DetectionMethod timestamp: float audio_data: Optional[np.ndarray] = Noneclass AudioCapture: """Audio capture using various backends"""def __init__(self, sample_rate: int = 16000, frame_length: int = 512):
    self.sample_rate = sample_rate
    self.frame_length = frame_length
    self.audio_queue = queue.Queue()
    self.recording = False
    self.audio_thread = None
    
    # Try to initialize audio backend
    self.backend = self._initialize_backend()

def _initialize_backend(self) -> str:
    """Initialize the best available audio backend"""
    
    # Try sounddevice first (if available)
    try:
        import sounddevice as sd
        # Test if sounddevice works
        sd.check_input_settings(
            device=None, 
            channels=1, 
            dtype='float32', 
            samplerate=self.sample_rate
        )
        logger.info("Using sounddevice for audio capture")
        return "sounddevice"
    except Exception as e:
        logger.debug(f"Sounddevice not available: {e}")
    
    # Try Termux microphone recording
    try:
        from termux_api import get_termux_api
        api = get_termux_api()
        if api.is_available():
            logger.info("Using Termux API for audio capture")
            return "termux"
    except Exception as e:
        logger.debug(f"Termux API not available: {e}")
    
    # Fallback to silence (no real audio)
    logger.warning("No audio backend available - using silence generator")
    return "silence"

def start_capture(self):
    """Start audio capture"""
    if self.recording:
        return
    
    self.recording = True
    
    if self.backend == "sounddevice":
        self.audio_thread = threading.Thread(target=self._sounddevice_capture, daemon=True)
    elif self.backend == "termux":
        self.audio_thread = threading.Thread(target=self._termux_capture, daemon=True)
    else:
        self.audio_thread = threading.Thread(target=self._silence_capture, daemon=True)
    
    self.audio_thread.start()
    logger.info("Audio capture started")

def stop_capture(self):
    """Stop audio capture"""
    self.recording = False
    if self.audio_thread:
        self.audio_thread.join(timeout=5.0)
    logger.info("Audio capture stopped")

def get_audio_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
    """Get next audio frame"""
    try:
        return self.audio_queue.get(timeout=timeout)
    except queue.Empty:
        return None

def _sounddevice_capture(self):
    """Capture audio using sounddevice"""
    try:
        import sounddevice as sd
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert to mono and add to queue
            audio_data = indata[:, 0] if indata.ndim > 1 else indata
            try:
                self.audio_queue.put_nowait(audio_data.copy())
            except queue.Full:
                pass  # Drop frame if queue is full
        
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_length,
            dtype='float32'
        ):
            while self.recording:
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Sounddevice capture error: {e}")

def _termux_capture(self):
    """Capture audio using Termux API"""
    try:
        from termux_api import get_termux_api
        api = get_termux_api()
        
        while self.recording:
            # Record short segments
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Record 1 second of audio
                success = api.record_audio(temp_path, duration=1, encoder="wav")
                
                if success and os.path.exists(temp_path):
                    # Read the recorded audio
                    audio_data = self._read_wav_file(temp_path)
                    if audio_data is not None:
                        # Split into frames
                        frames = len(audio_data) // self.frame_length
                        for i in range(frames):
                            start = i * self.frame_length
                            end = start + self.frame_length
                            frame = audio_data[start:end]
                            try:
                                self.audio_queue.put_nowait(frame)
                            except queue.Full:
                                pass
                
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                
    except Exception as e:
        logger.error(f"Termux capture error: {e}")

def _silence_capture(self):
    """Generate silence frames (fallback)"""
    logger.warning("Audio capture using silence generator - no real wake word detection")
    
    while self.recording:
        # Generate silence frame
        silence_frame = np.zeros(self.frame_length, dtype=np.float32)
        try:
            self.audio_queue.put_nowait(silence_frame)
        except queue.Full:
            pass
        
        # Wait appropriate time for frame rate
        time.sleep(self.frame_length / self.sample_rate)

def _read_wav_file(self, file_path: str) -> Optional[np.ndarray]:
    """Read WAV file and return audio data"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            
            # Convert to numpy array
            if sample_width == 1:
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = (audio_data.astype(np.float32) - 128) / 128
            elif sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768
            else:
                logger.error(f"Unsupported sample width: {sample_width}")
                return None
            
            return audio_data
            
    except Exception as e:
        logger.error(f"Error reading WAV file: {e}")
        return Noneclass PorcupineDetector: """Porcupine wake word detection engine"""def __init__(self, wake_word: str = "ultra", sensitivity: float = 0.5):
    self.wake_word = wake_word
    self.sensitivity = sensitivity
    self.porcupine = None
    self.available = False
    
    self._initialize_porcupine()

def _initialize_porcupine(self):
    """Initialize Porcupine engine"""
    try:
        import pvporcupine
        
        # Try to create Porcupine instance
        # Note: This requires Porcupine to be properly installed and licensed
        self.porcupine = pvporcupine.create(
            keywords=[self.wake_word],
            sensitivities=[self.sensitivity]
        )
        self.available = True
        logger.info("Porcupine wake word engine initialized")
        
    except ImportError:
        logger.info("Porcupine not available - install with: pip install pvporcupine")
    except Exception as e:
        logger.warning(f"Failed to initialize Porcupine: {e}")

def detect(self, audio_frame: np.ndarray) -> DetectionResult:
    """Detect wake word in audio frame"""
    if not self.available or self.porcupine is None:
        return DetectionResult(
            detected=False,
            confidence=0.0,
            engine=WakeWordEngine.PORCUPINE,
            method=DetectionMethod.TRUE_DETECTION,
            timestamp=time.time()
        )
    
    try:
        # Convert audio frame to required format
        if audio_frame.dtype != np.int16:
            audio_frame = (audio_frame * 32767).astype(np.int16)
        
        # Process audio frame
        keyword_index = self.porcupine.process(audio_frame)
        
        detected = keyword_index >= 0
        confidence = 1.0 if detected else 0.0
        
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            engine=WakeWordEngine.PORCUPINE,
            method=DetectionMethod.TRUE_DETECTION,
            timestamp=time.time(),
            audio_data=audio_frame
        )
        
    except Exception as e:
        logger.error(f"Porcupine detection error: {e}")
        return DetectionResult(
            detected=False,
            confidence=0.0,
            engine=WakeWordEngine.PORCUPINE,
            method=DetectionMethod.TRUE_DETECTION,
            timestamp=time.time()
        )

def cleanup(self):
    """Cleanup Porcupine resources"""
    if self.porcupine:
        try:
            self.porcupine.delete()
        except Exception as e:
            logger.error(f"Error cleaning up Porcupine: {e}")class MFCCDetector: """MFCC-based wake word detection (fallback)"""def __init__(self, wake_word: str = "ultra", threshold: float = 0.7):
    self.wake_word = wake_word
    self.threshold = threshold
    self.reference_mfcc = None
    self.frame_buffer = []
    self.buffer_size = 32  # Number of frames to buffer
    
    # Initialize reference pattern (placeholder)
    self._initialize_reference()

def _initialize_reference(self):
    """Initialize reference MFCC pattern"""
    # This is a placeholder - in a real implementation, this would be
    # trained from actual recordings of the wake word
    logger.warning("MFCC detector using placeholder reference - not true wake word detection")
    
    # Create a simple reference pattern
    self.reference_mfcc = np.random.random((13, 10))  # 13 MFCC coefficients, 10 frames

def _extract_mfcc(self, audio_frame: np.ndarray) -> np.ndarray:
    """Extract MFCC features from audio frame"""
    try:
        # Simple MFCC extraction (placeholder implementation)
        # In practice, this would use a proper MFCC implementation
        
        # Apply window
        windowed = audio_frame * np.hanning(len(audio_frame))
        
        # FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        # Mel filter bank (simplified)
        num_filters = 13
        mel_filters = np.random.random((num_filters, len(magnitude)))
        mel_energies = np.dot(mel_filters, magnitude)
        
        # Log and DCT (simplified)
        log_energies = np.log(mel_energies + 1e-10)
        mfcc = np.fft.dct(log_energies)
        
        return mfcc
        
    except Exception as e:
        logger.error(f"MFCC extraction error: {e}")
        return np.zeros(13)

def detect(self, audio_frame: np.ndarray) -> DetectionResult:
    """Detect wake word using MFCC features"""
    try:
        # Extract MFCC features
        mfcc = self._extract_mfcc(audio_frame)
        
        # Add to buffer
        self.frame_buffer.append(mfcc)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Simple pattern matching (placeholder)
        if len(self.frame_buffer) >= 10:  # Minimum frames for detection
            # This is a very basic similarity check
            # Real implementation would use DTW or other alignment methods
            similarity = np.random.random()  # Placeholder
            
            detected = similarity > self.threshold
            confidence = similarity
            
            return DetectionResult(
                detected=detected,
                confidence=confidence,
                engine=WakeWordEngine.MFCC,
                method=DetectionMethod.LISTEN_ONLY,  # Not true detection
                timestamp=time.time(),
                audio_data=audio_frame
            )
        
        return DetectionResult(
            detected=False,
            confidence=0.0,
            engine=WakeWordEngine.MFCC,
            method=DetectionMethod.LISTEN_ONLY,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"MFCC detection error: {e}")
        return DetectionResult(
            detected=False,
            confidence=0.0,
            engine=WakeWordEngine.MFCC,
            method=DetectionMethod.LISTEN_ONLY,
            timestamp=time.time()
        )class WakeWordDetector: """Main wake word detection system"""def __init__(self, config: Optional[WakeWordConfig] = None):
    self.config = config or WakeWordConfig()
    self.audio_capture = None
    self.detector = None
    self.running = False
    self.detection_thread = None
    self.callbacks = []
    
    # Initialize based on configuration
    if self.config.enabled:
        self._initialize_detector()
    
    logger.info(f"Wake word detector initialized (enabled: {self.config.enabled})")

def _initialize_detector(self):
    """Initialize the appropriate detection engine"""
    if self.config.engine == WakeWordEngine.PORCUPINE:
        self.detector = PorcupineDetector(
            wake_word=self.config.wake_word,
            sensitivity=self.config.sensitivity
        )
        
        # Fallback to MFCC if Porcupine not available
        if not self.detector.available:
            logger.info("Porcupine not available, falling back to MFCC")
            self.config.engine = WakeWordEngine.MFCC
            self.detector = MFCCDetector(
                wake_word=self.config.wake_word,
                threshold=self.config.detection_threshold
            )
    
    elif self.config.engine == WakeWordEngine.MFCC:
        self.detector = MFCCDetector(
            wake_word=self.config.wake_word,
            threshold=self.config.detection_threshold
        )
    
    # Initialize audio capture
    self.audio_capture = AudioCapture(
        sample_rate=self.config.sample_rate,
        frame_length=self.config.frame_length
    )

def add_callback(self, callback: Callable[[DetectionResult], None]):
    """Add detection callback"""
    self.callbacks.append(callback)

def remove_callback(self, callback: Callable[[DetectionResult], None]):
    """Remove detection callback"""
    if callback in self.callbacks:
        self.callbacks.remove(callback)

def start(self) -> bool:
    """Start wake word detection"""
    if not self.config.enabled:
        logger.info("Wake word detection is disabled")
        return False
    
    if self.running:
        logger.warning("Wake word detection already running")
        return True
    
    if not self.detector or not self.audio_capture:
        logger.error("Detector or audio capture not initialized")
        return False
    
    try:
        # Start audio capture
        self.audio_capture.start_capture()
        
        # Start detection thread
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        method_type = "LISTEN-ONLY" if self.config.engine == WakeWordEngine.MFCC else "TRUE DETECTION"
        logger.info(f"Wake word detection started ({self.config.engine.value}, {method_type})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start wake word detection: {e}")
        self.stop()
        return False

def stop(self):
    """Stop wake word detection"""
    if not self.running:
        return
    
    self.running = False
    
    # Stop audio capture
    if self.audio_capture:
        self.audio_capture.stop_capture()
    
    # Wait for detection thread
    if self.detection_thread:
        self.detection_thread.join(timeout=5.0)
    
    logger.info("Wake word detection stopped")

def _detection_loop(self):
    """Main detection loop"""
    logger.info("Wake word detection loop started")
    
    while self.running:
        try:
            # Get audio frame
            audio_frame = self.audio_capture.get_audio_frame(timeout=1.0)
            if audio_frame is None:
                continue
            
            # Run detection
            result = self.detector.detect(audio_frame)
            
            # Call callbacks if detection occurred
            if result.detected:
                logger.info(f"Wake word detected! (confidence: {result.confidence:.2f}, "
                          f"engine: {result.engine.value}, method: {result.method.value})")
                
                for callback in self.callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Detection callback error: {e}")
            
        except Exception as e:
            logger.error(f"Detection loop error: {e}")
            time.sleep(0.1)
    
    logger.info("Wake word detection loop ended")

def is_running(self) -> bool:
    """Check if detection is running"""
    return self.running

def get_config(self) -> WakeWordConfig:
    """Get current configuration"""
    return self.config

def update_config(self, new_config: WakeWordConfig):
    """Update configuration"""
    was_running = self.running
    
    if was_running:
        self.stop()
    
    self.config = new_config
    
    if self.config.enabled:
        self._initialize_detector()
        
        if was_running:
            self.start()

def cleanup(self):
    """Cleanup resources"""
    self.stop()
    
    if hasattr(self.detector, 'cleanup'):
        self.detector.cleanup()Global detector instance_wake_word_detector = Nonedef get_wake_word_detector() -> WakeWordDetector: """Get the global wake word detector instance""" global _wake_word_detector if _wake_word_detector is None: config = get_config() wake_config = WakeWordConfig( enabled=getattr(config.system, 'wake_word_enabled', False) ) _wake_word_detector = WakeWordDetector(wake_config) return _wake_word_detectordef enable_wake_word(enable: bool = True): """Enable or disable wake word detection""" detector = get_wake_word_detector() current_config = detector.get_config() current_config.enabled = enable detector.update_config(current_config)def set_wake_word_engine(engine: WakeWordEngine): """Set the wake word detection engine""" detector = get_wake_word_detector() current_config = detector.get_config() current_config.engine = engine detector.update_config(current_config)def add_wake_word_callback(callback: Callable[[DetectionResult], None]): """Add wake word detection callback""" get_wake_word_detector().add_callback(callback)if name == "main": # Test wake word detection print("Testing Ultra AI Wake Word Detection") print("=" * 50)# Create test configuration
test_config = WakeWordConfig(
    enabled=True,
    engine=WakeWordEngine.MFCC,  # Use MFCC for testing
    wake_word="ultra",
    sensitivity=0.5
)

# Create detector
detector = WakeWordDetector(test_config)

# Add test callback
def on_detection(result: DetectionResult):
    print(f"Wake word detected! Engine: {result.engine.value}, "
          f"Method: {result.method.value}, Confidence: {result.confidence:.2f}")
    if result.method == DetectionMethod.LISTEN_ONLY:
        print("NOTE: This is LISTEN-ONLY mode, not true wake word detection")

detector.add_callback(on_detection)

# Test detection
try:
    print(f"Starting wake word detection for '{test_config.wake_word}'...")
    print("Press Ctrl+C to stop")
    
    if detector.start():
        print("Detection started successfully")
        
        # Let it run for a while
        while True:
            time.sleep(1)
    else:
        print("Failed to start detection")

except KeyboardInterrupt:
    print("\nStopping wake word detection...")

finally:
    detector.cleanup()
    print("Wake word detection test completed")
