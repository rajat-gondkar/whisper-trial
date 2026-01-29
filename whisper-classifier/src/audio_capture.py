"""
Real-time audio capture module with Voice Activity Detection.
"""

import queue
import threading
import time
import numpy as np
from typing import Optional, Callable, Generator
from collections import deque

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("Please install sounddevice: pip install sounddevice")

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


class AudioCapture:
    """
    Real-time audio capture with buffering and voice activity detection.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 1.5,
        buffer_duration: float = 30,
        device: Optional[int] = None,
        vad_aggressiveness: int = 2,
        silence_threshold: float = 0.5,
        min_speech_duration: float = 0.3
    ):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sample rate (16000 Hz recommended for Whisper)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
            buffer_duration: Maximum buffer duration in seconds
            device: Audio input device index (None for default)
            vad_aggressiveness: VAD aggressiveness level (0-3)
            silence_threshold: Seconds of silence to end utterance
            min_speech_duration: Minimum speech duration to process
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.buffer_duration = buffer_duration
        self.device = device
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        
        # Calculate sizes
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.buffer_max_samples = int(sample_rate * buffer_duration)
        
        # Audio queue for processed chunks
        self.audio_queue: queue.Queue = queue.Queue()
        
        # Internal buffer for accumulating audio
        self.audio_buffer: deque = deque(maxlen=self.buffer_max_samples)
        
        # VAD setup
        self.vad = None
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # State
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        
        # Speech detection state
        self.is_speaking = False
        self.speech_buffer: list = []
        self.silence_samples = 0
        self.speech_samples = 0
        
        # VAD frame size (must be 10, 20, or 30 ms for webrtcvad)
        self.vad_frame_duration = 30  # ms
        self.vad_frame_samples = int(sample_rate * self.vad_frame_duration / 1000)
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono if needed
        audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        
        # Process with VAD if available
        if self.vad:
            self._process_with_vad(audio_data)
        else:
            self._process_simple(audio_data)
    
    def _process_with_vad(self, audio_data: np.ndarray):
        """Process audio with Voice Activity Detection."""
        # Convert to int16 for VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Process in VAD-compatible frames
        for i in range(0, len(audio_int16) - self.vad_frame_samples + 1, self.vad_frame_samples):
            frame = audio_int16[i:i + self.vad_frame_samples]
            
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            except Exception:
                is_speech = True  # Assume speech on error
            
            # Convert frame back to float32
            frame_float = frame.astype(np.float32) / 32767.0
            
            if is_speech:
                self.silence_samples = 0
                self.speech_samples += len(frame_float)
                self.speech_buffer.extend(frame_float)
                self.is_speaking = True
            else:
                self.silence_samples += len(frame_float)
                
                if self.is_speaking:
                    # Add some trailing silence
                    self.speech_buffer.extend(frame_float)
                    
                    # Check if silence threshold exceeded
                    silence_duration = self.silence_samples / self.sample_rate
                    speech_duration = self.speech_samples / self.sample_rate
                    
                    if silence_duration >= self.silence_threshold:
                        # End of utterance
                        if speech_duration >= self.min_speech_duration:
                            audio_chunk = np.array(self.speech_buffer, dtype=np.float32)
                            self.audio_queue.put(audio_chunk)
                        
                        # Reset state
                        self.speech_buffer = []
                        self.speech_samples = 0
                        self.is_speaking = False
    
    def _process_simple(self, audio_data: np.ndarray):
        """Simple processing without VAD - accumulate fixed-size chunks."""
        self.speech_buffer.extend(audio_data)
        
        # When we have enough samples, emit a chunk
        if len(self.speech_buffer) >= self.chunk_samples:
            audio_chunk = np.array(self.speech_buffer[:self.chunk_samples], dtype=np.float32)
            self.speech_buffer = self.speech_buffer[self.chunk_samples:]
            
            # Only emit if there's actual audio content (not silence)
            if np.abs(audio_chunk).max() > 0.01:
                self.audio_queue.put(audio_chunk)
    
    def start(self):
        """Start audio capture."""
        if self.is_recording:
            return
        
        with self._lock:
            self.is_recording = True
            self.speech_buffer = []
            self.silence_samples = 0
            self.speech_samples = 0
            self.is_speaking = False
            
            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=self.device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            )
            self.stream.start()
    
    def stop(self):
        """Stop audio capture."""
        with self._lock:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            # Process any remaining speech
            if self.speech_buffer and len(self.speech_buffer) >= self.sample_rate * self.min_speech_duration:
                audio_chunk = np.array(self.speech_buffer, dtype=np.float32)
                self.audio_queue.put(audio_chunk)
            
            self.speech_buffer = []
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Maximum time to wait for chunk
            
        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def audio_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio chunks continuously.
        
        Yields:
            Audio chunks as numpy arrays
        """
        while self.is_recording:
            chunk = self.get_audio_chunk(timeout=0.5)
            if chunk is not None:
                yield chunk
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class AudioCaptureSimple:
    """
    Simplified audio capture without VAD - uses fixed-duration chunks.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        device: Optional[int] = None,
        energy_threshold: float = 0.01
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device = device
        self.energy_threshold = energy_threshold
        
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self.buffer: list = []
        
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
    
    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self.buffer.extend(audio)
        
        while len(self.buffer) >= self.chunk_samples:
            chunk = np.array(self.buffer[:self.chunk_samples], dtype=np.float32)
            self.buffer = self.buffer[self.chunk_samples:]
            
            # Check if chunk has enough energy
            energy = np.sqrt(np.mean(chunk ** 2))
            if energy > self.energy_threshold:
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass  # Drop chunk if queue is full
    
    def start(self):
        if self.is_recording:
            return
        
        self.is_recording = True
        self.buffer = []
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.device,
            callback=self._callback,
            blocksize=int(self.sample_rate * 0.1)
        )
        self.stream.start()
    
    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def list_audio_devices():
    """List available audio input devices."""
    print("\n📱 Available Audio Input Devices:")
    print("-" * 50)
    
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']:.0f} Hz")
    print()


if __name__ == "__main__":
    # Test audio capture
    list_audio_devices()
    
    print("🎤 Testing audio capture for 5 seconds...")
    print("Speak something to test VAD detection.\n")
    
    capture = AudioCapture(
        sample_rate=16000,
        chunk_duration=1.5,
        vad_aggressiveness=2
    )
    
    capture.start()
    
    start_time = time.time()
    chunks_received = 0
    
    try:
        while time.time() - start_time < 5:
            chunk = capture.get_audio_chunk(timeout=0.5)
            if chunk is not None:
                chunks_received += 1
                duration = len(chunk) / 16000
                energy = np.sqrt(np.mean(chunk ** 2))
                print(f"  Chunk {chunks_received}: {duration:.2f}s, energy: {energy:.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
    
    print(f"\n✅ Test complete. Received {chunks_received} audio chunks.")
