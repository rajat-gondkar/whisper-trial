"""
Whisper-based speech-to-text transcription module.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False


@dataclass
class TranscriptionOutput:
    """Output from transcription."""
    text: str
    language: str
    language_probability: float
    duration: float
    processing_time_ms: float


class WhisperTranscriber:
    """
    Whisper-based transcription using faster-whisper for low latency.
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        language: Optional[str] = "en",
        beam_size: int = 5,
        best_of: int = 1,
        temperature: float = 0.0,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        condition_on_previous_text: bool = True,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute type for inference (int8, float16, float32)
            language: Language code or None for auto-detect
            beam_size: Beam search size
            best_of: Number of candidates
            temperature: Sampling temperature (can be list for fallback)
            vad_filter: Use VAD filtering
            word_timestamps: Enable word-level timestamps
            condition_on_previous_text: Use context from previous text
            compression_ratio_threshold: Threshold for compression ratio
            log_prob_threshold: Threshold for log probability
            no_speech_threshold: Threshold for no speech detection
        """
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.condition_on_previous_text = condition_on_previous_text
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold
        
        # Determine device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        self.device = device
        self.compute_type = compute_type
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        if FASTER_WHISPER_AVAILABLE:
            print(f"🔄 Loading faster-whisper model: {self.model_size} on {self.device}...")
            start = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            load_time = time.time() - start
            print(f"✅ Model loaded in {load_time:.2f}s")
            self.backend = "faster-whisper"
            
        elif OPENAI_WHISPER_AVAILABLE:
            print(f"🔄 Loading openai-whisper model: {self.model_size}...")
            start = time.time()
            
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            load_time = time.time() - start
            print(f"✅ Model loaded in {load_time:.2f}s")
            self.backend = "openai-whisper"
            
        else:
            raise ImportError(
                "No Whisper backend available. Please install:\n"
                "  pip install faster-whisper\n"
                "or\n"
                "  pip install openai-whisper"
            )
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionOutput:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio (should be 16000)
            
        Returns:
            TranscriptionOutput with text and metadata
        """
        start_time = time.perf_counter()
        
        # Ensure correct dtype
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        duration = len(audio) / sample_rate
        
        if self.backend == "faster-whisper":
            result = self._transcribe_faster_whisper(audio)
        else:
            result = self._transcribe_openai_whisper(audio, sample_rate)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return TranscriptionOutput(
            text=result['text'],
            language=result.get('language', self.language or 'en'),
            language_probability=result.get('language_probability', 1.0),
            duration=duration,
            processing_time_ms=processing_time
        )
    
    def _transcribe_faster_whisper(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe using faster-whisper with enhanced accuracy."""
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            word_timestamps=self.word_timestamps,
            condition_on_previous_text=self.condition_on_previous_text,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            initial_prompt="This is a technical conversation about programming and coding."
        )
        
        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        full_text = " ".join(text_parts)
        
        return {
            'text': full_text,
            'language': info.language,
            'language_probability': info.language_probability
        }
    
    def _transcribe_openai_whisper(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Transcribe using openai-whisper."""
        result = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            fp16=(self.device == "cuda")
        )
        
        return {
            'text': result['text'].strip(),
            'language': result.get('language', self.language or 'en'),
            'language_probability': 1.0
        }


class TranscriptionManager:
    """
    Manager for handling transcription with text cleanup.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize transcription manager with config."""
        trans_config = config.get('transcription', {})
        
        self.transcriber = WhisperTranscriber(
            model_size=trans_config.get('model_size', 'small'),
            device=trans_config.get('device', 'auto'),
            compute_type=trans_config.get('compute_type', 'int8'),
            language=trans_config.get('language', 'en'),
            beam_size=trans_config.get('beam_size', 5),
            best_of=trans_config.get('best_of', 1),
            temperature=trans_config.get('temperature', 0.0),
            vad_filter=trans_config.get('vad_filter', True),
            word_timestamps=trans_config.get('word_timestamps', False),
            condition_on_previous_text=trans_config.get('condition_on_previous_text', True),
            compression_ratio_threshold=trans_config.get('compression_ratio_threshold', 2.4),
            log_prob_threshold=trans_config.get('log_prob_threshold', -1.0),
            no_speech_threshold=trans_config.get('no_speech_threshold', 0.6)
        )
        
        self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)
    
    def transcribe(self, audio: np.ndarray) -> TranscriptionOutput:
        """Transcribe audio chunk."""
        return self.transcriber.transcribe(audio, self.sample_rate)


if __name__ == "__main__":
    # Test transcription
    print("🧪 Testing Whisper Transcription")
    print("-" * 40)
    
    # Create a simple test with synthesized audio
    import numpy as np
    
    # Create silence (for testing model loading)
    sample_rate = 16000
    duration = 2.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Add some noise to simulate audio
    audio += np.random.randn(len(audio)).astype(np.float32) * 0.01
    
    print("\nInitializing transcriber...")
    transcriber = WhisperTranscriber(
        model_size="tiny",  # Use tiny for quick testing
        device="cpu",
        compute_type="int8"
    )
    
    print("\nTranscribing test audio (silence with noise)...")
    result = transcriber.transcribe(audio, sample_rate)
    
    print(f"\n📝 Result:")
    print(f"   Text: '{result.text}'")
    print(f"   Language: {result.language} ({result.language_probability:.2%})")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Processing time: {result.processing_time_ms:.0f}ms")
    print(f"   Real-time factor: {result.processing_time_ms / (result.duration * 1000):.2f}x")
    
    print("\n✅ Transcription test complete!")
