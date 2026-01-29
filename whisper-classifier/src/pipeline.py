"""
Real-time processing pipeline integrating audio capture, transcription, and classification.
"""

import time
import threading
import queue
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

import numpy as np

from .audio_capture import AudioCapture, AudioCaptureSimple
from .transcription import TranscriptionManager, TranscriptionOutput
from .classifier import CodingClassifier, ClassificationResult
from .utils import (
    TranscriptionResult,
    TextCleaner,
    PerformanceMonitor,
    TranscriptionLogger
)


class RealTimePipeline:
    """
    Real-time pipeline for audio → transcription → classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        audio_config = config.get('audio', {})
        
        self.audio_capture = AudioCapture(
            sample_rate=audio_config.get('sample_rate', 16000),
            channels=audio_config.get('channels', 1),
            chunk_duration=audio_config.get('chunk_duration', 1.5),
            buffer_duration=audio_config.get('buffer_duration', 30),
            device=audio_config.get('device'),
            vad_aggressiveness=audio_config.get('vad_aggressiveness', 2),
            silence_threshold=audio_config.get('silence_threshold', 0.5),
            min_speech_duration=audio_config.get('min_speech_duration', 0.3)
        )
        
        self.transcription_manager = TranscriptionManager(config)
        self.classifier = CodingClassifier(config)
        self.text_cleaner = TextCleaner(config)
        self.performance_monitor = PerformanceMonitor()
        
        # Optional logging
        pipeline_config = config.get('pipeline', {})
        if pipeline_config.get('enable_logging', False):
            log_file = pipeline_config.get('log_file', 'transcription_log.jsonl')
            self.logger = TranscriptionLogger(log_file)
        else:
            self.logger = None
        
        # Pipeline state
        self.is_running = False
        self.result_callback: Optional[Callable[[TranscriptionResult], None]] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.sample_rate = audio_config.get('sample_rate', 16000)
    
    def set_result_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Set callback for processing results."""
        self.result_callback = callback
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        """
        Process a single audio chunk through the pipeline.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            TranscriptionResult or None if processing failed
        """
        # Start timing
        self.performance_monitor.start_timer()
        
        try:
            # Step 1: Transcribe
            transcription = self.transcription_manager.transcribe(audio_chunk)
            
            # Skip empty transcriptions
            if not transcription.text or not transcription.text.strip():
                return None
            
            # Step 2: Clean text
            cleaned_text = self.text_cleaner.clean(transcription.text)
            normalized_text = self.text_cleaner.normalize_for_classification(cleaned_text)
            
            # Skip if still empty after cleaning
            if not normalized_text.strip():
                return None
            
            # Step 3: Classify
            classification = self.classifier.classify(normalized_text)
            
            # Stop timing
            total_latency = self.performance_monitor.stop_timer()
            
            # Create result
            result = TranscriptionResult(
                transcription=cleaned_text,
                classification=classification.classification,
                confidence=classification.confidence,
                latency_ms=total_latency
            )
            
            # Record for monitoring
            self.performance_monitor.record_classification(result)
            
            # Log if enabled
            if self.logger:
                self.logger.log(result)
            
            return result
            
        except Exception as e:
            self.performance_monitor.stop_timer()
            print(f"❌ Processing error: {e}")
            return None
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        while self.is_running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                
                if audio_chunk is None:
                    continue
                
                # Process chunk
                result = self._process_audio_chunk(audio_chunk)
                
                if result and self.result_callback:
                    self.result_callback(result)
                    
            except Exception as e:
                if self.is_running:
                    print(f"❌ Pipeline error: {e}")
    
    def start(self):
        """Start the pipeline."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start audio capture
        self.audio_capture.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    
    def stop(self):
        """Stop the pipeline."""
        self.is_running = False
        
        # Stop audio capture
        self.audio_capture.stop()
        
        # Wait for processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        return self.performance_monitor.get_stats()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class SimplePipeline:
    """
    Simplified pipeline for synchronous processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified pipeline."""
        self.config = config
        
        self.transcription_manager = TranscriptionManager(config)
        self.classifier = CodingClassifier(config)
        self.text_cleaner = TextCleaner(config)
        self.performance_monitor = PerformanceMonitor()
    
    def process(self, audio: np.ndarray) -> Optional[TranscriptionResult]:
        """
        Process audio through the pipeline.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            TranscriptionResult or None
        """
        self.performance_monitor.start_timer()
        
        try:
            # Transcribe
            transcription = self.transcription_manager.transcribe(audio)
            
            if not transcription.text.strip():
                return None
            
            # Clean
            cleaned_text = self.text_cleaner.clean(transcription.text)
            
            if not cleaned_text.strip():
                return None
            
            # Classify
            classification = self.classifier.classify(
                self.text_cleaner.normalize_for_classification(cleaned_text)
            )
            
            # Create result
            latency = self.performance_monitor.stop_timer()
            
            result = TranscriptionResult(
                transcription=cleaned_text,
                classification=classification.classification,
                confidence=classification.confidence,
                latency_ms=latency
            )
            
            self.performance_monitor.record_classification(result)
            
            return result
            
        except Exception as e:
            self.performance_monitor.stop_timer()
            raise


if __name__ == "__main__":
    # Test pipeline
    from .utils import load_config
    
    print("🧪 Testing Pipeline")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        config = {
            'audio': {'sample_rate': 16000},
            'transcription': {'model_size': 'tiny', 'device': 'cpu'},
            'classification': {},
            'text_cleanup': {}
        }
    
    # Test simple pipeline with synthetic audio
    print("\nTesting SimplePipeline with synthetic audio...")
    
    pipeline = SimplePipeline(config)
    
    # Create test audio (silence)
    sample_rate = 16000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    audio += np.random.randn(len(audio)).astype(np.float32) * 0.01
    
    result = pipeline.process(audio)
    
    if result:
        print(f"\n📝 Result:")
        print(f"   Transcription: {result.transcription}")
        print(f"   Classification: {result.classification}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Latency: {result.latency_ms:.0f}ms")
    else:
        print("\n⚠️ No transcription result (audio may be silent)")
    
    print("\n✅ Pipeline test complete!")
