"""
Utility functions for the transcription and classification system.
"""

import os
import re
import yaml
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TranscriptionResult:
    """Result of transcription and classification."""
    transcription: str
    classification: str
    confidence: float
    latency_ms: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("whisper-classifier")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class TextCleaner:
    """Clean and normalize transcribed text."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('text_cleanup', {})
        self.filler_words = self.config.get('filler_words', [])
        self.remove_fillers = self.config.get('remove_filler_words', True)
        self.normalize_case = self.config.get('normalize_case', True)
        self.strip_punctuation = self.config.get('strip_punctuation', False)
        
        # Build filler word regex pattern
        if self.filler_words:
            # Sort by length (longest first) to avoid partial matches
            sorted_fillers = sorted(self.filler_words, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(re.escape(w) for w in sorted_fillers) + r')\b'
            self.filler_pattern = re.compile(pattern, re.IGNORECASE)
        else:
            self.filler_pattern = None
    
    def clean(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove filler words
        if self.remove_fillers and self.filler_pattern:
            text = self.filler_pattern.sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Strip punctuation if configured
        if self.strip_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize case for classification (keep original for display)
        # Note: We return cleaned but preserve case for display
        
        return text
    
    def normalize_for_classification(self, text: str) -> str:
        """Normalize text specifically for classification matching."""
        text = self.clean(text)
        if self.normalize_case:
            text = text.lower()
        return text


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.classifications: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
    
    def start_timer(self) -> float:
        """Start timing an operation."""
        self.start_time = time.perf_counter()
        return self.start_time
    
    def stop_timer(self) -> float:
        """Stop timer and return elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.latencies.append(elapsed)
        self.start_time = None
        return elapsed
    
    def record_classification(self, result: TranscriptionResult):
        """Record a classification result."""
        self.classifications.append(result.to_dict())
    
    def get_average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.latencies:
            return {
                "total_processed": 0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "coding_count": 0,
                "non_coding_count": 0
            }
        
        coding_count = sum(
            1 for c in self.classifications 
            if c.get('classification') == 'CODING'
        )
        
        return {
            "total_processed": len(self.classifications),
            "avg_latency_ms": round(sum(self.latencies) / len(self.latencies), 2),
            "min_latency_ms": round(min(self.latencies), 2),
            "max_latency_ms": round(max(self.latencies), 2),
            "coding_count": coding_count,
            "non_coding_count": len(self.classifications) - coding_count
        }


class TranscriptionLogger:
    """Log transcriptions to JSONL file."""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, result: TranscriptionResult):
        """Append result to log file."""
        with open(self.log_file, 'a') as f:
            f.write(result.to_json() + '\n')
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all logged results."""
        if not self.log_file.exists():
            return []
        
        results = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results


def get_audio_devices() -> List[Dict[str, Any]]:
    """Get list of available audio input devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices
    except Exception as e:
        return [{'error': str(e)}]


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
