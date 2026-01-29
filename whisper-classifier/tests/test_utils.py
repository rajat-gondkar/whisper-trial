"""
Tests for utility functions.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import clean_text, TranscriptionResult, Timer


class TestCleanText:
    """Tests for text cleaning function."""
    
    def test_removes_filler_words(self):
        """Test filler word removal."""
        text = "umm so like I want to umm write code"
        result = clean_text(text)
        assert "umm" not in result
        assert "like" not in result
    
    def test_preserves_content(self):
        """Test that meaningful content is preserved."""
        text = "Write Python code for sorting"
        result = clean_text(text, lowercase=False)
        assert "Write" in result or "write" in result.lower()
        assert "Python" in result or "python" in result.lower()
        assert "sorting" in result.lower()
    
    def test_removes_extra_whitespace(self):
        """Test whitespace normalization."""
        text = "Write   a    function"
        result = clean_text(text)
        assert "  " not in result
    
    def test_handles_empty_string(self):
        """Test empty string handling."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
    
    def test_lowercase_option(self):
        """Test lowercase conversion option."""
        text = "Write Python Code"
        
        lower_result = clean_text(text, lowercase=True)
        assert lower_result == lower_result.lower()
        
        # When lowercase=False, we still clean but preserve some case
        preserved_result = clean_text(text, lowercase=False)
        assert "Write" in preserved_result or "python" in preserved_result.lower()


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TranscriptionResult(
            transcription="write python code",
            classification="CODING",
            confidence=0.95,
            latency_ms=150.5
        )
        
        d = result.to_dict()
        
        assert d["transcription"] == "write python code"
        assert d["classification"] == "CODING"
        assert d["confidence"] == 0.95
        assert d["latency_ms"] == 150.5
    
    def test_to_json_string(self):
        """Test JSON string output."""
        result = TranscriptionResult(
            transcription="test",
            classification="NON-CODING",
            confidence=0.8,
            latency_ms=100
        )
        
        json_str = result.to_json_string()
        
        assert "transcription" in json_str
        assert "test" in json_str
        assert "NON-CODING" in json_str


class TestTimer:
    """Tests for Timer context manager."""
    
    def test_timer_measures_time(self):
        """Test that timer measures elapsed time."""
        import time
        
        with Timer() as t:
            time.sleep(0.1)
        
        # Should be approximately 100ms (with some tolerance)
        assert 80 <= t.elapsed_ms <= 200
    
    def test_timer_attributes(self):
        """Test timer has required attributes."""
        with Timer() as t:
            pass
        
        assert hasattr(t, 'start_time')
        assert hasattr(t, 'end_time')
        assert hasattr(t, 'elapsed_ms')
        assert t.elapsed_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
