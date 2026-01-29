# Whisper AI Implementation Guide

## Overview

This document provides a complete technical reference for the Whisper AI transcription system used in this project. It covers audio capture, voice activity detection, sentence segmentation, and all Whisper configurations.

---

## Architecture

The Whisper implementation consists of three main components:

1. **Audio Capture** (`audio_capture.py`) - Real-time microphone input with Voice Activity Detection
2. **Transcription** (`transcription.py`) - Whisper model integration and text generation
3. **Text Cleanup** (`utils.py`) - Post-processing and normalization

---

## 1. Audio Capture System

### 1.1 Core Configuration

Located in `config/config.yaml`:

```yaml
audio:
  sample_rate: 16000          # 16kHz - Whisper's native sample rate
  channels: 1                  # Mono audio (required by Whisper)
  chunk_duration: 1.5          # Seconds per audio chunk
  buffer_duration: 30          # Maximum buffer duration
  device: null                 # null = default mic, or specify device index
  vad_aggressiveness: 2        # Voice Activity Detection: 0-3
  silence_threshold: 0.5       # Seconds of silence to end utterance
  min_speech_duration: 0.3     # Minimum speech duration to process
```

### 1.2 How Sentence Detection Works

The system uses **Voice Activity Detection (VAD)** to automatically detect when speech starts and stops:

#### VAD Frame Processing
```python
# WebRTC VAD processes audio in fixed frames (30ms chunks)
vad_frame_duration = 30  # milliseconds
vad_frame_samples = int(sample_rate * vad_frame_duration / 1000)  # 480 samples at 16kHz
```

#### Speech Detection Logic

1. **Audio Conversion**
   - Input audio (float32, -1.0 to 1.0) → Convert to int16 for VAD
   - `audio_int16 = (audio_data * 32767).astype(np.int16)`

2. **Frame-by-Frame Analysis**
   ```python
   for each 30ms frame:
       is_speech = vad.is_speech(frame.tobytes(), sample_rate)
       
       if is_speech:
           - Reset silence counter
           - Accumulate frame to speech_buffer
           - Mark as speaking
       else:
           - Increment silence counter
           - Continue adding frames to buffer (trailing silence)
   ```

3. **Utterance Completion Detection**
   ```python
   silence_duration = silence_samples / sample_rate
   speech_duration = speech_samples / sample_rate
   
   if silence_duration >= silence_threshold (0.5s):
       if speech_duration >= min_speech_duration (0.3s):
           # Complete utterance detected
           send_to_transcription(speech_buffer)
       
       # Reset for next utterance
       clear_buffers()
   ```

### 1.3 When Sentences Stop

A sentence/utterance is considered complete when:

1. **Silence Detection**: ≥0.5 seconds of continuous silence detected by VAD
2. **Minimum Duration**: Total speech duration ≥0.3 seconds
3. **Audio Quality**: Frame has detectable energy (not pure silence)

**Key Parameters:**
- `silence_threshold: 0.5` - How long to wait after speech stops (500ms)
- `min_speech_duration: 0.3` - Filter out very short sounds (300ms)
- `vad_aggressiveness: 2` - Balance between sensitivity (0) and strictness (3)

### 1.4 Audio Streaming Flow

```
Microphone Input (continuous)
    ↓
100ms blocks (sounddevice)
    ↓
VAD Analysis (30ms frames)
    ↓
Speech Buffer (accumulation)
    ↓
Silence Detection (0.5s threshold)
    ↓
Complete Utterance → Queue for Transcription
```

### 1.5 Implementation Details

**AudioCapture Class (`audio_capture.py`)**

```python
class AudioCapture:
    def __init__(
        sample_rate=16000,           # Whisper requirement
        channels=1,                  # Mono
        chunk_duration=1.5,          # Not used with VAD
        vad_aggressiveness=2,        # 0-3 scale
        silence_threshold=0.5,       # 500ms of silence
        min_speech_duration=0.3      # 300ms minimum
    )
```

**Key Methods:**
- `_process_with_vad()` - Main VAD processing logic
- `_audio_callback()` - Receives audio from sounddevice
- `get_audio_chunk()` - Returns complete utterances for transcription

---

## 2. Whisper Transcription System

### 2.1 Model Configuration

Located in `config/config.yaml`:

```yaml
transcription:
  model_size: "medium"         # tiny, base, small, medium, large-v2, large-v3
  device: "auto"               # auto, cpu, cuda
  compute_type: "int8"         # int8, float16, float32
  language: "en"               # ISO language code or null for auto-detect
  beam_size: 5                 # Beam search width (1-10)
  best_of: 5                   # Number of candidates (1-10)
  temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Fallback temperatures
  vad_filter: true             # Whisper's internal VAD
  word_timestamps: false       # Disable for lower latency
  condition_on_previous_text: true  # Use context from previous text
  compression_ratio_threshold: 2.4  # Skip low-quality audio
  log_prob_threshold: -1.0     # Skip low-confidence segments
  no_speech_threshold: 0.6     # Skip if no speech detected
```

### 2.2 Model Selection Guide

| Model Size | RAM Usage | Speed (RTF) | Accuracy | Use Case |
|------------|-----------|-------------|----------|----------|
| tiny       | ~1 GB     | ~0.2x       | 70%      | Testing only |
| base       | ~1 GB     | ~0.3x       | 75%      | Low-end devices |
| small      | ~2 GB     | ~0.5x       | 82%      | Fast, good quality |
| **medium** | ~5 GB     | ~1.0x       | **86%**  | **Recommended** |
| large-v2   | ~10 GB    | ~2.0x       | 88%      | Best accuracy |
| large-v3   | ~10 GB    | ~2.0x       | 89%      | Latest version |

**RTF** = Real-Time Factor (1.0x = processes audio at same speed as input)

**Current Selection: `medium`**
- Best balance of accuracy and speed
- ~86% word error rate
- ~200-400ms processing latency
- 5GB RAM requirement

### 2.3 Whisper Parameters Explained

#### Core Parameters

**`beam_size: 5`**
- Beam search width for decoding
- Higher = more accurate but slower
- Range: 1-10
- **5 = optimal balance**

**`best_of: 5`**
- Number of candidate sequences
- Only used when temperature > 0
- Higher = better quality, slower
- **5 = good accuracy improvement**

**`temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`**
- Sampling randomness
- 0.0 = deterministic (greedy)
- Higher = more creative but less accurate
- **Array = fallback strategy**: Try 0.0 first, if fails, try 0.2, etc.

#### Quality Control Parameters

**`compression_ratio_threshold: 2.4`**
- Detects repetitive/low-quality output
- If text compression ratio > 2.4, segment is skipped
- Prevents hallucinations

**`log_prob_threshold: -1.0`**
- Minimum average log probability per token
- Segments below this are skipped
- -1.0 = accept most segments
- More negative = more permissive

**`no_speech_threshold: 0.6`**
- Probability threshold for "no speech" detection
- 0.6 = if >60% confidence of silence, skip segment
- Prevents processing pure silence/noise

#### Context Parameters

**`condition_on_previous_text: true`**
- Uses previous transcription as context
- Improves consistency and accuracy
- Helps with technical terms that repeat
- **Critical for multi-utterance conversations**

**`vad_filter: true`**
- Whisper's internal VAD (separate from WebRTC VAD)
- Pre-filters audio before transcription
- Removes silence at start/end
- Reduces hallucinations

**`word_timestamps: false`**
- When true, provides word-level timing
- **Disabled for lower latency** (~30% faster)
- Enable if you need precise word timing

### 2.4 Initial Prompt (Context Setting)

```python
initial_prompt = "This is a technical conversation about programming and coding."
```

**Purpose:**
- Guides Whisper's language model
- Improves recognition of technical terms
- Helps with programming jargon
- Sets expected vocabulary domain

**Customization:**
- Modify for different domains (medical, legal, etc.)
- Can include specific terminology
- Improves first-utterance accuracy

### 2.5 Transcription Flow

```
Audio Buffer (np.ndarray, float32)
    ↓
Normalization (ensure [-1.0, 1.0] range)
    ↓
Whisper Model (faster-whisper)
    ↓
Segments Generation (with timestamps)
    ↓
Text Assembly (join all segments)
    ↓
Return: text, language, confidence, timing
```

### 2.6 Implementation Code

**WhisperTranscriber Class (`transcription.py`)**

```python
class WhisperTranscriber:
    def __init__(self, **config_params):
        # Load faster-whisper model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        # Transcribe with all configured parameters
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
        
        # Collect segments
        text = " ".join(segment.text.strip() for segment in segments)
        
        return TranscriptionOutput(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=audio_duration,
            processing_time_ms=processing_time
        )
```

---

## 3. Text Cleanup System

### 3.1 Cleanup Configuration

```yaml
text_cleanup:
  remove_filler_words: true
  filler_words:
    - "um"
    - "umm"
    - "uh"
    - "uhh"
    - "ah"
    - "ahh"
    - "er"
    - "err"
    - "like"
    - "you know"
    - "i mean"
    - "sort of"
    - "kind of"
    - "basically"
  normalize_case: true         # Lowercase for consistency
  strip_punctuation: false     # Keep punctuation
```

### 3.2 Text Processing Pipeline

```
Raw Whisper Output
    ↓
Remove Filler Words (um, uh, etc.)
    ↓
Trim Whitespace
    ↓
Normalize Multiple Spaces
    ↓
Clean Text (ready for use)
```

---

## 4. Complete Integration Flow

### 4.1 End-to-End Process

```
[1] Microphone captures audio continuously (100ms blocks)
        ↓
[2] Audio converted to int16 for VAD analysis
        ↓
[3] VAD analyzes 30ms frames for speech detection
        ↓
[4] Speech frames accumulated in buffer
        ↓
[5] Silence detected (500ms threshold)
        ↓
[6] Complete utterance extracted (if >300ms speech)
        ↓
[7] Audio normalized to float32 [-1.0, 1.0]
        ↓
[8] Whisper transcribes with all configured parameters
        ↓
[9] Text cleaned (filler words removed)
        ↓
[10] Final transcribed sentence ready
```

### 4.2 Timing Breakdown

**Typical Latency for 2-second utterance:**

| Stage | Time | Description |
|-------|------|-------------|
| Audio Capture | Real-time | Continuous streaming |
| VAD Processing | ~10ms | Frame-by-frame analysis |
| Silence Detection | 500ms | Waiting for utterance end |
| Whisper (medium) | 200-400ms | Model inference |
| Text Cleanup | <5ms | String processing |
| **Total** | **~700-900ms** | From speech end to text |

### 4.3 Performance Optimization

**Current Optimizations:**
1. `compute_type: "int8"` - 4x faster than float32
2. `word_timestamps: false` - 30% latency reduction
3. `beam_size: 5` - Good accuracy without excessive computation
4. VAD pre-filtering - Reduces audio sent to Whisper
5. Medium model - Best accuracy/speed tradeoff

**Further Optimization Options:**
- Use `small` model → 2x faster, -4% accuracy
- Reduce `beam_size` to 3 → 20% faster, -1% accuracy
- Set `best_of: 1` → 30% faster, -2% accuracy
- Use GPU (`device: "cuda"`) → 3-5x faster with NVIDIA GPU

---

## 5. Implementation in New Project

### 5.1 Required Dependencies

```bash
pip install faster-whisper==1.1.0
pip install sounddevice==0.5.1
pip install webrtcvad==2.0.10
pip install numpy==2.4.1
pip install PyYAML==6.0.3
```

### 5.2 Minimal Implementation

```python
import numpy as np
from faster_whisper import WhisperModel
from audio_capture import AudioCapture

# 1. Initialize audio capture
audio = AudioCapture(
    sample_rate=16000,
    vad_aggressiveness=2,
    silence_threshold=0.5,
    min_speech_duration=0.3
)

# 2. Initialize Whisper
model = WhisperModel(
    "medium",
    device="auto",
    compute_type="int8"
)

# 3. Start capturing
audio.start()

# 4. Process utterances
while True:
    # Get complete utterance from VAD
    audio_chunk = audio.get_audio_chunk(timeout=1.0)
    
    if audio_chunk is not None:
        # Transcribe
        segments, info = model.transcribe(
            audio_chunk,
            language="en",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=True,
            vad_filter=True
        )
        
        # Get text
        text = " ".join(segment.text.strip() for segment in segments)
        print(f"Transcribed: {text}")
```

### 5.3 Configuration Template

Copy this to your `config.yaml`:

```yaml
audio:
  sample_rate: 16000
  channels: 1
  vad_aggressiveness: 2
  silence_threshold: 0.5
  min_speech_duration: 0.3

transcription:
  model_size: "medium"
  device: "auto"
  compute_type: "int8"
  language: "en"
  beam_size: 5
  best_of: 5
  temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  vad_filter: true
  word_timestamps: false
  condition_on_previous_text: true
  compression_ratio_threshold: 2.4
  log_prob_threshold: -1.0
  no_speech_threshold: 0.6

text_cleanup:
  remove_filler_words: true
  normalize_case: true
  strip_punctuation: false
```

---

## 6. Key Takeaways

### Sentence Detection Summary

**Sentences stop when:**
1. ✅ VAD detects ≥500ms of silence
2. ✅ Total speech duration ≥300ms
3. ✅ Audio has sufficient energy

**Controlled by:**
- `silence_threshold: 0.5` - Waiting time after speech
- `min_speech_duration: 0.3` - Minimum valid speech
- `vad_aggressiveness: 2` - Sensitivity level

### Critical Whisper Settings

**Must-have configs:**
- `sample_rate: 16000` - Whisper requirement
- `model_size: "medium"` - Best balance
- `compute_type: "int8"` - Speed optimization
- `condition_on_previous_text: true` - Consistency
- `temperature: [0.0, ...]` - Fallback strategy

**Quality controls:**
- `compression_ratio_threshold: 2.4` - Anti-hallucination
- `log_prob_threshold: -1.0` - Confidence filter
- `no_speech_threshold: 0.6` - Silence detection

### Files to Copy

For implementing in another project, copy these files:
1. `src/audio_capture.py` - Audio capture + VAD
2. `src/transcription.py` - Whisper integration
3. `src/utils.py` - Text cleanup (optional)
4. `config/config.yaml` - All configurations

---

**Last Updated:** January 2026  
**Project Version:** 1.0  
**Whisper Model:** medium (faster-whisper 1.1.0)
