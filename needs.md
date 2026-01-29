```markdown
# 🎙️ Real-Time Voice Transcription & Coding Question Classifier Agent — System Prompt

## 🎯 Objective

Design and implement a real-time AI system that:

1. Continuously listens to live microphone audio input.
2. Uses **OpenAI Whisper (or Whisper-based models)** to transcribe speech into text with **very low latency**.
3. Classifies each transcribed utterance **strictly** into:
   - **CODING QUESTION**
   - **NON-CODING QUESTION**

This classification must be **extremely precise** and **conservative** — only real coding implementation problems should be classified as coding questions.

---

## 🧠 Core Functional Requirements

### 1. Real-Time Audio Processing
- Capture live microphone audio in real time.
- Perform **streaming speech-to-text transcription** using Whisper.
- Transcription latency must be:
  - Target: **<300ms**
  - Max acceptable: **<700ms**

### 2. Transcription Accuracy
- Must correctly detect:
  - Programming keywords
  - Code structure language
  - Variable references
  - Algorithmic phrasing
  - Implementation-oriented questions

### 3. Real-Time Classification
- Immediately classify **every completed spoken segment** into:
  - `CODING`
  - `NON-CODING`

---

## 🧩 Strict Classification Rules

### A spoken query must be classified as **CODING** ONLY if:

It **explicitly requires writing actual executable code**, such as:

- "Write a Python function to reverse a linked list"
- "Give me Java code for Dijkstra's algorithm"
- "Implement binary search in C++"
- "Write SQL query to fetch duplicate rows"
- "Create a REST API using Flask"
- "Write a React component that displays a list"

#### Mandatory Conditions:
- Requires **actual program logic**
- Requires **code output**
- Requires **implementation**

---

### A spoken query must be classified as **NON-CODING** if it includes:

- General chit-chat  
- Casual conversation  
- System instructions  
- Explanations  
- Conceptual or theoretical questions  
- Career questions  
- Debug reasoning without code writing  
- Architecture discussions  
- Conceptual definitions  

#### Examples that MUST be classified as NON-CODING:

- "What is machine learning?"
- "Explain how blockchain works"
- "What is OOP?"
- "How does recursion work?"
- "What is the difference between stack and queue?"
- "Can you explain quicksort?"
- "Is Python better than Java?"
- "How do APIs work?"
- "What is overfitting?"
- "Tell me about AI"
- "How to prepare for coding interviews?"

---

### CRITICAL RULE:

If a question **does NOT require writing code**, it MUST be classified as **NON-CODING**, even if it is programming related.

---

## 🧪 Edge Case Handling

| Input Example | Expected Output |
|---------------|----------------|
| "Explain merge sort" | NON-CODING |
| "Write merge sort in python" | CODING |
| "How does HTTP work?" | NON-CODING |
| "Implement HTTP server in Node.js" | CODING |
| "What is recursion?" | NON-CODING |
| "Write recursive factorial function" | CODING |

---

## ⚙️ Processing Pipeline Architecture

### Step 1: Audio Capture
- Continuous microphone stream
- Buffer audio into 1–2 second chunks

### Step 2: Speech-to-Text (Whisper)
- Use **faster-whisper / whisper.cpp / Whisper small or medium**
- Apply:
  - Beam search
  - Noise filtering
  - Silence trimming

### Step 3: Text Cleanup
- Normalize casing
- Remove filler words ("umm", "uhh", etc.)
- Punctuation restoration

### Step 4: Classification Engine
- Rule-based + LLM-based classification:
  - Detect **explicit coding intent**
  - Identify **code writing verbs**:
    - write
    - implement
    - generate
    - build
    - create
    - code
    - program

### Step 5: Final Decision Output
- Output label:
```

CODING

```
or
```

NON-CODING

````

---

## 🧠 Classification Decision Logic

Classify as **CODING** if and only if:

- Transcription contains:
- "write code"
- "implement"
- "generate code"
- "create program"
- "build API"
- "write function"
- "code for"

AND

- The sentence structure clearly demands **actual implementation**

Else:

➡ Always return **NON-CODING**

---

## 🏗️ Performance Requirements

| Metric | Target |
|---------|--------|
| End-to-end latency | < 500 ms |
| Transcription WER | < 10% |
| Classification accuracy | > 98% |
| False positive rate | < 1% |

---

## 🛠️ Technology Recommendations

### Transcription
- faster-whisper
- whisper.cpp
- OpenAI Whisper small / medium

### Audio Streaming
- PyAudio
- sounddevice
- WebRTC (for browser-based)

### Classification
- Lightweight local LLM
- Regex + intent detection
- Distilled transformer classifier

---

## 🧪 Test Cases

| Voice Input | Expected Output |
|-------------|-----------------|
| "Write Python code for BFS" | CODING |
| "Explain BFS algorithm" | NON-CODING |
| "How does recursion work" | NON-CODING |
| "Implement recursion for factorial" | CODING |
| "What is a compiler?" | NON-CODING |
| "Create a compiler in C" | CODING |

---

## 🚨 Hard Safety Constraints

- NEVER classify theoretical coding questions as CODING.
- NEVER classify chit-chat as CODING.
- If ambiguity exists → default to **NON-CODING**.

---

## 📦 Output Format

The system must return:

```json
{
"transcription": "<final_text>",
"classification": "CODING | NON-CODING",
"confidence": 0.00 – 1.00
}
````

---

## 🧠 Final System Directive

Your classification must be **extremely strict**.

Only **real software development implementation requests** should ever be labeled as:

> CODING

Everything else must be:

> NON-CODING

No exceptions.

```

---

If you want, I can also generate:

- **Full working Python pipeline**
- **Streaming Whisper + classifier implementation**
- **Real-time microphone + low latency architecture**
- **Web-based live dashboard**

Just say 👍
::contentReference[oaicite:0]{index=0}
```
