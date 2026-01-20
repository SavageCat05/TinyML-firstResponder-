# Context: Audio-Based Emergency Intent Detection (TinyML)

## Overview

This project aims to build an **audio-first emergency intent detection system** using **TinyML**. The system listens to short audio snippets from a microphone, interprets **spoken words, loudness, and tone** (human-like cues), classifies the user’s **intent**, and triggers the appropriate **emergency response** (e.g., Police, Ambulance, Fire, Women Helpline).

The key constraint is **edge deployment**: the model must be **tiny, fast, int8-quantized**, and capable of running on **low-power devices** (microcontrollers or edge boards) with minimal latency.

---

## Core Objective

* Understand **intent**, not just keywords
* Use **acoustic cues** (stress, urgency, loudness, pitch variation)
* Operate in **real time** with short audio windows (≈1–2 seconds)
* Reliably classify emergencies even with:

  * Partial speech
  * Panic or shouting
  * Non-perfect grammar

---

## Input Characteristics

### Audio Input

* Source: **Microphone**
* Format: Mono PCM
* Sample Rate: 8–16 kHz (TinyML-friendly)
* Frame Size: 20–40 ms

### Human-Like Signals Considered (Important)

* **Lexical**: words like *help*, *fire*, *attack*, *hurt*
* **Prosodic**:

  * Loudness (RMS energy)
  * Pitch / pitch variance
  * Speaking rate
* **Emotional cues**:

  * Panic
  * Distress
  * Aggression

These cues together approximate how humans infer urgency.

---

## System Flow

```
Microphone
  ↓ (audio frames)
Thread-Safe Queue
  ↓
Sliding Buffer (≈1–2 sec window)
  ↓
TinyML Model (int8, on-device)
  ↓
Text / Intent Segments
  ↓
Emergency Action Output
```

---

## Processing Pipeline

### 1. Audio Capture

* Continuous microphone sampling
* Audio divided into short frames
* Frames pushed into a **thread-safe queue** to decouple capture and inference

---

### 2. Sliding Buffer

* Maintains a rolling window of ~1–2 seconds
* Enables:

  * Temporal context
  * Detection of urgency patterns (not single-frame decisions)

---

### 3. Feature Extraction (On-Device)

Minimal but expressive features:

* MFCCs or log-mel spectrograms
* Energy (loudness)
* Pitch-related features (if feasible)

Designed to be:

* Deterministic
* Low memory
* Real-time

---

### 4. TinyML Model

**Model Characteristics**:

* Architecture: CNN / CNN + lightweight RNN
* Quantization: **int8**
* Size: < 500 KB (target)
* Latency: < 100 ms per window

**Model Outputs**:

* Intent class probabilities

Example intent classes:

* Police Emergency
* Medical Emergency
* Fire Emergency
* Women Safety
* General Distress
* Non-Emergency / Noise

---

### 5. Text / Intent Segments

* Model may optionally output:

  * Keyword likelihoods
  * Confidence scores
* Temporal smoothing applied to avoid false triggers

---

### 6. Decision & Action Layer

Based on the classified intent:

| Intent           | Action             |
| ---------------- | ------------------ |
| Police           | Call **100**       |
| Ambulance        | Call **108**       |
| Fire             | Call **101**       |
| Women Safety     | Call **1091**      |
| General Distress | Escalation / Retry |

Fail-safe logic:

* Require repeated confirmation across windows
* Fallback to manual trigger if confidence is low

---

## Key Design Principles

* **Edge-first**: no cloud dependency
* **Human-like reasoning**: tone + words + urgency
* **Fail-safe by default**: minimize false negatives
* **Explainable**: confidence and reason for trigger

---

## Constraints & Assumptions

* Limited RAM and flash
* No full speech-to-text (STT) pipeline
* Works with noisy environments
* Prioritizes **speed and reliability** over linguistic completeness

---

## Intended Use Cases

* Wearables
* Smart safety devices
* Public emergency kiosks
* Assistive devices for vulnerable users

---

## Summary

This system mimics how humans recognize emergencies from **short, emotional, imperfect audio signals**, using TinyML to deliver **fast, reliable, on-device emergency intent detection** and immediate response triggering.
