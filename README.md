# TinyML Emergency Intent Detection System v1

A complete edge-deployable audio-based emergency intent detection system using TinyML for real-time classification on low-power devices.

## 🎯 Overview

This system listens to audio, and uses a TinyML CNN model to classify user intent into 6 emergency categories. When an emergency is detected with sufficient confidence, it automatically triggers the appropriate emergency response.

**Key Features:**
- **Edge-first design**: No cloud dependency, runs on microcontrollers
- **Real-time processing**: <100ms latency per inference
- **TinyML optimized**: <500KB quantized model
- **Fail-safe**: Temporal confirmation prevents false alarms (this means for eg out of 10 events, if 7 (eg threshold) show true -> model returns true)

## 🚨 Emergency Classes

The system detects 6 types of emergency intent:

1. **Police Emergency** → Calls 100 (Police)
2. **Medical Emergency** → Calls 108 (Ambulance)
3. **Fire Emergency** → Calls 101 (Fire)
4. **Women Safety** → Calls 1091 (Women Helpline)
5. **General Distress** → Escalation protocol
6. **Non-Emergency** → Continue monitoring

## 🏗️ System Architecture

```
Microphone → Audio Capture → Feature Extraction → TinyML Model → Decision Layer → Emergency Action
     ↓           ↓              ↓                    ↓             ↓               ↓
   16kHz PCM  Thread-Safe    Log-Mel/MFCCs      CNN (int8)   Confidence      Phone Call
   Mono       Queue         Loudness/Pitch    <500KB       Threshold       Trigger
```

### Core Components

- **Audio Capture**: Continuous sampling with 30ms frames, 1.5s sliding buffer
- **Feature Extraction**: MFCCs (describe how audio sounds to human ears), log-mel spectrograms, RMS energy, pitch variance
- **TinyML Model**: MobileNet-style CNN with inverted residuals, SE attention blocks, and int8 quantization (~41K params, ~161KB)
- **Decision Logic**: 70% confidence threshold + 3-window temporal confirmation
- **Emergency Actions**: Automatic calls to Indian emergency numbers

## 📋 Requirements

- Python 3.8+
- Microphone access
- ~32MB RAM (target for microcontroller deployment)

### Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Train the Model (Development)

```bash
python train_model.py
```

**Note**: Currently uses synthetic data for development. Production requires real emergency audio datasets.

### 2. Benchmark Performance

```bash
python run_emergency_detector.py benchmark
```

### 3. Run Emergency Detection

```bash
python run_emergency_detector.py run
```

The system will:
- 🎙️ Start listening for audio
- 🔄 Process 1.5-second windows continuously
- 🧠 Run TinyML inference every ~100ms
- 🚨 Trigger emergency calls when confident
- 📝 Log all detections

## 🔧 Configuration

Key parameters in `src/config.py`:

```python
SAMPLE_RATE = 16000          # 16kHz audio
SLIDING_WINDOW_SECONDS = 1.5 # 1.5s analysis window
CONFIDENCE_THRESHOLD = 0.7   # 70% confidence required
TEMPORAL_CONFIRMATIONS = 3   # 3 consecutive detections
MAX_MODEL_SIZE_KB = 500      # <500KB target
MAX_LATENCY_MS = 100         # <100ms latency target
```

## 📊 Performance Targets (v1 Spec)

- **Model Size**: <500KB (quantized)
- **Latency**: <100ms per inference
- **Memory**: <32MB RAM usage
- **Accuracy**: >70% confidence threshold
- **False Positive Rate**: Minimized via temporal smoothing

## 🛠️ Development

### Project Structure

```
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # System configuration
│   ├── audio_capture.py      # Audio capture & buffering
│   ├── feature_extraction.py # Acoustic feature extraction
│   ├── model.py              # TinyML CNN model
│   ├── emergency_detector.py # Main orchestrator
│   └── main.py               # Application interface
├── models/                   # Trained models directory
├── docs/
│   ├── v1_context.md         # System requirements
│   └── v1_workflow.mmd       # Processing flowchart
├── requirements.txt          # Python dependencies
├── run_emergency_detector.py # Main run script
├── train_model.py           # Training script
└── README.md                # This file
```

### Adding Emergency Audio Data

For production deployment, replace synthetic training data with real emergency audio:

1. Collect audio samples for each emergency class
2. Label with correct intent classes
3. Update training data generation in `train_model.py`
4. Retrain model with real data

### Microcontroller Deployment

The quantized TFLite model (`models/emergency_intent_model.tflite`) can be deployed to:

- **ESP32** with TensorFlow Lite Micro
- **Arduino Nano 33 BLE** with EloquentTinyML
- **Raspberry Pi Pico** with MicroPython
- **Coral Dev Board** for accelerated inference

## 📈 Monitoring & Logs

- **Console Output**: Real-time status and alerts
- **Emergency Log**: `emergency_log.txt` - all detections
- **Performance Metrics**: Latency, buffer status, inference count

## ⚠️ Safety & Fail-Safes

- **Temporal Confirmation**: Requires 3 consecutive detections
- **Confidence Threshold**: 70% minimum confidence
- **Action Cooldown**: 5-second gap between triggers
- **Manual Override**: Can be stopped with Ctrl+C
- **Fallback Logic**: Escalation for uncertain cases

## 🔬 Technical Details

### Feature Extraction
- **Log-Mel Spectrogram**: 40 mel bins, 50 time frames
- **MFCCs**: 13 coefficients for complementary features
- **Loudness**: RMS energy for urgency detection
- **Pitch**: Fundamental frequency and variance
- **Temporal**: Speaking rate and energy patterns

### Model Architecture Evolution

#### v1.0 - Simple CNN (Previous)
```
Input: (40, 50, 1) log-mel spectrogram
Conv2D(16, 3x3) → MaxPool → BatchNorm
Conv2D(32, 3x3) → MaxPool → BatchNorm
Conv2D(64, 3x3) → MaxPool → BatchNorm
GlobalAvgPool → Dense(64) → Dropout(0.3) → Dense(6, softmax)
```

#### v2.0 - MobileNet-Style (Current)
```
Input: (40, 50, 1) log-mel spectrogram
│
├── Stem: Conv2D(16, 3x3, stride=2) → BatchNorm → ReLU6
│
├── Block 1: Inverted Residual (expansion=2x, filters=24)
│   └── 1x1 Expand → 3x3 Depthwise → 1x1 Project
│
├── Block 2: Inverted Residual (expansion=4x, filters=32) + SE Attention
│   └── Squeeze-Excite recalibrates channel importance
│
├── Block 3: Inverted Residual (expansion=4x, filters=64) + SE Attention
│
└── Head: GlobalMaxPool → Dense(64) → Dropout(0.3) → Dense(6, softmax)
```

### Architecture Comparison

| Aspect | v1.0 Simple CNN | v2.0 MobileNet-Style |
|--------|-----------------|----------------------|
| **Parameters** | ~50-80K | 41,334 |
| **Model Size** | ~300-400KB | ~161KB |
| **Convolutions** | Standard Conv2D | Depthwise Separable |
| **Attention** | None | Squeeze-and-Excitation |
| **Activation** | ReLU | ReLU6 (quantization-friendly) |
| **Pooling** | Global Average | Global Max (better for sparse events) |
| **Skip Connections** | None | Residual connections |

### When v2.0 Works Better

1. **Noisy Environments** (traffic, crowds, TV background)
   - SE blocks dynamically suppress irrelevant frequencies
   - Focuses attention on voice/distress signal frequencies

2. **Sparse Audio Events** (sudden screams, single keywords)
   - Global Max Pooling captures peak activations
   - Better than averaging which dilutes brief signals

3. **Resource-Constrained Devices** (ESP32, Arduino Nano 33)
   - 2x smaller model with same or better accuracy
   - Depthwise separable convs = ~9x fewer FLOPs per layer

4. **Real-Time Edge Inference**
   - ReLU6 clipping makes int8 quantization more accurate
   - BatchNorm fusion reduces inference ops

5. **Gradient Flow During Training**
   - Inverted residuals prevent vanishing gradients
   - Better learns subtle emergency audio cues

### Quantization
- **Post-training quantization** to int8
- **Representative dataset** for calibration
- **Latency optimization** for real-time performance

## 📞 Emergency Integration

In production, emergency actions integrate with:
- **Telephony**: Automatic calls to emergency numbers
- **Location Services**: GPS coordinates included
- **Data Transmission**: Intent confidence and audio context
- **Verification**: Call recording and logging

## 🤝 Contributing

1. Follow the v1 specifications strictly
2. Maintain TinyML constraints (<500KB, <100ms)
3. Test on actual edge hardware
4. Include real emergency audio data
5. Document all changes

## 📜 License

This project implements the v1 specification for TinyML emergency detection. See docs/v1_context.md for detailed requirements.

---

**Version 1.0.0** - TinyML Emergency Intent Detection System