# TinyML First Responder - Speech-to-Text Distress Detection

A real-time audio-based emergency intent detection system using TinyML and Whisper for edge deployment.

---

## 📁 Project Structure

```
TinyML-firstResponder-/
├── src/
│   └── speech-to-text_.py    # Main distress detection script
├── models/
│   └── emergency_intent_model.tflite  # TinyML model (if available)
├── docs/
│   └── firstResponder context.md      # Project documentation
├── requirements.txt           # Python dependencies
├── emergency_log.txt          # Detection logs (auto-generated)
└── venv/                      # Virtual environment
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Working microphone
- Windows/Linux/macOS

### 1. Clone the Repository

```bash
git clone https://github.com/SavageCat05/TinyML-firstResponder-.git
cd TinyML-firstResponder-
```

### 2. Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Speech-to-Text Detector

```bash
python src/speech-to-text_.py
```

---

## 🎙️ How It Works

The `speech-to-text_.py` script:

1. **Captures audio** continuously from your microphone (16kHz, mono)
2. **Buffers audio** in 2.5-second chunks with 0.5s overlap
3. **Transcribes speech** using Whisper Tiny (int8 quantized, CPU-optimized)
4. **Detects distress keywords**: `help`, `emergency`, `save me`, `danger`, `please help`
5. **Logs alerts** to `distress_log.txt` with timestamps
6. **Voice commands**: Say "stop listening", "shutdown", or "exit program" to stop

### Output Examples

```
🎙️ Listening for distress calls...
📝 hello testing the microphone
🚨 DISTRESS: please help me
📝 everything is fine now
🛑 Voice command received. Stopping.
```

---

## ⚙️ Configuration

Edit the constants in `src/speech-to-text_.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `CHUNK_DURATION` | 2.5 | Audio chunk length (seconds) |
| `OVERLAP_DURATION` | 0.5 | Overlap between chunks (seconds) |
| `CONFIDENCE_THRESHOLD` | -1.0 | Whisper log probability threshold |
| `KEYWORDS` | ["help", "emergency", ...] | Distress trigger words |
| `STOP_WORDS` | ["stop listening", ...] | Voice commands to exit |

---

## 📊 Log Format

Logs are saved to `distress_log.txt`:

```
[2026-01-21 01:30:15] [INFO] hello testing
[2026-01-21 01:30:18] [ALERT] please help me
[2026-01-21 01:30:25] [INFO] Voice stop command received
```

---

## 🌐 Edge Device Simulation Guide

### : Wokwi (ESP32 Simulator)

[Wokwi](https://wokwi.com) simulates ESP32/Arduino with audio capabilities.

1. Go to [wokwi.com](https://wokwi.com)
2. Create a new ESP32 project
3. For TinyML audio, you'll need to:
   - Export the model to TFLite Micro format
   - Use the [TensorFlow Lite Micro library](https://github.com/tensorflow/tflite-micro)
   - Simulate with I2S microphone (INMP441)

**Example Wokwi diagram.json:**
```json
{
  "parts": [
    {"type": "esp32", "id": "esp32"},
    {"type": "inmp441", "id": "mic"}
  ],
  "connections": [
    ["esp32:GPIO25", "mic:WS"],
    ["esp32:GPIO26", "mic:SCK"],
    ["esp32:GPIO27", "mic:SD"]
  ]
}
```
---

## 🔧 Converting to Edge-Deployable Format

### Export to TFLite (Int8 Quantized)

```python
import tensorflow as tf

# Convert model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()

# Save
with open("emergency_model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

### Generate C Array for Microcontroller

```bash
xxd -i emergency_model_int8.tflite > model_data.h
```

---

## 📱 Hardware Deployment Targets

| Platform | RAM | Flash | Recommended |
|----------|-----|-------|-------------|
| ESP32-S3 | 512KB | 8MB | ✅ Best for audio |
| Arduino Nano 33 BLE | 256KB | 1MB | ✅ Good |
| Raspberry Pi Pico | 264KB | 2MB | ✅ Good |
| STM32F4 | 192KB | 1MB | ⚠️ Tight fit |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'sounddevice'` | `pip install sounddevice` |
| `No input device found` | Check microphone permissions |
| Whisper loads slowly | First run downloads model (~75MB) |
| High CPU usage | Use `compute_type="int8"` (already set) |

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push: `git push origin feature/new-feature`
5. Open Pull Request
