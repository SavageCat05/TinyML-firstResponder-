"""
Configuration constants for TinyML Emergency Intent Detection System v1
"""

# Audio Configuration
SAMPLE_RATE = 16000  # 16 kHz for TinyML compatibility
CHANNELS = 1  # Mono
FRAME_DURATION_MS = 30  # 30ms frames (between 20-40ms spec)
SLIDING_WINDOW_SECONDS = 1.5  # 1.5 second rolling window (within 1-2s spec)

# Feature Extraction
N_MFCC = 13  # Number of MFCC coefficients
N_MELS = 40  # Number of mel bins for log-mel spectrogram
HOP_LENGTH = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Frame hop size
N_FFT = 512  # FFT window size

# Model Configuration
MODEL_INPUT_SHAPE = (40, 50, 1)  # Time x Freq x Channel (log-mel spectrogram)
NUM_CLASSES = 6  # Emergency intent classes
MODEL_PATH = "models/emergency_intent_model.tflite"

# Intent Classes (matching v1 spec)
INTENT_CLASSES = [
    "police_emergency",      # 0
    "medical_emergency",     # 1
    "fire_emergency",        # 2
    "women_safety",          # 3
    "general_distress",      # 4
    "non_emergency_noise"    # 5
]

# Emergency Actions (Indian emergency numbers)
EMERGENCY_ACTIONS = {
    "police_emergency": {"number": "100", "description": "Police Emergency"},
    "medical_emergency": {"number": "108", "description": "Medical Emergency (Ambulance)"},
    "fire_emergency": {"number": "101", "description": "Fire Emergency"},
    "women_safety": {"number": "1091", "description": "Women Safety Helpline"},
    "general_distress": {"action": "escalation", "description": "General Distress - Escalation Required"},
    "non_emergency_noise": {"action": "ignore", "description": "Non-emergency audio"}
}

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for action
TEMPORAL_CONFIRMATIONS = 3  # Number of consecutive windows needed for confirmation

# Performance Constraints (v1 spec)
MAX_MODEL_SIZE_KB = 500  # < 500KB target
MAX_LATENCY_MS = 100     # < 100ms per window
TARGET_MEMORY_MB = 32    # Target RAM usage

# Logging
LOG_FILE = "emergency_log.txt"