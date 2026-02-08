"""
TinyML Emergency Intent Detection System v1

A complete edge-deployable audio-based emergency intent detection system
using TinyML for real-time classification on low-power devices.

Components:
- Audio capture with thread-safe queue and sliding buffer
- Feature extraction (MFCCs, log-mel, loudness, pitch)
- TinyML CNN model with int8 quantization
- Emergency action triggers with temporal smoothing

Target: <500KB model, <100ms latency, microcontroller deployment
"""

from .config import *
from .audio_capture import AudioCapture
from .feature_extraction import FeatureExtractor
from .emergency_detector import EmergencyDetector
from .main import TinyMLEmergencyApp

__version__ = "1.0.0"
__author__ = "TinyML Emergency Detection Team"