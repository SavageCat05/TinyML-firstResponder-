"""Pytest configuration and fixtures for TinyML Emergency Detector tests."""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_audio_data():
    """Fixture providing sample audio data for testing."""
    # Generate 1 second of dummy audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Create a simple sine wave as test audio
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples, False)
    audio = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def sample_spectrogram():
    """Fixture providing sample spectrogram data for testing."""
    from config import MODEL_INPUT_SHAPE

    # Generate dummy spectrogram data
    spectrogram = np.random.rand(*MODEL_INPUT_SHAPE).astype(np.float32)
    return spectrogram