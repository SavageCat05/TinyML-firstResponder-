"""
Feature extraction module for TinyML Emergency Intent Detection
Extracts MFCCs, log-mel spectrograms, loudness, and pitch features as per v1 spec
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
import logging
from .config import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT,
    MODEL_INPUT_SHAPE
)

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts acoustic features for emergency intent detection
    Designed for low memory and real-time processing
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mels = N_MELS
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT

        # Pre-compute mel filterbank for efficiency
        self.mel_filterbank = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )

        logger.info("Feature extractor initialized (Log-Mel Spectrogram only)")

    def extract_features(self, audio_window):
        """
        Extract log-mel spectrogram from audio window
        Returns features suitable for TinyML model input

        Args:
            audio_window: numpy array of audio samples (1.5 seconds), expected 16kHz mono

        Returns:
            numpy array: Log-Mel Spectrogram
        """
        if audio_window is None or len(audio_window) == 0:
            return None

        try:
            # Ensure proper audio format and normalize
            audio = audio_window.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # 1. Log-Mel Spectrogram (primary feature for CNN)
            log_mel = self._extract_log_mel(audio)
            
            return log_mel

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def _extract_log_mel(self, audio):
        """
        Extract log-mel spectrogram
        Returns shape suitable for CNN input: (time, freq, 1)
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann'
        )

        # Convert to power spectrogram
        power_spec = np.abs(stft) ** 2

        # Apply mel filterbank
        mel_spec = np.dot(self.mel_filterbank, power_spec)

        # Convert to log scale (add small epsilon to avoid log(0))
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] range for model input
        if np.max(log_mel) > np.min(log_mel):
            log_mel = (log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel))

        # Reshape for CNN input (freq, time, channel)
        # Target shape: MODEL_INPUT_SHAPE = (40, 50, 1)
        # 40 = mel frequency bins, 50 = time steps
        target_freq, target_time = MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1]

        # Resize frequency dimension
        if log_mel.shape[0] != target_freq:
            log_mel = librosa.resample(
                log_mel,
                orig_sr=log_mel.shape[0],
                target_sr=target_freq,
                axis=0
            )

        # Resize time dimension
        if log_mel.shape[1] != target_time:
            if log_mel.shape[1] > target_time:
                # Truncate
                log_mel = log_mel[:, :target_time]
            else:
                # Pad with zeros
                pad_width = target_time - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')

        # Add channel dimension
        log_mel = log_mel[..., np.newaxis]

        return log_mel.astype(np.float32)



    def get_model_input(self, features):
        """
        Prepare features for TinyML model input
        Returns the primary log-mel spectrogram for CNN
        """
        if features is None:
            return None

        # Since extract_features now directly returns log_mel
        return features