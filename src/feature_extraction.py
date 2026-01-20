"""
Feature extraction module for TinyML Emergency Intent Detection
Extracts MFCCs, log-mel spectrograms, loudness, and pitch features as per v1 spec
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
import logging
from .config import (
    SAMPLE_RATE, N_MFCC, N_MELS, HOP_LENGTH, N_FFT,
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
        self.n_mfcc = N_MFCC
        self.n_mels = N_MELS
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT

        # Pre-compute mel filterbank for efficiency
        self.mel_filterbank = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )

        logger.info("Feature extractor initialized")

    def extract_features(self, audio_window):
        """
        Extract all features from audio window
        Returns features suitable for TinyML model input

        Args:
            audio_window: numpy array of audio samples (1.5 seconds)

        Returns:
            dict: Dictionary containing all extracted features
        """
        if audio_window is None or len(audio_window) == 0:
            return None

        try:
            # Ensure proper audio format
            audio = audio_window.astype(np.float32)

            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            features = {}

            # 1. Log-Mel Spectrogram (primary feature for CNN)
            mel_spec = self._extract_log_mel(audio)
            features['log_mel'] = mel_spec

            # 2. MFCCs (alternative/complementary features)
            mfccs = self._extract_mfccs(audio)
            features['mfccs'] = mfccs

            # 3. Loudness/Energy features
            loudness_features = self._extract_loudness_features(audio)
            features.update(loudness_features)

            # 4. Pitch-related features
            pitch_features = self._extract_pitch_features(audio)
            features.update(pitch_features)

            # 5. Temporal features (speaking rate approximation)
            temporal_features = self._extract_temporal_features(audio)
            features.update(temporal_features)

            return features

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

    def _extract_mfccs(self, audio):
        """
        Extract MFCC coefficients
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)

        return mfccs.astype(np.float32)

    def _extract_loudness_features(self, audio):
        """
        Extract loudness/energy-based features
        """
        features = {}

        # RMS energy (loudness)
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)
        features['rms_energy'] = rms.flatten()

        # Zero-crossing rate (related to noisiness/percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)
        features['zero_crossing_rate'] = zcr.flatten()

        # Spectral centroid (brightness indicator)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_centroid'] = centroid.flatten()

        return features

    def _extract_pitch_features(self, audio):
        """
        Extract pitch-related features
        """
        features = {}

        try:
            # Fundamental frequency estimation using YIN algorithm
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )

            # Remove NaN values
            f0 = np.nan_to_num(f0, nan=0.0)

            features['fundamental_freq'] = f0
            features['pitch_variance'] = np.var(f0[f0 > 0]) if np.any(f0 > 0) else 0.0
            features['voiced_ratio'] = np.mean(voiced_flag) if len(voiced_flag) > 0 else 0.0

        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            features['fundamental_freq'] = np.zeros(10)  # Dummy values
            features['pitch_variance'] = 0.0
            features['voiced_ratio'] = 0.0

        return features

    def _extract_temporal_features(self, audio):
        """
        Extract temporal features (approximating speaking rate)
        """
        features = {}

        # Frame energy variation (tempo indicator)
        frame_length = int(self.sample_rate * 0.025)  # 25ms frames
        hop_length = int(self.sample_rate * 0.010)    # 10ms hop

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)

        # Energy peaks (potential speech segments)
        peaks, _ = find_peaks(frame_energies, height=np.mean(frame_energies), distance=10)
        features['energy_peaks_count'] = len(peaks)

        # Speaking rate approximation (peaks per second)
        window_duration = len(audio) / self.sample_rate
        features['speaking_rate'] = len(peaks) / window_duration if window_duration > 0 else 0.0

        return features

    def get_model_input(self, features):
        """
        Prepare features for TinyML model input
        Returns the primary log-mel spectrogram for CNN
        """
        if features is None or 'log_mel' not in features:
            return None

        return features['log_mel']