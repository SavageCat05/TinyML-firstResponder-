import numpy as np
import tensorflow as tf
import logging
import os
from pathlib import Path

from src.config import SAMPLE_RATE, SLIDING_WINDOW_SECONDS, NUM_CLASSES, MODEL_INPUT_SHAPE
from src.features.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)

class AudioDataGenerator:
    """
    Generates batches of audio data and corresponding labels for training.
    Includes placeholders for noise augmentation and synthetic mixing.
    """
    def __init__(self, data_dir=None, batch_size=32, shuffle=True):
        self.data_dir = data_dir # Directory where raw audio files might be stored
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_extractor = FeatureExtractor()

        # In a real scenario, you'd load file paths and labels here
        # For now, we'll simulate a dataset
        self.filepaths = [] # List of audio file paths
        self.labels = []    # Corresponding labels

        if self.data_dir:
            logger.info(f"Initialized data generator with data_dir: {self.data_dir}")
            # Placeholder for loading actual audio file paths and labels
            # e.g., self._load_filepaths_and_labels()
        else:
            logger.warning("No data_dir provided. Using dummy data generation.")
            self._generate_dummy_filepaths_and_labels(num_samples=1000)

        self.num_samples = len(self.filepaths)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_dummy_filepaths_and_labels(self, num_samples):
        """
        Generates dummy file paths and labels for demonstration.
        In a real scenario, these would point to actual audio files.
        """
        for i in range(num_samples):
            # Dummy file path, not actually used for loading
            self.filepaths.append(f"dummy_audio_{i}.wav")
            self.labels.append(np.random.randint(0, NUM_CLASSES))
        logger.info(f"Generated {num_samples} dummy audio samples.")

    def _load_and_process_audio(self, filepath, label):
        """
        Loads an audio file, applies preprocessing, and returns features and label.
        Placeholder for actual audio loading.
        """
        # Simulate loading raw audio
        # For noise augmentation, this is where raw audio would be mixed
        audio_duration_samples = int(SAMPLE_RATE * SLIDING_WINDOW_SECONDS)
        
        # Base audio (e.g., a speech segment or background noise)
        raw_audio = np.random.randn(audio_duration_samples).astype(np.float32) * 0.1

        # Apply noise augmentation if this is for training data
        if np.random.rand() < 0.5: # 50% chance to add "noise"
            noise = np.random.randn(audio_duration_samples).astype(np.float32) * 0.05
            raw_audio += noise
            logger.debug("Applied dummy noise augmentation to raw audio.")
        
        # Simulate synthetic emergency signals mixing (for specific labels)
        if label != 0 and np.random.rand() < 0.1: # 10% chance to add emergency sound to non-non_emergency
            t = np.linspace(0, SLIDING_WINDOW_SECONDS, audio_duration_samples, endpoint=False)
            emergency_sound = np.sin(2 * np.pi * 880 * t) * 0.2 # 880 Hz tone
            raw_audio += emergency_sound
            logger.debug(f"Applied dummy synthetic emergency sound for label {label}.")

        # Extract features
        features = self.feature_extractor.extract_features(raw_audio)
        if features is None:
            # Fallback for failed feature extraction
            return np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32), tf.keras.utils.to_categorical(label, num_classes=NUM_CLASSES)

        return features, tf.keras.utils.to_categorical(label, num_classes=NUM_CLASSES)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """
        Generates one batch of data.
        """
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []

        for i in batch_indices:
            filepath = self.filepaths[i]
            label = self.labels[i]
            
            x, y = self._load_and_process_audio(filepath, label)
            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def to_tf_dataset(self):
        """
        Converts the generator to a tf.data.Dataset for use with Keras model.fit.
        """
        output_signature = (
            tf.TensorSpec(shape=MODEL_INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
        )
        return tf.data.Dataset.from_generator(
            self._tf_generator,
            output_signature=output_signature
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _tf_generator(self):
        """
        Internal generator for tf.data.Dataset.
        """
        for i in range(self.num_samples):
            filepath = self.filepaths[i]
            label = self.labels[i]
            x, y = self._load_and_process_audio(filepath, label)
            yield x, y

# Example usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy generator
    data_generator = AudioDataGenerator(batch_size=16)
    
    # Test getting a batch
    first_batch_x, first_batch_y = data_generator[0]
    logger.info(f"Shape of first batch X: {first_batch_x.shape}")
    logger.info(f"Shape of first batch Y: {first_batch_y.shape}")
    logger.info(f"Example label: {first_batch_y[0]}")

    # Test iteration
    for i in range(3):
        batch_x, batch_y = data_generator[i]
        logger.info(f"Batch {i}: X shape {batch_x.shape}, Y shape {batch_y.shape}")

    # Test with tf.data.Dataset
    tf_dataset = data_generator.to_tf_dataset()
    for epoch in range(1):
        logger.info(f"Epoch {epoch+1}")
        for batch_x, batch_y in tf_dataset:
            logger.info(f"TF Dataset Batch: X shape {batch_x.shape}, Y shape {batch_y.shape}")
            break # Just get one batch to show it works
