import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import os
import json # Added import

from src.model.architecture import EmergencyIntentModel
from src.feature_extraction import FeatureExtractor
from src.config import (
    SAMPLE_RATE, SLIDING_WINDOW_SECONDS, MODEL_INPUT_SHAPE,
    NUM_CLASSES, MODEL_PATH, MAX_MODEL_SIZE_KB, INTENT_CLASSES
)
from src.engine.inference import TFLiteMicroInterpreter # Added import

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles training, evaluation, and quantization of the TinyML Emergency Intent Model.
    """
    def __init__(self, model_input_shape=MODEL_INPUT_SHAPE, num_classes=NUM_CLASSES):
        self.model_input_shape = model_input_shape
        self.num_classes = num_classes
        self.emergency_model = EmergencyIntentModel()
        self.emergency_model.build_model() # Build the model architecture

        # Feature extractor for data preprocessing
        self.feature_extractor = FeatureExtractor()

        # Path for saving models
        self.saved_model_dir = "models/saved_models"
        self.metadata_dir = "models/metadata"
        Path(self.saved_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metadata_dir).mkdir(parents=True, exist_ok=True)

    def _save_metadata(self):
        """
        Saves model metadata, such as class labels, to the metadata directory.
        """
        metadata_path = Path(self.metadata_dir) / "intent_classes.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(INTENT_CLASSES, f, indent=4)
            logger.info(f"Class labels metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")


    def _load_data(self):
        """
        Placeholder for loading and preparing dataset.
        This will involve src/data module for actual data generation/loading.
        For now, returns dummy data.
        """
        logger.info("Loading dummy dataset for training. Replace with actual data loading.")
        # Dummy data: 100 samples, matching model input shape
        num_samples = 100
        x_train = np.random.rand(num_samples, *self.model_input_shape).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, self.num_classes, num_samples), num_classes=self.num_classes)
        
        x_val = np.random.rand(num_samples // 10, *self.model_input_shape).astype(np.float32)
        y_val = tf.keras.utils.to_categorical(np.random.randint(0, self.num_classes, num_samples // 10), num_classes=self.num_classes)

        return x_train, y_train, x_val, y_val

    def _apply_noise_augmentation(self, x_data, noise_level=0.1):
        """
        Applies noise augmentation to the input data.
        This is a placeholder for more sophisticated noise mixing.
        """
        logger.info(f"Applying dummy noise augmentation with level {noise_level}")
        noise = np.random.normal(0, noise_level, x_data.shape).astype(np.float32)
        return x_data + noise

    def train(self, epochs=10, batch_size=32):
        """
        Trains the emergency intent model.
        """
        logger.info("Starting model training...")

        x_train, y_train, x_val, y_val = self._load_data()
        
        # Apply noise augmentation (as required by spec)
        x_train = self._apply_noise_augmentation(x_train)

        self.emergency_model.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=1
        )
        logger.info("Model training finished.")
        self._evaluate_model(x_val, y_val)

    def _evaluate_model(self, x_test, y_test):
        """
        Evaluates the trained model.
        """
        logger.info("Evaluating model...")
        loss, accuracy = self.emergency_model.model.evaluate(x_test, y_test, verbose=0)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def _representative_dataset_gen(self, num_calibration_samples=100):
        """
        Generates a representative dataset for Post-Training Quantization (PTQ).
        This method should yield preprocessed input data in the correct format.
        """
        logger.info(f"Generating representative dataset with {num_calibration_samples} samples.")
        for _ in range(num_calibration_samples):
            # For real data, this would involve loading raw audio,
            # applying feature extraction, and then yielding the result.
            # Here, we generate random data matching the input shape.
            dummy_audio = np.random.rand(int(SAMPLE_RATE * SLIDING_WINDOW_SECONDS)).astype(np.float32)
            processed_input = self.feature_extractor.extract_features(dummy_audio)
            
            if processed_input is not None:
                # TFLiteConverter expects a list of numpy arrays
                yield [processed_input]

    def quantize_and_save_model(self):
        """
        Applies post-training quantization and saves the .tflite model.
        Verifies model size against the target.
        """
        logger.info("Starting quantization process...")
        quantized_tflite_model_bytes = self.emergency_model.quantize_model(
            representative_dataset_gen=self._representative_dataset_gen(num_calibration_samples=100)
        )

        # Save quantized model
        model_save_path = self.emergency_model.model_path # Use the path from EmergencyIntentModel
        with open(model_save_path, 'wb') as f:
            f.write(quantized_tflite_model_bytes)

        # Verify file size
        model_size_bytes = len(quantized_tflite_model_bytes)
        model_size_kb = model_size_bytes / 1024

        if model_size_kb > MAX_MODEL_SIZE_KB:
            logger.error(f"Quantized model size ({model_size_kb:.2f} KB) exceeds target {MAX_MODEL_SIZE_KB} KB.")
            return False
        else:
            logger.info(f"Quantized model size ({model_size_kb:.2f} KB) is within target {MAX_MODEL_SIZE_KB} KB. ✓")
            return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    trainer = ModelTrainer()
    trainer.train(epochs=5)
    if trainer.quantize_and_save_model(): # Only save metadata if quantization is successful
        trainer._save_metadata()