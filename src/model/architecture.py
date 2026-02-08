"""
TinyML Model for Emergency Intent Detection
CNN architecture optimized for edge deployment with int8 quantization
"""

import tensorflow as tf
import numpy as np
import os
import logging
from pathlib import Path
from src.config import (
    MODEL_INPUT_SHAPE, NUM_CLASSES, MODEL_PATH,
    INTENT_CLASSES, MAX_MODEL_SIZE_KB
)
from src.model.layers import inverted_residual_block # Import the block

logger = logging.getLogger(__name__)

class EmergencyIntentModel:
    """
    TinyML CNN model for classifying emergency intent from audio features
    Designed for edge deployment with int8 quantization
    """

    def __init__(self):
        self.input_shape = MODEL_INPUT_SHAPE
        self.num_classes = NUM_CLASSES
        # Path for quantized TFLite model (edge deployment)
        self.model_path = os.path.join("models", "saved_models", "emergency_intent_model.tflite")
        # Path for Keras .h5 model (persistent weights for training/evaluation)
        self.h5_model_path = os.path.join("models", "saved_models", "emergency_intent_model.weights.h5")
        self.intent_classes = INTENT_CLASSES

        # Model will be loaded when needed
        self.model = None
        
        # TFLite interpreter for quantized model inference
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        # Ensure models directory exists
        Path(os.path.join("models", "saved_models")).mkdir(parents=True, exist_ok=True)

    def check_existing_model(self):
        """
        Check if a pre-trained model exists at the h5_model_path.
        Returns True if the file exists, False otherwise.
        """
        return os.path.exists(self.h5_model_path)

    def load_existing_model(self):
        """
        Load an existing model from the h5_model_path.
        Must build the architecture first, then load weights.
        Returns True if successful, False otherwise.
        """
        if not self.check_existing_model():
            logger.warning(f"No existing model found at {self.h5_model_path}")
            return False

        try:
            # Build the architecture first (required to load weights)
            if self.model is None:
                self.build_model()
            
            # Load the saved weights
            self.model.load_weights(self.h5_model_path)
            logger.info(f"Existing model found. Loading weights from {self.h5_model_path}...")
            print("Existing model found. Loading weights...")
            return True
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}")
            return False

    def save_model(self):
        """
        Save the current model weights to the h5_model_path.
        Called after training completes to persist the model for next run.
        """
        if self.model is None:
            logger.error("Cannot save model: No model has been built or trained.")
            return False

        try:
            self.model.save_weights(self.h5_model_path)
            logger.info(f"Model weights saved to {self.h5_model_path}")
            print(f"Model saved to {self.h5_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def build_model(self):
        """
        Build the TinyML v2 "MobileNet-style" architecture
        Features: Inverted Residuals, SE Blocks, Global Max Pooling
        """
        logger.info("Building TinyML v2 (MobileNet-style) model...")

        inputs = tf.keras.Input(shape=self.input_shape, name='log_mel_input')

        # Stem: Standard Conv to process input
        # Stride 2 to downsample time/freq dimensions early
        x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', 
                                 use_bias=False, kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)

        # Body: Inverted Residual Blocks from v1.1 spec
        
        # Block 1: Expansion 2x, 24 filters
        x = inverted_residual_block(x, filters=24, stride=1, expansion_ratio=2, use_se=False)
        
        # Block 2: Expansion 4x, 32 filters, SE
        x = inverted_residual_block(x, filters=32, stride=2, expansion_ratio=4, use_se=True)
        
        # Block 3: Expansion 4x, 64 filters, SE
        x = inverted_residual_block(x, filters=64, stride=1, expansion_ratio=4, use_se=True)

        # Head
        # No 1x1 Expansion in v1.1 spec

        # Global Max Pooling (Crucial for transient sounds like gunshots/screams)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

        # Dense head
        x = tf.keras.layers.Dense(64, activation=tf.keras.layers.ReLU(max_value=6))(x)

        # Dropout for regularization
        x = tf.keras.layers.Dropout(0.3)(x)

        # Output
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='emergency_intent_mobilenet_v2')

        # Compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        # Print model summary
        model.summary()
        logger.info(f"Model built with {model.count_params()} parameters")
        
        return model

    def quantize_model(self, representative_dataset_gen=None):
        """
        Apply post-training quantization to int8 and return the quantized TFLite model bytes.
        Also saves the quantized model to model_path.
        """
        if self.model is None:
            raise ValueError("Model must be built before quantization")

        logger.info("Applying int8 quantization...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset_gen:
            converter.representative_dataset = representative_dataset_gen
        else:
            converter.representative_dataset = self._create_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert to quantized TFLite model
        quantized_tflite_model = converter.convert()

        # Save quantized model
        Path(os.path.dirname(self.model_path)).mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            f.write(quantized_tflite_model)

        # Check model size
        model_size_kb = len(quantized_tflite_model) / 1024
        logger.info(f"Quantized model saved to {self.model_path}: {model_size_kb:.1f} KB")

        if model_size_kb > MAX_MODEL_SIZE_KB:
            logger.warning(f"Model size ({model_size_kb:.1f} KB) exceeds target ({MAX_MODEL_SIZE_KB} KB)")
        else:
            logger.info(f"Model size within target limits ✓")

        return quantized_tflite_model

    def _create_representative_dataset(self, num_samples=100):
        """
        Create representative dataset for quantization calibration
        """
        def representative_dataset():
            for _ in range(num_samples):
                # Generate representative log-mel spectrograms
                dummy_input = np.random.rand(1, *self.input_shape).astype(np.float32)
                yield [dummy_input]

        return representative_dataset

    def load_model(self):
        """
        Load the quantized TFLite model for inference
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load TFLite model
        with open(self.model_path, 'rb') as f:
            tflite_model = f.read()

        # Create interpreter
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info("TFLite model loaded successfully")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")

    def predict(self, input_data):
        """
        Run inference on input data
        Returns intent probabilities and predicted class
        """
        if not hasattr(self, 'interpreter') or self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load_model() first")

        # Prepare input data
        input_data = np.array(input_data, dtype=np.float32)

        # Quantize input if needed (for int8 models)
        if self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize output if needed
        if self.output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        # Convert to probabilities (softmax is applied in the model)
        probabilities = output_data[0]  # Remove batch dimension

        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.intent_classes[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

        return {
            'probabilities': probabilities,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_index': predicted_class_idx
        }

    def benchmark_latency(self, num_runs=100):
        """
        Benchmark model latency
        """
        if not hasattr(self, 'interpreter') or self.interpreter is None:
            raise RuntimeError("Model not loaded")

        import time

        latencies = []
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)

        for _ in range(num_runs):
            start_time = time.time()
            self.predict(dummy_input[np.newaxis, ...])
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        logger.info(f"Average latency: {avg_latency:.2f} ms")
        logger.info(f"Max latency: {max_latency:.2f} ms")

        return {
            'average_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'meets_target': avg_latency < 100  # 100ms target from spec
        }