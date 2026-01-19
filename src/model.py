"""
TinyML Model for Emergency Intent Detection
CNN architecture optimized for edge deployment with int8 quantization
"""

import tensorflow as tf
import numpy as np
import os
import logging
from pathlib import Path
from config import (
    MODEL_INPUT_SHAPE, NUM_CLASSES, MODEL_PATH,
    INTENT_CLASSES, MAX_MODEL_SIZE_KB
)

logger = logging.getLogger(__name__)

class EmergencyIntentModel:
    """
    TinyML CNN model for classifying emergency intent from audio features
    Designed for edge deployment with int8 quantization
    """

    def __init__(self):
        self.input_shape = MODEL_INPUT_SHAPE
        self.num_classes = NUM_CLASSES
        self.model_path = MODEL_PATH
        self.intent_classes = INTENT_CLASSES

        # Model will be loaded when needed
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)

    def build_model(self):
        """
        Build the TinyML CNN architecture
        Optimized for low latency and small model size
        """
        logger.info("Building TinyML CNN model...")

        inputs = tf.keras.Input(shape=self.input_shape, name='log_mel_input')

        # First convolutional block - small filters for efficiency
        x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Second convolutional block
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Third convolutional block - deeper features
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Global pooling to reduce parameters
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Dense layers - keep small for TinyML
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)  # Regularization for robustness

        # Output layer - softmax for probabilities
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='emergency_intent_cnn')

        # Compile with optimizer suitable for quantization
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        # Print model summary for verification
        model.summary()

        logger.info(f"Model built with {model.count_params()} parameters")
        return model

    def quantize_model(self, representative_data=None):
        """
        Apply post-training quantization to int8
        """
        if self.model is None:
            raise ValueError("Model must be built before quantization")

        logger.info("Applying int8 quantization...")

        # Create representative dataset for quantization calibration
        if representative_data is None:
            # Generate dummy representative data
            representative_data = self._create_representative_dataset()

        # Apply quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert to quantized TFLite model
        quantized_tflite_model = converter.convert()

        # Save quantized model
        with open(self.model_path, 'wb') as f:
            f.write(quantized_tflite_model)

        # Check model size
        model_size_kb = len(quantized_tflite_model) / 1024
        logger.info(f"Quantized model saved: {model_size_kb:.1f} KB")

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
        if self.interpreter is None:
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

    def get_model_info(self):
        """
        Get model information for monitoring
        """
        if self.interpreter is None:
            return {"status": "Model not loaded"}

        return {
            "model_path": self.model_path,
            "input_shape": self.input_details[0]['shape'].tolist(),
            "output_shape": self.output_details[0]['shape'].tolist(),
            "input_dtype": str(self.input_details[0]['dtype']),
            "output_dtype": str(self.output_details[0]['dtype']),
            "quantized": self.input_details[0]['dtype'] == np.int8
        }

    def benchmark_latency(self, num_runs=100):
        """
        Benchmark model latency
        """
        if self.interpreter is None:
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