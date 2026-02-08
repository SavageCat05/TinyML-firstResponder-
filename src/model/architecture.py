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
        self.model_path = os.path.join("models", "saved_models", "emergency_intent_model.tflite") # Updated path
        self.intent_classes = INTENT_CLASSES

        # Model will be loaded when needed
        self.model = None
        # Removed self.interpreter, self.input_details, self.output_details as they are now in TFLiteMicroInterpreter

        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)

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
        """
        if self.model is None:
            raise ValueError("Model must be built before quantization")

        logger.info("Applying int8 quantization...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset_gen:
            converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert to quantized TFLite model
        quantized_tflite_model = converter.convert()

        return quantized_tflite_model