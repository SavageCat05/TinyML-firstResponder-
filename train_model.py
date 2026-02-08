#!/usr/bin/env python3
"""
Model training and optimization script for TinyML Emergency Detection
Prepares the quantized model for edge deployment

Implements Persistent Model Check:
- At startup, checks for existing model at models/saved_models/emergency_intent_model.h5
- If exists: Loads weights and proceeds to evaluation/inference
- If not: Trains the model and saves it for next run
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.model.architecture import EmergencyIntentModel
from src.config import MODEL_INPUT_SHAPE, NUM_CLASSES

def create_training_data(num_samples=5000):
    """
    Create synthetic training data for development
    In production, replace with real emergency audio data
    """
    print(f"Generating {num_samples} synthetic training samples...")

    # Create diverse log-mel spectrogram patterns
    X_train = []
    y_train = []

    for i in range(num_samples):
        # Generate base spectrogram
        spectrogram = np.random.rand(*MODEL_INPUT_SHAPE).astype(np.float32)

        # Add class-specific patterns to simulate real audio features
        class_idx = i % NUM_CLASSES

        if class_idx == 0:  # Police - high energy, broad frequency
            spectrogram[:, :20, 0] += np.random.normal(0.3, 0.1, (MODEL_INPUT_SHAPE[0], 20))
            spectrogram[10:30, :, 0] += np.random.normal(0.2, 0.1, (20, MODEL_INPUT_SHAPE[1]))

        elif class_idx == 1:  # Medical - mid-range energy, some periodicity
            spectrogram[15:35, 10:30, 0] += np.random.normal(0.4, 0.15, (20, 20))
            # Add some vertical stripes for periodicity
            for j in range(0, MODEL_INPUT_SHAPE[1], 5):
                spectrogram[:, j:j+2, 0] += np.random.normal(0.1, 0.05, (MODEL_INPUT_SHAPE[0], 2))

        elif class_idx == 2:  # Fire - high frequency, crackling pattern
            spectrogram[:15, :, 0] += np.random.normal(0.5, 0.2, (15, MODEL_INPUT_SHAPE[1]))
            # Add noise-like pattern
            noise = np.random.normal(0, 0.3, MODEL_INPUT_SHAPE)
            spectrogram += noise

        elif class_idx == 3:  # Women Safety - urgent, high pitch
            spectrogram[:25, 25:, 0] += np.random.normal(0.4, 0.1, (25, MODEL_INPUT_SHAPE[1]-25))
            # Add rapid frequency changes
            for t in range(0, MODEL_INPUT_SHAPE[0], 3):
                freq_shift = np.random.normal(0, 5)
                spectrogram[t:t+3, :, 0] += freq_shift

        elif class_idx == 4:  # General Distress - varied, emotional
            # Mix of different patterns
            spectrogram[5:25, :, 0] += np.random.normal(0.2, 0.1, (20, MODEL_INPUT_SHAPE[1]))
            spectrogram[:, 15:35, 0] += np.random.normal(0.15, 0.08, (MODEL_INPUT_SHAPE[0], 20))

        # Class 5 (Non-emergency) - low energy, background noise
        # Leave as mostly random

        # Normalize
        spectrogram = np.clip(spectrogram, 0, 1)

        X_train.append(spectrogram)

        # One-hot label
        label = np.zeros(NUM_CLASSES)
        label[class_idx] = 1.0
        y_train.append(label)

    return np.array(X_train), np.array(y_train)

def train_and_quantize():
    """
    Complete training and quantization pipeline with persistent model check.
    
    - Checks for existing model at startup
    - If exists: Loads weights and skips training
    - If not: Trains and saves the model
    """
    print("TinyML Emergency Model Training & Quantization")
    print("=" * 55)

    # Initialize model
    model = EmergencyIntentModel()


    # === PERSISTENT MODEL CHECK ===
    # Check if existing model weights exist (cross-platform path)
    h5_model_path = model.h5_model_path
    
    if os.path.exists(h5_model_path):
        print("\n" + "=" * 55)
        print("Existing model found. Loading weights...")
        print("=" * 55)
        
        # Build architecture first (required to load weights)
        tf_model = model.build_model()
        
        # Load existing weights
        tf_model.load_weights(h5_model_path)
        print(f"Model weights loaded from: {h5_model_path}")
        
        # Skip training, proceed to evaluation
        print("\nSkipping training - using existing model.")
        print("To retrain, delete the existing model file or use --force-retrain flag.")
        
    else:
        print("\n" + "=" * 55)
        print("No existing model found. Starting training...")
        print("=" * 55)
        
        # Create training data
        X_train, y_train = create_training_data(num_samples=5000)

        print(f"Training data shape: {X_train.shape}")
        print(f"Labels shape: {y_train.shape}")

        # Build model
        tf_model = model.build_model()

        # Train model
        print("\nTraining model...")
        history = tf_model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Save model weights after training completes
        print(f"\nSaving model weights to: {h5_model_path}")
        # Ensure directory exists (cross-platform)
        Path(os.path.dirname(h5_model_path)).mkdir(parents=True, exist_ok=True)
        tf_model.save_weights(h5_model_path)
        print(f"Model saved to {h5_model_path}")

    # === EVALUATION (runs for both loaded and trained models) ===
    print("\nEvaluating model...")
    # Generate some test data for evaluation
    X_test, y_test = create_training_data(num_samples=500)
    loss, accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Quantize for edge deployment
    print("\nQuantizing model for edge deployment...")
    quantized_model = model.quantize_model()

    # Verify quantized model
    print("\nVerifying quantized model...")
    model.load_model()

    # Test inference (use test data, not X_train which may not exist)
    test_input = X_test[:5]  # Test with 5 samples
    print("Testing inference on 5 samples...")

    for i, sample in enumerate(test_input):
        result = model.predict(sample[np.newaxis, ...])
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        print(f"Sample {i+1}: {predicted_class} (confidence: {confidence:.3f})")

    # Benchmark latency
    print("\nBenchmarking latency...")
    latency_results = model.benchmark_latency(num_runs=50)

    print("\nTraining Complete!")
    print("=" * 30)
    print(f"Model saved: {model.model_path}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Target met: {'YES' if latency_results['meets_target'] else 'NO'} (<100ms)")
    print(f"Model size: {os.path.getsize(model.model_path)/1024:.1f} KB")
    print(f"Size target met: {'YES' if os.path.getsize(model.model_path) < 500*1024 else 'NO'} (<500KB)")

    return model

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    try:
        trained_model = train_and_quantize()
        print("\nTinyML Emergency Model ready for deployment!")
        print("   Run 'python run_emergency_detector.py benchmark' to test performance")
        print("   Run 'python run_emergency_detector.py run' to start detection")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)