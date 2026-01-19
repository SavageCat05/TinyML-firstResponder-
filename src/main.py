"""
Main application for TinyML Emergency Intent Detection System v1
Provides interface for training, testing, and running the emergency detector
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
import numpy as np

from .emergency_detector import EmergencyDetector
from .model import EmergencyIntentModel
from .config import (
    SAMPLE_RATE, SLIDING_WINDOW_SECONDS,
    NUM_CLASSES, INTENT_CLASSES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TinyMLEmergencyApp:
    """
    Main application class for the TinyML Emergency Detection System
    """

    def __init__(self):
        self.detector = None
        self.model = EmergencyIntentModel()

    def run_detector(self):
        """
        Run the emergency detection system
        """
        print("🚨 TinyML Emergency Intent Detection System v1")
        print("================================================")
        print(f"Target: Audio-based emergency detection on edge devices")
        print(f"Features: {NUM_CLASSES} intent classes, {SLIDING_WINDOW_SECONDS}s windows")
        print("================================================")

        try:
            # Initialize detector
            self.detector = EmergencyDetector()

            # Start detection
            self.detector.start()

            print("\n🎙️ Listening for emergency signals...")
            print("Press Ctrl+C to stop")
            print()

            # Keep running until interrupted
            while True:
                time.sleep(1)

                # Print status every 10 seconds
                if int(time.time()) % 10 == 0:
                    status = self.detector.get_status()
                    print(f"Status: Buffer {status['buffer_fill_percentage']:.1f}%, "
                          f"Latency {status['average_latency_ms']:.1f}ms, "
                          f"Inferences: {status['total_inferences']}")

        except KeyboardInterrupt:
            print("\n🛑 Stopping emergency detector...")
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
        finally:
            if self.detector:
                self.detector.stop()

    def train_model(self, epochs=50, batch_size=32):
        """
        Train the TinyML model (for development)
        """
        print("🔧 Training TinyML Emergency Intent Model")
        print("==========================================")

        try:
            # Build model
            model = self.model.build_model()

            # Generate dummy training data (in production, use real emergency audio data)
            print("Generating training data...")
            X_train, y_train = self._generate_training_data(num_samples=1000)

            # Train model
            print(f"Training on {len(X_train)} samples for {epochs} epochs...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )

            # Quantize and save model
            print("Quantizing model for edge deployment...")
            quantized_model = self.model.quantize_model()

            print(f"✅ Model trained and quantized successfully!")
            print(f"Model saved to: {self.model.model_path}")

            return history

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def benchmark_model(self):
        """
        Benchmark model performance and latency
        """
        print("📊 Benchmarking TinyML Model Performance")
        print("========================================")

        try:
            # Load model
            self.model.load_model()

            # Run latency benchmark
            latency_results = self.model.benchmark_latency(num_runs=100)

            print("Latency Results:")
            print(".2f")
            print(".2f")
            print(f"Meets target (<100ms): {'✅' if latency_results['meets_target'] else '❌'}")

            # Get model info
            model_info = self.model.get_model_info()
            print("\nModel Information:")
            print(f"Input shape: {model_info['input_shape']}")
            print(f"Output shape: {model_info['output_shape']}")
            print(f"Quantized: {'✅' if model_info.get('quantized') else '❌'}")

            # Test inference with dummy data
            dummy_input = np.random.rand(1, *model_info['input_shape'][1:]).astype(np.float32)
            result = self.model.predict(dummy_input)

            print("
Sample Inference:")
            print(f"Predicted class: {result['predicted_class']}")
            print(".3f")

            return latency_results

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            raise

    def _generate_training_data(self, num_samples=1000):
        """
        Generate synthetic training data for development
        In production, this would use real emergency audio datasets
        """
        from .config import MODEL_INPUT_SHAPE

        print("⚠️  Using synthetic training data (for development only)")
        print("   Production system requires real emergency audio datasets")

        # Generate random log-mel spectrograms
        X_train = np.random.rand(num_samples, *MODEL_INPUT_SHAPE).astype(np.float32)

        # Generate random one-hot labels
        y_train = np.zeros((num_samples, NUM_CLASSES))
        for i in range(num_samples):
            class_idx = np.random.randint(0, NUM_CLASSES)
            y_train[i, class_idx] = 1.0

        # Add some structure to make training more realistic
        # (This is just for development - real training needs actual audio data)
        for i in range(num_samples):
            # Add class-specific patterns
            class_idx = np.argmax(y_train[i])
            noise_level = 0.1

            if class_idx == 0:  # Police emergency - add high frequency noise
                X_train[i, :10, :, 0] += np.random.normal(0, noise_level, (10, MODEL_INPUT_SHAPE[1]))
            elif class_idx == 1:  # Medical emergency - add low frequency patterns
                X_train[i, -10:, :, 0] += np.random.normal(0, noise_level, (10, MODEL_INPUT_SHAPE[1]))
            # Add similar patterns for other classes...

        return X_train, y_train

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description='TinyML Emergency Intent Detection System v1'
    )
    parser.add_argument(
        'command',
        choices=['run', 'train', 'benchmark'],
        help='Command to execute'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size (default: 32)'
    )

    args = parser.parse_args()

    # Create application
    app = TinyMLEmergencyApp()

    if args.command == 'run':
        app.run_detector()
    elif args.command == 'train':
        app.train_model(epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == 'benchmark':
        app.benchmark_model()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()