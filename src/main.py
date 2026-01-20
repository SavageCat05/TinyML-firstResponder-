"""
Main application for TinyML Emergency Intent Detection System v1
Provides interface for training, testing, and running the emergency detector
"""

import argparse
import logging
import sys
import time
import numpy as np

from .emergency_detector import EmergencyDetector
from model import EmergencyIntentModel
from config import (
    SAMPLE_RATE,
    SLIDING_WINDOW_SECONDS,
    NUM_CLASSES,
    INTENT_CLASSES,
    MODEL_INPUT_SHAPE,
)

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TinyMLEmergencyApp:
    """
    Main application class for the TinyML Emergency Detection System
    """

    def __init__(self):
        self.detector = None
        self.model = EmergencyIntentModel()

    # -------------------------------------------------------------------
    # Run live detector
    # -------------------------------------------------------------------
    def run_detector(self):
        try:
            self.detector = EmergencyDetector()
            self.detector.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n[*] Stopping...")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
        finally:
            if self.detector:
                self.detector.stop()

    # -------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------
    def train_model(self, epochs=50, batch_size=32):
        print("🔧 Training TinyML Emergency Intent Model")
        print("========================================")

        try:
            model = self.model.build_model()

            print("Generating synthetic training data...")
            X_train, y_train = self._generate_training_data(num_samples=1000)

            print(
                f"Training on {len(X_train)} samples "
                f"for {epochs} epochs..."
            )
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
            )

            print("Quantizing model for edge deployment...")
            self.model.quantize_model()

            print("✅ Model trained and quantized successfully!")
            print(f"Model saved to: {self.model.model_path}")

            return history

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    # -------------------------------------------------------------------
    # Benchmark model
    # -------------------------------------------------------------------
    def benchmark_model(self):
        print("📊 Benchmarking TinyML Model Performance")
        print("========================================")

        try:
            self.model.load_model()

            latency_results = self.model.benchmark_latency(num_runs=100)

            print("\nLatency Results:")
            print(
                f"Average latency: "
                f"{latency_results['average_latency_ms']:.2f} ms"
            )
            print(
                f"Max latency: "
                f"{latency_results['max_latency_ms']:.2f} ms"
            )
            print(
                f"Meets target (<100ms): "
                f"{'✅' if latency_results['meets_target'] else '❌'}"
            )

            model_info = self.model.get_model_info()
            print("\nModel Information:")
            print(f"Input shape: {model_info['input_shape']}")
            print(f"Output shape: {model_info['output_shape']}")
            print(
                f"Quantized: "
                f"{'✅' if model_info.get('quantized') else '❌'}"
            )

            dummy_input = np.random.rand(
                1, *model_info["input_shape"][1:]
            ).astype(np.float32)

            result = self.model.predict(dummy_input)

            print("\nSample Inference:")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")

            return latency_results

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            raise

    # -------------------------------------------------------------------
    # Synthetic data generator
    # -------------------------------------------------------------------
    def _generate_training_data(self, num_samples=1000):
        print("⚠️  Using synthetic training data (development only)")
        print("   Real deployment requires real emergency audio data")

        X_train = np.random.rand(
            num_samples, *MODEL_INPUT_SHAPE
        ).astype(np.float32)

        y_train = np.zeros((num_samples, NUM_CLASSES))
        for i in range(num_samples):
            class_idx = np.random.randint(0, NUM_CLASSES)
            y_train[i, class_idx] = 1.0

        for i in range(num_samples):
            class_idx = np.argmax(y_train[i])
            noise_level = 0.1

            if class_idx == 0:
                X_train[i, :10, :, 0] += np.random.normal(
                    0, noise_level, (10, MODEL_INPUT_SHAPE[1])
                )
            elif class_idx == 1:
                X_train[i, -10:, :, 0] += np.random.normal(
                    0, noise_level, (10, MODEL_INPUT_SHAPE[1])
                )

        return X_train, y_train


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TinyML Emergency Intent Detection System v1"
    )
    parser.add_argument(
        "command",
        choices=["run", "train", "benchmark"],
        help="Command to execute",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )

    args = parser.parse_args()

    app = TinyMLEmergencyApp()

    if args.command == "run":
        app.run_detector()
    elif args.command == "train":
        app.train_model(args.epochs, args.batch_size)
    elif args.command == "benchmark":
        app.benchmark_model()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
