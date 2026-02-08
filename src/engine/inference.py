import tensorflow as tf
import numpy as np
import logging
import os

from src.config import INTENT_CLASSES

logger = logging.getLogger(__name__)

class TFLiteMicroInterpreter:
    """
    Orchestrates TFLite Micro model loading and inference.
    Handles input/output quantization/dequantization.
    """
    def __init__(self, model_path, intent_classes=INTENT_CLASSES):
        self.model_path = model_path
        self.intent_classes = intent_classes
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_quantized_input = False
        self.is_quantized_output = False

    def load_model(self):
        """
        Load the quantized TFLite model for inference.
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

        # Check quantization types
        self.is_quantized_input = self.input_details[0]['dtype'] == np.int8
        self.is_quantized_output = self.output_details[0]['dtype'] == np.int8

        logger.info("TFLite model loaded successfully")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")
        logger.info(f"Input dtype: {self.input_details[0]['dtype']}")
        logger.info(f"Output dtype: {self.output_details[0]['dtype']}")

    def predict(self, input_data):
        """
        Run inference on input data.
        Returns intent probabilities and predicted class.
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare input data
        input_data = np.array(input_data, dtype=np.float32)

        # Quantize input if needed (for int8 models)
        if self.is_quantized_input:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        else:
            input_data = input_data.astype(self.input_details[0]['dtype'])

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize output if needed
        if self.is_quantized_output:
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
        Get model information for monitoring.
        """
        if self.interpreter is None:
            return {"status": "Model not loaded"}

        return {
            "model_path": self.model_path,
            "input_shape": self.input_details[0]['shape'].tolist(),
            "output_shape": self.output_details[0]['shape'].tolist(),
            "input_dtype": str(self.input_details[0]['dtype']),
            "output_dtype": str(self.output_details[0]['dtype']),
            "quantized": self.is_quantized_input # Assuming input quantization implies overall model quantization
        }

    def benchmark_latency(self, num_runs=100):
        """
        Benchmark model latency.
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded")

        import time

        latencies = []
        # Get input shape from interpreter details
        input_shape = self.input_details[0]['shape']
        # Remove batch dimension if present (e.g., (1, 40, 50, 1) -> (40, 50, 1))
        if input_shape[0] == 1:
            dummy_input = np.random.rand(*input_shape[1:]).astype(np.float32)
        else:
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Add batch dimension for prediction
        dummy_input = np.expand_dims(dummy_input, axis=0)

        for _ in range(num_runs):
            start_time = time.time()
            self.predict(dummy_input)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        logger.info(f"Average latency: {avg_latency:.2f} ms")
        logger.info(f"Max latency: {max_latency:.2f} ms")

        # Assuming 100ms is the target for latency
        return {
            'average_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'meets_target': avg_latency < 100  # 100ms target from spec
        }