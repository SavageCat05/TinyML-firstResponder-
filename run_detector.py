import argparse
import logging
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components from our new architecture
from src.config import (
    MODEL_INPUT_SHAPE, SAMPLE_RATE, SLIDING_WINDOW_SECONDS,
    INTENT_CLASSES, CONFIDENCE_THRESHOLD, TEMPORAL_CONFIRMATIONS,
    EMERGENCY_ACTIONS
)
from src.model.architecture import EmergencyIntentModel
from src.features.feature_extraction import FeatureExtractor
# from src.engine.audio_stream import AudioStream (To be implemented)
# from src.engine.inference import TFLiteMicroInterpreter (To be implemented)

class EmergencyDetector:
    """
    Main class for the TinyML Emergency Intent Detection system.
    Orchestrates audio capture, feature extraction, and model inference.
    """
    def __init__(self):
        logger.info("Initializing EmergencyDetector...")
        self.model = EmergencyIntentModel()
        self.model.load_model() # Load the quantized TFLite model
        
        self.feature_extractor = FeatureExtractor()
        
        # self.audio_stream = AudioStream() # Placeholder
        # self.tflite_interpreter = TFLiteMicroInterpreter() # Placeholder

        self.temporal_confirmation_buffer = []
        self.last_prediction = "non_emergency_noise"
        self.consecutive_emergency_count = 0
        self.detection_cooldown_end_time = 0

    def _process_audio_chunk(self, audio_chunk):
        """
        Processes a single audio chunk: extracts features and runs inference.
        """
        # 1. Feature Extraction
        features = self.feature_extractor.extract_features(audio_chunk)
        if features is None:
            logger.warning("Failed to extract features from audio chunk.")
            return None

        model_input = self.feature_extractor.get_model_input(features)
        # Model expects a batch dimension: (1, 40, 50, 1)
        model_input = np.expand_dims(model_input, axis=0) 

        # 2. Model Inference
        # prediction_results = self.tflite_interpreter.predict(model_input) # Placeholder
        # For now, use the Keras model's predict method if TFLite model is loaded (self.model.predict)
        # This will be replaced by direct TFLiteMicroInterpreter call later.
        if self.model.interpreter is None:
             logger.error("TFLite model not loaded for inference.")
             return None
        
        prediction_results = self.model.predict(model_input)
        
        return prediction_results

    def _handle_prediction(self, prediction_results):
        """
        Handles the prediction results, applying temporal confirmation and actions.
        """
        predicted_class = prediction_results['predicted_class']
        confidence = prediction_results['confidence']
        
        current_time = time.time()

        if predicted_class != "non_emergency_noise" and confidence >= CONFIDENCE_THRESHOLD:
            logger.info(f"Potential emergency detected: {predicted_class} with confidence {confidence:.2f}")
            
            # Temporal confirmation logic
            if predicted_class == self.last_prediction:
                self.consecutive_emergency_count += 1
            else:
                self.consecutive_emergency_count = 1
                self.last_prediction = predicted_class
            
            if self.consecutive_emergency_count >= TEMPORAL_CONFIRMATIONS:
                if current_time > self.detection_cooldown_end_time:
                    logger.critical(f"CONFIRMED EMERGENCY: {predicted_class}!")
                    self._trigger_action(predicted_class)
                    self.consecutive_emergency_count = 0 # Reset after action
                    self.detection_cooldown_end_time = current_time + 10 # Cooldown period (e.g., 10 seconds)
                else:
                    logger.info(f"Emergency '{predicted_class}' confirmed but in cooldown period.")
            else:
                logger.info(f"Emergency '{predicted_class}' detected ({self.consecutive_emergency_count}/{TEMPORAL_CONFIRMATIONS})")
        else:
            if self.last_prediction != "non_emergency_noise" and self.consecutive_emergency_count > 0:
                logger.info(f"Non-emergency audio, resetting consecutive count for {self.last_prediction}.")
            self.last_prediction = "non_emergency_noise"
            self.consecutive_emergency_count = 0
            # logger.debug(f"Non-emergency audio: {predicted_class} (confidence {confidence:.2f})") # Too verbose for INFO level

    def _trigger_action(self, emergency_type):
        """
        Triggers the appropriate action based on the confirmed emergency type.
        """
        action_info = EMERGENCY_ACTIONS.get(emergency_type)
        if action_info:
            if "number" in action_info:
                logger.info(f"Action: Call {action_info['description']} at {action_info['number']}")
                # In a real system, integrate with communication modules
            elif "action" in action_info and action_info["action"] == "escalation":
                logger.info(f"Action: {action_info['description']}")
                # Implement further escalation logic (e.g., send push notification, activate siren)
            elif "action" in action_info and action_info["action"] == "ignore":
                logger.info(f"Action: {action_info['description']}")
        else:
            logger.warning(f"No specific action defined for emergency type: {emergency_type}")

    def run(self, duration_seconds=60):
        """
        Runs the emergency detector for a specified duration.
        In a real application, this would run indefinitely.
        """
        logger.info(f"Starting emergency detection loop for {duration_seconds} seconds.")
        start_time = time.time()

        # Placeholder for real-time audio capture
        # In a real system, self.audio_stream.start() would be called.
        # Here, we simulate audio chunks.
        
        audio_window_samples = int(SAMPLE_RATE * SLIDING_WINDOW_SECONDS)
        
        while (time.time() - start_time) < duration_seconds:
            # Simulate capturing an audio chunk
            # This should come from self.audio_stream.get_audio_chunk()
            simulated_audio_chunk = np.random.randn(audio_window_samples).astype(np.float32) * 0.1 # Small random noise
            
            # Simulate a strong emergency signal occasionally
            if np.random.rand() < 0.05: # 5% chance of simulating an emergency sound
                logger.info("Simulating an emergency sound...")
                # Add a strong sine wave burst or similar
                t = np.linspace(0, SLIDING_WINDOW_SECONDS, audio_window_samples, endpoint=False)
                emergency_sound = np.sin(2 * np.pi * 440 * t) * 0.5 # 440 Hz tone
                simulated_audio_chunk += emergency_sound

            prediction = self._process_audio_chunk(simulated_audio_chunk)
            if prediction:
                self._handle_prediction(prediction)
            
            # Simulate real-time delay
            time.sleep(SLIDING_WINDOW_SECONDS) # Process one window then wait for the next
        
        logger.info("Emergency detection loop finished.")

def main():
    parser = argparse.ArgumentParser(description="TinyML Emergency Intent Detection System")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds to run the detector (default: 60)")
    args = parser.parse_args()

    detector = EmergencyDetector()
    detector.run(duration_seconds=args.duration)

if __name__ == "__main__":
    main()
