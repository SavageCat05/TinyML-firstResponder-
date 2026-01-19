"""
Main Emergency Intent Detection System
Orchestrates audio capture, feature extraction, model inference, and emergency actions
"""

import time
import logging
import datetime
from collections import deque
import threading
from .audio_capture import AudioCapture
from .feature_extraction import FeatureExtractor
from .model import EmergencyIntentModel
from .config import (
    CONFIDENCE_THRESHOLD, TEMPORAL_CONFIRMATIONS,
    EMERGENCY_ACTIONS, LOG_FILE, MAX_LATENCY_MS
)

logger = logging.getLogger(__name__)

class EmergencyDetector:
    """
    Main orchestrator for TinyML Emergency Intent Detection System
    Implements the complete pipeline from audio to emergency action
    """

    def __init__(self):
        self.audio_capture = AudioCapture()
        self.feature_extractor = FeatureExtractor()
        self.model = EmergencyIntentModel()

        # Temporal smoothing for decision stability
        self.recent_predictions = deque(maxlen=TEMPORAL_CONFIRMATIONS)
        self.last_action_time = 0
        self.action_cooldown = 5.0  # 5 second cooldown between actions

        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.total_inferences = 0

        # Control flags
        self.is_running = False
        self.detection_thread = None

        # Emergency action handler (mock - would integrate with actual emergency services)
        self.emergency_handler = EmergencyActionHandler()

        logger.info("Emergency detector initialized")

    def start(self):
        """
        Start the emergency detection system
        """
        if self.is_running:
            logger.warning("Emergency detector already running")
            return

        try:
            # Load the model
            self.model.load_model()

            # Start audio capture
            self.audio_capture.start()

            # Start detection loop in background thread
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()

            logger.info("🚨 Emergency detection system started")

        except Exception as e:
            logger.error(f"Failed to start emergency detector: {e}")
            self.stop()
            raise

    def stop(self):
        """
        Stop the emergency detection system gracefully
        """
        if not self.is_running:
            return

        self.is_running = False

        # Stop audio capture
        self.audio_capture.stop()

        # Wait for detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

        logger.info("🛑 Emergency detection system stopped")

    def _detection_loop(self):
        """
        Main detection loop - runs in background thread
        """
        logger.info("Starting detection loop...")

        while self.is_running:
            try:
                start_time = time.time()

                # Get current audio window
                audio_window = self.audio_capture.get_audio_window()

                if audio_window is not None:
                    # Extract features
                    features = self.feature_extractor.extract_features(audio_window)

                    if features is not None:
                        # Prepare model input
                        model_input = self.feature_extractor.get_model_input(features)

                        if model_input is not None:
                            # Run inference
                            result = self.model.predict(model_input[np.newaxis, ...])

                            # Apply temporal smoothing
                            smoothed_result = self._apply_temporal_smoothing(result)

                            # Make decision and trigger action if needed
                            self._process_decision(smoothed_result)

                            # Record inference time for performance monitoring
                            inference_time = (time.time() - start_time) * 1000
                            self.inference_times.append(inference_time)
                            self.total_inferences += 1

                            # Log performance warning if exceeding latency target
                            if inference_time > MAX_LATENCY_MS:
                                logger.warning(f"High latency: {inference_time:.1f} ms")

                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(1.0)  # Longer delay on error

    def _apply_temporal_smoothing(self, current_result):
        """
        Apply temporal smoothing to reduce false positives
        Requires multiple consecutive confirmations
        """
        # Add current prediction to recent history
        self.recent_predictions.append(current_result)

        if len(self.recent_predictions) < TEMPORAL_CONFIRMATIONS:
            # Not enough history yet
            return None

        # Check if all recent predictions agree on the same class
        predicted_classes = [pred['predicted_class'] for pred in self.recent_predictions]

        if len(set(predicted_classes)) == 1:  # All predictions agree
            # Return the agreed-upon result with average confidence
            avg_confidence = np.mean([pred['confidence'] for pred in self.recent_predictions])
            agreed_result = self.recent_predictions[-1].copy()
            agreed_result['confidence'] = avg_confidence
            agreed_result['temporally_confirmed'] = True
            return agreed_result

        return None  # No temporal agreement yet

    def _process_decision(self, result):
        """
        Process the detection result and trigger appropriate action
        """
        if result is None:
            return

        predicted_class = result['predicted_class']
        confidence = result['confidence']
        temporally_confirmed = result.get('temporally_confirmed', False)

        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.debug(f"Low confidence: {predicted_class} ({confidence:.2f})")
            return

        # Check temporal confirmation
        if not temporally_confirmed:
            logger.debug(f"Temporal confirmation pending: {predicted_class}")
            return

        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            logger.debug("Action on cooldown")
            return

        # Trigger emergency action
        self._trigger_emergency_action(predicted_class, confidence, result)

        # Update last action time
        self.last_action_time = current_time

    def _trigger_emergency_action(self, intent_class, confidence, result):
        """
        Trigger the appropriate emergency action based on detected intent
        """
        if intent_class not in EMERGENCY_ACTIONS:
            logger.error(f"Unknown intent class: {intent_class}")
            return

        action_info = EMERGENCY_ACTIONS[intent_class]

        # Log the emergency detection
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{timestamp}] EMERGENCY DETECTED: {intent_class} "
            f"(confidence: {confidence:.2f}) - {action_info.get('description', 'Unknown')}"
        )

        logger.warning(log_entry)

        # Write to emergency log
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        # Print alert to console
        print(f"🚨🚨🚨 {log_entry}")

        # Trigger the actual emergency action
        self.emergency_handler.trigger_action(intent_class, action_info, confidence, result)

    def get_status(self):
        """
        Get current system status for monitoring
        """
        avg_latency = np.mean(list(self.inference_times)) if self.inference_times else 0
        buffer_fill = self.audio_capture.get_buffer_fill_percentage()

        return {
            "is_running": self.is_running,
            "buffer_fill_percentage": buffer_fill,
            "total_inferences": self.total_inferences,
            "average_latency_ms": avg_latency,
            "recent_predictions_count": len(self.recent_predictions),
            "last_action_time": self.last_action_time,
            "model_info": self.model.get_model_info()
        }


class EmergencyActionHandler:
    """
    Handles emergency actions (mock implementation)
    In production, this would integrate with actual emergency services
    """

    def __init__(self):
        self.action_log = []

    def trigger_action(self, intent_class, action_info, confidence, result):
        """
        Trigger the appropriate emergency action
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "number" in action_info:
            # Emergency phone call
            phone_number = action_info["number"]
            description = action_info["description"]

            print(f"📞 CALLING EMERGENCY: {phone_number} - {description}")
            print("⚠️  In production, this would initiate an actual emergency call!")

            # Mock emergency call
            self._mock_emergency_call(phone_number, description, confidence)

        elif action_info.get("action") == "escalation":
            # Escalation for general distress
            print("⚠️  GENERAL DISTRESS DETECTED - ESCALATION REQUIRED")
            print("📋 In production, this would trigger escalation protocols!")

            self._mock_escalation(confidence)

        elif action_info.get("action") == "ignore":
            # Non-emergency - just log
            print("ℹ️  Non-emergency audio detected - monitoring continues")

        # Log all actions
        action_record = {
            "timestamp": timestamp,
            "intent_class": intent_class,
            "confidence": confidence,
            "action": action_info
        }
        self.action_log.append(action_record)

    def _mock_emergency_call(self, number, description, confidence):
        """
        Mock emergency call (for development/testing)
        """
        print(f"   → Simulating call to {number} ({description})")
        print(f"   → Confidence level: {confidence:.2f}")
        print("   → Call would include location data and detected intent"
        # In production, this would:
        # 1. Get device location (GPS)
        # 2. Initiate phone call to emergency number
        # 3. Transmit location and intent data
        # 4. Record call for verification

    def _mock_escalation(self, confidence):
        """
        Mock escalation for general distress
        """
        print("   → Escalating to human operator")
        print(f"   → Confidence level: {confidence:.2f}")
        print("   → Would notify monitoring center for manual verification"
        # In production, this would:
        # 1. Alert human operators
        # 2. Provide audio context
        # 3. Enable manual intervention

    def get_action_history(self):
        """
        Get history of triggered actions
        """
        return self.action_log.copy()