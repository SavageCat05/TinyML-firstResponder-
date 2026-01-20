"""
Main Emergency Intent Detection System
Orchestrates audio capture, feature extraction, model inference, and emergency actions
"""

import time
import logging
import datetime
from collections import deque
import threading
import numpy as np
from .audio_capture import AudioCapture
from .feature_extraction import FeatureExtractor
from .model import EmergencyIntentModel
from .config import (
    CONFIDENCE_THRESHOLD, TEMPORAL_CONFIRMATIONS,
    EMERGENCY_ACTIONS, LOG_FILE, MAX_LATENCY_MS
)

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

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

        # Speech-to-text model
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            except Exception as e:
                pass

        # Temporal smoothing for decision stability
        self.recent_predictions = deque(maxlen=TEMPORAL_CONFIRMATIONS)
        self.last_action_time = 0
        self.action_cooldown = 5.0  # 5 second cooldown between actions

        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.total_inferences = 0
        self.last_latency_warning_time = 0

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

        except Exception as e:
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

    def _detection_loop(self):
        """
        Main detection loop - runs in background thread
        """
        print("\n[*] Listening for audio input...")
        print("[*] Press Ctrl+C to stop\n")

        while self.is_running:
            try:
                # Get current audio window
                audio_window = self.audio_capture.get_audio_window()

                if audio_window is not None:
                    # Try to get speech transcription
                    transcribed_text = None
                    if self.whisper_model:
                        try:
                            segments, info = self.whisper_model.transcribe(audio_window, language="en", beam_size=1)
                            transcribed_text = " ".join([segment.text for segment in segments]).strip()
                        except Exception:
                            pass
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(audio_window)

                    if features is not None:
                        model_input = self.feature_extractor.get_model_input(features)

                        if model_input is not None:
                            try:
                                result = self.model.predict(model_input[np.newaxis, ...])
                                smoothed_result = self._apply_temporal_smoothing(result)

                                # Only show input and decision
                                if smoothed_result is not None:
                                    detected_class = smoothed_result['predicted_class']
                                    confidence = smoothed_result['confidence']
                                    
                                    # Display: Input -> Output
                                    if transcribed_text:
                                        print(f"> Input: \"{transcribed_text}\"")
                                    print(f"  Result: {detected_class.upper()} ({confidence:.0%})\n")
                                    
                                    self._process_decision(smoothed_result, transcribed_text)

                            except Exception as e:
                                pass

                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

            except Exception:
                time.sleep(1.0)

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

    def _process_decision(self, result, transcribed_text=None):
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
            return

        # Check temporal confirmation
        if not temporally_confirmed:
            return

        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return

        # Trigger emergency action with transcribed text
        self._trigger_emergency_action(predicted_class, confidence, result, transcribed_text)

        # Update last action time
        self.last_action_time = current_time

    def _trigger_emergency_action(self, intent_class, confidence, result, transcribed_text=None):
        """
        Trigger the appropriate emergency action based on detected intent
        """
        if intent_class not in EMERGENCY_ACTIONS:
            return

        action_info = EMERGENCY_ACTIONS[intent_class]

        # Log the emergency detection
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_part = f" | Speech: '{transcribed_text}'" if transcribed_text else ""
        log_entry = (
            f"[{timestamp}] EMERGENCY: {intent_class} ({confidence:.2f}){text_part}"
        )

        # Write to emergency log
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        # Print only critical alerts
        if intent_class != "non_emergency_noise":
            print(f"\n*** ALERT: {intent_class.upper()} ({confidence:.0%}) ***\n")
            if transcribed_text:
                print(f"    Speech: \"{transcribed_text}\"\n")

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
        print("   → Call would include location data and detected intent")
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
        print("   → Would notify monitoring center for manual verification")
        # In production, this would:
        # 1. Alert human operators
        # 2. Provide audio context
        # 3. Enable manual intervention

    def get_action_history(self):
        """
        Get history of triggered actions
        """
        return self.action_log.copy()