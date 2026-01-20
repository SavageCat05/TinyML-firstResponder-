"""
Audio capture module for TinyML Emergency Intent Detection
Implements thread-safe audio queue and sliding buffer as per v1 specifications
"""

import numpy as np
import sounddevice as sd
import threading
import time
from queue import Queue
from collections import deque
import logging
from .config import (
    SAMPLE_RATE, CHANNELS, FRAME_DURATION_MS,
    SLIDING_WINDOW_SECONDS, HOP_LENGTH
)

logger = logging.getLogger(__name__)

class AudioCapture:
    """
    Handles continuous audio capture with thread-safe queue and sliding buffer
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.frame_duration_ms = FRAME_DURATION_MS

        # Calculate buffer sizes
        self.frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.window_samples = int(self.sample_rate * SLIDING_WINDOW_SECONDS)

        # Thread-safe queue for decoupling capture and processing
        self.audio_queue = Queue(maxsize=100)  # Prevent memory overflow

        # Sliding buffer for temporal context
        self.sliding_buffer = deque(maxlen=self.window_samples)

        # Control flags
        self.is_running = False
        self.stream = None
        self.capture_thread = None

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for audio stream - runs in separate thread
        """
        if status:
            pass  # Silently handle status messages

        # Convert to mono if needed and flatten
        if self.channels == 1:
            audio_data = indata.flatten()
        else:
            audio_data = np.mean(indata, axis=1)  # Convert to mono

        # Put audio frame in thread-safe queue
        try:
            self.audio_queue.put(audio_data, timeout=0.1)  # Non-blocking with timeout
        except:
            pass  # Silent fail on queue full

    def _capture_loop(self):
        """
        Main capture loop - runs in background thread
        """
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=self.frame_samples
            ) as stream:
                self.stream = stream
                while self.is_running:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting

        except Exception as e:
            logger.error(f"Audio capture error: {e}")
        finally:
            logger.info("🛑 Audio capture stopped")

    def start(self):
        """
        Start audio capture in background thread
        """
        if self.is_running:
            logger.warning("Audio capture already running")
            return

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Audio capture started successfully")

    def stop(self):
        """
        Stop audio capture gracefully
        """
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            self.stream.close()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        logger.info("Audio capture stopped")

    def get_audio_window(self):
        """
        Get the current sliding window of audio data
        Returns None if insufficient data
        """
        # Process any queued audio frames
        while not self.audio_queue.empty():
            try:
                frame = self.audio_queue.get_nowait()
                # Add frame to sliding buffer
                for sample in frame:
                    self.sliding_buffer.append(sample)
            except:
                break

        # Return buffer as numpy array if we have enough data
        if len(self.sliding_buffer) >= self.window_samples:
            return np.array(list(self.sliding_buffer))
        else:
            return None

    def get_buffer_fill_percentage(self):
        """
        Get current buffer fill percentage (for monitoring)
        """
        return (len(self.sliding_buffer) / self.window_samples) * 100

    def clear_buffer(self):
        """
        Clear the sliding buffer (useful for reset)
        """
        self.sliding_buffer.clear()