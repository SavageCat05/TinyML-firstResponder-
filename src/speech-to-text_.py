import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from queue import Queue
import datetime
import signal
import sys

# ======================
# CONFIG
# ======================
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.5
OVERLAP_DURATION = 0.5     # <-- NEW
CHANNELS = 1
LOG_FILE = "distress_log.txt"

CONFIDENCE_THRESHOLD = -1.0
KEYWORDS = ["help", "emergency", "save me", "danger", "please help", "call 911", "call ambulance"]
STOP_WORDS = ["stop listening", "shutdown", "exit program"]
# ======================
# MODEL
# ======================
model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

# ======================
# AUDIO QUEUE
# ======================
audio_queue = Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# ======================
# LOGGING
# ======================
def log_text(text, alert=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = "ALERT" if alert else "INFO"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{tag}] {text}\n")

# ======================
# GRACEFUL EXIT
# ======================
def shutdown(sig, frame):
    print("\n🛑 Shutting down listener...")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ======================
# STREAM LOOP
# ======================
print("🎙️ Listening for distress calls...")

buffer = np.zeros((0,), dtype=np.float32)
samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)
overlap_samples = int(SAMPLE_RATE * OVERLAP_DURATION)

last_logged_text = ""

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
    callback=audio_callback
):
    while True:
        data = audio_queue.get()
        buffer = np.concatenate((buffer, data.flatten()))

        if len(buffer) >= samples_per_chunk:
            audio_chunk = buffer[:samples_per_chunk]

            # keep overlap
            buffer = buffer[samples_per_chunk - overlap_samples:]

            segments, _ = model.transcribe(
                audio_chunk,
                language="en",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300
                )
            )

            for segment in segments:
                text = segment.text.strip().lower()

                # confidence filter
                if segment.avg_logprob < CONFIDENCE_THRESHOLD:
                    continue

                # debounce duplicate text
                if not text or text == last_logged_text:
                    continue

                last_logged_text = text

                # keyword detection
                is_distress = any(k in text for k in KEYWORDS)
                end_loop = any(word in text for word in STOP_WORDS)
                if is_distress:
                    print("🚨 DISTRESS:", text)
                    log_text(text, alert=True)
                elif end_loop:
                    print("🛑 Voice command received. Stopping.")
                    log_text("Voice stop command received")
                    sys.exit(0)
                else:
                    print("📝", text)
                    log_text(text)
