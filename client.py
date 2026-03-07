# ── Requirements ───────────────────────────────────────────────────────────────
# Python 3.9+ (M1 Mac native arm64 recommended)
# pip install sounddevice torch torchaudio requests soundfile numpy
#
# ffmpeg is NOT required — WAV is used throughout
# Silero VAD model (~2MB) is auto-downloaded from torch hub on first run
#
# Tested on: Apple M1, macOS Ventura/Sonoma, Python 3.11 (arm64)
# VAD runs on CPU — no GPU/MPS needed; uses ~50MB RAM
# ───────────────────────────────────────────────────────────────────────────────

import collections
import io
import queue
import threading

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch

# ── Config — change API_URL to your ngrok URL or local IP ─────────────────────
API_URL     = "https://glumly-unpredatory-sima.ngrok-free.dev"   # e.g. https://abc123.ngrok-free.app
ENDPOINT    = "/transcribe"                            # or "/translate"
SAMPLE_RATE = 16000                                    # Hz — required by MERaLiON

# ── VAD tuning ─────────────────────────────────────────────────────────────────
CHUNK_SAMPLES      = 512    # Silero VAD window size (32 ms at 16 kHz) — do not change
SPEECH_THRESHOLD   = 0.5    # VAD confidence threshold (0–1); raise to reduce false positives
SILENCE_DURATION   = 0.8    # seconds of silence before utterance is considered done
PRE_SPEECH_PAD     = 0.3    # seconds of audio kept before speech starts (avoids clipping)

SILENCE_CHUNKS     = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SAMPLES)
PRE_SPEECH_CHUNKS  = int(PRE_SPEECH_PAD   * SAMPLE_RATE / CHUNK_SAMPLES)

# ── Load Silero VAD ────────────────────────────────────────────────────────────
print("Loading Silero VAD model...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
vad_model.eval()
print("✓ VAD model ready\n")

# ── Shared audio queue (callback → main thread) ────────────────────────────────
audio_queue: queue.Queue = queue.Queue()


def audio_callback(indata: np.ndarray, frames: int, time, status):
    """sounddevice callback — runs in a separate thread."""
    if status:
        print(f"[sounddevice] {status}")
    # Queue a flat float32 copy of the incoming chunk
    audio_queue.put(indata[:, 0].copy().astype(np.float32))


def send_to_api(audio_array: np.ndarray):
    """Convert audio to WAV bytes and POST to the transcription API."""
    buf = io.BytesIO()
    sf.write(buf, audio_array, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)

    try:
        response = requests.post(
            f"{API_URL}{ENDPOINT}",
            files={"file": ("speech.wav", buf, "audio/wav")},
            timeout=120,  # MERaLiON can be slow — give it time
        )
        if response.ok:
            data = response.json()
            # /transcribe returns {"transcription": "..."}
            # /translate  returns {"result": "..."}
            text = data.get("transcription") or data.get("result", "")
            print(f"\n📝  {text}\n")
        else:
            print(f"\n[API error {response.status_code}]: {response.text}\n")
    except requests.exceptions.ConnectionError:
        print(f"\n[Connection failed] Is the server at {API_URL} reachable?\n")
    except Exception as exc:
        print(f"\n[Request error]: {exc}\n")


def process_audio():
    """
    Main VAD loop.
    - Accumulates audio chunks from the queue
    - Runs Silero VAD on each 512-sample window
    - Fires off a transcription request in a background thread when an
      utterance ends (speech followed by SILENCE_DURATION of silence)
    """
    # Ring buffer to capture audio just before speech starts
    pre_speech_buf: collections.deque = collections.deque(maxlen=PRE_SPEECH_CHUNKS)

    speech_chunks: list  = []
    in_speech: bool      = False
    silence_count: int   = 0

    print("🎙  Listening — speak now. Press Ctrl+C to stop.\n")

    while True:
        chunk = audio_queue.get()  # blocks until a chunk is available

        # Run VAD on this 512-sample window
        with torch.no_grad():
            prob = vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()

        is_speech = prob >= SPEECH_THRESHOLD

        if is_speech:
            if not in_speech:
                # Speech just started — prepend the pre-speech padding
                speech_chunks = list(pre_speech_buf)
                in_speech = True
                print("▶  Speech detected...", end="\r")
            speech_chunks.append(chunk)
            silence_count = 0

        else:  # silence
            if in_speech:
                speech_chunks.append(chunk)
                silence_count += 1

                if silence_count >= SILENCE_CHUNKS:
                    # Enough silence — utterance has ended
                    utterance = np.concatenate(speech_chunks)
                    duration  = len(utterance) / SAMPLE_RATE
                    print(f"⏹  Utterance ended ({duration:.1f}s) — sending to API...")

                    # Send in a daemon thread so we don't block VAD
                    threading.Thread(
                        target=send_to_api,
                        args=(utterance,),
                        daemon=True,
                    ).start()

                    # Reset state
                    speech_chunks = []
                    in_speech     = False
                    silence_count = 0
                    pre_speech_buf.clear()
            else:
                # No speech — keep filling the pre-speech ring buffer
                pre_speech_buf.append(chunk)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Connecting to API at: {API_URL}")
    try:
        r = requests.get(API_URL, timeout=5)
        print(f"✓ API reachable — {r.json()}\n")
    except Exception:
        print("⚠  Could not reach API. Check API_URL and ensure the server is running.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        ):
            process_audio()
    except KeyboardInterrupt:
        print("\nStopped.")
    except sd.PortAudioError as e:
        print(f"\n[Microphone error]: {e}")
        print("Run `python -m sounddevice` to list available devices.")
