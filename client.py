# ── Requirements ───────────────────────────────────────────────────────────────
# pip install sounddevice soundfile numpy requests websockets torch torchaudio
#
# Flow:
#   1. Press ENTER  → records ALERT_DURATION_S seconds of audio
#   2. POSTs audio  → gateway transcribes + notifies responder
#   3. Opens WS     → waits for responder to accept
#   4. Live call    → streams mic to gateway (transcribed every 5s)
#                     plays responder audio through speakers
# ───────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import uuid

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import websockets

# ── Config ─────────────────────────────────────────────────────────────────────
GATEWAY_URL     = "https://unmetropolitan-lupita-urban.ngrok-free.dev"
API_KEY         = "hamster67"
SAMPLE_RATE     = 16000
ALERT_DURATION  = 10       # seconds to record for the initial alert
CHUNK_SAMPLES       = 512    # audio chunk size sent over WebSocket (32 ms)
SILENCE_THRESHOLD   = 0.02   # RMS below this = silence, don't send (0.0 to disable)


# ── Audio helpers ──────────────────────────────────────────────────────────────

def record_alert(duration_s: int = ALERT_DURATION) -> np.ndarray:
    """Block and record `duration_s` seconds from the microphone."""
    print(f"🔴  Recording {duration_s}s — speak now...")
    audio = sd.rec(
        int(duration_s * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    for remaining in range(duration_s, 0, -1):
        print(f"    {remaining}s remaining...", end="\r")
        sd.sleep(1000)
    sd.wait()
    print("✅  Recording complete.                    ")
    return audio[:, 0]


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── Phase 1 + 2: POST alert to gateway ────────────────────────────────────────

def post_alert(audio: np.ndarray, incident_id: str) -> dict:
    """Send alert audio to gateway, return the response dict."""
    wav = to_wav_bytes(audio)
    print("📤  Sending alert to gateway...")
    resp = requests.post(
        f"{GATEWAY_URL}/v1/emergency/transcribe",
        headers={"X-API-Key": API_KEY},
        files={"file": ("alert.wav", wav, "audio/wav")},
        data={"incident_id": incident_id, "priority": "high"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Phase 3: Live WebSocket call ───────────────────────────────────────────────

async def live_call(incident_id: str) -> None:
    """Connect to gateway WS, stream mic audio, receive transcripts."""
    ws_url = (
        GATEWAY_URL
        .replace("https://", "wss://")
        .replace("http://", "ws://")
    ) + f"/ws/emergency/{incident_id}"

    print(f"🔌  Connecting to live call (incident {incident_id[:8]})...")

    async with websockets.connect(
        ws_url,
        additional_headers={"X-API-Key": API_KEY},
        ping_interval=20,
    ) as ws:

        # ── Receive loop (runs concurrently) ──────────────────────────────────
        async def receive_loop():
            try:
                async for message in ws:
                    if isinstance(message, str):
                        try:
                            msg = json.loads(message)
                        except Exception:
                            continue  # ignore malformed frames — don't crash the task
                        mtype = msg.get("type", "")

                        if mtype == "waiting":
                            print(f"⏳  {msg['message']}")

                        elif mtype == "accepted":
                            print(f"\n✅  {msg['message']}")
                            print("🎙   Speak freely. Press Ctrl+C to hang up.\n")

                        elif mtype == "transcript":
                            print(f"📝  [You said]: {msg['text']}")

                        elif mtype == "responder_message":
                            print(f"💬  [Responder]: {msg['text']}")

                        elif mtype == "responder_disconnected":
                            print(f"\n🔴  {msg['message']}")

                        elif mtype == "timeout":
                            print(f"\n⚠️   {msg['message']}")
                            return

                    elif isinstance(message, bytes) and len(message) > 0:
                        # Responder audio — play through speakers
                        try:
                            audio_chunk = np.frombuffer(message, dtype=np.float32)
                            sd.play(audio_chunk, samplerate=SAMPLE_RATE, blocking=False)
                        except Exception:
                            pass
            except websockets.ConnectionClosed:
                pass

        # ── Mic send loop ─────────────────────────────────────────────────────
        audio_queue: asyncio.Queue = asyncio.Queue()

        def mic_callback(indata: np.ndarray, frames: int, time, status):
            chunk = indata[:, 0].copy().astype(np.float32)
            # Gate: discard silent chunks to avoid padding ASR buffer with silence
            if SILENCE_THRESHOLD > 0.0 and float(np.sqrt(np.mean(chunk ** 2))) < SILENCE_THRESHOLD:
                return
            audio_queue.put_nowait(chunk)

        recv_task = asyncio.create_task(receive_loop())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=mic_callback,
        ):
            try:
                while not recv_task.done():
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        await ws.send(chunk.tobytes())
                    except asyncio.TimeoutError:
                        continue  # silence gap — keep looping, don't close WS
            except asyncio.CancelledError:
                pass
            except KeyboardInterrupt:
                print("\n📵  Hanging up...")
            finally:
                recv_task.cancel()
                await ws.close()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print(f"\n🚨  PAB Emergency Client")
    print(f"    Gateway : {GATEWAY_URL}")
    print(f"\nPress ENTER to trigger an emergency alert (records {ALERT_DURATION}s).")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            input()   # wait for button press (Enter)

            incident_id = str(uuid.uuid4())
            print(f"\n─── Incident {incident_id[:8]} ─────────────────────────────────────")

            # Phase 1: record alert
            audio = record_alert()

            # Phase 2: POST to gateway → ASR → responder notified
            try:
                result = post_alert(audio, incident_id)
                print(f"🗣   Transcript : {result.get('transcription', '(none)')}")
                print(f"📡   Delivery   : {result.get('responder_delivery', '?')}")
            except requests.RequestException as exc:
                print(f"❌  Alert failed: {exc}")
                continue

            # Use the event_id returned by the gateway (room is stored under this key)
            gateway_event_id = result.get("event_id", incident_id)

            # Phase 3: open WebSocket, wait for responder, go live
            try:
                asyncio.run(live_call(gateway_event_id))
            except KeyboardInterrupt:
                pass

            print(f"\n─── Call ended. Press ENTER for a new alert. ──────────────────\n")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

