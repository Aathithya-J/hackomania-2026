"""
gateway.py — PAB Emergency Transcription Gateway

Accepts audio uploads from authorized emergency systems, forwards them to the
MERaLiON ASR engine (api.py), and dispatches the transcript to responders.

Endpoints:
  POST /v1/emergency/transcribe   — submit audio, receive transcript + delivery ACK
  POST /v1/admin/replay           — retry all queued dead-letter events  [admin only]
  GET  /healthz                   — liveness probe

Run:
  uvicorn gateway:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from models import (
    ASRResult,
    DeliveryStatus,
    EmergencyTranscribeRequest,
    LanguageHint,
    Priority,
    ResponderDispatchPayload,
    TranscribeResponse,
)
from responder_dispatch import dispatch, replay_dead_letter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pab.gateway")

# ── Config ─────────────────────────────────────────────────────────────────────
ASR_BASE_URL:       str   = os.getenv("ASR_BASE_URL", "http://localhost:8000")
GATEWAY_API_KEY:    str   = os.getenv("GATEWAY_API_KEY", "")          # required in production
ADMIN_API_KEY:      str   = os.getenv("ADMIN_API_KEY", "")
MAX_FILE_MB:        int   = int(os.getenv("MAX_AUDIO_FILE_MB", "50"))
ASR_TIMEOUT_S:      float = float(os.getenv("ASR_TIMEOUT_S", "120.0"))
DISPATCH_BG:        bool  = os.getenv("DISPATCH_BACKGROUND", "true").lower() == "true"

MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PAB Emergency Transcription Gateway",
    version="1.0.0",
    description="Accepts emergency audio, transcribes via MERaLiON ASR, and dispatches to responders.",
)

# Shared HTTP session to ASR engine
_asr_session = requests.Session()

# ── Live call config ──────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000
ASR_BUFFER_SECONDS  = float(os.getenv("ASR_BUFFER_SECONDS", "5.0"))   # max buffer before forced flush
ASR_BUFFER_BYTES    = int(ASR_BUFFER_SECONDS * SAMPLE_RATE * 4)        # float32 = 4 bytes/sample
ASR_MIN_FLUSH_SECONDS = float(os.getenv("ASR_MIN_FLUSH_SECONDS", "1.0"))  # min content before silence flush
ASR_MIN_FLUSH_BYTES   = int(ASR_MIN_FLUSH_SECONDS * SAMPLE_RATE * 4)
LIVE_SILENCE_RMS    = float(os.getenv("LIVE_SILENCE_RMS", "0.015"))    # RMS below this = silence
SILENCE_FLUSH_CHUNKS = int(os.getenv("SILENCE_FLUSH_CHUNKS", "15"))    # ~480ms silence triggers early flush


# ── Incident room ─────────────────────────────────────────────────────────────

@dataclass
class IncidentRoom:
    incident_id:          str
    status:               str               = "pending"   # pending | active | closed
    created_at:           str               = ""
    caller_id:            Optional[str]     = None
    zone_id:              Optional[str]     = None
    priority:             Optional[str]     = None
    initial_transcript:   str               = ""
    emergency_ws:         Optional[WebSocket] = field(default=None, repr=False)
    responder_ws:         Optional[WebSocket] = field(default=None, repr=False)
    _accepted:            Optional[asyncio.Event] = field(default=None, repr=False)
    transcript_buffer:    list              = field(default_factory=list)  # transcripts before responder WS connects

    def accepted_event(self) -> asyncio.Event:
        if self._accepted is None:
            self._accepted = asyncio.Event()
        return self._accepted


_rooms: dict[str, IncidentRoom] = {}


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _check_auth(api_key: Optional[str], required_key: str, label: str = "API key") -> None:
    """Raise 401/403 if key is invalid. Skips check when required_key is empty (dev mode)."""
    if not required_key:
        return  # no key configured — open (dev mode)
    if not api_key:
        raise HTTPException(status_code=401, detail=f"Missing {label}")
    if api_key != required_key:
        raise HTTPException(status_code=403, detail=f"Invalid {label}")


# ── Utility ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_asr(file_bytes: bytes, filename: str, translate: bool) -> tuple[ASRResult, float]:
    """POST audio bytes to the MERaLiON ASR engine; return result + latency ms."""
    endpoint = "/translate" if translate else "/transcribe"
    url = f"{ASR_BASE_URL}{endpoint}"
    t0 = time.perf_counter()
    try:
        resp = _asr_session.post(
            url,
            files={"file": (filename, file_bytes, "audio/wav")},
            timeout=ASR_TIMEOUT_S,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if not resp.ok:
            logger.error("ASR engine returned %d: %s", resp.status_code, resp.text[:300])
            raise HTTPException(
                status_code=502,
                detail=f"ASR engine error {resp.status_code}: {resp.text[:200]}",
            )
        return ASRResult(**resp.json()), latency_ms
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Cannot reach ASR engine at {ASR_BASE_URL}")
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="ASR engine timed out")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/healthz", tags=["ops"])
def healthz():
    """Liveness probe — also checks ASR engine reachability."""
    asr_ok = False
    try:
        r = _asr_session.get(f"{ASR_BASE_URL}/", timeout=5)
        asr_ok = r.ok
    except Exception:
        pass
    return {
        "gateway": "ok",
        "asr_reachable": asr_ok,
        "asr_url": ASR_BASE_URL,
        "ts": _now_iso(),
    }


@app.post("/v1/emergency/transcribe", response_model=TranscribeResponse, tags=["emergency"])
async def emergency_transcribe(
    file:        UploadFile = File(..., description="Audio file (wav, flac, ogg, aiff)"),
    incident_id: Optional[str] = Form(None),
    caller_id:   Optional[str] = Form(None),
    zone_id:     Optional[str] = Form(None),
    priority:    Optional[str] = Form(None),
    language_hint: Optional[str] = Form(None),
    translate:   bool            = Form(False),
    x_api_key:   Optional[str]   = Header(None, alias="X-API-Key"),
):
    """
    Submit an audio clip from an emergency system.

    - Validates auth via `X-API-Key` header.
    - Forwards audio to MERaLiON ASR engine.
    - Dispatches transcript to configured responder webhook.
    - Returns event_id, transcription, and delivery status immediately.
    """
    _check_auth(x_api_key, GATEWAY_API_KEY)

    # ── Validate file size ─────────────────────────────────────────────────────
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(audio_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {MAX_FILE_MB} MB.",
        )

    # ── Parse and validate form metadata ──────────────────────────────────────
    meta = EmergencyTranscribeRequest(
        incident_id=incident_id,
        caller_id=caller_id,
        zone_id=zone_id,
        priority=Priority(priority) if priority else Priority.MEDIUM,
        language_hint=LanguageHint(language_hint) if language_hint else LanguageHint.AUTO,
        translate=translate,
    )

    event_id = str(uuid.uuid4())
    ts = _now_iso()
    logger.info(
        "Event %s | incident=%s caller=%s zone=%s priority=%s translate=%s",
        event_id, meta.incident_id, meta.caller_id, meta.zone_id, meta.priority, meta.translate,
    )

    # ── Call ASR engine ────────────────────────────────────────────────────────
    asr_result, asr_latency = _call_asr(
        audio_bytes,
        file.filename or "audio.wav",
        meta.translate,
    )
    transcription = asr_result.text
    translation = asr_result.result if meta.translate else None

    logger.info("Event %s transcribed in %.0f ms: %s", event_id, asr_latency, transcription[:80])

    # ── Dispatch to responder ──────────────────────────────────────────────────
    dispatch_payload = ResponderDispatchPayload(
        event_id=event_id,
        incident_id=meta.incident_id,
        caller_id=meta.caller_id,
        zone_id=meta.zone_id,
        priority=meta.priority,
        transcription=transcription,
        translation=translation,
        language_hint=meta.language_hint,
        ts=ts,
    )
    t_dispatch = time.perf_counter()
    delivery_status = dispatch(dispatch_payload, background=DISPATCH_BG)
    dispatch_latency = (time.perf_counter() - t_dispatch) * 1000

    # ── Create incident room (for Phase 3 live call) ─────────────────────────
    _rooms[event_id] = IncidentRoom(
        incident_id=event_id,
        created_at=ts,
        caller_id=meta.caller_id,
        zone_id=meta.zone_id,
        priority=meta.priority.value if meta.priority else None,
        initial_transcript=transcription,
    )
    logger.info("Incident room created: %s", event_id)

    # ── Return response ────────────────────────────────────────────────────────
    return TranscribeResponse(
        event_id=event_id,
        incident_id=meta.incident_id,
        transcription=transcription,
        translation=translation,
        language_hint=meta.language_hint,
        priority=meta.priority,
        responder_delivery=delivery_status,
        asr_latency_ms=round(asr_latency, 1),
        dispatch_latency_ms=round(dispatch_latency, 1),
        ts=ts,
    )


# ── Phase 2: Responder accepts ───────────────────────────────────────────────

@app.post("/v1/incident/{incident_id}/accept", tags=["responder"])
async def accept_incident(
    incident_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Responder calls this to pick up an incident.
    Notifies the emergency WebSocket that the call is active.
    """
    _check_auth(x_api_key, GATEWAY_API_KEY)
    room = _rooms.get(incident_id)
    if not room:
        raise HTTPException(status_code=404, detail="Incident not found")
    if room.status == "active":
        return {"status": "already_active", "incident_id": incident_id}
    room.status = "active"
    room.accepted_event().set()
    # ws_emergency will send the "accepted" message itself after unblocking —
    # no need to send it here too.
    logger.info("Incident %s accepted by responder", incident_id)
    return {"status": "ok", "incident_id": incident_id, "initial_transcript": room.initial_transcript}


@app.get("/v1/incident/{incident_id}/status", tags=["responder"])
def incident_status(incident_id: str, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Poll incident status — useful for responder dashboard before WebSocket connect."""
    _check_auth(x_api_key, GATEWAY_API_KEY)
    room = _rooms.get(incident_id)
    if not room:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {
        "incident_id": incident_id,
        "status": room.status,
        "initial_transcript": room.initial_transcript,
        "created_at": room.created_at,
        "priority": room.priority,
        "zone_id": room.zone_id,
    }


@app.get("/v1/incidents", tags=["responder"])
def list_incidents(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """List all active/pending incidents — for responder dashboard alert feed."""
    _check_auth(x_api_key, GATEWAY_API_KEY)
    return [
        {
            "incident_id": r.incident_id,
            "status": r.status,
            "initial_transcript": r.initial_transcript,
            "created_at": r.created_at,
            "priority": r.priority,
            "zone_id": r.zone_id,
        }
        for r in _rooms.values()
        if r.status in ("pending", "active")
    ]


# ── Phase 3: Live WebSocket call ──────────────────────────────────────────────

def _strip_speaker_prefix(text: str) -> str:
    """Remove leading '<Speaker...>: ' tag from a string."""
    return re.sub(r"^<[^>]+>\s*:\s*", "", text).strip()


async def _transcribe_chunk_and_relay(audio_bytes: bytes, room: IncidentRoom) -> None:
    """Transcribe a buffered audio chunk and push to both sides (transcribe-only for speed)."""
    try:
        # Convert raw float32 PCM bytes → WAV bytes
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio_array, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        loop = asyncio.get_running_loop()
        # Transcribe only during live call — translation doubles GPU load and adds latency.
        # Translation is still used on the initial alert POST for the triage card.
        transcribe_result, _ = await loop.run_in_executor(
            None, _call_asr, wav_bytes, "chunk.wav", False
        )

        text = transcribe_result.text.strip()
        if not text:
            return

        msg = {"type": "transcript", "text": text, "ts": _now_iso(), "source": "emergency"}
        logger.info("Live transcript [%s]: %s", room.incident_id[:8], text[:80])
        # Always buffer so responder gets transcripts even if WS connected late
        room.transcript_buffer.append(msg)
        for ws in [room.emergency_ws, room.responder_ws]:
            if ws:
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass
    except Exception as exc:
        logger.error("Live ASR error for incident %s: %s", room.incident_id, exc)


@app.websocket("/ws/emergency/{incident_id}")
async def ws_emergency(websocket: WebSocket, incident_id: str):
    """
    Emergency device connects here after posting the initial alert.
    - Waits for responder to accept
    - Then streams mic audio; gateway transcribes every ASR_BUFFER_SECONDS
    - Receives: JSON events (accepted, transcript, responder_message)
    - Sends:    binary PCM float32 audio chunks
    """
    await websocket.accept()
    room = _rooms.get(incident_id)
    if not room:
        await websocket.close(code=4004, reason="Incident not found")
        return

    room.emergency_ws = websocket
    audio_buffer = bytearray()
    silent_chunks = 0  # consecutive silent chunk counter for early flush

    try:
        await websocket.send_json({"type": "waiting", "message": "Waiting for responder to pick up..."})

        # Block until responder accepts (5 min timeout)
        try:
            await asyncio.wait_for(room.accepted_event().wait(), timeout=300.0)
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "timeout", "message": "No responder picked up within 5 minutes."})
            await websocket.close()
            return

        await websocket.send_json({"type": "accepted", "message": "Responder connected. You can speak freely."})

        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                chunk = data["bytes"]
                audio_buffer.extend(chunk)

                # Forward raw audio to responder in real-time
                if room.responder_ws:
                    try:
                        await room.responder_ws.send_bytes(chunk)
                    except Exception:
                        pass

                # Track silence for early flush
                chunk_array = np.frombuffer(chunk, dtype=np.float32)
                rms = float(np.sqrt(np.mean(chunk_array ** 2))) if len(chunk_array) else 0.0
                if rms < LIVE_SILENCE_RMS:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # Flush on silence (if we have enough content) or when buffer is full
                should_flush = (
                    len(audio_buffer) >= ASR_BUFFER_BYTES  # hard cap
                    or (silent_chunks >= SILENCE_FLUSH_CHUNKS and len(audio_buffer) >= ASR_MIN_FLUSH_BYTES)
                )
                if should_flush:
                    chunk_bytes = bytes(audio_buffer)
                    audio_buffer.clear()
                    silent_chunks = 0
                    asyncio.create_task(_transcribe_chunk_and_relay(chunk_bytes, room))

    except WebSocketDisconnect:
        logger.info("Emergency WS disconnected: %s", incident_id)
    finally:
        room.emergency_ws = None
        room.status = "closed"
        if room.responder_ws:
            try:
                await room.responder_ws.send_json({"type": "emergency_disconnected", "message": "Emergency caller hung up."})
            except Exception:
                pass


@app.websocket("/ws/responder/{incident_id}")
async def ws_responder(websocket: WebSocket, incident_id: str):
    """
    Responder connects here after accepting an incident.
    - Receives: JSON transcripts + binary emergency audio
    - Sends:    binary PCM float32 audio (relayed raw to emergency, no ASR)
    """
    await websocket.accept()
    room = _rooms.get(incident_id)
    if not room:
        await websocket.close(code=4004, reason="Incident not found")
        return
    if room.status not in ("active", "pending"):
        await websocket.close(code=4003, reason="Incident is closed")
        return

    room.responder_ws = websocket
    await websocket.send_json({
        "type": "connected",
        "incident_id": incident_id,
        "initial_transcript": room.initial_transcript,
        "priority": room.priority,
        "zone_id": room.zone_id,
    })

    # Flush any transcripts that arrived before the responder WS connected
    if room.transcript_buffer:
        logger.info("Flushing %d buffered transcripts to responder [%s]", len(room.transcript_buffer), incident_id[:8])
        for buffered_msg in room.transcript_buffer:
            try:
                await websocket.send_json({**buffered_msg, "buffered": True})
            except Exception:
                break

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                # Relay responder audio to emergency (raw, no ASR)
                if room.emergency_ws:
                    try:
                        await room.emergency_ws.send_bytes(data["bytes"])
                    except Exception:
                        pass

            elif "text" in data and data["text"]:
                # Responder typed message → forward to emergency as JSON
                if room.emergency_ws:
                    try:
                        await room.emergency_ws.send_json({
                            "type": "responder_message",
                            "text": data["text"],
                            "ts": _now_iso(),
                        })
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info("Responder WS disconnected: %s", incident_id)
    finally:
        room.responder_ws = None
        if room.emergency_ws:
            try:
                await room.emergency_ws.send_json({"type": "responder_disconnected", "message": "Responder disconnected."})
            except Exception:
                pass


@app.post("/v1/admin/replay", tags=["admin"])
def admin_replay_dead_letter(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
):
    """
    Retry all events in the dead-letter queue.
    Requires `X-Admin-Key` header.
    """
    _check_auth(x_admin_key, ADMIN_API_KEY, label="Admin key")
    summary = replay_dead_letter()
    logger.info("Dead-letter replay: %s", summary)
    return {"status": "ok", "replay": summary, "ts": _now_iso()}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gateway:app", host="0.0.0.0", port=8001, reload=False)
