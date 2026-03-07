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

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
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
