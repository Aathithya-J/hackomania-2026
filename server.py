"""
server.py — PAB Responder Server
---------------------------------
Receives alert transcripts from the gateway via POST /webhook.
Streams them to the React dashboard via SSE GET /stream.
Provides accept + incident list endpoints that proxy to the gateway.

Run:
    pip install fastapi uvicorn python-dotenv requests aiofiles websockets google-generativeai
    uvicorn server:app --host 0.0.0.0 --port 8002 --reload

Expose publicly:
    ngrok http 8002
    → set RESPONDER_WEBHOOK_URL=https://<this-ngrok-url>/webhook in gateway .env
"""

import asyncio
import json
import logging
import os
from datetime import datetime

import requests as http
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
PORT            = int(os.getenv("PORT", 8002))
GATEWAY_URL     = os.getenv("GATEWAY_URL", "")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

# ---------------------------------------------------------------------------
# Gemini setup (lazy import so server starts even if package missing)
# ---------------------------------------------------------------------------
_gemini_model = None

def _get_gemini_model():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        log.info("Gemini model loaded: gemini-1.5-flash")
        return _gemini_model
    except Exception as e:
        log.error("Failed to load Gemini model: %s", e)
        return None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="PAB Responder Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory SSE client registry  {client_id: asyncio.Queue}
_clients: dict[str, asyncio.Queue] = {}

# Track active background WS relay tasks
_active_relays: set[str] = set()


# ---------------------------------------------------------------------------
# Gemini triage analysis
# ---------------------------------------------------------------------------
async def analyze_with_gemini(transcript_text: str) -> dict:
    """
    Call Gemini to produce triage level, keywords, possible issues and actions.
    Falls back to a safe P3 default if anything goes wrong.
    """
    model = _get_gemini_model()
    if not model or not transcript_text.strip():
        log.warning("Gemini skipped — no model or empty transcript")
        return _fallback_triage(transcript_text)

    prompt = f"""You are an expert emergency medical dispatcher following SCDF (Singapore Civil Defence Force) triage protocols.

Analyze this emergency call transcript and return a triage assessment.

Transcript:
{transcript_text}

Triage levels:
- P1+  Immediately life-threatening (cardiac arrest, stopped breathing, severe trauma)
- P1   Critical emergency (chest pain, stroke, major bleeding, unconscious but breathing)
- P2   Emergency (fractures, moderate pain, altered consciousness, moderate breathing difficulty)
- P3   Ambulatory (minor injuries, mild symptoms, walking wounded)
- P4   Non-emergency (advice only, no immediate threat)

Return ONLY a valid JSON object with these exact keys:
{{
  "triage_level": "P1",
  "keywords": ["chest pain", "difficulty breathing"],
  "possible_issues": ["myocardial infarction", "angina"],
  "actions": [
    "Dispatch ambulance Priority 1",
    "Guide caller to keep patient still",
    "Advise caller to locate nearest AED",
    "Prepare advanced life support on arrival"
  ],
  "confidence": 0.87
}}

No markdown, no explanation, just the JSON object."""

    try:
        loop = asyncio.get_event_loop()
        # run_in_executor so we don't block the event loop
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        text = response.text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        log.info("Gemini triage result: %s", result)
        return result

    except Exception as e:
        log.error("Gemini analysis failed: %s", e)
        return _fallback_triage(transcript_text)


def _fallback_triage(transcript_text: str) -> dict:
    """
    Simple keyword-based fallback when Gemini is unavailable.
    """
    text = transcript_text.lower()
    critical = ["cardiac arrest", "heart attack", "not breathing", "cannot breathe",
                "unconscious", "collapsed", "chest pain", "stroke", "drowning",
                "choking", "stabbing", "gunshot", "overdose", "suicide"]
    urgent = ["bleeding", "fall", "fell", "seizure", "difficulty breathing",
              "shortness of breath", "allergic", "burn", "pain", "hurt"]

    found_critical = [k for k in critical if k in text]
    found_urgent   = [k for k in urgent   if k in text]

    if found_critical:
        level = "P1"
        actions = [
            "Dispatch ambulance Priority 1",
            "Keep caller on line",
            "Provide CPR/first aid instructions if needed",
            "Alert nearest AED-equipped responders",
        ]
    elif found_urgent:
        level = "P2"
        actions = [
            "Dispatch ambulance Priority 2",
            "Assess severity with caller",
            "Provide first aid guidance",
        ]
    else:
        level = "P3"
        actions = [
            "Assess situation with caller",
            "Dispatch appropriate resources",
        ]

    return {
        "triage_level": level,
        "keywords": found_critical + found_urgent,
        "possible_issues": [],
        "actions": actions,
        "confidence": None,
    }


# ---------------------------------------------------------------------------
# Background WS relay
# ---------------------------------------------------------------------------
async def _auto_responder_ws(incident_id: str, gw_ws_url: str, initial_transcript: str):
    """
    Auto-connect to the gateway WS after accept.
    Fans out all messages to SSE clients.
    """
    if incident_id in _active_relays:
        log.info("Relay already running for %s, skipping duplicate", incident_id)
        return
    _active_relays.add(incident_id)
    log.info("=== AUTO-CONNECTING WS for incident %s ===", incident_id)
    if initial_transcript:
        log.info("[%s] INITIAL TRANSCRIPT: %s", incident_id[:8], initial_transcript)
    try:
        async with websockets.connect(
            gw_ws_url,
            additional_headers={"X-API-Key": GATEWAY_API_KEY},
            ping_interval=20,
            open_timeout=15,
        ) as gw_ws:
            log.info("[%s] Gateway WS connected", incident_id[:8])
            async for message in gw_ws:
                if not isinstance(message, str):
                    continue
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                mtype = data.get("type", "")
                if mtype == "transcript":
                    log.info("[%s] TRANSCRIPT: %s", incident_id[:8], data.get("text", ""))
                elif mtype == "connected":
                    log.info("[%s] WS handshake OK — priority=%s zone=%s",
                             incident_id[:8], data.get("priority"), data.get("zone_id"))
                elif mtype == "emergency_disconnected":
                    log.info("[%s] Emergency caller hung up.", incident_id[:8])
                else:
                    log.info("[%s] GW event: %s", incident_id[:8], message[:200])

                data["incident_id"] = incident_id
                for q in list(_clients.values()):
                    try:
                        await q.put(data)
                    except Exception:
                        pass
    except Exception as exc:
        log.error("[%s] Auto-relay WS error: %s", incident_id[:8], exc)
    finally:
        _active_relays.discard(incident_id)
        log.info("[%s] Auto-relay WS closed", incident_id[:8])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _verify_secret(request: Request):
    if not WEBHOOK_SECRET:
        return
    auth_header   = request.headers.get("Authorization", "")
    secret_header = request.headers.get("X-Webhook-Secret", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if token != WEBHOOK_SECRET and secret_header != WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")


def _extract_transcript(data: dict) -> str:
    """
    Pull transcript text from whatever field the gateway uses.
    Gateway sends: transcription, transcript, text, or initial_transcript.
    """
    for field in ("transcription", "transcript", "text", "initial_transcript"):
        val = data.get(field, "")
        if val:
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_dashboard():
    return FileResponse("index.html", media_type="text/html")

@app.get("/index.html")
async def serve_dashboard_explicit():
    return FileResponse("index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "clients_connected": len(_clients),
        "gateway_url": GATEWAY_URL or "(not set)",
        "gemini_enabled": bool(GEMINI_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/incidents")
def list_incidents():
    if not GATEWAY_URL:
        raise HTTPException(status_code=503, detail="GATEWAY_URL not configured")
    try:
        r = http.get(
            f"{GATEWAY_URL}/v1/incidents",
            headers={"X-API-Key": GATEWAY_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except http.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Gateway unreachable: {exc}")


@app.post("/incident/{incident_id}/accept")
async def accept_incident(incident_id: str):
    if not GATEWAY_URL:
        raise HTTPException(status_code=503, detail="GATEWAY_URL not configured")
    try:
        r = http.post(
            f"{GATEWAY_URL}/v1/incident/{incident_id}/accept",
            headers={"X-API-Key": GATEWAY_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        result = r.json()
    except http.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Gateway error: {exc}")

    ws_url = (
        GATEWAY_URL
        .replace("https://", "wss://")
        .replace("http://", "ws://")
    ) + f"/ws/responder/{incident_id}"

    event = {
        "type": "accepted",
        "incident_id": incident_id,
        "initial_transcript": result.get("initial_transcript", ""),
        "ws_url": ws_url,
        "received_at": datetime.utcnow().isoformat(),
    }

    for q in list(_clients.values()):
        try:
            await q.put(event)
        except Exception:
            pass

    asyncio.create_task(
        _auto_responder_ws(incident_id, ws_url, result.get("initial_transcript", ""))
    )

    log.info("Incident %s accepted — WS relay started", incident_id)
    return {"status": "ok", "incident_id": incident_id, "ws_url": ws_url}


@app.post("/webhook")
async def receive_transcript(request: Request):
    """
    Gateway POSTs alert JSON here.
    We run Gemini triage on the transcript, then broadcast everything via SSE.
    """
    _verify_secret(request)

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    data.setdefault("received_at", datetime.utcnow().isoformat())

    # Build gateway WS / accept URLs for the dashboard
    if GATEWAY_URL and "event_id" in data:
        data["ws_url_responder"] = (
            GATEWAY_URL
            .replace("https://", "wss://")
            .replace("http://", "ws://")
        ) + f"/ws/responder/{data['event_id']}"
        data["accept_url"] = f"/incident/{data['event_id']}/accept"

    # ── Triage via Gemini ────────────────────────────────────────────────
    transcript_text = _extract_transcript(data)
    log.info("Running Gemini triage on transcript (%d chars): %s",
             len(transcript_text), transcript_text[:120])

    analysis = await analyze_with_gemini(transcript_text)

    # Normalise field names so the dashboard always sees the same keys
    data["triage_level"]         = analysis.get("triage_level", "P3")
    data["keywords"]             = analysis.get("keywords", [])
    data["possible_issues"]      = analysis.get("possible_issues", [])
    data["actions"]              = analysis.get("actions", [])
    if analysis.get("confidence") is not None:
        data["confidence"]       = analysis["confidence"]

    log.info("Webhook received — triage=%s — broadcasting to %d client(s)",
             data["triage_level"], len(_clients))
    log.info("Received data: %s", json.dumps(data, indent=2))

    dead = []
    for cid, q in _clients.items():
        try:
            await q.put(data)
        except Exception:
            dead.append(cid)
    for cid in dead:
        _clients.pop(cid, None)

    return JSONResponse({"status": "ok", "delivered_to": len(_clients)})


@app.get("/stream")
async def stream_transcripts(request: Request):
    client_id = f"{request.client.host}_{id(request)}"
    q: asyncio.Queue = asyncio.Queue()
    _clients[client_id] = q
    log.info("SSE client connected: %s  (total: %d)", client_id, len(_clients))

    async def event_generator():
        yield f"event: connected\ndata: {json.dumps({'client_id': client_id})}\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    if await request.is_disconnected():
                        break
        except asyncio.CancelledError:
            pass
        finally:
            _clients.pop(client_id, None)
            log.info("SSE client removed: %s  (total: %d)", client_id, len(_clients))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/call/{incident_id}")
async def ws_call_relay(client_ws: WebSocket, incident_id: str):
    await client_ws.accept()

    if not GATEWAY_URL:
        await client_ws.send_json({"type": "error", "message": "GATEWAY_URL not configured"})
        await client_ws.close()
        return

    gw_ws_url = (
        GATEWAY_URL
        .replace("https://", "wss://")
        .replace("http://", "ws://")
    ) + f"/ws/responder/{incident_id}"

    # If _auto_responder_ws is already running for this incident (started by accept),
    # stop it so we don't open two competing connections to the same gateway endpoint.
    # The dashboard's /ws/call connection will become the sole room.responder_ws.
    _active_relays.discard(incident_id)

    log.info("Dashboard WS relay for incident %s → %s", incident_id, gw_ws_url)

    try:
        async with websockets.connect(
            gw_ws_url,
            additional_headers={"X-API-Key": GATEWAY_API_KEY},
            ping_interval=20,
        ) as gw_ws:

            async def gateway_to_client():
                async for message in gw_ws:
                    if isinstance(message, str):
                        await client_ws.send_text(message)
                        try:
                            msg = json.loads(message)
                            if msg.get("type") == "transcript":
                                msg["incident_id"] = incident_id
                                for q in list(_clients.values()):
                                    try:
                                        await q.put(msg)
                                    except Exception:
                                        pass
                                log.info("[%s] transcript: %s",
                                         incident_id[:8], msg.get("text", "")[:80])
                        except Exception:
                            pass
                    elif isinstance(message, bytes):
                        await client_ws.send_bytes(message)

            async def client_to_gateway():
                while True:
                    msg = await client_ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        await gw_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await gw_ws.send(msg["text"])

            done, pending = await asyncio.wait(
                [asyncio.create_task(gateway_to_client()),
                 asyncio.create_task(client_to_gateway())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except websockets.exceptions.InvalidStatus as exc:
        log.error("Gateway WS rejected: %s", exc)
        try:
            await client_ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("WS relay error for incident %s: %s", incident_id, exc)
    finally:
        log.info("WS relay closed for incident %s", incident_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
