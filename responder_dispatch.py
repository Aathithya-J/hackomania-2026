"""
responder_dispatch.py — sends transcription events to responder webhooks.

Features:
  - Exponential backoff retry (configurable attempts + base delay)
  - Dead-letter queue: failed payloads saved as JSON-Lines to disk
  - HMAC-SHA256 request signing so responders can verify authenticity
  - Optional bearer-token auth header
  - Background thread dispatch so gateway doesn't block on slow endpoints
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from models import DeadLetterEntry, DeliveryStatus, ResponderDispatchPayload

load_dotenv()

logger = logging.getLogger("pab.dispatcher")

# ── Config from .env ───────────────────────────────────────────────────────────
RESPONDER_WEBHOOK_URL:  str = os.getenv("RESPONDER_WEBHOOK_URL", "")
WEBHOOK_SECRET:         str = os.getenv("WEBHOOK_SECRET", "")       # HMAC signing key

RETRY_ATTEMPTS:         int   = int(os.getenv("RETRY_ATTEMPTS", "4"))
RETRY_BASE_DELAY_S:     float = float(os.getenv("RETRY_BASE_DELAY_S", "1.0"))
DISPATCH_TIMEOUT_S:     float = float(os.getenv("DISPATCH_TIMEOUT_S", "10.0"))

DEAD_LETTER_FILE: Path = Path(os.getenv("DEAD_LETTER_FILE", "dead_letter_queue.jsonl"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sign_payload(body: bytes, secret: str) -> str:
    """Return hex HMAC-SHA256 signature of the raw JSON body."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _build_headers(body: bytes) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-PAB-Signature"] = _sign_payload(body, WEBHOOK_SECRET)
    return headers


def _write_dead_letter(
    payload: ResponderDispatchPayload,
    target_url: str,
    error: Optional[str],
    retry_count: int,
) -> None:
    """Append failed dispatch to dead-letter JSONL file for later replay."""
    entry = DeadLetterEntry(
        payload=payload,
        target_url=target_url,
        failed_at=_now_iso(),
        last_error=error,
        retry_count=retry_count,
    )
    try:
        with DEAD_LETTER_FILE.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        logger.warning("Dead-lettered event %s → %s", payload.event_id, DEAD_LETTER_FILE)
    except OSError as exc:
        logger.error("Could not write dead-letter entry: %s", exc)


# ── Core dispatch logic ────────────────────────────────────────────────────────

def _attempt_dispatch(
    payload: ResponderDispatchPayload,
    url: str,
    session: requests.Session,
) -> DeliveryStatus:
    """
    Try to POST `payload` to `url` with exponential backoff.
    Returns DELIVERED on success, QUEUED if all retries failed.
    """
    body = payload.model_dump_json().encode()
    headers = _build_headers(body)
    last_error: Optional[str] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.post(
                url,
                data=body,
                headers=headers,
                timeout=DISPATCH_TIMEOUT_S,
            )
            if resp.ok:
                logger.info(
                    "Dispatched event %s (attempt %d/%d) → HTTP %d",
                    payload.event_id, attempt, RETRY_ATTEMPTS, resp.status_code,
                )
                return DeliveryStatus.DELIVERED
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("Dispatch attempt %d failed: %s", attempt, last_error)

        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning("Dispatch attempt %d error: %s", attempt, last_error)

        if attempt < RETRY_ATTEMPTS:
            delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            logger.debug("Retrying in %.1fs...", delay)
            time.sleep(delay)

    # All retries exhausted — dead-letter
    _write_dead_letter(payload, url, last_error, RETRY_ATTEMPTS)
    return DeliveryStatus.QUEUED


# ── Public API ─────────────────────────────────────────────────────────────────

# Shared session for connection pooling across dispatches
_session = requests.Session()


def dispatch(
    payload: ResponderDispatchPayload,
    url: Optional[str] = None,
    background: bool = True,
) -> DeliveryStatus:
    """
    Send `payload` to the responder webhook.

    Args:
        payload:    The transcription event to forward.
        url:        Override webhook URL (defaults to RESPONDER_WEBHOOK_URL from .env).
        background: If True, dispatch runs in a daemon thread and this call
                    returns DeliveryStatus.QUEUED immediately (fire-and-forget).
                    If False, blocks and returns the real delivery status.

    Returns:
        DeliveryStatus — DELIVERED, QUEUED, or FAILED.
    """
    target = url or RESPONDER_WEBHOOK_URL
    if not target:
        logger.error("RESPONDER_WEBHOOK_URL is not set — cannot dispatch event %s", payload.event_id)
        _write_dead_letter(payload, "(no url)", "RESPONDER_WEBHOOK_URL not configured", 0)
        return DeliveryStatus.FAILED

    if background:
        def _run():
            _attempt_dispatch(payload, target, _session)

        t = threading.Thread(target=_run, daemon=True, name=f"dispatch-{payload.event_id[:8]}")
        t.start()
        return DeliveryStatus.QUEUED   # caller gets immediate ACK

    return _attempt_dispatch(payload, target, _session)


def replay_dead_letter(url: Optional[str] = None) -> dict[str, int]:
    """
    Re-attempt delivery of every event in the dead-letter file.
    Entries that succeed are removed; failures remain.

    Returns a summary dict: {"attempted": N, "delivered": N, "failed": N}
    """
    if not DEAD_LETTER_FILE.exists():
        return {"attempted": 0, "delivered": 0, "failed": 0}

    target = url or RESPONDER_WEBHOOK_URL
    lines = DEAD_LETTER_FILE.read_text(encoding="utf-8").splitlines()
    remaining: list[str] = []
    delivered = failed = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = DeadLetterEntry.model_validate_json(line)
            status = _attempt_dispatch(entry.payload, target or entry.target_url, _session)
            if status == DeliveryStatus.DELIVERED:
                delivered += 1
            else:
                remaining.append(line)
                failed += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not replay dead-letter entry: %s", exc)
            remaining.append(line)
            failed += 1

    # Rewrite file with only un-delivered entries
    DEAD_LETTER_FILE.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
    return {"attempted": len(lines), "delivered": delivered, "failed": failed}
