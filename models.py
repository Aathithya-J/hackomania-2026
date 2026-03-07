"""
models.py — shared Pydantic schemas for the PAB gateway.

Used by:
  - gateway.py          (request validation, response shaping)
  - responder_dispatch.py (outbound payload definition)
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class Priority(str, enum.Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class LanguageHint(str, enum.Enum):
    EN = "en"   # English
    ZH = "zh"   # Mandarin Chinese
    MS = "ms"   # Malay
    TA = "ta"   # Tamil
    AUTO = "auto"  # let ASR decide


class DeliveryStatus(str, enum.Enum):
    DELIVERED = "delivered"
    QUEUED    = "queued"     # responder endpoint down; saved to dead-letter queue
    FAILED    = "failed"     # exhausted retries


# ── Incoming request metadata ─────────────────────────────────────────────────
# (audio itself arrives as multipart; this models the extra form fields)

class EmergencyTranscribeRequest(BaseModel):
    """
    Metadata that accompanies an audio upload to POST /v1/emergency/transcribe.
    All fields are optional so callers without full context can still submit.
    """
    incident_id:    Optional[str]         = Field(None,  description="Caller-supplied incident reference")
    caller_id:      Optional[str]         = Field(None,  description="Device / operator identifier")
    zone_id:        Optional[str]         = Field(None,  description="Geographic / operational zone")
    priority:       Optional[Priority]    = Field(Priority.MEDIUM, description="Urgency level")
    language_hint:  Optional[LanguageHint]= Field(LanguageHint.AUTO, description="Expected speech language")
    translate:      bool                  = Field(False, description="If True, also return English translation")


# ── ASR engine responses (internal, from api.py) ──────────────────────────────

class ASRResult(BaseModel):
    """Raw result returned by api.py /transcribe or /translate."""
    transcription: Optional[str] = None
    result:        Optional[str] = None   # populated by /translate

    @property
    def text(self) -> str:
        return self.transcription or self.result or ""


# ── Gateway → caller response ─────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    """Response body returned by POST /v1/emergency/transcribe."""
    event_id:               str
    incident_id:            Optional[str]
    transcription:          str
    translation:            Optional[str]       = None
    language_hint:          Optional[LanguageHint]
    priority:               Optional[Priority]
    responder_delivery:     DeliveryStatus
    asr_latency_ms:         Optional[float]     = None
    dispatch_latency_ms:    Optional[float]     = None
    ts:                     str                 = Field(..., description="ISO-8601 UTC timestamp")


# ── Gateway → responder webhook payload ───────────────────────────────────────

class ResponderDispatchPayload(BaseModel):
    """
    Payload POSTed to the responder webhook.
    Keep this flat and self-contained so responders need no shared state.
    """
    event_id:       str
    incident_id:    Optional[str]
    caller_id:      Optional[str]
    zone_id:        Optional[str]
    priority:       Optional[Priority]
    transcription:  str
    translation:    Optional[str]   = None
    language_hint:  Optional[LanguageHint]
    ts:             str             = Field(..., description="ISO-8601 UTC timestamp of transcription")


# ── Dead-letter queue entry (persisted to disk on delivery failure) ────────────

class DeadLetterEntry(BaseModel):
    """Saved when a responder dispatch fails after all retries."""
    payload:        ResponderDispatchPayload
    target_url:     str
    failed_at:      str             = Field(..., description="ISO-8601 UTC timestamp")
    last_error:     Optional[str]   = None
    retry_count:    int             = 0
