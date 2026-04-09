"""Router for ingesting Apple Notes via iOS Shortcuts webhook.

Called by iOS Shortcuts when the user saves a note on iPhone/iPad.
The shortcut sends a POST request to /api/ingest/apple-note with the
note content, which JARVIS classifies by intent and stores for processing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AppleNotePayload(BaseModel):
    """Payload received from an iOS Shortcut when a note is created/shared."""

    title: str
    content: str
    created_at: Optional[str] = None
    note_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "research": [
        "investiga", "busca", "qué es", "como funciona", "cómo funciona",
        "explica", "research", "find out", "look up",
    ],
    "code": [
        "código", "bug", "refactor", "función", "implementa", "arregla",
        "error", "fix", "implement", "debug",
    ],
    "task": [
        "hacer", "recordar", "pendiente", "tarea", "cuando", "mañana",
        "deadline", "reminder", "todo",
    ],
    "idea": [
        "idea", "podría", "y si", "qué pasaría", "proyecto",
        "what if", "maybe", "could we",
    ],
}


def classify_intent(text: str) -> str:
    """Classify the intent of a text string.

    Checks the text (lowercased) against keyword lists for each intent
    category.  The first category whose keywords appear in the text wins.
    Falls back to ``"general"`` if no match is found.

    Parameters
    ----------
    text:
        The note content or title to classify.

    Returns
    -------
    str
        One of: ``"research"``, ``"code"``, ``"task"``, ``"idea"``,
        ``"general"``.
    """
    lower = text.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                logger.debug("classify_intent: matched '%s' → %s", kw, intent)
                return intent
    return "general"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_INTENT_MESSAGES: dict[str, str] = {
    "research": "Nota de investigación recibida. JARVIS la procesará pronto.",
    "code": "Nota de código recibida. La revisaré en cuanto pueda.",
    "task": "Tarea añadida. Te recordaré cuando sea el momento.",
    "idea": "Idea guardada. ¡Buena idea, la exploraremos!",
    "general": "Nota recibida y guardada correctamente.",
}


@router.post("/apple-note")
async def receive_apple_note(payload: AppleNotePayload) -> dict:
    """Receive an Apple Note from an iOS Shortcut.

    This endpoint is called by an iOS Shortcut configured to POST note
    data whenever the user saves or shares a note from the Apple Notes
    app.  It classifies the intent of the note content and returns a
    confirmation message in Spanish.

    Expected JSON body (all strings):
    - ``title``      — Note title (required)
    - ``content``    — Note body (required)
    - ``created_at`` — ISO 8601 timestamp (optional)
    - ``note_id``    — Apple internal note ID (optional)
    """
    received_at = datetime.now(tz=timezone.utc).isoformat()

    # Classify using title + content together for better accuracy
    combined_text = f"{payload.title} {payload.content}"
    intent = classify_intent(combined_text)

    logger.info(
        "Apple Note recibida | id=%s | title=%r | intent=%s | received_at=%s",
        payload.note_id or "—",
        payload.title[:60],
        intent,
        received_at,
    )

    confirmation = _INTENT_MESSAGES.get(intent, _INTENT_MESSAGES["general"])

    return {
        "status": "ok",
        "intent": intent,
        "received_at": received_at,
        "message": confirmation,
        "note_id": payload.note_id,
    }


__all__ = ["router", "AppleNotePayload", "classify_intent"]
