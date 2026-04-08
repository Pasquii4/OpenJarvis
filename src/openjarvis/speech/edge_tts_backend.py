"""Edge TTS backend — free Microsoft Edge text-to-speech via edge-tts.

Uses ``es-ES-AlvaroNeural`` as default voice for Spanish output.
Registered as ``edge-tts`` in the SpeechRegistry (for TTS discovery).
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List

from openjarvis.speech.tts import TTSBackend, TTSResult

logger = logging.getLogger(__name__)


class EdgeTTSBackend(TTSBackend):
    """Text-to-speech using Microsoft Edge TTS (free, no API key)."""

    backend_id = "edge-tts"

    def __init__(self, voice: str = "es-ES-AlvaroNeural") -> None:
        self._voice = voice

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        """Synthesize text to audio bytes using edge-tts."""
        import edge_tts

        voice = voice_id or self._voice
        rate = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            asyncio.run(communicate.save(str(tmp_path)))
            audio_bytes = tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

        return TTSResult(
            audio=audio_bytes,
            format=output_format,
            voice_id=voice,
        )

    def available_voices(self) -> List[str]:
        """Return a curated list of Spanish voices."""
        return [
            "es-ES-AlvaroNeural",
            "es-ES-ElviraNeural",
            "es-MX-DaliaNeural",
            "es-MX-JorgeNeural",
        ]

    def health(self) -> bool:
        """Check if edge-tts is importable."""
        try:
            import edge_tts  # noqa: F401

            return True
        except ImportError:
            return False


__all__ = ["EdgeTTSBackend"]
