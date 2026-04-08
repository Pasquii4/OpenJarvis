"""Groq inference engine — OpenAI-compatible Groq Cloud API with automatic fallback.

Registers as ``groq`` in the engine registry.  Uses the ``openai`` SDK
pointed at ``https://api.groq.com/openai/v1``.

When a request to the primary model fails (rate-limit, server error),
the engine automatically retries with a smaller, faster fallback model.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any, Dict, List

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    InferenceEngine,
    messages_to_dicts,
)

logger = logging.getLogger(__name__)

# Available Groq models — ordered by quality
GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # Default — best quality/speed ratio
    "llama-3.1-8b-instant",  # Fallback — fast for simple tasks
    "mixtral-8x7b-32768",  # Long-context alternative
]

_DEFAULT_MODEL = GROQ_MODELS[0]
_FALLBACK_MODEL = GROQ_MODELS[1]


@EngineRegistry.register("groq")
class GroqEngine(InferenceEngine):
    """Groq Cloud inference via OpenAI-compatible API with automatic fallback."""

    engine_id = "groq"
    is_cloud = True

    def __init__(self) -> None:
        self._client: Any = None
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        if self._api_key:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except ImportError:
                logger.warning(
                    "openai package not installed — Groq engine unavailable"
                )

    def health(self) -> bool:
        """Check if Groq API is reachable."""
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return well-known Groq models."""
        return list(GROQ_MODELS)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response, with automatic fallback on failure."""
        if self._client is None:
            raise EngineConnectionError(
                "Groq client not available — set GROQ_API_KEY "
                "and install openai>=1.30"
            )

        target_model = model or _DEFAULT_MODEL
        result = self._try_generate(
            messages,
            model=target_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if result is not None:
            return result

        # Fallback to smaller model
        if target_model != _FALLBACK_MODEL:
            logger.warning(
                "Groq model %r failed, falling back to %r",
                target_model,
                _FALLBACK_MODEL,
            )
            result = self._try_generate(
                messages,
                model=_FALLBACK_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if result is not None:
                return result

        raise EngineConnectionError(
            f"All Groq models failed ({target_model}, {_FALLBACK_MODEL})"
        )

    def _try_generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any] | None:
        """Attempt a single generate call; return None on failure."""
        try:
            # Strip unsupported kwargs for Groq
            kwargs.pop("response_format", None)

            create_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages_to_dicts(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Forward tool definitions if present
            tools = kwargs.pop("tools", None)
            if tools:
                create_kwargs["tools"] = tools

            t0 = time.monotonic()
            resp = self._client.chat.completions.create(**create_kwargs)
            elapsed = time.monotonic() - t0

            choice = resp.choices[0]
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            result: Dict[str, Any] = {
                "content": choice.message.content or "",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": (usage.total_tokens if usage else 0),
                },
                "model": resp.model,
                "finish_reason": choice.finish_reason or "stop",
                "cost_usd": 0.0,  # Groq is free-tier
                "ttft": elapsed,
            }

            # Extract tool_calls if present
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in choice.message.tool_calls
                ]

            return result

        except Exception as exc:
            logger.warning("Groq generate failed for model %r: %s", model, exc)
            return None

    def close(self) -> None:
        """Release resources."""
        self._client = None


__all__ = ["GroqEngine", "GROQ_MODELS"]
