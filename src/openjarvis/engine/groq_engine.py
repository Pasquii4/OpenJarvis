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
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    InferenceEngine,
    messages_to_dicts,
)
from openjarvis.engine._stubs import StreamChunk

logger = logging.getLogger(__name__)

# Available Groq models — ordered by quality
GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # Default — best quality/speed ratio
    "llama-3.1-8b-instant",  # Fallback — fast for simple tasks
    "gemma2-9b-it",  # Alternative — good for structured tasks
    "mixtral-8x7b-32768",  # Long-context alternative
]

_DEFAULT_MODEL = GROQ_MODELS[0]
_FALLBACK_MODEL = GROQ_MODELS[1]


@EngineRegistry.register("groq")
class GroqEngine(InferenceEngine):
    """Groq Cloud inference via OpenAI-compatible API with automatic fallback."""

    engine_id = "groq"
    is_cloud = True

    def __init__(self, api_key: str | None = None) -> None:
        self._client: Any = None
        self._async_client: Any = None
        
        # Resolve API key — fallback to env if placeholder or empty
        res_key = api_key or ""
        if not res_key or res_key.startswith("$"):
            res_key = os.environ.get("GROQ_API_KEY", "")
        
        self._api_key = res_key
        if self._api_key:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                self._async_client = openai.AsyncOpenAI(
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

    # -- Streaming methods -----------------------------------------------------

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield content tokens as they are generated via Groq streaming API."""
        if self._async_client is None:
            raise EngineConnectionError(
                "Groq async client not available — set GROQ_API_KEY "
                "and install openai>=1.30"
            )

        target_model = model or _DEFAULT_MODEL

        # Strip unsupported kwargs
        kwargs.pop("response_format", None)

        create_kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        tools = kwargs.pop("tools", None)
        if tools:
            create_kwargs["tools"] = tools

        try:
            response = await self._async_client.chat.completions.create(
                **create_kwargs
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            logger.warning(
                "Groq stream failed for model %r: %s", target_model, exc
            )
            raise EngineConnectionError(
                f"Groq streaming failed for {target_model}: {exc}"
            ) from exc

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield StreamChunks with content, tool_calls, and finish_reason."""
        if self._async_client is None:
            raise EngineConnectionError(
                "Groq async client not available — set GROQ_API_KEY "
                "and install openai>=1.30"
            )

        target_model = model or _DEFAULT_MODEL

        # Strip unsupported kwargs
        kwargs.pop("response_format", None)

        create_kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        tools = kwargs.pop("tools", None)
        if tools:
            create_kwargs["tools"] = tools

        try:
            response = await self._async_client.chat.completions.create(
                **create_kwargs
            )
            async for chunk in response:
                if not chunk.choices:
                    # Final chunk may carry only usage
                    if chunk.usage:
                        yield StreamChunk(
                            usage={
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            }
                        )
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                content = delta.content if delta else None
                finish = choice.finish_reason

                # Extract tool_calls from delta
                tc_list = None
                if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                    tc_list = [
                        {
                            "index": tc.index,
                            "id": getattr(tc, "id", None),
                            "function": {
                                "name": getattr(tc.function, "name", None)
                                if tc.function
                                else None,
                                "arguments": getattr(
                                    tc.function, "arguments", ""
                                )
                                if tc.function
                                else "",
                            },
                        }
                        for tc in delta.tool_calls
                    ]

                usage_data = None
                if chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

                if content or tc_list or finish or usage_data:
                    yield StreamChunk(
                        content=content,
                        tool_calls=tc_list,
                        finish_reason=finish,
                        usage=usage_data,
                    )
        except Exception as exc:
            logger.warning(
                "Groq stream_full failed for model %r: %s", target_model, exc
            )
            raise EngineConnectionError(
                f"Groq streaming failed for {target_model}: {exc}"
            ) from exc

    def close(self) -> None:
        """Release resources."""
        self._client = None
        self._async_client = None


__all__ = ["GroqEngine", "GROQ_MODELS"]
