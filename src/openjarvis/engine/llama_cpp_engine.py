"""llama.cpp inference engine — local GGUF inference with process management.

Registers as ``llama_cpp`` in the engine registry. Communicates with
``llama-server`` via its OpenAI-compatible HTTP API. Attempts to start
the server if it is not already running.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

import httpx

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._openai_compat import _OpenAICompatibleEngine
from openjarvis.engine._base import InferenceEngine, EngineConnectionError

logger = logging.getLogger(__name__)

@EngineRegistry.register("llama_cpp")
class LlamaCppEngine(_OpenAICompatibleEngine):
    """llama.cpp engine with automatic server management."""

    engine_id = "llama_cpp"
    _api_prefix = "/v1"

    def __init__(
        self,
        host: str | None = None,
        port: int = 8080,
        model_path: str = "",
        lora_path: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        binary_path: str = "",
        timeout: float = 600.0,
    ) -> None:
        self._host_addr = host or "127.0.0.1"
        self._port = port
        base_url = f"http://{self._host_addr}:{self._port}"
        super().__init__(host=base_url, timeout=timeout)
        
        self._model_path = model_path
        self._lora_path = lora_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._binary_path = binary_path or "llama-server"
        self._process: subprocess.Popen | None = None

    def health(self) -> bool:
        """Check if llama-server is healthy, try to start it if not."""
        if super().health():
            return True
        
        # If not healthy, try to start it if we have a model_path
        if self._model_path and (self._host_addr == "127.0.0.1" or self._host_addr == "localhost"):
            return self._start_server()
            
        return False

    def _start_server(self) -> bool:
        """Attempt to start the llama-server process."""
        if not self._model_path:
            logger.warning("llama_cpp: model_path not configured, cannot start server")
            return False

        full_model_path = os.path.expanduser(self._model_path)
        if not os.path.exists(full_model_path):
            logger.warning("llama_cpp: model_path %s does not exist", full_model_path)
            return False

        cmd = [
            self._binary_path,
            "-m", full_model_path,
            "--port", str(self._port),
            "--ctx-size", str(self._n_ctx),
            "--n-gpu-layers", str(self._n_gpu_layers),
        ]
        if self._lora_path:
            cmd.extend(["--lora", os.path.expanduser(self._lora_path)])
            
        # Add host if not default
        if self._host_addr not in ("127.0.0.1", "localhost"):
            cmd.extend(["--host", self._host_addr])

        logger.info("llama_cpp: starting server with command: %s", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            
            # Wait for server to become healthy
            max_retries = 30
            for i in range(max_retries):
                time.sleep(1.0)
                if super().health():
                    logger.info("llama_cpp: server started and healthy")
                    return True
                if self._process.poll() is not None:
                    logger.error("llama_cpp: server process exited prematurely")
                    return False
            
            logger.error("llama_cpp: server failed to become healthy after %d seconds", max_retries)
            return False
        except Exception as exc:
            logger.error("llama_cpp: failed to start server: %s", exc)
            return False

    def close(self) -> None:
        """Terminate the server process if we started it."""
        super().close()
        if self._process and self._process.poll() is None:
            logger.info("llama_cpp: terminating server process")
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

__all__ = ["LlamaCppEngine"]
