"""
JARVIS persistent memory — auto-index conversations, search context.
Uses MemoryHandle as backend. Storage path: ~/.jarvis/memory/
"""
import logging
from pathlib import Path
from datetime import datetime
from openjarvis.sdk import MemoryHandle
from openjarvis.core.config import load_config

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".jarvis" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

_handle: MemoryHandle | None = None

def get_memory() -> MemoryHandle:
    global _handle
    if _handle is None:
        config = load_config()
        # Override the db path to point to our JARVIS memory dir
        config.memory.db_path = str(MEMORY_DIR / "memory.db")
        _handle = MemoryHandle(config)
    return _handle

def index_conversation(user_input: str, assistant_response: str, agent: str, channel: str = "cli") -> None:
    """Index a completed conversation turn into persistent memory."""
    try:
        text = f"Usuario: {user_input}\nJARVIS: {assistant_response}"
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "channel": channel,
            "user": "Pau",
        }
        mem = get_memory()
        backend = mem._get_backend()
        backend.store(text, source="conversation", metadata=metadata)
    except Exception as e:
        logger.warning(f"Failed to index conversation: {e}")

def search_memory(query: str, top_k: int = 3) -> list[dict]:
    """Search memory for relevant past conversations. Returns list of {text, metadata, score}."""
    try:
        if not query.strip():
            return []
        return get_memory().search(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"Failed to search memory: {e}")
        return []

def format_memory_context(results: list[dict]) -> str:
    """Format memory search results as a compact context block for injection into prompts."""
    if not results:
        return ""
    lines = ["[MEMORIA RELEVANTE]"]
    for r in results:
        # handle metadata which might be nested or string
        meta = r.get("metadata") or {}
        ts = meta.get("timestamp", "")[:10] if isinstance(meta, dict) else ""
        text = r.get("content", r.get("text", ""))[:300]
        lines.append(f"- ({ts}) {text}")
    lines.append("[FIN MEMORIA]")
    return "\n".join(lines)

def stats_memory() -> int:
    try:
        mem = get_memory()
        backend = mem._get_backend()
        if hasattr(backend, "count"):
            return backend.count()
        return 0
    except Exception:
        return 0
