"""Tests for JARVIS persistent memory module."""
import pytest
from unittest.mock import MagicMock, patch

def test_index_and_search():
    with patch("openjarvis.memory.jarvis_memory.MemoryHandle") as MockHandle:
        mock_instance = MagicMock()
        MockHandle.return_value = mock_instance
        
        mock_backend = MagicMock()
        mock_instance._get_backend.return_value = mock_backend
        
        mock_instance.search.return_value = [
            {"text": "Usuario: hola\nJARVIS: hola Pau", "metadata": {"timestamp": "2026-04-08T09:00:00"}, "score": 0.9}
        ]
        from openjarvis.memory import jarvis_memory
        jarvis_memory._handle = mock_instance  # inject mock

        jarvis_memory.index_conversation("hola", "hola Pau", agent="native_react")
        mock_backend.store.assert_called_once()

        results = jarvis_memory.search_memory("hola", top_k=3)
        assert len(results) == 1
        assert "hola Pau" in results[0]["text"]

def test_format_memory_context_empty():
    from openjarvis.memory.jarvis_memory import format_memory_context
    assert format_memory_context([]) == ""

def test_format_memory_context_with_results():
    from openjarvis.memory.jarvis_memory import format_memory_context
    results = [{"text": "Usuario: test\nJARVIS: respuesta", "metadata": {"timestamp": "2026-04-08"}, "score": 0.8}]
    ctx = format_memory_context(results)
    assert "[MEMORIA RELEVANTE]" in ctx
    assert "2026-04-08" in ctx
    assert "[FIN MEMORIA]" in ctx
