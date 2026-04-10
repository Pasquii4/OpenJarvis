"""SQLite/FTS5 memory backend — zero-dependency default."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.events import EventType, get_event_bus
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult


def _check_fts5(conn: sqlite3.Connection) -> bool:
    """Return True if the SQLite build includes FTS5."""
    try:
        opts = conn.execute("PRAGMA compile_options").fetchall()
        return any("FTS5" in o[0].upper() for o in opts)
    except sqlite3.Error:
        return False


@MemoryRegistry.register("sqlite")
class SQLiteMemory(MemoryBackend):
    """Full-text search memory backend using SQLite FTS5.

    Uses the built-in ``sqlite3`` module — no extra dependencies.
    """

    backend_id: str = "sqlite"

    def __init__(self, db_path: str | Path = "") -> None:
        if not db_path:
            from openjarvis.core.config import DEFAULT_CONFIG_DIR

            db_path = str(DEFAULT_CONFIG_DIR / "memory.db")

        self._db_path = str(db_path)
        from openjarvis._rust_bridge import RUST_AVAILABLE

        if RUST_AVAILABLE:
            from openjarvis._rust_bridge import get_rust_module

            _rust = get_rust_module()
            self._rust_impl = _rust.SQLiteMemory(self._db_path)
            self._conn = None  # type: ignore[assignment]
        else:
            self._rust_impl = None
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id       TEXT PRIMARY KEY,
                content  TEXT NOT NULL,
                source   TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(
                content,
                source,
                tokenize='porter unicode61'
            );
        """)

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist *content* and return a unique document id."""
        meta_json = json.dumps(metadata) if metadata else None

        if self._rust_impl:
            doc_id = self._rust_impl.store(content, source, meta_json)
        else:
            import uuid
            import time

            doc_id = str(uuid.uuid4())
            with self._conn:
                self._conn.execute(
                    "INSERT INTO documents (id, content, source, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, content, source, meta_json or "{}", time.time()),
                )
                self._conn.execute(
                    "INSERT INTO documents_fts (content, source) VALUES (?, ?)",
                    (content, source),
                )

        bus = get_event_bus()
        bus.publish(
            EventType.MEMORY_STORE,
            {
                "backend": self.backend_id,
                "doc_id": doc_id,
                "source": source,
            },
        )
        return doc_id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Search via FTS5 MATCH with BM25 ranking."""
        if not query.strip():
            return []

        if self._rust_impl:
            from openjarvis._rust_bridge import retrieval_results_from_json

            results = retrieval_results_from_json(
                self._rust_impl.retrieve(query, top_k),
            )
        else:
            # Fallback Python implementation using FTS5
            results = []
            try:
                # Basic FTS5 search with BM25-like ordering
                cursor = self._conn.execute(
                    """
                    SELECT d.content, d.source, d.metadata, rank
                    FROM documents_fts f
                    JOIN documents d ON d.content = f.content AND d.source = f.source
                    WHERE documents_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """,
                    (query, top_k),
                )
                for row in cursor:
                    meta = json.loads(row[2]) if row[2] else {}
                    results.append(
                        RetrievalResult(
                            content=row[0],
                            score=float(row[3]),
                            source=row[1],
                            metadata=meta,
                        )
                    )
            except sqlite3.Error:
                # If FTS5 fails (e.g. syntax error in query), fallback to simple LIKE
                cursor = self._conn.execute(
                    """
                    SELECT content, source, metadata
                    FROM documents
                    WHERE content LIKE ? OR source LIKE ?
                    LIMIT ?
                """,
                    (f"%{query}%", f"%{query}%", top_k),
                )
                for row in cursor:
                    meta = json.loads(row[2]) if row[2] else {}
                    results.append(
                        RetrievalResult(
                            content=row[0],
                            score=0.0,
                            source=row[1],
                            metadata=meta,
                        )
                    )

        bus = get_event_bus()
        bus.publish(
            EventType.MEMORY_RETRIEVE,
            {
                "backend": self.backend_id,
                "query": query,
                "num_results": len(results),
            },
        )
        return results

    def delete(self, doc_id: str) -> bool:
        """Delete a document by id."""
        if self._rust_impl:
            return self._rust_impl.delete(doc_id)

        with self._conn:
            # Note: doc_id -> FTS sync is tricky without triggers or external content.
            # For simplicity, we just delete from documents.
            # A real implementation might need to rebuild FTS or use triggers.
            cursor = self._conn.execute(
                "SELECT content, source FROM documents WHERE id = ?", (doc_id,)
            )
            row = cursor.fetchone()
            if row:
                self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                self._conn.execute(
                    "DELETE FROM documents_fts WHERE content = ? AND source = ?",
                    (row[0], row[1]),
                )
                return True
        return False

    def clear(self) -> None:
        """Remove all stored documents."""
        if self._rust_impl:
            self._rust_impl.clear()
            return

        with self._conn:
            self._conn.execute("DELETE FROM documents")
            self._conn.execute("DELETE FROM documents_fts")

    def count(self) -> int:
        """Return the number of stored documents."""
        if self._rust_impl:
            return self._rust_impl.count()

        cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()


__all__ = ["SQLiteMemory"]
