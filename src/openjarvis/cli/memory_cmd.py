"""``jarvis memory`` — memory management subcommands."""

from __future__ import annotations

import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

from openjarvis.core.config import load_config
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage.chunking import ChunkConfig
from openjarvis.tools.storage.ingest import ingest_path


def _get_backend(backend_key: str | None = None):
    """Instantiate the configured (or overridden) memory backend."""
    config = load_config()
    key = backend_key or config.memory.default_backend

    # Ensure backends are registered
    import openjarvis.tools.storage  # noqa: F401

    if not MemoryRegistry.contains(key):
        raise click.ClickException(
            f"Memory backend '{key}' not found. "
            f"Available: {', '.join(MemoryRegistry.keys())}"
        )

    if key == "sqlite":
        return MemoryRegistry.create(key, db_path=config.memory.db_path)
    return MemoryRegistry.create(key)


@click.group()
def memory() -> None:
    """Manage the memory store."""


@memory.command()
@click.argument("path")
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
@click.option(
    "--chunk-size",
    default=512,
    type=int,
    help="Chunk size in tokens.",
)
@click.option(
    "--chunk-overlap",
    default=64,
    type=int,
    help="Overlap between chunks in tokens.",
)
def index(
    path: str,
    backend: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Index documents from a file or directory."""
    console = Console(stderr=True)
    target = Path(path)

    if not target.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise SystemExit(1)

    t0 = time.time()
    cfg = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    console.print(f"[cyan]Indexing[/cyan] {path} ...")
    chunks = ingest_path(target, config=cfg)

    if not chunks:
        console.print("[yellow]No indexable content found.[/yellow]")
        return

    mem = _get_backend(backend)
    try:
        for chunk in track(chunks, description="Storing chunks...", console=console):
            mem.store(
                chunk.content,
                source=chunk.source,
                metadata={
                    "offset": chunk.offset,
                    "index": chunk.index,
                },
            )
    finally:
        if hasattr(mem, "close"):
            mem.close()

    elapsed = time.time() - t0
    sources = {c.source for c in chunks}
    console.print(
        f"[green]Indexed {len(chunks)} chunks "
        f"from {len(sources)} file(s) "
        f"in {elapsed:.1f}s.[/green]"
    )


@memory.command()
@click.argument("query", nargs=-1, required=False)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of results to return.",
)
def search(query: tuple[str, ...], top_k: int) -> None:
    """Search the memory store."""
    query_str = " ".join(query)
    if not query_str:
        click.echo('Uso: jarvis memory search "<consulta>"')
        return
    from openjarvis.memory.jarvis_memory import search_memory
    results = search_memory(query_str, top_k=top_k)
    if not results:
        click.echo("No encontré nada relevante en memoria.")
        return
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        ts = meta.get("timestamp", "")[:16] if isinstance(meta, dict) else ""
        click.echo(f"\n[{i}] {ts}")
        click.echo(r.get("content", r.get("text", ""))[:500])


@memory.command()
def stats() -> None:
    """Show memory store statistics."""
    from openjarvis.memory.jarvis_memory import stats_memory
    count = stats_memory()
    from pathlib import Path
    click.echo(f"Memoria JARVIS: {count} conversaciones indexadas en {Path.home() / '.jarvis' / 'memory'}")
